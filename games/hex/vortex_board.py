from collections import namedtuple
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from scipy.sparse import dok_matrix
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from torch_geometric.utils import get_laplacian

HORIZONTAL_PLAYER = 0
VERTICAL_PLAYER = 1
EMPTY = -1
NEXT_PLAYER = 2
POINT_X = 3
POINT_Y = 4

N_STEPS = 10

WinState = namedtuple('WinState', ['is_ended', 'winner'])

def merge_nodes(merge_nodes, player_nodes, edge_index):
    """
    merge nodes by redirecting any connected edges, only the edges are changed
    """
    m_nodes = merge_nodes.copy()

    while m_nodes:
        node = m_nodes.pop()
        #if node not in merged:
        # get edges going out from the node
        c1_out_mask = (edge_index[0] == node)
        if c1_out_mask.any():
            # get connected nodes filled by same player
            c1_nodes = edge_index[1, c1_out_mask]
            c1_nodes = c1_nodes[np.in1d(c1_nodes, player_nodes)]

            # get all edges in and out of these nodes
            out_edge_mask = np.in1d(edge_index[0], c1_nodes)
            in_edge_mask = np.in1d(edge_index[1], c1_nodes)

            # form new edges to 2-hop adjacent nodes that are not filled by player
            c2_nodes = np.unique(edge_index[1, out_edge_mask])
            c2_nodes = c2_nodes[c2_nodes != node]
            #c2_nodes = c2_nodes[~np.in1d(c2_nodes, nodes)]
            new_edges = np.stack([c2_nodes, np.full_like(c2_nodes, node)])

            # print('Node: ', node)
            # print('Connected nodes: ', c1_nodes)
            # print('Remaining edges', edge_index[:, ~(out_edge_mask|in_edge_mask)].T)
            # print('New edges', new_edges.T)

            # remove all edges from merged nodes and add new edges
            edge_index = np.concatenate([
                edge_index[:, ~(out_edge_mask|in_edge_mask)],
                new_edges,
                new_edges[[1,0]]
            ], axis=1)
            edge_index = np.unique(edge_index, axis=1)
            
            # return the node to the queue if it is still connected
            c1_out_mask = (edge_index[0] == node)
            if c1_out_mask.any():
                # get connected nodes filled by same player
                c1_nodes = edge_index[1, c1_out_mask]
                c1_nodes = c1_nodes[np.in1d(c1_nodes, player_nodes)]
                if not c1_nodes.size == 0:
                    m_nodes.append(node)

    return edge_index

class VortexBoard():

    def __init__(self, edge_index, node_attr, n_steps, use_edge_weight):
        self.edge_index = edge_index
        self.node_attr = node_attr
        self.node_count = self.node_attr.shape[0]
        self.tnode_ndx = {
            "top": self.node_count-4,
            "bottom": self.node_count-3,
            "left": self.node_count-2,
            "right": self.node_count-1
        }
        self.n_steps = n_steps
        self.features = self.n_steps * 6
        self._nn_attr = None
        self.use_edge_weight = use_edge_weight

    @property
    def nn_attr(self):
        """ lazy calculation of input for the neural network
        """
        if self._nn_attr is None:
            self._nn_attr = self.get_nn_attr()
        return self._nn_attr

    @property
    def shape(self):
        return self.node_count, self.features

    def set_stone(self, node_ndx, player):
        """
        Sets a stone at the given node index for the given player
        """
        assert self.node_attr[node_ndx, :2].sum() == 0.
        assert player == HORIZONTAL_PLAYER or player == VERTICAL_PLAYER

        self.node_attr[node_ndx, player] = 1.

        assert self.node_attr[node_ndx, :2].sum() == 1.

    def get_node_state(self, node_ndx):
        """ 
        Gets the state of a node ie. which player has a stome on it
        """
        if self.node_attr[node_ndx, HORIZONTAL_PLAYER]:
            return HORIZONTAL_PLAYER
        elif self.node_attr[node_ndx, VERTICAL_PLAYER]:
            return VERTICAL_PLAYER
        else:
            return EMPTY

    def move(self, node_ndx):
        p = self.get_player()
        self.set_stone(node_ndx, p)
        self.node_attr[:, NEXT_PLAYER] = (self.node_attr[:, NEXT_PLAYER] + 1) % 2 # Toggle player

    def take_action(self, a):
        p = self.get_player()
        self.node_attr[:, p] += a.astype(np.float32) # Next move
        self.node_attr[:, NEXT_PLAYER] = (self.node_attr[:, NEXT_PLAYER] + 1) % 2 # Toggle player

    def get_vor_attr(self, player):
        """
        get the node attribute array froma single player perspective
        removes edges to other player nodes 

        Layers:
        0 - side 1 node of current player
        1 - side 2 node of current player
        2 - current player filled nodes
        """
        if player == HORIZONTAL_PLAYER:
            them = VERTICAL_PLAYER
            us = HORIZONTAL_PLAYER
            s1, s2 = self.tnode_ndx['left'], self.tnode_ndx['right']
        else:
            them = HORIZONTAL_PLAYER
            us = VERTICAL_PLAYER
            s1, s2 = self.tnode_ndx['top'], self.tnode_ndx['bottom']

        # set the stones
        v_attr = np.zeros((self.node_count, 3))
        v_attr[:, 2] = self.node_attr[:, us]
        #v_attr[s1] = np.array([1., 0, 0.])
        #v_attr[s2] = np.array([0, 1., 0.])
        # TODO: fix this after comparison
        v_attr[s1] = np.array([1., 0, 1.])
        v_attr[s2] = np.array([0, 1., 1.])

        # remove edges to other player filled nodes
        t_mask = self.node_attr[:, them] > 0.
        their_nodes = t_mask.nonzero()[0]
        m = ~(np.in1d(self.edge_index[0], their_nodes) | np.in1d(self.edge_index[1], their_nodes))
        v_edge_index = self.edge_index[:, m]

        # merge this player's connected internal nodes
        o_mask = self.node_attr[:-4, us] > 0.
        our_nodes = o_mask.nonzero()[0].tolist()
        v_edge_index = merge_nodes(our_nodes, our_nodes, v_edge_index)
        # merge this player's side nodes to remaining internal nodes
        o_mask = self.node_attr[:, us] > 0.
        all_our_nodes = o_mask.nonzero()[0] # our nodes including sides
        o_mask[:-4] = False
        side_nodes = o_mask.nonzero()[0].tolist()
        v_edge_index = merge_nodes(side_nodes, our_nodes, v_edge_index)

        # edges have weight 1. if both connected nodes are empty or 2. if they connect to one of our nodes
        v_edge_weight = np.ones_like(v_edge_index[0], dtype=np.float32)
        m = (np.in1d(v_edge_index[0], all_our_nodes) | np.in1d(v_edge_index[1], all_our_nodes))
        v_edge_weight[m] = 2.

        return v_attr, v_edge_index, v_edge_weight

    def get_player(self):
        return int(self.node_attr[0, NEXT_PLAYER])

    def get_nn_attr(self):
        """
        preprocess the attributes for the neural network

        runs n_steps of message passing Laplacian trnasformation for each side view
        """
        current_player = self.get_player()
        steps = []
        # do message passing and concatenate the timesteps
        # first n_steps x 3 layers are the current player
        # next n_steps x 3 layers are the other player
        for p in [current_player, (current_player+1)%2]:
            v_attr, v_edge_index, v_edge_weight = self.get_vor_attr(p)
            L = get_laplacian(
                edge_index=torch.tensor(v_edge_index), 
                edge_weight=torch.tensor(v_edge_weight) if self.use_edge_weight else None, 
                normalization='sym', 
                num_nodes=self.node_count
            )
            L = torch.sparse_coo_tensor(L[0], L[1])
            v_attr = torch.tensor(v_attr, dtype=torch.float32)

            for i in range(self.n_steps):
                steps.append(v_attr.numpy())
                if i < self.n_steps-1:
                    v_attr = L @ v_attr

        x = np.concatenate(steps, axis=1)
        # reverse the side node order for the horizontal player so that the first two side nodes are
        # always those of the current player, and the last two are those of the other player
        x1 = x.copy()
        if current_player == HORIZONTAL_PLAYER:
            x1[-4] = x[-2]
            x1[-3] = x[-1]
            x1[-2] = x[-4]
            x1[-1] = x[-3]
        return x1

    def get_available_actions(self):
        return ~((self.node_attr[:, HORIZONTAL_PLAYER] > 0.) | (self.node_attr[:, VERTICAL_PLAYER] > 0.))

    def tostring(self):
        return self.node_attr[:, :NEXT_PLAYER+1].tobytes()
   
    def copy(self):
        return VortexBoard(self.edge_index.copy(), self.node_attr.copy(), n_steps=self.n_steps, use_edge_weight=self.use_edge_weight)

    def check_game_over(self):
        """ checks whether HORIZONTAL_PLAYER has made a left-right connection or
        VERTICAL_PLAYER has made a top-bottom connection
        """
        def is_connected(player, start_node, end_node):
            # see if we can connect the start_node to the end_node
            # using a depth-first search
            todo = set([start_node])
            done = set()

            while todo:
                node = todo.pop()
                if node == end_node:
                    return True

                neighbourhood_mask = self.edge_index[0] == node
                neighbourhood_ndx = self.edge_index[1, neighbourhood_mask]
                connected_mask = self.node_attr[neighbourhood_ndx, player] > 0.
                n = set(neighbourhood_ndx[connected_mask])
                todo = todo.union(n - done)
                done.add(node)

            return False

        rval = np.zeros(2)
        if is_connected(VERTICAL_PLAYER, self.tnode_ndx["top"], self.tnode_ndx["bottom"]):
            rval[VERTICAL_PLAYER] = 1.
        elif is_connected(HORIZONTAL_PLAYER, self.tnode_ndx["left"], self.tnode_ndx["right"]):
            rval[HORIZONTAL_PLAYER] = 1.
        else:
            rval = None

        return rval

    def print_board_str(self):
        print(self.edge_index)
        print(self.node_attr)

    @classmethod
    def new_vortex_board(cls, size, n_steps=10, use_edge_weight=False):
        """ construct a new empty vortex board with approximately the same complexity
        as a hex board of size: size

        0 - HORIZONTAL_PLAYER
        1 - VERTICAL_PLAYER
        2 - NEXT_PLAYER
        3 - POINT_X
        4 - POINT_Y

        """
        min_dist = 3 / (size * 4)

        # set up the border points
        points = np.concatenate([
            np.linspace((0., 0.), (1., 0.), size)[:-1],
            np.linspace((0., 1.), (1., 1.), size)[1:],
            np.linspace((0., 0.), (0., 1.), size)[1:],
            np.linspace((1., 0.), (1., 1.), size)[:-1]
        ])
        left_border_ndx = (points[:, 0] == 0.0).nonzero()[0]
        right_border_ndx = (points[:, 0] == 1.0).nonzero()[0]
        bottom_border_ndx = (points[:, 1] == 0.0).nonzero()[0]
        top_border_ndx = (points[:, 1] == 1.0).nonzero()[0]

        # sample the inner points
        inner_point_count = size**2 - ((size - 1) * 4)
        for i in range(inner_point_count):
            while(True):
                p = np.random.random_sample((1, 2))
                dist = cdist(points, p, metric="euclidean").min()
                if dist > min_dist:
                    points = np.concatenate([points, p])
                    break

        # set up node attribues
        node_count = points.shape[0] + 4
        tnode_ndx = {
            "top": node_count-4,
            "bottom": node_count-3,
            "left": node_count-2,
            "right": node_count-1
        }
        node_attr = np.zeros((node_count, 5))
        # set up terminal (off-board) player nodes
        node_attr[tnode_ndx["top"], VERTICAL_PLAYER] = 1.
        node_attr[tnode_ndx["bottom"], VERTICAL_PLAYER] = 1.
        node_attr[tnode_ndx["left"], HORIZONTAL_PLAYER] = 1.
        node_attr[tnode_ndx["right"], HORIZONTAL_PLAYER] = 1.

        node_attr[:points.shape[0], POINT_X:] = points
        node_attr[-4:, POINT_X:] = np.array([
            [0.5, 1.2],
            [0.5, -0.2],
            [-0.2, 0.5],
            [1.2, 0.5]
        ])

        # build the adjacency matrix for the graph
        tri = Delaunay(points)
        adj = dok_matrix((node_count, node_count), dtype=np.int8)
        for s in tri.simplices:
            for i1 in range(3):
                i2 = (i1 + 1) % 3
                v1 = s[i1]
                v2 = s[i2]
                adj[v1, v2] = 1
                adj[v2, v1] = 1

        adj = adj.tocoo()
        edge_index = np.stack([adj.row, adj.col])

        # add the terminal edges to the border nodes
        def add_edges(edge_index, node_ndx, nodes):
            new_edge_index = np.stack([
                np.full_like(nodes, fill_value=node_ndx),
                nodes
            ])
            edge_index = np.concatenate([
                edge_index,
                new_edge_index,
                new_edge_index[[1, 0]]
            ], axis=1)
            return edge_index

        edge_index = add_edges(edge_index, tnode_ndx["top"], top_border_ndx)
        edge_index = add_edges(edge_index, tnode_ndx["bottom"], bottom_border_ndx)
        edge_index = add_edges(edge_index, tnode_ndx["left"], left_border_ndx)
        edge_index = add_edges(edge_index, tnode_ndx["right"], right_border_ndx)
        
        return cls(edge_index, node_attr, n_steps, use_edge_weight)



# edge_index = np.array([
#     [0,1,2,4,5,4,7,8,2,3],
#     [1,2,3,5,6,6,8,9,10,4]
# ])
# edge_index = np.concatenate([edge_index, edge_index[[1,0]]], axis=1)

# print(edge_index)
# nodes = np.array([1,2,5,6])
# print(nodes)
# edge_index = merge_nodes(nodes, edge_index)
# print(edge_index)