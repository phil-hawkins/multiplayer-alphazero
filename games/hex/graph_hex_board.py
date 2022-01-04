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

VERTICAL_PLAYER = 1
HORIZONTAL_PLAYER = -1
EMPTY_CELL = 0

WinState = namedtuple('WinState', ['is_ended', 'winner'])


class GraphHexBoard():
    """
    Hex Board played on an a graph.
    """

    def __init__(self, edge_index, node_attr, tri, vor_regions):
        """Set up initial board configuration."""
        self.winner = None
        self.edge_index = edge_index
        self.node_attr = node_attr
        self.tri = tri
        n = node_attr.size(0)
        if node_attr[n-4, 0] == VERTICAL_PLAYER:
            self.tnode_ndx = {
                "top": n-4,
                "bottom": n-3,
                "left": n-2,
                "right": n-1
            }
        else:
            self.tnode_ndx = {
                "left": n-4,
                "right": n-3,
                "top": n-2,
                "bottom": n-1
            }
        assert node_attr[self.tnode_ndx['top'], 0].long().numpy().item() == VERTICAL_PLAYER
        assert node_attr[self.tnode_ndx['bottom'], 0].long().numpy().item() == VERTICAL_PLAYER
        assert node_attr[self.tnode_ndx['left'], 0].long().numpy().item() == HORIZONTAL_PLAYER
        assert node_attr[self.tnode_ndx['right'], 0].long().numpy().item() == HORIZONTAL_PLAYER
        # Voronoi regions for each node
        self.vor_regions = vor_regions
        self.x = None

    @property
    def action_size(self):
        # action size is the number of playable cells on the board
        return self.node_attr.size(0) - len(self.tnode_ndx)

    @classmethod
    def new_vortex_board(cls, size, device="cpu"):
        """ construct a new empty vortex board with approximately the same complexity
        as a hex board of size: size
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
        node_attr = torch.zeros((node_count, 3), device=device)
        # set up terminal (off-board) player nodes
        node_attr[tnode_ndx["top"]] = torch.tensor([VERTICAL_PLAYER, 1., 0.])
        node_attr[tnode_ndx["bottom"]] = torch.tensor([VERTICAL_PLAYER, 0., 1.])
        node_attr[tnode_ndx["left"]] = torch.tensor([HORIZONTAL_PLAYER, 1., 0.])
        node_attr[tnode_ndx["right"]] = torch.tensor([HORIZONTAL_PLAYER, 0., 1.])

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
        edge_index = torch.stack([
            torch.tensor(adj.row, dtype=torch.long),
            torch.tensor(adj.col, dtype=torch.long)
        ]).to(device=device)

        # add the terminal edges to the border nodes
        def add_edges(edge_index, node_ndx, nodes):
            nodes = torch.tensor(nodes, dtype=torch.long, device=device)
            new_edge_index = torch.stack([
                torch.full_like(nodes, fill_value=node_ndx),
                nodes
            ])
            edge_index = torch.cat([
                edge_index,
                new_edge_index,
                new_edge_index[[1, 0]]
            ], dim=1)
            return edge_index

        edge_index = add_edges(edge_index, tnode_ndx["top"], top_border_ndx)
        edge_index = add_edges(edge_index, tnode_ndx["bottom"], bottom_border_ndx)
        edge_index = add_edges(edge_index, tnode_ndx["left"], left_border_ndx)
        edge_index = add_edges(edge_index, tnode_ndx["right"], right_border_ndx)

        # get the Voronoi regions for each node
        vor = Voronoi(points)
        vor_regions = []
        # regions for the outer points are not closed so we need to add some extra points
        # to create closed psuedo-regions for the UI
        for node_ndx, region_ndx in enumerate(vor.point_region):
            region_vert_ndx = vor.regions[region_ndx]
            if -1 in region_vert_ndx:
                i = region_vert_ndx.index(-1)
                pre = (i - 1) % len(region_vert_ndx)
                post = (i + 1) % len(region_vert_ndx)
                pre_pt = vor.vertices[region_vert_ndx[pre]]
                post_pt = vor.vertices[region_vert_ndx[post]]
                pre_verts = vor.vertices[region_vert_ndx[:i]]
                post_verts = vor.vertices[region_vert_ndx[i+1:]]

                if node_ndx in left_border_ndx and node_ndx in top_border_ndx:
                    missing_pts = np.array([[-0.1, pre_pt[1]], [-0.1, 1.1], [post_pt[0], 1.1]])
                elif node_ndx in top_border_ndx and node_ndx in right_border_ndx:
                    missing_pts = np.array([[post_pt[0], 1.1], [1.1, 1.1], [1.1, pre_pt[1]]])
                elif node_ndx in right_border_ndx and node_ndx in bottom_border_ndx:
                    missing_pts = np.array([[post_pt[0], -0.1], [1.1, -0.1], [1.1, pre_pt[1]]])
                elif node_ndx in bottom_border_ndx and node_ndx in left_border_ndx:
                    missing_pts = np.array([[pre_pt[0], -0.1], [-0.1, -0.1], [-0.1, post_pt[1]]])
                elif node_ndx in left_border_ndx:
                    missing_pts = np.array([[-0.1, pre_pt[1]], [-0.1, post_pt[1]]])
                elif node_ndx in right_border_ndx:
                    missing_pts = np.array([[1.1, pre_pt[1]], [1.1, post_pt[1]]])
                elif node_ndx in top_border_ndx:
                    missing_pts = np.array([[pre_pt[0], 1.1], [post_pt[0], 1.1]])
                elif node_ndx in bottom_border_ndx:
                    missing_pts = np.array([[pre_pt[0], -0.1], [post_pt[0], -0.1]])

                vor_region = np.concatenate([pre_verts, missing_pts, post_verts])
            else:
                vor_region = vor.vertices[region_vert_ndx]

            vor_regions.append(vor_region)

        return cls(edge_index, node_attr, tri, vor_regions)

    def reverse(self):
        """ return a board with players reversed
        note: graph embedding points are not changed
        """
        node_attr = self.node_attr.clone().detach()
        node_attr[:, 0] *= -1

        rval = GraphHexBoard(self.edge_index, node_attr, self.tri, self.vor_regions)

        return rval

    @property
    def cell_colours(self):
        cell_colours = np.stack([
            mcolors.to_rgba("red"),
            mcolors.to_rgba("linen"),
            mcolors.to_rgba("blue")
        ])

        return cell_colours[self.node_attr[:, 0].long() + 1]

    @property 
    def state_np(self):
        return self.node_attr[:-len(self.tnode_ndx), 0].numpy()

    @state_np.setter
    def state_np(self, np_state):
        self.node_attr[:-len(self.tnode_ndx), 0] = torch.tensor(np_state)

    def plot(self):
        vor = Voronoi(self.tri.points)

        plt.rcParams['figure.figsize'] = [10, 10]
        fig, ax = plt.subplots()

        voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='blue', line_width=2, line_alpha=0.8, point_size=2)

        patches = []
        for region in self.vor_regions:
            patches.append(Polygon(region))
        p = PatchCollection(patches, match_original=True, alpha=0.4, picker=1)
        p.set_facecolor(self.cell_colours)
        ax.add_collection(p)

        fig.canvas.mpl_connect('pick_event', self.on_pick_node)

        ax.triplot(self.tri.points[:, 0], self.tri.points[:, 1], self.tri.simplices, line_colors='gray')
        ax.plot(self.tri.points[:, 0], self.tri.points[:, 1], 'o')
        if self.x is not None:
            for n, p in enumerate(self.tri.points):
                x, y = p
                s = "{}: {:.2f}".format(n, self.x[n])
                ax.text(x, y, s, fontsize=12)

        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.show()

    def on_pick_node(self, event):
        artist = event.artist
        if isinstance(artist, PatchCollection):
            node_ndx = event.ind[0]
            print('class onpick node:', node_ndx)
            self.add_stone(node_ndx, 1)
            artist.set_facecolor(self.cell_colours)
            artist.figure.canvas.draw()

    def add_stone(self, action, player):
        assert self.node_attr[action, 0] == 0
        self.node_attr[action, 0] = player

    def get_valid_moves(self):
        """ Any zero value is a valid move
        """
        valids = self.node_attr[:-4, 0] == 0
        return list(valids.numpy())

    def get_win_state(self):
        """ checks whether player 1 has made a left-right connection or
        player -1 has made a top-bottom connection
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
                connected_mask = self.node_attr[neighbourhood_ndx, 0] == player
                n = set(neighbourhood_ndx[connected_mask].numpy())
                todo = todo.union(n - done)
                done.add(node)

            return False

        if is_connected(VERTICAL_PLAYER, self.tnode_ndx["top"], self.tnode_ndx["bottom"]):
            return WinState(True, VERTICAL_PLAYER)
        elif is_connected(HORIZONTAL_PLAYER, self.tnode_ndx["left"], self.tnode_ndx["right"]):
            return WinState(True, HORIZONTAL_PLAYER)

        return WinState(False, None)

    @property
    def state_representation(self):
        # get the playable node states and remove whitespace from string
        active_state = ''.join(str(self.state_np).split())
        
        return active_state

    def get_vortex_attr(self, hplays=True):
        VERTICAL_PLAYER = 1
        HORIZONTAL_PLAYER = -1
        F2O_WEIGHT = 1.
        F2F_WEIGHT = 1.

        n_nodes = self.node_attr.size(0)
        v_attr = torch.zeros(n_nodes, 3)

        if hplays:
            them = VERTICAL_PLAYER
            us = HORIZONTAL_PLAYER
            s1, s2 = n_nodes-2, n_nodes-1
        else:
            them = HORIZONTAL_PLAYER
            us = VERTICAL_PLAYER
            s1, s2 = n_nodes-4, n_nodes-3

        self.node_attr[18,0] = us

        # set the stones
        t_mask = self.node_attr[:, 0] == them
        o_mask = self.node_attr[:, 0] == us
        v_attr[o_mask, 2] = 1.
        v_attr[s1] = torch.tensor([1., 0, 0])
        v_attr[s2] = torch.tensor([0, 1., 0])

        # remove edges to other player filled nodes
        their_nodes = t_mask.nonzero().squeeze().numpy()
        ei_np = self.edge_index.numpy()
        m = ~(np.in1d(ei_np[0], their_nodes) | np.in1d(ei_np[1], their_nodes))
        edge_index = self.edge_index[:, m]

        # find filled-to-filled and open-to-filled edges
        our_nodes = o_mask.nonzero().squeeze().numpy()
        ei_np = edge_index.numpy()
        f2f_mask = np.in1d(ei_np[0], our_nodes) & np.in1d(ei_np[1], our_nodes)
        f2o_mask = np.in1d(ei_np[0], our_nodes) ^ np.in1d(ei_np[1], our_nodes)
        # o2o_edges = edge_index[:, ~(f2f_mask|f2o_mask)]
        # f2f_edges = edge_index[:, f2f_mask]
        # f2o_edges = edge_index[:, f2o_mask]
        edge_weights = torch.ones(edge_index.size(1))
        edge_weights[f2o_mask] = F2O_WEIGHT
        edge_weights[f2f_mask] = F2F_WEIGHT

        return v_attr, edge_index, edge_weights