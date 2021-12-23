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

HORIZONTAL_PLAYER = 0
VERTICAL_PLAYER = 1
NEXT_PLAYER = 2
POINT_X = 3
POINT_Y = 4

WinState = namedtuple('WinState', ['is_ended', 'winner'])

class VortexBoard():

    def __init__(self, edge_index, node_attr):
        self.edge_index = edge_index
        self.node_attr = node_attr
        self.node_count = self.node_attr.size(0)
        self.tnode_ndx = {
            "top": self.node_count-4,
            "bottom": self.node_count-3,
            "left": self.node_count-2,
            "right": self.node_count-1
        }

    @property
    def open_cells(self):
        return ~(self.node_attr[:, HORIZONTAL_PLAYER] > 0. | self.node_attr[:, VERTICAL_PLAYER] > 0.)

    def tostring(self):
        self.node_attr[:, :NEXT_PLAYER+1].tobytes()
   
    def copy(self):
        return VortexBoard(self.edge_index.copy(), self.node_attr.copy())

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
                n = set(neighbourhood_ndx[connected_mask].numpy())
                todo = todo.union(n - done)
                done.add(node)

            return False

        rval = np.zeros(2)
        if is_connected(VERTICAL_PLAYER, self.tnode_ndx["top"], self.tnode_ndx["bottom"]):
            rval[VERTICAL_PLAYER] = 1.
        elif is_connected(HORIZONTAL_PLAYER, self.tnode_ndx["left"], self.tnode_ndx["right"]):
            rval[HORIZONTAL_PLAYER] = 1.

        return rval

    def print_board_str(self):
        print(self.edge_index)
        print(self.node_attr)

    @classmethod
    def new_vortex_board(cls, size, device="cpu"):
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
        node_attr = torch.zeros((node_count, 5), device=device)
        # set up terminal (off-board) player nodes
        node_attr[tnode_ndx["top"], VERTICAL_PLAYER] = 1.
        node_attr[tnode_ndx["bottom"], VERTICAL_PLAYER] = 1,
        node_attr[tnode_ndx["left"], HORIZONTAL_PLAYER] = 1.
        node_attr[tnode_ndx["right"], HORIZONTAL_PLAYER] = 1.

        node_attr[:inner_point_count, POINT_X:] = points

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


        # # get the Voronoi regions for each node
        # vor = Voronoi(points)
        # vor_regions = []
        # # regions for the outer points are not closed so we need to add some extra points
        # # to create closed psuedo-regions for the UI
        # for node_ndx, region_ndx in enumerate(vor.point_region):
        #     region_vert_ndx = vor.regions[region_ndx]
        #     if -1 in region_vert_ndx:
        #         i = region_vert_ndx.index(-1)
        #         pre = (i - 1) % len(region_vert_ndx)
        #         post = (i + 1) % len(region_vert_ndx)
        #         pre_pt = vor.vertices[region_vert_ndx[pre]]
        #         post_pt = vor.vertices[region_vert_ndx[post]]
        #         pre_verts = vor.vertices[region_vert_ndx[:i]]
        #         post_verts = vor.vertices[region_vert_ndx[i+1:]]

        #         if node_ndx in left_border_ndx and node_ndx in top_border_ndx:
        #             missing_pts = np.array([[-0.1, pre_pt[1]], [-0.1, 1.1], [post_pt[0], 1.1]])
        #         elif node_ndx in top_border_ndx and node_ndx in right_border_ndx:
        #             missing_pts = np.array([[post_pt[0], 1.1], [1.1, 1.1], [1.1, pre_pt[1]]])
        #         elif node_ndx in right_border_ndx and node_ndx in bottom_border_ndx:
        #             missing_pts = np.array([[post_pt[0], -0.1], [1.1, -0.1], [1.1, pre_pt[1]]])
        #         elif node_ndx in bottom_border_ndx and node_ndx in left_border_ndx:
        #             missing_pts = np.array([[pre_pt[0], -0.1], [-0.1, -0.1], [-0.1, post_pt[1]]])
        #         elif node_ndx in left_border_ndx:
        #             missing_pts = np.array([[-0.1, pre_pt[1]], [-0.1, post_pt[1]]])
        #         elif node_ndx in right_border_ndx:
        #             missing_pts = np.array([[1.1, pre_pt[1]], [1.1, post_pt[1]]])
        #         elif node_ndx in top_border_ndx:
        #             missing_pts = np.array([[pre_pt[0], 1.1], [post_pt[0], 1.1]])
        #         elif node_ndx in bottom_border_ndx:
        #             missing_pts = np.array([[pre_pt[0], -0.1], [post_pt[0], -0.1]])

        #         vor_region = np.concatenate([pre_verts, missing_pts, post_verts])
        #     else:
        #         vor_region = vor.vertices[region_vert_ndx]

        #     vor_regions.append(vor_region)

        return cls(edge_index.numpy(), node_attr.numpy())
