import pyglet
import pyglet.shapes as shapes
from pyglet.window import mouse
from pyglet.event import EventDispatcher
from pyglet.gui.widgets import WidgetBase

import games.hex.vortex_board as vb
from neural_network import NeuralNetwork
from players.uninformed_mcts_player import UninformedMCTSPlayer, RolloutMCTSPlayer
from players.deep_mcts_player import DeepMCTSPlayer
from games.vortex import Vortex_5_20, Vortex_6_20, Vortex_7_20, Vortex_8_20, Vortex_9_20, Vortex_5_20_edge_weights, Vortex_4, Vortex_5_mctspt
from models.vornet import VorNet, VorNetBN, VorNetBN3

HCOLOUR = (225, 30, 30)
VCOLOUR = (30, 30, 255)
NODE_UI_WIDTH = 30
UI_PLAYER = vb.HORIZONTAL_PLAYER
AI_PLAYER = vb.VERTICAL_PLAYER
MCTS_SIMS = 500
CHECKPOINT = 0

class NodeUI(WidgetBase):
    def __init__(self, x, y, node_ndx):
        c = NODE_UI_WIDTH // 2
        super().__init__(x-c, y-c, NODE_UI_WIDTH, NODE_UI_WIDTH)
        self.node_ndx = node_ndx

    def on_mouse_press(self, x, y, buttons, modifiers):   
        if self._check_hit(x, y) and buttons == mouse.LEFT:
            self.dispatch_event('on_node_press', self.node_ndx)

NodeUI.register_event_type('on_node_press')

class BoardPieces(EventDispatcher):
    def __init__(self, board, window):
        self.board = board
        self.window = window
        self.node_pos = (self.board.node_attr[:, -2:] - 0.5) * 400 + 300
        self.nodes, self.node_buttons = self.create_board_nodes()
        self.edges = self.create_board_edges()
        self.stones = {}
        self.refresh_stones()
        self.window.push_handlers(self)
        self.push_handlers(self)

    def refresh_stones(self):
        r = 10
        for i, pos in enumerate(self.node_pos):
            if i not in self.stones.keys():
                x, y = pos
                ns = self.board.get_node_state(i)
                if ns == vb.HORIZONTAL_PLAYER:
                    self.stones[i] = shapes.Circle(x=x, y=y, radius=r, color=HCOLOUR)
                elif ns == vb.VERTICAL_PLAYER:
                    self.stones[i] = shapes.Circle(x=x, y=y, radius=r, color=VCOLOUR)

    def create_board_nodes(self):
        nodes = []
        node_buttons = []
        side_nodes_start = self.node_pos.shape[0] - 4

        for i, pos in enumerate(self.node_pos):
            x, y = pos
            r = 5 if i < side_nodes_start else 15
            nodes.append(shapes.Circle(x=x, y=y, radius=r))
            node_button = NodeUI(x, y, i)
            self.window.push_handlers(node_button)
            node_button.push_handlers(self)
            node_buttons.append(node_button)

        return nodes, node_buttons

    def create_board_edges(self):
        edges = []

        for edge in board.edge_index.T:
            x, y = self.node_pos[edge].T
            edges.append(shapes.Line(x[0], y[0], x[1], y[1]))

        return edges

    def on_draw(self):
        self.window.clear()
        for n in self.nodes:
            n.draw()
        for e in self.edges:
            e.draw()
        for s in self.stones.values():
            s.draw()

    def on_node_press(self, node_ndx):
        if self.board.get_node_state(node_ndx) == vb.EMPTY:
            self.board.move(node_ndx)
            self.refresh_stones()
            self.on_draw()
            
            game_over = game.check_game_over(self.board)
            if game_over is None:
                self.board = opponent.update_state(self.board)
                self.refresh_stones()
                self.on_draw()
                game_over = game.check_game_over(self.board)
            
            if game_over is not None:
                print("game over", game_over)
                #print(opponent.debug_stats)

BoardPieces.register_event_type('on_next_move')

window = pyglet.window.Window(600, 600)
window.set_caption('Vortex')
game = Vortex_5_mctspt()
nn = NeuralNetwork(game, VorNetBN, cuda=True)
nn.load(CHECKPOINT)
opponent = DeepMCTSPlayer(game, nn, simulations=MCTS_SIMS)
# opponent = RolloutMCTSPlayer(game, MCTS_SIMS)
board = game.get_initial_state()
board_pieces = BoardPieces(board, window)

pyglet.app.run()