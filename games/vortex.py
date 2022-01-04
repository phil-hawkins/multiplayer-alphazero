import numpy as np
import sys
from scipy.signal import correlate2d
sys.path.append("..")
from game import Game
import games.hex.vortex_board as vb

# Implementation Vortex game, 
# a connection game like Hex but played on a randomly generated Voronoi diagram.
class Vortex(Game):

    # Returns a blank Vortex board.
    # Layers:
    # 0 - HORIZONTAL_PLAYER
    # 1 - VERTICAL_PLAYER
    # 2 - NEXT_PLAYER
    # 3 - POINT_X
    # 4 - POINT_Y
    def get_initial_state(self):
        return vb.VortexBoard.new_vortex_board(size=5, n_steps=10)

    # Returns a 1d ndarray indicating open nodes. 
    def get_available_actions(self, s):
        return s.get_available_actions()
        
    # Update the state.
    def take_action(self, s, a):
        s = s.copy()
        s.take_action(a)
        return s

    # Check for a side to side connection
    def check_game_over(self, s):
        return s.check_game_over()

    # Return HORIZONTAL_PLAYER = 0, VERTICAL_PLAYER = 1
    def get_player(self, s):
        return s.get_player()

    # Fixed constant
    def get_num_players(self):
        return 2

    # Print a human-friendly visualization of the board.
    def visualize(self, s):
        s.print_board_str()

class Vortex7(Vortex):
    def get_initial_state(self):
        return vb.VortexBoard.new_vortex_board(size=7, n_steps=14)

class Vortex11(Vortex):
    def get_initial_state(self):
        return vb.VortexBoard.new_vortex_board(size=11, n_steps=22)

class Vortex_5_10(Vortex):
    def get_initial_state(self):
        return vb.VortexBoard.new_vortex_board(size=5, n_steps=10)

class Vortex_5_20(Vortex):
    def get_initial_state(self):
        return vb.VortexBoard.new_vortex_board(size=5, n_steps=20)

class Vortex_6_20(Vortex):
    def get_initial_state(self):
        return vb.VortexBoard.new_vortex_board(size=6, n_steps=20)

class Vortex_7_20(Vortex):
    def get_initial_state(self):
        return vb.VortexBoard.new_vortex_board(size=7, n_steps=20)

class Vortex_8_20(Vortex):
    def get_initial_state(self):
        return vb.VortexBoard.new_vortex_board(size=8, n_steps=20)

class Vortex_9_20(Vortex):
    def get_initial_state(self):
        return vb.VortexBoard.new_vortex_board(size=9, n_steps=20)