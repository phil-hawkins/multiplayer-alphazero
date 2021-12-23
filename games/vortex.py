import numpy as np
import sys
from scipy.signal import correlate2d
sys.path.append("..")
from game import Game
import hex.vortex_board as vb

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
        return vb.VortexBoard.new_vortex_board(size=5)

    # Returns a 1d ndarray indicating open nodes. 
    def get_available_actions(self, s):
        return s.open_cells
        
    # Update the state.
    def take_action(self, s, a):
        p = self.get_player(s)
        s = s.copy()
        s[:, p] += a.astype(np.float32) # Next move
        s[:, vb.NEXT_PLAYER] = (s[:, vb.NEXT_PLAYER] + 1) % 2 # Toggle player
        return s

    # Check for a side to side connection
    def check_game_over(self, s):
        return s.check_game_over()

    # Return 0 for X's turn or 1 for O's turn.
    def get_player(self, s):
        return int(s[0, vb.NEXT_PLAYER])

    # Fixed constant
    def get_num_players(self):
        return 2

    # Print a human-friendly visualization of the board.
    def visualize(self, s):
        s.print_board_str()

