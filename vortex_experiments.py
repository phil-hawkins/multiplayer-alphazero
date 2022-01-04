from itertools import permutations
import numpy as np

# Runs a match with the given game and list of players.
# Returns an array of points. Player number is the index into this array.
# For each match, a player gains a point if it wins, loses a point if it loses,
# and gains no points if it ties.
# the vortex board is gereated once for both players
def play_match(game, players, verbose=False):

    # permutations break the dependence on player order in measuring strength.
    matches = list(permutations(np.arange(len(players))))
    
    # Initialize scoreboard
    scores = np.zeros(game.get_num_players())

    # initialise the Vortex board
    vortex_board = game.get_initial_state()

    # Run the matches
    for order in matches:
        s = vortex_board.copy()

        for p in players:
            p.reset() # Clear player trees to make the next match fair

        #if verbose: game.visualize(s)
        game_over = game.check_game_over(s)

        while game_over is None:
            p = order[game.get_player(s)]
            if verbose: print("Player #{}'s turn.".format(p))
            s = players[p].update_state(s)
            #if verbose: game.visualize(s)
            game_over = game.check_game_over(s)

        scores[list(order)] += game_over
        if verbose: print("Δ" + str(game_over[list(order)]) + ", Current scoreboard: " + str(scores))


    if verbose: print("Final scores:", scores)
    return scores



if __name__ == "__main__":
    from players.human_player import HumanPlayer
    from neural_network import NeuralNetwork
    from players.uninformed_mcts_player import UninformedMCTSPlayer
    from players.deep_mcts_player import DeepMCTSPlayer
    from games.vortex import Vortex, Vortex7, Vortex11, Vortex_5_20, Vortex_5_10
    from models.vornet import VorNet


    # Change these variable 
    game = Vortex_5_20()
    nn = NeuralNetwork(game, VorNet, cuda=True)
    nn.load('136')
    deep_mcts = DeepMCTSPlayer(game, nn, simulations=1000)
    uninformed = UninformedMCTSPlayer(game, simulations=1000)
    
    players = [deep_mcts, uninformed]
    scores = []
    for _ in range(10):
        scores.append(play_match(game, players, verbose=True))
    
    scores = np.stack(scores).sum(axis=0)
    print('Deep MCTS:       ', scores[0])
    print('Uninformed MCTS: ', scores[1])