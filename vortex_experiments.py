
import os
import torch
import pandas as pd
from itertools import permutations
import numpy as np
from time import time
from absl import app, flags, logging

from neural_network import NeuralNetwork
from players.uninformed_mcts_player import UninformedMCTSPlayer, RolloutMCTSPlayer
from players.deep_mcts_player import DeepMCTSPlayer
from games.vortex import Vortex_5, Vortex_6, Vortex_7, Vortex_8, Vortex_9
from models.vornet import VorNet

def play_match(game, players, verbose=False):
    """
    A match is two games played with the same vortex board topology. On the first game player 0 plays first, on the second player 1 plays first.

    Returns: score as list of results:
    - player 0 first player 0 wins
    - player 0 first player 1 wins
    - player 1 first player 0 wins
    - player 1 first player 1 wins

    Args:
        game: game definition foe the Multiplayer AlphaZero framework
        players: list of player agents
        verbose: verbose reporting
    """

    # permutations to break the dependence on player order in measuring strength.
    matches = list(permutations(np.arange(len(players))))
    
    # Initialize scoreboard
    scores = np.zeros((len(matches), game.get_num_players()))

    # initialise the Vortex board
    vortex_board = game.get_initial_state()

    # Run the matches
    for i, order in enumerate(matches):
        s = vortex_board.copy()

        for p in players:
            p.reset() # Clear player trees to make the next match fair

        game_over = game.check_game_over(s)

        while game_over is None:
            p = order[game.get_player(s)]
            if verbose: print("Player #{}'s turn.".format(p))
            s = players[p].update_state(s)
            game_over = game.check_game_over(s)

        scores[i, list(order)] += game_over

    scores = list(scores.flatten().astype(int))
    return scores


def simulations_compare(nn, dmcts_sims, bmcts_player_cls, bmcts_sims, match_n):
    """
    **Evaluate model strength by comparing simulation requirements with a baseline MCTS agent**

    This runs the DeepMCTS model against a baseline (not neural networl based) MCTS agent 
    - the DeepMCTS agent has a range of simulations counts
    - the MCTS agent uses a range of simulation counts
    For each combination, *match_n* matches are played.

    Args:
        nn: neural network for Deep MCTS agent
        dmcts_sims: iterable of simulation counts for Deep MCTS agent
        bmcts_player_cls: class of the baseline agent
        bmcts_sims: iterable of simulation counts for baseline MCTS agent
        match_n: number of matches to run, each match is two games

    Returns:
        df: pandas dataframe of results of playouts

    """
    game = nn.game
    results = []

    for dmcts_sim in dmcts_sims:
        for bmcts_sim in bmcts_sims:
            deep_mcts = DeepMCTSPlayer(game, nn, simulations=dmcts_sim)
            rollout_mcts = bmcts_player_cls(game, simulations=bmcts_sim)
            players = [deep_mcts, rollout_mcts]

            print("DMCTS: {}, BMCTS: {}".format(dmcts_sim, bmcts_sim))

            for i in range(match_n):
                match = [dmcts_sim, bmcts_sim, i]
                tic = time()
                scores = play_match(game, players, verbose=False)
                toc = time()
                print("  match {}, score {}, time {:.2f}s".format(i, scores, toc-tic))
                results.append(match + scores)

    players = ['DMCTS', 'BMCTS']
    index = pd.MultiIndex.from_product([players, players], names=['first_player', 'score_player'])
    df = pd.DataFrame(data=results, columns=['dmcts_sim', 'bmcts_sim', 'match', 'p0fp_p0win', 'p0fp_p1win', 'p1fp_p0win', 'p1fp_p1win'])
    df = df.set_index(['dmcts_sim', 'bmcts_sim', 'match'])
    df.columns = index

    return df

def experiment_1():
    """
    Compare:
    - Agent 1: network trained by MCTS self play, then NN self play on 5x5 Vortex board - 100 simulations
    - Agent 2: uninformed MCTS without rollout - range of sims
    """
    dmcts_sims = [100]
    bmcts_sims = list(range(100, 401, 100)) + list(range(500, 2501, 500))
    match_n = 500
    nn = NeuralNetwork(Vortex_5(), VorNet, cuda=True)
    nn.load(99)

    results_df = simulations_compare(
        nn=nn,
        dmcts_sims=dmcts_sims, 
        bmcts_player_cls=UninformedMCTSPlayer,
        bmcts_sims=bmcts_sims, 
        match_n=match_n
    )
    return results_df

def experiment_2():
    """
    Compare:
    - Agent 1: network trained by MCTS self play, then NN self play on 5x5 Vortex board - 100 simulations
    - Agent 2: standard MCTS (with rollout) - range of sims
    """
    dmcts_sims = [100]
    bmcts_sims = list(range(100, 401, 100)) + list(range(500, 2501, 500))
    match_n = 500
    nn = NeuralNetwork(Vortex_5(), VorNet, cuda=True)
    nn.load(99)

    results_df = simulations_compare(
        nn=nn,
        dmcts_sims=dmcts_sims, 
        bmcts_player_cls=RolloutMCTSPlayer,
        bmcts_sims=bmcts_sims, 
        match_n=match_n
    )
    return results_df


def board_compare(Baseline_MCTSPlayer, match_n):
    # **Evaluate performance of DeepMCTS on larger boards than the 25 node board it was trained on**
    # 
    # The DeepMCTS agent is run against the uniformed MCTS agent. For each board size, *match_n* matches are played.

    dmcts_sim = 250
    bmcts_sims = range(250, 1001, 250)
    board_sizes = [Vortex_5, Vortex_6, Vortex_7, Vortex_8, Vortex_9]
    directory = "checkpoints/Vortex_5-VorNet"

    results = []
    for Game in board_sizes:
        board_size = Game.__name__
        game = Game()
        nn = NeuralNetwork(game, VorNet, cuda=True)
        nn.load(99, directory=directory)

        deep_mcts = DeepMCTSPlayer(game, nn, simulations=dmcts_sim)

        print("Board size: {}".format(board_size))

        for bmcts_sim in bmcts_sims:
            print("  bmcts_sim: {}".format(bmcts_sim))
            uninformed = Baseline_MCTSPlayer(game, simulations=bmcts_sim)
            players = [deep_mcts, uninformed]

            for i in range(match_n):
                scores = play_match(game, players, verbose=False)
                print("    match {}, score {}".format(i, scores))
                match = [board_size, bmcts_sim, i]
                results.append(match + scores)

    player_names = [p.__class__.__name__[:-6] for p in players]
    index = pd.MultiIndex.from_product([player_names, player_names], names=['first_player', 'score_player'])
    df = pd.DataFrame(data=results, columns=['board_size', 'bmcts_sim', 'match', 'p0fp_p0win', 'p0fp_p1win', 'p1fp_p0win', 'p1fp_p1win'])
    df = df.set_index(['board_size', 'bmcts_sim', 'match'])
    df.columns = index

    return df

def experiment_3():
    df = board_compare(Baseline_MCTSPlayer=UninformedMCTSPlayer, match_n=100)
    return df

def experiment_4():
    df = board_compare(Baseline_MCTSPlayer=RolloutMCTSPlayer, match_n=100)
    return df



'''

# %% [markdown]
# How does the curriculum trained model compare to the non-pretrained model?
# 
# (board size, iterations, training_steps)
# 
# curriculum:
# - vortex-5-20-gpu.json (5, 100, 1000)
# - vortex-6-20-gpu-curriculum.json (6, 10, 1000)
# - vortex-7-20-gpu-curriculum.json (7, 10, 1000)
# - vortex-8-20-gpu-curriculum.json (8, 10, 1000)
# - vortex-9-20-gpu-curriculum.json (9, 10, 10000)
# 
# non-pretrained:
# - vortex-9-20-gpu-nopt.json (9, 10, 10000)

# %%
nopt_checkpoints = range(10, 61, 10)
dmcts_sim=100
match_n=50

# %%
game = Vortex_9_20()
nn1 = NeuralNetwork(game, VorNet, cuda=True)
nn1.load(10, directory='checkpoints/Vortex_9_20-VorNet')
nn2 = NeuralNetwork(game, VorNet, cuda=True)
results = []

for nopt_checkpoint in nopt_checkpoints:
    nn2.load(nopt_checkpoint, directory='checkpoints/Vortex_9_20_NoPT-VorNet')

    deep_mcts1 = DeepMCTSPlayer(game, nn1, simulations=dmcts_sim)
    deep_mcts2 = DeepMCTSPlayer(game, nn2, simulations=dmcts_sim)
    players = [deep_mcts1, deep_mcts2]

    for i in range(match_n):
        match = [nopt_checkpoint, i]
        scores = play_match(game, players, verbose=False)
        print("  match {}, score {}".format(i, scores))
        results.append(match + scores)

# %%
df = pd.DataFrame(data=results, columns=['NoPT_checkpoint', 'match', 'p0fp_p0win', 'p0fp_p1win', 'p1fp_p0win', 'p1fp_p1win'])
df = df.set_index(['NoPT_checkpoint', 'match'])
players = ['curriculum', 'non-pretrained']
index = pd.MultiIndex.from_product([players, players], names=['first_player', 'score_player'])
df.columns = index
df.to_csv("./notebooks/results/curriculum.csv")
df.groupby(['NoPT_checkpoint']).sum()

# %%
df = pd.read_csv("./notebooks/results/curriculum.csv", skiprows=[1,2])
df.columns=['NoPT_checkpoint', 'match', 'p0fp_p0win', 'p0fp_p1win', 'p1fp_p0win', 'p1fp_p1win']
df = df.set_index(['NoPT_checkpoint', 'match'])
players = ['curriculum', 'non-pretrained']
index = pd.MultiIndex.from_product([players, players], names=['first_player', 'score_player'])
df.columns = index
df.to_csv("./notebooks/results/curriculum.csv")
df.groupby(['NoPT_checkpoint']).sum()
'''


FLAGS = flags.FLAGS
flags.DEFINE_integer('run_exp', 1, 'experiment to run')

def main(_argv):
    print("Running experiment {}".format(FLAGS.run_exp))
    start_time = time()
    experiments = {
        1 : experiment_1,
        2 : experiment_2,
        3 : experiment_3,
        4 : experiment_4
    }
    df = experiments[FLAGS.run_exp]()
    df.to_csv("./exp_results/experiment{}_{}-{}.csv".format(FLAGS.run_exp, start_time, time()))

if __name__ == '__main__':
    app.run(main)