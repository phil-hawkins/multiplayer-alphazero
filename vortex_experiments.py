
import os
import torch
import pandas as pd
from itertools import permutations
import numpy as np
from time import time

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
    start_time = time()
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
    results_df.to_csv("./exp_results/experiment1_{}-{}.csv".format(start_time, time()))

def experiment_2():
    """
    Compare:
    - Agent 1: network trained by MCTS self play, then NN self play on 5x5 Vortex board - 100 simulations
    - Agent 2: standard MCTS (with rollout) - range of sims
    """
    start_time = time()
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
    results_df.to_csv("./exp_results/experiment2_{}-{}.csv".format(start_time, time()))

'''
# # %%
# df = pd.read_csv("./exp_results/simulations_compare.csv", skiprows=[0,1]) #, index_col=[0,1])
# df = df.set_index(['dmcts_sim', 'umcts_sim', 'match'])
# df.columns = ['p0fp_p0win', 'p0fp_p1win', 'p1fp_p0win', 'p1fp_p1win']
# df
# players = ['DMCTS', 'UMCTS']
# index = pd.MultiIndex.from_product([players, players], names=['first_player', 'score_player'])
# df.columns = index
# df.groupby(['dmcts_sim', 'umcts_sim']).sum()

# %% [markdown]
# **Evaluate performance of DeepMCTS on larger boards than the 25 node board it was trained on**
# 
# The DeepMCTS agent is run against the uniformed MCTS agent. For each board size, *match_n* matches are played.

# %%
dmcts_sim = 40
umcts_sim = 320
board_sizes = [Vortex_5_20, Vortex_6_20, Vortex_7_20, Vortex_8_20, Vortex_9_20]
match_n = 100
directory = "checkpoints/Vortex_5_20-VorNet"

# %%
results = []
for Game in board_sizes:
    board_size = Game.__name__
    game = Game()
    nn = NeuralNetwork(game, VorNet, cuda=True)
    nn.load(checkpoint, directory=directory)

    deep_mcts = DeepMCTSPlayer(game, nn, simulations=dmcts_sim)
    uninformed = UninformedMCTSPlayer(game, simulations=umcts_sim)
    players = [deep_mcts, uninformed]

    print("Board size: {}".format(board_size))

    for i in range(match_n):
        match = [board_size, i]
        scores = play_match(game, players, verbose=False)
        print("  match {}, score {}".format(i, scores))
        results.append(match + scores)

# %%
players = [0,1]
index = pd.MultiIndex.from_product([players, players], names=['first_player', 'score_player'])
df = pd.DataFrame(data=results, columns=['board_size', 'match', 'p0fp_p0win', 'p0fp_p1win', 'p1fp_p0win', 'p1fp_p1win'])
df = df.set_index(['board_size', 'match'])
df.columns = index
df.to_csv("./notebooks/results/board_sizes.csv")
df.groupby(['board_size']).sum()


# %% [markdown]
# Simulating games to generate training data is time consuming on larger boards because untrained agents act close to randomly and this results in more moves required to reach a win state. 
# 
# Do models pretrained to smaller boards reduce this initial game simulation time?

# %%
# times for 60 games of 36 node Vortex, sides of 6, 100 simulations per action (tree search)
sim_times = [216, 215, 205, 205, 216, 204, 219, 209, 200, 203]
mean_game_time = sum(sim_times) / (len(sim_times) * 30)
print(mean_game_time)

# side 7
sim_times = [339, 366, 448, 518, 499, 490, 471, 457, 467, 452]
mean_game_time = sum(sim_times) / (len(sim_times) * 30)
print(mean_game_time)

# side 8
sim_times = [588, 583, 543, 527, 529, 523, 554, 537, 570, 564]
mean_game_time = sum(sim_times) / (len(sim_times) * 30)
print(mean_game_time)

# side 9
sim_times = [1085, 983, 988, 977, 1016, 933, 974, 1032, 971, 947]
mean_game_time = sum(sim_times) / (len(sim_times) * 30)
print(mean_game_time)

# side 9 - no pretraining
sim_times = [1017, 1088, 1118, 1079, 1076, 1085, 1084, 1029, 1125, 1031]
mean_game_time = sum(sim_times) / (len(sim_times) * 30)
print(mean_game_time)

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


experiment_1()