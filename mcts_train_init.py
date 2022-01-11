""" 
Generates initial training data by self play of MCTS and pretrains network on it to
create the first iteration checkpoint

"""

import json
import sys
import numpy as np

from models.senet import SENet
from models.senetbig import SENetBig
from models.vornet import VorNet, VorNetBN, VorNetBN3
from games.tictactoe import TicTacToe
from games.tictacmo import TicTacMo
from games.connect3x3 import Connect3x3
from games.vortex import Vortex_5_10, Vortex_5_20, Vortex_6_20, Vortex_7_20, Vortex_8_20, Vortex_9_20, Vortex_9_20_NoPT, Vortex_5_20_edge_weights, Vortex_5_mctspt
from neural_network import NeuralNetwork
from trainer2 import Trainer
from experiments import evaluate_against_uninformed, evaluate_against_mcts


# Load in a run configuration
with open(sys.argv[1], "r") as f:
    config = json.loads(f.read())

# Instantiate
game = globals()[config["game"]]()
model_class = globals()[config["model"]]
sims = config["num_simulations"]
cuda = config["cuda"]
nn = NeuralNetwork(game=game, model_class=model_class, lr=config["lr"],
    weight_decay=config["weight_decay"], batch_size=config["batch_size"], cuda=cuda)

# Generate training data from self play by MCTS with rollout
trainer = Trainer(
    game=game, 
    nn=nn, 
    num_simulations=2000,
    num_games=1000, 
    num_updates=1000, 
    buffer_size_limit=config["buffer_size_limit"], 
    cpuct=config["cpuct"],
    num_threads=config["num_threads"],
    mcts_selfplay=True)

print("Generating MCTS self play data and training")
trainer.policy_iteration(verbose=config["verbose"]) # One iteration of PI
print("Training examples:", len(trainer.training_data))
    
# Save checkpoint
iteration = 0
nn.save(name=iteration, training_data=trainer.training_data, error_log=trainer.error_log)


# Evaluate against MCTS agent with the same number of sims
evaluate_against_mcts(checkpoint=iteration, game=game, model_class=model_class,
        my_sims=sims, opponent_sims=sims, cuda=cuda)

# Evaluate how the current checkpoint performs against MCTS agents of increasing strength
# that do no use a heursitic.
for opponent_strength in [10, 20, 40, 80, 160, 320, 640, 1280]:
    evaluate_against_mcts(checkpoint=iteration, game=game, model_class=model_class,
        my_sims=sims, opponent_sims=opponent_strength, cuda=cuda)
