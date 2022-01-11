import json
import sys
import numpy as np

from models.senet import SENet
from models.senetbig import SENetBig
from models.vornet import VorNet, VorNetBN, VorNetBN3
from games.tictactoe import TicTacToe
from games.tictacmo import TicTacMo
from games.connect3x3 import Connect3x3
from games.vortex import Vortex_5_10, Vortex_5_20, Vortex_6_20, Vortex_7_20, Vortex_8_20, Vortex_9_20, Vortex_9_20_NoPT, Vortex_5_20_edge_weights
from neural_network import NeuralNetwork
from trainer import Trainer
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
trainer = Trainer(game=game, nn=nn, num_simulations=sims,
num_games=config["num_games"], num_updates=config["num_updates"], 
buffer_size_limit=config["buffer_size_limit"], cpuct=config["cpuct"],
num_threads=config["num_threads"])

# Logic for resuming training
checkpoints = nn.list_checkpoints()
if config["resume"]:
    if len(checkpoints) == 0:
        print("No existing checkpoints to resume.")
        quit()
    iteration = int(checkpoints[-1])
    trainer.training_data, trainer.error_log = nn.load(iteration, load_supplementary_data=True)
    #nn.load(iteration)
else:
    if len(checkpoints) != 0:
        print("Please delete the existing checkpoints for this game+model combination, or change resume flag to True.")
        quit()
    iteration = 0
    if "curriculum_checkpoint" in config:
        chk_iteration, chk_dir = config["curriculum_checkpoint"], config["curriculum_checkpoint_dir"]
        if config["verbose"]: 
            print("Continuing curriculum training from iteration {} ({}):".format(chk_iteration, chk_dir))
        nn.load(chk_iteration, directory=chk_dir)

# Training loop
while True:

    # Run multiple policy iterations to develop a checkpoint.
    for _ in range(config["ckpt_frequency"]):
        if config["verbose"]: print("Iteration:",iteration)
        trainer.policy_iteration(verbose=config["verbose"]) # One iteration of PI
        iteration += 1
        if config["verbose"]: print("Training examples:", len(trainer.training_data))
    
    # Save checkpoint
    nn.save(name=iteration, training_data=trainer.training_data, error_log=trainer.error_log)

    # Evaluate how the current checkpoint performs against MCTS agents of increasing strength
    # that do no use a heursitic.
    for opponent_strength in [10, 20, 40, 80, 160, 320, 640, 1280]:
        evaluate_against_mcts(checkpoint=iteration, game=game, model_class=model_class,
            my_sims=sims, opponent_sims=opponent_strength, cuda=cuda)
