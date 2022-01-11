import time
import numpy as np
from tqdm import trange
from multiprocessing.dummy import Pool as ThreadPool
import torch
from torch.utils.tensorboard import SummaryWriter

from mcts import MCTS, RolloutMCTS
from play import play_match
from players.uninformed_mcts_player import UninformedMCTSPlayer
from players.deep_mcts_player import DeepMCTSPlayer

# Object that coordinates AlphaZero training.
class Trainer:

    def __init__(self, game, nn, num_simulations, num_games, num_updates, buffer_size_limit, cpuct, num_threads, mcts_selfplay=False):
        self.game = game
        self.nn = nn
        self.num_simulations = num_simulations
        self.num_games = num_games
        self.num_updates = num_updates
        self.buffer_size_limit = buffer_size_limit
        self.training_data = np.zeros((0,3))
        self.cpuct = cpuct
        self.num_threads = num_threads
        self.error_log = []
        self.mcts_selfplay = mcts_selfplay


    # Does one game of self play and generates training samples.
    def self_play(self, temperature):
        s = self.game.get_initial_state()
        tree = RolloutMCTS(self.game) if self.mcts_selfplay else MCTS(self.game, self.nn)

        data = []
        scores = self.game.check_game_over(s)
        root = True
        alpha = 1
        weight = .25
        while scores is None:
            
            # Think
            for _ in range(self.num_simulations):
                tree.simulate(s, cpuct=self.cpuct)

            # Fetch action distribution and append training example template.
            dist = tree.get_distribution(s, temperature=temperature)

            # Add dirichlet noise to root
            if root:
                noise = np.random.dirichlet(np.array(alpha*np.ones_like(dist[:,1].astype(np.float32))))
                dist[:,1] = dist[:,1]*(1-weight) + noise*weight
                root = False

            data.append([s, dist[:,1], None]) # state, prob, outcome

            # Sample an action
            idx = np.random.choice(len(dist), p=dist[:,1].astype(np.float))
            a = tuple(dist[idx, 0])

            # Apply action
            available = self.game.get_available_actions(s)
            template = np.zeros_like(available)
            template[a] = 1
            s = self.game.take_action(s, template)

            # Check scores
            scores = self.game.check_game_over(s)

        # Update training examples with outcome
        for i, _ in enumerate(data):
            data[i][-1] = scores

        #return np.array(data)
        return np.array(data, dtype=object)


    # Performs one iteration of policy improvement.
    # Creates some number of games, then updates network parameters some number of times from that training data.
    def policy_iteration(self, verbose=False):
        temperature = 1   

        if verbose:
            print("SIMULATING " + str(self.num_games) + " games")
            start = time.time()
        if self.num_threads > 1:
            jobs = [temperature]*self.num_games
            pool = ThreadPool(self.num_threads)
            new_data = pool.map(self.self_play, jobs)
            pool.close()
            pool.join()
            self.training_data = np.concatenate([self.training_data] + new_data, axis=0)
        else:
            for _ in trange(self.num_games): # Self-play games
                new_data = self.self_play(temperature)
                self.training_data = np.concatenate([self.training_data, new_data], axis=0)
        if verbose:
            print("Simulating took " + str(int(time.time()-start)) + " seconds")

        # Prune oldest training samples if a buffer size limit is set.
        if self.buffer_size_limit is not None:
            self.training_data = self.training_data[-self.buffer_size_limit:,:]

        log_dir = "checkpoints/{}-{}/logs/mcts_init".format(self.nn.game.__class__.__name__, self.nn.model.module.__class__.__name__)
        writer = SummaryWriter(log_dir=log_dir)
        if verbose:
            print("TRAINING")
            start = time.time()
        mean_loss = None
        batch_size = self.nn.batch_size
        data_idx = np.arange(self.num_updates*batch_size) % len(self.training_data)
        np.random.shuffle(data_idx)

        with trange(self.num_updates) as steps:
            for step in steps:
                batch_idx = data_idx[step*batch_size:(step+1)*batch_size]
                batch = self.training_data[batch_idx]
                self.nn.train_step(batch, writer, step)
                new_loss = self.nn.latest_loss.item()
                mean_loss = new_loss if mean_loss is None else (mean_loss*step + new_loss)/(step+1)
                steps.set_postfix(loss=new_loss, mean_loss=mean_loss)
        self.error_log.append(mean_loss)
        writer.close()

        if verbose:
            print("Training took " + str(int(time.time()-start)) + " seconds")
            print("Average train error:", mean_loss)

