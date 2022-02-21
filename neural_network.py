import torch
import numpy as np
import os
import shutil
from games.hex.vortex_board import VortexBoard
from games.vortex import Vortex_5


# Object that manages interfacing data with the underlying PyTorch model, as well as checkpointing models.
class NeuralNetwork():

    def __init__(self, game, model_class, lr=1e-3, weight_decay=1e-8, batch_size=64, cuda=False):
        self.game = game
        self.batch_size = batch_size
        input_shape = game.get_initial_state().shape
        p_shape = game.get_available_actions(game.get_initial_state()).shape
        v_shape = (game.get_num_players(),)
        self.model = model_class(input_shape, p_shape, v_shape)
        self.cuda = cuda
        if self.cuda:
            self.model = self.model.to('cuda')
            self.model = torch.nn.DataParallel(self.model)
        if len(list(self.model.parameters())) > 0:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def get_batch_states(self, batch):
        states = np.stack(batch[:,0])
        if isinstance(batch[0][0], VortexBoard):
            nn_states = [s.nn_attr for s in states]
            nn_states = np.stack(nn_states)
        else:
            nn_states = states
            
        return nn_states, states

    # Incoming data is a numpy array containing (state, prob, outcome) tuples.
    def train(self, data):
        self.model.train()
        batch_size=self.batch_size
        idx = np.random.randint(len(data), size=batch_size)
        batch = data[idx]
        nn_states, states = self.get_batch_states(batch)
        x = torch.from_numpy(nn_states)
        p_pred, v_pred = self.model(x)
        p_gt, v_gt = batch[:,1], np.stack(batch[:,2])
        loss = self.loss(states, (p_pred, v_pred), (p_gt, v_gt))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.latest_loss = loss

    def train_step(self, batch, writer, step):
        self.model.train()
        nn_states, states = self.get_batch_states(batch)
        x = torch.from_numpy(nn_states)
        p_pred, v_pred = self.model(x)
        p_gt, v_gt = batch[:,1], np.stack(batch[:,2])
        loss = self.loss(states, (p_pred, v_pred), (p_gt, v_gt))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.latest_loss = loss
        writer.add_scalar('loss', loss, step)

    # Given a single state s, does inference to produce a distribution of valid moves P and a value V.
    def predict(self, s):
        self.model.eval()
        with torch.no_grad():
            if isinstance(s, VortexBoard):
                input_s = np.expand_dims(s.nn_attr, axis=0)
            else:
                input_s = np.array([s])
            input_s = torch.from_numpy(input_s)
            p_logits, v = self.model(input_s)
            p, v = self.get_valid_dist(s, p_logits[0]).cpu().numpy().squeeze(), v.cpu().numpy().squeeze() # EXP because log softmax
        return p, v


    # MSE + Cross entropy
    def loss(self, states, prediction, target):
        batch_size = len(states)
        p_pred, v_pred = prediction
        p_gt, v_gt = target
        v_gt = torch.from_numpy(v_gt.astype(np.float32))
        if self.cuda:
            v_gt = v_gt.cuda()
        v_loss = ((v_pred - v_gt)**2).sum() # Mean squared error
        p_loss = 0
        for i in range(batch_size):
            gt = torch.from_numpy(p_gt[i].astype(np.float32))
            if self.cuda:
                gt = gt.cuda()
            s = states[i]
            logits = p_pred[i]
            pred = self.get_valid_dist(s, logits, log_softmax=True)
            p_loss += -torch.sum(gt*pred)
        return p_loss + v_loss


    # Takes one state and logit set as input, produces a softmax/log_softmax over the valid actions.
    def get_valid_dist(self, s, logits, log_softmax=False):
        mask = torch.from_numpy(self.game.get_available_actions(s).astype(np.uint8)).bool()
        if self.cuda:
            mask = mask.cuda()
        selection = torch.masked_select(logits, mask)
        dist = torch.nn.functional.log_softmax(selection, dim=-1)
        if log_softmax:
            return dist
        return torch.exp(dist)


    # Saves the current network along with its current pool of training data and training error history.
    # Provide the name of the save file.
    def save(self, name, training_data, error_log, timing):
        network_name = self.model.module.__class__.__name__ if self.cuda else self.model.__class__.__name__
        directory = "checkpoints/{}-{}".format(self.game.__class__.__name__, network_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        network_path = "{}/{}.ckpt".format(directory, name)
        data_path = "{}/training.data".format(directory)
        if os.path.isfile(data_path):
            print(data_path + ' found, making backup')
            shutil.copy2(data_path, data_path+'.bak')
        else:
            print(data_path + ' not found')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'error_log': error_log,
            'timing': timing,
            }, network_path)
        torch.save({
            'training_data': training_data,
            }, data_path)


    # Loads the network at the given name.
    # Optionally, also load and return the training data and training error history.
    def load(self, name, load_supplementary_data=False, directory=None):
        network_name = self.model.module.__class__.__name__ if self.cuda else self.model.__class__.__name__
        if directory is None:
            directory = "checkpoints/{}-{}".format(self.game.__class__.__name__, network_name)
        network_path = "{}/{}.ckpt".format(directory, name)
        network_checkpoint = torch.load(network_path)
        self.model.load_state_dict(network_checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(network_checkpoint['optimizer_state_dict'])
        if load_supplementary_data:
            data_path = "{}/training.data".format(directory)
            data_checkpoint = torch.load(data_path)
            return data_checkpoint['training_data'], network_checkpoint['error_log']


    # Utility function for listing all available model checkpoints.
    def list_checkpoints(self):
        network_name = self.model.module.__class__.__name__ if self.cuda else self.model.__class__.__name__
        path = "checkpoints/{}-{}/".format(self.game.__class__.__name__, network_name)
        if  not os.path.isdir(path):
            return []
        return sorted([int(filename.split(".ckpt")[0]) for filename in os.listdir(path) if filename.endswith(".ckpt")])

