from torch.nn import Module, Linear, Tanh, ReLU, ModuleList
import numpy as np

class MLP(Module):
    def __init__(self, in_features, hidden_dims, hidden_layers, out_features):
        super().__init__()
        self._mlist = ModuleList([
            Linear(in_features, hidden_dims),
            ReLU()
        ])
        for _ in range(hidden_layers):
            self._mlist.append(Linear(hidden_dims, hidden_dims))
            self._mlist.append(ReLU())
        self._mlist.append(Linear(hidden_dims, out_features))

    def forward(self, x):
        for layer in self._mlist:
            x = layer(x)
        return x


# Interface for defining a PyTorch model.
# See the models folder for examples.
class VorNet(Module):

    def __init__(self, input_shape, p_shape, v_shape):
        super(VorNet, self).__init__()
        self.input_shape = input_shape
        self.p_shape = p_shape
        self.v_shape = v_shape
        B, F = input_shape
        self.policy_head = MLP(F, 256, 2, 1)
        self.value_head = MLP(4*F, 512, 2, np.prod(self.v_shape))

    # Simply define the forward pass.
    # Your input will be a batch of ndarrays representing board state.
    # Your output must have both a policy and value head, so you must return 2 tensors p, v in that order.
    # The policy head must be logits, and the value head must be passed through a tanh non-linearity.
    # Policy softmaxing is handled for you in neural_network.py.
    def forward(self, x):
        B, N, F = x.shape

        side_nodes = x[:, -4:, :]
        x = x.view(B*N, F)
        p = self.policy_head(x)
        p = p.view(B, np.prod(self.p_shape))
        
        side_nodes = side_nodes.view(B, 4*F)
        v = self.value_head(side_nodes).tanh()

        return p, v
