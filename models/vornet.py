from torch.nn import Module, Linear, BatchNorm1d, ReLU, ModuleList, ReLU6

class MLP(Module):
    def __init__(self, in_features, layer0_dims, hidden_dims, hidden_layers, out_features):
        super().__init__()
        self._mlist = ModuleList([
            Linear(in_features, layer0_dims),
            BatchNorm1d(layer0_dims),
            ReLU()
        ])

        in_dim = layer0_dims        
        for i in range(hidden_layers):
            self._mlist.append(Linear(in_dim, hidden_dims))
            self._mlist.append(BatchNorm1d(hidden_dims))
            self._mlist.append(ReLU())
            in_dim = hidden_dims
        self._mlist.append(Linear(in_dim, out_features))

    def forward(self, x):
        for layer in self._mlist:
            x = layer(x)
        return x

    @classmethod
    def policy_head(cls, in_features):
        return cls(in_features, 250, 250, 3, 1)

    @classmethod
    def value_head(cls, in_features):
        return cls(in_features, 1000, 250, 3, 2)

class VorNet(Module):
    def __init__(self, input_shape, p_shape, v_shape):
        super(VorNet, self).__init__()
        self.p_shape = p_shape
        assert v_shape[0] == 2
        B, F = input_shape

        self.input_bn = BatchNorm1d(F)
        self.policy_head = MLP.policy_head(F)
        self.value_head = MLP.value_head(4*F)

    def forward(self, x):
        B, N, F = x.shape
        assert N == self.p_shape[0]

        x = self.input_bn(x.view(B*N, F)).view(B, N, F)

        side_nodes = x[:, -4:, :]
        x = x.view(B*N, F)
        p = self.policy_head(x)
        p = p.view(B, N)
        
        side_nodes = side_nodes.view(B, 4*F)
        v = self.value_head(side_nodes).tanh()

        return p, v