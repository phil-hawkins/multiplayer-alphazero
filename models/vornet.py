from torch.nn import Module, Linear, BatchNorm1d, ReLU, ModuleList, Dropout

class MLP(Module):
    def __init__(self, in_features, hidden_dims, hidden_layers, out_features, dropout):
        super().__init__()
        self._mlist = ModuleList([
            Linear(in_features, hidden_dims),
            ReLU()
        ])
        for _ in range(hidden_layers):
            self._mlist.append(Linear(hidden_dims, hidden_dims))
            self._mlist.append(BatchNorm1d(hidden_dims))
            self._mlist.append(ReLU())
            self._mlist.append(Dropout(p=dropout))
        self._mlist.append(Linear(hidden_dims, out_features))

    def forward(self, x):
        for layer in self._mlist:
            x = layer(x)
        return x

    @classmethod
    def policy_head(cls, in_features):
        return cls(in_features, 512, 10, 1, 0.1)

    @classmethod
    def value_head(cls, in_features):
        return cls(in_features, 512, 10, 2, 0.1)

class VorNet(Module):
    def __init__(self, input_shape, p_shape, v_shape):
        super(VorNet, self).__init__()
        self.p_shape = p_shape
        assert v_shape == 2
        B, F = input_shape

        self.input_bn = BatchNorm1d(F)
        self.policy_head = MLP(F)
        self.value_head = MLP(4*F)

    def forward(self, x):
        B, N, F = x.shape
        assert N == self.p_shape

        x = self.input_bn(x.view(B*N, F)).view(B, N, F)

        side_nodes = x[:, -4:, :]
        x = x.view(B*N, F)
        p = self.policy_head(x)
        p = p.view(B, N)
        
        side_nodes = side_nodes.view(B, 4*F)
        v = self.value_head(side_nodes).tanh()

        return p, v