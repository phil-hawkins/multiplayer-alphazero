from torch.nn import Module, Linear, BatchNorm1d, ReLU, ModuleList
from torch_geometric.nn.conv import GCNConv, GINConv, GATConv, ChebConv, MessagePassing
from models.vornet import MLP

class GNNBaseline(Module):
    GNN = True

    def __init__(self, GNN, in_features, out_features, layers=19):
        super(GNNBaseline, self).__init__()

        self.out_features = out_features
        self._gnn_stack = ModuleList()
        for i in range(layers):
            self._gnn_stack.append(GNN(in_features, out_features))
            self._gnn_stack.append(BatchNorm1d(out_features))
            self._gnn_stack.append(ReLU())
            in_features = out_features

        self.policy_head = MLP.policy_head(2*out_features)
        self.value_head = MLP.value_head(2*4*out_features)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, edge_index, batch):
        B = batch.max().item() + 1
        assert B % 2 == 0, 'graphs must be paired, one for each player'
        N = (batch == 0).sum().item()
        assert batch.size(0) == B * N, 'All graphs in batch must have the same number of vertices'
        F = self.out_features

        for m in self._gnn_stack:
            if issubclass(m.__class__, MessagePassing):
                x = m(x, edge_index)
            else:
                x = m(x)

        B = B // 2
        x = x.view(B, 2, N, F)
        F = F * 2
        x = x.permute(0, 2, 3, 1).reshape(B, N, F)

        side_nodes = x[:, -4:, :]
        x = x.view(B*N, F)
        p = self.policy_head(x)
        p = p.view(B, N)
        
        side_nodes = side_nodes.view(B, 4*F)
        v = self.value_head(side_nodes).tanh()

        return p, v

class VornetGCN(GNNBaseline):
    def __init__(self, in_features=3, out_features=120, layers=19):
        super(VornetGCN, self).__init__(GCNConv, in_features, out_features, layers=layers)

class VornetGAT(GNNBaseline):
    def __init__(self, in_features=3, out_features=120, layers=19):
        super(VornetGCN, self).__init__(GATConv, in_features, out_features, layers=layers)

class VornetCheb(GNNBaseline):
    def __init__(self, in_features=3, out_features=120, layers=1, K=19):
        super(GNNBaseline, self).__init__()

        self.out_features = out_features
        self._gnn_stack = ModuleList()
        for i in range(layers):
            self._gnn_stack.append(ChebConv(in_features, out_features, K))
            self._gnn_stack.append(BatchNorm1d(out_features))
            self._gnn_stack.append(ReLU())
            in_features = out_features

        self.policy_head = MLP.policy_head(2*out_features)
        self.value_head = MLP.value_head(2*4*out_features)

class VornetGIN(GNNBaseline):
    def __init__(self, in_features=3, out_features=120, layers=19):
        super(GNNBaseline, self).__init__()

        self.out_features = out_features
        self._gnn_stack = ModuleList()
        for i in range(layers):
            nn = Linear(in_features=in_features, out_features=out_features)
            self._gnn_stack.append(GINConv(nn))
            self._gnn_stack.append(BatchNorm1d(out_features))
            self._gnn_stack.append(ReLU())
            in_features = out_features

        self.policy_head = MLP.policy_head(2*out_features)
        self.value_head = MLP.value_head(2*4*out_features)

class VornetGIN_3L(VornetGIN):
    def __init__(self, in_features=3, out_features=120):
        super(VornetGIN, self).__init__(in_features=3, out_features=120, layers=3)