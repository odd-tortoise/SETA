import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

# ──────────────────────────────────────────────────────────────────────────────
# 1.1. Abstract Base Class
# ──────────────────────────────────────────────────────────────────────────────

class DecisionNetwork(nn.Module):
    """
    Abstract base class for a decision network. Concretely,
    subclasses must implement:
      - forward(…)
      - reset()         (if they have any internal state to clear between sims)
    """
    def reset(self):
        """
        Optional: called once at the beginning of each new simulation.
        Default no‐op if subclass has no temporal hidden state.
        """
        pass


# ──────────────────────────────────────────────────────────────────────────────
# 1.2. MLP Decision Network
# ──────────────────────────────────────────────────────────────────────────────

class MLPDecisionNetwork(DecisionNetwork):
    """
    MLP-based decision network: input = [W_count, temperature, time_step] (shape (1,3))
    → hidden layers → output scalar (Softplus→positive).
    """
    def __init__(self, hidden_sizes: list = [32, 32]):
        super().__init__()
        dims = [3] + hidden_sizes + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: Tensor of shape (1, 3)
        returns : Tensor of shape (1,) after Softplus → nonnegative scalar
        """
        out = self.net(features)                # (1, 1)
        return F.softplus(out).view(-1)         # (1,)


# ──────────────────────────────────────────────────────────────────────────────
# 1.3. Linear Decision Network
# ──────────────────────────────────────────────────────────────────────────────

class LinearDecisionNetwork(DecisionNetwork):
    """
    Single linear layer from 3→1, followed by Softplus.
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.linear(features)             # (1, 1)
        return F.softplus(out).view(-1)         # (1,)


# ──────────────────────────────────────────────────────────────────────────────
# 1.4. LSTM Decision Network
# ──────────────────────────────────────────────────────────────────────────────

class LSTMDecisionNetwork(DecisionNetwork):
    """
    LSTMCell‐based: at each time step, feed [W_count, temp, t] into an LSTMCell,
    update (hx, cx) internally, then map hx→ scalar via a linear layer + Softplus.
    """
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size=3, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.reset()

    def reset(self):
        """
        Zero out hidden & cell states before each new simulation.
        """
        self.hx = None
        self.cx = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: Tensor of shape (1, 3)
        returns : Tensor of shape (1,) after Softplus
        """
        if self.hx is None or self.cx is None:
            device = features.device
            self.hx = torch.zeros(1, self.hidden_size, device=device)
            self.cx = torch.zeros(1, self.hidden_size, device=device)

        self.hx, self.cx = self.lstm_cell(features, (self.hx, self.cx))
        out = self.linear(self.hx)               # (1, 1)
        return F.softplus(out).view(-1)          # (1,)


# ──────────────────────────────────────────────────────────────────────────────
# 1.5. GNN Decision Network
# ──────────────────────────────────────────────────────────────────────────────

class GNNDecisionNetwork(DecisionNetwork):
    """
    GNN‐based: on each time‐step, receives:
      - ages:      Tensor shape (N, 1)  (each node’s age)
      - edge_index: LongTensor shape (2, E)
      - batch:     LongTensor shape (N,) (all zeros if single graph)
      - temp, t:   floats or 0‐d Tensors
    It applies a small stack of GraphSAGE layers to ages→h (N, hidden_dim),
    then does global_mean_pool(h, batch) → c (1, hidden_dim), 
    maps c→ z (1, vector_dim), concatenates [z, temp, t] → shape (1, vector_dim+2),
    then final MLP→ scalar (Softplus→positive).
    """
    def __init__(
        self,
        node_in_dim: int = 1,
        hidden_dim: int = 32,
        num_gnn_layers: int = 2,
        vector_dim: int = 5
    ):
        super().__init__()
        # build GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(node_in_dim, hidden_dim))
        for _ in range(num_gnn_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # map pooled embedding → vector_dim
        self.to_vector = nn.Linear(hidden_dim, vector_dim)

        # final MLP: (vector_dim + 2) → hidden_dim → 1 → Softplus
        self.mlp = nn.Sequential(
            nn.Linear(vector_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def reset(self):
        """
        No recurrent state, but define so that `.reset()` exists.
        """
        pass

    def forward(
        self,
        ages: torch.Tensor,        # (N, 1)
        edge_index: torch.Tensor,  # (2, E)
        batch: torch.Tensor,       # (N,)
        temp: float,
        t: float
    ) -> torch.Tensor:
        h = ages                                  # (N, 1)
        for conv in self.convs:
            h = conv(h, edge_index)               # (N, hidden_dim)
            h = F.relu(h)

        # global‐mean‐pool to (1, hidden_dim)
        c = global_mean_pool(h, batch)            # (1, hidden_dim)

        # project to (1, vector_dim)
        z = self.to_vector(c)                     # (1, vector_dim)

        extras = torch.tensor([[float(temp), float(t)]], device=ages.device)  # (1, 2)
        feat = torch.cat([z, extras], dim=1)       # (1, vector_dim+2)

        out = self.mlp(feat)                       # (1, 1)
        return out.view(-1)                        # (1,)
