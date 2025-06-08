import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import fields
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool

# Assume System and AgentState are defined elsewhere and imported here.

# ----------------------------
# 1) Abstract Base Class: ThinkerBase
# ----------------------------

class ThinkerBase(nn.Module, ABC):
    """
    Abstract base class for decision “Thinkers”.  
    Subclasses must implement:
      - build_network() -> nn.Module
      - extract_features(system, t: int) -> Tensor or tuple (for GNN)
      - process_output(raw: Tensor) -> Tensor(final)
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.net = self.build_network().to(device)

    def reset(self):
        """If the network has internal state (e.g. LSTMCell), reset here."""
        if hasattr(self.net, "reset"):
            self.net.reset()

    @abstractmethod
    def build_network(self) -> nn.Module:
        """Construct and return the internal nn.Module."""
        pass

    @abstractmethod
    def extract_features(self, system: Any, t: int) -> Any:
        """
        Given a System instance and time t, return the input(s) for self.net:
          - For MLP/Linear/LSTM: a single 1×D Tensor
          - For GNN: a tuple (node_feats, edge_index, batch)
        """
        pass

    @abstractmethod
    def process_output(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Given the raw output of self.net, return the final Tensor.
        E.g. apply Softplus or slice out components.
        """
        pass

    def forward(self, system: Any, t: int) -> torch.Tensor:
        feats = self.extract_features(system, t)
        # If extract_features returned a tuple (for GNN), unpack it
        if isinstance(feats, tuple):
            raw = self.net(*[f.to(self.device) for f in feats])
        else:
            raw = self.net(feats.to(self.device))
        return self.process_output(raw)


# ----------------------------
# 2) MLPThinker
# ----------------------------

class MLPThinker(ThinkerBase):
    """
    MLP Thinker: features = [W_count, temperature, t] → MLP → Softplus → scalar
    """

    def __init__(
        self,
        hidden_sizes: list = [32, 32],
        device: torch.device = torch.device("cpu")
    ):
        self.hidden_sizes = hidden_sizes
        super().__init__(device=device)

    def build_network(self) -> nn.Module:
        dims = [3] + self.hidden_sizes + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        return nn.Sequential(*layers)

    def extract_features(self, system: Any, t: int) -> torch.Tensor:
        W_count = system.types.count("W")
        temp = system.get_system_var("temperature")
        return torch.tensor([[W_count, temp, float(t)]], dtype=torch.float32, device=self.device)

    def process_output(self, raw: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(raw).view(-1)  # shape (1,)


# ----------------------------
# 3) LinearThinker
# ----------------------------

class LinearThinker(ThinkerBase):
    """
    Linear Thinker: features = [W_count, temperature, t] → Linear(3→1) → Softplus → scalar
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__(device=device)

    def build_network(self) -> nn.Module:
        return nn.Linear(3, 1)

    def extract_features(self, system: Any, t: int) -> torch.Tensor:
        W_count = system.types.count("W")
        temp = system.get_system_var("temperature")
        return torch.tensor([[W_count, temp, float(t)]], dtype=torch.float32, device=self.device)

    def process_output(self, raw: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(raw).view(-1)


# ----------------------------
# 4) LSTMThinker
# ----------------------------

class LSTMThinker(ThinkerBase):
    """
    LSTMThinker: uses an LSTMCell over the 3-dim feature vector.
      hidden_size param determines hidden dimension.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        device: torch.device = torch.device("cpu")
    ):
        self.hidden_size = hidden_size
        super().__init__(device=device)

    def build_network(self) -> nn.Module:
        # We wrap the LSTMCell and a final Linear in a small helper module
        return _LSTMWrapper(input_size=3, hidden_size=self.hidden_size, device=self.device)

    def extract_features(self, system: Any, t: int) -> torch.Tensor:
        W_count = system.types.count("W")
        temp = system.get_system_var("temperature")
        return torch.tensor([[W_count, temp, float(t)]], dtype=torch.float32, device=self.device)

    def process_output(self, raw: torch.Tensor) -> torch.Tensor:
        # raw is shape (1,1) coming from the linear after LSTMCell
        return nn.functional.softplus(raw).view(-1)


class _LSTMWrapper(nn.Module):
    """
    Internal helper: maintains LSTMCell hidden/cell state,
    then applies a final Linear → raw scalar.
    """

    def __init__(self, input_size: int, hidden_size: int, device: torch.device):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size).to(device)
        self.linear = nn.Linear(hidden_size, 1).to(device)
        self.device = device
        self.reset()

    def reset(self):
        self.hx = None
        self.cx = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (1, input_size)
        if self.hx is None or self.cx is None:
            self.hx = torch.zeros(1, self.lstm_cell.hidden_size, device=self.device)
            self.cx = torch.zeros(1, self.lstm_cell.hidden_size, device=self.device)
        self.hx, self.cx = self.lstm_cell(x, (self.hx, self.cx))
        out = self.linear(self.hx)  # (1,1)
        return out


# ----------------------------
# 5) GNNThinker
# ----------------------------

class GNNThinker(ThinkerBase):
    """
    GNNThinker: uses a small GraphSAGE network + global_mean_pool, then an MLP.
      Final output is Softplus of the MLP’s scalar output.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        num_gnn_layers: int = 2,
        vector_dim: int = 5,
        device: torch.device = torch.device("cpu")
    ):
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.vector_dim = vector_dim
        super().__init__(device=device)

    def build_network(self) -> nn.Module:
        return _GNNWrapper(
            hidden_dim=self.hidden_dim,
            num_gnn_layers=self.num_gnn_layers,
            vector_dim=self.vector_dim,
            device=self.device
        )

    def extract_features(self, system: Any, t: int):
        data: Data = system.get_graph_data().to(self.device)
        num_types = system.next_type_idx
        state_dim = len(fields(system.agents[0].state))  # number of fields in AgentState
        node_feats = data.x[:, num_types : num_types + state_dim]  # (N, state_dim)
        return node_feats, data.edge_index, data.batch, float(system.get_system_var("temperature")), float(t)

    def process_output(self, raw: torch.Tensor) -> torch.Tensor:
        # raw has shape (1,1) because _GNNWrapper applies Softplus at end
        return raw.view(-1)


class _GNNWrapper(nn.Module):
    """
    Internal helper for GNNThinker:
      - GraphSAGE layers → ReLU
      - global_mean_pool → (1, hidden_dim)
      - Linear(hidden_dim → vector_dim) → ReLU
      - Concatenate with [temp, t] → MLP → Softplus → (1,1)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_gnn_layers: int,
        vector_dim: int,
        device: torch.device
    ):
        super().__init__()
        self.device = device

        # Build GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(1, hidden_dim))
        for _ in range(num_gnn_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.global_pool = global_mean_pool

        # Project pooled embedding → vector_dim
        self.to_vector = nn.Linear(hidden_dim, vector_dim).to(device)

        # Final MLP: (vector_dim + 2) → hidden_dim → 1
        self.mlp = nn.Sequential(
            nn.Linear(vector_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        ).to(device)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        temperature: float,
        t: float
    ) -> torch.Tensor:
        """
        node_feats: (N, state_dim)
        edge_index: (2, E)
        batch:      (N,)
        temperature: float → will be converted to tensor inside
        t:           float → time step
        """
        h = node_feats.view(-1, 1)  # assume state_dim=1 (only 'age'); if more dims, adjust accordingly
        for conv in self.convs:
            h = conv(h, edge_index)
            h = nn.functional.relu(h)
        pooled = self.global_pool(h, batch)  # (1, hidden_dim)

        z = self.to_vector(pooled)  # (1, vector_dim)
        extras = torch.tensor([[temperature, t]], device=self.device)  # (1,2)
        feat = torch.cat([z, extras], dim=1)  # (1, vector_dim + 2)

        out = self.mlp(feat)  # (1,1), already Softplus
        return out  # (1,1)


# ----------------------------
# 6) Usage Example
# ----------------------------

if __name__ == "__main__":
    device = torch.device("cpu")

    # MLPThinker with custom hidden sizes:
    mlp_thinker = MLPThinker(hidden_sizes=[64, 64], device=device)
    print("MLPThinker:", mlp_thinker)

    # LinearThinker:
    lin_thinker = LinearThinker(device=device)
    print("LinearThinker:", lin_thinker)

    # LSTMThinker:
    lstm_thinker = LSTMThinker(hidden_size=32, device=device)
    print("LSTMThinker:", lstm_thinker)

    # GNNThinker:
    gnn_thinker = GNNThinker(hidden_dim=32, num_gnn_layers=2, vector_dim=5, device=device)
    print("GNNThinker:", gnn_thinker)

    # Example of resetting hidden state before simulation:
    lstm_thinker.reset()

    # In your Simulator, you can now call:
    # pred = thinker(system, t)  # returns a Tensor of shape (1,) for MLP/Linear/LSTM or (1,1) for GNN
