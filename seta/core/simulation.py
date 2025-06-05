import torch
from torch_geometric.utils import dense_to_sparse

from ..thinkers.decision_networks import (
    MLPDecisionNetwork,
    LinearDecisionNetwork,
    LSTMDecisionNetwork,
    GNNDecisionNetwork,
)

from ..agents.nodes import (
    WorkerAgent,
    SpawnerAgent
)

class Simulator:
    """
    Encapsulates one run of the agent‐based simulation for T steps at a fixed temperature.

    Attributes:
      - T             : total number of time steps
      - W_initial     : initial number of worker agents
      - S_initial     : initial number of spawner agents
      - decision_net  : a DecisionNetwork (MLP, Linear, LSTM, or GNN)
      - device        : torch.device
    """

    def __init__(
        self,
        T: int,
        W_initial: int,
        S_initial: int,
        decision_net: torch.nn.Module,
        device: torch.device,
    ):
        self.T = T
        self.W_initial = W_initial
        self.S_initial = S_initial
        self.decision_net = decision_net
        self.device = device

    def run(self, temperature: float) -> torch.Tensor:
        """
        Run a single simulation from t=0 … T-1 at fixed temperature.

        Returns:
          - Tensor of shape (T,) with predicted W‐counts at each time step.
        """

        temp_scalar = float(temperature)

        # 1) Initialize agent pools
        workers = [WorkerAgent(initial_age=0.0) for _ in range(self.W_initial)]
        spawners = [SpawnerAgent(initial_age=0.0) for _ in range(self.S_initial)]

        # If LSTMDecisionNetwork, clear its hidden state:
        if isinstance(self.decision_net, LSTMDecisionNetwork):
            self.decision_net.reset()

        preds = []

        for t in range(self.T):
            # — PHASE 1: SENSE — 
            current_temp = temp_scalar

            # — PHASE 2: EVOLVE — increment ages
            for w in workers:
                w.step_age()
            for s in spawners:
                s.step_age()

            # — PHASE 3: DECISION —
            if isinstance(self.decision_net, GNNDecisionNetwork):
                # Build a “chain graph” of (workers…spawner) in a line
                all_agents = workers + spawners    # list of Agent
                N = len(all_agents)
                ages_list = [agent.age for agent in all_agents]
                # ages_tensor: (N, 1)
                ages_tensor = torch.tensor(
                    ages_list, dtype=torch.float32, device=self.device
                ).unsqueeze(1)

                # Build “chain” adjacency (i ↔ i+1 for i=0..N-2)
                if N == 1:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                else:
                    row = []
                    col = []
                    for i_node in range(N - 1):
                        row += [i_node, i_node + 1]
                        col += [i_node + 1, i_node]
                    edge_index = torch.tensor([row, col], dtype=torch.long, device=self.device)

                batch = torch.zeros(N, dtype=torch.long, device=self.device)  # single‐graph batch

                pred_W_tensor = self.decision_net(
                    ages_tensor, edge_index, batch, current_temp, float(t)
                )
                pred_W_scalar = float(pred_W_tensor.item())

            else:
                # MLP / Linear / LSTM: input = [W_count, temp, t]
                W_count = len(workers)
                feature_tensor = torch.tensor(
                    [[W_count, current_temp, float(t)]],
                    dtype=torch.float32,
                    device=self.device,
                )  # shape (1, 3)

                pred_W_tensor = self.decision_net(feature_tensor)  # (1,)
                pred_W_scalar = float(pred_W_tensor.item())

            # — PHASE 4: REFLECT — spawn new workers if pred_W > current W_count
            W_count = len(workers)
            delta = pred_W_scalar - W_count
            if delta > 0.0:
                n_to_spawn = int(torch.ceil(torch.tensor(delta)).item())
                for _ in range(n_to_spawn):
                    workers.append(WorkerAgent(initial_age=0.0))

            preds.append(pred_W_tensor)

        return torch.cat(preds, dim=0)  # Tensor of shape (T,)

    def simulate_graphs(self, temperature: float):
        """
        Only meaningful if decision_net is GNNDecisionNetwork:
        Returns a list of (ages_tensor, edge_index) for each time step 0..T-1.
        Otherwise returns [].
        """
        if not isinstance(self.decision_net, GNNDecisionNetwork):
            return []

        temp_scalar = float(temperature)
        workers = [WorkerAgent(initial_age=0.0) for _ in range(self.W_initial)]
        spawners = [SpawnerAgent(initial_age=0.0) for _ in range(self.S_initial)]

        # Reset GNN hidden state (no‐op, but for uniformity)
        self.decision_net.reset()

        graphs = []

        for t in range(self.T):
            # Increment ages
            for w in workers:
                w.step_age()
            for s in spawners:
                s.step_age()

            all_agents = workers + spawners
            N = len(all_agents)
            ages_list = [agent.age for agent in all_agents]
            ages_tensor = torch.tensor(ages_list, dtype=torch.float32).unsqueeze(1)  # (N,1)

            if N == 1:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                row = []
                col = []
                for i_node in range(N - 1):
                    row += [i_node, i_node + 1]
                    col += [i_node + 1, i_node]
                edge_index = torch.tensor([row, col], dtype=torch.long)  # (2, 2*(N-1))

            graphs.append((ages_tensor.cpu(), edge_index.cpu()))

            # Keep evolving according to the GNN’s decision (but no plotting here)
            ages_tensor_device = ages_tensor.to(self.device)
            batch = torch.zeros(N, dtype=torch.long, device=self.device)
            pred_W_tensor = self.decision_net(
                ages_tensor_device, edge_index.to(self.device), batch, temp_scalar, float(t)
            )
            pred_W_scalar = float(pred_W_tensor.item())
            W_count = len(workers)
            delta = pred_W_scalar - W_count
            if delta > 0.0:
                n_to_spawn = int(torch.ceil(torch.tensor(delta)).item())
                for _ in range(n_to_spawn):
                    workers.append(WorkerAgent(initial_age=0.0))

        return graphs
