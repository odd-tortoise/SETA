import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#!/usr/bin/env python3
"""
Simulation + Training + Inference with Interchangeable Decision Networks,
including a GNN‐based variant and “full display” during inference.

Decision network options:
  • "mlp"    : feed‐forward MLP on [W_count, temperature, time_step]
  • "linear" : single linear layer on [W_count, temperature, time_step]
  • "lstm"   : LSTMCell on [W_count, temperature, time_step]
  • "gnn"    : GraphSAGE + pooling + MLP, using node ages in a chain graph

During inference, if `full_display=True`, the script:
  1) Plots a textual diagram of the chosen network architecture before simulation.
  2) Runs the simulation step‐by‐step, and every `display_interval` steps:
       • On the left: plots the current agent‐chain graph (node ages).
       • On the right: plots predicted W‐count so far vs. the ground‐truth logistic curve.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
import numpy as np

# PyTorch Geometric imports for GNN
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse

import networkx as nx  # for graph plotting


# ----------------------------
# 1. Agent Definitions
# ----------------------------

class Agent:
    """Base agent class tracking only 'age'."""
    def __init__(self, initial_age: float = 0.0):
        self.age = float(initial_age)

    def step_age(self):
        """Increment agent's age by 1.0 (one time step)."""
        self.age += 1.0


class WorkerAgent(Agent):
    """Worker agent (W)."""
    def __init__(self, initial_age: float = 0.0):
        super().__init__(initial_age)


class SpawnerAgent(Agent):
    """Spawner agent (S)."""
    def __init__(self, initial_age: float = 0.0):
        super().__init__(initial_age)


# ----------------------------
# 2. DecisionNetwork Variants
# ----------------------------

class MLPDecisionNetwork(nn.Module):
    """
    MLP-based decision network: input [W_count, temperature, time_step] → two hidden layers → output scalar
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
        returns: Tensor of shape (1,) after softplus for positivity
        """
        out = self.net(features)               # (1, 1)
        return nn.functional.softplus(out).view(-1)  # (1,)

    def reset(self):
        """No internal state to clear for MLP."""
        pass


class LinearDecisionNetwork(nn.Module):
    """
    Linear decision network: a single linear layer 3→1
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: Tensor of shape (1, 3)
        returns: Tensor of shape (1,) after softplus
        """
        out = self.linear(features)      # (1, 1)
        return nn.functional.softplus(out).view(-1)  # (1,)

    def reset(self):
        """No internal state to clear for linear."""
        pass


class LSTMDecisionNetwork(nn.Module):
    """
    LSTM-based decision network. At each time step, we feed the 3‐dim feature to an LSTMCell,
    keep hidden/cell states across time steps, then a linear layer maps hidden→1.
    """
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size=3, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.reset()

    def reset(self):
        """Zero out hidden and cell states; called at the start of each simulation."""
        self.hx = None  # Will be tensor (1, hidden_size)
        self.cx = None  # Will be tensor (1, hidden_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: Tensor of shape (1, 3)
        returns: Tensor of shape (1,) after softplus
        """
        if self.hx is None or self.cx is None:
            device = features.device
            self.hx = torch.zeros(1, self.hidden_size, device=device)
            self.cx = torch.zeros(1, self.hidden_size, device=device)

        self.hx, self.cx = self.lstm_cell(features, (self.hx, self.cx))
        out = self.linear(self.hx)  # (1, 1)
        return nn.functional.softplus(out).view(-1)  # (1,)


class GNNDecisionNetwork(nn.Module):
    """
    GNN‐based decision network: uses GraphSAGE on a chain graph of agents (by age).
    Node features = [age_i]; after GNN layers, do global_mean_pool → vector_dim=5 → concat [z, temp, t] → MLP→ scalar.
    """
    def __init__(self, node_in_dim: int = 1, hidden_dim: int = 32, num_gnn_layers: int = 2, vector_dim: int = 5):
        super().__init__()
        # Build GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(node_in_dim, hidden_dim))
        for _ in range(num_gnn_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Project pooled embedding to a vector_dim-sized vector
        self.to_vector = nn.Linear(hidden_dim, vector_dim)

        # Final MLP: (vector_dim + 2) → hidden_dim → 1
        self.mlp = nn.Sequential(
            nn.Linear(vector_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # ensure positive output
        )

    def forward(self, ages: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, temp: float, t: float) -> torch.Tensor:
        """
        ages      : Tensor of shape (N, 1) = node ages
        edge_index: LongTensor of shape (2, E)
        batch     : LongTensor of shape (N,)   (all zeros for single graph)
        temp, t   : floats or 0-dim tensors

        returns: Tensor of shape (1,) = predicted W_count Ŵ_t
        """
        h = ages  # (N, 1)
        for conv in self.convs:
            h = conv(h, edge_index)
            h = nn.functional.relu(h)  # (N, hidden_dim)

        # Global mean‐pool to (1, hidden_dim)
        c = global_mean_pool(h, batch)  # (1, hidden_dim)

        # Map to vector_dim‐vector
        z = self.to_vector(c)  # (1, vector_dim)

        extras = torch.tensor([[float(temp), float(t)]], device=ages.device)  # (1, 2)
        feat = torch.cat([z, extras], dim=1)  # (1, vector_dim + 2)

        out = self.mlp(feat)  # (1, 1)
        return out.view(-1)   # (1,)
    
    def reset(self):
        """No hidden state to clear, but define so .reset() exists."""
        pass


# ----------------------------
# 3. Simulator Logic (Not an nn.Module)
# ----------------------------

class Simulator:
    """
    Encapsulates the agent-based simulation. Holds:
      - T: number of time steps per simulation
      - W_initial: initial number of WorkerAgents
      - S_initial: initial number of SpawnerAgents
      - decision_net: an instance of one of the DecisionNetwork variants
      - device: torch.device for running the decision_net
    """

    def __init__(
        self,
        T: int,
        W_initial: int,
        S_initial: int,
        decision_net: nn.Module,
        device: torch.device,
    ):
        self.T = T
        self.W_initial = W_initial
        self.S_initial = S_initial
        self.decision_net = decision_net
        self.device = device

    def run(self, temperature: float) -> torch.Tensor:
        """
        Run a single simulation from t=0 to t=T-1 at a fixed temperature.
        Depending on decision_net type, either uses simple [W_count, temp, t] features
        or builds a chain-graph of node ages for the GNN.
        Returns a Tensor of shape (T,) with predicted W counts.
        """
        temp_scalar = float(temperature)

        # Initialize agents
        workers = [WorkerAgent(initial_age=0.0) for _ in range(self.W_initial)]
        spawners = [SpawnerAgent(initial_age=0.0) for _ in range(self.S_initial)]

        # If LSTM variant, reset internal states
        if isinstance(self.decision_net, LSTMDecisionNetwork):
            self.decision_net.reset()

        predicted_W_counts = []

        for t in range(self.T):
            # PHASE 1: SENSE
            current_temp = temp_scalar

            # PHASE 2: EVOLVE (increment ages)
            for w in workers:
                w.step_age()
            for s in spawners:
                s.step_age()

            # PHASE 3: DECISION
            if isinstance(self.decision_net, GNNDecisionNetwork):
                # Build chain graph: all workers first, then spawner as last node
                all_agents = workers + spawners  # a Python list
                N = len(all_agents)
                ages_list = [agent.age for agent in all_agents]
                ages_tensor = torch.tensor(ages_list, dtype=torch.float32, device=self.device).unsqueeze(1)  # (N, 1)

                # Build edges for a chain: i -- (i+1) for i in [0..N-2]
                if N == 1:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                else:
                    row = []
                    col = []
                    for i_node in range(N - 1):
                        row += [i_node, i_node + 1]
                        col += [i_node + 1, i_node]
                    edge_index = torch.tensor([row, col], dtype=torch.long, device=self.device)  # (2, 2*(N-1))

                batch = torch.zeros(N, dtype=torch.long, device=self.device)  # single graph
                pred_W_tensor = self.decision_net(ages_tensor, edge_index, batch, current_temp, float(t))
                pred_W_scalar = float(pred_W_tensor.item())

            else:
                # MLP, Linear, or LSTM on [W_count, temp, t]
                current_W_count = len(workers)
                t_scalar = float(t)
                feature_tensor = torch.tensor(
                    [[current_W_count, current_temp, t_scalar]],
                    dtype=torch.float32,
                    device=self.device,
                )  # (1, 3)
                pred_W_tensor = self.decision_net(feature_tensor)  # (1,)
                pred_W_scalar = float(pred_W_tensor.item())

            # PHASE 4: REFLECT (spawn if needed)
            W_count = len(workers)
            delta = pred_W_scalar - W_count
            if delta > 0.0:
                n_to_spawn = int(torch.ceil(torch.tensor(delta)).item())
                for _ in range(n_to_spawn):
                    workers.append(WorkerAgent(initial_age=0.0))

            predicted_W_counts.append(pred_W_tensor)

        return torch.cat(predicted_W_counts, dim=0)  # (T,)

    def simulate_graphs(self, temperature: float):
        """
        If decision_net is GNNDecisionNetwork, run and return a list of (ages_tensor, edge_index)
        for each time step; else return an empty list.
        """
        if not isinstance(self.decision_net, GNNDecisionNetwork):
            return []

        temp_scalar = float(temperature)
        workers = [WorkerAgent(initial_age=0.0) for _ in range(self.W_initial)]
        spawners = [SpawnerAgent(initial_age=0.0) for _ in range(self.S_initial)]
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
            ages_tensor = torch.tensor(ages_list, dtype=torch.float32).unsqueeze(1)  # (N, 1)

            # Build chain edges
            if N == 1:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                row = []
                col = []
                for i_node in range(N - 1):
                    row += [i_node, i_node + 1]
                    col += [i_node + 1, i_node]
                edge_index = torch.tensor([row, col], dtype=torch.long)  # (2, 2*(N-1))

            graphs.append((ages_tensor, edge_index))

            # Decision to spawn
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


# ----------------------------
# 4. CustomFunctionDataset
# ----------------------------

class CustomFunctionDataset(Dataset):
    """
    A Dataset generating (temperature, curve) pairs by calling a user-provided function:
        curve_fn(time_tensor, temperature) → Tensor of shape (T,)
    """

    def __init__(
        self,
        T: int,
        num_examples: int,
        temp_min: float,
        temp_max: float,
        curve_fn,
    ):
        super().__init__()
        self.T = T
        self.num_examples = num_examples
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.curve_fn = curve_fn

        # Random temperatures in [temp_min, temp_max]
        self.temperatures = temp_min + (temp_max - temp_min) * torch.rand(num_examples)

        # Precompute time grid
        self.t = torch.arange(0, T, dtype=torch.float32)  # (T,)

        # Precompute all curves
        self.curves = []
        for i in range(num_examples):
            temp = float(self.temperatures[i].item())
            curve = self.curve_fn(self.t, temp)  # expect Tensor (T,)
            if not (isinstance(curve, torch.Tensor) and curve.shape == (T,)):
                raise ValueError(f"curve_fn must return a torch.Tensor of shape ({T},). Got {type(curve)} {curve.shape}.")
            self.curves.append(curve)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        temp = float(self.temperatures[idx].item())
        curve = self.curves[idx]  # (T,)
        return temp, curve


# ----------------------------
# 5. Training Loop & Visualization with Early Stopping
# ----------------------------

def train_and_visualize(
    decision_net: nn.Module,
    simulator: Simulator,
    dataset: CustomFunctionDataset,
    device: torch.device,
    num_epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    output_model_path: str = "decision_net.pth",
    visualize: bool = True,
    curve_interval: int = 5,
    num_example_curves: int = 4,
    validation_split: float = 0.2,
    patience: int = 5
):
    """
    Train the decision network using the simulator to produce predicted curves,
    but obtain ground-truth from the custom function dataset. Implements early stopping.
    """

    total_size = len(dataset)
    val_size = int(validation_split * total_size)
    train_size = total_size - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(decision_net.parameters(), lr=lr)

    decision_net.to(device)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_weights = None
    no_improve_count = 0

    # Pick fixed examples from validation set for plotting
    if num_example_curves > len(val_set):
        raise ValueError("num_example_curves > validation set size")
    example_indices = list(range(num_example_curves))
    example_temps = []
    example_true_curves = []
    for idx in example_indices:
        temp_i, curve_i = val_set[idx]
        example_temps.append(temp_i)
        example_true_curves.append(curve_i.numpy())

    print("Starting training with early stopping (patience={} epochs) ...".format(patience))
    for epoch in range(num_epochs):
        decision_net.train()
        running_train_loss = 0.0

        for temps_batch, true_curves in train_loader:
            temps_batch = temps_batch.to(device)
            true_curves = true_curves.to(device)

            optimizer.zero_grad()

            batch_preds = []
            for i in range(temps_batch.shape[0]):
                temp_i = temps_batch[i].item()
                pred_curve_i = simulator.run(temp_i)
                batch_preds.append(pred_curve_i.unsqueeze(0))  # (1, T)

            batch_pred = torch.cat(batch_preds, dim=0)  # (batch_size, T)
            loss = criterion(batch_pred, true_curves)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        decision_net.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for temps_v, true_curves_v in val_loader:
                temps_v = temps_v.to(device)
                true_curves_v = true_curves_v.to(device)

                batch_preds_v = []
                for i in range(temps_v.shape[0]):
                    temp_iv = temps_v[i].item()
                    pred_curve_iv = simulator.run(temp_iv)
                    batch_preds_v.append(pred_curve_iv.unsqueeze(0))
                batch_pred_v = torch.cat(batch_preds_v, dim=0)
                loss_v = criterion(batch_pred_v, true_curves_v)
                running_val_loss += loss_v.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}  "
              f"Train Loss: {epoch_train_loss:.6f}  "
              f"Val Loss: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in decision_net.state_dict().items()}
            no_improve_count = 0
            best_epoch = epoch + 1
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}. No improvement for {patience} consecutive epochs.")
            break

        # Plot example curves every curve_interval epochs (or final epoch)
        if visualize and ((epoch + 1) % curve_interval == 0 or (epoch + 1) == num_epochs):
            decision_net.eval()
            with torch.no_grad():
                plt.figure(figsize=(8, 6))
                for i in range(num_example_curves):
                    temp_i = example_temps[i]
                    true_curve_i = example_true_curves[i]
                    pred_curve_i = simulator.run(temp_i).cpu().numpy()

                    t_axis = np.arange(simulator.T)
                    plt.plot(
                        t_axis,
                        true_curve_i,
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.6,
                        label=f"True #{i+1} (T={temp_i:.1f}°C)"
                    )
                    plt.plot(
                        t_axis,
                        pred_curve_i,
                        linestyle="-",
                        linewidth=2.0,
                        alpha=0.8,
                        label=f"Pred #{i+1} (T={temp_i:.1f}°C)"
                    )

                plt.title(f"Epoch {epoch+1}: Example Curves")
                plt.xlabel("Time Step")
                plt.ylabel("W Count")
                plt.legend(ncol=2, fontsize="small", loc="upper left")
                plt.grid(True)
                plt.tight_layout()

                fname = f"epoch_{epoch+1:03d}_curves.png"
                plt.savefig(fname)
                plt.show()

    # Load best weights
    if best_weights is not None:
        decision_net.load_state_dict(best_weights)
        print(f"Loaded best model from epoch {best_epoch} with Val Loss = {best_val_loss:.6f}")

    # Save the best model
    torch.save(decision_net.state_dict(), output_model_path)
    print(f"Best model saved to {output_model_path}")

    # Plot training & validation loss vs. epoch
    if visualize:
        epochs_range = np.arange(1, len(train_losses) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, train_losses, label="Train Loss", marker='o')
        plt.plot(epochs_range, val_losses,   label="Val Loss",   marker='x')
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("losses.png")
        plt.show()

        # Final single‐example plot
        temp_ex = example_temps[0]
        true_ex = example_true_curves[0]
        decision_net.eval()
        with torch.no_grad():
            pred_ex = simulator.run(temp_ex).cpu().numpy()

        t_axis = np.arange(simulator.T)
        plt.figure(figsize=(8, 5))
        plt.plot(t_axis, true_ex, label="Ground Truth", linewidth=2)
        plt.plot(t_axis, pred_ex, label="Predicted (Best Model)", linestyle="--", linewidth=2)
        plt.title(f"Best Model at Epoch {best_epoch} (T={temp_ex:.2f}°C)")
        plt.xlabel("Time Step")
        plt.ylabel("W Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("example_prediction_best.png")
        plt.show()

    return decision_net


# ----------------------------
# 6. Inference Function with Full Display
# ----------------------------

def plot_network_diagram(model: nn.Module):
    """
    Simple textual “diagram” of the model using matplotlib.
    """
    description = repr(model)
    plt.figure(figsize=(6, 6))
    plt.text(0.01, 0.99, description, family='monospace', fontsize=8, va='top')
    plt.axis('off')
    plt.title("Model Architecture")
    plt.tight_layout()
    plt.show()


def plot_chain_graph(ages_tensor: torch.Tensor, edge_index: torch.Tensor, ax, title=""):
    """
    Given ages (1D tensor) and edge_index (2×E tensor), plot the chain graph on ax.
    Node label = age, placed on a horizontal line.
    """
    ages = ages_tensor.numpy().flatten()
    N = len(ages)
    G = nx.Graph()
    for i in range(N):
        G.add_node(i, age=float(ages[i]))
    edges = edge_index.numpy().T
    G.add_edges_from(edges.tolist())

    pos = {i: (i, 0) for i in range(N)}
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=ages.min(), vmax=ages.max())
    node_colors = [cmap(norm(ages[i])) for i in range(N)]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)
    labels = {i: f"{int(ages[i])}" for i in range(N)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
    ax.set_title(title)
    ax.set_axis_off()


def run_inference(
    model_path: str,
    temperature: float,
    T: int,
    decision_type: str,
    hidden_sizes: list,
    lstm_hidden_size: int,
    W_initial: int,
    S_initial: int,
    device: torch.device,
    visualize: bool = True,
    full_display: bool = False,
    display_interval: int = 5  # plot every `display_interval` steps
):
    """
    Load a trained decision network from model_path, run one simulation at the given temperature,
    and optionally do “full display”:
      1) Plot network diagram.
      2) During simulation, every display_interval steps, plot [left: chain graph, right: predicted vs. true W_count].
    """

    if not os.path.exists(model_path):
        print(f"Error: model_path '{model_path}' does not exist.")
        sys.exit(1)

    # Instantiate the correct DecisionNetwork
    if decision_type == "mlp":
        decision_net = MLPDecisionNetwork(hidden_sizes=hidden_sizes).to(device)
    elif decision_type == "linear":
        decision_net = LinearDecisionNetwork().to(device)
    elif decision_type == "lstm":
        decision_net = LSTMDecisionNetwork(hidden_size=lstm_hidden_size).to(device)
    elif decision_type == "gnn":
        decision_net = GNNDecisionNetwork().to(device)
    else:
        raise ValueError(f"Unknown decision_type '{decision_type}'. Choose 'mlp','linear','lstm','gnn'.")

    decision_net.load_state_dict(torch.load(model_path, map_location=device))
    simulator = Simulator(
        T=T,
        W_initial=W_initial,
        S_initial=S_initial,
        decision_net=decision_net,
        device=device
    )
    decision_net.eval()

    # If full_display, plot the model diagram
    if full_display:
        plot_network_diagram(decision_net)

        # Precompute ground-truth logistic curve for this temperature
        t_grid = torch.arange(0, T, dtype=torch.float32)
        def logistic_curve(tensor, temp):
            T_max = 30.0
            K_max = 100.0
            r_max = 0.2
            alpha = temp / T_max
            K = K_max * alpha
            r = r_max * alpha
            exp_term = torch.exp(-r * tensor)
            return K / (1.0 + (K - 1.0) * exp_term)

        true_curve = logistic_curve(t_grid, temperature).numpy()

        # If GNN, we need to get chain-graphs at each time step
        if decision_type == "gnn":
            graphs = simulator.simulate_graphs(temperature)
        else:
            graphs = []

        # Now step through simulation manually, plotting every display_interval steps
        # We'll replicate Simulator.run with plotting hooks
        workers = [WorkerAgent(initial_age=0.0) for _ in range(W_initial)]
        spawners = [SpawnerAgent(initial_age=0.0) for _ in range(S_initial)]
        if isinstance(decision_net, LSTMDecisionNetwork):
            decision_net.reset()

        preds = []

        for t in range(T):
            # Evolve ages
            for w in workers:
                w.step_age()
            for s in spawners:
                s.step_age()

            # Decision
            if decision_type == "gnn":
                all_agents = workers + spawners
                N = len(all_agents)
                ages_list = [agent.age for agent in all_agents]
                ages_tensor = torch.tensor(ages_list, dtype=torch.float32, device=device).unsqueeze(1)
                if N == 1:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                else:
                    row = []
                    col = []
                    for i_node in range(N - 1):
                        row += [i_node, i_node + 1]
                        col += [i_node + 1, i_node]
                    edge_index = torch.tensor([row, col], dtype=torch.long, device=device)
                batch = torch.zeros(N, dtype=torch.long, device=device)
                pred_W_tensor = decision_net(ages_tensor, edge_index, batch, temperature, float(t))
                pred_W = float(pred_W_tensor.item())

            else:
                W_count = len(workers)
                feature_tensor = torch.tensor(
                    [[W_count, float(temperature), float(t)]],
                    dtype=torch.float32,
                    device=device
                )
                pred_W_tensor = decision_net(feature_tensor)
                pred_W = float(pred_W_tensor.item())

            preds.append(pred_W_tensor)

            # Plot every display_interval steps
            if full_display and (t % display_interval == 0 or t == T - 1):
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                # Left: graph (if GNN) or simple placeholder
                if decision_type == "gnn":
                    ages_cpu, edge_idx_cpu = graphs[t]
                    plot_chain_graph(ages_cpu, edge_idx_cpu, axes[0], title=f"t={t}: Chain Graph")
                else:
                    axes[0].text(0.5, 0.5, "No graph for this model", ha='center', va='center')
                    axes[0].axis('off')

                # Right: predicted vs. true W-count up to t
                pred_np = [float(p.item()) for p in preds]
                axes[1].plot(range(t + 1), true_curve[: t + 1], linestyle="--", label="True")
                axes[1].plot(range(t + 1), pred_np, linestyle="-", label="Pred")
                axes[1].set_xlabel("Time Step")
                axes[1].set_ylabel("W Count")
                axes[1].set_title(f"t={t}: Pred vs True")
                axes[1].legend()
                axes[1].grid(True)

                plt.suptitle(f"Simulation at t={t}")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()

            # Reflect (spawn)
            W_count = len(workers)
            delta = pred_W - W_count
            if delta > 0.0:
                n_to_spawn = int(torch.ceil(torch.tensor(delta)).item())
                for _ in range(n_to_spawn):
                    workers.append(WorkerAgent(initial_age=0.0))

        result_curve = torch.cat(preds).detach().cpu().numpy()


    else:
        # Simple inference without display
        with torch.no_grad():
            result_curve = simulator.run(temperature).cpu().numpy()

    if visualize and not full_display:
        t_axis = np.arange(T)
        plt.figure(figsize=(8, 5))
        plt.plot(t_axis, result_curve, label="Predicted W Count", linestyle="-", linewidth=2)
        plt.title(f"Inference: Predicted W Curve at Temperature={temperature:.2f}°C")
        plt.xlabel("Time Step")
        plt.ylabel("W Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("inference_prediction.png")
        plt.show()

    return result_curve


# ----------------------------
# 7. Main (with hardcoded arguments & custom curve_fn)
# ----------------------------

def main():
    # ------------------------
    # CONFIGURE HERE:
    # ------------------------

    # Choose mode: "train" or "infer"
    #mode = "train"
    mode = "infer"

    # Choose decision network type: "mlp", "linear", "lstm", or "gnn"
    # decision_type = "mlp"
    # decision_type = "linear"
    # decision_type = "lstm"
    decision_type = "gnn"

    # Common parameters:
    device_str = "cpu"  # or "cuda"
    device = torch.device(device_str)

    # ------------------------
    # CUSTOM CURVE FUNCTION:
    # ------------------------
    def logistic_curve_fn(time_tensor: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Example: logistic growth
          W_true(t) = K(T) / [1 + (K(T)-1)*exp(-r(T)*t)]
        """
        T_max = 30.0
        K_max = 100.0
        r_max = 0.2
        alpha = temperature / T_max
        K = K_max * alpha
        r = r_max * alpha
        exp_term = torch.exp(-r * time_tensor)            # (T,)
        W_t = K / (1.0 + (K - 1.0) * exp_term)            # (T,)
        return W_t

    # ------------------------
    # TRAINING PARAMETERS:
    # ------------------------
    T_train = 50
    epochs = 100
    batch_size = 8
    learning_rate = 1e-2
    num_examples = 100
    temp_min = 15.0
    temp_max = 30.0
    output_model_path = "decision_net.pth"
    hidden_sizes_train = [64, 64]  # for MLP
    lstm_hidden_size = 32          # for LSTM
    W_initial_train = 1
    S_initial_train = 1

    validation_split = 0.2
    patience = 10

    curve_interval = 20
    num_example_curves = 4

    # ------------------------
    # INFERENCE PARAMETERS:
    # ------------------------
    T_infer = 100
    temperature_infer = 22.5
    model_path_infer = "decision_net.pth"
    hidden_sizes_infer = [64, 64]
    lstm_hidden_size_infer = 32
    W_initial_infer = 1
    S_initial_infer = 1

    # For full display:
    full_display = True
    display_interval = 10  # plot every 10 steps

    # ------------------------
    # RUNNING:
    # ------------------------
    if mode == "train":
        print(f"=== Mode: TRAIN ({decision_type.upper()} decision network) ===")

        # 1) Create dataset
        dataset = CustomFunctionDataset(
            T=T_train,
            num_examples=num_examples,
            temp_min=temp_min,
            temp_max=temp_max,
            curve_fn=logistic_curve_fn
        )

        # 2) Instantiate decision network
        if decision_type == "mlp":
            decision_net = MLPDecisionNetwork(hidden_sizes=hidden_sizes_train)
        elif decision_type == "linear":
            decision_net = LinearDecisionNetwork()
        elif decision_type == "lstm":
            decision_net = LSTMDecisionNetwork(hidden_size=lstm_hidden_size)
        elif decision_type == "gnn":
            decision_net = GNNDecisionNetwork()
        else:
            raise ValueError(f"Unknown decision_type '{decision_type}'")

        # 3) Instantiate simulator
        simulator = Simulator(
            T=T_train,
            W_initial=W_initial_train,
            S_initial=S_initial_train,
            decision_net=decision_net,
            device=device
        )

        # 4) Train & visualize
        _ = train_and_visualize(
            decision_net=decision_net,
            simulator=simulator,
            dataset=dataset,
            device=device,
            num_epochs=epochs,
            batch_size=batch_size,
            lr=learning_rate,
            output_model_path=output_model_path,
            visualize=True,
            curve_interval=curve_interval,
            num_example_curves=num_example_curves,
            validation_split=validation_split,
            patience=patience
        )

    elif mode == "infer":
        print(f"=== Mode: INFER ({decision_type.upper()} decision network) ===")
        pred_curve = run_inference(
            model_path=model_path_infer,
            temperature=temperature_infer,
            T=T_infer,
            decision_type=decision_type,
            hidden_sizes=hidden_sizes_infer,
            lstm_hidden_size=lstm_hidden_size_infer,
            W_initial=W_initial_infer,
            S_initial=S_initial_infer,
            device=device,
            visualize=True,
            full_display=full_display,
            display_interval=display_interval
        )
        np.save("inference_pred_curve.npy", pred_curve)
        print("Predicted curve saved as 'inference_pred_curve.npy'")

    else:
        print("Unknown mode! Set mode = 'train' or 'infer'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
