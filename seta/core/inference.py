import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from .simulation import Simulator, WorkerAgent, SpawnerAgent
from ..thinkers.decision_networks import (
    MLPDecisionNetwork,
    LinearDecisionNetwork,
    LSTMDecisionNetwork,
    GNNDecisionNetwork,
)

def plot_network_diagram(model: torch.nn.Module):
    """
    Render a simple textual “repr(model)” as a plot.
    """
    desc = repr(model)
    plt.figure(figsize=(6, 6))
    plt.text(0.01, 0.99, desc, family="monospace", fontsize=8, va="top")
    plt.axis("off")
    plt.title("Model Architecture")
    plt.tight_layout()
    plt.show()

def plot_chain_graph(ages_tensor: torch.Tensor, edge_index: torch.Tensor, ax, title=""):
    """
    Given ages_tensor (shape N×1) and edge_index (2×E), draw a chain graph whose
    node‐colors (and node labels) reflect each node’s age.
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
    display_interval: int = 5
):
    """
    Load a saved DecisionNetwork from model_path, run one T‐step simulation at ‘temperature’,
    and optionally show “full display”—plotting the network architecture, then every
    display_interval steps show the chain graph + current pred vs. true curve.
    """

    if not os.path.exists(model_path):
        print(f"Error: '{model_path}' not found.")
        sys.exit(1)

    # 1) Instantiate the correct DecisionNetwork
    if decision_type == "mlp":
        decision_net = MLPDecisionNetwork(hidden_sizes).to(device)
    elif decision_type == "linear":
        decision_net = LinearDecisionNetwork().to(device)
    elif decision_type == "lstm":
        decision_net = LSTMDecisionNetwork(hidden_size=lstm_hidden_size).to(device)
    elif decision_type == "gnn":
        decision_net = GNNDecisionNetwork().to(device)
    else:
        raise ValueError("decision_type must be in {'mlp','linear','lstm','gnn'}")

    # 2) Load saved weights
    decision_net.load_state_dict(torch.load(model_path, map_location=device))
    simulator = Simulator(
        T=T,
        W_initial=W_initial,
        S_initial=S_initial,
        decision_net=decision_net,
        device=device
    )
    decision_net.eval()

    # Precompute ground‐truth logistic curve for plotting
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

    if full_display:
        # — Plot model architecture —
        plot_network_diagram(decision_net)

        # If GNN, get chain‐graphs at every time step
        if decision_type == "gnn":
            graphs = simulator.simulate_graphs(temperature)
        else:
            graphs = []

        # Now step through “by hand,” plotting at intervals
        workers = [WorkerAgent(initial_age=0.0) for _ in range(W_initial)]
        spawners = [SpawnerAgent(initial_age=0.0) for _ in range(S_initial)]
        if isinstance(decision_net, LSTMDecisionNetwork):
            decision_net.reset()

        preds = []

        for t in range(T):
            # increment ages
            for w in workers:
                w.step_age()
            for s in spawners:
                s.step_age()

            # make decision
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
                pred_W_tensor = decision_net(ages_tensor, edge_index, batch, float(temperature), float(t))
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

            # Plot every display_interval (or at final step)
            if (t % display_interval == 0) or (t == T - 1):
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # Left: chain graph (if GNN)
                if decision_type == "gnn":
                    ages_cuda, edge_idx_cuda = graphs[t]
                    plot_chain_graph(ages_cuda, edge_idx_cuda, axes[0], title=f"t={t}: Chain Graph")
                else:
                    axes[0].text(0.5, 0.5, "No graph available", ha="center", va="center")
                    axes[0].axis("off")

                # Right: pred vs. true curve up to t
                pred_vals = [float(p.item()) for p in preds]
                axes[1].plot(range(t + 1), true_curve[: t + 1], linestyle="--", label="True")
                axes[1].plot(range(t + 1), pred_vals, linestyle="-", label="Pred")
                axes[1].set_xlabel("Time Step")
                axes[1].set_ylabel("W Count")
                axes[1].set_title(f"t={t}: Pred vs True")
                axes[1].legend()
                axes[1].grid(True)

                plt.suptitle(f"Simulation at t={t}")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()

                # spawn new workers
            W_count = len(workers)
            delta = pred_W - W_count
            if delta > 0.0:
                n_to_spawn = int(torch.ceil(torch.tensor(delta)).item())
                for _ in range(n_to_spawn):
                    workers.append(WorkerAgent(initial_age=0.0))

        # Detach and convert to NumPy
        result_curve = torch.cat(preds).detach().cpu().numpy()

    else:
        # Simple “silent” inference
        with torch.no_grad():
            result_curve = simulator.run(temperature).cpu().numpy()

    if visualize and not full_display:
        t_axis = np.arange(T)
        plt.figure(figsize=(8, 5))
        plt.plot(t_axis, result_curve, label="Predicted W Count", linewidth=2)
        plt.title(f"Inference: Predicted W Curve at T={temperature:.2f}°C")
        plt.xlabel("Time Step")
        plt.ylabel("W Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("inference_prediction.png")
        plt.show()

    return result_curve
