
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#!/usr/bin/env python3
"""
Simulation + Training + Inference with Interchangeable Decision Networks

This script allows swapping between three types of decision networks:
  1. “mlp”: a multi‐layer perceptron (MLP)
  2. “linear”: a single linear layer
  3. “lstm”: an LSTMCell that carries hidden state across time steps

Everything else—Simulator, Dataset, training loop, inference—remains unchanged.
Choose the decision network by setting `decision_type` in main() to "mlp", "linear", or "lstm".
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
import numpy as np


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

    Usage:
      - Call reset() before each simulation to zero hidden/cell state.
      - Then, for t in 0..T-1, call forward(features) once per step.
    """
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        # LSTMCell takes input_size=3, hidden_size=hidden_size
        self.lstm_cell = nn.LSTMCell(input_size=3, hidden_size=hidden_size)
        # Map hidden state to scalar
        self.linear = nn.Linear(hidden_size, 1)
        self.reset()

    def reset(self):
        """Zero out hidden and cell states; called at the start of each simulation."""
        self.hx = None  # will become shape (1, hidden_size)
        self.cx = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: Tensor of shape (1, 3)
        returns: Tensor of shape (1,) after softplus
        Internally updates hidden/cell state.
        """
        # Initialize hx, cx on first call
        if self.hx is None or self.cx is None:
            self.hx = torch.zeros(1, self.hidden_size, device=features.device)
            self.cx = torch.zeros(1, self.hidden_size, device=features.device)
        # features: (1, 3)
        self.hx, self.cx = self.lstm_cell(features, (self.hx, self.cx))
        # hx: (1, hidden_size)
        out = self.linear(self.hx)            # (1, 1)
        return nn.functional.softplus(out).view(-1)  # (1,)


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
        Run a simulation from t=0 to t=T-1 at a fixed temperature.
        Returns a Tensor of shape (T,) with predicted W counts.
        """
        temp_scalar = float(temperature)

        # Initialize agents
        workers = [WorkerAgent(initial_age=0.0) for _ in range(self.W_initial)]
        spawners = [SpawnerAgent(initial_age=0.0) for _ in range(self.S_initial)]

        # Reset any internal state in decision_net (for LSTM variant)
        if hasattr(self.decision_net, "reset"):
            self.decision_net.reset()

        predicted_W_counts = []

        for t in range(self.T):
            # PHASE 1: SENSE
            current_temp = temp_scalar

            # PHASE 2: EVOLVE
            for w in workers:
                w.step_age()
            for s in spawners:
                s.step_age()

            current_W_count = len(workers)
            t_scalar = float(t)

            # PHASE 3: DECISION
            feature_tensor = torch.tensor(
                [[current_W_count, current_temp, t_scalar]],
                dtype=torch.float32,
                device=self.device,
            )  # shape = (1, 3)
            pred_W_tensor = self.decision_net(feature_tensor)  # shape = (1,)
            pred_W_scalar = float(pred_W_tensor.item())

            # PHASE 4: REFLECT
            delta = pred_W_scalar - current_W_count
            if delta > 0.0:
                n_to_spawn = int(torch.ceil(torch.tensor(delta)).item())
                for _ in range(n_to_spawn):
                    workers.append(WorkerAgent(initial_age=0.0))
            # (No removals if pred_W < current_W)

            predicted_W_counts.append(pred_W_tensor)

        return torch.cat(predicted_W_counts, dim=0)  # shape = (T,)


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
        """
        Args:
            T            : sequence length (number of time steps)
            num_examples : how many examples to generate
            temp_min     : minimum temperature
            temp_max     : maximum temperature
            curve_fn     : function f(time_tensor, temperature) → Tensor(T,)
        """
        super().__init__()
        self.T = T
        self.num_examples = num_examples
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.curve_fn = curve_fn

        # Random temperatures in [temp_min, temp_max]
        self.temperatures = temp_min + (temp_max - temp_min) * torch.rand(num_examples)

        # Precompute time grid
        self.t = torch.arange(0, T, dtype=torch.float32)  # shape = (T,)

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
    Train the decision network using the simulator to generate predictions and
    the dataset’s curves as ground-truth. Implements early stopping.

    Splits dataset into train/val, trains decision_net parameters, and tracks best model.
    """

    # 1) Split dataset into train/val
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

    # Choose fixed examples from validation for plotting
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
            temps_batch = temps_batch.to(device)      # (batch_size,)
            true_curves = true_curves.to(device)      # (batch_size, T)

            optimizer.zero_grad()

            batch_preds = []
            for i in range(temps_batch.shape[0]):
                temp_i = temps_batch[i].item()
                pred_curve_i = simulator.run(temp_i)  # (T,)
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

        # Check for improvement
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in decision_net.state_dict().items()}
            no_improve_count = 0
            best_epoch = epoch + 1
        else:
            no_improve_count += 1

        # Early stopping
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

    # Load best weights into the decision network
    if best_weights is not None:
        decision_net.load_state_dict(best_weights)
        print(f"Loaded best model from epoch {best_epoch} with Val Loss = {best_val_loss:.6f}")

    # Save the best decision network
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

        # Final single‐example plot using best model (first example)
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
# 6. Inference Function
# ----------------------------

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
):
    """
    Load a trained decision network from model_path, run one simulation at the given temperature,
    and (optionally) plot the resulting W‐count curve over time.
    """

    if not os.path.exists(model_path):
        print(f"Error: model_path '{model_path}' does not exist.")
        sys.exit(1)

    # Instantiate the correct DecisionNetwork and Simulator
    if decision_type == "mlp":
        decision_net = MLPDecisionNetwork(hidden_sizes=hidden_sizes).to(device)
    elif decision_type == "linear":
        decision_net = LinearDecisionNetwork().to(device)
    elif decision_type == "lstm":
        decision_net = LSTMDecisionNetwork(hidden_size=lstm_hidden_size).to(device)
    else:
        raise ValueError(f"Unknown decision_type '{decision_type}'. Choose 'mlp','linear', or 'lstm'.")

    decision_net.load_state_dict(torch.load(model_path, map_location=device))
    simulator = Simulator(
        T=T,
        W_initial=W_initial,
        S_initial=S_initial,
        decision_net=decision_net,
        device=device
    )
    decision_net.eval()

    with torch.no_grad():
        pred_curve = simulator.run(temperature).cpu().numpy()  # shape = (T,)

    if visualize:
        t_axis = np.arange(T)
        plt.figure(figsize=(8, 5))
        plt.plot(t_axis, pred_curve, label="Predicted W Count", linestyle="-", linewidth=2)
        plt.title(f"Inference: Predicted W Curve at Temperature={temperature:.2f}°C")
        plt.xlabel("Time Step")
        plt.ylabel("W Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("inference_prediction.png")
        plt.show()

    return pred_curve


# ----------------------------
# 7. Main (with hardcoded arguments & custom curve_fn)
# ----------------------------

def main():
    # ------------------------
    # CONFIGURE HERE:
    # ------------------------

    # Choose mode: "train" or "infer"
    mode = "train"
    # mode = "infer"

    # Choose decision network type: "mlp", "linear", or "lstm"
    #decision_type = "mlp"
    #decision_type = "linear"
    decision_type = "lstm"

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
    # Used if mode == "train"
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

    # Early stopping:
    validation_split = 0.2
    patience = 10

    # Plotting:
    curve_interval = 50
    num_example_curves = 4

    # ------------------------
    # INFERENCE PARAMETERS:
    # ------------------------
    # Used if mode == "infer"
    T_infer = 50
    temperature_infer = 22.5
    model_path_infer = "decision_net.pth"
    hidden_sizes_infer = [64, 64]
    lstm_hidden_size_infer = 32
    W_initial_infer = 1
    S_initial_infer = 1

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

        # 2) Instantiate decision network of chosen type
        if decision_type == "mlp":
            decision_net = MLPDecisionNetwork(hidden_sizes=hidden_sizes_train)
        elif decision_type == "linear":
            decision_net = LinearDecisionNetwork()
        elif decision_type == "lstm":
            decision_net = LSTMDecisionNetwork(hidden_size=lstm_hidden_size)
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
        )
        np.save("inference_pred_curve.npy", pred_curve)
        print("Predicted curve saved as 'inference_pred_curve.npy'")

    else:
        print("Unknown mode! Set mode = 'train' or 'infer'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
