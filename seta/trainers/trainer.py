import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

import time

from ..core.simulation import Simulator
from ..environment.environment import Environment

class Trainer:
    """
    Encapsulates the training loop, early stopping, and periodic plotting.
    """

    def __init__(
        self,
        decision_net: nn.Module,
        simulator : Simulator,
        dataset,
        device: torch.device,
        num_epochs: int = 50,
        batch_size: int = 16,
        lr: float = 1e-3,
        validation_split: float = 0.2,
        patience: int = 5,
        visualize: bool = True,
        curve_interval: int = 5,
        num_example_curves: int = 4,
        output_model_path: str = "decision_net.pth"
    ):
        """
        Args:
          - decision_net      : instance of DecisionNetwork (MLP, Linear, LSTM, or GNN)
          - simulator         : instance of Simulator (configured with same decision_net)
          - dataset           : instance of CustomFunctionDataset
          - device            : torch.device
          - num_epochs        : max number of epochs to train
          - batch_size        : batch size
          - lr                : learning rate for Adam
          - validation_split  : fraction of data to hold out as validation
          - patience          : early‐stopping patience
          - visualize         : whether to plot curves & losses
          - curve_interval    : how often (epochs) to plot example curves
          - num_example_curves: how many examples from validation to plot
          - output_model_path : where to save the best state_dict
        """
        self.decision_net = decision_net
        self.simulator = simulator
        self.dataset = dataset
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.validation_split = validation_split
        self.patience = patience

        self.visualize = visualize

        self.curve_interval = curve_interval
        self.num_example_curves = num_example_curves
        
        self.output_model_path = output_model_path

        # Prepare train/val split
        total = len(dataset)
        val_count = int(validation_split * total)
        train_count = total - val_count
        self.train_set, self.val_set = random_split(dataset, [train_count, val_count])

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False)

        # Pre‐select some validation examples for plotting
        if self.num_example_curves > len(self.val_set):
            raise ValueError("num_example_curves cannot exceed validation set size")
        self.example_indices = list(range(self.num_example_curves))
        self.example_temps = []
        self.example_true_curves = []
        for i in self.example_indices:
            temp_i, curve_i = self.val_set[i]
            self.example_temps.append(temp_i)
            self.example_true_curves.append(curve_i.numpy())

    def train(self):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.decision_net.parameters(), lr=self.lr)
        self.decision_net.to(self.device)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        train_losses = []
        val_losses = []

        print(f"Starting training (patience={self.patience} epochs) …")
        for epoch in range(1, self.num_epochs + 1):
            # — Training Phase —
            self.decision_net.train()
            running_train_loss = 0.0

            for envs_batch, true_curves in self.train_loader:
                envs_batch = envs_batch.to(self.device)
                true_curves = true_curves.to(self.device)

                optimizer.zero_grad()
                batch_preds = []

                for i in range(envs_batch.shape[0]):

                    env_i = envs_batch[i]
                    env = Environment(env_i)
                    pred_curve_i = self.simulator.run(env, "train", 1)    # Tensor (T,)
                    batch_preds.append(pred_curve_i.unsqueeze(0))  # shape (1, T)

                t0 = time.time()
           
                batch_pred = torch.cat(batch_preds, dim=0)  # shape (batch_size, T)
                loss = criterion(batch_pred, true_curves)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
                print("Batch update: ",time.time() - t0)

            train_loss = running_train_loss / len(self.train_loader)
            train_losses.append(train_loss)

            # — Validation Phase —
            self.decision_net.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for envs_v, true_curves_v in self.val_loader:
                    envs_v = envs_v.to(self.device)
                    true_curves_v = true_curves_v.to(self.device)

                    batch_preds_v = []
                    for i in range(envs_v.shape[0]):
                        env_iv = envs_v[i]
                        env = Environment(env_iv)
                        pred_curve_iv = self.simulator.run(env)
                        batch_preds_v.append(pred_curve_iv.unsqueeze(0))
                    batch_pred_v = torch.cat(batch_preds_v, dim=0)
                    loss_v = criterion(batch_pred_v, true_curves_v)
                    running_val_loss += loss_v.item()

            val_loss = running_val_loss / len(self.val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch:>3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.decision_net.state_dict().items()}
                no_improve = 0
                best_epoch = epoch
            else:
                no_improve += 1

            if no_improve >= self.patience:
                print(f"No improvement for {self.patience} epochs → early stopping at epoch {epoch}.")
                break

            # Plot example curves every self.curve_interval epochs (or final epoch)
            if self.visualize and (epoch % self.curve_interval == 0 or epoch == self.num_epochs):
                self._plot_example_curves(epoch)

        # Load best weights
        if best_state is not None:
            self.decision_net.load_state_dict(best_state)
            print(f"Loaded best model (epoch {best_epoch}) with Val Loss = {best_val_loss:.6f}")

        # Save best model
        torch.save(self.decision_net.state_dict(), self.output_model_path)
        print(f"Best model saved to '{self.output_model_path}'")

        # Plot train/val loss curves
        if self.visualize:
            self._plot_loss_curves(train_losses, val_losses)

        return best_val_loss

    def _plot_example_curves(self, epoch: int):
        """
        Plot a few validation examples: true vs. predicted curves.
        """
        self.decision_net.eval()
        with torch.no_grad():
            plt.figure(figsize=(8, 5))
            for i in range(self.num_example_curves):
                temp_i = self.example_temps[i]
                true_curve_i = self.example_true_curves[i]
                env = Environment(temp_i)
                pred_curve_i = self.simulator.run(env).cpu().numpy()

                t_axis = np.arange(self.simulator.T_max)
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

            plt.title(f"Epoch {epoch}: Example Curves")
            plt.xlabel("Time Step")
            plt.ylabel("W Count")
            plt.legend(ncol=2, fontsize="small", loc="upper left")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"epoch_{epoch:03d}_curves.png")
            plt.show()

    def _plot_loss_curves(self, train_losses, val_losses):
        """
        Plot training vs. validation loss over epochs.
        """
        epochs = np.arange(1, len(train_losses) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss", marker="o")
        plt.plot(epochs, val_losses,   label="Val Loss",   marker="x")
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("losses.png")
        plt.show()
