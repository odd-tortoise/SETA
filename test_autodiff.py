import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import copy

# ----- Ground Truth Dynamics -----
def true_a(temp): 
    return torch.sin(temp) * 0.5 + 1.0
true_b = -0.8

# ----- Neural Net + Trainable Parameter -----
class Thinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.b_raw = nn.Parameter(torch.tensor(0.0))  # unbounded raw param

    def forward(self, temp):
        # Clamp outputs to smaller range for stability
        a = torch.tanh(self.net(temp)) * 1 + 1  # a in [-0.5, 0.5]
        b = torch.tanh(self.b_raw) * 1      # b in [-0.5, 0.5]
        return a.squeeze(), b

# ----- ODE Function for torchdiffeq -----
class ODEFunc(nn.Module):
    def __init__(self, model, temps, times):
        super().__init__()
        self.model = model
        self.temps = temps
        self.times = times

    def forward(self, t, x):
        # Clamp index to avoid out-of-bounds
        raw_idx = (t / self.times[-1]) * (len(self.temps) - 1)
        idx = int(torch.clamp(raw_idx, 0, len(self.temps) - 1))
        temp_t = self.temps[idx].unsqueeze(0).unsqueeze(0)
        a_t, b = self.model(temp_t)
        return a_t * x + b * torch.sin(x)

# ----- Ground Truth Simulation -----
def generate_ground_truth(times, temps):
    x = torch.tensor([1.0])
    x_vals = []

    for i, t in enumerate(times):
        temp = temps[i].unsqueeze(0)
        a = true_a(temp)
        dxdt = a * x + true_b * torch.sin(x)
        dt = times[1] - times[0]
        x = x + dxdt * dt
        x_vals.append(x.item())

    return torch.tensor(x_vals)

# ----- Training with Restart on Blow-up -----
def train(model, times, temps, x_true, n_epochs=2000, lr=1e-2, max_retries=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []

    last_good_state = copy.deepcopy(model.state_dict())
    last_good_optimizer = copy.deepcopy(optimizer.state_dict())

    for epoch in range(n_epochs):
        for attempt in range(max_retries):
            try:
                optimizer.zero_grad()

                ode_func = ODEFunc(model, temps, times)
                x0 = torch.tensor([1.0])
                x_pred = odeint(
                    ode_func, x0, times, 
                    method='dopri5', atol=1e-5, rtol=1e-5
                )

                loss = loss_fn(x_pred.squeeze(), x_true)

                # Check for NaNs or huge values => blow up
                if (torch.isnan(loss) or
                    loss.item() > 1e4 or
                    torch.any(torch.isnan(x_pred)) or
                    torch.any(torch.abs(x_pred) > 1e3)):
                    raise ValueError("ODE blow-up detected")

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Save good state
                last_good_state = copy.deepcopy(model.state_dict())
                last_good_optimizer = copy.deepcopy(optimizer.state_dict())

                print(f"[Epoch {epoch}] Loss: {loss.item():.6f} | b: {model.b_raw.item():.6f}")
                losses.append(loss.item())
                break  # exit retry loop on success

            except Exception as e:
                print(f"[Epoch {epoch}] Attempt {attempt+1}/{max_retries} blow-up: {e}")

                # Restore last good state to avoid compounding errors
                model.load_state_dict(last_good_state)
                optimizer.load_state_dict(last_good_optimizer)

                # Reduce learning rate to stabilize training
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = old_lr * 0.5
                    print(f"Reducing learning rate from {old_lr} to {param_group['lr']}")

                if attempt == max_retries - 1:
                    raise RuntimeError(f"Training failed after {max_retries} attempts at epoch {epoch}")

    return losses, x_pred

# ----- Main -----
if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    # Setup time and temperature signal
    T, dt = 10.0, 0.05
    times = torch.arange(0, T, dt)
    temps = times.clone() + 1 # e.g., T(t) = t

    # Generate ground truth
    x_true = generate_ground_truth(times, temps)

    # Initialize model
    model = Thinker()

    # Train model
    losses, x_pred = train(model, times, temps, x_true, n_epochs=2000, lr=1e-2)

    # Plot results
    plt.figure(figsize=(10, 4))
    plt.plot(times, x_true, label="Ground Truth")
    plt.plot(times, x_pred.detach().squeeze(), '--', label="Predicted")
    plt.legend()
    plt.title("Neural ODE Simulation with Safe Restart & Gradient Clipping")
    plt.xlabel("Time")
    plt.ylabel("x(t)")
    plt.tight_layout()
    plt.show()
