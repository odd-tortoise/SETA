import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 1) Environment: wraps the worker/spawner simulator step‐by‐step
# ──────────────────────────────────────────────────────────────────────────────

class WorkerAgent:
    def __init__(self, initial_age: float = 0.0):
        self.age = float(initial_age)
    def step_age(self):
        self.age += 1.0

class SpawnerAgent:
    def __init__(self, initial_age: float = 0.0):
        self.age = float(initial_age)
    def step_age(self):
        self.age += 1.0

class RLEnvironment:
    """
    A simple RL environment where:
      - state s_t = [W_count, temperature, time_step]
      - action a_t ∈ {0,1,...,A_max} = spawn count that step
      - reward r_t = - (W_t - W*_t)^2, where W*_t is a reference logistic curve
      - episode length = T steps
    """

    def __init__(
        self,
        T: int,
        A_max: int,
        temp_min: float,
        temp_max: float,
        logistic_ref_fn,
        device: torch.device = torch.device("cpu")
    ):
        """
        Args:
          - T               : episode length (number of steps)
          - A_max           : maximum discrete spawn action
          - temp_min, temp_max : range of temperatures to sample each episode
          - logistic_ref_fn : function f(time_tensor, temperature) → Tensor(T,)
          - device          : torch.device
        """
        self.T = T
        self.A_max = A_max
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.logistic_ref_fn = logistic_ref_fn
        self.device = device

        # These will get reset each episode
        self.temperature = None
        self.workers = None
        self.spawners = None
        self.current_step = None
        self.reference_curve = None

    def reset(self, temperature: float):
        """
        Start a new episode at fixed temperature. Returns initial state s_0.
        """
        self.temperature = float(temperature)
        # initialize one worker (age=0) and one spawner (age=0)
        self.workers = [WorkerAgent(0.0)]
        self.spawners = [SpawnerAgent(0.0)]
        self.current_step = 0

        # Precompute reference logistic curve W*_0...W*_{T-1}
        t_grid = torch.arange(0, self.T, dtype=torch.float32, device=self.device)
        self.reference_curve = self.logistic_ref_fn(t_grid, self.temperature).cpu().numpy()  # shape (T,)

        # initial state: W_count=1, temp, t=0
        s0 = np.array([1.0, self.temperature, 0.0], dtype=np.float32)
        return s0

    def step(self, action: int):
        """
        Apply action = number of new workers to spawn at this time step.
        Returns:
          - next_state: np.array([W_count_next, temperature, t+1])
          - reward    : float
          - done      : bool
        """
        t = self.current_step

        # 1) Evolve ages
        for w in self.workers:
            w.step_age()
        for s in self.spawners:
            s.step_age()

        # 2) Apply spawn action
        if action > 0:
            for _ in range(action):
                self.workers.append(WorkerAgent(0.0))

        # 3) Compute reward (negative squared error)
        W_count = len(self.workers)  # after spawn
        W_ref = self.reference_curve[t]
        reward = - (W_count - W_ref) ** 2

        # 4) Advance time
        self.current_step += 1
        done = (self.current_step >= self.T)

        if not done:
            next_state = np.array([float(len(self.workers)), self.temperature, float(self.current_step)],
                                  dtype=np.float32)
        else:
            next_state = None

        return next_state, reward, done


# ──────────────────────────────────────────────────────────────────────────────
# 2) Policy Network Base + MLP Implementation
# ──────────────────────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    Abstract base class for a discrete‐action policy πθ(a | s).
    Must implement forward(s) → logits over actions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Given state tensor of shape (batch_size=1, state_dim),
        returns a tensor of shape (1, A_max+1) giving logits for each discrete action.
        Subclasses must override.
        """
        raise NotImplementedError


class MLPPolicy(PolicyNetwork):
    """
    A simple 3→hidden→(A_max+1) MLP policy.  
    Input: s_t = [W_count, temperature, time_step]
    Output: logits over actions {0,1,2,…,A_max}
    """
    def __init__(self, state_dim: int, hidden_sizes: list, A_max: int):
        super().__init__()
        dims = [state_dim] + hidden_sizes + [A_max + 1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-2], dims[-1]))  # final logits layer
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: Tensor of shape (1, 3)
        returns: logits Tensor of shape (1, A_max+1)
        """
        return self.net(state)


# ──────────────────────────────────────────────────────────────────────────────
# 3) REINFORCE Agent (On‐Policy Policy Gradient)
# ──────────────────────────────────────────────────────────────────────────────

class REINFORCEAgent:
    """
    Implements a simple REINFORCE (Monte‐Carlo policy gradient) update:
      - Collect one full episode (T steps)
      - Compute returns G_t = sum_{k=t to T-1} γ^{k-t} * r_k
      - Loss = - ∑_{t=0..T-1} [log πθ(a_t|s_t) * G_t]
    """

    def __init__(self, policy: PolicyNetwork, optimizer: torch.optim.Optimizer, gamma: float = 1.0):
        """
        Args:
          - policy    : instance of PolicyNetwork (e.g., MLPPolicy)
          - optimizer : torch optimizer for policy.parameters()
          - gamma     : discount factor in [0,1]
        """
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma

    def select_action(self, state: np.ndarray) -> tuple[int, torch.Tensor]:
        """
        Given a NumPy state (shape (3,)), convert to Tensor, run policy to get action distribution,
        sample an action, and return (action, log_prob).
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, 3)
        logits = self.policy(state_tensor)                                    # (1, A_max+1)
        probs = F.softmax(logits, dim=1)                                       # (1, A_max+1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()                                                 # shape ()
        log_prob = dist.log_prob(action)                                       # shape ()
        return int(action.item()), log_prob

    def update_policy(self, log_probs: list, rewards: list):
        """
        Perform one REINFORCE update given:
          - log_probs : list of length T of log π(a_t|s_t) tensors
          - rewards   : list of length T of scalar rewards
        """
        T = len(rewards)
        returns = []
        G = 0.0
        # Compute discounted returns backward:
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)  # shape (T,)

        # Normalize returns (optional, but often helpful):
        if returns.std(unbiased=False) > 1e-8:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        policy_loss = 0.0
        for log_p, Gt in zip(log_probs, returns):
            policy_loss += -log_p * Gt

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


# ──────────────────────────────────────────────────────────────────────────────
# 4) Trainer: orchestrates episodes, logs average returns, and optionally plots
# ──────────────────────────────────────────────────────────────────────────────

class RLTrainer:
    """
    Runs many episodes of REINFORCE to train the policy.
    """

    def __init__(
        self,
        env: RLEnvironment,
        agent: REINFORCEAgent,
        num_episodes: int = 1000,
        print_interval: int = 50,
        moving_avg_window: int = 100
    ):
        """
        Args:
          - env                : instance of RLEnvironment
          - agent              : instance of REINFORCEAgent
          - num_episodes       : total episodes to train
          - print_interval     : log stats every print_interval episodes
          - moving_avg_window  : window for computing moving average of episodic returns
        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.print_interval = print_interval
        self.moving_avg_window = moving_avg_window

        self.episode_returns = []

    def train(self):
        """
        Run full training loop over self.num_episodes.
        """
        for ep in range(1, self.num_episodes + 1):
            # Sample a random temperature from [temp_min, temp_max]
            temperature = float(np.random.uniform(self.env.temp_min, self.env.temp_max))

            # Reset environment
            state = self.env.reset(temperature)

            log_probs = []
            rewards = []
            ep_return = 0.0

            done = False
            while not done:
                # Agent selects action from policy
                action, log_p = self.agent.select_action(state)

                # Environment steps
                next_state, reward, done = self.env.step(action)

                log_probs.append(log_p)
                rewards.append(reward)
                ep_return += reward

                state = next_state if not done else None

            # After full episode, update policy
            self.agent.update_policy(log_probs, rewards)

            self.episode_returns.append(ep_return)

            # Logging
            if ep % self.print_interval == 0:
                recent_returns = self.episode_returns[-self.moving_avg_window:]
                avg_return = np.mean(recent_returns) if len(recent_returns) > 0 else self.episode_returns[-1]
                print(f"Episode {ep:>4d} | LastReturn = {ep_return:.2f} | "
                      f"AvgReturn({self.moving_avg_window}) = {avg_return:.2f}")

        # At the end, plot the learning curve
        self._plot_learning_curve()

    def _plot_learning_curve(self):
        """
        Plot episode returns and moving average.
        """
        returns = np.array(self.episode_returns)
        window = self.moving_avg_window
        if len(returns) >= window:
            moving_avg = np.convolve(returns, np.ones(window) / window, mode='valid')
        else:
            moving_avg = returns

        plt.figure(figsize=(8, 5))
        plt.plot(returns, label="Episode Return")
        if len(returns) >= window:
            plt.plot(np.arange(window - 1, len(returns)), moving_avg, label=f"MA({window})", color='red')
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("REINFORCE Training Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 5) Example Usage
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1) Define the same logistic reference function used earlier:
    def logistic_curve_fn(tensor: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Logistic growth: W*_t = K(T) / [1 + (K(T) - 1) * exp(-r(T)*t)]
        where K(T) = K_max * (T/T_max), r(T) = r_max * (T/T_max).
        """
        T_max = 30.0
        K_max = 100.0
        r_max = 0.2
        alpha = temperature / T_max
        K = K_max * alpha
        r = r_max * alpha
        exp_term = torch.exp(-r * tensor)
        return K / (1.0 + (K - 1.0) * exp_term)

    # 2) Create environment
    T = 50           # episode length
    A_max = 5        # allow spawning 0..5 workers per step
    temp_min = 15.0
    temp_max = 30.0
    device = torch.device("cpu")
    env = RLEnvironment(
        T=T,
        A_max=A_max,
        temp_min=temp_min,
        temp_max=temp_max,
        logistic_ref_fn=logistic_curve_fn,
        device=device
    )

    # 3) Build policy network & REINFORCE agent
    state_dim = 3            # [W_count, temp, t]
    hidden_sizes = [64, 64]
    policy = MLPPolicy(state_dim=state_dim, hidden_sizes=hidden_sizes, A_max=A_max).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    agent = REINFORCEAgent(policy=policy, optimizer=optimizer, gamma=1.0)

    # 4) Create trainer and run
    trainer = RLTrainer(
        env=env,
        agent=agent,
        num_episodes=1000,            # e.g. 500 episodes
        print_interval=50,
        moving_avg_window=25
    )
    trainer.train()

    # 5) After training, you can evaluate the final policy at a fixed temperature
    def evaluate_policy(env, agent, temperature):
        """
        Run a single episode with 'full display' to see how well it tracks the logistic curve.
        """
        state = env.reset(temperature)
        W_counts = []
        W_refs = env.reference_curve.copy()

        done = False
        while not done:
            # We pick the action with highest probability (greedy) during evaluation
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = agent.policy(state_tensor)
            probs = F.softmax(logits, dim=1)
            action = torch.argmax(probs, dim=1).item()

            next_state, reward, done = env.step(action)
            W_counts.append(state[0])  # W_count at this step
            state = next_state if not done else None

        # plot
        plt.figure(figsize=(6, 4))
        plt.plot(W_counts,     linestyle='-',  linewidth=2, label="Policy W_count")
        plt.plot(W_refs,       linestyle='--', linewidth=2, label="Logistic Reference")
        plt.xlabel("Time Step")
        plt.ylabel("Number of Workers")
        plt.title(f"Evaluation @ T={temperature:.1f}°C")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Example evaluation at 22.5°C
    evaluate_policy(env, agent, temperature=22.5)
