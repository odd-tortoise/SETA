import torch
from torch_geometric.utils import dense_to_sparse
from typing import Dict, List, Any, Union
from dataclasses import fields 

from ..agents.system import (
    System, SystemState
)

from ..dynamics.dynamic import (
    Dynamics
)

from ..thinkers.thinkers import (
    ThinkerBase
)

from ..environment.environment import (
    Environment
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
        T_max: int,
        system : System,
        system_dynamic: Dynamics,
        decision_net: ThinkerBase,
        act_rule : callable,
        device: torch.device,
    ):
        self.T_max = T_max
        self.system = system
        self.system_dynamics = system_dynamic

        # per ogni net una act rule (e più avanti un output per loss)
        self.decision_net = decision_net
        self.act_rule = act_rule

        self.device = device


    def run(
        self,
        environment: Environment,
        mode : str = "train",
        verbose: int = 0,
        output : str = None
    ) -> torch.Tensor:
        """
        Run a single simulation from t=0 … T-1 at fixed temperature.

        Args:
        - temperature    : float, fixed temperature for this simulation
        - verbose        : 0 = silent, 1 = major announcements, 2 = detailed step-by-step
        - return_history : if True, return a dict with extra diagnostics; otherwise, return only the (T,) tensor

        Returns:
        If return_history=False:
            - Tensor of shape (T,) with predicted W‐counts at each time step.

        If return_history=True:
            - Dict with keys:
                "pred_curve"   : Tensor (T,) of predicted W‐counts
                "pred_scalars" : List[float] of length T
                "worker_counts": List[int]   of length T
                "system_states": List[System] of length T (deep copies after evolve)
                "timestamps"   : List[int]   of length T
        """

        # 1) Initialize a fresh System
        self.system.initialize_system(environment,1,1)

        if verbose >= 1:
            print(f"\n=== Starting simulation - Mode: {mode} ===")
        
        if verbose >= 2:
            self.system.plot_graph(folder=output+"init")

        # 2) If using an LSTMThinker (or any Thinker with reset), reset now
        if hasattr(self.decision_net, "reset"):
            self.decision_net.reset()
            if verbose >= 2:
                print("Decision network internal state reset.")

        preds_run = []

        # 3) Main time‐step loop
        for t in range(self.T_max):
            # — PHASE 1: SENSE —
            if verbose >= 3:
                self.system.plot_graph(folder=output +f"{t}" )

            if verbose >= 2:
                print(f"[t={t}] Phase 1 (Sense)")

            self.system.read_env(env= environment)
            
            # — PHASE 2: EVOLVE —
            if verbose >= 2:
                print(f"[t={t}] Phase 2 (Evolve): applying dynamics to all agents.")

            self.system_dynamics.apply(system=self.system)
           
            if verbose >= 3:
                for idx, ag in enumerate(self.system.nodes):
                    print(f"    Node {idx} ({self.system.types[idx]}): {ag.get_state_dict()}")

            # — PHASE 3: DECISION —
            if verbose >= 2:
                print(f"[t={t}] Phase 3 (Think)")

            pred = self.decision_net.forward(system=self.system, t=t)
            preds_run.append(pred)
            

            # — PHASE 4: REFLECT —
            if verbose >= 2:
                print(f"[t={t}] Phase 4 (Act)")

            self.act_rule(self.system, pred)



            
      
        if verbose >= 1:
            print(f"=== Simulation complete.")
        
        if mode=="train":
            return torch.cat(preds_run, dim=0)  # Tensor of shape (T,)

