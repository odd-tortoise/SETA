import torch
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict, fields

@dataclass
class AgentState:
    """
    Base dataclass for an agent’s state.  
    Currently holds only 'age', but can be extended with new fields (workload, energy, etc.).
    """
    age: float = 0.0


@dataclass
class WorkerState(AgentState):
    """
    State for a WorkerAgent. Inherits 'age', and may add e.g. 'workload' in the future.
    """
    size: float = 1.0


@dataclass
class SpawnerState(AgentState):
    """
    State for a SpawnerAgent. Inherits 'age', and may add e.g. 'fertility' in the future.
    """
    fertility: float = 1.0




# ──────────────────────────────────────────────────────────────────────────────
# 1) Agent Definitions (using dataclass states)
# ──────────────────────────────────────────────────────────────────────────────

class Agent:
    """
    Base class for any agent. Holds a dataclass ‘state’ instance
    whose fields can be converted to a vector for ML / GNN usage.
    """
    def __init__(self, state: AgentState):
        """
        Args:
          - state: instance of AgentState (or subclass)
        """
        self.state = state

    def get_state_vector(self) -> torch.Tensor:
        """
        Convert the dataclass fields to a 1D tensor, in the declared field order.
        E.g., [age, workload, ...]
        """
        vals = [getattr(self.state, f.name) for f in fields(self.state)]
        return torch.tensor(vals, dtype=torch.float32)

    def set_state_vector(self, vec: Union[List[float], torch.Tensor]):
        """
        Overwrite this agent’s state from a list or tensor, respecting field order.
        """
        if isinstance(vec, torch.Tensor):
            vals = vec.tolist()
        else:
            vals = vec
        field_names = [f.name for f in fields(self.state)]
        if len(vals) != len(field_names):
            raise ValueError("Length of vector does not match number of state fields.")
        kwargs = {name: float(val) for name, val in zip(field_names, vals)}
        cls = type(self.state)
        self.state = cls(**kwargs)

    def get_state_dict(self) -> Dict[str, float]:
        """Return a dict of field_name→value for this agent’s state."""
        return asdict(self.state)


class WorkerAgent(Agent):
    """
    Worker agent. Uses WorkerState by default.
    """
    def __init__(self, state: Optional[WorkerState] = None):
        if state is None:
            state = WorkerState()
        super().__init__(state)


class SpawnerAgent(Agent):
    """
    Spawner agent. Uses SpawnerState by default.
    """
    def __init__(self, state: Optional[SpawnerState] = None):
        if state is None:
            state = SpawnerState()
        super().__init__(state)
