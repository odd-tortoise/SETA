from typing import Callable, Dict
from dataclasses import fields

# Assume System and Agent classes (with dataclass states) are already defined/imported.

class Dynamics:
    """
    Stores a set of evolution rules (one per agent type) and applies them
    to a System, updating each agent’s state in place.

    Each rule is a function of signature:
        rule(agent: Agent, system: System) -> None
    The function should modify agent.state (or other aspects of the system) directly.

    If no rule is registered for a given agent_type, that agent’s state is left unchanged.
    """

    def __init__(self):
        # Map from agent_type (str) to evolution function
        self.rules: Dict[str, Callable[[object, object], None]] = {}

    def register_rule(self, agent_type: str, rule_fn: Callable[[object, object], None]):
        """
        Register an evolution rule for all agents of type `agent_type`.

        Args:
          - agent_type: the same string used when adding agents to System (e.g., "W", "S").
          - rule_fn: function(agent: Agent, system: System) -> None
                     This function will be called for each agent of that type,
                     and should update agent.state (and/or system) in place.
        """
        self.rules[agent_type] = rule_fn

    def apply(self, system):
        """
        Apply the registered evolution rules to every agent in the given System.

        For each agent at index i:
          - Look up agent_type = system.types[i]
          - If a rule is registered for agent_type, call rule_fn(agent, system)
          - Otherwise, do nothing (agent’s state remains the same)
        """
        for idx, agent in enumerate(system.agents):
            agent_type = system.types[idx]
            rule_fn = self.rules.get(agent_type, None)
            if rule_fn is not None:
                rule_fn(agent, system)
            # If no rule is registered, that agent “does not evolve” this step
