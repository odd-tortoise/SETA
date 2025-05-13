# dynamics.py

import numpy as np

class DAESolver:
    """
    Very simple Euler integrator for a system of ODEs.
    'func' should be a callable f(t, y) returning dy/dt as same-shaped array.
    """
    def __init__(self, func):
        self.func = func

    def step(self, t: float, states: np.ndarray, dt: float) -> np.ndarray:
        """
        Advance states by one Euler step: y_{n+1} = y_n + dt * f(t_n, y_n)
        Args:
          t: current time
          states: np.ndarray of shape (N, state_dim)
          dt: timestep
        Returns:
          new_states: same shape as states
        """
        dydt = self.func(t, states)
        return states + dt * dydt
