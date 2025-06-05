# If you need the Agent classes here (or else import them from a separate module):
class Agent:
    def __init__(self, initial_age: float = 0.0):
        self.age = float(initial_age)
    def step_age(self):
        self.age += 1.0

class WorkerAgent(Agent):
    pass

class SpawnerAgent(Agent):
    pass

