import torch
from seta import (
    MLPThinker,
    System,
    Dynamics,
    Simulator,
    CustomFunctionDataset,
    Environment,
    Trainer)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def logistic_curve_fn(time_tensor: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Example logistic growth:
        W_true(t) = K(T) / [1 + (K(T) - 1) * exp(-r(T)*t)]
    """
    T_max = 30.0
    K_max = 100.0
    r_max = 0.2
    alpha = temperature / T_max
    K = K_max * alpha
    r = r_max * alpha
    exp_term = torch.exp(-r * time_tensor)  # shape (T,)
    return K / (1.0 + (K - 1.0) * exp_term)

def main():

    import pickle
    import numpy as np

    # Load the RBF interpolation from the file
    with open('rbf_interpolation_internodes.pkl', 'rb') as f:
        rbf_loaded = pickle.load(f)


    def curve_data(time_tensor: torch.Tensor, temperature: float) -> torch.Tensor:
        pred = []
        for el in time_tensor:
            pred.append(torch.tensor(rbf_loaded(np.array([el.item(),temperature]).reshape(1,-1))).float())
        return torch.cat(pred, dim=0)

    mode = "train"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Training Configuration ──────────────────────────────────────────
    T_max= 100
    epochs = 100
    batch_size = 8
    learning_rate = 1e-2
    num_examples = 80
    temp_min = 20.0
    temp_max = 40.0
    output_model_path = "decision_net.pth"
    
    validation_split = 0.2
    patience = 12
    curve_interval = 25
    num_example_curves = 4
    
    dataset = CustomFunctionDataset(
        T=T_max,
        num_examples=num_examples,
        temp_min=temp_min,
        temp_max=temp_max,
        curve_fn=curve_data
    )

    decision_net = MLPThinker([32,32])

        
    device = torch.device("cpu")
    system = System(device=device)
    dyn = Dynamics()

    def worker_rule(agent, system):
            """
            For a WorkerAgent:
            - increase 'workload' by 0.1 each step
            - if system.temperature > 20, increase 'age' by an extra 0.5
            """
            # agent.state is a WorkerState dataclass with fields (age, workload)
            agent.state.size += 0.1
            if system.get_system_var("temperature") > 20.0:
                agent.state.age += 0.5

    dyn.register_rule("W", worker_rule)

    def spawn_node_SAM(system, prediction):
        current_W = system.types.count("S")
        delta = prediction - current_W
        if delta > 0.0:
            n_to_spawn = int(torch.ceil(torch.tensor(delta)).item())
            for _ in range(n_to_spawn):
                system.add_node_SAM()


    sim = Simulator(
     T_max=T_max,
     system=system,
     system_dynamic= dyn,
     decision_net=decision_net,
     act_rule=spawn_node_SAM,
     device= "cpu"
    )

        # 4) Train
    trainer = Trainer(
        decision_net=decision_net,
        simulator=sim,
        dataset=dataset,
        device=device,
        num_epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        validation_split=validation_split,
        patience=patience,
        visualize=True,
        curve_interval=curve_interval,
        num_example_curves=num_example_curves,
        output_model_path=output_model_path
    )
    trainer.train()


    env_test = Environment(25)

    sim.run(env_test,"train",3)




if __name__ == "__main__":
    main()
