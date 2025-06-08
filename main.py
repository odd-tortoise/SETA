import torch
from seta import (
    MLPDecisionNetwork,
    LinearDecisionNetwork,
    LSTMDecisionNetwork,
    GNNDecisionNetwork, 
    Simulator,
    CustomFunctionDataset,
    Trainer,
    run_inference)

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
    # Choose mode = “train” or “infer”
    mode = "train"
    #mode = "infer"

    # Choose decision network type: "mlp", "linear", "lstm", or "gnn"
    decision_type = "mlp"  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Training Configuration ──────────────────────────────────────────
    T_train = 50
    epochs = 150
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
    validation_split = 0.2
    patience = 12
    curve_interval = 15
    num_example_curves = 4

    # ─── Inference Configuration ────────────────────────────────────────
    T_infer = 50
    temperature_infer = 22.5
    hidden_sizes_infer = [64, 64]
    lstm_hidden_size_infer = 32
    W_initial_infer = 1
    S_initial_infer = 1
    full_display = True
    display_interval = 10

    if mode == "train":
        # 1) Build dataset
        dataset = CustomFunctionDataset(
            T=T_train,
            num_examples=num_examples,
            temp_min=temp_min,
            temp_max=temp_max,
            curve_fn=logistic_curve_fn
        )

        # 2) Instantiate decision network
        if decision_type == "mlp":
            decision_net = MLPDecisionNetwork(hidden_sizes=hidden_sizes_train)
        elif decision_type == "linear":
            decision_net = LinearDecisionNetwork()
        elif decision_type == "lstm":
            decision_net = LSTMDecisionNetwork(hidden_size=lstm_hidden_size)
        elif decision_type == "gnn":
            decision_net = GNNDecisionNetwork()
        else:
            raise ValueError("Unknown decision_type")

        # 3) Instantiate simulator
        simulator = Simulator(
            T=T_train,
            W_initial=W_initial_train,
            S_initial=S_initial_train,
            decision_net=decision_net,
            device=device
        )

        # 4) Train
        trainer = Trainer(
            decision_net=decision_net,
            simulator=simulator,
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

    elif mode == "infer":
        run_inference(
            model_path=output_model_path,
            temperature=temperature_infer,
            T=T_infer,
            decision_type=decision_type,
            hidden_sizes=hidden_sizes_infer,
            lstm_hidden_size=lstm_hidden_size_infer,
            W_initial=W_initial_infer,
            S_initial=S_initial_infer,
            device=device,
            visualize=True,
            full_display=full_display,
            display_interval=display_interval
        )

    else:
        print("Unknown mode; must be 'train' or 'infer'.")

if __name__ == "__main__":
    main()
