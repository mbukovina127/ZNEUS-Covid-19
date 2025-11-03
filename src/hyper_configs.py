sweep_config = {
    "method": "random",  # or "grid", "bayes"
    "metric": {"name": "loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.01},
        "hidden_size_1": {"values": [8, 16, 32, 64]},
        "hidden_size_2": {"values": [8, 16, 32, 64]},
        "hidden_size_3": {"values": [8, 16, 32, 64]},
        "n_layers": {"values": [1, 2, 3]},
        "dropout": {"min": 0.0, "max": 0.5},
        "optimizer": {"values": ["Adam", "SGD"]},
        "batch_size": {"values": [32, 64, 128, 512]}
    }
}