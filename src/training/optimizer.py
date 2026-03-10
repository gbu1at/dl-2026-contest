import torch


def get_optimizer(cfg, model):
    opt_name = cfg["opt_name"]
    opt_params = cfg["opt_params"]

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), **opt_params)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), **opt_params)
    else:
        raise ValueError(f"Unknown optimizer {opt_name}")