import yaml
import os

def load_cfg(cfg_path: str) -> dict:
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Конфиг не найден: {cfg_path}")
    
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    return cfg