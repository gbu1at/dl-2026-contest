import logging
import os
from datetime import datetime
import json

class ExperimentLogger:
    def __init__(self, exp_dir, console=True):
        os.makedirs(exp_dir, exist_ok=True)
        self.exp_dir = exp_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(exp_dir, f"log_{timestamp}.txt")
        self.metrics_file = os.path.join(exp_dir, "metrics.json")

        self.logger = logging.getLogger("ExperimentLogger")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        if console:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.metrics = {}

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def log_metric(self, name, value, epoch=None):
        if name not in self.metrics:
            self.metrics[name] = []

        if epoch is not None:
            while len(self.metrics[name]) <= epoch:
                self.metrics[name].append(None)
            self.metrics[name][epoch] = value
        else:
            self.metrics[name].append(value)

        self._save_metrics()

    def _save_metrics(self):
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=4)

    def get_log_file(self):
        return self.log_file

    def get_metrics_file(self):
        return self.metrics_file


def get_logger(cfg):
    return ExperimentLogger(cfg["experiment_dir"])