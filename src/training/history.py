# src/training/history.py

import os
import json

from dataclasses import dataclass, field


HISTORY_KEYS = (
    "loss_train", "loss_val",
    "MAE_train", "MAE_val",
    "MSE_train", "MSE_val",
    "Edge_train", "Edge_val",
    "SSIM_train", "SSIM_val",
    "lr", "epoch_time",
)


class History:
    def __init__(self, data=None):
        data = data if data is not None else {} 
        self.data = data 
        self._data = {k: list(data.get(k, [])) for k in HISTORY_KEYS}

    def log_epoch(self, **kwargs):
        missing = set(HISTORY_KEYS) - set(kwargs)
        extra = set(kwargs) - set(HISTORY_KEYS)
        if missing:
            raise ValueError(f"Missing history keys: {missing}")
        if extra:
            raise ValueError(f"Unknown history keys: {extra}")
        for k, v in kwargs.items():
            self._data[k].append(v)

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data["loss_train"])

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self._data, f, indent=4)

    def truncate(self, start_epoch: int):
        self._data = {k: v[:start_epoch] for k, v in self._data.items()}
        print(f"\t|\t|- Resuming from epoch {len(self)}")

    @classmethod
    def load(cls, history_path: str) -> "History":
        if not os.path.exists(history_path):
            print(f"\t|\t|- No history file found at '{history_path}', starting fresh.")
            return cls()
        print(f"\t|\t|- Loading training history from '{history_path}'.")
        with open(history_path, "r") as f:
            raw = json.load(f)
        return cls(raw)


