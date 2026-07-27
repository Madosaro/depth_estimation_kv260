# src/config.py

import yaml

class ConfigNode:

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)

    def __getattr__(self, name: str):
        for value in self.__dict__.values():
            if isinstance(value, ConfigNode):
                try:
                    return getattr(value, name)
                except AttributeError:
                    continue 
                    
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __repr__(self):
        return f"ConfigNode({self.__dict__})"


def load_config(path: str) -> ConfigNode:
    print(f"\t|- Loading configuration from: {path}")
    with open(path, "r") as f:
        raw_dict = yaml.safe_load(f)
    return ConfigNode(raw_dict)