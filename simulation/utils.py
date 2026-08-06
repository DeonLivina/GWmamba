import yaml
from types import SimpleNamespace


def dict_to_namespace(d):
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            v = dict_to_namespace(v)
        setattr(ns, k, v)
    return ns


def load_config(config_path: str):
    with open(config_path) as f:
        data = yaml.safe_load(f)

    config = dict_to_namespace(data)
    return config
