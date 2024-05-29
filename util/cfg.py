import yaml
import ruamel.yaml


def load_cfg(cfg_path):
    with open(cfg_path) as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    return cfg


def save_cfg(cfg, cfg_path):
    try:
        ruamel_yaml = ruamel.yaml.YAML()
        ruamel_yaml.preserve_quotes = True
        with open(cfg_path, "w") as cfg_file:
            ruamel_yaml.dump(cfg, cfg_file)
        return True
    except:
        return False
