# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/073_wandb.ipynb.

# %% auto 0
__all__ = ['get_wandb_agent', 'run_wandb_agent', 'wandb_agent', 'update_run_config', 'get_sweep_config']

# %% ../nbs/073_wandb.ipynb 3

from fastcore.script import *

from .imports import *
from .utils import *


# %% ../nbs/073_wandb.ipynb 4
def wandb_agent(script_path, sweep, entity=None, project=None, count=None, run=True):
    "Run `wandb agent` with `sweep` and `script_path"
    try: import wandb
    except ImportError: raise ImportError('You need to install wandb to run sweeps!')
    if 'program' not in sweep.keys(): sweep["program"] = script_path
    sweep_id = wandb.sweep(sweep, entity=entity, project=project)
    entity = ifnone(entity, os.environ['WANDB_ENTITY'])
    project = ifnone(project, os.environ['WANDB_PROJECT'])
    print(f"\nwandb agent {entity}/{project}/{sweep_id}\n")
    if run: wandb.agent(sweep_id, function=None, count=count)

get_wandb_agent = named_partial("get_wandb_agent", wandb_agent, run=False)

run_wandb_agent = named_partial("run_wandb_agent", wandb_agent, run=True)

# %% ../nbs/073_wandb.ipynb 5
def update_run_config(config, new_config, verbose=False):
    "Update `config` with `new_config`"
    config_dict = config.copy()
    if hasattr(config_dict, "sweep"):
        del config_dict["sweep"]
    for k, v in new_config.items():
        if k in config_dict.keys():
            if verbose:
                print(f"config.{k} {config_dict[k]} updated to {new_config[k]}")
            config_dict[k] = new_config[k]
        elif (
            hasattr(config_dict, "arch_config")
            and k in config_dict["arch_config"].keys()
        ):
            if verbose:
                print(
                    f"config.arch_config.{k} {config['arch_config'][k]} updated to {new_config[k]}"
                )
            config["arch_config"][k] = new_config[k]
        else:
            warnings.warn(f"{k} not available in config or config.arch_config")
    return config_dict

# %% ../nbs/073_wandb.ipynb 6
def get_sweep_config(config):
    "Get sweep config from `config`"
    if not hasattr(config, "sweep") or not config["sweep"]: 
        return {}
    if isinstance(config["sweep"], str):
        file_path = Path(config["sweep"])
        file_path = to_root_path(file_path)
        sweep = yaml2dict(file_path, attrdict=False)
    else:
        sweep = attrdict2dict(config["sweep"])
    if hasattr(sweep, "sweep"):
        sweep = sweep["sweep"]
    return sweep
