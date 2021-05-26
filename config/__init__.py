import os
import copy
from config.parse_console import parse_arguments
from config.defaults import _C

_C.__init__()

def update_args(cfg_file=None, run_name=None, seed=None):
    if cfg_file and os.path.exists(cfg_file):
        _C.merge_from_file(cfg_file)
    if run_name is not None:
        _C.TRAIN.RUN_NAME = run_name
    if seed is not None:
        _C.TRAIN.SEED = seed
    return copy.deepcopy(_C)

__all__ = ["update_args", "parse_arguments"]
