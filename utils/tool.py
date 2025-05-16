import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import itertools
from itertools import product

def get_configs(configs):
    combinations = list(product(*configs.values()))
    list_configs = []
    for combo in combinations:
        config_dict = {key: value for key, value in zip(configs.keys(), combo)}
        list_configs.append(config_dict)
    return list_configs