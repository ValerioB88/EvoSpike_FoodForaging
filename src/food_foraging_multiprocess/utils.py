from dataclasses import dataclass, field
from typing import *
import numpy as np

from src.utils import EvolPar

shared_to_update = ['pos', 'direction']
shared_fixed = ['name', 'surname', 'radius', 'generation', 'color', 'fov', 'max_vision_dist', 'id', 'max_age', 'fertile_age_start', 'num_children', 'name_run', 'evol_pars', 'time_between_children', 'target_food', 'original_snn_state_dict']

@dataclass
class Message:
    info_to_pass_around: Dict = field(default_factory=lambda: {k: None for k in shared_to_update})
    index_food_eaten: List = None
    dead: bool = False
    reproduce: "self if the agent needs reproducing (it will be a parent) otherwise False" = False
    evol_pars_dict: dict = None
    original_snn_state_dict: None = None
    parent_info: dict = None


