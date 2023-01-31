from copy import deepcopy

import numpy as np
import torch
import abc
from typing import *
from src.utils import EvolPar


class AgentSNN(abc.ABC):
    network = None
    id = None
    lr_net = None
    original_snn_state_dict = None
    evol_pars: List[EvolPar] = None


    def __init__(self, snn_builder, input_to_spikes, output_spikes_to_action, lr_net_builder=None, evolve_substrate=False, evol_pars: List[EvolPar] = None, parent=None):
        self.input_to_spikes = input_to_spikes()
        self.output_spikes_to_action = output_spikes_to_action()
        self.network = snn_builder(self)
        self.evolve_substrate = evolve_substrate

        self.evol_pars = evol_pars

        if parent is None:
            for evo in self.evol_pars:
                self.__setattr__(evo.name, evo.init())

            self.lr_net = lr_net_builder(self.network) if lr_net_builder else None

        else:

            if self.evolve_substrate:
                self.network.load_state_dict(parent.original_snn_state_dict)
                for k, v in self.network.connections.items():
                    v.w += torch.normal(0, 0.1, v.w.shape)
                    if v.b is not None:
                        v.b += torch.normal(0, 0.1, v.b.shape)

            # Mutate lr_net

            ## ToDo: How does this work? This doesn't seem to be inherited by the parent!!!
            if lr_net_builder:
                print("WARNING!! THIS IS WRONG!!! LR_NET IS NEVER INHERITED!! HOW CAN IT EVOLVE?!")
                for net in self.lr_net.values():
                    for k, v in net.state_dict().items():
                        v += torch.normal(0, 0.05, v.shape)
                        # stop = 1

            # Mutate other properties
            # maybe randomly shuffle before mutating
            for par in self.evol_pars:
                self.__setattr__(par.name, parent.__getattribute__(par.name))
                if hasattr(self, par.name):
                    self.__setattr__(par.name, np.clip(parent.__getattribute__(par.name) + (np.random.randn(1)[0] * par.mutation_var), par.min_value, par.max_value))
                else:
                    assert False, f"{par.name} not found in agent!"

        # This is done so that the "learnt" net won't be inherited (which would be Lamarckian), only the net the agent is born with.
        self.original_snn_state_dict = deepcopy(self.network.state_dict())

    @abc.abstractmethod
    def reset(self):
        pass
