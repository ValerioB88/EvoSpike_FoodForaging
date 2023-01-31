from dataclasses import make_dataclass, field, dataclass
from multiprocessing import Process
from multiprocessing.connection import Connection
from typing import List

from src.food_foraging_multiprocess.utils import shared_fixed, shared_to_update ##
from src.utils import EvolPar

# Proxy Agents are not real agent, but they are just a trace of the real agents in the processes.
# They contain a Connection object which is one end of a pipe which other end is possessed by their real counterpart in the process
# They contain information needed for plotting (but NOT for computation).
ProxyAgent = make_dataclass('ProxyAgent',
                            [(s, 'typing.Any', None) for s in shared_fixed + shared_to_update] +
                            [('parent_id', int, None),
                             ('process', Process, None),
                             ('connection', Connection, None),
                             ('children_ids', List[int], field(default_factory=lambda: []))])
