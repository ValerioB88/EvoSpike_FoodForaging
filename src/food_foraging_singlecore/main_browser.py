from src.builders import build_snn
from src.food_foraging_singlecore.environment import Environment
from src.browser_utils.foraging_canvas import ForagingCanvas
from src.utils import food_foraging_input, food_foraging_spikes_action, Epoch
from src.browser_utils.additional_modules import *
from src.food_foraging_singlecore.agent import EvolPar, ForagingAgent
from collections import deque
from functools import partial
import uvicorn
import numpy as np

DT = 1
radians_num_sensors = 10
agent_class = partial(ForagingAgent,
                      input_to_spikes=partial(food_foraging_input,
                                              rad_num_sensors=radians_num_sensors,
                                              max_freq=60),
                      output_spikes_to_action=partial(food_foraging_spikes_action,
                                                      forward_displ_dt=0.7,
                                                      rad_displ_dt=0.3),
                      evol_pars=[EvolPar('max_energy_freq', 20, 0, 500, init=lambda: 120),
                                 EvolPar('max_radians_freq', 20, 0, 500, init=lambda: 120),
                                 EvolPar('color', 5, 0, 255, init=lambda: float(np.random.randint(0, 256)))],

                      snn_builder=partial(build_snn,
                                          dt=DT,
                                          n_input=2+radians_num_sensors,
                                          n_hidden=10,
                                          n_output=3,
                                          update_rule=None, # FFNA_one_hot_on_trace,
                                          # run_time=1,
                                          # w_type='random_unif',
                                          # output_scale=0.001,
                                          input_name='I',
                                          output_name='O'),
                      lr_net_builder=None, #partial(build_lr_net,
                                             # n_hidden=10,
                                             # init_type='none'),
                      evolve_substrate=True)

epochs = deque([Epoch(10000, [0.8] * 200)])
size = (500, 500)
env = Environment
env_args = dict(initial_population=100, epochs=epochs, agent_class=agent_class, size=size, dt=DT)


foraging_canvas = ForagingCanvas(canvas_height=size[0], canvas_width=size[1])
input = PlotInput(names=['energy', 'd_food', 'rad_food'], location='right')
output = PlotOutput(names=['forward', 'radians'], location='right')
output_spikes = PlotOutputSumSpikes(names=['forward', 'left', 'right'], location='right')

server = MyCustomAPI(env, [foraging_canvas, input, output, output_spikes], env_args)

if __name__ == "__main__":
    connected = False
    port = 8000
    while not connected:
        try:
            uvicorn.run("__main__:server", host="localhost", port=port)
            connected = True
        except:
            port += 1

# env.step()

