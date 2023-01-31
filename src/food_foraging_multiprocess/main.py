from src.builders import build_snn
from src.food_foraging_multiprocess.environment import Environment
from src.food_foraging_multiprocess.utils import shared_fixed, shared_to_update
from src.utils import Epoch, food_foraging_input, food_foraging_spikes_action, EvolPar, FoodEpochs, FoodFeedback
from functools import partial
from src.food_foraging_multiprocess.agent import ForagingAgentInProcess
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--gfx', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_known_args()[0]

DT = 1
radians_num_sensors = 10
agent_class = partial(ForagingAgentInProcess,
                      input_to_spikes=partial(food_foraging_input,
                                              rad_num_sensors=radians_num_sensors),
                      output_spikes_to_action=partial(food_foraging_spikes_action,
                                                      forward_displ_dt=3.7,
                                                      rad_displ_dt=0.3),
                      evol_pars=[EvolPar('max_energy_freq', 20, 0, 1000, init=lambda: 500),
                                 EvolPar('max_radians_freq', 20, 0, 1000, init=lambda: 500),
                                 EvolPar('color', 5, 0, 255, init=lambda: float(np.random.randint(0, 256)))],

                      snn_builder=partial(build_snn,
                                          dt=DT,
                                          n_input=2+radians_num_sensors,
                                          n_hidden=10,
                                          n_output=3,
                                          update_rule=None, #FFNA_one_hot_on_trace,
                                          # run_time=1,
                                          # w_type='random_unif',
                                          # output_scale=0.001,
                                          input_name='I',
                                          output_name='O'),
                      lr_net_builder=None, #partial(build_lr_net,
                                             # n_hidden=10,
                                             # init_type='none'),
                      evolve_substrate=True)

##########
min_food, max_food = 50, 120
x = np.arange(0, 100000, 1000)
food_manager = partial(FoodEpochs, epochs=[Epoch(ep_l, [0.8] * int(fd)) for ep_l, fd in zip(np.diff(x), np.exp(-0.5*(x/3000)**2) * (max_food - min_food) + min_food)], respawn_after=80)
# food_manager = partial(FoodEpochs, epochs=[Epoch(0, [0.8] * 80)], respawn_after=100)

############
food_manager = partial(FoodFeedback, start_with=120, check_every=10, stable_pop=80, min_food=30, max_food=120, k=0.02, respawn_after=80)

size = (500, 500)

env_args = dict(initial_population=100,
                agent_class=agent_class,
                tot_steps=100000,
                size=size,
                dt=DT,
                save_state=False,
                food_manager=food_manager,
                debug_mode=args.debug)
env = Environment


if args.gfx:
    import uvicorn
    from src.browser_utils.additional_modules import *
    from src.browser_utils.foraging_canvas import ForagingCanvas
    shared_to_update.extend(['energy', 'age', 'countdown_offspring_counter', 'inputs', 'outputs', 'sum_input_spikes', 'sum_output_spikes', 'sum_middle_spikes'])
    xmax = 50

    server = MyCustomAPI(env, [ForagingCanvas(canvas_height=size[0], canvas_width=size[1]),
                               PlotInput(names=['energy', 'd_food', 'rad_food'], location='right', xmax=xmax),
                               PlotOutput(names=['forward', 'radians'], location='right', xmax=xmax, title='output'),
                               # PlotOutputSumSpikes(names=['forward', 'left', 'right'], location='right', xmax=xmax, title='spike output'),
                               SpikePlot(names=['sum_input_spikes', 'sum_middle_spikes', 'sum_output_spikes'],
                                         location='left', width=480, height=200, xmax=xmax, add_random_noise=True)], env_args)
    connected = False
    port = 8000
    while not connected:
        try:
            uvicorn.run("__main__:server", host="localhost", port=port)
            connected = True
        except:
            port += 1
else:
    gen = iter(env(**env_args))

    while True:
        try:
            next(gen)
        except StopIteration:
            break
    print("FINISHED!")