import random
import abc
from collections import deque
from typing import List
import pandas as pd
import numpy as np
import torch
from typing import *
from dataclasses import dataclass


@dataclass
class EvolPar:
    name: str
    mutation_var: float
    min_value: float
    max_value: float
    init: Callable


@dataclass
class Epoch:
    duration: int
    energy_list: List[float]

def convert_ranges(value, input_range, output_range):
    return output_range[0] + (((value - input_range[0]) / (input_range[1] - input_range[0])) * (output_range[1] - output_range[0]))

class Countdown():
    counter = np.inf
    started = False

    def __init__(self, start_value=100, autostart=False, callback=None, randomness=0):
        self.randomness = randomness
        self.start_value = start_value
        self.counter = self.start_value + (np.random.randint(-self.randomness, self.randomness) if self.randomness != 0 else 0)
        self.autostart = autostart
        self.callback = callback

    def start(self):
        self.started = True
        self.counter = self.start_value + (np.random.randint(-self.randomness, self.randomness) if self.randomness != 0 else 0)

    def step(self):
        if self.started:
            self.counter -= 1
            if self.counter < 0:
                if self.autostart:
                    self.counter = self.start_value + (np.random.randint(-self.randomness, self.randomness) if self.randomness != 0 else 0)
                else:
                    self.counter = 0
                    self.callback() if self.callback is not None else None
                    self.started = False
        return self.counter


class FoodToken():
    radius = 5

    def __init__(self, pos, energy, model, respawn_after=100):
        self.pos = pos
        self.energy = energy
        self.model = model
        self.respawn_after = respawn_after
        self.eaten = False
        self.countdown_respawn = Countdown(self.respawn_after)

    def get_eaten(self):
        self.eaten = True
        self.pos = np.array([-10.0, -10.0])
        if self.respawn_after < np.inf:
            self.countdown_respawn.start()

    def step(self):
        self.countdown_respawn.step()
        if self.countdown_respawn.counter == 0 and self.eaten:
            self.pos = self.model.get_random_coord()
            self.eaten = False



class CountdownList():
    def __init__(self, cdl=None):
        self.countdown_list = cdl if cdl is not None else []

    def add(self, countdown):
        self.countdown_list.append(countdown)

    def step(self):
        for i in self.countdown_list:
            i.step()


class AgentDataCollector:
    counter = 0

    def __init__(self, name_collector, attributes, name_attrs, update_every=1):
        self.attributes = attributes
        self.name_collector = name_collector
        self.name_attrs = name_attrs
        self.update_every = update_every
        self.all_rows = []

    def update(self, a):
        self.counter += 1
        if self.counter >= self.update_every:
            self.all_rows.append([getattr(a, i) if isinstance(i, str) else i(a) for i in self.attributes])
            self.counter = 0

    def save(self, path):
        if self.all_rows:
            df = pd.DataFrame(self.all_rows)
            df.columns = self.name_attrs
            pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
        self.all_rows = []



class input_gaussian_sensors:
    def __init__(self, input_ranges, num_sensors: List[int], var_factor=1):
        """
        :param input_ranges: e.g. for 2 inputs [[-10, 2], [-50, 50]]
        :param num_sensors: a list, one for each input.
        """
        self.var_factor = var_factor
        self.num_sensors = num_sensors
        self.means = [np.linspace(r[0], r[1], self.num_sensors[obs_idx]) for obs_idx, r in enumerate(input_ranges)]
        range_obs = [mmax - mmin for [mmin, mmax] in input_ranges]

        self.var = [r / self.var_factor for r in range_obs]

    def __call__(self, inputs: List[int]):
        sens_value = []
        for obs_idx, oo in enumerate(inputs):
            for ii in range(self.num_sensors[obs_idx]):
                sens_value.append(np.exp(-1 / 2 * ((oo - self.means[obs_idx][ii]) / self.var[obs_idx]) ** 2))
        return torch.tensor(sens_value)

def convert_input_ranges(inputs: List[int], input_ranges, freq_ranges):
    return [convert_ranges(i, ir, fr) for i, ir, fr in zip(inputs, input_ranges, freq_ranges)]


def poisson2(datum, time, dt, clip=True):
    dist = torch.distributions.Poisson(rate=datum * dt / 1000)
    intervals = dist.sample(sample_shape=torch.Size([time//dt]))
    if clip:
        intervals = torch.clip(intervals, max=1)
    return intervals
def np_poisson2(datum, time, dt, clip=True):
    intervals = np.random.poisson(datum * dt / 1000, size=(time, len(datum)))
    if clip:
        intervals = np.clip(intervals, a_min=None, a_max=1)
    return intervals

class food_foraging_input:
    ## Inputs are energy (from 0 to 1), distance (from 0 to 1), radians (from -1 to 1)
    def __init__(self, rad_num_sensors):
        self.rad_sensors = input_gaussian_sensors([[-1, 1]], num_sensors=[rad_num_sensors], var_factor=4)

    def __call__(self, inputs, agent, dt=1, time_sp=1):
        norm_energy_dist = convert_input_ranges([inputs['energy'], inputs['distance']], [[0, 1], [0, 1]], freq_ranges=[[0, agent.max_energy_freq], [0, agent.max_energy_freq]])
        # i1 = poisson(torch.tensor(norm_en_dist, dtype=torch.float32), time=time_sp, dt=dt)
        i1 = np_poisson2(np.array(norm_energy_dist), time=time_sp, dt=dt)

        norm_rad = self.rad_sensors([inputs['radians']]) * agent.max_radians_freq
        # i2 = poisson(torch.tensor(norm_rad, dtype=torch.float32), time=time_sp, dt=dt)

        i2 = np_poisson2(np.array(norm_rad), time=time_sp, dt=dt)
        # return torch.zeros(1, 12, dtype=torch.uint8), dict(fq_en=0, fq_ds=0, fq_rad=[0]*10)
        # return torch.hstack((i1, i2)), dict(fq_en= norm_en_dist[0], fq_ds=norm_en_dist[1], fq_rad=norm_rad.tolist())
        return np.hstack((i1, i2)), dict(fq_en=norm_energy_dist[0], fq_ds=norm_energy_dist[1], fq_rad=norm_rad.tolist())

class food_foraging_spikes_action:
    def __init__(self, forward_displ_dt, rad_displ_dt):
        self.forward_displ_dt = forward_displ_dt
        self.rad_displ_dt = rad_displ_dt

    def __call__(self, sum_spikes):
        """
        The first spike column indicates the forward movement. Second and third will compete for left/riht direction.
        :param spikes:
        :param forward_displ_dt: how much displacement should we have for each spike in the first column
        :param rad_displ_dt: how much displacement should we have for each spike difference between third and second column
        :return:
         out_forward is the scaling of the forward vector, which can go from 0 to inf
         out_rotation instead is in radians, and should go from -2pi to 2pi (or equivalently from -inf to inf)
        """
        out_forward = sum_spikes[0] * self.forward_displ_dt
        out_rad = np.diff(sum_spikes[1:])[0] * self.rad_displ_dt
        return out_forward, out_rad


class FoodManager(abc.ABC):
    all_food: List[FoodToken] = None
    model: "Environment"

    @abc.abstractmethod
    def __init__(self, model, *args, **kwargs):
        pass

    @abc.abstractmethod
    def step(self) -> List[FoodToken]:
        pass

class FoodFeedback(FoodManager):
    def __init__(self, model, min_food=1, max_food=10000, start_with=100, check_every=100, stable_pop=100, food_energy=0.8, respawn_after=10, k=0.1):
        self.model = model
        self.k = k
        self.min_food = np.clip(min_food, a_min=0, a_max=None)
        self.max_food = max_food
        self.respawn_after = respawn_after
        self.food_energy = food_energy
        self.check_every = check_every
        self.check_save = check_every
        self.stable_pop = stable_pop
        self.all_food = [FoodToken(self.model.get_random_coord(), self.food_energy, self.model, self.respawn_after) for _ in range(start_with)]

    def step(self):
        if self.model.step_count % self.check_every == self.check_every - 1:
            if len(self.model.all_agents) < self.stable_pop:
                for _ in range(int((self.stable_pop - len(self.model.all_agents))*self.k)):
                    if len(self.all_food) > self.max_food:
                        break
                    self.all_food.append(FoodToken(self.model.get_random_coord(), self.food_energy, self.model, self.respawn_after))
                # self.check_every = self.check_save if self.check_every != self.check_save else 200
                # print(f"now we will check every {self.check_every}")

            if len(self.model.all_agents) > self.stable_pop:
                for _ in range(int((len(self.model.all_agents) - self.stable_pop)*self.k)):
                    if len(self.all_food) < self.min_food:
                        break
                    self.all_food.pop(random.randrange(len(self.all_food)))
                # self.check_every = self.check_save if self.check_every != self.check_save else 200
                # print(f"now we will check every {self.check_every}")

class FoodEpochs(FoodManager):

    step_start_epoch = 0

    def __init__(self, epochs: List[Epoch], model, respawn_after=10):
        self.model = model
        self.epochs = deque(epochs)
        self.respawn_after = respawn_after
        self.current_epoch = self.epochs.popleft()
        self.all_food = self.recreate_food_list()

    def recreate_food_list(self) -> List[FoodToken]:
        return [FoodToken(self.model.get_random_coord(), e, self.model, respawn_after=self.respawn_after) for e in self.current_epoch.energy_list]

    def step(self):
        if self.model.step_count - self.step_start_epoch >= self.current_epoch.duration:
            if len(self.epochs) != 0:
                self.model.message += f'step {self.model.step_count}: new epoch'
                self.step_start_epoch = self.model.step_count
                self.current_epoch = self.epochs.popleft()
                self.all_food = self.recreate_food_list()



class Logs():
    value = None

    def __repr__(self):
        return f'{self.value}'

    def __repl__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def __copy__(self):
        return self.value

    def __deepcopy__(self, memodict={}):
        return self.value

    def __eq__(self, other):
        return self.value == other

    def __add__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __radd__(self, other):
        return other + self.value

    def __rsub__(self, other):
        return other - self.value

    def __rfloordiv__(self, other):
        return other // self.value

    def __rtruediv__(self, other):
        return other / self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __floordiv__(self, other):
        return self.value // other

    def __truediv__(self, other):
        return self.value / other

    def __gt__(self, other):
        return self.value > other

    def __lt__(self, other):
        return self.value < other

    def __int__(self):
        return int(self.value)

    def __ge__(self, other):
        return self.value >= other

    def __le__(self, other):
        return self.value <= other

    def __float__(self):
        return float(self.value)

    def __pow__(self, power, modulo=None):
        return self.value ** power

    def __format__(self, format_spec):
        return format(self.value, format_spec)

class ExpMovingAverage(Logs):
    value = None
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def add(self, *args):
        if self.value is None:
            self.value = args[0]
        else:
            self.value = self.alpha * args[0] + (1 -    self.alpha) * self.value
        return self

