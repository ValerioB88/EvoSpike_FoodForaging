import torch
import numpy as np
import string
from src.utils import Countdown, CountdownList, EvolPar
from src.food_foraging_singlecore.environment import Environment
from src.agent import AgentSNN
import random

class ForagingAgent(AgentSNN):
    age = 0
    fov = np.pi / 1.5
    max_vision_dist = 100
    energy_depletion = 0.008
    radius = 5
    max_speed_forward = 5  # in px/time steps
    max_speed_rotation = 5  # in dg/time steps
    max_age = 500
    time_between_children = 100
    fertile_age_start = 90
    num_children = 0
    fertile = False
    target_food = -1

    def reset(self):
        self.network.load_state_dict(self.original_snn_state_dict)

    def __init__(
            self,
            id,
            model,
            pos,
            direction,
            parent: 'ForagingAgent' = None,
            collectors=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.evol_pars_dict = None
        self.model: Environment = model
        self.id = id
        if parent is not None:
            self.parent_id = parent.id
            self.surname = parent.surname
            self.generation = parent.generation + 1
            self.name_run = parent.name_run
        else:
            self.generation = 0
            self.parent_id = []
            self.name_run = self.model.name_run
            self.surname = ''.join(random.choices(string.ascii_lowercase, k=5))

        self.name = ''.join(random.choices(string.ascii_lowercase, k=5))
        self.inputs = []
        self.outputs = []
        self.children_ids = []
        self.sum_spikes = []
        self.pos = pos
        self.direction = direction
        # self.genome = genome
        # self.net = neat.nn.FeedForwardNetwork.create(genome, self.model.config)
        self.countdown_offspring = Countdown(self.time_between_children, randomness=25)
        self.countdown_list = CountdownList([self.countdown_offspring])
        self.energy = 1 # np.random.uniform(0.75, 1)
        # self.network_svg_path = self.model.data_folder + f'/nets/net_{self.id}'
        self.network_drawn = False

        self.die_callbacks = []
    def compute_vision(self, pop_idx):
        try:
            close_food_idx = np.where(self.model.food_dist_matrix[pop_idx] < self.max_vision_dist)[0]
        except IndexError:
            stop=1

        # close_food = [self.model.all_food[i] for i in np.where(self.model.food_dist_matrix[pop_idx] < self.max_vision_dist)[0]]
        if len(close_food_idx):
            ag_view = np.array([np.cos(self.direction), np.sin(self.direction)])
            all_rads = [ np.arccos(np.dot(ag_view, (self.model.all_food[i].pos - self.pos)) / (np.linalg.norm(ag_view) * self.model.food_dist_matrix[pop_idx][i])) for i in close_food_idx]

            in_range_idx = [(all_rads[idx], i) for idx, i in enumerate(close_food_idx) if all_rads[idx] < self.fov/2]
            if in_range_idx:
                rad, selected_food_idx = in_range_idx[np.argmin([self.model.food_dist_matrix[pop_idx][i[1]] for i in in_range_idx])]
                s = np.sign(np.cross(ag_view, (self.model.all_food[selected_food_idx].pos - self.pos)))
                rad *= s
                return self.model.food_dist_matrix[pop_idx][selected_food_idx], rad, selected_food_idx

        return self.max_vision_dist, 0, -1

    def step(self, pop_idx):
        dist, rad, self.target_food = self.compute_vision(pop_idx)
        d_input = (self.max_vision_dist - dist)/self.max_vision_dist
        rad_input = rad / (self.fov / 2)
        dict_inputs = dict(energy=self.energy, distance=d_input, radians=rad_input)
        self.inputs = [self.energy, d_input, rad_input]
        spike_input, freq_input = self.input_to_spikes(dict_inputs, self, dt=self.model.dt,  time_sp=self.model.dt)

        inputs = {
            self.network.input_name: torch.tensor(spike_input) # .cuda()
        }
        # self.network.cuda()
        with torch.no_grad():
            self.network.run(inputs=inputs, time=self.model.dt)
        all_spikes = self.network.monitors[self.network.output_name].get("s").float()
        self.sum_spikes = np.sum(all_spikes.numpy(), 0)[0]

        out_forward, out_rotation = self.output_spikes_to_action(self.sum_spikes)

        self.outputs = [out_forward, out_rotation]
        self.pos = np.array([self.pos[0] + np.cos(self.direction) * out_forward, self.pos[1] + np.sin(self.direction) * out_forward])
        self.direction += out_rotation

        [self.eat_food(self.model.all_food[i]) for i in np.where(self.model.food_dist_matrix[pop_idx] < self.radius * 2)[0]]

        if self.age > self.fertile_age_start and not self.fertile:
            self.fertile = True
            self.countdown_offspring.start()

        if self.age > self.fertile_age_start and (self.countdown_offspring.counter == 0 or self.countdown_offspring.counter == np.inf):
            self.evol_pars_dict = {ev.name: self.__getattribute__(ev.name) for ev in self.evol_pars}
            new_agent = self.model.prepare_new_agent(parent=self)
            self.model.all_agents.append(new_agent)
            self.children_ids.append(new_agent.id)
            self.countdown_offspring.start()
            self.num_children += 1
            # self.reproduce_asexual()

        if (self.pos[0] > self.model.world_max_size[0] - self.radius) or (self.pos[1] > self.model.world_max_size[1] - self.radius) or (self.pos[0] < 0 + self.radius) or (self.pos[1] < 0 + self.radius):
            self.pos = np.array([np.clip(self.pos[0], 0 + self.radius, self.model.world_max_size[0] - self.radius), np.clip(self.pos[1], 0 + self.radius, self.model.world_max_size[1] - self.radius)])

        self.energy -= self.energy_depletion
        self.age += 1

        if self.age > self.max_age or self.energy < 0:
            self.die()
        self.countdown_list.step()



    def eat_food(self, food):
        self.energy = np.clip(self.energy + food.energy, 0, 1)
        food.get_eaten()


    def die(self):
        if self.model.collect_data:
            self.model.collectors[ForagingAgent.die].update(self) if ForagingAgent.die in self.model.collectors else None

        # if len(self.model.all_agents) > 0:
        #     if self.id == self.model.selected_agent_idx:
        self.model.selected_agent_idx = np.clip(self.model.selected_agent_idx, 0, len(self.model.all_agents)-1)

                # self.model.selected_agent_idx = self.model.all_agents[np.argmin(np.abs([i.id - self.model.selected_agent_idx for i in self.model.all_agents]))].id

        self.model.all_agents.remove(self)



    def find_partner(self):
        pass
        # find reproductive parnet

