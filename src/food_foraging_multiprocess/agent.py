import random
import torch
import numpy as np
import string
from src.utils import Countdown, CountdownList
from src.agent import AgentSNN
from src.food_foraging_multiprocess.utils import Message


class ForagingAgentInProcess(AgentSNN):
    fov = np.pi / 1.5
    energy_depletion = 0.008
    radius = 5
    max_vision_dist = 100
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
        world_max_size,
        # model,
        pos,
        direction,
        parent: 'ForagingAgentInProcess' = None,
        collectors=None,
        name_run='',
        **kwargs
    ):
        super().__init__(parent=parent, **kwargs)
        self.sum_middle_spikes = []
        self.countdown_offspring_counter = np.inf
        self.world_max_size = world_max_size
        self.name_run = name_run
        self.input_spikes = []
        # self.model = model
        self.id = id
        # self.infoto_pass = info_to_pass
        if parent is not None:
            self.parent_id = parent.id
            self.surname = parent.surname
            self.generation = parent.generation + 1
        else:
            self.generation = 0
            self.parent_id = []
            self.surname = ''.join(random.choices(string.ascii_lowercase, k=5))

        self.name = ''.join(random.choices(string.ascii_lowercase, k=5))
        self.inputs = []
        self.outputs = []
        self.children_id = []
        self.output_spikes = []
        self.pos = pos
        self.age = np.random.randint(0, 100)
        self.sum_input_spikes = []
        self.sum_output_spikes = []
        self.direction = direction
        # self.genome = genome
        # self.net = neat.nn.FeedForwardNetwork.create(genome, self.model.config)
        self.countdown_offspring = Countdown(self.time_between_children, randomness=25)
        self.countdown_list = CountdownList([self.countdown_offspring])
        self.energy = 1  # np.random.uniform(0.75, 1)
        # self.network_svg_path = self.model.data_folder + f'/nets/net_{self.id}'
        self.network_drawn = False

        self.die_callbacks = []

    def compute_vision(self, food_dist, all_food_pos):
            close_food_idx = np.where(food_dist< self.max_vision_dist)[0]


            # close_food = [self.model.all_food[i] for i in np.where(food_dist < self.max_vision_dist)[0]]
            if len(close_food_idx):
                ag_view = np.array([np.cos(self.direction), np.sin(self.direction)])
                all_rads = [ np.arccos(np.dot(ag_view, (all_food_pos[i] - self.pos)) / (np.linalg.norm(ag_view) * food_dist[i])) for i in close_food_idx]

                in_range_idx = [(all_rads[idx], i) for idx, i in enumerate(close_food_idx) if all_rads[idx] < self.fov/2 ]
                if in_range_idx:
                    rad, selected_food_idx = in_range_idx[np.argmin([food_dist[i[1]] for i in in_range_idx])]
                    s = np.sign(np.cross(ag_view, (all_food_pos[selected_food_idx] - self.pos)))
                    rad *= s
                    return food_dist[selected_food_idx], rad, selected_food_idx

            return self.max_vision_dist, 0, -1



    def step(self, food_dist, all_food_pos):
        torch.set_num_threads(1)
        this_message = Message()

        dist, rad, self.target_food = self.compute_vision(food_dist, all_food_pos)
        d_input = (self.max_vision_dist - dist)/self.max_vision_dist
        rad_input = rad / (self.fov / 2)
        dict_inputs = dict(energy=self.energy, distance=d_input, radians=rad_input)
        self.inputs = [self.energy, d_input, rad_input]

        input_spikes, freq_input = self.input_to_spikes(dict_inputs,
                                                             self,
                                                             dt=1,  # ****
                                                             time_sp=1) # ****
                                                       # dt=self.model.dt,
                                                       # time_sp=self.model.dt)

        inputs = {
            self.network.input_name: torch.tensor(input_spikes)  # .cuda()
        }
        self.sum_input_spikes = np.sum(input_spikes, 0)
        with torch.no_grad():
            self.network.run(inputs=inputs,
                             time=1)  # ***
                             # time=self.model.dt)
        output_spikes = self.network.monitors[self.network.output_name].get("s").float().numpy()
        self.sum_middle_spikes = np.sum(self.network.monitors['H1'].get("s").float().numpy(), 0)[0]
        # self.sum_middle_spikes = None
        self.sum_output_spikes = np.sum(output_spikes, 0)[0]

        out_forward, out_rotation = self.output_spikes_to_action(self.sum_output_spikes)

        self.outputs = [out_forward, out_rotation]
        self.pos = np.array([self.pos[0] + np.cos(self.direction) * out_forward, self.pos[1] + np.sin(self.direction) * out_forward])
        self.direction += out_rotation


        ########## EAT FOOD ################
        this_message.index_food_eaten = [i for i in np.where(food_dist < self.radius * 2)[0]]
        ## ToDo: Pass Food Energy! (see comments below)
        # if this_message.index_food_eaten !=[]:
            # print(f"{self.id} ate food")
        # self.energy += [np.clip(self.energy + food.energy, 0, 1) for i in len(indexes_eatend_food)]
        self.energy += np.sum([np.clip(0.8, 0, 1) for _ in this_message.index_food_eaten])

        if self.age > self.fertile_age_start and not self.fertile:
            self.fertile = True
            self.countdown_offspring.start()

        if self.age > self.fertile_age_start and (self.countdown_offspring.counter == 0 or self.countdown_offspring.counter == np.inf):
            this_message.reproduce = True
            # this_message.evol_pars_dict = {ev.name: self.__getattribute__(ev.name) for ev in self.evol_pars}
            # this_message.original_snn_state_dict = self.original_snn_state_dict
            # this_message.parent_info = {info: self.__getattribute__(info) for info in share_info_proxy_parents}
            self.countdown_offspring.start()
            self.num_children += 1
            # self.reproduce_asexual()

        ########## MOVE WITH MAX SIZE #############
        if (self.pos[0] > self.world_max_size[0] - self.radius) or (self.pos[1] > self.world_max_size[1] - self.radius) or (self.pos[0] < 0 + self.radius) or (self.pos[1] < 0 + self.radius):
            self.pos = np.array([np.clip(self.pos[0], 0 + self.radius, self.world_max_size[0] - self.radius), np.clip(self.pos[1], 0 + self.radius, self.world_max_size[1] - self.radius)])

        self.energy = np.max((0, self.energy - self.energy_depletion))
        self.age += 1

        if self.age > self.max_age or self.energy <= 0:
            this_message.dead = True
            # print(f'{self.id} {self.name} {self.energy:.2f}')
            stop = 1
            # self.die()
        self.countdown_list.step()
        ## it's just easier to pass around this property
        self.countdown_offspring_counter = self.countdown_offspring.counter
        this_message.info_to_pass_around = {k: self.__getattribute__(k) for k in this_message.info_to_pass_around.keys()}
        # this_message.pos = self.pos
        # this_message.energy = self.energy
        self.pipe.send(this_message)


    # def get_nested_attributes(self, attr):
    #     at = attr.split('.')
    #     obj = self
    #     for i in at:
    #         obj = getattr(obj, i)
    #     return obj

    # def reproduce_asexual(self):
    #     ### reproduce agent info
    #     new_agent = self.model.spawn_new_agent(parent=self)
    #     if self.evolve_substrate:
    #         new_agent.network.load_state_dict(self.original_snn_state_dict)
    #         for k, v in new_agent.network.connections.items():
    #             v.w += torch.normal(0, 0.1, v.w.shape)
    #             if v.b is not None:
    #                 v.b += torch.normal(0, 0.1, v.b.shape)
    #     # This is done so that the "learnt" net won't be inherited (which would be Lamarckian), only the net the agent is born with.
    #     new_agent.original_snn_state_dict = deepcopy(new_agent.network.state_dict())
    #     #
    #     # Mutate lr_net
    #     if self.lr_net_builder:
    #         for net in new_agent.lr_net.values():
    #             for k, v in net.state_dict().items():
    #                 v += torch.normal(0, 0.05, v.shape)
    #                 # stop = 1
    #     #
    #     # Mutate other properties
    #     # maybe randomly shuffle before mutating
    #     for par in self.evol_pars:
    #         if hasattr(self, par.name):
    #             new_agent.__setattr__(par.name, np.max((np.min((self.__getattribute__(par.name) + (np.random.randn(1)[0] * par.mutation_var), par.max_value)), par.min_value)))
    #         else:
    #             assert False, f"{par.name} not found in agent!"
    #
    #     #
    #     #
    #     self.children_id.append(new_agent.id)
    #     self.countdown_offspring.start()
    #     self.num_children += 1
#


