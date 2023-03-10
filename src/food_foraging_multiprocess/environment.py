from typing import List, Callable
from src.food_foraging_multiprocess.proxies import ProxyAgent
from multiprocessing import Process, Pipe
import shutil
import numpy as np
import pathlib
from src.utils import AgentDataCollector
from src.food_foraging_multiprocess.utils import shared_to_update, shared_fixed
import os
from tqdm.auto import tqdm
import sty
from itertools import count
import dill
import random


class Environment():
    selected_agent_idx = 0
    running = True
    message = ''
    socket_message = None
    extinct = False

    def __init__(
            self,
            dt,
            initial_population=100,
            food_manager: Callable = None,
            agent_class=None,
            size=None,
            tot_steps=1000,
            name_run='sim',
            pop=None,  # either a path or an array of pop
            server_model=None,
            reset_on_extinction=False,
            collect_data=False,
            save_state=False,
            seed=0,
            debug_mode=False, # nobody dies, no new agent is born
            **kwargs
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.debug_mode = debug_mode
        self.tot_steps = tot_steps
        # self.manager = Manager()
        self.all_agents: List[ProxyAgent] = []
        # BaseManager.register('dict', dict)
        # self.manager = BaseManager()
        # manager.start()
        # inst = manager.SimpleClass()
        self.save_state = save_state
        self.processes: List[Process] = []
        self.dt = dt
        self.agent_indexer = count(1)
        self.agent_class = agent_class
        print(sty.fg.green + f"Name Run: {name_run}" + sty.rs.fg)
        self.collect_data = collect_data
        if self.collect_data:
            self.collectors = {
                # EvoAgent.die: AgentDataCollector('post_mortem', ['id', 'name_run', 'age'], name_attrs=['id', 'nr', 'age']),
                Environment.__iter__:
                    AgentDataCollector(name_collector='env_state',
                                       attributes=['step_count',
                                        # 'name_run',
                                       # lambda s: len(s.schedule.agents),
                                       lambda s: len([f for f in s.all_food if f.eaten])],
                                       name_attributes=['step',
                                        # 'name_run',
                                        # 'num_agents',
                                            'num_food'],
                                       update_every=25)}

        self.reset_on_extinction = reset_on_extinction
        self.server_model = server_model
        self.step_count = 0
        # self.all_food: List[FoodToken] = []
        # self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
        #                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
        #                           './config-feedforward')
        self.initial_pop = initial_population
        self.world_max_size = np.array(size) if size else np.array([100, 100])
        # self.schedule = MyRandomScheduler(self)
        self.name_run = name_run
        self.saved_model_folder = f'./data/{self.name_run}/save_model/'
        self.saved_pop_folder = f'./data/{self.name_run}/save_pop/'
        self.saved_collectors = f'./data/{self.name_run}/collectors/'

        self.net_svg_folder = f'./data/{self.name_run}/nets_svg/'
        self.food_manager = food_manager(model=self)
        self.pbar = tqdm(total=self.tot_steps, dynamic_ncols=True, colour="yellow")

        shutil.rmtree(self.net_svg_folder) if os.path.exists(self.net_svg_folder) else None
        pathlib.Path(self.saved_model_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.saved_pop_folder).mkdir(parents=True, exist_ok=True)

        pathlib.Path(self.net_svg_folder).mkdir(parents=True, exist_ok=True)
        # self.shr_all_food_pos = self.manager.list([i.pos.tolist() for i in self.all_food])
        self.make_agents(pop)

        ## Here we update the GLOBAL node indexer so that it never overlaps with those already used in the loaded pops (useful when the loaded pop is a mixture of pops coming from different simulations)
        if pop is not None:
            ######### THIS DOESN'T WORK, WE DON'T HAVE A SCHEDULE ANYMORE
            l = [list(a.genome.nodes.keys()) for a in self.all_agents]
            flat_l = [e for l in l for e in l]  # flatten list of lists... not sure htf it does that.
            # self.config.genome_config.node_indexer = count(max(flat_l) +1)

        self.running = True
        # self.save_model_state()
        print(f"RUN: {self.name_run}")

    def make_agents(self, pop=None):
        ######### THIS DOESN'T WORK, WE DON'T HAVE A SCHEDULE ANYMORE

        if pop is not None:
            if isinstance(pop, str):
                pop = dill.load(open(pop, 'rb'))
            self.agent_indexer = pop[0].model.agent_indexer
            for a in pop:
                self.schedule.add(a)
                a.model = self
        else:
            for i in range(self.initial_pop):
                self.spawn_new_agent()
        self.selected_agent_idx = 0

    def save_model_state(self):
        if self.save_state:
            collectors_cmd = None
            if self.collect_data:
                [c.save(self.saved_collectors + f'/{c.name_collector}/step{self.step_count}.csv') for k, c in self.collectors.items()]
                collectors_cmd = self.collectors
                self.collectors = None
            server_cmd = self.server_model  # this object can't be pickled ..
            self.server_model = None
            # pbar_cmd = self.pbar
            self.pbar = None

            dill.dump(self, open(self.saved_model_folder + f'/step{self.step_count}.pickle', 'wb'))
            dill.dump([i for i in self.schedule.agents], open(self.saved_pop_folder + f'/step{self.step_count}.pickle', 'wb'))
            self.message += 'saved in: ' + self.saved_model_folder + f'/step{self.step_count}.pickle\n'
            self.server_model = server_cmd
            # self.pbar = pbar_cmd
            self.collectors = collectors_cmd

    def stop(self):
        self.server_model.event_loop.stop() if self.server_model is not None else None
        self.server_model = None
        tqdm._instances.pop().close()

        # print("Epochs finished!")
        # self.message += 'epochs finished\n'
        self.running = False
        self.save_model_state()
        return "Finished"

    def __iter__(self):
        while True:
            self.collectors[Environment.step].update(self) if self.collect_data else None
            self.pbar.set_description("Steps")
            self.pbar.set_postfix_str(f"n. pop {len(self.all_agents)}; n. food {len(self.food_manager.all_food)}")
            self.pbar.update(1)
            print(sty.rs.fg, end="")
            # if self.step_count % 50 == 0:
            #     print(f"Step: {self.step_count}, n pop: {len(self.schedule.agents)}")
            self.step_count += 1
            self.message = ''

            if self.step_count % 1000 == 0:
                self.save_model_state()

            if self.step_count > self.tot_steps:
                return self.stop()
            # if self.step_count - self.step_start_epoch >= self.current_epoch.duration:
                # if len(self.all_epochs) == 0:
                #     return self.stop()
                # self.message += f'step {self.step_count}: new epoch'
                # self.step_start_epoch = self.step_count
                # self.current_epoch = self.all_epochs.popleft()
                # print(self.message)
                # self.spawn_food()
            self.food_manager.step()
            if not self.all_dead():
                self.food_dist_matrix = np.linalg.norm(np.array([[i.pos - j.pos for i in self.food_manager.all_food] for j in self.all_agents]), axis=2)

            all_food_pos = [f.pos for f in self.food_manager.all_food]
            for idx, pa in enumerate(self.all_agents):
                pa.connection.send((self.food_dist_matrix[idx], all_food_pos))

            # for id, p in self.pipes.items():
            #     received = p.recv()
            #     print(f"Agent {id} sent: {received}")
            for i, pa in enumerate(self.all_agents.copy()):
                message = pa.connection.recv()
                if message.dead and not self.debug_mode:
                    # if self.collect_data:
                    #     self.collectors[ForagingAgent.die].update(self) if ForagingAgent.die in self.collectors else None
                    # print(f"{pa.id} TERMINATED")
                    pa.process.kill()
                    self.all_agents.remove(pa)
                else:
                    [pa.__setattr__(k, v) for k, v in message.info_to_pass_around.items()]

                    for idx in message.index_food_eaten:
                        self.food_manager.all_food[idx].get_eaten()

                    if message.reproduce and not self.debug_mode:
                        new_proxy_agent = self.spawn_new_agent(parent=pa)
                        pa.children_ids.append(new_proxy_agent.id)
                        new_proxy_agent.parent_id = pa.id

            for f in self.food_manager.all_food:
                f.step()

            if self.all_dead():
                if self.reset_on_extinction:
                    self.pbar.disable = True
                    print("Extinctiong. Restarting") if not self.extinct else None
                    self.extinct = True
                    self.socket_message = "reset"
                    tqdm._instances.pop().close()
                    yield "Extinction"
                else:
                    self.server_model.event_loop.stop() if self.server_model is not None else None
            yield "Step"

        #
        # self.children_id.append(new_agent.id)
        # self.countdown_offspring.start()
        # self.num_children += 1

    #

    # def create_new_genome(self, genome_type, genome_config):
    #     g = genome_type(self.global_id)
    #     g.configure_new(genome_config)
    #     return g

    def get_random_coord(self):
        x = np.random.random() * self.world_max_size[0]
        y = np.random.rand() * self.world_max_size[1]
        return np.array([x, y])

    def all_dead(self):
        return len(self.all_agents) == 0

    def spawn_new_agent(self, pos=None, direction=None, parent=None):
        if pos is None:
            pos = self.get_random_coord()
        direction = np.random.uniform(0, 2 * np.pi) if direction is None else direction
        agent = self.agent_class(
            id=next(self.agent_indexer),
            world_max_size=self.world_max_size,
            pos=pos,
            direction=direction,
            parent=parent,
        )

        local_end, process_end = Pipe()

        proxy_agent = ProxyAgent(connection=local_end, process=Process(target=process_agent, args=(agent, process_end)))

        [proxy_agent.__setattr__(i, agent.__getattribute__(i)) for i in shared_fixed + shared_to_update]
        [proxy_agent.__setattr__(i, agent.__getattribute__(i)) for i in [i.name for i in agent.evol_pars]]
        self.all_agents.append(proxy_agent)
        proxy_agent.process.start()
        return proxy_agent


import torch
def process_agent(agent, conn):

    agent.pipe = conn
    while True:
        # food_dist = agent.input_queue.get()
        # print("in process agent")

        food_dist, all_food_pos = conn.recv()
        agent.step(food_dist, all_food_pos)