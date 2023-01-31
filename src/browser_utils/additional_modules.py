from browser_dashboard.modules.line_plot import LinePlot
from browser_dashboard.server import CustomAPI

import numpy as np
class ModelLinePlot():
    def index_render(self, values, model):
        return [[{'x': model.step_count, 'y': v} for idx, v in enumerate(dv)] for dv in  values]


class PlotInput(ModelLinePlot, LinePlot):
    def render(self, model):
        if not model.all_dead():
            values = self.reformat_values(model.all_agents[model.selected_agent_idx].inputs)
            return self.index_render(values, model)
        else:
            return []

class PlotOutput(ModelLinePlot, LinePlot):
    def render(self, model):
        if not model.all_dead():
            values = self.reformat_values(model.all_agents[model.selected_agent_idx].outputs)
            return self.index_render(values, model)
        else:
            return []

class PlotOutputSumSpikes(ModelLinePlot, LinePlot):
    def render(self, model):
        if not model.all_dead():
            values = self.reformat_values(model.all_agents[model.selected_agent_idx].sum_output_spikes)
            return self.index_render(values, model)
        else:
            return []

import numpy as np
class SpikePlot(LinePlot):
    def __init__(self, add_random_noise=False, names=None, **kwargs):
        self.spike_names = names
        self.add_random_noise = add_random_noise
        super().__init__(names=names, show_lines=['false'] * len(names), **kwargs)

    def render(self, model):
        if not model.all_dead():
            spikes = [model.all_agents[model.selected_agent_idx].__dict__[s] for s in self.spike_names]
            if len(spikes) > 0:
                return [[{'x': model.step_count + (np.random.uniform(-0.5, 0.5) if self.add_random_noise else 0), 'y': float(y)} for y in np.where(ss)[0]] if len(ss) > 0 else [] for ss  in spikes]

class MyCustomAPI(CustomAPI):
    def other_message(self, websocket, msg):
        if msg["type"] == 'command':
            m = self.model
            agents = m.all_agents

            if msg["command"] == "next":
                m.selected_agent_idx = 0 if m.selected_agent_idx + 1 > len(agents) - 1 else m.selected_agent_idx + 1
            if msg["command"] == "previous":
                # idx = find_pop_idx_from_id()
                m.selected_agent_idx = len(agents) - 1 if m.selected_agent_idx + -1 < 0 else m.selected_agent_idx - 1
            if msg["command"] == "kill":
                # idx = find_pop_idx_from_id()
                agents[m.selected_agent_idx].die()
            if msg["command"] == "offspring":
                # idx = find_pop_idx_from_id()
                idx = m.selected_agent_idx
                print("NOT IMPLEMENTED")
                # agents[idx].mutate()
            if msg["command"] == "save":
                self.model.save_model_state()


def find_pop_idx_from_id(model, id):
    for idx, a in enumerate(model.all_agents):
        if a.id == id:
            break
    return idx