from src.food_foraging_singlecore.agent import ForagingAgent
from src.utils import FoodToken
from browser_dashboard.modules.canvas import Canvas
from src.food_foraging_singlecore.environment import Environment


def object_draw(pop_idx, type="agent", obj: ForagingAgent = None, is_selected=False):
    if type == "agent":
        main_dict = {"type": "agent",
                     "NAME": f'{obj.name} {obj.surname}',
                     "pop_idx": pop_idx,
                     "r": obj.radius,
                     "generation": obj.generation,
                     "parent_id": obj.parent_id,
                     "children_id": obj.children_ids,
                     "color_hsl": f"hsl({obj.color}, 100%, 50%)",
                     "fov": obj.fov,
                     "mvisd": obj.max_vision_dist,
                     "energy": f'{obj.energy:.3f}',
                     "id": obj.id,
                     "dir": obj.direction,
                     "age": obj.age,
                     "max_age": obj.max_age,
                     "fertile_age_start": obj.fertile_age_start,
                     "time_between_children": obj.time_between_children,
                     "countdown_offspring": f"{obj.countdown_offspring_counter:.1f}",
                     "num_children": obj.num_children,
                     # "skin_c": str(obj.color_hsl),
                     "name_sim": obj.name_run,
                     "evol_pars":  {ep.name: obj.__getattribute__(ep.name) for ep in obj.evol_pars}
                     }
        return main_dict
    if type == "food":
        return {"type": "good_food", "r": FoodToken.radius, "color_hsl": "Green" if obj.energy > 0 else "Red", "selected": is_selected}
    # if type == "bad_food":
    #     return {"type": "bad_food", "r": FoodToken.radius, "color": "Red"}


class ForagingCanvas(Canvas):
    local_includes = ["src/browser_utils/ForagingCanvas.js"]
    portrayal_method = None

    def __init__(self, canvas_height=500, canvas_width=500, name_sim='', additional_info=''):
        self.portrayal_method = object_draw
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        new_element = f"new ForagingCanvas({self.canvas_width}, {self.canvas_height}, '{name_sim}', '{additional_info}')"
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model: Environment):
        space_state = []
        food_aimed = -1
        for idx, obj in enumerate(model.all_agents):
            sel = False
            portrayal = self.portrayal_method(idx, "agent", obj, is_selected=sel)
            portrayal["x"], portrayal["y"] = obj.pos

            if model.selected_agent_idx == idx:
                food_aimed = obj.target_food
                # if not obj.network_drawn:
                #     draw_net(model.config, obj.genome, view=False, filename=model.net_svg_folder + f'id{obj.id}')
                #     obj.network_drawn = True
                portrayal['selected'] = True
                # Add here all those info that it's computationally expensive to add for all agents
                portrayal['children_pos'] = [list(model.all_agents[i].pos) for i in obj.children_ids if i in model.all_agents]
                portrayal['pos'] = str(obj.pos.astype(int))
                portrayal['print_info'] = ['NAME', 'pop_idx', 'generation', 'id',  'age', 'parent_id', 'num_children', 'children_id', 'energy', 'countdown_offspring', 'pos', 'name_sim']


            space_state.append(portrayal)

        for idx, obj in enumerate(model.food_manager.all_food):
            portrayal = self.portrayal_method(idx, "food", obj, is_selected=True if food_aimed == idx else False)
            portrayal["x"], portrayal["y"] = obj.pos
            portrayal["print_info"], portrayal["y"] = obj.pos

            space_state.append(portrayal)

        info = {}
        info["pop_count"] = len(model.all_agents)
        info["food_count"] = len(model.food_manager.all_food)
        info["entities"] = space_state
        info["message"] = model.message
        info["step"] = model.step_count
        info["global_commands"] = model.socket_message

        return info
