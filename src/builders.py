import abc
import os
from copy import deepcopy
import sty
import torch
from functools import partial
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
import torch.nn as nn
from typing import Callable
import argparse


class MyNet(Network):
    def add_connection(
            self, connection, source: str, target: str):
        super().add_connection(connection, source, target)
        connection.key = (source, target)


def build_snn(agent, n_input, n_hidden, n_output, update_rule, input_name, output_name, dt=1.0):
    # Build network.
    network = MyNet(dt=dt)
    # network.output_scale = output_scale
    network.update_rule = update_rule
    network.input_name = input_name
    network.output_name = output_name
    # network.num_sensor_units = num_sensor_units
    # network.type_input = type_input
    network.agent = agent

    inpt = Input(n=n_input, traces=True) #.cuda()
    middle1 = LIFNodes(n=n_hidden, refrac=0, traces=True, thresh=-60) #.cuda()
    out = LIFNodes(n=n_output, refrac=0, traces=True, thresh=-60) #.cuda()
    ww = [dict(w=(0.7 + torch.randn(inpt.n, middle1.n) * 0.5)),
          dict(w=(0.7 + torch.randn(middle1.n, out.n) * 0.5))]
    # if w_type == 'zero':
    #     ww = [dict(w=torch.zeros(inpt.n, middle1.n)), dict(w=torch.zeros(middle1.n, out.n))]
    # elif w_type == 'random_unif':
    #     ww = [dict(w=None, wmin=-1, wmax=1), dict(w=None, wmin=-1, wmax=1)]
    # elif isinstance(w_type[0], nn.Parameter):
    #     ww = [dict(w=w_type[0]), dict(w=w_type[1])]
    # elif isinstance(w_type[0], Callable):
    #     ww = [dict(w=w_type[0](inpt.n, middle1.n)), dict(w=w_type[0](middle1.n, out.n))]
    # elif isinstance(w_type[0], torch.Tensor):
    #     ww = [dict(w=w_type[0]), dict(w=w_type[1])]


    inpt_middle = Connection(source=inpt,
                             target=middle1,
                             **ww[0],
                             use_bias=True,
                             update_rule=network.update_rule,
                             layer_index=0,
                             network=network
                             # norm=1
                             )


    middle_out = Connection(
        source=middle1,
        target=out,
        **ww[1],
        use_bias=True,
        update_rule=network.update_rule,
        layer_index=1,
        network=network
        # norm=1
    )

    network.add_layer(inpt, name="I")
    network.add_layer(middle1, name="H1")
    network.add_layer(out, name=network.output_name)
    network.add_connection(inpt_middle, source="I", target="H1")
    network.add_connection(middle_out, source="H1", target="O")
    network.add_monitor(
        Monitor(network.layers[network.output_name], ["s"], time=None),
        network.output_name)
    network.add_monitor(
        Monitor(network.layers['H1'], ["s"], time=None),
        'H1')

    # network.cuda()
    # network.run(inputs={'I': torch.tensor([0.]*12).cuda()}, time=1)
    return network



def build_lr_net(snn_network, n_hidden, init_type=None):
    def init_weights(m, c):
        if isinstance(m, nn.Linear):
            torch.nn.init.constant(m.weight, c)
            m.bias.data.fill_(c)

    lr_net = {}
    for l in snn_network.connections.keys():
        # The + 3 indicates the pre trace, the post trace, and the weight value
        lr_net[l] = nn.Sequential(nn.Linear(snn_network.connections[l].source.shape[-1]*snn_network.connections[l].target.shape[-1] + 3, n_hidden),
                                  nn.ReLU(),
                                  nn.Linear(n_hidden, n_hidden),
                                  nn.ReLU(),
                                  nn.Linear(n_hidden, 1))
        if init_type == 'zero':
            lr_net[l].apply(partial(init_weights, c=0))
        # lr_net[l].cuda()  # the cpu version seems faster!
    return lr_net






