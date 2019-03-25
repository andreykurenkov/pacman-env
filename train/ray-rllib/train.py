# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""The entry point for running an agent on an Atari 2600 domain.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune.registry import register_env
from ray.rllib.train import create_parser, run
import pacman_env
from pacman_env.pacmanAgents import LeftTurnAgent, GreedyAgent
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

END_REWARD = 1000
from ray.rllib.models import ModelCatalog, Model

class PacmanModel(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Define the layers of a custom model.

        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs",
                "prev_action", "prev_reward", "is_training".
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Model options.

        Returns:
            (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs]
                and [BATCH_SIZE, desired_feature_size].

        When using dict or tuple observation spaces, you can access
        the nested sub-observation batches here as well:

        Examples:
        """
        weights_initializer = slim.variance_scaling_initializer(
            factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

        obs = input_dict['obs']
        net = tf.cast(obs, tf.float32)
        net = tf.subtract(net, tf.reduce_min(net))
        net = tf.div(net, tf.reduce_max(net))
        net = slim.conv2d(
            net, 16, [4, 4], stride=1, weights_initializer=weights_initializer)
        net = slim.conv2d(
            net, 32, [4, 4], stride=1, weights_initializer=weights_initializer)
        net = slim.conv2d(
            net, 256, [11, 11], stride=1, weights_initializer=weights_initializer)
        net = slim.flatten(net)
        net = slim.fully_connected(
            net, 128, weights_initializer=weights_initializer)
        net_out = slim.fully_connected(
            net,
            num_outputs,
            activation_fn=None,
            weights_initializer=weights_initializer)

        return net_out, net

def create_pacman_environment(layout_name='originalClassic', stick_actions=False):
    """
    """
    layout = pacman_env.layout.getLayout(layout_name)
    if layout is None:
        raise ValueError('No suck layout as %s'%layout_name)
    ghosts = []
    for i in range(2):
        ghosts.append(pacman_env.ghostAgents.RandomGhost(i+1))
    #display = VizGraphics(includeInfoPane=False, zoom=0.4)
    display = pacman_env.matrixDisplay.PacmanGraphics(layout)
    #teacherAgents = [LeftTurnAgent(), GreedyAgent()]
    env = pacman_env.PacmanEnv(layout, ghosts, display)#, teacherAgents=teacherAgents)
    return env

if __name__ == '__main__':
    ModelCatalog.register_custom_model("PacmanModel", PacmanModel)
    register_env("pacman", lambda _: create_pacman_environment())
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
