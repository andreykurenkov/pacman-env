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

import pacman_env
from pacman_env.pacmanAgents import LeftTurnAgent, GreedyAgent

from absl import app
from absl import flags
from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.atari import run_experiment
from dopamine.atari.train import FLAGS
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

END_REWARD = 1000

class PacmanRainbowAgent(rainbow_agent.RainbowAgent):

  def _network_template(self, state):
    """Builds a convolutional network that outputs Q-value distributions.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    weights_initializer = slim.variance_scaling_initializer(
        factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

    net = tf.cast(state, tf.float32)
    net = tf.subtract(net, tf.reduce_min(net))
    net = tf.div(net, tf.reduce_max(net))
    net = slim.conv2d(
        net, 32, [8, 8], stride=2, weights_initializer=weights_initializer)
    net = slim.conv2d(
        net, 64, [4, 4], stride=1, weights_initializer=weights_initializer)
    net = slim.conv2d(
        net, 64, [3, 3], stride=1, weights_initializer=weights_initializer)
    net = slim.flatten(net)
    net = slim.fully_connected(
        net, 128, weights_initializer=weights_initializer)
    net = slim.fully_connected(
        net,
        self.num_actions * self._num_atoms,
        activation_fn=None,
        weights_initializer=weights_initializer)

    logits = tf.reshape(net, [-1, self.num_actions, self._num_atoms])
    probabilities = tf.contrib.layers.softmax(logits)
    q_values = tf.reduce_sum(self._support * probabilities, axis=2)
    return self._get_network_type()(q_values, logits, probabilities)

  def end_episode(self, reward):
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
    """
    if not self.eval_mode:
      self._store_transition(self._observation, self.action, reward*END_REWARD, True)

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
  env = pacman_env.PacmanEnv(layout, ghosts, display)
  return env

def create_agent(sess, environment, summary_writer=None):
  """Creates the agent.

  Args:
    sess: A `tf.Session` object for running associated ops.
    environment: An Atari 2600 Gym environment.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  if not FLAGS.debug_mode:
    summary_writer = None
  if FLAGS.agent_name == 'dqn':
    return dqn_agent.DQNAgent(sess, 
                              num_actions=environment.action_space.n,
                              observation_shape=environment.observation_space.shape,
                              summary_writer=summary_writer)
  elif FLAGS.agent_name == 'rainbow':
    #return rainbow_agent.RainbowAgent(sess, 
    agent = PacmanRainbowAgent(sess, 
                              num_actions=environment.action_space.n,
                              observation_shape=environment.observation_space.shape,
                              summary_writer=summary_writer)
    agent.env = environment
    return agent
  elif FLAGS.agent_name == 'implicit_quantile':
    return implicit_quantile_agent.ImplicitQuantileAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(FLAGS.agent_name))

def create_runner(base_dir, create_agent_fn):
  """Creates an experiment Runner.

  Args:
    base_dir: str, base directory for hosting all subdirectories.
    create_agent_fn: A function that takes as args a Tensorflow session and an
     Atari 2600 Gym environment, and returns an agent.

  Returns:
    runner: A `run_experiment.Runner` like object.

  Raises:
    ValueError: When an unknown schedule is encountered.
  """
  assert base_dir is not None
  # Continuously runs training and evaluation until max num_iterations is hit.
  if FLAGS.schedule == 'continuous_train_and_eval':
    return run_experiment.Runner(base_dir, create_agent_fn, create_pacman_environment)
  # Continuously runs training until max num_iterations is hit.
  elif FLAGS.schedule == 'continuous_train':
    return run_experiment.TrainRunner(base_dir, create_agent_fn, create_pacman_environment)
  else:
    raise ValueError('Unknown schedule: {}'.format(FLAGS.schedule))


def launch_experiment(create_runner_fn, create_agent_fn):
  """Launches the experiment.

  Args:
    create_runner_fn: A function that takes as args a base directory and a
      function for creating an agent and returns a `Runner`-like object.
    create_agent_fn: A function that takes as args a Tensorflow session and an  environment, and returns an agent.
  """
  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  runner = create_runner_fn(FLAGS.base_dir, create_agent_fn)
  runner.run_experiment()


def main(unused_argv):
  """Main method.

  Args:
    unused_argv: Arguments (unused).
  """
  tf.logging.set_verbosity(tf.logging.INFO)
  launch_experiment(create_runner, create_agent)


if __name__ == '__main__':
  flags.mark_flag_as_required('agent_name')
  flags.mark_flag_as_required('base_dir')
  app.run(main)
