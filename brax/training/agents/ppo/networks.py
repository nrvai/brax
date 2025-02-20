# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO networks."""

from typing import Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax.numpy as jp


@flax.struct.dataclass
class PPONetworks:
  policy_network: networks.FeedForwardNetwork
  value_network: networks.FeedForwardNetwork
  encoder_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(ppo_networks: PPONetworks):
  """Creates params and inference function for the PPO agent."""

  def make_policy(
      policy_params: types.Params,
      enc_params: types.Params,
      deterministic: bool = False
  ) -> types.Policy:
    policy_network = ppo_networks.policy_network
    encoder_network = ppo_networks.encoder_network
    parametric_action_distribution = ppo_networks.parametric_action_distribution

    def policy(
        observations: types.Observation, priv: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      encoding = encoder_network.apply(*enc_params, priv)

      pinput = jp.concatenate([observations, encoding], axis=-1)
      logits = policy_network.apply(*policy_params, pinput)

      if deterministic:
        return ppo_networks.parametric_action_distribution.mode(logits), {}
      raw_actions = parametric_action_distribution.sample_no_postprocessing(
          logits, key_sample
      )
      log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
      postprocessed_actions = parametric_action_distribution.postprocess(
          raw_actions
      )
      return postprocessed_actions, {
          'log_prob': log_prob,
          'raw_action': raw_actions,
          'encoding': pinput
      }

    return policy

  return make_policy


def make_ppo_networks(
    observation_size: types.ObservationSize,
    privileged_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    encoder_hidden_layer_sizes: Sequence[int] = (512, 512),
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
    encoder_obs_key: str = 'state',
) -> PPONetworks:
  """Make PPO networks with preprocessor."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size
  )
  encoder_size = 128
  full_obs_size = observation_size + encoder_size
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
      full_obs_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      obs_key=policy_obs_key,
  )
  value_network = networks.make_value_network(
      full_obs_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      obs_key=value_obs_key,
  )
  encoder_network = networks.make_policy_network(
      encoder_size,
      privileged_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=encoder_hidden_layer_sizes,
      activation=activation,
      obs_key=encoder_obs_key,
  )

  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      encoder_network=encoder_network,
      parametric_action_distribution=parametric_action_distribution,
  )