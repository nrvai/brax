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

import functools
from typing import Callable, Union

import numpy as np

import flax
import jax
import jax.numpy as jnp
import optax
from brax import envs
from brax.base import State, System
from brax.envs import Env, Wrapper
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper, VmapWrapper
from brax.training import gradients, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.v1 import envs as envs_v1

from brax.training.acting import Evaluator, generate_unroll


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    enc_normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


class HeightFieldWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.hf_data = self.env.unwrapped.sys.hfield_data * 1.0

    def env_fn(self, sys: System) -> Env:
        env = self.env
        env.unwrapped.sys = sys
        return env

    def step(self, state: State, action: jax.Array, *args) -> State:
        def update_hf_scale():
            hf_max_scale = jax.lax.cond(state.info["curriculum"] == 1, lambda: 2, lambda: 0)
            hf_max_scale = jax.lax.cond(state.info["curriculum"] == 2, lambda: 4, lambda: hf_max_scale)
            hf_max_scale = jax.lax.cond(state.info["curriculum"] == 3, lambda: 6, lambda: hf_max_scale)
            hf_max_scale = jax.lax.cond(state.info["curriculum"] > 3, lambda: 10, lambda: hf_max_scale)
            return jax.random.randint(sub_rng, (1,), 0, hf_max_scale + 1)[0] / 10.0

        rng, sub_rng = jax.random.split(state.info["rng"], 2)
        hf_scale = jax.lax.cond(
            state.info["step"] % 200 == 0,
            update_hf_scale,
            lambda: state.info["hf_scale"]
        )

        hf_data = self.hf_data * hf_scale
        sys = self.env.unwrapped.sys
        new_sys = sys.replace(hfield_data=hf_data)
        new_env = self.env_fn(new_sys)

        state.info["rng"] = rng
        state.info["hf_scale"] = hf_scale
        state.info["hf_data"] = hf_data

        return new_env.step(state, action, *args)


class StairsWrapper(Wrapper):
    def __init__(self, env: Env, key: jax.Array):
        super().__init__(env)
        self.key = key
        self.stair_configs = jnp.array([
            (2, 6, 0.4, 5),
            (2, 32, 0.4, 5),
            (8, 6, 0.5, 5),
            (8, 6, 1.0, 5)
        ])

    def get_pyramid_stairs(self, num_cycles=4, num_stairs=5, step_height=1.5, flat_area=5):
        nrows, ncols = 256, 256
        mid_point = nrows // 2
        hf_data = np.zeros((nrows, ncols))

        for i in range(nrows):
            for j in range(ncols):
                dist_x = max(0, abs(i - mid_point) - flat_area)
                dist_y = max(0, abs(j - mid_point) - flat_area)
                dist = max(dist_x, dist_y)

                stair_idx = (dist * num_cycles * num_stairs) // (mid_point - flat_area)
                direction = (stair_idx // num_stairs) % 2 == 0
                step_i = stair_idx % num_stairs
                height = step_i if direction else num_stairs - step_i
                hf_data[i, j] = height * step_height

        return hf_data.flatten()

    def step(self, state: State, action: jax.Array, *args) -> State:
        sys = self.env.unwrapped.sys
        key, sub_key = jax.random.split(self.key, 2)
        self.key = key

        stairs_id = jax.random.randint(sub_key, (1,), 0, state.info["curriculum"] + 1)[0]
        hf_data = sys.hfield_data * 0.0
        hf_data = jax.lax.cond(stairs_id == 1, lambda: self.get_pyramid_stairs(2, 6, 0.4), lambda: hf_data)
        hf_data = jax.lax.cond(stairs_id == 2, lambda: self.get_pyramid_stairs(2, 32, 0.4), lambda: hf_data)
        hf_data = jax.lax.cond(stairs_id == 3, lambda: self.get_pyramid_stairs(8, 6, 0.5), lambda: hf_data)
        hf_data = jax.lax.cond(stairs_id == 4, lambda: self.get_pyramid_stairs(8, 6, 1.0), lambda: hf_data)

        self.env.unwrapped.sys = sys.replace(hfield_data=sys.hfield_data * 0.0 + hf_data)
        return self.env.step(state, action, *args)


def ppo_train(
    environment: Union[envs_v1.Env, envs.Env],
    policy_hidden_layer_sizes: tuple,
    encoder_hidden_layer_sizes: tuple,
    value_hidden_layer_sizes: tuple,
    epoch_training_steps: int,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    run_eval: bool = True,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_epochs: int = 1,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    deterministic_eval: bool = False,
    normalize_advantage: bool = True,
    progress_fn: Callable = lambda *args: None,
    checkpoint: dict = None,
    curriculum: int = 0,
    **kwargs
):
    def init_training_components():
        ppo_network = ppo_networks.make_ppo_networks(
            state.obs.shape[-1],
            state.priv.shape[-1],
            env.action_size,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
            value_hidden_layer_sizes=value_hidden_layer_sizes,
            preprocess_observations_fn=running_statistics.normalize)

        make_policy = ppo_networks.make_inference_fn(ppo_network)

        optimizer = optax.adam(learning_rate=learning_rate)

        loss_fn = functools.partial(
            ppo_losses.compute_ppo_loss,
            ppo_network=ppo_network,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
            normalize_advantage=normalize_advantage)

        if checkpoint:
            policy_params = checkpoint["policy"]
            value_params = checkpoint["value"]
            normalizer_params = checkpoint["normalizer"]
            enc_params = checkpoint["encoder"]
            enc_normalizer_params = checkpoint["enc_normalizer"]
        else:
            policy_params = ppo_network.policy_network.init(key_policy)
            value_params = ppo_network.value_network.init(key_value)
            normalizer_params = running_statistics.init_state(specs.Array((state.obs.shape[-1] + 128), jnp.dtype("float32")))
            enc_params = ppo_network.encoder_network.init(key_enc)
            enc_normalizer_params = running_statistics.init_state(specs.Array(state.priv.shape[-1:], jnp.dtype("float32")))

        init_params = ppo_losses.PPONetworkParams(policy=policy_params, value=value_params, encoder=enc_params)
        optimizer_params = optimizer.init(init_params)

        training_state = TrainingState(
            optimizer_state=optimizer_params,
            params=init_params,
            normalizer_params=normalizer_params,
            enc_normalizer_params=enc_normalizer_params,
            env_steps=0
        )

        gradient_update_fn = gradients.gradient_update_fn(loss_fn, optimizer, pmap_axis_name=None, has_aux=True)

        return make_policy, training_state, gradient_update_fn

    def minibatch_step(carry, data: types.Transition,
                       normalizer_params: running_statistics.RunningStatisticsState,
                       enc_normalizer_params: running_statistics.RunningStatisticsState
                       ):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params,
            normalizer_params,
            enc_normalizer_params,
            data,
            key_loss,
            optimizer_state=optimizer_state
        )

        return (optimizer_state, params, key), metrics

    def sgd_step(carry, _, data: types.Transition,
                 normalizer_params: running_statistics.RunningStatisticsState,
                 enc_normalizer_params: running_statistics.RunningStatisticsState
                 ):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def shuffle(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)  # (minibatch * batch, unroll)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])  # (minibatch, batch, unroll)
            return x

        shuffled_data = jax.tree_util.tree_map(shuffle, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params, enc_normalizer_params=enc_normalizer_params),
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=num_minibatches
        )
        return (optimizer_state, params, key), metrics

    def training_step(carry, _):
        training_state, state, key = carry
        key, key_loss, key_generate_unroll = jax.random.split(key, 3)

        params = (training_state.normalizer_params, training_state.params.policy)
        enc_params = (training_state.enc_normalizer_params, training_state.params.encoder)
        policy = make_policy(params, enc_params)

        def unroll(carry, _):
            state, key = carry
            key, next_key = jax.random.split(key)
            next_state, data = generate_unroll(
                env,
                state,
                policy,
                key,
                unroll_length,
                extra_fields=("truncation",)
            )
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            unroll, (state, key_generate_unroll), (),
            length=batch_size * num_minibatches // num_envs
        )

        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)

        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.extras["policy_extras"]["encoding"]
            # jnp.concatenate([data.observation, data.extras["policy_extras"]["encoding"]], axis=-1)
        )

        enc_normalizer_params = running_statistics.update(
            training_state.enc_normalizer_params,
            data.priv)

        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step, data=data, normalizer_params=normalizer_params, enc_normalizer_params=enc_normalizer_params),
            (training_state.optimizer_state, training_state.params, key_loss), (),
            length=num_updates_per_batch
        )

        training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            enc_normalizer_params=enc_normalizer_params,
            env_steps=training_state.env_steps + env_steps_per_training_step
        )

        return (training_state, state, key), metrics

    def train_epoch(training_state, state, key):
        (training_state, state, key), loss_metrics = jax.lax.scan(
            training_step, (training_state, state, key), (), length=epoch_training_steps
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        train_metrics = {f"train/{k}": v for k, v in jax.tree_util.tree_map(jnp.mean, state.metrics).items()}
        train_state_info = {f"train/{k}": v for k, v in jax.tree_util.tree_map(jnp.mean, state.info).items()}
        return training_state, state, {**loss_metrics, **train_metrics, **train_state_info}

    def wrap_train_env(env):
        env = EpisodeWrapper(env, episode_length, action_repeat)
        env = HeightFieldWrapper(env)
        env = VmapWrapper(env)
        env = AutoResetWrapper(env)
        env.reset = jax.jit(env.reset)
        return env

    def wrap_eval_env(env):
        env = EpisodeWrapper(env, episode_length, action_repeat)
        env = VmapWrapper(env)
        env = AutoResetWrapper(env)
        env.reset = jax.jit(env.reset)
        return env

    env_steps_per_training_step = batch_size * unroll_length * num_minibatches * action_repeat

    key = jax.random.PRNGKey(seed=seed)
    key, env_key, key_policy, key_value, key_enc, epoch_key, eval_key, domain_key = jax.random.split(key, 8)
    key_envs = jax.random.split(env_key, num_envs)

    env = wrap_train_env(environment)
    state = env.reset(key_envs)
    make_policy, training_state, gradient_update_fn = init_training_components()

    current_step = 0
    curriculum = 0

    params = (training_state.normalizer_params, training_state.params.policy)
    enc_params = (training_state.enc_normalizer_params, training_state.params.encoder)
    metrics = {}
    if run_eval:
        evaluator = Evaluator(
            wrap_eval_env(environment),
            functools.partial(make_policy, deterministic=deterministic_eval),
            num_eval_envs=num_eval_envs, episode_length=episode_length,
            action_repeat=action_repeat, key=eval_key
        )
        metrics = evaluator.run_evaluation(params, enc_params, {})
        progress_fn(current_step, metrics, make_policy, training_state, curriculum)

    for _ in range(num_epochs):
        key, epoch_key = jax.random.split(key, 2)

        state.info["curriculum"] = jnp.full(num_envs, curriculum)
        (training_state, state, train_metrics) = train_epoch(training_state, state, epoch_key)
        current_step = int(training_state.env_steps)

        params = (training_state.normalizer_params, training_state.params.policy)
        enc_params = (training_state.enc_normalizer_params, training_state.params.encoder)
        metrics = evaluator.run_evaluation(params, enc_params, train_metrics) if run_eval else train_metrics

        progress_fn(current_step, metrics, make_policy, training_state, curriculum)

        _curr = curriculum
        if curriculum == 0 and metrics["train/episode_metrics"]["linear"] > 450:
            curriculum = 1
        elif curriculum == 1 and metrics["train/episode_metrics"]["linear"] > 450:
            curriculum = 2
        elif curriculum == 2 and metrics["train/episode_metrics"]["linear"] > 450:
            curriculum = 3
        elif curriculum == 3 and metrics["train/episode_metrics"]["linear"] > 450:
            curriculum = 4
        if _curr != curriculum:
            print(f"curriculum set to {curriculum}")

    return (make_policy, params, metrics)
