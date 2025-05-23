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

import flax
import jax
import jax.numpy as jnp
import optax
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper, VmapWrapper
from brax.training import gradients, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.v1 import envs as envs_v1

from brax.training.acting import Evaluator, generate_unroll, reccurent_generate_unroll
from brax.training.hf_wrapper import HeightFieldWrapper


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    enc_normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


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
    progress_fn: Callable = lambda *_: None,
    checkpoint: dict = None,
    curriculum: int = 0,
    reccurent_policy: bool = False,
    gru_layers: int = 0,
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
            preprocess_observations_fn=running_statistics.normalize,
            is_reccurent=reccurent_policy,
            num_gru_layers=gru_layers
        )

        make_policy = ppo_networks.make_inference_fn(ppo_network, is_reccurent=reccurent_policy)

        optimizer = optax.adam(learning_rate=learning_rate)

        loss_fn = functools.partial(
            ppo_losses.compute_ppo_loss,
            ppo_network=ppo_network,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
            normalize_advantage=normalize_advantage
        )
        generate_unroll_fn = reccurent_generate_unroll if reccurent_policy else generate_unroll

        if checkpoint:
            policy_params = checkpoint["policy"]
            value_params = checkpoint["value"]
            normalizer_params = checkpoint["normalizer"]
            enc_params = checkpoint["encoder"]
            enc_normalizer_params = checkpoint["enc_normalizer"]
        else:
            policy_params = ppo_network.policy_network.init(key_policy)
            value_params = ppo_network.value_network.init(key_value)
            normalizer_params = running_statistics.init_state(
                specs.Array(
                    (state.obs.shape[-1] + encoder_hidden_layer_sizes[-1]),
                    jnp.dtype("float32")
                )
            )
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

        return make_policy, training_state, generate_unroll_fn, gradient_update_fn

    def minibatch_step(
        carry,
        data: types.Transition,
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

    def sgd_step(
        carry,
        _,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
        enc_normalizer_params: running_statistics.RunningStatisticsState
    ):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def shuffle(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(shuffle, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                minibatch_step,
                normalizer_params=normalizer_params,
                enc_normalizer_params=enc_normalizer_params
            ),
            (optimizer_state, params, key_grad),
            shuffled_data,  # [batch, unroll_length, -1]
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
            next_state, data = generate_unroll_fn(env, state, policy, key, unroll_length, extra_fields=("truncation",))
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            unroll,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs
        )

        # (bs * nmb // nenvs, unroll_length, nenvs, -1)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)  # (bs * nmb // nenvs, nenvs, unroll_length, -1)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)  # (bs * nmb, unroll_length, -1)

        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.extras["policy_extras"]["encoding"]
        )

        enc_normalizer_params = running_statistics.update(training_state.enc_normalizer_params, data.priv)

        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step,
                data=data,
                normalizer_params=normalizer_params,
                enc_normalizer_params=enc_normalizer_params
            ),
            (training_state.optimizer_state, training_state.params, key_loss),
            (),
            length=num_updates_per_batch,
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
            training_step,
            (training_state, state, key),
            (),
            length=epoch_training_steps
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
    (
        key,
        env_key,
        key_policy,
        key_value,
        key_enc,
        epoch_key,
        eval_key,
        domain_key,
    ) = jax.random.split(key, 8)
    key_envs = jax.random.split(env_key, num_envs)

    env = wrap_train_env(environment)
    state = env.reset(key_envs)

    if reccurent_policy:
        hidden_size = policy_hidden_layer_sizes[0]
        state = state.replace(
            hidden_state=jnp.broadcast_to(
                jnp.zeros((1, 1, hidden_size)), (num_envs, gru_layers, hidden_size)
            )
        )

    make_policy, training_state, generate_unroll_fn, gradient_update_fn = init_training_components()
    training_spec = {
        "policy_hidden_layer_sizes": policy_hidden_layer_sizes,
        "encoder_hidden_layer_sizes": encoder_hidden_layer_sizes,
        "value_hidden_layer_sizes": value_hidden_layer_sizes,
        "is_recurrent": reccurent_policy,
        "gru_layers": gru_layers
    }

    current_step = 0
    curriculum = 0

    params = (training_state.normalizer_params, training_state.params.policy)
    enc_params = (training_state.enc_normalizer_params, training_state.params.encoder)
    metrics = {}
    if run_eval:
        evaluator = Evaluator(
            wrap_eval_env(environment),
            functools.partial(make_policy, deterministic=deterministic_eval),
            num_eval_envs=num_eval_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            key=eval_key
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

        progress_fn(current_step, metrics, make_policy, (training_state, training_spec), curriculum)

        _curr = curriculum
        if curriculum == 0 and metrics["train/linear"] > 0.91:
            curriculum = 1
        elif curriculum == 1 and metrics["train/linear"] > 0.9:
            curriculum = 2
        elif curriculum == 2 and metrics["train/linear"] > 0.89:
            curriculum = 3
        elif curriculum == 3 and metrics["train/linear"] > 0.85:
            curriculum = 4
        elif curriculum == 4 and metrics["train/linear"] > 0.85:
            curriculum = 5
        elif curriculum > 4 and metrics["train/linear"] > 0.82:
            curriculum += 1
        if _curr != curriculum:
            print(f"curriculum set to {curriculum}")

    return (make_policy, params, metrics)
