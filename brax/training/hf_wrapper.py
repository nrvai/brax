
import jax
import jax.numpy as jnp
from brax.base import State, System
from brax.envs import Env, Wrapper
from nrv_lab.sim.worldgen import get_curriculum_stairs, get_curriculum_slopes, get_curriculum_holes, get_curriculum_gutters


class HeightFieldWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.hf_data = self.env.unwrapped.sys.hfield_data * 1.0
        self.hf_size = self.env.unwrapped.sys.hfield_nrow * self.env.unwrapped.sys.hfield_ncol

    def env_fn(self, sys: System) -> Env:
        env = self.env
        env.unwrapped.sys = sys
        return env

    def reset(self, rng):
        state = super().reset(rng)
        state.info["hf_scale"] = 0.0
        state.info["hf_add"] = jnp.zeros(self.hf_size)
        state.info["hf_min"] = 0.0
        state.info["hf_max"] = 0.0
        state.info["hf_data"] = self.hf_data
        return state

    def step(self, state: State, action: jax.Array, *args) -> State:
        def get_hf_add(rng, curriculum):
            fn_rng, rng = jax.random.split(rng)
            fn_id = jax.random.randint(rng, (1,), 0, 4)[0]

            hf_add = get_curriculum_slopes(fn_rng, curriculum)
            hf_add = jax.lax.cond(fn_id == 1, lambda: get_curriculum_stairs(fn_rng, curriculum), lambda: hf_add)
            hf_add = jax.lax.cond(fn_id == 2, lambda: get_curriculum_holes(fn_rng, curriculum), lambda: hf_add)
            hf_add = jax.lax.cond(fn_id == 3, lambda: get_curriculum_gutters(fn_rng, curriculum), lambda: hf_add)
            return hf_add

        rng, scale_rng, add_rng = jax.random.split(state.info["rng"], 3)
        hf_scale = jax.lax.cond(
            state.info["steps"] == 0,
            lambda: jax.random.randint(scale_rng, (1,), 0, state.info["curriculum"] * 2)[0] / 100.0,
            lambda: state.info["hf_scale"]
        )
        hf_add = jax.lax.cond(
            state.info["steps"] == 0,
            lambda: get_hf_add(add_rng, state.info["curriculum"]),
            lambda: state.info["hf_add"]
        )

        hf_data = self.hf_data * hf_scale + hf_add
        sys = self.env.unwrapped.sys
        new_sys = sys.replace(hfield_data=hf_data)
        new_env = self.env_fn(new_sys)

        state.info["rng"] = rng
        state.info["hf_scale"] = hf_scale
        state.info["hf_add"] = hf_add
        state.info["hf_data"] = hf_data
        state.info["hf_min"] = hf_data.min()
        state.info["hf_max"] = hf_data.max()

        return new_env.step(state, action, *args)
