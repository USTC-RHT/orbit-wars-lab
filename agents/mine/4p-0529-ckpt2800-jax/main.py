"""Orbit Wars 4P JAX checkpoint agent.

Loads the Go_for_4P ckpt_002800 checkpoint directly. This is intended for
local visualization/debugging in orbit-wars-lab, not Kaggle submission speed.
"""

from __future__ import annotations

import os
import pickle
import sys
import threading
import traceback
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_HERE = Path(globals().get("__file__", ".")).resolve().parent
if not sys.path or sys.path[0] != str(_HERE):
    sys.path.insert(0, str(_HERE))


def _drop_foreign_jax_agent_modules() -> None:
    """Avoid fast-mode collisions with other bundled agents' jax_agent packages."""
    local_pkg = (_HERE / "jax_agent").resolve()
    for name, module in list(sys.modules.items()):
        if name != "jax_agent" and not name.startswith("jax_agent."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            sys.modules.pop(name, None)
            continue
        try:
            Path(module_file).resolve().relative_to(local_pkg)
        except ValueError:
            sys.modules.pop(name, None)


_drop_foreign_jax_agent_modules()

import jax
import jax.numpy as jnp

from jax_agent.fast_env import EnvParams
from jax_agent.features_jax import (
    FLEET_DIM,
    GLOBAL_DIM,
    MAP_CHANNELS,
    PLANET_DIM,
    RELATIVE_DIM,
    FeatureParams,
)
from jax_agent.model_jax import AutoregressiveActorCritic
from jax_agent.normalization_jax import NormalizerParams
from jax_agent.train_jax import (
    auto_deterministic_policy_actions,
    flip_actions,
    flip_state,
    jax_actions_to_kaggle,
    kaggle_obs_to_jax_state,
)


_ERROR_LOGGED = False
_RUNTIME = None
_RUNTIME_LOCK = threading.Lock()
_PRELOAD_THREAD = None


def _bind_local_jax_agent_symbols() -> None:
    """Rebind bundled jax_agent symbols after other agents mutate sys.modules."""
    global EnvParams, FeatureParams, NormalizerParams, AutoregressiveActorCritic
    global FLEET_DIM, GLOBAL_DIM, MAP_CHANNELS, PLANET_DIM, RELATIVE_DIM
    global auto_deterministic_policy_actions, flip_actions, flip_state
    global jax_actions_to_kaggle, kaggle_obs_to_jax_state

    _drop_foreign_jax_agent_modules()
    if not sys.path or sys.path[0] != str(_HERE):
        sys.path.insert(0, str(_HERE))

    fast_env = importlib.import_module("jax_agent.fast_env")
    features = importlib.import_module("jax_agent.features_jax")
    model_mod = importlib.import_module("jax_agent.model_jax")
    norm_mod = importlib.import_module("jax_agent.normalization_jax")
    train_mod = importlib.import_module("jax_agent.train_jax")

    EnvParams = fast_env.EnvParams
    FeatureParams = features.FeatureParams
    FLEET_DIM = features.FLEET_DIM
    GLOBAL_DIM = features.GLOBAL_DIM
    MAP_CHANNELS = features.MAP_CHANNELS
    PLANET_DIM = features.PLANET_DIM
    RELATIVE_DIM = features.RELATIVE_DIM
    AutoregressiveActorCritic = model_mod.AutoregressiveActorCritic
    NormalizerParams = norm_mod.NormalizerParams
    auto_deterministic_policy_actions = train_mod.auto_deterministic_policy_actions
    flip_actions = train_mod.flip_actions
    flip_state = train_mod.flip_state
    jax_actions_to_kaggle = train_mod.jax_actions_to_kaggle
    kaggle_obs_to_jax_state = train_mod.kaggle_obs_to_jax_state


@dataclass
class _Runtime:
    model: AutoregressiveActorCritic
    params: Any
    normalizer_state: Any
    norm_params: NormalizerParams
    feat_params: FeatureParams
    env_params: EnvParams
    infer_fn: Any = None

    @property
    def apply_fn(self):
        return self.model.apply

    def act(self, observation: Any) -> list[list[float | int]]:
        player = int(_read_field(observation, "player", 0) or 0)
        state = kaggle_obs_to_jax_state(observation, self.env_params)
        infer_state = flip_state(state, player) if player != 0 else state
        infer_fn = self.infer_fn or _make_infer_fn(self)
        actions = infer_fn(infer_state)
        if player != 0:
            actions = flip_actions(actions, player)
        return jax_actions_to_kaggle(jax.device_get(actions))


def _read_field(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set_agent_dir_from_config(configuration: Any) -> None:
    global _HERE
    raw_path = _read_field(configuration, "__raw_path__", None)
    if not raw_path:
        return
    agent_dir = Path(str(raw_path)).resolve().parent
    if agent_dir == _HERE:
        return
    _HERE = agent_dir
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))


def _load_runtime() -> _Runtime:
    _bind_local_jax_agent_symbols()
    ckpt_path = _HERE / "weights" / "ckpt_002800.pkl"
    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)
    cfg = ckpt["cfg"]
    model = AutoregressiveActorCritic(
        planet_dim=PLANET_DIM,
        relative_dim=RELATIVE_DIM,
        global_dim=GLOBAL_DIM,
        relative_count=int(cfg.env.max_planets) + 1,
        ship_bucket_count=int(cfg.env.ship_bucket_count),
        hidden_size=int(cfg.model.hidden_size),
        use_cnn=bool(cfg.model.use_cnn),
        map_channels=MAP_CHANNELS,
        fleet_dim=FLEET_DIM,
        max_fleet_tokens=int(getattr(cfg.model, "max_fleet_tokens", 16)),
        num_transformer_layers=int(getattr(cfg.model, "num_transformer_layers", 2)),
        num_heads=int(getattr(cfg.model, "num_heads", 4)),
        max_action_tokens=int(getattr(cfg.model, "max_action_tokens", 12)),
        end_bias_init=float(getattr(cfg.model, "end_bias_init", -3.0)),
    )
    env_params = EnvParams(
        episode_steps=int(cfg.env.episode_steps),
        max_planets=int(cfg.env.max_planets),
        max_fleets=512,
        max_actions=int(cfg.env.max_planets),
        num_players=4,
    )
    feat_params = FeatureParams.from_train_config(cfg)
    norm_params = NormalizerParams.from_train_config(cfg)
    runtime = _Runtime(
        model=model,
        params=jax.tree_util.tree_map(jnp.asarray, ckpt["params"]),
        normalizer_state=jax.tree_util.tree_map(jnp.asarray, ckpt["normalizer_state"]),
        norm_params=norm_params,
        feat_params=feat_params,
        env_params=env_params,
    )
    runtime.infer_fn = _make_infer_fn(runtime)
    return runtime


def _make_infer_fn(runtime: _Runtime):
    def infer(state):
        return auto_deterministic_policy_actions(
            runtime,
            runtime.model,
            state,
            runtime.feat_params,
            runtime.normalizer_state,
            runtime.norm_params,
        )

    return jax.jit(infer)


def _runtime() -> _Runtime:
    global _RUNTIME
    if _RUNTIME is None:
        with _RUNTIME_LOCK:
            if _RUNTIME is None:
                _RUNTIME = _load_runtime()
    return _RUNTIME


def _start_preload() -> None:
    global _PRELOAD_THREAD
    if _RUNTIME is not None:
        return
    if _PRELOAD_THREAD is not None and _PRELOAD_THREAD.is_alive():
        return
    _PRELOAD_THREAD = threading.Thread(target=_runtime, daemon=True)
    _PRELOAD_THREAD.start()


def agent(observation: Any, configuration: Any = None) -> list[list[float | int]]:
    global _ERROR_LOGGED
    try:
        _set_agent_dir_from_config(configuration)
        _start_preload()
        return _runtime().act(observation)
    except Exception:
        if not _ERROR_LOGGED:
            traceback.print_exc(file=sys.stderr)
            _ERROR_LOGGED = True
        raise


__all__ = ["agent"]
