"""JAX autoregressive fixed-pool 0525 ckpt_004000 agent for Orbit Wars Lab."""

from __future__ import annotations

import os
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax_compilation_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

_HERE = Path(globals().get("__file__", ".")).resolve().parent
_RL_ROOT = _HERE

_INFER_FN = None
_ENV_PARAMS = None
_FLIP_STATE = None
_FLIP_ACTIONS = None
_KAGGLE_OBS_TO_JAX_STATE = None
_JAX_ACTIONS_TO_KAGGLE = None
_LOAD_ERROR: BaseException | None = None
_LOAD_THREAD: threading.Thread | None = None
_LOAD_LOCK = threading.Lock()


def _is_local_module(module: Any) -> bool:
    file_name = getattr(module, "__file__", None)
    if not file_name:
        return False
    try:
        return Path(file_name).resolve().is_relative_to(_RL_ROOT)
    except OSError:
        return False


def _activate_local_imports() -> None:
    prefixes = ("jax_agent", "src")
    stale = any(
        (name == prefix or name.startswith(prefix + ".")) and not _is_local_module(module)
        for name, module in list(sys.modules.items())
        for prefix in prefixes
    )
    if stale:
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                sys.modules.pop(name, None)
    try:
        sys.path.remove(str(_RL_ROOT))
    except ValueError:
        pass
    sys.path.insert(0, str(_RL_ROOT))


def _set_agent_dir(config: Any = None) -> None:
    global _HERE, _RL_ROOT
    raw_path = config.get("__raw_path__") if isinstance(config, dict) else getattr(config, "__raw_path__", None)
    if raw_path:
        _HERE = Path(raw_path).resolve().parent
        _RL_ROOT = _HERE


def _load_model() -> None:
    global _INFER_FN, _ENV_PARAMS, _FLIP_STATE, _FLIP_ACTIONS, _KAGGLE_OBS_TO_JAX_STATE, _JAX_ACTIONS_TO_KAGGLE, _LOAD_ERROR
    t0 = time.time()
    try:
        _activate_local_imports()

        import jax
        import jax.numpy as jnp
        from flax.training.train_state import TrainState
        import optax

        from jax_agent.fast_env import (
            COMETS_PER_SPAWN,
            MAX_COMET_PATH,
            MAX_COMET_SPAWNS,
            EnvParams,
            EnvState,
        )
        from jax_agent.features_jax import FLEET_DIM, GLOBAL_DIM, PLANET_DIM, RELATIVE_DIM, FeatureParams
        from jax_agent.model_jax import AutoregressiveActorCritic
        from jax_agent.normalization_jax import NormalizerParams
        from jax_agent.train_jax import (
            auto_deterministic_policy_actions,
            flip_actions,
            flip_state,
            jax_actions_to_kaggle,
            kaggle_obs_to_jax_state,
        )

        ckpt_path = _HERE / "weights" / "ckpt_004000.pkl"
        with ckpt_path.open("rb") as f:
            ckpt = pickle.load(f)

        cfg = ckpt["cfg"]
        params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
        norm_state = ckpt["normalizer_state"]

        env_params = EnvParams(
            episode_steps=cfg.env.episode_steps,
            max_planets=cfg.env.max_planets,
        )
        feat_params = FeatureParams(
            max_planets=cfg.env.max_planets,
            ship_bucket_count=cfg.env.ship_bucket_count,
            max_ships=cfg.env.max_ships,
            max_production=cfg.env.max_production,
            episode_steps=cfg.env.episode_steps,
            target_filter_enabled=cfg.env.target_filter_enabled,
            attack_top_k=cfg.env.attack_top_k,
            defense_top_k=cfg.env.defense_top_k,
            max_fleet_tokens=getattr(cfg.model, "max_fleet_tokens", 16),
        )
        norm_params = NormalizerParams.from_train_config(cfg)

        model = AutoregressiveActorCritic(
            planet_dim=PLANET_DIM,
            relative_dim=RELATIVE_DIM,
            global_dim=GLOBAL_DIM,
            relative_count=cfg.env.max_planets + 1,
            ship_bucket_count=cfg.env.ship_bucket_count,
            hidden_size=cfg.model.hidden_size,
            use_cnn=cfg.model.use_cnn,
            fleet_dim=FLEET_DIM,
            max_fleet_tokens=feat_params.max_fleet_tokens,
            num_transformer_layers=getattr(cfg.model, "num_transformer_layers", 2),
            num_heads=getattr(cfg.model, "num_heads", 4),
            max_action_tokens=int(getattr(cfg.model, "max_action_tokens", 16)),
            end_bias_init=float(getattr(cfg.model, "end_bias_init", -2.0)),
        )
        train_state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.set_to_zero(),
        )

        @jax.jit
        def _infer(obs_state):
            return auto_deterministic_policy_actions(
                train_state,
                model,
                obs_state,
                feat_params,
                norm_state,
                norm_params,
            )

        p = env_params.max_planets
        f = env_params.max_fleets
        dummy_state = EnvState(
            planets=jnp.zeros((1, p, 7), dtype=jnp.float32),
            initial_planets=jnp.zeros((1, p, 7), dtype=jnp.float32),
            planet_alive=jnp.zeros((1, p), dtype=bool),
            fleets=jnp.zeros((1, f, 7), dtype=jnp.float32),
            fleet_alive=jnp.zeros((1, f), dtype=bool),
            step=jnp.zeros(1, dtype=jnp.int32),
            angular_velocity=jnp.zeros(1, dtype=jnp.float32),
            next_fleet_id=jnp.zeros(1, dtype=jnp.int32),
            done=jnp.zeros(1, dtype=bool),
            winner=jnp.full(1, -1, dtype=jnp.int32),
            overflow_fleets=jnp.zeros(1, dtype=jnp.int32),
            comet_paths=jnp.zeros((1, MAX_COMET_SPAWNS, COMETS_PER_SPAWN, MAX_COMET_PATH, 2), dtype=jnp.float32),
            comet_lengths=jnp.zeros((1, MAX_COMET_SPAWNS, COMETS_PER_SPAWN), dtype=jnp.int32),
            comet_ships=jnp.zeros((1, MAX_COMET_SPAWNS), dtype=jnp.float32),
            comet_valid=jnp.zeros((1, MAX_COMET_SPAWNS), dtype=bool),
            comet_path_index=jnp.full((1, MAX_COMET_SPAWNS), -2, dtype=jnp.int32),
            comet_slots=jnp.full((1, MAX_COMET_SPAWNS, COMETS_PER_SPAWN), -1, dtype=jnp.int32),
        )
        jax.block_until_ready(_infer(dummy_state))

        _ENV_PARAMS = env_params
        _FLIP_STATE = flip_state
        _FLIP_ACTIONS = flip_actions
        _KAGGLE_OBS_TO_JAX_STATE = kaggle_obs_to_jax_state
        _JAX_ACTIONS_TO_KAGGLE = jax_actions_to_kaggle
        _INFER_FN = _infer
        print(f"[agent] JAX autoregressive model ready in {time.time() - t0:.2f}s", file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001
        _LOAD_ERROR = e
        print(f"[agent] JAX autoregressive model load failed after {time.time() - t0:.2f}s: {e}", file=sys.stderr, flush=True)


def _ensure_loaded(config: Any = None, *, block: bool = False) -> None:
    global _LOAD_THREAD
    _set_agent_dir(config)
    with _LOAD_LOCK:
        if _INFER_FN is None and _LOAD_ERROR is None and _LOAD_THREAD is None:
            _LOAD_THREAD = threading.Thread(target=_load_model, daemon=True)
            _LOAD_THREAD.start()
    if block and _LOAD_THREAD is not None:
        _LOAD_THREAD.join()


def agent(observation, configuration=None):
    _ensure_loaded(configuration, block=True)

    if _LOAD_ERROR is not None or _INFER_FN is None:
        return []

    try:
        player = observation.player
        jax_state = _KAGGLE_OBS_TO_JAX_STATE(observation, _ENV_PARAMS)
        if player == 1:
            jax_state = _FLIP_STATE(jax_state)
        acts_jax = _INFER_FN(jax_state)
        if player == 1:
            acts_jax = _FLIP_ACTIONS(acts_jax)
        return _JAX_ACTIONS_TO_KAGGLE(acts_jax)
    except Exception:  # noqa: BLE001
        return []


__all__ = ["agent"]
