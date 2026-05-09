"""0508 checkpoint 300 agent for Orbit Wars Lab."""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path
from typing import Any

_HERE = Path(globals().get("__file__", ".")).resolve().parent
_CFG = None
_MODEL = None
_DEVICE = None
_STATE_NORMALIZER = None
_LOAD_ERROR: BaseException | None = None
_LOAD_THREAD: threading.Thread | None = None
_LOAD_LOCK = threading.Lock()


def _read(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set_agent_dir(config: Any = None) -> None:
    global _HERE
    raw_path = config.get("__raw_path__") if isinstance(config, dict) else getattr(config, "__raw_path__", None)
    if raw_path:
        _HERE = Path(raw_path).resolve().parent


def _checkpoint_path() -> Path:
    local = _HERE / "weights" / "ckpt_last.pt"
    if local.exists():
        return local
    raise FileNotFoundError(f"No checkpoint found at {local}")


def _load() -> None:
    global _CFG, _MODEL, _DEVICE, _STATE_NORMALIZER, _LOAD_ERROR
    try:
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))

        import torch
        import orbit_rl_0508_ckpt300
        import orbit_rl_0508_ckpt300.config
        import orbit_rl_0508_ckpt300.features
        import orbit_rl_0508_ckpt300.model
        import orbit_rl_0508_ckpt300.normalization
        import orbit_rl_0508_ckpt300.state
        import orbit_rl_0508_ckpt300.physics
        import orbit_rl_0508_ckpt300.opponents
        import orbit_rl_0508_ckpt300.ppo

        for module_name in list(sys.modules):
            if module_name.startswith("orbit_rl_0508_ckpt300"):
                sys.modules[module_name.replace("orbit_rl_0508_ckpt300", "src", 1)] = sys.modules[module_name]

        from orbit_rl_0508_ckpt300.config import load_config
        from orbit_rl_0508_ckpt300.features import CANDIDATE_DIM, GLOBAL_DIM, SELF_DIM, target_count
        from orbit_rl_0508_ckpt300.model import ActorCritic
        from orbit_rl_0508_ckpt300.normalization import StateNormalizer

        cfg = load_config(_HERE / "default_cfg.yaml")
        device = torch.device("cpu")
        model = ActorCritic(
            self_dim=SELF_DIM,
            candidate_dim=CANDIDATE_DIM,
            global_dim=GLOBAL_DIM,
            candidate_count=target_count(cfg.env),
            ship_bucket_count=cfg.env.ship_bucket_count,
            hidden_size=cfg.model.hidden_size,
        ).to(device)
        ckpt = torch.load(_checkpoint_path(), map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt))
        model.eval()

        state_normalizer = StateNormalizer(
            enabled=cfg.state_norm.enabled,
            clip=cfg.state_norm.clip,
            eps=cfg.state_norm.eps,
        )
        if isinstance(ckpt, dict) and "state_normalizer" in ckpt:
            state_normalizer.load_state_dict(ckpt["state_normalizer"])
        else:
            state_normalizer.enabled = False

        _CFG = cfg
        _MODEL = model
        _DEVICE = device
        _STATE_NORMALIZER = state_normalizer
    except BaseException as exc:
        _LOAD_ERROR = exc


def _ensure_loading() -> None:
    global _LOAD_THREAD
    if _MODEL is not None or _LOAD_ERROR is not None:
        return
    with _LOAD_LOCK:
        if _LOAD_THREAD is None:
            _LOAD_THREAD = threading.Thread(target=_load, name="0508-ckpt300-loader", daemon=True)
            _LOAD_THREAD.start()


def _fallback_agent(obs: Any) -> list[list[float | int]]:
    player = _read(obs, "player", 0)
    planets = _read(obs, "planets", []) or []
    initial_planets = _read(obs, "initial_planets", []) or planets
    comets = _read(obs, "comets", []) or []
    comet_ids = set(_read(obs, "comet_planet_ids", []) or [])
    angular_velocity = float(_read(obs, "angular_velocity", 0.0) or 0.0)
    sources = [p for p in planets if len(p) > 5 and p[1] == player and p[5] >= 12]
    targets = [p for p in planets if len(p) > 5 and p[1] != player]
    moves: list[list[float | int]] = []
    for src in sources:
        target = min(targets, key=lambda p: ((p[2] - src[2]) ** 2 + (p[3] - src[3]) ** 2, p[0]), default=None)
        if target is None:
            continue
        ships = min(int(src[5] * 0.6), max(1, int(target[5]) + 1))
        if ships <= 0:
            continue
        if int(target[1]) != player and ships <= int(target[5]):
            continue

        angle = _aim_angle(src, target, ships, initial_planets, angular_velocity, comets, comet_ids, planets)
        if angle is None:
            continue
        moves.append([src[0], float(angle), int(ships)])
    return moves


def _aim_angle(
    src: list,
    target: list,
    ships: int,
    initial_planets: list,
    angular_velocity: float,
    comets: list,
    comet_ids: set,
    planets: list,
) -> float | None:
    try:
        from orbit_rl_0508_ckpt300.physics import aim_with_prediction
        from orbit_rl_0508_ckpt300.state import Planet

        src_p = Planet(*src)
        target_p = Planet(*target)
        initial_by_id = {p[0]: Planet(*p) for p in initial_planets if len(p) >= 7}
        planet_objs = [Planet(*p) for p in planets if len(p) >= 7]
        result = aim_with_prediction(
            src_p,
            target_p,
            ships,
            initial_by_id,
            angular_velocity,
            comets,
            comet_ids,
            planet_objs,
        )
        return None if result is None else float(result[0])
    except Exception:
        return None


def _model_agent(obs: Any) -> list[list[float | int]]:
    from orbit_rl_0508_ckpt300.features import encode_turn
    from orbit_rl_0508_ckpt300.opponents import actions_from_policy

    batch = encode_turn(obs, _CFG.env)
    return actions_from_policy(
        batch,
        _MODEL,
        _DEVICE,
        deterministic=False,
        state_normalizer=_STATE_NORMALIZER,
    )


def agent(obs, config=None):
    _set_agent_dir(config)
    _ensure_loading()
    if _MODEL is None and _LOAD_THREAD is not None and _LOAD_THREAD.is_alive():
        _LOAD_THREAD.join()
    if _MODEL is None:
        return []
    return _model_agent(obs)


__all__ = ["agent"]
