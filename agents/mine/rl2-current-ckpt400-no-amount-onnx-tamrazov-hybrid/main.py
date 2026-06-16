"""Orbit Wars ONNX hybrid agent for no-amount-head ckpt_000400.

2-player games use the exported ONNX RL_2 no-amount-head JAX checkpoint.
4-player games delegate to the public tamrazov-starwars agent.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

_HERE = Path(globals().get("__file__", ".")).resolve().parent
if not sys.path or sys.path[0] != str(_HERE):
    sys.path.insert(0, str(_HERE))


def _drop_foreign_jax_agent_modules() -> None:
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


_ERROR_LOGGED = False
_RUNTIME = None
_RUNTIME_LOCK = threading.Lock()
_PRELOAD_THREAD = None
_TAMRAZOV = None
_ACTIVE_2P_CHECKPOINTS = ("ckpt_000400",)
_NEG_INF = -1.0e9
_MIN_LAUNCH_SHIPS = 10.0


@dataclass(frozen=True)
class _EnvParams:
    episode_steps: int
    max_planets: int

    @property
    def max_fleets(self) -> int:
        return 512


@dataclass
class _CheckpointRuntime:
    name: str
    meta: dict
    norm_state: dict[str, np.ndarray]
    norm_clip: float
    norm_eps: float
    start: Any
    step: Any
    tgt: Any
    tok: Any


@dataclass
class _StateArrays:
    planets: np.ndarray
    initial_planets: np.ndarray
    planet_alive: np.ndarray
    fleets: np.ndarray
    fleet_alive: np.ndarray
    step: np.ndarray
    angular_velocity: np.ndarray
    next_fleet_id: np.ndarray
    done: np.ndarray
    winner: np.ndarray
    overflow_fleets: np.ndarray
    comet_paths: np.ndarray
    comet_lengths: np.ndarray
    comet_ships: np.ndarray
    comet_valid: np.ndarray
    comet_path_index: np.ndarray
    comet_slots: np.ndarray


_PLANET_NORMALIZE_INDICES = np.asarray([5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], dtype=np.int32)
_RELATIVE_NORMALIZE_INDICES = np.asarray([1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21], dtype=np.int32)
_GLOBAL_NORMALIZE_INDICES = np.arange(11, dtype=np.int32)
_AIM_HORIZON = 60
_AIM_NEWTON_ITERS = 5
_AIM_NEWTON_MIN_DERIV = 1.0e-4
_AIM_NEWTON_MAX_STEP = 12.0
_AIM_TARGET_INSET = 1.0e-3
_MAX_COMET_SPAWNS = 5
_COMETS_PER_SPAWN = 4
_MAX_COMET_PATH = 40


def _set_agent_dir(config: Any = None) -> None:
    global _HERE
    raw_path = config.get("__raw_path__") if isinstance(config, dict) else getattr(config, "__raw_path__", None)
    if raw_path:
        _HERE = Path(raw_path).resolve().parent
    if not sys.path or sys.path[0] != str(_HERE):
        sys.path.insert(0, str(_HERE))


def _read_field(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _rows(obj: Any, key: str) -> list:
    return list(_read_field(obj, key, []) or [])


def _row_value(row: Any, idx: int, attr: str, default=None):
    try:
        return row[idx]
    except Exception:
        return getattr(row, attr, default)


def _is_four_player(observation: Any, configuration: Any = None) -> bool:
    for key in ("agentCount", "num_agents", "numAgents", "players"):
        value = _read_field(configuration, key, None)
        if value is not None:
            try:
                return int(value) >= 4
            except Exception:
                pass

    owners: set[int] = set()
    for row in _rows(observation, "planets"):
        owner = int(_row_value(row, 1, "owner", -1))
        if owner >= 0:
            owners.add(owner)
    for row in _rows(observation, "fleets"):
        owner = int(_row_value(row, 1, "owner", -1))
        if owner >= 0:
            owners.add(owner)
    return len(owners) >= 3 or any(owner >= 2 for owner in owners)


def _has_sendable_source(observation: Any) -> bool:
    player = int(_read_field(observation, "player", 0) or 0)
    for row in _rows(observation, "planets"):
        owner = int(_row_value(row, 1, "owner", -1))
        ships = float(_row_value(row, 5, "ship_count", 0.0) or 0.0)
        if owner == player and ships >= _MIN_LAUNCH_SHIPS:
            return True
    return False


def _load_tamrazov():
    global _TAMRAZOV
    if _TAMRAZOV is not None:
        return _TAMRAZOV
    path = _HERE / "tamrazov_main.py"
    spec = importlib.util.spec_from_file_location("_hybrid_tamrazov_main", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tamrazov module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _TAMRAZOV = module
    return module


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _safe_logits(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x, _NEG_INF).astype(np.float32, copy=False)


def _valid_logit_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (x > _NEG_INF * 0.5)


def _full_send_amount_np(ships: np.ndarray) -> np.ndarray:
    s = np.floor(ships).astype(np.float32, copy=False)
    return np.where(s >= _MIN_LAUNCH_SHIPS, s, 0.0).astype(np.float32, copy=False)


def _read_obs_field(obs: Any, key: str, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _mirror_planet_indices_np(count: int) -> np.ndarray:
    idx = np.arange(count, dtype=np.int32)
    mirror = (idx // 4) * 4 + (3 - (idx % 4))
    return np.where(mirror < count, mirror, idx)


def _rotate_angle_pi_np(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(-np.sin(angle), -np.cos(angle)).astype(np.float32, copy=False)


def _flip_owner_np(owner: np.ndarray) -> np.ndarray:
    return np.where(owner == 0.0, 1.0, np.where(owner == 1.0, 0.0, owner)).astype(owner.dtype, copy=False)


def _flip_state_arrays_np(state: _StateArrays) -> _StateArrays:
    planet_perm = _mirror_planet_indices_np(state.planets.shape[-2])
    planets = state.planets[:, planet_perm, :].copy()
    planets[..., 1] = _flip_owner_np(planets[..., 1])
    planets[..., 2] = 100.0 - planets[..., 2]
    planets[..., 3] = 100.0 - planets[..., 3]

    initial_planets = state.initial_planets[:, planet_perm, :].copy()
    initial_planets[..., 1] = _flip_owner_np(initial_planets[..., 1])
    initial_planets[..., 2] = 100.0 - initial_planets[..., 2]
    initial_planets[..., 3] = 100.0 - initial_planets[..., 3]

    fleets = state.fleets.copy()
    fleets[..., 1] = _flip_owner_np(fleets[..., 1])
    fleets[..., 2] = 100.0 - fleets[..., 2]
    fleets[..., 3] = 100.0 - fleets[..., 3]
    fleets[..., 4] = _rotate_angle_pi_np(fleets[..., 4])

    comet_paths = state.comet_paths.copy()
    comet_paths[..., 0] = 100.0 - comet_paths[..., 0]
    comet_paths[..., 1] = 100.0 - comet_paths[..., 1]

    safe_slots = np.clip(state.comet_slots, 0, state.planets.shape[-2] - 1)
    comet_slots = np.where(state.comet_slots >= 0, planet_perm[safe_slots], -1).astype(np.int32, copy=False)

    return _StateArrays(
        planets=planets,
        initial_planets=initial_planets,
        planet_alive=state.planet_alive[:, planet_perm].copy(),
        fleets=fleets,
        fleet_alive=state.fleet_alive,
        step=state.step,
        angular_velocity=state.angular_velocity,
        next_fleet_id=state.next_fleet_id,
        done=state.done,
        winner=state.winner,
        overflow_fleets=state.overflow_fleets,
        comet_paths=comet_paths,
        comet_lengths=state.comet_lengths,
        comet_ships=state.comet_ships,
        comet_valid=state.comet_valid,
        comet_path_index=state.comet_path_index,
        comet_slots=comet_slots,
    )


def _flip_actions_np(actions: np.ndarray) -> np.ndarray:
    out = actions.copy()
    ships = out[..., 2]
    out[..., 1] = np.where(ships > 0.0, _rotate_angle_pi_np(out[..., 1]), 0.0)
    return out


def _actions_to_kaggle_np(actions: np.ndarray) -> list[list[float | int]]:
    arr = np.asarray(actions[0])
    return [[float(r[0]), float(r[1]), int(r[2])] for r in arr if float(r[2]) > 0.0]


def _fleet_speed_np(ships: np.ndarray) -> np.ndarray:
    ratio = np.clip(np.log(np.maximum(ships, 1.0)) / np.log(1000.0), 0.0, 1.0)
    return np.minimum(1.0 + 5.0 * (ratio**1.5), 6.0).astype(np.float32, copy=False)


def _point_segment_distance_sq_np(px, py, x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    denom = np.maximum(dx * dx + dy * dy, 1.0e-9)
    t = np.clip(((px - x0) * dx + (py - y0) * dy) / denom, 0.0, 1.0)
    cx = x0 + t * dx
    cy = y0 + t * dy
    ddx = px - cx
    ddy = py - cy
    return ddx * ddx + ddy * ddy


def _comet_planet_mask_np(state: _StateArrays) -> np.ndarray:
    mask = np.zeros(state.planet_alive.shape, dtype=np.bool_)
    slots = state.comet_slots
    active = (slots >= 0) & (state.comet_path_index[:, :, None] >= -1)
    for b in range(slots.shape[0]):
        valid_slots = slots[b][active[b]]
        valid_slots = valid_slots[(valid_slots >= 0) & (valid_slots < mask.shape[1])]
        mask[b, valid_slots] = True
    return mask


def _target_max_turns_np(state: _StateArrays, target_idx: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full(target_idx.shape, horizon, dtype=np.float32)
    for b in range(target_idx.shape[0]):
        for i in range(target_idx.shape[1]):
            target = int(target_idx[b, i])
            best = 0
            found = False
            for g in range(_MAX_COMET_SPAWNS):
                if state.comet_path_index[b, g] < -1:
                    continue
                for c in range(_COMETS_PER_SPAWN):
                    if state.comet_slots[b, g, c] != target:
                        continue
                    remaining = int(state.comet_lengths[b, g, c] - state.comet_path_index[b, g])
                    best = max(best, remaining)
                    found = True
            if found:
                out[b, i] = float(min(horizon, max(0, best - 1)))
    return out


def _comet_pos_vel_at_time_np(state: _StateArrays, target_idx: np.ndarray, turns: np.ndarray, *, velocity: bool):
    out_shape = turns.shape
    px = np.zeros(out_shape, dtype=np.float32)
    py = np.zeros(out_shape, dtype=np.float32)
    vx = np.zeros(out_shape, dtype=np.float32)
    vy = np.zeros(out_shape, dtype=np.float32)
    valid = np.zeros(out_shape, dtype=np.bool_)
    flat_turns = turns.reshape(target_idx.shape + (-1,))
    for b in range(target_idx.shape[0]):
        for i in range(target_idx.shape[1]):
            target = int(target_idx[b, i])
            match = None
            for g in range(_MAX_COMET_SPAWNS):
                if state.comet_path_index[b, g] < -1:
                    continue
                for c in range(_COMETS_PER_SPAWN):
                    if state.comet_slots[b, g, c] == target and state.comet_lengths[b, g, c] > 0:
                        match = (g, c)
                        break
                if match is not None:
                    break
            if match is None:
                continue
            g, c = match
            path = state.comet_paths[b, g, c]
            length = int(state.comet_lengths[b, g, c])
            base = int(state.comet_path_index[b, g])
            for flat_j, turn in enumerate(flat_turns[b, i]):
                fi = base + float(turn)
                i0 = int(np.floor(fi))
                if i0 < 0 or i0 >= length:
                    continue
                frac = fi - float(i0)
                i1 = min(i0 + 1, length - 1)
                has_i1 = i0 + 1 < length
                pos0 = path[min(max(i0, 0), _MAX_COMET_PATH - 1)]
                pos1 = path[min(max(i1, 0), _MAX_COMET_PATH - 1)]
                pos = pos0 + frac * (pos1 - pos0) if has_i1 else pos0
                idx = np.unravel_index(flat_j, turns.shape[2:])
                full_idx = (b, i) + idx
                px[full_idx] = pos[0]
                py[full_idx] = pos[1]
                if velocity and has_i1:
                    vel = pos1 - pos0
                    vx[full_idx] = vel[0]
                    vy[full_idx] = vel[1]
                valid[full_idx] = True
    if velocity:
        return px, py, vx, vy, valid
    return px, py, valid


def _expand_like_time_np(x: np.ndarray, turns: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape + (1,) * (turns.ndim - x.ndim))


def _target_pos_vel_at_time_np(state: _StateArrays, target_idx: np.ndarray, turns: np.ndarray, *, velocity: bool):
    b_idx = np.arange(target_idx.shape[0])[:, None]
    planets = state.planets[b_idx, target_idx]
    initial = state.initial_planets[b_idx, target_idx]
    alive = state.planet_alive[b_idx, target_idx]

    base_x = _expand_like_time_np(planets[..., 2], turns)
    base_y = _expand_like_time_np(planets[..., 3], turns)
    dx = initial[..., 2] - 50.0
    dy = initial[..., 3] - 50.0
    orbit_r = np.sqrt(dx * dx + dy * dy)
    phi_cur = np.arctan2(planets[..., 3] - 50.0, planets[..., 2] - 50.0)
    omega = _expand_like_time_np(state.angular_velocity[:, None], turns)
    phi = _expand_like_time_np(phi_cur, turns) + omega * turns
    orbit_r_t = _expand_like_time_np(orbit_r, turns)
    rot_x = 50.0 + orbit_r_t * np.cos(phi)
    rot_y = 50.0 + orbit_r_t * np.sin(phi)
    comet_mask = _comet_planet_mask_np(state)
    is_comet = comet_mask[b_idx, target_idx]
    rotating = (orbit_r + initial[..., 4] < 50.0) & alive & (~is_comet)
    rotating_t = _expand_like_time_np(rotating, turns)
    px = np.where(rotating_t, rot_x, base_x)
    py = np.where(rotating_t, rot_y, base_y)
    valid = _expand_like_time_np(alive, turns)

    if velocity:
        zero = np.zeros_like(px)
        vx = np.where(rotating_t, -omega * orbit_r_t * np.sin(phi), zero)
        vy = np.where(rotating_t, omega * orbit_r_t * np.cos(phi), zero)
        comet_x, comet_y, comet_vx, comet_vy, comet_valid = _comet_pos_vel_at_time_np(state, target_idx, turns, velocity=True)
        is_comet_t = _expand_like_time_np(is_comet, turns)
        return (
            np.where(is_comet_t, comet_x, px).astype(np.float32, copy=False),
            np.where(is_comet_t, comet_y, py).astype(np.float32, copy=False),
            np.where(is_comet_t, comet_vx, vx).astype(np.float32, copy=False),
            np.where(is_comet_t, comet_vy, vy).astype(np.float32, copy=False),
            np.where(is_comet_t, comet_valid, valid),
        )

    comet_x, comet_y, comet_valid = _comet_pos_vel_at_time_np(state, target_idx, turns, velocity=False)
    is_comet_t = _expand_like_time_np(is_comet, turns)
    return (
        np.where(is_comet_t, comet_x, px).astype(np.float32, copy=False),
        np.where(is_comet_t, comet_y, py).astype(np.float32, copy=False),
        np.where(is_comet_t, comet_valid, valid),
    )


def _all_planet_pos_at_turns_np(state: _StateArrays, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    turns = np.arange(horizon + 1, dtype=np.float32)
    planets = state.planets
    initial = state.initial_planets
    dx = initial[..., 2] - 50.0
    dy = initial[..., 3] - 50.0
    orbit_r = np.sqrt(dx * dx + dy * dy)
    comet_mask = _comet_planet_mask_np(state)
    rotating = (orbit_r + initial[..., 4] < 50.0) & state.planet_alive & (~comet_mask)
    phi_cur = np.arctan2(planets[..., 3] - 50.0, planets[..., 2] - 50.0)
    phi = phi_cur[..., None] + state.angular_velocity[:, None, None] * turns
    px = np.where(rotating[..., None], 50.0 + orbit_r[..., None] * np.cos(phi), planets[..., None, 2])
    py = np.where(rotating[..., None], 50.0 + orbit_r[..., None] * np.sin(phi), planets[..., None, 3])

    for b in range(planets.shape[0]):
        for g in range(_MAX_COMET_SPAWNS):
            if state.comet_path_index[b, g] < -1:
                continue
            for c in range(_COMETS_PER_SPAWN):
                slot = int(state.comet_slots[b, g, c])
                length = int(state.comet_lengths[b, g, c])
                if slot < 0 or slot >= planets.shape[1] or length <= 0:
                    continue
                for t in range(horizon + 1):
                    pi = int(state.comet_path_index[b, g] + t)
                    if 0 <= pi < length:
                        px[b, slot, t] = state.comet_paths[b, g, c, pi, 0]
                        py[b, slot, t] = state.comet_paths[b, g, c, pi, 1]
    return px.astype(np.float32, copy=False), py.astype(np.float32, copy=False)


def _target_is_first_collision_np(state: _StateArrays, target_idx: np.ndarray, fx0, fy0, speed, cos_a, sin_a, hit_turn) -> np.ndarray:
    horizon = _AIM_HORIZON
    px_t, py_t = _all_planet_pos_at_turns_np(state, horizon)
    planets = state.planets
    alive = state.planet_alive
    speed = np.broadcast_to(speed, hit_turn.shape)
    out = np.ones(hit_turn.shape, dtype=np.bool_)
    large = 2 * (horizon + 1) + 1
    for idx in np.ndindex(hit_turn.shape):
        b = idx[0]
        target = int(target_idx[(b, idx[1])])
        first_planet = -1
        first_key = large
        in_bounds = True
        for t in range(1, horizon + 1):
            active = t <= int(hit_turn[idx])
            if not active:
                continue
            fx_prev = fx0[idx] + (float(t) - 1.0) * speed[idx] * cos_a[idx]
            fy_prev = fy0[idx] + (float(t) - 1.0) * speed[idx] * sin_a[idx]
            fx_cur = fx0[idx] + float(t) * speed[idx] * cos_a[idx]
            fy_cur = fy0[idx] + float(t) * speed[idx] * sin_a[idx]
            in_bounds = in_bounds and (0.0 <= fx_cur <= 100.0 and 0.0 <= fy_cur <= 100.0)
            for p in range(planets.shape[1]):
                if not alive[b, p]:
                    continue
                radius = planets[b, p, 4]
                move_d2 = _point_segment_distance_sq_np(px_t[b, p, t - 1], py_t[b, p, t - 1], fx_prev, fy_prev, fx_cur, fy_cur)
                move_hit = move_d2 < radius * radius
                sweep_active = (px_t[b, p, t] - px_t[b, p, t - 1]) ** 2 + (py_t[b, p, t] - py_t[b, p, t - 1]) ** 2 > 1.0e-9
                sweep_d2 = _point_segment_distance_sq_np(fx_cur, fy_cur, px_t[b, p, t - 1], py_t[b, p, t - 1], px_t[b, p, t], py_t[b, p, t])
                sweep_hit = sweep_active and sweep_d2 < radius * radius
                key = min(2 * t if move_hit else large, 2 * t + 1 if sweep_hit else large)
                if key < first_key:
                    first_key = key
                    first_planet = p
        lo = 2 * int(hit_turn[idx])
        hi = lo + 1
        out[idx] = in_bounds and first_key < large and first_planet == target and lo <= first_key <= hi
    return out


def _aim_with_prediction_np(src: np.ndarray, target: np.ndarray, target_idx: np.ndarray, state: _StateArrays, ships: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    horizon = _AIM_HORIZON
    clearance = 0.1
    speed = _fleet_speed_np(np.maximum(ships, 1.0))
    max_turns = _target_max_turns_np(state, target_idx, horizon)
    can_search = max_turns >= 1.0
    zeros = np.zeros_like(max_turns, dtype=np.float32)
    px0, py0, target0_valid = _target_pos_vel_at_time_np(state, target_idx, zeros, velocity=False)
    gap = src[..., 4] + clearance + target[..., 4]
    d0 = np.sqrt((px0 - src[..., 2]) ** 2 + (py0 - src[..., 3]) ** 2)
    t_star = np.clip((d0 - gap) / speed, 0.0, max_turns)

    for _ in range(_AIM_NEWTON_ITERS):
        px, py, vx, vy, pos_valid = _target_pos_vel_at_time_np(state, target_idx, t_star, velocity=True)
        rx = px - src[..., 2]
        ry = py - src[..., 3]
        dist = np.sqrt(np.maximum(rx * rx + ry * ry, 1.0e-9))
        f_val = dist - gap - speed * t_star
        radial_velocity = (rx * vx + ry * vy) / dist
        deriv = radial_velocity - speed
        raw_step = f_val / deriv
        max_step = np.minimum(np.maximum(max_turns, 1.0), _AIM_NEWTON_MAX_STEP)
        step = np.clip(raw_step, -max_step, max_step)
        t_newton = np.clip(t_star - step, 0.0, max_turns)
        use_newton = pos_valid & np.isfinite(t_newton) & np.isfinite(raw_step) & np.isfinite(deriv) & (np.abs(deriv) > _AIM_NEWTON_MIN_DERIV)
        t_star = np.where(use_newton, t_newton, t_star)

    has_solution = can_search & target0_valid
    t_floor = np.floor(t_star).astype(np.int32)
    candidates = np.stack([np.maximum(t_floor - 1, 1), np.maximum(t_floor, 1), t_floor + 1], axis=-1)
    candidate_turns = candidates.astype(np.float32)
    candidate_in_range = (candidate_turns >= 1.0) & (candidate_turns <= max_turns[..., None])
    t_star_for_candidates = np.broadcast_to(t_star[..., None], candidate_turns.shape)
    aim_turns = np.stack([t_star_for_candidates, np.maximum(candidate_turns - 1.0, 0.0), candidate_turns], axis=-1)
    aim_px, aim_py, aim_pos_valid = _target_pos_vel_at_time_np(state, target_idx, aim_turns, velocity=False)
    angle = np.arctan2(aim_py - src[..., None, None, 3], aim_px - src[..., None, None, 2])
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    candidate_turns_e = candidate_turns[..., None]
    candidates_e = candidates[..., None]
    fx0 = src[..., None, None, 2] + cos_a * (src[..., None, None, 4] + clearance)
    fy0 = src[..., None, None, 3] + sin_a * (src[..., None, None, 4] + clearance)
    fx_t = fx0 + candidate_turns_e * speed[..., None, None] * cos_a
    fy_t = fy0 + candidate_turns_e * speed[..., None, None] * sin_a
    sun_hit = _point_segment_distance_sq_np(50.0, 50.0, fx0, fy0, fx_t, fy_t) < 100.0
    first_collision_ok = _target_is_first_collision_np(state, target_idx, fx0, fy0, speed[..., None, None], cos_a, sin_a, candidates_e)
    fx_prev = fx0 + (candidate_turns_e - 1.0) * speed[..., None, None] * cos_a
    fy_prev = fy0 + (candidate_turns_e - 1.0) * speed[..., None, None] * sin_a
    px_old, py_old, target_pos_valid = _target_pos_vel_at_time_np(state, target_idx, candidate_turns_e - 1.0, velocity=False)
    move_d2 = _point_segment_distance_sq_np(px_old, py_old, fx_prev, fy_prev, fx_t, fy_t)
    target_hit_radius = np.maximum(target[..., None, None, 4] - _AIM_TARGET_INSET, 0.0)
    move_hits_target = move_d2 < target_hit_radius**2
    px_next, py_next, target_next_valid = _target_pos_vel_at_time_np(state, target_idx, candidate_turns_e, velocity=False)
    target_moved = (px_next - px_old) ** 2 + (py_next - py_old) ** 2 > 1.0e-9
    sweep_d2 = _point_segment_distance_sq_np(fx_t, fy_t, px_old, py_old, px_next, py_next)
    sweep_hits_target = target_moved & target_next_valid & (sweep_d2 < target_hit_radius**2)
    candidate_valid = (
        has_solution[..., None, None]
        & candidate_in_range[..., None]
        & aim_pos_valid
        & target_pos_valid
        & (~sun_hit)
        & first_collision_ok
        & (move_hits_target | sweep_hits_target)
        & (ships[..., None, None] > 0.0)
    )
    candidate_valid_flat = candidate_valid.reshape(candidate_valid.shape[:-2] + (9,))
    angle_flat = angle.reshape(angle.shape[:-2] + (9,))
    has_valid = candidate_valid_flat.any(axis=-1)
    chosen = np.argmax(candidate_valid_flat, axis=-1)
    chosen_angle = np.take_along_axis(angle_flat, chosen[..., None], axis=-1).squeeze(-1)
    direct_angle = np.arctan2(target[..., 3] - src[..., 3], target[..., 2] - src[..., 2])
    return np.where(has_valid, chosen_angle, direct_angle).astype(np.float32, copy=False), has_valid


def _import_onnxruntime():
    if sys.version_info[:2] == (3, 11):
        import onnxruntime as ort
        return ort

    existing = sys.modules.get("onnxruntime")
    if existing is not None and hasattr(existing, "SessionOptions"):
        return existing
    for name in list(sys.modules):
        if name == "onnxruntime" or name.startswith("onnxruntime."):
            del sys.modules[name]

    removed: list[tuple[int, str]] = []
    for entry in list(sys.path):
        if not entry:
            continue
        try:
            capi = Path(entry) / "onnxruntime" / "capi"
        except TypeError:
            continue
        if any(capi.glob("onnxruntime_pybind11_state.cpython-311-*.so")):
            while entry in sys.path:
                idx = sys.path.index(entry)
                removed.append((idx, sys.path.pop(idx)))
    try:
        import onnxruntime as ort
        if not hasattr(ort, "SessionOptions"):
            raise ImportError("imported onnxruntime without SessionOptions")
        return ort
    finally:
        for idx, entry in sorted(removed):
            sys.path.insert(min(idx, len(sys.path)), entry)


class _OnnxRuntime:
    def __init__(self, root: Path):
        ort = _import_onnxruntime()
        _drop_foreign_jax_agent_modules()

        import jax
        import jax.numpy as jnp

        from jax_agent.features_jax import FeatureParams, encode_batch
        from jax_agent.fast_env import EnvParams, EnvState, comet_arrays_from_observation

        self._jax = jax
        self._jnp = jnp
        self._FeatureParams = FeatureParams
        self._EnvParams = EnvParams
        self._EnvState = EnvState
        self._comet_arrays_from_observation = comet_arrays_from_observation
        self._encode_batch = None

        manifest = json.loads((root / "models" / "manifest.json").read_text(encoding="utf-8"))
        self.checkpoints: list[_CheckpointRuntime] = []
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.log_severity_level = 3
        providers = ["CPUExecutionProvider"]

        for item in manifest["checkpoints"]:
            ckpt_dir = root / "models" / Path(item["checkpoint"]).stem
            if _ACTIVE_2P_CHECKPOINTS and ckpt_dir.name not in _ACTIVE_2P_CHECKPOINTS:
                continue
            meta = json.loads((ckpt_dir / "meta.json").read_text(encoding="utf-8"))
            norm_npz = np.load(ckpt_dir / "normalizer.npz")
            norm_state = {key: np.asarray(norm_npz[key], dtype=np.float32) for key in norm_npz.files}
            self.checkpoints.append(
                _CheckpointRuntime(
                    name=Path(item["checkpoint"]).stem,
                    meta=meta,
                    norm_state=norm_state,
                    norm_clip=float(meta["state_norm_clip"]),
                    norm_eps=float(meta["state_norm_eps"]),
                    start=ort.InferenceSession(str(ckpt_dir / "auto_start.onnx"), sess_options, providers=providers),
                    step=ort.InferenceSession(str(ckpt_dir / "auto_step_base.onnx"), sess_options, providers=providers),
                    tgt=ort.InferenceSession(str(ckpt_dir / "auto_tgt_logits.onnx"), sess_options, providers=providers),
                    tok=ort.InferenceSession(str(ckpt_dir / "auto_token_embed.onnx"), sess_options, providers=providers),
                )
            )
        if not self.checkpoints:
            raise RuntimeError("no active ONNX checkpoints loaded")

        meta0 = self.checkpoints[0].meta
        self.max_planets = int(meta0["max_planets"])
        self.max_action_tokens = int(meta0["max_action_tokens"])
        self.env_params = EnvParams(
            episode_steps=int(meta0["episode_steps"]),
            max_planets=self.max_planets,
        )
        self.feat_params = FeatureParams(
            max_planets=self.max_planets,
            max_ships=float(meta0["max_ships"]),
            max_production=float(meta0["max_production"]),
            episode_steps=int(meta0["episode_steps"]),
            target_filter_enabled=bool(meta0["target_filter_enabled"]),
            attack_top_k=int(meta0["attack_top_k"]),
            defense_top_k=int(meta0["defense_top_k"]),
            max_fleet_tokens=int(meta0["max_fleet_tokens"]),
            fleet_selection=str(meta0.get("fleet_selection", "size")),
        )
        self._encode_batch = jax.jit(lambda state: encode_batch(state, self.feat_params))

    def _state_arrays_from_observation(self, observation: Any) -> _StateArrays:
        p_count = self.env_params.max_planets
        f_count = self.env_params.max_fleets
        planets = np.zeros((p_count, 7), dtype=np.float32)
        planet_alive = np.zeros((p_count,), dtype=np.bool_)
        planet_id_to_slot: dict[int, int] = {}
        for row in _rows(observation, "planets"):
            pid = int(_row_value(row, 0, "id", -1))
            if 0 <= pid < p_count:
                planets[pid] = [
                    pid,
                    int(_row_value(row, 1, "owner", -1)),
                    float(_row_value(row, 2, "x", 0.0)),
                    float(_row_value(row, 3, "y", 0.0)),
                    float(_row_value(row, 4, "radius", 0.0)),
                    float(_row_value(row, 5, "ship_count", 0.0) or 0.0),
                    float(_row_value(row, 6, "production", 0.0) or 0.0),
                ]
                planet_alive[pid] = True
                planet_id_to_slot[pid] = pid

        initial_planets = planets.copy()
        for row in _rows(observation, "initial_planets"):
            pid = int(_row_value(row, 0, "id", -1))
            if 0 <= pid < p_count:
                initial_planets[pid] = [
                    pid,
                    int(_row_value(row, 1, "owner", -1)),
                    float(_row_value(row, 2, "x", 0.0)),
                    float(_row_value(row, 3, "y", 0.0)),
                    float(_row_value(row, 4, "radius", 0.0)),
                    float(_row_value(row, 5, "ship_count", 0.0) or 0.0),
                    float(_row_value(row, 6, "production", 0.0) or 0.0),
                ]

        fleets = np.zeros((f_count, 7), dtype=np.float32)
        fleet_alive = np.zeros((f_count,), dtype=np.bool_)
        for i, row in enumerate(_rows(observation, "fleets")[:f_count]):
            fleets[i] = [
                int(_row_value(row, 0, "id", i)),
                int(_row_value(row, 1, "owner", -1)),
                float(_row_value(row, 2, "x", 0.0)),
                float(_row_value(row, 3, "y", 0.0)),
                float(_row_value(row, 4, "angle", 0.0)),
                int(_row_value(row, 5, "from_planet_id", -1)),
                float(_row_value(row, 6, "ship_count", 0.0) or 0.0),
            ]
            fleet_alive[i] = True

        comet_paths, comet_lengths, comet_ships, comet_valid, comet_path_index, comet_slots = self._comet_arrays_from_observation(
            observation,
            planet_id_to_slot,
            planets,
        )
        return _StateArrays(
            planets=planets[None],
            initial_planets=initial_planets[None],
            planet_alive=planet_alive[None],
            fleets=fleets[None],
            fleet_alive=fleet_alive[None],
            step=np.asarray([int(_read_obs_field(observation, "step", 0) or 0)], dtype=np.int32),
            angular_velocity=np.asarray([float(_read_obs_field(observation, "angular_velocity", 0.0) or 0.0)], dtype=np.float32),
            next_fleet_id=np.asarray([int(_read_obs_field(observation, "next_fleet_id", len(_rows(observation, "fleets"))) or 0)], dtype=np.int32),
            done=np.zeros((1,), dtype=np.bool_),
            winner=np.full((1,), -1, dtype=np.int32),
            overflow_fleets=np.zeros((1,), dtype=np.int32),
            comet_paths=comet_paths[None],
            comet_lengths=comet_lengths[None],
            comet_ships=comet_ships[None],
            comet_valid=comet_valid[None],
            comet_path_index=comet_path_index[None],
            comet_slots=comet_slots[None],
        )

    def _state_arrays_to_jax(self, state: _StateArrays):
        return self._EnvState(
            planets=self._jnp.asarray(state.planets, dtype=self._jnp.float32),
            initial_planets=self._jnp.asarray(state.initial_planets, dtype=self._jnp.float32),
            planet_alive=self._jnp.asarray(state.planet_alive, dtype=bool),
            fleets=self._jnp.asarray(state.fleets, dtype=self._jnp.float32),
            fleet_alive=self._jnp.asarray(state.fleet_alive, dtype=bool),
            step=self._jnp.asarray(state.step, dtype=self._jnp.int32),
            angular_velocity=self._jnp.asarray(state.angular_velocity, dtype=self._jnp.float32),
            next_fleet_id=self._jnp.asarray(state.next_fleet_id, dtype=self._jnp.int32),
            done=self._jnp.asarray(state.done, dtype=bool),
            winner=self._jnp.asarray(state.winner, dtype=self._jnp.int32),
            overflow_fleets=self._jnp.asarray(state.overflow_fleets, dtype=self._jnp.int32),
            comet_paths=self._jnp.asarray(state.comet_paths, dtype=self._jnp.float32),
            comet_lengths=self._jnp.asarray(state.comet_lengths, dtype=self._jnp.int32),
            comet_ships=self._jnp.asarray(state.comet_ships, dtype=self._jnp.float32),
            comet_valid=self._jnp.asarray(state.comet_valid, dtype=bool),
            comet_path_index=self._jnp.asarray(state.comet_path_index, dtype=self._jnp.int32),
            comet_slots=self._jnp.asarray(state.comet_slots, dtype=self._jnp.int32),
        )

    def _as_np(self, value, dtype=None) -> np.ndarray:
        arr = np.asarray(self._jax.device_get(value))
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def _encoded_to_np(self, encoded) -> dict[str, np.ndarray]:
        return {
            "planet_features": self._as_np(encoded.planet_features, np.float32),
            "relative_features": self._as_np(encoded.relative_features, np.float32),
            "global_features": self._as_np(encoded.global_features, np.float32),
            "map_features": self._as_np(encoded.map_features, np.float32),
            "candidate_mask": self._as_np(encoded.candidate_mask, np.bool_),
            "source_ids": self._as_np(encoded.source_ids, np.int32),
            "source_ships": self._as_np(encoded.source_ships, np.float32),
            "candidate_ids": self._as_np(encoded.candidate_ids, np.int32),
            "source_mask": self._as_np(encoded.source_mask, np.bool_),
            "owned_mask": self._as_np(encoded.owned_mask, np.bool_),
            "fleet_features": self._as_np(encoded.fleet_features, np.float32),
            "fleet_mask": self._as_np(encoded.fleet_mask, np.bool_),
        }

    def _normalize_encoded_np(self, encoded: dict[str, np.ndarray], ckpt: _CheckpointRuntime) -> dict[str, np.ndarray]:
        if not bool(ckpt.meta["state_norm_enabled"]):
            return encoded

        out = dict(encoded)
        planet_features = np.array(encoded["planet_features"], dtype=np.float32, copy=True)
        relative_features = np.array(encoded["relative_features"], dtype=np.float32, copy=True)
        global_features = np.array(encoded["global_features"], dtype=np.float32, copy=True)

        planet_values = planet_features[..., _PLANET_NORMALIZE_INDICES]
        planet_norm = (planet_values - ckpt.norm_state["planet_mean"]) / np.sqrt(ckpt.norm_state["planet_var"] + ckpt.norm_eps)
        planet_features[..., _PLANET_NORMALIZE_INDICES] = np.clip(planet_norm, -ckpt.norm_clip, ckpt.norm_clip)

        rel_values = relative_features[..., _RELATIVE_NORMALIZE_INDICES]
        rel_norm = (rel_values - ckpt.norm_state["relative_mean"]) / np.sqrt(ckpt.norm_state["relative_var"] + ckpt.norm_eps)
        rel_norm = np.clip(rel_norm, -ckpt.norm_clip, ckpt.norm_clip)
        real_candidates = (encoded["candidate_ids"] >= 0)[..., None]
        relative_features[..., _RELATIVE_NORMALIZE_INDICES] = np.where(
            real_candidates,
            rel_norm,
            relative_features[..., _RELATIVE_NORMALIZE_INDICES],
        )

        global_values = global_features[..., _GLOBAL_NORMALIZE_INDICES]
        global_norm = (global_values - ckpt.norm_state["global_mean"]) / np.sqrt(ckpt.norm_state["global_var"] + ckpt.norm_eps)
        global_features[..., _GLOBAL_NORMALIZE_INDICES] = np.clip(global_norm, -ckpt.norm_clip, ckpt.norm_clip)

        out["planet_features"] = planet_features
        out["relative_features"] = relative_features
        out["global_features"] = global_features
        return out

    def _encoded_inputs(self, encoded: dict[str, np.ndarray], initial_ships: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "planet_features": encoded["planet_features"],
            "relative_features": encoded["relative_features"],
            "global_features": encoded["global_features"],
            "candidate_mask": encoded["candidate_mask"],
            "initial_ships": initial_ships.astype(np.float32, copy=False),
            "map_features": encoded["map_features"],
            "fleet_features": encoded["fleet_features"],
            "fleet_mask": encoded["fleet_mask"],
        }

    def act(self, observation: Any) -> list[list[float | int]]:
        player = int(_read_field(observation, "player", 0) or 0)
        state_np = self._state_arrays_from_observation(observation)
        if player == 1:
            state_np = _flip_state_arrays_np(state_np)

        state = self._state_arrays_to_jax(state_np)
        raw_encoded = self._encode_batch(state)
        raw_encoded_np = self._encoded_to_np(raw_encoded)
        states = []
        encoded_for_actions = None
        for ckpt in self.checkpoints:
            encoded = self._normalize_encoded_np(raw_encoded_np, ckpt)
            if encoded_for_actions is None:
                encoded_for_actions = encoded
            initial_ships = np.concatenate(
                [
                    np.zeros((1, 1), dtype=np.float32),
                    encoded["source_ships"],
                ],
                axis=1,
            )
            out = ckpt.start.run(None, self._encoded_inputs(encoded, initial_ships))
            states.append(
                {
                    "ckpt": ckpt,
                    "encoded": encoded,
                    "state_emb": out[0],
                    "h": out[1],
                    "prev_tok": out[2],
                    "remaining": out[3],
                    "planet_out": out[4],
                    "relative_h": out[5],
                    "key_proj": out[6],
                    "src_key_proj": out[7],
                }
            )

        assert encoded_for_actions is not None
        actions = np.zeros((1, self.max_planets, 3), dtype=np.float32)
        ended = np.zeros((1,), dtype=np.bool_)
        batch_idx = np.arange(1)
        source_ids_np = encoded_for_actions["source_ids"]

        for step_i in range(self.max_action_tokens):
            step_outputs = []
            for st in states:
                out = st["ckpt"].step.run(
                    None,
                    {
                        "state_emb": st["state_emb"],
                        "h": st["h"],
                        "prev_tok": st["prev_tok"],
                        "remaining": st["remaining"],
                        "relative_features": st["encoded"]["relative_features"],
                        "source_mask": st["encoded"]["source_mask"],
                        "src_key_proj": st["src_key_proj"],
                    },
                )
                step_outputs.append(out)
                st["h_new"] = out[0]

            tv = ~ended
            end_logits = np.mean([out[1] for out in step_outputs], axis=0)
            src_logits = np.mean([out[2] for out in step_outputs], axis=0)
            no_valid_source = ~_valid_logit_mask(src_logits).any(axis=-1)
            end_t = tv & ((_sigmoid(end_logits) > 0.5) | no_valid_source)
            src_t = np.argmax(_safe_logits(src_logits), axis=-1).astype(np.int32)
            src_for_heads = np.where(tv & (~end_t), src_t, 0).astype(np.int32)

            tgt_logits_list = []
            for st in states:
                tgt_logits_list.append(
                    st["ckpt"].tgt.run(
                        None,
                        {
                            "h_new": st["h_new"],
                            "src_t": src_for_heads,
                            "planet_out": st["planet_out"],
                            "relative_h": st["relative_h"],
                            "key_proj": st["key_proj"],
                            "candidate_mask": st["encoded"]["candidate_mask"],
                        },
                    )[0]
                )
            tgt_logits = np.mean(tgt_logits_list, axis=0)
            tgt_t = np.argmax(_safe_logits(tgt_logits), axis=-1).astype(np.int32)
            tgt_for_heads = np.where(tv & (~end_t), tgt_t, 0).astype(np.int32)

            src_slot = np.clip(src_t - 1, 0, self.max_planets - 1).astype(np.int32)
            tgt_slot = np.clip(tgt_t - 1, 0, self.max_planets - 1).astype(np.int32)
            ships = _full_send_amount_np(states[0]["remaining"][batch_idx, np.clip(src_t, 0, self.max_planets)])
            src_planets = state_np.planets[batch_idx, src_slot]
            tgt_planets = state_np.planets[batch_idx, tgt_slot]
            angle_np, aim_hit_np = _aim_with_prediction_np(
                src_planets[:, None, :],
                tgt_planets[:, None, :],
                tgt_slot[:, None],
                state_np,
                ships[:, None],
            )
            source_mask_np = encoded_for_actions["source_mask"]
            src_valid = (src_t > 0) & source_mask_np[batch_idx, src_slot] & (ships >= _MIN_LAUNCH_SHIPS)
            tgt_valid = tgt_t > 0
            exec_t = tv & (~end_t) & src_valid & tgt_valid & aim_hit_np[:, 0]

            if bool(exec_t[0]):
                actions[0, step_i, 0] = float(source_ids_np[0, src_slot[0]])
                actions[0, step_i, 1] = float(angle_np[0, 0])
                actions[0, step_i, 2] = float(ships[0])

            for st in states:
                if bool(exec_t[0]):
                    st["remaining"][0, int(np.clip(src_t[0], 0, self.max_planets))] -= float(ships[0])
                st["prev_tok"] = st["ckpt"].tok.run(
                    None,
                    {
                        "src_t": src_for_heads,
                        "tgt_t": tgt_for_heads,
                        "end_t": end_t.astype(np.bool_),
                        "executed_t": exec_t.astype(np.bool_),
                        "step_i": np.asarray(step_i, dtype=np.int32),
                        "planet_out": st["planet_out"],
                    },
                )[0]
                st["h"] = st["h_new"]
            ended = ended | end_t
            if bool(np.all(ended)):
                break

        actions_out = actions
        if player == 1:
            actions_out = _flip_actions_np(actions)
        return _actions_to_kaggle_np(actions_out)


def _load_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        with _RUNTIME_LOCK:
            if _RUNTIME is None:
                _RUNTIME = _OnnxRuntime(_HERE)
    return _RUNTIME


def _start_preload() -> None:
    global _PRELOAD_THREAD
    if _RUNTIME is not None:
        return
    if _PRELOAD_THREAD is not None and _PRELOAD_THREAD.is_alive():
        return
    _PRELOAD_THREAD = threading.Thread(target=_load_runtime, daemon=True)
    _PRELOAD_THREAD.start()


def agent(observation: Any, configuration: Any = None) -> list[list[float | int]]:
    global _ERROR_LOGGED
    try:
        _set_agent_dir(configuration)
        if _is_four_player(observation, configuration):
            return _load_tamrazov().agent(observation, configuration)
        _start_preload()
        if not _has_sendable_source(observation):
            return []
        return _load_runtime().act(observation)
    except Exception:
        if not _ERROR_LOGGED:
            traceback.print_exc(file=sys.stderr)
            _ERROR_LOGGED = True
        raise


__all__ = ["agent"]
