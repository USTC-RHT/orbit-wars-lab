"""NumPy inference agent for ckpt_000900.

This submission path avoids importing JAX/Flax/Torch/ONNX Runtime at runtime.
Feature encoding and action decoding reuse the lightweight Python
implementation in src/.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(globals().get("__file__", ".")).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.config import EnvConfig
from src.features import (
    GLOBAL_DIM,
    DecisionContext,
    TurnBatch,
    encode_turn,
)
from src.normalization import CANDIDATE_NORMALIZE_INDICES, SELF_NORMALIZE_INDICES
from src.physics import aim_with_prediction
from src.state import GameState

_PARAMS: dict[str, np.ndarray] | None = None
_CFG: EnvConfig | None = None
_NORM: dict[str, np.ndarray] | None = None
_NORM_ENABLED = True
_NORM_CLIP = 5.0
_NORM_EPS = 1.0e-8
_ERROR_LOGGED = False


def _set_agent_dir(configuration: Any = None) -> None:
    global _HERE
    raw_path = None
    if isinstance(configuration, dict):
        raw_path = configuration.get("__raw_path__")
    elif configuration is not None:
        raw_path = getattr(configuration, "__raw_path__", None)
    if raw_path:
        _HERE = Path(raw_path).resolve().parent
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))


def _load() -> None:
    global _PARAMS, _CFG, _NORM, _NORM_ENABLED, _NORM_CLIP, _NORM_EPS
    if _PARAMS is not None:
        return

    raw_cfg = json.loads((_HERE / "agent_config.json").read_text(encoding="utf-8"))
    _CFG = EnvConfig(
        episode_steps=int(raw_cfg["episode_steps"]),
        ship_bucket_count=int(raw_cfg["ship_bucket_count"]),
        max_planets=int(raw_cfg["max_planets"]),
        max_ships=float(raw_cfg["max_ships"]),
        max_production=float(raw_cfg["max_production"]),
        target_filter_enabled=bool(raw_cfg["target_filter_enabled"]),
        attack_top_k=int(raw_cfg["attack_top_k"]),
        defense_top_k=int(raw_cfg["defense_top_k"]),
    )
    _NORM_ENABLED = bool(raw_cfg.get("state_norm_enabled", True))
    _NORM_CLIP = float(raw_cfg.get("state_norm_clip", 5.0))
    _NORM_EPS = float(raw_cfg.get("state_norm_eps", 1.0e-8))
    norm_npz = np.load(_HERE / "normalizer.npz")
    _NORM = {k: np.asarray(norm_npz[k], dtype=np.float32) for k in norm_npz.files}
    params_npz = np.load(_HERE / "model_params.npz")
    _PARAMS = {k: np.asarray(params_npz[k], dtype=np.float32) for k in params_npz.files}


def _normalize(batch: TurnBatch) -> TurnBatch:
    if not _NORM_ENABLED or _NORM is None or batch.self_features.shape[0] == 0:
        return batch

    self_features = batch.self_features.astype(np.float32, copy=True)
    candidate_features = batch.candidate_features.astype(np.float32, copy=True)
    global_features = batch.global_features.astype(np.float32, copy=True)

    self_features[:, SELF_NORMALIZE_INDICES] = _clip_norm(
        self_features[:, SELF_NORMALIZE_INDICES],
        _NORM["self_mean"],
        _NORM["self_var"],
    )
    real_planets = candidate_features[..., 1] > 0.5
    if real_planets.any():
        real_features = candidate_features[real_planets]
        real_features[:, CANDIDATE_NORMALIZE_INDICES] = _clip_norm(
            real_features[:, CANDIDATE_NORMALIZE_INDICES],
            _NORM["candidate_mean"],
            _NORM["candidate_var"],
        )
        candidate_features[real_planets] = real_features
    global_features[:, np.arange(GLOBAL_DIM)] = _clip_norm(
        global_features[:, np.arange(GLOBAL_DIM)],
        _NORM["global_mean"],
        _NORM["global_var"],
    )
    return TurnBatch(
        self_features=self_features,
        candidate_features=candidate_features,
        global_features=global_features,
        map_features=batch.map_features,
        candidate_mask=batch.candidate_mask,
        amount_mask=batch.amount_mask,
        contexts=batch.contexts,
        state=batch.state,
    )


def _clip_norm(values: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    values = (values - mean.astype(np.float32)) / np.sqrt(var.astype(np.float32) + _NORM_EPS)
    return np.clip(values, -_NORM_CLIP, _NORM_CLIP)


def _dense(x: np.ndarray, name: str) -> np.ndarray:
    assert _PARAMS is not None
    return x @ _PARAMS[f"{name}__kernel"] + _PARAMS[f"{name}__bias"]


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _layer_norm(x: np.ndarray, name: str) -> np.ndarray:
    assert _PARAMS is not None
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(var + 1.0e-6)
    return x * _PARAMS[f"{name}__scale"] + _PARAMS[f"{name}__bias"]


def _mlp(x: np.ndarray, prefix: str) -> np.ndarray:
    x = _dense(x, f"{prefix}__Dense_0")
    x = _layer_norm(x, f"{prefix}__LayerNorm_0")
    x = _relu(x)
    x = _dense(x, f"{prefix}__Dense_1")
    return _relu(x)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), 1.0e-12))
    return x / denom


def _run_model(batch: TurnBatch, selected_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert _PARAMS is not None
    self_features = batch.self_features.astype(np.float32, copy=False)
    candidate_features = batch.candidate_features.astype(np.float32, copy=False)
    global_features = batch.global_features.astype(np.float32, copy=False)
    candidate_mask = batch.candidate_mask.astype(bool, copy=False)

    self_h = _mlp(self_features, "MLP_0")
    cand_h = _mlp(candidate_features, "MLP_1")
    global_h = _mlp(global_features, "MLP_2")

    planet_mask = candidate_features[..., 1] > 0.5
    if planet_mask.shape[1] > 0:
        fallback = np.zeros_like(planet_mask)
        fallback[:, 0] = True
        safe_mask = np.where(planet_mask.any(axis=1, keepdims=True), planet_mask, fallback)
    else:
        safe_mask = planet_mask
    masked = np.where(safe_mask[..., None], cand_h, np.finfo(np.float32).min)
    max_pool = masked.max(axis=1)
    denom = np.maximum(safe_mask.sum(axis=1, keepdims=True).astype(np.float32), 1.0)
    mean_pool = (cand_h * safe_mask[..., None].astype(np.float32)).sum(axis=1) / denom
    map_h = 0.5 * (max_pool + mean_pool)
    context_h = np.concatenate([self_h, global_h, map_h], axis=-1)

    q = _relu(_dense(context_h, "Dense_0"))
    q = _dense(q, "Dense_1")
    q = _l2_normalize(q)
    keys = _dense(cand_h, "Dense_2")
    keys = keys.copy()
    keys[:, 0, :] = _PARAMS["noop_key"]
    keys = _l2_normalize(keys)
    target_logits = np.einsum("bh,bch->bc", q, keys) / np.sqrt(np.float32(keys.shape[-1]))
    target_logits = np.where(candidate_mask, target_logits, -1.0e9)

    fire_logits = _dense(_relu(_dense(context_h, "Dense_3")), "Dense_4").squeeze(-1)

    selected_targets = np.clip(selected_targets.astype(np.int64, copy=False), 0, cand_h.shape[1] - 1)
    selected_h = cand_h[np.arange(cand_h.shape[0]), selected_targets]
    amount_h = np.concatenate([context_h, selected_h], axis=-1)
    amount_logits = _dense(_relu(_dense(amount_h, "Dense_5")), "Dense_6")
    return fire_logits.astype(np.float32), target_logits.astype(np.float32), amount_logits.astype(np.float32)


def _actions_from_policy(batch: TurnBatch) -> list[list[float | int]]:
    if batch.self_features.shape[0] == 0:
        return []

    batch = _normalize(batch)
    n = batch.self_features.shape[0]
    zeros = np.zeros((n,), dtype=np.int64)
    fire_logits, target_logits, _amount_logits = _run_model(batch, zeros)

    has_target = batch.candidate_mask.copy()
    has_target[:, 0] = False
    fire = (1.0 / (1.0 + np.exp(-fire_logits)) > 0.5) & has_target.any(axis=1)

    target_mask = batch.candidate_mask.copy()
    target_mask[:, 0] = False
    masked_target_logits = np.where(target_mask, target_logits, -1.0e9)
    masked_target_logits = np.where(fire[:, None], masked_target_logits, -1.0e9)
    masked_target_logits[:, 0] = np.where(fire, masked_target_logits[:, 0], 0.0)
    targets = np.argmax(masked_target_logits, axis=1).astype(np.int64)

    _fire2, _target2, amount_logits = _run_model(batch, targets)
    row = np.arange(n)
    selected_amount_mask = batch.amount_mask[row, np.clip(targets, 0, batch.amount_mask.shape[1] - 1)]
    masked_amount_logits = np.where(selected_amount_mask, amount_logits, -1.0e9)
    no_amount = ~selected_amount_mask.any(axis=1)
    masked_amount_logits[no_amount] = 0.0
    amounts = np.argmax(masked_amount_logits, axis=1).astype(np.int64)

    moves: list[list[float | int]] = []
    for row_idx, context in enumerate(batch.contexts):
        if not bool(fire[row_idx]):
            continue
        move = _action_from_indices(batch.state, context, int(targets[row_idx]), int(amounts[row_idx]))
        if move is not None:
            moves.append(move)
    return moves


def _action_from_indices(
    state: GameState,
    context: DecisionContext,
    action_idx: int,
    amount_idx: int,
) -> list[float | int] | None:
    if action_idx == 0 or action_idx >= len(context.candidate_ids):
        return None
    if not context.candidate_mask[action_idx]:
        return None
    if amount_idx < 0 or amount_idx >= len(context.ship_buckets):
        return None
    if not context.amount_mask[action_idx, amount_idx]:
        return None
    ships = int(context.ship_buckets[amount_idx])
    if ships <= 0:
        return None

    planets_by_id = {planet.id: planet for planet in state.planets}
    src = planets_by_id.get(context.source_id)
    target = planets_by_id.get(context.candidate_ids[action_idx])
    if src is None or target is None:
        return None
    if target.owner != state.player and ships <= target.ships:
        return None

    initial_by_id = {planet.id: planet for planet in state.initial_planets}
    result = aim_with_prediction(
        src,
        target,
        ships,
        initial_by_id,
        state.angular_velocity,
        state.comets,
        set(state.comet_planet_ids),
        state.planets,
    )
    if result is None:
        return None
    angle, _eta, _hit_x, _hit_y = result
    if not math.isfinite(angle):
        return None
    return [context.source_id, float(angle), ships]


def agent(observation: Any, configuration: Any) -> list[list[float | int]]:
    global _ERROR_LOGGED
    try:
        _set_agent_dir(configuration)
        _load()
        assert _CFG is not None
        return _actions_from_policy(encode_turn(observation, _CFG))
    except Exception:
        if not _ERROR_LOGGED:
            traceback.print_exc(file=sys.stderr)
            _ERROR_LOGGED = True
        return []
