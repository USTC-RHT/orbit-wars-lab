#!/usr/bin/env python3
"""
逐步对比：JAX Flax 直接推理 vs onnx-hybrid numpy 版
对同一序列 observation，两路各自推理，比较每步动作是否一致。

用法：
  cd /home/dministrator/kaggle/orbit-wars-lab
  /home/dministrator/miniconda/envs/orbit-wars-lab/bin/python _cmp_jax_vs_onnx.py
"""
import sys, pickle, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

AGENT_DIR = Path(__file__).parent / "agents/mine/fixed-pool-tgtmask-0526-ckpt4800-onnx-hybrid"
PKL_PATH  = Path("/home/dministrator/kaggle/checkpoint/fixed_pool_with_tgtmask-0526_jax/ckpt_004800.pkl")

sys.path.insert(0, str(AGENT_DIR))

from jax_agent.fast_env import EnvParams
from jax_agent.features_jax import FeatureParams, encode_batch
from jax_agent.normalization_jax import normalize_encoded
from jax_agent.model_jax import AutoregressiveActorCritic
from jax_agent.train_jax import (
    jax_actions_to_kaggle, safe_logits,
    aim_with_prediction_jax, dynamic_ship_buckets,
    kaggle_obs_to_jax_state, flip_state, flip_actions,
)
from main import _OnnxEnsembleRuntime

# ── 加载 onnx-hybrid ────────────────────────────────────────────────────────
print("加载 onnx-hybrid runtime...")
ort_rt = _OnnxEnsembleRuntime(AGENT_DIR)
feat_params = ort_rt.feat_params
ckpt0 = ort_rt.checkpoints[0]
meta = ckpt0.meta
norm_state = ckpt0.norm_state
norm_params = ckpt0.norm_params

# ── 加载 JAX 参数 ─────────────────────────────────────────────────────────────
print("加载 JAX 参数...")
with open(PKL_PATH, "rb") as f:
    ckpt_pkl = pickle.load(f)
params = ckpt_pkl["params"]

model = AutoregressiveActorCritic(
    planet_dim=26, relative_dim=8, global_dim=11,
    relative_count=feat_params.max_planets + 1,
    ship_bucket_count=feat_params.ship_bucket_count,
    hidden_size=int(meta["hidden_size"]),
    use_cnn=bool(meta.get("use_cnn", False)),
    max_fleet_tokens=feat_params.max_fleet_tokens,
    num_transformer_layers=int(meta.get("num_transformer_layers", 2)),
    num_heads=int(meta.get("num_heads", 4)),
    max_action_tokens=int(meta["max_action_tokens"]),
    end_bias_init=float(meta.get("end_bias_init", -2.0)),
)

def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

def jax_act(obs: dict) -> list:
    """用 JAX Flax 模型直接推理，返回 kaggle 格式动作。"""
    player = int(obs.get("player", 0) or 0)
    state = kaggle_obs_to_jax_state(obs, ort_rt.env_params)
    if player == 1:
        state = flip_state(state)

    enc_raw = encode_batch(state, feat_params)
    encoded = normalize_encoded(enc_raw, norm_state, norm_params)

    b = 1
    initial_ships = jnp.concatenate([jnp.zeros((b,1), dtype=jnp.float32), encoded.source_ships], axis=1)
    jax_out = model.apply(
        {"params": params},
        encoded.planet_features, encoded.relative_features, encoded.global_features,
        encoded.candidate_mask, initial_ships,
        map_features=encoded.map_features, fleet_features=encoded.fleet_features, fleet_mask=encoded.fleet_mask,
        method=model.auto_start,
    )
    state_emb, h, prev_tok, remaining, planet_out, relative_h, key_proj, src_key_proj = jax_out[:8]

    P = feat_params.max_planets
    K = int(meta["max_action_tokens"])
    actions = np.zeros((1, P, 3), dtype=np.float32)
    ended = np.zeros((1,), dtype=bool)
    batch_idx = np.arange(1)
    source_ids_np = np.asarray(encoded.source_ids)
    source_mask_np = np.asarray(encoded.source_mask, dtype=bool)
    remaining_np = np.array(np.asarray(remaining), dtype=np.float32)

    for step_i in range(K):
        h_new, end_logits_t, src_logits_t = model.apply(
            {"params": params},
            state_emb, h, prev_tok, remaining,
            encoded.relative_features, encoded.source_mask, src_key_proj,
            method=model.auto_step_base,
        )
        tv = ~ended
        end_l = np.asarray(end_logits_t).reshape(-1)
        src_logits_np = np.asarray(src_logits_t)
        end_t = tv & (_sigmoid(end_l) > 0.5)
        src_t = np.argmax(np.where(np.isfinite(src_logits_np), src_logits_np, -1e9), axis=-1).astype(np.int32)
        src_for_heads = np.where(tv & (~end_t), src_t, 0).astype(np.int32)

        tgt_logits_t = model.apply(
            {"params": params},
            h_new, jnp.asarray(src_for_heads), planet_out, relative_h, key_proj,
            encoded.candidate_mask, method=model.auto_tgt_logits,
        )
        tgt_logits_np = np.asarray(tgt_logits_t)
        tgt_t = np.argmax(np.where(np.isfinite(tgt_logits_np), tgt_logits_np, -1e9), axis=-1).astype(np.int32)
        tgt_for_heads = np.where(tv & (~end_t), tgt_t, 0).astype(np.int32)

        amt_logits_t = model.apply(
            {"params": params},
            h_new, jnp.asarray(src_for_heads), jnp.asarray(tgt_for_heads), planet_out,
            relative_h, remaining, encoded.amount_mask, method=model.auto_amt_logits,
        )
        amt_logits_np = np.asarray(amt_logits_t)
        amt_t = np.argmax(np.where(np.isfinite(amt_logits_np), amt_logits_np, -1e9), axis=-1).astype(np.int32)

        src_slot = np.clip(src_t - 1, 0, P-1).astype(np.int32)
        tgt_slot = np.clip(tgt_t - 1, 0, P-1).astype(np.int32)

        source_buckets = np.asarray(dynamic_ship_buckets(
            jnp.asarray(remaining_np[batch_idx, np.clip(src_t, 0, P)]),
            feat_params.ship_bucket_count,
        ))
        ships = np.take_along_axis(source_buckets, amt_t[:, None], axis=-1).squeeze(-1)

        src_planets = np.asarray(state.planets)[batch_idx, src_slot]
        tgt_planets = np.asarray(state.planets)[batch_idx, tgt_slot]
        angle_jax, hit_jax = aim_with_prediction_jax(
            jnp.asarray(src_planets[:,None,:]), jnp.asarray(tgt_planets[:,None,:]),
            jnp.asarray(tgt_slot[:,None]), state, jnp.asarray(ships[:,None], dtype=jnp.float32),
        )
        angle_np = np.asarray(angle_jax[:,0], dtype=np.float32)
        hit_np   = np.asarray(hit_jax[:,0],   dtype=bool)

        src_valid = (src_t > 0) & source_mask_np[batch_idx, src_slot]
        tgt_valid = tgt_t > 0
        exec_t = tv & (~end_t) & src_valid & tgt_valid & hit_np

        if bool(exec_t[0]):
            actions[0, step_i, 0] = float(source_ids_np[0, src_slot[0]])
            actions[0, step_i, 1] = float(angle_np[0])
            actions[0, step_i, 2] = float(ships[0])
            remaining_np[0, int(np.clip(src_t[0], 0, P))] -= float(ships[0])
            remaining = jnp.asarray(remaining_np)

        prev_tok = model.apply(
            {"params": params},
            jnp.asarray(src_for_heads), jnp.asarray(tgt_for_heads),
            jnp.asarray(np.where(tv & (~end_t), amt_t, 0).astype(np.int32)),
            jnp.asarray(end_t), jnp.asarray(exec_t),
            jnp.asarray(step_i, dtype=jnp.int32),
            planet_out, method=model.auto_token_embed,
        )
        h = h_new
        ended = ended | end_t
        if bool(np.all(ended)):
            break

    result = jax_actions_to_kaggle(jnp.asarray(actions))
    if player == 1:
        result = jax_actions_to_kaggle(flip_actions(jnp.asarray(actions)))
    return result


# ── 预热 ──────────────────────────────────────────────────────────────────────
print("预热 JAX jit (encode_batch)...")
from kaggle_environments import make
env = make("orbit_wars", configuration={"seed": 13, "randomSeed": 13}, debug=False)
env.reset(num_agents=2)
# 跑第一步拿到 obs 做预热
init_states = env.step([[], []])
obs0 = init_states[0]["observation"]
_ = jax_act(obs0)
print("  预热完成")

# ── 重新开局，逐步对比 ──────────────────────────────────────────────────────────
env = make("orbit_wars", configuration={"seed": 13, "randomSeed": 13}, debug=False)
env.reset(num_agents=2)
env.step([[], []])  # step 0

print("\n逐步对比（最多 50 步）：")
print(f"{'step':>4}  {'JAX动作':^45}  {'ORT动作':^45}  {'一致':^4}")
print("-" * 105)

matches = 0
mismatches = 0
steps_compared = 0

for step in range(50):
    obs0 = env.state[0]["observation"]
    obs1 = env.state[1]["observation"]

    # JAX 路
    jax_a0 = jax_act(obs0)
    jax_a1 = jax_act(obs1)

    # ORT 路
    ort_a0 = ort_rt.act(obs0)
    ort_a1 = ort_rt.act(obs1)

    # 对比 P0
    def fmt(acts): return str([(int(a[0]), round(float(a[1]),3), int(a[2])) for a in acts])[:43]
    match0 = (sorted((int(a[0]),round(float(a[1]),4),int(a[2])) for a in jax_a0) ==
              sorted((int(a[0]),round(float(a[1]),4),int(a[2])) for a in ort_a0))
    match1 = (sorted((int(a[0]),round(float(a[1]),4),int(a[2])) for a in jax_a1) ==
              sorted((int(a[0]),round(float(a[1]),4),int(a[2])) for a in ort_a1))

    if jax_a0 or ort_a0:  # 有动作的步骤才打印
        tag = "✅" if match0 else "❌"
        print(f"{step+1:>4}  P0 JAX:{fmt(jax_a0)}  ORT:{fmt(ort_a0)}  {tag}")
        steps_compared += 1
        if match0: matches += 1
        else: mismatches += 1

    # 推进游戏（用 ORT 动作驱动，JAX 只负责对比）
    env.step([ort_a0, ort_a1])
    if env.state[0].get("status") == "DONE":
        print(f"  DONE at step {step+1}")
        break

print(f"\n{'='*50}")
print(f"有动作步数: {steps_compared}  一致: {matches}  不一致: {mismatches}")
print(f"一致率: {matches/max(steps_compared,1)*100:.1f}%")
