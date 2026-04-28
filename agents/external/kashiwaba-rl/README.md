# kashiwaba-rl Agent（强化学习 PPO）

## 策略概述

唯一的**端到端强化学习 agent**。没有手写规则，完全通过 PPO 自博弈训练学习策略。神经网络接收全局游戏状态，输出每颗己方星球的目标选择。

---

## 整体架构

```
游戏观察值 (obs)
      ↓
特征工程（固定维度向量）
      ↓
PlanetPolicy 神经网络（Attention + MLP）
      ↓
每颗己方星球的目标星球概率分布
      ↓
argmax 选择目标 → 计算角度 → 生成 moves
```

---

## 特征工程

将可变数量的星球/舰队状态编码为**固定维度向量**：

### `self_features`（我方星球特征，每颗星球 N 维）
- 当前舰船数、生产力
- 到太阳的距离、轨道角度
- 是否正在被攻击、周围我方兵力密度
- 星球坐标（归一化）

### `candidate_features`（候选目标特征，固定取最近 8 颗非我方星球）
- 目标舰船数、归属（中性/敌方）
- 与源星球的距离和角度
- 目标的生产力、是否旋转

### `global_features`（全局特征）
- 当前步数（归一化）
- 我方总舰船数 vs 敌方总舰船数
- 我方总生产力 vs 敌方总生产力
- 玩家数量（2P/4P）

---

## 神经网络结构（PlanetPolicy）

```
global_features ─────────────────────────────────────┐
                                                       ↓
self_features ──→ [MLP] ──→ [Self-Attention Layer] ──→ [MLP] ──→ 目标概率分布
                                    ↑
candidate_features ──→ [MLP] ──→ [Cross-Attention] ─┘
```

- **Cross-Attention**：每颗我方星球"关注"所有候选目标星球的特征
- **Self-Attention**：我方星球之间互相感知（协调多颗星球的目标选择）
- 输出为每颗我方星球 × 8 候选目标的 logits → softmax → 概率

---

## 训练方式（PPO 自博弈）

```yaml
algorithm: PPO
total_updates: 2000
num_envs: 8          # 并行游戏环境数
rollout_steps: 64    # 每次 rollout 收集 64 步

opponent: self       # 自博弈
self_play_update_interval: 50  # 每 50 次 update 更新对手权重

device: auto         # 自动选择 GPU（RTX 3070 Ti）
checkpoint_every: 50 # 每 50 次 update 保存一次
```

**自博弈流程**：
1. 当前 policy（student）与冻结的旧 policy（opponent）对战
2. 收集 rollout（state, action, reward, value, log_prob）
3. GAE 计算优势函数
4. 多 epoch PPO 更新 student policy
5. 每 50 次 update，将 student 权重复制给 opponent

---

## 推理流程（每回合）

```python
# 1. 从 obs 提取特征
self_feat, cand_feat, global_feat = featurize(obs, player_index)

# 2. 前向传播（不采样，用 argmax）
with torch.no_grad():
    logits = policy(self_feat, cand_feat, global_feat)
    target_indices = logits.argmax(dim=-1)  # 确定性策略

# 3. 生成 moves
for planet, target_idx in zip(my_planets, target_indices):
    target = candidates[target_idx]
    angle = atan2(target.y - planet.y, target.x - planet.x)
    moves.append([planet.id, angle, planet.ships // 2])  # 发送一半舰船
```

---

## 预训练权重

| 文件 | 说明 |
|------|------|
| `artifacts/orbit_wars_ppo/ckpt_002000.pt` | 本地训练 2000 步的最新权重 |
| `ckpt_002000.pt`（根目录） | Kaggle 上预训练的权重 |

加载方式：
```python
policy = PlanetPolicy(...)
checkpoint = torch.load("ckpt_002000.pt", map_location="cpu")
policy.load_state_dict(checkpoint["policy"])
```

---

## 与规则 agent 的对比

| 维度 | 规则 agent（如 tamrazov） | kashiwaba-rl |
|------|--------------------------|--------------|
| 策略来源 | 手写规则 + 参数调优 | 自博弈学习 |
| 太阳规避 | 明确判断 | 隐式学习（可能学到，也可能没有） |
| 多源协调 | 明确的群攻任务 | 通过 Self-Attention 隐式协调 |
| 解释性 | 高（可追溯每步决策原因） | 低（黑盒网络） |
| 上限 | 受限于设计者的规则完整性 | 理论上更高（但需要足够训练） |
| 推理速度 | 较慢（Python 模拟） | 快（GPU 矩阵运算） |

---

## 训练监控

```bash
# 查看训练进度
grep "update" /tmp/train.log | tail -10

# 绘制 loss + 收益曲线
cd agents/external/kashiwaba-rl
conda run -n orbit-wars-lab python plot_metrics.py
# 输出：artifacts/orbit_wars_ppo/training_curves.png
```
