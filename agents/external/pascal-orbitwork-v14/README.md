# pascal-orbitwork-v14 Agent

## 策略概述

该系列代码库中**功能最齐全、最激进的版本**。在 pilkwang-structured 的框架上新增了 4 源群攻、等待-出击时机协调、更长的预测视野，以及更强的 4 人局进攻策略。

---

## 框架来源

代码结构与 pilkwang-structured 完全相同。请先阅读 [pilkwang-structured/README.md](../pilkwang-structured/README.md) 了解核心框架。

本文档仅说明 v14 的**独有特性和参数差异**。

---

## 独有特性 1：Wait-Strike（等待出击）

### 问题背景
有时候现在出发无法确保胜利，但等待 2~4 回合让源星球多积累些舰船，或者等待敌方舰队先到达制造混战，再出发成功率会更高。

### 实现

```python
WAIT_STRIKE_DELAYS = (0, 2, 4, 6)  # 尝试延迟 0 / 2 / 4 / 6 回合出发

# 对每个 capture/swarm 任务：
for delay in WAIT_STRIKE_DELAYS:
    simulated_ships_at_launch = current_ships + production * delay
    simulated_arrival_turn = travel_turns + delay
    if simulated_ships_at_launch > target_at_arrival(simulated_arrival_turn):
        best_delay = delay
        break  # 取最短满足条件的延迟
```

**效果**：
- 不再盲目出发，而是找到**最优出击时间窗口**
- 可以利用敌方舰队碰撞后的防守空窗
- 避免仅差 1~2 艘船导致任务失败的情况

---

## 独有特性 2：4 源群攻（Four-Source Swarm）

pilkwang 最多协调 3 颗星球同时到达目标，v14 将上限提升至 **4 颗**：

```python
FOUR_SOURCE_SWARM_ENABLED = True
FOUR_SOURCE_SWARM_MIN_TARGET_SHIPS = 40  # 目标驻守 ≥ 40 艘才考虑 4 源
```

**适用场景**：目标是一个防守强大的敌方核心星球，单打独斗和 3 源群攻都不够，需要 4 路并进。

---

## 独有特性 3：更长的预测视野（HORIZON=180）

| 参数 | pilkwang | v14 | 效果 |
|------|---------|-----|------|
| `ROUTE_SEARCH_HORIZON` | 110 | **180** | 提前预测更远未来的星球归属 |

更长视野使得：
- 更准确地计算"目标未来 N 回合的归属"
- 在终局阶段做出更正确的"生产价值权重"计算
- 更好地规划星球链式控制（先拿 A 才能高效拿 B）

---

## 独有特性 4：更激进的 4 人局开局

```python
FOUR_PLAYER_ROTATING_SEND_RATIO = 0.72  # vs pilkwang 的 0.60
```

4 人局开局阶段，出征占领旋转星球时，发送己方可用兵力的 72%（而不是 60%），更快占领旋转星球，形成生产优势。

---

## 独有特性 5：彗星追击延伸

```python
MAX_COMET_CHASE_TURNS = 15  # vs pilkwang 的 10
```

愿意追逐距离更远（最多 15 回合路程）的彗星行星，避免错过高价值移动星球。

---

## 其他重要参数差异

| 参数 | pilkwang | v14 | 效果 |
|------|---------|-----|------|
| `MULTI_SOURCE_TOP_K` | 5 | **10** | 群攻评分时考虑更多候选源星球组合 |
| `HOSTILE_TARGET_VALUE_MULT` | 1.85 | **2.0** | 进攻敌方星球价值更高 |
| `GANG_UP_POST_BATTLE_DELAY` | 不存在 | **3** | 利用战场碰撞机会（延迟 3 回合） |

---

## 决策执行流程

```
1. 构建 World（HORIZON=180 的更长物理预测）
2. 生成所有任务候选（含 4-source swarm 候选）
3. 对每个候选任务尝试 4 种延迟（0/2/4/6），选最优出发时机
4. 按评分排序，贪心分配资源
5. 翻译为 moves（考虑延迟出发则不立刻生成 move，等到对应回合）
6. 时间截止控制（0.82s）
```

---

## 总结

v14 是该系列演化的最终形态，三大进步：
1. **Wait-Strike** — 不盲目出发，等最优时机
2. **4 源群攻** — 攻下任何坚固目标的终极手段
3. **更长视野** — 计划更深远，终局不迷路
