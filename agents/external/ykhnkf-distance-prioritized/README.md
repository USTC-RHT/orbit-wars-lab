# ykhnkf-distance-prioritized Agent（排行榜约 LB 1100）

## 策略概述

pilkwang-structured 同框架下的变体，**核心创新**是在进攻决策中引入了**敌方增援预测裕量**：进攻前预判敌方在我方舰队飞行途中会向目标星球追加多少援军，并提前将这部分兵力算入所需舰船数，使进攻更稳健。

---

## 框架来源

代码结构与 pilkwang-structured 完全相同。请先阅读 [pilkwang-structured/README.md](../pilkwang-structured/README.md) 了解核心框架。

本文档仅说明 ykhnkf 的**独有特性和参数差异**。

---

## 独有特性：敌方增援预测（Hostile Reinforce Margin）

### 问题背景
在 pilkwang 等基础 agent 中，占领一个星球所需舰船数 = 目标星球当前舰船数 + 1。
但如果目标星球在我方舰队飞行途中（10 回合路程）又从其他星球获得了增援，我方舰队到达时发现打不过，任务失败。

### 解决方案

```python
HOSTILE_REINFORCE_HORIZON = 8  # 预判未来 8 回合内的敌方增援

# 对每个进攻任务，额外加入增援预算：
hostile_reinforce_estimate = 预测目标在未来 8 回合内能接收到的敌方援军数量
extra_margin = min(hostile_reinforce_estimate * 0.25, 15)  # 取 25%，上限 15 艘
ships_needed_adjusted = ships_needed + extra_margin
```

**效果**：进攻时多带 25% 的预估敌援数（上限 15 艘），提高任务成功率，减少"飞到一半对方加了援军"的浪费。

---

## 参数对比（vs pilkwang-structured）

| 参数 | pilkwang | ykhnkf | 效果 |
|------|---------|--------|------|
| `HOSTILE_REINFORCE_HORIZON` | 不存在 | **8** | 预判 8 回合内敌方增援 |
| `HOSTILE_REINFORCE_MARGIN_RATIO` | 不存在 | **0.25** | 取预估增援的 25% 作为额外裕量 |
| `HOSTILE_REINFORCE_MARGIN_CAP` | 不存在 | **15** | 额外裕量上限 |
| `FINISHING_HOSTILE_SEND_BONUS` | 0 | **5** | 终局阶段强制多带 5 艘（更激进清场） |
| `AHEAD_ATTACK_MARGIN_BONUS` | 0.08 | **0.12** | 领先时进攻裕量更宽松 |
| `FINISHING_ATTACK_MARGIN_BONUS` | 0.08 | **0.12** | 终局进攻裕量更宽松 |

---

## 名称解读："distance-prioritized"

除了增援预测之外，ykhnkf 在目标优先级计算中**对近距离目标给予额外加分**（distance_bonus），优先进攻距离近的星球：
- 减少飞行途中被增援的风险
- 更快形成连片控制区域
- 与增援预测机制协同，共同降低进攻失败率

---

## 决策执行流程

（同 pilkwang-structured，唯一区别是在 `capture` 任务评分阶段加入 `hostile_reinforce_margin` 和 `distance_bonus`）

---

## 总结

ykhnkf-distance-prioritized 解决了 pilkwang 的一个关键缺陷：**盲目进攻，不考虑敌方会在路途中增援目标**。通过提前多带一点兵，使每次进攻都更有把握，减少无效消耗。
