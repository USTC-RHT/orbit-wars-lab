# yuriygreben-architect Agent

## 策略概述

pilkwang-structured / tamrazov 同框架下的**另一参数变体**。整体策略与 pilkwang 相同，重点差异在于更激进的**早期进攻**和更重视**消灭弱势对手**。

---

## 框架来源

代码结构与 pilkwang-structured 完全相同。请先阅读 [pilkwang-structured/README.md](../pilkwang-structured/README.md) 了解核心框架（WorldState、Mission、评分系统等）。

本文档仅说明 yuriygreben 的**参数特性和策略倾向**。

---

## 参数对比（vs pilkwang-structured）

| 参数 | pilkwang | yuriygreben | 效果 |
|------|---------|-------------|------|
| `HOSTILE_TARGET_VALUE_MULT` | 1.85 | **1.95** | 进攻敌方倾向略微更强 |
| `WEAK_ENEMY_THRESHOLD` | 40 | **60** | 更宽松地认定"弱势敌人"（船越少越应该打） |
| `ELIMINATION_BONUS` | 20.0 | **28.0** | 消灭某玩家后的额外奖励分更高 |
| `AHEAD_ATTACK_MARGIN_BONUS` | 0.08 | **0.12** | 我方领先时，进攻裕量更宽松 |
| `FINISHING_ATTACK_MARGIN_BONUS` | 0.08 | **0.10** | 终局阶段进攻裕量也更宽松 |

---

## 策略倾向解读

### 1. 更宽松的"弱势敌人"认定（WEAK_ENEMY_THRESHOLD=60）
pilkwang 只有当某玩家的总兵力 < 阈值时才认定为弱势。yuriygreben 将阈值提高到 60，意味着只要某方星球总兵力不占优（< 60），就会加大对该目标的进攻价值。效果：更早集中火力压垮一名玩家。

### 2. 更重视消灭（ELIMINATION_BONUS=28.0）
每当某个任务有望让某一玩家失去所有星球，评分额外加 28 分。比 pilkwang 的 20 分高出 40%，体现出"打死一个少一个敌人"的战略导向。

### 3. 领先时更大胆（AHEAD_ATTACK_MARGIN_BONUS=0.12）
当我方总兵力/生产力领先对手时，进攻裕量（ships_needed 的安全系数）放宽，可以用更少的优势兵力发动进攻，加速扩张。

---

## 决策执行流程

（完全同 pilkwang-structured，见该 README）

---

## 总结

yuriygreben-architect 的核心调参方向是：
> **见弱就打，打了就灭，领先时更激进扩张**

适合在己方已经占据优势的局面下扩大胜势，不擅长逆转劣势。
