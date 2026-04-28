# Starter Agent

## 策略概述

Kaggle 官方提供的入门 agent，属于**静止星球狙击者**策略。每回合向最近的静止星球（不绕太阳转的行星）发送舰队。

## 执行流程

```
每回合 → 调用引擎内置 starter_agent → 找最近静止目标 → 发送舰队
```

## 实现方式

```python
from kaggle_environments.envs.orbit_wars.orbit_wars import starter_agent as agent
```

直接 re-export Kaggle 引擎内置的 starter agent，无任何自定义逻辑。

## 特点

- 只考虑**静止星球**（轨道半径 + 行星半径 ≥ 50，不随时间转动）
- 不做路线规划、不预测星球位置、不考虑防御
- 比随机 agent 强，但不会处理动态旋转星球

## 用途

- 比 random 稍强的基准线
- 验证基础进攻能力
