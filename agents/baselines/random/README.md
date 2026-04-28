# Random Agent

## 策略概述

**完全随机**。每回合随机选择一个我方星球、随机选择一个目标方向和舰船数量，不做任何判断。

## 执行流程

```
每回合 → 直接调用引擎内置 random_agent → 返回随机 moves
```

## 实现方式

```python
from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent as agent
```

直接 re-export Kaggle 引擎内置的随机 agent，无任何自定义逻辑。

## 用途

- 作为最弱基准线，衡量其他 agent 的最低胜率门槛
- 用于调试对局框架是否正常工作
