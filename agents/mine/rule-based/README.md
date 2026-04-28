# Pilkwang Structured Split

这是 `orbit-wars-lab/agents/external/pilkwang-structured/main.py` 的拆分阅读版。
目标是保持策略逻辑不变，只把一个长文件拆成更清晰的模块。

## 模块关系

```text
main.py
  -> utils.py
  -> models.py
  -> world_model.py
       -> physics.py
       -> config.py
  -> strategy.py
       -> config.py
       -> models.py
       -> world_model.py

physics.py
  -> config.py

models.py
  无业务依赖

config.py
  无业务依赖
```

## 文件说明

- `config.py`：所有全局常量和调参参数，比如评分倍率、阶段阈值、时间预算。
- `models.py`：共享数据结构，包括 `Planet`、`Fleet`、`ShotOption`、`Mission`。
- `physics.py`：物理和瞄准逻辑，包括距离、舰队速度、太阳碰撞、旋转星球/彗星位置预测、ETA 估算。
- `world_model.py`：未来沙盘，负责到达账本、单星球时间线、最少派兵数、防守需求。
- `strategy.py`：策略和评分，负责生成任务、目标价值、派兵余量、贪心执行。
- `utils.py`：小工具，目前只有兼容 dict/对象读取的 `_read`。
- `main.py`：入口，提供 `build_world(obs)` 和 `agent(obs, config=None)`。

## 使用方式

作为包导入：

```python
from pilkwang_structured_split import agent
```

或者直接加载 `pilkwang_structured_split/main.py`，该文件包含导入兜底逻辑。

## 阅读顺序

建议按这个顺序看：

1. `config.py`
2. `models.py`
3. `physics.py`
4. `world_model.py`
5. `strategy.py`
6. `main.py`

