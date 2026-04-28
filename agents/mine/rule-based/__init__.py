"""拆分版 Pilkwang Structured agent。

入口仍然是 agent(obs, config=None)，模块拆分只改变代码组织，不改变策略逻辑。
"""

from .main import agent, build_world

__all__ = ["agent", "build_world"]
