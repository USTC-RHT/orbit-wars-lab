"""共享数据结构。

Planet/Fleet 是环境原始列表的具名包装；ShotOption/Mission 是策略层内部使用
的候选动作结构。
"""

from collections import namedtuple
from dataclasses import dataclass, field

Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
Fleet = namedtuple(
    "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
)


@dataclass(frozen=True)
class ShotOption:
    """一次具体的发兵候选：从一颗来源星球打向一颗目标星球。

    策略层会先生成很多 ShotOption，再把一个或多个 ShotOption 组合成
    Mission。单源进攻通常只有一个 option；多源合攻 swarm 会有两个或
    三个 option。

    score:
        这次发兵本身的吸引力。组合成 Mission 后，可能会重新计算组合分。
    needed:
        世界模型计算出的最低所需舰船数：到评估时刻必须至少派这么多，
        目标才会归我方。
    send_cap:
        这次发兵从 src_id 最多/倾向使用的舰船数。
    anchor_turn:
        时间锚点。用于 snipe 这类需要和敌舰同回合/近似同回合到达的任务。
    """
    score: float
    src_id: int
    target_id: int
    angle: float
    turns: int
    needed: int
    send_cap: int
    mission: str = "capture"
    anchor_turn: int | None = None


@dataclass
class Mission:
    """一个更高层的任务候选，可能由一次或多次发兵组成。

    plan_moves() 最后按 Mission 的 score 排序。某个 Mission 被接受后，
    其中的 option 会变成真实 move，并写入 planned_commitments。这样后续
    任务在模拟未来时，会把“我刚才已经决定发出的船”也算进去。
    """
    kind: str
    score: float
    target_id: int
    turns: int
    options: list[ShotOption] = field(default_factory=list)
