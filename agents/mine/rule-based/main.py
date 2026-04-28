"""拆分版 Kaggle 入口。

外部只需要调用 agent(obs, config=None)。
"""

import time
import sys
from pathlib import Path

try:
    # Kaggle 的 get_last_callable 会把这个文件当作“裸 Python 文件”执行，
    # 那时全局变量里可能没有 __name__/__package__，直接写相对导入会报
    # KeyError，而不是常见的 ImportError。这里先显式判断：只有真正作为
    # rule_based 包导入时，才使用 from .xxx 形式。
    if globals().get("__package__"):
        from .config import SOFT_ACT_DEADLINE
        from .models import Fleet, Planet
        from .strategy import plan_moves
        from .utils import _read
        from .world_model import WorldModel
    else:
        raise ImportError
except (ImportError, KeyError):
    # 作为 Kaggle 提交文件/本地脚本直接执行时，把当前目录加入 sys.path，
    # 再按普通模块导入同目录下拆分出来的文件。
    module_dir = Path(globals().get("__file__", ".")).resolve().parent
    sys.path.insert(0, str(module_dir))
    from config import SOFT_ACT_DEADLINE
    from models import Fleet, Planet
    from strategy import plan_moves
    from utils import _read
    from world_model import WorldModel


def build_world(obs):
    """把环境 obs 转成 WorldModel。

    raw_planets/raw_fleets 是列表形式，这里转换成命名元组，后续代码可用
    planet.x、fleet.angle 这类可读字段访问。
    """
    player = _read(obs, "player", 0)
    step = _read(obs, "step", 0) or 0
    raw_planets = _read(obs, "planets", []) or []
    raw_fleets = _read(obs, "fleets", []) or []
    ang_vel = _read(obs, "angular_velocity", 0.0) or 0.0
    raw_init = _read(obs, "initial_planets", []) or []
    comets = _read(obs, "comets", []) or []
    comet_ids = set(_read(obs, "comet_planet_ids", []) or [])

    planets = [Planet(*planet) for planet in raw_planets]
    fleets = [Fleet(*fleet) for fleet in raw_fleets]
    initial_planets = [Planet(*planet) for planet in raw_init]
    initial_by_id = {planet.id: planet for planet in initial_planets}

    return WorldModel(
        player=player,
        step=step,
        planets=planets,
        fleets=fleets,
        initial_by_id=initial_by_id,
        ang_vel=ang_vel,
        comets=comets,
        comet_ids=comet_ids,
    )


def agent(obs, config=None):
    """Kaggle 调用入口：构建世界模型并返回本回合动作列表。"""
    start_time = time.perf_counter()
    world = build_world(obs)
    if not world.my_planets:
        return []
    act_timeout = _read(config, "actTimeout", 1.0) if config is not None else 1.0
    soft_budget = min(SOFT_ACT_DEADLINE, max(0.55, act_timeout * 0.82))
    deadline = start_time + soft_budget
    return plan_moves(world, deadline=deadline)


__all__ = ["agent", "build_world"]
