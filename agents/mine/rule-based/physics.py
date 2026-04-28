"""物理和瞄准模块。

使用参数方程（来自 orbit_wars.py）描述旋转星球和彗星轨迹，
通过数值扫描求解舰队命中时刻和发射角度。
"""

import math

try:
    from .config import *
    from .utils import (
        dist, fleet_speed, is_static_planet, point_to_segment_distance,
        segment_hits_sun, LAUNCH_CLEARANCE,
    )
except ImportError:
    from config import *
    from utils import (
        dist, fleet_speed, is_static_planet, point_to_segment_distance,
        segment_hits_sun, LAUNCH_CLEARANCE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 参数方程：旋转星球位置
# ─────────────────────────────────────────────────────────────────────────────

def planet_pos(planet, initial_by_id, ang_vel, T):
    """旋转星球在 T 回合后的位置（T=0 为当前）。

    参数方程（来自 orbit_wars.py）：
      x = CENTER + r * cos(φ_cur + ω * T)
      y = CENTER + r * sin(φ_cur + ω * T)

    其中 φ_cur = atan2(planet.y - CENTER, planet.x - CENTER)，
    r = hypot(init.x - CENTER, init.y - CENTER)。
    仅当 r + init.radius < ROTATION_LIMIT（50.0）时旋转；否则返回当前坐标。
    """
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    dx = init.x - CENTER_X
    dy = init.y - CENTER_Y
    r = math.hypot(dx, dy)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    phi_cur = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    phi = phi_cur + ang_vel * T
    return CENTER_X + r * math.cos(phi), CENTER_Y + r * math.sin(phi)


# ─────────────────────────────────────────────────────────────────────────────
# 参数方程：彗星位置
# ─────────────────────────────────────────────────────────────────────────────

def comet_pos(planet_id, comets, T):
    """彗星在 T 回合后的位置（T=0 为当前），支持浮点 T。

    彗星路径由 orbit_wars.py generate_comet_paths 按等弧长（comet_speed
    单位/步）重采样生成，相邻两点距离恒为 comet_speed。对浮点 T 做线性
    插值，等价于彗星在相邻路径点间匀速飞行，与引擎物理完全一致。
    整数 T 时退化为直接查表，无精度损失。
    """
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        pi = group.get("path_index", 0)
        if idx >= len(paths):
            return None
        path = paths[idx]
        fi = pi + T
        i0 = int(fi)
        frac = fi - i0
        if i0 < 0 or i0 >= len(path):
            return None
        x0, y0 = path[i0][0], path[i0][1]
        if frac == 0.0 or i0 + 1 >= len(path):
            return x0, y0
        x1, y1 = path[i0 + 1][0], path[i0 + 1][1]
        return x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)
    return None


def comet_remaining_life(planet_id, comets):
    """彗星剩余路径长度（回合数）。"""
    for group in comets:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        pi = group.get("path_index", 0)
        if idx < len(paths):
            return max(0, len(paths[idx]) - pi)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 内部：按目标类型分发位置预测
# ─────────────────────────────────────────────────────────────────────────────

def _target_pos(target, initial_by_id, ang_vel, comets, comet_ids, T):
    """彗星走路径表，旋转星球走参数方程。"""
    if target.id in comet_ids:
        return comet_pos(target.id, comets, T)
    return planet_pos(target, initial_by_id, ang_vel, T)


# ─────────────────────────────────────────────────────────────────────────────
# 粗扫描 + 二分法求解命中时刻
# ─────────────────────────────────────────────────────────────────────────────

# 每次 aim_with_prediction 都会在策略评分里被调用很多次。
# 3000/50 很精细，但完整对局会明显变慢；固定 seed 测试里 180/16
# 与 600/24 的整数命中回合和角度结果一致，同时完整局耗时明显下降。
# 180 个粗扫点覆盖最多 60 回合，约每 0.33 回合采样一次；后续还会
# 二分精化并做整数回合碰撞验证，所以通常足够稳定。
_N_COARSE = 180   # 粗扫描采样点数
_N_BISECT = 16    # 二分迭代次数


def _aim_f(src, target, initial_by_id, ang_vel, comets, comet_ids, speed, T):
    """瞄准函数 f(T)：正值=舰队未到达目标圆，负值=已进入目标圆（命中）。

    f(T) = dist(src_center, P(T)) − D(T) − target.radius
    其中 D(T) = src.radius + LAUNCH_CLEARANCE + T * speed
    （直接瞄准 P(T) 发射时，舰队端点到 src 中心的距离）。

    推导：舰队端点 = src + D(T) * (P(T)−src)/L(T)，
    因此 |端点 − P(T)| = |L(T) − D(T)|；
    令 f = L − D − target.radius 即可检测前沿触及的首次过零点。
    """
    P = _target_pos(target, initial_by_id, ang_vel, comets, comet_ids, T)
    if P is None:
        return math.inf
    L = math.hypot(P[0] - src.x, P[1] - src.y)
    D = src.radius + LAUNCH_CLEARANCE + T * speed
    return L - D - target.radius


def aim_with_prediction(src, target, ships, initial_by_id, ang_vel,
                        comets, comet_ids, planets=None):
    """粗扫描 + 二分法求发射角度与命中回合。

    步骤：
      1. 在 (0, max_turns] 均匀采样 _N_COARSE 个点，
         检测 f(T) 的第一个正→负符号变化
         （即舰队端点"穿入"目标圆前沿的连续时刻）。
      2. 二分精化：_N_BISECT 次迭代，精确定位连续时刻 T*。
      3. 整数精挑：对 floor(T*) 和 floor(T*)+1 两个候选整数回合，
         用引擎同款线段–圆碰撞几何（舰队移动阶段 + 星球扫过阶段）
         验证并返回首个命中的 (θ, T, hit_x, hit_y)。

    参数 planets 保留兼容接口，不影响本算法。
    """
    speed = fleet_speed(max(1, int(ships)))
    max_turns = min(HORIZON, ROUTE_SEARCH_HORIZON)
    if target.id in comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, comets) - 1))
    if max_turns < 1:
        return None

    def f(T):
        return _aim_f(src, target, initial_by_id, ang_vel, comets, comet_ids, speed, T)

    # ── 1. 粗扫描：寻找首个正→负符号变化 ─────────────────────────────────────
    dt = max_turns / _N_COARSE
    prev = f(1e-6)
    t_lo = t_hi = None
    for i in range(1, _N_COARSE + 1):
        t = i * dt
        cur = f(t)
        if prev > 0 and cur <= 0:      # 正 → 非正（命中）
            t_lo = (i - 1) * dt
            t_hi = t
            break
        prev = cur

    if t_lo is None:
        return None

    # ── 2. 二分精化 ───────────────────────────────────────────────────────────
    for _ in range(_N_BISECT):
        mid = (t_lo + t_hi) * 0.5
        if f(t_lo) * f(mid) <= 0:
            t_hi = mid
        else:
            t_lo = mid

    t_star = (t_lo + t_hi) * 0.5

    # ── 3. 整数精挑：T* 周围三个整数值 ──────────────────────────────────────
    T_floor = int(t_star)
    candidates = sorted({
        max(1, T_floor - 1), max(1, T_floor), T_floor + 1
    })
    candidates = [T for T in candidates if 1 <= T <= max_turns]

    for T in candidates:
        # 引擎舰队移动阶段：使用 T-1 回合的星球位置（移动前）
        P = _target_pos(target, initial_by_id, ang_vel, comets, comet_ids, T - 1)
        if P is None:
            continue
        px, py = P

        theta = math.atan2(py - src.y, px - src.x)
        cos_a = math.cos(theta)
        sin_a = math.sin(theta)

        # 舰队实际出发点（引擎：planet.radius + 0.1）
        fx0 = src.x + cos_a * (src.radius + LAUNCH_CLEARANCE)
        fy0 = src.y + sin_a * (src.radius + LAUNCH_CLEARANCE)

        # 舰队在 T 回合末的实际位置
        fx_T = fx0 + T * speed * cos_a
        fy_T = fy0 + T * speed * sin_a

        # 太阳碰撞检测
        if segment_hits_sun(fx0, fy0, fx_T, fy_T):
            continue

        # T-1 回合末舰队位置（本步线段的起点）
        fx_prev = fx0 + (T - 1) * speed * cos_a
        fy_prev = fy0 + (T - 1) * speed * sin_a

        # 舰队移动碰撞：线段 F(T-1) → F(T) 与目标圆
        if point_to_segment_distance(px, py, fx_prev, fy_prev, fx_T, fy_T) < target.radius:
            return theta, T, px, py

        # 星球扫过阶段：目标从 P(T-1) 扫到 P(T) 时是否经过 F_T
        P_next = _target_pos(target, initial_by_id, ang_vel, comets, comet_ids, T)
        if P_next is not None and P_next != (px, py):
            pnx, pny = P_next
            if point_to_segment_distance(fx_T, fy_T, px, py, pnx, pny) < target.radius:
                return theta, T, pnx, pny

    return None
