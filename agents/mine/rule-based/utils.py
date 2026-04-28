"""通用小工具：几何计算、游戏公式、以及 obs 读取兼容层。"""

import math

try:
    from .config import (
        CENTER_X, CENTER_Y, ROTATION_LIMIT, MAX_SPEED,
        SUN_R, LAUNCH_CLEARANCE,
    )
except ImportError:
    from config import (
        CENTER_X, CENTER_Y, ROTATION_LIMIT, MAX_SPEED,
        SUN_R, LAUNCH_CLEARANCE,
    )


def _read(obs, key, default=None):
    """兼容 dict 和对象两种 obs/config 读取方式。"""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


# ─────────────────────────────────────────────────────────────────────────────
# 距离 / 星球属性
# ─────────────────────────────────────────────────────────────────────────────

def dist(ax, ay, bx, by):
    """计算棋盘上两个坐标点之间的欧氏距离。"""
    return math.hypot(ax - bx, ay - by)


def orbital_radius(planet):
    """计算星球到地图中心的距离，用来判断它是否处于旋转轨道。"""
    return dist(planet.x, planet.y, CENTER_X, CENTER_Y)


def is_static_planet(planet):
    """判断星球是否为静止星球：不在内圈旋转轨道内就视为静止。"""
    return orbital_radius(planet) + planet.radius >= ROTATION_LIMIT


# ─────────────────────────────────────────────────────────────────────────────
# 游戏公式
# ─────────────────────────────────────────────────────────────────────────────

def fleet_speed(ships):
    """根据游戏规则计算舰队速度。

    舰队越大飞得越快，但对数公式会限制最大速度并产生边际递减。
    这很关键：多派船不仅会提高战斗力，也可能改变到达时间。
    """
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


# ─────────────────────────────────────────────────────────────────────────────
# 几何 / 太阳碰撞检测
# ─────────────────────────────────────────────────────────────────────────────

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """计算点到线段的最短距离。"""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return dist(px, py, proj_x, proj_y)


def segment_hits_sun(x1, y1, x2, y2):
    """判断一条直线路径是否会穿过太阳真实半径。

    Kaggle 环境使用的规则是：
        point_to_segment_distance(sun_center, old_pos, new_pos) < SUN_RADIUS

    这里不再添加额外安全余量，保持和环境碰撞判定一致。
    """
    return point_to_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_R


def path_clear(sx, sy, sr, tx, ty, tr):
    """判断从来源到目标的直线路径是否安全（不穿过太阳真实半径）。

    sr/tr 分别是来源和目标的半径，路径从来源边缘算起。
    """
    angle = math.atan2(ty - sy, tx - sx)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    offset = sr + LAUNCH_CLEARANCE
    x0 = sx + cos_a * offset
    y0 = sy + sin_a * offset
    fly_dist = max(0.0, math.hypot(tx - sx, ty - sy) - offset - tr)
    x1 = x0 + cos_a * fly_dist
    y1 = y0 + sin_a * fly_dist
    return not segment_hits_sun(x0, y0, x1, y1)
