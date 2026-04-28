"""世界模型模块。

把当前观测到的星球和飞行舰队整理成未来 110 回合的沙盘：
- 推断已在飞舰队的目标和 ETA
- 为每颗星模拟未来 owner/ships 时间线
- 查询某回合目标状态
- 计算最少派多少船能占领/守住
"""

import math
from collections import defaultdict

try:
    from .config import *
    from .physics import *
except ImportError:
    from config import *
    from physics import *

def fleet_target_planet(fleet, planets):
    """推断一支已在飞行中的舰队最终会撞上哪颗星球，以及几回合后到达。

    这里不会读取“舰队目标”字段，而是用舰队当前位置、飞行角度和速度，把
    它看成一条射线。射线最先命中的星球，就是世界模型认为的目标星球。
    """
    # Project in-flight fleets by ray-circle hit timing to build a usable
    # arrival ledger.
    best_planet = None
    best_time = 1e9
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)

    for planet in planets:
        dx = planet.x - fleet.x
        dy = planet.y - fleet.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        radius_sq = planet.radius * planet.radius
        if perp_sq >= radius_sq:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))
        turns = hit_d / speed
        if turns <= HORIZON and turns < best_time:
            best_time = turns
            best_planet = planet

    if best_planet is None:
        return None, None
    return best_planet, int(math.ceil(best_time))


def build_arrival_ledger(fleets, planets):
    """为每颗星球建立未来到达账本。

    返回格式大致是：
        planet_id -> [(eta, owner, ships), ...]

    这个账本只包含当前已经在飞的舰队，不包含敌人未来可能新发出的舰队。
    """
    arrivals_by_planet = {planet.id: [] for planet in planets}
    for fleet in fleets:
        target, eta = fleet_target_planet(fleet, planets)
        if target is None:
            continue
        arrivals_by_planet[target.id].append((eta, fleet.owner, int(fleet.ships)))
    return arrivals_by_planet


def resolve_arrival_event(owner, garrison, arrivals):
    """结算同一回合抵达同一星球的所有舰队。

    规则要点：
    1. 同一玩家的舰队先合并。
    2. 到达舰队里兵力最多的两方互相抵消。
    3. 剩下的一方如果和驻军同阵营，就是增援；否则和驻军战斗。

    返回新的 (owner, garrison)。
    """
    # Match the environment's same-turn combat order: aggregate by owner, let
    # the top two attackers cancel, then resolve the survivor against garrison.
    by_owner = {}
    for _, attacker_owner, ships in arrivals:
        by_owner[attacker_owner] = by_owner.get(attacker_owner, 0) + ships

    if not by_owner:
        return owner, max(0.0, garrison)

    sorted_players = sorted(by_owner.items(), key=lambda item: item[1], reverse=True)
    top_owner, top_ships = sorted_players[0]

    if len(sorted_players) > 1:
        second_ships = sorted_players[1][1]
        if top_ships == second_ships:
            survivor_owner = -1
            survivor_ships = 0
        else:
            survivor_owner = top_owner
            survivor_ships = top_ships - second_ships
    else:
        survivor_owner = top_owner
        survivor_ships = top_ships

    if survivor_ships <= 0:
        return owner, max(0.0, garrison)

    if owner == survivor_owner:
        return owner, garrison + survivor_ships

    garrison -= survivor_ships
    if garrison < 0:
        return survivor_owner, -garrison
    return owner, garrison


def normalize_arrivals(arrivals, horizon):
    """清理并排序到达事件，只保留模拟窗口内的正数舰队。"""
    events = []
    for turns, owner, ships in arrivals:
        if ships <= 0:
            continue
        eta = max(1, int(math.ceil(turns)))
        if eta > horizon:
            continue
        events.append((eta, owner, int(ships)))
    events.sort(key=lambda item: item[0])
    return events


def simulate_planet_timeline(planet, arrivals, player, horizon):
    """为单颗星球模拟未来 horizon 回合的归属和驻军数量。

    这是世界模型的核心。它从当前 owner/ships 开始，每回合先生产，再处理
    本回合到达的舰队，最终得到：
    - owner_at: 每回合星球属于谁
    - ships_at: 每回合星球上有多少船
    - keep_needed: 如果这是我方星球，当前至少要留下多少船才能守住模拟窗口
    - fall_turn: 如果会失守，首次失守回合
    - holds_full: 即使一艘不派走也是否能守住
    """
    # Build one reusable future timeline so defense, capture, and evacuation
    # all query the same state model.
    horizon = max(0, int(math.ceil(horizon)))
    events = normalize_arrivals(arrivals, horizon)
    by_turn = defaultdict(list)
    for item in events:
        by_turn[item[0]].append(item)

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    min_owned = garrison if owner == player else 0.0
    first_enemy = None
    fall_turn = None

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production

        group = by_turn.get(turn, [])
        prev_owner = owner
        if group:
            if prev_owner == player and first_enemy is None:
                if any(item[1] not in (-1, player) for item in group):
                    first_enemy = turn
            owner, garrison = resolve_arrival_event(owner, garrison, group)
            if prev_owner == player and owner != player and fall_turn is None:
                fall_turn = turn

        owner_at[turn] = owner
        ships_at[turn] = max(0.0, garrison)
        if owner == player:
            min_owned = min(min_owned, garrison)

    keep_needed = 0
    holds_full = True

    if planet.owner == player:

        def survives_with_keep(keep):
            """假设当前只留下 keep 艘船，检查未来窗口内是否还能守住。"""
            sim_owner = planet.owner
            sim_garrison = float(keep)
            for turn in range(1, horizon + 1):
                if sim_owner != -1:
                    sim_garrison += planet.production
                group = by_turn.get(turn, [])
                if group:
                    sim_owner, sim_garrison = resolve_arrival_event(sim_owner, sim_garrison, group)
                    if sim_owner != player:
                        return False
            return sim_owner == player

        if survives_with_keep(int(planet.ships)):
            lo, hi = 0, int(planet.ships)
            while lo < hi:
                mid = (lo + hi) // 2
                if survives_with_keep(mid):
                    hi = mid
                else:
                    lo = mid + 1
            keep_needed = lo
        else:
            holds_full = False
            keep_needed = int(planet.ships)

    return {
        "owner_at": owner_at,
        "ships_at": ships_at,
        "keep_needed": keep_needed,
        "min_owned": max(0, int(math.floor(min_owned))) if planet.owner == player else 0,
        "first_enemy": first_enemy,
        "fall_turn": fall_turn,
        "holds_full": holds_full,
        "horizon": horizon,
    }


def state_at_timeline(timeline, arrival_turn):
    """从一条星球时间线中读取指定回合的 (owner, ships)。"""
    turn = max(0, int(math.ceil(arrival_turn)))
    turn = min(turn, timeline["horizon"])
    owner = timeline["owner_at"].get(turn, timeline["owner_at"][timeline["horizon"]])
    ships = timeline["ships_at"].get(turn, timeline["ships_at"][timeline["horizon"]])
    return owner, max(0.0, ships)


def count_players(planets, fleets):
    """根据星球和飞行舰队中出现的 owner 估算当前玩家数量。"""
    owners = set()
    for planet in planets:
        if planet.owner != -1:
            owners.add(planet.owner)
    for fleet in fleets:
        owners.add(fleet.owner)
    return max(2, len(owners))


def nearest_distance_to_set(px, py, planets):
    """计算某个坐标到一组星球的最近距离；空集合返回极大值。"""
    if not planets:
        return 10**9
    return min(dist(px, py, planet.x, planet.y) for planet in planets)


def indirect_features(planet, planets, player):
    """计算目标星球周边的“间接价值”特征。

    一颗星不仅因为自己的 production 有价值，也因为它靠近哪些星：
    - 靠近我方星球：更容易互相支援
    - 靠近中立星球：有扩张跳板价值
    - 靠近敌方星球：有前线压制价值

    返回三项累计值：(friendly, neutral, enemy)。
    """
    friendly = 0.0
    neutral = 0.0
    enemy = 0.0
    for other in planets:
        if other.id == planet.id:
            continue
        d = dist(planet.x, planet.y, other.x, other.y)
        if d < 1:
            continue
        factor = other.production / (d + 12.0)
        if other.owner == player:
            friendly += factor
        elif other.owner == -1:
            neutral += factor
        else:
            enemy += factor
    return friendly, neutral, enemy


class WorldModel:
    """每回合临时构建的局势沙盘。

    WorldModel 不预测敌人的未来意图；它把当前观测到的星球、飞行舰队、
    彗星路径、旋转参数整理成可查询的数据结构。策略层通过它回答：
    - 某颗星未来某回合是谁的？
    - 我最少派多少船能在某回合占下目标？
    - 我方星球至少要留下多少船防守？
    - 某个来源到某个目标能不能安全命中，几回合到？
    """
    def __init__(self, player, step, planets, fleets, initial_by_id, ang_vel, comets, comet_ids):
        """整理观测数据，并预先模拟每颗星球的基础未来时间线。"""
        self.player = player
        self.step = step
        self.planets = planets
        self.fleets = fleets
        self.initial_by_id = initial_by_id
        self.ang_vel = ang_vel
        self.comets = comets
        self.comet_ids = set(comet_ids)

        self.planet_by_id = {planet.id: planet for planet in planets}
        self.my_planets = [planet for planet in planets if planet.owner == player]
        self.enemy_planets = [planet for planet in planets if planet.owner not in (-1, player)]
        self.neutral_planets = [planet for planet in planets if planet.owner == -1]
        self.static_neutral_planets = [
            planet for planet in self.neutral_planets if is_static_planet(planet)
        ]

        self.num_players = count_players(planets, fleets)
        self.remaining_steps = max(1, TOTAL_STEPS - step)
        self.is_early = step < EARLY_TURN_LIMIT
        self.is_opening = step < OPENING_TURN_LIMIT
        self.is_late = self.remaining_steps < LATE_REMAINING_TURNS
        self.is_very_late = self.remaining_steps < VERY_LATE_REMAINING_TURNS
        self.is_four_player = self.num_players >= 4

        self.owner_strength = defaultdict(int)
        self.owner_production = defaultdict(int)
        for planet in planets:
            if planet.owner != -1:
                self.owner_strength[planet.owner] += int(planet.ships)
                self.owner_production[planet.owner] += int(planet.production)
        for fleet in fleets:
            self.owner_strength[fleet.owner] += int(fleet.ships)

        self.my_total = self.owner_strength.get(player, 0)
        self.enemy_total = sum(
            strength for owner, strength in self.owner_strength.items() if owner != player
        )
        self.max_enemy_strength = max(
            (strength for owner, strength in self.owner_strength.items() if owner != player),
            default=0,
        )
        self.my_prod = self.owner_production.get(player, 0)
        self.enemy_prod = sum(
            production
            for owner, production in self.owner_production.items()
            if owner != player
        )

        self.arrivals_by_planet = build_arrival_ledger(fleets, planets)
        self.base_timeline = {
            planet.id: simulate_planet_timeline(
                planet,
                self.arrivals_by_planet[planet.id],
                player,
                HORIZON,
            )
            for planet in planets
        }
        self.keep_needed_map = {
            planet.id: self.base_timeline[planet.id]["keep_needed"] for planet in planets
        }
        self.min_owned_map = {
            planet.id: self.base_timeline[planet.id]["min_owned"] for planet in planets
        }
        self.first_enemy_map = {
            planet.id: self.base_timeline[planet.id]["first_enemy"] for planet in planets
        }
        self.fall_turn_map = {
            planet.id: self.base_timeline[planet.id]["fall_turn"] for planet in planets
        }
        self.holds_full_map = {
            planet.id: self.base_timeline[planet.id]["holds_full"] for planet in planets
        }
        self.indirect_feature_map = {
            planet.id: indirect_features(planet, planets, player) for planet in planets
        }
        self.shot_cache = {}
        self.probe_candidate_cache = {}
        self.best_probe_cache = {}
        self.reaction_cache = {}
        self.exact_need_cache = {}

        self.total_visible_ships = sum(int(planet.ships) for planet in planets) + sum(
            int(fleet.ships) for fleet in fleets
        )
        self.total_production = sum(int(planet.production) for planet in planets)

    def is_static(self, planet_id):
        """按 planet_id 判断星球是否静止。"""
        return is_static_planet(self.planet_by_id[planet_id])

    def comet_life(self, planet_id):
        """查询彗星星球还存在/可追踪多少回合。"""
        return comet_remaining_life(planet_id, self.comets)

    def source_inventory_left(self, source_id, spent_total):
        """计算某个来源星球扣除本回合已花费后的剩余舰船。"""
        return max(0, int(self.planet_by_id[source_id].ships) - spent_total[source_id])

    def plan_shot(self, src_id, target_id, ships):
        """规划一次从 src 到 target、派 ships 艘的实际瞄准结果。

        返回 aim_with_prediction 的结果：
            (angle, turns, predicted_target_x, predicted_target_y)

        同一组 (src, target, ships) 会被大量重复查询，所以这里做缓存。
        """
        ships = int(ships)
        key = (src_id, target_id, ships)
        cached = self.shot_cache.get(key)
        if key in self.shot_cache:
            return cached
        src = self.planet_by_id[src_id]
        target = self.planet_by_id[target_id]
        result = aim_with_prediction(
            src,
            target,
            ships,
            self.initial_by_id,
            self.ang_vel,
            self.comets,
            self.comet_ids,
            self.planets,
        )
        self.shot_cache[key] = result
        return result

    def probe_ship_candidates(self, src_id, target_id, source_cap, hints=()):
        """生成少量“试探派兵数”候选，避免对 1..source_cap 全量搜索。

        派兵数会影响速度和 ETA，所以不能只算一次。这里围绕几个有意义的点
        取样：1~5、小半队、全队、目标驻军+若干余量，以及调用方传入的 hints。
        """
        cache = getattr(self, "probe_candidate_cache", None)
        if cache is None:
            cache = {}
            self.probe_candidate_cache = cache
        source_cap = max(1, int(source_cap))
        normalized_hints = tuple(
            int(math.ceil(hint))
            for hint in hints
            if hint is not None
        )
        cache_key = (src_id, target_id, source_cap, normalized_hints)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        target = self.planet_by_id[target_id]
        target_ships = max(1, int(math.ceil(target.ships)))

        values = set(range(1, min(6, source_cap) + 1))
        values.update(
            {
                source_cap,
                max(1, source_cap // 2),
                max(1, source_cap // 3),
                min(source_cap, PARTIAL_SOURCE_MIN_SHIPS),
                min(source_cap, target_ships + 1),
                min(source_cap, target_ships + 2),
                min(source_cap, target_ships + 4),
                min(source_cap, target_ships + 8),
            }
        )

        for hint in normalized_hints:
            base = max(1, min(source_cap, hint))
            for delta in (-2, -1, 0, 1, 2):
                candidate = base + delta
                if 1 <= candidate <= source_cap:
                    values.add(candidate)

        result = sorted(values)
        cache[cache_key] = result
        return result

    def best_probe_aim(
        self,
        src_id,
        target_id,
        source_cap,
        hints=(),
        min_turn=None,
        max_turn=None,
        anchor_turn=None,
        max_anchor_diff=None,
    ):
        """在若干试探派兵数中，找到最合适的可行瞄准方案。

        如果没有 anchor_turn，就偏好更早到达、派船更少的方案。
        如果有 anchor_turn，就偏好更接近该时间锚点的方案，用于 snipe、
        crash_exploit、救援等需要同步的任务。
        """
        cache_key = (
            src_id,
            target_id,
            max(1, int(source_cap)),
            tuple(hints),
            min_turn,
            max_turn,
            anchor_turn,
            max_anchor_diff,
        )
        cache = getattr(self, "best_probe_cache", None)
        if cache is None:
            cache = {}
            self.best_probe_cache = cache
        if cache_key in cache:
            return cache[cache_key]

        best = None
        best_key = None

        for ships in self.probe_ship_candidates(src_id, target_id, source_cap, hints=hints):
            aim = self.plan_shot(src_id, target_id, ships)
            if aim is None:
                continue

            angle, turns, dist_to_target, path_target = aim
            if min_turn is not None and turns < min_turn:
                continue
            if max_turn is not None and turns > max_turn:
                continue
            if (
                anchor_turn is not None
                and max_anchor_diff is not None
                and abs(turns - anchor_turn) > max_anchor_diff
            ):
                continue

            if anchor_turn is None:
                key = (turns, ships)
            else:
                key = (abs(turns - anchor_turn), turns, ships)

            if best_key is None or key < best_key:
                best_key = key
                best = (ships, (angle, turns, dist_to_target, path_target))

        cache[cache_key] = best
        return best

    def reaction_times(self, target_id):
        """估计我方和敌方最快分别多久能打到目标星球。"""
        cached = self.reaction_cache.get(target_id)
        if cached is not None:
            return cached

        target = self.planet_by_id[target_id]
        my_t = 10**9
        for planet in self.my_planets:
            seeded = self.best_probe_aim(planet.id, target.id, max(1, int(planet.ships)))
            if seeded is None:
                continue
            _, aim = seeded
            my_t = min(my_t, aim[1])

        enemy_t = 10**9
        for planet in self.enemy_planets:
            seeded = self.best_probe_aim(planet.id, target.id, max(1, int(planet.ships)))
            if seeded is None:
                continue
            _, aim = seeded
            enemy_t = min(enemy_t, aim[1])

        cached = (my_t, enemy_t)
        self.reaction_cache[target_id] = cached
        return cached

    def projected_state(self, target_id, arrival_turn, planned_commitments=None, extra_arrivals=()):
        """查询目标星球在某回合的预测状态。

        如果没有新增计划，直接读 base_timeline；如果有 planned_commitments
        或 extra_arrivals，就临时把这些到达事件加入账本重新模拟到 cutoff。
        """
        planned_commitments = planned_commitments or {}
        cutoff = max(1, int(math.ceil(arrival_turn)))
        if not planned_commitments.get(target_id) and not extra_arrivals:
            return state_at_timeline(self.base_timeline[target_id], cutoff)

        arrivals = [
            item
            for item in self.arrivals_by_planet.get(target_id, [])
            if item[0] <= cutoff
        ]
        arrivals.extend(
            item
            for item in planned_commitments.get(target_id, [])
            if item[0] <= cutoff
        )
        arrivals.extend(item for item in extra_arrivals if item[0] <= cutoff)

        target = self.planet_by_id[target_id]
        dyn = simulate_planet_timeline(target, arrivals, self.player, cutoff)
        return state_at_timeline(dyn, cutoff)

    def projected_timeline(self, target_id, horizon, planned_commitments=None, extra_arrivals=()):
        """在加入已计划发兵/额外到达事件后，重新模拟某颗星到 horizon 回合。"""
        planned_commitments = planned_commitments or {}
        horizon = max(1, int(math.ceil(horizon)))
        arrivals = [
            item for item in self.arrivals_by_planet.get(target_id, []) if item[0] <= horizon
        ]
        arrivals.extend(
            item for item in planned_commitments.get(target_id, []) if item[0] <= horizon
        )
        arrivals.extend(item for item in extra_arrivals if item[0] <= horizon)
        target = self.planet_by_id[target_id]
        return simulate_planet_timeline(target, arrivals, self.player, horizon)

    def hold_status(self, target_id, planned_commitments=None, horizon=HORIZON):
        """查询某颗我方星球在给定计划下的防守状态摘要。"""
        planned_commitments = planned_commitments or {}
        if planned_commitments.get(target_id):
            tl = self.projected_timeline(
                target_id,
                horizon,
                planned_commitments=planned_commitments,
            )
        else:
            tl = self.base_timeline[target_id]
        return {
            "keep_needed": tl["keep_needed"],
            "min_owned": tl["min_owned"],
            "first_enemy": tl["first_enemy"],
            "fall_turn": tl["fall_turn"],
            "holds_full": tl["holds_full"],
        }

    def _ownership_search_cap(self, eval_turn):
        """给“最少派多少船”二分搜索提供一个保守上界。"""
        productive_cap = self.total_production * max(2, eval_turn + 2)
        return max(32, int(self.total_visible_ships + productive_cap + 32))

    def min_ships_to_own_by(
        self,
        target_id,
        eval_turn,
        attacker_owner,
        arrival_turn=None,
        planned_commitments=None,
        extra_arrivals=(),
        upper_bound=None,
    ):
        """计算为了在 eval_turn 拥有目标，attacker_owner 最少要派多少船。

        这是策略里最重要的成本计算函数。它会把现有到达账本、已承诺的我方
        发兵、以及这次假设新增的舰队一起放进时间线里模拟。然后用二分搜索
        找到“刚好能让 attacker_owner 拥有目标”的最小 ships。

        如果 upper_bound 都不够，就返回 upper_bound + 1，表示当前来源预算不够。
        """
        planned_commitments = planned_commitments or {}
        eval_turn = max(1, int(math.ceil(eval_turn)))
        arrival_turn = eval_turn if arrival_turn is None else max(1, int(math.ceil(arrival_turn)))
        if arrival_turn > eval_turn:
            if upper_bound is not None:
                return max(1, int(upper_bound)) + 1
            return self._ownership_search_cap(eval_turn) + 1

        normalized_extra = tuple(
            (
                max(1, int(math.ceil(turns))),
                owner,
                int(ships),
            )
            for turns, owner, ships in extra_arrivals
            if ships > 0 and max(1, int(math.ceil(turns))) <= eval_turn
        )

        cache_key = None
        if (
            arrival_turn == eval_turn
            and not planned_commitments.get(target_id)
            and not normalized_extra
        ):
            cache_key = (target_id, eval_turn, attacker_owner)
            cached = self.exact_need_cache.get(cache_key)
            if cached is not None:
                return cached

        owner_before, ships_before = self.projected_state(
            target_id,
            eval_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=normalized_extra,
        )
        if owner_before == attacker_owner:
            if cache_key is not None:
                self.exact_need_cache[cache_key] = 0
            return 0

        def owns_at(ships):
            """假设额外派 ships 艘到达，检查 eval_turn 时是否归 attacker_owner。"""
            owner_after, _ = self.projected_state(
                target_id,
                eval_turn,
                planned_commitments=planned_commitments,
                extra_arrivals=normalized_extra + ((arrival_turn, attacker_owner, int(ships)),),
            )
            return owner_after == attacker_owner

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not owns_at(hi):
                return hi + 1
        else:
            hi = max(1, int(math.ceil(ships_before)) + 1)
            search_cap = self._ownership_search_cap(eval_turn)
            while hi <= search_cap and not owns_at(hi):
                hi *= 2
            if hi > search_cap:
                hi = search_cap
                if not owns_at(hi):
                    return hi + 1

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if owns_at(mid):
                hi = mid
            else:
                lo = mid + 1

        if cache_key is not None:
            self.exact_need_cache[cache_key] = lo
        return lo

    def min_ships_to_own_at(
        self,
        target_id,
        arrival_turn,
        attacker_owner,
        planned_commitments=None,
        extra_arrivals=(),
        upper_bound=None,
    ):
        """min_ships_to_own_by 的简化版本：到达时刻就是评估时刻。"""
        return self.min_ships_to_own_by(
            target_id,
            arrival_turn,
            attacker_owner,
            arrival_turn=arrival_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
            upper_bound=upper_bound,
        )

    def reinforcement_needed_to_hold_until(
        self,
        planet_id,
        arrival_turn,
        hold_until,
        planned_commitments=None,
        upper_bound=None,
    ):
        """计算为了让目标星球一直守到 hold_until，需要增援多少船。

        如果目标当前不是我方星球，就退化为“到 hold_until 时至少要归我方”。
        如果目标当前是我方星球，则模拟增援在 arrival_turn 到达后，检查之后
        每一回合是否都还属于我方。
        """
        planned_commitments = planned_commitments or {}
        target = self.planet_by_id[planet_id]
        arrival_turn = max(1, int(math.ceil(arrival_turn)))
        hold_until = max(arrival_turn, int(math.ceil(hold_until)))

        if target.owner != self.player:
            return self.min_ships_to_own_by(
                planet_id,
                hold_until,
                self.player,
                arrival_turn=arrival_turn,
                planned_commitments=planned_commitments,
                upper_bound=upper_bound,
            )

        def holds_with_reinforcement(ships):
            """假设 arrival_turn 增援 ships 艘，检查之后是否一直守到 hold_until。"""
            timeline = self.projected_timeline(
                planet_id,
                hold_until,
                planned_commitments=planned_commitments,
                extra_arrivals=((arrival_turn, self.player, int(ships)),),
            )
            for turn in range(arrival_turn, hold_until + 1):
                if timeline["owner_at"].get(turn) != self.player:
                    return False
            return True

        if upper_bound is not None:
            hi = max(1, int(upper_bound))
            if not holds_with_reinforcement(hi):
                return hi + 1
        else:
            hi = 1
            search_cap = self._ownership_search_cap(hold_until)
            while hi <= search_cap and not holds_with_reinforcement(hi):
                hi *= 2
            if hi > search_cap:
                hi = search_cap
                if not holds_with_reinforcement(hi):
                    return hi + 1

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if holds_with_reinforcement(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def ships_needed_to_capture(
        self,
        target_id,
        arrival_turn,
        planned_commitments=None,
        extra_arrivals=(),
    ):
        """计算我方在 arrival_turn 占下 target_id 所需的最少舰船。"""
        return self.min_ships_to_own_at(
            target_id,
            arrival_turn,
            self.player,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
        )
