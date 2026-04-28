"""全局配置和调参常量。

这里集中放所有阶段阈值、评分倍率、派兵余量、时间预算等参数。
其它模块只读取这些常量，不在运行时修改。
"""

# ============================================================
# 地图与物理规则
# ============================================================

# 棋盘是 100x100 的坐标系，中心点也是太阳中心。
BOARD = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0

# 环境发射舰队时，会从来源星球半径外再偏移 0.1，避免刚生成就撞回来源星。
LAUNCH_CLEARANCE = 0.1

# 太阳半径。舰队路径太靠近太阳会被判为不可用。
SUN_R = 10.0

# 舰队速度上限；实际速度由派出的 ships 数量按 fleet_speed() 计算。
MAX_SPEED = 6.0

# 旋转边界。轨道半径 + 星球半径 小于该值的星球会按 angular_velocity 旋转。
ROTATION_LIMIT = 50.0

# 发射点从来源星球边缘再外移 0.1，与游戏引擎 process_moves 硬编码的偏移一致。
# 引擎：start = planet_center + cos/sin * (radius + 0.1)
# 碰撞判定：路径段到星球中心距离 < planet.radius（严格小于）
# 不可修改此值，否则 ETA 估算与实际飞行距离不匹配。
LAUNCH_CLEARANCE = 0.1

# 游戏总回合数。
TOTAL_STEPS = 500

# 世界模型向前模拟的回合数。它不是预测敌人意图，只模拟已知舰队和生产。
SIM_HORIZON = 110

# 搜索移动目标安全拦截点时，最多往未来看多少回合。
ROUTE_SEARCH_HORIZON = 60

# HORIZON 是世界模型和部分策略共享的默认模拟窗口。
HORIZON = SIM_HORIZON


# ============================================================
# 游戏阶段划分
# ============================================================

# step < 40 视为早期，通常更偏向扩张中立星。
EARLY_TURN_LIMIT = 40

# step < 80 视为开局，旋转中立星和敌方目标会更谨慎。
OPENING_TURN_LIMIT = 80

# 剩余回合 < 60 进入后期，开始更重视眼前收益和消灭敌方。
LATE_REMAINING_TURNS = 60

# 剩余回合 < 25 进入极后期，部分补充扩张逻辑会关闭。
VERY_LATE_REMAINING_TURNS = 25

# ============================================================
# 反应时间与拦截容忍
# ============================================================

# 中立星：我方最快到达时间至少比敌方快 2 回合，才算安全中立星。
SAFE_NEUTRAL_MARGIN = 2

# 中立星：双方最快反应时间差不超过 2 回合，视为争抢目标。
CONTESTED_NEUTRAL_MARGIN = 2

# 移动目标瞄准时，允许 ETA 和目标预测回合相差 1 回合。
INTERCEPT_TOLERANCE = 1

# ============================================================
# 开局过滤：避免早期浪费兵力追差目标
# ============================================================

# 开局时，高产中立星达到该生产力阈值，且我方明显先手，会被优先放行。
SAFE_OPENING_PROD_THRESHOLD = 4

# 开局安全目标如果能在 10 回合内到达，认为足够近。
SAFE_OPENING_TURN_LIMIT = 10

# 开局追旋转中立星的最大舒适到达时间，超过后通常过滤。
ROTATING_OPENING_MAX_TURNS = 13

# 开局低产旋转中立星通常不值得抢。
ROTATING_OPENING_LOW_PROD = 2

# 四人局开局抢旋转中立星时，需要比敌人至少快这么多回合才考虑。
FOUR_PLAYER_ROTATING_REACTION_GAP = 3

# 四人局开局抢旋转中立星时，最多使用来源可用兵力的这个比例。
FOUR_PLAYER_ROTATING_SEND_RATIO = 0.62

# 四人局开局旋转中立星必须较快到达，否则容易被第三方搅局。
FOUR_PLAYER_ROTATING_TURN_LIMIT = 10

# ============================================================
# 彗星目标
# ============================================================

# 最多追击未来 10 回合内能命中的彗星，太久后路径收益不可靠。
COMET_MAX_CHASE_TURNS = 10

# ============================================================
# 基础评分成本与间接价值
# ============================================================

# 普通进攻评分中的时间成本权重：score = value / (ships + turns * weight + 1)。
ATTACK_COST_TURN_WEIGHT = 0.55

# snipe / crash_exploit 更依赖时机，时间成本略低。
SNIPE_COST_TURN_WEIGHT = 0.45

# 周边星球带来的间接战略价值缩放系数。
INDIRECT_VALUE_SCALE = 0.15

# 目标附近我方星球的价值权重：方便支援，但不是最重要。
INDIRECT_FRIENDLY_WEIGHT = 0.35

# 目标附近中立星的价值权重：占下后可继续扩张。
INDIRECT_NEUTRAL_WEIGHT = 0.9

# 目标附近敌方星的价值权重：作为前线跳板/压制点更有价值。
INDIRECT_ENEMY_WEIGHT = 1.25

# ============================================================
# 目标价值倍率：影响 target_value()
# ============================================================

# 静态中立星更稳定，长期经营价值更高。
STATIC_NEUTRAL_VALUE_MULT = 1.4

# 静态敌方星也更值得打，打下后更容易守和利用。
STATIC_HOSTILE_VALUE_MULT = 1.55

# 开局时旋转星风险较高，所以降低一点价值。
ROTATING_OPENING_VALUE_MULT = 0.9

# 敌方目标是双重收益：敌人少一颗，我方多一颗。
HOSTILE_TARGET_VALUE_MULT = 1.85

# 开局打敌方星更谨慎，倍率低于中后期。
OPENING_HOSTILE_TARGET_VALUE_MULT = 1.45

# 我方明显先手的中立星加分。
SAFE_NEUTRAL_VALUE_MULT = 1.2

# 双方都能差不多时间到的中立星降分，避免陷入抢夺消耗。
CONTESTED_NEUTRAL_VALUE_MULT = 0.7

# 早期更鼓励扩张中立星。
EARLY_NEUTRAL_VALUE_MULT = 1.2

# 彗星寿命短/不稳定，整体价值打折。
COMET_VALUE_MULT = 0.65

# 狙击任务的收益倍率。
SNIPE_VALUE_MULT = 1.12

# 多源合攻任务的收益倍率。
SWARM_VALUE_MULT = 1.05

# 增援任务的收益倍率。
REINFORCE_VALUE_MULT = 1.35

# 四人局敌方互撞后补刀任务的收益倍率。
CRASH_EXPLOIT_VALUE_MULT = 1.18

# 终结阶段打敌方目标额外加分。
FINISHING_HOSTILE_VALUE_MULT = 1.15

# 落后时降低追旋转中立星的意愿，避免继续分散兵力。
BEHIND_ROTATING_NEUTRAL_VALUE_MULT = 0.92

# ============================================================
# 派兵余量：影响 preferred_send()
# ============================================================

# 打中立星的基础安全余量。
NEUTRAL_MARGIN_BASE = 2

# 中立星产量越高，越值得多派一些保证拿下。
NEUTRAL_MARGIN_PROD_WEIGHT = 2

# 中立星余量上限，防止因为高产目标过度派兵。
NEUTRAL_MARGIN_CAP = 8

# 打敌方星的基础余量比中立星更高，因为敌方会继续生产且更可能有支援。
HOSTILE_MARGIN_BASE = 3

# 敌方星产量越高，派兵余量越大。
HOSTILE_MARGIN_PROD_WEIGHT = 2

# 敌方星余量上限。
HOSTILE_MARGIN_CAP = 12

# 静态目标更值得多派一点确保成功。
STATIC_TARGET_MARGIN = 4

# 争抢中立星额外加余量。
CONTESTED_TARGET_MARGIN = 5

# 四人局不确定性更大，额外加余量。
FOUR_PLAYER_TARGET_MARGIN = 3

# 超过该到达时间后，认为是长途进攻，需要逐步增加余量。
LONG_TRAVEL_MARGIN_START = 18

# 长途余量按 arrival_turns // LONG_TRAVEL_MARGIN_DIVISOR 增加。
LONG_TRAVEL_MARGIN_DIVISOR = 3

# 长途余量上限。
LONG_TRAVEL_MARGIN_CAP = 8

# 打彗星时减少余量，避免给短生命周期目标投入过多。
COMET_MARGIN_RELIEF = 6

# 终结阶段打敌方目标，额外多派一点提高确定性。
FINISHING_HOSTILE_SEND_BONUS = 3

# ============================================================
# 评分后处理倍率：影响 apply_score_modifiers()
# ============================================================

# 静态目标整体加分。
STATIC_TARGET_SCORE_MULT = 1.18

# 早期静态中立星额外加分，鼓励稳定扩张。
EARLY_STATIC_NEUTRAL_SCORE_MULT = 1.25

# 四人局旋转中立星降分，减少混战风险。
FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT = 0.84

# 静态中立星数量达到该阈值时，说明地图上稳定目标多。
DENSE_STATIC_NEUTRAL_COUNT = 4

# 稳定目标很多时，旋转中立星相对降分。
DENSE_ROTATING_NEUTRAL_SCORE_MULT = 0.86

# snipe 任务额外分数倍率。
SNIPE_SCORE_MULT = 1.12

# swarm 任务额外分数倍率。
SWARM_SCORE_MULT = 1.06

# crash_exploit 任务额外分数倍率。
CRASH_EXPLOIT_SCORE_MULT = 1.05

# ============================================================
# 补充进攻和后期过滤
# ============================================================

# 第一轮任务后，如果来源星剩余进攻预算至少这么多，才做补充进攻。
FOLLOWUP_MIN_SHIPS = 8

# 低产彗星在补充进攻阶段会被忽略。
LOW_VALUE_COMET_PRODUCTION = 1

# 后期进攻必须在剩余回合前留出该缓冲，避免到达太晚无收益。
LATE_CAPTURE_BUFFER = 5

# 极后期缓冲更小，允许更贴近终局的进攻。
VERY_LATE_CAPTURE_BUFFER = 3

# ============================================================
# 防守、救援、增援
# ============================================================

# 检查未来多少回合内的失守风险。
DEFENSE_LOOKAHEAD_TURNS = 28

# 防守/救援评分中的时间成本权重。
DEFENSE_COST_TURN_WEIGHT = 0.4

# 前线星球防守价值加成。
DEFENSE_FRONTIER_SCORE_MULT = 1.12

# 救援派兵的基础安全余量。
DEFENSE_SEND_MARGIN_BASE = 1

# 救援余量随目标产量增加。
DEFENSE_SEND_MARGIN_PROD_WEIGHT = 1

# 防守评分里，当前驻军/保存舰船的价值权重。
DEFENSE_SHIP_VALUE = 0.55

# 是否启用主动增援任务。
REINFORCE_ENABLED = True

# 只有 production 至少为该值的我方星球才考虑主动增援。
REINFORCE_MIN_PRODUCTION = 2

# 增援最多允许飞这么久，太慢赶不上防守价值。
REINFORCE_MAX_TRAVEL_TURNS = 22

# 增援计算出的最低需求上再多派的安全余量。
REINFORCE_SAFETY_MARGIN = 2

# 单次增援最多使用来源星当前舰船的比例。
REINFORCE_MAX_SOURCE_FRACTION = 0.75

# 剩余回合少于该值时不做主动增援，避免长期防守收益不足。
REINFORCE_MIN_FUTURE_TURNS = 40

# 增援希望目标在失守点之后继续守住多少回合。
REINFORCE_HOLD_LOOKAHEAD = 20

# 增援评分中的时间成本权重。
REINFORCE_COST_TURN_WEIGHT = 0.35

# ============================================================
# 收复 recapture
# ============================================================

# 星球失守后，最多在多少回合内尝试收复。
RECAPTURE_LOOKAHEAD_TURNS = 10

# 收复评分中的时间成本权重。
RECAPTURE_COST_TURN_WEIGHT = 0.52

# 收复任务整体价值倍率。
RECAPTURE_VALUE_MULT = 0.88

# 前线星球收复加成。
RECAPTURE_FRONTIER_MULT = 1.08

# 收复价值中，未来生产收益的权重。
RECAPTURE_PRODUCTION_WEIGHT = 0.6

# 收复价值中，当前/即时舰船收益的权重。
RECAPTURE_IMMEDIATE_WEIGHT = 0.4

# ============================================================
# 后方输送：rear source -> front/stage
# ============================================================

# 后方星球至少有这么多进攻预算，才考虑向前线输送。
REAR_SOURCE_MIN_SHIPS = 16

# 判断“后方”的距离比例阈值。
REAR_DISTANCE_RATIO = 1.25

# 中转星需要比后方星更靠近前线到这个比例，才适合作为 staging。
REAR_STAGE_PROGRESS = 0.78

# 双人局后方输送比例。
REAR_SEND_RATIO_TWO_PLAYER = 0.62

# 四人局/终结阶段后方输送比例更高。
REAR_SEND_RATIO_FOUR_PLAYER = 0.7

# 后方输送至少发送这么多船。
REAR_SEND_MIN_SHIPS = 10

# 后方输送最长允许飞行时间。
REAR_MAX_TRAVEL_TURNS = 40

# ============================================================
# 多源合攻 swarm
# ============================================================

# 单个来源至少愿意贡献这么多船，才作为 partial/swarm 候选。
PARTIAL_SOURCE_MIN_SHIPS = 6

# 每个目标最多保留评分最高的若干来源候选参与组合。
MULTI_SOURCE_TOP_K = 5

# 普通两源合攻允许到达时间相差多少回合。
MULTI_SOURCE_ETA_TOLERANCE = 2

# 多源合攻组合分数折扣，避免过度偏好复杂计划。
MULTI_SOURCE_PLAN_PENALTY = 0.97

# 打敌方星时多源合攻要求更同步。
HOSTILE_SWARM_ETA_TOLERANCE = 1

# 是否启用三源合攻。
THREE_SOURCE_SWARM_ENABLED = True

# 只有目标驻军至少这么多，才考虑三源合攻，避免小目标复杂化。
THREE_SOURCE_MIN_TARGET_SHIPS = 20

# 三源合攻允许到达时间相差多少回合。
THREE_SOURCE_ETA_TOLERANCE = 1

# 三源合攻额外分数折扣。
THREE_SOURCE_PLAN_PENALTY = 0.93

# ============================================================
# 主动防守 reserve 估算
# ============================================================

# 敌方星如果能在该回合数内打到我方星，就考虑主动留防。
PROACTIVE_DEFENSE_HORIZON = 12

# 单个敌星威胁下，按敌星舰船的该比例预留防守。
PROACTIVE_DEFENSE_RATIO = 0.18

# 多个敌星叠加威胁的观察窗口。
MULTI_ENEMY_PROACTIVE_HORIZON = 14

# 多敌星叠加威胁下，按窗口内敌舰总量的该比例预留。
MULTI_ENEMY_PROACTIVE_RATIO = 0.22

# 多敌星威胁 ETA 相差不超过该窗口，就视为可能叠加。
MULTI_ENEMY_STACK_WINDOW = 3

# 计算反应时间时，我方最多看离目标最近的几个来源。
REACTION_SOURCE_TOP_K_MY = 4

# 计算敌方反应时间时，敌方最多看离目标最近的几个来源。
REACTION_SOURCE_TOP_K_ENEMY = 4

# 主动防守时，只检查离我方星最近的几个敌星。
PROACTIVE_ENEMY_TOP_K = 3

# ============================================================
# 四人局碰撞利用 crash_exploit
# ============================================================

# 是否启用敌方互撞后补刀。
CRASH_EXPLOIT_ENABLED = True

# 两支敌方舰队总兵力至少达到该值，才值得检测碰撞机会。
CRASH_EXPLOIT_MIN_TOTAL_SHIPS = 10

# 两支不同敌人的 ETA 差在该窗口内，视为可能同场抵消。
CRASH_EXPLOIT_ETA_WINDOW = 2

# 敌方碰撞后延迟多少回合补刀。
CRASH_EXPLOIT_POST_CRASH_DELAY = 1

# ============================================================
# 后期清场和局势模式
# ============================================================

# 后期目标上现有舰船也算即时价值，按该权重加入 value。
LATE_IMMEDIATE_SHIP_VALUE = 0.6

# 敌方总兵力低于该值时，后期攻击该敌方目标有消灭奖励。
WEAK_ENEMY_THRESHOLD = 45

# 后期消灭弱敌的额外价值。
ELIMINATION_BONUS = 18.0

# domination = (我方总兵力 - 敌方总兵力) / 总兵力。
# 低于该阈值视为落后。
BEHIND_DOMINATION = -0.20

# 高于该阈值视为领先。
AHEAD_DOMINATION = 0.18

# 高于该阈值，且生产力也明显领先时，视为终结阶段。
FINISHING_DOMINATION = 0.35

# 终结阶段要求我方生产力至少是敌方生产力的这个倍数。
FINISHING_PROD_RATIO = 1.25

# 领先时提高派兵余量。
AHEAD_ATTACK_MARGIN_BONUS = 0.08

# 落后时降低派兵余量，减少过度投入。
BEHIND_ATTACK_MARGIN_PENALTY = 0.05

# 终结阶段提高派兵余量，确保清场攻击成功。
FINISHING_ATTACK_MARGIN_BONUS = 0.08

# ============================================================
# 撤离 doomed planet
# ============================================================

# 如果我方星球在该窗口内仍会失守，就进入撤离/最后一击逻辑。
DOOMED_EVAC_TURN_LIMIT = 24

# 星球至少剩这么多船，才值得执行撤离或最后一击。
DOOMED_MIN_SHIPS = 8

# ============================================================
# 单回合规划时间预算
# ============================================================

# agent 软截止时间，单位秒。超过后尽快返回已规划动作。
SOFT_ACT_DEADLINE = 0.82

# 只有剩余时间超过该值，才运行较重的组合搜索。
HEAVY_PHASE_MIN_TIME = 0.16

# 只有剩余时间超过该值，才运行可选补充阶段。
OPTIONAL_PHASE_MIN_TIME = 0.08

# 星球数量超过该值时，跳过部分重搜索，防止超时。
HEAVY_ROUTE_PLANET_LIMIT = 32
