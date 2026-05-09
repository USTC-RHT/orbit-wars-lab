"""Nearest opponent used by the local RL training loop.

This is intentionally separate from Kaggle's tutorial nearest-sniper baseline.
The training opponent sends a chunky fleet from every owned planet whenever it
has at least 12 ships, targeting the nearest planet not owned by itself.
"""

from __future__ import annotations

import math
from typing import Any

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet


def _read(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _distance(a: Planet, b: Planet) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def agent(obs: Any) -> list[list[float | int]]:
    player = int(_read(obs, "player", 0))
    raw_planets = _read(obs, "planets", []) or []
    planets = [Planet(*row) for row in raw_planets]

    sources = [planet for planet in planets if planet.owner == player]
    targets = [planet for planet in planets if planet.owner != player]
    moves: list[list[float | int]] = []

    for src in sources:
        if src.ships < 12 or not targets:
            continue
        target = min(targets, key=lambda planet: (_distance(src, planet), planet.id))
        ships = min(int(src.ships * 0.7), max(12, int(target.ships) + 2))
        if ships <= 0 or ships > src.ships:
            continue
        angle = math.atan2(target.y - src.y, target.x - src.x)
        moves.append([src.id, float(angle), int(ships)])

    return moves
