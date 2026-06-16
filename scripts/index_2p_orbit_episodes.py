from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DATE_RE = re.compile(r"^orbit-wars-episodes-(\d{4}-\d{2}-\d{2})$")


def summarize_episode(path: Path) -> dict[str, Any]:
    replay = json.loads(path.read_text(encoding="utf-8"))
    steps = replay.get("steps")
    first_step = steps[0] if isinstance(steps, list) and steps else []
    info = replay.get("info", {})
    return {
        "episode_id": info.get("EpisodeId") if isinstance(info, dict) else replay.get("id"),
        "player_count": len(first_step) if isinstance(first_step, list) else 0,
        "step_count": len(steps) if isinstance(steps, list) else 0,
        "rewards": json.dumps(replay.get("rewards"), separators=(",", ":")),
        "statuses": json.dumps(replay.get("statuses"), separators=(",", ":")),
    }


def iter_episode_files(data_root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for day_dir in sorted(data_root.glob("orbit-wars-episodes-*"), reverse=True):
        match = DATE_RE.match(day_dir.name)
        if not match:
            continue
        files_dir = day_dir / "files"
        if not files_dir.is_dir():
            continue
        for path in sorted(files_dir.glob("*.json")):
            rows.append((match.group(1), path))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a date-labeled CSV index for downloaded Orbit Wars episodes.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/dministrator/kaggle/orbit-wars-lab/data"))
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or (args.data_root / f"orbit_{int(args.players)}p_index.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "dataset",
        "path",
        "file",
        "size_bytes",
        "episode_id",
        "player_count",
        "step_count",
        "rewards",
        "statuses",
    ]

    kept = 0
    skipped = 0
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for date, path in iter_episode_files(args.data_root):
            try:
                summary = summarize_episode(path)
            except Exception:
                skipped += 1
                continue
            if int(args.players) > 0 and int(summary["player_count"]) != int(args.players):
                skipped += 1
                continue
            writer.writerow(
                {
                    "date": date,
                    "dataset": f"kaggle/orbit-wars-episodes-{date}",
                    "path": str(path),
                    "file": path.name,
                    "size_bytes": path.stat().st_size,
                    **summary,
                }
            )
            kept += 1
    print(f"wrote={output} kept={kept} skipped={skipped}")


if __name__ == "__main__":
    main()
