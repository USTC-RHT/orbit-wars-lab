from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def run_text(cmd: list[str], env: dict[str, str]) -> str:
    return subprocess.check_output(cmd, text=True, env=env)


def list_dataset_files(dataset: str, limit: int, env: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    page_token: str | None = None
    while len(rows) < limit:
        cmd = ["kaggle", "datasets", "files", dataset, "--page-size", str(min(200, limit - len(rows))), "--csv"]
        if page_token:
            cmd.extend(["--page-token", page_token])
        out = run_text(cmd, env)
        csv_lines: list[str] = []
        next_token = None
        for line in out.splitlines():
            if line.startswith("Next Page Token = "):
                next_token = line.split("=", 1)[1].strip()
            elif line.strip():
                csv_lines.append(line)
        if not csv_lines:
            break
        page_rows = [row for row in csv.DictReader(csv_lines) if row.get("name", "").endswith(".json")]
        rows.extend(page_rows)
        if not page_rows or not next_token:
            break
        page_token = next_token
    return rows[:limit]


def episode_summary(path: Path) -> dict[str, Any]:
    try:
        replay = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"valid_json": False, "error": str(exc), "player_count": 0}
    if not isinstance(replay, dict):
        return {"valid_json": False, "error": "root_not_object", "player_count": 0}
    steps = replay.get("steps")
    first_step = steps[0] if isinstance(steps, list) and steps else []
    info = replay.get("info", {})
    return {
        "valid_json": True,
        "episode_id": info.get("EpisodeId") if isinstance(info, dict) else replay.get("id"),
        "player_count": len(first_step) if isinstance(first_step, list) else 0,
        "step_count": len(steps) if isinstance(steps, list) else 0,
        "rewards": replay.get("rewards"),
        "statuses": replay.get("statuses"),
    }


def load_previous_filtered(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    filtered = manifest.get("filtered", [])
    if not isinstance(filtered, list):
        return []
    return [row for row in filtered if isinstance(row, dict) and row.get("name")]


def row_size(row: dict[str, Any]) -> int:
    try:
        return int(row.get("size", 0))
    except (TypeError, ValueError):
        return 0


def download_one(
    dataset: str,
    name: str,
    out_dir: Path,
    env: dict[str, str],
    timeout: int,
    players: int,
    keep_nonmatching: bool,
) -> dict[str, Any]:
    target = out_dir / name
    status = "exists"
    if not target.is_file():
        cmd = ["kaggle", "datasets", "download", dataset, "-f", name, "-p", str(out_dir), "-q"]
        try:
            proc = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                timeout=max(int(timeout), 1),
            )
        except subprocess.TimeoutExpired:
            return {"name": name, "kept": False, "status": f"timeout>{int(timeout)}s"}
        if proc.returncode != 0 or not target.is_file():
            msg = proc.stderr.strip() or proc.stdout.strip() or f"returncode={proc.returncode}"
            return {"name": name, "kept": False, "status": msg}
        status = "downloaded"

    summary = episode_summary(target)
    if int(players) > 0 and int(summary.get("player_count", 0)) != int(players):
        if not keep_nonmatching:
            target.unlink(missing_ok=True)
        return {"name": name, "kept": False, "status": "filtered_player_count", **summary}
    return {"name": name, "kept": True, "status": status, "path": str(target), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official Orbit Wars replay JSON files and keep only N-player games.")
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--files-per-day", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--keep-nonmatching", action="store_true")
    parser.add_argument("--retry-filtered", action="store_true")
    parser.add_argument("--max-size-bytes", type=int, default=0)
    parser.add_argument("--small-first", action="store_true")
    parser.add_argument("--out-root", type=Path, default=Path("/home/dministrator/kaggle/orbit-wars-lab/data"))
    parser.add_argument("--kaggle-config-dir", type=Path, default=Path("/home/dministrator/.config/kaggle"))
    args = parser.parse_args()

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(args.kaggle_config_dir)
    args.out_root.mkdir(parents=True, exist_ok=True)
    run_manifest: dict[str, Any] = {
        "dates": [],
        "players_filter": int(args.players),
        "files_per_day": int(args.files_per_day),
    }

    for date in args.dates:
        slug = f"orbit-wars-episodes-{date}"
        dataset = f"kaggle/{slug}"
        day_dir = args.out_root / slug
        out_dir = day_dir / "files"
        out_dir.mkdir(parents=True, exist_ok=True)
        selected_all = list_dataset_files(dataset, int(args.files_per_day), env)
        if args.small_first:
            selected_all = sorted(selected_all, key=lambda row: (row_size(row), row.get("name", "")))
        previous_filtered = [] if args.retry_filtered else load_previous_filtered(day_dir / "download_2p_manifest.json")
        previous_filtered_names = {row["name"] for row in previous_filtered}
        selected_all_names = {item.get("name") for item in selected_all}
        selected = [row for row in selected_all if row.get("name") not in previous_filtered_names]
        skipped_by_size = []
        if int(args.max_size_bytes) > 0:
            kept_by_size = []
            for row in selected:
                if row_size(row) > int(args.max_size_bytes):
                    skipped_by_size.append({**row, "kept": False, "status": "skipped_size_limit"})
                else:
                    kept_by_size.append(row)
            selected = kept_by_size
        skipped_filtered = [
            {**row, "kept": False, "status": "filtered_player_count_previous"}
            for row in previous_filtered
            if row.get("name") in selected_all_names
        ]
        (day_dir / "files_selected.json").write_text(json.dumps(selected_all, indent=2), encoding="utf-8")
        print(
            f"[{date}] selected={len(selected_all)} attempted={len(selected)} "
            f"skipped_previous_filtered={len(skipped_filtered)} "
            f"skipped_by_size={len(skipped_by_size)} dataset={dataset}",
            flush=True,
        )

        kept: list[dict[str, Any]] = []
        filtered: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(int(args.workers), 1)) as pool:
            futures = [
                pool.submit(
                    download_one,
                    dataset,
                    row["name"],
                    out_dir,
                    env,
                    int(args.timeout),
                    int(args.players),
                    bool(args.keep_nonmatching),
                )
                for row in selected
            ]
            for future in as_completed(futures):
                record = future.result()
                if record.get("kept"):
                    kept.append(record)
                elif record.get("status") == "filtered_player_count":
                    filtered.append(record)
                else:
                    failed.append(record)
                seen = len(kept) + len(filtered) + len(failed)
                if seen % 25 == 0 or seen == len(selected):
                    print(
                        f"[{date}] {seen}/{len(selected)} kept={len(kept)} "
                        f"filtered={len(filtered)} failed={len(failed)}",
                        flush=True,
                    )

        day_record = {
            "date": date,
            "dataset": dataset,
            "out_dir": str(out_dir),
            "selected": len(selected_all),
            "attempted_count": len(selected),
            "skipped_by_size_count": len(skipped_by_size),
            "kept_count": len(kept),
            "filtered_count": len(skipped_filtered) + len(filtered),
            "filtered_previous_count": len(skipped_filtered),
            "failed_count": len(failed),
            "kept": sorted(kept, key=lambda x: x["name"]),
            "filtered": sorted(skipped_filtered + filtered, key=lambda x: x["name"]),
            "skipped_by_size": sorted(skipped_by_size, key=lambda x: x["name"]),
            "failed": sorted(failed, key=lambda x: x["name"]),
        }
        (day_dir / "download_2p_manifest.json").write_text(json.dumps(day_record, indent=2), encoding="utf-8")
        run_manifest["dates"].append(day_record)

    (args.out_root / "download_2p_recent_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
