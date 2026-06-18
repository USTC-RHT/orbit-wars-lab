from __future__ import annotations

import argparse
import csv
import json
import re
import time
import zipfile
from pathlib import Path


DATE_RE = re.compile(r"orbit-wars-episodes-(\d{4}-\d{2}-\d{2})\.zip$")


def first_step_player_count(zf: zipfile.ZipFile, name: str) -> int:
    decoder = json.JSONDecoder()
    for read_size in (65536, 262144, 1048576, 4194304):
        with zf.open(name) as f:
            data = f.read(read_size)
        steps_idx = data.find(b'"steps"')
        if steps_idx < 0:
            continue
        colon_idx = data.find(b":", steps_idx)
        outer_list_idx = data.find(b"[", colon_idx)
        first_step_idx = data.find(b"[", outer_list_idx + 1)
        if colon_idx < 0 or outer_list_idx < 0 or first_step_idx < 0:
            continue
        try:
            first_step, _ = decoder.raw_decode(data[first_step_idx:].decode("utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue
        return len(first_step) if isinstance(first_step, list) else -2
    return -1


def episode_id_from_name(name: str) -> str:
    return Path(name).stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a filtered manifest for Orbit Wars replay zip files.")
    parser.add_argument("--zip-dir", type=Path, default=Path("/mnt/f/kaggle-orbit-wars-data/daily_zips_cli"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/f/kaggle-orbit-wars-data/manifests/orbit_2p_manifest.csv"))
    parser.add_argument("--players", type=int, default=2, help="Keep only this player count. Use 0 to keep all.")
    parser.add_argument("--limit-zips", type=int, default=0)
    parser.add_argument("--report-interval", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    zips = sorted(args.zip_dir.glob("orbit-wars-episodes-*.zip"))
    if int(args.limit_zips) > 0:
        zips = zips[: int(args.limit_zips)]

    fieldnames = [
        "date",
        "zip_path",
        "json_name",
        "episode_id",
        "player_count",
        "uncompressed_bytes",
        "compressed_bytes",
    ]
    total = kept = bad = 0
    by_players: dict[int, int] = {}
    started = time.monotonic()
    last_report = started

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for zip_index, zpath in enumerate(zips, start=1):
            match = DATE_RE.match(zpath.name)
            date = match.group(1) if match else ""
            zip_total = zip_kept = 0
            with zipfile.ZipFile(zpath) as zf:
                infos = [info for info in zf.infolist() if info.filename.endswith(".json")]
                for info in infos:
                    player_count = first_step_player_count(zf, info.filename)
                    total += 1
                    zip_total += 1
                    by_players[player_count] = by_players.get(player_count, 0) + 1
                    if player_count < 0:
                        bad += 1
                    if int(args.players) > 0 and player_count != int(args.players):
                        continue
                    kept += 1
                    zip_kept += 1
                    writer.writerow(
                        {
                            "date": date,
                            "zip_path": str(zpath),
                            "json_name": info.filename,
                            "episode_id": episode_id_from_name(info.filename),
                            "player_count": player_count,
                            "uncompressed_bytes": info.file_size,
                            "compressed_bytes": info.compress_size,
                        }
                    )
                    now = time.monotonic()
                    if now - last_report >= float(args.report_interval):
                        elapsed = now - started
                        rate = total / max(elapsed, 1e-9)
                        print(
                            f"[progress] zips={zip_index}/{len(zips)} scanned={total} kept={kept} "
                            f"bad={bad} rate={rate:.1f}/s players={dict(sorted(by_players.items()))}",
                            flush=True,
                        )
                        last_report = now
            print(f"[zip] {date} scanned={zip_total} kept={zip_kept} path={zpath}", flush=True)

    elapsed = time.monotonic() - started
    print(
        f"[done] output={args.output} zips={len(zips)} scanned={total} kept={kept} bad={bad} "
        f"elapsed={elapsed:.1f}s players={dict(sorted(by_players.items()))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
