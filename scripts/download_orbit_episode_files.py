from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def human_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size:.0f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def human_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours, rem = divmod(int(seconds + 0.5), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def make_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def list_files(dataset: str, page_size: int) -> list[dict[str, Any]]:
    api = make_api()
    rows: list[dict[str, Any]] = []
    page_token = None
    while True:
        response = api.dataset_list_files(dataset, page_token=page_token, page_size=page_size)
        for item in response.files:
            name = getattr(item, "name", "")
            if name.endswith(".json"):
                rows.append({"name": name, "size": int(getattr(item, "total_bytes", 0) or 0)})
        page_token = getattr(response, "next_page_token", None)
        if not page_token:
            break
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Orbit Wars replay JSON files with aggregate speed and ETA logs.")
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--out-root", type=Path, default=Path("/mnt/f/kaggle-orbit-wars-data/daily_files"))
    parser.add_argument("--kaggle-config-dir", type=Path, default=Path("/home/dministrator/.config/kaggle"))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--limit-per-day", type=int, default=0)
    parser.add_argument("--small-first", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-interval", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["KAGGLE_CONFIG_DIR"] = str(args.kaggle_config_dir)
    args.out_root.mkdir(parents=True, exist_ok=True)

    thread_local = threading.local()
    lock = threading.Lock()
    stats = {
        "completed_files": 0,
        "skipped_files": 0,
        "failed_files": 0,
        "completed_bytes": 0,
    }
    started = time.monotonic()

    def api_for_thread():
        api = getattr(thread_local, "api", None)
        if api is None:
            api = make_api()
            thread_local.api = api
        return api

    all_jobs: list[dict[str, Any]] = []
    for date in args.dates:
        dataset = f"kaggle/orbit-wars-episodes-{date}"
        day_dir = args.out_root / f"orbit-wars-episodes-{date}" / "files"
        day_dir.mkdir(parents=True, exist_ok=True)
        print(f"[list] {dataset}", flush=True)
        rows = list_files(dataset, int(args.page_size))
        if args.small_first:
            rows.sort(key=lambda row: (row["size"], row["name"]))
        if int(args.limit_per_day) > 0:
            rows = rows[: int(args.limit_per_day)]
        (day_dir.parent / "files_selected.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        for row in rows:
            all_jobs.append(
                {
                    "date": date,
                    "dataset": dataset,
                    "name": row["name"],
                    "size": int(row["size"]),
                    "out_dir": str(day_dir),
                    "path": str(day_dir / row["name"]),
                }
            )
        print(f"[list-done] {date} files={len(rows)} bytes={human_bytes(sum(r['size'] for r in rows))}", flush=True)

    total_files = len(all_jobs)
    total_bytes = sum(job["size"] for job in all_jobs)
    print(
        f"[start] dates={len(args.dates)} files={total_files} bytes={human_bytes(total_bytes)} "
        f"workers={int(args.workers)} out_root={args.out_root}",
        flush=True,
    )

    def report(tag: str = "progress") -> None:
        elapsed = max(time.monotonic() - started, 1e-9)
        with lock:
            completed_files = int(stats["completed_files"])
            skipped_files = int(stats["skipped_files"])
            failed_files = int(stats["failed_files"])
            completed_bytes = int(stats["completed_bytes"])
        avg_speed = completed_bytes / elapsed
        remaining = max(total_bytes - completed_bytes, 0)
        eta = remaining / avg_speed if avg_speed > 0 else 0.0
        pct = 100.0 * completed_bytes / total_bytes if total_bytes else 0.0
        print(
            f"[{tag}] files={completed_files}/{total_files} skipped={skipped_files} failed={failed_files} "
            f"bytes={pct:6.2f}% {human_bytes(completed_bytes)}/{human_bytes(total_bytes)} "
            f"avg={human_bytes(avg_speed)}/s elapsed={human_duration(elapsed)} eta={human_duration(eta)}",
            flush=True,
        )

    def download_one(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        path = Path(job["path"])
        expected = int(job["size"])
        if path.is_file() and not args.force:
            actual = path.stat().st_size
            if expected <= 0 or actual == expected:
                return "skipped", {**job, "actual": actual}
        api = api_for_thread()
        try:
            ok = api.dataset_download_file(
                job["dataset"],
                job["name"],
                path=job["out_dir"],
                force=bool(args.force),
                quiet=True,
            )
            actual = path.stat().st_size if path.is_file() else 0
            if not path.is_file() or (expected > 0 and actual != expected):
                return "failed", {**job, "actual": actual, "error": "size_mismatch_or_missing"}
            return "downloaded" if ok else "skipped", {**job, "actual": actual}
        except Exception as exc:
            return "failed", {**job, "error": repr(exc)}

    failures: list[dict[str, Any]] = []
    last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(int(args.workers), 1)) as pool:
        futures = [pool.submit(download_one, job) for job in all_jobs]
        for future in as_completed(futures):
            status, record = future.result()
            size = int(record.get("actual") or record.get("size") or 0)
            with lock:
                if status == "failed":
                    stats["failed_files"] += 1
                    failures.append(record)
                elif status == "skipped":
                    stats["skipped_files"] += 1
                    stats["completed_files"] += 1
                    stats["completed_bytes"] += size
                else:
                    stats["completed_files"] += 1
                    stats["completed_bytes"] += size
            now = time.monotonic()
            if now - last_report >= float(args.report_interval):
                report()
                last_report = now

    report(tag="done")
    manifest = {
        "dates": args.dates,
        "out_root": str(args.out_root),
        "workers": int(args.workers),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "failed": failures,
    }
    (args.out_root / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} files failed; see {args.out_root / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
