from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests


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


def load_kaggle_auth(config_dir: Path) -> tuple[str, str]:
    cfg_path = config_dir / "kaggle.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return str(cfg["username"]), str(cfg["key"])


def get_signed_url(dataset: str, auth: tuple[str, str]) -> str:
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}"
    response = requests.get(url, auth=auth, allow_redirects=False, stream=True, timeout=60)
    response.close()
    if response.status_code not in (301, 302, 303, 307, 308):
        body = response.text[:500] if hasattr(response, "text") else ""
        raise RuntimeError(f"expected redirect for {dataset}, got {response.status_code}: {body}")
    location = response.headers.get("Location")
    if not location:
        raise RuntimeError(f"redirect response for {dataset} did not include Location")
    return location


def remote_size(url: str) -> int:
    response = requests.head(url, allow_redirects=True, timeout=60)
    response.close()
    if response.status_code >= 400:
        return 0
    try:
        return int(response.headers.get("Content-Length", "0"))
    except ValueError:
        return 0


def download_signed_url(
    url: str,
    outfile: Path,
    report_interval: float,
    chunk_size: int = 8 * 1024 * 1024,
    max_attempts: int = 8,
) -> None:
    tmp = outfile.with_suffix(outfile.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_report = started
    last_bytes = tmp.stat().st_size if tmp.is_file() else 0
    written = last_bytes
    total = 0

    def report(final: bool = False) -> None:
        nonlocal last_report, last_bytes
        now = time.monotonic()
        elapsed = max(now - started, 1e-9)
        interval = max(now - last_report, 1e-9)
        inst = max(written - last_bytes, 0) / interval
        avg = written / elapsed
        remaining = max(total - written, 0) if total else 0
        eta = remaining / avg if avg > 0 and total else 0
        pct = 100.0 * written / total if total else 0.0
        tag = "done" if final else "progress"
        print(
            f"[{tag}] {outfile.name} {pct:6.2f}% "
            f"{human_bytes(written)}/{human_bytes(total) if total else 'unknown'} "
            f"inst={human_bytes(inst)}/s avg={human_bytes(avg)}/s "
            f"elapsed={human_duration(elapsed)} eta={human_duration(eta)}",
            flush=True,
        )
        last_report = now
        last_bytes = written

    print(f"[download] {outfile.name} -> {outfile.parent}", flush=True)
    for attempt in range(1, max_attempts + 1):
        headers = {}
        existing = tmp.stat().st_size if tmp.is_file() else 0
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
        try:
            response = requests.get(url, stream=True, headers=headers, timeout=(180, 180))
            try:
                if response.status_code == 416 and existing > 0:
                    tmp.replace(outfile)
                    report(final=True)
                    return
                response.raise_for_status()
                content_length = int(response.headers.get("Content-Length") or 0)
                if response.status_code == 206 and existing > 0:
                    total = existing + content_length
                    mode = "ab"
                    written = existing
                else:
                    total = content_length
                    mode = "wb"
                    written = 0
                print(
                    f"[attempt {attempt}/{max_attempts}] status={response.status_code} "
                    f"size={human_bytes(total) if total else 'unknown'} resume={human_bytes(existing)}",
                    flush=True,
                )
                with tmp.open(mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last_report >= report_interval:
                            report()
                if total and written < total:
                    raise RuntimeError(f"incomplete download {written}/{total}")
                tmp.replace(outfile)
                report(final=True)
                return
            finally:
                response.close()
        except Exception as exc:
            print(f"[retry] attempt={attempt}/{max_attempts} error={exc!r}", flush=True)
            if attempt == max_attempts:
                raise
            time.sleep(min(5 * attempt, 30))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Orbit Wars daily episode datasets as zip files with speed/ETA logs.")
    parser.add_argument("--dates", nargs="+", required=True, help="Dates like 2026-06-16.")
    parser.add_argument("--out-dir", type=Path, default=Path("/mnt/f/kaggle-orbit-wars-data/daily_zips"))
    parser.add_argument("--kaggle-config-dir", type=Path, default=Path("/home/dministrator/.config/kaggle"))
    parser.add_argument("--force", action="store_true", help="Redownload existing zip files.")
    parser.add_argument("--report-interval", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auth = load_kaggle_auth(args.kaggle_config_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    for index, date in enumerate(args.dates, start=1):
        dataset = f"kaggle/orbit-wars-episodes-{date}"
        outfile = args.out_dir / f"orbit-wars-episodes-{date}.zip"
        if outfile.is_file() and not args.force:
            print(f"[{index}/{len(args.dates)}] skip existing {outfile} size={human_bytes(outfile.stat().st_size)}", flush=True)
            continue

        print(f"[{index}/{len(args.dates)}] request signed URL for {dataset}", flush=True)
        day_started = time.monotonic()
        signed_url = get_signed_url(dataset, auth)
        print(f"[{index}/{len(args.dates)}] signed URL ready in {human_duration(time.monotonic() - day_started)}", flush=True)
        download_signed_url(signed_url, outfile, float(args.report_interval))
        elapsed = time.monotonic() - day_started
        print(f"[day-done] {date} elapsed={human_duration(elapsed)} file={outfile}", flush=True)

    print(f"[all-done] dates={len(args.dates)} elapsed={human_duration(time.monotonic() - started)} out_dir={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
