"""Download a file in byte ranges with resumable part files.

This is useful for public data hosts that advertise Accept-Ranges but frequently
close long TLS connections. Each chunk is saved independently; completed chunks
are skipped on rerun and then concatenated into the final file.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
from pathlib import Path


def _download_part(url: str, part: Path, start: int, end: int, expected: int, retries: int) -> tuple[Path, bool, str]:
    if part.is_file() and part.stat().st_size == expected:
        return part, True, "skip"
    tmp = part.with_suffix(part.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        if tmp.exists():
            tmp.unlink()
        cmd = [
            "curl.exe",
            "-L",
            "--fail",
            "--ssl-no-revoke",
            "--http1.1",
            "--retry",
            "3",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "60",
            "-r",
            f"{start}-{end}",
            "-o",
            str(tmp),
            url,
        ]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        size = tmp.stat().st_size if tmp.exists() else 0
        if proc.returncode == 0 and size == expected:
            tmp.replace(part)
            return part, False, f"ok attempt={attempt}"
        if tmp.exists():
            tmp.unlink()
        last = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        message = " | ".join(last)
    raise RuntimeError(f"failed part {part.name} range={start}-{end} expected={expected}: {message}")


def _concat(parts: list[Path], out: Path, expected_total: int) -> None:
    tmp = out.with_suffix(out.suffix + ".assembled")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("wb") as dst:
        for part in parts:
            with part.open("rb") as src:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
    if tmp.stat().st_size != expected_total:
        raise RuntimeError(f"assembled size mismatch: {tmp.stat().st_size} != {expected_total}")
    tmp.replace(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--part-size", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--force-assemble", action="store_true")
    args = parser.parse_args()

    if args.out.is_file() and args.out.stat().st_size == args.size and not args.force_assemble:
        print(f"[done] existing complete file: {args.out}")
        return 0

    part_dir = args.out.with_suffix(args.out.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    ranges: list[tuple[int, int, Path, int]] = []
    start = 0
    idx = 0
    while start < args.size:
        end = min(args.size - 1, start + args.part_size - 1)
        expected = end - start + 1
        ranges.append((start, end, part_dir / f"part_{idx:04d}.bin", expected))
        start = end + 1
        idx += 1

    print(
        {
            "parts": len(ranges),
            "part_size": args.part_size,
            "jobs": args.jobs,
            "out": str(args.out),
            "proxy": {key: os.environ.get(key, "") for key in ["HTTP_PROXY", "HTTPS_PROXY"]},
        },
        flush=True,
    )
    parts_by_index = [part for _, _, part, _ in ranges]
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        future_map = {
            pool.submit(_download_part, args.url, part, start, end, expected, args.retries): (i, start, end, part, expected)
            for i, (start, end, part, expected) in enumerate(ranges)
        }
        for future in cf.as_completed(future_map):
            i, start, end, part, expected = future_map[future]
            _, skipped, message = future.result()
            status = "skip" if skipped else "done"
            print(f"[{status}] {i + 1}/{len(ranges)} {part.name} {start}-{end} {expected} {message}", flush=True)

    _concat(parts_by_index, args.out, args.size)
    print(f"[assembled] {args.out} bytes={args.out.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
