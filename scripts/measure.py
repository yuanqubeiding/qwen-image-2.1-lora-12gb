#!/usr/bin/env python
"""Run a command and record host RAM, page file and GPU memory over its lifetime.

This is the sampler that produced `assets/measured_resources.csv`. It is generic —
give it any command:

    python scripts/measure.py --out run_dir -- python train.py --args ...
    python scripts/measure.py --out run_dir --interval 1 -- nvidia-smi

Writes into --out:
    resources.csv            one row per sample
    resources_summary.json   peaks and baselines
    command.log              the command's stdout+stderr

Requires psutil. GPU sampling needs nvidia-smi on PATH; without it the gpu column
is -1 and the run still completes.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil

FIELDS = ["t", "elapsed_s", "rss_mb", "ram_used_mb", "ram_avail_mb",
          "pagefile_used_mb", "gpu_used_mb", "cpu_pct"]


def gpu_used_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def tree_rss_mb(pproc):
    total = 0
    try:
        total += pproc.memory_info().rss
        for child in pproc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return total / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory for csv/json/log")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between samples")
    ap.add_argument("--cwd", default=None, help="working directory for the command")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="command to run, after --")
    args = ap.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given (put it after --)")

    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, "command.log"), "w", encoding="utf-8",
              errors="replace") as log:
        base_gpu = gpu_used_mb()
        vm0 = psutil.virtual_memory()
        sw0 = psutil.swap_memory()

        proc = subprocess.Popen(cmd, cwd=args.cwd, stdout=log,
                                stderr=subprocess.STDOUT)
        pproc = psutil.Process(proc.pid)
        pproc.cpu_percent(interval=None)  # prime the counter

        peak = {"rss_mb": 0.0, "ram_used_mb": 0.0, "pagefile_used_mb": 0.0,
                "gpu_used_mb": 0, "gpu_delta_mb": 0}
        min_avail = float("inf")
        t0 = time.time()
        samples = 0

        csv_path = os.path.join(args.out, "resources.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            while True:
                rc = proc.poll()
                elapsed = time.time() - t0
                rss = tree_rss_mb(pproc) if rc is None else 0.0
                vm = psutil.virtual_memory()
                sw = psutil.swap_memory()
                gpu = gpu_used_mb()
                row = {
                    "t": datetime.now().strftime("%H:%M:%S"),
                    "elapsed_s": round(elapsed, 1),
                    "rss_mb": round(rss, 1),
                    "ram_used_mb": round(vm.used / 2**20, 1),
                    "ram_avail_mb": round(vm.available / 2**20, 1),
                    "pagefile_used_mb": round(sw.used / 2**20, 1),
                    "gpu_used_mb": gpu,
                    "cpu_pct": round(pproc.cpu_percent(interval=None), 1) if rc is None else 0.0,
                }
                w.writerow(row)
                fh.flush()
                samples += 1

                peak["rss_mb"] = max(peak["rss_mb"], rss)
                peak["ram_used_mb"] = max(peak["ram_used_mb"], row["ram_used_mb"])
                peak["pagefile_used_mb"] = max(peak["pagefile_used_mb"], row["pagefile_used_mb"])
                peak["gpu_used_mb"] = max(peak["gpu_used_mb"], gpu)
                peak["gpu_delta_mb"] = max(peak["gpu_delta_mb"], gpu - base_gpu)
                min_avail = min(min_avail, row["ram_avail_mb"])

                if rc is not None:
                    break
                time.sleep(args.interval)

        duration = time.time() - t0
        summary = {
            "command": cmd,
            "exit_code": proc.poll(),
            "duration_s": round(duration, 1),
            "samples": samples,
            "interval_s": args.interval,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total_ram_mb": round(vm0.total / 2**20, 1),
            "baseline_gpu_mb": base_gpu,
            "baseline_ram_used_mb": round(vm0.used / 2**20, 1),
            "baseline_pagefile_used_mb": round(sw0.used / 2**20, 1),
            "peak_train_process_rss_mb": round(peak["rss_mb"], 1),
            "peak_system_ram_used_mb": round(peak["ram_used_mb"], 1),
            "min_ram_available_mb": round(min_avail, 1),
            "peak_pagefile_used_mb": round(peak["pagefile_used_mb"], 1),
            "peak_gpu_used_mb": peak["gpu_used_mb"],
            "peak_gpu_delta_mb": peak["gpu_delta_mb"],
        }
        with open(os.path.join(args.out, "resources_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["exit_code"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
