#!/usr/bin/env python3
"""Run a command and correlate LangGraph node events with host CPU/GPU use.

The command must write RESOURCE_TRACE_FILE events to a host-visible bind mount.
This monitor is intended to run on the Linux host when Ollama is a local
systemd service and the application runs in Docker.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure host resources during traced LangGraph tasks."
    )
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-devices", default="0,1")
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--gpu-active-threshold", type=float, default=1.0)
    parser.add_argument(
        "--app-pattern",
        default="tests/multitenant_test.py",
        help="Substring used to identify application processes in /proc.",
    )
    parser.add_argument(
        "--ollama-pattern",
        action="append",
        default=["ollama", "llama-server"],
        help="Substring identifying Ollama processes; may be repeated.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    return args


def read_host_cpu() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    values = [int(value) for value in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def process_cpu_times(
    patterns: list[str], executable_prefixes: list[str] | None = None
) -> dict[int, float]:
    ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    result: dict[int, float] = {}
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            arguments = [
                value.decode("utf-8", errors="replace")
                for value in (proc_dir / "cmdline").read_bytes().split(b"\0")
                if value
            ]
            if not arguments or int(proc_dir.name) == os.getpid():
                continue
            executable = Path(arguments[0]).name
            if executable_prefixes and not any(
                executable.startswith(prefix) for prefix in executable_prefixes
            ):
                continue
            cmdline = " ".join(arguments)
            if not any(pattern in cmdline for pattern in patterns):
                continue
            stat = (proc_dir / "stat").read_text(encoding="utf-8")
            close_paren = stat.rfind(")")
            fields = stat[close_paren + 2 :].split()
            user_ticks = int(fields[11])
            system_ticks = int(fields[12])
            result[int(proc_dir.name)] = (user_ticks + system_ticks) / ticks
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return result


def query_gpus(devices: list[str]) -> dict[str, dict[str, float]]:
    command = [
        "nvidia-smi",
        f"--id={','.join(devices)}",
        "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    metrics: dict[str, dict[str, float]] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        index, utilization, memory = fields
        metrics[index] = {
            "utilization_pct": float(utilization),
            "memory_mb": float(memory),
        }
    return metrics


def cpu_delta(
    current: dict[int, float], previous: dict[int, float]
) -> float:
    return sum(
        max(0.0, value - previous[pid])
        for pid, value in current.items()
        if pid in previous
    )


def take_sample(
    previous: dict[str, Any],
    app_pattern: str,
    ollama_patterns: list[str],
    gpu_devices: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    timestamp_ns = time.time_ns()
    host_total, host_idle = read_host_cpu()
    app_times = process_cpu_times([app_pattern], executable_prefixes=["python"])
    ollama_times = process_cpu_times(
        ollama_patterns, executable_prefixes=["ollama", "llama-server"]
    )
    gpus = query_gpus(gpu_devices)

    elapsed = max((timestamp_ns - previous["timestamp_ns"]) / 1e9, 1e-9)
    total_delta = host_total - previous["host_total"]
    idle_delta = host_idle - previous["host_idle"]
    host_busy = (
        100.0 * (total_delta - idle_delta) / total_delta if total_delta > 0 else 0.0
    )
    sample = {
        "start_ns": previous["timestamp_ns"],
        "end_ns": timestamp_ns,
        "interval_seconds": elapsed,
        "app_cpu_seconds": cpu_delta(app_times, previous["app_times"]),
        "ollama_cpu_seconds": cpu_delta(ollama_times, previous["ollama_times"]),
        "host_cpu_capacity_pct": host_busy,
        "gpu": gpus,
    }
    state = {
        "timestamp_ns": timestamp_ns,
        "host_total": host_total,
        "host_idle": host_idle,
        "app_times": app_times,
        "ollama_times": ollama_times,
    }
    return sample, state


def load_task_windows(trace_file: Path) -> list[dict[str, Any]]:
    starts: dict[str, dict[str, Any]] = {}
    windows: list[dict[str, Any]] = []
    for line in trace_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        task_id = event["task_id"]
        if event["phase"] == "start":
            starts[task_id] = event
        elif event["phase"] == "end" and task_id in starts:
            start = starts.pop(task_id)
            windows.append({
                "task_id": task_id,
                "trace_id": start["trace_id"],
                "task": start["label"],
                "node": start["node"],
                "start_ns": start["timestamp_ns"],
                "end_ns": event["timestamp_ns"],
                "status": event.get("status", "unknown"),
            })
    return sorted(windows, key=lambda item: item["start_ns"])


def summarize_task(
    window: dict[str, Any],
    samples: list[dict[str, Any]],
    gpu_devices: list[str],
    active_threshold: float,
) -> dict[str, Any]:
    start_ns, end_ns = window["start_ns"], window["end_ns"]
    wall_seconds = max((end_ns - start_ns) / 1e9, 0.0)
    app_cpu = 0.0
    ollama_cpu = 0.0
    covered_seconds = 0.0
    host_weighted = 0.0
    gpu_summary = {
        device: {"active_seconds": 0.0, "weighted_util": 0.0, "peak_memory_mb": 0.0}
        for device in gpu_devices
    }

    for sample in samples:
        overlap_ns = max(
            0,
            min(end_ns, sample["end_ns"]) - max(start_ns, sample["start_ns"]),
        )
        if overlap_ns == 0:
            continue
        overlap = overlap_ns / 1e9
        fraction = min(overlap / sample["interval_seconds"], 1.0)
        covered_seconds += overlap
        app_cpu += sample["app_cpu_seconds"] * fraction
        ollama_cpu += sample["ollama_cpu_seconds"] * fraction
        host_weighted += sample["host_cpu_capacity_pct"] * overlap
        for device in gpu_devices:
            gpu = sample["gpu"].get(device)
            if not gpu:
                continue
            utilization = gpu["utilization_pct"]
            gpu_summary[device]["weighted_util"] += utilization * overlap
            if utilization >= active_threshold:
                gpu_summary[device]["active_seconds"] += overlap
            gpu_summary[device]["peak_memory_mb"] = max(
                gpu_summary[device]["peak_memory_mb"], gpu["memory_mb"]
            )

    gpu_result = {}
    for device, gpu in gpu_summary.items():
        gpu_result[device] = {
            "active_time_pct": round(
                100.0 * gpu["active_seconds"] / covered_seconds, 2
            ) if covered_seconds else 0.0,
            "average_utilization_pct": round(
                gpu["weighted_util"] / covered_seconds, 2
            ) if covered_seconds else 0.0,
            "peak_memory_mb": round(gpu["peak_memory_mb"], 2),
        }

    return {
        "task_id": window["task_id"],
        "trace_id": window["trace_id"],
        "task": window["task"],
        "node": window["node"],
        "status": window["status"],
        "wall_time_ms": round(wall_seconds * 1000, 2),
        "app_cpu_time_ms": round(app_cpu * 1000, 2),
        "ollama_cpu_time_ms": round(ollama_cpu * 1000, 2),
        "host_cpu_capacity_pct": round(
            host_weighted / covered_seconds, 2
        ) if covered_seconds else 0.0,
        "sample_coverage_pct": round(
            min(100.0 * covered_seconds / wall_seconds, 100.0), 2
        ) if wall_seconds else 0.0,
        "gpu": gpu_result,
    }


def main() -> int:
    args = parse_args()
    trace_file = Path(args.trace_file).resolve()
    output = Path(args.output).resolve()
    gpu_devices = [item.strip() for item in args.gpu_devices.split(",") if item.strip()]
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_file.unlink(missing_ok=True)

    if not Path("/proc/stat").exists():
        raise SystemExit("This monitor must run on the Linux host.")
    try:
        query_gpus(gpu_devices)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Unable to query NVIDIA GPUs {gpu_devices}: {exc}") from exc

    total, idle = read_host_cpu()
    state = {
        "timestamp_ns": time.time_ns(),
        "host_total": total,
        "host_idle": idle,
        "app_times": process_cpu_times(
            [args.app_pattern], executable_prefixes=["python"]
        ),
        "ollama_times": process_cpu_times(
            args.ollama_pattern,
            executable_prefixes=["ollama", "llama-server"],
        ),
    }
    samples: list[dict[str, Any]] = []
    process = subprocess.Popen(args.command)
    try:
        while process.poll() is None:
            time.sleep(args.interval)
            sample, state = take_sample(
                state,
                args.app_pattern,
                args.ollama_pattern,
                gpu_devices,
            )
            samples.append(sample)
    except KeyboardInterrupt:
        process.terminate()
    finally:
        return_code = process.wait()

    if not trace_file.exists():
        raise SystemExit(
            f"No trace file was created at {trace_file}. Pass "
            "-e RESOURCE_TRACE_FILE=<container path> to docker compose run and "
            "bind-mount that path to the host trace file directory."
        )

    windows = load_task_windows(trace_file)
    tasks = [
        summarize_task(
            window,
            samples,
            gpu_devices,
            args.gpu_active_threshold,
        )
        for window in windows
    ]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": args.command,
        "sample_interval_seconds": args.interval,
        "gpu_active_threshold_pct": args.gpu_active_threshold,
        "concurrency_note": (
            "Overlapping task windows share host and GPU measurements; values are "
            "not exclusive per user when concurrency is greater than one."
        ),
        "tasks": tasks,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Resource report saved to: {output}")
    print(f"Traced tasks: {len(tasks)}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
