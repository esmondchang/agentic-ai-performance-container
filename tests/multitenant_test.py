#!/usr/bin/env python3
"""Concurrent workflow test with a separate persistent RAG store per tenant."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def percentile(values: list[float], rank: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((rank / 100) * (len(ordered) - 1))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate concurrent users with tenant-isolated FAISS stores."
    )
    parser.add_argument("--tenants", type=int, default=2)
    parser.add_argument("--users-per-tenant", type=int, default=2)
    parser.add_argument("--requests-per-user", type=int, default=1)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--query", default="Should I invest in this stock?")
    parser.add_argument("--tenant-root", default="./data/tenant_test")
    parser.add_argument("--output", default="multitenant_results.json")
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep tenant vector stores after the isolation audit.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("tenants", "users_per_tenant", "requests_per_user"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")


def run_test(args: argparse.Namespace) -> dict[str, Any]:
    from src.workflow import FinancialAgentWorkflow

    tenant_root = Path(args.tenant_root).resolve()
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    run_root = tenant_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    tenant_paths = {
        f"tenant-{number:03d}": run_root / f"tenant-{number:03d}"
        for number in range(1, args.tenants + 1)
    }
    for tenant_id, path in tenant_paths.items():
        path.mkdir(parents=True, exist_ok=False)
        (path / "tenant-id.txt").write_text(tenant_id, encoding="utf-8")

    start_event = threading.Event()

    def run_user(tenant_id: str, user_number: int) -> dict[str, Any]:
        store = tenant_paths[tenant_id]
        workflow = FinancialAgentWorkflow(persist_directory=str(store))
        start_event.wait()
        requests = []
        for request_number in range(1, args.requests_per_user + 1):
            started = time.perf_counter()
            result = workflow.analyze(ticker=args.ticker, query=args.query)
            requests.append({
                "request": request_number,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "ok": not bool(result.get("error")),
                "error": result.get("error"),
            })
        return {
            "tenant_id": tenant_id,
            "user_id": f"user-{user_number:03d}",
            "store": str(store),
            "requests": requests,
        }

    jobs = [
        (tenant_id, user_number)
        for tenant_id in tenant_paths
        for user_number in range(1, args.users_per_tenant + 1)
    ]
    print(
        f"Starting {len(jobs)} concurrent user(s): {args.tenants} tenant(s), "
        f"{args.users_per_tenant} user(s) per tenant, "
        f"{args.requests_per_user} request(s) per user"
    )

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(run_user, *job) for job in jobs]
        start_event.set()
        users = [future.result() for future in concurrent.futures.as_completed(futures)]
    duration = time.perf_counter() - started

    all_requests = [request for user in users for request in user["requests"]]
    latencies = [request["latency_ms"] for request in all_requests if request["ok"]]
    expected_markers = {
        tenant_id: (path / "tenant-id.txt").read_text(encoding="utf-8")
        for tenant_id, path in tenant_paths.items()
    }
    unique_paths = len({str(path) for path in tenant_paths.values()}) == args.tenants
    markers_match = all(key == value for key, value in expected_markers.items())
    stores_created = all((path / "faiss_index").is_dir() for path in tenant_paths.values())
    isolation_passed = unique_paths and markers_match and stores_created

    result = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "tenants": args.tenants,
        "users_per_tenant": args.users_per_tenant,
        "concurrent_users": len(jobs),
        "requests_per_user": args.requests_per_user,
        "total_requests": len(all_requests),
        "successful_requests": len(latencies),
        "failed_requests": len(all_requests) - len(latencies),
        "success_rate": round(len(latencies) / len(all_requests), 4),
        "duration_seconds": round(duration, 2),
        "throughput_requests_per_second": round(len(latencies) / duration, 4),
        "latency": {
            "average_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "p50_ms": round(percentile(latencies, 50), 2),
            "p95_ms": round(percentile(latencies, 95), 2),
        },
        "isolation_audit": {
            "passed": isolation_passed,
            "unique_tenant_paths": unique_paths,
            "tenant_markers_match": markers_match,
            "faiss_stores_created": stores_created,
            "tenant_paths": {key: str(value) for key, value in tenant_paths.items()},
        },
        "users": users,
        "_run_root": str(run_root),
    }
    return result


def main() -> int:
    args = parse_args()
    validate_args(args)
    result = run_test(args)
    run_root = result.pop("_run_root")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not args.keep_data:
        shutil.rmtree(run_root)

    print("\nMULTI-TENANT TEST RESULTS")
    print(f"Concurrent users : {result['concurrent_users']}")
    print(f"Total requests   : {result['total_requests']}")
    print(f"Success rate     : {result['success_rate']:.2%}")
    print(f"P50 / P95        : {result['latency']['p50_ms']:.2f} / {result['latency']['p95_ms']:.2f} ms")
    print(f"Isolation audit  : {'PASS' if result['isolation_audit']['passed'] else 'FAIL'}")
    print(f"Results saved to : {output}")
    print(f"Tenant data      : {'kept' if args.keep_data else 'removed after audit'}")

    return 0 if result["failed_requests"] == 0 and result["isolation_audit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
