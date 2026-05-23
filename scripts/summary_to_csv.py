#!/usr/bin/env python3
"""Convert benchmark summary.json to CSV."""

import json
import csv
import sys
from pathlib import Path


def convert_summary_to_csv(input_path, output_path=None):
    with open(input_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("No results found in JSON", file=sys.stderr)
        sys.exit(1)

    fieldnames = [
        "db", "index_type",
        "build_ms", "batch_ms", "batch_qps",
        "serial_qps", "recall",
        "mean_ms", "p50", "p95", "p99",
        "memory_mb",
    ]

    rows = []
    for r in results:
        rows.append({
            "db": r.get("db", ""),
            "index_type": r.get("index_type", ""),
            "build_ms": round(r.get("build_ms", 0), 2),
            "batch_ms": round(r.get("batch_ms", 0), 2),
            "batch_qps": round(r.get("batch_qps", 0), 2),
            "serial_qps": round(r.get("serial_qps", 0), 2),
            "recall": round(r.get("recall", 0), 4),
            "mean_ms": round(r.get("p50_ms", 0), 2),
            "p50": round(r.get("p50_ms", 0), 2),
            "p95": round(r.get("p95_ms", 0), 2),
            "p99": round(r.get("p99_ms", 0), 2),
            "memory_mb": round(r.get("memory_mb", 0), 2),
        })

    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {output_path}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <summary.json> [output.csv]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    convert_summary_to_csv(input_file, output_file)