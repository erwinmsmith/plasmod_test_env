#!/usr/bin/env python3
"""Prepare a sanitized Hugging Face release of the agent event data assets.

The source JSONL files are never modified. The output mirrors the source
layout, replaces credential-shaped strings recursively in JSON values, and
writes a release manifest with record counts and redaction totals.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE / "data" / "layer2_dynamic_events"

REDACTIONS: dict[str, re.Pattern[str]] = {
    "github_pat": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "openai_key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "slack_token": re.compile(r"xox[baprs]-[0-9A-Za-z-]{20,}"),
    # Some source-derived snippets contain malformed or truncated key blocks.
    # Redact from any private-key header through the matching footer, or through
    # the end of that JSON string when no footer is present.
    "private_key_block": re.compile(
        r"-----BEGIN [^-\n]{0,64}PRIVATE KEY(?: BLOCK)?-----"
        r"(?:.*?-----END [^-\n]{0,64}PRIVATE KEY(?: BLOCK)?-----|.*)",
        re.DOTALL,
    ),
}


def redact_string(value: str, counts: Counter[str]) -> str:
    for name, pattern in REDACTIONS.items():
        value, replaced = pattern.subn(f"[REDACTED_{name.upper()}]", value)
        counts[name] += replaced
    return value


def redact_value(value: Any, counts: Counter[str]) -> Any:
    if isinstance(value, str):
        return redact_string(value, counts)
    if isinstance(value, list):
        return [redact_value(item, counts) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item, counts) for key, item in value.items()}
    return value


def jsonl_files(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*.jsonl") if item.is_file())


def sanitize_jsonl(source: Path, destination: Path, counts: Counter[str]) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    with source.open("r", encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {source}:{line_number}: {exc}") from exc
            writer.write(json.dumps(redact_value(record, counts), ensure_ascii=False, separators=(",", ":")))
            writer.write("\n")
            records += 1
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a sanitized release copy of agent event JSONL data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    traces = input_root / "traces_collected"
    replay = input_root / "events.jsonl"
    if not traces.is_dir() or not replay.is_file():
        raise SystemExit(f"expected traces_collected/ and events.jsonl under {input_root}")
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {output_root}; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    counts: Counter[str] = Counter()
    trace_records = 0
    trace_files = jsonl_files(traces)
    for index, source in enumerate(trace_files, start=1):
        trace_records += sanitize_jsonl(source, output_root / source.relative_to(input_root), counts)
        if index % 1000 == 0 or index == len(trace_files):
            print(f"sanitized {index}/{len(trace_files)} trace shards", flush=True)
    replay_records = sanitize_jsonl(replay, output_root / "events.jsonl", counts)

    manifest = {
        "format": "jsonl",
        "release": "sanitized",
        "sources": {
            "traces_collected": {"files": len(trace_files), "records": trace_records},
            "events_jsonl": {"files": 1, "records": replay_records},
        },
        "redactions": dict(sorted(counts.items())),
        "redaction_markers": {name: f"[REDACTED_{name.upper()}]" for name in REDACTIONS},
    }
    (output_root / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
