from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "agent_native_ablation_benchmark.py"
SPEC = importlib.util.spec_from_file_location("agent_native_ablation_benchmark", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_smoke_launcher_supports_empty_default_arguments_under_nounset(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    launcher = SCRIPT.with_name("run_agent_native_ablation.sh")
    env = os.environ | {"PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}

    completed = subprocess.run(
        ["/bin/bash", str(launcher), "smoke", "--port", "18080"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_percentile_and_hash_vector_are_deterministic():
    assert MODULE.percentile([1, 2, 3, 4], 0.5) == 2.5
    first = MODULE.hash_vector("agent state transition")
    second = MODULE.hash_vector("agent state transition")
    assert first == second
    assert len(first) == 384
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9


def test_write_csv_requires_every_metric(tmp_path):
    fields = ["System", "Variant", "Metric"]
    path = tmp_path / "table.csv"
    MODULE.write_csv(path, fields, [{"System": "Plasmod", "Variant": "Full", "Metric": 1.25}])
    assert path.read_text(encoding="utf-8").splitlines() == [
        "System,Variant,Metric",
        "Plasmod,Full,1.25",
    ]
    with pytest.raises(RuntimeError, match="missing fields"):
        MODULE.write_csv(path, fields, [{"System": "Plasmod", "Variant": "Broken"}])


def test_every_ablation_table_has_unique_complete_headers():
    tables = [
        MODULE.WAL_FIELDS,
        MODULE.MATERIALIZATION_FIELDS,
        MODULE.EVIDENCE_FIELDS,
        MODULE.GOVERNANCE_FIELDS,
        MODULE.TIER_FIELDS,
    ]
    for fields in tables:
        assert fields[:2] == ["System", "Variant"]
        assert len(fields) == len(set(fields))
        assert len(fields) >= 8
    assert len(MODULE.MASTER_FIELDS) == len(set(MODULE.MASTER_FIELDS))
    assert MODULE.MASTER_FIELDS[:5] == [
        "System", "Module", "Original Variant", "Comparison Label", "Ablated Capability",
    ]


def test_ablation_matrix_covers_all_declared_variants_and_metrics():
    variants = [
        MODULE.recovery_variants(),
        MODULE.materialization_variants(),
        MODULE.evidence_variants(),
        MODULE.governance_variants(),
        MODULE.tier_variants(),
    ]
    assert [len(group) for group in variants] == [6, 7, 7, 6, 8]
    assert sum(len(group) for group in variants) == 34
    assert {variant.name for group in variants for variant in group} >= {
        "No-WAL", "WAL without replay", "No-materialization", "No-agent-state",
        "No-evidence", "No-proof-trace", "No-access-policy", "No-delete-propagation",
        "No-hot-cache", "No-promotion", "Hot-size-2000",
    }
    assert len(MODULE.all_variants()) == 34
    for variant in MODULE.all_variants():
        label, capability = MODULE.COMPARISON_LABELS[variant.group][variant.name]
        assert label
        assert capability


def test_master_table_has_common_metrics_and_explicit_not_applicable_cells(tmp_path):
    tables = {}
    field_sets = {
        "wal": MODULE.WAL_FIELDS,
        "materialization": MODULE.MATERIALIZATION_FIELDS,
        "evidence": MODULE.EVIDENCE_FIELDS,
        "governance": MODULE.GOVERNANCE_FIELDS,
        "tier": MODULE.TIER_FIELDS,
    }
    variants_by_group = {
        "wal": MODULE.recovery_variants(),
        "materialization": MODULE.materialization_variants(),
        "evidence": MODULE.evidence_variants(),
        "governance": MODULE.governance_variants(),
        "tier": MODULE.tier_variants(),
    }
    for group, variants in variants_by_group.items():
        tables[group] = []
        for variant in variants:
            fields = field_sets[group]
            tables[group].append({
                field: ("yes" if field == "Query Available During Recovery" else 1)
                for field in fields
            } | {"System": "Plasmod", "Variant": variant.name})
            variant_dir = tmp_path / "variants" / variant.slug
            variant_dir.mkdir(parents=True)
            (variant_dir / "common_metrics.json").write_text(json.dumps({
                "metrics": {field: 1 for field in MODULE.COMMON_FIELDS},
            }), encoding="utf-8")

    rows = MODULE.build_master_table(tmp_path, tables)
    assert len(rows) == 34
    assert all(all(row[field] not in (None, "") for field in MODULE.MASTER_FIELDS) for row in rows)
    wal_row = next(row for row in rows if row["Module"] == "wal")
    assert wal_row["WAL | Event Log Size"] == 1
    assert wal_row["EVIDENCE | Query p95 (ms)"] == MODULE.NOT_APPLICABLE


def test_governance_events_use_independent_sessions():
    private = MODULE.governance_event("private", "e1", "agent-a", "private")
    shared = MODULE.governance_event("shared", "e2", "agent-a", "restricted_shared")
    assert private["actor"]["session_id"] == "governance-private"
    assert shared["actor"]["session_id"] == "governance-shared"
    assert private["actor"]["session_id"] != shared["actor"]["session_id"]


def test_service_log_validation_rejects_s3_and_process_errors(tmp_path):
    log_dir = tmp_path / "variants" / "full"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "server.log"
    log_path.write_text("Plasmod started\ns3cold: cold lexical search returned=1\n", encoding="utf-8")
    MODULE.validate_service_logs(tmp_path)

    log_path.write_text(
        "s3cold: get memory id: s3 get do: dial tcp: can't assign requested address\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="fatal service log entries"):
        MODULE.validate_service_logs(tmp_path)
