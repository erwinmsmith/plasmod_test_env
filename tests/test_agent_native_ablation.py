from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "agent_native_ablation_benchmark.py"
SPEC = importlib.util.spec_from_file_location("agent_native_ablation_benchmark", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


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
