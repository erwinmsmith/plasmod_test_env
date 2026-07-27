from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import namedtuple

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


def test_preflight_accepts_only_declared_retention_modes():
    preflight = SCRIPT.with_name("preflight_agent_native_ablation.sh")
    completed = subprocess.run(
        ["/bin/bash", str(preflight), "full", "--retention", "discard-everything"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "invalid retention mode" in completed.stderr
    source = preflight.read_text(encoding="utf-8")
    assert 'PLASMOD_ABLATION_MIN_FREE_GB:-40' in source
    assert 'PLASMOD_ABLATION_MIN_FREE_GB:-250' in source


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


def test_physical_matrix_runs_one_shared_full_and_29_non_full_variants():
    physical = MODULE.physical_variants()

    assert len(physical) == 30
    assert physical[0] == MODULE.shared_full_variant()
    assert sum(variant == MODULE.shared_full_variant() for variant in physical) == 1
    assert all(
        not MODULE.is_group_full_variant(variant)
        for variant in physical[1:]
    )
    assert len(physical[1:]) == 29


def test_shared_full_baseline_round_trips_resume_state(tmp_path):
    baseline = MODULE.SharedFullBaseline(
        module_rows={
            "wal": {"System": "Plasmod", "Variant": "Full Plasmod", "Event Log Size": 8},
            "materialization": {"System": "Plasmod", "Variant": "Full Plasmod", "Write QPS": 10.0},
            "evidence": {"System": "Plasmod", "Variant": "Full Plasmod", "Query p95 (ms)": 2.0},
            "tier": {"System": "Plasmod", "Variant": "Full Tiering", "Hot Cache Size": 2000},
        },
        materialization_type_counts={"events": 8, "states": 2},
        state_ids={"state-2", "state-1"},
        artifact_ids={"artifact-1"},
        contexts={
            "state-1": ("agent-a", "workspace-a", "session-a"),
            "artifact-1": ("agent-b", "workspace-a", "session-b"),
        },
        evidence_totals=(3, 4, 5, 2),
        governance_measurement={"query_latency": 1.25, "private": False},
    )
    path = tmp_path / "shared_full_baseline.json"

    baseline.write(path)
    loaded = MODULE.SharedFullBaseline.read(path)

    assert loaded == baseline
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "shared-full-baseline-v1"
    assert payload["state_ids"] == ["state-1", "state-2"]


def test_metrics_only_retention_checkpoints_before_removing_variant_data(tmp_path, monkeypatch):
    variant = MODULE.Variant("wal", "No-WAL")
    variant_dir = tmp_path / "variants" / variant.slug
    data_dir = variant_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "badger.data").write_bytes(b"database-bytes")
    for name in (
        "capabilities.json", "measurements.json", "common_metrics.json",
        "server.log", "recovery.json",
    ):
        (variant_dir / name).write_text("{}", encoding="utf-8")
    manager = MODULE.RetentionManager(tmp_path, "metrics-only", disk_floor_gb=10)
    deleted_prefixes = []
    monkeypatch.setattr(manager, "_delete_s3_prefix", deleted_prefixes.append)
    fields = ["System", "Variant", "Metric"]
    row = {"System": "Plasmod", "Variant": "No-WAL", "Metric": 1.5}

    manager.record_variant(variant, "wal", fields, row)

    assert not data_dir.exists()
    assert (variant_dir / "result_checkpoint.json").exists()
    assert (variant_dir / "METRICS_RETAINED").exists()
    assert (variant_dir / "common_metrics.json").exists()
    assert deleted_prefixes == [variant]
    assert manager.load_variant_row(variant, "wal", fields) == row


def test_full_retention_keeps_variant_data_while_writing_checkpoint(tmp_path, monkeypatch):
    variant = MODULE.Variant("tier", "No-hot-cache")
    data_dir = tmp_path / "variants" / variant.slug / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "warm.db").write_bytes(b"warm-tier")
    manager = MODULE.RetentionManager(tmp_path, "full", disk_floor_gb=10)
    monkeypatch.setattr(manager, "_delete_s3_prefix", lambda _variant: pytest.fail("unexpected S3 cleanup"))
    fields = ["System", "Variant", "Metric"]
    row = {"System": "Plasmod", "Variant": "No-hot-cache", "Metric": 2.0}

    manager.record_variant(variant, "tier", fields, row)

    assert data_dir.exists()
    assert (data_dir.parent / "result_checkpoint.json").exists()
    assert not (data_dir.parent / "METRICS_RETAINED").exists()


def test_metrics_only_refuses_cleanup_when_metric_artifact_is_missing(
        tmp_path, monkeypatch):
    variant = MODULE.Variant("evidence", "No-proof-trace")
    variant_dir = tmp_path / "variants" / variant.slug
    data_dir = variant_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "badger.data").write_bytes(b"database-bytes")
    for name in ("capabilities.json", "measurements.json", "server.log"):
        (variant_dir / name).write_text("{}", encoding="utf-8")
    manager = MODULE.RetentionManager(tmp_path, "metrics-only")
    monkeypatch.setattr(
        manager,
        "_delete_s3_prefix",
        lambda _variant: pytest.fail("cleanup must not start"),
    )
    fields = ["System", "Variant", "Metric"]
    row = {"System": "Plasmod", "Variant": variant.name, "Metric": 2.5}

    with pytest.raises(RuntimeError, match="missing retained metric artifacts"):
        manager.record_variant(variant, "evidence", fields, row)

    assert data_dir.exists()
    assert (variant_dir / "result_checkpoint.json").exists()
    assert not (variant_dir / "METRICS_RETAINED").exists()


def test_retention_reclaimed_bytes_uses_allocated_size_for_sparse_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sparse = data_dir / "sparse.data"
    with sparse.open("wb") as handle:
        handle.seek(1024**3)
        handle.write(b"x")
    allocated = sparse.stat().st_blocks * 512

    assert MODULE.RetentionManager._directory_size(data_dir) == allocated
    assert allocated < sparse.stat().st_size


def test_prepare_variant_removes_incomplete_s3_prefix_before_restart(
        tmp_path, monkeypatch):
    variant = MODULE.Variant("tier", "Warm-only")
    manager = MODULE.RetentionManager(tmp_path, "full")
    removed = []
    monkeypatch.setattr(manager, "_delete_s3_prefix", removed.append)

    manager.prepare_variant(variant)

    assert removed == [variant]


def test_offline_recovery_reset_preserves_wal_and_removes_materialized_state(tmp_path):
    server = MODULE.PlasmodProcess(MODULE.shared_full_variant(), tmp_path, 18080)
    server.data_dir.mkdir(parents=True)
    (server.data_dir / "wal.log").write_text("wal entries", encoding="utf-8")
    for name in (
        "000001.vlog", "000002.sst", "DISCARD", "KEYREGISTRY", "LOCK", "MANIFEST",
        "consistency_checkpoint.json", "derivation.log", "hard_delete_tasks.json",
    ):
        (server.data_dir / name).write_text(name, encoding="utf-8")

    server.offline_reset_materialized_state()

    assert (server.data_dir / "wal.log").read_text(encoding="utf-8") == "wal entries"
    assert sorted(path.name for path in server.data_dir.iterdir()) == ["wal.log"]


def test_explicit_replay_restart_disables_startup_recovery_temporarily(
        tmp_path, monkeypatch):
    variant = MODULE.Variant("wal", "Full Plasmod", {
        "PLASMOD_RECOVERY_REPLAY": "true",
    })
    server = MODULE.PlasmodProcess(variant, tmp_path, 18080)
    calls = []

    monkeypatch.setattr(server, "stop", lambda: calls.append(("stop", None)))
    monkeypatch.setattr(
        server,
        "offline_reset_materialized_state",
        lambda: calls.append(("offline_reset", None)),
    )

    def fake_start(*, fresh):
        calls.append(("start", fresh, variant.env["PLASMOD_RECOVERY_REPLAY"]))

    monkeypatch.setattr(server, "start", fake_start)

    server.restart_for_explicit_replay()

    assert calls == [
        ("stop", None),
        ("offline_reset", None),
        ("start", False, "false"),
    ]
    assert variant.env["PLASMOD_RECOVERY_REPLAY"] == "true"


def test_mark_run_started_replaces_stale_terminal_markers(tmp_path):
    (tmp_path / "FAILED").write_text("old failure", encoding="utf-8")
    (tmp_path / "COMPLETE").write_text("old completion", encoding="utf-8")

    MODULE.mark_run_started(tmp_path)

    assert not (tmp_path / "FAILED").exists()
    assert not (tmp_path / "COMPLETE").exists()
    assert json.loads((tmp_path / "RUNNING").read_text(encoding="utf-8"))["status"] == "running"


def test_metrics_only_disk_guard_stops_before_safety_floor(tmp_path, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        MODULE.shutil,
        "disk_usage",
        lambda _path: usage(100 * 1024**3, 91 * 1024**3, 9 * 1024**3),
    )
    manager = MODULE.RetentionManager(tmp_path, "metrics-only", disk_floor_gb=10)

    with pytest.raises(RuntimeError, match="disk safety floor"):
        manager.ensure_capacity("test variant")


def test_metrics_only_resume_reclaims_incomplete_local_data_before_disk_guard(
        tmp_path, monkeypatch):
    usage = namedtuple("usage", "total used free")
    incomplete = MODULE.Variant("wal", "In-memory WAL")
    incomplete_dir = tmp_path / "variants" / incomplete.slug
    incomplete_data = incomplete_dir / "data"
    incomplete_data.mkdir(parents=True)
    (incomplete_data / "000001.vlog").write_bytes(b"stale")
    (incomplete_dir / "server.log").write_text("interrupted", encoding="utf-8")
    checkpointed = MODULE.Variant("wal", "No-WAL")
    checkpointed_dir = tmp_path / "variants" / checkpointed.slug
    checkpointed_dir.mkdir(parents=True)
    (checkpointed_dir / "result_checkpoint.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        MODULE.shutil,
        "disk_usage",
        lambda _path: usage(
            100 * 1024**3,
            91 * 1024**3,
            (9 if incomplete_data.exists() else 11) * 1024**3,
        ),
    )
    manager = MODULE.RetentionManager(tmp_path, "metrics-only", disk_floor_gb=10)

    with pytest.raises(RuntimeError, match="disk safety floor"):
        manager.ensure_capacity("run start")

    MODULE.reclaim_metrics_only_resume_local_data(manager)

    manager.ensure_capacity("run start")
    assert not incomplete_data.exists()
    assert (incomplete_dir / "server.log").exists()
    assert (checkpointed_dir / "result_checkpoint.json").exists()


def test_recovery_replay_timeout_scales_with_formal_full_workload():
    assert MODULE.recovery_replay_timeout_s(8) == 300
    assert MODULE.recovery_replay_timeout_s(641_979) >= 6_480


def test_recovery_replay_timeout_has_headroom_for_full_projection_replay():
    assert MODULE.recovery_replay_timeout_s(641_979) >= 12_960


def test_recovery_reset_timeout_has_headroom_for_disk_cleanup():
    assert MODULE.recovery_reset_timeout_s(8) == 300
    assert MODULE.recovery_reset_timeout_s(641_979) >= 21_600


def test_plasmod_process_uses_checkpoint_flush_interval_suited_for_disk_benchmark(
        tmp_path, monkeypatch):
    captured_env = {}

    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

        def send_signal(self, _signal):
            pass

        def wait(self, _timeout=None):
            return 0

    def fake_popen(_args, **kwargs):
        captured_env.update(kwargs["env"])
        return FakeProcess()

    def fake_http_json(_base, _method, path, _body=None, timeout=60.0):
        if path == "/healthz":
            return {"status": "ok"}
        if path == "/v1/admin/capabilities":
            return {
                "capabilities": {
                    "wal_mode": "memory",
                    "recovery_replay": True,
                    "hot_cache_size": 2000,
                }
            }
        raise AssertionError(f"unexpected request path {path}")

    monkeypatch.setattr(MODULE.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(MODULE, "http_json", fake_http_json)
    server = MODULE.PlasmodProcess(
        MODULE.Variant("wal", "In-memory WAL", {
            "PLASMOD_WAL_MODE": "memory",
            "PLASMOD_RECOVERY_REPLAY": "true",
        }),
        tmp_path,
        18080,
    )

    server.start(fresh=True)

    assert captured_env["PLASMOD_CONSISTENCY_CHECKPOINT_FLUSH_INTERVAL"] == "1s"


@pytest.mark.parametrize("event_limit", [0, 641_979])
def test_ingest_workload_uses_formal_timeouts_for_large_event_runs(
        tmp_path, monkeypatch, event_limit):
    captured_ingest_timeouts = []
    captured_query_timeouts = []

    class FakeServer:
        base = "http://127.0.0.1:18080"
        variant_dir = tmp_path

        def rss_mb(self):
            return 10.0

    def fake_iter_events(limit):
        assert limit == event_limit
        yield {"identity": {"event_id": "source-1"}, "payload": {"text": "one"}}
        yield {"identity": {"event_id": "source-2"}, "payload": {"text": "two"}}

    def fake_http_json(_base, _method, path, _body=None, timeout=60.0):
        if path == "/v1/ingest/events":
            captured_ingest_timeouts.append(timeout)
            return {
                "status": "accepted",
                "event_id": f"event-{len(captured_ingest_timeouts)}",
                "materialization_latency_ms": 1.0,
                "memory_id": f"mem-{len(captured_ingest_timeouts)}",
            }
        if path == "/v1/query":
            captured_query_timeouts.append(timeout)
            return {"objects": [f"mem-{len(captured_ingest_timeouts)}"]}
        if path == "/v1/admin/runtime/state":
            return {"state": {"events": 2, "objects": 2, "edges": 0, "latest_states": 0}}
        raise AssertionError(f"unexpected request path {path}")

    monkeypatch.setattr(MODULE, "iter_events", fake_iter_events)
    monkeypatch.setattr(MODULE, "http_json", fake_http_json)

    data = MODULE.ingest_workload(
        FakeServer(),
        MODULE.Variant("wal", "In-memory WAL"),
        event_limit,
        0,
    )

    assert data.writes == 2
    assert captured_ingest_timeouts == [300.0, 300.0]
    assert captured_query_timeouts == [300.0, 300.0]


def test_measure_recovery_resets_offline_before_explicit_replay(
        tmp_path, monkeypatch):
    calls = []

    class FakeServer:
        base = "http://127.0.0.1:18080"
        variant_dir = tmp_path
        variant = MODULE.Variant("wal", "Full Plasmod", {
            "PLASMOD_RECOVERY_REPLAY": "true",
        })

        def restart_for_explicit_replay(self):
            calls.append("restart_for_explicit_replay")

    def fake_http_json(_base, _method, path, _body=None, timeout=60.0):
        calls.append(path)
        if path == "/v1/admin/recovery/reset":
            raise AssertionError("materialized recovery reset must be offline")
        if path == "/v1/admin/replay":
            return {"status": "ok", "state": {"objects": 0, "edges": 0, "latest_states": 0}}
        if path == "/v1/admin/runtime/state":
            return {"state": {"objects": 0, "edges": 0, "latest_states": 0, "events": 0}}
        raise AssertionError(f"unexpected request path {path}")

    monkeypatch.setattr(MODULE, "http_json", fake_http_json)
    before = MODULE.RunData(writes=641_979)

    MODULE.measure_recovery(FakeServer(), FakeServer.variant, before)

    assert calls[0] == "restart_for_explicit_replay"
    assert "/v1/admin/replay" in calls


def test_measure_recovery_uses_replay_returned_state(tmp_path, monkeypatch):
    calls = []

    class FakeServer:
        base = "http://127.0.0.1:18080"
        variant_dir = tmp_path

        def restart_for_explicit_replay(self):
            pass

    def fake_http_json(_base, _method, path, _body=None, timeout=60.0):
        calls.append(path)
        if path == "/v1/admin/recovery/reset":
            raise AssertionError("materialized recovery reset must be offline")
        if path == "/v1/admin/replay":
            return {
                "status": "ok",
                "scanned_entries": 3,
                "recovery_time_ms": 1500,
                "replay_throughput_events_s": 2,
                "duplicate_objects": 0,
                "state": {"objects": 4, "edges": 2, "latest_states": 1, "events": 3},
            }
        if path == "/v1/admin/runtime/state":
            raise AssertionError("runtime state should come from replay response")
        raise AssertionError(f"unexpected request path {path}")

    monkeypatch.setattr(MODULE, "http_json", fake_http_json)
    monkeypatch.setattr(MODULE, "query", lambda *args, **kwargs: {})

    variant = MODULE.Variant("wal", "Full Plasmod", {"PLASMOD_RECOVERY_REPLAY": "true"})
    before = MODULE.RunData(
        writes=3,
        event_ids={"e1", "e2", "e3"},
        state_ids={"s1"},
        edge_ids={"edge1", "edge2"},
    )

    row = MODULE.measure_recovery(FakeServer(), variant, before)

    assert "/v1/admin/runtime/state" not in calls
    assert row["Recovered Objects (%)"] == 100
    assert row["Recovered Relations (%)"] == 100
    assert row["Recovered Latest State (%)"] == 100


def test_recovery_resume_uses_variant_checkpoints_without_restarting_servers(
        tmp_path, monkeypatch):
    baseline_row = {
        field: (
            "Plasmod" if field == "System"
            else "Full Plasmod" if field == "Variant"
            else "yes" if field == "Query Available During Recovery"
            else 1
        )
        for field in MODULE.WAL_FIELDS
    }
    baseline = MODULE.SharedFullBaseline(
        module_rows={"wal": baseline_row},
        materialization_type_counts={},
        state_ids=set(),
        artifact_ids=set(),
        contexts={},
        evidence_totals=(0, 0, 0, 0),
        governance_measurement={},
    )
    manager = MODULE.RetentionManager(tmp_path, "full")
    expected = [baseline_row]
    for variant in MODULE.recovery_variants()[1:]:
        row = {
            field: (
                "Plasmod" if field == "System"
                else variant.name if field == "Variant"
                else "yes" if field == "Query Available During Recovery"
                else 2
            )
            for field in MODULE.WAL_FIELDS
        }
        manager.record_variant(variant, "wal", MODULE.WAL_FIELDS, row)
        expected.append(row)

    monkeypatch.setattr(
        MODULE,
        "PlasmodProcess",
        lambda *_args, **_kwargs: pytest.fail("checkpointed variant was restarted"),
    )

    assert MODULE.run_recovery(
        tmp_path, 18080, 8, 3, baseline, manager) == expected


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
    shared_dir = tmp_path / "variants" / MODULE.shared_full_variant().slug
    shared_dir.mkdir(parents=True)
    (shared_dir / "common_metrics.json").write_text(json.dumps({
        "metrics": {field: 7 for field in MODULE.COMMON_FIELDS},
    }), encoding="utf-8")
    for group, variants in variants_by_group.items():
        tables[group] = []
        for variant in variants:
            fields = field_sets[group]
            tables[group].append({
                field: ("yes" if field == "Query Available During Recovery" else 1)
                for field in fields
            } | {"System": "Plasmod", "Variant": variant.name})
            if not MODULE.is_group_full_variant(variant):
                variant_dir = tmp_path / "variants" / variant.slug
                variant_dir.mkdir(parents=True)
                (variant_dir / "common_metrics.json").write_text(json.dumps({
                    "metrics": {field: 1 for field in MODULE.COMMON_FIELDS},
                }), encoding="utf-8")

    rows = MODULE.build_master_table(tmp_path, tables)
    assert len(rows) == 34
    assert all(all(row[field] not in (None, "") for field in MODULE.MASTER_FIELDS) for row in rows)
    full_rows = [row for row in rows if row["Comparison Label"] == "Full"]
    assert len(full_rows) == 5
    assert {row["Common | Event Count"] for row in full_rows} == {7}
    wal_row = next(row for row in rows if row["Module"] == "wal")
    assert wal_row["WAL | Event Log Size"] == 1
    assert wal_row["EVIDENCE | Query p95 (ms)"] == MODULE.NOT_APPLICABLE


def test_measure_tier_scales_query_timeout_for_large_replayed_dataset(monkeypatch):
    captured_timeouts = []
    captured_modes = []

    class FakeServer:
        base = "http://127.0.0.1:18080"

        def rss_mb(self):
            return 123.0

    def fake_http_json(_base, _method, path, _body=None, timeout=60.0):
        if path == "/v1/admin/tier/archive":
            return {"archive_latency_ms": 12.0}
        raise AssertionError(f"unexpected request path {path}")

    def fake_query(
        _base,
        _text,
        _vector,
        target_ids=None,
        response_mode="structured_evidence",
        requester="",
        include_cold=False,
        workspace="plasmod-ablation",
        session="",
        timeout=60.0,
    ):
        captured_timeouts.append(timeout)
        captured_modes.append((response_mode, include_cold))
        if response_mode == "objects_only":
            return {"objects": list(target_ids or [])}, 1.0
        return {
            "retrieval": {
                "hot_candidate_count": 1,
                "warm_candidate_count": 2,
                "cold_candidate_count": 3,
            },
            "diagnostics": {"promotion_latency_ms": 4.0},
        }, 1.0

    monkeypatch.setattr(MODULE, "http_json", fake_http_json)
    monkeypatch.setattr(MODULE, "query", fake_query)
    data = MODULE.RunData(
        writes=641_979,
        memory_ids={"mem-1"},
        latest_query_samples=[(
            "agent state",
            MODULE.hash_vector("agent state"),
            "mem-1",
            "agent-1",
            "plasmod-ablation",
            "session-1",
        )],
    )

    MODULE.measure_tier(FakeServer(), MODULE.shared_full_variant(), data)

    assert MODULE.formal_query_timeout_s(8) == 60.0
    assert MODULE.formal_query_timeout_s(data.writes) == 300.0
    assert captured_modes == [("objects_only", True), ("objects_only", False)]
    assert captured_timeouts == [300.0, 300.0]


def test_plasmod_start_allows_large_data_restart_to_become_healthy(tmp_path, monkeypatch):
    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

    now = 0.0

    def fake_time():
        return now

    def fake_sleep(seconds):
        nonlocal now
        now += seconds

    def fake_http_json(_base, _method, path, _body=None, timeout=60.0):
        if path == "/healthz":
            if now < 60.0:
                raise RuntimeError("not healthy yet")
            return {"status": "ok"}
        if path == "/v1/admin/capabilities":
            return {"capabilities": {
                "wal_mode": "file",
                "recovery_replay": True,
                "recovery_projection": "full",
                "materialization_profile": "full",
                "evidence_profile": "full",
                "governance_profile": "full",
                "tier_profile": "full",
                "hot_cache_size": 2000,
            }}
        raise AssertionError(f"unexpected request path {path}")

    monkeypatch.setattr(MODULE.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(MODULE.time, "time", fake_time)
    monkeypatch.setattr(MODULE.time, "sleep", fake_sleep)
    monkeypatch.setattr(MODULE, "http_json", fake_http_json)

    process = MODULE.PlasmodProcess(MODULE.shared_full_variant(), tmp_path, 18080)

    process.start(fresh=True)

    assert now >= 60.0


def test_plasmod_start_keeps_checkpoint_buffering_enabled(tmp_path, monkeypatch):
    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

    captured_env = {}

    def fake_popen(*args, **kwargs):
        captured_env.update(kwargs["env"])
        return FakeProcess()

    def fake_http_json(_base, _method, path, _body=None, timeout=60.0):
        if path == "/healthz":
            return {"status": "ok"}
        if path == "/v1/admin/capabilities":
            return {"capabilities": {
                "wal_mode": "file",
                "recovery_replay": True,
                "recovery_projection": "full",
                "materialization_profile": "full",
                "evidence_profile": "full",
                "governance_profile": "full",
                "tier_profile": "full",
                "hot_cache_size": 2000,
            }}
        raise AssertionError(f"unexpected request path {path}")

    monkeypatch.setattr(MODULE.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(MODULE, "http_json", fake_http_json)

    process = MODULE.PlasmodProcess(MODULE.shared_full_variant(), tmp_path, 18080)

    process.start(fresh=True)

    assert captured_env["PLASMOD_CONSISTENCY_CHECKPOINT_FLUSH_INTERVAL"] != "0"
    assert captured_env["PLASMOD_CONSISTENCY_CHECKPOINT_FLUSH_INTERVAL"] != "off"


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
