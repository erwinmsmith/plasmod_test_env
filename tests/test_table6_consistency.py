import unittest
import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import layer2_dynamic_event_benchmark as benchmark


def healthy_status(mode: str, **overrides):
    status = {
        "status": "ok",
        "mode": mode,
        "supported_modes": list(benchmark.PLASMOD_MODES.values()),
        "data_path_active": True,
        "queue_depth": 0,
        "latest_lsn": 1,
        "visible_watermark": 1,
        "pending": 0,
        "retrying": 0,
        "failed": 0,
        "sla_breaches": 0,
        "last_error": "",
    }
    status.update(overrides)
    return status


class FakeHTTP:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def ingest_result(mode: str, visibility: str, *, sla_ms=None, visible_lag_ms=10.0):
    ack = {
        "consistency_mode": benchmark.PLASMOD_MODES[mode],
        "visibility_status": visibility,
        "lsn": 1,
    }
    if sla_ms is not None:
        ack["freshness_sla_ms"] = sla_ms
    return benchmark.IngestResult(
        system="Plasmod",
        event_type="memory",
        event_id="event",
        object_id="object",
        expected_ids={"mem_event"},
        write_start_ms=90.0,
        write_ack_ms=100.0,
        write_latency_ms=10.0,
        ok=True,
        ack=ack,
        first_visible_ms=100.0 + visible_lag_ms,
    )


class Table6ConsistencyContractTest(unittest.TestCase):
    def test_run_id_rejects_mismatched_resume_contract_without_overwriting_metadata(self):
        with TemporaryDirectory() as tmp:
            common = [
                "run", "--tables", "6", "--systems", "plasmod",
                "--output-dir", tmp, "--run-id", "immutable-run",
                "--events-per-rate", "10", "--query-limit", "2",
            ]
            with patch.object(benchmark, "run_table6", return_value=[]):
                benchmark.run_tables(benchmark.parse_args(common + ["--fixed-write-rate", "10"]))

            metadata_path = Path(tmp) / "immutable-run" / "run_metadata.json"
            before = metadata_path.read_text()
            with patch.object(benchmark, "run_table6", return_value=[]):
                with self.assertRaisesRegex(RuntimeError, "run contract mismatch"):
                    benchmark.run_tables(benchmark.parse_args(common + ["--fixed-write-rate", "100"]))
            self.assertEqual(metadata_path.read_text(), before)

    def test_run_failure_writes_diagnostic_artifact(self):
        with TemporaryDirectory() as tmp:
            args = benchmark.parse_args([
                "run", "--tables", "6", "--systems", "plasmod",
                "--output-dir", tmp, "--run-id", "failed-run",
            ])
            with patch.object(benchmark, "run_table6", side_effect=RuntimeError("injected failure")):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    benchmark.run_tables(args)

            failure_path = Path(tmp) / "failed-run" / "run_failure.json"
            failure = json.loads(failure_path.read_text())
            self.assertEqual(failure["error_type"], "RuntimeError")
            self.assertIn("injected failure", failure["error"])
            self.assertIn("traceback", failure)

    def test_hash_embeddings_are_prewarmed_and_reused_from_disk_cache(self):
        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "embeddings.sqlite3"
            embedder = benchmark.make_embedding_provider(
                "hash",
                benchmark.DEFAULT_EMBEDDER_MODEL,
                benchmark.DEFAULT_EMBEDDER_VOCAB,
                cache_path,
                2,
            )

            first = embedder.prewarm_texts(iter(["alpha", "beta", "alpha"]))
            second = embedder.prewarm_texts(iter(["alpha", "beta"]))

            self.assertIsInstance(embedder, benchmark.CachedEmbedder)
            self.assertEqual(first["new_embeddings"], 2)
            self.assertEqual(second["new_embeddings"], 0)
            self.assertEqual(second["cached_embeddings"], 2)
            self.assertEqual(embedder.embed_one("alpha"), embedder.embed_one("alpha"))

    def test_table6_launcher_disables_periodic_retrieval_flush_by_default(self):
        launcher = Path(__file__).parents[1] / "scripts" / "start_plasmod_table6.sh"

        self.assertIn(
            "PLASMOD_FLUSH_INTERVAL='${PLASMOD_FLUSH_INTERVAL:-0}'",
            launcher.read_text(),
        )

    def test_table6_launcher_skips_unneeded_vector_projection_by_default(self):
        launcher = Path(__file__).parents[1] / "scripts" / "start_plasmod_table6.sh"

        self.assertIn(
            "PLASMOD_SKIP_VECTOR_INDEX='${PLASMOD_SKIP_VECTOR_INDEX:-1}'",
            launcher.read_text(),
        )

    def test_table6_launcher_reserves_projection_capacity_for_fixed_rate_run(self):
        launcher = Path(__file__).parents[1] / "scripts" / "start_plasmod_table6.sh"

        self.assertIn(
            "PLASMOD_CONSISTENCY_WORKERS='${PLASMOD_CONSISTENCY_WORKERS:-10}'",
            launcher.read_text(),
        )

    def test_table6_full_launcher_pins_validated_experiment_contract(self):
        launcher = Path(__file__).parents[1] / "scripts" / "run_table6_full.sh"
        text = launcher.read_text()

        for expected in [
            ".venv/bin/python",
            "--tables 6",
            "--systems plasmod milvus",
            "--events-per-rate 0",
            "--fixed-write-rate 100",
            "--query-qps 5",
            "--query-limit 5000",
            "--bounded-sla-ms 1000",
            "--embedding-cache",
            "--embedding-batch-size 512",
            "--milvus-visibility-policy deferred",
            "--milvus-index-type FLAT",
            "--milvus-payload-json-bytes 0",
            "--reset-between-runs",
        ]:
            self.assertIn(expected, text)

    def test_milvus_collection_setup_uses_extended_load_timeout(self):
        class FakeClient:
            def __init__(self):
                self.load_timeouts = []

            def has_collection(self, _name):
                return True

            def load_collection(self, _name, timeout=None):
                self.load_timeouts.append(timeout)

        adapter = benchmark.MilvusAdapter.__new__(benchmark.MilvusAdapter)
        adapter.client = FakeClient()
        adapter.collection_name = "existing"
        adapter.timeout = 30.0

        adapter._ensure_collection(drop=False)

        self.assertEqual(adapter.client.load_timeouts, [120.0])

    def test_milvus_collection_setup_recovers_when_create_times_out_after_creation(self):
        class FakeClient:
            def __init__(self):
                self.created = False
                self.load_timeouts = []

            def has_collection(self, _name):
                return self.created

            def create_collection(self, _name, **_kwargs):
                self.created = True
                raise TimeoutError("created but initial load timed out")

            def load_collection(self, _name, timeout=None):
                self.load_timeouts.append(timeout)

        adapter = benchmark.MilvusAdapter.__new__(benchmark.MilvusAdapter)
        adapter.client = FakeClient()
        adapter.collection_name = "created-after-timeout"
        adapter.timeout = 30.0
        adapter.index_type = "FLAT"

        adapter._ensure_collection(drop=False)

        self.assertEqual(adapter.client.load_timeouts, [120.0])

    def test_milvus_reset_does_not_hide_collection_drop_failure(self):
        class FakeClient:
            def has_collection(self, _name):
                return True

            def drop_collection(self, _name, timeout=None):
                raise RuntimeError("drop failed")

            def load_collection(self, _name, timeout=None):
                raise AssertionError("stale collection must not be loaded")

        adapter = benchmark.MilvusAdapter.__new__(benchmark.MilvusAdapter)
        adapter.client = FakeClient()
        adapter.collection_name = "stale"
        adapter.timeout = 30.0

        with self.assertRaisesRegex(RuntimeError, "drop failed"):
            adapter._ensure_collection(drop=True)

    def test_milvus_ingest_inserts_each_event_once(self):
        class FakeEmbedder:
            def embed_one(self, _text):
                return [0.0] * benchmark.EMBEDDING_DIM

        class FakeClient:
            def __init__(self):
                self.inserts = []

            def insert(self, collection_name, rows, timeout=None):
                self.inserts.append((collection_name, rows, timeout))

        adapter = benchmark.MilvusAdapter.__new__(benchmark.MilvusAdapter)
        adapter.client = FakeClient()
        adapter.collection_name = "single-insert"
        adapter.timeout = 30.0
        adapter.embedder = FakeEmbedder()
        adapter.visibility_policy = "deferred"
        adapter.payload_json_bytes = 0
        adapter.mu = __import__("threading").Lock()

        result = adapter.ingest({
            "identity": {"event_id": "event"},
            "object": {"object_id": "object"},
            "payload": {"text": "hello"},
        })

        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(adapter.client.inserts), 1)

    def test_ingest_completion_callback_ignores_cancelled_future(self):
        observed = []
        future = Future()
        self.assertTrue(future.cancel())
        benchmark.notify_ingest_completion(
            future,
            {"identity": {"event_id": "event"}},
            lambda _event, result: observed.append(result),
        )

        self.assertEqual(observed, [])

    def test_ingest_completion_callback_propagates_unexpected_errors(self):
        future = Future()
        future.set_result(ingest_result("bounded", "pending", sla_ms=1000))

        with self.assertRaisesRegex(RuntimeError, "unexpected callback failure"):
            benchmark.notify_ingest_completion(
                future,
                {"identity": {"event_id": "event"}},
                lambda _event, _result: (_ for _ in ()).throw(
                    RuntimeError("unexpected callback failure")
                ),
            )

    def test_ingest_with_rate_propagates_async_completion_callback_errors(self):
        class SuccessfulAdapter:
            name = "Successful"
            timeout = 0.1

            def ingest(self, _event):
                return ingest_result("bounded", "pending", sla_ms=1000)

        with self.assertRaisesRegex(RuntimeError, "completion callback failed"):
            benchmark.ingest_with_rate(
                SuccessfulAdapter(),
                [{"identity": {"event_id": "event"}}],
                rate_eps=0,
                workers=1,
                on_complete=lambda _event, _result: (_ for _ in ()).throw(
                    RuntimeError("unexpected callback failure")
                ),
                total_events=1,
            )

    def test_freshness_trial_cleans_up_executors_after_ingest_failure(self):
        shutdown_calls = []

        class RecordingExecutor(ThreadPoolExecutor):
            def shutdown(self, wait=True, *, cancel_futures=False):
                shutdown_calls.append((wait, cancel_futures))
                return super().shutdown(wait=wait, cancel_futures=cancel_futures)

        class FailingAdapter:
            name = "Failing"
            timeout = 0.1

            def ingest(self, _event):
                raise RuntimeError("primary write failure")

            def query(self, _query):
                raise AssertionError("query should not run after ingest failure")

        with patch.object(benchmark, "ThreadPoolExecutor", RecordingExecutor):
            with self.assertRaisesRegex(RuntimeError, "primary write failure"):
                benchmark.run_freshness_trial(
                    FailingAdapter(),
                    [{"identity": {"event_id": "event"}}],
                    run_id="cleanup",
                    write_rate=0,
                    query_qps=0,
                    workers=1,
                    query_limit=1,
                    visible_timeout_ms=10,
                    visible_poll_ms=1,
                    total_events=1,
                )

        self.assertIn((False, True), shutdown_calls)
        self.assertIn((True, True), shutdown_calls)

    def test_mode_switch_rejects_inactive_data_path(self):
        adapter = benchmark.PlasmodAdapter("http://plasmod.test")
        adapter.http = FakeHTTP(
            [healthy_status("strict_visible", data_path_active=False)]
        )

        with self.assertRaisesRegex(RuntimeError, "data path"):
            adapter.set_visibility_mode("strict")

    def test_drained_status_waits_for_final_projection_slot_release(self):
        adapter = benchmark.PlasmodAdapter("http://plasmod.test")
        adapter.active_mode = "bounded_staleness"
        adapter.http = FakeHTTP(
            [
                healthy_status("bounded_staleness", queue_depth=1),
                healthy_status("bounded_staleness", queue_depth=0),
            ]
        )

        status = adapter.consistency_status(require_drained=True)

        self.assertEqual(status["queue_depth"], 0)
        self.assertEqual(len(adapter.http.calls), 2)

    def test_ingest_rejects_ack_mode_mismatch(self):
        adapter = benchmark.PlasmodAdapter("http://plasmod.test")
        adapter.active_mode = "strict_visible"
        adapter.http = FakeHTTP(
            [
                {
                    "consistency_mode": "eventual_visibility",
                    "visibility_status": "visible",
                    "lsn": 1,
                }
            ]
        )
        event = {
            "identity": {"event_id": "event"},
            "object": {"object_id": "object"},
            "access": {"consistency": "strict_visible"},
            "payload": {"text": "payload"},
        }

        result = adapter.ingest(event)

        self.assertFalse(result.ok)
        self.assertIn("consistency_mode", result.error)

    def test_reset_preserves_mode_and_clears_projection(self):
        adapter = benchmark.PlasmodAdapter("http://plasmod.test")
        adapter.active_mode = "bounded_staleness"
        adapter.http = FakeHTTP(
            [
                {"status": "ok", "consistency_projection": "reset"},
                healthy_status(
                    "bounded_staleness", latest_lsn=0, visible_watermark=0
                ),
            ]
        )

        adapter.reset()

        self.assertEqual(adapter.last_consistency_status["mode"], "bounded_staleness")
        self.assertEqual(adapter.last_consistency_status["pending"], 0)

    def test_strict_mode_rejects_stale_query(self):
        adapter = benchmark.PlasmodAdapter("http://plasmod.test")
        adapter.active_mode = "strict_visible"
        adapter.http = FakeHTTP([healthy_status("strict_visible")])
        query = benchmark.QueryResult(
            "Plasmod", "visibility", 1.0, True, False, True
        )

        with self.assertRaisesRegex(RuntimeError, "strict"):
            benchmark.validate_table6_mode_semantics(
                adapter, "strict", [ingest_result("strict", "visible")], [query], 1000
            )

    def test_bounded_mode_rejects_sla_violation(self):
        adapter = benchmark.PlasmodAdapter("http://plasmod.test")
        adapter.active_mode = "bounded_staleness"
        adapter.http = FakeHTTP([healthy_status("bounded_staleness")])

        with self.assertRaisesRegex(RuntimeError, "SLA"):
            benchmark.validate_table6_mode_semantics(
                adapter,
                "bounded",
                [
                    ingest_result(
                        "bounded", "pending", sla_ms=1000, visible_lag_ms=1001
                    )
                ],
                [],
                1000,
            )

    def test_eventual_mode_accepts_unbounded_visibility_lag(self):
        adapter = benchmark.PlasmodAdapter("http://plasmod.test")
        adapter.active_mode = "eventual_visibility"
        adapter.http = FakeHTTP([healthy_status("eventual_visibility")])

        benchmark.validate_table6_mode_semantics(
            adapter,
            "eventual",
            [ingest_result("eventual", "pending", visible_lag_ms=5000)],
            [],
            1000,
        )


if __name__ == "__main__":
    unittest.main()
