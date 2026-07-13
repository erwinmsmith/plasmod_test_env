import unittest
from concurrent.futures import Future
from pathlib import Path

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
    def test_table6_launcher_disables_periodic_retrieval_flush_by_default(self):
        launcher = Path(__file__).parents[1] / "scripts" / "start_plasmod_table6.sh"

        self.assertIn(
            "PLASMOD_FLUSH_INTERVAL='${PLASMOD_FLUSH_INTERVAL:-0}'",
            launcher.read_text(),
        )

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

    def test_mode_switch_rejects_inactive_data_path(self):
        adapter = benchmark.PlasmodAdapter("http://plasmod.test")
        adapter.http = FakeHTTP(
            [healthy_status("strict_visible", data_path_active=False)]
        )

        with self.assertRaisesRegex(RuntimeError, "data path"):
            adapter.set_visibility_mode("strict")

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
