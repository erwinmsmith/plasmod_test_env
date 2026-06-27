# Layer 2 Dynamic Event Stream Data

This directory is the data drop zone for Layer 2 experiments: Dynamic Event Stream and State Visibility.

Keep all prepared Layer 2 datasets in `plasmod_test_env`, not in the `Plasmod` core repository. The core repository should only contain database/runtime implementation code. Synthetic streams, replay traces, query workloads, and ground-truth files belong here.

## Layout

| Path | Purpose |
|---|---|
| `synthetic/` | Synthetic Agent Event Stream files for controllable write-load and visibility experiments. |
| `replay_traces/research_agent/` | Replayable research-agent traces covering memory, artifacts, provenance, and evidence relations. |
| `replay_traces/tool_use_agent/` | Replayable tool-use traces covering tool results, state updates, and latest-state queries. |
| `replay_traces/multi_agent_collaboration/` | Replayable multi-agent traces covering shared/private memory, scopes, artifacts, and conflict relations. |
| `queries/` | Query workload files for latest state, memory, artifact lookup, provenance, relation, and scope-aware retrieval. |
| `ground_truth/` | Expected visibility timestamps, versions, relations, state snapshots, and correctness labels. |
| `manifests/` | Run manifests describing dataset versions, generation parameters, scale, and file checksums. |

## Suggested File Names

- Synthetic events: `synthetic/events_<rate>eps_<sessions>s_<payload>.jsonl`
- Replay traces: `replay_traces/<trace_type>/session_<id>.jsonl`
- Queries: `queries/queries_<dataset_or_trace>.jsonl`
- Ground truth: `ground_truth/ground_truth_<dataset_or_trace>.jsonl`
- Manifest: `manifests/<dataset_or_trace>.json`

## Expected Records

Event records should include the fields needed by the Layer 2 plan, including `event_id`, `session_id`, `agent_id`, `timestamp_ms`, `event_type`, `object_id`, `payload`, `payload_size`, `has_embedding`, `embedding_dim`, `parents`, `visibility`, `trigger_materialization`, and `relation_type` where applicable.

Query records should include `query_id`, `timestamp_ms`, `query_type`, `session_id`, `agent_id`, `scope`, `target_object_id`, `expected_object_id`, `expected_version`, `expected_relation_ids`, and `visibility_requirement` where applicable.

Ground-truth records should include the timing fields required for metric computation: `t_event_created`, `t_write_start`, `t_write_ack`, `t_materialized`, `t_first_visible`, and `t_query`.

Large data files are intentionally git-ignored. Commit only documentation, manifests when they are small, and scripts needed to reproduce or consume the data.
