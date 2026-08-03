# INTAKE — rerun dual-sink guard (`rrd` + `serve` silently loses 3,314× the data)

- **Date:** 2026-07-21
- **From:** Tools & DevEnv agent (Monday run, branch `agent/tools-devenv-20260721`)
- **Verdict (orchestrator):** _(unfilled)_

## What

A 12-line pre-flight check for `tanitad/replay/rr_log.py::RerunLogger.__init__`
that **refuses** `rrd=... AND serve=...` in the same recording, with a message
that names the measurement and the two workarounds.

- `rr_dual_sink_guard.py` — `check_sinks(rrd, serve, allow_stub_rrd=False)`
- `tests/test_rr_dual_sink_guard.py` — 5 falsifiers, standalone-runnable

## Why

Mission P1 has carried "dual-sink (serve+rrd) empty file" as an open TanitResim
bug with no measurement. This run measured it.

**Root cause (rerun-sdk 0.34.1, confirmed against the SDK's own docstring):**
`rr.save()` installs the file sink; `rr.serve_grpc()` — called immediately
after it in `RerunLogger.__init__` — *replaces* the sink set
("This _replaces_ existing sinks"). Every per-step record then goes only to the
gRPC stream. The `.rrd` retains the blueprint and the static BEV axes.

**Measured** (200 windows × 3 arms × 256×256 RGB, RTX 4060 box, CPU path,
`../rrd_bench/results.json`, rerun 0.34.1 / py3.13.5):

| case | bytes | B/window | verdict |
|---|---:|---:|---|
| `rrd` only, jpeg85 | 10,593,179 | 52,966 | correct |
| `rrd` only, raw | 40,107,580 | 200,538 | correct (jpeg85 = **3.79× smaller**, 17 % slower) |
| **`rrd` + `serve`** | **3,196** | **16** | **stub — 3,314× data loss** |

The file is **non-zero**, which is why the bug survived: every "did the artifact
get written?" check passes. It is a legend with no data.

## Why a guard and not a fix

The obvious repair — `rr.set_sinks(rr.FileSink(p), rr.GrpcSink(url))` after
`serve_grpc()` — was prototyped and **deadlocks**: the process produced no
output and was killed at 120 s, because the GrpcSink connects back to the
in-process server hosted by the same thread. A real tee needs two
`RecordingStream`s and an explicit `recording=` on every `rr.log` call in
`rr_log.py` — a genuine refactor, now backlogged (Tools&DevEnv P0#3) with the
falsifier pre-registered: *dual-sink `.rrd` must land within 5 % of the
single-sink `.rrd` for identical input*.

Silently shipping a stub is the worse failure: a researcher opens the `.rrd`
next week, sees an empty timeline, and debugs the model instead of the sink.

## Evidence / tests run

- `pytest tests` in this package: **5 passed, 0.01 s** (standalone).
- Benchmark harness + raw numbers: `../rrd_bench/bench_rrd.py`, `results.json`.
- Repro of the loss is in the harness itself (case `dual_sink_rrd_plus_serve`).

## Proposed target location

`stack/tanitad/replay/rr_log.py` — two changes:

1. add the `allow_stub_rrd: bool = False` parameter to `RerunLogger.__init__`;
2. call `check_sinks(rrd, serve, allow_stub_rrd)` **before `rr.init`** (failing
   afterwards would leave the stub file on disk — the exact artefact we are
   trying to stop shipping). Inline the function or import it; it has no deps.

`stack/scripts/replay_app.py` should surface `--allow-stub-rrd` so the opt-out
is reachable from the CLI.

## Risk / rollback

Low, but **behaviour-changing**: any caller passing both `--rrd` and `--serve`
today starts failing. That is the intent — those callers are currently getting
a stub — but it will surface in anyone's muscle memory. Rollback is deleting
the single `check_sinks(...)` line.

Not covered: the two-`RecordingStream` tee (backlogged), and whether the
0.34.1 Viewer-MCP server changes this picture (untested; the MCP attaches to a
viewer, not to the sink stack).
