"""Fail-loud guard for the rerun dual-sink data loss (measured 2026-07-21).

The bug
-------
``RerunLogger(rrd=..., serve=...)`` accepts both sinks and appears to work.
It does not. In rerun-sdk 0.34.1, ``rr.save()`` installs the file sink and
``rr.serve_grpc()`` **replaces** it (its own docstring: "This _replaces_
existing sinks"). Every per-step record after that point goes only to the gRPC
stream; the ``.rrd`` keeps just the blueprint and the static BEV axes.

Measured on 200 windows x 3 arms x 256x256 RGB
(``Tools&DevEnv/Implementation/rrd_bench/results.json``):

    rrd only, jpeg85           10,593,179 B   (52,966 B/window)
    rrd + serve (dual sink)         3,196 B   (16 B/window)   <-- 3,314x loss

The file is **not zero bytes**, which is why the bug survived: every "is the
artifact there?" check passes. It is a stub with a legend and no data.

Why this is a guard and not a fix
---------------------------------
The obvious repair — re-attach both sinks with
``rr.set_sinks(rr.FileSink(p), rr.GrpcSink(url))`` after ``serve_grpc()`` —
was prototyped and **deadlocks**: the process hangs indefinitely (killed at
120 s with no output) because the GrpcSink connects back to the in-process
server the same thread is hosting. A working tee needs two RecordingStreams
and a logger that takes an explicit ``recording=`` on every ``rr.log`` call —
a real refactor of ``rr_log.py``, backlogged with a measured falsifier.

Until then the correct behaviour is to **refuse the combination loudly**
rather than hand back a stub. ``allow_stub_rrd=True`` is the explicit opt-out
for anyone who genuinely wants only the live viewer.
"""
from __future__ import annotations

DUAL_SINK_MSG = (
    "rerun 0.34.1 cannot serve a live viewer AND write a usable .rrd from one "
    "recording: serve_grpc() REPLACES the file sink, so the .rrd would contain "
    "only the blueprint (measured 3,196 B vs 10,593,179 B for the same 200 "
    "windows — a 3,314x silent data loss, and the file is non-zero so it looks "
    "written).\n"
    "  Pick one sink:  --rrd PATH   (artifact, scrubbable later)\n"
    "                  --serve PORT (live viewer; use the viewer's own Save "
    "button for an artifact)\n"
    "  Or run the replay twice, once per sink.\n"
    "  Override with allow_stub_rrd=True only if you accept a stub .rrd."
)


def check_sinks(rrd, serve, allow_stub_rrd: bool = False) -> None:
    """Raise before any logging happens if both sinks were requested.

    Called at the top of ``RerunLogger.__init__``, i.e. before ``rr.init`` —
    failing after the recording is open would leave the stub file on disk,
    which is the exact artefact we are trying to stop shipping."""
    if rrd is not None and serve is not None and not allow_stub_rrd:
        raise ValueError(DUAL_SINK_MSG)
