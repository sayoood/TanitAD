"""Tests for gpu_tripwire. The parity probes need a real CUDA device, so the
substantive ones skip on a CPU-only box — but the *contract* (never raises,
reports honestly, exit-code policy) is asserted everywhere, because the whole
point of this tool is that it must not silently no-op."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TOOLS))
# the parity probes import the real model, which lives in the sibling stack/
sys.path.insert(0, str(_TOOLS.parent / "stack"))
import gpu_tripwire  # noqa: E402

torch = pytest.importorskip("torch")
HAS_CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")


def test_run_never_raises_and_reports_availability():
    rep = gpu_tripwire.run()
    assert isinstance(rep.available, bool)
    assert rep.available == HAS_CUDA
    if not rep.available:
        assert rep.error, "a CPU-only run must say WHY it produced no probes"


def test_no_cuda_is_a_loud_skip_by_default_and_fails_under_require(monkeypatch):
    """The policy that keeps this honest: a dev laptop may skip, CI may not."""
    monkeypatch.setattr(gpu_tripwire, "run",
                        lambda **_: gpu_tripwire.GpuReport(
                            available=False, error="no CUDA device visible"))
    assert gpu_tripwire.main([]) == 0
    assert gpu_tripwire.main(["--require-cuda"]) == 1


def test_format_report_is_ascii_only():
    """The Windows cp1252 console lesson: any non-ASCII marker crashes the tool
    on the dev box before it can report anything."""
    rep = gpu_tripwire.GpuReport(
        available=True, device_name="X", torch_version="2.11",
        probes=[gpu_tripwire.Probe("P1", True, "fine", 1e-7),
                gpu_tripwire.Probe("P2", False, "broken")],
        encode_batch1_ms=1.23, wall_s=1.0)
    text = gpu_tripwire.format_report(rep)
    text.encode("ascii")            # raises if a glyph slipped in
    assert "[PASS] P1" in text and "[FAIL] P2" in text


def test_report_to_dict_is_json_serializable():
    rep = gpu_tripwire.GpuReport(available=False, error="none here")
    json.dumps(rep.to_dict())


@needs_cuda
def test_all_parity_probes_pass_on_this_card():
    rep = gpu_tripwire.run()
    assert rep.error is None, rep.error
    names = {p.name for p in rep.probes}
    assert names == {"P1_encode_parity", "P2_imagine_parity",
                     "P3_i2_on_device", "P4_backward_finite"}
    assert not rep.failures, gpu_tripwire.format_report(rep)


@needs_cuda
def test_an_impossible_tolerance_fails_the_parity_probes():
    """Falsifier: the probes must be capable of failing. With tol=0 the fp32
    CPU-vs-CUDA deviation (~1e-6) has to trip P1/P2 — if this passes, the
    comparison is not actually comparing anything."""
    rep = gpu_tripwire.run(tol=0.0)
    failed = {p.name for p in rep.failures}
    assert {"P1_encode_parity", "P2_imagine_parity"} <= failed


@needs_cuda
def test_latency_proxy_is_measured():
    rep = gpu_tripwire.run()
    assert rep.encode_batch1_ms is not None and rep.encode_batch1_ms > 0
