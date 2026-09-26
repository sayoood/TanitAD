"""The GPU-gap memory limit must be REACHABLE FROM THE CALLER, and raising it must not weaken the
standing invariant.

⛔ THE DEFECT THIS PINS (MEASURED 2026-09-20/26). ``limit_mib`` was threaded correctly through
``gap_verdict``, ``device_policy`` and ``GpuGapLauncher`` — and passed by NOBODY. Both construction
sites (``contract.py``, ``navsim/benchmark.py``) omitted it and the CLI had no flag, so the gate always
ran at the hardcoded 1024 MiB. On this desktop box the idle WDDM compositor holds 1,175-5,812 MiB with
no training process alive at all, so the gate could never open: W7 sampled 0 gaps in 12 samples and
every model arm silently fell back to CPU, where a 416x1024 ResNet rollout is ~100x too slow to use.
That is the ``CLAUDE.md`` class "built, tested, and unreachable from its caller" — sixth instance.

⭐ Expectations are written as LITERALS, never as expressions over the code under test, and each
positive assertion is paired with a control that must read a known value.
"""
from __future__ import annotations

import pytest

from taniteval.bench import gpu_gap as G
from taniteval.bench.cli import build_parser


def _probe(pids, used):
    return lambda: (list(pids), used)


# ---------------------------------------------------------------- the resolver

def test_default_is_1024_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv(G.ENV_MEM_LIMIT, raising=False)
    assert G.resolve_mem_limit_mib() == 1024.0          # literal, not DEFAULT_MEM_LIMIT_MIB


def test_env_is_read(monkeypatch):
    monkeypatch.setenv(G.ENV_MEM_LIMIT, "4300")
    assert G.resolve_mem_limit_mib() == 4300.0


def test_explicit_argument_beats_env(monkeypatch):
    monkeypatch.setenv(G.ENV_MEM_LIMIT, "9999")
    assert G.resolve_mem_limit_mib(4300) == 4300.0


@pytest.mark.parametrize("raw", ["", "  ", "abc", "0", "-1", "inf", "nan_but_not", "1e400"])
def test_unusable_env_falls_back_and_never_disables_the_gate(monkeypatch, raw):
    """⛔ A gate reading 0 would refuse forever; one reading inf would never refuse. Both are worse
    than the default, so a malformed value is IGNORED rather than honoured."""
    monkeypatch.setenv(G.ENV_MEM_LIMIT, raw)
    assert G.resolve_mem_limit_mib() == 1024.0


# ------------------------------------------------- reachability from the CLI

def test_cli_exposes_the_flag_and_carries_the_value():
    a = build_parser().parse_args([
        "navsim_v2", "--ckpt", "none", "--split", "warmup_two_stage", "--gpu-mem-limit-mib", "4300"])
    assert a.gpu_mem_limit_mib == 4300.0


def test_cli_default_is_none_so_the_resolver_decides():
    """None (not 1024) so that $TANITAD_GPU_MEM_LIMIT_MIB still applies when the flag is absent —
    a CLI default of 1024 would silently override the env and re-create the original bug."""
    a = build_parser().parse_args(["navsim_v2", "--ckpt", "none", "--split", "warmup_two_stage"])
    assert a.gpu_mem_limit_mib is None


# --------------------------------------- the behaviour change, and its control

def test_the_regression_arm_a_desktop_box_is_refused_at_the_old_default(monkeypatch):
    """MUTATION ARM: this is the old, broken behaviour. 3,000 MiB of desktop compositor against the
    hardcoded 1,024 limit => REFUSE, which is why no model arm could ever use the GPU here."""
    monkeypatch.delenv(G.ENV_MEM_LIMIT, raising=False)
    pol = G.device_policy("auto", probe=_probe([], 3000), cuda_available=lambda: True)
    assert pol["decision"] == "REFUSE"
    assert pol["record"]["limit_mib"] == 1024.0


def test_threading_the_limit_opens_the_gate(monkeypatch):
    """The FIX: the same box, the PI's 4,300 MiB gate => CUDA. Only the caller-supplied limit differs
    from the arm above, so this isolates the reachability fix itself."""
    monkeypatch.delenv(G.ENV_MEM_LIMIT, raising=False)
    pol = G.device_policy("auto", probe=_probe([], 3000), cuda_available=lambda: True, limit_mib=4300)
    assert pol["decision"] == "CUDA"
    assert pol["record"]["limit_mib"] == 4300.0


def test_a_raised_limit_still_refuses_above_it(monkeypatch):
    """Control: 5,812 MiB (W7's measured desktop peak) is still above 4,300 => REFUSE. Proves the
    gate is a real threshold and not merely switched off by being made configurable."""
    monkeypatch.delenv(G.ENV_MEM_LIMIT, raising=False)
    pol = G.device_policy("auto", probe=_probe([], 5812), cuda_available=lambda: True, limit_mib=4300)
    assert pol["decision"] == "REFUSE"


def test_training_process_still_refuses_at_any_limit(monkeypatch):
    """⛔ THE DISCRIMINATING CONTROL, and the one that matters most: the standing invariant is
    "never add load to a box that is training". A live training pid must REFUSE even with an
    absurdly high limit and zero foreign memory. If this ever passes as CUDA, the fix has traded a
    usability gap for a correctness hole."""
    monkeypatch.delenv(G.ENV_MEM_LIMIT, raising=False)
    pol = G.device_policy("auto", probe=_probe([4242], 0), cuda_available=lambda: True,
                          limit_mib=10 ** 9)
    assert pol["decision"] == "REFUSE"
    assert "training process alive" in pol["reason"]


def test_launcher_resolves_env_too(monkeypatch):
    monkeypatch.setenv(G.ENV_MEM_LIMIT, "4300")
    assert G.GpuGapLauncher("cpu", probe=_probe([], 0), log=lambda *_: None).limit_mib == 4300.0


def test_gap_verdict_stayed_pure(monkeypatch):
    """``gap_verdict`` is the literal-tested predicate: it must NOT read the environment, so its
    callers hand it an already-resolved number. Env set to a value that would flip the answer."""
    monkeypatch.setenv(G.ENV_MEM_LIMIT, "99999")
    v = G.gap_verdict([], 3000, limit_mib=1024, cuda_available=True)
    assert v["gap"] is False and v["limit_mib"] == 1024
