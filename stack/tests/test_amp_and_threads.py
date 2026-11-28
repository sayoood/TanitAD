"""P4-8 / P4-9 — the autocast dtype is PROBED, and OMP is set before torch.

Two defects of the same shape: a value that decides how the run computes, chosen
by assumption and recorded nowhere. Two runs on two machines could differ in
numerical precision and in thread count with byte-identical launch lines.
"""
import os
import pathlib
import re
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import train_v6_staged as T  # noqa: E402


# ---------------------------------------------------------------- P4-9
def test_omp_is_set_before_torch_is_imported():
    """⛔ THE WHOLE POINT. torch sizes its thread pool AT IMPORT, so a setdefault
    after `import torch` changes the environment and nothing else — the guard
    was inert where it used to live, ~7,700 lines below the import."""
    src = pathlib.Path(T.__file__).read_text(encoding="utf-8")
    omp = src.index('os.environ.setdefault("OMP_NUM_THREADS"')
    imp = re.search(r"^import torch$", src, re.M).start()
    assert omp < imp, (
        "OMP_NUM_THREADS must be set BEFORE `import torch`; it is currently set "
        f"at offset {omp}, after the import at {imp}")


def test_omp_is_setdefault_so_the_launcher_still_wins():
    src = pathlib.Path(T.__file__).read_text(encoding="utf-8")
    assert 'os.environ.setdefault("OMP_NUM_THREADS"' in src
    assert 'os.environ["OMP_NUM_THREADS"] =' not in src, (
        "must not OVERRIDE an explicit value from the launcher")


def test_the_resolved_thread_settings_are_recorded():
    p = T.run_provenance()
    assert p["omp_num_threads"] == os.environ.get("OMP_NUM_THREADS", "<unset>")
    assert isinstance(p["torch_num_threads"], (int, str))


# ---------------------------------------------------------------- P4-8
def test_amp_off_resolves_to_fp32_and_says_why():
    r = T.resolve_amp_dtype("cuda", want_amp=False)
    assert r["name"] == "fp32" and r["autocast"] is False
    assert "no-amp" in r["reason"]


def test_cpu_never_autocasts_and_says_why():
    r = T.resolve_amp_dtype("cpu", want_amp=True)
    assert r["autocast"] is False
    assert "not cuda" in r["reason"]


def test_bf16_capable_device_selects_bf16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    r = T.resolve_amp_dtype("cuda", want_amp=True)
    assert r["dtype"] is torch.bfloat16 and r["autocast"] is True
    assert r["bf16_supported"] is True


def test_no_bf16_falls_back_to_fp32_NOT_fp16(monkeypatch):
    """⛔ THE LOAD-BEARING ONE. fp16 autocast without a GradScaler underflows
    gradients to zero and trains a WORSE model without failing. The fallback
    must be fp32, and it must say so."""
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    r = T.resolve_amp_dtype("cuda", want_amp=True)
    assert r["name"] == "fp32", "must NOT silently use fp16"
    assert r["dtype"] is not torch.float16
    assert r["autocast"] is False
    assert "GradScaler" in r["reason"], "the refusal must explain itself"


def test_a_broken_probe_degrades_safely_and_is_visible(monkeypatch):
    """A probe that raises must not take the run down, and must not be read as
    'bf16 is fine' — the conservative branch is the safe one here."""
    def boom():
        raise RuntimeError("driver exploded")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", boom)
    r = T.resolve_amp_dtype("cuda", want_amp=True)
    assert r["name"] == "fp32" and r["bf16_supported"] is False
    assert "probe failed" in r["reason"]


def test_the_autocast_call_uses_the_resolved_spec_not_a_hardcode():
    """⚠️ Pins the WIRING, not just the helper. A correct resolver that nothing
    calls is the defect this item exists to fix (cf. the dead ds_val)."""
    src = pathlib.Path(T.__file__).read_text(encoding="utf-8")
    assert 'dtype=amp_spec["dtype"] or torch.float32' in src
    assert "torch.autocast(dev_type, dtype=torch.bfloat16" not in src, (
        "the hardcoded bf16 autocast is still present in the step loop")


def test_the_reordering_ACTUALLY_changes_the_thread_pool():
    """⭐ MEASURED, not argued: the fix has a 2.67x effect, and the old code was
    demonstrably INERT.

    A source-order assertion proves the lines moved; it does not prove the move
    mattered. This runs two clean subprocesses with OMP_NUM_THREADS UNSET in the
    environment, so the module's own setdefault is the only source of the value:

        module imported FIRST (the fixed order) -> torch.get_num_threads() == 6
        torch  imported FIRST (the old order)   -> torch.get_num_threads() == 16

    ⚠️ Note the environment must be scrubbed. With OMP_NUM_THREADS already
    exported both orders read 6, and the test would pass while measuring
    nothing — the same empty-verification family as TRAIN-C10.
    """
    import subprocess

    stack_dir = pathlib.Path(T.__file__).resolve().parents[1]   # .../stack
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    env["PYTHONIOENCODING"] = "utf-8"

    def probe(torch_first: bool) -> int:
        pre = "import torch\n" if torch_first else ""
        code = (f"{pre}import sys\n"
                f"sys.path.insert(0, r'{stack_dir / 'scripts'}')\n"
                f"sys.path.insert(0, r'{stack_dir}')\n"
                "import train_v6_staged\n"
                "import torch\n"
                "print(torch.get_num_threads())\n")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, encoding="utf-8", errors="replace",
                             env=env, cwd=str(stack_dir), timeout=300)
        assert out.returncode == 0, out.stderr[-2000:]
        return int(out.stdout.strip().splitlines()[-1])

    fixed = probe(torch_first=False)
    stale = probe(torch_first=True)
    assert fixed == 6, f"the fixed order must yield 6 threads, got {fixed}"
    assert stale != fixed, (
        "torch-first and module-first give the SAME thread count, so this box "
        "cannot demonstrate the effect and the test is measuring nothing")
