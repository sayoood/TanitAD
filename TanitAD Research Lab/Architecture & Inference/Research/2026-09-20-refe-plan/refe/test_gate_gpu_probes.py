#!/usr/bin/env python3
"""Two launch-gate defects that only a REAL model on a REAL GPU exposes (MEASURED 2026-10-12, the P2 pod's launch gate):

1. G-DVB read --amp at the backbone's OUTPUT.  The backbone ends in a LayerNorm, which CUDA autocast always runs in float32, so a correct
   bf16 trainer read 'built float32'.  Fix: gate_checks.amp_probe_module = the backbone's first nn.Linear.
2. G-CKPT spawned two real trainings while the gate process still held the smoke model + its cached CUDA blocks (24.11 GiB): both died
   CUDA OOM.  Fix: launch_gate frees the smoke capture (deep) before check_ckpt.  R43 (2026-10-16, P6 pod): the free leaves this process's
   CUDA CONTEXT (394-474 MiB) on the card, and P6's compile-free smoke needs > 43.6 GiB allocated of 44.43: U and H1 died OOM again.
   Fix: check_ckpt runs BEFORE the build and the smoke, i.e. before this process opens a CUDA context at all.

Run: python refe/test_gate_gpu_probes.py  ->  ZZTEST_PASS <n>.  The CUDA check needs a GPU; without one it reports SKIP_CUDA and the suite
says so in its final line (a skipped check is not a passed one).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def order_ok(src: str) -> bool:
    """launch_gate.py run_checks: check_ckpt is CALLED before the first instrumented_run (the build/smoke that open this process's CUDA
    context), and its evidence is emitted (R43)."""
    i = src.find("def run_checks(")
    end = src.find("\ndef ", i + 10)
    body = src[i:end]
    j = body.find("C.check_ckpt(")
    k = body.find("C.instrumented_run(")
    return i >= 0 and 0 < j < k and "emit(ckpt_ev)" in body


def free_deep_ok(mod) -> bool:
    cap = types.SimpleNamespace(model=object(), first_batch=(1,), opt=object(), sched=object(), loaders=[1], samplers=[2], metrics=[3])
    mod._free(cap, deep=True)
    return cap.model is None and cap.first_batch is None and cap.opt is None and cap.sched is None and cap.loaders == [] \
        and cap.samplers == [] and cap.metrics == [3]


def ckpt_mb_ok(gc_mod, profile: dict, want: tuple) -> bool:
    """R44: check_ckpt hands its trainings the profile's micro-batch (accumulation scaled, effective batch kept, --compile dropped).
    The trainings are faked (rc 1), so this reads the argv check_ckpt BUILDS, on any box."""
    import tempfile
    seen = []
    real = gc_mod._train_sub
    gc_mod._train_sub = lambda args, log: (seen.append(list(args)), 1)[1]
    try:
        with tempfile.TemporaryDirectory() as td:
            ev = gc_mod.check_ckpt(["--batch", "4", "--accum", "2", "--compile", "--out", "x", "--resume"], Path(td) / "ckpt", profile)
    finally:
        gc_mod._train_sub = real
    if not seen or ev.get("status") != "FAIL":
        return False
    return all(a[a.index("--batch") + 1] == want[0] and a[a.index("--accum") + 1] == want[1] and "--compile" not in a and "--resume" not in a
               for a in seen) and ev["details"].get("batch_accum") == [int(want[0]), int(want[1])]


def det_boot_ok(src: str) -> bool:
    """gate_checks._train_sub launches train.py THROUGH the deterministic bootstrap, with the cuBLAS workspace pinned."""
    i = src.find("def _train_sub(")
    body = src[i:src.find("\ndef ", i + 10)]
    return ("torch.use_deterministic_algorithms(True)" in src and '"-c", DET_BOOT' in body and 'CUBLAS_WORKSPACE_CONFIG=":4096:8"' in body)


def cuda_probe(gc_mod):
    """On a GPU, under bf16 autocast: the real backbone class's OUTPUT is float32 (why the old probe failed) and the probe module reads bf16."""
    import torch
    if not torch.cuda.is_available():
        return "SKIP_CUDA"
    from model import REFeConfig, VitS16
    cfg = REFeConfig.for_backbone("vits16")
    bb = VitS16(cfg).cuda().eval()
    seen = {}
    gc_mod.amp_probe_module(bb).register_forward_hook(lambda _m, _i, o: seen.__setitem__("probe", o.dtype))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = bb(torch.rand(1, 3, 64, 64, device="cuda"))
    return out.dtype == torch.float32 and seen.get("probe") == torch.bfloat16


def main() -> int:
    import gate_checks as GC
    import launch_gate as LG
    src_lg = (HERE / "launch_gate.py").read_text(encoding="utf-8")
    src_gc = (HERE / "gate_checks.py").read_text(encoding="utf-8")
    n, fails = 0, []

    def t(name, cond):
        nonlocal n
        if cond is True:
            n += 1
        else:
            fails.append(f"{name}: {cond}")

    t("free_before_ckpt", order_ok(src_lg))
    t("free_deep_clears_pins", free_deep_ok(LG))
    t("probe_hooked_via_helper", "amp_probe_module(self.backbone).register_forward_hook" in src_gc)
    t("ckpt_runs_deterministic", det_boot_ok(src_gc))
    t("ckpt_p6_micro_batch_2", LG.PROFILES["refe_p6"].get("ckpt_micro_batch") == 2 and "ckpt_micro_batch" not in LG.PROFILES["refe_p2"])
    t("ckpt_argv_scaled", ckpt_mb_ok(GC, {"ckpt_steps": 4, "ckpt_micro_batch": 2}, ("2", "4")))
    t("ckpt_argv_default_unchanged", ckpt_mb_ok(GC, {"ckpt_steps": 4}, ("4", "2")))
    cu = cuda_probe(GC)
    skipped = cu == "SKIP_CUDA"
    if not skipped:
        t("cuda_probe_reads_bf16_where_output_reads_fp32", cu)

    # mutations: each must make the matching check FAIL
    muts = [
        ("ckpt_after_smoke", lambda: order_ok(src_lg.replace(
            '        ckpt_ev = C.check_ckpt(s_argv, work / "ckpt", prof2, arm=arm, rehearsal=rehearsal)\n', "").replace(
            "emit(ckpt_ev)", 'emit(C.check_ckpt(s_argv, work / "ckpt", prof2, arm=arm, rehearsal=rehearsal))'))),
        ("ckpt_not_emitted", lambda: order_ok(src_lg.replace("emit(ckpt_ev)", "pass"))),
        ("ckpt_mb_key_ignored", lambda: ckpt_mb_ok(GC, {"ckpt_steps": 4, "ckpt_micro_batch": 2}, ("4", "2"))),
        ("ckpt_plain_launch", lambda: det_boot_ok(src_gc.replace('[sys.executable, "-c", DET_BOOT, str(HERE / "train.py")]',
                                                               '[sys.executable, str(HERE / "train.py")]'))),
        ("ckpt_no_cublas_pin", lambda: det_boot_ok(src_gc.replace(', CUBLAS_WORKSPACE_CONFIG=":4096:8"', ""))),
        ("ckpt_no_deterministic_algorithms", lambda: det_boot_ok(src_gc.replace("torch.use_deterministic_algorithms(True)", "pass"))),
        ("probe_back_on_output", lambda: "amp_probe_module(self.backbone).register_forward_hook" in src_gc.replace(
            "amp_probe_module(self.backbone).register_forward_hook", "self.backbone.register_forward_hook")),
    ]
    for name, fn in muts:
        if fn() is True:
            fails.append(f"mutation {name} stayed GREEN")
        else:
            n += 1
    if not skipped:                     # the probe helper reverted to the backbone itself must read float32 on the GPU
        GC2 = types.SimpleNamespace(amp_probe_module=lambda bb: bb)
        if cuda_probe(GC2) is True:
            fails.append("mutation probe_helper_returns_backbone stayed GREEN")
        else:
            n += 1
    if fails:
        print("ZZTEST_FAIL", fails)
        return 1
    print(f"ZZTEST_PASS {n}" + ("  (SKIP_CUDA: the GPU check did NOT run)" if skipped else "  (incl. the CUDA probe on this GPU)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
