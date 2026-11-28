"""R44: G-CKPT's round trip runs at the profile's `ckpt_micro_batch` (P6: 2, accumulation scaled to keep the effective batch).
argv: <refe dir>.  Exact-anchor replacements; every anchor must occur exactly once."""
import sys
from pathlib import Path

R = Path(sys.argv[1])


def sub(name, pairs):
    p = R / name
    t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert t.count(old) == 1, (name, t.count(old), old[:80])
        t = t.replace(old, new)
    p.write_bytes(t.encode("utf-8"))


sub("gate_checks.py", [
    ('    def with_out(tag, extra):\n'
     '        # --compile is dropped: the resume contract is independent of it, and three compiles would cost ~6 min on the pod\n'
     '        return _drop(_drop(_set(_set(argv, "--out", str(work / tag)), "--steps", str(n)), "--resume"), "--compile") + extra\n',
     '    def with_out(tag, extra):\n'
     '        # --compile is dropped: the resume contract is independent of it, and three compiles would cost ~6 min on the pod\n'
     '        a_ = _drop(_drop(_set(_set(argv, "--out", str(work / tag)), "--steps", str(n)), "--resume"), "--compile")\n'
     '        mb = profile.get("ckpt_micro_batch")\n'
     '        if mb:\n'
     '            # R44 (MEASURED 2026-10-16, P6 pod): under DET_BOOT\'s deterministic kernels P6\'s eager micro-batch of 4 needs >= 43.70 GiB ALLOCATED of\n'
     '            # the A40\'s 44.43 ALONE on the card (U died OOM with no other process; H1 survived the same step: the edge).  The resume contract\n'
     '            # (model, optimiser, data position, loss trajectory bit-identical) is independent of the micro-batch exactly as it is of --compile,\n'
     '            # so the round trip runs at micro-batch `mb` with the accumulation scaled to keep the effective batch.\n'
     '            b0, k0 = int(a_[a_.index("--batch") + 1]), int(a_[a_.index("--accum") + 1])\n'
     '            if (b0 * k0) % int(mb):\n'
     '                raise ValueError(f"ckpt_micro_batch {mb} does not divide the smoke\'s effective batch {b0} x {k0}")\n'
     '            a_ = _set(_set(a_, "--batch", str(mb)), "--accum", str(b0 * k0 // int(mb)))\n'
     '        return a_ + extra\n'
     '    w_ = with_out("U", [])\n'
     '    det["batch_accum"] = [int(w_[w_.index("--batch") + 1]), int(w_[w_.index("--accum") + 1])] if "--batch" in w_ and "--accum" in w_ else None\n'),
])

sub("launch_gate.py", [
    ('                "live_terms": P6_TERMS,\n',
     '                "live_terms": P6_TERMS,\n'
     '                # R44: G-CKPT\'s round trip at micro-batch 2 x accum 4 (the smoke\'s effective 8): batch 4 under deterministic kernels needs >= 43.70 GiB of 44.43\n'
     '                "ckpt_micro_batch": 2,\n'),
])

sub("test_gate_gpu_probes.py", [
    ('def det_boot_ok(src: str) -> bool:\n',
     'def ckpt_mb_ok(gc_mod, profile: dict, want: tuple) -> bool:\n'
     '    """R44: check_ckpt hands its trainings the profile\'s micro-batch (accumulation scaled, effective batch kept, --compile dropped).\n'
     '    The trainings are faked (rc 1), so this reads the argv check_ckpt BUILDS, on any box."""\n'
     '    import tempfile\n'
     '    seen = []\n'
     '    real = gc_mod._train_sub\n'
     '    gc_mod._train_sub = lambda args, log: (seen.append(list(args)), 1)[1]\n'
     '    try:\n'
     '        with tempfile.TemporaryDirectory() as td:\n'
     '            ev = gc_mod.check_ckpt(["--batch", "4", "--accum", "2", "--compile", "--out", "x", "--resume"], Path(td) / "ckpt", profile)\n'
     '    finally:\n'
     '        gc_mod._train_sub = real\n'
     '    if not seen or ev.get("status") != "FAIL":\n'
     '        return False\n'
     '    return all(a[a.index("--batch") + 1] == want[0] and a[a.index("--accum") + 1] == want[1] and "--compile" not in a and "--resume" not in a\n'
     '               for a in seen) and ev["details"].get("batch_accum") == [int(want[0]), int(want[1])]\n'
     '\n'
     '\n'
     'def det_boot_ok(src: str) -> bool:\n'),
    ('    t("ckpt_runs_deterministic", det_boot_ok(src_gc))\n',
     '    t("ckpt_runs_deterministic", det_boot_ok(src_gc))\n'
     '    t("ckpt_p6_micro_batch_2", LG.PROFILES["refe_p6"].get("ckpt_micro_batch") == 2 and "ckpt_micro_batch" not in LG.PROFILES["refe_p2"])\n'
     '    t("ckpt_argv_scaled", ckpt_mb_ok(GC, {"ckpt_steps": 4, "ckpt_micro_batch": 2}, ("2", "4")))\n'
     '    t("ckpt_argv_default_unchanged", ckpt_mb_ok(GC, {"ckpt_steps": 4}, ("4", "2")))\n'),
    ('        ("ckpt_plain_launch", lambda: det_boot_ok(',
     '        ("ckpt_mb_key_ignored", lambda: ckpt_mb_ok(GC, {"ckpt_steps": 4, "ckpt_micro_batch": 2}, ("4", "2"))),\n'
     '        ("ckpt_plain_launch", lambda: det_boot_ok('),
])
print("patched r44")
