"""R43: G-CKPT runs FIRST in run_checks, before this process opens a CUDA context; p6_pod.sh exports expandable_segments.
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


sub("launch_gate.py", [
    ('    b_argv, s_argv = derived_argvs(prof, argv, work, gate_sets, gate_pdm, gate_bank)\n'
     '    need_b = any(c in want for c in ("G-DVB", "G-CLOCK"))\n',
     '    b_argv, s_argv = derived_argvs(prof, argv, work, gate_sets, gate_pdm, gate_bank)\n'
     '    prof2 = dict(prof)\n'
     '    # G-CKPT runs FIRST (R43, MEASURED 2026-10-16, P6 pod): its round trip runs two real trainings in SUBPROCESSES, and the CUDA context this\n'
     '    # process opens for the build/smoke (394-474 MiB, which no free can release) is taken from them.  P6\'s compile-free smoke needs > 43.6 GiB\n'
     '    # ALLOCATED of the A40\'s 44.43: U and H1 died OOM with the smoke freed (0.00-0.14 GiB reserved), with and without expandable_segments.\n'
     '    # check_ckpt reads only s_argv + the profile, never the build or the smoke capture, so running it before them changes none of its inputs.\n'
     '    ckpt_ev = None\n'
     '    if "G-CKPT" in want:\n'
     '        _log_gpu_reserved("before G-CKPT")\n'
     '        ckpt_ev = C.check_ckpt(s_argv, work / "ckpt", prof2, arm=arm, rehearsal=rehearsal)\n'
     '    need_b = any(c in want for c in ("G-DVB", "G-CLOCK"))\n'),
    ('    s = C.instrumented_run(s_argv, mode="smoke", work=work / "smoke", arm=arm) if need_s else None\n'
     '    prof2 = dict(prof)\n',
     '    s = C.instrumented_run(s_argv, mode="smoke", work=work / "smoke", arm=arm) if need_s else None\n'),
    ('    if "G-CKPT" in want:\n'
     '        # the round trip runs TWO real trainings in subprocesses: the smoke model, its optimizer and the caching allocator behind them\n'
     '        # must leave THIS process\'s GPU first (MEASURED 2026-10-12, P2 pod: this process held 24.11 GiB, U and H1 both died CUDA OOM on 44 GiB)\n'
     '        if s is not None:\n'
     '            _free(s, deep=True)\n'
     '        _log_gpu_reserved("before G-CKPT")\n'
     '        emit(C.check_ckpt(s_argv, work / "ckpt", prof2, arm=arm, rehearsal=rehearsal))\n',
     '    if "G-CKPT" in want:\n'
     '        # computed FIRST, above (R43; the 2026-10-12 P2 fix, a deep free of the smoke before check_ckpt, left the context behind)\n'
     '        emit(ckpt_ev)\n'),
    ('def _log_gpu_reserved(when: str) -> None:\n'
     '    try:\n'
     '        import torch\n'
     '        if torch.cuda.is_available():\n'
     '            print(f"  GPU reserved by the gate process {when}: {torch.cuda.memory_reserved() / 2 ** 30:.2f} GiB", flush=True)\n',
     'def _log_gpu_reserved(when: str) -> None:\n'
     '    try:\n'
     '        import torch\n'
     '        if torch.cuda.is_available():\n'
     '            init = torch.cuda.is_initialized()\n'
     '            res = torch.cuda.memory_reserved() / 2 ** 30 if init else 0.0\n'
     '            print(f"  GPU reserved by the gate process {when}: {res:.2f} GiB (CUDA initialised in this process: {init})", flush=True)\n'),
])

sub("test_gate_gpu_probes.py", [
    ('2. G-CKPT spawned two real trainings while the gate process still held the smoke model + its cached CUDA blocks (24.11 GiB): both died\n'
     '   CUDA OOM.  Fix: launch_gate frees the smoke capture (deep) before check_ckpt.\n',
     '2. G-CKPT spawned two real trainings while the gate process still held the smoke model + its cached CUDA blocks (24.11 GiB): both died\n'
     '   CUDA OOM.  Fix: launch_gate frees the smoke capture (deep) before check_ckpt.  R43 (2026-10-16, P6 pod): the free leaves this process\'s\n'
     '   CUDA CONTEXT (394-474 MiB) on the card, and P6\'s compile-free smoke needs > 43.6 GiB allocated of 44.43: U and H1 died OOM again.\n'
     '   Fix: check_ckpt runs BEFORE the build and the smoke, i.e. before this process opens a CUDA context at all.\n'),
    ('def order_ok(src: str) -> bool:\n'
     '    """launch_gate.py: inside the G-CKPT branch, the deep free of the smoke capture comes BEFORE check_ckpt."""\n'
     '    i = src.find(\'if "G-CKPT" in want:\')\n'
     '    j = src.find("C.check_ckpt(", i)\n'
     '    k = src.find("_free(s, deep=True)", i)\n'
     '    return i >= 0 and j > i and i < k < j\n',
     'def order_ok(src: str) -> bool:\n'
     '    """launch_gate.py run_checks: check_ckpt is CALLED before the first instrumented_run (the build/smoke that open this process\'s CUDA\n'
     '    context), and its evidence is emitted (R43)."""\n'
     '    i = src.find("def run_checks(")\n'
     '    end = src.find("\\ndef ", i + 10)\n'
     '    body = src[i:end]\n'
     '    j = body.find("C.check_ckpt(")\n'
     '    k = body.find("C.instrumented_run(")\n'
     '    return i >= 0 and 0 < j < k and "emit(ckpt_ev)" in body\n'),
    ('        ("free_removed", lambda: order_ok(src_lg.replace("            _free(s, deep=True)\\n", ""))),\n'
     '        ("free_after_ckpt", lambda: order_ok(src_lg.replace("_free(s, deep=True)", "pass").replace(\n'
     '            \'emit(C.check_ckpt(s_argv, work / "ckpt", prof2, arm=arm, rehearsal=rehearsal))\',\n'
     '            \'emit(C.check_ckpt(s_argv, work / "ckpt", prof2, arm=arm, rehearsal=rehearsal)); _free(s, deep=True)\'))),\n',
     '        ("ckpt_after_smoke", lambda: order_ok(src_lg.replace(\n'
     '            \'        ckpt_ev = C.check_ckpt(s_argv, work / "ckpt", prof2, arm=arm, rehearsal=rehearsal)\\n\', "").replace(\n'
     '            "emit(ckpt_ev)", \'emit(C.check_ckpt(s_argv, work / "ckpt", prof2, arm=arm, rehearsal=rehearsal))\'))),\n'
     '        ("ckpt_not_emitted", lambda: order_ok(src_lg.replace("emit(ckpt_ev)", "pass"))),\n'),
])

sub("p6_pod.sh", [
    ('export REFE_BACKBONE_ROOT="$DATA/backbones" PYTHONUNBUFFERED=1\n',
     'export REFE_BACKBONE_ROOT="$DATA/backbones" PYTHONUNBUFFERED=1\n'
     '# R43 (MEASURED 2026-10-16, this pod): a gate smoke died CUDA OOM with 5.79 GiB reserved-but-unallocated beside 37.72 GiB allocated; the launch\n'
     '# argv peaks at 41.29 GiB allocated of 44.43, so fragmentation alone can end a 7-day run.  The gate, every instrument and the launch inherit it.\n'
     'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"\n'),
])
print("patched r43")
