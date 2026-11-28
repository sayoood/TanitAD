"""Mutation proof for PREREG_V8F_TAC's instruments: every mutation must turn stack/tests/test_v8f_tac.py RED; the unmutated tree must be
GREEN before and after, and every file is restored byte-identical.   python mutate_tac.py <repo root> <python exe>"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
PY = sys.argv[2]
MODEL = ROOT / "stack" / "tanitad" / "models" / "v8f_mini.py"
EVAL = ROOT / "stack" / "scripts" / "v8f_mini_eval.py"
READ = ROOT / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "code" / "v8f_tac_read.py"
QUEUE = READ.parent / "v8f_mini_box2_queue.cmd"
GUARD = READ.parent / "box2_ops" / "v8f_guard.sh"
READ_G = READ.parent / "v8f_tac_groups_read.py"
MUT = [
    (MODEL, "the override is ignored (O always reads T's own code)",
     "            code = code_override.to(device=code.device, dtype=code.dtype)\n", "            pass\n"),
    (MODEL, "out['code'] records T's code, not the code O read",
     "        out[\"code\"] = code                                               # the code O actually read\n",
     "        out[\"code\"] = self._code_pred(tout)\n"),
    (MODEL, "a wrong-shaped override is not refused",
     "            if code_override.shape != code.shape:\n", "            if False:\n"),
    (EVAL, "the shuffle's donor is the window itself",
     "                    dci, dt_, _, _ = wins[int(tperm[i])]\n", "                    dci, dt_, _, _ = wins[i]\n"),
    (EVAL, "the oracle feeds T's own code, not the GT code",
     "                    code = torch.where(has[:, None], model._code_gt(b[\"lab_t\"]).to(code0.dtype), code0)\n",
     "                    code = code0\n"),
    (EVAL, "--tac with --nav-shuffle / --plan beta is not refused",
     "    if a.tac != \"pred\" and (a.plan == \"beta\" or a.nav_shuffle):\n", "    if False:\n"),
    (READ, "USED needs only one checkpoint", "    tac_used = \"USED\" if all(used.values())", "    tac_used = \"USED\" if any(used.values())"),
    (READ, "DECORATIVE whenever nothing is USED", "else (\"DECORATIVE\" if all(inert.values()) else", "else (\"DECORATIVE\" if not any(used.values()) else"),
    (READ, "LARGE ignores the training-seed floor", "and abs(d[\"delta\"]) > 2 * F[arm] for d in ds)", "for d in ds)"),
    (READ, "LARGE needs only one seed", "        large = all(d[\"delta\"] < 0", "        large = any(d[\"delta\"] < 0"),
    (READ, "the reproducibility check is dropped", "            if dev > TOL_REPRO:\n", "            if False:\n"),
    (READ, "the did-the-shuffle-happen check is dropped", "            if sh is None or float((sh > 1e-6).mean()) < MIN_CHANGED:\n",
     "            if sh is None:\n"),
    (QUEUE, "the queue has no tac dispatch", 'if /i "%~1"=="tac" goto :tac', "rem no tac"),
    (QUEUE, "the queue's shuffle eval runs pred", "  call :tacone %%A %%S shuffle tacshuf", "  call :tacone %%A %%S pred tacshuf"),
    (GUARD, "the guard's end test is unanchored", "grep -q -E '^(ZZV8FMINI|ZZBETA|ZZV8FTAC)-(DONE|STOP)'",
     "grep -q -E 'ZZV8FMINI-DONE|ZZV8FMINI-STOP|ZZBETA-DONE|ZZBETA-STOP|ZZV8FTAC-DONE|ZZV8FTAC-STOP'"),
    # ---- PREREG_V8F_TAC Amendment 1: the per-group oracle
    (MODEL, "A1: the londec group loses an index (no partition)", '    "londec": tuple(range(N_GOAL + N_LAT, _C0)),',
     '    "londec": tuple(range(N_GOAL + N_LAT, _C0 - 1)),'),
    (MODEL, "A1: an unknown group name is not refused", '            raise ValueError(f"unknown code group',
     '            continue; ValueError(f"unknown code group'),
    (EVAL, "A1: the group selection is ignored (the whole code gets GT)", "                        code = torch.where(sel[None, :], code, code0)\n",
     "                        pass\n"),
    (EVAL, "A1: --tac-groups without --tac oracle is not refused", '    if a.tac_groups != "all" and a.tac != "oracle":', "    if False:"),
    (READ_G, "A1: CLASSIFIER needs no half share", 'if carries(arm, "decisions") and share_ge(arm, "decisions", 0.5):',
     'if carries(arm, "decisions"):'),
    (READ_G, "A1: SPEED needs no half share", 'elif carries(arm, "lonspeed") and share_ge(arm, "lonspeed", 0.5):', 'elif carries(arm, "lonspeed"):'),
    (READ_G, "A1: CARRIES ignores the training-seed floor",
     '                   and abs(rows[f"{arm}_s{s}"]["groups"][g]["paired"]["delta"]) > 2 * F[arm] for s in SEEDS)',
     '                   for s in SEEDS)'),
    (READ_G, "A1: the manipulation record is not checked",
     '                    if m.get("mode") != "oracle" or m.get("groups") != g or m.get("n_code_dims_replaced") != sizes[g]:',
     "                    if False:"),
    (QUEUE, "A1: the queue has no tacgrp dispatch", 'if /i "%~1"=="tacgrp" goto :tacgrp', "rem no tacgrp"),
]


def run():
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "stack/tests/test_v8f_tac.py"], cwd=ROOT,
                       capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    return r.returncode, r.stdout.strip().splitlines()[-1]


orig = {p: p.read_bytes() for p in (MODEL, EVAL, READ, QUEUE, GUARD, READ_G)}
rc, tail = run()
print("baseline:", rc, tail)
assert rc == 0
red = 0
try:
    for path, name, a, b in MUT:
        src = orig[path]                                  # bytes: the queue is CRLF and must stay so
        assert src.count(a.encode()) == 1, name
        path.write_bytes(src.replace(a.encode(), b.encode()))
        rc, tail = run()
        red += rc != 0
        print(f"{'RED ' if rc else 'SURVIVED'} [{path.name}] {name}: {tail}")
        path.write_bytes(orig[path])
finally:
    for p, b in orig.items():
        p.write_bytes(b)
rc, tail = run()
print("restored:", rc, tail, "| byte-identical:", all(p.read_bytes() == b for p, b in orig.items()))
print(f"{red}/{len(MUT)} RED")
sys.exit(0 if red == len(MUT) and rc == 0 else 1)
