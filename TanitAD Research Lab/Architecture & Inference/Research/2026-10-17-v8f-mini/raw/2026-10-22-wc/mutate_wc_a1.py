"""Mutation proof for PREREG_V8F_WC's instruments: every mutation must turn stack/tests/test_v8f_wc.py RED; the unmutated tree must be
GREEN before and after, and every file is restored byte-identical.   python mutate_wc.py <repo root> <python exe>"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
PY = sys.argv[2]
MODEL = ROOT / "stack" / "tanitad" / "models" / "v8f_mini.py"
TRAIN = ROOT / "stack" / "scripts" / "v8f_mini_train.py"
CODE = ROOT / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "code"
WC, QUEUE = CODE / "v8f_wc_read.py", CODE / "v8f_mini_box2_queue.cmd"
MUT = [
    (MODEL, "built_levers misreports w_cons", '"w_tac": cfg.w_tac, "w_cons": cfg.w_cons}}', '"w_tac": cfg.w_tac, "w_cons": 0.5}}'),
    (MODEL, "the loss ignores w_cons", "+ cfg.w_cons * T[\"cons\"])", "+ 0.5 * T[\"cons\"])"),
    (MODEL, "the constraint term leaks into O's head", "        T[\"cons\"] = (hub * m).sum() / m.sum().clamp_min(1.0)\n",
     "        T[\"cons\"] = (hub * m).sum() / m.sum().clamp_min(1.0) + 1e-3 * sum(p.sum() for p in self.head_o.parameters())\n"),
    (TRAIN, "make_config drops w_cons", "w_cons=a.w_cons, w_traj=a.w_traj", "w_cons=0.5, w_traj=a.w_traj"),
    (TRAIN, "declared-vs-built does not compare w_cons", '"w_ckpt", "w_tac", "w_cons", "w_traj", "w_pen",', '"w_ckpt", "w_tac", "w_traj", "w_pen",'),
    (WC, "SUCCESS ignores the training-seed floor", "and abs(d[\"delta\"]) > 2 * F)", ")"),
    (WC, "the verdict needs only one seed", "verdict = \"SUCCESS\" if all(rows[f\"s{s}\"][\"passed\"] for s in seeds)",
     "verdict = \"SUCCESS\" if any(rows[f\"s{s}\"][\"passed\"] for s in seeds)"),
    (WC, "stage s0 always continues", "        go = rows[\"s0\"][\"passed\"]\n", "        go = True\n"),
    (WC, "the mechanism is always OK", "        mech = bool(dm[\"delta\"] < 0 and dm[\"separated\"])\n", "        mech = True\n"),
    (WC, "T's values are read from the GT oracle", "\"t_cons\": np.asarray(za[\"tac_code\"], np.float64)[:, C0:C0 + N_CONS]",
     "\"t_cons\": np.asarray(zo[\"tac_code\"], np.float64)[:, C0:C0 + N_CONS]"),
    (WC, "the built-w_cons check is dropped", "            if w is None or float(w) != WC:\n", "            if False:\n"),
    (WC, "the argv check is dropped", "            if got is None or float(got) != WC:\n", "            if False:\n"),
    (WC, "M2b's argv is not checked", "            if _flag(argv, \"--w-cons\") is not None:\n", "            if False:\n"),
    (WC, "the GT-equality check is dropped",
     "        if not np.array_equal(v[\"has\"], has) or not np.array_equal(v[\"g_cons\"][has], G[has]):\n", "        if False:\n"),
    (WC, "Amendment 1: the all-goal fallback is dropped", "            if \"tac_code\" not in zo.files:                 # Amendment 1",
     "            if False:                 # Amendment 1"),
    (WC, "Amendment 1: check-inputs never loads the evals", "        Z, _, eid = load_evals(a.eval_dir, (0, 1))\n", "        return {}\n"),
    (WC, "the step-budget check is dropped", "    if len({d[\"steps\"] for d in sm.values()}) != 1:\n", "    if False:\n"),
    (WC, "the selfcheck gate is not read", "        if v[\"selfcheck.json\"].get(\"declared_vs_built\") != \"PASS\":\n", "        if False:\n"),
    (WC, "the seed floor is zero", "    return abs(m[0] - m[1])\n", "    return 0.0\n"),
    (WC, "lonspeed is read as a contiguous slice", '"lonspeed": (12, 13, 14, 15, 22, 23, 24, 25)', '"lonspeed": tuple(range(12, 20))'),
    (QUEUE, "trainwc drops the lever", 'set "TW="%PY%" "%TRAIN%" --arm M2b --w-cons %WC% --seed %~2',
     'set "TW="%PY%" "%TRAIN%" --arm M2b --seed %~2'),
    (QUEUE, "the real trainwc run is not the printed command", '%TW% >> "%OUT%\\train.log" 2>&1',
     '"%PY%" "%TRAIN%" --arm M2b --seed %~2 --out "%OUT%" %A% >> "%OUT%\\train.log" 2>&1'),
    (QUEUE, "gatewc drops the lever", 'set "GW="%PY%" "%TRAIN%" --arm M2b --w-cons %WC% --out', 'set "GW="%PY%" "%TRAIN%" --arm M2b --out'),
    (QUEUE, "WC is 2.0", 'set "WC=4.0"', 'set "WC=2.0"'),
    (QUEUE, "evalwc's GT oracle runs T's code", "call :tacone M2b-wc %~2 oracle tacorc", "call :tacone M2b-wc %~2 pred tacorc"),
    (QUEUE, "the queue has no gatewc dispatch", 'if /i "%~1"=="gatewc" goto :gatewc', "rem no gatewc"),
]


def run():
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "stack/tests/test_v8f_wc.py"], cwd=ROOT,
                       capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    return r.returncode, r.stdout.strip().splitlines()[-1]


rc, last = run()
print(f"BASELINE rc={rc} {last}")
assert rc == 0, "the unmutated tree must be GREEN"
orig = {p: p.read_bytes() for p in {m[0] for m in MUT}}
red = 0
for path, name, old, new in MUT:
    b = orig[path]
    o, n = old.encode(), new.encode()
    if path.suffix == ".cmd":
        o, n = o.replace(b"\n", b"\r\n"), n.replace(b"\n", b"\r\n")
    assert b.count(o) == 1, f"{name}: anchor found {b.count(o)} times in {path.name}"
    path.write_bytes(b.replace(o, n))
    try:
        rc, last = run()
    finally:
        path.write_bytes(b)
    red += rc != 0
    print(f"{'RED  ' if rc != 0 else 'GREEN'} {path.name}: {name} -> {last}")
assert all(p.read_bytes() == orig[p] for p in orig), "a file was not restored"
rc, last = run()
print(f"AFTER rc={rc} {last}")
print(f"{red} / {len(MUT)} mutations RED")
sys.exit(0 if red == len(MUT) and rc == 0 else 1)
