"""Mutation proof for PREREG_V8F_CZ's instruments: every mutation must turn stack/tests/test_v8f_cz.py RED; the unmutated tree must be
GREEN before and after, and every file is restored byte-identical.   python mutate_cz.py <repo root> <python exe>"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
PY = sys.argv[2]
MODEL = ROOT / "stack" / "tanitad" / "models" / "v8f_mini.py"
TRAIN = ROOT / "stack" / "scripts" / "v8f_mini_train.py"
EVAL = ROOT / "stack" / "scripts" / "v8f_mini_eval.py"
CODE = ROOT / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "code"
GATE, CZ, QUEUE = CODE / "v8f_cz_gate_read.py", CODE / "v8f_cz_read.py", CODE / "v8f_mini_box2_queue.cmd"
MUT = [
    (MODEL, "zero mode still masks the absent slots", "            m = lt[\"has\"][:, None].expand_as(tgt)\n",
     "            m = torch.isfinite(tgt) & lt[\"has\"][:, None]\n"),
    (MODEL, "the absent-slot target is 1, not 0", "            tgt = torch.where(torch.isfinite(tgt), tgt, torch.zeros_like(tgt))\n",
     "            tgt = torch.where(torch.isfinite(tgt), tgt, torch.ones_like(tgt))\n"),
    (MODEL, "built_levers misreports the lever", "                \"cons_absent\": cfg.cons_absent,\n", "                \"cons_absent\": \"mask\",\n"),
    (MODEL, "an unknown cons_absent is accepted", "        if self.cons_absent not in (\"mask\", \"zero\"):\n", "        if False:\n"),
    (TRAIN, "the M2b-cz arm does not move the lever", "    \"M2b-cz\": dict(navonly=False, consistency=True, wm=\"b\", cons_absent=\"zero\"),",
     "    \"M2b-cz\": dict(navonly=False, consistency=True, wm=\"b\", cons_absent=\"mask\"),"),
    (TRAIN, "the --cons-absent override is ignored", "        spec[\"cons_absent\"] = a.cons_absent\n", "        pass\n"),
    (TRAIN, "make_config drops the lever", "kin_mode=a.kin_mode, cons_absent=lv[\"cons_absent\"])", "kin_mode=a.kin_mode, cons_absent=\"mask\")"),
    (TRAIN, "declared-vs-built does not compare the lever", ", (\"cons_absent\", lv[\"cons_absent\"])):", "):"),
    (TRAIN, "the does-nothing check is dropped",
     "    if lv[\"cons_absent\"] == \"zero\" and not (cfg.w_cons > 0.0 and cfg.w_tac > 0.0):\n", "    if False:\n"),
    (TRAIN, "the flag has no declared-vs-built entry", ", \"cons_absent\": \"lever:cons_absent\",", ","),
    (EVAL, "absmask zeroes EVERY constraint slot",
     "                    code[:, c0:] = torch.where(absent, torch.zeros_like(code0[:, c0:]), code0[:, c0:])\n",
     "                    code[:, c0:] = 0.0\n"),
    (EVAL, "absmask zeroes the PRESENT slots", "torch.where(absent, torch.zeros_like(code0[:, c0:]), code0[:, c0:])",
     "torch.where(~absent, torch.zeros_like(code0[:, c0:]), code0[:, c0:])"),
    (EVAL, "the absent-slot fraction is not recorded", "if a.tac == \"absmask\" else None}", "if False else None}"),
    (GATE, "PASS ignores the training-seed floor", "and abs(rows[f\"{arm}_s{s}\"][\"absmask\"][\"delta\"]) > 2 * F[arm] for s in SEEDS)",
     "for s in SEEDS)"),
    (GATE, "PASS needs only one seed", "        ok = all(rows[f\"{arm}_s{s}\"][\"absmask\"][\"delta\"] < 0",
     "        ok = any(rows[f\"{arm}_s{s}\"][\"absmask\"][\"delta\"] < 0"),
    (GATE, "the manipulation check is dropped", "                    if rec.get(\"mode\") != \"absmask\" or fr is None or not 0.0 < float(fr) < 1.0:\n",
     "                    if False:\n"),
    (GATE, "the tacpred drift check is dropped", "            if abs(got - r[\"ade6_pred\"]) > 1e-3:\n", "            if False:\n"),
    (CZ, "SUCCESS ignores the training-seed floor", "and abs(rows[f\"s{s}\"][\"delta\"][\"delta\"]) > 2 * F for s in SEEDS)", "for s in SEEDS)"),
    (CZ, "SUCCESS needs only one seed", "    success = all(rows[f\"s{s}\"][\"delta\"][\"delta\"] < 0", "    success = any(rows[f\"s{s}\"][\"delta\"][\"delta\"] < 0"),
    (CZ, "the mechanism is always OK", "mechanism=bool(sh[\"M2b-cz\"] <= 0.5 * sh[\"M2b\"])", "mechanism=True"),
    (CZ, "the lever provenance check is dropped", "        if got != WANT_LEVER[arm]:\n", "        if False:\n"),
    (CZ, "the step-budget check is dropped", "    if len({d[\"steps\"] for d in S.values()}) != 1:\n", "    if False:\n"),
    (CZ, "the seed floor is zero", "    F = abs(Z[(\"M2b\", 0)][\"ade6\"].mean() - Z[(\"M2b\", 1)][\"ade6\"].mean())\n", "    F = 0.0\n"),
    (QUEUE, "tacabs runs T's code unmodified", "call :tacone %%A %%S absmask tacabs", "call :tacone %%A %%S pred tacabs"),
    (QUEUE, "the queue has no gatearm dispatch", 'if /i "%~1"=="gatearm" goto :gatearm', "rem no gatearm"),
    (QUEUE, "tacabsarm runs the full oracle", "call :tacone %~2 %~3 absmask tacabs", "call :tacone %~2 %~3 oracle tacabs"),
]


def run():
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "stack/tests/test_v8f_cz.py"], cwd=ROOT,
                       capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    return r.returncode, r.stdout.strip().splitlines()[-1]


orig = {p: p.read_bytes() for p in (MODEL, TRAIN, EVAL, GATE, CZ, QUEUE)}
rc, tail = run()
print("baseline:", rc, tail, flush=True)
assert rc == 0
red = 0
try:
    for path, name, a, b in MUT:
        src = orig[path]                                  # bytes: the queue is CRLF and must stay so
        assert src.count(a.encode()) == 1, name
        path.write_bytes(src.replace(a.encode(), b.encode()))
        rc, tail = run()
        red += rc != 0
        print(f"{'RED ' if rc else 'SURVIVED'} [{path.name}] {name}: {tail}", flush=True)
        path.write_bytes(orig[path])
finally:
    for p, b in orig.items():
        p.write_bytes(b)
rc, tail = run()
print("restored:", rc, tail, "| byte-identical:", all(p.read_bytes() == b for p, b in orig.items()))
print(f"{red}/{len(MUT)} RED")
sys.exit(0 if red == len(MUT) and rc == 0 else 1)
