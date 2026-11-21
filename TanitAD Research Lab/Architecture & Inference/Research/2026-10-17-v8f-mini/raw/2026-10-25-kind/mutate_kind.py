"""Mutation proof for PREREG_V8F_KIND's instruments (and its Amendment 1): every mutation must turn stack/tests/test_v8f_kind.py RED; the unmutated tree must be
GREEN before and after, and every file is restored byte-identical.

    python mutate_kind.py <repo root> <python exe> [--check]     (--check: only verify that every anchor occurs exactly once, then exit)

The mutants live in the model (the partition and `code_group_index`), the reader (the committed table, the bars, every refusal, the paired estimator, LAMBDA, the SPEED side-read,
and Amendment 1: the stratum labels, the within-stratum bootstrap, the 10-episode floor, a stratum read that FEEDS the outcome), the stratum builder, the queue (CRLF, edited as bytes)
and the chain. Each mutation is one exact substring replaced once; a mutation whose anchor is not found exactly once aborts the run.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1])
PY = sys.argv[2]
CHECK = "--check" in sys.argv
PKG = ROOT / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
CODE = PKG / "code"
MODEL = ROOT / "stack" / "tanitad" / "models" / "v8f_mini.py"
READER = CODE / "v8f_kind_read.py"
QUEUE = CODE / "v8f_mini_box2_queue.cmd"
CHAIN = CODE / "box2_ops" / "chain_v8f_kind.sh"
BUILDER = PKG / "raw" / "2026-10-25-kind" / "mk_window_strata.py"

MUT = [
    # ---- the partition and code_group_index (model)
    (MODEL, "a cell moves between kinds: speed holds turn_dyaw_deg and loses v_hi_ms", '("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")),',
     '("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "turn_dyaw_deg")),'),
    (MODEL, "geom loses stop_y_m (the kinds no longer tile the 26 cells)", '"lon_d_reach_m", "stop_x_m", "stop_y_m")),', '"lon_d_reach_m", "stop_x_m")),'),
    (MODEL, "code_group_index does not take a kind", "        elif part in CODE_KINDS:                    # PREREG_V8F_KIND: a kind of constraint cell\n",
     "        elif False:                    # PREREG_V8F_KIND: a kind of constraint cell\n"),
    (MODEL, "a kind named like a group", '    "timegap": tuple(_C0 + CI[n] for n in ("lead_tg_s", "lead_tg_min_s")),', '    "latcons": tuple(_C0 + CI[n] for n in ("lead_tg_s", "lead_tg_min_s")),'),
    # ---- the committed table, bars and gains (reader)
    (READER, "the 2F bar is 0.2400", "BAR_2F = 0.2415 ", "BAR_2F = 0.2400 "),
    (READER, "the additivity bar is 0.75", "ADD_BAR = 0.8 ", "ADD_BAR = 0.75 "),
    (READER, "B defaults to 1,000", "\nB = 2000\n", "\nB = 1000\n"),
    (READER, "a gain exactly at 2F does not clear", 'return bool(g["point"] >= bar and g["separated"] and g["point"] > 0)', 'return bool(g["point"] > bar and g["separated"] and g["point"] > 0)'),
    (READER, "a gain need not be separated", 'return bool(g["point"] >= bar and g["separated"] and g["point"] > 0)', 'return bool(g["point"] >= bar and g["point"] > 0)'),
    (READER, "a kind carries at ANY seed, not both", "carriers = [k for k in KINDS if all(ok[s][k] for s in SEEDS)]", "carriers = [k for k in KINDS if any(ok[s][k] for s in SEEDS)]"),
    (READER, "DIFFUSE needs A strictly above 0.8", "elif all(x is not None and x >= add_bar for x in A):", "elif all(x is not None and x > add_bar for x in A):"),
    (READER, "an undefined A reads 0.0", 'A = (s_gain / g_union["point"]) if g_union["point"] > 0 else None', 'A = (s_gain / g_union["point"]) if g_union["point"] > 0 else 0.0'),
    (READER, "gain_k has the wrong sign", 'gain = {k: PGT.paired(pf, own, R[f"kind-only-{k}"]["ade6"], eid, n_boot) for k in KINDS}',
     'gain = {k: PGT.paired(pf, R[f"kind-only-{k}"]["ade6"], own, eid, n_boot) for k in KINDS}'),
    (READER, "necessity_k has the wrong sign", 'nec = {k: PGT.paired(pf, R[f"kind-loo-{k}"]["ade6"], allo, eid, n_boot) for k in KINDS}',
     'nec = {k: PGT.paired(pf, allo, R[f"kind-loo-{k}"]["ade6"], eid, n_boot) for k in KINDS}'),
    (READER, "the union arm is the full oracle", '        return "constraints"', '        return "all"'),
    (READER, "LAMBDA is read against T's own code, not the full oracle", "lam = along[suf] - along[ALL]", "lam = along[suf] - along[OWN]"),
    (READER, "the SPEED side-read needs only one seed", "if all(x >= SPEED_HI for x in g):", "if any(x >= SPEED_HI for x in g):"),
    # ---- the refusals (reader)
    (READER, "the identity control is not enforced", "if not same or not _exact_zero(r):", "if False:"),
    (READER, "the checkpoint sha256 is not compared", "        if have != want:", "        if False:"),
    (READER, "a missing tactical / strategic family is not refused", "        if not raw.get(fam):", "        if False:"),
    (READER, "the window list (eid) is not compared", "refuse(f\"eval_{E['name']}.npz is not on the banked window list (eid differs): the window set or the episode list moved\")", "pass"),
    (READER, "the committed window / episode counts are not checked", "    if (n, ne) != (a.n_windows, a.n_episodes):", "    if False:"),
    (READER, "the no-v9-row windows may differ between oracle arms", "        if any(not np.array_equal(x, fb[0]) for x in fb):", "        if False:"),
    (READER, "the manipulation record is not checked", '        if m.get("mode") != "oracle" or m.get("groups") != spec or m.get("n_code_dims_replaced") != n or alpha is None or abs(float(alpha) - 1.0) > 1e-9:',
     "        if False:"),
    (READER, "the scored checkpoint is not checked", "    ok, ck = _ckpt_ok(js, c)", "    ok, ck = True, None"),
    (READER, "the known-value control (F = 0.1207) is dropped", 'if abs(F - PGT.KNOWN["F"]) > PGT.KNOWN_TOL["F"]:', "if False:"),
    (READER, "the union-is-constraints check is dropped", "    if union != cons or len(union) != len(set(union)):", "    if False:"),
    # ---- Amendment 1: the strata (reader)
    (READER, "CURVE windows get the JUNCTION code", "    s[(np.asarray(curve) == 1) & np.asarray(has, bool)] = 1\n", "    s[(np.asarray(curve) == 1) & np.asarray(has, bool)] = 0\n"),
    (READER, "the 10-episode floor is 3", "MIN_STRATUM_EPISODES = 10 ", "MIN_STRATUM_EPISODES = 3 "),
    (READER, "the floor counts the episodes of the whole eval, not the stratum's", "    ne = int(np.unique(eid[mask]).size)\n    own = arms", "    ne = int(np.unique(eid).size)\n    own = arms"),
    (READER, "the within-stratum pairing is broken", 'r = PGT.paired(pf, own, E["ade6"][mask], eid[mask], n_boot)', 'r = PGT.paired(pf, own, E["ade6"][mask][::-1], eid[mask], n_boot)'),
    (READER, "the contribution to the pooled gain is a mean, not n_stratum / N", 'float(d[s == si].sum() / n_all)', 'float(d[s == si].mean())'),
    (READER, "THE STRATUM READ FEEDS THE OUTCOME (A moves with the CURVE stratum)", '    out_name = dec["outcome"]\n',
     '    out_name = dec["outcome"]\n    if not a.no_strata:\n        dec["A"][0] = dec["A"][0] + 1e-9 * strata["strata"]["CURVE"]["seeds"]["s0"]["gain_union"]["point"]\n'),
    (READER, "a stratum that is not its own labels' function is accepted", "    if not np.array_equal(s, stratum_codes(cu, ti, has.astype(bool))):", "    if False:"),
    (READER, "the strata file's v9-row presence is not compared with the evaluator's", "        if not np.array_equal(has.astype(bool), ~np.asarray(fb, bool)):", "        if False:"),
    (READER, "the strata file's episode index is not compared", '    if not np.array_equal(np.asarray(z["eid"]), np.asarray(eid)):', "    if False:"),
    (READER, "the strata file's tick is not compared", '    if frame is not None and not np.array_equal(2 * np.asarray(z["t"], np.int64), np.asarray(frame, np.int64)):', "    if False:"),
    (READER, "the strata file's release is not checked", '    if prov.get("release_md5") != RELEASE_MD5:', "    if False:"),
    (READER, "--check-inputs does not validate the strata file", "    if not a.no_strata:\n", "    if False:\n"),
    (BUILDER, "the builder gives JUNCTION windows the CURVE code", '    s[(turn_is_turn == 1) & has] = CODE["JUNCTION"]', '    s[(turn_is_turn == 1) & has] = CODE["CURVE"]'),
    (BUILDER, "the builder reads the NOW row's age wrong (a_T = i % 2)", "a_t = np.minimum(np.minimum(i % 3, np.maximum(tt - WINDOW_START, 0)), MAX_AGE_T)", "a_t = np.minimum(np.minimum(i % 2, np.maximum(tt - WINDOW_START, 0)), MAX_AGE_T)"),
    # ---- the queue (CRLF, edited as bytes)
    (QUEUE, "the only-<kind> loop drops leadgap", "for %%K in (speed angle tte timegap geom leadgap) do call :kindorc M2b %~1 %%K only-%%K || exit /b 1",
     "for %%K in (speed angle tte timegap geom) do call :kindorc M2b %~1 %%K only-%%K || exit /b 1"),
    (QUEUE, "the leave-one-out arm runs the single kind", "call :kindorc M2b %~1 all-%%K loo-%%K || exit /b 1", "call :kindorc M2b %~1 %%K loo-%%K || exit /b 1"),
    (QUEUE, "the union arm is not the constraints", "call :kindorc M2b %~1 constraints union || exit /b 1", "call :kindorc M2b %~1 decisions union || exit /b 1"),
    (QUEUE, "the oracle arms run at alpha 0.5", "--tac oracle --tac-groups %~3 --tac-alpha 1 %EC% %ER%\"", "--tac oracle --tac-groups %~3 --tac-alpha 0.5 %EC% %ER%\""),
    (QUEUE, "the own arm is an oracle eval", '-kind-own"\r\nset "EK="%PY%" "%EVAL%" run --ckpt "%OUT%\\ckpt_last.pt" --tag %TG% %EC% %ER% --project --families"',
     '-kind-own"\r\nset "EK="%PY%" "%EVAL%" run --ckpt "%OUT%\\ckpt_last.pt" --tag %TG% --tac oracle --tac-groups all --tac-alpha 1 %EC% %ER% --project --families"'),
    (QUEUE, "the own arm is tagged as the full oracle", 'set "TG=%~1_s%~2-kind-own"', 'set "TG=%~1_s%~2-kind-all"'),
    (QUEUE, "the kind stage is not dispatched", 'if /i "%~1"=="kind" goto :kind', "rem no kind"),
    (QUEUE, "an existing stage is edited (compare)", "echo [ok] compare", "echo [ok] compare."),
    # ---- the chain
    (CHAIN, "the pinned checkpoint sha256 is wrong", "CKPT_SHA=(601cb435", "CKPT_SHA=(701cb435"),
    (CHAIN, "the chain waits for another predecessor", "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_dscale.out}", "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_hist.out}"),
    (CHAIN, "the busy marker is not taken atomically", '( set -C; echo "v8F-mini PREREG_V8F_KIND', '( echo "v8F-mini PREREG_V8F_KIND'),
    (CHAIN, "the chain runs another queue stage", "queue kindrun $EVAL_POLLS kind", "queue kindrun $EVAL_POLLS tacgrp2"),
    (CHAIN, "the union eval is not pulled", 'SUFFIXES="$SUFFIXES kind-union"', 'SUFFIXES="$SUFFIXES kind-uni"'),
    (CHAIN, "check-inputs no longer stops the chain", '|| stop "the reader cannot use the M2b side (check-inputs)"', "|| true"),
    (CHAIN, "the checkpoint is not re-asserted after the evals", 'ckptsha "after the evals"', "true"),
    (CHAIN, "the npz of an eval is not pulled", 'for x in json npz log; do pull "eval/eval_M2b_s$s-$suf.$x"', 'for x in json log; do pull "eval/eval_M2b_s$s-$suf.$x"'),
]

orig = {p: p.read_bytes() for p in {m[0] for m in MUT}}
for path, name, a, b in MUT:
    n = orig[path].count(a.encode())
    if n != 1:
        print(f"ANCHOR FOUND {n}x (need exactly 1): [{path.name}] {name}", flush=True)
        sys.exit(3)
print(f"{len(MUT)} anchors, each found exactly once", flush=True)
if CHECK:
    sys.exit(0)
BASE = Path(tempfile.gettempdir()) / "pt_kind_mut"
BASE.mkdir(exist_ok=True)
_run = [0]


def run():
    _run[0] += 1
    r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp", str(BASE / f"m{_run[0]}"), "stack/tests/test_v8f_kind.py"], cwd=ROOT,
                       capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"})
    lines = [x for x in r.stdout.strip().splitlines() if x.startswith(("FAILED", "ERROR")) or " passed" in x or " failed" in x]
    return r.returncode, " | ".join(x[:150] for x in (lines[-2:] if lines else [r.stdout.strip().splitlines()[-1]]))


rc, tail = run()
print("baseline:", rc, tail, flush=True)
assert rc == 0, "the unmutated tree must be green"
red = 0
survived = []
try:
    for path, name, a, b in MUT:
        path.write_bytes(orig[path].replace(a.encode(), b.encode()))
        rc, tail = run()
        red += rc != 0
        if rc == 0:
            survived.append(name)
        print(f"{'RED     ' if rc else 'SURVIVED'} [{path.name}] {name}: {tail}", flush=True)
        path.write_bytes(orig[path])
finally:
    for p, b in orig.items():
        p.write_bytes(b)
rc, tail = run()
print("restored:", rc, tail, "| byte-identical:", all(p.read_bytes() == b for p, b in orig.items()))
print(f"{red}/{len(MUT)} RED" + (f"  SURVIVED: {survived}" if survived else ""))
sys.exit(0 if red == len(MUT) else 1)
