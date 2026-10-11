#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the refcv8 LAUNCH argv from the profile-passing smoke argv + the ladder's readings.

    refcv8_assemble_argv.py --base <refcv8-wpb-smoke.argv.json> --code <CODE tree> --run-out <dir> \
        --recipe-json F --k-tac K --w-map-hires W --b1-ref-ap2m V \
        (--stage measure | --gs-measure <gs_measure.json>) --out-json <argv.json>
    refcv8_assemble_argv.py --audit <argv.json> [--gs-measure <gs_measure.json>]
    refcv8_assemble_argv.py --self-test

Two stages (SPEC_REFCV8 10.3, ruled 2026-10-11): `--stage measure` writes the 100-step G-LIVE MEASUREMENT argv
(`--grad-share-every 50 --grad-share-at 50`: one full-size gradient-share reading inside a 100-step smoke); the
smoke's `gs_measure.json` ({"reading_s", "s_per_step"}) then picks the launch `--grad-share-at` by the rule in
`pick_gs_schedule`. There is NO escape for an unmeasured reading. `--r8-speed-enc8` / `--r8-critic-drivable` are OUT
of refcv8 (PI 2026-10-11: the ladder stops after 11 runs, L4b / L5 are not run) and are refused by name.

Every override is traced to its SPEC clause in assemble(). The result is checked against LITERALS
authored here from SPEC_REFCV8 3.2 / 10.3 (never derived from the base file) and against the flags
the launch gate refuses by name. Exit 0 = argv written; 2 = refused. Stdlib only; stdout is LF.
"""
from __future__ import annotations

import argparse
import json
import sys

#: R8-RECIPE (SPEC_REFCV8 10.3): the only flags L1's reading may set
RECIPE_FLAGS = ("--r8-n-alloc", "--r8-alloc-emit", "--w-r8-alloc-l1", "--w-r8-listwise", "--w-r8-sat",
                "--w-r8-subscore", "--w-r8-v9-cons", "--r8-lat-prior-dropout", "--r8-prior-free-group")
#: refused by NAME (launch_gate.py PROFILES['refcv8'] forbidden_levers)
FORBIDDEN = ("--r8-derange-feed", "--r8-rc-roll", "--r8-roll-targets", "--r8-roll-speed-input",
             "--speed-max-sidecar-v6", "--speed-max-sidecar-v6-eval",
             # PI 2026-10-11 ("You can stop the ladder after 11 runs"): L4b / L5 are not run, their entry rule cannot be met
             "--r8-speed-enc8", "--r8-critic-drivable", "--w-r8-drivable")
#: literal expectations (SPEC_REFCV8 3.2 'required values' + R8-WARMUP), written out, not computed
LITERALS = {"--r8-v9-md5": "f63ece410b725febb8a5242cf2b01d3c",
            "--r8-v9-eval-md5": "6b5c7f207cffc3b7eb3cd527fd433599",
            "--r8-v9-lat-variant": "a", "--r8-rc-noise-along-m": "2.0", "--r8-rc-noise-lat-m": "0.75",
            "--r8-rc-dropout": "0.3", "--r8-nav-args-dropout": "0.5", "--r8-speed-unknown-p": "0.45",
            "--r8-alloc-emit-start": "1000", "--w-r8-cons": "0.05", "--lr": "1e-4", "--warmup": "1000",
            "--opt": "dd", "--r8-speed-input": "n2", "--eval-every": "500", "--log-every": "50",
            # R8-WARMUP CLOSED (MM 2026-10-11, SPEC 10.3): EMA + the B1 guard step + the run length + the gradient-share terms
            "--ema-decay": "0.9998", "--b1-guard-step": "1000", "--steps": "30000",
            "--grad-share-terms": "tac_v6,map_hires", "--grad-share-every": "250"}
#: the launch `--grad-share-at` rows: the long schedule, or the short one when the MEASURED reading is too dear
GS_AT_LONG = "1000,10000,20000,30000"
GS_AT_SHORT = "1000,30000"
GS_BUDGET_FRAC = 0.03       # 4 readings may cost at most 3 % of the projected run wall-clock
GS_N_LONG = 4
REQUIRED_FLAGS = ("--refcv8", "--init-from", "--r8-v9-labels", "--r8-v9-labels-eval", "--r8-nav-from-v9",
                  "--r8-rc-variant", "--join-defect-masks", "--grad-share-every", "--max-speed-input-v6",
                  "--r8-speed-input", "--pose-sync-sidecar", "--ema-decay", "--b1-guard-ref-ap2m",
                  "--b1-guard-step", "--grad-share-terms", "--grad-share-at")
#: model-only is NOT on Thor; ckpt.pt is the full checkpoint, registry md5 d5f104ee54ba6b2861e38030b4f6fcf1
INIT_FROM = "/home/nvidia/refcv7_run/runs/refcv7-r101-s0/ckpt.pt"


def pick_gs_schedule(reading_s, s_per_step, steps=30000):
    """SPEC 10.3 (ruled 2026-10-11): {1000, 10000, 20000, 30000} iff 4 x the MEASURED full-size reading cost is <= 3 % of
    the projected run wall-clock (steps x projected s/step); otherwise {1000, 30000}."""
    if not (reading_s and reading_s > 0 and s_per_step and s_per_step > 0):
        raise SystemExit("REFUSED: gs_measure.json must carry a POSITIVE reading_s and s_per_step (an unmeasured "
                         "reading has no escape; run the G-LIVE measurement smoke)")
    return GS_AT_LONG if GS_N_LONG * float(reading_s) <= GS_BUDGET_FRAC * steps * float(s_per_step) else GS_AT_SHORT


def pairs(a):
    out, i = [], 0
    while i < len(a):
        j = i + 1
        if a[i].startswith("--"):
            while j < len(a) and not a[j].startswith("--"):
                j += 1
        out.append((a[i], list(a[i + 1:j])))
        i = j
    return out


def flat(ps):
    return [t for f, v in ps for t in [f, *v]]


def set_flag(ps, flag, vals):
    """replace (first hit, drop duplicates) or append; vals None = remove."""
    out, done = [], False
    for f, v in ps:
        if f == flag:
            if not done and vals is not None:
                out.append((f, [str(x) for x in vals]))
            done = True
        else:
            out.append((f, v))
    if not done and vals is not None:
        out.append((flag, [str(x) for x in vals]))
    return out


def get(ps, flag):
    v = [v for f, v in ps if f == flag]
    return v[-1] if v else None


def assemble(base, *, code, run_out, recipe, k_tac, w_map, b1_ref, gs_at, stage="launch", steps=30000,
             warmup=1000, init_from=INIT_FROM):
    ps = pairs(base)
    why = []

    def put(flag, vals, clause):
        nonlocal ps
        ps = set_flag(ps, flag, vals)
        why.append(f"{flag} {'' if vals is None else ' '.join(map(str, vals))}  <- {clause}")

    put("--out", [run_out], "run dir (new; never the smoke dir)")
    put("--steps", [steps], "SPEC_REFCV8 10.3: --steps 30000 binds (also the cosine horizon)")
    put("--init-from", [init_from], "SPEC 3.1: warm start from refcv7-50,400")
    put("--warmup", [warmup], "R8-WARMUP: re-warm over 1,000 steps (SPEC 10.3)")
    put("--lr", ["1e-4"], "R8-WARMUP: head peak 1e-4 (trunk 5e-5 = --encoder-lr-mult default 0.5)")
    put("--w-tac-v6", [repr(round(1.0 * k_tac, 6))], "R8-BUDGET: 1.0 x k (ladder L2, SPEC_WPB_LADDER 4.2)")
    put("--w-map-hires", [repr(float(w_map))], "R8-BUDGET: 4.0 iff B-MAPW else 1.0 (ladder L3)")
    put("--ema-decay", ["0.9998"], "R8-WARMUP CLOSED (MM 2026-10-11): EMA evaluated at every in-run eval and saved")
    put("--b1-guard-ref-ap2m", [repr(float(b1_ref))], "SPEC 10.1 B1 guard: the refcv7-50,400 EMA box3d AP@2 m (cite artifact + md5)")
    put("--b1-guard-step", ["1000"], "SPEC 10.1 B1 guard step")
    put("--r8-alloc-emit-start", ["1000"], "S_emit = the warmup length (MM 2026-10-11)")
    put("--grad-share-terms", ["tac_v6,map_hires"], "SPEC 10.3 gradient share on the main run")
    if stage == "measure":
        put("--steps", [100], "G-LIVE MEASUREMENT smoke: 100 steps (SPEC 3.2 item 5)")
        put("--grad-share-every", [50], "measurement smoke: one reading inside the smoke")
        put("--grad-share-at", [50], "measurement smoke: the reading at row 50; wall-clock recorded")
    else:
        put("--grad-share-every", [250], "SPEC 10.3")
        put("--grad-share-at", [gs_at], "SPEC 10.3: picked by pick_gs_schedule from the MEASURED reading")
    put("--join-defect-masks", [f"{code}/stack/tanitad/configs/refcv8_join_label_defects.json"],
        "MM 2026-10-04 item 6; path under the GATED tree, not the smoke tree")
    for f, v in recipe.items():
        if f not in RECIPE_FLAGS:
            raise SystemExit(f"REFUSED: recipe flag {f} is not an R8-RECIPE flag {RECIPE_FLAGS}")
        put(f, [] if v is True else None if v is False else [v], "R8-RECIPE: ladder L1 / SPEC_WPB reading")
    return ps, why


#: SPEC_REFCV8 sec. 10.1 (MM ruling 2026-10-11): min of refcv7-r101-s0's in-run box3d_ap2m over its last 22 eval rows,
#: from inrun_curve.json md5 74436ec4f705c22cc7f6f4cc1ea6a024. A literal: the launch argv must carry exactly this.
B1_REF_REGISTERED = "0.21211"


def audit(ps, gs_measure=None, stage="launch"):
    errs = []
    names = [f for f, _ in ps]
    try:
        if float((get(ps, "--b1-guard-ref-ap2m") or ["0"])[0]) <= 0:
            errs.append("--b1-guard-ref-ap2m must be a positive float (the refcv7-50,400 reference value)")
        elif stage == "launch" and float((get(ps, "--b1-guard-ref-ap2m") or ["0"])[0]) != float(B1_REF_REGISTERED):
            errs.append(f"--b1-guard-ref-ap2m must be the registered {B1_REF_REGISTERED} (SPEC 10.1), not a re-derived value")
    except ValueError:
        errs.append("--b1-guard-ref-ap2m not numeric")
    if stage == "measure":
        for f, want in (("--steps", "100"), ("--grad-share-every", "50"), ("--grad-share-at", "50")):
            if (get(ps, f) or [None])[0] != want:
                errs.append(f"measurement argv: {f} must be {want}")
    else:
        if gs_measure is None:
            errs.append("no gs_measure.json: the launch --grad-share-at cannot be checked against the measured reading")
        else:
            try:
                want = pick_gs_schedule(gs_measure.get("reading_s"), gs_measure.get("s_per_step"))
                if (get(ps, "--grad-share-at") or [None])[0] != want:
                    errs.append(f"--grad-share-at must be {want} for the measured reading, got {get(ps, '--grad-share-at')}")
            except SystemExit as e:
                errs.append(str(e))
    for f in FORBIDDEN:
        if f in names:
            errs.append(f"forbidden lever {f} present")
    for f in REQUIRED_FLAGS:
        if f not in names:
            errs.append(f"required flag {f} missing")
    for f, want in LITERALS.items():
        if stage == "measure" and f in ("--steps", "--grad-share-every"):
            continue
        got = get(ps, f)
        if got is None or got[:1] != [want]:
            errs.append(f"{f} must be {want!r}, got {got}")
    if (get(ps, "--r8-n-alloc") or ["0"])[0] != "0" and (get(ps, "--w-r8-alloc-l1") or ["0"])[0] in ("0", "0.0"):
        errs.append("--r8-n-alloc > 0 needs --w-r8-alloc-l1 > 0 (SPEC 3.2 item 1)")
    for f in ("--w-r8-sat", "--w-r8-listwise", "--w-r8-subscore"):
        if f not in names:
            errs.append(f"{f} must be STATED in the argv even when 0 (SPEC 3.2 item 1)")
    if "--r8-alloc-emit" in names and "--r8-alloc-emit-start" not in names:
        errs.append("emission needs --r8-alloc-emit-start (MM ruling Q1)")
    for f in ("--w-tac-v6", "--w-map-hires"):
        try:
            if float((get(ps, f) or ["0"])[0]) <= 0:
                errs.append(f"{f} must be > 0")
        except ValueError:
            errs.append(f"{f} not numeric")
    return errs


def self_test():
    base = ["--arm", "hier", "--opt", "dd", "--lr", "1e-4", "--warmup", "2000", "--eval-every", "500",
            "--log-every", "50", "--steps", "50400", "--out", "/x/smoke", "--w-tac-v6", "1.0",
            "--w-map-hires", "1.0", "--refcv8", "--init-from", "/old", "--r8-v9-labels", "a",
            "--r8-v9-labels-eval", "b", "--r8-nav-from-v9", "--r8-rc-variant", "A50",
            "--join-defect-masks", "/old/j.json", "--grad-share-every", "50", "--max-speed-input-v6",
            "--r8-speed-input", "n2", "--pose-sync-sidecar", "p", "--w-r8-cons", "0.05",
            "--r8-v9-md5", LITERALS["--r8-v9-md5"], "--r8-v9-eval-md5", LITERALS["--r8-v9-eval-md5"],
            "--r8-v9-lat-variant", "a", "--r8-rc-noise-along-m", "2.0", "--r8-rc-noise-lat-m", "0.75",
            "--r8-rc-dropout", "0.3", "--r8-nav-args-dropout", "0.5", "--r8-speed-unknown-p", "0.45",
            "--r8-alloc-emit-start", "2000", "--w-r8-sat", "0.1", "--w-r8-listwise", "1.0",
            "--w-r8-subscore", "0.5", "--r8-n-alloc", "32", "--r8-alloc-emit", "--w-r8-alloc-l1", "1.0"]
    kw = dict(code="/C", run_out="/R", recipe={"--w-r8-sat": "0.1"}, k_tac=51, w_map=4.0, b1_ref=0.21211,
              gs_at=GS_AT_LONG)
    GSM = {"reading_s": 800.0, "s_per_step": 10.0}
    # analytic boundary of the schedule rule, written as LITERALS: 4 x 2250 = 9000 = 0.03 x 30000 x 10 -> long; 2251 -> short
    assert pick_gs_schedule(2250.0, 10.0) == "1000,10000,20000,30000"
    assert pick_gs_schedule(2251.0, 10.0) == "1000,30000"
    assert pick_gs_schedule(788.0, 10.5) == "1000,10000,20000,30000"      # the tiny-rig 788 s reading would qualify
    assert pick_gs_schedule(2500.0, 10.0) == "1000,30000"
    for bad_m in ({}, {"reading_s": 0, "s_per_step": 10}, {"reading_s": 5, "s_per_step": None}):
        try:
            pick_gs_schedule(bad_m.get("reading_s"), bad_m.get("s_per_step"))
            raise AssertionError("an unmeasured reading was accepted")
        except SystemExit:
            pass
    ps, _ = assemble(base, **kw)
    assert not audit(ps, GSM), audit(ps, GSM)
    pm, _ = assemble(base, **{**kw, "gs_at": None}, stage="measure")
    assert not audit(pm, None, "measure"), audit(pm, None, "measure")
    assert get(pm, "--steps") == ["100"] and get(pm, "--grad-share-every") == ["50"] and get(pm, "--grad-share-at") == ["50"]
    # LITERAL expectations
    assert get(ps, "--w-tac-v6") == ["51.0"], get(ps, "--w-tac-v6")
    assert get(ps, "--w-map-hires") == ["4.0"]
    assert get(ps, "--warmup") == ["1000"] and get(ps, "--steps") == ["30000"] and get(ps, "--out") == ["/R"]
    assert get(ps, "--ema-decay") == ["0.9998"] and get(ps, "--b1-guard-step") == ["1000"]
    assert get(ps, "--r8-alloc-emit-start") == ["1000"] and get(ps, "--grad-share-every") == ["250"]
    assert get(ps, "--grad-share-at") == ["1000,10000,20000,30000"] and get(ps, "--b1-guard-ref-ap2m") == ["0.21211"]
    assert get(ps, "--join-defect-masks") == ["/C/stack/tanitad/configs/refcv8_join_label_defects.json"]
    # deliberate regressions: each must be REFUSED
    bads = (ps + [("--speed-max-sidecar-v6", ["s"])], set_flag(ps, "--w-r8-cons", ["0"]),
            set_flag(ps, "--r8-speed-input", None), set_flag(ps, "--w-r8-alloc-l1", ["0"]),
            set_flag(ps, "--warmup", ["2000"]), set_flag(ps, "--r8-alloc-emit-start", ["0"]),
            set_flag(ps, "--r8-alloc-emit-start", ["2000"]),                       # the pre-2026-10-11 S_emit
            ps + [("--r8-speed-enc8", [])], ps + [("--r8-critic-drivable", [])],   # OUT by the PI's 11-run ruling
            set_flag(ps, "--ema-decay", None), set_flag(ps, "--b1-guard-ref-ap2m", None),
            set_flag(ps, "--b1-guard-ref-ap2m", ["0"]), set_flag(ps, "--b1-guard-step", ["500"]),
            set_flag(ps, "--steps", ["12000"]), set_flag(ps, "--grad-share-every", ["2000"]),
            set_flag(ps, "--grad-share-at", [GS_AT_SHORT]),                         # wrong for GSM (long qualifies)
            set_flag(ps, "--grad-share-terms", ["all"]))
    for i, bad in enumerate(bads):
        assert audit(bad, GSM), f"deliberate regression {i} passed the audit"
    assert audit(ps, None), "a launch argv with no gs_measure.json was accepted (the unmeasured escape is gone)"
    try:
        assemble(base, **{**kw, "recipe": {"--lr": "9"}})
        raise AssertionError("a non-recipe flag was accepted as recipe")
    except SystemExit:
        pass
    # helpers: a metrics file with a reading at row 50 and rows to 100; and the refusals
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "metrics.jsonl")
        rows = [{"step": 0, "elapsed_s": 5.0}, {"step": 50, "elapsed_s": 905.0, "gs_fp32_replay_s": 788.0, "gs_measure_s": 560.0},
                {"step": 100, "elapsed_s": 955.0}]
        open(mp, "w").write("\n".join(json.dumps(r) for r in rows))
        m = read_gs_measure(mp)
        assert m["reading_s"] == 788.0 and abs(m["s_per_step"] - 1.0) < 1e-9, m          # (955 - 905) / 50 = 1.0, excl. the reading
        open(mp, "w").write("\n".join(json.dumps(r) for r in (rows[0], rows[2])))
        try:
            read_gs_measure(mp)
            raise AssertionError("a smoke with no reading was accepted")
        except SystemExit:
            pass
        cp = os.path.join(td, "c.json")
        json.dump({"binding": True, "exit_status": 0, "closure_sha256": "a" * 64, "outside_tree": []}, open(cp, "w"))
        assert assert_closure(cp) == []
        json.dump({"binding": False, "exit_status": 1, "closure_sha256": "a" * 64, "outside_tree": ["x"]}, open(cp, "w"))
        assert len(assert_closure(cp)) == 3
        rp = os.path.join(td, "r.json")
        json.dump({"verdict": "PASS"}, open(rp, "w"))
        assert assert_record(rp, "map") == []
        json.dump({"RESULT": "FAIL", "binding": True}, open(rp, "w"))                      # refcv7's own box record read FAIL
        assert len(assert_record(rp, "box")) == 1
        try:
            render_template("--x\n{UNSET}\n", {})
            raise AssertionError("an unreplaced placeholder was accepted")
        except SystemExit:
            pass
    print(f"SELFTEST-OK: 1 positive, 1 measurement argv, {len(bads)} regressions refused, 4 schedule boundaries, 1 non-recipe flag refused")


# ------------------------------------------------------------------------------------------------------------------ #
# helpers the shell kit calls (stdlib only). Every verdict is a VALUE read from an ARTIFACT, never a status code.       #
# ------------------------------------------------------------------------------------------------------------------ #
def render_template(text, subs):
    """One token per line; `#` lines are comments; `DATA-ROOT <dir>` lines go to the closure wrapper. `{K}` is
    replaced by subs[K]; an unreplaced `{...}` is REFUSED."""
    args, roots = [], []
    for line in text.split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        for k, v in subs.items():
            line = line.replace("{" + k + "}", v)
        if "{" in line and "}" in line:
            raise SystemExit(f"REFUSED: unreplaced placeholder in template line {line!r}")
        if line.startswith("DATA-ROOT "):
            roots.append(line[len("DATA-ROOT "):].strip())
        else:
            args.append(line)
    return args, roots


def read_gs_measure(metrics_path):
    """-> {reading_s, s_per_step, gs_row_step, last_step, n_rows}. reading_s = the largest `gs_fp32_replay_s` (else
    `gs_measure_s`) of any row; s_per_step = the interval AFTER the reading row (so it excludes the reading), read from the
    trainer's own cumulative `elapsed_s`. Refuses (SystemExit) when the smoke never reached a reading or step 100."""
    rows = []
    with open(metrics_path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except ValueError:
                    pass
    g = [r for r in rows if any(k in r for k in ("gs_fp32_replay_s", "gs_measure_s"))]
    if not g:
        raise SystemExit("REFUSED: the measurement smoke never reached a gradient-share reading (no gs_*_s key in any row)")
    reading_s = max(float(r.get("gs_fp32_replay_s", r.get("gs_measure_s", 0.0))) for r in g)
    gstep = min(int(r["step"]) for r in g)
    after = [r for r in rows if "elapsed_s" in r and "step" in r and int(r["step"]) > gstep]
    if not after:
        raise SystemExit("REFUSED: no row after the reading row: s/step cannot be read (the smoke must run past the reading)")
    last = max(after, key=lambda r: int(r["step"]))
    base = [r for r in rows if "elapsed_s" in r and int(r.get("step", -1)) == gstep]
    if not base:
        raise SystemExit("REFUSED: the reading row carries no elapsed_s")
    d_steps = int(last["step"]) - gstep
    sps = (float(last["elapsed_s"]) - float(base[0]["elapsed_s"])) / d_steps
    if not (reading_s > 0 and sps > 0 and d_steps > 0):
        raise SystemExit("REFUSED: non-positive reading_s / s_per_step")
    return {"reading_s": reading_s, "s_per_step": sps, "gs_row_step": gstep, "last_step": int(last["step"]),
            "n_rows": len(rows), "source": str(metrics_path)}


def assert_closure(path):
    """The closure JSON of closure_run.py --binding: binding true, exit_status 0, a closure sha, no outside_tree module."""
    d = json.load(open(path, encoding="utf-8"))
    errs = []
    if d.get("binding") is not True:
        errs.append("closure.binding is not true")
    if d.get("exit_status") != 0:
        errs.append(f"closure.exit_status is {d.get('exit_status')!r}, not 0")
    if len(str(d.get("closure_sha256", ""))) != 64:
        errs.append("closure.closure_sha256 is not a 64-char digest")
    if d.get("outside_tree"):
        errs.append(f"closure.outside_tree is non-empty ({len(d['outside_tree'])} modules)")
    return errs


def assert_record(path, kind):
    """The harness record: map = top-level `verdict` PASS (map_hires_overfit.py); box = `RESULT` PASS (g_box_overfit.py).
    MAIN arm only (SPEC_REFCV7 A19): the box record must list exactly the `main` arm."""
    d = json.load(open(path, encoding="utf-8"))
    errs = []
    if kind == "map":
        if d.get("verdict") != "PASS":
            errs.append(f"map record verdict is {d.get('verdict')!r}, not PASS")
    elif kind == "box":
        if d.get("RESULT") != "PASS":
            errs.append(f"box record RESULT is {d.get('RESULT')!r}, not PASS")
        if d.get("binding") is not True:
            errs.append("box record is not a --binding record")
    else:
        errs.append(f"unknown kind {kind!r}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--base")
    ap.add_argument("--code")
    ap.add_argument("--run-out")
    ap.add_argument("--stage", choices=("launch", "measure"), default="launch")
    ap.add_argument("--b1-ref-ap2m", type=float)
    ap.add_argument("--gs-measure", help="gs_measure.json from the G-LIVE measurement smoke: {reading_s, s_per_step}")
    ap.add_argument("--recipe-json")
    ap.add_argument("--k-tac", type=float)
    ap.add_argument("--w-map-hires", type=float)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--out-json")
    ap.add_argument("--audit", help="audit an ALREADY-assembled argv json (the one the gate will bind)")
    ap.add_argument("--render-template")
    ap.add_argument("--sub", action="append", default=[])
    ap.add_argument("--args-out")
    ap.add_argument("--data-roots-out")
    ap.add_argument("--read-gs-measure", help="metrics.jsonl of the measurement smoke -> --out-json gs_measure.json")
    ap.add_argument("--assert-closure")
    ap.add_argument("--assert-record")
    ap.add_argument("--kind", choices=("map", "box"))
    a = ap.parse_args()
    sys.stdout.reconfigure(newline="\n")
    if a.render_template:
        subs = dict(s.split("=", 1) for s in a.sub)
        args, roots = render_template(open(a.render_template, encoding="utf-8").read(), subs)
        open(a.args_out, "w", encoding="utf-8", newline="\n").write("\n".join(args) + "\n")
        open(a.data_roots_out, "w", encoding="utf-8", newline="\n").write("\n".join(roots) + ("\n" if roots else ""))
        print(f"RENDERED {len(args)} args, {len(roots)} data roots")
        return 0
    if a.read_gs_measure:
        m = read_gs_measure(a.read_gs_measure)
        open(a.out_json, "w", encoding="utf-8", newline="\n").write(json.dumps(m, indent=1))
        print(f"GSMEASURE reading_s={m['reading_s']:.1f} s_per_step={m['s_per_step']:.3f}")
        return 0
    if a.assert_closure or a.assert_record:
        errs = assert_closure(a.assert_closure) if a.assert_closure else assert_record(a.assert_record, a.kind)
        print("\n".join("REFUSED " + e for e in errs) if errs else "ASSERT-OK")
        return 2 if errs else 0
    if a.self_test:
        self_test()
        return 0
    if a.audit:
        doc = json.load(open(a.audit, encoding="utf-8"))
        gsm = json.load(open(a.gs_measure, encoding="utf-8")) if a.gs_measure else None
        errs = audit(pairs(doc["argv"] if isinstance(doc, dict) else doc), gsm, a.stage)
        print("\n".join("REFUSED " + e for e in errs) if errs else "AUDIT-OK")
        return 2 if errs else 0
    for k in ("base", "code", "run_out", "recipe_json", "k_tac", "w_map_hires", "out_json", "b1_ref_ap2m"):
        if getattr(a, k) is None:
            raise SystemExit(f"REFUSED: --{k.replace('_', '-')} is required")
    doc = json.load(open(a.base, encoding="utf-8"))
    base = doc["argv"] if isinstance(doc, dict) else doc
    recipe = json.load(open(a.recipe_json, encoding="utf-8"))
    gsm = json.load(open(a.gs_measure, encoding="utf-8")) if a.gs_measure else None
    if a.stage == "launch" and gsm is None:
        raise SystemExit("REFUSED: --gs-measure is required for the launch stage (no unmeasured-reading escape)")
    gs_at = None if a.stage == "measure" else pick_gs_schedule(gsm.get("reading_s"), gsm.get("s_per_step"))
    ps, why = assemble(base, code=a.code, run_out=a.run_out, recipe=recipe, k_tac=a.k_tac,
                       w_map=a.w_map_hires, b1_ref=a.b1_ref_ap2m, gs_at=gs_at, stage=a.stage, warmup=a.warmup)
    errs = audit(ps, gsm, a.stage)
    print("\n".join("SET " + w for w in why))
    if errs:
        print("\n".join("REFUSED " + e for e in errs))
        return 2
    out = dict(doc, argv=flat(ps)) if isinstance(doc, dict) else flat(ps)
    with open(a.out_json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    print(f"ASSEMBLED {len(flat(ps))} tokens -> {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
