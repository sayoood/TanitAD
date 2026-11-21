#!/usr/bin/env python3
"""soup_dry_run.py -- the heavy layer of soup_read.py on REAL banked dumps, with STAND-INS (PREREG_V8F_SOUP section 13, launch precondition 4; the K-RC pattern of the LAUNCH-INTENT reader's --dry-run).

NO SOUP IS READ, ROLLED OUT OR EVALUATED.  The four banked LAUNCH-INTENT runs (A, Arep, CTRL = DAG-OACC, REP; both seeds; F:/TanitAD-scratch/launch_intent_t1_out/runs.json) are the member slots, and:
    SOUP_s  := A_s      (the main pair is then A_s - DAG-OACC_s: the banked LAUNCH-INTENT 'A_CTRL' pair)
    SOUPR_s := Arep_s   (the replicate pair is then A_rep,s - A_s: NOT a banked pair; machinery only)
    SOUP4_s := A_s      (plumbing only: the four-member SG code path)
    t0_dev  := the box-2 standard T0 of CTRL and A (no dev-box standard T0 of either exists, so K-T0DEV's dev-minus-box-2 content reads 0.0 BY CONSTRUCTION here and is not tested)
What this proves is that the heavy layer (SoupHeavy, read_core, the guards, K-RC, SG, the slow-draw flag, K-TREE, the per-run pipeline controls) runs on real dumps and reproduces the banked
LAUNCH-INTENT read of record (read_A1/LAUNCH_INTENT_READ.json) with 0 differences.  It does NOT prove any soup property.  The outcome it prints is the machinery's output on stand-ins: NOT A RESULT.

    python soup_dry_run.py --out dry_run_soup_read.json [--work-dir DIR]          rc 0 = every check passed | 1 = a check failed | 2 = a pinned input is not the pinned file
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
W = os.path.dirname(HERE)
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import soup_read as R  # noqa: E402
import numpy as np  # noqa: E402

LI_OUT = "F:/TanitAD-scratch/launch_intent_t1_out"
RUNS, RUNS_SHA = f"{LI_OUT}/runs.json", "98d2cd3d3c18263396c26a12f54b5f60e7fd50e85b3b7edac6cc7eab1a4991e2"
BANKED, BANKED_SHA = f"{LI_OUT}/read_A1/LAUNCH_INTENT_READ.json", "e451e1a6ba44e1d30e97561456476dc4223cd916bd3be96163cec4b02844bf23"
ORC_DIR = "E:/Projects/TanitAD-artifacts/dagorc_rebase/read"
LI_PATH = f"{HERE}/ref/launch_intent_read.py"
SPEC, BUILDER = f"{W}/soups.json", f"{W}/soup_build.py"
COMPOSE_KEYS = ("flags", "guards", "R-CRAWL_rows", "R-MOVE_rows", "domain", "R-NEW_fires", "R-NOBASE_violations", "T0_point", "KEEP_point")


def jn(x):
    """JSON round trip (numpy scalars -> python), so a record compares with the banked file by value."""
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def standin_manifest(runs):
    arms = {r: copy.deepcopy(runs["arms"][r]) for r in ("A", "Arep", "CTRL", "REP")}
    for r in ("CTRL", "A"):
        for s in ("0", "1"):
            arms[r][s]["t0_dev"] = dict(arms[r][s]["t0"])
    arms["SOUP"] = {s: copy.deepcopy(arms["A"][s]) for s in ("0", "1")}
    arms["SOUPR"] = {s: copy.deepcopy(arms["Arep"][s]) for s in ("0", "1")}
    arms["SOUP4"] = {s: copy.deepcopy(arms["A"][s]) for s in ("0", "1")}
    return {"seeds": [0, 1], "arms": arms, "build": runs.get("build")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", default="")
    a = ap.parse_args(argv)
    t_start = time.time()
    LI = R.load_li(LI_PATH)
    ctl = LI.Ctl()
    for p, want in ((RUNS, RUNS_SHA), (BANKED, BANKED_SHA), (BUILDER, None), (SPEC, None)):
        got = R.sha256_file(p) if os.path.isfile(p) else None
        ctl.add(f"PIN:{os.path.basename(p)}", got is not None and (want is None or got == want), {"path": p, "sha256": got, "pin": want})
    if ctl.failed():
        print("SOUP DRY RUN STOPPED: a banked input is not the pinned file:", json.dumps({k: v["detail"] for k, v in ctl.failed().items()}), file=sys.stderr)
        return 2
    runs, banked = json.load(open(RUNS, encoding="utf-8")), json.load(open(BANKED, encoding="utf-8"))
    man = standin_manifest(runs)
    H = R.make_heavy(LI, ORC_DIR, SPEC, BUILDER, R.sha256_file(BUILDER), BANKED, work_dir=a.work_dir or None, standin=True)
    sp = H.static_problems()
    ctl.add("STATIC", not sp, sp or "staged readers, Read A's banked inputs, DAG-ORC's PREREG text, RF0 pin")
    pr, rf0, det = H.rf0(R.rollout_files(man))
    ctl.add("K-P0", not pr, pr or det)
    res = R.read_core(H, LI, ctl, man, rf0)
    out = {"DRY_RUN": True, "note": "banked LAUNCH-INTENT runs stand in for the soups; no soup is read, rolled out or evaluated; nothing below is a soup result",
           "standin_map": {"SOUP_s": "A_s (LAUNCH-INTENT)", "SOUPR_s": "A_rep,s", "SOUP4_s": "A_s (plumbing only)", "t0_dev (CTRL, A)": "the box-2 standard T0 (K-T0DEV content NOT tested)"},
           "pins": {"runs.json": RUNS_SHA, "LAUNCH_INTENT_READ.json": BANKED_SHA}}
    chk = {}

    # 1. the stand-in main pair IS the banked LAUNCH-INTENT pair: SOUP - CTRL = A - CTRL, SOUP - P10 = A - P10, SOUPR - P10 = Arep - P10: every key of every row, 0 differences
    cmp_t = {}
    for mine_k, bank_k in (("SOUP_CTRL", "A_CTRL"), ("SOUP_P10", "A_P10"), ("SOUPR_P10", "Arep_P10"), ("SOUP4_P10", "A_P10")):
        for s in R.SEEDS:
            d = LI.row_diffs(LI.trim(res["tabs"][mine_k][s]), banked["read_a"][bank_k][f"s{s}"])
            cmp_t[f"{mine_k} vs banked {bank_k}:s{s}"] = {"n_rows": len(banked["read_a"][bank_k][f"s{s}"]), "n_differences": len(d), "differences": d[:6]}
    out["standin_tables_vs_banked"] = cmp_t
    ctl.add("DRY:standin-tables-vs-banked", all(v["n_differences"] == 0 for v in cmp_t.values()), {k: v["n_differences"] for k, v in cmp_t.items()})

    # 2. the stand-in main pair's composition (guards, flags, rows, T0 and KEEP points) equals the banked A main pair, key by key
    cmp_c = {f"s{s}": {k: (jn(res["pairs"][s]["main"][k]) == banked["rows"]["A"][f"s{s}"]["pairs"]["main"][k]) for k in COMPOSE_KEYS} for s in R.SEEDS}
    out["main_pair_composition_vs_banked"] = cmp_c
    ctl.add("DRY:main-pair-composition", all(all(v.values()) for v in cmp_c.values()), cmp_c)

    # 3. RF, the bars, and the beside slow-draw flag of the main pair
    ctl.add("DRY:RF", jn(res["floors"]["RF"]) == banked["floors"]["RF"], "RF equals the banked floors.RF")
    ctl.add("DRY:bars", jn(res["floors"]["bar"]) == banked["floors"]["bar"], "the 19 bars equal the banked floors.bar")
    ctl.add("DRY:RF0", jn(res["floors"]["RF0"]) == banked["floors"]["RF0"], "RF0 equals the banked floors.RF0")
    sl = {f"s{s}": jn(res["slow"][f"main:SOUP-CTRL:s{s}"]) == banked["beside"]["A"]["slow_draw"][f"main:A-CTRL:s{s}"] for s in R.SEEDS}
    out["slow_draw_main_vs_banked"] = sl
    ctl.add("DRY:slow-draw-main", all(sl.values()), sl)
    ctl.add("DRY:slow-draw-no-errors", not res["slow_errors"], res["slow_errors"])

    # 4. SG: an analytic identity on real windows.  SOUPR := Arep with members (A, Arep): x - mean(A, Arep) = (Arep - A) / 2, so SG = half the paired Arep - A contrast on ADE6 (T.PB, same bootstrap).
    #    T.PB ROUNDS ITS OUTPUTS TO 4 DECIMALS (MEASURED by the first run of this check: residuals of exactly 5e-5), so a residual of up to 7.5e-5 is rounding, and the identity is read at TWO scales:
    #    (a) the heavy layer's SG as computed (wiring), against half the paired contrast, within the rounding bound 1.5e-4 (half-rounding of each side: 7.5e-5; the SG itself is ~0.2 m here, so this reads the wiring to ~1e-3 relative);
    #    (b) soup_gain itself on the same windows multiplied by 1e4 (PB is positively homogeneous in the data at a fixed seed, so every statistic scales by 1e4): the residual is then the same rounding bound
    #        against values of order 1, a relative 1e-4 test of the formula.
    RB, SCALE = 1.5e-4, 1e4
    ident, ident_scaled = {}, {}
    for s in R.SEEDS:
        K, T = H.K, H.T
        ca, ar = K.comps(H.recs["A"][s]["dump"]["cl"], H.recs["A"][s]["dump"]["g"])["ade6_m"], K.comps(H.recs["Arep"][s]["dump"]["cl"], H.recs["Arep"][s]["dump"]["g"])["ade6_m"]
        eid = H.recs["A"][s]["dump"]["eid"]
        mk = np.isfinite(ca) & np.isfinite(ar)
        full = T.PB(ar[mk], ca[mk], eid[mk], n_boot=H.n_boot, seed=0)
        sg = res["sg"][f"SOUPR:s{s}"]
        ident[f"s{s}"] = {k: abs(sg[k] - 0.5 * float(full[k])) for k in ("delta", "lo", "hi")}
        ident[f"s{s}"]["n_windows"] = sg["n_windows"]
        ident[f"s{s}"]["n_dropped"] = sg["n_dropped"]
        fk = T.PB(SCALE * ar[mk], SCALE * ca[mk], eid[mk], n_boot=H.n_boot, seed=0)
        sk = R.soup_gain(SCALE * ar, [SCALE * ca, SCALE * ar], eid, T.PB, H.n_boot)
        ident_scaled[f"s{s}"] = {k: abs(sk[k] - 0.5 * float(fk[k])) for k in ("delta", "lo", "hi")}
        ident_scaled[f"s{s}"]["magnitude_delta"] = abs(float(fk["delta"]))
        ident_scaled[f"s{s}"]["n_dropped"] = sk["n_dropped"]
    out["sg_identity_half_of_Arep_minus_A"] = {"rounding_bound": RB, "as_computed": ident, f"scaled_x{int(SCALE)}": ident_scaled}
    ctl.add("DRY:SG-identity", all(max(v["delta"], v["lo"], v["hi"]) <= RB and v["n_dropped"] == 0 for v in ident.values()), ident)
    ctl.add("DRY:SG-identity-scaled", all(max(v["delta"], v["lo"], v["hi"]) <= RB and v["n_dropped"] == 0 and v["magnitude_delta"] > 0.1 for v in ident_scaled.values()), ident_scaled)
    zeros = {f"s{s}": H.sg("CTRL", ("CTRL", "CTRL"), s) for s in R.SEEDS}
    ctl.add("DRY:SG-zero", all(v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 and not v["separated"] for v in zeros.values()), "a 'soup' of identical members: x - mean(x, x) reads exactly 0 [0, 0]")
    four = {f"s{s}": res["sg"][f"SOUP4:s{s}"] for s in R.SEEDS}
    ctl.add("DRY:SG-four-members", all(v is not None and np.isfinite([v["delta"], v["lo"], v["hi"]]).all() and v["n_dropped"] == 0 for v in four.values()), four)

    # 5. K-RC is a control that can FAIL: the same tables against a banked file with ONE thing changed.  (contrast() is memoised: this re-reads the file only.)
    import tempfile
    mut = {}
    work = tempfile.mkdtemp(prefix="soup_krc_mut_", dir=a.work_dir or None)
    cases = {"A_CTRL ADE6 delta +0.0001 (s1)": ("A_CTRL", "s1", "ADE6", "delta", lambda v: v + 0.0001),
             "REP_P10 yaw-rate separated flipped (s0)": ("REP_P10", "s0", "yaw-rate MAE", "separated", lambda v: not v),
             "CTRL_P10 mean_arm +1e-9 (s0)": ("CTRL_P10", "s0", "ADE6", "mean_arm", lambda v: v + 1e-9)}
    for name, (tab, sd, lab, key, fn) in cases.items():
        b2 = copy.deepcopy(banked)
        b2["read_a"][tab][sd][lab][key] = fn(b2["read_a"][tab][sd][lab][key])
        pth = os.path.join(work, f"banked_mut_{len(mut)}.json")
        with open(pth, "w", encoding="utf-8", newline="\n") as f:
            json.dump(b2, f)
        c2, keep = LI.Ctl(), H.banked_path
        H.banked_path = pth
        try:
            H.k_rc(c2)
        finally:
            H.banked_path = keep
        failed = sorted(k for k, v in c2.items.items() if not v["PASS"])
        mut[name] = {"failed_controls": failed, "as_required": failed == [f"K-RC:{tab}"]}
    b3 = copy.deepcopy(banked)
    b3["floors"]["RF"]["ADE6"] = round(b3["floors"]["RF"]["ADE6"] + 0.0001, 4)
    pth = os.path.join(work, "banked_mut_rf.json")
    with open(pth, "w", encoding="utf-8", newline="\n") as f:
        json.dump(b3, f)
    c3, keep = LI.Ctl(), H.banked_path
    H.banked_path = pth
    try:
        H.k_rc(c3)
    finally:
        H.banked_path = keep
    f3 = sorted(k for k, v in c3.items.items() if not v["PASS"])
    mut["RF ADE6 +0.0001"] = {"failed_controls": f3, "as_required": f3 == ["K-RC:RF"]}
    out["k_rc_deliberate_regression"] = mut
    ctl.add("DRY:K-RC-fails-when-it-should", all(v["as_required"] for v in mut.values()), mut)

    # 6. the controls: every one the real read adds for a soup slot ran on the stand-ins (names listed), the ones that cannot apply are named
    names = sorted(ctl.items)
    must = [f"K-ZERO:{r}-{r}:s{s}" for r in ("SOUP", "SOUPR", "SOUP4", "CTRL", "A") for s in R.SEEDS] + [f"K-RC:{k}" for k in list(R.K_RC_TABLES) + ["RF"]]
    for tag in ("SOUP_s0", "SOUP_s1", "SOUPR_s0", "SOUPR_s1", "SOUP4_s0", "SOUP4_s1"):
        must += [f"{c}:{tag}" for c in ("T1-CONFIG", "T1-CONFIG-logged", "K4", "T0F-CKPT", "T0-CKPT", "DUMP-WINDOWS", "GRID-COUNTS", "R2-ADE6", "K-TREE")]
    for x in ("SOUP", "SOUPR", "SOUP4"):
        for s in R.SEEDS:
            must += [f"C4-WINDOWS:{x}-vs-CTRL:s{s}", f"C3-FLOOR:{x}-vs-CTRL:s{s}"]
    for r in ("CTRL", "A"):
        for s in R.SEEDS:
            must += [f"T0DEV-CKPT:{r}_s{s}", f"K-T0DEV-BESIDE:{r}_s{s}"]
    for pair in (("SOUP", "CTRL"), ("SOUPR", "A")):
        for s in R.SEEDS:
            must += [f"ADE-LINK:{pair[0]}-{pair[1]}:s{s}", f"T0-WINDOWS:{pair[0]}-{pair[1]}:s{s}"]
    missing = [c for c in must if c not in ctl.items]
    ctl.add("DRY:controls-all-ran", not missing, {"missing": missing, "n_required": len(must)})
    out["controls_not_applicable_to_standins"] = {
        "K-SOUP-REC / K-MID / K-BASIN (x6 soup slots)": "read a soup's own files (soup.json, ckpt, member checkpoints); a banked LAUNCH-INTENT run is not a soup",
        "K-XINIT:s0, K-XINIT:s1, K-ZERO-W": "weights-only builder controls on the XINIT soups and a member; no soup file is read in a stand-in run",
        "K-T0DEV (content)": "no dev-box standard T0 of CTRL / A exists: t0_dev is the box-2 T0, so dev-minus-box-2 reads 0.0 by construction; the plumbing (T0DEV-CKPT, the swap in guard_rows / speed) ran",
        "the soup's own rollouts": "the stand-in rollouts are LAUNCH-INTENT's: K-TREE read their done records against the A checkpoint, which is the stand-in's own binding",
        "listed by the heavy layer (H.na)": H.na}
    out["controls_PASS"] = ctl.flags()
    out["n_controls"], out["failed_controls"] = len(ctl.items), {k: str(v["detail"])[:400] for k, v in ctl.failed().items()}
    d = res["decision"]
    out["standin_machinery_outcome"] = {"NOTE": "NOT A SOUP RESULT: LAUNCH-INTENT's A and A_rep on stand-in pairs", "OUTCOME": d["OUTCOME"], "per_seed": d["per_seed"], "route": res["route"]}
    out["wall_s"] = round(time.time() - t_start, 1)
    out["reader_sha256"] = R.sha256_file(os.path.join(HERE, "soup_read.py"))
    out["li_reader_sha256"] = R.sha256_file(LI_PATH)
    ok = not ctl.failed()
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("ZZSOUP-DRY", "OK" if ok else "NOT-OK", json.dumps({"n_controls": out["n_controls"], "failed": out["failed_controls"], "wall_s": out["wall_s"]}, default=str)[:1500], flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
