"""ds_read_x.py -- the DSCALE10K X_s0 read (the 10,000 data-scale clips alone): "do the 10k clips by themselves teach a worse lateral tactical decision (CONTENT), or does the cost appear only in the mixture?"
RULES: F:/TanitAD-scratch/datascale/blockx/MM_RULINGS_X_READ.md (sha256 e6e3f24cc79fb65098af3e6b166ebb9375d2c65ca939e5c4e937e989a58aecca, written before any X_s0 evaluation number existed).  The file is checked by sha256 on every run and is never edited.
CPU, zero training.  Writes RESULT_DSCALE10K_X.json + .md.  ONE SEED (seed 0): every label carries "(one seed)".
COMMITTED RULES (a transcription of R1-R4; the ruling file is the authority)
  R1  Deciding cell TAC-LAT, T1-kin (the block-1 / 3 instrument, unchanged).  Pairs X_s0 - B_s0 (deciding) and X_s0 - U_s0 (beside, no bar), at stage A (matched to block 3) and stage B (the recipe of record).
      Estimator: paired episode-cluster bootstrap over the 141 eval episodes, B 2,000, seed 0 ("another draw of EPISODES").  F_cell = |TAC-LAT(B_rep) - TAC-LAT(B_s0)| at the same stage
      (stage A 0.0767 from block 3; stage B recomputed here from the banked block-1 stage-B arms).  One seed, one-pair floor.
  R2  Label (higher TAC-LAT is better; Delta = X - B_s0 at stage A decides):
        X-CONTENT  : separated AND Delta <= -F_cell         X-NO-COST : not separated OR |Delta| < F_cell         X-GAIN : separated AND Delta >= +F_cell
      A stage-B label that differs from stage A's is printed STAGE-SENSITIVE with both labels; the stage-A label still names the next lever.
  R3  Every pair, both stages: LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC in full, never pooled; F_cell from B_rep; fog unscored (PI ruling 2026-11-17).
  R4  Controls that must read known values: K-VLM-ZERO (X's VLM-class macro-F1 at stage A <= B_s0's minus the block-1 floor, OR <= 0.05; else the mask did not hold), K1 (X - X reads 0.0 [0, 0]),
      K-RE (B_s0, U_s0, B_rep TAC-LAT at both stages reproduce the banked block-1 / 3 values exactly), K-EVALX on X at both stages, C-JOIN-EQ quoted from the chain's gate record.
  Outcome precedence (as block 3): BLOCKED > NOT-YET > label; a failed control marks the read "CONTROL FAILED, not quotable".
TIERS: LONGITUDINAL / LATERAL / TACTICAL = T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene); STRATEGIC = T0 at the window's first tick; goal-class F1 = T1-kin tick 0 (a tactical-head diagnostic).
Usage:  python ds_read_x.py --out RESULT_DSCALE10K_X.json [--n-boot 2000]"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import ds_lib as L  # noqa: E402
import ds_read as R  # noqa: E402  (block-1 module, unedited: pure logic, loaders, K-RE helpers)
import ds_read_b2 as B2  # noqa: E402  (block-2 module, unedited: K-RE helpers, md_cell, fam_cells)

OR, FAMILIES, HEADLINE, FAMILY = R.OR, R.FAMILIES, R.HEADLINE, R.FAMILY
CELL = "TAC-LAT"
STAGES = ("A", "B")
RULINGS = "F:/TanitAD-scratch/datascale/blockx/MM_RULINGS_X_READ.md"
RULINGS_SHA256 = "e6e3f24cc79fb65098af3e6b166ebb9375d2c65ca939e5c4e937e989a58aecca"
DIAG = "F:/TanitAD-scratch/datascale/diag"
VLM_ABS_BAR = 0.05                                       # R4: "OR at most 0.05"
LABELS = ("X-CONTENT", "X-NO-COST", "X-GAIN")
PAIRS = {"X-B_s0": ("X_s0", "B_s0"), "X-U_s0": ("X_s0", "U_s0")}       # name -> (arm, reference); the first is deciding, the second beside (no bar)
NEXT = {"X-CONTENT": "Zero-training audit of the 10k clips' lateral labels and inputs against the corpus clips: compare their class shares, label builder fields and nav alignment.  UM_s0 (rebalancing) would not be expected to remove a content defect: the MM tells the PI this before UM_s0 runs.",
        "X-NO-COST": "The 10k clips alone teach TAC-LAT as well as the corpus.  The U cost appears only in the mixture, so the mixture arm UM_s0 is the discriminating lever (it waits for the PI's go).",
        "X-GAIN": "As X-NO-COST, but stronger: the cost is a mixture effect.  UM_s0 first."}
CAVEAT = ("R5: ONE seed; one-pair floor.  X has 10,000 clips against B's 4,512 at the same steps, so X sees each window about 1.8 times (19,164 x 64 / 678,482 bank windows, ESTIMATED) against B's 4.0: the same confound direction as U against B, "
          "and block 3 showed that passes do not explain U's TAC-LAT cost.  A CONTENT label is read with that caveat and a NO-COST is conservative.  The geometry goal classes and the STRATEGIC cells are read but do not decide.")


# ------------------------------------------------------------------------------------------------------------------------------------------ pure logic
def rulings_problem(path=None, want=RULINGS_SHA256):
    """-> reason or None: the frozen ruling file is present and hashes to the committed sha256"""
    p = path or RULINGS
    if not os.path.isfile(p):
        return f"the ruling file {p} is missing"
    got = L.sha256_file(p)
    return None if got == want else f"the ruling file {p} hashes to {got[:12]}, the committed sha256 is {want[:12]}"


def label_x(delta, separated, f_cell):
    """R2 for one stage: delta = TAC-LAT(X) - TAC-LAT(B_s0) (higher is better), separated = the paired interval excludes 0, f_cell = |B_rep - B_s0| at the same stage.
    X-CONTENT: separated AND delta <= -F; X-GAIN: separated AND delta >= +F; X-NO-COST: not separated OR |delta| < F.  A floor that is not a positive finite number (a bit-identical replicate is not a floor), or a non-finite delta, is NOT-YET."""
    if f_cell is None or not math.isfinite(f_cell) or f_cell <= 0.0 or delta is None or not math.isfinite(delta) or separated is None:
        return "NOT-YET"
    if not separated:
        return "X-NO-COST"
    if delta <= -f_cell:
        return "X-CONTENT"
    if delta >= f_cell:
        return "X-GAIN"
    return "X-NO-COST"


def stage_labels(rowA, rowB, fA, fB):
    """the label at stage A (decides) and at stage B (beside) from the two X - B_s0 TAC-LAT rows; STAGE-SENSITIVE iff both are real labels and differ"""
    la = label_x((rowA or {}).get("delta"), (rowA or {}).get("separated"), fA)
    lb = label_x((rowB or {}).get("delta"), (rowB or {}).get("separated"), fB)
    sens = bool(la in LABELS and lb in LABELS and la != lb)
    return {"label": la, "label_stage_A": la, "label_stage_B": lb, "stage_sensitive": sens, "F_cell": {"A": fA, "B": fB},
            "delta": {"A": (rowA or {}).get("delta"), "B": (rowB or {}).get("delta")}, "separated": {"A": (rowA or {}).get("separated"), "B": (rowB or {}).get("separated")}}


def label_text(lab):
    """the label as printed: '(one seed)' always; STAGE-SENSITIVE with both labels when they differ"""
    if lab["label"] not in LABELS:
        return lab["label"]
    t = f"{lab['label']} (one seed)"
    if lab["stage_sensitive"]:
        t += f" STAGE-SENSITIVE (stage A {lab['label_stage_A']}, stage B {lab['label_stage_B']})"
    return t


def outcome_x(lab, blocked=None, ade_given=True, problems=None):
    """BLOCKED (a refused bank item) first, then NOT-YET (the deciding cell is unread), then the label; a failed control makes the label not quotable"""
    if blocked:
        return ["BLOCKED"]
    if not ade_given or lab is None or lab["label"] not in LABELS:
        return ["NOT-YET"]
    if problems:
        return [f"{label_text(lab)} (CONTROL FAILED: {len(problems)} problem(s); not quotable)"]
    return [label_text(lab)]


def k_vlm_zero(x_vlm, b_vlm, floor, abs_bar=VLM_ABS_BAR):
    """R4: X's VLM-class macro-F1 at stage A is at most B_s0's minus the block-1 floor, OR at most 0.05.  X scoring VLM classes like B_s0 means the mask did not hold.  Any non-finite input fails (a missing number is not a pass)."""
    if not all(v is not None and math.isfinite(v) for v in (x_vlm, b_vlm, floor)):
        return {"pass": False, "reason": "a macro-F1 or the floor is not a finite number", "x_vlm": x_vlm, "b_s0_vlm": b_vlm, "floor": floor}
    bar = b_vlm - floor
    ok_floor, ok_abs = bool(x_vlm <= bar), bool(x_vlm <= abs_bar)
    return {"pass": bool(ok_floor or ok_abs), "x_vlm": x_vlm, "b_s0_vlm": b_vlm, "floor": floor, "bar_b_minus_floor": bar, "bar_abs": abs_bar, "le_b_minus_floor": ok_floor, "le_abs": ok_abs,
            "rule": "X_s0 VLM macro-F1 <= B_s0 - block-1 floor, OR <= 0.05"}


def self_pair_problems(rows):
    """K1 on real cells: an arm minus itself reads exactly 0.0 [0, 0] on every cell that is OK, and at least one cell is OK"""
    ok = {c: v for c, v in rows.items() if v.get("status") == "OK"}
    bad = [f"{c}: {v.get('delta')} [{v.get('lo')}, {v.get('hi')}]" for c, v in ok.items() if not (v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0)]
    return (["no cell is OK"] if not ok else []) + bad


def kevalx_problem_x(path, arms, stages=STAGES):
    """-> reason or None: the KEVALX.json at `path` is a PASS over exactly the real and donor passes of `arms` at `stages` (50c70ae8 vs 008f48b7)"""
    if not os.path.isfile(path):
        return f"{path}: KEVALX.json missing"
    k = json.load(open(path, encoding="utf-8"))
    want = {f"{a}.{s} {v}" for a in arms for s in stages for v in ("real", "donor")}
    if not k.get("pass") or set(k.get("items", {})) != want or not all(v["identical"] for v in k["items"].values()):
        return f"{path}: KEVALX.json is not a PASS over {sorted(want)}"
    return None


def route_problem_x(dirs, arms_by_dir):
    """-> reason or None: every route pass (real, donor, zero, realx, donorx) of every (arm, stage) exists in its folder, and K-EVALX passed in every folder.  dirs {key: folder}; arms_by_dir {key: [(arm, stage)]}"""
    miss = [f"{a}.{s} {v}" for k, items in arms_by_dir.items() for a, s in items for v in B2.ROUTE_VARIANTS if not os.path.exists(f"{dirs[k]}/eval_DS-{a}-{s}_{v}.npz")]
    if miss:
        return f"T0 route passes missing: {miss}"
    for k, items in arms_by_dir.items():
        arms = sorted({a for a, _ in items})
        stages = sorted({s for _, s in items})
        why = kevalx_problem_x(f"{dirs[k]}/KEVALX.json", arms, tuple(stages)) if k == "X" else None
        if k != "X":                           # the banked reference folders carry their own records (block 1: stage B; block 3: stage A); their item sets are larger than the pairs read here
            p = f"{dirs[k]}/KEVALX.json"
            kk = json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None
            why = None if kk and kk.get("pass") and all(f"{a}.{s} {v}" in kk["items"] and kk["items"][f"{a}.{s} {v}"]["identical"] for a, s in items for v in ("real", "donor")) else f"{p}: missing or not a PASS over the reference passes"
        if why:
            return f"K-EVALX: {why} (the zero passes are not admissible)"
    return None


def cjoin_problem(checks):
    """C-JOIN-EQ quoted from the chain's gate record (`gate/checks.json` banked with the arm): -> reason or None.  PASS = pass is true and the problem list is empty"""
    c = (checks or {}).get("C-JOIN-EQ")
    if not c:
        return "no C-JOIN-EQ entry in the gate record"
    if c.get("pass") is not True or c.get("problems"):
        return f"C-JOIN-EQ did not pass: {c.get('problems')}"
    return None


def kre_f1(computed, banked, tol=1e-12):
    """K-RE of the goal-class F1 tool: the block-1 B_s0 / B_rep VLM and GEO macro-F1 at each stage equal the banked RESULT_DIAG.json values exactly"""
    bad, n = [], 0
    for st in STAGES:
        for g in ("VLM", "GEO"):
            for who, key in (("B_s0", "b"), ("B_rep", "a")):
                n += 1
                x, y = computed[st][who][g], banked["pairs"][st]["floor_Brep-B_s0"][g][key]
                if not (math.isfinite(x) and abs(x - y) <= tol * max(1.0, abs(y))):
                    bad.append(f"stage {st} {who} {g}: recomputed {x!r} != banked {y!r}")
    return {"pass": bool(n == 8 and not bad), "rows_compared": n, "mismatches": bad}


def fam_rows_x(t1_rows, strat_rows):
    """one pair at one stage: the T1 cell rows plus (when read) every STRATEGIC cell row"""
    r = dict(t1_rows)
    r.update(strat_rows or {})
    return r


def pair_family_status(rows):
    """{family: family_status}: each family from its own headline cells (the card's rule, unedited), never pooled"""
    return {f: R.family_status(rows, HEADLINE[f]) for f in FAMILIES}


def md_pair_rows(rows, cells):
    out = ["| cell | Delta [95 % interval] sep F verdict | n windows |", "|---|---|---|"]
    for c in cells:
        v = rows.get(c)
        out.append(f"| {c} | {B2.md_cell(v)} | {(v or {}).get('n_windows', 0)} |")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------------------------------------------ goal-class F1 (K-VLM-ZERO)
def f1_part(CI, nb, xdir):
    """the per-goal-class F1 tool of diag/diag_read.py, called unedited (counts, f1_from, group_macro, boot_f1; the ticks loader pointed at this folder for X_s0).  Macro classes = B_s0's support (label side, arm-free), as the diag addendum.
    -> {stage: {arm: {VLM, GEO}}, floors, pairs (bootstrap, beside), oracle control}"""
    path = list(sys.path)
    sys.path.insert(0, DIAG)
    import diag_read as DG
    sys.path[:] = path                                    # diag_read puts read2 first on sys.path: restore it
    DG.ST[9] = xdir                                       # seed slot 9 = this folder (t1/ticks/ticks_DS-X_s0-<stage>_m10_causal_heads.npz)
    J = np.load(DG.INP + "/join_eval141.npz")
    ref = DG.load_ticks0(0, "B_s0", "B")
    assert [str(x) for x in J["clip"]] == ref["clip_names"], "join clip order differs from the ticks clip_names"
    dd = DG.load_eid(DG.dump_dir(0, "B_s0", "B"))
    assert np.array_equal(dd["clip_index"], ref["ci"]) and np.array_equal(dd["ws"], ref["t0"])
    eid = dd["eid"]
    row = J["off"][ref["ci"]] + ref["t0"]
    gy, gw = J["goal_y"][row], J["goal_w"][row]
    bits = np.arange(22)
    wm = ((gw[:, None].astype(np.int64) >> bits) & 1).astype(bool)
    ym = ((gy[:, None].astype(np.int64) >> bits) & 1).astype(bool)
    in_macro = [c for c in range(22) if (wm[:, c] & ym[:, c]).sum() > 0]
    geo = np.array([c for c in in_macro if DG.GOAL22[c] in DG.GEO], int)
    vlm = np.array([c for c in in_macro if DG.GOAL22[c] not in DG.GEO], int)
    groups = {"VLM": vlm, "GEO": geo}
    arms = {}
    for st in STAGES:
        for seed, a in ((0, "B_s0"), (0, "U_s0"), (0, "B_rep"), (9, "X_s0")):
            arms[f"{a}.{st}"] = DG.counts(DG.load_ticks0(seed, a, st, ref)["t_goal_prob"], gy, gw)
    out = {"macro_vlm": [DG.GOAL22[c] for c in vlm], "macro_geometry": [DG.GOAL22[c] for c in geo], "n_windows": int(len(eid)), "stage": {}, "floor": {}, "pairs": {}, "threshold": DG.THR}
    for st in STAGES:
        mac = {}
        for a in ("B_s0", "U_s0", "B_rep", "X_s0"):
            f = DG.f1_from(*[x.sum(0) for x in arms[f"{a}.{st}"]])
            mac[a] = {g: float(DG.group_macro(f, idx)) for g, idx in groups.items()}
        out["stage"][st] = mac
        out["floor"][st] = {g: abs(mac["B_rep"][g] - mac["B_s0"][g]) for g in groups}
        pr = [(f"X_s0.{st}", f"B_s0.{st}"), (f"X_s0.{st}", f"U_s0.{st}")]
        res = DG.boot_f1(CI, {k: v for k, v in arms.items() if k.endswith("." + st)}, pr, eid, groups, nb)
        out["pairs"][st] = {"X-B_s0": res[pr[0]], "X-U_s0": res[pr[1]]}
    ora = DG.counts(np.where(ym, 1.0, 0.0), gy, gw)
    fo = DG.f1_from(*[x.sum(0) for x in ora])
    out["oracle_control"] = {"every_supported_class_f1_is_1": bool(all(fo[c] == 1.0 for c in in_macro)), "min_f1": float(min(fo[c] for c in in_macro))}
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--b1-result", default=L.B1_ROOT + "/out/RESULT_DSCALE10K_B1.json")
    ap.add_argument("--b3-result", default=L.B3_ROOT + "/out/RESULT_DSCALE10K_B3.json")
    ap.add_argument("--diag-result", default=DIAG + "/RESULT_DIAG.json")
    ap.add_argument("--dumps0", default=L.B1_ROOT + "/t1", help="block-1 T1 dumps (B_s0, U_s0, B_rep, stages A and B)")
    ap.add_argument("--dumpsx", default=L.T1, help="X_s0 T1 dumps (stages A and B) and tick files")
    ap.add_argument("--t0route", default=L.OUT + "/t0route", help="X_s0's own route passes (stages A and B) and its KEVALX.json")
    ap.add_argument("--evidence", default=L.HERE + "/evidence", help="X_s0.<stage>.checks.json (the chain's gate record, banked with the arm and sha256-verified by make_banker_x.py)")
    ap.add_argument("--rulings", default=None)
    ap.add_argument("--n-boot", type=int, default=L.NB)
    a = ap.parse_args(argv)
    t_start = time.time()
    CI = L.load_ci()
    dpath = lambda arm, st: f"{a.dumpsx if arm in L.ARMS_X else a.dumps0}/dump_{L.t1_tag(arm, st)}"  # noqa: E731
    ARMS_ST = [(x, s) for x in L.REF_ARMS_X for s in STAGES] + [(x, s) for x in L.ARMS_X for s in STAGES]
    ready = {L.item(arm, st): os.path.isdir(dpath(arm, st)) for arm, st in ARMS_ST}
    rep = {"tool": "ds_read_x.py", "block": "X", "rulings_file": RULINGS, "rulings_sha256": RULINGS_SHA256, "code_of_record": L.COMMIT, "seed": "0 (ONE SEED: reported as one seed)",
           "hypothesis": "DATA-HURTS cause: CONTENT (the 10k clips alone teach a worse TAC-LAT) vs MIXTURE (the cost appears only when mixed)", "steps": {"X_s0": L.STEPS, "B_s0 / U_s0 / B_rep": L.STEPS},
           "tier": {"LONGITUDINAL / LATERAL / TACTICAL": "T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)", "STRATEGIC": "T0 at the window's first tick (--ages fresh)",
                    "goal-class F1 (K-VLM-ZERO)": "T1-kin tick 0 (tactical-head diagnostic, never driving performance)"},
           "evidence_class": "MEASURED (ours; dumps and T0 passes, zero training); block-1 / 3 rows reproduced under K-RE",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only; training-run variance = F_cell (one replicate pair, B_rep); ONE seed",
           "committed_rules": __doc__.split("COMMITTED RULES")[1].split("TIERS:")[0].strip(), "r5_caveat": CAVEAT, "fog": "unscored (PI ruling 2026-11-17)", "items_ready": ready, "controls": {"problems": []}}
    prob = rep["controls"]["problems"]
    why = rulings_problem(a.rulings)
    if why:
        prob.append("RULINGS: " + why)
    miss = [k for k, v in ready.items() if not v]
    if miss:
        rep["outcome"], rep["reason"] = ["NOT-YET"], f"T1 dumps missing: {miss}"
        json.dump(rep, open(a.out, "w"), indent=1, default=float)
        print("WROTE", a.out, "NOT-YET", miss)
        return 1
    b1 = json.load(open(a.b1_result, encoding="utf-8")) if os.path.isfile(a.b1_result) else None
    b3 = json.load(open(a.b3_result, encoding="utf-8")) if os.path.isfile(a.b3_result) else None
    diag = json.load(open(a.diag_result, encoding="utf-8")) if os.path.isfile(a.diag_result) else None
    for nm, v, p in (("K-RE (block 1)", b1, a.b1_result), ("K-RE (block 3 stage-A floor)", b3, a.b3_result), ("K-RE (goal F1)", diag, a.diag_result)):
        if v is None:
            prob.append(f"{nm} NOT-GIVEN: {p} is missing")
    import obd_geom as G
    geom = G.build()
    dg = R.load_dump_n(dpath("B_s0", "A"), L.EVAL_N_EPISODES)
    J = geom.join(dg)
    glat, glon = geom.labels(dg["g"])
    cells = {}
    for arm, st in ARMS_ST:
        cells[(arm, st)] = R.load_arm_cells(geom, dg, J, glat, glon, dpath(arm, st), L.EVAL_N_EPISODES)
        print(f"loaded {arm}.{st} {time.time() - t_start:.0f}s", flush=True)
    eid = dg["eid"]
    rep["controls"]["K1_self_pair"] = {"synthetic": OR.identity_control(CI, eid)}
    if not rep["controls"]["K1_self_pair"]["synthetic"]["pass"]:
        prob.append("K1: a reader self-pair did not read exactly 0.0")
    for st in STAGES:                                       # K1 on the real X cells
        sp = self_pair_problems(R.pair_cells(CI, cells[("X_s0", st)], cells[("X_s0", st)], eid, None, a.n_boot))
        rep["controls"]["K1_self_pair"][f"X-X.{st}"] = {"pass": not sp, "problems": sp[:6]}
        if sp:
            prob.append(f"K1: X - X at stage {st} did not read exactly 0.0 [0, 0] on every cell")
    F = {st: OR.floors_from(cells[("B_rep", st)], cells[("B_s0", st)]) for st in STAGES}
    rep["floors"] = {"definition": "F_cell = |B_rep - B_s0| per cell at the same stage (one replicate pair; B_rep is seed 2)", "stage_A": F["A"], "stage_B": F["B"],
                     "TAC-LAT": {"A": F["A"][CELL], "B": F["B"][CELL]}}
    pairs, t1 = {}, {}
    for st in STAGES:
        for p, (x, y) in PAIRS.items():
            pairs[f"{p}.{st}"] = R.pair_cells(CI, cells[(x, st)], cells[(y, st)], eid, F[st], a.n_boot)
        pairs[f"U-B_s0.{st}"] = R.pair_cells(CI, cells[("U_s0", st)], cells[("B_s0", st)], eid, F[st], a.n_boot)          # for K-RE
        pairs[f"B_rep-B_s0.{st}"] = R.pair_cells(CI, cells[("B_rep", st)], cells[("B_s0", st)], eid, None, a.n_boot)      # for K-RE
    t1 = {k: dict(v) for k, v in pairs.items()}
    # K-RE (T1 cells and floors): the block-1 / 3 stage rows recomputed through this pipeline equal the banked ones, every field exactly
    kre = {}
    if b1 is not None:
        for st in STAGES:
            for p in (f"U-B_s0.{st}", f"B_rep-B_s0.{st}"):
                banked = {c: v for c, v in b1["pairs"][p].items() if c in OR.CELLS}
                kre[f"B1 {p} (T1 cells)"] = B2.kre_b1(R.slim({c: v for c, v in t1[p].items() if c in OR.CELLS}), banked)
            kre[f"B1 floor stage {st}"] = B2.kre_b1({c: {"floor": v} for c, v in F[st].items()}, {c: {"floor": v} for c, v in b1["floors"][f"stage_{st}"].items()})
    if b3 is not None:
        n, bad = R.kre_compare({CELL: {"floor": F["A"][CELL]}}, {CELL: {"floor": b3["floors"]["stage_A"][CELL]}}, ("floor",))
        kre["B3 TAC-LAT stage-A floor (the 0.0767 of block 3)"] = {"pass": bool(n == 1 and not bad), "rows_compared": n, "mismatches": bad}
        bk = {c: v for c, v in b3["pairs"]["U-B_s0.A"].items() if c in OR.CELLS}
        kre["B3 U-B_s0.A (T1 cells)"] = B2.kre_b1(R.slim({c: v for c, v in t1["U-B_s0.A"].items() if c in OR.CELLS}), bk)
    # STRATEGIC (T0 route passes): X's own folder, the references' folders (block 3 for stage A, block 1 for stage B)
    SS, Gr, ci, tt = R.load_strategic_helpers()
    rdirs = {"X": a.t0route, "R3": L.B3_ROOT + "/out/t0route", "R1": L.B1_ROOT + "/out/t0route"}
    strat = {"tier": rep["tier"]["STRATEGIC"], "headline": list(HEADLINE["STRATEGIC"]), "status": "NOT-YET", "not_in_this_card": {
        "detected_information": {"status": "n/a", "reason": "the recipe of record has no ctx head: no output"}, "derived_goals": {"status": "not in this arm"},
        "strategic_intent_6_30s": {"status": "not computable", "reason": "no output beyond 80 m of arc"}},
        "note": "the C0-type v9 nav shares its source with RC-A, so S1 is echo-prone for every arm alike: the Delta between arms is read, never the level"}
    arms_by_dir = {"X": [("X_s0", s) for s in STAGES], "R3": [(x, "A") for x in L.REF_ARMS_X], "R1": [(x, "B") for x in L.REF_ARMS_X]}
    why = route_problem_x(rdirs, arms_by_dir)
    strat_pairs = {}
    if why:
        strat["reason"] = why
    else:
        dirof = lambda arm, st: a.t0route if arm in L.ARMS_X else (rdirs["R3"] if st == "A" else rdirs["R1"])  # noqa: E731
        RA = {(arm, st): R.route_arrays(dirof(arm, st), arm, st, SS, Gr, ci, tt) for arm, st in ARMS_ST}
        sf = {st: R.strat_floors(RA[("B_rep", st)], RA[("B_s0", st)]) for st in STAGES}
        sp = {}
        for st in STAGES:
            for p, (x, y) in {**PAIRS, "U-B_s0": ("U_s0", "B_s0")}.items():
                sp[f"{p}.{st}"] = R.strat_pair(CI, RA[(x, st)], RA[(y, st)], eid, sf[st], a.n_boot)
        strat.update({"status": "OK", "floors": sf, "n_scorable_80m": {f"{k[0]}.{k[1]}": int(np.isfinite(v["S_CKPT_real@80"]).sum()) for k, v in RA.items()}, "best_constant_80m_reference": 0.794,
                      "pairs": {p: R.slim(v) for p, v in sp.items() if not p.startswith("U-B")}})
        strat_pairs = {p: v for p, v in sp.items() if not p.startswith("U-B")}
        for st, key in (("B", "U-B_s0.B"),):
            if b1 is not None and "STRATEGIC" in b1.get("families", {}) and "U-B_s0.B" in b1["families"]["STRATEGIC"].get("block", {}):
                kre["B1 STRATEGIC U-B_s0.B"] = B2.kre_strat_b1(R.slim(sp[key]), b1["families"]["STRATEGIC"]["block"]["U-B_s0.B"])
        if b3 is not None and "U-B_s0.A" in b3["families"]["STRATEGIC"].get("block", {}).get("pairs", {}):
            kre["B3 STRATEGIC U-B_s0.A"] = B2.kre_strat_b1(R.slim(sp["U-B_s0.A"]), b3["families"]["STRATEGIC"]["block"]["pairs"]["U-B_s0.A"])
        else:
            prob.append("K-RE: no banked block-3 stage-A STRATEGIC row to reproduce")
        for p in PAIRS:
            for st in STAGES:
                pairs[f"{p}.{st}"].update({c: v for c, v in sp[f"{p}.{st}"].items() if c in HEADLINE["STRATEGIC"]})
    rep["controls"]["K-RE"] = {"pass": bool(kre) and all(v["pass"] for v in kre.values()), "per_check": kre,
                               "note": "TAC-LAT of B_s0, U_s0 and B_rep at both stages enter through the pair rows U - B_s0 and B_rep - B_s0 (every field exactly equal) and the floors; the absolute per-arm means are not banked rows"}
    if not rep["controls"]["K-RE"]["pass"]:
        prob.append("K-RE: a recomputed block-1 / 3 row differs from the banked row, or no banked row was available")
    kx = {"X": kevalx_problem_x(f"{a.t0route}/KEVALX.json", ("X_s0",))}
    rep["controls"]["K-EVALX"] = {"X_s0 stages A and B": kx["X"] or "PASS", "reference folders": "checked inside the route-pass completeness check (block 1 stage B, block 3 stage A): " + (why or "PASS")}
    if kx["X"]:
        prob.append("K-EVALX on X_s0: " + kx["X"])
    # K-VLM-ZERO: the goal-class F1 tool, X's VLM macro-F1 at stage A
    f1 = None
    try:
        f1 = f1_part(CI, a.n_boot, a.dumpsx.rsplit("/t1", 1)[0] if a.dumpsx.endswith("/t1") else os.path.dirname(a.dumpsx))
    except (OSError, AssertionError, KeyError, ValueError) as e:
        rep["controls"]["K-VLM-ZERO"] = {"pass": False, "reason": f"{type(e).__name__}: {e}"}
        prob.append(f"K-VLM-ZERO: the goal-class F1 read failed: {type(e).__name__}: {e}")
    if f1 is not None:
        kz = k_vlm_zero(f1["stage"]["A"]["X_s0"]["VLM"], f1["stage"]["A"]["B_s0"]["VLM"], f1["floor"]["A"]["VLM"])
        kz["stage_B_beside"] = k_vlm_zero(f1["stage"]["B"]["X_s0"]["VLM"], f1["stage"]["B"]["B_s0"]["VLM"], f1["floor"]["B"]["VLM"])
        rep["controls"]["K-VLM-ZERO"] = kz
        if not kz["pass"]:
            prob.append("K-VLM-ZERO: X_s0 scores VLM classes like B_s0 (macro-F1 above both bars): the VLM mask did not hold")
        if not f1["oracle_control"]["every_supported_class_f1_is_1"]:
            prob.append("K-VLM-ZERO instrument: the oracle arm did not read F1 1.0 on every supported class")
        if diag is not None:
            rep["controls"]["K-RE_goal_F1"] = kre_f1(f1["stage"], diag["f1"])
            if not rep["controls"]["K-RE_goal_F1"]["pass"]:
                prob.append("K-RE (goal F1): the block-1 B_s0 / B_rep macro-F1 differ from the banked RESULT_DIAG.json")
        rep["goal_f1"] = f1
    # C-JOIN-EQ
    cj = {}
    for st in STAGES:
        p = f"{a.evidence}/X_s0.{st}.checks.json"
        if os.path.isfile(p):
            rec = json.load(open(p, encoding="utf-8"))
            cj[st] = {"file": p, "sha256": L.sha256_file(p), "C-JOIN-EQ": rec.get("C-JOIN-EQ"), "problem": cjoin_problem(rec)}
        else:
            cj[st] = {"file": p, "problem": "the chain's gate record is missing (make_banker_x.py writes it)"}
    rep["controls"]["C-JOIN-EQ"] = {"quoted_from": "the chain's gate record banked with the arm (gate/checks.json, sha256 equal to the banker ledger's entry)", "stages": cj, "pass": all(v["problem"] is None for v in cj.values())}
    if not rep["controls"]["C-JOIN-EQ"]["pass"]:
        prob.append("C-JOIN-EQ: " + "; ".join(f"stage {s}: {v['problem']}" for s, v in cj.items() if v["problem"]))
    # the label (stage A decides)
    rows = {st: pairs[f"X-B_s0.{st}"].get(CELL) for st in STAGES}
    lab = stage_labels(rows["A"], rows["B"], F["A"][CELL], F["B"][CELL]) if all(r and r.get("status") == "OK" for r in rows.values()) else None
    blocked = {f"{x}.{s}": L.bank_problem(x, s) for x, s in ARMS_ST if L.bank_problem(x, s)}
    rep["blocked_items"] = blocked
    outcome = outcome_x(lab, list(blocked) or None, lab is not None, prob)
    rep["outcome"] = outcome
    rep["deciding"] = None if lab is None else {"cell": f"{CELL} (T1-kin, higher is better)", "pair": "X_s0 - B_s0", **lab, "rows": {st: R.slim({CELL: rows[st]})[CELL] for st in STAGES},
                                               "X_minus_U_s0_beside": {st: R.slim({CELL: pairs[f"X-U_s0.{st}"][CELL]})[CELL] for st in STAGES}}
    rep["next_step"] = [f"{lab['label']}: {NEXT[lab['label']]}"] if lab and lab["label"] in NEXT else ["no label: see the reason fields"]
    # four families, both stages, both pairs, never pooled
    fam_rows = {p: fam_rows_x({c: v for c, v in t1[p].items() if c in OR.CELLS}, strat_pairs.get(p)) for p in (f"{q}.{s}" for q in PAIRS for s in STAGES)}
    rep["family_status"] = {p: pair_family_status(r) for p, r in fam_rows.items()}
    n_w, n_e = int(len(eid)), int(len(np.unique(eid)))
    rep["families"] = {f: {"tier": rep["tier"]["STRATEGIC"] if f == "STRATEGIC" else rep["tier"]["LONGITUDINAL / LATERAL / TACTICAL"], "headline": list(HEADLINE[f]),
                           "cells": {p: {c: R.slim({c: v})[c] for c, v in B2.fam_cells(r, f).items()} for p, r in fam_rows.items()}, "status": {p: rep["family_status"][p][f] for p in fam_rows},
                           "n_windows": n_w, "n_episodes": n_e, "estimator": "paired episode-cluster bootstrap", "evidence_class": "MEASURED"} for f in FAMILIES}
    rep["families"]["STRATEGIC"]["block"] = strat
    rep["families"]["PRIMARY (ADE / FDE, beside, decide nothing)"] = {"cells": {p: {c: R.slim({c: v})[c] for c, v in B2.fam_cells(r, "PRIMARY").items()} for p, r in fam_rows.items()}}
    rep["pairs"] = {p: R.slim(v) for p, v in pairs.items()}
    rep["not_built_beside_rows"] = {"DK-VIOL counts": "NOT COMPUTABLE here: beside row (OBD o4), routed to the W-FAM-STRAT1-TTC-TACCONF instrument (MM ruling, block 2)",
                                    "per-country / platform eval breakdown (card 8)": "named work item, not in the cell set (diag/diag_read.py --part strata, with X_s0 added, is the instrument)"}
    rep["wall_s"] = round(time.time() - t_start, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    md = ["# DSCALE10K X_s0 read (the 10,000 clips alone against B_s0 and U_s0), stages A and B, ONE SEED (seed 0), MEASURED", "",
          f"Outcome: **{outcome}**.  Controls: problems {prob or 'none'}.  141 episodes / 2,399 windows; paired episode-cluster bootstrap B {a.n_boot} seed 0; ruling file sha256 {RULINGS_SHA256[:12]}.", ""]
    if lab:
        d = rep["deciding"]
        md += [f"## Deciding cell {CELL} (T1-kin, higher is better), X_s0 - B_s0", "",
               "| stage | Delta [95 % interval] | separated | F_cell | label |", "|---|---|---|---|---|"]
        for st in STAGES:
            r = rows[st]
            md.append(f"| {st}{' (decides)' if st == 'A' else ' (beside)'} | {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['separated']} | {F[st][CELL]:.4f} | {lab['label_stage_' + st]} |")
        md += ["", f"Label: **{label_text(lab)}**.  X_s0 - U_s0 beside (no bar): " + "; ".join(f"stage {st} {B2.md_cell(pairs[f'X-U_s0.{st}'][CELL])}" for st in STAGES), "", "Next: " + rep["next_step"][0], "", CAVEAT, ""]
    if f1 is not None:
        kz = rep["controls"]["K-VLM-ZERO"]
        md += [f"K-VLM-ZERO (stage A): X_s0 VLM macro-F1 {kz['x_vlm']:.4f}, B_s0 {kz['b_s0_vlm']:.4f}, block-1 floor {kz['floor']:.4f}, bars {kz['bar_b_minus_floor']:.4f} / {kz['bar_abs']}: **{'PASS' if kz['pass'] else 'FAIL'}**.  GEO macro-F1 beside: X_s0 {f1['stage']['A']['X_s0']['GEO']:.4f}, B_s0 {f1['stage']['A']['B_s0']['GEO']:.4f}.", ""]
    for f in FAMILIES:
        for st in STAGES:
            for p in PAIRS:
                md += [f"## {f} ({rep['families'][f]['tier']}): {p}.{st} = {PAIRS[p][0]} - {PAIRS[p][1]}, n = {n_w} windows / {n_e} episodes, status {rep['families'][f]['status'][f'{p}.{st}']['status']}", "",
                       md_pair_rows(rep["families"][f]["cells"][f"{p}.{st}"], list(rep["families"][f]["cells"][f"X-B_s0.{st}"])), ""]
    md += ["## STRATEGIC members not in this card", "", json.dumps(strat["not_in_this_card"])]
    open(a.out.replace(".json", ".md"), "w", encoding="utf-8").write("\n".join(md))
    print("WROTE", a.out, "|", outcome, "| problems:", len(prob), "|", rep["wall_s"], "s")
    return 1 if prob else 0


if __name__ == "__main__":
    sys.exit(main())
