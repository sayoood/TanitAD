"""ds_read_mix.py -- the DSCALE mixture-arm read: UM_s0 against UE_s0 and BL_s0 (deciding), B_s0 and U_s0 (beside), stage A, ONE SEED (seed 0), MEASURED.  CPU, zero training.  Writes RESULT_DSCALE_MIX.json + .md.
CARD: card_mix/CARD_DSCALE_MIX.md (frozen, sha256 0bc241f5...), sections 5 and 6; MM rulings card_mix/MM_RULINGS_DSCALE_MIX.md; AMENDMENT UM-A1 card_mix/AMENDMENT_UM_A1_KTACSRC.md (sha256 b2b2c4f0..., PI-accepted 2026-11-18).
COMMITTED RULES (card 5.1-5.5, quoted as the rule)
  Pairs (stage A, the same 2,399 windows / 141 episodes): deciding UM_s0 - UE_s0 and UM_s0 - BL_s0; beside UM_s0 - B_s0 and UM_s0 - U_s0; reproduced as controls (K-RE) UE - BL, U - B_s0 and BL - B_s0 from block 3.
  Tiers: LONGITUDINAL / LATERAL / TACTICAL = T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene); STRATEGIC = T0 at the window's first tick; T's latcons and ADE6 = T0 beside.
  Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2,000, seed 0 ("another draw of EPISODES"); training-run variance only through F_cell = |B_rep - B_s0| per cell at stage A (ONE replicate pair); ONE seed.
  Cell rule: Delta = UM - X.  Gain: separated on the better side and |Delta| > F_cell.  Cost: separated on the worse side and |Delta| > F_cell.  Family gain: at least one headline gain and no headline cost; family cost: at least one headline cost.
  DECIDING LABEL (TAC-LAT, T1-kin, stage A; TAC-LAT is accuracy, better = higher):
    DILUTION-CONFIRMED: UM - UE TAC-LAT is a gain (separated better AND Delta > F_cell) AND UM - BL TAC-LAT is not separated worse (its interval's upper end >= 0).
    DILUTION-PARTIAL: UM - UE separated better, and not DILUTION-CONFIRMED (Delta <= F_cell, or UM - BL still separated worse).
    NOT-DILUTION: UM - UE not separated.      REBALANCE-HURTS: UM - UE separated worse.
  PRECEDENCE: BLOCKED (a gate FAIL, a refused argv, K-TACSRC failing, the post-training Adam-state count != 252) > CONTROL-FAILED (K1, K-RE, K-EVALX; "not quotable") > NOT-YET (an item unread) > the label.
  BESIDE LABELS (never change the deciding label): STRAT-KEPT (UM - UE STRATEGIC not a family cost AND UM - BL STRATEGIC a family gain); TAC-LON-PRICE (UM - UE TAC-LON is a cost); a LONGITUDINAL / LATERAL family cost in UM - UE is named with its cell.
  K-TACSRC UNDER AMENDMENT UM-A1: every committed field of UM_s0's summary.json (tac_src_w 0.0, tac_src_list_sha256 37f4f9ee..., tac_src_weight_share_listed exactly 1.0, tac_src_rows_other > 0) EXCEPT the empty-batch bound, which is exact equality with
    card_mix/um_replay/um_empty_replay.json (empty batches 10 = 10, rows listed 1,223,523 and other 2,708,437 equal).  As committed the line reads "tac_src_empty_batches 10 > 5, BLOCKED as committed"; it is printed beside EVERY UM label together
    with "read under UM-A1 (PI-accepted 2026-11-18)".
  ONE SEED: no label is quotable as a lever effect until the replicate (UM_s1 + UE_s1, BL_s1) exists (H-ESTIM-SEED-1).
Usage:  python ds_read_mix.py --out RESULT_DSCALE_MIX.json [--n-boot 2000]"""
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
import ds_read as R  # noqa: E402
import ds_read_b2 as B2  # noqa: E402

OR, FAMILIES, HEADLINE, FAMILY = R.OR, R.FAMILIES, R.HEADLINE, R.FAMILY
CARD_DIR = os.environ.get("DS_CARD_MIX", "F:/TanitAD-scratch/datascale/card_mix")
PINS_MIX = {"CARD_DSCALE_MIX.md": "0bc241f5c4e2292a56e4e8405305c76f8f18d82988c90485604d54f8e2702df0",
            "AMENDMENT_UM_A1_KTACSRC.md": "b2b2c4f0d189359e19d60ff5619ecfe36ab92b5c41d800d9630560ba6ce860d6",
            "um_replay/um_empty_replay.json": "b8b9bc51a9accd07a10495b94bb56af14aa6703d7f5937274720c22c119592f7"}
LIST_SHA = "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1"
EMPTY_BOUND = 5
A1_TAG = "read under UM-A1 (PI-accepted 2026-11-18)"
CELL = "TAC-LAT"
LABELS = ("DILUTION-CONFIRMED", "DILUTION-PARTIAL", "NOT-DILUTION", "REBALANCE-HURTS")
ARMS_ALL = ("B_s0", "U_s0", "B_rep", "UE_s0", "BL_s0", "UM_s0")
ARMS_REF = ("B_s0", "U_s0", "B_rep", "UE_s0", "BL_s0")
PAIRS = {                                    # name -> (arm, reference); stage A, seed 0
    "UM-UE.A": ("UM_s0", "UE_s0"),           # DECIDING: the rebalancing at matched data, steps, batches and non-tactical terms
    "UM-BL.A": ("UM_s0", "BL_s0"),           # DECIDING: is the tactical cost against BL gone
    "UM-B_s0.A": ("UM_s0", "B_s0"),          # beside
    "UM-U_s0.A": ("UM_s0", "U_s0"),          # beside
    "UE-BL.A": ("UE_s0", "BL_s0"),           # K-RE (B3)
    "U-B_s0.A": ("U_s0", "B_s0"),            # K-RE (B3)
    "BL-B_s0.A": ("BL_s0", "B_s0"),          # K-RE (B3)
}
UM_PAIRS = ("UM-UE.A", "UM-BL.A", "UM-B_s0.A", "UM-U_s0.A")
KRE_PAIRS = ("UE-BL.A", "U-B_s0.A", "BL-B_s0.A")
MECH_REF = {"UM-UE.A": "UE_s0", "UM-BL.A": "BL_s0", "UM-B_s0.A": "B_s0", "UM-U_s0.A": "U_s0", "UE-BL.A": "BL_s0", "U-B_s0.A": "B_s0", "BL-B_s0.A": "B_s0"}
NEXT = {
    "DILUTION-CONFIRMED": "1. The replicate pair UM_s1 + UE_s1 (and BL_s1 for the UM - BL side) before the register row is quoted as a lever effect. 2. UM through stage B (OBD-CTL fine-tune, 3,000 steps, its own DAgger bank): the deployable read. "
                          "3. Split the share: a lateral-only weight against a VLM-only renormalisation of the goal term. 4. The VLM-CoT pass on the 10k (PI). If STRAT-KEPT also holds, UM is the candidate base of the next data-scale rung.",
    "DILUTION-PARTIAL": "1. The replicate pair first (about 1.1 F is the expected size even under dilution). 2. If UM - BL is still separated worse: the visit form, which needs the sampler-weight arm (code in ChunkedSampler, a per-source draw rate, its G-DVB entry, an amendment).",
    "NOT-DILUTION": "1. X_s0 (built and staged, waiting on the PI's go): does 10k content alone cost TAC-LAT? 2. The sampler-weight arm (visits). 3. The per-country read of UM (Spain and Sweden cost TAC-LAT at both seeds in block 2).",
    "REBALANCE-HURTS": "1. The VLM-CoT pass on the 10k: fill the masked bits instead of removing the live ones. 2. X_s0. 3. The replicate of UM.",
}


# ------------------------------------------------------------------------------------------------------------------------------------------ pure logic
def tag_um(text):
    """the UM-A1 line, printed beside EVERY UM label (the original K-TACSRC reading and the amendment)"""
    return f"{text} [{A1_TAG}; K-TACSRC as committed: tac_src_empty_batches 10 > 5, BLOCKED as committed]"


def row_ok(r):
    return bool(r) and r.get("status") == "OK" and r.get("delta") is not None and math.isfinite(r["delta"]) and r.get("lo") is not None and r.get("hi") is not None


def row_gain(r):
    """TAC-LAT is accuracy (higher is better): separated better AND Delta > F_cell (strict)"""
    return row_ok(r) and bool(r["separated"]) and r["delta"] > 0 and r.get("floor") is not None and abs(r["delta"]) > r["floor"]


def row_sep_better(r):
    return row_ok(r) and bool(r["separated"]) and r["delta"] > 0


def row_sep_worse(r):
    return row_ok(r) and bool(r["separated"]) and r["delta"] < 0


def label_mix(um_ue, um_bl):
    """the deciding label from the two TAC-LAT rows (UM - UE, UM - BL; dicts delta / lo / hi / separated / floor).  An unreadable row gives NOT-YET."""
    if not (row_ok(um_ue) and row_ok(um_bl)):
        return "NOT-YET"
    if row_gain(um_ue):
        return "DILUTION-CONFIRMED" if um_bl["hi"] >= 0 else "DILUTION-PARTIAL"
    if row_sep_better(um_ue):
        return "DILUTION-PARTIAL"
    if row_sep_worse(um_ue):
        return "REBALANCE-HURTS"
    return "NOT-DILUTION"


def outcome_mix(lab, blocked, control_problems, unread):
    """precedence: BLOCKED > CONTROL-FAILED > NOT-YET > the label"""
    if blocked:
        return ["BLOCKED"]
    if control_problems:
        return ["CONTROL-FAILED (not quotable)"]
    if unread or lab is None or lab == "NOT-YET":
        return ["NOT-YET"]
    return [lab]


def strat_kept(fam_um_ue, fam_um_bl):
    """STRAT-KEPT: UM - UE STRATEGIC is not a family cost AND UM - BL STRATEGIC is a family gain (family_status dicts)"""
    if "NOT-YET" in (fam_um_ue.get("status"), fam_um_bl.get("status")):
        return None
    return bool(fam_um_ue["status"] != "cost" and fam_um_bl["status"] == "gain")


def tac_lon_price(row):
    return bool(row and row.get("status") == "OK" and row.get("cost"))


def other_family_costs(fam_status):
    """a LONGITUDINAL or LATERAL family cost in UM - UE, with its cells"""
    return {f: fam_status[f]["costs"] for f in ("LONGITUDINAL", "LATERAL") if fam_status.get(f, {}).get("status") == "cost"}


def ktacsrc(summary, replay):
    """K-TACSRC under UM-A1 -> {pass_under_UM_A1, problems_under_UM_A1, as_committed: {pass, line}, fields}.  Every committed field except the empty-batch bound is read as committed; the empty-batch count, rows listed and rows other
    are EXACT equal to the replay's."""
    s, rp = summary or {}, replay or {}
    problems = []
    if s.get("tac_src_w") != 0.0:
        problems.append(f"tac_src_w {s.get('tac_src_w')!r} != 0.0")
    if s.get("tac_src_list_sha256") != LIST_SHA:
        problems.append(f"tac_src_list_sha256 {str(s.get('tac_src_list_sha256'))[:12]} != {LIST_SHA[:12]}")
    if s.get("tac_src_weight_share_listed") != 1.0:
        problems.append(f"tac_src_weight_share_listed {s.get('tac_src_weight_share_listed')!r} != exactly 1.0")
    ro = s.get("tac_src_rows_other")
    if not (isinstance(ro, (int, float)) and ro > 0):
        problems.append(f"tac_src_rows_other {ro!r} is not > 0")
    eb = s.get("tac_src_empty_batches")
    ok_committed_bound = isinstance(eb, int) and eb <= EMPTY_BOUND
    for k_s, k_r in (("tac_src_empty_batches", "empty_batches_replayed"), ("tac_src_rows_listed", "rows_listed_replayed"), ("tac_src_rows_other", "rows_other_replayed")):
        if k_s not in s or k_r not in rp or s[k_s] != rp[k_r]:
            problems.append(f"{k_s} {s.get(k_s)!r} != the replay's {k_r} {rp.get(k_r)!r} (UM-A1 requires exact equality)")
    fields = {k: s.get(k) for k in ("tac_src_w", "tac_src_list_sha256", "tac_src_weight_share_listed", "tac_src_rows_listed", "tac_src_rows_other", "tac_src_empty_batches")}
    fields["replay"] = {k: rp.get(k) for k in ("empty_batches_replayed", "rows_listed_replayed", "rows_other_replayed")}
    return {"pass_under_UM_A1": not problems, "problems_under_UM_A1": problems, "fields": fields,
            "as_committed": {"pass": bool(ok_committed_bound and not [p for p in problems if "UM-A1" not in p]),
                             "line": f"tac_src_empty_batches {eb} > {EMPTY_BOUND}, BLOCKED as committed" if not ok_committed_bound else f"tac_src_empty_batches {eb} <= {EMPTY_BOUND}"}}


def gate_problems(checks, post):
    """-> list: a gate FAIL (any entry of the banked gate/checks.json not pass) or the post-training Adam-state count != 252 of 252"""
    out = []
    if not isinstance(checks, dict) or not checks:
        out.append("gate checks.json missing or empty")
    else:
        bad = sorted(k for k, v in checks.items() if not (isinstance(v, dict) and v.get("pass") is True))
        out += [f"gate check not a pass: {k}" for k in bad]
    if not isinstance(post, dict) or post.get("registered") != 252 or post.get("with_adam_state") != 252 or post.get("problems"):
        out.append(f"post-training Adam-state count is not 252 of 252 with no problems: {post}")
    return out


def rulings_problem(card_dir=None):
    """-> reason or None: the frozen card, the amendment and the replay record hash to their committed sha256"""
    import hashlib
    d = card_dir or CARD_DIR
    for rel, want in PINS_MIX.items():
        p = f"{d}/{rel}"
        if not os.path.isfile(p):
            return f"{rel} missing at {p}"
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if got != want:
            return f"{rel} sha256 {got[:8]} != the committed {want[:8]}"
    return None


def kevalx_problem(t0route, arms, stage="A"):
    p = f"{t0route}/KEVALX.json"
    if not os.path.isfile(p):
        return "KEVALX.json missing"
    k = json.load(open(p, encoding="utf-8"))
    want = {f"{a}.{stage} {v}" for a in arms for v in ("real", "donor")}
    if not k.get("pass") or set(k.get("items", {})) != want or not all(v["identical"] for v in k["items"].values()):
        return f"KEVALX.json is not a PASS over {sorted(want)}"
    return None


def route_missing(t0route, arms):
    return [f"{a}.A {v}" for a in arms for v in B2.ROUTE_VARIANTS if not os.path.exists(f"{t0route}/eval_DS-{a}-A_{v}.npz")]


def pair_family_status(rows):
    return {f: R.family_status(rows, HEADLINE[f]) for f in FAMILIES}


def fam_rows_all(pairs, strat):
    out = {}
    for p, rows in pairs.items():
        r = dict(rows)
        if strat.get("status") == "OK" and p in strat["pairs"]:
            r.update(strat["pairs"][p])
        out[p] = r
    return out


def ade_means(Z, names):
    return {n: float(np.asarray(Z[(n, 0)]["ade6"], np.float64).mean()) for n in names}


def kre_ade6(recomputed, banked):
    """K-RE of the stage-A ADE6 means of the five block-3 arms (and F_A) against RESULT_DSCALE10K_B3.json, every field exactly"""
    rows = {k: {"v": v} for k, v in recomputed.items()}
    bk = {k: {"v": v} for k, v in banked.items() if k in recomputed}
    n, bad = R.kre_compare(rows, bk, ("v",))
    return {"pass": bool(n == len(recomputed) and not bad), "rows_compared": n, "mismatches": bad}


def kre_floor(recomputed, banked):
    n, bad = R.kre_compare({k: {"f": v} for k, v in recomputed.items()}, {k: {"f": v} for k, v in banked.items()}, ("f",))
    return {"pass": bool(n == len(banked) and not bad), "rows_compared": n, "mismatches": bad[:12]}


def md_pair_rows(rows, cells):
    out = ["| cell | Delta [95 % interval] sep F verdict | n windows |", "|---|---|---|"]
    for c in cells:
        v = rows.get(c)
        out.append(f"| {c} | {B2.md_cell(v)} | {(v or {}).get('n_windows', 0)} |")
    return "\n".join(out)


def goal_f1(CI, nb, dumpsm, dumps3, dumps0):
    """BESIDE (never decides): per-goal-class macro F1, VLM vs geometry bits, at stage A, for UM / UE / BL / B_s0 / U_s0 / B_rep, via diag/diag_read.py called unedited.  Power-limited, as diag/RESULT_DIAG.md says."""
    DIAG = "F:/TanitAD-scratch/datascale/diag"
    path = list(sys.path)
    sys.path.insert(0, DIAG)
    import diag_read as DG
    sys.path[:] = path
    DG.ST[0], DG.ST[3], DG.ST[9] = os.path.dirname(dumps0), os.path.dirname(dumps3), os.path.dirname(dumpsm)
    J = np.load(DG.INP + "/join_eval141.npz")
    ref = DG.load_ticks0(0, "B_s0", "A")
    dd = DG.load_eid(DG.dump_dir(0, "B_s0", "A"))
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
    where = {"B_s0": 0, "U_s0": 0, "B_rep": 0, "UE_s0": 3, "BL_s0": 3, "UM_s0": 9}
    arms = {f"{a}.A": DG.counts(DG.load_ticks0(sd, a, "A", ref)["t_goal_prob"], gy, gw) for a, sd in where.items()}
    mac = {a: {g: float(DG.group_macro(DG.f1_from(*[x.sum(0) for x in arms[f"{a}.A"]]), idx)) for g, idx in groups.items()} for a in where}
    pr = [("UM_s0.A", "UE_s0.A"), ("UM_s0.A", "BL_s0.A")]
    res = DG.boot_f1(CI, arms, pr, eid, groups, nb)
    return {"macro_vlm": [DG.GOAL22[c] for c in vlm], "macro_geometry": [DG.GOAL22[c] for c in geo], "n_windows": int(len(eid)), "macro_f1": mac,
            "pairs": {"UM-UE": res[pr[0]], "UM-BL": res[pr[1]]}, "tier": "T0 diagnostic beside (tick 0 goal head, threshold 0.5)", "status": "OK"}


# ------------------------------------------------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--b3-result", default=L.B3_ROOT + "/out/RESULT_DSCALE10K_B3.json")
    ap.add_argument("--dumps0", default=L.B1_ROOT + "/t1", help="block-1 T1 dumps (B_s0.A, U_s0.A, B_rep.A)")
    ap.add_argument("--dumps3", default=L.B3_ROOT + "/t1", help="block-3 T1 dumps (UE_s0.A, BL_s0.A)")
    ap.add_argument("--dumpsm", default=L.T1, help="UM_s0.A T1 dump (this folder)")
    ap.add_argument("--t0mech", default=L.OUT + "/t0mech", help="ONE folder: the six reference arms' stage-A mechanism evals (sync_refs.py) and UM's")
    ap.add_argument("--t0route", default=L.OUT + "/t0route", help="UM_s0's stage-A route passes (+ KEVALX.json over UM)")
    ap.add_argument("--t0route-ref", default=L.B3_ROOT + "/out/t0route", help="the five reference arms' stage-A route passes (read only; + their KEVALX.json)")
    ap.add_argument("--um-summary", default=L.CK + "/UM_s0.A/summary.json")
    ap.add_argument("--banker", default=L.HERE + "/banker/UM_s0.A.json")
    ap.add_argument("--evidence", default=L.HERE + "/evidence")
    ap.add_argument("--card-dir", default=CARD_DIR)
    ap.add_argument("--n-boot", type=int, default=L.NB)
    ap.add_argument("--no-f1", action="store_true", help="skip the beside goal-F1 block")
    a = ap.parse_args(argv)
    t_start = time.time()
    CI = L.load_ci()
    dpath = lambda arm, st="A": f"{a.dumpsm if arm == 'UM_s0' else (a.dumps3 if arm in ('UE_s0', 'BL_s0') else a.dumps0)}/dump_{L.t1_tag(arm, st)}"  # noqa: E731
    ready = {L.item(arm, "A"): os.path.isdir(dpath(arm)) for arm in ARMS_ALL}
    rep = {"tool": "ds_read_mix.py", "card": "card_mix/CARD_DSCALE_MIX.md (frozen; sha256 0bc241f5...; MM rulings 2026-11-18)", "amendment": "UM-A1 (card_mix/AMENDMENT_UM_A1_KTACSRC.md, sha256 b2b2c4f0...): " + A1_TAG,
           "block": "mix", "seed": "0 (ONE SEED: reported as one seed)", "code_of_record": L.COMMIT, "hypothesis": "H-DATA-SCALE-1 sub-hypothesis (dilution): the cost travels with the corpus share of T's tactical supervision",
           "steps": {"UM_s0": L.STEPS_ARM["UM_s0"], "UE_s0": L.STEPS_ARM["UE_s0"], "BL_s0": L.STEPS_ARM["BL_s0"], "B_s0 / U_s0 / B_rep": L.STEPS["A"]},
           "tier": {"LONGITUDINAL / LATERAL / TACTICAL": "T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene)", "STRATEGIC": "T0 at the window's first tick (--ages fresh)",
                    "ADE6 / T's latcons / goal F1 (beside)": "T0 (world-model diagnostic, never driving performance)"},
           "evidence_class": "MEASURED (ours; dumps and T0 evals, zero training); block-3 rows reproduced under K-RE (B3)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only; training-run variance = F_cell (one replicate pair, B_rep); ONE seed; inference variance not applicable (deterministic planner)",
           "committed_rules": __doc__.split("COMMITTED RULES")[1].split("Usage:")[0].strip(), "items_ready": ready, "controls": {"problems": []}, "um_a1": A1_TAG}
    prob = rep["controls"]["problems"]
    blocked = {}
    unread = []
    why = rulings_problem(a.card_dir)
    rep["rulings_check"] = why or "card, amendment and replay record hash to their committed sha256"
    if why:
        blocked["rulings"] = why
    for x in ARMS_ALL:
        w = L.bank_problem(x, "A")
        if w:
            blocked[L.item(x, "A")] = w
    # K-TACSRC under UM-A1, the banked gate record and the post-training Adam-state count
    kt = None
    try:
        sm = json.load(open(a.um_summary, encoding="utf-8"))
        bk = json.load(open(a.banker, encoding="utf-8"))
        if L.sha256_file(a.um_summary) != bk.get("summary_sha256"):
            blocked["K-TACSRC"] = "UM_s0's summary.json differs from the banked sha256"
        replay = json.load(open(f"{a.card_dir}/um_replay/um_empty_replay.json", encoding="utf-8"))
        kt = ktacsrc(sm, replay)
        if not kt["pass_under_UM_A1"]:
            blocked["K-TACSRC"] = "; ".join(kt["problems_under_UM_A1"])
    except (OSError, ValueError, KeyError) as e:
        blocked["K-TACSRC"] = f"{type(e).__name__}: {e}"
    rep["k_tacsrc"] = kt
    try:
        gp = gate_problems(json.load(open(f"{a.evidence}/UM_s0.A.checks.json", encoding="utf-8")), json.load(open(f"{a.evidence}/UM_s0.A.post.json", encoding="utf-8")))
    except (OSError, ValueError) as e:
        gp = [f"banked gate / post evidence unreadable: {e}"]
    rep["gate_and_post"] = {"problems": gp, "source": "banked gate/checks.json and post.json of UM_s0.A (HF private, ledger sha256-checked by make_banker_mix.py)"}
    if gp:
        blocked["gate / post"] = "; ".join(gp)
    miss = [k for k, v in ready.items() if not v]
    rep["blocked_items"] = blocked
    if miss:
        rep["outcome"], rep["reason"] = [tag_um("NOT-YET" if not blocked else "BLOCKED")], f"T1 dumps missing: {miss}"
        json.dump(rep, open(a.out, "w"), indent=1, default=float)
        print("WROTE", a.out, rep["outcome"], miss)
        return 1
    b3 = json.load(open(a.b3_result, encoding="utf-8")) if os.path.isfile(a.b3_result) else None
    if b3 is None:
        prob.append(f"K-RE (B3) NOT-GIVEN: {a.b3_result} is missing, so no block-3 row can be checked")
    import obd_geom as G
    geom = G.build()
    dg = R.load_dump_n(dpath("B_s0"), L.EVAL_N_EPISODES)
    J = geom.join(dg)
    glat, glon = geom.labels(dg["g"])
    cells = {}
    for arm in ARMS_ALL:
        cells[arm] = R.load_arm_cells(geom, dg, J, glat, glon, dpath(arm), L.EVAL_N_EPISODES)
        print(f"loaded {arm}.A {time.time() - t_start:.0f}s", flush=True)
    eid = dg["eid"]
    rep["controls"]["K1_synthetic"] = OR.identity_control(CI, eid)
    F = OR.floors_from(cells["B_rep"], cells["B_s0"])
    rep["floors"] = {"definition": "F_cell = |B_rep - B_s0| per cell at stage A (one replicate pair, B_rep seed 2)", "stage_A": F}
    pairs = {p: R.pair_cells(CI, cells[x], cells[y], eid, F, a.n_boot) for p, (x, y) in PAIRS.items()}
    self_um = R.pair_cells(CI, cells["UM_s0"], cells["UM_s0"], eid, F, a.n_boot)
    k1_bad = [c for c, v in self_um.items() if v.get("status") == "OK" and not (v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0)]
    rep["controls"]["K1_self_pair"] = {"pass": bool(rep["controls"]["K1_synthetic"]["pass"] and not k1_bad and any(v.get("status") == "OK" for v in self_um.values())), "UM_vs_UM_cells_not_exactly_zero": k1_bad}
    if not rep["controls"]["K1_self_pair"]["pass"]:
        prob.append("K1: UM vs itself (or the synthetic identity) did not read exactly 0.0 [0, 0]")
    # STRATEGIC (T0 route passes): UM's here, the five references' in read3
    SS, Gr, ci, t = R.load_strategic_helpers()
    strat = {"tier": rep["tier"]["STRATEGIC"], "headline": list(HEADLINE["STRATEGIC"]), "status": "NOT-YET", "not_in_this_card": {
        "detected_information": {"status": "n/a", "reason": "no ctx head in the recipe of record: no output"}, "derived_goals": {"status": "not in this arm"},
        "strategic_intent_6_30s": {"status": "not computable", "reason": "no output beyond 80 m of arc"}},
        "note": "the C0-type v9 nav shares its source with RC-A, so S1 is echo-prone for every arm alike: the Delta between arms is read, never the level. STRATEGIC is reported in full (the strategic pause ended 2026-11-16)."}
    mm_ref, mm_um = route_missing(a.t0route_ref, ARMS_REF), route_missing(a.t0route, ("UM_s0",))
    kx_ref, kx_um = kevalx_problem(a.t0route_ref, ARMS_REF), kevalx_problem(a.t0route, ("UM_s0",))
    rep["controls"]["K-EVALX"] = {"references (read3 stage A, five arms)": kx_ref or "PASS", "UM_s0 stage A": kx_um or "PASS"}
    if mm_ref or mm_um:
        strat["reason"] = f"T0 route passes missing: {mm_ref + mm_um}"
        unread.append("STRATEGIC route passes")
    elif kx_ref or kx_um:
        strat["reason"] = f"K-EVALX: {kx_ref or kx_um} (the zero passes are not admissible)"
        prob.append("K-EVALX: " + str(kx_ref or kx_um))
    else:
        R_ = {arm: R.route_arrays(a.t0route if arm == "UM_s0" else a.t0route_ref, arm, "A", SS, Gr, ci, t) for arm in ARMS_ALL}
        sf = R.strat_floors(R_["B_rep"], R_["B_s0"])
        sp = {p: R.strat_pair(CI, R_[x], R_[y], eid, sf, a.n_boot) for p, (x, y) in PAIRS.items()}
        strat.update({"status": "OK", "floors": sf, "n_scorable_80m": {arm: int(np.isfinite(R_[arm]["S_CKPT_real@80"]).sum()) for arm in R_}, "best_constant_80m_reference": 0.794, "pairs": {p: R.slim(v) for p, v in sp.items()}})
        for p in PAIRS:
            pairs[p].update({c: v for c, v in sp[p].items() if c in HEADLINE["STRATEGIC"]})
    # T0 beside: ADE6 and T's latcons
    import ds_t0 as T
    nm = lambda arm: T.name_of(arm, "A")  # noqa: E731
    HR, GR, PGT, WC, pairfn = T.modules()
    mech, ade = {"tier": rep["tier"]["ADE6 / T's latcons / goal F1 (beside)"]}, {"tier": rep["tier"]["ADE6 / T's latcons / goal F1 (beside)"]}
    kre = {}
    try:
        names = tuple(nm(x) for x in ARMS_ALL)
        Zn, eid0 = PGT.load_all(a.t0mech, (0,), names, WC)
        Z = {(x, 0): Zn[(nm(x), 0)] for x in ARMS_ALL}
        means = ade_means(Z, ARMS_ALL)
        f_a = float(abs(means["B_rep"] - means["B_s0"]))
        ade.update({"mean_ade6_m": means, "F_A": f_a, "n_windows": int(len(eid0)), "n_episodes": int(len(np.unique(eid0))),
                    "pairs": {p: dict(PGT.paired(pairfn, Z[(x, 0)]["ade6"], Z[(y, 0)]["ade6"], eid0, a.n_boot), pair=f"{x} - {y}") for p, (x, y) in PAIRS.items()}})
        if b3 is not None:
            bm = b3["ade6_stage_A_T0"]
            kre["B3 ADE6 means (five arms) and F_A"] = kre_ade6({**{k: v for k, v in means.items() if k in ARMS_REF}, "F_A": f_a}, {**bm["mean_ade6_m"], "F_A": bm["F_A"]})
    except (RuntimeError, OSError, KeyError, ValueError) as e:
        ade["status"] = "NOT-YET"
        ade["reason"] = f"{type(e).__name__}: {e}"
        unread.append("ADE6 (T0 mechanism evals)")
    rep["ade6_stage_A_T0"] = ade
    try:
        fl = T.mech_floor(a.t0mech, nm("B_rep"), nm("B_s0"))
        mech["stage_A_T_floor"] = fl
        mech["pairs"] = {}
        mp = {}
        for p, (x, y) in PAIRS.items():
            m = T.mech_pair(a.t0mech, nm(x), nm(y), 0, a.n_boot, ref_name=nm(MECH_REF[p]))
            mp[p] = m
            mech["pairs"][p] = {"r": m["r"], "rmse_slot": m["rmse_slot"], "mse_delta": m["mse_delta"], "t_rule": B2.label_by_units(T.decide_t, m, fl["floor_mse"]), "frame": m.get("frame"), "beside": m["beside"]}
        mech["identity_control"] = T.identity_control(a.t0mech, nm("UM_s0"))
        if not mech["identity_control"]["pass"]:
            prob.append("K1-T0: UM against itself did not read r exactly 0 [0, 0]")
        if b3 is not None and "pairs" in b3.get("mechanism_stage_A_T0", {}):
            bmech = b3["mechanism_stage_A_T0"]
            rows, bks = {}, {}
            for p in KRE_PAIRS:
                rows[p], bks[p] = B2.flat_mech(mp[p]), B2.flat_mech(bmech["pairs"][p])
            rows["floor"], bks["floor"] = {"floor_mse": fl["floor_mse"]}, {"floor_mse": bmech["stage_A_T_floor"]["floor_mse"]}
            n, bad = R.kre_compare(rows, bks, tuple(rows[KRE_PAIRS[0]]) + ("floor_mse",))
            kre["B3 mechanism rows (UE-BL, U-B_s0, BL-B_s0, floor)"] = {"pass": bool(n >= 15 and not bad), "rows_compared": n, "mismatches": bad[:12]}
    except (RuntimeError, OSError, KeyError, ValueError) as e:
        mech["status"] = "NOT-YET"
        mech["reason"] = f"{type(e).__name__}: {e}"
        unread.append("T's latcons (T0 mechanism evals)")
    rep["mechanism_stage_A_T0"] = mech
    # K-RE (B3): the block-3 rows reproduced exactly
    if b3 is not None:
        for p in KRE_PAIRS:
            kre[f"B3 {p} (T1 cells + STRATEGIC S1 / S2 rows)"] = B2.kre_b1(R.slim(pairs[p]), b3["pairs"][p])
        kre["B3 stage-A floors (24 T1 cells)"] = kre_floor(F, b3["floors"]["stage_A"])
        if strat.get("status") == "OK":
            bs = b3["families"]["STRATEGIC"]["block"]
            for p in KRE_PAIRS:
                kre[f"B3 STRATEGIC all cells {p}"] = B2.kre_strat_b1(R.slim(sp[p]), bs["pairs"][p])
            kre["B3 STRATEGIC floors"] = kre_floor(strat["floors"], bs["floors"])
    rep["controls"]["K-RE_B3"] = {"pass": bool(kre) and all(v["pass"] for v in kre.values()), "per_check": kre}
    if not rep["controls"]["K-RE_B3"]["pass"]:
        prob.append("K-RE (B3): a recomputed block-3 row differs from the banked row, or no banked row was available")
    # the deciding label
    r_ue, r_bl = pairs["UM-UE.A"].get(CELL), pairs["UM-BL.A"].get(CELL)
    lab = label_mix(r_ue, r_bl)
    if lab == "NOT-YET":
        unread.append("TAC-LAT rows")
    labels = outcome_mix(lab, list(blocked) or None, prob, unread)
    rep["outcome"] = [tag_um(x) for x in labels]
    rep["h_dilution"] = {"label": lab, "cell": CELL, "tier": "T1-kin", "F_cell": F.get(CELL), "UM-UE": R.slim({CELL: r_ue})[CELL] if r_ue else None, "UM-BL": R.slim({CELL: r_bl})[CELL] if r_bl else None,
                         "um_bl_upper_end": None if not row_ok(r_bl) else r_bl["hi"], "um_ue_gain": row_gain(r_ue), "precedence": "BLOCKED > CONTROL-FAILED > NOT-YET > label",
                         "unread": unread, "blocked": list(blocked), "control_problems": list(prob)}
    rep["next_step"] = [f"{x.split(' (')[0].split(' [')[0]}: {NEXT[x.split(' (')[0].split(' [')[0]]}" for x in labels if x.split(" (")[0].split(" [")[0] in NEXT] or ["no label: see the reason fields"]
    fam_rows = fam_rows_all(pairs, strat)
    fs = {p: pair_family_status(rows) for p, rows in fam_rows.items()}
    rep["family_status"] = fs
    rep["beside_labels"] = {"STRAT-KEPT": strat_kept(fs["UM-UE.A"]["STRATEGIC"], fs["UM-BL.A"]["STRATEGIC"]), "TAC-LON-PRICE": tac_lon_price(pairs["UM-UE.A"].get("TAC-LON")),
                            "other_family_costs_in_UM_minus_UE": other_family_costs(fs["UM-UE.A"])}
    n_w, n_e = int(len(eid)), int(len(np.unique(eid)))
    rep["families"] = {}
    for f in FAMILIES:
        rep["families"][f] = {"tier": rep["tier"]["STRATEGIC"] if f == "STRATEGIC" else rep["tier"]["LONGITUDINAL / LATERAL / TACTICAL"], "headline": list(HEADLINE[f]),
                              "cells_stage_A": {p: {c: R.slim({c: v})[c] for c, v in B2.fam_cells(rows, f).items()} for p, rows in fam_rows.items()},
                              "status_stage_A": {p: fs[p][f] for p in PAIRS}, "bands": "operative knots 1-10 (operative_0_2s:*), tactical knots 11-30 (tactical_2_6s:*, TAC-LAT, TAC-LON); never pooled",
                              "n_windows": n_w, "n_episodes": n_e, "estimator": "paired episode-cluster bootstrap", "evidence_class": "MEASURED"}
    rep["families"]["STRATEGIC"]["block"] = strat
    rep["families"]["PRIMARY (ADE / FDE, beside, decide nothing)"] = {"cells_stage_A": {p: {c: R.slim({c: v})[c] for c, v in B2.fam_cells(rows, "PRIMARY").items()} for p, rows in fam_rows.items()}}
    rep["pairs"] = {p: R.slim(v) for p, v in pairs.items()}
    if a.no_f1:
        rep["goal_f1_beside"] = {"status": "skipped (--no-f1)"}
    else:
        try:
            rep["goal_f1_beside"] = goal_f1(CI, a.n_boot, a.dumpsm, a.dumps3, a.dumps0)
        except Exception as e:  # noqa: BLE001  (beside, never decides: any failure is recorded, not hidden)
            rep["goal_f1_beside"] = {"status": "NOT-READ", "reason": f"{type(e).__name__}: {e}"}
    rep["not_built_beside_rows"] = {"DK-VIOL counts": "NOT COMPUTABLE here: beside row, outside the 24 cells; routed to the W-FAM-STRAT1-TTC-TACCONF instrument (MM ruling, block 2)",
                                    "per-country / platform eval breakdown": "named work item (card section 9 and DILUTION NOT next lever 3), not in the cell set",
                                    "stage B": "UM_s0 has stage A only (card section 4); every number here is stage A"}
    rep["wall_s"] = round(time.time() - t_start, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    tl = lambda p: B2._lab(mech.get("pairs", {}).get(p, {}).get("t_rule"))  # noqa: E731
    md = [f"# DSCALE mixture arm: UM_s0 against UE_s0 and BL_s0 (deciding), B_s0 and U_s0 (beside), stage A, ONE SEED (seed 0), MEASURED", "",
          f"Outcome: **{rep['outcome']}**", "", f"Controls: problems {prob or 'none'}; blocked {blocked or 'none'}; unread {unread or 'none'}.  141 episodes / 2,399 windows; paired episode-cluster bootstrap B {a.n_boot} seed 0; tiers: LONGITUDINAL / LATERAL / TACTICAL T1-kin, "
          "STRATEGIC T0, ADE6 / latcons / goal F1 T0 beside.  ONE seed: not quotable as a lever effect before the replicate.", "",
          "## K-TACSRC", "", f"* As committed: **{kt['as_committed']['line'] if kt else 'unread'}**.", f"* {A1_TAG}: exact equality with the replay (empty batches, rows listed, rows other) and every other committed field: "
          f"**{'PASS' if kt and kt['pass_under_UM_A1'] else 'FAIL'}** {json.dumps(kt['fields']) if kt else ''}", "",
          f"## Deciding label ({CELL}, T1-kin, stage A; F_cell = {F.get(CELL):.4f})", ""]
    for p in ("UM-UE.A", "UM-BL.A"):
        md.append(f"* {p}: {B2.md_cell(pairs[p].get(CELL))}")
    md += ["", f"Label: **{tag_um(lab)}**", "", f"Beside labels: {json.dumps(rep['beside_labels'])} ({A1_TAG})", "",
           "T's latcons (T0, beside; T-DATA-LIMITED rule, per pair): " + "; ".join(f"{p}: {tl(p)}" for p in PAIRS), "",
           "Family status per pair: " + json.dumps({p: {f: v["status"] for f, v in s.items()} for p, s in fs.items() if p in UM_PAIRS}), ""]
    for f in FAMILIES:
        for p in UM_PAIRS:
            md += [f"## {f} ({rep['families'][f]['tier']}): {p} = {PAIRS[p][0]} - {PAIRS[p][1]}, n = {n_w} windows / {n_e} episodes ({A1_TAG})", "",
                   md_pair_rows(rep["families"][f]["cells_stage_A"][p], list(rep["families"][f]["cells_stage_A"]["UM-UE.A"])), ""]
    md += ["## STRATEGIC members not in this card", "", json.dumps(strat["not_in_this_card"])]
    open(a.out.replace(".json", ".md"), "w", encoding="utf-8").write("\n".join(md))
    print("WROTE", a.out, "|", rep["outcome"], "| problems:", len(prob), "|", rep["wall_s"], "s")
    return 1 if (prob or blocked) else 0


if __name__ == "__main__":
    sys.exit(main())
