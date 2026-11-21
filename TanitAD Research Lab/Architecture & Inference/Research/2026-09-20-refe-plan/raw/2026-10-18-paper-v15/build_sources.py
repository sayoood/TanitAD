#!/usr/bin/env python3
"""build_sources.py -- Paper v1.5 (section 7.21) claim extractor.

Reads every number quoted in section 7.21 straight out of the RAW JSON named in the landed packages
(never out of a RESULT.md), formats it exactly as the paper prints it, and writes:
  * SOURCES.md          claim -> JSON path -> key -> value -> as printed
  * claims_values.json  the same, machine-readable (consumed by verify_paper_numbers.py)

Usage:  python -I build_sources.py [RP_RAW_ROOT] [OUT_DIR]
  RP_RAW_ROOT  default: the repo path  TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw
               (on the dev box: C:/Users/Admin/paper15/s/PK/raw, a short-path extraction of the mirror tip 5fb509a)
The digests of the files read (of the tip's bytes) are written by build_docs.py into SOURCES.md and files_read.json.
"""
import hashlib
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/paper15/s/PK/raw"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
MINUS = "\u2212"

_cache = {}
_hash = {}


def load(rel):
    if rel not in _cache:
        p = os.path.join(ROOT, rel)
        raw = open(p, "rb").read()
        _hash[rel] = hashlib.sha256(raw).hexdigest()
        txt = raw.decode("utf-8")
        if rel.endswith(".jsonl"):
            _cache[rel] = [json.loads(l) for l in txt.splitlines() if l.strip()]
        elif rel.endswith(".log") or rel.endswith(".txt"):
            _cache[rel] = txt
        else:
            _cache[rel] = json.loads(txt)
    return _cache[rel]


def get(rel, path):
    d = load(rel)
    for part in path.split("/"):
        if part == "":
            continue
        if isinstance(d, list):
            d = d[int(part)]
        else:
            d = d[part]
    return d


def fnum(x, nd, signed=False):
    s = f"{abs(x):.{nd}f}"
    if x < 0 and float(s) != 0.0:
        return MINUS + s
    if x < 0:  # rounds to zero but is negative: keep the sign the JSON has
        return MINUS + s
    return ("+" + s) if signed else s


CLAIMS = []


def LG(cid, text, rel, pattern, template, nds, signed=(), kind="MEASURED (POST-HOC, from a log line)", tier="", note="", negate=()):
    """a number read out of a plain-text log line by regex; template uses {0},{1},.. = the formatted groups"""
    import re
    txt = load(rel)
    m = re.search(pattern, txt, re.M)
    if not m:
        raise SystemExit(f"{cid}: pattern not found in {rel}")
    vals = [(-float(g) if i in negate else float(g)) for i, g in enumerate(m.groups())]
    fm = [fnum(v, nds[i], i in signed) for i, v in enumerate(vals)]
    CLAIMS.append(dict(id=cid, text=text, file=rel, path="regex " + pattern, value=vals, printed=template.format(*fm), kind=kind, tier=tier, note=note))
    return vals


def S(cid, text, rel, path, nd=2, signed=False, kind="MEASURED", tier="T1", note="", plain=False, count=False):
    """a scalar claim (plain=True: print an int without thousands separators, e.g. a seed; count=True: the value is len(list))"""
    v = get(rel, path)
    if count:
        v = len(v)
    if isinstance(v, int) and not isinstance(v, bool) and plain:
        printed = str(v)
    else:
        printed = fnum(v, nd, signed) if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else str(v))
    CLAIMS.append(dict(id=cid, text=text, file=rel, path=path, value=v, printed=printed, kind=kind, tier=tier, note=note))
    return v


def T(cid, text, rel, base, keys=("point", "lo", "hi"), nd=2, signed=False, kind="MEASURED", tier="T1", note=""):
    """a point + 95 % interval claim; keys are sub-paths relative to base"""
    vals = [get(rel, base + "/" + k) for k in keys]
    p, lo, hi = vals
    printed = f"{fnum(p, nd, signed)} [{fnum(lo, nd, signed)}, {fnum(hi, nd, signed)}]"
    CLAIMS.append(dict(id=cid, text=text, file=rel, path=base + "  {" + ", ".join(keys) + "}", value=vals, printed=printed,
                       kind=kind, tier=tier, note=note))
    return vals


def D(cid, text, value, nd=2, signed=False, formula="", kind="DERIVED", tier="T1", note=""):
    """a derived claim (arithmetic on claims above); formula is shown in SOURCES.md"""
    printed = fnum(value, nd, signed) if isinstance(value, float) else str(value)
    CLAIMS.append(dict(id=cid, text=text, file="(arithmetic)", path=formula, value=value, printed=printed, kind=kind, tier=tier, note=note))
    return value


# =============================================================================================== (a) P2 full table
FA = "2026-10-16-p2-full-table/anatomy.json"
FP2 = "2026-10-16-p2-full-table/P2_MEASURED_from_full.json"
FSP = "2026-10-16-p2-full-table/same_pick_proof.json"
TA = "T1 NAVSIM v1.1 navtest, P2 seed 0, paired log-cluster bootstrap 10,000 / seed 20260927"
S("a01", "tokens in the full navtest table", FA, "n_tokens", tier=TA)
S("a02", "bootstrap resamples", FA, "estimator/B", tier=TA)
S("a03", "bootstrap seed", FA, "estimator/seed", tier=TA, plain=True)
S("a04", "log clusters", FA, "estimator/n_logs", tier=TA)
pick = T("a05", "P2 pick, mean PDMS x100", FA, "mean_pdms_x100/P2_pick", tier=TA)
orc = T("a06", "best-of-64 oracle, mean PDMS x100", FA, "mean_pdms_x100/best_of_64_oracle", tier=TA)
rnd = T("a07", "random selector (mean over the 64), PDMS x100", FA, "mean_pdms_x100/random_selector_mean_over_64", tier=TA)
gap = T("a08", "oracle minus pick, PDMS points", FA, "mean_pdms_x100/oracle_minus_pick", tier=TA)
nz = T("a09", "... from tokens whose pick is non-zero", FA, "oracle_gap_by_cause_at_pick_pdms_pts/pick_nonzero", nd=3, tier=TA)
dz = T("a10", "... from DAC-zero picks", FA, "oracle_gap_by_cause_at_pick_pdms_pts/zero_DAC", nd=3, tier=TA)
nzc = T("a11", "... from NC-zero picks", FA, "oracle_gap_by_cause_at_pick_pdms_pts/zero_NC", nd=3, tier=TA)
bz = T("a12", "... from both-zero picks", FA, "oracle_gap_by_cause_at_pick_pdms_pts/zero_both", nd=3, tier=TA)
S("a13", "... from other-zero picks", FA, "oracle_gap_by_cause_at_pick_pdms_pts/zero_other/point", tier=TA)
S("a14", "tokens whose best proposal is itself zero (no non-zero proposal)", FA, "n_best_of_64_zero", tier=TA)
S("a15", "zero picks", FA, "n_pick_zero", tier=TA)
S("a16", "zero picks, DAC-only", FA, "n_by_cause_at_pick/zero_DAC", tier=TA)
S("a17", "zero picks, NC-only", FA, "n_by_cause_at_pick/zero_NC", tier=TA)
S("a18", "zero picks, both", FA, "n_by_cause_at_pick/zero_both", tier=TA)
S("a19", "non-zero picks", FA, "n_by_cause_at_pick/pick_nonzero", tier=TA)
S("a20", "tokens where the oracle strictly beats the pick", FA, "n_oracle_strictly_better_than_pick", tier=TA)
S("a21", "non-zero proposals per token (all tokens), mean", FA, "n_proposals_with_pdms_gt_0_per_token/mean", nd=1, tier=TA)
S("a22", "non-zero proposals per token (all tokens), median", FA, "n_proposals_with_pdms_gt_0_per_token/median", nd=0, tier=TA)
sel = T("a23", "oracle-on-SEL ceiling (section 7.20(c) quantity), recomputed from the full table", FP2,
        "Q1_SEL_PROP/oracle_on_SEL_gain_pdms_pts", tier=TA)
S("a24", "SEL zeros among the 509", FP2, "Q1_SEL_PROP/n_SEL", tier=TA)
S("a25", "PROP zeros among the 509", FP2, "Q1_SEL_PROP/n_PROP", tier=TA)
S("a26", "same-pick proof: tokens whose pose is bit-identical to the banked P2 pick", FSP, "pose/n_bit_identical", tier="T1 identity gate")
S("a27", "same-pick proof: max abs pose difference (m)", FSP, "pose/max_abs_m", nd=1, tier="T1 identity gate")
S("a28", "same-pick proof: tokens whose PDMS at the pick equals the banked csv exactly", FSP, "pdms_at_pick_vs_banked_csv/n_exact", tier="T1 identity gate")
S("a29", "same-pick proof: PASS", FSP, "PASS", tier="T1 identity gate")
share_nz = D("a30", "share of the oracle-pick gap that is ranking among non-zero picks (%)", nz[0] / gap[0] * 100, nd=1,
             formula="a09 / a08 * 100", kind="DERIVED (ESTIMATED reading: from the decomposition, not from an arm)")
share_z = D("a31", "share of the gap that sits in zero picks (%)", (dz[0] + nzc[0] + bz[0]) / gap[0] * 100, nd=1,
            formula="(a10 + a11 + a12) / a08 * 100", kind="DERIVED (ESTIMATED reading)")
D("a32", "sum of the three zero-pick terms (PDMS points), equals the SEL ceiling a23", dz[0] + nzc[0] + bz[0], nd=3,
  formula="a10 + a11 + a12  (point estimates)")
D("a33", "sum of the four terms (PDMS points), equals a08", nz[0] + dz[0] + nzc[0] + bz[0], nd=3,
  formula="a09 + a10 + a11 + a12  (point estimates)")
D("a34", "P2 pick to the matched published row 94.55: headroom to climb (PDMS points)", 94.55 - pick[0], nd=2,
  formula="94.55 (PUBLISHED, library 2609.06055 Tab. A13, as in section 7.20(b)) - a05  (point only, no interval)",
  kind="DERIVED (point arithmetic; no interval)")
D("a35", "... as a share of the oracle-pick gap (%)", (94.55 - pick[0]) / gap[0] * 100, nd=1,
  formula="(94.55 - a05) / a08 * 100", kind="DERIVED (point arithmetic; no interval)")

# =============================================================================================== (b) free points
FB = "2026-10-17-free-point-navtest/raw/result/freepoint_result.json"
FBB = "2026-10-17-free-point-navtest/raw/result/families/fp_BASE_families.json"
FBR = "2026-10-17-free-point-navtest/raw/result/families/fp_R0B_families.json"
FBC = "2026-10-17-free-point-navtest/raw/result/families/fp_C0A_families.json"
FBM = "2026-10-17-free-point-navtest/raw/result/families/fp_M25_families.json"
TB = "T1 NAVSIM v1.1 navtest, paired log-cluster bootstrap 10,000 / seed 20260927, one training seed per state"
S("b01", "IC1 tokens compared (inputs identical)", FB, "identity/IC1/IC1_tokens_compared", tier=TB)
S("b02", "IC1 mismatches", FB, "identity/IC1/IC1_mismatches", tier=TB)
S("b03", "IC3: BASE pose equal to the banked u400 pose (tokens of 12,146)", FB, "identity/IC3_IC4/IC3/n_equal", tier=TB)
S("b04", "IC3: fraction equal", FB, "identity/IC3_IC4/IC3/frac_equal", nd=4, tier=TB)
S("b05", "IC3: bar", FB, "identity/IC3_IC4/IC3/bar_frac", nd=2, tier=TB)
T("b06", "IC4b: BASE (re-picked) minus banked u400, PDMS x100", FB, "identity/IC3_IC4/IC4/IC4b_paired_diff_BASEpod_minus_banked",
  keys=("mean", "ci95/0", "ci95/1"), nd=4, signed=True, tier=TB)
S("b07", "IC4b: BASE re-picked mean PDMS x100", FB, "identity/IC3_IC4/IC4/IC4b_BASE_pod_mean", nd=4, tier=TB)
S("b08", "IC4b: banked u400 mean PDMS x100", FB, "identity/IC3_IC4/IC4/IC4b_banked_u400_mean", nd=4, tier=TB)
S("b08a", "IC3: BASE pose disagreements with the banked u400 (near-ties)", FB, "identity/IC3_IC4/IC3/n_disagree", tier=TB)
S("b08b", "IC3: largest near-tie margin among the disagreements", FB, "identity/IC3_IC4/IC3/neartie_margin_quantiles_50_90_99_max/3", nd=6, tier=TB)
S("b08c", "banked u400 zero-PDMS picks (pairs_vs_banked_u400, base side)", FB, "pairs_vs_banked_u400/BASE/primary/zero_base", tier=TB)
S("b09", "IC5: forward + harness bit-identical", FB, "identity/IC5/pass", tier=TB)
S("b10", "D1 (shuffled picks) agreement", FB, "identity/D1_D3/D1_shuffled_pick_agreement", nd=3, tier=TB)
S("b11a", "D3 tokens", FB, "identity/D1_D3/D3/tokens", tier=TB, count=True)
S("b11", "D3 (swapped camera slots) pose agreement, bar: must FAIL (< 0.5)", FB, "identity/D1_D3/D3/frac_equal_with_swap", nd=1, tier=TB)
S("b12", "D3 has teeth", FB, "identity/D1_D3/D3_has_teeth", tier=TB)
for pt, nm in (("BASE", "BASE (deployed scorer, r0 step0)"), ("C0A", "C0A (SFT-continued control, arm A, update 1000)"),
               ("R0B", "R0B (EPO arm B, update 1000)"), ("M25", "M25 (logit mixing alpha 0.25)")):
    pass
lv = {}
for k, pt in (("b13", "BASE"), ("b14", "C0A"), ("b15", "R0B"), ("b16", "M25")):
    lv[pt] = S(k, f"{pt} mean PDMS x100 (levels_x100)", FB, f"levels_x100/{pt}/PDMS", tier=TB)
S("b17", "zero-PDMS picks, BASE", FB, "pairs_vs_BASE_pod/R0B/primary/zero_base", tier=TB)
S("b18", "zero-PDMS picks, C0A", FB, "pairs_vs_BASE_pod/C0A/primary/zero_arm", tier=TB)
S("b19", "zero-PDMS picks, R0B", FB, "pairs_vs_BASE_pod/R0B/primary/zero_arm", tier=TB)
S("b20", "zero-PDMS picks, M25", FB, "pairs_vs_BASE_pod/M25/primary/zero_arm", tier=TB)
D("b21", "extra zero-PDMS picks, R0B minus BASE", 566 - 530, nd=0, formula="b19 - b17", kind="DERIVED", tier=TB)
kk = ("mean", "ci95/0", "ci95/1")
T("b22", "R0B - BASE: PDMS", FB, "pairs_vs_BASE_pod/R0B/primary", keys=kk, signed=True, tier=TB)
S("b23", "R0B - BASE PDMS: better tokens", FB, "pairs_vs_BASE_pod/R0B/primary/better", tier=TB)
S("b24", "R0B - BASE PDMS: worse tokens", FB, "pairs_vs_BASE_pod/R0B/primary/worse", tier=TB)
S("b25", "R0B - BASE PDMS: tied tokens", FB, "pairs_vs_BASE_pod/R0B/primary/tied", tier=TB)
for k, sub in (("b26", "NC"), ("b27", "DAC"), ("b28", "EP"), ("b29", "TTC"), ("b30", "C"), ("b31", "DDC")):
    T(k, f"R0B - BASE: {sub}", FB, f"pairs_vs_BASE_pod/R0B/subscores/{sub}", keys=kk, signed=True, tier=TB)
S("b32", "R0B adverse sub-scores", FB, "pairs_vs_BASE_pod/R0B/adverse_subscores", tier=TB)
T("b33", "C0A - BASE: PDMS", FB, "pairs_vs_BASE_pod/C0A/primary", keys=kk, signed=True, tier=TB)
T("b34", "C0A - BASE: EP", FB, "pairs_vs_BASE_pod/C0A/subscores/EP", keys=kk, signed=True, tier=TB)
T("b35", "M25 - BASE: PDMS", FB, "pairs_vs_BASE_pod/M25/primary", keys=kk, signed=True, tier=TB)
T("b36", "M25 - BASE: EP", FB, "pairs_vs_BASE_pod/M25/subscores/EP", keys=kk, signed=True, tier=TB)
S("b37", "C0A adverse sub-scores", FB, "pairs_vs_BASE_pod/C0A/adverse_subscores", tier=TB)
S("b38", "M25 adverse sub-scores", FB, "pairs_vs_BASE_pod/M25/adverse_subscores", tier=TB)
T("b39", "lever view R0B - C0A: PDMS", FB, "lever_view/R0B-C0A/primary", keys=kk, signed=True, tier=TB)
T("b40", "lever view R0B - C0A: NC", FB, "lever_view/R0B-C0A/subscores/NC", keys=kk, signed=True, tier=TB)
T("b41", "lever view R0B - C0A: EP", FB, "lever_view/R0B-C0A/subscores/EP", keys=kk, signed=True, tier=TB)
T("b42", "lever view R0B - C0A: TTC", FB, "lever_view/R0B-C0A/subscores/TTC", keys=kk, signed=True, tier=TB)
S("b43", "row", FB, "row", tier=TB)
S("b44", "R0B deployable-safe", FB, "deployable_safe_vs_BASE_pod/R0B/safe", tier=TB)
S("b45", "R0B PDMS lower bound > 0", FB, "deployable_safe_vs_BASE_pod/R0B/conditions/PDMS_lo_gt_0", tier=TB)
S("b46", "R0B NC upper bound >= 0", FB, "deployable_safe_vs_BASE_pod/R0B/conditions/NC_hi_ge_0", tier=TB)
S("b47", "R0B TTC upper bound >= 0", FB, "deployable_safe_vs_BASE_pod/R0B/conditions/TTC_hi_ge_0", tier=TB)
S("b48", "M25 deployable-safe", FB, "deployable_safe_vs_BASE_pod/M25/safe", tier=TB)
S("b49", "family components compared (R0B)", FB, "families_adverse_vs_BASE/R0B/components_compared", tier=TB)
S("b50", "family components flagged adverse, R0B (count of the list)", FB, "families_adverse_vs_BASE/R0B/adverse", tier=TB + " (unpaired interval non-overlap screen)", count=True)
S("b51", "family components flagged adverse, C0A (count of the list)", FB, "families_adverse_vs_BASE/C0A/adverse", tier=TB + " (unpaired interval non-overlap screen)", count=True)
S("b52", "family components flagged adverse, M25 (count of the list)", FB, "families_adverse_vs_BASE/M25/adverse", tier=TB + " (unpaired interval non-overlap screen)", count=True)


# C0A and M25 sub-score deltas for the levels table
for pt in ("C0A", "M25"):
    for sub in ("NC", "DAC", "TTC"):
        T(f"b_{pt}_{sub}", f"{pt} - BASE: {sub}", FB, f"pairs_vs_BASE_pod/{pt}/subscores/{sub}", keys=kk, signed=True, tier=TB)
S("b_dd_r", "R0B - BASE: DDC adverse flag", FB, "pairs_vs_BASE_pod/R0B/subscores/DDC/adverse", tier=TB)

TF = "T1 families6, single-arm interval, log-cluster bootstrap 2,000 resamples (cluster unit log_name, 136 clusters); unpaired"


def FAM(cid, text, block, metric, nd):
    for lab, rel in (("BASE", FBB), ("R0B", FBR)):
        p = f"families/refcv6/{block}/ci/components/{metric}"
        T(f"{cid}{lab[0]}", f"{text}, {lab}", rel, p, keys=("mean", "lo", "hi"), nd=nd, tier=TF)


FAM("c", "speed MAE (m/s)", "longitudinal", "speed_mae_mps", 4)
FAM("d", "speed bias (m/s)", "longitudinal", "speed_bias_mps", 4)
FAM("e", "along-track MAE (m)", "longitudinal", "along_mae_m", 4)
FAM("f", "along-track final bias (m)", "longitudinal", "along_final_bias_m", 4)
FAM("g", "acceleration MAE (m/s2)", "longitudinal", "accel_mae_mps2", 4)
FAM("h", "heading MAE (deg)", "lateral", "heading_mae_deg", 3)
FAM("i", "yaw-rate MAE (deg/s)", "lateral", "yaw_rate_mae_degps", 3)
FAM("j", "curvature MAE (1/m)", "lateral", "curvature_mae_1pm", 5)
FAM("k", "cross-track MAE (m)", "lateral", "cross_mae_m", 4)
for blk, nm in (("lateral_decision", "lateral decision"), ("longitudinal_decision", "longitudinal decision"),
                ("maneuver_5way_collapsed", "5-way manoeuvre")):
    for lab, rel in (("BASE", FBB), ("R0B", FBR)):
        for mt in ("accuracy", "kappa"):
            T(f"t_{blk[:3]}_{mt[:3]}_{lab[0]}", f"TACTICAL {nm} {mt}, {lab}", rel, f"families/refcv6/tactical/{blk}/ci/{mt}",
              keys=("mean", "lo", "hi"), nd=4, tier=TF + "; the package prose says tactical rows carry no interval, the JSON carries one")

# =============================================================================================== (c) R0d + SO-CARVE
R0 = "2026-10-18-r0d-outcome/raw/"
TR = "RL-val selection tier (3,000 sets / 75 logs; NOT T1), arm B minus its own arm A, paired log-cluster bootstrap 10,000 / seed 20260927, one seed"
chk = {"u250": R0 + "rlval_B_vs_ownA_u250.json", "u500": R0 + "rlval_B_vs_ownA_u500.json",
       "u750": R0 + "rlval_B_vs_ownA_u750.json", "u1000": R0 + "rlval_B_vs_ownA.json"}
for u, rel in chk.items():
    T(f"r_{u}_ttc", f"R0d {u}: TTC B - A (x100)", rel, "paired_vs_base/B/quantities/TTC", keys=kk, nd=3, signed=True, tier=TR)
    T(f"r_{u}_nc", f"R0d {u}: NC B - A (x100)", rel, "paired_vs_base/B/quantities/NC", keys=kk, nd=3, signed=True, tier=TR)
    T(f"r_{u}_pdms", f"R0d {u}: PDMS B - A (x100)", rel, "paired_vs_base/B/primary", keys=kk, nd=3, signed=True, tier=TR)
    S(f"r_{u}_g3", f"R0d {u}: g3 pick/fan-mean path (bound >= 0.894)", rel, "arms/B/g3/pick_over_fan_mean_path", nd=4, tier=TR)
    S(f"r_{u}_g3s", f"R0d {u}: g3 shortest-bin share (bound <= 0.55)", rel, "arms/B/g3/shortest_bin_share", nd=4, tier=TR)
for u, rel in chk.items():
    S(f"r_{u}_g5ttc", f"R0d {u}: g5 TTC bound held (TTC may not fall by > 0.5)", rel, "paired_vs_base/B/g5/TTC_fall_ok", tier=TR)
    S(f"r_{u}_g5nc", f"R0d {u}: g5 NC bound held", rel, "paired_vs_base/B/g5/NC_fall_ok", tier=TR)
T("r_u1000_ep", "R0d u1000: EP B - A (x100)", chk["u1000"], "paired_vs_base/B/quantities/EP", keys=kk, nd=3, signed=True, tier=TR)
T("r_u1000_ddc", "R0d u1000: DDC B - A (x100)", chk["u1000"], "paired_vs_base/B/quantities/DDC", keys=kk, nd=3, signed=True, tier=TR)
_g3 = [get(rel, "arms/B/g3/pick_over_fan_mean_path") for rel in chk.values()]
_sb = [get(rel, "arms/B/g3/shortest_bin_share") for rel in chk.values()]
D("r_g3_min", "R0d g3 pick/fan-mean path, minimum over the four checks", min(_g3), nd=3, formula="min over r_u*_g3", kind="DERIVED", tier=TR)
D("r_g3_max", "R0d g3 pick/fan-mean path, maximum over the four checks", max(_g3), nd=3, formula="max over r_u*_g3", kind="DERIVED", tier=TR)
D("r_sb_min", "R0d g3 shortest-bin share, minimum over the four checks", min(_sb), nd=3, formula="min over r_u*_g3s", kind="DERIVED", tier=TR)
D("r_sb_max", "R0d g3 shortest-bin share, maximum over the four checks", max(_sb), nd=3, formula="max over r_u*_g3s", kind="DERIVED", tier=TR)
_sft = R0 + "sft.jsonl"
_di = next(i for i, d in enumerate(load(_sft)) if d.get("event") == "data")
S("r_train_sets", "R0d RL-train sets (data event of R0d's own training log)", _sft, f"{_di}/train_sets", tier="R0d's own training log")
S("r_train_ho", "R0d train rows inside held-out logs (data event)", _sft, f"{_di}/train_rows_in_heldout_logs", tier="R0d's own training log")
S("r_kill", "R0d kill code", R0 + "KILLED.json", "code", tier="pre-registered kill rule")
S("r_kill_why", "R0d kill reason", R0 + "KILLED.json", "why", tier="pre-registered kill rule")
SO = R0 + "safeobj_outcome.json"
S("r_out", "R0d pre-registered outcome", SO, "arms/R0d/outcome", tier="pre-registered outcome table")
S("r_S", "R0d bar S verdict", SO, "arms/R0d/S/verdict", tier="pre-registered outcome table")
S("r_G", "R0d bar G verdict", SO, "arms/R0d/G/verdict", tier="pre-registered outcome table")
S("r_gttc", "R0d in-sample signed window mean of the dual's violation g, TTC (updates 651..750, four 25-update means)", SO,
  "arms/R0d/log/numbers/window_g_ttc_mean/value", nd=6, tier="R0d's own training log (in-sample)")
S("r_gnc", "R0d in-sample signed window mean g, NC", SO, "arms/R0d/log/numbers/window_g_nc_mean/value", nd=6, tier="R0d's own training log (in-sample)")
S("r_lamt", "R0d max lambda TTC (cap 10.0)", SO, "arms/R0d/log/numbers/lam_seen_max_ttc_last/value", nd=4, tier="R0d's own training log (in-sample)")
S("r_lamn", "R0d max lambda NC (cap 10.0)", SO, "arms/R0d/log/numbers/lam_seen_max_nc_last/value", nd=4, tier="R0d's own training log (in-sample)")
S("r_pr1", "R0d post-ramp updates with lambda > 0", SO, "arms/R0d/log/numbers/post_ramp_updates_with_lambda_gt_0/value", tier="R0d's own training log (in-sample)")
S("r_pr2", "R0d post-ramp updates", SO, "arms/R0d/log/numbers/post_ramp_updates/value", tier="R0d's own training log (in-sample)")

KC = R0 + "kill_checks.jsonl"
S("r_lc_rate", "R0d u1000 lane-change rate (tactical; count, no interval)", KC, "3/lane_change/rate", nd=4, tier=TR)
S("r_lc_n", "R0d u1000 lane changes (of 3,000 sets)", KC, "3/lane_change/n_lane_change", tier=TR)
S("r_lc_base", "label base rate of lane changes", KC, "3/lane_change/base_rate", nd=4, tier=TR)
S("r_lc_prec", "R0d u1000 lane-change precision", KC, "3/lane_change/precision", nd=3, tier=TR)
S("r_lc_rec", "R0d u1000 lane-change recall", KC, "3/lane_change/recall", nd=3, tier=TR)
S("r_lc_breach", "R0d u1000 K5 lane-change breach", KC, "3/lane_change/breach", tier=TR)
SOC = "2026-10-18-so-carve/logs/"
LG("s_ph_g", "post-hoc: R0d post-ramp g (updates 226..750, 21 rows of 25 updates), mean and se (x100)", SOC + "r0d_insample_posthoc.log",
   r"^post-ramp 226-750\s+con_g_ttc\s+n=21 mean (\S+)\s+sd \S+\s+se (\S+)", "{0} (se {1})", (3, 3), signed=(), tier="R0d's own training log (in-sample), POST-HOC")
LG("s_ph_p", "post-hoc: R0d post-ramp pick gap B - A (updates 226..750, 21 rows), mean and se (x100)", SOC + "r0d_insample_posthoc.log",
   r"^post-ramp 226-750\s+con_p1_gap_ttc\s+n=21 mean (\S+)\s+sd \S+\s+se (\S+)", "{0} (se {1})", (3, 3), signed=(0,), negate=(0,), tier="R0d's own training log (in-sample), POST-HOC")
LG("s_c3_1e6", "A2 control-3 diagnosis at the u1000 state: diff at T=1e-6", SOC + "a2_control3_diag.log",
   r"^1e-06 L \S+ P1 \S+ diff (\S+)", "{0}", (1,), tier="RL-val selection tier, u1000 state, POST-HOC")
LG("s_c3_1e10", "A2 control-3 diagnosis at the u1000 state: diff at T=1e-10", SOC + "a2_control3_diag.log",
   r"^1e-10 L \S+ P1 \S+ diff (\S+)", "{0}", (1,), tier="RL-val selection tier, u1000 state, POST-HOC")

def DL(cid, text, values, nd, signed, formula, kind="DERIVED", tier="", note=""):
    p, lo, hi = values
    printed = f"{fnum(p, nd, signed)} [{fnum(lo, nd, signed)}, {fnum(hi, nd, signed)}]"
    CLAIMS.append(dict(id=cid, text=text, file="(arithmetic)", path=formula, value=values, printed=printed, kind=kind, tier=tier, note=note))
    return values


def P(cid, text, value, printed, where, tier="PUBLISHED, single run, no interval in the source"):
    CLAIMS.append(dict(id=cid, text=text, file=where, path="banked PDF (library.json sha256), table row as named", value=value, printed=printed,
                       kind="PUBLISHED", tier=tier, note=""))
    return value


# CLOVER (library key 2605.15120) Table 5: 64 proposals on 12,146 navtest scenes, true evaluator PDMS; values read in the banked PDF
CL = "library 2605.15120, Table 5 (also Table 23)"
P("p01", "CLOVER Table 5, DrivoR baseline: Oracle@64 PDMS", 0.9933, "0.9933", CL)
P("p02", "CLOVER Table 5, DrivoR baseline: selected PDMS", 0.9369, "0.9369", CL)
P("p03", "CLOVER Table 5, DrivoR baseline: Gap (Oracle - Selected), printed", 0.0564, "5.64", CL + "; printed 0.0564, shown here x100")
P("p04", "CLOVER Table 5, Stage 1: Oracle@64 PDMS", 0.9976, "0.9976", CL)
P("p05", "CLOVER Table 5, Stage 1: selected PDMS", 0.9413, "0.9413", CL)
P("p06", "CLOVER Table 5, Stage 1: Gap, printed", 0.0563, "5.63", CL + "; printed 0.0563, shown here x100")
P("p07", "CLOVER Table 5, Stage 2: Gap, printed (its selected PDMS is 0.9448 in Table 5 and 0.9443 in Table 23)", 0.0495, "4.95", CL + "; printed 0.0495, shown here x100")
D("p08", "ours (P2, a08) over CLOVER baseline gap", gap[0] / 5.64, nd=1, formula="a08 / p03", kind="DERIVED", tier="point arithmetic across non-comparable arms")
D("p09", "ours (P2, a08) over CLOVER Stage-2 gap", gap[0] / 4.95, nd=1, formula="a08 / p07", kind="DERIVED", tier="point arithmetic across non-comparable arms")

# R0d pick loss on RL-val (negated B - A) beside the unconstrained R0's, same tier
_t = get(chk["u1000"], "paired_vs_base/B/quantities/TTC")
DL("r_u1000_loss", "R0d u1000: loss of the pick's TTC (= minus B - A), x100", [-_t["mean"], -_t["ci95"][1], -_t["ci95"][0]], 3, True,
   "-(r_u1000_ttc): point, minus upper, minus lower", tier=TR)

SCJ = "2026-10-18-so-carve/socarve.json"
_pl = get(SCJ, "per_log_r0_u1000/val")
_n = sum(x["n_sets"] for x in _pl)
_top = max(_pl, key=lambda x: x["L_P2_TTC"])
D("s_pl_pool", "post-hoc: val-pool mean loss of E_pi[TTC] (set-weighted, x100)", sum(x["n_sets"] * x["L_P2_TTC"] for x in _pl) / _n, nd=3,
  formula="sum_l n_l * L_l / sum_l n_l over socarve.json per_log_r0_u1000/val (13 logs, 62 sets)", kind="DERIVED (POST-HOC arithmetic)", tier="ESTIMATED harness read, not T1")
D("s_pl_top_n", "post-hoc: sets in the highest-loss val log", _top["n_sets"], nd=0, formula="argmax_l L_l", kind="DERIVED (POST-HOC arithmetic)", tier="ESTIMATED harness read, not T1")
D("s_pl_top", "post-hoc: that log's contribution to the pool mean (x100)", _top["n_sets"] * _top["L_P2_TTC"] / _n, nd=3,
  formula="n_top * L_top / 62", kind="DERIVED (POST-HOC arithmetic)", tier="ESTIMATED harness read, not T1")
D("s_pl_rest_n", "post-hoc: the other val sets", _n - _top["n_sets"], nd=0, formula="62 - n_top", kind="DERIVED (POST-HOC arithmetic)", tier="ESTIMATED harness read, not T1")
D("s_pl_rest", "post-hoc: mean loss of the other 57 sets (x100)",
  (sum(x["n_sets"] * x["L_P2_TTC"] for x in _pl) - _top["n_sets"] * _top["L_P2_TTC"]) / (_n - _top["n_sets"]), nd=3, signed=False,
  formula="(sum - n_top L_top) / (62 - n_top)", kind="DERIVED (POST-HOC arithmetic)", tier="ESTIMATED harness read, not T1")
S("s_rz_g", "RZ-1 R0d in-sample window mean of the dual's quantity g (probability units)", SCJ, "RZ1_R0d_in_sample/con_g_ttc/mean", nd=6,
  tier="R0d's own training log (in-sample)")
S("s_rz_p1", "RZ-1 R0d in-sample window mean of the pick's TTC gap B - A (x100)", SCJ, "RZ1_R0d_in_sample/p1_gap_ttc_x100_window_mean", nd=3,
  signed=False, tier="R0d's own training log (in-sample)")
D("s_rz_g100", "RZ-1 window mean of the dual's quantity g, x100 (negative = no loss)", get(SCJ, "RZ1_R0d_in_sample/con_g_ttc/mean") * 100, nd=3, signed=False,
  formula="s_rz_g * 100", kind="DERIVED", tier="R0d's own training log (in-sample)")
D("s_rz_p1loss", "RZ-1 window mean of the pick's TTC loss (= minus B - A), x100", -get(SCJ, "RZ1_R0d_in_sample/p1_gap_ttc_x100_window_mean"), nd=3, signed=True,
  formula="-(s_rz_p1)", kind="DERIVED", tier="R0d's own training log (in-sample)")
S("s_rz_tag", "RZ-1 tag", SCJ, "RZ1_R0d_in_sample/tag", tier="R0d's own training log (in-sample)")
DEC = "DECISION_R0_u1000_P2_TTC"
S("s_lab", "SO-CARVE main label", SCJ, DEC + "/label", tier="ESTIMATED harness read, not T1")
S("s_rho", "SO-CARVE main rho", SCJ, DEC + "/rho", nd=2, tier="ESTIMATED harness read, not T1")
for k, nm in (("L_tr", "train pool"), ("L_c", "log-disjoint carve (m = 13 of 30, mean over draws)"), ("L_va", "val pool (13 logs, 62 sets)")):
    pass
S("s_ntr_logs", "harness train pool: logs", SCJ, "states/r0_u1000/P2_TTC/n_logs_train", tier="ESTIMATED harness read, not T1")
S("s_ntr_sets", "harness train pool: sets", SCJ, "states/r0_u1000/P2_TTC/n_sets_train", tier="ESTIMATED harness read, not T1")
S("s_nva_logs", "harness val pool: logs", SCJ, "states/r0_u1000/P2_TTC/n_logs_val", tier="ESTIMATED harness read, not T1")
S("s_nva_sets", "harness val pool: sets", SCJ, "states/r0_u1000/P2_TTC/n_sets_val", tier="ESTIMATED harness read, not T1")
S("s_Ltr", "main read: L_tr (x100)", SCJ, DEC + "/L_tr", nd=3, tier="ESTIMATED harness read, not T1")
S("s_Lc", "main read: L_carve (x100), carve m = 13 of 30 logs, mean over 10,000 draws", SCJ, "states/r0_u1000/P2_TTC/carve/13/L_c", nd=3, tier="ESTIMATED harness read, not T1")
S("s_Lva", "main read: L_val pool (x100)", SCJ, DEC + "/L_va", nd=3, tier="ESTIMATED harness read, not T1")
S("s_Gtv", "main read: G_tv", SCJ, DEC + "/G_tv", nd=3, tier="ESTIMATED harness read, not T1")
S("s_Gtv_lo", "main read: G_tv lower", SCJ, DEC + "/ci/G_tv/0", nd=3, tier="ESTIMATED harness read, not T1")
S("s_Gtv_hi", "main read: G_tv upper", SCJ, DEC + "/ci/G_tv/1", nd=3, tier="ESTIMATED harness read, not T1")
S("s_F1", "main read: F1 (carve in-sample)", SCJ, DEC + "/F1_in_sample_carve", tier="ESTIMATED harness read, not T1")
S("s_F2", "main read: F2 (gap unresolved)", SCJ, DEC + "/F2_unresolved_gap", tier="ESTIMATED harness read, not T1")
A1 = "2026-10-18-so-carve/fullval_a1.json"
TA1 = "RL-val selection tier, full 3,000 sets / 75 logs, R0 live dumps (val side MEASURED), paired log-cluster bootstrap 10,000 / seed 20260927"
for u in ("u250", "u500", "u1000"):
    T(f"w_{u}_p2", f"A1 {u}: loss of E_pi[TTC] (the dual's quantity, T=0.1), full RL-val", A1, f"states/r0_{u}/P2_TTC",
      keys=("L_va_full", "ci_L_va_full/0", "ci_L_va_full/1"), nd=3, signed=True, tier=TA1)
    T(f"w_{u}_p1", f"A1 {u}: loss of the pick's TTC (the bar's quantity), full RL-val", A1, f"states/r0_{u}/P1_TTC",
      keys=("L_va_full", "ci_L_va_full/0", "ci_L_va_full/1"), nd=3, signed=True, tier=TA1)
    T(f"w_{u}_G", f"A1 {u}: G_tv_full (val minus train) of the dual's quantity", A1, f"states/r0_{u}/P2_TTC",
      keys=("G_tv_full", "ci_G_tv_full/0", "ci_G_tv_full/1"), nd=3, signed=True, tier=TA1)
S("w_u1000_gap", "A1 u1000 A1_GAP", A1, "states/r0_u1000/P2_TTC/A1_GAP", tier=TA1)
A2 = "2026-10-18-so-carve/tempsweep_a2.json"
TA2 = "RL-val selection tier, full 3,000 sets, R0 states; paired log-cluster bootstrap 10,000 / seed 20260927; PROVISIONAL (control 3 failed)"
for T_ in ("0.1", "0.05", "0.02", "0.01"):
    for u in ("u250", "u500", "u1000"):
        T(f"v_{T_}_{u}", f"A2 T={T_} {u}: L_T - L_pick", A2, f"states/r0_{u}/{T_}", keys=("diff", "ci_diff/0", "ci_diff/1"), nd=3, signed=True, tier=TA2)
S("v_ok_u1000_002", "A2 T=0.02 u1000 within 0.25 of the pick", A2, "states/r0_u1000/0.02/abs_diff_le_0.25", tier=TA2)
S("v_ok_u1000_005", "A2 T=0.05 u1000 within 0.25 of the pick", A2, "states/r0_u1000/0.05/abs_diff_le_0.25", tier=TA2)
S("v_ok_u1000_01", "A2 T=0.1 u1000 within 0.25 of the pick", A2, "states/r0_u1000/0.1/abs_diff_le_0.25", tier=TA2)
S("v_c3", "A2 control 3 (T=1e-4 reproduces the pick to 1e-6), u1000", A2, "controls/T1e-4_equals_P1_u1000", tier=TA2)
for u in ("u250", "u500", "u1000"):
    S(f"v_c3_{u}", f"A2 T=1e-4 {u}: L_T - L_pick", A2, f"states/r0_{u}/0.0001/diff", nd=3, signed=True, tier=TA2)
S("v_c3_n", "A2 T=1e-4 u1000: picks differing from the argmax", A2, "controls/T1e-4_n_picks_differing_from_argmax_u1000", tier=TA2)
S("v_out", "A2 outcome field in the JSON", A2, "outcome", tier=TA2)
S("v_pa", "A2 PICK_ALIGNED T=0.02", A2, "PICK_ALIGNED/0.02", tier=TA2)

# =============================================================================================== (d) SELRULE-DAC confirm
FD = "2026-10-16-selrule-dac/confirm_out_a8/CONFIRM.json"
FDV = "2026-10-16-selrule-dac/FIT_DIAGNOSTIC_VETO_REACH.json"
TD = "T1 NAVSIM v1.1 navtest, P2 full table, paired log-cluster bootstrap 10,000 / seed 20260927, one training seed"
T("d01", "SELRULE-DAC V(0.8) minus B: PDMS x100", FD, "result/compare/PDMS", keys=kk, nd=4, signed=True, tier=TD)
T("d02", "... EP", FD, "result/compare/subscores/EP", keys=kk, nd=4, signed=True, tier=TD)
S("d03", "... TTC (mean)", FD, "result/compare/subscores/TTC/mean", nd=4, signed=True, tier=TD)
T("d04", "... DAC", FD, "result/compare/subscores/DAC", keys=kk, nd=4, signed=True, tier=TD)
S("d05", "picks changed", FD, "result/compare/picks_changed", tier=TD)
S("d06", "zero tokens, B", FD, "result/compare/zero_tokens/b_zero", tier=TD)
S("d07", "zero tokens, V(0.8)", FD, "result/compare/zero_tokens/a_zero", tier=TD)
S("d08", "DAC=0 picks, B", FD, "result/compare/dac0/b", tier=TD)
S("d09", "DAC=0 picks, V(0.8)", FD, "result/compare/dac0/a", tier=TD)
S("d10", "NC=0 picks, B and V(0.8)", FD, "result/compare/nc0/a", tier=TD)
S("d11", "zeros rescued (b0 -> a nonzero)", FD, "result/compare/zero_tokens/rescued_b0_to_a_nonzero", tier=TD)
S("d12", "new zeros created", FD, "result/compare/zero_tokens/new_zero_b_nonzero_to_a0", tier=TD)
S("d13", "PDMS-changing tokens: better", FD, "result/compare/PDMS/better", tier=TD)
S("d14", "PDMS-changing tokens: worse", FD, "result/compare/PDMS/worse", tier=TD)
S("d15", "adverse PDMS sub-scores", FD, "result/compare/adverse_subscores", tier=TD)
S("d16", "family components compared", FD, "families/components_compared", tier=TD)
S("d17", "family components adverse", FD, "families/adverse", tier=TD)
S("d18", "outcome", FD, "outcome/outcome", tier=TD)
T("d19", "oracle O-DAC (true DAC bit replaces the DAC head) minus B", FD, "result/oracles/O-DAC/delta_vs_B", keys=kk, nd=3, signed=True, tier=TD)
S("d20", "O-DAC picks changed", FD, "result/oracles/O-DAC/picks_changed_vs_B", tier=TD)
S("d21", "O-DAC share of the SEL ceiling", FD, "result/oracles/O-DAC/share_of_SEL_ceiling_3.54", nd=3, tier=TD)
T("d22", "oracle O-NC minus B", FD, "result/oracles/O-NC/delta_vs_B", keys=kk, nd=3, signed=True, tier=TD)
T("d23", "oracle O-both minus B", FD, "result/oracles/O-both/delta_vs_B", keys=kk, nd=3, signed=True, tier=TD)
S("d24", "SEL ceiling used by the oracles", FD, "result/oracles/O-DAC/ceiling_pts", nd=4, tier=TD)
S("d25", "B mean PDMS x100", FD, "result/ceilings_x100/B", nd=2, tier=TD)
S("d26", "chosen-rule mean PDMS x100", FD, "result/chosen/PDMS_x100", nd=2, tier=TD)
S("d27", "best-of-64 x100", FD, "result/ceilings_x100/best_of_64", nd=4, tier=TD)
D("d28", "PDMS gain as a share of the O-DAC reach (%)", get(FD, "result/compare/PDMS/mean") / get(FD, "result/oracles/O-DAC/delta_vs_B/mean") * 100,
  nd=1, formula="d01 / d19 * 100  (point estimates)", tier=TD)
S("d29", "disjointness: RL-val logs intersect navtest logs", FD, "disjointness/intersection", tier=TD)
T("d30", "fit set (RL-val, u400 scorer, pre-P2 fan): V(0.8) minus B", FD, "fit_rl_val/delta_vs_B", keys=kk, nd=3, signed=True,
  tier="RL-val selection tier; a TRANSFER to P2 (Amendment 1, A1)")
S("d31", "fit set: B picks with true DAC = 0 (of 3,000 sets)", FDV, "B_picks_with_true_DAC0", tier="RL-val selection tier (u400 scorer)")
S("d32", "median p_DAC at those picks", FDV, "p_DAC_at_B_pick_quantiles_among_DAC0_picks/0.5", nd=3, tier="RL-val selection tier (u400 scorer)")
S("d33", "p_DAC 25th percentile at those picks", FDV, "p_DAC_at_B_pick_quantiles_among_DAC0_picks/0.25", nd=3, tier="RL-val selection tier (u400 scorer)")
S("d34", "p_DAC 75th percentile at those picks", FDV, "p_DAC_at_B_pick_quantiles_among_DAC0_picks/0.75", nd=3, tier="RL-val selection tier (u400 scorer)")
S("d35", "of those picks, p_DAC below 0.8", FDV, "DAC0_picks_where_B_pick_has_p_DAC_below_tau/0.8", tier="RL-val selection tier (u400 scorer)")
S("d36", "of those picks, p_DAC below 0.9", FDV, "DAC0_picks_where_B_pick_has_p_DAC_below_tau/0.9", tier="RL-val selection tier (u400 scorer)")


# =============================================================================================== output
def md_cell(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


def write_outputs():
    rows = []
    for c in CLAIMS:
        v = c["value"]
        if isinstance(v, list):
            vs = "[" + ", ".join(repr(x) for x in v) + "]"
        else:
            vs = repr(v)
        rows.append(f"| {c['id']} | {md_cell(c['text'])} | `{md_cell(c['file'])}` | `{md_cell(c['path'])}` | {md_cell(vs)} | **{md_cell(c['printed'])}** | {md_cell(c['kind'])} | {md_cell(c['tier'])} |")
    hdr = ("| id | claim | JSON file (under `RP/raw/`) | key path | value in the JSON (full precision) | as printed in 7.21 | class | tier / estimator |\n"
           "|---|---|---|---|---|---|---|---|")
    files = "\n".join(f"| `{k}` | `{v}` |" for k, v in sorted(_hash.items()))
    with open(os.path.join(OUT, "claims_table.md"), "w", encoding="utf-8", newline="\n") as f:
        # file digests are written by build_docs.py (SOURCES.md section 2, files_read.json) from the tip's bytes, not from this local extract
        f.write(hdr + "\n" + "\n".join(rows) + "\n")
    with open(os.path.join(OUT, "claims_values.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(CLAIMS, f, indent=1, ensure_ascii=False)
    print(len(CLAIMS), "claims;", len(_hash), "files hashed")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    write_outputs()
    # print for the author
    for c in CLAIMS:
        print(f"{c['id']:>16} | {c['printed']:<34} | {c['text']}")
