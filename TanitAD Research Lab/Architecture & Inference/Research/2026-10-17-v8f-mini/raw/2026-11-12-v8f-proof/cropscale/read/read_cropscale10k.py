#!/usr/bin/env python
"""read_cropscale10k.py -- read of card PREREG_V8F_PROOF_ADD106 (CROPSCALE10K): the head D-H5 of the lead read fit on all eligible train clips (fit seeds 2, 3, 4: heads L2, L3, L4), read at T0 (the gap error) and at
T1-kin/hb0.2 (the reader-fed lead rule, form VD5c of the wrapper variant 28bc31f21ecb4ccb, arms G2, G3, G4, GEc and the reference E3c), pre-roll of 30 steps (6 s), two model seeds.
Derived from visseed/read_visseed10k.py (card ADD104) and visens/read_visens10k.py (card ADD105): functions kept or adapted; the engine of the model reader EC, R, HS, H is imported, not edited.
Stage controls: runs g0 to g10 of card section 5, the grid read and the STRATEGIC family of the arms and the banked references, and the T0 read; writes CONTROLS_FILE. With every control PASS the values are written
  under `payload`, else under `withheld` and no name exists. The stage prints only the control lines (g<n> PASS|FAIL) and a last line; the engine's own output goes to <out>.engine.log.
Stage names: reads CONTROLS_FILE, refuses it unless g0 to g10 are all present and PASS, writes the names of card sections 3 and 4 into RESULT.json and a names log, binds the result to the controls file by sha256.
Usage: python read_cropscale10k.py --stage controls --pull PULL --code CODE --grid-dir DIR --w W --tree TREE --out CONTROLS.json --wrapper-file F --wrapper-check FILE --wrapper-hashes FILE
          --eval-join J.npz [--cs-root R] [--gap-dir D] [--gap-record REC] [--tests-enc F] [--tests-fit F] [--tests-read F] [--n-boot 2000]
       python read_cropscale10k.py --stage names --controls CONTROLS.json --out RESULT.json     |   --selftest
Last line: ZZ READ-CROPSCALE10K <stage> rc <rc> controls_all_pass <bool>"""
import argparse
import contextlib
import copy
import importlib
import io
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("SR_ESTREAD_DIR"), os.path.join(_HERE, "..", "..", "estclose_read"), _HERE):
    if _p and os.path.isfile(os.path.join(_p, "read_estclose10k_n2.py")):
        sys.path.insert(0, os.path.abspath(_p))
        break
sys.path.insert(0, _HERE)
try:
    import read_estclose10k_n2 as EC        # noqa: E402
    HS, R, np = EC.HS, EC.R, EC.np
except ImportError:                          # the unit tests of the name rules and controls need numpy only
    EC = HS = R = None
    import numpy as np                       # noqa: E402,F401


def _sibling(name, dirs):
    """imports the module `name` from the first of the folders `dirs` (relative to this file) that holds it: the pod kit folders kit_fit and kit_enc, or the dev-box folders fit and enc"""
    for d in dirs:
        p = os.path.abspath(os.path.join(_HERE, d))
        if os.path.isfile(os.path.join(p, name + ".py")):
            if p not in sys.path:
                sys.path.insert(0, p)
            return importlib.import_module(name)
    raise ImportError("%s not found in %s" % (name, list(dirs)))


MSG = _sibling("make_scale_gapfiles", (os.path.join("..", "kit_fit"), os.path.join("..", "fit")))      # g2 and g3 (fit kit of this card)
ENCC = _sibling("ctl_g0g1", (os.path.join("..", "kit_enc"), os.path.join("..", "enc")))                # g0 and g1 (encode kit of this card)

CARD, PRE, PRE_T0 = "PREREG_V8F_PROOF_ADD106", "CROPSCALE10K", "CROPSCALE"
SEEDS = (0, 1)                                                                 # model seeds
LSEEDS = (2, 3, 4)                                                             # fit seeds of the large fits (and of the three 500-clip fits of card ADD104)
ARMS3 = ("G2", "G3", "G4")
XARMS = ARMS3 + ("GEc",)                                                       # the four arms that get names
ALLARMS = XARMS + ("E3c",)                                                     # the five arms of the ten rollouts
RUN = {"G2": "RQ50KF5P30Hb20gG2", "G3": "RQ50KF5P30Hb20gG3", "G4": "RQ50KF5P30Hb20gG4", "GEc": "RQ50KF5P30Hb20gGEc", "E3c": "RQ50KF5P30Hb20gE3c"}
GAPFILE = {"G2": "VGAPL_s2", "G3": "VGAPL_s3", "G4": "VGAPL_s4", "GEc": "VGAPLE", "E3c": "VGAPSE3"}      # pod_cropscale10k_queue.sh:211-223
B102 = {"VD5c": "RQ50KF5P30Hb20vVD5c"}                                          # banked arm of card ADD102
B104 = {"F2": "RQ50KF5P30Hb20fF2", "F3": "RQ50KF5P30Hb20fF3", "F4": "RQ50KF5P30Hb20fF4"}    # banked arms of card ADD104 (the 500-clip fits of seeds 2, 3, 4)
B105 = {"E5c": "RQ50KF5P30Hb20eE5c"}                                            # banked arm of card ADD105
V0_RUN = "RQ50KF5P30Hb20vV0"                                                   # banked control run of card ADD102: the reader's own gap, logged at every lead step
BANKED = {"RR": "RQ50KF5P30Hb20", "RLc": "RLcKF5P30Hb20"}
SCALE_PAIR = {"G2": "F2", "G3": "F3", "G4": "F4", "GEc": "E3c"}                # the 500-clip arm of the same kind
C_EXPECTED = {"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0169, "F2": 0.0138, "F3": 0.0185, "F4": 0.0123, "E5c": 0.0138}      # control g6 (card section 5)
MEASURES = (("C", "CONTACT"), ("U", "UNSAFE"), ("F", "FALLING-BEHIND"))
H_CARD = 0.0477
TOL_W2 = 1e-3
WRAPPER_SHA = "28bc31f21ecb4ccb"
CHECK_LAST = "ZZ CHECK-WRAPPER-V PASS"
TIER = "T0 (a probe on logged frames; a diagnostic of a reader, not driving performance) for the gap error; T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s), for the rule"
FAM_REASON = ("the rule changes the planned speed along the path through the distance to the lead; it does not move the path, so the lateral family is expected to be unmoved by it, "
              "and the tactical family is not read by the rule; the strategic family is read from the strategic grid of the engine")
CONTROLS = tuple("g%d" % i for i in range(11))
REF_RUNS = {"RR": BANKED["RR"], "RLc": BANKED["RLc"]}                          # references of the strategic family (the third is the 500-clip arm of the same kind)
HEADS_MID = ("O0", "O1", "M2", "M3", "M4")                                      # the five 500-clip fits on record: fit seeds 0, 1 (card ADD99) and 2, 3, 4 (card ADD104)
HEADS_L = ("L2", "L3", "L4")                                                    # the three large fits
HEADS = HEADS_MID + HEADS_L
T0_EXPECTED = {"O0": 3.960, "O1": 3.913, "M2": 4.112, "M3": 4.079, "M4": 4.056}   # control g9: the record of card ADD105, fit seeds 0 to 4
T0_ROWS = 3841
T0_TOL = 1e-3
N_BOOT_T0 = 2000
BOOT_SEED = 106
WS = "/workspace/v8proof"
G0_CLAUSES = ("g0_all_in_eligible", "g0_none_is_evaluation", "g0_pilot_500_unchanged", "g0_selection_is_pilot_50", "g0_left_out_listed", "g0_no_duplicates", "g0_record_matches_file")
G1_CLAUSES = ("g1_new_files_shape_type", "g1_new_hash_in_manifest", "g1_pilot_unchanged", "g1_identity_three_clips")
G3_PARTS = ("v0", "meta", "build", "mean_large", "mean_mid", "mid_binding")


# ============================================================================================ names
def _bracket(iv, up, down, mid):
    return up if iv[1] < 0.0 else (down if iv[0] > 0.0 else mid)


def names_of(arm, d_rr_c, d_rlc_c, d_vd5c_c, d_rr_f, c_rr, c_rlc):
    """the four names of one arm (card section 4). d_*: interval [lo, hi] of C(X)-C(RR), C(X)-C(RLc), C(X)-C(VD5c), F(X)-F(RR); c_rr, c_rlc: pooled point values of C of the references."""
    p = PRE + "-" + arm + "-"
    H_ = 0.5 * (c_rr - c_rlc)
    n1 = p + ("CUTS-CONTACT" if d_rr_c[1] < 0.0 else ("MORE-CONTACT" if d_rr_c[0] > 0.0 else "CONTACT-NOT-SEPARATED-FROM-RR"))
    n2 = p + ("WITHIN-HALF-THE-GAP" if d_rlc_c[1] <= H_ else ("HALF-THE-GAP-NOT-REACHED" if d_rlc_c[0] > H_ else "HALF-THE-GAP-NOT-DECIDED"))
    n3 = p + ("LESS-CONTACT-THAN-VD5c" if d_vd5c_c[1] < 0.0 else ("MORE-CONTACT-THAN-VD5c" if d_vd5c_c[0] > 0.0 else "CONTACT-NOT-SEPARATED-FROM-VD5c"))
    n4 = p + ("FALLS-BEHIND-MORE" if d_rr_f[0] > 0.0 else ("FALLS-BEHIND-LESS" if d_rr_f[1] < 0.0 else "FALLING-BEHIND-NOT-SEPARATED"))
    return {arm + "_CONTACT": n1, arm + "_HALF": n2, arm + "_VD5C": n3, arm + "_PRICE": n4}


def holds_name(names):
    """card name by the count rule (card section 4) over G2, G3, G4: HOLDS when all three are CUTS-CONTACT, at least two are WITHIN-HALF-THE-GAP and none is HALF-THE-GAP-NOT-REACHED; DOES-NOT-HOLD when one arm is
    not CUTS-CONTACT or one is HALF-THE-GAP-NOT-REACHED; else NOT-DECIDED."""
    cuts = [names[a + "_CONTACT"].endswith("-" + a + "-CUTS-CONTACT") for a in ARMS3]
    within = [names[a + "_HALF"].endswith("-" + a + "-WITHIN-HALF-THE-GAP") for a in ARMS3]
    notreached = [names[a + "_HALF"].endswith("-" + a + "-HALF-THE-GAP-NOT-REACHED") for a in ARMS3]
    if all(cuts) and sum(within) >= 2 and not any(notreached):
        return PRE + "-HOLDS"
    if (not all(cuts)) or any(notreached):
        return PRE + "-DOES-NOT-HOLD"
    return PRE + "-NOT-DECIDED"


def scale_name(label, ivs):
    """the effect of the larger fit for one measure over the three pairs of the same fit seed (card section 4): LOWERS when at least two pairs have the upper end below 0 and none has the lower end above 0;
    RAISES when at least two have the lower end above 0 and none has the upper end below 0; else NOT-SEPARATED. ivs: three intervals [lo, hi] of the pair (large minus 500-clip)."""
    below = sum(1 for iv in ivs if iv[1] < 0.0)
    above = sum(1 for iv in ivs if iv[0] > 0.0)
    if below >= 2 and above == 0:
        return PRE + "-SCALE-LOWERS-" + label
    if above >= 2 and below == 0:
        return PRE + "-SCALE-RAISES-" + label
    return PRE + "-SCALE-" + label + "-NOT-SEPARATED"


def mean_name(label, iv):
    """the mean pair GEc minus E3c for one measure (card section 4)"""
    return PRE + "-MEAN-" + _bracket(iv, "LOWERS-" + label, "RAISES-" + label, label + "-NOT-SEPARATED")


def t0_name(iv, large_errs, mid_errs):
    """the T0 name (card section 3). iv: interval [lo, hi] of the ratio mean(L2, L3, L4) / mean(500-clip fits of seeds 2, 3, 4); large_errs: errors of the three large fits; mid_errs: errors of the five 500-clip fits on record."""
    if iv[1] < 1.0:
        return PRE_T0 + ("-T0-GAP-ERROR-CUT" if all(e < min(mid_errs) for e in large_errs) else "-T0-GAP-ERROR-CUT-WITHIN-FIT-NOISE")
    if iv[0] > 1.0:
        return PRE_T0 + "-T0-MORE-GAP-ERROR"
    return PRE_T0 + "-T0-GAP-ERROR-NOT-SEPARATED"


def all_names(payload):
    out = {}
    pr = payload["pairs"]
    for a in XARMS:
        out.update(names_of(a, pr[a + "-VS-RR"]["C"]["interval"], pr[a + "-VS-RLc"]["C"]["interval"], pr[a + "-VS-VD5c"]["C"]["interval"], pr[a + "-VS-RR"]["F"]["interval"],
                            payload["levels"]["RR"]["C"], payload["levels"]["RLc"]["C"]))
    out["CARD"] = holds_name(out)
    for m, label in MEASURES:
        out["SCALE_" + label] = scale_name(label, [pr[a + "-VS-" + SCALE_PAIR[a]][m]["interval"] for a in ARMS3])
        out["MEAN_" + label] = mean_name(label, pr["GEc-VS-" + SCALE_PAIR["GEc"]][m]["interval"])
    t = payload["t0"]
    out["T0"] = t0_name(t["interval"], [t["errors_m"][h] for h in HEADS_L], [t["errors_m"][h] for h in HEADS_MID])
    chk = card_check(payload, out)
    if not chk["PASS"]:
        raise SystemExit("REFUSED: the names are not the ones the intervals give: %s" % chk["differences"])
    return out


def _sg(iv):
    return -1 if iv[1] < 0.0 else (1 if iv[0] > 0.0 else 0)


def derive_all(payload):
    """independent second derivation of every name from the intervals and errors of the payload (written apart from names_of, holds_name, scale_name, mean_name and t0_name: signs and counts)"""
    pr, lv = payload["pairs"], payload["levels"]
    H = 0.5 * (lv["RR"]["C"] - lv["RLc"]["C"])
    out, cut, within, notreached = {}, 0, 0, 0
    for x in XARMS:
        s_rr = _sg(pr[x + "-VS-RR"]["C"]["interval"])
        lo, hi = pr[x + "-VS-RLc"]["C"]["interval"]
        s_vd = _sg(pr[x + "-VS-VD5c"]["C"]["interval"])
        s_f = _sg(pr[x + "-VS-RR"]["F"]["interval"])
        out[x + "_CONTACT"] = "%s-%s-%s" % (PRE, x, {-1: "CUTS-CONTACT", 1: "MORE-CONTACT", 0: "CONTACT-NOT-SEPARATED-FROM-RR"}[s_rr])
        out[x + "_HALF"] = "%s-%s-%s" % (PRE, x, "WITHIN-HALF-THE-GAP" if hi <= H else ("HALF-THE-GAP-NOT-REACHED" if lo > H else "HALF-THE-GAP-NOT-DECIDED"))
        out[x + "_VD5C"] = "%s-%s-%s" % (PRE, x, {-1: "LESS-CONTACT-THAN-VD5c", 1: "MORE-CONTACT-THAN-VD5c", 0: "CONTACT-NOT-SEPARATED-FROM-VD5c"}[s_vd])
        out[x + "_PRICE"] = "%s-%s-%s" % (PRE, x, {1: "FALLS-BEHIND-MORE", -1: "FALLS-BEHIND-LESS", 0: "FALLING-BEHIND-NOT-SEPARATED"}[s_f])
        if x in ARMS3:
            cut += int(s_rr == -1)
            within += int(hi <= H)
            notreached += int(lo > H)
    if cut == 3 and within >= 2 and notreached == 0:
        out["CARD"] = PRE + "-HOLDS"
    elif cut < 3 or notreached > 0:
        out["CARD"] = PRE + "-DOES-NOT-HOLD"
    else:
        out["CARD"] = PRE + "-NOT-DECIDED"
    for m, label in MEASURES:
        sg = [_sg(pr[x + "-VS-" + SCALE_PAIR[x]][m]["interval"]) for x in ARMS3]
        out["SCALE_" + label] = PRE + ("-SCALE-LOWERS-" + label if (sg.count(-1) >= 2 and sg.count(1) == 0) else ("-SCALE-RAISES-" + label if (sg.count(1) >= 2 and sg.count(-1) == 0) else "-SCALE-" + label + "-NOT-SEPARATED"))
        s = _sg(pr["GEc-VS-" + SCALE_PAIR["GEc"]][m]["interval"])
        out["MEAN_" + label] = PRE + "-MEAN-" + {-1: "LOWERS-" + label, 1: "RAISES-" + label, 0: label + "-NOT-SEPARATED"}[s]
    t = payload["t0"]
    lo, hi = t["interval"]
    small = min(t["errors_m"][h] for h in HEADS_MID)
    below_all = sum(1 for h in HEADS_L if t["errors_m"][h] < small) == len(HEADS_L)
    out["T0"] = PRE_T0 + ("-T0-GAP-ERROR-CUT" if (hi < 1.0 and below_all) else ("-T0-GAP-ERROR-CUT-WITHIN-FIT-NOISE" if hi < 1.0 else ("-T0-MORE-GAP-ERROR" if lo > 1.0 else "-T0-GAP-ERROR-NOT-SEPARATED")))
    return out


def card_check(payload, got):
    """re-derives every name from the intervals of the payload by derive_all and compares with the names given; also the point ratio of the T0 read against its errors. {"PASS", "differences"}"""
    exp = derive_all(payload)
    diff = ["%s: got %s, expected %s" % (k, got.get(k), v) for k, v in sorted(exp.items()) if got.get(k) != v]
    diff += ["%s: name present that the card does not have" % k for k in sorted(got) if k not in exp]
    t = payload["t0"]
    want = (sum(t["errors_m"][h] for h in HEADS_L) / len(HEADS_L)) / (sum(t["errors_m"][h] for h in HEADS_MID[2:]) / 3.0)
    if abs(t["ratio"] - want) > 1e-9:
        diff.append("T0 ratio %s is not the ratio of the errors %s" % (t["ratio"], want))
    return {"PASS": not diff, "differences": diff}


def h_sensitivity(payload):
    """notes, no name: the half-the-gap name of each arm again with H = 0.0477 as the card prints it; `differs` is true when that name is not the one made with the point values"""
    pr = payload["pairs"]
    h = 0.5 * (payload["levels"]["RR"]["C"] - payload["levels"]["RLc"]["C"])
    out = {"H_point_values": h, "H_card": H_CARD, "differs": False}
    for arm in XARMS:
        lo, hi = pr[arm + "-VS-RLc"]["C"]["interval"]
        f = lambda H: "WITHIN" if hi <= H else ("NOT-REACHED" if lo > H else "NOT-DECIDED")      # noqa: E731
        out[arm] = {"with_point_values": f(h), "with_H_card": f(H_CARD)}
        out["differs"] = bool(out["differs"] or f(h) != f(H_CARD))
    return out


def assemble_controls(controls, payload):
    """the controls file. The values some controls carry (key `values`) are moved into the payload (or `withheld`) so that nothing but the flags and failure texts stands in the controls themselves."""
    ok = bool(controls) and sorted(controls) == sorted(CONTROLS) and all(isinstance(v, dict) and v.get("PASS") is True for v in controls.values())
    vals = {k: v["values"] for k, v in controls.items() if isinstance(v, dict) and "values" in v}
    flat = {k: ({kk: vv for kk, vv in v.items() if kk != "values"} if isinstance(v, dict) else v) for k, v in controls.items()}
    pl = dict(payload)
    pl["control_values"] = vals
    res = {"card": CARD, "tier": TIER, "controls": flat, "controls_all_pass": ok}
    res["payload" if ok else "withheld"] = pl
    return res


# ============================================================================================ T0 read (card section 3) and its controls g9, g10
def sha_file(p):
    return MSG.sha_file(p)


def clip_of_rows(J):
    return np.repeat(np.arange(len(J["clip"]), dtype=np.int64), np.asarray(J["n"], np.int64))


def t0_rows(J, D, heads):
    """-> (lab, G, sel, n_label_differs). The join rows with a lead in the label (lab_present and a finite lab_gap) and a finite output of every head in `heads`; the label fields are read from the dump of the first head and
    compared with the others (`n_label_differs` counts rows where they differ)."""
    h0 = heads[0]
    lab = MSG.on_join_rows(J, D[h0], "lab_gap").astype(np.float64)
    labp = MSG.on_join_rows(J, D[h0], "lab_present")
    G = np.stack([MSG.on_join_rows(J, D[h], "gap").astype(np.float64) for h in heads], 0)
    nd = 0
    for h in heads[1:]:
        l2, p2 = MSG.on_join_rows(J, D[h], "lab_gap").astype(np.float64), MSG.on_join_rows(J, D[h], "lab_present")
        nd += int((~np.isclose(l2, lab, equal_nan=True, rtol=0, atol=0)).sum() + (p2 != labp).sum())
    sel = labp & np.isfinite(lab) & np.isfinite(G).all(axis=0)
    return lab, G, sel, nd


def cluster_sums(ci, vals, n_clips):
    """vals [k, rows] summed per clip -> [n_clips, k]"""
    out = np.zeros((n_clips, vals.shape[0]))
    for i in range(vals.shape[0]):
        out[:, i] = np.bincount(ci, weights=vals[i], minlength=n_clips)
    return out


def boot_ratio(s_num, s_den, n_boot=N_BOOT_T0, seed=BOOT_SEED):
    """cluster bootstrap of the ratio sum(s_num) / sum(s_den) over the clips (the models, here the fits, held fixed): n_boot draws of the clips with replacement, a fixed generator seed; the two-sided 95 % percentile interval."""
    s_num, s_den = np.asarray(s_num, np.float64), np.asarray(s_den, np.float64)
    nc = len(s_num)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, nc, size=(n_boot, nc))
    den = s_den[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = s_num[idx].sum(axis=1) / den
    r = r[np.isfinite(r)]
    if r.size < n_boot // 2:
        return None
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def t0_five(J, D):
    """control g9's measure: on the rows with a lead in the label and an output of all five 500-clip fits, the mean absolute gap error of each. -> dict"""
    lab, G, sel, nd = t0_rows(J, D, HEADS_MID)
    e = np.abs(G[:, sel] - lab[sel]).mean(axis=1) if sel.any() else np.full(len(HEADS_MID), np.nan)
    return {"n_rows": int(sel.sum()), "n_label_fields_differ": nd, "errors_m": {h: float(e[i]) for i, h in enumerate(HEADS_MID)}}


def t0_read(J, D, n_boot=N_BOOT_T0, seed=BOOT_SEED):
    """the card's T0 read: on the rows with a lead in the label and an output of all eight heads the mean absolute gap error of every fit, the ratio mean(L2, L3, L4) / mean(M2, M3, M4) with its cluster-bootstrap interval over the
    evaluation clips, and the printed-with-no-name facts (correlation of each head's gap with the label gap). The name comes from payload["t0"] by all_names."""
    lab, G, sel, nd = t0_rows(J, D, HEADS)
    out = {"tier": "T0", "n_rows": int(sel.sum()), "n_label_fields_differ": nd, "n_boot": n_boot, "seed": seed, "interval_level": "two-sided 95 % percentile", "errors_m": None, "ratio": None, "interval": None}
    if not sel.any():
        return out
    E = np.abs(G[:, sel] - lab[sel])
    out["errors_m"] = {h: float(E[i].mean()) for i, h in enumerate(HEADS)}
    ci = clip_of_rows(J)[sel]
    S = cluster_sums(ci, E, len(J["clip"]))
    il, im = [HEADS.index(h) for h in HEADS_L], [HEADS.index(h) for h in ("M2", "M3", "M4")]
    s_num, s_den = S[:, il].mean(axis=1), S[:, im].mean(axis=1)
    out["ratio"] = float(s_num.sum() / s_den.sum())
    out["interval"] = boot_ratio(s_num, s_den, n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["correlation_with_label_gap"] = {h: float(np.corrcoef(G[i, sel], lab[sel])[0, 1]) for i, h in enumerate(HEADS)}
    small = min(out["errors_m"][h] for h in HEADS_MID)
    out["smallest_500_clip_error_m"] = small
    out["each_large_below_smallest_500_clip"] = {h: bool(out["errors_m"][h] < small) for h in HEADS_L}
    rec_small = min(T0_EXPECTED.values())
    out["each_large_below_recorded_smallest_%s" % ("%.3f" % rec_small)] = {h: bool(out["errors_m"][h] < rec_small) for h in HEADS_L}
    return out


def ctl_g9(five):
    """g9 (the T0 measure): the mean absolute gap error of the five 500-clip fits, recomputed on the rows with a lead in the label and an output of all five, equals 3.960, 3.913, 4.112, 4.079, 4.056 m (fit seeds 0 to 4)
    within 1e-3 m, on 3,841 rows; the label fields of the five dumps agree on every row (an added clause: the measure is not defined otherwise)."""
    bad = []
    if not isinstance(five, dict) or not five.get("errors_m"):
        return {"PASS": False, "failures": ["no T0 measure"]}
    for h, want in T0_EXPECTED.items():
        got = five["errors_m"].get(h)
        if got is None or not np.isfinite(got) or abs(got - want) > T0_TOL + 1e-12:
            bad.append("%s: error %s, expected %.3f within %g" % (h, got, want, T0_TOL))
    if five.get("n_rows") != T0_ROWS:
        bad.append("%s rows, expected %d" % (five.get("n_rows"), T0_ROWS))
    if five.get("n_label_fields_differ") != 0:
        bad.append("the label fields of the five dumps differ on %s rows" % five.get("n_label_fields_differ"))
    return {"PASS": not bad, "failures": bad, "values": {"errors_m": five.get("errors_m"), "n_rows": five.get("n_rows")}}


def perm_ratio(J, D_perm, D_l2, n_boot=N_BOOT_T0, seed=BOOT_SEED):
    """g10's measure: on the rows with a lead in the label and an output of both, the mean absolute gap error of the permuted-label fit divided by that of L2, cluster bootstrap over the clips. -> dict"""
    D = {"P": D_perm, "L2": D_l2}
    lab, G, sel, nd = t0_rows(J, D, ("L2", "P"))
    out = {"n_rows": int(sel.sum()), "n_label_fields_differ": nd, "n_boot": n_boot, "seed": seed, "ratio": None, "interval": None, "error_perm_m": None, "error_L2_m": None}
    if not sel.any():
        return out
    E = np.abs(G[:, sel] - lab[sel])
    S = cluster_sums(clip_of_rows(J)[sel], E, len(J["clip"]))
    out["error_L2_m"], out["error_perm_m"] = float(E[0].mean()), float(E[1].mean())
    out["ratio"] = float(S[:, 1].sum() / S[:, 0].sum())
    out["interval"] = boot_ratio(S[:, 1], S[:, 0], n_boot, seed)
    return out


def ctl_g10(pr, perm_facts=None, perm_meta=None, l2_facts=None, other_heads=None):
    """g10 (deliberate regression): the lower end of the ratio of the permuted-label fit's error to L2's (cluster bootstrap over the evaluation clips, 2,000 draws, fixed generator seed) is above 1.0. Added clauses that make
    `the fit with permuted labels` checkable: its record says permuted labels, seed 2, arm D, head H5, the fit clips and selection clips of L2, its head file differs from the eight on record, and its dump meta names its head file."""
    bad = []
    iv = (pr or {}).get("interval")
    if not iv or not (iv[0] > 1.0):
        bad.append("the ratio of the permuted-label error to that of L2 has the interval %s: its lower end is not above 1.0" % (iv,))
    if (pr or {}).get("n_label_fields_differ") not in (0,):
        bad.append("the label fields of the permuted-label dump and the dump of L2 differ on %s rows" % (pr or {}).get("n_label_fields_differ"))
    if perm_facts is not None:
        f = perm_facts
        if "error" in f:
            bad.append("the permuted-label fit record is unreadable: %s" % f["error"])
        else:
            if f.get("permute_labels") is not True or "--permute-labels" not in (f.get("argv") or []):
                bad.append("the record of the permuted-label fit does not say permuted labels (flag %s)" % f.get("permute_labels"))
            if f.get("seed") != 2 or f.get("arm") != MSG.FIT_ARM or f.get("head") != MSG.FIT_HEAD:
                bad.append("the permuted-label fit is for seed %s arm %s head %s, not seed 2 arm D head H5" % (f.get("seed"), f.get("arm"), f.get("head")))
            if l2_facts is not None and ("error" in l2_facts or f.get("fit_clips") != l2_facts.get("fit_clips") or f.get("sel_clips") != l2_facts.get("sel_clips")):
                bad.append("the permuted-label fit does not read the fit clips and the selection clips of L2")
            hd = f.get("head_disk_sha256")
            if not (isinstance(hd, str) and len(hd) == 64) or hd != f.get("head_pt_sha256_record"):
                bad.append("the head file of the permuted-label fit on disk is not the one of its record")
            if other_heads is not None and hd in other_heads:
                bad.append("the head file of the permuted-label fit is the same file as another fit's")
            dm = perm_meta
            if not (isinstance(dm, dict) and dm.get("head_pt_sha256") == hd and dm.get("seed") == 2 and dm.get("permute_labels") is True):
                bad.append("the dump meta of the permuted-label fit does not name its head file, seed 2 and permuted labels")
    return {"PASS": not bad, "failures": bad, "values": {"ratio_perm_over_L2": pr}}


# ============================================================================================ controls g0 to g4
def ctl_g0(enc, disk):
    """g0 (lists): ENC_CONTROLS.json says g0 passes with each of the seven clauses of the encode kit present and passing, and the clauses read again from the files on disk now pass (the five of the card: every clip of the
    large list is eligible, none is an evaluation clip, the pilot's 500 lines are its first lines, the selection set is the pilot's 50 clips, the clips left out are listed with a reason; and the two the encode kit adds)."""
    bad = []
    if not isinstance(enc, dict):
        bad.append("no ENC_CONTROLS.json")
    else:
        g = (enc.get("controls") or {}).get("g0")
        if enc.get("pass_g0") is not True or not isinstance(g, dict) or g.get("pass") is not True:
            bad.append("ENC_CONTROLS.json does not read g0 PASS")
        for c in G0_CLAUSES:
            if not isinstance(g, dict) or not isinstance(g.get(c), dict) or g[c].get("pass") is not True:
                bad.append("ENC_CONTROLS.json: clause %s is missing or not PASS" % c)
    if not isinstance(disk, dict):
        bad.append("g0 could not be read again from disk now")
    else:
        for c in G0_CLAUSES:
            if not isinstance(disk.get(c), dict) or disk[c].get("pass") is not True:
                bad.append("now from disk: clause %s is not PASS" % c)
        if disk.get("pass") is not True:
            bad.append("now from disk: g0 is not PASS")
    return {"PASS": not bad, "failures": bad}


def ctl_g1(enc, identity):
    """g1 (tokens): ENC_CONTROLS.json says g1 passes with its four clauses present and passing; the cheap clause (three pilot clips encoded again against the pilot's tokens, 1e-3) is read again from disk now. The
    re-hash of every token file (about 49 GB) is not repeated here: the encode kit's own run is the record."""
    bad = []
    if not isinstance(enc, dict):
        bad.append("no ENC_CONTROLS.json")
    else:
        g = (enc.get("controls") or {}).get("g1")
        if enc.get("pass_g1") is not True or not isinstance(g, dict) or g.get("pass") is not True:
            bad.append("ENC_CONTROLS.json does not read g1 PASS")
        for c in G1_CLAUSES:
            if not isinstance(g, dict) or not isinstance(g.get(c), dict) or g[c].get("pass") is not True:
                bad.append("ENC_CONTROLS.json: clause %s is missing or not PASS" % c)
        if enc.get("pass_all") is not True:
            bad.append("ENC_CONTROLS.json does not read pass_all")
    if not isinstance(identity, dict) or identity.get("pass") is not True:
        bad.append("now from disk: the identity of the three pilot clips is not PASS")
    return {"PASS": not bad, "failures": bad}


def ctl_g2(rec, F):
    """g2 (fits): the gap-file record has g2 PASS, g2 computed again from the fit records and head files on disk now passes (MSG.g2_check on MSG.collect), and the fit records and head files are the files the record saw."""
    if not isinstance(rec, dict):
        return {"PASS": False, "failures": ["no gap-file record"]}
    bad = []
    if (rec.get("g2") or {}).get("PASS") is not True:
        bad.append("the record's own g2 is not PASS")
    now = MSG.g2_check(F)
    bad += ["now: " + x for x in now["failures"]]
    seen = rec.get("fits") or {}
    for kind in ("large", "mid", "old"):
        for k, f in (F.get(kind) or {}).items():
            s = (seen.get(kind) or {}).get(k) or {}
            if s.get("record_sha256") != f.get("record_sha256") or s.get("head_disk_sha256") != f.get("head_disk_sha256"):
                bad.append("%s %s: the fit record or head file is not the one the gap-file record saw" % (kind, k))
    return {"PASS": not bad, "failures": bad, "now_clauses": now.get("clauses")}


def ctl_g3(rec, gap_dir, eval_join_sha, F, now):
    """g3 (gap files): the record has g3 PASS with its six parts; the parts computed again from the dumps, the gap files and the dump metas on disk now pass (MSG.v0_check, build_check, meta_check, mean_check for both mean
    files, mid_binding_check); the gap files and dumps on disk are the files the record read; the record is for this evaluation join. now: the facts of the files read from disk (see g23_now)."""
    if not isinstance(rec, dict):
        return {"PASS": False, "failures": ["no gap-file record"]}
    bad = []
    g = rec.get("g3") or {}
    if g.get("PASS") is not True or sorted((g.get("parts") or {})) != sorted(G3_PARTS) or not all(v is True for v in (g.get("parts") or {}).values()):
        bad.append("the record's own g3 is not PASS with its six parts")
    if rec.get("card") != CARD:
        bad.append("the record is for card %s" % rec.get("card"))
    try:
        parts = {"v0": MSG.v0_check(now), "meta": MSG.meta_check(now, F), "build": MSG.build_check(now), "mean_large": MSG.mean_check(now["means"][MSG.MEAN_L]["facts"]),
                 "mean_mid": MSG.mean_check(now["means"][MSG.MEAN_M]["facts"]), "mid_binding": MSG.mid_binding_check(F)}
    except (KeyError, TypeError) as ex:
        return {"PASS": False, "failures": bad + ["the facts of g3 cannot be read from disk now: %s: %s" % (type(ex).__name__, ex)]}
    for k, v in parts.items():
        bad += ["now %s: %s" % (k, x) for x in v["failures"]]
    for nm in MSG.L_NAMES + [MSG.MEAN_L, MSG.MEAN_M]:
        fr = (rec.get("files") or {}).get(nm) or {}
        if (now["files"].get(nm) or {}).get("file_sha256") != fr.get("file_sha256") or not fr.get("file_sha256"):
            bad.append("%s: the gap file on disk is not the one of the record" % nm)
    for nm in MSG.L_NAMES:
        if (now["files"].get(nm) or {}).get("dump_sha256") != ((rec.get("files") or {}).get(nm) or {}).get("dump_sha256"):
            bad.append("%s: the dump on disk is not the one the record read" % nm)
    for nm, key in ((MSG.MEAN_L, "large_dumps"), (MSG.MEAN_M, "mid_dumps")):
        named = (((rec.get("files") or {}).get(nm)) or {}).get("dump_sha256") or {}
        for s in MSG.LARGE_SEEDS:
            if named.get("fit%d" % s) != (F[key]["s%d" % s] or {}).get("dump_sha256"):
                bad.append("%s: the dump of fit %d on disk is not the one the record read" % (nm, s))
    if eval_join_sha and rec.get("eval_join_sha256") != eval_join_sha:
        bad.append("the record is for another evaluation join")
    return {"PASS": not bad, "failures": bad, "now_parts": {k: v["PASS"] for k, v in parts.items()}}


WRAP_RE = re.compile(r"wrapper sha256 ([0-9a-f]{64})")


def ctl_g4(wrapper_sha16, check_text, hashes_text, run_shas):
    """g4 (wrapper): the wrapper file is 28bc31f21ecb4ccb, the pod check record of card ADD102 names it with a pass (last line of CHECK_WRAPPER_V_DONE.txt and the hash in CHECK_WRAPPER_V.hashes), and every run text of the
    ten rollouts names that wrapper hash (ten expected)."""
    lines = [x for x in (check_text or "").splitlines() if x.strip()]
    bad = []
    if wrapper_sha16 != WRAPPER_SHA:
        bad.append("the wrapper file is %s, not %s" % (wrapper_sha16, WRAPPER_SHA))
    if not lines or lines[-1].strip() != CHECK_LAST:
        bad.append("the last line of the pod check is %r" % (lines[-1].strip() if lines else None))
    if WRAPPER_SHA not in (hashes_text or ""):
        bad.append("the pod check hashes do not name %s" % WRAPPER_SHA)
    other = [k for k, v in run_shas.items() if v is not None and not v.startswith(WRAPPER_SHA)]
    if other:
        bad.append("run files with another wrapper hash: %s" % sorted(other))
    n_found = sum(1 for v in run_shas.values() if v is not None)
    if n_found != len(run_shas) or len(run_shas) != 10:
        bad.append("run files with a wrapper hash: %d of %d (ten expected)" % (n_found, len(run_shas)))
    return {"PASS": not bad, "failures": bad, "wrapper_sha16": wrapper_sha16, "pod_check_last_line": lines[-1].strip() if lines else None, "run_files_found": n_found, "run_files_expected": len(run_shas)}


# ============================================================================================ control g5 (rule logs), g6, g7, g8
def rows_of(logs):
    """-> (ci, j, present, g_src, c_src) over the logs (tick log and later-look log) of one rollout."""
    cols = {k: [] for k in ("ci", "j", "present", "g_src", "c_src")}
    for z in logs:
        for k in cols:
            cols[k].append(np.asarray(z[k]))
    return (np.concatenate(cols["ci"]).astype(np.int64), np.concatenate(cols["j"]).astype(np.int64), np.concatenate(cols["present"]).astype(bool),
            np.concatenate(cols["g_src"]).astype(np.float64), np.concatenate(cols["c_src"]).astype(np.float64))


def lead_multi(rows):
    """{(ci, j): [g_src, ...]} of the rows with a lead: a frame logged more than once keeps every value (they differ in the float noise of the reader)"""
    ci, j, pres, g, _ = rows
    m = {}
    for i in np.flatnonzero(pres):
        m.setdefault((int(ci[i]), int(j[i])), []).append(float(g[i]))
    return m


def gapfile_ref(path):
    """{(clip index, frame): gap or None} from the gap file as it lies on disk (a plain loop over its rows; None where it has no output)"""
    z = MSG.load_npz(path)
    ci, fr, g, h = np.asarray(z["clip_index"]), np.asarray(z["frame"]), np.asarray(z["gap"], np.float64), np.asarray(z["has_out"], bool)
    return {(int(ci[i]), int(fr[i])): (float(g[i]) if (h[i] and np.isfinite(g[i])) else None) for i in range(len(ci))}


def w2_one(rows, ref_gap, v0_multi, meta_expect=None, meta=None):
    """control g5 on one rollout. rows from rows_of; ref_gap {(ci, frame): gap or None} from the arm's gap file; v0_multi {(ci, j): [gap, ...]} of the control run V0 of card ADD102. On every row with a lead: where the file has an
    output the gap the rule received equals the gap of the file within 1e-3 m; where it has none it equals a gap of V0 at the same clip and frame within 1e-3 m, any of them. The closing input is non-zero on a positive share of
    the lead rows. The rule log meta names the gap file hash and the closing switch of the arm."""
    ci, j, pres, g, c = rows
    n_head = n_bad_head = n_noout = n_ref_missing = n_bad_ref = 0
    for i in np.flatnonzero(pres):
        k = (int(ci[i]), int(j[i]))
        hg = ref_gap.get(k) if ref_gap is not None else None
        if hg is not None:
            n_head += 1
            n_bad_head += int(abs(g[i] - hg) > TOL_W2)
        else:
            n_noout += 1
            refs = v0_multi.get(k)
            if not refs:
                n_ref_missing += 1
            else:
                n_bad_ref += int(min(abs(g[i] - r) for r in refs) > TOL_W2)
    n_lead = int(pres.sum())
    nz_lead = int((c[pres] != 0).sum())
    bad = []
    if n_lead == 0:
        bad.append("no rows with a lead")
    if ref_gap is not None and n_head == 0:
        bad.append("no lead row with a gap-file output")
    if n_bad_head:
        bad.append("%d lead rows with a gap-file output differ from the gap of the file by more than %g m" % (n_bad_head, TOL_W2))
    if n_ref_missing or n_bad_ref:
        bad.append("lead rows checked against V0: %d have no V0 row, %d differ from every V0 value by more than %g m" % (n_ref_missing, n_bad_ref, TOL_W2))
    if nz_lead == 0:
        bad.append("the closing input is 0 on every lead row")
    if meta_expect is not None:
        for k, v in meta_expect.items():
            if str((meta or {}).get(k)) != str(v):
                bad.append("rule log meta %s is %s, expected %s" % (k, (meta or {}).get(k), v))
    return {"PASS": not bad, "failures": bad, "n_rows": int(len(ci)), "n_lead_rows": n_lead, "n_lead_rows_with_head_output": n_head, "n_lead_rows_without_head_output": n_noout,
            "n_bad_head": n_bad_head, "n_ref_missing": n_ref_missing, "n_bad_ref": n_bad_ref, "n_lead_rows_closing_nonzero": nz_lead,
            "share_of_rule_steps_with_a_lead": (n_lead / len(ci)) if len(ci) else None, "share_of_lead_steps_with_nonzero_closing": (nz_lead / n_lead) if n_lead else None}


def ctl_g5(per_run):
    """g5 (each arm is what the card says): all ten rollouts (five arms, two model seeds) present and each passes w2_one; the shares of rule steps go into the payload (no name)."""
    want = sorted("%s_s%d" % (a, m) for a in ALLARMS for m in SEEDS)
    bad = []
    if sorted(per_run) != want:
        bad.append("rollouts read: %s, expected %s" % (sorted(per_run), want))
    for k, v in sorted(per_run.items()):
        bad += ["%s: %s" % (k, x) for x in v.get("failures", [])]
    shares = {k: {"share_of_rule_steps_with_a_lead": v.get("share_of_rule_steps_with_a_lead"), "share_of_lead_steps_with_nonzero_closing": v.get("share_of_lead_steps_with_nonzero_closing"),
                  "n_rows": v.get("n_rows"), "n_lead_rows": v.get("n_lead_rows"), "n_lead_rows_closing_nonzero": v.get("n_lead_rows_closing_nonzero")} for k, v in per_run.items()}
    return {"PASS": not bad and bool(per_run) and all(v.get("PASS") is True for v in per_run.values()), "failures": bad, "per_run_pass": {k: v.get("PASS") for k, v in sorted(per_run.items())}, "values": {"rollout_shares": shares}}


def ctl_g6(levels):
    """the reader recomputes C of RR, RLc, VD5c, F2, F3, F4 and E5c from the banked rollouts: 0.0954, 0.0000, 0.0169, 0.0138, 0.0185, 0.0123, 0.0138 to four decimals"""
    got = {k: levels.get(k) for k in C_EXPECTED}
    bad = ["%s: %s, expected %.4f" % (k, v, C_EXPECTED[k]) for k, v in got.items() if not (v is not None and round(float(v), 4) == C_EXPECTED[k])]
    return {"PASS": not bad, "failures": bad, "C": got, "expected": dict(C_EXPECTED)}


def ctl_g7(texts):
    """g7: the three lead_pytest.out files (kit_enc, kit_fit, kit_read) each read `N passed` with N > 0 and no `failed` and no `error`. texts {"kit_enc": text, "kit_fit": text, "kit_read": text}"""
    bad, per = [], {}
    for k in ("kit_enc", "kit_fit", "kit_read"):
        t = texts.get(k) or ""
        m = re.findall(r"(\d+) passed", t)
        ok = bool(m) and int(m[-1]) > 0 and not re.search(r"\d+ (failed|error)", t)
        per[k] = int(m[-1]) if (m and ok) else None
        if not ok:
            bad.append("%s: the test output does not read N passed without failed or error" % k)
    return {"PASS": not bad, "failures": bad, "passed": per}


def ctl_g8(grid_all_pass, strat_a, strat_b):
    ok = grid_all_pass is True and strat_a is True and strat_b is True
    return {"PASS": bool(ok), "grid_all_pass": grid_all_pass, "strategic_stage_A_pass": strat_a, "strategic_stage_B_pass": strat_b}


# ============================================================================================ the pod part
def load_logs(pull, run, m):
    d = os.path.join(pull, "I10%s_s%d" % (run, m), "ticks")
    names = ["rule_log_I10%s_s%d.npz" % (run, m), "rule_log_mid_I10%s_s%d.npz" % (run, m)]
    zs = []
    for nm in names:
        p = os.path.join(d, nm)
        if not os.path.isfile(p):
            raise SystemExit("REFUSED: no rule log %s" % p)
        with np.load(p, allow_pickle=False) as z:
            zs.append({k: z[k] for k in z.files})
    return zs


def meta_of(z):
    return {k: (z[k].item() if z[k].shape == () else None) for k in ("vis_gap_file_sha256", "vis_closing") if k in z}


def run_file_sha(pull, run, m):
    p = os.path.join(pull, "I10%s_s%d" % (run, m), "ticks", "run_I10%s_s%d.txt" % (run, m))
    if not os.path.isfile(p):
        return None
    mm = WRAP_RE.search(io.open(p, encoding="utf-8", errors="replace").read())
    return mm.group(1) if mm else None


def g5_from_files(a, rec):
    """-> per-run dict for ctl_g5. Arm X at model seed m: the gap file of the arm (VGAPL_s<S>, VGAPLE, VGAPSE3) as it lies on disk, its sha256 = the record's (named by the meta of the rule log), closing switch 1; V0 of card ADD102
    for the rows with no output."""
    per = {}
    for m in SEEDS:
        v0_multi = lead_multi(rows_of(load_logs(a.pull, V0_RUN, m)))
        for arm in ALLARMS:
            ref = gapfile_ref(os.path.join(a.gap_dir, GAPFILE[arm] + ".npz"))
            sha = (((rec or {}).get("files") or {}).get(GAPFILE[arm]) or {}).get("file_sha256")
            logs = load_logs(a.pull, RUN[arm], m)
            per["%s_s%d" % (arm, m)] = w2_one(rows_of(logs), ref, v0_multi, meta_expect={"vis_gap_file_sha256": sha, "vis_closing": 1}, meta=meta_of(logs[0]))
    return per


def measure(a, E, log):
    """the pod part: grid read (levels, pairs with intervals, families) of the ten runs (five arms, RR, RLc, VD5c, F2, F3, F4, E5c); then the strategic grid. -> dict"""
    H, Q3, RH = E
    g30 = R.jl(os.path.join(a.grid_dir, "grid_p30.json"))
    J = H.load_join()
    keys = R.grid_keys(g30, [str(c) for c in J["clip"]])
    runs = dict(RUN)
    runs.update(BANKED)
    runs.update(B102)
    runs.update(B104)
    runs.update(B105)
    for k, r in runs.items():
        R.ALL_RUNS.setdefault(r, ("RLc" if k == "RLc" else "RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    R.ALL_RUNS.setdefault(EC.SETTINGS["AGE"]["runs"]["RR"], ("RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    pairs_def = {}
    for x in XARMS:
        pairs_def[x + "-VS-RR"] = (RUN[x], BANKED["RR"])
        pairs_def[x + "-VS-RLc"] = (RUN[x], BANKED["RLc"])
        pairs_def[x + "-VS-VD5c"] = (RUN[x], B102["VD5c"])
        pairs_def[x + "-VS-" + SCALE_PAIR[x]] = (RUN[x], runs[SCALE_PAIR[x]])
    pairs_def["E3c-VS-RLc"] = (RUN["E3c"], BANKED["RLc"])                                  # falling behind against RLc for every arm (printed, no name)
    spread = "R0Hb20"
    run_list = [spread] + sorted(set(runs.values()))
    G = R.read_grid(a, E, "30", keys, run_list, spread, pairs_def, {}, log)
    lv = {k: {c: float(np.mean(G["levels"][r][c])) for c in ("C", "U", "F")} for k, r in runs.items()}
    lv_seed = {k: {c: [float(v) for v in np.asarray(G["levels"][r][c]).reshape(-1)[:2]] for c in ("C", "U", "F")} for k, r in runs.items()}
    pairs = {nm: R.ucf(p["headway"]) for nm, p in G["pairs"].items()}
    fams = {nm: {f_: R.reading(R.cells_of(p["family"]), f_) for f_ in R.NAMED} for nm, p in G["pairs"].items()}
    # the STRATEGIC family as the reader of card ADD102 reads it (engine strategic_grid with its own controls, then the pair read at tick t0 + 20 on the arcs of 30, 50 and 80 m):
    # G2, G3, G4 and GEc each against RR, RLc and the 500-clip arm of the same kind (F2, F3, F4, E3c)
    strat_runs = set(RUN.values()) | set(REF_RUNS.values()) | {runs[v] for v in SCALE_PAIR.values()}
    p_of = {"R0": 0}
    p_of.update({r: 30 for r in strat_runs})
    SR, cells_s, eid_s, S = R.strategic_grid(a, keys, sorted(p_of), p_of, log, {})
    sa, sb = bool(SR.get("stage_A_pass")), bool(SR.get("stage_B_pass"))
    strat = {"controls": dict({k: v for k, v in SR["controls"].items() if k != "s3"}, s3=SR["controls"]["s3"].get("pass")), "pairs": {}}
    if sa and sb:
        for x in XARMS:
            for ref, rn in list(REF_RUNS.items()) + [(SCALE_PAIR[x], runs[SCALE_PAIR[x]])]:
                pr = S.pair_read(cells_s, RUN[x], rn, eid_s, ticks=(2,))
                pos = pr["tick2"]["pos"]
                strat["pairs"][x + "-VS-" + ref] = {"A": RUN[x], "B": rn, "pos_30_50_80m_tick_t0_plus_20": [{"arc_m": arc, "mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "verdict": b["verdict"],
                                                                                                           "n_per_seed": b.get("n_per_seed")} for arc, b in zip(S.ARCS, pos)],
                                                    "reading": R.family_of([c["verdict"] for c in pos])}
    return {"levels": lv, "levels_per_seed": lv_seed, "pairs": pairs, "families": fams, "grid_all_pass": G["controls"].get("ALL_PASS") is True, "strategic": strat, "strategic_a": sa, "strategic_b": sb}


def g23_now(a, rec):
    """the facts g2 and g3 are computed from, read from disk now: the fit records, head files, lists and dump metas (MSG.collect), and the gap files, dumps and join (the build facts again: v0 stats, second-routine comparison
    of every single file, the means against a second routine). -> (F, now) with `now` in the layout of a gap-file record."""
    P = {"fit_root": a.fit_root, "large_list": a.large_list, "pilot_list": a.pilot_list, "eval_list": a.eval_list, "old_fits": a.old_fits, "mid_root": a.mid_root, "rec104": a.rec104}
    F = MSG.collect(P)
    J = MSG.load_npz(a.eval_join)
    DL = {s: MSG.load_npz(F["large_dumps"]["s%d" % s]["dump_file"]) for s in MSG.LARGE_SEEDS}
    DM = {s: MSG.load_npz(F["mid_dumps"]["s%d" % s]["dump_file"]) for s in MSG.MID_SEEDS}
    now = {"card": MSG.CARD, "files": {}, "dumps": {}, "means": {}}
    for s in MSG.LARGE_SEEDS:
        nm = "VGAPL_s%d" % s
        gp = os.path.join(a.gap_dir, nm + ".npz")
        arr = MSG.load_npz(gp)
        tab = MSG.frame_table(J, DL[s])
        m = F["large_dumps"]["s%d" % s]
        now["dumps"][nm] = {"dump_file": m["dump_file"], "dump_dir_name": m["dump_dir_name"], "dump_meta": m["dump_meta"], "e0": MSG.e0_stats(J, DL[s], tab, s, a.expect_clips)}
        f = MSG.measure_file(J, DL[s], arr, m["dump_file"], m["dump_dir_name"], gp)
        f["fit_seed"] = s
        now["files"][nm] = f
    for nm, dd in ((MSG.MEAN_L, DL), (MSG.MEAN_M, DM)):
        gp = os.path.join(a.gap_dir, nm + ".npz")
        now["means"][nm] = {"facts": MSG.mean_facts(J, [dd[s] for s in MSG.LARGE_SEEDS], MSG.load_npz(gp))}
        now["files"][nm] = {"file_sha256": sha_file(gp)}
    return F, now


def dumps_for_t0(F, a):
    """the eight dumps for the T0 read: O0, O1 (fit seeds 0, 1 of card ADD99), M2, M3, M4 (500-clip fits of card ADD104), L2, L3, L4 (large fits)"""
    D = {}
    for s in (0, 1):
        D["O%d" % s] = MSG.load_npz(os.path.join(a.old_dump_root, MSG.DUMP_DIR, "reader_s%d_score.npz" % s))
    for s in MSG.LARGE_SEEDS:
        D["M%d" % s] = MSG.load_npz(F["mid_dumps"]["s%d" % s]["dump_file"])
        D["L%d" % s] = MSG.load_npz(F["large_dumps"]["s%d" % s]["dump_file"])
    return D


def left_out_of(path):
    r = MSG.read_json(path) or {}
    return {"n_large_list": r.get("n_large_list"), "n_fit": r.get("n_fit"), "n_sel": r.get("n_sel"), "n_left_out": len(r.get("left_out") or []), "left_out_counts": r.get("left_out_counts"),
            "left_out": [{"clip": x.get("clip"), "reason": x.get("reason")} for x in (r.get("left_out") or [])]}


def fit_selection(rec, perm_facts):
    out = dict((rec or {}).get("fit_facts_for_reader") or {})
    if isinstance(perm_facts, dict) and "error" not in perm_facts:
        out["P2perm"] = {"fit_seed": perm_facts.get("seed"), "kept_epoch": perm_facts.get("kept_epoch"), "kept_sel_loss": perm_facts.get("kept_sel_loss"),
                         "n_fit_clips": len(perm_facts.get("fit_clips") or []), "n_sel_clips": len(perm_facts.get("sel_clips") or []), "head_pt_sha256": perm_facts.get("head_pt_sha256_record")}
    return out


def run_controls(a):
    rd = lambda p: io.open(p, encoding="utf-8").read() if p and os.path.isfile(p) else ""      # noqa: E731
    elog = a.out + ".engine.log"
    if os.path.exists(elog):
        raise SystemExit("REFUSED: %s exists" % elog)
    with io.open(elog, "w", encoding="utf-8", newline="\n") as fh, contextlib.redirect_stdout(fh):
        E = R.engine(a)
        m = measure(a, E, lambda s: print(s, flush=True))
    rec = MSG.read_json(a.gap_record)
    ejs = sha_file(a.eval_join) if a.eval_join and os.path.isfile(a.eval_join) else None
    enc = MSG.read_json(a.enc_controls)
    try:
        disk0 = ENCC.g0(a.eligible, a.eval_list, a.pilot_list, a.large_list, a.lists_record)
    except (OSError, ValueError, KeyError) as ex:
        disk0 = {"error": "%s: %s" % (type(ex).__name__, ex)}
    try:
        ids = ENCC.read_ids(os.path.join(a.pilot_run, "lists", "train_final_ids.txt"))[:3]
        ident = ENCC.g1_identity(ids, os.path.join(a.pilot_run, "tok", "D"), os.path.join(a.cs_root, "check_tok", "tok", "D"))
    except (OSError, ValueError) as ex:
        ident = {"error": "%s: %s" % (type(ex).__name__, ex)}
    F = now = None
    c2 = c3 = {"PASS": False, "failures": ["g2 and g3 could not be read from disk"]}
    t0 = five = prr = perm_facts = perm_meta = None
    try:
        F, now = g23_now(a, rec)
        c2 = ctl_g2(rec, F)
        c3 = ctl_g3(rec, a.gap_dir, ejs, F, now)
        J = MSG.load_npz(a.eval_join)
        D = dumps_for_t0(F, a)
        five = t0_five(J, D)
        t0 = t0_read(J, D)
        pd = os.path.join(a.fit_root, "dump", MSG.DUMP_DIR + "_perm")
        perm_dump = MSG.load_npz(os.path.join(pd, "reader_s2_score.npz"))
        perm_meta = MSG.read_json(os.path.join(pd, "reader_s2_meta.json"))
        perm_facts = MSG.fit_facts(MSG.fit_dir(os.path.join(a.fit_root, "fits"), 2, "_perm"))
        prr = perm_ratio(J, perm_dump, D["L2"])
    except (OSError, ValueError, KeyError, TypeError, SystemExit) as ex:
        c3 = {"PASS": False, "failures": c3["failures"] + ["reading from disk stopped: %s: %s" % (type(ex).__name__, ex)]}
    heads_on_record = {f.get("head_pt_sha256_record") for kind in ("large", "mid", "old") for f in (F or {}).get(kind, {}).values()} if F else None
    c9 = ctl_g9(five)
    c10 = ctl_g10(prr, perm_facts, perm_meta, (F or {}).get("large", {}).get("s2"), heads_on_record)
    c5 = ctl_g5(g5_from_files(a, rec))
    run_shas = {"%s_s%d" % (arm, s): run_file_sha(a.pull, RUN[arm], s) for arm in ALLARMS for s in SEEDS}
    wsha = sha_file(a.wrapper_file)[:16] if a.wrapper_file and os.path.isfile(a.wrapper_file) else None
    controls = {"g0": ctl_g0(enc, disk0), "g1": ctl_g1(enc, ident), "g2": c2, "g3": c3, "g4": ctl_g4(wsha, rd(a.wrapper_check), rd(a.wrapper_hashes), run_shas), "g5": c5,
                "g6": ctl_g6({k: m["levels"][k]["C"] for k in C_EXPECTED}), "g7": ctl_g7({"kit_enc": rd(a.tests_enc), "kit_fit": rd(a.tests_fit), "kit_read": rd(a.tests_read)}),
                "g8": ctl_g8(m["grid_all_pass"], m["strategic_a"], m["strategic_b"]), "g9": c9, "g10": c10}
    payload = {"levels": m["levels"], "levels_per_seed": m["levels_per_seed"], "pairs": m["pairs"], "families": m["families"], "strategic": m["strategic"], "family_note": FAM_REASON, "t0": t0,
               "t0_five_500_clip_rows": five, "fit_selection": fit_selection(rec, perm_facts), "large_list": left_out_of(a.lists_record),
               "unsafe_rate": {nm: p["U"] for nm, p in sorted(m["pairs"].items())}, "falling_behind_against_RLc": {x: m["pairs"][x + "-VS-RLc"]["F"] for x in ALLARMS}}
    return assemble_controls(controls, payload)


def fmt_iv(p):
    return "C %s U %s F %s" % (p["C"]["interval"], p["U"]["interval"], p["F"]["interval"])


def print_sections(payload, names):
    L = []
    t = payload["t0"]
    L.append("T0 gap error (m) on %s rows with a lead in the label and an output of all eight heads: %s" % (t["n_rows"], json.dumps(t["errors_m"], sort_keys=True)))
    L.append("T0 ratio mean(L2, L3, L4) / mean(M2, M3, M4) = %.4f, interval %s (%s draws, seed %s, %s); the name is %s" % (t["ratio"], t["interval"], t["n_boot"], t["seed"], t["interval_level"], names["T0"]))
    L.append("T0 second clause: each large fit below the smallest 500-clip error %.4f: %s" % (t["smallest_500_clip_error_m"], json.dumps(t["each_large_below_smallest_500_clip"], sort_keys=True)))
    L.append("T0 correlation of each head's gap with the label gap: " + json.dumps(t.get("correlation_with_label_gap"), sort_keys=True))
    L.append("kept epoch, selection loss and fit clips per fit: " + json.dumps(payload.get("fit_selection"), sort_keys=True))
    lo = payload.get("large_list") or {}
    L.append("large list: %s lines, %s clips left out: %s %s" % (lo.get("n_large_list"), lo.get("n_left_out"), json.dumps(lo.get("left_out_counts"), sort_keys=True), json.dumps(lo.get("left_out"))))
    pr, st, fm = payload["pairs"], (payload.get("strategic") or {}).get("pairs") or {}, payload.get("families") or {}
    for x in XARMS:
        row = [x]
        for ref in ("RR", "RLc", "VD5c", SCALE_PAIR[x]):
            nm = x + "-VS-" + ref
            if nm in pr:
                row.append("vs %s: %s" % (ref, fmt_iv(pr[nm])))
            row.append("families(%s) %s strategic %s" % (ref, json.dumps(fm.get(nm) or {}, sort_keys=True), (st.get(nm) or {}).get("reading")))
        L.append("   ".join(row))
    for ref in ("RR", "RLc"):
        for x in XARMS:
            if x + "-VS-" + ref not in st:
                L.append("strategic %s: not read" % (x + "-VS-" + ref))
    L.append("levels (pooled over the two model seeds): " + json.dumps(payload.get("levels"), sort_keys=True))
    L.append("levels per model seed: " + json.dumps(payload.get("levels_per_seed"), sort_keys=True))
    L.append("falling behind against RLc: " + json.dumps(payload.get("falling_behind_against_RLc"), sort_keys=True))
    L.append("rollout shares: " + json.dumps((payload.get("control_values") or {}).get("g5", {}).get("rollout_shares"), sort_keys=True))
    return L


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage", choices=("controls", "names"), required=True)
    for n in ("pull", "code", "grid-dir", "w", "tree", "wrapper-file", "wrapper-check", "wrapper-hashes", "controls", "eval-join", "gap-record", "gap-dir", "fit-root", "large-list", "lists-record", "pilot-list",
              "eval-list", "eligible", "old-fits", "old-dump-root", "mid-root", "rec104", "enc-controls", "tests-enc", "tests-fit", "tests-read", "pilot-run", "cs-root"):
        ap.add_argument("--" + n, default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--expect-clips", type=int, default=141)
    a = ap.parse_args(argv)
    a.n_boot_control = 300
    if os.path.exists(a.out):
        print("REFUSED: %s exists" % a.out)
        return 2
    if a.stage == "controls":
        a.cs_root = a.cs_root or WS + "/cropscale"
        a.pilot_run = a.pilot_run or WS + "/cropread/run1"
        a.gap_dir = a.gap_dir or a.cs_root + "/gapfiles"
        a.gap_record = a.gap_record or a.gap_dir + "/" + MSG.RECORD_NAME
        a.fit_root = a.fit_root or a.cs_root + "/fitroot"
        a.large_list = a.large_list or a.cs_root + "/lists/train_final.txt"
        a.lists_record = a.lists_record or a.cs_root + "/lists/LISTS_LARGE_RECORD.json"
        a.pilot_list = a.pilot_list or a.pilot_run + "/lists/train_final.txt"
        a.eval_list = a.eval_list or a.pilot_run + "/lists/eval141.txt"
        a.eligible = a.eligible or WS + "/cropread/enc_out/lists/train_eligible.txt"
        a.old_fits = a.old_fits or a.pilot_run + "/fits"
        a.old_dump_root = a.old_dump_root or a.pilot_run + "/dump"
        a.mid_root = a.mid_root or WS + "/visseed10k/fitroot"
        a.rec104 = a.rec104 or WS + "/visseed10k/gapfiles/VISSEED_GAPFILES_RECORD.json"
        a.enc_controls = a.enc_controls or a.cs_root + "/ENC_CONTROLS.json"
        a.tests_enc = a.tests_enc or a.cs_root + "/kit_enc/lead_pytest.out"
        a.tests_fit = a.tests_fit or a.cs_root + "/kit_fit/lead_pytest.out"
        a.tests_read = a.tests_read or a.cs_root + "/kit_read/lead_pytest.out"
        res = run_controls(a)
        io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o)))
        for k in CONTROLS:
            print("%s %s" % (k, "PASS" if (res["controls"].get(k) or {}).get("PASS") else "FAIL"))
        print("ZZ READ-CROPSCALE10K controls rc 0 controls_all_pass %s" % res["controls_all_pass"])
        return 0
    cj = json.load(io.open(a.controls, encoding="utf-8"))
    if cj.get("card") != CARD or cj.get("controls_all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != sorted(CONTROLS) \
            or any(v.get("PASS") is not True for v in cj.get("controls", {}).values()):
        print("REFUSED: the controls file is not all-pass with g0 to g10 present: no name is written")
        print("ZZ READ-CROPSCALE10K names rc 3 controls_all_pass False")
        return 3
    logp = a.log or (os.path.splitext(a.out)[0] + "_names.log")
    if os.path.exists(logp):
        print("REFUSED: %s exists" % logp)
        return 2
    names = all_names(cj["payload"])
    res = {"card": CARD, "tier": TIER, "controls_file": os.path.abspath(a.controls), "controls_sha256": sha_file(a.controls), "names": names, "h_note": h_sensitivity(cj["payload"]), "values": cj["payload"]}
    io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1))
    lines = [names[k] for k in sorted(names)] + print_sections(cj["payload"], names) + ["ZZ READ-CROPSCALE10K names rc 0 controls_all_pass True"]
    io.open(logp, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    for ln in lines:
        print(ln)
    return 0


def selftest():
    ok = True
    n = names_of("G2", [-0.05, -0.01], [0.0, 0.004], [-0.03, 0.001], [-0.02, 0.03], 0.0954, 0.0)               # H = 0.0477
    ok &= n["G2_CONTACT"].endswith("G2-CUTS-CONTACT") and n["G2_HALF"].endswith("WITHIN-HALF-THE-GAP") and n["G2_VD5C"].endswith("CONTACT-NOT-SEPARATED-FROM-VD5c") and n["G2_PRICE"].endswith("FALLING-BEHIND-NOT-SEPARATED")
    ok &= scale_name("CONTACT", [[-0.02, -0.01], [-0.03, -0.001], [-0.01, 0.02]]) == PRE + "-SCALE-LOWERS-CONTACT"
    ok &= scale_name("UNSAFE", [[-0.02, -0.01], [0.001, 0.02], [-0.01, 0.02]]) == PRE + "-SCALE-UNSAFE-NOT-SEPARATED"
    ok &= t0_name([0.8, 0.95], [3.0, 3.1, 3.2], [3.9, 4.0, 4.1, 3.95, 4.05]) == PRE_T0 + "-T0-GAP-ERROR-CUT"
    ok &= t0_name([0.8, 0.95], [3.0, 3.1, 3.95], [3.9, 4.0, 4.1, 3.95, 4.05]) == PRE_T0 + "-T0-GAP-ERROR-CUT-WITHIN-FIT-NOISE"
    ok &= t0_name([1.01, 1.2], [3.0, 3.1, 3.2], [3.9, 4.0]) == PRE_T0 + "-T0-MORE-GAP-ERROR" and t0_name([0.9, 1.0], [3.0, 3.1, 3.2], [3.9, 4.0]) == PRE_T0 + "-T0-GAP-ERROR-NOT-SEPARATED"
    ok &= ctl_g6({"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0169, "F2": 0.0138, "F3": 0.0185, "F4": 0.0123, "E5c": 0.0138})["PASS"]
    ok &= not ctl_g6({"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0169, "F2": 0.0138, "F3": 0.0186, "F4": 0.0123, "E5c": 0.0138})["PASS"]
    rng = np.random.RandomState(3)
    sn, sd = rng.rand(141) + 1.0, rng.rand(141) + 2.0
    ok &= boot_ratio(sn, sd, 500, 5) == boot_ratio(sn, sd, 500, 5)
    print("ZZ SELFTEST-READ-CROPSCALE10K ok: %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
