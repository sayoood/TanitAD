#!/usr/bin/env python3
"""TF-N7 navtest build (PREREG_TF_N7.md sections 3, 4, 6): the plans of the cascade arms from the banked TF-N5 N5-A plan / seam. CPU only. NO human fallback.

The residual set is the tokens where the BANKED N5-A plan is DAC = 0 under the builder tracked scorer (all 12,146 are scored: C-DACMAP6, C-SET). On each one the build runs the
frozen candidate generator of `n7_lib.LEVERS` (REPAIR-D / L1 iterates to cap 20 at margins 0.05 / 0.10 / 0.20, label and target time-stretch, lateral offsets, mild stretch + offset,
deep stretch), every iterate / grid point scored by the builder (tracked LQR scorer, nuPlan-DB rebuild), TWICE (C-DET), and the CASCADE of the card picks per arm:
first tier (T0 TF-N6, T1 L2, T2 stretch, T3 offsets, T4 deep stretch s >= 0.5) with a certified candidate, then the maximum builder PDMS, ties to the earliest generation order.
Certificate = the TF-N6 N6-T one (builder tracked scorer vs the N5-A builder score): DAC = 1, NC and DDC not worse, builder PDMS strictly better; TTC and C free.
Arms: N7-T3 (PRIMARY, tiers T0-T3), N7-T (SECONDARY, T0-T4), N7-X (deliberate regression: certificate inverted, must change 0 tokens).
The TF-N6 selection is re-run live on the same residual tokens (the pinned n6_nt_build capture + N6-T selector) so that C-N6SAME compares against the N6 pipeline itself.

  reach   python n7_nt_build.py reach [--real --fixture <json> --out <json>]    card tables read back (LEVERS, TIERS, certificate, grids, flag audit), the three selectors
          (read back + behavioural signature on synthetic candidate sets), the two N6 repair objects; --real: every tier executes on a real NAVTRAIN token.
  derive  python n7_nt_build.py derive --route cache|navtrain --n5a-plans <npz> --n2r-plans <npz> --rolls "<RJ5 rows glob>" --out <shard.jsonl> --shard i/N
              [--mutate none|strict|nc_drop|ddc_drop|pdms_drop|dac_inverted|tie_latest|tier_reverse|floor_off|t4_in_t3|drop_T1|drop_T0|levers_off --smoke-allow-mutation]
  merge   python n7_nt_build.py merge ... -> n7_plans_{N7T3,N7T,N7X}.npz, seams refe_tfn7_{n7t3,n7t,n7x}.npz, n7_build_report.json (C-COVB, C-DACMAP6, C-DERIV, C-LABELEGO, C-SET,
          C-REPLAY, C-DET, C-SELECT, C-CERT-B7, C-ORDER, C-X0, C-SEAMFN, C-UNCH, C-N6SAME, C-REACH). Prints ZZN7_BUILD PASS|FAIL.
  unch    python n7_nt_build.py unch --out-dir <dir> ...   C-UNCH again on the files ON DISK."""
from __future__ import annotations

import argparse
import functools
import glob
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
os.environ["N4_RUN"] = os.environ.get("N4_RUN") or RUN_DIR
for _p in (os.path.join(RUN_DIR, "n6"), os.path.join(RUN_DIR, "n5"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import n5_env  # noqa: E402,F401  (BLAS threads = 1 + the pinned tree on sys.path, before numpy)
import numpy as np  # noqa: E402

import n4_nt_build as B4  # noqa: E402  (the PINNED A1 builder: load_rolls, db_ego_fn, ctx_stream, make_repair("N4D"), check_arm)
import n5_nt_build as B5  # noqa: E402  (the PINNED TF-N5 builder: make_repair / check_arm for L1, _npz, read_rows)
import n6_nt_build as B6  # noqa: E402  (the PINNED TF-N6 build: the N6-T certificate CARD6, capture, selector, load_harness_csv)
import n7_lib as L  # noqa: E402  (the generator levers; chain copy of n7_gen/n7_lib.py with the n5_env import)
import n7_select as S  # noqa: E402  (TIERS, describe, the dev cascade)

sha1_int, sha256, load_rolls, done_tokens, _npz, read_rows = B4.sha1_int, B4.sha256, B4.load_rolls, B4.done_tokens, B5._npz, B5.read_rows
load_harness_csv, cert_clauses, cert_ok = B6.load_harness_csv, B6.cert_clauses, B6.cert_ok

# ------------------------------------------------------------------------------------------------------------------------- the card tables (PREREG_TF_N7.md section 3)
# Written out by hand from the card. They are NOT imported from n7_lib / n7_select: C-SELECT reads the generator's own tables back against these.
_S_GRID = (0.95, 0.90, 0.85, 0.80)
_OFF_DY = (0.05, -0.05, 0.10, -0.10, 0.20, -0.20, 0.30, -0.30, 0.50, -0.50, 0.75, -0.75, 1.0, -1.0)
_OFF_RAMP = (0.2, 1.0, 2.0)
LEVERS_CARD = {
    "D_base": {"kind": "D", "cap": 5, "margin": 0.05}, "L1_base": {"kind": "L1", "cap": 5, "margin": 0.05},
    "D_c5_m05": {"kind": "D", "cap": 5, "margin": 0.05}, "L1_c5_m05": {"kind": "L1", "cap": 5, "margin": 0.05},
    "D_c5_m10": {"kind": "D", "cap": 5, "margin": 0.10}, "L1_c5_m10": {"kind": "L1", "cap": 5, "margin": 0.10},
    "D_c5_m20": {"kind": "D", "cap": 5, "margin": 0.20}, "L1_c5_m20": {"kind": "L1", "cap": 5, "margin": 0.20},
    "D_c10_m05": {"kind": "D", "cap": 10, "margin": 0.05}, "L1_c10_m05": {"kind": "L1", "cap": 10, "margin": 0.05},
    "D_c10_m10": {"kind": "D", "cap": 10, "margin": 0.10}, "L1_c10_m10": {"kind": "L1", "cap": 10, "margin": 0.10},
    "D_c10_m20": {"kind": "D", "cap": 10, "margin": 0.20}, "L1_c10_m20": {"kind": "L1", "cap": 10, "margin": 0.20},
    "D_c20_m05": {"kind": "D", "cap": 20, "margin": 0.05}, "L1_c20_m05": {"kind": "L1", "cap": 20, "margin": 0.05},
    "D_c20_m10": {"kind": "D", "cap": 20, "margin": 0.10}, "L1_c20_m10": {"kind": "L1", "cap": 20, "margin": 0.10},
    "D_c20_m20": {"kind": "D", "cap": 20, "margin": 0.20}, "L1_c20_m20": {"kind": "L1", "cap": 20, "margin": 0.20},
    "STR_label": {"kind": "stretch_label", "s": list(_S_GRID)},
    "STR_target": {"kind": "stretch_target", "s": list(_S_GRID)},
    "OFF_label": {"kind": "offset_label", "dy": list(_OFF_DY), "ramp": list(_OFF_RAMP)},
    "STR_deep": {"kind": "stretch_label", "s": [0.70, 0.60, 0.50, 0.40, 0.30, 0.20]},
    "STR_OFF": {"kind": "stretch_offset_label", "s": [0.90, 0.80], "dy": list(_OFF_DY), "ramp": list(_OFF_RAMP)},
}
TIERS_CARD = [
    ("T0_N6", [("D_c20_m05", 5), ("L1_c20_m05", 5)]),
    ("T1_L2", [("D_c20_m05", 20), ("D_c20_m10", 20), ("D_c20_m20", 20), ("L1_c20_m05", 20), ("L1_c20_m10", 20), ("L1_c20_m20", 20)]),
    ("T2_STR", [("STR_label", None), ("STR_target", None)]),
    ("T3_OFF", [("OFF_label", None), ("STR_OFF", None)]),
    ("T4_DEEP", [("STR_deep", None)]),
]
FLOOR = 0.5                                                                       # T4: s >= 0.5 (0.4, 0.3, 0.2 are generated by the lever and never selectable)
# the TF-N6 PI-approved tolerance certificate, literal; C-CERT-B7 requires == B6.CARD6["N6T"] == n7_lib.CERT_TOL
CERT = {"dac": 1.0, "nc": True, "ddc": True, "ttc": False, "c": False, "pdms_gt": True}
CERT_X = dict(CERT, dac=0.0)
ARM_SPEC = {                                                                      # arm -> (cert, number of tiers, floor)
    "N7T3": (CERT, 4, None),
    "N7T": (CERT, 5, FLOOR),
    "N7X": (CERT_X, 5, FLOOR),
}
ARMS = tuple(ARM_SPEC)
CARD7 = {arm: {"cert": dict(c), "tiers": [(n, list(map(tuple, lv))) for n, lv in TIERS_CARD[:nt]], "floor": fl, "tie": "earliest"} for arm, (c, nt, fl) in ARM_SPEC.items()}
RUN_LEVERS = tuple(sorted({lv for _n, lvs in TIERS_CARD for lv, _c in lvs}))     # the levers the derive runs (cap 5 / 10 are a prefix k <= cap of the cap 20 run)
SEAM_LABEL = {"N7T3": "refe_tfn7_n7t3", "N7T": "refe_tfn7_n7t", "N7X": "refe_tfn7_n7x"}
OLD_SEAM = "refe_tfn5_n5a"
N_CDERIV, CDERIV_BAR_M, N_RESID_EXPECT = 200, 1e-9, 20
EPS = 1e-12
MUTATIONS = ("none", "strict", "nc_drop", "ddc_drop", "pdms_drop", "dac_inverted", "tie_latest", "tier_reverse", "floor_off", "t4_in_t3", "drop_T1", "drop_T0", "levers_off")
SEL_MUTATIONS = ("strict", "nc_drop", "ddc_drop", "pdms_drop", "dac_inverted", "tie_latest", "tier_reverse", "floor_off", "t4_in_t3", "drop_T1", "drop_T0")
D_ARM, L1_ARM = "N4D", "N5A"                                                     # the banked repair objects of the live TF-N6 re-run
TIER_NAMES = tuple(n for n, _ in TIERS_CARD)


def _norm(x):
    """JSON-normalised copy (tuples -> lists), for readbacks"""
    return json.loads(json.dumps(x))


# ----------------------------------------------------------------------------------------------------------------------------------- the cascade selector
def card_cascade(row: dict, *, spec: dict, tiers: list, floor, tie: str = "earliest"):
    """row = {'ref': builder score of the N5-A plan, 'levers': {lever: [{'k', 'plan', 'm'} ...]}}. The first tier with a certified candidate; in it the maximum builder PDMS, ties to
    the EARLIEST generation order (levers in the tier's order, then k). Returns {'tier','lever','k','param','plan','m'} or None. T4 candidates below `floor` are never taken."""
    ref = row["ref"]
    for name, levers in tiers:
        best = None
        for lv, cap in levers:
            for x in row["levers"].get(lv, []):
                if cap is not None and x["k"] > cap:
                    continue
                p = S.describe(lv, x["k"])
                if name == "T4_DEEP" and floor is not None and p["s"] < floor:
                    continue
                if not cert_ok(x["m"], ref, spec):
                    continue
                if best is None or x["m"]["pdms"] > best["m"]["pdms"] or (tie == "latest" and x["m"]["pdms"] == best["m"]["pdms"]):
                    best = {"tier": name, "lever": lv, "k": x["k"], "param": p, "plan": x["plan"], "m": x["m"]}
        if best is not None:
            return best
    return None


def make_selector(arm: str, mutation: str = "none"):
    c = CARD7[arm]
    spec, tiers, floor, tie = dict(c["cert"]), [(n, list(lv)) for n, lv in c["tiers"]], c["floor"], c["tie"]
    if mutation == "strict":
        spec.update(ttc=True, c=True)
    elif mutation == "nc_drop":
        spec["nc"] = False
    elif mutation == "ddc_drop":
        spec["ddc"] = False
    elif mutation == "pdms_drop":
        spec["pdms_gt"] = False
    elif mutation == "dac_inverted":
        spec["dac"] = 0.0 if c["cert"]["dac"] == 1.0 else 1.0
    elif mutation == "tie_latest":
        tie = "latest"
    elif mutation == "tier_reverse":
        tiers = tiers[::-1]
    elif mutation == "floor_off":
        floor = None
    elif mutation == "t4_in_t3":
        tiers = [(n, list(lv)) for n, lv in TIERS_CARD]
        floor = FLOOR
    elif mutation == "drop_T1":
        tiers = [t for t in tiers if t[0] != "T1_L2"]
    elif mutation == "drop_T0":
        tiers = [t for t in tiers if t[0] != "T0_N6"]
    return functools.partial(card_cascade, spec=spec, tiers=tiers, floor=floor, tie=tie)


def sel_readback(fn) -> dict:
    kw = fn.keywords
    return _norm({"cert": kw["spec"], "tiers": kw["tiers"], "floor": kw["floor"], "tie": kw["tie"]})


def _cd(k, **m):
    base = {"nc": 1.0, "dac": 1.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0, "ep": 0.5, "pdms": 0.5}
    base.update(m)
    return {"k": k, "plan": None, "m": base}


REF0 = {"nc": 1.0, "dac": 0.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0, "ep": 0.0, "pdms": 0.0}
REF_NC = dict(REF0, nc=0.5)
SCENARIOS = {                              # name -> (ref, {lever: candidates}); every clause, every tier, the cap-5 prefix, the floor, the tie rule move at least one entry
    "t0_beats_later_tiers": (REF0, {"L1_c20_m05": [_cd(2, pdms=0.4)], "D_c20_m10": [_cd(3, pdms=0.9)], "OFF_label": [_cd(5, pdms=0.95)]}),
    "t1_when_t0_empty": (REF0, {"D_c20_m10": [_cd(3, pdms=0.6)], "L1_c20_m20": [_cd(2, pdms=0.8)], "STR_label": [_cd(1, pdms=0.99)]}),
    "cap5_prefix": (REF0, {"D_c20_m05": [_cd(6, pdms=0.6)], "STR_label": [_cd(1, pdms=0.99)]}),
    "tie_earliest_lever_order": (REF0, {"D_c20_m10": [_cd(4, pdms=0.5)], "D_c20_m05": [_cd(9, pdms=0.5)]}),
    "tie_earliest_k": (REF0, {"L1_c20_m20": [_cd(3, pdms=0.5), _cd(5, pdms=0.5)]}),
    "t2_stretch": (REF0, {"STR_target": [_cd(1, pdms=0.7)], "OFF_label": [_cd(2, pdms=0.9)]}),
    "t3_offsets": (REF0, {"OFF_label": [_cd(8, pdms=0.6)], "STR_OFF": [_cd(3, pdms=0.8)], "STR_deep": [_cd(1, pdms=0.99)]}),
    "t4_only": (REF0, {"STR_deep": [_cd(1, pdms=0.6)]}),
    "t4_floor_below": (REF0, {"STR_deep": [_cd(3, pdms=0.9)]}),
    "t4_floor_at": (REF0, {"STR_deep": [_cd(2, pdms=0.5), _cd(4, pdms=0.9)]}),
    "t4_two_s": (REF0, {"STR_deep": [_cd(0, pdms=0.5), _cd(1, pdms=0.7), _cd(2, pdms=0.6)]}),
    "ttc_c_free": (REF0, {"D_c20_m05": [_cd(1, ttc=0.0, c=0.0, pdms=0.4)]}),
    "nc_worse": (REF0, {"D_c20_m05": [_cd(1, nc=0.0, pdms=0.4)]}),
    "nc_equal_below_one": (REF_NC, {"D_c20_m05": [_cd(1, nc=0.5, pdms=0.4)]}),
    "ddc_worse": (REF0, {"D_c20_m05": [_cd(1, ddc=0.5, pdms=0.4)]}),
    "pdms_not_improved": (dict(REF0, pdms=0.4), {"D_c20_m05": [_cd(1, pdms=0.4)]}),
    "dac0_candidate": (REF0, {"D_c20_m05": [_cd(1, dac=0.0, pdms=0.2)]}),
    "nothing": (REF0, {"D_c20_m05": [_cd(1, dac=0.0, pdms=0.0)], "STR_deep": [_cd(1, dac=0.0, pdms=0.0)]}),
}


def selector_signature(fn) -> dict:
    out = {}
    for name, (ref, lev) in SCENARIOS.items():
        r = fn({"ref": ref, "levers": lev})
        out[name] = None if r is None else [r["tier"], r["lever"], r["k"]]
    return out


def check_selector(arm: str, fn) -> dict:
    """read-back + behavioural signature compared with the card (arm); ok False with the reasons on any difference"""
    rb, bad = sel_readback(fn), []
    want = _norm({"cert": CARD7[arm]["cert"], "tiers": CARD7[arm]["tiers"], "floor": CARD7[arm]["floor"], "tie": CARD7[arm]["tie"]})
    for k, v in want.items():
        if rb.get(k) != v:
            bad.append(f"{k}: built {rb.get(k)!r} != card {v!r}")
    if getattr(fn, "func", None) is not card_cascade:
        bad.append("selector function is not n7_nt_build.card_cascade")
    sg, ref = selector_signature(fn), selector_signature(make_selector(arm))
    diff = [k for k in ref if sg.get(k) != ref[k]]
    if diff:
        bad.append("behavioural signature differs from the card selector's on " + ",".join(diff))
    return {"arm": arm, "ok": not bad, "bad": bad, "readback": rb, "signature_sha1": hashlib.sha1(json.dumps(sg, sort_keys=True).encode()).hexdigest()}


# ------------------------------------------------------------------------------------------------------------------------------- card readback (C-SELECT, C-CERT-B7)
FLAG_PLUMBING = {"help", "route", "n5a_plans", "n5a_seam", "n5a_report", "n2r_plans", "expect_residual_json", "replay_expect_json", "resid_tokens", "rolls", "metric_cache_root",
                 "navtrain_tokens", "out", "shard", "rows", "harness_csv_n5a", "reach_json", "expect_residual", "real", "fixture", "out_dir", "n6_report", "n6t_plans",
                 "mutate", "smoke_allow_mutation", "mode"}


def audit_flags(parser, argv_names: set | None = None) -> list:
    """Every optional flag of the build CLI that is not file / routing plumbing is a lever; a flag whose DEFAULT disables something the card names (a floor of None, a tier list of
    none, a cap of 0 ...) is only safe if the arm's argv names it. The build has NO lever flags (the generator and the cascade are card constants), so any non-plumbing flag is
    returned as a finding (with its default and whether the chain's argv names it)."""
    argv_names = argv_names or set()
    out = []
    for act in parser._actions:                                                    # noqa: SLF001
        if act.dest in FLAG_PLUMBING or not act.option_strings:
            continue
        named = any(o in argv_names for o in act.option_strings)
        out.append({"flag": act.option_strings[0], "default": act.default, "named_in_argv": named, "disabling_default": act.default in (None, False, 0, 0.0, "", "none", ())})
    return out


def check_card7(parser=None, argv_names: set | None = None) -> dict:
    """the generator's own tables read back against the card tables above; the certificate three ways; the grids of n7_select against n7_lib; the flag audit"""
    bad = []
    if _norm(L.LEVERS) != _norm(LEVERS_CARD):
        diff = sorted(k for k in set(L.LEVERS) | set(LEVERS_CARD) if _norm(L.LEVERS.get(k)) != _norm(LEVERS_CARD.get(k)))
        bad.append("n7_lib.LEVERS != card table on " + ",".join(diff[:6]))
    if _norm(S.TIERS) != _norm(TIERS_CARD):
        bad.append("n7_select.TIERS != card tiers")
    if set(RUN_LEVERS) - set(L.LEVERS):
        bad.append("a card tier names a lever that n7_lib does not define")
    n6t = {k: B6.CARD6["N6T"][k] for k in CERT}
    if _norm(n6t) != _norm(CERT):
        bad.append(f"card certificate {CERT} != CARD6 N6T {n6t}")
    if _norm(L.CERT_TOL) != _norm(CERT):
        bad.append(f"card certificate {CERT} != n7_lib.CERT_TOL {L.CERT_TOL}")
    if tuple(B6.CARD6["N6T"]["order"]) != ("teacher",) or B6.CARD6["N6T"]["tie"] != "earliest":
        bad.append("CARD6 N6T order/tie is not teacher-only / earliest")
    # the grids n7_select.describe() decodes must be the grids n7_lib scores (a divergence would label the wrong parameter)
    pairs = [("S_GRID", S.S_GRID, L.LEVERS["STR_label"]["s"]), ("S_DEEP", S.S_DEEP, L.LEVERS["STR_deep"]["s"]), ("OFF_DY", S.OFF_DY, L.LEVERS["OFF_label"]["dy"]),
             ("OFF_RAMP", S.OFF_RAMP, L.LEVERS["OFF_label"]["ramp"]), ("S_MILD", S.S_MILD, L.LEVERS["STR_OFF"]["s"]), ("OFF_DY(STR_OFF)", S.OFF_DY, L.LEVERS["STR_OFF"]["dy"])]
    for nm, a, b in pairs:
        if _norm(a) != _norm(b):
            bad.append(f"n7_select.{nm} != the grid n7_lib scores")
    # certificate behaviour: n7_lib.cert_clauses, B6.cert_clauses and the card certificate agree on a clause grid
    import itertools
    ref = dict(REF0, pdms=0.3)
    for dac, nc, ddc, ttc, c, pdms in itertools.product((0.0, 1.0), (0.0, 0.5, 1.0), (0.5, 1.0), (0.0, 1.0), (0.0, 1.0), (0.2, 0.3, 0.4)):
        m = {"dac": dac, "nc": nc, "ddc": ddc, "ttc": ttc, "c": c, "ep": 0.5, "pdms": pdms}
        if B6.cert_ok(m, ref, B6.CARD6["N6T"]) != L.cert_ok(m, ref, L.CERT_TOL) or B6.cert_ok(m, ref, B6.CARD6["N6T"]) != cert_ok(m, ref, CERT):
            bad.append("certificate behaviour differs between n6 / n7_lib / card on " + json.dumps(m))
            break
    for arm in ARMS:
        r = check_selector(arm, make_selector(arm))
        if not r["ok"]:
            bad += [f"selector {arm}: {b}" for b in r["bad"]]
    fl = audit_flags(parser, argv_names) if parser is not None else []
    flags_bad = [f for f in fl if f["disabling_default"] and not f["named_in_argv"]]
    for f in flags_bad:
        bad.append(f"flag {f['flag']} has a disabling default ({f['default']!r}) and is not named in the arm's argv")
    return {"ok": not bad, "bad": bad, "levers": len(L.LEVERS), "tiers": [n for n, _ in S.TIERS], "floor": FLOOR, "flag_audit": fl}


# --------------------------------------------------------------------------------------------------------------------------------------- candidate generation
def gen_levers(cx, t20, label, ref) -> dict:
    """every lever of the cascade on one token -> {lever: [{'k', 'plan' (ndarray), 'm'}]} in generation order; the SAME function (n7_lib.run_lever) the development ran"""
    out = {}
    for name in RUN_LEVERS:
        out[name] = [{"k": x["k"], "plan": np.asarray(x["plan"], float), "m": x["m"]} for x in L.run_lever(name, L.LEVERS[name], cx, t20, label, ref)]
    return out


def levers_equal(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(len(a[n]) == len(b[n]) and all(x["k"] == y["k"] and np.array_equal(np.asarray(x["plan"]), np.asarray(y["plan"])) and x["m"] == y["m"]
                                                                  for x, y in zip(a[n], b[n])) for n in a)


def levers_json(lev: dict) -> dict:
    return {n: [{"k": x["k"], "plan": np.asarray(x["plan"]).tolist(), "m": x["m"]} for x in c] for n, c in lev.items()}


def choose_all(row: dict, sels: dict, cx, plan_dtype=float) -> dict:
    """per arm: the cascade choice with its plan, the recorded score and a SEPARATE re-score of the chosen plan (C-CERT-B7)"""
    out = {}
    for arm, fn in sels.items():
        c = fn(row)
        out[arm] = None if c is None else {"tier": c["tier"], "lever": c["lever"], "k": c["k"], "param": c["param"], "plan": np.asarray(c["plan"], plan_dtype).tolist(),
                                           "m": c["m"], "m_recheck": cx.score([np.asarray(c["plan"], float)])[0][0]}
    return out


def make_objects(mutation: str = "none"):
    sels = {arm: make_selector(arm, mutation if mutation in SEL_MUTATIONS else "none") for arm in ARMS}
    return sels


def n6_live(cx, t20, ref, fd, fl, sel6) -> dict:
    """the TF-N6 pipeline itself on this token: the pinned capture (REPAIR-D at A1 settings, L1 at the TF-N5 frozen parameters) + the pinned N6-T selector"""
    cands, repl = B6.capture(cx, t20, ref, fd, fl)
    c = sel6(cands, ref)
    return {"repl": {k: v["returns_plan"] for k, v in repl.items()}, "chosen": None if c is None else {"src": c["src"], "k": c["k"], "plan": np.asarray(c["plan"]).tolist(), "m": c["m"]},
            "n_cands": len(cands)}


def token_row(cx, t20, label, ref, sels, fd, fl, sel6, mutation="none") -> dict:
    """the residual-token record: both generator runs (C-DET), the arms' choices, the live N6 selection. Shared by the derive, the chain's reach stage and the smoke"""
    rec = {}
    lev = {} if mutation == "levers_off" else gen_levers(cx, t20, label, ref)
    lev2 = {} if mutation == "levers_off" else gen_levers(cx, t20, label, ref)         # C-DET: the whole generator run a SECOND time
    rec["det_equal"] = bool(lev) and levers_equal(lev, lev2)
    rec["levers"] = levers_json(lev)
    row = {"ref": ref, "levers": lev}
    rec["chosen"] = choose_all(row, sels, cx)
    rec["n6_live"] = n6_live(cx, t20, ref, fd, fl, sel6)
    return rec


# -------------------------------------------------------------------------------------------------------------------------------------------------------- reach
def cmd_reach_real_instances(fixture: str) -> dict:
    """C-REACH: EVERY tier executes on a real NAVTRAIN token through the build's own code path (gen_levers + the card selector): the token's chosen tier / lever / k / parameter / plan
    equal the development's (fixture), a selector whose tier list lacks that tier no longer chooses it (a mutated tier list DROPS its tokens), N7-T3 equals N7-T except on the T4
    token (where it chooses nothing) and N7-X chooses nothing."""
    import n4_geom as G
    fx = json.load(open(fixture, encoding="utf-8"))
    sels = make_objects()
    out = {"fixture_sha256": sha256(fixture), "instances": {}, "tiers_executed": [], "ok": True}
    for tier in TIER_NAMES:
        inst = fx["instances"][tier]
        cx = None
        for _lg, _tk, c in G.contexts({inst["log"]: [inst["token"]]}):
            cx = c
        t20, label = np.asarray(inst["t20"], float), np.asarray(inst["label"], float)
        ref = cx.score([label])[0][0]
        lev = gen_levers(cx, t20, label, ref)
        row = {"ref": ref, "levers": lev}
        ch = {arm: sels[arm](row) for arm in ARMS}
        want = inst["expect"]
        got = ch["N7T"]
        rec = {"token": inst["token"], "tier_expected": tier, "tier_got": None if got is None else got["tier"], "dac_ref": float(ref["dac"])}
        rec["match"] = bool(got is not None and (got["tier"], got["lever"], got["k"]) == (want["tier"], want["lever"], want["k"]) and got["param"] == want["param"])
        rec["plan_sha256_match"] = bool(got is not None and hashlib.sha256(np.ascontiguousarray(got["plan"], float).tobytes()).hexdigest() == inst["plan_sha256"])
        full = sels["N7T"]
        dropped = functools.partial(card_cascade, spec=full.keywords["spec"], tiers=[t for t in full.keywords["tiers"] if t[0] != tier], floor=FLOOR, tie="earliest")
        d = dropped(row)
        rec["dropped_tier_removes_it"] = bool(d is None or d["tier"] != tier)                  # a tier list without this tier no longer chooses at it (the token is dropped from it)
        rec["dropped_plan_changes"] = bool(d is None or not np.array_equal(d["plan"], got["plan"]))     # informational: a later tier may reach the same plan
        rec["t3_expected"] = None if tier == "T4_DEEP" else [want["tier"], want["lever"], want["k"]]
        t3 = ch["N7T3"]
        rec["t3_ok"] = bool((tier == "T4_DEEP" and t3 is None) or (tier != "T4_DEEP" and t3 is not None and [t3["tier"], t3["lever"], t3["k"]] == rec["t3_expected"]))
        rec["x_none"] = ch["N7X"] is None
        rec["base_dac0"] = bool(ref["dac"] == 0.0)
        if tier == "T0_N6":                                                                  # determinism on the cheapest token (the derive's C-DET covers every residual token)
            rec["deterministic"] = levers_equal(lev, gen_levers(cx, t20, label, ref))
        rec["ok"] = bool(rec["match"] and rec["plan_sha256_match"] and rec["dropped_tier_removes_it"] and rec["t3_ok"] and rec["x_none"] and rec["base_dac0"]
                         and rec.get("deterministic", True))
        out["instances"][tier] = rec
        if rec["ok"]:
            out["tiers_executed"].append(tier)
        out["ok"] = out["ok"] and rec["ok"]
    out["ok"] = bool(out["ok"] and out["tiers_executed"] == list(TIER_NAMES))
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("reach", "derive", "merge", "unch"))
    ap.add_argument("--route", choices=("cache", "navtrain"), default="cache")
    ap.add_argument("--n5a-plans")
    ap.add_argument("--n5a-seam")
    ap.add_argument("--n5a-report", default="")
    ap.add_argument("--n2r-plans")
    ap.add_argument("--expect-residual-json", default="")
    ap.add_argument("--replay-expect-json", default="")
    ap.add_argument("--resid-tokens", default="")
    ap.add_argument("--rolls", default="")
    ap.add_argument("--metric-cache-root", default="")
    ap.add_argument("--navtrain-tokens", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--mutate", choices=MUTATIONS, default="none")
    ap.add_argument("--smoke-allow-mutation", action="store_true")
    ap.add_argument("--rows", nargs="*", default=[])
    ap.add_argument("--harness-csv-n5a", default="")
    ap.add_argument("--reach-json", default="")
    ap.add_argument("--n6-report", default="")
    ap.add_argument("--n6t-plans", default="")
    ap.add_argument("--expect-residual", type=int, default=N_RESID_EXPECT)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--fixture", default=os.path.join(HERE, "fixture_navtrain_tiers.json"))
    ap.add_argument("--out-dir", default="")
    return ap


def cmd_reach(a) -> int:
    fd = B4.make_repair(D_ARM)
    fl = B5.make_repair(B5.CARD[L1_ARM])
    reps = [dict(B4.check_arm(D_ARM, fd), what="REPAIR-D at A1 settings (live TF-N6 re-run)"), dict(B5.check_arm(L1_ARM, fl), what="L1 at the TF-N5 frozen parameters (live TF-N6 re-run)")]
    sels = make_objects()
    reps += [dict(check_selector(arm, sels[arm]), what="selector " + arm) for arm in ARMS]
    c7 = check_card7(build_parser(), {"--route"})
    reps.append({"what": "card tables read back (LEVERS, TIERS, certificate, grids, flag audit)", "ok": c7["ok"], "bad": c7["bad"]})
    ok = all(r["ok"] for r in reps)
    real = cmd_reach_real_instances(a.fixture) if a.real else None
    ok = ok and (real is None or real["ok"])
    print("ZZN7_REACH", "PASS" if ok else "FAIL", json.dumps({r.get("what"): r["ok"] for r in reps}), flush=True)
    for r in reps:
        for b in r["bad"]:
            print("ZZN7_REACH BAD", r.get("what"), b, flush=True)
    if real is not None:
        print("ZZN7_REACH REAL", json.dumps(real), flush=True)
    if a.out:
        json.dump({"ok": bool(ok), "arms": reps, "real": real}, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------------------------------------------------ derive
def cmd_derive(a) -> int:
    if a.mutate != "none" and not a.smoke_allow_mutation:
        raise SystemExit("REFUSING: --mutate is a navtrain-smoke instrument and needs --smoke-allow-mutation (the chain never passes either)")
    sels = make_objects(a.mutate)
    fd = B4.make_repair(D_ARM)
    fl = B5.make_repair(B5.CARD[L1_ARM])
    sel6 = B6.make_selector("N6T")
    chk = [B4.check_arm(D_ARM, fd), B5.check_arm(L1_ARM, fl)] + [check_selector(arm, sels[arm]) for arm in ARMS] + [check_card7()]
    if not all(c["ok"] for c in chk) and not a.smoke_allow_mutation:
        raise SystemExit("REFUSING: the built generator / selector differs from the card: " + json.dumps([c["bad"] for c in chk if not c["ok"]]))
    z5 = np.load(a.n5a_plans, allow_pickle=False)
    toks, logs, p5 = [str(t) for t in z5["tokens"]], [str(x) for x in z5["logs"]], np.asarray(z5["plans20"], float)
    z2 = np.load(a.n2r_plans, allow_pickle=False)
    t2 = [str(t) for t in z2["tokens"]]
    ix = {t: i for i, t in enumerate(toks)}
    ix2 = {t: i for i, t in enumerate(t2)}
    if set(toks) - set(ix2):
        raise SystemExit("REFUSING: N5-A plan tokens are not all in the N2-R plan file")
    p2 = np.asarray(z2["plans20"], float)
    rolls = load_rolls(a.rolls)
    if a.navtrain_tokens:
        keep = set(json.load(open(a.navtrain_tokens, encoding="utf-8"))["tokens"])
        toks = [t for t in toks if t in keep]
    si, sn = (int(x) for x in a.shard.split("/"))
    cder = set(sorted(toks, key=sha1_int)[:N_CDERIV])
    done = done_tokens(a.out)
    todo = {t: logs[ix[t]] for t in toks if sha1_int(t) % sn == si and t not in done}
    need_roll = {t for t in todo if t in cder}
    if a.resid_tokens:                                  # smoke: the residual tokens are known (their rolls are needed up front)
        need_roll |= set(json.load(open(a.resid_tokens, encoding="utf-8"))["tokens"]) & set(todo)
    missing = [t for t in need_roll if t not in rolls]
    if missing:
        raise SystemExit(f"REFUSING: {len(missing)} tokens have no RJ5 target row, e.g. {missing[:3]}")
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        with open(a.out, "rb") as fb:
            fb.seek(-1, 2)
            torn = fb.read(1) != b"\n"
        if torn:
            open(a.out, "a", encoding="utf-8").write("\n")
    import n4_geom as G
    n, t0 = 0, time.time()
    with open(a.out, "a", encoding="utf-8") as fo:
        for tok, lg, cx in B4.ctx_stream(a, todo):
            ts, rec = time.time(), {"log": lg, "token": tok}
            try:
                plan5 = p5[ix[tok]]
                ref = cx.score([plan5])[0][0]
                rec["ref"] = ref
                if tok in cder:
                    rec["cderiv_maxdev_m"] = float(np.abs(cx.labels(rolls[tok])["R"] - p2[ix2[tok]]).max())
                resid = ref["dac"] == 0.0
                rec["residual"] = bool(resid)
                if resid:
                    if tok not in rolls:
                        raise SystemExit(f"REFUSING: residual token {tok} has no RJ5 target row")
                    rec["n5a_equals_n2r_plan"] = bool(np.array_equal(plan5, p2[ix2[tok]]))
                    rec.update(token_row(cx, rolls[tok], plan5, ref, sels, fd, fl, sel6, a.mutate))
                rec["label_ego"] = cx.label_ego_src if cx.label_ego_src else "none"
                rec["t_s"] = round(time.time() - ts, 3)
            except G.LabelEgoMissing:
                raise                                                      # REFUSE: never an error row, never a cache-ego fallback
            except Exception as e:                                    # noqa: BLE001
                rec = {"log": lg, "token": tok, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            n += 1
    print("ZZN7_DERIVE_SHARD_DONE", a.shard, n, round(time.time() - t0, 1), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------------------------------------- merge
def check_unch(out_dir: str, n5a_plans: str, n5a_seam: str, applied: dict, n_resid: int, n_expect: int) -> dict:
    """C-UNCH on FILES: for every arm the plan20 AND the seam pose array of every token outside the applied set is bit-identical to the banked N5-A's, the changed set equals the
    applied set exactly; the residual count equals the card's; N7-X (applied = []) has plan and seam arrays bit-identical to N5-A's."""
    zb, sb = _npz(n5a_plans), _npz(n5a_seam)
    bt = [str(t) for t in zb["tokens"]]
    bi = {t: i for i, t in enumerate(bt)}
    sbi = {str(t): i for i, t in enumerate(sb["token"])}
    out = {"pass": True, "arms": {}, "n_residual": n_resid, "n_expected": n_expect}
    for arm in ARMS:
        fp, sp = os.path.join(out_dir, f"n7_plans_{arm}.npz"), os.path.join(out_dir, SEAM_LABEL[arm] + ".npz")
        if not (os.path.exists(fp) and os.path.exists(sp)):
            out["arms"][arm] = {"pass": False, "why": "missing file"}
            out["pass"] = False
            continue
        z, s = _npz(fp), _npz(sp)
        zt, stt = [str(t) for t in z["tokens"]], [str(t) for t in s["token"]]
        changed_p = {t for i, t in enumerate(zt) if t in bi and not np.array_equal(z["plans20"][i], zb["plans20"][bi[t]])}
        changed_s = {t for i, t in enumerate(stt) if t in sbi and not np.array_equal(s["poses"][i], sb["poses"][sbi[t]])}
        want = set(applied.get(arm, []))
        ok = changed_p == want and changed_s == want and set(zt) == set(bi) and set(stt) == set(sbi)
        rec = {"changed_plan20": len(changed_p), "changed_seam": len(changed_s), "applied": len(want), "tokens": len(zt), "seam_tokens": len(stt),
               "unflagged_changed": sorted((changed_p | changed_s) - want)[:5], "flagged_unchanged": sorted(want - (changed_p & changed_s))[:5]}
        for k in ("token", "fingerprint", "sampling"):
            if k in sb and not np.array_equal(s[k], sb[k]):
                ok = False
                rec["seam_field_differs"] = k
        if not want:
            rec["bit_identical_to_n5a"] = bool(changed_p == set() and changed_s == set())
        rec["pass"] = bool(ok)
        out["arms"][arm] = rec
        out["pass"] = out["pass"] and ok
    out["residual_count_ok"] = bool(n_resid == n_expect)
    out["pass"] = bool(out["pass"] and out["residual_count_ok"])
    return out


def recompute_chosen(row: dict, arm: str):
    """the CARD selector (NOT the row's recorded one) on the stored candidates of one row -> (tier, lever, k) or None"""
    c = make_selector(arm)({"ref": row["ref"], "levers": row["levers"]})
    return None if c is None else (c["tier"], c["lever"], c["k"])


def dev_cascade(row: dict, arm: str):
    """the DEVELOPMENT cascade (n7_select.cascade, which the n7_gen tests pin) on the stored candidates: the cross-check that the build's selector and the generator's agree"""
    tiers = S.TIERS[:4] if arm == "N7T3" else S.TIERS
    c = S.cascade({"ref": row["ref"], "levers": row["levers"]}, tiers=tiers, floor=None if arm == "N7T3" else FLOOR)
    return None if c is None else (c["tier"], c["lever"], c["k"])


def order_violations(row: dict, rec_ch: dict) -> list:
    """C-ORDER, written independently of card_cascade: the chosen tier is the FIRST tier with any certified candidate (flat generation order), the choice has the maximum builder PDMS
    among that tier's certified candidates and, among ties, the smallest generation index; N7-T equals N7-T3 wherever N7-T3 applies and N7-T takes a T4 candidate only where N7-T3 took none"""
    bad = []
    for arm in ("N7T3", "N7T"):
        ntiers = 4 if arm == "N7T3" else 5
        got, found = rec_ch.get(arm), None
        for ti, (tname, levers) in enumerate(TIERS_CARD[:ntiers]):
            gen = []
            for lv, cap in levers:
                for x in row["levers"].get(lv, []):
                    if cap is not None and x["k"] > cap:
                        continue
                    if tname == "T4_DEEP" and S.describe(lv, x["k"])["s"] < FLOOR:
                        continue
                    if cert_ok(x["m"], row["ref"], CERT):
                        gen.append((lv, x["k"], x["m"]["pdms"]))
            if gen:
                top = max(p for _l, _k, p in gen)
                first = [g for g in gen if g[2] == top][0]
                found = (tname, first[0], first[1])
                break
        gid = None if got is None else (got["tier"], got["lever"], got["k"])
        if gid != found:
            bad.append((arm, "tier/choice is not the first certified tier's best-earliest candidate", gid, found))
    t3, t = rec_ch.get("N7T3"), rec_ch.get("N7T")
    if t3 is not None and (t is None or (t["tier"], t["lever"], t["k"]) != (t3["tier"], t3["lever"], t3["k"]) or t["plan"] != t3["plan"]):
        bad.append(("N7T", "N7-T differs from N7-T3 where N7-T3 applies"))
    if t3 is None and t is not None and t["tier"] != "T4_DEEP":
        bad.append(("N7T", "N7-T takes a non-T4 candidate where N7-T3 took none"))
    return bad


def cmd_merge(a, parser=None) -> int:
    import n4_geom as G
    ND = G.ND
    rows, n_err, n_dup = read_rows(a.rows)
    z5 = _npz(a.n5a_plans)
    bt, bl, bp = [str(t) for t in z5["tokens"]], [str(x) for x in z5["logs"]], np.asarray(z5["plans20"], float)
    sb = _npz(a.n5a_seam)
    order = [str(t) for t in sb["token"]]
    ix = {t: i for i, t in enumerate(bt)}
    rep: dict = {"n_tokens": len(bt), "n_rows": len(rows), "n_error_lines": n_err, "n_duplicate_lines": n_dup, "controls": {}, "diagnostics": {}, "files": {}}
    C = rep["controls"]
    levers_off = any(not r.get("levers") for r in rows.values() if r.get("residual"))
    C["C-COVB"] = {"pass": set(rows) == set(bt) and n_err == 0 and n_dup == 0, "missing": len(set(bt) - set(rows)), "extra": len(set(rows) - set(bt)),
                   "error_lines": n_err, "duplicate_lines": n_dup}
    ok_t = [t for t in bt if t in rows]
    hc = load_harness_csv(a.harness_csv_n5a)
    mism = [t for t in bt if t not in hc or (t in rows and rows[t]["ref"]["dac"] != hc[t]["dac"])]
    C["C-DACMAP6"] = {"pass": not mism and len(hc) >= len(bt), "tokens": len(bt), "mismatches": len(mism), "examples": mism[:5],
                      "what": "builder tracked DAC of the banked N5-A plan == harness DAC of N5-A, all tokens", "reference": os.path.basename(a.harness_csv_n5a)}
    other = {m: max((abs(rows[t]["ref"][m] - hc[t][m]) for t in ok_t if t in hc), default=None) for m in ("nc", "ddc", "ttc", "c", "pdms")}
    rep["diagnostics"]["builder_vs_harness_max_abs_all_tokens"] = other
    cd = sorted(bt, key=sha1_int)[:N_CDERIV]
    dev = [rows[t].get("cderiv_maxdev_m", float("nan")) for t in cd if t in rows]
    C["C-DERIV"] = {"pass": len(dev) == len(cd) and bool(np.nanmax(dev) <= CDERIV_BAR_M) and not any(np.isnan(dev)), "n": len(dev), "expected_n": len(cd),
                    "max_dev_m": float(np.nanmax(dev)) if dev else None, "bar_m": CDERIV_BAR_M}
    resid = sorted(t for t in ok_t if rows[t].get("residual"))
    lab_rows = [t for t in ok_t if "cderiv_maxdev_m" in rows[t] or rows[t].get("residual")]
    not_db = sorted(t for t in lab_rows if rows[t].get("label_ego") != "db")
    C["C-LABELEGO"] = {"pass": not not_db, "label_rows": len(lab_rows), "db": len(lab_rows) - len(not_db), "not_db": len(not_db), "examples": not_db[:5],
                       "what": "every row that built labels (C-DERIV tokens + residual tokens x every lever iterate) used the nuPlan-DB ego"}
    if a.n5a_report:
        banked = json.load(open(a.n5a_report, encoding="utf-8"))
        exp_res, exp_src = sorted(banked["unrepaired_l1"]["N5A"]), "banked TF-N5 A1 build report unrepaired_l1.N5A"
    else:
        exp_res, exp_src = sorted(json.load(open(a.expect_residual_json, encoding="utf-8"))["tokens"]), "--expect-residual-json (smoke)"
    harness_dac0 = sorted(t for t in bt if t in hc and hc[t]["dac"] == 0.0)
    flag_bad = sorted(t for t in ok_t if bool(rows[t].get("residual")) != (rows[t]["ref"]["dac"] == 0.0))
    C["C-SET"] = {"pass": bool(resid == exp_res and (not a.n5a_report or harness_dac0 == exp_res) and not flag_bad), "builder_residual": len(resid), "expected": len(exp_res),
                  "residual_flag_disagrees_with_builder_dac": flag_bad[:5], "harness_n5a_zero_dac": len(harness_dac0), "expected_source": exp_src,
                  "missing_from_builder": sorted(set(exp_res) - set(resid))[:5], "extra_in_builder": sorted(set(resid) - set(exp_res))[:5],
                  "what": "builder tracked N5-A DAC-0 set == the banked TF-N5 A1 residual (and, on navtest, == the harness N5-A zero-DAC set)"}
    expect = json.load(open(a.replay_expect_json, encoding="utf-8")) if a.replay_expect_json else {}
    bad_replay = []
    for t in resid:
        r = rows[t]
        live = r.get("n6_live")
        if not r.get("levers") or live is None:
            bad_replay.append((t, "no generator / live N6 record"))
            continue
        for src, key in (("D", "d_returns_plan"), ("L1", "l1_returns_plan")):
            want = expect.get(key)
            if want is not None and live["repl"][src] != (t in set(want)):
                bad_replay.append((t, f"{src} returns_plan {live['repl'][src]} != banked"))
        d0 = r["levers"].get("D_c20_m05") or []
        if not d0 or d0[0]["k"] != 0 or not np.array_equal(np.asarray(d0[0]["plan"], float), bp[ix[t]]) or not r.get("n5a_equals_n2r_plan"):
            bad_replay.append((t, "iterate 0 of the generator is not the banked N5-A (= N2-R) plan"))
    C["C-REPLAY"] = {"pass": not bad_replay and bool(expect), "residual": len(resid), "violations": bad_replay[:6],
                     "what": "the live TF-N6 re-run reproduces the banked repair results (REPAIR-D and L1 return the same applied / not applied as TF-N4 A1 / TF-N5 A1 on every residual "
                             "token) and iterate 0 of the generator is the banked plan"}
    det = {t: bool(rows[t].get("det_equal")) for t in resid}
    C["C-DET"] = {"pass": bool(det) and all(det.values()) and not levers_off, "checked": len(det), "equal": int(sum(det.values())),
                  "what": "the whole generator (every lever of every tier, every iterate / grid point and its builder score) run twice on every residual token gives bit-identical plans and scores"}
    c7 = check_card7(parser or build_parser(), {"--route"})
    bad_sel, bad_dev = [], []
    for t in resid:
        r = rows[t]
        for arm in ARMS:
            got = r["chosen"].get(arm)
            got_id = None if got is None else (got["tier"], got["lever"], got["k"])
            if recompute_chosen(r, arm) != got_id:
                bad_sel.append((t, arm, got_id, recompute_chosen(r, arm)))
            if arm != "N7X" and dev_cascade(r, arm) != got_id:
                bad_dev.append((t, arm, got_id, dev_cascade(r, arm)))
    C["C-SELECT"] = {"pass": not bad_sel and not bad_dev and c7["ok"], "tokens": len(resid), "violations": len(bad_sel), "examples": bad_sel[:4], "dev_cascade_disagrees": bad_dev[:4],
                     "card_tables_readback_ok": c7["ok"], "card_tables_bad": c7["bad"], "flag_audit": c7["flag_audit"], "levers": c7["levers"], "tiers": c7["tiers"], "floor": FLOOR,
                     "what": "the card selector run in the merge on the stored candidates picks exactly what the derive recorded, per arm and token; the n7_select.cascade of the "
                             "development agrees; n7_lib.LEVERS / n7_select.TIERS read back equal the card's tables; no flag has a disabling default"}
    applied = {arm: sorted(t for t in resid if rows[t]["chosen"].get(arm) is not None) for arm in ARMS}
    tier_of = {arm: {t: rows[t]["chosen"][arm]["tier"] for t in applied[arm]} for arm in ARMS}
    bad_cert = {arm: [t for t in applied[arm] if not cert_ok(rows[t]["chosen"][arm]["m_recheck"], rows[t]["ref"], dict(CARD7[arm]["cert"], tie="earliest", order=("teacher",)))
                      or rows[t]["chosen"][arm]["m_recheck"] != rows[t]["chosen"][arm]["m"]] for arm in ARMS}
    n6t = {k: B6.CARD6["N6T"][k] for k in CERT}
    cert_eq = _norm(n6t) == _norm(CERT) == _norm(L.CERT_TOL) and _norm(CARD7["N7T3"]["cert"]) == _norm(CARD7["N7T"]["cert"]) == _norm(CERT) and CARD7["N7X"]["cert"]["dac"] == 0.0
    C["C-CERT-B7"] = {"pass": not any(bad_cert.values()) and cert_eq, "violations": {k: len(v) for k, v in bad_cert.items()}, "examples": {k: v[:3] for k, v in bad_cert.items()},
                      "certificate_equals_card6_n6t": cert_eq,
                      "what": "every applied plan, re-scored by a separate scorer call, reproduces its score and passes the arm's CARD certificate against the N5-A builder score (DAC; NC and "
                              "DDC not worse; builder PDMS strictly better; TTC and C free); the certificate equals CARD6 N6T and n7_lib.CERT_TOL"}
    bad_order = []
    for t in resid:
        bad_order += [(t,) + v for v in order_violations(rows[t], rows[t]["chosen"])]
    C["C-ORDER"] = {"pass": not bad_order, "violations": bad_order[:4], "what": "tier order and tie rule: the choice is the first certified tier's maximum-builder-PDMS, earliest-generated "
                    "candidate (independent re-derivation); N7-T == N7-T3 wherever N7-T3 applies; N7-T takes a T4 candidate only where N7-T3 took none"}
    C["C-X0"] = {"pass": not applied["N7X"], "n7x_applied": len(applied["N7X"]),
                 "what": "the deliberate-regression arm (certificate inverted: DAC = 0 required) changes 0 tokens, so its seam is bit-identical to N5-A (a DAC-0 plan has builder PDMS 0)"}
    # ---- C-N6SAME: the TF-N6 selection, live (pinned n6 capture + N6-T selector) and banked (the N6 build report and the N6-T plan file) == the cascade's T0 choice
    bad_same = []
    for t in resid:
        r = rows[t]
        live = (r.get("n6_live") or {}).get("chosen")
        for arm in ("N7T3", "N7T"):
            ch = r["chosen"].get(arm)
            t0 = ch if (ch is not None and ch["tier"] == "T0_N6") else None
            if (live is None) != (t0 is None):
                bad_same.append((t, arm, "N6-T live selection and the T0 choice disagree on whether the token is certified"))
            elif live is not None:
                want_lv = "D_c20_m05" if live["src"] == "D" else "L1_c20_m05"
                if (t0["lever"], t0["k"]) != (want_lv, live["k"]) or t0["plan"] != live["plan"] or t0["m"] != live["m"]:
                    bad_same.append((t, arm, "T0 choice differs from the live N6-T choice (lever, k, plan or score)"))
    banked = {"report": None, "plans": None}
    n6_report, n6t_plans = getattr(a, "n6_report", ""), getattr(a, "n6t_plans", "")
    if n6_report:
        n6r = json.load(open(n6_report, encoding="utf-8"))
        n6t_toks = sorted(n6r["repaired"]["N6T"])
        banked["report"] = True
        for t in n6t_toks:
            if t not in resid:
                bad_same.append((t, "report", "N6-T token is not in the residual set"))
                continue
            rb = n6r["residual_table"][t]["resolved_by"]
            for arm in ("N7T3", "N7T"):
                ch = rows[t]["chosen"].get(arm)
                src = None if ch is None else ("D" if ch["lever"].startswith("D_") else "L1")
                if ch is None or ch["tier"] != "T0_N6" or [src, ch["k"], "teacher"] != rb:
                    bad_same.append((t, arm, "not the banked N6-T choice " + json.dumps(rb)))
                    continue
                cm = [c for c in n6r["residual_table"][t]["candidates"] if (c["src"], c["k"]) == (rb[0], rb[1])][0]
                if any(ch["m"][k] != cm[k] for k in ("dac", "nc", "ddc", "ttc", "c", "pdms")):
                    bad_same.append((t, arm, "metrics differ from the banked N6 candidate"))
        extra = sorted(t for t in resid if rows[t]["chosen"].get("N7T3") is not None and rows[t]["chosen"]["N7T3"]["tier"] == "T0_N6" and t not in n6t_toks)
        if extra:
            bad_same.append((extra[0], "report", f"{len(extra)} tokens are T0-certified here but N6-T did not certify them"))
        if n6t_plans:
            zt = _npz(n6t_plans)
            tix = {str(x): i for i, x in enumerate(zt["tokens"])}
            want_sha = n6r.get("files", {}).get("n6_plans_N6T.npz")
            banked["plans"] = True
            if want_sha and sha256(n6t_plans) != want_sha:
                bad_same.append(("*", "plans", "sha256 of the N6-T plan file != the N6 build report"))
            for t in n6t_toks:
                for arm in ("N7T3", "N7T"):
                    ch = rows[t]["chosen"].get(arm)
                    if ch is None or t not in tix or not np.array_equal(np.asarray(ch["plan"], float), np.asarray(zt["plans20"][tix[t]], float)):
                        bad_same.append((t, arm, "plan is not bit-identical to the banked N6-T plan"))
            changed_n6 = {str(x) for i, x in enumerate(zt["tokens"]) if not np.array_equal(zt["plans20"][i], bp[ix[str(x)]])} if all(str(x) in ix for x in zt["tokens"]) else None
            if changed_n6 is not None and changed_n6 != set(n6t_toks):
                bad_same.append(("*", "plans", "the banked N6-T plan file changed a different token set than the N6 report names"))
    C["C-N6SAME"] = {"pass": not bad_same, "violations": bad_same[:6], "n_violations": len(bad_same), "live_checked_tokens": len(resid), "banked_report_checked": bool(banked["report"]),
                     "banked_plans_checked": bool(banked["plans"]), "n6t_tokens": sorted(n6r["repaired"]["N6T"]) if n6_report else None,
                     "what": "the tokens N6-T certified keep the TF-N6 plan bit for bit in N7-T3 and N7-T (T0 == N6): the live N6 selection equals the T0 choice on every residual token, "
                             "and the banked N6 report (choices and scores) and the banked N6-T plan file agree"}
    os.makedirs(a.out_dir, exist_ok=True)
    Pl = {arm: bp.copy() for arm in ARMS}
    for arm in ARMS:
        for t in applied[arm]:
            Pl[arm][ix[t]] = np.asarray(rows[t]["chosen"][arm]["plan"], float)
        fp = os.path.join(a.out_dir, f"n7_plans_{arm}.npz")
        np.savez(fp, tokens=np.array(bt), logs=np.array(bl), plans20=Pl[arm], arm=np.array(arm))
        rep["files"][os.path.basename(fp)] = sha256(fp)
    p8 = np.stack([ND.to_navsim(bp[ix[t]]).astype(np.float32) for t in order])
    bad_fn = [order[i] for i in range(len(order)) if not np.array_equal(p8[i], sb["poses"][i])]
    C["C-SEAMFN"] = {"pass": not bad_fn, "tokens": len(order), "mismatches": len(bad_fn), "examples": bad_fn[:3],
                     "what": "ND.to_navsim reproduces the banked N5-A seam poses bit for bit on every token, so the changed rows are made by the function that made the banked ones"}
    for arm in ARMS:
        ap = set(applied[arm])
        p8a = np.stack([ND.to_navsim(Pl[arm][ix[t]]).astype(np.float32) if t in ap else sb["poses"][i] for i, t in enumerate(order)])
        old = str(sb["arm"])
        new_arm = (old[: -len(OLD_SEAM)] if old.endswith(OLD_SEAM) else old + "_") + SEAM_LABEL[arm]
        sp = os.path.join(a.out_dir, SEAM_LABEL[arm] + ".npz")
        np.savez(sp, token=sb["token"], fingerprint=sb["fingerprint"], poses=p8a, sampling=sb["sampling"], arm=np.array(new_arm))
        rep["files"][os.path.basename(sp)] = sha256(sp)
    u = check_unch(a.out_dir, a.n5a_plans, a.n5a_seam, applied, len(resid), a.expect_residual)
    C["C-UNCH"] = u
    if a.reach_json:
        rj = json.load(open(a.reach_json, encoding="utf-8"))
        real = rj.get("real") or {}
        C["C-REACH"] = {"pass": bool(rj.get("ok") and real.get("ok") and real.get("tiers_executed") == list(TIER_NAMES)), "arms_ok": {x.get("what"): x["ok"] for x in rj.get("arms", [])},
                        "tiers_executed": real.get("tiers_executed"), "real": {k: v.get("ok") for k, v in (real.get("instances") or {}).items()}}
    else:
        C["C-REACH"] = {"pass": False, "why": "no --reach-json (the chain's reach_real stage)"}
    rep["residual"] = resid
    rep["repaired"] = applied
    rep["tier_of"] = tier_of
    rep["counts"] = {arm: {"n_teacher_tol": len(applied[arm]), "by_tier": {tn: sum(1 for v in tier_of[arm].values() if v == tn) for tn in TIER_NAMES}} for arm in ARMS}
    best = {}
    for t in resid:                  # per residual token: what the best DAC = 1 candidate failed, or that none reaches DAC = 1 (the blocking class, card section 5)
        r, tab = rows[t], {}
        d1_fail = {}
        for tname, lvs in TIERS_CARD:
            n_c = n_d1 = n_cert = 0
            for lv, cap in lvs:
                for x in r["levers"].get(lv, []):
                    if cap is not None and x["k"] > cap:
                        continue
                    n_c += 1
                    cl = cert_clauses(x["m"], r["ref"], CERT)
                    if x["m"]["dac"] == 1.0:
                        n_d1 += 1
                        if tname == "T4_DEEP" and S.describe(lv, x["k"])["s"] < FLOOR:
                            d1_fail.setdefault(tname, set()).add("below_floor")
                            continue
                        if all(cl.values()):
                            n_cert += 1
                        else:
                            d1_fail.setdefault(tname, set()).add(",".join(sorted(k for k, v in cl.items() if not v)))
            tab[tname] = {"n_candidates": n_c, "n_dac1": n_d1, "n_certified": n_cert}
        ch = r["chosen"].get("N7T")
        any_d1 = any(v["n_dac1"] for v in tab.values())
        best[t] = {"resolved_by": None if ch is None else [ch["tier"], ch["lever"], ch["k"], ch["param"]], "resolved_by_n7t3": (lambda c: None if c is None else [c["tier"], c["lever"], c["k"]])(r["chosen"].get("N7T3")),
                   "tiers": tab, "ref_pdms": r["ref"]["pdms"],
                   "blocking_class": None if ch is not None else ("no candidate in any tier reaches DAC = 1" if not any_d1 else
                                                                  "DAC-1 candidates exist but fail: " + "; ".join(f"{k}: {sorted(v)}" for k, v in sorted(d1_fail.items()))),
                   "blocking_class_n7t3": None if r["chosen"].get("N7T3") is not None else ("no candidate in tiers T0-T3 reaches DAC = 1" if not any(tab[n]["n_dac1"] for n in TIER_NAMES[:4]) else
                                                                                           "DAC-1 candidates exist in T0-T3 but fail: " + "; ".join(f"{k}: {sorted(v)}" for k, v in sorted(d1_fail.items()) if k != "T4_DEEP"))}
    rep["residual_table"] = best
    ts = [r["t_s"] for r in rows.values() if "t_s" in r]
    rep["timing"] = {"n": len(ts), "t_s_mean": float(np.mean(ts)) if ts else None, "cpu_h_sum": float(np.sum(ts) / 3600.0) if ts else None}
    allpass = all(c["pass"] for c in C.values())
    rep["verdict"] = "PASS" if allpass else "FAIL"
    json.dump(rep, open(os.path.join(a.out_dir, "n7_build_report.json"), "w", encoding="utf-8"), indent=1)
    print("ZZN7_BUILD", rep["verdict"], json.dumps({k: v["pass"] for k, v in C.items()}), "applied", {k: len(v) for k, v in applied.items()}, "counts", json.dumps(rep["counts"]),
          "residual", len(resid), flush=True)
    return 0 if allpass else 1


def cmd_unch(a) -> int:
    rep = json.load(open(os.path.join(a.out_dir, "n7_build_report.json"), encoding="utf-8"))
    u = check_unch(a.out_dir, a.n5a_plans, a.n5a_seam, rep["repaired"], len(rep["residual"]), a.expect_residual)
    print("ZZN7_UNCH", "PASS" if u["pass"] else "FAIL", json.dumps(u["arms"]), flush=True)
    if a.out:
        json.dump(u, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0 if u["pass"] else 1


def main() -> int:
    ap = build_parser()
    a = ap.parse_args()
    return {"reach": cmd_reach, "derive": cmd_derive, "merge": lambda x: cmd_merge(x, ap), "unch": cmd_unch}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
