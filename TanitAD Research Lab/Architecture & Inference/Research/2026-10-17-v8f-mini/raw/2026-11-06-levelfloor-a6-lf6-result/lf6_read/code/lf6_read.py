#!/usr/bin/env python3
"""lf6_read.py -- the T1-kin read of PREREG_LEVELFLOOR_A6.md (sha256 fbf5de02...): the bounded-lift floor F_B at kappa 1.0, v_f(tau) = min(kappa * v_hat [lead cap], v0 + 1.0 * tau, v_O(tau) + DELTA), DELTA in {0.5, 1.0}, read with the logged speed
(LFOB05, LFOB10: CEILING) and with the trained LLR heads (LFB05, LFB10: the only admissible arms). Planner C0_s{s}, lead cap from LLR_s{s}, 3 m/s band and every other constant as in LF075 / LF100.
NOT a text clone: this module LOADS lfo_read.py (the A4 ceiling rules P_O / G_O / F_O and the CEILING-CLEARS / -HARMS / -SHORT / F-UNREACHED labels) and, through it, lf_read.py (the A3 rules for LFB: P, G, F, D read A over four families, LEAK-BORNE on `logged`, the reach
control) BY SHA-PINNED PATH and calls them unchanged. New here: the 24 rollouts and their flags, the bounded-lift instruments and controls (share of bound ticks where the DELTA term binds, mean lift where bound, s' - s <= DELTA * t on every knot), the attribution
beside (LFB{DELTA} - RILF100, LFOB{DELTA} - RILFO100, from the banked A5 arrays in lf5_read/raw/arrays, sha256-pinned), the joint reading of card section 5 in its order 1 to 5, and the multiplicity note (five heads-floor doses).
Every ceiling-arm result carries the label CEILING (a privileged level input, not admissible at inference; never a driving result, never a counted DOMINANT). Only LFB05 / LFB10 can count.
Modes:  production [--out-root D]  |  ka-a5  (the banked A5 arrays in the A6 slots reproduce LF5_READ.json's LFO100 rules/label and LF100 rules/label exactly, and the attribution rows against itself are exactly zero; raw/KA_READER_ka-a5.json)
usage: python -I lf6_read.py --mode production [--workers 4]"""
import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
MC = f"{ART}/dagland/motion_c"
PKG = f"{MC}/lf6_read"
RDB = f"{ART}/v8f-recover"
LFO_READ_PY, LFO_READ_SHA = f"{MC}/lfo_read/code/lfo_read.py", "35dae285c262943a43c6bd52e5eee6dcd7782440725c446c9a7d215fdba728c6"
LFO_READ_JSON, LFO_READ_JSON_SHA = f"{MC}/lfo_read/raw/LFO_READ.json", "f6c936aae77b6f1f240cad2e20d19319a2663aa0baf041bc34a4b4ae1c2d4387"
LF5_READ_JSON, LF5_READ_JSON_SHA = f"{MC}/lf5_read/raw/LF5_READ.json", "0bb8cf99dc5cc846fa0ee5fa641ea1f5992efa1e23be54fa4fc03dad9da0f4df"
LF5_ARR = f"{MC}/lf5_read/raw/arrays"                                                         # the banked A5 arrays (landed package lf5_read)
LF5_ARR_SHA = {"RILF100_s0": "e1c65f34db1bfaf39960c9d38b1ef7be9741be13302d8e406dd99aa202da23b9", "RILF100_s1": "c60df4ad2d295fac48b4e071d2ddb16d171cef079a92ffa5ac7830e88beb04c4",
               "RILF100C_s0": "b557da901dcec23ea4c64f5ad2e19f1548bb124ea402fd36c16e62f13ee35799", "RILF100C_s1": "08283a0938a4e7d156e9cd70d867ed799ba18645b6122696d12430fbb11c2a1e",
               "RILFO100_s0": "2b20d4cae72fb04b71ed9a88d5670ff3c8a877924c0342e21ca2ae4439d03803", "RILFO100_s1": "3cdfa07b693562b23be2ed6a68738c49182038bd67342c1bf085953d8ab20ba3",
               "RILFO100C_s0": "53aa0198b40e3fcf492b3ee1ede734cfff76d90b556bc47eb45fb3001e2705b8", "RILFO100C_s1": "e9c637cf8f70f2ea5f0857a0026d45e77e6aaabaf1ab3236f9a2df1561a2c0ee"}
LF5_KA_ARR_SHA = {"LF100_s0": "6948b9a58335f2fe434a52359b2ddf04d050e3801d59ee5033843c069d1b2731", "LF100_s1": "c912881e7170333fa3ea3c502baba4ed044f9583c7095be35f75b10b334fa4c9",
                   "LF100C_s0": "2a642a5478932cd66aa9ef39e82fe16c6a5cc9fa146d97361cb06c3dcbc87852", "LF100C_s1": "20c55a49bf520aed8cde423bd7bd7512bed858fc65b2a3db9da44d87d5064ffd"}      # ka-a5 only
CARD, CARD_FROZEN = f"{MC}/a6_prereg/PREREG_LEVELFLOOR_A6.md", f"{MC}/a6_prereg/FROZEN_SHA.txt"
CARD_SHA = "fbf5de02f1444e3c4b283a61c32c4be051dd146dbe4f52847df856160de92eb1"
BUILD_PINS = os.environ.get("LF6_BUILD_PINS", f"{MC}/lf6_build/PINS.txt")
OUT_ROOT, WORK_DIR = "F:/TanitAD-scratch/motion_c/lf6_read/out", "F:/TanitAD-scratch/motion_c/lf6_read/work"
SEEDS, KAPPA, KAPPA_FLAG, BASE_SMOOTH = (0, 1), 1.0, "1.0", "sg72"
DELTAS = {"05": 0.5, "10": 1.0}                                                                 # the doses of the card (m/s), fixed a priori
DELTA_FLAG = {"05": "0.5", "10": "1.0"}
REPORT_ORDER = ("05", "10")                                                                    # the smaller intervention is named first
BOUND_TOL_M = 1e-6                                                                             # s' - s <= DELTA * t + BOUND_TOL_M on every knot (card section 2)
CEIL_ARMS = {"LFOB05": ("RILFOB05", "05"), "LFOB10": ("RILFOB10", "10")}
LFB_ARMS = {"LFB05": ("RILFB05", "LFB05", "05"), "LFB10": ("RILFB10", "LFB10", "10")}
CEILING_NOTE = "CEILING: the level input is the human's logged route speed (exact), a privileged channel that is NOT admissible at inference. Not a driving result, not a capability claim, never a counted DOMINANT; it ranks levers (A6 card)."
MULTIPLICITY = ("The heads-floor family has now read five doses (kappa 0.50, 0.75 and 1.0, then delta 0.5 and 1.0 at kappa 1): a counted DOMINANT names its kappa and delta and says that five were read. "
                "Adaptivity, disclosed: the form and the delta doses were chosen after looking at eval-141 rollouts and this read is on the same windows (A6 card).")
LFO = LFR = LR = rr = SR = GD = PCO = None


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def init():
    """Load lfo_read.py BY FILE PATH (sha256 checked), run its init() (which loads lf_read.py, llr_read.py and the RECOVER-NEXT reader by pinned path), add the doses B05 / B10 (kappa 1.0) to lf_read's tables, load lf6_precheck.py."""
    global LFO, LFR, LR, rr, SR, GD, PCO
    got = sha256_file(LFO_READ_PY)
    if got != LFO_READ_SHA:
        raise SystemExit(f"REFUSED: {LFO_READ_PY} is {got}, not the pinned {LFO_READ_SHA}")
    LFO = _load("lfo_read_for_lf6", LFO_READ_PY)
    LFO.init()
    LFR, LR, rr, SR, GD = LFO.LFR, LFO.LR, LFO.rr, LFO.SR, LFO.GD
    for d in DELTAS:                                                                           # lf_read.py's dose tables gain the doses of A6 (its rules are unchanged); the dose key B05 / B10 reads kappa 1.0
        LFR.KAPPA["B" + d], LFR.KAPPA_FLAG["B" + d] = KAPPA, KAPPA_FLAG
    PCO = _load("lf6_precheck_for_read", f"{PKG}/code/lf6_precheck.py")
    return LR


# ------------------------------------------------------------------------------------------------ the 24 rollouts (card section 3), in PRIORITY order
def floor_ckpt(s):
    return f"{RDB}/M2b_s{s}_recft_LLR/ckpt_last.pt"


def rollouts():
    """[(tag, base, composite, seed, scene, feasible, source, delta)] -- 24 rows. A killed chain yields the deciding admissible arms first (LFB10, LFB05 on reindexed), then the ceiling (LFOB10, LFOB05), then LFB10 / LFB05 on `logged`."""
    rows = []
    for base, scene, source, dk in (("RILFB10", "reindexed", "heads", "10"), ("RILFB05", "reindexed", "heads", "05"), ("RILFOB10", "reindexed", "logged", "10"), ("RILFOB05", "reindexed", "logged", "05"),
                                    ("LFB10", "logged", "heads", "10"), ("LFB05", "logged", "heads", "05")):
        for comp in (False, True):
            for s in SEEDS:
                rows.append((f"{base}{'C' if comp else ''}_s{s}", base, comp, s, scene, "kbox" if comp else "off", source, DELTAS[dk]))
    return rows


def delta_key(delta):
    return {v: k for k, v in DELTAS.items()}[float(delta)]


def expected_flags(source, delta, scene, feas, s):
    return (f"--smooth {BASE_SMOOTH} --scene {scene}" + (f" --feasible {feas}" if feas != "off" else "") + f" --level-floor on --floor-kappa {KAPPA_FLAG} --floor-ckpt {floor_ckpt(s)} --floor-vhat-source {source}"
            + f" --floor-lift-cap {DELTA_FLAG[delta_key(delta)]}")


# ------------------------------------------------------------------------------------------------ pure layer 1: instruments and controls
def _err_block(vh, v_log, sel):
    sel = sel & np.isfinite(vh) & np.isfinite(v_log) & (v_log > 0) & (vh > 0)
    if not sel.any():
        return {"n": 0, "mae_mps": None, "relative_mae": None, "log_ratio_mean": None, "log_ratio_std": None, "share_vhat_exceeds_logged": None}
    e, lr = vh[sel] - v_log[sel], np.log(vh[sel] / v_log[sel])
    return {"n": int(sel.sum()), "mae_mps": float(np.abs(e).mean()), "relative_mae": float((np.abs(e) / v_log[sel]).mean()), "log_ratio_mean": float(lr.mean()), "log_ratio_std": float(lr.std()), "share_vhat_exceeds_logged": float((e > 0).mean())}


def bound_block(t, delta):
    """The bounded-lift instruments of one tick dump (needs lf_bind, lf_dterm, lf_dterm_knots, lf_s, lf_sp): the share of bound ticks where the DELTA term is the strict minimum at a knot where F lifts, the mean number of such knots, and the card's bound
    s' - s <= DELTA * t over every knot of every tick (the max slack; <= BOUND_TOL_M holds), s' >= s, and that an unbound tick has s' == s."""
    bind, dt = np.asarray(t["lf_bind"], bool), np.asarray(t["lf_dterm"], bool)
    s, sp = np.asarray(t["lf_s"], np.float64), np.asarray(t["lf_sp"], np.float64)
    tk = 0.2 * np.arange(1, s.shape[1] + 1)[None, :]
    slack = float(np.nanmax(sp - s - float(delta) * tk)) if s.size else None
    kn = np.asarray(t["lf_dterm_knots"], np.float64)
    return {"delta": float(delta), "n_ticks": int(len(bind)), "n_bound": int(bind.sum()), "dterm_share_of_bound": None if not bind.any() else float(dt[bind].mean()), "n_dterm_bound_ticks": int((dt & bind).sum()),
            "mean_dterm_knots_where_bound": None if not bind.any() else float(kn[bind].mean()), "max_sp_minus_s_minus_delta_t_m": slack, "bound_holds": bool(slack is not None and slack <= BOUND_TOL_M),
            "sp_ge_s": bool((sp >= s).all()), "unbound_ticks_have_sp_equal_s": bool(np.array_equal(sp[~bind], s[~bind])), "dterm_only_on_bound_ticks": bool(not (dt & ~bind).any())}


def tick_instruments6(t, source, delta):
    """The A3 instruments (bind overall and in the M1 ticks, cap, lift where bound incl. the negative chord lift, v_f against the logged speed) plus the error of v_hat against the logged speed on the same frames (the heads' MAE, relative MAE and overshoot
    share on a heads arm; zero by construction on a ceiling arm), plus the bounded-lift block. Beside, never deciding."""
    r = LFR.tick_instruments(t)
    v_log, ok = LFR.logged_speed(t["gt"], t["gt_ok"])
    vh = np.asarray(t["lf_vhat"], np.float64)
    m1 = LFR.m1_ticks(t["v0"], v_log, ok)
    r["vhat_vs_logged"] = {"all_ticks": _err_block(vh, v_log, np.ones(len(vh), bool)), "m1_ticks": _err_block(vh, v_log, m1)}
    if source == "logged":
        r["oracle"] = LFO.oracle_block(t, KAPPA)
    r["bounded"] = bound_block(t, delta)
    return r


def tick_controls6(t, source, delta):
    """Tool-versus-card consistency of one tick dump. heads: lf_read's controls; logged: lfo_read's (lf_vf stays the scalar first term kappa * v_hat with the lead cap, so both read unchanged); plus the A6 bound: s' - s <= DELTA * t + 1e-6 on every knot, s' >= s,
    the DELTA term binding only on bound ticks, the lift cap in the ticks file equal to DELTA, and no object-dtype key in the file."""
    c = LFR.tick_controls(t, KAPPA) if source == "heads" else LFO.tick_controls(t, KAPPA)
    need = ("lf_dterm", "lf_dterm_knots", "lf_s", "lf_sp", "lf_bind", "floor_lift_cap")
    if not all(k in t for k in need):
        c["lf6_fields_present"] = False
        return c
    c["lf6_fields_present"] = True
    b = bound_block(t, delta)
    c["bound_s_prime_minus_s_le_delta_t"] = b["bound_holds"]
    c["s_prime_ge_s"] = b["sp_ge_s"]
    c["unbound_ticks_have_s_prime_equal_s"] = b["unbound_ticks_have_sp_equal_s"]
    c["dterm_only_on_bound_ticks"] = b["dterm_only_on_bound_ticks"]
    c["floor_lift_cap_in_ticks_equals_delta"] = bool(float(t["floor_lift_cap"]) == float(delta))
    c["no_object_dtype_member"] = bool(all(getattr(v, "dtype", None) is None or v.dtype.kind != "O" for v in t.values()))
    return c


# ------------------------------------------------------------------------------------------------ pure layer 2: the joint reading (card section 5, in its order 1 to 5)
def joint_reading(lfb, lfob):
    """Card section 5, exactly one case, read in the card's order. ``lfb`` = {"05": A3 label of LFB05, "10": ...}; ``lfob`` = {"05": A4 label of LFOB05, "10": ...}. Returns {case, counted_dominant, doses, estimator_levers_signalled, next}."""
    hold = [d for d in REPORT_ORDER if lfb.get(d) == "D-HOLDS"]
    if hold:
        names = " and ".join(f"delta {DELTAS[d]:.1f}" for d in hold)
        return {"case": "LFB-COUNTED-DOMINANT", "counted_dominant": True, "doses": [DELTAS[d] for d in hold], "estimator_levers_signalled": None,
                "next": (f"COUNTED DOMINANT at kappa 1 with {names} (D holds at both seeds with P and G, not LEAK-BORNE)" + (", both reported, delta 0.5 (the smaller intervention) named first" if len(hold) == 2 else "")
                         + "; signal the PI for scale (data and size), with the video evidence. " + MULTIPLICITY)}
    clears = [d for d in REPORT_ORDER if lfob.get(d) == "CEILING-CLEARS"]
    if clears:
        names = " and ".join(f"delta {DELTAS[d]:.1f}" for d in clears)
        return {"case": "BOUNDED-FORM-WORKS-GIVEN-A-GOOD-LEVEL", "counted_dominant": False, "doses": [DELTAS[d] for d in clears], "estimator_levers_signalled": True,
                "next": (f"LFOB is CEILING-CLEARS at {names}: the bounded form works given a good level, and the heads are the bottleneck; the estimator levers (the A40, a stronger estimator on more data) go to the PI, with the ceiling and the heads' measured error as the case; "
                         "the next CPU arm is a temporal filter of the heads' level at that delta, in its own addendum")}
    labels = [lfob.get(d) for d in REPORT_ORDER]
    if all(x == "CEILING-HARMS" for x in labels):
        return {"case": "BOUNDING-DOES-NOT-REMOVE-THE-HARM", "counted_dominant": False, "doses": [], "estimator_levers_signalled": False,
                "next": ("LFOB is CEILING-HARMS at both doses: bounding the lift does not remove the operator's harm; the progress-floor family is exhausted on CPU, and T2 re-perception (option (c)) and the A40 are reported to the PI; "
                         "the next CPU item is pulled from BACKLOG.md")}
    if all(x in ("CEILING-SHORT", "CEILING-HARMS") for x in labels):
        return {"case": "THE-BOUND-REMOVES-THE-CLOSURE", "counted_dominant": False, "doses": [], "estimator_levers_signalled": False,
                "next": ("LFOB is CEILING-SHORT at both doses, or SHORT at one and HARMS at the other: the bound removes the closure; the floor's closure needs a lift larger than O's plan supports, and no measured estimator supports such a lift; "
                         "the family is exhausted on CPU, and T2 and the A40 are reported to the PI")}
    un = [f"delta {DELTAS[d]:.1f}" for d in REPORT_ORDER if lfob.get(d) == "F-UNREACHED"]
    return {"case": "F-UNREACHED", "counted_dominant": False, "doses": [], "estimator_levers_signalled": False,
            "next": f"F-UNREACHED at {', '.join(un) or 'a dose'} (labels {dict(zip(REPORT_ORDER, labels))}): the instrument is checked first"}


# ------------------------------------------------------------------------------------------------ the read (arrays)
def read_ceiling_arm(arm_ar, att_ar, n_boot, reach, cache, log=print):
    """One ceiling arm. ``arm_ar`` = {seed: {RILFO, RILFOC}} (the A6 rollouts); ``att_ar`` = {seed: {RILF, RILFC}}: the attribution base, here the banked A5 RILFO100 / RILFO100C (the same exact level at the unbounded lift). Rules P_O / G_O / F_O and the label
    are lfo_read.read_ceiling's, unchanged."""
    arrays = {s: {**cache["base"][s], **arm_ar[s], **att_ar[s]} for s in SEEDS}
    res = LFO.read_ceiling(arrays, n_boot, reach, cache["c"], log)
    res["kappa"] = KAPPA
    return res


def relabel(b):
    """In this read the attribution slot RILF holds the banked A5 RILFO100: rename lfo_read's keys for what they are here."""
    for s, d in (b.get("seeds") or {}).items():
        for old, new in (("RILFO_minus_RILF075", "ARM_minus_RILFO100"), ("RILFOC_minus_RILF075C", "ARMC_minus_RILFO100C"), ("RILF075_minus_RIC0_beside", "RILFO100_minus_RIC0_beside")):
            if old in d:
                d[new] = d.pop(old)
        if "family_tables" in d and "RILFO_minus_RILF075" in d["family_tables"]:
            d["family_tables"]["ARM_minus_RILFO100"] = d["family_tables"].pop("RILFO_minus_RILF075")
        d["attribution_rows_changed_vs_RILFO100"] = d.pop("attribution_rows_changed_vs_RILF075", None)
    return b


def attribution_heads(arrays, n_boot):
    """Beside, never deciding (card section 3): LFB{DELTA} - RILF100, bare and composite, per seed (19 rows each): what the bound changed against the unbounded heads floor. ``arrays`` = {seed: {role: array dict}} with RILFB, RILFBC, RILF100, RILF100C."""
    out = {}
    for s in SEEDS:
        ar = arrays[s]
        eid = ar["C0"]["eid"]
        cls, _ = GD.classify(ar["C0"]["v0"], ar["C0"]["g"])
        out[f"s{s}"] = {"RILFB_minus_RILF100": LFO._rows(ar, eid, cls, "RILFB", "RILF100", n_boot)[0], "RILFBC_minus_RILF100C": LFO._rows(ar, eid, cls, "RILFBC", "RILF100C", n_boot)[0]}
    return out


def reused_arrays(s):
    z, ctl = LFO.reused_arrays(s)                                                      # C0 C0C C0REP RIC0 RIC0C (pinned) + RILF = RILF075, RILFC = RILF075C (pinned; unused here, their pins still hold)
    for role in ("RILF100", "RILF100C", "RILFO100", "RILFO100C"):
        tag = f"{role}_s{s}"
        p = f"{LF5_ARR}/{tag}.npz"
        got = sha256_file(p)
        ctl[tag] = {"path": p, "sha256": got, "pinned": LF5_ARR_SHA[tag], "PASS": got == LF5_ARR_SHA[tag]}
        z[role] = rr.read_arrays(p)
    return z, ctl


# ------------------------------------------------------------------------------------------------ controls
def frozen_card_sha():
    for line in open(CARD_FROZEN, encoding="utf-8"):
        t = line.split()
        if len(t) >= 2 and t[0] == os.path.basename(CARD):
            return t[1]
    return None


def load_ticks_strict(path):
    """The pinned loader's condition: allow_pickle=False. An object-dtype member (the LF5 tool's floor_vhat_noise_seed = None at sigma 0) raises here."""
    t = np.load(path, allow_pickle=False)
    return {k: t[k] for k in t.files}


def run_controls(out_root, raw):
    ctl, ckpts = {}, {}
    try:
        tool_pin = LFR.parse_pins(BUILD_PINS).get("patched_tool")
    except Exception:
        tool_pin = None
    ctl["patched_tool_pin"] = {"PASS": bool(tool_pin and len(tool_pin) == 64), "patched_tool": tool_pin}
    for tag, base, comp, s, scene, feas, source, delta in rollouts():
        c = {"PASS": False}
        try:
            d = json.load(open(f"{out_root}/done_{tag}.txt", encoding="utf-8"))
            h = rr.run_header(out_root, tag)
            fe = (h.get("feasible") or {}).get("mode", "off") if isinstance(h.get("feasible"), dict) else ("off" if not h.get("feasible") else str(h.get("feasible")))
            sm = h["smooth"]["mode"] if isinstance(h.get("smooth"), dict) else h.get("smooth")
            fc, _ = PCO.check_rollout(out_root, tag)
            td = load_ticks_strict(f"{out_root}/{tag}/ticks/ticks_{tag}.npz")
            tc = tick_controls6(td, source, delta)
            c = {"flags_exact": d["flags"] == expected_flags(source, delta, scene, feas, s), "tool_sha256_is_patched_tool": bool(tool_pin and d["tool_sha256"] == tool_pin),
                 "ckpt_is_C0": d["ckpt"].replace("\\", "/").endswith(f"/M2b_s{s}_recft_C0/ckpt_last.pt"), "header_scene": h["scene"] == scene, "header_feasible": fe == feas, "header_smooth": sm == BASE_SMOOTH,
                 "header_n_windows_2399": int(h["n_windows"]) == 2399, "header_kin_yaw": h["kin_mode"] == "yaw", "ckpt_sha256": d["ckpt_sha256"], "threads": d.get("threads")}
            c.update({f"floor_{k}": v for k, v in fc.items()})
            c.update({f"ticks_{k}": v for k, v in tc.items()})
            ckpts.setdefault(s, set()).add(d["ckpt_sha256"])
            c["PASS"] = all(v for k, v in c.items() if isinstance(v, bool))
        except Exception as e:
            c["error"] = f"{type(e).__name__}: {e}"
        ctl[f"rollout_{tag}"] = c
    ctl["one_checkpoint_per_seed"] = {"PASS": bool(len(ckpts) == len(SEEDS) and all(len(v) == 1 and next(iter(v)) == PCO.C0_SHA[s] for s, v in ckpts.items())), "ckpt_sha256": {f"s{s}": sorted(v) for s, v in ckpts.items()}}
    try:
        pc = json.load(open(f"{raw}/LF6_PRECHECK.json", encoding="utf-8"))
        ctl["LF6_PRECHECK"] = {"PASS": bool(pc["PASS"] is True and all(pc["seeds"][f"s{s}"]["C0"]["ckpt_sha256"] == PCO.C0_SHA[s] and pc["seeds"][f"s{s}"]["LLR"]["ckpt_sha256"] == PCO.llr_sha_of(s) for s in SEEDS))}
    except Exception as e:
        ctl["LF6_PRECHECK"] = {"PASS": False, "error": f"{type(e).__name__}: {e}"}
    got, fz = sha256_file(CARD), frozen_card_sha()
    ctl["card_a6"] = {"PASS": bool(got == CARD_SHA == fz), "sha256": got, "pinned": CARD_SHA, "frozen_sha_file": fz}
    got = LR.card_rules_sha()
    ctl["cl_card_rules"] = {"PASS": got == LR.CARD_RULES_SHA, "sha256": got, "pinned": LR.CARD_RULES_SHA}
    ctl["lfo_read_py"] = {"PASS": sha256_file(LFO_READ_PY) == LFO_READ_SHA}
    ctl["lfo_read_json"] = {"PASS": sha256_file(LFO_READ_JSON) == LFO_READ_JSON_SHA}
    return ctl


def extract_new(out_root, arr_dir, workers):
    jobs = [(r[0], r[3]) for r in rollouts()]
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(rr.extract, tag, s, out_root, arr_dir): tag for tag, s in jobs}
        for f in cf.as_completed(futs):
            f.result()
    return {tag: f"{arr_dir}/{tag}.npz" for tag, _ in jobs}


# ------------------------------------------------------------------------------------------------ RESULTS.md
def fmt(r):
    return "%+.4f [%+.4f, %+.4f]%s" % (r["delta"], r["lo"], r["hi"], " *" if r["separated"] else "")


def render_md(res):
    sh = lambda c: "None" if c is None else "%.3f" % (float(c) + 0.0)
    J = res["joint"]
    L = [f"# LF6 read: the bounded-lift floor F_B at kappa 1.0 (A6), T1-kin, generated {res.get('clock', '')}", "", f"**Joint reading (A6 section 5): {J['case']}.** {J['next']}", "",
         "**LFB (the only admissible arms): " + "; ".join(f"delta {DELTAS[d]:.1f}: {res['LFB']['LFB' + d]['outcome']['label']} (counts as a result: {res['LFB']['LFB' + d]['outcome'].get('counts_as_result')})" for d in REPORT_ORDER) + f".** {MULTIPLICITY}", "",
         "Ceiling arms (every line CEILING): " + "; ".join(f"{k} {v['outcome']['label']}" for k, v in res["ceiling"].items()) + ".", "",
         CEILING_NOTE, "", f"Tier: {res['tier']}. Estimator: {res['estimator']}. `*` = interval excludes 0. STRATEGIC: {res['strategic_family']}", ""]
    for k, v in res["ceiling"].items():
        o = v["outcome"]
        L += [f"## {k} (CEILING): **{o['label']}**", ""]
        if v["rules"] is None:
            L += ["No rule is read (F-UNREACHED).", ""]
            continue
        R = v["rules"]
        P, G, F = R["P_O"], R["G_O"], R["F_O"]
        L += ["| rule | s0 | s1 | holds |", "|---|---|---|---|",
              "| P_O share of M1 closed >= 0.5, separated-better | " + " | ".join(f"sep {P['per_seed'][s]['separated_better']}, share {sh(P['per_seed'][s]['share_closed'])}" for s in ("s0", "s1")) + f" | **{P['PASS']}** |",
              "| G_O ADE6 <= +0.0604 m, no non-crawl row separated-worse | " + " | ".join(f"ADE6 {G['per_seed'][s]['ade6_delta']:+.4f}, worse: {', '.join(G['per_seed'][s]['non_crawl_separated_worse']) or 'none'}" for s in ("s0", "s1")) + f" | **{G['PASS']}** |",
              "| F_O (beside) | " + " | ".join(f"{F['per_seed'][s]['abs_R_minus_C0']:.4f} vs {F['per_seed'][s]['abs_C0rep_minus_C0']:.4f}" for s in ("s0", "s1")) + f" | **{F['PASS']}** |",
              f"| ceiling of D (never counted): read A {R['ceiling_of_D']['ceiling_of_D_read_a_both_seeds']} | " + " | ".join(f"D1 {R['ceiling_of_D']['by_seed'][s]['D1']}, D2 {R['ceiling_of_D']['by_seed'][s]['D2']}, D3 {R['ceiling_of_D']['by_seed'][s]['D3']}" for s in ("s0", "s1")) + " | counted: False |", ""]
        L += ["| row | family | s0 | s1 |", "|---|---|---|---|"] + [f"| {lab} | {SR.FAMILY[lab]} | " + " | ".join(fmt(v["seeds"][s]["RILFO_minus_RIC0"][lab]) for s in ("s0", "s1")) + " |" for lab in GD.LABELS] + [""]
        L += [f"Attribution (beside): {k} - RILFO100 (the same exact level at the unbounded lift)", "", "| row | family | s0 | s1 |", "|---|---|---|---|"] + [f"| {lab} | {SR.FAMILY[lab]} | " + " | ".join(fmt(v["seeds"][s]["ARM_minus_RILFO100"][lab]) for s in ("s0", "s1")) + " |" for lab in GD.LABELS] + [""]
    for name, d in res["LFB"].items():
        L += [f"## {name} (admissible, vision-only): **{d['outcome']['label']}** (the A3 rules)", "", f"Next (A3 wording; the joint reading above governs where they differ): {d['outcome']['next']}.", ""]
        if d["rules"]:
            R = d["rules"]
            L += [f"P {R['P']['PASS']}, G {R['G']['PASS']}, F {R['F']['PASS']}, D (read A, four families) {R['D']['DOMINANT_COUNTED']}, LEAK-BORNE {R['LEAK_BORNE']['LEAK_BORNE']}.", ""]
            L += ["| row | family | s0 | s1 |", "|---|---|---|---|"] + [f"| {lab} | {SR.FAMILY[lab]} | " + " | ".join(fmt(d["seeds"][s]["RILF_minus_RIC0"][lab]) for s in ("s0", "s1")) + " |" for lab in GD.LABELS] + [""]
        att = res["attribution_beside"][name]
        L += [f"Attribution (beside): {name} - RILF100 (the unbounded heads floor, A5)", "", "| row | family | s0 | s1 |", "|---|---|---|---|"] + [f"| {lab} | {SR.FAMILY[lab]} | " + " | ".join(fmt(att[s]["RILFB_minus_RILF100"][lab]) for s in ("s0", "s1")) + " |" for lab in GD.LABELS] + [""]
    L += ["## Instruments (beside, never deciding)", "", "| rollout | bind overall | bind in M1 ticks (n) | cap overall | mean lift where bound, m/s | share of bound ticks where the delta term binds | max (s' - s - delta t), m | v_hat vs logged: MAE m/s, relative MAE, share v_hat > logged (n) |",
          "|---|---|---|---|---|---|---|---|"]
    for tag, v in res["instruments"].items():
        e, lw, bd = v["vhat_vs_logged"]["all_ticks"], v["lift_where_bound_overall"], v["bounded"]
        L.append(f"| {tag} | {sh(v['bind_rate_overall']['rate'])} | {sh(v['bind_rate_in_m1_ticks']['rate'])} ({v['bind_rate_in_m1_ticks']['n']}) | {sh(v['cap_rate_overall']['rate'])} | {'None' if lw['mean_lift_mps'] is None else '%.3f' % lw['mean_lift_mps']} | "
                 f"{sh(bd['dterm_share_of_bound'])} | {'None' if bd['max_sp_minus_s_minus_delta_t_m'] is None else '%.2e' % bd['max_sp_minus_s_minus_delta_t_m']} | {sh(e['mae_mps'])}, {sh(e['relative_mae'])}, {sh(e['share_vhat_exceeds_logged'])} ({e['n']}) |")
    L += ["", "## Controls", "", "| control | PASS |", "|---|---|"] + [f"| {k} | {v.get('PASS')} |" for k, v in res.get("controls", {}).items()]
    L += ["", "## Variance named", ""] + [f"* {k}: {v}" for k, v in res["variance_questions"].items()]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("production", "ka-a5"), default="production")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--work-dir", default=WORK_DIR)
    ap.add_argument("--raw", default=f"{PK}/raw")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--results-md", default=f"{PK}/RESULTS.md")
    a = ap.parse_args(argv)
    init()
    rr.PK = PK
    rr.WORK = f"{a.work_dir}/rig"
    os.makedirs(a.raw, exist_ok=True)
    arr_dir = f"{a.raw}/arrays"
    res = {"card": "PREREG_LEVELFLOOR_A6.md (section 4: the ceiling arms by lfo_read.py's rules, LFB by lf_read.py's A3 rules)", "card_sha256": sha256_file(CARD), "reader": os.path.abspath(__file__).replace("\\", "/"), "reader_sha256": sha256_file(__file__),
           "mode": a.mode, "tier": "T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2)", "estimator": f"paired episode-cluster bootstrap B={a.n_boot} seed 0 (gapdecomp.decompose); answers another draw of EPISODES only",
           "clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "strategic_family": rr.STRATEGIC_NOTE, "ceiling_note": CEILING_NOTE, "multiplicity": MULTIPLICITY,
           "variance_questions": {"episodes": "answered by the paired episode-cluster bootstrap (every interval here)",
                                  "training_run": "O is C0's own checkpoint in every contrast with C0 / RIC0, so O's training variance is absent from it; the heads come from one LLR run per seed (not measured); F_O / F read against the C0REP floor",
                                  "inference": "every A6 arm is deterministic (no noise arm): the interval answers another draw of EPISODES only"}}
    base, rctl = {}, {}
    for s in SEEDS:
        base[s], c = reused_arrays(s)
        rctl.update(c)
    ctl = {"reused_arrays": {"PASS": all(v["PASS"] for v in rctl.values()), "files": rctl}}
    if a.mode == "ka-a5":                                                              # the banked A5 arrays in the A6 slots reproduce LF5_READ.json
        ctl["lf5_read_json"] = {"PASS": sha256_file(LF5_READ_JSON) == LF5_READ_JSON_SHA}
        ka_ar = {}
        for s in SEEDS:
            ka_ar[s] = {}
            for role in ("LF100", "LF100C"):
                tag = f"{role}_s{s}"
                p = f"{LF5_ARR}/{tag}.npz"
                got = sha256_file(p)
                ctl["reused_arrays"]["files"][tag] = {"path": p, "sha256": got, "pinned": LF5_KA_ARR_SHA[tag], "PASS": got == LF5_KA_ARR_SHA[tag]}
                ka_ar[s][role] = rr.read_arrays(p)
        ctl["reused_arrays"]["PASS"] = all(v["PASS"] for v in ctl["reused_arrays"]["files"].values())
        cache = {"base": {s: {k: v for k, v in base[s].items() if k in ("C0", "C0C", "C0REP", "RIC0", "RIC0C")} for s in SEEDS}, "c": {}}
        arm = read_ceiling_arm({s: {"RILFO": base[s]["RILFO100"], "RILFOC": base[s]["RILFO100C"]} for s in SEEDS}, {s: {"RILF": base[s]["RILFO100"], "RILFC": base[s]["RILFO100C"]} for s in SEEDS}, a.n_boot, None, cache)
        arm = relabel(arm)
        lfa = {s: {**cache["base"][s], "RILF": base[s]["RILF100"], "RILFC": base[s]["RILF100C"], "LF": ka_ar[s]["LF100"], "LFC": ka_ar[s]["LF100C"]} for s in SEEDS}
        dose = LFR.read_dose("B05", lfa, a.n_boot, None, {})
        ref = json.load(open(LF5_READ_JSON, encoding="utf-8"))
        j = lambda x: json.dumps(x, sort_keys=True, default=rr._jd)
        chk = {f"ceiling_{k}_equals_LF5_READ_LFO100": j(arm["rules"][k]) == j(ref["ceiling"]["LFO100"]["rules"][k]) for k in ("P_O", "G_O", "F_O", "ceiling_of_D")}
        chk["ceiling_label_equals_LF5_READ_LFO100"] = arm["outcome"]["label"] == ref["ceiling"]["LFO100"]["outcome"]["label"]
        chk.update({f"heads_{k}_equals_LF5_READ_LF100": j(dose["rules"][k]) == j(ref["LF100"]["rules"][k]) for k in ("P", "G", "F", "D", "LEAK_BORNE")})
        chk["heads_label_equals_LF5_READ_LF100"] = dose["outcome"]["label"] == ref["LF100"]["outcome"]["label"]
        zero = all(r["delta"] == 0.0 and not r["separated"] for s in SEEDS for r in arm["seeds"][f"s{s}"]["ARM_minus_RILFO100"].values())
        chk["attribution_of_the_banked_arm_against_itself_is_exactly_zero"] = bool(zero)
        res.update({"controls": ctl, "ka_checks": chk, "KA_PASS": bool(all(chk.values()) and ctl["reused_arrays"]["PASS"] and ctl["lf5_read_json"]["PASS"]), "label_ceiling": arm["outcome"]["label"], "label_heads": dose["outcome"]["label"]})
        json.dump(res, open(f"{a.raw}/KA_READER_ka-a5.json", "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
        print(f"KA ka-a5: {sum(chk.values())}/{len(chk)} checks hold; ceiling label {arm['outcome']['label']}; heads label {dose['outcome']['label']}")
        for k, v in chk.items():
            if not v:
                print("  RED", k)
        return 0 if res["KA_PASS"] else 2
    ctl.update(run_controls(a.out_root, a.raw))
    res["controls"] = ctl
    bad = [k for k, v in ctl.items() if not v.get("PASS")]
    if bad:
        res["refused"] = f"controls failed: {bad}"
        json.dump(res, open(f"{a.raw}/LF6_READ.json", "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
        print("REFUSED:", res["refused"])
        return 3
    inst = {}
    for tag, base_, comp, s, scene, feas, source, delta in rollouts():
        inst[tag] = tick_instruments6(load_ticks_strict(f"{a.out_root}/{tag}/ticks/ticks_{tag}.npz"), source, delta)
    res["instruments"] = inst
    paths = extract_new(a.out_root, arr_dir, a.workers)
    new = {s: {r[1] + ("C" if r[2] else ""): rr.read_arrays(paths[r[0]]) for r in rollouts() if r[3] == s} for s in SEEDS}
    cache = {"base": {s: {k: v for k, v in base[s].items() if k in ("C0", "C0C", "C0REP", "RIC0", "RIC0C")} for s in SEEDS}, "c": {}}
    att = {s: {"RILF": base[s]["RILFO100"], "RILFC": base[s]["RILFO100C"]} for s in SEEDS}
    res["ceiling"] = {}
    for arm in ("LFOB05", "LFOB10"):
        b_, dk = CEIL_ARMS[arm]
        reach = LFR.reach_control({f"s{s}": inst[f"{b_}_s{s}"] for s in SEEDS})
        arm_ar = {s: {"RILFO": new[s][b_], "RILFOC": new[s][b_ + "C"]} for s in SEEDS}
        res["ceiling"][arm] = relabel(read_ceiling_arm(arm_ar, att, a.n_boot, reach, cache))
        print(arm, res["ceiling"][arm]["outcome"]["label"], flush=True)
    res["LFB"], res["attribution_beside"], dose_cache = {}, {}, {}
    for arm in ("LFB05", "LFB10"):
        rb, lb, dk = LFB_ARMS[arm]
        lfa = {s: {**cache["base"][s], "RILF": new[s][rb], "RILFC": new[s][rb + "C"], "LF": new[s][lb], "LFC": new[s][lb + "C"]} for s in SEEDS}
        reach = LFR.reach_control({f"s{s}": inst[f"{rb}_s{s}"] for s in SEEDS})
        res["LFB"][arm] = LFR.read_dose("B" + dk, lfa, a.n_boot, reach, dose_cache)
        res["attribution_beside"][arm] = attribution_heads({s: {**cache["base"][s], "RILFB": new[s][rb], "RILFBC": new[s][rb + "C"], "RILF100": base[s]["RILF100"], "RILF100C": base[s]["RILF100C"]} for s in SEEDS}, a.n_boot)
        print(arm, res["LFB"][arm]["outcome"]["label"], flush=True)
    res["joint"] = joint_reading({dk: res["LFB"]["LFB" + dk]["outcome"]["label"] for dk in REPORT_ORDER}, {dk: res["ceiling"]["LFOB" + dk]["outcome"]["label"] for dk in REPORT_ORDER})
    res["heads_error_beside"] = {t: inst[t]["vhat_vs_logged"]["all_ticks"] for t in inst if t.startswith("RILFB")}
    json.dump(res, open(f"{a.raw}/LF6_READ.json", "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
    open(a.results_md, "w", encoding="utf-8", newline="\n").write(render_md(res))
    print("JOINT", res["joint"]["case"], "| LFB05", res["LFB"]["LFB05"]["outcome"]["label"], "| LFB10", res["LFB"]["LFB10"]["outcome"]["label"], "| LFOB05", res["ceiling"]["LFOB05"]["outcome"]["label"], "| LFOB10", res["ceiling"]["LFOB10"]["outcome"]["label"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
