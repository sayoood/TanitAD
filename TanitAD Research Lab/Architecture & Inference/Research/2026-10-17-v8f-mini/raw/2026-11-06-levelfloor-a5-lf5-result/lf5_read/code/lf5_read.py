#!/usr/bin/env python3
"""lf5_read.py -- the T1-kin read of PREREG_LEVELFLOOR_A5.md (sha256 2aac1dbe...): the level floor F at kappa 1.0 with the logged speed (LFO100, CEILING), the logged speed under a priori noise (LFN100x10, LFN100x20, CEILING) and the trained heads (LF100, the
only admissible arm). Planner C0_s{s}, lead cap from LLR_s{s}, operator F, the 3 m/s band and every constant as in LF075.
NOT a text clone: this module LOADS lfo_read.py (the A4 ceiling rules P_O / G_O / F_O, the CEILING-CLEARS / -HARMS / -SHORT / F-UNREACHED labels) and, through it, lf_read.py (the A3 rules for LF100: P, G, F, D read A over four families, LEAK-BORNE on `logged`,
the reach control) BY SHA-PINNED PATH and calls them unchanged. New here: the 20 rollouts and their flags, the noise instruments, the spec read sigma* (card section 4), the joint reading of card section 5, and the multiplicity note (three heads-floor doses).
Every ceiling-arm result carries the label CEILING (a privileged level input, not admissible at inference; never a driving result, never a counted DOMINANT). Only LF100 can count.
Modes:  production [--out-root D] | ka-a4  (the banked RILFO075 arrays in the LFO100 slot reproduce the A4 read's P_O / G_O / F_O / ceiling-of-D exactly; raw/KA_READER_ka-a4.json)
usage: python -I lf5_read.py --mode production [--workers 4]"""
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
PKG = f"{MC}/lf5_read"
RDB = f"{ART}/v8f-recover"
LFO_READ_PY, LFO_READ_SHA = f"{MC}/lfo_read/code/lfo_read.py", "35dae285c262943a43c6bd52e5eee6dcd7782440725c446c9a7d215fdba728c6"
LFO_READ_JSON, LFO_READ_JSON_SHA = f"{MC}/lfo_read/raw/LFO_READ.json", "f6c936aae77b6f1f240cad2e20d19319a2663aa0baf041bc34a4b4ae1c2d4387"
LFO_ARR = f"{MC}/lfo_read/raw/arrays"
LFO_ARR_SHA = {"RILFO075_s0": "be4ad91bad0335ab59d8674f141f09de82e00b2ae7aa538c707eecd8b38d95f8", "RILFO075_s1": "d6878a24f1cdddaac53952022b9359dd63189036d3523b5ee5b8ab14b3ac9fb7",
               "RILFO075C_s0": "2735fd1b0f862ce1122a82615b50c831c4155ac36fde7f416a57f8cd15e56d55", "RILFO075C_s1": "e2ebe2266f3d74661c8d17e62995e2ccc5f5c9b2e0fddf4e8a2a487dce7451e7"}
CARD, CARD_FROZEN = f"{MC}/a5_prereg/PREREG_LEVELFLOOR_A5.md", f"{MC}/a5_prereg/FROZEN_SHA.txt"
CARD_SHA = "2aac1dbea651ac82f599d874fce48f26d2cd83d19a8ec62f3353a28389333310"
BUILD_PINS = os.environ.get("LF5_BUILD_PINS", f"{MC}/lf5_build/PINS.txt")
OUT_ROOT, WORK_DIR = "F:/TanitAD-scratch/motion_c/lf5_read/out", "F:/TanitAD-scratch/motion_c/lf5_read/work"
SEEDS, KAPPA, KAPPA_FLAG, BASE_SMOOTH, NOISE_SEED = (0, 1), 1.0, "1.0", "sg72", 20261106
SIGMAS = (0.0, 0.10, 0.20)
SIG_FLAG = {0.10: "0.10", 0.20: "0.20"}
CEIL_ARMS = {"LFO100": ("RILFO100", 0.0), "LFN100x10": ("RILFN100x10", 0.10), "LFN100x20": ("RILFN100x20", 0.20)}
CEILING_NOTE = "CEILING: the level input is the human's logged route speed (exact, or under a priori noise), a privileged channel that is NOT admissible at inference. Not a driving result, not a capability claim, never a counted DOMINANT; it ranks levers (A5 card)."
MULTIPLICITY = "The heads-floor family has now read three doses (kappa 0.50, 0.75, 1.0): a counted DOMINANT names its dose and says that three were read (A5 card section 4)."
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
    """Load lfo_read.py BY FILE PATH (sha256 checked), run its init() (which loads lf_read.py, llr_read.py and the RECOVER-NEXT reader by pinned path), add the dose 100 to lf_read's tables, load lf5_precheck.py."""
    global LFO, LFR, LR, rr, SR, GD, PCO
    got = sha256_file(LFO_READ_PY)
    if got != LFO_READ_SHA:
        raise SystemExit(f"REFUSED: {LFO_READ_PY} is {got}, not the pinned {LFO_READ_SHA}")
    LFO = _load("lfo_read_for_lf5", LFO_READ_PY)
    LFO.init()
    LFR, LR, rr, SR, GD = LFO.LFR, LFO.LR, LFO.rr, LFO.SR, LFO.GD
    LFR.KAPPA["100"], LFR.KAPPA_FLAG["100"] = 1.0, "1.0"                 # lf_read.py's dose tables gain the dose 100 (its rules are unchanged)
    PCO = _load("lf5_precheck_for_read", f"{PKG}/code/lf5_precheck.py")
    return LR


# ------------------------------------------------------------------------------------------------ the 20 rollouts (card section 3), in PRIORITY order
def floor_ckpt(s):
    return f"{RDB}/M2b_s{s}_recft_LLR/ckpt_last.pt"


def rollouts():
    """[(tag, base, composite, seed, scene, feasible, source, sigma)] -- 20 rows. A killed chain yields the deciding admissible arm first, then the ceiling, then the noise arms, then LF100 on `logged`."""
    rows = []
    for base, scene, source, sigma in (("RILF100", "reindexed", "heads", None), ("RILFO100", "reindexed", "logged", 0.0), ("RILFN100x10", "reindexed", "logged", 0.10), ("RILFN100x20", "reindexed", "logged", 0.20), ("LF100", "logged", "heads", None)):
        for comp in (False, True):
            for s in SEEDS:
                rows.append((f"{base}{'C' if comp else ''}_s{s}", base, comp, s, scene, "kbox" if comp else "off", source, sigma))
    return rows


def expected_flags(source, sigma, scene, feas, s):
    f = f"--smooth {BASE_SMOOTH} --scene {scene}" + (f" --feasible {feas}" if feas != "off" else "") + f" --level-floor on --floor-kappa {KAPPA_FLAG} --floor-ckpt {floor_ckpt(s)} --floor-vhat-source {source}"
    if source == "logged" and sigma and sigma > 0:
        f += f" --floor-vhat-noise {SIG_FLAG[sigma]} --floor-vhat-noise-seed {NOISE_SEED}"
    return f


# ------------------------------------------------------------------------------------------------ pure layer 1: instruments
def _err_block(vh, v_log, sel):
    sel = sel & np.isfinite(vh) & np.isfinite(v_log) & (v_log > 0) & (vh > 0)
    if not sel.any():
        return {"n": 0, "mae_mps": None, "relative_mae": None, "log_ratio_mean": None, "log_ratio_std": None, "share_vhat_exceeds_logged": None}
    e, lr = vh[sel] - v_log[sel], np.log(vh[sel] / v_log[sel])
    return {"n": int(sel.sum()), "mae_mps": float(np.abs(e).mean()), "relative_mae": float((np.abs(e) / v_log[sel]).mean()), "log_ratio_mean": float(lr.mean()), "log_ratio_std": float(lr.std()), "share_vhat_exceeds_logged": float((e > 0).mean())}


def tick_instruments5(t, source, sigma):
    """The A3 instruments (bind overall and in the M1 ticks, cap, lift where bound incl. the negative chord lift) plus the error of v_hat against the logged speed on the same frames: for the noise arms the REALISED noise (MAE, relative MAE, std of log(v_hat / v_log)),
    for LF100 the heads' MAE, relative MAE and overshoot share. Beside, never deciding."""
    r = LFR.tick_instruments(t)
    v_log, ok = LFR.logged_speed(t["gt"], t["gt_ok"])
    vh = np.asarray(t["lf_vhat"], np.float64)
    m1 = LFR.m1_ticks(t["v0"], v_log, ok)
    r["vhat_vs_logged"] = {"all_ticks": _err_block(vh, v_log, np.ones(len(vh), bool)), "m1_ticks": _err_block(vh, v_log, m1)}
    if source == "logged" and (sigma or 0.0) == 0.0:
        r["oracle"] = LFO.oracle_block(t, KAPPA)
    if source == "logged" and (sigma or 0.0) > 0.0:
        r["noise"] = {"sigma": sigma, "realised_log_ratio_std": r["vhat_vs_logged"]["all_ticks"]["log_ratio_std"], "n_eps": int(len(t["lf_eps"])), "eps_mean": float(np.mean(t["lf_eps"])), "eps_std": float(np.std(t["lf_eps"]))}
    return r


def tick_controls5(t, source, sigma):
    """Tool-versus-card consistency of one tick dump. heads: lf_read's controls; logged: lfo_read's; with noise the 'v_hat is the logged speed' control is replaced by v_hat == v_log * exp(sigma * eps) and v_log == the reader's logged speed."""
    if source == "heads":
        return LFR.tick_controls(t, KAPPA)
    c = LFO.tick_controls(t, KAPPA)
    if (sigma or 0.0) > 0.0 and c.get("fields_present"):
        v_log, ok = LFR.logged_speed(t["gt"], t["gt_ok"])
        vl, vh, eps = np.asarray(t["lf_vlog"], np.float64), np.asarray(t["lf_vhat"], np.float64), np.asarray(t["lf_eps"], np.float64)
        both = np.isfinite(vl) & np.isfinite(v_log)
        c["v_hat_is_the_logged_speed_of_the_dumped_gt"] = bool(both.any() and np.abs(vl[both] - v_log[both]).max() <= 1e-5 and np.array_equal(np.isnan(vl), ~ok))
        fin = np.isfinite(vl) & np.isfinite(vh)
        c["v_hat_is_v_log_times_exp_sigma_eps"] = bool(np.abs(vh[fin] - vl[fin] * np.exp(sigma * eps[fin])).max() <= 1e-4 * max(1.0, float(np.abs(vl[fin]).max())))
        c["one_draw_per_tick_row"] = bool(len(eps) == len(vh) and np.isfinite(eps).all())
    return c


# ------------------------------------------------------------------------------------------------ pure layer 2: sigma*, the joint reading (card sections 4-5)
def sigma_star(holds):
    """The largest sigma in {0, 0.10, 0.20} at which P_O and G_O both hold at both seeds (card section 4). ``holds`` = {sigma: bool}. -> {sigma_star, holds, monotone}."""
    ok = [s for s in SIGMAS if holds.get(s) is True]
    seq = [holds.get(s) is True for s in SIGMAS]
    return {"sigma_star": max(ok) if ok else None, "holds_by_sigma": {str(s): holds.get(s) for s in SIGMAS}, "monotone": bool(all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))),
            "definition": "largest sigma in {0, 0.10, 0.20} at which P_O and G_O both hold at both seeds; reported, never deciding (one noise draw)"}


def joint_reading(lfo_label, lf_label, sstar):
    """Card section 5, exactly one case. ``lf_label`` = LF100's A3 label (D-HOLDS = counted DOMINANT). Returns {case, counted_dominant, estimator_levers_signalled, next}."""
    if lf_label == "D-HOLDS":
        return {"case": "LF100-COUNTED-DOMINANT", "counted_dominant": True, "estimator_levers_signalled": None,
                "next": "COUNTED DOMINANT at kappa 1.0 with P and G, not LEAK-BORNE; signal the PI for scale (data and size), with the video evidence. " + MULTIPLICITY}
    if lfo_label == "CEILING-CLEARS":
        if sstar is not None and sstar >= 0.10:
            return {"case": "FORM-WORKS-GIVEN-A-GOOD-LEVEL", "counted_dominant": False, "estimator_levers_signalled": True,
                    "next": f"the form works given a good level and sigma* = {sstar:.2f} is the spec: the estimator levers (the A40, a stronger estimator on more data) go to the PI with the ceiling, sigma* and the heads' measured error as the case; the next CPU arm is a temporal filter of the heads' level, in its own addendum"}
        return {"case": "FORM-NEEDS-AN-EXACT-LEVEL", "counted_dominant": False, "estimator_levers_signalled": False,
                "next": "sigma* = 0: the form needs an exact level; reported as such, and no estimator lever is signalled on this evidence"}
    if lfo_label == "CEILING-HARMS":
        return {"case": "OPERATOR-IS-THE-PROBLEM", "counted_dominant": False, "estimator_levers_signalled": False,
                "next": "matching the human's speed along O's path harms on the named rows, so the operator is the problem; the next arm changes F, in its own addendum; no estimator lever is signalled"}
    if lfo_label == "CEILING-SHORT":
        return {"case": "PROGRESS-FLOOR-FORM-EXHAUSTED", "counted_dominant": False, "estimator_levers_signalled": False,
                "next": "even the human's exact speed, reached by the 1.0 m/s^2 ramp along O's path, does not close half the crawl: the progress-floor form is exhausted; what remains goes to the PI: T2 re-perception (option (c)) and the A40"}
    return {"case": "F-UNREACHED", "counted_dominant": False, "estimator_levers_signalled": False, "next": "the instrument is checked first"}


# ------------------------------------------------------------------------------------------------ the read (arrays)
def read_ceiling_arm(arm_ar, att_ar, n_boot, reach, cache, log=print):
    """One ceiling arm. ``arm_ar`` = {seed: {RILFO, RILFOC}}; ``att_ar`` = {seed: {RILF, RILFC}}: the attribution base (the heads-floor arm of the same kappa). Rules P_O / G_O / F_O and the label are lfo_read.read_ceiling's, unchanged."""
    arrays = {s: {**cache["base"][s], **arm_ar[s], **att_ar[s]} for s in SEEDS}
    return LFO.read_ceiling(arrays, n_boot, reach, cache["c"], log)


def relabel(b):
    """In this read the attribution slot RILF holds the heads-floor arm of the same dose (RILF100): rename lfo_read's keys for what they are here."""
    for s, d in (b.get("seeds") or {}).items():
        for old, new in (("RILFO_minus_RILF075", "ARM_minus_RILF100"), ("RILFOC_minus_RILF075C", "ARMC_minus_RILF100C"), ("RILF075_minus_RIC0_beside", "RILF100_minus_RIC0_beside")):
            if old in d:
                d[new] = d.pop(old)
        d["attribution_rows_changed_vs_RILF100"] = d.pop("attribution_rows_changed_vs_RILF075", None)
    return b


def extra_contrasts(arrays, n_boot):
    """Beside, never deciding (card section 3 bases): the dose effects RILFO100 - RILFO075 and RILF100 - RILF075, per seed (19 rows each)."""
    out = {}
    for s in SEEDS:
        ar = arrays[s]
        eid = ar["C0"]["eid"]
        cls, _ = GD.classify(ar["C0"]["v0"], ar["C0"]["g"])
        out[f"s{s}"] = {"RILFO100_minus_RILFO075": LFO._rows(ar, eid, cls, "RILFO100", "RILFO075", n_boot)[0], "RILF100_minus_RILF075": LFO._rows(ar, eid, cls, "RILF100", "RILF", n_boot)[0]}
    return out


def reused_arrays(s):
    z, ctl = LFO.reused_arrays(s)                                                      # C0 C0C C0REP RIC0 RIC0C (pinned) + RILF = RILF075, RILFC = RILF075C (pinned)
    for role, tag in (("RILFO075", f"RILFO075_s{s}"), ("RILFO075C", f"RILFO075C_s{s}")):
        p = f"{LFO_ARR}/{tag}.npz"
        got = sha256_file(p)
        ctl[tag] = {"path": p, "sha256": got, "pinned": LFO_ARR_SHA[tag], "PASS": got == LFO_ARR_SHA[tag]}
        z[role] = rr.read_arrays(p)
    return z, ctl


# ------------------------------------------------------------------------------------------------ controls
def frozen_card_sha():
    for line in open(CARD_FROZEN, encoding="utf-8"):
        t = line.split()
        if len(t) >= 2 and t[0] == os.path.basename(CARD):
            return t[1]
    return None


def run_controls(out_root, raw):
    ctl, ckpts, eps = {}, {}, {}
    try:
        tool_pin = LFR.parse_pins(BUILD_PINS).get("patched_tool")
    except Exception:
        tool_pin = None
    ctl["patched_tool_pin"] = {"PASS": bool(tool_pin and len(tool_pin) == 64), "patched_tool": tool_pin}
    for tag, base, comp, s, scene, feas, source, sigma in rollouts():
        c = {"PASS": False}
        try:
            d = json.load(open(f"{out_root}/done_{tag}.txt", encoding="utf-8"))
            h = rr.run_header(out_root, tag)
            fe = (h.get("feasible") or {}).get("mode", "off") if isinstance(h.get("feasible"), dict) else ("off" if not h.get("feasible") else str(h.get("feasible")))
            sm = h["smooth"]["mode"] if isinstance(h.get("smooth"), dict) else h.get("smooth")
            fc, _ = PCO.check_rollout(out_root, tag)
            t = np.load(f"{out_root}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True)
            td = {k: t[k] for k in t.files}
            tc = tick_controls5(td, source, sigma)
            c = {"flags_exact": d["flags"] == expected_flags(source, sigma, scene, feas, s), "tool_sha256_is_patched_tool": bool(tool_pin and d["tool_sha256"] == tool_pin),
                 "ckpt_is_C0": d["ckpt"].replace("\\", "/").endswith(f"/M2b_s{s}_recft_C0/ckpt_last.pt"), "header_scene": h["scene"] == scene, "header_feasible": fe == feas, "header_smooth": sm == BASE_SMOOTH,
                 "header_n_windows_2399": int(h["n_windows"]) == 2399, "header_kin_yaw": h["kin_mode"] == "yaw", "ckpt_sha256": d["ckpt_sha256"], "threads": d.get("threads")}
            c.update({f"floor_{k}": v for k, v in fc.items()})
            c.update({f"ticks_{k}": v for k, v in tc.items()})
            ckpts.setdefault(s, set()).add(d["ckpt_sha256"])
            if (sigma or 0.0) > 0:
                eps[tag] = np.asarray(td["lf_eps"], np.float64)
            c["PASS"] = all(v for k, v in c.items() if isinstance(v, bool))
        except Exception as e:
            c["error"] = f"{type(e).__name__}: {e}"
        ctl[f"rollout_{tag}"] = c
    same = {}
    for tag, e in eps.items():                                                          # the draws are identical between the bare and the composite rollout of one seed (card section 2, known answer 2), and between seeds (one generator seed, one window order)
        if "C_s" not in tag:
            ctag = tag.replace("_s", "C_s")
            same[f"{tag}_vs_{ctag}"] = bool(ctag in eps and len(eps[ctag]) == len(e) and np.array_equal(eps[ctag], e))
    ctl["noise_draws_identical_bare_vs_composite"] = {"PASS": bool(same) and all(same.values()), "pairs": same}
    ctl["one_checkpoint_per_seed"] = {"PASS": bool(len(ckpts) == len(SEEDS) and all(len(v) == 1 and next(iter(v)) == PCO.C0_SHA[s] for s, v in ckpts.items())), "ckpt_sha256": {f"s{s}": sorted(v) for s, v in ckpts.items()}}
    try:
        pc = json.load(open(f"{raw}/LF5_PRECHECK.json", encoding="utf-8"))
        ctl["LF5_PRECHECK"] = {"PASS": bool(pc["PASS"] is True and all(pc["seeds"][f"s{s}"]["C0"]["ckpt_sha256"] == PCO.C0_SHA[s] and pc["seeds"][f"s{s}"]["LLR"]["ckpt_sha256"] == PCO.llr_sha_of(s) for s in SEEDS))}
    except Exception as e:
        ctl["LF5_PRECHECK"] = {"PASS": False, "error": f"{type(e).__name__}: {e}"}
    got, fz = sha256_file(CARD), frozen_card_sha()
    ctl["card_a5"] = {"PASS": bool(got == CARD_SHA == fz), "sha256": got, "pinned": CARD_SHA, "frozen_sha_file": fz}
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
    L = [f"# LF5 read: the level floor at kappa 1.0 (A5), T1-kin, generated {res.get('clock', '')}", "", f"**Joint reading (A5 section 5): {J['case']}.** {J['next']}", "",
         f"**LF100 (the only admissible arm): {res['LF100']['outcome']['label']}** (counts as a result: {res['LF100']['outcome'].get('counts_as_result')}). {MULTIPLICITY}", "",
         f"Ceiling arms (every line CEILING): " + "; ".join(f"{k} {v['outcome']['label']}" for k, v in res["ceiling"].items()) + f". sigma* = {res['sigma_star']['sigma_star']} (holds by sigma {res['sigma_star']['holds_by_sigma']}; monotone {res['sigma_star']['monotone']}; reported, never deciding).", "",
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
    d = res["LF100"]
    L += ["## LF100 (admissible, vision-only): the A3 rules", "", f"Next (A3 wording; the joint reading above governs where they differ): {d['outcome']['next']}.", ""]
    if d["rules"]:
        R = d["rules"]
        L += [f"P {R['P']['PASS']}, G {R['G']['PASS']}, F {R['F']['PASS']}, D (read A, four families) {R['D']['DOMINANT_COUNTED']}, LEAK-BORNE {R['LEAK_BORNE']['LEAK_BORNE']}.", ""]
        L += ["| row | family | s0 | s1 |", "|---|---|---|---|"] + [f"| {lab} | {SR.FAMILY[lab]} | " + " | ".join(fmt(d["seeds"][s]["RILF_minus_RIC0"][lab]) for s in ("s0", "s1")) + " |" for lab in GD.LABELS] + [""]
    L += ["## Instruments (beside, never deciding)", "", "| rollout | bind overall | bind in M1 ticks (n) | cap overall | mean lift where bound, m/s | v_hat vs logged: MAE m/s, relative MAE, log-ratio std, share v_hat > logged (n) |", "|---|---|---|---|---|---|"]
    for tag, v in res["instruments"].items():
        e, lw = v["vhat_vs_logged"]["all_ticks"], v["lift_where_bound_overall"]
        L.append(f"| {tag} | {sh(v['bind_rate_overall']['rate'])} | {sh(v['bind_rate_in_m1_ticks']['rate'])} ({v['bind_rate_in_m1_ticks']['n']}) | {sh(v['cap_rate_overall']['rate'])} | {'None' if lw['mean_lift_mps'] is None else '%.3f' % lw['mean_lift_mps']} | "
                 f"{sh(e['mae_mps'])}, {sh(e['relative_mae'])}, {sh(e['log_ratio_std'])}, {sh(e['share_vhat_exceeds_logged'])} ({e['n']}) |")
    sp = res["spec_read"]
    L += ["", "## Spec read (reported, never deciding)", "", f"sigma* = {sp['sigma_star']}. The heads' realised error on the same frames (LF100 reindexed rollouts, std of log(v_hat / v_log) and relative MAE): " + "; ".join(f"{t}: {sh(x['log_ratio_std'])} / {sh(x['relative_mae'])}" for t, x in sp["heads_error_by_rollout"].items()) + ".",
          "Noise arms' realised log-ratio std: " + "; ".join(f"{t}: {sh(x)}" for t, x in sp["noise_realised_std_by_rollout"].items()) + ".", ""]
    L += ["## Controls", "", "| control | PASS |", "|---|---|"] + [f"| {k} | {v.get('PASS')} |" for k, v in res.get("controls", {}).items()]
    L += ["", "## Variance named", ""] + [f"* {k}: {v}" for k, v in res["variance_questions"].items()]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("production", "ka-a4"), default="production")
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
    res = {"card": "PREREG_LEVELFLOOR_A5.md (section 4: the ceiling arms by lfo_read.py's rules, LF100 by lf_read.py's A3 rules)", "card_sha256": sha256_file(CARD), "reader": os.path.abspath(__file__).replace("\\", "/"), "reader_sha256": sha256_file(__file__),
           "mode": a.mode, "tier": "T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2)", "estimator": f"paired episode-cluster bootstrap B={a.n_boot} seed 0 (gapdecomp.decompose); answers another draw of EPISODES only",
           "clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "strategic_family": rr.STRATEGIC_NOTE, "ceiling_note": CEILING_NOTE, "multiplicity": MULTIPLICITY,
           "variance_questions": {"episodes": "answered by the paired episode-cluster bootstrap (every interval here)",
                                  "training_run": "O is C0's own checkpoint in every contrast with C0 / RIC0, so O's training variance is absent from it; the heads come from one LLR run per seed (not measured); F_O / F read against the C0REP floor",
                                  "inference": "LFO100 and LF100 are deterministic. The noise arms carry ONE noise draw (one generator seed 20261106, the same draws at both seeds): a difference between noise levels smaller than a second draw would show is not resolved; sigma* is reported, never deciding"}}
    base, rctl = {}, {}
    for s in SEEDS:
        base[s], c = reused_arrays(s)
        rctl.update(c)
    ctl = {"reused_arrays": {"PASS": all(v["PASS"] for v in rctl.values()), "files": rctl}}
    if a.mode == "ka-a4":
        arrays = {s: {**base[s], "RILFO": base[s]["RILFO075"], "RILFOC": base[s]["RILFO075C"]} for s in SEEDS}      # the banked RILFO075 arrays in the LFO100 slot; the attribution slot RILF = RILF075 as in A4
        arm = LFO.read_ceiling(arrays, a.n_boot, None)
        ref = json.load(open(LFO_READ_JSON, encoding="utf-8"))["arm"]
        j = lambda x: json.dumps(x, sort_keys=True, default=rr._jd)
        chk = {f"{k}_equals_LFO_READ": j(arm["rules"][k]) == j(ref["rules"][k]) for k in ("P_O", "G_O", "F_O", "ceiling_of_D")}
        chk["label_equals_LFO_READ"] = arm["outcome"]["label"] == ref["outcome"]["label"]
        chk["label_is_CEILING_SHORT"] = arm["outcome"]["label"] == "CEILING-SHORT"
        res.update({"controls": ctl, "ka_checks": chk, "KA_PASS": bool(all(chk.values()) and ctl["reused_arrays"]["PASS"]), "label": arm["outcome"]["label"]})
        json.dump(res, open(f"{a.raw}/KA_READER_ka-a4.json", "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
        print(f"KA ka-a4: {sum(chk.values())}/{len(chk)} checks hold; label {arm['outcome']['label']}")
        for k, v in chk.items():
            if not v:
                print("  RED", k)
        return 0 if res["KA_PASS"] else 2
    ctl.update(run_controls(a.out_root, a.raw))
    res["controls"] = ctl
    bad = [k for k, v in ctl.items() if not v.get("PASS")]
    if bad:
        res["refused"] = f"controls failed: {bad}"
        json.dump(res, open(f"{a.raw}/LF5_READ.json", "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
        print("REFUSED:", res["refused"])
        return 3
    inst = {}
    for tag, base_, comp, s, scene, feas, source, sigma in rollouts():
        t = np.load(f"{a.out_root}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True)
        inst[tag] = tick_instruments5({k: t[k] for k in t.files}, source, sigma)
    res["instruments"] = inst
    paths = extract_new(a.out_root, arr_dir, a.workers)
    new = {s: {r[1] + ("C" if r[2] else ""): rr.read_arrays(paths[r[0]]) for r in rollouts() if r[3] == s} for s in SEEDS}
    cache = {"base": {s: {k: v for k, v in base[s].items() if k in ("C0", "C0C", "C0REP", "RIC0", "RIC0C")} for s in SEEDS}, "c": {}}
    att = {s: {"RILF": new[s]["RILF100"], "RILFC": new[s]["RILF100C"]} for s in SEEDS}
    res["ceiling"] = {}
    for arm, (b_, sigma) in CEIL_ARMS.items():
        reach = LFR.reach_control({f"s{s}": inst[f"{b_}_s{s}"] for s in SEEDS})
        arm_ar = {s: {"RILFO": new[s][b_], "RILFOC": new[s][b_ + "C"]} for s in SEEDS}
        res["ceiling"][arm] = relabel(read_ceiling_arm(arm_ar, att, a.n_boot, reach, cache))
        print(arm, res["ceiling"][arm]["outcome"]["label"], flush=True)
    lfa = {s: {**cache["base"][s], "RILF": new[s]["RILF100"], "RILFC": new[s]["RILF100C"], "LF": new[s]["LF100"], "LFC": new[s]["LF100C"]} for s in SEEDS}
    reach = LFR.reach_control({f"s{s}": inst[f"RILF100_s{s}"] for s in SEEDS})
    res["LF100"] = LFR.read_dose("100", lfa, a.n_boot, reach, {})
    res["extra_contrasts_beside"] = extra_contrasts({s: {**cache["base"][s], "RILF": base[s]["RILF"], "RILFO075": base[s]["RILFO075"], "RILFO100": new[s]["RILFO100"], "RILF100": new[s]["RILF100"]} for s in SEEDS}, a.n_boot)
    holds = {sg: (res["ceiling"][arm]["outcome"]["label"] != "F-UNREACHED" and bool(res["ceiling"][arm]["rules"]["P_O"]["PASS"] and res["ceiling"][arm]["rules"]["G_O"]["PASS"])) for arm, (_b, sg) in CEIL_ARMS.items()}
    res["sigma_star"] = sigma_star(holds)
    res["joint"] = joint_reading(res["ceiling"]["LFO100"]["outcome"]["label"], res["LF100"]["outcome"]["label"], res["sigma_star"]["sigma_star"])
    res["spec_read"] = {"sigma_star": res["sigma_star"]["sigma_star"], "heads_error_by_rollout": {t: inst[t]["vhat_vs_logged"]["all_ticks"] for t in inst if t.startswith("RILF100_")},
                        "noise_realised_std_by_rollout": {t: inst[t]["vhat_vs_logged"]["all_ticks"]["log_ratio_std"] for t in inst if t.startswith("RILFN")}}
    json.dump(res, open(f"{a.raw}/LF5_READ.json", "w", encoding="utf-8", newline="\n"), indent=1, default=rr._jd)
    open(a.results_md, "w", encoding="utf-8", newline="\n").write(render_md(res))
    print("JOINT", res["joint"]["case"], "| LF100", res["LF100"]["outcome"]["label"], "| sigma*", res["sigma_star"]["sigma_star"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
