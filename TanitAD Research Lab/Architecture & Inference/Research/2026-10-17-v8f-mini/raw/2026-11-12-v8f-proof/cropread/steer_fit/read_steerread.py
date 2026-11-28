#!/usr/bin/env python
"""read_steerread.py -- read of card PREREG_V8F_PROOF_ADD100 (STEERREAD): controls s1..s8, then the names. Tier T0. Numpy only (no torch).

  python read_steerread.py --stage controls --gaprange-result GAPRANGE_RESULT.json --s7-pytest-out FILE [--root S_ROOT] [--run1 R1] [--box-dir D]
  python read_steerread.py --stage names [--root S_ROOT] [--run1 R1] [--box-dir D]
  python read_steerread.py --selftest

The estimator and the wrapper are those of card ADD99 (read_cropread.load_ctxs / measure_dumps / cut_ratio / cut_name, kinclose4 through them), unchanged.
Stage controls writes <root>/read/STEERREAD_CONTROLS.json (s1..s8, one line each). Stage names refuses unless all eight passed (nothing is opened) and writes
<root>/read/STEERREAD_RESULT.json. Last line: ZZ READ-STEERREAD <stage> rc <rc> controls_all_pass <bool>.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import steer_common as SC  # noqa: E402  (puts the kit of card ADD99 on sys.path)
import read_cropread as RC  # noqa: E402

K2, K4, K6 = RC.K2, RC.K4, RC.K6
CFT = SC.CFT
CARD = SC.CARD
CTL_ORDER = ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8")
KS = RC.KS
EXPECTED_FITS = ["S_H5_s0", "S_H5d_s0", "S_H5d_s0_perm", "S_H5_s1", "S_H5d_s1"]
CUT_STEM, CUT_STEM_TOK = "STEERREAD-CUT-BY-2", "STEERREAD-TOKENS-ONLY-CUT-BY-2"
PLANTED_CUTS = ((0.4, "REACHED"), (0.9, "NOT-REACHED"))
S6_SD_RATIO_MIN = 0.8


# ============================================================================================ name rules of card section 4
def vs_fixed_name(cut):
    """ratio S-H5d / D-H5: BEATS-FIXED when the upper end is below 1, WORSE-THAN-FIXED when the lower end is above 1, else NOT-SEPARATED-FROM-FIXED (also when there is no interval)."""
    ci = cut.get("ci")
    if not ci or not np.all(np.isfinite(ci)):
        return "STEERREAD-NOT-SEPARATED-FROM-FIXED"
    if ci[1] < 1.0:
        return "STEERREAD-BEATS-FIXED"
    if ci[0] > 1.0:
        return "STEERREAD-WORSE-THAN-FIXED"
    return "STEERREAD-NOT-SEPARATED-FROM-FIXED"


def vs_bound_name(cut):
    """ratio S-H5d / B-H5: BELOW-BOUND when the lower end is above 1, else NOT-SEPARATED-FROM-BOUND."""
    ci = cut.get("ci")
    if ci and np.all(np.isfinite(ci)) and ci[0] > 1.0:
        return "STEERREAD-BELOW-BOUND"
    return "STEERREAD-NOT-SEPARATED-FROM-BOUND"


# ============================================================================================ controls
def read_lists_safe(run1):
    try:
        return SC.read_lists(run1), None
    except SystemExit as e:
        return None, str(e)


def control_s1(root, run1):
    """every localiser's record lists the clips it read; no train clip is steered by a localiser that read it; no evaluation clip is read by a localiser or by a head fit."""
    bad = []
    j = RC.jpass(os.path.join(root, "ctl", "S1.json"))
    if not j["pass"]:
        bad.append("ctl/S1.json: " + j["detail"])
    lists, err = read_lists_safe(run1)
    if err:
        return {"pass": False, "detail": {"failures": bad + [err]}}
    rows, ev = lists
    assign = SC.steer_assignment(rows, ev)
    recs = {}
    for n in SC.LOC_NAMES:
        p = os.path.join(root, "loc", n, "LOC_RECORD.json")
        if os.path.isfile(p):
            recs[n] = SC.read_json(p)
    bad += SC.audit_assignment(assign, recs, rows, ev)
    train = {c for c, _r in rows}
    evs = set(ev)
    sha_train = SC.sha256_file(SC.paths(run1)["train_list"])
    fits = {}
    for n in EXPECTED_FITS:
        p = os.path.join(root, "fits", n, "FIT_RECORD.json")
        if not os.path.isfile(p):
            bad.append("fit record missing: " + n)
            continue
        r = SC.read_json(p)
        read = list(r["clips_read"]["fit"]) + list(r["clips_read"]["sel"])
        if not set(read) <= train:
            bad.append("%s read a clip outside the train list" % n)
        if set(read) & evs:
            bad.append("%s read an evaluation clip" % n)
        if r.get("train_list_sha256") != sha_train:
            bad.append("%s: train list sha256 differs from the file" % n)
        fits[n] = {"n_clips_read": len(read), "hparams_equal_card": r.get("hparams_equal_card")}
    return {"pass": not bad, "detail": {"failures": bad, "n_localiser_records": len(recs), "head_fits": fits, "s1_json": j["detail"]}}


def actual_steer_shas(root, clips):
    out, miss = {}, []
    for c in clips:
        p = os.path.join(root, "steer", c + ".npy")
        if os.path.isfile(p):
            out[c] = SC.sha256_file(p)
        else:
            miss.append(c)
    return out, miss


def control_s2(root, run1):
    """steering is the localiser's output: the first half (ctl/S2.json, 6 evaluation clips reproduced within 1e-3 degrees) and the sha256 of every steering file used carried by the
    data side (ctl/S8.json or tok/S/MANIFEST.json) and by ctl/S1.json equal the sha256 of the files on disk."""
    bad = []
    lists, err = read_lists_safe(run1)
    if err:
        return {"pass": False, "detail": {"failures": [err]}}
    rows, ev = lists
    clips = [c for c, _r, _l in SC.steer_assignment(rows, ev)]
    j = RC.jpass(os.path.join(root, "ctl", "S2.json"))
    if not j["pass"]:
        bad.append("ctl/S2.json: " + j["detail"])
    else:
        s2 = SC.read_json(os.path.join(root, "ctl", "S2.json"))
        if s2.get("clips") != SC.first_eval_by_sha(ev):
            bad.append("ctl/S2.json clips are not the first 6 evaluation clips by sha256 hex of the clip id")
        if s2.get("tol_deg") != SC.S2_TOL_DEG:
            bad.append("ctl/S2.json tolerance is not 1e-3 degrees")
        for c, v in s2.get("per_clip", {}).items():
            if not (isinstance(v.get("max_abs_diff_deg"), (int, float)) and v["max_abs_diff_deg"] <= SC.S2_TOL_DEG):
                bad.append("ctl/S2.json: %s difference above 1e-3 degrees" % c)
    act, miss = actual_steer_shas(root, clips)
    if miss:
        bad.append("%d steering file(s) missing, first %s" % (len(miss), miss[0]))
    carried = {}
    for label, p in (("S1", os.path.join(root, "ctl", "S1.json")), ("S8", os.path.join(root, "ctl", "S8.json")), ("tok_S_MANIFEST", os.path.join(root, "tok", "S", "MANIFEST.json"))):
        if os.path.isfile(p):
            m = SC.find_steer_shas(SC.read_json(p), clips)
            carried[label] = len(m)
            if m:
                wrong = [c for c in clips if m.get(c) != act.get(c)]
                if wrong:
                    bad.append("%s carries a steering sha256 that differs from the file (%d clips, first %s)" % (label, len(wrong), wrong[0]))
    if not carried.get("S8") and not carried.get("tok_S_MANIFEST"):
        bad.append("neither ctl/S8.json nor tok/S/MANIFEST.json carries the steering sha256 map")
    return {"pass": not bad, "detail": {"failures": bad, "carried_counts": carried, "n_files_hashed": len(act), "s2_json": j["detail"]}}


def control_s5(present_dir, ep_map, B):
    """control c3 of card ADD99 read again, and the planted cut checks: the present reader's error scaled to 0.4 gives -REACHED, to 0.9 gives -NOT-REACHED."""
    c3 = RC.control_c3(present_dir, ep_map, B)
    raws = RC.raw_dumps(present_dir)
    cr = [K4.make_ctx(D, ep_map) for D in raws]
    planted, ok = [], bool(c3["pass"])
    for scale, want in PLANTED_CUTS:
        cp = [K4.make_ctx(RC.scaled_dump(D, scale), ep_map) for D in raws]
        cut = RC.cut_ratio(cp, cr, min(B, RC.CUT_B))
        name = RC.cut_name(cut, "S5")
        planted.append({"scale": scale, "ratio": cut.get("ratio"), "ci": cut.get("ci"), "name": name, "wanted": "S5-" + want})
        ok = ok and name == "S5-" + want
    return {"pass": bool(ok), "detail": {"c3": c3, "planted_cuts": planted}}


def control_s6(root, ep_map, B):
    """S-H5d fit with permuted label gaps (seed 0): not the name REACHES, and the standard deviation of its gap error is at least 0.8 times that of the label gap."""
    d = os.path.join(root, "dump", "S_H5d_perm")
    if not os.path.isfile(os.path.join(d, "reader_s0_score.npz")):
        return {"pass": False, "detail": {"failures": ["no dump " + d]}}
    c = K4.make_ctx(K2.CF.load_dump(os.path.join(d, "reader_s0_score.npz")), ep_map)
    P = RC.measure_dumps([c], B)
    m = K4.meas(c)
    sd_lab = float(c["LG"][c["ci"], c["fr"]][m].std())
    sd_err = float(P["gap_error"]["ALL"]["sd"])
    name = RC.form_name(P, "S6")
    ok = bool(name != "S6-REACHES" and sd_err >= S6_SD_RATIO_MIN * sd_lab)
    return {"pass": ok, "detail": {"name": name, "gap_error_sd_m": sd_err, "label_gap_sd_m": sd_lab, "ratio": sd_err / sd_lab if sd_lab > 0 else None, "min_ratio": S6_SD_RATIO_MIN}}


def run_controls(a, log=print):
    out = a.out or os.path.join(a.root, "read", "STEERREAD_CONTROLS.json")
    tm = bool(getattr(a, "test_mode", False))
    if a.boot != 2000 and not tm:
        raise SystemExit("REFUSED: s4 compares with a bootstrap of 2,000 draws; --boot must be 2000")
    if os.path.exists(out):
        raise SystemExit("REFUSED: %s exists" % out)
    K2.load_tools(a.closeread_tools)
    ep_map = RC.read_json(a.episode_map)
    ctl = {"s1": control_s1(a.root, a.run1), "s2": control_s2(a.root, a.run1),
           "s3": RC.jpass(os.path.join(a.root, "ctl", "S3.json")),
           "s4": RC.control_c2(a.present_dump, ep_map, a.gaprange_result, a.boot, check_shas=not tm),
           "s5": control_s5(a.present_dump, ep_map, a.boot), "s6": control_s6(a.root, ep_map, a.boot),
           "s7": RC.control_c6(a.s7_pytest_out), "s8": RC.jpass(os.path.join(a.root, "ctl", "S8.json"))}
    ctl = {n: ctl[n] for n in CTL_ORDER}
    allp = bool(all(ctl[n]["pass"] for n in CTL_ORDER))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    K2.write_json(out, {"card": CARD, "controls": ctl, "controls_all_pass": allp, "boot": a.boot})
    for n in CTL_ORDER:
        log("%s %s" % (n, "PASS" if ctl[n]["pass"] else "FAIL"))
    log("ZZ READ-STEERREAD controls rc %d controls_all_pass %s" % (0 if allp else 1, allp))
    return 0 if allp else 1


def check_controls(C):
    """-> (ok, why): s1..s8 all present with pass True and the file says controls_all_pass."""
    ctl = C.get("controls", {})
    bad = [n for n in CTL_ORDER if not (isinstance(ctl.get(n), dict) and ctl[n].get("pass") is True)]
    if bad or C.get("controls_all_pass") is not True:
        return False, "controls not passed: %s" % (bad or "file says controls_all_pass false")
    return True, ""


# ============================================================================================ values printed with no name
def _abs_stats(e):
    e = np.abs(np.asarray(e, np.float64))
    return {"median": float(np.median(e)), "p90": float(np.percentile(e, 90))} if len(e) else {"median": float("nan"), "p90": float("nan")}


def localiser_stats(run1_box_dir, s_root, present_dump):
    """the localiser's error (steering minus box centre, degrees) on frames with a lead box (has_lead and finite angles) of the evaluation clips, on two frame sets:
    `scored` = rows of the present reader's seed 0 score dump marked valid (the frames the heads are read on); `all_frames` = every frame of the clips of that dump.
    Per set: n, median and 90th percentile of the absolute error in azimuth and in elevation, the share of the frames whose box centre lies inside the W16 window of arm S
    (absolute azimuth difference <= 8 degrees and elevation difference <= 7.95 degrees)."""
    z = np.load(os.path.join(present_dump, "reader_s0_score.npz"), allow_pickle=False)
    clips = [str(c) for c in z["clips"]]
    cid, fr, val = np.asarray(z["cid"], np.int64), np.asarray(z["frame"], np.int64), np.asarray(z["valid"], bool)
    steer = SC.SteerStore(s_root)
    acc = {"scored": ([], []), "all_frames": ([], [])}
    for i, c in enumerate(clips):
        az, el, ok = SC.load_box(run1_box_dir, c)
        st = steer.get(c, len(az))
        scored = np.zeros(len(az), bool)
        scored[fr[(cid == i) & val]] = True
        d = st.astype(np.float64) - np.stack([az, el], 1)
        for key, m in (("scored", ok & scored), ("all_frames", ok)):
            acc[key][0].append(d[m, 0])
            acc[key][1].append(d[m, 1])
    out = {}
    for key, (da, de) in acc.items():
        da, de = np.concatenate(da), np.concatenate(de)
        inside = (np.abs(da) <= SC.W16_HALF_AZ_DEG) & (np.abs(de) <= SC.W16_HALF_EL_DEG)
        out[key] = {"n": int(len(da)), "az_abs": _abs_stats(da), "el_abs": _abs_stats(de), "share_box_centre_inside_w16": float(inside.mean()) if len(da) else float("nan")}
    return out


def form_block(root, arm, head, ep_map, boot, stem, ref_ctx):
    d = os.path.join(root, "dump", "%s_%s" % (arm, head))
    ctxs = RC.load_ctxs(d, ep_map)
    P = RC.measure_dumps(ctxs, boot)
    f = {"P": P, "name": RC.form_name(P, stem), "fits": RC.fits_summary(root, arm, head)}
    f["split_row154"] = K6.measure(ctxs, [K6.frame_set(c) for c in ctxs], ep_map, boot)["reader"]
    pr = [RC.parts_raw(D) for D in RC.raw_dumps(d)]
    pooled = {}
    for k in ("m", "s", "f"):
        v = np.concatenate([pa[k][pa["idx"]] for _st, pa in pr])
        pooled[k] = {"rms": float(np.sqrt(np.mean(v ** 2))), "sd": float(v.std())}
    f["parts_adD98_raw_gap"] = {"per_seed": [st for st, _pa in pr], "pooled": pooled}
    return f, ctxs


def print_form(key, f, log):
    P = f["P"]
    for b, ge in P["gap_error"].items():
        if "sd_ci" in ge:
            log("GAPERR %s %-4s sd %.2f [%.2f, %.2f] mean %+.2f n %d per-seed sd %s" % (key, b, ge["sd"], ge["sd_ci"][0], ge["sd_ci"][1], ge["mean"], ge["n"], ["%.2f" % x["sd"] for x in ge["per_seed"]]))
        else:
            log("GAPERR %s %-4s n %d (no interval)" % (key, b, ge["n"]))
    r = RC.rg10(P)
    if r:
        log("RG10 %s ALL r %+.3f [%+.3f, %+.3f] per-seed r %s n %d" % (key, r["r_pooled"], r["r_ci"][0], r["r_ci"][1], ["%+.3f" % x for x in r["r_per_seed"]], r["n_frames"]))
    sp = f["split_row154"]
    log("SPLIT %s const_abs %.2f fast_abs %.2f step_abs %.2f const_prop %.3f fast_prop %.3f" % (key, sp["const_abs"]["value"], sp["fast_abs"]["value"], sp["step_abs"]["value"], sp["const_prop"]["value"], sp["fast_prop"]["value"]))
    pp = f["parts_adD98_raw_gap"]["pooled"]
    log("PARTS-ADD98 %s raw gap rms m %.2f s %.2f f %.2f (sd m %.2f s %.2f f %.2f); present reader as the rule reads it %s (a different object)" % (
        key, pp["m"]["rms"], pp["s"]["rms"], pp["f"]["rms"], pp["m"]["sd"], pp["s"]["sd"], pp["f"]["sd"], RC.PRESENT_RULE_PARTS_M))
    if f.get("cut", {}).get("ci"):
        c = f["cut"]
        log("CUT %s vs present ratio %.3f [%.3f, %.3f] sd %.2f present %.2f n %d" % (key, c["ratio"], c["ci"][0], c["ci"][1], c["sd_pilot"], c["sd_present"], c["n"]))
    log("SEL-LOSS %s %s" % (key, f["fits"]))


def run_names(a, log=print):
    cpath = a.controls or os.path.join(a.root, "read", "STEERREAD_CONTROLS.json")
    rpath = a.out or os.path.join(a.root, "read", "STEERREAD_RESULT.json")
    if not os.path.isfile(cpath):
        log("REFUSED: no controls file; nothing opened")
        return 2
    C = RC.read_json(cpath)
    ok, why = check_controls(C)
    if not ok:
        log("REFUSED: %s; nothing opened" % why)
        log("ZZ READ-STEERREAD names rc 2 controls_all_pass False")
        return 2
    if os.path.exists(rpath):
        raise SystemExit("REFUSED: %s exists" % rpath)
    K2.load_tools(a.closeread_tools)
    ep_map = RC.read_json(a.episode_map)
    ref_ctx = RC.load_ctxs(a.present_dump, ep_map)
    B = a.boot
    res = {"card": CARD, "tier": "T0", "controls_sha256": K2.sha256_file(cpath), "boot": B, "names": {}, "forms": {}}
    P_ref = RC.measure_dumps(ref_ctx, B)
    res["present_reader"] = {"RG10_ALL": RC.rg10(P_ref), "gap_error": P_ref["gap_error"]}
    ctx_of = {}
    for head, stem, cut_stem in (("H5d", "STEERREAD-S-H5d", CUT_STEM), ("H5", "STEERREAD-S-H5", CUT_STEM_TOK)):
        key = "S-" + head
        f, ctxs = form_block(a.root, "S", head, ep_map, B, stem, ref_ctx)
        f["cut"] = RC.cut_ratio(ctxs, ref_ctx, min(B, RC.CUT_B))
        f["cut_name"] = RC.cut_name(f["cut"], cut_stem)
        res["forms"][key] = f
        ctx_of[key] = ctxs
        res["names"][key + "-cut"] = f["cut_name"]
        res["names"][key] = f["name"]
    res["names"]["CARD"] = res["names"]["S-H5d-cut"]
    d_ctx = RC.load_ctxs(os.path.join(a.run1, "dump", "D_H5"), ep_map)
    b_ctx = RC.load_ctxs(os.path.join(a.run1, "dump", "B_H5"), ep_map)
    res["vs_D_H5"] = RC.cut_ratio(ctx_of["S-H5d"], d_ctx, min(B, RC.CUT_B))
    res["vs_B_H5"] = RC.cut_ratio(ctx_of["S-H5d"], b_ctx, min(B, RC.CUT_B))
    res["names"]["vs-fixed"] = vs_fixed_name(res["vs_D_H5"])
    res["names"]["vs-bound"] = vs_bound_name(res["vs_B_H5"])
    res["localiser"] = localiser_stats(a.box_dir, a.root, a.present_dump)
    K2.write_json(rpath, res)
    log("NAME " + res["names"]["CARD"])
    for k in ("S-H5-cut", "S-H5d", "S-H5", "vs-fixed", "vs-bound"):
        log("NAME " + res["names"][k])
    for key, f in res["forms"].items():
        print_form(key, f, log)
    for tag in ("vs_D_H5", "vs_B_H5"):
        c = res[tag]
        if c.get("ci"):
            log("RATIO S-H5d %s ratio %.3f [%.3f, %.3f] sd S %.2f other %.2f n %d" % (tag, c["ratio"], c["ci"][0], c["ci"][1], c["sd_pilot"], c["sd_present"], c["n"]))
    for key, s in res["localiser"].items():
        log("LOCALISER %s n %d az abs median %.2f p90 %.2f el abs median %.2f p90 %.2f share box centre inside W16 %.3f" % (
            key, s["n"], s["az_abs"]["median"], s["az_abs"]["p90"], s["el_abs"]["median"], s["el_abs"]["p90"], s["share_box_centre_inside_w16"]))
    log("ZZ READ-STEERREAD names rc 0 controls_all_pass True")
    return 0


# ============================================================================================ self-test (planted values)
def write_world(root, run1, tmp_box, T=100, n_clips=24):
    """present dumps, reference dumps of D and B, arm S dumps, box and steering files of a synthetic world. Returns (dumps, eps, expected)."""
    dumps, eps = RC.make_world(n_clips=n_clips, T=T)
    RC.write_dumps(os.path.join(root, "present"), dumps)
    K2.write_json(os.path.join(root, "epm.json"), eps)
    for base, tag, scale in ((run1, "D_H5", 0.9), (run1, "B_H5", 0.2), (root, "S_H5d", 0.4), (root, "S_H5", 0.9)):
        RC.write_dumps(os.path.join(base, "dump", tag), {k: RC.scaled_dump(D, scale) for k, D in dumps.items()})
    D0 = dumps[0]
    cid, fr, lp, val = np.asarray(D0["cid"]), np.asarray(D0["frame"]), np.asarray(D0["lab_present"], bool), np.asarray(D0["valid"], bool)
    os.makedirs(tmp_box, exist_ok=True)
    os.makedirs(os.path.join(root, "steer"), exist_ok=True)
    n_in, n_all = 0, 0
    for i, c in enumerate([str(x) for x in D0["clips"]]):
        has = np.zeros(T, bool)
        has[fr[(cid == i) & lp]] = True
        az = np.where(has, 3.0, np.nan)
        el = np.where(has, -1.0, np.nan)
        np.savez(os.path.join(tmp_box, c + ".npz"), has_lead=has, ctr_az=az, ctr_el=el)
        err_az = np.full(T, 9.0 if i == 0 else 1.0)
        st = np.stack([np.where(has, az + err_az, 0.0), np.where(has, el - 0.5, 0.0)], 1).astype(np.float32)
        np.save(os.path.join(root, "steer", c + ".npy"), st)
        sc = np.zeros(T, bool)
        sc[fr[(cid == i) & val]] = True
        n_in += int((has & sc & (np.abs(err_az) <= SC.W16_HALF_AZ_DEG)).sum())
        n_all += int((has & sc).sum())
    return dumps, eps, {"share_scored": n_in / n_all, "n_scored": n_all}


def write_controls(path, passes):
    ctl = {n: {"pass": bool(passes.get(n, True)), "detail": "planted"} for n in CTL_ORDER}
    K2.write_json(path, {"card": CARD, "controls": ctl, "controls_all_pass": all(ctl[n]["pass"] for n in CTL_ORDER)})


def selftest():
    import tempfile
    K2.load_tools(os.environ.get("KINCLOSE_CRTOOLS") or os.environ.get("CROPREAD_CRTOOLS") or next((p for p in (RC.CR_TOOLS, os.path.join(HERE, "..", "..", "closeread")) if os.path.isfile(os.path.join(p, "closeread_fit.py"))), None))
    bad, lines = [], []

    def chk(name, cond):
        lines.append("%s %s" % ("ok  " if cond else "FAIL", name))
        if not cond:
            bad.append(name)
    tmp = os.path.join(HERE, "tmp_selftest")
    os.makedirs(tmp, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp) as base:
        root, run1, box = os.path.join(base, "s"), os.path.join(base, "run1"), os.path.join(base, "box")
        dumps, eps, exp = write_world(root, run1, box)

        def args(ctl_path, out):
            return argparse.Namespace(root=root, run1=run1, box_dir=box, controls=ctl_path, out=out, present_dump=os.path.join(root, "present"), episode_map=os.path.join(root, "epm.json"), closeread_tools=None, boot=150)
        good = os.path.join(base, "c_good.json")
        write_controls(good, {})
        o = []
        rc = run_names(args(good, os.path.join(base, "r_good.json")), log=o.append)
        R = RC.read_json(os.path.join(base, "r_good.json"))
        N = R["names"]
        chk("S-H5d error scaled to 0.4 against the present reader: STEERREAD-CUT-BY-2-REACHED (the card name)", rc == 0 and N["CARD"] == "STEERREAD-CUT-BY-2-REACHED")
        chk("S-H5 scaled to 0.9: STEERREAD-TOKENS-ONLY-CUT-BY-2-NOT-REACHED", N["S-H5-cut"] == "STEERREAD-TOKENS-ONLY-CUT-BY-2-NOT-REACHED")
        chk("per-form names carry the form: " + N["S-H5d"] + ", " + N["S-H5"], N["S-H5d"].startswith("STEERREAD-S-H5d-") and N["S-H5"].startswith("STEERREAD-S-H5-"))
        chk("S-H5d (0.4) against D-H5 (0.9): ratio 0.444, STEERREAD-BEATS-FIXED", abs(R["vs_D_H5"]["ratio"] - 0.4 / 0.9) < 1e-9 and N["vs-fixed"] == "STEERREAD-BEATS-FIXED")
        chk("S-H5d (0.4) against B-H5 (0.2): ratio 2, STEERREAD-BELOW-BOUND", abs(R["vs_B_H5"]["ratio"] - 2.0) < 1e-9 and N["vs-bound"] == "STEERREAD-BELOW-BOUND")
        L = R["localiser"]["scored"]
        chk("localiser error printed: median 1.00 / 0.50 degrees, inside-W16 share %.4f" % exp["share_scored"],
            abs(L["az_abs"]["median"] - 1.0) < 1e-6 and abs(L["el_abs"]["median"] - 0.5) < 1e-6 and L["n"] == exp["n_scored"] and abs(L["share_box_centre_inside_w16"] - exp["share_scored"]) < 1e-12
            and any(x.startswith("LOCALISER scored") for x in o))
        for n in CTL_ORDER:
            bp = os.path.join(base, "c_bad_%s.json" % n)
            write_controls(bp, {n: False})
            ob = []
            rcb = run_names(args(bp, os.path.join(base, "r_bad_%s.json" % n)), log=ob.append)
            chk("control %s failed: names refuses (rc 2), nothing written, nothing printed but the refusal" % n,
                rcb == 2 and not os.path.exists(os.path.join(base, "r_bad_%s.json" % n)) and all(("NAME" not in x and "GAPERR" not in x) for x in ob))
        for cut, name in (({"ci": [0.9, 0.99]}, "STEERREAD-BEATS-FIXED"), ({"ci": [0.5, 0.99]}, "STEERREAD-BEATS-FIXED"), ({"ci": [1.01, 1.4]}, "STEERREAD-WORSE-THAN-FIXED"),
                          ({"ci": [0.8, 1.2]}, "STEERREAD-NOT-SEPARATED-FROM-FIXED"), ({"ci": None}, "STEERREAD-NOT-SEPARATED-FROM-FIXED")):
            chk("vs-fixed rule %s -> %s" % (cut["ci"], name), vs_fixed_name(cut) == name)
        # planted values: upper end exactly 1 is not BEATS; lower end exactly 1 is not BELOW-BOUND
        chk("vs-fixed: upper end 1.0 is not BEATS-FIXED", vs_fixed_name({"ci": [0.7, 1.0]}) == "STEERREAD-NOT-SEPARATED-FROM-FIXED")
        chk("vs-bound: lower end 1.0 is not BELOW-BOUND, 1.01 is", vs_bound_name({"ci": [1.0, 2.0]}) == "STEERREAD-NOT-SEPARATED-FROM-BOUND" and vs_bound_name({"ci": [1.01, 2.0]}) == "STEERREAD-BELOW-BOUND")
        s5 = control_s5(os.path.join(root, "present"), eps, 150)
        chk("s5: control c3 and the planted cuts (0.4 REACHED, 0.9 NOT-REACHED) pass on the world", s5["pass"])
        # s6 form: a shuffled-label dump is not REACHES and its error sd is at least 0.8 of the label sd
        def shuffled(D):
            lg = np.asarray(D["lab_gap"], np.float64)
            pres = np.asarray(D["lab_present"], bool) & np.isfinite(lg)
            g = np.asarray(D["gap"], np.float64).copy()
            g[pres] = lg[pres][np.random.default_rng(1).permutation(int(pres.sum()))]
            return dict(D, gap=g)
        RC.write_dumps(os.path.join(root, "dump", "S_H5d_perm"), {0: shuffled(dumps[0])})
        chk("s6 on a shuffled-label S_H5d dump passes", control_s6(root, eps, 100)["pass"])
        RC.write_dumps(os.path.join(root, "dump", "S_H5d_perm"), {0: RC.planted_dump(dumps[0], 0.0)})
        chk("s6 on a dump that reads the label gap fails", not control_s6(root, eps, 100)["pass"])
    for ln in lines:
        print(ln)
    print("ZZ STEERREAD-SELFTEST rc %d" % (1 if bad else 0))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage", choices=("controls", "names"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=SC.S_ROOT)
    ap.add_argument("--run1", default=SC.RUN1)
    ap.add_argument("--box-dir", default=SC.BOX_DIR)
    ap.add_argument("--present-dump", default=RC.PRESENT_DUMP)
    ap.add_argument("--episode-map", default=RC.EP_MAP)
    ap.add_argument("--closeread-tools", default=RC.CR_TOOLS)
    ap.add_argument("--gaprange-result", default=None)
    ap.add_argument("--s7-pytest-out", default=None)
    ap.add_argument("--controls", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "controls":
        if not a.gaprange_result:
            raise SystemExit("REFUSED: --gaprange-result is required")
        return run_controls(a)
    if a.stage == "names":
        return run_names(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
