#!/usr/bin/env python
"""read_cropread.py -- read of card PREREG_V8F_PROOF_ADD99 (CROPREAD): controls c1..c7 and p1, then the names. Tier T0. Numpy only (no torch).

  python read_cropread.py --stage controls --gaprange-result GAPRANGE_RESULT.json --c6-pytest-out FILE [--root R] [--episode-map M] [--closeread-tools DIR] [--boot 2000]
  python read_cropread.py --stage names [--root R] [--controls R/read/CROPREAD_CONTROLS.json] ...
  python read_cropread.py --selftest

Stage controls assembles c1 to c7 and p1 (writes read/CROPREAD_CONTROLS.json, prints one line per control, opens no value of an arm).
Stage names refuses unless c1..c7 all passed (nothing is opened); with p1 failed it opens arm D only. The estimator is the one of row 152 (kinclose4.measure, kinclose4.r_name) called through the
wrapper `load_ctxs` / `measure_dumps`, which only swaps the dumps. Last line: ZZ READ-CROPREAD <stage> rc <rc> controls_all_pass <bool>.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("CROPREAD_KC"), HERE, os.path.join(HERE, "..", "..", "kinclose"), os.path.join(HERE, "..", "..", "errpart"), os.path.join(HERE, "..", "..", "closeread")):
    if _p and os.path.isdir(_p) and os.path.abspath(_p) not in sys.path:
        sys.path.append(os.path.abspath(_p))
import kinclose2 as K2  # noqa: E402
import kinclose3 as K3  # noqa: E402
import kinclose4 as K4  # noqa: E402
import kinclose6 as K6  # noqa: E402
import make_part_joins as MMJ  # noqa: E402

CARD = "PREREG_V8F_PROOF_ADD99"
ROOT = "/workspace/v8proof/cropread"
PRESENT_DUMP = "/workspace/v8proof/closeread/dump/both"
EP_MAP = "/workspace/v8proof/closeread/episode_map.json"
CR_TOOLS = "/workspace/v8proof/closeread/tools"
KS = (0, 1)
FORMS = (("D", "H5"), ("D", "H1"), ("B", "H5"), ("B", "H1"))
CTL_ORDER = ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "p1")
CTL_MUST = CTL_ORDER[:7]
TOOL_SHA16 = {"kinclose4.py": "9a734658232bc327", "kinclose6.py": "f5be6f8077f6f563"}      # card section 6
GAPRANGE_SHA16 = "9965504cfde0dae4"                                                      # card c2
C2_TOL = 1e-6
C3_SDS = ((0.25, "REACHES"), (8.0, "BELOW"))
C5_SD_RATIO_MIN = 0.8
GAP_FILL_M = 1000.0         # gap written on frames without a label in a planted dump: a step the run rule stops at
CUT_B, CUT_SEED, CUT_RATIO = 2000, 99, 0.5
PRESENT_RULE_PARTS_M = (5.68, 6.31, 1.65)     # card section 6, the present reader as the rule reads it (a different object)
BAD_WORDS = re.compile(r"\b(failed|errors?)\b", re.I)


# ============================================================================================ wrapper around the tool of row 152
def sha16(path):
    return K2.sha256_file(path)[:16]


def load_ctxs(dump_dir, ep_map, ks=KS):
    """the two dumps of a read (fit seeds in place of reader seeds) -> contexts of kinclose4.make_ctx."""
    return [K4.make_ctx(K2.CF.load_dump(os.path.join(dump_dir, "reader_s%d_score.npz" % k)), ep_map) for k in ks]


def measure_dumps(ctxs, B):
    return K4.measure(ctxs, B)


def rg10(P):
    return P["arms"]["ALL"].get("RG10")


def form_name(P, stem):
    """per-form name by the rule of the tool on RG10 over all scored frames: <stem>-REACHES / -BELOW / -NOT-DECIDED."""
    return K4.r_name(rg10(P), stem)


def card_name(P):
    n = K4.r_name(rg10(P), "X")
    return "CROPREAD-PILOT-PASS" if n.endswith("-REACHES") else ("CROPREAD-PILOT-FAIL" if n.endswith("-BELOW") else "CROPREAD-PILOT-NOT-DECIDED")


def cut_name(cut, stem):
    ci = cut.get("ci")
    if not ci or not np.all(np.isfinite(ci)):
        return stem + "-NOT-DECIDED"
    if ci[1] <= CUT_RATIO:
        return stem + "-REACHED"
    if ci[0] > CUT_RATIO:
        return stem + "-NOT-REACHED"
    return stem + "-NOT-DECIDED"


def cut_ratio(cp_list, cr_list, B=CUT_B, seed=CUT_SEED):
    """ratio of the standard deviation of the gap error of the pilot to that of the present reader on the frames both measure, pooled over the two seeds;
    interval: paired bootstrap over evaluation episodes (one episode draw, both seeds and both readers), percentile 2.5 / 97.5."""
    S_p, S_r, eps = [], [], []
    for cp, cr in zip(cp_list, cr_list):
        if cp["clips"] != cr["clips"] or cp["T"] != cr["T"]:
            raise SystemExit("REFUSED: pilot and present dump are not on the same clips")
        scored_r = np.zeros((len(cr["clips"]), cr["T"]), bool)
        scored_r[cr["ci"], cr["fr"]] = True
        both_r = cr["RM"] & cr["LM"]
        sel = K4.meas(cp) & scored_r[cp["ci"], cp["fr"]] & both_r[cp["ci"], cp["fr"]]
        ep_p = K4.gap_err(cp)[sel]
        ep_r = (cr["G"] - cr["LG"])[cp["ci"][sel], cp["fr"][sel]]
        S_p.append(ep_p)
        S_r.append(ep_r)
        eps.append(np.asarray(cp["ep"])[sel])
    alle = sorted(set(np.concatenate(eps).tolist())) if sum(len(e) for e in eps) else []
    out = {"n": int(sum(len(e) for e in S_p)), "n_episodes": int(len(alle))}
    if out["n"] < 2 or len(alle) < 2:
        out.update({"ratio": float("nan"), "ci": None})
        return out
    emap = {e: i for i, e in enumerate(alle)}
    ne = len(alle)

    def sums(e, ep):
        idx = np.asarray([emap[x] for x in ep.tolist()], np.int64)
        return np.stack([np.bincount(idx, minlength=ne).astype(np.float64), np.bincount(idx, e, ne), np.bincount(idx, e * e, ne)])

    Sp = [sums(e, ep) for e, ep in zip(S_p, eps)]
    Sr = [sums(e, ep) for e, ep in zip(S_r, eps)]

    def sd(t):
        return float(np.sqrt(max(t[2] / t[0] - (t[1] / t[0]) ** 2, 0.0))) if t[0] > 0 else float("nan")

    def ratio(w):
        a, b = sd(sum(s @ w for s in Sp)), sd(sum(s @ w for s in Sr))
        return a, b, (a / b if b > 0 else float("nan"))
    a0, b0, r0 = ratio(np.ones(ne))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        d.append(ratio(np.bincount(rng.integers(0, ne, ne), minlength=ne).astype(np.float64))[2])
    d = np.asarray(d)
    d = d[np.isfinite(d)]
    out.update({"ratio": r0, "sd_pilot": a0, "sd_present": b0, "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) else None, "B": int(B), "seed": seed})
    return out


# ============================================================================================ planted dumps (controls c3, selftest)
def planted_dump(D, sd, seed=3):
    """the label gap plus a white error of standard deviation sd (exact over the frames with a label) on the frames with a label; GAP_FILL_M on the others; every other field as in D."""
    lg = np.asarray(D["lab_gap"], np.float64)
    pres = np.asarray(D["lab_present"], bool) & np.isfinite(lg)
    z = np.random.default_rng(seed).standard_normal(len(lg))
    zn = (z[pres] - z[pres].mean()) / z[pres].std()
    gap = np.full(len(lg), GAP_FILL_M)
    gap[pres] = lg[pres] + sd * zn
    return dict(D, gap=gap)


def scaled_dump(D, scale):
    """the present reader's error scaled by `scale` on the frames with a label."""
    lg = np.asarray(D["lab_gap"], np.float64)
    pres = np.asarray(D["lab_present"], bool) & np.isfinite(lg)
    g = np.asarray(D["gap"], np.float64).copy()
    g[pres] = lg[pres] + scale * (g[pres] - lg[pres])
    return dict(D, gap=g)


def raw_dumps(dump_dir, ks=KS):
    return [K2.CF.load_dump(os.path.join(dump_dir, "reader_s%d_score.npz" % k)) for k in ks]


# ============================================================================================ parts of card ADD98 on the raw gap
def parts_raw(D):
    """clip offset m, slow drift s, fast part f of card ADD98 on the dump's raw gap (no map, no minimum), by the routine of make_part_joins.part_arrays over the dump rows (a clip's rows are
    consecutive frames). Mix frames: label gap present and in 0..60 m, valid read, finite raw gap. -> statistics per part and the number of mix frames."""
    lg = np.asarray(D["lab_gap"], np.float64)
    g = np.asarray(D["gap"], np.float64)
    mix = np.asarray(D["lab_present"], bool) & np.isfinite(lg) & (lg >= 0.0) & (lg <= MMJ.GAP_MAX_M) & np.asarray(D["valid"], bool) & np.isfinite(g)
    pa = MMJ.part_arrays(g, mix, np.asarray(D["cid"], np.int64), np.where(mix, lg, 0.0))
    ix = pa["idx"]
    out = {"n_mix_frames": int(len(ix)), "n_runs": int(len(pa["length"]))}
    for k in ("m", "s", "f"):
        v = pa[k][ix]
        out[k] = {"rms": float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan"), "sd": float(v.std()) if len(v) else float("nan"), "mean": float(v.mean()) if len(v) else float("nan")}
    return out, pa


def parts_plain(D, half=MMJ.HALF_WINDOW, jump=MMJ.RUN_JUMP_M):
    """the same parts by plain loops (written for the test of parts_raw against the routine of make_part_joins). -> {dump row: (m, s, f)}"""
    lg = np.asarray(D["lab_gap"], np.float64)
    g = np.asarray(D["gap"], np.float64)
    mix = np.asarray(D["lab_present"], bool) & np.isfinite(lg) & (lg >= 0.0) & (lg <= MMJ.GAP_MAX_M) & np.asarray(D["valid"], bool) & np.isfinite(g)
    cid = np.asarray(D["cid"], np.int64)
    res = {}
    for c in sorted(set(cid[mix].tolist())):
        rows = [i for i in range(len(cid)) if cid[i] == c and mix[i]]
        e = [g[i] - lg[i] for i in rows]
        m = sum(e) / len(e)
        r = [x - m for x in e]
        runs, cur = [], [0]
        for k in range(1, len(rows)):
            if rows[k] == rows[k - 1] + 1 and abs(lg[rows[k]] - lg[rows[k - 1]]) <= jump:
                cur.append(k)
            else:
                runs.append(cur)
                cur = [k]
        runs.append(cur)
        for ks in runs:
            for j, k in enumerate(ks):
                lo, hi = max(j - half, 0), min(j + half, len(ks) - 1)
                s = sum(r[ks[i]] for i in range(lo, hi + 1)) / (hi - lo + 1)
                res[rows[k]] = (m, s, r[k] - s)
    return res


# ============================================================================================ controls
def read_json(p):
    return K2.read_json(p)


def jpass(path):
    """top-level boolean `pass` of a control record written by the data side."""
    if not os.path.isfile(path):
        return {"pass": False, "detail": "missing file " + path}
    try:
        v = read_json(path).get("pass")
    except Exception as e:      # noqa: BLE001
        return {"pass": False, "detail": "unreadable: %s" % e}
    return {"pass": v is True, "detail": "pass=%r in %s sha16 %s" % (v, os.path.basename(path), sha16(path))}


def hex_strings(o):
    if isinstance(o, str):
        return {o} if re.fullmatch(r"[0-9a-f]{64}", o) else set()
    if isinstance(o, dict):
        return set().union(*[hex_strings(v) for v in o.values()]) if o else set()
    if isinstance(o, (list, tuple)):
        return set().union(*[hex_strings(v) for v in o]) if o else set()
    return set()


def expected_fits(p1_pass):
    names = ["D_%s_s%d" % (h, k) for h in ("H5", "H1") for k in KS] + ["D_H5_s0_perm"]
    if p1_pass:
        names += ["B_%s_s%d" % (h, k) for h in ("H5", "H1") for k in KS]
    return names


def control_c2(present_dir, ep_map, gaprange_path, B, check_shas=True):
    """the wrapper on the present reader's dumps equals GAPRANGE_RESULT.json (RG10 ALL: r and both ends; gap error ALL: sd and both ends) within 1e-6."""
    ref = read_json(gaprange_path)
    P = measure_dumps(load_ctxs(present_dir, ep_map), B)
    got = {"r": rg10(P)["r_pooled"], "r_lo": rg10(P)["r_ci"][0], "r_hi": rg10(P)["r_ci"][1], "sd": P["gap_error"]["ALL"]["sd"], "sd_lo": P["gap_error"]["ALL"]["sd_ci"][0], "sd_hi": P["gap_error"]["ALL"]["sd_ci"][1]}
    rr, rg = ref["arms"]["ALL"]["RG10"], ref["gap_error"]["ALL"]
    want = {"r": rr["r_pooled"], "r_lo": rr["r_ci"][0], "r_hi": rr["r_ci"][1], "sd": rg["sd"], "sd_lo": rg["sd_ci"][0], "sd_hi": rg["sd_ci"][1]}
    diff = {k: abs(float(got[k]) - float(want[k])) for k in got}
    ok = bool(all(v <= C2_TOL for v in diff.values()))
    det = {"values_present_reader": got, "values_gaprange_file": want, "max_abs_diff": max(diff.values()), "tol": C2_TOL}
    if check_shas:
        t4, t6 = sha16(K4.__file__), sha16(K6.__file__)
        det["tool_sha16"] = {"kinclose4.py": t4, "kinclose6.py": t6}
        det["gaprange_sha16"] = sha16(gaprange_path)
        ok = bool(ok and t4 == TOOL_SHA16["kinclose4.py"] and t6 == TOOL_SHA16["kinclose6.py"] and det["gaprange_sha16"] == GAPRANGE_SHA16)
    return {"pass": ok, "detail": det}


def control_c3(present_dir, ep_map, B):
    """the label gap plus a white error of 0.25 m gets REACHES, plus 8 m gets BELOW (name rule on RG10, all scored frames)."""
    raws = raw_dumps(present_dir)
    lv, ok = [], True
    for sd, want in C3_SDS:
        ctxs = [K4.make_ctx(planted_dump(D, sd, seed=3 + k), ep_map) for k, D in enumerate(raws)]
        P = measure_dumps(ctxs, B)
        name = form_name(P, "C3")
        lv.append({"planted_sd_m": sd, "name": name, "r": rg10(P)["r_pooled"] if rg10(P) else None, "r_ci": rg10(P)["r_ci"] if rg10(P) else None, "wanted": want})
        ok = ok and name == "C3-" + want
    return {"pass": bool(ok), "detail": {"levels": lv}}


def control_c4(root, p1_pass):
    lp, ep = os.path.join(root, "lists", "train_final.txt"), os.path.join(root, "lists", "eval141.txt")
    bad = []
    if not (os.path.isfile(lp) and os.path.isfile(ep)):
        return {"pass": False, "detail": {"failures": ["missing lists"]}}
    rows = [ln.split() for ln in open(lp, encoding="utf-8").read().splitlines() if ln.strip()]
    train = [r[0] for r in rows]
    ev = [ln.strip() for ln in open(ep, encoding="utf-8") if ln.strip()]
    both = sorted(set(train) & set(ev))
    if both:
        bad.append("%d clip(s) in both lists" % len(both))
    sha_file = K2.sha256_file(lp)
    sha_join = hashlib.sha256("\n".join(train).encode("utf-8")).hexdigest()
    lr_path = os.path.join(root, "lists", "LISTS_RECORD.json")
    in_rec = bool(os.path.isfile(lr_path) and ({sha_file, sha_join} & hex_strings(read_json(lr_path))))
    if not in_rec:
        bad.append("the sha256 of the train list is not in LISTS_RECORD.json")
    recs = {}
    for p in sorted(glob.glob(os.path.join(root, "fits", "*", "FIT_RECORD.json"))):
        recs[os.path.basename(os.path.dirname(p))] = read_json(p)
    miss = [n for n in expected_fits(p1_pass) if n not in recs]
    if miss:
        bad.append("fit records missing: %s" % miss)
    ts = set(train)
    for n, r in recs.items():
        read = list(r["clips_read"]["fit"]) + list(r["clips_read"]["sel"])
        if not set(read) <= ts:
            bad.append("%s read a clip outside the train list" % n)
        if set(read) & set(ev):
            bad.append("%s read an evaluation clip" % n)
        if r.get("train_list_sha256") != sha_file:
            bad.append("%s: train list sha256 differs from the file" % n)
        if r.get("eval_list_sha256") != K2.sha256_file(ep):
            bad.append("%s: evaluation list sha256 differs from the file" % n)
    return {"pass": not bad, "detail": {"failures": bad, "n_fit_records": len(recs), "train_list_sha256": sha_file, "train_list_in_lists_record": in_rec,
                                      "hparams_equal_card_per_fit": {n: r.get("hparams_equal_card") for n, r in recs.items()}}}


def control_c5(root, ep_map, B):
    """the permuted-label fit of D-H5 (seed 0) does not get REACHES and its gap error has a standard deviation of at least 0.8 times that of the label gap on the same frames."""
    d = os.path.join(root, "dump", "D_H5_perm")
    if not os.path.isfile(os.path.join(d, "reader_s0_score.npz")):
        return {"pass": False, "detail": {"failures": ["no dump " + d]}}
    c = K4.make_ctx(K2.CF.load_dump(os.path.join(d, "reader_s0_score.npz")), ep_map)
    P = measure_dumps([c], B)
    m = K4.meas(c)
    sd_lab = float(c["LG"][c["ci"], c["fr"]][m].std())
    sd_err = float(P["gap_error"]["ALL"]["sd"])
    name = form_name(P, "C5")
    ok = bool(name != "C5-REACHES" and sd_err >= C5_SD_RATIO_MIN * sd_lab)
    return {"pass": ok, "detail": {"name": name, "gap_error_sd_m": sd_err, "label_gap_sd_m": sd_lab, "ratio": sd_err / sd_lab if sd_lab > 0 else None, "min_ratio": C5_SD_RATIO_MIN}}


def control_c6(path):
    if not path or not os.path.isfile(path):
        return {"pass": False, "detail": "missing pytest output " + str(path)}
    t = open(path, encoding="utf-8", errors="replace").read()
    ok = bool(re.search(r"\b\d+ passed\b", t) and not BAD_WORDS.search(t))
    return {"pass": ok, "detail": "pytest output sha16 %s; has 'passed' and no 'failed' or 'error': %s" % (sha16(path), ok)}


def run_controls(a, log=print):
    out = a.out or os.path.join(a.root, "read", "CROPREAD_CONTROLS.json")
    tm = bool(getattr(a, "test_mode", False))
    if a.boot != 2000 and not tm:
        raise SystemExit("REFUSED: c2 compares with a bootstrap of 2,000 draws; --boot must be 2000")
    if os.path.exists(out):
        raise SystemExit("REFUSED: %s exists" % out)
    K2.load_tools(a.closeread_tools)
    ep_map = read_json(a.episode_map)
    ctl = {n: jpass(os.path.join(a.root, "ctl", n.upper() + ".json")) for n in ("c1", "c7", "p1")}
    ctl["c2"] = control_c2(a.present_dump, ep_map, a.gaprange_result, a.boot, check_shas=not tm)
    ctl["c3"] = control_c3(a.present_dump, ep_map, a.boot)
    ctl["c4"] = control_c4(a.root, bool(ctl["p1"]["pass"]))
    ctl["c5"] = control_c5(a.root, ep_map, a.boot)
    ctl["c6"] = control_c6(a.c6_pytest_out)
    ctl = {n: ctl[n] for n in CTL_ORDER}
    allp = bool(all(ctl[n]["pass"] for n in CTL_MUST))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    K2.write_json(out, {"card": CARD, "controls": ctl, "controls_all_pass": allp, "p1_pass": bool(ctl["p1"]["pass"]), "boot": a.boot})
    for n in CTL_ORDER:
        log("%s %s" % (n, "PASS" if ctl[n]["pass"] else "FAIL"))
    log("ZZ READ-CROPREAD controls rc %d controls_all_pass %s p1 %s" % (0 if allp else 1, allp, "PASS" if ctl["p1"]["pass"] else "FAIL"))
    return 0 if allp else 1


# ============================================================================================ names
def check_controls(C):
    """-> (ok, why): c1..c7 all present with pass True and the file says controls_all_pass."""
    ctl = C.get("controls", {})
    bad = [n for n in CTL_MUST if not (isinstance(ctl.get(n), dict) and ctl[n].get("pass") is True)]
    if "p1" not in ctl:
        bad.append("p1(missing)")
    if bad or C.get("controls_all_pass") is not True:
        return False, "controls not passed: %s" % (bad or "file says controls_all_pass false")
    return True, ""


def fits_summary(root, arm, head):
    out = {}
    for k in KS:
        p = os.path.join(root, "fits", "%s_%s_s%d" % (arm, head, k), "FIT_RECORD.json")
        if os.path.isfile(p):
            r = read_json(p)
            out["s%d" % k] = {"kept_epoch": r["kept_epoch"], "kept_sel_loss": r["kept_sel_loss"], "n_epochs": len(r["history"])}
    return out


def run_names(a, log=print):
    cpath = a.controls or os.path.join(a.root, "read", "CROPREAD_CONTROLS.json")
    rpath = a.out or os.path.join(a.root, "read", "CROPREAD_RESULT.json")
    if not os.path.isfile(cpath):
        log("REFUSED: no controls file; nothing opened")
        return 2
    C = read_json(cpath)
    ok, why = check_controls(C)
    if not ok:
        log("REFUSED: %s; nothing opened" % why)
        log("ZZ READ-CROPREAD names rc 2 controls_all_pass False")
        return 2
    if os.path.exists(rpath):
        raise SystemExit("REFUSED: %s exists" % rpath)
    p1 = bool(C["controls"]["p1"]["pass"])
    K2.load_tools(a.closeread_tools)
    ep_map = read_json(a.episode_map)
    ref_ctx = load_ctxs(a.present_dump, ep_map)
    P_ref = measure_dumps(ref_ctx, a.boot)
    forms = [f for f in FORMS if f[0] == "D" or p1]
    res = {"card": CARD, "tier": "T0", "controls_sha256": K2.sha256_file(cpath), "boot": a.boot, "p1_pass": p1, "names": {}, "forms": {}, "present_reader": {"RG10_ALL": rg10(P_ref), "gap_error": P_ref["gap_error"]}}
    if not p1:
        res["arm_B"] = {"opened": False, "reason": "p1 failed: arm B stays unopened and is reported as failed with no value"}
        log("p1 failed: arm B stays unopened (failed, no value)")
    ctxs_of = {}
    for arm, head in forms:
        key = "%s-%s" % (arm, head)
        ctxs = load_ctxs(os.path.join(a.root, "dump", "%s_%s" % (arm, head)), ep_map)
        ctxs_of[key] = ctxs
        P = measure_dumps(ctxs, a.boot)
        f = {"P": P, "name": form_name(P, "CROPREAD-" + key), "fits": fits_summary(a.root, arm, head)}
        Ms = [K6.frame_set(c) for c in ctxs]
        f["split_row154"] = K6.measure(ctxs, Ms, ep_map, a.boot)["reader"]
        pr = [parts_raw(D) for D in raw_dumps(os.path.join(a.root, "dump", "%s_%s" % (arm, head)))]
        pooled = {}
        for k in ("m", "s", "f"):
            v = np.concatenate([pa[k][pa["idx"]] for _st, pa in pr])
            pooled[k] = {"rms": float(np.sqrt(np.mean(v ** 2))), "sd": float(v.std())}
        f["parts_adD98_raw_gap"] = {"per_seed": [st for st, _pa in pr], "pooled": pooled}
        res["forms"][key] = f
        res["names"][key] = f["name"]
        if head == "H5":
            cut = cut_ratio(ctxs, ref_ctx, min(a.boot, CUT_B))
            stem = "CROPREAD-CUT-BY-2" if arm == "D" else "CROPREAD-BOUND-CUT-BY-2"
            f["cut"] = cut
            res["names"][stem if arm == "D" else "BOUND"] = cut_name(cut, stem)
    res["names"]["CARD"] = card_name(res["forms"]["D-H5"]["P"])
    K2.write_json(rpath, res)
    # prints: names first, then the values with no name
    log("NAME " + res["names"]["CARD"])
    for k, v in res["names"].items():
        if k != "CARD":
            log("NAME " + v)
    for key, f in res["forms"].items():
        P = f["P"]
        for b, ge in P["gap_error"].items():
            if "sd_ci" in ge:
                log("GAPERR %s %-4s sd %.2f [%.2f, %.2f] mean %+.2f n %d per-seed sd %s" % (key, b, ge["sd"], ge["sd_ci"][0], ge["sd_ci"][1], ge["mean"], ge["n"], ["%.2f" % x["sd"] for x in ge["per_seed"]]))
            else:
                log("GAPERR %s %-4s n %d (no interval)" % (key, b, ge["n"]))
        r = rg10(P)
        if r:
            log("RG10 %s ALL r %+.3f [%+.3f, %+.3f] per-seed r %s n %d" % (key, r["r_pooled"], r["r_ci"][0], r["r_ci"][1], ["%+.3f" % x for x in r["r_per_seed"]], r["n_frames"]))
        sp = f["split_row154"]
        log("SPLIT %s const_abs %.2f fast_abs %.2f step_abs %.2f const_prop %.3f fast_prop %.3f" % (key, sp["const_abs"]["value"], sp["fast_abs"]["value"], sp["step_abs"]["value"], sp["const_prop"]["value"], sp["fast_prop"]["value"]))
        pp = f["parts_adD98_raw_gap"]["pooled"]
        log("PARTS-ADD98 %s raw gap rms m %.2f s %.2f f %.2f (sd m %.2f s %.2f f %.2f); present reader as the rule reads it %s (a different object)" % (
            key, pp["m"]["rms"], pp["s"]["rms"], pp["f"]["rms"], pp["m"]["sd"], pp["s"]["sd"], pp["f"]["sd"], PRESENT_RULE_PARTS_M))
        if "cut" in f and f["cut"].get("ci"):
            log("CUT %s ratio %.3f [%.3f, %.3f] sd pilot %.2f present %.2f n %d" % (key, f["cut"]["ratio"], f["cut"]["ci"][0], f["cut"]["ci"][1], f["cut"]["sd_pilot"], f["cut"]["sd_present"], f["cut"]["n"]))
        log("SEL-LOSS %s %s" % (key, f["fits"]))
    log("ZZ READ-CROPREAD names rc 0 controls_all_pass True")
    return 0


# ============================================================================================ self-test (planted faults)
def make_world(seed=5, n_clips=24, T=100, dt=0.2):
    """present-reader-like dumps of two seeds: label gap with piecewise constant closing speed, steps, holes; the reader gap is the label plus white noise of 1.5 m."""
    rng = np.random.default_rng(seed)
    clips = ["wclip-%03d" % i for i in range(n_clips)]
    cols = {n: [] for n in ("cid", "clip_index", "frame", "p", "gap", "clos", "valid", "lab_present", "lab_gap", "lab_closing", "lab_run")}
    for i in range(n_clips):
        v = np.repeat(rng.normal(0.0, 2.0, T // 50 + 1), 50)[:T]
        g = np.clip(35.0 + np.cumsum(v * dt) + rng.normal(0, 0.02, T), 3.0, 58.0)
        ok = np.ones(T, bool)
        ok[: 6 + i % 5] = False
        clo, run = np.zeros(T), np.zeros(T, np.int64)
        for j in range(T):
            c = K3.port_label_closing(g, ok, j, dt)
            if c is None:
                continue
            idx, f = 1, j - 1
            while f >= 0 and idx < 5 and ok[f] and abs(g[f + 1] - g[f]) <= 3.0:
                idx, f = idx + 1, f - 1
            run[j], clo[j] = idx, c
        for k, vv in (("cid", np.full(T, i)), ("clip_index", np.full(T, 1000 + i)), ("frame", np.arange(T)), ("p", np.full(T, 0.9)), ("gap", g + rng.normal(0, 1.5, T)), ("clos", rng.normal(0, 1.0, T)),
                      ("valid", np.arange(T) >= 4), ("lab_present", ok), ("lab_gap", np.where(ok, g, np.nan)), ("lab_closing", clo), ("lab_run", run)):
            cols[k].append(vv)
    D = {k: np.concatenate(v) for k, v in cols.items()}
    D["clips"], D["dt_s"], D["seed"] = np.asarray(clips), np.float64(dt), np.int64(0)
    eps = {c: "ep%03d" % (i // 2) for i, c in enumerate(clips)}
    return {0: D, 1: dict(D, seed=np.int64(1), gap=np.asarray(D["gap"]) + np.random.default_rng(seed + 1).normal(0, 0.3, len(D["gap"])))}, eps


def write_dumps(d, dumps):
    os.makedirs(d, exist_ok=True)
    for k, D in dumps.items():
        np.savez(os.path.join(d, "reader_s%d_score.npz" % k), **D)


def write_controls(path, passes, p1=True):
    ctl = {n: {"pass": bool(passes.get(n, True)), "detail": "planted"} for n in CTL_MUST}
    ctl["p1"] = {"pass": bool(p1), "detail": "planted"}
    K2.write_json(path, {"card": CARD, "controls": ctl, "controls_all_pass": all(ctl[n]["pass"] for n in CTL_MUST)})


def selftest():
    import tempfile
    K2.load_tools(os.environ.get("CROPREAD_CRTOOLS") or next((p for p in (CR_TOOLS, os.path.join(HERE, "..", "..", "closeread")) if os.path.isfile(os.path.join(p, "closeread_fit.py"))), None))
    bad, lines = [], []

    def chk(name, cond):
        lines.append("%s %s" % ("ok  " if cond else "FAIL", name))
        if not cond:
            bad.append(name)
    dumps, eps = make_world()
    tmp = os.path.join(HERE, "tmp_selftest")
    os.makedirs(tmp, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp) as root:
        write_dumps(os.path.join(root, "present"), dumps)
        K2.write_json(os.path.join(root, "epm.json"), eps)

        def forms_dir(tag, make):
            write_dumps(os.path.join(root, "dump", tag), {k: make(D) for k, D in dumps.items()})

        def args(ctl_path, **kw):
            return argparse.Namespace(root=root, controls=ctl_path, out=kw.get("out"), present_dump=os.path.join(root, "present"), episode_map=os.path.join(root, "epm.json"),
                                      closeread_tools=None, boot=150)

        label = lambda D: planted_dump(D, 0.0)                    # noqa: E731
        forms_dir("D_H5", label)
        forms_dir("D_H1", lambda D: scaled_dump(D, 0.4))
        forms_dir("B_H5", lambda D: scaled_dump(D, 0.9))
        forms_dir("B_H1", lambda D: scaled_dump(D, 0.4))
        out = []
        lg = lambda s: out.append(s)                             # noqa: E731
        good = os.path.join(root, "c_good.json")
        write_controls(good, {})
        rc = run_names(args(good, out=os.path.join(root, "r_good.json")), log=lg)
        R = read_json(os.path.join(root, "r_good.json"))
        chk("a dump equal to the label gap gives the card name PASS", rc == 0 and R["names"]["CARD"] == "CROPREAD-PILOT-PASS")
        chk("per-form names: D-H5 (the label gap) REACHES, D-H1 carries a per-form name",
            R["names"]["D-H5"] == "CROPREAD-D-H5-REACHES" and R["names"]["D-H1"].startswith("CROPREAD-D-H1-"))
        cp = [K4.make_ctx(scaled_dump(D, 0.4), eps) for D in dumps.values()]
        cr = [K4.make_ctx(D, eps) for D in dumps.values()]
        c04 = cut_ratio(cp, cr, B=200)
        chk("cut ratio of an error scaled to 0.4 is 0.4 and the name is REACHED (ratio %.4f)" % c04["ratio"], abs(c04["ratio"] - 0.4) < 1e-9 and cut_name(c04, "S") == "S-REACHED")
        cp = [K4.make_ctx(scaled_dump(D, 0.9), eps) for D in dumps.values()]
        c09 = cut_ratio(cp, cr, B=200)
        chk("cut ratio of an error scaled to 0.9 is 0.9 and the name is NOT-REACHED (ratio %.4f)" % c09["ratio"], abs(c09["ratio"] - 0.9) < 1e-9 and cut_name(c09, "S") == "S-NOT-REACHED")
        chk("D-H5 pilot of the label gap: cut name REACHED (error sd 0 against the present reader)", R["names"]["CROPREAD-CUT-BY-2"].endswith("-REACHED"))
        chk("arm B opened when p1 passed", "B-H5" in R["forms"] and R["names"]["BOUND"].startswith("CROPREAD-BOUND-CUT-BY-2-"))
        # a failed control blocks names
        for n in CTL_MUST:
            bp = os.path.join(root, "c_bad_%s.json" % n)
            write_controls(bp, {n: False})
            o = []
            rcb = run_names(args(bp, out=os.path.join(root, "r_bad_%s.json" % n)), log=o.append)
            chk("control %s failed: names refuses (rc 2), nothing written, nothing printed but the refusal" % n, rcb == 2 and not os.path.exists(os.path.join(root, "r_bad_%s.json" % n)) and all(("NAME" not in x and "GAPERR" not in x) for x in o))
        # p1 failed leaves B unopened
        pp = os.path.join(root, "c_p1.json")
        write_controls(pp, {}, p1=False)
        o = []
        rcp = run_names(args(pp, out=os.path.join(root, "r_p1.json")), log=o.append)
        Rp = read_json(os.path.join(root, "r_p1.json"))
        chk("p1 failed: arm D opened, arm B unopened with no value", rcp == 0 and "D-H5" in Rp["forms"] and not any(k.startswith("B-") for k in Rp["forms"]) and Rp["arm_B"]["opened"] is False
            and any("arm B stays unopened" in x for x in o) and "BOUND" not in Rp["names"])
        # planted c3 levels on the world
        for sd, want in C3_SDS:
            P = measure_dumps([K4.make_ctx(planted_dump(D, sd, seed=3 + k), eps) for k, D in enumerate(dumps.values())], 150)
            chk("c3: label gap plus white error %.2f m gets %s (name %s)" % (sd, want, form_name(P, "C3")), form_name(P, "C3") == "C3-" + want)
        # a constant estimate gives NOT-DECIDED, never PASS
        P = measure_dumps([K4.make_ctx(dict(D, gap=np.full(len(D["gap"]), 25.0)), eps) for D in dumps.values()], 100)
        chk("a constant gap reads NOT-DECIDED (r undefined), not REACHES", form_name(P, "K") == "K-NOT-DECIDED")
        # c5-like: a permuted-label style estimate (label shuffled across frames) is not REACHES and its error sd is at least 0.8 of the label sd
        def shuffled(D):
            lg = np.asarray(D["lab_gap"], np.float64)
            pres = np.asarray(D["lab_present"], bool) & np.isfinite(lg)
            g = np.asarray(D["gap"], np.float64).copy()
            g[pres] = lg[pres][np.random.default_rng(1).permutation(int(pres.sum()))]
            return dict(D, gap=g)
        c = K4.make_ctx(shuffled(dumps[0]), eps)
        P = measure_dumps([c], 100)
        m = K4.meas(c)
        chk("c5 form: shuffled labels do not get REACHES and the error sd is at least 0.8 of the label sd", form_name(P, "C5") != "C5-REACHES" and P["gap_error"]["ALL"]["sd"] >= 0.8 * float(c["LG"][c["ci"], c["fr"]][m].std()))
    for ln in lines:
        print(ln)
    print("ZZ CROPREAD-SELFTEST rc %d" % (1 if bad else 0))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage", choices=("controls", "names"))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--present-dump", default=PRESENT_DUMP)
    ap.add_argument("--episode-map", default=EP_MAP)
    ap.add_argument("--closeread-tools", default=CR_TOOLS)
    ap.add_argument("--gaprange-result", default=None)
    ap.add_argument("--c6-pytest-out", default=None)
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
