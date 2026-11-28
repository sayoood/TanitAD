#!/usr/bin/env python
"""preshead_t0.py -- card PREREG_V8F_PROOF_ADD118 (PRESHEAD-T0). Tier T0: a presence head on the narrow-window tokens against the label on the evaluation frames of card
ADD116 (a diagnostic of the head, not driving performance). CPU only; reads the dumps and records of the fit queue, banked rule logs with their side files and the
evaluation join; runs no fit and no rollout. The reader of card ADD116, preserr_t0.py (sha256 begins fae2c532698cf0ba), is imported unchanged for the frames, the labels,
the counts per clip and the rates.

Forms (a frame is marked when the value is at least 0.5): READER = the token reader's p; HEAD = the new head's p; HEAD-OR-READER = the larger of the two.
Head seed m is read on the frames of model seed m. Pooled = mean of the two seeds. Intervals: bootstrap over clips, the same draw for both seeds and all forms.

  python preshead_t0.py --stage controls --preserr FILE --t1 DIR --join FILE --root DIR --ref-fit-record FILE --tests-fit FILE --tests-read FILE --out CONTROLS.json
  python preshead_t0.py --stage names --controls CONTROLS.json --out RESULT.json            refuses unless p0 to p6 all pass
The controls stage prints p<i> PASS|FAIL and one summary line; no value.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD118"
TIER = "T0 (a presence head's reads on recorded evaluation frames against the label; a diagnostic of the head, not driving performance)"
PRE = "PRESHEAD-T0-"
PRESERR_SHA16 = "fae2c532698cf0ba"
SOURCE_SHA16 = "61bf2f4e8e7ec61f"
CONTROLS = ("p0", "p1", "p2", "p3", "p4", "p5", "p6")
FITS = (("D_H5_s0", 0, False), ("D_H5_s1", 1, False), ("D_H5_s0_perm", 0, True))
HPARAMS = {"lr": 1e-3, "weight_decay": 0.05, "batch": 256, "epochs": 8, "warmup_epochs": 1, "beta_m": 1.0, "token_drop": 0.1, "d_model": 256, "n_heads": 4}
EXPECT = {"M": 0.1558, "X": 0.2244, "n_frames": 10865, "n_L": 3383, "n_not_L": 7482}            # row 180, control p4
ID_TOL = 1e-4
BAR_M, BAR_X = 0.0779, 0.4488                                                                  # row 180: half of 0.1558, twice 0.2244
AUC_PERM = (0.40, 0.60)
FORMS = ("READER", "HEAD", "HEAD-OR-READER")
NAMED = ("HEAD", "HEAD-OR-READER")


def sha_file(p):
    h = hashlib.sha256()
    with io.open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(p):
    with io.open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load_preserr(path):
    spec = importlib.util.spec_from_file_location("preserr_t0_of_card_add116", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows_of(JL, F):
    off = np.asarray(JL["off"]).astype(np.int64)
    return off[F["ci"]] + F["j"]


def with_p(F, p):
    G = dict(F)
    G["p"] = np.asarray(p, np.float64)
    return G


def tests_ok(path, before=None):
    bad = []
    tt = io.open(path, encoding="utf-8", errors="replace").read() if path and os.path.isfile(path) else ""
    hits = re.findall(r"(\d+) passed", tt)
    if not (hits and int(hits[-1]) > 0):
        bad.append("%s has no count of passed tests above 0" % os.path.basename(path or "(none)"))
    if re.search(r"\b\d+ (failed|errors?)\b", tt):
        bad.append("%s has a failed or error count" % os.path.basename(path))
    if before is not None and tt and os.path.getmtime(path) > before:
        bad.append("%s was written after the step it gates" % os.path.basename(path))
    return bad


def read_values(PE, per, perm, n_clip, n_boot, seed=20261122):
    """per: list over seeds of (F reader frames, L, g, p_head [frames]); perm: list over seeds of p of the permuted fit -> the payload of the card"""
    i5 = PE.GRID.index(0.5)
    fs = {"READER": [F for F, _, _, _ in per], "HEAD": [with_p(F, ph) for F, _, _, ph in per], "HEAD-OR-READER": [with_p(F, np.maximum(F["p"], ph)) for F, _, _, ph in per],
          "PERMUTED": [with_p(F, pp) for (F, _, _, _), pp in zip(per, perm)]}
    stats = {k: [PE.clip_stats(Fk, L, g, n_clip) for Fk, (_, L, g, _) in zip(v, per)] for k, v in fs.items()}
    has = np.zeros(n_clip, bool)
    for s in stats["READER"]:
        has |= (s["nL"] + s["nN"]) > 0
    idx = np.flatnonzero(has)
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx)) for _ in range(n_boot)]).astype(np.float64)
    pt, bs = {}, {}
    for k, st in stats.items():
        sub = [{q: v[..., idx] for q, v in s.items()} for s in st]
        pt[k], bs[k] = PE.measures(sub, np.ones((1, len(idx)))), PE.measures(sub, w)
    num = lambda x: float(x) if np.isfinite(x) else None
    out = {"n_clips_with_frames": int(len(idx)), "n_boot": int(n_boot), "bar": {"M": BAR_M, "X": BAR_X}, "forms": {}}
    for k in ("READER", "HEAD", "HEAD-OR-READER", "PERMUTED"):
        st = stats[k]
        out["forms"][k] = {
            "pooled": {"M": {"value": num(pt[k]["M"][i5, 0]), "interval": PE._iv(bs[k]["M"][i5])}, "X": {"value": num(pt[k]["X"][i5, 0]), "interval": PE._iv(bs[k]["X"][i5])},
                       "M_large_gaps": num(pt[k]["M_large"][0]), "M_small_gaps": num(pt[k]["M_small"][0]),
                       "by_t": [{"t": t, "M": num(pt[k]["M"][q, 0]), "X": num(pt[k]["X"][q, 0])} for q, t in enumerate(PE.GRID)]},
            "per_seed": [{"n_frames": int(len(Fk["p"])), "n_L": int(L.sum()), "n_not_L": int((~L).sum()), "M": num(PE._div(s["miss"][i5].sum(), s["nL"].sum())),
                          "X": num(PE._div(s["mark"][i5].sum(), s["nN"].sum())), "auc": PE.auc(Fk["p"], L),
                          "bins": [{"lo": lo, "hi": hi, "n_L": int(s["nLb"][b].sum()), "n_miss": int(s["missb"][b].sum())} for b, (lo, hi) in enumerate(PE.BINS)]}
                         for Fk, (_, L, g, _), s in zip(fs[k], per, st)]}
    out["differences"] = {}
    for k in NAMED:
        out["differences"][k] = {q: {"value": num(pt[k][q][i5, 0] - pt["READER"][q][i5, 0]), "interval": PE._iv(bs[k][q][i5] - bs["READER"][q][i5])} for q in ("M", "X")}
    xr = pt["READER"]["X"][i5, 0]
    q = int(np.nanargmin(np.abs(pt["HEAD"]["X"][:, 0] - xr)))
    out["head_at_the_reader_extra_mark_rate"] = {"t": PE.GRID[q], "M": num(pt["HEAD"]["M"][q, 0]), "X": num(pt["HEAD"]["X"][q, 0]), "reader_M": num(pt["READER"]["M"][i5, 0]), "reader_X": num(xr)}
    return out


def names_of(payload):
    """card section 4"""
    out, meets, cuts = {}, [], []
    for k in NAMED:
        P, D = payload["forms"][k]["pooled"], payload["differences"][k]
        m, x, iv = P["M"]["value"], P["X"]["value"], D["M"]["interval"]
        ok_x = x is not None and x <= BAR_X
        if m is not None and m <= BAR_M and ok_x:
            out[k] = PRE + k + "-MEETS-BAR"
            meets.append((x, k))
        elif iv is not None and iv[1] < 0.0 and ok_x:
            out[k] = PRE + k + "-CUTS-MISSES-NOT-TO-BAR"
            cuts.append((m, k))
        else:
            out[k] = PRE + k + "-NO-CUT-OF-MISSES"
    if meets:
        out["OVERALL"], out["named_form"] = PRE + "BAR-MET", sorted(meets)[0][1]                # of two the lower pooled X
    elif cuts:
        out["OVERALL"], out["named_form"] = PRE + "MISSES-CUT-BAR-NOT-MET", sorted(cuts)[0][1]  # the lower miss rate goes to the T1-kin card
    else:
        out["OVERALL"], out["named_form"] = PRE + "NO-CUT", None
    return out


def run_controls(a, now=None):
    PE = load_preserr(a.preserr)
    c, inputs = {}, {"preserr_sha256": sha_file(a.preserr), "tool_sha256": sha_file(os.path.abspath(__file__))}
    kit = os.path.join(a.root, "kit_fit")
    bad0 = []
    if not inputs["preserr_sha256"].startswith(PRESERR_SHA16):
        bad0.append("the reader of card ADD116 has the hash %s, the card names %s" % (inputs["preserr_sha256"][:16], PRESERR_SHA16))
    try:
        g = jl(os.path.join(kit, "PRES_FIT_GEN_RECORD.json"))
        vsha = sha_file(os.path.join(kit, "pres_fit.py"))
        if g.get("pass_p0") is not True or g.get("every_substitution_once") is not True:
            bad0.append("the generator record does not read pass_p0 and every substitution once")
        if g.get("n_substitutions") != 9 or g.get("n_source_lines_changed") != 9 or len(g.get("substitutions") or []) != 9:
            bad0.append("the generator record does not list 9 substitutions on 9 source lines")
        if not str(g.get("source_sha256", "")).startswith(SOURCE_SHA16):
            bad0.append("the generator record names another source hash")
        if g.get("variant_sha256") != vsha:
            bad0.append("the fit tool in the kit does not have the hash of the generator record")
    except (OSError, ValueError) as ex:
        vsha = None
        bad0.append("generator record or fit tool cannot be read: %s" % type(ex).__name__)
    c["p0"] = {"pass": not bad0, "failures": bad0}
    bad1, bad2, recs = [], [], {}
    try:
        ref = jl(a.ref_fit_record)
    except (OSError, ValueError) as ex:
        ref = {}
        bad1.append("the fit record of card ADD106 cannot be read: %s" % type(ex).__name__)
    for name, seed, perm in FITS:
        try:
            r = jl(os.path.join(a.root, "fits", name, "FIT_RECORD.json"))
        except (OSError, ValueError) as ex:
            bad2.append("%s: no fit record (%s)" % (name, type(ex).__name__))
            continue
        recs[name] = r
        for k in ("train_list_sha256", "eval_list_sha256"):
            if not r.get(k) or r.get(k) != ref.get(k):
                bad1.append("%s: %s is not that of the fit record of card ADD106" % (name, k))
        ns, npos = r.get("n_samples") or {}, r.get("n_positive") or {}
        if r.get("tool_sha256") != vsha:
            bad2.append("%s: the fit ran with another tool hash" % name)
        if r.get("hparams") != HPARAMS or r.get("hparams_equal_card") is not True:
            bad2.append("%s: hyper-parameters are not those of the card" % name)
        if r.get("label") != "presence" or r.get("card") != CARD or r.get("seed") != seed or bool(r.get("permute_labels")) != perm or r.get("arm") != "D" or r.get("head") != "H5":
            bad2.append("%s: label, card, seed, arm, head or the permute switch is not as the card states" % name)
        for role in ("fit", "sel"):
            if not (isinstance(ns.get(role), int) and isinstance(npos.get(role), int) and 0 < npos[role] < ns[role]):
                bad2.append("%s: the %s samples do not hold both classes" % (name, role))
        ks = r.get("kept_sel_loss")
        if not (isinstance(ks, (int, float)) and math.isfinite(ks)):
            bad2.append("%s: the kept loss is not finite" % name)
    c["p1"] = {"pass": not bad1 and len(recs) == len(FITS), "failures": bad1 + ([] if len(recs) == len(FITS) else ["a fit record is absent"])}
    c["p2"] = {"pass": not bad2, "failures": bad2}
    JL = PE.npz(a.join)
    n_clip, n_rows = len(np.asarray(JL["n"])), int(np.asarray(JL["n"]).sum())
    inputs["join_sha256"] = sha_file(a.join)
    frames, bad4, files = [], [], []
    for m in PE.SEEDS:
        rows, b1, fl = PE.load_seed(a.t1, m)
        files += fl
        bad4 += ["seed %d: %s" % (m, x) for x in b1]
        F = PE.frames_of(rows)
        maps, L, g = PE.label_of(JL, F["ci"], F["j"])
        if int((~maps).sum()):
            bad4.append("seed %d: %d frames map to no join row" % (m, int((~maps).sum())))
        frames.append((F, L, g, rows_of(JL, F) if not int((~maps).sum()) else None))
    bad5, dumps = [], {}
    for name, seed, perm in FITS:
        d = os.path.join(a.root, "dump", name)
        try:
            with np.load(os.path.join(d, "pres.npz"), allow_pickle=False) as z:
                p = np.asarray(z["p"], np.float64)
            meta = jl(os.path.join(d, "pres_meta.json"))
        except (OSError, ValueError, KeyError) as ex:
            bad5.append("%s: no dump (%s)" % (name, type(ex).__name__))
            continue
        files.append(os.path.join(d, "pres.npz"))
        if len(p) != n_rows or meta.get("n_rows") != n_rows:
            bad5.append("%s: %d rows, the join has %d" % (name, len(p), n_rows))
            continue
        if not meta.get("head_pt_sha256") or meta.get("head_pt_sha256") != (recs.get(name) or {}).get("head_pt_sha256"):
            bad5.append("%s: the dump does not name the head hash of its fit record" % name)
        if meta.get("eval_join_sha256") != inputs["join_sha256"]:
            bad5.append("%s: the dump was made on another join" % name)
        for m in (PE.SEEDS if perm else (seed,)):
            F, L, g, row = frames[m]
            if row is not None and int((~np.isfinite(p[row])).sum()):
                bad5.append("%s: %d frames of the read of seed %d have no finite p" % (name, int((~np.isfinite(p[row])).sum()), m))
        dumps[name] = p
    c["p5"] = {"pass": not bad5 and len(dumps) == len(FITS), "failures": bad5}
    per, perm_p, bad3 = [], [], []
    usable = len(dumps) == len(FITS) and all(fr[3] is not None for fr in frames)
    if usable:
        for m in PE.SEEDS:
            F, L, g, row = frames[m]
            per.append((F, L, g, dumps["D_H5_s%d" % m][row]))
            pp = dumps["D_H5_s0_perm"][row]
            perm_p.append(pp)
            au = PE.auc(pp, L)
            if au is None or not (AUC_PERM[0] <= au <= AUC_PERM[1]):
                bad3.append("frames of seed %d: the area under the curve of the permuted fit lies outside %.2f to %.2f" % (m, AUC_PERM[0], AUC_PERM[1]))
    else:
        bad3.append("the permuted dump or the frames cannot be read")
    c["p3"] = {"pass": not bad3, "failures": bad3}
    if usable and not bad4:
        st = [PE.clip_stats(F, L, g, n_clip) for F, L, g, _ in per]
        pt = PE.measures(st, np.ones((1, n_clip)))
        i5 = PE.GRID.index(0.5)
        if not (abs(float(pt["M"][i5, 0]) - EXPECT["M"]) <= ID_TOL and abs(float(pt["X"][i5, 0]) - EXPECT["X"]) <= ID_TOL):
            bad4.append("form READER does not reproduce the pooled rates of row 180 within %g" % ID_TOL)
        for m, (F, L, g, _) in enumerate(per):
            if (len(F["p"]), int(L.sum()), int((~L).sum())) != (EXPECT["n_frames"], EXPECT["n_L"], EXPECT["n_not_L"]):
                bad4.append("seed %d: the counts of frames are not those of row 180" % m)
    elif not bad4:
        bad4.append("the frames cannot be read")
    c["p4"] = {"pass": not bad4, "failures": bad4}
    try:
        qs = float(io.open(os.path.join(a.root, "PRESHEAD_START.txt"), encoding="utf-8").readline().split()[0])
    except (OSError, ValueError, IndexError):
        qs = None
    bad6 = tests_ok(a.tests_fit, qs) + tests_ok(a.tests_read, now)
    if qs is None:
        bad6.append("the start marker of the fit queue cannot be read")
    c["p6"] = {"pass": not bad6, "failures": bad6}
    ok = all(c[k]["pass"] for k in CONTROLS)
    inputs["files"] = {os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f): sha_file(f) for f in files}
    res = {"card": CARD, "tier": TIER, "controls": c, "all_pass": ok, "inputs": inputs}
    if ok:
        res["payload"] = read_values(PE, per, perm_p, n_clip, a.n_boot)
        res["payload"]["fits"] = {k: {"n_samples": r["n_samples"], "n_positive": r["n_positive"], "kept_epoch": r.get("kept_epoch"), "kept_sel_loss": r.get("kept_sel_loss"), "seconds": r.get("seconds")}
                                  for k, r in recs.items()}
    return res


def print_names(payload, names):
    f = lambda x: "none" if x is None else "%.4f" % x
    iv = lambda x: "none" if x is None else "[%.4f, %.4f]" % (x[0], x[1])
    L = ["card %s, tier %s" % (CARD, TIER), "clips with frames %d, bootstrap draws %d, bar: M at most %.4f and X at most %.4f" % (payload["n_clips_with_frames"], payload["n_boot"], BAR_M, BAR_X)]
    for k, r in payload["fits"].items():
        L.append("fit %s: samples %s, with a lead %s, kept epoch %s, kept loss %s, seconds %s" % (k, r["n_samples"], r["n_positive"], r["kept_epoch"], f(r["kept_sel_loss"]), "%.0f" % r["seconds"] if r.get("seconds") else "none"))
    for k in ("READER", "HEAD", "HEAD-OR-READER", "PERMUTED"):
        P = payload["forms"][k]["pooled"]
        L.append("form %s pooled: M %s %s, X %s %s; M at gaps from 20 m %s, below 20 m %s" % (k, f(P["M"]["value"]), iv(P["M"]["interval"]), f(P["X"]["value"]), iv(P["X"]["interval"]), f(P["M_large_gaps"]), f(P["M_small_gaps"])))
        for m, s in enumerate(payload["forms"][k]["per_seed"]):
            L.append("form %s seed %d: frames %d, with a lead %d, without %d, M %s, X %s, auc %s; bins (lo-hi m: misses/leads) %s" % (
                k, m, s["n_frames"], s["n_L"], s["n_not_L"], f(s["M"]), f(s["X"]), f(s["auc"]), ", ".join("%g-%g: %d/%d" % (b["lo"], b["hi"], b["n_miss"], b["n_L"]) for b in s["bins"])))
    for k in NAMED:
        D = payload["differences"][k]
        L.append("%s minus READER: M %s %s, X %s %s" % (k, f(D["M"]["value"]), iv(D["M"]["interval"]), f(D["X"]["value"]), iv(D["X"]["interval"])))
    L.append("form HEAD by threshold: " + "; ".join("t %.1f M %s X %s" % (r["t"], f(r["M"]), f(r["X"])) for r in payload["forms"]["HEAD"]["pooled"]["by_t"]))
    h = payload["head_at_the_reader_extra_mark_rate"]
    L.append("no name: HEAD at the threshold whose extra-mark rate is nearest to the reader's: t %.1f, M %s, X %s (READER: M %s, X %s)" % (h["t"], f(h["M"]), f(h["X"]), f(h["reader_M"]), f(h["reader_X"])))
    for k in NAMED:
        L.append("NAME %s" % names[k])
    L.append("NAME %s (form for the T1-kin card: %s)" % (names["OVERALL"], names["named_form"]))
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage", choices=("controls", "names"), required=True)
    for nm in ("preserr", "t1", "join", "root", "ref-fit-record", "tests-fit", "tests-read", "controls"):
        ap.add_argument("--" + nm, default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        print("REFUSED: %s exists" % a.out)
        return 2
    if a.stage == "controls":
        import time
        res = run_controls(a, now=time.time())
        io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1))
        for k in CONTROLS:
            print("%s %s" % (k, "PASS" if res["controls"][k]["pass"] else "FAIL"))
        print("ZZ PRESHEAD-T0 controls all_pass %s" % res["all_pass"])
        return 0
    cj = jl(a.controls)
    if cj.get("card") != CARD or cj.get("all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != sorted(CONTROLS) \
            or any((v or {}).get("pass") is not True for v in cj["controls"].values()):
        print("REFUSED: the controls file is not all-pass with p0 to p6 present: no name is written")
        print("ZZ PRESHEAD-T0 names refused")
        return 3
    names = names_of(cj["payload"])
    io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps({"card": CARD, "tier": TIER, "controls_sha256": sha_file(a.controls), "names": names, "values": cj["payload"]}, indent=1))
    for ln in print_names(cj["payload"], names) + ["ZZ PRESHEAD-T0 names done"]:
        print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
