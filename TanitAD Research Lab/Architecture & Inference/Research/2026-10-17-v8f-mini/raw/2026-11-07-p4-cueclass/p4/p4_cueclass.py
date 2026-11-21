#!/usr/bin/env python3
"""p4_cueclass.py -- the P4 instrument of PREREG_P4_CUECLASS.md (frozen sha256 in ../PREREG_P4_CUECLASS.sha256).  Zero training, dev-box CPU, CUDA_VISIBLE_DEVICES=-1.

    python -I p4_cueclass.py verify-card
    python -I p4_cueclass.py fit      --featureset V+E --source banked [--pops pstop,pmove]
    python -I p4_cueclass.py fit      --featureset L_<tag> --source raw:<dir with V_L_pstop.npy, V_L_pmove.npy>
    python -I p4_cueclass.py compare  --a L_<tag> --b V+E
    python -I p4_cueclass.py controls

fit     : refits the trigger probe's LINEAR probe (tp_lib.cv_oof, the probe's own folds / C grid) on <featureset>+E, class-blind, then reads the out-of-fold scores per class
          (LIGHT / LEAD / SELF, plus MIXED / UNKNOWN and ALL) -> P4_<featureset>.json (+ P4_<featureset>_oof.npz holding the scores and the bootstrap vectors).
compare : paired Delta = AUC_c(a) - AUC_c(b), contrast D = Delta_c - Delta_SELF, the card's verdict rule -> P4_COMPARE_<a>_vs_<b>.json.
controls: card section 8 (a)-(d).
Estimator: episode(clip)-cluster bootstrap, B 2000, seed 0; the SAME resample for both feature sets and every class.  It answers "another draw of EPISODES", not "another training run".
Evidence class of every output: MEASURED, tier T0 (a representation probe on logged windows)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
THEORY = os.path.dirname(HERE).replace("\\", "/")
TP = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe"
RAW = f"{TP}/raw"
TP_LIB_SHA256 = "44b8797740af5cb01dfbdb37d35f4caba80783393c932536127824ca98ee8932"
CARD, CARD_SHA = f"{THEORY}/PREREG_P4_CUECLASS.md", f"{THEORY}/PREREG_P4_CUECLASS.sha256"
LOG_MARK = b"## LOG (append-only after freeze)\n"
K_PCA, B_BOOT, SEED = 64, 2000, 0
CLASSES = ("LIGHT", "LEAD", "SELF")
ALL_CLASSES = ("LIGHT", "LEAD", "SELF", "MIXED", "UNKNOWN")
FLOOR_POS, FLOOR_NEG, FLOOR_CLIPS = 30, 30, 20


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tp():
    """The probe's own tp_lib (imported, never copied), pinned by sha256."""
    got = sha256_file(f"{TP}/tp_lib.py")
    if got != TP_LIB_SHA256:
        raise RuntimeError(f"trigger_probe/tp_lib.py sha256 {got} != pinned {TP_LIB_SHA256}")
    if TP not in sys.path:
        sys.path.insert(0, TP)
    import tp_lib
    return tp_lib


# ------------------------------------------------------------------------------------------------ card
def card_body_sha():
    b = open(CARD, "rb").read()
    assert b.count(LOG_MARK) == 1, "log marker missing or repeated"
    return hashlib.sha256(b[:b.index(LOG_MARK)]).hexdigest()


def verify_card():
    want = open(CARD_SHA, encoding="utf-8").read().split()[0]
    got = card_body_sha()
    return {"ok": got == want, "card_sha256": got, "frozen_sha256": want}


# ------------------------------------------------------------------------------------------------ classes (card section 3)
def class_masks(tl, lead_now, no_lead):
    """LIGHT = TL & no_lead; LEAD = ~TL & lead_now; SELF = ~TL & no_lead; MIXED = TL & lead_now; UNKNOWN = lead-unknown (neither lead_now nor no_lead).  A partition."""
    tl, lead_now, no_lead = (np.asarray(x, bool) for x in (tl, lead_now, no_lead))
    if np.any(lead_now & no_lead):
        raise ValueError("lead_now and no_lead overlap")
    m = {"LIGHT": tl & no_lead, "LEAD": ~tl & lead_now, "SELF": ~tl & no_lead, "MIXED": tl & lead_now, "UNKNOWN": ~lead_now & ~no_lead}
    if int(sum(v.sum() for v in m.values())) != len(tl) or np.any(sum(v.astype(int) for v in m.values()) != 1):
        raise AssertionError("the five cells do not partition the windows")
    return m


def load_pop(pop):
    L = tp()
    A = np.load(f"{RAW}/windows_train.npz", allow_pickle=False)
    idx = np.load(f"{RAW}/sample_{pop}.npz")["idx"]
    clip = A["clip_i"][idx].astype(np.int64)
    sha = A["episode_sha12"][clip]
    return {"idx": idx, "y": A["go"][idx].astype(int), "clip": clip, "outer": L.outer_fold(sha), "inner": L.inner_fold(sha), "E": A["E"][idx].astype(np.float64),
            "masks": class_masks(A["TL_ANY"][idx], A["lead_now"][idx], A["no_lead"][idx])}


# ------------------------------------------------------------------------------------------------ features
def features(pop, source, d, featureset):
    """-> S [6, n, 64] float32 (fold f: PCA fitted on the outer-train rows; fold 5: on all rows).  'banked' = trigger_probe/raw/pca_V_<pop>.npz; 'raw:<dir>' = V_L_<pop>.npy through the probe's StreamPCA."""
    if source == "banked":
        return np.load(f"{RAW}/pca_V_{pop}.npz")["S"]
    if source.startswith("raw:"):
        L = tp()
        cache = f"{HERE}/P4_{featureset}_pca_{pop}.npz"
        if os.path.exists(cache):
            return np.load(cache)["S"]
        X = np.load(f"{source[4:]}/V_L_{pop}.npy", mmap_mode="r")
        n = len(d["y"])
        if X.shape != (n, 40 * 1024):
            raise ValueError(f"{pop}: feature file shape {X.shape} != ({n}, 40960)")
        read = lambda r: np.asarray(X[r], np.float32)
        S = np.zeros((6, n, K_PCA), np.float32)
        for f in range(6):
            rows = np.flatnonzero(d["outer"] != f) if f < 5 else np.arange(n)
            p = L.StreamPCA(k=K_PCA, seed=0).fit(read, rows, X.shape[1])
            S[f] = p.transform(read, np.arange(n))
            if f == 5:
                np.savez(f"{HERE}/P4_{featureset}_pcacomp_{pop}_all.npz", mu=p.mu, comp=p.comp, S=p.S)
            print(f"pca {featureset} {pop} fold {f}", flush=True)
        np.savez(cache, S=S)
        return S
    raise ValueError(source)


def fit_oof(S, d, extra=None, labels=None):
    """The probe's own linear refit on [PCA-64 ++ E (++ extra)]: one class-blind probe per (feature set, population)."""
    L = tp()
    E = d["E"] if extra is None else np.concatenate([d["E"], extra], 1)
    oof, hps = L.cv_oof(lambda f: np.concatenate([S[f].astype(np.float64), E], 1), d["y"] if labels is None else labels, d["outer"], d["inner"], "linear")
    return oof, hps


# ------------------------------------------------------------------------------------------------ per-class AUC, bootstrap vectors
def _ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def class_stats(oof, y, d, masks=None, B=B_BOOT, labels_for_auc=None):
    """Per class: fold-mean AUC (the card's statistic) with a bootstrap vector, the pooled AUC beside it, and the counts / evaluability.  One clip resample for every class."""
    L = tp()
    y = np.asarray(d["y"] if labels_for_auc is None else labels_for_auc)
    masks = dict(d["masks"]) if masks is None else dict(masks)
    masks["ALL"] = np.ones(len(y), bool)
    cnt, inv = L.boot_counts(d["clip"], B, SEED)
    stats, boots = {}, {}
    for c, m in masks.items():
        ix = np.flatnonzero(m)
        pos, neg = int(y[ix].sum()), int(len(ix) - y[ix].sum())
        nclip = int(len(np.unique(d["clip"][ix])))
        row = {"n": int(len(ix)), "pos": pos, "neg": neg, "clips": nclip, "evaluable": bool(pos >= FLOOR_POS and neg >= FLOOR_NEG and nclip >= FLOOR_CLIPS)}
        if len(ix) and pos and neg:
            fa, pa = L.FoldWAuc(oof[ix], y[ix], d["outer"][ix]), L.WAuc(oof[ix], y[ix])
            W = cnt[:, inv[ix]]
            bf = np.array([fa.auc(W[b]) for b in range(B)])
            bp = np.array([pa.auc(W[b]) for b in range(B)])
            row["foldmean"] = {"auc": fa.auc(), "lo": _ci(bf)[0], "hi": _ci(bf)[1]}
            row["pooled"] = {"auc": pa.auc(), "lo": _ci(bp)[0], "hi": _ci(bp)[1]}
            boots[c] = bf
        stats[c] = row
    return stats, boots


# ------------------------------------------------------------------------------------------------ the verdict rule (card section 7)
def sep_pos(x):
    return x["lo"] > 0


def sep_neg(x):
    return x["hi"] < 0


def verdict_pop(d):
    """d: {'LIGHT','LEAD','SELF': {delta,lo,hi}, 'D_LIGHT','D_LEAD': {delta,lo,hi}}.  C1 LIGHT sep>0; C2 LEAD sep>0; C3 SELF not sep>0; C4 both contrasts > 0 (points)."""
    li, le, se = d["LIGHT"], d["LEAD"], d["SELF"]
    c1 = sep_pos(li)
    c2 = sep_pos(le)
    c3 = not sep_pos(se)
    c4 = d["D_LIGHT"]["delta"] > 0 and d["D_LEAD"]["delta"] > 0
    holds = bool(c1 and c2 and c3 and c4)
    sub = None
    if not holds:
        if any(sep_neg(x) for x in (li, le, se)):
            sub = "F-e"
        elif sep_pos(se):
            sub = "F-c"
        elif not c1 and not c2:
            sub = "F-a"
        elif c1 != c2:
            sub = "F-b"
        else:
            sub = "F-d"
    return {"holds": holds, "C1": bool(c1), "C2": bool(c2), "C3": bool(c3), "C4": bool(c4), "subcase": sub,
            "contrast_separated": bool(d["D_LIGHT"]["lo"] > 0 and d["D_LEAD"]["lo"] > 0), "self_uninformative": bool(se["hi"] >= min(li["delta"], le["delta"]))}


def verdict_overall(per_pop):
    """per_pop: {pop: None (not evaluable; reason elsewhere) or the dict verdict_pop reads}.  -> {'verdict': HOLDS | FAILS | NOT EVALUABLE, ...}."""
    res = {p: (None if v is None else verdict_pop(v)) for p, v in per_pop.items()}
    ev = {p: r for p, r in res.items() if r is not None}
    if not ev:
        return {"verdict": "NOT EVALUABLE", "per_pop": res, "subcase": None}
    if all(r["holds"] for r in ev.values()):
        return {"verdict": "HOLDS", "per_pop": res, "subcase": None}
    first = next(r for r in ev.values() if not r["holds"])
    return {"verdict": "FAILS", "per_pop": res, "subcase": first["subcase"]}


# ------------------------------------------------------------------------------------------------ commands
def cmd_fit(featureset, source, pops):
    out = {"featureset": featureset, "source": source, "card": verify_card(), "tp_lib_sha256": TP_LIB_SHA256, "pops": {}}
    if not out["card"]["ok"]:
        raise SystemExit("the card on disk does not match its frozen sha256: refusing to fit")
    arrs = {}
    for pop in pops:
        t0 = time.time()
        d = load_pop(pop)
        S = features(pop, source, d, featureset)
        oof, hps = fit_oof(S, d)
        stats, boots = class_stats(oof, d["y"], d)
        out["pops"][pop] = {"n": int(len(d["y"])), "n_clips": int(len(np.unique(d["clip"]))), "classes": stats, "hp_per_fold": {str(k): v["hp"] for k, v in hps.items()}, "seconds": time.time() - t0}
        arrs[f"oof_{pop}"] = oof
        for c, v in boots.items():
            arrs[f"boot_{pop}_{c}"] = v
        for c in (*ALL_CLASSES, "ALL"):
            s = stats[c]
            print(featureset, pop, c, "n", s["n"], "foldmean", None if "foldmean" not in s else round(s["foldmean"]["auc"], 4), flush=True)
    np.savez(f"{HERE}/P4_{featureset}_oof.npz", **arrs)
    json.dump(out, open(f"{HERE}/P4_{featureset}.json", "w"), indent=1)
    return out


def deltas(a_arr, b_arr, pop, stats_a, stats_b):
    """Paired Delta per class and contrast D from two fit outputs; None when a class is not evaluable."""
    if not all(stats_a[c]["evaluable"] and stats_b[c]["evaluable"] for c in CLASSES):
        return None
    dl = {}
    bd = {}
    for c in CLASSES:
        bd[c] = a_arr[f"boot_{pop}_{c}"] - b_arr[f"boot_{pop}_{c}"]
        pt = stats_a[c]["foldmean"]["auc"] - stats_b[c]["foldmean"]["auc"]
        lo, hi = _ci(bd[c])
        dl[c] = {"delta": pt, "lo": lo, "hi": hi}
    for c in ("LIGHT", "LEAD"):
        v = bd[c] - bd["SELF"]
        lo, hi = _ci(v)
        dl[f"D_{c}"] = {"delta": dl[c]["delta"] - dl["SELF"]["delta"], "lo": lo, "hi": hi}
    return dl


def compare_arrays(a_arr, b_arr, sa, sb, pops):
    per_pop, detail = {}, {}
    for pop in pops:
        dl = deltas(a_arr, b_arr, pop, sa["pops"][pop]["classes"], sb["pops"][pop]["classes"])
        per_pop[pop] = dl
        detail[pop] = dl if dl is not None else "NOT EVALUABLE: a class is below the floor (30 GO, 30 not-GO, 20 clips)"
    return verdict_overall(per_pop), detail


def cmd_compare(a, b):
    sa, sb = json.load(open(f"{HERE}/P4_{a}.json")), json.load(open(f"{HERE}/P4_{b}.json"))
    pops = [p for p in sa["pops"] if p in sb["pops"]]
    aa, bb = dict(np.load(f"{HERE}/P4_{a}_oof.npz")), dict(np.load(f"{HERE}/P4_{b}_oof.npz"))
    v, detail = compare_arrays(aa, bb, sa, sb, pops)
    out = {"a": a, "b": b, "card": verify_card(), "estimator": "episode-cluster bootstrap, B 2000, seed 0, paired (same resample for both feature sets and all classes); answers 'another draw of episodes' only", "verdict": v["verdict"],
           "subcase": v["subcase"], "per_pop_verdict": v["per_pop"], "deltas": detail, "pooled_beside": {p: {c: {"a": sa["pops"][p]["classes"][c].get("pooled"), "b": sb["pops"][p]["classes"][c].get("pooled")} for c in CLASSES} for p in pops}}
    json.dump(out, open(f"{HERE}/P4_COMPARE_{a}_vs_{b}.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("verdict", "subcase", "per_pop_verdict")}, indent=1))
    return out


def planted(d, mask, amp=0.5, seed=3):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(len(d["y"]))
    return (z + amp * (2 * d["y"] - 1) * mask.astype(float))[:, None]


def cmd_controls(pops=("pstop", "pmove")):
    L = tp()
    res = {"card": verify_card(), "a": {}, "b": {}, "c": {}, "d": {}}
    for pop in pops:
        d = load_pop(pop)
        S = np.load(f"{RAW}/pca_V_{pop}.npz")["S"]
        y = d["y"]
        # (a) reproduction of the banked V+E
        oof, _ = fit_oof(S, d)
        bank = np.load(f"{RAW}/oof_{pop}_V_linear.npz")["V+E"]
        banked = json.load(open(f"{RAW}/res_{pop}_V_linear.json"))["arms"]["V+E"]
        pooled, fm = L.WAuc(oof, y).auc(), L.FoldWAuc(oof, y, d["outer"]).auc()
        target = 0.680 if pop == "pstop" else 0.785
        ra = {"max_abs_score_diff": float(np.abs(oof - bank).max()), "pooled": pooled, "pooled_banked": banked["auc"], "foldmean": fm, "foldmean_banked": banked["foldmean"]["auc"], "briefed_round_number": target}
        ra["pass"] = bool(ra["max_abs_score_diff"] <= 1e-6 and abs(pooled - banked["auc"]) <= 1e-6 and abs(fm - banked["foldmean"]["auc"]) <= 1e-6 and abs(pooled - target) <= 1e-3 and abs(fm - target) <= 1e-3)
        res["a"][pop] = ra
        stats_base, boots_base = class_stats(oof, y, d)
        base_arr = {f"boot_{pop}_{c}": boots_base[c] for c in boots_base}
        # (b) label shuffle (seed 1): the whole pipeline on permuted labels, classes read from the same masks
        ys = L.shuffle_labels(y, 1)
        oofs, _ = fit_oof(S, d, labels=ys)
        st, _ = class_stats(oofs, ys, d, labels_for_auc=ys)
        rb = {c: st[c]["foldmean"]["auc"] for c in CLASSES if "foldmean" in st[c]}
        allcov = st["ALL"]["foldmean"]["lo"] <= 0.5 <= st["ALL"]["foldmean"]["hi"]
        res["b"][pop] = {"class_foldmean_auc": rb, "all": st["ALL"]["foldmean"], "pass": bool(all(abs(v - 0.5) <= 0.05 for v in rb.values()) and allcov)}
        # (c) planted signal and leak
        pl = planted(d, np.ones(len(y), bool))
        oofp, _ = fit_oof(S, d, extra=pl)
        stp, _ = class_stats(oofp, y, d)
        oofl, _ = fit_oof(S, d, extra=y[:, None].astype(np.float64))
        stl, _ = class_stats(oofl, y, d)
        res["c"][pop] = {"planted": {c: stp[c]["foldmean"]["auc"] for c in CLASSES}, "leak": {c: stl[c]["foldmean"]["auc"] for c in CLASSES},
                         "pass": bool(all(stp[c]["foldmean"]["auc"] >= 0.74 for c in CLASSES) and all(stl[c]["foldmean"]["auc"] >= 0.99 for c in CLASSES))}
        # (d) class-specific plants through compare
        res["d"][pop] = {}
        M = d["masks"]
        variants = {"LIGHT+LEAD": (M["LIGHT"] | M["LEAD"], 3), "SELF": (M["SELF"], 3), "ALL3": (M["LIGHT"] | M["LEAD"] | M["SELF"], 3), "NOISE": (np.zeros(len(y), bool), 4)}
        for nm, (mask, seed) in variants.items():
            o, _ = fit_oof(S, d, extra=planted(d, mask, seed=seed))
            sv, bv = class_stats(o, y, d)
            arr = {f"boot_{pop}_{c}": bv[c] for c in bv}
            v, det = compare_arrays(arr, base_arr, {"pops": {pop: {"classes": sv}}}, {"pops": {pop: {"classes": stats_base}}}, [pop])
            res["d"][pop][nm] = {"verdict": v["verdict"], "subcase": v["subcase"], "deltas": det[pop]}
        r = res["d"][pop]
        self_sep = lambda k: isinstance(r[k]["deltas"], dict) and sep_pos(r[k]["deltas"]["SELF"])
        r["pass"] = bool(r["LIGHT+LEAD"]["verdict"] == "HOLDS" and r["SELF"]["verdict"] == "FAILS" and self_sep("SELF") and r["ALL3"]["verdict"] == "FAILS" and self_sep("ALL3") and r["NOISE"]["verdict"] == "FAILS")
        print(pop, "controls a/b/c/d", res["a"][pop]["pass"], res["b"][pop]["pass"], res["c"][pop]["pass"], r["pass"], flush=True)
    res["all_pass"] = bool(all(res[k][p]["pass"] for k in "abcd" for p in pops))
    json.dump(res, open(f"{HERE}/P4_CONTROLS.json", "w"), indent=1, default=float)
    print("ALL CONTROLS PASS" if res["all_pass"] else "A CONTROL FAILED", flush=True)
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("verify-card")
    f = sp.add_parser("fit")
    f.add_argument("--featureset", required=True)
    f.add_argument("--source", required=True)
    f.add_argument("--pops", default="pstop,pmove")
    c = sp.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    sp.add_parser("controls")
    a = ap.parse_args(argv)
    if a.cmd == "verify-card":
        r = verify_card()
        print(json.dumps(r))
        return 0 if r["ok"] else 1
    if a.cmd == "fit":
        cmd_fit(a.featureset, a.source, a.pops.split(","))
    elif a.cmd == "compare":
        cmd_compare(a.a, a.b)
    else:
        cmd_controls()
    return 0


if __name__ == "__main__":
    sys.exit(main())
