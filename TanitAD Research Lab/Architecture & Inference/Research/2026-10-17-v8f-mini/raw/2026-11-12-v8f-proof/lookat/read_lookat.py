#!/usr/bin/env python
"""read_lookat.py -- read of card PREREG_V8F_PROOF_ADD107 (LOOKAT-T0). CPU, numpy only. Tier T0 (recorded frames; a diagnostic of a perception head, not driving performance).

  python read_lookat.py --stage controls --root R --out-root O --box-dir B --eval-join J --gapfiles f1,f2,f3,f4,f5 --out O/read/LOOKAT_CONTROLS.json
  python read_lookat.py --stage names --controls O/read/LOOKAT_CONTROLS.json --out O/read/LOOKAT_RESULT.json
  python read_lookat.py --selftest

Stage controls: runs k0 to k6 of card section 5 and prints ONLY `k<n> PASS|FAIL` lines (plus the one count line k4 asks for, when it applies) and the last line. Every value goes to the controls file: under `payload` when
all seven pass, under `withheld` when one fails (then no name exists). Stage names: refuses unless k0 to k6 are all present and PASS; writes the names of section 3 and the no-name tables, and derives the name a second time apart.
Last line: ZZ READ-LOOKAT <stage> rc <rc> controls_all_pass <bool>"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD107"
PRE = "LOOKAT-T0"
SEEDS = (0, 1, 2)
FITS = ("s0", "s1", "s2", "perm")
CONTROLS = tuple("k%d" % i for i in range(7))
N_BOOT, BOOT_SEED = 2000, 107
EXPECT = {"fit": 450, "sel": 50, "eval": 141, "rows": 3841, "outside": 486}
HP_CARD = {"lr": 1e-3, "weight_decay": 0.05, "batch": 256, "epochs": 40, "warmup_epochs": 1, "beta_m": 0.25, "token_drop": 0.1, "d_model": 256, "n_heads": 4}
CROP_FIT_SHA16 = "61bf2f4e8e7ec61f"
S8_OUT_BAR = 0.50
GAP_BINS = ((0, 10), (10, 20), (20, 40), (40, 60))
AZ_BINS = ((0, 4), (4, 8), (8, 16), (16, None))
TIER = "T0 (recorded frames, no rollout; a diagnostic of a perception head, not driving performance)"
FAMILIES = ("LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC: not read (n = 0 for each; reason: no rollout and no plan in this card, card section 4)")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(p):
    with io.open(p, encoding="utf-8") as fh:
        return json.load(fh)


def ids_of(p):
    with io.open(p, encoding="utf-8") as fh:
        return [ln.split()[0] for ln in fh if ln.strip()]


# ============================================================================================ rows and values
def load_npz(p):
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def row_index(J, names, cidx, frame):
    """join row of (clip name list `names`, clip index, frame) triples; -1 where not in the join."""
    jn = {str(c): i for i, c in enumerate(J["clip"])}
    of = np.array([jn.get(str(c), -1) for c in names], np.int64)
    ji = of[np.asarray(cidx, np.int64)]
    off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
    jc = np.where(ji >= 0, ji, 0)
    fr = np.asarray(frame, np.int64)
    ok = (ji >= 0) & (fr >= 0) & (fr < n[jc])
    return np.where(ok, off[jc] + fr, -1), ok


def on_rows(J, D, key, cname="clips", cidx="clip_index"):
    nrow = int(np.asarray(J["n"], np.int64).sum())
    row, ok = row_index(J, D[cname], D[cidx], D["frame"])
    v = np.asarray(D[key])
    out = np.zeros(nrow, bool) if v.dtype == bool else np.full(nrow, np.nan)
    out[row[ok]] = v[ok]
    return out


def label_mask(J):
    g = np.asarray(J["lead_gap"], np.float64)
    return (np.asarray(J["lead_valid"]) == 1) & np.isfinite(g) & (g >= 0.0) & (g <= 60.0)


def row_set(J, gapfiles):
    """label lead (valid, finite, 0 to 60 m) and an output with a finite gap in all gap files. -> bool [rows]"""
    sel = label_mask(J).copy()
    for G in gapfiles:
        sel &= on_rows(J, G, "has_out").astype(bool) & np.isfinite(on_rows(J, G, "gap"))
    return sel


def box_on_rows(J, boxes):
    """boxes {clip: (has, az, el)} -> arrays on the join rows (has False and 0 where no box)"""
    n = np.asarray(J["n"], np.int64)
    off = np.asarray(J["off"], np.int64)
    nrow = int(n.sum())
    has, az, el = np.zeros(nrow, bool), np.zeros(nrow), np.zeros(nrow)
    for i, c in enumerate(J["clip"]):
        b = boxes.get(str(c))
        if b is None:
            continue
        s = slice(off[i], off[i] + n[i])
        has[s], az[s], el[s] = b[0], b[1], b[2]
    return {"has": has, "az": az, "el": el}


def cluster_sum(ci, m, nc):
    return np.bincount(ci, weights=np.asarray(m, np.float64), minlength=nc)


def boot_diff(h_a, h_b, r, n_boot=N_BOOT, seed=BOOT_SEED):
    """paired cluster bootstrap of (sum h_a - sum h_b) / sum r over clips: the same draws for both, RandomState(seed), two-sided 95 % percentile. -> [lo, hi]"""
    h_a, h_b, r = (np.asarray(x, np.float64) for x in (h_a, h_b, r))
    nc = len(r)
    idx = np.random.RandomState(seed).randint(0, nc, size=(n_boot, nc))
    den = r[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (h_a[idx].sum(axis=1) - h_b[idx].sum(axis=1)) / den
    d = d[np.isfinite(d)]
    if d.size < n_boot // 2:
        return None
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def _stats(daz, dele, gap, az):
    h4, h8 = (daz <= 4) & (dele <= 4), (daz <= 8) & (dele <= 8)
    out = {"S4": float(h4.mean()), "S8": float(h8.mean()), "med_d_az": float(np.median(daz)), "p90_d_az": float(np.percentile(daz, 90)),
           "med_d_el": float(np.median(dele)), "p90_d_el": float(np.percentile(dele, 90))}
    outside = np.abs(az) > 8
    out["S8_out"] = float(h8[outside].mean()) if outside.any() else None
    out["by_gap"] = []
    for lo, hi in GAP_BINS:
        m = (gap >= lo) & ((gap < hi) if hi < 60 else (gap <= hi))
        out["by_gap"].append({"bin": "%d-%d m" % (lo, hi), "n": int(m.sum()), "S4": float(h4[m].mean()) if m.any() else None, "S8": float(h8[m].mean()) if m.any() else None})
    out["by_az"] = []
    for lo, hi in AZ_BINS:
        m = (np.abs(az) > lo) & ((np.abs(az) <= hi) if hi else True)
        if lo == 0:
            m = np.abs(az) <= hi
        out["by_az"].append({"bin": ("%d-%d deg" % (lo, hi)) if hi else "above %d deg" % lo, "n": int(m.sum()), "S4": float(h4[m].mean()) if m.any() else None, "S8": float(h8[m].mean()) if m.any() else None})
    return out, h4, h8, outside


def compute_values(J, sel, box, preds, n_boot=N_BOOT, seed=BOOT_SEED):
    """J the evaluation join, sel the row set, box the label centres on the join rows, preds {fit: {az, el on the join rows}} for s0, s1, s2, perm. -> payload (all values of the read)."""
    n = np.asarray(J["n"], np.int64)
    nc = len(n)
    ci_all = np.repeat(np.arange(nc), n)
    rows = np.flatnonzero(sel)
    ci = ci_all[rows]
    az, el, has = box["az"][rows], box["el"][rows], box["has"][rows]
    gap = np.asarray(J["lead_gap"], np.float64)[rows]
    outside = np.abs(az) > 8
    P = {"n_rows": int(len(rows)), "n_has_box": int(has.sum()), "n_outside": int(outside.sum()), "n_el_off8": int((np.abs(el + 2) > 8).sum()), "n_clips": int(nc), "n_boot": n_boot, "seed": seed,
         "interval_level": "two-sided 95 % percentile, paired cluster bootstrap over the evaluation clips, fits held fixed", "tier": TIER, "families": FAMILIES}
    P["n_az_beyond_target_clip"] = int((np.abs(az) > 40).sum())             # the target is clipped at 32 degrees: d_az <= 8 needs |az| <= 40 unless the output extrapolates
    P["n_el_beyond_target_clip"] = int(((el < -26) | (el > 22)).sum())
    P["n_az_beyond_window"] = int((np.abs(az) > 16).sum())                  # outside the 32-degree window as well (no pixel of the lead in any window)
    P["n_el_off8_note"] = "rows with |el + 2| > 8 (S8_fixed is then computed as defined, not by the closed form)"
    P["S8_fixed_closed_form"] = float((len(rows) - int(outside.sum())) / max(len(rows), 1))
    P["n_pred_missing"] = {k: int((~(np.isfinite(preds[k]["az"][rows]) & np.isfinite(preds[k]["el"][rows]))).sum()) for k in preds}
    sums = {"r": cluster_sum(ci, np.ones(len(rows)), nc), "out_r": cluster_sum(ci, outside, nc)}
    fx, h4f, h8f, _ = _stats(np.abs(az), np.abs(el + 2), gap, az)
    P["fixed"] = fx
    sums["fixed_h4"] = cluster_sum(ci, h4f, nc)
    P["fits"], P["intervals"], hh = {}, {}, {}
    for k in preds:
        pa, pe = preds[k]["az"][rows], preds[k]["el"][rows]
        st, h4, h8, _ = _stats(np.abs(pa - az), np.abs(pe - el), gap, az)
        P["fits"][k] = st
        hh[k] = h4
        sums[k] = {"h4": cluster_sum(ci, h4, nc), "h8o": cluster_sum(ci, h8 & outside, nc)}
    for k in preds:
        if k != "perm":
            P["intervals"][k] = {"diff": P["fits"][k]["S4"] - fx["S4"], "iv": boot_diff(sums[k]["h4"], sums["fixed_h4"], sums["r"], n_boot, seed)}
    if "perm" in preds and "s0" in preds:
        P["intervals"]["perm_vs_s0"] = {"diff": P["fits"]["perm"]["S4"] - P["fits"]["s0"]["S4"], "iv": boot_diff(sums["perm"]["h4"], sums["s0"]["h4"], sums["r"], n_boot, seed)}
    P["sums"] = {k: (v.tolist() if isinstance(v, np.ndarray) else {a: b.tolist() for a, b in v.items()}) for k, v in sums.items()}
    return P


# ============================================================================================ names (two derivations)
def name_of(P):
    """card section 3, first derivation (from the values of the payload)."""
    ivs = [P["intervals"][k]["iv"] for k in ("s0", "s1", "s2")]
    s8 = [P["fits"][k]["S8_out"] for k in ("s0", "s1", "s2")]
    if any(i is None for i in ivs) or any(v is None for v in s8):
        return PRE + "-NOT-SEPARATED"
    c1 = all(i[0] > 0.0 for i in ivs)
    c2 = all(v >= S8_OUT_BAR for v in s8)
    if c1 and c2:
        return PRE + "-LOCATES"
    if c1:
        return PRE + "-CENTRES-ONLY"
    if any(i[1] < 0.0 for i in ivs):
        return PRE + "-DOES-NOT-LOCATE"
    return PRE + "-NOT-SEPARATED"


def name_second(P):
    """second derivation, written apart: from the per-clip sums only, plain python loops, hand-made percentiles. -> (name, {fit: [lo, hi]})."""
    S = P["sums"]
    r = S["r"]
    nc = len(r)
    idx = np.random.RandomState(P["seed"]).randint(0, nc, size=(P["n_boot"], nc)).tolist()
    R = float(sum(r))

    def pct(v, q):
        v = sorted(v)
        pos = (len(v) - 1) * q / 100.0
        a = int(pos)
        b = min(a + 1, len(v) - 1)
        return v[a] + (v[b] - v[a]) * (pos - a)

    def iv(ha, hb):
        v = []
        for row in idx:
            den = sum(r[j] for j in row)
            if den > 0:
                v.append((sum(ha[j] for j in row) - sum(hb[j] for j in row)) / den)
        return [pct(v, 2.5), pct(v, 97.5)]
    out, c1, hiNeg, c2 = {}, True, False, True
    for k in ("s0", "s1", "s2"):
        lo, hi = iv(S[k]["h4"], S["fixed_h4"])
        out[k] = [lo, hi]
        c1 = c1 and lo > 0.0
        hiNeg = hiNeg or hi < 0.0
        o = float(sum(S["out_r"]))
        c2 = c2 and o > 0 and (float(sum(S[k]["h8o"])) / o) >= S8_OUT_BAR
    nm = "LOCATES" if (c1 and c2) else ("CENTRES-ONLY" if c1 else ("DOES-NOT-LOCATE" if hiNeg else "NOT-SEPARATED"))
    return PRE + "-" + nm, out


def derive_name(P):
    n1 = name_of(P)
    n2, ivs = name_second(P)
    agree = n1 == n2 and all(P["intervals"][k]["iv"] is not None and np.allclose(P["intervals"][k]["iv"], ivs[k], atol=1e-9, rtol=0) for k in ivs)
    return n1, n2, agree


# ============================================================================================ controls (pure functions of facts)
def _res(bad, **kw):
    d = {"PASS": not bad, "failures": bad}
    d.update(kw)
    return d


def ctl_k0(f, ex=EXPECT):
    """lists and targets: pilot list hash and shape, 450 + 50 clips, every clip has a box file with the frame count of its token file, the four fit records read the same clips."""
    bad = []
    if f.get("train_list_sha") != f.get("rec99_train_list_sha"):
        bad.append("train list hash differs from the fit record of card ADD99")
    if f.get("eval_list_sha") != f.get("rec99_eval_list_sha"):
        bad.append("eval list hash differs from the fit record of card ADD99")
    if (f.get("n_fit"), f.get("n_sel")) != (ex["fit"], ex["sel"]):
        bad.append("train list has %s fit and %s sel clips, expected %d and %d" % (f.get("n_fit"), f.get("n_sel"), ex["fit"], ex["sel"]))
    if f.get("n_eval") != ex["eval"]:
        bad.append("eval list has %s clips, expected %d" % (f.get("n_eval"), ex["eval"]))
    if f.get("n_overlap"):
        bad.append("%s clips are in both lists" % f.get("n_overlap"))
    nb = [c for c, v in (f.get("box_vs_tok") or {}).items() if v is None or v[0] != v[1]]
    if nb or len(f.get("box_vs_tok") or {}) != ex["fit"] + ex["sel"] + ex["eval"]:
        bad.append("box files: %d clips without a box file or with another frame count (of %d checked)" % (len(nb), len(f.get("box_vs_tok") or {})))
    if not f.get("records_read_list_clips"):
        bad.append("a fit record's clips_read differs from the train list")
    return _res(bad, n_box_checked=len(f.get("box_vs_tok") or {}), first_bad_clips=nb[:3])


def ctl_k1(f, ex=EXPECT):
    """tokens: each of the 641 token files of arm D has the manifest's byte size, frame count and sha256."""
    bad = []
    if f.get("manifest_error"):
        bad.append(f["manifest_error"])
    if f.get("n_checked") != ex["fit"] + ex["sel"] + ex["eval"]:
        bad.append("%s token files checked, expected %d" % (f.get("n_checked"), ex["fit"] + ex["sel"] + ex["eval"]))
    for key in ("missing", "size", "frames", "sha"):
        if f.get(key):
            bad.append("%d token files with %s differing (first %s)" % (len(f[key]), key, f[key][0]))
    if not f.get("sha_checked"):
        bad.append("sha256 not verified")
    return _res(bad)


def ctl_k2(recs, heads, ex=EXPECT):
    """fits: four records (s0, s1, s2, perm) with the settings of card section 2, head files equal to the records, the three real fits not permuted, one tool and one imported crop_fit for all four."""
    bad = []
    for k in FITS:
        r = recs.get(k)
        if not r:
            bad.append("%s: no fit record" % k)
            continue
        want_seed = {"s0": 0, "s1": 1, "s2": 2, "perm": 0}[k]
        if r.get("seed") != want_seed:
            bad.append("%s: seed %s" % (k, r.get("seed")))
        if bool(r.get("permute_targets")) != (k == "perm"):
            bad.append("%s: permute_targets is %s" % (k, r.get("permute_targets")))
        if r.get("hparams") != HP_CARD or r.get("hparams_equal_card") is not True:
            bad.append("%s: settings are not those of card section 2" % k)
        if r.get("card") != CARD:
            bad.append("%s: card %s" % (k, r.get("card")))
        if heads.get(k) is None or heads.get(k) != r.get("head_pt_sha256"):
            bad.append("%s: head file does not match its record" % k)
        if (r.get("n_samples") or {}).get("fit", 0) <= 0:
            bad.append("%s: no fit samples" % k)
        if len(r.get("history") or []) != HP_CARD["epochs"]:
            bad.append("%s: %s epochs on record" % (k, len(r.get("history") or [])))
        if r.get("clips_read", {}).get("fit") is None or len(r["clips_read"]["fit"]) != ex["fit"] or len(r["clips_read"]["sel"]) != ex["sel"]:
            bad.append("%s: clips read are not %d + %d" % (k, ex["fit"], ex["sel"]))
    if len({(r or {}).get("tool_sha256") for r in recs.values()}) != 1 or len({(r or {}).get("crop_fit_sha256") for r in recs.values()}) != 1:
        bad.append("the four records do not share one locator tool and one crop_fit.py")
    cf = {(r or {}).get("crop_fit_sha256", "")[:16] for r in recs.values()}
    if cf != {CROP_FIT_SHA16}:
        bad.append("crop_fit.py imported is not the tool of card ADD99 (sha256 begins %s)" % CROP_FIT_SHA16)
    return _res(bad)


def ctl_k3(P):
    """deliberate regression: the upper end of the interval of S4_perm - S4_seed0 is below 0."""
    iv = (P.get("intervals") or {}).get("perm_vs_s0", {}).get("iv")
    return _res([] if (iv is not None and iv[1] < 0.0) else ["the permuted fit does not locate worse than seed 0 (interval %s)" % (iv,)])


def ctl_k4(P, ex=EXPECT):
    """the reference recomputes: 3,841 rows, all with a box centre, 486 outside rows, S8_fixed as the closed form when no row has |el + 2| > 8, a finite output of all four fits on every row."""
    bad = []
    if P["n_rows"] != ex["rows"]:
        bad.append("%d rows, expected %d" % (P["n_rows"], ex["rows"]))
    if P["n_has_box"] != P["n_rows"]:
        bad.append("%d rows have no box centre" % (P["n_rows"] - P["n_has_box"]))
    if P["n_outside"] != ex["outside"]:
        bad.append("%d outside rows, expected %d" % (P["n_outside"], ex["outside"]))
    closed = (ex["rows"] - ex["outside"]) / ex["rows"]
    if P["n_el_off8"] == 0 and abs(P["fixed"]["S8"] - closed) > 1e-9:
        bad.append("S8_fixed %.12f differs from the closed form %.12f" % (P["fixed"]["S8"], closed))
    if any(P["n_pred_missing"].values()):
        bad.append("rows without an output: %s" % P["n_pred_missing"])
    return _res(bad, n_el_off8=P["n_el_off8"])


def ctl_k5(I):
    """identity with the tool of card ADD99 (written by locator_fit.py --identity)."""
    bad = []
    if not isinstance(I, dict) or I.get("control") != "k5":
        return _res(["IDENTITY_K5.json missing or not of control k5"])
    if not (isinstance(I.get("abs_diff"), (int, float)) and I["abs_diff"] < 1e-4):
        bad.append("selection loss differs by %s (bar 1e-4)" % I.get("abs_diff"))
    if I.get("n_sel_samples") != I.get("n_sel_samples_record") or I.get("head_pt_sha256") != I.get("head_pt_sha256_record"):
        bad.append("sample count or head file differs from the record")
    if (I.get("crop_fit_sha256") or "")[:16] != CROP_FIT_SHA16:
        bad.append("crop_fit.py of the identity run is not the tool of card ADD99")
    if I.get("PASS") is not True:
        bad.append("the identity run did not pass")
    return _res(bad)


def ctl_k6(text):
    """kit tests: the lead's pytest output reads `N passed`, N > 0, with no failed and no error."""
    t = text or ""
    m = re.findall(r"(\d+) passed", t)
    ok = bool(m) and int(m[-1]) > 0 and not re.search(r"\d+ (failed|error)", t)
    return _res([] if ok else ["the test output does not read N passed without failed or error"], passed=int(m[-1]) if m else None)


# ============================================================================================ disk (pod)
def tok_n(root, c):
    return int(np.load(os.path.join(root, "tok", "D", c + ".npy"), mmap_mode="r").shape[0])


def box_n(box_dir, c):
    p = os.path.join(box_dir, c + ".npz")
    return int(len(np.load(p)["has_lead"])) if os.path.exists(p) else None


def gather_k0(a):
    tp, ep = os.path.join(a.root, "lists", "train_final.txt"), os.path.join(a.root, "lists", "eval141.txt")
    rows = [ln.split() for ln in io.open(tp, encoding="utf-8") if ln.strip()]
    fit, sel = [r[0] for r in rows if r[1] == "fit"], [r[0] for r in rows if r[1] == "sel"]
    ev = ids_of(ep)
    r99 = read_json(os.path.join(a.root, "fits", "D_H5_s0", "FIT_RECORD.json"))
    f = {"train_list_sha": sha_file(tp), "rec99_train_list_sha": r99.get("train_list_sha256"), "eval_list_sha": sha_file(ep), "rec99_eval_list_sha": r99.get("eval_list_sha256"),
         "n_fit": len(fit), "n_sel": len(sel), "n_eval": len(ev), "n_overlap": len(set(fit + sel) & set(ev)), "box_vs_tok": {}}
    for c in fit + sel + ev:
        try:
            nb = box_n(a.box_dir, c)
            f["box_vs_tok"][c] = None if nb is None else (nb, tok_n(a.root, c))
        except (OSError, ValueError, KeyError):
            f["box_vs_tok"][c] = None
    f["records_read_list_clips"] = True
    for k in FITS:
        try:
            r = read_json(fit_record_path(a, k))
            if r.get("clips_read", {}).get("fit") != fit or r.get("clips_read", {}).get("sel") != sel:
                f["records_read_list_clips"] = False
        except (OSError, ValueError):
            f["records_read_list_clips"] = False
    return f, fit + sel + ev


def gather_k1(a, clips):
    f = {"missing": [], "size": [], "frames": [], "sha": [], "n_checked": 0, "sha_checked": False, "manifest_error": None}
    mp = a.manifest if a.manifest and os.path.isfile(a.manifest) else None
    for cand in (os.path.join(a.root, "tok", "D", "MANIFEST.json"), "/workspace/v8proof/cropread/enc_out/tok/D/MANIFEST.json"):
        if mp is None and os.path.isfile(cand):
            mp = cand
    if mp is None:
        f["manifest_error"] = "no MANIFEST.json of arm D found"
        return f
    man = read_json(mp)["clips"]
    todo = []
    for c in clips:
        m = man.get(c)
        p = os.path.join(a.root, "tok", "D", c + ".npy")
        if m is None or not os.path.exists(p):
            f["missing"].append(c)
            continue
        if os.path.getsize(p) != m["bytes"]:
            f["size"].append(c)
        elif tok_n(a.root, c) != m["n_frames"]:
            f["frames"].append(c)
        else:
            todo.append((c, p, m["sha256"]))
    if a.no_sha:
        f["sha_checked"] = False
    else:
        with concurrent.futures.ThreadPoolExecutor(4) as ex:
            for (c, p, want), got in zip(todo, ex.map(lambda t: sha_file(t[1]), todo)):
                if got != want:
                    f["sha"].append(c)
        f["sha_checked"] = True
    f["n_checked"] = len(clips)
    return f


def fit_record_path(a, k):
    return os.path.join(a.out_root, "fits", "LOC_s%s%s" % (k[1:] if k != "perm" else "0", "_perm" if k == "perm" else ""), "FIT_RECORD.json")


def load_all(a):
    J = load_npz(a.eval_join)
    gfs = [load_npz(p) for p in a.gapfiles.split(",")]
    sel = row_set(J, gfs)
    boxes = {}
    for c in [str(c) for c in J["clip"]]:
        p = os.path.join(a.box_dir, c + ".npz")
        if os.path.exists(p):
            z = np.load(p)
            has = z["has_lead"].astype(bool) & np.isfinite(z["ctr_az"]) & np.isfinite(z["ctr_el"])
            boxes[c] = (has, np.where(has, z["ctr_az"], 0.0), np.where(has, z["ctr_el"], 0.0))
    box = box_on_rows(J, boxes)
    preds = {}
    for k in FITS:
        nm = "LOC_s%s%s" % (k[1:] if k != "perm" else "0", "_perm" if k == "perm" else "")
        Pz = load_npz(os.path.join(a.out_root, "pred", nm + ".npz"))
        preds[k] = {"az": on_rows(J, Pz, "az_pred"), "el": on_rows(J, Pz, "el_pred")}
    return J, sel, box, preds


def run_controls(a):
    ctl, P = {}, None
    f0, clips = gather_k0(a)
    ctl["k0"] = ctl_k0(f0, a.expect)
    try:
        ctl["k1"] = ctl_k1(gather_k1(a, clips), a.expect)
    except (OSError, ValueError, KeyError) as ex:
        ctl["k1"] = _res(["tokens could not be checked: %s: %s" % (type(ex).__name__, ex)])
    recs, heads = {}, {}
    for k in FITS:
        p = fit_record_path(a, k)
        try:
            recs[k] = read_json(p)
            heads[k] = sha_file(os.path.join(os.path.dirname(p), "head.pt"))
        except (OSError, ValueError):
            recs[k], heads[k] = None, None
    ctl["k2"] = ctl_k2(recs, heads, a.expect)
    try:
        J, sel, box, preds = load_all(a)
        P = compute_values(J, sel, box, preds, a.n_boot)
        ctl["k3"], ctl["k4"] = ctl_k3(P), ctl_k4(P, a.expect)
    except (OSError, ValueError, KeyError, TypeError) as ex:
        ctl["k3"] = ctl["k4"] = _res(["the read could not be computed: %s: %s" % (type(ex).__name__, ex)])
    try:
        ctl["k5"] = ctl_k5(read_json(os.path.join(a.out_root, "IDENTITY_K5.json")))
    except (OSError, ValueError):
        ctl["k5"] = _res(["IDENTITY_K5.json is missing"])
    tp = os.path.join(a.out_root, "kit", "lead_pytest.out")
    ctl["k6"] = ctl_k6(io.open(tp, encoding="utf-8", errors="replace").read() if os.path.isfile(tp) else "")
    return assemble(ctl, P)


def assemble(ctl, P):
    allp = sorted(ctl) == sorted(CONTROLS) and all(v["PASS"] for v in ctl.values())
    res = {"card": CARD, "tier": TIER, "controls": ctl, "controls_all_pass": bool(allp)}
    res["payload" if allp else "withheld"] = P
    return res


# ============================================================================================ output
def fmt_iv(iv):
    return "[%.4f, %.4f]" % (iv[0], iv[1]) if iv else "None"


def print_sections(P, name):
    L = ["rows %d (outside rows %d, rows with |el + 2| > 8: %d); %d evaluation clips; %d draws, RandomState(%d), %s" % (P["n_rows"], P["n_outside"], P["n_el_off8"], P["n_clips"], P["n_boot"], P["seed"], P["interval_level"]),
         "fixed window (az 0, el -2): S4 %.4f S8 %.4f" % (P["fixed"]["S4"], P["fixed"]["S8"])]
    for k in FITS:
        s = P["fits"][k]
        iv = P["intervals"].get(k) or P["intervals"].get("perm_vs_s0")
        what = "S4 - S4_fixed" if k != "perm" else "S4_perm - S4_seed0"
        L.append("fit %s: S4 %.4f S8 %.4f S4_fixed %.4f S8_fixed %.4f S8_out %s; %s %.4f interval %s" % (k, s["S4"], s["S8"], P["fixed"]["S4"], P["fixed"]["S8"], ("%.4f" % s["S8_out"]) if s["S8_out"] is not None else "None", what, iv["diff"], fmt_iv(iv["iv"])))
    for k in ("fixed",) + FITS:
        s = P["fixed"] if k == "fixed" else P["fits"][k]
        L.append("no-name %s: median d_az %.3f d_el %.3f; p90 d_az %.3f d_el %.3f" % (k, s["med_d_az"], s["med_d_el"], s["p90_d_az"], s["p90_d_el"]))
        for key, lab in (("by_gap", "gap"), ("by_az", "|az|")):
            L.append("no-name %s by %s: %s" % (k, lab, "; ".join("%s n %d S4 %s S8 %s" % (b["bin"], b["n"], "%.3f" % b["S4"] if b["S4"] is not None else "-", "%.3f" % b["S8"] if b["S8"] is not None else "-") for b in s[key])))
    L.append("no-name label facts: rows with |az| > 16 (outside both windows): %d; rows whose target is clipped beyond reach of d_az <= 8 (|az| > 40): %d; of d_el <= 8 (el outside -26 to 22): %d" % (P["n_az_beyond_window"], P["n_az_beyond_target_clip"], P["n_el_beyond_target_clip"]))
    L.append(FAMILIES)
    L.append("name: " + name)
    return L


def run_names(a):
    cj = read_json(a.controls)
    if cj.get("card") != CARD or cj.get("controls_all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != sorted(CONTROLS) \
            or any(v.get("PASS") is not True for v in cj.get("controls", {}).values()):
        print("REFUSED: the controls file is not all-pass with k0 to k6 present: no name is written")
        print("ZZ READ-LOOKAT names rc 3 controls_all_pass False")
        return 3
    P = cj["payload"]
    n1, n2, agree = derive_name(P)
    if not agree:
        print("STOP: the two derivations of the name disagree: %s | %s" % (n1, n2))
        print("ZZ READ-LOOKAT names rc 4 controls_all_pass True")
        return 4
    logp = a.log or (os.path.splitext(a.out)[0] + "_names.log")
    if os.path.exists(a.out) or os.path.exists(logp):
        print("REFUSED: %s or %s exists" % (a.out, logp))
        return 2
    res = {"card": CARD, "tier": TIER, "name": n1, "second_derivation_agrees": True, "controls_file": os.path.abspath(a.controls), "controls_sha256": sha_file(a.controls), "values": P}
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    lines = print_sections(P, n1) + ["ZZ READ-LOOKAT names rc 0 controls_all_pass True"]
    with io.open(logp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    for ln in lines:
        print(ln)
    return 0


# ============================================================================================ selftest world
def synth_world(kind, seed=1, nc=30, nfr=40):
    """synthetic evaluation world. kind in good | centres | bad_seed1 | same. -> (J, sel, box, preds, expected name)"""
    rng = np.random.RandomState(seed)
    nrow = nc * nfr
    J = {"clip": np.asarray(["c%02d" % i for i in range(nc)]), "n": np.full(nc, nfr), "off": np.arange(nc) * nfr, "lead_gap": rng.uniform(0, 70, nrow), "lead_valid": (rng.rand(nrow) < 0.9).astype(np.int8)}
    far = rng.rand(nrow) < 0.15
    az = np.where(far, rng.choice([-1, 1], nrow) * rng.uniform(10, 25, nrow), rng.normal(0, 3, nrow))
    el = -2 + rng.normal(0, 2.5, nrow)
    box = {"has": np.ones(nrow, bool), "az": az, "el": el}
    fr = np.tile(np.arange(nfr), nc)
    hist = fr >= 4
    gf = {"clip_index": np.repeat(np.arange(nc), nfr), "frame": fr, "gap": np.where(hist, 10.0, np.nan), "has_out": hist, "clips": J["clip"]}
    sel = row_set(J, [gf] * 5)

    def good(s):
        r = np.random.RandomState(s)
        return az + r.normal(0, 1.0, nrow), el + r.normal(0, 1.0, nrow)

    def centres(s):
        a, e = good(s)
        return np.where(np.abs(az) > 8, np.clip(az, -3, 3), a), e
    preds = {}
    for i, k in enumerate(("s0", "s1", "s2")):
        a, e = {"good": good, "centres": centres, "bad_seed1": good, "same": lambda s: (np.zeros(nrow), np.full(nrow, -2.0))}[kind](10 + i)
        if kind == "bad_seed1" and k == "s1":
            a, e = np.full(nrow, 20.0), np.full(nrow, -2.0)
        preds[k] = {"az": np.where(hist, a, np.nan), "el": np.where(hist, e, np.nan)}
    r = np.random.RandomState(99)
    preds["perm"] = {"az": np.where(hist, r.normal(0, 8, nrow), np.nan), "el": np.where(hist, -2 + r.normal(0, 8, nrow), np.nan)}
    exp = {"good": "LOCATES", "centres": "CENTRES-ONLY", "bad_seed1": "DOES-NOT-LOCATE", "same": "NOT-SEPARATED"}[kind]
    return J, sel, box, preds, PRE + "-" + exp


def clean_facts(ex=EXPECT):
    n = ex["fit"] + ex["sel"] + ex["eval"]
    k0 = {"train_list_sha": "a", "rec99_train_list_sha": "a", "eval_list_sha": "b", "rec99_eval_list_sha": "b", "n_fit": ex["fit"], "n_sel": ex["sel"], "n_eval": ex["eval"], "n_overlap": 0,
          "box_vs_tok": {"c%d" % i: (5, 5) for i in range(n)}, "records_read_list_clips": True}
    k1 = {"missing": [], "size": [], "frames": [], "sha": [], "n_checked": n, "sha_checked": True, "manifest_error": None}
    rec = {"card": CARD, "seed": 0, "permute_targets": False, "hparams": dict(HP_CARD), "hparams_equal_card": True, "head_pt_sha256": "h", "n_samples": {"fit": 10, "sel": 2},
           "history": [{}] * 40, "clips_read": {"fit": ["x"] * ex["fit"], "sel": ["y"] * ex["sel"]}, "tool_sha256": "t", "crop_fit_sha256": CROP_FIT_SHA16 + "0" * 48}
    recs = {k: dict(rec, seed={"s0": 0, "s1": 1, "s2": 2, "perm": 0}[k], permute_targets=(k == "perm")) for k in FITS}
    ident = {"control": "k5", "abs_diff": 2e-6, "n_sel_samples": 7, "n_sel_samples_record": 7, "head_pt_sha256": "z", "head_pt_sha256_record": "z", "crop_fit_sha256": CROP_FIT_SHA16 + "0" * 48, "PASS": True}
    return k0, k1, recs, {k: "h" for k in FITS}, ident, "== 55 passed in 3.2s ==\n"


def selftest():
    ok = True
    for kind in ("good", "centres", "bad_seed1", "same"):
        J, sel, box, preds, want = synth_world(kind)
        P = compute_values(J, sel, box, preds, 300)
        n1, n2, agree = derive_name(P)
        good = n1 == want and agree
        print("world %s -> %s (%s)" % (kind, n1, "ok" if good else "WRONG, expected %s, second %s" % (want, n2)))
        ok &= good
    k0, k1, recs, heads, ident, txt = clean_facts()
    ok &= ctl_k0(k0)["PASS"] and ctl_k1(k1)["PASS"] and ctl_k2(recs, heads)["PASS"] and ctl_k5(ident)["PASS"] and ctl_k6(txt)["PASS"]
    J, sel, box, preds, _ = synth_world("good")
    P = compute_values(J, sel, box, preds, 300)
    ok &= ctl_k3(P)["PASS"]
    faults = {
        "k0": [ctl_k0(dict(k0, train_list_sha="q")), ctl_k0(dict(k0, n_sel=49)), ctl_k0(dict(k0, box_vs_tok=dict(k0["box_vs_tok"], c3=(5, 6)))), ctl_k0(dict(k0, box_vs_tok=dict(k0["box_vs_tok"], c3=None))), ctl_k0(dict(k0, records_read_list_clips=False))],
        "k1": [ctl_k1(dict(k1, sha=["c1"])), ctl_k1(dict(k1, missing=["c1"])), ctl_k1(dict(k1, sha_checked=False)), ctl_k1(dict(k1, n_checked=640))],
        "k2": [ctl_k2(dict(recs, perm=dict(recs["perm"], permute_targets=False)), heads), ctl_k2(dict(recs, s1=dict(recs["s1"], permute_targets=True)), heads), ctl_k2(recs, dict(heads, s2="x")),
               ctl_k2(dict(recs, s0=dict(recs["s0"], hparams_equal_card=False)), heads), ctl_k2(dict(recs, s0=dict(recs["s0"], crop_fit_sha256="0" * 64)), heads), ctl_k2({k: v for k, v in recs.items() if k != "s2"}, heads)],
        "k3": [ctl_k3(dict(P, intervals=dict(P["intervals"], perm_vs_s0={"diff": 0.0, "iv": [-0.1, 0.1]}))), ctl_k3({"intervals": {}})],
        "k4": [ctl_k4(P)],
        "k5": [ctl_k5(dict(ident, abs_diff=1e-3)), ctl_k5(dict(ident, n_sel_samples=8)), ctl_k5({})],
        "k6": [ctl_k6("== 3 passed, 1 failed in 1s =="), ctl_k6(""), ctl_k6("1 error in 0.1s"), ctl_k6("0 passed")]}
    for k, rs in faults.items():
        red = all(not r["PASS"] for r in rs)
        print("fault %s: %d planted, all red: %s" % (k, len(rs), red))
        ok &= red
    ex = {"fit": 450, "sel": 50, "eval": 141, "rows": P["n_rows"], "outside": P["n_outside"]}
    ok &= ctl_k4(P, ex)["PASS"] and not ctl_k4(P, dict(ex, rows=ex["rows"] + 1))["PASS"] and not ctl_k4(P, dict(ex, outside=ex["outside"] + 1))["PASS"]
    ok &= not ctl_k4(dict(P, n_has_box=P["n_rows"] - 1), ex)["PASS"] and not ctl_k4(dict(P, n_pred_missing={"s0": 1}), ex)["PASS"]
    closed = (P["n_rows"] - P["n_outside"]) / P["n_rows"]
    ok &= not ctl_k4(dict(P, n_el_off8=0, fixed=dict(P["fixed"], S8=closed + 1e-6)), ex)["PASS"]
    ok &= len(assemble({"k0": {"PASS": True}}, P).get("withheld", {})) > 0 and "payload" not in assemble({"k0": {"PASS": True}}, P)
    print("ZZ SELFTEST-READ-LOOKAT ok: %s" % ok)
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage", choices=("controls", "names"), required=True)
    ap.add_argument("--root", default="/workspace/v8proof/cropread/run1")
    ap.add_argument("--out-root", default="/workspace/v8proof/lookat")
    ap.add_argument("--box-dir", default="/workspace/v8proof/cropread/enc_out/box")
    ap.add_argument("--eval-join", default="/workspace/v8proof/hierrep/pins/join_eval141.npz")
    ap.add_argument("--gapfiles", default=",".join(["/workspace/v8proof/visrule10k/gapfiles/VGAP_s0.npz", "/workspace/v8proof/visrule10k/gapfiles/VGAP_s1.npz",
                                                     "/workspace/v8proof/visseed10k/gapfiles/VGAPF_s2.npz", "/workspace/v8proof/visseed10k/gapfiles/VGAPF_s3.npz", "/workspace/v8proof/visseed10k/gapfiles/VGAPF_s4.npz"]))
    ap.add_argument("--manifest", default="")
    ap.add_argument("--no-sha", action="store_true")
    ap.add_argument("--controls", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    for k in EXPECT:
        ap.add_argument("--expect-" + k, type=int, default=EXPECT[k])
    a = ap.parse_args(argv)
    a.expect = {k: getattr(a, "expect_" + k) for k in EXPECT}
    if a.stage == "controls":
        if os.path.exists(a.out):
            print("REFUSED: %s exists" % a.out)
            return 2
        res = run_controls(a)
        with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(res, indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o)))
        for k in CONTROLS:
            print("%s %s" % (k, "PASS" if (res["controls"].get(k) or {}).get("PASS") else "FAIL"))
        nel = (res["controls"].get("k4") or {}).get("n_el_off8")
        if nel:
            print("k4 note: %d rows have |el + 2| > 8 (S8_fixed is computed as defined)" % nel)
        print("ZZ READ-LOOKAT controls rc 0 controls_all_pass %s" % res["controls_all_pass"])
        return 0
    return run_names(a)


if __name__ == "__main__":
    sys.exit(main())
