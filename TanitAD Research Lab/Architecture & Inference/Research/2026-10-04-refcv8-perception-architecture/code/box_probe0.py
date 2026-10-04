"""WP-D PROBE-0 (zero GPU): what the refcv7 box heads' OWN eval packs say about the levers, before any training.

Inputs (NOT in the repo; the diagnostics package's Thor outputs, md5-verified on copy -- see raw/box_probe0.json
"inputs"): `<packs_dir>/eval_{s5000,s15000,s20000,s30000,final}.packs.pkl` = the trainer's own `window_packs` for
box3d and agent on EVAL-DIAG (1,061 labelled windows / 137 episodes of eval139), banked by
`…/2026-10-04-refcv7-map-box-diagnostics/code/run_diag.py`; and the run checkpoint
`D:/refcv7_eval_kit/ckpt/ckpt_50400.pt` for the box3d head's learned reference points.

Tier: OPEN-LOOP PERCEPTION DIAGNOSTIC (teacher-forced forward, no planner). Estimator: episode-cluster bootstrap over
the 137 episodes (B = 1000, percentile 95 %), PAIRED where two arms share windows. It answers "another draw of
EPISODES" only -- one training seed, no replicate (H-ESTIM-SEED-1).

What it measures (each is a question a design lever depends on):
  Q1 the box AP curve on the decision-grade 1,061 windows at 5k/15k/20k/30k/50.4k (was the head still improving?);
  Q2 duplicates at the TRAIN P = R gate: boxes per detected object, split by whether the object has an IGNORE row
     within 2 m (the training's presence-exemption radius) and by range;
  Q3 score vs localisation: among the slots within 2 m of a POSITIVE, is the top-scored one the nearest?
  Q4 ceilings with the SAME boxes: (a) admissible centre-distance NMS; (b) ORACLE within-object quality re-ranking
     (score x exp(-d^2 / 2 s^2), d = distance to the nearest GT, applied ONLY to slots that have a GT within 2 m, so
     it can re-order near-duplicates but can never reject a far false positive) -- a CEILING for a quality-aware
     presence target, inadmissible as a result;
  Q5 localisation error split LONGITUDINAL / LATERAL by range (greedy TPs at 4 m, slots above the P = R gate), next
     to the ANALYTIC depth-per-image-row of raw/geom_bound.json;
  Q6 the box3d head's learned reference points: reach (anchors within the +-4 m tanh offset of each POSITIVE),
     offset saturation, and which anchors ever produce a TP.

Controls (must read known values; the script STOPS if one fails):
  C1 the vectorised greedy matcher == `detection_metrics.greedy_rows` (the trainer's own) on 60 windows x 2 thresholds;
  C2 AP@2 m (class-agnostic, all ranges) at 50,400 == the diagnostics' T0 (box3d 0.2483, agent 0.1314) to 1e-4;
  C3 a paired bootstrap of an arm against ITSELF reads exactly 0;
  C4 the oracle with s -> infinity (q == 1) reproduces T0 AP exactly; NMS with radius 0 reproduces T0 AP exactly.
Run: PYTHONPATH=<tip>/stack python box_probe0.py --packs <dir> --ckpt <ckpt_50400.pt>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
CKPTS = [("5000", "eval_s5000"), ("15000", "eval_s15000"), ("20000", "eval_s20000"), ("30000", "eval_s30000"),
         ("50400", "eval_final")]
THRS = [0.5, 1.0, 2.0, 4.0]
BANDS = [("0_20", 0.0, 20.0), ("20_40", 20.0, 40.0), ("40_60", 40.0, 60.0)]
PR_GATE = {"box3d": 0.2567, "agent": 0.2307}     # TRAIN-DIAG P = R gates, diagnostics RESULT sec. 2.3 (MEASURED)
T0_AP2 = {"box3d": 0.2483, "agent": 0.1314}      # diagnostics B_box.json T0 AP@2 m (MEASURED)
B_BOOT = 1000


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def band_of(x):
    """== detection_metrics._band_of (the last band is closed at 60 m)."""
    for name, lo, hi in BANDS:
        if lo <= x < hi or (name == BANDS[-1][0] and x == hi):
            return name
    return None


# --------------------------------------------------------------------------------------------------------- #
# greedy matching (vectorised; semantics of box3d_head.box3d_match_rows with use_z False, score presence)      #
# --------------------------------------------------------------------------------------------------------- #
def greedy(pk, thr, score=None, keep=None):
    """-> list of (conf, kind, band, slot, g) exactly like detection_metrics.greedy_rows(pk, thr) when score/keep
    are None. `score` overrides the ranking score (oracle / transforms); `keep` restricts the slots (NMS)."""
    p = sig(pk["logit"]) if score is None else np.asarray(score, np.float64)
    sel = np.arange(len(p)) if keep is None else np.nonzero(keep)[0]
    if sel.size == 0:
        return []
    pos, ign = pk["pos"], pk["ign"]
    gi = np.nonzero(pos | ign)[0]
    order = sel[np.argsort(-p[sel], kind="stable")]
    rows = []
    if gi.size == 0:
        return [(float(p[s]), 0, band_of(float(pk["xy"][s, 0])), int(s), -1) for s in order]
    gxy = pk["gt_xy"][gi].astype(np.float64)
    pxy = pk["xy"].astype(np.float64)
    D = np.sqrt(((pxy[order][:, None, :] - gxy[None, :, :]) ** 2).sum(-1))      # [n_sel, A]
    taken = np.zeros(gi.size, bool)
    match = np.full(order.size, -1, np.int64)
    # a slot with NO target within thr can never take one, whatever the order -> only candidates loop
    for r in np.nonzero(D.min(1) <= thr)[0]:
        d = np.where(taken, np.inf, D[r])
        j = int(np.argmin(d))
        if d[j] <= thr:
            taken[j] = True
            match[r] = j
    for r, s in enumerate(order):
        j = match[r]
        if j >= 0:
            g = int(gi[j])
            if pos[g]:
                rows.append((float(p[s]), 1, band_of(float(pk["gt_xy"][g, 0])), int(s), g))
            else:
                rows.append((float(p[s]), -1, None, int(s), g))
        else:
            rows.append((float(p[s]), 0, band_of(float(pk["xy"][s, 0])), int(s), -1))
    return rows


# --------------------------------------------------------------------------------------------------------- #
# AP with episode weights (pre-sorted once; a bootstrap resample is a weighted cumsum)                         #
# --------------------------------------------------------------------------------------------------------- #
class APTable:
    def __init__(self, rows_by_win, npos_by_win, ep_of_win, band=None):
        conf, hit, ep = [], [], []
        for wi, rows in enumerate(rows_by_win):
            for (c, k, b, _s, _g) in rows:
                if k < 0:
                    continue
                if band is not None:
                    if k == 1 and b != band:
                        continue          # a TP of another band is not a row of this band's list
                    if k == 0 and b != band:
                        continue          # an FP is assigned to the band of ITS OWN centre (detection_metrics)
                conf.append(c)
                hit.append(k)
                ep.append(ep_of_win[wi])
        conf = np.asarray(conf, np.float64)
        order = np.argsort(-conf, kind="stable")
        self.hit = np.asarray(hit, np.float64)[order]
        self.ep = np.asarray(ep, np.int64)[order]
        self.npos_ep = np.zeros(int(max(ep_of_win)) + 1, np.float64)
        for wi, n in enumerate(npos_by_win):
            self.npos_ep[ep_of_win[wi]] += n

    def ap(self, w_ep=None):
        w = np.ones(self.hit.size) if w_ep is None else w_ep[self.ep]
        n_gt = float(self.npos_ep.sum() if w_ep is None else (self.npos_ep * w_ep).sum())
        if n_gt <= 0 or self.hit.size == 0:
            return float("nan")
        tp = np.cumsum(w * self.hit)
        fp = np.cumsum(w * (1.0 - self.hit))
        rec = tp / n_gt
        prec = tp / np.maximum(tp + fp, 1e-12)
        return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def npos_band(pk, band):
    if band is None:
        return int(pk["pos"].sum())
    lo, hi = [(a, b) for (n, a, b) in BANDS if n == band][0]
    x = pk["gt_xy"][:, 0]
    return int((pk["pos"] & (x >= lo) & (x < hi)).sum())


def boot_weights(n_ep, B, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_ep, size=(B, n_ep))
    W = np.zeros((B, n_ep))
    for b in range(B):
        W[b] = np.bincount(idx[b], minlength=n_ep)
    return W


def ci(v):
    v = np.asarray(v, np.float64)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [None, None]


# --------------------------------------------------------------------------------------------------------- #
def nms_keep(pk, radius):
    p = sig(pk["logit"])
    order = np.argsort(-p, kind="stable")
    xy = pk["xy"].astype(np.float64)
    keep = np.zeros(len(p), bool)
    if radius <= 0:
        keep[:] = True
        return keep
    kept = []
    for s in order:
        if kept:
            d = np.sqrt(((xy[kept] - xy[s]) ** 2).sum(-1))
            if (d < radius).any():
                continue
        kept.append(s)
        keep[s] = True
    return keep


def oracle_score(pk, sigma, radius=2.0):
    """ORACLE (uses GT): within-object quality re-ranking. Slots with a GT (POS or IGN) within `radius` get
    p * exp(-d^2 / 2 sigma^2); others keep p. sigma = inf -> p (control C4)."""
    p = sig(pk["logit"])
    gi = np.nonzero(pk["pos"] | pk["ign"])[0]
    if gi.size == 0 or not np.isfinite(sigma):
        return p
    d = np.sqrt(((pk["xy"][:, None, :].astype(np.float64) - pk["gt_xy"][gi][None].astype(np.float64)) ** 2).sum(-1))
    dn = d.min(1)
    q = np.exp(-dn ** 2 / (2 * sigma ** 2))
    return np.where(dn <= radius, p * q, p)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--packs", required=True)
    ap_.add_argument("--ckpt", required=True)
    ap_.add_argument("--boot", type=int, default=B_BOOT)
    a = ap_.parse_args()
    t0 = time.time()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tanitad.eval import detection_metrics as det        # the trainer's own (C1)

    packs, md5 = {}, {}
    for step, tag in CKPTS:
        f = Path(a.packs) / f"{tag}.packs.pkl"
        md5[tag] = hashlib.md5(f.read_bytes()).hexdigest()
        with open(f, "rb") as fh:
            packs[step] = pickle.load(fh)
    out = {"inputs": {"packs_md5": md5, "ckpt": Path(a.ckpt).name, "n_boot": a.boot,
                      "evidence": "MEASURED (diagnostics packs, refcv7-r101-s0); oracle rows are CEILINGS"},
           "controls": {}, "Q1_ap_curve": {}, "Q2_duplicates": {}, "Q3_score_vs_loc": {}, "Q4_ceilings": {},
           "Q5_loc_by_range": {}, "Q6_anchors": {}}

    # episode index (sha12 -> int) shared by every checkpoint (same windows)
    ref = packs["50400"]["box3d"]
    eps = sorted({pk["sha12"] for pk in ref})
    eix = {e: i for i, e in enumerate(eps)}
    for step in packs:
        for hd in ("box3d", "agent"):
            assert [pk["sha12"] for pk in packs[step][hd]] == [pk["sha12"] for pk in ref], "window order differs"
    ep_of_win = [eix[pk["sha12"]] for pk in ref]
    W = boot_weights(len(eps), a.boot)

    # ---- C1 ----------------------------------------------------------------------------------------------- #
    n_c1 = 0
    for hd in ("box3d", "agent"):
        for pk in packs["50400"][hd][:30]:
            for thr in (0.5, 2.0):
                mine = greedy(pk, thr)
                theirs = det.greedy_rows(pk, thr)
                # kind / slot / gt EXACT, in the same order; conf to 1e-6 (theirs is torch float32 sigmoid,
                # mine float64 -- the 7th decimal differs, MEASURED on the first run of this control)
                assert [(k, s, g) for (c, k, b, s, g) in mine] == [(k, s, g) for (c, k, b, s, g) in theirs],                     f"C1 FAIL {hd} thr {thr}"
                assert max(abs(x[0] - y[0]) for x, y in zip(mine, theirs)) < 1e-6, f"C1 conf FAIL {hd} {thr}"
                n_c1 += 1
    out["controls"]["C1_greedy_equals_trainer"] = {"windows_x_thr": n_c1, "pass": True}

    # ---- Q1 + C2: AP curve with paired bootstrap ------------------------------------------------------------ #
    tables = {}
    for hd in ("box3d", "agent"):
        for step in packs:
            P = packs[step][hd]
            for thr in THRS:
                rows = [greedy(pk, thr) for pk in P]
                for band in [None] + [b[0] for b in BANDS]:
                    tables[(hd, step, thr, band)] = APTable(rows, [npos_band(pk, band) for pk in P], ep_of_win, band)
        ap2 = tables[(hd, "50400", 2.0, None)].ap()
        assert abs(ap2 - T0_AP2[hd]) < 1e-4, f"C2 FAIL {hd} {ap2}"
        out["controls"][f"C2_ap2_{hd}"] = {"measured": ap2, "expected": T0_AP2[hd], "pass": True}
    self_d = [tables[("box3d", "50400", 2.0, None)].ap(w) - tables[("box3d", "50400", 2.0, None)].ap(w)
              for w in W[:50]]
    assert max(abs(x) for x in self_d) == 0.0
    out["controls"]["C3_self_paired_zero"] = {"max_abs": 0.0, "pass": True}
    for hd in ("box3d", "agent"):
        cur = {}
        for thr in THRS:
            for band in [None] + [b[0] for b in BANDS]:
                key = f"ap{thr:g}_{band or 'all'}"
                cur[key] = {s: tables[(hd, s, thr, band)].ap() for s in packs}
                fin = tables[(hd, "50400", thr, band)]
                for base in ("30000", "5000"):
                    bt = tables[(hd, base, thr, band)]
                    d = [fin.ap(w) - bt.ap(w) for w in W]
                    cur[key][f"d_50400_minus_{base}"] = {"point": fin.ap() - bt.ap(), "ci95": ci(d)}
        out["Q1_ap_curve"][hd] = cur

    # ---- Q2 duplicates at the P = R gate -------------------------------------------------------------------- #
    for hd in ("box3d", "agent"):
        g = PR_GATE[hd]
        P = packs["50400"][hd]
        tot = {"tp": 0, "dup": 0, "fp_other": 0, "fp_exempt": 0, "fp": 0}
        strat = {"ign_within_2m": {"tp": 0, "dup": 0}, "no_ign_within_2m": {"tp": 0, "dup": 0}}
        by_band = {b[0]: {"tp": 0, "dup": 0} for b in BANDS}
        for pk in P:
            rows = greedy(pk, 2.0, keep=sig(pk["logit"]) >= g)
            det_g = {gg for (c, k, b, s, gg) in rows if k == 1}
            posi = np.nonzero(pk["pos"])[0]
            igni = np.nonzero(pk["ign"])[0]
            gxy = pk["gt_xy"].astype(np.float64)
            dup_of = {}
            for (c, k, b, s, gg) in rows:
                if k != 0:
                    continue
                tot["fp"] += 1
                if pk["exempt"][s]:
                    tot["fp_exempt"] += 1
                if posi.size:
                    d = np.sqrt(((gxy[posi] - pk["xy"][s].astype(np.float64)) ** 2).sum(-1))
                    j = int(np.argmin(d))
                    if d[j] <= 2.0 and int(posi[j]) in det_g:
                        dup_of[int(posi[j])] = dup_of.get(int(posi[j]), 0) + 1
                        tot["dup"] += 1
                        continue
                tot["fp_other"] += 1
            for gg in det_g:
                tot["tp"] += 1
                has_ign = igni.size and (np.sqrt(((gxy[igni] - gxy[gg]) ** 2).sum(-1)) <= 2.0).any()
                key = "ign_within_2m" if has_ign else "no_ign_within_2m"
                strat[key]["tp"] += 1
                strat[key]["dup"] += dup_of.get(gg, 0)
                bnd = band_of(float(gxy[gg, 0]))
                if bnd:
                    by_band[bnd]["tp"] += 1
                    by_band[bnd]["dup"] += dup_of.get(gg, 0)
        f = lambda d: (d["tp"] + d["dup"]) / d["tp"] if d["tp"] else None
        out["Q2_duplicates"][hd] = {
            "gate": g, **tot, "boxes_per_detected_object": f(tot),
            "dup_share_of_fp": tot["dup"] / tot["fp"] if tot["fp"] else None,
            "exempt_share_of_fp": tot["fp_exempt"] / tot["fp"] if tot["fp"] else None,
            "by_ignore_neighbour": {k: {**v, "boxes_per_detected_object": f(v)} for k, v in strat.items()},
            "by_band": {k: {**v, "boxes_per_detected_object": f(v)} for k, v in by_band.items()}}

    # ---- Q3 score vs localisation among near slots ---------------------------------------------------------- #
    for hd in ("box3d", "agent"):
        n_obj = n_top_is_nearest = 0
        rhos = []
        top_d, best_d = [], []
        for pk in packs["50400"][hd]:
            p = sig(pk["logit"])
            xy = pk["xy"].astype(np.float64)
            for gg in np.nonzero(pk["pos"])[0]:
                d = np.sqrt(((xy - pk["gt_xy"][gg].astype(np.float64)) ** 2).sum(-1))
                near = np.nonzero(d <= 2.0)[0]
                if near.size < 2:
                    continue
                n_obj += 1
                it = near[np.argmax(p[near])]
                ib = near[np.argmin(d[near])]
                n_top_is_nearest += int(it == ib)
                top_d.append(d[it])
                best_d.append(d[ib])
                if near.size >= 3:
                    rp = np.argsort(np.argsort(-p[near]))
                    rd = np.argsort(np.argsort(d[near]))
                    rhos.append(float(np.corrcoef(rp, rd)[0, 1]))
        out["Q3_score_vs_loc"][hd] = {
            "n_objects_with_2plus_slots_within_2m": n_obj,
            "top_scored_is_nearest_frac": n_top_is_nearest / n_obj if n_obj else None,
            "chance_frac_if_random": None,
            "median_dist_of_top_scored_m": float(np.median(top_d)) if top_d else None,
            "median_dist_of_nearest_m": float(np.median(best_d)) if best_d else None,
            "mean_within_object_spearman_score_rank_vs_dist_rank": float(np.mean(rhos)) if rhos else None,
            "n_objects_with_3plus": len(rhos),
            "reading": "spearman +1 = higher score <-> nearer (quality-aligned); 0 = score blind to localisation"}

    # ---- Q4 ceilings with the same boxes --------------------------------------------------------------------- #
    for hd in ("box3d", "agent"):
        P = packs["50400"][hd]
        res = {}
        arms = {"T0": dict(score=None, keep=None)}
        for r in (1.0, 2.0, 3.0):
            arms[f"nms_r{r:g}"] = dict(keep_fn=lambda pk, r=r: nms_keep(pk, r))
        for s in (0.5, 1.0):
            arms[f"ORACLE_q_s{s:g}"] = dict(score_fn=lambda pk, s=s: oracle_score(pk, s))
            arms[f"ORACLE_q_s{s:g}+nms_r2"] = dict(score_fn=lambda pk, s=s: oracle_score(pk, s),
                                                  keep_fn=lambda pk: nms_keep(pk, 2.0))
        # C4
        for thr in (0.5, 2.0):
            a0 = APTable([greedy(pk, thr) for pk in P], [npos_band(pk, None) for pk in P], ep_of_win).ap()
            a1 = APTable([greedy(pk, thr, score=oracle_score(pk, float("inf"))) for pk in P],
                         [npos_band(pk, None) for pk in P], ep_of_win).ap()
            a2 = APTable([greedy(pk, thr, keep=nms_keep(pk, 0.0)) for pk in P],
                         [npos_band(pk, None) for pk in P], ep_of_win).ap()
            assert abs(a0 - a1) < 1e-12 and abs(a0 - a2) < 1e-12, "C4 FAIL"
        out["controls"][f"C4_identity_oracle_and_nms0_{hd}"] = {"pass": True}
        base_tabs = {}
        for name, spec in arms.items():
            res[name] = {}
            for thr in THRS:
                rows = []
                for pk in P:
                    sc = spec["score_fn"](pk) if "score_fn" in spec else None
                    kp = spec["keep_fn"](pk) if "keep_fn" in spec else None
                    rows.append(greedy(pk, thr, score=sc, keep=kp))
                tab = APTable(rows, [npos_band(pk, None) for pk in P], ep_of_win)
                if name == "T0":
                    base_tabs[thr] = tab
                    res[name][f"ap{thr:g}"] = tab.ap()
                else:
                    d = [tab.ap(w) - base_tabs[thr].ap(w) for w in W[:300]]
                    res[name][f"ap{thr:g}"] = tab.ap()
                    res[name][f"d_vs_T0_ap{thr:g}"] = {"point": tab.ap() - base_tabs[thr].ap(), "ci95": ci(d)}
        out["Q4_ceilings"][hd] = res

    # ---- Q5 localisation error split by range ---------------------------------------------------------------- #
    for hd in ("box3d", "agent"):
        g = PR_GATE[hd]
        acc = {b[0]: {"dlon": [], "dlat": []} for b in BANDS}
        for pk in packs["50400"][hd]:
            for (c, k, b, s, gg) in greedy(pk, 4.0, keep=sig(pk["logit"]) >= g):
                if k != 1 or b is None:
                    continue
                d = pk["xy"][s].astype(np.float64) - pk["gt_xy"][gg].astype(np.float64)
                acc[b]["dlon"].append(abs(d[0]))
                acc[b]["dlat"].append(abs(d[1]))
        out["Q5_loc_by_range"][hd] = {
            b: {"n": len(v["dlon"]),
                "median_abs_dlon_m": float(np.median(v["dlon"])) if v["dlon"] else None,
                "median_abs_dlat_m": float(np.median(v["dlat"])) if v["dlat"] else None,
                "p75_abs_dlon_m": float(np.percentile(v["dlon"], 75)) if v["dlon"] else None,
                "p75_abs_dlat_m": float(np.percentile(v["dlat"], 75)) if v["dlat"] else None}
            for b, v in acc.items()}
    gb = PKG / "raw" / "geom_bound.json"
    if gb.exists():
        G = json.loads(gb.read_text(encoding="utf-8"))
        out["Q5_loc_by_range"]["ANALYTIC_depth_per_image_row_m_h1.45"] = {
            r["band"]: r["by_height"]["h1.45"]["ground_depth_per_image_row_m_at_centre"] for r in G["bands"][:3]}
        out["Q5_loc_by_range"]["ANALYTIC_lateral_per_px_m"] = {
            r["band"]: r["by_height"]["h1.45"]["lateral_m_per_px_at_centre"] for r in G["bands"][:3]}

    # ---- Q6 anchors -------------------------------------------------------------------------------------------- #
    import torch
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"]
    anc = sd["_perception.box_refpts.xy"].float().numpy().astype(np.float64)
    del sd
    reach, tp_slots, off_sat, offs = [], set(), 0, []
    reach_band = {b[0]: [] for b in BANDS}
    g = PR_GATE["box3d"]
    for pk in packs["50400"]["box3d"]:
        for gg in np.nonzero(pk["pos"])[0]:
            gxy = pk["gt_xy"][gg].astype(np.float64)
            n = int(((np.abs(anc[:, 0] - gxy[0]) <= 4.0) & (np.abs(anc[:, 1] - gxy[1]) <= 4.0)).sum())
            reach.append(n)
            bnd = band_of(float(gxy[0]))
            if bnd:
                reach_band[bnd].append(n)
        for (c, k, b, s, gg) in greedy(pk, 2.0, keep=sig(pk["logit"]) >= g):
            if k == 1:
                tp_slots.add(s)
                o = pk["xy"][s].astype(np.float64) - anc[s]
                offs.append(np.abs(o))
                off_sat += int((np.abs(o) > 3.6).any())
    offs = np.asarray(offs)
    out["Q6_anchors"] = {
        "n_anchors": int(anc.shape[0]),
        "anchor_abs_y_gt_12m": int((np.abs(anc[:, 1]) > 12).sum()),
        "reach_per_positive": {"median": float(np.median(reach)), "p10": float(np.percentile(reach, 10)),
                               "frac_zero": float(np.mean(np.asarray(reach) == 0)), "n": len(reach),
                               "by_band_median": {k: (float(np.median(v)) if v else None)
                                                  for k, v in reach_band.items()}},
        "tp_at_gate_distinct_slots": len(tp_slots),
        "tp_offsets_median_abs_m": offs.mean(0).tolist() if len(offs) else None,
        "tp_offset_saturated_frac_(>3.6m_on_an_axis)": off_sat / max(len(offs), 1),
        "tp_slot_anchor_abs_y_gt_12m": int(sum(1 for s in tp_slots if abs(anc[s, 1]) > 12)),
        "anchor_displacement_from_seed0_init_m": "see raw/refpts_by_step.json (median 0.031 m at 50,400)"}
    out["wall_s"] = time.time() - t0
    p = PKG / "raw" / "box_probe0.json"
    p.write_text(json.dumps(out, indent=1, allow_nan=True), encoding="utf-8")
    print("wrote", p, f"{out['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
