#!/usr/bin/env python3
"""THE THREE NUMBERS THAT DECIDE WHETHER QUANTIZATION ACTUALLY WORKED.

Run against the banked (v0, v_hi) join. Zero GPU, stdlib + numpy only.

  (i)   RESIDUAL AFTER QUANTIZATION -- the ego-coupling panel recomputed on
        q(v_hi) beside the raw figures, so the reduction is visible.
  (ii)  THE LOAD-BEARING ONE -- fit v_hi_raw <- (q(v_hi), v0). If the raw value
        is recoverable from the bin plus v0, the quantization bought NOTHING and
        the leak is intact. Reported against two same-breath controls:
          * v_hi_raw <- v0 alone            (must read a KNOWN, non-zero value)
          * v_hi_raw <- v_hi_raw            (must read EXACTLY R^2 = 1)
          * v_hi_raw <- (shuffled q, v0)    (must collapse onto the v0-alone row)
        Every R^2 is OUT-OF-FOLD (5-fold, clip-disjoint): an in-sample R^2 on a
        per-bin regressor is optimistic by construction and would overstate
        recoverability, i.e. would make the quantization look MORE cosmetic than
        it is. In-sample is printed beside it so the gap is visible.
  (iii) BIN OCCUPANCY -- histogram, entropy, max-bin share. One dominant bin =>
        near-constant channel that cannot help. Near-unique bins => the raw value
        in disguise.

PLUS the ladder-selection evidence that PINS `POSTED_LIMIT_STEPS_KMH`, and the
two behavioural checks the raw value cannot pass:
  * CEILING VIOLABILITY -- the raw ego-derived ceiling is violated on 0 clips BY
    CONSTRUCTION; a real posted limit can be exceeded.
  * LOW-SPEED DISTORTION -- snapping UP from a STOPPED ego reports the lowest
    posted limit, which is not the road's limit. Quantified, not hand-waved.

ASCII-only output (cp1252 dev box).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Users/Admin/tanitad-wt/stack")
sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")

from tanitad.refs import max_speed_input as MSI  # noqa: E402

KMH = 3.6
CANDIDATE_LADDERS = {
    "L4_PI_start": (30, 50, 70, 100),
    "L5_PI_plus_130": (30, 50, 70, 100, 130),
    "L8_PINNED": tuple(MSI.POSTED_LIMIT_STEPS_KMH),
    "L10_plus_90_110": (20, 30, 50, 70, 80, 90, 100, 110, 120, 130),
    "L13_every_10": (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130),
}


# --- regression utilities (numpy only; no sklearn dependency) ----------------

def _ols_fit(X, y):
    X1 = np.column_stack([X, np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def _ols_pred(beta, X):
    return np.column_stack([X, np.ones(len(X))]) @ beta


def _perbin_fit(q, v0, y):
    """The STRONGEST small regressor available to an attacker: one OLS line in v0
    per bin (i.e. the full bin x v0 interaction). If the raw value survives THIS,
    it survives a linear reading of the bin."""
    out = {}
    for b in np.unique(q):
        m = q == b
        if m.sum() >= 3:
            out[float(b)] = _ols_fit(v0[m, None], y[m])
        else:                       # too few rows to fit: fall back to the mean
            out[float(b)] = np.array([0.0, float(y[m].mean()) if m.any() else 0.0])
    return out


def _perbin_pred(model, q, v0, fallback):
    p = np.empty(len(q))
    for i, (b, v) in enumerate(zip(q, v0)):
        beta = model.get(float(b))
        p[i] = fallback if beta is None else float(beta[0] * v + beta[1])
    return p


def _score(y, p):
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"r2": 1.0 - ss_res / ss_tot, "rmse_ms": float(np.sqrt(((y - p) ** 2).mean())),
            "mae_ms": float(np.abs(y - p).mean())}


def kfold_r2(fit, pred, y, k=5, seed=0):
    """Clip-disjoint K-fold. Each clip is one row, so a random fold IS
    clip-disjoint -- stated because it would NOT be if rows were windows."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    folds = np.array_split(order, k)
    p = np.empty(len(y))
    for f in folds:
        tr = np.setdiff1d(order, f, assume_unique=False)
        model = fit(tr)
        p[f] = pred(model, f)
    return _score(y, p), p


def run(join_json: Path, out_json: Path) -> dict:
    d = json.load(open(join_json, encoding="utf-8"))
    assert d["units"] == "m_s", f"join file declares units {d['units']!r}, not m_s"
    rows = d["rows"]
    v_hi = np.array([r["v_hi_ms"] for r in rows], float)
    v0 = np.array([r["v0_ms"] for r in rows], float)
    v_lo = np.array([r["v_lo_ms"] for r in rows], float)
    rc = np.array([r["road_class"] or "?" for r in rows])
    n = len(v_hi)
    out: dict = {"units": "m_s", "n_clips": int(n), "source": str(join_json),
                 "pinned_steps_kmh": list(MSI.POSTED_LIMIT_STEPS_KMH)}

    # ---------------------------------------------------------------- ladders
    lad = {}
    for name, kmh in CANDIDATE_LADDERS.items():
        steps = tuple(k / KMH for k in kmh)
        q, over = MSI.quantize_up_array(v_hi, steps)
        vals, cnt = np.unique(q, return_counts=True)
        p = cnt / cnt.sum()
        slack = (q - v_hi) * KMH
        r2 = _score(v_hi, _ols_pred(_ols_fit(np.column_stack([q, v0]), v_hi),
                                    np.column_stack([q, v0])))
        lad[name] = {
            "steps_kmh": list(kmh), "k": len(kmh),
            "entropy_bits": float(-(p * np.log2(p)).sum()),
            "max_bin_share": float(p.max()), "n_bins_used": int(len(vals)),
            "slack_p50_kmh": float(np.median(slack)),
            "slack_p95_kmh": float(np.percentile(slack, 95)),
            "over_ceiling_frac": float(over.mean()),
            "insample_r2_vhi_from_q_v0": r2["r2"],
            "insample_rmse_ms": r2["rmse_ms"],
        }
    out["ladder_selection"] = lad

    # -------------------------------------------------- the PINNED quantizer
    q, over = MSI.quantize_up_array(v_hi)
    out["quantized_over_ceiling"] = {
        "n": int(over.sum()), "frac": float(over.mean()),
        "note": ("clips whose realised v_hi EXCEEDS the top posted limit "
                 "(130 km/h). The raw ego-derived ceiling is violated on 0 clips "
                 "BY CONSTRUCTION; this is the property a real limit has.")}

    # ------------------------------------------- (i) residual after quantize
    def coupling(x, tag):
        dv = x - v0
        return {"tag": tag, "n": int(n),
                "corr_with_v0": float(np.corrcoef(x, v0)[0, 1]),
                "minus_v0_mean_ms": float(dv.mean()),
                "minus_v0_median_ms": float(np.median(dv)),
                "minus_v0_p95_ms": float(np.percentile(dv, 95)),
                "frac_differs_gt_1ms": float((np.abs(dv) > 1.0).mean()),
                "frac_below_v0": float((x < v0).mean())}

    out["i_ego_coupling"] = {"raw": coupling(v_hi, "v_hi RAW"),
                             "quantized": coupling(q, "quantize_up(v_hi)")}

    # ------------------------------------------------- (ii) RECOVERABILITY
    def _lin(cols):
        def fit(tr):
            return _ols_fit(np.column_stack([c[tr] for c in cols]), v_hi[tr])
        def pred(model, te):
            return _ols_pred(model, np.column_stack([c[te] for c in cols]))
        return fit, pred

    def _perbin(qq):
        def fit(tr):
            return (_perbin_fit(qq[tr], v0[tr], v_hi[tr]), float(v_hi[tr].mean()))
        def pred(model, te):
            m, fb = model
            return _perbin_pred(m, qq[te], v0[te], fb)
        return fit, pred

    rng = np.random.default_rng(1234)
    q_shuf = q[rng.permutation(n)]

    arms = {
        "CONTROL_vhi_from_vhi_must_be_1": _lin([v_hi]),
        "CONTROL_vhi_from_v0_alone": _lin([v0]),
        "CONTROL_vhi_from_shuffled_q_and_v0": _lin([q_shuf, v0]),
        "vhi_from_q_alone": _lin([q]),
        "vhi_from_q_and_v0_linear": _lin([q, v0]),
        "vhi_from_q_and_v0_PERBIN": _perbin(q),
        "REFERENCE_vhi_from_RAW_and_v0": _lin([v_hi, v0]),
    }
    rec = {}
    for name, (fit, pred) in arms.items():
        oof, _ = kfold_r2(fit, pred, v_hi, k=5, seed=0)
        ins = _score(v_hi, pred(fit(np.arange(n)), np.arange(n)))
        rec[name] = {"oof": oof, "insample": ins}
    out["ii_recoverability"] = rec
    base = rec["CONTROL_vhi_from_v0_alone"]["oof"]
    best = max(rec["vhi_from_q_and_v0_linear"]["oof"]["r2"],
               rec["vhi_from_q_and_v0_PERBIN"]["oof"]["r2"])
    best_rmse = min(rec["vhi_from_q_and_v0_linear"]["oof"]["rmse_ms"],
                    rec["vhi_from_q_and_v0_PERBIN"]["oof"]["rmse_ms"])
    out["ii_headline"] = {
        "r2_v0_alone": base["r2"], "rmse_v0_alone_ms": base["rmse_ms"],
        "r2_bin_plus_v0_best": best, "rmse_bin_plus_v0_best_ms": best_rmse,
        "delta_r2_the_future_info_the_bin_adds": best - base["r2"],
        "rmse_reduction_ms": base["rmse_ms"] - best_rmse,
        "unrecovered_variance_share": 1.0 - best,
        "reading": ("1 - r2_bin_plus_v0 is the share of the RAW future value the "
                    "bin + ego does NOT reveal (quantization NOT cosmetic if it is "
                    "materially above 0). delta_r2 is the FUTURE information the "
                    "bin adds beyond the ego present -- the residual leak."),
    }

    # can the BIN itself be read off v0? (if yes, the channel adds nothing at
    # deployment, because the model already knows its own speed)
    bins_sorted = np.unique(q)
    b_idx = np.searchsorted(bins_sorted, q)
    def _bin_fit(tr):
        # nearest-centroid in v0 per bin -- the simplest honest classifier
        cen = {}
        for b in np.unique(b_idx[tr]):
            cen[int(b)] = float(v0[tr][b_idx[tr] == b].mean())
        maj = int(np.bincount(b_idx[tr]).argmax())
        return cen, maj
    def _bin_pred(model, te):
        cen, maj = model
        keys = np.array(sorted(cen)); vals = np.array([cen[k] for k in keys])
        if len(keys) == 0:
            return np.full(len(te), maj)
        return keys[np.abs(v0[te][:, None] - vals[None, :]).argmin(axis=1)]
    rngf = np.random.default_rng(0)
    order = rngf.permutation(n)
    pb = np.empty(n, int)
    for f in np.array_split(order, 5):
        tr = np.setdiff1d(order, f)
        pb[f] = _bin_pred(_bin_fit(tr), f)
    maj_share = float(np.bincount(b_idx).max() / n)
    out["ii_bin_from_v0"] = {
        "oof_accuracy": float((pb == b_idx).mean()),
        "majority_class_baseline": maj_share,
        "note": ("if the bin is readable off the ego's own v0, the channel tells "
                 "the model nothing it does not already have.")}

    # -------------------------------------------------- (iii) bin occupancy
    vals, cnt = np.unique(q, return_counts=True)
    p = cnt / cnt.sum()
    out["iii_occupancy"] = {
        "bins_ms": [round(float(v), 4) for v in vals],
        "bins_kmh": [round(float(v) * KMH) for v in vals],
        "counts": [int(c) for c in cnt],
        "shares": [round(float(x), 5) for x in p],
        "entropy_bits": float(-(p * np.log2(p)).sum()),
        "max_entropy_bits": float(np.log2(len(MSI.POSTED_LIMIT_STEPS_MS))),
        "max_bin_share": float(p.max()),
        "max_bin_kmh": int(round(float(vals[p.argmax()]) * KMH)),
        "n_bins_used": int(len(vals)),
        "n_bins_available": len(MSI.POSTED_LIMIT_STEPS_MS)}

    # ------------------------------- the low-speed distortion, quantified
    stopped = v_lo <= 0.5
    out["low_speed_distortion"] = {
        "frac_clips_ceiling_le_30kmh": float((q <= 30 / KMH + 1e-9).mean()),
        "frac_clips_with_a_stop_in_band": float(stopped.mean()),
        "of_those_frac_ceiling_le_30kmh":
            float((q[stopped] <= 30 / KMH + 1e-9).mean()) if stopped.any() else None,
        "by_road_class": {
            k: {"n": int((rc == k).sum()),
                "frac_ceiling_le_30kmh":
                    float((q[rc == k] <= 30 / KMH + 1e-9).mean()),
                "median_ceiling_kmh": float(np.median(q[rc == k]) * KMH)}
            for k in sorted(set(rc.tolist()))},
        "note": ("snapping UP from a STOPPED or crawling ego reports the LOWEST "
                 "posted limit, which is a property of the traffic, not of the "
                 "road. A deployed map service would return 50 km/h for an urban "
                 "clip whose ego never exceeded 5 km/h. This is a train/deploy "
                 "mismatch in the OPPOSITE direction to the optimism the "
                 "quantization fixes, and it is the largest single caveat on "
                 "this channel.")}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _p(o):
    print("=" * 78)
    print("LADDER SELECTION (the evidence that PINS the step set)   n=%d" % o["n_clips"])
    print("%-18s %3s %8s %8s %9s %9s %8s %8s" %
          ("ladder", "k", "H(bits)", "maxbin", "slack50", "slack95", "over%", "R2(q,v0)"))
    for k, v in o["ladder_selection"].items():
        print("%-18s %3d %8.3f %8.3f %8.1fk %8.1fk %7.2f%% %8.4f" %
              (k, v["k"], v["entropy_bits"], v["max_bin_share"],
               v["slack_p50_kmh"], v["slack_p95_kmh"],
               100 * v["over_ceiling_frac"], v["insample_r2_vhi_from_q_v0"]))
    print()
    print("=" * 78)
    print("(i) EGO COUPLING -- raw vs quantized")
    for k in ("raw", "quantized"):
        c = o["i_ego_coupling"][k]
        print("  %-20s corr(v0)=%.4f  -v0 mean=%+.3f med=%+.3f p95=%+.3f  "
              ">1m/s=%.1f%%  below v0=%.1f%%"
              % (c["tag"], c["corr_with_v0"], c["minus_v0_mean_ms"],
                 c["minus_v0_median_ms"], c["minus_v0_p95_ms"],
                 100 * c["frac_differs_gt_1ms"], 100 * c["frac_below_v0"]))
    print()
    print("=" * 78)
    print("(ii) RECOVERABILITY -- v_hi_raw <- ?   (5-fold OUT-OF-FOLD)")
    for k, v in o["ii_recoverability"].items():
        print("  %-38s R2=%7.4f rmse=%6.3f m/s   (in-sample R2=%7.4f)"
              % (k, v["oof"]["r2"], v["oof"]["rmse_ms"], v["insample"]["r2"]))
    h = o["ii_headline"]
    print("  ---- HEADLINE ----")
    print("  v0 alone            R2=%.4f  rmse=%.3f m/s" % (h["r2_v0_alone"], h["rmse_v0_alone_ms"]))
    print("  bin + v0 (best)     R2=%.4f  rmse=%.3f m/s" % (h["r2_bin_plus_v0_best"], h["rmse_bin_plus_v0_best_ms"]))
    print("  delta R2 (leak)     %+.4f      rmse cut %.3f m/s" % (h["delta_r2_the_future_info_the_bin_adds"], h["rmse_reduction_ms"]))
    print("  UNRECOVERED share   %.4f" % h["unrecovered_variance_share"])
    b = o["ii_bin_from_v0"]
    print("  bin readable off v0: oof acc %.4f vs majority %.4f"
          % (b["oof_accuracy"], b["majority_class_baseline"]))
    print()
    print("=" * 78)
    print("(iii) BIN OCCUPANCY  H=%.3f / %.3f bits max   maxbin=%.3f at %d km/h   %d/%d bins used"
          % (o["iii_occupancy"]["entropy_bits"], o["iii_occupancy"]["max_entropy_bits"],
             o["iii_occupancy"]["max_bin_share"], o["iii_occupancy"]["max_bin_kmh"],
             o["iii_occupancy"]["n_bins_used"], o["iii_occupancy"]["n_bins_available"]))
    for kmh, c, s in zip(o["iii_occupancy"]["bins_kmh"], o["iii_occupancy"]["counts"],
                         o["iii_occupancy"]["shares"]):
        print("    %3d km/h  n=%5d  %6.2f%%  %s" % (kmh, c, 100 * s, "#" * int(60 * s)))
    print()
    print("OVER-CEILING (quantized): n=%d  %.2f%%   [raw: 0 by construction]"
          % (o["quantized_over_ceiling"]["n"], 100 * o["quantized_over_ceiling"]["frac"]))
    ls = o["low_speed_distortion"]
    print("LOW-SPEED DISTORTION: %.1f%% of clips get a ceiling <= 30 km/h; "
          "%.1f%% have a stop in band" % (100 * ls["frac_clips_ceiling_le_30kmh"],
                                          100 * ls["frac_clips_with_a_stop_in_band"]))
    for k, v in ls["by_road_class"].items():
        print("    %-14s n=%4d  ceiling<=30km/h on %5.1f%%  median ceiling %.0f km/h"
              % (k, v["n"], 100 * v["frac_ceiling_le_30kmh"], v["median_ceiling_kmh"]))


if __name__ == "__main__":
    _p(run(Path(sys.argv[1]), Path(sys.argv[2])))
