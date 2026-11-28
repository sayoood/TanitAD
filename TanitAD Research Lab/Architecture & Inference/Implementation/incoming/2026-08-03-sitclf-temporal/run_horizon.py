"""THE ANTICIPATION HORIZON — how far ahead is this label predictable at all?

THE QUESTION THE PI'S ITEM 3 NAMES
----------------------------------
"the head's window is [t-0.7 s, t] while the label's evidence window is [onset, onset+4 s]".
`run_temporal.py` attacks the HISTORY side of that mismatch (H-T1, the window). This attacks the
other side: the LEAD. `y(t) = 1` iff an onset falls in `(t, t + lead_s]`, and `lead_s = 3.0` has been
a frozen constant since the original pre-registration, never varied, never characterised. If the
camera arm's skill collapses with lead, then the ceiling is the HORIZON and not the modality — and
no encoder, window or head fixes that.

⛔ THIS IS A CHARACTERISATION, NOT A RE-SELECTION. `lead_s = 3.0` lives in the parent
PRE_REGISTRATION.md §2.4 and §7 forbids re-sweeping a §2 constant after a held-out number is seen.
Nothing here selects a new `lead_s`, changes a deployed constant, or re-derives any banked label:
**every horizon is reported, none is chosen**, the detectors and their frozen thresholds are
untouched, and the deployed value stays 3.0. What is produced is a curve describing how predictable
the task is as a function of how far ahead it asks — which is a property of the TASK, and is exactly
what "is the horizon the problem?" requires to answer.

PRE-REGISTERED OUTCOMES (both committed before the numbers were read)
  HORIZON-LIMITED  AP-lift over the permuted-feature null RISES materially as the lead shortens
                   (short-lead lift clearly above the 3.0 s lift). => the label asks for more
                   anticipation than the front camera can supply; the lever is the horizon /
                   label definition, and a temporal encoder would not fix it.
  HORIZON-FLAT     lift is roughly constant across 1.0-5.0 s. => the horizon is NOT the binding
                   constraint and the ceiling is elsewhere.
  ⚠️ POWER MOVES WITH THE LEAD. A shorter lead has fewer positive frames; a longer lead has more.
  Positive frames AND positive CLUSTERS are reported per horizon, and the C-POW bar of 40 clusters
  is applied per horizon, because a lift that "improves" only because its denominator shrank is the
  same defect as a recall-only frontier.

The representation is held at the REFERENCE recipe (appearance PCA-16, WIN 8, 129 params/head) so
the only thing moving is the label's horizon. The PCA does not depend on the labels, so one basis
per fold serves every horizon and the arms stay exactly comparable.

usage:
  python run_horizon.py --substrate <npz> --out results_horizon.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from tanitad.data.situations import (anticipation_target,            # noqa: E402
                                     detect_intersection,
                                     detect_lane_change,
                                     detect_roundabout, kinematics)
from tanitad.eval.ap_ci import (ap_episode_cluster_bootstrap,        # noqa: E402
                                ap_lift, average_precision,
                                paired_ap_episode_cluster_bootstrap)
from tanitad.eval.sitclf import (causal_window, clip_runs,           # noqa: E402
                                 cluster_folds, ridge_scores)
from tanitad.eval.sitclf_deploy import precision_recall_at_budget    # noqa: E402

SITS = ("lane_change", "roundabout", "intersection")
# build_substrate.py:64-67 — the SAME caches in the SAME order, so clip k here is clip k there
CACHES = (r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74",
          r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836")
LEADS = (1.0, 2.0, 3.0, 4.0, 5.0)
RANK, WIN = 16, 8                       # the REFERENCE recipe, held fixed
SEL_FRAC = 0.20
LAMBDAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, "replace").decode(enc, "replace"), flush=True)


def events_per_clip(limit_frames):
    """The three situations' events for every clip, in build_substrate.py's order."""
    files = []
    for root in CACHES:
        files += sorted(glob.glob(os.path.join(root, "ep_*.pt")))
    out = []
    for f in files:
        P = np.asarray(torch.load(f, map_location="cpu", weights_only=True,
                                  mmap=True)["poses"]).astype(np.float64)
        K = kinematics(P)
        out.append((int(K["T"]), {
            "lane_change": detect_lane_change(K),
            "roundabout": detect_roundabout(K, bracket=True),
            "intersection": detect_intersection(K, cross=None)[0]}))
    tot = sum(t for t, _ in out)
    if tot != limit_frames:
        raise SystemExit(f"C-FID: rebuilt {tot} frames but substrate has {limit_frames}; "
                         "the clip order or cache set does not match build_substrate.py")
    log(f"events rebuilt for {len(out)} clips, {tot:,} frames — C-FID OK")
    return out


def targets_at(per_clip, lead_s):
    Y, V = [], []
    for T, ev in per_clip:
        y = np.zeros((T, 3), bool)
        v = np.zeros((T, 3), bool)
        for i, s in enumerate(SITS):
            y[:, i], v[:, i] = anticipation_target(T, ev[s], lead_s=lead_s)
        Y.append(y)
        V.append(v)
    return np.concatenate(Y), np.concatenate(V)


def fit_pca(F, rows, r, device="cpu"):
    A = torch.from_numpy(np.asarray(F[rows], dtype=np.float32)).to(device)
    mu = A.mean(0, keepdim=True)
    A = A - mu
    _u, _s, v = torch.svd_lowrank(A, q=r + 16, niter=4)
    del A
    if device != "cpu":
        torch.cuda.empty_cache()
    return mu.cpu().numpy(), v[:, :r].cpu().numpy().astype(np.float32)


def project(F, mu, W):
    img = (np.asarray(F, dtype=np.float32) - mu) @ W
    img /= max(float(np.abs(img).mean()), 1e-6)
    return img


def mean_ap(y, s, v):
    vals = []
    for i in range(y.shape[1]):
        m = v[:, i]
        if m.sum() < 50 or y[m, i].sum() < 5:
            continue
        vals.append(average_precision(y[m, i], s[m, i]))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--substrate", required=True)
    ap_.add_argument("--out", default="results_horizon.json")
    ap_.add_argument("--n-boot", type=int, default=2000)
    ap_.add_argument("--device", default=None)
    a = ap_.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    z = np.load(a.substrate)
    F = z["F"]
    cc = z["clip_cluster"]
    st, en = clip_runs(cc)
    folds = cluster_folds(cc, 2, seed=0)
    log(f"substrate {F.shape[0]:,} x {F.shape[1]}  {len(st)} clips  device={device}")
    per_clip = events_per_clip(F.shape[0])

    # the clip permutation that defines the null (run_temporal.py verbatim)
    perm_clip = np.random.default_rng(0).permutation(len(st))
    shuf_rows = np.arange(len(cc))
    for i, (s0, e0) in enumerate(zip(st, en)):
        j = perm_clip[i]
        n = min(e0 - s0, en[j] - st[j])
        shuf_rows[s0:s0 + n] = st[j] + np.arange(n)
        if e0 - s0 > n:
            shuf_rows[s0 + n:e0] = st[j] + n - 1

    splits = []
    for f in (0, 1):
        te = folds == f
        tr = ~te
        tr_cl = np.unique(cc[tr])
        perm = np.random.default_rng(0 + f).permutation(tr_cl)
        sel_cl = set(int(x) for x in perm[:max(1, int(round(SEL_FRAC * len(tr_cl))))])
        is_sel = np.array([int(x) in sel_cl for x in cc])
        splits.append((tr & ~is_sel, tr & is_sel, te))

    # ONE feature build per (fold, real/null) — the PCA does not depend on the labels
    feats = {}
    for f in (0, 1):
        for shuf in (False, True):
            src = F[shuf_rows] if shuf else F
            mu, W = fit_pca(src, np.flatnonzero(splits[f][0]), RANK, device=device)
            X, _ = causal_window(project(src, mu, W), st, en, WIN)
            feats[(f, shuf)] = X
            log(f"  features built fold={f} shuf={shuf}: {X.shape}")
    _, hist_ok = causal_window(np.zeros((len(cc), 1), np.float32), st, en, WIN)

    res = {"_what": "anticipation HORIZON characterisation of the reference vision arm",
           "_not_a_reselection": ("lead_s = 3.0 remains the deployed constant; every horizon is "
                                  "reported and none is selected. No label constant is swept."),
           "representation": {"pca_rank": RANK, "win": WIN, "params_per_head": RANK * WIN + 1,
                              "arm": "ridge_app16_w8 (the reference recipe)"},
           "estimator": "paired_ap_episode_cluster_bootstrap (taniteval draws)",
           "n_boot": a.n_boot, "leads_s": list(LEADS), "per_situation": {s: {} for s in SITS}}

    for lead in LEADS:
        Yb, Vb = targets_at(per_clip, lead)
        V = Vb & hist_ok[:, None]
        Y = Yb.astype(np.float32)
        Yi = Yb.astype(np.int64)
        Vf = V.astype(np.float32)
        scores = {}
        for shuf in (False, True):
            out = np.zeros_like(Y)
            for f in (0, 1):
                X = feats[(f, shuf)]
                fit, sel, te = splits[f]
                best = (-np.inf, 1.0)
                for lam in LAMBDAS:
                    m_ = mean_ap(Yi[sel], ridge_scores(X[fit], Y[fit], Vf[fit], X[sel], lam=lam),
                                 V[sel])
                    if np.isfinite(m_) and m_ > best[0]:
                        best = (m_, lam)
                out[te] = ridge_scores(X[fit], Y[fit], Vf[fit], X[te], lam=best[1])
            scores["null" if shuf else "real"] = out
        for i, s in enumerate(SITS):
            m = V[:, i]
            yv = Yi[m, i]
            eid = cc[m]
            if yv.sum() < 5:
                res["per_situation"][s][f"lead_{lead}"] = {"_status": "NO_POSITIVES"}
                continue
            sc = scores["real"][m, i].astype(np.float64)
            nl = scores["null"][m, i].astype(np.float64)
            n_cl = int(len(np.unique(eid[yv > 0])))
            r_ = {"lead_s": lead, "n_scorable": int(m.sum()), "n_pos": int(yv.sum()),
                  "n_clusters_with_a_positive": n_cl,
                  "C_POW_pass": bool(n_cl >= 40),
                  "base_rate": round(float(yv.mean()), 6),
                  "ap": round(average_precision(yv, sc), 5),
                  "ap_lift": ap_episode_cluster_bootstrap(yv, sc, eid, n_boot=a.n_boot,
                                                          lift=True),
                  "null_ap_lift": round(ap_lift(yv, nl), 5),
                  "op_5pct": precision_recall_at_budget(yv, sc, np.ones(yv.size, bool)),
                  "paired_vs_null": paired_ap_episode_cluster_bootstrap(
                      yv, sc, nl, eid, n_boot=a.n_boot, lift=True)}
            res["per_situation"][s][f"lead_{lead}"] = r_
            p = r_["paired_vs_null"]
            log(f"  lead {lead:>3}s {s:>13}: pos {r_['n_pos']:>5,} in {n_cl:>3} clusters "
                f"base {r_['base_rate']:.5f}  AP {r_['ap']:.5f}  lift "
                f"{r_['ap_lift']['point']:.3f} [{r_['ap_lift']['lo']:.3f},"
                f"{r_['ap_lift']['hi']:.3f}]  P@5% {r_['op_5pct']['precision']:.4f} "
                f"(fires {r_['op_5pct']['n_alarm']}/{r_['op_5pct']['n_pos']})  vs-null "
                f"{p['delta']:+.3f} [{p['lo']:+.3f},{p['hi']:+.3f}]"
                f"{' SEP' if p['separated'] else ''}")
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")

    # the pre-registered read: does skill over the null move with the horizon?
    verdict = {}
    for s in SITS:
        rows = [(l, res["per_situation"][s].get(f"lead_{l}")) for l in LEADS]
        ok = [(l, r) for l, r in rows if r and r.get("C_POW_pass")]
        if len(ok) < 2:
            verdict[s] = {"VERDICT": "UNDERPOWERED_C_POW",
                          "powered_leads": [l for l, _ in ok]}
            continue
        d = {l: r["paired_vs_null"]["delta"] for l, r in ok}
        short, long_ = min(d), max(d)
        verdict[s] = {"powered_leads": list(d), "delta_vs_null_by_lead": d,
                      "short_minus_long": round(d[short] - d[long_], 5),
                      "VERDICT": ("HORIZON_LIMITED" if d[short] > 1.5 * max(d[long_], 1e-9)
                                  else "HORIZON_FLAT")}
        log(f"  {s}: {verdict[s]['VERDICT']} — vs-null delta by lead {d}")
    res["verdict"] = verdict
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
