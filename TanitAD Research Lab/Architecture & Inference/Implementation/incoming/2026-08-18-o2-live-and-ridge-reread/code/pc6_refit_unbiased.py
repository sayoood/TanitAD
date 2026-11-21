"""RE-READ every banked ``pc6_ridge_*.json`` under an UNPENALISED INTERCEPT.

⛔ ZERO GPU. Pure CPU refits on the banked latent caches.

THE DEFECT BEING CORRECTED (C92, repaired in `pc6_linear_readout.ridge_fit`
2026-08-18). Callers append a ones-column for the bias and hand the whole design
matrix to the ridge solve, where it sat inside ``alpha * np.eye(d)``. The
intercept was therefore shrunk like any other coefficient, so as alpha grows the
prediction collapses toward **ZERO, not toward the MEAN**. The readout could not
express the constant predictor K1 scores it against ⇒ **a no-signal arm scored
WORSE THAN A CONSTANT BY CONSTRUCTION.** That biases the FLOOR, so it taints
**K1 FAIL verdicts specifically**.

⭐ THE DEFECT IS VISIBLE IN THE BANKED FILES THEMSELVES, WHICH IS THE FIRST
CHECK THIS SCRIPT MAKES. Every banked ``alpha_inner_mae`` table rises to
~16-19 m at alpha=1e5 while the train-median constant scores 5.13 m and
mean(y) ~ 15 m. An MAE that converges to *mean(|y|)* rather than to the
constant's error IS the collapse-toward-zero signature.

WHAT THIS SCRIPT DOES, PER ARM — three fits, so the change is ATTRIBUTABLE:

  A. INCUMBENT   intercept_col=None, alpha re-selected  -> must REPRODUCE the
                 banked JSON bit-exactly. If it does not, nothing else here is
                 trustworthy and the script says so loudly.
  B. REPAIRED-A  intercept_col=-1, alpha HELD at the banked choice  -> isolates
                 the intercept penalty from the alpha re-selection.
  C. REPAIRED    intercept_col=-1, alpha RE-SELECTED on the same inner split
                 -> the honest re-read. This is the number that replaces the
                 banked one.

⛔ WHY NOT JUST FLIP THE DEFAULT. Banked results reproduce bit-exactly through
the incumbent path and `ll1_ladder.py` asserts that reproduction to 1e-4.
Mutating the default would rewrite the meaning of every committed
``pc6_ridge_*.json`` while leaving the filenames identical. The honest fix is to
RE-READ the banked numbers, which is this script, and it does not modify
`pc6_linear_readout.py` at all — it imports it.

ESTIMATOR. Unchanged: ``taniteval.ci.paired_episode_cluster_bootstrap`` over the
episode clusters, seed 0, n_boot 2000. ⛔ ``overlapping_holdout_se`` is
forbidden and is not used anywhere in this path.

TIER. T0-DIAGNOSTIC. A ridge readout on a latent is a world-model diagnostic and
is never a driving number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _imports(pc6_dir: Path, sp2_dir: Path):
    sys.path.insert(0, str(sp2_dir))
    sys.path.insert(0, str(pc6_dir))
    import pc6_linear_readout as PC6                              # noqa: E402
    import sp2_probe as SP                                        # noqa: E402
    from taniteval.ci import (episode_cluster_bootstrap,           # noqa: E402
                              paired_episode_cluster_bootstrap)
    return PC6, SP, episode_cluster_bootstrap, paired_episode_cluster_bootstrap


def build(cache: Path, split_json: Path, SP):
    """Design matrices — VERBATIM the data path of ``pc6_linear_readout.main``."""
    blob = torch.load(cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    decl = json.loads(Path(split_json).read_text("utf-8"))
    ev_c, tr_c = set(decl["eval_clips"]), set(decl["train_clips"])

    def pack(sel_clips):
        idx = [i for i, r in enumerate(rows) if r["clip_id"] in sel_clips]
        g = np.array([SP.gt_lead_gap(rows[i]["agents"])
                      if SP.gt_lead_gap(rows[i]["agents"]) is not None
                      else np.nan for i in idx])
        keep = ~np.isnan(g)
        idx = [i for i, k in zip(idx, keep) if k]
        X = np.stack([rows[i]["cells"].numpy().reshape(-1).astype(np.float64)
                      for i in idx])
        return X, g[keep], np.array([rows[i]["episode_uid"] for i in idx]), \
            np.array([rows[i]["clip_id"] for i in idx])

    Xtr, ytr, _etr, ctr = pack(tr_c)
    Xev, yev, eev, _cev = pack(ev_c)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0

    def prep(X):
        return np.concatenate([(X - mu) / sd, np.ones((X.shape[0], 1))], 1)

    # C-CONST / C-EPMEAN, verbatim
    const_m = float(np.median(ytr))
    ep_of = np.array([rows[i]["clip_id"] for i, r in enumerate(rows)
                      if r["clip_id"] in ev_c
                      and SP.gt_lead_gap(r["agents"]) is not None])
    epmean = np.full(len(yev), const_m, dtype=np.float64)
    for c in np.unique(ep_of):
        pos = np.nonzero(ep_of == c)[0]
        tot = float(np.sum(yev[pos]))
        for k in pos:
            epmean[k] = ((tot - yev[k]) / (pos.size - 1)) if pos.size > 1 \
                else const_m
    return {"Ztr": prep(Xtr), "Zev": prep(Xev), "ytr": ytr, "yev": yev,
            "eev": eev, "ctr": ctr, "const_m": const_m, "epmean": epmean,
            "meta": meta, "memory_shape": list(rows[0]["cells"].shape)}


def inner_mask(ctr, seed, inner_frac):
    rng = np.random.default_rng(seed)
    clips = np.array(sorted(set(ctr.tolist())))
    rng.shuffle(clips)
    n_in = max(1, int(round(len(clips) * inner_frac)))
    inner = set(clips[:n_in].tolist())
    return np.array([c in inner for c in ctr]), n_in


def score(D, pred, ecb, pcb, n_boot):
    e_arm = np.abs(pred - D["yev"])
    e_con = np.abs(D["const_m"] - D["yev"])
    e_ep = np.abs(D["epmean"] - D["yev"])
    arm = ecb(e_arm, D["eev"], n_boot=n_boot)
    k1 = pcb(e_arm, e_con, D["eev"], n_boot=n_boot)
    k5 = pcb(e_arm, e_ep, D["eev"], n_boot=n_boot)
    return {"ridge_err_m": round(float(e_arm.mean()), 4),
            "ridge_ci": [arm["lo"], arm["hi"]],
            "ridge_median_m": round(float(np.median(e_arm)), 4),
            "c_const_err_m": round(float(e_con.mean()), 4),
            "K1_delta": k1["delta"], "K1_lo": k1["lo"], "K1_hi": k1["hi"],
            "K1_separated": k1["separated"],
            "K1_PASSES": bool(k1["separated"] and k1["delta"] < 0),
            "K5_delta": k5["delta"], "K5_separated": k5["separated"],
            "K5_PASSES": bool(k5["separated"] and k5["delta"] < 0),
            "corr_pred_gt": round(float(np.corrcoef(pred, D["yev"])[0, 1]), 4),
            "pred_sd_m": round(float(pred.std()), 3)}


def fit_arm(D, PC6, ecb, pcb, alphas, seed, inner_frac, n_boot, hold_alpha):
    m_in, n_in = inner_mask(D["ctr"], seed, inner_frac)
    Ztr, ytr = D["Ztr"], D["ytr"]

    def select(icol):
        best, best_mae, tried = None, np.inf, {}
        for al in alphas:
            w = PC6.ridge_fit(Ztr[~m_in], ytr[~m_in], al, intercept_col=icol)
            mae = float(np.abs(Ztr[m_in] @ w - ytr[m_in]).mean())
            tried[f"{al:g}"] = round(mae, 4)
            if mae < best_mae:
                best, best_mae = al, mae
        return best, tried

    res = {"inner_split_clips": n_in}
    # A — INCUMBENT (must reproduce the banked file)
    a_inc, t_inc = select(None)
    p_inc = D["Zev"] @ PC6.ridge_fit(Ztr, ytr, a_inc, intercept_col=None)
    res["incumbent"] = {"alpha_chosen": a_inc, "alpha_inner_mae": t_inc,
                        **score(D, p_inc, ecb, pcb, n_boot)}
    # B — REPAIRED, alpha HELD at the banked choice (isolates the penalty)
    p_hold = D["Zev"] @ PC6.ridge_fit(Ztr, ytr, hold_alpha, intercept_col=-1)
    res["repaired_alpha_held"] = {"alpha_chosen": hold_alpha,
                                  **score(D, p_hold, ecb, pcb, n_boot)}
    # C — REPAIRED, alpha re-selected (the honest re-read)
    a_rep, t_rep = select(-1)
    p_rep = D["Zev"] @ PC6.ridge_fit(Ztr, ytr, a_rep, intercept_col=-1)
    res["repaired"] = {"alpha_chosen": a_rep, "alpha_inner_mae": t_rep,
                       **score(D, p_rep, ecb, pcb, n_boot)}
    return res


ARMS = [
    ("s11250", "v6F-SW-30k@11250", "sp2/cache_s11250/latents.pt"),
    ("s09000", "v6F-SW-30k@9000", "sp2/cache_s09000/latents.pt"),
    ("nullmatched", "RANDOM-LATENT-NULL-MATCHED", "sp2/cache_nullmatched/latents.pt"),
    ("orc010", "GT-ORACLE-CELLS-n0.1@11250", "pc/cache_orc010/latents.pt"),
    ("orcdir", "GT-ORACLE-DIRECT-n0.1@11250", "pc/cache_orcdir/latents.pt"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True, help="dir holding pc/ and sp2/")
    ap.add_argument("--banked", required=True, help="dir of banked pc6_ridge_*.json")
    ap.add_argument("--pc6-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5])
    ap.add_argument("--inner-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)

    scratch, banked = Path(a.scratch), Path(a.banked)
    pc6_dir = Path(a.pc6_dir)
    PC6, SP, ecb, pcb = _imports(pc6_dir, scratch / "pc")

    split = scratch / "sp2" / "p3_selection.json"
    results, repro_ok = {}, True
    for tag, label, rel in ARMS:
        if a.only and tag not in a.only:
            continue
        cache = scratch / rel
        bank_p = banked / f"pc6_ridge_{tag}.json"
        if not cache.exists():
            print(f"MISSING CACHE {tag}: {cache}")
            continue
        bank = json.loads(bank_p.read_text("utf-8"))
        D = build(cache, split, SP)
        r = fit_arm(D, PC6, ecb, pcb, a.alphas, a.seed, a.inner_frac,
                    a.n_boot, bank["alpha_chosen"])
        # --- the reproduction gate ---
        chk = {k: (r["incumbent"][k], bank[k]) for k in
               ("alpha_chosen", "ridge_err_m", "K1_delta", "K1_lo", "K1_hi",
                "K1_separated", "K1_PASSES", "corr_pred_gt")}
        ok = all(abs(x - y) < 1e-6 if isinstance(x, (int, float))
                 and not isinstance(x, bool) else x == y
                 for x, y in chk.values())
        repro_ok &= ok
        r["banked"] = {k: bank.get(k) for k in
                       ("arm", "run_stamp", "alpha_chosen", "ridge_err_m",
                        "ridge_ci", "K1_delta", "K1_lo", "K1_hi",
                        "K1_separated", "K1_PASSES", "K5_PASSES",
                        "corr_pred_gt", "c_const_err_m")}
        r["reproduces_banked"] = ok
        r["reproduction_check"] = {k: {"refit": v[0], "banked": v[1]}
                                   for k, v in chk.items()}
        r["arm"] = label
        r["run_stamp"] = D["meta"].get("run_stamp")
        r["n_eval_windows"] = int(D["Zev"].shape[0])
        r["n_eval_clusters"] = int(len(np.unique(D["eev"])))
        r["n_train_windows"] = int(D["Ztr"].shape[0])
        r["memory_shape"] = D["memory_shape"]
        r["c_const_m_value"] = round(D["const_m"], 4)
        results[tag] = r

        b, inc, hld, rep = r["banked"], r["incumbent"], \
            r["repaired_alpha_held"], r["repaired"]
        print(f"\n=== {tag}  ({label})   reproduces_banked={ok}")
        print(f"  BANKED     err {b['ridge_err_m']:7.3f}  K1 {b['K1_delta']:+7.3f} "
              f"[{b['K1_lo']:+.3f},{b['K1_hi']:+.3f}] sep={b['K1_separated']!s:5} "
              f"{'PASS' if b['K1_PASSES'] else 'FAIL':4}  alpha {b['alpha_chosen']:g}")
        print(f"  incumbent  err {inc['ridge_err_m']:7.3f}  K1 {inc['K1_delta']:+7.3f} "
              f"[{inc['K1_lo']:+.3f},{inc['K1_hi']:+.3f}] sep={inc['K1_separated']!s:5} "
              f"{'PASS' if inc['K1_PASSES'] else 'FAIL':4}  alpha {inc['alpha_chosen']:g}")
        print(f"  rep(hold)  err {hld['ridge_err_m']:7.3f}  K1 {hld['K1_delta']:+7.3f} "
              f"[{hld['K1_lo']:+.3f},{hld['K1_hi']:+.3f}] sep={hld['K1_separated']!s:5} "
              f"{'PASS' if hld['K1_PASSES'] else 'FAIL':4}  alpha {hld['alpha_chosen']:g}")
        print(f"  REPAIRED   err {rep['ridge_err_m']:7.3f}  K1 {rep['K1_delta']:+7.3f} "
              f"[{rep['K1_lo']:+.3f},{rep['K1_hi']:+.3f}] sep={rep['K1_separated']!s:5} "
              f"{'PASS' if rep['K1_PASSES'] else 'FAIL':4}  alpha {rep['alpha_chosen']:g}"
              f"   r {rep['corr_pred_gt']:+.4f}")
        sys.stdout.flush()

    payload = {
        "_evidence_class": "MEASURED (ours; refits of the banked pc6 latent caches "
                           "on CPU, no GPU, no re-encode)",
        "eval_tier": "T0-DIAGNOSTIC (WM diagnostic — never a driving number)",
        "estimator": "taniteval.ci.paired_episode_cluster_bootstrap (seed 0)",
        "forbidden": "overlapping_holdout_se",
        "defect": "C92 — penalised intercept collapses predictions toward ZERO "
                  "not the MEAN; biases the FLOOR; taints K1 FAIL verdicts",
        "all_incumbent_fits_reproduce_banked": repro_ok,
        "arms": results,
    }
    Path(a.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nALL INCUMBENT FITS REPRODUCE BANKED: {repro_ok}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
