"""⭐ THE DERIVED CLAIM THE REPAIR PUTS AT RISK — arm vs NULL, paired, both fitted
under an UNPENALISED INTERCEPT.

WHAT THIS TESTS. `2026-08-17-O234-DESIGN-RESEARCH.md` §3.4 reads the banked pc6
table as: *"the v6 latent beats a random-vector null by 1.6-1.8 m with positive
correlation"* (arm 6.713 m vs null 8.534 m). That comparison is between two fits
that were BOTH produced under the penalised intercept, and the defect does not
bias both equally: it hurts a NO-SIGNAL arm hardest, because with the bias term
shrunk the only way to reach y's ~15 m level is to load the FEATURES — so the
noise arm is forced to hallucinate variance. (MEASURED: the incumbent null's
pred_sd is 8.47 m against a GT sd of 6.20 m — 1.37x the spread of the truth.)

⇒ the arm-vs-null MARGIN is exactly the quantity the defect most inflates, and
re-reading each arm separately is not enough — the comparison itself must be
re-run, PAIRED, on identical windows.

PAIRING IS VALID BY CONSTRUCTION AND IS VERIFIED HERE. `pA_null_matched.py`
takes the REAL cache and replaces ONLY the `cells` tensor with `torch.randn`,
keeping every window, clip_id, episode and target. This script ASSERTS the two
eval target vectors are elementwise identical before pairing them; if they are
not, it refuses rather than reporting an unpaired comparison as a paired one.

ESTIMATOR: taniteval.ci.paired_episode_cluster_bootstrap, seed 0, n_boot 2000.
⛔ overlapping_holdout_se is forbidden and unused.
TIER: T0-DIAGNOSTIC.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pc6_refit_unbiased import build, inner_mask, _imports          # noqa: E402


def fit(D, PC6, alphas, seed, inner_frac, icol):
    m_in, _ = inner_mask(D["ctr"], seed, inner_frac)
    best, best_mae = None, np.inf
    for al in alphas:
        w = PC6.ridge_fit(D["Ztr"][~m_in], D["ytr"][~m_in], al, intercept_col=icol)
        mae = float(np.abs(D["Ztr"][m_in] @ w - D["ytr"][m_in]).mean())
        if mae < best_mae:
            best, best_mae = al, mae
    w = PC6.ridge_fit(D["Ztr"], D["ytr"], best, intercept_col=icol)
    return D["Zev"] @ w, best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--pc6-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5])
    ap.add_argument("--inner-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    scratch = Path(a.scratch)
    PC6, SP, ecb, pcb = _imports(Path(a.pc6_dir), scratch / "pc")
    split = scratch / "sp2" / "p3_selection.json"

    Dn = build(scratch / "sp2/cache_nullmatched/latents.pt", split, SP)
    out = {"_evidence_class": "MEASURED (ours; paired refits on the banked caches, CPU)",
           "eval_tier": "T0-DIAGNOSTIC",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap (seed 0)",
           "forbidden": "overlapping_holdout_se",
           "claim_under_test": ("2026-08-17-O234-DESIGN-RESEARCH.md 3.4: 'the v6 "
                                "latent beats a random-vector null by 1.6-1.8 m'"),
           "null_construction": ("pA_null_matched.py — the REAL cache with `cells` "
                                 "replaced by torch.randn(seed 0); every window, "
                                 "clip and target preserved"),
           "arms": {}}

    for tag, rel in (("s11250", "sp2/cache_s11250/latents.pt"),
                     ("s09000", "sp2/cache_s09000/latents.pt")):
        Da = build(scratch / rel, split, SP)
        # ⛔ refuse to pair unless the windows really are identical
        same = (Da["yev"].shape == Dn["yev"].shape
                and bool(np.array_equal(Da["yev"], Dn["yev"]))
                and bool(np.array_equal(Da["eev"], Dn["eev"])))
        if not same:
            print(f"{tag}: ⛔ WINDOWS DIFFER — refusing to report an unpaired "
                  f"comparison as paired")
            out["arms"][tag] = {"paired_valid": False}
            continue

        row = {"paired_valid": True, "n_eval_windows": int(len(Da["yev"])),
               "n_eval_clusters": int(len(np.unique(Da["eev"])))}
        for mode, icol in (("incumbent", None), ("repaired", -1)):
            pa, aa = fit(Da, PC6, a.alphas, a.seed, a.inner_frac, icol)
            pn, an = fit(Dn, PC6, a.alphas, a.seed, a.inner_frac, icol)
            ea, en = np.abs(pa - Da["yev"]), np.abs(pn - Dn["yev"])
            d = pcb(ea, en, Da["eev"], n_boot=a.n_boot)
            row[mode] = {
                "arm_alpha": aa, "null_alpha": an,
                "arm_err_m": round(float(ea.mean()), 4),
                "null_err_m": round(float(en.mean()), 4),
                "arm_minus_null_m": d["delta"], "lo": d["lo"], "hi": d["hi"],
                "separated": d["separated"],
                "arm_beats_null": bool(d["separated"] and d["delta"] < 0),
                "arm_pred_sd": round(float(pa.std()), 4),
                "null_pred_sd": round(float(pn.std()), 4),
                "gt_sd": round(float(Da["yev"].std()), 4)}
            r = row[mode]
            print(f"{tag:8} {mode:10} arm {r['arm_err_m']:7.3f}  null "
                  f"{r['null_err_m']:7.3f}  arm-null {r['arm_minus_null_m']:+7.3f} "
                  f"[{r['lo']:+.3f},{r['hi']:+.3f}] sep={r['separated']!s:5} "
                  f"-> {'ARM BEATS NULL' if r['arm_beats_null'] else 'arm does NOT beat null'}"
                  f"   (null pred_sd {r['null_pred_sd']:.3f} vs gt_sd {r['gt_sd']:.3f})")
        out["arms"][tag] = row

    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
