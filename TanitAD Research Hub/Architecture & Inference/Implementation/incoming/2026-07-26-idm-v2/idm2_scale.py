"""IDM-v2 -- is the monocular scale gain a PER-CORPUS CONSTANT (metric grounding
with one number per camera works) or PER-CLIP (you need per-clip geometry)?

Two things, both honest fit/apply splits (never oracle-on-val):

A. Cross-domain probe gains.  Fit the w9 ridge on PhysicalAI TRAIN, apply to
   comma; fit the per-clip affine y ~ a*p + b on each VAL clip and look at the
   DISTRIBUTION of `a`.  If `a` clusters tightly by corpus, one constant per
   camera is enough.  If it scatters within a corpus, per-clip geometry is
   required.

B. The zero-training deployment fix: take the PERSISTED idm_head_v1 (A0), fit a
   per-domain affine on the 68 TRAIN episodes it has also never seen, and apply
   it to the 36 VAL episodes.  This is what a per-corpus calibration constant
   would buy today, with no retraining at all.

Writes /root/idm2/out/scale.json
"""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.path.insert(0, "/root/idm2")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")
import idm2_lib as L                      # noqa: E402
import idm_head as ih                     # noqa: E402
from idm2_diag_probe import ridge_fit_predict, standardise   # noqa: E402

DEV = "cuda"
K = 8


def afit(p, y):
    A = np.stack([p, np.ones(p.size)], 1)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(c[0]), float(c[1])


def main():
    tr_tags, va_tags = L.split_tags()
    tr = L.build_set(tr_tags, k=K, stride=1)
    va = L.build_set(va_tags, k=K, stride=2)
    Str = tr["S"].numpy().astype(np.float64)
    Sva = va["S"].numpy().astype(np.float64)
    out = {"n_train_eps": len(tr_tags), "n_val_eps": len(va_tags)}

    # ---------------- A. cross-domain probe gains ------------------------ #
    Zt = tr["Z"][:, K - 4:K + 5].to(DEV).float().reshape(tr["Z"].shape[0], -1)
    Zv = va["Z"][:, K - 4:K + 5].to(DEV).float().reshape(va["Z"].shape[0], -1)
    gains = {}
    for src in ("pai", "cm"):
        m = torch.tensor(tr["dom"] == src, device=DEV)
        Xt, (Xv,) = standardise(Zt[m], [Zv])
        y = torch.tensor(Str[tr["dom"] == src, 0], device=DEV,
                         dtype=torch.float32).unsqueeze(1)
        mu = y.mean()
        pf = ridge_fit_predict(Xt, y - mu, [Xv], [1e4])
        p = pf[1e4][0][:, 0] + float(mu)
        rec = {}
        for dst in ("pai", "cm"):
            dm = va["dom"] == dst
            per = {}
            for tag in np.unique(va["eid"][dm]):
                cm_ = va["eid"] == tag
                a, b = afit(p[cm_], Sva[cm_, 0])
                per[tag] = {"a": a, "b": b, "v_mean": float(Sva[cm_, 0].mean())}
            A = np.array([v["a"] for v in per.values()])
            B = np.array([v["b"] for v in per.values()])
            # one CONSTANT per corpus, fitted on the pooled corpus windows
            ac, bc = afit(p[dm], Sva[dm, 0])
            rec[dst] = {
                "r2_raw": L.chan_metrics(p[dm], Sva[dm, 0])["r2"],
                "r2_corpus_affine": L.chan_metrics(ac * p[dm] + bc, Sva[dm, 0])["r2"],
                "r2_corpus_scale_only": L.chan_metrics(
                    float((p[dm] * Sva[dm, 0]).sum() / (p[dm] ** 2).sum()) * p[dm],
                    Sva[dm, 0])["r2"],
                "corpus_a": ac, "corpus_b": bc,
                "per_clip_a_mean": float(A.mean()), "per_clip_a_std": float(A.std()),
                "per_clip_a_cv": float(A.std() / max(abs(A.mean()), 1e-9)),
                "per_clip_a_min": float(A.min()), "per_clip_a_max": float(A.max()),
                "per_clip_b_mean": float(B.mean()), "per_clip_b_std": float(B.std()),
                "n_clips": int(A.size), "per_clip": per}
            print(f"probe[{src}] -> {dst}: R2 raw {rec[dst]['r2_raw']:+.4f} | "
                  f"corpus-affine {rec[dst]['r2_corpus_affine']:+.4f} "
                  f"(a={ac:.3f} b={bc:+.3f}) | corpus-scale-only "
                  f"{rec[dst]['r2_corpus_scale_only']:+.4f} | per-clip a "
                  f"{A.mean():.3f}+-{A.std():.3f} (cv {A.std()/abs(A.mean()):.3f}, "
                  f"{A.min():.3f}..{A.max():.3f})", flush=True)
        gains[src] = rec
        del Xt, Xv
        torch.cuda.empty_cache()
    out["cross_domain_gains"] = gains

    # ---------------- B. A0 + per-corpus calibration --------------------- #
    d = torch.load("/root/idmval/idm_head_v1.pt", weights_only=False)
    h = ih.IDMHead(**d["config"]["head_kwargs"]).to(DEV)
    h.load_state_dict(d["state_dict"]); h.eval()

    @torch.no_grad()
    def a0(Z):
        return torch.cat([h(Z[i:i + 1024])["scalars"].cpu()
                          for i in range(0, Z.shape[0], 1024)]).numpy().astype(np.float64)
    Ptr = a0(tr["Z"][:, K - 4:K + 5].to(DEV).float())
    Pva = a0(va["Z"][:, K - 4:K + 5].to(DEV).float())
    cal = {}
    for j, nm in ((0, "speed"),):
        coef = {dd: afit(Ptr[tr["dom"] == dd, j], Str[tr["dom"] == dd, j])
                for dd in ("pai", "cm")}
        p = Pva[:, j].copy()
        for dd, (a, b) in coef.items():
            m = va["dom"] == dd
            p[m] = a * p[m] + b
        pg = Pva[:, j].copy()
        ag, bg = afit(Ptr[:, j], Str[:, j])
        pg = ag * pg + bg
        cal[nm] = {
            "coef_per_domain_fitted_on_TRAIN": {k: list(v) for k, v in coef.items()},
            "coef_global_fitted_on_TRAIN": [ag, bg],
            "raw": L.chan_metrics(Pva[:, j], Sva[:, j]),
            "global_affine": L.chan_metrics(pg, Sva[:, j]),
            "per_domain_affine": L.chan_metrics(p, Sva[:, j]),
            "raw_per_domain": {dd: L.chan_metrics(Pva[va["dom"] == dd, j],
                                                  Sva[va["dom"] == dd, j])
                               for dd in ("pai", "cm")},
            "cal_per_domain": {dd: L.chan_metrics(p[va["dom"] == dd],
                                                  Sva[va["dom"] == dd, j])
                               for dd in ("pai", "cm")},
            "boot_paired_mae_cal_minus_raw": L.paired_mae(p, Pva[:, j], Sva[:, j],
                                                          va["eid"]),
        }
        r = cal[nm]
        print(f"\nA0 {nm}: raw R2 {r['raw']['r2']:+.4f} MAE {r['raw']['mae']:.3f} "
              f"-> +global affine {r['global_affine']['r2']:+.4f} "
              f"MAE {r['global_affine']['mae']:.3f} "
              f"-> +PER-CORPUS affine {r['per_domain_affine']['r2']:+.4f} "
              f"MAE {r['per_domain_affine']['mae']:.3f}")
        print(f"   coefs (fit on TRAIN eps): {r['coef_per_domain_fitted_on_TRAIN']}")
        print(f"   paired dMAE {r['boot_paired_mae_cal_minus_raw']['delta']:+.4f} "
              f"[{r['boot_paired_mae_cal_minus_raw']['lo']:+.4f},"
              f"{r['boot_paired_mae_cal_minus_raw']['hi']:+.4f}] "
              f"sep={r['boot_paired_mae_cal_minus_raw']['separated']}")
        for dd in ("pai", "cm"):
            print(f"   {dd}: R2 {r['raw_per_domain'][dd]['r2']:+.4f} "
                  f"MAE {r['raw_per_domain'][dd]['mae']:.3f} -> "
                  f"R2 {r['cal_per_domain'][dd]['r2']:+.4f} "
                  f"MAE {r['cal_per_domain'][dd]['mae']:.3f}")
    out["a0_calibration"] = cal
    L.jdump(out, "/root/idm2/out/scale.json")


if __name__ == "__main__":
    main()
