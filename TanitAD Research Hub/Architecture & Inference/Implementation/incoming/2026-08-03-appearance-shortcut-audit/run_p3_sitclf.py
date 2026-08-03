"""D-APPEAR P3 — does the appearance shortcut THREATEN the vision-only scenario classifier?

Pre-registration: ``Project Steering/PREREG_APPEARANCE_SHORTCUT.md`` §3.

THE CONCERN, stated exactly
    Situation labels are a DETERMINISTIC FUNCTION OF THE EGO POSE TRACK
    (``stack/tanitad/data/situations.py``; ``scripts/emit_situation_labels.py:54-62`` reads only
    ``d["poses"]``). The PI's binding ruling -- *labels may use ego; inference is VISION-ONLY* --
    makes ``head_img`` the only deployable arm.

    If a still frame reads ``speed`` at 93 % of the full latent (the comma2k19 finding), then a
    "vision-only" classifier can score by the indirect route
        APPEARANCE  ->  SPEED  ->  the EGO-DERIVED LABEL
    without ever perceiving the situation. That is the SAME family as the leak test the PI made
    binding -- *does an input at inference contain something the label was derived from* -- with
    one extra hop.

⛔ SCOPE. A separate stream owns the scenario classifier. This script MEASURES and ESCALATES.
   It edits nothing under the sitclf tree and overturns no sitclf verdict.

⚠️ WHAT THIS IS NOT. It is not a re-run of ``head_img``. The deployed head consumes an 8-frame
   window (offsets -7..0, ``sc_train.py:37``) through a PCA block; every arm here is a SINGLE
   FRAME, so all arms are internally comparable to each other and NONE of them is the deployed
   number. The comparison that matters is between the arms, not against the banked AP.

usage:
  OMP_NUM_THREADS=6 python run_p3_sitclf.py --n-boot 2000 --out results_p3_sitclf.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))

SITCLF_NPZ = Path(r"C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.npz")
SITCLF_META = Path(r"C:/Users/Admin/tanitad-data/eval/sitclf_b4_substrate.meta.json")
EPC_TRAIN = Path(r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74")
EPC_VAL = Path(r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836")
STILL_CACHE = Path(r"C:/Users/Admin/tanitad-data/eval/dappear_sitclf_still32.pt")
SIDE = 32
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# --------------------------------------------------------------------------- #
def build_still(clusters_needed) -> dict:
    """Per-FRAME 32x32 grey still features for every clip in the sitclf substrate.

    Same luma + exact-average-pool recipe as the P1 substrate, on the LATEST RGB triplet of
    the D-015 9-channel stack (one row = one timestep)."""
    if STILL_CACHE.exists():
        log(f"loading {STILL_CACHE}")
        return torch.load(STILL_CACHE, map_location="cpu", weights_only=False)
    LUMA = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    out = {}
    for n, (tag, cid) in enumerate(sorted(clusters_needed)):
        base = EPC_TRAIN if tag == 0 else EPC_VAL
        idx = cid if tag == 0 else cid - 400
        e = torch.load(base / f"ep_{idx:05d}.pt", map_location="cpu", weights_only=False)
        x = e["frames_u8"][:, 6:9].float() / 255.0
        g = (x * LUMA).sum(1, keepdim=True)
        g = torch.nn.functional.avg_pool2d(g, g.shape[-1] // SIDE)
        out[(tag, cid)] = g.reshape(g.shape[0], -1).half()
        if (n + 1) % 50 == 0:
            log(f"  still {n+1}/{len(clusters_needed)}")
    torch.save(out, STILL_CACHE)
    log(f"cached -> {STILL_CACHE}")
    return out


class PrimalRidge:
    """Exact ridge in the PRIMAL, whole alpha path from one eigendecomposition.

    The dual is wrong here: n ~ 66 000 frames against D <= 2 048 features, so the dual would be
    a 66 000^2 eigenproblem for a rank-2 048 map. The primal Gram is D x D and the alpha path is
    free after one ``eigh``. Features are centred and standardised on TRAIN, and the intercept is
    carried by the target mean.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor, *, device="cpu"):
        self.mu = X.mean(0, keepdim=True)
        self.sd = X.std(0, keepdim=True).clamp_min(1e-6)
        Xs = ((X - self.mu) / self.sd).to(device, torch.float32)
        self.ymu = float(y.mean())
        yc = (y - self.ymu).to(device, torch.float32)[:, None]
        G = (Xs.T @ Xs).double().cpu()
        self.Xty = (Xs.T @ yc).double().cpu()
        w, V = torch.linalg.eigh(G)
        self.w, self.V = w.clamp_min(0.0), V

    def coef(self, alpha: float) -> torch.Tensor:
        return self.V @ ((self.V.T @ self.Xty) / (self.w[:, None] + float(alpha)))

    def predict(self, X: torch.Tensor, alpha: float) -> np.ndarray:
        Xs = ((X - self.mu) / self.sd).double()
        return (Xs @ self.coef(alpha)).numpy().ravel() + self.ymu

    @staticmethod
    def alpha_grid(lo=-2, hi=8, per_decade=2):
        k = int(round((hi - lo) * per_decade)) + 1
        return [float(10.0 ** e) for e in np.linspace(lo, hi, k)]


def fit_score(Xf, yf, Xs, ys, XF, yF, Xh, *, device="cpu", metric="ap"):
    """Fit on (fit, full), select alpha on the inner split, return the held-out score."""
    from tanitad.eval.ap_ci import average_precision
    from tanitad.eval.accel_probe import r2_score
    inner = PrimalRidge(Xf, yf, device=device)
    best_a, best_s = None, -1e18
    for al in PrimalRidge.alpha_grid():
        p = inner.predict(Xs, al)
        s = average_precision(ys.numpy(), p) if metric == "ap" else r2_score(p, ys.numpy())
        if s > best_s:
            best_a, best_s = al, s
    del inner
    full = PrimalRidge(XF, yF, device=device)
    return full.predict(Xh, best_a), float(best_a), float(best_s)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_p3_sitclf.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from tanitad.eval import ap_ci as APCI
    from tanitad.eval.accel_probe import r2_score

    z = np.load(SITCLF_NPZ, allow_pickle=True)
    meta = json.load(open(SITCLF_META))
    F, Y, V, E = z["F"], z["Y"], z["V"], z["E"]
    cc, ct, tt = z["clip_cluster"], z["cache_tag"], z["t"]
    sits = [str(s) for s in z["situations"]]
    v_true = E[:, 0] * meta["ego_scale"][0]

    clusters = sorted({(int(x), int(y)) for x, y in zip(ct, cc)})
    still = build_still(clusters)
    S = np.zeros((len(cc), SIDE * SIDE), dtype=np.float16)
    for (tag, cid), g in still.items():
        m = (ct == tag) & (cc == cid)
        S[m] = g.numpy()[tt[m]]
    log(f"still features {S.shape}; latents {F.shape}")

    # clip-disjoint split -- the clip is the independent unit (I3)
    ho = (cc % 3) == 0
    tr = ~ho
    inner_sel = tr & ((cc % 9) == 1)
    inner_fit = tr & ~inner_sel
    log(f"train {tr.sum()} / heldout {ho.sum()} frames; "
        f"{len(np.unique(cc[tr]))} / {len(np.unique(cc[ho]))} clips")

    Ft = torch.from_numpy(np.asarray(F, np.float32))
    St = torch.from_numpy(np.asarray(S, np.float32))
    Vt = torch.from_numpy(v_true.astype(np.float32))

    res = {"meta": {
        "prereg": "Project Steering/PREREG_APPEARANCE_SHORTCUT.md#3",
        "substrate": str(SITCLF_NPZ), "encoder": meta["trunk"]["ckpt"],
        "encoder_step": meta["trunk"]["step"], "n_frames": int(len(cc)),
        "n_clips": len(np.unique(cc)), "situations": sits,
        "label_provenance": "deterministic function of the ego pose track "
                            "(tanitad/data/situations.py; emit_situation_labels.py:54-62)",
        "scope": "SINGLE-FRAME diagnostic arms; NOT a re-run of the deployed head_img "
                 "(which uses an 8-frame window through PCA). Arms are comparable to each "
                 "other, not to the banked sitclf AP.",
        "estimator": "episode(clip)-cluster bootstrap on AP-lift (tanitad/eval/ap_ci.py)",
        "n_boot": a.n_boot}, "speed_readability": {}, "situations": {}}

    # --- step 0: how well does a STILL FRAME read speed on this substrate? --- #
    vp, alpha, sel = fit_score(St[inner_fit], Vt[inner_fit], St[inner_sel], Vt[inner_sel],
                               St[tr], Vt[tr], St, device=a.device, metric="r2")
    fp, alpha_f, sel_f = fit_score(Ft[inner_fit], Vt[inner_fit], Ft[inner_sel], Vt[inner_sel],
                                   Ft[tr], Vt[tr], Ft, device=a.device, metric="r2")
    res["speed_readability"] = {
        "note": "the shortcut's FIRST HOP, measured on the sitclf substrate itself",
        "still32_speed_r2_heldout": round(r2_score(vp[ho], v_true[ho]), 5),
        "latent_speed_r2_heldout": round(r2_score(fp[ho], v_true[ho]), 5),
        "still32_alpha": alpha, "latent_alpha": alpha_f,
        "still32_inner_r2": round(sel, 5), "latent_inner_r2": round(sel_f, 5)}
    log(f"speed read: still32 R2 {res['speed_readability']['still32_speed_r2_heldout']:+.4f}  "
        f"latent R2 {res['speed_readability']['latent_speed_r2_heldout']:+.4f}")

    # --- the arms, per situation --- #
    for si, sname in enumerate(sits):
        ok = V[:, si].astype(bool)
        y = Y[:, si].astype(np.float32)
        h = ho & ok
        f_, s_, F_ = inner_fit & ok, inner_sel & ok, tr & ok
        if h.sum() < 50 or y[h].sum() < 5:
            res["situations"][sname] = {"status": "UNPOWERED",
                                        "n_heldout": int(h.sum()),
                                        "n_pos_heldout": int(y[h].sum())}
            continue
        yt = torch.from_numpy(y)
        blocks = {
            "img_latent": Ft,
            "img_still32": St,
            "ego_speed_true": Vt[:, None],
            "speed_from_appearance": torch.from_numpy(vp.astype(np.float32))[:, None],
        }
        blocks["speed_plus_img"] = torch.cat([Vt[:, None], Ft], 1)
        rec = {"n_heldout": int(h.sum()), "n_pos_heldout": int(y[h].sum()),
               "base_rate_heldout": round(float(y[h].mean()), 6), "arms": {}}
        scores = {}
        for aname, X in blocks.items():
            p, al, isel = fit_score(X[f_], yt[f_], X[s_], yt[s_], X[F_], yt[F_], X,
                                    device=a.device, metric="ap")
            scores[aname] = p
            rec["arms"][aname] = {
                "n_features": int(X.shape[1]), "alpha": al, "inner_ap": round(isel, 6),
                "null_is_degenerate_for_1d": bool(X.shape[1] == 1),
                "ap_lift": APCI.ap_episode_cluster_bootstrap(
                    y[h], p[h], cc[h], n_boot=a.n_boot, lift=True)}
            # its own permuted-feature null
            # ⛔ DEGENERATE FOR 1-D ARMS, AND SAID SO IN THE OUTPUT. AP is RANK-based, so a
            # 1-feature model whose coefficient is fitted on scrambled pairs still ranks the
            # held-out rows by that same single feature (possibly reversed) and can reproduce
            # the arm's AP exactly. The permuted-feature null is only meaningful where the
            # fitted DIRECTION can become random, i.e. for high-dimensional blocks. It is
            # reported for every arm for symmetry and is used for NO verdict.
            g = np.random.default_rng(7)
            perm = g.permutation(int(F_.sum()))
            Xp = X[F_][perm]
            pn, _, _ = fit_score(Xp[:len(Xp) * 2 // 3], yt[F_][:len(Xp) * 2 // 3],
                                 Xp[len(Xp) * 2 // 3:], yt[F_][len(Xp) * 2 // 3:],
                                 Xp, yt[F_], X, device=a.device, metric="ap")
            rec["arms"][aname]["ap_lift_shuffled_null"] = APCI.ap_episode_cluster_bootstrap(
                y[h], pn[h], cc[h], n_boot=min(a.n_boot, 500), lift=True)
            rec["arms"][aname]["paired_vs_null"] = APCI.paired_ap_episode_cluster_bootstrap(
                y[h], p[h], pn[h], cc[h], n_boot=min(a.n_boot, 500), lift=True)
            log(f"{sname:14s} {aname:22s} AP-lift "
                f"{rec['arms'][aname]['ap_lift']['point']:+.4f} "
                f"[{rec['arms'][aname]['ap_lift']['lo']:+.4f},"
                f"{rec['arms'][aname]['ap_lift']['hi']:+.4f}]  "
                f"null {rec['arms'][aname]['ap_lift_shuffled_null']['point']:+.4f}")
        # the pre-registered verdict quantities
        # ⚠️ MY OWN PRE-REGISTRATION UNDER-SPECIFIED THIS AND I AM NOT SILENTLY REDEFINING IT.
        # ``ap_lift`` is AP / base_rate, so CHANCE IS 1.0, NOT 0.0. A ratio of raw lifts
        # therefore inherits a +1 offset in both terms and is biased towards "THREATENED":
        # an arm at chance (1.00) scores 0.50 of an arm at 2.00 while contributing NOTHING.
        # The defensible statistic is the ratio of EXCESS lift (lift - 1). Both are reported;
        # the verdict is taken on the excess form and the raw form is shown beside it.
        lift = {k: rec["arms"][k]["ap_lift"]["point"] for k in blocks}
        exc = {k: v - 1.0 for k, v in lift.items()}
        share_raw = (lift["speed_from_appearance"] / lift["img_latent"]
                     if lift["img_latent"] > 0 else float("nan"))
        share_exc = (exc["speed_from_appearance"] / exc["img_latent"]
                     if exc["img_latent"] > 0 else float("nan"))
        incremental = exc["speed_plus_img"] - exc["ego_speed_true"]   # == lift difference
        keeps = (incremental / exc["img_latent"] if exc["img_latent"] > 0 else float("nan"))
        if not np.isfinite(share_exc):
            verdict = "VOID (img_latent is at or below chance)"
        elif share_exc >= 0.50:
            verdict = "THREATENED"
        elif keeps >= 0.70:
            verdict = "NOT THREATENED"
        else:
            verdict = "MIXED"
        rec["verdict"] = {
            "ap_lift_note": "ap_lift = AP / base_rate, so CHANCE = 1.0. 'excess' = lift - 1.",
            "shortcut_share_EXCESS_lift": round(float(share_exc), 5),
            "shortcut_share_RAW_lift_as_preregistered": round(float(share_raw), 5),
            "incremental_ap_lift_of_vision_over_speed": round(float(incremental), 5),
            "incremental_share_of_img_excess_lift": round(float(keeps), 5),
            "PREREG_OUTCOME": verdict,
            "prereg_defect": "the pre-registration wrote '>= 50 % of img_latent's AP-lift' "
                             "without saying that AP-lift's chance value is 1.0, not 0. The "
                             "raw-lift form is reported unchanged so the registered rule is "
                             "auditable; the verdict uses the excess form."}
        rec["paired_img_vs_speed_from_appearance"] = APCI.paired_ap_episode_cluster_bootstrap(
            y[h], scores["img_latent"][h], scores["speed_from_appearance"][h], cc[h],
            n_boot=a.n_boot, lift=True)
        rec["paired_speedplusimg_vs_speed"] = APCI.paired_ap_episode_cluster_bootstrap(
            y[h], scores["speed_plus_img"][h], scores["ego_speed_true"][h], cc[h],
            n_boot=a.n_boot, lift=True)
        res["situations"][sname] = rec
        log(f">> {sname}: shortcut_share(excess) {share_exc:.4f} "
            f"(raw {share_raw:.4f})  incremental {incremental:+.4f} -> {verdict}")
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))

    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
