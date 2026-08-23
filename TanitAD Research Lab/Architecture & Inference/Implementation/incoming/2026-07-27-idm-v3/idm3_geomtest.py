"""IDM v3 — the SHARP 0-GPU test of the geometry hypothesis.

`idm3_labels.py` ran the naive version and it came back UNDERPOWERED and with
the WRONG SIGN (PAI rel_bias vs cam_h r = +0.433, CI [-0.369, +0.773], n = 14).
Three things are wrong with that test and all three are fixed here:

  1. **n = 14 was needlessly small.** `idm_head_v1` was trained on pod3's
     tr_a/tr_b/cm tags and has seen NONE of these 104 episodes, so ALL 40
     PhysicalAI clips are held out from it, not just the 14 in the v2 val split.
     n: 14 -> 40.
  2. **The regression-to-the-prior term swamps it.** rel_bias vs the clip's mean
     speed is r = -0.605 [-0.847, -0.311], CI-SEPARATED — that is the measured
     shrinkage (gain 0.830) of `IDM_DIAGNOSIS.md` §5.3. Any camera-height effect
     rides on top of it, so the admissible test is a PARTIAL correlation
     controlling for v_mean, not a marginal one.
  3. **A correlation is not the hypothesis.** Equation (1) of `idm3_geom.py`
     says v = (f*h)*PHI, so a head that assumes a single height h_bar emits
     v_hat = v * h_bar/h. The DIRECT test is therefore: does multiplying the
     prediction by h/h_bar reduce the error? That is a one-parameter, no-fit
     correction with a sign fixed IN ADVANCE by the physics, and it is scored
     with a paired episode-cluster bootstrap on the same windows.

BOTH OUTCOMES ARE COMMITTED IN ADVANCE (brief: pre-register, never re-define):
  * PASS  = the physical correction v_hat * (h/h_bar) reduces paired speed MAE
            with a CI that excludes 0, AND the partial correlation of rel_bias
            with h controlling for v_mean is negative and CI-separated.
  * FAIL  = it does not. Then the per-clip speed bias is NOT camera height, the
            PI's geometry hypothesis is REFUTED for the speed channel as
            mediated by mount height, and I report that plainly rather than
            hunting for a variant that fires.

The deliberately-failing input required by the brief is included: the SAME
correction applied with the heights SHUFFLED across clips must not help.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/root/idm2")
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")

import idm2_lib as L          # noqa: E402
import idm_head as ih         # noqa: E402
import idm3_geom as GEO       # noqa: E402
from taniteval import ci as tci  # noqa: E402

DEV = "cuda"
KBUILD = 8


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@torch.no_grad()
def a0_predict_all(tags, batch=1024):
    """A0's speed prediction on EVERY window of `tags` (stride 2), plus gt."""
    d = torch.load("/root/idmval/idm_head_v1.pt", weights_only=False)
    h = ih.IDMHead(**d["config"]["head_kwargs"]).to(DEV)
    h.load_state_dict(d["state_dict"])
    h.eval()
    st = L.build_set(tags, k=KBUILD, stride=2, want_seq=False)
    Z = st["Z"][:, KBUILD - 4:KBUILD + 5].to(DEV).float()
    S = []
    for i in range(0, Z.shape[0], batch):
        S.append(h(Z[i:i + batch])["scalars"].cpu())
    del Z, h
    torch.cuda.empty_cache()
    return torch.cat(S).numpy().astype(np.float64), st


def partial_corr(y, x, z):
    """corr(y, x | z) — residualise both on [1, z] then correlate."""
    A = np.stack([np.ones_like(z), z], 1)
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ry, rx)[0, 1])


def boot_stat(fn, tags_of_unit, n_units, n_boot=2000, seed=0):
    """Episode-cluster bootstrap of an arbitrary statistic over CLIPS."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n_units, n_units)
        v = fn(i)
        if np.isfinite(v):
            vals.append(v)
    v = np.array(vals)
    return {"point": float(fn(np.arange(n_units))),
            "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)), "n_boot": int(v.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/idm3/out/geomtest_v3.json")
    a = ap.parse_args()
    res = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    pai = sorted(t for t in L.all_tags() if t.startswith("pai_"))
    log(f"PhysicalAI clips held out from A0: {len(pai)}")
    S, st = a0_predict_all(pai)
    gt = st["S"].numpy().astype(np.float64)
    eid = st["eid"]
    tab = GEO.load_table()

    # ---- per-clip aggregates ----------------------------------------------
    rows = []
    for t in pai:
        m = eid == t
        if m.sum() == 0:
            continue
        g = GEO.geom_for_tag(t, tab)
        vg, vp = gt[m, 0], S[m, 0]
        mu = float(vg.mean())
        if mu < 1.0:
            continue
        rows.append({"tag": t, "n": int(m.sum()), "v_mean": mu,
                     "bias": float((vp - vg).mean()),
                     "rel_bias": float((vp - vg).mean() / mu),
                     "cam_h": g["cam_h_m"], "rig": g["rig"],
                     "pitch": g["pitch_down_rad"],
                     "metric_gain": g["f_eff_px"] * g["cam_h_m"]})
    n = len(rows)
    h = np.array([r["cam_h"] for r in rows])
    vm = np.array([r["v_mean"] for r in rows])
    rb = np.array([r["rel_bias"] for r in rows])
    log(f"usable clips n={n}  cam_h {h.min():.3f}-{h.max():.3f} "
        f"(cv {h.std()/h.mean():.3%})  v_mean {vm.min():.1f}-{vm.max():.1f}")

    # ---- (2) partial correlation ------------------------------------------
    marg = float(np.corrcoef(rb, h)[0, 1])
    part = partial_corr(rb, h, vm)
    part_log = partial_corr(rb, np.log(h), np.log(vm))
    ci_m = boot_stat(lambda i: float(np.corrcoef(rb[i], h[i])[0, 1]), None, n)
    ci_p = boot_stat(lambda i: partial_corr(rb[i], h[i], vm[i]), None, n)
    res["correlation"] = {
        "n_clips": n,
        "cam_h_cv": float(h.std() / h.mean()),
        "marginal_relbias_vs_camh": {"r": marg, **ci_m},
        "PARTIAL_relbias_vs_camh_given_vmean": {"r": part, **ci_p},
        "partial_log_log": part_log,
        "relbias_vs_vmean": float(np.corrcoef(rb, vm)[0, 1]),
        "camh_vs_vmean": float(np.corrcoef(h, vm)[0, 1]),
        "PREDICTED_SIGN": "negative (v_hat = v * h_bar/h  =>  rel_bias falls with h)",
    }
    log("marginal  r(rel_bias, h)          = %+.3f [%+.3f,%+.3f]"
        % (ci_m["point"], ci_m["lo"], ci_m["hi"]))
    log("PARTIAL   r(rel_bias, h | v_mean) = %+.3f [%+.3f,%+.3f]   (predicted: NEGATIVE)"
        % (ci_p["point"], ci_p["lo"], ci_p["hi"]))
    log("nuisance  r(rel_bias, v_mean)     = %+.3f ;  r(h, v_mean) = %+.3f"
        % (res["correlation"]["relbias_vs_vmean"],
           res["correlation"]["camh_vs_vmean"]))

    # ---- (3) THE DIRECT TEST: apply the physical correction ---------------
    hbar = float(np.exp(np.mean(np.log(h))))          # geometric mean height
    hw = np.array([GEO.geom_for_tag(t, tab)["cam_h_m"] for t in eid])
    base = np.abs(S[:, 0] - gt[:, 0])
    corr = np.abs(S[:, 0] * (hw / hbar) - gt[:, 0])
    inv = np.abs(S[:, 0] * (hbar / hw) - gt[:, 0])    # the OPPOSITE sign

    rng = np.random.default_rng(7)
    perm = rng.permutation(len(pai))
    shuf_map = {t: GEO.geom_for_tag(pai[perm[i]], tab)["cam_h_m"]
                for i, t in enumerate(pai)}
    hs = np.array([shuf_map[t] for t in eid])
    shuf = np.abs(S[:, 0] * (hs / hbar) - gt[:, 0])

    res["h_bar_geometric_mean_m"] = hbar
    res["direct_correction"] = {}
    for nm, arr in (("correct_h_over_hbar", corr),
                    ("inverse_hbar_over_h", inv),
                    ("SHUFFLED_negative_control", shuf)):
        d = tci.paired_episode_cluster_bootstrap(arr, base, eid, n_boot=2000,
                                                 seed=0, reduce="mean")
        sep = (d["lo"] > 0) or (d["hi"] < 0)
        res["direct_correction"][nm] = {
            "delta_mae": float(d["mean"]) if "mean" in d else float(d.get("point", np.nan)),
            "lo": float(d["lo"]), "hi": float(d["hi"]), "separated": bool(sep),
            "mae_base": float(base.mean()), "mae_corrected": float(arr.mean())}
        log("DIRECT %-28s dMAE %+.4f [%+.4f,%+.4f] %s  (%.3f -> %.3f m/s)"
            % (nm, res["direct_correction"][nm]["delta_mae"], d["lo"], d["hi"],
               "SEPARATED" if sep else "not sep", base.mean(), arr.mean()))

    # ---- how big COULD it be? the oracle per-clip scale --------------------
    orc = np.zeros_like(base)
    for r in rows:
        m = eid == r["tag"]
        k = float((S[m, 0] * gt[m, 0]).sum() / max((S[m, 0] ** 2).sum(), 1e-9))
        orc[m] = np.abs(S[m, 0] * k - gt[m, 0])
    d = tci.paired_episode_cluster_bootstrap(orc, base, eid, n_boot=2000, seed=0,
                                             reduce="mean")
    res["oracle_per_clip_scale"] = {"delta_mae": float(d.get("mean", np.nan)),
                                    "lo": float(d["lo"]), "hi": float(d["hi"]),
                                    "mae": float(orc.mean())}
    log("ORACLE per-clip scale  dMAE %+.4f [%+.4f,%+.4f]  (%.3f -> %.3f m/s)"
        % (res["oracle_per_clip_scale"]["delta_mae"], d["lo"], d["hi"],
           base.mean(), orc.mean()))

    # is the oracle scale factor explained by h?
    ks = []
    for r in rows:
        m = eid == r["tag"]
        ks.append(float((S[m, 0] * gt[m, 0]).sum() / max((S[m, 0] ** 2).sum(), 1e-9)))
    ks = np.array(ks)
    res["oracle_k_vs_geometry"] = {
        "k_mean": float(ks.mean()), "k_std": float(ks.std()),
        "k_range": [float(ks.min()), float(ks.max())],
        "r_k_vs_camh": float(np.corrcoef(ks, h)[0, 1]),
        "r_k_vs_vmean": float(np.corrcoef(ks, vm)[0, 1]),
        "partial_k_camh_given_vmean": partial_corr(ks, h, vm),
        "PREDICTED": "if scale error IS mount height, k should track h with r>0",
    }
    log("ORACLE k: mean %.3f std %.3f range [%.3f,%.3f] | r(k,h)=%+.3f "
        "r(k,v_mean)=%+.3f partial(k,h|v)=%+.3f"
        % (ks.mean(), ks.std(), ks.min(), ks.max(),
           res["oracle_k_vs_geometry"]["r_k_vs_camh"],
           res["oracle_k_vs_geometry"]["r_k_vs_vmean"],
           res["oracle_k_vs_geometry"]["partial_k_camh_given_vmean"]))

    res["per_clip"] = rows
    L.jdump(res, a.out)


if __name__ == "__main__":
    main()
