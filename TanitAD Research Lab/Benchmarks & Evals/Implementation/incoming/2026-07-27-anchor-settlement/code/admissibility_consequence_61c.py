"""The ADMISSIBILITY consequence on the SECOND comma substrate — measured
through the shipped API, not through a private reimplementation.

WHY A SECOND SUBSTRATE. `resettle_anchor.py` measures admissibility on the
ANCHOR's corpus (`comma2k19-val-76b6e94a97a1`), where it drops 50 of 2,992
windows and moves R2 by +0.007 — nearly a no-op. That number alone would invite
exactly the wrong conclusion ("admissibility does not matter"). On
`comma2k19-val-61c46fca8f7f` `cm_[40:70]` — `idm_head_v1`'s own held-out comma
clips, which the content probe confirms are disjoint from its training clips —
the same rule removes **every** impossible label and collapses the label's own
std by a factor of 20. **The consequence is corpus-dependent; the correctness is
not.** Both are reported, per corpus, never pooled.

WHAT THIS ALSO IS: the first consumer of `comma2k19.yaw_rate_from_heading` /
`heading_admissible_centers`. The API is exercised on real corpus data here, not
only on fixtures — an admissibility contract nobody calls is the defect it was
written to fix.

⛔ Nothing retrained, nothing re-encoded: the latents produced by the
`heading-default` pass are reused (they are a pure function of an md5-pinned
encoder and an md5-pinned cache). ⛔ pod1/pod2/pod3 untouched — dev box only.

ESTIMATOR: `taniteval.ci.episode_cluster_bootstrap` (callable r2 reducer),
B = 2000, unit = the episode. `overlapping_holdout_se` is never called.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[6]
for p in (REPO / "stack", REPO / "stack" / "scripts", REPO / "taniteval"):
    sys.path.insert(0, str(p))

import idm_head as ih                                              # noqa: E402
from taniteval import ci as tci                                    # noqa: E402
from tanitad.data.comma2k19 import (HEADING_OBSERVABLE_V_MPS,      # noqa: E402
                                    KEEP_INADMISSIBLE_YAW_REASON,
                                    admissible_from_poses,
                                    heading_admissible_centers,
                                    hold_heading_through_standstill,
                                    yaw_rate_from_heading)

HEAD_PT = (REPO / "TanitAD Research Hub" / "Architecture & Inference" /
           "Implementation" / "incoming" / "2026-07-25-idm-youtube-validation" /
           "idm_head_v1.pt")
HEAD_MD5 = "fa4462f0b898b036be729c790278b823"
LAT_DIR = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
               r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
               r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad\lat_cm40_69")
OUT = HERE.parent / "raw" / "admissibility_consequence_61c.json"

CM_LO, CM_HI = 40, 70
K, STRIDE, DT = 4, 2, 0.1
N_BOOT = 2000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def md5_of(p, chunk=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def spearman(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def chan_metrics(pred, gt):
    p = np.asarray(pred, np.float64); g = np.asarray(gt, np.float64)
    err = p - g
    mad = float(np.median(np.abs(g - np.median(g))))
    return {"r2": float(1.0 - (err ** 2).sum()
                        / max(((g - g.mean()) ** 2).sum(), 1e-12)),
            "rho": spearman(p, g), "mae": float(np.abs(err).mean()),
            "medae": float(np.median(np.abs(err))),
            "nmedae": float(np.median(np.abs(err)) / max(mad, 1e-12)),
            "gt_std": float(g.std()), "gt_mad": mad,
            "n_impossible_gt1p5": int((np.abs(g) > 1.5).sum()),
            "n": int(g.size)}


def boot_r2(pred, gt, eid):
    p = np.asarray(pred, np.float64); g = np.asarray(gt, np.float64)

    def _r2(idx):
        i = idx.astype(np.int64); gg, pp = g[i], p[i]
        return float(1.0 - ((pp - gg) ** 2).sum()
                     / max(((gg - gg.mean()) ** 2).sum(), 1e-12))
    _r2.__name__ = "r2"
    return tci.episode_cluster_bootstrap(np.arange(p.size, dtype=np.float64),
                                         eid, reduce=_r2, n_boot=N_BOOT, seed=0)


@torch.no_grad()
def main():
    assert md5_of(HEAD_PT) == HEAD_MD5, "⛔ deployed head md5 mismatch"
    ci_md5 = md5_of(tci.__file__)
    assert Path(tci.__file__).resolve().is_relative_to((REPO / "taniteval").resolve()), \
        f"⛔ estimator not from the repo: {tci.__file__}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tags = [f"cm_{i:05d}" for i in range(CM_LO, CM_HI)]
    missing = [t for t in tags if not (LAT_DIR / f"{t}.pt").exists()]
    assert not missing, (f"latents missing: {missing[:3]}… regenerate with "
                         f"…/2026-07-27-heading-default/code/"
                         f"rescore_idm_head_v1_comma.py (~90 s)")

    d = torch.load(HEAD_PT, map_location="cpu", weights_only=False)
    head = ih.IDMHead(**d["config"]["head_kwargs"]).to(device).eval()
    head.load_state_dict(d["state_dict"])

    P, LEG, REP, ADM, EID, VMAXOBS = [], [], [], [], [], {}
    for tag in tags:
        L = torch.load(LAT_DIR / f"{tag}.pt", weights_only=False)
        z = L["z"].float(); poses = L["poses"].float()
        t = ih.valid_centers(z.shape[0], K, ih.DEFAULT_HORIZONS, STRIDE)
        offs = torch.arange(-K, K + 1)
        Zwin = z[t[:, None] + offs[None, :]]

        yaw_raw = poses[:, 2].numpy().astype(np.float64)
        v = poses[:, 3].numpy().astype(np.float64)
        yaw_fix, obs = hold_heading_through_standstill(
            yaw_raw, v, v_min=HEADING_OBSERVABLE_V_MPS)
        # the shipped API — observability is a pure function of the poses …
        assert np.array_equal(obs, admissible_from_poses(poses.numpy()))
        tn = t.numpy()
        # … and this is the ONE definition of the centred-difference rule
        adm = heading_admissible_centers(obs, tn)

        # LEGACY and REPAIRED both need the pre-2026-07-27 policy, because
        # reproducing a published number is exactly the acknowledged case.
        leg, adm2 = yaw_rate_from_heading(
            yaw_raw, obs, tn, dt=DT, admissibility="keep",
            allow_inadmissible=True, reason=KEEP_INADMISSIBLE_YAW_REASON)
        rep, _ = yaw_rate_from_heading(
            yaw_fix, obs, tn, dt=DT, admissibility="keep",
            allow_inadmissible=True, reason=KEEP_INADMISSIBLE_YAW_REASON)
        assert np.array_equal(adm, adm2)

        out = []
        for i in range(0, Zwin.shape[0], 512):
            out.append(head(Zwin[i:i + 512].to(device))["scalars"].cpu())
        P.append(torch.cat(out)[:, 1].double().numpy())
        LEG.append(leg); REP.append(rep); ADM.append(adm)
        EID += [tag] * tn.size
        VMAXOBS[tag] = {"n_observable_frames": int(obs.sum()),
                        "v_max": float(v.max()),
                        "n_windows": int(tn.size),
                        "n_admissible_windows": int(adm.sum())}

    P = np.concatenate(P); LEG = np.concatenate(LEG); REP = np.concatenate(REP)
    ADM = np.concatenate(ADM); EID = np.array(EID)
    assert P.size == 4140, P.size
    log(f"windows {P.size}  admissible {int(ADM.sum())}  "
        f"dropped {int((~ADM).sum())}")

    # ⭐ the DEFAULT policy, on the REAL substrate: NaN must reach the metric.
    # ⛔ Run over EVERY clip, not the first one — `cm_00040` happens to be fully
    # observable, so a first-clip check would have reported "no NaN" and quietly
    # contradicted the prose. A check that cannot see the condition it is cited
    # for is class C13.
    nan_counts, nan_total, clips_with_nan = {}, 0, 0
    for tag in tags:
        L = torch.load(LAT_DIR / f"{tag}.pt", weights_only=False)
        poses = L["poses"].float()
        obs = admissible_from_poses(poses.numpy())
        tn = ih.valid_centers(poses.shape[0], K, ih.DEFAULT_HORIZONS,
                              STRIDE).numpy()
        rate, _ = yaw_rate_from_heading(poses[:, 2].double().numpy(), obs, tn,
                                        dt=DT)          # DEFAULT policy
        c = int(np.isnan(rate).sum())
        nan_counts[tag] = c
        nan_total += c
        clips_with_nan += int(c > 0)
    # and the payoff, on the clip that caused C42: a metric over it is NaN
    worst = max(nan_counts, key=nan_counts.get)
    Lw = torch.load(LAT_DIR / f"{worst}.pt", weights_only=False)
    pw = Lw["poses"].float()
    rw, _ = yaw_rate_from_heading(
        pw[:, 2].double().numpy(), admissible_from_poses(pw.numpy()),
        ih.valid_centers(pw.shape[0], K, ih.DEFAULT_HORIZONS, STRIDE).numpy(),
        dt=DT)
    mean_abs_worst = float(np.mean(np.abs(rw)))

    res = {
        "what": "the ADMISSIBILITY consequence on idm_head_v1's OWN held-out "
                "comma clips — the second corpus, measured through the shipped "
                "comma2k19 admissibility API",
        "date": "2026-07-27", "agent": "anchor-settlement",
        "evidence_class": "MEASURED (ours; dev box RTX 4060, this script)",
        "tier": "decision-grade for the comma channel on THIS substrate",
        "host": "dev box only — pod1/pod2/pod3 not touched",
        "estimator": "taniteval.ci.episode_cluster_bootstrap (callable r2 "
                     "reducer), B=2000, unit = the episode. "
                     "overlapping_holdout_se NOT used.",
        "ci_py": tci.__file__, "ci_py_md5": ci_md5,
        "substrate": {
            "cache": "comma2k19-val-61c46fca8f7f",
            "tags": f"cm_[{CM_LO}:{CM_HI}]", "n_clips": len(tags),
            "n_windows": int(P.size), "k": K, "stride": STRIDE,
            "content_disjoint_from_training": True,
            "content_disjointness_source": "raw/anchor_overlap.json -> "
                                           "self_overlap_61c = 0 by sha256 of "
                                           "raw pose bytes AND raw frame bytes",
        },
        "label_protocols": {
            "legacy": "cached heading (arctan2 of ENU velocity), all windows",
            "repaired": f"hold_heading_through_standstill, v_min="
                        f"{HEADING_OBSERVABLE_V_MPS}, all windows — the "
                        f"anchor's protocol",
            "strict_admissible": "repaired AND heading_admissible_centers "
                                 "(observable at t-1, t, t+1)",
        },
        "api_check": {
            "swept": "all 30 clips (NOT the first — cm_00040 is fully "
                     "observable, so a first-clip check would report 'no NaN' "
                     "and quietly contradict the claim: class C13)",
            "default_policy_returns_nan_on_real_data": bool(nan_total > 0),
            "n_nan_windows_total": int(nan_total),
            "n_clips_with_any_nan": int(clips_with_nan),
            "worst_clip": worst,
            "n_nan_in_worst_clip": int(nan_counts[worst]),
            "mean_abs_yaw_rate_over_worst_clip_is_nan": bool(
                math.isnan(mean_abs_worst)),
            "reading": "the DEFAULT path produces NaN where the label is "
                       "undefined, so a metric over it is visibly undefined "
                       "rather than plausible-looking",
        },
        "results": {}, "per_episode": VMAXOBS,
    }

    for name, gt, mask in (("legacy", LEG, np.ones_like(ADM)),
                           ("repaired", REP, np.ones_like(ADM)),
                           ("strict_admissible", REP, ADM)):
        m = mask.astype(bool)
        b = chan_metrics(P[m], gt[m])
        b["r2_ci"] = boot_r2(P[m], gt[m], EID[m])
        b["n_episodes"] = int(len(set(EID[m])))
        b["n_dropped"] = int((~m).sum())
        res["results"][name] = b
        log(f"  {name:18s} n={b['n']:5d} R2 {b['r2']:+.6f} "
            f"CI[{b['r2_ci']['lo']:+.3f},{b['r2_ci']['hi']:+.3f}] "
            f"rho {b['rho']:+.4f} nMedAE {b['nmedae']:.3f} "
            f"gt_std {b['gt_std']:.4f} imposs {b['n_impossible_gt1p5']}")

    res["wholly_stationary_clips"] = {
        t: s for t, s in VMAXOBS.items() if s["n_observable_frames"] == 0}
    Path(OUT).write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
