"""T1 — kinematic clip + fan-size sweep over ALREADY-CACHED fans. ZERO GPU.

Two published motivations, both DEMONSTRATED:
  * CoverNet — current-state-conditioned set construction: minADE5 2.62 -> 2.02
    and the same coverage with ~half the trajectories.
  * LLM-Assist — PDM-Closed at 15 proposals scores 92.51 and at 8,505 scores
    77.78, with progress RISING while TTC and comfort collapse. A big fan is an
    adversarial search against the scorer's approximation error.
Published fan sizes: PDM 15 · PLUTO 20 · DiffusionDrive 20 (saturated) ·
Slow-Brain plateau K~18-24. Ours: 256, single-stage, unfiltered.

⚠️ THE BAND COMES FROM PHYSICS AND OUR OWN HEAD, NEVER FROM HELD-OUT ERROR.
``FlagshipV15Head.select`` already computes a reachable-speed clamp for the GOAL
(``reach = sel_accel_max * horizons[-1] * 0.1`` = 2.5 m/s^2 * 2.0 s = 5.0 m/s,
``flagship_v15.py:139,455``). T1 applies the IDENTICAL clamp to the CANDIDATES.
Nothing is tuned on ade_0_2s.

TWO SURFACES, because of what is on this host:
  A. ``taniteval/results/fan_refc-xl-30k.pt`` — REF-C-XL's EMITTED 256-candidate
     fan for all 881 canonical val windows, WITH its real per-candidate
     ``logits``. Geometry + a realisable picker => the full test.
  B. ``…/v5-imagination-selection/raw/v5_v4_windows_reduced.pt`` — flagship-v4's
     real per-candidate ``fan_err4`` on the same 881 windows. No geometry and no
     usable score ranking (its ``base_rank`` is rank-0-real + fan order after,
     by that script's own documentation), so v4 supports the COVERAGE side of
     the sweep only. Stated, not glossed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "taniteval"))
from taniteval import ci as _ci  # noqa: E402

SEL_ACCEL_MAX = 2.5      # flagship_v15.V4Config.sel_accel_max
HORIZON_S = 2.0


def ade_of(fan_err, pick):
    return np.take_along_axis(fan_err, pick[:, None], 1)[:, 0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    # ---------------- surface A: REF-C-XL emitted fan + real logits ----------
    fd = torch.load(REPO / "taniteval/results/fan_refc-xl-30k.pt",
                    map_location="cpu", weights_only=False)
    fan = fd["fan"].double().numpy()               # [W, C, 4, 2]
    gt = fd["gt"].double().numpy()                 # [W, 4, 2]
    logit = fd["logits"].double().numpy()          # [W, C]
    v0 = fd["v0"].double().numpy()                 # [W]
    eid = [str(e) for e in fd["eid"]]
    W, C = fan.shape[0], fan.shape[1]
    fe = np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)   # [W, C]

    # fidelity: the dump's own recorded pick must be the argmax of its logits
    fid = float((logit.argmax(1) == fd["sel"].numpy()).mean())
    base_pick = logit.argmax(1)
    base_ade = ade_of(fe, base_pick)

    # implied terminal mean speed of each candidate over the 2 s horizon
    v_term = np.linalg.norm(fan[:, :, -1, :], axis=-1) / HORIZON_S      # [W, C]
    reach = SEL_ACCEL_MAX * HORIZON_S                                   # 5.0 m/s
    lo = np.maximum(v0 - reach, 0.0)[:, None]
    hi = (v0 + reach)[:, None]
    keep = (v_term >= lo) & (v_term <= hi)

    def masked_pick(mask):
        m = np.where(mask, logit, -np.inf)
        dead = ~mask.any(1)
        p = m.argmax(1)
        p[dead] = base_pick[dead]                  # empty set -> unfiltered pick
        return p, dead

    clip_pick, dead = masked_pick(keep)
    clip_ade = ade_of(fe, clip_pick)
    orc = fe.argmin(1)
    orc_kept = np.take_along_axis(keep, orc[:, None], 1)[:, 0]
    fe_masked = np.where(keep, fe, np.inf)

    clip = dict(
        band=dict(rule="v_term in [max(0, v0 - reach), v0 + reach], "
                       "reach = sel_accel_max * horizon = 2.5 * 2.0 = 5.0 m/s",
                  provenance="flagship_v15.py:139,455 — the head's OWN goal "
                             "clamp, applied to the candidates; NOT tuned"),
        frac_candidates_removed=round(float(1.0 - keep.mean()), 4),
        frac_windows_with_empty_survivor_set=round(float(dead.mean()), 4),
        oracle_survives_frac=round(float(orc_kept.mean()), 4),
        oracle_ade_unfiltered=round(float(fe.min(1).mean()), 4),
        oracle_ade_after_clip=round(
            float(np.where(dead, fe.min(1), fe_masked.min(1)).mean()), 4),
        as_trained_ade=round(float(base_ade.mean()), 4),
        clipped_ade=round(float(clip_ade.mean()), 4),
        paired=_ci.paired_episode_cluster_bootstrap(clip_ade, base_ade, eid,
                                                    n_boot=2000),
        miss_at_2m_as_trained=round(float((base_ade > 2.0).mean()), 4),
        miss_at_2m_clipped=round(float((clip_ade > 2.0).mean()), 4))

    # ---------------- fan-size sweep, surface A (coverage + realisable pick) -
    rng = np.random.default_rng(0)
    Ks = [4, 8, 16, 20, 32, 64, 128, 256]
    sweepA = {}
    for K in Ks:
        oa, pa, ma = [], [], []
        for s in range(8 if K < C else 1):
            if K == C:
                sub = np.tile(np.arange(C), (W, 1))
            else:
                sub = np.argsort(rng.random((W, C)), axis=1)[:, :K]
            fs = np.take_along_axis(fe, sub, 1)
            ls = np.take_along_axis(logit, sub, 1)
            pick = np.take_along_axis(sub, ls.argmax(1)[:, None], 1)[:, 0]
            ad = ade_of(fe, pick)
            oa.append(fs.min(1).mean()); pa.append(ad.mean())
            ma.append((ad > 2.0).mean())
        sweepA[f"K{K}"] = dict(
            oracle_ade=round(float(np.mean(oa)), 4),
            realisable_pick_ade=round(float(np.mean(pa)), 4),
            realisable_pick_ade_sd_over_seeds=round(float(np.std(pa)), 4),
            miss_at_2m=round(float(np.mean(ma)), 4))

    # ---------------- fan-size sweep, surface B (v4, COVERAGE only) ----------
    v4 = torch.load(REPO / "TanitAD Research Hub/Architecture & Inference/"
                    "Implementation/incoming/2026-07-26-v5-imagination-selection/"
                    "raw/v5_v4_windows_reduced.pt",
                    map_location="cpu", weights_only=False)
    fe4 = v4["fan_err4"].double().numpy()
    sel4 = v4["ref_sel_idx"].numpy()
    rng2 = np.random.default_rng(0)
    sweepB = {}
    for K in Ks:
        oa = []
        for s in range(8 if K < 256 else 1):
            sub = (np.tile(np.arange(256), (fe4.shape[0], 1)) if K == 256
                   else np.argsort(rng2.random(fe4.shape), axis=1)[:, :K])
            oa.append(np.take_along_axis(fe4, sub, 1).min(1).mean())
        sweepB[f"K{K}"] = dict(oracle_ade=round(float(np.mean(oa)), 4))

    res = dict(
        what="T1 — kinematic clip + fan-size sweep on cached fans (zero GPU)",
        surfaceA=dict(
            source="taniteval/results/fan_refc-xl-30k.pt (refc-xl-30k, 881 "
                   "canonical val windows, 256 emitted candidates + real logits)",
            selftest_pick_is_argmax_of_logits=round(fid, 6),
            kinematic_clip=clip, fan_size_sweep=sweepA),
        surfaceB=dict(
            source="v5_v4_windows_reduced.pt (flagship-v4-fromscratch-30k, "
                   "same 881 windows, real per-candidate fan_err4)",
            limitation="no fan geometry and no usable score ranking on this "
                       "host, so only the COVERAGE side of the sweep is "
                       "computable for v4",
            as_trained_ade=round(float(ade_of(fe4, sel4).mean()), 4),
            oracle_ade_full_fan=round(float(fe4.min(1).mean()), 4),
            fan_size_sweep_coverage_only=sweepB),
        reading=dict(
            published_anchor="LLM-Assist: PDM-Closed 15 proposals 92.51 -> "
                             "8,505 proposals 77.78 CLS-NR (PDF-VERBATIM)",
            note="a shrinking oracle is the COST of a smaller fan; the "
                 "realisable pick is the BENEFIT. Only surface A can show both."))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
