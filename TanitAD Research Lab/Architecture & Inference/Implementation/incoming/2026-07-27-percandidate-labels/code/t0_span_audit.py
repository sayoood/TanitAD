"""T0 — is the 181 km/h span in the ANCHORS or manufactured by the OFFSET HEAD?

The two answers imply different fixes (rebuild the vocabulary vs clamp the
refinement), so this runs first and it is pure measurement.

Three independent legs, none of which needs a pod:

  L1 STRUCTURAL.  ``build_refc_anchors.py --data-root`` calls
     ``refc.furthest_point_sample(pool, n)``, whose last line is
     ``return pool[chosen]`` — FPS *selects members of the pool*, it does not
     synthesise centroids (contrast k-means). The pool is
     ``refb_labels.waypoint_targets`` over every window of the cached corpus.
     ⇒ every anchor IS, verbatim, one real human 2 s trajectory. An anchor can
     therefore only be a 181 km/h plan if a 181 km/h human window exists.

  L2 EMPIRICAL, anchor side.  Rebuild the pool + a 256-anchor FPS vocabulary
     from a real PhysicalAI episode cache and measure the span directly.

  L3 EMPIRICAL, offset side.  The committed REF-C-XL fan dump
     ``taniteval/results/fan_refc-xl-30k.pt`` holds the **emitted** 256-candidate
     fan (``fan = x_in + offset``, ``refc.py:539-540`` — the identical mechanism
     v4 inherits) for all 881 canonical val windows. Measure its span.

⚠️ WHAT THIS CANNOT DO. v4's own emitted fan and its ``flagship_v4_anchors_dense.pt``
live on the eval pod, and the brief forbids touching a pod. L3 measures the
*mechanism* on a matched architecture (same decoder, same 256 anchors, same
unbounded ``offset_head``), not on v4's own tensor. Stated, not glossed.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))

from pcl_common import N_STEPS, TRAIN_DIR  # noqa: E402


def waypoint_pool(ep_dir: Path, horizons: tuple[int, ...],
                  max_eps: int = 0) -> torch.Tensor:
    """refb_labels.waypoint_targets over EVERY window — build_refc_anchors's pool."""
    import refb_labels
    files = sorted(glob.glob(str(ep_dir / "ep_*.pt")))
    if max_eps:
        files = files[:max_eps]
    max_h = max(horizons)
    pool = []
    for f in files:
        poses = torch.load(f, map_location="cpu", weights_only=False)["poses"].float()
        n = poses.shape[0] - max_h
        if n <= 0:
            continue
        idx = torch.arange(n)
        fut = torch.stack([poses[idx + k] for k in range(1, max_h + 1)], dim=1)
        pool.append(refb_labels.waypoint_targets(poses[idx], fut, horizons))
    return torch.cat(pool, 0)


def span_stats(term_xy: np.ndarray) -> dict:
    """Stats of a set of 2 s terminal ego-frame waypoints [N, 2]."""
    along = term_xy[..., 0]
    disp = np.linalg.norm(term_xy, axis=-1)
    v = disp / 2.0                                    # mean speed over 2 s [m/s]
    return dict(
        n=int(term_xy.shape[0]),
        along_min_m=round(float(along.min()), 3),
        along_max_m=round(float(along.max()), 3),
        along_span_m=round(float(along.max() - along.min()), 3),
        along_p99_m=round(float(np.percentile(along, 99)), 3),
        disp_max_m=round(float(disp.max()), 3),
        implied_mean_speed_max_ms=round(float(v.max()), 3),
        implied_mean_speed_max_kmh=round(float(v.max() * 3.6), 2),
        implied_mean_speed_p99_kmh=round(float(np.percentile(v, 99) * 3.6), 2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eps", type=int, default=0)
    ap.add_argument("--n-anchors", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors-out", default=None)
    a = ap.parse_args(argv)

    from tanitad.refs.refc import furthest_point_sample

    # ---- L2: the anchor side ------------------------------------------------
    hz = tuple(range(1, N_STEPS + 1))
    pool = waypoint_pool(TRAIN_DIR, hz, a.max_eps)          # [M, 20, 2]
    sub = pool
    if pool.shape[0] > 200_000:                             # build_refc_anchors cap
        g = torch.Generator().manual_seed(0)
        sub = pool[torch.randperm(pool.shape[0], generator=g)[:200_000]]
    anchors = furthest_point_sample(sub, a.n_anchors, seed=0).contiguous()
    if a.anchors_out:
        torch.save({"anchors": anchors, "method": "fps",
                    "horizons": list(hz), "n_anchors": a.n_anchors,
                    "pool_size": int(sub.shape[0]),
                    "source": str(TRAIN_DIR), "seed": 0}, a.anchors_out)

    # anchors are members of the pool -> verify, do not assume.
    # (torch.cdist in fp32 over a 40-dim vector with ~40 m entries carries ~4e-2
    #  of rounding, so an approximate distance CANNOT establish exact membership.
    #  Compare in float64 against each anchor's own nearest pool row instead.)
    flat_pool = sub.reshape(sub.shape[0], -1).double()
    flat_anc = anchors.reshape(anchors.shape[0], -1).double()
    membership_max_dist = 0.0
    n_exact = 0
    for i in range(flat_anc.shape[0]):
        dd = (flat_pool - flat_anc[i]).abs().max(dim=1).values
        j = int(dd.argmin())
        membership_max_dist = max(membership_max_dist, float(dd[j]))
        n_exact += int(bool(torch.equal(flat_pool[j], flat_anc[i])))

    # ---- L3: the offset side (committed REF-C-XL emitted fan) ---------------
    fan_path = REPO / "taniteval/results/fan_refc-xl-30k.pt"
    fd = torch.load(fan_path, map_location="cpu", weights_only=False)
    fan = fd["fan"].numpy()                                  # [881, 256, 4, 2]
    gt = fd["gt"].numpy()                                    # [881, 4, 2]
    term_fan = fan[:, :, -1, :].reshape(-1, 2)
    per_window_along_span = (fan[:, :, -1, 0].max(1) - fan[:, :, -1, 0].min(1))

    res = dict(
        what="T0 span audit — anchors (demonstration-derived) vs the "
             "unbounded offset head",
        L1_structural=dict(
            claim="anchors are VERBATIM real human 2 s trajectories",
            evidence=("refc.furthest_point_sample returns pool[chosen] — it "
                      "SELECTS pool members, never synthesises a centroid; the "
                      "pool is refb_labels.waypoint_targets over every window "
                      "of the cached corpus (build_refc_anchors.episode_traj_pool)"),
            verified_numerically=dict(
                max_linf_from_anchor_to_its_nearest_pool_member=round(
                    membership_max_dist, 12),
                n_anchors_bitwise_identical_to_a_pool_row=int(n_exact),
                n_anchors=int(anchors.shape[0]),
                exact_membership=bool(n_exact == anchors.shape[0]))),
        L2_anchor_side=dict(
            corpus=str(TRAIN_DIR),
            warning=("NOT the parity key e438721ae894 — this is the dev-box "
                     "cache 14231cd29c74. A corpus-property measurement only."),
            pool=span_stats(pool[:, -1, :].numpy()),
            anchors_fps256=span_stats(anchors[:, -1, :].numpy())),
        L3_offset_side=dict(
            source=str(fan_path),
            arm="refc-xl-30k (same V15Decoder `x = x_in + offset`, same 256 "
                "anchors, same unbounded offset_head that v4 inherits)",
            windows=int(fan.shape[0]), candidates=int(fan.shape[1]),
            emitted_fan=span_stats(term_fan),
            realised_future_gt=span_stats(gt[:, -1, :]),
            mean_per_window_along_span_m=round(
                float(per_window_along_span.mean()), 3),
            max_per_window_along_span_m=round(
                float(per_window_along_span.max()), 3)),
    )
    p = span_stats(pool[:, -1, :].numpy())
    f = span_stats(term_fan)
    gt_max_along = float(gt[:, -1, 0].max())
    fan_along = fan[:, :, -1, 0]
    res["verdict"] = dict(
        anchor_max_kmh=p["implied_mean_speed_max_kmh"],
        emitted_fan_max_kmh=f["implied_mean_speed_max_kmh"],
        ratio_emitted_over_demonstration=round(
            f["implied_mean_speed_max_kmh"] / p["implied_mean_speed_max_kmh"], 3),
        val_gt_max_along_m=round(gt_max_along, 3),
        frac_emitted_candidates_beyond_val_gt_max=round(
            float((fan_along > gt_max_along).mean()), 5),
        frac_windows_with_any_candidate_beyond_val_gt_max=round(
            float((fan_along > gt_max_along).any(axis=1).mean()), 5),
        frac_emitted_candidates_beyond_50ms_180kmh=round(
            float((fan_along / 2.0 > 50.0).mean()), 6))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"L1": res["L1_structural"]["verified_numerically"],
                      "pool": res["L2_anchor_side"]["pool"],
                      "anchors": res["L2_anchor_side"]["anchors_fps256"],
                      "emitted": res["L3_offset_side"]["emitted_fan"],
                      "gt": res["L3_offset_side"]["realised_future_gt"],
                      "verdict": res["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
