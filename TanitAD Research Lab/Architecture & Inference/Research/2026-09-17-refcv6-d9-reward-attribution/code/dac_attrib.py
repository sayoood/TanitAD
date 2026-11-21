"""D9 attribution: did the RL policy leave the DRIVABLE AREA more than the cold start?

THE HYPOTHESIS, read from source (`tanitad/rl/pdm_proxy.py`):
    pdms = NC * DAC * (w_ep EP + w_ttc TTC + w_c C) / (w_ep + w_ttc + w_c)
DAC is a BINARY MULTIPLIER and `dac_from_drivable` returns 1 when no map exists. The
RL-TRAIN split had no SAM3 maps, so DAC was identically 1 for every candidate in every
anchor group. GRPO's advantage is computed WITHIN a group, so a term with zero
within-group variance contributes EXACTLY ZERO to the gradient -- not a weak constraint,
an absent one. The policy was free to maximise progress with no road-boundary term.

⇒ PREDICTION: the RL arms' plans put MORE waypoints on non-drivable ground than the cold
start's, on the identical windows. If they do not, this hypothesis is WRONG and the cause
of H-DDV2RL-2's FAIL-HARM is something else.

RUNNABLE BECAUSE: the T1 rolls were on the B1 eval split, and all 41 held-out clips are
inside the 139 AND have a SAM3 map (measured 2026-09-17). Zero GPU.

⛔ CONTROLS -- a number here without them is not evidence:
  * THE HUMAN'S OWN FUTURE PATH is the gold control: the human drove on the road, so this
    must read SMALL. If it does not, the frame or the map is wrong and NO arm comparison
    below is readable. This validates the pipeline, not the arms.
  * `ha0` (hold action: straight ahead at v0) is a second, dumber reference.
  * EVERY index convention below was pinned by MEASUREMENT, never assumed:
      - `ha0` reads exactly v0*t with y == 0  => metres, ego frame at t0, +x fwd/+y LEFT,
        the same convention the SAM3 grid declares (x in [0,60), y in [-16,16), 0.5 m).
      - `pose_last[w]` == `ep_poses[ws[w]]` at residual 0.000000 => ws IS the NOW row.
      - `ep_poses` == raw `poses[2:]` at residual 0.000000 => raw frame = ws + (n_stack-1).
  * The episode -> clip mapping is SELF-VERIFYING twice: the payload's own poses must
    reproduce `ep_poses`, and `ClipMapGT.check_pose_alignment` must return ALIGNED.

A POINT test on ego centres, not the footprint, so no yaw derivation can corrupt it.
It is therefore a LOWER bound on off-road contact -- directionally identical to DAC.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\Admin\tanitad-wt-bevtac\stack")
from tanitad.data import semantic_map_gt as SMG            # noqa: E402

RUN = pathlib.Path("C:/Users/Admin/tanitad-caches/ddv2rl-l1-20260917")
HELD = pathlib.Path("C:/Users/Admin/tanitad-caches/ddv2rl-20260915/heldout41_eps")
MAPS = pathlib.Path("D:/Projects/TanitAD-artifacts/sam3-maps-eval")
ARMS = {"base": "t1_base_dump", "rl-s0": "t1_l1-rl-s0_dump", "rl-s1": "t1_l1-rl-s1_dump"}

Y_HALF, CELL, THRESH, N_STACK = 16.0, 0.5, 0.5, 3
DRIVABLE_CH = SMG.CHANNELS.index("drivable")
GT_TICKS = np.array([4, 9, 14, 19])        # gt is 0.1 s; the plan is 0.5 s x 4

clips = sorted(p.name.split(".")[0] for p in HELD.glob("*.v2ep.pt"))
assert len(clips) == 41, f"expected 41 held-out clips, got {len(clips)}"


def offroad(xy: np.ndarray, frac: np.ndarray, seen: np.ndarray) -> tuple[int, int]:
    """(n_offroad, n_scored) for points [N, 2] in metres, ego frame at t0."""
    ix = np.floor(xy[:, 0] / CELL).astype(np.int64)
    iy = np.floor((xy[:, 1] + Y_HALF) / CELL).astype(np.int64)
    h, w = seen.shape
    inside = (ix >= 0) & (ix < h) & (iy >= 0) & (iy < w)
    ixc, iyc = np.clip(ix, 0, h - 1), np.clip(iy, 0, w - 1)
    sn = seen[ixc, iyc] & inside            # off-grid / unseen carries no evidence
    off = sn & (frac[DRIVABLE_CH][ixc, iyc] < THRESH)
    return int(off.sum()), int(sn.sum())


KEYS = list(ARMS) + ["human", "ha0"]
tot = {k: [0, 0] for k in KEYS}
per_ep = {k: [] for k in KEYS}             # per-episode rates, for the cluster read
win = {k: [] for k in KEYS}                # PER-WINDOW rates -- the estimator's input
win_eid: list[int] = []                    # the cluster label: one per window
n_ep = n_win = 0
align_max = 0.0
refused: list[str] = []

for ei, clip in enumerate(clips):
    dumps = {a: RUN / d / f"ep{ei:03d}.npz" for a, d in ARMS.items()}
    decs = {a: RUN / d / "decisions" / f"ep{ei:03d}.npz" for a, d in ARMS.items()}
    if not all(p.exists() for p in list(dumps.values()) + list(decs.values())):
        refused.append(f"ep{ei:03d}: a dump is missing"); continue
    z = {a: np.load(p) for a, p in dumps.items()}
    zd = {a: np.load(p) for a, p in decs.items()}
    ep_poses = zd["base"]["ep_poses"]

    # ---- mapping check 1: the payload's own poses must reproduce ep_poses ----------
    pay = torch.load(HELD / f"{clip}.v2ep.pt", map_location="cpu", weights_only=False)
    P = np.asarray(pay["poses"], dtype=np.float64)
    off = N_STACK - 1
    if P.shape[0] - off != ep_poses.shape[0] or \
            np.abs(P[off:, :2] - ep_poses[:, :2]).max() > 1e-4:
        refused.append(f"ep{ei:03d}: payload poses do not reproduce ep_poses"); continue
    # ---- mapping check 2: the MAP must align to this episode's frame axis ----------
    try:
        mp = MAPS / f"{SMG.sha12(clip)}{SMG.GT_SUFFIX}"
        rep = SMG.open_path(mp, clip).check_pose_alignment(P)
    except Exception as exc:                                   # noqa: BLE001
        refused.append(f"ep{ei:03d}: {type(exc).__name__} {exc}"); continue
    if rep.get("verdict") != "ALIGNED":
        refused.append(f"ep{ei:03d}: alignment {rep.get('verdict')}"); continue
    align_max = max(align_max, float(rep["aligned_max_m"]))

    gt_map = SMG.open_path(mp, clip)
    ws = zd["base"]["ws"]
    for a in ARMS:                      # paired or it is not a comparison
        assert np.array_equal(zd[a]["ws"], ws), f"ep{ei:03d}: {a} windows differ"

    acc = {k: [0, 0] for k in KEYS}
    for wi in range(len(ws)):
        raw = SMG.raw_frame_index(int(ws[wi]), N_STACK)
        mf = gt_map.read(np.asarray([int(raw)]))
        frac = np.asarray(mf.cart[0], dtype=np.float32)
        seen = np.asarray(mf.seen[0], dtype=bool)
        w_rate: dict[str, float | None] = {}
        for a in ARMS:
            o, s = offroad(z[a]["os"][wi], frac, seen); acc[a][0] += o; acc[a][1] += s
            w_rate[a] = o / s if s else None
        o, s = offroad(z["base"]["ha0"][wi], frac, seen); acc["ha0"][0] += o; acc["ha0"][1] += s
        w_rate["ha0"] = o / s if s else None
        if (zd["base"]["gt_future_valid_ext"][wi][GT_TICKS] > 0).all():
            # ⛔ gt_future_ext is in the EPISODE frame, not the ego frame at t0 that
            # `os`/`ha0` use -- MEASURED: gt[0] - pose_last == v0 * 0.1 m. Untransformed
            # this control read 0.1741 off-road, i.e. it measured the frame error and
            # not the human, and it correctly voided the whole comparison.
            pl = zd["base"]["pose_last"][wi]
            d = zd["base"]["gt_future_ext"][wi][GT_TICKS, :2] - pl[None, :2]
            c, s_ = np.cos(-pl[2]), np.sin(-pl[2])
            g_ego = np.stack([d[:, 0] * c - d[:, 1] * s_, d[:, 0] * s_ + d[:, 1] * c], 1)
            o, s = offroad(g_ego, frac, seen)
            acc["human"][0] += o; acc["human"][1] += s
            w_rate["human"] = o / s if s else None
        # a window enters the estimator only if EVERY arm scored it -- otherwise the
        # arms are not on the same windows and the pairing that makes the estimator
        # valid is gone.
        if all(w_rate.get(k) is not None for k in KEYS):
            for k in KEYS:
                win[k].append(w_rate[k])
            win_eid.append(ei)
        n_win += 1
    for k in KEYS:
        tot[k][0] += acc[k][0]; tot[k][1] += acc[k][1]
        if acc[k][1]:
            per_ep[k].append(acc[k][0] / acc[k][1])
    n_ep += 1

print(f"\nepisodes scored {n_ep}/41   windows {n_win}   worst map-pose residual {align_max:.4f} m")
if refused:
    print(f"REFUSED {len(refused)}:")
    for r in refused[:8]:
        print("   ", r)

out = {}
print(f"\n{'arm':8s} {'offroad':>9s} {'scored':>9s} {'rate':>8s} {'per-ep mean':>12s}")
for k in ("human", "ha0", "base", "rl-s0", "rl-s1"):
    o, s = tot[k]
    r = o / s if s else float("nan")
    m = float(np.mean(per_ep[k])) if per_ep[k] else float("nan")
    out[k] = {"n_offroad": o, "n_scored": s, "rate": r, "per_episode_mean": m,
              "n_episodes": len(per_ep[k])}
    print(f"{k:8s} {o:>9d} {s:>9d} {r:>8.4f} {m:>12.4f}")

h = out["human"]["rate"]
print(f"\n⛔ PIPELINE CONTROL -- the human's own driven path reads {h:.4f} off-road.")
print("   This must be SMALL. If it is not, the frame or the map is wrong and nothing below is readable.")
print("\nTHE TEST:")
CONTROL_BAR = 0.05
if not out["base"]["n_scored"] or not np.isfinite(h) or h > CONTROL_BAR:
    print(f"   ⛔ INCONCLUSIVE -- the pipeline control reads {h:.4f} (bar: <= {CONTROL_BAR}).")
    print("   NO arm comparison is reported. A broken control means a broken instrument, and")
    print("   a delta computed through it would be a measurement of the defect, not the arms.")
else:
    # ⛔ THE ONLY ADMISSIBLE ESTIMATOR. A raw difference of rates is not a result:
    # both arms are scored on the SAME windows, so the pairing must be kept and the
    # clustering must be by EPISODE.
    sys.path.insert(0, r"C:\Users\Admin\tanitad-wt-bevtac\taniteval")
    from taniteval.ci import paired_episode_cluster_bootstrap as PB   # noqa: E402

    eid = np.asarray(win_eid)
    print(f"   estimator: paired_episode_cluster_bootstrap, n_boot 2000, seed 0, "
          f"clustered by episode (n={len(set(win_eid))} episodes, {len(eid)} windows)\n")
    ci = {}
    for lab, x, y in (("rl-s0 - base", "rl-s0", "base"),
                      ("rl-s1 - base", "rl-s1", "base"),
                      ("base  - human", "base", "human"),
                      ("ha0   - human", "ha0", "human")):
        r = PB(np.asarray(win[x]), np.asarray(win[y]), eid, n_boot=2000, seed=0)
        ci[lab] = {k: r.get(k) for k in ("delta", "lo", "hi", "separated")}
        flag = "  <== SEPARATED" if r.get("separated") else "  (not separated)"
        print(f"   {lab:14s} {r['delta']:+8.4f}  [{r['lo']:+8.4f}, {r['hi']:+8.4f}]{flag}")
    out["_paired_ci"] = ci
    print("\n   ⚠️ n = 2 RL seeds. A separated CI answers 'would another draw of EPISODES")
    print("   say this?' -- not 'would another TRAINING RUN say this?'. Read the two seeds.")

pathlib.Path("C:/Users/Admin/qland/dac_attrib.json").write_text(
    json.dumps({"_what": "D9 attribution: off-road waypoint rate, cold start vs RL, identical windows",
                "_hypothesis": "DAC was a constant multiplier (no maps on the RL-train split), so it "
                               "had zero within-group variance and contributed exactly zero to GRPO's "
                               "advantage -- the policy had no road-boundary term at all.",
                "_tier": "T1 plans (self-action open loop), scored against SAM3 map GT",
                "_evidence_class": "MEASURED (ours)",
                "_point_test": "ego-centre waypoints, NOT the footprint -- a LOWER bound on contact",
                "_n": {"episodes": n_ep, "windows": n_win},
                "_worst_map_pose_residual_m": align_max,
                "_refused": refused,
                "arms": out}, indent=1) + "\n", encoding="utf-8", newline="\n")
print("\nwrote dac_attrib.json")
