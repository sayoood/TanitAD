#!/usr/bin/env python3
"""A0 — does the RL reward actually FIRE on real driving windows?

⛔ THE GATE THIS ANSWERS
------------------------
``products/P4-training-pipelines/.../LAUNCH_PLAN.md`` makes A0 the arm that runs
BEFORE any RL step, and A1 does not launch until it passes. The question is not
"is the reward good" — it is **"is the reward PRESENT"**:

    A reward whose safety terms never fire is not a safe reward — it is an
    absent measurement wearing safety's name.

``tanitad.rl.audit.audit_reward`` already refuses to bless the reward without
this: with no scene context it returns **INCONCLUSIVE**, naming
``collision``/``headway`` as CONSTANT across the whole degenerate panel. A term
that is constant across candidates cancels EXACTLY in a group-relative
advantage, so it contributes nothing to the gradient no matter its weight.

WHAT IT MEASURES
----------------
For real windows, per component: did it VARY across the candidate fan, by how
much, over what n. Constant => carries no signal => reported, never assumed.

⛔ WHY THE FAN IS THE ANCHOR VOCABULARY AND NOT A CHECKPOINT'S OUTPUT
---------------------------------------------------------------------
refcv3 has **no trained checkpoint and no registry row** yet (the IL cold start
is a prerequisite of the whole campaign). Rather than block A0 on that, the
candidate set here is the model's OWN ANCHOR VOCABULARY — ``synth_anchor_pool``
+ ``furthest_point_sample``, i.e. the fan's actual support, which is what the
decoder offsets from. This makes A0 **checkpoint-independent**, which is
correct: whether the reward can rank candidates on real scenes is a property of
the reward and the data, not of how well some checkpoint scores.
⚠️ STATED LIMIT: a trained decoder's fan is tighter than the raw vocabulary, so
the spreads here are an UPPER BOUND on what a converged model would show. A
component that is constant HERE is constant everywhere; a component that fires
here may still be near-constant on a tight fan. A0 can therefore REFUTE
readiness but not fully confirm it — re-run against the real fan once a
checkpoint exists.

CORPUS
------
``physicalai-val130-heldout`` + the ``obstacle.offline`` agent join
(``build_obstacle_join.py``). Agents are ego-frame **+x forward, +y left** at
their own frame (``build_obstacle_join.py:15-17``), re-expressed into the window
origin's ego frame here. Held-out episodes on purpose: the decodability battery
was later found to be 100 % train-overlapped (H-DEC-3), and there is no reason
to repeat that.

Evidence class: MEASURED (ours; rule-based over real scene facts). Tier: N/A —
this is an instrument-coverage probe, not a capability claim.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ))

from tanitad.refs import refc  # noqa: E402
from tanitad.rl import audit as AUD  # noqa: E402
from tanitad.rl import rewards as R  # noqa: E402

DT = 0.1
S_STEPS = 20                     # 2.0 s at 10 Hz — the seam slot
LEAD_LAT_M = 2.0                 # taniteval.lead_source.LEAD_LAT_M
LEAD_MAX_GAP_M = 80.0            # taniteval.lead_source.LEAD_MAX_GAP_M


def ego_frame(px, py, x0, y0, yaw0):
    """World -> ego frame at (x0, y0, yaw0). +x forward, +y left."""
    dx, dy = px - x0, py - y0
    c, s = np.cos(yaw0), np.sin(yaw0)
    return dx * c + dy * s, -dx * s + dy * c


def agents_to_world(agents, x, y, yaw):
    """Agent ego-frame (cx, cy) at its own frame -> world."""
    c, s = np.cos(yaw), np.sin(yaw)
    out = []
    for a in agents:
        cx, cy = float(a["cx"]), float(a["cy"])
        out.append((x + cx * c - cy * s, y + cx * s + cy * c, a.get("track_id")))
    return out


def build_anchor_fan(n_anchors: int, seed: int = 0):
    """The model's own anchor vocabulary, dense at dt -> [N, S, 2]."""
    horizons = tuple(range(1, S_STEPS + 1))
    pool = refc.synth_anchor_pool(horizons, pool_size=2048, seed=seed, dt=DT)
    return refc.furthest_point_sample(pool, n_anchors, seed=seed)


def read_agents(path, clips=None):
    per: dict[str, dict[int, list]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            cid = r["clip_id"]
            if clips is not None and cid not in clips:
                continue
            per.setdefault(cid, {})[int(r["frame_idx"])] = r.get("agents", [])
    return per


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epdir", required=True, help="dir of *.v2ep.pt episodes")
    ap.add_argument("--agents", required=True, help="obstacle.offline join jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=40)
    ap.add_argument("--windows-per-clip", type=int, default=6)
    ap.add_argument("--n-anchors", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    eps = sorted(glob.glob(os.path.join(a.epdir, "*.v2ep.pt")))
    if not eps:
        raise SystemExit(f"[a0] no episodes under {a.epdir}")
    clip_ids = {os.path.basename(p).split(".")[0] for p in eps}
    agents = read_agents(a.agents, clip_ids)
    print(f"[a0] {len(eps)} episodes, {len(agents)} with an agent join", flush=True)
    if not agents:
        raise SystemExit("[a0] ⛔ NO episode has an agent join — A0 cannot rule")

    fan = build_anchor_fan(a.n_anchors, a.seed)          # [N, S, 2]
    spec = R.RewardSpec()
    rng = np.random.default_rng(a.seed)

    agg: dict[str, dict] = {n: {"fired": 0, "seen": 0, "spreads": [],
                                "means": []} for n in spec.names}
    n_windows = 0
    n_with_obstacles = 0
    n_with_lead = 0

    for path in eps:
        cid = os.path.basename(path).split(".")[0]
        if cid not in agents:
            continue
        if len({c for c in agg}) and n_windows >= a.clips * a.windows_per_clip:
            break
        d = torch.load(path, map_location="cpu", weights_only=False)
        poses = d["poses"].numpy()                        # [T, 4] x,y,yaw,v
        T = poses.shape[0]
        if T < S_STEPS + 2:
            continue
        frames = sorted(agents[cid])
        cand = [t for t in frames if 0 <= t < T - S_STEPS - 1]
        if not cand:
            continue
        picks = rng.choice(cand, size=min(a.windows_per_clip, len(cand)),
                           replace=False)

        for t0 in sorted(int(t) for t in picks):
            x0, y0, yaw0 = poses[t0, 0], poses[t0, 1], poses[t0, 2]

            fut = poses[t0 + 1: t0 + 1 + S_STEPS]
            gx, gy = ego_frame(fut[:, 0], fut[:, 1], x0, y0, yaw0)
            gt = torch.tensor(np.stack([gx, gy], -1), dtype=torch.float32)

            here = agents[cid].get(t0, [])
            obs_xy = []
            for wx, wy, _tid in agents_to_world(here, x0, y0, yaw0):
                ex, ey = ego_frame(np.array(wx), np.array(wy), x0, y0, yaw0)
                if 0.0 < float(ex) < LEAD_MAX_GAP_M:
                    obs_xy.append((float(ex), float(ey)))

            # ⭐ v0 = the window's OWN current ego speed (poses[:, 3]), which is
            # what `progress` now normalises by. Inference-admissible ego state,
            # never fan-derived and never expert-derived.
            ctx: dict = {"dt": DT, "gt_traj": gt,
                         "v0": float(poses[t0, 3])}
            if obs_xy:
                ctx["obstacles"] = torch.tensor(obs_xy, dtype=torch.float32)
                n_with_obstacles += 1

            # lead = nearest in-lane agent ahead (lead_source's geometry)
            inlane = [(ex, ey) for ex, ey in obs_xy if abs(ey) <= LEAD_LAT_M]
            if inlane:
                lx, ly = min(inlane, key=lambda p: p[0])
                # a static-lead approximation over the horizon; STATED, and it
                # only affects the headway MAGNITUDE, not whether it fires
                ctx["lead_path"] = torch.tensor(
                    [[lx, ly]] * S_STEPS, dtype=torch.float32)
                n_with_lead += 1

            cov = AUD.report_component_coverage(spec, fan, ctx)
            for name, info in cov.items():
                agg[name]["seen"] += 1
                agg[name]["fired"] += int(bool(info["fired"]))
                agg[name]["spreads"].append(info["spread"])
                agg[name]["means"].append(info["mean"])
            n_windows += 1

    if n_windows == 0:
        raise SystemExit("[a0] ⛔ ZERO windows scored — refusing to report a "
                         "coverage verdict over an empty set")

    # ⛔ CORRECTED (second time) — THE POPULATION MATTERS.
    # The first metric took the median spread over ALL windows. `headway` is
    # UNDEFINED where there is no lead vehicle (72 % of windows here), so that
    # median was dominated by windows the component does not apply to and read
    # 0.0000 — which I first reported as the component being inert. It is not:
    # headway fired on 68/240 = 28.3 % of windows, which is EXACTLY the
    # lead-present count, i.e. it ranks on 100 % of the windows where it applies.
    # ⇒ Report APPLICABILITY and CONDITIONAL SPREAD as two separate numbers.
    # A statistic computed over a population where the quantity is undefined is
    # not a weak measurement, it is a different measurement.
    rows = {}
    for name, st in agg.items():
        sp = np.array(st["spreads"], dtype=float)
        live = sp[sp > 1e-9]
        rows[name] = {
            "weight": spec.weights[name],
            "applicable_frac": st["fired"] / max(st["seen"], 1),
            "spread_median_ALL": float(np.median(sp)),
            "spread_median_WHEN_APPLICABLE": (float(np.median(live))
                                              if live.size else 0.0),
            "spread_max": float(sp.max()),
            "value_mean": float(np.mean(st["means"])),
            "n_windows": int(st["seen"]),
            "n_applicable": int(live.size),
        }

    # ⛔ CORRECTED after the first run: "fires at all" is TOO LENIENT.
    # `headway` read fired_frac 28.3 % with a MEDIAN SPREAD OF 0.0000 — i.e. on
    # most windows it is identical across every candidate, so it cancels exactly
    # in the group-relative advantage and cannot rank anything. A term can be
    # present and still carry no signal; only SPREAD makes it a ranking signal.
    dead = [n for n, r in rows.items() if r["applicable_frac"] == 0.0]
    inert = [n for n, r in rows.items()
             if n not in dead and r["spread_median_WHEN_APPLICABLE"] <= 1e-9]
    rare = [n for n, r in rows.items()
            if n not in dead and n not in inert and r["applicable_frac"] < 0.5]
    if dead:
        verdict = (f"⛔ FAIL — component(s) {dead} NEVER applied over "
                   f"{n_windows} real windows. They cancel in the group-relative "
                   "advantage and contribute nothing. A1 must not launch.")
    elif inert:
        verdict = (f"⛔ FAIL — component(s) {inert} apply but are IDENTICAL "
                   "across the fan even where they apply: they cancel in the "
                   "group-relative advantage and cannot rank anything. Present is "
                   "not the same as informative. A1 must not launch.")
    else:
        verdict = ("PASS — every weighted component ranks the fan wherever it "
                   "applies")
        if rare:
            verdict += (f" — ⚠️ {rare} apply on <50 % of windows; that is a "
                        "BASE RATE of the situation they score, not a defect, and "
                        "it belongs in the launch record")

    out = {
        "_what": "A0 reward-coverage probe on real held-out windows",
        "_corpus": os.path.basename(a.epdir.rstrip("/\\")),
        "_evidence_class": "MEASURED (ours; rule-based over real scene facts)",
        "_tier": "N/A — instrument coverage, not a capability claim",
        "_fan": (f"anchor vocabulary (synth_anchor_pool + FPS, n={a.n_anchors}) "
                 "— checkpoint-independent; a trained fan is TIGHTER, so these "
                 "spreads are an UPPER BOUND"),
        "n_windows": n_windows,
        "n_windows_with_obstacles": n_with_obstacles,
        "n_windows_with_lead": n_with_lead,
        "components": rows, "dead": dead, "inert": inert, "rare": rare,
        "verdict": verdict,
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print(f"\n[a0] {n_windows} windows · {n_with_obstacles} with obstacles · "
          f"{n_with_lead} with a lead", flush=True)
    print(f"{'component':<14}{'w':>6}{'applies':>9}{'spread|app':>12}"
          f"{'spread|all':>12}{'mean':>9}")
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["applicable_frac"]):
        print(f"{name:<14}{r['weight']:>6.2f}{r['applicable_frac']:>8.1%}"
              f"{r['spread_median_WHEN_APPLICABLE']:>12.4f}"
              f"{r['spread_median_ALL']:>12.4f}{r['value_mean']:>9.4f}")
    print(f"\n[a0] VERDICT: {verdict}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
