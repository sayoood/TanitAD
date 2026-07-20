"""SC-13 real-window anticipation probe — Opponent Analyzer run #4.

Question (P0.1 of the Opponent-Analyzer backlog): SC-13's numbers so far are a
DESIGN ORACLE (two archetype policies in a toy sim). Does OUR checkpoint show
any *anticipation* signal on REAL held-out windows — i.e. does the world model's
imagined 2 s future shorten BEFORE the ego actually brakes, and does that beat a
detection-free reactive baseline?

Proxy honesty: the canonical PhysicalAI val has no object labels, so we cannot
label "stationary lead" directly. We label the OBSERVABLE CONSEQUENCE instead —
a sustained ego deceleration event — which is the response SC-13 measures
(braking-onset lead time). Scenario identity is therefore INFER, not FACT.

Three rollout arms on identical anchors (the deficit signal is always
D = CV_forward(2s) - pred_forward(2s); positive = imagined slowdown):
  informed : TRUE future actions fed to the rollout. LEAKS the braking command
             -- reported only as an upper bound / sanity check, never as
             evidence of anticipation.
  held     : future actions = the LAST OBSERVED action repeated ("keep doing
             what I am doing"). The real test: does the imagined future still
             shorten when a hazard is ahead, with no command telling it to?
  blind    : actions held AND vision replaced by a constant mean frame. Control
             that isolates how much of `held` comes from VISION rather than
             from the ego-state channels alone.
  reactive : -(v(t) - v(t-0.5s))/0.5 -- current decel; no model, no detection.
             The "react to what already happened" floor SC-13 must beat.

Labels at anchor t (10 Hz), all requiring v(t) >= 5 m/s:
  BRAKE_NEAR : v drops >= DROP within the next 2 s   (inside the model horizon)
  BRAKE_FAR  : v drops >= DROP in the 2-3 s window AND < 0.5 m/s inside 0-2 s
               -> braking has NOT started and lies OUTSIDE the 2 s rollout:
               the pure anticipation test
  CRUISE     : |v(t+k) - v(t)| <= 0.5 for all k in 0..3 s

Pre-registered falsifier: on BRAKE_FAR, AUROC(held) <= AUROC(reactive) + 0.02,
or AUROC(held) <= AUROC(blind) + 0.02 => no VISION-driven anticipation advantage
over reaction on real data; SC-13's H15 claim stays oracle-only and must be
escalated to the closed loop.

Usage: PYTHONPATH=/root/taniteval:/root/TanitAD/stack python3 sc13_real_probe.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import data, loaders, rollout                      # noqa: E402
from taniteval.registry import MODELS                             # noqa: E402
from driving_diagnostic import baseline_waypoints, gt_ego_waypoints  # noqa: E402
from tanitad.models.metric_dynamics import rollout_decode         # noqa: E402

VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
WINDOW, FWD_K, HZ = 8, 20, 10
LOOKAHEAD = 30            # need v(t+30) = +3 s for the BRAKE_FAR label
DROP = 2.0                # m/s sustained speed drop = a braking event (0-2 s)
DROP_FAR = 1.5            # 2-3 s window: rarer, so a slightly lower bar
VMIN = 5.0                # ignore near-standstill anchors


def auroc(pos, neg):
    """Mann-Whitney AUROC, ties counted at 0.5. pos/neg are 1-D tensors."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    p = pos.reshape(-1, 1)
    n = neg.reshape(1, -1)
    return float(((p > n).float() + 0.5 * (p == n).float()).mean())


def _boot_ci(pos, neg, gen, n=2000, lo=0.025, hi=0.975):
    """Percentile bootstrap CI for the AUROC, resampling BOTH classes."""
    if len(pos) < 2 or len(neg) < 2:
        return [float("nan"), float("nan")]
    vals = []
    for _ in range(n):
        ip = torch.randint(len(pos), (len(pos),), generator=gen)
        iN = torch.randint(len(neg), (len(neg),), generator=gen)
        vals.append(auroc(pos[ip], neg[iN]))
    v = torch.tensor(vals).sort().values
    return [round(float(v[int(lo * n)]), 3), round(float(v[int(hi * n)]), 3)]


@torch.no_grad()
def _mean_frame(episodes, device, n=64):
    """A single constant [1, W, C, H, W] window: the mean frame of the val set.
    Feeding it strips scene content while keeping the input shape/statistics."""
    acc, k = None, 0
    for ep in episodes:
        f = torch.as_tensor(ep.feats[:n]).float()
        if f.max() > 1.5:
            f = f / 255.0
        s = f.mean(dim=0)
        acc = s if acc is None else acc + s
        k += 1
    m = (acc / k).to(device)                      # [C,H,W] or [tokens,d]
    return m[None, None].expand(1, WINDOW, *m.shape).contiguous()


@torch.no_grad()
def collect(entry, L, episodes, device, stride):
    """rollout.collect, but keeping the anchor time index + speed profile."""
    assert L["traj_capable"], f"{entry['key']} has no grounded rollout head"
    model, sro = L["model"], L["step_readout"]
    out = {k: [] for k in ("informed", "held", "blind", "cv", "gt", "t",
                           "v0", "vfut", "eid")}
    mean_frame = _mean_frame(episodes, device)
    for ep in episodes:
        feats = ep.feats
        T = min(feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        starts = [t for t in range(0, T - WINDOW - LOOKAHEAD, stride)]
        for i in range(0, len(starts), 8):
            ch = starts[i:i + 8]
            last = torch.tensor([t + WINDOW - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(feats[t:t + WINDOW])
                              for t in ch]).to(device)
            if fw.dtype == torch.uint8:
                fw = fw.float().div_(255.0)
            elif fw.dtype == torch.float16:
                fw = fw.float()
            aw = torch.stack([ep.actions[t:t + WINDOW] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + WINDOW:t + WINDOW + FWD_K]
                              for t in ch]).to(device)
            aw, fa = rollout.append_ego(aw, fa, ep.poses, last,
                                        bool(entry.get("speed_input")),
                                        bool(entry.get("yaw_input")),
                                        bool(entry.get("dyn_input")), device)
            # held actions: repeat the last OBSERVED action across the future.
            # The ego channels appended above are already constant, so this
            # holds exactly the commanded (steer, accel/jerk) dims too.
            fa_held = aw[:, -1:].expand(-1, FWD_K, -1).contiguous()
            states = model.encode_window(fw)
            for name, st, act in (("informed", states, fa),
                                  ("held", states, fa_held)):
                wp, _ = rollout_decode(model.predictor, st, aw, act, sro, FWD_K)
                out[name].append(wp[:, FWD_K - 1].cpu().float())        # [b,2]
            # blind control: same held actions, scene replaced by a constant
            # mean frame -> whatever survives is NOT coming from vision.
            fb = mean_frame.expand(fw.shape[0], -1, -1, -1, -1).contiguous()
            wp, _ = rollout_decode(model.predictor, model.encode_window(fb),
                                   aw, fa_held, sro, FWD_K)
            out["blind"].append(wp[:, FWD_K - 1].cpu().float())
            out["cv"].append(baseline_waypoints(ep.poses, last)
                             ["constant_velocity"][:, -1].float())      # 2 s
            out["gt"].append(gt_ego_waypoints(ep.poses, last)[:, -1].float())
            out["t"].append(last.clone())
            out["v0"].append(ep.poses[last, 3].float())
            # speed profile v(t+1..t+LOOKAHEAD) relative to the anchor
            out["vfut"].append(torch.stack(
                [ep.poses[last + k, 3].float() for k in range(1, LOOKAHEAD + 1)],
                dim=1))                                                 # [b,30]
            out["eid"].extend([ep.episode_id] * len(ch))
        # reactive signal needs v(t-5); poses are per-episode so fold it in here
    eids = out.pop("eid")
    packed = {k: torch.cat(v) for k, v in out.items()}
    packed["eid"] = eids
    # reactive baseline: current 0.5 s decel, recomputed per episode
    react, off = [], 0
    for ep in episodes:
        T = min(ep.feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        starts = [t for t in range(0, T - WINDOW - LOOKAHEAD, stride)]
        last = torch.tensor([t + WINDOW - 1 for t in starts])
        react.append(-(ep.poses[last, 3].float()
                       - ep.poses[last - 5, 3].float()) / 0.5)
        off += len(starts)
    packed["reactive"] = torch.cat(react)
    assert len(packed["reactive"]) == len(packed["v0"]), "anchor misalignment"
    return packed


def analyse(p):
    v0, vf = p["v0"], p["vfut"]
    drop_near = v0 - vf[:, :20].min(dim=1).values      # 0-2 s
    drop_far = v0 - vf[:, 20:30].min(dim=1).values     # 2-3 s
    swing_all = (vf - v0[:, None]).abs().max(dim=1).values
    fast = v0 >= VMIN
    brake_near = fast & (drop_near >= DROP)
    brake_far = fast & (drop_far >= DROP_FAR) & (drop_near < 0.75)
    cruise = fast & (swing_all <= 0.5)

    sig = {arm: p["cv"][:, 0] - p[arm][:, 0]           # forward deficit vs CV
           for arm in ("informed", "held", "blind")}
    sig["gt_oracle"] = p["cv"][:, 0] - p["gt"][:, 0]
    sig["reactive"] = p["reactive"]
    res = {"n_windows": int(len(v0)), "n_fast": int(fast.sum()),
           "n_brake_near": int(brake_near.sum()),
           "n_brake_far": int(brake_far.sum()), "n_cruise": int(cruise.sum()),
           "drop_threshold_mps": DROP, "drop_far_threshold_mps": DROP_FAR,
           "vmin_mps": VMIN}
    g = torch.Generator().manual_seed(0)
    for lbl, m in (("brake_near", brake_near), ("brake_far", brake_far)):
        row = {f"auroc_{k}": auroc(v[m], v[cruise]) for k, v in sig.items()}
        # bootstrap CI over EVENTS (the scarce class) -- n_brake_far is small
        row.update({f"ci95_{k}": _boot_ci(v[m], v[cruise], g)
                    for k, v in sig.items()})
        # confound check: are the event anchors simply at a different speed?
        row["median_v0_event"] = float(v0[m].median()) if m.any() else float("nan")
        row["median_v0_cruise"] = float(v0[cruise].median()) if cruise.any() else float("nan")
        row.update({
            f"median_{k}_event": (float(v[m].median()) if m.any()
                                  else float("nan")) for k, v in sig.items()})
        row.update({
            f"median_{k}_cruise": (float(v[cruise].median()) if cruise.any()
                                   else float("nan")) for k, v in sig.items()})
        res[lbl] = row
    # context: 2 s ADE of each arm vs GT on the same anchors
    err = lambda a: float((a - p["gt"]).norm(dim=-1).mean())
    res["ade2s_m"] = {arm: err(p[arm]) for arm in
                      ("informed", "held", "blind")}
    res["ade2s_m"]["cv"] = err(p["cv"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flagship-30k")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--val", default=VAL)
    ap.add_argument("--out", default="/root/taniteval/results/sc13_real_probe.json")
    a = ap.parse_args()
    entry = [m for m in MODELS if m["key"] == a.model][0]
    files = data.list_val_episodes(a.val, a.episodes)
    assert files, f"no val episodes under {a.val}"
    t0 = time.time()
    L = loaders.load(entry, "cuda")
    eps = (data.load_frames(files) if L["feed"] == "frames"
           else data.load_features(files, L["feed"], "cuda"))
    p = collect(entry, L, eps, "cuda", a.stride)
    torch.save(p, a.out.replace(".json", "_windows.pt"))   # raw substrate
    res = analyse(p)
    res.update(model=a.model, ckpt=entry["ckpt"], episodes=len(files),
               stride=a.stride, wallclock_s=round(time.time() - t0, 1),
               val=a.val)
    print(json.dumps(res, indent=2))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[sc13] wrote {a.out}")


if __name__ == "__main__":
    main()
