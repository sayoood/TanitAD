"""SC-13 anticipation probe, run #5 — the decisive re-test of run #4's positive.

WHAT RUN #4 LEFT OPEN
---------------------
Run #4 measured, in-domain (PhysicalAI val, stride 2, 3,241 anchors, n=23
BRAKE_FAR events), speed-matched AUROC `held` 0.723/0.740 vs `reactive`
0.434/0.450 — i.e. the model's imagined 2 s future appeared to shorten 2-3 s
BEFORE a deceleration that had not started and lay OUTSIDE the rollout horizon.
The same probe on comma2k19 did NOT replicate (held 0.538 ~ blind 0.608 ~
reactive 0.588) and the pre-registered falsifier fired. Two candidate readings
survived: (i) the in-domain positive is real and comma2k19 failed for domain /
competence reasons (there CV beats the model outright), or (ii) the in-domain
positive was n=23 noise and/or not actually driven by vision.

THIS RUN ATTACKS (ii) DIRECTLY, ON THE CORPUS WHERE THE MODEL IS COMPETENT.

  1. VOLUME.  --stride 1 (was 2) over the same 40 canonical val episodes:
     ~2x the anchors, ~2x the events, same corpus, same checkpoint. The
     stride-2 subset is recoverable exactly (anchor start index t, even t), so
     run #4's numbers are re-derivable from this run's substrate as a
     replication check rather than a remembered figure.

  2. STRONGER VISION CONTROLS.  Run #4's only vision control (`blind`) is a
     constant MEAN FRAME — far off-manifold, so it may UNDERSTATE vision by
     breaking the encoder rather than by removing the hazard. Two controls with
     correct input statistics are added:

       shuffled : a REAL 8-frame window from a DIFFERENT EPISODE, actions held.
                  Real scene, real motion, real statistics — only decorrelated
                  from this anchor's hazard. *** This is the control the claim
                  must beat. ***
       frozen   : this anchor's OWN last real frame repeated 8x, actions held.
                  Real scene, NO motion. Separates "the model reads the hazard
                  from the current scene" from "the model reads it from motion".
                  Note frozen is NOT a falsifier: a static stopped-lead cue
                  SHOULD survive it. It is diagnostic, not adversarial.

  3. HONEST INTERVALS.  Run #4 bootstrapped ANCHORS. Anchors 0.1 s apart in one
     episode are near-duplicates, so an anchor-level CI is anticonservative and
     n=23 "events" are not 23 independent facts. This run reports the
     EPISODE-CLUSTER bootstrap (resample the 40 val episodes with replacement,
     pool their anchors, rescore) as the decision-grade interval, per
     CLAUDE.md, and keeps the anchor-level one only for comparability with
     run #4. Differences between arms are bootstrapped PAIRED, on the same
     resample — never combined from two marginal intervals.

PRE-REGISTERED FALSIFIERS (both outcomes committed before the run)
------------------------------------------------------------------
On BRAKE_FAR, speed-matched, in-domain:
  F-A (volume) : AUROC(held) - AUROC(reactive) <= +0.10  (run #4 saw +0.29)
                 => the in-domain positive does not survive more anchors; it
                    was small-n noise. SC-13's H15 claim stays oracle-only.
  F-B (vision) : AUROC(held) - max(AUROC(shuffled), AUROC(blind)) <= +0.02
                 => nothing scene-specific is driving it; whatever anticipation
                    exists comes from ego kinematics, not from the world model
                    reading THIS scene. SC-13 escalates to the closed loop.
  SURVIVES only if BOTH margins are exceeded AND the paired episode-cluster
  bootstrap CI of each difference excludes 0.

Arms (all scored by the same deficit signal D = CV_forward(2s) - arm_forward(2s),
positive = the imagined future is SHORTER than constant-velocity):
  informed : true future actions. LEAKS the braking command. Upper bound only.
  held     : last observed action repeated. THE CLAIM.
  blind    : held + constant mean frame            (run #4's control)
  shuffled : held + real window, different episode  (NEW - the decisive control)
  frozen   : held + own last frame repeated 8x      (NEW - motion vs content)
  reactive : -(v(t)-v(t-0.5s))/0.5. No model. The floor.

Usage (eval pod):
  PYTHONPATH=/workspace/TanitAD/taniteval:/workspace/TanitAD/stack \
  OMP_NUM_THREADS=6 python3 sc13_probe_v5.py \
      --ckpt /workspace/v1_modelonly.pt --stride 1 \
      --out /workspace/sc13v5/sc13_v1_stride1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

STACK = os.environ.get("TANITAD_STACK", "/workspace/TanitAD/stack")
TEV = os.environ.get("TANITEVAL_ROOT", "/workspace/TanitAD/taniteval")
for _p in (TEV, STACK, os.path.join(STACK, "scripts")):
    if _p not in sys.path:
        sys.path.append(_p)

from taniteval import data, loaders, rollout                         # noqa: E402
from taniteval.registry import MODELS                                # noqa: E402
from driving_diagnostic import baseline_waypoints, gt_ego_waypoints  # noqa: E402
from tanitad.models.metric_dynamics import rollout_decode            # noqa: E402

VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
WINDOW, FWD_K, HZ = 8, 20, 10
LOOKAHEAD = 30            # need v(t+30) = +3 s for the BRAKE_FAR label
DROP = 2.0                # sustained speed drop inside 0-2 s = braking event
DROP_FAR = 1.5            # 2-3 s window: rarer, so a slightly lower bar
VMIN = 5.0                # ignore near-standstill anchors
BANK = 48                 # real windows held aside for the `shuffled` control

ARMS = ("informed", "held", "blind", "shuffled", "frozen")


@torch.no_grad()
def _mean_frame(episodes, device, n=64):
    """One constant [1,W,...] window: the val-set mean frame (run #4's control)."""
    acc, k = None, 0
    for ep in episodes:
        f = torch.as_tensor(ep.feats[:n]).float()
        if f.max() > 1.5:
            f = f / 255.0
        acc = f.mean(dim=0) if acc is None else acc + f.mean(dim=0)
        k += 1
    m = (acc / k).to(device)
    return m[None, None].expand(1, WINDOW, *m.shape).contiguous()


@torch.no_grad()
def _shuffle_bank(episodes, gen):
    """A pool of REAL windows tagged with their episode, for `shuffled`.

    Kept on CPU in the cache's own dtype (uint8 for frames) — BANK=48 windows
    of 9x256x256 would be ~0.9 GB in float32 and there is no reason to hold
    that resident on the GPU."""
    wins, owners = [], []
    per = max(1, BANK // max(1, len(episodes)))
    for ei, ep in enumerate(episodes):
        T = min(ep.feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        hi = T - WINDOW - LOOKAHEAD
        if hi <= 1:
            continue
        for _ in range(per):
            t = int(torch.randint(hi, (1,), generator=gen))
            wins.append(torch.as_tensor(ep.feats[t:t + WINDOW]).clone())
            owners.append(ei)
    return torch.stack(wins), torch.tensor(owners)


def _to_float(x):
    if x.dtype == torch.uint8:
        return x.float().div_(255.0)
    return x.float() if x.dtype == torch.float16 else x


@torch.no_grad()
def collect(entry, L, episodes, device, stride):
    """rollout.collect, keeping the anchor start index, episode and v-profile."""
    assert L["traj_capable"], f"{entry['key']} has no grounded rollout head"
    model, sro = L["model"], L["step_readout"]
    keys = ARMS + ("cv", "gt", "t", "sidx", "v0", "vfut")
    out = {k: [] for k in keys}
    eids = []
    mean_frame = _mean_frame(episodes, device)
    gen = torch.Generator().manual_seed(1234)
    bank, bank_ep = _shuffle_bank(episodes, gen)
    print(f"[probe] shuffle bank: {tuple(bank.shape)} {bank.dtype} "
          f"over {len(set(bank_ep.tolist()))} episodes", flush=True)

    for ei, ep in enumerate(episodes):
        feats = ep.feats
        T = min(feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        starts = [t for t in range(0, T - WINDOW - LOOKAHEAD, stride)]
        for i in range(0, len(starts), 8):
            ch = starts[i:i + 8]
            b = len(ch)
            last = torch.tensor([t + WINDOW - 1 for t in ch])
            fw = _to_float(torch.stack(
                [torch.as_tensor(feats[t:t + WINDOW]) for t in ch]).to(device))
            aw = torch.stack([ep.actions[t:t + WINDOW] for t in ch]).to(device)
            fa = torch.stack([ep.actions[t + WINDOW:t + WINDOW + FWD_K]
                              for t in ch]).to(device)
            aw, fa = rollout.append_ego(aw, fa, ep.poses, last,
                                        bool(entry.get("speed_input")),
                                        bool(entry.get("yaw_input")),
                                        bool(entry.get("dyn_input")), device)
            # held: repeat the last OBSERVED action. The appended ego channels
            # are already constant, so the commanded dims are held exactly too.
            fa_held = aw[:, -1:].expand(-1, FWD_K, -1).contiguous()

            st_real = model.encode_window(fw)
            for name, st, act in (("informed", st_real, fa),
                                  ("held", st_real, fa_held)):
                wp, _ = rollout_decode(model.predictor, st, aw, act, sro, FWD_K)
                out[name].append(wp[:, FWD_K - 1].cpu().float())

            # --- vision controls, all with actions HELD -------------------
            # blind: constant mean frame (off-manifold; run #4's control)
            views = {"blind": mean_frame.expand(b, -1, -1, -1, -1).contiguous()}
            # frozen: this anchor's own last real frame, repeated (no motion)
            views["frozen"] = fw[:, -1:].expand(-1, WINDOW, *fw.shape[2:]
                                                ).contiguous()
            # shuffled: a real window from a DIFFERENT episode (decorrelated)
            pick = []
            other = torch.nonzero(bank_ep != ei).flatten()
            for _ in range(b):
                pick.append(int(other[torch.randint(len(other), (1,),
                                                    generator=gen)]))
            views["shuffled"] = _to_float(bank[torch.tensor(pick)].to(device))

            for name, v in views.items():
                wp, _ = rollout_decode(model.predictor, model.encode_window(v),
                                       aw, fa_held, sro, FWD_K)
                out[name].append(wp[:, FWD_K - 1].cpu().float())

            out["cv"].append(baseline_waypoints(ep.poses, last)
                             ["constant_velocity"][:, -1].float())
            out["gt"].append(gt_ego_waypoints(ep.poses, last)[:, -1].float())
            out["t"].append(last.clone())
            out["sidx"].append(torch.tensor(ch))       # anchor START index
            out["v0"].append(ep.poses[last, 3].float())
            out["vfut"].append(torch.stack(
                [ep.poses[last + k, 3].float() for k in range(1, LOOKAHEAD + 1)],
                dim=1))
            eids.extend([ei] * b)
        print(f"[probe] ep {ei + 1}/{len(episodes)} "
              f"anchors={len(starts)}", flush=True)

    packed = {k: torch.cat(v) for k, v in out.items()}
    packed["eidx"] = torch.tensor(eids)
    packed["episode_ids"] = [ep.episode_id for ep in episodes]
    # reactive floor: the current 0.5 s deceleration, per episode
    react = []
    for ep in episodes:
        T = min(ep.feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        st = [t for t in range(0, T - WINDOW - LOOKAHEAD, stride)]
        last = torch.tensor([t + WINDOW - 1 for t in st])
        react.append(-(ep.poses[last, 3].float()
                       - ep.poses[last - 5, 3].float()) / 0.5)
    packed["reactive"] = torch.cat(react)
    assert len(packed["reactive"]) == len(packed["v0"]), "anchor misalignment"
    return packed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flagship-30k",
                    help="registry key whose ARCH/flags to use")
    ap.add_argument("--ckpt", default=None,
                    help="override the registry ckpt path (the eval pod was "
                         "reprovisioned; /root/models is gone)")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--val", default=VAL)
    ap.add_argument("--out", default="/workspace/sc13v5/sc13_v5.json")
    a = ap.parse_args()

    entry = dict([m for m in MODELS if m["key"] == a.model][0])
    if a.ckpt:
        entry["ckpt"] = a.ckpt
    files = data.list_val_episodes(a.val, a.episodes)
    assert files, f"no val episodes under {a.val}"
    t0 = time.time()
    L = loaders.load(entry, "cuda")
    eps = (data.load_frames(files) if L["feed"] == "frames"
           else data.load_features(files, L["feed"], "cuda"))
    p = collect(entry, L, eps, "cuda", a.stride)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sub = a.out.replace(".json", "_windows.pt")
    torch.save(p, sub)
    meta = dict(model=a.model, ckpt=entry["ckpt"], episodes=len(files),
                stride=a.stride, val=a.val, n_anchors=int(len(p["v0"])),
                wallclock_s=round(time.time() - t0, 1), arms=list(ARMS),
                window=WINDOW, fwd_k=FWD_K, bank=BANK)
    with open(a.out, "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"[sc13v5] substrate -> {sub}")


if __name__ == "__main__":
    main()
