"""flagship v1.5 trainer — head-only, on the FROZEN v1 world model.

The encoder is not in this process's gradient graph at all: its states were
precomputed once by ``v15_prep.py states``. The predictor IS loaded (frozen, in
eval, ``requires_grad_(False)``) because conditioning (b) rolls it forward under
the probe action vocabulary every step. Only ``FlagshipV15Head`` is optimised.

Two independent ablation axes:

  ``--cond``  WHERE the conditioning comes from (architecture)
    a     frozen encoder states only                       (the drop-in stage)
    ab    + imagined future latents from the frozen
            predictor rolled under the probe actions       (the novel part)
    abc   + the VTARGET and v2.1 ROUTE goal tokens,
            goal-dropout 0.5 on each                       (the full v1.5)

  ``--label-set``  WHICH LABEL GENERATION feeds (c) — separates a gain from the
    architecture from a gain from the repaired labels, which is the thing Sayed
    expects and the thing an ablation can actually demonstrate:
    v21     the repaired labels (validated VTARGET mint + refb_labels v2.1)
    legacy  the pre-repair labels (raw jittery mint + silent-straight route)

Loss: the DiffusionDrive recipe REF-C validated — anchor-classification CE
against the GT-nearest anchor + L1 reconstruction from that anchor, with the
truncated-denoise refinement live in the same forward.

In-training validation runs the EXACT TanitEval window protocol (first 40 val
episodes, window 8, stride 8, waypoints 5/10/15/20) so the curve is in the same
units as the gates.

Usage (pod2):
  PYTHONPATH=/workspace/TanitAD/stack python3 train_flagship_v15.py \
    --states-train /workspace/v15/states_train.pt \
    --poses-train  /workspace/v15/poses_train.pt \
    --states-val   /workspace/v15/states_val.pt \
    --poses-val    /workspace/v15/poses_val.pt \
    --trunk /workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt \
    --anchors /workspace/v15/anchors256.pt --probes /workspace/v15/probes8.pt \
    --cond abc --steps 20000 --batch 64 --out /workspace/experiments/flagship-v15-abc
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from tanitad.models.flagship_v15 import (SPEED_SCALE, FlagshipV15Head,
                                         imagine_probes, param_breakdown,
                                         v15_ablation_config, v15_losses)
from v15_prep import HORIZONS, K_MAX, WINDOW, load_frozen_v1

def _ego(dxy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * dxy[:, 0] + s * dxy[:, 1],
                        -s * dxy[:, 0] + c * dxy[:, 1]], dim=-1)


# --------------------------------------------------------------------------- #
# dataset — cached states + pose-derived targets, zero disk I/O per item       #
# --------------------------------------------------------------------------- #
class V15Dataset(Dataset):
    """Windows over the cached frozen states + the precomputed label artifact.

    Item: states [W, S] fp16 · actions [W, 3] · v0 [] · traj_tgt [4, 2] ·
    vt_band [] long · route [] long · route_graded [] · ep/last.

    ``label_set``:
      ``v21``     the REPAIRED labels — fixed VTARGET mint (smoothed track,
                  enforced lookahead floor, DROPPED where untrustworthy) and
                  ``refb_labels`` v2.1 route (adaptive horizon, explicit
                  ROUTE_UNKNOWN, net_dyaw in the decision) + graded target.
      ``legacy``  the pre-repair labels — the planner_p2 VTARGET mint off the
                  raw jittery track with its silent hold-speed fallback, and the
                  v1 route labeler that silently emits STRAIGHT for the ~70 % of
                  windows it cannot judge. This arm exists ONLY to measure what
                  the repairs bought; it is not a shipping configuration.
    """

    def __init__(self, states_pt: str, poses_pt: str, labels_pt: str,
                 stride: int = 1, episodes: int = 0, label_set: str = "v21"):
        sd = torch.load(states_pt, weights_only=False)
        pd = torch.load(poses_pt, weights_only=False)
        ld = torch.load(labels_pt, weights_only=False)
        n = min(len(sd["eids"]), len(pd["eids"]), len(ld["eids"]))
        if not (sd["eids"][:n] == pd["eids"][:n] == ld["eids"][:n]):
            raise SystemExit("state / pose / label caches disagree on episode "
                             "order — refusing to train on misaligned labels")
        self.states = sd["states"]
        self.trunk_ckpt = sd.get("trunk_ckpt")
        self.label_set = label_set
        n_ep = n if not episodes else min(episodes, n)
        self.index: list[tuple[int, int]] = []
        self.traj: list[torch.Tensor] = []
        self.acts: list[torch.Tensor] = []
        self.v0: list[torch.Tensor] = []
        vt_key = "vt_band_v2" if label_set == "v21" else "vt_band_raw"
        self.vband = ld[vt_key]
        self.vspeed = ld["vt_v2" if label_set == "v21" else "vt_raw"]
        if label_set == "v21":
            self.route = ld["route_v21"]
            self.rgraded = ld["route_graded"]
        else:                              # v1: no UNKNOWN, no graded signal
            self.route = ld["route_legacy"]
            self.rgraded = [torch.zeros_like(x, dtype=torch.float32)
                            for x in ld["route_legacy"]]
        self.label_stats = ld.get("stats", {})
        for e in range(n_ep):
            po = torch.as_tensor(pd["poses"][e], dtype=torch.float32)
            ac = torch.as_tensor(pd["actions"][e], dtype=torch.float32)
            n_w = po.shape[0] - WINDOW - K_MAX
            self.acts.append(ac)
            self.v0.append(po[:, 3])
            if n_w <= 0:
                self.traj.append(torch.zeros(0, len(HORIZONS), 2))
                continue
            last = torch.arange(n_w) + WINDOW - 1
            yaw = po[last, 2]
            wps = [_ego(po[last + k, :2] - po[last, :2], yaw) for k in HORIZONS]
            self.traj.append(torch.stack(wps, dim=1))          # [n, 4, 2]
            if self.vband[e].shape[0] != n_w:
                raise SystemExit(
                    f"label/window count mismatch on episode {e}: "
                    f"{self.vband[e].shape[0]} labels vs {n_w} windows")
            self.index.extend((e, int(x)) for x in range(0, n_w, stride))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        e, t = self.index[i]
        last = t + WINDOW - 1
        v0 = self.v0[e][last]
        st = torch.as_tensor(self.states[e][t:t + WINDOW]).float()
        a = self.acts[e][t:t + WINDOW]                          # [W, 2]
        a3 = torch.cat([a, (v0 / SPEED_SCALE).expand(WINDOW, 1)], dim=-1)
        return {"states": st, "actions": a3, "v0": v0,
                "traj_tgt": self.traj[e][t], "vt_band": self.vband[e][t],
                "vt_speed": self.vspeed[e][t],
                "route": self.route[e][t], "route_graded": self.rgraded[e][t],
                "ep": e, "last": last}


# --------------------------------------------------------------------------- #
# eval — the TanitEval window protocol, on cached states                       #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(head, predictor, ds_val, probes, cfg, device, stride=8,
             batch=64, episodes=40, steps=None):
    """ADE/FDE/miss at the TanitEval waypoints over the first ``episodes`` val
    episodes with stride 8 — the same windows ``taniteval`` scores."""
    head.eval()
    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < episodes and t % stride == 0]
    preds, gts, v0s, eids, fans = [], [], [], [], []
    for b0 in range(0, len(sel), batch):
        idx = sel[b0:b0 + batch]
        items = [ds_val[i] for i in idx]
        st = torch.stack([x["states"] for x in items]).to(device)
        ac = torch.stack([x["actions"] for x in items]).to(device)
        v0 = torch.stack([x["v0"] for x in items]).to(device)
        vb = torch.stack([x["vt_band"] for x in items]).to(device)
        rt = torch.stack([x["route"] for x in items]).to(device)
        rg = torch.stack([x["route_graded"] for x in items]).to(device)
        vs = torch.stack([x["vt_speed"] for x in items]).to(device)
        imag = None
        if cfg.cond_imagination:
            imag = imagine_probes(predictor, st, ac, probes, cfg.imag_read,
                                  v0 / SPEED_SCALE)
        out = head(st, v0, imagined=imag, vt_band=vb, route=rt,
                   route_graded=rg, vt_speed=vs, steps=steps)
        preds.append(out["traj"].float().cpu())
        fans.append(out["anchor_traj"].float().cpu())
        gts.append(torch.stack([x["traj_tgt"] for x in items]))
        v0s.append(v0.cpu())
        eids.extend(x["ep"] for x in items)
    head.train()
    p, g = torch.cat(preds), torch.cat(gts)
    err = (p - g).norm(dim=-1)                                  # [N, 4]
    # oracle-in-fan: the best proposal AVAILABLE, vs the one SELECTED. The gap
    # separates "cannot propose it" from "cannot rank it" (the REF-C failure).
    fan = torch.cat(fans)
    fan_err = (fan - g[:, None]).norm(dim=-1).mean(dim=-1)      # [N, A]
    oracle = fan_err.min(dim=1).values
    sel = err.mean(dim=1)
    return {"n": int(p.shape[0]),
            "oracle_ade@2s": float(oracle.mean()),
            "sel_gap@2s": float((sel - oracle).mean()),
            "frac_sel_2x_worse": float((sel > 2.0 * oracle).float().mean()),
            "ade@0.5s": float(err[:, 0].mean()),
            "ade@1s": float(err[:, :2].mean()),
            "ade@1.5s": float(err[:, :3].mean()),
            "ade@2s": float(err.mean()),
            "fde@2s": float(err[:, -1].mean()),
            "miss@2m": float((err[:, -1] > 2.0).float().mean()),
            "_err": err, "_eid": eids}


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-train", required=True)
    ap.add_argument("--poses-train", required=True)
    ap.add_argument("--states-val", required=True)
    ap.add_argument("--poses-val", required=True)
    ap.add_argument("--labels-train", required=True)
    ap.add_argument("--labels-val", required=True)
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cond", choices=("a", "ab", "abc"), default="abc")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--label-set", choices=("v21", "legacy"), default="v21",
                    help="v21 = the REPAIRED labels (fixed VTARGET mint + "
                         "refb_labels v2.1 route). legacy = the pre-repair "
                         "labels (raw jittery VTARGET mint + the v1 route "
                         "labeler that silently emits STRAIGHT) — the control "
                         "arm that measures what the repairs bought.")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    dev = a.device
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = v15_ablation_config(states=True,
                              imagination=a.cond in ("ab", "abc"),
                              vtarget=a.cond == "abc")

    # --- frozen trunk: only the predictor is needed at train time -----------
    trunk, _grounding, trunk_step = load_frozen_v1(a.trunk, dev)
    predictor = trunk.predictor
    del trunk.encoder                       # states are cached; free the ViT
    torch.cuda.empty_cache()

    anc = torch.load(a.anchors, weights_only=False)
    anchors = anc["anchors"] if isinstance(anc, dict) else anc
    prb = torch.load(a.probes, weights_only=False)
    probes = (prb["probes"] if isinstance(prb, dict) else prb).to(dev)

    head = FlagshipV15Head(cfg).to(dev)
    head.load_anchors(anchors.to(dev))
    pb = param_breakdown(head)
    print(f"[v15] cond={a.cond} head params={pb['total']:,} {pb}", flush=True)

    ds = V15Dataset(a.states_train, a.poses_train, a.labels_train,
                    episodes=a.episodes, label_set=a.label_set)
    ds_val = V15Dataset(a.states_val, a.poses_val, a.labels_val,
                        episodes=40, label_set=a.label_set)
    print(f"[data] label_set={a.label_set} train windows={len(ds)} "
          f"val windows={len(ds_val)} label_stats={json.dumps(ds.label_stats)}",
          flush=True)

    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, drop_last=True,
                    num_workers=a.workers, persistent_workers=a.workers > 0,
                    pin_memory=True, prefetch_factor=4 if a.workers else None)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)

    (out / "config.json").write_text(json.dumps({
        "arch": "flagship-v1.5 (frozen v1 trunk + REF-C anchored-diffusion head)",
        "cond": a.cond, "cfg": dataclasses.asdict(cfg), "args": vars(a),
        "trunk": {"ckpt": a.trunk, "step": trunk_step, "frozen": True},
        "anchors": {k: v for k, v in anc.items() if k != "anchors"}
        if isinstance(anc, dict) else {},
        "probes": {k: v for k, v in prb.items() if k != "probes"}
        if isinstance(prb, dict) else {},
        "param_breakdown": pb,
        "label_set": a.label_set, "label_stats_train": ds.label_stats,
        "optimizer": {"kind": "AdamW", "lr": a.lr, "wd": 0.01,
                      "warmup": a.warmup, "schedule": "cosine"},
        "loss_weights": {"traj": 1.0, "anchor_cls": 1.0, "refined_cls": 1.0},
        "scoring_fix": ("selection ranks the REFINED fan via sel_score "
                        "(refined confidence + gated longitudinal term), NOT "
                        "the t=0 anchor logits REF-C selects on"),
    }, indent=2, default=str), encoding="utf-8")

    log_f = (out / "train_log.jsonl").open("a")
    ckpt_p = out / "ckpt.pt"
    step = 0
    if ckpt_p.exists():
        ck = torch.load(ckpt_p, map_location=dev, weights_only=False)
        head.load_state_dict(ck["head"]); opt.load_state_dict(ck["opt"])
        step = int(ck["step"]) + 1
        print(f"[resume] step {step}", flush=True)

    anchors_d = head.decoder.anchors
    it = iter(dl)
    t0 = time.time()
    best = float("inf")
    while step < a.steps:
        lr = cosine_lr(step, a.steps, a.warmup, a.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        st = b["states"].to(dev, non_blocking=True)
        ac = b["actions"].to(dev, non_blocking=True)
        v0 = b["v0"].to(dev, non_blocking=True)
        vb = b["vt_band"].to(dev, non_blocking=True)
        rt = b["route"].to(dev, non_blocking=True)
        rg = b["route_graded"].to(dev, non_blocking=True)
        vs = b["vt_speed"].to(dev, non_blocking=True)
        tgt = b["traj_tgt"].to(dev, non_blocking=True)

        imag = None
        if cfg.cond_imagination:
            imag = imagine_probes(predictor, st, ac, probes, cfg.imag_read,
                                  v0 / SPEED_SCALE)
        o = head(st, v0, imagined=imag, vt_band=vb, route=rt, route_graded=rg,
                 vt_speed=vs)
        L = v15_losses(o, anchors_d, tgt)
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        gn = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0))
        opt.step()

        if step % a.log_every == 0 or step == a.steps - 1:
            row = {"step": step, "lr": round(lr, 7),
                   "loss": round(float(L["loss"]), 5),
                   "traj": round(float(L["traj"]), 5),
                   "cls": round(float(L["cls"]), 5),
                   "cls_refined": round(float(L["cls_refined"]), 5),
                   "anchor_acc": round(float(L["anchor_acc"]), 4),
                   "rank_acc": round(float(L["rank_acc"]), 4),
                   "train_ade": round(float(L["ade"]), 4),
                   "oracle_ade": round(float(L["oracle_ade"]), 4),
                   "sel_gap": round(float(L["sel_gap"]), 4),
                   "sel_2x_worse": round(
                       float(L["frac_sel_2x_worse_than_oracle"]), 4),
                   "gnorm": round(gn, 3),
                   "elapsed_s": round(time.time() - t0, 1),
                   **{k: round(v, 4) for k, v in o["telemetry"].items()}}
            print(json.dumps(row), flush=True)
            log_f.write(json.dumps(row) + "\n"); log_f.flush()

        if step > 0 and step % a.eval_every == 0:
            ev = evaluate(head, predictor, ds_val, probes, cfg, dev)
            ev.pop("_err"); ev.pop("_eid")
            row = {"step": step, "val": {k: round(v, 5) if isinstance(v, float)
                                         else v for k, v in ev.items()}}
            print(json.dumps(row), flush=True)
            log_f.write(json.dumps(row) + "\n"); log_f.flush()
            if ev["ade@2s"] < best:
                best = ev["ade@2s"]
                torch.save({"head": head.state_dict(), "step": step,
                            "cfg": dataclasses.asdict(cfg), "val": ev},
                           out / "ckpt_best.pt")

        if step > 0 and step % a.save_every == 0:
            tmp = ckpt_p.with_suffix(".tmp")
            torch.save({"head": head.state_dict(), "opt": opt.state_dict(),
                        "step": step, "cfg": dataclasses.asdict(cfg)}, tmp)
            tmp.replace(ckpt_p)
        step += 1

    tmp = ckpt_p.with_suffix(".tmp")
    torch.save({"head": head.state_dict(), "opt": opt.state_dict(),
                "step": step - 1, "cfg": dataclasses.asdict(cfg)}, tmp)
    tmp.replace(ckpt_p)
    ev = evaluate(head, predictor, ds_val, probes, cfg, dev)
    ev.pop("_err"); ev.pop("_eid")
    (out / "metrics.json").write_text(json.dumps(
        {"final_step": step - 1, "val": ev, "best_val_ade2s": best,
         "param_breakdown": pb, "cond": a.cond,
         "wallclock_s": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "cond": a.cond, "val": ev}), flush=True)


if __name__ == "__main__":
    main()
