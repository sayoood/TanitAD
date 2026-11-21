"""Roll a P1-gate checkpoint over the FULL held-out eval split and dump the
per-window arrays the four-family panel needs.

⛔ Everything that touches the model or the data is the TRAINER'S OWN code path
(`refc_v3_train` imported as a module, `_pin_trainer_cfg`, `V3Dataset`,
`build_v2_providers`, `waypoint_targets`) -- a second construction path would be
a second contract, and this gate is about whether one flag moved the model, not
about whether I can rebuild it.

The emitted plan is `anchor_traj[argmax(anchor_logits)]` -- the fan the decoder
actually emitted, selected by the head that ranks it. T1 = SELF-ACTION OPEN
LOOP: the planner is fed measured state at t0 and rolls its own plan. Nothing
here is closed loop.
"""
from __future__ import annotations
import argparse, json, os, sys, types
import numpy as np
import torch

STACK = r"C:/Users/Admin/tanitad-p1gate/stack"
sys.path.insert(0, STACK)
sys.path.insert(0, STACK + "/scripts")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import refc_v3_train as T                                    # noqa: E402
from tanitad.refs import refc_v3 as v3                       # noqa: E402
from tanitad.data.v2_dataset import build_v2_providers       # noqa: E402
from tanitad.data.v2_dataset import stable_episode_id        # noqa: E402
import refb_labels                                            # noqa: E402
from tanitad.data import v7_labels as v7l                     # noqa: E402


def args_from_launch(run: str) -> argparse.Namespace:
    """Rebuild the EXACT launch args from the banked preflight argv.

    ⛔ NOT from `config.json`. The stamp is a RECORD, not a round-trip: it
    renders `anchors` as the artifact stamp (a dict, not a path) and does not
    carry every flag that changes a tensor SHAPE. Rebuilding from it silently
    produced a 128-anchor model for a 117-anchor checkpoint and a 5-wide
    measurement head for a 6-wide one -- i.e. a DIFFERENT MODEL that would have
    been scored as this one. The preflight file holds the argv that actually ran.
    """
    arm = os.path.basename(run.rstrip("/\\"))            # e.g. head_s0
    name, seed = arm.rsplit("_s", 1)
    pf = json.load(open(os.path.join(os.path.dirname(run),
                                     f"preflight_s{seed}.json"),
                        encoding="utf-8"))
    argv = pf["arms"][name]["argv"]
    ns = T.build_parser().parse_args(argv)
    ns.workers = 0
    return ns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-windows", type=int, default=100000)
    ap.add_argument("--stride", type=int, default=1,
                    help="score every Nth window. Windows inside an "
                         "episode overlap heavily and the estimator's "
                         "unit is the EPISODE, so a stride costs "
                         "little power and keeps all 35 clusters.")
    a = ap.parse_args()

    args = args_from_launch(a.run)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    art = T._read_anchor_artifact(args)
    cfg = T._pin_trainer_cfg(
        v3.refc_v3_sized_config(args.size, hier=args.arm == "hier"), args)
    T._check_anchor_artifact_against_cfg(art, cfg, args)
    if args.graft_lan or args.goal_str:
        from tanitad.refs import refc
        cfg.core.lan = refc.LanConfig(k=len(args.lan_arclengths))
    model = v3.RefCV3Model(cfg).to(dev)
    T._pin_refcv5_seams(cfg, args)
    model = v3.RefCV3Model(cfg).to(dev)
    sd = torch.load(a.run + "/ckpt.pt", map_location=dev, weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[dump] load: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    e_eps = build_v2_providers([args.eval_cache], lru_size=8)
    want_lan = bool(args.graft_lan or args.goal_str)
    dcls = T.lan_dataset_class(T.V3Dataset) if want_lan else T.V3Dataset
    kw = dict(window=cfg.core.window, max_horizon=20,
              channels=cfg.core.encoder.in_channels)
    if want_lan:
        kw["lan_cfg"] = T.DataLanConfig(
            arclengths_m=tuple(args.lan_arclengths),
            min_lead_m=args.lan_min_lead_m)
    ds = dcls(e_eps, **kw)
    ds.u8_frames = bool(getattr(args, "u8_batches", False))
    e_lab, e_man = v7l.load_v7_labels(args.eval_labels, allow_oracle_nav=True)
    ds.v7_by_sid = {stable_episode_id(l.clip_id): l for l in e_lab}
    ds.v7_dt = 0.1
    if getattr(args, "nav_from_v7", False):
        ds.enable_nav_from_v7(e_man)
    print(f"[dump] {len(e_eps)} eval providers, {len(ds)} windows")

    P, G, V, E, SV = [], [], [], [], []
    PL, FE, FV, NW = [], [], [], []
    EXTRA: dict[str, list] = {}
    # the dataset's window index -> NOW frame is t + w - 1 (see V3Dataset)
    keep = list(range(0, len(ds), max(a.stride, 1)))
    now_frames = np.array([ds.index[i][1] + cfg.core.window - 1 for i in keep])
    ep_of_win = np.array([int(ds.episodes[ds.index[i][0]].episode_id)
                          for i in keep])
    ds_scored = torch.utils.data.Subset(ds, keep)
    print(f"[dump] stride {a.stride}: scoring {len(keep)} of {len(ds)} windows")
    dl = torch.utils.data.DataLoader(ds_scored, batch_size=args.batch,
                                     shuffle=False, num_workers=0)
    n = 0
    with torch.no_grad():
        for batch in dl:
            frames = T.frames_to_device(batch["frames"], dev)
            pose_last = batch["pose_last"].to(dev)
            fut_ext = batch["future_poses_ext"].to(dev)
            fut_valid = batch["future_valid_ext"].to(dev)
            nav_cmd = batch["nav_cmd"].to(dev)
            lan = batch["lan"].to(dev) if "lan" in batch else None
            v0 = pose_last[:, 3]
            ego_state = None
            if getattr(cfg, "ego_state_inject", False):
                ego_state = v3.ego_state_from_batch(
                    {"pose_last": pose_last, "actions": batch["actions"]},
                    device=dev)
            out = model(frames, nav_cmd=nav_cmd, v0=v0,
                        steps=cfg.core.decoder.diffusion_steps, lan=lan,
                        ego_state=ego_state)
            bank = out["anchor_traj"]                       # [B, N, S, 2]
            sel = out["anchor_logits"].argmax(dim=1)        # [B]
            pred = bank.gather(
                1, sel[:, None, None, None].expand(
                    -1, 1, bank.shape[2], bank.shape[3])).squeeze(1)
            tgt = refb_labels.waypoint_targets(pose_last, fut_ext,
                                               cfg.core.trajectory.horizons)
            sv = torch.stack([fut_valid[:, h - 1]
                              for h in cfg.core.trajectory.horizons], dim=1)
            P.append(pred.float().cpu().numpy())
            G.append(tgt.float().cpu().numpy())
            SV.append(sv.cpu().numpy())
            V.append(v0.float().cpu().numpy())
            E.append(batch["episode_id"].cpu().numpy())
            for k, src in (("lat_tac_pred", out.get("lat_logits_tac")),
                           ("lon_tac_pred", out.get("lon_logits_tac")),
                           ("route_pred", out.get("route_logits"))):
                if src is not None:
                    EXTRA.setdefault(k, []).append(
                        src.argmax(-1).cpu().numpy())
            for k, bk in (("lat_tac_gt", "lat_v7"), ("lon_tac_gt", "lon_v7"),
                          ("route_gt", "route_target")):
                if bk in batch:
                    EXTRA.setdefault(k, []).append(
                        batch[bk].cpu().numpy())
            PL.append(pose_last.float().cpu().numpy())
            FE.append(fut_ext.float().cpu().numpy())
            FV.append(fut_valid.cpu().numpy())
            n += pred.shape[0]
            if n >= a.max_windows:
                break
    d = dict(pred=np.concatenate(P), gt=np.concatenate(G),
             slot_valid=np.concatenate(SV), v0=np.concatenate(V),
             eid=np.concatenate(E),
             pose_last=np.concatenate(PL), fut_ext=np.concatenate(FE),
             fut_valid=np.concatenate(FV),
             now_frame=now_frames[:n], ep_of_win=ep_of_win[:n],
             horizons=np.array(cfg.core.trajectory.horizons))
    for k, v in EXTRA.items():
        d[k] = np.concatenate(v)
    assert (d["eid"] == d["ep_of_win"]).all(), \
        "window order does not match ds.index -- the frame join would be wrong"
    np.savez_compressed(a.out, **d)
    print(f"[dump] wrote {a.out}: n_windows={d['pred'].shape[0]} "
          f"n_episodes={len(set(d['eid'].tolist()))} "
          f"S={d['pred'].shape[1]} horizons={d['horizons'].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
