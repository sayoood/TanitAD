"""refcv7-r101-s0 map + box diagnostics -- ONE pass of ONE checkpoint over ONE window set (SPEC.md).

The model is built by the run's own eval loader (`tanitad.eval.refcv7_loader.build_model`, STRICT load, G-DVB,
stamp checks); every window goes through the trainer's own `compute_losses_v3` in eval mode under no_grad.
The 10 cm logits are captured by wrapping `map_head_hires.map_hires_loss_row` -- the ORIGINAL still runs and its
own `inter`/`union` per class x band is compared, every batch, with this script's (CONTROL C1). The box rows
are the trainer's own `_det_pack_box3d` / `_det_pack_agent` packs, pickled as they are.

Outputs (small, aggregated on the fly -- no logit dump): <out>/<tag>.acc.pt (per-episode counts, histograms,
M-f census, optional 1 % TRAIN cell subsample), <tag>.packs.pkl, <tag>.json (record + controls).

Env (set by the caller, BEFORE import): REFCV6_REPO=<launch tree>, REFCV6_KIT=/home/nvidia,
REFCV6_REMAP_OVERRIDES=<json> for the TRAIN subsets. ⛔ Prints sha12 only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def sha12(c):
    return hashlib.sha256(str(c).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["eval_inrun", "eval_diag", "train_diag", "train_calib256"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--k-per-ep", type=int, default=8)
    ap.add_argument("--fit", default=None, help="JSON with fitted decision params (delta, tau_phat, tau_q)")
    ap.add_argument("--subsample", type=float, default=0.0)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--no-mf", action="store_true", help="skip the M-f census (milestone passes)")
    a = ap.parse_args()

    import numpy as np
    import torch
    import diag_metrics as dm
    from tanitad.eval import refcv7_loader as L
    tr = L.trainer()
    mhr = tr._mhr
    det = tr._det_metrics
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    torch.manual_seed(0)

    config = L.load_config(Path(a.run_dir) / "config.json")
    model, cfg, args, mrec = L.build_model(config, a.ckpt, device="cuda", strict=True)
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config)
    sha_of_eid = dict(e_ds._vis1_sha12)               # episode_id -> sha12 (enable_vis1)
    print(f"[diag] built: step {mrec['state_dict']['step']} episodes {len(e_eps)} windows {len(e_ds)} "
          f"build {time.time() - t_start:.0f}s", flush=True)

    # ---- the window set ------------------------------------------------------------------------- #
    if a.split == "eval_inrun":
        idx = L.inrun_eval_perm(e_ds, int(args.eval_batches), int(args.batch))
        batch, drop_last = int(args.batch), True
    elif a.split == "train_calib256":
        idx, n_miss = det.calib_indices(e_ds.index, lambda e: sha_of_eid[int(e_ds.episodes[e].episode_id)])
        if n_miss:
            raise SystemExit(f"[diag] {n_miss} calibration windows missing from the subset cache")
        batch, drop_last = a.batch, False
    else:
        by_ep = {}
        for i, (e_i, t) in enumerate(e_ds.index):
            by_ep.setdefault(int(e_i), []).append((int(t), i))
        idx = []
        K = int(a.k_per_ep)
        for e_i in sorted(by_ep, key=lambda e: sha_of_eid[int(e_ds.episodes[e].episode_id)]):
            ts = sorted(by_ep[e_i])
            for j in range(K):
                idx.append(ts[int((j + 0.5) * len(ts) / K)][1])
        batch, drop_last = a.batch, False
    win_sha = [sha_of_eid[int(e_ds.episodes[e_ds.index[i][0]].episode_id)] for i in idx]
    win_t = [int(e_ds.index[i][1]) for i in idx]
    episodes = sorted(set(win_sha))
    ep_pos = {s: k for k, s in enumerate(episodes)}
    n_ep = len(episodes)
    print(f"[diag] split {a.split}: {len(idx)} windows over {n_ep} episodes, batch {batch}", flush=True)

    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(e_ds, idx), batch_size=batch, shuffle=False,
                                     num_workers=a.workers, drop_last=drop_last,
                                     prefetch_factor=(2 if a.workers > 0 else None))

    # ---- decisions ------------------------------------------------------------------------------ #
    br = model._map_hires
    cw = model._map_hires_class_weight.detach().float().cuda()
    lw = dm.log_w(cw, "cuda")
    H, W = (int(v) for v in br.cfg.out_hw)
    nb = (H + dm.BAND_ROWS - 1) // dm.BAND_ROWS
    M = dm.band_matrix(H, "cuda")
    decisions = {"pc": ("argmax", (-lw).tolist()), "raw": ("argmax", [0.0] * 8)}
    fit = None
    if a.fit:
        fit = json.load(open(a.fit))
        decisions["off"] = ("argmax", [float(-lw[c]) + float(fit["delta"][c]) for c in range(8)])
        decisions["thr_phat"] = ("thr_phat", [float(v) for v in fit["tau_phat_logit"]])
        decisions["thr_q"] = ("thr_q", [float(v) for v in fit["tau_q_logit"]])
    stats = ["pred", "gt", "inter"] + [f"tp{s}{k}" for k in dm.TOL_K for s in ("p", "g")]
    acc = {d: {s: np.zeros((n_ep, 8, nb)) for s in stats} for d in decisions}
    sup_cells = np.zeros((n_ep, nb))
    hist = {"phat": torch.zeros(8, nb, 2, dm.HIST_NB, dtype=torch.int64, device="cuda"),
            "q": torch.zeros(8, nb, 2, dm.HIST_NB, dtype=torch.int64, device="cuda")}
    runs = {c: np.zeros((nb, dm.RUN_CAP + 1), np.int64) for c in dm.THIN}
    reg = {"gt_drivable": torch.zeros(nb, len(dm.REG_BUCKETS), dtype=torch.int64, device="cuda"),
           "pred_pc_drivable": torch.zeros(nb, len(dm.REG_BUCKETS), dtype=torch.int64, device="cuda")}
    sub = {"z": [], "y": [], "band": [], "ep": []}
    c1 = {"max_abs_diff": 0.0, "n_batches": 0, "n_compared": 0}
    CTX = {}

    orig = mhr.map_hires_loss_row

    def wrapped(logits, codes, *, class_weight=None, lift_valid_025=None, with_metrics=True,
                decision_rule="raw"):
        row = orig(logits, codes, class_weight=class_weight, lift_valid_025=lift_valid_025,
                   with_metrics=with_metrics, decision_rule=decision_rule)
        with torch.no_grad():
            z = logits.detach().float()
            cd = codes.to(z.device)
            lv = (torch.ones_like(cd, dtype=torch.bool) if lift_valid_025 is None
                  else mhr.lift_valid_to_fine(lift_valid_025, tuple(z.shape[2:])).to(z.device))
            sup = (cd != 255) & lv
            y = cd.long()
            G = dm.onehot_masks(y, sup)
            wins = CTX["wins"]                       # window positions of the labelled rows
            eps = [ep_pos[win_sha[w]] for w in wins]
            sup_b = dm.per_window_bands(sup[:, None], M)[:, 0].cpu().numpy()
            for j, e in enumerate(eps):
                sup_cells[e] += sup_b[j]
            masks = dm.decision_masks(z, sup, lw, decisions)
            for d, P in masks.items():
                tc = dm.tolerant_counts(P, G, M)
                tcn = {s: tc[s].cpu().numpy() for s in stats}
                for j, e in enumerate(eps):
                    for s in stats:
                        acc[d][s][e] += tcn[s][j]
            # CONTROL C1: the trainer's own inter/union (declared rule + raw) vs mine, this batch
            if with_metrics and "map_hires_inter_lane_0_20" in row:
                bks = mhr.band_keys_for_rows(H)
                for d, ik, uk in (("pc", "inter", "union"), ("raw", "interraw", "unionraw")):
                    mi = dm.per_window_bands(masks[d] & G, M).sum(0).cpu().numpy()
                    mu = dm.per_window_bands(masks[d] | G, M).sum(0).cpu().numpy()
                    for ci in range(8):
                        for bi, bk in enumerate(bks):
                            for mine, key in ((mi, ik), (mu, uk)):
                                theirs = float(row[mhr.per_class_key(key, ci, bk)])
                                c1["max_abs_diff"] = max(c1["max_abs_diff"], abs(theirs - float(mine[ci, bi])))
                                c1["n_compared"] += 1
                c1["n_batches"] += 1
            # histograms (M-c iii, M-d)
            hist["phat"] += dm.score_histograms(dm.class_logits(z - lw.view(1, -1, 1, 1)), G, sup, nb)
            hist["q"] += dm.score_histograms(dm.class_logits(z), G, sup, nb)
            if not a.no_mf:
                cn, sn = cd.cpu().numpy(), sup.cpu().numpy()
                for c in dm.THIN:
                    runs[c] += dm.run_length_hist(cn, sn, c, nb)
                edge = (cd == 5) & sup
                reg["gt_drivable"] += dm.registration_hist(edge, dm.boundary((cd == 1) & sup, sup), nb)
                reg["pred_pc_drivable"] += dm.registration_hist(edge, dm.boundary(masks["pc"][:, 1], sup), nb)
            if a.subsample > 0:
                for j, w in enumerate(wins):
                    g = torch.Generator(device="cpu").manual_seed(1000003 + int(w))
                    m = sup[j].cpu()
                    pick = (torch.rand(m.shape, generator=g) < float(a.subsample)) & m
                    ii = pick.nonzero(as_tuple=False)
                    sub["z"].append(z[j][:, ii[:, 0], ii[:, 1]].T.half().cpu())
                    sub["y"].append(y[j][ii[:, 0], ii[:, 1]].to(torch.uint8).cpu())
                    sub["band"].append((ii[:, 0] // dm.BAND_ROWS).to(torch.uint8))
                    sub["ep"].append(torch.full((ii.shape[0],), ep_pos[win_sha[w]], dtype=torch.int16))
        return row

    mhr.map_hires_loss_row = wrapped

    packs = {"box3d": [], "agent": []}
    inrun_acc, nb_done = {}, 0
    t_loop = time.time()
    n_batches = len(dl)
    pos = 0
    for bi, b in enumerate(dl):
        if a.max_batches and bi >= a.max_batches:
            break
        nbatch = int(b["map_fine_label"].shape[0])
        wins_all = list(range(pos, pos + nbatch))
        pos += nbatch
        lab = b["map_fine_label"].bool().tolist()
        CTX["wins"] = [w for w, l in zip(wins_all, lab) if l]
        el = tr.compute_losses_v3(model, b, "cuda", mode=args.mode, ablate_frames=args.ablate_frames)
        for hd in ("box3d", "agent"):
            for pk in el.get(f"_det_pack_{hd}") or []:
                pk = dict(pk)
                pk["sha12"] = sha_of_eid.get(int(pk["ep"])) if pk.get("ep") is not None else None
                packs[hd].append(pk)
        if a.split == "eval_inrun":
            for k, v in el.items():
                if torch.is_tensor(v) and v.ndim == 0:
                    inrun_acc[k] = inrun_acc.get(k, 0.0) + float(v.detach())
                elif isinstance(v, (int, float, bool)):
                    inrun_acc[k] = inrun_acc.get(k, 0.0) + float(v)
        nb_done += 1
        if bi % 10 == 0 or bi == n_batches - 1:
            el_s = time.time() - t_loop
            print(f"[diag] ZZ{bi + 1}-{n_batches}ZZ {el_s:.0f}s ({el_s / (bi + 1):.2f} s/batch) "
                  f"cuda_max {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    mhr.map_hires_loss_row = orig

    # ---- bank ----------------------------------------------------------------------------------- #
    accd = {"decisions": {d: list(v) for d, v in decisions.items()},
            "acc": {d: {s: torch.from_numpy(v) for s, v in acc[d].items()} for d in acc},
            "sup_cells": torch.from_numpy(sup_cells), "episodes": episodes,
            "hist": {k: v.cpu() for k, v in hist.items()},
            "runs": {c: torch.from_numpy(v) for c, v in runs.items()},
            "reg": {k: v.cpu() for k, v in reg.items()},
            "class_weight": cw.cpu(), "band_keys": list(mhr.band_keys_for_rows(H)), "out_hw": [H, W]}
    if a.subsample > 0 and sub["z"]:
        accd["sub"] = {k: torch.cat(v) for k, v in sub.items()}
    torch.save(accd, out_dir / f"{a.tag}.acc.pt")
    with open(out_dir / f"{a.tag}.packs.pkl", "wb") as fh:
        pickle.dump(packs, fh, protocol=4)
    rec = {"tag": a.tag, "split": a.split, "ckpt": a.ckpt, "step": mrec["state_dict"]["step"],
           "n_windows": len(idx), "n_windows_done": pos, "n_batches_done": nb_done, "batch": batch,
           "n_episodes": n_ep, "window_sha12_t_sha256": hashlib.sha256(
               json.dumps(list(zip(win_sha, win_t))).encode()).hexdigest(),
           "decisions": {d: list(v) for d, v in decisions.items()}, "fit": fit,
           "control_C1": c1, "n_packs": {k: len(v) for k, v in packs.items()},
           "loader_departures": mrec.get("departures"), "dvb": mrec.get("declared_vs_built"),
           "state_dict": {k: mrec["state_dict"][k] for k in ("strict", "missing", "unexpected", "step")},
           "dataset": {"n_episodes": drec.get("n_episodes"), "n_windows": drec.get("n_windows"),
                       "map_fine": drec.get("map_fine")},
           "cuda_max_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
           "wall_s": time.time() - t_start, "loop_s": time.time() - t_loop}
    if a.split == "eval_inrun" and nb_done:
        erow = tr._eval_row_from_acc(inrun_acc, nb_done, model)
        for hd, pk in packs.items():
            if pk:
                for k, v in det.summarise(pk, hd).items():
                    erow[k] = None if (isinstance(v, float) and v != v) else round(float(v), 5)
        rec["inrun_row"] = {k: v for k, v in erow.items() if not k.startswith("eval__det_pack")}
    with open(out_dir / f"{a.tag}.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print(f"[diag] DONE {a.tag}: C1 max_abs_diff {c1['max_abs_diff']} over {c1['n_compared']} "
          f"values / {c1['n_batches']} batches; wall {rec['wall_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
