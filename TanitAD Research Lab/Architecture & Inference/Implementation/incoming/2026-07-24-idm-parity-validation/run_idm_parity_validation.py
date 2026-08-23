"""ON-TARGET, LARGER-SCALE confirmation of the YouTube-IDM GO: does pseudo-label
pretraining help a WM on the ACTUAL PARITY corpus (physicalai-train-e438721ae894)?
The comma/rig-B proxies said GO (~96% of ceiling); this validates it on the domain
the WM is actually fine-tuned on, with readable YAW (unlike comma), at ~6x the scale.

⚠ PARITY FIREWALL: this is a SIDE de-risk. It reads parity-domain EPISODES as data
with its OWN IDM split; it does NOT create or affect any WM parity arm and does NOT
re-select the canonical WM episode selection. Licensing-FREE (parity, not YouTube).

Design (same paired 3-arm pretraining-benefit ablation as the proxy run):
  substrate : v1 frozen encoder.  model: IDMHead temporal trunk (small dynamics WM).
  labeler L : v1 + IDMHead{rigA[:60]+rigB[:60]+comma[:40]}  (the multi-domain pipeline)
  pretrain D: parity clips HELD OUT from L -> rigA[60:200] + rigB[60:220] (~300 clips),
              pseudo-labeled by L (realistic parity pseudo-quality).
  downstream: parity HELD-OUT = physicalai-VAL (episode-disjoint), low-data finetune
              (15 clips) + test (65). BOTH rigs. speed + YAW + trajectory.
  arms (paired/seed, identical finetune): FLOOR (random) / PSEUDO (pretrain D pseudo) /
              CEILING (pretrain D real).  KEY: fraction_of_ceiling per metric.

PRE-REG: pseudo beats floor (CI-separated) AND captures a substantial fraction of the
real-label ceiling ON PARITY -> scale-up de-risked on our actual target (GO ->
decision-grade); no benefit on parity -> a caveat the proxies missed, reported.
pod3, venv python, gpu_lock idm-parity-validation. Reuses cached v1 latents; encodes
only the additional parity chunk.
"""
from __future__ import annotations
import argparse, json, statistics as st, sys, time
from pathlib import Path
import torch

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
import idm_head as ih                                              # noqa: E402
import run_idm_proof as R                                          # noqa: E402

CACHED = "/workspace/tmp/branchb_eval/lat_flagshipv1"
PARITY = "/workspace/tmp/idm_parity/lat"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get_latent(tag, path, enc, readout, device):
    for d in (CACHED, PARITY):
        f = Path(d) / f"{tag}.pt"
        if f.exists():
            return torch.load(f, weights_only=False)
    ep = R._load_ep(path)
    z = R.encode_frames(enc, readout, ep["frames_u8"], device, batch=48)
    out = {"z": z, "poses": ep["poses"].float(), "actions": ep["actions"].float()}
    Path(PARITY).mkdir(parents=True, exist_ok=True)
    torch.save(out, Path(PARITY) / f"{tag}.pt")
    return out


def windows(tagpaths, enc, readout, device, k=4, stride=2):
    out = []
    for j, (tag, path) in enumerate(tagpaths):
        d = get_latent(tag, path, enc, readout, device)
        zw, sc, tj = ih.build_windows(d["z"].float(), d["poses"].float(),
                                      d["actions"].float(), k=k, stride=stride)
        if zw.shape[0]: out.append((zw, sc, tj))
        if j and j % 40 == 0: log(f"    windows {j}/{len(tagpaths)}")
    return out


def cat(lst): return (torch.cat([x[0] for x in lst]), torch.cat([x[1] for x in lst]),
                      torch.cat([x[2] for x in lst]))


def train_head(train, device, *, epochs, batch=256, lr=3e-4, wd=0.01, seed=0, init_state=None):
    torch.manual_seed(seed)
    Z, S, T = train
    std = ih.Standardizer.fit(S)
    head = ih.IDMHead(state_dim=2048, horizons=ih.DEFAULT_HORIZONS).to(device)
    if init_state is not None: head.load_state_dict(init_state)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = Z.shape[0]
    Z, S, T = Z.to(device), S.to(device), T.to(device)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * max(1, n // batch))
    g = torch.Generator(device=device).manual_seed(seed + 1)
    for _ in range(epochs):
        head.train(); perm = torch.randperm(n, generator=g, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            ld = ih.idm_loss(head(Z[idx]), S[idx], T[idx], std)
            opt.zero_grad(set_to_none=True); ld["loss"].backward(); opt.step(); sched.step()
    head.eval(); return head


@torch.no_grad()
def pseudo_targets(labeler, clips, device, b=2048):
    out = []
    for zw, sc, tj in clips:
        ps, pt = [], []
        for i in range(0, zw.shape[0], b):
            o = labeler(zw[i:i + b].to(device)); ps.append(o["scalars"].cpu()); pt.append(o["traj"].cpu())
        out.append((zw, torch.cat(ps), torch.cat(pt)))
    return out


@torch.no_grad()
def ev(head, clips, device):
    m = ih.evaluate(head, *cat(clips), device=device)
    return {"speed_r2": m["r2"]["speed"], "yaw_r2": m["r2"]["yaw_rate"],
            "ade_2s": m["ade_2s"], "n": m["n"]}


def frac(fl, ps, ce):
    return round((fl - ps) / (fl - ce), 3) if abs(fl - ce) > 1e-6 else None   # for ADE (lower better)


def frac_up(fl, ps, ce):
    return round((ps - fl) / (ce - fl), 3) if abs(ce - fl) > 1e-6 else None    # for R2 (higher better)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagship-ckpt", default="/workspace/tmp/idm/ckpt.pt")
    ap.add_argument("--train-cache", default="/workspace/pai_epcache/physicalai-train-e438721ae894")
    ap.add_argument("--val-cache", default="/workspace/pai_epcache/physicalai-val-f1b378f295ae")
    ap.add_argument("--comma-cache", default="/workspace/data/comma2k19-val-61c46fca8f7f")
    ap.add_argument("--train-rig-table", default="/workspace/tmp/idm/rig_table.json")
    ap.add_argument("--val-order", default="/workspace/tmp/val_clip_order.tsv")
    ap.add_argument("--calib-root", default="/workspace/pai_build")
    ap.add_argument("--work", default="/workspace/tmp/idm_parity")
    ap.add_argument("--out", default="/workspace/tmp/idm_parity/results_idm_parity_validation.json")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--pt-epochs", type=int, default=25)
    ap.add_argument("--ft-epochs", type=int, default=60)
    ap.add_argument("--rigA-pt-hi", type=int, default=200)
    ap.add_argument("--rigB-pt-hi", type=int, default=220)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.work).mkdir(parents=True, exist_ok=True); t0 = time.time()
    enc, ro, _ = R.load_encoder(args.flagship_ckpt, device)

    train_rig = json.loads(Path(args.train_rig_table).read_text())
    ta, tb = R.select_episodes(train_rig, args.train_cache, 400, 400)
    val_rig = R.build_rig_table(args.val_order, args.calib_root, str(Path(args.work) / "val_rig_table.json"))
    va, vb = R.select_episodes(val_rig, args.val_cache, 400, 400)
    comma = sorted(Path(args.comma_cache).glob("ep_*.pt"))
    A = lambda i, p: (f"tr_a_{i:05d}", p)
    B = lambda i, p: (f"tr_b_{i:05d}", p)
    # tagpaths
    rigA = [A(i, p) for i, (t, p) in enumerate(ta[:args.rigA_pt_hi])]
    rigB = [B(i, p) for i, (t, p) in enumerate(tb[:args.rigB_pt_hi])]
    CM = [(f"cm_{i:05d}", str(p)) for i, p in enumerate(comma[:80])]
    VA = [(f"va_a_{i:05d}", p) for i, (t, p) in enumerate(va)]     # 26 rigA val (cached)
    VB = [(f"va_b_{i:05d}", p) for i, (t, p) in enumerate(vb)]     # 54 rigB val (cached)

    # labeler L: multi-domain head on a labeled chunk (held out from pretrain D)
    log("building labeler {rigA[:60]+rigB[:60]+comma[:40]} ...")
    lab = train_head(cat(windows(rigA[:60] + rigB[:60] + CM[:40], enc, ro, device)),
                     device, epochs=50, seed=0)

    # pretrain corpus D: parity clips HELD OUT from L
    log("building pretrain corpus D = rigA[60:200]+rigB[60:220] (parity, held out from L) ...")
    D = windows(rigA[60:args.rigA_pt_hi] + rigB[60:args.rigB_pt_hi], enc, ro, device)
    D_pseudo = pseudo_targets(lab, D, device)
    log(f"  D = {len(D)} clips, {cat(D)[0].shape[0]} windows")

    # downstream: parity VAL (both rigs), low-data finetune + test
    valA = windows(VA, enc, ro, device); valB = windows(VB, enc, ro, device)
    ft = valA[:8] + valB[:7]                                       # 15 finetune clips (mixed)
    te = valA[8:] + valB[7:]                                       # 65 test clips
    log(f"  downstream: finetune {len(ft)} clips ({cat(ft)[0].shape[0]} win), "
        f"test {len(te)} ({cat(te)[0].shape[0]} win)")

    seeds = list(range(args.seeds))
    per_seed = []
    agg = {"floor": {"speed_r2": [], "yaw_r2": [], "ade_2s": []},
           "pseudo": {"speed_r2": [], "yaw_r2": [], "ade_2s": []},
           "ceiling": {"speed_r2": [], "yaw_r2": [], "ade_2s": []}}
    D_real, ft_cat = cat(D), cat(ft)
    D_ps_cat = cat(D_pseudo)
    for s in seeds:
        h_floor = train_head(ft_cat, device, epochs=args.ft_epochs, seed=s)
        h_ps = train_head(D_ps_cat, device, epochs=args.pt_epochs, seed=s)
        h_pseudo = train_head(ft_cat, device, epochs=args.ft_epochs, seed=s,
                              init_state={k: v.clone() for k, v in h_ps.state_dict().items()})
        h_re = train_head(D_real, device, epochs=args.pt_epochs, seed=s)
        h_ceiling = train_head(ft_cat, device, epochs=args.ft_epochs, seed=s,
                               init_state={k: v.clone() for k, v in h_re.state_dict().items()})
        e = {a: ev(h, te, device) for a, h in
             (("floor", h_floor), ("pseudo", h_pseudo), ("ceiling", h_ceiling))}
        for a in agg:
            for m in agg[a]: agg[a][m].append(e[a][m])
        per_seed.append({"seed": s, **e})
        log(f"  seed {s}: floor(sp {e['floor']['speed_r2']:+.3f} yaw {e['floor']['yaw_r2']:+.3f}) "
            f"pseudo(sp {e['pseudo']['speed_r2']:+.3f} yaw {e['pseudo']['yaw_r2']:+.3f}) "
            f"ceiling(sp {e['ceiling']['speed_r2']:+.3f} yaw {e['ceiling']['yaw_r2']:+.3f})")

    def ms(a, m): return (round(st.mean(agg[a][m]), 4), round(st.pstdev(agg[a][m]), 4))
    res = {"meta": {"experiment": "idm_parity_validation", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "flagship_md5": R.md5_of(args.flagship_ckpt), "device": device,
                    "seeds": args.seeds, "pt_epochs": args.pt_epochs, "ft_epochs": args.ft_epochs,
                    "parity_firewall": "SIDE de-risk; own IDM split of parity-domain data; no WM arm, no re-selection of canonical WM episodes; licensing-free (parity, not YouTube).",
                    "pretrain_clips": len(D), "pretrain_windows": int(cat(D)[0].shape[0]),
                    "downstream": "physicalai-val (both rigs), finetune 15 / test 65",
                    "pre_reg": "pseudo beats floor (all seeds) AND substantial fraction of ceiling ON PARITY -> GO decision-grade; else caveat"},
           "arms_mean_std": {a: {m: ms(a, m) for m in ("speed_r2", "yaw_r2", "ade_2s")} for a in agg},
           "per_seed": per_seed}
    # fractions per metric (paired per-seed means)
    fl, ps, ce = agg["floor"], agg["pseudo"], agg["ceiling"]
    res["fraction_of_ceiling"] = {
        "speed_r2": frac_up(st.mean(fl["speed_r2"]), st.mean(ps["speed_r2"]), st.mean(ce["speed_r2"])),
        "yaw_r2": frac_up(st.mean(fl["yaw_r2"]), st.mean(ps["yaw_r2"]), st.mean(ce["yaw_r2"])),
        "ade_2s": frac(st.mean(fl["ade_2s"]), st.mean(ps["ade_2s"]), st.mean(ce["ade_2s"]))}
    res["pseudo_beats_floor_all_seeds"] = {
        m: all(ps[m][i] > fl[m][i] for i in range(len(seeds))) if m != "ade_2s"
        else all(ps[m][i] < fl[m][i] for i in range(len(seeds))) for m in ("speed_r2", "yaw_r2", "ade_2s")}
    res["ceiling_beats_floor_all_seeds"] = {
        m: all(ce[m][i] > fl[m][i] for i in range(len(seeds))) if m != "ade_2s"
        else all(ce[m][i] < fl[m][i] for i in range(len(seeds))) for m in ("speed_r2", "yaw_r2", "ade_2s")}
    sp_ok = res["pseudo_beats_floor_all_seeds"]["speed_r2"]
    sp_frac = res["fraction_of_ceiling"]["speed_r2"]
    res["verdict"] = ("GO-decision-grade (pseudo beats floor all seeds on parity AND captures "
                      f"{sp_frac:.0%} of speed ceiling)" if sp_ok and sp_frac and sp_frac >= 0.5
                      else "NO/caveat (pseudo did not clear floor+50% on parity)")
    Path(args.out).write_text(json.dumps(res, indent=2))
    log(f"ARMS speed_r2: floor {ms('floor','speed_r2')} pseudo {ms('pseudo','speed_r2')} ceiling {ms('ceiling','speed_r2')}")
    log(f"ARMS yaw_r2:   floor {ms('floor','yaw_r2')} pseudo {ms('pseudo','yaw_r2')} ceiling {ms('ceiling','yaw_r2')}")
    log(f"FRACTION_OF_CEILING {res['fraction_of_ceiling']}")
    log(f"VERDICT {res['verdict']}")
    log(f"WROTE {args.out}  ({time.time()-t0:.0f}s)")
    log("IDM_PARITY_VALIDATION_DONE")


if __name__ == "__main__":
    main()
