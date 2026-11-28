"""P4 — the DECISION READ: does pretraining the (small) world-model on
YouTube-pilot PSEUDO-labeled video lift downstream driving vs NO-YouTube pretrain,
measured on the ACTUAL PARITY held-out split?

Mirrors run_idm_parity_validation.py EXACTLY, swapping only the pretrain corpus D:
  parity-held-out   ->   YouTube-pilot pseudo-labeled latents (yt_*.pt).

  substrate : v1 frozen encoder.   model: IDMHead temporal trunk (small dyn. WM).
  labeler L : v1 + IDMHead{parity rigA[:60]+rigB[:60]+comma[:40]}  (identical recipe)
  downstream: physicalai-VAL (episode-disjoint, BOTH rigs), low-data finetune 15 /
              test 65 — the SAME split parity-validation uses (parity firewall).
  arms (paired per seed, identical finetune):
     FLOOR     random-init                 -> finetune(real) -> test   (no-YouTube)
     PSEUDO_YT pretrain(YouTube pilot pseudo)-> finetune(real) -> test (the read)
     [ref, if parity latents cached] PSEUDO_PARITY / CEILING_PARITY  (same-domain)

⚠ PARITY FIREWALL: reads parity-domain VAL episodes as data with its OWN IDM split;
creates NO WM parity arm and does NOT re-select the canonical WM episode selection.

PRE-REGISTERED (both outcomes committed before running — see NOTE.md):
  metric = downstream parity-val test speed_r2 (primary) + traj ade + yaw (caveat).
  WIN   : PSEUDO_YT beats FLOOR on speed_r2 for ALL seeds AND the clip-cluster
          bootstrap 95% CI of the (PSEUDO_YT - FLOOR) speed_r2 gap excludes 0
          -> the YouTube domain transfers -> the full harvest is justified.
  BOUND : no separation -> name the gap (domain too far / yield too low / label
          noise) -> the full harvest is NOT yet justified.
  A 2-3 seed pilot is a DIRECTIONAL read, explicitly not decision-grade for the
  full commitment.
"""
from __future__ import annotations
import argparse, json, statistics as st, sys, time
from pathlib import Path
import torch

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
import idm_head as ih                                               # noqa: E402
import run_idm_proof as R                                           # noqa: E402

CACHED_DIRS = ["/workspace/tmp/branchb_eval/lat_flagshipv1",
               "/workspace/tmp/idm_parity/lat"]
YT_LAT = "/workspace/tmp/yt_pilot/latents"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get_latent(tag, path, enc, ro, device):
    for d in CACHED_DIRS:
        f = Path(d) / f"{tag}.pt"
        if f.exists():
            return torch.load(f, weights_only=False)
    ep = R._load_ep(path)
    z = R.encode_frames(enc, ro, ep["frames_u8"], device, batch=48)
    out = {"z": z, "poses": ep["poses"].float(), "actions": ep["actions"].float()}
    Path(CACHED_DIRS[1]).mkdir(parents=True, exist_ok=True)
    torch.save(out, Path(CACHED_DIRS[1]) / f"{tag}.pt")
    return out


def windows(tagpaths, enc, ro, device, k=4, stride=2):
    out = []
    for tag, path in tagpaths:
        d = get_latent(tag, path, enc, ro, device)
        zw, sc, tj = ih.build_windows(d["z"].float(), d["poses"].float(),
                                      d["actions"].float(), k=k, stride=stride)
        if zw.shape[0]:
            out.append((zw, sc, tj))
    return out


def yt_windows(latents_dir, k=4, stride=2):
    out = []
    for p in sorted(Path(latents_dir).glob("yt_*.pt")):
        d = torch.load(p, weights_only=False)
        z = d["z"].float()
        zw, sc, tj = ih.build_windows(z, torch.zeros(z.shape[0], 4),
                                      torch.zeros(z.shape[0], 2), k=k, stride=stride)
        if zw.shape[0]:
            out.append((zw, sc, tj))
    return out


def cat(lst):
    return (torch.cat([x[0] for x in lst]), torch.cat([x[1] for x in lst]),
            torch.cat([x[2] for x in lst]))


def train_head(train, device, *, epochs, batch=256, lr=3e-4, wd=0.01, seed=0,
               init_state=None):
    torch.manual_seed(seed)
    Z, S, T = train
    std = ih.Standardizer.fit(S)
    head = ih.IDMHead(state_dim=2048, horizons=ih.DEFAULT_HORIZONS).to(device)
    if init_state is not None:
        head.load_state_dict(init_state)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = Z.shape[0]; Z, S, T = Z.to(device), S.to(device), T.to(device)
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
            o = labeler(zw[i:i + b].to(device))
            ps.append(o["scalars"].cpu()); pt.append(o["traj"].cpu())
        out.append((zw, torch.cat(ps), torch.cat(pt)))
    return out


@torch.no_grad()
def eval_full(head, clips, device):
    m = ih.evaluate(head, *cat(clips), device=device)
    return {"speed_r2": m["r2"]["speed"], "yaw_r2": m["r2"]["yaw_rate"],
            "ade_2s": m["ade_2s"], "n": m["n"]}


@torch.no_grad()
def speed_preds_by_clip(head, clips, device, b=2048):
    """Return pooled (pred_speed, gt_speed, clip_idx) over test clips for the
    clip-cluster bootstrap."""
    preds, gts, cidx = [], [], []
    for ci, (zw, sc, tj) in enumerate(clips):
        for i in range(0, zw.shape[0], b):
            o = head(zw[i:i + b].to(device))
            preds.append(o["scalars"][:, 0].cpu())
            gts.append(sc[i:i + b, 0])
            cidx.append(torch.full((min(b, zw.shape[0] - i),), ci, dtype=torch.long))
    return torch.cat(preds), torch.cat(gts), torch.cat(cidx)


def r2(pred, gt):
    return ih.r2_score(pred, gt)


def bootstrap_gap_ci(pf, pp, gt, cidx, n_clips, n_boot=2000, seed=0):
    """Clip-cluster bootstrap of the (pseudo - floor) speed_r2 gap: resample test
    CLIPS with replacement, recompute pooled speed_r2 for each arm on the same
    resample, take the delta. Returns (mean, lo2.5, hi97.5, frac_gap_gt0)."""
    g = torch.Generator().manual_seed(seed)
    # precompute per-clip masks
    masks = [(cidx == c) for c in range(n_clips)]
    gaps = []
    for _ in range(n_boot):
        pick = torch.randint(0, n_clips, (n_clips,), generator=g)
        sel = torch.cat([masks[int(c)].nonzero(as_tuple=True)[0] for c in pick])
        gaps.append(r2(pp[sel], gt[sel]) - r2(pf[sel], gt[sel]))
    gaps = torch.tensor(gaps)
    return (float(gaps.mean()), float(gaps.quantile(0.025)),
            float(gaps.quantile(0.975)), float((gaps > 0).float().mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagship-ckpt", default="/workspace/tmp/idm/ckpt.pt")
    ap.add_argument("--train-cache", default="/workspace/pai_epcache/physicalai-train-e438721ae894")
    ap.add_argument("--val-cache", default="/workspace/pai_epcache/physicalai-val-f1b378f295ae")
    ap.add_argument("--comma-cache", default="/workspace/data/comma2k19-val-61c46fca8f7f")
    ap.add_argument("--train-rig-table", default="/workspace/tmp/idm/rig_table.json")
    ap.add_argument("--val-order", default="/workspace/tmp/val_clip_order.tsv")
    ap.add_argument("--calib-root", default="/workspace/pai_build")
    ap.add_argument("--yt-latents", default=YT_LAT)
    ap.add_argument("--work", default="/workspace/tmp/yt_pilot/results")
    ap.add_argument("--out", default="/workspace/tmp/yt_pilot/results/results_youtube_pilot_downstream.json")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--pt-epochs", type=int, default=25)
    ap.add_argument("--ft-epochs", type=int, default=60)
    ap.add_argument("--with-parity-ref", action="store_true",
                    help="also compute same-domain PSEUDO_PARITY/CEILING refs if latents cached")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.work).mkdir(parents=True, exist_ok=True); t0 = time.time()
    enc, ro, _ = R.load_encoder(args.flagship_ckpt, device)

    # ---- labeler L (identical recipe to parity/proxy) ----
    train_rig = json.loads(Path(args.train_rig_table).read_text())
    ta, tb = R.select_episodes(train_rig, args.train_cache, 400, 400)
    A = lambda i, p: (f"tr_a_{i:05d}", p); B = lambda i, p: (f"tr_b_{i:05d}", p)
    rigA = [A(i, p) for i, (t, p) in enumerate(ta[:200])]
    rigB = [B(i, p) for i, (t, p) in enumerate(tb[:220])]
    comma = sorted(Path(args.comma_cache).glob("ep_*.pt"))
    CM = [(f"cm_{i:05d}", str(p)) for i, p in enumerate(comma[:80])]
    log("building labeler L {rigA[:60]+rigB[:60]+comma[:40]} ...")
    lab = train_head(cat(windows(rigA[:60] + rigB[:60] + CM[:40], enc, ro, device)),
                     device, epochs=50, seed=0)

    # ---- pretrain corpus D = YouTube pilot pseudo ----
    D_yt = yt_windows(args.yt_latents)
    if not D_yt:
        raise RuntimeError(f"no YouTube latents in {args.yt_latents} — run harvest+pseudo_label first")
    D_yt_pseudo = pseudo_targets(lab, D_yt, device)
    n_yt_win = int(cat(D_yt)[0].shape[0])
    log(f"D_youtube = {len(D_yt)} clips, {n_yt_win} windows (pseudo-labeled by L)")

    # ---- optional same-domain reference corpus ----
    D_par = D_par_pseudo = None
    if args.with_parity_ref:
        cached = all((Path(CACHED_DIRS[0]) / f"tr_a_{i:05d}.pt").exists() for i in (60, 199)) and \
                 all((Path(CACHED_DIRS[0]) / f"tr_b_{i:05d}.pt").exists() for i in (60, 219))
        if cached:
            D_par = windows(rigA[60:200] + rigB[60:220], enc, ro, device)
            D_par_pseudo = pseudo_targets(lab, D_par, device)
            log(f"D_parity(ref) = {len(D_par)} clips (cached)")
        else:
            log("parity ref latents not fully cached -> skipping same-domain ref arms")

    # ---- downstream split: parity VAL, finetune 15 / test 65 (SAME as parity-val) ----
    val_rig = R.build_rig_table(args.val_order, args.calib_root,
                                str(Path(args.work) / "val_rig_table.json"))
    va, vb = R.select_episodes(val_rig, args.val_cache, 400, 400)
    VA = [(f"va_a_{i:05d}", p) for i, (t, p) in enumerate(va)]
    VB = [(f"va_b_{i:05d}", p) for i, (t, p) in enumerate(vb)]
    valA = windows(VA, enc, ro, device); valB = windows(VB, enc, ro, device)
    ft = valA[:8] + valB[:7]; te = valA[8:] + valB[7:]
    ft_cat = cat(ft)
    log(f"downstream: finetune {len(ft)} clips ({ft_cat[0].shape[0]} win), "
        f"test {len(te)} clips ({cat(te)[0].shape[0]} win)")

    seeds = list(range(args.seeds))
    arms = ["floor", "pseudo_yt"] + (["pseudo_parity", "ceiling_parity"] if D_par else [])
    agg = {a: {"speed_r2": [], "yaw_r2": [], "ade_2s": []} for a in arms}
    per_seed = []
    ci_records = []
    D_yt_ps_cat = cat(D_yt_pseudo)
    D_par_ps_cat = cat(D_par_pseudo) if D_par_pseudo else None
    D_par_cat = cat(D_par) if D_par else None

    for s in seeds:
        h_floor = train_head(ft_cat, device, epochs=args.ft_epochs, seed=s)
        h_yt_pt = train_head(D_yt_ps_cat, device, epochs=args.pt_epochs, seed=s)
        h_yt = train_head(ft_cat, device, epochs=args.ft_epochs, seed=s,
                          init_state={k: v.clone() for k, v in h_yt_pt.state_dict().items()})
        heads = {"floor": h_floor, "pseudo_yt": h_yt}
        if D_par:
            h_pp = train_head(D_par_ps_cat, device, epochs=args.pt_epochs, seed=s)
            heads["pseudo_parity"] = train_head(ft_cat, device, epochs=args.ft_epochs, seed=s,
                                                init_state={k: v.clone() for k, v in h_pp.state_dict().items()})
            h_re = train_head(D_par_cat, device, epochs=args.pt_epochs, seed=s)
            heads["ceiling_parity"] = train_head(ft_cat, device, epochs=args.ft_epochs, seed=s,
                                                 init_state={k: v.clone() for k, v in h_re.state_dict().items()})
        e = {a: eval_full(h, te, device) for a, h in heads.items()}
        for a in arms:
            for m in agg[a]:
                agg[a][m].append(e[a][m])
        # clip-cluster bootstrap CI of the (pseudo_yt - floor) speed_r2 gap, this seed
        pf, gt, ci = speed_preds_by_clip(h_floor, te, device)
        pp, _gt2, _ci2 = speed_preds_by_clip(h_yt, te, device)
        gmean, glo, ghi, fg = bootstrap_gap_ci(pf, pp, gt, ci, len(te), seed=s)
        ci_records.append({"seed": s, "gap_mean": round(gmean, 4),
                           "gap_ci95": [round(glo, 4), round(ghi, 4)],
                           "frac_boot_gt0": round(fg, 3),
                           "ci_excludes_0": bool(glo > 0)})
        per_seed.append({"seed": s, **{a: e[a] for a in arms}})
        log(f"  seed {s}: floor sp {e['floor']['speed_r2']:+.3f} | "
            f"pseudo_yt sp {e['pseudo_yt']['speed_r2']:+.3f} | "
            f"gap {gmean:+.3f} CI95[{glo:+.3f},{ghi:+.3f}]")

    def ms(a, m): return [round(st.mean(agg[a][m]), 4),
                          round(st.pstdev(agg[a][m]), 4)]
    beats_all = all(agg["pseudo_yt"]["speed_r2"][i] > agg["floor"]["speed_r2"][i]
                    for i in range(len(seeds)))
    ci_all = all(r["ci_excludes_0"] for r in ci_records)
    fl_sp, yt_sp = st.mean(agg["floor"]["speed_r2"]), st.mean(agg["pseudo_yt"]["speed_r2"])
    frac_ceiling = None
    if D_par:
        ce_sp = st.mean(agg["ceiling_parity"]["speed_r2"])
        if abs(ce_sp - fl_sp) > 1e-6:
            frac_ceiling = round((yt_sp - fl_sp) / (ce_sp - fl_sp), 3)

    win = bool(beats_all and ci_all)
    res = {
        "meta": {"experiment": "youtube_idm_pilot_downstream",
                 "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "flagship_md5": R.md5_of(args.flagship_ckpt), "device": device,
                 "seeds": args.seeds, "pt_epochs": args.pt_epochs, "ft_epochs": args.ft_epochs,
                 "pretrain_youtube_clips": len(D_yt), "pretrain_youtube_windows": n_yt_win,
                 "downstream": "physicalai-val both rigs, finetune 15 / test 65 (parity firewall)",
                 "parity_firewall": "own IDM split of parity-domain VAL; no WM arm; no re-selection of canonical WM episodes",
                 "pre_reg": "WIN = pseudo_yt beats floor all seeds AND clip-bootstrap 95% CI of speed_r2 gap excludes 0; else BOUND",
                 "caveat": "2-3 seed pilot = DIRECTIONAL read, not decision-grade for the full harvest"},
        "arms_mean_std": {a: {m: ms(a, m) for m in ("speed_r2", "yaw_r2", "ade_2s")} for a in arms},
        "per_seed": per_seed,
        "bootstrap_ci_speed_r2_gap": ci_records,
        "pseudo_yt_beats_floor_all_seeds": beats_all,
        "ci_excludes_0_all_seeds": ci_all,
        "fraction_of_ceiling_speed_r2": frac_ceiling,
        "verdict": ("WIN — YouTube-pilot pseudo-pretraining lifts downstream speed_r2 "
                    "(all seeds + CI-separated); domain transfers -> full harvest justified (directional)"
                    if win else
                    "BOUND — no CI-separated lift from YouTube-pilot pretraining on parity-val "
                    "speed_r2; full harvest NOT yet justified (name the gap: domain shift / yield / label noise)"),
    }
    Path(args.out).write_text(json.dumps(res, indent=2))
    log(f"ARMS speed_r2: {{a: mean_std}} -> " +
        json.dumps({a: ms(a, 'speed_r2') for a in arms}))
    log(f"BOOTSTRAP {ci_records}")
    log(f"FRACTION_OF_CEILING(speed_r2) {frac_ceiling}")
    log(f"VERDICT {res['verdict']}")
    log(f"WROTE {args.out} ({time.time()-t0:.0f}s)")
    log("YT_PILOT_DOWNSTREAM_DONE")


if __name__ == "__main__":
    main()
