"""DECISIVE held-out-rig transfer eval for Branch B (own dynamics encoder).

Design: `…/incoming/2026-07-22-own-dynamics-encoder/{DESIGN,PRE_REGISTRATION}.md`
+ `dynenc-branchB/BRANCHB_LAUNCH.md` ("held-out-rig eval reuses the camcond
harness with THIS encoder in place of the warm-started one").

Question (pre-registered, both outcomes committed):
  Does Branch B — the from-scratch, GAIA-2 ALL-BLOCK camera-conditioned, multi-rig
  video-SSL encoder (step 40k) — recover cross-rig ego-motion recovery where the
  warm-start suffix-conditioning ABLATION did NOT (cross-rig speed R2 ~ -2.06/-2.25)
  and where multi-domain co-training alone did NOT (-1.61 / -1.65)?

Protocol (faithful to the camcond harness downstream probe — same IDMHead, same
build_windows/evaluate, same rigA(+comma)->rigB splits):
  * FROZEN encoder produces latents z[T,2048]; a FRESH IDMHead is fit on
    rigA(+comma) TRAIN windows ONLY (never rig-B); eval on held-out rig-B.
  * TWO encoders on IDENTICAL windows for a clean paired contrast:
      branchB      : CameraConditionedEncoder (fed TRUE per-clip cam params)
      flagshipv1   : plain ViT+readout (the -2.4 frozen control, no conditioning)
  * TWO eval regimes for the cross-rig set:
      _train_ : rig-B clips = the SAME train-cache clips the ablation used
                (paired-to-baseline; BUT Branch B saw these in SSL+supervised
                training -> best-case, flagged).
      _val_   : rig-B clips from the physicalai-VAL cache (f1b378f295ae), which
                Branch B NEVER trained on (episode-disjoint) -> the leak-controlled
                honest cross-rig number. Same rig GEOMETRY (multi-rig coverage is
                the GAIA-2 design), episode-disjoint.
  * CONTEXT (not a transfer number): Branch B's OWN trained idm_head on rig-B,
    labelled INSAMPLE (head trained on rig-B -> a ceiling, not transfer).

Gate (frozen): cross speed R2>0.9 AND yaw R2>0.9 AND ADE@2s<1.5x in-domain.
Honest bounds: episode-cluster bootstrap 95% CI on cross-rig speed R2 (over the
rig-B eval CLIPS) + paired branchB-minus-flagship dR2 CI on identical clips.
Run on pod3 (A40), venv python, under gpu_lock branchb-transfer.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import torch

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
import idm_head as ih                                              # noqa: E402
import run_idm_proof as R                                          # noqa: E402
from tanitad.models.dynamics_encoder import (                     # noqa: E402
    DynEncConfig, DynamicsEncoderModel, normalize_cam_params)
from train_dynamics_encoder import clip_cam_raw                   # noqa: E402


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# encoders                                                                    #
# --------------------------------------------------------------------------- #
def load_branchb(ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = DynEncConfig(**ck["cfg"])
    model = DynamicsEncoderModel(cfg)
    miss, unexp = model.load_state_dict(ck["model"], strict=True)
    enc = model.encoder.to(device).eval()
    head = model.idm_head.to(device).eval()
    for p in list(enc.parameters()) + list(head.parameters()):
        p.requires_grad_(False)
    log(f"branchB: state_dim {enc.state_dim}, step {ck.get('step')}, "
        f"missing {len(miss)} unexpected {len(unexp)}")
    return enc, head, enc.state_dim, int(ck.get("step", -1))


@torch.no_grad()
def encode_branchb(enc, frames_u8, cam16, device, batch=48):
    cam = cam16.to(device)
    zs = []
    for i in range(0, frames_u8.shape[0], batch):
        fb = frames_u8[i:i + batch].to(device).float().div_(255.0)
        camb = cam.unsqueeze(0).expand(fb.shape[0], -1)
        zs.append(enc(fb, camb).half().cpu())
    return torch.cat(zs)


@torch.no_grad()
def encode_flagship(enc, readout, frames_u8, device, batch=48):
    return R.encode_frames(enc, readout, frames_u8, device, batch=batch)


# --------------------------------------------------------------------------- #
# latents + windows                                                           #
# --------------------------------------------------------------------------- #
def cam_for(rig: str, cy: float):
    """EXACT train_dynamics_encoder.build_clip_specs convention Branch B trained on."""
    if rig == "a":
        raw, kn = clip_cam_raw(cy or 542.0, 1.0, pitch=0.02, height=1.30)
    elif rig == "b":
        raw, kn = clip_cam_raw(cy or 753.0, 1.0, pitch=-0.03, height=1.60)
    else:  # comma rectilinear
        raw, kn = clip_cam_raw(128.0, 0.0, pitch=0.01, height=1.20)
    return normalize_cam_params(raw, kn)


def encode_clips(which, clips, latent_dir, device, enc_objs):
    """clips: list of (tag, path, rig, cy). Encode once per encoder, cache to disk."""
    Path(latent_dir).mkdir(parents=True, exist_ok=True)
    for j, (tag, path, rig, cy) in enumerate(clips):
        lf = Path(latent_dir) / f"{tag}.pt"
        if lf.exists():
            continue
        d = R._load_ep(path)
        if which == "branchB":
            z = encode_branchb(enc_objs[0], d["frames_u8"], cam_for(rig, cy), device)
        else:
            z = encode_flagship(enc_objs[0], enc_objs[1], d["frames_u8"], device)
        torch.save({"z": z, "poses": d["poses"].float(),
                    "actions": d["actions"].float()}, lf)
        del d
        if j % 20 == 0:
            log(f"  [{which}] encoded {j}/{len(clips)} {tag} z{tuple(z.shape)}")


def clip_windows(latent_dir, tags, k=4, stride=2):
    """-> list of per-clip (Zwin,S,T); drops empties."""
    out = []
    for tag in tags:
        d = torch.load(Path(latent_dir) / f"{tag}.pt", weights_only=False)
        zw, sc, tj = ih.build_windows(d["z"].float(), d["poses"].float(),
                                      d["actions"].float(), k=k, stride=stride)
        if zw.shape[0]:
            out.append((zw, sc, tj))
    return out


def cat_clips(lst):
    return (torch.cat([x[0] for x in lst]), torch.cat([x[1] for x in lst]),
            torch.cat([x[2] for x in lst]))


# --------------------------------------------------------------------------- #
# head fit (identical to ih.train_head fitting) + per-clip speed for bootstrap #
# --------------------------------------------------------------------------- #
def fit_head(train_cat, state_dim, device, epochs=10, batch=256, lr=3e-4,
             wd=0.01, seed=0):
    torch.manual_seed(seed)
    Ztr, Str, Ttr = train_cat
    std = ih.Standardizer.fit(Str)
    head = ih.IDMHead(state_dim=state_dim, horizons=ih.DEFAULT_HORIZONS).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = Ztr.shape[0]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * max(1, n // batch))
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            ld = ih.idm_loss(head(Ztr[idx].to(device)), Str[idx].to(device),
                             Ttr[idx].to(device), std)
            opt.zero_grad(set_to_none=True)
            ld["loss"].backward()
            opt.step()
            sched.step()
    head.eval()
    return head


@torch.no_grad()
def speed_by_clip(head, clips, device, batch=1024):
    """-> list of (pred_speed[ni], gt_speed[ni]) numpy-free tensors, per clip."""
    out = []
    for zw, sc, _tj in clips:
        preds = []
        for i in range(0, zw.shape[0], batch):
            preds.append(head(zw[i:i + batch].to(device))["scalars"][:, 0].cpu())
        out.append((torch.cat(preds) if preds else sc[:, 0:0], sc[:, 0]))
    return out


def pooled_r2(pairs):
    p = torch.cat([a for a, _ in pairs]).double()
    g = torch.cat([b for _, b in pairs]).double()
    ss_res = ((g - p) ** 2).sum()
    ss_tot = ((g - g.mean()) ** 2).sum().clamp_min(1e-12)
    return float(1 - ss_res / ss_tot)


def bootstrap_r2(pairs, n_boot=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = len(pairs)
    vals = []
    for _ in range(n_boot):
        idx = torch.randint(m, (m,), generator=g).tolist()
        vals.append(pooled_r2([pairs[i] for i in idx]))
    vals = sorted(vals)
    lo, hi = vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]
    return {"point": pooled_r2(pairs), "ci95": [lo, hi], "n_clips": m}


def paired_bootstrap_dr2(pairs_bb, pairs_fs, n_boot=2000, seed=0):
    """dR2 = R2(branchB) - R2(flagship) on the SAME resampled clips (paired)."""
    assert len(pairs_bb) == len(pairs_fs)
    g = torch.Generator().manual_seed(seed)
    m = len(pairs_bb)
    vals = []
    for _ in range(n_boot):
        idx = torch.randint(m, (m,), generator=g).tolist()
        vals.append(pooled_r2([pairs_bb[i] for i in idx])
                    - pooled_r2([pairs_fs[i] for i in idx]))
    vals = sorted(vals)
    lo, hi = vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]
    point = pooled_r2(pairs_bb) - pooled_r2(pairs_fs)
    frac_pos = sum(v > 0 for v in vals) / len(vals)
    return {"point": point, "ci95": [lo, hi], "frac_boot_positive": frac_pos}


def verdict(cross_metrics, indom_ade):
    r2 = cross_metrics["r2"]
    return {"cross_speed_r2": r2["speed"], "cross_yaw_r2": r2["yaw_rate"],
            "cross_steer_r2": r2["steer"], "cross_ade_2s": cross_metrics["ade_2s"],
            "in_domain_ade_2s": indom_ade,
            "ade_ratio": cross_metrics["ade_2s"] / max(indom_ade, 1e-9),
            "PASS": bool(r2["speed"] > 0.9 and r2["yaw_rate"] > 0.9
                         and cross_metrics["ade_2s"] < 1.5 * indom_ade)}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branchb-ckpt", default="/workspace/experiments/dynenc-branchB/ckpt.pt")
    ap.add_argument("--flagship-ckpt", default="/workspace/tmp/idm/ckpt.pt")
    ap.add_argument("--train-cache", default="/workspace/pai_epcache/physicalai-train-e438721ae894")
    ap.add_argument("--val-cache", default="/workspace/pai_epcache/physicalai-val-f1b378f295ae")
    ap.add_argument("--comma-cache", default="/workspace/data/comma2k19-val-61c46fca8f7f")
    ap.add_argument("--train-rig-table", default="/workspace/tmp/idm/rig_table.json")
    ap.add_argument("--val-order", default="/workspace/tmp/val_clip_order.tsv")
    ap.add_argument("--calib-root", default="/workspace/pai_build")
    ap.add_argument("--work", default="/workspace/tmp/branchb_eval")
    ap.add_argument("--out", default="/workspace/tmp/branchb_eval/results_branchb_transfer.json")
    ap.add_argument("--n-cross-train", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.work).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- rig tables + clip selection ----
    train_rig = json.loads(Path(args.train_rig_table).read_text())
    ta, tb = R.select_episodes(train_rig, args.train_cache, 400, 400)
    val_rig = R.build_rig_table(args.val_order, args.calib_root,
                                str(Path(args.work) / "val_rig_table.json"))
    va, vb = R.select_episodes(val_rig, args.val_cache, 400, 400)
    comma = sorted(Path(args.comma_cache).glob("ep_*.pt"))

    def cy_of(table, tag):
        idx = int(tag.split("_")[-1]); return table[str(idx)]["cy"]

    # clip lists: (tag, path, rig, cy)  -- tags are namespaced per split
    train_a = [(f"tr_a_{i:05d}", p, "a", cy_of(train_rig, t)) for i, (t, p) in enumerate(ta[:100])]
    train_b = [(f"tr_b_{i:05d}", p, "b", cy_of(train_rig, t)) for i, (t, p) in enumerate(tb[:args.n_cross_train])]
    comma_c = [(f"cm_{i:05d}", str(p), "c", 128.0) for i, p in enumerate(comma[:80])]
    val_a = [(f"va_a_{i:05d}", p, "a", cy_of(val_rig, t)) for i, (t, p) in enumerate(va)]
    val_b = [(f"va_b_{i:05d}", p, "b", cy_of(val_rig, t)) for i, (t, p) in enumerate(vb)]
    all_clips = train_a + train_b + comma_c + val_a + val_b
    log(f"clips: train_a {len(train_a)} train_b {len(train_b)} comma {len(comma_c)} "
        f"val_a {len(val_a)} val_b {len(val_b)}")

    # ---- encode with both encoders ----
    from tanitad.config import flagship4b_config  # noqa
    bb_enc, bb_head, S_bb, bb_step = load_branchb(args.branchb_ckpt, device)
    assert S_bb == 2048, f"unexpected branchB state_dim {S_bb}"
    fs_enc, fs_ro, fs_meta = R.load_encoder(args.flagship_ckpt, device)

    dirs = {"branchB": str(Path(args.work) / "lat_branchB"),
            "flagshipv1": str(Path(args.work) / "lat_flagshipv1")}
    log("encoding (branchB)...")
    encode_clips("branchB", all_clips, dirs["branchB"], device, (bb_enc, bb_head))
    log("encoding (flagshipv1)...")
    encode_clips("flagshipv1", all_clips, dirs["flagshipv1"], device, (fs_enc, fs_ro))

    tags = {"train_a": [c[0] for c in train_a], "train_b": [c[0] for c in train_b],
            "comma": [c[0] for c in comma_c], "val_a": [c[0] for c in val_a],
            "val_b": [c[0] for c in val_b]}

    # ---- experiments ----
    md5 = R.md5_of
    results = {"meta": {
        "experiment": "branchb_heldout_rig_transfer",
        "design": "PRE_REGISTRATION.md + BRANCHB_LAUNCH.md",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device, "branchb_ckpt_md5": md5(args.branchb_ckpt),
        "branchb_step": bb_step, "flagship_ckpt_md5": md5(args.flagship_ckpt),
        "head_fit_epochs": args.epochs, "seed": args.seed,
        "pass_rule": "cross speed R2>0.9 AND yaw R2>0.9 AND ADE<1.5x in-domain",
        "n_clips": {k: len(v) for k, v in tags.items()},
        "baselines_external": {
            "ablation_ON_rig_rigB_speed_r2": -2.253,
            "ablation_ON_multirig_rigB_speed_r2": -2.057,
            "single_domain_lightft_rigB_speed_r2": -1.65,
            "multirig_lightft_rigB_speed_r2": -1.61,
            "frozen_flagship_rig_rigB_speed_r2_priorartifact": -2.465},
        "CAVEAT_train_cross": ("Branch B trained (SSL + supervised IDM) on ALL rig-B "
            "train-cache clips -> the _train_ cross set is best-case, not zero-shot. "
            "The _val_ cross set (physicalai-val, episode-disjoint from training) is "
            "the leak-controlled number."),
        "CAVEAT_thesis": ("held-out RIG here = rig-B GEOMETRY that Branch B saw during "
            "multi-rig SSL (GAIA-2 'conditioning + multi-rig, both required'); the head "
            "never saw rig-B. This validates the FULL multi-rig recipe, NOT transfer to "
            "a never-seen rig (that is the deferred YouTube-scale question).")},
        "arms": {}, "headline": {}}

    # experiment specs: name -> (train_tags, {evalname:(tags, is_cross)}, indom_key)
    EXP = {
        "rig_train":      (["train_a_0:60"], {"indom": ("train_a_60:100", False),
                            "cross": ("train_b", True)}, "indom"),
        "multirig_train": (["train_a_0:60", "comma_0:40"], {"indomA": ("train_a_60:100", False),
                            "indomC": ("comma_40:80", False), "cross": ("train_b", True)}, "indomA"),
        "rig_val":        (["train_a_0:60"], {"indom": ("val_a", False),
                            "cross": ("val_b", True)}, "indom"),
        "multirig_val":   (["train_a_0:60", "comma_0:40"], {"indomA": ("val_a", False),
                            "cross": ("val_b", True)}, "indomA"),
    }

    def resolve(spec):
        if spec == "train_a_0:60": return tags["train_a"][:60]
        if spec == "train_a_60:100": return tags["train_a"][60:100]
        if spec == "comma_0:40": return tags["comma"][:40]
        if spec == "comma_40:80": return tags["comma"][40:80]
        return tags[spec]

    # cache per-clip windows per (encoder, tagset)
    def wins(which, tagset):
        return clip_windows(dirs[which], resolve(tagset))

    for exp, (tr_specs, evals, indom_key) in EXP.items():
        results["arms"][exp] = {}
        cross_pairs = {}   # which -> per-clip speed pairs (for bootstrap)
        for which in ("branchB", "flagshipv1"):
            tr_clips = []
            for s in tr_specs:
                tr_clips += wins(which, s)
            head = fit_head(cat_clips(tr_clips), 2048, device, epochs=args.epochs,
                            seed=args.seed)
            val = {}
            for ename, (tagset, is_cross) in evals.items():
                ev_clips = wins(which, tagset)
                Zc, Sc, Tc = cat_clips(ev_clips)
                val[ename] = ih.evaluate(head, Zc, Sc, Tc, device=device)
                if is_cross:
                    cross_pairs[which] = speed_by_clip(head, ev_clips, device)
            vd = verdict(val["cross"], val[indom_key]["ade_2s"])
            results["arms"][exp][which] = {"val": val, "verdict": vd}
            log(f"{exp} [{which}] cross speedR2={vd['cross_speed_r2']:.3f} "
                f"yawR2={vd['cross_yaw_r2']:.3f} ade_ratio={vd['ade_ratio']:.2f} "
                f"PASS={vd['PASS']}")
        # bootstrap CIs + paired dR2 on the cross set
        bb_ci = bootstrap_r2(cross_pairs["branchB"], seed=args.seed)
        fs_ci = bootstrap_r2(cross_pairs["flagshipv1"], seed=args.seed)
        dr2 = paired_bootstrap_dr2(cross_pairs["branchB"], cross_pairs["flagshipv1"],
                                   seed=args.seed)
        results["arms"][exp]["cross_speed_r2_ci"] = {
            "branchB": bb_ci, "flagshipv1": fs_ci, "paired_dr2_bb_minus_fs": dr2}
        log(f"{exp} CI branchB speedR2 {bb_ci['point']:.3f} {bb_ci['ci95']} | "
            f"flagship {fs_ci['point']:.3f} {fs_ci['ci95']} | dR2 {dr2['point']:.3f} "
            f"{dr2['ci95']} fracpos {dr2['frac_boot_positive']:.3f}")

    # ---- Branch B OWN head, zero-shot on rig-B (INSAMPLE ceiling, not transfer) ----
    zs = {}
    for cross_name, tagset in (("train_b", "train_b"), ("val_b", "val_b")):
        ev = clip_windows(dirs["branchB"], tags[tagset])
        Zc, Sc, Tc = cat_clips(ev)
        zs[cross_name] = ih.evaluate(bb_head, Zc, Sc, Tc, device=device)
        log(f"branchB OWN-head INSAMPLE {cross_name}: speedR2={zs[cross_name]['r2']['speed']:.3f} "
            f"yawR2={zs[cross_name]['r2']['yaw_rate']:.3f} ade={zs[cross_name]['ade_2s']:.3f}")
    results["branchB_own_head_INSAMPLE"] = zs

    # ---- headline ----
    def hl(exp):
        a = results["arms"][exp]
        return {"branchB_cross_speed_r2": a["branchB"]["verdict"]["cross_speed_r2"],
                "flagshipv1_cross_speed_r2": a["flagshipv1"]["verdict"]["cross_speed_r2"],
                "branchB_cross_yaw_r2": a["branchB"]["verdict"]["cross_yaw_r2"],
                "branchB_ade_ratio": a["branchB"]["verdict"]["ade_ratio"],
                "branchB_PASS": a["branchB"]["verdict"]["PASS"],
                "paired_dr2": a["cross_speed_r2_ci"]["paired_dr2_bb_minus_fs"],
                "branchB_speed_r2_ci95": a["cross_speed_r2_ci"]["branchB"]["ci95"]}
    results["headline"] = {e: hl(e) for e in EXP}

    Path(args.out).write_text(json.dumps(results, indent=2))
    log(f"WROTE {args.out}  ({time.time()-t0:.0f}s)")
    log("BRANCHB_TRANSFER_DONE")


if __name__ == "__main__":
    main()
