"""Characterize flagship-v1's FROZEN encoder as a cross-rig / IDM substrate — the
pivot evidence for Sayed (own-encoder PIVOT: v1 usable AS-IS vs flagship-warm-started).

Context: the Branch B transfer eval (`RESULTS_branchB.md`) gave v1 frozen a MIXED
cross-rig read: multirig_val (head=rigA+comma) speed R2 +0.657 but rig_val
(head=rigA-only) -1.169. This script resolves whether the -1.17 is a HEAD-DIVERSITY
artifact (fixable cheaply with a multi-domain readout) or an ENCODER ceiling, adds
FULL IDM readout (speed/yaw/accel/steer), a 3rd domain (comma) as a cross target, and
a GEOMETRIC-ROBUSTNESS probe (±dv px vertical pitch-proxy shift).

FACTUAL CORRECTION (verified by code): flagship-v1 (train_flagship4b, WM recipe) trained
WITHOUT geom_augment — that augmentation lives ONLY in train_dynamics_encoder.py (the
Branch B trainer). So for v1, CLEAN frames ARE its training distribution; the aug arm is a
ROBUSTNESS probe, not a distribution match. For Branch B, aug IS its matched condition, so
the BB aug arm doubles as the closure of the RESULTS_branchB in-domain-weakness caveat.

Reuses the cached CLEAN latents from the transfer eval (/workspace/tmp/branchb_eval/) and
only re-encodes the AUG arm. Converged head (epochs=50; flagship in-domain ~0.9 = the C5
validity check). Bootstrap: episode-cluster over eval clips. Run on pod3, venv python,
gpu_lock v1-encoder-char.
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

F_EFF = 266.0
CLEAN_DIR = {"branchB": "/workspace/tmp/branchb_eval/lat_branchB",
             "flagshipv1": "/workspace/tmp/branchb_eval/lat_flagshipv1"}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cam_for(rig, cy):
    if rig == "a":   raw, kn = clip_cam_raw(cy or 542.0, 1.0, pitch=0.02, height=1.30)
    elif rig == "b": raw, kn = clip_cam_raw(cy or 753.0, 1.0, pitch=-0.03, height=1.60)
    else:            raw, kn = clip_cam_raw(128.0, 0.0, pitch=0.01, height=1.20)
    return raw, kn


def aug_shift(frames_u8, cam_raw, dv):
    """Vertical roll by dv px (rig-pitch proxy) + consistent cam update (geom_augment)."""
    f = torch.roll(frames_u8, shifts=dv, dims=2)
    if dv > 0: f[:, :, :dv] = 0
    elif dv < 0: f[:, :, dv:] = 0
    c = cam_raw.clone(); c[2] = c[2] + dv
    c[3] = c[3] + math.atan2(float(dv), F_EFF)
    return f, c


DV_CYCLE = [-12, -8, -5, 5, 8, 12]   # deterministic per-clip nonzero shifts


# ---- encoders ---- #
def load_branchb(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = DynamicsEncoderModel(DynEncConfig(**ck["cfg"])); m.load_state_dict(ck["model"], strict=True)
    enc = m.encoder.to(device).eval(); head = m.idm_head.to(device).eval()
    for p in list(enc.parameters()) + list(head.parameters()): p.requires_grad_(False)
    return enc, head, int(ck.get("step", -1))


@torch.no_grad()
def enc_bb(enc, frames, cam16, device, b=48):
    cam = cam16.to(device); out = []
    for i in range(0, frames.shape[0], b):
        fb = frames[i:i + b].to(device).float().div_(255.0)
        out.append(enc(fb, cam.unsqueeze(0).expand(fb.shape[0], -1)).half().cpu())
    return torch.cat(out)


# ---- latents: clean (reuse) or aug (encode) ---- #
def get_latent(which, tag, spec, enc_objs, device, aug):
    """spec=(path,rig,cy). aug=None -> reuse clean cache; else re-encode with per-clip dv."""
    if aug is None:
        return torch.load(Path(CLEAN_DIR[which]) / f"{tag}.pt", weights_only=False)
    adir = Path(f"/workspace/tmp/v1char/lat_{which}_aug"); adir.mkdir(parents=True, exist_ok=True)
    lf = adir / f"{tag}.pt"
    if lf.exists(): return torch.load(lf, weights_only=False)
    path, rig, cy = spec
    d = R._load_ep(path)
    dv = DV_CYCLE[aug % len(DV_CYCLE)]
    raw, kn = cam_for(rig, cy)
    f, raw2 = aug_shift(d["frames_u8"], raw, dv)
    if which == "branchB":
        z = enc_bb(enc_objs[0], f, normalize_cam_params(raw2, kn), device)
    else:
        z = R.encode_frames(enc_objs[0], enc_objs[1], f, device, batch=48)
    out = {"z": z, "poses": d["poses"].float(), "actions": d["actions"].float()}
    torch.save(out, lf); del d, f
    return out


def clip_windows(which, tagspecs, enc_objs, device, aug_map=None, k=4, stride=2):
    """tagspecs: list of (tag, spec). aug_map: None(all clean) or dict tag->augidx."""
    out = []
    for j, (tag, spec) in enumerate(tagspecs):
        aug = None if aug_map is None else aug_map.get(tag)
        d = get_latent(which, tag, spec, enc_objs, device, aug)
        zw, sc, tj = ih.build_windows(d["z"].float(), d["poses"].float(),
                                      d["actions"].float(), k=k, stride=stride)
        if zw.shape[0]: out.append((zw, sc, tj))
    return out


def cat(lst): return (torch.cat([x[0] for x in lst]), torch.cat([x[1] for x in lst]),
                      torch.cat([x[2] for x in lst]))


def fit_head(train_cat, device, epochs=50, batch=256, lr=3e-4, wd=0.01, seed=0):
    torch.manual_seed(seed); Z, S, T = train_cat
    std = ih.Standardizer.fit(S)
    head = ih.IDMHead(state_dim=2048, horizons=ih.DEFAULT_HORIZONS).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = Z.shape[0]; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * max(1, n // batch))
    for _ in range(epochs):
        head.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            ld = ih.idm_loss(head(Z[idx].to(device)), S[idx].to(device), T[idx].to(device), std)
            opt.zero_grad(set_to_none=True); ld["loss"].backward(); opt.step(); sched.step()
    head.eval(); return head


@torch.no_grad()
def speed_by_clip(head, clips, device, b=1024):
    out = []
    for zw, sc, _ in clips:
        pr = [head(zw[i:i + b].to(device))["scalars"][:, 0].cpu() for i in range(0, zw.shape[0], b)]
        out.append((torch.cat(pr), sc[:, 0]))
    return out


def pooled_r2(pairs):
    p = torch.cat([a for a, _ in pairs]).double(); g = torch.cat([b for _, b in pairs]).double()
    return float(1 - ((g - p) ** 2).sum() / ((g - g.mean()) ** 2).sum().clamp_min(1e-12))


def boot_ci(pairs, n=2000, seed=0):
    G = torch.Generator().manual_seed(seed); m = len(pairs); v = []
    for _ in range(n):
        idx = torch.randint(m, (m,), generator=G).tolist(); v.append(pooled_r2([pairs[i] for i in idx]))
    v.sort(); return {"point": pooled_r2(pairs), "ci95": [v[int(.025 * n)], v[int(.975 * n)]], "n_clips": m}


def full(m):  # full IDM readout row
    return {"n": m["n"], "speed_r2": m["r2"]["speed"], "yaw_r2": m["r2"]["yaw_rate"],
            "accel_r2": m["r2"]["long_accel"], "steer_r2": m["r2"]["steer"],
            "speed_mae": m["mae"]["speed"], "yaw_mae": m["mae"]["yaw_rate"],
            "accel_mae": m["mae"]["long_accel"], "ade_2s": m["ade_2s"]}


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
    ap.add_argument("--work", default="/workspace/tmp/v1char")
    ap.add_argument("--out", default="/workspace/tmp/v1char/results_v1_encoder_char.json")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.work).mkdir(parents=True, exist_ok=True); t0 = time.time()

    train_rig = json.loads(Path(args.train_rig_table).read_text())
    ta, tb = R.select_episodes(train_rig, args.train_cache, 400, 400)
    val_rig = R.build_rig_table(args.val_order, args.calib_root, str(Path(args.work) / "val_rig_table.json"))
    va, vb = R.select_episodes(val_rig, args.val_cache, 400, 400)
    comma = sorted(Path(args.comma_cache).glob("ep_*.pt"))
    cyt = lambda tab, t: tab[str(int(t.split("_")[-1]))]["cy"]
    # tag -> (path,rig,cy). Tags MUST match the cached clean latents from the transfer eval.
    TR_A = [(f"tr_a_{i:05d}", (p, "a", cyt(train_rig, t))) for i, (t, p) in enumerate(ta[:100])]
    TR_B = [(f"tr_b_{i:05d}", (p, "b", cyt(train_rig, t))) for i, (t, p) in enumerate(tb[:120])]
    CM = [(f"cm_{i:05d}", (str(p), "c", 128.0)) for i, p in enumerate(comma[:80])]
    VA_B = [(f"va_b_{i:05d}", (p, "b", cyt(val_rig, t))) for i, (t, p) in enumerate(vb)]
    log(f"clips tr_a {len(TR_A)} tr_b {len(TR_B)} comma {len(CM)} va_b {len(VA_B)}")

    bb_enc, bb_head, bb_step = load_branchb(args.branchb_ckpt, device)
    fs_enc, fs_ro, _ = R.load_encoder(args.flagship_ckpt, device)
    E = {"branchB": (bb_enc,), "flagshipv1": (fs_enc, fs_ro)}
    aug_all = lambda tags: {t: i for i, (t, _) in enumerate(tags)}   # per-clip dv by index

    res = {"meta": {"experiment": "v1_encoder_substrate_char",
                    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "branchb_step": bb_step, "branchb_md5": R.md5_of(args.branchb_ckpt),
                    "flagship_md5": R.md5_of(args.flagship_ckpt), "epochs": args.epochs,
                    "dv_cycle": DV_CYCLE, "device": device,
                    "note": "flagship-v1 trained WITHOUT geom_augment (WM recipe); aug arm = robustness probe. "
                            "BranchB trained WITH aug -> its aug arm is the matched-condition caveat closure.",
                    "pass_rule_gate": "cross speed R2>0.9 AND yaw R2>0.9 AND ADE<1.5x in-domain",
                    "pivot_rule": "v1 usable AS-IS iff (best converged multi-domain head) rig-B AND comma "
                                  "cross speed R2>0.5 AND yaw R2>0.5, robust to aug; else warm-start-longer."},
           "flagshipv1": {}, "branchB": {}}

    def run_head(which, train_tags, evals, seed=0):
        """evals: name -> (tags, aug_map or None). Returns {name: full-readout} + cross speed pairs."""
        eo = E[which]
        tr = []
        for tags, amap in train_tags: tr += clip_windows(which, tags, eo, device, amap)
        head = fit_head(cat(tr), device, epochs=args.epochs, seed=seed)
        val = {}; pairs = {}
        for name, (tags, amap) in evals.items():
            cl = clip_windows(which, tags, eo, device, amap)
            Z, S, T = cat(cl); val[name] = full(ih.evaluate(head, Z, S, T, device=device))
            if name.startswith("cross"): pairs[name] = speed_by_clip(head, cl, device)
        return val, pairs

    # ---------- flagship-v1 characterization ---------- #
    log("=== flagship-v1 ===")
    # H1 rigA-only head (clean) -> rig-B, comma cross + in-domain
    v, _ = run_head("flagshipv1", [([t for t in TR_A[:60]], None)],
                    {"indom": (TR_A[60:100], None), "cross_rigB": (VA_B, None), "cross_comma": (CM[:40], None)})
    res["flagshipv1"]["H1_rigAonly_clean"] = v
    # H2 rigA+comma head (clean) -> rig-B
    v, p = run_head("flagshipv1", [(TR_A[:60], None), (CM[:40], None)],
                    {"indom": (TR_A[60:100], None), "cross_rigB": (VA_B, None)})
    res["flagshipv1"]["H2_rigA_comma_clean"] = v
    res["flagshipv1"]["H2_cross_rigB_speed_ci"] = boot_ci(p["cross_rigB"])
    # H3 rigA+rigB head (clean) -> comma  (symmetric: two fisheye rigs -> rectilinear)
    v, _ = run_head("flagshipv1", [(TR_A[:60], None), (TR_B[:40], None)],
                    {"indom": (TR_A[60:100], None), "cross_comma": (CM[40:80], None)})
    res["flagshipv1"]["H3_rigA_rigB_clean_to_comma"] = v
    # H1-AUG robustness: clean rigA head, eval AUG in-domain + AUG rig-B
    v, _ = run_head("flagshipv1", [(TR_A[:60], None)],
                    {"indom_aug": (TR_A[60:100], aug_all(TR_A[60:100])),
                     "cross_rigB_aug": (VA_B, aug_all(VA_B))})
    res["flagshipv1"]["H1_rigAonly_AUGeval"] = v
    for k in res["flagshipv1"]:
        if isinstance(res["flagshipv1"][k], dict) and "cross_rigB" in res["flagshipv1"][k]:
            c = res["flagshipv1"][k]
            log(f"  fs {k}: indom speed {c.get('indom',{}).get('speed_r2','-')}, "
                f"rigB speed {c['cross_rigB']['speed_r2']:+.3f} yaw {c['cross_rigB']['yaw_r2']:+.3f}")

    # ---------- Branch B aug-caveat closure ---------- #
    log("=== branchB (aug caveat closure) ===")
    v, _ = run_head("branchB", [(TR_A[:60], None)],
                    {"indom": (TR_A[60:100], None), "cross_rigB": (VA_B, None)})
    res["branchB"]["H1_clean"] = v
    v, _ = run_head("branchB", [(TR_A[:60], aug_all(TR_A[:60]))],
                    {"indom_aug": (TR_A[60:100], aug_all(TR_A[60:100])),
                     "cross_rigB_aug": (VA_B, aug_all(VA_B))})
    res["branchB"]["H1_AUGmatched"] = v
    # own-head clean vs aug on val rig-B
    own = {}
    for nm, amap in (("clean", None), ("aug", aug_all(VA_B))):
        cl = clip_windows("branchB", VA_B, E["branchB"], device, amap)
        Z, S, T = cat(cl); own[nm] = full(ih.evaluate(bb_head, Z, S, T, device=device))
    res["branchB"]["own_head_val_rigB"] = own
    log(f"  bb clean indom speed {res['branchB']['H1_clean']['indom']['speed_r2']:+.3f} | "
        f"aug indom speed {res['branchB']['H1_AUGmatched']['indom_aug']['speed_r2']:+.3f} | "
        f"own-head clean {own['clean']['speed_r2']:+.3f} aug {own['aug']['speed_r2']:+.3f}")

    Path(args.out).write_text(json.dumps(res, indent=2))
    log(f"WROTE {args.out}  ({time.time()-t0:.0f}s)")
    log("V1_ENCODER_CHAR_DONE")


if __name__ == "__main__":
    main()
