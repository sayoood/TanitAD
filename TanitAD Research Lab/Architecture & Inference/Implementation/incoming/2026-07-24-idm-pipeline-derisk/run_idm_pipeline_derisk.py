"""DE-RISK (NOT scale-up) the YouTube-scale IDM-pretraining path: wire the end-to-end
pseudo-labeling pipeline (v1 frozen encoder -> validated multi-domain IDM head ->
per-frame speed/yaw/accel/traj pseudo-labels) and MEASURE pseudo-label quality on a
HELD-OUT DOMAIN as a YouTube proxy.

Follows `../v1-encoder-char/RESULTS_v1_encoder_char.md` (v1 + multi-domain head = the
cheap IDM substrate). This asks the next question: are the pseudo-labels this pipeline
emits on an UNSEEN domain good enough to pretrain a WM on?

PROXY DOMAINS reachable on pod3 (L2D not present):
  * comma2k19 (rectilinear, different vehicle) = the cross-CLASS proxy (closest to a
    genuinely different YouTube rig). Held out: head trains on {rigA + rigB} (fisheye).
  * PhysicalAI rig-B VAL (episode-disjoint) = cross-rig same-class proxy. Held out:
    head trains on {rigA + comma}.
NO target-domain labels are used to fit the labeler (zero-shot) — the YouTube condition.
We ALSO report Pearson r^2 (= affine-calibrated ceiling) to separate "signal is there but
mis-scaled" (recoverable with the design's weak speed prior / speedometer OCR) from "no
signal". Reuses the v1 CLEAN latent cache (/workspace/tmp/branchb_eval/lat_flagshipv1).
Converged head (epochs=50). pod3, venv python, gpu_lock idm-pipeline-derisk.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
import idm_head as ih                                              # noqa: E402
import run_idm_proof as R                                          # noqa: E402

V1_LAT = "/workspace/tmp/branchb_eval/lat_flagshipv1"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# THE PIPELINE (the reusable de-risk artifact)                                #
# --------------------------------------------------------------------------- #
class PseudoLabeler:
    """v1 frozen encoder + a trained IDM head -> per-frame dynamics pseudo-labels.
    label_clip(frames_u8[,cam]) -> per-center-frame {t, speed, yaw_rate, long_accel,
    traj[H,2] ego-frame}. Encoder is purely visual (cam unused for v1)."""

    def __init__(self, enc, readout, head, device, k=4, stride=1):
        self.enc, self.readout, self.head = enc, readout, head
        self.device, self.k, self.stride = device, k, stride

    @torch.no_grad()
    def _encode(self, frames_u8, batch=48):
        return R.encode_frames(self.enc, self.readout, frames_u8, self.device, batch=batch)

    @torch.no_grad()
    def label_clip(self, frames_u8):
        z = self._encode(frames_u8).float()                       # [T,2048]
        T = z.shape[0]
        centers = ih.valid_centers(T, self.k, ih.DEFAULT_HORIZONS, self.stride)
        if centers.numel() == 0:
            return {"centers": [], "speed": [], "yaw_rate": [], "long_accel": [], "traj": []}
        offs = torch.arange(-self.k, self.k + 1)
        win = z[centers[:, None] + offs[None, :]]                 # [N,2k+1,2048]
        out = self.head(win.to(self.device))
        sc = out["scalars"].cpu(); tj = out["traj"].cpu()
        return {"centers": centers.tolist(),
                "speed": sc[:, 0].tolist(), "yaw_rate": sc[:, 1].tolist(),
                "long_accel": sc[:, 3].tolist(), "traj": tj.tolist()}


def load_latent_windows(tags, k=4, stride=2):
    """Reuse cached v1 latents -> concatenated (Z,S,T) + per-clip list."""
    clips = []
    for tag in tags:
        d = torch.load(Path(V1_LAT) / f"{tag}.pt", weights_only=False)
        zw, sc, tj = ih.build_windows(d["z"].float(), d["poses"].float(),
                                      d["actions"].float(), k=k, stride=stride)
        if zw.shape[0]: clips.append((zw, sc, tj))
    return clips


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


# --------------------------------------------------------------------------- #
# pseudo-label QUALITY vs GT (the de-risk number)                             #
# --------------------------------------------------------------------------- #
def r2(pred, gt):
    p, g = pred.double(), gt.double()
    return float(1 - ((g - p) ** 2).sum() / ((g - g.mean()) ** 2).sum().clamp_min(1e-12))


def pearson_r2(pred, gt):
    p, g = pred.double(), gt.double()
    pc, gc = p - p.mean(), g - g.mean()
    r = (pc * gc).sum() / (pc.norm() * gc.norm()).clamp_min(1e-12)
    return float(r * r)                                           # affine-calibrated ceiling


def calib(pred, gt):
    """OLS gt = a*pred + b (a weak per-domain scale prior would recover this)."""
    p, g = pred.double(), gt.double()
    A = torch.stack([p, torch.ones_like(p)], 1)
    sol = torch.linalg.lstsq(A, g.unsqueeze(1)).solution.squeeze(1)
    return float(sol[0]), float(sol[1])


@torch.no_grad()
def quality(head, clips, device, b=2048):
    Z, S, T = cat(clips)
    ps, pt = [], []
    for i in range(0, Z.shape[0], b):
        o = head(Z[i:i + b].to(device)); ps.append(o["scalars"].cpu()); pt.append(o["traj"].cpu())
    ps = torch.cat(ps); pt = torch.cat(pt)                       # [N,4], [N,H,2]
    res = {"n": int(Z.shape[0]), "channels": {}}
    for j, name in ((0, "speed"), (1, "yaw_rate"), (3, "long_accel")):
        a, bmb = calib(ps[:, j], S[:, j])
        res["channels"][name] = {
            "r2_zeroshot": r2(ps[:, j], S[:, j]),
            "pearson_r2_calibceiling": pearson_r2(ps[:, j], S[:, j]),
            "mae": float((ps[:, j].double() - S[:, j].double()).abs().mean()),
            "bias_meanerr": float((ps[:, j].double() - S[:, j].double()).mean()),
            "calib_slope": a, "calib_intercept": bmb}
    # trajectory: ADE + per-horizon + longitudinal/lateral decomposition (ego x=long, y=lat)
    de = (pt.double() - T.double()).norm(dim=-1)                 # [N,H]
    lon = (pt[..., 0].double() - T[..., 0].double()).abs()      # [N,H]
    lat = (pt[..., 1].double() - T[..., 1].double()).abs()
    res["trajectory"] = {
        "ade_2s": float(de.mean()), "de_per_horizon": [float(x) for x in de.mean(0)],
        "longitudinal_ade": float(lon.mean()), "lateral_ade": float(lat.mean()),
        "traj_x_r2": r2(pt[:, -1, 0], T[:, -1, 0]),             # 2s waypoint long R2
        "traj_y_r2": r2(pt[:, -1, 1], T[:, -1, 1])}             # 2s waypoint lat R2
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagship-ckpt", default="/workspace/tmp/idm/ckpt.pt")
    ap.add_argument("--train-cache", default="/workspace/pai_epcache/physicalai-train-e438721ae894")
    ap.add_argument("--val-cache", default="/workspace/pai_epcache/physicalai-val-f1b378f295ae")
    ap.add_argument("--comma-cache", default="/workspace/data/comma2k19-val-61c46fca8f7f")
    ap.add_argument("--train-rig-table", default="/workspace/tmp/idm/rig_table.json")
    ap.add_argument("--val-order", default="/workspace/tmp/val_clip_order.tsv")
    ap.add_argument("--calib-root", default="/workspace/pai_build")
    ap.add_argument("--work", default="/workspace/tmp/idm_derisk")
    ap.add_argument("--out", default="/workspace/tmp/idm_derisk/results_idm_pipeline_derisk.json")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.work).mkdir(parents=True, exist_ok=True); t0 = time.time()

    # tags in the v1 cache (from the transfer eval): tr_a_*, tr_b_*, cm_*, va_b_*
    TR_A = [f"tr_a_{i:05d}" for i in range(100)]
    TR_B = [f"tr_b_{i:05d}" for i in range(120)]
    CM = [f"cm_{i:05d}" for i in range(80)]
    VA_B = [f"va_b_{i:05d}" for i in range(54)]

    fs_enc, fs_ro, _ = R.load_encoder(args.flagship_ckpt, device)
    res = {"meta": {"experiment": "idm_pipeline_derisk", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "flagship_md5": R.md5_of(args.flagship_ckpt), "epochs": args.epochs, "device": device,
                    "pipeline": "v1 frozen encoder -> multi-domain IDM head -> per-frame pseudo-labels",
                    "proxy_note": "L2D absent on pod3; proxies = comma (cross-CLASS) + rigB-val (cross-rig same-class). "
                                  "Zero-shot (no target labels). pearson_r2 = affine-calibrated ceiling (weak speed prior).",
                    "pre_reg": "held-out speed/traj R2 > ~0.7 -> de-risked; else name the gap"},
           "heldout_domains": {}}

    # ---- leave-one-domain-out pseudo-label quality ---- #
    # (A) HELD-OUT comma: head = rigA + rigB (all labeled fisheye) -> pseudo-label comma
    log("head {rigA+rigB} -> pseudo-label HELD-OUT comma ...")
    head_ab = fit_head(cat(load_latent_windows(TR_A[:60] + TR_B[:60])), device, epochs=args.epochs)
    res["heldout_domains"]["comma_crossCLASS"] = {
        "head_trained_on": "rigA+rigB (fisheye)", "target": "comma (rectilinear)",
        "in_domain_speed_ceiling_ref": 0.592,          # comma-cotrained head, results_camcond_multirig
        **quality(head_ab, load_latent_windows(CM), device)}

    # (B) HELD-OUT rig-B val: head = rigA + comma -> pseudo-label rig-B (disjoint)
    log("head {rigA+comma} -> pseudo-label HELD-OUT rigB-val ...")
    head_ac = fit_head(cat(load_latent_windows(TR_A[:60] + CM[:40])), device, epochs=args.epochs)
    res["heldout_domains"]["rigBval_crossRIG"] = {
        "head_trained_on": "rigA+comma", "target": "rigB-val (episode-disjoint)",
        **quality(head_ac, load_latent_windows(VA_B), device)}

    # in-domain reference (head=rigA -> rigA held-out) for the "how much worse cross-domain" ratio
    log("in-domain reference (head rigA -> rigA held-out) ...")
    head_a = fit_head(cat(load_latent_windows(TR_A[:60])), device, epochs=args.epochs)
    res["in_domain_reference_rigA"] = quality(head_a, load_latent_windows(TR_A[60:100]), device)

    # ---- WIRE-IT proof: emit actual per-frame pseudo-labels for sample held-out clips ---- #
    log("emitting sample pseudo-labels (pipeline end-to-end) ...")
    train_rig = json.loads(Path(args.train_rig_table).read_text())
    _, tb = R.select_episodes(train_rig, args.train_cache, 400, 400)
    comma_paths = sorted(Path(args.comma_cache).glob("ep_*.pt"))
    labeler = PseudoLabeler(fs_enc, fs_ro, head_ab, device, stride=1)     # comma labeler
    samples = {}
    for nm, p in (("comma_sample", str(comma_paths[0])),):
        d = R._load_ep(p); pl = labeler.label_clip(d["frames_u8"])
        # GT at the same centers for a quick sanity line
        gt_speed = [float(d["poses"][c, 3]) for c in pl["centers"]]
        samples[nm] = {"path": p, "n_frames": int(d["frames_u8"].shape[0]),
                       "n_labeled": len(pl["centers"]),
                       "pseudo_speed_head": pl["speed"][:8], "gt_speed_head": gt_speed[:8],
                       "pseudo_traj0": pl["traj"][0] if pl["traj"] else None}
    (Path(args.work) / "sample_pseudolabels.json").write_text(json.dumps(samples, indent=2))
    res["wire_it_proof"] = {"emitted": list(samples), "file": str(Path(args.work) / "sample_pseudolabels.json"),
                            "example": samples[list(samples)[0]]}

    # ---- readiness verdict ---- #
    def verdict(d):
        sp = d["channels"]["speed"]; tr = d["trajectory"]
        return {"speed_r2_zeroshot": sp["r2_zeroshot"], "speed_pearson_r2": sp["pearson_r2_calibceiling"],
                "traj_x_r2": tr["traj_x_r2"], "ade_2s": tr["ade_2s"],
                "meets_0p7_speed_zeroshot": sp["r2_zeroshot"] > 0.7,
                "meets_0p7_speed_calibrated": sp["pearson_r2_calibceiling"] > 0.7}
    res["readiness"] = {k: verdict(v) for k, v in res["heldout_domains"].items()}

    Path(args.out).write_text(json.dumps(res, indent=2))
    for k, v in res["readiness"].items():
        log(f"  {k}: speed R2 zeroshot {v['speed_r2_zeroshot']:+.3f} / calib-ceiling {v['speed_pearson_r2']:.3f} | "
            f"traj_x R2 {v['traj_x_r2']:+.3f} ade {v['ade_2s']:.2f} | >0.7 zeroshot={v['meets_0p7_speed_zeroshot']}")
    log(f"WROTE {args.out}  ({time.time()-t0:.0f}s)")
    log("IDM_PIPELINE_DERISK_DONE")


if __name__ == "__main__":
    main()
