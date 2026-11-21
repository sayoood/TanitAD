"""Reconstruct the ego trajectory of ONE YouTube dashcam clip with the IDM, and
gather every GT-free cross-check signal the clip affords.

PIPELINE (each step reuses the shipped module — nothing re-implemented):
  1. GeoCalib per-video intrinsics  ->  geocalib_intrinsics.decode_canonical_geocalib
     (privacy: yt_pilot_common.Anonymizer blurs faces/plates/bodies at FULL RES
      before any downscale; decoder thread_type=NONE, never AUTO -> no CUDA
      deadlock)                                     -> frames [T,3,256,256] u8 @10 Hz
  2. comma2k19.stack_frames(3)                      -> [T-2, 9, 256, 256]
  3. frozen flagship-v1 encoder+readout             -> z [T-2, 2048]
  4. idm_head_v1 (PERSISTED labeler, non-causal 9-frame window)
                                                    -> speed / yaw_rate + 2 s traj
  5. flagship-v1 strategic + tactical policy brains -> route / maneuver for the HUD
  6. cross-check signals with NO model in the loop:
       - HUD ROI strips (burned-in GPS speed / distance / elapsed clock) kept at
         source resolution for the reader in hud_ocr.py
       - optical-flow yaw rate from the FAR FIELD (above the horizon, where
         translation-induced flow vanishes and flow is pure rotation)
       - focus-of-expansion, which independently locates the projection centre

The raw mp4 is DELETED as soon as decoding finishes (privacy design, unchanged).

Usage (pod3):
  PYTHONPATH=/workspace/TanitAD/stack /workspace/venv/bin/python yt_idm_reconstruct.py \
     --mp4 /workspace/tmp/yt_val/work/raw.mp4 --work /workspace/tmp/yt_val \
     --head /workspace/tmp/yt_val/results/idm_head_v1.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/TanitAD/stack")
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
sys.path.insert(0, "/workspace/tmp/yt_val/scripts")

import idm_head as ih                                              # noqa: E402
import run_idm_proof as R                                          # noqa: E402
import yt_pilot_common as YC                                       # noqa: E402
import geocalib_intrinsics as GC                                   # noqa: E402
from tanitad.data.comma2k19 import stack_frames                    # noqa: E402

TARGET_HZ = 10.0
POLICY_WINDOW = 8            # flagship state-window for the policy brains


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# HUD strip pass — SAME temporal grid as decode_canonical_geocalib             #
# --------------------------------------------------------------------------- #
def decode_hud_strips(mp4: str, rois: dict, target_hz: float = TARGET_HZ):
    """Second decode pass replicating decode_canonical_geocalib's resampling
    EXACTLY (pts grid, `next_t += dt`, thread_type=NONE) so strip index i is the
    same instant as canonical frame i. Returns {name: uint8 [T,h,w,3]}.

    These strips are the burned-in telemetry HUD only (rendered digits on an
    opaque bar). They are the clip's independently-derivable GT and never leave
    this machine as imagery: hud_ocr.py reads numbers off them and they are
    deleted with the rest of the transient imagery."""
    import av
    out = {k: [] for k in rois}
    with av.open(str(mp4)) as c:
        st = c.streams.video[0]
        st.thread_type = GC.DECODE_THREAD_TYPE          # NONE — never AUTO
        tb = st.time_base
        try:
            src_fps = float(st.average_rate) if st.average_rate else None
        except Exception:
            src_fps = None
        dt = 1.0 / target_hz
        next_t = 0.0
        stride = max(1, int(round((src_fps or target_hz) / target_hz))) if src_fps else 1
        fi = -1
        for fr in c.decode(st):
            fi += 1
            if fr.pts is not None and tb is not None:
                if float(fr.pts * tb) + 1e-6 < next_t:
                    continue
            else:
                if fi % stride != 0:
                    continue
            rgb = fr.to_ndarray(format="rgb24")
            for k, (x0, y0, x1, y1) in rois.items():
                out[k].append(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
            next_t += dt
    return {k: np.stack(v) for k, v in out.items() if v}


# --------------------------------------------------------------------------- #
# optical flow cross-checks (NO model in the loop)                             #
# --------------------------------------------------------------------------- #
def flow_signals(vid_u8: torch.Tensor, f_eff: float, horizon_row: float,
                 dt: float = 1.0 / TARGET_HZ, hud_mask_rows: int | None = None):
    """Per-frame optical-flow yaw rate + focus-of-expansion from the canonical
    256 px frames.

    FAR FIELD (rows well above the horizon) is at effectively infinite depth, so
    its flow is PURE CAMERA ROTATION: du = f * omega_z * dt (u grows right; a
    LEFT yaw, +omega by the repo's +y-left convention, sweeps the scene right).
    That makes the far-field median du a translation-free yaw-rate estimate that
    shares no machinery with the IDM.

    Returns dict of numpy arrays indexed like vid[1:]."""
    import cv2
    g = vid_u8.float().mean(1).byte().numpy()             # [T,256,256] luma
    T, H, W = g.shape
    hr = int(round(horizon_row))
    far0, far1 = max(2, hr - 100), max(6, hr - 18)        # sky / distant band
    gnd0, gnd1 = min(H - 4, hr + 24), (hud_mask_rows if hud_mask_rows else H)
    yaw, div, ok = [], [], []
    prev = g[0]
    for t in range(1, T):
        cur = g[t]
        fl = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 21, 3, 5,
                                          1.1, 0)
        prev = cur
        du = fl[far0:far1, :, 0]
        dv = fl[far0:far1, :, 1]
        # texture gate: featureless sky gives ~0 flow and would bias to 0
        tex = cv2.Laplacian(cur[far0:far1, :], cv2.CV_32F).__abs__()
        m = tex > 6.0
        if m.sum() < 200:
            yaw.append(np.nan); div.append(np.nan); ok.append(False)
        else:
            yaw.append(float(np.median(du[m])) / (f_eff * dt))
            div.append(float(np.median(dv[m])) / (f_eff * dt))
            ok.append(True)
    return {"flow_yaw_rate": np.array(yaw), "flow_pitch_rate": np.array(div),
            "flow_ok": np.array(ok),
            "band_rows": [far0, far1, gnd0, gnd1]}


# --------------------------------------------------------------------------- #
def build_flagship(ckpt_path: str, device: str):
    """Full flagship-v1 WorldModel (for the strategic/tactical HUD brains).
    action_dim=3 per taniteval registry `flagship-30k` (speed_input=True)."""
    from tanitad.config import flagship4b_config
    from tanitad.models.fourbrain import WorldModel
    cfg = flagship4b_config()
    object.__setattr__(cfg.predictor, "action_dim", 3)
    if cfg.tactical_pred is not None:
        object.__setattr__(cfg.tactical_pred, "action_dim", 3)
    object.__setattr__(cfg.encoder, "grad_checkpoint", False)
    model = WorldModel(cfg)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in list(missing) + list(unexpected)
           if k.split(".")[0] in ("encoder", "readout", "strategic_policy",
                                  "tactical_policy")]
    assert not bad, f"policy/encoder weights did not load cleanly: {bad[:8]}"
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"flagship v1 built: missing={len(missing)} unexpected={len(unexpected)} "
        f"(none in encoder/readout/policies)")
    return model, int(ck.get("step", -1)) if isinstance(ck, dict) else -1


@torch.no_grad()
def policy_decode(model, frames9_u8: torch.Tensor, centers: np.ndarray,
                  device: str, batch: int = 24):
    """Decoded intent at each center: strategic route + tactical maneuver, using
    the causal POLICY_WINDOW state window ENDING at the center (identical call
    chain to taniteval.corpus_overlay.episode_rollouts)."""
    routes, mans = {}, {}
    valid = [int(t) for t in centers if t - POLICY_WINDOW + 1 >= 0]
    for i in range(0, len(valid), batch):
        ch = valid[i:i + batch]
        fw = torch.stack([frames9_u8[t - POLICY_WINDOW + 1:t + 1] for t in ch])
        fw = fw.to(device).float().div_(255.0)
        states = model.encode_window(fw)
        follow = torch.zeros(len(ch), dtype=torch.long, device=device)
        sf = model.strategic_policy(states, follow)
        r = sf["route_logits"].argmax(-1).cpu().tolist()
        tf = model.tactical_policy(states, sf["ctx"])
        m = tf["maneuver_logits"].argmax(-1).cpu().tolist()
        for j, t in enumerate(ch):
            routes[t], mans[t] = int(r[j]), int(m[j])
    return routes, mans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4", required=True)
    ap.add_argument("--work", default="/workspace/tmp/yt_val")
    ap.add_argument("--head", required=True, help="idm_head_v1.pt")
    ap.add_argument("--enc-ckpt", default="/workspace/tmp/idm/ckpt.pt")
    ap.add_argument("--cascades", default="/workspace/tmp/yt_pilot/cascades")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--encode-batch", type=int, default=32)
    ap.add_argument("--hud-rois", default=None,
                    help="JSON {name:[x0,y0,x1,y1]} at SOURCE resolution")
    ap.add_argument("--keep-raw", action="store_true",
                    help="debug only; default DELETES the mp4 after decode")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    work = Path(args.work)
    (work / "results").mkdir(parents=True, exist_ok=True)

    # ---- 1. geometry + privacy + canonical decode ------------------------ #
    anon = YC.Anonymizer(cascade_dir=args.cascades if
                         Path(args.cascades).exists() else None)
    anon.reset()
    est = GC.GeoCalibEstimator()
    intr = est.estimate_from_video(args.mp4, n_frames=GC.DEFAULT_N_FRAMES,
                                   anonymizer=None)
    log(f"GeoCalib: vfov {intr.vfov_deg:.2f} hfov {intr.hfov_deg:.2f} "
        f"conf={intr.confidence} mad={intr.vfov_mad_deg} "
        f"fallback={intr.fallback_used} (n_used {intr.n_frames_used}/"
        f"{intr.n_frames_total})")
    anon.reset()
    vid, meta = GC.decode_canonical_geocalib(args.mp4, anon, estimated=intr,
                                             size=256, target_hz=TARGET_HZ,
                                             max_frames=args.max_frames)
    log(f"decoded {tuple(vid.shape)} f_eff={meta['achieved_f_eff']} "
        f"fully_canonical={meta['fully_canonical']} anon={meta['anon']}")

    # where the canonical square landed in the source frame (does a burned-in
    # HUD survive into the model's input?)
    from tanitad.data.calib import focal_crop_size
    W0, H0 = intr.est_width, intr.est_height
    f_src = intr.focal_px(width=W0, height=H0)
    c = focal_crop_size(f_src, H0, W0, 256)
    crop_box = [(W0 - c) // 2, (H0 - c) // 2, (W0 - c) // 2 + c, (H0 - c) // 2 + c]
    log(f"canonical crop at source res {W0}x{H0}: side {c} box {crop_box}")

    # ---- HUD strips on the identical temporal grid ----------------------- #
    hud = {}
    if args.hud_rois:
        rois = {k: tuple(v) for k, v in json.loads(args.hud_rois).items()}
        hud = decode_hud_strips(args.mp4, rois)
        for k, v in hud.items():
            log(f"HUD strip '{k}': {v.shape}")
            np.save(work / "work" / f"hud_{k}.npy", v)

    if not args.keep_raw:
        os.remove(args.mp4)                      # privacy: raw video destroyed
        log(f"DELETED raw video {args.mp4}")

    # ---- 2/3. stack + frozen encode -------------------------------------- #
    f9 = stack_frames(vid, 3)                                  # [T-2,9,256,256]
    enc, ro, emeta = R.load_encoder(args.enc_ckpt, device)
    z = R.encode_frames(enc, ro, f9, device, batch=args.encode_batch)
    log(f"encoded z {tuple(z.shape)} (state_dim {emeta['state_dim']})")
    del enc, ro
    torch.cuda.empty_cache()

    # ---- 4. IDM reconstruction (persisted labeler) ----------------------- #
    hd = torch.load(args.head, map_location="cpu", weights_only=False)
    head = ih.IDMHead(**hd["config"]["head_kwargs"]).to(device)
    head.load_state_dict(hd["state_dict"])
    head.eval()
    k = hd["config"]["window_k"]
    zf = z.float()
    Tz = zf.shape[0]
    centers = ih.valid_centers(Tz, k, ih.DEFAULT_HORIZONS, stride=1).numpy()
    offs = torch.arange(-k, k + 1)
    Zwin = zf[torch.as_tensor(centers)[:, None] + offs[None, :]]
    preds_s, preds_t = [], []
    with torch.no_grad():
        for i in range(0, Zwin.shape[0], 512):
            o = head(Zwin[i:i + 512].to(device))
            preds_s.append(o["scalars"].cpu())
            preds_t.append(o["traj"].cpu())
    scal = torch.cat(preds_s).numpy()                # [N,4]
    traj = torch.cat(preds_t).numpy()                # [N,H,2]
    log(f"IDM: {len(centers)} windows; speed mean {scal[:,0].mean():.2f} "
        f"m/s p05 {np.percentile(scal[:,0],5):.2f} p95 "
        f"{np.percentile(scal[:,0],95):.2f}; |yaw| mean "
        f"{np.abs(scal[:,1]).mean():.4f} rad/s")

    # ---- 5. decoded intent for the HUD ----------------------------------- #
    model, step = build_flagship(args.enc_ckpt, device)
    routes, mans = policy_decode(model, f9, centers, device)
    del model
    torch.cuda.empty_cache()
    log(f"policy decode: {len(mans)} centers")

    # ---- 6. flow cross-checks -------------------------------------------- #
    f_eff = float(meta["achieved_f_eff"])
    fl = flow_signals(vid, f_eff, horizon_row=128.0)
    ok = fl["flow_ok"]
    log(f"flow: {ok.sum()}/{len(ok)} frames usable; |yaw| mean "
        f"{np.nanmean(np.abs(fl['flow_yaw_rate'])):.4f} rad/s")

    # ---- persist --------------------------------------------------------- #
    torch.save({"frames_u8": vid, "centers": centers}, work / "work" / "canon.pt")
    np.savez(work / "results" / "idm_reconstruction.npz",
             centers=centers, scalars=scal, traj=traj,
             route=np.array([routes.get(int(t), -1) for t in centers]),
             maneuver=np.array([mans.get(int(t), -1) for t in centers]),
             flow_yaw_rate=fl["flow_yaw_rate"], flow_ok=fl["flow_ok"],
             flow_pitch_rate=fl["flow_pitch_rate"])
    summary = {
        "meta": {
            "experiment": "idm_youtube_validation_reconstruct",
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mp4_deleted": not args.keep_raw,
            "target_hz": TARGET_HZ,
            "n_frames_10hz": int(vid.shape[0]),
            "n_windows": int(len(centers)),
            "encoder": {"ckpt": args.enc_ckpt, "step": emeta["ckpt_step"],
                        "state_dim": emeta["state_dim"]},
            "idm_head": {"path": args.head,
                         "name": hd["config"]["name"],
                         "params": hd.get("params"),
                         "val": hd.get("val", {})},
            "geometry": {"source_res": [W0, H0], "geocalib": intr.as_dict(),
                         "achieved_f_eff": meta["achieved_f_eff"],
                         "fully_canonical": meta["fully_canonical"],
                         "crop_side_src_px": int(c), "crop_box_src": crop_box},
            "privacy": {"anon_stats": meta["anon"],
                        "blur": "full-res Haar face/plate/body BEFORE downscale"},
        },
        "idm": {
            "speed_mps": {"mean": float(scal[:, 0].mean()),
                          "std": float(scal[:, 0].std()),
                          "p05": float(np.percentile(scal[:, 0], 5)),
                          "p50": float(np.percentile(scal[:, 0], 50)),
                          "p95": float(np.percentile(scal[:, 0], 95)),
                          "min": float(scal[:, 0].min()),
                          "max": float(scal[:, 0].max()),
                          "frac_in_0_45_band": float(((scal[:, 0] >= -1) &
                                                      (scal[:, 0] <= 45)).mean())},
            "yaw_rate_radps": {"abs_mean": float(np.abs(scal[:, 1]).mean()),
                               "min": float(scal[:, 1].min()),
                               "max": float(scal[:, 1].max())},
            "long_disp_2s_m": {"mean": float(traj[:, -1, 0].mean()),
                               "p05": float(np.percentile(traj[:, -1, 0], 5)),
                               "p95": float(np.percentile(traj[:, -1, 0], 95))},
        },
        "flow": {
            "n_usable": int(fl["flow_ok"].sum()), "n_total": int(len(fl["flow_ok"])),
            "abs_yaw_mean_radps": float(np.nanmean(np.abs(fl["flow_yaw_rate"]))),
            "band_rows_far_gnd": fl["band_rows"],
        },
    }
    (work / "results" / "reconstruct_summary.json").write_text(
        json.dumps(summary, indent=2))
    log("WROTE results/reconstruct_summary.json + idm_reconstruction.npz")
    print("YT_IDM_RECONSTRUCT_DONE", flush=True)


if __name__ == "__main__":
    main()
