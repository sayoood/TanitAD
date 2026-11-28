"""v2 corpus pool scorer — CHEAP EGOMOTION ONLY, camera-aligned first-20 s window.

Phase-1 selection-design tool (NO camera download). Scores every locally-available
egomotion clip on the SAME 20 s window the camera covers, so each clip's maneuver
signature matches what the trainer will actually see when the camera is fetched.

WHY THE FIRST 20 s (MEASURED, align_probe on 40 clips 2026-07-24):
  Each clip_id has (a) an egomotion parquet spanning the whole recording
  (~114 s mean, ~35 Hz native) and (b) ONE front_wide mp4 of ~20.1 s (605 frames
  @ ~30 fps). The camera excerpt sits at the START of the egomotion recording:
  cam_start_frac = (cam_t0 - ego_t0)/ego_span was 0.000-0.008 for all 40 probed
  clips. build_episode() resamples the camera's 20.1 s span to 10 Hz -> ~201
  poses, drops the first n_stack-1=2 (stacking) -> ~199 poses (matches the parity
  cache: 472627/2376 = 198.9 mean). We reproduce that window from egomotion alone
  by taking t_query = linspace(ego_t0, ego_t0 + 20.1 s, 201) and poses[2:].

FAITHFULNESS: reuses tanitad.data.physicalai.signals_at (the exact pose
convention: x, y, quaternion-yaw, hypot(vx,vy)) and scripts/refb_labels
(the exact maneuver classes/thresholds) verbatim — no re-implementation. The
aggregate over all 18,988 valid&train clips reproduces the parity profile
(turns 17.2 % vs parity 14.25 %; speed regime 7.1/51/42 vs 7.8/46/46), which is
the cross-check that the window and labeler are right.

Output: v2_pool_scored.parquet (one row per clip; maneuver histogram, turn/
junction/stop presence, curvature, speed regime, country/hour). ~3.5 min on the
dev box for 197 chunks / ~19 k clips.

Usage:
  python score_v2_pool.py [--root <physicalai root>] [--out v2_pool_scored.parquet]
"""
from __future__ import annotations
import argparse, glob, io, math, os, re, sys, time, zipfile
import numpy as np, pandas as pd, torch

# refb_labels + tanitad live under stack/; add to path (append, never shadow stdlib)
_STACK = os.environ.get("TANITAD_STACK",
                        r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack")
sys.path.insert(0, _STACK); sys.path.append(os.path.join(_STACK, "scripts"))
import refb_labels as rl                                   # noqa: E402
from tanitad.data.physicalai import signals_at             # noqa: E402

HZ = 10.0
CAM_S = 20.1        # measured front_wide excerpt length (605 frames / ~30 fps)
K_STACK = 2         # n_stack-1 poses dropped by build_episode


def _detect_unit(span: float) -> float:
    for cand in (1e9, 1e6, 1e3):                           # ns / us / ms -> s
        if span / cand > 1.0:
            return cand
    return 1.0


def window_poses(ego: pd.DataFrame):
    """First-20 s camera-aligned poses[T,4] via the build_episode convention."""
    t = np.sort(ego["timestamp"].to_numpy(np.float64))
    unit = _detect_unit(t[-1] - t[0])
    t0, t_end = t[0], min(t[-1], t[0] + CAM_S * unit)
    n_target = max(int((t_end - t0) / unit * HZ), K_STACK + 21)
    _, poses = signals_at(ego, np.linspace(t0, t_end, n_target))
    return torch.from_numpy(poses)[K_STACK:], (t_end - t0) / unit


def score(poses: torch.Tensor):
    T = poses.shape[0]
    if T <= rl.LABEL_HORIZON:
        return None
    v, yaw = poses[:, 3], poses[:, 2]
    m1 = rl.maneuver_labels(poses, horizon=rl.LABEL_HORIZON)          # v1 (headline)
    m2 = rl.maneuver_labels_v2(poses, horizon=rl.LABEL_HORIZON)       # v2 curvature-gated
    h1 = torch.bincount(m1, minlength=5).tolist()
    h2 = torch.bincount(m2, minlength=5).tolist()
    dyaw = (yaw[1:] - yaw[:-1]); dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi
    junction = False
    for tt in range(0, T, 5):                                         # v2.1 junction proxy
        if rl.route_from_future_v21(poses, tt)["reason"] == "tight_transient":
            junction = True; break
    return dict(
        T=T, mean_v=float(v.mean()), stop_frac=float((v < 0.5).float().mean()),
        dist_m=float(v.sum()) * 0.1,
        net_head=abs(math.degrees(float(((yaw[-1] - yaw[0] + math.pi) % (2 * math.pi)) - math.pi))),
        cum_head=math.degrees(float(dyaw.abs().sum())), junction=int(junction),
        stopped=float((v < 1.0).float().mean()),
        city=float(((v >= 1) & (v <= 12)).float().mean()),
        hw=float((v > 12).float().mean()),
        lk=h1[0], tl=h1[1], tr=h1[2], ac=h1[3], bs=h1[4],
        lk2=h2[0], tl2=h2[1], tr2=h2[2], ac2=h2[3], bs2=h2[4], nlab=int(sum(h1)),
        has_stop=int((v < 0.5).any()), has_brake=int(h1[4] > 0), has_turn=int(h1[1] + h1[2] > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\Admin\tanitad-data\physicalai")
    ap.add_argument("--out", default="v2_pool_scored.parquet")
    a = ap.parse_args()
    cat = (pd.read_parquet(os.path.join(a.root, "clip_index.parquet"))
           .join(pd.read_parquet(os.path.join(a.root, "metadata", "data_collection.parquet"))))
    zips = sorted(glob.glob(os.path.join(a.root, "labels", "egomotion", "egomotion.chunk_*.zip")))
    print(f"[score] {len(zips)} local egomotion chunks", flush=True)
    rows, t0, nbad = [], time.time(), 0
    for zi, zp in enumerate(zips):
        ch = int(re.search(r"chunk_(\d+)", os.path.basename(zp)).group(1))
        try:
            zf = zipfile.ZipFile(zp)
        except Exception as e:
            print(f"[score] chunk {ch} zip fail: {e}", flush=True); continue
        with zf as z:
            for name in z.namelist():
                if not name.endswith(".egomotion.parquet"):
                    continue
                cid = name.split("/")[-1].split(".")[0]
                try:
                    poses, span = window_poses(pd.read_parquet(io.BytesIO(z.read(name))))
                    s = score(poses)
                except Exception:
                    nbad += 1; continue
                if s is None:
                    continue
                s.update(clip_id=cid, chunk=ch, win_s=span); rows.append(s)
        if zi % 20 == 0:
            print(f"[score] chunk {zi}/{len(zips)} (#{ch}) clips={len(rows)} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    meta = cat.reset_index().rename(columns={"index": "clip_id"})
    meta["clip_id"] = meta["clip_id"].astype(str)
    df = df.merge(meta[["clip_id", "country", "hour_of_day", "platform_class",
                        "split", "clip_is_valid"]], on="clip_id", how="left")
    df.to_parquet(a.out)
    print(f"[score] WROTE {a.out} n={len(df)} bad={nbad} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
