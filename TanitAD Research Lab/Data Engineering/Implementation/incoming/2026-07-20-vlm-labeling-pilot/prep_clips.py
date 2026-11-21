#!/usr/bin/env python3
"""Prep the Cosmos-Drive-Dreams pilot clips: sample keyframes + derive ego kinematics.

Non-destructive: reads /root/cosmos_data, writes only to /root/vlm_pilot.
"""
import glob, io, json, os, re, subprocess, tarfile
import numpy as np

GEN = "/root/cosmos_data/pairs/generation"
VP = "/root/cosmos_data/pairs/vehicle_pose"
OUT = "/root/vlm_pilot"
FRAMES = os.path.join(OUT, "frames")
NFRAMES = 8
W, H = 896, 512

os.makedirs(FRAMES, exist_ok=True)


CLIP_RE = re.compile(r"^(?P<uuid>[0-9a-fA-F-]{36})_(?P<t0>\d+)_(?P<t1>\d+)_(?P<cam>\d+)_(?P<weather>.+)$")


def clip_key(fn):
    """generation file: <uuid>_<t0>_<t1>_<camidx>_<Weather>.mp4 -> pose key <uuid>_<t0>_<t1>.

    Weather may itself contain '_' (e.g. 'Golden_hour'), so parse structurally.
    """
    base = os.path.basename(fn)[:-4]
    m = CLIP_RE.match(base)
    if not m:
        parts = base.split("_")
        return "_".join(parts[:-2]), parts[-2], parts[-1]
    return ("%s_%s_%s" % (m.group("uuid"), m.group("t0"), m.group("t1")),
            m.group("cam"), m.group("weather"))


def ego_kinematics(key):
    """Return speed profile (m/s) + summary from the vehicle_pose tar, or None."""
    tp = os.path.join(VP, key + ".tar")
    if not os.path.exists(tp):
        return None
    try:
        tf = tarfile.open(tp)
        names = sorted(n for n in tf.getnames() if n.endswith(".npy"))
        P = []
        for n in names:
            a = np.load(io.BytesIO(tf.extractfile(n).read()), allow_pickle=True)
            P.append(np.asarray(a, dtype=np.float64))
        tf.close()
        if len(P) < 4:
            return None
        # translation column of each 4x4
        xyz = np.stack([p[:3, 3] for p in P], 0)
        # clip wall duration from the key timestamps. Units are MICROseconds:
        # (t1-t0)=2e7 -> 20 s, giving 300 poses @ 15 Hz and plausible road speeds.
        parts = key.split("_")
        t0, t1 = float(parts[-2]), float(parts[-1])
        dur_s = (t1 - t0) / 1e6
        n = len(P)
        dt = dur_s / n
        d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        speed = d / dt
        # yaw from rotation column
        yaw = np.array([np.arctan2(p[1, 0], p[0, 0]) for p in P])
        yaw_un = np.unwrap(yaw)
        yaw_rate = np.diff(yaw_un) / dt
        return dict(
            n_poses=n, dur_s=round(dur_s, 3), hz=round(1.0 / dt, 2),
            speed_mps=[round(float(v), 3) for v in speed],
            v_mean=round(float(np.mean(speed)), 2),
            v_start=round(float(np.mean(speed[: max(1, int(1 / dt))])), 2),
            v_end=round(float(np.mean(speed[-max(1, int(1 / dt)):])), 2),
            v_max=round(float(np.max(speed)), 2),
            v_min=round(float(np.min(speed)), 2),
            path_len_m=round(float(np.sum(d)), 1),
            yaw_change_deg=round(float(np.degrees(yaw_un[-1] - yaw_un[0])), 1),
            yaw_rate_max_dps=round(float(np.degrees(np.abs(yaw_rate).max())), 1),
            accel_mean=round(float((np.mean(speed[-5:]) - np.mean(speed[:5])) / max(dur_s, 1e-6)), 3),
        )
    except Exception as e:
        return dict(error=str(e)[:200])


def extract_frames(mp4, cid):
    d = os.path.join(FRAMES, cid)
    os.makedirs(d, exist_ok=True)
    got = sorted(glob.glob(os.path.join(d, "*.jpg")))
    if len(got) >= NFRAMES:
        return got[:NFRAMES]
    # probe duration
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "format=duration", "-of", "csv=p=0", mp4],
                        capture_output=True, text=True, timeout=60)
    try:
        dur = float(pr.stdout.strip())
    except Exception:
        dur = 5.0
    for i in range(NFRAMES):
        ts = dur * (i + 0.5) / NFRAMES
        op = os.path.join(d, "f%02d.jpg" % i)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "%.3f" % ts, "-i", mp4,
                        "-frames:v", "1", "-vf", "scale=%d:%d" % (W, H), "-q:v", "3", op],
                       capture_output=True, timeout=120)
    return sorted(glob.glob(os.path.join(d, "*.jpg")))


def main():
    mp4s = sorted(glob.glob(os.path.join(GEN, "*.mp4")))
    print("clips found:", len(mp4s))
    manifest = []
    for i, mp4 in enumerate(mp4s):
        key, camidx, weather = clip_key(mp4)
        cid = os.path.basename(mp4)[:-4]
        frames = extract_frames(mp4, cid)
        kin = ego_kinematics(key)
        rec = dict(clip_id=cid, pose_key=key, cam_idx=camidx,
                   weather_filename_gt=weather, mp4=mp4,
                   frames=frames, n_frames=len(frames), kinematics=kin)
        manifest.append(rec)
        if i < 3 or i % 12 == 0:
            print("[%2d] %s w=%-8s frames=%d v_mean=%s hz=%s yaw=%s" % (
                i, cid[:28], weather, len(frames),
                (kin or {}).get("v_mean"), (kin or {}).get("hz"),
                (kin or {}).get("yaw_change_deg")))
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    ok = [m for m in manifest if m["n_frames"] == NFRAMES]
    kok = [m for m in manifest if m["kinematics"] and "v_mean" in (m["kinematics"] or {})]
    print("\nSUMMARY: %d clips, %d with %d frames, %d with kinematics" % (
        len(manifest), len(ok), NFRAMES, len(kok)))
    if kok:
        vs = [m["kinematics"]["v_mean"] for m in kok]
        hz = {m["kinematics"]["hz"] for m in kok}
        print("  v_mean range: %.1f - %.1f m/s (%.0f - %.0f km/h)" % (
            min(vs), max(vs), min(vs) * 3.6, max(vs) * 3.6))
        print("  pose hz seen:", sorted(hz)[:5])
    from collections import Counter
    print("  weather GT:", Counter(m["weather_filename_gt"] for m in manifest))


if __name__ == "__main__":
    main()
