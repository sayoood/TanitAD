"""QA of the finished video: encoding facts, frame accounting, camera-rate/provenance statistics from the per-frame manifests, and a
PROVENANCE test that every displayed front frame is ONE real source JPEG (no blend of neighbours, i.e. no interpolation).

    python qa_video.py <video.mp4> [--n 36] [--stills]
Writes qa_report.json (+ stills/*.png with --stills).  Provenance test: for random output frames, the front panel region that no overlay ever
touches (front y in [110, 400]: sky / buildings, above the horizon, below the title bars) is compared with (a) the frame's own source JPEG and
(b) the ADJACENT source JPEG (+-0.1 s).  A real frame matches its own source far better than its neighbour; a motion-interpolated blend would
match neither well.
"""
import csv
import glob
import json
import os
import subprocess
import sys
import zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEG = os.path.join(HERE, "segments")
MAN = os.path.join(HERE, "manifests")
DRV = "E:"
MINI = f"{DRV}/Projects/TanitAD/data/nuplan-camera/mini"
INTRO_FR, OUTRO_FR = 80, 60
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def grab(video, idx, fps=10):
    t = max(0.0, idx / fps - 0.02)      # -ss seeks to the FIRST frame with pts >= t, i.e. frame idx (pts = idx / fps)
    out = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{t:.4f}", "-i", video, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                         capture_output=True, check=True).stdout
    return cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)


def main():
    video = sys.argv[1]
    n_prov = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 36
    runs = json.load(open(os.path.join(HERE, "video_runs.json")))
    rep = {"video": video, "bytes": os.path.getsize(video)}
    pr = json.loads(subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                                    "stream=codec_name,profile,width,height,r_frame_rate,nb_read_frames,duration,bit_rate,pix_fmt", "-of", "json", video],
                                   capture_output=True, text=True, check=True).stdout)["streams"][0]
    rep["ffprobe"] = pr
    done = {int(os.path.basename(f)[4:7]): int(open(f).read()) for f in glob.glob(os.path.join(SEG, "clip*.mp4.done"))}
    rows = []
    for ri in sorted(done):
        with open(os.path.join(MAN, f"clip{ri:03d}.csv"), encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    expected = INTRO_FR + OUTRO_FR + sum(done.values())
    rep["frames"] = {"expected": expected, "decoded": int(pr["nb_read_frames"]), "match": expected == int(pr["nb_read_frames"]),
                     "clips": len(done), "clip_frames_sum": sum(done.values()), "manifest_rows": len(rows)}
    # camera rate / provenance statistics (manifest = what was drawn)
    d_ms, bad = [], 0
    prev = None
    for r in rows:
        key = r["clip"]
        ts = int(r["f0_ts_us"])
        if prev is not None and prev[0] == key:
            d = (ts - prev[1]) / 1000.0
            d_ms.append(d)
            bad += not (95.0 <= d <= 105.0)
        prev = (key, ts)
    d_ms = np.array(d_ms)
    rep["front_frame_spacing_ms"] = {"n": int(len(d_ms)), "min": float(d_ms.min()), "median": float(np.median(d_ms)), "max": float(d_ms.max()),
                                     "outside_95_105": int(bad), "rate_hz_from_median": float(1000.0 / np.median(d_ms))}
    rep["f0_ego_pose_dt_us_max_abs"] = int(max(abs(int(r["f0_ego_pose_dt_us"])) for r in rows))
    cams = {}
    for ch, pre in (("CAM_L0", "L"), ("CAM_R0", "R"), ("CAM_B0", "B")):
        real = [float(r[f"{pre}_dt_ms"]) for r in rows if r[f"{pre}_mode"] == "real"]
        held = [float(r[f"{pre}_dt_ms"]) for r in rows if r[f"{pre}_mode"] == "held"]
        cams[ch] = {"frames_real_10hz": len(real), "frames_held_2hz": len(held), "real_max_dt_to_front_ms": max(real) if real else None,
                    "held_max_dt_to_token_ms": max(held) if held else None}
    rep["side_rear_cameras"] = cams
    tot = len(rows)
    rep["share_all_four_real"] = float(sum(1 for r in rows if r["L_mode"] == "real" and r["R_mode"] == "real" and r["B_mode"] == "real") / tot)
    sp = np.array([abs(float(r["speed_db_mps"]) - float(r["speed_token_mps"])) for r in rows if r["sub"] == "0"])
    rep["speed_db_vs_token_abs_diff_mps"] = {"n": int(len(sp)), "median": float(np.median(sp)), "p95": float(np.percentile(sp, 95)), "max": float(sp.max())}
    # provenance: own source vs adjacent source
    rng = np.random.default_rng(20261011)
    # moving frames only (> 3 m/s): at standstill the neighbouring source frame is nearly identical, so the test cannot discriminate
    cand = [i for i, r in enumerate(rows) if float(r["speed_db_mps"]) > 3.0 and not (r["token_idx"] == "0" and int(r["sub"]) < 8)]
    pick = rng.choice(cand, size=min(n_prov, len(cand)), replace=False)
    cum, c = {}, 0
    for ri in sorted(done):
        cum[ri] = c
        c += done[ri]
    res = []
    zips = {}
    for i in sorted(pick):
        r = rows[i]
        ri = int(r["clip"])
        gi = INTRO_FR + cum[ri] + int(r["frame_in_clip"])
        log = runs[ri - 1]["log"]
        zp = f"{MINI}/{log}_CAM_F0.zip"
        if zp not in zips:
            zips[zp] = zipfile.ZipFile(zp)
        z = zips[zp]
        src = cv2.imdecode(np.frombuffer(z.read(r["f0_jpg"]), np.uint8), cv2.IMREAD_COLOR)
        names = [r2["f0_jpg"] for r2 in rows if r2["clip"] == r["clip"]]
        j = names.index(r["f0_jpg"])
        nb = names[j + 1] if j + 1 < len(names) else names[j - 1]
        adj = cv2.imdecode(np.frombuffer(z.read(nb), np.uint8), cv2.IMREAD_COLOR)
        fr = grab(video, gi)[40:40 + 1080, 0:1920]
        sl = (slice(110, 400), slice(0, 1900))
        res.append({"global_frame": int(gi), "clip": ri, "psnr_own_source_db": psnr(fr[sl], src[sl]), "psnr_adjacent_source_db": psnr(fr[sl], adj[sl])})
    own = np.array([x["psnr_own_source_db"] for x in res])
    adj = np.array([x["psnr_adjacent_source_db"] for x in res])
    rep["provenance_psnr"] = {"n": len(res), "own_min": float(own.min()), "own_median": float(np.median(own)), "adjacent_median": float(np.median(adj)),
                              "adjacent_max": float(adj.max()), "own_beats_adjacent_every_frame": bool((own > adj).all()),
                              "min_gap_db": float((own - adj).min()), "samples": res}
    json.dump(rep, open(os.path.join(HERE, "qa_report.json"), "w"), indent=1)
    if "--stills" in sys.argv:
        os.makedirs(os.path.join(HERE, "stills"), exist_ok=True)
        want = (("still_1_turn_4cams_real.png", 4, 4, 2), ("still_2_failure_token_right_turn.png", 15, 8, 3), ("still_3_speed_lane_keeping_4cams_real.png", 2, 4, 2))
        for name, ri, k, sub in want:
            if ri not in done:
                continue
            r = next(r for r in rows if int(r["clip"]) == ri and int(r["token_idx"]) == k and int(r["sub"]) == sub)
            gi = INTRO_FR + cum[ri] + int(r["frame_in_clip"])
            cv2.imwrite(os.path.join(HERE, "stills", name), grab(video, gi))
            print("still", name, "clip", ri, "token_idx", k, "global frame", gi)
    print(json.dumps({k: rep[k] for k in ("frames", "front_frame_spacing_ms", "provenance_psnr")}, indent=1)[:2500])


if __name__ == "__main__":
    main()
