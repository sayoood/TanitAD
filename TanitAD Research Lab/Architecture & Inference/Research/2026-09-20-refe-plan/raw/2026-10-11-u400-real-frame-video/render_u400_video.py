"""REFe SFT-4 B u400 -- planning behaviour on REAL camera frames (no synthesised frame anywhere).

What is different from raw/2026-10-01-final-video/render_sync_video.py (which motion-interpolated 2 Hz frames to 10 fps with DIS optical flow):
  * every displayed camera image is ONE decoded sensor JPEG, shown unmodified (front: native 1920x1080; side/rear: area-downscaled to 480x270);
  * front camera: the raw nuPlan CAM_F0 stream, native 10.000 Hz, each frame composited at its OWN timestamp;
  * the plan of token k is transformed into the ego pose AT THAT FRAME'S TIMESTAMP -- the high-rate (100 Hz) ego_pose row the DB itself links to
    that image (image.ego_pose_token) -- and projected with the DB camera calibration INCLUDING its Caltech lens distortion (applied to the
    projected POINTS; the image is never warped);
  * side/rear: real 10 Hz where a raw zip exists (one navtest log), otherwise the 2 Hz OpenScene frame HELD (labelled), never warped;
  * logged agents come from the DB lidar_box table at their real 20 Hz (nearest lidar_pc to the camera frame; no interpolation).

    python render_u400_video.py prep                                   -> video_meta.json (small per-token metadata)
    python render_u400_video.py still --run 3 --token 5 --sub 2 --out still.png [--mock] [--no-pose3d] [--debug-boxes]
    python render_u400_video.py render --out E:/.../u400_realframes.mp4 [--workers 4] [--runs 1,2,3]
Evidence class of the overlays: a RENDERING of the re-run u400 proposals (fidelity-gated against the banked plans) and banked per-token harness
scores.  The BEV agents and the map are rendering aids / logged ground truth, never a model input.
"""
import argparse
import collections
import csv
import gzip
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time

import cv2
import numpy as np

import u400_lib as L
from u400_lib import (BEVP, FRONT_XY, H, HUD, PANELS, PANEL_H, PANEL_W, PANEL_Y, TOP, W, R)

HERE = L.HERE
SEG = os.path.join(HERE, "segments")
MAN = os.path.join(HERE, "manifests")
FPS = L.FPS


# ------------------------------------------------------------------------------------------------- inputs
def prep():
    runs = json.load(open(L.RUNS))
    want = {t for r in runs for t in r["tokens"]}
    exp = json.load(gzip.open(L.EXPORT, "rt", encoding="utf-8"))["tokens"]
    meta = {}
    for t in want:
        x = exp[t]
        meta[t] = {"log": x["log_name"], "map": x["map_name"], "cmd": int(np.argmax(x["ego_statuses"][-1]["driving_command"])),
                   "v_tok": float(np.hypot(*x["ego_statuses"][-1]["ego_velocity"])), "hum": x["human_future_poses"], "t_us": x["timestamps_us"][-1]}
    json.dump(meta, open(L.META, "w"))
    print("wrote", L.META, len(meta))


class Ctx:
    def __init__(self, mock=False):
        self.runs = json.load(open(L.RUNS))
        self.meta = json.load(open(L.META))
        self.mock = mock
        self.props = {}
        if not mock and os.path.exists(L.PROPS):
            P = np.load(L.PROPS)
            for i, t in enumerate(P["token"]):
                self.props[str(t)] = (P["proposals"][i].astype(np.float64), P["logits"][i], int(P["pick"][i]))
        else:       # also used while inference is still running: per-log dumps
            d = os.path.join(HERE, "inference")
            for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                if f.endswith(".dump.npz"):
                    P = np.load(os.path.join(d, f))
                    for i, t in enumerate(P["token"]):
                        self.props[str(t)] = (P["proposals"][i].astype(np.float64), P["logits"][i], int(P["pick"][i]))
        self.truth = {}
        want = set(self.meta)
        for r in csv.DictReader(open(L.CSV, encoding="utf-8")):
            if r["token"] in want:
                self.truth[r["token"]] = {"score": float(r["score"]), "valid": r["valid"] == "True",
                                          "sub": [float(r[k]) for k in ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                                                                        "time_to_collision_within_bound", "comfort", "driving_direction_compliance")]}

    def plan_for(self, tok):
        hum = np.array(self.meta[tok]["hum"], dtype=np.float64)
        if self.mock or tok not in self.props:
            rng = np.random.default_rng(abs(hash(tok)) % (2 ** 32))
            prop = np.stack([hum + np.c_[rng.normal(0, 0.25 * (k % 7), 8).cumsum() * 0.1, rng.normal(0, 0.4, 8).cumsum() * 0.1 * (k % 5), np.zeros(8)]
                             for k in range(64)])
            prop[0] = hum
            sc, pick = rng.random(64), 0
            sc[0] = 1.0
        else:
            prop, logits, pick = self.props[tok]
            sc = L.agg_lane(logits)
            assert int(np.argmax(sc)) == pick, f"pick != argmax of the scorer rule for {tok}"
        nrm = (sc - sc.min()) / max(float(sc.max() - sc.min()), 1e-9)
        paths = [L.hermite(prop[i])[0] for i in range(64)]
        le, re_ = L.ribbon_edges(paths[pick])
        return {"paths": paths, "nrm": nrm, "order": np.argsort(sc), "pick": pick, "sel_edges": (le, re_), "hum": L.hermite(hum)[0],
                "ticks": prop[pick][:, :2], "sc": sc}


# ------------------------------------------------------------------------------------------------- one clip
def small(img, w=PANEL_W, h=PANEL_H):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def panel_label(canvas, x, y, label, mode):
    real = mode == "real"
    cv2.rectangle(canvas, (x, y), (x + PANEL_W, y + 26), (0, 0, 0), -1)
    R.text(canvas, label, (x + 8, y + 19), 0.55, (235, 235, 240), 1, False)
    tag = "real 10 Hz" if real else "2 Hz (held)"
    R.text(canvas, tag, (x + 120, y + 19), 0.55, (120, 235, 120) if real else (60, 170, 255), 1, False)


def compose_frame(ld, ctx, run, ri, nruns, k, ci, sub, plan, pose_k, t_k, i0k, opts, held_cache, clip_t0):
    """one video frame: front camera frame `ci` (its own timestamp and ego pose) + the plan of token k"""
    tok = run["tokens"][k]
    f0 = ld.img["CAM_F0"]
    pose_c = L.Pose(f0["pose"][ci])
    canvas = np.full((H, W, 3), 18, np.uint8)
    fx, fy = FRONT_XY
    frame = ld.read_jpg("CAM_F0", f0["fn"][ci])
    assert frame.shape == (1080, 1920, 3), frame.shape
    canvas[fy:fy + 1080, fx:fx + 1920] = frame
    front = canvas[fy:fy + 1080, fx:fx + 1920]
    L.draw_front(front, ld, ld.cam["CAM_F0"], plan, pose_k, pose_c, opts["pose3d"], ci == i0k)
    if opts.get("debug_boxes"):
        li = int(np.argmin(np.abs(ld.lid_ts - f0["ts"][ci])))
        bx = ld.boxes.get(li)
        if bx is not None:
            for b in bx:
                c_, s_ = math.cos(b[2]), math.sin(b[2])
                loc = np.array([[sx * b[4] / 2, sy * b[3] / 2] for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
                g = np.stack([b[0] + c_ * loc[:, 0] - s_ * loc[:, 1], b[1] + s_ * loc[:, 0] + c_ * loc[:, 1]], 1)
                for zz in (b[6] - b[7] / 2, b[6] + b[7] / 2):
                    Pg = np.c_[g, np.full(4, zz)]
                    Pb = (Pg - pose_c.T) @ (pose_c.Rm if opts["pose3d"] else pose_c.Rz)
                    uv, ok = ld.cam["CAM_F0"].project(Pb, near=2.0)
                    if ok.all():
                        cv2.polylines(front, [np.round(uv).astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
    # labels on the front panel
    cv2.rectangle(canvas, (fx, fy), (fx + 560, fy + 30), (0, 0, 0), -1)
    R.text(canvas, "FRONT  CAM_F0  real frame, 10 Hz, native 1920x1080", (fx + 8, fy + 21), 0.55, (235, 235, 240), 1, False)
    cv2.rectangle(canvas, (fx + 1920 - 520, fy), (fx + 1920, fy + 30), (0, 0, 0), -1)
    cv2.rectangle(canvas, (fx + 1920 - 512, fy + 8), (fx + 1920 - 492, fy + 22), (205, 190, 40), -1)
    R.text(canvas, "drivable area: map rendering aid, not a model input", (fx + 1920 - 486, fy + 21), 0.5, (225, 225, 230), 1, False)
    if ci == i0k:
        cv2.rectangle(canvas, (fx, fy + 1080 - 34), (fx + 520, fy + 1080), (0, 0, 0), -1)
        R.text(canvas, "this is the camera frame the planner saw (token time)", (fx + 8, fy + 1080 - 12), 0.55, (120, 235, 255), 1, False)
    if k == 0 and (ci - run["_i0"][0]) < 8:
        cv2.rectangle(canvas, (fx + 560, fy + 40), (fx + 1360, fy + 90), (20, 20, 20), -1)
        R.text(canvas, f"NEW CLIP {ri + 1}/{nruns}  (cut)  {run['cams']}", (fx + 580, fy + 75), 0.9, (80, 220, 255), 2, False)
    # side / rear panels
    srcs = {}
    for ch, label, x in PANELS:
        mode = ld.src[ch][1]
        if mode == "real":
            j = ld.nearest(ch, int(f0["ts"][ci]))
            nm, dt = ld.img[ch]["fn"][j], abs(int(ld.img[ch]["ts"][j]) - int(f0["ts"][ci])) / 1000.0
            img = small(ld.read_jpg(ch, nm))
        else:
            if ch not in held_cache or held_cache[ch][0] != k:
                j = ld.nearest(ch, int(t_k))
                nm = ld.img[ch]["fn"][j]
                dt = abs(int(ld.img[ch]["ts"][j]) - int(t_k)) / 1000.0
                held_cache[ch] = (k, small(ld.read_jpg(ch, nm)), nm, dt)
            _, img, nm, dt = held_cache[ch]
        canvas[PANEL_Y:PANEL_Y + PANEL_H, x:x + PANEL_W] = img
        panel_label(canvas, x, PANEL_Y, label, mode)
        srcs[ch] = (mode, nm, dt)
    # BEV
    bev = R.Bev(BEVP, ppm=L.BEV_PPM, x0=L.BEV_X0)
    li = int(np.argmin(np.abs(ld.lid_ts - f0["ts"][ci])))
    L.draw_bev_base(canvas, bev, ld, pose_c, ld.boxes.get(li)[:, :6] if li in ld.boxes else None)
    L.draw_bev_plan(canvas, bev, plan, pose_k, pose_c)
    L.bev_overlays_static(canvas, bev)
    # title + HUD
    cv2.rectangle(canvas, (0, 0), (W, TOP), (10, 10, 12), -1)
    R.text(canvas, L.TITLE, (16, 28), 0.8, (240, 240, 245), 2, False)
    sc = plan["sc"]
    tr = ctx.truth.get(tok)
    d = {"cmd": ctx.meta[tok]["cmd"], "speed": pose_c.speed, "token": tok, "t": (f0["ts"][ci] - clip_t0) / 1e6, "age": (f0["ts"][ci] - t_k) / 1e6,
         "run": ri + 1, "nruns": nruns, "log": ld.log, "f0_ts_s": (f0["ts"][ci] - f0["ts"][0]) / 1e6, "f0_name": f0["fn"][ci][:12],
         "pick": plan["pick"], "pred_sel": float(sc[plan["pick"]]), "pred_min": float(sc.min()), "pred_max": float(sc.max()), "true": tr,
         "clip_prog": (f0["ts"][ci] - clip_t0) / max(run["_dur_us"], 1)}
    L.hud(canvas, d)
    return canvas, {"ts": int(f0["ts"][ci]), "name": f0["fn"][ci], "pose_dt_us": int(f0["ets"][ci] - f0["ts"][ci]), "srcs": srcs,
                    "speed_db": pose_c.speed, "speed_tok": ctx.meta[tok]["v_tok"]}


def prepare_run(ld, ctx, run):
    toks = run["tokens"]
    kidx = [ld.lid_tok[t] for t in toks]
    t_k = [int(ld.lid_ts[i]) for i in kidx]
    assert all(abs(t_k[a + 1] - t_k[a] - 500000) < 20000 for a in range(len(t_k) - 1)), "tokens not 0.5 s apart in the DB"
    i0 = [ld.nearest("CAM_F0", t) for t in t_k]
    i1 = i0[1:] + [i0[-1] + 5]
    run["_i0"], run["_i1"] = i0, i1
    run["_dur_us"] = int(ld.img["CAM_F0"]["ts"][i1[-1] - 1] - ld.img["CAM_F0"]["ts"][i0[0]]) + 100000
    return kidx, t_k


def render_run(ld, ctx, run, ri, nruns, opts, out_path=None, still=None):
    kidx, t_k = prepare_run(ld, ctx, run)
    n = len(run["tokens"])
    ld.boxes = {}
    ld.load_boxes(t_k[0] - 400000, t_k[-1] + 900000)
    f0 = ld.img["CAM_F0"]
    clip_t0 = int(f0["ts"][run["_i0"][0]])
    writer = None
    if still is None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cmd = [L.FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-c:v", "libx264", "-preset", opts.get("preset", "slow"), "-crf", str(opts.get("crf", 18)), "-threads", "6", "-pix_fmt", "yuv420p",
               "-g", "20", "-video_track_timescale", "10000", "-movflags", "+faststart", out_path]
        writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rows, held_cache, nframes = [], {}, 0
    for k in range(n):
        plan = ctx.plan_for(run["tokens"][k])
        pose_k = L.Pose(ld.lid_pose[kidx[k]])
        for sub, ci in enumerate(range(run["_i0"][k], run["_i1"][k])):
            if still is not None and (k, sub) != still:
                continue
            canvas, info = compose_frame(ld, ctx, run, ri, nruns, k, ci, sub, plan, pose_k, t_k[k], run["_i0"][k], opts, held_cache, clip_t0)
            if still is not None:
                return canvas
            writer.stdin.write(canvas.tobytes())
            rows.append([ri + 1, nframes, k, run["tokens"][k], sub, info["ts"], info["name"], info["pose_dt_us"]] +
                        [x for ch in ("CAM_L0", "CAM_R0", "CAM_B0") for x in (info["srcs"][ch][0], info["srcs"][ch][1], f"{info['srcs'][ch][2]:.2f}")] +
                        [f"{info['speed_db']:.3f}", f"{info['speed_tok']:.3f}"])
            nframes += 1
    writer.stdin.close()
    writer.wait()
    os.makedirs(MAN, exist_ok=True)
    with open(os.path.join(MAN, f"clip{ri + 1:03d}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "frame_in_clip", "token_idx", "token", "sub", "f0_ts_us", "f0_jpg", "f0_ego_pose_dt_us", "L_mode", "L_jpg", "L_dt_ms",
                    "R_mode", "R_jpg", "R_dt_ms", "B_mode", "B_jpg", "B_dt_ms", "speed_db_mps", "speed_token_mps"])
        w.writerows(rows)
    return nframes


def log_task(args):
    log, ris, opts = args
    ctx = Ctx(mock=opts.get("mock", False))
    runs = ctx.runs
    t0 = time.time()
    ld = L.LogData(log, pose3d=opts["pose3d"])
    pts = np.concatenate([ld.lid_pose[[ld.lid_tok[t] for t in runs[ri]["tokens"]]][:, :2] for ri in ris])
    ld.load_map(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max(), ctx.meta[runs[ris[0]]["tokens"][0]]["map"])
    out = {}
    for ri in ris:
        seg = os.path.join(SEG, f"clip{ri + 1:03d}.mp4")
        if os.path.exists(seg + ".done"):
            out[ri] = int(open(seg + ".done").read())
            continue
        nf = render_run(ld, ctx, runs[ri], ri, len(runs), opts, out_path=seg)
        open(seg + ".done", "w").write(str(nf))
        out[ri] = nf
        print(f"  clip {ri + 1}/{len(runs)} {log[-11:]}: {nf} frames  ({time.time() - t0:.0f} s)", flush=True)
    return out


# ------------------------------------------------------------------------------------------------- cards
def stats_line(ctx):
    sc = np.array([ctx.truth[t]["score"] for r in ctx.runs for t in r["tokens"] if t in ctx.truth])
    return len(sc), 100 * sc.mean(), int((sc < 0.5).sum()), 100 * float(np.median(sc))


def card_frames(kind, ctx, nclips, nframes_total):
    n, mean, nfail, med = stats_line(ctx)
    secs = nframes_total / FPS
    out = []
    nfr = 80 if kind == "intro" else 60
    for f in range(nfr):
        a = min(1.0, f / 8.0)
        c = np.full((H, W, 3), 16, np.uint8)

        def sh(col):
            return tuple(int(v * a) for v in col)
        y = 260
        R.text(c, "TanitAD  -  REFe SFT-4 arm B, checkpoint u400: what the planner sees and chooses", (110, y), 2.0, sh((240, 240, 245)), 3, False)
        R.text(c, L.TITLE, (110, y + 90), 1.35, sh((90, 255, 90)), 2, False)
        lines = ([f"{nclips} clips of consecutive NAVSIM navtest tokens (0.5 s apart), {secs / 60:.1f} min, from the navtest logs that have RAW nuPlan camera frames.",
                  "EVERY camera image is a real sensor frame at its own timestamp. No optical-flow interpolation, no warping, no synthesised frames.",
                  "The plan of each token is projected into the ego pose of EACH video frame (DB ego_pose, 100 Hz) with the DB camera calibration incl. lens distortion.",
                  "",
                  "front camera: raw 10 Hz.   side / rear: raw 10 Hz in the clips labelled \"real 10 Hz\"; otherwise the 2 Hz frame is HELD and labelled \"2 Hz (held)\".",
                  "BEV: nuPlan map + logged agents (DB lidar_box, 20 Hz) are rendering aids; they are NOT model inputs.   Drivable area = NAVSIM's DAC polygons.",
                  "Plans: u400 re-run on exactly these tokens; reproduces the banked full-navtest plans (fidelity gate in RESULT_VIDEO.md)."]
                 if kind == "intro" else
                 [f"These {n} tokens: mean true PDMS {mean:.2f} (median {med:.1f}); {nfail} tokens score < 0.5 (shown in red as FAILURE).",
                  "This is a sample of 10 logs, NOT the benchmark: the full 12,146-token navtest result of this checkpoint is PDMS 88.25.",
                  "True per-token scores: NAVSIM harness, this checkpoint's executed plan (u400 full-navtest CSV).",
                  "The predicted PDMS on screen is the scorer head's own estimate and can differ from the true score."])
        for i, s in enumerate(lines):
            R.text(c, s, (110, y + 210 + 74 * i), 1.0, sh((215, 215, 222)), 1, False)
        out.append(c)
    return out


def write_card(path, frames):
    cmd = [L.FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
           "-preset", "slow", "-crf", "18", "-threads", "6", "-pix_fmt", "yuv420p", "-g", "20", "-video_track_timescale", "10000", "-movflags", "+faststart", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()


# ------------------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "still", "render", "cards", "concat"))
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--token", type=int, default=0)
    ap.add_argument("--sub", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-pose3d", action="store_true")
    ap.add_argument("--debug-boxes", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--runs", default=None, help="1-based comma list of clips to render (default: all)")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="slow")
    a = ap.parse_args()
    opts = {"pose3d": not a.no_pose3d, "debug_boxes": a.debug_boxes, "mock": a.mock, "crf": a.crf, "preset": a.preset}
    if a.cmd == "prep":
        return prep()
    ctx = Ctx(mock=a.mock)
    if a.cmd == "still":
        ri = a.run - 1
        run = ctx.runs[ri]
        ld = L.LogData(run["log"], pose3d=opts["pose3d"])
        pts = ld.lid_pose[[ld.lid_tok[t] for t in run["tokens"]]][:, :2]
        ld.load_map(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max(), ctx.meta[run["tokens"][0]]["map"])
        img = render_run(ld, ctx, run, ri, len(ctx.runs), opts, still=(a.token, a.sub))
        cv2.imwrite(a.out, img)
        print("wrote", a.out, img.shape)
        return 0
    if a.cmd == "cards":
        nfr = int(sum(int(open(os.path.join(SEG, f)).read()) for f in os.listdir(SEG) if f.endswith(".done")))
        write_card(os.path.join(SEG, "00_intro.mp4"), card_frames("intro", ctx, len(ctx.runs), nfr))
        write_card(os.path.join(SEG, "99_outro.mp4"), card_frames("outro", ctx, len(ctx.runs), nfr))
        print("cards written")
        return 0
    if a.cmd == "concat":
        sel_c = sorted(int(x) - 1 for x in a.runs.split(",")) if a.runs else range(len(ctx.runs))
        files = [os.path.join(SEG, "00_intro.mp4")] + [os.path.join(SEG, f"clip{i + 1:03d}.mp4") for i in sel_c] + [os.path.join(SEG, "99_outro.mp4")]
        assert all(os.path.exists(f) for f in files), [f for f in files if not os.path.exists(f)]
        lst = os.path.join(SEG, "concat.txt")
        open(lst, "w").write("".join(f"file '{f.replace(chr(92), '/')}'\n" for f in files))
        subprocess.check_call([L.FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", a.out])
        print("wrote", a.out, os.path.getsize(a.out))
        return 0
    # render
    os.makedirs(SEG, exist_ok=True)
    sel = set(int(x) - 1 for x in a.runs.split(",")) if a.runs else set(range(len(ctx.runs)))
    by_log = collections.OrderedDict()
    for i, r in enumerate(ctx.runs):
        if i in sel:
            by_log.setdefault(r["log"], []).append(i)
    tasks = sorted(((lg, ris, opts) for lg, ris in by_log.items()), key=lambda t: -sum(len(ctx.runs[i]["tokens"]) for i in t[1]))
    # split big logs so four workers stay busy
    split = []
    for lg, ris, o in tasks:
        tot = sum(len(ctx.runs[i]["tokens"]) for i in ris)
        parts = 2 if tot > 150 else 1
        step = (len(ris) + parts - 1) // parts
        for p in range(parts):
            part = ris[p * step:(p + 1) * step]
            if part:
                split.append((lg, part, o))
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = {}
        for r in pool.imap_unordered(log_task, split):
            res.update(r)
    tot = sum(res.values())
    print(f"ZZRENDER_DONE clips {len(res)} frames {tot} ({tot / FPS:.1f} s of video) in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
