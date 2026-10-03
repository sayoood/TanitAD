"""Synchronised playback: the camera images ADVANCE in time while the planner's 64 hypotheses and its selected plan stay
fixed in the WORLD, so the plan is seen to be driven through (and the next plan replaces it every 0.5 s).

For each run of consecutive navtest tokens (2 Hz) of one log:
  * the four camera images are motion-interpolated (DIS optical flow) from the 2 Hz frames to 10 fps;
  * the plan of token k (64 hypotheses on NAVSIM's 0.5 s grid, ego frame of k) is transformed with the logged ego poses into
    the INTERPOLATED ego frame of every video frame and projected with the camera calibration -- so it is geometrically
    consistent with the moving image; at the next token the new plan takes over;
  * the BEV is drawn in the current ego frame; logged agents are interpolated between frames by track token (they are a
    rendering aid, never a model input);
  * the NAVSIM driving command, ego speed and the scorer's predicted PDMS are read per token.

Evidence: a rendering of banked proposals (eval/proptable/navtest_final/proposals.npz from the full-navtest run) and logged data.
    python render_sync_video.py --plans full --out sync.mp4 [--runs LOG:START:LEN,...] [--max-runs N]
    python render_sync_video.py --plans human-mock --still ... (pipeline test: the human future stands in for the plan)
"""
import argparse
import collections
import gzip
import json
import math
import os
import pickle
import subprocess
import sys
import zipfile

import cv2
import numpy as np

import render_final_video as R

W, H = R.W, R.H
FPS = 10
SUB = 5                                     # video frames per 0.5 s token step
DATA, EXPORT, LOGS = R.DATA, R.EXPORT, R.LOGS
CMD = R.CMD
FRONT, BEVP = R.FRONT, R.BEVP


def _true_scores():
    """the harness's per-token scores of the EXECUTED plan (full-navtest CSV), when it exists; {} otherwise"""
    import csv
    p = f"{DATA}/score/refe_navtest_final/refe_navtest_final.csv"
    if not os.path.exists(p):
        return {}
    keys = ("score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
            "time_to_collision_within_bound", "comfort")
    return {r["token"]: {k: float(r[k]) for k in keys} for r in csv.DictReader(open(p, encoding="utf-8"))
            if r.get("token") not in (None, "average")}


TRUE = _true_scores()


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Pose:
    def __init__(self, frame):
        self.T = np.array(frame["ego2global_translation"][:2], dtype=np.float64)
        self.yaw = yaw_of(frame["ego2global_rotation"])


def lerp_pose(p0, p1, a):
    q = Pose.__new__(Pose)
    q.T = (1 - a) * p0.T + a * p1.T
    q.yaw = p0.yaw + a * wrap(p1.yaw - p0.yaw)
    return q


def rot(y):
    c, s = math.cos(y), math.sin(y)
    return np.array([[c, -s], [s, c]])


def ego_to_ego(xy, p_from, p_to):
    g = xy @ rot(p_from.yaw).T + p_from.T
    return (g - p_to.T) @ rot(p_to.yaw)


class Flow:
    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def field(self, A, B, scale=0.25):
        a = cv2.cvtColor(cv2.resize(A, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(cv2.resize(B, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        return self.dis.calc(a, b, None)


def interp_frame(A, B, flow, a):
    """A, B same size (BGR); flow at A's pixels (from a lower-scale estimate); returns the frame at fraction a in [0, 1]"""
    if a <= 1e-6:
        return A
    h, w = A.shape[:2]
    f = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) * (w / (flow.shape[1]))
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ma = cv2.remap(A, gx - a * f[..., 0], gy - a * f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mb = cv2.remap(B, gx + (1 - a) * f[..., 0], gy + (1 - a) * f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(ma, 1 - a, mb, a, 0)


class LogCache:
    def __init__(self, log):
        self.log = log
        fr = pickle.load(open(f"{LOGS}/{log}.pkl", "rb"))
        self.by_token = {f["token"]: f for f in fr}
        self.zips = {}

    def image(self, frame, cam, size):
        if cam not in self.zips:
            self.zips[cam] = zipfile.ZipFile(f"{DATA}/frames/{self.log}_{cam}.zip")
        name = os.path.basename(frame["cams"][cam]["data_path"])
        im = cv2.imdecode(np.frombuffer(self.zips[cam].read(name), np.uint8), cv2.IMREAD_COLOR)
        return cv2.resize(im, size, interpolation=cv2.INTER_AREA)


CAMS = (("CAM_F0", (1280, 720)), ("CAM_L0", (480, 270)), ("CAM_R0", (480, 270)), ("CAM_B0", (480, 270)))


def find_runs(exp, min_len=40):
    by = collections.defaultdict(list)
    for t, v in exp.items():
        by[v["log_name"]].append((v["timestamps_us"][-1], t))
    runs = []
    for log, lst in by.items():
        lst.sort()
        cur = [lst[0]]
        for a, b in zip(lst, lst[1:]):
            if abs((b[0] - a[0]) / 1e6 - 0.5) < 0.02:
                cur.append(b)
            else:
                if len(cur) >= min_len:
                    runs.append((log, [x[1] for x in cur]))
                cur = [b]
        if len(cur) >= min_len:
            runs.append((log, [x[1] for x in cur]))
    return runs


def run_score(exp, toks):
    cmds = [int(np.argmax(exp[t]["ego_statuses"][-1]["driving_command"])) for t in toks]
    spd = [float(np.hypot(*exp[t]["ego_statuses"][-1]["ego_velocity"])) for t in toks]
    moving = np.mean(np.array(spd) > 3.0)
    turns = np.mean(np.array(cmds) != 1)
    return float(moving + 2.0 * min(turns, 0.5)), cmds, spd


def plans_provider(mode, exp):
    if mode == "full":
        P = np.load(f"{DATA}/proptable/navtest_final/proposals.npz")
        idx = {str(t): i for i, t in enumerate(P["token"])}

        def get(tok):
            i = idx[tok]
            return P["proposals"][i].astype(np.float64), R.agg_score(P["logits"][i]), int(P["pick"][i])
        return get

    def mock(tok):
        hum = np.array(exp[tok]["human_future_poses"], dtype=np.float64)
        rng = np.random.default_rng(abs(hash(tok)) % (2 ** 32))
        prop = np.stack([hum + np.c_[rng.normal(0, 0.25 * (k % 7), 8).cumsum() * 0.1, rng.normal(0, 0.4, 8).cumsum() * 0.1 * (k % 5), np.zeros(8)] for k in range(64)])
        sc = rng.random(64)
        sc[0] = 1.0
        prop[0] = hum
        return prop, sc, 0
    return mock


def render_run(writer, log, toks, exp, get_plan, run_idx, n_runs, flow, still_at=None):
    lc = LogCache(log)
    frames = [lc.by_token[t] for t in toks]
    poses = [Pose(f) for f in frames]
    bev = R.Bev(BEVP)
    imgs_prev = None
    sink = []
    for k in range(len(toks) - 1):
        fk, fk1 = frames[k], frames[k + 1]
        prop, sc, pick = get_plan(toks[k])
        hum = np.array(exp[toks[k]]["human_future_poses"], dtype=np.float64)
        paths = [R.densify(prop[kk]) for kk in range(64)]
        hpath = R.densify(hum)
        nrm = (sc - sc.min()) / max(float(sc.max() - sc.min()), 1e-9)
        order = np.argsort(sc)
        cmd = int(np.argmax(fk["driving_command"]))
        v = float(np.hypot(*exp[toks[k]]["ego_statuses"][-1]["ego_velocity"]))
        A = {c: lc.image(fk, c, sz) for c, sz in CAMS}
        B = {c: lc.image(fk1, c, sz) for c, sz in CAMS}
        FL = {c: flow.field(A[c], B[c]) for c, _ in CAMS}
        cam_k = R.Cam(fk["cams"]["CAM_F0"], fk["lidar2ego_translation"], fk["lidar2ego_rotation"])
        tr1 = {str(t): i for i, t in enumerate(fk1["anns"]["track_tokens"])}
        for m in range(SUB):
            a = m / SUB
            pt = lerp_pose(poses[k], poses[k + 1], a)
            canvas = np.full((H, W, 3), 18, np.uint8)
            fx, fy, fw, fh = FRONT
            canvas[fy:fy + fh, fx:fx + fw] = interp_frame(A["CAM_F0"], B["CAM_F0"], FL["CAM_F0"], a)
            for i2, (c, label) in enumerate((("CAM_L0", "LEFT"), ("CAM_R0", "RIGHT"), ("CAM_B0", "REAR"))):
                canvas[R.BOTTOM_Y:R.BOTTOM_Y + 270, i2 * 480:(i2 + 1) * 480] = interp_frame(A[c], B[c], FL[c], a)
                R.text(canvas, f"{label} camera", (i2 * 480 + 10, R.BOTTOM_Y + 24), 0.6)
            R.text(canvas, "FRONT camera (2 Hz sensor frames, motion-interpolated to 10 fps)", (14, 30), 0.7)
            # BEV background
            bx, by, bw, bh = BEVP
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (28, 28, 30), -1)
            for r in range(10, 60, 10):
                cv2.line(canvas, bev.pt(r, -29), bev.pt(r, 29), (50, 50, 54), 1)
                R.text(canvas, f"{r} m", (bx + 6, bev.pt(r, 0)[1] - 3), 0.4, (120, 120, 125), 1, False)
            cv2.line(canvas, bev.pt(-8, 0), bev.pt(57, 0), (50, 50, 54), 1)
            colors = {"vehicle": (220, 140, 60), "pedestrian": (60, 160, 255), "bicycle": (80, 220, 220), "generic_object": (150, 150, 150)}
            for j, (b, nme, trk) in enumerate(zip(np.array(fk["anns"]["gt_boxes"]), fk["anns"]["gt_names"], fk["anns"]["track_tokens"])):
                g0 = np.array([b[:2]]) @ rot(poses[k].yaw).T + poses[k].T
                yaw_g = b[6] + poses[k].yaw
                j1 = tr1.get(str(trk))
                if j1 is not None:
                    b1 = np.array(fk1["anns"]["gt_boxes"][j1])
                    g1 = np.array([b1[:2]]) @ rot(poses[k + 1].yaw).T + poses[k + 1].T
                    g0 = (1 - a) * g0 + a * g1
                    yaw_g = yaw_g + a * wrap(b1[6] + poses[k + 1].yaw - yaw_g)
                e = (g0 - pt.T) @ rot(pt.yaw)
                ye = yaw_g - pt.yaw
                if -10 < e[0, 0] < 60 and abs(e[0, 1]) < 30:
                    cv2.fillPoly(canvas, [bev.poly(R.box_corners(e[0, 0], e[0, 1], ye, b[3], b[4]))], colors.get(str(nme), (150, 150, 150)), cv2.LINE_AA)
            R.text(canvas, "BEV (current ego frame); logged agents are not a model input", (bx + 8, by + 24), 0.5, (200, 200, 205))
            # overlays: the plan of token k, fixed in the world
            front = canvas[fy:fy + fh, fx:fx + fw]
            sc_ = fw / 1920.0
            tt = paths[0][1]
            allt = np.full(len(tt), 99.0)

            def to_now(xy):
                return ego_to_ego(xy, poses[k], pt)

            sel = to_now(paths[pick][0])
            ov = front.copy()
            R.draw_ribbon_cam(ov, cam_k, sel, allt, 99.0, (80, 220, 90), sc_)
            front[:] = cv2.addWeighted(ov, 0.30, front, 0.70, 0)
            for kk in order:
                if kk != pick:
                    R.draw_path_cam(front, cam_k, to_now(paths[kk][0]), allt, 99.0, R.turbo(0.15 + 0.85 * nrm[kk]), 2, sc_)
            hnow = to_now(hpath[0])
            R.draw_path_cam(front, cam_k, hnow, allt, 99.0, (255, 255, 255), 3, sc_)
            R.draw_path_cam(front, cam_k, sel, allt, 99.0, (0, 0, 0), 9, sc_)
            R.draw_path_cam(front, cam_k, sel, allt, 99.0, (90, 255, 90), 5, sc_)
            for j in range(8):
                p3 = np.array([[*to_now(prop[pick][j:j + 1, :2])[0], R.GROUND_Z]])
                uv, z = cam_k.project(p3)
                if z[0] > 2.0:
                    c = tuple((uv[0] * sc_).astype(int))
                    cv2.circle(front, c, 7, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(front, c, 5, (255, 255, 255), -1, cv2.LINE_AA)
            for kk in order:
                if kk != pick:
                    cv2.polylines(canvas, [bev.poly(to_now(paths[kk][0]))], False, R.turbo(0.15 + 0.85 * nrm[kk]), 1, cv2.LINE_AA)
            cv2.polylines(canvas, [bev.poly(hnow)], False, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.polylines(canvas, [bev.poly(sel)], False, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.polylines(canvas, [bev.poly(sel)], False, (90, 255, 90), 4, cv2.LINE_AA)
            off = R.EGO_L / 2 - R.EGO_REAR
            cv2.fillPoly(canvas, [bev.poly(R.box_corners(off, 0.0, 0.0, R.EGO_L, R.EGO_W))], (90, 255, 90), cv2.LINE_AA)
            cv2.polylines(canvas, [bev.poly(R.box_corners(off, 0.0, 0.0, R.EGO_L, R.EGO_W))], True, (255, 255, 255), 1, cv2.LINE_AA)
            ly = by + 650
            for j, (lab, col) in enumerate((("selected plan", (90, 255, 90)), ("human future", (255, 255, 255)))):
                cv2.line(canvas, (bx + 12, ly + 18 * j - 4), (bx + 40, ly + 18 * j - 4), col, 3, cv2.LINE_AA)
                R.text(canvas, lab, (bx + 48, ly + 18 * j), 0.45, (220, 220, 225), 1, False)
            for j in range(6):
                cv2.rectangle(canvas, (bx + 330 + j * 18, ly - 12), (bx + 346 + j * 18, ly + 2), R.turbo(0.15 + 0.85 * j / 5), -1)
            R.text(canvas, "hypotheses: predicted PDMS low -> high", (bx + 330, ly + 20), 0.4, (220, 220, 225), 1, False)
            ix, iy = 1440, 720
            cv2.rectangle(canvas, (ix, iy), (W, H), (24, 24, 28), -1)
            cv2.rectangle(canvas, (0, 990), (1440, H), (24, 24, 28), -1)
            R.text(canvas, "NAV COMMAND (agent input)", (ix + 14, iy + 28), 0.55, (200, 200, 205), 1, False)
            for j, nm in enumerate(CMD[:3]):
                on = cmd == j
                x0 = ix + 14 + j * 150
                cv2.rectangle(canvas, (x0, iy + 40), (x0 + 140, iy + 78), (90, 255, 90) if on else (60, 60, 66), -1 if on else 1)
                R.text(canvas, nm, (x0 + 10, iy + 66), 0.6, (0, 0, 0) if on else (160, 160, 165), 2 if on else 1, False)
            R.text(canvas, f"ego speed {v * 3.6:5.1f} km/h", (ix + 14, iy + 112), 0.6)
            R.text(canvas, f"{log[:10]}..{log[-11:]}", (ix + 14, iy + 136), 0.45, (150, 150, 155), 1, False)
            R.text(canvas, f"time in run {(k + a) * 0.5:6.1f} s   (plan #{k + 1}, replaced every 0.5 s)", (ix + 14, iy + 166), 0.5, (230, 230, 235), 1, False)
            R.text(canvas, f"selected: hypothesis #{pick}, predicted PDMS {100 * sc[pick]:5.1f}", (ix + 14, iy + 196), 0.55, (90, 255, 90))
            R.text(canvas, f"predicted range of the 64: {100 * sc.min():4.1f} .. {100 * sc.max():4.1f}", (ix + 14, iy + 224), 0.5, (200, 200, 205), 1, False)
            tr = TRUE.get(toks[k])
            if tr is not None:                       # the harness's own score of the EXECUTED plan (full-navtest CSV)
                bad = tr["score"] == 0.0
                R.text(canvas, f"NAVSIM PDMS of the executed plan: {100 * tr['score']:5.1f}", (ix + 14, iy + 256), 0.55,
                       (80, 80, 255) if bad else (230, 230, 235))
                R.text(canvas, "NC {no_at_fault_collisions:.0f}  DAC {drivable_area_compliance:.0f}  EP {ego_progress:.2f}  "
                       "TTC {time_to_collision_within_bound:.0f}  C {comfort:.0f}".format(**tr), (ix + 14, iy + 282), 0.5,
                       (80, 80, 255) if bad else (200, 200, 205), 1, False)
            R.text(canvas, f"run {run_idx + 1}/{n_runs}", (ix + 14, iy + 340), 0.55, (230, 230, 235))
            cv2.rectangle(canvas, (14, 1050), (1426, 1062), (60, 60, 66), 1)
            cv2.rectangle(canvas, (14, 1050), (14 + int(1412 * (k + a) / (len(toks) - 1)), 1062), (90, 255, 90), -1)
            R.text(canvas, "REFe final model  |  NAVSIM navtest  |  plan fixed in the world, camera advancing", (14, 1030), 0.55, (200, 200, 205), 1, False)
            if still_at is not None:
                sink.append(canvas)
                if len(sink) > still_at:
                    return sink[still_at]
            else:
                writer.stdin.write(canvas.tobytes())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", choices=("full", "human-mock"), default="full")
    ap.add_argument("--out", default="sync.mp4")
    ap.add_argument("--runs", default=None, help="LOG:START:LEN,... (token indices within the consecutive run list is not needed: START = index in the log's run)")
    ap.add_argument("--max-runs", type=int, default=6)
    ap.add_argument("--run-len", type=int, default=100, help="tokens per run (100 = 50 s)")
    ap.add_argument("--still", type=int, default=None)
    ap.add_argument("--still-out", default="sync_still.png")
    ap.add_argument("--test-run-len", type=int, default=12)
    a = ap.parse_args()
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    runs = find_runs(exp)
    scored = sorted(((run_score(exp, t)[0], log, t) for log, t in runs), key=lambda x: -x[0])
    chosen, seen = [], set()
    for s, log, t in scored:
        if log in seen:
            continue
        seen.add(log)
        mid = max(0, len(t) // 2 - a.run_len // 2)
        chosen.append((log, t[mid:mid + a.run_len]))
        if len(chosen) >= a.max_runs:
            break
    if a.plans == "full":
        P = np.load(f"{DATA}/proptable/navtest_final/proposals.npz")
        have = {str(t) for t in P["token"]}
        chosen = [(l, [t for t in ts]) for l, ts in chosen if all(t in have for t in ts)]
    get = plans_provider(a.plans, exp)
    flow = Flow()
    if a.still is not None:
        log, toks = chosen[0]
        fr = render_run(None, log, toks[:a.test_run_len], exp, get, 0, 1, flow, still_at=a.still)
        cv2.imwrite(a.still_out, fr)
        print("wrote", a.still_out, "run", log, len(toks))
        return 0
    cmd = [R.FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
    writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i, (log, toks) in enumerate(chosen):
        render_run(writer, log, toks, exp, get, i, len(chosen), flow)
        print(f"run {i + 1}/{len(chosen)} done: {log} ({len(toks)} tokens)", flush=True)
    writer.stdin.close()
    writer.wait()
    print("ZZSYNC_DONE", a.out, f"runs {len(chosen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
