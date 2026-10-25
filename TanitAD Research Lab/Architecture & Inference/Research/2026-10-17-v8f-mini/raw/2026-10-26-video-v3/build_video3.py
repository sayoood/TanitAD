"""build_video3.py -- assemble viz3/v8f_evidence_v3.mp4 (1920x1080, H.264, 10 fps) from the scenes of v3_scenes.py.   CPU only, ONE python process (no pool).
  python build_video3.py [--dry] [--qa-only] [--tags a,b]   ;   end of a full run prints ZZVIDEO3-DONE, a refusal prints ZZVIDEO3-STOP <why>
Also writes data/onscreen_text.json (every drawn string of every unique frame) and data/numreg.json (every registered on-screen number) for CONTROLS_V3."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

FPS = 10
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
OUTV = f"{HERE}/v8f_evidence_v3.mp4"
QADIR = f"{HERE}/_qa_tmp"
MIN_FREE = 3.0e9


def schedule():
    J, chapters = [], []

    def add(fn, args, reps, qa=None):
        J.append({"fn": fn, "args": args, "reps": int(max(1, round(reps))), "qa": qa})

    def hold(fn, args, sec, qa=None):
        add(fn, args, sec * FPS, qa)

    def sweep(fn, mk, hold_s, n=8, qa=None, step_s=0.1):
        for q in range(1, n + 1):
            last = q == n
            add(fn, mk(q / n), (hold_s * FPS) if last else step_s * FPS, qa if last else None)

    def chapter(name):
        chapters.append({"name": name, "frame": sum(j["reps"] for j in J)})

    chapter("1 title")
    hold("title_card", (), 8.0, "01_title")
    chapter("2 who listens to what")
    for st, sec, qa in ((0, 3.5, "02a_listen_howto"), (1, 5.0, "02b_listen_S"), (2, 6.0, "02c_listen_Tlat"), (3, 5.5, "02d_listen_O"), (4, 3.5, "02e_listen_navpath")):
        hold("listen_frame", (st,), sec, qa)
    sweep("qnav_frame", lambda p: (p,), 4.8, qa="02f_qnav")
    chapter("3 nav counterfactual")
    hold("method_card", (), 4.5, "03a_method")
    for name, tag in (("JUNCTION", "03b_junction"), ("NO-TURN", "03d_noturn")):
        for k in range(3):
            sweep("cf_all", lambda p, k=k, name=name: (name, k, p), 1.2)
        hold("cf_all", (name, 3, 1.0), 4.5, f"{tag}_all")
        sweep("cf_bcd", lambda p, name=name: (name, p), 5.7, qa=f"{tag}_bcd")
    for name, tag in (("CURVE", "03e_curve"), ("FAILURE(worst)", "03f_failure")):
        sweep("cf_four", lambda p, name=name: (name, p), 5.2, qa=tag)
    hold("cf_summary", (), 8.0, "03g_summary")
    chapter("4 better plans, with HACC")
    for name, tag in (("JUNCTION", "04a_plans_junction"), ("CURVE", "04b_plans_curve"), ("NO-TURN", "04c_plans_noturn"), ("FAILURE(worst)", "04d_plans_failure")):
        sweep("plans_frame", lambda p, name=name: (name, p), 4.5, qa=tag)
    hold("plans_table_card", (), 9.0, "04e_plans_table")
    hold("plans_gap_card", (), 10.0, "04f_plans_gaps")
    chapter("5 dominance over the kinematic prior")
    sweep("dom_table", lambda p: (p,), 11.0, n=15, qa="05a_dom_table", step_s=0.3)
    sweep("dom_accel", lambda p: (p,), 11.0, n=10, qa="05b_dom_accel", step_s=0.2)
    hold("dom_mech", (), 12.0, "05c_dom_mech")
    hold("dom_caveats", (), 14.0, "05d_dom_caveats")
    chapter("6 strategic layer at its own tick")
    for name, tag in (("JUNCTION", "06a_S_junction"), ("NO-TURN", "06b_S_noturn")):
        sweep("s_frame", lambda p, name=name: (name, p), 4.2, qa=tag)
    hold("s_card", (), 8.5, "06c_S_card")
    chapter("7 closing")
    hold("closing_card", (), 9.0, "07_closing")
    return J, chapters


def free_c():
    return shutil.disk_usage("C:/").free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--qa-only", action="store_true", help="render only the tagged frames (first of each tag), no video")
    ap.add_argument("--tags", default="")
    a = ap.parse_args()
    J, ch = schedule()
    nfr = sum(j["reps"] for j in J)
    print(f"{len(J)} unique frames, {nfr} video frames = {nfr / FPS:.1f} s", flush=True)
    for c in ch:
        print("  chapter", c["name"], f"t = {c['frame'] / FPS:.1f} s")
    json.dump({"fps": FPS, "chapters": [{"name": c["name"], "t_s": c["frame"] / FPS} for c in ch], "n_frames": nfr, "duration_s": nfr / FPS, "unique_frames": len(J)},
              open(f"{HERE}/video_chapters.json", "w"), indent=1)
    if a.dry:
        return
    if free_c() < MIN_FREE:
        print(f"ZZVIDEO3-STOP C: free {free_c() / 1e9:.2f} GB < 3 GB before the render")
        raise SystemExit(2)
    import numpy as np  # noqa: F401
    from PIL import Image
    import v3_scenes as V
    import v3_lib as L
    import v3_draw as DR
    os.makedirs(QADIR, exist_ok=True)
    want = set(t for t in a.tags.split(",") if t)
    texts, numreg = [], []
    tmp = OUTV + ".tmp.mp4"
    ff = None
    if not a.qa_only:
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(FPS), "-i", "-",
                               "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp], stdin=subprocess.PIPE)
    t0 = time.time()
    try:
        for n, j in enumerate(J):
            if a.qa_only and not (j["qa"] and (not want or j["qa"] in want)):
                continue
            if n % 20 == 0 and free_c() < MIN_FREE:
                raise RuntimeError(f"C: free {free_c() / 1e9:.2f} GB < 3 GB at frame {n}")
            key = f"{j['fn']}{tuple(round(x, 4) if isinstance(x, float) else x for x in j['args'])}"
            L.CUR["scene"] = key
            n0 = len(L.REG)
            img = V.SCENES[j["fn"]](*j["args"])
            texts.append({"scene": key, "fn": j["fn"], "texts": list(DR.LAST_TEXTS)})
            numreg.extend(L.REG[n0:])
            if ff is not None:
                buf = img.tobytes()
                for _ in range(j["reps"]):
                    ff.stdin.write(buf)
            if j["qa"]:
                Image.fromarray(img).save(f"{QADIR}/qa_{j['qa']}.png")
            if n % 10 == 0:
                print(f"{n}/{len(J)} {time.time() - t0:.0f}s", flush=True)
    except BaseException as e:
        if ff is not None:
            ff.kill()
        print(f"ZZVIDEO3-STOP {type(e).__name__}: {e}", flush=True)
        raise
    json.dump(texts, open(f"{HERE}/data/onscreen_text{'_qa' if a.qa_only else ''}.json", "w"))
    json.dump(numreg, open(f"{HERE}/data/numreg{'_qa' if a.qa_only else ''}.json", "w"))
    if ff is not None:
        ff.stdin.close()
        rc = ff.wait()
        if rc != 0:
            print(f"ZZVIDEO3-STOP ffmpeg rc={rc}; no output written")
            raise SystemExit(3)
        os.replace(tmp, OUTV)
        print("wrote", OUTV, f"{os.path.getsize(OUTV) / 1e6:.1f} MB", f"{time.time() - t0:.0f}s", flush=True)
    print(f"registered numbers: {len(numreg)}; drawn-text frames: {len(texts)}")
    print("ZZVIDEO3-DONE", flush=True)


if __name__ == "__main__":
    main()
