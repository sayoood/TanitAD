"""build_video.py -- assemble OUT/v8f_evidence_v2.mp4 (1920x1080, H.264, 10 fps) from the scenes of v2_scenes.py.   CPU only.
  python build_video.py [--workers 4] [--dry]"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FPS = 10
OUTV = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/viz2/v8f_evidence_v2.mp4"
OUTD = os.path.dirname(OUTV)


def schedule():
    J, chapters = [], []

    def add(fn, args, reps, qa=None):
        J.append({"fn": fn, "args": args, "reps": int(max(1, round(reps))), "qa": qa})

    def hold(fn, args, sec, qa=None):
        add(fn, args, sec * FPS, qa)

    def sweep(fn, mk, hold_s, n=8, qa=None):
        for q in range(1, n + 1):
            last = q == n
            add(fn, mk(q / n), (hold_s * FPS) if last else 1, qa if last else None)

    def chapter(name):
        chapters.append({"name": name, "frame": sum(j["reps"] for j in J)})

    chapter("1 title")
    hold("title_card", (), 6.5, "01_title")
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
    chapter("4 better plans")
    for name, tag in (("JUNCTION", "04a_plans_junction"), ("CURVE", "04b_plans_curve"), ("NO-TURN", "04c_plans_noturn"), ("FAILURE(worst)", "04d_plans_failure")):
        sweep("plans_frame", lambda p, name=name: (name, p), 3.2, qa=tag)
    hold("plans_table_card", (), 7.0, "04e_plans_table")
    hold("plans_gap_card", (), 7.0, "04f_plans_gaps")
    chapter("5 strategic layer at its own tick")
    for name, tag in (("JUNCTION", "05a_S_junction"), ("NO-TURN", "05b_S_noturn")):
        sweep("s_frame", lambda p, name=name: (name, p), 4.2, qa=tag)
    hold("s_card", (), 7.5, "05c_S_card")
    chapter("6 closing")
    hold("closing_card", (), 6.5, "06_closing")
    return J, chapters


def _job(j):
    import v2_scenes as V
    from PIL import Image  # noqa: F401
    return V.SCENES[j["fn"]](*j["args"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    J, ch = schedule()
    nfr = sum(j["reps"] for j in J)
    print(f"{len(J)} unique frames, {nfr} video frames = {nfr / FPS:.1f} s", flush=True)
    for c in ch:
        print("  chapter", c["name"], f"t = {c['frame'] / FPS:.1f} s")
    json.dump({"fps": FPS, "chapters": [{"name": c["name"], "t_s": c["frame"] / FPS} for c in ch], "n_frames": nfr, "duration_s": nfr / FPS,
               "unique_frames": len(J)}, open(f"{OUTD}/video_chapters.json", "w"), indent=1)
    if a.dry:
        return
    import shutil
    free = shutil.disk_usage("C:/").free
    if free < 1.5e9:
        raise SystemExit(f"free space on C: is {free / 1e9:.2f} GB < 1.5 GB: stopping")
    from multiprocessing import Pool
    from PIL import Image
    tmp = OUTV + ".tmp.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp], stdin=subprocess.PIPE)
    t0 = time.time()
    with Pool(a.workers) as p:
        for n, img in enumerate(p.imap(_job, J, chunksize=1)):
            buf = img.tobytes()
            for _ in range(J[n]["reps"]):
                ff.stdin.write(buf)
            if J[n]["qa"]:
                Image.fromarray(img).save(f"{OUTD}/qa_{J[n]['qa']}.png")
            if n % 25 == 0:
                print(f"{n}/{len(J)} {time.time() - t0:.0f}s", flush=True)
    ff.stdin.close()
    rc = ff.wait()
    if rc != 0:
        raise SystemExit(f"ffmpeg failed rc={rc}; no output written")
    os.replace(tmp, OUTV)
    print("wrote", OUTV, f"{os.path.getsize(OUTV) / 1e6:.1f} MB", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
