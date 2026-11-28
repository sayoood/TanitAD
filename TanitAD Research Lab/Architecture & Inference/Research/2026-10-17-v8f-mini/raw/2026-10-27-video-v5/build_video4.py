"""build_video4.py -- assemble viz4/v8f_evidence_v4.mp4 (1920x1080, H.264, 10 fps) = the v3 video (build_video3.py's chapters 1-6 and the closing, unchanged except the header string, the title card and the closing card)
PLUS ONE NEW CHAPTER, "7 T1-kin: the model on its own motion" (v4_scenes.py), inserted before the closing (now chapter 8).   CPU only, ONE python process (no pool).
  python build_video4.py [--dry] [--qa-only] [--tags a,b] [--chapter7]   ;   end of a full run prints ZZVIDEO4-DONE, a refusal prints ZZVIDEO4-STOP <why>
--chapter7 renders only the chapter-7 scenes (no video) for CONTROLS_V4; the full run also writes data/onscreen_text.json, data/numreg.json and data/ch7_lines_digest.json (sha1 of the plotted arrays of every chapter-7 frame)."""
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
OUTV = f"{HERE}/v8f_evidence_v4.mp4"
QADIR = HERE
MIN_FREE = 0.5e9          # v4: C: has ~2 GB free and the video is ~10 MB; no frame sequence is ever written to C: (frames are piped to ffmpeg)


def schedule():
    J, chapters = [], []

    cur = {"chap": ""}

    def add(fn, args, reps, qa=None):
        J.append({"fn": fn, "args": args, "reps": int(max(1, round(reps))), "qa": qa, "chap": cur["chap"]})

    def hold(fn, args, sec, qa=None):
        add(fn, args, sec * FPS, qa)

    def sweep(fn, mk, hold_s, n=8, qa=None, step_s=0.1):
        for q in range(1, n + 1):
            last = q == n
            add(fn, mk(q / n), (hold_s * FPS) if last else step_s * FPS, qa if last else None)

    def chapter(name):
        cur["chap"] = name
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
    chapter("7 T1-kin: the model on its own motion")                                         # NEW in v4
    hold("t1k_intro", (), 14.0, "07a_intro")
    hold("t1k_rule", (), 14.0, "07b_rule")
    import v4_lib as V4
    for i, sc in enumerate(V4.SCN):
        tg = sc["tag"].replace("(", "").replace(")", "").replace(" ", "")
        add("t1k_frame", (i, 0), 0.6 * FPS, "07c%d_%s_start" % (i + 1, tg) if i in (0, 4) else None)
        for f in range(1, 30):
            add("t1k_frame", (i, f), 0.2 * FPS, "07c%d_%s_mid" % (i + 1, tg) if (f == 18 and i in (0, 3)) else None)
        add("t1k_frame", (i, 30), 5.0 * FPS, "07c%d_%s_end" % (i + 1, tg))
    hold("t1k_hist", (), 14.0, "07d_hist")
    hold("t1k_card", ("causal",), 14.0, "07e_card_causal")
    hold("t1k_card", ("strict",), 14.0, "07f_card_strict")
    hold("t1k_lat", (), 16.0, "07g_card_lateral")
    chapter("8 closing")
    hold("closing_card", (), 9.0, "08_closing")
    return J, chapters


def free_c():
    return shutil.disk_usage("C:/").free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--qa-only", action="store_true", help="render only the tagged frames (first of each tag), no video")
    ap.add_argument("--tags", default="")
    ap.add_argument("--chapter7", action="store_true", help="render only the chapter-7 scenes (no video) for CONTROLS_V4")
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
        print(f"ZZVIDEO4-STOP C: free {free_c() / 1e9:.2f} GB < 3 GB before the render")
        raise SystemExit(2)
    import numpy as np  # noqa: F401
    from PIL import Image
    import v3_scenes as V
    import v4_scenes as V7                       # registers the chapter-7 scenes in V.SCENES
    import v3_lib as L
    import v3_draw as DR
    import hashlib
    want = set(t for t in a.tags.split(",") if t)
    texts, numreg, digest = [], [], {}
    tmp = OUTV + ".tmp.mp4"
    ff = None
    if not a.qa_only and not a.chapter7:
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(FPS), "-i", "-",
                               "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp], stdin=subprocess.PIPE)
    t0 = time.time()
    try:
        for n, j in enumerate(J):
            if a.qa_only and not (j["qa"] and (not want or j["qa"] in want)):
                continue
            if a.chapter7 and not j["chap"].startswith("7"):
                continue
            if n % 20 == 0 and free_c() < MIN_FREE:
                raise RuntimeError(f"C: free {free_c() / 1e9:.2f} GB < 3 GB at frame {n}")
            key = f"{j['fn']}{tuple(round(x, 4) if isinstance(x, float) else x for x in j['args'])}"
            L.CUR["scene"] = key
            n0 = len(L.REG)
            img = V.SCENES[j["fn"]](*j["args"])
            texts.append({"scene": key, "fn": j["fn"], "chap": j["chap"], "texts": list(DR.LAST_TEXTS)})
            if j["chap"].startswith("7"):
                hh = hashlib.sha1()
                for gid, x_, y_ in V7.LAST_LINES:
                    hh.update(gid.encode() + np.ascontiguousarray(x_, np.float64).tobytes() + np.ascontiguousarray(y_, np.float64).tobytes())
                digest[key] = {"sha1": hh.hexdigest(), "n_lines": len(V7.LAST_LINES)}
            numreg.extend(L.REG[n0:])
            if ff is not None:
                buf = img.tobytes()
                for _ in range(j["reps"]):
                    ff.stdin.write(buf)
            if j["qa"] and (a.qa_only or not a.chapter7) and (a.qa_only or j["qa"].startswith(("07", "08", "01"))):
                Image.fromarray(img).save(f"{QADIR}/qa_ch7_{j['qa']}.png" if j["qa"].startswith("07") else f"{QADIR}/qa_{j['qa']}.png")
            if n % 10 == 0:
                print(f"{n}/{len(J)} {time.time() - t0:.0f}s", flush=True)
    except BaseException as e:
        if ff is not None:
            ff.kill()
        print(f"ZZVIDEO4-STOP {type(e).__name__}: {e}", flush=True)
        raise
    sfx = "_ch7" if a.chapter7 else "_qa" if a.qa_only else ""
    json.dump(texts, open(f"{HERE}/data/onscreen_text{sfx}.json", "w"))
    json.dump(numreg, open(f"{HERE}/data/numreg{sfx}.json", "w"))
    json.dump(digest, open(f"{HERE}/data/ch7_lines_digest{sfx}.json", "w"))
    if ff is not None:
        ff.stdin.close()
        rc = ff.wait()
        if rc != 0:
            print(f"ZZVIDEO4-STOP ffmpeg rc={rc}; no output written")
            raise SystemExit(3)
        os.replace(tmp, OUTV)
        print("wrote", OUTV, f"{os.path.getsize(OUTV) / 1e6:.1f} MB", f"{time.time() - t0:.0f}s", flush=True)
    print(f"registered numbers: {len(numreg)}; drawn-text frames: {len(texts)}")
    print("ZZVIDEO4-DONE", flush=True)


if __name__ == "__main__":
    main()
