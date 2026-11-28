"""make_reel_r8.py -- stitch the rule-picked episode videos into ONE reel with title cards (CPU; ffmpeg concat, stream copy).

    python make_reel_r8.py --picks <raw>/PICKS_step40000ema.json --videos-dir E:/refcv8_eval_kit/videos --out E:/refcv8_eval_kit/videos/refcv8_ema40k_reel.mp4

Cards are encoded with the SAME encoder settings as the episode videos (render_r8_video.open_ffmpeg), so the concat demuxer can stream-copy.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_r8_video as R  # noqa: E402

NAMES = {"cruise_false_brake": "cruise false-braking", "sharp_turn": "sharp turn", "stop_slowdown": "stop / slow-down"}
RULE = {
    "cruise_false_brake": "Rule R1: most windows where GT cruises (|v6-v0| < 1.5 m/s) but the plan decelerates",
    "sharp_turn": "Rule R2: largest heading change over any 3 s of the logged path, inside the plan windows",
    "stop_slowdown": "Rule R3: logged speed falls from >= 5 to < 1 m/s; largest drop inside the plan-window coverage",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", required=True)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=40000)
    ap.add_argument("--heads", action="store_true",
                    help="use the *_HEADS.mp4 episode videos (rendered with --heads-dir) and say so on the cards")
    a = ap.parse_args()
    sfx = "_HEADS" if a.heads else ""
    pk = json.load(open(a.picks, encoding="utf-8"))
    vd = Path(a.videos_dir)
    tmp = vd / "_cache" / ("reel_parts" + sfx)
    tmp.mkdir(parents=True, exist_ok=True)
    parts = []
    intro = R.title_card([f"refcv8  EMA step {a.step:,}  -  judgement reel",
                          "3 full eval139 episodes, picked BY RULE from banked metrics (never by how the plan looks)",
                          "T1 self-action OPEN LOOP on logged frames: a diagnostic, not driving performance"],
                         ["panels: front camera (cylindrical) | metric BEV | GT map | tactical decoder | metrics + speed plot",
                          "GT green, refcv8 orange; shipped re-timed plan = 30k EMA layer (cyan); refcv7-50,400 violet dashed; causal prior grey",
                          "10 Hz native, one plan snapshot every 5th frame (held between); strategic OFF (--no-strategic)",
                          ("refcv8 MAP-head (right half of the MAP panel) and BOX-head (orange outlines, p>=0.20)" if a.heads else "map-head / box-head outputs: pending dump"),
                          ("from a CPU head dump; box-head presence never exceeds 0.40 here (display threshold, not a detector)" if a.heads else "(see the _HEADS reel for the head layers)")])
    p = tmp / "00_intro.mp4"
    R.write_card(str(p), intro, 5.0)
    parts.append(p)
    order = ["cruise_false_brake", "sharp_turn", "stop_slowdown"]
    for k, key in enumerate(order, 1):
        ci = pk["picks"][key]
        st = pk["pick_stats"][key]
        extra = []
        if key == "cruise_false_brake":
            extra = [f"{st['n_flagged']} of {st['n_gt_cruising']} GT-cruising windows flagged as false-brake"]
        elif key == "sharp_turn":
            extra = [f"max heading change {st['max_abs_dyaw_3s_rad']:.2f} rad over 3 s",
                     "NOTE: the nav-command input is constant per episode (all 139 episodes):",
                     "here TURN_R for the whole clip, while the first manoeuvre (t = 1-4 s) is a LEFT turn"]
        else:
            extra = [f"logged speed drop {st['stop_drop_mps']:.1f} m/s to standstill"]
        card = R.title_card([f"{k}/3   episode {ci}: {NAMES[key]}", RULE[key]], extra)
        pc = tmp / f"{k:02d}_card.mp4"
        R.write_card(str(pc), card, 4.0)
        parts.append(pc)
        parts.append(vd / f"refcv8_ema{a.step // 1000}k_ep{ci}_{key}{sfx}.mp4")
    end = R.title_card(["end of reel"], ["definitions: ADE = mean slot error vs logged GT path; false-brake = GT cruising AND plan decel (cruise_brake_retime_r8.py)",
                                         ("map-head / box-head panels: CPU head dump (dump_heads_r8.py), deterministic functions of the frames" if a.heads else "map-head / box-head panels fill once dump_heads_r8.py has run on the GPU")])
    pe = tmp / "99_end.mp4"
    R.write_card(str(pe), end, 3.0)
    parts.append(pe)
    lst = tmp / "concat.txt"
    lst.write_text("".join(f"file '{Path(x).as_posix()}'\n" for x in parts), encoding="utf-8")
    cmd = [R.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", a.out]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"ffmpeg concat exited {rc}")
    print(json.dumps({"reel": a.out, "parts": [str(x) for x in parts]}, indent=1))


if __name__ == "__main__":
    main()
