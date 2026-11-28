"""render_video.py -- the hierarchy-attention video and the key-frame figure.  CPU, matplotlib/Agg + ffmpeg.

  python render_video.py frame --clip 0 --i 30 --png test.png     one video frame (QA)
  python render_video.py keyfig [--seed M2b_s0]                    fig_attention_levels.png  (the 4 banked eval windows of the selection rule)
  python render_video.py video  [--workers 4] [--fps 4]            v8f_hierarchy_attention.mp4   (needs viz_data.py first)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from viz_lib import *  # noqa: E402,F401,F403

SCRATCH = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/viz"
WORK, OUT = f"{SCRATCH}/work", f"{SCRATCH}/out"
VIZ_DIR = f"{WORK}/viz"
REBUILD = "C:/Users/Admin/tanitad-data/refav1-eval141-256rb"
SEL = json.load(open(f"{OUT}/selection.json"))
ORDER = [("JUNCTION", "JUNCTION  (median own-code ADE6 of its stratum)"), ("CURVE", "CURVE  (median own-code ADE6 of its stratum)"),
         ("NO-TURN", "NO-TURN  (median own-code ADE6 of its stratum)"), ("FAILURE(worst)", "FAILURE CASE  (single worst own-code ADE6 of all 2,399 windows)")]
RULE = ("Selection rule (declared before rendering): per stratum the lower-median own-code ADE6 window of seed 0 (banked eval, 141 episodes / 2,399 windows); "
        "plus the single worst window; the WHOLE clip is rendered; nothing swapped after viewing.")
_CACHE: dict = {}


# ------------------------------------------------------------------------------------------------ data access
def clip_key(k):
    return SEL[ORDER[k][0]]["clip"]


def load_clip(seed, clip):
    key = (seed, clip)
    if key not in _CACHE:
        import torch
        import torchvision
        z = dict(np.load(f"{VIZ_DIR}/{seed}__{clip}.npz"))
        o = torch.load(f"{REBUILD}/{clip}.v2ep.pt", map_location="cpu", weights_only=False)
        buf, lens = o["jpeg_buf"], o["jpeg_len"]
        offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
        frames = {}
        need = set()
        for t, a_t, a_s in zip(z["t"], z["a_t"], z["a_s"]):
            need.update((int(t), int(t - a_t), int(t - a_s)))
        for j in sorted(need):
            frames[j] = torchvision.io.decode_png(buf[offs[2 * j]:offs[2 * j + 1]].clone()).permute(1, 2, 0).numpy()
        ext = bev_extent(z["gt"], np.concatenate([z["traj30"], np.nan_to_num(z["traj30_oracle"])]), z["ck_now"])
        _CACHE[key] = (z, frames, ext)
    return _CACHE[key]


def row_of(z, i):
    return {k: v[i] for k, v in z.items()}


# ------------------------------------------------------------------------------------------------ one video frame
def render_frame(k: int, i: int, seed="M2b_s0", W=1920, H=1080):
    clip = clip_key(k)
    z, frames, ext = load_clip(seed, clip)
    r = row_of(z, i)
    t, a_t, a_s = int(r["t"]), int(r["a_t"]), int(r["a_s"])
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    name, desc = ORDER[k]
    sel = SEL[name]
    # title
    fig.text(10 / W, 1 - 14 / H, "TanitAD v8F-mini  |  arm M2b, training seed 0 (M2b_s0)  |  T0 OPEN-LOOP: the true past is fed, nothing the model outputs feeds back  |  QUALITATIVE, 1 clip",
             fontsize=11.5, fontweight="bold", va="center")
    fig.text(10 / W, 1 - 34 / H, f"clip {clip[:8]}  |  {desc}  |  t = {0.2 * t:4.1f} s of {0.2 * 100:.0f} s  |  playback 0.8x real time",
             fontsize=9.5, color=MUTED, va="center")
    # top row: the three levels on the frame each last ticked at
    tick = {"S": t - a_s, "T": t - a_t, "O": t}
    age = {"S": a_s, "T": a_t, "O": 0}
    for n, lv in enumerate(("S", "T", "O")):
        x0 = 640 * n
        sc = r[f"scene_{lv}"]
        ax = axpx(fig, x0 + 6, 96, 628, 252, W, H)
        draw_overlay(ax, frames[tick[lv]], sc[-1].reshape(4, 10), lv, n_annot=3, total_mass=1.0)
        cm = float(r[f"cond_{lv}"].sum())
        fig.text((x0 + 8) / W, 1 - 62 / H, LV_TITLE[lv], fontsize=12.5, fontweight="bold", color=LV_COL[lv], va="center")
        tk = "reads the frame at NOW" if lv == "O" else f"last ticked {0.2 * age[lv]:.1f} s ago  (frame t-{age[lv]}), held since"
        fig.text((x0 + 8) / W, 1 - 82 / H, f"{tk}  |  tokens: 4 history frames x 40", fontsize=8.8, color=MUTED, va="center")
        mx = float(sc[-1].max())
        fig.text((x0 + 8) / W, 1 - 358 / H,
                 f"overlay: attention on the 40 tick-frame tokens (cell = 64x64 px; % = share of ALL its attention)\n"
                 f"scene {100 * (1 - cm):.0f}% | cond tokens {100 * cm:.0f}% | tick frame {100 * sc[-1].sum():.0f}% | 3 older frames {100 * sc[:-1].sum():.0f}% (not drawn)",
                 fontsize=8.1, color=INK, va="top", linespacing=1.45)
    # bottom-left BEV
    bx, by, bw, bh = 10, 412, 712, 574
    info = (f"speed {r['v']:.1f} m/s ({3.6 * r['v']:.0f} km/h)  |  nav input: {['FOLLOW', 'TURN_L', 'TURN_R'][int(np.argmax(r['nav'][:3]))]}"
            + (f" in {100 * r['nav'][3]:.0f} m" if r["nav"][4] > 0.5 else "")
            + f"  |  limit {'%d km/h' % (30, 50, 100, 120)[int(np.argmax(r['limit']))] if r['limit_ok'] else 'n/a'}")
    fig.text((bx + 4) / W, 1 - 402 / H, "BEV at NOW  (6 s ahead)   " + info, fontsize=9.2, fontweight="bold", va="center")
    abev = axpx(fig, bx + 40, by + 8, bw - 52, bh - 84, W, H)
    draw_bev(abev, r, ext)
    oracle = f"   |   T0-oracle (GT code) ADE6 {r['ade6_oracle']:.2f} m" if np.isfinite(r["ade6_oracle"]) else ""
    fig.text((bx + 4) / W, 1 - (by + bh - 22) / H, f"O own-code ADE6 {r['ade6']:.2f} m (video tick schedule){oracle}", fontsize=9.5, color=LV_COL["O"], fontweight="bold", va="center")
    if t == sel["t"]:
        fig.text((bx + 4) / W, 1 - (by + bh - 2) / H, f"<< SELECTED EVAL WINDOW #{sel['window']}: banked ADE6 seed 0 = {sel['ade6_seed0_banked']:.2f} m, seed 1 = {sel['ade6_seed1_banked']:.2f} m",
                 fontsize=8.8, color="#b30000", fontweight="bold", va="center")
    # middle: T's code
    draw_code(fig, (736, 412, 528, 574), r, W, H)
    # right: attention bars
    draw_cond_bars(fig, [(1274, 426, 640, 184), (1274, 616, 640, 184), (1274, 806, 640, 184)], r, W, H)
    # timeline
    at = axpx(fig, 50, 1008, 1840, 30, W, H)
    ts = 0.2 * z["t"]
    at.plot(ts, z["ade6"], color=LV_COL["O"], lw=1.8)
    if np.isfinite(z["ade6_oracle"]).any():
        at.plot(ts, z["ade6_oracle"], "--", color="#009E73", lw=1.2)
    at.axvline(0.2 * t, color=INK, lw=1.6)
    at.axvline(0.2 * sel["t"], color="#b30000", lw=1.0, ls=":")
    at.set_xlim(ts[0], ts[-1])
    at.set_ylim(0, max(1.0, float(np.nanmax(z["ade6"])) * 1.1))
    at.tick_params(labelsize=7, colors=MUTED, length=2)
    at.set_ylabel("ADE6 (m)", fontsize=7.5, color=MUTED)
    at.set_xlabel("")
    for s_ in ("top", "right"):
        at.spines[s_].set_visible(False)
    fig.text(52 / W, 1 - 998 / H, "own-code ADE6 per step (orange)   T0-oracle-code (green dashed)   selected eval window (red dotted)   cursor = now", fontsize=7.4, color=MUTED, va="center")
    fig.text(10 / W, 1 - 1060 / H, RULE, fontsize=7.4, color=MUTED, va="center")
    fig.text(10 / W, 1 - 1073 / H, "Attention = mean over 4 heads, 4 queries and the Reader's layers. Attention is NOT an explanation (cond.mean is also added to every query; a token can be a sink).",
             fontsize=7.4, color=MUTED, va="center")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def render_card(lines, W=1920, H=1080):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="#10141a")
    y = 0.80
    for txt, fs, col, bold in lines:
        fig.text(0.06, y, txt, fontsize=fs, color=col, fontweight="bold" if bold else "normal", va="top", wrap=True)
        y -= 0.012 + fs * 0.0021 * (txt.count("\n") + 1) * 1.55
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def cards(seed):
    s = json.load(open(f"{OUT}/fig_attention_stats.json"))["seeds"][seed]
    f = lambda lv, k: 100 * s[lv][k]["mean"][0]
    title = [("TanitAD v8F-mini: what each level of the hierarchy looks at", 30, "#ffffff", True),
             ("arm M2b (seed 0)   |   tier T0: OPEN-LOOP, teacher-forced on the true past   |   QUALITATIVE: one clip per row, attention is not an explanation", 14, "#aeb7c4", False),
             ("S strategic (1 Hz)  ->  T tactical (2 Hz)  ->  O operative (every step), each a learned-query cross-attention over  [cond tokens ; 4 frames x 40 DINOv3 tokens].\n"
              "Each panel is drawn on the frame its level LAST TICKED on (sample-and-hold).", 14, "#e8ecf2", False),
             ("Colour = level identity:   S purple     T blue     O orange.   Cell opacity = attention / max cell of that panel.", 14, "#e8ecf2", False),
             ("Selection rule (declared before rendering): per stratum the lower-median own-code ADE6 window (seed 0, banked eval: 141 episodes, 2,399 windows);\n"
              "plus the single worst window; the whole clip is rendered; no clip was swapped after viewing.", 13, "#aeb7c4", False),
             (f"Aggregate over ALL 2,399 windows (seed 0):  attention on scene tokens   S {100 - f('S', 'cond_mass'):.0f}%   T {100 - f('T', 'cond_mass'):.0f}%   O {100 - f('O', 'cond_mass'):.0f}%;   "
              f"effective tokens attended   S {f('S', 'effective_tokens_exp_entropy') / 100 * 1:.1f}   T {s['T']['effective_tokens_exp_entropy']['mean'][0]:.1f}   O {s['O']['effective_tokens_exp_entropy']['mean'][0]:.1f}  (of 160).", 13, "#e8ecf2", False)]
    return title


# ------------------------------------------------------------------------------------------------ key-frame figure (the BANKED eval windows, eval's own cycle ages)
def keyfig(seed="M2b_s0"):
    import torch
    import glob
    from attn_extract import EvalWorld, load_model
    model, cfg = load_model(seed)
    ew = EvalWorld(cfg, lru=2, cache=f"{WORK}/cache_viz")
    fig = plt.figure(figsize=(19.2, 12.9), dpi=100, facecolor=BG)
    W, H = 1920, 1290
    fig.text(10 / W, 1 - 18 / H, f"Hierarchy attention at the 4 pre-declared eval windows  |  arm M2b ({seed})  |  T0 open-loop  |  QUALITATIVE", fontsize=13, fontweight="bold", va="center")
    fig.text(10 / W, 1 - 40 / H, RULE, fontsize=8.5, color=MUTED, va="center")
    for k, (name, desc) in enumerate(ORDER):
        sel = SEL[name]
        w, clip, t = sel["window"], sel["clip"], sel["t"]
        ch = np.load(f"{WORK}/stats/chunks/{clip}.npz")
        j = int(np.nonzero(ch["win"] == w)[0][0])
        a_t, a_s = int(ch[f"{seed}__a_t"][j]), int(ch[f"{seed}__a_s"][j])
        rr = {"gt": ew.X["gt"][w], "traj30": ch[f"{seed}__traj30"][j], "ck_now": ch[f"{seed}__ck_now"][j], "traj30_oracle": None}
        z, frames, ext = load_clip(seed, clip) if os.path.exists(f"{VIZ_DIR}/{seed}__{clip}.npz") else (None, None, None)
        import torchvision
        o = torch.load(f"{REBUILD}/{clip}.v2ep.pt", map_location="cpu", weights_only=False)
        buf, lens = o["jpeg_buf"], o["jpeg_len"]
        offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
        fr = lambda jj: torchvision.io.decode_png(buf[offs[2 * jj]:offs[2 * jj + 1]].clone()).permute(1, 2, 0).numpy()
        tick = {"S": t - a_s, "T": t - a_t, "O": t}
        y0 = 62 + k * 306
        ncd = {"S": 3, "T": 4, "O": 6}
        for n, lv in enumerate(("S", "T", "O")):
            keys = sorted(kk for kk in ch.files if kk.startswith(f"{seed}__att_{lv}"))
            a = np.mean([ch[kk][j].mean(axis=(0, 1)) for kk in keys], 0)           # [S] mean over heads, queries, layers
            sc = a[ncd[lv]:].reshape(4, 40)
            ax = axpx(fig, 8 + 500 * n, y0 + 26, 492, 197, W, H)
            draw_overlay(ax, fr(tick[lv]), sc[-1].reshape(4, 10), lv, n_annot=3)
            ag = {"S": a_s, "T": a_t, "O": 0}[lv]
            fig.text((8 + 500 * n) / W, 1 - (y0 + 13) / H, f"{lv}  frame t-{ag}  ({0.2 * ag:.1f} s old)   scene {100 * sc.sum():.0f}% | cond {100 * a[:ncd[lv]].sum():.0f}% | tick frame {100 * sc[-1].sum():.0f}%",
                     fontsize=8.6, fontweight="bold", color=LV_COL[lv], va="center")
        ab = axpx(fig, 1540, y0 + 14, 370, 262, W, H)
        draw_bev(ab, rr, bev_extent(rr["gt"][None], rr["traj30"][None], rr["ck_now"][None]), show_oracle=False)
        ab.legend().remove()
        fig.text(10 / W, 1 - (y0 + 238) / H,
                 f"{desc}  |  banked eval window #{w}, clip {clip[:8]}, t = {0.2 * t:.1f} s, ages a_T={a_t}, a_S={a_s} (eval cycles them)  |  own-code ADE6: seed 0 {sel['ade6_seed0_banked']:.2f} m, seed 1 {sel['ade6_seed1_banked']:.2f} m",
                 fontsize=9.3, color=INK, va="center")
    fig.savefig(f"{OUT}/fig_attention_levels.png", dpi=100, facecolor=BG)
    print("wrote fig_attention_levels.png")


# ------------------------------------------------------------------------------------------------ video
def _job(j):
    kind, a, b = j
    if kind == "frame":
        return render_frame(a, b)
    return render_card(a)


def video(workers=4, fps=4, seed="M2b_s0", out=f"{OUT}/v8f_hierarchy_attention.mp4", limit=0):
    from multiprocessing import Pool
    jobs, reps = [], []
    c = cards(seed)
    jobs.append(("card", c, 0)); reps.append(4 * fps)                            # 4 s title
    for k in range(4):
        name, desc = ORDER[k]
        sel = SEL[name]
        jobs.append(("card", [(f"Clip {k + 1} of 4:  {desc}", 24, "#ffffff", True),
                              (f"clip {sel['clip']}   |   selected eval window #{sel['window']} at t = {0.2 * sel['t']:.1f} s   |   banked own-code ADE6: seed 0 {sel['ade6_seed0_banked']:.2f} m, seed 1 {sel['ade6_seed1_banked']:.2f} m",
                               14, "#aeb7c4", False),
                              ("The whole clip follows. Every step (0.2 s) is evaluated with the physical tick schedule: S every 1 s, T every 0.5 s, O every step.", 14, "#e8ecf2", False)], 0))
        reps.append(2 * fps)
        z = np.load(f"{VIZ_DIR}/{seed}__{sel['clip']}.npz")
        n = len(z["t"])
        for i in range(n if not limit else min(limit, n)):
            jobs.append(("frame", k, i)); reps.append(1)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(fps), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out + ".tmp.mp4"], stdin=subprocess.PIPE)
    t0 = time.time()
    with Pool(workers) as p:
        for n, img in enumerate(p.imap(_job, jobs, chunksize=1)):
            for _ in range(reps[n]):
                ff.stdin.write(img.tobytes())
            if n % 20 == 0:
                print(f"{n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    ff.stdin.close()
    rc = ff.wait()
    if rc != 0:
        raise SystemExit(f"ffmpeg failed rc={rc}; the previous {out} (if any) was NOT overwritten")
    os.replace(out + ".tmp.mp4", out)
    print("wrote", out, os.path.getsize(out) / 1e6, "MB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("frame", "keyfig", "video"))
    ap.add_argument("--clip", type=int, default=0)
    ap.add_argument("--i", type=int, default=20)
    ap.add_argument("--png", default=f"{OUT}/_test_frame.png")
    ap.add_argument("--seed", default="M2b_s0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "frame":
        from PIL import Image
        t0 = time.time()
        Image.fromarray(render_frame(a.clip, a.i, a.seed)).save(a.png)
        print("frame", time.time() - t0, a.png)
    elif a.cmd == "keyfig":
        keyfig(a.seed)
    else:
        video(a.workers, a.fps, a.seed, limit=a.limit)
