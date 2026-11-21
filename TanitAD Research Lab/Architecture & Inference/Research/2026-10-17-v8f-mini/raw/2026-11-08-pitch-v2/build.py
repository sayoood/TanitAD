"""Render the TanitAD pitch film scene by scene (CPU only), then concatenate.

  python build.py still <scene_idx> t1,t2,...     -> PNG stills + a montage in the scratch dir
  python build.py render [idx ...] [--workers N]  -> scene_XX.mp4 in the scratch render dir
  python build.py concat                          -> video/out/tanitad_pitch.mp4
"""
import os, sys, subprocess, shutil, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from fx import *
import scenes

SCR = "F:/TanitAD-scratch/pitch"
RDIR = SCR + "/render"
OUT = "F:/TanitAD-scratch/pitch/out_v2"
FADE = 0.5
STARTS, tot = [], 0.0
for _n, _d, _f in scenes.SCENES:
    STARTS.append(tot)
    tot += _d
TOTAL = tot


def frame(si, t):
    name, dur, fn = scenes.SCENES[si]
    g = STARTS[si] + t
    cv0 = Canvas()
    particles(cv0, g)
    if si not in (0, len(scenes.SCENES) - 1):
        cv0.text(110, 1030, "TanitAD", 20, TXT3, "semi", tracking=2.5)
    cv0.line([(110, 1066), (1810, 1066)], mix(LINE, BG0, 0.5), 2, cap=False)
    pw = 1700 * g / TOTAL
    if pw > 2:
        cv0.line([(110, 1066), (110 + pw, 1066)], mix(TXT3, BG0, 0.2), 2, cap=False)
    cv = Canvas(cv0.im)
    fn(cv, t)
    env = min(ep(t, 0, FADE), 1 - ep(t, dur - FADE, dur))
    im = cv.im if env >= 0.999 else Image.blend(cv0.im, cv.im, env)
    return im.reduce(SS)


def render(si):
    name, dur, fn = scenes.SCENES[si]
    os.makedirs(RDIR, exist_ok=True)
    out = f"{RDIR}/scene_{si + 1:02d}_{name}.mp4"
    n = int(round(dur * FPS))
    cmd = [shutil.which("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-profile:v", "high", "-threads", "2", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for k in range(n):
        p.stdin.write(frame(si, k / FPS).tobytes())
    p.stdin.close()
    p.wait()
    return f"scene {si + 1} {name}: {n} frames, {time.time() - t0:.0f} s, rc {p.returncode}"


def still(si, ts):
    os.makedirs(SCR + "/stills", exist_ok=True)
    ims = []
    for t in ts:
        im = frame(si, t)
        im.save(f"{SCR}/stills/s{si + 1:02d}_{t:05.1f}.png")
        ims.append(im.resize((960, 540), Image.LANCZOS))
    cols = 2
    rows = (len(ims) + 1) // 2
    sheet = Image.new("RGB", (960 * cols, 540 * rows))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * 960, (i // cols) * 540))
    sheet.save(f"{SCR}/stills/sheet_s{si + 1:02d}.png")
    print("sheet", f"{SCR}/stills/sheet_s{si + 1:02d}.png")


def concat():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(f for f in os.listdir(RDIR) if f.startswith("scene_") and f.endswith(".mp4"))
    assert len(files) == len(scenes.SCENES), files
    lst = RDIR + "/list.txt"
    with open(lst, "w", newline="\n") as fh:
        for f in files:
            fh.write(f"file '{RDIR}/{f}'\n")
    subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", OUT + "/tanitad_pitch.mp4"], check=True)
    print("wrote", OUT + "/tanitad_pitch.mp4")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "still":
        still(int(sys.argv[2]), [float(x) for x in sys.argv[3].split(",")])
    elif cmd == "render":
        args = sys.argv[2:]
        workers = 5
        if "--workers" in args:
            i = args.index("--workers")
            workers = int(args[i + 1])
            del args[i:i + 2]
        idx = [int(a) for a in args] or list(range(len(scenes.SCENES)))
        from multiprocessing import Pool
        with Pool(workers) as pool:
            for r in pool.imap_unordered(render, sorted(idx, key=lambda i: -scenes.SCENES[i][1])):
                print(r, flush=True)
    elif cmd == "concat":
        concat()
