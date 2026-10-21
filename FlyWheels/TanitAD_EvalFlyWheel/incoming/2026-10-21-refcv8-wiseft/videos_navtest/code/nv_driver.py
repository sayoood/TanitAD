"""nv_driver.py -- render all six clips (parallel processes), the cards, and the combined reel with chapter markers, for one EMA arm.
usage: nv_driver.py --ema-arm LEGAL|RT --out DIR [--reel] [--no-reel]"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
PY = sys.executable
ap = argparse.ArgumentParser()
ap.add_argument("--ema-arm", default="LEGAL"); ap.add_argument("--out", required=True); ap.add_argument("--no-reel", action="store_true")
ap.add_argument("--crf", default="18")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
sel = json.load(open(HERE.parent / "selection.json"))[a.ema_arm]["rows"]
wanted = json.load(open("C:/Users/Admin/refcv8_pf/navsim_video/work/wanted.json"))
logs = list(wanted.keys())
byrank = {r["log"]: r for r in sel}
env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
procs, names = [], []
t0 = time.time()
for li, log in enumerate(logs):
    r = byrank[log]
    nm = f"navtest_{a.ema_arm}_r{r['rank']}_{r['tier']}_{log}.mp4"
    names.append(nm)
    procs.append(subprocess.Popen([PY, "-I", str(HERE / "nv_render.py"), "clip", "--ema-arm", a.ema_arm, "--clip", str(li), "--crf", a.crf, "--out", str(out / nm)],
                                  env=env, stdout=open(out / f"{nm}.log", "w"), stderr=subprocess.STDOUT))
subprocess.run([PY, "-I", str(HERE / "nv_render.py"), "cards", "--ema-arm", a.ema_arm, "--crf", a.crf, "--out", str(out / "cards")], env=env, check=True)
rcs = [p.wait() for p in procs]
print("clip exit codes", rcs, "%.0fs" % (time.time() - t0), flush=True)
assert all(rc == 0 for rc in rcs), "a clip render failed"
if not a.no_reel:
    ff = subprocess.run(["where", "ffmpeg"], capture_output=True, text=True).stdout.split("\n")[0].strip()
    parts = [out / "cards" / "title.mp4"] + [out / n for n in names] + [out / "cards" / "closing.mp4"]
    lst = out / "concat.txt"
    lst.write_text("".join(f"file '{str(p).replace(chr(92), '/')}'\n" for p in parts), encoding="utf-8")
    raw = out / f"_reel_raw_{a.ema_arm}.mp4"
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(raw)], check=True)

    def dur(p):
        r = subprocess.run([ff.replace("ffmpeg.exe", "ffprobe.exe"), "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(p)], capture_output=True, text=True)
        return float(r.stdout.strip())
    t, meta = 0.0, [";FFMETADATA1"]
    labels = ["Title card"] + [f"{byrank[logs[i]]['tier']} r{byrank[logs[i]]['rank']}: {logs[i]}" for i in range(len(logs))] + ["Summary"]
    for p, lab in zip(parts, labels):
        d = dur(p)
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(t * 1000)}", f"END={int((t + d) * 1000)}", f"title={lab}"]
        t += d
    (out / "chapters.txt").write_text("\n".join(meta) + "\n", encoding="utf-8")
    final = out / f"navtest_REEL_{a.ema_arm}_combined.mp4"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(raw), "-i", str(out / "chapters.txt"), "-map_metadata", "1", "-map_chapters", "1", "-c", "copy", "-movflags", "+faststart", str(final)], check=True)
    os.remove(raw)
    print("reel", final, "total %.1fs" % t, flush=True)
