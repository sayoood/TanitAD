"""P6 real-frame video, fully autonomous (no agent waits on it):
wait for dump.DONE / dump.FAILED -> fidelity gate (+ slice) -> render clips (resumable, 3 workers) -> cards -> concat -> QA + stills -> 1080p -> write_result -> cleanup.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -u run_all.py          (launched detached by launch_detached.py; log: run_all.log)

Writes pipeline_done.json on success, pipeline_FAILED.json on any failure (and GATE_FAIL.json if the fidelity gate fails: then NO render happens).
Before each stage it waits while free RAM < 6 GB. No GPU is used (CUDA_VISIBLE_DEVICES=-1). Resumable: re-run the same command; finished stages are
recorded in stages_done.json and finished clips carry .done markers.  Nothing here runs git.
"""
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
OUTDIR = os.path.join(HERE, "out")
OUT = f"{OUTDIR}/p6_realframes.mp4"
OUT1080 = f"{OUTDIR}/p6_realframes_1080p.mp4"
SEG = os.path.join(HERE, "segments")
MK = "F:/TanitAD_p6read_work/markers"
STATE = os.path.join(HERE, "stages_done.json")
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
for k in ("P6_WORK", "P6_PROPS", "P6_RUNS", "P6_GATE", "P6_GATE_DUMP", "P6_GATE_MARK", "P6_GATE_NTOK"):
    ENV.pop(k, None)       # the production run never inherits a test override


def say(m):
    print(f"[{time.strftime('%F %T')}] {m}", flush=True)


def wait_ram(who, min_gb=6.0, poll=30):
    t0, warned = time.time(), False
    while psutil.virtual_memory().available < min_gb * 2 ** 30:
        if not warned:
            say(f"{who}: free RAM {psutil.virtual_memory().available / 2 ** 30:.1f} GB < {min_gb} GB, waiting")
            warned = True
        time.sleep(poll)
    if warned:
        say(f"{who}: resumed after {time.time() - t0:.0f} s")


def fail(stage, why, rc=1):
    json.dump({"failed": stage, "why": why, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(os.path.join(HERE, "pipeline_FAILED.json"), "w"), indent=1)
    say(f"PIPELINE FAILED at {stage}: {why}")
    sys.exit(rc)


def done_stages():
    return json.load(open(STATE)) if os.path.exists(STATE) else []


def stage(name, fn):
    if name in done_stages():
        say(f"{name}: already done, skipped")
        return
    wait_ram(name)
    t0 = time.time()
    say(f"{name} ...")
    fn()
    json.dump(done_stages() + [name], open(STATE, "w"))
    say(f"{name} done ({time.time() - t0:.0f} s)")


def sh(args, name):
    rc = subprocess.call(args, cwd=HERE, env=ENV, stdin=subprocess.DEVNULL)
    if rc != 0:
        fail(name, f"rc {rc}: {' '.join(map(str, args))[:300]}", rc)


def s_wait():
    while True:
        if os.path.exists(f"{MK}/dump.FAILED"):
            fail("dump", "second navtest pass wrote dump.FAILED: " + open(f"{MK}/dump.FAILED", encoding="utf-8", errors="replace").read()[-1500:])
        if os.path.exists(f"{MK}/dump.DONE"):
            break
        time.sleep(60)
    d = json.load(open(f"{MK}/dump.DONE"))
    say(f"dump.DONE: {d}")
    if d.get("poses_bit_identical") is not True:
        fail("dump", f"dump.DONE reports poses_bit_identical {d.get('poses_bit_identical')}: the second pass is not bit-identical to the first; STOPPED, report to the Master Mind: {d}")


def s_gate():
    for f in ("GATE_FAIL.json", "GATE_FAIL_detail.json", "FIDELITY_GATE.json"):
        if os.path.exists(os.path.join(HERE, f)):
            os.remove(os.path.join(HERE, f))       # own earlier output of this stage; re-derived below
    rc = subprocess.call([PY, "-u", "gate_fidelity.py"], cwd=HERE, env=ENV, stdin=subprocess.DEVNULL)
    if rc != 0:
        fail("fidelity gate", "GATE_FAIL.json written, no render", rc)


def s_render():
    sh([PY, "-u", "render_p6_video.py", "render", "--workers", "3", "--out", "x"], "render")
    n = len([f for f in os.listdir(SEG) if f.startswith("clip") and f.endswith(".mp4.done")])
    if n != 94:
        fail("render", f"{n} clip segments done, expected 94")


def s_cards():
    sh([PY, "-u", "render_p6_video.py", "cards"], "cards")


def s_concat():
    os.makedirs(OUTDIR, exist_ok=True)
    sh([PY, "-u", "render_p6_video.py", "concat", "--out", OUT], "concat")


def s_qa():
    sh([PY, "-u", "qa_video.py", OUT, "--stills"], "qa")
    qa = json.load(open(os.path.join(HERE, "qa_report.json")))
    if not qa["frames"]["match"] or not qa["provenance_psnr"]["own_beats_adjacent_every_frame"]:
        fail("qa", f"frames {qa['frames']}, own_beats_adjacent {qa['provenance_psnr']['own_beats_adjacent_every_frame']}")


def s_1080():
    sh([FFMPEG, "-y", "-loglevel", "error", "-i", OUT, "-vf", "scale=1920:1080:flags=lanczos", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-threads", "3",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT1080], "1080p")


def s_result():
    sh([PY, "-u", "write_result.py", OUT, OUT1080], "write_result")


def s_cleanup():
    """own scratch only, after the final mp4 passed QA: the clip/card segments (re-creatable by re-running run_all with stages_done.json cleared)"""
    sp = os.path.abspath(SEG).replace("\\", "/")
    assert sp.startswith("F:/TanitAD_p6video/") and sp.endswith("/segments"), sp
    dels = []
    for f in sorted(os.listdir(SEG)):
        p = os.path.join(SEG, f)
        if os.path.isfile(p):
            dels.append({"path": p.replace("\\", "/"), "bytes": os.path.getsize(p)})
            os.remove(p)
    os.rmdir(SEG)
    json.dump({"deleted": dels, "total_bytes": sum(d["bytes"] for d in dels), "n": len(dels)}, open(os.path.join(HERE, "DELETIONS.json"), "w"), indent=1)
    say(f"deleted {len(dels)} segment files, {sum(d['bytes'] for d in dels) / 1e6:.0f} MB (DELETIONS.json)")


def main():
    say(f"run_all start, pid {os.getpid()}")
    for f in ("pipeline_FAILED.json", "pipeline_done.json"):
        if os.path.exists(os.path.join(HERE, f)):
            os.remove(os.path.join(HERE, f))     # stale result of an earlier attempt
    for nm, fn in (("wait_dump", s_wait), ("gate", s_gate), ("render", s_render), ("cards", s_cards), ("concat", s_concat), ("qa", s_qa), ("1080p", s_1080),
                   ("write_result", s_result)):
        stage(nm, fn)
    json.dump({"ok": True, "video": OUT, "video_1080p": OUT1080, "bytes": os.path.getsize(OUT), "bytes_1080p": os.path.getsize(OUT1080),
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(os.path.join(HERE, "pipeline_done.json"), "w"), indent=1)
    stage("cleanup", s_cleanup)
    say(f"ZZPIPELINE_DONE {OUT} {os.path.getsize(OUT)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as e:      # noqa: BLE001  never die silently
        import traceback
        fail("exception", traceback.format_exc()[-1500:])
