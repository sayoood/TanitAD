"""End-to-end driver, fully autonomous (so no agent has to wait on it):  inference gate -> render clips -> cards -> concat -> QA + stills.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -u run_pipeline.py > pipeline.log 2>&1 &

Step 0 waits for run_inference.py to finish (inference/FIDELITY_GATE.json) or abort (GATE_FAIL.json -> the pipeline STOPS: a video of a different
decision than the scored one is not this model).  Every step is resumable (clip segments carry .done markers).  Writes pipeline_done.json.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
OUT = "E:/Projects/TanitAD/data/videos/u400_realframes.mp4"


def run(args, name):
    t0 = time.time()
    print(f"[{time.strftime('%T')}] {name} ...", flush=True)
    rc = subprocess.call([PY, "-u"] + args, cwd=HERE)
    print(f"[{time.strftime('%T')}] {name} rc={rc} ({time.time() - t0:.0f} s)", flush=True)
    if rc != 0:
        json.dump({"failed": name, "rc": rc}, open(os.path.join(HERE, "pipeline_done.json"), "w"))
        sys.exit(rc)


def main():
    gate_ok, gate_fail = os.path.join(HERE, "inference", "FIDELITY_GATE.json"), os.path.join(HERE, "GATE_FAIL.json")
    while not (os.path.exists(gate_ok) or os.path.exists(gate_fail)):
        time.sleep(30)
    if os.path.exists(gate_fail):
        print("FIDELITY GATE FAILED -- stopping", flush=True)
        json.dump({"failed": "fidelity gate"}, open(os.path.join(HERE, "pipeline_done.json"), "w"))
        return 2
    g = json.load(open(gate_ok))["overall"]
    print("fidelity gate:", g, flush=True)
    assert g["pass"], g
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    run(["render_u400_video.py", "render", "--workers", "4", "--out", "x"], "render clips")
    run(["render_u400_video.py", "cards"], "cards")
    run(["render_u400_video.py", "concat", "--out", OUT], "concat")
    run(["qa_video.py", OUT, "--stills"], "QA + stills")
    json.dump({"ok": True, "video": OUT, "bytes": os.path.getsize(OUT), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(os.path.join(HERE, "pipeline_done.json"), "w"))
    print("ZZPIPELINE_DONE", OUT, os.path.getsize(OUT), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
