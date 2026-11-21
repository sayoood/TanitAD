# Banked from pod4 (ran the 5k milestone autonomously end-to-end).
# TARGET_STEP env selects the milestone (default 10000).
"""5k-milestone P-battery watcher (pod4). One-shot.

Waits until the HF-mirrored v6F checkpoint reaches step >= 5000 (the new pod5
push loop uploads it every ~20 min), then: pauses aug120 if it is mid-run
(per-batch resume-safe by design), pulls the checkpoint, runs the P-battery
chain AND the speed-echo control, pushes both result sets to HF, and restarts
aug120 if it had remaining work. GPU is never shared: the battery only starts
after the aug120 PID is gone.
"""
import json, os, shutil, signal, subprocess, time
TARGET = int(os.environ.get("TARGET_STEP", "10000"))
from huggingface_hub import HfApi, hf_hub_download

R = "Sayood/tanitad-v6"
STACK = "/workspace/TanitAD/stack"
ENV = {**os.environ, "PYTHONPATH": STACK, "OMP_NUM_THREADS": "6"}
PY = "/workspace/a2venv/bin/python"


def hf_step() -> int:
    try:
        p = hf_hub_download(R, "v6F-SW-30k/config.json", repo_type="model",
                            force_download=True)
        _ = json.load(open(p))
        q = hf_hub_download(R, "v6F-SW-30k/train_log.jsonl", repo_type="model",
                            force_download=True)
        steps = [json.loads(l).get("step", 0) for l in open(q) if l.strip()]
        return max((s for s in steps if isinstance(s, int)), default=0)
    except Exception as e:                                  # noqa: BLE001
        print("HFSTEP_FAIL", type(e).__name__, flush=True)
        return 0


while True:
    s = hf_step()
    print(time.strftime("%H:%M:%S"), "hf_step", s, flush=True)
    if s >= TARGET:
        break
    time.sleep(600)

# pause aug120 if running (explicit pid via pgrep -f is banned; scan cmdline)
paused = None
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline").read()
    except OSError:
        continue
    if "aug120.py" in cmd and int(pid) != os.getpid():
        paused = int(pid)
        os.kill(paused, signal.SIGTERM)
        print("PAUSED_AUG120", paused, flush=True)
while paused:
    time.sleep(5)
    if not os.path.exists(f"/proc/{paused}"):
        break

os.makedirs("/workspace/v6Fck", exist_ok=True)
for f in ("ckpt.pt", "config.json"):
    p = hf_hub_download(R, f"v6F-SW-30k/{f}", repo_type="model",
                        force_download=True)
    shutil.copyfile(p, f"/workspace/v6Fck/{f}")
print("CKPT5K_PULLED", flush=True)

env = {**ENV, "S": STACK, "PY": PY, "CKPT": "/workspace/v6Fck/ckpt.pt",
       "OUT": "/workspace/experiments/v6F-pb-milestone"}
rc = subprocess.call(["bash", "scripts/p_battery_chain.sh"], cwd=STACK,
                     env=env)
print("PB5K_RC", rc, flush=True)
rc2 = subprocess.call(
    [PY, "-u", "scripts/probe_latent_state.py", "--ckpt",
     "/workspace/v6Fck/ckpt.pt", "--v2-val-cache",
     "/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl",
     "--require-parity", "--out", "/workspace/experiments/v6F-pb-milestone-echoctl",
     "--ks", "5,10,15,20", "--episodes", "40", "--stride", "8",
     "--frame-h", "256", "--frame-w", "640", "--frame-hfov", "120.0",
     "--projection", "cylindrical", "--speed-echo-control"],
    cwd=STACK, env=ENV)
print("ECHO5K_RC", rc2, flush=True)

api = HfApi()
for src, dst in (("/workspace/experiments/v6F-pb-milestone", "pbattery_" + str(TARGET) + "/battery"),
                 ("/workspace/experiments/v6F-pb-milestone-echoctl",
                  "pbattery_" + str(TARGET) + "/echoctl")):
    if os.path.isdir(src):
        api.upload_folder(folder_path=src, path_in_repo=dst, repo_id=R,
                          repo_type="model")
print("PB5K_PUSHED", flush=True)

if paused:
    subprocess.Popen([PY, "-u", "/workspace/aug120.py"],
                     stdout=open("/workspace/aug120.log", "ab"),
                     stderr=subprocess.STDOUT, env=ENV,
                     start_new_session=True)
    print("AUG120_RESTARTED", flush=True)
print("WATCH5K_DONE", flush=True)
