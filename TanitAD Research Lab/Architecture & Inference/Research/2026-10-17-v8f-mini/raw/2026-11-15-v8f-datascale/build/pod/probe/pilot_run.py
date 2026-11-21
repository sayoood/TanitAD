"""DSCALE10K block-0 pilot driver (pod). Called by pilot.sh AFTER the STRAT1 chain ended and the GPU is below 3,500 MiB. Does NOT start block 1.
Steps: (b) GPU probes (reach probe on U, G-HYG records of B and U, the trainer's own --selfcheck-only on both: its real-data G-LIVE / G-DVB)
       (c) 2 x 600 steps (B_s0 and U_s0 stage-A argv, --steps 600): s/step from the trainer's cumulative `sec`, peak GPU, peak RSS, bytes read from disk / s
       (d) the 200-clip DAgger-bank timing on B's pilot checkpoint
       (e) PILOT_REPORT.json with the projected block-1 wall time (the card's 10 h stop rule)."""
import glob
import json
import os
import subprocess
import sys
import threading
import time

D = "/root/dscale10k"
PY = "/root/venv/bin/python"
TREE = D + "/tree"
ENV = dict(os.environ, PYTHONPATH=":".join([TREE + "/stack", TREE + "/stack/scripts", TREE, D + "/study"]), HF_HUB_DISABLE_XET="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
ENV.pop("CUDA_VISIBLE_DEVICES", None)
REP = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "evidence_class": "MEASURED on the pod (RTX 4000 Ada), this file", "steps": {}}
os.makedirs(D + "/pilot", exist_ok=True)
os.makedirs(D + "/probe", exist_ok=True)


def save():
    json.dump(REP, open(D + "/PILOT_REPORT.json", "w"), indent=1, default=str)


def fail(why):
    REP["status"] = "FAIL"
    REP["why"] = why
    save()
    print("ZZ DSCALE10K PILOT FAIL ZZ " + why[:300], flush=True)
    sys.exit(1)


def sh(cmd, log, timeout=None):
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=TREE, timeout=timeout, stdin=subprocess.DEVNULL)
    return r.returncode


def free_gb():
    s = os.statvfs("/root")
    return s.f_bavail * s.f_frsize / 1e9


def gpu_used():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
    except Exception:  # noqa: BLE001
        return -1


def argv_of(name):
    return json.load(open(f"{D}/argv/{name}.json"))["argv"]


def setf(t, f, vals):
    t = list(t)
    if f not in t:
        return t + [f] + list(vals)
    i = t.index(f)
    j = i + 1
    while j < len(t) and not (t[j].startswith("--") and len(t[j]) > 2):
        j += 1
    return t[:i + 1] + list(vals) + t[j:]


def delf(t, f):
    t = list(t)
    if f not in t:
        return t
    i = t.index(f)
    j = i + 1
    while j < len(t) and not (t[j].startswith("--") and len(t[j]) > 2):
        j += 1
    return t[:i] + t[j:]


if free_gb() < 60:
    fail(f"free disk {free_gb():.0f} GB < 60 GB guard")
REP["free_gb_start"] = round(free_gb(), 1)
for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
    if os.path.exists(p):
        REP["memory_limit"] = open(p).read().strip()
REP.setdefault("memory_limit", "no cgroup limit file found (cat memory.max: absent)")
REL_U = D + "/merged/v9_labels_union.manifest.json"

# ---- (b) GPU probes
for nm, extra in (("B_s0.A", []), ("U_s0.A", ["--reach"])):
    out = f"{D}/probe/{nm}.json"
    rc = sh([PY, D + "/probe/reach_probe.py", f"{D}/argv/{nm}.json", out, REL_U] + extra, f"{D}/probe/{nm}.log", 3000)
    REP["steps"]["probe_" + nm] = {"rc": rc, "out": out}
    save()
    if rc != 0:
        fail(f"reach_probe {nm} rc {rc}: see {D}/probe/{nm}.log")
for nm in ("B_s0.A", "U_s0.A"):
    t = setf(argv_of(nm), "--out", [f"{D}/pilot/selfcheck_{nm}"])
    t = [x for x in t if x != "--fit-join"] + ["--selfcheck-only"]
    rc = sh([PY, TREE + "/stack/scripts/v8f_mini_train.py"] + t, f"{D}/pilot/selfcheck_{nm}.log", 3000)
    sc = f"{D}/pilot/selfcheck_{nm}/selfcheck.json"
    ok = os.path.isfile(sc)
    live = {}
    if ok:
        s = json.load(open(sc))
        live = {"declared_vs_built": s.get("declared_vs_built"), "live_pass": (s.get("live") or {}).get("pass"), "groups_grad": (s.get("live") or {}).get("groups_grad"), "n_trainable_tensors": (s.get("live") or {}).get("n_trainable")}
    REP["steps"]["selfcheck_" + nm] = {"rc": rc, "selfcheck_json": sc if ok else None, **live}
    save()
    if rc != 0 or not ok:
        fail(f"trainer --selfcheck-only {nm} rc {rc}: see {D}/pilot/selfcheck_{nm}.log")

# ---- (c) the 600-step pilots
def run_pilot(tag, nm):
    t = argv_of(nm)
    t = setf(t, "--out", [f"{D}/pilot/{tag}"])
    t = delf(t, "--fit-join")
    t = delf(t, "--steps-mult")
    t = setf(t, "--steps", ["600"])
    t = setf(t, "--log-every", ["50"])
    t = setf(t, "--ckpt-every", ["600"])
    log = f"{D}/pilot/{tag}.log"
    f = open(log, "w")
    p = subprocess.Popen([PY, TREE + "/stack/scripts/v8f_mini_train.py"] + t, stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=TREE, stdin=subprocess.DEVNULL)
    g_peak, rss_peak, io0, t0 = 0, 0, None, time.time()
    while p.poll() is None:
        time.sleep(2)
        g_peak = max(g_peak, gpu_used())
        try:
            st = open(f"/proc/{p.pid}/status").read()
            rss_peak = max(rss_peak, int([x for x in st.splitlines() if x.startswith("VmRSS")][0].split()[1]) // 1024)
            io = int([x for x in open(f"/proc/{p.pid}/io").read().splitlines() if x.startswith("read_bytes")][0].split()[1])
            io0 = io if io0 is None else io0
            io_last = io
        except Exception:  # noqa: BLE001
            pass
    wall = time.time() - t0
    rows = []
    lp = f"{D}/pilot/{tag}/train_log.jsonl"
    if os.path.isfile(lp):
        for x in open(lp, encoding="utf-8"):
            if x.strip().startswith("{") and '"sec"' in x:
                r = json.loads(x)
                if "step" in r and "sec" in r:
                    rows.append(r)
    rows.sort(key=lambda r: r["step"])
    d = {"rc": p.returncode, "wall_s": round(wall, 1), "gpu_mib_peak_2s_sampled": g_peak, "rss_mib_peak": rss_peak, "steps": [r["step"] for r in rows], "sec_cum": [round(r["sec"], 2) for r in rows]}
    if len(rows) >= 2:
        d["s_per_step_steady"] = (rows[-1]["sec"] - rows[0]["sec"]) / (rows[-1]["step"] - rows[0]["step"])
        d["s_per_step_overall"] = rows[-1]["sec"] / rows[-1]["step"]
    if io0 is not None:
        d["disk_read_MB_per_s_mean_over_wall"] = round((io_last - io0) / 1e6 / max(wall, 1.0), 1)
        d["disk_read_GB_total"] = round((io_last - io0) / 1e9, 2)
    d["ckpt"] = f"{D}/pilot/{tag}/ckpt_last.pt" if os.path.isfile(f"{D}/pilot/{tag}/ckpt_last.pt") else None
    return d


for tag, nm in (("pB", "B_s0.A"), ("pU", "U_s0.A")):
    d = run_pilot(tag, nm)
    REP["steps"]["pilot_" + tag] = d
    save()
    if d["rc"] != 0 or d["ckpt"] is None or "s_per_step_steady" not in d:
        fail(f"pilot {tag} rc {d['rc']} ckpt {d['ckpt']}: see {D}/pilot/{tag}.log")

# ---- (d) the 200-clip DAgger bank timing, from B's pilot checkpoint
bank_out = f"{D}/pilot/bank200"
t0 = time.time()
rc = sh([PY, TREE + "/stack/scripts/v8f_dagger_bank.py", "gen", "--ckpt", REP["steps"]["pilot_pB"]["ckpt"], "--m2b-argv", f"{D}/argv/B_s0.A.json", "--cache-dir", "/root/data/cache_pod",
         "--tracks", "/root/data/cache_pod/native_tracks.pt", "--join-train", "/root/strat1/inputs/join_train.npz", "--eval-grid", "/root/strat1/eval_grid_windows.json", "--out-dir", bank_out,
         "--limit-clips", "200", "--n-shards", "1", "--device", "cuda"], f"{D}/pilot/bank200.log", 3 * 3600)
bank_s = time.time() - t0
REP["steps"]["bank200"] = {"rc": rc, "wall_s": round(bank_s, 1), "out": bank_out, "note": "200 clips of B's train list, --limit-clips (PARTIAL bank); the generator's own controls ran"}
save()
if rc != 0:
    REP["steps"]["bank200"]["warning"] = "rc != 0: see pilot/bank200.log; projection uses the wall time up to the failure and is marked unreliable"

# ---- (e) projection of block 1 (ESTIMATED from the measured s/step)
sB, sU = REP["steps"]["pilot_pB"]["s_per_step_steady"], REP["steps"]["pilot_pU"]["s_per_step_steady"]
bankB, bankU = bank_s * 4512 / 200.0, bank_s * 14512 / 200.0 * 1.0
A_B, A_U, SB_B, SB_U = sB * 19164, sU * 19164, sB * 3000, sU * 3000
b1 = 2 * (A_B + SB_B + bankB) + (A_U + SB_U + bankU) + A_U
REP["projection_block1"] = {
    "evidence_class": "ESTIMATED from the 600-step pilots (stage-B s/step taken equal to the stage-A pilot rate: conservative; bank wall scaled linearly in clips; the union bank at the B clip rate)",
    "arms": "B_s0 (A+B+bank), B_rep (A+B+bank), U_s0 (A+B+bank), R_s0 (A)",
    "s_per_step": {"B": sB, "U": sU, "U_over_B": sU / sB},
    "stage_A_h": {"B": A_B / 3600, "U": A_U / 3600}, "stage_B_h": {"B": SB_B / 3600, "U": SB_U / 3600},
    "bank_h": {"B_4512_clips": bankB / 3600, "U_14512_clips": bankU / 3600},
    "block1_wall_h": b1 / 3600,
    "stop_rule_10h": "STOP, back to the MM / PI" if b1 / 3600 > 10.0 else "within 10 h",
    "mitigation_trigger_U_over_1p5B": bool(sU / sB > 1.5),
}
REP["status"] = "DONE"
REP["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save()
print("ZZ DSCALE10K PILOT DONE ZZ", flush=True)
