"""ATTACK diag_rank_distinctness.py: construct rank divergences it does NOT catch.
Expectations are LITERALS ("the instrument must return 1 / RANKS_ARE_A_RELABELLED_COPY"), never an
expression over the instrument's own logic."""
import json, os, shutil, subprocess, sys, math, numpy as np
SRC = r"D:/Projects/TanitAD/data/refe_targets_4cam"
PY  = r"C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/bd7d00af-b98e-42f1-a53c-cb4113059b0f/scratchpad/driverl-venv/Scripts/python.exe"
DIA = r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/diag_rank_distinctness.py"
TMP = os.path.abspath("attack_banks")

def rows(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def write(p, rs):
    with open(p, "w", encoding="utf-8") as f:
        for r in rs: f.write(json.dumps(r) + "\n")
def run(d):
    r = subprocess.run([PY, DIA, "--targets", d], capture_output=True, text=True)
    tail = [l for l in r.stdout.strip().splitlines() if l.startswith("RANKS_")]
    return r.returncode, (tail[-1] if tail else "??")

r0 = rows(SRC + "/targets_rank0.jsonl"); r1 = rows(SRC + "/targets_rank1.jsonl")

# --- first, how big IS the real novelty? (context for attack B) ---------------
k0 = {(r["log_name"], r["token"], r["step"]): r for r in r0}
seps = np.array([np.linalg.norm(np.array(r["traj"])[-1,:2] - np.array(k0[(r["log_name"],r["token"],r["step"])]["traj"])[-1,:2])
                 for r in r1 if (r["log_name"],r["token"],r["step"]) in k0])
print(f"REAL rank0-vs-rank1 4 s endpoint separation: min {seps.min():.4f} m  p10 {np.percentile(seps,10):.3f}  "
      f"median {np.median(seps):.3f}  max {seps.max():.3f} m")
for th in (0.01, 0.10, 0.50):
    print(f"   keys separated by < {th:4.2f} m: {(seps<th).sum():4d} / {len(seps)} ({100*(seps<th).mean():.1f} %)")

# --- ATTACK A: a THIRD rank that is a relabelled copy of rank 0 ---------------
d = os.path.join(TMP, "A_three_ranks"); os.makedirs(d, exist_ok=True)
write(d+"/targets_rank0.jsonl", r0); write(d+"/targets_rank1.jsonl", r1)
write(d+"/targets_rank2.jsonl", [dict(r, rank=2) for r in r0])   # the EXACT historical defect
code, verdict = run(d)
print(f"\nATTACK A  rank2 = relabelled copy of rank0 (3 rank files)")
print(f"   instrument -> exit {code}  {verdict}   EXPECTED: exit 1 RANKS_ARE_A_RELABELLED_COPY")
print(f"   {'** MISSED **' if code == 0 else 'caught'}")

# --- ATTACK B: rank1 = rank0 with a 1e-9 m perturbation -----------------------
d = os.path.join(TMP, "B_epsilon"); os.makedirs(d, exist_ok=True)
eps = []
for r in r0:
    t = [[c + 1e-9 for c in pt] for pt in r["traj"]]
    g = [c + 1e-9 for c in r["goal"]]
    eps.append(dict(r, rank=1, traj=t, goal=g))
write(d+"/targets_rank0.jsonl", r0); write(d+"/targets_rank1.jsonl", eps)
code, verdict = run(d)
print(f"\nATTACK B  rank1 = rank0 + 1e-9 m on every coordinate (zero new information)")
print(f"   instrument -> exit {code}  {verdict}   EXPECTED: exit 1 (a 1 nm 'route variant' is not one)")
print(f"   {'** MISSED **' if code == 0 else 'caught'}")

# --- ATTACK C: rank1 real, but its EGO vector copied from rank0 ---------------
d = os.path.join(TMP, "C_ego_copied"); os.makedirs(d, exist_ok=True)
cc = [dict(r, ego=k0[(r["log_name"],r["token"],r["step"])]["ego"]) for r in r1
      if (r["log_name"],r["token"],r["step"]) in k0]
write(d+"/targets_rank0.jsonl", r0); write(d+"/targets_rank1.jsonl", cc)
code, verdict = run(d)
n_ego_diff = sum(1 for r in r1 if (r["log_name"],r["token"],r["step"]) in k0
                 and r["ego"] != k0[(r["log_name"],r["token"],r["step"])]["ego"])
print(f"\nATTACK C  rank1 keeps its own traj/goal but its EGO is copied from rank0")
print(f"   (on the REAL data ego differs on {n_ego_diff}/{len(r1)} keys, so this is a real corruption)")
print(f"   instrument -> exit {code}  {verdict}   EXPECTED: exit 1 (the ego is a model INPUT)")
print(f"   {'** MISSED **' if code == 0 else 'caught'}")

# --- CONTROL: the instrument DOES catch the historical defect at rank1 --------
d = os.path.join(TMP, "D_control"); os.makedirs(d, exist_ok=True)
write(d+"/targets_rank0.jsonl", r0); write(d+"/targets_rank1.jsonl", [dict(r, rank=1) for r in r0])
code, verdict = run(d)
print(f"\nCONTROL   rank1 = relabelled copy of rank0 (the historical defect, at position 1)")
print(f"   instrument -> exit {code}  {verdict}   EXPECTED: exit 1 RANKS_ARE_A_RELABELLED_COPY")
print(f"   {'CAUGHT -- the probe is sensitive' if code == 1 else '** the probe itself is inert **'}")
