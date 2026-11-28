#!/usr/bin/env python3
"""mutate_dk141.py -- deliberate-regression arms for dk141 controls.  Each mutant is a text patch of ONE source line, written to a scratch copy (nothing in stack/, taniteval/ or
tip_export/ is touched); the controls run in FAST mode (B 50) against it and the mutant is KILLED when the run is not all-green.  The unmutated run is the baseline and must be
green except the bootstrap-bound cell check (C0), which is excluded from the verdict in fast mode.
usage: python -I mutate_dk141.py --scratch <dir> --out raw/DK141_MUTATION.json
"""
import argparse, json, os, subprocess, sys
HOME = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PY = sys.executable
LM = HOME + "/tip_export/taniteval/taniteval/lead_metrics.py"
DKP = HOME + "/dk141.py"
MUTANTS = [
 ("L1 gap without half the lead length", "lm", "gap = along - float(lead_len) / 2.0", "gap = along"),
 ("L2 headway = max instead of min", "lm", "headway[i] = np.nanmin(g)", "headway[i] = np.nanmax(g)"),
 ("L3 corridor disabled", "lm", "(np.abs(lat) < lat_m)", "(np.abs(lat) < 1e9)"),
 ("L4 time gap = headway * speed", "lm", "time_gap[i] = headway[i] / speeds[i]", "time_gap[i] = headway[i] * speeds[i]"),
 ("L5 closing-rate sign flipped", "lm", "rate = -np.diff(g[idx]) / (np.diff(idx) * dt)", "rate = np.diff(g[idx]) / (np.diff(idx) * dt)"),
 ("L6 LEAD-VIOL inequality flipped", "lm", "return (p < thr) & (h >= thr)", "return (p > thr) & (h >= thr)"),
 ("L7 behind-the-lead steps admitted (gap>=0 dropped)", "lm", " & (gap >= 0.0) & (np.abs(lat) < lat_m)", " & (np.abs(lat) < lat_m)"),
 ("D1 TTC violation threshold 0", "dk", "TTC_VIOL_S = 1.5 ", "TTC_VIOL_S = 0.0 "),
 ("D2 oracle without half the lead length", "dk", "gap = lead_x - lens[:, None] / 2.0", "gap = lead_x"),
 ("D3 at-risk condition broken", "dk", "at_risk = is_lead & np.isfinite(th) & (th >= TIGHT_TG_S)", "at_risk = is_lead & np.isfinite(th) & (th >= 99.0)"),
 ("D4 join frame = t instead of 2t", "dk", '"frame": z["frame"].astype(np.int64),', '"frame": z["frame"].astype(np.int64) // 2,'),
 ("D5 paired rate delta sign flipped", "dk", "**ci.paired_episode_cluster_bootstrap(va[ok], vb[ok], eid[ok]", "**ci.paired_episode_cluster_bootstrap(vb[ok], va[ok], eid[ok]"),
 ("D6 non-finite time gaps scored as 0 (free flow)", "dk", "tight = np.where(np.isfinite(tg), (tg < TIGHT_TG_S).astype(float), nan)", "tight = np.where(True, (tg < TIGHT_TG_S).astype(float), nan)"),
 ("D7 pred shifted one step", "dk", "preds = {n: v[0] for n, v in arms.items()}", "preds = {n: np.concatenate([v[0][:, 1:], v[0][:, -1:]], axis=1) for n, v in arms.items()}"),
]


def run(extra, tag):
    out = f"{ARGS.scratch}/ctl_{tag}.json"
    cmd = [PY, "-I", ARGS.dk, "controls", "--out", out, "--n-boot", "50"] + extra
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    fails = [l for l in p.stdout.splitlines() if l.startswith("FAIL ")]
    return p.returncode, fails, p.stderr.strip()[-300:]


ap = argparse.ArgumentParser(); ap.add_argument("--scratch", required=True); ap.add_argument("--out", required=True); ap.add_argument("--dk", default=DKP)
ARGS = ap.parse_args()
os.makedirs(ARGS.scratch, exist_ok=True)
res = []
rc, fails, err = run([], "baseline")
base_ok = rc in (0, 3) and all(("C0 banked read reproduced" in f) for f in fails)
res.append({"mutant": "BASELINE (unmutated)", "rc": rc, "fails": fails, "baseline_green_except_bootstrap_cells": bool(base_ok)})
print("baseline rc", rc, "fails", fails, "OK" if base_ok else "BASELINE NOT GREEN", flush=True)
for i, (name, kind, old, new) in enumerate(MUTANTS):
    if kind == "lm":
        src = open(LM, encoding="utf-8").read()
        if src.count(old) != 1:
            res.append({"mutant": name, "status": "PATCH-NOT-APPLIED", "occurrences": src.count(old)}); print(name, "PATCH-NOT-APPLIED", flush=True); continue
        pth = f"{ARGS.scratch}/lm_m{i}.py"; open(pth, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        rc, fails, err = run(["--mutant-lead-metrics", pth], f"m{i}")
    else:
        src = open(DKP, encoding="utf-8").read()
        if src.count(old) != 1:
            res.append({"mutant": name, "status": "PATCH-NOT-APPLIED", "occurrences": src.count(old)}); print(name, "PATCH-NOT-APPLIED", flush=True); continue
        pth = f"{ARGS.scratch}/dk_m{i}.py"; open(pth, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
        ARGS_dk = ARGS.dk; ARGS.dk = pth
        rc, fails, err = run([], f"m{i}")
        ARGS.dk = ARGS_dk
    real = [f for f in fails if "C0 banked read reproduced" not in f]
    killed = rc == 2 or bool(real)
    res.append({"mutant": name, "rc": rc, "killed": killed, "failing_controls": [f[5:] for f in real], "refusal": err if rc == 2 else None})
    print(("KILLED   " if killed else "SURVIVED "), name, "rc", rc, "|", (real[:2] or [err[:120]]), flush=True)
nk = sum(1 for r in res[1:] if r.get("killed"))
out = {"schema": "dk141-mutation-v1", "baseline_ok": bool(base_ok), "n_mutants": len(MUTANTS), "n_killed": nk, "results": res, "mode": "FAST (B 50); C0 bootstrap-cell check excluded from the verdict"}
json.dump(out, open(ARGS.out, "w", encoding="utf-8", newline="\n"), indent=1)
print(f"MUTATION {nk}/{len(MUTANTS)} killed; baseline_ok={base_ok}")
sys.exit(0 if (base_ok and nk == len(MUTANTS)) else 4)
