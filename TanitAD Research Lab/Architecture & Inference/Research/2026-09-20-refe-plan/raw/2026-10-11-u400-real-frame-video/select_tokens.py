"""Pick the video's runs: every run of >= MIN_RUN consecutive navtest tokens (0.5 s apart) in the navtest logs that have RAW 10 Hz
front-camera frames (nuPlan-mini CAM_F0 zips).  Writes video_runs.json, video_tokens.json (+ one token file per log) and RUNS.md.

Evidence: selection only; the tags come from the banked W3 export (driving command, ego speed, human future) and the banked
per-token harness CSV of THIS checkpoint (u400 full navtest).   python select_tokens.py
"""
import collections
import gzip
import json
import os

import numpy as np
import pandas as pd

DRV = os.environ.get("REFE_DRIVE", "E:")
EXPORT = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
CSV = f"{DRV}/Projects/TanitAD/data/refe_navtest/score/refe_stage2_sft4_B_u400_navtest_full/refe_stage2_sft4_B_u400_navtest_full.csv"
MINI = f"{DRV}/Projects/TanitAD/data/nuplan-camera/mini"
S4 = f"{DRV}/Projects/TanitAD/data/nuplan-camera/scenarios4"
LOGS = f"{DRV}/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
MIN_RUN = 6
OUT = os.path.dirname(os.path.abspath(__file__))
CMD = ["LEFT", "STRAIGHT", "RIGHT", "UNKNOWN"]


def main():
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    csv = pd.read_csv(CSV).set_index("token")
    have_f0 = {n[:-len("_CAM_F0.zip")] for n in os.listdir(MINI) if n.endswith("_CAM_F0.zip")}
    have_f0 |= {n[:-len("_CAM_F0.zip")] for n in os.listdir(S4) if n.endswith("_CAM_F0.zip")}
    navtest_logs = {n[:-4] for n in os.listdir(LOGS) if n.endswith(".pkl")}
    cams4 = {l for l in have_f0 if all(os.path.exists(f"{S4}/{l}_CAM_{c}.zip") for c in ("B0", "L0", "R0"))}
    logs = sorted(have_f0 & navtest_logs)
    print("navtest logs with raw F0:", len(logs), "| with all four raw cameras:", sorted(cams4))
    by = collections.defaultdict(list)
    for t, x in exp.items():
        if x["log_name"] in logs:
            by[x["log_name"]].append((x["timestamps_us"][-1], t))
    runs = []
    for log in sorted(logs, key=lambda l: (l not in cams4, l)):          # the four-real-camera log first
        lst = sorted(by[log])
        cur = [lst[0]]
        parts = []
        for a, b in zip(lst, lst[1:]):
            if abs((b[0] - a[0]) / 1e6 - 0.5) < 0.02:
                cur.append(b)
            else:
                parts.append(cur)
                cur = [b]
        parts.append(cur)
        for r in parts:
            if len(r) < MIN_RUN:
                continue
            toks = [t for _, t in r]
            cmd = [int(np.argmax(exp[t]["ego_statuses"][-1]["driving_command"])) for t in toks]
            v = np.array([np.hypot(*exp[t]["ego_statuses"][-1]["ego_velocity"]) for t in toks])
            hf = np.array([exp[t]["human_future_poses"][-1] for t in toks])
            sc = csv.loc[toks, "score"].values
            tags = []
            if (v < 0.5).mean() >= 0.5:
                tags.append("stop")
            if any(c != 1 for c in cmd) or np.degrees(np.abs(hf[:, 2]).max()) >= 30:
                tags.append("turn")
            if np.abs(hf[:, 1]).max() >= 2.5 and np.degrees(np.abs(hf[:, 2]).max()) < 15 and all(c == 1 for c in cmd):
                tags.append("lane-change?")
            if v.max() >= 10:
                tags.append("speed>=10m/s")
            if (sc < 0.5).any():
                tags.append("FAILURE")
            runs.append({"log": log, "tokens": toks, "t0_us": r[0][0], "n": len(toks), "cams": "4 raw" if log in cams4 else "F0 raw + 2Hz held",
                         "tags": tags, "cmd": dict(collections.Counter(CMD[c] for c in cmd)), "v_min": float(v.min()), "v_max": float(v.max()),
                         "score_mean": float(sc.mean()), "score_min": float(sc.min()), "n_score_lt_0.5": int((sc < 0.5).sum())})
    allt = [t for r in runs for t in r["tokens"]]
    json.dump(runs, open(f"{OUT}/video_runs.json", "w"), indent=1)
    json.dump({"tokens": allt}, open(f"{OUT}/video_tokens.json", "w"))
    os.makedirs(f"{OUT}/tokens_by_log", exist_ok=True)
    for log in sorted({r["log"] for r in runs}):
        json.dump({"tokens": [t for r in runs if r["log"] == log for t in r["tokens"]]}, open(f"{OUT}/tokens_by_log/{log}.json", "w"))
    secs = len(allt) * 0.5
    L = [f"# Run list ({len(runs)} runs, {len(allt)} tokens = {secs:.0f} s = {secs / 60:.2f} min of 10 Hz video)\n",
         f"Rule: every run of >= {MIN_RUN} consecutive navtest tokens (0.5 s apart) in the {len(logs)} navtest logs that have raw nuPlan CAM_F0 frames.\n",
         "| # | log | tokens | s | cameras | cmd | v m/s | true PDMS mean/min | tags |", "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(runs):
        L.append(f"| {i + 1} | {r['log']} | {r['n']} | {r['n'] * 0.5:.1f} | {r['cams']} | {r['cmd']} | {r['v_min']:.1f}-{r['v_max']:.1f} | "
                 f"{100 * r['score_mean']:.1f} / {100 * r['score_min']:.1f} | {', '.join(r['tags'])} |")
    open(f"{OUT}/RUNS.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"{len(runs)} runs, {len(allt)} tokens, {secs:.0f} s; tags:", collections.Counter(t for r in runs for t in r["tags"]))


if __name__ == "__main__":
    main()
