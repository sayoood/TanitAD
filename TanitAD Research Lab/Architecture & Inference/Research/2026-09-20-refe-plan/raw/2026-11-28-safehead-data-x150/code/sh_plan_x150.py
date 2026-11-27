#!/usr/bin/env python3
"""x150 plan: +150 logs on top of the 35 cached (PI approval 2026-11-27, "+150 logs"). CPU, stdlib; no download here.

Differences from sh_plan.py (same eligibility, same frame_names/need files, same PLAN.json schema):
  * excludes the 35 logs in F:/safehead_cache/index.csv (read only);
  * includes EVERY remaining held-out log (src/splits/sft_heldout_logs.txt) that has a local DB, split=heldout;
  * fills with train logs: local-complete first (no download), then the other eligible logs of the held-out archives, then archives by
    eligible-logs-per-GB (recomputed as the plan fills) until N_TOTAL logs.
Run with SH_WORK=F:/safehead_cache/x150 (plan/ goes there). Reads meta/src from F:/safehead_cache (read only).
Exits 2 and writes plan/STOP_OVER_200GB if the transfer exceeds 200 GB.
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sh_lib as L  # noqa: E402

N_TOTAL = 150
CAP_BYTES = 200e9
OLD = "F:/safehead_cache"
LOCAL_DIRS = ["E:/Projects/TanitAD/data/nuplan-camera/scenarios4", "E:/Projects/TanitAD/data/nuplan-camera/mini"]


def main():
    assert os.path.abspath(L.W).replace("\\", "/").rstrip("/") != OLD, "refusing to plan into the original cache root"
    logs, toks = L.read_yaml_split()
    cached = {r["log"] for r in csv.DictReader(open(f"{OLD}/index.csv"))}
    held_all = [l.strip() for l in open(f"{OLD}/src/splits/sft_heldout_logs.txt") if l.strip()]
    cmd = json.load(open(L.COMMANDS))
    yaw = L.load_json_gz(L.YAW_TRAIN)["table"]
    tokset = set(toks)
    cand = {}
    for lg in logs:
        if lg in cached:
            continue
        p = L.db_path(lg)
        if not p or "/test/" in p:
            continue
        rows = L.lidar_rows(p)
        t = L.guard_tokens(p, [r[0] for r in rows if r[0] in tokset])
        t = [x for x in t if f"{lg}/{x}" in cmd and x in yaw]
        if t:
            cand[lg] = {"db": p, "tokens": sorted(t)}
    print(f"cached {len(cached)}; eligible NEW logs {len(cand)}; tokens {sum(len(v['tokens']) for v in cand.values())}")
    m = json.load(open(f"{OLD}/meta/trainval_0-199.json"))
    l2a = {l: k.replace("openscene_sensor_trainval", "openscene_sensor_trainval_camera") for k, v in m.items() for l in v}
    tree = json.load(open(f"{OLD}/meta/hf_tree_openscene-v1.1_openscene_sensor_trainval_camera.json"))
    size = {x["path"].rsplit("/", 1)[1][:-4]: x["size"] for x in tree}
    held_c = [l for l in held_all if l in cand]
    held_missing = [l for l in held_all if l not in cand and l not in cached]
    sel = {}
    for lg in held_c:
        sel[lg] = ("heldout", l2a[lg])
    by_arch = {}
    for lg in cand:
        by_arch.setdefault(l2a[lg], []).append(lg)
    # local-complete train logs (no download)
    local = [lg for lg in sorted(cand) if lg not in held_all and all(any(os.path.exists(f"{d}/{lg}_{ch}.zip") for d in LOCAL_DIRS) for ch in L.CAMS)]
    for lg in local:
        if len(sel) < N_TOTAL:
            sel[lg] = ("train", "local")
    arch_sel = sorted({l2a[h] for h in held_c})
    for a in arch_sel:                                          # free riders of the forced held-out archives
        for lg in sorted(by_arch[a]):
            if lg not in sel and lg not in held_all and len(sel) < N_TOTAL:
                sel[lg] = ("train", a)
    while len(sel) < N_TOTAL:
        rest = [a for a in by_arch if a not in arch_sel and any(l not in sel and l not in held_all for l in by_arch[a])]
        if not rest:
            break
        a = max(rest, key=lambda a: len([l for l in by_arch[a] if l not in sel and l not in held_all]) / (size[a] / 1e9))
        for lg in sorted(by_arch[a]):
            if lg not in sel and lg not in held_all and len(sel) < N_TOTAL:
                sel[lg] = ("train", a)
        arch_sel.append(a)
    # a log's archive when it was chosen via 'local' is 'local'; archives list = those with any non-local log
    arch_sel = [a for a in arch_sel if any(v[1] == a for v in sel.values())]
    nt = sum(1 for v in sel.values() if v[0] == "train")
    nh = sum(1 for v in sel.values() if v[0] == "heldout")
    print(f"selected: train {nt}, heldout {nh} (held-out without a local DB / not navtrain-eligible: {held_missing}), archives {len(arch_sel)}")
    plan = {"version": 1, "n_train": nt, "n_heldout": nh, "archives": [], "logs": []}
    tot = 0
    os.makedirs(L.wp("plan"), exist_ok=True)
    for a in arch_sel:
        lgs = sorted(l for l, v in sel.items() if v[1] == a)
        need = {}
        for lg in lgs:
            fn = L.frame_names(cand[lg]["db"], cand[lg]["tokens"])
            for t, rel in fn.items():
                if rel:
                    for r in rel:
                        need[r] = lg
        json.dump(need, open(L.wp("plan", f"need_{a}.json"), "w"))
        tot += size[a]
        plan["archives"].append({"name": a, "url": L.HF_BASE + a + ".tgz", "bytes": size[a], "logs": lgs, "n_need_frames": len(need)})
    for lg, (sp, src) in sorted(sel.items(), key=lambda kv: (kv[1][0] != "train", kv[0])):
        c = cand[lg]
        fn = L.frame_names(c["db"], c["tokens"])
        toks_ok = [t for t in c["tokens"] if fn[t]]
        plan["logs"].append({"log": lg, "split": sp, "source": src, "db": c["db"], "n_tokens": len(toks_ok), "tokens": toks_ok, "n_nocam": len(c["tokens"]) - len(toks_ok)})
    plan["total_download_bytes"] = tot
    plan["n_tokens"] = sum(x["n_tokens"] for x in plan["logs"])
    plan["n_frames_needed"] = sum(a["n_need_frames"] for a in plan["archives"])
    json.dump(plan, open(L.wp("plan", "PLAN.json"), "w"), indent=0)
    with open(L.wp("plan", "DOWNLOAD_FILES.md"), "w", encoding="utf-8") as f:
        f.write("# x150: exact public files (HF dataset OpenDriveLab/OpenScene, public, no token needed)\n\n| archive | bytes | GB | logs | held-out | needed frames |\n|---|---|---|---|---|---|\n")
        for a in plan["archives"]:
            nh_ = sum(1 for l in a["logs"] if sel[l][0] == "heldout")
            f.write(f"| {a['url']} | {a['bytes']} | {a['bytes'] / 1e9:.2f} | {len(a['logs'])} | {nh_} | {a['n_need_frames']} |\n")
        f.write(f"\nTOTAL transfer (upper bound; the stream stops once all needed frames are found) {tot} B = {tot / 1e9:.2f} GB over {len(plan['archives'])} archives. "
                f"Logs {len(plan['logs'])} (train {nt}, heldout {nh}); local-complete logs (no download): {sum(1 for v in sel.values() if v[1] == 'local')}; "
                f"tokens {plan['n_tokens']}; frames to extract {plan['n_frames_needed']} (~{plan['n_frames_needed'] * 0.215 / 1024:.1f} GB peak if all kept; the chain deletes per log).\n")
    print(f"archives {len(arch_sel)}, transfer {tot / 1e9:.2f} GB, logs {len(plan['logs'])}, tokens {plan['n_tokens']}, frames {plan['n_frames_needed']}")
    if tot > CAP_BYTES:
        open(L.wp("plan", "STOP_OVER_200GB"), "w").write(str(tot))
        print("STOP: transfer exceeds 200 GB")
        sys.exit(2)


if __name__ == "__main__":
    main()
