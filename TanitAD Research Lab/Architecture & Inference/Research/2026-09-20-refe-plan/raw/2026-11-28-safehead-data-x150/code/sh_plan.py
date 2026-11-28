#!/usr/bin/env python3
"""Choose the logs and write PLAN.json + DOWNLOAD_FILES.md (CPU, stdlib; no download here).

Selection (PI approval 2026-11-26): 29 TRAIN logs + 6 HELD-OUT logs (sft_heldout_logs.txt), all from navtrain.yaml, each with a local nuPlan DB.
  * local-complete train logs first (4-camera frames already on disk: nuplan-camera/scenarios4 + mini)  -> no download
  * held-out: the 6 smallest OpenScene trainval camera archives that hold a held-out log with a local DB
  * train: every other eligible log in those archives, then archives by eligible-logs-per-GB until 29 train logs
Public source: HF dataset OpenDriveLab/OpenScene, openscene-v1.1/openscene_sensor_trainval_camera/openscene_sensor_trainval_camera_<i>.tgz
(log -> archive from openscene_sensor_trainval_0-199.json, same repo). Streamed (curl | tar), only the needed frames are extracted; nothing else is written.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sh_lib as L  # noqa: E402

N_TRAIN, N_HELD = 29, 6
LOCAL_DIRS = ["E:/Projects/TanitAD/data/nuplan-camera/scenarios4", "E:/Projects/TanitAD/data/nuplan-camera/mini"]


def main():
    logs, toks = L.read_yaml_split()
    held_all = [l.strip() for l in open("F:/safehead_cache/src/splits/sft_heldout_logs.txt") if l.strip()]
    cmd = json.load(open(L.COMMANDS))
    yaw = L.load_json_gz(L.YAW_TRAIN)["table"]
    tokset = set(toks)
    # navtest logs never enter (eval only)
    navtest_logs = set()      # navtest DBs live under splits/test/; a navtrain log whose DB is only there is skipped below
    cand = {}
    for lg in logs:
        p = L.db_path(lg)
        if not p or '/test/' in p or lg in navtest_logs:
            continue
        rows = L.lidar_rows(p)
        t = L.guard_tokens(p, [r[0] for r in rows if r[0] in tokset])
        t = [x for x in t if f"{lg}/{x}" in cmd and x in yaw]
        if t:
            cand[lg] = {"db": p, "tokens": sorted(t)}
    print(f"eligible logs (navtrain, local DB, tokens with command+yaw+margin): {len(cand)}; tokens {sum(len(v['tokens']) for v in cand.values())}")
    m = json.load(open(L.wp("meta", "trainval_0-199.json")))
    l2a = {l: k.replace("openscene_sensor_trainval", "openscene_sensor_trainval_camera") for k, v in m.items() for l in v}
    tree = json.load(open(L.wp("meta", "hf_tree_openscene-v1.1_openscene_sensor_trainval_camera.json")))
    size = {x["path"].rsplit("/", 1)[1][:-4]: x["size"] for x in tree}
    # local-complete logs: all four cameras zipped on disk
    local = []
    for lg in sorted(cand):
        ok = True
        for ch in L.CAMS:
            if not any(os.path.exists(f"{d}/{lg}_{ch}.zip") for d in LOCAL_DIRS):
                ok = False
        if ok:
            local.append(lg)
    held_c = [l for l in held_all if l in cand]
    by_arch = {}
    for lg in cand:
        by_arch.setdefault(l2a[lg], []).append(lg)
    sel = {}                                                   # log -> (split, source)
    for lg in local:
        sel[lg] = ("train", "local")
    # held-out: smallest archives holding a held-out candidate
    arch_sel, n_held = [], 0
    for a in sorted({l2a[h] for h in held_c}, key=lambda a: size[a]):
        if n_held >= N_HELD:
            break
        hl = [h for h in held_c if l2a[h] == a]
        if lg_taken := [h for h in hl if h in sel]:
            continue
        sel[hl[0]] = ("heldout", a)
        n_held += 1
        arch_sel.append(a)
    # train: every other eligible log in the held-out archives
    for a in arch_sel:
        for lg in sorted(by_arch[a]):
            if lg not in sel and lg not in held_all and sum(1 for v in sel.values() if v[0] == "train") < N_TRAIN:
                sel[lg] = ("train", a)
    # then the archives with the most eligible non-held logs per GB
    rest = sorted((a for a in by_arch if a not in arch_sel), key=lambda a: -(len([l for l in by_arch[a] if l not in held_all and l not in sel]) / (size[a] / 1e9)))
    for a in rest:
        nt = sum(1 for v in sel.values() if v[0] == "train")
        if nt >= N_TRAIN:
            break
        add = [l for l in sorted(by_arch[a]) if l not in sel and l not in held_all][:N_TRAIN - nt]
        if add:
            for lg in add:
                sel[lg] = ("train", a)
            arch_sel.append(a)
    nt = sum(1 for v in sel.values() if v[0] == "train")
    nh = sum(1 for v in sel.values() if v[0] == "heldout")
    print(f"selected: train {nt}, heldout {nh}, archives {len(arch_sel)}")
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
    json.dump(plan, open(L.wp("plan", "PLAN.json"), "w"), indent=0)
    with open(L.wp("plan", "DOWNLOAD_FILES.md"), "w", encoding="utf-8") as f:
        f.write("# Exact public files (HF dataset OpenDriveLab/OpenScene, public, no token needed)\n\n| archive | bytes | GB | logs | held-out | needed frames |\n|---|---|---|---|---|---|\n")
        for a in plan["archives"]:
            nh_ = sum(1 for l in a["logs"] if sel[l][0] == "heldout")
            f.write(f"| {a['url']} | {a['bytes']} | {a['bytes'] / 1e9:.2f} | {len(a['logs'])} | {nh_} | {a['n_need_frames']} |\n")
        f.write(f"\nTOTAL transfer {tot} B = {tot / 1e9:.2f} GB over {len(plan['archives'])} archives (largest {max(a['bytes'] for a in plan['archives']) / 1e9:.2f} GB; streamed, never stored). "
                f"Downloaded logs: {sum(1 for v in sel.values() if v[1] != 'local')}; local-complete logs (no download): {len(local)}.\n")
    print(f"archives {len(arch_sel)}, transfer {tot / 1e9:.2f} GB, logs {len(plan['logs'])}, tokens {plan['n_tokens']}")
    for x in plan["logs"]:
        print(x["split"], x["source"], x["log"], x["n_tokens"], x["n_nocam"])


if __name__ == "__main__":
    main()
