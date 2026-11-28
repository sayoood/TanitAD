#!/usr/bin/env python3
"""L2-FRESH-STEPA scoping probe for the UNSEEN-STRATUM DATA JOB (Amendment 2, branch "data job"): where would frames for the A1.5(ii) logs come from?  READ-ONLY, CPU: directory listings, the token table
and (for a small sample) READ-ONLY nuPlan DB opens.  No download, no network, no pod, no write outside C:/Users/Admin/par18/l2fresh/raw/.

LOCATION 1 (local): the nuPlan DB name sets on this box (trainval/, test/, top-level, extra), the navtrain log set (token table), the candidate set U = trainval DBs NOT named in the navtrain token table, and every
local FRAME root (listing only) matched against U.   LOCATION 2 (remote/recorded): what the repo's own data scripts say about where sensor blobs live (cited by file:line in the report), and the OpenScene
navtrain archive's own member list on disk (`openscene_probe/members.txt`), matched against U.  (The HF dataset cards are read separately; this script touches no network.)

    python scope_unseen_stratum.py
"""
import collections
import glob
import gzip
import json
import os
import sqlite3
import sys

PK = "C:/Users/Admin/par18/l1prep/work/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
DB = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
EXTRA = "C:/Users/Admin/p2heldout_work/dbs"
DATA = "D:/Projects/TanitAD/data"
OUT = "C:/Users/Admin/par18/l2fresh/raw/scope_unseen_stratum.json"
FRAME_ROOTS = {"refe_proxy/pixels": f"{DATA}/refe_proxy/pixels", "refe_m5/pixels": f"{DATA}/refe_m5/pixels", "nuplan-camera/mini": f"{DATA}/nuplan-camera/mini", "nuplan-camera/scenarios": f"{DATA}/nuplan-camera/scenarios",
               "nuplan-camera/scenarios4": f"{DATA}/nuplan-camera/scenarios4", "nuplan-camera/probe": f"{DATA}/nuplan-camera/probe", "refe_navtrain": f"{DATA}/refe_navtrain", "refe_navtrain10": f"{DATA}/refe_navtrain10",
               "refe_navtest": f"{DATA}/refe_navtest", "refe_m5r": f"{DATA}/refe_m5r", "refe_consistent": f"{DATA}/refe_consistent", "heldout_frames": "C:/Users/Admin/p2heldout_work/frames_heldout", "videos": f"{DATA}/videos"}
CAMS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0")


def names(pattern):
    return {os.path.basename(p)[:-3] for p in glob.glob(pattern)}


def rd(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        h = f.readline().rstrip("\n").split("\t")
        for l in f:
            yield dict(zip(h, l.rstrip("\n").split("\t")))


def log_of_member(n):
    """a frame-root entry name -> the nuPlan log name it holds: '<log>_CAM_F0.zip' or '<log>' or '<log>.zip'"""
    n = os.path.basename(n.rstrip("/"))
    for c in CAMS:
        if n.endswith(f"_{c}.zip"):
            return n[:-len(f"_{c}.zip")]
    return n[:-4] if n.endswith(".zip") else n


def main():
    tv, tl, te = names(f"{DB}/trainval/*.db"), names(f"{DB}/*.db"), names(f"{DB}/test/*.db")
    ex = names(f"{EXTRA}/*.db")
    rows = list(rd(PK + "/raw/rl_pool/navtrain_token_table.tsv.gz"))
    nt = {r["log"] for r in rows if r["log"]}
    nt_logless_tokens = sum(1 for r in rows if not r["log"])
    held = set(open(PK + "/splits/sft_heldout_logs.txt").read().split())
    pdm = {r["log"] for r in rows if r["log"] and r["heldout_log"] == "0" and int(r["served_train_sets"] or 0) > 0}
    res = {"what": "scoping probe for the unseen-stratum data job (read-only, no network)", "evidence": "MEASURED (directory listings + token table; nothing downloaded)"}
    res["db_name_sets"] = {"splits/trainval": len(tv), "splits/test": len(te), "splits/top_level": len(tl), "extra_p2heldout_work/dbs": len(ex), "union_all": len(tv | te | tl | ex)}
    res["navtrain_token_table"] = {"rows": len(rows), "named_logs": len(nt), "logless_tokens": nt_logless_tokens, "heldout_24": len(held), "heldout_in_named": len(held & nt), "M_PDM_logs_served_train_sets_gt0": len(pdm)}
    res["overlaps"] = {"navtrain_named_in_trainval": len(nt & tv), "navtrain_named_in_top_level": len(nt & tl), "navtrain_named_in_test": len(nt & te), "navtrain_named_in_extra": len(nt & ex),
                       "navtrain_named_in_NO_local_db": len(nt - (tv | tl | te | ex)), "trainval_and_top_level": len(tv & tl), "trainval_and_test": len(tv & te)}
    U = sorted(tv - nt)
    U_tl = sorted(tl - nt - tv)
    U_all = sorted((tv | tl | ex) - nt)
    res["candidate_sets"] = {"U_trainval_minus_navtrain (A1.5(ii) by name)": len(U), "top_level_minus_navtrain_minus_trainval": len(U_tl), "(trainval|top_level|extra)_minus_navtrain": len(U_all),
                             "the_number_1167_in_the_ruling_and_A1.5": 1167, "len(U)==1167": len(U) == 1167, "M_PDM_logs==1167": len(pdm) == 1167}
    # LOCATION 1: every local frame root, LISTING ONLY (depth 1), matched against U and against navtrain
    Uset, NTset = set(U_all), nt
    roots = {}
    for k, p in FRAME_ROOTS.items():
        if not os.path.isdir(p):
            roots[k] = {"exists": False}
            continue
        try:
            ents = os.listdir(p)
        except Exception as e:  # noqa: BLE001
            roots[k] = {"exists": True, "error": str(e)}
            continue
        lg = {log_of_member(e) for e in ents}
        roots[k] = {"exists": True, "entries": len(ents), "distinct_log_like_names": len(lg), "names_in_U": len(lg & Uset), "names_in_navtrain": len(lg & NTset),
                    "names_in_heldout_24": len(lg & held), "example": sorted(ents)[:2]}
    res["location_1_local_frame_roots"] = roots
    # LOCATION 2a: the OpenScene navtrain archive that was probed on this box: its member list is on disk
    mp = f"{DATA}/openscene_probe/members.txt"
    if os.path.exists(mp):
        lg2 = set()
        for line in open(mp, encoding="utf-8", errors="replace"):
            parts = line.rstrip("\n").split("/")
            if len(parts) >= 2 and parts[1]:
                lg2.add(parts[1])
        res["location_2a_openscene_navtrain_current_25_members"] = {"distinct_logs": len(lg2), "in_U": len(lg2 & Uset), "in_navtrain": len(lg2 & NTset), "in_heldout_24": len(lg2 & held)}
    # a small READ-ONLY look at U's DBs: do they carry the four camera channels with image rows (i.e. is a per-log frame fetch list derivable from the DB alone)?
    sample = U[:: max(1, len(U) // 24)][:24]
    probe = []
    for lg in sample:
        p = f"{DB}/trainval/{lg}.db"
        try:
            c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            ch = {r[0]: r[1] for r in c.execute("SELECT c.channel, COUNT(i.token) FROM camera c LEFT JOIN image i ON i.camera_token=c.token GROUP BY c.channel")}
            n_lpc = c.execute("SELECT COUNT(*) FROM lidar_pc").fetchone()[0]
            c.close()
            probe.append({"log": lg, "lidar_pc_rows": n_lpc, "image_rows_by_channel": {k: ch.get(k, 0) for k in CAMS}, "size_mb": round(os.path.getsize(p) / 1e6, 1)})
        except Exception as e:  # noqa: BLE001
            probe.append({"log": lg, "error": str(e)[:120]})
    res["db_sample_of_U"] = {"n": len(probe), "all_four_channels_have_images": all(("error" not in x and all(x["image_rows_by_channel"][c] > 0 for c in CAMS)) for x in probe), "rows": probe}
    res["U_first_5"] = U[:5]
    json.dump(res, open(OUT, "w"), indent=1)
    json.dump({"U_trainval_minus_navtrain": U}, open(OUT.replace(".json", "_U_names.json"), "w"))
    print(json.dumps({k: v for k, v in res.items() if k != "db_sample_of_U"}, indent=1))
    print("DB sample all four channels:", res["db_sample_of_U"]["all_four_channels_have_images"], "n", len(probe))


if __name__ == "__main__":
    sys.exit(main())
