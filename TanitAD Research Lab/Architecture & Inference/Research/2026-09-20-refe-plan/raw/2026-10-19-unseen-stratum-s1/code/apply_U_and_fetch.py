"""S1 step 4-5: apply the FROZEN rule to the 1,167 U logs (STRUCTURAL reads only: lidar_pc token/timestamp/scene, CAM_F0 + CAM_L0 image timestamps,
scene.roadblock_ids emptiness) and count candidate tokens per log; then list the exact jpg names a seeded 123-log x 4-token x 4-camera sample implies.
NO ego_pose, NO scenario_tag, NO label, NO score is read for U.  No network.  sqlite read-only.

    python apply_U_and_fetch.py     -> raw/U_candidates_per_log.json, raw/U_candidate_tokens.json.gz, raw/fetch_list_s2_s4.{json,txt}
"""
import gzip
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unseen_token_rule as R  # noqa: E402

DB = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval"
UNAMES = "C:/Users/Admin/par18/l2fresh/raw/scope_unseen_stratum_U_names.json"
NAVTRAIN_YAML = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/splits/navtrain.yaml"
RAW = "C:/Users/Admin/par18/unseen_s1/raw"
FETCH_SEED = 20261020          # PROVISIONAL: the S3 prereg fixes the real seed and sampler; this one only sizes S2 / S4
N_LOGS, N_TOK = 123, 4


def navtrain_yaml_logs():
    out, sec = set(), None
    for l in open(NAVTRAIN_YAML, encoding="utf-8"):
        if l.startswith("log_names"):
            sec = "L"
        elif l.startswith("tokens"):
            break
        elif sec == "L":
            m = re.match(r"\s*-\s*'?([^']+)'?", l)
            if m:
                out.add(m.group(1))
    return out


def main():
    t0 = time.time()
    U = json.load(open(UNAMES))["U_trainval_minus_navtrain"]
    ny = navtrain_yaml_logs()
    res = {"what": "the frozen token rule applied to U (1,167 val-split logs not named in the navtrain token table); structural reads only", "evidence": "MEASURED (read-only DB; no ego_pose, no scenario_tag, no label)",
           "n_U_logs": len(U), "U_logs_named_in_navtrain_yaml_log_names": len(set(U) & ny), "navtrain_yaml_log_names": len(ny)}
    per, toks, fail = {}, {}, []
    for n, lg in enumerate(U):
        try:
            rec = R.read_log(f"{DB}/{lg}.db")
            c = R.candidates(rec)
        except Exception as e:  # noqa: BLE001
            fail.append((lg, str(e)[:100]))
            continue
        per[lg] = {"n_lidar_pc": len(rec["tokens"]), "n_grid_frames": int(c["n_frames"]), "n_candidates": len(c["tokens"]), "n_dropped_route": int(c["n_dropped_route"]),
                   "phase_lidar_idx": (int(c["frame_idx"][0]) % 10 if len(c["frame_idx"]) else None)}
        toks[lg] = c["tokens"]
        if n % 200 == 0:
            print("ZZ", n, f"{time.time()-t0:.0f}s", flush=True)
    cnt = np.array([v["n_candidates"] for v in per.values()])
    res.update({"n_logs_ok": len(per), "n_logs_failed": len(fail), "failed": fail[:5], "n_candidates_total": int(cnt.sum()),
                "candidates_per_log": {"min": int(cnt.min()), "p5": float(np.percentile(cnt, 5)), "p25": float(np.percentile(cnt, 25)), "median": float(np.median(cnt)), "p75": float(np.percentile(cnt, 75)),
                                       "p95": float(np.percentile(cnt, 95)), "max": int(cnt.max()), "mean": float(cnt.mean())},
                "n_logs_with_zero_candidates": int((cnt == 0).sum()), "n_logs_lt_4_candidates": int((cnt < 4).sum()),
                "phase_mod10_counts": {str(k): int(sum(1 for v in per.values() if v["phase_lidar_idx"] == k)) for k in range(10)},
                "n_route_dropped_windows_total": int(sum(v["n_dropped_route"] for v in per.values()))})
    json.dump({"summary": res, "per_log": per}, open(f"{RAW}/U_candidates_per_log.json", "w"))
    with gzip.open(f"{RAW}/U_candidate_tokens.json.gz", "wt") as f:
        json.dump(toks, f)
    print(json.dumps(res, indent=1), flush=True)

    # ---- the fetch list: seeded 123 logs x 4 tokens x 4 cameras ------------------------------------------------------------
    ok_logs = sorted(lg for lg in per if per[lg]["n_candidates"] >= N_TOK)
    rng = np.random.default_rng(FETCH_SEED)
    chosen = [ok_logs[i] for i in sorted(rng.choice(len(ok_logs), size=N_LOGS, replace=False))]
    items, miss_tok = [], 0
    for lg in chosen:
        rec = R.read_log(f"{DB}/{lg}.db")
        c = R.candidates(rec)
        tsmap = {t: int(rec["ts"][i]) for t, i in zip(c["tokens"], c["lp_idx"])}
        r2 = np.random.default_rng([FETCH_SEED, int(hashlib.sha256(lg.encode()).hexdigest()[:8], 16)])
        order = r2.permutation(len(c["tokens"]))
        got = 0
        for j in order:
            t = c["tokens"][int(j)]
            fr = R.nearest_jpgs(f"{DB}/{lg}.db", tsmap[t])
            if fr is None:
                miss_tok += 1
                continue
            items.append({"log": lg, "token": t, "ts_us": tsmap[t], "jpgs": {ch: {"path": fr[ch][0], "dt_ms": round(fr[ch][1], 2)} for ch in R.CAMERAS}})
            got += 1
            if got == N_TOK:
                break
    paths = [it["jpgs"][ch]["path"] for it in items for ch in R.CAMERAS]
    lst = sorted(set(paths))
    fl = {"what": "the jpgs S2 / S4 would fetch for a SEEDED PROVISIONAL sample of 123 U logs x 4 tokens x 4 cameras (CAM_F0, CAM_L0, CAM_R0, CAM_B0 = the cameras of Step A's bundle: refe/planner.py:54 CAMERAS, make_stepA_bundle.py:82). The final S4 list is drawn by S3's SEALED sampler; this one sizes S2/S4 and names the archives to look up.",
          "evidence": "MEASURED names and dt from the DB (structural reads only); size ESTIMATED", "seed": FETCH_SEED, "n_logs": len(chosen), "tokens_per_log": N_TOK, "n_tokens": len(items), "n_jpgs": len(paths), "n_unique_paths": len(lst),
          "tokens_skipped_for_unresolvable_frames": miss_tok, "max_dt_ms": max(it["jpgs"][ch]["dt_ms"] for it in items for ch in R.CAMERAS), "path_layout": "<log>/<CAM>/<hash>.jpg (DB image.filename_jpg; same layout as raw/readtwo_fetch_list.txt)",
          "list_sha256": hashlib.sha256("\n".join(lst).encode()).hexdigest(), "logs": chosen, "items": items}
    json.dump(fl, open(f"{RAW}/fetch_list_s2_s4.json", "w"))
    open(f"{RAW}/fetch_list_s2_s4.txt", "w", newline="\n").write("\n".join(lst) + "\n")
    print("fetch list", {k: v for k, v in fl.items() if k not in ("items", "logs")}, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
