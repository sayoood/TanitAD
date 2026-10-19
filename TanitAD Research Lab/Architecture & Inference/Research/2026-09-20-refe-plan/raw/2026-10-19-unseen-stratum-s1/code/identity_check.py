"""S1 step 2-3: the IDENTITY check of the frozen rule on the navtrain logs that are in the val split, plus the RED arms.
READ-ONLY, CPU.  No label: the only 'truth' is the set of navtrain TOKENS per log (token table), which is a token list, not a score.

    python identity_check.py            -> raw/identity_214.json, logs/identity_214.log
"""
import collections
import glob
import gzip
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unseen_token_rule as R  # noqa: E402

PK = "C:/Users/Admin/par18/l1prep/work/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
DB = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval"
OUT = "C:/Users/Admin/par18/unseen_s1/raw/identity_214.json"
if len(sys.argv) > 2:           # python identity_check.py <db_dir> <tag>   e.g. the 248 top-level navtrain DBs (out-of-sample)
    DB, OUT = sys.argv[1], f"C:/Users/Admin/par18/unseen_s1/raw/identity_{sys.argv[2]}.json"


def load_known():
    by = collections.defaultdict(set)
    with gzip.open(PK + "/raw/rl_pool/navtrain_token_table.tsv.gz", "rt") as f:
        h = f.readline().rstrip("\n").split("\t")
        for l in f:
            r = dict(zip(h, l.rstrip("\n").split("\t")))
            if r["log"]:
                by[r["log"]].add(r["token"])
    return by


def dist(x):
    x = np.asarray(x, dtype=float)
    q = np.percentile(x, [0, 5, 25, 50, 75, 95, 100])
    return {"n": int(len(x)), "min": q[0], "p5": q[1], "p25": q[2], "median": q[3], "p75": q[4], "p95": q[5], "max": q[6], "mean": float(x.mean()),
            "n_eq_1": int((x >= 1 - 1e-12).sum()), "n_eq_0": int((x <= 1e-12).sum())}


def run_variant(recs, known, **kw):
    rec_l, prec_l, ncand, hits, nk = [], [], [], 0, 0
    per = {}
    for lg, rec in recs.items():
        c = R.candidates(rec, **kw)
        cs, ks = set(c["tokens"]), known[lg]
        h = len(cs & ks)
        hits += h
        nk += len(ks)
        ncand.append(len(cs))
        rec_l.append(h / len(ks))
        prec_l.append(h / len(cs) if cs else float("nan"))
        per[lg] = {"n_known": len(ks), "n_cand": len(cs), "hit": h}
    tot_c = int(sum(ncand))
    return {"params": {k: v for k, v in kw.items()}, "n_candidates_total": tot_c, "n_known_total": nk, "hits_total": hits,
            "recall_micro": hits / nk, "precision_micro": hits / tot_c if tot_c else None,
            "recall_per_log": dist(rec_l), "precision_per_log": dist([p for p in prec_l if p == p]), "n_cand_per_log": dist(ncand)}, per


def main():
    t0 = time.time()
    known = load_known()
    logs = sorted(l for l in known if os.path.exists(f"{DB}/{l}.db"))
    print("navtrain named logs", len(known), "with a trainval DB", len(logs), "known tokens in them", sum(len(known[l]) for l in logs), flush=True)
    recs = {lg: R.read_log(f"{DB}/{lg}.db") for lg in logs}
    print("read", len(recs), f"{time.time()-t0:.0f}s", flush=True)
    res = {"what": "S1 identity check of unseen_token_rule on the navtrain logs that sit in the val split (trainval DBs)", "evidence": "MEASURED (read-only DB + token table; no label)",
           "n_logs": len(logs), "n_known_tokens": sum(len(known[l]) for l in logs)}

    # ---- the frozen rule -------------------------------------------------------------------------------------------------
    frozen, per = run_variant(recs, known)
    res["frozen_rule"] = frozen
    # why does a known token miss?  (cadence / window edge / route)
    why = collections.Counter()
    miss_logs = collections.Counter()
    edge = {"k_min_known": 10**9, "k_max_gap_known": 10**9, "n_known_at_k3": 0, "n_known_at_kM11": 0}
    hist_start, hist_end = collections.Counter(), collections.Counter()   # known tokens by post-drop list position k (3..8) and by gap-from-end (10..15)
    offs = collections.Counter()
    for lg, rec in recs.items():
        fi = R.frame_indices(rec)
        c = R.candidates(rec)
        fpos = {int(i): k for k, i in enumerate(fi)}
        offs[int(fi[0]) % 10 if len(fi) else -1] += 1
        idx = {t: i for i, t in enumerate(rec["tokens"])}
        m = len(fi)
        cs = set(c["tokens"])
        for t in known[lg]:
            if t in cs:
                k = fpos[idx[t]]
                edge["k_min_known"] = min(edge["k_min_known"], k)
                edge["k_max_gap_known"] = min(edge["k_max_gap_known"], m - 1 - k)
                edge["n_known_at_k3"] += int(k == 3)
                edge["n_known_at_kM11"] += int(k == m - 11)
                hist_start[k] += 1
                hist_end[m - 1 - k] += 1
                continue
            miss_logs[lg] += 1
            i = idx.get(t)
            if i is None:
                why["token not a lidar_pc of the log DB"] += 1
            elif i not in fpos:
                why["off the 2 Hz grid (cadence)"] += 1
            else:
                k = fpos[i]
                if k < 3:
                    why["window: fewer than 3 history frames"] += 1
                elif k > m - 11:
                    why["window: fewer than 10 future frames"] += 1
                else:
                    why["no route (scene.roadblock_ids empty)"] += 1
    res["frozen_rule"]["miss_reasons"] = dict(why)
    res["frozen_rule"]["n_logs_with_a_miss"] = len(miss_logs)
    # the evidence for the dropped first grid frame: where do known tokens sit in the UNdropped grid (positions 0..7)?
    pre = collections.Counter()
    for lg, rec in recs.items():
        fi0 = R.frame_indices(rec, drop_first=0)
        pos = {int(i): k for k, i in enumerate(fi0)}
        idx = {t: i for i, t in enumerate(rec["tokens"])}
        for t in known[lg]:
            k = pos.get(idx[t])
            if k is not None and k < 8:
                pre[k] += 1
    res["frozen_rule"]["known_by_position_in_UNDROPPED_grid_(0..7)"] = {k: pre.get(k, 0) for k in range(8)}
    res["frozen_rule"]["edges_of_known_tokens"] = edge
    res["frozen_rule"]["known_by_position_from_start_(k=3..9)"] = {k: hist_start.get(k, 0) for k in range(3, 10)}
    res["frozen_rule"]["known_by_gap_from_end_(10..16)"] = {k: hist_end.get(k, 0) for k in range(10, 17)}
    res["frozen_rule"]["phase_offset_mod10_of_first_frame"] = dict(sorted(offs.items()))
    nfr = [R.candidates(r)["n_frames"] for r in recs.values()]
    res["frozen_rule"]["frames_per_log_M"] = dist(nfr)
    res["frozen_rule"]["route_dropped_windows_total"] = int(sum(R.candidates(r)["n_dropped_route"] for r in recs.values()))
    res["frozen_rule"]["navtrain_share_of_candidates_per_log"] = res["frozen_rule"]["precision_per_log"]

    # ---- RED arms --------------------------------------------------------------------------------------------------------
    red = {}
    # cadence: phase shifted by +-1 lidar_pc (off-by-one), a fixed phase 0 (no pairing), stride 9 / 11 / 20 / 5
    phases = {lg: R.first_paired(rec) for lg, rec in recs.items()}

    def with_phase(delta=0, fixed=None, **kw):
        out = {}
        for lg, rec in recs.items():
            ph = (fixed if fixed is not None else phases[lg] + delta)
            c = R.candidates(rec, phase=(ph if ph >= 0 else ph % 10), **kw)
            out[lg] = c
        return out

    def score(cmap):
        hits = sum(len(set(cmap[lg]["tokens"]) & known[lg]) for lg in recs)
        nk = sum(len(known[lg]) for lg in recs)
        nc = sum(len(cmap[lg]["tokens"]) for lg in recs)
        rl = [len(set(cmap[lg]["tokens"]) & known[lg]) / len(known[lg]) for lg in recs]
        return {"recall_micro": hits / nk, "precision_micro": hits / nc if nc else None, "n_candidates_total": nc, "recall_per_log_median": float(np.median(rl)), "recall_per_log_max": float(max(rl)), "n_logs_recall_ge_0.5": int(sum(1 for x in rl if x >= 0.5))}

    red["cadence_phase_plus1"] = score(with_phase(+1))
    red["cadence_phase_minus1"] = score(with_phase(-1))
    red["cadence_phase_plus5_(half_a_frame)"] = score(with_phase(+5))
    red["cadence_phase_fixed0_(ignore_camera_pairing)"] = score(with_phase(fixed=0))
    for st in (9, 11, 20, 5):
        out = {lg: R.candidates(rec, stride=st, phase=phases[lg], drop_first=1) for lg, rec in recs.items()}
        red[f"cadence_stride_{st}"] = score(out)
    # the two corrections made on the 214 logs, switched back off one at a time, and a wrong pairing tolerance
    red["phase_from_CAM_F0_only(the first version)"] = score({lg: R.candidates(rec, cams=("CAM_F0",)) for lg, rec in recs.items()})
    red["phase_from_CAM_L0_only"] = score({lg: R.candidates(rec, cams=("CAM_L0",)) for lg, rec in recs.items()})
    red["first_grid_frame_NOT_dropped(drop_first=0)"] = score({lg: R.candidates(rec, drop_first=0) for lg, rec in recs.items()})
    red["pairing_tolerance_40ms(every_lidar_pc_pairs)"] = score({lg: R.candidates(rec, tol_us=40_000) for lg, rec in recs.items()})
    red["pairing_tolerance_5ms(no_lidar_pc_pairs)"] = score({lg: R.candidates(rec, tol_us=5_000) for lg, rec in recs.items()})
    # window: history / future (frames) -- the count must shift by a fixed amount per log
    base_n = frozen["n_candidates_total"]
    for name, (h, f) in {"history_3": (3, 10), "history_5": (5, 10), "history_6": (6, 10), "future_9": (4, 9), "future_11": (4, 11), "future_8": (4, 8), "brief_2s_history_4s_future(5,8)": (5, 8), "history_8(4s)_future_10": (8, 10)}.items():
        out = {lg: R.candidates(rec, num_history=h, num_future=f) for lg, rec in recs.items()}
        s = score(out)
        s["shift_vs_frozen"] = s["n_candidates_total"] - base_n
        s["shift_per_log"] = s["shift_vs_frozen"] / len(recs)
        red["window_" + name] = s
    out = {lg: R.candidates(rec, has_route=False) for lg, rec in recs.items()}
    s = score(out)
    s["shift_vs_frozen"] = s["n_candidates_total"] - base_n
    red["route_filter_off"] = s
    res["red_arms"] = red
    res["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    json.dump(per, open(OUT.replace(".json", "_per_log.json"), "w"))
    f = res["frozen_rule"]
    print(json.dumps({k: f[k] for k in ("n_candidates_total", "n_known_total", "hits_total", "recall_micro", "precision_micro", "miss_reasons", "n_logs_with_a_miss", "edges_of_known_tokens", "phase_offset_mod10_of_first_frame", "known_by_position_from_start_(k=3..9)", "known_by_gap_from_end_(10..16)", "known_by_position_in_UNDROPPED_grid_(0..7)")}, indent=1))
    print("recall per log", f["recall_per_log"])
    print("precision per log", f["precision_per_log"])
    print("cand per log", f["n_cand_per_log"])
    for k, v in red.items():
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})


if __name__ == "__main__":
    main()
