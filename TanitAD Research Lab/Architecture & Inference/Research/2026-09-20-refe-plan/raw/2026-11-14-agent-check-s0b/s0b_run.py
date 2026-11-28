"""S0-B driver.  `--smoke`: ONE log, prints/writes ONLY counts, timings, shapes and control booleans (NO gain / capture / precision
numbers).  Full mode (MM only, after the freeze): writes s0b_result.json + S0B_RESULT.md + per-set flag arrays to --out.

  python s0b_run.py --smoke --out <dir>
  python s0b_run.py --out <dir> [--cid-json <s0b_cid result>]

CPU only; one process; CUDA_VISIBLE_DEVICES=-1.  Reads (read-only): the fit pack and local nuPlan DBs (mode=ro&immutable=1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import s0b_raster as RS  # noqa: E402
import s0b_readout as RO  # noqa: E402

PACK = "F:/tanitad_scratch/p6fit/fit_pack.npz"
PACK_SHA_PREFIX = "e58653c5"
DB_ROOTS = ["E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval/"]
ARMS = ("R-true", "R-t0", "R-cv", "R-dyn", "R-true-2Hz", "C-SHUF", "C-REG-zero", "C-REG-one")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def db_path(log):
    for r in DB_ROOTS:
        if os.path.exists(r + log + ".db"):
            return r + log + ".db"
    return None


def covered_sets(pack, found_logs):
    """Sorted (log, token) list of fit-pack sets with a local DB and banked truth -> row indices in that order."""
    have = set(found_logs)
    logs, toks = pack["log"], pack["token"]
    hp = pack["has_pdm"]
    rows = [i for i in range(len(logs)) if str(logs[i]) in have and bool(hp[i]) and db_path(str(logs[i])) is not None]
    rows.sort(key=lambda i: (str(logs[i]), str(toks[i])))
    return rows


def process_log(log, rows, pack, fp, out, t_acc, shuf_partner, want_shuf=True):
    """Fills out[arm] = {'nc': (n,64), 'ttc': (n,64)}, t0 overlap, status for the sets `rows` of one log (order = rows)."""
    t = time.time()
    ldb = RS.LogDB(db_path(log))
    t_acc["db_load"] += time.time() - t
    i0s = [ldb.pc_index.get(str(pack["token"][r])) for r in rows]
    tracks = {}

    def get_tracks(j):
        if j not in tracks:
            tracks[j] = RS.gather_set_tracks(ldb, i0s[j]) if i0s[j] is not None else None
        return tracks[j]

    res = []
    for j, r in enumerate(rows):
        rec = {"status": "ok"}
        t = time.time()
        st = get_tracks(j)
        t_acc["query"] += time.time() - t
        if st is None:
            rec["status"] = "no_token" if i0s[j] is None else "short_future"
            res.append(rec)
            continue
        t = time.time()
        poses = RO.proposals_10hz(pack["traj_xy"][r], pack["yaw"][r])
        ft = RO.footprint_tables(poses, fp)
        t_acc["footprint"] += time.time() - t
        rec["ft"] = {"partial": ft.partial_share, "none_in_grid": ft.none_in_grid_share, "cells": ft.n_cells_mean,
                     "k_nc": int(ft.idx_nc.shape[-1]), "k_ttc": int(ft.idx_ttc.shape[-1])}
        t = time.time()
        arms = RS.build_arms(st)
        t_acc["raster"] += time.time() - t
        arms["C-REG-zero"] = RS.reg_raster("zero")
        arms["C-REG-one"] = RS.reg_raster("one")
        if want_shuf and shuf_partner is not None and shuf_partner[j] >= 0:
            t = time.time()
            stp = get_tracks(int(shuf_partner[j]))
            if stp is not None:
                arms["C-SHUF"] = RS.build_arms(stp, which=("R-true",))["R-true"]
            t_acc["raster"] += time.time() - t
        t = time.time()
        rec["flags"] = {}
        for arm, Rg in arms.items():
            nc, ttc, o0 = RO.flags_for_raster(Rg, ft)
            rec["flags"][arm] = (nc, ttc, o0)
        t_acc["readout"] += time.time() - t
        rec["n_tracks"] = int(st.raw_x.shape[0])
        rec["raster_shape"] = tuple(arms["R-true"].shape)
        res.append(rec)
    return res


def run(args):
    t_all = time.time()
    pack_sha = sha256_file(PACK)
    assert pack_sha.startswith(PACK_SHA_PREFIX), f"fit pack sha mismatch {pack_sha}"
    pack = np.load(PACK, allow_pickle=False)
    pack = {k: pack[k] for k in ("log", "token", "traj_xy", "yaw", "logits", "pdm", "pdm_names", "is_heldout", "has_pdm")}
    names = [str(n) for n in pack["pdm_names"]]
    found = [l.strip() for l in open(HERE / "found_logs.txt", encoding="utf-8") if l.strip()]
    rows_all = covered_sets(pack, found)
    sel = RO.load_sel_common(args.sel_dir)
    fp = RO.read_pacifica()
    logs_all = np.array([str(pack["log"][r]) for r in rows_all])
    by_log = {}
    for j, l in enumerate(logs_all):
        by_log.setdefault(l, []).append(j)
    use_logs = sorted(by_log)
    if args.smoke or args.fullpath:
        use_logs = [next(l for l in use_logs if len(by_log[l]) >= 3)]
    t_acc = dict(db_load=0.0, query=0.0, footprint=0.0, raster=0.0, readout=0.0)
    recs = {}
    for li, log in enumerate(use_logs):
        js = by_log[log]
        rows = [rows_all[j] for j in js]
        recs[log] = (js, process_log(log, rows, pack, fp, None, t_acc, GLOBAL_PARTNER_LOCAL(rows_all, logs_all, js, args.smoke or args.fullpath)))
        if not args.smoke:
            print(f"[{li + 1}/{len(use_logs)}] {log} sets={len(js)} t={time.time() - t_all:.0f}s", flush=True)
    if args.smoke:
        return smoke_report(args, recs, t_acc, rows_all, by_log, t_all)
    return full_report(args, pack, names, sel, fp, rows_all, logs_all, recs, t_acc, t_all, pack_sha)


_PARTNERS = {}


def GLOBAL_PARTNER_LOCAL(rows_all, logs_all, js, smoke):
    """Per-log partner indices (LOCAL to the log's set list) from ONE global derangement stream (rng 20261114, sorted logs)."""
    if "p" not in _PARTNERS:
        _PARTNERS["p"] = RO.shuffle_partners(logs_all, seed=20261114)
    p = _PARTNERS["p"]
    if smoke:                                   # smoke: one log; partner stream is irrelevant to results, only needs to exist
        loc = RO.shuffle_partners(logs_all[js], seed=20261114)
        return loc
    first = js[0]
    return np.where(p[js] >= 0, p[js] - first, -1)


def smoke_report(args, recs, t_acc, rows_all, by_log, t_all):
    (log, (js, res)), = recs.items()
    ok = [r for r in res if r["status"] == "ok"]
    n = len(ok)
    creg_exact = all((not r["flags"]["C-REG-zero"][0].any()) and (not r["flags"]["C-REG-zero"][1].any()) and r["flags"]["C-REG-one"][0].all()
                     for r in ok)
    rep = {"smoke": True, "log": log, "covered_sets_total": len(rows_all), "covered_logs_total": len(by_log),
           "sets_in_smoke_log": len(js), "sets_ok": n, "status_counts": {s: sum(r["status"] == s for r in res) for s in ("ok", "no_token", "short_future")},
           "n_tracks_median": float(np.median([r["n_tracks"] for r in ok])) if ok else None,
           "raster_gather_table_shape": list(ok[0]["raster_shape"]) if ok else None,
           "footprint_cells_k_nc_max": max(r["ft"]["k_nc"] for r in ok), "footprint_cells_k_ttc_max": max(r["ft"]["k_ttc"] for r in ok),
           "c_shuf_sets": sum("C-SHUF" in r["flags"] for r in ok), "creg_controls_structurally_ok": bool(creg_exact),
           "timing_s": {k: round(v, 3) for k, v in t_acc.items()}, "wall_s": round(time.time() - t_all, 2)}
    per_set = (t_acc["query"] + t_acc["footprint"] + t_acc["raster"] + t_acc["readout"]) / max(n, 1)
    nlogs = len(by_log)
    est = per_set * len(rows_all) + t_acc["db_load"] * nlogs
    rep["per_set_s"] = round(per_set, 3)
    rep["ESTIMATED_full_cpu_s"] = round(est)
    rep["ESTIMATED_full_cpu_h"] = round(est / 3600, 2)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "smoke_report.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    return 0


def full_report(args, pack, names, sel, fp, rows_all, logs_all, recs, t_acc, t_all, pack_sha):
    S = len(rows_all)
    ix = {n: i for i, n in enumerate(names)}
    R = np.array(rows_all)
    ok = np.zeros(S, bool)
    flags = {a: [np.zeros((S, 64), bool), np.zeros((S, 64), bool)] for a in ARMS}
    have = {a: np.zeros(S, bool) for a in ARMS}
    o0 = {a: np.zeros(S, bool) for a in ARMS}
    fts = np.full((S, 3), np.nan)
    status = {}
    for log, (js, res) in recs.items():
        for j, rec in zip(js, res):
            status[rec["status"]] = status.get(rec["status"], 0) + 1
            if rec["status"] != "ok":
                continue
            ok[j] = True
            fts[j] = (rec["ft"]["partial"], rec["ft"]["none_in_grid"], rec["ft"]["cells"])
            for a, (nc, ttc, o) in rec["flags"].items():
                flags[a][0][j], flags[a][1][j] = nc, ttc
                have[a][j] = True
                o0[a][j] = o
    logits = pack["logits"][R].astype(np.float32)
    rule = sel.agg_torch(logits, lane=False)
    p6 = np.argmax(rule, axis=-1)
    pdm = pack["pdm"][R].astype(np.float64)
    nc_t, ttc_t, pd_t = pdm[..., ix["nc"]], pdm[..., ix["ttc"]], pdm[..., ix["pdms"]]
    ho = pack["is_heldout"][R].astype(bool)
    cid = {"status": "NOT RUN"}
    if args.cid_json and os.path.exists(args.cid_json):
        cid = json.load(open(args.cid_json, encoding="utf-8"))
    cid_status = cid.get("status", "NOT RUN")
    result = {"pack_sha256": pack_sha, "footprint": vars(fp), "n_covered_sets": S, "status_counts": status, "c_id": cid,
              "columns": {}, "outcome": {}, "timing_s": {k: round(v, 1) for k, v in t_acc.items()}}
    for cname, cm in (("all_covered", ok), ("heldout_covered", ok & ho)):
        m = np.nonzero(cm)[0]
        if len(m) == 0:
            result["columns"][cname] = {"n_sets": 0}
            result["outcome"][cname] = {"row": "NOT RUN", "reasons": ["no sets in this column"]}
            continue
        lg = logs_all[m]
        O = RO.oracle_gains(rule[m], pd_t[m], nc_t[m], ttc_t[m], p6[m])
        col = {"O": {"NC+TTC": float(O["both"].mean()), "NC": float(O["nc_only"].mean()), "TTC": float(O["ttc_only"].mean())}}
        gains = {"both": {}, "nc_only": {}, "ttc_only": {}}
        allf, t0s, fid = {}, {}, {}
        for a in ARMS:
            mm = m[have[a][m]]
            gset = RO.set_gains(rule[mm], pd_t[mm], p6[mm], flags[a][0][mm], flags[a][1][mm])
            for k in gains:
                gains[k][a] = (mm, gset[k])
            allf[a] = float(gset["all_flagged"].mean()) if len(mm) else None
            t0s[a] = float(o0[a][mm].mean()) if len(mm) else None
            fl = flags[a][0][mm] | flags[a][1][mm]
            fid[a] = RO.gfid(fl, nc_t[mm], ttc_t[mm], rule[mm])
        rep = {}
        for k, oname in (("both", "both"), ("nc_only", "nc_only"), ("ttc_only", "ttc_only")):
            part = {}
            for a in ARMS:
                mm, g = gains[k][a]
                if len(mm) == 0:
                    part[a] = {"n_sets": 0, "gain": None, "capture": None}
                    continue
                Oa = RO.oracle_gains(rule[mm], pd_t[mm], nc_t[mm], ttc_t[mm], p6[mm])[oname]
                part[a] = RO.column_report(logs_all[mm], {a: g}, Oa)[a]
                part[a]["n_sets"] = int(len(mm))
            rep[k] = part
        col.update({"arms_both": rep["both"], "arms_nc_only_vs_O_NC": rep["nc_only"], "arms_ttc_only_vs_O_TTC": rep["ttc_only"],
                    "all_flagged_share": allf, "footprint_overlap_at_t0_share": t0s, "g_fid": fid,
                    "n_sets": int(len(m)), "n_logs": int(len(set(lg.tolist()))),
                    "c_shuf_excluded_single_set_log_sets": int(len(m) - have["C-SHUF"][m].sum()),
                    "footprint_partial_out_of_grid_share": float(np.nanmean(fts[m, 0])),
                    "footprint_none_in_grid_share": float(np.nanmean(fts[m, 1]))})
        flat = {a: rep["both"][a] for a in ARMS}
        row, why = RO.outcome_row(flat, cid_status)
        result["columns"][cname] = col
        result["outcome"][cname] = {"row": row, "reasons": why}
    result["PRIMARY_COLUMN_NOTE"] = "outcome for BOTH columns is reported; the card names the 3,976 covered sets (all_covered); the held-out column is the clean read (MM Q-A). MM confirms which decides at the freeze."
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "s0b_result.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(out / "s0b_flags.npz", rows=R, ok=ok, logs=logs_all, **{f"{a}__nc": np.packbits(flags[a][0], axis=1) for a in ARMS},
                        **{f"{a}__ttc": np.packbits(flags[a][1], axis=1) for a in ARMS})
    with open(out / "S0B_RESULT.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(render_md(result))
    if args.fullpath:
        print("FULLPATH-SELFTEST OK (code path executed on ONE log; numbers not printed; delete the out dir)", sorted(result["columns"]), f"{time.time() - t_all:.0f}s")
    else:
        print("DONE", {k: v["row"] for k, v in result["outcome"].items()}, f"{time.time() - t_all:.0f}s")
    return 0


def render_md(res):
    L = ["# S0-B result (MEASURED, dev-box CPU; navtrain fit-pack covered sets; PDMS x 100 vs P6's pick)", "",
         f"pack sha256 {res['pack_sha256'][:16]}..., sets {res['n_covered_sets']}, status {res['status_counts']}, C-ID {res['c_id'].get('status')}", ""]
    for c, col in res["columns"].items():
        if not col.get("n_sets"):
            L += [f"## {c}: no sets", ""]
            continue
        L += [f"## {c} (n={col['n_sets']}, logs={col['n_logs']})  outcome: **{res['outcome'][c]['row']}** {res['outcome'][c]['reasons']}",
              f"oracle O: {col['O']}", "", "| arm | gain [95% CI] | capture vs O_NC+TTC [95% CI] | all-flagged |", "|---|---|---|---|"]
        for a, r in col["arms_both"].items():
            L.append(f"| {a} | {r['gain']:+.4f} {r['gain_ci']} | {r['capture']} {r['capture_ci']} | {col['all_flagged_share'][a]} |")
        L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fullpath", action="store_true", help="selftest of the full-mode code path on ONE log; prints no numbers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sel-dir", default=os.environ.get("S0B_SEL_DIR", str(HERE / "ref")))
    ap.add_argument("--cid-json", default=None)
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
