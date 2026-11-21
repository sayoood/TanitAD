#!/usr/bin/env python3
"""TF-2b rollouts (PREREG_TF2B.md): the registered teacher protocol (eval/teacher_navtest_seam.roll, 10 Hz, CPU, rank 0,
step 0, per-token scenario) on navtrain held-out samples.  Runs inside eval/ through eval/run_p4.py --script.
  --samples <tf2b_samples.txt> --shard i/N --out <jsonl>      the fresh 687 (resumable; one JSON row per sample)
  --control --p5s <p5s_cubic.npz> --out <json> [--limit n]    C0: re-roll every 34th banked sample (30 of 1,018) and
                                                              compare with the banked t20 (PREREG_P4 C3 bar)
Row: {log, token, traj20 [20,3], buffer_s, history_clamped, v0, sec}  or  {log, token, error}.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import teacher_navtest_seam as S  # noqa: E402


def groups(keys):
    by = {}
    for lg, tok in keys:
        by.setdefault(lg, []).append(tok)
    return by


def run(keys, on_row):
    NS = S._S["NS"]
    dbs = NS.index_dbs(db_root=os.path.dirname(S.TRAINVAL_DB))          # the DBs sit in several split folders
    for lg, toks in sorted(groups(keys).items()):
        db = dbs.get(lg, "")
        if not os.path.exists(db):
            for tok in sorted(toks):
                on_row({"log": lg, "token": tok, "error": "no local DB"})
            continue
        pos = S._log_positions(db)
        scs = {sc.scenario_name: sc for sc in NS.build_scenarios_for_log(db, sorted(toks))}
        for tok in sorted(toks):
            rec = {"log": lg, "token": tok}
            try:
                t1 = time.time()
                g = S.roll(scs[tok], pos.get(tok))
                rec.update({"traj20": np.asarray(g["traj20"], np.float64).tolist(), "buffer_s": g["buffer_s"],
                            "history_clamped": g["history_clamped"], "v0": g["v0"], "sec": round(time.time() - t1, 2)})
            except Exception as e:                                               # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            on_row(rec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--p5s")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", default="1")
    a = ap.parse_args()
    S._init("cpu", int(a.threads), 0)
    if a.control:
        z = np.load(a.p5s)
        allk = sorted(zip(z["logs"].tolist(), z["tokens"].tolist()))
        ref = {(l, t): z["labels20_t20_tc20_tw"][i, 0] for i, (l, t) in enumerate(zip(z["logs"].tolist(), z["tokens"].tolist()))}
        keys = allk[::34]
        if a.limit:
            keys = keys[:a.limit]
        out = []

        def on(rec):
            if "traj20" in rec:
                d = np.linalg.norm(np.asarray(rec["traj20"])[:, :2] - ref[(rec["log"], rec["token"])][:, :2], axis=1)
                rec["max_dxy_m"] = float(d.max())
            out.append(rec)
            print(json.dumps({k: rec.get(k) for k in ("token", "max_dxy_m", "sec", "error")}), flush=True)
        t0 = time.time()
        run(keys, on)
        ok = [r for r in out if "max_dxy_m" in r]
        m = np.array([r["max_dxy_m"] for r in ok]) if ok else np.array([np.inf])
        rep = {"what": "C0: my rollout vs the BANKED rank-0 10 Hz target, max |dxy| over 20 poses (PREREG_P4 C3 bar)",
               "n": len(out), "n_ok": len(ok), "n_error": len(out) - len(ok), "median_max_dxy_m": float(np.median(m)),
               "p90_max_dxy_m": float(np.percentile(m, 90)), "max_max_dxy_m": float(m.max()),
               "share_le_0.25": float((m <= 0.25).mean()), "seconds": round(time.time() - t0, 1), "rows": out}
        rep["pass"] = bool(rep["n_error"] == 0 and rep["median_max_dxy_m"] <= 0.05 and rep["share_le_0.25"] >= 0.90)
        json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)
        print("ZZTF2B_C0", "PASS" if rep["pass"] else "FAIL", rep["n_ok"], rep["median_max_dxy_m"], rep["share_le_0.25"])
        return 0
    keys = [tuple(x.strip().split(":")) for x in open(a.samples, encoding="utf-8") if x.strip()]
    si, sn = (int(x) for x in a.shard.split("/"))
    keys = keys[si::sn]
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                done.add((r["log"], r["token"]))
    keys = [k for k in keys if k not in done]
    if a.limit:
        keys = keys[:a.limit]
    n = [0]
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as fo:
        def on(rec):
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            n[0] += 1
        run(keys, on)
    print("ZZTF2B_ROLL_DONE", a.shard, n[0], round(time.time() - t0, 1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
