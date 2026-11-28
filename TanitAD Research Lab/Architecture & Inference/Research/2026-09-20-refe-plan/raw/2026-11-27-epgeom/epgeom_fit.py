#!/usr/bin/env python3
"""EP-GEOM fit (PREREG_EPGEOM.md section 3): fit (surrogate, tau) on the navtrain cache TRAIN split only, check on the 6 held-out logs.
Reads ONLY props8, logits, pick, tr_* from F:/safehead_cache/out/<log>.npz (never `s`). Writes FROZEN_PARAMS.json + FIT_REPORT.json. No navtest file is opened.

    epgeom_fit.py --index F:/TanitAD_safehead_tmp/adapted/index.csv --out-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os

import numpy as np

import epgeom_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
ARR = ("logits", "props8", "pick", "tr_pdms", "tr_nc", "tr_dac", "tr_ttc", "tr_ep", "tr_c", "tr_ddc")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load(index):
    rows = list(csv.DictReader(open(index, newline="")))
    files = {}
    for r in rows:
        files.setdefault(r["file"], set()).add(r["token"])
    out = {k: [] for k in ARR}
    out["token"], out["log"], out["split"] = [], [], []
    for f in sorted(files):
        z = np.load(f)
        tok = z["token"]
        keep = np.array([t in files[f] for t in tok])
        for k in ARR:
            out[k].append(z[k][keep])
        out["token"].append(tok[keep])
        out["log"] += [str(z["log"])] * int(keep.sum())
        out["split"] += [str(z["split"])] * int(keep.sum())
    for k in ARR + ("token",):
        out[k] = np.concatenate(out[k])
    out["log"], out["split"] = np.array(out["log"]), np.array(out["split"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="F:/TanitAD_safehead_tmp/adapted/index.csv")
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    d = load(a.index)
    tr, ho = d["split"] == "train", d["split"] == "heldout"
    best, table = L.fit(d["logits"][tr], d["props8"][tr], d["pick"][tr], d["tr_pdms"][tr])
    base_tr = float(L.pdms_at(d["tr_pdms"][tr], d["pick"][tr]).mean())
    base_ho = float(L.pdms_at(d["tr_pdms"][ho], d["pick"][ho]).mean())
    agg_agree = float((L.aggregate_v1(d["logits"]).argmax(1) == d["pick"]).mean())      # CTRL-AGG (descriptive)

    def run(m, **kw):
        return L.select(d["logits"][m], d["props8"][m], d["pick"][m], best["surrogate"], best["tau"], **kw)

    pk_tr, pk_ho = run(tr), run(ho)
    new_tr = float(L.pdms_at(d["tr_pdms"][tr], pk_tr).mean())
    new_ho = float(L.pdms_at(d["tr_pdms"][ho], pk_ho).mean())
    pk_id = L.select(d["logits"], d["props8"], d["pick"], best["surrogate"], 1.5)         # CTRL-ID: tau above every q
    ctrl_id = bool(np.array_equal(pk_id, d["pick"]))
    red_rate = float((run(tr, roll=1) != pk_tr).mean())                                    # CTRL-RED

    def dci(m, pk):
        v = (L.pdms_at(d["tr_pdms"][m], pk) - L.pdms_at(d["tr_pdms"][m], d["pick"][m])) * 100
        return L.Boot(list(d["log"][m])).ci(v)

    notfit = new_tr <= base_tr
    heldout_ok = new_ho >= base_ho
    outcome = "NULL-FIT" if notfit else ("PASS-HELDOUT" if heldout_ok else "NULL-HELDOUT")
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    frozen = {"surrogate": best["surrogate"], "tau": best["tau"], "fit_outcome": outcome, "fitted_at_utc": now,
              "fit_split": "train (navtrain cache)", "n_train_tokens": int(tr.sum()), "n_heldout_tokens": int(ho.sum()),
              "n_train_logs": int(len(set(d["log"][tr]))), "n_heldout_logs": int(len(set(d["log"][ho]))),
              "index_sha256": sha(a.index), "code_sha256": {f: sha(os.path.join(HERE, f)) for f in ("epgeom_lib.py", "epgeom_fit.py")},
              "tie_rule": "exact float32 g equality; C1 pick if tied else argmax C1 aggregate over tied", "empty_set": "C1 pick"}
    rep = {"tier": "T0 (single-step open-loop planning labels, tracked PDM, navtrain cache; train-distribution tokens for C1)", "evidence": "MEASURED",
           "fitted": best,
           "train": {"c1_mean_pdms": base_tr, "rule_mean_pdms": new_tr, "delta_x100_ci": dci(tr, pk_tr), "switch_rate": float((pk_tr != d["pick"][tr]).mean())},
           "heldout": {"c1_mean_pdms": base_ho, "rule_mean_pdms": new_ho, "delta_x100_ci": dci(ho, pk_ho),
                       "switch_rate": float((pk_ho != d["pick"][ho]).mean()), "not_below_c1": heldout_ok},
           "controls": {"CTRL-ID_tau1.5_bitwise_all_tokens": ctrl_id, "CTRL-RED_train_pick_change_rate": red_rate, "CTRL-AGG_numpy_argmax_eq_stored_pick": agg_agree},
           "fit_table": table, "outcome": outcome}
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(frozen, open(os.path.join(a.out_dir, "FROZEN_PARAMS.json"), "w"), indent=1)
    json.dump(rep, open(os.path.join(a.out_dir, "FIT_REPORT.json"), "w"), indent=1)
    with open(os.path.join(a.out_dir, "FROZEN_PARAMS.sha256"), "w") as f:
        f.write(sha(os.path.join(a.out_dir, "FROZEN_PARAMS.json")) + " *FROZEN_PARAMS.json\n")
    print(json.dumps({k: rep[k] for k in ("fitted", "train", "heldout", "controls", "outcome")}, indent=1))
    return 0 if ctrl_id else 3


if __name__ == "__main__":
    raise SystemExit(main())
