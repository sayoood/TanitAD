#!/usr/bin/env python3
"""p4_counts.py -- PRE-FREEZE window COUNTS per class for PREREG_P4_CUECLASS.md.  COUNTS ONLY: this script never opens an OOF-score or PCA file and computes no AUC.
Reads the banked trigger probe's raw/windows_train.npz + raw/sample_{pstop,pmove}.npz (W-A) and raw/eval141_features.npz (W-B) plus the label files named in the card.
Writes P4_COUNTS_PREFREEZE.json next to this file.   CUDA_VISIBLE_DEVICES=-1   python -I p4_counts.py"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW = "E:/Projects/TanitAD-artifacts/dagland/trigger_probe/raw"
EVAL_TL = "E:/Projects/TanitAD-artifacts/_s2build-copy-20260919/release/v72/s2_labels_v7.2_eval.jsonl.gz"
EVAL_LEAD = "E:/Projects/TanitAD-artifacts/v9labels/inputs/lead_eval139.npz"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def classes(tl, lead_now, no_lead):
    """The P4 partition (card section 3): LIGHT = TL & no_lead; LEAD = ~TL & lead_now; SELF = ~TL & no_lead; MIXED = TL & lead_now; UNKNOWN = lead-unknown."""
    unk = ~lead_now & ~no_lead
    return {"LIGHT": tl & no_lead, "LEAD": ~tl & lead_now, "SELF": ~tl & no_lead, "MIXED": tl & lead_now, "UNKNOWN": unk}


def cell(m, y, clip):
    return {"n": int(m.sum()), "pos": int((m & y).sum()), "neg": int((m & ~y).sum()), "clips": int(len(np.unique(clip[m])))}


def eval_flags(ev):
    tl = {}
    with gzip.open(EVAL_TL, "rt", encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            gk = (r.get("g_tac") or {}).get("goals") or {}
            tl[r["clip_id"]] = any(k.startswith("TRAFFIC_LIGHT_REACT") for k in gk)
    L = np.load(EVAL_LEAD, allow_pickle=False)
    ls, lk, ll, lg = L["sha12"], L["k"], L["lead"], L["gap"]
    n = len(ev["clip"])
    TL, LN, NL = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    for i in range(n):
        c = str(ev["clip"][i])
        TL[i] = tl.get(c, False)
        s = hashlib.sha256(c.encode()).hexdigest()[:12]
        rows = np.flatnonzero((ls == s) & (lk == 2 * int(ev["t0"][i])))           # cache step t <-> row k = 2t (build_label_join's mapping)
        if len(rows):
            r = rows[0]
            LN[i] = bool(ll[r] == 1 and np.isfinite(lg[r]))
            NL[i] = bool(ll[r] == 0)
    return TL, LN, NL


def main():
    A = np.load(f"{RAW}/windows_train.npz", allow_pickle=False)
    out = {"note": "COUNTS ONLY (no AUC, no OOF file opened)", "files_sha256": {"windows_train.npz": sha256_file(f"{RAW}/windows_train.npz"), "eval141_features.npz": sha256_file(f"{RAW}/eval141_features.npz"),
                                                                              "s2_labels_v7.2_eval.jsonl.gz": sha256_file(EVAL_TL), "lead_eval139.npz": sha256_file(EVAL_LEAD)}, "W_A": {}, "W_B": {}}
    for pop in ("pstop", "pmove"):
        idx = np.load(f"{RAW}/sample_{pop}.npz")["idx"]
        tl, ln, nl, y, clip = A["TL_ANY"][idx], A["lead_now"][idx], A["no_lead"][idx], A["go"][idx].astype(bool), A["clip_i"][idx]
        assert not np.any(ln & nl)
        C = classes(tl, ln, nl)
        assert int(sum(m.sum() for m in C.values())) == len(idx), "the five cells do not partition the sample"
        out["W_A"][pop] = {"n": int(len(idx)), "n_clips": int(len(np.unique(clip))), "classes": {k: cell(m, y, clip) for k, m in C.items()}}
    ev = np.load(f"{RAW}/eval141_features.npz", allow_pickle=False)
    TL, LN, NL = eval_flags(ev)
    eid = ev["eid"]
    for pop, sel in (("pstop", ev["v0"] < 0.5), ("pmove", (ev["v0"] >= 0.5) & (ev["v0"] < 6.0))):
        C = classes(TL[sel], LN[sel], NL[sel])
        y = ev["go"][sel].astype(bool)
        out["W_B"][pop] = {"n": int(sel.sum()), "n_episodes": int(len(np.unique(eid[sel]))), "classes": {k: cell(m, y, eid[sel]) for k, m in C.items()}}
    json.dump(out, open(f"{HERE}/P4_COUNTS_PREFREEZE.json", "w"), indent=1)
    for w in ("W_A", "W_B"):
        for pop, d in out[w].items():
            print(w, pop, "n", d["n"], {k: (v["n"], v["pos"], v["neg"], v["clips"]) for k, v in d["classes"].items()})


if __name__ == "__main__":
    main()
