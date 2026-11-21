#!/usr/bin/env python3
"""build_windows.py -- step 1 of the trigger probe (PREREG_FROZEN.md sections 1-3, 6).  Labels / ego features / strata flags for ALL 306,566 canonical TRAIN windows, then the stratified sample.
No image feature is read here.  Writes raw/windows_train.npz and raw/sample_{pstop,pmove}.npz.   CUDA_VISIBLE_DEVICES=-1   python -I build_windows.py
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
CD = "E:/Projects/TanitAD-artifacts/dagland/crawl_diag"
TL_LABELS = "E:/Projects/TanitAD-artifacts/_s2build-copy-20260919/release/v72/s2_labels_v7.2_train.jsonl.gz"
LEAD = "C:/Users/Admin/box2_ship/v9ship/lead_train4572.npz"
RAW = f"{HERE}/raw"
N_PER_CLASS = 6000
DT = 0.2
sys.path.insert(0, HERE)
import tp_lib as L  # noqa: E402


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(RAW, exist_ok=True)
    t0 = time.time()
    sys.path.insert(0, CD)
    import importlib.util
    spec = importlib.util.spec_from_file_location("crawl_diag", f"{CD}/crawl_diag.py")
    cd = importlib.util.module_from_spec(spec)
    sys.modules["crawl_diag"] = cd
    spec.loader.exec_module(cd)
    env = cd.Env()
    tr, tracks = cd.train_inputs(env)          # REFUSES unless the window-list sha256 equals e144adcd...8168
    D = env.tool().D
    assert tr["digest"] == cd.WINDOW_LIST_SHA
    clips, first = tr["clips"], tr["first"]
    print("windows", tr["n_windows"], "clips", len(clips), "first", first, "digest ok", tr["digest"][:12], f"{time.time() - t0:.0f}s", flush=True)

    fk = D._fk()
    clip_i, tt, v0s, gos, go24, lead_now, lead_dep, no_lead, E8, EH = [], [], [], [], [], [], [], [], [], []
    leadz = np.load(LEAD, allow_pickle=False)
    lead_sha = np.asarray(leadz["sha12"])
    order = np.argsort(lead_sha, kind="stable")
    ls, lk, ll, lg = lead_sha[order], np.asarray(leadz["k"])[order], np.asarray(leadz["lead"])[order], np.asarray(leadz["gap"])[order]
    for ci, c in enumerate(clips):
        tr_ = tracks[c]
        pg = tr_[0::2]
        T = len(pg)
        kin3, omega = D.yaw_kin_from_track(tr_)
        gt, gok = D.gt_future(pg)
        ts = np.arange(first, T - D.HORIZON)
        assert gok[ts].all()
        g = gt[ts]
        v0 = kin3[ts, 0].astype(np.float64)
        go = L.go_label(v0, g, 29)
        g24 = L.go_label(v0, g, 11)
        p10 = fk.kin_features(kin3[ts].astype(np.float64)).astype(np.float32)
        assert p10.shape == (len(ts), 8) and np.isfinite(p10).all(), (c, p10.shape)
        v_full = kin3[:, 0].astype(np.float64)
        cols = [p10]
        for k in range(1, 6):
            ik = np.maximum(ts - k, 2)          # POST-FREEZE FIX (E_H sensitivity only): a window with t < 7 has no full 1.0 s history; the missing steps are held at row 2 (rows 0-1 carry NaN omega/v; a negative index used to WRAP)
            cols.append(v_full[ik][:, None].astype(np.float32))
            cols.append(omega[ik][:, None].astype(np.float32))
        eh = np.concatenate(cols, 1)
        # lead (build_label_join's mapping: cache step t <-> row k = 2t)
        s12 = D.sha12(c)
        a, b = np.searchsorted(ls, s12, "left"), np.searchsorted(ls, s12, "right")
        if b > a:
            kk, ldv, gpv = lk[a:b], ll[a:b], lg[a:b]
            gap_by = dict(zip(kk.tolist(), np.where(ldv == 1, gpv, np.nan).tolist()))
            has_row = dict(zip(kk.tolist(), ldv.tolist()))
            gnow = np.array([gap_by.get(2 * int(t), np.nan) for t in ts])
            gfut = np.array([gap_by.get(2 * (int(t) + 10), np.nan) for t in ts])
            ln = np.isfinite(gnow)
            nl = np.array([(has_row.get(2 * int(t), -1) == 0) for t in ts])
            ld = ln & np.isfinite(gfut) & ((gfut - gnow) >= 2.0)
        else:
            ln = np.zeros(len(ts), bool); nl = np.zeros(len(ts), bool); ld = np.zeros(len(ts), bool)
        clip_i.append(np.full(len(ts), ci)); tt.append(ts); v0s.append(v0); gos.append(go); go24.append(g24)
        lead_now.append(ln); no_lead.append(nl); lead_dep.append(ld); E8.append(p10); EH.append(eh)
    A = {"clip_i": np.concatenate(clip_i).astype(np.int32), "t": np.concatenate(tt).astype(np.int32), "v0": np.concatenate(v0s), "go": np.concatenate(gos),
         "go24": np.concatenate(go24), "lead_now": np.concatenate(lead_now), "no_lead": np.concatenate(no_lead), "lead_depart": np.concatenate(lead_dep),
         "E": np.concatenate(E8), "EH": np.concatenate(EH)}
    assert len(A["t"]) == tr["n_windows"]
    # control: the vectorised label equals the independent python loop of mlaunch on a sample, and the GT-based v0 agrees with crawl_diag's own v0
    assert np.allclose(A["v0"], tr["v0"]) and np.array_equal(A["clip_i"], tr["clip_i"]), "v0 / clip order differs from crawl_diag.train_inputs"
    # TL flags (clip level)
    tl = {k: np.zeros(len(clips), bool) for k in ("TL_RED", "TL_GREEN", "TL_YELLOW", "TL_ANY")}
    cidx = {c: i for i, c in enumerate(clips)}
    n_lab = 0
    with gzip.open(TL_LABELS, "rt", encoding="utf-8") as f:
        for ln_ in f:
            r = json.loads(ln_)
            n_lab += 1
            i = cidx.get(r["clip_id"])
            if i is None:
                continue
            gk = (r.get("g_tac") or {}).get("goals") or {}
            tl["TL_RED"][i] = "TRAFFIC_LIGHT_REACT_RED" in gk
            tl["TL_GREEN"][i] = "TRAFFIC_LIGHT_REACT_GREEN" in gk
            tl["TL_YELLOW"][i] = "TRAFFIC_LIGHT_REACT_YELLOW" in gk
            tl["TL_ANY"][i] = any(k.startswith("TRAFFIC_LIGHT_REACT") for k in gk)
    for k, v in tl.items():
        A[k] = v[A["clip_i"]]
    A["NO_TL"] = ~A["TL_ANY"]
    A["episode_sha12"] = np.array([D.sha12(c) for c in clips])
    A["clips"] = np.array(clips)
    np.savez(f"{RAW}/windows_train.npz", **A)
    prov = {"window_list_sha256": tr["digest"], "n_windows": int(tr["n_windows"]), "n_clips": len(clips), "tl_label_file_sha256": sha256_file(TL_LABELS), "tl_label_records": n_lab,
            "tl_clip_counts_in_train": {k: int(v.sum()) for k, v in tl.items()}, "lead_file_sha256": sha256_file(LEAD), "windows_npz_sha256": sha256_file(f"{RAW}/windows_train.npz")}
    pops = {"pstop": A["v0"] < 0.5, "pmove": (A["v0"] >= 0.5) & (A["v0"] < 6.0)}
    rng = np.random.default_rng(0)
    for nm, pm in pops.items():
        idx_all = np.flatnonzero(pm)
        sel = []
        for cls in (1, 0):
            ii = idx_all[A["go"][idx_all] == cls]
            n = min(len(ii), N_PER_CLASS)
            sel.append(np.sort(rng.choice(ii, size=n, replace=False)))
        sel = np.sort(np.concatenate(sel))
        np.savez(f"{RAW}/sample_{nm}.npz", idx=sel)
        prov[nm] = {"n_population": int(pm.sum()), "n_go": int(A["go"][pm].sum()), "go_rate": float(A["go"][pm].mean()), "n_clips_population": int(len(np.unique(A["clip_i"][pm]))),
                    "sample_n": int(len(sel)), "sample_n_go": int(A["go"][sel].sum()), "sample_n_clips": int(len(np.unique(A["clip_i"][sel]))), "go24_rate": float(A["go24"][pm].mean())}
    json.dump(prov, open(f"{RAW}/windows_provenance.json", "w"), indent=1)
    print(json.dumps(prov, indent=1), f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
