"""fresh2_lhg_fit.py -- PREREG_FRESH2 s3 arm `LH_G`: LH's recipe (`refcv9_lonhead/code/lh_fit.py`, IMPORTED UNEDITED: folds_of, cv, freeze, load_frozen,
predict_logits6; features [standardised z_tac, v0/10]; the sealed config grid {logistic lambda 1e-3..100, MLP wd 1e-4..1e-2} chosen by out-of-fold CE;
5 folds by EPISODE, numpy default_rng(0)) refit on the L1C FIT rows (8,178 rows / 290 episodes, `lh_fit.load_fresh(..., lead_clips_only=True)`)
PLUS the FRESH2 GROW rows.  CONFIRM is never read here (its clips are excluded by sha12 before any row is formed, and asserted).

GROW rows = the row rule of lh_fit.load_fresh applied to FRESH2: every GROW window whose GT LON6 label (lh_klabel.trainer_lon_targets on the FRESH2
v9 release built WITH the lead pass) is one of the six LON6 classes, on clips the lead pass covers (clip__has_lead; lh_fit's lead-clips-only rule);
z_tac and v0 = the FRESH2 bank's z_tac / v0_in (refcv8 EMA-48,852, the roll that fed v0_in).  GROW episode ids are offset by +100000 so the two pulls'
integer clip indices are disjoint before folds_of.

  python fresh2_lhg_fit.py --selftest     # KNOWN VALUES on the first pull alone: lh_fit.cv must reproduce LH_FIT_INFO.json's LH table exactly and the
                                          # re-frozen blob must hash to the registered LH_FROZEN.pt (6fd3ea1e...d8f6)
  python fresh2_lhg_fit.py --fit          # LH_G -> F:/l1c/fresh2/lhg/LH_G_FROZEN.pt (+ .sha256, LH_G_FIT_INFO.json)
CPU only.  No clip id is printed or written.  NEW file (FRESH2 agent, 2026-10-27).
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
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.dont_write_bytecode = True
import numpy as np

LONHEAD = Path("C:/Users/Admin/refcv8_pf/refcv9_lonhead")
for _p in ("C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58/stack", str(LONHEAD / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import lh_fit as LF  # noqa: E402
import lh_klabel as KL  # noqa: E402

PKG = Path(__file__).resolve().parent.parent
OUT = Path("F:/l1c/fresh2/lhg")
LH_SHA = "6fd3ea1e3700a15b6506176bc855355d0cc1eefb3d87d390584e791d98e1d8f6"
LH_FIT_SHA = "fb330f7f23b2fea249e1527e804c454062a06ffa0cbe4b612ff551f57f5b62d9"
FR2 = {"release": "F:/l1c/fresh2/kit_fresh2_lead/v9/v9_labels_eval.npz", "sidecar": "F:/l1c/kit_fresh2/data/clock_sidecar.jsonl",
       "dumps": {0: "F:/l1c/tree/step48852ema_fresh2/dump_s0"}, "windows": "F:/l1c/kit_fresh2/windows_s0.json"}
FR2_BANK = Path("F:/l1c/bank/ztac_fresh2.npz")
SPLIT = Path("F:/l1c/fresh2/split.json")
GROW_EID_OFFSET = 100000


def sha12(c: str) -> str:
    return hashlib.sha256(c.encode("utf-8")).hexdigest()[:12]


def lh_prov(D: dict) -> dict:
    """lh_fit.main's provenance dict, rebuilt key for key (same order), so a re-freeze of the first pull is byte-comparable."""
    return {"script_sha256": LF.sha256_file(LF.__file__), "klabel_sha256": LF.sha256_file(Path(LF.__file__).parent / "lh_klabel.py"), **D["sources"],
            "prereg_sha256": LF.sha256_file(LONHEAD / "PREREG_LONHEAD_C.md"), "label_function": "lh_klabel.trainer_lon_targets (refc_v3_train.py:4646-4674)"}


def load_grow(which: str = "grow") -> dict:
    """lh_fit.load_fresh's row rule on FRESH2, restricted to the GROW clips (CONFIRM never forms a row)."""
    split = json.load(open(SPLIT, encoding="utf-8"))
    keep12 = {sha12(c) for c in split[which]}
    other12 = {sha12(c) for c in split["confirm" if which == "grow" else "grow"]}
    rel = Path(FR2["release"])
    z_rel = np.load(rel, allow_pickle=False)
    nolead_s12 = {str(s) for s, h in zip(z_rel["clip__sha12"], z_rel["clip__has_lead"]) if not bool(h)}
    r = KL.trainer_lon_targets(FR2, 0, release=str(rel), md5=KL.md5_file(rel))
    bk = np.load(FR2_BANK, allow_pickle=False)
    meta = json.loads(str(bk["meta_json"]))
    W = int(meta["window_W"])
    ws_b = bk["t"].astype(np.int64) + W - 1
    idx = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(bk["e_i"], ws_b))}
    j = np.array([idx.get((int(c), int(w)), -1) for c, w in zip(r["clip_index"], r["ws"])])
    if (j < 0).any() or len(idx) != len(ws_b):
        raise SystemExit(f"[lhg] {int((j < 0).sum())} FRESH2 windows have no z_tac bank row (or duplicate bank keys)")
    dump = Path(FR2["dumps"][0])
    ex = np.load(dump / "refcv6_extras.npz")
    v0 = bk["v0_in"].astype(np.float64)[j]
    dv0 = float(np.abs(v0 - ex["v0"].astype(np.float64)).max())
    jx = json.load(open(dump / "refcv6_extras.npz.json", encoding="utf-8"))
    i2s = {int(k): str(v) for k, v in jx["clip_sha12_by_index"].items()}
    s12 = np.array([i2s[int(c)] for c in r["clip_index"]])
    assert keep12 <= set(i2s.values()) and not (keep12 & other12), "split sha12 sets vs the FRESH2 dump"
    in_set = np.isin(s12, sorted(keep12))
    lead_ok = ~np.isin(s12, sorted(nolead_s12))
    lab = r["lon"]
    keep = np.isin(lab, LF.LON6_V7) & in_set & lead_ok
    assert not np.isin(s12[keep], sorted(other12)).any(), "a row of the other split leaked in"
    y = np.array([LF.LON6_V7.index(int(v)) if v in LF.LON6_V7 else -1 for v in lab])
    return {"Z": bk["z_tac"].astype(np.float64)[j][keep], "v0": v0[keep], "y": y[keep], "eid": r["clip_index"][keep] + GROW_EID_OFFSET, "ws": r["ws"][keep],
            "n_windows_in_split": int(in_set.sum()), "n_rows": int(keep.sum()), "n_episodes": int(len(np.unique(r["clip_index"][keep]))),
            "n_clips_in_split": int(len(keep12)), "n_split_clips_without_lead": int(len(keep12 & nolead_s12)),
            "class_counts": {f"{LF.LON6_V7[k]} {KL.LON7[LF.LON6_V7[k]]}": int((y[keep] == k).sum()) for k in range(6)},
            "label_census_all_split_windows": {str(int(v)): int((lab[in_set] == v).sum()) for v in np.unique(lab[in_set])},
            "checks": {**r["checks"], "bank_v0_in_vs_extras_v0_maxabs": dv0, "bank_final": bool(meta.get("final")), "bank_n_windows": int(meta.get("n_windows", -1))},
            "sources": {"release": str(rel), "release_md5": KL.md5_file(rel), "release_has_lead_clips": int(z_rel["clip__has_lead"].sum()),
                        "release_n_clips": int(len(z_rel["clip__has_lead"])), "bank": str(FR2_BANK), "bank_sha256": LF.sha256_file(FR2_BANK),
                        "split_sha256": LF.sha256_file(SPLIT), "which": which}}


def selftest() -> int:
    t0 = time.time()
    assert LF.sha256_file(LF.__file__) == LH_FIT_SHA, "lh_fit.py is not the registered file"
    info = json.load(open(LONHEAD / "LH_FIT_INFO.json", encoding="utf-8"))
    D = LF.load_fresh(LF.FRESH_LEAD_RELEASE, False, True)
    fold = LF.folds_of(D["eid"])
    res = LF.cv(D, D["y"], fold)
    want = info["LH"]["table"]
    cmp = {}
    for k, v in want.items():
        g = res["table"][k]
        cmp[k] = {"mean_ce_oof_equal": g["mean_ce_oof"] == v["mean_ce_oof"], "fold_ce_equal": g["fold_ce"] == v["fold_ce"],
                  "epochs_equal": g.get("early_stop_epochs") == v.get("early_stop_epochs"), "got": g["mean_ce_oof"], "want": v["mean_ce_oof"]}
    blob = LF.freeze(D, D["y"], res["chosen_cfg"], "LH", lh_prov(D))
    sha = hashlib.sha256(blob).hexdigest()
    ok = {"population_8178_290": D["n_rows"] == 8178 and D["n_episodes"] == 290, "class_counts": D["class_counts"] == info["population"]["class_counts"],
          "cv_table_exact": all(x["mean_ce_oof_equal"] and x["fold_ce_equal"] and x["epochs_equal"] for x in cmp.values()),
          "chosen_logistic_100": res["chosen"] == info["LH"]["chosen"] == "logistic:100", "refreeze_sha256_eq_LH_FROZEN": sha == LH_SHA}
    out = {"tool": "fresh2_lhg_fit.py --selftest", "known_values": "LH_FIT_INFO.json (sha256 a7e53f5e...) + LH_FROZEN.pt (6fd3ea1e...d8f6)", "checks": ok,
           "ALL_PASS": all(ok.values()), "table_compare": cmp, "refreeze_sha256": sha, "wall_s": round(time.time() - t0, 1)}
    (PKG / "result").mkdir(exist_ok=True)
    (PKG / "result" / "SELFTEST_LHG_RECIPE.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"ZZLHGSELFTEST{'PASS' if out['ALL_PASS'] else 'FAIL'}ZZ {ok}", flush=True)
    return 0 if out["ALL_PASS"] else 2


def fit() -> int:
    t0 = time.time()
    assert LF.sha256_file(LF.__file__) == LH_FIT_SHA, "lh_fit.py is not the registered file"
    A = LF.load_fresh(LF.FRESH_LEAD_RELEASE, False, True)                       # the L1C FIT rows
    assert A["n_rows"] == 8178 and A["n_episodes"] == 290
    G = load_grow("grow")
    for k, v in G["checks"].items():
        if k.startswith(("windows_sha12", "release_sid")):
            assert v is True, (k, v)
    assert G["checks"]["bank_v0_in_vs_extras_v0_maxabs"] == 0.0 and G["checks"]["bank_final"] and G["checks"]["n_join_refused"] == 0
    assert not (set(np.unique(A["eid"]).tolist()) & set(np.unique(G["eid"]).tolist()))
    D = {"Z": np.vstack([A["Z"], G["Z"]]), "v0": np.concatenate([A["v0"], G["v0"]]), "eid": np.concatenate([A["eid"], G["eid"]])}
    y = np.concatenate([A["y"], G["y"]])
    counts = {f"{LF.LON6_V7[k]} {KL.LON7[LF.LON6_V7[k]]}": int((y == k).sum()) for k in range(6)}
    missing = [k for k, n in counts.items() if n < 10]
    if missing:
        raise SystemExit(f"[lhg] REFUSING (lh_fit's rule): LON6 classes with < 10 rows: {missing}")
    fold = LF.folds_of(D["eid"])
    res = LF.cv(D, y, fold)
    prov = {"script_sha256": LF.sha256_file(LF.__file__), "klabel_sha256": LF.sha256_file(Path(LF.__file__).parent / "lh_klabel.py"),
            "first_pull_sources": A["sources"], "grow_sources": G["sources"], "prereg_FRESH2_sha256": LF.sha256_file(PKG / "PREREG_FRESH2.md"),
            "this_script_sha256": LF.sha256_file(__file__), "label_function": "lh_klabel.trainer_lon_targets (refc_v3_train.py:4646-4674)",
            "rows": "L1C FIT rows (lh_fit.load_fresh lead_clips_only) + FRESH2 GROW rows (same row rule); GROW eid offset +100000"}
    b1 = LF.freeze(D, y, res["chosen_cfg"], "LH_G", prov)
    b2 = LF.freeze(D, y, res["chosen_cfg"], "LH_G", prov)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "LH_G_FROZEN.pt"
    tmp = p.with_suffix(".pt.tmp")
    tmp.write_bytes(b1)
    os.replace(tmp, p)
    h = LF.sha256_file(p)
    (OUT / "LH_G_FROZEN.sha256").write_text(f"{h} *LH_G_FROZEN.pt\n# frozen {time.strftime('%FT%T%z')} BEFORE any CONFIRM row was scored\n", encoding="utf-8")
    bl = LF.load_frozen(p, h)
    lg = LF.predict_logits6(bl, D["Z"], D["v0"])
    info = {"tool": "fresh2_lhg_fit.py --fit", "prereg": "PREREG_FRESH2.md s3 arm LH_G", "script_sha256": LF.sha256_file(__file__), "lh_fit_sha256": LF.sha256_file(LF.__file__),
            "population": {"first_pull_rows": A["n_rows"], "first_pull_episodes": A["n_episodes"], "first_pull_class_counts": A["class_counts"],
                           "grow_rows": G["n_rows"], "grow_episodes": G["n_episodes"], "grow_clips": G["n_clips_in_split"], "grow_clips_without_lead": G["n_split_clips_without_lead"],
                           "grow_windows": G["n_windows_in_split"], "grow_class_counts": G["class_counts"], "grow_label_census_all_windows": G["label_census_all_split_windows"],
                           "total_rows": int(len(y)), "total_episodes": int(len(np.unique(D["eid"]))), "class_counts": counts},
            "d_features": int(D["Z"].shape[1] + 1), "grow_checks": G["checks"], "folds": {"k": 5, "seed": 0, "by": "episode (clip); GROW eid offset +100000"},
            "cv": res, "chosen": res["chosen"], "frozen": {"file": str(p), "sha256": h, "refreeze_bytes_identical": bool(b1 == b2),
                                                            "known_value_reload": {"train_rows_mean_ce": float(LF.ce_rows(LF.log_softmax(lg), y).mean()),
                                                                                   "train_rows_argmax_acc": float((lg.argmax(1) == y).mean())}},
            "wall_s": round(time.time() - t0, 1)}
    (OUT / "LH_G_FIT_INFO.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    (PKG / "result").mkdir(exist_ok=True)
    (PKG / "result" / "LH_G_FIT_INFO.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    print(f"ZZLHGFITZZ chosen {res['chosen']} ce {res['chosen_cfg']['mean_ce_oof']:.4f} rows {len(y)} (first {A['n_rows']} + grow {G['n_rows']}) sha {h[:16]}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    raise SystemExit(selftest() if a.selftest else fit())
