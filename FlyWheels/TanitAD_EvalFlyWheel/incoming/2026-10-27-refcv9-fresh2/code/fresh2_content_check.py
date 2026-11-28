"""fresh2_content_check.py -- PREREG_FRESH2 step 3 content assertions BEFORE stage 1: every FRESH2 bank is asserted on its CONTENT (a bank of zeros, a constant
feature or an all-IGNORE label file must not pass).  Reads the FRESH2 artifacts only (labels by count, never a score).  Writes result/CONTENT_CHECKS.json.
  z_tac bank   final, n == windows json, finite, no all-zero row, per-dim sd > 0 on >= 99 % of dims, v0_in finite and in [0, 60] m/s, == dump extras v0
  features     a0 / psi_dot0 finite on every window, not all zero, plausible ranges, anchor rule calibrated
  obstacle     parquet present + non-empty for the clips the fetch kept
  LON6 labels  per split (CONFIRM / GROW): class census of the trainer's lon target on the lead release; all six classes present in GROW's fit rows
  K-HEAD GT    per split: S6-valid / compliant / kinematic decel / cruise counts (no score)
No clip id is printed or written.  NEW file (FRESH2 agent, 2026-10-27).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
for _p in ("C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58/stack", "C:/Users/Admin/refcv8_pf/refcv9_lonhead/code", "C:/Users/Admin/refcv8_pf/refcv9_l2/code"):
    sys.path.insert(0, _p)
import numpy as np  # noqa: E402

PKG = Path(__file__).resolve().parent.parent
BANK = Path("F:/l1c/bank/ztac_fresh2.npz")
DUMP = Path("F:/l1c/tree/step48852ema_fresh2/dump_s0")
FEAT = Path("F:/l1c/fresh2/features/result")
REL = Path("F:/l1c/fresh2/kit_fresh2_lead/v9/v9_labels_eval.npz")
WJ = Path("F:/l1c/kit_fresh2/windows_s0.json")
OBS = Path("F:/l1c/fresh2/obstacle_fresh2")
SPLIT = Path("F:/l1c/fresh2/split.json")


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def main():
    import l2_common as C
    import lh_klabel as KL
    out, ok = {}, {}
    bk = np.load(BANK, allow_pickle=False)
    meta = json.loads(str(bk["meta_json"]))
    want = sum(len(v) for v in json.load(open(WJ, encoding="utf-8")).values())
    Z = bk["z_tac"].astype(np.float64)
    v0 = bk["v0_in"].astype(np.float64)
    sd = Z.std(0)
    ex = np.load(DUMP / "refcv6_extras.npz")
    out["ztac_bank"] = {"final": meta.get("final"), "n_windows": int(len(Z)), "want": want, "d": int(Z.shape[1]), "all_finite": bool(np.isfinite(Z).all()),
                        "rows_all_zero": int((np.abs(Z).max(1) == 0).sum()), "frac_dims_sd_gt_0": float((sd > 0).mean()), "mean_abs": float(np.abs(Z).mean()),
                        "sd_median": float(np.median(sd)), "n_distinct_rows_first_1000": int(len({r.tobytes() for r in Z[:1000]})),
                        "v0_min_med_max": [float(v0.min()), float(np.median(v0)), float(v0.max())], "v0_eq_dump_extras_v0": bool(np.array_equal(v0, ex["v0"].astype(np.float64))),
                        "cuda_max_memory_allocated_gib": meta.get("cuda_max_memory_allocated_gib"), "roll_wall_s": meta.get("roll_wall_s"), "prereg_sha256": meta.get("prereg_sha256")}
    b = out["ztac_bank"]
    ok["ztac"] = (b["final"] is True and b["n_windows"] == want and b["all_finite"] and b["rows_all_zero"] == 0 and b["frac_dims_sd_gt_0"] >= 0.99
                  and b["n_distinct_rows_first_1000"] == min(1000, want) and np.isfinite(v0).all() and 0 <= b["v0_min_med_max"][0] and b["v0_min_med_max"][2] <= 60
                  and b["v0_eq_dump_extras_v0"] and str(b["prereg_sha256"]).startswith("956f87cd"))
    ft = np.load(FEAT / "lha0_features.npz")
    cal = json.load(open(FEAT / "LHA0_CALIBRATION.json", encoding="utf-8"))
    a0, psi = ft["a0"].astype(np.float64), ft["psi_dot0"].astype(np.float64)
    out["features"] = {"calibration_verdict": cal["verdict"], "anchor_rule": cal.get("anchor_rule"), "n": int(len(a0)), "a0_finite": bool(np.isfinite(a0).all()),
                       "psi_finite": bool(np.isfinite(psi).all()), "a0_frac_nonzero": float((a0 != 0).mean()), "psi_frac_nonzero": float((psi != 0).mean()),
                       "a0_p1_p50_p99": [float(x) for x in np.percentile(a0, [1, 50, 99])], "psi_p1_p50_p99": [float(x) for x in np.percentile(psi, [1, 50, 99])],
                       "order_eq_bank": bool(np.array_equal(ft["e_i"].astype(np.int64), bk["e_i"].astype(np.int64)) and np.array_equal(ft["t"].astype(np.int64), bk["t"].astype(np.int64)))}
    f = out["features"]
    ok["features"] = (f["calibration_verdict"] == "CALIBRATED" and f["n"] == want and f["a0_finite"] and f["psi_finite"] and f["a0_frac_nonzero"] > 0.5
                      and f["psi_frac_nonzero"] > 0.5 and -12 < f["a0_p1_p50_p99"][0] and f["a0_p1_p50_p99"][2] < 12 and f["order_eq_bank"])
    split = json.load(open(SPLIT, encoding="utf-8"))
    s12 = {k: {sha12(c) for c in split[k]} for k in ("confirm", "grow")}
    obs = {k: {"n_clips": len(split[k]), "obstacle_present": sum(1 for c in split[k] if (OBS / f"{c}.parquet").is_file()),
               "obstacle_nonempty": sum(1 for c in split[k] if (OBS / f"{c}.parquet").is_file() and (OBS / f"{c}.parquet").stat().st_size > 0)} for k in split}
    out["obstacle"] = obs
    ok["obstacle"] = all(v["obstacle_present"] == v["obstacle_nonempty"] and v["obstacle_present"] >= 0.9 * v["n_clips"] for v in obs.values())
    kit = {"release": str(REL), "md5": KL.md5_file(REL), "sidecar": "F:/l1c/kit_fresh2/data/clock_sidecar.jsonl", "dumps": {0: str(DUMP)}, "windows": str(WJ)}
    r = KL.trainer_lon_targets(kit, 0)
    jx = json.load(open(DUMP / "refcv6_extras.npz.json", encoding="utf-8"))
    i2s = {int(k): str(v) for k, v in jx["clip_sha12_by_index"].items()}
    rs12 = np.array([i2s[int(c)] for c in r["clip_index"]])
    zr = np.load(REL, allow_pickle=False)
    nolead = {str(s) for s, h in zip(zr["clip__sha12"], zr["clip__has_lead"]) if not bool(h)}
    E = C.load_train_dump(DUMP, "fresh2")
    assert np.array_equal(E["clip"], r["clip_index"]) and np.array_equal(E["ws"], r["ws"])
    lab = {}
    for k in ("confirm", "grow"):
        m = np.isin(rs12, sorted(s12[k]))
        ml = m & ~np.isin(rs12, sorted(nolead))
        cen = {f"{v} {KL.LON7[v] if 0 <= v < len(KL.LON7) else 'IGNORE'}": int((r["lon"][ml] == v).sum()) for v in sorted(set(r["lon"][ml].tolist()))}
        valid, comp, gt = np.asarray(E["valid_s6"], bool), np.asarray(E["compliant"], bool), np.asarray(E["gt"])
        lab[k] = {"windows": int(m.sum()), "windows_on_lead_clips": int(ml.sum()), "clips_without_lead": len(s12[k] & nolead), "lon6_census_lead_clips": cen,
                  "kinematic": {"s6_valid": int((valid & m).sum()), "compliant": int((valid & comp & m).sum()), "decel": int((valid & comp & m & (gt == "decel")).sum()),
                                "cruise": int((valid & comp & m & (gt == "cruise")).sum()), "accel": int((valid & comp & m & (gt == "accel")).sum())}}
    out["labels"] = {"release_md5": kit["md5"], "release_has_lead_clips": int(zr["clip__has_lead"].sum()), "release_n_clips": int(len(zr["clip__has_lead"])),
                     "label_checks": r["checks"], "by_split": lab}
    six = (0, 1, 3, 4, 5, 7)
    ok["labels"] = (r["checks"]["n_join_refused"] == 0 and r["checks"]["windows_sha12_eq_dump_sha12"] and r["checks"]["release_sid_eq_stable_episode_id"]
                    and all(any(key.startswith(f"{c} ") and n > 0 for key, n in lab["grow"]["lon6_census_lead_clips"].items()) for c in six)
                    and all(lab[k]["kinematic"]["decel"] > 0 and lab[k]["kinematic"]["cruise"] > 0 for k in lab))
    out["ALL_PASS"] = bool(all(ok.values()))
    out["checks"] = ok
    (PKG / "result").mkdir(exist_ok=True)
    (PKG / "result" / "CONTENT_CHECKS.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"ZZCONTENT{'PASS' if out['ALL_PASS'] else 'FAIL'}ZZ {ok} | confirm {lab['confirm']['kinematic']} grow {lab['grow']['kinematic']}", flush=True)
    return 0 if out["ALL_PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
