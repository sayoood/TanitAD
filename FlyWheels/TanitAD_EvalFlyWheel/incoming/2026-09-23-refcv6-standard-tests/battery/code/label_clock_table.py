"""SPEC A5: per-window TACTICAL label tables under BOTH label clocks, built ONCE with the 82c2331
trainer's own V3Dataset (the A16 fix), over the FULL eval index. CPU only, no model, no frames
(except the C2 sample, which decodes a few full items).

usage (the loader resolves its tree from REFCV6_REPO at import, so set it and PYTHONPATH first):
  REFCV6_REPO=C:/Users/Admin/ev6_82c2331 PYTHONPATH="C:/Users/Admin/ev6_82c2331/stack;C:/Users/Admin/ev6_82c2331/taniteval" \
  python label_clock_table.py --config <resume config.json> --extras <old-tree roll refcv6_extras.npz> --out-dir <dir>

OLD       = `legacy_label_clock = True`: (t + w - 1) * 0.1 s, the clock the pre-switch checkpoints trained on.
CORRECTED = `enable_clip_clock(sidecar)`: grid_start + (t + w - 1 + n_stack - 1) * dt (pose-dt fallback counted).
The direct computation mirrors `V3Dataset.__getitem__`'s label block line for line (82c2331
refc_v3_train.py:3083-3122) and is checked against full `__getitem__` items (C2).

Controls (SPEC A5, literals; any failure REFUSES -- exit 2 -- and the tables are not written):
  C1  OLD table == the labels in the OLD-tree roll's extras, bit for bit (NaN-aware), every extras window
  C2  direct computation == full __getitem__ on >= 20 windows over >= 5 clips, both clocks
  C3  OLD vs CORRECTED band membership (lat_v7 != IGNORE) must differ on > 0 windows; the full-index
      count is reported against A16's MEASURED 598 / 5,699
  C4  two independent CORRECTED builds are bit-identical (mirror)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_loader as L  # noqa: E402

L.bootstrap()


def build(config: dict, *, legacy: bool):
    """The eval dataset exactly as the battery builds it (refcv6_loader.build_eval_dataset, which now
    calls enable_clip_clock where the 82c2331 trainer does), then the clock switch."""
    tr = L.trainer()
    from tanitad.refs import refc_v3 as v3
    args, _argv, _arec = L.parse_args(config)
    # the same pre-pin sequence as refcv6_loader._build_model (train():6714-6726): the refusals and the
    # anchor artifact (it resolves units onto args before the pin); no weights are read
    tr._check_nav_from_v7_args(args)
    tr._check_max_speed_args(args)
    tr._check_goal_point_args(args)
    tr.check_effective_weights(args)
    tr._read_anchor_artifact(args)
    cfg = tr._pin_trainer_cfg(
        v3.refc_v3_smoke_config(args.arm == "hier") if args.smoke else
        v3.refc_v3_sized_config(args.size, hier=args.arm == "hier"), args)
    ds, eps, rec = L.build_eval_dataset(None, cfg, args, config, with_perception_targets=False)
    if not hasattr(ds, "legacy_label_clock"):
        raise SystemExit("[labels] this tree's V3Dataset has no legacy_label_clock -- not an A16 tree")
    ds.legacy_label_clock = bool(legacy)
    return ds, rec, int(cfg.core.window)


def direct_labels(ds, e_i: int, t: int):
    """V3Dataset.__getitem__'s label block (82c2331 refc_v3_train.py:3083-3122), without frames."""
    from tanitad.data import v7_labels as v7l
    ep = ds.episodes[e_i]
    lab = ds.v7_by_sid.get(int(ep.episode_id))
    n = len(v7l.TAC_GOAL_TOKENS)
    if lab is None:
        lat = lon = v7l.IGNORE_ID
        gy, gw = (0.0,) * n, (v7l.IGNORE_W,) * n
    else:
        now = ds._now_s(ep, t)
        lat, lon = v7l.tactical_class_ids(lab, now)
        if ds.tac_goal_targets:
            gy, gw = v7l.tactical_goal_targets(lab, now, negatives=ds.tac_goal_negatives,
                                               sidecar=ds.cot_negative_sidecar)
        else:
            gy, gw = (np.nan,) * n, (np.nan,) * n
    return int(lat), int(lon), np.asarray(gy, np.float32), np.asarray(gw, np.float32)


def table(ds, W: int) -> dict:
    rows = {"e_i": [], "t": [], "ws": [], "lat": [], "lon": [], "gy": [], "gw": []}
    for (e_i, t) in ds.index:
        lat, lon, gy, gw = direct_labels(ds, int(e_i), int(t))
        rows["e_i"].append(int(e_i)); rows["t"].append(int(t)); rows["ws"].append(int(t) + W - 1)
        rows["lat"].append(lat); rows["lon"].append(lon); rows["gy"].append(gy); rows["gw"].append(gw)
    return {k: np.asarray(v) for k, v in rows.items()}


def same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=(a.dtype.kind == "f")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--extras", required=True, help="an OLD-tree roll's refcv6_extras.npz (C1)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--c2-n", type=int, default=24)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = L.load_config(a.config)
    rec = {"tool": "label_clock_table.py", "amendment": "A5", "repo": str(L.REPO),
           "config": a.config, "config_md5": L.md5_file(a.config), "extras": a.extras,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    ds_o, r_o, W = build(config, legacy=True)
    ds_c, r_c, _ = build(config, legacy=False)
    ds_c2, _, _ = build(config, legacy=False)
    rec["label_clock_old_build"] = r_o.get("label_clock")
    rec["label_clock_corrected_build"] = r_c.get("label_clock")
    rec["window"] = W
    T_o, T_c, T_c2 = table(ds_o, W), table(ds_c, W), table(ds_c2, W)
    ctl = {}
    # ---- C4 mirror ---------------------------------------------------------------------- #
    ctl["C4_mirror_corrected_twice_bit_identical"] = all(same(T_c[k], T_c2[k]) for k in T_c)
    # ---- C3 the two clocks must differ -------------------------------------------------- #
    ign = -100
    mem_o, mem_c = T_o["lat"] != ign, T_c["lat"] != ign
    both = mem_o & mem_c
    ctl["C3"] = {"n_windows_full_index": int(len(mem_o)),
                 "n_in_band_old": int(mem_o.sum()), "n_in_band_corrected": int(mem_c.sum()),
                 "n_membership_changed": int((mem_o != mem_c).sum()),
                 "n_in_band_either": int((mem_o | mem_c).sum()),
                 "n_lat_class_changed_in_both": int((both & (T_o["lat"] != T_c["lat"])).sum()),
                 "n_lon_class_changed_in_both": int((both & (T_o["lon"] != T_c["lon"])).sum()),
                 "n_goal_rows_changed": int(sum(0 if (same(T_o["gy"][i], T_c["gy"][i]) and same(T_o["gw"][i], T_c["gw"][i])) else 1
                                                for i in range(len(mem_o)))),
                 "reference_A16_MEASURED": "598 / 5,699 (10.5 %) of tactical-supervised eval windows change band"}
    ctl["C3"]["pass"] = ctl["C3"]["n_membership_changed"] > 0
    # ---- C1 OLD table == the old-tree roll's extras -------------------------------------- #
    ex = np.load(a.extras)
    key = {(int(e), int(w)): i for i, (e, w) in enumerate(zip(T_o["e_i"], T_o["ws"]))}
    idx = [key.get((int(c), int(w))) for c, w in zip(ex["clip_index"], ex["ws"])]
    missing = sum(1 for i in idx if i is None)
    ok_idx = np.asarray([i for i in idx if i is not None])
    sel = np.asarray([i is not None for i in idx])
    c1 = {"n_extras_windows": int(len(idx)), "n_not_in_table": int(missing)}
    if missing == 0:
        c1["lat_equal"] = same(ex["lat_v7"].astype(np.int64), T_o["lat"][ok_idx].astype(np.int64))
        c1["lon_equal"] = same(ex["lon_v7"].astype(np.int64), T_o["lon"][ok_idx].astype(np.int64))
        c1["goal_y_equal"] = same(ex["tac_goal_y"].astype(np.float32), T_o["gy"][ok_idx].astype(np.float32))
        c1["goal_w_equal"] = same(ex["tac_goal_w"].astype(np.float32), T_o["gw"][ok_idx].astype(np.float32))
        c1["corrected_differs_on_these_windows"] = int((T_o["lat"][ok_idx] != T_c["lat"][ok_idx]).sum())
    c1["pass"] = bool(missing == 0 and c1.get("lat_equal") and c1.get("lon_equal")
                      and c1.get("goal_y_equal") and c1.get("goal_w_equal"))
    ctl["C1_old_table_equals_old_tree_extras"] = c1
    # ---- C2 direct == full __getitem__ ------------------------------------------------- #
    rng = np.random.default_rng(0)
    eps_all = sorted({int(e) for e in T_o["e_i"]})
    pick_eps = rng.choice(eps_all, size=min(6, len(eps_all)), replace=False)
    cand = [i for i, e in enumerate(T_o["e_i"]) if int(e) in set(int(x) for x in pick_eps)]
    # prefer windows where the two clocks disagree, so C2 exercises the corrected branch for real
    dis = [i for i in cand if T_o["lat"][i] != T_c["lat"][i]]
    rest = [i for i in cand if i not in set(dis)]
    pick = (list(rng.choice(dis, size=min(len(dis), a.c2_n // 2), replace=False)) if dis else []) + \
        list(rng.choice(rest, size=min(len(rest), a.c2_n - min(len(dis), a.c2_n // 2)), replace=False))
    c2 = {"n_windows": len(pick), "n_clips": int(len({int(T_o["e_i"][i]) for i in pick})),
          "n_where_clocks_disagree": int(sum(1 for i in pick if T_o["lat"][i] != T_c["lat"][i])),
          "mismatches": []}
    for dsx, T, nm in ((ds_o, T_o, "old"), (ds_c, T_c, "corrected")):
        for i in pick:
            it = dsx[int(i)]
            got = (int(it["lat_v7"]), int(it["lon_v7"]), it["tac_goal_y"].numpy().astype(np.float32),
                   it["tac_goal_w"].numpy().astype(np.float32))
            if not (got[0] == int(T["lat"][i]) and got[1] == int(T["lon"][i])
                    and same(got[2], T["gy"][i].astype(np.float32)) and same(got[3], T["gw"][i].astype(np.float32))):
                c2["mismatches"].append({"clock": nm, "row": int(i)})
    c2["pass"] = bool(len(pick) >= 20 and c2["n_clips"] >= 5 and not c2["mismatches"])
    ctl["C2_direct_equals_getitem"] = c2
    rec["controls"] = ctl
    rec["verdict"] = ("PASS" if (ctl["C4_mirror_corrected_twice_bit_identical"] and ctl["C3"]["pass"]
                                  and c1["pass"] and c2["pass"]) else "REFUSED")
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(rec, open(out / "label_clock_record.json", "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({"verdict": rec["verdict"], "C1": c1.get("pass"), "C2": c2["pass"], "C3": ctl["C3"],
                      "C4": ctl["C4_mirror_corrected_twice_bit_identical"]}, indent=1, default=str))
    if rec["verdict"] != "PASS":
        raise SystemExit(2)
    np.savez_compressed(out / "label_tables.npz",
                        e_i=T_o["e_i"], t=T_o["t"], ws=T_o["ws"],
                        lat_old=T_o["lat"], lon_old=T_o["lon"], gy_old=T_o["gy"], gw_old=T_o["gw"],
                        lat_corrected=T_c["lat"], lon_corrected=T_c["lon"], gy_corrected=T_c["gy"],
                        gw_corrected=T_c["gw"])
    h = hashlib.sha256((out / "label_tables.npz").read_bytes()).hexdigest()
    rec["label_tables_sha256"] = h
    json.dump(rec, open(out / "label_clock_record.json", "w", encoding="utf-8"), indent=1, default=str)


if __name__ == "__main__":
    main()
