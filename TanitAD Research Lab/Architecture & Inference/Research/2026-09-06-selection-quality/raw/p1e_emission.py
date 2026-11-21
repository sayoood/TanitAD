"""P1e - LANE-CHANGE EMISSION AND RECALL, with the LABEL PREVALENCE BESIDE IT.

Reads the freshly rolled refcv4b step-40284 dump.  `lat_pred_nav_true` /
`lat_label` are the v7.2 8-way tactical class ids (IGNORE_ID -> -100), so the
emission rate and the prevalence are read on the SAME windows.

Also runs the DETERMINISM replicate the stability number needs: the arm's
planner must be deterministic given its inputs, or "jitter" is partly the
inference seed and not the model.
"""
import collections
import gzip
import json
import os
import sys

import numpy as np

import _env  # noqa: F401

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
LAB = r"C:\Users\Admin\navcomp\data\s2_labels_v7.2_eval.jsonl.gz"


def load(dump):
    fs = sorted(f for f in os.listdir(dump)
                if f.startswith("ep") and f.endswith(".npz"))
    out = collections.defaultdict(list)
    for f in fs:
        d = np.load(os.path.join(dump, "decisions", f), allow_pickle=True)
        for k in d.files:
            out[k].append(d[k])
        out["_epi"].append(np.full(len(d["ws"]), int(f[2:5])))
    return {k: np.concatenate(v) for k, v in out.items()}, fs


def table(title, names, gt, pr, flag=()):
    print(f"\n  {title}")
    print(f"    {'class':<16}{'n_true':>9}{'n_pred':>9}{'recall':>9}"
          f"{'precision':>11}")
    for i, nm in enumerate(names):
        nt = int((gt == i).sum())
        npd = int((pr == i).sum())
        tp = int(((gt == i) & (pr == i)).sum())
        rec = tp / nt if nt else float("nan")
        pre = tp / npd if npd else float("nan")
        m = "   <==" if nm in flag else ""
        print(f"    {nm:<16}{nt:>9}{npd:>9}{rec:>9.4f}{pre:>11.4f}{m}")
    print(f"    [read-control] classes with n_true > 0: "
          f"{sum(1 for i in range(len(names)) if (gt==i).sum()>0)} / "
          f"{len(names)};  with n_pred > 0: "
          f"{sum(1 for i in range(len(names)) if (pr==i).sum()>0)} / "
          f"{len(names)}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--arm-json", required=True)
    ap.add_argument("--replicate-dump", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tanitad.models.vocab_v7 import (TACTICAL_LAT_ACTIONS_V7 as LAT,
                                         TACTICAL_LON_ACTIONS_V7 as LON)
    D, files = load(a.dump)
    n = len(D["ws"])
    print(f"[dump] {a.dump}   {len(files)} episodes, {n} windows")

    res = {}
    for axis, names, pk, lk in (("LATERAL", LAT, "lat_pred_nav_true",
                                 "lat_label"),
                                ("LONGITUDINAL", LON, "lon_pred_nav_true",
                                 "lon_label")):
        pr, gt = D[pk], D[lk]
        val = gt != -100
        print(f"\n=== v7.2 TACTICAL {axis} HEAD ===")
        print(f"  windows with a valid label: {int(val.sum())} / {n} "
              f"({100.0*val.mean():.2f} %)  (IGNORE_ID -100 elsewhere)")
        print(f"  EMISSION over ALL {n} windows (label or not):")
        for i, nm in enumerate(names):
            c = int((pr == i).sum())
            m = ("   <== LANE CHANGE" if nm.startswith("LANE_CHANGE")
                 or nm == "ABORT_LC" else "")
            print(f"    {nm:<16}{c:>9} / {n}   {100.0*c/n:7.4f} %{m}")
        table(f"AGREEMENT on the {int(val.sum())} labelled windows",
              names, gt[val], pr[val],
              flag=("LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC",
                    "YIELD_MERGE"))
        res[axis] = {
            "n_windows": int(n), "n_labelled": int(val.sum()),
            "emission": {nm: int((pr == i).sum())
                         for i, nm in enumerate(names)},
            "label_prevalence": {nm: int((gt[val] == i).sum())
                                 for i, nm in enumerate(names)},
            "recall": {nm: (float(((gt[val] == i) & (pr[val] == i)).sum() /
                                  max(1, (gt[val] == i).sum()))
                            if (gt[val] == i).sum() else None)
                       for i, nm in enumerate(names)}}

    # --- what does the model emit where the VLM CoT flagged a lane change? ---
    with open(a.arm_json, encoding="utf-8") as fh:
        man = json.load(fh)["refcv3"]["manifest"]
    clip_of = {int(e["file_index"]): e["clip_id"] for e in man["episodes"]}
    lc_clips = set()
    with gzip.open(LAB, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            t = (r.get("cot_tokens") or {}).get("lane_change")
            g = (r.get("g_tac") or {}).get("goals") or {}
            if t not in (None, "", False) or ({"LANE_CHANGE_L",
                                               "LANE_CHANGE_R"} & set(g)):
                lc_clips.add(r["clip_id"])
    mask = np.array([clip_of.get(int(e), "") in lc_clips for e in D["_epi"]])
    print(f"\n=== windows whose CLIP the VLM CoT / goal layer flags as a LANE "
          f"CHANGE ===")
    print(f"  {int(mask.sum())} windows over "
          f"{len({clip_of[int(e)] for e in D['_epi'][mask]}) if mask.any() else 0}"
          f" clips of the {len(files)} in this eval set")
    if mask.any():
        c = collections.Counter(D["lat_pred_nav_true"][mask].tolist())
        print(f"  the model's LATERAL emission there: "
              f"{ {LAT[k]: v for k, v in sorted(c.items())} }")
        cl = collections.Counter(D["lat_label"][mask].tolist())
        print(f"  the v7.2 LABEL there:               "
              f"{ {(LAT[k] if k>=0 else 'IGNORE'): v for k, v in sorted(cl.items())} }")
        res["cot_lane_change_windows"] = {
            "n_windows": int(mask.sum()),
            "pred": {LAT[k]: v for k, v in sorted(c.items())},
            "label": {(LAT[k] if k >= 0 else "IGNORE"): v
                      for k, v in sorted(cl.items())}}

    # --- determinism replicate ---------------------------------------------
    if a.replicate_dump:
        R, rf = load(a.replicate_dump)
        m = min(len(R["ws"]), len(D["ws"]))
        k = len(R["ws"])
        same_sel = bool(np.array_equal(D["sel_idx"][:k], R["sel_idx"]))
        same_lat = bool(np.array_equal(D["lat_pred_nav_true"][:k],
                                       R["lat_pred_nav_true"]))
        print(f"\n=== DETERMINISM REPLICATE ({len(rf)} episodes, {k} windows, "
              f"a SEPARATE process, same flags, same seed) ===")
        print(f"  sel_idx bit-identical            : {same_sel}")
        print(f"  lat_pred_nav_true bit-identical  : {same_lat}")
        print(f"  => inference-run variance on this arm is "
              f"{'EXACTLY ZERO on these windows, so every switch measured '
                 'below is the model reacting to its input, not a sampler'
                 if same_sel else 'NON-ZERO -- the jitter number must be read '
                 'against this floor'}")
        res["determinism"] = {"n_windows": int(k), "sel_idx_identical":
                              same_sel, "lat_pred_identical": same_lat}

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
