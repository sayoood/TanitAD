"""Compare a control build (ctlA / ctlB / wrong) with the existing v9 release rows, per field, for the control clips found in the
release. Reports per-field equality (bitwise, NaN-equal) and the max abs diff for float fields."""
import json, sys
import numpy as np
OUT = "/root/geo10k/out"
mode = sys.argv[1]
rel = np.load("/root/geo10k/ref/tanitad-refcv8-v9-labels/release/v9_labels_train.npz", allow_pickle=False)
mine = np.load(f"{OUT}/v9_labels_{mode}.npz", allow_pickle=False)
rs, r0, rn = rel["clip__sid"], rel["clip__row0"], rel["clip__n_rows"]
rpos = {int(s): i for i, s in enumerate(rs)}
ms, m0, mn = mine["clip__sid"], mine["clip__row0"], mine["clip__n_rows"]
fields = sorted(f for f in mine.files if f.startswith("row__"))
common = [i for i, s in enumerate(ms) if int(s) in rpos]
print(f"[g4 {mode}] control clips {len(ms)}; present in release {len(common)}")
per_field = {f: {"equal": True, "max_abs": 0.0, "n_clips_differ": 0} for f in fields}
miss = [f for f in fields if f not in rel.files]
clip_diff = {}
for f in fields:
    if f not in rel.files:
        per_field[f]["equal"] = None
        continue
    A, B = mine[f], rel[f]
    for i in common:
        j = rpos[int(ms[i])]
        a, b = A[m0[i]:m0[i] + mn[i]], B[r0[j]:r0[j] + rn[j]]
        if a.shape != b.shape:
            eq, d = False, float("inf")
        else:
            eq = bool(np.array_equal(a, b, equal_nan=True)) if a.dtype.kind in "fc" else bool(np.array_equal(a, b))
            d = 0.0
            if not eq and a.dtype.kind in "fiu":
                aa, bb = a.astype(np.float64), b.astype(np.float64)
                m = ~(np.isnan(aa) & np.isnan(bb))
                d = float(np.nanmax(np.abs(np.where(m, aa - bb, 0.0)))) if m.any() else 0.0
                if np.isnan(aa[m]).any() != np.isnan(bb[m]).any() or (np.isnan(aa[m]) ^ np.isnan(bb[m])).any():
                    d = float("inf")
        if not eq:
            per_field[f]["equal"] = False
            per_field[f]["n_clips_differ"] += 1
            per_field[f]["max_abs"] = max(per_field[f]["max_abs"], d)
            clip_diff.setdefault(int(ms[i]), set()).add(f)
n_eq = sum(1 for v in per_field.values() if v["equal"] is True)
rep = {"mode": mode, "n_control_clips": len(ms), "n_in_release": len(common), "n_fields": len(fields), "n_fields_equal_all_clips": n_eq,
       "fields_not_in_release": miss, "differing_fields": {f: v for f, v in per_field.items() if v["equal"] is False},
       "n_clips_with_any_diff": len(clip_diff)}
json.dump(rep, open(f"{OUT}/compare_{mode}.json", "w"), indent=1, default=str)
print(f"[g4 {mode}] fields equal on all clips: {n_eq}/{len(fields)}; clips with any diff: {len(clip_diff)}/{len(common)}")
for f, v in list(rep["differing_fields"].items())[:60]:
    print("   DIFF", f, v)
