"""Do the two in-band rules agree on EVERY eval window? The goal targets use
`v7_labels.window_in_band` (|t - t0| <= tol); the loader's lat/lon/SPEED_BAND labels use
`t0 - tol <= t <= t0 + tol`. A float boundary could split them. Counted over all windows of
every labelled eval episode (str_ext_steps=0 -> the widest window set)."""
import os
from tanitad.data.refav1_loader import RefAV1Windows
from tanitad.data.v7_labels import window_in_band
root = os.environ.get("TANITAD_REFAV1_EVAL_ROOT", "C:/Users/Admin/tanitad-data/refav1-eval141")
lbl = os.environ.get("TANITAD_V72_EVAL_LABELS", "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz")
ld = RefAV1Windows(f"{root}/refav1-fp8-eval", f"{root}/eps", op_window=4, op_steps=30,
                   str_ext_steps=0, lru=1, seed=0, labels_path=lbl, goal_targets=True)
n = agree = n_in = 0
for ei, t in ld.windows:
    rec = ld._v7rec.get(ld.clip_id[ld.names[ei]])
    row = ld._lab.get(ld.clip_id[ld.names[ei]])
    if rec is None or row is None:
        continue
    a = window_in_band(rec, t * ld.dt)
    b = row[3] <= t * ld.dt <= row[4]
    n += 1; agree += int(a == b); n_in += int(b)
print({"n_windows_labelled": n, "agree": agree, "disagree": n - agree, "in_band": n_in})
