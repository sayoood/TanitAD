r"""Training-window prevalence of the STOP_POINT goal and the lon STOP class, from the DAG-OACC argv's own training join.
join_train.npz = box 2 D:\tanitad\runs\v8f_mini_full\join_train.npz (the file the DAG-OACC argv names via --join-train, built from the 4,572-clip v9 release md5 83e30bd3).
Windows exactly as the trainer: clips minus `--exclude 540:600` (sorted by sha12), D.train_windows(hist 4 -> first window t = 3, t in [3, T-31)); label read at row (ci, t) (a_t = 0; T's tick may be older by 0-max_age_t frames: UNVERIFIED effect, same labels shifted a few frames)."""
import json, sys, math
import numpy as np
from tanitad.data import v8f_mini_data as D
J = D.load_join(sys.argv[1])
clips_all = [str(c) for c in J["clip"].tolist()]
xs = sorted(clips_all, key=D.sha12)[540:600]
ex = set(xs)
clips = [c for c in clips_all if c not in ex]
wins = D.train_windows(None, J, D.select_train_clips(clips, 1.0), hist=4, start=None)
rows = np.array([int(J["off"][ci]) + t for ci, t, _, _ in wins])
has = J["has"][rows].astype(bool)
gy, gw = J["goal_y"][rows].astype(np.int64), J["goal_w"][rows].astype(np.int64)
obs = has & (((gw >> 6) & 1) == 1)
pos = obs & (((gy >> 6) & 1) == 1)
neg = obs & ~pos
la, lc = J["lon_allowed"][rows].astype(np.int64), J["lon_cls"][rows].astype(np.int64)
valid = has & (la != 0)
lstop = valid & (((la >> 2) & 1) == 1)
lstop_single = valid & (la == 4)
res = {"n_clips_all": len(clips_all), "n_excluded": len(ex), "n_clips_train": len(clips), "n_windows": len(wins), "n_has": int(has.sum()),
       "goal_stop": {"n_observed": int(obs.sum()), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "prevalence": float(pos.sum() / obs.sum()), "W_sqrt_neg_over_pos": float(math.sqrt(neg.sum() / pos.sum()))},
       "lon_stop": {"n_valid": int(valid.sum()), "n_stop_rows(allowed contains STOP)": int(lstop.sum()), "n_stop_only_allowed": int(lstop_single.sum()), "n_lon_cls_eq_2": int((valid & (lc == 2)).sum()),
                    "prevalence": float(lstop.sum() / valid.sum()), "W_sqrt_other_over_stop": float(math.sqrt((valid.sum() - lstop.sum()) / lstop.sum()))},
       "lon_cls_hist(valid rows, LON9 order)": [int((valid & (lc == k)).sum()) for k in range(7)]}
print(json.dumps(res, indent=1))
open(sys.argv[2], "w", encoding="utf-8").write(json.dumps(res, indent=1))
