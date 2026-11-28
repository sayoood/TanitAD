"""tac_echo_check.py -- is the TACTICAL lateral-head accuracy gain of the per-frame nav feed vision skill, or the fed token?

Reads the nav-feed experiment's banked decision dumps (raw/run/dump_<arm>/decisions/ep*.npz) and prints, per arm:
  * the crosstab fed nav token -> head lateral prediction and fed token -> lateral LABEL;
  * head lateral accuracy against a CONSTANT majority-class control (the no-information value on these windows);
  * the share of windows where the prediction equals the token's modal prediction (an echo index).
The nav_zero / nav_shuffled head passes are NOT run in this lane (their arrays are the -1 sentinel), so no intervention read
exists here.  CPU, numpy only.   python -I tac_echo_check.py <raw/run dir> --out <json>
"""
import argparse, collections, glob, json, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--out", required=True); a = ap.parse_args()
res = {}
for arm in ("dump_clip_s0", "dump_clip_s1", "dump_perframe_full_s0", "dump_perframe_full_s1", "dump_perframe_token_s0", "dump_perframe_token_s1"):
    fs = sorted(glob.glob(os.path.join(a.run_dir, arm, "decisions", "ep*.npz")))
    if not fs:
        res[arm] = {"status": "ABSENT"}; continue
    R = collections.defaultdict(list)
    for f in fs:
        z = np.load(f)
        for k in ("nav_cmd", "lat_pred_nav_true", "lat_pred_nav_zero", "lat_pred_nav_shuffled", "lat_label"):
            R[k].append(z[k])
    R = {k: np.concatenate(v) for k, v in R.items()}
    ok = R["lat_label"] >= 0
    fed, pred, lab = R["nav_cmd"][ok], R["lat_pred_nav_true"][ok], R["lat_label"][ok]
    major = collections.Counter(lab.tolist()).most_common(1)[0][0]
    tok_mode = {int(v): collections.Counter(pred[fed == v].tolist()).most_common(1)[0][0] for v in set(fed.tolist())}
    res[arm] = {
        "n_windows_with_lat_label": int(ok.sum()), "n_episode_files": len(fs),
        "crosstab_fed_to_pred": {int(v): dict(sorted(collections.Counter(pred[fed == v].tolist()).items())) for v in sorted(set(fed.tolist()))},
        "crosstab_fed_to_label": {int(v): dict(sorted(collections.Counter(lab[fed == v].tolist()).items())) for v in sorted(set(fed.tolist()))},
        "head_lat_acc": round(float((pred == lab).mean()), 4),
        "constant_majority_class": int(major), "constant_majority_acc": round(float((lab == major).mean()), 4),
        "echo_index_pred_eq_token_mode": round(float(np.mean([tok_mode[int(t)] == int(p) for t, p in zip(fed, pred)])), 4),
        "nav_zero_pass_run": bool((R["lat_pred_nav_zero"] >= 0).any()), "nav_shuffled_pass_run": bool((R["lat_pred_nav_shuffled"] >= 0).any()),
    }
json.dump(res, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
for k, v in res.items():
    print(k, {x: v.get(x) for x in ("n_windows_with_lat_label", "head_lat_acc", "constant_majority_acc", "echo_index_pred_eq_token_mode", "nav_zero_pass_run")})
