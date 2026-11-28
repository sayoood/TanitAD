#!/bin/bash
# process-level OFF-identity through online_labeller._job: tip vs overlay(OFF) vs overlay(ON, twice).   usage: run_identity.sh <scenes_dir> <work_dir>
export A2_GATE_DATA="$(cd "$(dirname "$0")/.." && pwd)/gate_data" PYTHONDONTWRITEBYTECODE=1
PY=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
S="${1:?}"; W="${2:?}"; mkdir -p "$W"
$PY "$(dirname "$0")/job_identity_child.py" F:/a2_scratch/tip_refe "$S" "$W/job_tip_off.npz" 0
$PY "$(dirname "$0")/job_identity_child.py" F:/a2_scratch/merged_refe "$S" "$W/job_ov_off.npz" 0
$PY "$(dirname "$0")/job_identity_child.py" F:/a2_scratch/merged_refe "$S" "$W/job_ov_on.npz" 1
$PY "$(dirname "$0")/job_identity_child.py" F:/a2_scratch/merged_refe "$S" "$W/job_ov_on2.npz" 1
$PY - "$W" <<'PYEOF'
import sys, numpy as np, json
W = sys.argv[1]
a, b, c, d = (np.load(f"{W}/job_{n}.npz") for n in ("tip_off", "ov_off", "ov_on", "ov_on2"))
lk = [k for k in a.files if k.startswith("labels")]
res = dict(n_scenes=len(lk), tip_vs_overlay_off_bytes_equal=all(a[k].tobytes() == b[k].tobytes() for k in lk),
           overlay_off_vs_on_labels_bytes_equal=all(b[k].tobytes() == c[k].tobytes() for k in lk),
           on_vs_on_two_processes_bytes_equal=all(c[k].tobytes() == d[k].tobytes() for k in c.files),
           off_has_no_behaviour_arrays=not any(k.startswith(("beh", "act")) for k in b.files), on_has_behaviour_arrays=sum(k.startswith("beh") for k in c.files))
print(json.dumps(res, indent=1)); json.dump(res, open(f"{W}/job_identity_result.json", "w"), indent=1)
PYEOF
