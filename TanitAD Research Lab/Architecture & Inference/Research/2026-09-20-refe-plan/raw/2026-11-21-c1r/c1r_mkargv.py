"""MM 2026-11-21: C1r's launch argv (C1's seed-1 replicate, PI go 2026-11-20) from the pinned template launch_gate.p5c_argv("C1r") in the P5-C package,
with the same paths as ARGS_C1.txt.  Writes ARGS_C1r.txt and asserts the C1 -> C1r difference is exactly --seed (0 -> 1) and --out (p5c_c1 -> p5c_c1r).
Pod defaults: PKG = the landed refe dir's parent, DIR = /workspace/data/refe_p5c (holds ARGS_C1.txt).  Local run: C1R_PKG_REFE=<.../refe>  C1R_DIR=<dir holding ARGS_C1.txt>  C1R_OUT=<ARGS_C1r.txt path>."""
import hashlib, os, sys
REFE = os.environ.get("C1R_PKG_REFE", "/workspace/data/refe_p5c/code_33824420/refe")
DIR = os.environ.get("C1R_DIR", "/workspace/data/refe_p5c")
sys.path.insert(0, REFE)
import launch_gate as L  # noqa: E402
assert os.path.dirname(os.path.abspath(L.__file__)) == os.path.abspath(REFE), L.__file__
c1 = open(os.path.join(DIR, "ARGS_C1.txt")).read().split()


def val(argv, flag):
    return argv[argv.index(flag) + 1]


paths = {"targets": val(c1, "--targets"), "images": val(c1, "--images"), "calib": val(c1, "--calib"), "metric_cache": val(c1, "--metric-cache"),
         "init_from": val(c1, "--init-from"), "ego_yaw_table": val(c1, "--ego-yaw-table"), "traj_label_table": val(c1, "--traj-label-table"),
         "out_root": "/workspace/data/refe_runs", "label_workers": val(c1, "--label-workers")}
a = L.p5c_argv("C1r", paths=paths)
assert len(a) == len(c1), (len(a), len(c1))
d = [(c1[i - 1], c1[i], a[i]) for i in range(len(a)) if a[i] != c1[i]]
assert d == [("--seed", "0", "1"), ("--out", "/workspace/data/refe_runs/p5c_c1", "/workspace/data/refe_runs/p5c_c1r")], d
out = os.environ.get("C1R_OUT", os.path.join(DIR, "ARGS_C1r.txt"))
open(out, "w", newline="\n").write(" ".join(a) + "\n")
print("ZZC1R_ARGV_OK", hashlib.sha256(" ".join(a).encode()).hexdigest())
