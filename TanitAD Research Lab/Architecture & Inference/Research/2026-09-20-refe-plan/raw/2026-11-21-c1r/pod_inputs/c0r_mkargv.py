"""MM 2026-11-18: C0r's launch argv (P5-C's zero-lever replicate, seed 1) from the pinned template launch_gate.p5c_argv("C0r") in the P5-C package,
with the same paths as ARGS_C0.txt (mk_argv.py). Writes ARGS_C0r.txt and asserts the C0 -> C0r difference is exactly --seed (0 -> 1) and --out (p5c_c0 -> p5c_c0r).
Run only when the P5-C read triggers C0r (C0R_DECISION.txt = RUN), by post_rep_chain.sh."""
import hashlib, json, os, sys
PKG = "/workspace/data/refe_p5c/code_33824420"
sys.path.insert(0, PKG + "/refe")
import launch_gate as L  # noqa: E402
assert L.__file__.startswith(PKG), L.__file__
cfg = json.load(open("/workspace/data/refe_runs/p7_reserve_head/config.json"))
p7 = cfg["argv"][1:]


def val(flag):
    return p7[p7.index(flag) + 1]


paths = {"targets": val("--targets"), "images": val("--images"), "calib": val("--calib"), "metric_cache": val("--metric-cache"),
         "init_from": "/workspace/data/refe_runs/p7_reserve_head/model_final.pt", "ego_yaw_table": val("--ego-yaw-table"),
         "out_root": "/workspace/data/refe_runs", "label_workers": val("--label-workers")}
for k, v in paths.items():
    if k not in ("out_root", "label_workers"):
        assert os.path.exists(v), (k, v)
a = L.p5c_argv("C0r", paths=paths)
c0 = open("/workspace/data/refe_p5c/ARGS_C0.txt").read().split()
assert len(a) == len(c0), (len(a), len(c0))
d = [(c0[i - 1], c0[i], a[i]) for i in range(len(a)) if a[i] != c0[i]]
assert d == [("--seed", "0", "1"), ("--out", "/workspace/data/refe_runs/p5c_c0", "/workspace/data/refe_runs/p5c_c0r")], d
open("/workspace/data/refe_p5c/ARGS_C0r.txt", "w").write(" ".join(a) + "\n")
print("ZZC0R_ARGV_OK argv_sha16", hashlib.sha256(" ".join(a).encode()).hexdigest()[:16])
