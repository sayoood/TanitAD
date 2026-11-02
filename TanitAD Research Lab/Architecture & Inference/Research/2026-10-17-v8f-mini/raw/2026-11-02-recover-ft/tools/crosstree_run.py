"""usage: crosstree_run.py <tree> <out> [extra trainer flags...]  -- run that tree's v8f_mini_train.main on the synthetic world (6 steps, CPU, 1 thread)."""
import sys

tree, out, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path[:0] = [f"{tree}/stack", f"{tree}/taniteval", f"{tree}/stack/scripts"]
import torch
import v8f_mini_train as TR

torch.set_num_threads(1)
COMMON = ["--synthetic", "--batch", "16", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b"]
print("trainer", TR.__file__)
sys.exit(TR.main(["--out", out, *COMMON, "--steps", "6", "--ego-perturb-p", "0", "--dagger-p", "0", *extra]))
