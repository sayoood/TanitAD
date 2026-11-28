import json, os, sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
TREE = Path("E:/Projects/TanitAD-artifacts/orcchain_build/tree")
for p in (TREE / "stack", TREE / "taniteval", TREE / "stack" / "scripts"):
    sys.path.insert(0, str(p))
import torch
torch.set_num_threads(1)
import v8f_mini_train as TR
from tanitad.data import v8f_mini_data as D
root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
A = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--log-every", "1", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw"]
src = root / "M2b_s0"
ck = src / "ckpt_last.pt"
assert ck.exists()
def ft(seed, p):
    return ["--init-ckpt", str(ck), "--steps", "3000", "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", p, "--ego-perturb-vscale", "0.5", "1.0",
            "--ego-perturb-vmax", "8.0", "--ego-perturb-adrop", "1.5", "--ego-perturb-yawrate-std", "2.0", "--ego-perturb-seed", str(seed)]
def gate(tag, seed, p):
    try:
        rc = TR.main(["--arm", "M2b", "--seed", "0", "--grid", "pool4x10", "--out", str(root / tag), *A, *ft(seed, p), "--selfcheck-only"])
        return f"{tag} rc={rc}"
    except SystemExit as e:
        return f"{tag} REFUSED: {str(e)[:200]}"
print("CURRENT trainer:")
for tag, seed, p in (("c_orc_s0", 0, "0.3"), ("c_orc_s1", 1, "0.3"), ("c_orc0_s0", 0, "0"), ("c_orc0_s1", 1, "0")):
    print(" ", gate(tag, seed, p))
orig = D.EgoPerturb.problems
def relaxed(self):
    bad = self.law_problems()
    if self.active and self.kin_mode != "yaw":
        bad.append(f"ego_perturb_p {self.p} needs --kin-mode yaw (the lateral state is the measured yaw rate); got {self.kin_mode!r}")
    if self.active and self.vscale == (1.0, 1.0) and self.adrop == 0.0 and self.yawrate_std_deg == 0.0:
        bad.append("ego_perturb_p > 0 but vscale = (1, 1), adrop = 0 and yawrate_std = 0: the perturbation moves nothing -- a lever that is declared and does nothing")
    return bad
D.EgoPerturb.problems = relaxed
print("RELAXED (the p = 0 'declared and does nothing' clause skipped, in-process only):")
for tag, seed, p in (("r_orc0_s0", 0, "0"), ("r_orc0_s1", 1, "0")):
    print(" ", gate(tag, seed, p))
sc = json.load(open(root / "r_orc0_s0" / "selfcheck.json")); print("probe:", sc["ego_perturb_probe"])
print("levers.ego_perturb:", {k: v for k, v in sc["levers"]["ego_perturb"].items() if k.startswith("ego_") or k == "kin_mode"})
