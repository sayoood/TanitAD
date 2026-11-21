"""verify_token.py ARM STAGE (pod, CPU): the runner's own refusal logic (``token_problems``) applied to the minted token with EXACTLY the arguments chk_arm.sh run passes, plus three tampered copies that must be refused.
Writes /root/dscale10k/gate/verify_<ARM>.<STAGE>.json and prints PASS / FAIL. Starts nothing."""
import copy
import json
import os
import shutil
import sys

D = "/root/dscale10k"
TREE = D + "/tree"
for p in (TREE + "/stack", TREE + "/stack/scripts", TREE, D + "/study"):
    if p not in sys.path:
        sys.path.insert(0, p)
import dscale10k_check as C  # noqa: E402

arm, stage = sys.argv[1], sys.argv[2]
COMMIT = "cdbff971fa6dad4d20b937fe8abda6cb9f622adb"
argv_p = f"{D}/argv/{arm}.A.json" if stage == "A" else f"{D}/runs/{arm}/argv_B.json"
toks = json.load(open(argv_p))["argv"]
gd = f"{D}/gate/tokens/{arm}.{stage}"
extras = {"evidence": f"{D}/gate/evidence_{arm}.json", "dry_record": f"{D}/dry/{arm}.A.dry.json"}
if arm in ("U_s0", "R_s0"):
    extras.update({"merge_record": f"{D}/merged/merge_record.json", "--union-release": f"{D}/merged/v9_labels_union.npz", "tenk_v9": f"{D}/in/labels/v9_labels_datascale10k.npz"})
if arm == "R_s0":
    extras["r_release"] = f"{D}/merged/v9_labels_R_s0.npz"
tok_p = gd + "/gate_token.json"
good = C.token_problems(tok_p, arm, stage, toks, TREE, COMMIT, extras)
regs = []
tmp = D + "/tmp/verify_token_copy"
os.makedirs(tmp, exist_ok=True)
t = json.load(open(tok_p))
t2 = copy.deepcopy(t)
t2["checks"]["G-LIVE"]["pass"] = False
json.dump(t2, open(tmp + "/t_edit.json", "w"))
regs.append(("token_body_edited", C.token_problems(tmp + "/t_edit.json", arm, stage, toks, TREE, COMMIT, extras)))
regs.append(("argv_one_flag_added", C.token_problems(tok_p, arm, stage, toks + ["--lr", "1e-3"], TREE, COMMIT, extras)))
regs.append(("other_commit", C.token_problems(tok_p, arm, stage, toks, TREE, "deadbeef", extras)))
regs.append(("runner_without_the_bound_extra_files", C.token_problems(tok_p, arm, stage, toks, TREE, COMMIT, None)))
shutil.rmtree(tmp, ignore_errors=True)                                                                  # own temporary copy
out = {"arm": arm, "stage": stage, "token": tok_p, "problems": good, "body_sha256": t.get("body_sha256"), "bound_data": sorted(t.get("data", {})), "required_checks": sorted(C.required_checks(arm, stage)),
       "checks_pass": {k: v.get("pass") for k, v in t["checks"].items()}, "regression": [{"arm": n, "red": bool(p), "problems": p[:2]} for n, p in regs]}
out["pass"] = not good and all(r["red"] for r in out["regression"])
json.dump(out, open(f"{D}/gate/verify_{arm}.{stage}.json", "w"), indent=1)
print("verify", arm, stage, "PASS" if out["pass"] else "FAIL", good[:2], [r["red"] for r in out["regression"]])
