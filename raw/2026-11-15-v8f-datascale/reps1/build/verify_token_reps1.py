"""verify_token_reps1.py ARM A (pod, CPU): the runner's own refusal logic (``token_problems``) applied to the minted token of UE_s1 / UM_s1 with EXACTLY the arguments chk_arm_reps1.sh run passes, plus tampered copies
that must be refused. FORK of blockum/verify_token_um.py and block3/verify_token3.py (checker from blockreps1/study, tree and commit by arm). Writes gate/verify_<ARM>.A.json, prints PASS / FAIL, rc 1 on FAIL. Starts nothing."""
import copy
import json
import os
import shutil
import sys

D = "/root/dscale10k"
arm, stage = sys.argv[1], sys.argv[2]
assert arm in ("UE_s1", "UM_s1") and stage == "A", "verify_token_reps1.py serves UE_s1 / UM_s1 stage A"
TREE = D + ("/tree" if arm == "UE_s1" else "/tree_um")
COMMIT = "cdbff971fa6dad4d20b937fe8abda6cb9f622adb" if arm == "UE_s1" else "648d4ca3cab588d738f153872473bd3cb6f5e40e"
for p in (TREE + "/stack", TREE + "/stack/scripts", TREE, D + "/study", D + "/blockreps1/study"):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
import dscale10k_check as C  # noqa: E402
if os.path.realpath(C.__file__) != os.path.realpath(D + "/blockreps1/study/dscale10k_check.py"):
    sys.exit(f"REFUSED: imported the checker from {C.__file__}, not {D}/blockreps1/study/dscale10k_check.py (the one mint and run use)")

toks = json.load(open(f"{D}/argv/{arm}.A.json"))["argv"]
gd = f"{D}/gate/tokens/{arm}.{stage}"
extras = {"evidence": f"{D}/gate/evidence_{arm}.json", "dry_record": f"{D}/dry/{arm}.A.dry.json", "merge_record": f"{D}/merged/merge_record.json",
          "--union-release": f"{D}/merged/v9_labels_union.npz", "tenk_v9": f"{D}/in/labels/v9_labels_datascale10k.npz"}
if arm == "UM_s1":
    extras.update({"tacsrc_list": f"{D}/mix/bset_clips.txt", "ue_argv": f"{D}/argv/UE_s1.A.json", "b_dry": f"{D}/dry/B_s0.A.dry.json"})
tok_p = gd + "/gate_token.json"
good = C.token_problems(tok_p, arm, stage, toks, TREE, COMMIT, extras)
regs = []
tmp = D + "/tmp/verify_token_reps1_copy"
os.makedirs(tmp, exist_ok=True)
t = json.load(open(tok_p))
t2 = copy.deepcopy(t)
t2["checks"]["G-LIVE"]["pass"] = False
json.dump(t2, open(tmp + "/t_edit.json", "w"))
regs.append(("token_body_edited", C.token_problems(tmp + "/t_edit.json", arm, stage, toks, TREE, COMMIT, extras)))
regs.append(("argv_one_flag_added", C.token_problems(tok_p, arm, stage, toks + ["--lr", "1e-3"], TREE, COMMIT, extras)))
regs.append(("argv_seed_changed_to_0", C.token_problems(tok_p, arm, stage, [("0" if toks[i - 1] == "--seed" else x) for i, x in enumerate(toks)], TREE, COMMIT, extras)))
if arm == "UM_s1":
    regs.append(("list_file_missing", C.token_problems(tok_p, arm, stage, toks, TREE, COMMIT, {**extras, "tacsrc_list": D + "/mix/bset_clips.txt.nonexistent"})))
regs.append(("other_commit", C.token_problems(tok_p, arm, stage, toks, TREE, "deadbeef", extras)))
regs.append(("runner_without_the_bound_extra_files", C.token_problems(tok_p, arm, stage, toks, TREE, COMMIT, None)))
shutil.rmtree(tmp, ignore_errors=True)                                  # own temporary copy
out = {"arm": arm, "stage": stage, "token": tok_p, "problems": good, "body_sha256": t.get("body_sha256"), "bound_data": sorted(t.get("data", {})), "required_checks": sorted(C.required_checks(arm, stage)),
       "checks_pass": {k: v.get("pass") for k, v in t["checks"].items()}, "regression": [{"arm": n, "red": bool(p), "problems": p[:2]} for n, p in regs]}
out["pass"] = not good and all(r["red"] for r in out["regression"])
json.dump(out, open(f"{D}/gate/verify_{arm}.{stage}.json", "w"), indent=1)
print("verify", arm, stage, "PASS" if out["pass"] else "FAIL", good[:2], [r["red"] for r in out["regression"]])
sys.exit(0 if out["pass"] else 1)
