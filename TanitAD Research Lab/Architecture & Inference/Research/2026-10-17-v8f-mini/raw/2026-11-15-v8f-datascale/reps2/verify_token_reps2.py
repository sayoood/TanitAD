"""verify_token_reps2.py ARM A (pod, CPU): the runner's own refusal logic (``token_problems``) applied to the minted token of UE_s2 / UE_s3 with EXACTLY the arguments chk_arm_reps2.sh run passes, plus tampered copies
that must be refused. FORK of verify_token_reps1.py (checker from blockreps2/study). Writes gate/verify_<ARM>.A.json, prints PASS / FAIL, rc 1 on FAIL. Starts nothing.
DS_ROOT / DS_GD / DS_REFARGV / DS_EVIDENCE / DS_DRYREC override for a dry run (as chk_arm_reps2.sh); DS_TREE the tree."""
import copy
import json
import os
import shutil
import sys

D = os.environ.get("DS_ROOT", "/root/dscale10k")
arm, stage = sys.argv[1], sys.argv[2]
assert arm in ("UE_s2", "UE_s3") and stage == "A", "verify_token_reps2.py serves UE_s2 / UE_s3 stage A"
TREE = os.environ.get("DS_TREE", "/root/dscale10k/tree")
COMMIT = "cdbff971fa6dad4d20b937fe8abda6cb9f622adb"
RPS = os.environ.get("DS_RP", D + "/blockreps2") + "/study"
for p in (TREE + "/stack", TREE + "/stack/scripts", TREE, "/root/dscale10k/study", RPS):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
import dscale10k_check as C  # noqa: E402
if os.path.realpath(C.__file__) != os.path.realpath(RPS + "/dscale10k_check.py"):
    sys.exit(f"REFUSED: imported the checker from {C.__file__}, not {RPS}/dscale10k_check.py (the one mint and run use)")

toks = json.load(open(f"{D}/argv/{arm}.A.json"))["argv"]
gd = os.environ.get("DS_GD", f"{D}/gate/tokens/{arm}.{stage}")
ref = os.environ.get("DS_REFARGV", f"{D}/argv/UE_s0.A.json")
extras = {"evidence": os.environ.get("DS_EVIDENCE", f"{D}/gate/evidence_{arm}.json"), "dry_record": os.environ.get("DS_DRYREC", f"{D}/dry/{arm}.A.dry.json"), "merge_record": f"{D}/merged/merge_record.json",
          "--union-release": f"{D}/merged/v9_labels_union.npz", "tenk_v9": f"{D}/in/labels/v9_labels_datascale10k.npz", "rep_ref_argv": ref}
tok_p = gd + "/gate_token.json"
good = C.token_problems(tok_p, arm, stage, toks, TREE, COMMIT, extras)
regs = []
tmp = D + "/tmp/verify_token_reps2_copy"
os.makedirs(tmp, exist_ok=True)
t = json.load(open(tok_p))
t2 = copy.deepcopy(t)
t2["checks"]["G-LIVE"]["pass"] = False
json.dump(t2, open(tmp + "/t_edit.json", "w"))
regs.append(("token_body_edited", C.token_problems(tmp + "/t_edit.json", arm, stage, toks, TREE, COMMIT, extras)))
regs.append(("argv_one_flag_added", C.token_problems(tok_p, arm, stage, toks + ["--lr", "1e-3"], TREE, COMMIT, extras)))
regs.append(("argv_seed_changed_to_0", C.token_problems(tok_p, arm, stage, [("0" if toks[i - 1] == "--seed" else x) for i, x in enumerate(toks)], TREE, COMMIT, extras)))
other_seed = str(5 - int(toks[toks.index("--seed") + 1]))                # UE_s2 <-> 3, UE_s3 <-> 2
regs.append(("argv_seed_of_the_other_new_arm", C.token_problems(tok_p, arm, stage, [(other_seed if toks[i - 1] == "--seed" else x) for i, x in enumerate(toks)], TREE, COMMIT, extras)))
regs.append(("other_commit", C.token_problems(tok_p, arm, stage, toks, TREE, "deadbeef", extras)))
regs.append(("runner_without_the_bound_extra_files", C.token_problems(tok_p, arm, stage, toks, TREE, COMMIT, None)))
regs.append(("runner_without_the_reference_argv", C.token_problems(tok_p, arm, stage, toks, TREE, COMMIT, {k: v for k, v in extras.items() if k != "rep_ref_argv"})))
regs.append(("reference_argv_file_missing", C.token_problems(tok_p, arm, stage, toks, TREE, COMMIT, {**extras, "rep_ref_argv": ref + ".nonexistent"})))
shutil.rmtree(tmp, ignore_errors=True)                                  # own temporary copy
out = {"arm": arm, "stage": stage, "token": tok_p, "problems": good, "body_sha256": t.get("body_sha256"), "bound_data": sorted(t.get("data", {})), "required_checks": sorted(C.required_checks(arm, stage)),
       "checks_pass": {k: v.get("pass") for k, v in t["checks"].items()}, "regression": [{"arm": n, "red": bool(p), "problems": p[:2]} for n, p in regs]}
out["pass"] = not good and all(r["red"] for r in out["regression"])
vp = os.environ.get("DS_VERIFY_OUT", f"{D}/gate/verify_{arm}.{stage}.json")
json.dump(out, open(vp, "w"), indent=1)
print("verify", arm, stage, "PASS" if out["pass"] else "FAIL", good[:2], [r["red"] for r in out["regression"]])
sys.exit(0 if out["pass"] else 1)
