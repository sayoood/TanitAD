#!/usr/bin/env python3
"""verify_blobs: after apply_all.sh, print the git blob of every touched file and check the pins the patches promise.  usage: verify_blobs.py <PK dir>
Exit 0 iff: every BASE blob was the tip's (checked by the patch scripts), rl_reward.py is byte-unchanged (cd5c5144), and reversing the trainer patch gives 7c8071ea (I5)."""
import json
import os
import sys

import patchlib as PL

TOUCHED = ["refe/scorer_finetune.py", "refe/launch_gate.py", "refe/gate_checks.py", "refe/rls1_checks.py", "refe/rls1_pod.sh",
           "refe/test_rl_s1b.py", "refe/test_rl_r0c.py", "refe/test_rl_epo.py", "refe/test_launch_gate.py", "refe/test_rls1_chain.py", "refe/test_rls1_queue.py",
           "refe/test_rl_safeobj.py", "eval/rls1d_arms.json", "eval/safeobj_read.py", "eval/test_safeobj_read.py", "eval/test_safeobj_gate.py"]
UNCHANGED = {"refe/rl_reward.py": "cd5c5144"}


def main():
    pk = PL.pk_arg()
    out, bad = {}, []
    for rel in TOUCHED:
        out[rel] = PL.git_blob(PL.read(os.path.join(pk, rel)))
    for rel, pre in UNCHANGED.items():
        b = PL.git_blob(PL.read(os.path.join(pk, rel)))
        out[rel] = b
        if not b.startswith(pre):
            bad.append(f"{rel} changed: {b} (must start with {pre})")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from patch_trainer import TRAINER_EDITS
    src = PL.read(os.path.join(pk, "refe/scorer_finetune.py"))
    for old, new in reversed(TRAINER_EDITS):
        if src.count(new) != 1:
            bad.append(f"trainer reversal: a patch text occurs {src.count(new)} times")
            break
        src = src.replace(new, old)
    else:
        if not PL.git_blob(src).startswith("7c8071ea"):
            bad.append(f"I5: the reversed trainer is {PL.git_blob(src)}, not 7c8071ea")
    print(json.dumps({"blobs": out, "problems": bad}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
