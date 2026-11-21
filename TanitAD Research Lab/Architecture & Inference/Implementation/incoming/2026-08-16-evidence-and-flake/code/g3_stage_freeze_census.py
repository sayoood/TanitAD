"""Per-stage x per-group TRAINABLE-parameter census, with the interpretation
head ON — the measurement that decides whether the freeze audit overstates.

WHY THIS SCRIPT EXISTS
----------------------
`EVIDENCE_AND_FLAKE.md` §2.6 measured the `interp`/S-J defect on the TINY test
stack (62 parameter TENSORS). That is enough to establish the STRUCTURE but not
the SCALE, and the brief for the fix asked for the number that actually ships.
This runs the same census at the **production geometry** (default `V6Config`
plus `agent_slots=True`), for **every** stage, and reports per group.

⛔ NO GPU. Construction + `requires_grad` flips only — no forward, no backward.

REPRODUCING THE BEFORE ARM
--------------------------
The fix is STAGED, NOT COMMITTED, so `HEAD` still carries the pre-fix module:

    git show HEAD:stack/tanitad/models/v6.py > <tmp>/v6.py   # pre-fix
    # ...and run this script against a tree carrying that file.

Both arms in `raw/stage_freeze_census.json` were produced by THIS script, on
this dev box, one before the `v6.py` edit and one after.

⚠️ The emitted block is delimited by opaque markers (`ZZBEGINZZ`/`ZZENDZZ`) and
the parser never greps the stream for words the command contains — the
CLAUDE.md self-matching-filter trap, which has now cost this programme three
separate false readings (most recently `xfailed` matching a grep for `failed`).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "stack"))

from tanitad.models.v6 import (  # noqa: E402
    MODULE_GROUPS, STAGE_GROUPS, STAGES, V6Config, V6Stack,
    apply_stage_freeze, stage_trainable_groups)


def census(label: str) -> dict:
    out: dict = {"_label": label}

    # ---- 1. the ALIAS, checked rather than assumed ----------------------
    # `STAGE_GROUPS["S-J"] is MODULE_GROUPS` was an IDENTITY alias: editing
    # MODULE_GROUPS silently edited S-J. Tuples are immutable so nothing could
    # mutate ACROSS it — the hazard is the invisible one-way coupling, which is
    # exactly how `interp` entered S-J in 06b8782 without S-J's line changing.
    out["alias"] = {
        "S-J_is_MODULE_GROUPS": STAGE_GROUPS["S-J"] is MODULE_GROUPS,
        "MODULE_GROUPS_type": type(MODULE_GROUPS).__name__,
        "MODULE_GROUPS": list(MODULE_GROUPS),
        "STAGE_GROUPS": {k: list(v) for k, v in STAGE_GROUPS.items()},
    }
    try:
        from tanitad.models.v6 import LADDER_UNTRAINED_GROUPS
        out["alias"]["LADDER_UNTRAINED_GROUPS"] = sorted(
            LADDER_UNTRAINED_GROUPS)
    except ImportError:
        out["alias"]["LADDER_UNTRAINED_GROUPS"] = None   # pre-fix tree

    # ---- 2. ⛔ THE SAFETY INVARIANT --------------------------------------
    # The live 30k v6F S-W run resumes TENSOR-STRICT. 87,893,449 / 405 is what
    # a broken strict resume would kill, so it is measured on BOTH arms and the
    # fix is admissible only if it is bit-inert here.
    torch.manual_seed(0)
    default = V6Stack(V6Config())
    out["default_build"] = {
        "params": int(sum(p.numel() for p in default.parameters())),
        "state_dict_keys": len(default.state_dict()),
    }
    del default

    # ---- 3. the head ON, at the PRODUCTION geometry ----------------------
    torch.manual_seed(0)
    slot = V6Stack(V6Config(agent_slots=True))
    per_group_params = {g: 0 for g in MODULE_GROUPS}
    per_group_tensors = {g: 0 for g in MODULE_GROUPS}
    for n, p in slot.named_parameters():
        per_group_params[slot.group_of(n)] += int(p.numel())
        per_group_tensors[slot.group_of(n)] += 1
    out["slot_build"] = {
        "params": int(sum(p.numel() for p in slot.parameters())),
        "state_dict_keys": len(slot.state_dict()),
        "per_group_params": per_group_params,
        "per_group_tensors": per_group_tensors,
    }

    stages = {}
    for st in STAGES:
        rep = apply_stage_freeze(slot, st)
        stages[st] = {
            "declared_trainable_groups": list(stage_trainable_groups(st)),
            "audit_trainable_groups": rep["trainable_groups"],
            "per_group_trainable": {g: rep["per_group"][g]["trainable"]
                                    for g in MODULE_GROUPS},
            "per_group_frozen": {g: rep["per_group"][g]["frozen"]
                                 for g in MODULE_GROUPS},
            "n_trainable": rep["n_trainable"],
            "n_frozen": rep["n_frozen"],
            "n_trainable_tensors": sum(
                1 for p in slot.parameters() if p.requires_grad),
        }
    out["stages_with_head_ON"] = stages
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=["before", "after"])
    a = ap.parse_args()
    print("ZZBEGINZZ")
    print(json.dumps(census(a.label), indent=1, sort_keys=True))
    print("ZZENDZZ")
