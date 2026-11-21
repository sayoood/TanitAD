"""Do the two structural findings survive at the CURRENT branch tip?

⛔ An absence claim goes stale the moment someone lands code. This package was
measured on a worktree detached at `faecb29`; the branch moved to `bf8dad9`
while it ran. This re-asserts, against the NEWER blobs read out of git, that

  * no trainer/model file constructs the map head or the 3-D box head, and
  * `build_parser` still exposes no flag for the SPEC §4 tactical behaviour
    decoder or the SPEC §5 four-value max-speed input.

Every check is a POSITIVE assertion in both directions: a marker that MUST be
present is counted in the same breath as one that must be absent, so a blob
that could not be read is distinguishable from a genuine zero (CLAUDE.md: a
count of 0 from a file that could not be READ is not an absence).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

WT = Path(os.environ.get("TANITAD_WT", "C:/Users/Admin/tanitad-wt-e2e"))
FILES = ["stack/scripts/refc_v3_train.py", "stack/tanitad/refs/refc_v3.py",
         "stack/tanitad/refs/refc.py"]
ABSENT = {"perception_heads":
          r"BEVMapBranch|Box3DSlotDecoder|Box3DMemory|map_soft_ce|box3d_set_loss|BEVLift"}
PRESENT = {"file_really_read": r"def |import "}   # the same-breath control


def blob(ref: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=WT,
                       capture_output=True)
    return None if r.returncode else r.stdout.decode("utf-8", "replace")


def main(refs=("faecb29", None)):
    tip = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         cwd=WT, capture_output=True, text=True).stdout.strip()
    heads = subprocess.run(["git", "rev-parse", "--short",
                            "agent/arch-inf-20260803"], cwd=WT,
                           capture_output=True, text=True).stdout.strip()
    out = {"evidence_class": "MEASURED", "worktree_head": tip,
           "branch_tip": heads, "refs": {}}
    for ref in [r for r in refs if r] + ([heads] if heads else []):
        per = {}
        for f in FILES:
            src = blob(ref, f)
            if src is None:
                per[f] = {"read": False}
                continue
            row = {"read": True, "n_lines": src.count("\n")}
            for k, rx in PRESENT.items():
                row[k] = len(re.findall(rx, src))
            for k, rx in ABSENT.items():
                row[k] = len(re.findall(rx, src))
            row["tac_decoder_v6_mentions"] = len(
                re.findall(r"tac_decoder_v6", src))
            row["max_speed_onehot_v6_mentions"] = len(
                re.findall(r"max_speed_onehot_v6", src))
            if f.endswith("refc_v3_train.py"):
                row["cli_flags"] = sorted(set(
                    re.findall(r'add_argument\("(--[a-z0-9-]+)', src)))
                row["n_cli_flags"] = len(row["cli_flags"])
                row["has_tac_decoder_flag"] = any(
                    "tac-decoder" in x for x in row["cli_flags"])
                row["has_max_speed_onehot_flag"] = any(
                    "max-speed-onehot" in x for x in row["cli_flags"])
                row["has_any_perception_flag"] = any(
                    ("map" in x or "box" in x or "sam3" in x or "cuboid" in x
                     or "percep" in x) for x in row["cli_flags"])
            per[f] = row
        out["refs"][ref] = per
    # the verdict, derived
    v = {}
    for ref, per in out["refs"].items():
        tr = per.get("stack/scripts/refc_v3_train.py", {})
        unread = [f for f, r in per.items() if not r.get("read")]
        v[ref] = {
            "unreadable_files": unread,
            "perception_heads_referenced_anywhere": sum(
                r.get("perception_heads", 0) for r in per.values()
                if r.get("read")),
            "trainer_mentions_tac_decoder_v6":
                tr.get("tac_decoder_v6_mentions"),
            "trainer_has_tac_decoder_flag": tr.get("has_tac_decoder_flag"),
            "trainer_has_max_speed_onehot_flag":
                tr.get("has_max_speed_onehot_flag"),
            "trainer_has_any_perception_flag":
                tr.get("has_any_perception_flag"),
            "n_cli_flags": tr.get("n_cli_flags"),
            "INCONCLUSIVE": bool(unread),
        }
    out["verdict"] = v
    p = Path(__file__).resolve().parents[1] / "raw" / "tip_recheck.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
