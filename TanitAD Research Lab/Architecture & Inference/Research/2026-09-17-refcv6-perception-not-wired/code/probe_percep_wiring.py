"""MEASURE whether the refcv6 perception heads reach the TRAINER -- not whether
they exist.

⛔ Every check here is a POSITIVE assertion with a same-breath control that must
read non-zero, because a `grep -c` of 0 from a file that could not be read is
indistinguishable from a genuine absence.

Run:  python probe_percep_wiring.py <stack dir> <out.json>
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path


def _count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text))


def main(stack: str, out_path: str) -> int:
    S = Path(stack)
    trainer = S / "scripts" / "refc_v3_train.py"
    src = trainer.read_text(encoding="utf-8", errors="replace")
    assert len(src) > 100_000, "the trainer did not read -- refusing to report absence"

    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "compute_losses_v3"), None)
    assert fn is not None, "compute_losses_v3 not found -- the probe is pointed wrong"
    body = ast.get_source_segment(src, fn) or ""
    # ⛔ Match the identifier, not a quoting style. My first version required
    # DOUBLE QUOTES and read zero loss keys -- the same-breath control caught it
    # immediately, which is the whole reason the control is there.
    loss_keys = sorted(set(re.findall(r"loss_[a-z0-9_]+", body)))

    # ⭐ the same-breath controls: each MUST be non-zero, or the probe is blind
    controls = {
        "def main": _count(src, r"\bdef main\b"),
        "add_argument": _count(src, r"add_argument"),
        "loss_traj_in_compute_losses_v3": int("loss_traj" in loss_keys),
    }
    assert all(v > 0 for v in controls.values()), f"a control read zero: {controls}"

    calls = {
        "map_soft_ce": _count(src, r"\bmap_soft_ce\b"),
        "box3d_set_loss": _count(src, r"\bbox3d_set_loss\b"),
        "collate_map_targets": _count(src, r"\bcollate_map_targets\b"),
        "collate_box_targets": _count(src, r"\bcollate_box_targets\b"),
        "MapGTStore": _count(src, r"\bMapGTStore\b"),
        "zh_for_frame": _count(src, r"\bzh_for_frame\b"),
    }
    imports = {
        m: _count(src, rf"(?:from|import)\s+tanitad\.(?:models|data)\.{m}\b")
        for m in ("bev_encoder", "box3d_head", "bev_lift",
                  "perception_targets", "semantic_map_gt")
    }
    weights = {
        "_w_map_reads": _count(src, r'getattr\(\s*m\s*,\s*"_w_map"'),
        "_w_map_assignments": _count(src, r"_w_map\s*="),
        "_w_box3d_reads": _count(src, r'getattr\(\s*m\s*,\s*"_w_box3d"'),
        "_w_box3d_assignments": _count(src, r"_w_box3d\s*="),
    }
    flags = {
        f: _count(src, rf'add_argument\("--{f}"')
        for f in ("w-map", "w-box3d", "trunk-pretrained", "no-trunk-pretrained")
    }

    # the heads DO exist and ARE tested -- the island half of the finding
    island = {}
    for rel, sym in (("tanitad/models/bev_encoder.py", "def map_soft_ce"),
                     ("tanitad/models/box3d_head.py", "def box3d_set_loss"),
                     ("tanitad/data/perception_targets.py", "def collate_map_targets")):
        t = (S / rel).read_text(encoding="utf-8", errors="replace")
        island[rel] = {"bytes": len(t), "defines": sym in t}
    assert all(v["defines"] for v in island.values()), "the heads are missing too?"

    # trunk_pretrained is a real field with no flag
    refc = (S / "tanitad" / "refs" / "refc.py").read_text(encoding="utf-8", errors="replace")
    pretrained = {
        "field_declared_in_refc": _count(refc, r"^\s*trunk_pretrained:\s*bool"),
        "consumed_in_refc": _count(refc, r'getattr\(cfg,\s*"trunk_pretrained"'),
        "stamped_in_trainer_config": _count(src, r'"trunk_pretrained"'),
        "cli_flag": flags["trunk-pretrained"] + flags["no-trunk-pretrained"],
    }

    rep = {
        "evidence_class": "MEASURED",
        "question": ("do the refcv6 perception heads reach the TRAINER's loss, "
                     "or only exist as tested modules?"),
        "compute_losses_v3_loss_keys": loss_keys,
        "trainer_calls": calls,
        "trainer_imports_of_perception_modules": imports,
        "perception_loss_weights": weights,
        "cli_flags": flags,
        "trunk_pretrained": pretrained,
        "the_heads_themselves": island,
        "controls_all_nonzero": controls,
        "verdict": (
            "THE PERCEPTION BRANCH IS A TESTED ISLAND WITH NO BRIDGE TO THE TRAINER"
            if (sum(calls.values()) == 0 and sum(imports.values()) == 0
                and weights["_w_map_assignments"] == 0
                and weights["_w_box3d_assignments"] == 0)
            else "PARTIALLY OR FULLY WIRED -- re-read, this probe's premise has changed"),
    }
    Path(out_path).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8",
                              newline="\n")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
