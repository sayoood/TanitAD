"""Stage-5 cell entry for the RALIGN arm (PREREG_T2_RALIGN.md section 5): the banked stage-5 cell with the cross-fitted token map in front of the policy. Shipped to box 2 (pkg/code/).

    python -I ralign_cell.py [--executor base|hold] [--hold-eps 0.1] <the t2v8f_run.py rollout arguments>      (the driver sets T2V8F_S5_MANIFEST for every child)

It does NOT edit or copy the banked cell: it imports `t2v8f_stage5_cell` (already on box 2, byte-identical to code/), lets its `install5` register `<NAME>_s0/_s1` and put the composite
(sg72 + kbox) on them, and then `ralign_policy.install_align` wraps `build_arm` once more so that an `AlignPolicy` carrying the OTHER fold's map sits around the composite. Only the
`arm5.encoder.kind == "frozen_dinov3_ralign"` manifest is accepted here; the banked cell and driver refuse that kind by name (ENCODER_UNSUPPORTED), so the arm cannot run without its map.
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ENV_MANIFEST = "T2V8F_S5_MANIFEST"
KIND = "frozen_dinov3_ralign"
SEEDS = (0, 1)


def arm5_from_manifest_ra(path):
    """-> (name, {arm: ckpt file}, ralign block). Refuses a manifest that is not an RALIGN one."""
    a = json.load(open(path, encoding="utf-8")).get("arm5") or {}
    if not a.get("name") or not a.get("ckpt_file"):
        raise SystemExit(f"[ralign_cell] {path} carries no arm5.name / arm5.ckpt_file: this is not a MANIFEST5.json")
    kind = (a.get("encoder") or {}).get("kind")
    if kind != KIND or not isinstance(a.get("ralign"), dict):
        raise SystemExit(f"[ralign_cell] ENCODER_UNSUPPORTED {kind!r}: the RALIGN cell builds arm5.encoder.kind == {KIND!r} with an arm5.ralign block only")
    return a["name"], dict(a["ckpt_file"]), a["ralign"]


def install_ra(R, manifest_path):
    """Register the arm (banked `install5`), then the map (`install_align`). Idempotent; `R` is the module `t2v8f_run`."""
    import ralign_policy as RP
    import t2v8f_stage5_cell as C5
    name, ckpt_file, block = arm5_from_manifest_ra(manifest_path)
    C5.install5(R, name, ckpt_file)
    RP.install_align(R, block, os.path.dirname(os.path.abspath(manifest_path)).replace("\\", "/"), tuple(f"{name}_s{s}" for s in SEEDS))
    return R


def run_cell(argv):
    path = os.environ.get(ENV_MANIFEST)
    if not path or not os.path.isfile(path):
        raise SystemExit(f"[ralign_cell] {ENV_MANIFEST} is not set to an existing MANIFEST5.json ({path!r}); the driver sets it for every cell")
    import t2v8f_run as R
    import t2v8f_stage2_cell as C
    install_ra(R, path)
    return C.run_cell(argv)


if __name__ == "__main__":
    run_cell(sys.argv[1:])
