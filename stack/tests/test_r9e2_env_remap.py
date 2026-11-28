"""test_r9e2_env_remap.py -- the E2 / D1 Thor read env must hand the eval loader a REAL remap (FIXED 2026-11-12).

MEASURED 2026-11-12 10:27Z, the first real E2 roll on Thor: with R9E2_REMAP = an empty {} json, refcv7_loader REFUSED to evaluate K0 because the run's argv
names the refcv8 train4572/eval139 side files and the kit's defaults are refcv6's -- on exactly the four flags pinned below.  The env now points at the landed
Thor remap stack/ops/refcv8_eval_remap_a2_thor.json (the E1 dump chain's REFCV6_REMAP_OVERRIDES).  Static: reads the two landed files; stdlib + pytest only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "ops" / "r9e2_read" / "r9e2_read_env.thor.sh"
REMAP = ROOT / "ops" / "refcv8_eval_remap_a2_thor.json"
#: the loader's refusal list, MEASURED 2026-11-12 (runs/E2_READ_K2M1N20/E2_K0_pf/roll_s0.log): flag -> the file the K0 argv names
REFUSED = {"--agent-rig-extrinsics": "refcv8_train4572_eval139_extrinsics.json", "--vis1-sidecar": "vis1_sidecar_refcv8b1_train4572_eval139.npz",
           "--nav-compliance-tau-file": "nav_compliance_tau_train4572.json", "--clip-clock-sidecar": "refcv8_clip_clock_sidecar_ts33.jsonl"}


def _remap_line():
    lines = [ln for ln in ENV.read_text(encoding="utf-8").splitlines() if re.match(r"\s*export\s+R9E2_REMAP=", ln)]
    assert len(lines) == 1, lines
    return lines[0]


def test_the_env_points_r9e2_remap_at_the_landed_thor_remap_inside_the_read_tree():
    ln = _remap_line()
    assert '"$R9E2_TREE/stack/ops/refcv8_eval_remap_a2_thor.json"' in ln
    assert "empty_remap" not in ln.split("#")[0]


def test_r9e2_tree_is_set_before_the_remap_uses_it():
    txt = ENV.read_text(encoding="utf-8")
    assert txt.index("export R9E2_TREE=") < txt.index("export R9E2_REMAP=")


def test_the_landed_remap_maps_every_flag_the_loader_refused_to_the_file_the_argv_names():
    m = json.loads(REMAP.read_text(encoding="utf-8"))
    for flag, name in REFUSED.items():
        assert flag in m, flag
        assert m[flag].startswith("/home/nvidia/") and m[flag].endswith("/" + name), (flag, m[flag])


def test_regression_arm_an_empty_remap_maps_none_of_them():
    assert not [f for f in REFUSED if f in json.loads("{}")]
