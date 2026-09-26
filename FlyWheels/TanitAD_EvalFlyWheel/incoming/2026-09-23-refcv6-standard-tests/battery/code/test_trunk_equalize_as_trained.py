"""D-REFCV6-EQUALIZE-DROPPED / SPEC_REFCV7 FIX-3: a pre-fix refcv6 checkpoint must be rebuilt with its
trunk AS TRAINED (0 equalised rows), never as declared (43), on ANY tree -- including a post-FIX one.

Every assertion reads the TREE UNDER TEST: the `stack/scripts/refc_v3_train.py` the battery loader imports
(REFCV6_REPO / PYTHONPATH), never another package's staging copy.
  * post-FIX tree (the helper exists): the helper returns the literal (0, "DROPPED...") on the run's real
    config.json and (43, "stamped (post-fix record)") on a post-fix record; the pin DECLARES 43 on the trunk
    and the loader's override sets it to 0 with the literal record;
  * pre-FIX tree (no helper): the loader records "pre-FIX-3 tree" and changes nothing -- asserted, not skipped.
Integration (RUN_EQ_INTEGRATION=1; needs both trees, the kit and ckpt_30000.pt): three deterministic rolls,
each its own process, compared by a fourth (`eq_as_trained_check.py`): the post-FIX tree WITH the override
must equal the pre-fix tree BIT FOR BIT, and WITHOUT it (trunk as declared) must differ.

run:  PYTHONPATH="<tree>/stack;<tree>/taniteval" REFCV6_REPO=<tree> python -m pytest -q test_trunk_equalize_as_trained.py
"""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcv6_loader as L  # noqa: E402

KIT_CONFIG = Path("D:/refcv6_eval_kit/ckpt/config.json")
PRE_TREE, POST_TREE = "C:/Users/Admin/ev6", "C:/Users/Admin/ev6_postfix"
CKPT = "D:/refcv6_eval_kit/ckpt/ckpt_30000.pt"


def test_fake_pre_fix_tree_is_left_alone():
    tr = types.SimpleNamespace()
    cfg = types.SimpleNamespace(core=types.SimpleNamespace(encoder=types.SimpleNamespace()))
    out, rec = L.trunk_rows_as_trained(tr, cfg, {})
    assert out is cfg and not hasattr(cfg.core.encoder, "trunk_equalize_bottom_rows")
    assert rec["status"].startswith("pre-FIX-3 tree")


def test_fake_post_fix_tree_gets_the_trained_rows():
    tr = types.SimpleNamespace(trunk_equalize_rows_as_trained=lambda config: (0, "DROPPED: test"))
    cfg = types.SimpleNamespace(core=types.SimpleNamespace(encoder=types.SimpleNamespace(
        trunk_equalize_bottom_rows=43)))
    out, rec = L.trunk_rows_as_trained(tr, cfg, {})
    assert out.core.encoder.trunk_equalize_bottom_rows == 0
    assert rec == {"status": "set as trained", "declared_by_pin": 43, "as_trained": 0, "why": "DROPPED: test"}


def _pinned_cfg(tr, config):
    """The loader's own pre-pin sequence + the tree's pin (train():6714-6749), no weights read."""
    from tanitad.refs import refc_v3 as v3
    args = L.parse_args(config)[0]
    tr._check_nav_from_v7_args(args)
    tr._check_max_speed_args(args)
    tr._check_goal_point_args(args)
    tr.check_effective_weights(args)
    tr._read_anchor_artifact(args)
    return tr._pin_trainer_cfg(v3.refc_v3_smoke_config(args.arm == "hier") if args.smoke else
                               v3.refc_v3_sized_config(args.size, hier=args.arm == "hier"), args)


def test_the_tree_under_test_rebuilds_the_trunk_as_trained():
    if not KIT_CONFIG.exists():
        pytest.skip("the run's config.json (kit) is not on this box")
    config = json.load(open(KIT_CONFIG, encoding="utf-8"))
    tr = L.trainer()                                          # the TREE UNDER TEST's refc_v3_train.py
    cfg = _pinned_cfg(tr, config)
    out, rec = L.trunk_rows_as_trained(tr, cfg, config)
    if hasattr(tr, "trunk_equalize_rows_as_trained"):         # post-FIX tree
        rows, why = tr.trunk_equalize_rows_as_trained(config)
        assert rows == 0 and why.startswith("DROPPED")
        assert tr.trunk_equalize_rows_as_trained(
            {"seams": {"trunk_equalize_bottom_rows": 43}, "argv": []}) == (43, "stamped (post-fix record)")
        assert (rec["status"], rec["declared_by_pin"], rec["as_trained"]) == ("set as trained", 43, 0)
        assert out.core.encoder.trunk_equalize_bottom_rows == 0
    else:                                                     # pre-FIX tree: asserted, not skipped
        assert rec["status"].startswith("pre-FIX-3 tree")
        assert not hasattr(out.core.encoder, "trunk_equalize_bottom_rows")


@pytest.mark.skipif(os.environ.get("RUN_EQ_INTEGRATION") != "1", reason="integration: set RUN_EQ_INTEGRATION=1")
def test_post_fix_rebuild_is_bit_identical_only_with_the_override(tmp_path):
    for p in (PRE_TREE, POST_TREE, CKPT):
        if not Path(p).exists():
            pytest.skip(f"{p} absent")
    for arm in ("R", "A", "B"):                               # three processes, each frees its model
        subprocess.run([sys.executable, str(HERE / "eq_as_trained_check.py"), "roll", "--arm", arm,
                        str(tmp_path)], timeout=8 * 3600)
    subprocess.run([sys.executable, str(HERE / "eq_as_trained_check.py"), "compare", str(tmp_path)], timeout=600)
    rec = json.load(open(tmp_path / "eq_as_trained_record.json", encoding="utf-8"))
    assert rec["checks"]["A_bit_identical_to_R_on_every_key"], rec["A_equals_R"]
    assert rec["checks"]["B_differs_from_R_on_traj"], rec["B_vs_R"]
    assert rec["checks"]["A_record_set_as_trained_43_to_0"], rec["trunk_records"]
    assert rec["verdict"] == "PASS"
