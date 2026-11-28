"""⛔⛔ FIX-3's other side: a PRE-FIX checkpoint must be rebuilt AS TRAINED.

D-REFCV6-EQUALIZE-DROPPED (2026-09-26): with `--image-hw`, `--equalize-bottom-rows N` never
reached the trunk before FIX-3, so refcv6-r101-s0 (argv 43, `--image-hw 416 1024`) TRAINED with
an un-equalised trunk. FIX-3 makes the pin honour argv -- which, applied to that run's recorded
argv, would evaluate a model its weights never were (bottom 43 rows zeroed at eval, never in
training), with a strict 0/0 load and no refusal anywhere. `refcv3_arm.rebuild_config` therefore
rebuilds a record that predates the `seams.trunk_equalize_bottom_rows` stamp with the rows its
trunk REALLY zeroed (`refc_v3_train.trunk_equalize_rows_as_trained`), and says so in `source`.

Literals: the live run's argv tail (`--image-hw 416 1024 ... --equalize-bottom-rows 43`) must
rebuild 0; a post-fix record stamped 43 must rebuild 43. The deliberate-regression arm removes
the resolver and must read 43 -- the silent train/eval mismatch.
"""
from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("torch")


def _arm():
    import importlib.util
    import sys
    root = pathlib.Path(__file__).resolve().parents[2]
    src = root / "taniteval" / "tools" / "refcv3_arm.py"
    if not src.is_file():
        pytest.skip(f"{src} not present in this checkout")
    spec = importlib.util.spec_from_file_location("refcv3_arm_eq_under_test", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refcv3_arm_eq_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


#: the smallest argv carrying refcv6-r101-s0's two relevant levers (its own tail, verbatim)
LIVE = ["--arm", "hier", "--smoke", "--trunk", "timm", "--trunk-name", "resnet18.a1_in1k",
        "--no-trunk-pretrained", "--image-hw", "416", "1024", "--equalize-bottom-rows", "43",
        "--out", "x"]


def test_a_PRE_FIX_record_rebuilds_the_trunk_UN_equalised_and_says_why():
    arm = _arm()
    cfg, _args, src = arm.rebuild_config({"argv": LIVE})
    assert cfg.core.encoder.trunk_equalize_bottom_rows == 0
    assert "TRUNK equalize_bottom_rows -> 0" in src and "DROPPED" in src


def test_a_POST_FIX_record_rebuilds_what_it_stamped():
    arm = _arm()
    cfg, _args, src = arm.rebuild_config({"argv": LIVE,
                                          "seams": {"trunk_equalize_bottom_rows": 43}})
    assert cfg.core.encoder.trunk_equalize_bottom_rows == 43
    assert "TRUNK equalize_bottom_rows" not in src


def test_DELIBERATE_REGRESSION_without_the_resolver_the_eval_trunk_would_equalise(monkeypatch):
    arm = _arm()
    tr = arm.trainer()
    monkeypatch.delattr(tr, "trunk_equalize_rows_as_trained")
    cfg, _args, _src = arm.rebuild_config({"argv": LIVE})
    assert cfg.core.encoder.trunk_equalize_bottom_rows == 43, \
        "the regression arm did not reproduce the silent train/eval mismatch"
