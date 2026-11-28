"""§12 refusal 1 in both halves — and the mutation that matters is the REAL defect.

⭐ The `_MUT_a_dropped_field_makes_two_arms_ONE_arm` test reproduces the exact
`--image-hw` mechanism MEASURED on 2026-09-17: a config builder that rebuilds a
dataclass from a **hand-written field list** and silently drops what it does not
mention. A parsed-namespace check passes it; the built-config check does not.
"""
from __future__ import annotations

import argparse
import dataclasses as dc

import pytest

from tanitad.train.one_variable_v2 import (
    OneVariableViolation, config_diff, namespace_diff,
    refuse_more_than_one_variable,
)


@dc.dataclass
class Enc:
    image_size: int = 256
    image_width: int = 640
    trunk_name: str = "resnet101.a1_in1k"
    trunk_fuse: str = "concat1x1"


@dc.dataclass
class Cfg:
    encoder: Enc = dc.field(default_factory=Enc)
    seed: int = 0


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk-name", default="resnet101.a1_in1k")
    ap.add_argument("--trunk-fuse", default="concat1x1")
    ap.add_argument("--image-hw", nargs=2, type=int, default=None)
    ap.add_argument("--out", default="/tmp/x")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def _build_good(ns):
    """The CORRECT builder: `dataclasses.replace`, so no field can be dropped."""
    c = Cfg()
    c.encoder = dc.replace(c.encoder, trunk_name=ns.trunk_name,
                           trunk_fuse=ns.trunk_fuse)
    if ns.image_hw:
        c.encoder = dc.replace(c.encoder, image_size=ns.image_hw[0],
                               image_width=ns.image_hw[1])
    c.seed = ns.seed
    return c


def _build_dropping(ns):
    """⛔ THE REAL DEFECT: rebuild from a HAND-WRITTEN field list that mentions
    neither `trunk_name` nor `trunk_fuse`, so `--image-hw` resets both."""
    c = _build_good(ns)
    if ns.image_hw:
        c.encoder = Enc(image_size=ns.image_hw[0], image_width=ns.image_hw[1])
    return c


A = ["--trunk-name", "resnet101.a1_in1k", "--out", "/tmp/a"]
B = ["--trunk-name", "resnet34", "--out", "/tmp/b"]
HW = ["--image-hw", "256", "1024"]


def _call(argv_a, argv_b, build, **kw):
    return refuse_more_than_one_variable(
        argv_a, argv_b, lever="trunk_name",
        expected_from="resnet101.a1_in1k", expected_to="resnet34",
        build_parser=_parser, build_config=build,
        bookkeeping=("out", "seed"), **kw)


def test_a_clean_one_variable_pair_PASSES():
    r = _call(A, B, _build_good)
    assert r["verdict"] == "ONE-VARIABLE"
    assert r["built_paths_differing"] == ["encoder.trunk_name"]


def test_MUT_a_dropped_field_makes_two_arms_ONE_ARM_and_is_REFUSED():
    """⭐⭐ THE ONE THAT EARNS THE MODULE — the real 2026-09-17 defect.

    Both arms pass `--image-hw`, as every 1024 run does. The builder drops
    `trunk_name`. The PARSED namespaces still differ exactly as declared, so the
    old half-check passes; the BUILT configs are identical, so the arms are one
    arm."""
    # the old half alone is happy: the namespaces really do differ in the lever
    nd = namespace_diff(_parser().parse_args(A + HW), _parser().parse_args(B + HW))
    assert nd["trunk_name"] == ("resnet101.a1_in1k", "resnet34")
    # and the built configs are IDENTICAL, which is the defect
    assert config_diff(_build_dropping(_parser().parse_args(A + HW)),
                       _build_dropping(_parser().parse_args(B + HW))) == {}
    with pytest.raises(OneVariableViolation, match="BUILT CONFIGS ARE IDENTICAL"):
        _call(A + HW, B + HW, _build_dropping)


def test_the_SAME_pair_passes_once_the_builder_is_fixed():
    """⭐ The discriminating control: the refusal must be about the BUILDER, not
    about `--image-hw` being present."""
    r = _call(A + HW, B + HW, _build_good)
    assert r["built_paths_differing"] == ["encoder.trunk_name"]


def test_MUT_a_lever_that_moves_the_WRONG_WAY_is_REFUSED():
    """A pair differing in exactly one key is not evidence the lever is ON."""
    with pytest.raises(OneVariableViolation, match="moved"):
        _call(B, A, _build_good)          # reversed: resnet34 -> resnet101


def test_MUT_a_lever_that_does_not_move_at_all_is_REFUSED():
    with pytest.raises(OneVariableViolation, match="does not differ"):
        _call(A, list(A), _build_good)


def test_MUT_a_SECOND_argv_lever_is_REFUSED_unless_declared():
    with pytest.raises(OneVariableViolation, match="beyond the lever"):
        _call(A, B + ["--trunk-fuse", "last"], _build_good)


def test_a_declared_CONSTITUTIVE_key_is_allowed_and_a_bookkeeping_one_is_ignored():
    """`out` differs on every pair by construction; a constitutive key is one the
    trainer's own guard forces to move WITH the lever."""
    r = _call(A, B + ["--trunk-fuse", "last"], _build_good,
              constitutive=("trunk_fuse",),
              built_allowed=("encoder.trunk_fuse",))
    assert r["verdict"] == "ONE-VARIABLE"


def test_MUT_an_UNDECLARED_built_difference_is_REFUSED():
    """⛔ The built half must bite in BOTH directions: too few built differences
    (the arms are one arm) and too many (a second lever hiding downstream)."""
    def build_extra(ns):
        c = _build_good(ns)
        if ns.trunk_name == "resnet34":
            c.encoder = dc.replace(c.encoder, image_width=99)   # undeclared
        return c
    with pytest.raises(OneVariableViolation, match="does not declare"):
        _call(A, B, build_extra)
