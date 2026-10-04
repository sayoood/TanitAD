"""refcv8 WP-B -- the ``refcv8`` launch-gate profile and the canonical SMOKE argv (``stack/ops/runs.d/
refcv8-wpb-smoke.argv.json``).

Pinned:
1. the profile INHERITS every refcv7 rule (required values / flags / positives / forbidden levers are supersets);
2. the canonical smoke argv passes every argv-level rule, its only PI-DECISION is the route checkpoint (the Master
   Mind's provisional ruling), and its OPEN ITEMS are named -- so the token cannot read PASS while the recipe is open;
3. each refcv8 rule REFUSES its violation (one mutation per rule);
4. the argv is a refcv8 argv the TRAINER accepts (parse + ``_pin_refcv8``), with exactly the stated delta from refcv7.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from _refcv8_rig import trainer
import launch_gate as LG                                                      # noqa: E402 (scripts/ on sys.path)
from tanitad.refs import refcv8_conditioning as r8c

STACK = Path(__file__).resolve().parents[1]
ARGV_FILE = STACK / "ops" / "runs.d" / "refcv8-wpb-smoke.argv.json"
R7_FILE = STACK / "ops" / "runs.d" / "refcv7-r101-s0.argv.json"


def _argv():
    return list(json.loads(ARGV_FILE.read_text(encoding="utf-8"))["argv"])


def test_the_profile_inherits_every_refcv7_rule():
    r7, r8 = LG.PROFILES["refcv7"], LG.PROFILES["refcv8"]
    assert set(r7["required_flags"]) < set(r8["required_flags"])
    assert all(r8["required_values"][k] == v for k, v in r7["required_values"].items())
    assert set(r7["required_positive"]) < set(r8["required_positive"])
    assert set(r7["forbidden_levers"]) < set(r8["forbidden_levers"])
    assert r8["eval_loader"] == r7["eval_loader"] and r8["box_required"] == r7["box_required"]


def test_the_smoke_argv_passes_every_argv_rule_with_the_RC_as_its_only_PI_decision():
    prof = LG.PROFILES["refcv8"]
    refusals, pending = LG.profile_argv_rules(prof, _argv())
    # refcv7's FIX-4 rule still binds: the banked tau RECORD is a host-side input of the real gate run (its labels
    # sha256 is checked against the file the host reads), so without it this is the one refusal, and it must be there
    tau = [r for r in refusals if r.startswith("--graft-nav-compliance is passed without a RECORDED tau")]
    assert len(tau) == 1
    refusals = [r for r in refusals if r not in tau]
    assert refusals == [], refusals
    # two PI decisions: the RC (the MM's provisional ruling) and N2 vs N3 (PI decision 2; N2 = the default)
    assert len(pending) == 2, pending
    assert any(p.startswith("--r8-rc-variant A50") for p in pending)
    assert any(p.startswith("--r8-speed-input n2") for p in pending)
    ids = [it["id"] for it in prof["open_items"]]
    assert ids == ["R8-RECIPE", "R8-BUDGET", "R8-WARMUP", "R8-INHERITED-RECORDS"]
    assert len(LG.open_item_reasons(prof)) == 4                              # the token cannot read PASS yet


@pytest.mark.parametrize("mutate,needle", [
    (lambda a: LG.set_flag(a, "--w-r8-cons", None), "--w-r8-cons"),
    (lambda a: LG.set_flag(a, "--init-from", None), "--init-from"),
    (lambda a: LG.set_flag(a, "--join-defect-masks", None), "--join-defect-masks"),
    (lambda a: LG.set_flag(a, "--r8-rc-noise-along-m", ["1.0"]), "--r8-rc-noise-along-m"),
    (lambda a: LG.set_flag(a, "--r8-rc-dropout", ["0.2"]), "--r8-rc-dropout"),
    (lambda a: LG.set_flag(a, "--r8-v9-md5", ["0" * 32]), "--r8-v9-md5"),
    (lambda a: LG.set_flag(a, "--r8-derange-feed", []), "--r8-derange-feed"),
    (lambda a: LG.set_flag(a, "--r8-roll-targets", ["tac"]), "--r8-roll-targets"),
    (lambda a: LG.set_flag(a, "--residual-prior", ["off"]), "--residual-prior"),          # a refcv7 rule still binds
    # X3 (SPEC_REFCV8 8.3, MM ruling Q5)
    (lambda a: LG.set_flag(a, "--r8-speed-input", None), "--r8-speed-input"),
    (lambda a: LG.set_flag(a, "--r8-speed-unknown-p", ["0.2"]), "--r8-speed-unknown-p"),
    (lambda a: LG.set_flag(a, "--r8-roll-speed-input", []), "--r8-roll-speed-input"),
])
def test_each_refcv8_rule_refuses_its_violation(mutate, needle):
    refusals, _p = LG.profile_argv_rules(LG.PROFILES["refcv8"], mutate(_argv()))
    assert any(needle in r for r in refusals), refusals


V8_SIDECAR = "/home/nvidia/data/refcv6_speed_max_v8_train.jsonl"


@pytest.mark.parametrize("flag", ["--speed-max-sidecar-v6", "--speed-max-sidecar-v6-eval"])
def test_RED_a_mutation_that_re_attaches_the_v8_sidecar_goes_RED_at_the_gate_and_at_the_trainer(flag):
    """MM item 1: "a mutation that re-attaches the sidecar must go RED at the gate". The canonical argv is GREEN on the
    rule (no forbidden lever); the same argv with refcv7's sidecar re-attached is refused by NAME at the gate (the
    lever rule, whatever the value) AND by the trainer's own pin -- two independent places."""
    prof = LG.PROFILES["refcv8"]
    green, _d = LG.forbidden_lever_reasons(prof, _argv())
    assert green == [], green
    bad = LG.set_flag(_argv(), flag, [V8_SIDECAR])
    red, det = LG.forbidden_lever_reasons(prof, bad)
    assert det["argv_hits"] == [flag] and any("FUTURE-MAX sidecar" in r for r in red), red
    refusals, _p = LG.profile_argv_rules(prof, bad)
    assert any(flag in r and "ORACLE" in r for r in refusals), refusals
    T = trainer()
    with pytest.raises(SystemExit, match="FUTURE-MAX sidecar"):
        T._pin_refcv8(types.SimpleNamespace(refcv8=r8c.R8Config()), T.build_parser().parse_args(bad))


def test_the_canonical_argv_feeds_N2_and_no_sidecar():
    a = _argv()
    assert LG.flag_values(a, "--r8-speed-input") == ["n2"] and LG.flag_values(a, "--r8-speed-unknown-p") == ["0.45"]
    assert not any(t.startswith("--speed-max-sidecar") for t in a)
    assert "--max-speed-input-v6" in a and "--speed-ceiling-filter" in a          # the channel and the ceiling stay


def test_the_argv_is_a_refcv8_argv_the_trainer_accepts_with_exactly_the_stated_delta():
    T = trainer()
    a = _argv()
    args = T.build_parser().parse_args(a)
    cfg = types.SimpleNamespace(refcv8=r8c.R8Config())
    T._pin_refcv8(cfg, args)
    r = cfg.refcv8
    assert r.enable and r.n_alloc == 32 and r.w_cons == 0.05 and r.w_listwise == 1.0 and r.speed_input == "n2"
    rec = json.loads(ARGV_FILE.read_text(encoding="utf-8"))
    r7 = json.loads(R7_FILE.read_text(encoding="utf-8"))["argv"]
    assert rec["derived_from"]["n_tokens"] == len(r7)
    changed = {c["flag"] for c in rec["changes_vs_refcv7"]}
    for f, v in LG.flag_pairs(r7):                     # every refcv7 token not named as a change is kept verbatim
        if f not in changed:
            assert LG.flag_values(a, f) == list(v), f
