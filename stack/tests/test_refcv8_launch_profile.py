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


#: PI rulings 2026-10-11: Q-A no route checkpoint, Q-B the quantized label (refcv7's two sidecars) is the speed input --
#: all LITERALS. hold_r8label (the trainer's `label` value) ALSO moved the smoke JSON itself onto Q-A / Q-B, so for those
#: keys `_ruled_argv` is the identity on it; the EMA / B1 / geometry keys are still applied in memory.
_RULED = {"--r8-rc-variant": None, "--r8-rc-noise-along-m": None, "--r8-rc-noise-lat-m": None, "--r8-rc-dropout": None,
          "--r8-no-rc": [], "--r8-speed-input": ["label"],
          "--speed-max-sidecar-v6": ["/home/nvidia/data/refcv6_speed_max_v8_train.jsonl"],
          "--speed-max-sidecar-v6-eval": ["/home/nvidia/data/refcv6_speed_max_v8_eval.jsonl"],
          "--ema-decay": ["0.9998"], "--b1-guard-ref-ap2m": ["0.21211"], "--b1-guard-step": ["1000"],
          "--steps": ["48852"], "--batch": ["16"], "--eval-batches": ["8"], "--eval-every": ["500"],
          # SPEC_REFCV8 A2 (PI 2026-10-12, one full pass over the 4,572-clip train view): the data set, as LITERALS
          "--v2-cache": ["/home/nvidia/data/refcv8-b1-416x1024-train-4572"],
          "--agent-rig-extrinsics": ["/home/nvidia/data/refcv8_train4572_eval139_extrinsics.json"],
          "--pose-sync-sidecar": ["/home/nvidia/data/refcv8_pose_sync_sidecar.jsonl"],
          "--clip-clock-sidecar": ["/home/nvidia/data/refcv8_clip_clock_sidecar_ts33.jsonl"],
          "--r8-v9-labels": ["/home/nvidia/refcv8_v9labels/release_train4572/v9_labels_train.npz"], "--r8-v9-md5": ["83e30bd3791f0767707eade5f4671759"],
          "--vis1-sidecar": ["/home/nvidia/data/refcv8/vis1_sidecar_refcv8b1_train4572_eval139.npz"],
          # SPEC_REFCV8 A3: the nav tau re-derived on the 4,572 view
          "--nav-compliance-tau-file": ["/home/nvidia/data/refcv8/nav_compliance_tau_train4572.json"], "--nav-compliance-tau-rad": ["0.1806411302089691"],
          }


def _ruled_argv():
    a = _argv()
    for f, v in _RULED.items():
        a = LG.set_flag(a, f, v)
    return a


def test_the_profile_inherits_every_refcv7_rule():
    r7, r8 = LG.PROFILES["refcv7"], LG.PROFILES["refcv8"]
    assert set(r7["required_flags"]) < set(r8["required_flags"])
    # SPEC_REFCV8 A2 overrides exactly ONE refcv7 value: the VIS-1 sidecar, rebuilt over the 4,572-clip train view
    # (the trainer refuses any absent clip). Written as a literal set -- every other refcv7 value is inherited verbatim.
    overridden = {k for k, v in r7["required_values"].items() if r8["required_values"][k] != v}
    assert overridden == {"--vis1-sidecar"}, overridden
    assert r8["required_values"]["--vis1-sidecar"] == "/home/nvidia/data/refcv8/vis1_sidecar_refcv8b1_train4572_eval139.npz"
    assert set(r7["required_positive"]) < set(r8["required_positive"])
    assert set(r7["forbidden_levers"]) < set(r8["forbidden_levers"])
    assert r8["eval_loader"] == r7["eval_loader"] and r8["box_required"] == r7["box_required"]


def test_the_ruled_smoke_argv_passes_every_argv_rule_with_NO_pending_PI_decision():
    prof = LG.PROFILES["refcv8"]
    refusals, pending = LG.profile_argv_rules(prof, _ruled_argv())
    # refcv7's FIX-4 rule still binds: the banked tau RECORD is a host-side input of the real gate run (its labels
    # sha256 is checked against the file the host reads), so without it this is the one refusal, and it must be there
    tau = [r for r in refusals if r.startswith("--graft-nav-compliance is passed without a RECORDED tau")]
    assert len(tau) == 1
    refusals = [r for r in refusals if r not in tau]
    assert refusals == [], refusals
    # both PI items are CLOSED by the PI's rulings of 2026-10-11 (Q-A no RC, Q-B the label is the speed input)
    assert pending == [], pending
    # ... and the SUPERSEDED argv (RC variant A50 + N2, no sidecar) is refused, not pending
    old = _argv()
    for f, v in (("--r8-no-rc", None), ("--r8-speed-input", ["n2"]), ("--speed-max-sidecar-v6", None),
                 ("--speed-max-sidecar-v6-eval", None), ("--r8-rc-variant", ["A50"]),
                 ("--r8-rc-noise-along-m", ["2.0"]), ("--r8-rc-noise-lat-m", ["0.75"]), ("--r8-rc-dropout", ["0.3"])):
        old = LG.set_flag(old, f, v)
    old_refusals, _p = LG.profile_argv_rules(prof, old)
    assert any("--r8-rc-variant" in r for r in old_refusals) and any("--r8-speed-input" in r for r in old_refusals)
    ids = [it["id"] for it in prof["open_items"]]
    # R8-WARMUP was CLOSED by the Master Mind on 2026-10-11 (SPEC_REFCV8 sec. 10.3): three items still wait on evidence
    assert ids == ["R8-RECIPE", "R8-BUDGET", "R8-INHERITED-RECORDS"]
    assert "R8-WARMUP" not in ids
    assert len(LG.open_item_reasons(prof)) == 3                              # the token cannot read PASS yet


def test_S_emit_is_the_warmup_length_and_both_are_1000_literals():
    """R8-WARMUP closed (SPEC_REFCV8 sec. 10.3): S_emit == the warmup length == 1,000. Literals, not derived."""
    prof = LG.PROFILES["refcv8"]
    assert prof["required_values"]["--r8-alloc-emit-start"] == "1000"
    a = _argv()
    assert a[a.index("--warmup") + 1] == "1000"
    assert a[a.index("--r8-alloc-emit-start") + 1] == "1000"
    why = prof["required_values_why"]["--r8-alloc-emit-start"]
    assert "--warmup 1000" in why and "2000" not in why


@pytest.mark.parametrize("mutate,needle", [
    (lambda a: LG.set_flag(a, "--w-r8-cons", None), "--w-r8-cons"),
    (lambda a: LG.set_flag(a, "--init-from", None), "--init-from"),
    (lambda a: LG.set_flag(a, "--join-defect-masks", None), "--join-defect-masks"),
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
    refusals, _p = LG.profile_argv_rules(LG.PROFILES["refcv8"], mutate(_ruled_argv()))
    assert any(needle in r for r in refusals), refusals


V8_SIDECAR = "/home/nvidia/data/refcv6_speed_max_v8_train.jsonl"


@pytest.mark.parametrize("flag", ["--speed-max-sidecar-v6", "--speed-max-sidecar-v6-eval"])
def test_Q_B_the_label_sidecar_is_admitted_at_the_gate_and_only_refcv7s_files_pass(flag):
    """PI Q-B 2026-10-11 LIFTED the by-name refusal of the v8 sidecar (it is the speed channel's source, as the 4-way
    quantized label). The gate admits it as a lever, requires it, and pins the path to refcv7's launch record. (The
    trainer-side pin that still refuses it with `--r8-speed-input n2` is hold_r8label's to flip, with its own test.)"""
    prof = LG.PROFILES["refcv8"]
    green, d = LG.forbidden_lever_reasons(prof, _ruled_argv())
    assert green == [] and d["argv_hits"] == [], green
    missing, _p = LG.profile_argv_rules(prof, LG.set_flag(_ruled_argv(), flag, None))
    assert any(flag in r for r in missing), missing                          # required
    other, _p = LG.profile_argv_rules(prof, LG.set_flag(_ruled_argv(), flag, ["/x/other.jsonl"]))
    assert any(flag in r for r in other), other                               # not refcv7's file


def test_the_canonical_argv_feeds_the_label_with_refcv7s_sidecars_and_no_route_checkpoint():
    """PI 2026-10-11: the speed source is the road-type label (refcv7's two sidecars, verbatim) and there is no RC."""
    a = _argv()
    r7 = LG.flag_pairs(json.loads(R7_FILE.read_text(encoding="utf-8"))["argv"])
    assert LG.flag_values(a, "--r8-speed-input") == ["label"] and LG.flag_values(a, "--r8-speed-unknown-p") == ["0.45"]
    assert LG.flag_values(a, "--speed-max-sidecar-v6") == ["/home/nvidia/data/refcv6_speed_max_v8_train.jsonl"]
    assert LG.flag_values(a, "--speed-max-sidecar-v6-eval") == ["/home/nvidia/data/refcv6_speed_max_v8_eval.jsonl"]
    assert dict((f, list(v)) for f, v in r7)["--speed-max-sidecar-v6"] == LG.flag_values(a, "--speed-max-sidecar-v6")
    assert "--r8-no-rc" in a and "--r8-nav-from-v9" in a                          # nav command + turn distance stay
    assert not any(t.startswith("--r8-rc-") for t in a)
    assert "--max-speed-input-v6" in a and "--speed-ceiling-filter" in a          # the channel and the ceiling stay


def test_the_argv_is_a_refcv8_argv_the_trainer_accepts_with_exactly_the_stated_delta():
    T = trainer()
    a = _argv()
    args = T.build_parser().parse_args(a)
    cfg = types.SimpleNamespace(refcv8=r8c.R8Config())
    T._pin_refcv8(cfg, args)
    r = cfg.refcv8
    assert r.enable and r.n_alloc == 32 and r.w_cons == 0.05 and r.w_listwise == 1.0 and r.speed_input == "label"
    rec = json.loads(ARGV_FILE.read_text(encoding="utf-8"))
    r7 = json.loads(R7_FILE.read_text(encoding="utf-8"))["argv"]
    assert rec["derived_from"]["n_tokens"] == len(r7)
    changed = {c["flag"] for c in rec["changes_vs_refcv7"]}
    for f, v in LG.flag_pairs(r7):                     # every refcv7 token not named as a change is kept verbatim
        if f not in changed:
            assert LG.flag_values(a, f) == list(v), f
