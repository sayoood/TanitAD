#!/usr/bin/env python
"""Q4 — provenance closure by CONSTRUCTION.

Two halves, and the second is the one that matters:

  **A. Seams the guard DOES cover.** For each, build a real model, take the
  real `_seam_stamp`, then construct a record/model DISAGREEMENT and assert
  `assert_seams_are_built` REFUSES. A guard that refuses nothing is not a guard.

  **B. Seams the guard is SILENT about.** The same construction, for stamp
  fields that decide which refcv6 arm ran. A field that can be falsified in the
  record with no refusal is FALSE PROVENANCE waiting to happen — the exact
  defect `assert_seams_are_built`'s own docstring was written to close.

⛔ Method notes — and the third one is here because THIS FILE'S FIRST VERSION
   FAILED IT:
  * the stamp is the REAL `_seam_stamp(cfg, args)` output, mutated one key at a
    time — never a hand-built dict, which would test my typing, not the code.
  * every arm carries a SAME-BREATH CONTROL: the unmutated stamp must PASS.
    A probe where even the clean stamp refuses is INCONCLUSIVE, not a finding.
  * ⛔⛔ **EVERY ARM ASSERTS THAT IT ACTUALLY CHANGED THE VALUE.** Run 1 of this
    probe scored `trunk_imagenet_norm=True`, `encoder_lr_mult=0.5` and
    `weight_decay=1e-4` as "SILENT" while the clean stamp ALREADY held those
    values — three INERT arms reported as findings. That is the advisory's
    *"a control that cannot come out the other way is not a control"* committed
    inside the instrument written to hunt it. An arm whose `new` equals the
    clean value now ABORTS the run as INVALID.
  * `assert_seams_are_built` raises `SystemExit`; "refused" means that exception
    was raised, and the message is recorded so a refusal for the WRONG reason is
    visible rather than scored as a pass.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import pathlib
import sys

import torch

TRAINER = pathlib.Path(r"D:/Projects/TanitAD/stack/scripts/refc_v3_train.py")
_MISSING = object()


def _load():
    spec = importlib.util.spec_from_file_location("refc_v3_train", TRAINER)
    m = importlib.util.module_from_spec(spec)
    sys.modules["refc_v3_train"] = m
    spec.loader.exec_module(m)
    return m


def build(m, extra: list[str]):
    from tanitad.refs import refc_v3 as v3
    p = m.build_parser()
    args = p.parse_args(["--smoke", "--arm", "hier", "--out", "unused"] + extra)
    cfg = m._pin_trainer_cfg(v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    model = v3.RefCV3Model(cfg)
    stamp = m._seam_stamp(cfg, args)
    return args, cfg, model, stamp


def check(m, model, stamp) -> tuple[bool, str]:
    """-> (refused?, message)."""
    try:
        m.assert_seams_are_built(model, stamp)
        return False, ""
    except SystemExit as ex:
        return True, str(ex)[:500]
    except Exception as ex:                       # any other raise is a refusal
        return True, f"<{type(ex).__name__}> {str(ex)[:400]}"


def run_arm(m, model, stamp, rows, label, key, new, why, *, must_differ=True):
    """Mutate ONE stamp key and record whether the guard refuses.

    ⛔ Aborts INVALID if the mutation does not change the value — an inert arm
    reads exactly like a silent guard.
    """
    old = stamp.get(key, _MISSING)
    if must_differ and old is not _MISSING and old == new:
        raise SystemExit(
            f"INVALID ARM {label}: stamp[{key!r}] is ALREADY {new!r}; this "
            f"mutation changes nothing and would score as SILENT for free.")
    s = copy.deepcopy(stamp)
    s[key] = new
    refused, msg = check(m, model, s)
    rows.append({"arm": label, "key": key,
                 "clean_value": "<ABSENT>" if old is _MISSING else old,
                 "falsified_to": new, "refused": refused, "message": msg,
                 "why_it_is_a_regression": why})
    print(f"  {'REFUSED ' if refused else '⛔ SILENT'}  {label:<48} "
          f"{key}: {old!r} -> {new!r}")
    if refused:
        first = [ln for ln in msg.split("\n") if ln.strip().startswith("-")]
        if first:
            print(f"             {first[0].strip()[:150]}")
    return refused


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    m = _load()
    rows: list[dict] = []

    # ================= PASS 1: the in-repo refc trunk ==================== #
    args, cfg, model, stamp = build(m, [])
    refused, msg = check(m, model, stamp)
    print("PASS 1 — refc trunk.  CONTROL (clean stamp, clean model):",
          "⛔ REFUSED — INCONCLUSIVE" if refused else "passes (correct)")
    if refused:
        print("  ", msg)
        raise SystemExit("INCONCLUSIVE: the clean arm already refuses.")
    print(f"  stamp keys: {len(stamp)}   "
          f"encoder={type(model.core.encoder).__name__}")

    print("\n  --- A: seams assert_seams_are_built claims to cover ---")
    run_arm(m, model, stamp, rows, "A-sampler-ddim-no-denoiser",
            "sampler", "ddim",
            "the 2026-09-06 defect verbatim: a record claiming a denoiser the "
            "weights do not contain")
    run_arm(m, model, stamp, rows, "A-w_u0-positive-no-control-head",
            "w_u0", 1.0,
            "a stamped reconstruction weight with nothing to supervise")
    run_arm(m, model, stamp, rows, "A-trunk-timm-on-refc-encoder",
            "trunk", "timm",
            "a record claiming an ImageNet prior the weights do not have")
    run_arm(m, model, stamp, rows, "A-cross_agent-true-no-agent-attn",
            "cross_agent", True,
            "a stamped agent seam whose tokens reach nothing")
    run_arm(m, model, stamp, rows, "A-agents-block-no-detector",
            "agents", {"n_queries": 100},
            "a record claiming a detector that was never built")
    run_arm(m, model, stamp, rows, "A-ego_history-enabled-not-built",
            "ego_history", {"enable": True, "kind": "gru"},
            "a record claiming an input the model never reads")
    run_arm(m, model, stamp, rows, "A-goal_point-inject-no-head",
            "goal_point", {"goal_point_inject": True, "graft_gp_point": False,
                           "w_goal_point": 1.0, "gp_slot": 0},
            "a record claiming an E15 head the weights do not contain")

    print("\n  --- B: RECIPE fields — the optimiser is ALREADY BUILT at the "
          "call site (refc_v3_train.py:7049 vs :7106) ---")
    run_arm(m, model, stamp, rows, "B-opt-dd-on-a-plain-Adam-run",
            "opt", "dd",
            "SPEC §2's optimiser. config.json is the only record of the "
            "recipe, and the optimiser object exists 57 lines before the "
            "guard runs, so this IS checkable.")
    run_arm(m, model, stamp, rows, "B-encoder_lr_mult-2.0",
            "encoder_lr_mult", 2.0,
            "the encoder learning-rate multiplier the run reports.")
    run_arm(m, model, stamp, rows, "B-weight-decay-1e-2",
            "weight_decay", 1e-2,
            "SPEC §2: wd 1e-4, NOT torch AdamW's 1e-2. Under `--opt adam` the "
            "real wd is 0.0 and the stamp says 1e-4 either way.")

    ok2, _ = check(m, model, stamp)
    print(f"\n  CONTROL (clean stamp again): "
          f"{'⛔ now refuses — INVALID' if ok2 else 'still passes'}")
    if ok2:
        raise SystemExit("INVALID: the arms contaminated the shared stamp.")

    # ========== PASS 2: a real timm resnet34, RANDOM init ================ #
    print("\n== PASS 2: a real timm resnet34, randomly initialised ==")
    a2, c2, m2, s2 = build(m, ["--trunk", "timm", "--trunk-name", "resnet34",
                               "--trunk-in-channels", "3",
                               "--no-trunk-pretrained"])
    r2, msg2 = check(m, m2, s2)
    print("  CONTROL (clean):", "⛔ REFUSED" if r2 else "passes (correct)")
    if r2:
        print("  ", msg2)
        raise SystemExit("INCONCLUSIVE: the timm control arm already refuses.")
    print(f"  encoder={type(m2.core.encoder).__name__}  trunk={s2.get('trunk')!r} "
          f"trunk_name={s2.get('trunk_name')!r} "
          f"pretrained={s2.get('trunk_pretrained')!r} "
          f"imagenet_norm={s2.get('trunk_imagenet_norm')!r} "
          f"in_channels={s2.get('trunk_in_channels')!r}")
    run_arm(m, m2, s2, rows, "B2-trunk_name-r34-record-says-r101",
            "trunk_name", "resnet101.a1_in1k",
            "SPEC §10.2: resnet101 is the PRIMARY arm, resnet34 the comparison "
            "run. The record is the only thing that says which one ran.")
    run_arm(m, m2, s2, rows, "B2-pretrained-random-record-says-ImageNet",
            "trunk_pretrained", True,
            "the registered ImageNet KNOCKOUT arm. If the record can claim a "
            "prior the weights do not have, 'the prior helps' is unfalsifiable.")
    run_arm(m, m2, s2, rows, "B2-in_channels-3-record-says-9",
            "trunk_in_channels", 9,
            "§10.3's K-frame history is read off in_channels; the record would "
            "name a different input stack (K=3 vs K=1).")

    # PASS 3: build with ImageNet norm OFF, then claim it ON.
    print("\n== PASS 3: timm trunk with ImageNet normalisation OFF ==")
    try:
        a3, c3, m3, s3 = build(m, ["--trunk", "timm", "--trunk-name",
                                   "resnet34", "--trunk-in-channels", "3",
                                   "--no-trunk-pretrained",
                                   "--no-trunk-imagenet-norm"])
    except SystemExit as ex:
        print(f"  (no such flag / refused: {str(ex)[:160]}) — arm NOT RUN")
        s3 = None
    if s3 is not None:
        r3, _ = check(m, m3, s3)
        print("  CONTROL (clean):", "⛔ REFUSED" if r3 else "passes (correct)")
        print(f"  imagenet_norm in stamp = {s3.get('trunk_imagenet_norm')!r}")
        if not r3:
            run_arm(m, m3, s3, rows,
                    "B3-imagenet_norm-OFF-record-says-ON",
                    "trunk_imagenet_norm", True,
                    "advisory class A: a record claiming the checkpoint's "
                    "declared preprocessing it did not apply. Class A measured "
                    "that omission at rel L2 0.562 on REFe.")

    n_sil = sum(1 for r in rows if not r["refused"])
    print(f"\n  arms: {len(rows)}   REFUSED: {len(rows) - n_sil}   "
          f"SILENT: {n_sil}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"stamp_keys_pass1": sorted(stamp), "arms": rows},
            indent=2, default=str), encoding="utf-8")
        print(f"[artifact] {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
