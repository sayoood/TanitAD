#!/usr/bin/env python3
"""Fixture worlds for the DAG-ORC (DAG-OACC base) gate asserter and run-record checks, built from DAG-OACC's REAL banked gate / run artifacts (v8f-dagoacc/M2b_s<S>_dagoacc/...):

  gate/selfcheck.json  DAG-OACC's own gate selfcheck (real G-LIVE, dagger_probe, acc_probe, info.dagger, the built levers of the trained-M2b fine-tune with --ego-perturb-p 0)
  argv.json            DAG-OACC's banked training argv (the ONE-LEVER reference)
  summary.json         DAG-OACC's banked run summary (the gate comparator B)

REP world = DAG-OACC's gate selfcheck with the argv lines changed to REP's (--seed 10 / 11): everything else is the real record.  A world = the same with the PERTURBATION grafted in:
the real ego_perturb_probe of an ego-on gate (the DAG-ORC build's fixtures/dagorc_s0_gate/selfcheck.json, a CPU gate of the real trainer), levers.ego_perturb at p 0.3 + the law, levers.ego_perturb_with_dagger True, and the
argv lines with the perturbation and the composition flag.  The grafted probe is therefore a real record of the trainer's own probe, not a shape typed here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "box2_ship"))
import gate_dagoaccorc_assert as G                      # noqa: E402

OACC = Path("E:/Projects/TanitAD-artifacts/v8f-dagoacc")
DAG1 = Path("E:/Projects/TanitAD-artifacts/v8f-dagger")
EGO_FIX = Path("E:/Projects/TanitAD-artifacts/dag_orc_chain/fixtures/dagorc_s0_gate/selfcheck.json")
PYEXE, TRAIN = r"D:\tanitad\venv\Scripts\python.exe", r"D:\tanitad\repo\stack\scripts\v8f_mini_train.py"
RUNS = r"D:\tanitad\runs\v8f_mini_full"
W_ACC = "49.6969"


def load(p):
    return json.load(open(p, encoding="utf-8"))


def line(argv) -> str:
    """A queue-printed trainer line: quoted python and script, the argv tokens (paths with no spaces need no quotes; argv_of strips quotes anyway)."""
    return f'"{PYEXE}" "{TRAIN}" ' + " ".join(argv)


def arm_argv(arm: str, seed: int, ref=None) -> list:
    """The registered training argv of ``arm`` built from DAG-OACC's banked argv, with --out set to the arm's own run directory."""
    ref = ref or load(OACC / f"M2b_s{seed}_dagoacc" / "argv.json")["argv"]
    exp = G.expected_from_ref(ref, arm, seed)
    k = ref.index("--out")
    # drop() removed --out in expected_from_ref: put it back at the reference's position
    out = f"{RUNS}\\M2b_s{seed}_{arm}"
    return exp[:k] + ["--out", out] + exp[k:]


def build(root: Path, arm: str, seed: int, mutate=None, line_mutate=None, pin_mutate=None, gate_mutate=None, gate_sha_override=None) -> dict:
    """Write one world under ``root``; return the argument dict for ``check`` / ``token``. ``mutate(sc)`` edits the selfcheck dict; ``line_mutate(argv_train)`` edits the planned training argv (the gate / pin lines follow)."""
    root.mkdir(parents=True, exist_ok=True)
    sc = copy.deepcopy(load(OACC / f"M2b_s{seed}_dagoacc" / "gate" / "selfcheck.json"))
    p = G.ARMS[arm]
    train = arm_argv(arm, seed)
    if line_mutate:
        train = line_mutate(list(train))
    k = train.index("--out")
    gate = train[:k + 1] + [f"{RUNS}\\gate_M2b-{arm}_s{seed}"] + train[k + 2:] + ["--selfcheck-only"]
    if gate_mutate:
        gate = gate_mutate(list(gate))
    sc["argv_sha"] = gate_sha_override or hashlib.sha256(" ".join(gate).encode()).hexdigest()[:16]
    if p > 0.0:
        ego = load(EGO_FIX)
        sc["ego_perturb_probe"] = copy.deepcopy(ego["ego_perturb_probe"])
        sc["ego_perturb_probe"]["p_declared"] = p
        sc["levers"]["ego_perturb"] = {**sc["levers"]["ego_perturb"], "ego_perturb_p": p, **G.LAW_VALUES, "ego_perturb_seed": seed}
        sc["levers"]["ego_perturb_with_dagger"] = True
    if mutate:
        mutate(sc)
    files = {"selfcheck.json": json.dumps(sc), "gate_line.txt": line(gate), "train_line.txt": line(train), "pin_line.txt": line(pin_mutate(list(train)) if pin_mutate else train)}
    for n, t in files.items():
        (root / n).write_text(t, encoding="utf-8", newline="\n")
    banked_init = load(OACC / f"M2b_s{seed}_dagoacc" / "launch_gate_token.json")["init_ckpt_sha256"]
    bank_sha = load(OACC / f"M2b_s{seed}_dagoacc" / "argv.json")["argv"]
    bank_sha = bank_sha[bank_sha.index("--dagger-bank-sha256") + 1]
    return dict(mode="check", selfcheck=str(root / "selfcheck.json"), gate_line=str(root / "gate_line.txt"), train_line=str(root / "train_line.txt"), pin_line=str(root / "pin_line.txt"), arm=arm, seed=str(seed),
                banked_init_sha=banked_init, bank_sha=bank_sha, bank_json=str(DAG1 / f"bank_s{seed}" / "bank.json"), w_acc=W_ACC, b_summary=str(OACC / f"M2b_s{seed}_dagoacc" / "summary.json"),
                ref_argv_json=str(OACC / f"M2b_s{seed}_dagoacc" / "argv.json"), expect="", commit="", queue_sha="", key_sha_json="{}", run="", out="")


class NS:
    def __init__(self, d):
        self.__dict__.update(d)


def run_check(d: dict) -> list:
    return G.check(NS(d))
