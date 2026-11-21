#!/usr/bin/env python3
"""Self-test infrastructure of the LAUNCH-INTENT chain: result bookkeeping, scratch worlds built from DAG-OACC's REAL banked records, module loading (original or MUTATED text).

A MUTANT is a deliberate regression: a world / log / file with ONE thing wrong that the registered check must REFUSE for the reason it was built to trip. A mutant that is accepted (or refused for another reason) is a SURVIVOR and fails the run.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                  # noqa: E402

SHIP = C.SHIP
SCR = C.SCRATCH / "selftest"
ARMS = C.LI_ARMS
OACC, DAG1, REPDIR = C.OACC, C.DAG1, C.REPDIR
RUNS = r"D:\tanitad\runs\v8f_mini_full"
PYEXE, TRAIN = r"D:\tanitad\venv\Scripts\python.exe", r"D:\tanitad\repo\stack\scripts\v8f_mini_train.py"
W_ACC, LI_W = C.WACC, float(C.LI_W)
QUIET = "--quiet" in sys.argv
RESULTS, MUTANTS = [], []


def T(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    if not QUIET or not cond:
        print(("ok   " if cond else "FAIL ") + name + ("" if cond else f"  <- {str(detail)[:400]}"), flush=True)


def M(name, killed, why=""):
    MUTANTS.append((name, bool(killed)))
    if not QUIET or not killed:
        print(("killed   " if killed else "SURVIVOR ") + name + ("" if killed else f"  <- {str(why)[:400]}"), flush=True)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


def _rw_retry(func, path, _exc):
    """shutil.rmtree onerror: git writes read-only object files; clear the flag and retry (otherwise a re-run finds the old world still there)."""
    os.chmod(path, 0o700)
    func(path)


def fresh(sub) -> Path:
    d = SCR / sub
    if d.exists():
        shutil.rmtree(d, onerror=_rw_retry)
    d.mkdir(parents=True)
    return d


def load(p):
    return json.load(open(p, encoding="utf-8"))


def put(p: Path, text):
    """Write bytes (or utf-8 text, no newline translation) creating the parent."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))


class NS:
    def __init__(self, d):
        self.__dict__.update(d)


# ---------------------------------------------------------------------------------------------------------------------------------------- module loading (original or mutated text)
_MODS = {}


def _load_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def mods(gtext: str | None = None, ktext: str | None = None):
    """(G, K) loaded from a scratch directory. Original text when None; a MUTATED text otherwise. K imports G from its own directory, so both are written there."""
    gt = gtext if gtext is not None else (SHIP / "gate_launchintent_assert.py").read_text(encoding="utf-8")
    kt = ktext if ktext is not None else (SHIP / "v8f_launchintent_chain_checks.py").read_text(encoding="utf-8")
    key = (hashlib.sha256(gt.encode()).hexdigest(), hashlib.sha256(kt.encode()).hexdigest())
    if key in _MODS:
        return _MODS[key]
    d = SCR / "mods" / (key[0][:8] + "_" + key[1][:8])
    d.mkdir(parents=True, exist_ok=True)
    (d / "gate_launchintent_assert.py").write_bytes(gt.encode("utf-8"))
    (d / "v8f_launchintent_chain_checks.py").write_bytes(kt.encode("utf-8"))
    K = _load_file(d / "v8f_launchintent_chain_checks.py", "k_" + key[1][:8] + key[0][:4])
    _MODS[key] = (K.G, K)
    return _MODS[key]


# ---------------------------------------------------------------------------------------------------------------------------------------- the gate world (DAG-OACC's REAL gate selfcheck + the launch lever grafted in)
def line(argv) -> str:
    return f'"{PYEXE}" "{TRAIN}" ' + " ".join(argv)


def ctrl_dir(arm: str, seed: int) -> Path:
    """The run's PAIR CONTROL's banked directory: DAG-OACC_s for A_s, REP_s (DAG-ORC r2's M2b_s<s>_dagoaccorcrep) for A_rep,s."""
    return OACC / f"M2b_s{seed}_dagoacc" if arm == "launchintent" else REPDIR / f"M2b_s{seed}_dagoaccorcrep"


def arm_argv(G, seed: int, out=None, arm: str = "launchintent") -> list:
    """The registered training argv built from the PAIR CONTROL's banked argv, with --out set to the arm's own run directory at the reference's position."""
    ref = load(ctrl_dir(arm, seed) / "argv.json")["argv"]
    exp = G.expected_from_ref(ref, seed, arm)
    k = ref.index("--out")
    return exp[:k] + ["--out", out or f"{RUNS}\\M2b_s{seed}_{arm}"] + exp[k:]


def probe_record(W=LI_W, w_traj=1.0):
    term = 2.5
    want = w_traj * W * term / 3.0
    return {"launch_intent_w_declared": W, "launch_n": 3, "launch_eligible": 64, "launch_n_forced": 32, "term_forced": term, "term_at_zero_weight": None, "total_w": 19.5, "total_0": 19.5 - want, "contribution": want * 1.00001,
            "w_traj_x_W_x_term_over_3": want, "head_o_grad_diff": 0.37, "read_o_grad_diff": 0.11, "natural_contribution": 0.2, "natural_term": 0.5, "natural_head_o_grad_diff": 0.05,
            "control_repeat_total_diff": 0.0, "control_repeat_grad_diff": 0.0, "control_allfalse_term": 0.0, "control_allfalse_total_diff": 0.0, "control_allfalse_grad_diff": 0.0,
            "control_prior_grad_diff": 0.0, "control_wm_grad_diff": None, "pass": True}


def mask_record(W=LI_W):
    return {"launch_intent_w": W, "v0_max": 6.0, "dv": 1.0, "dt": 0.2, "knots": [1, 12], "axis": "along (x of the ego frame at NOW)", "mask_from": "the LOGGED kin3[:, 0] and the GT future, taken before any ego stage"}


def graft_levers(rec: dict, W=LI_W, n=3, e=64):
    lv = rec["levers"]
    lv["launch_intent_w"] = W
    lv["launch_intent"] = mask_record(W)
    lv["launch_n"], lv["launch_eligible"] = n, e


def edit(path: str, value=None, delete=False):
    """A mutator for a dict: set / delete the dotted ``path``."""
    ks = path.split(".")

    def f(d):
        cur = d
        for k in ks[:-1]:
            cur = cur[k]
        if delete:
            cur.pop(ks[-1], None)
        else:
            cur[ks[-1]] = value
    return f


def gate_world(G, root: Path, seed: int, mutate=None, line_mutate=None, pin_mutate=None, gate_mutate=None, sha_override=None, over=None, arm: str = "launchintent") -> dict:
    """Write one world under ``root``; return the argument dict for ``check`` / ``token``.  The base records are the PAIR CONTROL's (DAG-OACC_s for A_s, REP_s for A_rep,s)."""
    root.mkdir(parents=True, exist_ok=True)
    cd = ctrl_dir(arm, seed)
    sc = copy.deepcopy(load(cd / "gate" / "selfcheck.json"))
    train = arm_argv(G, seed, arm=arm)
    if line_mutate:
        train = line_mutate(list(train))
    k = train.index("--out")
    gate = train[:k + 1] + [f"{RUNS}\\gate_M2b-{arm}_s{seed}"] + train[k + 2:] + ["--selfcheck-only"]
    if gate_mutate:
        gate = gate_mutate(list(gate))
    sc["argv_sha"] = sha_override or hashlib.sha256(" ".join(gate).encode()).hexdigest()[:16]
    graft_levers(sc)
    sc["launch_intent_probe"] = probe_record()
    if mutate:
        mutate(sc)
    files = {"selfcheck.json": json.dumps(sc), "gate_line.txt": line(gate), "train_line.txt": line(train), "pin_line.txt": line(pin_mutate(list(train)) if pin_mutate else train)}
    for n, t in files.items():
        (root / n).write_bytes(t.encode("utf-8"))
    tok = load(cd / "launch_gate_token.json")
    ref = load(cd / "argv.json")["argv"]
    d = dict(mode="check", selfcheck=str(root / "selfcheck.json"), gate_line=str(root / "gate_line.txt"), train_line=str(root / "train_line.txt"), pin_line=str(root / "pin_line.txt"), arm=arm, seed=str(seed),
             banked_init_sha=tok["init_ckpt_sha256"], bank_sha=ref[ref.index("--dagger-bank-sha256") + 1], bank_json=str(DAG1 / f"bank_s{seed}" / "bank.json"), w_acc=W_ACC,
             b_summary=str(cd / "summary.json"), ref_argv_json=str(cd / "argv.json"), expect="", commit="", queue_sha="", key_sha_json="{}", run="", prereg_sha="", reader_sha="", out="")
    if over:
        d.update(over)
    return d


def run_gate(G, d: dict) -> list:
    return quiet(G.check, NS(d))


# ---------------------------------------------------------------------------------------------------------------------------------------- the run-record world (DAG-OACC's REAL run records + the launch keys grafted in)
COUNTS = {0: (30727, 191203), 1: (30674, 191396)}      # PREREG section 3: the simulated sampler stream at the A seeds (launch rows, rows)
COUNTS_REP = {0: (30790, 191311), 1: (30601, 191122)}  # FIXTURE values for A_rep,s (trainer seeds 10 / 11 draw a different stream; f = 0.1610 / 0.1601, inside f +- 0.02): NOT measured, only shaped like the registered ones


def record_world(G, K, root: Path, seed: int, mut_sm=None, mut_sc=None, mut_planned=None, mut_aj=None, mut_pin=None, over=None, arm: str = "launchintent") -> dict:
    root.mkdir(parents=True, exist_ok=True)
    n, e = (COUNTS if arm == "launchintent" else COUNTS_REP)[seed]
    planned = arm_argv(G, seed, arm=arm)
    if mut_planned:
        planned = mut_planned(list(planned))
    sha16 = hashlib.sha256(" ".join(planned).encode()).hexdigest()[:16]
    base = ctrl_dir(arm, seed)
    sm, sc = copy.deepcopy(load(base / "summary.json")), copy.deepcopy(load(base / "selfcheck.json"))
    for rec in (sm, sc):
        rec["argv_sha"] = sha16
        graft_levers(rec, LI_W, n, e)
    sm["launch_n"], sm["launch_eligible"], sm["launch_rows"] = n, e, e
    sm["launch_intent_w"] = LI_W
    sm["launch_weight_share"] = K.share_of(n, e, LI_W)
    aj = {"argv": planned, "sha": sha16}
    if mut_sm:
        mut_sm(sm)
    if mut_sc:
        mut_sc(sc)
    if mut_aj:
        mut_aj(aj)
    pin = mut_pin(list(planned)) if mut_pin else planned
    (root / "summary.json").write_bytes(json.dumps(sm).encode())
    (root / "selfcheck.json").write_bytes(json.dumps(sc).encode())
    (root / "argv.json").write_bytes(json.dumps(aj).encode())
    (root / "train_line.txt").write_bytes(line(planned).encode())
    (root / "pin_line.txt").write_bytes(line(pin).encode())
    tok = load(base / "launch_gate_token.json")
    ref = load(base / "argv.json")["argv"]
    d = dict(run_dir=str(root), train_line=str(root / "train_line.txt"), pin_line=str(root / "pin_line.txt"), seed=str(seed), arm=arm, banked_init_sha=tok["init_ckpt_sha256"], bank_sha=ref[ref.index("--dagger-bank-sha256") + 1], w_acc=W_ACC,
             steps="3000", expect="", summary=str(root / "summary.json"), window_sha=load(base / "summary.json")["train_window_list_sha256"])
    if over:
        d.update(over)
    return d


def sm_with_counts(K, n, e, rows=None, share="auto"):
    def f(sm):
        sm["launch_n"], sm["launch_eligible"], sm["launch_rows"] = n, e, (e if rows is None else rows)
        sm["launch_weight_share"] = K.share_of(n, e, LI_W) if share == "auto" else share
    return f


# ---------------------------------------------------------------------------------------------------------------------------------------- the REP world (DAG-ORC r2's REAL REP pulls of BOTH seeds: the predecessor ended DONE 2026-10-28 07:44)
def rep_world(root: Path, seed: int, mut=None) -> dict:
    """Returns {"rep_dir", "oacc_dir"}; ``mut(files)`` edits the dict of parsed json files {summary, selfcheck, argv, token} before they are written."""
    real = REPDIR / f"M2b_s{seed}_dagoaccorcrep"
    o = OACC / f"M2b_s{seed}_dagoacc"
    files = {"summary": load(real / "summary.json"), "selfcheck": load(real / "selfcheck.json"), "argv": load(real / "argv.json"), "token": load(real / "launch_gate_token.json")}
    rd, od = root / f"M2b_s{seed}_dagoaccorcrep", root / f"M2b_s{seed}_dagoacc"
    if mut:
        mut(files)
    for dd in (rd, od):
        dd.mkdir(parents=True, exist_ok=True)
    for name, f in (("summary", "summary.json"), ("selfcheck", "selfcheck.json"), ("argv", "argv.json"), ("token", "launch_gate_token.json")):
        (rd / f).write_bytes(json.dumps(files[name]).encode())
    for f in ("summary.json", "argv.json", "launch_gate_token.json", "selfcheck.json"):
        shutil.copyfile(o / f, od / f)
    return {"rep_dir": str(rd), "oacc_dir": str(od)}


def rep_args(w: dict, seed: int, log="", over=None) -> dict:
    tok = load(OACC / f"M2b_s{seed}_dagoacc" / "launch_gate_token.json")
    ref = load(OACC / f"M2b_s{seed}_dagoacc" / "argv.json")["argv"]
    d = dict(rep_dir=w["rep_dir"], oacc_dir=w["oacc_dir"], seed=str(seed), init_sha=tok["init_ckpt_sha256"], bank_sha=ref[ref.index("--dagger-bank-sha256") + 1], w_acc=W_ACC,
             window_sha=load(OACC / f"M2b_s{seed}_dagoacc" / "summary.json")["train_window_list_sha256"], log=log)
    if over:
        d.update(over)
    return d


# ---------------------------------------------------------------------------------------------------------------------------------------- a synthetic COMPLETE DAG-ORC r2 chain log (the real one is still running when this is built)
R2_RUNS = ("M2b_s0_dagoaccorc", "M2b_s0_dagoaccorcrep", "M2b_s1_dagoaccorc", "M2b_s1_dagoaccorcrep")


def r2_log(done=True, extra_after="", drop_pull=None, drop_rr=None, drop_kreach=None, stop=None, guard=False, floor=False, two_done=False, fail_line=None, done_line=None) -> str:
    """Lines of the DAG-ORC r2 chain's log in the shapes the REAL log has (MEASURED in chain_v8f_dagoaccorc_r2.out: ' pulled to ', 'run record: PASS | <arm> s<S> |', 'K-REACH: PASS | <arm> |'), then its DONE marker."""
    L = ["10-28 03:48 waiting for the r1 chain (x)", "10-28 03:48 predecessor verdict: GO the r1 chain ended on its K-REACH stop"]
    for run in R2_RUNS:
        s, arm = run.split("_")[1][1:], run.split("_")[2]
        if run != drop_pull:
            L.append(f"10-28 05:19 {run} pulled to /e/Projects/TanitAD-artifacts/v8f-dag-orc-oacc/{run} (checkpoint sha256 {'ab' * 32})")
        if run != drop_rr:
            L.append(f"DAG-ORC (DAG-OACC base) run record: PASS | {arm} s{s} | trainer seed {10 + int(s) if arm.endswith('rep') else s} | argv sha16 7c6d0c0e8f3cf78c | steps 3000 | ego_perturb_p 0.0 | w_acc 49.6969")
        if arm != drop_kreach:
            L.append(f"DAG-ORC (DAG-OACC base) K-REACH: PASS | {arm} | swap 14659/144643 = 0.1013 (must lie in 0.1 +- 0.02) | ego 0/82118 | overlap 0 | flag False")
    if fail_line:
        L.append(fail_line)
    if guard:
        L.append("ZZGUARD-STOP TEMP 10-28 06:00 USB HDD: 5 of the last 20 samples >= 61 C, max 64 C")
    if floor:
        L.append("ZZDISK-FLOOR-STOP pulling x would leave less than y")
    if stop:
        L.append(stop)
    dl = done_line or ("ZZV8FMINI-DONE 10-28 08:30 tip " + "c" * 40 + ": PREREG_V8F_DAG_ORC Amendment 1 (DAG-OACC base) r2: 4 runs (dagoaccorc 0 dagoaccorcrep 0 dagoaccorc 1 dagoaccorcrep 1) each through its gate (boot unchanged 2026-10-17 18:43:26)")
    if done:
        L.append(dl)
        if two_done:
            L.append(dl)
    return "\n".join(L) + "\n" + extra_after
