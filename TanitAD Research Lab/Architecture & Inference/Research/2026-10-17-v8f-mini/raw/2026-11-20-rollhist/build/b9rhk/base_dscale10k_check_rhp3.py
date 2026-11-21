"""[ROLLHIST P03 build: dscale10k_check_rhp3.py = dscale10k_check_rh.py 69ff1a0f + the RHp3 kind (patch_rhp3.py); PREREG_V8F_ROLLHIST Amendment 7 DRAFT]
PREREG_V8F_DSCALE10K study checker (card section 5; MM freeze entry 2026-11-16: NO ``stack/`` change). Mints ``dscale10k-gate-token-v1``.

The pattern of ``stack/scripts/strat1_check.py`` (token = a record written ONLY by ``mint`` after every required check read PASS, bound to the commit, the
argv sha256, every data file, the tree, the trainer and this file; ``run`` REFUSES (rc 2) without a matching token). What is new here:

  * the accepted v9-md5 set is STUDY-LOCAL (``accepted_md5_set``): corpus ``83e30bd3...``, 10k ``39522144...``, plus the union / R md5 only after C-MERGE passed.
    ``ACCEPTED_TRAIN_MD5`` in ``v8f_mini_data.py`` has no reader (MM ruling 1b), so THIS file is the reader.
  * ``argv_diff_problems``: arm vs B_s0 may differ only in the flags the card section 5 lists (``--exclude`` on a union argv = G-R3).
  * data checks C-MERGE, C-JOIN-EQ, C-MASK (G-R2), C-DISJ (G-R1), C-HOLD, C-EXCL, C-STEPS, C-NCLIPS, C-CHUNK, C-TRAIN-SHA, report-only C-NAVCOV, C-LEADCOV, C-INIT, C-SESS.

Every ``*_problems`` function returns a list of strings, EMPTY = pass, and has a deliberate-regression arm in ``test_dscale10k.py`` that must make it non-empty.

POD-SIDE, NOT YET RUN (they need the pod's data or the trainer's GPU run): ``dry`` against the real union cache, the real-data G-LIVE reach probe (this file only
DECIDES from its record: ``reach_problems``), G-EVAL, G-CKPT, G-SUITE, the clock residuals of G-CLOCK, C-CACHE, C-SESS. They are marked ``"pod_side"`` in the evidence schema below.

Subcommands: ``diff`` (argv vs base), ``dry`` (build_data record), ``mint``, ``run``.

THIS FILE (dscale10k_check_rh.py, ROLLHIST gate build B3, 2026-11-20) is the reps1 copy (``blockreps1/study/dscale10k_check.py``, sha256 ac37c1b6..., itself a descendant of the
stage-B checker ``study/dscale10k_check.py`` 7467e5d7... that minted the U_s0 / U_s1 stage-B tokens) plus FOUR arm kinds, stage B only: ``RH_s0``, ``RH_s1`` (the roll-history DAgger arm)
and ``B0_s0``, ``B0_s1`` (its p = 0 control). Every other arm, check and token field is UNCHANGED (the identity test proves it). What is new:
  * ``rh_argv_problems``   argv == the arm's ``U_sK.B.tmpl.json`` except exactly: RH: the two declarations (``--dagger-bank-readout <token>`` ``--dagger-bank-kappa 0.2``), the arm's own
                           ``--out``, ``--init-ckpt`` (its OWN stage-A model ``runs/UE_sK/A/ckpt_last.pt``, K-B2) and the bank / bank sha256; B0: ``--dagger-p 0``, no bank flag, no declaration, own ``--out`` / ``--init-ckpt``.
  * ``rh_bank_generator_problems``  bank.json sha256 == ``--dagger-bank-sha256``; the bank's ``generator`` record (as ``DaggerSwap.built_levers`` reads it: absent = ``pinned`` / ``none``) equals the declared token and kappa.
  * ``rh_pair_problems``   ``stack/scripts/rollhist_pair_check.py`` PASS on the pair. That script hard-codes ``causal_g1x``: for ``causal_g1`` this checker REFUSES (a committed-code change is the MM's) and never patches it in process.
  * the token binds the template, the pair argv and the pair-check file besides everything the stage-B token binds; the declared readout token is recorded in ``checks["C-RH-PAIR"]``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import dscale10k_merge as M  # noqa: E402

TOKEN_SCHEMA = "dscale10k-gate-token-v1"
WRITTEN_BY = "dscale10k_check.py mint"
CARD_SHA256_PREFIX = "48a74901"                     # the FROZEN card (MM freeze 2026-11-16); the full value is the MM's. UNVERIFIED here: only the prefix was given
CODE_COMMIT = "648d4ca3"                             # blockum: the UM_s0 code of record (child of cdbff971)
TAC_FLAGS = ("--tac-src-w", "--tac-src-clips", "--tac-src-clips-sha256")
TAC_W = "0.0"
TAC_LIST = "/root/dscale10k/mix/bset_clips.txt"
TAC_SHA = "37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1"     # B_s0's MEASURED train_sha256 = the byte sha256 of bset_clips.txt (card section 4)
UM_OUT = "/root/dscale10k/runs/UM_s0/A"
N_TENK = 10000
CORPUS_JOIN_MD5_PREFIX = "9ebc7a48"                 # join_train.npz (M via STRAT1); the full md5 is not in the card: prefix only
BASE_PARAMS = 7_214_495                              # M2b trainable parameters (card 5, G-HYG; M2b summary)
RUN_STEPS_A = 19164
STAGE_B_STEPS = 3000
CHUNK = 48
B_NCLIPS, U_NCLIPS, X_NCLIPS = 4512, 14512, 10000
GOAL22 = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "YIELD", "STOP_POINT", "SPEED_BAND", "CORRIDOR_OFFSET", "EVADE_IN_CORRIDOR",
          "OVERTAKE_VEHICLE", "MERGE", "GAP_TARGET", "REACT_ON_ONCOMING", "TAKE_EXIT_L", "TAKE_EXIT_R", "TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED",
          "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT_GREEN", "LANE_CHANGE_L", "LANE_CHANGE_R")

DATA_FLAGS = ("--cache-dir", "--v9-train", "--v9-train-md5", "--lead-train", "--speed-train", "--tracks", "--join-train")
UNION_FLAG_DELTA = set(DATA_FLAGS) | {"--out"}
#: arm -> set ("B" = 4,512; "U" = union 14,512; "R" = union with the 10k labels deranged; "X" = the 10k alone), stage-B yes/no, seed, dagger-seed, stage-A step rule
ARMS = {
    "B_s0": dict(set="B", stage_b=True, seed=0, dagger_seed=0, steps="mult"),
    "U_s0": dict(set="U", stage_b=True, seed=0, dagger_seed=0, steps="fixed"),
    "B_rep": dict(set="B", stage_b=True, seed=2, dagger_seed=0, steps="mult"),
    "R_s0": dict(set="R", stage_b=False, seed=0, dagger_seed=None, steps="fixed"),
    "B_s1": dict(set="B", stage_b=True, seed=1, dagger_seed=1, steps="mult"),
    "U_s1": dict(set="U", stage_b=True, seed=1, dagger_seed=1, steps="fixed"),
    "UE_s0": dict(set="U", stage_b=False, seed=0, dagger_seed=None, steps="ue"),
    "BL_s0": dict(set="B", stage_b=False, seed=0, dagger_seed=None, steps="ue"),
    "X_s0": dict(set="X", stage_b=True, seed=0, dagger_seed=0, steps="fixed"),
    "UM_s0": dict(set="U", stage_b=False, seed=0, dagger_seed=None, steps="ue"),      # blockum (PREREG_V8F_DSCALE_MIX section 4): UE_s0's argv + the three --tac-src-* flags
    "UE_s1": dict(set="U", stage_b=False, seed=1, dagger_seed=None, steps="ue"),      # REPS1: the seed-1 replicate of UE_s0 (Amendment UM-A2 DRAFT)
    "UM_s1": dict(set="U", stage_b=False, seed=1, dagger_seed=None, steps="ue"),      # REPS1: the seed-1 replicate of UM_s0
    "B_s3": dict(set="B", stage_b=True, seed=3, dagger_seed=3, steps="mult"), "U_s3": dict(set="U", stage_b=True, seed=3, dagger_seed=3, steps="fixed"),
    "B_s4": dict(set="B", stage_b=True, seed=4, dagger_seed=4, steps="mult"), "U_s4": dict(set="U", stage_b=True, seed=4, dagger_seed=4, steps="fixed"),
}
MIX_ARMS = ("UM_s0", "UM_s1")
#: ROLLHIST stage-B arms (this file): arm -> (kind, seed). Union set, stage B only; the template is ``argv/U_s<seed>.B.tmpl.json``; the init is the arm's own ``runs/UE_s<seed>/A/ckpt_last.pt``.
RH_ARMS = {"RH_s0": ("RH", 0), "RH_s1": ("RH", 1), "B0_s0": ("B0", 0), "B0_s1": ("B0", 1), "RHp3_s0": ("RHp3", 0), "RHp3_s1": ("RHp3", 1)}
#: ROLLHIST next arm (card 3.4 PARTIAL row, Amendment 7): RHp3 = the RH arm with --dagger-p EXACTLY 0.3 on the SAME bank. RH_LIKE = the kinds that carry a bank and the two declarations.
RH_LIKE = ("RH", "RHp3")
RHP3_P = "0.3"
for _a, (_k, _s) in RH_ARMS.items():
    ARMS[_a] = dict(set="U", stage_b=True, seed=_s, dagger_seed=_s if _s else 0, steps="fixed")
RH_TOKENS = ("causal_g1x", "causal_g1")
RH_KAPPA = "0.2"
RH_DECL = ("--dagger-bank-readout", "--dagger-bank-kappa")
RH_PAIR_CHECK_HARDCODED = "causal_g1x"
NCLIPS = {"B": B_NCLIPS, "U": U_NCLIPS, "R": U_NCLIPS, "X": X_NCLIPS}
STAGE_B_OWN = {"--init-ckpt", "--dagger-bank", "--dagger-bank-sha256"}          # the arm's OWN stage-A checkpoint and bank (card 2.1)


def sha_file(p):
    return M.sha256_file(p)


def md5_file(p):
    return M.md5_file(p)


# ------------------------------------------------------------------------------------------------ argv
def parse_argv(toks) -> dict:
    """``["--a", "1", "--flag", "--b", "x", "y"]`` -> ``{"--a": ["1"], "--flag": [], "--b": ["x", "y"]}`` (insertion order). A repeated flag or a leading value is refused."""
    out, cur = {}, None
    for t in toks:
        if isinstance(t, str) and t.startswith("--") and len(t) > 2:
            if t in out:
                raise ValueError(f"flag {t} appears twice")
            out[t] = []
            cur = t
        else:
            if cur is None:
                raise ValueError(f"value {t!r} before any flag")
            out[cur].append(str(t))
    return out


def argv_sha(toks):
    return hashlib.sha256(json.dumps(list(toks)).encode("utf-8")).hexdigest()


def argv_diff_problems(base_toks, toks, arm: str, stage: str = "A", ue_steps: int | None = None) -> list:
    """The G-DVB argv refusal (card section 5): ``toks`` (arm ``arm``, stage ``stage``) may differ from the B_s0 argv of the SAME stage ONLY in the flags the card lists.

    Allowed (and pinned) per arm: ``--out``; on a union / R / X arm the seven data flags, ``--fit-join`` added, ``--exclude`` removed, ``--steps`` added at 19,164
    (UE: ``ue_steps``; BL: B set + ``--steps`` = ``ue_steps``); on a different seed ``--seed`` (stage B: ``--dagger-seed``) at the arm's own value; stage B: this arm's own
    ``--init-ckpt`` / ``--dagger-bank`` / ``--dagger-bank-sha256``. Anything else, in either direction, is refused."""
    if arm not in ARMS:
        return [f"unknown arm {arm!r}"]
    spec = ARMS[arm]
    try:
        b, a = parse_argv(base_toks), parse_argv(toks)
    except ValueError as e:
        return [f"argv does not parse: {e}"]
    bad = []
    union = spec["set"] in ("U", "R", "X")
    allow_diff = {"--out"}
    allow_added, allow_removed = set(), set()
    if union:
        allow_diff |= set(DATA_FLAGS)
        allow_added |= {"--fit-join", "--steps"}
        allow_removed |= {"--exclude"}
    if arm in MIX_ARMS:
        allow_added |= set(TAC_FLAGS)
    if spec["steps"] in ("fixed", "ue") and not union:
        allow_added |= {"--steps"}                                                      # BL_s0: the B set at UE's step count
    if stage == "B":
        allow_diff |= STAGE_B_OWN
        allow_added |= STAGE_B_OWN
    allow_diff.add("--seed")                                                           # always allowed to differ, ALWAYS pinned to the arm's own value below
    pinned = {"--seed": [str(spec["seed"])]}
    if stage == "B" and spec["dagger_seed"] is not None:
        allow_diff.add("--dagger-seed")
        allow_added.add("--dagger-seed")
        pinned["--dagger-seed"] = [str(spec["dagger_seed"])]
    if spec["steps"] == "fixed" and stage == "A":
        pinned["--steps"] = [str(RUN_STEPS_A)]
    if spec["steps"] == "ue" and stage == "A":
        if ue_steps is None:
            bad.append("UE / BL arms need ue_steps (4 x S_U from the dry build)")
        else:
            pinned["--steps"] = [str(int(ue_steps))]
    if arm in MIX_ARMS:
        pinned.update({"--tac-src-w": [TAC_W], "--tac-src-clips": [TAC_LIST], "--tac-src-clips-sha256": [TAC_SHA]})
        for f in TAC_FLAGS:
            if f not in a:
                bad.append(f"UM_s0 argv has no {f}")
    if stage == "A" and spec["steps"] == "mult" and "--steps" in a:
        bad.append("--steps on a matched-epochs (--steps-mult) arm: C-STEPS")
    for f in a:
        if f not in b and f not in allow_added:
            bad.append(f"flag {f} {a[f]} is not in the B_s0 argv and is not an allowed addition for {arm}")
    for f in b:
        if f not in a and f not in allow_removed:
            bad.append(f"flag {f} {b[f]} of the B_s0 argv is missing from the {arm} argv")
    for f in a:
        if f in b and a[f] != b[f] and f not in allow_diff:
            bad.append(f"flag {f}: {a[f]} differs from the B_s0 value {b[f]} (not an allowed difference for {arm})")
    for f, v in pinned.items():
        if f in a and a[f] != v:
            bad.append(f"flag {f} reads {a[f]}, the card pins {v} for {arm}")
        if f not in a and f in ("--steps",):
            bad.append(f"{arm} stage {stage} has no {f} (the card pins {v})")
    if stage == "A" and "--seed" not in a:
        bad.append("no --seed on the argv")
    if union and "--exclude" in a:
        bad.append(f"C-EXCL / G-R3: --exclude {a['--exclude']} on a {spec['set']}-set argv (the 60 holdout clips are removed from the union files by id)")
    if not union and "--exclude" not in a:
        bad.append("C-EXCL: a B-set arm without --exclude (the 60 holdout clips would train)")
    if not union and a.get("--exclude") not in (None, ["540:600"]):
        bad.append(f"C-EXCL: B-set --exclude reads {a.get('--exclude')}, expected ['540:600']")
    if union and "--fit-join" not in a:
        bad.append("a union arm without --fit-join: the join would be read from a prebuilt file")
    if union and a.get("--join-train") == b.get("--join-train"):
        bad.append("a union arm that names the corpus --join-train path: it would read (or overwrite) the corpus join")
    return bad


# ------------------------------------------------------------------------------------------------ blockum: C-ARGV-MIX, C-TACSRC-LIST (card sections 4 and 7)
def mix_argv_problems(ue_toks, toks) -> list:
    """C-ARGV-MIX: UM's argv differs from UE_s0.A's ONLY in --out (= UM_OUT) and the three --tac-src-* flags at exactly the pinned values; lambda 0 needs a list, lambda 1 with a list does nothing (G-RM5 / G-RM6)."""
    try:
        u, a = parse_argv(ue_toks), parse_argv(toks)
    except ValueError as e:
        return [f"argv does not parse: {e}"]
    bad = []
    added = [f for f in a if f not in u]
    if sorted(added) != sorted(TAC_FLAGS):
        bad.append(f"flags added to UE_s0's argv: {sorted(added)}, expected exactly {sorted(TAC_FLAGS)}")
    bad += [f"flag {f} of UE_s0's argv is missing" for f in u if f not in a]
    for f in u:
        if f in a and a[f] != u[f] and f != "--out":
            bad.append(f"flag {f}: {a[f]} differs from UE_s0's {u[f]}")
    exp_out = [(u.get("--out") or [""])[0].replace("/runs/UE_", "/runs/UM_")]
    if a.get("--out") != exp_out or exp_out == u.get("--out"):
        bad.append(f"--out reads {a.get('--out')}, expected {exp_out}")
    for f, v in (("--tac-src-w", [TAC_W]), ("--tac-src-clips", [TAC_LIST]), ("--tac-src-clips-sha256", [TAC_SHA])):
        if a.get(f) != v:
            bad.append(f"{f} reads {a.get(f)}, the card pins {v}")
    lam = a.get("--tac-src-w", ["1.0"])[0]
    has_list = bool(a.get("--tac-src-clips"))
    try:
        lf = float(lam)
    except ValueError:
        return bad + [f"--tac-src-w {lam!r} is not a number"]
    if lf != 1.0 and not has_list:
        bad.append("G-RM5: --tac-src-w != 1.0 without --tac-src-clips")
    if lf == 1.0 and has_list:
        bad.append("G-RM6: --tac-src-clips with --tac-src-w 1.0 (declared, does nothing)")
    return bad


def tacsrc_list_problems(raw: bytes, declared_sha: str, train_ids, b_ids) -> list:
    """C-TACSRC-LIST: the list parses to 4,512 ids, all in the union train clips, none of the 10,000 (= it equals B_s0's train clip list; union minus list = 10,000), sorted-join sha256 = byte sha256 = the pinned value."""
    bad = []
    sha = hashlib.sha256(raw).hexdigest()
    if sha != declared_sha:
        bad.append(f"G-RM4: the list file hashes {sha[:12]}, declared {str(declared_sha)[:12]}")
    if sha != TAC_SHA:
        bad.append(f"the list hashes {sha[:12]}, the card's {TAC_SHA[:12]}")
    if raw.endswith(b"\n"):
        bad.append("the list has a trailing newline")
    ids = raw.decode("utf-8").split("\n")
    if len(ids) != B_NCLIPS or len(set(ids)) != len(ids):
        bad.append(f"the list holds {len(ids)} ids ({len(set(ids))} distinct), expected {B_NCLIPS}")
    tr = set(train_ids)
    if not set(ids) <= tr:
        bad.append(f"{len(set(ids) - tr)} listed ids are not in the union train clips")
    if set(ids) != set(b_ids):
        bad.append("the list is not B_s0's train clip list")
    if len(tr) - len(set(ids) & tr) != N_TENK:
        bad.append(f"union train clips minus the list = {len(tr) - len(set(ids) & tr)}, expected {N_TENK} (the 10k)")
    if ids != sorted(ids, key=M.sha12):
        bad.append("the list is not in sha12 order")
    if M.train_sha256(ids) != sha:
        bad.append("the sorted-join sha256 of the ids != the byte sha256 of the file")
    return bad


def mix_regression(base_toks, ue_toks, toks, raw: bytes, train_ids, b_ids, ue_steps, arm="UM_s0") -> list:
    """G-RM1..G-RM7 (card section 4): each mutated argv / list must be REFUSED; the unmutated control must read clean. Returns [{"arm", "red", "expect", "problems"}]."""
    def setf(t, f, v):
        t = list(t)
        t[t.index(f) + 1] = v
        return t

    def full(t, r=raw):
        return (argv_diff_problems(base_toks, t, arm, "A", ue_steps) + mix_argv_problems(ue_toks, t)
                + tacsrc_list_problems(r, (parse_argv(t).get("--tac-src-clips-sha256") or [""])[0], train_ids, b_ids))

    ids = raw.decode().split("\n")
    no_list = [x for i, x in enumerate(toks) if not (x in ("--tac-src-clips", "--tac-src-clips-sha256") or (i > 0 and toks[i - 1] in ("--tac-src-clips", "--tac-src-clips-sha256")))]
    arms = [("G-RM1_steps_61569", full(setf(toks, "--steps", "61569"))), (f"G-RM2_seed_{ARMS[arm]['seed'] ^ 1}", full(setf(toks, "--seed", str(ARMS[arm]["seed"] ^ 1)))), ("G-RM3_w_0.5", full(setf(toks, "--tac-src-w", "0.5"))),
            ("G-RM4_list_one_clip_removed", full(toks, "\n".join(ids[1:]).encode())), ("G-RM5_w0_without_list", full(no_list)),
            ("G-RM6_w1_with_list", full(setf(toks, "--tac-src-w", "1.0"))), ("G-RM7_exclude_540_600", full(toks + ["--exclude", "540:600"]))]
    ctl = full(toks)
    res = [{"arm": "G-RM0_control_unmutated", "red": not ctl, "expect": "clean", "problems": list(ctl)[:3]}]
    res += [{"arm": n, "red": bool(p), "expect": "refused", "problems": list(p)[:2]} for n, p in arms]
    return res


# ------------------------------------------------------------------------------------------------ accepted md5 set (MM ruling 1b)
def accepted_md5_set(c_merge_pass: bool = False, union_md5: str | None = None, r_md5: str | None = None) -> dict:
    """md5 -> role. The union md5 enters ONLY when C-MERGE passed (the merged file equals the merge tool's output from the two pinned inputs, corpus side original bytes)."""
    s = {M.CORPUS_V9_MD5: "corpus", M.TENK_V9_MD5: "tenk"}
    if c_merge_pass and union_md5:
        s[union_md5] = "union"
    if c_merge_pass and r_md5:
        s[r_md5] = "R"
    return s


ROLE_OF_SET = {"B": "corpus", "U": "union", "R": "R", "X": "tenk"}


def md5_problems(arm: str, toks, accepted: dict, check_file: bool = True) -> list:
    """C-MD5: ``--v9-train-md5`` in the accepted set, of the role this arm's data set needs, and equal to the md5 of the ``--v9-train`` file (recomputed now)."""
    a = parse_argv(toks)
    bad = []
    v = (a.get("--v9-train-md5") or [""])[0]
    if v not in accepted:
        return [f"--v9-train-md5 {v!r} is not in the accepted set {sorted(accepted)}"]
    want = ROLE_OF_SET[ARMS[arm]["set"]]
    if accepted[v] != want:
        bad.append(f"--v9-train-md5 is the {accepted[v]!r} release; arm {arm} (set {ARMS[arm]['set']}) needs the {want!r} one")
    if check_file:
        p = (a.get("--v9-train") or [""])[0]
        if not os.path.isfile(p):
            bad.append(f"--v9-train {p!r} is not a file")
        elif md5_file(p) != v:
            bad.append(f"--v9-train {p!r} has md5 {md5_file(p)}, not the declared {v}")
    return bad


# ------------------------------------------------------------------------------------------------ counts: C-STEPS, C-NCLIPS, C-CHUNK, C-TRAIN-SHA
def total_steps(toks, rec) -> int:
    a = parse_argv(toks)
    st = int((a.get("--steps") or ["0"])[0])
    if st:
        return st
    return int(rec["S"]) * int((a.get("--steps-mult") or ["1"])[0])


def steps_problems(arm: str, toks, rec, ue_steps: int | None = None, stage: str = "A", u_rec=None) -> list:
    """C-STEPS. Stage A: 19,164 on B / U / R / X; UE and BL run 4 x S_U (and the same number). Stage B: 3,000.
    MM ruling 2026-11-17 (block 3, MM_RULING_BLOCK3_STEPS.md; card lines 75, 87, 195): UE_s0 = 4 x S_U of ITS OWN (union) record; BL_s0 = UE_s0's total, read from the UNION dry record ``u_rec``
    (U_s0's), never from the B record ``rec``. BL without ``u_rec`` is refused. When ``ue_steps`` is given (the evidence's value) it must equal that target."""
    spec, bad = ARMS[arm], []
    t = total_steps(toks, rec)
    if stage == "B":
        return [] if t == STAGE_B_STEPS else [f"stage B runs {t} steps, the card says {STAGE_B_STEPS}"]
    if spec["steps"] == "ue":
        src = rec if spec["set"] == "U" else u_rec
        s_u = None if src is None else src.get("S_U")
        if s_u is None:
            return [f"{arm}: the union S_U is not recorded (UE: the dry build of the union arm; BL: the union dry record u_rec, U_s0's, is required: the B record's S_U is the B set's)"]
        if spec["set"] != "U" and src.get("n_clips") != NCLIPS["U"]:
            return [f"{arm}: the record the target is read from has n_clips {src.get('n_clips')}, not the union's {NCLIPS['U']} (a target from the B record)"]
        target = 4 * int(s_u)
        if t != target:
            bad.append(f"{arm} runs {t} steps, 4 x S_U of the union record = {target}")
        if ue_steps is not None and int(ue_steps) != target:
            bad.append(f"{arm}: the pair's step count {ue_steps} is not 4 x S_U of the union record ({target})")
    elif t != RUN_STEPS_A:
        bad.append(f"{arm} stage A runs {t} steps, the card says {RUN_STEPS_A}")
    return bad


def nclips_problems(arm: str, rec) -> list:
    want = NCLIPS[ARMS[arm]["set"]]
    return [] if rec.get("n_clips") == want else [f"built n_clips {rec.get('n_clips')} != {want} for arm {arm}"]


def chunk_problems(rec) -> list:
    """The sampler chunk the trainer builds is ``max(4, min(48, lru - 2))`` (v8f_mini_train.py:3320): it must be exactly 48."""
    lru = rec.get("lru")
    if lru is None:
        return ["the dry build record has no lru"]
    ch = max(4, min(48, int(lru) - 2))
    bad = [] if ch == CHUNK else [f"sampler chunk {ch} (lru {lru}) != {CHUNK}"]
    if rec.get("sampler_chunk") is not None and rec["sampler_chunk"] != ch:
        bad.append(f"recorded sampler_chunk {rec['sampler_chunk']} != recomputed {ch}")
    return bad


def train_sha_problems(arm: str, rec) -> list:
    """The record's ``train_sha256`` against the digest recomputed from the clip list the build actually used; B arms against the M2b summary's value."""
    bad = []
    clips = rec.get("train_clips")
    if clips is None:
        return ["the dry build record carries no train clip list"]
    if len(clips) != rec.get("n_clips"):
        bad.append(f"the clip list holds {len(clips)} clips, n_clips {rec.get('n_clips')}")
    if M.train_sha256(clips) != rec.get("train_sha256"):
        bad.append("train_sha256 of the record != the digest of its own clip list")
    if ARMS[arm]["set"] == "B" and rec.get("train_sha256") != M.B_TRAIN_SHA256:
        bad.append(f"B set train_sha256 {rec.get('train_sha256')} != the M2b summary's {M.B_TRAIN_SHA256}")
    return bad


# ------------------------------------------------------------------------------------------------ leakage and holdout: C-DISJ (G-R1), C-HOLD
def load_id_list(path) -> list:
    """.txt: one id per line; .json: a list of ids, or a dict whose values are lists (UNVERIFIED shape of the digest json files: flattened)."""
    if str(path).endswith(".json"):
        o = json.load(open(path, encoding="utf-8"))
        if isinstance(o, dict):
            o = [x for v in o.values() if isinstance(v, list) for x in v]
        return [str(x) for x in o]
    return [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]


def disjoint_problems(train_ids, eval_ids, eval_sha12=()) -> list:
    """C-DISJ: ``assert_disjoint(U set, eval-141)`` by clip id AND by stable sid, and by sha12 against the digest lists. Expected 0 and 0 and 0."""
    tr, ev = set(train_ids), set(eval_ids)
    bad = []
    if tr & ev:
        bad.append(f"G-R1: {len(tr & ev)} eval clip ids are in the train set (first: {sorted(tr & ev)[:3]})")
    ts = {M.stable_episode_id(c) for c in tr}
    es = {M.stable_episode_id(c) for c in ev}
    if ts & es and not (tr & ev):
        bad.append(f"{len(ts & es)} eval stable sids are in the train set")
    t12 = {M.sha12(c) for c in tr}
    e12 = set(eval_sha12) | {M.sha12(c) for c in ev}
    if t12 & e12 and not (tr & ev):
        bad.append(f"{len(t12 & e12)} sha12 digests of the eval lists are in the train set")
    if not ev:
        bad.append("the eval id list is empty: C-DISJ reads nothing")
    return bad


def holdout_problems(train_ids, holdout) -> list:
    hit = set(train_ids) & set(holdout)
    bad = [f"C-HOLD: {len(hit)} of the {len(holdout)} RA-1 holdout clips are in the train set (first: {sorted(hit)[:3]})"] if hit else []
    if not holdout:
        bad.append("C-HOLD: the holdout list is empty")
    return bad


def sess_report(train_ids, eval_ids, session_of=None) -> dict:
    """C-SESS (report). ``session_of``: clip id -> session / drive key, or None when no such key exists ("not computable")."""
    if session_of is None:
        return {"computable": False, "note": "no session / drive key in clip_index.parquet or the clip metadata was supplied: the card keeps the residual risk in words"}
    es = {session_of[c] for c in eval_ids if c in session_of}
    hit = sorted(c for c in train_ids if session_of.get(c) in es)
    return {"computable": True, "n_shared": len(hit), "clips": hit[:50]}


# ------------------------------------------------------------------------------------------------ C-MASK (G-R2)
def vlm_mask_from_manifest(man: dict, exact: bool = False) -> int:
    """Bit mask over ``goal_w`` of the VLM-provenance goal classes. ``exact``: provenance exactly ``vlm`` (excludes ``sam3|vlm``). UNVERIFIED which the MM wants: default includes both."""
    prov = man["vocab"]["GOAL_PROV"]
    m = 0
    for i, g in enumerate(GOAL22):
        p = prov.get(g, "")
        if (p == "vlm") if exact else ("vlm" in p.split("|")):
            m |= 1 << i
    return m


def mask_problems(goal_w, is_tenk_row, vlm_mask: int) -> list:
    """C-MASK on any per-row ``goal_w`` array: 10k rows with a VLM-class bit of weight 1 must be exactly 0; the corpus control must be > 0."""
    gw = np.asarray(goal_w).astype(np.uint64)
    t = np.asarray(is_tenk_row, bool)
    hit = (gw & np.uint64(vlm_mask)) != 0
    n10, nc = int(hit[t].sum()), int(hit[~t].sum())
    bad = []
    if n10:
        per = {GOAL22[i]: int((((gw[t] >> np.uint64(i)) & np.uint64(1)) != 0).sum()) for i in range(22) if (vlm_mask >> i) & 1}
        bad.append(f"G-R2: {n10} 10k rows carry a VLM-class goal_w bit (imputed negatives); per bit {per}")
    if nc <= 0:
        bad.append("C-MASK control: no corpus row carries a VLM-class goal_w bit (the probe reads nothing)")
    return bad


def mask_problems_release(union_npz, tenk_npz, manifest_json=None, exact: bool = False) -> list:
    U = M.load_npz(union_npz)
    T = M.load_npz(tenk_npz)
    man = json.load(open(manifest_json or (os.path.splitext(str(union_npz))[0] + ".manifest.json"), encoding="utf-8"))
    tsid = set(T["clip__sid"].tolist())
    is_t = np.zeros(len(U["row__k"]), bool)
    for sid, r0, n in zip(U["clip__sid"].tolist(), U["clip__row0"].tolist(), U["clip__n_rows"].tolist()):
        if sid in tsid:
            is_t[r0:r0 + n] = True
    return mask_problems(U["row__goal_w"], is_t, vlm_mask_from_manifest(man, exact))


# ------------------------------------------------------------------------------------------------ C-JOIN-EQ, coverage report
JOIN_SKIP = ("clip", "off", "n")


def join_eq_problems(Ju: dict, Jc: dict, skip=()) -> list:
    """C-JOIN-EQ: every array of the union join restricted to the corpus clips equals the corpus join's (bit-equal; reports the differing element count)."""
    bad = []
    iu = {str(c): i for i, c in enumerate(Ju["clip"])}
    nu, nc = int(np.sum(Ju["n"])), int(np.sum(Jc["n"]))
    skip = {str(x) for x in skip}                       # pod-side patch 2026-11-16: the 60 holdout clips are in the corpus join and absent from the union BY DESIGN (card 2.2)
    miss = [str(c) for c in Jc["clip"] if str(c) not in iu and str(c) not in skip]
    if miss:
        return [f"{len(miss)} corpus-join clips are not in the union join (first: {miss[:3]})"]
    for k, vc in Jc.items():
        if k in JOIN_SKIP or not isinstance(vc, np.ndarray) or vc.ndim == 0:           # 0-d entries (n_clips_missing_labels) are per-file counts, not per-clip data
            continue
        if k not in Ju:
            bad.append(f"union join has no array {k}")
            continue
        vu = Ju[k]
        n_diff = 0
        for ci, c in enumerate(Jc["clip"]):
            if str(c) in skip:
                continue
            j = iu[str(c)]
            if vc.shape[0] == nc and vc.shape[0] != len(Jc["clip"]):
                a = vc[int(Jc["off"][ci]):int(Jc["off"][ci]) + int(Jc["n"][ci])]
                b = vu[int(Ju["off"][j]):int(Ju["off"][j]) + int(Ju["n"][j])]
            else:
                a, b = vc[ci:ci + 1], vu[j:j + 1]
            if a.shape != b.shape or a.dtype != b.dtype:
                n_diff += max(a.size, b.size)
            elif a.tobytes() != b.tobytes():
                with np.errstate(invalid="ignore"):
                    ne = (a != b) & ~(np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else (a != b)
                n_diff += int(ne.sum()) or 1
        if n_diff:
            bad.append(f"C-JOIN-EQ: array {k}: {n_diff} differing elements on the corpus clips")
    return bad


def coverage_report(J: dict, tenk_ids) -> dict:
    """C-NAVCOV / C-LEADCOV (report only): per set the has-row fraction of steps, the clips with no lead data, the lead_valid fraction."""
    tk = set(tenk_ids)
    isT = np.array([str(c) in tk for c in J["clip"]])
    out = {}
    for nm, sel in (("corpus", ~isT), ("tenk", isT)):
        steps = has = lv = 0
        no_lead_clips = 0
        for ci in np.where(sel)[0]:
            o, n = int(J["off"][ci]), int(J["n"][ci])
            steps += n
            has += int(J["has"][o:o + n].sum())
            lv += int((J["lead_valid"][o:o + n] == 1).sum())
            no_lead_clips += int(not np.isfinite(J["lead_gap"][o:o + n]).any())
        out[nm] = {"n_clips": int(sel.sum()), "steps": steps, "nav_has_row_frac": has / max(steps, 1), "lead_valid_frac": lv / max(steps, 1), "clips_without_lead_gap": no_lead_clips}
    return out


# ------------------------------------------------------------------------------------------------ G-CLOCK, G-HYG, reach table, C-INIT, C-CACHE
def clock_problems(res_tenk, res_corpus, margin: float = 0.0) -> list:
    """G-CLOCK: the per-clip residual of v9 ``now_s`` against the cache frame time: the 10k p99 must lie within the corpus p99. Regression arm: a +0.2 s shift of the 10k clock."""
    a, b = np.abs(np.asarray(res_tenk, float)), np.abs(np.asarray(res_corpus, float))
    if not len(a) or not len(b):
        return ["G-CLOCK: an empty residual list"]
    p10k, pc = float(np.percentile(a, 99)), float(np.percentile(b, 99))
    return [] if p10k <= pc + margin else [f"G-CLOCK: 10k p99 residual {p10k:.4g} s exceeds the corpus p99 {pc:.4g} s"]


def hyg_problems(rec_arm: dict, rec_base: dict) -> list:
    """G-HYG addition: the arm's built cfg and trainable parameter count equal B_s0's (M2b: 7,214,495)."""
    bad = []
    if rec_base.get("n_trainable") != BASE_PARAMS:
        bad.append(f"B_s0 reads {rec_base.get('n_trainable')} trainable parameters, the card says {BASE_PARAMS}")
    if rec_arm.get("n_trainable") != rec_base.get("n_trainable"):
        bad.append(f"trainable parameters {rec_arm.get('n_trainable')} != B_s0's {rec_base.get('n_trainable')}")
    if rec_arm.get("cfg") != rec_base.get("cfg"):
        bad.append("the built cfg differs from B_s0's")
    return bad


STAGE_A_TERMS = ("ckpt", "tac_lat", "tac_lon", "goal_geom", "goal_vlm", "cons_lat", "cons_lon", "cons_lc", "traj", "pen", "pen_self", "wm")
STAGE_B_TERMS = ("acc", "dagger")
MASKED_ZERO = {"goal_vlm": "grad", "cons_lc": "value"}              # exactly 0.0 on a 10k-only batch (card 2.4 table)


def reach_problems(rec: dict, stage_b: bool = False, kinds: tuple = ("corpus", "tenk", "mixed")) -> list:
    """G-LIVE reach table (card 2.4). ``rec = {batch_kind: {term: {"finite": bool, "value": float, "grad": float}}, "control": {"goal_vlm_grad": float}}`` for the batch kinds
    ``corpus``, ``tenk``, ``mixed``. Expected: every term finite on every kind; every term live (grad > 0) on ``tenk`` except ``goal_vlm`` (grad exactly 0.0) and ``cons_lc``
    (contribution exactly 0.0); a term live on mixed batches and dead on 10k-only batches outside those two is a blocker; the corpus control (a TL-positive batch) reads > 0."""
    terms = STAGE_A_TERMS + (STAGE_B_TERMS if stage_b else ())
    bad = []
    for kind in kinds:                                                        # a B-set arm has no 10k windows: kinds = ("corpus",) (gate patch 2026-11-17)
        k = rec.get(kind)
        if not k:
            bad.append(f"reach table: no {kind} batch record")
            continue
        for t in terms:
            r = k.get(t)
            if r is None:
                bad.append(f"reach table: {kind}: term {t} is absent")
                continue
            if not r.get("finite", False):
                bad.append(f"reach table: {kind}: term {t} is not finite")
            if kind == "tenk" and t in MASKED_ZERO:
                key = MASKED_ZERO[t]
                if r.get(key) != 0.0:
                    bad.append(f"reach table: tenk: masked term {t} reads {key} {r.get(key)}, expected exactly 0.0")
            elif t in ("goal_vlm", "cons_lc") and kind != "tenk":
                continue                                                      # live only where a batch holds a positive: the control below carries that claim
            elif not (r.get("grad", 0.0) > 0.0):
                bad.append(f"reach table: {kind}: term {t} is dead (grad {r.get('grad')})")
    c = (rec.get("control") or {}).get("goal_vlm_grad")
    if not (c is not None and c > 0.0):
        bad.append(f"reach table: the corpus control (a TL-positive batch) reads goal_vlm grad {c}, expected > 0")
    return bad


T_TERMS = ("tac_lat", "tac_lon", "goal_geom", "goal_vlm", "cons_lat", "cons_lon", "cons_lc")
NON_T_TERMS = ("ckpt", "traj", "pen", "pen_self", "wm", "acc")


def reach_um_problems(rec: dict) -> list:
    """G-LIVE reach table of UM_s0 at lambda 0 (card section 7), measured FRESH on the UM tree: kinds corpus / tenk / mixed + the lambda-1 control on the tenk batch.
    tenk (lambda 0): every T term reads value and grad EXACTLY 0.0, every non-T term stays live. mixed: T terms live (the corpus rows carry them). corpus: as the parent card's table.
    control: the same tenk batch at lambda 1 reproduces ``tenk_ref`` (probe2/UE_s0.A.json) to rtol 1e-3 on the T grads."""
    bad = reach_problems({"corpus": rec.get("corpus"), "control": rec.get("control")}, False, ("corpus",)) if rec.get("corpus") else ["reach table: no corpus record"]
    for kind in ("tenk", "mixed"):
        k = rec.get(kind)
        if not k:
            bad.append(f"reach um: no {kind} record")
            continue
        for t in T_TERMS + NON_T_TERMS:
            r = k.get(t)
            if r is None:
                if t in ("wm", "pen", "pen_self"):
                    continue
                bad.append(f"reach um: {kind}: term {t} absent")
                continue
            if not r.get("finite", False):
                bad.append(f"reach um: {kind}: {t} not finite")
            if kind == "tenk" and t in T_TERMS:
                if r.get("value") != 0.0 or r.get("grad") != 0.0:
                    bad.append(f"reach um: tenk at lambda 0: T term {t} reads value {r.get('value')} grad {r.get('grad')}, expected exactly 0.0 / 0.0")
            elif t in NON_T_TERMS or (kind == "mixed" and t not in ("goal_vlm", "cons_lc")):
                if not (r.get("grad", 0.0) > 0.0):
                    bad.append(f"reach um: {kind}: term {t} is dead (grad {r.get('grad')})")
    w1, ref = rec.get("tenk_w1"), rec.get("tenk_ref")
    if not w1 or not ref:
        bad.append("reach um: the lambda-1 control or its reference is missing")
    else:
        for t in ("tac_lat", "tac_lon", "goal_geom", "cons_lat", "cons_lon"):
            a, b = (w1.get(t) or {}).get("grad"), (ref.get(t) or {}).get("grad")
            if a is None or b is None or not (b > 0.0) or abs(a - b) > 1e-3 * abs(b):
                bad.append(f"reach um: lambda-1 control {t} grad {a} does not reproduce the reference {b}")
        for t in ("goal_vlm", "cons_lc"):
            if (w1.get(t) or {}).get("grad") != 0.0:
                bad.append(f"reach um: lambda-1 control {t} grad reads {(w1.get(t) or {}).get('grad')}, expected 0.0 on 10k rows")
    return bad


def state_dict_sha256(sd: dict) -> str:
    """C-INIT: a step-0 state dict's identity (sorted keys; tensor bytes, dtype and shape)."""
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        arr = v.detach().cpu().contiguous().view(-1).numpy() if hasattr(v, "detach") else np.ascontiguousarray(v)
        h.update(k.encode("utf-8") + str(tuple(getattr(v, "shape", ()))).encode() + str(getattr(v, "dtype", "")).encode() + arr.tobytes())
    return h.hexdigest()


def cache_problems(expected: dict, actual: dict) -> list:
    """C-CACHE: ``{relative .pt path: sha256}`` of the manifest against the ones recomputed (pod-side walk, not run here)."""
    bad = [f"cache file {k}: sha256 differs" for k in expected if k in actual and expected[k] != actual[k]]
    bad += [f"cache file {k}: absent" for k in expected if k not in actual]
    return bad[:50]


# ------------------------------------------------------------------------------------------------ C-MERGE (rebuild + verify)
def c_merge_problems(rec_path: str) -> tuple:
    """Re-runs the verification of ``merge_record.json``'s outputs against its inputs (hashes recomputed), then re-runs the merge into a scratch folder and compares every
    output's sha256 with the recorded (determinism: the merged file equals the merge tool's output from the two pinned inputs). Returns (problems, union md5)."""
    rec = json.load(open(rec_path, encoding="utf-8"))
    d = os.path.dirname(os.path.abspath(rec_path))
    inp = {k: v["path"] for k, v in rec["inputs"].items()}
    bad = []
    for k, v in rec["inputs"].items():
        if not os.path.isfile(v["path"]) or sha_file(v["path"]) != v["sha256"]:
            bad.append(f"input {k} differs from the recorded sha256")
    for k, v in rec["outputs"].items():
        if not os.path.isfile(os.path.join(d, k)) or sha_file(os.path.join(d, k)) != v["sha256"]:
            bad.append(f"output {k} differs from the recorded sha256")
    if inp["v9_corpus"] and md5_file(inp["v9_corpus"]) != M.CORPUS_V9_MD5 or md5_file(inp["v9_tenk"]) != M.TENK_V9_MD5:
        bad.append("a v9 input md5 is not the pinned one")
    if bad:
        return bad, None
    ids = M.holdout_ids(inp["corpus_manifest"], *rec["holdout"]["range"])
    o = lambda n: os.path.join(d, n)
    bad += M.verify_v9(inp["v9_corpus"], inp["v9_tenk"], o("v9_labels_union.npz"), ids)
    bad += M.verify_lead(inp["lead_corpus"], inp["lead_tenk"], o("lead_union.npz"), ids)
    bad += M.verify_speed(inp["speed_corpus"], inp["speed_tenk"], o("speed_union.jsonl"), ids)
    bad += M.verify_tracks(inp["tracks_corpus"], inp["tracks_tenk"], o("native_tracks_union.pt"), ids)
    scratch = tempfile.mkdtemp(prefix="ds10k_rebuild_", dir=os.environ.get("DS10K_TMP") or None)
    try:
        s = lambda n: os.path.join(scratch, n)
        M.merge_manifests(inp["corpus_manifest"], inp["tenk_manifest"], ids, s("manifest.jsonl"))
        M.merge_v9(inp["v9_corpus"], inp["v9_tenk"], ids, s("v9_labels_union.npz"))
        M.merge_lead(inp["lead_corpus"], inp["lead_tenk"], ids, s("lead_union.npz"))
        M.merge_speed(inp["speed_corpus"], inp["speed_tenk"], ids, s("speed_union.jsonl"))
        M.merge_tracks(inp["tracks_corpus"], inp["tracks_tenk"], ids, s("native_tracks_union.pt"))
        for n in ("manifest.jsonl", "v9_labels_union.npz", "v9_labels_union.manifest.json", "lead_union.npz", "speed_union.jsonl", "native_tracks_union.pt"):
            if sha_file(s(n)) != sha_file(o(n)):
                bad.append(f"rebuild: {n} differs from the merge tool's output of the pinned inputs")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)                                  # own temporary data
    return bad, md5_file(o("v9_labels_union.npz"))


# ------------------------------------------------------------------------------------------------ the dry build (pod-side: needs the trainer, torch and the real cache)
def dry_build(toks, root: str) -> dict:
    """Run the trainer's own ``build_data`` on the argv (no model, no step) and record what the gate needs. ``root`` holds ``stack/`` (the exported code of record).
    Writes the arm's ``--join-train`` file (the trainer's rule with ``--fit-join``). On the pod, UNION arms only after the union view exists: NOT YET RUN on real data."""
    for p in (os.path.join(root, "stack"), os.path.join(root, "stack", "scripts"), root):
        if p not in sys.path:
            sys.path.insert(0, p)
    import v8f_mini_train as T
    a = T.build_parser().parse_args([str(t) for t in toks])
    os.makedirs(a.out, exist_ok=True)
    cfg = T.make_config(a)
    store, J, wins, clips, info = T.build_data(a, cfg)
    S = math.ceil(info["n_windows_all_clips"] / a.batch)
    lru = int(info["lru"])
    return {"n_clips": int(info["n_clips"]), "n_clips_all": int(info["n_clips_all"]), "n_windows": int(info["n_windows"]), "n_windows_all_clips": int(info["n_windows_all_clips"]),
            "train_sha256": info["train_sha256"], "train_clips": sorted(clips, key=M.sha12), "lru": lru, "sampler_chunk": max(4, min(48, lru - 2)), "S": S, "S_U": S,
            "steps_mult": int(a.steps_mult), "batch": int(a.batch), "total_steps": int(a.steps or S * a.steps_mult), "v9_md5": info.get("v9_md5"),
            "n_excluded": int(info.get("n_excluded", 0)), "excluded_in_train": int(info.get("excluded_in_train", 0)), "argv_sha256": argv_sha(toks)}


# ------------------------------------------------------------------------------------------------ stage B: the bank (C-BANK) and the trainer's own stage-B G-LIVE (G-DAGGER)  [gate patch 2026-11-17]
BANK_SHARDS = 4                       # MM ruling 2026-11-17: every bank is built as the recipe of record builds banks: 4 shards, the 4 processes at the same time, then merge


def bank_problems(bank_dir, init_ckpt, dry) -> list:
    """C-BANK: the stage-B bank is complete (not PARTIAL), has n_shards == 4 with every shard file re-hashed, covers the arm's TRAIN list (clips and windows of the dry build), carries
    PASSing K-B4 / K-B5, zero K-B0 / K-B1 mismatches in every shard, an eval-141 overlap of 0, coverage >= 0.5, and was generated from this arm's own --init-ckpt (K-B2). The trainer re-checks
    K-B2 / K-B3 / COVERAGE at launch; this is the checker's own read of the same file, so a bank that the trainer would only refuse after the stage-B build is refused at the token."""
    bad = []
    bj, pj = os.path.join(bank_dir or "", "bank.json"), os.path.join(bank_dir or "", "PROVENANCE.json")
    if not (os.path.isfile(bj) and os.path.isfile(pj)):
        return [f"no bank.json / PROVENANCE.json under {bank_dir!r}: the bank was not built (or a control failed at the merge)"]
    b, p = json.load(open(bj, encoding="utf-8")), json.load(open(pj, encoding="utf-8"))
    if b.get("n_shards") != BANK_SHARDS:
        bad.append(f"bank n_shards {b.get('n_shards')} != {BANK_SHARDS} (the recipe of record's bank build)")
    if b.get("partial") or (b.get("common") or {}).get("partial"):
        bad.append("the bank is PARTIAL (--limit-clips): it may not train an arm")
    com = b.get("common") or {}
    tr = com.get("train") or {}
    if dry:
        if tr.get("n_clips") != dry.get("n_clips") or tr.get("n_windows") != dry.get("n_windows"):
            bad.append(f"bank TRAIN list {tr.get('n_clips')} clips / {tr.get('n_windows')} windows != the dry build's {dry.get('n_clips')} / {dry.get('n_windows')}")
        if tr.get("train_sha256") != dry.get("train_sha256"):
            bad.append("bank train_sha256 differs from the dry build's")
    else:
        bad.append("no dry record to compare the bank's TRAIN list with")
    if (com.get("eval") or {}).get("overlap") != 0 or (com.get("eval") or {}).get("n_clips") != 141:
        bad.append(f"bank eval-141 overlap {(com.get('eval') or {}).get('overlap')}, n_clips {(com.get('eval') or {}).get('n_clips')}")
    cov = b.get("coverage") or {}
    if cov.get("blocked") or not (cov.get("fraction", 0.0) >= 0.5):
        bad.append(f"bank COVERAGE {cov.get('fraction')} < 0.5 (BLOCKED)")
    if init_ckpt and os.path.isfile(init_ckpt) and com.get("ckpt_sha256") != sha_file(init_ckpt):
        bad.append("K-B2: the bank's generating checkpoint is not this arm's --init-ckpt")
    for sh in b.get("shards") or []:
        f = os.path.join(bank_dir, sh.get("file", ""))
        if not os.path.isfile(f) or sha_file(f) != sh.get("file_sha256"):
            bad.append(f"shard {sh.get('k')}: file absent or its sha256 differs from bank.json")
    if len(b.get("shards") or []) != BANK_SHARDS:
        bad.append(f"bank lists {len(b.get('shards') or [])} shards")
    ctl = p.get("controls") or {}
    for k in ("K-B4", "K-B5"):
        if not (ctl.get(k) or {}).get("PASS"):
            bad.append(f"control {k} does not read PASS in PROVENANCE.json")
    for i in range(BANK_SHARDS):
        sp = os.path.join(bank_dir, f"shard_{i:03d}of{BANK_SHARDS:03d}.provenance.json")
        if not os.path.isfile(sp):
            bad.append(f"shard {i}: no provenance file")
            continue
        s = json.load(open(sp, encoding="utf-8"))
        if (s.get("kb0_tick0_identity") or {}).get("mismatch") != 0:
            bad.append(f"shard {i}: K-B0 mismatches {(s.get('kb0_tick0_identity') or {}).get('mismatch')}")
        if (s.get("kb1_replay") or {}).get("mismatch_chunks") != 0:
            bad.append(f"shard {i}: K-B1 mismatched chunks {(s.get('kb1_replay') or {}).get('mismatch_chunks')}")
    if p.get("bank_sha256") != sha_file(bj):
        bad.append("PROVENANCE.json bank_sha256 != the sha256 of bank.json")
    return bad


def dagger_problems(sc: dict, toks) -> list:
    """G-DAGGER: the trainer's own ``--selfcheck-only`` record of the STAGE-B argv: declared_vs_built PASS; ``acc_probe`` PASS with the declared w_acc and a non-zero head_o gradient; ``dagger_probe`` active
    and PASS with a non-zero loss / gradient difference and every control EXACTLY 0.0 (the p = 0 identity, the repeat, the frozen prior, the world-model auxiliary); the bank controls it read (K-B2 / K-B3 /
    coverage) present and not blocked. Deliberate regression: a swap whose ``apply`` is a no-op makes the trainer's probe raise (the selfcheck then holds no passing record) -> RED here."""
    bad = []
    if sc.get("declared_vs_built") != "PASS":
        bad.append(f"declared_vs_built reads {sc.get('declared_vs_built')!r}")
    ap, dp = sc.get("acc_probe") or {}, sc.get("dagger_probe") or {}
    if not ap.get("pass"):
        bad.append("acc_probe does not read pass")
    elif not (ap.get("head_o_grad_diff", 0.0) > 0.0 and abs(float(ap.get("w_acc_declared", -1)) - 49.6969) < 1e-9):
        bad.append(f"acc_probe: w_acc {ap.get('w_acc_declared')}, head_o_grad_diff {ap.get('head_o_grad_diff')}")
    if dp.get("dagger") == "off" or not dp.get("pass") or "skipped" in dp:
        bad.append(f"dagger_probe is not active and passing: {str(dp)[:200]}")
    else:
        if not (dp.get("loss_diff", 0.0) != 0.0 and dp.get("grad_diff_all", 0.0) > 0.0):
            bad.append(f"dagger_probe: loss_diff {dp.get('loss_diff')}, grad_diff_all {dp.get('grad_diff_all')}: the swap does not reach the loss")
        for k in ("control_repeat_loss_diff", "control_repeat_grad_diff", "control_p0_loss_diff", "control_p0_grad_diff", "control_prior_grad_diff"):
            if dp.get(k) != 0.0:
                bad.append(f"dagger_probe control {k} reads {dp.get(k)}, expected exactly 0.0")
        if dp.get("control_wm_grad_diff") not in (0.0, None):
            bad.append(f"dagger_probe control control_wm_grad_diff reads {dp.get('control_wm_grad_diff')}")
        if dp.get("control_p0_same_object") is not True:
            bad.append("dagger_probe: p = 0 did not return the same batch object")
    ctl = (sc.get("info") or {}).get("dagger") or {}
    if not ctl or (ctl.get("coverage") or {}).get("blocked") or not (ctl.get("coverage") or {}).get("fraction", 0) >= 0.5:
        bad.append("the selfcheck carries no bank controls, or the bank coverage is blocked")
    pa = parse_argv(toks).get("--dagger-p")
    try:
        pv = float(pa[0]) if pa else 0.0
    except ValueError:                                                      # ROLLHIST P03 build: a non-numeric dose refuses, it does not crash
        pv = 0.0
    if not pv > 0:
        bad.append("the stage-B argv has no --dagger-p > 0")
    return bad


# ------------------------------------------------------------------------------------------------ ROLLHIST stage B (RH / B0): argv, bank generator record, pair, selfcheck levers  [B3 gate build 2026-11-20]
def readout_token(readout, acc_gain=None, path_exec=False) -> str:
    """Byte-for-byte the spelling of ``tanitad.data.v8f_dagger.readout_token`` at the port commit (cdbff971 + lever C; byte-identical blob to 2bda9e1a) (the ONE spelling; a test compares the two over a grid)."""
    if readout == "pinned":
        if acc_gain is not None or path_exec:
            raise ValueError("readout pinned takes no gain and no path-exec")
        return "pinned"
    if readout != "causal_g":
        raise ValueError(f"unknown readout {readout!r}")
    if acc_gain is None:
        raise ValueError("readout causal_g needs a gain")
    return f"causal_g{float(acc_gain):g}" + ("x" if path_exec else "")


def _norm_path(p) -> str:
    return str(p).replace("\\", "/")


def rh_stage_a_ckpt_problems(init: str, seed: int) -> list:
    """K-B2: the arm's ``--init-ckpt`` is ITS OWN stage-A model ``.../runs/UE_s<seed>/A/ckpt_last.pt`` (RH and B0 of one seed start from the same UE_s<seed>.A checkpoint)."""
    want = f"/runs/UE_s{seed}/A/ckpt_last.pt"
    return [] if _norm_path(init).endswith(want) else [f"K-B2: --init-ckpt {init!r} is not this arm's stage A (a path ending {want})"]


def rh_argv_problems(base_toks, tmpl_toks, toks, arm: str, token: str | None = None) -> list:
    """C-ARGV of an RH / B0 arm. (1) the arm's ``U_s<seed>.B.tmpl.json`` is itself an accepted stage-B argv of U_s<seed> against the B_s0 stage-B template (the existing ``argv_diff_problems``);
    (2) ``toks`` equals the template EXCEPT exactly, RH: the two declarations (added; token in RH_TOKENS, kappa 0.2), the own ``--out`` (names the arm), the own ``--init-ckpt`` (UE_s<seed>.A), the bank and
    its sha256 (filled: no ``@`` placeholder, sha 64 hex), ``--dagger-p`` and ``--dagger-seed`` as the template; B0: ``--dagger-p 0``, no bank flag, no declaration, ``--dagger-seed`` absent or the default 0,
    the own ``--out`` / ``--init-ckpt``. A fifth changed, added or removed flag is refused."""
    if arm not in RH_ARMS:
        return [f"{arm!r} is not an RH / B0 arm"]
    kind, seed = RH_ARMS[arm]
    bad = [f"template: {x}" for x in argv_diff_problems(base_toks, tmpl_toks, f"U_s{seed}", "B")]
    try:
        t, a = parse_argv(tmpl_toks), parse_argv(toks)
    except ValueError as e:
        return bad + [f"argv does not parse: {e}"]
    dag_bank = {"--dagger-bank", "--dagger-bank-sha256"}
    added, removed = set(a) - set(t), set(t) - set(a)
    if kind in RH_LIKE:
        exp_added, exp_removed, own = set(RH_DECL), set(), {"--out", "--init-ckpt"} | dag_bank | ({"--dagger-p"} if kind == "RHp3" else set())
    else:
        exp_added, exp_removed, own = set(), dag_bank | {"--dagger-seed"}, {"--out", "--init-ckpt", "--dagger-p"}
    for f in sorted(added - exp_added):
        bad.append(f"flag {f} {a[f]} is not in the template and is not an allowed addition for {kind}")
    for f in sorted(exp_added - added):
        bad.append(f"{kind} argv does not name {f}")
    for f in sorted(removed - exp_removed):
        bad.append(f"flag {f} {t[f]} of the template is missing from the {kind} argv")
    for f in sorted(set(a) & set(t)):
        if a[f] != t[f] and f not in own:
            bad.append(f"flag {f}: {a[f]} differs from the template's {t[f]} (not an allowed difference for {kind})")
    if "--out" in a:
        if a["--out"] == t.get("--out") or f"/{arm}/" not in _norm_path(" ".join(a["--out"]) + "/"):
            bad.append(f"--out {a['--out']} is not the arm's own (must differ from the template's and name /{arm}/)")
    else:
        bad.append("no --out")
    init = (a.get("--init-ckpt") or [""])[0]
    if not init or init.startswith("@"):
        bad.append(f"--init-ckpt {init!r} is not filled")
    else:
        bad += rh_stage_a_ckpt_problems(init, seed)
    if kind in RH_LIKE:
        for f in ("--dagger-bank", "--dagger-bank-sha256"):
            v = (a.get(f) or [""])[0]
            if not v or v.startswith("@"):
                bad.append(f"{f} {v!r} is not filled")
        sh = (a.get("--dagger-bank-sha256") or [""])[0]
        if sh and not sh.startswith("@") and not (len(sh) == 64 and all(c in "0123456789abcdefABCDEF" for c in sh)):
            bad.append(f"--dagger-bank-sha256 {sh!r} is not 64 hex digits")
        tk = (a.get("--dagger-bank-readout") or [""])[0]
        if tk not in RH_TOKENS:
            bad.append(f"--dagger-bank-readout {tk!r} is not one of {RH_TOKENS}")
        if token is not None and tk != token:
            bad.append(f"--dagger-bank-readout {tk!r} is not the declared token {token!r}")
        if a.get("--dagger-bank-kappa") != [RH_KAPPA]:
            bad.append(f"--dagger-bank-kappa reads {a.get('--dagger-bank-kappa')}, expected [{RH_KAPPA!r}]")
        if kind == "RH":
            if a.get("--dagger-p") != t.get("--dagger-p") or not (a.get("--dagger-p") and float(a["--dagger-p"][0]) > 0):
                bad.append(f"--dagger-p reads {a.get('--dagger-p')}, the template's {t.get('--dagger-p')} (> 0 required)")
        elif a.get("--dagger-p") != [RHP3_P]:                                                    # RHp3: the dose is EXACTLY 0.3, a literal, never "the template's" and never "any p > 0"
            bad.append(f"RHp3 --dagger-p reads {a.get('--dagger-p')}, the card's only dose for this arm is [{RHP3_P!r}]")
    else:
        for f in sorted((dag_bank | set(RH_DECL)) & set(a)):
            bad.append(f"B0 names {f} {a[f]}: a bank flag / declaration at --dagger-p 0 is declared and does nothing (the trainer refuses it)")
        if a.get("--dagger-p") != ["0"]:
            bad.append(f"B0 --dagger-p reads {a.get('--dagger-p')}, expected ['0']")
        if "--dagger-seed" in a and a["--dagger-seed"] != ["0"]:
            bad.append(f"B0 --dagger-seed {a['--dagger-seed']}: a non-default seed at --dagger-p 0 is declared and does nothing (trainer refuses)")
        if token is not None:
            bad.append("B0 takes no readout token (a declaration at --dagger-p 0 is refused by the trainer)")
    return bad


def rh_bank_generator_problems(bank_dir, declared_sha: str, token: str, kappa: str = RH_KAPPA) -> list:
    """The RH bank is the bank the argv declares: ``bank.json``'s sha256 == ``--dagger-bank-sha256``, and its ``generator`` record, read as ``DaggerSwap.built_levers`` reads it (a record that is absent reads
    ``pinned`` / ``none``), equals the declared readout token and kappa; every shard's provenance carries the same record."""
    bj = os.path.join(bank_dir or "", "bank.json")
    if not os.path.isfile(bj):
        return [f"no bank.json under {bank_dir!r}"]
    bad = []
    if sha_file(bj) != declared_sha:
        bad.append(f"bank.json sha256 {sha_file(bj)[:12]} != --dagger-bank-sha256 {str(declared_sha)[:12]}")
    b = json.load(open(bj, encoding="utf-8"))

    def built(gen):
        gen = gen or {}
        try:
            return readout_token(gen.get("readout", "pinned"), gen.get("acc_gain"), bool(gen.get("path_exec", False))), str(gen.get("kappa_max", "none"))
        except ValueError as e:
            return f"<unreadable generator record: {e}>", str(gen.get("kappa_max", "none"))

    gen_b = (b.get("common") or {}).get("generator")         # ROLLHIST Amendment 5: the record the trainer reads (DaggerBank header = bank.json["common"], v8f_dagger.py:366)
    tk, kp = built(gen_b)
    if tk != token:
        bad.append(f"the bank's generator record reads readout {tk!r}, the argv declares {token!r}")
    if kp != kappa:
        bad.append(f"the bank's generator record reads kappa {kp!r}, the argv declares {kappa!r}")
    for sh in b.get("shards") or []:
        sp = os.path.join(bank_dir, f"shard_{int(sh.get('k', 0)):03d}of{int(b.get('n_shards', 0)):03d}.provenance.json")
        if os.path.isfile(sp) and json.load(open(sp, encoding="utf-8")).get("generator") != gen_b:
            bad.append(f"shard {sh.get('k')}: its provenance generator record differs from bank.json's")
    if "generator" in b and b["generator"] != gen_b:
        bad.append("bank.json carries a top-level generator record that differs from common.generator (the trainer reads only common)")
    return bad


def rh_pair_problems(rh_toks, b0_toks, token: str, bank_sha: str | None, pair_check_path: str) -> list:
    """``rollhist_pair_check.pair_problems`` (the committed stack/scripts file, loaded from ``pair_check_path``) on the pair. That file hard-codes ``causal_g1x``: for any other token this REFUSES with a message
    and does NOT patch the committed code in process (the committed-code change, a flag, is the MM's)."""
    if token != RH_PAIR_CHECK_HARDCODED:
        return [f"readout {token!r}: stack/scripts/rollhist_pair_check.py hard-codes {RH_PAIR_CHECK_HARDCODED!r} (RH_READOUT / DECL), so it rejects every other declaration; this checker will not patch it in process. "
                f"It needs a flag (a committed-code change: the MM's) before {token!r} can be minted."]
    if not (pair_check_path and os.path.isfile(pair_check_path)):
        return [f"no rollhist_pair_check.py at {pair_check_path!r}"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("_rh_pair_check_loaded", pair_check_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if m.RH_READOUT != RH_PAIR_CHECK_HARDCODED:
        return [f"the loaded pair check declares {m.RH_READOUT!r}, expected {RH_PAIR_CHECK_HARDCODED!r}"]
    return list(m.pair_problems(rh_toks, b0_toks, bank_sha256=bank_sha))


def _find_key(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                yield v
            yield from _find_key(v, key)
    elif isinstance(o, list):
        for v in o:
            yield from _find_key(v, key)


def rh_levers_problems(sc: dict, kind: str, token: str | None) -> list:
    """The trainer's BUILT levers (selfcheck ``levers``): RH carries the declared readout token and kappa; B0 reads the defaults (``pinned`` / ``None``)."""
    ro = {str(x) for x in _find_key(sc.get("levers", {}), "dagger_bank_readout")}
    kp = {str(x) for x in _find_key(sc.get("levers", {}), "dagger_bank_kappa")}
    bad = []
    if kind in RH_LIKE:
        if token not in ro:
            bad.append(f"the built levers do not carry dagger_bank_readout {token!r} (read {sorted(ro)})")
        if RH_KAPPA not in kp:
            bad.append(f"the built levers do not carry dagger_bank_kappa {RH_KAPPA!r} (read {sorted(kp)})")
    elif ro - {"pinned", "None"}:
        bad.append(f"B0 built levers carry a non-default dagger_bank_readout {sorted(ro)}")
    return bad


def b0_selfcheck_problems(sc: dict, toks) -> list:
    """G-DAGGER-B0: the trainer's own ``--selfcheck-only`` of the B0 argv: declared_vs_built PASS, acc_probe pass with the declared w_acc and a live head_o gradient, default dagger levers, ``--dagger-p`` 0."""
    bad = []
    if sc.get("declared_vs_built") != "PASS":
        bad.append(f"declared_vs_built reads {sc.get('declared_vs_built')!r}")
    ap = sc.get("acc_probe") or {}
    if not ap.get("pass"):
        bad.append("acc_probe does not read pass")
    elif not (ap.get("head_o_grad_diff", 0.0) > 0.0 and abs(float(ap.get("w_acc_declared", -1)) - 49.6969) < 1e-9):
        bad.append(f"acc_probe: w_acc {ap.get('w_acc_declared')}, head_o_grad_diff {ap.get('head_o_grad_diff')}")
    bad += rh_levers_problems(sc, "B0", None)
    if parse_argv(toks).get("--dagger-p") != ["0"]:
        bad.append("the B0 argv does not name --dagger-p 0")
    return bad


# ------------------------------------------------------------------------------------------------ the token
REQUIRED_TRAINER = ("G-HYG", "G-LIVE", "G-CLOCK", "G-EVAL", "G-CKPT", "G-SUITE")
REQUIRED_DATA = ("C-ARGV", "C-MD5", "C-STEPS", "C-NCLIPS", "C-CHUNK", "C-TRAIN-SHA", "C-DISJ", "C-HOLD")
REQUIRED_UNION = ("C-MERGE", "C-JOIN-EQ", "C-MASK")
REQUIRED_MIX = ("C-ARGV-MIX", "C-TACSRC-LIST")        # blockum
REQUIRED_RH_RH = ("C-BANK", "G-DAGGER", "C-RH-PAIR", "C-RH-BANKGEN")           # ROLLHIST RH: the stage-B checks + the pair + the bank's generator record
REQUIRED_RH_B0 = ("C-RH-PAIR", "G-DAGGER-B0")                                   # ROLLHIST B0: no bank, no DAgger probe; the pair + the trainer's own selfcheck of the p = 0 argv
REQUIRED_STAGE_B = ("C-BANK", "G-DAGGER")             # gate patch 2026-11-17: the bank (n_shards 4, complete, controls) and the trainer's own stage-B G-LIVE (acc + dagger probes)
TREE_DIRS = (("stack", "tanitad"), ("stack", "scripts"), ("stack", "tests"), ("taniteval"),)
TREE_EXT = (".py", ".txt", ".json", ".sh")
TRAINER_REL = os.path.join("stack", "scripts", "v8f_mini_train.py")
DATA_BIND_FLAGS = ("--v9-train", "--lead-train", "--speed-train", "--tracks", "--join-train", "--p10", "--init-ckpt", "--tac-src-clips")


def required_checks(arm: str, stage: str = "A") -> tuple:
    if arm in RH_ARMS:
        return REQUIRED_DATA + REQUIRED_UNION + REQUIRED_TRAINER + (REQUIRED_RH_RH if RH_ARMS[arm][0] in RH_LIKE else REQUIRED_RH_B0)
    return REQUIRED_DATA + (REQUIRED_UNION if ARMS[arm]["set"] in ("U", "R", "X") else ()) + REQUIRED_TRAINER + (REQUIRED_STAGE_B if stage == "B" else ()) + (REQUIRED_MIX if arm in MIX_ARMS else ())


def tree_manifest_sha(root) -> str:
    out = {}
    for parts in TREE_DIRS:
        base = os.path.join(root, *([parts] if isinstance(parts, str) else parts))
        for dp, dn, fn in os.walk(base):
            dn[:] = sorted(d for d in dn if d != "__pycache__")
            for f in sorted(fn):
                if f.endswith(TREE_EXT):
                    out[os.path.relpath(os.path.join(dp, f), root).replace(os.sep, "/")] = sha_file(os.path.join(dp, f))
    return hashlib.sha256(json.dumps(out, sort_keys=True).encode("utf-8")).hexdigest()


def data_binding(toks) -> tuple:
    a = parse_argv(toks)
    d, bad = {}, []
    for fl in DATA_BIND_FLAGS:
        v = (a.get(fl) or [""])[0]
        if not v:
            continue
        if not os.path.isfile(v):
            if fl == "--join-train":                       # built by the dry build / --fit-join: bound once it exists
                continue
            bad.append(f"{fl} {v!r} is not a file")
        else:
            d[fl] = {"path": v, "size": os.path.getsize(v), "sha256": sha_file(v), "md5": md5_file(v)}
    c = (a.get("--cache-dir") or [""])[0]
    m = os.path.join(c, "train", "manifest.jsonl") if c else ""
    if m and os.path.isfile(m):
        d["cache/train/manifest.jsonl"] = {"path": m, "size": os.path.getsize(m), "sha256": sha_file(m), "md5": md5_file(m)}
    elif c:
        bad.append(f"the cache has no train/manifest.jsonl ({m})")
    bank = (a.get("--dagger-bank") or [""])[0]
    if bank:
        bj = os.path.join(bank, "bank.json")
        if not os.path.isfile(bj):
            bad.append("the dagger bank has no bank.json")
        else:
            d["bank/bank.json"] = {"path": bj, "sha256": sha_file(bj)}
            if (a.get("--dagger-bank-sha256") or [""])[0] != sha_file(bj):
                bad.append("--dagger-bank-sha256 differs from the sha256 of bank.json")
    return d, bad


def body_sha(tok):
    return hashlib.sha256(json.dumps({k: v for k, v in tok.items() if k != "body_sha256"}, sort_keys=True).encode("utf-8")).hexdigest()


def binding(arm, stage, toks, root, commit, extra_files: dict) -> tuple:
    d, bad = data_binding(toks)
    for k, p in (extra_files or {}).items():
        if p and os.path.isfile(p):
            d[k] = {"path": p, "sha256": sha_file(p), "md5": md5_file(p)}
        elif p:
            bad.append(f"bound file {k} {p!r} is absent")
    tr = os.path.join(root, TRAINER_REL)
    b = {"arm": arm, "stage": stage, "commit": commit, "argv_sha256": argv_sha(toks), "data": d, "tree_manifest_sha256": tree_manifest_sha(root) if os.path.isdir(root) else None,
         "trainer_sha256": sha_file(tr) if os.path.isfile(tr) else None, "checker_sha256": sha_file(os.path.abspath(__file__)), "merge_tool_sha256": sha_file(M.__file__),
         "card_sha256_prefix": CARD_SHA256_PREFIX}
    if b["trainer_sha256"] is None:
        bad.append("trainer not found under --root")
    return b, bad


def write_token(path, b, checks, arm):
    miss = [c for c in required_checks(arm, b.get("stage", "A")) if not (checks.get(c) or {}).get("pass")]
    if miss:
        raise ValueError(f"refusing to write a token: checks not PASS: {miss}")
    tok = {"schema": TOKEN_SCHEMA, "written_by": WRITTEN_BY, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": checks, **b}
    tok["body_sha256"] = body_sha(tok)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".part", "w", encoding="utf-8") as f:
        json.dump(tok, f, indent=1, sort_keys=True)
    os.replace(path + ".part", path)
    return tok


def token_problems(token_path, arm, stage, toks, root, commit, extra_files=None) -> list:
    """[] iff a token written by ``mint`` still binds this arm, stage, commit, argv, every data file, the tree, the trainer, this checker and the merge tool."""
    try:
        tok = json.load(open(token_path, encoding="utf-8"))
    except (OSError, ValueError):
        return ["no gate token"]
    p = []
    if tok.get("schema") != TOKEN_SCHEMA or tok.get("written_by") != WRITTEN_BY:
        p.append("token not written by the checker")
    if tok.get("body_sha256") != body_sha(tok):
        p.append("token body was edited (body_sha256)")
    miss = [c for c in required_checks(arm, stage) if not (tok.get("checks", {}).get(c) or {}).get("pass")]
    if miss:
        p.append(f"token checks not PASS: {miss}")
    want, bad = binding(arm, stage, toks, root, commit, extra_files)
    p += list(bad)
    for k, v in want.items():
        if k == "data":
            for name in sorted(set(v) | set(tok.get("data", {}))):
                if v.get(name) != tok.get("data", {}).get(name):
                    p.append(f"token data {name} differs from the current file")
        elif tok.get(k) != v:
            p.append(f"token {k} differs from the current {k}")
    return p


def mint(a) -> list:
    """Compute every check from the evidence on disk and write the token iff all required ones pass. Returns the problems (empty = token written)."""
    arm, stage, spec = a.arm, a.stage, ARMS[a.arm]
    toks = json.load(open(a.argv, encoding="utf-8"))
    base = json.load(open(a.base_argv, encoding="utf-8"))
    toks, base = (toks["argv"] if isinstance(toks, dict) else toks), (base["argv"] if isinstance(base, dict) else base)
    ev = json.load(open(a.evidence, encoding="utf-8")) if a.evidence else {}
    dry = json.load(open(a.dry, encoding="utf-8")) if a.dry else None
    checks = {}

    def put(name, problems, **kw):
        checks[name] = {"pass": not problems, "problems": list(problems), **kw}

    ue = ev.get("ue_steps")
    kind = RH_ARMS[arm][0] if arm in RH_ARMS else None
    if kind:
        tmpl = json.load(open(a.template, encoding="utf-8")) if getattr(a, "template", "") else None
        tmpl = tmpl["argv"] if isinstance(tmpl, dict) else tmpl
        pair = json.load(open(a.pair_argv, encoding="utf-8")) if getattr(a, "pair_argv", "") else None
        pair = pair["argv"] if isinstance(pair, dict) else pair
        rtok = getattr(a, "readout_token", "") or None
        pre = ([f"{arm} is a stage-B arm: stage {stage} refused"] if stage != "B" else []) + ([] if tmpl else ["no --template (the arm's U_sK.B.tmpl.json)"])
        if kind in RH_LIKE and not rtok:
            pre.append("an RH arm needs --readout-token (causal_g1x | causal_g1): the declaration is a CLI argument of the mint")
        put("C-ARGV", pre + (rh_argv_problems(base, tmpl, toks, arm, rtok) if tmpl else []))
        if pair is None:
            put("C-RH-PAIR", ["no --pair-argv (the other arm's filled argv: RH <-> B0 of the same seed)"])
        else:
            p_tok = rtok if kind in RH_LIKE else ((parse_argv(pair).get("--dagger-bank-readout") or [""])[0])
            rh_t, b0_t = (toks, pair) if kind in RH_LIKE else (pair, toks)
            bsha = (parse_argv(rh_t).get("--dagger-bank-sha256") or [""])[0] or None
            pc = getattr(a, "pair_check", "")
            put("C-RH-PAIR", rh_pair_problems(rh_t, b0_t, p_tok, bsha, pc), readout_token=p_tok, pair_argv_sha256=argv_sha(pair), pair_check_sha256=sha_file(pc) if pc and os.path.isfile(pc) else None)
    else:
        put("C-ARGV", argv_diff_problems(base, toks, arm, stage, ue))
    if arm in MIX_ARMS:
        ue_t = json.load(open(a.ue_argv, encoding="utf-8"))
        ue_t = ue_t["argv"] if isinstance(ue_t, dict) else ue_t
        raw = open(a.tacsrc_list, "rb").read() if a.tacsrc_list and os.path.isfile(a.tacsrc_list) else b""
        bdry = json.load(open(a.b_dry, encoding="utf-8")) if a.b_dry else {}
        udry = json.load(open(a.dry, encoding="utf-8")) if a.dry else {}
        mp = mix_argv_problems(ue_t, toks)
        rg = mix_regression(base, ue_t, toks, raw, udry.get("train_clips") or [], bdry.get("train_clips") or [], ue, arm)
        put("C-ARGV-MIX", mp + [f"deliberate-regression arm {r['arm']} did not read as required" for r in rg if not r["red"]], regression=rg, ue_argv_sha256=argv_sha(ue_t))
        put("C-TACSRC-LIST", tacsrc_list_problems(raw, (parse_argv(toks).get("--tac-src-clips-sha256") or [""])[0], udry.get("train_clips") or [], bdry.get("train_clips") or []),
            list_sha256=hashlib.sha256(raw).hexdigest(), b_dry_sha256=sha_file(a.b_dry) if a.b_dry else None)
    union = spec["set"] in ("U", "R", "X")
    c_merge_ok, union_md5, r_ok = False, None, False
    if union:
        if a.merge_record:
            mb, union_md5 = c_merge_problems(a.merge_record)
            put("C-MERGE", mb, union_md5=union_md5, merge_record_sha256=sha_file(a.merge_record))
            c_merge_ok = not mb
        else:
            put("C-MERGE", ["no --merge-record: C-MERGE not run (pod-side)"])
        if spec["set"] == "R":
            rp = []
            if not (a.r_release and a.union_release and a.tenk_v9 and os.path.isfile(a.r_release)):
                rp.append("R arm: --r-release / --union-release / --tenk-v9 are needed to verify the derangement")
            else:
                rp = ["derangement: " + x for x in M.verify_derange(a.union_release, a.r_release, a.tenk_v9)]
            if rp:
                checks["C-MERGE"]["problems"] += rp
                checks["C-MERGE"]["pass"] = False
            r_ok = not rp and c_merge_ok
    r_md5 = md5_file(a.r_release) if spec["set"] == "R" and a.r_release and os.path.isfile(a.r_release) else None
    acc = accepted_md5_set(c_merge_ok, union_md5, r_md5 if r_ok else None)
    put("C-MD5", md5_problems(arm, toks, acc), accepted_roles=sorted(acc.values()), union_md5=union_md5)
    if dry is None:
        for c in ("C-STEPS", "C-NCLIPS", "C-CHUNK", "C-TRAIN-SHA", "C-DISJ", "C-HOLD"):
            put(c, ["no --dry record: the dry build_data has not run (pod-side)"])
    else:
        ue_rec = json.load(open(a.ue_dry, encoding="utf-8")) if getattr(a, "ue_dry", "") else None
        put("C-STEPS", steps_problems(arm, toks, dry, ue, stage, ue_rec), ue_dry_sha256=sha_file(a.ue_dry) if getattr(a, "ue_dry", "") else None)
        put("C-NCLIPS", nclips_problems(arm, dry))
        put("C-CHUNK", chunk_problems(dry))
        put("C-TRAIN-SHA", train_sha_problems(arm, dry), train_sha256=dry.get("train_sha256"))
        tr = dry.get("train_clips") or []
        if a.eval_ids and a.holdout_manifest:
            ids = M.holdout_ids(a.holdout_manifest, *a.holdout)
            dig = []
            for p in a.digest_json or []:
                dig += [M.sha12(x) if len(x) > 12 else x for x in load_id_list(p)]
            put("C-DISJ", disjoint_problems(tr, load_id_list(a.eval_ids), dig))
            hp = holdout_problems(tr, ids)
            if list(a.holdout) == [M.HOLD_LO, M.HOLD_HI] and M.holdout_digest(ids) != M.EXCLUDED_SHA256:
                hp.append("the reproduced holdout digest != the M2b summary's excluded_sha256 (wrong manifest)")
            put("C-HOLD", hp)
        else:
            put("C-DISJ", ["no --eval-ids / --holdout-manifest (pod-side)"])
            put("C-HOLD", ["no --holdout-manifest (pod-side)"])
    if union:
        if a.union_release and a.tenk_v9 and os.path.isfile(a.union_release):
            rel = a.r_release if spec["set"] == "R" and a.r_release else a.union_release
            put("C-MASK", mask_problems_release(rel, a.tenk_v9))
        else:
            put("C-MASK", ["no --union-release / --tenk-v9"])
        if a.corpus_join and a.union_join and os.path.isfile(a.corpus_join) and os.path.isfile(a.union_join):
            if not os.path.basename(a.corpus_join) or md5_file(a.corpus_join)[:8] != CORPUS_JOIN_MD5_PREFIX:
                put("C-JOIN-EQ", [f"the corpus join md5 {md5_file(a.corpus_join)} does not start with {CORPUS_JOIN_MD5_PREFIX}"])
            else:
                put("C-JOIN-EQ", join_eq_problems(dict(np.load(a.union_join, allow_pickle=True)), dict(np.load(a.corpus_join, allow_pickle=True)),
                                                  skip=M.holdout_ids(a.holdout_manifest, *a.holdout) if a.holdout_manifest else ()))
        else:
            put("C-JOIN-EQ", ["no --corpus-join / --union-join (pod-side: the union join is written by the dry build)"])
    # the trainer-side checks: DECIDED here from the evidence they produce
    hy = ev.get("hyg")
    put("G-HYG", hyg_problems(hy["arm"], hy["base"]) if hy else ["no hyg evidence (pod-side)"])
    kinds = ("corpus",) if spec["set"] == "B" else ("corpus", "tenk", "mixed")
    if arm in MIX_ARMS:
        put("G-LIVE", reach_um_problems(ev["reach"]) if ev.get("reach") else ["no reach-table evidence (pod-side real-data smoke)"])
    else:
        put("G-LIVE", reach_problems(ev["reach"], False, kinds) if ev.get("reach") else ["no reach-table evidence (pod-side real-data smoke)"])
    if stage == "B":
        bank_dir = (parse_argv(toks).get("--dagger-bank") or [""])[0]
        init = (parse_argv(toks).get("--init-ckpt") or [""])[0]
        if kind != "B0":
            put("C-BANK", bank_problems(bank_dir, init, dry), n_shards=BANK_SHARDS)
        sc = json.load(open(a.selfcheck, encoding="utf-8")) if getattr(a, "selfcheck", "") and os.path.isfile(a.selfcheck) else None
        no_sc = ["no --selfcheck: the trainer's --selfcheck-only record of THIS stage-B argv (real bank, real init checkpoint)"]
        if kind == "B0":
            put("G-DAGGER-B0", b0_selfcheck_problems(sc, toks) if sc else no_sc, selfcheck_sha256=sha_file(a.selfcheck) if sc else None)
        else:
            put("G-DAGGER", (dagger_problems(sc, toks) + (rh_levers_problems(sc, "RH", rtok) if kind in RH_LIKE else [])) if sc else no_sc,
                selfcheck_sha256=sha_file(a.selfcheck) if sc else None)
        if kind in RH_LIKE:
            put("C-RH-BANKGEN", rh_bank_generator_problems(bank_dir, (parse_argv(toks).get("--dagger-bank-sha256") or [""])[0], rtok or "", RH_KAPPA), readout_token=rtok, kappa=RH_KAPPA)
    ck = ev.get("clock")
    if not union:
        put("G-CLOCK", [], note="B set: no new label clock (the inherited corpus clock)")
    else:
        put("G-CLOCK", clock_problems(ck["tenk"], ck["corpus"]) if ck else ["no clock evidence (pod-side)"])
    for g in ("G-EVAL", "G-CKPT", "G-SUITE"):
        r = ev.get(g)
        put(g, [] if r and r.get("pass") is True else [f"no passing {g} record (pod-side)"], record_sha256=hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() if r else None)
    if "c_init" in ev:
        checks["C-INIT"] = {"pass": True, "report": ev["c_init"]}
    extra = {"--union-release": a.union_release, "tenk_v9": a.tenk_v9, "r_release": a.r_release, "merge_record": a.merge_record, "dry_record": a.dry, "evidence": a.evidence,
             "selfcheck": getattr(a, "selfcheck", ""), "tacsrc_list": getattr(a, "tacsrc_list", ""), "ue_argv": getattr(a, "ue_argv", ""), "b_dry": getattr(a, "b_dry", ""),
             "template": getattr(a, "template", ""), "pair_argv": getattr(a, "pair_argv", ""), "pair_check": getattr(a, "pair_check", "")}
    b, bad = binding(arm, stage, toks, a.root, a.commit, {k: v for k, v in extra.items() if v})
    fails = [f"check {c} did not pass: {json.dumps(checks.get(c))[:500]}" for c in required_checks(arm, stage) if not checks.get(c, {}).get("pass")]
    os.makedirs(a.gate_dir, exist_ok=True)
    json.dump(checks, open(os.path.join(a.gate_dir, "checks.json"), "w", encoding="utf-8"), indent=1)
    if bad or fails:
        return ["binding: " + x for x in bad] + fails
    write_token(os.path.join(a.gate_dir, "gate_token.json"), b, checks, arm)
    return []


def run_arm(a) -> int:
    """The runner: REFUSE (rc 2) unless the token binds exactly the argv on disk, the commit, every data file and the tree; then start the trainer under the same argv."""
    import subprocess
    toks = json.load(open(a.argv, encoding="utf-8"))
    toks = toks["argv"] if isinstance(toks, dict) else toks
    g = lambda k: getattr(a, k, "") or ""
    extra = {"--union-release": g("union_release"), "tenk_v9": g("tenk_v9"), "r_release": g("r_release"), "merge_record": g("merge_record"), "dry_record": g("dry"), "evidence": g("evidence"),
             "selfcheck": g("selfcheck"), "tacsrc_list": g("tacsrc_list"), "ue_argv": g("ue_argv"), "b_dry": g("b_dry"),
             "template": g("template"), "pair_argv": g("pair_argv"), "pair_check": g("pair_check")}                                               # gate patch 2026-11-17: the same extra files mint bound
    bad = token_problems(os.path.join(a.gate_dir, "gate_token.json"), a.arm, a.stage, toks, a.root, a.commit, {k: v for k, v in extra.items() if v})
    if bad:
        print(f"REFUSED {a.arm}: " + "; ".join(bad))
        return 2
    if a.arm in RH_ARMS:                                                  # ROLLHIST: stage B only, and the declared token is the one the token records
        tk = json.load(open(os.path.join(a.gate_dir, "gate_token.json"), encoding="utf-8"))
        rec = ((tk.get("checks") or {}).get("C-RH-PAIR") or {}).get("readout_token")
        if a.stage != "B":
            print(f"REFUSED {a.arm}: a stage-B arm")
            return 2
        if RH_ARMS[a.arm][0] in RH_LIKE and g("readout_token") != rec:
            print(f"REFUSED {a.arm}: --readout-token {g('readout_token')!r} is not the token's {rec!r}")
            return 2
    print(f"TOKEN-OK {a.arm}")
    env = dict(os.environ)
    env.update({"PYTHONPATH": os.path.join(a.root, "stack") + os.pathsep + a.root, "CUDA_VISIBLE_DEVICES": a.gpu, "HF_HUB_DISABLE_XET": "1"})
    return subprocess.run([a.py, os.path.join(a.root, TRAINER_REL)] + list(toks), cwd=a.root, env=env).returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("diff"); p.add_argument("--arm", required=True, choices=sorted(ARMS)); p.add_argument("--stage", default="A", choices=("A", "B"))
    p.add_argument("--argv", required=True); p.add_argument("--base-argv", required=True); p.add_argument("--ue-steps", type=int)
    p = sub.add_parser("dry"); p.add_argument("--argv", required=True); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    for nm in ("mint", "run"):
        p = sub.add_parser(nm)
        p.add_argument("--arm", required=True, choices=sorted(ARMS)); p.add_argument("--stage", default="A", choices=("A", "B"))
        p.add_argument("--argv", required=True); p.add_argument("--root", required=True); p.add_argument("--commit", required=True); p.add_argument("--gate-dir", required=True)
        p.add_argument("--py", default=sys.executable); p.add_argument("--gpu", default="0")
        if nm == "run":
            p.add_argument("--evidence", default=""); p.add_argument("--dry", default=""); p.add_argument("--merge-record", default=""); p.add_argument("--union-release", default="")
            p.add_argument("--tenk-v9", default=""); p.add_argument("--r-release", default=""); p.add_argument("--selfcheck", default="")
            p.add_argument("--tacsrc-list", default=""); p.add_argument("--ue-argv", default=""); p.add_argument("--b-dry", default="")
            p.add_argument("--template", default=""); p.add_argument("--pair-argv", default=""); p.add_argument("--pair-check", default=""); p.add_argument("--readout-token", default="")
        if nm == "mint":
            p.add_argument("--base-argv", required=True); p.add_argument("--evidence", default=""); p.add_argument("--dry", default="")
            p.add_argument("--merge-record", default=""); p.add_argument("--union-release", default=""); p.add_argument("--tenk-v9", default=""); p.add_argument("--r-release", default="")
            p.add_argument("--corpus-join", default=""); p.add_argument("--union-join", default=""); p.add_argument("--eval-ids", default=""); p.add_argument("--digest-json", nargs="*")
            p.add_argument("--ue-dry", default=""); p.add_argument("--selfcheck", default=""); p.add_argument("--tacsrc-list", default=""); p.add_argument("--ue-argv", default=""); p.add_argument("--b-dry", default=""); p.add_argument("--template", default=""); p.add_argument("--pair-argv", default="")
            p.add_argument("--pair-check", default=""); p.add_argument("--readout-token", default=""); p.add_argument("--holdout-manifest", default=""); p.add_argument("--holdout", type=lambda s: tuple(int(x) for x in s.split(":")), default=(M.HOLD_LO, M.HOLD_HI))
    a = ap.parse_args(argv)
    if a.cmd == "diff":
        ld = lambda p: (lambda o: o["argv"] if isinstance(o, dict) else o)(json.load(open(p, encoding="utf-8")))
        bad = argv_diff_problems(ld(a.base_argv), ld(a.argv), a.arm, a.stage, a.ue_steps)
        print("PASS" if not bad else "REFUSED: " + "; ".join(bad))
        return 0 if not bad else 1
    if a.cmd == "dry":
        toks = json.load(open(a.argv, encoding="utf-8"))
        toks = toks["argv"] if isinstance(toks, dict) else toks
        rec = dry_build(toks, a.root)
        json.dump(rec, open(a.out, "w", encoding="utf-8"))
        print(json.dumps({k: v for k, v in rec.items() if k != "train_clips"}))
        return 0
    if a.cmd == "mint":
        bad = mint(a)
        print("TOKEN-WRITTEN " + a.arm if not bad else "REFUSED: " + "; ".join(bad))
        return 0 if not bad else 1
    return run_arm(a)


if __name__ == "__main__":
    sys.exit(main())
