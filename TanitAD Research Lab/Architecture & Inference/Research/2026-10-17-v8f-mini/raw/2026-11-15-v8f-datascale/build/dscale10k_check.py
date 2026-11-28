"""PREREG_V8F_DSCALE10K study checker (card section 5; MM freeze entry 2026-11-16: NO ``stack/`` change). Mints ``dscale10k-gate-token-v1``.

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
CODE_COMMIT = "cdbff971"
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
    "B_s3": dict(set="B", stage_b=True, seed=3, dagger_seed=3, steps="mult"), "U_s3": dict(set="U", stage_b=True, seed=3, dagger_seed=3, steps="fixed"),
    "B_s4": dict(set="B", stage_b=True, seed=4, dagger_seed=4, steps="mult"), "U_s4": dict(set="U", stage_b=True, seed=4, dagger_seed=4, steps="fixed"),
}
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


def steps_problems(arm: str, toks, rec, ue_steps: int | None = None, stage: str = "A") -> list:
    """C-STEPS. Stage A: 19,164 on B / U / R / X; UE and BL run 4 x S_U (and the same number). Stage B: 3,000."""
    spec, bad = ARMS[arm], []
    t = total_steps(toks, rec)
    if stage == "B":
        return [] if t == STAGE_B_STEPS else [f"stage B runs {t} steps, the card says {STAGE_B_STEPS}"]
    if spec["steps"] == "ue":
        s_u = rec.get("S_U")
        if s_u is None:
            return ["UE / BL: the union S_U is not recorded (dry build of a union arm)"]
        if t != 4 * int(s_u):
            bad.append(f"{arm} runs {t} steps, 4 x S_U = {4 * int(s_u)}")
        if ue_steps is not None and t != int(ue_steps):
            bad.append(f"{arm} runs {t} steps, the UE / BL pair must match ({ue_steps})")
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


def join_eq_problems(Ju: dict, Jc: dict) -> list:
    """C-JOIN-EQ: every array of the union join restricted to the corpus clips equals the corpus join's (bit-equal; reports the differing element count)."""
    bad = []
    iu = {str(c): i for i, c in enumerate(Ju["clip"])}
    nu, nc = int(np.sum(Ju["n"])), int(np.sum(Jc["n"]))
    miss = [str(c) for c in Jc["clip"] if str(c) not in iu]
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


def reach_problems(rec: dict, stage_b: bool = False) -> list:
    """G-LIVE reach table (card 2.4). ``rec = {batch_kind: {term: {"finite": bool, "value": float, "grad": float}}, "control": {"goal_vlm_grad": float}}`` for the batch kinds
    ``corpus``, ``tenk``, ``mixed``. Expected: every term finite on every kind; every term live (grad > 0) on ``tenk`` except ``goal_vlm`` (grad exactly 0.0) and ``cons_lc``
    (contribution exactly 0.0); a term live on mixed batches and dead on 10k-only batches outside those two is a blocker; the corpus control (a TL-positive batch) reads > 0."""
    terms = STAGE_A_TERMS + (STAGE_B_TERMS if stage_b else ())
    bad = []
    for kind in ("corpus", "tenk", "mixed"):
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


# ------------------------------------------------------------------------------------------------ the token
REQUIRED_TRAINER = ("G-HYG", "G-LIVE", "G-CLOCK", "G-EVAL", "G-CKPT", "G-SUITE")
REQUIRED_DATA = ("C-ARGV", "C-MD5", "C-STEPS", "C-NCLIPS", "C-CHUNK", "C-TRAIN-SHA", "C-DISJ", "C-HOLD")
REQUIRED_UNION = ("C-MERGE", "C-JOIN-EQ", "C-MASK")
TREE_DIRS = (("stack", "tanitad"), ("stack", "scripts"), ("stack", "tests"), ("taniteval"),)
TREE_EXT = (".py", ".txt", ".json", ".sh")
TRAINER_REL = os.path.join("stack", "scripts", "v8f_mini_train.py")
DATA_BIND_FLAGS = ("--v9-train", "--lead-train", "--speed-train", "--tracks", "--join-train", "--p10", "--init-ckpt")


def required_checks(arm: str) -> tuple:
    return REQUIRED_DATA + (REQUIRED_UNION if ARMS[arm]["set"] in ("U", "R", "X") else ()) + REQUIRED_TRAINER


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
    miss = [c for c in required_checks(arm) if not (checks.get(c) or {}).get("pass")]
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
    miss = [c for c in required_checks(arm) if not (tok.get("checks", {}).get(c) or {}).get("pass")]
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
    put("C-ARGV", argv_diff_problems(base, toks, arm, stage, ue))
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
        put("C-STEPS", steps_problems(arm, toks, dry, ue, stage))
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
                put("C-JOIN-EQ", join_eq_problems(dict(np.load(a.union_join, allow_pickle=True)), dict(np.load(a.corpus_join, allow_pickle=True))))
        else:
            put("C-JOIN-EQ", ["no --corpus-join / --union-join (pod-side: the union join is written by the dry build)"])
    # the trainer-side checks: DECIDED here from the evidence they produce
    hy = ev.get("hyg")
    put("G-HYG", hyg_problems(hy["arm"], hy["base"]) if hy else ["no hyg evidence (pod-side)"])
    put("G-LIVE", reach_problems(ev["reach"], spec["stage_b"] and stage == "B") if ev.get("reach") else ["no reach-table evidence (pod-side real-data smoke)"])
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
    extra = {"--union-release": a.union_release, "tenk_v9": a.tenk_v9, "r_release": a.r_release, "merge_record": a.merge_record, "dry_record": a.dry, "evidence": a.evidence}
    b, bad = binding(arm, stage, toks, a.root, a.commit, {k: v for k, v in extra.items() if v})
    fails = [f"check {c} did not pass: {json.dumps(checks[c])[:500]}" for c in required_checks(arm) if not checks.get(c, {}).get("pass")]
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
    bad = token_problems(os.path.join(a.gate_dir, "gate_token.json"), a.arm, a.stage, toks, a.root, a.commit)
    if bad:
        print(f"REFUSED {a.arm}: " + "; ".join(bad))
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
        if nm == "mint":
            p.add_argument("--base-argv", required=True); p.add_argument("--evidence", default=""); p.add_argument("--dry", default="")
            p.add_argument("--merge-record", default=""); p.add_argument("--union-release", default=""); p.add_argument("--tenk-v9", default=""); p.add_argument("--r-release", default="")
            p.add_argument("--corpus-join", default=""); p.add_argument("--union-join", default=""); p.add_argument("--eval-ids", default=""); p.add_argument("--digest-json", nargs="*")
            p.add_argument("--holdout-manifest", default=""); p.add_argument("--holdout", type=lambda s: tuple(int(x) for x in s.split(":")), default=(M.HOLD_LO, M.HOLD_HI))
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
