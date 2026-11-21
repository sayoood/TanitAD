#!/usr/bin/env python3
"""S1 end-to-end READ DRIVER of the auxiliary drivable-area head (eval/PREREG_AUX_DRIVABLE_S1.md, IN FORCE with Amendments A1-A6).

It turns BANKED ARM EVIDENCE (private HF repo Sayood/tanitad-refe-s1aux) into the pre-registered S1 read. It is a DRIVER: the estimators and the P1/P2/P3/selection reads are
`eval/aux_s1_readout.py` (tool aux_s1_readout/1, harness commit 20fa3548) and the head outputs are `eval/aux_navtest_forward.py`; this file fetches, verifies, runs those
instruments with the PINNED settings, and applies the prereg's DECISION RULES (K2, K3, K6, the P3 decision incl. the A2 floor, H2 / H1 promotion) to their outputs.

    python s1_read.py read --cache <dir> --out <dir>                    # discover the arms on HF, fetch, run, decide
    python s1_read.py read --arms N-s0 N-s1 N-s2 --cache <dir> --out <dir>
    python s1_read.py read --plumbing-check H2-s0 ...                   # run the head path on those arms as PLUMBING: values are written to raw files, never to S1_READ

RULES THIS FILE ENFORCES (each has a RED mutation in test_s1_read.py)
  * COMPLETENESS (A5: "the first S1 read happens after all registered arms exist"): all NINE arms N/H1/H2 x s0,s1,s2 must be present and clean, else the status is INCOMPLETE,
    every read names the arms it lacks, and NO outcome label (K2_*, H2_*, P3_*, *_PROMOTED) is emitted. An arm whose evidence contradicts its label is REFUSED, never read.
  * FETCH: every file is verified against the listing's size AND sha256 (LFS) / git blob id (non-LFS); a mismatch deletes the file and refuses the arm.
  * PINS: 10,000 resamples, seed 20260927, K2 0.80, P3 0.90 / +0.05, K6 0.9 x step-0 at 10 % of the steps, K3 3 % / 1 GB, rule navsim_v1; asserted against the harness.
  * A2: H2_EVALUATED / H2_PROVISIONAL need the paired (trained - initial head) pooled-AUC 95 % lower bound > 0, else P3_AT_FLOOR (re-derived here, compared with the harness).
  * TIER: every number is T0-like (held-out selection / perception on teacher-forced banked sets), never driving performance.
Every place where the prereg's wording admits two readings is in AMBIGUITIES, written into the output; none is resolved silently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))

TOOL = "s1_read/1"
HARNESS_TOOL = "aux_s1_readout/1"
HARNESS_COMMIT = "20fa3548"
REPO = "Sayood/tanitad-refe-s1aux"
PREREG = "eval/PREREG_AUX_DRIVABLE_S1.md"

# ------------------------------------------------------------------------------------------------------------------ pins (the prereg's numbers, restated)
N_BOOT = 10000                      # S1 prereg section 2 "harness commit" row; draft "Estimator"
SEED = 20260927
K2_IOU_MIN = 0.80                   # draft K2 / P1
P3_FLOOR, P3_MARGIN = 0.90, 0.05    # draft P3
K6_FACTOR, K6_FRAC = 0.9, 0.10      # draft K6: aux_bce >= 0.9 x step-0 after 10 % of the steps
K3_TIME, K3_GB = 0.03, 1.0          # draft K3: step time +3 %, peak memory +1 GB
N_UPDATES = 400                     # proposal section 2 / S1 prereg section 1 (init + 400 updates)
SEEDS = (0, 1, 2)
PAIRS_NEEDED, PAIRS_NEEDED_PROVISIONAL = 2, 3          # S1 prereg 4.3 (2): 2 of 3 seed pairs; 3 of 3 under H2_PROVISIONAL
RULE = "navsim_v1"
LOG_LEVEL = "95 %"
TIER = "T0-like: held-out selection and perception on teacher-forced banked sets (S1_POD_RUNBOOK l.5); never driving performance"
ESTIMATOR = f"paired log-cluster bootstrap, {N_BOOT} resamples, {LOG_LEVEL}, seed {SEED} (aux_s1_readout / proxy_eval.boot's draw scheme)"
QUESTION = "answers 'would another draw of LOGS say this?' ONLY; training-run variance is the N-vs-N seed floor (H-ESTIM-SEED-1); inference variance: REFe scoring is deterministic, ASSERTED not measured"

REGISTERED = tuple(f"{c}-s{s}" for c in ("N", "H1", "H2") for s in SEEDS)
ARM_RE = re.compile(r"^(N|H1|H2)-s([0-2])$")
G12 = "G12_aux_drv_live_and_identity"

# fixed on-disk inputs of the dev box (the same ones the P3-floor run used)
ART = "D:/Projects/TanitAD-artifacts"
DEFAULT_CTX_CACHE = f"{ART}/aux_navtest_forward/c3_full_ctx"
DEFAULT_FLOOR_DIR = f"{ART}/aux_navtest_forward/p3_floor_full/heads"
DEFAULT_HELDOUT_LOGS = f"{ART}/_tlane_work/ho/refe_heldout/heldout_logs.txt"
HELDOUT_LOGS_SHA256 = "b814998deeb40a9d7839cce2807f8f282ede76c7f2c5ccce97adc064b27f4902"      # the PASS token's --heldout-logs binding (N-s0)
DEFAULT_PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"                          # the venv the P3 floors were produced in (CPU, torch 2.7.1)
C3_TABLE, C3_TOKENS = "raw/2026-10-11-c3-selection-headroom/table.npz", "raw/2026-10-11-c3-selection-headroom/tokens.json"
U400_CSV = "raw/p2_r2_instrument/baseline/u400_navtest_full.csv"
FLOOR_SHA_FILE = "raw/2026-10-13-aux-p3-floor/full_1058/head_pred_sha256.txt"

ARM_SMALL = ("arms_manifest.jsonl", "run_check.json", "PREFLIGHT_PASS.json", "sft.jsonl", "token_verify.json", "launch_gate.log")
ARM_DUMP = ("dump/scores_update400_B.npz",)
ARM_HEAD = ("aux_drv.pt", "dump/head_pred_update400.npz", "dump/head_pred_step0.npz")
LABEL_FILES = ("labels/s1_heldout_pdm.tar", "labels/MANIFEST_s1_labels_sha256.txt", "labels/pdm_complete_heldout.json")
NOT_FETCHED = {"model_sft_B.pt": "1.29 GB per arm; no read needs it: selection reads the banked score dumps, perception reads aux_drv.pt on the cached u400 ctx (the trunk is frozen, "
                                "S1 prereg 4.1), and A2.3 fixes the ctx of record to the planner's fp32 ctx",
               "scorer_states_last.pt": "34 MB optimiser/scorer states; no read needs it",
               "dump/scores_update400_A.npz / _base.npz / step0_*": "the prereg reads arm B (the arm the harness docstring names); step0_B is compared by LISTING sha256 only"}

CLAUSES = {
    "completeness": "S1 prereg A5 l.298: 'The first S1 read happens after all registered arms exist'; A3.3 l.233-237: all nine registered arms run, both H1 replicates are reported",
    "seed_floor": "S1 prereg 4.3 l.108 + draft 'Replicates': floor = largest |delta| between two N seeds, measured only with all three N runs; l.126: fewer than three N => capped",
    "sel_pair": "S1 prereg 4.3 l.108: each seed pair Hx-sK minus N-sK on held-out pick PDMS and pick DAC; comparator of record = the SAME-seed N (section 2 l.48)",
    "k2": "draft K2 + P1; S1 prereg 4.1 table l.91-96 (P1 row l.93: navtest P1 on the 1,058 C3 tokens; P1-HO is the fallback) and l.98 (navtest governs)",
    "p3": "S1 prereg 3 R2 (pooled AUC), R3 (H2_PROVISIONAL), A2.1 l.154-156 (paired floor), A2.3 (fp32 planner ctx)",
    "k6": "draft K6: aux_bce >= 0.9 x its step-0 value after 10 % of the arm's steps => stop",
    "k3": "draft K3: measured step time rises > 3 % or peak memory > 1 GB; proposal 2 / S1_POD_RUNBOOK S1-D6 (read afterwards from the arm evidence)",
    "k5": "draft K5: the zero-weight / detached exact controls must be bit-identical (the G12 preflight fields); the shuffle limb cannot fire (no C-shuffle arm, S1 prereg 1 l.39)",
    "h1_id": "S1 prereg 4.3 l.124: H1's scorer is bit-identical to N by construction; any H1-vs-N difference in selection is a defect, investigated before any read",
    "promote_h2": "S1 prereg 4.3 l.110-118: conditions 1-4 (l.111-114), otherwise-not-promoted (l.116), C-shuffle amendment (l.118)",
    "promote_h1": "S1 prereg 4.3 l.120-122",
    "families": "draft 'Four families' + S1 prereg 4.2 l.100-104 (navtest families not computed at S1; STRATEGIC PAUSED, PI 2026-09-19 item 25)",
}

# every place the prereg's words admit two readings. `used` is what the code does; `alt` is what it does NOT do; `sensitivity` = the driver recomputes the outcome under `alt`.
AMBIGUITIES = [
    {"id": "AMB-1", "where": "S1 prereg 4.3 cond 1 (l.111) and cond 2 (l.112)", "text": "'the P3 decision on C3 is H2_EVALUATED or H2_PROVISIONAL' - there are three H2 runs, each with its own P3 decision; the text has one decision",
     "used": "all three H2 seeds must read EVALUATED or PROVISIONAL; the regime is PROVISIONAL (3 of 3 pairs) if ANY seed is PROVISIONAL", "alt": "at least 2 of 3 seeds in {EVALUATED, PROVISIONAL}", "sensitivity": True},
    {"id": "AMB-2", "where": "S1 prereg 4.3 H1 paragraph (l.120-122)", "text": "'K2 does not fire on H1's head' / 'K6 did not fire' - there are three H1 runs",
     "used": "no K2 / K6 on ANY of the three H1 seeds (the phrase 'any H2 run' in cond 4 is the model)", "alt": "H1-s0 only (the S1_POD_RUNBOOK S1-D3 drop rule reads H1-s0)", "sensitivity": True},
    {"id": "AMB-3", "where": "draft K6", "text": "'aux_bce >= 0.9 x its step-0 value after 10 % of the arm's steps' - which logged number is the step-0 value and which row is 'after 10 %'",
     "used": "step-0 value = G12 bce_declared (the head's BCE on a real batch before any update); the row = the first 'train' event with update >= ceil(0.10 x 400) = 40, i.e. update 50 (a 25-update mean)",
     "alt": "the update-25 row; or ln 2 as the step-0 value", "sensitivity": True},
    {"id": "AMB-4", "where": "draft K3", "text": "'step time rises > 3 % or peak memory > 1 GB at micro-batch 4' - the S1 argv uses --batch 8; the comparison arm and the memory quantity are not named",
     "used": "memory: G12 head_retained_gb > 1.0 OR preflight gpu_mem_gb(H) - gpu_mem_gb(N same seed) > 1.0; time: run-mean s_per_update at update 400 AND the median per-window s/update (A7.4), each H vs the same-seed N, > +3 %; any limb fires K3",
     "alt": "N = the mean of the three N seeds; micro-batch 4 timing (not measured at S1)", "sensitivity": False},
    {"id": "AMB-5", "where": "draft K5", "text": "'the zero-weight / detached exact controls are not bit-identical to the comparator' is a check on an exact test, not on a banked number",
     "used": "read from the arm's PREFLIGHT_PASS.json G12 raw fields (a head arm cannot have trained without them); a missing or false field REFUSES the arm (an evidence-integrity failure, not an outcome); the shuffle limb is NOT_APPLICABLE",
     "alt": "treat a K5 failure as a fifth promotion condition", "sensitivity": False},
    {"id": "AMB-6", "where": "S1 prereg 4.3 (pick PDMS / pick DAC)", "text": "the held-out selection rule is not named in the S1 prereg",
     "used": "navsim_v1 (the draft's rule; the harness docstring's invocation; selection_pair passes L=None, which navsim_v1_lane cannot use)", "alt": "navsim_v1_lane", "sensitivity": False},
    {"id": "AMB-7", "where": "S1 prereg 4.3 l.124", "text": "'investigated before any read' - blocks which reads",
     "used": "an H1 identity defect makes the H1 outcome H1_IDENTITY_DEFECT (no promotion label) and is stamped on the H2 outcome; the H2 conditions are still computed",
     "alt": "withhold the H2 outcome as well", "sensitivity": False},
    {"id": "AMB-8", "where": "S1 prereg 3 R3 / A2.1", "text": "the P3 decision names are H2_*; an H1 arm also gets a P3 number from the harness",
     "used": "the P3 DECISION is taken for H2 arms only; an H1 arm's P3 is reported, undecided (H1's promotion rule does not use P3)", "alt": "decide P3 on H1 too", "sensitivity": False},
    {"id": "AMB-9", "where": "S1 prereg 4.3 cond 2 + the harness classify()", "text": "'separated and above the seed floor': floor = largest |delta| between two N seeds of the paired MEAN delta; the test is the pair's CI lower bound > floor",
     "used": "the harness's classify() (re-derived here and compared); the capped label 'separated, single seed, not yet a lever claim' counts as NOT above the floor", "alt": "-", "sensitivity": False},
]

NOT_COMPUTED = [
    "E1 / E2 / E3 on the full navtest (PDMS, DAC-zero share, zeroed share): the planner seam does not apply H2's feed (S1 prereg 4.2); H2's verdict is a HELD-OUT SELECTION verdict, never an ADOPT",
    "inference variance: REFe scoring is deterministic, ASSERTED by a second pass of the same checkpoint, not measured here",
    "C-shuffle: not an S1 arm (Master Mind ruling 2026-10-13) => every H2 gain carries 'capacity not excluded'; K5's shuffle limb cannot fire",
    "navtest PDMS for the H arms; the speed / along-track / heading / cross-track / curvature / yaw-rate / headway / decision-accuracy components of the four families (they need the navtest planner seam)",
]


class FetchRefused(RuntimeError):
    """A fetched file does not match the listing's size / sha256 / git blob id (or the listing offers nothing to verify against)."""


class EvidenceRefused(RuntimeError):
    """An arm's evidence contradicts its label, or a harness output breaks a pin."""


# ================================================================================================================ arms and the read table
# PREREG_AUX_DRIVABLE_S1.md Amendment A7 (2026-10-15, before any H-arm read): the Master Mind's rulings.  `used` GOVERNS; `alt` is supplementary and never decides.
A7_RULINGS = {"AMB-1": "A7.1: all three H2 seeds (used); 2 of 3 is supplementary",
              "AMB-2": "A7.2: all three H1 seeds (used); A3.3 says the S1-D3 drop rule is not exercised, so 'H1-s0 only' does not arise",
              "AMB-3": "A7.3: G12 bce_declared vs the first train row with update >= 40 (used); update-25 row and ln 2 are supplementary",
              "AMB-4": "A7.4: read at batch 8 (stricter for memory); time limb = run-mean OR median-window, either > +3 % fires"}
for _a in AMBIGUITIES:
    if _a["id"] in A7_RULINGS:
        _a["ruled"] = A7_RULINGS[_a["id"]]


def parse_arm(name: str) -> tuple:
    m = ARM_RE.match(name)
    if not m:
        raise ValueError(f"arm name {name!r} is not N-s<k> / H1-s<k> / H2-s<k> with k in 0..2")
    return m.group(1), int(m.group(2))


def missing_arms(present) -> list:
    return [a for a in REGISTERED if a not in set(present)]


def reads_table(present) -> list:
    """Every read, the arms it needs, which of them are missing, and its prereg clause. status: INCOMPLETE (named arms missing) | READY (all its arms present; read only when the whole set is)."""
    p = set(present)
    N = [f"N-s{s}" for s in SEEDS]
    rows = [("SEED_FLOOR", "seed_floor", N)]
    for s in SEEDS:
        for c in ("H1", "H2"):
            rows.append((f"SEL_{c}_s{s}", "sel_pair", [f"{c}-s{s}"] + N))
    for s in SEEDS:
        for c in ("H1", "H2"):
            rows.append((f"K2_{c}_s{s}", "k2", [f"{c}-s{s}"]))
            rows.append((f"P3_{c}_s{s}", "p3", [f"{c}-s{s}"]))
            rows.append((f"K6_{c}_s{s}", "k6", [f"{c}-s{s}"]))
            rows.append((f"K3_{c}_s{s}", "k3", [f"{c}-s{s}", f"N-s{s}"]))
            rows.append((f"K5_{c}_s{s}", "k5", [f"{c}-s{s}"]))
    for s in SEEDS:
        rows.append((f"H1_IDENTITY_s{s}", "h1_id", [f"H1-s{s}", f"N-s{s}"]))
    rows.append(("PROMOTE_H2", "promote_h2", [f"H2-s{s}" for s in SEEDS] + N))
    rows.append(("PROMOTE_H1", "promote_h1", [f"H1-s{s}" for s in SEEDS] + N))
    rows.append(("FAMILIES", "families", list(REGISTERED)))
    rows.append(("S1_OVERALL", "completeness", list(REGISTERED)))
    out = []
    for rid, cl, need in rows:
        miss = [a for a in need if a not in p]
        out.append({"id": rid, "clause": CLAUSES[cl], "arms_needed": need, "arms_missing": miss, "status": "INCOMPLETE" if miss else "READY"})
    return out


# ================================================================================================================ fetch + verify
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(path: str) -> str:
    h = hashlib.sha1()
    h.update(f"blob {os.path.getsize(path)}\0".encode())
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def verify_file(path: str, meta: dict) -> dict:
    """meta = the listing row {size, sha256 (LFS) | None, blob_id | None}. Size AND content hash must match; raises FetchRefused otherwise."""
    size = os.path.getsize(path)
    if size != meta.get("size"):
        raise FetchRefused(f"{path}: size {size} != listing size {meta.get('size')}")
    sha = sha256_file(path)
    if meta.get("sha256"):
        if sha != meta["sha256"]:
            raise FetchRefused(f"{path}: sha256 {sha[:16]}... != listing LFS sha256 {meta['sha256'][:16]}...")
        how, expected = "lfs_sha256", meta["sha256"]
    elif meta.get("blob_id"):
        b = git_blob_sha1(path)
        if b != meta["blob_id"]:
            raise FetchRefused(f"{path}: git blob id {b[:12]} != listing blob id {meta['blob_id'][:12]}")
        how, expected = "git_blob_sha1", meta["blob_id"]
    else:
        raise FetchRefused(f"{path}: the listing carries neither an LFS sha256 nor a git blob id: nothing to verify against")
    return {"size": size, "sha256": sha, "verified_against": how, "expected": expected}


def fetch_verified(remote, listing: dict, rel: str, cache: str) -> dict:
    """Download `rel` into cache/rel (or reuse the cached copy after RE-VERIFYING it) and verify it against the listing. A mismatch deletes the file and raises."""
    if rel not in listing:
        raise FetchRefused(f"{rel}: not in the remote listing")
    dest = os.path.join(cache, *rel.split("/"))
    meta = listing[rel]
    if os.path.isfile(dest):
        try:
            row = verify_file(dest, meta)
            row.update({"path": rel, "cached": True})
            return row
        except FetchRefused:
            os.remove(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    remote.download(rel, tmp)
    try:
        row = verify_file(tmp, meta)
    except FetchRefused:
        os.remove(tmp)
        raise
    os.replace(tmp, dest)
    row.update({"path": rel, "cached": False})
    return row


class HFRemote:
    """The private HF repo. The token is read from the environment (HF_TOKEN) at call time and is never stored, logged or written."""

    def __init__(self, repo: str = REPO):
        self.repo = repo
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    def _token(self):
        t = os.environ.get("HF_TOKEN")
        if not t:
            raise FetchRefused("HF_TOKEN is not set in the environment")
        return t

    def listing(self) -> dict:
        from huggingface_hub import HfApi
        info = HfApi(token=self._token()).model_info(self.repo, files_metadata=True)
        out = {}
        for s in info.siblings:
            lfs = getattr(s, "lfs", None)
            sha = lfs.get("sha256") if isinstance(lfs, dict) else (getattr(lfs, "sha256", None) if lfs else None)
            out[s.rfilename] = {"size": s.size, "sha256": sha, "blob_id": getattr(s, "blob_id", None)}
        return out

    def download(self, rel: str, dest: str) -> None:
        from huggingface_hub import hf_hub_download
        tmpdir = dest + ".dl"
        p = hf_hub_download(self.repo, rel, token=self._token(), local_dir=tmpdir)
        os.replace(p, dest)
        shutil.rmtree(tmpdir, ignore_errors=True)


class LocalRemote:
    """A directory standing in for the repo (tests, offline re-runs). `lies` = {rel: wrong-listing-row-overrides} to simulate a corrupt listing / file."""

    def __init__(self, root: str, lies: dict | None = None):
        self.root, self.lies = root, lies or {}

    def listing(self) -> dict:
        out = {}
        for d, _dn, fs in os.walk(self.root):
            for f in fs:
                p = os.path.join(d, f)
                rel = os.path.relpath(p, self.root).replace(os.sep, "/")
                big = os.path.getsize(p) > 4096 or rel.endswith((".npz", ".pt", ".tar"))
                out[rel] = {"size": os.path.getsize(p), "sha256": sha256_file(p) if big else None, "blob_id": git_blob_sha1(p)}
                out[rel].update(self.lies.get(rel, {}))
        return out

    def download(self, rel: str, dest: str) -> None:
        shutil.copyfile(os.path.join(self.root, *rel.split("/")), dest)


def arm_paths(arm: str, listing: dict) -> tuple:
    """-> (paths to fetch, paths absent from the listing). The PASS token is whichever arms/<ARM>/PASS_*.json the listing holds (exactly one expected)."""
    cls, _ = parse_arm(arm)
    want = [f"arms/{arm}/{f}" for f in ARM_SMALL + ARM_DUMP]
    if cls != "N":
        want += [f"arms/{arm}/{f}" for f in ARM_HEAD]
    tok = sorted(k for k in listing if re.fullmatch(rf"arms/{re.escape(arm)}/PASS_[0-9a-f]+\.json", k))
    want += tok[:1] if len(tok) == 1 else []
    absent = [w for w in want if w not in listing]
    if len(tok) != 1:
        absent.append(f"arms/{arm}/PASS_<id>.json (found {len(tok)})")
    return want, absent


def discover_arms(listing: dict) -> list:
    return [a for a in REGISTERED if f"arms/{a}/run_check.json" in listing]


# ================================================================================================================ evidence
def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_evidence(arm: str, cache: str) -> dict:
    d = os.path.join(cache, "arms", arm)
    tok = [f for f in os.listdir(d) if re.fullmatch(r"PASS_[0-9a-f]+\.json", f)]
    return {"manifest": read_jsonl(os.path.join(d, "arms_manifest.jsonl")), "run_check": read_json(os.path.join(d, "run_check.json")),
            "preflight": read_json(os.path.join(d, "PREFLIGHT_PASS.json")), "sft": read_jsonl(os.path.join(d, "sft.jsonl")),
            "token_verify": read_json(os.path.join(d, "token_verify.json")), "token": read_json(os.path.join(d, tok[0])) if len(tok) == 1 else None,
            "token_file": tok[0] if len(tok) == 1 else None, "token_sha256": sha256_file(os.path.join(d, tok[0])) if len(tok) == 1 else None}


def argv_flags(argv: list) -> dict:
    out, i = {}, 0
    while i < len(argv):
        t = argv[i]
        if t.startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[t], i = argv[i + 1], i + 2
            else:
                out[t], i = "", i + 1
        else:
            i += 1
    return out


def argv_sha256(argv: list) -> str:
    return hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()


def argv_class(flags: dict) -> str:
    if "--aux-drv" not in flags:
        return "N"
    return "H2" if flags.get("--aux-drv-feed") == "on" else "H1"


VARYING = ("--dump-scores", "--out", "--seed", "--aux-drv", "--w-aux-drv", "--aux-drv-lr", "--aux-drv-labels", "--aux-drv-feed")


def argv_core(flags: dict) -> tuple:
    return tuple(sorted((k, v) for k, v in flags.items() if k not in VARYING))


def check_arm_evidence(arm: str, ev: dict, listing: dict | None = None) -> tuple:
    """-> (problems, warnings). A problem means the arm's evidence contradicts its label or the registration: the arm is REFUSED (never read)."""
    cls, seed = parse_arm(arm)
    P, W = [], []
    rows = [r for r in ev["manifest"] if r.get("arm") == arm]
    if not rows:
        P.append(f"ID_MANIFEST_NO_ROW: arms_manifest.jsonl has no row for {arm} (arms named: {sorted({r.get('arm') for r in ev['manifest']})})")
    elif ev["manifest"][-1].get("arm") != arm:
        P.append(f"ID_MANIFEST_LAST_ROW: the manifest's last row names {ev['manifest'][-1].get('arm')!r}, not {arm}")
    rc = ev["run_check"]
    if rc.get("arm") != arm or rc.get("ok") is not True or rc.get("problems"):
        P.append(f"ID_RUNCHECK: run_check arm={rc.get('arm')!r} ok={rc.get('ok')} problems={rc.get('problems')}")
    tk = ev.get("token")
    if tk is None:
        P.append("ID_TOKEN_MISSING: no single PASS_<id>.json")
    else:
        b = tk.get("binding", {})
        if tk.get("verdict") != "PASS" or b.get("profile") != "refe_aux_s1":
            P.append(f"ID_TOKEN_PROFILE: verdict={tk.get('verdict')} profile={b.get('profile')}")
        argv = b.get("argv", [])
        if argv_sha256(argv) != b.get("argv_sha256"):
            P.append("ID_ARGV_SHA: sha256(compact JSON of the token's argv) != the token's argv_sha256")
        if rows and not any(r.get("argv_sha256") == b.get("argv_sha256") and r.get("token_sha256") == ev.get("token_sha256") for r in rows):
            P.append("ID_TOKEN_MANIFEST: no manifest row of this arm carries this token's file sha256 and argv sha256")
        fl = argv_flags(argv)
        if argv_class(fl) != cls:
            P.append(f"ID_ARGV_CLASS: the argv is an {argv_class(fl)} arm, the label says {cls}")
        if fl.get("--seed") != str(seed):
            P.append(f"ID_ARGV_SEED: argv --seed {fl.get('--seed')} != label seed {seed}")
        out = fl.get("--out", "").rstrip("/").split("/")[-1]
        if out != arm:
            P.append(f"ID_ARGV_OUT: argv --out ends in {out!r}, not {arm}")
        if fl.get("--dump-scores", "").rstrip("/").split("/")[-2:-1] != [arm]:
            P.append(f"ID_ARGV_DUMP: argv --dump-scores {fl.get('--dump-scores')!r} is not under .../{arm}/dump")
        if fl.get("--max-updates") != str(N_UPDATES) or fl.get("--eval-every") != str(N_UPDATES):
            P.append(f"ID_ARGV_UPDATES: --max-updates {fl.get('--max-updates')} --eval-every {fl.get('--eval-every')} (registered {N_UPDATES})")
        if cls != "N" and (fl.get("--w-aux-drv") != "0.1" or fl.get("--aux-drv-lr") not in ("2e-4", "0.0002")):
            P.append(f"ID_ARGV_HEADFLAGS: --w-aux-drv {fl.get('--w-aux-drv')} --aux-drv-lr {fl.get('--aux-drv-lr')} (registered 0.1 / 2e-4)")
        if cls == "N" and any(k.startswith("--aux-drv") or k == "--w-aux-drv" for k in fl):
            P.append("ID_ARGV_N_HAS_HEAD: an N arm names a head flag")
    sft = ev["sft"]
    start = [r for r in sft if r.get("event") == "start"]
    done = [r for r in sft if r.get("event") == "done"]
    if len(start) != 1 or start[0].get("updates") != N_UPDATES:
        P.append(f"ID_SFT_START: start events {[r.get('updates') for r in start]} (registered exactly one with updates == {N_UPDATES})")
    if len(done) != 1 or done[0].get("updates") != N_UPDATES:
        P.append(f"ID_SFT_DONE: done events {[r.get('updates') for r in done]}")
    tags = {r.get("tag") for r in sft if r.get("event") == "eval"}
    if not {"step0", "update400"} <= tags:
        P.append(f"ID_SFT_EVAL_TAGS: eval tags {sorted(tags)} (need step0 and update400)")
    for r in sft:
        if r.get("event") == "score_dump" and any(f"/runs/{arm}/" not in v for v in r.get("files", {}).values()):
            P.append(f"ID_SFT_DUMP_PATH: a score_dump path is not under /runs/{arm}/: {sorted(r['files'].values())[:1]}")
    data = [r for r in sft if r.get("event") == "data"]
    if len(data) != 1 or data[0].get("heldout_sets") != 3137 or data[0].get("heldout_logs") != 24 or data[0].get("train_rows_in_heldout_logs") != 0 or data[0].get("train_sets") != 174912:
        P.append(f"ID_SFT_DATA: data event {({k: data[0].get(k) for k in ('train_sets', 'heldout_sets', 'heldout_logs', 'train_rows_in_heldout_logs')} if data else None)} "
                 f"(registered 174912 / 3137 / 24 / 0)")
    has_aux_rows = any(r.get("event") == "aux_eval" for r in sft) or any("aux_bce" in r for r in sft)
    if cls == "N" and has_aux_rows:
        P.append("ID_N_HAS_AUX_ROWS: an N arm's log carries aux_eval / aux_bce rows")
    if cls != "N" and not has_aux_rows:
        P.append("ID_HEAD_NO_AUX_ROWS: a head arm's log carries no aux_eval / aux_bce rows")
    pf = ev["preflight"]
    if pf.get("verdict") != "PASS":
        P.append(f"ID_PREFLIGHT: verdict {pf.get('verdict')}")
    g12 = pf.get("gates", {}).get(G12)
    if cls == "N" and g12 is not None:
        P.append("ID_N_HAS_G12: an N arm's preflight carries gate G12")
    if cls != "N":
        P.extend(k5_problems(cls, g12))
    if listing is not None:
        has_head_files = any(k.startswith(f"arms/{arm}/") and (k.endswith("aux_drv.pt") or "/head_pred_" in k) for k in listing)
        if cls == "N" and has_head_files:
            P.append("ID_N_HAS_HEAD_FILES: an N arm's HF directory holds aux_drv.pt / head_pred dumps")
    return P, W


K5_FIELDS = {"H1": ("reference_replay_identical", "scorer_params_bit_identical_2steps", "identity_probe_reads_one_ulp", "score_to_head_grads_all_none", "bce_near_ln2"),
             "H2": ("feed_zero_init", "feed_step0_equal_base", "feed_probe_reads_difference", "score_to_head_grads_all_none", "bce_near_ln2")}


def k5_problems(cls: str, g12) -> list:
    """K5 (exact controls) as an evidence-integrity check on the G12 raw fields. A missing or false field is a problem (the arm is refused)."""
    if not isinstance(g12, dict):
        return ["K5_G12_MISSING: a head arm's PREFLIGHT_PASS.json has no G12 gate"]
    bad = [f for f in K5_FIELDS[cls] if g12.get(f) is not True]
    out = [f"K5_FIELD_NOT_TRUE: G12.{f} = {g12.get(f)!r}" for f in bad]
    if g12.get("ok") is not True:
        out.append(f"K5_G12_NOT_OK: G12.ok = {g12.get('ok')!r}")
    return out


def check_cross_arm(evs: dict, shas: dict, listing: dict | None = None) -> tuple:
    """-> (problems {arm: [..]}, warnings). Parity across arms: the argv core (everything but the seed / out / dump / head flags) is identical; N seeds do not share a score dump;
    an H2 arm's scores differ from its N (the feed acts); step0_B is the common init (listing sha256)."""
    P: dict = {}
    W: list = []
    cores = {a: argv_core(argv_flags(ev["token"]["binding"]["argv"])) for a, ev in evs.items() if ev.get("token")}
    if cores:
        ref_arm = sorted(cores)[0]
        for a, c in sorted(cores.items()):
            if c != cores[ref_arm]:
                diff = sorted(set(c) ^ set(cores[ref_arm]))[:3]
                P.setdefault(a, []).append(f"PARITY_ARGV_CORE: {a}'s argv differs from {ref_arm}'s outside the seed / out / dump / head flags: {diff}")
    ns = sorted(a for a in shas if a.startswith("N-"))
    for i, a in enumerate(ns):
        for b in ns[i + 1:]:
            if shas[a] == shas[b]:
                P.setdefault(b, []).append(f"DUP_SCORES: {a} and {b} have the SAME scores_update400_B.npz sha256: one file under two arm names")
    for a in sorted(shas):
        if a.startswith("H2-") and shas.get("N-" + a.split("-")[1]) == shas[a]:
            P.setdefault(a, []).append(f"H2_SCORES_EQUAL_N: {a}'s scores are bit-identical to its same-seed N: the feed did not act (or the dump is N's)")
    if listing is not None:
        s0 = {a: listing.get(f"arms/{a}/dump/scores_step0_B.npz", {}).get("sha256") for a in evs}
        if len({v for v in s0.values() if v}) > 1:
            W.append(f"STEP0_NOT_COMMON: the arms' scores_step0_B.npz differ (listing sha256): {s0}")
    return P, W


# ================================================================================================================ decision rules (pure)
def p3_decide(auc_head, auc_scorer, floor_lo) -> str:
    """S1 prereg R3 + A2.1. Thresholds on the POINT estimates; the floor is an ADDED condition on the two outcomes that would promote; H2_KILLED is unchanged."""
    thr = max(P3_FLOOR, auc_scorer + P3_MARGIN)
    if auc_head >= thr:
        base = "H2_EVALUATED"
    elif auc_head < auc_scorer:
        return "H2_KILLED"
    else:
        base = "H2_PROVISIONAL"
    if floor_lo is None:
        return "FLOOR_UNREAD"
    if not (floor_lo > 0.0):
        return "P3_AT_FLOOR"
    return base


def classify(read: dict, floor: dict | None, q: str) -> str:
    lo, hi = read["ci95"]
    if hi < 0:
        return "adverse"
    if not lo > 0:
        return "not_separated"
    if floor is None or not floor.get("measured"):
        return "separated, single seed, not yet a lever claim"
    return "separated and above the seed floor" if lo > floor["floor"].get(q, float("inf")) else "separated but within the seed floor"


def k2_from(nav_arm, ho_arm) -> dict:
    """K2: IoU in [0, 40) m over all labelled frames < 0.80. The NAVTEST P1 governs (S1 prereg l.98); P1-HO only when the navtest instrument is unavailable."""
    def one(a):
        if not a:
            return None
        iou = a["p1"]["all"]["0-40"]["est"]
        if iou is None or not math.isfinite(iou):
            return {"decision": "K2_UNDEFINED", "iou_0_40": iou}
        return {"decision": "K2_FIRES" if iou < K2_IOU_MIN else "K2_CLEARED", "iou_0_40": iou}
    n, h = one(nav_arm), one(ho_arm)
    if n is not None:
        out = dict(n, source="navtest P1 (governs)")
        if h is not None:
            out["p1_ho"] = h
            out["navtest_and_ho_agree"] = n["decision"] == h["decision"]
        if nav_arm.get("k2", {}).get("decision") not in (None, n["decision"]):
            raise EvidenceRefused(f"the harness's k2 {nav_arm['k2']['decision']} disagrees with the re-derived {n['decision']}")
        return out
    if h is not None:
        return dict(h, source="P1-HO (the navtest instrument was not available; S1 prereg 4.1 l.93)")
    return {"decision": "K2_UNREAD", "source": None}


def read_k6(rows: list, preflight: dict) -> dict:
    """AMB-3. Primary: first train row with update >= ceil(0.10 x 400) vs G12 bce_declared."""
    g12 = preflight.get("gates", {}).get(G12) or {}
    bce0 = g12.get("bce_declared")
    train = sorted((r for r in rows if r.get("event") == "train" and "aux_bce" in r), key=lambda r: r["update"])
    need = math.ceil(K6_FRAC * N_UPDATES)
    cand = [r for r in train if r["update"] >= need]
    if bce0 is None or not cand:
        return {"decision": "K6_UNREAD", "reason": f"bce_declared={bce0}, rows at update >= {need}: {len(cand)}"}
    row = cand[0]
    fires = row["aux_bce"] >= K6_FACTOR * bce0
    alt = {}
    alt["row_update25_vs_bce_declared"] = bool(train[0]["aux_bce"] >= K6_FACTOR * bce0)
    alt["row_used_vs_ln2"] = bool(row["aux_bce"] >= K6_FACTOR * math.log(2.0))
    return {"decision": "K6_FIRES" if fires else "K6_CLEARED", "aux_bce_at_row": row["aux_bce"], "row_update": row["update"], "bce_step0": bce0, "bar": K6_FACTOR * bce0,
            "alt_fires": alt, "sensitive_to_ambiguity": any(v != fires for v in alt.values()), "ambiguity": "AMB-3"}


def window_rates(rows: list) -> list:
    """A7.4: per-window s/update from the trainer's CUMULATIVE s_per_update ((time.time() - t0) / upd, scorer_finetune.py:1371): window (u1, u2] =
    (s2*u2 - s1*u1) / (u2 - u1), the first from update 0.  Its median is robust to a transient CPU-contention window that the run-mean averages in."""
    t = sorted((r for r in rows if r.get("event") == "train" and "s_per_update" in r), key=lambda r: r["update"])
    out, pu, ps = [], 0, 0.0
    for r in t:
        u, s = r["update"], r["s_per_update"]
        if u > pu:
            out.append((s * u - ps * pu) / (u - pu))
            pu, ps = u, s
    return out


def read_k3(h_pf: dict, n_pf: dict, h_rows: list, n_rows: list) -> dict:
    """AMB-4. Memory: G12 head_retained_gb > 1 or the preflight gpu_mem_gb delta vs the same-seed N > 1. Time: run-mean s_per_update at update 400 vs the same-seed N > +3 %."""
    g12 = h_pf.get("gates", {}).get(G12) or {}
    ret = g12.get("head_retained_gb")
    dgb = (h_pf["gpu_mem_gb"] - n_pf["gpu_mem_gb"]) if ("gpu_mem_gb" in h_pf and "gpu_mem_gb" in n_pf) else None

    def last(rows):
        t = [r for r in rows if r.get("event") == "train" and "s_per_update" in r]
        return max(t, key=lambda r: r["update"]) if t else None
    lh, ln = last(h_rows), last(n_rows)
    rise = (lh["s_per_update"] / ln["s_per_update"] - 1.0) if (lh and ln and ln["s_per_update"] > 0) else None
    wh, wn = window_rates(h_rows), window_rates(n_rows)
    mh, mn = (statistics.median(wh) if wh else None), (statistics.median(wn) if wn else None)
    rise_med = (mh / mn - 1.0) if (mh is not None and mn) else None
    mem = bool(ret is not None and ret > K3_GB)
    gpu = bool(dgb is not None and dgb > K3_GB)
    tim = bool(rise is not None and rise > K3_TIME)
    tmed = bool(rise_med is not None and rise_med > K3_TIME)
    unread = ret is None and dgb is None and rise is None and rise_med is None
    return {"decision": "K3_UNREAD" if unread else ("K3_FIRES" if (mem or gpu or tim or tmed) else "K3_CLEARED"), "head_retained_gb": ret, "limb_head_retained_gt_1gb": mem,
            "gpu_mem_gb_delta_vs_N": dgb, "limb_gpu_delta_gt_1gb": gpu, "step_time_rise_vs_N": rise, "limb_time_gt_3pct": tim,
            "step_time_rise_vs_N_median_window": rise_med, "limb_time_median_window_gt_3pct": tmed, "n_windows": [len(wh), len(wn)],
            "update_rows": [lh["update"] if lh else None, ln["update"] if ln else None], "ambiguity": "AMB-4", "ruling": "A7.4"}


def read_h1_identity(sel_pair: dict, sha_h1: str, sha_n: str, arrays_equal) -> dict:
    """S1 prereg l.124. Selection-identical == every paired quantity reads EXACTLY 0.0 with CI [0, 0] (the harness's self-pair signature)."""
    qs = sel_pair["paired_vs_n"]["quantities"]
    sel_ident = all(v["mean"] == 0.0 and v["ci95"] == [0.0, 0.0] for v in qs.values())
    scorer_ident = (sha_h1 == sha_n) or bool(arrays_equal)
    return {"selection_identical": sel_ident, "scores_file_sha_equal": sha_h1 == sha_n, "scores_arrays_bit_identical": bool(arrays_equal) if sha_h1 != sha_n else True,
            "defect": not sel_ident, "warning": ("scores differ bitwise although selection is identical" if (sel_ident and not scorer_ident) else None), "ambiguity": "AMB-7"}


def promote_h2(p3: dict, cls: dict, k2: dict, k3: dict, k6: dict, floor_measured: bool, c1_mode: str = "all") -> dict:
    """S1 prereg 4.3 conditions 1-4. Inputs per seed: p3 {s: decision}, cls {s: {"PDMS": label, "DAC": label}}, k2/k3/k6 {s: decision string}. c1_mode 'all' (used) or 'majority' (AMB-1 alt)."""
    seeds = sorted(p3)
    ok_set = ("H2_EVALUATED", "H2_PROVISIONAL")
    n_ok = sum(p3[s] in ok_set for s in seeds)
    c1 = (n_ok == len(seeds)) if c1_mode == "all" else (n_ok >= 2)
    prov = any(p3[s] == "H2_PROVISIONAL" for s in seeds if p3[s] in ok_set)
    regime = ("H2_PROVISIONAL" if prov else "H2_EVALUATED") if c1 else None
    need = PAIRS_NEEDED_PROVISIONAL if regime == "H2_PROVISIONAL" else PAIRS_NEEDED
    n_above = sum(cls[s]["PDMS"] == "separated and above the seed floor" for s in seeds)
    c2 = c1 and floor_measured and n_above >= need
    adverse = sorted(f"{s}:{q}" for s in seeds for q in ("PDMS", "DAC") if cls[s][q] == "adverse")
    c3 = not adverse
    fired = sorted(f"{s}:{name}" for name, d in (("K2", k2), ("K3", k3), ("K6", k6)) for s in seeds if d[s] not in ("K2_CLEARED", "K3_CLEARED", "K6_CLEARED"))
    c4 = not fired
    ok = bool(c1 and c2 and c3 and c4)
    return {"outcome": "H2_PROMOTED_TO_S2" if ok else "H2_NOT_PROMOTED", "regime": regime,
            "conditions": {"1_p3": {"ok": c1, "per_seed": {str(s): p3[s] for s in seeds}, "mode": c1_mode},
                           "2_pdms_pairs": {"ok": c2, "n_above_floor": n_above, "needed": need, "floor_measured": floor_measured, "per_seed": {str(s): cls[s]["PDMS"] for s in seeds}},
                           "3_no_adverse": {"ok": c3, "adverse": adverse},
                           "4_no_k2_k3_k6": {"ok": c4, "not_cleared": fired}},
            "stamps": ["capacity not excluded (no C-shuffle arm at S1)", "a HELD-OUT SELECTION verdict (T0-like); never an ADOPT of the draft's rule (S1 prereg 4.2)"],
            "if_promoted": "the first amendment is C-shuffle (derangement seed 20261011) before any S2 spend (S1 prereg 4.3 l.118)"}


def promote_h1(k2: dict, k6: dict, identity_defect: bool, seeds=None) -> dict:
    seeds = sorted(k2) if seeds is None else sorted(seeds)
    if identity_defect:
        return {"outcome": "H1_IDENTITY_DEFECT", "note": "H1's scorer must be bit-identical to N by construction; investigate BEFORE any H1 read (S1 prereg l.124). No promotion label is emitted."}
    bad = sorted(f"{s}:{n}" for n, d in (("K2", k2), ("K6", k6)) for s in seeds if d[s] not in ("K2_CLEARED", "K6_CLEARED"))
    return {"outcome": "H1_PROMOTED_TO_S2" if not bad else "H1_NOT_PROMOTED", "not_cleared": bad, "seeds_considered": seeds}


# ================================================================================================================ harness output checks
def check_harness_json(js: dict, kind: str) -> None:
    """The estimator pin: a harness output that does not say 10,000 resamples and seed 20260927 is refused."""
    if js.get("tool") != HARNESS_TOOL:
        raise EvidenceRefused(f"{kind}: tool {js.get('tool')!r} != {HARNESS_TOOL!r}")
    est = js.get("estimator", "")
    if f"{N_BOOT} resamples" not in est or f"seed {SEED}" not in est:
        raise EvidenceRefused(f"{kind}: estimator {est[:120]!r} does not carry '{N_BOOT} resamples' and 'seed {SEED}'")


def check_pins_against_harness(S) -> None:
    """The driver's pins equal the harness module's (S = the imported aux_s1_readout)."""
    got = {"N_BOOT": S.N_BOOT, "SEED": S.SEED, "K2_IOU_MIN": S.K2_IOU_MIN, "P3_AUC_FLOOR": S.P3_AUC_FLOOR, "P3_AUC_MARGIN": S.P3_AUC_MARGIN, "TOOL": S.TOOL}
    want = {"N_BOOT": N_BOOT, "SEED": SEED, "K2_IOU_MIN": K2_IOU_MIN, "P3_AUC_FLOOR": P3_FLOOR, "P3_AUC_MARGIN": P3_MARGIN, "TOOL": HARNESS_TOOL}
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        raise EvidenceRefused(f"pin drift between s1_read and aux_s1_readout (harness, driver): {bad}")


# ================================================================================================================ instruments (subprocess)
def find_eval_dir(explicit: str | None = None) -> str:
    for c in (explicit, os.environ.get("S1_EVAL_DIR"), HERE, os.path.join(os.path.dirname(os.path.dirname(HERE)), "eval"), os.path.join(os.path.dirname(HERE), "eval")):
        if c and os.path.isfile(os.path.join(c, "aux_s1_readout.py")) and os.path.isfile(os.path.join(c, "aux_navtest_forward.py")):
            return os.path.abspath(c)
    raise EvidenceRefused("cannot find eval/ (aux_s1_readout.py + aux_navtest_forward.py): pass --eval-dir or set S1_EVAL_DIR")


def cmd_selection(py, ev_dir, scores: dict, pdm_dir, out) -> list:
    """The harness's `selection` subcommand, launched through THIS file's `_harness` shim (see load_scores_copy: rl_val_eval.load_scores retains the whole score array per row)."""
    return [py, os.path.join(HERE, "s1_read.py"), "_harness", ev_dir, "selection", "--scores"] + [f"{a}={p}" for a, p in sorted(scores.items())] + \
           ["--pdm", pdm_dir, "--rule", RULE, "--n-boot", str(N_BOOT), "--out", out]


def cmd_forward(py, ev_dir, heads: dict, out_dir, ctx_cache, device="cpu") -> list:
    return [py, os.path.join(ev_dir, "aux_navtest_forward.py"), "--out", out_dir, "--ctx-cache", ctx_cache, "--device", device, "--no-gpu-lock", "--heads"] + \
           [f"{a}={p}" for a, p in sorted(heads.items())]


def cmd_perception_navtest(py, ev_dir, pkg, preds: dict, inits: dict, out) -> list:
    return [py, os.path.join(ev_dir, "aux_s1_readout.py"), "perception", "--labels-set", "navtest", "--dac-csv", os.path.join(pkg, U400_CSV),
            "--c3-table", os.path.join(pkg, C3_TABLE), "--c3-tokens", os.path.join(pkg, C3_TOKENS), "--n-boot", str(N_BOOT), "--pred"] + \
           [f"{a}={p}" for a, p in sorted(preds.items())] + ["--pred-init"] + [f"{a}={p}" for a, p in sorted(inits.items())] + ["--out", out]


def cmd_perception_heldout(py, ev_dir, preds: dict, inits: dict, heldout_logs, out) -> list:
    return [py, os.path.join(ev_dir, "aux_s1_readout.py"), "perception", "--labels-set", "train", "--heldout-logs", heldout_logs, "--n-boot", str(N_BOOT), "--pred"] + \
           [f"{a}={p}" for a, p in sorted(preds.items())] + ["--pred-init"] + [f"{a}={p}" for a, p in sorted(inits.items())] + ["--out", out]


def run_cmd(cmd: list, log_path: str) -> dict:
    t0 = time.perf_counter()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("HF_TOKEN", None)                                       # the instruments never need the token
    with open(log_path, "w", encoding="utf-8", newline="\n") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=os.path.dirname(cmd[1]))
    with open(log_path, encoding="utf-8", errors="replace") as f:
        tail = [x.rstrip() for x in f.readlines()[-3:]]
    return {"rc": p.returncode, "seconds": round(time.perf_counter() - t0, 2), "log": os.path.basename(log_path), "tail": tail}


def safe_extract_tar(tar_path: str, dest: str) -> list:
    os.makedirs(dest, exist_ok=True)
    names = []
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            n = m.name
            if os.path.isabs(n) or ".." in n.replace("\\", "/").split("/") or not (m.isfile() or m.isdir()):
                raise FetchRefused(f"{tar_path}: refused member {n!r}")
            names.append(n)
        try:
            t.extractall(dest, filter="data")
        except TypeError:                                                  # Python < 3.11.4: no `filter` argument; the member names were validated above
            t.extractall(dest)
    return names


def floor_inputs(floor_dir: str, pkg: str) -> dict:
    """{seed: path} of the A2.4 initial-head predictions (CPU passes), sha256-checked against the banked head_pred_sha256.txt."""
    pins = {}
    with open(os.path.join(pkg, FLOOR_SHA_FILE), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                h, _, name = line.strip().partition(" *")
                pins[name] = h
    out = {}
    for s in SEEDS:
        name = f"head_pred_H2-s{s}.npz"
        p = os.path.join(floor_dir, name)
        if not os.path.isfile(p):
            raise EvidenceRefused(f"FLOOR_INPUT_MISSING: {p} (regenerate with raw/2026-10-13-aux-p3-floor/make_init_heads.py + aux_navtest_forward.py --heads, CPU)")
        if sha256_file(p) != pins.get(name):
            raise EvidenceRefused(f"FLOOR_INPUT_SHA: {p} != the banked sha256 in {FLOOR_SHA_FILE}")
        out[s] = p
    return out


# ================================================================================================================ the memory shim for rl_val_eval.load_scores
def load_scores_copy(path, RV):
    """rl_val_eval.load_scores, row for row, with two defects removed: it reads `z["scores"]` (a full 10.7 MiB array, decompressed again) INSIDE the per-row loop and keeps a VIEW of it,
    so 3,137 rows pin 3,137 copies of the whole array (33.5 GB per arm: MemoryError on the real S1 dumps). Here each array is read ONCE and each row is copied.
    Same keys, same ckpt, same float64 scores, same space, same duplicate-key error; `check_loader_equivalence` proves it on every run."""
    import numpy as np
    if not path.endswith(".npz"):
        return RV.load_scores(path)
    z = np.load(path, allow_pickle=False)
    a = {k: z[k] for k in z.files}
    sp = str(a["space"]) if "space" in a else "logit"
    if sp not in ("logit", "prob"):
        raise ValueError(f"space must be logit or prob, got {sp!r}")
    out = {}
    for i in range(len(a["token"])):
        k = RV.key_of((a["log"][i], a["token"][i], a["step"][i], a["rank"][i]))
        if k in out:
            raise ValueError(f"duplicate scores for set {RV.kstr(k)} in {path}")
        out[k] = {"ckpt": int(a["ckpt_step"][i]), "scores": np.array(a["scores"][i], np.float64), "space": sp}
    return out


def check_loader_equivalence(RV, load_fn=None, n=48, tmpdir=None) -> None:
    """The shim equals the original on a 48-set npz (small enough for the original to run): same keys, ckpt, space, bit-identical scores, and an OWNED (not a view) score row."""
    import numpy as np
    import tempfile
    load_fn = load_fn or (lambda p: load_scores_copy(p, RV))
    rng = np.random.default_rng(7)
    keys = [(f"LOG{i % 5}", f"tok{i:03d}", i % 3, i % 2) for i in range(n)]
    sc = rng.normal(size=(n, 6, 7))
    with tempfile.TemporaryDirectory(dir=tmpdir) as d:
        p = os.path.join(d, "s.npz")
        RV.write_scores_npz(p, keys, [100 + i for i in range(n)], sc)
        o, c = RV.load_scores(p), load_fn(p)
        ok = set(o) == set(c) and all(o[k]["ckpt"] == c[k]["ckpt"] and o[k]["space"] == c[k]["space"] and o[k]["scores"].dtype == c[k]["scores"].dtype
                                      and np.array_equal(o[k]["scores"], c[k]["scores"]) and c[k]["scores"].base is None for k in o)
    if not ok:
        raise EvidenceRefused("the load_scores shim does not reproduce rl_val_eval.load_scores on the 48-set check")


def harness_shim(argv: list) -> int:
    """`s1_read.py _harness <eval_dir> <harness args...>`: install load_scores_copy into rl_val_eval, prove it, then run aux_s1_readout.main(args)."""
    ev_dir, rest = argv[0], argv[1:]
    sys.path.insert(0, ev_dir)
    sys.path.insert(0, os.path.join(os.path.dirname(ev_dir), "refe"))
    import rl_val_eval as RV
    import aux_s1_readout as S
    check_loader_equivalence(RV)
    RV.load_scores = lambda path: load_scores_copy(path, RV)
    return S.main(rest)


# ================================================================================================================ families
def families_block(sel: dict | None) -> dict:
    """Per binding family: what S1 measured (held-out, T0-like, paired CI) and what it cannot, with the reason. STRATEGIC is PAUSED by the PI (item 25): not a pass, not omitted."""
    comp = {}
    if sel:
        for name, pair in sorted(sel.get("pairs", {}).items()):
            q = pair["paired_vs_n"]["quantities"]
            comp[name] = {"LONGITUDINAL": {k: q[k] for k in ("EP", "TTC")}, "LATERAL": {"DAC": q["DAC"]}, "n_sets": pair["n_sets"], "n_logs": pair["n_logs"],
                          "TACTICAL": {"head_arm": {k: pair["arms"]["head"].get(k) for k in ("PDMS_x100", "fan_random_x100", "fan_best_x100", "skill")},
                                       "n_arm": {k: pair["arms"]["n"].get(k) for k in ("PDMS_x100", "fan_random_x100", "fan_best_x100", "skill")}}}
    return {
        "tier": TIER, "estimator": ESTIMATOR,
        "LONGITUDINAL": {"measured_at_S1": "EP and TTC sub-scores of the PICK on the 3,137 held-out sets (paired H - N, same seed)",
                         "not_computed": "speed MAE/bias, along-track MAE/bias, headway/time-gap (navtest families instrument; the planner seam does not apply H2's feed, S1 prereg 4.2)"},
        "LATERAL": {"measured_at_S1": "DAC sub-score of the PICK on the held-out sets (paired)", "not_computed": "heading / cross-track / curvature / yaw-rate MAE (navtest families instrument; same reason)"},
        "TACTICAL": {"measured_at_S1": "selection quality as pick / random / best-of-64 on the held-out sets, per arm (the random control is the arm's own fan mean)",
                     "not_computed": "lateral/longitudinal/5-way decision accuracy and kappa, command adherence (navtest families instrument); the paired CI of (actual - random) is not produced by the harness"},
        "STRATEGIC": {"status": "PAUSED", "ruling": "PI 2026-09-19, PI_DECISION_QUEUE.md item 25: switched off temporarily until tactical and operative drive with high quality; reported as PAUSED, not a pass, not omitted, not extended to the other three"},
        "components_by_pair": comp,
    }


# ================================================================================================================ decide (pure)
def decide(present, refused, evid, sel, nav, ho, h1id_inputs) -> dict:
    """The outcome section. `present` = arms with clean evidence; `refused` = {arm: [problems]}. Returns status INCOMPLETE / REFUSED with NO outcome label, or READ with the outcomes.
    evid {arm: {"preflight","sft"}}; sel / nav / ho = the harness JSONs (or None); h1id_inputs {seed: {"sha_h1","sha_n","arrays_equal"}}."""
    miss = missing_arms(present)
    reads = reads_table(present)
    if refused:
        return {"status": "REFUSED", "refused_arms": refused, "missing_arms": miss, "reads": reads,
                "note": "an arm's evidence contradicts its label or the registration: it is not read; no outcome label is emitted"}
    if miss:
        return {"status": "INCOMPLETE", "missing_arms": miss, "reads": reads, "H2": None, "H1": None,
                "note": "S1 prereg A5: the first read happens after all registered arms exist; no outcome label (K2_*, H2_*, P3_*, *_PROMOTED) is emitted from a partial arm set"}
    if sel is None:
        raise EvidenceRefused("a complete arm set needs the selection read")
    check_harness_json(sel, "selection")
    for nm, js in (("perception_navtest", nav), ("perception_heldout", ho)):
        if js is not None:
            check_harness_json(js, nm)
    floor = sel["seed_floor"]
    per = {"p3": {}, "k2": {}, "k3": {}, "k6": {}}
    cls = {}
    for s in SEEDS:
        for c in ("H1", "H2"):
            a = f"{c}-s{s}"
            nav_a = nav["arms"].get(a) if nav else None
            ho_a = ho["arms"].get(a) if ho else None
            per["k2"][a] = k2_from(nav_a, ho_a)
            per["k6"][a] = read_k6(evid[a]["sft"], evid[a]["preflight"])
            per["k3"][a] = read_k3(evid[a]["preflight"], evid[f"N-s{s}"]["preflight"], evid[a]["sft"], evid[f"N-s{s}"]["sft"])
            p3 = (nav_a or {}).get("p3")
            if p3 is None or "decision" not in p3:
                per["p3"][a] = {"decision": "P3_NOT_READ", "reason": "perception on the 1,058 C3 tokens was not produced for this arm: perception gate unread (S1 prereg 4.1 l.96)"}
            else:
                d = p3["decision"]
                lo = d.get("floor_lower_bound")
                local = p3_decide(d["auc_head"], d["auc_scorer"], lo)
                if local != d["decision"]:
                    raise EvidenceRefused(f"{a}: the harness's P3 decision {d['decision']} disagrees with the re-derived {local}")
                per["p3"][a] = {"decision": local if c == "H2" else "NOT_DECIDED_FOR_H1", "reported_label": local, "auc_head": d["auc_head"], "auc_scorer": d["auc_scorer"],
                                "evaluate_threshold": d["evaluate_threshold"], "floor_lower_bound": lo, "ambiguity": "AMB-8"}
        for c in ("H1", "H2"):
            key = f"{c}-s{s}-minus-N-s{s}"
            pair = sel["pairs"][key]
            cls[f"{c}-s{s}"] = {q: classify(pair["paired_vs_n"]["quantities"][q], floor, q) for q in ("PDMS", "DAC")}
            if cls[f"{c}-s{s}"] != pair["classification"]:
                raise EvidenceRefused(f"{key}: the harness's classification {pair['classification']} disagrees with the re-derived {cls[f'{c}-s{s}']}")
    h1id = {}
    for s in SEEDS:
        i = h1id_inputs[s]
        h1id[s] = read_h1_identity(sel["pairs"][f"H1-s{s}-minus-N-s{s}"], i["sha_h1"], i["sha_n"], i.get("arrays_equal"))
    defect = any(v["defect"] for v in h1id.values())

    def by_seed(cname, tbl, key=None):
        return {s: (tbl[f"{cname}-s{s}"][key] if key else tbl[f"{cname}-s{s}"]["decision"]) for s in SEEDS}
    kw = dict(p3=by_seed("H2", per["p3"]), cls={s: cls[f"H2-s{s}"] for s in SEEDS}, k2=by_seed("H2", per["k2"]), k3=by_seed("H2", per["k3"]), k6=by_seed("H2", per["k6"]),
              floor_measured=bool(floor["measured"]))
    h2 = promote_h2(**kw)
    alt = promote_h2(**kw, c1_mode="majority")
    h2["sensitivity_AMB-1"] = {"outcome_under_alt_majority": alt["outcome"], "differs": alt["outcome"] != h2["outcome"]}
    h2["h1_identity_defect_open"] = defect
    h1_kw = dict(k2=by_seed("H1", per["k2"]), k6=by_seed("H1", per["k6"]), identity_defect=defect)
    h1 = promote_h1(**h1_kw)
    h1_alt = promote_h1(k2={0: h1_kw["k2"][0]}, k6={0: h1_kw["k6"][0]}, identity_defect=defect, seeds=[0])
    h1["sensitivity_AMB-2"] = {"outcome_under_alt_H1_s0_only": h1_alt["outcome"], "differs": h1_alt["outcome"] != h1["outcome"]}
    for r in reads:
        r["status"] = "READ"
    return {"status": "READ", "missing_arms": [], "reads": reads, "H2": h2, "H1": h1, "per_arm": per, "classification": cls, "h1_identity": {str(k): v for k, v in h1id.items()},
            "seed_floor": floor, "families": families_block(sel)}


# ================================================================================================================ output
def render_md(rd: dict) -> str:
    o = rd["outcome"]
    L = [f"# S1 read (`{TOOL}`)", "",
         f"**Status: {o['status']}.** Tier: {TIER}. Estimator: {ESTIMATOR}. It {QUESTION}.", "",
         f"Registered arms ({len(REGISTERED)}): present and clean {rd['arms']['present'] or 'none'}; missing {o['missing_arms'] or 'none'}; "
         f"refused {sorted(o.get('refused_arms', {})) or 'none'}.", ""]
    if o["status"] == "INCOMPLETE":
        L += [f"> {o['note']}", ""]
    if o["status"] == "REFUSED":
        L += [f"> {o['note']}"] + [f"> - {a}: {p}" for a, ps in sorted(o["refused_arms"].items()) for p in ps] + [""]
    L += ["## Reads", "", "| read | status | arms needed | arms missing | prereg clause |", "|---|---|---|---|---|"]
    for r in o["reads"]:
        L.append(f"| {r['id']} | {r['status']} | {', '.join(r['arms_needed'])} | {', '.join(r['arms_missing']) or '-'} | {r['clause']} |")
    if o["status"] == "READ":
        L += ["", "## Outcome", "", f"- **H2: {o['H2']['outcome']}** (regime {o['H2']['regime']}); conditions: " + json.dumps(o["H2"]["conditions"], sort_keys=True),
              f"- stamps: {'; '.join(o['H2']['stamps'])}", f"- **H1: {o['H1']['outcome']}** " + json.dumps({k: v for k, v in o["H1"].items() if k != "outcome"}, sort_keys=True)]
    comp = rd.get("components", {}).get("seed_floor")
    if comp:
        L += ["", "## Component (a rig property of the N arms; NOT an outcome)", "", f"Seed floor, measured={comp['measured']} (n N seeds {comp['n_n_seeds']}): `{json.dumps(comp['floor'], sort_keys=True)}`"]
    L += ["", "## Families (binding; per family, never pooled)", "", "| family | status at S1 | not computed |", "|---|---|---|"]
    fam = rd["families"]
    for k in ("LONGITUDINAL", "LATERAL", "TACTICAL"):
        L.append(f"| {k} | {fam[k]['measured_at_S1']} | {fam[k]['not_computed']} |")
    L.append(f"| STRATEGIC | PAUSED ({fam['STRATEGIC']['ruling']}) | - |")
    L += ["", "## Prereg ambiguities (none resolved silently)", "", "| id | where | text | used | not used |", "|---|---|---|---|---|"]
    for a in AMBIGUITIES:
        L.append(f"| {a['id']} | {a['where']} | {a['text']} | {a['used']} | {a['alt']} |")
    L += ["", "## Not computed", ""] + [f"- {x}" for x in NOT_COMPUTED]
    L += ["", "## Timings (s)", "", "```", json.dumps(rd["timings_s"], indent=1, sort_keys=True), "```", ""]
    return "\n".join(L)


# ================================================================================================================ main
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_harness":
        return harness_shim(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("read")
    p.add_argument("--arms", nargs="*", default=None, help="arms to fetch and check (default: every registered arm that has run_check.json on HF)")
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--remote", default="hf", help="'hf' or 'local:<dir>'")
    p.add_argument("--eval-dir", default=None)
    p.add_argument("--python", default=DEFAULT_PY)
    p.add_argument("--ctx-cache", default=DEFAULT_CTX_CACHE)
    p.add_argument("--floor-dir", default=DEFAULT_FLOOR_DIR)
    p.add_argument("--heldout-logs", default=DEFAULT_HELDOUT_LOGS)
    p.add_argument("--skip-navtest", action="store_true", help="do not run the navtest head forward / perception (P3 NOT READ => H2 is not promoted; K2 falls back to P1-HO)")
    p.add_argument("--plumbing-check", nargs="*", default=[], help="head arms to run the head path on as PLUMBING; values go to raw files, never into S1_READ")
    a = ap.parse_args(argv)
    return cmd_read(a)


PATH_ARGS = ("out", "cache", "ctx_cache", "floor_dir", "heldout_logs", "eval_dir")


def abs_paths(a, names=PATH_ARGS) -> None:
    """Every path an instrument subprocess receives must be ABSOLUTE: run_cmd runs them with cwd = the eval dir.  2026-10-15: a relative --out sent
    perception_navtest.json to eval/raw/..., the step returned 0, and the read crashed on the missing file (selection failed outright)."""
    for n in names:
        v = getattr(a, n, None)
        if isinstance(v, str) and v and not os.path.isabs(v):
            setattr(a, n, os.path.abspath(v))


def cmd_read(a) -> int:
    abs_paths(a)
    T: dict = {}
    t_all = time.perf_counter()
    ev_dir = find_eval_dir(a.eval_dir)
    pkg = os.path.dirname(ev_dir)
    sys.path.insert(0, ev_dir)
    import aux_s1_readout as S                                   # noqa: E402
    check_pins_against_harness(S)
    os.makedirs(a.out, exist_ok=True)
    inst = os.path.join(a.out, "instrument")
    os.makedirs(inst, exist_ok=True)
    remote = HFRemote() if a.remote == "hf" else LocalRemote(a.remote.split(":", 1)[1])
    t0 = time.perf_counter()
    listing = remote.listing()
    T["listing"] = round(time.perf_counter() - t0, 2)
    arms = list(a.arms) if a.arms else discover_arms(listing)
    for x in arms:
        parse_arm(x)
    manifest, partial, problems, evs, shas = [], {}, {}, {}, {}
    t0 = time.perf_counter()
    for arm in arms:
        want, absent = arm_paths(arm, listing)
        if absent:
            partial[arm] = absent
            continue
        for rel in want:
            manifest.append(fetch_verified(remote, listing, rel, a.cache))
        evs[arm] = load_evidence(arm, a.cache)
        shas[arm] = sha256_file(os.path.join(a.cache, "arms", arm, "dump", "scores_update400_B.npz"))
    lab_rows = [fetch_verified(remote, listing, rel, a.cache) for rel in LABEL_FILES]
    manifest += lab_rows
    T["fetch"] = round(time.perf_counter() - t0, 2)
    tar_sha = next(r["sha256"] for r in lab_rows if r["path"].endswith("s1_heldout_pdm.tar"))
    mf = open(os.path.join(a.cache, "labels", "MANIFEST_s1_labels_sha256.txt"), encoding="utf-8").read()
    if f"{tar_sha}  s1_heldout_pdm.tar" not in mf:
        raise FetchRefused("s1_heldout_pdm.tar's sha256 is not the one in MANIFEST_s1_labels_sha256.txt")
    pdm_cmp = read_json(os.path.join(a.cache, "labels", "pdm_complete_heldout.json"))
    if pdm_cmp.get("verdict") != "COMPLETE" or pdm_cmp.get("eligible") != 3137:
        raise EvidenceRefused(f"pdm_complete_heldout.json: {pdm_cmp.get('verdict')} eligible {pdm_cmp.get('eligible')}")
    t0 = time.perf_counter()
    pdm_root = os.path.join(a.cache, "labels", "x")
    if not os.path.isdir(os.path.join(pdm_root, "heldout_pdm")):
        safe_extract_tar(os.path.join(a.cache, "labels", "s1_heldout_pdm.tar"), pdm_root)
    T["untar_pdm"] = round(time.perf_counter() - t0, 2)
    pdm_dir = os.path.join(pdm_root, "heldout_pdm")
    for arm in evs:
        pr, _w = check_arm_evidence(arm, evs[arm], listing)
        if pr:
            problems[arm] = pr
    xp, xw = check_cross_arm(evs, shas, listing)
    for arm, pr in xp.items():
        problems.setdefault(arm, []).extend(pr)
    clean = [x for x in evs if x not in problems]
    json.dump(manifest, open(os.path.join(a.out, "fetch_manifest.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)

    comps, sel, nav, ho, h1id_inputs, steps = {}, None, None, None, {}, {}
    full = not missing_arms(clean) and not problems
    n_arms = sorted(x for x in clean if x.startswith("N-"))
    t0 = time.perf_counter()
    sel_arms = sorted(clean) if full else n_arms                  # an incomplete set only ever sees its N arms (the seed floor is a rig property, not an outcome)
    if len(sel_arms) >= 2 or (full and sel_arms):
        sc = {x: os.path.join(a.cache, "arms", x, "dump", "scores_update400_B.npz") for x in sel_arms}
        sel_json = os.path.join(inst, "selection.json" if full else "selection_N_only.json")
        steps["selection"] = run_cmd(cmd_selection(a.python, ev_dir, sc, pdm_dir, sel_json), os.path.join(inst, "selection.log"))
        if steps["selection"]["rc"] == 0:
            js = read_json(sel_json)
            check_harness_json(js, "selection")
            if full:
                sel = js
            elif len(n_arms) >= 3:
                comps["seed_floor"] = js["seed_floor"]
    T["selection"] = round(time.perf_counter() - t0, 2)
    if full and not a.skip_navtest:
        head = {x: os.path.join(a.cache, "arms", x, "aux_drv.pt") for x in clean if not x.startswith("N-")}
        t0 = time.perf_counter()
        hd = os.path.join(inst, "heads")
        steps["forward"] = run_cmd(cmd_forward(a.python, ev_dir, head, hd, a.ctx_cache), os.path.join(inst, "forward.log"))
        T["forward"] = round(time.perf_counter() - t0, 2)
        fl = floor_inputs(a.floor_dir, pkg)
        preds = {x: os.path.join(hd, f"head_pred_{x}.npz") for x in head}
        inits = {x: fl[parse_arm(x)[1]] for x in head}
        t0 = time.perf_counter()
        steps["perception_navtest"] = run_cmd(cmd_perception_navtest(a.python, ev_dir, pkg, preds, inits, os.path.join(inst, "perception_navtest.json")), os.path.join(inst, "perception_navtest.log"))
        T["perception_navtest"] = round(time.perf_counter() - t0, 2)
        if steps["perception_navtest"]["rc"] == 0:
            nav = read_json(os.path.join(inst, "perception_navtest.json"))
    if full:
        if sha256_file(a.heldout_logs) != HELDOUT_LOGS_SHA256:
            raise EvidenceRefused("the held-out logs file is not the one the PASS tokens bind (sha256 b814998d...)")
        hp = {x: os.path.join(a.cache, "arms", x, "dump", "head_pred_update400.npz") for x in clean if not x.startswith("N-")}
        hi = {x: os.path.join(a.cache, "arms", x, "dump", "head_pred_step0.npz") for x in hp}
        t0 = time.perf_counter()
        steps["perception_heldout"] = run_cmd(cmd_perception_heldout(a.python, ev_dir, hp, hi, a.heldout_logs, os.path.join(inst, "perception_heldout.json")), os.path.join(inst, "perception_heldout.log"))
        T["perception_heldout"] = round(time.perf_counter() - t0, 2)
        if steps["perception_heldout"]["rc"] == 0:
            ho = read_json(os.path.join(inst, "perception_heldout.json"))
        import numpy as np
        for s in SEEDS:
            ph, pn = (os.path.join(a.cache, "arms", f"{c}-s{s}", "dump", "scores_update400_B.npz") for c in ("H1", "N"))
            eq = None
            if shas[f"H1-s{s}"] != shas[f"N-s{s}"]:
                zh, zn = np.load(ph, allow_pickle=False), np.load(pn, allow_pickle=False)
                eq = zh.files == zn.files and all(np.array_equal(zh[k], zn[k]) for k in zh.files)
            h1id_inputs[s] = {"sha_h1": shas[f"H1-s{s}"], "sha_n": shas[f"N-s{s}"], "arrays_equal": eq}
    plumb = {}
    for arm in a.plumbing_check:
        try:
            plumb[arm] = plumbing_check(a, arm, clean, ev_dir, pkg, inst)
        except (EvidenceRefused, FetchRefused, OSError) as e:                 # a plumbing failure is a finding about the plumbing, never a crash of the read
            plumb[arm] = {"status": "FAILED", "error": f"{type(e).__name__}: {str(e)[:300]}"}
    evid = {x: {"preflight": evs[x]["preflight"], "sft": evs[x]["sft"]} for x in clean}
    if full and any(v["rc"] != 0 for v in steps.values()):
        raise EvidenceRefused(f"an instrument step failed: { {k: v['rc'] for k, v in steps.items()} }")
    outcome = decide(clean, problems, evid, sel, nav, ho, h1id_inputs)
    if outcome["status"] == "READ" and a.skip_navtest:
        outcome["note"] = "navtest instrument skipped: P3 NOT READ => H2 is not promoted (S1 prereg 4.1 l.96)"
    T["total"] = round(time.perf_counter() - t_all, 2)
    rd = {"tool": TOOL, "prereg": {"file": PREREG, "amendments": "A1-A6", "harness": f"{HARNESS_TOOL} @ {HARNESS_COMMIT}"}, "tier": TIER, "estimator": ESTIMATOR, "question_answered": QUESTION,
          "pins": {"n_boot": N_BOOT, "seed": SEED, "k2_iou_min": K2_IOU_MIN, "p3_floor": P3_FLOOR, "p3_margin": P3_MARGIN, "k6": [K6_FACTOR, K6_FRAC], "k3": [K3_TIME, K3_GB], "rule": RULE},
          "arms": {"registered": list(REGISTERED), "requested": arms, "present": sorted(clean), "partial_on_hf": partial, "evidence_problems": problems},
          "fetch": {"repo": REPO, "n_files": len(manifest), "bytes": sum(r["size"] for r in manifest), "all_verified": all(r.get("verified_against") for r in manifest),
                    "manifest": "fetch_manifest.json", "not_fetched": NOT_FETCHED, "warnings": xw},
          "outcome": outcome, "components": ({"seed_floor": dict(comps["seed_floor"], not_an_outcome=True, reads_only="N arms")} if comps else {}), "families": families_block(sel),
          "ambiguities": AMBIGUITIES, "not_computed": NOT_COMPUTED, "instrument_steps": steps, "instrument_failures": {k: v["rc"] for k, v in steps.items() if v["rc"] != 0}, "plumbing": plumb, "timings_s": T}
    json.dump(rd, open(os.path.join(a.out, "S1_READ.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True, default=str)
    with open(os.path.join(a.out, "S1_READ.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(render_md(rd))
    failed = {k: v["rc"] for k, v in steps.items() if v["rc"] != 0}
    print(f"ZZS1READ_{outcome['status']} present {sorted(clean)} missing {outcome['missing_arms']} refused {sorted(problems)} instrument_failures {failed or 'none'} total_s {T['total']}", flush=True)
    if failed:
        return 4
    return 0 if outcome["status"] in ("READ", "INCOMPLETE") else 3


def plumbing_check(a, arm, clean, ev_dir, pkg, inst) -> dict:
    """Run the head path (navtest forward -> perception on the 1,058 C3 tokens, with the arm's own initial-head floor) on ONE head arm. Only status, rc and timings are returned."""
    cls, seed = parse_arm(arm)
    if cls == "N" or arm not in clean:
        return {"status": "SKIPPED", "reason": "not a clean head arm" if cls != "N" else "an N arm has no head"}
    d = os.path.join(inst, "plumbing", arm)
    os.makedirs(d, exist_ok=True)
    out = {"values": "written to instrument/plumbing/<ARM>/*.json for plumbing only; NOT read into S1_READ, NOT used for any decision"}
    fwd = run_cmd(cmd_forward(a.python, ev_dir, {arm: os.path.join(a.cache, "arms", arm, "aux_drv.pt")}, os.path.join(d, "heads"), a.ctx_cache), os.path.join(d, "forward.log"))
    out["forward"] = fwd
    if fwd["rc"] == 0:
        fl = floor_inputs(a.floor_dir, pkg)
        per = run_cmd(cmd_perception_navtest(a.python, ev_dir, pkg, {arm: os.path.join(d, "heads", f"head_pred_{arm}.npz")}, {arm: fl[seed]}, os.path.join(d, "perception_navtest.json")),
                      os.path.join(d, "perception.log"))
        out["perception_navtest"] = per
        ident = os.path.join(d, "heads", "identity.json")
        if os.path.isfile(ident):
            out["navtest_forward_identity_pass"] = read_json(ident).get("identity", {}).get("PASS")
    return out


if __name__ == "__main__":
    sys.exit(main())
