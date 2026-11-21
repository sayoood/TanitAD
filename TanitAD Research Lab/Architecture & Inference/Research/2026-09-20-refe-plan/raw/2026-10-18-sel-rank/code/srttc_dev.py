#!/usr/bin/env python3
"""SR-TTC DEV GATE applier.  CPU only.  Implements the dev gate of SR-TTC's OWN registration (MM_DECISIONS_EPR1B.md D5), NOT EP-R1's bar:

    PREREG_SELRANK_ARM.md  (sha256 cd73c4d4...)   + PREREG_SELRANK_ARM.amendment1.md + PREREG_SELRANK_ARM.amendment2.md   + MM_DECISIONS_R1.md
    all cited below as  <file>:<line>  (line numbers of the files as landed at the tip 884c05b).

WHAT IT DOES (one read, ever)
  Applies the two fitted TTC rows of c* (`ttc_row_seed0.npy`, `ttc_row_seed1.npy`, written by `srttc_r1.py run`) to P2's held-out PRE-HEAD features, replacing ONLY column 3 of the six logits
  (stored + feat . (row - P2's row): the SAME expression `srttc_r1.forward_z6` / `score_sets` fitted with), picks by the planner's own rule NC*DAC*(5EP+5TTC+2C)/12 (float64, stable, lowest index
  on ties) and reads PDMS BY LOOKUP `pdms[token, new_pick]`: the fan is frozen (C4).

THE REGISTERED BAR (PREREG_SELRANK_ARM.md:93, the only place a dev bar is defined for SR-TTC)
    paired out-of-sample TOTAL gain >= +0.30 PDMS x100 on BOTH seeds   AND   C0, C1, C2, C4 pass.
    "24 logs cannot support a lower-bound rule" (:93): the interval is REPORTED, never a bar.
  Verdict: PASS  = controls green AND both seeds >= +0.30;   FAIL = controls green AND either seed < +0.30 (PREREG_SELRANK_ARM.md:102 DEV-FAIL: "dev gain < +0.30 on either seed");
           VOID  = a control is RED ("each is a gate; a miss voids the arm", :67).
  Terms the registration leaves UNDEFINED are NOT invented; they are listed under `registration_gaps` in the result (see GAPS below).

EVERY BAR TERM -> ITS REGISTRATION LINE  (also written into the result as `registration_trace`)
    gain >= +0.30, both seeds ........ PREREG_SELRANK_ARM.md:93        (seed-floor range 0.33-0.44: :26, :93; reported |seed0-seed1| vs the 0.44 floor: amendment2.md:197, a DISCLOSURE not a bar)
    C0 identity ...................... PREREG_SELRANK_ARM.md:69        unchanged TTC row -> P2's pick on every token, gain exactly 0.0
    C1 reachability .................. PREREG_SELRANK_ARM.md:70        non-zero TTC-row gradient, EXACTLY 0.0 on frozen modules, zero-weight control 0.0 (recorded numbers of r1_result.json,
                                                                       RE-EVALUATED here), a launch-gate PASS token bound to the R1 code and bank (MM_DECISIONS_R1.md:24), and the applier-side
                                                                       reach (the TTC column moves, the other five are bit-identical, a negated-row positive control changes picks)
    C2 negative control .............. PREREG_SELRANK_ARM.md:71, :93   TTC labels shuffled across the proposals of each set, same fit: dev gain <= 0 on each seed, NO tolerance (MM_DECISIONS_EPR1B.md D7)
    C3 replicate ..................... PREREG_SELRANK_ARM.md:72        two training seeds {0,1}, same flags; the claim needs BOTH to clear the bar
    C4 fan frozen .................... PREREG_SELRANK_ARM.md:73        re-dumped proposals equal the table's, max abs pose diff 0.0 m on all tokens
    C5 estimator ..................... PREREG_SELRANK_ARM.md:74        paired log-cluster bootstrap, 10,000 resamples, seed 20260927 (zero_tail.Boot), here 24 logs
    c* and its sha256 written first .. PREREG_SELRANK_ARM.md:8-10      "written down before the gate set is read"; gate read ONCE; the 8 unchosen configs are never scored on the gate
    c* from the registered grid ...... PREREG_SELRANK_ARM.md:6, :39     lr {1e-4,3e-4,1e-3} x anchor {0.03,0.1,0.3}, K = 8
    epochs/batch/criterion/seeds ..... MM_DECISIONS_R1.md:14-16         Adam, 10 epochs, 256 sets/batch; criterion pooled_oof; seeds {0,1}; ties -> first in grid order (:20)
    the LC call before the gate ...... PREREG_SELRANK_ARM.amendment2.md:169-177, :182   FLAT -> read; RISING/INCONCLUSIVE below f_max -> scale up, do NOT read; f = 100 % (= f_max, :7) -> read and record the call
    gate-set admissibility ........... PREREG_SELRANK_ARM.amendment2.md:245-248       (1) P2HO_DONE PARTIAL false, 1,705 tokens / 3,137 sets, (2) every gate of the package PASSES, (3) the 333 first-run tokens
                                                                       bit-identical, (4) the table is landed with its sha256 (the pins below); "if any of the four fails, the dev gate is NOT read"
    fit-set size per fraction ........ PREREG_SELRANK_ARM.amendment2.md:98-101        logs 117 / 292 / 584 / 1,167
  Outcome rows: PREREG_SELRANK_ARM.md:98-103 (section 7) and amendment2.md:206-207 (A2.6).

GAPS (registration silent; reported, never filled in)
    G1  C2's shuffle seed is not registered (srttc_c2.py uses 20261018 = EP-R1's, an UNREGISTERED choice made before any data; any seed gives a null).
    G2  C2 has no tolerance and no replicate: a null fit has a gain of random sign, so C2 can read RED by chance; the registration does not say what a RED C2 costs beyond "voids" (:67).
    G3  Whether a VOID that arises AFTER the outcomes were opened consumes the one dev read: the registration is silent (EP-R1's applier treats the attempt as the read).
    G4  A2.6 (amendment2.md:207) says the RISING/INCONCLUSIVE + DEV-FAIL row is "only reachable when f_max < 100 %"; with f_max = 100 % and f = 100 % a non-FLAT call + FAIL is reachable and unregistered.
    G5  The launch-gate token's "bound to the commit" (:70) cannot be re-checked without git; this applier checks the token's HMAC, verdict, profile, stage, fraction and the srttc_r1.py / bank-manifest
        bindings, not the commit blobs (that is `srttc_r1_gate.py verify`'s job at launch).
    G6  No grid-edge rule (EP-R1 has E1 in PREREG_EPR1 s8; the SR-TTC documents contain no such rule): `c_star_at_grid_edge` is REPORTED only.

EVIDENCE CLASS of anything this prints: MEASURED on the dev table (SELECTION tier, a GATE and not a claim: MM_DECISIONS_EPR1 D3).  Only the navtest primary is quotable.

THE ORDER IS THE PROTOCOL (each refusal fires BEFORE any dev OUTCOME is opened; `pdms`, `sub`, `valid` are touched only after READ_STARTED is written)
  0. the read is once  1. the code is the landed code  2. c* result: registered sha256, registered object  3. LC call (A2.4)  4. C2 files  5. gate-set admissibility (A2.10)
  6. dev input pins + the no-outcome controls (integrity, C0, C1, C4): a RED one stops the process, the read is NOT spent  7. READ_STARTED written EXCLUSIVELY
  8. only then: outcomes, formula check, C0 outcome half, the two arms, C2, the verdict.

CLI
    srttc_dev.py run --run-dir C:/Users/Admin/srttc_r1/run/f10 --registered-result-sha256 <hex> [--r1-gate-token F] [--gate-key-file F] [--p2ho-dir D] [--out-dir D] [--dev-dir D] [--p2-ckpt P]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import hmac
import io
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import srttc_r1 as R   # noqa: E402  (the LANDED fit code; used for FUNCTIONS only: Cfg.sha256, row_bytes, boot_draws, paired_delta, canon_json.  Registered VALUES are re-declared below.)

SCHEMA = "srttc-dev-v1"
STAGE = "DEV"
R1_SCHEMA = "srttc-r1-v1"

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# REGISTERED VALUES, hard-coded HERE with their source line, independent of srttc_r1.py (the gate compares the built object against the registration, not against itself)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
TTC_COL = 3                                         # S0_HANDOFF.md:24 (labels6 order NC, DAC, EP, TTC, C, DDC); MM_DECISIONS_R1.md (TTC = column 3)
N_PROPOSALS, FEAT_DIM, N_COMP = 64, 256, 6
DEV_BAR_X100 = 0.30                                 # PREREG_SELRANK_ARM.md:93
SEED_FLOOR_RANGE = (0.33, 0.44)                     # PREREG_SELRANK_ARM.md:26, :93 ; reported against |gain_s0 - gain_s1| (amendment2.md:197)
SEED_KEYS = ("0", "1")                              # PREREG_SELRANK_ARM.md:72 (C3)
REG_SEEDS = [0, 1]
BOOT_B, BOOT_SEED = 10000, 20260927                 # PREREG_SELRANK_ARM.md:74 (C5)
REG_GRID_LR = (1e-4, 3e-4, 1e-3)                    # PREREG_SELRANK_ARM.md:6, :39
REG_GRID_ANCHOR = (0.03, 0.1, 0.3)                  # PREREG_SELRANK_ARM.md:6, :39
REG_K = 8                                           # PREREG_SELRANK_ARM.md:39
REG_EPOCHS, REG_BATCH_SETS = 10, 256                # MM_DECISIONS_R1.md:14
REG_CRITERION = "pooled_oof"                        # MM_DECISIONS_R1.md:15
REG_FRACTIONS = (10, 25, 50, 100)                   # amendment2.md:7 (nested sizes), A2.1 item 4
F_MAX = 100                                         # amendment2.md:7, MM_DECISIONS_R1.md:23
LOGS_BY_FRACTION = {10: 117, 25: 292, 50: 584, 100: 1167}   # amendment2.md:98-101 (A2.2 table, MEASURED there)
LC_DELTA_FLAT = 0.010                               # amendment2.md:171
LC_PREFIXES = [25, 50, 100]                         # amendment2.md:164
LC_BOTH_RULE = "flat"                               # MM_DECISIONS_R1.md:19
DEV_N_TOKENS, DEV_N_LOGS, DEV_N_SETS = 1705, 24, 3137   # amendment2.md:245 (A2.10)
N_RECOMPUTE_CHECK = 333                             # amendment2.md:247 (A2.10 condition 3)
FORMULA_TOL = 1e-6                                  # pdms == NC*DAC*(5EP+5TTC+2C)/12 on the dev table, MEASURED max abs diff 0.0 (P2HO_DONE G4b)
FEATCHECK_TOL = 2e-3                                # P2's head reproduces the stored logits from the features; MEASURED 1.45e-3 (fp16 input), a next-token control reads 15.2
C2_SHUFFLE_SEED = 20261018                          # GAP G1 : UNREGISTERED for SR-TTC (EP-R1's value); recorded in the C2 result, not read here
SRTTC_R1_SHA256 = "b841347d90c298ec597f10afbbd284d68258df55c98cbac8cbd9a0a444cc5d0b"    # the LANDED srttc_r1.py at tip 884c05b; the fit and this applier must be the same code
GATE_PROFILE, GATE_KIND, GATE_SCHEMA = "srttc-r1", "launch-gate-token", "tanitad.launch_gate/1"     # srttc_r1_gate.py:50-52

DEV_DIR = "D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18"
DEV_FILES = ("table_p2_heldout.npz", "prehead_features_p2_heldout.npz", "proposals.npz")
DEV_PINS = {   # MANIFEST.sha256.json of raw/2026-10-18-p2-heldout-table (landed; A2.10 condition 4)
    "table_p2_heldout.npz": "8eacc19b35b12aee9b0ab047716c901399a711d22ddd3c25cd890ecf2f02004b",
    "prehead_features_p2_heldout.npz": "e37b9c554bf2c2546e958ff763dca23b71e0b77036a271d6a003c8ba2672f522",
    "proposals.npz": "a37422102add66ca386fdadfafa670b121cca1c67ed9dde997232982bb36de4b",
}
P2HO_DIR = ("D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-p2-heldout-table")
DEFAULT_P2_CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"
DEFAULT_GATE_ROOT = "C:/Users/Admin/srttc_r1/gate"

COMPS = ("nc", "dac", "ep", "ttc", "c", "ddc")
SUB_NAMES = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance")

READ_STARTED = "srttc_dev_READ_STARTED.json"
RESULT_NAME = "srttc_dev_result.json"
SIDECAR_NAME = "srttc_dev_result.sha256"
R1_RESULT, LC_RESULT = "r1_result.json", "lc_result.json"
C2_RESULT = "r1_c2_result.json"
C2_ROW_FMT = "ttc_row_c2shuffle_seed{sd}.npy"

REGISTRATION_TRACE = {
    "bar_gain_both_seeds": "PREREG_SELRANK_ARM.md:93 (>= +0.30 PDMS x100 on BOTH seeds)",
    "no_lower_bound_rule": "PREREG_SELRANK_ARM.md:93 (24 logs cannot support a lower-bound rule: interval reported, never a bar)",
    "C0": "PREREG_SELRANK_ARM.md:69", "C1": "PREREG_SELRANK_ARM.md:70 ; launch-gate token MM_DECISIONS_R1.md:24",
    "C2": "PREREG_SELRANK_ARM.md:71, :93 (no tolerance: MM_DECISIONS_EPR1B.md D7)", "C3": "PREREG_SELRANK_ARM.md:72", "C4": "PREREG_SELRANK_ARM.md:73", "C5": "PREREG_SELRANK_ARM.md:74",
    "c_star_written_first": "PREREG_SELRANK_ARM.md:8-10", "grid": "PREREG_SELRANK_ARM.md:6, :39", "r1_params": "MM_DECISIONS_R1.md:14-16, :20",
    "lc_before_gate": "PREREG_SELRANK_ARM.amendment2.md:169-177, :182", "gate_set": "PREREG_SELRANK_ARM.amendment2.md:245-248", "fit_set_size": "PREREG_SELRANK_ARM.amendment2.md:98-101",
    "outcome_rows": "PREREG_SELRANK_ARM.md:98-103 ; PREREG_SELRANK_ARM.amendment2.md:206-207", "seed_gap_disclosure": "PREREG_SELRANK_ARM.amendment2.md:197",
    "void_means": "PREREG_SELRANK_ARM.md:67 ('each is a gate; a miss voids the arm')",
}
REGISTRATION_GAPS = {
    "G1": "C2 shuffle seed is not registered for SR-TTC (srttc_c2.py: 20261018, EP-R1's value, chosen before any data)",
    "G2": "C2 has no tolerance and no replicate; a null shuffled fit has a gain of random sign, so C2 can read RED by chance (MM D7: no tolerance)",
    "G3": "the registration does not say whether a VOID that arises after the outcomes were opened consumes the one dev read (this applier: the attempt consumes it)",
    "G4": "amendment2.md:207 says non-FLAT + DEV-FAIL is 'only reachable when f_max < 100 %'; with f_max = 100 % it is reachable at f = 100 % and unregistered",
    "G5": "the launch-gate token's commit binding is not re-checked here (needs git); HMAC, verdict, profile, stage, fraction, srttc_r1.py and bank-manifest bindings are",
    "G6": "no grid-edge rule exists in the SR-TTC documents; c_star_at_grid_edge is reported only",
}


class DevRefusal(RuntimeError):
    """A registered precondition is not met.  Raised BEFORE any dev outcome is opened (unless the message says otherwise); nothing was read."""


class DevGuardError(RuntimeError):
    """A file access that would touch navtest / the S0 bank, or anything outside the allow-list."""


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# PathGuard + the ONLY functions that touch the file system (`_io_*`; a test enforces that by AST).  The dev applier never reads navtest or the fit bank.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
FORBIDDEN_TOKENS = ("navtest", "table_p2_full", "srttc_s0", ".sets.npz", "s1_read_cache", "ranked_fit_logs")


def _cmp(p: str) -> str:
    return os.path.normpath(os.path.abspath(p)).replace("\\", "/").lower()


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


class DevGuard:
    def __init__(self, read_dirs=(), read_files=(), write_dirs=()):
        self.read_dirs = [(_cmp(d), _cmp(os.path.realpath(d))) for d in read_dirs]
        self.read_files = [(_cmp(f), _cmp(os.path.realpath(f))) for f in read_files]
        self.write_dirs = [(_cmp(d), _cmp(os.path.realpath(d))) for d in write_dirs]

    @staticmethod
    def forbidden(path: str) -> list:
        forms = {str(path).replace("\\", "/").lower(), _cmp(path), _cmp(os.path.realpath(path))}
        return sorted({t for f in forms for t in FORBIDDEN_TOKENS if t in f})

    def _allowed(self, path: str, dirs, files) -> bool:
        a, r = _cmp(path), _cmp(os.path.realpath(path))
        return any(_under(a, da) or _under(r, dr) for da, dr in dirs) or any(a == fa or r == fr for fa, fr in files)

    def check_read(self, path: str) -> str:
        hit = self.forbidden(path)
        if hit:
            raise DevGuardError(f"REFUSED read of {path!r}: it names a navtest / fit-bank token {hit} (the dev applier never reads those)")
        if not self._allowed(path, self.read_dirs, self.read_files):
            raise DevGuardError(f"REFUSED read of {path!r}: outside the allow-list (dev dir, the run dir, the package dir, the named inputs, the output dir)")
        return path

    def check_write(self, path: str) -> str:
        hit = self.forbidden(path)
        if hit:
            raise DevGuardError(f"REFUSED write to {path!r}: forbidden token {hit}")
        if not self._allowed(path, self.write_dirs, ()):
            raise DevGuardError(f"REFUSED write to {path!r}: outside the output dir")
        return path


def _io_read_bytes(guard: DevGuard, path: str) -> bytes:
    guard.check_read(path)
    with open(path, "rb") as f:
        return f.read()


def _io_sha256(guard: DevGuard, path: str) -> str:
    guard.check_read(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _io_npz(guard: DevGuard, path: str):
    guard.check_read(path)
    return np.load(path, allow_pickle=False)


def _io_exists(guard: DevGuard, path: str) -> bool:
    guard.check_read(path)
    return os.path.exists(path)


def _io_stat(guard: DevGuard, path: str) -> tuple:
    guard.check_read(path)
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns)


def _io_torch_load(guard: DevGuard, path: str):
    guard.check_read(path)
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def _io_makedirs(guard: DevGuard, d: str) -> None:
    guard.check_write(os.path.join(d, "x"))
    os.makedirs(d, exist_ok=True)


def _io_write_bytes(guard: DevGuard, path: str, data: bytes, exclusive: bool = False) -> str:
    guard.check_write(path)
    with open(path, "xb" if exclusive else "wb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def canon(obj) -> bytes:
    return R.canon_json(_jsonable(obj))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# options
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class DevOptions:
    run_dir: str                                        # SR-TTC R1's run dir: r1_result.json, ttc_row_seed{0,1}.npy, lc_result.json, r1_c2_result.json, ttc_row_c2shuffle_seed{0,1}.npy
    registered_result_sha256: str                       # sha256 of r1_result.json written down BEFORE this read (PREREG_SELRANK_ARM.md:8)
    dev_dir: str = DEV_DIR
    p2_ckpt: str = DEFAULT_P2_CKPT
    out_dir: str = ""                                   # default: run_dir (the confirm applier is given the result path explicitly)
    p2ho_dir: str = P2HO_DIR                            # the landed held-out-table package (A2.10 attestations)
    r1_gate_token: str = ""                             # default <DEFAULT_GATE_ROOT>/f<fraction>_run/r1_gate_token.json
    gate_key_file: str = ""                             # default: $TANITAD_GATE_KEY_FILE or ~/.config/tanitad/launch_gate.key
    dev_pins: dict = dataclasses.field(default_factory=lambda: dict(DEV_PINS))
    expect_tokens: int | None = DEV_N_TOKENS
    expect_logs: int | None = DEV_N_LOGS
    expect_sets: int | None = DEV_N_SETS
    srttc_r1_sha256: str | None = SRTTC_R1_SHA256
    expect_fit_logs: int | None = None                  # None -> LOGS_BY_FRACTION[fraction]
    reg_epochs: int = REG_EPOCHS
    reg_batch_sets: int = REG_BATCH_SETS
    gate_profile: str = GATE_PROFILE
    featcheck_tol: float = FEATCHECK_TOL
    require_gate_token: bool = True
    require_p2ho: bool = True
    allow_r1_pin_overrides: bool = False
    chunk: int = 512

    def resolved_out_dir(self) -> str:
        return self.out_dir or self.run_dir

    def token_path(self, fraction: int) -> str:
        return self.r1_gate_token or f"{DEFAULT_GATE_ROOT}/f{fraction}_run/r1_gate_token.json"

    def key_path(self) -> str:
        return self.gate_key_file or os.environ.get("TANITAD_GATE_KEY_FILE") or os.path.join(os.path.expanduser("~"), ".config", "tanitad", "launch_gate.key")

    def guard(self) -> DevGuard:
        files = [self.p2_ckpt, os.path.abspath(R.__file__)]
        for f in (self.r1_gate_token, self.gate_key_file):
            if f:
                files.append(f)
        files.append(self.key_path())
        dirs = [self.dev_dir, self.run_dir, self.resolved_out_dir()]
        if self.p2ho_dir:
            dirs.append(self.p2ho_dir)
        if not self.r1_gate_token:
            dirs.append(DEFAULT_GATE_ROOT)
        return DevGuard(read_dirs=dirs, read_files=files, write_dirs=[self.resolved_out_dir()])

    def pins_overridden(self) -> list:
        out = []
        for k, d in (("dev_pins", DEV_PINS), ("expect_tokens", DEV_N_TOKENS), ("expect_logs", DEV_N_LOGS), ("expect_sets", DEV_N_SETS), ("srttc_r1_sha256", SRTTC_R1_SHA256),
                     ("expect_fit_logs", None), ("reg_epochs", REG_EPOCHS), ("reg_batch_sets", REG_BATCH_SETS), ("gate_profile", GATE_PROFILE), ("featcheck_tol", FEATCHECK_TOL),
                     ("require_gate_token", True), ("require_p2ho", True), ("allow_r1_pin_overrides", False)):
            if getattr(self, k) != d:
                out.append(k)
        return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# small loaders
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_p2_head(guard: DevGuard, ckpt: str) -> tuple:
    """-> (W6 [6,256] float32, b6 [6] float32): ALL six rows of P2's score_head (the TTC row is the one replaced; the others are needed for the integrity check)."""
    sd = _io_torch_load(guard, ckpt)["model"]
    W, b = sd["score_head.weight"].detach().float().clone().numpy(), sd["score_head.bias"].detach().float().clone().numpy()
    if W.shape != (N_COMP, FEAT_DIM) or b.shape != (N_COMP,):
        raise DevRefusal(f"P2 score_head is {W.shape}/{b.shape}, expected ({N_COMP}, {FEAT_DIM}) (a 7-output SFT-4 head is not P2)")
    return W, b


def load_row(guard: DevGuard, path: str, expect_sha256: str | None = None) -> tuple:
    """-> (w [256] float32, b [1] float32, sha256).  Layout `srttc_r1.row_bytes`: float32[257] = weight[3,:256] then bias[3], .npy.  The hash is taken over the exact bytes that are parsed."""
    raw = _io_read_bytes(guard, path)
    sha = sha256_bytes(raw)
    if expect_sha256 is not None and sha != expect_sha256:
        raise DevRefusal(f"{path}: sha256 {sha} != the recorded {expect_sha256} (the row is not the one the fit wrote)")
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    if arr.dtype != np.float32 or arr.shape != (FEAT_DIM + 1,) or not np.isfinite(arr).all():
        raise DevRefusal(f"{path}: expected finite float32[{FEAT_DIM + 1}], got {arr.dtype}{arr.shape}")
    return arr[:FEAT_DIM].copy(), arr[FEAT_DIM:].copy(), sha


def load_json(guard: DevGuard, path: str) -> tuple:
    raw = _io_read_bytes(guard, path)
    return json.loads(raw.decode("utf-8")), sha256_bytes(raw)


def _need(doc: dict, *path):
    cur = doc
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            raise DevRefusal(f"result file lacks {'/'.join(path)}")
        cur = cur[p]
    return cur


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the dev table: INPUTS (no outcome) and OUTCOMES (opened only after READ_STARTED)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class DevInputs:
    token: np.ndarray
    log: np.ndarray
    log_names: list
    log_idx: np.ndarray
    feat: torch.Tensor              # [N,64,256] fp16: the INPUT of P2's score_head (same forward as the logits)
    z6: torch.Tensor                # [N,64,6] float32: the stored logits of P2
    pick_banked: np.ndarray         # [N] int64: the dumped pick of P2 (planner, rule navsim_v1)
    props_table: np.ndarray         # [N,64,8,3] float32
    props_dump: np.ndarray          # [N,64,8,3] float32: the dump the features came from (proposals.npz)
    pick_dump: np.ndarray
    token_feat: np.ndarray
    log_feat: np.ndarray
    token_dump: np.ndarray
    log_dump: np.ndarray
    provenance: dict

    @property
    def n(self) -> int:
        return len(self.token)

    @property
    def n_logs(self) -> int:
        return len(self.log_names)


@dataclasses.dataclass
class DevOutcomes:
    pdms: np.ndarray                # [N,64] float64
    sub: np.ndarray                 # [N,64,6] float64 (nc, dac, ep, ttc, c, ddc)
    safe: np.ndarray                # [N,64] bool: NC >= 1 and DAC >= 1
    formula: dict


def load_dev_inputs(guard: DevGuard, opts: DevOptions) -> DevInputs:
    paths = {n: os.path.join(opts.dev_dir, n) for n in DEV_FILES}
    sha, stats = {}, {}
    for n, p in paths.items():
        if not _io_exists(guard, p):
            raise DevRefusal(f"dev input {p} does not exist")
        sha[n] = _io_sha256(guard, p)                                              # BEFORE np.load
        pin = opts.dev_pins.get(n)
        if pin is not None and sha[n] != pin:                                       # MUTATION-TARGET-SRDEV-DEVSHA
            raise DevRefusal(f"dev input {n}: sha256 {sha[n]} != the landed {pin}")
        stats[n] = _io_stat(guard, p)
    T, F, P = (_io_npz(guard, paths[n]) for n in DEV_FILES)
    token, log = np.array([str(x) for x in T["token"]]), np.array([str(x) for x in T["log"]])
    N = len(token)
    if opts.expect_tokens is not None and N != opts.expect_tokens:
        raise DevRefusal(f"dev table has {N} tokens, expected {opts.expect_tokens} (A2.10: the full gate set, amendment2.md:245)")
    if len(set(token.tolist())) != N:
        raise DevRefusal("dev table: duplicate tokens")
    log_names, log_idx = np.unique(log, return_inverse=True)                      # sorted names: exactly selrank_dev.py's Boot(log_idx, len(ulogs))
    if opts.expect_logs is not None and len(log_names) != opts.expect_logs:
        raise DevRefusal(f"dev table has {len(log_names)} logs, expected {opts.expect_logs}")
    logits = T["logits"]
    props_t = T["proposals"]
    pick = T["pick"].astype(np.int64)
    feat = F["prehead_features"]
    token_f, log_f = np.array([str(x) for x in F["token"]]), np.array([str(x) for x in F["log"]])
    token_d, log_d = np.array([str(x) for x in P["token"]]), np.array([str(x) for x in P["log"]])
    props_d, pick_d = P["proposals"], P["pick"].astype(np.int64)
    if feat.dtype != np.float16 or feat.shape != (N, N_PROPOSALS, FEAT_DIM):
        raise DevRefusal(f"pre-head features {feat.dtype}{feat.shape} != float16[{N},{N_PROPOSALS},{FEAT_DIM}]")
    if logits.dtype != np.float32 or logits.shape != (N, N_PROPOSALS, N_COMP):
        raise DevRefusal(f"table logits {logits.dtype}{logits.shape} != float32[{N},{N_PROPOSALS},{N_COMP}]")
    if props_t.shape != (N, N_PROPOSALS, 8, 3) or props_d.shape != (N, N_PROPOSALS, 8, 3):
        raise DevRefusal(f"proposals shapes {props_t.shape} / {props_d.shape} != ({N},{N_PROPOSALS},8,3)")
    if not (np.isfinite(logits).all() and np.isfinite(feat.astype(np.float32)).all()):
        raise DevRefusal("non-finite logit or feature in the dev inputs")
    if pick.shape != (N,) or pick.min() < 0 or pick.max() >= N_PROPOSALS:
        raise DevRefusal("table pick out of range")
    for n, p in paths.items():
        if _io_stat(guard, p) != stats[n]:
            raise DevRefusal(f"{n} changed on disk between verification and load")
    prov = {"dev_dir": opts.dev_dir, "sha256": sha, "n_tokens": N, "n_logs": len(log_names)}
    feat, logits, props_t, props_d = np.array(feat), np.array(logits), np.array(props_t), np.array(props_d)      # materialise, then release the archive handles
    for z in (T, F, P):
        z.close()
    return DevInputs(token=token, log=log, log_names=log_names.tolist(), log_idx=np.asarray(log_idx, np.int64), feat=torch.from_numpy(feat), z6=torch.from_numpy(logits),
                     pick_banked=pick, props_table=props_t, props_dump=props_d, pick_dump=pick_d, token_feat=token_f, log_feat=log_f, token_dump=token_d, log_dump=log_d, provenance=prov)


def safe_np(sub: np.ndarray) -> np.ndarray:
    """NC >= 1 and DAC >= 1 (the unsafe-pick rate's definition, as `epr1.safe_np`)."""
    return (sub[..., 0] >= 1.0) & (sub[..., 1] >= 1.0)


def read_outcomes(guard: DevGuard, table_path: str) -> DevOutcomes:
    """THE DEV READ.  Opens the OUTCOME arrays (pdms, sub, valid, sub_names) of the held-out table and checks the PDMS formula.  Called only after READ_STARTED exists."""
    T = _io_npz(guard, table_path)
    names = tuple(str(x) for x in T["sub_names"])
    if names != SUB_NAMES:
        raise DevRefusal(f"table sub_names {names} != {SUB_NAMES} (the component order is the bank's label order; TTC would not be column {TTC_COL})")
    pdms, sub, valid = T["pdms"].astype(np.float64), T["sub"].astype(np.float64), T["valid"]
    if not (np.isfinite(pdms).all() and np.isfinite(sub).all() and bool(valid.all())):
        raise DevRefusal("dev table: a non-finite or invalid (token, proposal) cell")
    if sub.min() < 0.0 or sub.max() > 1.0 or pdms.min() < 0.0 or pdms.max() > 1.0:
        raise DevRefusal("dev table: a score outside [0,1]")
    T.close()
    form = sub[..., 0] * sub[..., 1] * (5.0 * sub[..., 2] + 5.0 * sub[..., 3] + 2.0 * sub[..., 4]) / 12.0
    d = float(np.abs(pdms - form).max())
    ctl = float(np.abs(pdms - form * sub[..., 5]).max())
    if not d <= FORMULA_TOL:
        raise DevRefusal(f"dev table: pdms != NC*DAC*(5EP+5TTC+2C)/12 (max abs diff {d:.3e} > {FORMULA_TOL:g}); the PDMS lookup would be on the wrong scale")
    return DevOutcomes(pdms=pdms, sub=sub, safe=safe_np(sub), formula={"max_abs_diff": d, "ddc_in_product_control_max_abs_diff": ctl, "tol": FORMULA_TOL})


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the applier: ONLY the named column changes
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def apply_row(z6_stored: torch.Tensor, feat: torch.Tensor, row_w, row_b, p2_w, p2_b, col: int = TTC_COL) -> torch.Tensor:
    """New logits with ONLY column `col` changed: stored + feat . (row_w - p2_w) + (row_b - p2_b).  The SAME expression as srttc_r1.forward_z6 / score_sets (so the applier evaluates the
    object the fit trained), and bit-identical to the stored logits when the row equals P2's (the delta is exactly 0.0)."""
    dw = torch.as_tensor(np.asarray(row_w, np.float32) - np.asarray(p2_w, np.float32))
    db = torch.as_tensor(np.asarray(row_b, np.float32) - np.asarray(p2_b, np.float32))
    out = z6_stored.clone()
    out[..., col] = out[..., col] + (feat.float() @ dw + db)             # MUTATION-TARGET-SRDEV-APPLYCOL
    return out


def apply_rows(z6: torch.Tensor, feat: torch.Tensor, rows: dict, W6: np.ndarray, b6: np.ndarray, chunk: int = 512) -> np.ndarray:
    """rows: {col: (w [256], b [1])}.  -> [N,64,6] float32 logits with ONLY the named columns replaced."""
    out = []
    for s in range(0, z6.shape[0], chunk):
        z, f = z6[s:s + chunk], feat[s:s + chunk]
        for col, (w, b) in rows.items():
            z = apply_row(z, f, w, b, W6[col], b6[col:col + 1], col=col)       # MUTATION-TARGET-SRDEV-APPLY
        out.append(z.numpy() if isinstance(z, torch.Tensor) else np.asarray(z))
    return np.concatenate(out) if out else np.zeros((0, N_PROPOSALS, N_COMP), np.float32)


def pick_logits(z6: np.ndarray) -> np.ndarray:
    """The planner's rule on logits: NC*DAC*(5EP+5TTC+2C)/12 over float64 sigmoids, stable argsort descending, first element (lowest index on ties)  == srttc_r1.picks_np."""
    return R.picks_np(z6)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# controls that need NO outcome
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def integrity_featcheck(inp: DevInputs, W6: np.ndarray, b6: np.ndarray, tol: float, chunk: int = 512) -> dict:
    """amendment2.md:247 (2) 'pre-head feature check', RE-MEASURED: P2's own head, applied to the cached features, reproduces the stored logits (all six columns).
    Negative control: the SAME comparison against the logits of the next token must read far above the tolerance (the probe can read a mismatch)."""
    dev, neg = 0.0, np.inf
    for s in range(0, inp.n, chunk):
        f, z = inp.feat[s:s + chunk].float(), inp.z6[s:s + chunk]
        re = torch.nn.functional.linear(f, torch.from_numpy(W6), torch.from_numpy(b6))
        dev = max(dev, float((re - z).abs().max()))
    if inp.n > 1:
        f, z = inp.feat[:-1].float(), inp.z6[1:]
        neg = 0.0
        for s in range(0, inp.n - 1, chunk):
            re = torch.nn.functional.linear(f[s:s + chunk], torch.from_numpy(W6), torch.from_numpy(b6))
            neg = max(neg, float((re - z[s:s + chunk]).abs().max()))
    return {"max_abs_featcheck": dev, "tol": tol, "negative_control_next_token_max_abs": neg, "negative_control_reads_mismatch": bool(neg > 10 * tol), "pass": bool(dev <= tol and neg > 10 * tol)}   # MUTATION-TARGET-SRDEV-FEATCHECK


def c0_identity(inp: DevInputs, W6: np.ndarray, b6: np.ndarray, chunk: int = 512) -> dict:
    """C0 (PREREG_SELRANK_ARM.md:69): with the TTC row UNCHANGED (P2's own row) the logits are bit-identical to the stored ones and the dev pick equals P2's pick on every token.
    The gain on identical picks is exactly 0.0 by construction (checked again on the outcomes, `c0_gain_zero`)."""
    ttc = {TTC_COL: (W6[TTC_COL].copy(), b6[TTC_COL:TTC_COL + 1].copy())}
    z_id = apply_rows(inp.z6, inp.feat, ttc, W6, b6, chunk)
    stored = inp.z6.numpy()
    rec = {"logits_bit_identical_ttc_row_is_p2": bool(np.array_equal(z_id, stored))}
    pk_stored = pick_logits(stored)
    rec["picks_equal_banked_p2_pick"] = int((pk_stored == inp.pick_banked).sum())
    rec["picks_equal_banked_p2_pick_all"] = bool((pk_stored == inp.pick_banked).all())
    rec["picks_identity_ttc_row"] = int((pick_logits(z_id) == inp.pick_banked).sum())
    rec["n_tokens"] = inp.n
    rec["pass"] = bool(rec["logits_bit_identical_ttc_row_is_p2"] and rec["picks_equal_banked_p2_pick_all"] and rec["picks_identity_ttc_row"] == inp.n)   # MUTATION-TARGET-SRDEV-C0
    return rec


def c1_applier_reach(inp: DevInputs, rows_by_seed: dict, W6: np.ndarray, b6: np.ndarray, chunk: int = 512) -> dict:
    """C1, applier side ('built, tested, unreachable from its caller'): the fitted row REACHES the logits and the pick, and reaches nothing else.
    For every seed: the TTC column differs on > 0 entries; the other five columns are bit-identical (exactly 0.0 delta: the zero control).
    POSITIVE CONTROL (the probe can read a change): the NEGATED P2 TTC row must change the pick on > 0 tokens -- the line that proves logits -> pick is wired."""
    stored = inp.z6.numpy()
    others = [c for c in range(N_COMP) if c != TTC_COL]
    per_seed = {}
    ok = True
    for sd, (w, b) in rows_by_seed.items():
        z = apply_rows(inp.z6, inp.feat, {TTC_COL: (w, b)}, W6, b6, chunk)
        changed = int((z[..., TTC_COL] != stored[..., TTC_COL]).sum())
        others_equal = all(bool(np.array_equal(z[..., c], stored[..., c])) for c in others)
        per_seed[sd] = {"ttc_column_entries_changed": changed, "other_five_columns_bit_identical": others_equal,
                        "max_abs_delta_ttc_logit": float(np.abs(z[..., TTC_COL].astype(np.float64) - stored[..., TTC_COL]).max())}
        ok = ok and changed > 0 and others_equal                                                  # MUTATION-TARGET-SRDEV-C1-REACH
    z_neg = apply_rows(inp.z6, inp.feat, {TTC_COL: (-W6[TTC_COL], -b6[TTC_COL:TTC_COL + 1])}, W6, b6, chunk)
    pos = int((pick_logits(z_neg) != pick_logits(stored)).sum())                    # MUTATION-TARGET-SRDEV-C1-POS
    ok = ok and pos > 0
    return {"per_seed": per_seed, "positive_control_negated_p2_row_changed_picks": pos, "pass": bool(ok)}


def c1_recorded(r1_doc: dict) -> dict:
    """C1, training side (PREREG_SELRANK_ARM.md:70): the gradient numbers `srttc_r1.reachability_check` RECORDED in r1_result.json, RE-EVALUATED here (a result file with a 'GREEN' flag is not evidence;
    the numbers are).  non-zero finite TTC-row gradient (full, pair-only, anchor-only); EXACTLY 0.0 on the six frozen rows and the features in all four probes; the zero-weight control reads
    EXACTLY 0.0 on the TTC row; one optimiser step moved the row."""
    rep = r1_doc.get("c1_reachability")
    if not isinstance(rep, dict):
        return {"pass": False, "reasons": ["r1_result.json has no c1_reachability record"]}
    why = []
    for nm in ("full", "pair_only", "anchor_only"):
        r = rep.get(nm) or {}
        for k in ("ttc_row_w", "ttc_row_b"):
            v = r.get(k)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0):                    # MUTATION-TARGET-SRDEV-C1-GRAD
                why.append(f"{nm}.{k} = {v!r} (must be finite and > 0)")
    for nm in ("full", "pair_only", "anchor_only", "zero_control"):
        r = rep.get(nm) or {}
        for k in ("frozen_rows_w", "frozen_rows_b", "features"):
            if r.get(k) != 0.0:                                                                        # MUTATION-TARGET-SRDEV-C1-FROZEN
                why.append(f"{nm}.{k} = {r.get(k)!r} (must be EXACTLY 0.0)")
    z = rep.get("zero_control") or {}
    if z.get("ttc_row_w") != 0.0 or z.get("ttc_row_b") != 0.0:                                         # MUTATION-TARGET-SRDEV-C1-ZERO
        why.append(f"zero_control TTC-row gradient = {z.get('ttc_row_w')!r}/{z.get('ttc_row_b')!r} (the probe must read EXACTLY 0.0 with both loss weights 0)")
    mv = rep.get("update_moved_ttc_row_max_abs")
    if not (isinstance(mv, (int, float)) and mv > 0.0):
        why.append(f"update_moved_ttc_row_max_abs = {mv!r} (one optimiser step must move the row)")
    return {"pass": not why, "reasons": why, "recorded": {k: rep.get(k) for k in ("n_pairs_in_probe_batch", "update_moved_ttc_row_max_abs")}}


def c4_fan_frozen(inp: DevInputs) -> dict:
    """C4 (PREREG_SELRANK_ARM.md:73): the fan is frozen.  The dump the features came from (proposals.npz) holds EXACTLY the table's 64 proposals on every token (max abs pose diff 0.0 m), in the same token
    and log order as the table and the feature file, with the same pick -- so `pdms[token, new_pick]` is the PDMS of the pose the new pick selects.
    Negative control: against the NEXT token's proposals the diff must read > 0."""
    same_tok = bool(np.array_equal(inp.token, inp.token_feat) and np.array_equal(inp.token, inp.token_dump))
    same_log = bool(np.array_equal(inp.log, inp.log_feat) and np.array_equal(inp.log, inp.log_dump))
    d = np.abs(inp.props_table.astype(np.float64) - inp.props_dump.astype(np.float64))
    mx = float(d.max()) if d.size else 0.0
    neg = float(np.abs(inp.props_table[:-1].astype(np.float64) - inp.props_dump[1:].astype(np.float64)).reshape(inp.n - 1, -1).max(1).min()) if inp.n > 1 else float("inf")
    pick_eq = bool(np.array_equal(inp.pick_banked, inp.pick_dump))
    return {"tokens_aligned": same_tok, "logs_aligned": same_log, "proposals_max_abs_m": mx, "n_tokens_bit_identical": int((d.reshape(inp.n, -1).max(1) == 0.0).sum()),
            "picks_equal": pick_eq, "negative_control_next_token_min_max_abs_m": neg, "negative_control_reads_FAIL": bool(neg > 0.0),
            "pass": bool(same_tok and same_log and mx == 0.0 and pick_eq and neg > 0.0)}                  # MUTATION-TARGET-SRDEV-C4-EXACT


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the launch-gate token (C1, PREREG_SELRANK_ARM.md:70 / MM_DECISIONS_R1.md:24).  HMAC primitives re-implemented byte-for-byte from srttc_r1_gate.py:117-128 (pinned by a known-answer test)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def _hmac_ok(token: dict, key: bytes) -> bool:
    body = {k: v for k, v in token.items() if k != "hmac_sha256"}
    got = str(token.get("hmac_sha256", ""))
    return len(got) == 64 and hmac.compare_digest(hmac.new(key, _canonical_json(body), hashlib.sha256).hexdigest(), got)


def check_gate_token(guard: DevGuard, opts: DevOptions, r1: dict) -> dict:
    """The R1 launch-gate PASS token must exist and be BOUND to the R1 that produced r1_result.json: HMAC verifies (so it was minted by the gate on this host, never hand-written), verdict PASS,
    profile `srttc-r1`, stage `run`, the same fraction, the token's srttc_r1.py sha256 == r1_result.code_sha256, the token's bank-manifest sha256 == r1_result.inputs.bank_manifest_sha256.
    Never raises: the consequence of a failure is a RED C1 (VOID), decided by the caller.  GAP G5: the commit blobs are not re-checked."""
    doc = r1["doc"]
    path = opts.token_path(int(doc["fraction_pct"]))
    out = {"token_path": path, "required": bool(opts.require_gate_token)}
    if not opts.require_gate_token:
        out.update({"pass": True, "note": "NOT REQUIRED (test override; recorded in pins_overridden)"})
        return out
    why = []
    try:
        tok, tsha = load_json(guard, path)
        out["token_sha256"] = tsha
    except Exception as ex:                                                                 # noqa: BLE001
        out.update({"pass": False, "reasons": [f"token unreadable: {type(ex).__name__}: {str(ex)[:160]}"]})
        return out
    try:
        kt = _io_read_bytes(guard, opts.key_path()).decode("ascii").strip()
        key = bytes.fromhex(kt) if len(kt) == 64 else b""
    except Exception as ex:                                                                 # noqa: BLE001
        key = b""
        why.append(f"gate key unreadable ({type(ex).__name__}): the HMAC cannot be verified, so the token is not evidence")
    if not key:
        why.append("gate key is not 64 hex chars")
    elif not _hmac_ok(tok, key):                                                            # MUTATION-TARGET-SRDEV-TOKEN-HMAC
        why.append("HMAC does not verify: the token was edited or signed with another key")
    if tok.get("kind") != GATE_KIND or tok.get("schema") != GATE_SCHEMA:
        why.append("not a launch-gate token")
    if tok.get("verdict") != "PASS":                                                        # MUTATION-TARGET-SRDEV-TOKEN-VERDICT
        why.append(f"token verdict is {tok.get('verdict')!r}, not PASS")
    b = tok.get("binding") or {}
    if b.get("profile") != opts.gate_profile:                                               # MUTATION-TARGET-SRDEV-TOKEN-PROFILE
        why.append(f"token profile {b.get('profile')!r} != {opts.gate_profile!r} (a synthetic-profile token can never certify a real run)")
    if b.get("stage") != "run" or b.get("fraction") != doc["fraction_pct"]:
        why.append(f"token stage/fraction {b.get('stage')}/{b.get('fraction')} != run/{doc['fraction_pct']}")
    if b.get("srttc_r1_sha256") != doc.get("code_sha256"):                                  # MUTATION-TARGET-SRDEV-TOKEN-CODE
        why.append(f"token binds srttc_r1.py {str(b.get('srttc_r1_sha256'))[:12]} but r1_result.json was written by {str(doc.get('code_sha256'))[:12]}")
    if b.get("bank_manifest_sha256") != doc["inputs"].get("bank_manifest_sha256"):           # MUTATION-TARGET-SRDEV-TOKEN-MANIFEST
        why.append("token binds a different S0 bank manifest than r1_result.json used")
    out.update({"pass": not why, "reasons": why, "bound": {k: b.get(k) for k in ("profile", "commit", "stage", "fraction", "argv_sha256", "srttc_r1_sha256", "bank_manifest_sha256")},
                "created_utc": tok.get("created_utc")})
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# scoring: PDMS by lookup, paired log-cluster bootstrap (the SAME draws for every quantity)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _x100(d: dict) -> dict:
    return {k: (100.0 * v if k in ("point", "lo", "hi") else v) for k, v in d.items()}


def per_log(inp: DevInputs, v: np.ndarray) -> np.ndarray:
    return np.bincount(inp.log_idx, weights=np.asarray(v, np.float64), minlength=inp.n_logs)


def score_arm(inp: DevInputs, out: DevOutcomes, new_pick: np.ndarray, base_pick: np.ndarray, draws: np.ndarray) -> dict:
    """Paired (new pick vs base pick) TOTAL gain in PDMS x100, the six components, the unsafe-pick and zero-pick rates; every interval from the SAME log draws.
    `pool_gain_x100` is DESCRIPTIVE (the numerator is restricted to tokens whose base pick has PDMS > 0; the denominator is all tokens) so the dev gain can be set beside the reach table (PREREG :84-86)."""
    ar = np.arange(inp.n)
    den = per_log(inp, np.ones(inp.n))
    pn, pb = out.pdms[ar, new_pick], out.pdms[ar, base_pick]                         # MUTATION-TARGET-SRDEV-SCORE-LOOKUP
    nz = (pb > 0.0).astype(np.float64)
    rec = {"changed_picks": int((new_pick != base_pick).sum()), "n_tokens": inp.n,
           "new_pdms_x100": 100.0 * float(pn.mean()), "base_pdms_x100": 100.0 * float(pb.mean()),
           "gain_x100": _x100(R.paired_delta(per_log(inp, pn), per_log(inp, pb), den, draws)),          # MUTATION-TARGET-SRDEV-GAIN-SIGN: num_a = NEW, num_b = BASE
           "pool_gain_x100": _x100(R.paired_delta(per_log(inp, pn * nz), per_log(inp, pb * nz), den, draws))}
    comps = {}
    for k, nm in enumerate(COMPS):
        a, b = out.sub[ar, new_pick, k], out.sub[ar, base_pick, k]
        comps[nm] = {"new_x100": 100.0 * float(a.mean()), "base_x100": 100.0 * float(b.mean()), "delta_x100": _x100(R.paired_delta(per_log(inp, a), per_log(inp, b), den, draws))}
    rec["components"] = comps
    un, ub = (~out.safe[ar, new_pick]).astype(np.float64), (~out.safe[ar, base_pick]).astype(np.float64)
    rec["unsafe_pick_rate"] = {"new": float(un.mean()), "base": float(ub.mean()), "delta": R.paired_delta(per_log(inp, un), per_log(inp, ub), den, draws)}
    zn, zb = (pn == 0.0).astype(np.float64), (pb == 0.0).astype(np.float64)
    rec["zero_pdms_pick_rate"] = {"new": float(zn.mean()), "base": float(zb.mean()), "delta": R.paired_delta(per_log(inp, zn), per_log(inp, zb), den, draws)}
    return rec


def c0_gain_zero(inp: DevInputs, out: DevOutcomes, base_pick: np.ndarray, draws: np.ndarray) -> dict:
    """C0's outcome half (PREREG_SELRANK_ARM.md:69 'the gain reads exactly 0.0'): identical picks give a gain of EXACTLY 0.0 (point and both ends) and 0 changed picks."""
    r = score_arm(inp, out, base_pick.copy(), base_pick, draws)
    g = r["gain_x100"]
    return {"gain_point": g["point"], "gain_lo": g["lo"], "gain_hi": g["hi"], "changed_picks": r["changed_picks"], "pass": bool(g["point"] == 0.0 and g["lo"] == 0.0 and g["hi"] == 0.0 and r["changed_picks"] == 0)}   # MUTATION-TARGET-SRDEV-C0-GAIN


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the registered objects: c* result, LC call, C2 rows, the gate-set attestations
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def recompute_cstar(grid: list) -> str:
    """c* = argmax over the 9 configs of the MEAN over seeds {0,1} of the POOLED out-of-fold concordance (MM_DECISIONS_R1.md:15); ties -> the first in grid order, lr ascending then anchor
    ascending (MM_DECISIONS_R1.md:20).  -> the config id."""
    order = [(lr, a) for lr in REG_GRID_LR for a in REG_GRID_ANCHOR]
    by = {(g["lr"], g["anchor"]): g for g in grid}
    best, best_v = None, -math.inf
    for key in order:
        g = by[key]
        v = float(np.mean([g["per_seed"][s]["pooled"] for s in SEED_KEYS]))
        if v > best_v:
            best, best_v = key, v
    return f"lr{best[0]:g}_anchor{best[1]:g}"


def check_r1_result(guard: DevGuard, opts: DevOptions, W6: np.ndarray, b6: np.ndarray) -> dict:
    """Everything about r1_result.json that must hold BEFORE a dev outcome is opened.  -> {doc, sha256, rows: {seed: (w, b, sha)}, star}"""
    path = os.path.join(opts.run_dir, R1_RESULT)
    if not _io_exists(guard, path):
        raise DevRefusal(f"{path} does not exist: there is no c* to gate")
    doc, sha = load_json(guard, path)
    if sha != opts.registered_result_sha256:                                                     # MUTATION-TARGET-SRDEV-REGSHA
        raise DevRefusal(f"r1_result.json sha256 {sha} != the REGISTERED {opts.registered_result_sha256}: c* and its sha256 must be written down before the dev gate is read (PREREG_SELRANK_ARM.md:8)")
    if _need(doc, "schema") != R1_SCHEMA or _need(doc, "stage") != "R1":
        raise DevRefusal(f"result schema/stage {doc.get('schema')}/{doc.get('stage')} != {R1_SCHEMA}/R1")
    frac = _need(doc, "fraction_pct")
    if frac not in REG_FRACTIONS:
        raise DevRefusal(f"fraction {frac} not in the registered nested sizes {REG_FRACTIONS} (amendment2.md:7)")
    p = _need(doc, "params")
    if [int(s) for s in _need(doc, "params", "seeds")] != REG_SEEDS:                             # MUTATION-TARGET-SRDEV-SEEDS
        raise DevRefusal(f"fit seeds {p['seeds']} != the registered {REG_SEEDS} (C3, PREREG_SELRANK_ARM.md:72: both seeds are required)")
    if not opts.allow_r1_pin_overrides and _need(doc, "inputs", "pins_overridden"):              # MUTATION-TARGET-SRDEV-PINS
        raise DevRefusal(f"the fit overrode registered pins {doc['inputs']['pins_overridden']}: it is not the registered object")
    if _need(doc, "code_sha256") != opts.srttc_r1_sha256:                                        # MUTATION-TARGET-SRDEV-CODESHA
        raise DevRefusal(f"the fit's code sha256 {doc['code_sha256']} != the landed srttc_r1.py {opts.srttc_r1_sha256}")
    reg_params = {"grid_lr": list(REG_GRID_LR), "grid_anchor": list(REG_GRID_ANCHOR), "k": REG_K, "epochs": opts.reg_epochs, "batch_sets": opts.reg_batch_sets, "criterion": REG_CRITERION}
    bad = {k: (p.get(k), v) for k, v in reg_params.items() if p.get(k) != v}
    if bad:                                                                                      # MUTATION-TARGET-SRDEV-PARAMS
        raise DevRefusal(f"r1_result.json params differ from the registered ones (found, registered): {bad}  [PREREG_SELRANK_ARM.md:6,:39; MM_DECISIONS_R1.md:14-15]")
    cs = _need(doc, "c_star")
    star = R.Cfg(lr=cs["lr"], anchor=cs["anchor"], epochs=p["epochs"], batch_sets=p["batch_sets"])
    if star.sha256() != cs["config_sha256"]:                                                     # MUTATION-TARGET-SRDEV-CFGSHA
        raise DevRefusal("c* does not reproduce its recorded config sha256")
    if (cs["lr"], cs["anchor"]) not in {(lr, a) for lr in REG_GRID_LR for a in REG_GRID_ANCHOR}:   # MUTATION-TARGET-SRDEV-GRID
        raise DevRefusal(f"c* (lr {cs['lr']}, anchor {cs['anchor']}) is not one of the registered 9 configurations (PREREG_SELRANK_ARM.md:6)")
    grid = _need(doc, "grid")
    if sorted((g["lr"], g["anchor"]) for g in grid) != sorted((lr, a) for lr in REG_GRID_LR for a in REG_GRID_ANCHOR):
        raise DevRefusal("the recorded grid is not the registered 9 configurations")
    if recompute_cstar(grid) != cs["id"]:                                                        # MUTATION-TARGET-SRDEV-CSTAR
        raise DevRefusal(f"c* {cs['id']} is not the argmax of the recorded grid ({recompute_cstar(grid)}): the result file is inconsistent with its own rule (MM_DECISIONS_R1.md:15,:20)")
    want_logs = opts.expect_fit_logs if opts.expect_fit_logs is not None else LOGS_BY_FRACTION[frac]
    if _need(doc, "inputs", "n_logs") != want_logs:                                              # MUTATION-TARGET-SRDEV-NLOGS
        raise DevRefusal(f"the fit used {doc['inputs']['n_logs']} logs; the registered S_{frac} has {want_logs} (amendment2.md:98-101)")
    p2_row_sha = sha256_bytes(R.row_bytes(W6[TTC_COL], b6[TTC_COL:TTC_COL + 1]))
    if _need(doc, "inputs", "p2_row_sha256") != p2_row_sha:                                      # MUTATION-TARGET-SRDEV-P2ROW
        raise DevRefusal("the recorded P2 TTC-row sha256 differs from the TTC row of THIS checkpoint: the rows were fitted from a different P2")
    fr = _need(doc, "final_rows")
    rows = {}
    for sd in SEED_KEYS:
        ent = fr.get(sd)
        if not ent:
            raise DevRefusal(f"result lacks final_rows[{sd}]")
        w, b, rsha = load_row(guard, os.path.join(opts.run_dir, ent["file"]), ent["sha256"])
        rows[sd] = (w, b, rsha)
    return {"doc": doc, "sha256": sha, "rows": rows, "star": star, "p2_row_sha256": p2_row_sha, "c_star_at_grid_edge": {"lr_upper": bool(cs["lr"] == REG_GRID_LR[-1]), "lr_lower": bool(cs["lr"] == REG_GRID_LR[0]),
                                                  "anchor_upper": bool(cs["anchor"] == REG_GRID_ANCHOR[-1]), "anchor_lower": bool(cs["anchor"] == REG_GRID_ANCHOR[0])}}


def lc_call_registered(delta_by_seed: dict) -> dict:
    """amendment2.md:169-173 + MM_DECISIONS_R1.md:19.  RISING: lower bound of delta > 0 on BOTH seeds.  FLAT: upper bound < 0.010 on BOTH seeds.  INCONCLUSIVE: anything else.
    An interval wholly inside (0, 0.010) satisfies both: FLAT (the registered both-rule)."""
    rising = all(delta_by_seed[s]["lo"] > 0.0 for s in SEED_KEYS)
    flat = all(delta_by_seed[s]["hi"] < LC_DELTA_FLAT for s in SEED_KEYS)
    if rising and flat:
        call = "FLAT"
    elif rising:
        call = "RISING"
    elif flat:
        call = "FLAT"
    else:
        call = "INCONCLUSIVE"
    return {"call": call, "rising": rising, "flat": flat}


def check_lc(guard: DevGuard, opts: DevOptions, r1: dict) -> dict:
    """A2.4 items 3-4 (amendment2.md:169-177, :182): the learning-curve call at THIS size is banked before the gate, and the gate is read only when the call is FLAT or the size is f_max.
    The call is RECOMPUTED from the recorded intervals; a recorded call that disagrees is refused.  RISING/INCONCLUSIVE below f_max: the registered action is to scale up, so the gate is NOT read."""
    path = os.path.join(opts.run_dir, LC_RESULT)
    if not _io_exists(guard, path):                                                              # MUTATION-TARGET-SRDEV-LC-REQUIRED
        raise DevRefusal(f"{path} does not exist: the LC call at this size is banked before the gate is read (amendment2.md:177, :182). Nothing was read.")
    lc, sha = load_json(guard, path)
    d = r1["doc"]
    if lc.get("schema") != R1_SCHEMA or lc.get("stage") != "LC" or lc.get("fraction_pct") != d["fraction_pct"]:
        raise DevRefusal("lc_result.json is not the LC stage of this fraction")
    if lc.get("bank_manifest_sha256") != d["inputs"]["bank_manifest_sha256"] or (lc.get("c_star") or {}).get("config_sha256") != d["c_star"]["config_sha256"]:
        raise DevRefusal("lc_result.json was produced on a different bank or a different c* than r1_result.json")
    if lc.get("prefixes_pct") != LC_PREFIXES:
        raise DevRefusal(f"lc_result.json prefixes {lc.get('prefixes_pct')} != the registered {LC_PREFIXES} (amendment2.md:164)")
    bs = lc.get("bootstrap") or {}
    if bs.get("B") != BOOT_B or bs.get("seed") != BOOT_SEED:
        raise DevRefusal("lc_result.json bootstrap is not 10,000 resamples at seed 20260927 (amendment2.md:166)")
    dl = lc.get("delta_100_minus_50") or {}
    if set(dl) != set(SEED_KEYS):
        raise DevRefusal("lc_result.json lacks the LC interval of a seed (both seeds are required: amendment2.md:169)")
    mine = lc_call_registered(dl)
    rec = (lc.get("call") or {}).get("call")
    if rec != mine["call"]:                                                                      # MUTATION-TARGET-SRDEV-LC-CALL
        raise DevRefusal(f"lc_result.json records the call {rec!r} but its own intervals give {mine['call']!r}: inconsistent")
    frac = d["fraction_pct"]
    allowed = bool(mine["call"] == "FLAT" or frac >= F_MAX)                                      # MUTATION-TARGET-SRDEV-LC-SCALEUP
    if not allowed:
        raise DevRefusal(f"LC call {mine['call']} at f = {frac} % < f_max = {F_MAX} %: the registered action is to scale to the next nested size (amendment2.md:176); the dev gate is NOT read. Nothing was read.")
    return {"path": path, "sha256": sha, "call": mine["call"], "rising": mine["rising"], "flat": mine["flat"], "fraction_pct": frac, "f_max": F_MAX,
            "delta_100_minus_50": {s: dl[s] for s in SEED_KEYS}, "p100_reproduces_r1": lc.get("p100_reproduces_r1")}


def check_c2_files(guard: DevGuard, opts: DevOptions, r1: dict) -> dict:
    """C2 is a bar condition (PREREG_SELRANK_ARM.md:93): the shuffled-label rows and their result must exist, and belong to THIS c* and THIS bank.  Written by `srttc_c2.py`."""
    path = os.path.join(opts.run_dir, C2_RESULT)
    if not _io_exists(guard, path):
        raise DevRefusal(f"{path} does not exist: C2 (the shuffled-label negative control) is a dev-gate condition (PREREG_SELRANK_ARM.md:71, :93); run `srttc_c2.py` first. Nothing was read.")
    doc, sha = load_json(guard, path)
    if _need(doc, "schema") != R1_SCHEMA or _need(doc, "stage") != "C2" or _need(doc, "c_star", "config_sha256") != r1["doc"]["c_star"]["config_sha256"]:   # MUTATION-TARGET-SRDEV-C2BIND
        raise DevRefusal("r1_c2_result.json is not the C2 stage of this c*")
    if _need(doc, "bank_manifest_sha256") != r1["doc"]["inputs"]["bank_manifest_sha256"]:
        raise DevRefusal("r1_c2_result.json was fitted on a different bank than r1_result.json")
    if _need(doc, "fraction_pct") != r1["doc"]["fraction_pct"]:
        raise DevRefusal("r1_c2_result.json is for a different fraction than r1_result.json")
    rows = {}
    for sd in SEED_KEYS:
        ent = _need(doc, "final_rows", sd)
        w, b, rsha = load_row(guard, os.path.join(opts.run_dir, ent["file"]), ent["sha256"])
        rows[sd] = (w, b, rsha)
    return {"sha256": sha, "rows": rows, "doc": doc, "shuffle_seed": doc.get("shuffle_seed")}


def check_a210(guard: DevGuard, opts: DevOptions, dev_sha: dict) -> dict:
    """A2.10 (amendment2.md:245-248): the gate is admissible only if ALL FOUR hold; if any fails the dev gate is NOT read.
       (1) P2HO_DONE reports PARTIAL false and 1,705 tokens / 3,137 sets;  (2) every gate of the held-out package PASSES;  (3) the 333 first-run tokens recompute bit-identically;
       (4) the table is landed with its sha256 (MANIFEST.sha256.json equals the pins AND the files in the dev dir)."""
    d = opts.p2ho_dir
    n_tok = opts.expect_tokens if opts.expect_tokens is not None else DEV_N_TOKENS
    n_log = opts.expect_logs if opts.expect_logs is not None else DEV_N_LOGS
    done, done_sha = load_json(guard, os.path.join(d, "P2HO_DONE"))
    rc, rc_sha = load_json(guard, os.path.join(d, "recompute_check_333.json"))
    man, man_sha = load_json(guard, os.path.join(d, "MANIFEST.sha256.json"))
    cov, pr = done.get("coverage") or {}, done.get("proofs") or {}
    why = []
    if done.get("PARTIAL") is not False:                                                         # MUTATION-TARGET-SRDEV-A210-PARTIAL
        why.append(f"(1) P2HO_DONE.PARTIAL is {done.get('PARTIAL')!r}, must be false")
    if done.get("n_tokens") != n_tok or cov.get("n_tokens_banked") != n_tok or cov.get("n_tokens_blocked") != 0:
        why.append(f"(1) tokens: n_tokens {done.get('n_tokens')}, banked {cov.get('n_tokens_banked')}, blocked {cov.get('n_tokens_blocked')} (need {n_tok}, {n_tok}, 0)")
    if cov.get("n_logs_banked") != n_log:
        why.append(f"(1) logs banked {cov.get('n_logs_banked')} != {n_log}")
    si = done.get("sets_index") or {}
    if opts.expect_sets is not None and (si.get("n_sets") != opts.expect_sets or si.get("n_sets_with_row") != opts.expect_sets):
        why.append(f"(1) sets: {si.get('n_sets')}/{si.get('n_sets_with_row')} != {opts.expect_sets}")
    proofs_ok = (str(pr.get("disjointness", "")).endswith("PASS") and pr.get("fidelity_A") is True and pr.get("dgdump") is True and pr.get("G3") is True
                 and pr.get("G4a_pick_from_logits") == f"{n_tok}/{n_tok}" and isinstance(pr.get("G4b_pdms_formula_max_abs_diff"), (int, float))
                 and pr["G4b_pdms_formula_max_abs_diff"] <= FORMULA_TOL)
    if not proofs_ok:                                                                            # MUTATION-TARGET-SRDEV-A210-PROOFS
        why.append(f"(2) a gate of the held-out package is not PASS: {json.dumps(pr)[:400]}")
    zeros = all(rc.get(k) == 0.0 for k in ("logits_max_abs", "proposals_max_abs", "pdms_max_abs", "sub_max_abs", "pdms_at_pick_max_abs"))
    if not (rc.get("PASS") is True and rc.get("n_compared") == N_RECOMPUTE_CHECK and rc.get("pick_equal") == N_RECOMPUTE_CHECK and rc.get("log_equal") is True and zeros):   # MUTATION-TARGET-SRDEV-A210-333
        why.append(f"(3) recompute_check_333 is not a bit-identical PASS over {N_RECOMPUTE_CHECK} tokens: {json.dumps(rc)[:400]}")
    files = man.get("files") or {}
    for n in DEV_FILES:
        landed = (files.get(n) or {}).get("sha256")
        if landed != opts.dev_pins.get(n) or landed != dev_sha.get(n):                           # MUTATION-TARGET-SRDEV-A210-MANIFEST
            why.append(f"(4) {n}: landed manifest {str(landed)[:12]} / pin {str(opts.dev_pins.get(n))[:12]} / file {str(dev_sha.get(n))[:12]} disagree")
    if why:
        raise DevRefusal("A2.10 (amendment2.md:248): the dev gate is NOT read. " + " | ".join(why))
    return {"dir": d, "p2ho_done_sha256": done_sha, "recompute_check_333_sha256": rc_sha, "manifest_sha256": man_sha, "n_tokens": done["n_tokens"], "n_sets": si.get("n_sets"),
            "n_logs": cov["n_logs_banked"], "all_four": True}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the verdict and the outcome rows (committed in advance)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def bar_check(gains: dict) -> dict:
    """Paired total gain >= +0.30 PDMS x100 on BOTH seeds (point estimates; 24 logs support no lower-bound rule): PREREG_SELRANK_ARM.md:93."""
    per = {sd: bool(gains[sd] >= DEV_BAR_X100) for sd in SEED_KEYS}                              # MUTATION-TARGET-SRDEV-BAR
    return {"bar_x100": DEV_BAR_X100, "per_seed_meets": per, "both_seeds": bool(all(per[sd] for sd in SEED_KEYS))}      # MUTATION-TARGET-SRDEV-BOTH-SEEDS


def verdict_of(controls_pass: bool, bar: dict) -> str:
    if not controls_pass:                                                                        # MUTATION-TARGET-SRDEV-VERDICT
        return "VOID"
    return "PASS" if bar["both_seeds"] else "FAIL"


def outcome_rows(verdict: str, lc: dict) -> dict:
    """The rows the DEV gate can decide: PREREG_SELRANK_ARM.md:98-103 (section 7) and amendment2.md:206-207 (A2.6).  Each names its next arm (Rule Zero); nothing is promoted."""
    fired, notes = [], {}
    call, frac = lc["call"], lc["fraction_pct"]
    if verdict == "VOID":
        fired.append("VOID")
    elif verdict == "PASS":
        fired.append("DEV-PASS")
        if call != "FLAT":
            notes["a2_6_row1_iv"] = ("LC call not FLAT: A2.6 row 1 adds '(iv) refit at the next nested size' to ADOPT-SIGNAL / NOT PROVEN; "
                                     + ("not applicable at f = f_max = 100 %" if frac >= F_MAX else "available"))
    else:
        fired.append("DEV-FAIL")
        if call == "FLAT":
            fired.append("HEAD/FEATURE-LIMITED")
        else:
            notes["a2_6_row3"] = ("GAP G4: LC call %s at f = %s %% with f_max = 100 %%. amendment2.md:207 states the DATA-LIMITED row as 'only reachable when f_max < 100 %%'; "
                                  "it is NOT applied here and the MM must rule on it before reading this FAIL as data-limited." % (call, frac))
    nxt = {"VOID": "fix the instrument and re-gate: G3 (registration silent) means the MM decides whether the one dev read was consumed; no number from this run is read",
           "DEV-PASS": "navtest confirm, read ONCE (PREREG_SELRANK_ARM.md:94), only after BOTH dev gates are read or one arm is formally abandoned (MM_DECISIONS_EPR1.md D1)",
           "DEV-FAIL": ("feature-vs-capacity probe, CPU, minutes: a 2-layer TTC probe on the cached pre-head features with 5-fold-by-log CV; if its contender concordance stays <= 0.72 the features do not "
                        "carry the TTC signal and the lever moves to the trunk (a TTC gradient into the trunk, the P6 pattern), not the head (PREREG_SELRANK_ARM.md:102)"),
           "HEAD/FEATURE-LIMITED": "amendment2.md:206: more fit data of this kind is NOT the lever (MEASURED plateau); the subsample is NOT the explanation; run the probe above on the cached S_f features with the A2.2 folds"}
    return {"fired": fired, "next_arm": {k: nxt[k] for k in fired}, "notes": notes}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_dev(opts: DevOptions) -> dict:
    t0 = time.time()
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    guard = opts.guard()
    out_dir = opts.resolved_out_dir()
    result_path, marker_path = os.path.join(out_dir, RESULT_NAME), os.path.join(out_dir, READ_STARTED)
    # --- 0. the read is once
    for p in (result_path, marker_path):
        if _io_exists(guard, p):                                                           # MUTATION-TARGET-SRDEV-ONCE
            raise DevRefusal(f"{p} exists: the dev gate is read ONCE (the attempt consumes the read, not only its success; PREREG_SELRANK_ARM.md:9). Move the directory away only on a recorded Master Mind decision.")
    # --- 1. the code is the landed code
    if opts.srttc_r1_sha256 is not None:
        got = sha256_bytes(_io_read_bytes(guard, os.path.abspath(R.__file__)))
        if got != opts.srttc_r1_sha256:
            raise DevRefusal(f"srttc_r1.py sha256 {got} != the landed {opts.srttc_r1_sha256}: this applier is pinned to the code the rows were fitted with")
    # --- 2. the registered objects
    W6, b6 = load_p2_head(guard, opts.p2_ckpt)
    r1 = check_r1_result(guard, opts, W6, b6)
    lc = check_lc(guard, opts, r1)
    c2 = check_c2_files(guard, opts, r1)
    # --- 3. the dev INPUTS (no outcome), the gate-set admissibility, and the controls that need none
    inp = load_dev_inputs(guard, opts)
    a210 = check_a210(guard, opts, inp.provenance["sha256"]) if opts.require_p2ho else {"skipped": True, "note": "NOT REQUIRED (test override; recorded in pins_overridden)"}
    rows_ttc = {sd: (r1["rows"][sd][0], r1["rows"][sd][1]) for sd in SEED_KEYS}
    c1_rec, c1_tok, c1_app = c1_recorded(r1["doc"]), check_gate_token(guard, opts, r1), c1_applier_reach(inp, rows_ttc, W6, b6, opts.chunk)
    ctl = {"integrity_featcheck": integrity_featcheck(inp, W6, b6, opts.featcheck_tol, opts.chunk), "C0_pick_identity": c0_identity(inp, W6, b6, opts.chunk),
           "C1_reach": {"recorded_training_gradients": c1_rec, "launch_gate_token": c1_tok, "applier_reach": c1_app, "pass": bool(c1_rec["pass"] and c1_tok["pass"] and c1_app["pass"])},
           "C4_fan_frozen": c4_fan_frozen(inp)}
    red = [k for k, v in ctl.items() if not v["pass"]]
    if red:                                                                                  # MUTATION-TARGET-SRDEV-REDGATE
        raise DevRefusal(f"control(s) RED before the dev outcomes were opened: {red} -- the instrument is wrong; the read was NOT spent. Detail: {json.dumps(_jsonable({k: ctl[k] for k in red}))[:1500]}")
    # --- 4. the read is consumed from here on
    started = {"schema": SCHEMA, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "r1_result_sha256": r1["sha256"], "srttc_r1_sha256": opts.srttc_r1_sha256}
    _io_makedirs(guard, out_dir)
    _io_write_bytes(guard, marker_path, canon(started), exclusive=True)
    out = read_outcomes(guard, os.path.join(opts.dev_dir, "table_p2_heldout.npz"))
    draws = R.boot_draws(inp.n_logs)
    stored = inp.z6.numpy()
    base_pick = pick_logits(stored)
    ctl["C0_gain_zero"] = c0_gain_zero(inp, out, base_pick, draws)
    # --- 5. the arms
    per_seed, c2_per_seed = {}, {}
    for sd in SEED_KEYS:
        w, b, _sha = r1["rows"][sd]
        z_new = apply_rows(inp.z6, inp.feat, {TTC_COL: (w, b)}, W6, b6, opts.chunk)
        new_pick = pick_logits(z_new)                                                           # MUTATION-TARGET-SRDEV-PICK-NEW
        per_seed[sd] = score_arm(inp, out, new_pick, base_pick, draws)
        cw, cb, _ = c2["rows"][sd]
        zc = apply_rows(inp.z6, inp.feat, {TTC_COL: (cw, cb)}, W6, b6, opts.chunk)
        c2_per_seed[sd] = score_arm(inp, out, pick_logits(zc), base_pick, draws)
    ctl["C2_negative"] = {"per_seed_gain_x100": {sd: c2_per_seed[sd]["gain_x100"] for sd in SEED_KEYS}, "per_seed_changed_picks": {sd: c2_per_seed[sd]["changed_picks"] for sd in SEED_KEYS},
                          "rule": "the shuffled-label rows must give a dev gain <= 0 on each seed, no tolerance (PREREG_SELRANK_ARM.md:71; MM_DECISIONS_EPR1B.md D7)",
                          "vacuous_seeds_zero_changed_picks": [sd for sd in SEED_KEYS if c2_per_seed[sd]["changed_picks"] == 0], "shuffle_seed": c2["shuffle_seed"],
                          "pass": bool(all(c2_per_seed[sd]["gain_x100"]["point"] <= 0.0 for sd in SEED_KEYS))}   # MUTATION-TARGET-SRDEV-C2
    replicate = {"seeds": REG_SEEDS, "rows_distinct": bool(r1["rows"]["0"][2] != r1["rows"]["1"][2]),
                 "rule": "C3 (PREREG_SELRANK_ARM.md:72): two seeds, same flags (refused earlier if not); the claim needs BOTH seeds to clear the bar (the bar below). rows_distinct is a DISCLOSURE, not a gate"}
    controls_pass = bool(all(v["pass"] for v in ctl.values()))
    gains = {sd: per_seed[sd]["gain_x100"]["point"] for sd in SEED_KEYS}
    bar = bar_check(gains)
    bar["seed_gap_abs"] = abs(gains["0"] - gains["1"])
    bar["seed_floor_range"] = list(SEED_FLOOR_RANGE)
    bar["seed_gap_vs_0_44_floor"] = "DISCLOSURE (amendment2.md:197), not a bar: gap %.3f vs the training-seed floor range %s" % (bar["seed_gap_abs"], list(SEED_FLOOR_RANGE))
    verdict = verdict_of(controls_pass, bar)
    rows_fired = outcome_rows(verdict, lc)
    result = {
        "schema": SCHEMA, "stage": STAGE, "tier": "SELECTION (dev gate; not a claim: MM_DECISIONS_EPR1 D3)",
        "inputs": {"run_dir": opts.run_dir, "r1_result_sha256": r1["sha256"], "registered_result_sha256": opts.registered_result_sha256, "c_star": r1["doc"]["c_star"],
                   "c_star_at_grid_edge": r1["c_star_at_grid_edge"], "fraction_pct": r1["doc"]["fraction_pct"], "rows_sha256": {sd: r1["rows"][sd][2] for sd in SEED_KEYS},
                   "c2_result_sha256": c2["sha256"], "c2_rows_sha256": {sd: c2["rows"][sd][2] for sd in SEED_KEYS}, "lc": {k: lc[k] for k in ("sha256", "call", "rising", "flat", "fraction_pct", "f_max")},
                   "p2_ckpt": opts.p2_ckpt, "p2_ttc_row_sha256": r1["p2_row_sha256"], "gate_set_admissibility_a210": a210, "dev": inp.provenance, "dev_formula_check": out.formula,
                   "srttc_r1_sha256": opts.srttc_r1_sha256, "pins_overridden": opts.pins_overridden()},
        "estimator": {"name": "paired log-cluster bootstrap (srttc_r1.boot_draws / paired_delta == zero_tail.Boot draws)", "B": BOOT_B, "seed": BOOT_SEED, "n_logs": inp.n_logs, "n_tokens": inp.n,
                      "answers": "another draw of 24 LOGS only. The two fit seeds answer head-fit noise (H-ESTIM-SEED-1); inference is deterministic. 24 logs support no lower-bound rule: intervals are reported, never a bar."},
        "controls": ctl, "replicate_c3": replicate, "primary": {"per_seed": per_seed, "bar": bar},
        "dev_gate": {"verdict": verdict, "bar_x100": DEV_BAR_X100, "both_seeds_meet_bar": bar["both_seeds"], "controls_all_pass": controls_pass,
                     "strategic_family": "PAUSED (PI 2026-09-19, item 25): never a pass"},
        "outcome_rows": rows_fired, "registration_trace": REGISTRATION_TRACE, "registration_gaps": REGISTRATION_GAPS,
        "run_meta": {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(time.time() - t0, 1), "torch": torch.__version__, "numpy": np.__version__,
                     "threads": torch.get_num_threads(), "chunk": opts.chunk},
    }
    blob = canon(result)
    sha = _io_write_bytes(guard, result_path, blob, exclusive=True)
    _io_write_bytes(guard, os.path.join(out_dir, SIDECAR_NAME), (sha + "  " + RESULT_NAME + "\n").encode("utf-8"), exclusive=True)
    print(f"[{time.strftime('%H:%M:%S')}] SR-TTC dev gate {verdict}: gains x100 {gains}, wrote {result_path} sha256 {sha}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SR-TTC dev-gate applier: apply c*'s two TTC rows to P2's held-out pre-head features, PDMS by lookup, paired log-cluster bootstrap. Read ONCE.")
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--run-dir", required=True, help="SR-TTC R1's run dir (r1_result.json, ttc_row_seed{0,1}.npy, lc_result.json, r1_c2_result.json, ttc_row_c2shuffle_seed{0,1}.npy)")
    ap.add_argument("--registered-result-sha256", required=True, help="the sha256 of r1_result.json written down BEFORE this read (PREREG_SELRANK_ARM.md:8)")
    ap.add_argument("--r1-gate-token", default="", help="default C:/Users/Admin/srttc_r1/gate/f<fraction>_run/r1_gate_token.json")
    ap.add_argument("--gate-key-file", default="", help="default $TANITAD_GATE_KEY_FILE or ~/.config/tanitad/launch_gate.key")
    ap.add_argument("--p2ho-dir", default=P2HO_DIR, help="the landed raw/2026-10-18-p2-heldout-table package (A2.10 attestations)")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--dev-dir", default=DEV_DIR)
    ap.add_argument("--p2-ckpt", default=DEFAULT_P2_CKPT)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    opts = DevOptions(run_dir=a.run_dir, registered_result_sha256=a.registered_result_sha256, dev_dir=a.dev_dir, p2_ckpt=a.p2_ckpt, out_dir=a.out_dir, p2ho_dir=a.p2ho_dir,
                      r1_gate_token=a.r1_gate_token, gate_key_file=a.gate_key_file)
    try:
        run_dev(opts)
    except (DevRefusal, DevGuardError) as ex:
        print(f"REFUSED: {ex}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
