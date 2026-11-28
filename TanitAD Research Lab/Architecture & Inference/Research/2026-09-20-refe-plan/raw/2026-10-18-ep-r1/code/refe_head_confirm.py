#!/usr/bin/env python3
"""The SHARED navtest applier for the REFE head rows (MM_DECISIONS_EPR1 D1: ONE navtest open for the whole head programme).  CPU only.  It serves SR-TTC alone, EP-R1 alone and the JOINT (EP-R1
row + SR-TTC's TTC row of the same seed) in one look, each against ITS OWN registered bar.  Nothing is fitted here.

INPUTS   the refit rows (`ep_row_seed{0,1}.npy` from EP-R1's `epr1.py`, `ttc_row_seed{0,1}.npy` from SR-TTC's `srttc_r1.py`), a navtest FEATURE BANK (P2's `score_head` INPUT, `[N,64,256]` fp16, one
         file per chunk, from `srttc_s0_dump.py` in its fan path, with the matching dump for the C4 proof; see NAVTEST_BANK_SCOPING.md), and the landed `table_p2_full.npz` (sha256 pinned).
WHAT IT DOES per reading  recomputes the six logits with ONLY the named columns replaced (`epr1.apply_row`: stored + feat . (row - P2's row)), applies the planner's rule
         NC*DAC*(5EP+5TTC+2C)/12 over sigmoids (`planner.py:558-561`; `epr1.picks_np`, float64, lowest index on ties; it reproduces the planner's float32 pick on 12,146 / 12,146 navtest tokens), and writes a
         W3-format SEAM per reading (`token, fingerprint, poses[N,8,3], sampling, arm`; poses = the table's executed pose of the new pick), ready for `eval/score_navtest_refe.py` (the harness cross-check)
         and for `families6`.  The fan is frozen (C4), so the PDMS of a pick is the lookup `pdms[token, pick]` in the table.

TWO STAGES, and the look is spent only by the second
  picks   no outcome array is opened (a test runs it on a table that has no `pdms` / `sub` at all).  Writes the seams, the picks, the controls, `ready_for_score.json`.  Repeatable.
  read    SPENDS THE LOOK: exclusive-creates the ledger file, THEN opens `pdms` / `sub`, does the lookups, the paired log-cluster bootstrap (the SAME draws for every quantity), the registered bar clauses
          and the PREREG s8 outcome triggers.  The ledger is outside the out dir by design: one open for the whole programme.  A second `read` is refused.

IT REFUSES TO START (before it opens anything but the small JSON files)
  * unless BOTH dev-gate result files exist (EP-R1's `epr1_dev_result.json`, SR-TTC's), OR a formal-abandonment file exists for ONE arm (MM_DECISIONS_EPR1 D1).  Both abandoned: nothing to read.
  * for an arm whose dev result says FAIL or VOID, or whose arm is abandoned: that arm's ALONE reading is not opened (EP-R1 is read iff EP-R1 alone passed its dev gate; SR-TTC follows its own registration).
    If no reading is open, the look stays unspent and this refuses.
  * unless every row file hashes to the sha256 the dev result recorded (the rows read now are the rows that passed), the result files hash to the REGISTERED sha256, and the P2 head matches.
  * unless the feature bank is complete (every navtest token once), every chunk's sha256 is in the bank manifest, P2's head reproduces the stored logits from the features (<= 2e-3), and the C4 dump's
    proposals equal the table's exactly.

CLI
    refe_head_confirm.py picks|read --ep-run-dir D --ep-registered-result-sha256 H --ttc-run-dir D --ttc-registered-result-sha256 H --ttc-dev-result F --bank-dir D --out-dir O [--look-ledger F (read)]
                         [--ep-abandon F] [--ttc-abandon F] [--table T] [--p2-ckpt P] [--banked-p2-seam S]
    refe_head_confirm.py manifest --bank-dir D          (hash every *.npz of the bank into MANIFEST.sha256.json)
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import os
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import epr1 as R          # noqa: E402
import epr1_dev as D      # noqa: E402

SCHEMA = "refe-head-confirm-v1"
NAVTEST_TABLE = "D:/Projects/TanitAD-artifacts/p2_full_table/table_p2_full.npz"
NAVTEST_TABLE_SHA256 = "33441082b7044c951e3cef5fb898cac50797039aed93560b2712e3ff4c34906e"      # raw/2026-10-16-p2-full-table/MANIFEST.sha256.json
NAVTEST_BANKED_SEAM = "D:/Projects/TanitAD/data/refe_navtest/seams/stage2_p2_full.npz"
N_TOKENS, N_LOGS = 12146, 136
EP_COL, TTC_COL = R.EP_COL, R.TTC_COL
SEED_KEYS = D.SEED_KEYS
BAR_X100 = 0.30                       # PREREG_EPR1 s7 and PREREG_SELRANK_ARM s6: navtest total paired gain point >= +0.30
FEATCHECK_TOL = D.FEATCHECK_TOL
DUMP_LOGIT_TOL = 1e-4                 # the landed DG-DUMP logit tolerance on the GPU (MEASURED 0.0)
LOOK_FILE = "NAVTEST_LOOK_SPENT.json"
BANK_MANIFEST = "MANIFEST.sha256.json"
FEAT_SUFFIX, DUMP_SUFFIX = "_feat.npz", "_dump.npz"
COLUMNS = {"P2": (), "EPR1": (EP_COL,), "SRTTC": (TTC_COL,), "JOINT": (EP_COL, TTC_COL)}
DEV_VERDICTS = ("PASS", "FAIL", "VOID")
ARMS = ("EP-R1", "SR-TTC")


class ConfirmRefusal(RuntimeError):
    """A registered precondition is not met.  Nothing outcome-bearing was opened."""


class LookNotOpen(ConfirmRefusal):
    """The navtest look is not open (a dev gate is unread, or no arm passed)."""


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# guard + the ONLY file-system functions (this module's own `_io_*`; the generic ones are epr1_dev's, called with this module's guard)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
FORBIDDEN_TOKENS = (".sets.npz", "s1_read_cache", "ranked_fit_logs")      # MUTATION-TARGET-CONFIRM-FORBID      # the fit bank is never read here; navtest IS this module's business
FORBIDDEN_SEGMENTS = ("srttc_s0",)       # the S0 run dir, matched as a whole PATH SEGMENT: a substring match refused this module's own seams (`seam_SRTTC_s0.npz`), MEASURED by the first test run


class ConfirmGuard(D.DevGuard):
    @staticmethod
    def forbidden(path: str) -> list:
        forms = {str(path).replace("\\", "/").lower(), D._cmp(path), D._cmp(os.path.realpath(path))}
        hit = {t for f in forms for t in FORBIDDEN_TOKENS if t in f}
        hit |= {sg for f in forms for sg in FORBIDDEN_SEGMENTS if sg in f.split("/")}
        return sorted(hit)


def _io_listdir(guard, d: str) -> list:
    guard.check_read(d)
    return sorted(os.listdir(d))


def _io_getsize(guard, path: str) -> int:
    guard.check_read(path)
    return os.path.getsize(path)


_io_read_bytes, _io_sha256, _io_npz, _io_exists, _io_makedirs, _io_write_bytes, _io_torch_load = (D._io_read_bytes, D._io_sha256, D._io_npz, D._io_exists, D._io_makedirs, D._io_write_bytes,
                                                                                                  D._io_torch_load)
sha256_bytes, canon = D.sha256_bytes, D.canon


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# options
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class ConfirmOptions:
    out_dir: str
    bank_dir: str
    ep_run_dir: str | None = None
    ep_registered_result_sha256: str | None = None
    ep_dev_result: str | None = None                     # default <ep_run_dir>/epr1_dev_result.json
    ep_abandon: str | None = None
    ttc_run_dir: str | None = None
    ttc_registered_result_sha256: str | None = None
    ttc_dev_result: str | None = None                    # SR-TTC's dev applier does not exist yet: its path is always explicit
    ttc_abandon: str | None = None
    table: str = NAVTEST_TABLE
    table_sha256: str | None = NAVTEST_TABLE_SHA256
    p2_ckpt: str = R.DEFAULT_P2_CKPT
    banked_p2_seam: str | None = None
    look_ledger: str | None = None
    expect_tokens: int | None = N_TOKENS
    expect_logs: int | None = N_LOGS
    featcheck_tol: float = FEATCHECK_TOL
    dump_logit_tol: float = DUMP_LOGIT_TOL
    require_dump: bool = True                            # C4 needs the dump; False only in tests of other checks
    epr1_sha256: str | None = D.EPR1_SHA256

    def ep_dev_path(self) -> str | None:
        return self.ep_dev_result or (os.path.join(self.ep_run_dir, D.RESULT_NAME) if self.ep_run_dir else None)

    def guard(self) -> ConfirmGuard:
        files = [self.table, self.p2_ckpt, os.path.abspath(R.__file__)] + [f for f in (self.ep_dev_path(), self.ep_abandon, self.ttc_dev_result, self.ttc_abandon, self.banked_p2_seam,
                                                                                     self.look_ledger) if f]
        dirs = [self.bank_dir, self.out_dir] + [d for d in (self.ep_run_dir, self.ttc_run_dir) if d]
        writes = [self.out_dir]
        if self.look_ledger:
            writes.append(os.path.dirname(os.path.abspath(self.look_ledger)))
        return ConfirmGuard(read_dirs=dirs, read_files=files, write_dirs=writes)

    def pins_overridden(self) -> list:
        out = []
        for k, d in (("table_sha256", NAVTEST_TABLE_SHA256), ("expect_tokens", N_TOKENS), ("expect_logs", N_LOGS), ("featcheck_tol", FEATCHECK_TOL), ("dump_logit_tol", DUMP_LOGIT_TOL),
                     ("epr1_sha256", D.EPR1_SHA256)):
            if getattr(self, k) != d:
                out.append(k)
        if not self.require_dump:
            out.append("require_dump")
        return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# is the look open?  (MM_DECISIONS_EPR1 D1)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def read_abandon(guard, path: str, arm: str) -> dict:
    """A formal abandonment: the Master Mind's record, with the register row that says why."""
    doc, sha = D.load_json(guard, path)
    if doc.get("arm") != arm:
        raise ConfirmRefusal(f"abandonment file {path} is for {doc.get('arm')!r}, not {arm!r}")
    if doc.get("abandoned") is not True:
        raise ConfirmRefusal(f"abandonment file {path}: 'abandoned' is not true")
    for k in ("register_row", "reason", "decided_by"):
        if not isinstance(doc.get(k), str) or not doc[k].strip():
            raise ConfirmRefusal(f"abandonment file {path} lacks a non-empty '{k}' (a register row saying why is part of the formal abandonment)")
    return {"status": "ABANDONED", "arm": arm, "path": path, "sha256": sha, "register_row": doc["register_row"], "reason": doc["reason"], "decided_by": doc["decided_by"]}


def read_dev_result(guard, path: str, arm: str) -> dict:
    """A dev-gate result file.  EP-R1's own is `epr1-dev-v1`.  SR-TTC's dev applier does not exist at the time of writing, so its contract is stated here: top-level `dev_gate.verdict` in
    PASS/FAIL/VOID, `inputs.rows_sha256` {"0","1"} (sha256 of ttc_row_seed{0,1}.npy as applied) and `inputs.r1_result_sha256` (sha256 of r1_result.json)."""
    doc, sha = D.load_json(guard, path)
    if arm == "EP-R1" and (doc.get("schema") != D.SCHEMA or doc.get("stage") != D.STAGE):
        raise ConfirmRefusal(f"{path} is not an EP-R1 dev-gate result (schema/stage {doc.get('schema')}/{doc.get('stage')})")
    verdict = (doc.get("dev_gate") or {}).get("verdict")
    if verdict not in DEV_VERDICTS:
        raise ConfirmRefusal(f"{path}: dev_gate.verdict {verdict!r} is not one of {DEV_VERDICTS}")
    inp = doc.get("inputs") or {}
    rows = inp.get("rows_sha256") or {}
    if set(rows) != set(SEED_KEYS):
        raise ConfirmRefusal(f"{path}: inputs.rows_sha256 must carry seeds {SEED_KEYS}")
    result_sha = inp.get("epr1_result_sha256") if arm == "EP-R1" else inp.get("r1_result_sha256")
    if not result_sha:
        raise ConfirmRefusal(f"{path}: inputs.{'epr1_result_sha256' if arm == 'EP-R1' else 'r1_result_sha256'} is missing")
    return {"status": "READ", "arm": arm, "path": path, "sha256": sha, "verdict": verdict, "rows_sha256": rows, "result_sha256": result_sha}


def arm_state(guard, arm: str, dev_path: str | None, abandon_path: str | None) -> dict:
    ab = bool(abandon_path) and _io_exists(guard, abandon_path)
    dv = bool(dev_path) and _io_exists(guard, dev_path)
    if ab and dv:
        raise ConfirmRefusal(f"{arm}: BOTH a dev-gate result and an abandonment file exist: ambiguous, the Master Mind resolves it before the look opens")
    if ab:
        return read_abandon(guard, abandon_path, arm)
    if not dv:                                                                                 # MUTATION-TARGET-CONFIRM-DVMISSING
        raise LookNotOpen(f"{arm}: no dev-gate result file ({dev_path}) and no formal-abandonment file ({abandon_path}). The navtest look opens only after BOTH dev gates have been read, or one arm is "
                          f"formally abandoned with a register row (MM_DECISIONS_EPR1 D1). Nothing was read.")
    return read_dev_result(guard, dev_path, arm)


def eligible_readings(ep: dict, ttc: dict) -> tuple:
    """-> (list of reading families open, {family: reason it is not}).  EP-R1 alone iff EP-R1 passed its dev gate; SR-TTC alone iff SR-TTC passed ITS; the JOINT is EP-R1's registered SECONDARY: it rides EP-R1's
    membership (EP-R1 passed) and needs an SR-TTC c* row that was dev-read (not abandoned).  P2 (nothing replaced) is the baseline whenever anything is open."""
    open_, why = [], {}
    if ep["status"] == "READ" and ep["verdict"] == "PASS":                                    # MUTATION-TARGET-CONFIRM-EPELIG
        open_.append("EPR1")
    else:
        why["EPR1"] = "EP-R1 abandoned" if ep["status"] == "ABANDONED" else f"EP-R1 dev gate {ep['verdict']}: EP-R1 is read on navtest iff EP-R1 alone passed it (PREREG_EPR1 s8)"
    if ttc["status"] == "READ" and ttc["verdict"] == "PASS":                                  # MUTATION-TARGET-CONFIRM-TTCELIG
        open_.append("SRTTC")
    else:
        why["SRTTC"] = "SR-TTC abandoned" if ttc["status"] == "ABANDONED" else f"SR-TTC dev gate {ttc['verdict']}: SR-TTC follows its own registration (navtest opens iff its dev gate passed)"
    if "EPR1" in open_ and ttc["status"] == "READ":                                           # MUTATION-TARGET-CONFIRM-JOINTELIG
        open_.append("JOINT")
    else:
        why["JOINT"] = ("the JOINT is EP-R1's secondary and EP-R1 did not pass its dev gate" if "EPR1" not in open_ else "SR-TTC has no dev-read c* row (abandoned): the secondary is not computed")
    if not any(f in open_ for f in ("EPR1", "SRTTC")):
        raise LookNotOpen("neither arm passed its dev gate (or is abandoned): the navtest look stays UNSPENT. Reasons: " + json.dumps(why))
    return ["P2"] + open_, why


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the registered objects: result files, rows, bindings to the dev result
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_arm_rows(guard, arm: str, run_dir: str, registered_sha: str | None, state: dict, W6: np.ndarray, b6: np.ndarray, col: int, result_name: str, schema: str) -> dict:
    """The arm's c* result file and its two rows, bound to the dev result: registered sha256 == the file's == the one the dev result recorded; every row hashes to what the dev gate applied;
    the P2 row recorded at fit time is THIS checkpoint's."""
    path = os.path.join(run_dir, result_name)
    if not _io_exists(guard, path):
        raise ConfirmRefusal(f"{arm}: {path} does not exist")
    doc, sha = D.load_json(guard, path)
    if not registered_sha:
        raise ConfirmRefusal(f"{arm}: no registered result sha256 given (c* and its sha256 are written down before any gate is read)")
    if sha != registered_sha:                                                                  # MUTATION-TARGET-CONFIRM-REGSHA
        raise ConfirmRefusal(f"{arm}: {result_name} sha256 {sha} != the REGISTERED {registered_sha}")
    if sha != state["result_sha256"]:                                                          # MUTATION-TARGET-CONFIRM-DEVCHAIN
        raise ConfirmRefusal(f"{arm}: {result_name} sha256 {sha} != the one its dev-gate result recorded ({state['result_sha256']}): the rows are not the rows that were gated")
    if doc.get("schema") != schema or doc.get("stage") != "R1":
        raise ConfirmRefusal(f"{arm}: result schema/stage {doc.get('schema')}/{doc.get('stage')} != {schema}/R1")
    if sha256_bytes(R.row_bytes(W6[col], b6[col:col + 1])) != (doc.get("inputs") or {}).get("p2_row_sha256"):          # MUTATION-TARGET-CONFIRM-P2ROW
        raise ConfirmRefusal(f"{arm}: the recorded P2 row sha256 differs from column {col} of THIS checkpoint: the rows were fitted from a different P2")
    rows = {}
    for sd in SEED_KEYS:
        ent = doc["final_rows"][sd]
        w, b, rsha = D.load_row(guard, os.path.join(run_dir, ent["file"]), ent["sha256"])
        if rsha != state["rows_sha256"][sd]:                                                   # MUTATION-TARGET-CONFIRM-ROWSHA
            raise ConfirmRefusal(f"{arm}: row seed {sd} sha256 {rsha} != the one the dev gate applied ({state['rows_sha256'][sd]})")
        rows[sd] = (w, b, rsha)
    return {"doc": doc, "sha256": sha, "rows": rows}


def joint_compat(ep_doc: dict, ttc_doc: dict) -> list:
    """Reasons the two rows were NOT fitted on identical logs (PREREG_EPR1 s4/s8: the joint combines rows fitted on the same sets).  Empty list = compatible."""
    out = []
    pairs = (("fraction_pct", ep_doc.get("fraction_pct"), ttc_doc.get("fraction_pct")),
             ("bank_manifest_sha256", (ep_doc.get("inputs") or {}).get("bank_manifest_sha256"), (ttc_doc.get("inputs") or {}).get("bank_manifest_sha256")),
             ("ranked_fit_logs_sha256", (ep_doc.get("inputs") or {}).get("ranked_fit_logs_sha256"), (ttc_doc.get("inputs") or {}).get("ranked_fit_logs_sha256")),
             ("replacements", (ep_doc.get("inputs") or {}).get("replacements"), (ttc_doc.get("inputs") or {}).get("replacements")))
    for nm, a, b in pairs:
        if a is None or a != b:
            out.append(f"{nm} differs or is missing ({a!r} vs {b!r})")
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the navtest table (INPUT members only) and the feature bank
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class NavTable:
    token: np.ndarray
    log: np.ndarray
    fingerprint: np.ndarray
    logits: np.ndarray            # [N,64,6] float32 (P2's stored logits)
    proposals: np.ndarray         # [N,64,8,3] float32: executed, NAVSIM-converted
    pick: np.ndarray              # [N] int64: P2's planner pick
    sha256: str
    log_names: list
    log_idx: np.ndarray

    @property
    def n(self) -> int:
        return len(self.token)


def load_table_inputs(guard, opts: ConfirmOptions) -> NavTable:
    """Opens token, log, fingerprint, logits, proposals, pick.  NEVER pdms / sub / valid / sub_names (those are the look)."""
    if not _io_exists(guard, opts.table):
        raise ConfirmRefusal(f"navtest table {opts.table} does not exist")
    sha = _io_sha256(guard, opts.table)
    if opts.table_sha256 is not None and sha != opts.table_sha256:                           # MUTATION-TARGET-CONFIRM-TABLESHA
        raise ConfirmRefusal(f"navtest table sha256 {sha} != the landed {opts.table_sha256}")
    T = _io_npz(guard, opts.table)
    token, log = np.array([str(x) for x in T["token"]]), np.array([str(x) for x in T["log"]])
    fp, logits, props, pick = np.array([str(x) for x in T["fingerprint"]]), np.array(T["logits"]), np.array(T["proposals"]), T["pick"].astype(np.int64)
    T.close()
    n = len(token)
    if opts.expect_tokens is not None and n != opts.expect_tokens:
        raise ConfirmRefusal(f"navtest table has {n} tokens, expected {opts.expect_tokens}")
    if len(set(token.tolist())) != n:
        raise ConfirmRefusal("navtest table: duplicate tokens")
    names, idx = np.unique(log, return_inverse=True)
    if opts.expect_logs is not None and len(names) != opts.expect_logs:
        raise ConfirmRefusal(f"navtest table has {len(names)} logs, expected {opts.expect_logs}")
    if logits.dtype != np.float32 or logits.shape != (n, R.N_PROPOSALS, R.N_COMP) or props.shape != (n, R.N_PROPOSALS, 8, 3) or pick.shape != (n,):
        raise ConfirmRefusal(f"navtest table shapes/dtypes: logits {logits.dtype}{logits.shape}, proposals {props.shape}, pick {pick.shape}")
    if not np.isfinite(logits).all() or not np.isfinite(props).all():
        raise ConfirmRefusal("navtest table: non-finite logit or pose")
    return NavTable(token=token, log=log, fingerprint=fp, logits=logits, proposals=props, pick=pick, sha256=sha, log_names=names.tolist(), log_idx=np.asarray(idx, np.int64))


def read_bank_manifest(guard, bank_dir: str) -> dict:
    p = os.path.join(bank_dir, BANK_MANIFEST)
    if not _io_exists(guard, p):
        raise ConfirmRefusal(f"{p} does not exist: every bank chunk must be listed with its sha256 (run `refe_head_confirm.py manifest`)")
    doc, sha = D.load_json(guard, p)
    files = {}
    for rel, ent in doc["files"].items():
        files[os.path.basename(str(ent.get("path", rel)).replace("\\", "/"))] = ent["sha256"]
    return {"files": files, "sha256": sha}


def list_bank_chunks(guard, bank_dir: str) -> list:
    names = [n for n in _io_listdir(guard, bank_dir) if n.endswith(FEAT_SUFFIX)]
    if not names:
        raise ConfirmRefusal(f"no *{FEAT_SUFFIX} chunk in {bank_dir}")
    return [n[:-len(FEAT_SUFFIX)] for n in names]


def make_bank_manifest(guard, bank_dir: str) -> dict:
    files = {}
    for n in _io_listdir(guard, bank_dir):
        if n.endswith(".npz"):
            p = os.path.join(bank_dir, n)
            files[n] = {"path": p.replace("\\", "/"), "sha256": _io_sha256(guard, p), "bytes": _io_getsize(guard, p)}
    return {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "what": "navtest P2 pre-head feature bank: sha256 of every chunk npz", "n_files": len(files), "files": files}


def iter_bank(guard, opts: ConfirmOptions, man: dict):
    """Yields (chunk name, feat dict, dump dict | None), each file hashed against the manifest BEFORE np.load."""
    for stem in list_bank_chunks(guard, opts.bank_dir):
        out = []
        for suffix, needed in ((FEAT_SUFFIX, True), (DUMP_SUFFIX, opts.require_dump)):
            fn = stem + suffix
            p = os.path.join(opts.bank_dir, fn)
            if not _io_exists(guard, p):
                if needed:
                    raise ConfirmRefusal(f"bank chunk {stem}: {fn} is missing")
                out.append(None)
                continue
            if fn not in man["files"]:
                raise ConfirmRefusal(f"bank file {fn} is not in the manifest (refused: unverified)")
            sha = _io_sha256(guard, p)
            if sha != man["files"][fn]:                                                        # MUTATION-TARGET-CONFIRM-BANKSHA
                raise ConfirmRefusal(f"bank file {fn}: sha256 {sha} != manifest {man['files'][fn]}")
            z = _io_npz(guard, p)
            out.append({k: z[k] for k in z.files})
            z.close()
        yield stem, out[0], out[1]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the pass over the bank: every reading's picks, the controls, no outcome
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def build_plan(rows: dict, families: list) -> dict:
    """{reading name: {col: (w, b)}} for the open families.  Names: P2, EPR1_s0, EPR1_s1, SRTTC_s0, ..., JOINT_s0, JOINT_s1."""
    plan = {"P2": {}}
    for fam in families:
        if fam == "P2":
            continue
        for sd in SEED_KEYS:
            r = {}
            if fam in ("EPR1", "JOINT"):
                r[EP_COL] = rows["EP-R1"][sd][:2]
            if fam in ("SRTTC", "JOINT"):
                r[TTC_COL] = rows["SR-TTC"][sd][:2]
            plan[f"{fam}_s{sd}"] = r
    return plan


def process_bank(chunks, tab: NavTable, plan: dict, W6: np.ndarray, b6: np.ndarray, opts: ConfirmOptions, chunk: int = 512) -> dict:
    """One pass.  -> {picks: {name: [N] int16 in TABLE order}, controls...}.  `chunks` yields (stem, feat dict, dump dict | None)."""
    N = tab.n
    row_of = {t: i for i, t in enumerate(tab.token.tolist())}
    picks = {nm: np.full(N, -1, np.int16) for nm in plan}
    seen = np.zeros(N, bool)
    stats = {"featcheck_max": 0.0, "featcheck_neg_min": np.inf, "c0_logits_bit_identical": True, "c0_picks_equal_table_pick": 0, "c1_entries_changed": {nm: 0 for nm in plan},
             "c1_named_cols_only": {nm: True for nm in plan}, "c1_pos_changed_picks": 0, "c4_max_abs": 0.0, "c4_tokens_checked": 0, "c4_pick_equal": True, "c4_logit_max_abs": 0.0,
             "c4_log_equal": True, "chunks": []}
    ident = {EP_COL: (W6[EP_COL].copy(), b6[EP_COL:EP_COL + 1].copy()), TTC_COL: (W6[TTC_COL].copy(), b6[TTC_COL:TTC_COL + 1].copy())}
    neg = {EP_COL: (-W6[EP_COL], -b6[EP_COL:EP_COL + 1])}
    for stem, F, Dp in chunks:
        tok = np.array([str(x) for x in F["token"]])
        if len(set(tok.tolist())) != len(tok):
            raise ConfirmRefusal(f"bank chunk {stem}: duplicate tokens")
        miss = [t for t in tok.tolist() if t not in row_of]
        if miss:
            raise ConfirmRefusal(f"bank chunk {stem}: {len(miss)} tokens are not navtest-table tokens (e.g. {miss[:3]})")
        rows = np.array([row_of[t] for t in tok.tolist()], np.int64)
        if seen[rows].any():                                                                   # MUTATION-TARGET-CONFIRM-DUPTOKEN
            raise ConfirmRefusal(f"bank chunk {stem}: a token already appeared in an earlier chunk")
        seen[rows] = True
        if not np.array_equal(np.array([str(x) for x in F["log"]]), tab.log[rows]):
            raise ConfirmRefusal(f"bank chunk {stem}: log names differ from the table's for the same tokens")
        feat = torch.from_numpy(np.asarray(F["prehead_features"]))
        if feat.dtype != torch.float16 or tuple(feat.shape[1:]) != (R.N_PROPOSALS, R.FEAT_DIM) or not torch.isfinite(feat.float()).all():
            raise ConfirmRefusal(f"bank chunk {stem}: features {feat.dtype}{tuple(feat.shape)} are not finite float16[n,64,256]")
        z6 = torch.from_numpy(tab.logits[rows])
        # integrity: P2's own head reproduces the stored logits from these features
        for s in range(0, len(rows), chunk):
            re = torch.nn.functional.linear(feat[s:s + chunk].float(), torch.from_numpy(W6), torch.from_numpy(b6))
            stats["featcheck_max"] = max(stats["featcheck_max"], float((re - z6[s:s + chunk]).abs().max()))
        if len(rows) > 1:
            re = torch.nn.functional.linear(feat[:-1].float(), torch.from_numpy(W6), torch.from_numpy(b6))
            stats["featcheck_neg_min"] = min(stats["featcheck_neg_min"], float((re - z6[1:]).abs().max()))
        # C0 identity: P2's own rows replace nothing
        z_id = D.apply_rows(z6, feat, ident, W6, b6, chunk)
        stats["c0_logits_bit_identical"] = bool(stats["c0_logits_bit_identical"] and np.array_equal(z_id, tab.logits[rows]))
        pk_id = D.pick_logits(z_id)
        stats["c0_picks_equal_table_pick"] += int((pk_id == tab.pick[rows]).sum())
        # the readings
        stored = tab.logits[rows]
        for nm, named in plan.items():
            z = D.apply_rows(z6, feat, named, W6, b6, chunk) if named else z_id
            picks[nm][rows] = D.pick_logits(z).astype(np.int16)
            cols = set(named)
            others = [c for c in range(R.N_COMP) if c not in cols]
            stats["c1_named_cols_only"][nm] = bool(stats["c1_named_cols_only"][nm] and all(np.array_equal(z[..., c], stored[..., c]) for c in others))
            stats["c1_entries_changed"][nm] += int(sum(int((z[..., c] != stored[..., c]).sum()) for c in cols))
        zp = D.apply_rows(z6, feat, neg, W6, b6, chunk)
        stats["c1_pos_changed_picks"] += int((D.pick_logits(zp) != D.pick_logits(stored)).sum())
        # C4: the dump the features came from holds the table's proposals
        if Dp is not None:
            dt = np.array([str(x) for x in Dp["token"]])
            if not np.array_equal(dt, tok):
                raise ConfirmRefusal(f"bank chunk {stem}: the dump's token order differs from the feature file's")
            d = np.abs(np.asarray(Dp["proposals"], np.float64) - tab.proposals[rows].astype(np.float64))
            stats["c4_max_abs"] = max(stats["c4_max_abs"], float(d.max()) if d.size else 0.0)
            stats["c4_pick_equal"] = bool(stats["c4_pick_equal"] and np.array_equal(Dp["pick"].astype(np.int64), tab.pick[rows]))
            stats["c4_logit_max_abs"] = max(stats["c4_logit_max_abs"], float(np.abs(Dp["logits"].astype(np.float64) - stored.astype(np.float64)).max()))
            stats["c4_log_equal"] = bool(stats["c4_log_equal"] and np.array_equal(np.array([str(x) for x in Dp["log"]]), tab.log[rows]))
            stats["c4_tokens_checked"] += len(rows)
        stats["chunks"].append({"chunk": stem, "n": int(len(rows))})
    if not seen.all():                                                                         # MUTATION-TARGET-CONFIRM-INCOMPLETE
        raise ConfirmRefusal(f"the feature bank is INCOMPLETE: {int((~seen).sum())} of {N} navtest tokens have no features (first {tab.token[~seen][:3].tolist()})")
    stats["picks"] = picks
    stats["n_tokens_covered"] = int(seen.sum())
    return stats


def controls_from_stats(stats: dict, tab: NavTable, plan: dict, opts: ConfirmOptions) -> dict:
    N = tab.n
    pk_planner = np.concatenate([D.pick_logits(tab.logits[s:s + 1024]) for s in range(0, N, 1024)])      # chunked: float64 sigmoid of the whole 12,146 x 64 x 6 table is ~1 GB of temporaries
    c0_rule = bool((pk_planner == tab.pick).all())
    feat = {"max_abs_featcheck": stats["featcheck_max"], "tol": opts.featcheck_tol, "negative_control_next_token_min_max_abs": stats["featcheck_neg_min"],
            "pass": bool(stats["featcheck_max"] <= opts.featcheck_tol and stats["featcheck_neg_min"] > 10 * opts.featcheck_tol)}
    c0 = {"table_logits_rule_pick_equals_planner_pick": int((pk_planner == tab.pick).sum()), "n_tokens": N, "logits_bit_identical_with_p2_rows": stats["c0_logits_bit_identical"],
          "picks_with_p2_rows_equal_planner_pick": stats["c0_picks_equal_table_pick"],
          "p2_reading_equals_planner_pick": bool((stats["picks"]["P2"].astype(np.int64) == tab.pick).all()),
          "pass": bool(c0_rule and stats["c0_logits_bit_identical"] and stats["c0_picks_equal_table_pick"] == N and (stats["picks"]["P2"].astype(np.int64) == tab.pick).all())}   # MUTATION-TARGET-CONFIRM-C0
    c1 = {"per_reading": {nm: {"entries_changed_in_named_columns": stats["c1_entries_changed"][nm], "other_columns_bit_identical": stats["c1_named_cols_only"][nm]} for nm in plan if nm != "P2"},
          "positive_control_negated_p2_ep_row_changed_picks": stats["c1_pos_changed_picks"]}
    reach = bool(all(v["entries_changed_in_named_columns"] > 0 and v["other_columns_bit_identical"] for v in c1["per_reading"].values()))   # MUTATION-TARGET-CONFIRM-C1
    c1["pass"] = bool(reach and stats["c1_pos_changed_picks"] > 0 and stats["c1_named_cols_only"].get("P2", True))
    if opts.require_dump:
        c4 = {"tokens_checked": stats["c4_tokens_checked"], "proposals_max_abs_m": stats["c4_max_abs"], "picks_equal": stats["c4_pick_equal"], "logs_equal": stats["c4_log_equal"],
              "dump_logits_vs_table_max_abs": stats["c4_logit_max_abs"], "dump_logit_tol": opts.dump_logit_tol,
              "pass": bool(stats["c4_tokens_checked"] == N and stats["c4_max_abs"] == 0.0 and stats["c4_pick_equal"] and stats["c4_log_equal"] and stats["c4_logit_max_abs"] <= opts.dump_logit_tol)}   # MUTATION-TARGET-CONFIRM-C4
    else:
        c4 = {"pass": False, "note": "dump not required (test mode)"}
    return {"integrity_featcheck": feat, "C0_identity": c0, "C1_reach": c1, "C4_fan_frozen": c4}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# seams
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def seam_arrays(tab: NavTable, pick: np.ndarray, arm: str) -> dict:
    ar = np.arange(tab.n)
    return {"token": tab.token, "fingerprint": tab.fingerprint, "poses": tab.proposals[ar, pick.astype(np.int64)].astype(np.float32), "sampling": np.array([8, 0.5]), "arm": np.array(arm)}


def npz_bytes(arrays: dict) -> bytes:
    """In-memory only: the bytes of an .npz.  The file is written by `_io_write_bytes`."""
    bio = io.BytesIO()
    np.savez(bio, **arrays)
    return bio.getvalue()


def label_of(name: str) -> str:
    return "refe_head_" + name.lower()


def write_outputs(guard, opts: ConfirmOptions, tab: NavTable, stats: dict, plan: dict, ctl: dict, extra: dict) -> dict:
    out = opts.out_dir
    seam_dir = os.path.join(out, "seams")
    _io_makedirs(guard, seam_dir)
    seams = {}
    for nm in plan:
        blob = npz_bytes(seam_arrays(tab, stats["picks"][nm], f"REFE_HEAD_{nm}"))
        p = os.path.join(seam_dir, f"seam_{nm}.npz")
        seams[nm] = {"path": p.replace("\\", "/"), "sha256": _io_write_bytes(guard, p, blob, exclusive=True), "label": label_of(nm),
                     "changed_picks_vs_p2": int((stats["picks"][nm] != stats["picks"]["P2"]).sum()), "picks_sha256": sha256_bytes(stats["picks"][nm].tobytes())}
    ready = {"what": "seams ready for eval/score_navtest_refe.py (labels MUST start with 'refe'); the picks are table lookups, the harness score is the independent instrument cross-check",
             "WARNING": "scoring a seam through the harness READS THE NAVTEST OUTCOMES. Run these commands only AFTER the `read` stage has spent the shared look (MM_DECISIONS_EPR1 D1); before it, they would be a second, unledgered look.",
             "readings": [{"reading": nm, "label": seams[nm]["label"], "seam": seams[nm]["path"],
                           "command": f"python eval/score_navtest_refe.py --label {seams[nm]['label']} --seam {seams[nm]['path']}"} for nm in plan]}
    _io_write_bytes(guard, os.path.join(out, "ready_for_score.json"), canon(ready), exclusive=True)
    return seams


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the registered setup, shared by both stages
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def prepare(opts: ConfirmOptions) -> dict:
    guard = opts.guard()
    if opts.epr1_sha256 is not None:
        got = sha256_bytes(_io_read_bytes(guard, os.path.abspath(R.__file__)))
        if got != opts.epr1_sha256:
            raise ConfirmRefusal(f"epr1.py sha256 {got} != the landed {opts.epr1_sha256}")
    # 1. is the look open?
    ep = arm_state(guard, "EP-R1", opts.ep_dev_path(), opts.ep_abandon)
    ttc = arm_state(guard, "SR-TTC", opts.ttc_dev_result, opts.ttc_abandon)
    if ep["status"] == "ABANDONED" and ttc["status"] == "ABANDONED":
        raise LookNotOpen("both arms are formally abandoned: there is nothing to read")
    families, why = eligible_readings(ep, ttc)
    # 2. the registered rows, bound to the dev results
    W6, b6 = D.load_p2_head(guard, opts.p2_ckpt)
    arm_rows, docs = {}, {}
    if "EPR1" in families or "JOINT" in families:
        a = load_arm_rows(guard, "EP-R1", opts.ep_run_dir, opts.ep_registered_result_sha256, ep, W6, b6, EP_COL, "epr1_result.json", R.SCHEMA)
        arm_rows["EP-R1"], docs["EP-R1"] = a["rows"], a
    if "SRTTC" in families or "JOINT" in families:
        a = load_arm_rows(guard, "SR-TTC", opts.ttc_run_dir, opts.ttc_registered_result_sha256, ttc, W6, b6, TTC_COL, "r1_result.json", "srttc-r1-v1")
        arm_rows["SR-TTC"], docs["SR-TTC"] = a["rows"], a
    if "JOINT" in families:
        bad = joint_compat(docs["EP-R1"]["doc"], docs["SR-TTC"]["doc"])
        if bad:                                                                                # MUTATION-TARGET-CONFIRM-COMPAT
            families.remove("JOINT")
            why["JOINT"] = "EP-R1 and SR-TTC were not fitted on identical logs: " + "; ".join(bad)
    plan = build_plan(arm_rows, families)
    return {"guard": guard, "ep": ep, "ttc": ttc, "families": families, "why_not": why, "W6": W6, "b6": b6, "arm_rows": arm_rows, "docs": docs, "plan": plan}


def run_picks_core(opts: ConfirmOptions, prep: dict) -> dict:
    guard = prep["guard"]
    tab = load_table_inputs(guard, opts)
    man = read_bank_manifest(guard, opts.bank_dir)
    stats = process_bank(iter_bank(guard, opts, man), tab, prep["plan"], prep["W6"], prep["b6"], opts)
    ctl = controls_from_stats(stats, tab, prep["plan"], opts)
    banked = None
    if opts.banked_p2_seam:
        z = _io_npz(guard, opts.banked_p2_seam)
        bt = {str(t): i for i, t in enumerate(z["token"])}
        rows = [bt.get(str(t), -1) for t in tab.token]
        if min(rows) < 0:
            banked = {"performed": True, "pass": False, "reason": "tokens missing from the banked seam"}
        else:
            ours = seam_arrays(tab, stats["picks"]["P2"], "x")["poses"]
            d = float(np.abs(ours.astype(np.float64) - z["poses"][rows].astype(np.float64)).max())
            fp_ok = bool(all(str(z["fingerprint"][r]) == tab.fingerprint[i] for i, r in enumerate(rows)))
            banked = {"performed": True, "max_abs_pose_m": d, "fingerprints_equal": fp_ok, "pass": bool(d == 0.0 and fp_ok)}
        z.close()
        ctl["banked_p2_seam"] = banked
    return {"tab": tab, "man": man, "stats": stats, "ctl": ctl}


def controls_pass(ctl: dict) -> bool:
    return bool(all(v.get("pass", False) for v in ctl.values()))


def stage_picks(opts: ConfirmOptions) -> dict:
    t0 = time.time()
    guard = opts.guard()
    if _io_exists(guard, os.path.join(opts.out_dir, "confirm_picks_report.json")):
        raise ConfirmRefusal(f"{opts.out_dir}/confirm_picks_report.json exists: use a fresh output directory")
    prep = prepare(opts)
    core = run_picks_core(opts, prep)
    if not controls_pass(core["ctl"]):                                                                  # MUTATION-TARGET-CONFIRM-CTLGATE
        red = [k for k, v in core["ctl"].items() if not v.get("pass", False)]
        raise ConfirmRefusal(f"control(s) RED: {red}. No seam is written. Detail: {json.dumps(D._jsonable({k: core['ctl'][k] for k in red}))[:1500]}")
    seams = write_outputs(guard, opts, core["tab"], core["stats"], prep["plan"], core["ctl"], {})
    rep = {"schema": SCHEMA, "stage": "PICKS", "look": {"spent": False, "note": "no outcome array was opened (pdms / sub / valid)"}, "inputs": inputs_record(opts, prep, core),
           "readings_open": prep["families"], "readings_not_open": prep["why_not"], "controls": core["ctl"], "seams": seams,
           "pick_agreement_with_p2": {nm: int((core["stats"]["picks"][nm] == core["stats"]["picks"]["P2"]).sum()) for nm in prep["plan"]},
           "run_meta": {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(time.time() - t0, 1), "torch": torch.__version__, "numpy": np.__version__}}
    _io_write_bytes(guard, os.path.join(opts.out_dir, "confirm_picks_report.json"), canon(rep), exclusive=True)
    _io_write_bytes(guard, os.path.join(opts.out_dir, "picks.npz"), npz_bytes({"token": core["tab"].token, **core["stats"]["picks"]}), exclusive=True)
    print(f"[{time.strftime('%H:%M:%S')}] picks stage done: readings {prep['families']}, seams in {opts.out_dir}/seams", flush=True)
    return rep


def inputs_record(opts: ConfirmOptions, prep: dict, core: dict) -> dict:
    return {"navtest_table": opts.table, "navtest_table_sha256": core["tab"].sha256, "bank_dir": opts.bank_dir, "bank_manifest_sha256": core["man"]["sha256"], "n_bank_chunks": len(core["stats"]["chunks"]),
            "p2_ckpt": opts.p2_ckpt, "ep_dev": {k: prep["ep"].get(k) for k in ("status", "verdict", "sha256", "path", "register_row")},
            "ttc_dev": {k: prep["ttc"].get(k) for k in ("status", "verdict", "sha256", "path", "register_row")},
            "rows_sha256": {arm: {sd: v[2] for sd, v in rows.items()} for arm, rows in prep["arm_rows"].items()},
            "result_sha256": {arm: d["sha256"] for arm, d in prep["docs"].items()}, "epr1_sha256": opts.epr1_sha256, "pins_overridden": opts.pins_overridden()}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the read: SPENDS THE LOOK
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def adverse(iv: dict) -> bool:
    """An interval wholly below 0."""
    return bool(iv["hi"] < 0.0)


def bar_clauses(rec: dict, family: str) -> dict:
    """The registered navtest clauses for ONE seed's reading vs P2.  EP-R1 (PREREG_EPR1 s7): total point >= +0.30, lower > 0, no NC/DAC/TTC/C/DDC wholly below 0, the EP component not wholly below 0.
    SR-TTC (PREREG_SELRANK_ARM s6): total point >= +0.30, lower > 0, no sub-score of the six wholly below 0.  families6 (longitudinal / lateral) is evaluated OUTSIDE this module (PENDING)."""
    g = rec["gain_x100"]
    comps = rec["components"]
    names = ("nc", "dac", "ttc", "c", "ddc") if family == "EPR1" else D.COMPS
    c = {"total_point_ge_bar": bool(g["point"] >= BAR_X100), "total_lower_gt_0": bool(g["lo"] > 0.0),                # MUTATION-TARGET-CONFIRM-BAR
         "no_subscore_wholly_below_0": bool(not any(adverse(comps[nm]["delta_x100"]) for nm in names)),
         "adverse_subscores": [nm for nm in D.COMPS if adverse(comps[nm]["delta_x100"])]}
    if family == "EPR1":
        c["ep_component_not_wholly_below_0"] = bool(not adverse(comps["ep"]["delta_x100"]))
    c["clears_without_families6"] = bool(c["total_point_ge_bar"] and c["total_lower_gt_0"] and c["no_subscore_wholly_below_0"] and c.get("ep_component_not_wholly_below_0", True))
    return c


def navtest_rows_ep(rec: dict) -> list:
    """PREREG_EPR1 s8 navtest triggers fired by ONE seed's EP-R1 reading (several can fire; none is promoted, and the families6 clause is outside).  E-ADOPT is `clears_without_families6` of BOTH seeds
    (decided by the caller), and is only CANDIDATE until families6 is read."""
    g = rec["gain_x100"]
    c = rec["components"]
    fired = []
    if g["lo"] > 0.0 and g["point"] < BAR_X100:
        fired.append("E-SMALL")
    if g["point"] > 0.0 and g["lo"] <= 0.0:
        fired.append("E-NOTPROVEN")
    ep_up = bool(c["ep"]["delta_x100"]["lo"] > 0.0)
    if ep_up and any(adverse(c[nm]["delta_x100"]) for nm in ("ttc", "nc", "dac")):
        fired.append("E-TRADE")
    if (not ep_up) and g["point"] <= 0.0:
        fired.append("E-MECH")
    if g["hi"] < 0.0:
        fired.append("E-REFUTED")
    return fired


def stage_read(opts: ConfirmOptions) -> dict:
    t0 = time.time()
    guard = opts.guard()
    if not opts.look_ledger:
        raise ConfirmRefusal("`read` needs --look-ledger: ONE ledger file outside the out dir records the single navtest look of the whole head programme")
    if _io_exists(guard, opts.look_ledger):
        raise ConfirmRefusal(f"{opts.look_ledger} exists: the navtest look is SPENT (MM_DECISIONS_EPR1 D1: one open for the whole head programme). A second read needs a recorded Master Mind decision.")
    if _io_exists(guard, os.path.join(opts.out_dir, "confirm_picks_report.json")):
        raise ConfirmRefusal(f"{opts.out_dir}/confirm_picks_report.json exists: `read` writes its own seams; use a fresh output directory")
    prep = prepare(opts)
    core = run_picks_core(opts, prep)                                           # still no outcome
    if not controls_pass(core["ctl"]):                                                                  # MUTATION-TARGET-CONFIRM-CTLGATE-READ
        red = [k for k, v in core["ctl"].items() if not v.get("pass", False)]
        raise ConfirmRefusal(f"control(s) RED before the look was spent: {red}. The look is UNSPENT.")
    seams = write_outputs(guard, opts, core["tab"], core["stats"], prep["plan"], core["ctl"], {})
    # ---- THE LOOK IS SPENT HERE, before pdms / sub are touched
    _io_makedirs(guard, os.path.dirname(os.path.abspath(opts.look_ledger)))
    ledger = {"schema": SCHEMA, "spent": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "readings": prep["families"], "table_sha256": core["tab"].sha256, "bank_manifest_sha256": core["man"]["sha256"],
              "rows_sha256": {arm: {sd: v[2] for sd, v in rows.items()} for arm, rows in prep["arm_rows"].items()}, "out_dir": opts.out_dir,
              "picks_sha256": {nm: sha256_bytes(core["stats"]["picks"][nm].tobytes()) for nm in prep["plan"]}}
    _io_write_bytes(guard, opts.look_ledger, canon(ledger), exclusive=True)                                  # MUTATION-TARGET-CONFIRM-LEDGER
    out = D.read_outcomes(guard, opts.table)
    tab = core["tab"]
    nav = types.SimpleNamespace(n=tab.n, log_idx=tab.log_idx, n_logs=len(tab.log_names))
    draws = R.boot_draws(nav.n_logs)
    P = core["stats"]["picks"]
    base = P["P2"].astype(np.int64)
    readings = {}
    for nm in prep["plan"]:
        if nm == "P2":
            continue
        rec = D.score_arm(nav, out, P[nm].astype(np.int64), base, draws)
        fam = nm.split("_")[0]
        if fam in ("EPR1", "SRTTC"):
            rec["bar_clauses"] = bar_clauses(rec, fam)
        if fam == "EPR1":
            rec["navtest_rows_fired"] = navtest_rows_ep(rec)
        if fam == "JOINT":
            sd = nm.split("_s")[1]
            rec["vs_ep_r1_alone"] = D.score_arm(nav, out, P[nm].astype(np.int64), P[f"EPR1_s{sd}"].astype(np.int64), draws)["gain_x100"]
            rec["vs_ttc_alone"] = D.score_arm(nav, out, P[nm].astype(np.int64), P[f"SRTTC_s{sd}"].astype(np.int64), draws)["gain_x100"] if f"SRTTC_s{sd}" in P else None
            rec["label"] = "SECONDARY (PREREG_EPR1 s7/s8): never decides the primary; bar clauses below are informational"
            rec["bar_clauses_informational"] = bar_clauses(rec, "EPR1")
        readings[nm] = rec
    primaries = {}
    for fam, arm in (("EPR1", "EP-R1"), ("SRTTC", "SR-TTC")):
        if fam in prep["families"]:
            both = all(readings[f"{fam}_s{sd}"]["bar_clauses"]["clears_without_families6"] for sd in SEED_KEYS)                    # MUTATION-TARGET-CONFIRM-BOTH
            primaries[arm] = {"both_seeds_clear_all_clauses_except_families6": bool(both),
                              "per_seed": {sd: readings[f"{fam}_s{sd}"]["bar_clauses"] for sd in SEED_KEYS},
                              "families6": "PENDING: run families6 on seams/seam_" + fam + "_s{0,1}.npz; the longitudinal / lateral 'separated adverse' clause is NOT evaluated here. Verdict is a CANDIDATE until then.",
                              "tactical": "reported by families6", "strategic": "PAUSED (PI 2026-09-19, item 25): never a pass", "distance_keeping": "UNAVAILABLE until lead-agent tracks reach families6.py"}
    for arm, other, ofam in (("EP-R1", "SR-TTC", "SRTTC"), ("SR-TTC", "EP-R1", "EPR1")):                 # MM_DECISIONS_EPR1 D1: a report quoting one primary states the other beside it
        if arm in primaries:
            primaries[arm]["other_primary_beside_it"] = ({other: primaries[other]["both_seeds_clear_all_clauses_except_families6"]} if other in primaries
                                                         else {other: "not read: " + str(prep["why_not"].get(ofam, "not eligible"))})
    res = {"schema": SCHEMA, "stage": "READ", "tier": "T1 NAVSIM v1.1 navtest, self-action open loop (EVAL_DOCTRINE)", "look": {"spent": True, "ledger": opts.look_ledger, "ledger_sha256": sha256_bytes(canon(ledger))},
           "inputs": inputs_record(opts, prep, core), "readings_open": prep["families"], "readings_not_open": prep["why_not"], "controls": core["ctl"],
           "estimator": {"name": "paired log-cluster bootstrap (epr1.boot_draws / paired_delta == zero_tail.Boot draws)", "B": R.BOOT_B, "seed": R.BOOT_SEED, "n_logs": nav.n_logs, "n_tokens": nav.n,
                         "answers": "another draw of 136 LOGS only. Two fit seeds per row answer head-fit noise (H-ESTIM-SEED-1); inference is deterministic; one P2 training seed."},
           "p2_pick": {"pdms_x100": 100.0 * float(out.pdms[np.arange(nav.n), base].mean()), "dev_note": "P2's own pick, the paired baseline of every reading"},
           "readings": readings, "primaries": primaries,
           "multiplicity_disclosure": "ONE navtest open reads up to three registered rows (SR-TTC alone, EP-R1 alone, the JOINT), each against its own bar, with no family-wise correction (MM_DECISIONS_EPR1 D1). "
                                      "Any report quoting either primary states the other reading and its outcome beside it.",
           "seams": seams, "run_meta": {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(time.time() - t0, 1)}}
    _io_write_bytes(guard, os.path.join(opts.out_dir, "confirm_read_result.json"), canon(res), exclusive=True)
    _io_write_bytes(guard, os.path.join(opts.out_dir, "confirm_picks_report.json"), canon({"schema": SCHEMA, "stage": "PICKS(in read)", "controls": core["ctl"], "seams": seams}), exclusive=True)
    print(f"[{time.strftime('%H:%M:%S')}] LOOK SPENT; read stage done: {opts.out_dir}/confirm_read_result.json", flush=True)
    return res


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Shared navtest applier for the REFE head rows (SR-TTC alone, EP-R1 alone, the joint) in ONE look. CPU only.")
    ap.add_argument("cmd", choices=("picks", "read", "manifest"))
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--ep-run-dir", default=None)
    ap.add_argument("--ep-registered-result-sha256", default=None)
    ap.add_argument("--ep-dev-result", default=None)
    ap.add_argument("--ep-abandon", default=None)
    ap.add_argument("--ttc-run-dir", default=None)
    ap.add_argument("--ttc-registered-result-sha256", default=None)
    ap.add_argument("--ttc-dev-result", default=None)
    ap.add_argument("--ttc-abandon", default=None)
    ap.add_argument("--table", default=NAVTEST_TABLE)
    ap.add_argument("--p2-ckpt", default=R.DEFAULT_P2_CKPT)
    ap.add_argument("--banked-p2-seam", default=NAVTEST_BANKED_SEAM)
    ap.add_argument("--look-ledger", default=None)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "manifest":
        g = ConfirmGuard(read_dirs=[a.bank_dir], write_dirs=[a.bank_dir])
        man = make_bank_manifest(g, a.bank_dir)
        _io_write_bytes(g, os.path.join(a.bank_dir, BANK_MANIFEST), canon(man), exclusive=True)
        print(f"wrote {a.bank_dir}/{BANK_MANIFEST}: {man['n_files']} files")
        return 0
    opts = ConfirmOptions(out_dir=a.out_dir or os.path.join(a.bank_dir, "confirm_out"), bank_dir=a.bank_dir, ep_run_dir=a.ep_run_dir, ep_registered_result_sha256=a.ep_registered_result_sha256,
                          ep_dev_result=a.ep_dev_result, ep_abandon=a.ep_abandon, ttc_run_dir=a.ttc_run_dir, ttc_registered_result_sha256=a.ttc_registered_result_sha256,
                          ttc_dev_result=a.ttc_dev_result, ttc_abandon=a.ttc_abandon, table=a.table, p2_ckpt=a.p2_ckpt, banked_p2_seam=a.banked_p2_seam, look_ledger=a.look_ledger)
    try:
        {"picks": stage_picks, "read": stage_read}[a.cmd](opts)
    except (ConfirmRefusal, D.DevRefusal, D.DevGuardError) as ex:
        print(f"REFUSED: {ex}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
