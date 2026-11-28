"""EP-R1 (+ the S2 head fit it needs): fit ONLY the EP row of P2's `score_head` as a CONDITIONAL expectation (EP given NC = DAC = 1) on the cached S0 features; choose `c*` by 5-fold grouped-by-log CV.

REGISTERED SOURCE: PREREG_EPR1.md (this package).  Mirrors SR-TTC R1 (`srttc_r1.py`, `MM_DECISIONS_R1.md`) wherever the registered text does not differ; the generic machinery below
(PathGuard, the `_io_*` functions, folds, manifest, frozen-rule helpers, bootstrap, LC call) is BYTE-IDENTICAL to the template and pinned by `template_pins.json` (a test fails if it drifts).

WHAT IT DOES
  run : (1) verifies every bank unit's sha256 against bank/MANIFEST.sha256.json BEFORE np.load  (2) builds the folds from the registered ranked_fit_logs.tsv
        (3) C1 + C6 reachability / conditional-target controls on a real batch (aborts if RED)  (4) for each of the 9 registered (lr, anchor) configurations x seeds x 5 folds fits the EP row
        (257 parameters: weight[2,:] and bias[2] of P2's score_head, initialised from P2's checkpoint) with the registered loss  (5) scores the out-of-fold SAFE-CONTENDER concordance
        and the pick-level read  (6) picks c*  (7) refits c* on ALL fit-set logs for each seed and writes the rows.  Writes epr1_result.json; refuses to overwrite one (c* is chosen once).
  lc  : the A2.4-style learning-curve read at c*, on the FIT set only.  Writes epr1_lc_result.json.   (no authority over S0 scale-up)
  c2  : the shuffled-label negative control at c*: EP labels shuffled among the SAFE proposals of each set, same fit.  Writes epr1_c2_result.json and the shuffled rows.

THE LOSS (PREREG_EPR1 s3).  safe_i = NC_i >= 1 and DAC_i >= 1.  Anchor: mean over SAFE proposals of BCE-with-logits(EP logit, EP label).  Pair term: over the FROZEN top-8 contenders (bank membership,
  never refreshed) pairs (i, j) with safe_i, safe_j and y_i > y_j, softplus(-(z_i - z_j)), mean over the batch's pairs.  total = pair_w * pair + anchor * anchor_term.
  The EP logit of an UNSAFE proposal receives no gradient (C6).  Frozen and PROVED frozen: every other row of score_head, the features.

NEVER READS the dev-gate (P2's held-out table) or the navtest tables.  Every file access goes through `_io_*` below, behind a PathGuard that (a) refuses any path containing a forbidden token
  and (b) allows only the bank dir, the plan dir, the named input files and the output dir.  tests (e) enforce both, and enforce by AST that no other function touches the file system.

CLI
    epr1.py run --fraction 10 [--out-dir D] [--epochs 10] [--batch-sets 256] [--seeds 0,1] [--criterion pooled_oof|fold_mean] [--threads 8]
    epr1.py lc  --fraction 10 [--out-dir D]
    epr1.py c2  --fraction 10 [--out-dir D]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import math
import os
import sys
import time

import numpy as np
import torch

SCHEMA = "epr1-v1"

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# REGISTERED constants (each with its source).  Changing one of these changes the experiment.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
NC_COL, DAC_COL, EP_COL, TTC_COL, C_COL, DDC_COL = 0, 1, 2, 3, 4, 5   # S0_HANDOFF + the bank header (label_order = nc, dac, ep, ttc, c, ddc); EP = column 2
SAFE_MIN = 1.0                                # PREREG_EPR1 s3: safe = NC >= 1.0 and DAC >= 1.0 on the clipped labels (NC = 0.5 is NOT safe)
TARGET_ID = "ep_given_nc_dac_eq_1"            # recorded in every config sha256
C2_SHUFFLE_SEED = 20261018                    # PREREG_EPR1 s6 C2: the within-set shuffle of EP labels among SAFE proposals
LABEL_ORDER = ("nc", "dac", "ep", "ttc", "c", "ddc")
N_PROPOSALS, FEAT_DIM, N_COMP = 64, 256, 6    # S0_HANDOFF: feat [S,64,256] fp16
K = 8                                         # section 2: K = 8 fixed (the bank's top8 width)
N_FOLDS = 5                                   # R1: 5-fold grouped-by-log CV
FOLD_SALT = "SR-TTC/A2/v1"                    # A2.1 item 3
N_FIT_LOGS = 1167                             # A2.1 item 2
FRACTIONS = (10, 25, 50, 100)                 # A2.1 item 4 (percent)
GRID_LR = (1e-4, 3e-4, 1e-3)                  # section 2 grid
GRID_ANCHOR = (0.03, 0.1, 0.3)                # section 2 grid
SEEDS = (0, 1)                                # A2.4 item 2 / C3: two training seeds
MIN_LOGS_PER_FOLD = 20                        # A2.2
LC_PREFIXES = (25, 50, 100)                   # A2.4 item 2 (percent of each fold's training logs, key order)
LC_DELTA_FLAT = 0.010                         # A2.4 item 3
BOOT_B = 10000                                # A2.4 item 2 / C5
BOOT_SEED = 20260927                          # A2.4 item 2 / C5 (zero_tail.Boot)
RULE = "navsim_v1"                            # the frozen selection rule NC*DAC*(5EP+5TTC+2C)/12
RANKED_LOGS_SHA256 = "a61275037a3d2b06903dc7e1ad0f6f99fb740f668a6ec7488fdef6fe901ce9b1"   # run/plan/ranked_fit_logs.tsv == raw/a2_subsample/ranked_fit_logs.tsv at the tip

# UNREGISTERED choices: named parameters (see Options), listed as ambiguities.  They are defaults, not facts.
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SETS = 256
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ROW_CONSISTENCY_TOL = 1e-2                    # |stored logit3 - (feat . p2_w + p2_b)| (S0_HANDOFF measured ~1e-3 on the held-out features)
ROW_CONSISTENCY_MAX_SETS = 2048

DEFAULT_BANK_DIR = "C:/Users/Admin/srttc_s0/work/bank"
DEFAULT_MANIFEST = DEFAULT_BANK_DIR + "/MANIFEST.sha256.json"
DEFAULT_RANKED = "C:/Users/Admin/srttc_s0/run/plan/ranked_fit_logs.tsv"
DEFAULT_S0_MARKER = "C:/Users/Admin/srttc_s0/run/S0_DONE"
DEFAULT_P2_CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"
DEFAULT_OUT_ROOT = "C:/Users/Admin/epr1/run"


class PathGuardError(RuntimeError):
    """A file access that would touch the dev-gate / navtest tables, or anything outside the allow-list."""


class BankError(RuntimeError):
    """The bank, the manifest, the folds or the P2 row failed an integrity check."""


class ReachabilityError(RuntimeError):
    """C1 went RED: the loss does not reach the TTC row, or it reaches something frozen."""


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# PathGuard + the ONLY functions allowed to touch the file system (`_io_*`; a test enforces that by AST)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
FORBIDDEN_TOKENS = ("heldout", "held_out", "held-out", "p2ho", "table_p2", "navtest", "gate_table", "dev_gate", "devgate", "dev-gate",
                    "dgdump", "fidelity", "featcheck", "s1_read_cache")


def _cmp(p: str) -> str:
    return os.path.normpath(os.path.abspath(p)).replace("\\", "/").lower()


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


class PathGuard:
    """Hard path guard.  Forbidden tokens are checked on the raw string, the absolute path and the symlink-resolved path (so `bank/../x` and a junction cannot smuggle one in)."""

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
            raise PathGuardError(f"REFUSED read of {path!r}: it names a dev-gate / navtest table token {hit} (SR-TTC R1 never reads those)")
        if not self._allowed(path, self.read_dirs, self.read_files):
            raise PathGuardError(f"REFUSED read of {path!r}: outside the allow-list (bank dir, plan dir, named inputs, output dir)")
        return path

    def check_write(self, path: str) -> str:
        hit = self.forbidden(path)
        if hit:
            raise PathGuardError(f"REFUSED write to {path!r}: forbidden token {hit}")
        if not self._allowed(path, self.write_dirs, ()):
            raise PathGuardError(f"REFUSED write to {path!r}: outside the output dir")
        return path


def _io_read_bytes(guard: PathGuard, path: str) -> bytes:
    guard.check_read(path)
    with open(path, "rb") as f:
        return f.read()


def _io_sha256(guard: PathGuard, path: str) -> str:
    guard.check_read(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _io_npz(guard: PathGuard, path: str):
    guard.check_read(path)
    return np.load(path, allow_pickle=False)


def _io_listdir(guard: PathGuard, d: str) -> list:
    guard.check_read(d)
    return sorted(os.listdir(d))


def _io_exists(guard: PathGuard, path: str) -> bool:
    guard.check_read(path)
    return os.path.exists(path)


def _io_stat(guard: PathGuard, path: str) -> tuple:
    guard.check_read(path)
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns)


def _io_torch_load(guard: PathGuard, path: str):
    guard.check_read(path)
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def _io_makedirs(guard: PathGuard, d: str) -> None:
    guard.check_write(os.path.join(d, "x"))
    os.makedirs(d, exist_ok=True)


def _io_write_bytes(guard: PathGuard, path: str, data: bytes, exclusive: bool = False) -> str:
    guard.check_write(path)
    with open(path, "xb" if exclusive else "wb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canon_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=True).encode("utf-8")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# options / config
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Cfg:
    """One training configuration.  lr and anchor are the registered grid axes; the rest are named, UNREGISTERED parameters."""
    lr: float
    anchor: float
    epochs: int = DEFAULT_EPOCHS
    batch_sets: int = DEFAULT_BATCH_SETS
    k: int = K
    pair_w: float = 1.0

    def id(self) -> str:
        return f"lr{self.lr:g}_anchor{self.anchor:g}"

    def sha256(self) -> str:
        return sha256_bytes(canon_json({**dataclasses.asdict(self), "optimizer": "adam", "betas": list(ADAM_BETAS), "eps": ADAM_EPS, "weight_decay": 0.0, "row": EP_COL, "target": TARGET_ID, "safe_min": SAFE_MIN}))


def grid(epochs: int = DEFAULT_EPOCHS, batch_sets: int = DEFAULT_BATCH_SETS) -> list:
    return [Cfg(lr=lr, anchor=a, epochs=epochs, batch_sets=batch_sets) for lr in GRID_LR for a in GRID_ANCHOR]


@dataclasses.dataclass
class Options:
    fraction: int = 10
    bank_dir: str = DEFAULT_BANK_DIR
    manifest: str = DEFAULT_MANIFEST
    ranked_logs: str = DEFAULT_RANKED
    p2_ckpt: str = DEFAULT_P2_CKPT
    out_dir: str = ""
    s0_marker: str | None = DEFAULT_S0_MARKER      # None only in tests; the CLI always sets it
    seeds: tuple = SEEDS
    epochs: int = DEFAULT_EPOCHS
    batch_sets: int = DEFAULT_BATCH_SETS
    criterion: str = "pooled_oof"                  # AMBIGUITY: R1 says "mean out-of-fold concordance", A2.4 says "pooled"; both are always reported
    both_rule: str = "flat"                        # AMBIGUITY: LC interval wholly inside (0, delta) satisfies RISING and FLAT
    min_logs_per_fold: int = MIN_LOGS_PER_FOLD
    replacements_file: str | None = None           # A2.1 item 7: {"dropped": [log...], "added": [log...]}; a decision of the Master Mind, never automatic
    threads: int = 8
    # pins; the CLI never changes them, tests do (and the result records any override)
    ranked_sha256: str | None = RANKED_LOGS_SHA256
    expected_n_logs: int | None = N_FIT_LOGS
    row_tol: float = ROW_CONSISTENCY_TOL
    reach_n_sets: int = 512

    def resolved_out_dir(self) -> str:
        return self.out_dir or f"{DEFAULT_OUT_ROOT}/f{self.fraction}"

    def guard(self) -> PathGuard:
        files = [self.manifest, self.ranked_logs, self.p2_ckpt, os.path.abspath(__file__)]
        if self.s0_marker:
            files.append(self.s0_marker)
        if self.replacements_file:
            files.append(self.replacements_file)
        return PathGuard(read_dirs=[self.bank_dir, self.resolved_out_dir()], read_files=files, write_dirs=[self.resolved_out_dir()])

    def pins_overridden(self) -> list:
        out = []
        if self.ranked_sha256 != RANKED_LOGS_SHA256:
            out.append("ranked_sha256")
        if self.expected_n_logs != N_FIT_LOGS:
            out.append("expected_n_logs")
        if self.min_logs_per_fold != MIN_LOGS_PER_FOLD:
            out.append("min_logs_per_fold")
        if self.row_tol != ROW_CONSISTENCY_TOL:
            out.append("row_tol")
        if tuple(self.seeds) != SEEDS:
            out.append("seeds")
        if self.s0_marker != DEFAULT_S0_MARKER:
            out.append("s0_marker")
        return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# folds (A2.1 / A2.2)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def fold_of_rank(rank: int) -> int:
    """A2.2: fold = rank_in_key_order mod 5."""
    return int(rank) % N_FOLDS


def log_key(log: str) -> str:
    return hashlib.sha256((FOLD_SALT + "\n" + log).encode("utf-8")).hexdigest()


def load_ranked_logs(guard: PathGuard, path: str, expected_sha256: str | None, expected_n: int | None) -> tuple:
    """-> (rows, sha256).  rows: dict(rank, log, key, fold, pct) in key order.  Verifies the pin, the key of every log, the ascending key order and fold == rank mod 5."""
    raw = _io_read_bytes(guard, path)
    sha = sha256_bytes(raw)
    if expected_sha256 is not None and sha != expected_sha256:
        raise BankError(f"ranked_fit_logs.tsv sha256 {sha} != the registered {expected_sha256}")
    lines = raw.decode("utf-8").splitlines()
    head = lines[0].split("\t")
    want = ["rank", "log", "key_sha256", "fold", "first_fraction", "n_fit_tokens", "n_fit_sets"]
    if head != want:
        raise BankError(f"ranked_fit_logs.tsv header {head} != {want}")
    rows = []
    for i, ln in enumerate(lines[1:]):
        c = ln.split("\t")
        rank, log, key, fold, ff = int(c[0]), c[1], c[2], int(c[3]), float(c[4])
        if rank != i:
            raise BankError(f"ranked_fit_logs.tsv: rank {rank} at row {i}")
        if key != log_key(log):
            raise BankError(f"ranked_fit_logs.tsv: key of {log} does not equal sha256({FOLD_SALT!r} + newline + log)")
        if rows and not rows[-1]["key"] < key:
            raise BankError("ranked_fit_logs.tsv: not in ascending key order")
        if fold != fold_of_rank(rank):
            raise BankError(f"ranked_fit_logs.tsv: fold {fold} != rank {rank} mod {N_FOLDS}")
        rows.append({"rank": rank, "log": log, "key": key, "fold": fold, "pct": int(round(ff * 100))})
    if expected_n is not None and len(rows) != expected_n:
        raise BankError(f"ranked_fit_logs.tsv has {len(rows)} logs, expected {expected_n}")
    if len({r["log"] for r in rows}) != len(rows):
        raise BankError("ranked_fit_logs.tsv: duplicate log")
    return rows, sha


def select_logs(rows: list, pct: int, replacements: dict | None) -> tuple:
    """S_f = logs with first_fraction <= f (A2.1 item 4), then the A2.1 item 7 replacements if the Master Mind supplied them.  -> (selected rows in key order, report)."""
    sel = [r for r in rows if r["pct"] <= pct]
    by_log = {r["log"]: r for r in rows}
    dropped = sorted((replacements or {}).get("dropped", []))
    added = sorted((replacements or {}).get("added", []))
    for lg in dropped:
        if lg not in {r["log"] for r in sel}:
            raise BankError(f"replacements: dropped log {lg} is not in S_{pct}")
    for lg in added:
        if lg not in by_log:
            raise BankError(f"replacements: added log {lg} is not a fit log")
        if lg in {r["log"] for r in sel}:
            raise BankError(f"replacements: added log {lg} is already in S_{pct}")
    if len(dropped) > 0.05 * len(sel):
        raise BankError(f"A2.1 item 7: {len(dropped)} replacements > 5 % of {len(sel)} selected logs; STOP and report (the sample would drift toward availability)")
    keep = [r for r in sel if r["log"] not in set(dropped)] + [by_log[lg] for lg in added]
    keep.sort(key=lambda r: r["rank"])
    return keep, {"n_selected": len(sel), "dropped": dropped, "added": added}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# bank: manifest, integrity, loading
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_manifest(guard: PathGuard, path: str) -> tuple:
    """-> ({unit basename: sha256}, sha256 of the manifest file).  Only `*.sets.npz` entries are kept; a unit basename listed twice with different sha256 is refused."""
    raw = _io_read_bytes(guard, path)
    doc = json.loads(raw.decode("utf-8"))
    table: dict = {}
    for rel, ent in doc["files"].items():
        base = os.path.basename(str(ent.get("path", rel)).replace("\\", "/"))
        if not base.endswith(".sets.npz"):
            continue    # only bank UNITS are looked up by basename; the real S0 post-step manifest also lists run/batches/<unit>/audit.json etc., one per batch, SAME basename, different content
        if base in table and table[base] != ent["sha256"]:
            raise BankError(f"manifest lists {base} twice with different sha256")
        table[base] = ent["sha256"]
    return table, sha256_bytes(raw)


def aggregate_np(z6: np.ndarray) -> np.ndarray:
    """The frozen rule, the bank dump's own expression: float64 sigmoid, p0*p1*(5p2+5p3+2p4)/12."""
    p = 1.0 / (1.0 + np.exp(-z6.astype(np.float64)))
    return p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12.0


def top8_np(z6_set: np.ndarray) -> np.ndarray:
    """One set's frozen top-8 exactly as srttc_s0_dump.py computed it: stable argsort, descending, first 8."""
    return np.argsort(-aggregate_np(z6_set), kind="stable")[:K].astype(np.int16)


def picks_np(z6: np.ndarray) -> np.ndarray:
    """[S,64,6] -> [S] pick = first element of the stable descending argsort (lowest index on ties), chunked."""
    out = np.empty(len(z6), np.int64)
    for s in range(0, len(z6), 4096):
        agg = aggregate_np(z6[s:s + 4096])
        out[s:s + 4096] = np.argsort(-agg, axis=1, kind="stable")[:, 0]
    return out


@dataclasses.dataclass
class FitData:
    log_names: list                 # selected logs in key order (index = position)
    log_rank: np.ndarray            # [L] rank in key order
    log_fold: np.ndarray            # [L]
    log_idx: np.ndarray             # [N] index into log_names
    fold: np.ndarray                # [N]
    feat: torch.Tensor              # [N,64,256] fp16
    z6: torch.Tensor                # [N,64,6] fp32 (the stored logits)
    y: torch.Tensor                 # [N,64] fp32 (EP label as banked: 0 on violators)
    safe: torch.Tensor              # [N,64] bool (NC >= 1 and DAC >= 1): the proposals that carry the conditional target
    pdms: torch.Tensor              # [N,64] fp32 NC*DAC*(5EP+5TTC+2C)/12 from the banked sub-scores (formula MEASURED to reproduce the table, ep_common.pdms_formula)
    top8: torch.Tensor              # [N,8] int64 (frozen contender membership)
    provenance: dict

    @property
    def n_sets(self) -> int:
        return int(self.feat.shape[0])

    @property
    def n_logs(self) -> int:
        return len(self.log_names)


def safe_np(lb: np.ndarray) -> np.ndarray:
    """[..., 6] clipped labels -> bool [...]: NC >= SAFE_MIN and DAC >= SAFE_MIN (the EP-1 'geometry-safe' set; NC = 0.5 is not safe)."""
    return (lb[..., NC_COL] >= SAFE_MIN) & (lb[..., DAC_COL] >= SAFE_MIN)


def pdms_np(lb: np.ndarray) -> np.ndarray:
    """NC * DAC * (5 EP + 5 TTC + 2 C) / 12 (DDC at weight 0), float32."""
    return (lb[..., NC_COL] * lb[..., DAC_COL] * (5.0 * lb[..., EP_COL] + 5.0 * lb[..., TTC_COL] + 2.0 * lb[..., C_COL]) / 12.0).astype(np.float32)


def pair_counts_np(y8: np.ndarray, s8: np.ndarray) -> np.ndarray:
    """Admitted contender pairs per set: both safe and y_i > y_j."""
    return ((s8[:, :, None] & s8[:, None, :]) & (y8[:, :, None] > y8[:, None, :])).sum((1, 2))


def load_p2_row(guard: PathGuard, ckpt: str) -> tuple:
    ck = _io_torch_load(guard, ckpt)
    sd = ck["model"]
    W, b = sd["score_head.weight"].detach().float().clone().numpy(), sd["score_head.bias"].detach().float().clone().numpy()
    if W.shape != (N_COMP, FEAT_DIM) or b.shape != (N_COMP,):
        raise BankError(f"P2 score_head is {W.shape}/{b.shape}, expected ({N_COMP}, {FEAT_DIM}) (a 7-output SFT-4 head is not P2)")
    return W, b


def load_fit_data(guard: PathGuard, opts: Options, selected: list, manifest: dict, W6: np.ndarray, b6: np.ndarray) -> FitData:
    """Verify (manifest sha256 BEFORE np.load) and load every unit, keep the sets of the selected logs, run the integrity checks."""
    sel_pos = {r["log"]: i for i, r in enumerate(selected)}
    units = [n for n in _io_listdir(guard, opts.bank_dir) if n.endswith(".sets.npz")]
    if not units:
        raise BankError(f"no *.sets.npz in {opts.bank_dir}")
    stats: dict = {}
    rows_per_unit: dict = {}
    meta: dict = {"unit_sha256": {}, "ckpt": set(), "rule": set()}
    # pass 1: sha256 against the manifest, then the small arrays only
    for n in units:
        p = os.path.join(opts.bank_dir, n)
        if n not in manifest:
            raise BankError(f"unit {n} is not in the manifest (refused: unverified)")
        sha = _io_sha256(guard, p)
        if sha != manifest[n]:
            raise BankError(f"unit {n}: sha256 {sha} != manifest {manifest[n]}")
        stats[n] = _io_stat(guard, p)
        meta["unit_sha256"][n] = sha
        z = _io_npz(guard, p)
        for key in ("log", "label_order", "rule", "ckpt"):
            if key not in z.files:
                raise BankError(f"unit {n}: missing array {key}")
        if [str(x) for x in z["label_order"]] != list(LABEL_ORDER):
            raise BankError(f"unit {n}: label_order {list(z['label_order'])} != {list(LABEL_ORDER)} (EP would not be column {EP_COL})")
        meta["rule"].add(str(z["rule"]))
        meta["ckpt"].add(str(z["ckpt"]))
        logs = z["log"]
        rows_per_unit[n] = np.array([str(x) in sel_pos for x in logs], dtype=bool)
        z.close()
    if meta["rule"] != {RULE}:
        raise BankError(f"bank rule {sorted(meta['rule'])} != the frozen {RULE!r}")
    if len(meta["ckpt"]) != 1:
        raise BankError(f"bank units were scored by different checkpoints: {sorted(meta['ckpt'])}")
    N = int(sum(m.sum() for m in rows_per_unit.values()))
    present = set()
    feat = torch.empty((N, N_PROPOSALS, FEAT_DIM), dtype=torch.float16)
    z6 = np.empty((N, N_PROPOSALS, N_COMP), np.float32)
    y = np.empty((N, N_PROPOSALS), np.float32)
    safe = np.empty((N, N_PROPOSALS), bool)
    pdms = np.empty((N, N_PROPOSALS), np.float32)
    top8 = np.empty((N, K), np.int64)
    log_idx = np.empty(N, np.int64)
    off = 0
    n_odd = 0
    n_ties_fixed = 0
    for n in units:
        mask = rows_per_unit[n]
        if not mask.any():
            continue
        p = os.path.join(opts.bank_dir, n)
        if _io_stat(guard, p) != stats[n]:
            raise BankError(f"unit {n} changed on disk between verification and load")
        z = _io_npz(guard, p)
        f, lo, tp, lb, lg = z["feat"][mask], z["logits"][mask], z["top8"][mask], z["labels6"][mask], z["log"][mask]
        z.close()
        S = len(lg)
        if f.dtype != np.float16 or f.shape[1:] != (N_PROPOSALS, FEAT_DIM) or lo.shape[1:] != (N_PROPOSALS, N_COMP) or tp.shape[1:] != (K,) or lb.shape[1:] != (N_PROPOSALS, N_COMP):
            raise BankError(f"unit {n}: shapes/dtypes {f.dtype}{f.shape} {lo.shape} {tp.shape} {lb.shape} do not match the registered bank format")
        if not (np.isfinite(lo).all() and np.isfinite(lb).all() and np.isfinite(f.astype(np.float32)).all()):
            raise BankError(f"unit {n}: non-finite feature / logit / label")
        if lb.min() < 0.0 or lb.max() > 1.0:
            raise BankError(f"unit {n}: a label outside [0,1]")
        # the frozen contender membership must be what the frozen rule says about the stored logits (vectorised, per-set fallback for a float tie)
        agg = aggregate_np(lo)
        again = np.argsort(-agg, axis=1, kind="stable")[:, :K]
        bad = np.nonzero((again != tp.astype(np.int64)).any(axis=1))[0]
        for i in bad:
            if not np.array_equal(top8_np(lo[i]), tp[i]):
                raise BankError(f"unit {n}: stored top8 of set {int(i)} is not the frozen rule applied to its stored logits")
            n_ties_fixed += 1
        feat[off:off + S] = torch.from_numpy(f)
        z6[off:off + S] = lo
        y[off:off + S] = lb[:, :, EP_COL]
        safe[off:off + S] = safe_np(lb)
        pdms[off:off + S] = pdms_np(lb)
        top8[off:off + S] = tp.astype(np.int64)
        log_idx[off:off + S] = [sel_pos[str(x)] for x in lg]
        nd = lb[:, :, (NC_COL, DAC_COL)]
        n_odd += int(((nd != 0.0) & (nd != 0.5) & (nd != 1.0)).sum())       # integrity: NC / DAC values outside {0, 0.5, 1} are COUNTED, never refused
        present.update(str(x) for x in np.unique(lg))
        off += S
    assert off == N
    missing = [r["log"] for r in selected if r["log"] not in present]
    if missing:
        raise BankError(f"{len(missing)} selected logs have no set in the bank (A2.1 item 7: supply --replacements-file or finish S0): e.g. {missing[:3]}")
    ranks = np.array([r["rank"] for r in selected], np.int64)
    data = FitData(log_names=[r["log"] for r in selected], log_rank=ranks, log_fold=np.array([fold_of_rank(r) for r in ranks], np.int64), log_idx=log_idx,
                   fold=np.array([fold_of_rank(ranks[i]) for i in log_idx], np.int64), feat=feat, z6=torch.from_numpy(z6), y=torch.from_numpy(y), safe=torch.from_numpy(safe),
                   pdms=torch.from_numpy(pdms), top8=torch.from_numpy(top8), provenance={})
    # P2-row consistency: the checkpoint row must be the row that produced the stored logits (a wrong row differs by O(1), a right one by ~1e-3)
    step = max(1, N // ROW_CONSISTENCY_MAX_SETS)
    ii = np.arange(0, N, step)[:ROW_CONSISTENCY_MAX_SETS]
    chk = data.feat[torch.from_numpy(ii)].float() @ torch.from_numpy(W6[EP_COL].copy()) + float(b6[EP_COL])
    row_dev = float((chk - data.z6[torch.from_numpy(ii), :, EP_COL]).abs().max())
    if not row_dev <= opts.row_tol:
        raise BankError(f"P2 row from {opts.p2_ckpt} does not reproduce the stored EP logits: max |feat.w+b - logit2| = {row_dev:.3e} > {opts.row_tol:g} "
                        f"(wrong checkpoint, or the bank was scored by a different head)")
    data.provenance = {"bank_ckpt": sorted(meta["ckpt"])[0], "rule": RULE, "n_units_verified": len(units), "unit_sha256": meta["unit_sha256"],
                       "row_consistency_max_abs": row_dev, "row_consistency_n_sets": int(len(ii)), "nc_dac_entries_outside_0_half_1": n_odd,
                       "top8_vectorised_tie_rechecked_sets": n_ties_fixed, "n_sets": N, "n_logs": len(selected)}
    return data


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the model, the loss, the C1 / C6 reachability and conditional-target checks
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class RowHead(torch.nn.Module):
    """P2's score_head as EP-R1 sees it.  The ONLY trainable tensors are row_w [256] and row_b [1] = weight[2,:] and bias[2], initialised from P2.
    The new EP logit is the STORED logit plus feat . (row_w - p2_w) + (row_b - p2_b): the delta is exactly 0.0 until a step is taken (C0 identity, bit-for-bit) and has the same
    gradient as the absolute parameterisation.  `probe=True` adds ALL six rows of P2's head as leaf parameters feeding the non-EP columns, so the frozen-row gradient is MEASURABLE
    (a loss that leaked into another column would then read non-zero there)."""

    def __init__(self, W6, b6, probe: bool = False, perturb: bool = False):
        super().__init__()
        W6 = torch.as_tensor(np.asarray(W6), dtype=torch.float32)
        b6 = torch.as_tensor(np.asarray(b6), dtype=torch.float32)
        self.register_buffer("p2_w", W6[EP_COL].clone())
        self.register_buffer("p2_b", b6[EP_COL:EP_COL + 1].clone())
        self.row_w = torch.nn.Parameter(W6[EP_COL].clone())
        self.row_b = torch.nn.Parameter(b6[EP_COL:EP_COL + 1].clone())
        self.probe = probe
        if perturb:   # a point where d(loss)/d(feat) is non-zero if the features were not detached (at the P2 row the delta, hence that gradient, is exactly 0)
            with torch.no_grad():
                self.row_w += 1e-2 * torch.cos(torch.arange(FEAT_DIM, dtype=torch.float32))
                self.row_b += 1e-2
        if probe:
            self.frozen_W6 = torch.nn.Parameter(W6.clone())
            self.frozen_b6 = torch.nn.Parameter(b6.clone())

    def trainable(self) -> list:
        return [self.row_w, self.row_b]


def forward_z6(head: RowHead, feat: torch.Tensor, z6_stored: torch.Tensor) -> torch.Tensor:
    """[B,64,256] features, [B,64,6] stored logits -> [B,64,6] logits with ONLY column 2 (EP) depending on the trainable row."""
    feat = feat.detach().float()                                                  # MUTATION-TARGET-DETACH: the trunk and the features are frozen
    delta = feat @ (head.row_w - head.p2_w) + (head.row_b - head.p2_b)
    z_ep = z6_stored[..., EP_COL] + delta
    if head.probe:
        z_oth = torch.nn.functional.linear(feat, head.frozen_W6, head.frozen_b6)
        cols = [z_ep if c == EP_COL else z_oth[..., c] for c in range(N_COMP)]
    else:
        cols = [z_ep if c == EP_COL else z6_stored[..., c] for c in range(N_COMP)]
    return torch.stack(cols, dim=-1)


def apply_row(z6_stored: torch.Tensor, feat: torch.Tensor, row_w, row_b, p2_w, p2_b, col: int = EP_COL) -> torch.Tensor:
    """New logits with ONLY column `col` changed: stored + feat . (row_w - p2_w) + (row_b - p2_b).  Bit-identical to the stored logits when the row equals P2's.
    The dev-gate / confirm appliers use it twice for the JOINT arm (col = EP_COL with the EP-R1 row, col = 3 with SR-TTC's TTC row)."""
    dw = torch.as_tensor(np.asarray(row_w, np.float32) - np.asarray(p2_w, np.float32))
    db = torch.as_tensor(np.asarray(row_b, np.float32) - np.asarray(p2_b, np.float32))
    out = z6_stored.clone()
    out[..., col] = out[..., col] + (feat.float() @ dw + db)
    return out


def pairwise_term(z6: torch.Tensor, y: torch.Tensor, safe: torch.Tensor, idx8: torch.Tensor) -> torch.Tensor:
    """Pairwise logistic loss on the EP logit over SAFE contender pairs with differing EP: pair (i,j) has safe_i, safe_j and y_i > y_j; loss softplus(-(z_i - z_j)); mean over the batch's pairs."""
    z = z6[..., EP_COL]
    z8, y8, s8 = torch.gather(z, 1, idx8), torch.gather(y, 1, idx8), torch.gather(safe, 1, idx8)
    both = s8.unsqueeze(2) & s8.unsqueeze(1)                                      # MUTATION-TARGET-PAIR-SAFE: both members of a pair must be safe
    hi = both & (y8.unsqueeze(2) > y8.unsqueeze(1))
    d = z8.unsqueeze(2) - z8.unsqueeze(1)
    li = torch.nn.functional.softplus(-d) * hi
    return li.sum() / hi.sum().clamp(min=1)


def anchor_term(z6: torch.Tensor, y: torch.Tensor, safe: torch.Tensor) -> torch.Tensor:
    """BCE (with logits) of the EP logit against the EP label on the SAFE proposals only (all 64 positions of every set), mean over the safe entries."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(z6[..., EP_COL], y, reduction="none")
    m = safe.to(bce.dtype)                                                        # MUTATION-TARGET-ANCHOR-SAFE: only safe proposals carry the conditional target
    return (bce * m).sum() / m.sum().clamp(min=1.0)


def total_loss(z6, y, safe, idx8, anchor_w: float, pair_w: float = 1.0):
    pair = pairwise_term(z6, y, safe, idx8)
    anchor = anchor_term(z6, y, safe)
    loss = anchor_w * anchor
    loss = loss + pair_w * pair             # MUTATION-TARGET-PAIR: the pairwise term
    return loss, pair.detach(), anchor.detach()


def _grad_report(head: RowHead, feat_leaf: torch.Tensor) -> dict:
    def n(t):
        return 0.0 if t is None else float(t.abs().sum())
    fw = head.frozen_W6.grad
    fb = head.frozen_b6.grad
    return {"ep_row_w": n(head.row_w.grad), "ep_row_b": n(head.row_b.grad), "frozen_rows_w": n(fw), "frozen_rows_b": n(fb),
            "frozen_rows_w_per_row": [0.0 if fw is None else float(fw[r].abs().sum()) for r in range(N_COMP)], "features": n(feat_leaf.grad)}


def _probe_grads(W6, b6, batch: dict, loss_fn, pair_w: float, anchor_w: float, y=None, safe=None) -> dict:
    """One backward pass at a perturbed row; -> the gradient report PLUS the raw EP-row gradients (for the bit-identity checks)."""
    head = RowHead(W6, b6, probe=True, perturb=True)
    feat = batch["feat"].clone().float().requires_grad_(True)
    z6 = forward_z6(head, feat, batch["z6"])
    loss, _p, _a = loss_fn(z6, batch["y"] if y is None else y, batch["safe"] if safe is None else safe, batch["top8"], anchor_w, pair_w)
    if loss.requires_grad:
        loss.backward()
    rep = _grad_report(head, feat)
    gw = head.row_w.grad if head.row_w.grad is not None else torch.zeros_like(head.row_w)
    gb = head.row_b.grad if head.row_b.grad is not None else torch.zeros_like(head.row_b)
    rep["_gw"], rep["_gb"] = gw.detach().clone(), gb.detach().clone()
    return rep


def reachability_check(W6, b6, batch: dict, cfg: Cfg, loss_fn=None) -> dict:
    """C1 (the 'built, tested, unreachable from its caller' class) and C6 (the conditional-target identities).  On a real batch {feat fp16, z6, y, safe, top8}:
       full    : the configured loss         -> non-zero gradient on the EP row; EXACTLY 0.0 on all six frozen rows and on the features
       pair    : pairwise term alone         -> non-zero on the EP row           (delete the pairwise line and THIS goes RED)
       anchor  : masked BCE anchor alone     -> non-zero on the EP row
       zero    : both weights 0.0 (CONTROL)  -> EXACTLY 0.0 everywhere           (a probe stuck at non-zero cannot pass this)
       update  : one real optimiser step on the real fit path moves the EP row and nothing else.
       C6 (i)  : EP labels of UNSAFE proposals replaced by other values -> the EP-row gradient is BIT-IDENTICAL   (drop either safe mask and THIS goes RED)
       C6 (ii) : a batch where every proposal is unsafe                -> the EP-row gradient is EXACTLY 0.0       (drop either safe mask and THIS goes RED)
       C6 (iii): EP labels of SAFE proposals changed                   -> the gradient CHANGES                     (positive control: (i) is not vacuous)
    Raises ReachabilityError listing every failure; returns the report when GREEN."""
    loss_fn = loss_fn or total_loss
    fails: list = []
    feat0, z60, y0, s0, idx0 = batch["feat"], batch["z6"], batch["y"], batch["safe"], batch["top8"]
    y80, s80 = torch.gather(y0, 1, idx0), torch.gather(s0, 1, idx0)
    n_pairs = int(((s80.unsqueeze(2) & s80.unsqueeze(1)) & (y80.unsqueeze(2) > y80.unsqueeze(1))).sum())
    if n_pairs == 0:
        raise ReachabilityError("the probe batch has no safe contender pair with differing EP: the pairwise term cannot be tested (choose a batch with contrast)")
    n_unsafe = int((~s0).sum())
    if n_unsafe == 0:
        raise ReachabilityError("the probe batch has no unsafe proposal: the conditional-target identities (C6) cannot be tested")
    rep: dict = {"n_pairs_in_probe_batch": n_pairs, "n_unsafe_in_probe_batch": n_unsafe}
    full, pair, anch, zero = (_probe_grads(W6, b6, batch, loss_fn, cfg.pair_w, cfg.anchor), _probe_grads(W6, b6, batch, loss_fn, cfg.pair_w, 0.0),
                              _probe_grads(W6, b6, batch, loss_fn, 0.0, 1.0), _probe_grads(W6, b6, batch, loss_fn, 0.0, 0.0))
    strip = lambda r: {k: v for k, v in r.items() if not k.startswith("_")}      # noqa: E731
    rep.update({"full": strip(full), "pair_only": strip(pair), "anchor_only": strip(anch), "zero_control": strip(zero)})
    for name, r in (("full", full), ("pair_only", pair), ("anchor_only", anch)):
        if not (r["ep_row_w"] > 0.0 and math.isfinite(r["ep_row_w"])):
            fails.append(f"{name}: gradient on the EP row weight is {r['ep_row_w']!r} (must be finite and > 0)")
        if not (r["ep_row_b"] > 0.0 and math.isfinite(r["ep_row_b"])):
            fails.append(f"{name}: gradient on the EP row bias is {r['ep_row_b']!r} (must be finite and > 0)")
    for name, r in (("full", full), ("pair_only", pair), ("anchor_only", anch), ("zero_control", zero)):
        if r["frozen_rows_w"] != 0.0 or r["frozen_rows_b"] != 0.0:
            fails.append(f"{name}: a FROZEN row of score_head received a gradient (w {r['frozen_rows_w']!r}, b {r['frozen_rows_b']!r}, per row {r['frozen_rows_w_per_row']}); must be EXACTLY 0.0")
        if r["features"] != 0.0:
            fails.append(f"{name}: the features received a gradient ({r['features']!r}); the trunk must get EXACTLY 0.0")
    if zero["ep_row_w"] != 0.0 or zero["ep_row_b"] != 0.0:
        fails.append(f"zero_control: weights (0, 0) still gave an EP-row gradient ({zero['ep_row_w']!r}); the probe is not reading the loss")
    # --- C6: the conditional target
    g = torch.Generator().manual_seed(11)
    y_junk = torch.rand(y0.shape, generator=g)
    y_unsafe_changed = torch.where(s0, y0, y_junk)
    inv = _probe_grads(W6, b6, batch, loss_fn, cfg.pair_w, cfg.anchor, y=y_unsafe_changed)
    same = bool(torch.equal(inv["_gw"], full["_gw"]) and torch.equal(inv["_gb"], full["_gb"]))
    rep["c6_unsafe_label_invariance_bit_identical"] = same
    if not same:
        fails.append("C6 (i) unsafe-label invariance: replacing the EP labels of UNSAFE proposals changed the EP-row gradient (the target is not conditional on NC = DAC = 1)")
    allun = _probe_grads(W6, b6, batch, loss_fn, cfg.pair_w, cfg.anchor, y=y_junk, safe=torch.zeros_like(s0))
    rep["c6_all_unsafe_ep_row_grad"] = {"w": allun["ep_row_w"], "b": allun["ep_row_b"]}
    if allun["ep_row_w"] != 0.0 or allun["ep_row_b"] != 0.0:
        fails.append(f"C6 (ii) all-unsafe batch: EP-row gradient ({allun['ep_row_w']!r}, {allun['ep_row_b']!r}) is not EXACTLY 0.0 (an unsafe proposal still carries a target)")
    y_safe_changed = torch.where(s0, 1.0 - y0, y0)
    chg = _probe_grads(W6, b6, batch, loss_fn, cfg.pair_w, cfg.anchor, y=y_safe_changed)
    rep["c6_safe_label_change_changes_gradient"] = bool(not torch.equal(chg["_gw"], full["_gw"]))
    if not rep["c6_safe_label_change_changes_gradient"]:
        fails.append("C6 (iii) positive control: changing the EP labels of SAFE proposals did NOT change the gradient, so the invariance in (i) would mean nothing")
    # the real fit path moves the row (and only the row): one step with lr = 1e-3 on the same batch
    mini = FitData(log_names=["probe"], log_rank=np.zeros(1, np.int64), log_fold=np.zeros(1, np.int64), log_idx=np.zeros(len(feat0), np.int64), fold=np.zeros(len(feat0), np.int64),
                   feat=feat0, z6=z60, y=y0, safe=s0, pdms=torch.zeros_like(y0), top8=idx0, provenance={})
    w, b = fit_row(mini, np.arange(len(feat0)), dataclasses.replace(cfg, lr=1e-3, epochs=1, batch_sets=len(feat0)), seed=0, W6=W6, b6=b6, loss_fn=loss_fn)
    moved = float(np.abs(w - np.asarray(W6)[EP_COL]).max())
    rep["update_moved_ep_row_max_abs"] = moved
    if not moved > 0.0:
        fails.append("update: one optimiser step at lr = 1e-3 did not move the EP row (the loss is not wired to the optimiser)")
    if fails:
        raise ReachabilityError("C1/C6 RED: " + " | ".join(fails))
    return rep


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# fit / score
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def fit_row(data: FitData, train_idx: np.ndarray, cfg: Cfg, seed: int, W6, b6, loss_fn=None) -> tuple:
    """Fit the EP row on the sets `train_idx`.  Adam (no weight decay, no schedule), minibatches of `cfg.batch_sets` sets, shuffled each epoch by a generator seeded from `seed`.
    -> (row_w [256] float32, row_b [1] float32) as numpy."""
    loss_fn = loss_fn or total_loss
    head = RowHead(W6, b6, probe=False)
    opt = torch.optim.Adam(head.trainable(), lr=cfg.lr, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=0.0)
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(len(train_idx)))
    tr = torch.as_tensor(np.asarray(train_idx), dtype=torch.long)
    n = len(tr)
    for _ in range(cfg.epochs):
        order = tr[torch.randperm(n, generator=g)]
        for s in range(0, n, cfg.batch_sets):
            b = order[s:s + cfg.batch_sets]
            z6 = forward_z6(head, data.feat[b], data.z6[b])
            loss, _p, _a = loss_fn(z6, data.y[b], data.safe[b], data.top8[b], cfg.anchor, cfg.pair_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head.row_w.detach().clone().numpy(), head.row_b.detach().clone().numpy()


def score_sets(data: FitData, idx: np.ndarray, row_w: np.ndarray, row_b: np.ndarray, p2_w: np.ndarray, p2_b: np.ndarray, chunk: int = 2048, y=None) -> dict:
    """Safe-contender concordance on `idx`, per set, the picks under the frozen rule, and the pick-level read.  Concordant: z_i > z_j for y_i > y_j, both safe, both in the frozen top-8; ties count 0.5.
    `y` overrides the labels the concordance is scored against (the C2 stage trains on shuffled labels and scores against the real ones).
    -> {num, den, picks, pick_p2, pd_new, pd_p2, unsafe_new, unsafe_p2, p_safe_sum, p_safe_n, p_unsafe_sum, p_unsafe_n} (per set unless noted)."""
    yy = data.y if y is None else y
    keys = ("num", "den", "pk", "pk0", "pdn", "pdp", "un", "up", "pss", "psn", "pus", "pun")
    acc = {k: [] for k in keys}
    with torch.no_grad():
        for s in range(0, len(idx), chunk):
            ii = torch.as_tensor(np.asarray(idx[s:s + chunk]), dtype=torch.long)
            z6 = apply_row(data.z6[ii], data.feat[ii], row_w, row_b, p2_w, p2_b)
            t8 = data.top8[ii]
            z8 = torch.gather(z6[..., EP_COL], 1, t8)
            y8 = torch.gather(yy[ii], 1, t8)
            s8 = torch.gather(data.safe[ii], 1, t8)
            m = (s8.unsqueeze(2) & s8.unsqueeze(1)) & (y8.unsqueeze(2) > y8.unsqueeze(1))
            d = z8.unsqueeze(2) - z8.unsqueeze(1)
            acc["num"].append((((d > 0) & m).sum((1, 2)).double() + 0.5 * ((d == 0) & m).sum((1, 2)).double()).numpy())
            acc["den"].append(m.sum((1, 2)).double().numpy())
            pk, pk0 = picks_np(z6.numpy()), picks_np(data.z6[ii].numpy())
            acc["pk"].append(pk)
            acc["pk0"].append(pk0)
            rows = np.arange(len(pk))
            pd, sf = data.pdms[ii].numpy(), data.safe[ii].numpy()
            acc["pdn"].append(pd[rows, pk].astype(np.float64))
            acc["pdp"].append(pd[rows, pk0].astype(np.float64))
            acc["un"].append((~sf[rows, pk]).astype(np.float64))
            acc["up"].append((~sf[rows, pk0]).astype(np.float64))
            p = torch.sigmoid(z6[..., EP_COL]).double().numpy()
            acc["pss"].append((p * sf).sum(1))
            acc["psn"].append(sf.sum(1).astype(np.float64))
            acc["pus"].append((p * ~sf).sum(1))
            acc["pun"].append((~sf).sum(1).astype(np.float64))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)               # noqa: E731
    return {"num": cat(acc["num"]), "den": cat(acc["den"]), "picks": cat(acc["pk"]), "pick_p2": cat(acc["pk0"]), "pd_new": cat(acc["pdn"]), "pd_p2": cat(acc["pdp"]),
            "unsafe_new": cat(acc["un"]), "unsafe_p2": cat(acc["up"]), "p_safe_sum": cat(acc["pss"]), "p_safe_n": cat(acc["psn"]), "p_unsafe_sum": cat(acc["pus"]), "p_unsafe_n": cat(acc["pun"])}


def _bin(data: FitData, idx: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.bincount(data.log_idx[idx], weights=v, minlength=data.n_logs)


def _pick_level(sets: float, pdn: float, pdp: float, un: float, up: float) -> dict:
    return {"new_pdms_x100": 100.0 * pdn / sets, "p2_pdms_x100": 100.0 * pdp / sets, "delta_pdms_x100": 100.0 * (pdn - pdp) / sets, "unsafe_rate_new": un / sets, "unsafe_rate_p2": up / sets}


def oof_run(data: FitData, cfg: Cfg, seed: int, W6, b6, train_pct: int = 100, fit_y=None) -> dict:
    """5-fold grouped-by-log CV at one (config, seed).  train_pct < 100 keeps only the first ceil(train_pct % x n) TRAINING logs of each fold in key order (learning curve).
    `fit_y` replaces the labels the row is TRAINED on (C2: shuffled among the safe proposals); scoring stays on the real labels.
    -> pooled and per-fold OOF concordance, per-log numerators / denominators (for the bootstrap), the pick-level read (per-log sums), the violator-leak reads (C7)."""
    p2_w, p2_b = np.asarray(W6, np.float32)[EP_COL], np.asarray(b6, np.float32)[EP_COL:EP_COL + 1]
    train_data = data if fit_y is None else dataclasses.replace(data, y=fit_y)
    z = lambda: np.zeros(data.n_logs)                                          # noqa: E731
    num_log, den_log, n_log, pdn_log, pdp_log, un_log, up_log = z(), z(), z(), z(), z(), z(), z()
    per_fold, changed, n_oof = [], 0, 0
    pss = psn = pus = pun = 0.0
    for k in range(N_FOLDS):
        oof = np.nonzero(data.fold == k)[0]
        tr_logs = [i for i in np.argsort(data.log_rank, kind="stable") if data.log_fold[i] != k]
        if train_pct < 100:
            tr_logs = tr_logs[:max(1, math.ceil(train_pct / 100.0 * len(tr_logs)))]
        tr = np.nonzero(np.isin(data.log_idx, np.array(tr_logs, np.int64)))[0]
        w, b = fit_row(train_data, tr, cfg, seed, W6, b6)
        st = score_sets(data, oof, w, b, p2_w, p2_b)
        num_log += _bin(data, oof, st["num"])
        den_log += _bin(data, oof, st["den"])
        n_log += _bin(data, oof, np.ones(len(oof)))
        pdn_log += _bin(data, oof, st["pd_new"])
        pdp_log += _bin(data, oof, st["pd_p2"])
        un_log += _bin(data, oof, st["unsafe_new"])
        up_log += _bin(data, oof, st["unsafe_p2"])
        pss += float(st["p_safe_sum"].sum())
        psn += float(st["p_safe_n"].sum())
        pus += float(st["p_unsafe_sum"].sum())
        pun += float(st["p_unsafe_n"].sum())
        per_fold.append(float(st["num"].sum() / st["den"].sum()) if st["den"].sum() > 0 else float("nan"))
        changed += int((st["picks"] != st["pick_p2"]).sum())
        n_oof += len(oof)
    pick = _pick_level(float(n_log.sum()), float(pdn_log.sum()), float(pdp_log.sum()), float(un_log.sum()), float(up_log.sum()))
    return {"pooled": float(num_log.sum() / den_log.sum()), "fold_mean": float(np.mean(per_fold)), "per_fold": per_fold, "num_log": num_log, "den_log": den_log,
            "n_pairs": float(den_log.sum()), "oof_picks_changed": changed, "n_oof_sets": n_oof, "n_log": n_log, "pdn_log": pdn_log, "pdp_log": pdp_log, "un_log": un_log, "up_log": up_log,
            "pick_level": pick, "leak_c7": {"mean_sigmoid_ep_on_safe": pss / max(psn, 1.0), "mean_sigmoid_ep_on_unsafe": pus / max(pun, 1.0)}}


def p2_reference(data: FitData, W6, b6) -> dict:
    """P2's own EP row on the same OOF pairs.  IN-SAMPLE for P2 (it trained on these sets): biased upward and in P2's favour at the pick (PREREG_EPR1 s5 row 10)."""
    p2_w, p2_b = np.asarray(W6, np.float32)[EP_COL], np.asarray(b6, np.float32)[EP_COL:EP_COL + 1]
    st = score_sets(data, np.arange(data.n_sets), p2_w, p2_b, p2_w, p2_b)
    per_fold = []
    for k in range(N_FOLDS):
        m = data.fold == k
        per_fold.append(float(st["num"][m].sum() / st["den"][m].sum()))
    n = float(data.n_sets)
    return {"pooled": float(st["num"].sum() / st["den"].sum()), "fold_mean": float(np.mean(per_fold)), "per_fold": per_fold, "n_pairs": float(st["den"].sum()),
            "pick_level": _pick_level(n, float(st["pd_new"].sum()), float(st["pd_p2"].sum()), float(st["unsafe_new"].sum()), float(st["unsafe_p2"].sum())),
            "leak_c7": {"mean_sigmoid_ep_on_safe": float(st["p_safe_sum"].sum() / max(st["p_safe_n"].sum(), 1.0)), "mean_sigmoid_ep_on_unsafe": float(st["p_unsafe_sum"].sum() / max(st["p_unsafe_n"].sum(), 1.0))},
            "note": "IN-SAMPLE for P2 (P2 trained on these sets): biased upward; NOT a reference for any bar"}


def shuffle_safe_labels(y: torch.Tensor, safe: torch.Tensor, seed: int) -> torch.Tensor:
    """C2: permute the EP labels among the SAFE proposals of each set (the safe mask and each set's label multiset are unchanged).  Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    yn, sn = y.numpy().copy(), safe.numpy()
    for i in range(len(yn)):
        idx = np.nonzero(sn[i])[0]
        if len(idx) > 1:
            yn[i, idx] = yn[i, idx[rng.permutation(len(idx))]]
    return torch.from_numpy(yn)


def row_bytes(w: np.ndarray, b: np.ndarray) -> bytes:
    """float32 [257] = weight[2,:] then bias[2], as a .npy (header + data: deterministic, so the sha256 is reproducible)."""
    bio = io.BytesIO()
    np.save(bio, np.concatenate([np.asarray(w, np.float32), np.asarray(b, np.float32).reshape(1)]))
    return bio.getvalue()


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# bootstrap (A2.4): mirrors zero_tail.Boot (draws = default_rng(seed).integers(0, n_logs, (B, n_logs)), percentile 2.5 / 97.5)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def boot_draws(n_logs: int, B: int = BOOT_B, seed: int = BOOT_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_logs, size=(B, n_logs))


def paired_delta(num_a: np.ndarray, num_b: np.ndarray, den: np.ndarray, draws: np.ndarray) -> dict:
    """Delta = c_a - c_b with c = sum(num)/sum(den) over the same OOF pairs; paired log-cluster bootstrap on the SAME draws."""
    d_all = den.sum()
    point = float(num_a.sum() / d_all - num_b.sum() / d_all)
    bd = den[draws].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(bd > 0, (num_a[draws].sum(axis=1) - num_b[draws].sum(axis=1)) / bd, np.nan)
    ok = ~np.isnan(v)
    return {"point": point, "lo": float(np.percentile(v[ok], 2.5)), "hi": float(np.percentile(v[ok], 97.5)), "B": int(len(draws)), "n_nan": int((~ok).sum())}


def lc_call(per_seed: dict, both_rule: str) -> dict:
    """A2.4 item 3.  RISING: lower bound of delta > 0 on both seeds.  FLAT: upper bound < 0.010 on both seeds.  INCONCLUSIVE: anything else.
    An interval wholly inside (0, 0.010) satisfies BOTH definitions: `both_rule` ('flat' | 'rising') decides, and the flags are always reported."""
    rising = all(v["lo"] > 0.0 for v in per_seed.values())
    flat = all(v["hi"] < LC_DELTA_FLAT for v in per_seed.values())
    if rising and flat:
        call = "FLAT" if both_rule == "flat" else "RISING"
    elif rising:
        call = "RISING"
    elif flat:
        call = "FLAT"
    else:
        call = "INCONCLUSIVE"
    return {"call": call, "rising": rising, "flat": flat, "both_true": bool(rising and flat), "both_rule": both_rule, "delta_flat_threshold": LC_DELTA_FLAT}


def boot_pick_level(r: dict, draws: np.ndarray) -> dict:
    """The pick-level read at c* with the paired log-cluster bootstrap: Delta mean PDMS x100 (new - P2's pick) and Delta unsafe-pick rate (new - P2's)."""
    d = paired_delta(r["pdn_log"], r["pdp_log"], r["n_log"], draws)
    u = paired_delta(r["un_log"], r["up_log"], r["n_log"], draws)
    return {"delta_pdms_x100": {k: (100.0 * v if k in ("point", "lo", "hi") else v) for k, v in d.items()}, "delta_unsafe_rate": u}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _setup(opts: Options) -> tuple:
    torch.set_num_threads(max(1, int(opts.threads)))
    torch.use_deterministic_algorithms(True)
    guard = opts.guard()
    out = opts.resolved_out_dir()
    if opts.fraction not in FRACTIONS:
        raise BankError(f"fraction {opts.fraction} not in the registered nested sizes {FRACTIONS}")
    if opts.s0_marker is not None and not _io_exists(guard, opts.s0_marker):
        raise BankError(f"{opts.s0_marker} does not exist: S0 is not done, EP-R1 does not start")
    manifest, man_sha = load_manifest(guard, opts.manifest)
    rows, ranked_sha = load_ranked_logs(guard, opts.ranked_logs, opts.ranked_sha256, opts.expected_n_logs)
    repl = None
    if opts.replacements_file:
        repl = json.loads(_io_read_bytes(guard, opts.replacements_file).decode("utf-8"))
    selected, repl_rep = select_logs(rows, opts.fraction, repl)
    per_fold_logs = [sum(1 for r in selected if fold_of_rank(r["rank"]) == k) for k in range(N_FOLDS)]
    if min(per_fold_logs) < opts.min_logs_per_fold:
        raise BankError(f"A2.2: a fold has {min(per_fold_logs)} logs < the minimum {opts.min_logs_per_fold} (logs per fold {per_fold_logs})")
    W6, b6 = load_p2_row(guard, opts.p2_ckpt)
    data = load_fit_data(guard, opts, selected, manifest, W6, b6)
    return guard, out, manifest, man_sha, ranked_sha, selected, repl_rep, per_fold_logs, W6, b6, data


def _probe_batch(data: FitData, n: int) -> dict:
    y8, s8 = torch.gather(data.y, 1, data.top8), torch.gather(data.safe, 1, data.top8)
    has = (((s8.unsqueeze(2) & s8.unsqueeze(1)) & (y8.unsqueeze(2) > y8.unsqueeze(1))).sum((1, 2)) > 0).nonzero().flatten()
    if len(has) == 0:
        raise ReachabilityError("no set of the fit data has a safe contender pair with differing EP")
    ii = has[:n]
    return {"feat": data.feat[ii], "z6": data.z6[ii], "y": data.y[ii], "safe": data.safe[ii], "top8": data.top8[ii]}


def _x100(d: dict) -> dict:
    return {k: (100.0 * v if k in ("point", "lo", "hi") else v) for k, v in d.items()}


def run_r1(opts: Options) -> dict:
    t0 = time.time()
    guard, out, manifest, man_sha, ranked_sha, selected, repl_rep, per_fold_logs, W6, b6, data = _setup(opts)
    _io_makedirs(guard, out)
    result_path = os.path.join(out, "epr1_result.json")
    stale = [result_path] + [os.path.join(out, f"ep_row_seed{sd}.npy") for sd in opts.seeds]
    for p in stale:
        if _io_exists(guard, p):
            raise BankError(f"{p} exists: c* is chosen ONCE and never re-drawn (move the directory away to rerun)")
    base = Cfg(lr=GRID_LR[-1], anchor=GRID_ANCHOR[1], epochs=opts.epochs, batch_sets=opts.batch_sets)
    reach = reachability_check(W6, b6, _probe_batch(data, opts.reach_n_sets), base)
    _log(f"C1/C6 GREEN: {json.dumps({k: reach[k] for k in ('n_pairs_in_probe_batch', 'n_unsafe_in_probe_batch', 'update_moved_ep_row_max_abs')})}")
    y8, s8 = np.take_along_axis(data.y.numpy(), data.top8.numpy(), axis=1), np.take_along_axis(data.safe.numpy(), data.top8.numpy(), axis=1)
    pc = pair_counts_np(y8, s8)
    fold_table = [{"fold": k, "logs": per_fold_logs[k], "sets": int((data.fold == k).sum()), "contrast_sets": int(((data.fold == k) & (pc > 0)).sum()),
                   "pairs": int(pc[data.fold == k].sum())} for k in range(N_FOLDS)]
    ref = p2_reference(data, W6, b6)
    cfgs = grid(opts.epochs, opts.batch_sets)
    rows, rr = [], {}
    for ci, cfg in enumerate(cfgs):
        per_seed = {}
        for sd in opts.seeds:
            r = oof_run(data, cfg, sd, W6, b6)
            rr[(ci, sd)] = r
            per_seed[str(sd)] = {k: r[k] for k in ("pooled", "fold_mean", "per_fold", "n_pairs", "oof_picks_changed", "n_oof_sets", "pick_level", "leak_c7")}
        score = {c: float(np.mean([per_seed[str(s)][c] for s in opts.seeds])) for c in ("pooled", "fold_mean")}
        rows.append({"id": cfg.id(), "lr": cfg.lr, "anchor": cfg.anchor, "config_sha256": cfg.sha256(), "per_seed": per_seed, "mean_over_seeds": score,
                     "score": score["pooled" if opts.criterion == "pooled_oof" else "fold_mean"]})
        _log(f"config {ci + 1}/{len(cfgs)} {cfg.id()} pooled {score['pooled']:.5f} fold_mean {score['fold_mean']:.5f}")

    def best(key: str) -> int:
        vals = [r["mean_over_seeds"][key] for r in rows]
        return int(np.argmax(vals))           # first maximum = grid order (lr ascending, anchor ascending)
    i_star = best("pooled" if opts.criterion == "pooled_oof" else "fold_mean")
    ties = [r["id"] for r in rows if r["score"] == rows[i_star]["score"]]
    star = cfgs[i_star]
    draws = boot_draws(data.n_logs)
    pick_boot = {str(sd): boot_pick_level(rr[(i_star, sd)], draws) for sd in opts.seeds}
    final = {}
    for sd in opts.seeds:
        w, b = fit_row(data, np.arange(data.n_sets), star, sd, W6, b6)
        blob = row_bytes(w, b)
        name = f"ep_row_seed{sd}.npy"
        sha = _io_write_bytes(guard, os.path.join(out, name), blob, exclusive=True)
        final[str(sd)] = {"file": name, "sha256": sha, "bytes": len(blob), "layout": "float32[257] = weight[2,:256] then bias[2], .npy",
                          "max_abs_change_vs_p2": float(max(np.abs(w - W6[EP_COL]).max(), np.abs(b - b6[EP_COL]).max())), "fitted_on": "ALL fit-set sets of S_f"}
    code_sha = sha256_bytes(_io_read_bytes(guard, os.path.abspath(__file__)))     # the code sha256 (own source; a guard-listed file)
    result = {
        "schema": SCHEMA, "stage": "R1", "fraction_pct": opts.fraction,
        "inputs": {"bank_dir": opts.bank_dir, "bank_manifest": opts.manifest, "bank_manifest_sha256": man_sha, "ranked_fit_logs_sha256": ranked_sha, "p2_ckpt": opts.p2_ckpt,
                   "p2_row_sha256": sha256_bytes(row_bytes(W6[EP_COL], b6[EP_COL:EP_COL + 1])), "bank_provenance": data.provenance, "replacements": repl_rep,
                   "n_logs": data.n_logs, "n_sets": data.n_sets, "fold_table": fold_table, "pins_overridden": opts.pins_overridden()},
        "params": {"grid_lr": list(GRID_LR), "grid_anchor": list(GRID_ANCHOR), "k": K, "seeds": list(opts.seeds), "epochs": opts.epochs, "batch_sets": opts.batch_sets,
                   "optimizer": "adam", "betas": list(ADAM_BETAS), "eps": ADAM_EPS, "criterion": opts.criterion, "folds": N_FOLDS, "fold_rule": "rank_in_key_order mod 5",
                   "target": TARGET_ID, "safe_min": SAFE_MIN, "threads": opts.threads, "torch": torch.__version__, "numpy": np.__version__},
        "c1_c6_reachability": reach, "p2_row_reference_oof": ref, "grid": rows,
        "c_star": {"id": star.id(), "lr": star.lr, "anchor": star.anchor, "config_sha256": star.sha256(), "score": rows[i_star]["score"], "criterion": opts.criterion,
                   "tied_at_max": ties, "tie_break": "first in grid order (lr ascending, anchor ascending)",
                   "by_pooled_oof": cfgs[best("pooled")].id(), "by_fold_mean": cfgs[best("fold_mean")].id(), "criteria_agree": best("pooled") == best("fold_mean"),
                   "c_star_at_grid_edge": bool(star.lr == GRID_LR[-1])},
        "pick_level_at_c_star": pick_boot, "final_rows": final, "code_sha256": code_sha,
        "run_meta": {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(time.time() - t0, 1)},
    }
    _io_write_bytes(guard, result_path, canon_json(result), exclusive=True)
    _log(f"c* = {star.id()} score {rows[i_star]['score']:.5f}; wrote {result_path}")
    return result


def _load_r1(guard, out: str, man_sha: str, opts: Options) -> dict:
    r1 = json.loads(_io_read_bytes(guard, os.path.join(out, "epr1_result.json")).decode("utf-8"))
    if r1["inputs"]["bank_manifest_sha256"] != man_sha:
        raise BankError("the bank manifest changed since epr1_result.json was written: this read must use the SAME bank (rerun R1 for the new bank)")
    if r1["fraction_pct"] != opts.fraction:
        raise BankError(f"epr1_result.json is for f = {r1['fraction_pct']} %, not {opts.fraction} %")
    cs = r1["c_star"]
    star = Cfg(lr=cs["lr"], anchor=cs["anchor"], epochs=r1["params"]["epochs"], batch_sets=r1["params"]["batch_sets"])
    if star.sha256() != cs["config_sha256"]:
        raise BankError("c* from epr1_result.json does not reproduce its recorded config sha256")
    return r1, star


def run_lc(opts: Options) -> dict:
    """The learning-curve read at c*, on the FIT set only: nested prefixes p of each fold's training logs (key order), both seeds, pooled OOF concordance on the SAME OOF pairs,
    Delta = c(100 %) - c(50 %), paired log-cluster bootstrap.  No exponent is fitted, nothing is extrapolated.  It has NO authority over S0 scale-up (PREREG_EPR1 s5 row 9)."""
    t0 = time.time()
    guard, out, manifest, man_sha, ranked_sha, selected, repl_rep, per_fold_logs, W6, b6, data = _setup(opts)
    r1, star = _load_r1(guard, out, man_sha, opts)
    cs = r1["c_star"]
    draws = boot_draws(data.n_logs)
    pts: dict = {}
    for sd in opts.seeds:
        for p in LC_PREFIXES:
            r = oof_run(data, star, sd, W6, b6, train_pct=p)
            pts[(sd, p)] = r
            _log(f"LC seed {sd} p {p} % pooled {r['pooled']:.5f}")
    per_seed, table = {}, {}
    for sd in opts.seeds:
        a, b = pts[(sd, 100)], pts[(sd, 50)]
        per_seed[str(sd)] = paired_delta(a["num_log"], b["num_log"], a["den_log"], draws)
        table[str(sd)] = {str(p): {"pooled": pts[(sd, p)]["pooled"], "fold_mean": pts[(sd, p)]["fold_mean"], "n_pairs": pts[(sd, p)]["n_pairs"]} for p in LC_PREFIXES}
    p100_ok = {str(sd): bool(abs(pts[(sd, 100)]["pooled"] - r1["grid"][[g["id"] for g in r1["grid"]].index(cs["id"])]["per_seed"][str(sd)]["pooled"]) <= 1e-9) for sd in opts.seeds}
    if not all(p100_ok.values()):
        _log("WARNING: LC p = 100 % does not reproduce R1's score at c* (different thread count / machine?)")
    result = {"schema": SCHEMA, "stage": "LC", "fraction_pct": opts.fraction, "c_star": cs, "bank_manifest_sha256": man_sha,
              "prefixes_pct": list(LC_PREFIXES), "table": table, "delta_100_minus_50": per_seed, "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "estimator": "paired log-cluster bootstrap"},
              "call": lc_call(per_seed, opts.both_rule), "p100_reproduces_r1": p100_ok, "n_logs": data.n_logs,
              "note": "p = 25 % is descriptive. No learning-curve exponent is fitted or quoted, nothing is extrapolated. No authority over S0 scale-up.",
              "run_meta": {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(time.time() - t0, 1)}}
    _io_write_bytes(guard, os.path.join(out, "epr1_lc_result.json"), canon_json(result), exclusive=True)
    _log(f"LC call {result['call']['call']}")
    return result


def run_c2(opts: Options) -> dict:
    """C2 negative control at c*: the SAME fit with the EP labels shuffled among the SAFE proposals of each set (seeded), scored against the REAL labels.  Writes the shuffled rows the dev
    applier needs (their dev gain must be <= 0) and the OOF read."""
    t0 = time.time()
    guard, out, manifest, man_sha, ranked_sha, selected, repl_rep, per_fold_logs, W6, b6, data = _setup(opts)
    r1, star = _load_r1(guard, out, man_sha, opts)
    y_shuf = shuffle_safe_labels(data.y, data.safe, C2_SHUFFLE_SEED)
    per_seed, final = {}, {}
    for sd in opts.seeds:
        r = oof_run(data, star, sd, W6, b6, fit_y=y_shuf)
        per_seed[str(sd)] = {k: r[k] for k in ("pooled", "fold_mean", "per_fold", "n_pairs", "oof_picks_changed", "n_oof_sets", "pick_level", "leak_c7")}
        _log(f"C2 seed {sd} pooled OOF concordance (real labels) {r['pooled']:.5f}")
        tr_data = dataclasses.replace(data, y=y_shuf)
        w, b = fit_row(tr_data, np.arange(data.n_sets), star, sd, W6, b6)
        blob = row_bytes(w, b)
        name = f"ep_row_c2shuffle_seed{sd}.npy"
        final[str(sd)] = {"file": name, "sha256": _io_write_bytes(guard, os.path.join(out, name), blob, exclusive=True), "bytes": len(blob)}
    result = {"schema": SCHEMA, "stage": "C2", "fraction_pct": opts.fraction, "c_star": r1["c_star"], "bank_manifest_sha256": man_sha, "shuffle_seed": C2_SHUFFLE_SEED,
              "shuffle": "EP labels permuted among the SAFE proposals of each set; trained on shuffled, scored on real", "per_seed": per_seed, "final_rows": final,
              "p2_row_reference_oof": r1["p2_row_reference_oof"], "run_meta": {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(time.time() - t0, 1)}}
    _io_write_bytes(guard, os.path.join(out, "epr1_c2_result.json"), canon_json(result), exclusive=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EP-R1 (conditional EP row fit + 5-fold grouped-by-log CV), the learning-curve read and the C2 control. Never reads the dev-gate or navtest tables.")
    ap.add_argument("cmd", choices=("run", "lc", "c2"))
    ap.add_argument("--fraction", type=int, required=True, choices=FRACTIONS, help="nested fit-set size in percent (A2.1); MUST equal the fraction SR-TTC R1 wrote its c* at")
    ap.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    ap.add_argument("--manifest", default=None, help="default: <bank-dir>/MANIFEST.sha256.json")
    ap.add_argument("--ranked-logs", default=DEFAULT_RANKED)
    ap.add_argument("--p2-ckpt", default=DEFAULT_P2_CKPT)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--s0-marker", default=DEFAULT_S0_MARKER)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="UNREGISTERED in SR-TTC; frozen by PREREG_EPR1 s5 row 2")
    ap.add_argument("--batch-sets", type=int, default=DEFAULT_BATCH_SETS, help="frozen by PREREG_EPR1 s5 row 2")
    ap.add_argument("--criterion", choices=("pooled_oof", "fold_mean"), default="pooled_oof")
    ap.add_argument("--both-rule", choices=("flat", "rising"), default="flat")
    ap.add_argument("--replacements-file", default=None)
    ap.add_argument("--threads", type=int, default=8)
    return ap


def options_from_args(a: argparse.Namespace) -> Options:
    return Options(fraction=a.fraction, bank_dir=a.bank_dir, manifest=a.manifest or (a.bank_dir.rstrip("/\\") + "/MANIFEST.sha256.json"), ranked_logs=a.ranked_logs,
                   p2_ckpt=a.p2_ckpt, out_dir=a.out_dir, s0_marker=a.s0_marker, seeds=tuple(int(s) for s in a.seeds.split(",")), epochs=a.epochs, batch_sets=a.batch_sets,
                   criterion=a.criterion, both_rule=a.both_rule, replacements_file=a.replacements_file, threads=a.threads)


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    opts = options_from_args(a)
    {"run": run_r1, "lc": run_lc, "c2": run_c2}[a.cmd](opts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
