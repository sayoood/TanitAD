#!/usr/bin/env python3
"""EP-R1 DEV GATE applier (PREREG_EPR1 s7, s8; MM_DECISIONS_EPR1 D1-D3).  CPU only.  Mirrors `selrank_dev.py` (SR-TTC): the SAME held-out table (P2's own navtrain HELD-OUT
64-proposal table, 1,705 tokens / 24 logs), the SAME estimator (zero_tail.Boot draws: 10,000 log-cluster resamples, seed 20260927, paired), the SAME C-checks.  Nothing is fitted here.

WHAT IT DOES (one read, ever)
  Applies the two fitted EP rows of c* (`ep_row_seed0.npy`, `ep_row_seed1.npy`) to P2's held-out PRE-HEAD features, replacing ONLY column 2 of the six logits
  (`epr1.apply_row`: stored + feat . (row - P2's row)), picks by the planner's own rule NC*DAC*(5EP+5TTC+2C)/12 (`planner.py:558-561`; `epr1.picks_np`: float64, stable, lowest index on ties;
  this reproduces the planner's float32 pick on 12,146 / 12,146 navtest tokens, `same_pick_proof.json`) and reads PDMS BY LOOKUP `pdms[token, new_pick]` -- the fan is frozen (C4).
  BAR (frozen): paired out-of-sample total gain >= +0.30 PDMS x100 on BOTH seeds, AND C0, C1, C2, C4 pass.  24 logs support no lower-bound rule: the interval is REPORTED, never a bar.
  Reported alongside, not bars: NC, DAC, EP, TTC, C, DDC components per seed, changed picks, the unsafe-pick rate, the JOINT number (EP-R1 row + SR-TTC's c* TTC row of the same seed;
  SECONDARY, labelled so, never decides anything) and the violator-leak read (C7).

THE ORDER IS THE PROTOCOL (each refusal below fires BEFORE any dev OUTCOME is opened; the outcome arrays `pdms`, `sub`, `valid` are touched only after READ_STARTED is written)
  1. the c* result file's sha256 must equal the sha256 REGISTERED beforehand (`--registered-result-sha256`): "c* and its sha256 are written before the dev gate is read" (PREREG s7).
  2. the result must be the registered object: schema/stage, seeds {0,1}, no overridden pins, c*'s config sha256 reproduces, the fit's code sha256 == the landed epr1.py pin, both row files hash to
     the recorded sha256, and the recorded P2-row sha256 equals the EP row of THIS checkpoint (the rows were fitted from this P2).
  3. E1: c* at the grid edge (lr = 1e-3) is REFUSED unless a Master Mind waiver naming that c* exists ("the dev gate is NOT read on the edge c* before EP-R1b has run, unless the MM records why").
  4. the C2 shuffled-label rows and result must exist (C2 is a bar condition: no C2, no read).
  5. every dev input file is hashed against its landed sha256 BEFORE np.load; token/log order is identical across the three files.
  6. the controls that need no outcome run first and GATE the read: integrity (P2's head reproduces the stored logits from the features), C0 (identity), C1 (reach), C4 (fan frozen).
     A RED control stops the process BEFORE the dev table's outcomes are opened -- a broken instrument does not spend the read.
  7. READ_STARTED is written EXCLUSIVELY; a second attempt is refused whatever happened to the first (the read is consumed by the attempt, not by its success).
  8. only then: pdms / sub / valid, the formula check, C2 (needs outcomes), the primary, the components, the JOINT secondary, the outcome rows of PREREG s8.

EVIDENCE CLASS of anything this prints: MEASURED on the dev table (SELECTION tier, a GATE and not a claim: the dev set is not virgin for EP-R1, MM_DECISIONS_EPR1 D3).  Only the navtest primary is quotable.

CLI
    epr1_dev.py run --run-dir C:/Users/Admin/epr1/run/f10 --registered-result-sha256 <hex> [--ttc-run-dir C:/Users/Admin/srttc_r1/run/f10 --ttc-registered-result-sha256 <hex>]
                    [--mm-edge-waiver waiver.json] [--out-dir D] [--dev-dir D] [--p2-ckpt P]
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

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import epr1 as R   # noqa: E402  (the LANDED fit code; frozen: apply_row, picks_np, boot_draws, paired_delta, safe_np, Cfg, row_bytes)

SCHEMA = "epr1-dev-v1"
STAGE = "DEV"
DEV_BAR_X100 = 0.30                      # PREREG_EPR1 s7: paired total gain >= +0.30 PDMS x100 on BOTH seeds (inherited from SR-TTC's dev floor, the 0.33-0.44 training-seed range)
SEED_FLOOR_RANGE = (0.33, 0.44)          # PREREG_SELRANK_ARM s6/s7 (INHERITED): reported against |gain_s0 - gain_s1|, never a bar
EP_COL, TTC_COL = R.EP_COL, R.TTC_COL
COMPS = ("nc", "dac", "ep", "ttc", "c", "ddc")
SUB_NAMES = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
SEED_KEYS = ("0", "1")
FEATCHECK_TOL = 2e-3                     # featcheck.json MEASURED 1.45e-3 (fp16 input); a shuffled-row control reads 15.2
FORMULA_TOL = 1e-6                       # G4b: pdms == NC*DAC*(5EP+5TTC+2C)/12, MEASURED max abs diff 0.0
EPR1_SHA256 = "a751c63eb8e38343a7ec1286114a158046ce4ec13390b2eea0a4b1d2237c38d8"   # the LANDED epr1.py (CODE_SHA256.json at 54f0258); the fit and this applier must be the same code

DEV_DIR = "D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18"
DEV_FILES = ("table_p2_heldout.npz", "prehead_features_p2_heldout.npz", "proposals.npz")
DEV_PINS = {   # MANIFEST.sha256.json of raw/2026-10-18-p2-heldout-table (landed, A2.10 condition 4)
    "table_p2_heldout.npz": "8eacc19b35b12aee9b0ab047716c901399a711d22ddd3c25cd890ecf2f02004b",
    "prehead_features_p2_heldout.npz": "e37b9c554bf2c2546e958ff763dca23b71e0b77036a271d6a003c8ba2672f522",
    "proposals.npz": "a37422102add66ca386fdadfafa670b121cca1c67ed9dde997232982bb36de4b",
}
DEV_N_TOKENS, DEV_N_LOGS = 1705, 24      # A2.10: the full gate set (3,137 sets / 1,705 tokens / 24 logs; the table has one row per token)

READ_STARTED = "epr1_dev_READ_STARTED.json"
RESULT_NAME = "epr1_dev_result.json"
SIDECAR_NAME = "epr1_dev_result.sha256"


class DevRefusal(RuntimeError):
    """A registered precondition is not met.  Raised BEFORE any dev outcome is opened (unless the message says otherwise); nothing was read."""


class DevGuardError(RuntimeError):
    """A file access that would touch navtest / the S0 bank, or anything outside the allow-list."""


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# PathGuard + the ONLY functions that touch the file system (`_io_*`; a test enforces that by AST).  The dev applier never reads navtest or the fit bank.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
FORBIDDEN_TOKENS = ("navtest", "table_p2_full", "gate_table", "srttc_s0", ".sets.npz", "s1_read_cache", "ranked_fit_logs")


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
            raise DevGuardError(f"REFUSED read of {path!r}: outside the allow-list (dev dir, the run dirs, the named inputs, the output dir)")
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
    run_dir: str
    registered_result_sha256: str
    dev_dir: str = DEV_DIR
    p2_ckpt: str = R.DEFAULT_P2_CKPT
    out_dir: str = ""                                   # default: run_dir (the confirm applier looks for the dev result next to the rows)
    ttc_run_dir: str | None = None                      # SR-TTC R1's run dir (r1_result.json + ttc_row_seed{0,1}.npy); None -> the JOINT is not computed
    ttc_registered_result_sha256: str | None = None
    mm_edge_waiver: str | None = None
    dev_pins: dict = dataclasses.field(default_factory=lambda: dict(DEV_PINS))
    expect_tokens: int | None = DEV_N_TOKENS
    expect_logs: int | None = DEV_N_LOGS
    epr1_sha256: str | None = EPR1_SHA256
    featcheck_tol: float = FEATCHECK_TOL
    chunk: int = 512

    def resolved_out_dir(self) -> str:
        return self.out_dir or self.run_dir

    def guard(self) -> DevGuard:
        files = [self.p2_ckpt, os.path.abspath(R.__file__)]
        if self.mm_edge_waiver:
            files.append(self.mm_edge_waiver)
        dirs = [self.dev_dir, self.run_dir, self.resolved_out_dir()]
        if self.ttc_run_dir:
            dirs.append(self.ttc_run_dir)
        return DevGuard(read_dirs=dirs, read_files=files, write_dirs=[self.resolved_out_dir()])

    def pins_overridden(self) -> list:
        out = []
        if self.dev_pins != DEV_PINS:
            out.append("dev_pins")
        if self.expect_tokens != DEV_N_TOKENS:
            out.append("expect_tokens")
        if self.expect_logs != DEV_N_LOGS:
            out.append("expect_logs")
        if self.epr1_sha256 != EPR1_SHA256:
            out.append("epr1_sha256")
        if self.featcheck_tol != FEATCHECK_TOL:
            out.append("featcheck_tol")
        return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# small loaders
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_p2_head(guard: DevGuard, ckpt: str) -> tuple:
    """-> (W6 [6,256] float32, b6 [6] float32): ALL six rows of P2's score_head (the EP row is the one replaced; the others are needed for the integrity check and the JOINT's TTC row)."""
    sd = _io_torch_load(guard, ckpt)["model"]
    W, b = sd["score_head.weight"].detach().float().clone().numpy(), sd["score_head.bias"].detach().float().clone().numpy()
    if W.shape != (R.N_COMP, R.FEAT_DIM) or b.shape != (R.N_COMP,):
        raise DevRefusal(f"P2 score_head is {W.shape}/{b.shape}, expected ({R.N_COMP}, {R.FEAT_DIM}) (a 7-output SFT-4 head is not P2)")
    return W, b


def load_row(guard: DevGuard, path: str, expect_sha256: str | None = None) -> tuple:
    """-> (w [256] float32, b [1] float32, sha256).  Layout `epr1.row_bytes`: float32[257] = weight[col,:256] then bias[col], .npy.  The hash is taken over the exact bytes that are parsed."""
    raw = _io_read_bytes(guard, path)
    sha = sha256_bytes(raw)
    if expect_sha256 is not None and sha != expect_sha256:
        raise DevRefusal(f"{path}: sha256 {sha} != the recorded {expect_sha256} (the row is not the one the fit wrote)")
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    if arr.dtype != np.float32 or arr.shape != (R.FEAT_DIM + 1,) or not np.isfinite(arr).all():
        raise DevRefusal(f"{path}: expected finite float32[{R.FEAT_DIM + 1}], got {arr.dtype}{arr.shape}")
    return arr[:R.FEAT_DIM].copy(), arr[R.FEAT_DIM:].copy(), sha


def load_json(guard: DevGuard, path: str) -> tuple:
    raw = _io_read_bytes(guard, path)
    return json.loads(raw.decode("utf-8")), sha256_bytes(raw)


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
        if pin is not None and sha[n] != pin:                                       # MUTATION-TARGET-DEVSHA
            raise DevRefusal(f"dev input {n}: sha256 {sha[n]} != the landed {pin}")
        stats[n] = _io_stat(guard, p)
    T, F, P = (_io_npz(guard, paths[n]) for n in DEV_FILES)
    token, log = np.array([str(x) for x in T["token"]]), np.array([str(x) for x in T["log"]])
    N = len(token)
    if opts.expect_tokens is not None and N != opts.expect_tokens:
        raise DevRefusal(f"dev table has {N} tokens, expected {opts.expect_tokens} (A2.10: the full gate set)")
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
    if feat.dtype != np.float16 or feat.shape != (N, R.N_PROPOSALS, R.FEAT_DIM):
        raise DevRefusal(f"pre-head features {feat.dtype}{feat.shape} != float16[{N},{R.N_PROPOSALS},{R.FEAT_DIM}]")
    if logits.dtype != np.float32 or logits.shape != (N, R.N_PROPOSALS, R.N_COMP):
        raise DevRefusal(f"table logits {logits.dtype}{logits.shape} != float32[{N},{R.N_PROPOSALS},{R.N_COMP}]")
    if props_t.shape != (N, R.N_PROPOSALS, 8, 3) or props_d.shape != (N, R.N_PROPOSALS, 8, 3):
        raise DevRefusal(f"proposals shapes {props_t.shape} / {props_d.shape} != ({N},{R.N_PROPOSALS},8,3)")
    if not (np.isfinite(logits).all() and np.isfinite(feat.astype(np.float32)).all()):
        raise DevRefusal("non-finite logit or feature in the dev inputs")
    if pick.shape != (N,) or pick.min() < 0 or pick.max() >= R.N_PROPOSALS:
        raise DevRefusal("table pick out of range")
    for n, p in paths.items():
        if _io_stat(guard, p) != stats[n]:
            raise DevRefusal(f"{n} changed on disk between verification and load")
    prov = {"dev_dir": opts.dev_dir, "sha256": sha, "n_tokens": N, "n_logs": len(log_names), "pins_overridden": opts.pins_overridden()}
    feat, logits, props_t, props_d = np.array(feat), np.array(logits), np.array(props_t), np.array(props_d)      # materialise, then release the archive handles
    for z in (T, F, P):
        z.close()
    return DevInputs(token=token, log=log, log_names=log_names.tolist(), log_idx=np.asarray(log_idx, np.int64), feat=torch.from_numpy(feat), z6=torch.from_numpy(logits),
                     pick_banked=pick, props_table=props_t, props_dump=props_d, pick_dump=pick_d, token_feat=token_f, log_feat=log_f, token_dump=token_d, log_dump=log_d, provenance=prov)


def load_dev_outcomes(guard: DevGuard, opts: DevOptions) -> DevOutcomes:
    """THE DEV READ.  Opens pdms / sub / valid.  Called only after READ_STARTED exists."""
    return read_outcomes(guard, os.path.join(opts.dev_dir, "table_p2_heldout.npz"))


def read_outcomes(guard, table_path: str) -> DevOutcomes:
    """Opens the OUTCOME arrays (pdms, sub, valid, sub_names) of a 64-proposal table and checks the PDMS formula.  Shared with `refe_head_confirm.py` (the navtest table has the same layout)."""
    T = _io_npz(guard, table_path)
    names = tuple(str(x) for x in T["sub_names"])
    if names != SUB_NAMES:
        raise DevRefusal(f"table sub_names {names} != {SUB_NAMES} (the component order is the bank's label order; EP would not be column {EP_COL})")
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
    return DevOutcomes(pdms=pdms, sub=sub, safe=R.safe_np(sub), formula={"max_abs_diff": d, "ddc_in_product_control_max_abs_diff": ctl, "tol": FORMULA_TOL})


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the applier: ONLY the named columns change
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def apply_rows(z6: torch.Tensor, feat: torch.Tensor, rows: dict, W6: np.ndarray, b6: np.ndarray, chunk: int = 512) -> np.ndarray:
    """rows: {col: (w [256], b [1])}.  -> [N,64,6] float32 logits with ONLY the named columns replaced (stored + feat . (row - P2's row) + (b - P2's b), `epr1.apply_row`).
    Identity: a row equal to P2's leaves the column bit-identical (the delta is exactly 0.0)."""
    out = []
    for s in range(0, z6.shape[0], chunk):
        z, f = z6[s:s + chunk], feat[s:s + chunk]
        for col, (w, b) in rows.items():
            z = R.apply_row(z, f, w, b, W6[col], b6[col:col + 1], col=col)       # MUTATION-TARGET-APPLY
        out.append(z.numpy() if isinstance(z, torch.Tensor) else np.asarray(z))
    return np.concatenate(out) if out else np.zeros((0, R.N_PROPOSALS, R.N_COMP), np.float32)


def pick_logits(z6: np.ndarray) -> np.ndarray:
    """The planner's rule on logits: NC*DAC*(5EP+5TTC+2C)/12 over sigmoids, first max (`epr1.picks_np`)."""
    return R.picks_np(z6)


def sigmoid64(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# controls that need NO outcome
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def integrity_featcheck(inp: DevInputs, W6: np.ndarray, b6: np.ndarray, tol: float, chunk: int = 512) -> dict:
    """P2's own head, applied to the cached features, reproduces the stored logits (all six columns) -- the features are the INPUT of THIS head, from the same forward.
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
    return {"max_abs_featcheck": dev, "tol": tol, "negative_control_next_token_max_abs": neg, "negative_control_reads_mismatch": bool(neg > 10 * tol), "pass": bool(dev <= tol and neg > 10 * tol)}


def c0_identity(inp: DevInputs, W6: np.ndarray, b6: np.ndarray, chunk: int = 512, ttc_rows: bool = True) -> dict:
    """C0 (PREREG s6; known-value identity of the brief): with the EP row UNCHANGED (P2's own row) the logits are bit-identical to the stored ones and the dev pick equals P2's pick on every
    token; with P2's EP and TTC rows (the JOINT with nothing replaced) the same.  The gain on identical picks is exactly 0.0 by construction (checked again on the outcomes)."""
    ep = {EP_COL: (W6[EP_COL].copy(), b6[EP_COL:EP_COL + 1].copy())}
    z_ep = apply_rows(inp.z6, inp.feat, ep, W6, b6, chunk)
    stored = inp.z6.numpy()
    rec = {"logits_bit_identical_ep_row_is_p2": bool(np.array_equal(z_ep, stored))}
    pk_stored = pick_logits(stored)
    rec["picks_equal_banked_p2_pick"] = int((pk_stored == inp.pick_banked).sum())
    rec["picks_equal_banked_p2_pick_all"] = bool((pk_stored == inp.pick_banked).all())
    rec["picks_identity_ep_row"] = int((pick_logits(z_ep) == inp.pick_banked).sum())
    ok = rec["logits_bit_identical_ep_row_is_p2"] and rec["picks_equal_banked_p2_pick_all"] and rec["picks_identity_ep_row"] == inp.n
    if ttc_rows:
        both = {EP_COL: ep[EP_COL], TTC_COL: (W6[TTC_COL].copy(), b6[TTC_COL:TTC_COL + 1].copy())}
        z_j = apply_rows(inp.z6, inp.feat, both, W6, b6, chunk)
        rec["logits_bit_identical_joint_rows_are_p2"] = bool(np.array_equal(z_j, stored))
        rec["picks_identity_joint"] = int((pick_logits(z_j) == inp.pick_banked).sum())
        ok = ok and rec["logits_bit_identical_joint_rows_are_p2"] and rec["picks_identity_joint"] == inp.n
    rec["n_tokens"] = inp.n
    rec["pass"] = bool(ok)
    return rec


def c1_reach(inp: DevInputs, rows_by_seed: dict, W6: np.ndarray, b6: np.ndarray, chunk: int = 512) -> dict:
    """C1, applier side ('built, tested, unreachable from its caller'): the fitted row REACHES the logits and the pick, and reaches nothing else.
    For every seed: the EP column differs from the stored one on > 0 entries; the other five columns are bit-identical (exactly 0.0 delta: the zero control).
    POSITIVE CONTROL (the probe can read a change): the NEGATED P2 EP row must change the pick on > 0 tokens -- this is the line that proves logits -> pick is wired."""
    stored = inp.z6.numpy()
    others = [c for c in range(R.N_COMP) if c != EP_COL]
    per_seed = {}
    ok = True
    for sd, (w, b) in rows_by_seed.items():
        z = apply_rows(inp.z6, inp.feat, {EP_COL: (w, b)}, W6, b6, chunk)
        ep_changed = int((z[..., EP_COL] != stored[..., EP_COL]).sum())
        others_equal = all(bool(np.array_equal(z[..., c], stored[..., c])) for c in others)
        per_seed[sd] = {"ep_column_entries_changed": ep_changed, "other_five_columns_bit_identical": others_equal,
                        "max_abs_delta_ep_logit": float(np.abs(z[..., EP_COL].astype(np.float64) - stored[..., EP_COL]).max())}
        ok = ok and ep_changed > 0 and others_equal
    z_neg = apply_rows(inp.z6, inp.feat, {EP_COL: (-W6[EP_COL], -b6[EP_COL:EP_COL + 1])}, W6, b6, chunk)
    pos = int((pick_logits(z_neg) != pick_logits(stored)).sum())                    # MUTATION-TARGET-C1-POS
    ok = ok and pos > 0
    return {"per_seed": per_seed, "positive_control_negated_p2_row_changed_picks": pos, "pass": bool(ok)}


def c4_fan_frozen(inp: DevInputs) -> dict:
    """C4: the fan is frozen.  The dump the features came from (proposals.npz) holds EXACTLY the table's 64 proposals on every token (max abs pose diff 0.0 m), in the same token and
    log order as the table and the feature file, with the same pick -- so `pdms[token, new_pick]` is the PDMS of the pose the new pick selects.
    Negative control: against the NEXT token's proposals the diff must read > 0."""
    same_tok = bool(np.array_equal(inp.token, inp.token_feat) and np.array_equal(inp.token, inp.token_dump))
    same_log = bool(np.array_equal(inp.log, inp.log_feat) and np.array_equal(inp.log, inp.log_dump))
    d = np.abs(inp.props_table.astype(np.float64) - inp.props_dump.astype(np.float64))
    mx = float(d.max()) if d.size else 0.0
    neg = float(np.abs(inp.props_table[:-1].astype(np.float64) - inp.props_dump[1:].astype(np.float64)).reshape(inp.n - 1, -1).max(1).min()) if inp.n > 1 else float("inf")
    pick_eq = bool(np.array_equal(inp.pick_banked, inp.pick_dump))
    return {"tokens_aligned": same_tok, "logs_aligned": same_log, "proposals_max_abs_m": mx, "n_tokens_bit_identical": int((d.reshape(inp.n, -1).max(1) == 0.0).sum()),
            "picks_equal": pick_eq, "negative_control_next_token_min_max_abs_m": neg, "negative_control_reads_FAIL": bool(neg > 0.0),
            "pass": bool(same_tok and same_log and mx == 0.0 and pick_eq and neg > 0.0)}                  # MUTATION-TARGET-C4-EXACT


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# scoring: PDMS by lookup, paired log-cluster bootstrap (the SAME draws for every quantity)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _x100(d: dict) -> dict:
    return {k: (100.0 * v if k in ("point", "lo", "hi") else v) for k, v in d.items()}


def per_log(inp: DevInputs, v: np.ndarray) -> np.ndarray:
    return np.bincount(inp.log_idx, weights=np.asarray(v, np.float64), minlength=inp.n_logs)


def score_arm(inp: DevInputs, out: DevOutcomes, new_pick: np.ndarray, base_pick: np.ndarray, draws: np.ndarray) -> dict:
    """Paired (new pick vs base pick) total gain in PDMS x100, the six components, the unsafe-pick and zero-pick rates; every interval from the SAME log draws."""
    ar = np.arange(inp.n)
    den = per_log(inp, np.ones(inp.n))
    pn, pb = out.pdms[ar, new_pick], out.pdms[ar, base_pick]
    rec = {"changed_picks": int((new_pick != base_pick).sum()), "n_tokens": inp.n,
           "new_pdms_x100": 100.0 * float(pn.mean()), "base_pdms_x100": 100.0 * float(pb.mean()),
           "gain_x100": _x100(R.paired_delta(per_log(inp, pn), per_log(inp, pb), den, draws))}          # MUTATION-TARGET-GAIN-SIGN: num_a = NEW, num_b = BASE
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


def leak_read(out: DevOutcomes, z_new: np.ndarray, z_base: np.ndarray) -> dict:
    """C7 on the dev table (REPORTED, never a bar): mean sigmoid(EP logit) on safe vs unsafe proposals, new vs P2's row."""
    def m(z):
        p = sigmoid64(z[..., EP_COL])
        return {"on_safe": float(p[out.safe].mean()) if out.safe.any() else float("nan"), "on_unsafe": float(p[~out.safe].mean()) if (~out.safe).any() else float("nan")}
    return {"new": m(z_new), "p2": m(z_base), "n_safe_proposals": int(out.safe.sum()), "n_unsafe_proposals": int((~out.safe).sum())}


def c0_gain_zero(inp: DevInputs, out: DevOutcomes, base_pick: np.ndarray, draws: np.ndarray) -> dict:
    """C0's outcome half: identical picks give a gain of EXACTLY 0.0 (point and both ends) and 0 changed picks."""
    r = score_arm(inp, out, base_pick.copy(), base_pick, draws)
    g = r["gain_x100"]
    return {"gain_point": g["point"], "gain_lo": g["lo"], "gain_hi": g["hi"], "changed_picks": r["changed_picks"], "pass": bool(g["point"] == 0.0 and g["lo"] == 0.0 and g["hi"] == 0.0 and r["changed_picks"] == 0)}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the registered objects: c* result, rows, C2, SR-TTC's rows
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _need(doc: dict, *path):
    cur = doc
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            raise DevRefusal(f"result file lacks {'/'.join(path)}")
        cur = cur[p]
    return cur


def check_waiver(guard: DevGuard, path: str | None, star_cfg_sha: str) -> dict | None:
    """E1 (PREREG s8): the Master Mind's recorded reason for reading the dev gate on a grid-edge c*.  Must name THIS c*."""
    if not path:
        return None
    doc, sha = load_json(guard, path)
    for k in ("waiver_of", "reason", "decided_by", "c_star_config_sha256"):
        if not doc.get(k):
            raise DevRefusal(f"E1 waiver {path} lacks a non-empty '{k}'")
    if doc["waiver_of"] != "E1":
        raise DevRefusal(f"waiver {path} waives {doc['waiver_of']!r}, not E1")
    if doc["c_star_config_sha256"] != star_cfg_sha:
        raise DevRefusal("E1 waiver names a different c* than the result under test")
    return {"path": path, "sha256": sha, "decided_by": doc["decided_by"], "reason": doc["reason"]}


def check_r1_result(guard: DevGuard, opts: DevOptions, W6: np.ndarray, b6: np.ndarray) -> dict:
    """Everything about epr1_result.json that must hold BEFORE a dev outcome is opened.  -> {doc, sha256, rows: {seed: (w, b, sha)}, waiver}"""
    path = os.path.join(opts.run_dir, "epr1_result.json")
    if not _io_exists(guard, path):
        raise DevRefusal(f"{path} does not exist: there is no c* to gate")
    doc, sha = load_json(guard, path)
    if sha != opts.registered_result_sha256:                                                     # MUTATION-TARGET-REGSHA
        raise DevRefusal(f"epr1_result.json sha256 {sha} != the REGISTERED {opts.registered_result_sha256}: c* and its sha256 must be written down before the dev gate is read (PREREG s7)")
    if _need(doc, "schema") != R.SCHEMA or _need(doc, "stage") != "R1":
        raise DevRefusal(f"result schema/stage {doc.get('schema')}/{doc.get('stage')} != {R.SCHEMA}/R1")
    if _need(doc, "fraction_pct") not in R.FRACTIONS:
        raise DevRefusal(f"fraction {doc['fraction_pct']} not in {R.FRACTIONS}")
    if [int(s) for s in _need(doc, "params", "seeds")] != list(R.SEEDS):
        raise DevRefusal(f"fit seeds {doc['params']['seeds']} != the registered {list(R.SEEDS)} (both seeds are required downstream, C3)")
    if _need(doc, "inputs", "pins_overridden"):
        raise DevRefusal(f"the fit overrode registered pins {doc['inputs']['pins_overridden']}: it is not the registered object")
    if _need(doc, "code_sha256") != opts.epr1_sha256:
        raise DevRefusal(f"the fit's code sha256 {doc['code_sha256']} != the landed epr1.py {opts.epr1_sha256}")
    cs = _need(doc, "c_star")
    p = _need(doc, "params")
    star = R.Cfg(lr=cs["lr"], anchor=cs["anchor"], epochs=p["epochs"], batch_sets=p["batch_sets"])
    if star.sha256() != cs["config_sha256"]:
        raise DevRefusal("c* does not reproduce its recorded config sha256")
    p2_row_sha = sha256_bytes(R.row_bytes(W6[EP_COL], b6[EP_COL:EP_COL + 1]))
    if _need(doc, "inputs", "p2_row_sha256") != p2_row_sha:
        raise DevRefusal("the recorded P2 EP-row sha256 differs from the EP row of THIS checkpoint: the rows were fitted from a different P2")
    waiver = None
    if star.lr == R.GRID_LR[-1]:                                       # recomputed, not trusted from the flag   # MUTATION-TARGET-E1
        waiver = check_waiver(guard, opts.mm_edge_waiver, cs["config_sha256"])
        if waiver is None:
            raise DevRefusal("E1: c* sits at the grid edge (lr = 1e-3). The dev gate is NOT read on the edge c* before EP-R1b has run, unless the Master Mind records why "
                             "(--mm-edge-waiver, naming this c*). Nothing was read.")
    fr = _need(doc, "final_rows")
    rows = {}
    for sd in SEED_KEYS:
        ent = fr.get(sd)
        if not ent:
            raise DevRefusal(f"result lacks final_rows[{sd}]")
        w, b, rsha = load_row(guard, os.path.join(opts.run_dir, ent["file"]), ent["sha256"])
        rows[sd] = (w, b, rsha)
    return {"doc": doc, "sha256": sha, "rows": rows, "waiver": waiver, "star": star, "p2_row_sha256": p2_row_sha}


def check_c2_files(guard: DevGuard, opts: DevOptions, r1: dict) -> dict:
    """C2 is a bar condition: the shuffled-label rows and their result must exist, and belong to THIS c* and THIS bank."""
    path = os.path.join(opts.run_dir, "epr1_c2_result.json")
    if not _io_exists(guard, path):
        raise DevRefusal(f"{path} does not exist: C2 (the shuffled-label negative control) is a dev-gate condition; run `epr1.py c2` first. Nothing was read.")
    doc, sha = load_json(guard, path)
    if _need(doc, "stage") != "C2" or _need(doc, "c_star", "config_sha256") != r1["doc"]["c_star"]["config_sha256"]:
        raise DevRefusal("epr1_c2_result.json is not the C2 stage of this c*")
    if _need(doc, "bank_manifest_sha256") != r1["doc"]["inputs"]["bank_manifest_sha256"]:
        raise DevRefusal("epr1_c2_result.json was fitted on a different bank than epr1_result.json")
    rows = {}
    for sd in SEED_KEYS:
        ent = _need(doc, "final_rows", sd)
        w, b, rsha = load_row(guard, os.path.join(opts.run_dir, ent["file"]), ent["sha256"])
        rows[sd] = (w, b, rsha)
    return {"sha256": sha, "rows": rows, "doc": doc}


def check_ttc_rows(guard: DevGuard, opts: DevOptions, r1: dict, W6: np.ndarray, b6: np.ndarray) -> dict:
    """The JOINT's second row (SECONDARY).  Never a refusal: any miss is reported with its reason and the joint is NOT computed (PREREG s8: 'not computed, with that reason named')."""
    if not opts.ttc_run_dir:
        return {"available": False, "reason": "no SR-TTC run dir given (SR-TTC has no c* row at the time)"}
    try:
        path = os.path.join(opts.ttc_run_dir, "r1_result.json")
        if not _io_exists(guard, path):
            return {"available": False, "reason": f"{path} does not exist (SR-TTC R1 has not written c*)"}
        doc, sha = load_json(guard, path)
        if not opts.ttc_registered_result_sha256:
            return {"available": False, "reason": "no --ttc-registered-result-sha256: SR-TTC's c* sha256 was not registered beforehand"}
        if sha != opts.ttc_registered_result_sha256:
            return {"available": False, "reason": f"r1_result.json sha256 {sha} != the registered {opts.ttc_registered_result_sha256}"}
        if doc.get("schema") != "srttc-r1-v1" or doc.get("stage") != "R1":
            return {"available": False, "reason": f"SR-TTC result schema/stage {doc.get('schema')}/{doc.get('stage')}"}
        e, t = r1["doc"], doc
        for nm, a, b in (("fraction_pct", e["fraction_pct"], t.get("fraction_pct")), ("bank_manifest_sha256", e["inputs"]["bank_manifest_sha256"], t["inputs"].get("bank_manifest_sha256")),
                         ("ranked_fit_logs_sha256", e["inputs"]["ranked_fit_logs_sha256"], t["inputs"].get("ranked_fit_logs_sha256")),
                         ("replacements", e["inputs"]["replacements"], t["inputs"].get("replacements"))):
            if a != b:
                return {"available": False, "reason": f"EP-R1 and SR-TTC were not fitted on identical sets/logs: {nm} differs ({a!r} vs {b!r}); the joint would combine rows fitted on different data"}
        if t["inputs"].get("p2_row_sha256") != sha256_bytes(R.row_bytes(W6[TTC_COL], b6[TTC_COL:TTC_COL + 1])):
            return {"available": False, "reason": "SR-TTC's recorded P2 TTC-row sha256 differs from this checkpoint's"}
        rows = {}
        for sd in SEED_KEYS:
            ent = t["final_rows"][sd]
            w, b, rsha = load_row(guard, os.path.join(opts.ttc_run_dir, ent["file"]), ent["sha256"])
            rows[sd] = (w, b, rsha)
        return {"available": True, "rows": rows, "result_sha256": sha, "c_star": t["c_star"], "fraction_pct": t["fraction_pct"]}
    except (DevRefusal, KeyError) as ex:
        return {"available": False, "reason": f"SR-TTC rows unusable: {ex}"}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the verdict and the outcome rows of PREREG s8 (committed in advance)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def bar_check(gains: dict) -> dict:
    """Paired total gain >= +0.30 PDMS x100 on BOTH seeds (point estimates; 24 logs support no lower-bound rule)."""
    per = {sd: bool(gains[sd] >= DEV_BAR_X100) for sd in SEED_KEYS}
    return {"bar_x100": DEV_BAR_X100, "per_seed_meets": per, "both_seeds": bool(all(per[sd] for sd in SEED_KEYS))}      # MUTATION-TARGET-BOTH-SEEDS


def verdict_of(controls_pass: bool, bar: dict) -> str:
    if not controls_pass:
        return "VOID"
    return "PASS" if bar["both_seeds"] else "FAIL"


def outcome_rows(verdict: str, per_seed: dict, r1_doc: dict, lc: dict | None) -> dict:
    """The PREREG s8 rows that DEV can decide.  A (build void) / D-PASS / D-TRADE (per seed) / D-LEAK (OOF numbers of the fit, both seeds, whatever the dev gain) / D-FLAT / D-DATA (needs the LC call).
    Several rows can fire together; none is promoted over another here."""
    fired, notes = [], {}
    if verdict == "VOID":
        fired.append("A")
    if verdict == "PASS":
        fired.append("D-PASS")
    trade = {}
    for sd in SEED_KEYS:
        g = per_seed[sd]["gain_x100"]["point"]
        ep = per_seed[sd]["components"]["ep"]["delta_x100"]["point"]
        tt = per_seed[sd]["components"]["ttc"]["delta_x100"]["point"]
        trade[sd] = bool(g < DEV_BAR_X100 and ep > 0.0 and tt < 0.0)                            # MUTATION-TARGET-TRADE
    notes["d_trade_per_seed"] = trade
    if verdict != "VOID" and any(trade.values()):
        fired.append("D-TRADE")
    pk = r1_doc.get("pick_level_at_c_star", {})
    leak = {sd: (pk.get(sd, {}).get("delta_unsafe_rate", {}).get("lo", float("-inf")) > 0.0) for sd in SEED_KEYS}
    notes["d_leak_oof_lo_above_0_per_seed"] = leak
    if all(leak.values()):                                                                   # MUTATION-TARGET-LEAK
        fired.append("D-LEAK")
    if verdict == "FAIL" and "D-TRADE" not in fired and "D-LEAK" not in fired:
        if lc is None:
            notes["d_flat_or_data"] = "UNDETERMINED: no epr1_lc_result.json (the LC call decides D-FLAT vs D-DATA)"
        else:
            call = lc.get("call", {}).get("call")
            f = r1_doc["fraction_pct"]
            if call == "FLAT":
                fired.append("D-FLAT")
            elif call in ("RISING", "INCONCLUSIVE") and f < 100:
                fired.append("D-DATA")
            else:
                notes["d_flat_or_data"] = f"LC call {call} at f = {f} %: no registered row fires (RISING/INCONCLUSIVE at f = 100 %)"
    nxt = {"A": "fix and re-gate; no number from this run is read",
           "D-PASS": "navtest confirm (PREREG s7), only after BOTH dev gates are read or one arm is formally abandoned (MM_DECISIONS_EPR1 D1)",
           "D-TRADE": "EP-R1t (admit a pair only if TTC_i >= TTC_j); read the JOINT dev number: if it clears +0.30 on both seeds a NEW registration EP-R1J names the pair the primary",
           "D-LEAK": "EP-R1u: a NEW registration adding an unsafe-proposal anchor with target 0 at weight u in {0.03, 0.1, 0.3}",
           "D-FLAT": "feature-vs-capacity probe, CPU: a 2-layer EP probe on the cached features with the same folds",
           "D-DATA": "the probe above AND re-fit at the next nested size (SR-TTC's S0 increment); dev gate read #2, disclosed, the maximum"}
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
        if _io_exists(guard, p):                                                           # MUTATION-TARGET-ONCE
            raise DevRefusal(f"{p} exists: the dev gate is read ONCE (the attempt consumes the read, not only its success). Move the directory away only on a recorded Master Mind decision.")
    # --- 1. the code is the landed code
    if opts.epr1_sha256 is not None:
        got = sha256_bytes(_io_read_bytes(guard, os.path.abspath(R.__file__)))
        if got != opts.epr1_sha256:
            raise DevRefusal(f"epr1.py sha256 {got} != the landed {opts.epr1_sha256}: this applier is pinned to the code the rows were fitted with")
    # --- 2. the registered objects
    W6, b6 = load_p2_head(guard, opts.p2_ckpt)
    r1 = check_r1_result(guard, opts, W6, b6)
    c2 = check_c2_files(guard, opts, r1)
    ttc = check_ttc_rows(guard, opts, r1, W6, b6)
    lc = None
    lc_path = os.path.join(opts.run_dir, "epr1_lc_result.json")
    if _io_exists(guard, lc_path):
        lc = load_json(guard, lc_path)[0]
    # --- 3. the dev INPUTS (no outcome) and the controls that need none
    inp = load_dev_inputs(guard, opts)
    rows_ep = {sd: (r1["rows"][sd][0], r1["rows"][sd][1]) for sd in SEED_KEYS}
    ctl = {"integrity_featcheck": integrity_featcheck(inp, W6, b6, opts.featcheck_tol, opts.chunk), "C0_pick_identity": c0_identity(inp, W6, b6, opts.chunk),
           "C1_reach": c1_reach(inp, rows_ep, W6, b6, opts.chunk), "C4_fan_frozen": c4_fan_frozen(inp)}
    red = [k for k, v in ctl.items() if not v["pass"]]
    if red:                                                                                  # MUTATION-TARGET-REDGATE
        raise DevRefusal(f"control(s) RED before the dev outcomes were opened: {red} -- the instrument is wrong; the read was NOT spent. Detail: {json.dumps(_jsonable({k: ctl[k] for k in red}))[:1500]}")
    # --- 4. the read is consumed from here on
    started = {"schema": SCHEMA, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "c_star_result_sha256": r1["sha256"], "epr1_sha256": opts.epr1_sha256}
    _io_makedirs(guard, out_dir)
    _io_write_bytes(guard, marker_path, canon(started), exclusive=True)
    out = load_dev_outcomes(guard, opts)
    draws = R.boot_draws(inp.n_logs)
    stored = inp.z6.numpy()
    base_pick = pick_logits(stored)
    ctl["C0_gain_zero"] = c0_gain_zero(inp, out, base_pick, draws)
    # --- 5. the arms
    per_seed, c2_per_seed, z_new = {}, {}, {}
    for sd in SEED_KEYS:
        w, b, _sha = r1["rows"][sd]
        z_new[sd] = apply_rows(inp.z6, inp.feat, {EP_COL: (w, b)}, W6, b6, opts.chunk)
        new_pick = pick_logits(z_new[sd])                                                       # MUTATION-TARGET-PICK-NEW
        per_seed[sd] = score_arm(inp, out, new_pick, base_pick, draws)
        per_seed[sd]["leak_c7_dev"] = leak_read(out, z_new[sd], stored)
        cw, cb, _ = c2["rows"][sd]
        zc = apply_rows(inp.z6, inp.feat, {EP_COL: (cw, cb)}, W6, b6, opts.chunk)
        c2_per_seed[sd] = score_arm(inp, out, pick_logits(zc), base_pick, draws)
    ctl["C2_negative"] = {"per_seed_gain_x100": {sd: c2_per_seed[sd]["gain_x100"] for sd in SEED_KEYS}, "per_seed_changed_picks": {sd: c2_per_seed[sd]["changed_picks"] for sd in SEED_KEYS},
                          "rule": "the shuffled-label rows must give a dev gain <= 0 on each seed", "pass": bool(all(c2_per_seed[sd]["gain_x100"]["point"] <= 0.0 for sd in SEED_KEYS))}   # MUTATION-TARGET-C2
    controls_pass = bool(all(v["pass"] for v in ctl.values()))
    gains = {sd: per_seed[sd]["gain_x100"]["point"] for sd in SEED_KEYS}
    bar = bar_check(gains)
    bar["seed_gap_abs"] = abs(gains["0"] - gains["1"])
    bar["seed_floor_range_inherited"] = list(SEED_FLOOR_RANGE)
    verdict = verdict_of(controls_pass, bar)
    # --- 6. the JOINT secondary
    joint = {"label": "SECONDARY (PREREG s7/s8): EP-R1 row + SR-TTC's c* TTC row of the same seed. Never a bar, never selects, never promoted.", "computed": ttc["available"]}
    if ttc["available"]:
        joint["ttc_result_sha256"], joint["ttc_c_star"], joint["per_seed"] = ttc["result_sha256"], ttc["c_star"], {}
        for sd in SEED_KEYS:
            tw, tb, _ = ttc["rows"][sd]
            w, b, _s = r1["rows"][sd]
            zj = apply_rows(inp.z6, inp.feat, {EP_COL: (w, b), TTC_COL: (tw, tb)}, W6, b6, opts.chunk)
            pj = pick_logits(zj)
            rec = score_arm(inp, out, pj, base_pick, draws)
            rec["vs_ep_r1_alone"] = score_arm(inp, out, pj, pick_logits(z_new[sd]), draws)["gain_x100"]
            rec["meets_dev_bar_informational"] = bool(rec["gain_x100"]["point"] >= DEV_BAR_X100)
            joint["per_seed"][sd] = rec
        joint["both_seeds_meet_bar_informational"] = bool(all(joint["per_seed"][sd]["meets_dev_bar_informational"] for sd in SEED_KEYS))
    else:
        joint["reason"] = ttc["reason"]
    rows_fired = outcome_rows(verdict, per_seed, r1["doc"], lc)
    result = {
        "schema": SCHEMA, "stage": STAGE, "tier": "SELECTION (dev gate; not a claim: MM_DECISIONS_EPR1 D3)",
        "inputs": {"run_dir": opts.run_dir, "epr1_result_sha256": r1["sha256"], "registered_result_sha256": opts.registered_result_sha256, "c_star": r1["doc"]["c_star"],
                   "fraction_pct": r1["doc"]["fraction_pct"], "rows_sha256": {sd: r1["rows"][sd][2] for sd in SEED_KEYS}, "c2_result_sha256": c2["sha256"],
                   "c2_rows_sha256": {sd: c2["rows"][sd][2] for sd in SEED_KEYS}, "p2_ckpt": opts.p2_ckpt, "p2_ep_row_sha256": r1["p2_row_sha256"], "mm_edge_waiver": r1["waiver"],
                   "lc_result_present": lc is not None, "dev": inp.provenance, "dev_formula_check": out.formula, "epr1_sha256": opts.epr1_sha256, "pins_overridden": opts.pins_overridden()},
        "estimator": {"name": "paired log-cluster bootstrap (epr1.boot_draws / paired_delta == zero_tail.Boot draws)", "B": R.BOOT_B, "seed": R.BOOT_SEED, "n_logs": inp.n_logs, "n_tokens": inp.n,
                      "answers": "another draw of 24 LOGS only. The two fit seeds answer head-fit noise (H-ESTIM-SEED-1); inference is deterministic. 24 logs support no lower-bound rule: intervals are reported, never a bar."},
        "controls": ctl, "primary": {"per_seed": per_seed, "bar": bar},
        "dev_gate": {"verdict": verdict, "bar_x100": DEV_BAR_X100, "both_seeds_meet_bar": bar["both_seeds"], "controls_all_pass": controls_pass,
                     "strategic_family": "PAUSED (PI 2026-09-19, item 25): never a pass"},
        "joint_secondary": joint, "outcome_rows": rows_fired,
        "run_meta": {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elapsed_s": round(time.time() - t0, 1), "torch": torch.__version__, "numpy": np.__version__,
                     "threads": torch.get_num_threads(), "chunk": opts.chunk},
    }
    blob = canon(result)
    sha = _io_write_bytes(guard, result_path, blob, exclusive=True)
    _io_write_bytes(guard, os.path.join(out_dir, SIDECAR_NAME), (sha + "  " + RESULT_NAME + "\n").encode("utf-8"), exclusive=True)
    print(f"[{time.strftime('%H:%M:%S')}] dev gate {verdict}: gains x100 {gains}, wrote {result_path} sha256 {sha}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EP-R1 dev-gate applier: apply c*'s two EP rows to P2's held-out pre-head features, PDMS by lookup, paired log-cluster bootstrap. Read ONCE.")
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--run-dir", required=True, help="EP-R1's run dir (epr1_result.json, ep_row_seed{0,1}.npy, epr1_c2_result.json, ep_row_c2shuffle_seed{0,1}.npy)")
    ap.add_argument("--registered-result-sha256", required=True, help="the sha256 of epr1_result.json written down BEFORE this read (PREREG s7)")
    ap.add_argument("--ttc-run-dir", default=None, help="SR-TTC R1's run dir (r1_result.json, ttc_row_seed{0,1}.npy); omitted -> the JOINT secondary is not computed")
    ap.add_argument("--ttc-registered-result-sha256", default=None)
    ap.add_argument("--mm-edge-waiver", default=None, help="JSON {waiver_of: E1, reason, decided_by, c_star_config_sha256}; required iff c* is at the grid edge")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--dev-dir", default=DEV_DIR)
    ap.add_argument("--p2-ckpt", default=R.DEFAULT_P2_CKPT)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    opts = DevOptions(run_dir=a.run_dir, registered_result_sha256=a.registered_result_sha256, dev_dir=a.dev_dir, p2_ckpt=a.p2_ckpt, out_dir=a.out_dir, ttc_run_dir=a.ttc_run_dir,
                      ttc_registered_result_sha256=a.ttc_registered_result_sha256, mm_edge_waiver=a.mm_edge_waiver)
    try:
        run_dev(opts)
    except (DevRefusal, DevGuardError) as ex:
        print(f"REFUSED: {ex}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
