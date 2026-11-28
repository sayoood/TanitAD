"""r9pf_common.py -- shared pieces of PREREG_REFCV9_INMODEL_PA (sha256 67d17860...) PHASE A.  NEW file (EvalFlyWheel 2026-10-26).

* the prereg seal (every tool refuses unless the prereg hashes to 67d17860... AND to its .sha256 file, both 64 hex),
* the R9-PF code: `refcv9_progress.py` is imported BY FILE from a COPY of the patched tree (F:/refcv9_r9pf_scratch/tree/stack), never from the
  original F:/refcv9_inmodel and never from G:; `tanitad` itself comes from the eval tree tree_fb47a58 (the tree every beside arm was scored with),
* the calibration rows (the FRESH 300-clip split, H2cn's loader `l1c_fit_score.load_fresh`, S6 compliant rows = every slot valid + a defined regime),
* the frozen head file (write / load / predict) and the R9 composition (head -> decision -> profile -> monotone progress -> re-time).
No raw clip id is ever printed or written (sha12 only, and only inside npz files).
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np

WORK = Path("C:/Users/Admin/refcv8_pf/refcv9_r9pf")
CODE = WORK / "code"
HEADS = WORK / "heads"
RESULT = WORK / "result"
SCRATCH = Path("F:/refcv9_r9pf_scratch")
PREREG = WORK / "PREREG_REFCV9_INMODEL_PA.md"
PREREG_SHA = "67d178600a87ba128d8b5e76adbbf32a6fcfe7ac5374e38f1375ddd1fe918da8"
R9_TREE = SCRATCH / "tree" / "stack"
R9_FILE = R9_TREE / "tanitad/refs/refcv9_progress.py"
R9_ORIG_FILE = Path("F:/refcv9_inmodel/tree/stack/tanitad/refs/refcv9_progress.py")
SUBDIR = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub")
FRESH_BANK = Path("F:/l1c/bank/ztac_fresh.npz")
FRESH_BANK_SHA = "c07c7849c7ba897aea29ee0068d970ae5d511a13f88f4113f25c033a3fa20755"     # H2CN_FREEZE_INFO.json provenance.fresh_bank
EVAL_BANK = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/bank")
CKPT = "E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt"
FRESH_DUMP = Path("F:/l1c/tree/step48852ema_fresh/dump_s0")
HZ8 = [5, 10, 15, 20, 30, 40, 50, 60]
T8 = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
SLOTS = {"S2": [0, 1, 2, 3], "S6": [1, 3, 4, 5, 6, 7]}
ARMS = ("R9PF", "R9PF-TL", "R9PF-CONSTZ", "R9PF-SHUF")
FIT_SEEDS = (0, 1, 2)
HEAD_SCHEMA = "tanitad.refcv9_r9pf.head/1"
# LON6 (l2_common): v7 ids (FOLLOW, CRUISE, BRAKE_TO, CREEP, HOLD, ACCELERATE) -> regime (0 decel, 1 cruise, 2 accel)
LON6_V7_IDS = (0, 1, 3, 4, 5, 7)
LON6_TO_REGIME = (1, 1, 0, 0, 0, 2)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hex64(s: str) -> str:
    try:
        t = "".join(c for c in str(s).split()[0] if c in "0123456789abcdefABCDEF").lower()
    except Exception:          # noqa: BLE001
        return ""
    return t if len(t) == 64 else ""


def check_prereg() -> dict:
    """REFUSES (SystemExit) unless the prereg hashes to the sealed value AND to its .sha256 file (both sides 64 hex)."""
    got = sha256_file(PREREG) if PREREG.is_file() else ""
    seal = ""
    try:
        seal = hex64(PREREG.with_suffix(".sha256").read_text(encoding="utf-8-sig"))
    except Exception:          # noqa: BLE001
        pass
    if len(got) != 64 or len(seal) != 64:
        raise SystemExit(f"[r9pf] INCONCLUSIVE prereg seal: file '{got}' seal '{seal}' -- refusing")
    if not (got == seal == PREREG_SHA):
        raise SystemExit(f"[r9pf] the prereg does not match its seal ({got} / {seal} / {PREREG_SHA}) -- refusing")
    return {"file": str(PREREG), "sha256": got, "seal_matches": True}


def setup_paths() -> None:
    sys.path.insert(0, str(SUBDIR))
    import ship_r8_sub_common as C
    C.setup_paths()


_R9 = {}


def r9p():
    """refcv9_progress.py from the COPY of the patched tree, loaded BY FILE (tanitad itself = the eval tree's).  Refuses unless the copy is
    byte-equal to the banked original (sha256, both 64 hex)."""
    if "m" in _R9:
        return _R9["m"]
    a, b = sha256_file(R9_FILE), sha256_file(R9_ORIG_FILE)
    if len(a) != 64 or len(b) != 64 or a != b:
        raise SystemExit(f"[r9pf] refcv9_progress.py copy {a} != original {b} -- refusing")
    spec = importlib.util.spec_from_file_location("r9pf_refcv9_progress", str(R9_FILE))
    m = importlib.util.module_from_spec(spec)
    sys.modules["r9pf_refcv9_progress"] = m
    spec.loader.exec_module(m)
    _R9["m"], _R9["sha256"] = m, a
    return m


def regime_from_lon_v7(lon_v7) -> np.ndarray:
    """v7 lon label id -> regime (LON6 groups); any id outside LON6 -> -1."""
    lut = {v: r for v, r in zip(LON6_V7_IDS, LON6_TO_REGIME)}
    out = np.full(len(lon_v7), -1, np.int64)
    for i, x in enumerate(np.asarray(lon_v7, np.float64)):
        if np.isfinite(x) and int(round(x)) in lut and abs(x - round(x)) < 1e-6:
            out[i] = lut[int(round(x))]
    return out


def regime_independent(p5, p6, v0) -> np.ndarray:
    """The MD/FD cells' class (s6_lon_decomp.plan_lon_class: |P6 - P5| / 1.0 s - v0, +-1.5 m/s), float64 numpy, as ints 0/1/2."""
    v_end = np.linalg.norm(np.asarray(p6, np.float64)[:, :2] - np.asarray(p5, np.float64)[:, :2], axis=-1) / 1.0
    dv = v_end - np.asarray(v0, np.float64)
    return np.where(dv <= -1.5, 0, np.where(dv >= 1.5, 2, 1)).astype(np.int64), dv


# ------------------------------------------------------------------------------------------------ the frozen head
def ser(obj) -> bytes:
    import torch
    f = io.BytesIO()
    torch.save(obj, f)
    return f.getvalue()


def head_payload(arm: str, fit_seed: int, decision: str, sd32: dict, fit: dict, constz_mean, provenance: dict) -> dict:
    import torch
    R = r9p()
    cfg = R.R9Config(enable=True, decision=decision, hidden=128, decel_tau=None, w_prog=1.0, w_regime=1.0, calib_stage=True, seed=20261026)
    return {"schema": HEAD_SCHEMA, "arm": arm, "fit_seed": int(fit_seed), "decision": decision, "r9_config": cfg.to_dict(),
            "d_lat": 512, "n_slots": 8, "horizons_ticks": list(HZ8), "regimes": list(R.REGIMES), "scalar_names": list(R.SCALAR_NAMES),
            "state_dict": {k: v.detach().clone().contiguous() for k, v in sd32.items()},
            "constz_mean": (None if constz_mean is None else torch.from_numpy(np.ascontiguousarray(np.asarray(constz_mean, np.float32)))),
            "fit": fit, "provenance": provenance,
            "use": "model.enable_refcv9(R9Config(**r9_config)); model.r9_prog.load_state_dict(state_dict) (float32, eval mode).  "
                   "R9PF-CONSTZ: z_tac := constz_mean at eval.  Decision 'trunk_lon' = the trunk lon head argmax grouped to regimes."}


def load_head(path, registered_sha: str | None = None):
    """-> (head module float32 eval, blob, sha).  With registered_sha, REFUSES unless sha256(file) equals it (both 64 hex)."""
    import torch
    got = sha256_file(path)
    if registered_sha is not None:
        reg = hex64(registered_sha)
        if len(got) != 64 or len(reg) != 64:
            raise SystemExit(f"[r9pf] INCONCLUSIVE head sha256: file '{got}' registered '{reg}' -- refusing")
        if got != reg:
            raise SystemExit(f"[r9pf] head {path} hashes to {got}, registered {reg} -- refusing")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("schema") != HEAD_SCHEMA:
        raise SystemExit(f"[r9pf] head schema {blob.get('schema')!r}")
    R = r9p()
    head = R.R9ProgressHead(int(blob["d_lat"]), int(blob["n_slots"]), int(blob["r9_config"]["hidden"]))
    head.load_state_dict(blob["state_dict"], strict=True)
    head.eval()
    return head, blob, got


def compose(head, blob: dict, z, x9, prior8, plan8, lon_logits_v7=None) -> dict:
    """The R9 plan exactly as RefCV3Model._r9_forward computes it (eval mode, float32, CPU): s_P = arc_at_slots(prior8); head_forward
    (decision regime or trunk_lon); s_t = monotone(s_P + dS[r]); traj = retime(plan8, s_t).  z := constz_mean for R9PF-CONSTZ."""
    import torch
    R = r9p()
    cfg = R.R9Config(**blob["r9_config"])
    z = np.asarray(z, np.float32)
    if blob.get("constz_mean") is not None:
        z = np.broadcast_to(blob["constz_mean"].numpy().astype(np.float32), z.shape).copy()
    with torch.no_grad():
        zt = torch.from_numpy(np.ascontiguousarray(z))
        xt = torch.from_numpy(np.ascontiguousarray(np.asarray(x9, np.float32)))
        s_p = R.arc_at_slots(torch.from_numpy(np.ascontiguousarray(np.asarray(prior8, np.float32))))
        lon = None if lon_logits_v7 is None else torch.from_numpy(np.ascontiguousarray(np.asarray(lon_logits_v7, np.float32)))
        o = R.head_forward(head, cfg, zt, xt, s_p, lon)
        traj = R.retime(torch.from_numpy(np.ascontiguousarray(np.asarray(plan8, np.float32))), o["r9_s_target"]).to(torch.float32)
    return {"traj": traj.numpy(), "decision": o["r9_decision"].numpy().astype(np.int64), "s_target": o["r9_s_target"].numpy(),
            "s_prior": s_p.numpy(), "regime_logits": o["r9_regime_logits"].numpy(), "profiles": o["r9_profiles"].numpy()}


def x9_inmodel(win, lim) -> np.ndarray:
    """The head's 9 scalars with the IN-MODEL builders (refcv9_progress.ego_t0 + admissible_scalars, float32) from the observed pose window
    [N, 8, 4] and the fed ceiling (non-finite = invalid).  Cross-checked against progress_scalar.feats (H2cn's builder) in A1."""
    import torch
    R = r9p()
    pw = torch.from_numpy(np.ascontiguousarray(np.asarray(win, np.float32)))
    v0, a0, om = R.ego_t0(pw, 8, 0.1)
    lt = torch.from_numpy(np.ascontiguousarray(np.asarray(lim, np.float32)))
    return R.admissible_scalars(v0, a0, om, lt, None).numpy()


def jdump(obj, path) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ AMENDMENT 1 (MM, 2026-10-26)
AMEND1 = WORK / "PREREG_REFCV9_INMODEL_PA_AMENDMENT1.md"
AMEND1_SHA = "1c25a0640def6378ed33221d016dad7925a6784d5c39eb2082b8d198cd1714ac"
FIX_PIN = WORK / "FIX_R9_RNG.sha256"
FIX_TREE = SCRATCH / "launch_tree_fix"


def check_amendment1() -> dict:
    got = sha256_file(AMEND1) if AMEND1.is_file() else ""
    seal = hex64(AMEND1.with_suffix(".sha256").read_text(encoding="utf-8-sig")) if AMEND1.with_suffix(".sha256").is_file() else ""
    if len(got) != 64 or len(seal) != 64 or not (got == seal == AMEND1_SHA):
        raise SystemExit(f"[r9pf] AMENDMENT1 seal mismatch / INCONCLUSIVE ({got} / {seal} / {AMEND1_SHA}) -- refusing")
    return {"file": str(AMEND1), "sha256": got, "seal_matches": True}


def check_fix_pin() -> dict:
    """AMENDMENT1 change 1: every '<sha256>  <path>' line of FIX_R9_RNG.sha256 re-hashes now (both 64 hex) -- else REFUSE."""
    pins, bad = {}, []
    for ln in FIX_PIN.read_text(encoding="utf-8-sig").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        w, p = ln.split(None, 1)
        want, got = hex64(w), (sha256_file(p.strip()) if Path(p.strip()).is_file() else "")
        pins[p.strip()] = want
        if len(want) != 64 or len(got) != 64 or want != got:
            bad.append({"path": p.strip(), "pinned": want, "now": got})
    if bad or len(pins) < 3:
        raise SystemExit(f"[r9pf] FIX_R9_RNG pin check failed: {bad} -- refusing")
    return {"file": str(FIX_PIN), "sha256": sha256_file(FIX_PIN), "pins": pins, "ALL_MATCH": True}
