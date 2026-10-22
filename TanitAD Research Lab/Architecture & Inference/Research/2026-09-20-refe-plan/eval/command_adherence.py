#!/usr/bin/env python3
"""TACTICAL "command adherence" row of eval/PREREG_P6.md, EXACTLY as defined by PREREG_P6 "Amendment 2026-10-22 (b)" and
raw/2026-10-22-p6-read-gaps/MM_RULING_P6_READ_GAPS.md section E3 / G2c (P6-ADHERENCE, 2026-10-22). 0 GPU, CPU only.

THE DEFINITION (binding, copied, not re-derived):
  classing    the canonical labeller `tanitad.refs.refc_tactical.factor_from_kinematics`, reached exactly the way
              `taniteval/taniteval/four_families.py::tactical_from_trajectory` reaches it (`maneuver_kinematics` -> the LATERAL
              factor: lane_keep / turn_left / turn_right = STRAIGHT / LEFT / RIGHT). NO new threshold lives in this file.
  plan        the EXECUTED plan after the A7 transform (the plan PDMS scores) = the seam npz's `poses` [N, 8, 3]. The raw proposal is DESCRIPTIVE only.
  command     NAVSIM's `driving_command` one-hot (LEFT, STRAIGHT, RIGHT, UNKNOWN), the last `ego_statuses` entry of the W3 export.
  statistics  accuracy AND Cohen's kappa on ALL tokens (12,146: LEFT 2,501 / STRAIGHT 8,070 / RIGHT 1,575 / UNKNOWN 0) AND on the TURN-ACTIVE
              subset (LEFT u RIGHT commands, 4,076 tokens). Turn-active accuracy is the HEADLINE cell. The 3x3 confusion is always reported.
  floor       the random-proposal floor: the mean adherence of the 64 proposals per token, from the `--dump-proposals` dump.
  interval    the paired log-cluster bootstrap, 10,000 resamples, seed 20260927, against C1 (SFT-4 arm B u400) and C2 (P2) SEPARATELY, on identical tokens.
  authority   NONE. TACTICAL is a reported family; it never decides PREREG_P6's rule (PDMS vs C1/C2 and the goal bar).

THE ESTIMATOR (found, not chosen): `proxy_eval.boot` (eval/proxy_eval.py:313-327), the function `stage3_full.py` calls at line 216 with
N_BOOT, SEED = 10000, 20260927 (stage3_full.py:77); selection_quality.py uses the same one (selection_quality.py:38,45,164,186).
`taniteval/ci.py::episode_cluster_bootstrap` (ci.py:225, rng.choice draws, n_boot 2000 in the families blocks) is the FAMILIES block's estimator and is NOT used here.
`proxy_eval.boot` resamples LOGS with the models held fixed and is only defined for a per-token MEAN. Accuracy is a mean, so it goes through the
registered resample stream unchanged; Cohen's kappa is a function of the 3x3 confusion, so this file uses the SAME resample stream (default_rng(seed),
`rng.integers(0, G, G)` per resample, logs sorted) and sums per-log confusion matrices instead of concatenating tokens. That is the identical
resample, not a second estimator: the accuracy interval it gives is bit-equal to `proxy_eval.boot` on the same tokens (gated on every read, tested).
It answers ONE question, "would another draw of LOGS say this?"; training variance (one seed per arm, H-ESTIM-SEED-1) is unmeasured; inference is deterministic.

    python eval/command_adherence.py read    --name p6_full --dump <seams/stage2_p6_full.proposals.npz> --base u400 --also p2 [--out <json>]
    python eval/command_adherence.py control                      # known-value + RED controls on the BANKED P2 / u400 arms; touches no P6 file

Every step prints ZZADH_* markers. Any failed gate = INVALID (written into the JSON, exit 1), never a number. Evidence class of a number this file
produces: MEASURED (ours, with its input artifact). Tier: T1 (NAVSIM v1.1 plan, self-action open loop; the command is an INPUT of the agent).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")          # a CPU instrument: it must never take GPU memory on a box that may be training

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))

DATA = Path(os.environ.get("REFE_NAVTEST_DIR", "E:/Projects/TanitAD/data/refe_navtest"))
EXPORT = os.environ.get("REFE_NAVTEST_EXPORT", "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz")
P2_TABLE = os.environ.get("REFE_P2_TABLE", "D:/Projects/TanitAD-artifacts/p2_full_table/table_p2_full.npz")      # P2's banked 64-proposal table (a floor control)
REPO = Path(os.environ.get("TANITAD_REPO") or (HERE.parents[4] if len(HERE.parents) > 4 else "E:/Projects/TanitAD"))

# ---------------------------------------------------------------- the definition, as constants (each one cited)
CMD_NAMES = ("LEFT", "STRAIGHT", "RIGHT", "UNKNOWN")         # NAVSIM driving_command one-hot order, MEASURED: refe/planner.py:184-186 (DRIVING_COMMANDS)
LEFT, STRAIGHT, RIGHT, UNKNOWN = range(4)
PLAN_NAMES = CMD_NAMES[:3]
# refc_tactical.LAT_CLASSES = ("lane_keep", "turn_left", "turn_right") -> the command space. The ONLY mapping this file owns (tested, mutation-tested).
LAT_TO_CMD = np.array([STRAIGHT, LEFT, RIGHT], dtype=np.int64)
EXPECT_COUNTS = {"LEFT": 2501, "STRAIGHT": 8070, "RIGHT": 1575, "UNKNOWN": 0}      # MM_RULING_P6_READ_GAPS E3 Q2
EXPECT_N, EXPECT_TURN, EXPECT_LOGS = 12146, 4076, 136
N_POSES, DT_S = 8, 0.5                                        # NAVSIM v1 TrajectorySampling (refe_navtest_seam.py NAVSIM_DT, NAVSIM_N); families6.py:76 dt_s=0.5
N_BOOT, SEED = 10000, 20260927                                # stage3_full.py:77
M = 64
TIER = "T1 = NAVSIM v1.1 PDMS-side plan, an ego pseudo-simulation of an open-loop plan against logged agents (self-action open loop, not closed loop); the command is an INPUT of the agent"
ESTIMATOR = ("paired log-cluster bootstrap over the navtest logs (cluster unit log_name), 10,000 resamples, 95 %, seed 20260927: the resample stream of "
             "proxy_eval.boot (proxy_eval.py:313; called by stage3_full.py:216), applied to per-log confusion matrices so Cohen's kappa shares it. It answers "
             "'another draw of LOGS' and nothing else; training variance (one seed per arm, H-ESTIM-SEED-1) is unmeasured; inference is deterministic.")
AUTHORITY = ("NONE. TACTICAL is a reported family; it carries no decision authority over PREREG_P6's rule (PDMS vs C1/C2 and the goal bar). "
             "Quote it only with the estimator and n above.")
DEFINITION = {
    "source": ["eval/PREREG_P6.md 'Amendment 2026-10-22 (b)' (Command adherence is DEFINED here)",
               "raw/2026-10-22-p6-read-gaps/MM_RULING_P6_READ_GAPS.md section E3 / G2c, Q1-Q5"],
    "classing": "tanitad.refs.refc_tactical.factor_from_kinematics via taniteval.four_families.maneuver_kinematics (the tactical_from_trajectory path); LATERAL factor; no new thresholds",
    "plan": "the executed (A7) plan = seam poses [N,8,3]; the raw proposal is descriptive only",
    "statistics": "accuracy AND Cohen's kappa on all tokens AND on the turn-active subset (LEFT u RIGHT commands); turn-active accuracy is the headline cell; 3x3 confusion always",
    "floor": "mean adherence of the 64 proposals per token (--dump-proposals dump)",
    "interval": "paired log-cluster bootstrap, B = 10,000, against C1 and C2 separately",
    "authority": AUTHORITY,
}


class PlanFormatError(ValueError):
    """the plan is not in the format the definition was written for (NAVSIM 8 poses at 0.5 s, ego frame): STOP, this is a question for the Master Mind"""


# ---------------------------------------------------------------- the canonical labeller (imported, never re-implemented)
_LAB = None


def labeller():
    """-> (refc_tactical module, four_families module). The repo's `stack/` and `taniteval/` go AFTER sys.path (as families6.py does) so a caller's tree is never shadowed."""
    global _LAB
    if _LAB is None:
        for root in (REPO / "stack", REPO / "taniteval"):
            if root.is_dir() and str(root) not in sys.path:
                sys.path.append(str(root))
        from tanitad.refs import refc_tactical as RT
        from taniteval import four_families as FF
        _LAB = (RT, FF)
    return _LAB


def labeller_provenance() -> dict:
    RT, FF = labeller()
    out = {"factor_from_kinematics": f"{RT.factor_from_kinematics.__module__}.factor_from_kinematics",
           "maneuver_kinematics": f"{FF.maneuver_kinematics.__module__}.maneuver_kinematics",
           "thresholds": {k: getattr(RT, k) for k in ("YAW_TURN_RAD", "DV_ACCEL_MS", "DV_BRAKE_MS", "STOP_V_MS", "MOVING_V_MS", "LABEL_HORIZON")},
           "lateral_gate": "v1: |dyaw| > YAW_TURN_RAD (kappa=None), dyaw = path tangent at the horizon in the ego frame",
           "lat_classes": list(RT.LAT_CLASSES), "lat_to_command": {RT.LAT_CLASSES[i]: CMD_NAMES[int(LAT_TO_CMD[i])] for i in range(3)},
           "files": {}}
    for nm, mod in (("refc_tactical.py", RT), ("four_families.py", FF)):
        p = Path(mod.__file__)
        out["files"][nm] = {"path": str(p), "sha256_lf": hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
    return out


# ---------------------------------------------------------------- inputs
def check_plan_format(poses, what: str = "plan") -> np.ndarray:
    """[N, 8, >=2] finite. Anything else is a STOP: the definition fixes the labeller's call for REFe's NAVSIM plan (8 poses, 0.5 s, ego frame) and nothing else."""
    a = np.asarray(poses)
    if a.ndim != 3 or a.shape[1] != N_POSES or a.shape[2] < 2:
        raise PlanFormatError(f"{what}: shape {a.shape}, expected [N, {N_POSES}, 3] (NAVSIM v1: 8 poses at {DT_S} s, ego frame x forward / y left). "
                              f"Any other pose count, rate or frame needs a parameter the definition does not fix: ask the Master Mind, do not choose.")
    if not np.isfinite(a[..., :2]).all():
        raise PlanFormatError(f"{what}: non-finite xy")
    return a


def plan_classes(poses, chunk: int = 200_000):
    """executed plans [N,8,3] -> (class in COMMAND space {LEFT, STRAIGHT, RIGHT} [N] int64, provenance dict).
    One call of the canonical path per chunk, float32 exactly as the families path feeds it (`torch.as_tensor(win['pred_dense']).float()`)."""
    import torch
    RT, FF = labeller()
    a = check_plan_format(poses)
    out = np.empty(a.shape[0], dtype=np.int64)
    held = never = 0
    for s in range(0, a.shape[0], chunk):
        w = torch.as_tensor(np.ascontiguousarray(a[s:s + chunk, :, :2])).float()
        dyaw, dv, v0, v1, prov = FF.maneuver_kinematics(w, DT_S)
        lat, _lon = RT.factor_from_kinematics(dyaw, dv, v0, v1)
        out[s:s + chunk] = LAT_TO_CMD[lat.numpy()]
        held += prov["n_heading_held_from_last_moving_step"]
        never += prov["n_never_moved"]
    return out, {"n": int(a.shape[0]), "n_heading_held_from_last_moving_step": int(held), "n_never_moved": int(never), "dt_s": DT_S, "n_poses": N_POSES}


def load_export(path=None) -> dict:
    return json.load(gzip.open(path or EXPORT, "rt", encoding="utf-8"))["tokens"]


def commands_and_logs(tokens, exp: dict):
    """-> (cmd [N] in {0,1,2,3}, logs [N] str). STRICT: the one-hot must be finite, binary and sum to 1, or the token is refused (never defaulted)."""
    cmd = np.empty(len(tokens), dtype=np.int64)
    logs = []
    for i, t in enumerate(tokens):
        rec = exp[t]
        st = rec.get("ego_statuses") or []
        c = st[-1].get("driving_command") if st else None
        if c is None:
            raise ValueError(f"token {t}: no driving_command in ego_statuses[-1] of the NAVSIM agent input")
        v = np.asarray(c, dtype=np.float64).reshape(-1)
        if v.size != 4 or not np.isfinite(v).all() or not set(np.unique(v).tolist()) <= {0.0, 1.0} or v.sum() != 1.0:
            raise ValueError(f"token {t}: driving_command {c!r} is not a one-hot 4-vector")
        cmd[i] = int(np.argmax(v))
        logs.append(str(rec["log_name"]))
    return cmd, np.array(logs)


def load_seam(path) -> dict:
    """seam npz -> {token: row} with poses. Mirrors families6.py: rows whose `source` is 'cv_standin' are not model rows and are excluded (and counted)."""
    z = np.load(path, allow_pickle=False)
    tok = [str(t) for t in z["token"]]
    src = [str(s) for s in z["source"]] if "source" in z.files else ["model"] * len(tok)
    keep = [i for i in range(len(tok)) if src[i] != "cv_standin"]
    poses = check_plan_format(z["poses"][keep], f"{path} poses")
    return {"path": str(path), "tokens": [tok[i] for i in keep], "poses": poses, "n_cv_standin_dropped": len(tok) - len(keep)}


def align(seam: dict, tokens) -> np.ndarray:
    idx = {t: i for i, t in enumerate(seam["tokens"])}
    miss = [t for t in tokens if t not in idx]
    if miss:
        raise ValueError(f"{seam['path']}: {len(miss)} of {len(tokens)} tokens absent (e.g. {miss[:3]})")
    return seam["poses"][np.array([idx[t] for t in tokens])]


# ---------------------------------------------------------------- the statistics (counts in, numbers out)
def confusion(cmd, plan) -> np.ndarray:
    """3x3, rows = command (LEFT, STRAIGHT, RIGHT), cols = executed-plan class. UNKNOWN-command tokens are NOT in it (they are counted by the caller)."""
    cm = np.zeros((3, 3), dtype=np.int64)
    m = cmd < 3
    np.add.at(cm, (cmd[m], plan[m]), 1)
    return cm


def stats_from_counts(C):
    """C [..., 3, 3] (counts or expected counts) -> dict of arrays n, trace, accuracy, kappa. Cohen's kappa (po - pe) / (1 - pe), NaN where 1 - pe <= 1e-9 (never a fake 1.0:
    `four_families._kappa_k` returns None there). Over the 3 command classes; a class absent from both raters adds 0 to pe, so this equals the programme's `_kappa_k(k=3)`."""
    C = np.asarray(C, dtype=np.float64)
    n = C.sum(axis=(-1, -2))
    tr = np.trace(C, axis1=-2, axis2=-1)
    row, col = C.sum(-1), C.sum(-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = tr / n
        pe = (row * col).sum(-1) / (n * n)
        kap = np.where((1.0 - pe) > 1e-9, (acc - pe) / (1.0 - pe), np.nan)
    return {"n": n, "trace": tr, "accuracy": acc, "kappa": kap}


_W: dict = {}


def resample_weights(G: int, n: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """[n, G] multiplicities of each log in each resample: proxy_eval.boot's stream (default_rng(seed); one `rng.integers(0, G, G)` per resample; logs sorted)"""
    key = (G, n, seed)
    if key not in _W:
        rng = np.random.default_rng(seed)
        W = np.empty((n, G), dtype=np.float64)
        for b in range(n):
            W[b] = np.bincount(rng.integers(0, G, G), minlength=G)
        _W[key] = W
    return _W[key]


def grouped(cmd, plan_or_frac, logs, mask=None):
    """per-log [G,3,3] confusion. `plan_or_frac` is a class vector [N] (integer counts) or class FRACTIONS [N,3] (the floor: each token spreads one unit over the 64 proposals' classes)"""
    cmd, logs = np.asarray(cmd), np.asarray(logs)
    mask = np.ones(len(cmd), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    ul = sorted(set(logs[mask & (cmd < 3)].tolist()))                      # a log enters only through a token that has a command class (UNKNOWN never does)
    pos = {l: i for i, l in enumerate(ul)}
    C = np.zeros((len(ul), 3, 3), dtype=np.float64)
    sel = np.flatnonzero(mask & (cmd < 3))
    g = np.array([pos[l] for l in logs[sel]], dtype=np.int64)
    P = np.asarray(plan_or_frac)
    if P.ndim == 1:
        np.add.at(C, (g, cmd[sel], P[sel]), 1.0)
    else:
        for j in range(3):
            np.add.at(C, (g, cmd[sel], np.full(len(sel), j)), P[sel, j])
    return ul, C


def _ci(x, alpha=2.5):
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    if not ok.any():
        return [None, None], int((~ok).sum())
    return [float(np.percentile(x[ok], alpha)), float(np.percentile(x[ok], 100 - alpha))], int((~ok).sum())


def cell(C, n: int = N_BOOT, seed: int = SEED, what: str = "") -> dict:
    """level block for per-log counts C [G,3,3]: full-set point estimate + percentile CI from the registered resample stream"""
    G = C.shape[0]
    point = stats_from_counts(C.sum(0))
    st = stats_from_counts(np.einsum("bg,gij->bij", resample_weights(G, n, seed), C))
    acc_ci, _ = _ci(st["accuracy"])
    kap_ci, n_nan = _ci(st["kappa"])
    cm = C.sum(0)
    return {"n": float(point["n"]) if not float(point["n"]).is_integer() else int(point["n"]), "n_logs": int(G),
            "accuracy": {"point": float(point["accuracy"]), "ci95": acc_ci},
            "kappa": {"point": None if not np.isfinite(point["kappa"]) else float(point["kappa"]), "ci95": kap_ci, "n_undefined_resamples": n_nan},
            "confusion_command_rows_plan_cols": cm.tolist(), "rows": list(PLAN_NAMES), "cols": list(PLAN_NAMES),
            "per_command_recall": {PLAN_NAMES[i]: (None if cm[i].sum() == 0 else float(cm[i, i] / cm[i].sum())) for i in range(3)},
            "per_command_n": {PLAN_NAMES[i]: float(cm[i].sum()) for i in range(3)},
            "_what": what}


def paired(Ca, Cb, n: int = N_BOOT, seed: int = SEED) -> dict:
    """arm - base on IDENTICAL tokens, the SAME resamples (that is what 'paired' means here). Delta accuracy = (trace_a - trace_b) / n (the per-token mean of the
    difference indicator, as proxy_eval.boot computes it); delta kappa = kappa(arm) - kappa(base) per resample."""
    assert Ca.shape == Cb.shape, (Ca.shape, Cb.shape)
    G = Ca.shape[0]
    W = resample_weights(G, n, seed)
    A, B = np.einsum("bg,gij->bij", W, Ca), np.einsum("bg,gij->bij", W, Cb)
    sa, sb = stats_from_counts(A), stats_from_counts(B)
    pa, pb = stats_from_counts(Ca.sum(0)), stats_from_counts(Cb.sum(0))
    d_acc_b = (sa["trace"] - sb["trace"]) / sa["n"]
    d_kap_b = sa["kappa"] - sb["kappa"]
    d_acc = (float(pa["trace"]) - float(pb["trace"])) / float(pa["n"])
    d_kap = float(pa["kappa"] - pb["kappa"])
    a_ci, _ = _ci(d_acc_b)
    k_ci, n_nan = _ci(d_kap_b)
    return {"n": float(pa["n"]), "n_logs": int(G), "accuracy_delta": {"point": d_acc, "ci95": a_ci, "separated": bool(a_ci[0] > 0 or a_ci[1] < 0)},
            "kappa_delta": {"point": d_kap, "ci95": k_ci, "separated": None if k_ci[0] is None else bool(k_ci[0] > 0 or k_ci[1] < 0), "n_undefined_resamples": n_nan}}


# ---------------------------------------------------------------- the random-proposal floor
def proposal_fractions(proposals, chunk: int = 20_000) -> np.ndarray:
    """proposals [N,64,8,3] -> [N,3] fractions of the 64 proposals in each COMMAND-space class (LEFT, STRAIGHT, RIGHT); rows sum to 1"""
    P = np.asarray(proposals)
    if P.ndim != 4 or P.shape[1] != M:
        raise PlanFormatError(f"proposals {P.shape}: expected [N, {M}, {N_POSES}, 3]")
    out = np.zeros((P.shape[0], 3), dtype=np.float64)
    for s in range(0, P.shape[0], chunk):
        blk = P[s:s + chunk]
        cls, _ = plan_classes(blk.reshape(-1, N_POSES, blk.shape[-1]))
        cls = cls.reshape(blk.shape[0], M)
        for j in range(3):
            out[s:s + chunk, j] = (cls == j).mean(1)
    return out


def load_dump(path) -> dict:
    z = np.load(path, allow_pickle=False)
    for k in ("token", "proposals", "pick"):
        if k not in z.files:
            raise ValueError(f"{path}: key {k!r} absent (keys {z.files}); not a --dump-proposals dump")
    return {"path": str(path), "tokens": [str(t) for t in z["token"]], "proposals": z["proposals"], "pick": z["pick"].astype(np.int64)}


# ---------------------------------------------------------------- the read
def _cells(cmd, plan, logs, n_boot, seed, label):
    ta = (cmd == LEFT) | (cmd == RIGHT)
    ul_a, Ca = grouped(cmd, plan, logs)
    ul_t, Ct = grouped(cmd, plan, logs, ta)
    return ul_a, Ca, ul_t, Ct, {"all_tokens": cell(Ca, n_boot, seed, f"{label}: all tokens"),
                                "turn_active": cell(Ct, n_boot, seed, f"{label}: turn-active (LEFT u RIGHT commands) = HEADLINE")}


def verify_boot_equal(cmd, plan_by_arm: dict, logs, tokens, arm_cells: dict, pair_cells: dict, n_boot, seed) -> dict:
    """gate: the accuracy intervals this file reports are bit-equal to proxy_eval.boot (the registered function) on the same tokens, levels AND paired differences"""
    try:
        import proxy_eval as PE
    except Exception as e:                                                  # noqa: BLE001
        return {"ok": None, "status": f"NOT VERIFIED: proxy_eval not importable ({type(e).__name__}: {e})"}
    tl = {t: str(l) for t, l in zip(tokens, logs)}
    bad, n_checked = [], 0
    for sub in ("all_tokens", "turn_active"):
        mask = (cmd < 3) if sub == "all_tokens" else ((cmd == LEFT) | (cmd == RIGHT))
        sel = [i for i in np.flatnonzero(mask)]
        ind = {a: {tokens[i]: float(plan[i] == cmd[i]) for i in sel} for a, plan in plan_by_arm.items()}
        tls = {tokens[i]: tl[tokens[i]] for i in sel}
        for a, cells_ in arm_cells.items():
            mu, lo, hi = PE.boot(ind[a], tls, n=n_boot, seed=seed)
            got = cells_[sub]["accuracy"]
            n_checked += 1
            if (mu, lo, hi) != (got["point"], got["ci95"][0], got["ci95"][1]):
                bad.append({"level": a, "subset": sub, "PE": [mu, lo, hi], "ours": [got["point"], *got["ci95"]]})
        for (a, b), pc in pair_cells.items():
            d = {t: ind[a][t] - ind[b][t] for t in ind[a]}
            mu, lo, hi = PE.boot(d, tls, n=n_boot, seed=seed)
            got = pc[sub]["accuracy_delta"]
            n_checked += 1
            if (mu, lo, hi) != (got["point"], got["ci95"][0], got["ci95"][1]):
                bad.append({"pair": f"{a}-{b}", "subset": sub, "PE": [mu, lo, hi], "ours": [got["point"], *got["ci95"]]})
    return {"ok": not bad, "n_checked": n_checked, "mismatches": bad, "status": "bit-equal to proxy_eval.boot" if not bad else "MISMATCH"}


def kappa_vs_programme(cmd, plan) -> dict:
    """gate: kappa and the confusion equal the programme's own `four_families._kappa_k` / `_class_report` on the same label vectors (1e-12)"""
    RT, FF = labeller()
    m = cmd < 3
    ours = stats_from_counts(confusion(cmd, plan))
    theirs = FF._kappa_k(cmd[m], plan[m], 3)
    rep = FF._class_report(cmd[m], plan[m], list(PLAN_NAMES))
    ok_cm = rep["confusion_gt_rows_pred_cols"] == confusion(cmd, plan).tolist()
    ok_k = (theirs is None and not np.isfinite(ours["kappa"])) or (theirs is not None and abs(float(ours["kappa"]) - theirs) < 1e-12)
    return {"ok": bool(ok_cm and ok_k), "kappa_ours": float(ours["kappa"]), "kappa_programme": theirs, "confusion_equal": bool(ok_cm)}


def run_read(arm: dict, comparators: list, dump: dict | None = None, exp: dict | None = None, expect_full: bool = True,
             n_boot: int = N_BOOT, seed: int = SEED, verify: bool = True, include_human: bool = True) -> dict:
    """arm = {"name", "seam" (path)}; comparators = [{"role": "C1"|"C2"|..., "name", "seam"}]. Returns the whole result JSON object (with `INVALID` set on any failed gate)."""
    t0 = time.time()
    exp = exp if exp is not None else load_export()
    names = [arm["name"]] + [c["name"] for c in comparators]
    if len(set(names)) != len(names):
        raise ValueError(f"arm and comparator names must be distinct (a collision would silently overwrite one seam): {names}")
    seams = {arm["name"]: load_seam(arm["seam"])}
    for c in comparators:
        seams[c["name"]] = load_seam(c["seam"])
    tokens = sorted(seams[arm["name"]]["tokens"])
    invalid: list = []
    for nm, s in seams.items():
        if set(s["tokens"]) != set(tokens) or len(s["tokens"]) != len(set(s["tokens"])):
            invalid.append(f"token set of {nm} differs from the arm's ({len(s['tokens'])} vs {len(tokens)} rows): pairing needs IDENTICAL tokens")
    if invalid:
        return {"INVALID": invalid, "definition": DEFINITION}
    cmd, logs = commands_and_logs(tokens, exp)
    counts = {CMD_NAMES[i]: int((cmd == i).sum()) for i in range(4)}
    n_turn = int(((cmd == LEFT) | (cmd == RIGHT)).sum())
    gates = {"G_FORMAT": {"ok": True, "n_poses": N_POSES, "dt_s": DT_S, "detail": "every plan is [N, 8, 3], finite xy (check_plan_format)"}}
    full_ok = counts == EXPECT_COUNTS and n_turn == EXPECT_TURN and len(set(logs.tolist())) == EXPECT_LOGS and len(tokens) == EXPECT_N
    gates["G_COUNTS"] = {"ok": bool(full_ok or not expect_full), "observed": counts, "n_turn_active": n_turn, "n_tokens": len(tokens), "n_logs": len(set(logs.tolist())),
                         "expected": {**EXPECT_COUNTS, "n": EXPECT_N, "turn_active": EXPECT_TURN, "logs": EXPECT_LOGS}, "enforced": bool(expect_full),
                         "note": "the full-navtest class counts of the ruling; a subset run must pass --subset-ok and is then NOT the registered read"}
    if not gates["G_COUNTS"]["ok"]:
        invalid.append("command class counts differ from the ruling's 2,501 / 8,070 / 1,575 / 0 (4,076 turn-active) on 12,146 tokens / 136 logs")
    gates["G_LABELLER"] = {"ok": True, **labeller_provenance()}

    plans, provs = {}, {}
    for nm, s in seams.items():
        plans[nm], provs[nm] = plan_classes(align(s, tokens))
    arm_cells, groups = {}, {}
    out_arms = {}
    for nm in seams:
        ul_a, Ca, ul_t, Ct, cl = _cells(cmd, plans[nm], logs, n_boot, seed, nm)
        arm_cells[nm], groups[nm] = cl, {"all_tokens": Ca, "turn_active": Ct}
        out_arms[nm] = {"executed_plan_A7": cl, "labeller_provenance": provs[nm],
                        "n_cv_standin_rows_dropped": seams[nm]["n_cv_standin_dropped"], "seam": seams[nm]["path"]}
    gates["G_KAPPA_PROGRAMME"] = kappa_vs_programme(cmd, plans[arm["name"]])
    if not gates["G_KAPPA_PROGRAMME"]["ok"]:
        invalid.append("kappa / confusion differ from four_families._kappa_k / _class_report")

    # descriptive raw (pre-A7) proposal: A7 touches ONLY poses[..., -1, 2] (refe/planner.py:159-167) and the labeller reads ONLY xy (four_families.maneuver_kinematics),
    # so the two are identical BY CONSTRUCTION; measured here by zeroing the whole heading column.
    a0 = align(seams[arm["name"]], tokens)
    z = a0.copy()
    z[..., 2:] = 0.0
    raw_cls, _ = plan_classes(z)
    out_arms[arm["name"]]["raw_proposal_descriptive"] = {
        "status": "IDENTICAL_BY_CONSTRUCTION", "n_class_differences_with_heading_column_zeroed": int((raw_cls != plans[arm["name"]]).sum()),
        "why": "A7 (planner.repair_last_heading) rewrites only the last pose's heading; the canonical labeller reads only xy. The pre-A7 plan is not banked in the seam or the dump (both hold executed forms), and cannot differ in class.",
        "descriptive_only": True}

    pairs, pair_cells = {}, {}
    for c in comparators:
        key = f"{arm['name']}_minus_{c['role']}_{c['name']}"
        pairs[key] = {sub: paired(groups[arm["name"]][sub], groups[c["name"]][sub], n_boot, seed) for sub in ("all_tokens", "turn_active")}
        pair_cells[(arm["name"], c["name"])] = pairs[key]

    # the random-proposal floor (arm's own dump)
    floor = {"status": "NOT COMPUTED: no --dump-proposals dump given for this arm (the floor is the mean adherence of its 64 proposals per token)"}
    if dump is not None:
        d_idx = {t: i for i, t in enumerate(dump["tokens"])}
        if set(d_idx) != set(tokens):
            invalid.append(f"proposal dump token set differs from the seam's ({len(d_idx)} vs {len(tokens)})")
            floor = {"status": "INVALID: token sets differ"}
        else:
            order = np.array([d_idx[t] for t in tokens])
            P, pk = dump["proposals"][order], dump["pick"][order]
            at_pick = P[np.arange(len(tokens)), pk]
            same = bool(np.array_equal(at_pick, a0.astype(at_pick.dtype)))
            gates["G_DUMP_PICK"] = {"ok": same, "detail": "dump proposals[pick] == the arm's seam poses, bit for bit (the dump IS the run that was scored)"}
            if not same:
                invalid.append("dump proposals[pick] != seam poses: the dump is not this arm's run")
            F = proposal_fractions(P)
            fl = {}
            for sub, mask in (("all_tokens", None), ("turn_active", (cmd == LEFT) | (cmd == RIGHT))):
                ul, Cf = grouped(cmd, F, logs, mask)
                ul2, Ca = grouped(cmd, plans[arm["name"]], logs, mask)
                assert ul == ul2
                fl[sub] = {"floor": cell(Cf, n_boot, seed, f"random-proposal floor: {sub}"), "actual_minus_floor": paired(Ca, Cf, n_boot, seed)}
                fl[sub]["floor_ge_actual_point"] = bool(fl[sub]["floor"]["accuracy"]["point"] >= float(stats_from_counts(Ca.sum(0))["accuracy"]))
            floor = {"definition": "mean over the 64 proposals' per-token adherence (the exact expectation of a skill-free selector); kappa is the kappa of the EXPECTED confusion matrix",
                     "n_proposals": M, "dump": dump["path"], **fl,
                     "note": "a floor at or above the actual selection is a FINDING (selection adds no adherence), not an instrument failure; it is flagged, never gated"}
    # controls that must read their known values on THIS arm (reported; the kappa identity is also a gate)
    sw = np.where(cmd == LEFT, RIGHT, np.where(cmd == RIGHT, LEFT, cmd))
    ctl = {}
    _, _, _, Cs, cs = _cells(sw, plans[arm["name"]], logs, n_boot, seed, "RED arm: LEFT<->RIGHT commands swapped")
    ctl["swap_LR_commands"] = {"turn_active_accuracy": cs["turn_active"]["accuracy"], "turn_active_kappa": cs["turn_active"]["kappa"], "all_tokens_accuracy": cs["all_tokens"]["accuracy"],
                               "real_turn_active_accuracy": arm_cells[arm["name"]]["turn_active"]["accuracy"]["point"],
                               "drops": bool(cs["turn_active"]["accuracy"]["point"] < arm_cells[arm["name"]]["turn_active"]["accuracy"]["point"]),
                               "expected": "turn-active accuracy falls sharply (a plan that follows its command is wrong on every swapped turn)"}
    const = np.full(len(tokens), STRAIGHT, dtype=np.int64)
    _, _, _, Ck, ck = _cells(cmd, const, logs, n_boot, seed, "zero arm: constant-STRAIGHT plan")
    k_all = ck["all_tokens"]["kappa"]["point"]
    k_turn = ck["turn_active"]["kappa"]["point"]
    ctl["constant_straight_plan"] = {"kappa_all_tokens": k_all, "kappa_turn_active": k_turn, "accuracy_all_tokens": ck["all_tokens"]["accuracy"]["point"],
                                     "accuracy_turn_active": ck["turn_active"]["accuracy"]["point"],
                                     "kappa_is_zero": bool(k_all is not None and abs(k_all) < 1e-12 and k_turn is not None and abs(k_turn) < 1e-12)}
    gates["G_KAPPA_ZERO_ID"] = {"ok": ctl["constant_straight_plan"]["kappa_is_zero"], "detail": "a constant-STRAIGHT plan has kappa 0 (an identity, independent of the arm)"}
    if not gates["G_KAPPA_ZERO_ID"]["ok"]:
        invalid.append("constant-STRAIGHT plan did not give kappa = 0")

    human = None
    if include_human:
        hum = np.asarray([exp[t]["human_future_poses"] for t in tokens], dtype=np.float64)
        hcls, hprov = plan_classes(hum)
        _, _, _, _, hc = _cells(cmd, hcls, logs, n_boot, seed, "human future (descriptive reference)")
        human = {"status": "DESCRIPTIVE REFERENCE, not part of the definition: the same labeller on the logged human future vs the same command", "cells": hc, "labeller_provenance": hprov}

    if verify:
        gates["G_PE_BOOT_EQUAL"] = verify_boot_equal(cmd, plans, logs, tokens, arm_cells, pair_cells, n_boot, seed)
        if gates["G_PE_BOOT_EQUAL"].get("ok") is False:
            invalid.append("accuracy intervals are not bit-equal to proxy_eval.boot")
    else:
        gates["G_PE_BOOT_EQUAL"] = {"ok": None, "status": "NOT VERIFIED (--no-verify-boot)"}

    res = {"instrument": "eval/command_adherence.py (P6-ADHERENCE)", "definition": DEFINITION, "tier": TIER, "estimator": ESTIMATOR, "authority": AUTHORITY,
           "evidence_class": "MEASURED (ours) on the named seams; a control read on banked arms is a known-value control, not a P6 look",
           "arm": arm["name"], "comparators": [{"role": c["role"], "name": c["name"]} for c in comparators],
           "tokens": {"n": len(tokens), "n_logs": len(set(logs.tolist())), "command_counts": counts, "turn_active_n": n_turn},
           "arms": out_arms, "paired_vs_comparators": pairs, "random_proposal_floor": floor, "controls": ctl, "reference_human_future": human,
           "strategic_family": "PAUSED (PI 2026-09-19, PI_DECISION_QUEUE item 25): not computed here, never a pass",
           "gates": gates, "n_boot": n_boot, "seed": seed, "seconds": round(time.time() - t0, 1)}
    if invalid:
        res["INVALID"] = invalid
    if dump is None:
        res["INCOMPLETE"] = ["random-proposal floor NOT computed: no --dump-proposals dump given. The floor is a required row of the definition (E3 Q5); a read without it is incomplete, never a pass"]
    return res


# ---------------------------------------------------------------- banked-arm controls
def families_lateral_control(arm_name: str, seam_path, exp: dict) -> dict:
    """known-value control: this file's labeller application on (executed plan, human future) reproduces the BANKED families block's lateral confusion matrix and
    heading-held / never-moved counts (stage3_full's md5-pinned families JSON for the arm), exactly"""
    import stage3_full as S3
    b = S3.pick_baseline(arm_name)
    fam = json.load(open(b["families"], encoding="utf-8"))["families"]["refcv6"]["tactical"]
    s = load_seam(seam_path)
    toks = sorted(s["tokens"])
    plan = align(s, toks)
    hum = np.asarray([exp[t]["human_future_poses"] for t in toks], dtype=np.float64)
    pc, pprov = plan_classes(plan)
    gc, gprov = plan_classes(hum)
    cm = np.zeros((3, 3), dtype=np.int64)                                   # COMMAND space (LEFT, STRAIGHT, RIGHT), rows = human-future class, cols = plan class
    np.add.at(cm, (gc, pc), 1)
    # the banked block is in the LABELLER's class order (lane_keep, turn_left, turn_right). `spec` is an INDEPENDENT literal copy of the definition's mapping
    # (lane_keep -> STRAIGHT, turn_left -> LEFT, turn_right -> RIGHT), deliberately NOT derived from LAT_TO_CMD: a mutated LAT_TO_CMD must make this control RED.
    spec = (1, 0, 2)
    banked = np.asarray(fam["lateral_decision"]["confusion_gt_rows_pred_cols"], dtype=np.int64)
    banked_cmd = np.zeros((3, 3), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            banked_cmd[spec[i], spec[j]] = banked[i, j]
    prov = fam["kinematics_provenance"]
    ok = (cm.tolist() == banked_cmd.tolist() and pprov["n_heading_held_from_last_moving_step"] == prov["pred"]["n_heading_held_from_last_moving_step"]
          and pprov["n_never_moved"] == prov["pred"]["n_never_moved"] and gprov["n_heading_held_from_last_moving_step"] == prov["gt"]["n_heading_held_from_last_moving_step"]
          and gprov["n_never_moved"] == prov["gt"]["n_never_moved"])
    return {"arm": arm_name, "ok": bool(ok), "ours_confusion_command_space": cm.tolist(), "banked_confusion_command_space": banked_cmd.tolist(),
            "banked_confusion_labeller_order": banked.tolist(), "families_json": str(b["families"]),
            "pred_provenance_ours": pprov, "pred_provenance_banked": prov["pred"], "gt_provenance_ours": gprov, "gt_provenance_banked": prov["gt"]}


def control(out=None, n_boot: int = N_BOOT, verify: bool = True) -> dict:
    """P2 (C2) read against u400 (C1) on their BANKED navtest seams, with P2's banked 64-proposal table as the floor control. No P6 file is touched."""
    import stage3_full as S3
    exp = load_export()
    p2, u4 = S3.pick_baseline("p2"), S3.pick_baseline("u400")
    assert p2["pin_ok"] and u4["pin_ok"], (p2["pin_ok"], u4["pin_ok"])
    res = run_read({"name": "p2", "seam": p2["seam"]}, [{"role": "C1", "name": "u400", "seam": u4["seam"]}], dump=load_dump(P2_TABLE), exp=exp, n_boot=n_boot, verify=verify)
    res["families_lateral_control"] = {a: families_lateral_control(a, b["seam"], exp) for a, b in (("p2", p2), ("u400", u4))}
    for a_, v in res["families_lateral_control"].items():
        if not v["ok"]:
            res.setdefault("INVALID", []).append(f"banked families lateral confusion of {a_} not reproduced")
    res["control_inputs"] = {"p2_seam": str(p2["seam"]), "p2_seam_md5": p2["seam_md5"], "u400_seam": str(u4["seam"]), "u400_seam_md5": u4["seam_md5"], "p2_table": P2_TABLE, "export": EXPORT}
    if out:
        Path(out).write_text(json.dumps(res, indent=1, default=_j), encoding="utf-8")
    return res


def _j(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _resolve_comp(spec: str, role: str) -> dict:
    """a BASELINES key of stage3_full ('u400', 'p2'; md5-verified seam) or a path to a seam npz"""
    if Path(spec).suffix == ".npz":
        return {"role": role, "name": Path(spec).stem, "seam": spec}
    import stage3_full as S3
    b = S3.pick_baseline(spec)
    if not b["pin_ok"]:
        raise SystemExit(f"comparator {spec!r}: the pinned files are absent or their md5 differs (pin_ok False); refusing")
    return {"role": role, "name": spec, "seam": b["seam"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("read")
    r.add_argument("--name", required=True)
    r.add_argument("--seam", default=None, help="default: <REFE_NAVTEST_DIR>/seams/stage2_<name>.npz")
    r.add_argument("--dump", default=None, help="the --dump-proposals npz (the random-proposal floor)")
    r.add_argument("--base", default="u400", help="C1: a stage3_full BASELINES key or a seam npz")
    r.add_argument("--also", default="p2", help="C2: a stage3_full BASELINES key or a seam npz")
    r.add_argument("--out", default=None)
    r.add_argument("--subset-ok", action="store_true", help="skip the full-navtest class-count gate (NOT the registered read)")
    r.add_argument("--no-verify-boot", action="store_true")
    r.add_argument("--n-boot", type=int, default=N_BOOT)
    c = sp.add_parser("control")
    c.add_argument("--out", default=None)
    c.add_argument("--no-verify-boot", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "control":
        res = control(a.out, verify=not a.no_verify_boot)
    else:
        seam = a.seam or str(DATA / "seams" / f"stage2_{a.name}.npz")
        comps = [_resolve_comp(a.base, "C1")] + ([_resolve_comp(a.also, "C2")] if a.also else [])
        dump_path = a.dump or (str(DATA / "seams" / f"stage2_{a.name}.proposals.npz") if (DATA / "seams" / f"stage2_{a.name}.proposals.npz").exists() else None)
        res = run_read({"name": a.name, "seam": seam}, comps, dump=load_dump(dump_path) if dump_path else None, expect_full=not a.subset_ok,
                       n_boot=a.n_boot, verify=not a.no_verify_boot)
        out = a.out or str(PKG / "raw" / "p2_readout" / a.name / f"command_adherence_{a.name}.json")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1, default=_j), encoding="utf-8")
    inv = res.get("INVALID")
    inc = res.get("INCOMPLETE")
    if "arms" in res:
        h = res["arms"][res["arm"]]["executed_plan_A7"]["turn_active"]
        print("ZZADH_READ", res["arm"], f"turn-active acc {h['accuracy']['point']:.4f} [{h['accuracy']['ci95'][0]:.4f}, {h['accuracy']['ci95'][1]:.4f}] n={h['n']}",
              f"kappa {h['kappa']['point']:.4f}", "INVALID" if inv else "ok")
    if "arms" in res and isinstance(res.get("random_proposal_floor"), dict) and "turn_active" in res["random_proposal_floor"]:
        fl = res["random_proposal_floor"]["turn_active"]
        print("ZZADH_FLOOR", f"turn-active floor {fl['floor']['accuracy']['point']:.4f} [{fl['floor']['accuracy']['ci95'][0]:.4f}, {fl['floor']['accuracy']['ci95'][1]:.4f}]",
              "floor>=actual" if fl["floor_ge_actual_point"] else "actual>floor")
    if inv:
        print("ZZADH_INVALID", json.dumps(inv))
    if inc and not inv:
        print("ZZADH_INCOMPLETE", json.dumps(inc))
    return 1 if inv else (2 if inc else 0)


if __name__ == "__main__":
    sys.exit(main())
