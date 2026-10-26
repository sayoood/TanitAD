"""hsgate_common.py -- PREREG_HSGATE (2331e1f3...): the pieces BOTH phases use, so the FIT side and the SCORE side build the gate's inputs with ONE code path.
NEW file (EvalFlyWheel / refcv9-hsgate agent, 2026-10-26).  CPU only.  Nothing registered is edited; the battery / ship_r8_sub code is IMPORTED.

  * seal checks (the prereg; the registered H2cn stage; the registered per-frame RETIME table)
  * the battery lane exactly as ship_r8_sub_forward.forward builds it (pf_lane.configure -> D8.RS + the sha-verified deploy table)
  * P_F (Amendment 5): compose(RT, P_os8, P_ss8) = eval_polyline(polyline_geometry(P_os8), arc(P_ss8)), then RS.retime_plan(.., prior8, T8, table, cast=True)
  * P_H (Amendment 4): h2cn_progress(AL, prior8, f2, f6) -> eval_polyline(polyline_geometry(plan8), s8) cast to the plan's float32
    (both functions are IMPORTED from ship_r8_sub/ship_r8_sub_forward.py -- the code that built the landed packages)
  * prior8 exactly as ship_r8_sub_forward's surface_inputs_hook builds it (kinematic_prior 'ha0_ext_pose' on the decision window)
  * the 15 gate features (PREREG_HSGATE s1) and the gate decision g(x) = 1[sigmoid(w.x + b) >= 0.5]
  * the L2 logistic regression (float64, torch LBFGS strong-Wolfe, zero init, torch seed 0)
  * ADE_S6 = refav1_arm._components 'ade_m' (the battery's ADE) on the S6 positions
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent                                                    # C:/Users/Admin/refcv8_pf/refcv9_hsgate
SCRATCH = Path("F:/refcv9_hsgate_scratch")
PREREG = WORK / "PREREG_HSGATE.md"
PREREG_SEAL = WORK / "PREREG_HSGATE.sha256"
PREREG_SHA_PREFIX = "2331e1f3"
SUB = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub")                    # READ-ONLY (imported, never written)
REG_H2CN = SUB / "REGISTERED_H2CN_STAGE.sha256"
H2CN_STAGE = SUB / "H2CN_STAGE_FROZEN.pt"
H2CN_STAGE_SHA = "f293b7bdc06aaeec19a10fd8cac9af993c4116401034a9d4ff9c28649e256f32"   # the brief's literal (README pinned-inputs table)
REG_PF = Path("C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256")
DEPLOY_TABLE_SHA = "3319b003fecc83577430f07dad762bce0a9a0b8594e9748e9a61bc4d264136b7"  # README pinned-inputs table (registered per-frame RETIME deploy table)

FRESH_TRACE = Path("F:/refcv9_ssfresh/bank/trace_selfsharp_fresh_s0.npz")
FRESH_LABELS = Path("F:/refcv9_ssfresh/bank/labels_selfsharp_fresh_s0.npz")
FRESH_KLABSRC = Path("F:/refcv9_ssfresh/bank/KLABSRC.json")
FRESH_CHAIN_LOG = Path("F:/refcv9_ssfresh/logs/chain.log")
FRESH_BANK = Path("F:/l1c/bank/ztac_fresh.npz")
FRESH_EXTRAS = Path("F:/l1c/tree/step48852ema_fresh/dump_s0/refcv6_extras.npz")
FRESH_N_WINDOWS = 10456
EVAL_SS_TRACE = "C:/Users/Admin/refcv8_pf/refcv9_l2/selfsharp/bank/trace_selfsharp_s{seed}.npz"     # == F:/refcv9_selfsharp/trace_selfsharp_s{seed}.npz (sha256-equal)
EVAL_L1_BANK = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/bank")
EVAL_EXTRAS = "C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/dump_s{seed}/refcv6_extras.npz"
LANDED_H8 = SUB / "out" / "arms" / "h2cn" / "H2CN_DEPLOY8" / "DEPLOY8_POS_step48852ema_pf_s{seed}.npz"
LANDED_FX = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub_fx/out/arms/factored_scored/FACTORED_DEPLOY8/DEPLOY8_POS_step48852ema_pf_s{seed}.npz")

LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)                         # PREREG_HSGATE s1
OOF_FOLD_SEED, CV_FOLD_SEED, LR_SEED = 0, 1, 0
LON6_V7_IDS = (0, 1, 3, 4, 5, 7)
LON6 = ("FOLLOW", "CRUISE", "BRAKE_TO", "CREEP", "HOLD", "ACCELERATE")
LOGIT_KEY = "nav_true.tacv6_lon_logits"
V0_KEY = "v0"
S6 = [1, 3, 4, 5, 6, 7]
FEATURES = ([f"p_{c}" for c in LON6] + [f"onehot_argmax_{c}" for c in LON6] + ["v0_over_10", "d_rel_arc_S6_last", "d_times_v0_over_10"])
FEATURE_SPEC = {
    "n": 15, "names": FEATURES,
    "p": "softmax over l2_common.lon6(extras['nav_true.tacv6_lon_logits'][row]) in float64, max-subtracted (h2cnl_freeze.lon6_softmax)",
    "onehot": "one-hot of argmax over the same 6 LON6 logits (numpy argmax: first maximum on ties)",
    "v0": "extras['v0'][row] (float32 widened exactly to float64) / 10",
    "d": "(arcF - arcH) / max(arcH, 1.0), arc* = along_track_localise.arc(P[:, S6 slots])[:, -1] (cumulative polyline length from the origin through the 6 "
         "staged S6 positions, float64), P_F = the FACTORED_DEPLOY8 construction, P_H = the H2CN_DEPLOY8 construction (OOF H2cn on FIT rows)",
    "d_times_v0": "d * v0 / 10",
    "row": "the extras row of the SAME native forward, joined on (clip index, ws)",
    "standardise": "x_std = (x - mu) / sd, mu = mean over FIT rows, sd = std over FIT rows + 1e-9 (l1_fit_score._std)",
}
DECISION = "g = 1[sigmoid(w . x_std + b) >= 0.5]  (computed as 1[1/(1+exp(-(w . x_std + b))) >= 0.5] in float64)"
OBJECTIVE = ("mean_i BCEwithLogits(w . x_std_i + b, y_i) + 0.5 * lambda * ||w||^2 (b unpenalised); float64; torch.optim.LBFGS(lr=1, max_iter=5000, "
             "history_size=50, tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn='strong_wolfe'), w = 0, b = 0 init, torch.manual_seed(0)")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def first_token(p: Path) -> str:
    try:
        return Path(p).read_text(encoding="utf-8").split()[0].lstrip("*").strip()
    except Exception:          # noqa: BLE001
        return ""


def check_prereg() -> dict:
    """REFUSES unless sha256(PREREG_HSGATE.md) == the first token of PREREG_HSGATE.sha256 (both 64 hex, prefix 2331e1f3)."""
    got = sha256_file(PREREG) if PREREG.is_file() else ""
    want = first_token(PREREG_SEAL)
    if len(got) != 64 or len(want) != 64:
        raise SystemExit(f"[hsgate] INCONCLUSIVE prereg seal: file '{got}' seal '{want}' -- refusing")
    if got != want or not got.startswith(PREREG_SHA_PREFIX):
        raise SystemExit(f"[hsgate] PREREG_HSGATE.md hashes to {got}, the seal says {want} -- refusing")
    return {"file": str(PREREG), "sha256": got, "seal_matches": True}


def check_h2cn_stage() -> dict:
    reg, got = first_token(REG_H2CN), sha256_file(H2CN_STAGE)
    if not (len(reg) == len(got) == 64 and reg == got == H2CN_STAGE_SHA):
        raise SystemExit(f"[hsgate] the registered H2cn stage does not verify (registered '{reg}', file '{got}', literal '{H2CN_STAGE_SHA}')")
    return {"file": str(H2CN_STAGE), "sha256": got, "registered_file": str(REG_H2CN), "matches_registered_and_literal": True}


def env_check() -> dict:
    e = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE", "TMP", "TEMP")}
    if e["CUDA_VISIBLE_DEVICES"] != "-1":
        raise SystemExit("[hsgate] CPU only: CUDA_VISIBLE_DEVICES must be -1")
    if e["PYTHONDONTWRITEBYTECODE"] != "1":
        raise SystemExit("[hsgate] PYTHONDONTWRITEBYTECODE=1 is required (the imported ship_r8_sub modules must not get a __pycache__ written beside them)")
    return e


def lane() -> dict:
    """The battery lane EXACTLY as ship_r8_sub_forward.forward sets it up (pf_lane: read_registered, verify_installed, mods, configure; D8.RS deploy table)."""
    sys.path.insert(0, str(SUB))
    import ship_r8_sub_common as C
    C.setup_paths()
    sys.path.insert(0, str(C.PF_HOLD / "code"))
    import pf_lane as PL
    import ship_r8_sub_forward as SF                                   # h2cn_progress / compose / T8 / HZ8 (import-time: two sha256 reads, no writes)
    reg_pf = PL.read_registered(REG_PF)
    PFPKG = C.PF_HOLD / "pfpkg"
    PL.verify_installed(PFPKG, reg_pf)
    M = PL.mods()
    PL.configure(M, PFPKG, reg_pf)
    RT, D8, TFM = M["RT"], M["D8"], M["TFM"]
    RS = D8.RS
    table = RS.load_deploy_table()                                    # sha-verified (sidecar)
    import l1_fit_score as F
    if F.AL.RT is not RT:
        raise SystemExit("[hsgate] two copies of a0k0_headroom_retime_r8 are loaded")
    tsha = str(table.get("sha256", ""))
    tfile = sha256_file(RS.DEPLOY_TABLE_PATH)
    if not (len(tsha) == 64 and tsha == tfile == DEPLOY_TABLE_SHA):
        raise SystemExit(f"[hsgate] the deploy table does not verify: table '{tsha}' file '{tfile}' literal '{DEPLOY_TABLE_SHA}'")
    return {"C": C, "PL": PL, "SF": SF, "M": M, "RT": RT, "RS": RS, "D8": D8, "TFM": TFM, "F": F, "AL": F.AL, "PS": F.PS, "table": table,
            "rec": {"registered_pf_table_sha256": reg_pf, "deploy_table_file": str(RS.DEPLOY_TABLE_PATH), "deploy_table_sha256": tfile,
                    "retime_stage_file": str(D8.RS_FILE), "retime_stage_sha256": sha256_file(D8.RS_FILE),
                    "ship_r8_sub_forward_sha256": SF.SCRIPT_SHA, "ship_r8_sub_common_sha256": SF.COMMON_SHA}}


def prior8_of(RT, SF, win) -> np.ndarray:
    """ship_r8_sub_forward.surface_inputs_hook's prior8, verbatim: kinematic_prior 'ha0_ext_pose' on the decision window, 8 model slots, float32."""
    import torch
    from tanitad.models import kinematic_prior as KP
    from tanitad.models.ego_history import ego_channels_from_poses
    pw = torch.from_numpy(np.ascontiguousarray(win, dtype=np.float32))
    a0, kp = KP.prior_controls("ha0_ext_pose", pw)
    v0 = ego_channels_from_poses(pw, RT.W_OBS, dt=RT.DT)[:, -1, 0]
    return KP.prior_path(a0, kp, v0, SF.HZ8).numpy().astype(np.float32)


def pf_positions(L, P_os8, P_ss8, prior8) -> tuple:
    """Amendment 5: P_fx = compose(RT, P_os8, P_ss8) (float32), staged = RS.retime_plan(P_fx, prior8, T8, table, cast=True / False).  Returns (P_fx, pos32, pos64)."""
    P_os8 = np.ascontiguousarray(P_os8).astype(np.float32)
    P_ss8 = np.ascontiguousarray(P_ss8).astype(np.float32)
    P_fx = L["SF"].compose(L["RT"], P_os8, P_ss8)
    T8 = L["SF"].T8
    return P_fx, L["RS"].retime_plan(P_fx, prior8, T8, L["table"], cast=True), L["RS"].retime_plan(P_fx, prior8, T8, L["table"], cast=False)


def ph_positions(L, plan8, prior8, f2, f6) -> tuple:
    """Amendment 4: s8 = h2cn_progress(AL, prior8, f2, f6) (max(s,0) + running max); positions = eval_polyline(polyline_geometry(plan8), s8); (pos32, pos64, s8, changed)."""
    s8, changed = L["SF"].h2cn_progress(L["AL"], prior8, f2, f6)
    p64 = L["RT"].eval_polyline(L["RT"].polyline_geometry(plan8), s8)
    return p64.astype(np.asarray(plan8).dtype), p64, s8, changed


def load_l2_common():
    sys.path.insert(0, str(SUB))
    import h2cnl_freeze as HL
    return HL.load_l2_common(), HL


def lon6_softmax(L2, HL, logits8):
    return HL.lon6_softmax(L2, logits8)


def features(L2, HL, AL, logits8, v0, PF6, PH6) -> tuple:
    """The 15 PREREG_HSGATE features (unstandardised), float64 [n, 15] + the parts (arcF, arcH, d)."""
    p = lon6_softmax(L2, HL, logits8)
    lg6 = L2.lon6(logits8)
    oh = np.eye(6)[np.argmax(lg6, axis=1)]
    v = np.asarray(v0, np.float32).astype(np.float64) / 10.0
    aF = AL.arc(np.asarray(PF6, np.float64))[:, -1]
    aH = AL.arc(np.asarray(PH6, np.float64))[:, -1]
    d = (aF - aH) / np.maximum(aH, 1.0)
    X = np.concatenate([p, oh, v[:, None], d[:, None], (d * v)[:, None]], axis=1)
    if X.shape[1] != 15 or not np.isfinite(X).all():
        raise SystemExit(f"[hsgate] features: shape {X.shape}, finite {bool(np.isfinite(X).all())}")
    return X, {"arcF": aF, "arcH": aH, "d": d}


def standardise_fit(X):
    mu, sd = X.mean(0), X.std(0) + 1e-9                                  # l1_fit_score._std, verbatim
    return mu, sd


def lr_fit(Xs, y, lam) -> dict:
    """L2 logistic regression on standardised X (OBJECTIVE).  Deterministic: zero init, full batch, torch seed 0."""
    import torch
    torch.manual_seed(LR_SEED)
    x = torch.from_numpy(np.ascontiguousarray(Xs, np.float64))
    t = torch.from_numpy(np.asarray(y, np.float64))
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=5000, history_size=50, tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
    lam_t = float(lam)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, t) + 0.5 * lam_t * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    loss = closure()
    gnorm = float(torch.cat([w.grad.abs(), b.grad.abs()]).max())
    st = opt.state[opt._params[0]]
    return {"w": w.detach().numpy().copy(), "b": float(b.detach().numpy()[0]), "lambda": lam_t, "loss": float(loss.detach()), "max_abs_grad": gnorm,
            "n_iter": int(st.get("n_iter", -1)), "func_evals": int(st.get("func_evals", -1))}


def gate_logit(gate: dict, X) -> np.ndarray:
    Xs = (np.asarray(X, np.float64) - np.asarray(gate["mu"], np.float64)) / np.asarray(gate["sd"], np.float64)
    return Xs @ np.asarray(gate["w"], np.float64) + float(gate["b"])


def gate_decide(gate: dict, X) -> np.ndarray:
    z = gate_logit(gate, X)
    return (1.0 / (1.0 + np.exp(-z))) >= 0.5


def ade6(ra, P6, G6, dt) -> np.ndarray:
    """The battery's ADE (refav1_arm._components 'ade_m': float32 torch norm, mean over the instants) on S6 positions."""
    c = ra._components(np.asarray(P6, np.float32), np.asarray(G6, np.float32), float(dt))
    return np.asarray(c["ade_m"], np.float64)


def jdump(obj, p: Path) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


def canonical_bytes(obj) -> bytes:
    """Deterministic serialisation for the frozen gate (sorted keys, repr-exact floats, no timestamps)."""
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=True).encode("utf-8")


def chain_status(path: Path = FRESH_CHAIN_LOG) -> dict:
    """The fresh SELFSHARP chain's own markers.  ZZSFENDZZ ... rc=<n> is written by its EXIT trap (always)."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:      # noqa: BLE001
        return {"readable": False, "error": type(e).__name__}
    tags = [ln.split()[0] for ln in lines if ln.startswith("ZZSF")]
    end = [ln for ln in lines if ln.startswith("ZZSFENDZZ")]
    bad = [ln for ln in lines if ln.split()[0] in ("ZZSFABORTZZ", "ZZSFSTOPZZ", "ZZSFNOLOCKZZ")]
    rc = None
    if end:
        for tok in end[-1].split():
            if tok.startswith("rc="):
                rc = tok[3:]
    return {"readable": True, "n_lines": len(lines), "tags": tags, "ended": bool(end), "end_rc": rc, "failure_markers": bad, "last": lines[-1] if lines else ""}
