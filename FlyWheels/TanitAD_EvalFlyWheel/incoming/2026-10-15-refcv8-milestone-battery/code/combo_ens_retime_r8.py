"""combo_ens_retime_r8.py -- do the K-draw ensemble (ENS-K) and the arc-length re-timing (RETIME) STACK on refcv8's E1 error?  (PREREG_COMBO.md)
CPU only, banked dumps + FROZEN gains only.  NEW file; nothing existing is edited (it IMPORTS kdraw_e1_r8 and a0k0_headroom_retime_r8).

ONE COMMAND (memory-gated wrapper):   sh <PKG>/code/run_combo_ens_retime_r8.sh [tree] [eval|train]
  == CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' \
       python -X utf8 combo_ens_retime_r8.py run --tree step30000ema_unc --gains eval --out-dir <PKG>/raw/combo_ens_retime

Arms (S2 grid, ceiling-compliant mask, plans [N,4,2]):  ENS-K = mean of K draws; RET-alone-K = per-window mean over the K draws of retime(os_s); COMBO-K = retime(ENS-K, HP, g)
with the FROZEN gains g applied UNCHANGED (never refitted here); beside: COMBO-G-K (gated table) and RT-THEN-ENS-K (retime each draw, then average: no transfer).
K=4 on the DISJOINT sets A={0,1,2,3}, B={4,5,6,7} decides the label; K=2 (A2={0,1}, B2={4,5}) and K=8 are descriptive.  Verdict rules = PREREG_COMBO.md sections 4-5, coded in
`decide`.  The registered read is the FINAL checkpoint with TRAIN-fitted gains (`--tree step48852ema_unc --gains train`); the 30k read with eval-fitted gains is EXPLORATORY.
TIER: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).  ESTIMATOR: paired episode-cluster bootstrap, models FIXED, B 2000, seed 0 -- answers "another draw
of EPISODES" only; the disjoint draw sets A/B are the inference-run replicate; one training seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PKG = HERE.parent
OUT = PKG / "raw" / "combo_ens_retime"
PREREG = OUT / "PREREG_COMBO.md"
#: pinned AFTER PREREG_COMBO.md was written and BEFORE any combined number existed (PREREG_COMBO.sha256 holds the same value)
PREREG_SHA256 = "a888e7cc99c3346dcbe71244f11c1d6f6ce654bccec9aeb7ec764b7050701287"

import kdraw_e1_r8 as KD  # noqa: E402  (pure helpers: ensemble_mean, load_dump, arm_verdict, ...)
import a0k0_headroom_retime_r8 as RT  # noqa: E402  (retime_apply, polyline_geometry, frozen_weights, acc_lat, ...)

N_BOOT, BOOT_SEED = 2000, 0
SETS = {"4A": (0, 1, 2, 3), "4B": (4, 5, 6, 7), "2A": (0, 1), "2B": (4, 5), "8": tuple(range(8))}
K4 = ("4A", "4B")
LATERAL_KEYS = tuple(RT.LATERAL_KEYS)
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
QUESTION = ("paired episode-cluster bootstrap, models FIXED, B 2000, seed 0: answers 'another draw of EPISODES' only; the disjoint draw sets A/B are the inference-run replicate; "
            "NOT another training run (single training seed)")
GAINS_EVAL_SHA256 = "4ecde275554b1aaae84a4328294f791f28426f451832f9933fa34859eedcda4c"
TRAIN_GAINS = PKG / "raw" / "retime_trainfit" / "FROZEN_GAINS_RETIME_train.json"
TREE30 = "step30000ema_unc"
ON_PATH_TOL_M = 1e-6
G0_TOL_M = 1e-6
#: literals (MEASURED; written out, never derived from the code under test).  Sources: raw/kdraw/KDRAW_E1.json (single draws, ENS, pose cells), raw/a0k0_retime/FORWARD_RETIME_step30000ema_unc.json
#: (FROZEN single-draw RETIME), the 2026-09-26 model-free measurement (R0926)
LIT_K1_ECHO = {0: (-0.0085, -0.0229, 0.0065), 1: (-0.0076, -0.0216, 0.0073)}
LIT_K1_POSE = {0: (-0.0136, -0.0286, 0.0018), 1: (-0.0127, -0.0272, 0.0024)}
LIT_K1R_POSE = {0: (-0.0320, -0.0427, -0.0211), 1: (-0.0315, -0.0420, -0.0207)}
LIT_K1R_ECHO = {0: (-0.0269, -0.0373, -0.0161), 1: (-0.0264, -0.0367, -0.0158)}
LIT_ENS_POSE = {"2A": (-0.0189, -0.0335, -0.0035), "4A": (-0.0225, -0.0373, -0.0071), "4B": (-0.0210, -0.0363, -0.0056)}
LIT_R0926 = (0.0055, 0.0036, 0.0076)
LIT_POSE_ECHO_MEAN_ADE = 0.2858


def mutation() -> str:
    """Deliberate-regression hook, read at CALL time so tests can set it.  '' in production.  Recorded in the output."""
    return os.environ.get("COMBO_TEST_MUTATION", "")


# ============================================================================================================== pure functions (unit-tested)
def retime(P, HP, W) -> np.ndarray:
    """The re-timing of plan(s) P [n,S,2] with prior HP and gains W (scalar / [S] / [n,S]).  Mutation 'skip_retime' passes the plan through unchanged."""
    if mutation() == "skip_retime":
        return np.array(P, copy=True)
    return RT.retime_apply(P, HP, W)


def walker_positions(O, HP) -> np.ndarray:
    """INDEPENDENT, loop-based (shares no code with RT.eval_polyline / prior_arc / retime_*): the point of the polyline (0,0) -> O_1 -> ... -> O_S at the arc length
    that the PRIOR's chord-sum reaches at each instant (the position gain 0 must produce); past the last vertex along the last positive-length segment (+x if degenerate).
    Returns float64 [n, S, 2]."""
    O, HP = np.asarray(O, np.float64), np.asarray(HP, np.float64)
    n, S, _ = O.shape
    out = np.zeros((n, S, 2))
    for i in range(n):
        V = [(0.0, 0.0)] + [(float(O[i, k, 0]), float(O[i, k, 1])) for k in range(S)]
        L = [0.0]
        for k in range(S):
            L.append(L[-1] + math.hypot(V[k + 1][0] - V[k][0], V[k + 1][1] - V[k][1]))
        u_end = (1.0, 0.0)
        for k in range(S - 1, -1, -1):
            d = L[k + 1] - L[k]
            if d > 0:
                u_end = ((V[k + 1][0] - V[k][0]) / d, (V[k + 1][1] - V[k][1]) / d)
                break
        W = [(0.0, 0.0)] + [(float(HP[i, k, 0]), float(HP[i, k, 1])) for k in range(S)]
        m = 0.0
        for k in range(S):
            m += math.hypot(W[k + 1][0] - W[k][0], W[k + 1][1] - W[k][1])
            if m >= L[S]:
                out[i, k] = (V[S][0] + (m - L[S]) * u_end[0], V[S][1] + (m - L[S]) * u_end[1])
            else:
                j = max(jj for jj in range(S) if L[jj] <= m)
                d = L[j + 1] - L[j]
                ux, uy = (((V[j + 1][0] - V[j][0]) / d, (V[j + 1][1] - V[j][1]) / d) if d > 0 else (0.0, 0.0))
                out[i, k] = (V[j][0] + (m - L[j]) * ux, V[j][1] + (m - L[j]) * uy)
    return out


def check_g1(P, HP, g=1.0) -> dict:
    """C-G1: the gain vector 1 must reproduce the plan BIT FOR BIT (array level).  g is a parameter so the test can feed a wrong gain and see RED."""
    S = np.asarray(P).shape[1]
    out = RT.retime_apply(P, HP, np.full(S, float(g)))
    same = bool(np.array_equal(out, np.asarray(P)))
    return {"bit_equal": same, "max_abs_m": float(np.abs(out.astype(np.float64) - np.asarray(P, np.float64)).max()), "PASS": same}


def check_g0(P, HP) -> dict:
    """C-G0: gain vector 0 vs the independent walker (float64 positions, no cast)."""
    S = np.asarray(P).shape[1]
    a = RT.retime_apply(P, HP, np.zeros(S), cast=False)
    b = walker_positions(P, HP)
    err = float(np.abs(a - b).max())
    return {"max_abs_m": err, "tol_m": G0_TOL_M, "PASS": bool(err <= G0_TOL_M)}


def rt_then_ens(os_list, HP, W) -> np.ndarray:
    """retime every draw with the gains, THEN average (float64 mean, float32): the order swap that involves no transfer."""
    return KD.ensemble_mean(np.stack([retime(P, HP, W) for P in os_list]))


def mean_over_draws(arrs) -> np.ndarray:
    return np.mean(np.stack([np.asarray(a, np.float64) for a in arrs]), axis=0)


def stacking_core(os_by_seed: dict, seeds, HP, W, ade_fn, cell_fn) -> dict:
    """THE glue that analyse() and the tests share: ENS-K of `seeds`, COMBO-K = retime(ENS-K), RET-alone = per-window mean over the draws of ADE(retime(os_s)), and the two paired
    stacking cells ST_ens = COMBO - ENS (does RETIME add on top of the ensemble?) and ST_ret = COMBO - RET-alone (does the ensemble add on top of RETIME?).
    ade_fn(plan) -> per-window ADE [n]; cell_fn(new, ref) -> paired cell.  Returns plans, ADE vectors, cells."""
    ens = KD.ensemble_mean(np.stack([os_by_seed[s] for s in seeds]))
    combo = retime(ens, HP, W)
    ade_ens, ade_combo = ade_fn(ens), ade_fn(combo)
    ade_ret = mean_over_draws([ade_fn(retime(os_by_seed[s], HP, W)) for s in seeds])
    return {"ens": ens, "combo": combo, "ade_ens": np.asarray(ade_ens), "ade_combo": np.asarray(ade_combo), "ade_ret": ade_ret,
            "ST_ens": cell_fn(ade_combo, ade_ens), "ST_ret": cell_fn(ade_combo, ade_ret)}


def additivity(single: float, ens: float, ret: float, combo: float) -> dict:
    """How additive are the two levers?  All inputs are mean ADE (m) on the same windows/draw set.  dENS = ens - single, dRET = ret - single (negative = better); additive prediction of the
    combo = single + dENS + dRET.  share_of_additive = (combo - single) / (dENS + dRET);  ens_on_top_of_ret = combo - ret as a share of dENS;  ret_on_top_of_ens = combo - ens as a share of dRET."""
    dE, dR = ens - single, ret - single
    return {"single": single, "dENS": dE, "dRET": dR, "additive_combo": single + dE + dR, "combo": combo, "combo_gain": combo - single,
            "share_of_additive": (combo - single) / (dE + dR) if (dE + dR) != 0 else None,
            "ens_on_top_of_ret_m": combo - ret, "ens_on_top_of_ret_share_of_dENS": (combo - ret) / dE if dE != 0 else None,
            "ret_on_top_of_ens_m": combo - ens, "ret_on_top_of_ens_share_of_dRET": (combo - ens) / dR if dR != 0 else None}


def floor_of(mean_a: float, mean_b: float) -> float:
    return abs(float(mean_a) - float(mean_b))


def sep_better(c: dict | None) -> bool:
    return bool(c and "delta" in c and c["delta"] < 0 and c["hi"] < 0)


def sep_worse(c: dict | None) -> bool:
    return bool(c and "delta" in c and c["lo"] > 0)


def stack_verdict(cell_a: dict | None, cell_b: dict | None, floor_combo: float, floor_cmp: float) -> dict:
    """PREREG 4.2: a stacking cell STACKS iff on BOTH sets it is separated better AND |delta| > 2 x max(floor_combo, floor_cmp).  `worse` = separated worse on >= 1 set."""
    if not (cell_a and cell_b and "delta" in cell_a and "delta" in cell_b):
        return {"stacks": False, "worse_on_any_set": False, "status": "PENDING"}
    cs = (cell_a, cell_b)
    f2 = 2.0 * max(floor_combo, floor_cmp)
    sep = [sep_better(c) for c in cs]
    beyond = [bool(abs(c["delta"]) > f2) for c in cs]
    return {"stacks": bool(all(sep) and all(beyond)), "worse_on_any_set": bool(any(sep_worse(c) for c in cs)), "separated_better_by_set": sep, "beyond_2x_floor_by_set": beyond,
            "two_x_floor_m": f2}


def nr_family_fail(rollup: dict | None) -> set:
    return {f for f, v in (rollup or {}).items() if str(v).startswith("FAIL")}


def decide(ev: dict) -> dict:
    """PREREG section 5, K=4, sets A and B.  Precedence HARMS > STACKS > NO-STACK.  `ev` keys:
        st_ens, st_ret : stack_verdict dicts           e1p : {"A": cell, "B": cell, "floor_combo": f}
        acc : {"S2": {"HOLDS": True/False/None, "moved": bool}, "S6K2": {...} | None}
        nr, nr_ens : {"A": {family: "PASS"|"FAIL: ..."}, "B": {...}} | None  (None = not evaluated -> a descriptive label, never STACKS)"""
    reasons_h, reasons_ns = [], []
    if ev["st_ens"].get("worse_on_any_set"):
        reasons_h.append("h1: COMBO separated WORSE than ENS-4 alone on >=1 set")
    if ev["st_ret"].get("worse_on_any_set"):
        reasons_h.append("h1: COMBO separated WORSE than RET-alone-4 on >=1 set")
    acc = ev.get("acc") or {}
    for key in ("S2", "S6K2"):
        a = acc.get(key)
        if a and a.get("HOLDS") is False:
            reasons_h.append(f"h2: ACC-LAT FAILS ({key})")
    nr, nr_ens = ev.get("nr"), ev.get("nr_ens")
    if nr is not None:
        for s in ("A", "B"):
            introduced = nr_family_fail(nr.get(s)) - nr_family_fail((nr_ens or {}).get(s))
            if introduced:
                reasons_h.append(f"h3: COMBO fails NR family {sorted(introduced)} on set {s} where ENS-4 alone passes")
    if reasons_h:
        return {"label": "HARMS", "reasons": reasons_h}
    e1 = KD.arm_verdict(ev["e1p"]["A"], ev["e1p"]["B"], ev["e1p"]["floor_combo"])["verdict"]
    if not (ev["st_ens"].get("stacks") and ev["st_ret"].get("stacks")):
        which = [n for n, v in (("ENS-4 alone", ev["st_ens"]), ("RET-alone-4", ev["st_ret"])) if not v.get("stacks")]
        reasons_ns.append("not separated better (beyond 2x floor, both sets) than: " + ", ".join(which))
    if e1 != "PASS":
        reasons_ns.append(f"E1-PASS (vs pose) not met: {e1}")
    s2, s6 = acc.get("S2"), acc.get("S6K2")
    if not s2 or s2.get("HOLDS") is not True or not s2.get("moved"):
        reasons_ns.append("ACC-LAT S2 K=4 not a non-vacuous HOLD")
    if not s6 or s6.get("HOLDS") is not True or not s6.get("moved"):
        reasons_ns.append("ACC-LAT S6 K=2 not a non-vacuous HOLD (or UNAVAILABLE)")
    if nr is None:
        reasons_ns.append("NR not evaluated")
    elif not all(v == "PASS" for s in ("A", "B") for v in (nr.get(s) or {}).values()):
        reasons_ns.append("NR not PASS in every family on both sets")
    if reasons_ns:
        return {"label": "NO-STACK", "reasons": reasons_ns}
    return {"label": "STACKS", "reasons": []}


# ============================================================================================================== load / guards
def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_prereg() -> str:
    if not PREREG.exists():
        raise SystemExit(f"[combo] REFUSING: {PREREG} is missing")
    got = sha256_file(PREREG)
    if got != PREREG_SHA256:
        raise SystemExit(f"[combo] REFUSING: PREREG_COMBO.md sha256 {got} != the value pinned in the script {PREREG_SHA256} (the pre-registration was edited after pinning)")
    return got


def load_gains(kind: str) -> tuple[dict, dict]:
    if kind == "eval":
        p = Path(RT.FROZEN_JSON)
        meta = {"kind": "eval", "label": "EVAL-FITTED GAINS (in-sample on the eval episodes; EXPLORATORY ONLY)", "file": str(p), "sha256": sha256_file(p)}
        if meta["sha256"] != GAINS_EVAL_SHA256:
            raise SystemExit(f"[combo] REFUSING: {p} sha256 {meta['sha256']} != the registered eval-gains file {GAINS_EVAL_SHA256}")
        return json.load(open(p, encoding="utf-8")), meta
    if kind == "train":
        p = TRAIN_GAINS
        side = p.with_suffix(".sha256")
        if not p.exists() or not side.exists():
            raise SystemExit(f"[combo] REFUSING: the train-fitted gains are not there yet ({p} / {side.name}); they are written by chain_retime_trainfit_r8.sh (ZZTFPOSTENDZZ rc=0)")
        fz = json.load(open(p, encoding="utf-8"))
        want, got = side.read_text().strip(), sha256_file(p)
        if want != got:
            raise SystemExit(f"[combo] REFUSING: {p.name} sha256 {got} != recorded {want}")
        if fz.get("fit_source") != "train" or fz.get("fit_guard", {}).get("n_overlap_with_eval139") != 0:
            raise SystemExit("[combo] REFUSING: the gains file is not a train-fitted file (fit_source != 'train' or overlap with eval139 != 0)")
        return fz, {"kind": "train", "label": "CLEAN GAINS (train-fitted)", "file": str(p), "sha256": got, "fit_guard": fz.get("fit_guard"), "fallback": fz.get("fallback")}
    raise SystemExit(f"[combo] unknown gains kind {kind!r}")


def fmt(c, dp=4):
    return KD.fc(c, dp)


# ============================================================================================================== the analysis
def analyse(tree: str, gains_kind: str, kd_root: str, out_dir: Path) -> dict:
    t0 = time.time()
    prereg_sha = check_prereg()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"COMBO_{tree}_{gains_kind}"
    if (out_dir / f"{stem}.json").exists() and not mutation():
        raise SystemExit(f"[combo] REFUSING to overwrite {out_dir / (stem + '.json')} (a result for this tree/gains already exists)")
    leftovers = [p.name for p in out_dir.iterdir() if p.is_file() and p.name not in ("PREREG_COMBO.md", "PREREG_COMBO.sha256") and p.suffix in (".json", ".md")
                 and not p.name.startswith(("HANDOFF", "RESULTS", "COMBO_"))]
    fz, gmeta = load_gains(gains_kind)
    tools = RT.tools_load()                       # FIRST: r8_repoint configures the loader environment (CP.get_readers) before anything imports the loader
    import compliant_reread_r8 as CR
    PR, ra = tools["PR"], tools["ra"]
    rc = tools["rc"]
    res = {"tool": "combo_ens_retime_r8.py", "generated": time.strftime("%FT%T%z"), "tier": TIER, "estimator": QUESTION, "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "tree": tree,
           "kd_root": kd_root, "gains": gmeta, "prereg": "raw/combo_ens_retime/PREREG_COMBO.md", "prereg_sha256": prereg_sha, "script_sha256": sha256_file(__file__),
           "test_mutation": mutation() or None, "unexpected_files_in_out_dir_at_start": leftovers,
           "status": ("EXPLORATORY (eval-fitted gains, post-hoc arm, milestone checkpoint); bar of record = single-draw `os`" if gains_kind == "eval" else
                      ("CLEAN-GAINS read of a milestone checkpoint (not the registered read)" if tree != "step48852ema_unc" else "REGISTERED READ (final checkpoint, train-fitted gains)"))}
    # ---- seeds / dumps ------------------------------------------------------------------------------------------------------------------------
    dumps, avail = {}, {}
    for s in range(8):
        ok, d = KD.seed_available(tree, s, kd_root)
        avail[s] = {"available": ok, "dump": d}
        if ok:
            dumps[s] = KD.load_dump(d)
    res["seeds"] = avail
    missing = [s for s in range(8) if s not in dumps]
    if missing:
        raise SystemExit(f"[combo] REFUSING: draws {missing} are not available under {kd_root} / the battery archive (KDRAW_VERIFY.json ok=true required)")
    ref = dumps[0]
    # ---- panel-side inputs (the retime script's own builders: the SAME prior, the SAME mask as FORWARD_RETIME) -------------------------------------------
    ctx = {s: CR.prepare_seed(tree, s, tools, rc) for s in (0, 1)}
    first = {}
    S2 = {s: RT.surface_inputs(tools, ctx[s], "S2", first, tree) for s in (0, 1)}
    try:
        S6 = {s: RT.surface_inputs(tools, ctx[s], "S6", first, tree) for s in (0, 1)}
    except (Exception, SystemExit) as exc:                              # noqa: BLE001 -- recorded, not hidden
        S6 = None
        res["s6_unavailable"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    sha_mask = ctx[0]["sha"]
    del ctx                                         # the raw extras are not needed again; free the commit before the bootstrap phase
    import gc
    gc.collect()
    HP, ECHO, G, dt = S2[0]["X"]["HP"], S2[0]["X"]["HX"], S2[0]["X"]["G"], S2[0]["dt"]
    v0, mask, eid = S2[0]["v0"], np.asarray(S2[0]["mask"], bool), np.asarray(ref["eid"])
    ctl = {}
    ctl["C-MASK"] = {"sha": sha_mask, "n_compliant": int(mask.sum()), "n_dump": int(len(mask)), "n_episodes": int(len(np.unique(eid[mask])))}
    ctl["C-MASK"]["PASS"] = bool(sha_mask == KD.MASK_SHA and ctl["C-MASK"]["n_compliant"] == 4159 and ctl["C-MASK"]["n_dump"] == 4754 and ctl["C-MASK"]["n_episodes"] == 138)
    pan0 = KD.load_panel_arms(f"{KD.ARCHIVE}/{tree}/panel_s0", ("b_refcv7_50400_s0", "b_refcv7_50400_s1"))
    pan1 = KD.load_panel_arms(f"{KD.ARCHIVE}/{tree}/panel_s1", ())
    ctl["C-ALIGN"] = {"S2_ws_eq_dump": bool(np.array_equal(S2[0]["ws"], ref["ws"])), "S2_clip_eq_dump": bool(np.array_equal(S2[0]["clip"], ref["clip_index"])),
                      "S2_eid_eq_dump": bool(np.array_equal(S2[0]["eid"], ref["eid"])), "G_eq_dump": bool(np.array_equal(G, ref["g"])), "echo_eq_dump": bool(np.array_equal(ECHO, ref["ha0_ext"])),
                      "pan0_ws_eq": bool(np.array_equal(pan0["ws"], ref["ws"])), "pan0_os_eq_dump0": bool(np.array_equal(pan0["os"], dumps[0]["os"])),
                      "pan1_os_eq_dump1": bool(np.array_equal(pan1["os"], dumps[1]["os"])), "RT_O_s0_eq_dump0": bool(np.array_equal(S2[0]["X"]["O"], dumps[0]["os"])),
                      "RT_O_s1_eq_dump1": bool(np.array_equal(S2[1]["X"]["O"], dumps[1]["os"])), "HP_shape_ok": bool(HP.shape == dumps[0]["os"].shape)}
    ctl["C-ALIGN"]["PASS"] = bool(all(ctl["C-ALIGN"].values()))
    bad_id = {s: [k for k in ("ws", "eid", "clip_index", "g", "ha0_ext", "v0") if not np.array_equal(d[k], ref[k])] for s, d in dumps.items()}
    bad_id = {s: b for s, b in bad_id.items() if b}
    ctl["C-ID"] = {"seeds_checked": sorted(dumps), "mismatches": bad_id, "PASS": not bad_id}
    pl2 = {f"{i}-{j}": float(np.linalg.norm(dumps[i]["os"].astype(np.float64) - dumps[j]["os"].astype(np.float64), axis=-1).mean()) for i in sorted(dumps) for j in sorted(dumps) if i < j}
    ctl["C-DIST"] = {"min_mean_pair_plan_l2_m": min(pl2.values()), "PASS": bool(min(pl2.values()) > 1e-4)}
    os_ = {s: dumps[s]["os"] for s in range(8)}

    def comps(P, Gt=None, dtt=None):
        return PR.components(ra, np.ascontiguousarray(P, dtype=np.float32), G if Gt is None else Gt, dt if dtt is None else dtt)

    def cell(a, b, m=None, e=None):
        m = mask if m is None else m
        e = eid if e is None else e
        return PR.paired_cell(np.asarray(a)[m], np.asarray(b)[m], np.asarray(e)[m], N_BOOT, BOOT_SEED)

    echo_c, pose_c = comps(ECHO), comps(HP)
    r09 = PR.paired_cell(pose_c["ade_m"], echo_c["ade_m"], eid, N_BOOT, BOOT_SEED)
    ctl["C-R0926"] = {"got": fmt(r09), "lit": LIT_R0926, "pose_echo_mean_ade_compliant": float(pose_c["ade_m"][mask].mean()), "lit_mean": LIT_POSE_ECHO_MEAN_ADE}
    ctl["C-R0926"]["PASS"] = bool(KD.lit_equal(r09, LIT_R0926) and abs(ctl["C-R0926"]["pose_echo_mean_ade_compliant"] - LIT_POSE_ECHO_MEAN_ADE) <= 5e-5 + 1e-7)
    # ---- gains -------------------------------------------------------------------------------------------------------------------------------
    Wd = RT.frozen_weights(fz, "S2", v0)
    W, Wg = Wd["RETIME_FROZEN30k"], Wd["GATED_RETIME_FROZEN30k"]
    res["gains_S2"] = {"RETIME": [float(x) for x in W], "fit_set": "ALL windows of the surface (see gains.file)"}
    # ---- single-draw (K=1) objects ----------------------------------------------------------------------------------------------------------
    c_os, c_rt, rt_plan = {}, {}, {}
    cs = {}
    for s in range(8):
        c_os[s] = comps(os_[s])
        rt_plan[s] = retime(os_[s], HP, W)
        c_rt[s] = comps(rt_plan[s])
        cs[s] = {"os_E1": cell(c_os[s]["ade_m"], echo_c["ade_m"]), "os_E1P": cell(c_os[s]["ade_m"], pose_c["ade_m"]),
                 "RET_E1": cell(c_rt[s]["ade_m"], echo_c["ade_m"]), "RET_E1P": cell(c_rt[s]["ade_m"], pose_c["ade_m"]), "RET_vs_os": cell(c_rt[s]["ade_m"], c_os[s]["ade_m"]),
                 "mean_ade_os": float(c_os[s]["ade_m"][mask].mean()), "mean_ade_RET": float(c_rt[s]["ade_m"][mask].mean())}
    # C-G1 / C-G0 / C-K1 / C-K1R
    ens_plans = {n: KD.ensemble_mean(np.stack([os_[s] for s in ss])) for n, ss in SETS.items()}
    ctl["C-G1"] = {n: check_g1(p, HP) for n, p in ens_plans.items()}
    ctl["C-G1"]["single_draws"] = {str(s): check_g1(os_[s], HP) for s in (0, 1)}
    ctl["C-G1"]["PASS"] = bool(all(v["PASS"] for v in ctl["C-G1"].values() if isinstance(v, dict) and "PASS" in v) and all(v["PASS"] for v in ctl["C-G1"]["single_draws"].values()))
    ctl["C-G0"] = {n: check_g0(p, HP) for n, p in ens_plans.items() if n in ("4A", "4B")}
    ctl["C-G0"]["PASS"] = bool(all(v["PASS"] for v in ctl["C-G0"].values() if isinstance(v, dict) and "PASS" in v))
    ctl["C-K1"] = {f"s{s}": {"echo": fmt(cs[s]["os_E1"]), "pose": fmt(cs[s]["os_E1P"]), "lit_echo": LIT_K1_ECHO[s], "lit_pose": LIT_K1_POSE[s],
                             "ok": bool(KD.lit_equal(cs[s]["os_E1"], LIT_K1_ECHO[s]) and KD.lit_equal(cs[s]["os_E1P"], LIT_K1_POSE[s]))} for s in (0, 1)}
    ctl["C-K1"]["PASS"] = bool(all(v["ok"] for v in ctl["C-K1"].values() if isinstance(v, dict)))
    k1r_applies = bool(tree == TREE30 and gains_kind == "eval")
    if k1r_applies:
        ctl["C-K1R"] = {f"s{s}": {"pose": fmt(cs[s]["RET_E1P"]), "echo": fmt(cs[s]["RET_E1"]), "lit_pose": LIT_K1R_POSE[s], "lit_echo": LIT_K1R_ECHO[s],
                                  "ok": bool(KD.lit_equal(cs[s]["RET_E1P"], LIT_K1R_POSE[s]) and KD.lit_equal(cs[s]["RET_E1"], LIT_K1R_ECHO[s]))} for s in (0, 1)}
        ctl["C-K1R"]["PASS"] = bool(all(v["ok"] for v in ctl["C-K1R"].values() if isinstance(v, dict)))
    else:
        ctl["C-K1R"] = {"PASS": None, "N/A": "literals exist only for the EMA-30k tree with the eval-fitted gains file"}
    # ---- the sets -----------------------------------------------------------------------------------------------------------------------------
    sets, comp_of = {}, {}
    for n, ss in SETS.items():
        core = stacking_core(os_, ss, HP, W, lambda P: comps(P)["ade_m"], cell)
        ens, combo = core["ens"], core["combo"]
        combo_g = retime(ens, HP, Wg)
        rte = rt_then_ens([os_[s] for s in ss], HP, W)
        C = {"ENS": comps(ens), "COMBO": comps(combo), "COMBO_G": comps(combo_g), "RT_THEN_ENS": comps(rte)}
        comp_of[n] = {"C": C, "combo": combo, "ens": ens}
        ade_ret = core["ade_ret"]
        ade_os = mean_over_draws([c_os[s]["ade_m"] for s in ss])
        rec = {"seeds": list(ss), "K": len(ss), "mean_ade": {k: float(C[k]["ade_m"][mask].mean()) for k in C}, "mean_ade_RET_alone": float(ade_ret[mask].mean()),
               "mean_ade_single_os": float(ade_os[mask].mean())}
        for arm in C:
            rec[f"{arm}_E1P"], rec[f"{arm}_E1"] = cell(C[arm]["ade_m"], pose_c["ade_m"]), cell(C[arm]["ade_m"], echo_c["ade_m"])
        rec["RET_alone_E1P"], rec["RET_alone_E1"] = cell(ade_ret, pose_c["ade_m"]), cell(ade_ret, echo_c["ade_m"])
        rec["ST_ens"], rec["ST_ret"] = core["ST_ens"], core["ST_ret"]             # ST_ens: does RETIME add on top of the ensemble?  ST_ret: does the ensemble add on top of RETIME?
        best_is_ens = rec["mean_ade"]["ENS"] <= rec["mean_ade_RET_alone"]
        rec["best_single_lever"] = "ENS" if best_is_ens else "RET_alone"
        rec["additivity"] = additivity(rec["mean_ade_single_os"], rec["mean_ade"]["ENS"], rec["mean_ade_RET_alone"], rec["mean_ade"]["COMBO"])
        rec["ST_best"] = rec["ST_ens"] if best_is_ens else rec["ST_ret"]
        rec["ST_G_ens"] = cell(C["COMBO_G"]["ade_m"], C["ENS"]["ade_m"])
        rec["TRANSFER_combo_minus_rt_then_ens"] = cell(C["COMBO"]["ade_m"], C["RT_THEN_ENS"]["ade_m"])
        rec["ST_rtthen_ens"], rec["ST_rtthen_ret"] = cell(C["RT_THEN_ENS"]["ade_m"], C["ENS"]["ade_m"]), cell(C["RT_THEN_ENS"]["ade_m"], ade_ret)
        rec["geometry_combo"] = RT.retime_diag(ens, HP, W)
        L_ens = RT.polyline_geometry(ens)["L"][:, -1]
        L_dr = mean_over_draws([RT.polyline_geometry(os_[s])["L"][:, -1] for s in ss])
        ok = mask & (L_dr > 0)
        rec["arc_length_shrinkage_ens_over_mean_draw"] = float(L_ens[ok].mean() / L_dr[ok].mean())
        rec["C-ONPATH_max_dist_m"] = rec["geometry_combo"]["max_dist_to_path_f64_m"]
        # lateral / full rows of the combo against the unmodified ensemble plan (the ACC-LAT object) and against the mean single-draw os
        rows = {}
        for fam, key, label, unit, lower, nr_surf in PR.METRICS:
            mean_os_k = mean_over_draws([c_os[s][key] for s in ss])
            rows[key] = {"family": fam, "label": label, "vs_ens": cell(C["COMBO"][key], C["ENS"][key]), "vs_mean_single_os": cell(C["COMBO"][key], mean_os_k)}
        rec["rows"] = rows
        sets[n] = rec
    # ---- C-ONPATH / C-TRI / C-ENS ----------------------------------------------------------------------------------------------------------
    ctl["C-ONPATH"] = {n: sets[n]["C-ONPATH_max_dist_m"] for n in sets}
    ctl["C-ONPATH"]["PASS"] = bool(all(v <= ON_PATH_TOL_M for k, v in ctl["C-ONPATH"].items() if k != "PASS"))
    tri = {}
    for n, ss in SETS.items():
        per = mean_over_draws([c_os[s]["ade_m"] for s in ss])
        tri[n] = bool((comp_of[n]["C"]["ENS"]["ade_m"] <= per + 1e-6).all())
    ctl["C-TRI"] = {**tri, "PASS": bool(all(tri.values()))}
    if tree == TREE30:
        ens_chk = {n: {"got": fmt(sets[n]["ENS_E1P"]), "lit": LIT_ENS_POSE[n], "ok": KD.lit_equal(sets[n]["ENS_E1P"], LIT_ENS_POSE[n])} for n in LIT_ENS_POSE}
        ctl["C-ENS"] = {**ens_chk, "PASS": bool(all(v["ok"] for v in ens_chk.values()))}
    else:
        kj = PKG / "raw" / "kdraw" / tree / "KDRAW_E1.json"
        if kj.exists():
            kd = json.load(open(kj, encoding="utf-8"))
            ens_chk = {}
            for n, nm in (("4A", "ENS-4A"), ("4B", "ENS-4B")):
                b = (kd.get("arms", {}).get(nm) or {}).get("E1_pose")
                ens_chk[n] = {"got": fmt(sets[n]["ENS_E1P"]), "kdraw": fmt(b), "ok": bool(b and KD.lit_equal(sets[n]["ENS_E1P"], (b["delta"], b["lo"], b["hi"]), tol=1e-6))}
            ctl["C-ENS"] = {**ens_chk, "PASS": bool(all(v["ok"] for v in ens_chk.values())), "source": str(kj)}
        else:
            ctl["C-ENS"] = {"PASS": None, "UNVERIFIED": f"{kj} absent: ENS-K alone could not be cross-checked against the K-draw analysis"}
    # ---- floors / stacking / E1 --------------------------------------------------------------------------------------------------------------
    def floors(a, b):
        return {"COMBO": floor_of(sets[a]["mean_ade"]["COMBO"], sets[b]["mean_ade"]["COMBO"]), "ENS": floor_of(sets[a]["mean_ade"]["ENS"], sets[b]["mean_ade"]["ENS"]),
                "RET_alone": floor_of(sets[a]["mean_ade_RET_alone"], sets[b]["mean_ade_RET_alone"])}

    fl4, fl2 = floors("4A", "4B"), floors("2A", "2B")
    st4 = {"ST_ens": stack_verdict(sets["4A"]["ST_ens"], sets["4B"]["ST_ens"], fl4["COMBO"], fl4["ENS"]),
           "ST_ret": stack_verdict(sets["4A"]["ST_ret"], sets["4B"]["ST_ret"], fl4["COMBO"], fl4["RET_alone"])}
    st2 = {"ST_ens": stack_verdict(sets["2A"]["ST_ens"], sets["2B"]["ST_ens"], fl2["COMBO"], fl2["ENS"]),
           "ST_ret": stack_verdict(sets["2A"]["ST_ret"], sets["2B"]["ST_ret"], fl2["COMBO"], fl2["RET_alone"])}
    e1p4 = {"A": sets["4A"]["COMBO_E1P"], "B": sets["4B"]["COMBO_E1P"], "floor_combo": fl4["COMBO"]}
    e1_4 = KD.arm_verdict(sets["4A"]["COMBO_E1"], sets["4B"]["COMBO_E1"], fl4["COMBO"])
    e1p_v = KD.arm_verdict(e1p4["A"], e1p4["B"], fl4["COMBO"])
    # ---- NR (K=4 sets): single refcv7 draws (bar_NR_metric) + the refcv7 2-draw mean as the secondary comparator --------------------------------
    r7 = {0: np.asarray(pan0["b_refcv7_50400_s0"], np.float32), 1: np.asarray(pan0["b_refcv7_50400_s1"], np.float32)}
    r7["ens2"] = KD.ensemble_mean(np.stack([r7[0], r7[1]]))
    r7c = {k: comps(v) for k, v in r7.items()}
    fam_of = {key: fam for fam, key, _l, _u, _lo, _nr in PR.METRICS}
    nr = {}
    for n in K4:
        for arm in ("COMBO", "ENS"):
            Ca = comp_of[n]["C"][arm]
            per_metric, per_metric2, rows = {}, {}, {}
            for fam, key, label, unit, lower, nr_surf in PR.METRICS:
                row = {"family": fam, "label": label, "vs_r7_s0": cell(Ca[key], r7c[0][key]), "vs_r7_s1": cell(Ca[key], r7c[1][key]), "vs_r7_ens2": cell(Ca[key], r7c["ens2"][key])}
                rows[key] = row
                if "S2" in nr_surf:
                    per_metric[key] = PR.bar_NR_metric({"0": row["vs_r7_s0"], "1": row["vs_r7_s1"]})
                    per_metric2[key] = {"verdict": ("FAIL (separated worse)" if sep_worse(row["vs_r7_ens2"]) else "PASS (not separated worse)")}
            nr[f"{arm}-{n}"] = {"rows": rows, "headline": {k: v["verdict"] for k, v in per_metric.items()}, "rollup": PR.rollup_NR(per_metric, fam_of),
                                "rollup_vs_r7_ens2": PR.rollup_NR(per_metric2, fam_of)}
    nr_ev = {"A": nr["COMBO-4A"]["rollup"], "B": nr["COMBO-4B"]["rollup"]}
    nr_ens_ev = {"A": nr["ENS-4A"]["rollup"], "B": nr["ENS-4B"]["rollup"]}
    nr2_caveat = {s: sorted(nr_family_fail(nr[f"COMBO-4{s}"]["rollup_vs_r7_ens2"]) - nr_family_fail(nr[f"COMBO-4{s}"]["rollup"])) for s in ("A", "B")}
    # ---- ACC-LAT -----------------------------------------------------------------------------------------------------------------------------
    acc_s2 = RT.acc_lat({n[-1]: {k: sets[n]["rows"][k]["vs_ens"] for k in LATERAL_KEYS} for n in K4})
    moved_s2 = bool(sep_better(sets["4A"]["ST_ens"]) and sep_better(sets["4B"]["ST_ens"]))
    acc_s2_v = {"HOLDS": acc_s2["HOLDS"], "moved": moved_s2, "separated_worse_rows": acc_s2["separated_worse_at_any_seed"], "undetermined": acc_s2["undetermined"]}
    s6k2 = None
    if S6 is not None:
        try:
            s6k2 = s6_k2_block(S6, fz, PR, ra, N_BOOT)
        except (Exception, SystemExit) as exc:                          # noqa: BLE001 -- recorded, not hidden
            res["s6_k2_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    acc = {"S2": acc_s2_v, "S6K2": (s6k2 or {}).get("acc_verdict")}
    ev = {"st_ens": st4["ST_ens"], "st_ret": st4["ST_ret"], "e1p": e1p4, "acc": acc, "nr": nr_ev, "nr_ens": nr_ens_ev}
    label = decide(ev)
    # descriptive K=2 label (no NR): decides nothing
    ev2 = {"st_ens": st2["ST_ens"], "st_ret": st2["ST_ret"], "e1p": {"A": sets["2A"]["COMBO_E1P"], "B": sets["2B"]["COMBO_E1P"], "floor_combo": fl2["COMBO"]}, "acc": acc, "nr": None, "nr_ens": None}
    label2 = decide(ev2)
    if label2["label"] == "NO-STACK" and label2["reasons"] == ["NR not evaluated"]:
        label2 = {"label": "STACKING CELLS STACK (K=2 descriptive; NR is not evaluated at K=2, so no STACKS label is issued)", "reasons": []}
    # ---- mode-disagreement diagnostic (combo vs ens curvature on the subset where the K endpoints disagree > 1 m) ---------------------------------
    md = {}
    for n in K4:
        sp = KD.spread_2s(np.stack([os_[s] for s in SETS[n]]))
        sub = mask & (sp > KD.SPREAD_M)
        rec = {"frac_spread_gt_1m_compliant": float((sp[mask] > KD.SPREAD_M).mean()), "n_disagree_compliant": int(sub.sum())}
        if sub.sum() >= 30 and len(np.unique(eid[sub])) >= 5:
            C = comp_of[n]["C"]
            rec["combo_minus_ens_curvature_on_disagree"] = PR.paired_cell(C["COMBO"]["LAT_curvature_mae_1pm_valid"][sub], C["ENS"]["LAT_curvature_mae_1pm_valid"][sub], eid[sub], N_BOOT, BOOT_SEED)
            rec["combo_minus_ens_ade_on_disagree"] = PR.paired_cell(C["COMBO"]["ade_m"][sub], C["ENS"]["ade_m"][sub], eid[sub], N_BOOT, BOOT_SEED)
        md[n] = rec
    hard = [k for k in ("C-MASK", "C-ALIGN", "C-ID", "C-DIST", "C-R0926", "C-G1", "C-G0", "C-K1", "C-K1R", "C-ONPATH", "C-TRI", "C-ENS") if ctl.get(k, {}).get("PASS") is False]
    unverified = [k for k, v in ctl.items() if isinstance(v, dict) and v.get("UNVERIFIED")]
    qual = [res["status"], gmeta["label"], f"tree {tree}", "single training seed", "S6 K=4 UNTESTED (S6 draws exist for inference seeds 0,1 only)",
            "transfer unmodelled: gains fitted on single-draw plans" + ("; fitted on EMA-30k plans, applied to " + tree if tree != TREE30 else "")]
    if any(nr2_caveat.values()):
        qual.append(f"NR2 caveat (refcv7 2-draw mean comparator): COMBO-4 separated worse in {nr2_caveat} where the single-draw comparators pass")
    if unverified:
        qual.append("UNVERIFIED controls: " + ", ".join(unverified))
    res.update({"controls": ctl, "void_controls_failed": hard, "unverified": unverified, "single_draws": {str(s): cs[s] for s in cs}, "sets": sets, "floors_K4": fl4, "floors_K2": fl2,
                "stacking_K4": st4, "stacking_K2": st2, "E1_K4": {"vs_pose": e1p_v, "vs_echo": e1_4}, "NR": nr, "NR2_caveat": nr2_caveat, "ACC_LAT": {"S2_K4": acc_s2, "S6_K2": s6k2, "verdicts": acc},
                "mode_disagreement": md, "outcome": {**label, "qualifiers": qual, "descriptive_K2": label2},
                "echo_mean_ade_compliant": float(echo_c["ade_m"][mask].mean()), "pose_mean_ade_compliant": float(pose_c["ade_m"][mask].mean()), "wall_s": round(time.time() - t0, 1)})
    if hard:
        res["outcome"] = {"label": "VOID (controls failed: " + ", ".join(hard) + ")", "would_have_been": label, "qualifiers": qual}
    (out_dir / f"{stem}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(render(res), encoding="utf-8")
    return res


def s6_k2_block(S6: dict, fz: dict, PR, ra, n_boot: int) -> dict:
    """S6 (compliant) K=2 on draws {0,1}: COMBO-2(S6) = retime(mean(os_s0, os_s1)) with the frozen S6 gains.  ONE set, no replicate."""
    A = S6[0]
    O0, O1 = A["X"]["O"], S6[1]["X"]["O"]
    HP, G, HX = A["X"]["HP"], A["X"]["G"], A["X"]["HX"]
    eid, mask, dt, v0 = A["eid"], np.asarray(A["mask"], bool), A["dt"], A["v0"]
    W = RT.frozen_weights(fz, "S6", v0)["RETIME_FROZEN30k"]

    def comp(P):
        return PR.components(ra, np.ascontiguousarray(P, dtype=np.float32), G, dt)

    def cell(a, b):
        return PR.paired_cell(np.asarray(a)[mask], np.asarray(b)[mask], np.asarray(eid)[mask], n_boot, BOOT_SEED)

    ens = KD.ensemble_mean(np.stack([O0, O1]))
    combo = retime(ens, HP, W)
    Ce, Cc, C0, C1, Cx, Cp = comp(ens), comp(combo), comp(O0), comp(O1), comp(HX), comp(HP)
    rows = {k: cell(Cc[k], Ce[k]) for k in LATERAL_KEYS}
    acc = RT.acc_lat({"S6K2": rows})
    ade_move = cell(Cc["ade_m"], Ce["ade_m"])
    moved = sep_better(ade_move)
    return {"n_windows": int(mask.sum()), "n_episodes": int(len(np.unique(np.asarray(eid)[mask]))), "draws": [0, 1],
            "ADE_combo_minus_ens": ade_move, "ADE_combo_minus_pose": cell(Cc["ade_m"], Cp["ade_m"]), "ADE_combo_minus_echo": cell(Cc["ade_m"], Cx["ade_m"]),
            "ADE_ens_minus_echo": cell(Ce["ade_m"], Cx["ade_m"]), "ADE_ens_minus_pose": cell(Ce["ade_m"], Cp["ade_m"]),
            "lateral_vs_ens": rows, "lateral_vs_os": {f"s{s}": {k: cell(Cc[k], Cs[k]) for k in LATERAL_KEYS} for s, Cs in ((0, C0), (1, C1))},
            "acc_lat": acc, "acc_verdict": {"HOLDS": acc["HOLDS"], "moved": bool(moved), "separated_worse_rows": acc["separated_worse_at_any_seed"], "undetermined": acc["undetermined"]},
            "gains_S6": [float(x) for x in W], "geometry": RT.retime_diag(ens, HP, W)}


# ============================================================================================================== markdown
def render(r: dict) -> str:
    o = r["outcome"]
    L = [f"# COMBO_ENS_RETIME -- {r['tree']} / gains: {r['gains']['kind']}", "",
         f"Tier: {r['tier']}.  Estimator: {r['estimator']}.", f"Status: {r['status']}.", "Bar of record: single-draw `os` (BAR-R8-E1, vs `ha0_ext_pose`, D-R8-E1-REF).  A combo is a NEW arm (inference cost up to xK).", "",
         f"prereg sha256 `{r['prereg_sha256']}`; script sha256 `{r['script_sha256'][:12]}`; gains `{r['gains']['file']}` sha256 `{r['gains']['sha256'][:12]}`.", "",
         f"## OUTCOME (K=4, sets A and B): **{o['label']}**", ""]
    for q in o.get("qualifiers", []):
        L.append(f"* qualifier: {q}")
    for q in o.get("reasons", []):
        L.append(f"* reason: {q}")
    if r.get("void_controls_failed"):
        L.append(f"* **VOID: controls failed -- {r['void_controls_failed']}** (would have been {o.get('would_have_been')})")
    L += ["", "## Controls", "", "| control | PASS |", "|---|---|"]
    for k, v in r["controls"].items():
        if isinstance(v, dict):
            L.append(f"| {k} | {v.get('PASS')}{' UNVERIFIED' if v.get('UNVERIFIED') else ''}{' (N/A: ' + v['N/A'] + ')' if v.get('N/A') else ''} |")
    L += ["", f"Echo `ha0_ext` mean ADE (compliant) {r['echo_mean_ade_compliant']:.4f} m; pose `ha0_ext_pose` {r['pose_mean_ade_compliant']:.4f} m.", "",
          "## E1 cells vs `ha0_ext_pose` (PRIMARY, D-R8-E1-REF) and vs `ha0_ext` (beside), ADE 0-2 s, compliant S2", "", "| set | seeds | arm | vs pose | vs echo | mean ADE |", "|---|---|---|---|---|---|"]
    for n, s in r["sets"].items():
        for arm, lab in (("ENS", "ENS-K"), ("COMBO", "COMBO-K"), ("COMBO_G", "COMBO-G-K (beside)"), ("RT_THEN_ENS", "RT-THEN-ENS-K (beside)")):
            L.append(f"| {n} | {s['seeds']} | {lab} | {fmt(s[arm + '_E1P'])} | {fmt(s[arm + '_E1'])} | {s['mean_ade'][arm]:.4f} |")
        L.append(f"| {n} | {s['seeds']} | RET-alone (mean of {s['K']} single draws) | {fmt(s['RET_alone_E1P'])} | {fmt(s['RET_alone_E1'])} | {s['mean_ade_RET_alone']:.4f} |")
    L += ["", "## STACKING (paired, ADE 0-2 s, compliant S2)", "", "| set | COMBO - ENS-K (adds RETIME) | COMBO - RET-alone (adds ENS) | COMBO - best single (" + "which" + ") | COMBO - RT-THEN-ENS (transfer) |", "|---|---|---|---|---|"]
    for n, s in r["sets"].items():
        L.append(f"| {n} | {fmt(s['ST_ens'])} | {fmt(s['ST_ret'])} | {fmt(s['ST_best'])} ({s['best_single_lever']}) | {fmt(s['TRANSFER_combo_minus_rt_then_ens'])} |")
    L += ["", "## ADDITIVITY (mean ADE, compliant S2; negative = better than the single-draw `os` of the same draws)", "",
          "| set | single os | dENS | dRET | additive combo | observed combo | share of additive | ENS on top of RET (m, share of dENS) | RET on top of ENS (m, share of dRET) |", "|---|---|---|---|---|---|---|---|---|"]
    for n, s in r["sets"].items():
        a = s["additivity"]
        L.append(f"| {n} | {a['single']:.4f} | {a['dENS']:+.4f} | {a['dRET']:+.4f} | {a['additive_combo']:.4f} | {a['combo']:.4f} | {a['share_of_additive']:.2f} | "
                 f"{a['ens_on_top_of_ret_m']:+.4f} ({a['ens_on_top_of_ret_share_of_dENS']:.2f}) | {a['ret_on_top_of_ens_m']:+.4f} ({a['ret_on_top_of_ens_share_of_dRET']:.2f}) |")
    L += ["", f"K=4 floors (m): {r['floors_K4']}.  Stacking verdict K=4: {json.dumps(r['stacking_K4'])}", f"E1 K=4 vs pose: {r['E1_K4']['vs_pose']['verdict']} ; vs echo: {r['E1_K4']['vs_echo']['verdict']}",
          f"K=2 descriptive label: {o.get('descriptive_K2', {}).get('label')} ({o.get('descriptive_K2', {}).get('reasons')})", "", "## ACC-LAT", ""]
    L.append(f"* S2 K=4 (COMBO - ENS-K, five LATERAL rows, sets A and B): {r['ACC_LAT']['verdicts']['S2']}")
    s6 = r["ACC_LAT"]["S6_K2"]
    if s6:
        L.append(f"* S6 K=2 draws {{0,1}} (n {s6['n_windows']} w / {s6['n_episodes']} ep): {r['ACC_LAT']['verdicts']['S6K2']}; ADE combo - ens {fmt(s6['ADE_combo_minus_ens'])}; combo - pose {fmt(s6['ADE_combo_minus_pose'])}")
        for k, c in s6["lateral_vs_ens"].items():
            L.append(f"  * {k}: {fmt(c, 4)}")
    else:
        L.append(f"* S6 K=2: UNAVAILABLE ({r.get('s6_unavailable') or r.get('s6_k2_error')})")
    L += ["", "## NR vs refcv7-50,400 (S2 headline rows; PASS = not separated worse at BOTH refcv7 draws)", ""]
    for k, v in r["NR"].items():
        L.append(f"* {k}: {v['rollup']}   (vs refcv7 2-draw mean: {v['rollup_vs_r7_ens2']})   (STRATEGIC: NOT APPLICABLE, n = 0)")
    L += ["", "## Diagnostics", ""]
    for n, s in r["sets"].items():
        L.append(f"* {n}: arc-length shrinkage of the averaged plan {s['arc_length_shrinkage_ens_over_mean_draw']:.4f}; combo geometry nonmonotone {s['geometry_combo']['frac_nonmonotone_windows']:.3f}, "
                 f"extrapolated {s['geometry_combo']['frac_extrapolated_points']:.3f}, on-path max {s['geometry_combo']['max_dist_to_path_f64_m']:.2e} m")
    for n, m in r["mode_disagreement"].items():
        L.append(f"* mode disagreement {n}: {m}")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--tree", default=TREE30)
    ap.add_argument("--gains", choices=("eval", "train"), default="eval")
    ap.add_argument("--kd-root", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[combo] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    out = Path(a.out_dir) if a.out_dir else OUT
    if mutation() and out.resolve() == OUT.resolve():
        raise SystemExit("[combo] COMBO_TEST_MUTATION is set: refusing to write into the production raw/combo_ens_retime directory")
    r = analyse(a.tree, a.gains, a.kd_root or f"{KD.KIT}/kdraw/{a.tree}", out)
    print("OUTCOME", r["outcome"]["label"], "WROTE", out, f"{r['wall_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
