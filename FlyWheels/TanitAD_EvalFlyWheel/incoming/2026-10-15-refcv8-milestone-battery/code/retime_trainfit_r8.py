"""retime_trainfit_r8.py -- gains for the arc-length RETIME lever fitted on refcv8's TRAIN clips, frozen, scored UNCHANGED on the eval139 EMA-30k trees.

NEW glue module (PREREG_TRAINFIT.md, sha256 pinned below); `a0k0_headroom_retime_r8.py` (RT) is imported and NOT edited.  CPU only, zero GPU.
    python retime_trainfit_r8.py panels                   # roll dumps (E:/refcv8_eval_kit/trainfit/<TREE>/dump_s{0,1}) -> panel_s*, s6_s*
    python retime_trainfit_r8.py freeze                   # fit on ALL train windows, seeds pooled -> raw/retime_trainfit/FROZEN_GAINS_RETIME_train.json
    python retime_trainfit_r8.py forward --tree step30000ema_unc | step30000_unc     # frozen train gains on the banked eval trees -> FORWARD_TRAINFIT_<tree>.json
    python retime_trainfit_r8.py gap                      # train vs eval os ADE (and os / ha0_ext), unpaired episode-cluster bootstrap
    python retime_trainfit_r8.py report                   # outcome label (PREREG section 5) -> RESULTS_TRAINFIT.md
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' python -X utf8 retime_trainfit_r8.py ...
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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a0k0_headroom_retime_r8 as RT  # noqa: E402

PKG = HERE.parent
TF = PKG / "raw" / "retime_trainfit"
PREREG = TF / "PREREG_TRAINFIT.md"
PREREG_SHA256 = "648cdee6e68f70c2a91f518dbcce2999e846d5de6759a7deaf8a7d07311096f6"      # v0 + AMENDMENT 1 (2026-10-19T03:26:11)
PREREG_V0 = TF / "PREREG_TRAINFIT.v0.md"
PREREG_V0_SHA256 = "be16e4dd9ab2c5b45cb571b083f6f7e97217b8134ebbd83c3732b629f0e4cf7e"   # the registered text the amendment appends to
FROZEN_TRAIN = TF / "FROZEN_GAINS_RETIME_train.json"
TRAIN_ROOT = os.environ.get("TF_TRAIN_ROOT", "E:/refcv8_eval_kit/trainfit")
TRAIN_TREE = "step30000ema_train139"
EVAL_WINDOWS_SHA12 = "E:/refcv8_eval_kit/windows/s2_windows.json.sha12.json"
TRAIN_LABELS = "E:/refcv6_eval_kit/data/a6/s2_labels_v8_train.jsonl.gz"
SEEDS = (0, 1)
SURFS = ("S2", "S6")
ARM_RENAME = {"RETIME_FROZEN30k": "RETIME_TRAINFIT", "GATED_RETIME_FROZEN30k": "GATED_RETIME_TRAINFIT"}
A6_SAMPLE_N = 139
A6_SAMPLE_FINGERPRINT = "6086d9cc0c90281619eb0bb63736a73e012322b48dddf5b4ed0363008a0ee3ed"


def sha12(c) -> str:
    return hashlib.sha256(str(c).encode()).hexdigest()[:12]


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_prereg() -> None:
    if sha256_file(PREREG) != PREREG_SHA256:
        raise SystemExit(f"[trainfit] {PREREG} does not hash to the registered sha256 {PREREG_SHA256}: the gains must be produced under the registered text")


# ------------------------------------------------------------------------------------------ guards (pure; pinned by the pytest)
def assert_train_only(fit_sha12, eval_sha12, train_sha12) -> dict:
    """The fit windows' clips must be (a) disjoint from eval139 and (b) inside train4572.  Refuses otherwise (fitting on eval = the leak this test removes)."""
    fit, ev, tr = set(fit_sha12), set(eval_sha12), set(train_sha12)
    overlap, outside = sorted(fit & ev), sorted(fit - tr)
    if overlap:
        raise SystemExit(f"[trainfit] REFUSING: {len(overlap)} fit clips are eval139 clips (first {overlap[:3]}) -- gains must be fitted on TRAIN clips only")
    if outside:
        raise SystemExit(f"[trainfit] REFUSING: {len(outside)} fit clips are not in train4572 (first {outside[:3]})")
    if not fit:
        raise SystemExit("[trainfit] REFUSING: no fit clips")
    fp = hashlib.sha256("\n".join(sorted(fit)).encode()).hexdigest()
    return {"n_fit_clips": len(fit), "n_overlap_with_eval139": 0, "n_outside_train4572": 0, "fit_clips_fingerprint": fp,
            "is_registered_A6_sample": bool(len(fit) == A6_SAMPLE_N and fp == A6_SAMPLE_FINGERPRINT)}


def assert_frozen_is_train(fz: dict) -> None:
    if fz.get("fit_source") != "train" or fz.get("fit_guard", {}).get("n_overlap_with_eval139") != 0:
        raise SystemExit("[trainfit] REFUSING: the frozen gains file is not a train-fitted file (fit_source != 'train' or overlap with eval139 != 0)")


def load_id_sets() -> tuple[set, set]:
    ev = set(json.load(open(EVAL_WINDOWS_SHA12, encoding="utf-8")))
    tr = set()
    with gzip.open(TRAIN_LABELS, "rt", encoding="utf-8") as f:
        for line in f:
            tr.add(sha12(json.loads(line)["clip_id"]))
    return ev, tr


# ------------------------------------------------------------------------------------------ the fit (RT.freeze's per-surface body, on arrays)
def fit_surface(Os: list, HP, G, v0, mask) -> dict:
    """Os = list of plans [n, S, 2], one per inference seed (POOLED; same prior HP, same GT G).  Returns {'all': ..., 'compliant_info_only': ...} like RT.freeze."""
    import s6_lon_decomp as SD
    v0 = np.asarray(v0)
    vb = RT.v0_bins(v0)
    rec = {}
    for set_name in ("all", "compliant_info_only"):
        sel = np.ones(len(v0), bool) if set_name == "all" else np.asarray(mask, bool)
        O2 = np.concatenate([O[sel] for O in Os])
        B2, G2 = np.concatenate([HP[sel]] * len(Os)), np.concatenate([G[sel]] * len(Os))
        vb2 = np.concatenate([vb[sel]] * len(Os))
        st = {}
        geo2, M2 = RT.polyline_geometry(O2), RT.prior_arc(B2)
        rt = RT.retime_fit_pre(geo2, M2, G2, stats=st)
        gated = {}
        for b, (_lo, _hi, name) in enumerate(SD.V0_EDGES):
            nu = int((vb[sel] == b).sum())
            if nu >= RT.MIN_BIN_FIT:
                mb = vb2 == b
                gated[name] = {"bin": b, "n_fit_unique": nu, "g": RT.retime_fit_pre(RT._geo_take(geo2, mb), M2[mb], G2[mb], stats=st).tolist()}
            else:
                gated[name] = {"bin": b, "n_fit_unique": nu, "g": rt.tolist(), "fallback": "RETIME table"}
        Wg = np.stack([np.asarray(gated[n]["g"]) for _a, _b, n in SD.V0_EDGES])[vb2]
        rec[set_name] = {"n_fit_rows_pooled": int(len(O2)), "RETIME": rt.tolist(), "GATED_RETIME": gated, "fit_stats": st,
                         "in_sample_pooled_mean_ade": {"os": float(RT.ade_np(O2, G2).mean()), "ha0_ext_pose": float(RT.ade_np(B2, G2).mean()),
                                                       "RETIME": float(RT.ade_np(RT.retime_apply(O2, B2, rt), G2).mean()),
                                                       "GATED_RETIME": float(RT.ade_np(RT.retime_apply(O2, B2, Wg), G2).mean())}}
        if not (np.all(np.asarray(rt) >= RT.RT_BOX[0]) and np.all(np.asarray(rt) <= RT.RT_BOX[1])):
            raise SystemExit("[trainfit] a gain is outside its box")
    return rec


def freeze_arrays(per_surf: dict, fit_sha12, eval_sha12, train_sha12, fallback: bool = False, guard=None) -> dict:
    """per_surf = {surf: {'Os': [..], 'HP', 'G', 'v0', 'mask', 'instants_s', 'hz', 'n_episodes'}}.  The guard runs FIRST (the mutation test swaps it for a no-op)."""
    g = (guard or assert_train_only)(fit_sha12, eval_sha12, train_sha12)       # resolved at CALL time so the mutation test can swap the module attribute
    out = {"fit_source": "train", "fit_guard": g, "fallback": ("FALLBACK_SINGLE_SEED" if fallback else None), "surfaces": {}}
    for surf, a in per_surf.items():
        rec = {"instants_s": a["instants_s"], "horizons_steps": a["hz"], "n_windows_unique": int(len(a["v0"])), "n_episodes": int(a["n_episodes"])}
        rec.update(fit_surface(a["Os"], a["HP"], a["G"], a["v0"], a["mask"]))
        out["surfaces"][surf] = rec
    return out


# ------------------------------------------------------------------------------------------ outcome label (PREREG section 5), pure
def outcome(fwd: dict, arm: str = "RETIME_TRAINFIT") -> dict:
    """fwd = a FORWARD_TRAINFIT_<tree>.json dict (verdict tree).  Compliant set, both seeds.  Labels exactly as PREREG_TRAINFIT section 5."""
    b = fwd["bars"][f"compliant|{arm}"]
    e1, e1p = b["FZR-E1"], b["FZR-E1P"]
    e1p_ok = all(x.get("pass_separated") for x in e1p["per_inference_seed"].values()) and len(e1p["per_inference_seed"]) == 2
    nr = b["FZR-NR"]["per_family"]
    nr_ok = all(v == "PASS" for v in nr.values())
    nr_fail = any(str(v).startswith("FAIL") for v in nr.values())
    lat = b["FZR-LAT"]
    sec = lambda s, surf: fwd["seeds"][str(s)]["surfaces"][surf]["sets"]["compliant"]["arms"][arm]["minus_os"]["ade_m"]  # noqa: E731
    sep_better = lambda c: bool(c["delta"] < 0 and c["hi"] < 0)  # noqa: E731
    moves = all(sep_better(sec(s, "S2")) for s in SEEDS)
    moved6 = all(sep_better(sec(s, "S6")) for s in SEEDS)
    lat_label = RT.acc_verdict(lat, moved6)
    fail = (lat_label == "FAILS") or nr_fail or (not moves)
    pass_ = (not fail) and str(e1["verdict"]).startswith("PASS") and e1p_ok and lat_label == "HOLDS" and nr_ok
    label = "FAIL" if fail else ("DEPLOYABLE-PASS" if pass_ else "PARTIAL")
    return {"label": label, "E1_verdict": e1["verdict"], "E1P_separated_both_seeds": e1p_ok, "ACC_LAT_label": lat_label, "NR_per_family": nr,
            "MOVES_S2_vs_os_separated_better_both_seeds": moves, "MOVED_S6_vs_os_separated_better_both_seeds": moved6,
            "fail_reasons": [r for r, c in (("ACC-LAT FAILS", lat_label == "FAILS"), ("NR family FAIL", nr_fail), ("NOT MOVES (S2 arm - os not separated better at both seeds)", not moves)) if c]}


# ------------------------------------------------------------------------------------------ AMENDMENT 1: constant-gain arms and known-value controls (pure)
KNOWN_TOL_M = 1e-6                                           # = RT.RT_ON_PATH_TOL_M
CONST_ARM_NAMES = ("CONST_G1", "CONST_G0", "CONST_G05")


def const_weights(T: int, g1: float = 1.0, g0: float = 0.0, g05: float = 0.5) -> dict:
    """The three constant-gain arms as 1-D gain vectors (the mutation tests pass a wrong g1 / g0 to prove the controls withhold)."""
    return {"CONST_G1": np.full(T, g1), "CONST_G0": np.full(T, g0), "CONST_G05": np.full(T, g05)}


def walker_positions(O, HP) -> np.ndarray:
    """INDEPENDENT, loop-based: the point of the os polyline (V_0=(0,0), V_k=O_k) at the PRIOR's chord-sum arc length M_k (W_0=(0,0), W_k=HP_k), per window and
    instant; past the last vertex along the last positive-length segment's direction (+x if the whole polyline is degenerate).  Shares no code with
    RT.eval_polyline / RT.prior_arc / RT.retime_*.  This is the position CONST_G0 must produce.  Returns [n, S, 2] float64."""
    import math
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


def known_value_arrays(O, HP, g1: float = 1.0, g0: float = 0.0) -> dict:
    """Array-level known values: CONST_G1 (all gains `g1`, default 1) bit-equal to os after the cast; CONST_G0 (all gains `g0`, default 0, float64, no cast)
    equal to the independent walker (os path at the prior's arc length) -- the arc length per instant asserted against the prior's."""
    T = np.asarray(O).shape[1]
    P1 = RT.retime_apply(O, HP, np.full(T, g1))
    P0 = RT.retime_apply(O, HP, np.full(T, g0), cast=False)
    exp0 = walker_positions(O, HP)
    return {"n_windows": int(len(O)), "g1_bit_equal_os": bool(np.array_equal(P1, O)),
            "g1_max_abs_m": float(np.max(np.abs(P1.astype(np.float64) - np.asarray(O, np.float64)))),
            "g0_vs_walker_max_abs_m": float(np.max(np.abs(P0 - exp0)))}


def assert_known_values(kv: dict, tol: float = KNOWN_TOL_M) -> None:
    """WITHHOLD (SystemExit) unless every cell reads CONST_G1 == os bit for bit and CONST_G0 == the prior-arc-length walker within `tol`."""
    bad = [f"{k}: g1_bit_equal={v['g1_bit_equal_os']} g1_max={v['g1_max_abs_m']:.3g} g0_walker_max={v['g0_vs_walker_max_abs_m']:.3g}"
           for k, v in kv.items() if not (v["g1_bit_equal_os"] and v["g1_max_abs_m"] == 0.0 and v["g0_vs_walker_max_abs_m"] <= tol)]
    if bad or not kv:
        raise SystemExit("[trainfit] WITHHELD: a known-value control did not read its known value: " + ("; ".join(bad) if bad else "no cell was checked"))


def assert_const_g1_cells(res: dict) -> int:
    """WITHHOLD unless CONST_G1 - os is exactly 0.0 with a zero-width CI on EVERY METRICS row of every (seed, surface, set) cell.  Returns the rows checked."""
    n = 0
    for s, sr in res["seeds"].items():
        for surf, ss in sr["surfaces"].items():
            for set_name, r in ss["sets"].items():
                cells = r["arms"]["CONST_G1"]["minus_os"]
                if "ade_m" not in cells or "delta" not in cells["ade_m"]:
                    raise SystemExit(f"[trainfit] WITHHELD: CONST_G1 has no ADE cell at seed {s} {surf} {set_name}")
                for k, c in cells.items():
                    if "delta" in c:
                        if not (c["delta"] == 0.0 and c["lo"] == 0.0 and c["hi"] == 0.0):
                            raise SystemExit(f"[trainfit] WITHHELD: CONST_G1 - os row {k} at seed {s} {surf} {set_name} reads {c['delta']!r} [{c['lo']!r}, {c['hi']!r}], not exactly 0")
                        n += 1
                    elif c.get("status") != "REFUSED":
                        raise SystemExit(f"[trainfit] WITHHELD: CONST_G1 row {k} at seed {s} {surf} {set_name} has no interval and is not REFUSED")
    return n


def const_reading(vs: dict, arm: str = "RETIME_TRAINFIT") -> dict:
    """AMENDMENT 1 A1.3: train-fit - CONST_G05 ADE, compliant set, S2 and S6, both seeds (four cells).  vs = {'<surf>|<set>|s<seed>': {arm: {metric: cell}}}."""
    cells = {f"{surf}|s{s}": vs[f"{surf}|compliant|s{s}"][arm]["ade_m"] for surf in SURFS for s in SEEDS}
    if any("delta" not in c for c in cells.values()):
        return {"reading": "UNDETERMINED", "cells": cells}
    if all(c["lo"] <= 0.0 <= c["hi"] for c in cells.values()):
        rd, sent = "INDISTINGUISHABLE", "refcv8's residual over the prior is over-scaled, and the fitted values add nothing beyond about half. The fitted gains are then not the evidence."
    elif all(c["delta"] < 0 and c["hi"] < 0 for c in cells.values()):
        rd, sent = "FITTED-SHAPE-MATTERS", "the fitted shape carries information beyond a constant 0.5."
    elif any(c["lo"] > 0 for c in cells.values()):
        rd, sent = "CONST-BEATS-FIT", "at least one cell has the constant 0.5 separated better than the fit."
    else:
        rd, sent = "MIXED", "some cells separate the fit from the constant, not all four, none the other way."
    return {"reading": rd, "sentence": sent, "cells": cells}


# ------------------------------------------------------------------------------------------ unpaired episode-cluster bootstrap (train vs eval: different clips)
def cluster_stats(x, eid):
    x, eid = np.asarray(x, np.float64), np.asarray(eid)
    ok = np.isfinite(x)
    u, inv = np.unique(eid[ok], return_inverse=True)
    return np.bincount(inv, weights=x[ok], minlength=len(u)), np.bincount(inv, minlength=len(u)).astype(np.float64)


def unpaired_gap(a_by_arm: dict, b_by_arm: dict, num="os", den="ha0_ext", n_boot=2000, seed=0) -> dict:
    """a_by_arm / b_by_arm = {'os': (values, eid), 'ha0_ext': (values, eid)} for two DIFFERENT clip sets.  Independent episode resampling within each set;
    the per-set ratio num/den uses the SAME resampled episodes inside a set (paired within the set)."""
    rng = np.random.default_rng(seed)
    S = {k: {arm: cluster_stats(*v[arm]) for arm in (num, den)} for k, v in (("a", a_by_arm), ("b", b_by_arm))}

    def draw(k):
        n = len(S[k][num][0])
        i = rng.integers(0, n, n)
        return {arm: S[k][arm][0][i].sum() / S[k][arm][1][i].sum() for arm in (num, den)}
    pt = {k: {arm: S[k][arm][0].sum() / S[k][arm][1].sum() for arm in (num, den)} for k in S}
    bs = [(draw("a"), draw("b")) for _ in range(n_boot)]
    ci = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]  # noqa: E731
    out = {"n_boot": n_boot, "seed": seed, "estimator": "independent episode-cluster bootstrap within each set (two different clip samples: no pairing exists)"}
    for arm in (num, den):
        out[f"{arm}_mean_a"], out[f"{arm}_mean_b"] = float(pt["a"][arm]), float(pt["b"][arm])
        out[f"{arm}_ci_a"], out[f"{arm}_ci_b"] = ci([x[0][arm] for x in bs]), ci([x[1][arm] for x in bs])
        d = [x[0][arm] - x[1][arm] for x in bs]
        out[f"{arm}_diff_a_minus_b"] = {"delta": float(pt["a"][arm] - pt["b"][arm]), "lo": ci(d)[0], "hi": ci(d)[1]}
    ra, rb = pt["a"][num] / pt["a"][den], pt["b"][num] / pt["b"][den]
    rd = [x[0][num] / x[0][den] - x[1][num] / x[1][den] for x in bs]
    out["ratio_num_over_den"] = {"a": float(ra), "b": float(rb), "diff_a_minus_b": {"delta": float(ra - rb), "lo": ci(rd)[0], "hi": ci(rd)[1]}}
    return out


# ------------------------------------------------------------------------------------------ IO-bound commands (need the stack; exercised by the real run)
def _patch_archive(root: str):
    import clamp_posthoc_r8 as CP
    import compliant_reread_r8 as CR
    CP.ARCHIVE = root
    CR.ARCHIVE = root
    return CP, CR


def train_surface_inputs(tools: dict, ctx: dict, surf: str, first: dict, tree: str) -> dict:
    """RT.surface_inputs without the refcv7 'R' arm (train panels have no banked baseline), plus the clip sha12 of every window."""
    P = ctx["panels"][surf]
    rows, eid, A, dt = P["rows"], P["eid"], P["A"], P["dt"]
    clip, v0, ws = ctx["clip"][rows], ctx["v0"][rows], ctx["ws"][rows]
    hz = [int(h) for h in P["man"]["grid"]["horizons_steps"]]
    dec = RT.load_decisions(tools["NR"], P["dir"], ws, v0)
    REF, V = RT.references(dec, A["ha0_ext"], hz)
    key = np.stack([clip, ws])
    if surf in first:
        if not (all(np.array_equal(REF[k], first[surf][k]) for k in REF) and np.array_equal(first[surf]["_key"], key) and np.array_equal(first[surf]["G"], A["g"])):
            raise SystemExit(f"[trainfit] {tree} {surf}: references / GT / window order differ between inference seeds")
    else:
        first[surf] = {**REF, "_key": key, "G": A["g"]}
    meta = json.load(open(ctx["dump"] + ".json", encoding="utf-8"))["clip_sha12_by_index"]
    return {"X": {"O": A["os"], "HX": A["ha0_ext"], "HP": REF["HP"], "G": A["g"]}, "eid": eid, "clip": clip, "clip_sha12": [meta[str(int(c))] for c in clip], "v0": v0,
            "ws": ws, "mask": ctx["comp"][rows], "dt": dt, "hz": hz, "dec": dec, "V": V, "dir": P["dir"]}


def cmd_panels() -> int:
    import clamp_posthoc_r8 as CP
    import r8_battery_patch as BP
    _L, P = BP.apply()
    for s in SEEDS:
        d, root = f"{TRAIN_ROOT}/{TRAIN_TREE}/dump_s{s}", f"{TRAIN_ROOT}/{TRAIN_TREE}"
        if not os.path.exists(f"{d}/manifest.json"):
            raise SystemExit(f"[trainfit panels] {d} has no manifest: the seed-{s} roll has not run")
        rec = P.build_panel_dump(d, f"{root}/panel_s{s}", os.path.join(d, "refcv6_extras.npz"), baselines={})
        s6 = P.build_s6_dump(d, f"{root}/s6_s{s}", baselines={})
        print(f"panels seed {s}: S2 {rec['n_windows']} windows / {rec['n_episodes']} eps; S6 {s6.get('n_windows')}", flush=True)
    del CP
    print("ZZTFPANELSDONEZZ")
    return 0


def _load_train(tools, seeds):
    CP, CR = _patch_archive(TRAIN_ROOT)
    first, per, ctxs = {}, {s: {} for s in SURFS}, {}
    for seed in seeds:
        ctx = CR.prepare_seed(TRAIN_TREE, seed, tools, tools["rc"])
        for surf in SURFS:
            S = train_surface_inputs(tools, ctx, surf, first, TRAIN_TREE)
            if not (S["dec"]["chk"]["PASS"] and S["V"]["PASS"]):
                raise SystemExit(f"[trainfit] {surf} s{seed}: decisions / validation failed")
            per[surf][seed] = S
        ctxs[seed] = {"mask_sha": ctx["sha"]}
        del ctx
    return per, ctxs


def cmd_freeze() -> int:
    check_prereg()
    tools = RT.tools_load()
    have = [s for s in SEEDS if os.path.exists(f"{TRAIN_ROOT}/{TRAIN_TREE}/s6_s{s}/manifest.json")]
    if 0 not in have:
        raise SystemExit("[trainfit freeze] the seed-0 train panels do not exist")
    seeds = tuple(have) if len(have) == 2 else (0,)
    per, ctxs = _load_train(tools, seeds)
    ev, tr = load_id_sets()
    fit_ids = set(per["S2"][0]["clip_sha12"])
    per_surf = {}
    for surf in SURFS:
        s0 = per[surf][0]
        per_surf[surf] = {"Os": [per[surf][s]["X"]["O"] for s in seeds], "HP": s0["X"]["HP"], "G": s0["X"]["G"], "v0": s0["v0"], "mask": s0["mask"],
                          "instants_s": [h / 10 for h in s0["hz"]], "hz": s0["hz"], "n_episodes": len(np.unique(s0["eid"]))}
    out = freeze_arrays(per_surf, fit_ids, ev, tr, fallback=(len(seeds) < 2))
    out.update({"tool": "retime_trainfit_r8.py freeze", "generated": time.strftime("%FT%T"), "prereg_sha256": PREREG_SHA256, "script_sha256": sha256_file(__file__),
                "tree": f"{TRAIN_ROOT}/{TRAIN_TREE}", "seeds_pooled": list(seeds), "mask_sha_by_seed": ctxs,
                "fit_set": "ALL train windows of the surface, inference seeds pooled", "prior": "ha0_ext_pose",
                "input_files": {f"{surf}|s{s}": RT._sha_dir(per[surf][s]["dir"]) for surf in SURFS for s in seeds},
                "n_windows_by_surface": {surf: int(len(per[surf][0]["v0"])) for surf in SURFS},
                "n_compliant_by_surface": {surf: int(per[surf][0]["mask"].sum()) for surf in SURFS}})
    RT.jdump(out, FROZEN_TRAIN)
    sh = sha256_file(FROZEN_TRAIN)
    (TF / "FROZEN_GAINS_RETIME_train.sha256").write_text(sh + "\n")
    print(f"WROTE {FROZEN_TRAIN} sha256 {sh}")
    for surf in SURFS:
        print(surf, "RETIME g", np.round(np.asarray(out["surfaces"][surf]["all"]["RETIME"]), 3).tolist(), out["surfaces"][surf]["all"]["in_sample_pooled_mean_ade"])
    return 0


def const_extras(tree: str, fz: dict, tools: dict) -> tuple[dict, dict]:
    """Array-level known values (all windows, every seed x surface) and the paired train-fit - CONST_G05 cells (every METRICS row, both sets), on the banked eval tree."""
    import compliant_reread_r8 as CR
    PR = tools["PR"]
    keys = [m[1] for m in PR.METRICS]
    kv, vs, first = {}, {}, {}
    for seed in SEEDS:
        ctx = CR.prepare_seed(tree, seed, tools, tools["rc"])
        for surf in SURFS:
            S = RT.surface_inputs(tools, ctx, surf, first, tree)
            fwts = RT.frozen_weights(fz, surf, S["v0"])             # the patched version: includes the CONST arms
            kv[f"{surf}|s{seed}"] = known_value_arrays(S["X"]["O"], S["X"]["HP"])
            for set_name in RT.SETS:
                sel = S["mask"] if set_name == "compliant" else np.ones(len(S["eid"]), bool)
                X = {k: v[sel] for k, v in S["X"].items()}
                e = S["eid"][sel]
                comp = lambda P: PR.components(tools["ra"], P, X["G"], S["dt"])  # noqa: E731
                c5 = comp(RT.retime_apply(X["O"], X["HP"], fwts["CONST_G05"]))
                cell = {}
                for arm, key in (("RETIME_TRAINFIT", "RETIME_FROZEN30k"), ("GATED_RETIME_TRAINFIT", "GATED_RETIME_FROZEN30k")):
                    W = fwts[key]
                    cA = comp(RT.retime_apply(X["O"], X["HP"], W if W.ndim == 1 else W[sel]))
                    cell[arm] = {k: RT.slim(PR.paired_cell(cA[k], c5[k], e, RT.N_BOOT)) for k in keys}
                vs[f"{surf}|{set_name}|s{seed}"] = cell
        del ctx
    return kv, vs


def cmd_forward(tree: str) -> int:
    check_prereg()
    fz = json.load(open(FROZEN_TRAIN, encoding="utf-8"))
    assert_frozen_is_train(fz)
    want = (TF / "FROZEN_GAINS_RETIME_train.sha256").read_text().strip()
    if sha256_file(FROZEN_TRAIN) != want:
        raise SystemExit("[trainfit forward] the frozen gains file differs from the sha256 recorded at freeze time")
    RT.FROZEN_JSON, RT.OUT = FROZEN_TRAIN, TF
    cache = {}
    orig_tl, orig_fw = RT.tools_load, RT.frozen_weights

    def tools_load_once():
        if "t" not in cache:
            cache["t"] = orig_tl()
        return cache["t"]

    def frozen_weights_plus_const(fz_, surf, v0):                   # AMENDMENT 1: the constant arms ride the SAME forward path
        d = orig_fw(fz_, surf, v0)
        d.update(const_weights(len(fz_["surfaces"][surf]["all"]["RETIME"])))
        return d
    RT.tools_load, RT.frozen_weights = tools_load_once, frozen_weights_plus_const
    rc = RT.forward(tree)
    p = TF / f"FORWARD_RETIME_{tree}.json"
    res = json.load(open(p, encoding="utf-8"))
    res["role"] = ("DEPLOYABLE TEST (VERDICT TREE): gains fitted on TRAIN clips, frozen, scored on eval139 EMA-30k" if tree == "step30000ema_unc" else
                   "BESIDE (not in the verdict): the same frozen train gains on another checkpoint of the same run (raw 30k)")
    res["fit_source"], res["fit_guard"], res["prereg_sha256"] = fz["fit_source"], fz["fit_guard"], PREREG_SHA256
    res["note_arm_names"] = ("RETIME_TRAINFIT / GATED_RETIME_TRAINFIT = the arms RT.forward calls *_FROZEN30k, with the TRAIN-fitted tables; "
                             "CONST_G1 / CONST_G0 = known-value controls, CONST_G05 = the no-fit comparator (AMENDMENT 1)")
    for k in list(res["bars"]):
        for old, new in ARM_RENAME.items():
            if k.endswith("|" + old):
                res["bars"][k[: -len(old)] + new] = res["bars"].pop(k)
    for sr in res["seeds"].values():
        for ss in sr["surfaces"].values():
            for r in ss["sets"].values():
                for old, new in ARM_RENAME.items():
                    if old in r["arms"]:
                        r["arms"][new] = r["arms"].pop(old)
    withheld = TF / f"WITHHELD_{tree}.json"
    try:
        n_rows = assert_const_g1_cells(res)
        kv, vs = const_extras(tree, fz, cache["t"])
        assert_known_values(kv)
    except SystemExit as e:
        RT.jdump({"tree": tree, "WITHHELD": str(e), "generated": time.strftime("%FT%T"), "prereg_sha256": PREREG_SHA256}, withheld)
        os.remove(p)
        stale = TF / f"FORWARD_TRAINFIT_{tree}.json"
        if stale.exists():
            os.remove(stale)                                          # a withheld run leaves NO usable forward file behind
        print(f"{e}\nZZTFWITHHELDZZ {tree}", flush=True)
        return 4
    if withheld.exists():
        os.remove(withheld)
    res["const"] = {"ALL_PASS": True, "tolerance_m": KNOWN_TOL_M, "g1_cell_rows_checked_exactly_zero": n_rows, "known_value_arrays": kv, "vs_const_g05": vs,
                    "reading": {arm: const_reading(vs, arm) for arm in ("RETIME_TRAINFIT", "GATED_RETIME_TRAINFIT")}}
    out = TF / f"FORWARD_TRAINFIT_{tree}.json"
    RT.jdump(res, out)
    os.remove(p)
    print(f"WROTE {out} controls={res['controls']['ALL_PASS']} const_controls=PASS reading={res['const']['reading']['RETIME_TRAINFIT']['reading']}")
    return rc


def cmd_gap() -> int:
    check_prereg()
    tools = RT.tools_load()
    PR, ra = tools["PR"], tools["ra"]
    out = {"tool": "retime_trainfit_r8.py gap", "prereg_sha256": PREREG_SHA256, "tier": RT.TIER, "cells": {}}
    ptrain, _ = _load_train(tools, SEEDS)
    import compliant_reread_r8 as CR
    import clamp_posthoc_r8 as CP
    CP.ARCHIVE = CR.ARCHIVE = "C:/Users/Admin/refcv8_eval_archive/battery"
    first = {}
    for seed in SEEDS:
        ctx = CR.prepare_seed("step30000ema_unc", seed, tools, tools["rc"])
        for surf in SURFS:
            Se = RT.surface_inputs(tools, ctx, surf, first, "step30000ema_unc")
            St = ptrain[surf][seed]
            for set_name in ("all", "compliant"):
                def side(S, ext):
                    sel = np.ones(len(S["eid"]), bool) if set_name == "all" else S["mask"]
                    return {arm: (PR.components(ra, S["X"][k], S["X"]["G"], S["dt"])["ade_m"][sel], np.asarray(S["eid"])[sel]) for arm, k in (("os", "O"), ("ha0_ext", "HX"))}
                out["cells"][f"{surf}|{set_name}|s{seed}"] = unpaired_gap(side(St, False), side(Se, True))
        del ctx
    out["a"], out["b"] = "TRAIN (A6 139 clips)", "EVAL139"
    RT.jdump(out, TF / "GAP_train_vs_eval.json")
    print("WROTE", TF / "GAP_train_vs_eval.json")
    return 0


def _f(c) -> str:
    if not c or "delta" not in c:
        return "n/a"
    return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" + (" **SEP**" if c.get("separated") else "")


def cmd_report() -> int:
    check_prereg()
    fz = json.load(open(FROZEN_TRAIN, encoding="utf-8"))
    ev = json.load(open(RT.OUT / "FROZEN_GAINS_RETIME_ema30k.json", encoding="utf-8"))
    fwd = json.load(open(TF / "FORWARD_TRAINFIT_step30000ema_unc.json", encoding="utf-8"))
    if (TF / "WITHHELD_step30000ema_unc.json").exists() or not fwd.get("const", {}).get("ALL_PASS") or not fwd["controls"]["ALL_PASS"]:
        raise SystemExit("[trainfit report] REFUSING: the forward run is withheld or a control did not pass; no outcome label is produced")
    res = {a: outcome(fwd, a) for a in ("RETIME_TRAINFIT", "GATED_RETIME_TRAINFIT")}
    rd = fwd["const"]["reading"]
    L = ["# RESULTS_TRAINFIT -- RETIME gains fitted on TRAIN clips, frozen, scored on eval139 (EMA-30k, T1 open loop, single training seed)", "",
         f"prereg (v0 + AMENDMENT 1) sha256 `{PREREG_SHA256}`; frozen gains sha256 `{sha256_file(FROZEN_TRAIN)}`; fit source = {fz['fit_source']}, fit clips {fz['fit_guard']['n_fit_clips']}, "
         f"overlap with eval139 {fz['fit_guard']['n_overlap_with_eval139']}, registered A6 sample: {fz['fit_guard']['is_registered_A6_sample']}; seeds pooled {fz['seeds_pooled']} {fz.get('fallback') or ''}",
         "Estimator: paired episode-cluster bootstrap over eval episodes (B 2000, seed 0, gains and models fixed) = 'another draw of EVAL EPISODES' only; not another train-clip draw, training run or inference run.", "",
         f"## OUTCOME (arm RETIME, committed table PREREG section 5): **{res['RETIME_TRAINFIT']['label']}**", "", "```", json.dumps(res["RETIME_TRAINFIT"], indent=1), "```",
         f"GATED_RETIME (beside, cannot rescue or sink): {res['GATED_RETIME_TRAINFIT']['label']}", "",
         f"## CONST_G05 reading (AMENDMENT 1 A1.3; does not change the label): RETIME **{rd['RETIME_TRAINFIT']['reading']}** | GATED_RETIME {rd['GATED_RETIME_TRAINFIT']['reading']}",
         rd["RETIME_TRAINFIT"].get("sentence", ""), "", "train-fit - CONST_G05, ADE, compliant, EMA-30k:", ""]
    for k, c in rd["RETIME_TRAINFIT"]["cells"].items():
        L.append(f"* {k}: {_f(c)}")
    kv = fwd["const"]["known_value_arrays"]
    L += ["", f"## Known-value controls: PASS (else the run is withheld). CONST_G1 - os exactly 0.0 with zero-width CI on {fwd['const']['g1_cell_rows_checked_exactly_zero']} rows; "
          f"CONST_G1 bit-equal os and CONST_G0 == independent prior-arc-length walker (<= {KNOWN_TOL_M} m) on {len(kv)} cells: max walker error {max(v['g0_vs_walker_max_abs_m'] for v in kv.values()):.3g} m.", "",
          "## Gains per instant: train-fit RETIME vs eval-fit (FROZEN_GAINS_RETIME_ema30k.json, all eval windows, seeds pooled) and the constants", ""]
    for surf in SURFS:
        tr, evg = fz["surfaces"][surf]["all"]["RETIME"], ev["surfaces"][surf]["all"]["RETIME"]
        inst = fz["surfaces"][surf]["instants_s"]
        L += [f"{surf}: instant(s) " + " ".join(f"{x:g}" for x in inst), f"* train-fit : " + " ".join(f"{x:.3f}" for x in tr), f"* eval-fit  : " + " ".join(f"{x:.3f}" for x in evg),
              f"* CONST_G1/G05/G0: 1 / 0.5 / 0", ""]
    L += ["## Same table, S2 and S6, ceiling-compliant, EMA-30k (ADE; arm minus reference; s0 | s1)", ""]
    for surf in SURFS:
        L += [f"### {surf}", "", "| arm | - ha0_ext s0 | s1 | - ha0_ext_pose s0 | s1 | - os s0 | s1 |", "|---|---|---|---|---|---|---|"]
        for arm in ("RETIME_TRAINFIT", "GATED_RETIME_TRAINFIT", "CONST_G05", "CONST_G1", "CONST_G0"):
            row = [fwd["seeds"][str(s)]["surfaces"][surf]["sets"]["compliant"]["arms"][arm] for s in SEEDS]
            L.append(f"| {arm}{' (control)' if arm in ('CONST_G1', 'CONST_G0') else ' (comparator)' if arm == 'CONST_G05' else ''} | "
                     + " | ".join([_f(row[0]["minus_ha0_ext"]), _f(row[1]["minus_ha0_ext"]), _f(row[0]["minus_ha0_ext_pose"]), _f(row[1]["minus_ha0_ext_pose"]),
                                   _f(row[0]["minus_os"]["ade_m"]), _f(row[1]["minus_os"]["ade_m"])]) + " |")
        L.append("")
    L += ["## All four families, RETIME_TRAINFIT, compliant (arm - os, arm - refcv7-50,400; s0 | s1). STRATEGIC: NOT APPLICABLE, n = 0 (--no-strategic).", ""]
    PRm = RT.tools_load()["PR"]
    for surf in SURFS:
        L += [f"### {surf}", "", "| family | row | - os s0 | s1 | - refcv7 s0 | s1 |", "|---|---|---|---|---|---|"]
        for fam, key, _lab, _u, _lo, surfs_ in PRm.METRICS:
            if surf not in surfs_:
                continue
            a = [fwd["seeds"][str(s)]["surfaces"][surf]["sets"]["compliant"]["arms"]["RETIME_TRAINFIT"] for s in SEEDS]
            L.append(f"| {fam} | {key} | " + " | ".join([_f(a[0]["minus_os"][key]), _f(a[1]["minus_os"][key]), _f(a[0]["NR_minus_refcv7"][key]), _f(a[1]["NR_minus_refcv7"][key])]) + " |")
        L.append("")
    gp = TF / "GAP_train_vs_eval.json"
    if gp.exists():
        g = json.load(open(gp, encoding="utf-8"))
        L += ["## Train vs eval139 os ADE (A = TRAIN, B = EVAL; unpaired episode-cluster bootstrap). Headline = os / ha0_ext per set (difficulty-normalised)", ""]
        for k, c in g["cells"].items():
            r = c["ratio_num_over_den"]
            L.append(f"* {k}: os ADE A {c['os_mean_a']:.4f} B {c['os_mean_b']:.4f} (diff {c['os_diff_a_minus_b']['delta']:+.4f} [{c['os_diff_a_minus_b']['lo']:+.4f}, {c['os_diff_a_minus_b']['hi']:+.4f}]); "
                     f"os/ha0_ext A {r['a']:.4f} B {r['b']:.4f} (diff {r['diff_a_minus_b']['delta']:+.4f} [{r['diff_a_minus_b']['lo']:+.4f}, {r['diff_a_minus_b']['hi']:+.4f}])")
        L.append("")
    L += ["Full cells (both sets, raw-30k beside, GATED, all rows): `FORWARD_TRAINFIT_step30000ema_unc.json`, `FORWARD_TRAINFIT_step30000_unc.json`; gap: `GAP_train_vs_eval.json`."]
    (TF / "RESULTS_TRAINFIT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("WROTE", TF / "RESULTS_TRAINFIT.md", res["RETIME_TRAINFIT"]["label"], rd["RETIME_TRAINFIT"]["reading"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("panels", "freeze", "forward", "gap", "report"))
    ap.add_argument("--tree")
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("[trainfit] set CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    return {"panels": cmd_panels, "freeze": cmd_freeze, "gap": cmd_gap, "report": cmd_report, "forward": lambda: cmd_forward(a.tree)}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
