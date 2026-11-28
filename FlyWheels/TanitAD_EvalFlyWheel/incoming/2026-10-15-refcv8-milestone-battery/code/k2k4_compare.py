"""k2k4_compare.py -- the K2 / K4 known-value controls of AMENDMENT A-R8-CLAMP-1 section 3, as COMPARISON CODE (CPU; numpy) + the episode-prefix planner.

    python k2k4_compare.py plan    --banked <tree dir, e.g. .../battery/step30000ema_unc> --windows-json E:/refcv8_eval_kit/windows/s2_windows.json \
                                   --out-windows <subset.json> --out-plan <plan.json> [--n-episodes N]
    python k2k4_compare.py compare --banked <tree dir> --off-dump <flag-OFF roll dump dir> --on-dump <flag-ON roll dump dir> --out K2K4_<step>.json \
                                   [--control-dump <fb47a58-tree roll dump dir>] [--gpu-posthoc] [--step <label>]

WHAT IT CHECKS (every criterion below was written BEFORE any GPU number existed; it is not editable by the result):
  K2  FLAG-OFF PARITY.  The clamp-carrying tree (de4e15b), flag OFF, rolled with the SAME seed on a PREFIX of the banked dump's episodes, must reproduce the
      banked `dump_s0/refcv6_extras.npz :: nav_true.traj` BIT-FOR-BIT on every window (rows keyed by (clip_index, ws), exactly as nr_matched_instrument
      keys them).  PASS iff >= 3 episodes, no window missing from the banked dump, and `np.array_equal(..., equal_nan=True)` on all rows.
      SECONDARY (reported, does not change the K2 verdict): every other extras key present in both, EXCEPT `nav_shuffled.*` -- the nav shuffle is a
      permutation over the WHOLE window list (`refcv3_arm.shuffle_nav`), so a prefix roll legitimately feeds other shuffled tokens.
      WHY A PREFIX: refcv3_arm seeds the generator ONCE (`torch.manual_seed(infer_seed)`, refcv3_arm.py:1999) and draws sequentially, so a window's DDIM
      draw depends on every window before it.  Only the first K episodes of the dump, rolled from window 0 with the SAME arm set (`full`), see the same stream.
  K4  ON vs POST-HOC.  The flag-ON roll of the same prefix: `<cond>.traj` must agree within 1e-6 m with the vendored landed function
      (`clamp_posthoc_r8` + `_vendored_de4e15b_refcv6_selection.py`) applied to the flag-OFF roll's plan, per condition in {nav_true, nav_shuffled,
      nav_flipped, nav_zero}, with limits derived by `clamp_posthoc_r8.fed_ceiling` and v0 from the extras.  PASS iff max |delta| <= 1e-6 on ALL rows of
      ALL four conditions AND >= 5 nav_true rows are BINDING (a K4 in which nothing binds proves nothing about the clamp arithmetic: INCONCLUSIVE).
      SECONDARY (reported): K4a ON `traj_precap` == OFF `traj` bit-for-bit (the clamp perturbs nothing upstream); K4b the forward's `plan_v_lim_ms` ==
      the post-hoc `fed_ceiling` limit; K4c the forward's `plan_cap_binding` == the post-hoc binding set.
      ⚠ RESOLUTION NOTE (committed now): 1e-6 m is BELOW float32 spacing for any waypoint beyond ~8 m (1 ulp = 3.8e-6 m at 32-64 m, 1.5e-5 at 128-256 m).
      On a binding row the GPU forward and the CPU post-hoc can differ by 1 ulp and miss 1e-6.  The max error in ULPs is reported beside the verdict and
      `--gpu-posthoc` re-runs the SAME function on the GPU so device numerics are separable from logic.  Nobody may relax the tolerance after seeing the
      number: that is a Master Mind decision recorded as an amendment.
      ⭐ MASTER MIND DECISION, 2026-10-18 ~20:40 Berlin, BEFORE any K4 number exists (no K4 has ever run): a condition PASSES iff every value is finite AND
      (max |d| <= 1e-6 m  OR  max float32 ULP over ALL rows <= ULP_TOL = 4).  A 4-ulp band is <= 6e-5 m even at 256 m, i.e. rounding of the same float32
      arithmetic on two devices; a wrong clamp moves a waypoint by centimetres (the post-hoc S2 shift was 0.063 m mean), five orders of magnitude above it.
      The brief's 1e-6 m (my number, never in A-R8-CLAMP-1, which says only "equals") is kept as the first disjunct so a bit-identical result reads the same.
  CONTROL (optional): the LAUNCH tree (fb47a58, `roll_seed_r8.py`, i.e. the banked code path) rolled on the same prefix vs the banked dump.  If K2 FAILS and the
      control also fails, the ROLL RIG is not bit-reproducible on this box (GPU nondeterminism) and K2 says nothing about the tree; if the control passes and K2
      fails, the TREE changes flag-OFF numerics.
TIER: T1 self-action open loop.  ESTIMATOR: none -- these are deterministic identity checks, not an interval; "n" is windows, not a sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TOL_M = 1e-6
ULP_TOL = 4.0          # Master Mind 2026-10-18 ~20:40, before any K4 data: see the RESOLUTION NOTE in the module docstring
MIN_EPISODES = 3
MIN_BINDING = 5
K4_CONDS = ("nav_true", "nav_shuffled", "nav_flipped", "nav_zero")
WINDOW_LIST_DEPENDENT = ("nav_shuffled.",)
SKIP_KEYS = ("clip_index", "ws")
INPUT_KEYS = ("v0", "v_max_ms", "v_max_valid")
PRIMARY = "nav_true.traj"
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"


# ------------------------------------------------------------------------------------------------ small helpers
def sha12(clip_id: str) -> str:
    return hashlib.sha256(str(clip_id).encode()).hexdigest()[:12]


def load_extras(path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "refcv6_extras.npz"
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def row_index(ex: dict) -> dict:
    """{(clip_index, ws): row} -- the key nr_matched_instrument.load_extras builds (its :88)."""
    key = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(ex["clip_index"], ex["ws"]))}
    if len(key) != len(ex["ws"]):
        raise SystemExit("[k2k4] duplicate (clip_index, ws) rows in an extras file")
    return key


def _rows_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2, b2 = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    if a.dtype.kind == "f":
        return ((a2 == b2) | (np.isnan(a2) & np.isnan(b2))).all(axis=1)
    return (a2 == b2).all(axis=1)


def _max_abs(a: np.ndarray, b: np.ndarray):
    if a.dtype.kind not in "fiu":
        return None
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    d = d[np.isfinite(d)]
    return float(d.max()) if d.size else 0.0


# ------------------------------------------------------------------------------------------------ K2
def k2_compare(banked: dict, new: dict, *, primary: str = PRIMARY, dependent_prefixes=WINDOW_LIST_DEPENDENT,
               min_episodes: int = MIN_EPISODES, expect_no_new_keys: bool = True) -> dict:
    bk, nk = row_index(banked), row_index(new)
    episodes = sorted({c for c, _ in nk})
    missing = [k for k in nk if k not in bk]
    rec: dict = {"n_new_rows": len(nk), "n_banked_rows": len(bk), "n_episodes": len(episodes), "episodes_clip_index": episodes,
                 "windows_absent_from_banked": len(missing), "primary": primary, "tolerance": "bit-for-bit (np.array_equal, NaN==NaN)"}
    if not nk or missing or len(episodes) < min_episodes or primary not in new or primary not in banked:
        rec["verdict"] = "INCONCLUSIVE"
        rec["why"] = (f"rows {len(nk)}, episodes {len(episodes)} (need >= {min_episodes}), windows absent from the banked dump {len(missing)}, "
                      f"primary key present: new={primary in new} banked={primary in banked}")
        return rec
    order = np.array([bk[k] for k in nk])
    shared = [k for k in new if k in banked and k not in SKIP_KEYS]
    rec["only_in_new"] = sorted(k for k in new if k not in banked and k not in SKIP_KEYS)
    rec["only_in_banked"] = sorted(k for k in banked if k not in new and k not in SKIP_KEYS)
    by_key: dict = {}
    for k in shared:
        a, b = banked[k][order], new[k]
        if a.shape != b.shape or a.dtype != b.dtype:
            by_key[k] = {"equal": False, "why": f"shape/dtype banked {a.shape}/{a.dtype} vs new {b.shape}/{b.dtype}"}
            continue
        eq = _rows_equal(a, b)
        by_key[k] = {"equal": bool(eq.all()), "n_rows_unequal": int((~eq).sum()), "max_abs_diff": _max_abs(a, b)}
    p = by_key[primary]
    rec["primary_result"] = p
    dep = [k for k in by_key if any(k.startswith(x) for x in dependent_prefixes)]
    sec = {k: v for k, v in by_key.items() if k != primary and k not in dep}
    rec["secondary_keys_compared"] = len(sec)
    rec["secondary_unequal_keys"] = sorted(k for k, v in sec.items() if not v["equal"])
    rec["secondary_all_equal"] = not rec["secondary_unequal_keys"]
    rec["window_list_dependent_excluded"] = {k: by_key[k] for k in dep}
    rec["by_key"] = by_key
    rec["key_set_note"] = ("OFF mode: no key beyond the banked set expected" if expect_no_new_keys and not rec["only_in_new"] else
                           f"extra keys in the new extras: {rec['only_in_new'][:12]}")
    rec["verdict"] = "PASS" if p["equal"] else "FAIL"
    if not p["equal"]:
        rec["why"] = f"{primary}: {p.get('n_rows_unequal')} of {len(nk)} rows differ, max |d| {p.get('max_abs_diff')}"
    return rec


# ------------------------------------------------------------------------------------------------ K4
def _ulps(a: np.ndarray, b: np.ndarray, rows: np.ndarray):
    if not rows.any():
        return 0.0
    x, y = a[rows].astype(np.float32), b[rows].astype(np.float32)
    sp = np.spacing(np.maximum(np.abs(x), np.abs(y)))
    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(sp > 0, d / sp, 0.0)
    return float(np.nanmax(u)) if u.size else 0.0


def k4_compare(off: dict, on: dict, *, ceiling_fn, clamp_fn, tol: float = TOL_M, conds=K4_CONDS, min_binding: int = MIN_BINDING,
               min_episodes: int = MIN_EPISODES) -> dict:
    """`ceiling_fn(v_max_ms[N], v_max_valid[N]) -> lim float32 [N]` (inf = no ceiling);
    `clamp_fn(plan[N,S,2], lim[N], v0[N]) -> (clamped[N,S,2] float32, binding[N] bool)`.  Both are the POST-HOC route (vendored function)."""
    ko, kn = row_index(off), row_index(on)
    rec: dict = {"tolerance_m": tol, "conds": list(conds), "n_rows": len(ko), "n_episodes": len({c for c, _ in ko})}
    if set(ko) != set(kn) or not ko or rec["n_episodes"] < min_episodes:
        rec["verdict"] = "INCONCLUSIVE"
        rec["why"] = f"OFF rows {len(ko)} / ON rows {len(kn)} / same window set {set(ko) == set(kn)} / episodes {rec['n_episodes']} (need >= {min_episodes})"
        return rec
    oo = np.array([kn[k] for k in ko])
    for k in INPUT_KEYS:
        if k not in off or k not in on or not np.array_equal(off[k], on[k][oo], equal_nan=True):
            rec["verdict"] = "INCONCLUSIVE"
            rec["why"] = f"the per-window input {k!r} differs between the OFF and ON extras (or is absent): the two rolls did not see the same windows"
            return rec
    lim = np.asarray(ceiling_fn(off["v_max_ms"], off["v_max_valid"]), np.float32)
    v0 = off["v0"].astype(np.float32)
    rec["n_rows_finite_limit"] = int(np.isfinite(lim).sum())
    by: dict = {}
    aux = {"K4a_precap_equals_off_plan": {}, "K4b_forward_limit_equals_fed_ceiling": {}, "K4c_forward_binding_equals_posthoc": {}}
    worst, worst_ulp, ok_all, n_bind_ref = 0.0, 0.0, True, None
    for c in conds:
        pk = f"{c}.traj"
        if pk not in off or pk not in on:
            by[c] = {"present": False}
            ok_all = False
            continue
        plan = off[pk]
        out, binding = clamp_fn(plan, lim, v0)
        got = on[pk][oo]
        finite = bool(np.isfinite(got).all() and np.isfinite(out).all())
        d = np.abs(out.astype(np.float64) - got.astype(np.float64))
        mx = float(np.nanmax(d)) if d.size else 0.0
        by[c] = {"present": True, "max_abs_m": mx, "n_rows_over_tol": int((d.reshape(len(d), -1).max(axis=1) > tol).sum()),
                 "n_binding_rows_posthoc": int(binding.sum()), "all_finite": finite,
                 "max_ulp_on_binding_rows": _ulps(out, got, binding), "max_ulp_all_rows": _ulps(out, got, np.ones(len(out), bool))}
        by[c]["pass"] = bool(finite and (mx <= tol or by[c]["max_ulp_all_rows"] <= ULP_TOL))
        ok_all = ok_all and by[c]["pass"]
        worst, worst_ulp = max(worst, mx), max(worst_ulp, by[c]["max_ulp_all_rows"])
        if c == "nav_true":
            n_bind_ref = int(binding.sum())
        pre = on.get(f"{c}.traj_precap")
        if pre is not None:
            aux["K4a_precap_equals_off_plan"][c] = bool(np.array_equal(pre[oo], plan, equal_nan=True))
        fl = on.get(f"{c}.plan_v_lim_ms")
        if fl is not None:
            aux["K4b_forward_limit_equals_fed_ceiling"][c] = bool(np.array_equal(fl[oo], lim, equal_nan=True))
        fb = on.get(f"{c}.plan_cap_binding")
        if fb is not None:
            aux["K4c_forward_binding_equals_posthoc"][c] = bool(np.array_equal(fb[oo] > 0.5, binding))
    rec["by_cond"], rec["aux"] = by, aux
    rec["aux_all_true"] = bool(all(all(v.values()) and bool(v) for v in aux.values()))
    rec["max_abs_m_all"], rec["max_ulp_all"] = worst, worst_ulp
    rec["n_binding_rows_nav_true"] = n_bind_ref
    if n_bind_ref is None or n_bind_ref < min_binding:
        rec["verdict"] = "INCONCLUSIVE"
        rec["why"] = f"only {n_bind_ref} nav_true rows BIND on this prefix (need >= {min_binding}): a K4 in which nothing binds proves nothing"
    else:
        rec["verdict"] = "PASS" if ok_all else "FAIL"
        if not ok_all:
            rec["why"] = (f"max |d| {worst:.3e} m over {len(conds)} conditions (tolerance {tol:g} m or <= {ULP_TOL:g} ulp); max {worst_ulp:.2f} float32 ulp "
                          f"(or a non-finite value / a missing condition): MORE than the float32 resolution band, so not device rounding.")
    rec["ulp_tol"] = ULP_TOL
    return rec


# ------------------------------------------------------------------------------------------------ the post-hoc route (vendored function)
def default_posthoc(device: str = "cpu"):
    """-> (ceiling_fn, clamp_fn, record).  Uses clamp_posthoc_r8's own loader/ceiling/apply (the vendored LANDED function, blob-verified)."""
    import clamp_posthoc_r8 as CP
    import torch
    V, vrec = CP.load_vendored()
    rc = CP.run_cfg()

    def ceiling_fn(vmax, valid):
        lim, _ = CP.fed_ceiling(vmax, valid, rc)
        return lim.numpy()

    def clamp_fn(plan, lim, v0):
        t = torch.from_numpy(plan.astype(np.float32)).to(device)
        out, diag = V.cap_plan_to_limit(t, torch.from_numpy(np.asarray(lim, np.float32)).to(device), horizons=rc["horizons"], tick_s=rc["tick_s"],
                                        v0=torch.from_numpy(np.asarray(v0, np.float32)).to(device), brake_ms2=float(rc["brake_ms2"]))
        return out.cpu().numpy(), diag["plan_cap_binding"].cpu().numpy().astype(bool)

    return ceiling_fn, clamp_fn, {"device": device, "vendored": vrec, "horizons": rc["horizons"], "tick_s": rc["tick_s"], "brake_ms2": rc["brake_ms2"]}


# ------------------------------------------------------------------------------------------------ the prefix planner
def plan_prefix(banked_dump: str, windows_json: str, n_episodes: int | None = None, *, min_episodes: int = MIN_EPISODES,
                min_binding: int = 10, max_episodes: int = 15, posthoc=None) -> tuple[dict, dict]:
    """-> (subset windows {clip_id: [ws]}, plan record).  The subset is the first K episodes of the banked dump IN DUMP ORDER.

    K is `n_episodes` when given, else the smallest K >= `min_episodes` whose prefix holds >= `min_binding` rows that the post-hoc clamp BINDS on
    (so K4 is not vacuous), capped at `max_episodes`.  Uses only the banked OFF output + the committed K4 rule; no GPU number is involved."""
    root = Path(banked_dump)
    ex = load_extras(root / "dump_s0")
    man = json.loads((root / "dump_s0" / "manifest.json").read_text(encoding="utf-8"))
    sh = json.loads((root / "dump_s0" / "refcv6_extras.npz.json").read_text(encoding="utf-8"))["clip_sha12_by_index"]
    allw = json.loads(Path(windows_json).read_text(encoding="utf-8"))
    idxs = sorted({int(c) for c in ex["clip_index"]})
    by_sha = {sha12(e["clip_id"]): e["clip_id"] for e in man["episodes"]}
    ep_rows = []
    for ci in idxs[:max_episodes]:
        want = sh.get(str(ci))
        cid = by_sha.get(want)
        if not want or cid is None:
            raise SystemExit(f"[k2k4 plan] clip_index {ci}: sha12 {want!r} matches no manifest episode -- the index<->clip map is not provable")
        rows = np.nonzero(ex["clip_index"] == ci)[0]
        ws = sorted(int(x) for x in ex["ws"][rows])
        if allw.get(cid) is None or sorted(int(x) for x in allw[cid]) != ws:
            raise SystemExit(f"[k2k4 plan] {cid}: the banked windows are not the windows-json grid for this clip")
        ep_rows.append({"clip_index": ci, "clip_id": cid, "sha12": want, "n_windows": len(ws), "ws": ws, "rows": rows})
    binding_rows = None
    if posthoc is None and n_episodes is None:
        posthoc = default_posthoc()
    if posthoc is not None:
        ceiling_fn, clamp_fn = posthoc[0], posthoc[1]
        lim = np.asarray(ceiling_fn(ex["v_max_ms"], ex["v_max_valid"]), np.float32)
        _, bnd = clamp_fn(ex["nav_true.traj"], lim, ex["v0"].astype(np.float32))
        binding_rows = bnd
        for e in ep_rows:
            e["n_binding_posthoc_nav_true"] = int(bnd[e["rows"]].sum())
    k, chosen_by = None, None
    if n_episodes is not None:
        k, chosen_by = int(n_episodes), "explicit --n-episodes"
        if k < min_episodes:
            raise SystemExit(f"[k2k4 plan] --n-episodes {k} < {min_episodes}")
    else:
        cum = 0
        for j, e in enumerate(ep_rows, 1):
            cum += e["n_binding_posthoc_nav_true"]
            if j >= min_episodes and cum >= min_binding:
                k, chosen_by = j, f"smallest prefix >= {min_episodes} episodes with >= {min_binding} post-hoc-binding nav_true rows"
                break
        if k is None:
            k, chosen_by = len(ep_rows), f"CAP {max_episodes}: fewer than {min_binding} binding rows were found (K4 may be INCONCLUSIVE)"
    keep = ep_rows[:k]
    sub = {e["clip_id"]: e["ws"] for e in keep}
    rec = {"banked_dump": str(root), "n_episodes": k, "chosen_by": chosen_by, "n_windows": int(sum(e["n_windows"] for e in keep)),
           "episodes": [{kk: vv for kk, vv in e.items() if kk not in ("ws", "rows")} for e in keep],
           "n_binding_posthoc_nav_true_in_prefix": (int(sum(e.get("n_binding_posthoc_nav_true", 0) for e in keep)) if binding_rows is not None else None),
           "prefix_rule": "first K episodes in DUMP ORDER, rolled from window 0 with the full arm set: refcv3_arm seeds once (refcv3_arm.py:1999), so only a prefix sees the same DDIM stream",
           "ws_is": "provider frame index of the window origin (refcv3_arm manifest 'ws_is')"}
    return sub, rec


def _atomic_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def overall(k2: dict, k4: dict, control: dict | None = None) -> dict:
    v2, v4 = k2.get("verdict"), k4.get("verdict")
    if v2 == "PASS" and v4 == "PASS":
        o = "PASS: the flag-OFF clamp tree reproduces the banked dump bit-for-bit AND the forward clamp equals the post-hoc clamp -- clamp-ON GPU rolls are admissible"
    elif "FAIL" in (v2, v4):
        o = "FAIL: " + "; ".join(x for x in (f"K2 {v2}" if v2 != "PASS" else "", f"K4 {v4}" if v4 != "PASS" else "") if x) + " -- no clamp-ON number is admissible"
    else:
        o = f"INCONCLUSIVE (K2 {v2}, K4 {v4}) -- not admissible"
    out = {"verdict": o.split(":")[0].split(" ")[0], "text": o}
    if control is not None:
        out["control_launch_tree_vs_banked"] = control.get("verdict")
        if v2 == "FAIL":
            out["k2_fail_reading"] = ("ROLL RIG NOT BIT-REPRODUCIBLE (the fb47a58 control also fails): K2 does not indict the tree" if control.get("verdict") == "FAIL"
                                      else "the rig reproduces the banked dump with the launch tree, so the clamp tree CHANGES flag-OFF numerics" if control.get("verdict") == "PASS"
                                      else "control inconclusive")
    return out


def _cmd_plan(a) -> int:
    sub, rec = plan_prefix(a.banked, a.windows_json, a.n_episodes)
    _atomic_json(sub, a.out_windows)
    _atomic_json(rec, a.out_plan)
    print(json.dumps({"n_episodes": rec["n_episodes"], "n_windows": rec["n_windows"], "chosen_by": rec["chosen_by"],
                      "n_binding_posthoc": rec["n_binding_posthoc_nav_true_in_prefix"]}))
    return 0


def _cmd_compare(a) -> int:
    banked = load_extras(Path(a.banked) / "dump_s0")
    off, on = load_extras(a.off_dump), load_extras(a.on_dump)
    res: dict = {"schema": "r8-k2k4/1", "step": a.step, "created": time.strftime("%FT%T%z"), "tier": TIER,
                 "estimator": "none: deterministic identity checks on windows (no interval, no sample)", "banked": str(a.banked),
                 "off_dump": str(a.off_dump), "on_dump": str(a.on_dump), "tolerance_m": TOL_M}
    res["K2"] = k2_compare(banked, off)
    ceiling_fn, clamp_fn, prec = default_posthoc("cpu")
    res["posthoc_cpu"] = prec
    res["K4"] = k4_compare(off, on, ceiling_fn=ceiling_fn, clamp_fn=clamp_fn)
    res["K2_on_precap_vs_banked"] = None
    if "nav_true.traj_precap" in on:
        bk, nk = row_index(banked), row_index(on)
        if all(k in bk for k in nk):
            o = np.array([bk[k] for k in nk])
            res["K2_on_precap_vs_banked"] = {"nav_true_traj_precap_equals_banked_nav_true_traj": bool(np.array_equal(on["nav_true.traj_precap"], banked["nav_true.traj"][o]))}
    if a.gpu_posthoc:
        try:
            import torch
            if torch.cuda.is_available():
                c2, f2, p2 = default_posthoc("cuda")
                res["K4_gpu_posthoc"] = k4_compare(off, on, ceiling_fn=c2, clamp_fn=f2)
                res["K4_gpu_posthoc"]["note"] = "the SAME vendored function on the GPU: separates device numerics from logic (diagnostic; does not replace K4)"
            else:
                res["K4_gpu_posthoc"] = {"verdict": "NOT RUN", "why": "no CUDA device visible"}
        except Exception as exc:                                  # noqa: BLE001 -- recorded
            res["K4_gpu_posthoc"] = {"verdict": "NOT RUN", "why": f"{type(exc).__name__}: {str(exc)[:200]}"}
    ctl = None
    if a.control_dump:
        ctl = k2_compare(banked, load_extras(a.control_dump))
        res["CONTROL_launch_tree_flag_free"] = ctl
    res["OVERALL"] = overall(res["K2"], res["K4"], ctl)
    _atomic_json(res, a.out)
    print(json.dumps({"K2": res["K2"]["verdict"], "K4": res["K4"]["verdict"], "OVERALL": res["OVERALL"]["verdict"], "out": str(a.out)}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--banked", required=True)
    p.add_argument("--windows-json", required=True)
    p.add_argument("--out-windows", required=True)
    p.add_argument("--out-plan", required=True)
    p.add_argument("--n-episodes", type=int, default=None)
    c = sub.add_parser("compare")
    c.add_argument("--banked", required=True)
    c.add_argument("--off-dump", required=True)
    c.add_argument("--on-dump", required=True)
    c.add_argument("--control-dump", default=None)
    c.add_argument("--out", required=True)
    c.add_argument("--step", default="")
    c.add_argument("--gpu-posthoc", action="store_true")
    a = ap.parse_args(argv)
    return _cmd_plan(a) if a.cmd == "plan" else _cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
