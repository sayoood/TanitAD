"""deploy8_forward_r8.py -- the 8-SLOT DEPLOYED re-timing stage (SHIP-R8 AMENDMENT 2) scored on one eval139 tree.  CPU only, zero GPU.

NEW glue module.  `a0k0_headroom_retime_r8.py` (RT), `retime_trainfit_r8.py` (TFM) and `compliant_reread_r8.py` (CR) are imported and NOT edited; RT.forward
does ALL the scoring (its own bootstrap, bars, NR / LAT machinery); only the positions of the ARMS are replaced, via four monkeypatches (tools_load,
surface_inputs, frozen_weights, retime_apply + retime_diag) -- the same device `retime_trainfit_r8.cmd_forward` and `deploy_forward_r8.py` use.

    python deploy8_forward_r8.py --tree step30000ema_unc                  # -> raw/retime_deploy/FORWARD_DEPLOY8_<tree>.json + DEPLOY8_POS_<tree>_s{0,1}.npz
    python deploy8_forward_r8.py --tree step30000ema_unc --mutate-knot 1:0.01   # deliberate regression: the per-surface identity control MUST go RED
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH='<stack overlay with tanitad/retime_stage.py>;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' python -X utf8 deploy8_forward_r8.py ...

THE FUNCTION (AMENDMENT 2): ONE call per window on all 8 model slots (0.5, 1, 1.5, 2, 3, 4, 5, 6 s) of the model's own plan `out["traj"]` (= the dump's `nav_true.traj`,
== the decisions' `plan_full_nav_true`), against the prior `ha0_ext_pose` rolled at the same 8 slots, with Amendment 1's table (`tanitad.retime_stage`,
configs/refcv8_retime_deploy_gains.json).  Each surface is scored on that call's outputs AT ITS OWN INSTANTS: S2 = slots [0,1,2,3], S6 = slots [1,3,4,5,6,7].

ARMS
  RETIME_DEPLOY8            the condition's arm: the 8-slot call, indexed at the surface's slots
  RETIME_DEPLOY_PERSURFACE  Amendment 1's table on the PER-SURFACE geometry (RT.retime_apply on the surface's own 4 / 6 points) -- beside, not in the condition
  CONST_G1                  the 8-slot call with g == 1 everywhere: the known-value control, must equal os EXACTLY (zero-width CI on every row)
CONTROLS (any failure WITHHOLDS: no FORWARD_DEPLOY8 / npz is left behind, WITHHELD_DEPLOY8_<tree>.json is written, rc 4)
  (a) RT.forward's own controls (decisions / validation / os-cell-equals-banked / on-path -- the on-path of the 8-slot arm is measured against the 8-point os polyline)
  (b) CONST_G1 - os == 0.0 with a zero-width CI on every METRICS row, both sets, both seeds, both surfaces
  (c) 8-slot array known values: g == 1 bit-equal plan8; g == 0 equal to TFM.walker_positions (an independent loop-based walker) on all 8 slots
  (d) THE SCORING-PATH IDENTITY: RETIME_DEPLOY_PERSURFACE's S2 rows equal FORWARD_TRAINFIT_<tree>.json's RETIME_TRAINFIT S2 rows EXACTLY (an identity: the deployed S2
      table IS the train S2 table), so the arm path scores exactly like the registered one.  MUTATION (--mutate-knot): perturbing a knot must break it.
  (e) row alignment: the panel `os` == plan8[:, slots] bit for bit; the dump's `nav_true.traj` == the decisions' `plan_full_nav_true`; the 8-slot prior at the surface's
      slots == RT's HP bit for bit; the model slot horizons are consistent across surfaces
  (f) the deploy table equals the frozen train gains at its knots (read from the frozen file, not from the stage's own sidecar) and verifies its sha256 sidecar
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
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
import retime_trainfit_r8 as TFM  # noqa: E402

PKG = HERE.parent
DEP = PKG / "raw" / "retime_deploy"
TF = PKG / "raw" / "retime_trainfit"
SEEDS = (0, 1)
SURFS = ("S2", "S6")
SETS = ("compliant", "all")
ARMS = ("RETIME_DEPLOY8", "RETIME_DEPLOY_PERSURFACE", "CONST_G1")
HZ8 = [5, 10, 15, 20, 30, 40, 50, 60]                         # the model's anchor_horizons (ticks of 0.1 s)
T8 = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])      # the 8 slot instants, as literals
SLOTS = {"S2": [0, 1, 2, 3], "S6": [1, 3, 4, 5, 6, 7]}
FROZEN_SHA = "b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee"
# dev-box fallbacks, in order: the build overlay (byte-identical to the landed stack/tanitad/retime_stage.py, git blob 62a847bc), then the original hold (consumed on landing)
STAGE_FALLBACKS = (Path("C:/lgs/retime_stage_scratch/stack/tanitad/retime_stage.py"), Path("C:/Users/Admin/qland/hold_retimestage/stage/stack/tanitad/retime_stage.py"))
CONTRACT = (
    "A consumer selects a surface's RETIME_DEPLOY8 positions like this.  Open DEPLOY8_POS_<tree>_s<seed>.npz (one file per INFERENCE seed).  For surface S in {S2, S6}: "
    "`pos = z[f'{S}_pos']` is [n, T, 2] float32 -- the re-timed plan at the surface's instants (`{S}_instants_s`; model slots `{S}_slots`), i.e. the 8-slot call's output "
    "indexed at those slots, cast to float32 exactly as RT.retime_apply casts.  Rows are ALL windows of the surface in RT's `surface_inputs` order (the same order as the "
    "panel's `os`, which is `{S}_os`, and as `RT.surface_inputs(...)['X']['O']`); `{S}_clip`, `{S}_ws`, `{S}_eid` are the clip index, window origin (provider frame) and "
    "episode id of each row.  The COMPLIANT windows are `sel = z[f'{S}_compliant']` (bool [n]; RT's `S['mask']`, = `ctx['comp'][rows]`): `pos[sel]` is what RT scores on the "
    "`compliant` set, `pos` the `all` set.  `{S}_pos8` is the full [n, 8, 2] output (S6's rows are a subset of S2's windows but stored per surface), `{S}_plan8` / `{S}_prior8` "
    "are the call's inputs, `{S}_pos64` the float64 result before the cast.  To score an arm like RT does: `comp = PR.components(ra, pos[sel], G[sel], dt)` and "
    "`PR.paired_cell(...)` with episode ids `{S}_eid[sel]`."
)


def _load_stage():
    """The runtime stage, from (in order) $RETIME_STAGE_FILE, the landed `tanitad.retime_stage`, the dev-box fallbacks.  Its path and sha256 are recorded."""
    cands = []
    if os.environ.get("RETIME_STAGE_FILE"):
        cands.append(Path(os.environ["RETIME_STAGE_FILE"]))
    try:
        sp = importlib.util.find_spec("tanitad.retime_stage")
        if sp and sp.origin:
            cands.append(Path(sp.origin))
    except Exception:
        pass
    cands.extend(STAGE_FALLBACKS)
    for c in cands:
        if c.is_file():
            spec = importlib.util.spec_from_file_location("_retime_stage_d8", c)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, c
    raise SystemExit("[deploy8] retime_stage.py not found (set RETIME_STAGE_FILE)")


RS, RS_FILE = _load_stage()


# ------------------------------------------------------------------------------------------ pure helpers (pinned by the pytest)
def exact_equal(a, b, path: str = "") -> list:
    """Deep EXACT comparison (== on floats, NaN == NaN).  Returns the list of mismatching paths ([] = identical).  Dict key sets must match."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: key only on one side")
            else:
                out += exact_equal(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += exact_equal(x, y, f"{path}[{i}]")
        return out
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def count_leaves(x) -> int:
    if isinstance(x, dict):
        return sum(count_leaves(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(count_leaves(v) for v in x)
    return 1


def s2_identity(res: dict | None, ref: dict | None, arm: str = "RETIME_DEPLOY_PERSURFACE", ref_arm: str = "RETIME_TRAINFIT") -> dict:
    """`arm`'s S2 rows (both inference seeds x both sets) EXACTLY equal `ref_arm`'s S2 rows of `ref`.  An absent side is status ABSENT, pass False."""
    if res is None or ref is None:
        return {"status": "ABSENT", "pass": False, "n_leaves_compared": 0, "mismatches": ["the reference or the result is absent"]}
    mism, n = [], 0
    for s in SEEDS:
        for st in SETS:
            try:
                a = res["seeds"][str(s)]["surfaces"]["S2"]["sets"][st]["arms"][arm]
                b = ref["seeds"][str(s)]["surfaces"]["S2"]["sets"][st]["arms"][ref_arm]
            except KeyError as e:
                mism.append(f"seed {s} {st}: missing {e}")
                continue
            mism += [f"s{s}|{st}{m}" for m in exact_equal(a, b)]
            n += count_leaves(a)
    return {"status": "COMPARED", "pass": bool(not mism and n > 0), "n_leaves_compared": n, "n_mismatches": len(mism), "mismatches": mism[:20]}


def mutate_table(table: dict, idx: int, delta: float) -> dict:
    """DELIBERATE REGRESSION: add `delta` to one knot gain."""
    t = {"knots_s": np.array(table["knots_s"], np.float64), "gains": np.array(table["gains"], np.float64)}
    t["gains"][idx] += delta
    return t


def aggregate_controls(c: dict) -> bool:
    need = ("rt_forward", "const_g1_exactly_os", "known_values_8slot", "persurface_identity_vs_forward_trainfit", "row_alignment",
            "table_matches_frozen_train")
    return all(c.get(k) is True for k in need)


class _Rows(np.ndarray):
    """A gain 'vector' that is really a ROW INDEX [n, T]: RT.forward hands arms' gains through `W[sel]`, so this carries which rows of the surface were selected,
    and the patched `retime_apply` / `retime_diag` return the pre-computed 8-slot positions of exactly those rows.  The subclass IS the marker."""


class _RowsD8(_Rows):
    KIND = "D8"


class _RowsG1(_Rows):
    KIND = "G1"


def rows_gain(n: int, T: int, cls):
    return np.repeat(np.arange(n, dtype=np.int64)[:, None], T, axis=1).view(cls)


def diag8(plan8, prior8, pos64_8, pos32_8, g8, cast_dtype) -> dict:
    """RT.retime_diag's schema for the 8-slot function: geometry of the re-timed plan against the 8-POINT os polyline."""
    geo = RT.polyline_geometry(plan8)
    d = RT.dist_to_path(pos64_8, geo)
    d32 = RT.dist_to_path(pos32_8.astype(np.float64), geo)
    L, M = geo["L"][:, 1:], RT.prior_arc(prior8)
    s = RT.retime_progress(L, M, g8)
    LS = geo["L"][:, -1:]
    bound32 = float(np.finfo(cast_dtype).eps * np.abs(pos64_8).max()) if np.dtype(cast_dtype).kind == "f" else float("nan")
    Gb = np.broadcast_to(np.asarray(g8, np.float64), s.shape)
    return {"n_windows": int(len(s)), "max_dist_to_path_f64_m": float(d.max()), "max_dist_to_path_after_cast_m": float(d32.max()), "float32_rounding_bound_m": bound32,
            "frac_nonmonotone_windows": float((np.diff(s, axis=1) < -1e-12).any(1).mean()), "frac_extrapolated_points": float((s > LS + 1e-9).mean()),
            "frac_fallback_dir_windows": float(geo["fallback"].mean()), "max_abs_g_minus_1": float(np.abs(Gb - 1.0).max()), "mean_abs_g_minus_1": float(np.abs(Gb - 1.0).mean()),
            "geometry_over": "the 8-slot call: re-timed positions at all 8 slots against the 8-point os polyline"}


# ------------------------------------------------------------------------------------------ the run
def forward(tree: str, mutate: tuple | None = None) -> int:
    t_start = time.time()
    train_path = TF / "FROZEN_GAINS_RETIME_train.json"
    train_fz = json.load(open(train_path, encoding="utf-8"))
    TFM.assert_frozen_is_train(train_fz)
    if TFM.sha256_file(train_path) != FROZEN_SHA or (TF / "FROZEN_GAINS_RETIME_train.sha256").read_text().strip() != FROZEN_SHA:
        raise SystemExit("[deploy8] REFUSING: the frozen train gains do not hash to their pinned sha256")
    table = RS.load_deploy_table()                                    # sha-verified (sidecar)
    stage_meta = {"stage_file": str(RS_FILE), "stage_sha256": hashlib.sha256(Path(RS_FILE).read_bytes()).hexdigest(), "table_file": str(RS.DEPLOY_TABLE_PATH),
                  "table_sha256": table["sha256"]}
    ctl_table = {}
    k_all = [(f"S2.RETIME[{i}]", train_fz["surfaces"]["S2"]["all"]["RETIME"][i]) for i in range(4)] + \
            [(f"S6.RETIME[{i}]", train_fz["surfaces"]["S6"]["all"]["RETIME"][i]) for i in range(2, 6)]
    ctl_table["knots_equal_frozen_train_exactly"] = bool(table["knots_s"].tolist() == T8.tolist() and table["gains"].tolist() == [v for _n, v in k_all])
    ctl_table["knot_sources"] = [n for n, _v in k_all]
    sib = DEP / "FROZEN_GAINS_RETIME_deploy.json"
    if sib.exists():                                                   # informational: the sibling's per-surface derivation of the same table
        sd = json.load(open(sib, encoding="utf-8"))
        try:
            ctl_table["info_agrees_with_FROZEN_GAINS_RETIME_deploy"] = bool(all(
                np.allclose(RS.gain_at(sd["surfaces"][s]["instants_s"], table), sd["surfaces"][s]["RETIME_DEPLOY"], atol=1e-12, rtol=0) for s in SURFS))
        except Exception as e:  # noqa: BLE001
            ctl_table["info_agrees_with_FROZEN_GAINS_RETIME_deploy"] = f"UNREADABLE: {type(e).__name__}"
    if mutate:
        table = mutate_table(table, mutate[0], mutate[1])
    work = DEP / ("mutation8" if mutate else "_d8_work")
    work.mkdir(parents=True, exist_ok=True)
    RT.FROZEN_JSON, RT.OUT = train_path, work
    import compliant_reread_r8 as CR
    import torch
    from tanitad.models import kinematic_prior as KP
    from tanitad.models.ego_history import ego_channels_from_poses

    cache, cur = {}, {"seed": None, "surf": None}
    stash, kv, align, store = {}, {}, {}, {}
    orig_tl, orig_si, orig_ps = RT.tools_load, RT.surface_inputs, CR.prepare_seed
    orig_apply, orig_diag = RT.retime_apply, RT.retime_diag
    const1, const0 = {"knots_s": [0.5], "gains": [1.0]}, {"knots_s": [0.5], "gains": [0.0]}

    def tools_load_once():
        if "t" not in cache:
            cache["t"] = orig_tl()
        return cache["t"]

    def prepare_seed_rec(tree_, seed, tools, rc):
        cur["seed"] = int(seed)
        return orig_ps(tree_, seed, tools, rc)

    def surface_inputs_hook(tools, ctx, surf, first, tree_):
        S = orig_si(tools, ctx, surf, first, tree_)
        seed, P = cur["seed"], ctx["panels"][surf]
        rows, pdir = P["rows"], S["dir"]
        grid = json.load(open(os.path.join(pdir, "manifest.json"), encoding="utf-8"))["grid"]
        slots, hz = [int(x) for x in grid["slots"]], [int(x) for x in grid["horizons_steps"]]
        inst = np.asarray(grid["instants_s"], np.float64)
        a = {"slots_equal_expected": slots == SLOTS[surf], "slot_horizons_consistent_with_model": [HZ8[i] for i in slots] == hz,
             "instants_equal_T8_at_slots": bool(np.array_equal(inst, T8[slots]))}
        plan8 = np.ascontiguousarray(ctx["plan"][rows])                 # the dump's nav_true.traj, [n, 8, 2] float32
        _m, files, _e = tools["NR"].panel_layout(pdir)
        dd = [np.load(os.path.join(pdir, "decisions", os.path.basename(f))) for f in files]
        dec_plan = np.concatenate([d["plan_full_nav_true"] for d in dd])
        dec_ws = np.concatenate([d["ws"] for d in dd]).astype(np.int64)
        a["dump_traj_eq_decisions_plan_full_nav_true"] = bool(plan8.shape == dec_plan.shape and np.array_equal(plan8, dec_plan))
        a["decisions_ws_eq_rows_ws"] = bool(np.array_equal(dec_ws, S["ws"]))
        a["panel_os_eq_plan8_at_slots"] = bool(np.array_equal(S["X"]["O"], plan8[:, slots]))
        pw = torch.from_numpy(np.ascontiguousarray(S["dec"]["win"], dtype=np.float32))
        a0, kp = KP.prior_controls("ha0_ext_pose", pw)
        v0 = ego_channels_from_poses(pw, RT.W_OBS, dt=RT.DT)[:, -1, 0]
        prior8 = KP.prior_path(a0, kp, v0, HZ8).numpy().astype(np.float32)
        a["prior8_at_slots_eq_RT_HP"] = bool(np.array_equal(prior8[:, slots], S["X"]["HP"]))
        pos64 = RS.retime_plan(plan8, prior8, T8, table, cast=False)
        pos32 = RS.retime_plan(plan8, prior8, T8, table, cast=True)
        g1_64, g1_32 = RS.retime_plan(plan8, prior8, T8, const1, cast=False), RS.retime_plan(plan8, prior8, T8, const1, cast=True)
        g0 = RS.retime_plan(plan8, prior8, T8, const0, cast=False)
        kv[f"{surf}|s{seed}"] = {"n_windows": int(len(plan8)), "g1_bit_equal_plan8": bool(g1_32.tobytes() == plan8.tobytes()),
                                 "g1_max_abs_m": float(np.max(np.abs(g1_32.astype(np.float64) - plan8.astype(np.float64)))),
                                 "g0_vs_walker_max_abs_m": float(np.max(np.abs(g0 - TFM.walker_positions(plan8, prior8))))}
        align[f"{surf}|s{seed}"] = a
        g8 = RS.gain_at(T8, table)
        stash[(seed, surf)] = {"D8": (pos32, pos64), "G1": (g1_32, g1_64), "plan8": plan8, "prior8": prior8, "g8": g8, "slots": slots}
        cur["surf"] = surf
        store[(seed, surf)] = {"os": S["X"]["O"], "plan8": plan8, "prior8": prior8, "pos8": pos32, "pos64_8": pos64, "clip": S["clip"], "ws": S["ws"], "eid": np.asarray(S["eid"]),
                               "compliant": np.asarray(S["mask"], bool), "slots": np.asarray(slots), "instants": inst, "rows": np.asarray(rows)}
        return S

    def frozen_weights_d8(_fz, surf, _v0):
        n = len(stash[(cur["seed"], surf)]["plan8"])
        T = len(SLOTS[surf])
        t_s = np.array({"S2": [0.5, 1.0, 1.5, 2.0], "S6": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}[surf])
        return {"RETIME_DEPLOY8": rows_gain(n, T, _RowsD8), "RETIME_DEPLOY_PERSURFACE": RS.gain_at(t_s, table), "CONST_G1": rows_gain(n, T, _RowsG1)}

    def retime_apply_d8(O, B, Gn, cast=True):
        if isinstance(Gn, _Rows):
            st = stash[(cur["seed"], cur["surf"])]
            rows = np.asarray(Gn)[:, 0].astype(np.int64)
            pos = st[Gn.KIND][0 if cast else 1][:, st["slots"]][rows]
            if not np.array_equal(np.asarray(O), np.asarray(store[(cur["seed"], cur["surf"])]["os"])[rows]):
                raise SystemExit("[deploy8] the rows RT.forward selected do not carry the os the 8-slot positions were built from (row misalignment)")
            return pos
        return orig_apply(O, B, Gn, cast)

    def retime_diag_d8(O, B, Gn):
        if isinstance(Gn, _Rows):
            st = stash[(cur["seed"], cur["surf"])]
            rows = np.asarray(Gn)[:, 0].astype(np.int64)
            g8 = st["g8"] if Gn.KIND == "D8" else np.ones(8)
            return diag8(st["plan8"][rows], st["prior8"][rows], st[Gn.KIND][1][rows], st[Gn.KIND][0][rows], g8, st["plan8"].dtype)
        return orig_diag(O, B, Gn)

    RT.tools_load, RT.surface_inputs, RT.frozen_weights = tools_load_once, surface_inputs_hook, frozen_weights_d8
    RT.retime_apply, RT.retime_diag, CR.prepare_seed = retime_apply_d8, retime_diag_d8, prepare_seed_rec
    try:
        rc = RT.forward(tree)
    finally:
        RT.retime_apply, RT.retime_diag, CR.prepare_seed = orig_apply, orig_diag, orig_ps
    p = work / f"FORWARD_RETIME_{tree}.json"
    res = json.load(open(p, encoding="utf-8"))
    os.remove(p)

    controls = {"rt_forward": bool(res["controls"]["ALL_PASS"] and rc == 0), "table_matches_frozen_train": bool(ctl_table["knots_equal_frozen_train_exactly"]), "table": ctl_table}
    try:
        controls["const_g1_cell_rows_checked_exactly_zero"] = TFM.assert_const_g1_cells(res)
        controls["const_g1_exactly_os"] = True
    except SystemExit as e:
        controls["const_g1_exactly_os"], controls["const_g1_error"] = False, str(e)
    bad = [f"{k}: g1_bit_equal={v['g1_bit_equal_plan8']} g1_max={v['g1_max_abs_m']:.3g} g0_walker_max={v['g0_vs_walker_max_abs_m']:.3g}" for k, v in kv.items()
           if not (v["g1_bit_equal_plan8"] and v["g1_max_abs_m"] == 0.0 and v["g0_vs_walker_max_abs_m"] <= 1e-6)]
    controls["known_values_8slot"] = bool(kv and not bad)
    controls["known_value_arrays_8slot"] = kv
    if bad:
        controls["known_values_error"] = "; ".join(bad)
    controls["row_alignment"] = bool(align and all(all(v for v in a.values()) for a in align.values()))
    controls["row_alignment_detail"] = align
    tfp = TF / f"FORWARD_TRAINFIT_{tree}.json"
    ref = json.load(open(tfp, encoding="utf-8")) if tfp.exists() else None
    ident = s2_identity(res, ref)
    ident["reference"] = f"raw/retime_trainfit/FORWARD_TRAINFIT_{tree}.json arm RETIME_TRAINFIT, S2, seeds 0 and 1, compliant and all-window sets"
    ident["reference_controls_ALL_PASS"] = (ref or {}).get("controls", {}).get("ALL_PASS")
    controls["persurface_identity_vs_forward_trainfit"] = bool(ident["pass"] and ident["reference_controls_ALL_PASS"] is True)
    controls["persurface_identity_vs_forward_trainfit_detail"] = ident
    controls["ALL_PASS"] = aggregate_controls(controls)
    d8 = {"differs_from_persurface": {s: bool(exact_equal(res["seeds"]["0"]["surfaces"][s]["sets"]["compliant"]["arms"]["RETIME_DEPLOY8"],
                                                           res["seeds"]["0"]["surfaces"][s]["sets"]["compliant"]["arms"]["RETIME_DEPLOY_PERSURFACE"])) for s in SURFS}}
    res["controls_rt"] = res["controls"]
    res["controls"] = controls
    res.update({"tool": "deploy8_forward_r8.py", "tree_name": tree, "n_boot": RT.N_BOOT, "bootstrap_seed": 0,
                "role": ("MUTATION RUN (deliberate regression; NOT a result)" if mutate else
                         "SHIP-R8 AMENDMENT 2: the DEPLOYED 8-slot stage RETIME_DEPLOY8 (Amendment 1 table, frozen TRAIN gains sha256 b208c9c1... UNCHANGED), scored on eval139 " + tree),
                "mutation": ({"knot_index": mutate[0], "delta": mutate[1]} if mutate else None),
                "deploy8": {"function": "ONE call per window on all 8 model slots; prior ha0_ext_pose at the same 8 slots; each surface scored on the call's outputs at its own instants",
                            "model_slot_horizons_ticks": HZ8, "slot_instants_s": T8.tolist(), "surface_slots": SLOTS, "deploy_table_knots_s": T8.tolist(),
                            "deploy_table_gains": [float(x) for x in table["gains"]], **stage_meta, "consumer_contract": CONTRACT, **d8},
                "frozen_train_gains_sha256": FROZEN_SHA,
                "arm_names": "RETIME_DEPLOY8 = the condition's arm; RETIME_DEPLOY_PERSURFACE = Amendment 1 table on per-surface geometry (beside); CONST_G1 = 8-slot known-value control",
                "script_sha256": TFM.sha256_file(__file__), "wall_s": round(time.time() - t_start, 1), "generated": time.strftime("%FT%T")})
    if mutate:
        mp = work / f"MUTATION8_{tree}.json"
        RT.jdump({"tree": tree, "mutation": res["mutation"], "persurface_identity_vs_forward_trainfit_pass": controls["persurface_identity_vs_forward_trainfit"],
                  "n_mismatches_vs_forward_trainfit": ident.get("n_mismatches"), "first_mismatches": ident.get("mismatches", [])[:5], "ALL_PASS": controls["ALL_PASS"],
                  "mutation_detected": bool(not controls["persurface_identity_vs_forward_trainfit"] and not controls["ALL_PASS"]), "generated": res["generated"]}, mp)
        print(f"MUTATION {mp} detected={not controls['ALL_PASS']} (identity_vs_trainfit={controls['persurface_identity_vs_forward_trainfit']})", flush=True)
        print(f"ZZDEP8MUT_{'RED' if not controls['ALL_PASS'] else 'GREEN'}ZZ {tree}", flush=True)
        return 0 if not controls["ALL_PASS"] else 5
    out, wh = DEP / f"FORWARD_DEPLOY8_{tree}.json", DEP / f"WITHHELD_DEPLOY8_{tree}.json"
    if not controls["ALL_PASS"]:
        RT.jdump({"tree": tree, "WITHHELD": {k: v for k, v in controls.items() if v is False or k.endswith("_error") or k.endswith("_detail")}, "generated": res["generated"]}, wh)
        if out.exists():
            os.remove(out)
        print(f"WITHHELD: a control did not pass -> {wh}\nZZDEP8WITHHELDZZ {tree}", flush=True)
        return 4
    if wh.exists():
        os.remove(wh)
    for seed in SEEDS:                                                   # the npz: positions per inference seed, self-describing
        arrs = {}
        for surf in SURFS:
            s = store[(seed, surf)]
            arrs.update({f"{surf}_pos": s["pos8"][:, s["slots"]], f"{surf}_pos64": s["pos64_8"][:, s["slots"]], f"{surf}_pos8": s["pos8"], f"{surf}_os": s["os"],
                         f"{surf}_plan8": s["plan8"], f"{surf}_prior8": s["prior8"], f"{surf}_clip": s["clip"], f"{surf}_ws": s["ws"], f"{surf}_eid": s["eid"],
                         f"{surf}_compliant": s["compliant"], f"{surf}_slots": s["slots"], f"{surf}_instants_s": s["instants"], f"{surf}_rows": s["rows"]})
        meta = {"tool": "deploy8_forward_r8.py", "tree": tree, "inference_seed": seed, "arm": "RETIME_DEPLOY8", "tier": RT.TIER, "table_sha256": table["sha256"],
                "stage_sha256": stage_meta["stage_sha256"], "model_slot_horizons_ticks": HZ8, "slot_instants_s": T8.tolist(), "surface_slots": SLOTS, "consumer_contract": CONTRACT}
        arrs["meta_json"] = np.array(json.dumps(meta))
        np.savez(DEP / f"DEPLOY8_POS_{tree}_s{seed}.npz", **arrs)
    RT.jdump(res, out)
    chk = json.load(open(out, encoding="utf-8"))                          # assert on the ARTIFACT
    assert chk["controls"]["ALL_PASS"] is True and chk["bars"]["compliant|RETIME_DEPLOY8"]["FZR-E1"] and chk["seeds"]["1"]["surfaces"]["S6"]["sets"]["compliant"]["arms"]["RETIME_DEPLOY8"]
    for seed in SEEDS:
        z = np.load(DEP / f"DEPLOY8_POS_{tree}_s{seed}.npz")
        assert z["S2_pos"].shape[1:] == (4, 2) and z["S6_pos"].shape[1:] == (6, 2) and z["S2_compliant"].dtype == bool
    try:
        os.rmdir(work)
    except OSError:
        pass
    print(f"WROTE {out} controls=ALL_PASS wall={res['wall_s']}s", flush=True)
    print(f"ZZDEP8OKZZ {tree}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--mutate-knot", default=None, help="IDX:DELTA -- deliberate regression (e.g. 1:0.01); writes raw/retime_deploy/mutation8/ only")
    a = ap.parse_args()
    mut = None
    if a.mutate_knot:
        i, d = a.mutate_knot.split(":")
        mut = (int(i), float(d))
        if not 0 <= mut[0] <= 7:
            raise SystemExit("--mutate-knot must index one of the eight knots (0..7)")
    return forward(a.tree, mut)


if __name__ == "__main__":
    sys.exit(main())
