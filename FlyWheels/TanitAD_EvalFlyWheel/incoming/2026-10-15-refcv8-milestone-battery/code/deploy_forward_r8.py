"""deploy_forward_r8.py -- the PER-SURFACE form of the deployed re-timing stage, `RETIME_DEPLOY` (SHIP-R8 AMENDMENT 1) scored on one eval139 tree, CPU only, zero GPU.

AMENDMENT 2 (2026-10-19 ~16:10) changed the SCORED arm to `RETIME_DEPLOY8` (ONE call on all 8 model slots; built by the sibling `deploy8_forward_r8.py`).  THIS file's per-surface form
(each surface re-timed on its own 4 / 6 points) is therefore printed BESIDE the condition, never scored; it also carries the S2 identity control against FORWARD_TRAINFIT.

NEW glue module.  `a0k0_headroom_retime_r8.py` (RT) and `retime_trainfit_r8.py` (TFM) are imported and NOT edited; RT.forward's gain source is monkeypatched
(the same device `retime_trainfit_r8.cmd_forward` uses).

    python deploy_forward_r8.py --tree step30000ema_unc                     # -> raw/retime_deploy/FORWARD_DEPLOY_<tree>.json
    python deploy_forward_r8.py --tree step30000ema_unc --mutate-s2-knot 1:0.01   # deliberate regression: must make the S2 identity control FAIL
    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' python -X utf8 deploy_forward_r8.py ...

The stage: s'(t) = s_prior(t) + g(t) * (s_os(t) - s_prior(t)) on the os plan's own path; the prior is `ha0_ext_pose`; g(t) is the DEPLOYED table
(raw/retime_deploy/FROZEN_GAINS_RETIME_deploy.json, built from the frozen TRAIN gains sha256 b208c9c1...).  Arms in the file:
  RETIME_DEPLOY   the per-surface form of AMENDMENT 1 (one gain function of time; S6's 1 s / 2 s gains are the S2 values); BESIDE under AMENDMENT 2
  RETIME_TRAINFIT the per-surface train tables (beside; the S2 identity control compares against it)
  CONST_G1        known-value control: gains 1 == os exactly (zero-width CI on every row)
  CONST_G0        known-value control: gains 0 == the prior's arc length on the os path (independent walker)
  CONST_G05       the no-fit flat-0.5 comparator, beside
CONTROLS (any failure WITHHOLDS: no FORWARD_DEPLOY file is left behind, WITHHELD_DEPLOY_<tree>.json is written, rc 4):
  (a) RT.forward's own controls (decisions / validation / os-cell-equals-banked / on-path);  (b) CONST_G1 - os exactly 0 on every row;  (c) array-level known values;
  (d) THE S2 IDENTITY: RETIME_DEPLOY's S2 rows equal RETIME_TRAINFIT's S2 rows EXACTLY, both in this file and in raw/retime_trainfit/FORWARD_TRAINFIT_<tree>.json
      (an identity, because the deployed S2 table IS the train S2 table);  (e) the frozen train gains and the deployed table hash to their pinned values.
"""
from __future__ import annotations

import argparse
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
import build_deploy_gains_r8 as BD  # noqa: E402
import retime_trainfit_r8 as TFM  # noqa: E402

PKG = HERE.parent
DEP = PKG / "raw" / "retime_deploy"
TF = PKG / "raw" / "retime_trainfit"
SEEDS = (0, 1)
SURFS = ("S2", "S6")
SETS = ("compliant", "all")
IDENT_ARM, DEPLOY_ARM, TRAIN_ARM = "RETIME_DEPLOY", "RETIME_DEPLOY", "RETIME_TRAINFIT"
CONST_ARMS = TFM.CONST_ARM_NAMES


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


def s2_identity(deploy_res: dict | None, ref_res: dict | None, ref_arm: str = TRAIN_ARM, deploy_arm: str = DEPLOY_ARM) -> dict:
    """RETIME_DEPLOY's S2 rows (both inference seeds x both sets) EXACTLY equal `ref_arm`'s S2 rows of `ref_res`.  `ref_res` None -> status ABSENT, pass False
    (an absent reference is never a pass)."""
    if deploy_res is None or ref_res is None:
        return {"status": "ABSENT", "pass": False, "n_leaves_compared": 0, "mismatches": ["the reference or the deploy result is absent"]}
    mism, n = [], 0
    for s in SEEDS:
        for st in SETS:
            try:
                a = deploy_res["seeds"][str(s)]["surfaces"]["S2"]["sets"][st]["arms"][deploy_arm]
                b = ref_res["seeds"][str(s)]["surfaces"]["S2"]["sets"][st]["arms"][ref_arm]
            except KeyError as e:
                mism.append(f"seed {s} {st}: missing {e}")
                continue
            mism += [f"s{s}|{st}{m}" for m in exact_equal(a, b)]
            n += count_leaves(a)
    return {"status": "COMPARED", "pass": bool(not mism and n > 0), "n_leaves_compared": n, "n_mismatches": len(mism), "mismatches": mism[:20]}


def surface_arm_gains(deploy: dict, train_fz: dict, surf: str) -> dict:
    """The per-surface gain vectors of every arm in the file (float64)."""
    tab = np.asarray(deploy["surfaces"][surf]["RETIME_DEPLOY"], np.float64)
    out = {DEPLOY_ARM: tab, TRAIN_ARM: np.asarray(train_fz["surfaces"][surf]["all"]["RETIME"], np.float64)}
    out.update(TFM.const_weights(len(tab)))
    return out


def mutate_deploy(deploy: dict, knot_index: int, delta: float) -> dict:
    """DELIBERATE REGRESSION: add `delta` to one deployed knot and rebuild the per-surface tables from the perturbed knots.  (S2's instants are knots, so S2's table moves.)"""
    import copy
    d = copy.deepcopy(deploy)
    d["knots"]["gain"][knot_index] += delta
    for surf, node in d["surfaces"].items():
        node["RETIME_DEPLOY"] = BD.surface_table(node["instants_s"], d["knots"]["t_s"], d["knots"]["gain"])
    return d


def aggregate_controls(c: dict) -> bool:
    """ALL_PASS only if EVERY named control is True; a control that is absent from the dict is a fail (a missing control is not a pass)."""
    need = ("rt_forward", "const_g1_exactly_os", "known_values", "s2_identity_vs_forward_trainfit", "s2_identity_in_file", "gains_pinned")
    return all(c.get(k) is True for k in need)


# ------------------------------------------------------------------------------------------ the run
def _gains_pinned(deploy_path: Path, train_path: Path) -> dict:
    ok_train = TFM.sha256_file(train_path) == BD.SOURCE_SHA256 and (TF / "FROZEN_GAINS_RETIME_train.sha256").read_text().strip() == BD.SOURCE_SHA256
    dep_sha = TFM.sha256_file(deploy_path)
    ok_dep = dep_sha == BD.DEPLOY_SHA256_FILE.read_text().strip()
    return {"train_sha256": BD.SOURCE_SHA256, "train_pinned": bool(ok_train), "deploy_sha256": dep_sha, "deploy_matches_sha256_file": bool(ok_dep)}


def forward(tree: str, mutate: tuple | None = None) -> int:
    TFM.check_prereg()
    train_fz = json.load(open(BD.SOURCE, encoding="utf-8"))
    TFM.assert_frozen_is_train(train_fz)
    pin = _gains_pinned(BD.DEPLOY_JSON, BD.SOURCE)
    if not (pin["train_pinned"] and pin["deploy_matches_sha256_file"]):
        raise SystemExit(f"[deploy-forward] REFUSING: a gains file differs from its pinned sha256 ({pin})")
    deploy = json.load(open(BD.DEPLOY_JSON, encoding="utf-8"))
    bad = BD.check_deploy(deploy)
    if bad:
        raise SystemExit("[deploy-forward] REFUSING: the deploy table fails AMENDMENT 1's checks: " + "; ".join(bad))
    if mutate:
        deploy = mutate_deploy(deploy, mutate[0], mutate[1])
    out_dir = DEP / ("mutation" if mutate else "")
    RT.FROZEN_JSON, RT.OUT = BD.SOURCE, out_dir                      # RT.forward reads the TRAIN file for its metadata and writes FORWARD_RETIME_<tree>.json into OUT
    cache, kv, calls = {}, {}, []
    orig_tl, orig_si = RT.tools_load, RT.surface_inputs

    def tools_load_once():
        if "t" not in cache:
            cache["t"] = orig_tl()
        return cache["t"]

    def surface_inputs_rec(tools, ctx, surf, first, tree_):
        S = orig_si(tools, ctx, surf, first, tree_)
        i = len(calls)
        calls.append(surf)
        if surf != SURFS[i % len(SURFS)]:
            raise SystemExit(f"[deploy-forward] surface order {calls} is not seed-major (S2, S6, S2, S6): the known-value bookkeeping would mislabel")
        kv[f"{surf}|s{SEEDS[i // len(SURFS)]}"] = TFM.known_value_arrays(S["X"]["O"], S["X"]["HP"])
        return S

    def frozen_weights_deploy(_fz, surf, _v0):
        return surface_arm_gains(deploy, train_fz, surf)
    RT.tools_load, RT.surface_inputs, RT.frozen_weights = tools_load_once, surface_inputs_rec, frozen_weights_deploy
    t0 = time.time()
    rc = RT.forward(tree)
    p = out_dir / f"FORWARD_RETIME_{tree}.json"
    res = json.load(open(p, encoding="utf-8"))
    os.remove(p)
    controls = {"rt_forward": bool(res["controls"]["ALL_PASS"] and rc == 0), "gains_pinned": bool(pin["train_pinned"] and pin["deploy_matches_sha256_file"]), "pins": pin}
    try:
        controls["const_g1_cell_rows_checked_exactly_zero"] = TFM.assert_const_g1_cells(res)
        controls["const_g1_exactly_os"] = True
    except SystemExit as e:
        controls["const_g1_exactly_os"], controls["const_g1_error"] = False, str(e)
    try:
        TFM.assert_known_values(kv)
        controls["known_values"] = True
    except SystemExit as e:
        controls["known_values"], controls["known_values_error"] = False, str(e)
    controls["known_value_arrays"] = kv
    tfp = TF / f"FORWARD_TRAINFIT_{tree}.json"
    ref = json.load(open(tfp, encoding="utf-8")) if tfp.exists() else None
    ident = s2_identity(res, ref)
    ident["reference"] = f"raw/retime_trainfit/FORWARD_TRAINFIT_{tree}.json arm {TRAIN_ARM}, S2, seeds 0 and 1, compliant and all-window sets"
    ident["reference_controls_ALL_PASS"] = (ref or {}).get("controls", {}).get("ALL_PASS")
    controls["s2_identity_vs_forward_trainfit"] = bool(ident["pass"] and ident["reference_controls_ALL_PASS"] is True)
    controls["s2_identity_vs_forward_trainfit_detail"] = ident
    infile = s2_identity(res, res)
    controls["s2_identity_in_file"] = bool(infile["pass"])
    controls["s2_identity_in_file_detail"] = infile
    controls["ALL_PASS"] = aggregate_controls(controls)
    s6_diff = exact_equal(res["seeds"]["0"]["surfaces"]["S6"]["sets"]["compliant"]["arms"][DEPLOY_ARM],
                          res["seeds"]["0"]["surfaces"]["S6"]["sets"]["compliant"]["arms"][TRAIN_ARM])
    controls["info_s6_deploy_differs_from_trainfit"] = bool(s6_diff)
    res["controls_rt"] = res["controls"]
    res["controls"] = controls
    res.update({"tool": "deploy_forward_r8.py", "tree_name": tree, "n_boot": RT.N_BOOT, "bootstrap_seed": 0,
                "role": ("MUTATION RUN (deliberate regression; NOT a result)" if mutate else
                         "BESIDE the condition (SHIP-R8 AMENDMENT 2 scores RETIME_DEPLOY8 = one 8-slot call): the PER-SURFACE form RETIME_DEPLOY of AMENDMENT 1, frozen TRAIN gains "
                         "(sha256 b208c9c1...) UNCHANGED, scored on eval139 " + tree),
                "mutation": ({"knot_index": mutate[0], "delta": mutate[1]} if mutate else None),
                "deploy_gains": str(BD.DEPLOY_JSON), "deploy_gains_sha256": pin["deploy_sha256"], "deploy_tables": {s: deploy["surfaces"][s]["RETIME_DEPLOY"] for s in SURFS},
                "train_tables": {s: train_fz["surfaces"][s]["all"]["RETIME"] for s in SURFS},
                "arm_names": "RETIME_DEPLOY = the per-surface form of Amendment 1 (BESIDE under Amendment 2); RETIME_TRAINFIT = per-surface train tables, beside; CONST_G1/CONST_G0 = known-value controls; CONST_G05 = flat 0.5 comparator, beside",
                "script_sha256": TFM.sha256_file(__file__), "wall_s": round(time.time() - t0, 1), "generated": time.strftime("%FT%T")})
    if mutate:
        mp = DEP / "mutation" / f"MUTATION_{tree}.json"
        RT.jdump({"tree": tree, "mutation": res["mutation"], "s2_identity_vs_forward_trainfit_pass": controls["s2_identity_vs_forward_trainfit"],
                  "s2_identity_in_file_pass": controls["s2_identity_in_file"], "n_mismatches_vs_forward_trainfit": ident.get("n_mismatches"),
                  "first_mismatches": ident.get("mismatches", [])[:5], "ALL_PASS": controls["ALL_PASS"],
                  "mutation_detected": bool(not controls["s2_identity_vs_forward_trainfit"] and not controls["ALL_PASS"]), "generated": res["generated"]}, mp)
        print(f"MUTATION {mp} detected={not controls['ALL_PASS']} (identity_vs_trainfit={controls['s2_identity_vs_forward_trainfit']}, in_file={controls['s2_identity_in_file']})", flush=True)
        print(f"ZZDEPMUT_{'RED' if not controls['ALL_PASS'] else 'GREEN'}ZZ {tree}", flush=True)
        return 0 if not controls["ALL_PASS"] else 5
    out, wh = DEP / f"FORWARD_DEPLOY_{tree}.json", DEP / f"WITHHELD_DEPLOY_{tree}.json"
    if not controls["ALL_PASS"]:
        RT.jdump({"tree": tree, "WITHHELD": {k: v for k, v in controls.items() if v is False or k.endswith("_error") or k.endswith("_detail")}, "generated": res["generated"]}, wh)
        if out.exists():
            os.remove(out)
        print(f"WITHHELD: a control did not pass -> {wh}\nZZDEPWITHHELDZZ {tree}", flush=True)
        return 4
    if wh.exists():
        os.remove(wh)
    RT.jdump(res, out)
    chk = json.load(open(out, encoding="utf-8"))                      # assert on the ARTIFACT
    assert chk["controls"]["ALL_PASS"] is True and chk["seeds"]["1"]["surfaces"]["S6"]["sets"]["compliant"]["arms"][DEPLOY_ARM]
    print(f"WROTE {out} controls=ALL_PASS wall={res['wall_s']}s", flush=True)
    print(f"ZZDEPOKZZ {tree}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--mutate-s2-knot", default=None, help="IDX:DELTA -- deliberate regression (e.g. 1:0.01); writes raw/retime_deploy/mutation/ only")
    a = ap.parse_args()
    mut = None
    if a.mutate_s2_knot:
        i, d = a.mutate_s2_knot.split(":")
        mut = (int(i), float(d))
        if not 0 <= mut[0] <= 3:
            raise SystemExit("--mutate-s2-knot must index one of the four S2 knots (0..3)")
    return forward(a.tree, mut)


if __name__ == "__main__":
    sys.exit(main())
