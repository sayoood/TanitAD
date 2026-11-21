#!/usr/bin/env python3
"""refav1_add_floor.py — add the ``ha0_ext`` ECHO control to a BANKED refav1 dump, post hoc.

WHY (D-REFAV1-CCOS-EVAL, 2026-09-05): ``stack/tanitad/eval/echo_gate.py`` refuses a panel
that lacks ``ha0_ext`` (``REQUIRED_REFERENCES = ("ha", "ha0_ext")``), and every refav1 dump
banked before this date carries ``ha`` / ``ha0`` only. ``ha0_ext`` is MODEL-FREE — it needs the
episode's recorded speed and curvature at t0 and nothing else — so it is added here with
ZERO GPU, through the SAME functions ``refav1_arm.run_dump`` uses for the new dumps
(``hold_ext_controls`` + ``paths_from_controls``), so the banked and the fresh dumps carry one
definition of the control.

⛔ NEVER IN PLACE. The banked dump is read-only; this writes a NEW dump directory
(``<out>/ep*.npz`` + ``decisions/`` + ``manifest.json``) with the extra arm, and records in the
manifest that the arm was added post hoc, from which episode files, and with what checks.

THE CHECKS (each must pass for every window, or the tool refuses):
  * the dump's ``v0`` == ``poses[2t, 3]`` of the episode file (bit-exact float32);
  * the dump's ``ha_controls`` == ``_kin_actions(t-1)`` recomputed from the episode
    (so the episode file is the one the dump was produced from);
  * the new ``ha0_ext`` differs from ``ha`` ONLY in its curvature channel (a0 identical).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import shutil
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_arm():
    spec = importlib.util.spec_from_file_location(
        "refav1_arm_for_floor", os.path.join(_HERE, "refav1_arm.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="the BANKED dump dir (read-only)")
    ap.add_argument("--episodes", required=True, help="v2ep episode dir")
    ap.add_argument("--out", required=True, help="NEW dump dir to write")
    a = ap.parse_args(argv)
    ra = _load_arm()
    import torch

    man_path = os.path.join(a.dump, "manifest.json")
    with open(man_path, encoding="utf-8") as fh:
        man = json.load(fh)
    if "ha0_ext" in man.get("arms", []):
        raise SystemExit(f"{a.dump} already carries ha0_ext — nothing to add")
    if os.path.isdir(a.out) and glob.glob(os.path.join(a.out, "ep*.npz")):
        raise SystemExit(f"{a.out} already holds a dump; refusing to overwrite")
    os.makedirs(os.path.join(a.out, "decisions"), exist_ok=True)
    rec_units = ((man.get("action_units") or {}).get("recorded")) or "kappa"
    k = int(man["grid"]["horizon_k"])
    dt = float(man["grid"]["dt_s"])
    eps = {e["file_index"]: e for e in man["episodes"]}
    files = sorted(glob.glob(os.path.join(a.dump, "ep*.npz")))
    n_win, max_v0, max_ha, n_kdiff = 0, 0.0, 0.0, 0
    for f in files:
        fi = int(os.path.basename(f)[2:5])
        e = eps[fi]
        o = torch.load(os.path.join(a.episodes, f"{e['name']}.v2ep.pt"),
                       map_location="cpu", weights_only=False)
        v = o["poses"][:, 3].float()
        kap = o["actions"][:, 0].float()
        with np.load(f) as d:
            arrs = {kk: d[kk] for kk in d.files}
        dec_f = os.path.join(a.dump, "decisions", os.path.basename(f))
        with np.load(dec_f) as d:
            dec = {kk: d[kk] for kk in d.files}
        ws = arrs["ws"].astype(int).ravel()
        v0s = arrs["v0"].astype(np.float32).ravel()
        outs, ctrls = [], []
        for i, t in enumerate(ws):
            v0 = float(v[2 * t])
            max_v0 = max(max_v0, abs(np.float32(v0) - v0s[i]))
            # the dump's held action must be reproducible from THIS episode file
            hold = torch.stack([(v[2 * t] - v[2 * t - 2]) / dt, kap[2 * t - 2]]).float()
            max_ha = max(max_ha, float(np.abs(dec["ha_controls"][i, 0] - hold.numpy()).max()))
            ext = ra.hold_ext_controls(None, v, kap, int(t), dt=dt)
            n_kdiff += int(float(ext[1]) != float(hold[1]))
            assert float(ext[0]) == float(hold[0]), "a0 must be ha's a0 exactly"
            p = ra.paths_from_controls(ext[None].expand(k, 2), v0, dt, k,
                                       action_units=rec_units)
            outs.append(p.float().cpu().numpy())
            ctrls.append(ext[None].expand(k, 2).float().cpu().numpy()[None])
        arrs["ha0_ext"] = np.concatenate(outs).astype(np.float32)
        dec["ha0_ext_controls"] = np.concatenate(ctrls).astype(np.float32)
        np.savez_compressed(os.path.join(a.out, os.path.basename(f)), **arrs)
        np.savez_compressed(os.path.join(a.out, "decisions", os.path.basename(f)), **dec)
        n_win += len(ws)
    if max_v0 > 0.0:
        raise SystemExit(f"v0 mismatch {max_v0} — wrong episode files; output NOT valid")
    if max_ha > 1e-6:
        raise SystemExit(f"ha_controls not reproducible from the episodes (max {max_ha})")
    man["arms"] = list(man["arms"]) + ["ha0_ext"]
    man.setdefault("tiers", {})["ha0_ext"] = ra.ARM_TIERS["ha0_ext"]
    man.setdefault("arm_meaning", {})["ha0_ext"] = ra.ARM_MEANING["ha0_ext"]
    man["floors_added_post_hoc"] = {
        "tool": "taniteval/tools/refav1_add_floor.py", "arm": "ha0_ext",
        "source_dump": os.path.abspath(a.dump), "episodes": os.path.abspath(a.episodes),
        "n_windows": n_win, "checks": {"v0_max_abs_diff": float(max_v0),
                                       "ha_controls_max_abs_diff": float(max_ha),
                                       "n_windows_kappa_differs_from_ha": int(n_kdiff)},
        "rule": ra.hold_ext_controls.__doc__}
    for extra in os.listdir(a.dump):            # carry any other sidecar files verbatim
        src = os.path.join(a.dump, extra)
        if os.path.isfile(src) and not extra.startswith("ep") and extra != "manifest.json":
            shutil.copy2(src, os.path.join(a.out, extra))
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=str)
    print(f"[add_floor] {n_win} windows / {len(files)} episodes -> {a.out}; v0 max|d| "
          f"{max_v0}, ha_controls max|d| {max_ha:.2e}, kappa differs from ha on "
          f"{n_kdiff}/{n_win} windows", flush=True)


if __name__ == "__main__":
    main()
