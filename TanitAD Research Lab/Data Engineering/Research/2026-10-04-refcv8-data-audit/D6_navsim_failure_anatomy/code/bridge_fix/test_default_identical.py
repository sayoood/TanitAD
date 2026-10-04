"""T-DEFAULT (SPEC_P1P2 s8): the bridge COPY is the base plus INSERTIONS only; with the new flags absent it is the base.

Run (tanitad venv):  PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/tanitad/Scripts/python.exe test_default_identical.py
Writes test_default_identical.json next to it.  Exits non-zero on any failure.

Checks
  1. every non-equal diff opcode between `<file>.base` and `<file>` is an INSERT (no base line deleted or changed), for
     refcv7_bridge.py and run_bridge7.py; boot7.py and gpu_lock.py are byte-identical to their base.
  2. the base files are byte-identical to the LIVE package files at test time (so the .base copies really are the base).
  3. AST: inside run_model7 every use of the new export is under `if EXPORT_FAN["on"]`; in run_bridge7.main every new
     statement referencing export_fan / nav_override / gpu_lock_job is under an `if` on that very flag.
  4. runtime (needs the clean tree): importing the COPY gives ARMS7 == base ARMS7 + {R7_NAVFOLLOW, R7_NAVFLIP}, every base arm entry unchanged,
     EXPORT_FAN defaults to off, `_template` accepts the new arms, and apply_nav_override maps (left->right, right->left, follow->follow) for flip
     and everything -> follow for follow.
  5. MUTATION: a copy with one base line deleted must FAIL check 1 (the checker is not vacuous).
"""
from __future__ import annotations

import ast
import difflib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/code")
res = {"checks": []}


def check(name, ok, detail=""):
    res["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
    print(("OK   " if ok else "FAIL "), name, detail)


def insert_only(base_bytes: bytes, new_bytes: bytes):
    a = base_bytes.decode("utf-8").split("\n")
    b = new_bytes.decode("utf-8").split("\n")
    ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != "equal"]
    return all(o[0] == "insert" for o in ops), [(o[0], o[2] - o[1], o[4] - o[3]) for o in ops]


def _raises(m, out):
    o2 = dict(out)
    try:
        m.export_fan_from_out(o2, {})
    except m.RefusedInput:
        return True
    return False


def main():
    for f in ("refcv7_bridge.py", "run_bridge7.py"):
        base = open(os.path.join(HERE, f + ".base"), "rb").read()
        new = open(os.path.join(HERE, f), "rb").read()
        ok, ops = insert_only(base, new)
        check(f"1 {f}: diff vs base is INSERT-only", ok, str(ops))
    for f in ("boot7.py", "gpu_lock.py"):
        check(f"1 {f}: byte-identical to base", open(os.path.join(HERE, f), "rb").read() == open(os.path.join(HERE, f + ".base"), "rb").read())
    for f in ("refcv7_bridge.py", "run_bridge7.py", "boot7.py", "gpu_lock.py"):
        live = open(os.path.join(LIVE, f), "rb").read()
        check(f"2 {f}.base == live package file", live == open(os.path.join(HERE, f + ".base"), "rb").read())
    # 3. AST
    src = open(os.path.join(HERE, "refcv7_bridge.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    rm = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_model7"][0]
    bad = []
    for node in ast.walk(rm):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == "fan":
            bad.append(node.lineno)
    gated = [n for n in rm.body if isinstance(n, ast.If) and "EXPORT_FAN" in ast.dump(n.test)]
    inside = set()
    for g in gated:
        for n in ast.walk(g):
            if hasattr(n, "lineno"):
                inside.add(n.lineno)
    check("3a run_model7: every res['fan'] use is inside `if EXPORT_FAN[...]`", bool(bad) and all(l in inside for l in bad), f"uses at {bad}")
    rb = ast.parse(open(os.path.join(HERE, "run_bridge7.py"), encoding="utf-8").read())
    mainf = [n for n in ast.walk(rb) if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    flag_names = ("export_fan", "nav_override", "gpu_lock_job", "gpu_lock_pid", "fh_fan")
    uncovered = []
    gate_ifs = [n for n in ast.walk(mainf) if isinstance(n, ast.If) and any(fn in ast.dump(n.test) for fn in flag_names)]
    covered = set()
    for g in gate_ifs:
        for n in ast.walk(g):
            if hasattr(n, "lineno"):
                covered.add(n.lineno)
    base_lines = set(open(os.path.join(HERE, "run_bridge7.py.base"), encoding="utf-8").read().split("\n"))
    for i, ln in enumerate(open(os.path.join(HERE, "run_bridge7.py"), encoding="utf-8").read().split("\n"), 1):
        if any(fn in ln for fn in flag_names) and ln not in base_lines and i not in covered:
            uncovered.append((i, ln.strip()[:70]))
    # the only allowed uncovered mentions are the argparse declarations and the fh_fan opener (both inert when the flag is absent)
    allowed = [u for u in uncovered if "add_argument" in u[1] or u[1].startswith("fh_fan") or u[1].startswith("if a.export_fan") or "if a.gpu_lock_job" in u[1] or u[1].startswith("if spec.get") or u[1].startswith("if a.export_fan")]
    check("3b run_bridge7: every new flag use is gated by an `if` on that flag (or is an argparse declaration)", len(uncovered) == len(allowed), f"ungated: {[u for u in uncovered if u not in allowed]}")
    # 4. runtime import of the COPY
    sys.path.insert(0, HERE)
    try:
        spec_ = importlib.util.spec_from_file_location("refcv7_bridge_copy", os.path.join(HERE, "refcv7_bridge.py"))
        m = importlib.util.module_from_spec(spec_)
        sys.modules["refcv7_bridge_copy"] = m
        spec_.loader.exec_module(m)
        spec0 = importlib.util.spec_from_file_location("refcv7_bridge_base", os.path.join(LIVE, "refcv7_bridge.py"))
        m0 = importlib.util.module_from_spec(spec0)
        spec0.loader.exec_module(m0)
        new_arms = set(m.ARMS7) - set(m0.ARMS7)
        check("4a ARMS7 = base + {R7_NAVFOLLOW, R7_NAVFLIP}", new_arms == {"R7_NAVFOLLOW", "R7_NAVFLIP"}, str(sorted(new_arms)))
        check("4b every base arm entry unchanged", all(m.ARMS7[k] == m0.ARMS7[k] for k in m0.ARMS7))
        check("4c EXPORT_FAN defaults off", m.EXPORT_FAN == {"on": False})
        check("4d _template accepts the new arms", all(m._template(a) == "R6_A1" for a in new_arms))
        from tanitad.refs import refb
        ix = {n: refb.NAV_COMMANDS.index(n) for n in ("follow", "left", "right")}
        def nv(n):
            return {"nav_index": ix[n], "name": n, "navsim_argmax": 0}
        flip = {n: m.apply_nav_override(nv(n), "flip")["name"] for n in ("follow", "left", "right")}
        foll = {n: m.apply_nav_override(nv(n), "follow")["name"] for n in ("follow", "left", "right")}
        check("4e flip: left<->right, follow unchanged", flip == {"follow": "follow", "left": "right", "right": "left"}, str(flip))
        check("4f follow: everything -> follow", foll == {"follow": "follow", "left": "follow", "right": "follow"}, str(foll))
        check("4g NAV_COMMANDS index map is the one the base rows use (left=1)", ix["left"] == 1)
    except Exception as e:                                                   # noqa: BLE001
        check("4 runtime import of the copy", False, repr(e)[:300])
    # 6/7. the converter and the export on SYNTHETIC `out` dicts built around REAL banked rows (no GPU, no model)
    try:
        import numpy as np
        import torch
        rows = []
        with open("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/raw/milestones/step30000/bridge_navhard/rows_R7_A1.jsonl", encoding="utf-8") as fh:
            for i, ln in enumerate(fh):
                if i >= 20:
                    break
                rows.append(json.loads(ln))
        dmax = max(float(np.abs(m.knots_to_navsim(np.asarray(r["knots"], dtype=np.float64)) - np.asarray(r["poses"])).max()) for r in rows)
        check("6 knots_to_navsim(banked knots) reproduces the banked seam poses (20 rows)", dmax <= 1e-5, f"max abs {dmax:.3g}")
        r = rows[3]
        rng = np.random.default_rng(0)
        cand = rng.normal(size=(1, 181, 8, 2)).astype(np.float32) * 5.0
        cand[0, 42] = np.asarray(r["knots"], dtype=np.float32)
        out = {"r7_candidates": torch.from_numpy(cand), "r7_n_fan": 117, "sel_idx": torch.tensor([42]),
               "sel_score_v3": torch.randn(1, 117), "reach_keep": torch.ones(1, 117, dtype=torch.bool),
               "r7_score": torch.randn(1, 181), "r7_sel_idx": torch.tensor([150]), "r7_pick_is_wta": torch.tensor([True]),
               "traj_r7": torch.from_numpy(cand[:, 150])}
        fan = m.export_fan_from_out(out, {"sel_idx_base": 41})
        d2 = float(np.abs(fan["cands_poses"][42] - np.asarray(r["poses"], dtype=np.float32)).max())
        check("7a export: cands_poses[sel_idx] == the banked emitted poses (synthetic out around a real row)", d2 <= 1e-5, f"max abs {d2:.3g}")
        check("7b export: shapes [181,8,2] / [181,8,3], n_fan 117, WTA flag", fan["cands_knots"].shape == (181, 8, 2) and fan["cands_poses"].shape == (181, 8, 3) and fan["n_fan"] == 117 and fan["r7_pick_is_wta"] is True)
        o117 = {k: v for k, v in out.items() if k not in ("r7_candidates", "r7_n_fan", "r7_score", "r7_sel_idx", "r7_pick_is_wta", "traj_r7")}
        o117["anchor_traj"] = torch.from_numpy(cand[:, :117])
        f117 = m.export_fan_from_out(o117, {"sel_idx_base": 41})
        d3 = float(np.abs(f117["cands_poses"][42] - np.asarray(r["poses"], dtype=np.float32)).max())
        check("7c as-launched model (NO r7 heads): universe = anchor_traj FAN117, r7 fields None, cands_poses[sel_idx] == banked poses",
              f117["cands_poses"].shape == (117, 8, 3) and f117["universe"] == "FAN117" and f117["r7_score"] is None and f117["r7_sel_idx"] is None and f117["traj_r7_poses"] is None and d3 <= 1e-5,
              f"max abs {d3:.3g}")
        check("7d export REFUSES an `out` with neither r7_candidates nor anchor_traj", _raises(m, {k: v for k, v in o117.items() if k != "anchor_traj"}))
    except Exception as e:                                                   # noqa: BLE001
        check("6/7 converter + export on banked rows", False, repr(e)[:300])
    # 5. mutation: delete one base line from the copy -> check 1 must fail
    new = open(os.path.join(HERE, "run_bridge7.py"), "rb").read().decode().split("\n")
    mut = "\n".join(new[:200] + new[201:]).encode()
    ok, ops = insert_only(open(os.path.join(HERE, "run_bridge7.py.base"), "rb").read(), mut)
    check("5 MUTATION (one base line deleted) is REJECTED by the insert-only check", not ok, str(ops))
    res["n_ok"] = sum(c["ok"] for c in res["checks"])
    res["n"] = len(res["checks"])
    json.dump(res, open(os.path.join(HERE, "test_default_identical.json"), "w"), indent=1)
    print(res["n_ok"], "/", res["n"])
    return 0 if res["n_ok"] == res["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
