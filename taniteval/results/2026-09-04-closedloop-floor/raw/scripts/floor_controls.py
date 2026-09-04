#!/usr/bin/env python3
"""RAW bit-identity controls for the closed-loop trivial floor (2026-09-04).

Two controls, both of which must read a value decided BEFORE the run:

* DETERMINISM  — `cl_ha0` run twice with identical flags must be bit-identical.
  (The existing panel's repro control reads exactly 0.0 with a zero-width CI on
  437 windows; the floor has to reproduce that property, not merely be close.)
* RENDER INDEPENDENCE — `cl_ha0` under the HQ render (cull 0.95 + sky 0.3) and
  under the MORNING render (neither) must ALSO be bit-identical. A policy that
  reads no pixels cannot move when the pixels change, and this is the positive
  proof of `canon_provenance.reads_pixels = False`. It is also what licenses ONE
  floor run being used as the bar for panels rendered differently.

⛔ Compared at the RAW trajectory level, before any metric — a metric-level zero
can be produced by a scorer that dropped both arms.
"""
import json, sys
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())


def compare(a, b, label):
    ra, rb = a["rollouts"], b["rollouts"]
    out = {"label": label, "n_rollouts_a": len(ra), "n_rollouts_b": len(rb)}
    if len(ra) != len(rb):
        out["verdict"] = "SHAPE MISMATCH"
        return out
    n = ident = 0
    dplan = dego = dv = dsteer = daccel = 0.0
    for x, y in zip(ra, rb):
        assert x["start_frame"] == y["start_frame"], (x["start_frame"], y["start_frame"])
        for sx, sy in zip(x["steps"], y["steps"]):
            n += 1
            same = True
            for i in range(4):
                for j in range(2):
                    d = abs(sx["plan"][i][j] - sy["plan"][i][j])
                    dplan = max(dplan, d)
                    same &= (d == 0.0)
            for i in range(4):
                d = abs(sx["ego"][i] - sy["ego"][i])
                dego = max(dego, d)
                same &= (d == 0.0)
            for k, acc in (("v", "dv"), ("steer", "dsteer"), ("accel", "daccel")):
                d = abs(sx[k] - sy[k])
                same &= (d == 0.0)
                if acc == "dv":
                    dv = max(dv, d)
                elif acc == "dsteer":
                    dsteer = max(dsteer, d)
                else:
                    daccel = max(daccel, d)
            ident += int(same)
    out.update({"n_steps": n, "bit_identical_steps": ident,
                "max_abs_dplan": dplan, "max_abs_dego": dego, "max_abs_dv": dv,
                "max_abs_dsteer": dsteer, "max_abs_daccel": daccel,
                "verdict": "BIT-IDENTICAL" if ident == n else "DIVERGED",
                "render_a": ra[0].get("render_quality"),
                "render_b": rb[0].get("render_quality")})
    return out


def floor_invariants(a, label):
    """The floor arm's OWN known-value controls, read off the produced record."""
    out = {"label": label, "arm": a["arm"]}
    steps = [s for r in a["rollouts"] for s in r["steps"]]
    out["n_steps"] = len(steps)
    out["n_rollouts"] = len(a["rollouts"])
    out["f_eff_is_null"] = a["f_eff"] is None
    out["ckpt_is_null"] = a["ckpt"] is None
    out["canon_reads_pixels"] = a["canon"]["reads_pixels"]
    if a["arm"] == "cl_ha0":
        out["expected"] = {"steer": 0.0, "accel": 0.0, "kappa_plan": 0.0,
                           "v_target_minus_v": 0.0,
                           "v_constant_within_rollout": True,
                           "plan_lateral_y": 0.0}
        out["measured"] = {
            "max_abs_steer": max(abs(s["steer"]) for s in steps),
            "max_abs_accel": max(abs(s["accel"]) for s in steps),
            "max_abs_kappa_plan": max(abs(s["kappa_plan"]) for s in steps),
            "max_abs_v_target_minus_v": max(abs(s["v_target"] - s["v"]) for s in steps),
            "v_constant_within_rollout": all(
                len({s["v"] for s in r["steps"]}) == 1 for r in a["rollouts"]),
            "max_abs_plan_lateral_y": max(abs(p[1]) for s in steps for p in s["plan"]),
        }
        out["all_controls_exact"] = (
            out["measured"]["max_abs_steer"] == 0.0
            and out["measured"]["max_abs_accel"] == 0.0
            and out["measured"]["max_abs_kappa_plan"] == 0.0
            and out["measured"]["max_abs_v_target_minus_v"] == 0.0
            and out["measured"]["v_constant_within_rollout"]
            and out["measured"]["max_abs_plan_lateral_y"] == 0.0)
    out["t0_per_rollout"] = {str(r["start_frame"]): r.get("floor_t0")
                             for r in a["rollouts"]}
    return out


if __name__ == "__main__":
    F = Path(sys.argv[1])
    P = F / "panel_cl_ha0" / "rollouts_cl_ha0_empty.json"
    res = {
        "controls": [
            compare(load(F / "repro_cl_ha0" / "rollouts_cl_ha0_empty.json"), load(P),
                    "DETERMINISM: cl_ha0 re-run, identical flags"),
            compare(load(F / "morn_cl_ha0" / "rollouts_cl_ha0_empty.json"), load(P),
                    "RENDER INDEPENDENCE: cl_ha0 under the MORNING render vs the HQ render"),
        ],
        "invariants": [floor_invariants(load(F / f"panel_{m}" / f"rollouts_{m}_empty.json"), m)
                       for m in ("cl_ha0", "cl_ha", "cl_ha0_ext")],
    }
    Path(sys.argv[2]).write_text(json.dumps(res, indent=2))
    for c in res["controls"]:
        print("ZZCTRL", c["label"][:34], c["verdict"],
              c.get("bit_identical_steps"), "/", c.get("n_steps"), "ZZ")
    for v in res["invariants"]:
        print("ZZINV", v["arm"], v.get("all_controls_exact"), v["n_steps"], "ZZ")
