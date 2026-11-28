#!/usr/bin/env python3
"""P2: does the top-2 kinematic gate read ANY scene input? Asserted at SOURCE, two ways.

M23 claims the gate uses "only the candidate's own waypoints -- no scene input, no new
perception, no gradient step". This script tests that claim by MECHANISM, not by docstring:

  PROBE A (static, AST): walk `rewards.py`, resolve the call graph reachable from
      `_kinematic_feasibility` and `_comfort`, and collect EVERY `ctx` key those
      functions read. Compare against the partition of the ctx key set into
      SCENE-DERIVED vs CONSTANT.

  PROBE B (dynamic, permutation): call the two components twice on the SAME candidate
      waypoints -- once with the true ctx, once with every scene fact GARBLED
      (v0 scrambled, lead_path/obstacles replaced, gt_traj replaced). If the outputs
      are BITWISE identical, no scene fact reached the component. Same shape as the
      repo's FORBIDDEN_FUTURE_CTX permutation test.

  CONTROL (must FAIL, or the probe is not sensitive): `_headway` and `_collision`
      are known scene readers. The same garble MUST change them. A permutation test
      in which nothing moves proves nothing about the component -- it may only prove
      the garble did not take. The positive control is mandatory.

ASCII-only output (cp1252 console).
"""
import ast
import json
import os
import sys

REPO = os.environ.get("TANITAD_REPO", "C:/Users/Admin/refcv4b_repo")
REWARDS = os.path.join(REPO, "stack", "tanitad", "rl", "rewards.py")
PROBE = os.path.join(REPO, "stack", "scripts", "rl_fan_rerank_probe.py")

# SCENE = derived from the observed scene or the ego's measured state.
# CONST = a fixed programme constant carrying no scene fact.
SCENE_KEYS = {"v0", "lead_path", "obstacles", "gt_traj", "lead_xy", "lead_track"}
CONST_KEYS = {"dt", "lead_len_m", "a_max", "kappa_max", "jerk_max", "lat_acc_max",
              "min_motion_m", "ttc_min_s", "ade_scale_m", "d_safe_m", "r_veh_m",
              "veh_r_m", "d_safe", "headway_target_s"}

out = {"probe": "verify_no_scene_input", "repo": REPO}


def _fn_defs(tree):
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def _ctx_keys_and_calls(node):
    """Every ctx.get('k') / ctx['k'] read, and every call name, inside `node`."""
    keys, calls, bare = set(), set(), []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if (isinstance(f, ast.Attribute) and f.attr == "get"
                    and isinstance(f.value, ast.Name) and f.value.id == "ctx"):
                if n.args and isinstance(n.args[0], ast.Constant):
                    keys.add(n.args[0].value)
                else:
                    bare.append("dynamic-get")
            elif isinstance(f, ast.Name):
                calls.add(f.id)
            elif isinstance(f, ast.Attribute):
                calls.add(f.attr)
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                and n.value.id == "ctx"):
            if isinstance(n.slice, ast.Constant):
                keys.add(n.slice.value)
            else:
                bare.append("dynamic-subscript")
    return keys, calls, bare


def probe_a():
    src = open(REWARDS, "r", encoding="utf-8").read()
    defs = _fn_defs(ast.parse(src))
    seen, keys, bare = set(), set(), []
    stack = ["_kinematic_feasibility", "_comfort"]
    while stack:
        name = stack.pop()
        if name in seen or name not in defs:
            continue
        seen.add(name)
        k, calls, cb = _ctx_keys_and_calls(defs[name])
        keys |= k
        bare += ["%s:%s" % (name, x) for x in cb]
        for c in calls:
            if c in defs and c not in seen:
                stack.append(c)
    ctrl = set()
    for name in ("_headway", "_collision"):
        if name in defs:
            k, _c, _b = _ctx_keys_and_calls(defs[name])
            ctrl |= k
    return {
        "reachable_functions": sorted(seen),
        "ctx_keys_read": sorted(keys),
        "ctx_keys_scene": sorted(keys & SCENE_KEYS),
        "ctx_keys_const": sorted(keys & CONST_KEYS),
        "ctx_keys_unclassified": sorted(keys - SCENE_KEYS - CONST_KEYS),
        "dynamic_ctx_access": bare,
        "control_scene_readers_keys": sorted(ctrl),
        "control_reads_scene": bool(ctrl & SCENE_KEYS),
    }


def probe_a_gate_expression():
    """The gate score must be EXACTLY feasibility + comfort; the candidate set must come
    from the model's own sel_score. Read both out of the probe source."""
    src = open(PROBE, "r", encoding="utf-8").read()
    tree = ast.parse(src)
    found = {"r_kin_rhs": None, "rank_src": None, "gate_expr": None,
             "gate_ks": None, "reach_mask": None}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    if t.id == "r_kin":
                        found["r_kin_rhs"] = ast.unparse(n.value)
                    if t.id == "rank" and found["rank_src"] is None:
                        found["rank_src"] = ast.unparse(n.value)
                    if t.id == "GATE_KS":
                        found["gate_ks"] = ast.unparse(n.value)
                if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                        and t.value.id == "pick" and isinstance(t.slice, ast.JoinedStr)):
                    found["gate_expr"] = ast.unparse(n.value)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == "masked_fill":
            found["reach_mask"] = ast.unparse(n)
    return found


def probe_b():
    import torch
    sys.path.insert(0, os.path.join(REPO, "stack"))
    sys.path.insert(0, REPO)
    from tanitad.rl import rewards as RW

    g = torch.Generator().manual_seed(7)
    B, N, S = 6, 32, 5
    traj = (torch.randn(B, N, S, 2, generator=g) * 3.0).cumsum(dim=-2)
    traj[..., 0, :] = 0.0                      # origin, as with_origin() makes it

    def mk_ctx(scramble):
        gg = torch.Generator().manual_seed(99 if scramble else 1)
        v0 = torch.rand(B, 1, generator=gg) * (60.0 if scramble else 20.0)
        lead = torch.randn(B, 1, S, 2, generator=gg) * (50.0 if scramble else 5.0)
        lead = lead + (500.0 if scramble else 0.0)
        return {"dt": 0.5, "v0": v0, "lead_len_m": 4.5, "lead_path": lead,
                "obstacles": lead[:, :, :1, :],
                "gt_traj": torch.randn(B, 4, 2, generator=gg) * (40.0 if scramble else 4.0)}

    ctx_t, ctx_s = mk_ctx(False), mk_ctx(True)
    res = {}
    for name in ("feasibility", "comfort", "headway", "collision", "progress"):
        fn = RW.COMPONENTS[name].fn
        a = fn(traj, ctx_t).detach()
        b = fn(traj, ctx_s).detach()
        res[name] = {"bitwise_identical_under_scene_garble": bool(torch.equal(a, b)),
                     "max_abs_diff": float((a - b).abs().max()),
                     "mean_true": float(a.mean())}
    a = (RW.COMPONENTS["feasibility"].fn(traj, ctx_t)
         + RW.COMPONENTS["comfort"].fn(traj, ctx_t)).detach()
    b = (RW.COMPONENTS["feasibility"].fn(traj, ctx_s)
         + RW.COMPONENTS["comfort"].fn(traj, ctx_s)).detach()
    res["GATE_SCORE_r_kin"] = {
        "bitwise_identical_under_scene_garble": bool(torch.equal(a, b)),
        "max_abs_diff": float((a - b).abs().max())}
    res["GATE_ARGMAX"] = {"identical": bool(torch.equal(a.argmax(dim=1), b.argmax(dim=1))),
                          "max_abs_diff": 0.0}
    return res


if __name__ == "__main__":
    out["probe_A_ast"] = probe_a()
    out["probe_A_gate_expression"] = probe_a_gate_expression()
    out["probe_B_permutation"] = probe_b()

    A, Bp = out["probe_A_ast"], out["probe_B_permutation"]
    verdict = {
        "A_no_scene_ctx_key_read": A["ctx_keys_scene"] == [],
        "A_no_unclassified_key": A["ctx_keys_unclassified"] == [],
        "A_no_dynamic_ctx_access": A["dynamic_ctx_access"] == [],
        "A_positive_control_reads_scene": A["control_reads_scene"],
        "B_gate_score_invariant":
            Bp["GATE_SCORE_r_kin"]["bitwise_identical_under_scene_garble"],
        "B_positive_control_headway_moved":
            not Bp["headway"]["bitwise_identical_under_scene_garble"],
        "B_positive_control_collision_moved":
            not Bp["collision"]["bitwise_identical_under_scene_garble"],
    }
    verdict["NO_SCENE_INPUT_HOLDS"] = all(verdict.values())
    out["verdict"] = verdict
    dst = sys.argv[1] if len(sys.argv) > 1 else "no_scene_input.json"
    json.dump(out, open(dst, "w", encoding="utf-8"), indent=2)
    print("=== P2 VERIFY: does the top-2 kinematic gate read scene input? ===")
    print("A(ast)  reachable fns   :", ",".join(A["reachable_functions"]))
    print("A(ast)  ctx keys read   :", A["ctx_keys_read"])
    print("A(ast)  of which SCENE  :", A["ctx_keys_scene"], "CONST:", A["ctx_keys_const"])
    print("A(ast)  unclassified    :", A["ctx_keys_unclassified"])
    print("A(ast)  dynamic access  :", A["dynamic_ctx_access"])
    print("A(ast)  CONTROL scene rd:", A["control_scene_readers_keys"])
    ge = out["probe_A_gate_expression"]
    print("A(ast)  r_kin rhs       :", ge["r_kin_rhs"])
    print("A(ast)  rank source     :", ge["rank_src"])
    print("A(ast)  reach mask      :", ge["reach_mask"])
    print("A(ast)  gate expression :", ge["gate_expr"])
    print("A(ast)  GATE_KS         :", ge["gate_ks"])
    for k, v in Bp.items():
        ident = v.get("bitwise_identical_under_scene_garble", v.get("identical"))
        print("B(perm) %-22s identical=%-5s maxdiff=%.6e"
              % (k, ident, v.get("max_abs_diff", float("nan"))))
    print("--- verdict ---")
    for k, v in verdict.items():
        print("  %-42s %s" % (k, v))
    print("wrote", dst)
    sys.exit(0 if verdict["NO_SCENE_INPUT_HOLDS"] else 3)
