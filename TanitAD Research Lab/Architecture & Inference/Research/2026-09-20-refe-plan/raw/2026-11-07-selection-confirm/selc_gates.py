#!/usr/bin/env python3
"""Preflight gates for the H-SEL-1 confirmation (CPU, seconds). Writes gate_preflight.json; exit 0 only if every gate is green.

G-RULE   the frozen rule in selc_rule.py == p2_score.py's own `rule_vec` / `rule_loop`, re-executed FROM ITS SOURCE TEXT (ast-extracted, so p2_score's
         top-level read is never re-run and p2_result.json is never touched), on the 1,058 C3 tokens: vectorised 1058/1058, literal loop 1058/1058,
         banked pick (delta = 0) 1058/1058 == C3's banked pick; RED arm: the guard sign flipped must DISAGREE with the loop on >= 1 token.
G-PIN    PREREG_SEL.md (the rule's frozen source) still hashes to e00fa28a...
"""
import ast
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
YSEL = "E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107"
sys.path.insert(0, YSEL)
sys.path.insert(0, HERE)
import sel_common as sc  # noqa: E402
import sel_rules as sr  # noqa: E402
import selc_rule as R  # noqa: E402

PREREG_SEL_SHA = "e00fa28a7a5d6352b6f5a1782b34966bd75d88ad6b99ce7a614264581db0b7f0"


def p2_functions(ns):
    src = open(YSEL + "/p2_score.py", encoding="utf-8").read()
    tree = ast.parse(src)
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("rule_vec", "rule_loop")]
    assert len(fns) == 2
    exec(compile(ast.Module(body=fns, type_ignores=[]), "p2_score.py", "exec"), ns)
    return ns["rule_vec"], ns["rule_loop"]


def main():
    g = {}
    g["G_PIN_prereg_sel_sha256"] = hashlib.sha256(open(YSEL + "/PREREG_SEL.md", "rb").read()).hexdigest()
    g["G_PIN_PASS"] = g["G_PIN_prereg_sel_sha256"] == PREREG_SEL_SHA
    D = sc.load()
    L, v0, ax0, v4 = D["logits"], D["v0"], D["ax0"], D["v4_all"]
    N = len(L)
    ns = {"np": np, "sc": sc, "sr": sr, "L": L, "v0": v0, "ax0": ax0, "v4": v4, "r": np.arange(N), "s": sr.speed_ratio(v4, v0),
          "GATE": (v0 > 5.0) & (np.abs(ax0) < 0.3), "P": sr.probs(L), "agg": sc.agg_np(L, np.float64)}
    vec0, loop0 = p2_functions(ns)
    a_vec = R.rule_vec(L, v4, v0, ax0)
    o_vec = vec0(0.20, 0.05)
    a_loop = np.array([R.rule_loop(L, v4, v0, ax0, i) for i in range(N)])
    o_loop = np.array([loop0(i, 0.20, 0.05) for i in range(N)])
    red = R.rule_vec(L, v4, v0, ax0, flip_guard=True)
    g.update(n=N, G_RULE_vec_copy_eq_p2=int((a_vec == o_vec).sum()), G_RULE_loop_copy_eq_p2=int((a_loop == o_loop).sum()),
             G_RULE_loop_eq_vec=int((a_loop == a_vec).sum()), G_RULE_banked_eq_c3_pick=int((R.banked_pick(L) == D["pick"]).sum()),
             G_RULE_delta0_eq_c3_pick=int((R.rule_vec(L, v4, v0, ax0, delta=0.0) == D["pick"]).sum()),
             G_RULE_RED_flipguard_disagree=int((red != a_loop).sum()), n_changed_c3=int((a_vec != D["pick"]).sum()),
             G_RULE_step_speed_eq_sel_common=bool(np.array_equal(R.step_speed4(D["traj"]), sc.step_speed(D["traj"]))))
    g["G_RULE_PASS"] = bool(g["G_RULE_vec_copy_eq_p2"] == N and g["G_RULE_loop_copy_eq_p2"] == N and g["G_RULE_loop_eq_vec"] == N
                            and g["G_RULE_banked_eq_c3_pick"] == N and g["G_RULE_delta0_eq_c3_pick"] == N and g["G_RULE_RED_flipguard_disagree"] > 0
                            and g["G_RULE_step_speed_eq_sel_common"])
    g["PASS"] = bool(g["G_PIN_PASS"] and g["G_RULE_PASS"])
    json.dump(g, open(os.path.join(HERE, "gate_preflight.json"), "w"), indent=1)
    print(json.dumps(g, indent=1))
    return 0 if g["PASS"] else 2


if __name__ == "__main__":
    sys.exit(main())
