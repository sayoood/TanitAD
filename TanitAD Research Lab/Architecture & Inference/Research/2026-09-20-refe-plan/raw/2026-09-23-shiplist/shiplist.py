"""Compute the ship list from the IMPORT GRAPH, not from memory.

POD_HANDOFF's own warning: "a ship list that under-counts is how a pod ends up debugging an import
at 2 a.m." It was written when there were 7 files, corrected to 9, and has since gained
`build_teacher_rollouts.py`, `navtrain_scenarios.py`, `route_rank.py` and `augment_search.py`. A
hand-maintained list drifts every time the package does -- so derive it.

Walks local imports transitively from the entry points the pod actually runs, over refe/ and code/,
by AST. Anything reachable must ship; anything unreachable is evidence, not runtime.
"""
import ast, os, sys

PKG = ("D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/"
       "Research/2026-09-20-refe-plan")
DIRS = [os.path.join(PKG, "refe"), os.path.join(PKG, "code")]

ENTRY = [
    "refe/train.py",                    # the training run
    "refe/build_teacher_rollouts.py",   # rank-0 targets
    "refe/augment_search.py",           # the augmented half
    "refe/build_scorer_targets.py",     # PDM targets
    "refe/planner.py",                  # scoring REFe in their harness
    "refe/validate_model.py",
]
GATES = [
    "refe/diag_consumer_conformance.py", "refe/diag_rank_distinctness.py",
    "refe/diag_architecture.py", "refe/diag_rope.py", "refe/diag_schedule.py",
    "refe/diag_guard_audit.py", "refe/diag_planner_holds.py",
    "refe/diag_signal_consistency.py",
    "refe/diag_train_resume.py",        # 2026-09-23 (R20): pod_train.sh refuses to launch without it
    "refe/diag_calib.py",               # 2026-09-23 (R22): per-sample rigs vs an OpenCV reference
]

local = {}
for d in DIRS:
    for f in os.listdir(d):
        if f.endswith(".py"):
            local[f[:-3]] = os.path.join(d, f)


def imports_of(path):
    try:
        t = ast.parse(open(path, encoding="utf-8").read())
    except Exception:
        return set()
    out = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            out.add(n.module.split(".")[0])
    return {m for m in out if m in local}


def closure(seeds):
    seen, stack = set(), list(seeds)
    while stack:
        p = stack.pop()
        rel = os.path.relpath(p, PKG).replace("\\", "/")
        if rel in seen:
            continue
        seen.add(rel)
        for m in imports_of(p):
            stack.append(local[m])
    return seen


run = closure([os.path.join(PKG, e) for e in ENTRY])
gate = closure([os.path.join(PKG, g) for g in GATES]) - run
allpy = {os.path.relpath(p, PKG).replace("\\", "/") for p in local.values()}
rest = allpy - run - gate

print(f"RUNTIME ({len(run)}) -- must ship:")
for f in sorted(run):
    print("   ", f)
print(f"\nGATES + their extra deps ({len(gate)}):")
for f in sorted(gate):
    print("   ", f)
print(f"\nNOT reachable ({len(rest)}) -- evidence, not runtime:")
for f in sorted(rest):
    print("   ", f)

HANDOFF9 = {"refe/model.py", "refe/load_dinov3.py", "refe/train.py", "refe/build_targets.py",
            "refe/build_scorer_targets.py", "refe/score_proposals.py", "refe/planner.py",
            "refe/validate_model.py", "code/augment_routes.py"}
missing = sorted(run - HANDOFF9)
print(f"\n!! REACHABLE BUT ABSENT from POD_HANDOFF's 9-file list ({len(missing)}):")
for f in missing:
    print("   ", f)
