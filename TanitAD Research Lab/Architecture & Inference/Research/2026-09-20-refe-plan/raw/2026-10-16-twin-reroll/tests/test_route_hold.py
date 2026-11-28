#!/usr/bin/env python3
"""Route hold for lane-rank twins (P6 D7 option 3, 2026-10-16): a re-ranking regression must go RED.

No nuPlan needed: a fake `driverl_runtime_map_features` with THEIR route-build loop and THEIR `_choose_route_edge` rule (nearest at the
first roadblock, outgoing-edge continuity after it) on a straight two-lane road, lane A at y=0 and lane B at y=3.5, four roadblocks.
The ego drives from A toward B over a rollout, and the route is rebuilt from the CURRENT pose every step, exactly as
`feature_builder._build_map_features` does.

    python test_route_hold.py            # the tests, then the two deliberate-regression arms (each must make a test FAIL)
    pytest -q test_route_hold.py         # the tests only
"""
from __future__ import annotations

import ast
import importlib
import math
import os
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.get("ROUTE_HOLD_CODE_DIR", HERE.parent / "code"))
sys.path.insert(0, str(CODE))


# ------------------------------------------------------------------ a fake map module with THEIR rule and THEIR build loop
class Edge:
    def __init__(self, eid, y, x0, x1):
        self.id, self.coords = eid, [(float(x), float(y)) for x in range(x0, x1 + 1, 5)]
        self.outgoing_edges, self.incoming_edges = [], []


class Block:
    def __init__(self, edges):
        self.interior_edges = edges


def _fake_module():
    M = types.ModuleType("fake_driverl_runtime_map_features")
    M._edge_centerline_coords = lambda e: e.coords

    def heading(e):
        c = e.coords
        return math.atan2(c[-1][1] - c[0][1], c[-1][0] - c[0][0])

    def anchor_score(e, anchor):          # THEIR score: distance + 5 * |heading error|
        d = min(math.hypot(x - anchor.x, y - anchor.y) for x, y in e.coords)
        return d + 5.0 * abs(math.atan2(math.sin(heading(e) - anchor.yaw), math.cos(heading(e) - anchor.yaw)))

    def continuity(e, prev):
        g = math.hypot(e.coords[0][0] - prev.coords[-1][0], e.coords[0][1] - prev.coords[-1][1])
        return g + 5.0 * abs(heading(e) - heading(prev))

    def choose(edges, anchor, prev_edge):  # THEIR _choose_route_edge, structurally verbatim
        cands = [e for e in edges if M._edge_centerline_coords(e)]
        if not cands:
            return None
        if prev_edge is None:
            return min(cands, key=lambda e: M._route_edge_anchor_score(e, anchor))
        out = {e.id for e in prev_edge.outgoing_edges}
        linked = [e for e in cands if e.id in out]
        return min(linked or cands, key=lambda e: continuity(e, prev_edge))

    def build(blocks, anchor):             # THEIR _sample_route_points_from_map loop (start index, then one edge per block)
        scores = [min(M._route_edge_anchor_score(e, anchor) for e in b.interior_edges) for b in blocks]
        s = scores.index(min(scores))
        out, prev = [], None
        for b in blocks[s:]:
            e = M._choose_route_edge(list(b.interior_edges), anchor, prev)   # looked up on the MODULE at call time
            if e is not None:
                out.append(e.id)
                prev = e
        return out

    M._route_edge_anchor_score, M._choose_route_edge, M.build = anchor_score, choose, build
    return M


def _road():
    blocks = []
    for i in range(4):
        a, b = Edge(f"A{i}", 0.0, 50 * i, 50 * i + 50), Edge(f"B{i}", 3.5, 50 * i, 50 * i + 50)
        blocks.append(Block([a, b]))
    for i in range(3):
        for ln in (0, 1):
            e, n = blocks[i].interior_edges[ln], blocks[i + 1].interior_edges[ln]
            e.outgoing_edges.append(n)
            n.incoming_edges.append(e)
    return blocks


def _anchor(x, y):
    return types.SimpleNamespace(x=x, y=y, yaw=0.0)


def _fresh_patch(M, rank):
    P = importlib.reload(importlib.import_module("route_lane_rank_patch"))
    P._RANK = rank
    M._choose_route_edge = P.make_patched(M._choose_route_edge, M)
    return P


def _rollout(P, M, blocks, hold):
    """40 steps; the ego moves 2 m forward and 0.1 m toward lane B per step (ends at y = 3.9, past lane B)."""
    routes = []
    with P.held_route(enabled=hold):
        for k in range(40):
            routes.append(M.build(blocks, _anchor(5.0 + 2.0 * k, 0.1 * k)))
    return routes


# ------------------------------------------------------------------ tests
def test_held_route_never_reranks():
    """THE REGRESSION TEST: under the hold every build is the step-0 rank-1 chain (lane B), never the logged lane A."""
    M, blocks = _fake_module(), _road()
    P = _fresh_patch(M, 1)
    hold = os.environ.get("REFE_ROUTE_HOLD", "1") != "0"   # the deliberate-regression arm flips this from outside
    routes = _rollout(P, M, blocks, hold)
    assert routes[0] == ["B0", "B1", "B2", "B3"], routes[0]
    bad = [(k, r) for k, r in enumerate(routes) if any(e.startswith("A") for e in r)]
    assert not bad, f"the route re-ranked back onto the logged lane at {len(bad)} of 40 steps, first at step {bad[0][0]}: {bad[0][1]}"
    assert P.LAST_HOLD is not None and P.LAST_HOLD["chain_ids"] == ("B0", "B1", "B2", "B3")
    assert P.LAST_HOLD["builds"] == 40 and P.LAST_HOLD["misses"] == 0


def test_unheld_reproduces_the_defect():
    """The CONTROL: without the hold, once the ego is nearer lane B, 'rank 1' is lane A again -- the measured D7 failure mode."""
    M, blocks = _fake_module(), _road()
    P = _fresh_patch(M, 1)
    routes = _rollout(P, M, blocks, hold=False)
    assert routes[0][0] == "B0"
    flipped = [k for k, r in enumerate(routes) if r[0].startswith("A")]
    assert flipped and flipped[0] == 18, flipped[:3]      # y = 1.8 > 1.75: lane B is now nearest, so rank 1 = lane A


def test_rank0_is_untouched_inside_a_hold():
    """Rank 0 (and goal-horizon twins, set_rank(0)) keep THEIR per-step nearest-lane rule exactly, hold scope or not."""
    M, blocks = _fake_module(), _road()
    ref = [M.build(blocks, _anchor(5.0 + 2.0 * k, 0.1 * k)) for k in range(40)]
    M2, blocks2 = _fake_module(), _road()
    P = _fresh_patch(M2, 0)
    assert _rollout(P, M2, blocks2, hold=True) == ref


def test_outside_a_scope_is_the_old_patch():
    """Single-build callers (build_scorer_targets, pdm_route_rank's oracle) see the pre-fix behaviour: rank-k at the GIVEN anchor."""
    M, blocks = _fake_module(), _road()
    P = _fresh_patch(M, 1)
    assert M.build(blocks, _anchor(5.0, 0.0))[0] == "B0"
    assert M.build(blocks, _anchor(5.0, 3.5))[0] == "A0"
    assert P._HOLD is None


def test_held_route_rejoins_after_a_miss():
    """A build that starts on a roadblock the step-0 chain never covered falls back to the un-held rule there and rejoins the chain."""
    M, blocks = _fake_module(), _road()
    P = _fresh_patch(M, 1)
    with P.held_route(enabled=True):
        first = M.build(blocks[1:], _anchor(55.0, 0.0))            # step 0 starts at block 1
        later = M.build(blocks, _anchor(5.0, 3.5))                 # a later build starting at block 0 (not covered)
    assert first == ["B1", "B2", "B3"] and later[1:] == ["B1", "B2", "B3"], (first, later)
    assert P.LAST_HOLD["misses"] == 1


def test_rollout_enters_the_hold_scope():
    """build_teacher_rollouts.rollout_from_frame runs its WHOLE closed loop (`_roll`) inside `held_route()` when the patch is installed."""
    src = (CODE / "build_teacher_rollouts.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "rollout_from_frame")
    assert "held_route" in ast.unparse(fn), "rollout_from_frame never builds a held_route() scope"
    withs = [n for n in ast.walk(fn) if isinstance(n, ast.With)]
    inside = [w for w in withs if any(isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_roll" for b in w.body for c in ast.walk(b))]
    assert inside, "the rollout loop `_roll(...)` is not inside a `with` scope"
    loose = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_roll"]
    assert len(loose) == 1, f"_roll is called {len(loose)} times in rollout_from_frame (expected exactly once, inside the hold)"
    assert "for k in range(span)" not in ast.unparse(fn), "the closed loop moved back into rollout_from_frame, outside the hold"


TESTS = [test_held_route_never_reranks, test_unheld_reproduces_the_defect, test_rank0_is_untouched_inside_a_hold,
         test_outside_a_scope_is_the_old_patch, test_held_route_rejoins_after_a_miss, test_rollout_enters_the_hold_scope]


def _run() -> int:
    fails = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {t.__name__}: {e}")
    return fails


def _mutation_arms() -> int:
    """Each arm is a deliberate regression and MUST turn at least one test RED; an arm that stays green proves the test blind."""
    import shutil
    import tempfile
    bad = 0
    # arm 1: the hold switched off (REFE_ROUTE_HOLD=0) -> test_held_route_never_reranks must FAIL
    r = subprocess.run([sys.executable, __file__, "--tests-only"], env={**os.environ, "REFE_ROUTE_HOLD": "0"}, capture_output=True, text=True)
    red = "FAIL test_held_route_never_reranks" in r.stdout
    print(f"MUTATION hold_off: {'RED (good)' if red else 'STILL GREEN (BAD)'}")
    bad += not red
    # arm 2: the rollout loop moved back OUT of the scope (the pre-fix build_teacher_rollouts) -> test_rollout_enters_the_hold_scope must FAIL
    with tempfile.TemporaryDirectory() as td:
        for f in ("route_lane_rank_patch.py", "build_teacher_rollouts.py"):
            shutil.copy(CODE / f, Path(td) / f)
        p = Path(td) / "build_teacher_rollouts.py"
        s = p.read_text(encoding="utf-8")
        s2 = s.replace("    with _hold:\n        out, goal_used = _roll(", "    if True:\n        out, goal_used = _roll(")
        assert s2 != s, "mutation arm 2 could not find its target line"
        p.write_text(s2, encoding="utf-8")
        r = subprocess.run([sys.executable, __file__, "--tests-only"], env={**os.environ, "ROUTE_HOLD_CODE_DIR": td}, capture_output=True, text=True)
        red = "FAIL test_rollout_enters_the_hold_scope" in r.stdout
        print(f"MUTATION loop_outside_scope: {'RED (good)' if red else 'STILL GREEN (BAD)'}")
        bad += not red
        # arm 3: the patch re-ranks inside the scope (chain lookup removed) -> test_held_route_never_reranks must FAIL
        q = Path(td) / "route_lane_rank_patch.py"
        s = q.read_text(encoding="utf-8")
        s3 = s.replace('if h is not None and h["chain_ids"] is not None:', 'if False:')
        assert s3 != s, "mutation arm 3 could not find its target line"
        q.write_text(s3, encoding="utf-8")
        p.write_text((CODE / "build_teacher_rollouts.py").read_text(encoding="utf-8"), encoding="utf-8")
        r = subprocess.run([sys.executable, __file__, "--tests-only"], env={**os.environ, "ROUTE_HOLD_CODE_DIR": td}, capture_output=True, text=True)
        red = "FAIL test_held_route_never_reranks" in r.stdout
        print(f"MUTATION chain_lookup_removed: {'RED (good)' if red else 'STILL GREEN (BAD)'}")
        bad += not red
    return bad


if __name__ == "__main__":
    f = _run()
    if "--tests-only" in sys.argv:
        raise SystemExit(1 if f else 0)
    m = _mutation_arms()
    print(f"ZZROUTE_HOLD_TESTS {'PASS' if f == 0 and m == 0 else 'FAIL'} tests_failed={f} mutation_arms_green={m}")
    raise SystemExit(1 if (f or m) else 0)
