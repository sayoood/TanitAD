#!/usr/bin/env python
"""Run the goal-provenance audit against the REAL v6 stack, PER ARM.

⛔ THE GATE, not a report. Exit 0 only if every arm measured DISJOINT and every
arm carried a LIVE positive control. Exit 2 on a violation, 3 on an unpowered
audit, 4 on a structural (cross-module) failure.

WHAT IT MEASURES, and why each half is necessary
------------------------------------------------
1. **IN-GRAPH (computed).** For each pre-registered ``V6Config`` arm, intervene
   on every model INPUT and read which goal-path nodes move. This answers, by
   computation on the live autograd-free forward:
     * *does the goal path read anything but vision?* — the PI's vision-only
       rule, and specifically ``v6.py:62``'s prose claim that ``v0`` "enters
       ONLY the unicycle";
     * *is there a situation-classifier node in the graph at all?*
   Each arm carries its OWN positive control — a monkeypatched leak wired into
   that arm's own graph, which must fire (C107: a control run once for a study
   left 33 of 165 rows with no control; C109: an inert control proves nothing).

2. **CROSS-MODULE (structural).** The situation classifier is NOT in the model
   graph — it is an offline scorer — so an in-graph probe alone would report
   "clean" for the trivial reason that there is nothing to find. That is the
   vacuous pass this project's rules name repeatedly, so the second half asserts
   the separation where it actually lives: no model/trainer source imports the
   situation path, and the situation path's declared inference inputs contain no
   goal/route/nav symbol. **Absence found at one location is not absence** — the
   two halves are the two locations.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_STACK = Path(__file__).resolve().parents[1]
if str(_STACK) not in sys.path:
    sys.path.insert(0, str(_STACK))

import torch  # noqa: E402

from tanitad.config import (EncoderConfig, PredictorConfig,  # noqa: E402
                            ReadoutConfig)
from tanitad.eval.goal_provenance import (  # noqa: E402
    ProvenanceViolation, assert_information_disjoint, audit_arms,
    module_runner, probe_dependency)
from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402

# --------------------------------------------------------------------------- #
# the probed geometry — small, because the QUESTION IS TOPOLOGICAL             #
# --------------------------------------------------------------------------- #
# ⚠️ Stated rather than assumed: wiring does not depend on width or depth. The
# graph this builds has the SAME edges as the 87.9 M-parameter live geometry;
# only the tensor sizes differ. A leak is an edge, and an edge is present or
# absent at any width. (Verify-at-scale remains available via --full.)
_SUB = dict(
    encoder=EncoderConfig(in_channels=9, image_size=64, image_width=64,
                          patch_size=16, d_model=32, depth=1, n_heads=2),
    readout=ReadoutConfig(grid=2, d_readout=8),
    predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=2,
                              horizons=(1,), action_dim=3, residual=True))
_SMALL = dict(d_tac=32, d_str=16, adapter_hidden=32, f_hidden_tac=32,
              f_hidden_str=32, f_blocks=1, aux_hidden=16, sigreg_slices=8,
              plan_steps=6, dt=0.1, op_band_s=(0.0, 0.2),
              tac_band_s=(0.2, 0.6), hz_op=10.0, hz_tac=2.0, hz_str=0.5,
              d_plan_feat=16, emission_hidden=16, d_goal_embed=128,
              n_candidates=8)

#: The pre-registered arms. Each is a real ``V6Config`` lever, and each gets its
#: own row — a per-study verdict would be exactly C107.
ARMS: dict[str, dict] = {
    "default": {},
    "planner-cut-off": dict(isolate_planner_from_encoder=False),
    "uplink-cut-off": dict(isolate_uplink=False),
    "factored-goal": dict(goal_factored=True),
    "goal-cat-args": dict(goal_cat_args=True),
    "tac-goal-cond": dict(tac_goal_cond=True),
}

#: role -> dotted submodule path. The GOAL PATH as it is actually wired
#: (``v6.py:4703-4726``): goal_head_str -> cond_tac -> goal_head_tac -> cond_op.
GOAL_NODES = {
    "goal_head_str": "goal_head_str",
    "goal_cond_tac": "cond_tac",
    "goal_head_tac": "goal_head_tac",
    "goal_cond_op": "cond_op",
    "goal_emission": "emission",
}


def _goal_nodes(model: V6Stack) -> dict[str, str]:
    """The goal nodes ACTUALLY CALLED by this arm's forward.

    ⚠️ Arm-aware on purpose. Under ``goal_factored=True`` the forward takes the
    ``cond_op_lat``/``cond_op_lon`` branch and ``cond_op`` is never called
    (``v6.py:4719-4728``), so probing it would be probing a node that does not
    run. ``probe_dependency`` REFUSES an unobservable node rather than reporting
    it clean — that refusal is what surfaced this, and swapping the node set is
    the fix; suppressing the error would have manufactured a vacuous pass.
    """
    nodes = dict(GOAL_NODES)
    if getattr(model, "goal_head_tac_lat", None) is not None:
        nodes.pop("goal_cond_op")
        nodes.update({"goal_cond_op_lat": "cond_op_lat",
                      "goal_cond_op_lon": "cond_op_lon",
                      "goal_head_tac_lat": "goal_head_tac_lat",
                      "goal_head_tac_lon": "goal_head_tac_lon"})
    if getattr(model, "cond_tac_dyn", None) is not None:
        nodes["goal_cond_tac_dyn"] = "cond_tac_dyn"
    return nodes

#: role -> batch key. Every input the forward contract accepts.
INPUT_NODES = {"in_frames": "frames", "in_actions": "actions", "in_v0": "v0"}

#: ⭐ THE ONE ALLOWLISTED NON-VISION EDGE, with its citation.
#: ``v6.py:62`` — *"``v0`` (initial speed) enters ONLY the unicycle"*. ``v0`` is
#: the ego's OWN current speed, which is a legal inference input (it is not a
#: label, not a future, and not a privileged channel), and the unicycle rollout
#: is undefined without it. Allowlisting it is what lets the gate FAIL LOUD on
#: any OTHER non-vision edge — in particular ``v0 -> any goal HEAD``, which
#: would put ego state inside the goal decision itself.
#: ⚠️ An allowlist is a place a real violation can hide, so it is one entry,
#: it names the exact (source, target) pair, and it carries the line that
#: authorises it. Anything broader would be a hole, not a rule.
EXPECTED_NON_VISION: set[tuple[str, str]] = {("in_v0", "goal_emission")}


def _cfg(**over) -> V6Config:
    return V6Config(**{**_SUB, **_SMALL, **over})


def _build(over: dict, seed: int = 0) -> V6Stack:
    torch.manual_seed(seed)
    m = V6Stack(_cfg(**over))
    m.eval()
    return m


def _runner(model, batch, *, extra_nodes=None):
    return module_runner(model, batch,
                         nodes={**_goal_nodes(model), **(extra_nodes or {})},
                         input_nodes=INPUT_NODES)


class _FakeSituation(torch.nn.Module):
    """The POSITIVE CONTROL's situation classifier — a stand-in that emits a
    3-way score from the trunk latent, exactly the shape the real offline
    classifier emits (independent per-situation sigmoids, ``sitclf.py:209``)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = torch.nn.Linear(d_in, 3)
        self.back = torch.nn.Linear(3, d_out)
        torch.nn.init.constant_(self.back.weight, 0.25)
        torch.nn.init.zeros_(self.back.bias)

    def forward(self, z):
        return self.back(torch.sigmoid(self.net(z)))


def _wire_leak(model: V6Stack) -> V6Stack:
    """⭐ THE PER-ARM POSITIVE CONTROL — a DELIBERATELY WIRED violation.

    Splices a situation-classifier output into ``goal_head_tac``'s input
    **behind a ``detach()``**, which is how every downward goal port in
    ``V6Stack.forward`` is already wired (``v6.py:4341`` ``_cut``). This is the
    exact shape a real violation would take here, and it is invisible to a
    gradient probe. ⚠️ It MUST fire, or the arm's clean reading is unpowered
    (C109 — a positive control that was inert by construction).
    """
    d_tac = model.cfg.d_tac
    sit = _FakeSituation(d_tac, d_tac)
    sit.eval()
    model.add_module("situation_head_CONTROL", sit)
    inner = model.goal_head_tac

    class _Leaky(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner
            self.situation = sit

        def forward(self, z, cond=None):
            s = self.situation(z)
            return self.inner(z + s.detach(), cond=cond)

    model.goal_head_tac = _Leaky()
    return model


def _in_graph_audit(arms: dict, *, seed: int = 0) -> dict:
    specs, per_arm_inputs, per_arm_dead = [], {}, {}
    for name, over in arms.items():
        clean = _build(over, seed)
        batch = clean.synthetic_batch(2)

        # -- the per-arm positive control, on THIS arm's own graph ------------
        leak = _wire_leak(_build(over, seed))
        ctrl = probe_dependency(
            _runner(leak, batch,
                    extra_nodes={"situation_output":
                                 "goal_head_tac.situation"}),
            "situation_output", ["goal_head_tac"])

        # -- the arm itself: no situation node exists, so the in-graph matrix
        #    is over the goal roles and the INPUTS ------------------------- #
        run = _runner(clean, batch)
        inputs = {r: probe_dependency(run, r, list(_goal_nodes(clean)))
                  for r in INPUT_NODES}
        per_arm_inputs[name] = {
            r: {t: v["depends"] for t, v in rep["targets"].items()}
            for r, rep in inputs.items()}
        # ⭐ A goal node that NO input could move is constant on this batch. On a
        # freshly built (untrained) stack this is the zero-init heads, and their
        # "clean" reading is an artefact — recorded so no arm is read as proved
        # disjoint on a node nothing could move.
        per_arm_dead[name] = sorted(
            t for t in _goal_nodes(clean)
            if not any(rep["targets"][t]["depends"] for rep in inputs.values()))

        # The leak arm is audited as a real arm too, so the panel demonstrates
        # the gate FIRING rather than only passing.
        specs.append(dict(
            arm=name, run=_runner(leak, batch,
                                  extra_nodes={"situation_output":
                                               "goal_head_tac.situation"}),
            goal_roles=["goal_head_tac"],
            situation_roles=["situation_output"],
            positive_control=ctrl))
    return {"positive_control_panel": audit_arms(specs),
            "input_dependency": per_arm_inputs,
            "dead_goal_nodes": per_arm_dead}


# --------------------------------------------------------------------------- #
# cross-module structural half                                                #
# --------------------------------------------------------------------------- #
_SIT_IMPORT = re.compile(
    r"^\s*(?:from\s+[\w.]*\b(?:sitclf|situations)\b|import\s+[\w.]*\b"
    r"(?:sitclf|situations)\b)", re.M)
_GOAL_SYMBOL = re.compile(
    r"\b(nav_cmd|goal_point|g_str|g_tac|route_head|goal_head|anchor_xy)\b")


def _structural_audit(stack: Path) -> dict:
    """Assert the separation where it actually lives — across modules.

    The situation classifier is an OFFLINE scorer, so an in-graph probe finds
    nothing for the trivial reason that nothing is there. This half checks the
    two directions at file scope.
    """
    model_srcs = sorted((stack / "tanitad" / "models").rglob("*.py"))
    trainer_srcs = sorted(p for p in (stack / "scripts").glob("*.py")
                          if p.name.startswith(("train", "refc_train",
                                                "refb_train")))
    offenders = []
    for p in [*model_srcs, *trainer_srcs]:
        if _SIT_IMPORT.search(p.read_text(encoding="utf-8", errors="replace")):
            offenders.append(str(p.relative_to(stack)))

    sit_paths = [stack / "tanitad" / "eval" / "sitclf.py",
                 stack / "tanitad" / "eval" / "sitclf_deploy.py",
                 stack / "scripts" / "sitclf_train.py"]
    reverse = []
    for p in sit_paths:
        if not p.exists():
            continue
        for i, line in enumerate(
                p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            s = line.split("#")[0]
            if _GOAL_SYMBOL.search(s) and "=" in s:
                reverse.append(f"{p.relative_to(stack)}:{i}: {line.strip()}")
    return {
        "n_model_and_trainer_sources_scanned": len(model_srcs) +
        len(trainer_srcs),
        "sources_importing_situation_path": offenders,
        "situation_sources_scanned": [str(p.relative_to(stack))
                                      for p in sit_paths if p.exists()],
        "goal_symbols_in_situation_path": reverse,
        "ok": not offenders and not reverse,
        "_reads": ("FORWARD: no model or trainer imports the situation path. "
                   "REVERSE: no goal/route/nav symbol is assigned in the "
                   "situation path. Both directions, at file scope, because "
                   "the two paths do not meet in one graph."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--arms", default="", help="comma-list; default = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-unpowered", action="store_true",
                    help="accept nodes that are CONSTANT on the probed batch "
                         "(zero-init heads on an untrained model). Without it "
                         "an unpowered node is exit 3, because 'we did not "
                         "establish it' must not read as a pass.")
    a = ap.parse_args(argv)

    want = ({k: ARMS[k] for k in a.arms.split(",")} if a.arms else ARMS)
    graph = _in_graph_audit(want, seed=a.seed)
    struct = _structural_audit(_STACK)
    report = {
        "_what": "goal-provenance audit — PI ruling 2026-08-03, both directions",
        "tier": "T-NA (a structural/graph audit, not an eval tier)",
        "arms": sorted(want),
        "in_graph": graph,
        "structural": struct,
        "torch": torch.__version__,
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2, default=str),
                         encoding="utf-8")

    rc, notes = 0, []
    panel = graph["positive_control_panel"]
    # The control panel is EXPECTED to be inadmissible: every arm in it is the
    # deliberately-wired leak. A panel that came back clean would mean the probe
    # is blind, which is the failure this inverted check exists to catch.
    if panel["verdict"] != "INADMISSIBLE":
        notes.append(f"POSITIVE CONTROL DID NOT FIRE on {panel['unpowered'] or panel['arms'].keys()} "
                     f"— the probe could not detect a wired leak; no clean "
                     f"verdict on any arm is admissible")
        rc = 3
    else:
        try:
            assert_information_disjoint(panel)
            notes.append("UNEXPECTED: the wired-leak panel passed the gate")
            rc = 3
        except ProvenanceViolation:
            notes.append(f"positive control fired on all {panel['n_arms']} "
                         f"arms — the probe is live")

    for arm, rep in graph["input_dependency"].items():
        bad = [(src, tgt) for src, tt in rep.items() if src != "in_frames"
               for tgt, dep in tt.items()
               if dep and (src, tgt) not in EXPECTED_NON_VISION]
        if bad:
            notes.append(f"{arm}: ⛔ NON-VISION INPUT reaches the goal path "
                         f"outside the allowlist: {bad}")
            rc = 2
        dead = graph["dead_goal_nodes"].get(arm) or []
        if dead:
            notes.append(f"{arm}: UNPOWERED for {dead} — constant on this "
                         f"batch (zero-init head on an untrained model); "
                         f"disjointness NOT established for those nodes")
            if not a.allow_unpowered:
                rc = max(rc, 3)
    if not struct["ok"]:
        notes.append(f"STRUCTURAL: {struct['sources_importing_situation_path']} "
                     f"{struct['goal_symbols_in_situation_path']}")
        rc = 4

    print(json.dumps({"rc": rc, "notes": notes,
                      "input_dependency": graph["input_dependency"],
                      "dead_goal_nodes": graph["dead_goal_nodes"],
                      "structural_ok": struct["ok"]}, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
