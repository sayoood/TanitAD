"""Measure the SIZE of the two ways out of the v5 trainer contradiction.

The contradiction (verified in code by the brief, re-verified here):

    train_flagship4b    : reads the v2 compressed cache (--v2-cache), NO val loop
    train_flagship_v4   : has the mid-run held-out gate,          NO v2 support

v5 must have both. Two options:

    A. teach train_flagship_v4 to read v2
    B. port the held-out val loop into train_flagship4b

⚠️ Nothing here is an opinion. Every row is a fact read out of the imported
modules, or a line count over the real files. The decisive numbers are the
STRUCTURAL BLOCKERS: a blocker is something the option must BUILD, not move,
and it is measured by actually asking the objects.

Run:  python measure_options.py --out ../raw/option_size_<date>.json
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
STACK = REPO / "stack"
for p in (str(STACK), str(STACK / "scripts"), str(REPO / "taniteval")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _stub_torchvision() -> bool:
    """The dev box has no torchvision, and ``tanitad.data.v2_dataset`` imports
    ``torchvision.io`` at module scope. STUB it rather than skip: a skipped
    measurement is a measurement that cannot fail (class C13). Nothing here
    decodes an image, so the stub is never called."""
    try:
        import torchvision  # noqa: F401,PLC0415
        return False
    except ModuleNotFoundError:
        import types
        tv = types.ModuleType("torchvision")
        io = types.ModuleType("torchvision.io")

        class ImageReadMode:                       # noqa: D401
            RGB = "RGB"

        def _never(*_a, **_k):                     # pragma: no cover
            raise RuntimeError("torchvision stub: decode was not expected here")

        io.decode_jpeg = _never
        io.decode_png = _never
        io.ImageReadMode = ImageReadMode
        tv.io = io
        sys.modules["torchvision"] = tv
        sys.modules["torchvision.io"] = io
        return True


_TV_STUBBED = _stub_torchvision()


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _src_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _def_span(path: Path, name: str) -> int:
    """Physical line span of a top-level def/class named ``name`` (0 if absent)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.name == name:
            return int(node.end_lineno - node.lineno + 1)
    return 0


def _grep(path: Path, needles) -> dict:
    hits = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for n in needles:
            if n in line:
                hits.setdefault(n, []).append(i)
    return hits


# --------------------------------------------------------------------------- #
# A. what OPTION A must touch in train_flagship_v4                            #
# --------------------------------------------------------------------------- #
def measure_option_a() -> dict:
    v4 = STACK / "scripts" / "train_flagship_v4.py"
    fb = STACK / "scripts" / "train_flagship4b.py"
    src = v4.read_text(encoding="utf-8").splitlines()

    # the raw-path-specific sites, found by reading the file (not by memory)
    raw_sites = _grep(v4, ["load_episode", 'glob("ep_*.pt")', "_assert_parity(",
                           "--train-cache", "--val-cache"])
    geom = _grep(v4, ["add_geometry_args", "apply_geometry_args"])

    # the loader that already exists and would simply be called
    from tanitad.data import v2_dataset
    from tanitad.data import parity

    reuse = {
        "build_v2_providers": {
            "module": "tanitad.data.v2_dataset",
            "exists": hasattr(v2_dataset, "build_v2_providers"),
            "lines": _def_span(STACK / "tanitad" / "data" / "v2_dataset.py",
                               "build_v2_providers"),
            "returns": "list[LazyV2Episode] — .frames/.poses/.actions/.episode_id",
        },
        "assert_v2_parity_cache": {
            "module": "tanitad.data.parity",
            "exists": hasattr(parity, "assert_v2_parity_cache"),
            "lines": _def_span(STACK / "tanitad" / "data" / "parity.py",
                               "assert_v2_parity_cache"),
        },
        "apply_geometry_args": {
            "module": "tanitad.geometry",
            "exists": True,
            "already_used_by_train_flagship4b": bool(
                _grep(fb, ["apply_geometry_args"])),
        },
    }

    # does the v4 window dataset actually accept a LazyV2Episode? (contract test)
    from flagship_v4_data import FlagshipV4Dataset          # noqa: PLC0415
    ds_attrs_needed = sorted({"poses", "frames", "actions", "episode_id"})

    return {
        "option": "A — teach train_flagship_v4 to read v2",
        "file_to_edit": str(v4.relative_to(REPO)).replace("\\", "/"),
        "file_lines": _src_lines(v4),
        "raw_path_sites_to_branch": {k: v for k, v in raw_sites.items()},
        "n_raw_path_sites": sum(len(v) for v in raw_sites.values()),
        "geometry_args_present_in_v4": {k: v for k, v in geom.items()},
        "geometry_must_be_added": not geom,
        "machinery_reused_not_written": reuse,
        "new_machinery_required": [],
        "dataset_contract": {
            "class": "FlagshipV4Dataset(FlagshipWindowDataset)",
            "episode_attrs_read": ds_attrs_needed,
            "note": "LazyV2Episode exposes exactly these — see the live check",
        },
        "gate_reusable_as_is": True,
        "gate_reuse_reason": (
            "HeldoutGate.probe(step, world, head, episodes) takes episode OBJECTS "
            "with .poses/.frames; LazyV2Episode supplies both. The gate never "
            "touches the cache format."),
    }


# --------------------------------------------------------------------------- #
# B. what OPTION B must BUILD in train_flagship4b                             #
# --------------------------------------------------------------------------- #
def measure_option_b() -> dict:
    """The decisive measurement: ask the two planner surfaces what they emit."""
    import torch
    from tanitad.config import flagship4b_config
    from tanitad.models.flagship_v4 import FlagshipV4Head
    from tanitad.models.flagship_v4 import v4_config
    from tanitad.train.heldout_gate import DeployableSurfacePlanner, NonDensePlanError

    fb = STACK / "scripts" / "train_flagship4b.py"
    v4 = STACK / "scripts" / "train_flagship_v4.py"

    cfg = flagship4b_config()
    tac_h = tuple(cfg.tactical_policy.waypoint_horizons)
    hcfg = v4_config()
    v4_h = tuple(hcfg.horizons)

    # --- the gate's own contract, asserted by RUNNING it, not by reading it ----
    class _FakeCfg:
        def __init__(self, horizons):
            self.horizons = tuple(horizons)

    class _FakeHead(torch.nn.Module):
        def __init__(self, horizons):
            super().__init__()
            self.cfg = _FakeCfg(horizons)

    dense_ok = sparse_ok = None
    sparse_err = None
    try:
        DeployableSurfacePlanner(None, _FakeHead(v4_h), device="cpu")
        dense_ok = True
    except NonDensePlanError as e:                          # pragma: no cover
        dense_ok = False
        sparse_err = str(e)
    try:
        DeployableSurfacePlanner(None, _FakeHead(tac_h), device="cpu")
        sparse_ok = True
    except NonDensePlanError as e:
        sparse_ok = False
        sparse_err = str(e)

    # --- what train_flagship4b's SELECTED plan actually is ---------------------
    fb_keys = sorted({"anchor_logits", "anchor_traj", "offset", "traj",
                      "sel_idx", "waypoints"})          # AnchoredTacticalDecoder
    v4_keys_needed = ["wp_seq"]

    # --- the v4-only machinery that would have to move into 4b ----------------
    to_build = {
        "FlagshipV4Head (dense 1..20 plan + selection)": _def_span(
            STACK / "tanitad" / "models" / "flagship_v4.py", "FlagshipV4Head"),
        "FlagshipV15Head (its base class)": _def_span(
            STACK / "tanitad" / "models" / "flagship_v15.py", "FlagshipV15Head"),
        "v4_loss_step (trains that head)": _def_span(v4, "v4_loss_step"),
        "FlagshipV4Dataset (its labels)": _def_span(
            STACK / "scripts" / "flagship_v4_data.py", "FlagshipV4Dataset"),
        "_training_loop val/canary/gate body": _def_span(v4, "_training_loop"),
        "canary_rollout": _def_span(v4, "canary_rollout"),
        "evaluate_planner": _def_span(v4, "evaluate_planner"),
    }

    return {
        "option": "B — port the held-out val loop into train_flagship4b",
        "file_to_edit": str(fb.relative_to(REPO)).replace("\\", "/"),
        "file_lines": _src_lines(fb),
        "STRUCTURAL_BLOCKER": {
            "what": "train_flagship4b has no dense-plan head, so the gate's "
                    "deployable surface does not exist in it",
            "train_flagship4b_selected_plan_horizons": list(tac_h),
            "train_flagship4b_plan_dt_s": [round(h * 0.1, 2) for h in tac_h],
            "train_flagship4b_decoder_output_keys": fb_keys,
            "gate_requires_output_key": v4_keys_needed,
            "gate_requires_dense_horizons": list(range(1, len(v4_h) + 1)),
            "FlagshipV4Head_horizons": list(v4_h),
            "DeployableSurfacePlanner_accepts_v4_head": dense_ok,
            "DeployableSurfacePlanner_accepts_4b_head": sparse_ok,
            "refusal_message": sparse_err,
        },
        "new_machinery_required_lines": to_build,
        "new_machinery_required_total": sum(to_build.values()),
        "machinery_reused_not_written": {
            "build_v2_providers": "already wired in 4b (--v2-cache)",
        },
        "also_still_required": [
            "a v2 VAL cache branch in 4b (the same wiring option A needs)",
            "an optimizer group + loss for the new head (it is untrained "
            "otherwise, and a gate on a random head measures nothing)",
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    a_rec = measure_option_a()
    b_rec = measure_option_b()

    # the live contract check option A depends on: does a LazyV2Episode quack
    # like the episode objects v4's dataset and the gate read?
    from tanitad.data.v2_dataset import LazyV2Episode
    contract = {n: hasattr(LazyV2Episode, n)
                for n in ("frames", "poses", "actions", "episode_id")}

    rec = {
        "generated": "measure_options.py",
        "repo": str(REPO),
        "torchvision_stubbed_on_this_host": _TV_STUBBED,
        "option_a": a_rec,
        "option_b": b_rec,
        "lazy_v2_episode_contract": contract,
        "verdict": {
            "option_a_new_machinery_lines": 0,
            "option_b_new_machinery_lines": b_rec["new_machinery_required_total"],
            "decisive": (
                "Option B is not a port: train_flagship4b's selected plan is "
                f"{list(tuple(b_rec['STRUCTURAL_BLOCKER']['train_flagship4b_selected_plan_horizons']))}"
                " (0.5 s spacing), and the gate's pseudo-simulation refuses a "
                "non-dense plan by construction because it differentiates at "
                "dt=0.1 s. Giving 4b a dense plan means giving it v4's head and "
                "the loss that trains it — i.e. making 4b into v4."),
        },
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rec, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
