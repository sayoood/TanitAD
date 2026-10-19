"""r7m_patch.py -- run the refcv8 battery wrapper (`run_battery_r8.py`) on a REFCV7 checkpoint, with the refcv7 REMAP.

(EvalFlyWheel, PREREG_R7_MATCHED_PHASE_TURN.md, 2026-10-18.)  NEW file; nothing that a running script imports is edited.

WHY.  `r8_repoint.REMAP_OVERRIDES` (`refcv8_eval_remap_devbox.json`) re-points SEVEN eval-side flags at the REFCV8 files
(extrinsics, clip-clock sidecar `ts33`, vis1 sidecar, nav-compliance tau file, v9 labels, pose-sync sidecar, defect masks).  A refcv7
argv names the refcv6/7 files for the first four, and the banked refcv7-50,400 roll used the LOADER DEFAULTS for them (its roll record:
"Thor data paths REMAPPED to the eval kit D:\\refcv6_eval_kit for ['--agent-join', '--agent-rig-extrinsics', '--anchors',
'--clip-clock-sidecar', '--eval-cache', '--eval-labels', '--join3d', '--map-gt-root', '--map-hires-class-weights',
'--nav-compliance-tau-file', '--speed-max-sidecar-v6-eval', '--vis1-sidecar']"; no override file).  Feeding refcv7 the refcv8 files would
change its eval inputs, so the refcv7 pass uses an EMPTY override file: the loader defaults, i.e. exactly what the banked roll used.

HOW.  `install()` (called first in the parent AND in the roll child, because monkey-patches are per process):
  1. `r8_repoint.REMAP_OVERRIDES` := `refcv7_eval_remap_none.json` (`{}`), read by `configure_env()` / `describe()` at call time;
  2. `r8_battery_patch.rewrite_child_cmd` := the original, then `roll_seed_r8.py` -> `roll_seed_r7m.py` (the child that calls install()).
A refcv7 argv carries no `--pose-sync-sidecar`, so `r8_repoint.patch_loader` is a pass-through; the seam-guard patch returns the SAME tensor
(it only drops a training alarm's raise); the micro-batch patch only adds kwargs refcv7 never passes.  The known-value control
(`compare_r7_control.py`) is what shows that this really reproduces the banked refcv7-50,400 numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NONE_JSON = HERE / "refcv7_eval_remap_none.json"
CHILD = HERE / "roll_seed_r7m.py"
_DONE = {"v": False}


def install() -> None:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    if _DONE["v"]:
        return
    import r8_repoint
    import r8_battery_patch as BP
    r8_repoint.REMAP_OVERRIDES = NONE_JSON
    orig = BP.rewrite_child_cmd

    def rewrite_child_cmd(cmd, **kw):
        out = [str(c) for c in orig(cmd, **kw)]
        for i, c in enumerate(out):
            if c.replace("\\", "/").endswith("/roll_seed_r8.py"):
                out[i] = str(CHILD)
                BP._REC["child_rewrites"].append({"from": c, "to": str(CHILD), "by": "r7m_patch"})
                break
        return out

    BP.rewrite_child_cmd = rewrite_child_cmd
    _DONE["v"] = True


def disable_ceiling() -> None:
    """DIAGNOSTIC / candidate path (`R7M_CEIL_OFF=1`): switch the max-speed CEILING FILTER off in the main forward.

    MEASURED 2026-10-18 (KNOWN_VALUE_CONTROL): the banked refcv7-50,400 roll has the ceiling filter INERT (filter-on == filter-off in 4,754 / 4,754
    windows, `refcv6_extras.npz`), the launch-tree path has it ACTING (36 / 137 windows on the 4-clip subset).  This sets
    `speed_ceiling_filter = False` on every module that carries it, right after `refcv7_loader.build_model` -- exactly what the Refcv6Proxy's own
    `filter_off` forward does for one call.  Must run AFTER `r8_battery_patch.apply_child()` (the loader reads its environment at import time).
    The proxy then finds no filter module and records no `filter_off` arm (its `filteroff` flag is `... and bool(filter_mods)`)."""
    from tanitad.eval import refcv7_loader as L
    if getattr(L.build_model, "_r7m_ceil_off", False):
        return
    orig = L.build_model

    def build_model(*a, **k):
        m, c, ar, r = orig(*a, **k)
        n = 0
        for mod in m.modules():
            if isinstance(getattr(mod, "speed_ceiling_filter", None), bool) and mod.speed_ceiling_filter:
                mod.speed_ceiling_filter = False
                n += 1
        r["r7m_ceiling_off"] = {"modules_switched_off": n, "env": "R7M_CEIL_OFF=1"}
        print(f"[r7m] speed-ceiling filter switched OFF on {n} module(s)", flush=True)
        return m, c, ar, r

    build_model._r7m_ceil_off = True
    L.build_model = build_model


def disable_e9_ceiling() -> None:
    """The MECHANISM-FAITHFUL emulation of the banked refcv7 roll (`R7M_E9_NO_CEIL=1`): E9's final re-rank does not apply the ceiling mask.

    MEASURED 2026-10-18 (KNOWN_VALUE_CONTROL*): in the banked refcv7-50,400 roll the final selection `sel_idx` equals the filter-OFF selection in 4,754 / 4,754
    windows while the core's pre-graft `sel_idx_base` HONOURS the ceiling (the launch-tree path with the filter ON reproduces `sel_idx_base` exactly but changes
    `sel_idx` in 26 % of the smoke windows; with the filter OFF everywhere `sel_idx` matches but `sel_idx_base` does not).  The launch tree's `e9_rank`
    (refc_v3.py:1025-1043) applies `out["ceil_keep"]` (SPEC_REFCV7 A2: "exported so E9 applies the same ceiling", refc.py:3863/4016); the code that rolled the banked
    dump evidently did not.  This drops `ceil_keep` from `out` for the E9 call only, so the core keeps its ceiling and E9 ignores it.  UNVERIFIED against the
    banked roll's own source (its tree no longer exists); the known-value control is the test.  Must run after `apply_child()`."""
    from tanitad.refs import refc_v3 as V3
    if getattr(V3.e9_rank, "_r7m_no_ceil", False):
        return
    orig = V3.e9_rank

    def e9_rank(blended, out):
        if "ceil_keep" in out:
            out = {k: v for k, v in out.items() if k != "ceil_keep"}
        return orig(blended, out)

    e9_rank._r7m_no_ceil = True
    V3.e9_rank = e9_rank
    print("[r7m] E9 re-rank: ceil_keep ignored (R7M_E9_NO_CEIL=1)", flush=True)
