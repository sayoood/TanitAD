"""RUNG A1 / DELIVERABLE 1 — port ``ha0_ext`` into the REF-C harness.

Applies to ONE tree (``--root <repo-or-mirror-root>``). Idempotent: re-running
on an already-patched tree exits 0 with "already applied". Every replacement is
an EXACT-MATCH assertion, so a drifted source REFUSES rather than half-patching.

Why a script and not hand edits: the G: mount drops individual files with
errno 22 for minutes at a time, so the edit must be replayable and verifiable
by read-back rather than trusted once.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# refav1_arm.py — generalise the ONE ha0_ext derivation with an explicit stride #
# --------------------------------------------------------------------------- #
RA_OLD = '''def hold_ext_controls(loader, v, kap, t: int, *, dt: float | None = None):
    """``ha0_ext`` (refav1 form): the (a0, kappa0) of the MEASURED t0 state.

    ``a0 = (v[2t] - v[2t-2]) / dt`` — the SAME backward difference
    `hold_action_controls` holds (the forward one, ``v[2t+2]``, is future);
    ``kappa0 = kap[2t]`` — the recorded curvature AT t0, where ``ha`` holds
    ``kap[2t-2]`` (`echo_gate.ha_finite_diff_accel`'s "reads the steer at
    t0-1" weakness, closed). Every index is <= 2t: nothing recorded after t0
    enters. ⚠️ `echo_gate.ha0_ext`'s STRONGER form is built on the corpus's own
    measured ``ax``; the v2ep carries none, so this is the sharpest ADMISSIBLE
    form on these inputs, and `ARM_MEANING` names it as such.
    """
    if t < 1:
        raise ValueError("ha0_ext needs t >= 1 (one closed step before t0)")
    import torch
    dt = float(dt if dt is not None else loader.dt)
    a0 = (v[2 * t] - v[2 * t - 2]) / dt
    return torch.stack([a0, kap[2 * t]]).float()                 # [2]
'''

RA_NEW = '''def hold_ext_controls(loader, v, kap, t: int, *, dt: float | None = None,
                      stride: int = 2):
    """``ha0_ext`` (refav1 form): the (a0, kappa0) of the MEASURED t0 state.

    ``a0 = (v[i0] - v[i0-stride]) / dt`` — the SAME backward difference
    `hold_action_controls` holds (the forward one, ``v[i0+stride]``, is future);
    ``kappa0 = kap[i0]`` — the recorded curvature AT t0, where ``ha`` holds
    ``kap[i0-stride]`` (`echo_gate.ha_finite_diff_accel`'s "reads the steer at
    t0-1" weakness, closed). Every index is <= i0: nothing recorded after t0
    enters. ⚠️ `echo_gate.ha0_ext`'s STRONGER form is built on the corpus's own
    measured ``ax``; the v2ep carries none, so this is the sharpest ADMISSIBLE
    form on these inputs, and `ARM_MEANING` names it as such.

    ⭐ ``stride`` EXISTS SO THERE IS ONE IMPLEMENTATION, NOT TWO (Rung A1,
    2026-09-05). refav1 windows a 0.1 s pose array on its own 0.2 s operative
    tick, so its origin is ``i0 = 2t`` — the DEFAULT, and with it this function
    is bit-identical to the pre-Rung-A1 file. ``refcv3_arm.py`` indexes the same
    array at the native 0.1 s tick and calls this with ``stride=1, dt=0.1``,
    passing ``loader=None`` (the loader is read ONLY for its ``dt``). A control
    re-implemented beside the harness that uses it is a control that can drift
    away from it, and then the gate measures the drift instead of the model —
    `echo_gate.ha0_ext`'s own docstring says exactly this about its shared base.
    Pinned by ``stack/tests/test_refcv3_ha0_ext_shared.py``.
    """
    stride = int(stride)
    i0 = stride * int(t)
    if i0 < stride:
        raise ValueError("ha0_ext needs one closed step before t0 "
                         f"(stride={stride}, t={t} -> i0={i0})")
    import torch
    dt = float(dt if dt is not None else loader.dt)
    a0 = (v[i0] - v[i0 - stride]) / dt
    return torch.stack([a0, kap[i0]]).float()                    # [2]
'''

# --------------------------------------------------------------------------- #
# refcv3_arm.py                                                                #
# --------------------------------------------------------------------------- #
RC_EDITS: list[tuple[str, str, str]] = [
    # ---- (1) the tier stamp ------------------------------------------------
    (
        "ARM_TIERS",
        '''ARM_TIERS = {"os": "T1", "os_navshuf": "T1", "os_navzero": "T1", "ha": "T1",
             "ha0": "T1", "oracle_sel": "T0", "os_navflip": "T1"}''',
        '''ARM_TIERS = {"os": "T1", "os_navshuf": "T1", "os_navzero": "T1", "ha": "T1",
             "ha0": "T1", "ha0_ext": "T1", "oracle_sel": "T0",
             "os_navflip": "T1"}''',
    ),
    # ---- (2) ARM_MEANING: add ha0_ext AND apply echo_gate.py's retraction ---
    (
        "ARM_MEANING",
        '''    "ha0": "T1 — ⭐ CONSTANT VELOCITY: a = 0, kappa = 0 at the measured v0, i.e. "
           "a straight line at constant speed. The STRONGEST TRIVIAL BASELINE, "
           "the echo test's real bar, and the ONLY arm bit-comparable with "
           "refav1's (zero is zero in either action unit).",''',
        '''    "ha0": "T1 — ⭐ CONSTANT VELOCITY: a = 0, kappa = 0 at the measured v0, i.e. "
           "a straight line at constant speed. The STRONGEST TRIVIAL BASELINE "
           "and the ONLY arm bit-comparable with refav1's (zero is zero in "
           "either action unit). ⛔ NOT 'the echo test's real bar' — that line "
           "stood here and stack/tanitad/eval/echo_gate.py RETRACTS it by name: "
           "`ha` is 2.2x harder, and the bar is `ha` AND `ha0_ext` TOGETHER.",
    "ha0_ext": "T1 — ⭐⭐ THE ECHO CONTROL: constant a0 AND constant curvature "
               "k0, both read at the MEASURED t0 (a0 the backward difference "
               "of v, k0 the recorded channel AT t0 — where `ha` holds the one "
               "at t0-1). The SHARPENED form of `ha`, and the OTHER HALF of the "
               "acceptance bar: beating `ha` while TYING `ha0_ext` means the "
               "gain is echo, and the gate reads it that way "
               "(stack/tanitad/eval/echo_gate.py::ha0_ext). Derived by the ONE "
               "shared implementation refav1_arm.hold_ext_controls, called here "
               "with stride=1 — never re-derived (Rung A1, 2026-09-05).",''',
    ),
    # ---- (3) the arm list --------------------------------------------------
    (
        "arms list",
        '''    arms = ["os", "ha", "ha0"]''',
        '''    # ⭐ `ha0_ext` is NOT optional: refcv5's acceptance bar is "beat BOTH `ha`
    # AND `ha0_ext`", and until Rung A1 this harness computed neither the arm
    # nor the number, so half the bar was unreadable on the REF-C surface.
    arms = ["os", "ha", "ha0", "ha0_ext"]''',
    ),
    # ---- (4) the roll ------------------------------------------------------
    (
        "roll",
        '''            ha0 = integrate_select(hold_v0_controls(n_f), v0, grid,
                                   action_units=units)''',
        '''            ha0 = integrate_select(hold_v0_controls(n_f), v0, grid,
                                   action_units=units)
            # ⭐⭐ ha0_ext: THE ECHO CONTROL, and the other half of the bar.
            # ⛔ The (a0, k0) derivation is NOT written here. It is
            # `refav1_arm.hold_ext_controls` — the SAME call the refav1 harness
            # makes — invoked at this corpus's native 0.1 s tick (stride=1);
            # `loader=None` because that argument is read only for its `dt`,
            # which is passed explicitly. A second implementation of a shared
            # kinematic is how two "independent" checks come to agree on a
            # wrong answer (echo_gate.ha0_ext's own docstring).
            ext = ra.hold_ext_controls(None, v_ep, kap_ep, t0,
                                       dt=DT_FRAME, stride=1)
            ha0_ext = integrate_select(ext[None].expand(n_f, 2), v0, grid,
                                       action_units=units)''',
    ),
    (
        "accumulate",
        '''            acc["ha"].append(ha)
            acc["ha0"].append(ha0)''',
        '''            acc["ha"].append(ha)
            acc["ha0"].append(ha0)
            acc["ha0_ext"].append(ha0_ext)''',
    ),
    # ---- (5) the manifest: the rule travels with the number ----------------
    (
        "manifest rules",
        '''        "hold_action_rule": hold_controls.__doc__,
        "hold_v0_rule": hold_v0_controls.__doc__,''',
        '''        "hold_action_rule": hold_controls.__doc__,
        "hold_v0_rule": hold_v0_controls.__doc__,
        # ⭐ the SHARED derivation's own docstring, not a paraphrase — so the
        # record names the one implementation both harnesses call.
        "ha0_ext_rule": ra.hold_ext_controls.__doc__,
        "ha0_ext_call": ("refav1_arm.hold_ext_controls(None, v_ep, kap_ep, t0, "
                         "dt=0.1, stride=1) -> integrate_select — the SAME "
                         "function refav1_arm.py calls at stride=2; pinned "
                         "bit-equal by "
                         "stack/tests/test_refcv3_ha0_ext_shared.py"),''',
    ),
    (
        "manifest units",
        '''            "applies_to": ["ha"],''',
        '''            # `ha0_ext` holds a RECORDED channel-1 value too, so the unit
            # conversion applies to it exactly as it does to `ha`.
            "applies_to": ["ha", "ha0_ext"],''',
    ),
    # ---- (6) the paired blocks: the bar becomes readable --------------------
    (
        "pairs",
        '''        ("ha0", "ha", "paired_ha_minus_ha0"),
        ("os", "oracle_sel", "paired_oraclesel_minus_os"))''',
        '''        ("ha0", "ha", "paired_ha_minus_ha0"),
        # ⭐⭐ THE OTHER HALF OF refcv5's ACCEPTANCE BAR. `os - ha0_ext` is the
        # margin over the ego-extrapolation echo; an arm that beats `ha` while
        # TYING this one has echoed, not driven (echo_gate.ha0_ext). The
        # nav-zero pairing is the DEPLOYMENT-relevant form of the same read.
        ("ha0_ext", "os", "paired_os_minus_ha0ext"),
        ("ha0_ext", "os_navzero", "paired_os_navzero_minus_ha0ext"),
        ("os", "oracle_sel", "paired_oraclesel_minus_os"))''',
    ),
    # ---- (7) the floor note + the ha0_ext ≡ ha0 diagnostic -----------------
    (
        "floor note",
        '''    triv["degenerate_arms_excluding_floor"] = [
        x for x in triv["degenerate_arms"] if x != "ha0"]
    triv["_floor_note"] = ("`ha0` is expected in degenerate_arms — it IS the "
                           "constant-velocity plan. VOID-RISK is raised only on "
                           "degenerate_arms_excluding_floor.")''',
        '''    triv["degenerate_arms_excluding_floor"] = [
        x for x in triv["degenerate_arms"] if x not in ("ha0", "ha0_ext")]
    triv["_floor_note"] = (
        "`ha0` is expected in degenerate_arms — it IS the constant-velocity "
        "plan. `ha0_ext` is excluded for a DIFFERENT reason: it is a "
        "constant-(a0, k0) extrapolation, so its trivial fraction is a "
        "property of HOW STRAIGHT AND STEADY THE EGO WAS AT t0 on this corpus, "
        "not a defect of the arm — escalating on it would make the loudest "
        "warning in this tool fire on a control and be learned as noise. Both "
        "stay IN degenerate_arms and are read through `ha0_ext_vs_ha0` below. "
        "VOID-RISK is raised only on degenerate_arms_excluding_floor.")
    # ⭐ THE READ THAT MATTERS FOR THE BAR: if `ha0_ext` is bit-identical to
    # `ha0` on most windows then "beat BOTH `ha` and `ha0_ext`" has collapsed
    # into "beat `ha0`", and the echo control is adding nothing on this surface.
    # Say so as a number rather than letting a reader assume two controls.
    _ext = (triv["arms"].get("ha0_ext") or {})
    _ext_vs_ha0 = (_ext.get("identical_to") or {}).get("ha0")
    triv["ha0_ext_vs_ha0"] = {
        "identical_frac": (None if _ext_vs_ha0 is None
                           else float(_ext_vs_ha0["frac"])),
        "n_identical": None if _ext_vs_ha0 is None else int(_ext_vs_ha0["n"]),
        "n": _ext.get("n"),
        "means": ("the ECHO control and the CONSTANT-VELOCITY floor are the "
                  "same path on that fraction of windows; on those windows the "
                  "acceptance bar has ONE control in it, not two"),
    }
    if _ext_vs_ha0 is not None:
        _p(f"  ha0_ext ≡ ha0 on {_ext_vs_ha0['n']}/{_ext.get('n')} windows "
           f"({_ext_vs_ha0['frac']:.2%}) — on those the echo control adds "
           f"nothing beyond the constant-velocity floor, and 'beat BOTH' "
           f"collapses to 'beat ha0'.")''',
    ),
]


def _apply(path: Path, edits, label: str) -> str:
    raw = path.read_bytes()
    if b"\r" in raw:
        raise SystemExit(f"[{label}] REFUSING: {path} contains CR — this "
                         f"patcher round-trips through universal newlines and "
                         f"would rewrite every line ending")
    src = path.read_text(encoding="utf-8")
    done, todo = [], []
    for name, old, new in edits:
        if new in src:
            done.append(name)
        elif old in src:
            todo.append((name, old, new))
        else:
            raise SystemExit(
                f"[{label}] REFUSING: anchor {name!r} not found and its "
                f"replacement is not present either — the source has drifted. "
                f"Fix the anchor, never the file.")
    if not todo:
        return f"[{label}] already applied ({len(done)}/{len(edits)})"
    for name, old, new in todo:
        if src.count(old) != 1:
            raise SystemExit(f"[{label}] REFUSING: anchor {name!r} occurs "
                             f"{src.count(old)} times, need exactly 1")
        src = src.replace(old, new, 1)
    path.write_text(src, encoding="utf-8", newline="")
    back = path.read_text(encoding="utf-8")          # READ-BACK VERIFICATION
    missing = [n for n, _o, nw in edits if nw not in back]
    if missing:
        raise SystemExit(f"[{label}] read-back FAILED for {missing}")
    return (f"[{label}] applied {len(todo)} edit(s) "
            f"({[n for n, _, _ in todo]}), {len(done)} already present; "
            f"read-back OK, {len(back)} chars")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    print(_apply(root / "taniteval" / "tools" / "refav1_arm.py",
                 [("hold_ext_controls stride", RA_OLD, RA_NEW)], "refav1_arm"))
    print(_apply(root / "taniteval" / "tools" / "refcv3_arm.py",
                 RC_EDITS, "refcv3_arm"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
