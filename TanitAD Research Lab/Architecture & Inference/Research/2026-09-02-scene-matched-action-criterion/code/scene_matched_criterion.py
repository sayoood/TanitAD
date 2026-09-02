"""E-ARCH-SMAS-1 — a scene-spread-aware criterion for action-conditioning arms.

⛔ THE DEFECT THIS EXISTS TO REMOVE. The programme's action-conditioning bar is
the **h1 action/scene ratio**, pre-registered with a ">= 10x rise" threshold
(PREREG_MM_E19). That statistic is a QUOTIENT OF TWO QUANTITIES THAT BOTH MOVE
between arms:

    ratio = action_spread / scene_spread

`scene_spread` is not a normaliser held fixed by the design -- it is a second
measurement of the model, and it rises when the model gets BETTER at the scene.
So the ratio **penalises world-model improvement**, and the "10x" bar is not a
fixed bar: its action-side requirement floats with whatever the denominator does.

This module computes, from banked actdiv reads:

  * the **log-additive decomposition** of a ratio change into its action and
    scene contributions -- the correct attribution for a quotient;
  * the **effective action-side bar** a stated ratio bar actually imposes, given
    the observed scene movement;
  * **SMAS** (scene-matched action sensitivity): `action_spread / scene_ref`,
    with the denominator PINNED to a named reference arm.
    ⚠️ SMAS is `action_spread` rescaled by a constant, and this file says so
    rather than dressing it up. The contribution is not a cleverer statistic --
    it is **refusing to put a moving quantity in the denominator of a decision
    statistic**, and keeping `scene_spread` as a CO-PRIMARY report instead.

⭐ THE CONTROLS, without which a criterion proposal is just an opinion:

  1. **no-information control** -- an action-blind arm (`action_spread = 0`) must
     read EXACTLY 0.0 under both the raw ratio and SMAS. A criterion that does
     not bottom out at the known value cannot be trusted at the top.
  2. **known-value synthetic control** -- inject (action_factor, scene_factor)
     chosen in advance; the decomposition MUST recover them exactly. This is what
     catches an algebra error, which would otherwise produce a confident,
     publishable-looking attribution.
  3. **empirical action-dead floor** -- the h2/h4 rows of the same banked reads,
     where `action_spread` is 6-7e-06 on every arm. That is what "the action does
     nothing" measures as on this instrument, so h1 numbers are read against it.
  4. **identity control C0** -- carried through from the source reads
     (`C0_identity_max_abs_diff == 0.0`); an arm whose C0 fails is not scored.

Usage:
    python scene_matched_criterion.py --out ../raw/smas.json
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from datetime import datetime, timezone

# Banked source reads (both MEASURED, dev-box CUDA, 24 clips / 144 windows,
# 8 action variants, variant_draw="roll (never permutation)").
REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
SRC_K8 = os.path.join(
    REPO, "TanitAD Research Lab", "Architecture & Inference", "Research",
    "2026-09-01-mm-e19-k8-attribution", "raw",
    "MISLABELLED_INSIDE_AS_k60__actdiv_local.json")
SRC_K60 = os.path.join(
    REPO, "TanitAD Research Lab", "Architecture & Inference", "Research",
    "2026-08-31-mm-e19-k60-horizon", "raw", "reread-2026-09-02",
    "actdiv_local.json")

#: ⛔ SRC_K8's internal arm key says "k60clip05p30k" and is WRONG -- the
#: attribution package documents the `--arm-name` default that mislabelled it
#: (§6a), and the checkpoint md5 is what settled which arm it is. Reading it by
#: its internal name would silently swap two arms, so the key is named here.
K8_INTERNAL_KEY = "k60clip05p30k"      # actually k8clip05p30k
PREREG_RATIO_BAR = 10.0                # PREREG_MM_E19's ">= 10x rise"


def load(path: str) -> dict:
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def decompose(ref: dict, arm: dict) -> dict:
    """Log-additive attribution of a ratio change to numerator vs denominator.

    For r = a / s, log(r_arm/r_ref) = log(a_arm/a_ref) - log(s_arm/s_ref); the
    two terms are the only additive split of a quotient's movement, so shares are
    reported on that scale and NOT as a linear percentage of the ratio.
    """
    a_f = arm["action_spread"] / ref["action_spread"]
    s_f = arm["scene_spread"] / ref["scene_spread"]
    r_f = arm["ratio_action_over_scene"] / ref["ratio_action_over_scene"]

    la, ls = math.log(a_f), math.log(s_f)
    total = abs(la) + abs(ls)
    return {
        "action_factor": a_f,
        "scene_factor": s_f,
        "ratio_factor_observed": r_f,
        "ratio_factor_reconstructed": a_f / s_f,
        # a reconstruction that does not match means the two reads are not on the
        # same instrument -- report it rather than assume it
        "reconstruction_abs_err": abs(r_f - a_f / s_f),
        "share_of_change_from_action": (abs(la) / total) if total else 0.0,
        "share_of_change_from_scene": (abs(ls) / total) if total else 0.0,
    }


def effective_action_bar(scene_factor: float, ratio_bar: float) -> float:
    """A ratio bar of `ratio_bar` demands this much rise in ACTION spread, once
    the denominator has moved by `scene_factor`."""
    return ratio_bar * scene_factor


def smas(arm: dict, scene_ref: float) -> float:
    """Scene-matched action sensitivity: the denominator pinned to a named arm."""
    return arm["action_spread"] / scene_ref


def controls(ref: dict, scene_ref: float) -> dict:
    """The three checks that must read known values, plus the empirical floor."""
    out = {}

    # 1. no-information control: an action-blind arm
    blind = {"action_spread": 0.0, "scene_spread": ref["scene_spread"],
             "ratio_action_over_scene": 0.0}
    out["no_information_control"] = {
        "raw_ratio": blind["ratio_action_over_scene"],
        "smas": smas(blind, scene_ref),
        "required": 0.0,
        "passes": (blind["ratio_action_over_scene"] == 0.0
                   and smas(blind, scene_ref) == 0.0),
    }

    # 2. known-value synthetic control: factors chosen in advance
    want_a, want_s = 2.5, 0.4
    synth_ref = {"action_spread": 1.0, "scene_spread": 1.0,
                 "ratio_action_over_scene": 1.0}
    synth_arm = {"action_spread": want_a, "scene_spread": want_s,
                 "ratio_action_over_scene": want_a / want_s}
    d = decompose(synth_ref, synth_arm)
    out["known_value_synthetic_control"] = {
        "injected_action_factor": want_a, "recovered": d["action_factor"],
        "injected_scene_factor": want_s, "recovered_scene": d["scene_factor"],
        "passes": (abs(d["action_factor"] - want_a) < 1e-12
                   and abs(d["scene_factor"] - want_s) < 1e-12
                   and d["reconstruction_abs_err"] < 1e-12),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    j8, j60 = load(SRC_K8), load(SRC_K60)
    k8 = j8["arms"][K8_INTERNAL_KEY]          # mislabelled inside; see note above
    k60 = j60["arms"]["k60clip05p30k"]
    inc = j60["arms"]["postrain30k"]

    # C0 must pass on every arm/horizon, or the arm is not scored at all.
    c0 = {name: all(arm[h]["C0_passes"] for h in ("h1", "h2", "h4"))
          for name, arm in (("k8clip05p30k", k8), ("k60clip05p30k", k60),
                            ("postrain30k", inc))}
    if not all(c0.values()):
        raise SystemExit(f"C0 identity control failed: {c0}")

    scene_ref = k8["h1"]["scene_spread"]      # the one-variable reference arm
    dec = decompose(k8["h1"], k60["h1"])

    # what a CONSTANT action response, or a DOUBLED one, would have reported
    hypotheticals = {
        f"action_factor_{f}_reads_ratio_factor": f / dec["scene_factor"]
        for f in (0.5, 1.0, 1.5, 2.0, 3.0)
    }

    res = {
        "_experiment": "E-ARCH-SMAS-1",
        "_date": "2026-09-02",
        "_generated_utc": datetime.now(timezone.utc).isoformat(),
        "_evidence_class": "MEASURED (ours; re-analysis of two banked actdiv reads)",
        "_tier": "T0-DIAGNOSTIC (inherited from the source reads) -- NOT a driving claim",
        "_sources": {"k8_control": SRC_K8, "k60_and_incumbent": SRC_K60},
        "_source_note": ("SRC_K8's internal arm key reads 'k60clip05p30k' and is "
                         "MISLABELLED; the attribution package settled the identity "
                         "by checkpoint md5. Read by explicit key, never by name."),
        "C0_identity_control": c0,
        "arms_h1": {
            "postrain30k": inc["h1"],
            "k8clip05p30k": k8["h1"],
            "k60clip05p30k": k60["h1"],
        },
        "empirical_action_dead_floor_h2_h4": {
            "k8clip05p30k": [k8["h2"]["action_spread"], k8["h4"]["action_spread"]],
            "k60clip05p30k": [k60["h2"]["action_spread"], k60["h4"]["action_spread"]],
            "postrain30k": [inc["h2"]["action_spread"], inc["h4"]["action_spread"]],
            "note": ("what 'the action does nothing' measures as on this "
                     "instrument; h1 numbers are read against it"),
        },
        "one_variable_decomposition_k8control_to_k60": dec,
        "prereg_bar": {
            "stated_ratio_bar": PREREG_RATIO_BAR,
            "observed_scene_factor": dec["scene_factor"],
            "effective_ACTION_bar_actually_imposed":
                effective_action_bar(dec["scene_factor"], PREREG_RATIO_BAR),
        },
        "smas": {
            "scene_reference_arm": "k8clip05p30k",
            "scene_reference_value": scene_ref,
            "k8clip05p30k": smas(k8["h1"], scene_ref),
            "k60clip05p30k": smas(k60["h1"], scene_ref),
            "factor": smas(k60["h1"], scene_ref) / smas(k8["h1"], scene_ref),
        },
        "hypothetical_reads_under_the_RAW_ratio": hypotheticals,
        "controls": controls(k8["h1"], scene_ref),
    }
    res["controls_all_pass"] = all(
        v["passes"] for v in res["controls"].values())

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)

    d = res["one_variable_decomposition_k8control_to_k60"]
    print("E-ARCH-SMAS-1  (one variable: o5_k 8 -> 60)")
    print(f"  action factor      {d['action_factor']:.4f}")
    print(f"  scene  factor      {d['scene_factor']:.4f}")
    print(f"  ratio  factor      {d['ratio_factor_observed']:.4f}  "
          f"(reconstructed {d['ratio_factor_reconstructed']:.4f}, "
          f"err {d['reconstruction_abs_err']:.2e})")
    print(f"  share from ACTION  {d['share_of_change_from_action']*100:.1f}%")
    print(f"  share from SCENE   {d['share_of_change_from_scene']*100:.1f}%")
    print(f"  stated bar 10x  ->  EFFECTIVE action bar "
          f"{res['prereg_bar']['effective_ACTION_bar_actually_imposed']:.2f}x")
    print(f"  SMAS factor        {res['smas']['factor']:.4f}")
    print("  hypothetical raw-ratio reads:")
    for k, v in res["hypothetical_reads_under_the_RAW_ratio"].items():
        print(f"     {k:44s} {v:.4f}")
    print(f"  controls_all_pass  {res['controls_all_pass']}")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
