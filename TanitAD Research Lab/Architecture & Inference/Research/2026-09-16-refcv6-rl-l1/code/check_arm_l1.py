"""Assert on a finished **L1 / D9** arm's ARTIFACTS (never on its exit code). Exit 0 = clean.

Everything ``…/2026-09-15-ddv2-rl-prep/code/check_arm.py`` checks (I1 present + finite +
parameters moved, I2 the NORL coefficient is exactly zero with the logged residual inside
float32 round-off, I3 a positive advantage on >= 90 % of steps, the artifacts exist), PLUS the
four known-value checks this lever needs (PREREG §12):

* **I7 — THE LEVER IS ON, AND IS THE ONE CLAIMED.** ``run.json``'s ``il.form`` /
  ``il.lambda_scale`` / ``grad_clip`` match what was asked for, and EVERY logged step agrees
  with them. A silently-release arm reporting a matched-anchor result is the failure mode this
  whole package exists to avoid.
* **I8 — THE MATCHED TERM IS THE ONE THAT DROVE.** On a ``matched_anchor`` arm the logged
  ``il_mean_m`` equals ``il_matched_anchor_m`` and NOT ``il_all_modes_m`` (they differ by
  construction: the matched anchor is the nearest by definition).
* **I9 — THE MATCH IS NOT DEGENERATE.** ``match_modal_frac`` < 1.0 on some step, i.e. the
  batch's windows do not all match one anchor. If they did, "mode preserving" would be a label
  on a term that supervises one fixed mode forever, which is a different object.
* **I10 — CLIPPING IS REAL AND IS NOT EVERYTHING.** With the clip on, no step's
  ``grad_norm_clipped`` exceeds the max-norm; and the clip is reported as binding on at least
  one step **or** the run's maximum pre-clip norm stayed under it (so "never clipped" is a
  measured statement, not a silently dead flag).

⛔ EVERY CHECK HERE IS PROVEN REACHABLE BY MUTATION in
``stack/tests/test_ddv2_il.py::test_MUTATION_each_L1_integrity_check_fires_on_its_own_defect`` —
a checker that cannot fail has never passed anything.

Usage::

    python check_arm_l1.py <run_dir> --steps 600 --arm rl --il-form matched --grad-clip 100
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

#: ``--il-form`` -> the ``il.form`` / ``lambda_scale`` the run record must carry.
#: MUST stay identical to ``stack/scripts/ddv2_rl_refcv5.py:IL_FORM_FLAGS``; pinned by
#: ``test_the_checker_and_the_runner_agree_on_what_each_flag_means``.
IL_FORM_FLAGS = {"release": ("release_all_modes", 1.0),
                 "matched": ("matched_anchor", 1.0),
                 "lambda": ("release_all_modes_lambda", 0.1)}

NUM_KEYS = ("loss", "rl_part", "il_mean_m", "grad_norm", "param_delta_norm", "reward_mean")
ARTIFACTS = (("ckpt.pt", 10 ** 9), ("config.json", 1000), ("run.json", 100))


def load(run_dir: str) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    mpath = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    run: dict = {}
    rpath = os.path.join(run_dir, "run.json")
    if os.path.exists(rpath):
        with open(rpath, encoding="utf-8") as fh:
            run = json.load(fh)
    return rows, run


def check(rows: list[dict], run: dict, *, steps: int, arm: str, il_form: str,
          grad_clip: float, run_dir: str | None = None) -> list[str]:
    """-> the list of problems; empty == every check holds."""
    bad: list[str] = []
    want_form, want_scale = IL_FORM_FLAGS[il_form]

    # ---- I1 ----------------------------------------------------------------
    if len(rows) != steps:
        bad.append(f"I1 metrics rows {len(rows)} != steps {steps}")
    nonfinite = [r["step"] for r in rows
                 if not all(math.isfinite(float(r[k])) for k in NUM_KEYS)]
    if nonfinite:
        bad.append(f"I1 non-finite at steps {nonfinite[:5]}")
    if rows and not rows[-1]["param_delta_norm"] > 0:
        bad.append("I1 parameters did not move")

    # ---- I2 / I3 (unchanged from the 2026-09-15 checker, D-1 amended form) --
    if arm == "norl":
        coef_nz = [r["step"] for r in rows if r["rl_coef_abs_sum"] != 0.0]
        if coef_nz:
            bad.append(f"I2 NORL policy-gradient COEFFICIENT non-zero at steps {coef_nz[:5]}")
        roundoff = [r["step"] for r in rows
                    if abs(r["rl_part"]) > 1e-6 * max(1.0, abs(r["loss"]))]
        if roundoff:
            bad.append(f"I2 NORL logged rl_part above float32 round-off at steps {roundoff[:5]}")
    else:
        pos = sum(1 for r in rows if r["frac_positive_after_bar"] > 0)
        if rows and pos / len(rows) < 0.90:
            bad.append(f"I3 positive advantage on only {pos}/{len(rows)} steps")

    # ---- I7 the lever is on, and is the one claimed ------------------------
    rec = run.get("il") or {}
    if rec.get("form") != want_form:
        bad.append(f"I7 run.json il.form {rec.get('form')!r} != {want_form!r}")
    if rec.get("lambda_scale") != want_scale:
        bad.append(f"I7 run.json il.lambda_scale {rec.get('lambda_scale')} != {want_scale}")
    want_clip = None if float(grad_clip) == 0.0 else float(grad_clip)
    if run.get("grad_clip") != want_clip:
        bad.append(f"I7 run.json grad_clip {run.get('grad_clip')} != {want_clip}")
    off = [r["step"] for r in rows
           if r.get("il_form") != want_form or r.get("il_lambda_scale") != want_scale
           or r.get("grad_clip") != want_clip]
    if off:
        bad.append(f"I7 logged steps disagree with the declared lever at {off[:5]}")

    # ---- I8 the matched term is the one that drove -------------------------
    if want_form == "matched_anchor":
        wrong = [r["step"] for r in rows
                 if abs(r["il_mean_m"] - r["il_matched_anchor_m"]) > 1e-6 * max(1.0, r["il_mean_m"])]
        if wrong:
            bad.append(f"I8 il_mean_m is not the MATCHED term at steps {wrong[:5]}")
        same = [r["step"] for r in rows
                if abs(r["il_all_modes_m"] - r["il_matched_anchor_m"]) <= 1e-9]
        if rows and len(same) == len(rows):
            bad.append("I8 the two IL statistics are identical on every step — the matched "
                       "gather is selecting all chains, or N was read as 1")
    else:
        wrong = [r["step"] for r in rows
                 if abs(r["il_mean_m"] - r["il_all_modes_m"]) > 1e-6 * max(1.0, r["il_mean_m"])]
        if wrong:
            bad.append(f"I8 il_mean_m is not the ALL-MODES term at steps {wrong[:5]}")

    # ---- I9 the match is not degenerate ------------------------------------
    if rows and all(r.get("match_modal_frac", 1.0) >= 1.0 for r in rows):
        bad.append("I9 every batch matched a single anchor on every step — the match is "
                   "degenerate and 'mode preserving' would be a label, not a mechanism")

    # ---- I10 clipping is real and is not everything ------------------------
    if want_clip:
        over = [r["step"] for r in rows if r["grad_norm_clipped"] > want_clip * (1 + 1e-5)]
        if over:
            bad.append(f"I10 grad_norm_clipped above the max-norm at steps {over[:5]}")
        clipped = sum(1 for r in rows if r.get("grad_clipped"))
        peak = max((r["grad_norm"] for r in rows), default=0.0)
        if rows and clipped == 0 and peak > want_clip:
            bad.append(f"I10 no step reports clipping although the peak pre-clip norm was "
                       f"{peak:.3f} > {want_clip} — the clip is a dead flag")
    else:
        if any(r.get("grad_clipped") for r in rows):
            bad.append("I10 a run declared UNCLIPPED reports clipped steps")

    if run_dir is not None:
        for f, minsize in ARTIFACTS:
            p = os.path.join(run_dir, f)
            if not os.path.exists(p) or os.path.getsize(p) < minsize:
                bad.append(f"{f} missing or too small")
    return bad


def summarise(rows: list[dict], run: dict, bad: list[str], run_dir: str) -> dict:
    last = rows[-1] if rows else {}
    return {
        "run_dir": os.path.basename(run_dir), "rows": len(rows),
        "il": run.get("il"), "grad_clip": run.get("grad_clip"),
        "first_reward": rows[0]["reward_mean"] if rows else None,
        "last_reward": last.get("reward_mean"),
        "il_mean_m_first_last": [rows[0]["il_mean_m"], last.get("il_mean_m")] if rows else None,
        "chain_spread_m_first_last": ([rows[0]["chain_endpoint_spread_m"],
                                       last.get("chain_endpoint_spread_m")] if rows else None),
        "grad_norm_max": max((r["grad_norm"] for r in rows), default=None),
        "steps_clipped": sum(1 for r in rows if r.get("grad_clipped")),
        "match_distinct_anchors_max": max((r.get("match_n_distinct_anchors_matched", 0)
                                           for r in rows), default=None),
        "last_delta": last.get("param_delta_norm"),
        "human_nc_eq_1_frac_pooled": ((sum(r["human_nc_eq_1_frac"] for r in rows) / len(rows))
                                      if rows else None),
        "problems": bad,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--arm", choices=("rl", "norl"), required=True)
    ap.add_argument("--il-form", choices=tuple(IL_FORM_FLAGS), required=True)
    ap.add_argument("--grad-clip", type=float, required=True)
    a = ap.parse_args(argv)
    rows, run = load(a.run_dir)
    bad = check(rows, run, steps=a.steps, arm=a.arm, il_form=a.il_form,
                grad_clip=a.grad_clip, run_dir=a.run_dir)
    print(json.dumps(summarise(rows, run, bad, a.run_dir)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
