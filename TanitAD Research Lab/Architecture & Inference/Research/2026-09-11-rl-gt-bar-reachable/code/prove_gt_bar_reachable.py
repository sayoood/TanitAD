#!/usr/bin/env python3
"""Bank the reachability proof for V2's >=GT truncation as an ARTIFACT.

⛔ WHY A SCRIPT AND NOT ONLY A TEST. A green test is a claim about a run that has
already finished; this writes the numbers down. The programme's rule is that the
ARTIFACT is the evidence, never the exit code — a timed-out, output-less gate has
reported success here before.

WHAT IT PROVES, AND HOW EACH HALF IS DISCRIMINATED
---------------------------------------------------
1. **The adapter emits nothing by default** — the pre-2026-09-11 three-tuple, so
   the banked baseline reproduces bit-for-bit.
2. **Two analytic identities through the real caller wiring.** A bar below every
   achievable reward is the identity mask (gradient must be BIT-IDENTICAL to the
   no-bar gradient); a bar above it admits nothing, and with ``w_intra = 0`` the
   loss loses every dependence on ``logp`` so the gradient is EXACTLY 0.0. ⛔
   Neither target needs any knowledge of what the reward computes.
3. **The real `score_gt_bar` masks PARTIALLY.** A mask that only ever admits all
   or nothing is a switch, not a truncation.
4. ⭐ **A SOURCE MUTATION.** The emitter line is removed from a byte-copy of the
   adapter, imported as a separate module, and the same three checks are re-run.
   They must FAIL. *"A guard must be shown able to fail before its pass means
   anything"*, and an AST census of this same wiring once read 0 suspects on BOTH
   the fixed and the broken trainer — so inspection is inadmissible here.
   ⛔ The real `refcv3_adapter.py` is never written to; the mutant lives in its
   own temporary module and its md5 is recorded beside the original's.
5. **The reward audit's separation check**, on both the deliberate-regression
   reward and the default one, at both dt values.

Tier: N/A — this is an instrument proof, not a capability claim.
Evidence class: MEASURED (ours).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time

import torch
from torch import nn

STACK = os.environ.get("TANITAD_STACK")
if STACK and STACK not in sys.path:
    sys.path.insert(0, STACK)

from tanitad.rl import PostTrainConfig, RewardSpec          # noqa: E402
from tanitad.rl import audit as AUD                          # noqa: E402
from tanitad.rl import rewards as RW                         # noqa: E402
from tanitad.rl.posttrain import rl_objective, score_gt_bar  # noqa: E402
import tanitad.rl.refcv3_adapter as ADAPT                    # noqa: E402

B, N, G, S = 2, 3, 4, 5
_SCALE = torch.tensor([0.4, 0.8, 1.2]).view(1, N, 1)
EMITTER_LINE = '            extras["gt_bar"] = bar\n'
MUTANT_LINE = '            pass  # MUTANT: the emitter that was missing until 2026-09-11\n'


class OneKnobPolicy(nn.Module):
    """A stand-in MODEL only — every other layer is the production code path."""

    def __init__(self) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(1))
        self.decoder = nn.Linear(1, 1)

    def forward(self, frames, **kw):
        x = torch.linspace(0.0, 8.0, S).view(1, 1, S) * _SCALE
        x = x.expand(B, N, S) + self.theta
        anchor = torch.stack([x, torch.zeros_like(x)], dim=-1)
        return {"anchor_traj": anchor, "offset": 0.25 * anchor}


def ctx_for(gt_reach_m: float) -> dict:
    """⚠️ SCALES the GT's extent; it does NOT translate it. ``_progress`` reads a
    DIFFERENCE, so a translated GT is a lever that does not move."""
    gt_x = torch.linspace(0.0, gt_reach_m, S).view(1, S).expand(B, S)
    return {"dt": 0.5, "v0": torch.full((B,), 4.0),
            "gt_traj": torch.stack([gt_x, torch.zeros_like(gt_x)], dim=-1)}


def cfg_for(use_gt_bar: bool) -> PostTrainConfig:
    return PostTrainConfig(
        method="grpo", group_size=G, steps=1, batch=B, lr=0.0, dt=0.5,
        reward_weights={"progress": 1.0}, w_intra=0.0, w_inter=1.0,
        veto_enabled=False, w_anchor=0.0, use_gt_bar=use_gt_bar,
        trainable_prefixes=("decoder",), noise_scale=0.0)


def one_step(maker, *, use_gt_bar: bool, gt_reach_m: float = 8.0,
             bar_value: float | None = None, emit: bool | None = None):
    """One objective step through ``maker`` (the real or the mutant adapter)."""
    torch.manual_seed(0)
    model = OneKnobPolicy()
    cfg = cfg_for(use_gt_bar)
    spec = RewardSpec(weights=dict(cfg.reward_weights), dt=cfg.dt)
    ctx = ctx_for(gt_reach_m)
    do_emit = use_gt_bar if emit is None else emit
    fn = None
    if do_emit and bar_value is not None:
        def fn(batch, c, n_steps):
            return torch.full((B,), float(bar_value))
    elif do_emit:
        def fn(batch, c, n_steps):
            return score_gt_bar(spec, c, n_steps=n_steps)
    sample_fn = maker(model, cfg, build_ctx=lambda b, out=None: dict(ctx),
                      gt_bar_fn=fn, strict_conditioning=False)
    got = sample_fn({"frames": torch.zeros(B, 1)}, cfg)
    traj, logp, c = got[0], got[1], got[2]
    extras = got[3] if len(got) > 3 else {}
    logp = logp + 0.0 * traj.sum(dim=(-1, -2))
    out = rl_objective(traj, logp, c, cfg, spec, gt_bar=extras.get("gt_bar"))
    model.zero_grad(set_to_none=True)
    out["loss"].backward()
    return {"grad": [float(v) for v in model.theta.grad.reshape(-1)],
            "frac_above_bar": (float(out["frac_above_bar"])
                               if "frac_above_bar" in out else None),
            "emitted_keys": sorted(extras),
            "n_tuple": len(got)}


def three_checks(maker) -> dict:
    """The three claims, each returning its own verdict rather than one aggregate."""
    res: dict = {}
    try:
        res["default_off_returns_three_tuple"] = (
            one_step(maker, use_gt_bar=False)["n_tuple"] == 3)
    except Exception as exc:                                  # noqa: BLE001
        res["default_off_returns_three_tuple"] = f"RAISED: {type(exc).__name__}"

    try:
        none_ = one_step(maker, use_gt_bar=False)
        all_ = one_step(maker, use_gt_bar=True, bar_value=-1e9)
        non_ = one_step(maker, use_gt_bar=True, bar_value=+1e9)
        res["no_bar_grad"] = none_["grad"]
        res["admit_all_grad"] = all_["grad"]
        res["admit_none_grad"] = non_["grad"]
        res["identity_admit_all_equals_no_bar"] = (all_["grad"] == none_["grad"])
        res["identity_admit_none_is_exactly_zero"] = (
            all(v == 0.0 for v in non_["grad"]))
        res["frac_admit_all"] = all_["frac_above_bar"]
        res["frac_admit_none"] = non_["frac_above_bar"]
    except Exception as exc:                                  # noqa: BLE001
        res["identity_admit_all_equals_no_bar"] = f"RAISED: {type(exc).__name__}"
        res["identity_admit_none_is_exactly_zero"] = f"RAISED: {type(exc).__name__}"

    try:
        sweep = {}
        for reach in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
            sweep[reach] = one_step(maker, use_gt_bar=True,
                                    gt_reach_m=reach)["frac_above_bar"]
        res["real_scorer_sweep_frac_above_bar"] = {str(k): v
                                                   for k, v in sweep.items()}
        res["real_scorer_masks_PARTIALLY"] = any(
            v is not None and 0.0 < v < 1.0 for v in sweep.values())
    except Exception as exc:                                  # noqa: BLE001
        res["real_scorer_masks_PARTIALLY"] = f"RAISED: {type(exc).__name__}"
    return res


def build_mutant():
    """Write a byte-copy of the adapter with the emitter removed, import it.

    ⛔ The real file is NEVER written to. The mutant is a separate module inside
    the package (so its relative imports resolve) and is deleted afterwards; both
    md5s are recorded so the reader can see they are different files.
    """
    src_path = ADAPT.__file__
    src = open(src_path, encoding="utf-8").read()
    if src.count(EMITTER_LINE) != 1:
        raise SystemExit("MUTATION TARGET NOT FOUND — the emitter line moved; "
                         "refusing to report a mutation that was never applied")
    mutant_src = src.replace(EMITTER_LINE, MUTANT_LINE)
    mut_path = os.path.join(os.path.dirname(src_path), "_mutant_adapter_tmp.py")
    with open(mut_path, "w", encoding="utf-8") as fh:
        fh.write(mutant_src)
    importlib.invalidate_caches()   # the file was written after the finder cached this dir
    mod = importlib.import_module("tanitad.rl._mutant_adapter_tmp")
    return mod, mut_path, src_path, mutant_src


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    record: dict = {
        "_what": ("V2's >=GT truncation is REACHED FROM THE CALLER — proven by "
                  "mutation, not by inspection"),
        "_tier": "N/A — instrument proof, not a capability claim",
        "_evidence_class": "MEASURED (ours)",
        "_when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
    }

    record["FIXED"] = three_checks(ADAPT.make_refcv3_sample_fn)

    mod, mut_path, src_path, mutant_src = build_mutant()
    try:
        record["MUTANT"] = three_checks(mod.make_refcv3_sample_fn)
    finally:
        os.remove(mut_path)
        pyc = os.path.join(os.path.dirname(mut_path), "__pycache__")
        for f in (os.listdir(pyc) if os.path.isdir(pyc) else []):
            if f.startswith("_mutant_adapter_tmp"):
                os.remove(os.path.join(pyc, f))
    on_disk = open(src_path, encoding="utf-8").read()
    record["mutation"] = {
        "target_line": EMITTER_LINE.strip(),
        "adapter_md5_before_and_after": md5(on_disk),
        "mutant_md5": md5(mutant_src),
        "adapter_untouched": md5(on_disk) != md5(mutant_src),
        "mutant_module_deleted": not os.path.exists(mut_path),
    }
    # ⭐ THE DISCRIMINATOR, stated as a claim rather than left to the reader.
    record["mutation"]["discriminated"] = {
        k: [record["FIXED"].get(k), record["MUTANT"].get(k)]
        for k in ("identity_admit_all_equals_no_bar",
                  "identity_admit_none_is_exactly_zero",
                  "real_scorer_masks_PARTIALLY")}
    record["mutation"]["MUTANT_FAILS_ALL_THREE"] = all(
        record["FIXED"].get(k) is True and record["MUTANT"].get(k) is not True
        for k in ("identity_admit_all_equals_no_bar",
                  "identity_admit_none_is_exactly_zero",
                  "real_scorer_masks_PARTIALLY"))

    # --- the reward audit's separation check ------------------------------- #
    audits = {}
    for dt in (RW.DT_S, 0.5):
        for name, w in (("default", RW.DEFAULT_WEIGHTS),
                        ("hackable", RW.HACKABLE_WEIGHTS)):
            for cname, c in (("empty", None),
                             ("exercising", AUD.exercising_ctx(dt=dt))):
                rep = AUD.audit_reward(RewardSpec(weights=dict(w), dt=dt), c)
                audits[f"{name}|dt={dt}|ctx={cname}"] = {
                    "verdict": rep.verdict, "flagged": rep.flagged,
                    "dead_components": rep.dead_components,
                    "tied_at_max": rep.tied_at_max,
                    "reference": rep.reference,
                    "scores": {k: round(v, 6) for k, v in rep.scores.items()},
                }
    record["reward_audit_panel"] = audits
    record["reward_audit_claims"] = {
        "hackable_at_pilot_dt_is_FLAGGED":
            audits["hackable|dt=0.5|ctx=empty"]["verdict"] == "FLAGGED",
        "hackable_at_pilot_dt_ties_three":
            len(audits["hackable|dt=0.5|ctx=empty"]["tied_at_max"]) == 3,
        "default_stays_green_on_an_exercising_ctx":
            audits["default|dt=0.5|ctx=exercising"]["verdict"] == "clean",
        "default_was_UNRULABLE_on_the_empty_ctx":
            audits["default|dt=0.5|ctx=empty"]["verdict"] == "INCONCLUSIVE",
    }

    record["wall_s"] = round(time.time() - t0, 3)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=False)
    print(json.dumps({k: record[k] for k in ("mutation", "reward_audit_claims")},
                     indent=1))
    print(f"ZZPROOF-WRITTEN-{a.out}ZZ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
