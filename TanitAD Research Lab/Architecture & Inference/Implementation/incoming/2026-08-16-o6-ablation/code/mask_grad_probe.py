"""⭐ THE PAIRED SINGLE-BATCH GRADIENT PROBE — 0 GPU, S-W, attributable.

WHAT THIS ANSWERS. The PI asked *"can we measure whether our measures work?"*
This is the cheapest instrument that bites: fixed batch, fixed ``generator`` AND
fixed ``sigreg_generator``, two arms differing in ONE loss weight, read on the
PARAMETER GRADIENT rather than on the printed loss.

⛔ WHY IT WAS IMPOSSIBLE BEFORE 2026-08-16. ``SigReg`` drew its slice directions
from the GLOBAL RNG (``sigreg.py:70``), so two arms of any in-process A/B got
DIFFERENT directions and every shared parameter's gradient was perturbed between
them. The lever's own effect and the resample were summed and unattributable.
``142ce34`` made ``o6`` reproducible from a passed generator; this probe is the
first thing that can use it. ``arm_noise_floor`` below RE-MEASURES that noise on
the incumbent path, so the claim "it was impossible" is a number in this
artifact, not a citation.

THE ESTIMATOR — exact, not statistical. Gradients are LINEAR in the loss, so
with everything else held bit-identical

    g(lever ON) - g(lever OFF)  ==  the lever term's OWN gradient, exactly.

No fitting, no resampling. The uncertainty in this file is across SEEDS (model
init + batch), which is the only place uncertainty actually lives.

⭐ THE READING THAT MATTERS IS THE DIRECTION, NOT THE NORM.

  cos(g_term, g_rest) ~ 0   the term adds an INDEPENDENT signal — it moves the
                            trunk somewhere the rest of the objective does not
  cos(g_term, g_rest) > 0   REDUNDANT — it amplifies what the rest already says
  cos(g_term, g_rest) < 0   it FIGHTS the rest of the objective
  ||g_term|| == 0           a true NULL — the lever does not reach the parameter

⛔ AND ~0 IS NOT AUTOMATICALLY "INDEPENDENT". Two random vectors in D dimensions
have |cos| ~ 1/sqrt(D) by chance. At D ~ 1e5 that is ~0.003, so a cosine of 0.05
is 16x chance and NOT orthogonal. Every cosine here is therefore reported
against ``chance_level`` = 1/sqrt(D) and as a MULTIPLE of it, and the full
pairwise term x term cosine matrix is computed so "O3 is orthogonal" can be
checked against "everything here is orthogonal" — which would make the statement
about the dimension, not about O3.

CONTROLS (a probe that always finds something has found nothing):
  N1  the SAME arm twice, seeded            -> delta must be EXACTLY 0
  N2  a structurally inert lever (``t1_latent`` in S-W, which ``for_stage``
      forces to 0)                          -> delta must be EXACTLY 0
  N3  the SAME arm twice on the INCUMBENT global-RNG path -> must be NON-zero;
      this is the positive control AND the sensitivity floor the fix removed

Run:  PYTHONUTF8=1 python mask_grad_probe.py --out ../raw/mask_grad_probe.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")   # CLAUDE.md: torch spawns ~113

import torch

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[6]
_STACK = _ROOT / "stack"
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402
from train_v6_staged import (  # noqa: E402
    V6LossWeights, synthetic_train_batch, v6_loss_step)

STAGE = "S-W"

#: The S-W levers, as (name, weight-fields, incumbent value). O1's three
#: sub-weights are one TERM in the loss dict, so they move together.
LEVERS: dict[str, tuple[tuple[str, ...], tuple[float, ...]]] = {
    "o1": (("o1_ctrl", "o1_fact", "o1_scene"), (1.0, 1.0, 0.3)),
    "o2": (("o2_nearfield",), (1.0,)),
    "o3": (("o3_masked",), (1.0,)),
    "o5": (("o5_rollout",), (1.0,)),
    "o6": (("o6_sigreg",), (0.1,)),
}

#: v6.py MODULE_GROUPS. ``masked_cells`` and ``sigreg`` are both "aux", so the
#: two levers under test are separated by PREFIX as well as by group.
TRUNK_GROUPS = ("encoder", "readout", "predictor_op")


#: ⛔ THE READOUT GRID IS LOAD-BEARING FOR O3 AND NEARLY COST ME A FALSE
#: FINDING. ``sample_cell_block_mask`` places ``n_blocks`` blocks of
#: ``block_hw`` on a ``grid x grid`` field. At grid=2 with the default 2x2
#: blocks the block IS the whole grid -> mask_rate 1.0 -> every cell is
#: replaced by the learned mask token in ``MaskedCellPredictor.forward``
#: (v6.py:2089) -> the module's input no longer depends on ``cells`` at all ->
#: O3's gradient to the trunk is EXACTLY ZERO. That is a FIXTURE ARTIFACT, not
#: a property of O3. The live v6F S-W run is ``--readout-grid 4`` (mask rate
#: 0.469), which is what GRID_LIVE reproduces. GRID_DEGENERATE is kept and run
#: as a labelled control precisely because the artifact is instructive.
GRID_LIVE = 4
GRID_DEGENERATE = 2


def _sub_cfgs(grid: int = GRID_LIVE) -> dict:
    return dict(
        encoder=EncoderConfig(in_channels=9, image_size=64, image_width=64,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=grid, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=2,
                                  action_dim=3))


_BASE_KW = dict(
    d_tac=32, d_str=16, adapter_hidden=32, f_hidden_tac=32, f_hidden_str=32,
    f_blocks=1, aux_hidden=16, sigreg_slices=8, plan_steps=6, dt=0.1,
    op_band_s=(0.0, 0.2), tac_band_s=(0.2, 0.6), hz_op=10.0, hz_tac=2.0,
    hz_str=0.5, d_plan_feat=16, emission_hidden=16, d_goal_embed=128,
    n_candidates=8)


def build(seed: int, grid: int = GRID_LIVE, **over) -> V6Stack:
    torch.manual_seed(seed)
    s = V6Stack(V6Config(**(copy.deepcopy(_BASE_KW) | _sub_cfgs(grid) | over)))
    s.train()
    return s


def make_batch(stack: V6Stack, seed: int) -> dict:
    b = synthetic_train_batch(stack, batch=2, k=4, seed=seed)
    b["gt_wp"] = torch.zeros(2, 2, 2)
    return b


def grads(stack: V6Stack, batch: dict, weights: V6LossWeights, *,
          sig_seed: int | None = 11, gen_seed: int = 5) -> tuple[dict, float, dict]:
    """One backward. Returns {param_name: flat grad}, the scalar loss, the log.

    Missing grads become ZEROS, not omissions: a parameter a term does not
    reach must contribute 0 to every norm and cosine, and dropping the key
    instead would silently change the vector's support between arms.
    """
    stack.zero_grad(set_to_none=True)
    out = v6_loss_step(
        stack, batch, stage=STAGE, o1_k=2, o5_k=2, weights=weights,
        generator=torch.Generator().manual_seed(gen_seed),
        sigreg_generator=(None if sig_seed is None
                          else torch.Generator().manual_seed(sig_seed)))
    out["loss"].backward()
    g = {n: (p.grad.detach().reshape(-1).clone() if p.grad is not None
             else torch.zeros(p.numel()))
         for n, p in stack.named_parameters()}
    loss = float(out["loss"].detach())
    stack.zero_grad(set_to_none=True)
    return g, loss, dict(out["log"])


def _cat(g: dict, names: list[str]) -> torch.Tensor:
    if not names:
        return torch.zeros(0)
    return torch.cat([g[n] for n in names]).double()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float | None:
    na, nb = float(a.norm()), float(b.norm())
    if na < 1e-30 or nb < 1e-30:
        return None                     # undefined, and that IS the finding
    return float((a @ b) / (na * nb))


def _sub(ga: dict, gb: dict) -> dict:
    return {k: ga[k] - gb[k] for k in ga}


def probe_seed(build_seed: int, batch_seed: int, grid: int = GRID_LIVE) -> dict:
    """Every S-W lever's isolated gradient, on one (model, batch) pair."""
    stack = build(build_seed, grid)
    batch = make_batch(stack, batch_seed)
    names = [n for n, _ in stack.named_parameters()]
    groups = {n: stack.group_of(n) for n in names}
    trunk = [n for n in names if groups[n] in TRUNK_GROUPS]
    mcell = [n for n in names if n.startswith("masked_cells.")]
    sigp = [n for n in names if n.startswith("sigreg.")]
    full = V6LossWeights()
    g_full, loss_full, log_full = grads(stack, batch, full)
    n_all = int(_cat(g_full, names).numel())
    n_trunk = int(_cat(g_full, trunk).numel())

    res: dict = {
        "build_seed": build_seed, "batch_seed": batch_seed,
        "readout_grid": grid, "n_cells": int(stack.cfg.n_cells),
        # ⛔ stamped in the record for the same reason `rank_ceiling` is: a
        # mask rate of 1.0 makes O3's trunk gradient zero BY CONSTRUCTION, so
        # no O3 reading may be quoted without it.
        "o3_mask_rate": log_full.get("o3_mask_rate"),
        "o3_n_masked": log_full.get("o3_n_masked"),
        "loss_full": loss_full,
        "n_params_all": n_all, "n_params_trunk": n_trunk,
        "n_params_masked_cells": int(_cat(g_full, mcell).numel()),
        "n_params_sigreg": int(_cat(g_full, sigp).numel()),
        "chance_cos_all": 1.0 / math.sqrt(max(n_all, 1)),
        "chance_cos_trunk": 1.0 / math.sqrt(max(n_trunk, 1)),
        "levers": {}, "cos_matrix_trunk": {},
    }

    # ---- each lever's isolated gradient, by exact linearity ----------------
    iso: dict[str, dict] = {}
    for lev, (fields, on) in LEVERS.items():
        off = V6LossWeights(**{f: 0.0 for f in fields})
        g_off, loss_off, _ = grads(stack, batch, off)
        g_lev = _sub(g_full, g_off)                       # the term's OWN grad
        iso[lev] = g_lev
        v_lev_t, v_off_t = _cat(g_lev, trunk), _cat(g_off, trunk)
        v_full_t = _cat(g_full, trunk)
        rec = {
            "weights_zeroed": list(fields), "incumbent_weight": list(on),
            "loss_full": loss_full, "loss_lever_off": loss_off,
            "loss_delta": loss_full - loss_off,
            # --- direction: the reading that matters ---------------------
            "cos_term_vs_rest_TRUNK": _cos(v_lev_t, v_off_t),
            "cos_armON_vs_armOFF_TRUNK": _cos(v_full_t, v_off_t),
            "cos_term_vs_rest_ALL": _cos(_cat(g_lev, names),
                                         _cat(g_off, names)),
            # --- magnitude ------------------------------------------------
            "norm_term_TRUNK": float(v_lev_t.norm()),
            "norm_rest_TRUNK": float(v_off_t.norm()),
            "norm_full_TRUNK": float(v_full_t.norm()),
            "relative_pull_TRUNK": (float(v_lev_t.norm() / v_off_t.norm())
                                    if float(v_off_t.norm()) > 1e-30 else None),
            "norm_term_masked_cells": float(_cat(g_lev, mcell).norm()),
            "norm_rest_masked_cells": float(_cat(g_off, mcell).norm()),
            "norm_term_sigreg": float(_cat(g_lev, sigp).norm()),
            "norm_rest_sigreg": float(_cat(g_off, sigp).norm()),
            "per_group": {},
        }
        for grp in sorted(set(groups.values())):
            gn = [n for n in names if groups[n] == grp]
            a, b = _cat(g_lev, gn), _cat(g_off, gn)
            rec["per_group"][grp] = {
                "n_params": int(a.numel()),
                "norm_term": float(a.norm()), "norm_rest": float(b.norm()),
                "cos_term_vs_rest": _cos(a, b)}
        res["levers"][lev] = rec

    # ---- ⭐ is O3 special, or is EVERYTHING orthogonal at this D? ----------
    for a_name in LEVERS:
        res["cos_matrix_trunk"][a_name] = {
            b_name: _cos(_cat(iso[a_name], trunk), _cat(iso[b_name], trunk))
            for b_name in LEVERS}
    return res


def controls(build_seed: int, batch_seed: int, grid: int = GRID_LIVE) -> dict:
    """⛔ A probe that always finds something has found nothing."""
    stack = build(build_seed, grid)
    batch = make_batch(stack, batch_seed)
    names = [n for n, _ in stack.named_parameters()]
    full = V6LossWeights()

    ga, _, _ = grads(stack, batch, full)
    gb, _, _ = grads(stack, batch, full)
    n1 = float(_cat(_sub(ga, gb), names).norm())

    # N2 — a lever ``for_stage("S-W")`` forces to zero: the knob turns, the
    # loss cannot see it. This is the null the probe MUST report as null.
    gt1, _, _ = grads(stack, batch, V6LossWeights(t1_latent=0.0))
    gt2, _, _ = grads(stack, batch, V6LossWeights(t1_latent=1.0))
    n2 = float(_cat(_sub(gt1, gt2), names).norm())

    # N3 — the incumbent global-RNG path: the SAME arm twice MUST differ.
    # ⚠️ Reported on the TRUNK support as well as on all params, because every
    # lever norm above is trunk-restricted and comparing an all-param noise
    # figure against a trunk-restricted signal would be a scope mismatch —
    # the `df`/`effective_rank` error class in a third costume.
    groups = {n: stack.group_of(n) for n in names}
    trunk = [n for n in names if groups[n] in TRUNK_GROUPS]
    ha, _, _ = grads(stack, batch, full, sig_seed=None)
    hb, _, _ = grads(stack, batch, full, sig_seed=None)
    dd = _sub(ha, hb)
    n3 = float(_cat(dd, names).norm())
    n3_trunk = float(_cat(dd, trunk).norm())
    ref = float(_cat(ha, names).norm())
    ref_trunk = float(_cat(ha, trunk).norm())

    return {
        "N1_same_arm_twice_seeded": {
            "grad_delta_l2": n1, "expect": "EXACTLY 0",
            "pass": n1 == 0.0,
            "note": "the harness itself is deterministic; without this every "
                    "number above could be resample noise"},
        "N2_structurally_inert_lever_t1_in_SW": {
            "grad_delta_l2": n2, "expect": "EXACTLY 0", "pass": n2 == 0.0,
            "note": "t1_latent is forced to 0 by V6LossWeights.for_stage('S-W'), "
                    "so moving it 0.0 -> 1.0 changes the LAUNCH LINE and nothing "
                    "else. The probe must report no effect."},
        "N3_POSITIVE_same_arm_twice_on_the_INCUMBENT_path": {
            "grad_delta_l2": n3, "grad_norm_reference": ref,
            "grad_delta_l2_TRUNK": n3_trunk,
            "grad_norm_reference_TRUNK": ref_trunk,
            "relative": (n3 / ref if ref > 0 else None),
            "relative_TRUNK": (n3_trunk / ref_trunk if ref_trunk > 0 else None),
            "expect": "NON-zero", "pass": n3 > 0.0,
            "note": "sigreg_generator=None is the pre-142ce34 code. This is the "
                    "NOISE FLOOR every in-process A/B used to sit on, and any "
                    "lever effect smaller than it was unmeasurable."},
    }


def summarise(runs: list[dict], noise_floor: float | None = None) -> dict:
    """Across-seed summary — paired, PER LEVER, never pooled into one score."""
    summary = {}
    for lev in LEVERS:
        cos = [r["levers"][lev]["cos_term_vs_rest_TRUNK"] for r in runs]
        rel = [r["levers"][lev]["relative_pull_TRUNK"] for r in runs]
        nt = [r["levers"][lev]["norm_term_TRUNK"] for r in runs]
        cv = [c for c in cos if c is not None]
        chance = runs[0]["chance_cos_trunk"]
        summary[lev] = {
            "cos_term_vs_rest_TRUNK": {
                "per_seed": cos,
                "mean": (sum(cv) / len(cv)) if cv else None,
                "min": min(cv) if cv else None, "max": max(cv) if cv else None,
                "chance_level": chance,
                "mean_over_chance": ((abs(sum(cv) / len(cv)) / chance)
                                     if cv else None),
                "sign_consistent": (all(c > 0 for c in cv)
                                    or all(c < 0 for c in cv)) if cv else None},
            "relative_pull_TRUNK": {
                "per_seed": rel,
                "mean": (sum(x for x in rel if x is not None)
                         / max(len([x for x in rel if x is not None]), 1))},
            "norm_term_TRUNK": {"per_seed": nt, "min": min(nt), "max": max(nt)},
            "reaches_trunk": all(x > 0 for x in nt),
            # ⭐ was this lever measurable BEFORE 142ce34? The pre-fix noise
            # floor (control N3) perturbed every shared parameter between two
            # arms; a lever whose own gradient is smaller than that floor was
            # not attributable at all.
            "vs_pre_fix_noise_floor": (
                {"noise_floor_l2_TRUNK": noise_floor,
                 "term_norm_TRUNK_mean": sum(nt) / len(nt),
                 "ratio_term_over_noise": (sum(nt) / len(nt)) / noise_floor,
                 "measurable_before_the_fix":
                     (sum(nt) / len(nt)) / noise_floor > 1.0}
                if noise_floor else None),
        }
    return summary


def _geometry(runs: list[dict], ctl: dict, label: str) -> dict:
    nf = ctl["N3_POSITIVE_same_arm_twice_on_the_INCUMBENT_path"][
        "grad_delta_l2_TRUNK"]
    return {"label": label,
            "readout_grid": runs[0]["readout_grid"],
            "n_cells": runs[0]["n_cells"],
            "o3_mask_rate": runs[0]["o3_mask_rate"],
            "controls": ctl,
            "summary_across_seeds": summarise(runs, nf),
            "per_seed": runs}


def _report(geo: dict) -> None:
    runs, summary = geo["per_seed"], geo["summary_across_seeds"]
    print(f"\n{'=' * 72}\n{geo['label']}  (readout_grid={geo['readout_grid']}, "
          f"n_cells={geo['n_cells']}, o3_mask_rate={geo['o3_mask_rate']})\n"
          f"{'=' * 72}")
    print("--- CONTROLS ---")
    for k, v in geo["controls"].items():
        print(f"  {k}: delta={v['grad_delta_l2']:.6g} "
              f"expect={v['expect']} pass={v['pass']}")
    ch = runs[0]["chance_cos_trunk"]
    print(f"\n--- cos(term, rest) on the TRUNK  (chance = {ch:.6g}) ---")
    for lev, s in summary.items():
        c = s["cos_term_vs_rest_TRUNK"]
        m = c["mean"]
        oc = c["mean_over_chance"]
        print(f"  {lev}: cos={('None' if m is None else f'{m:+.5f}')} "
              f"({'n/a' if oc is None else f'{oc:.0f}x'} chance)  "
              f"sign_consistent={c['sign_consistent']}  "
              f"rel_pull={s['relative_pull_TRUNK']['mean']:.5f}  "
              f"reaches_trunk={s['reaches_trunk']}")
    nf = geo["controls"]["N3_POSITIVE_same_arm_twice_on_the_INCUMBENT_path"]
    print(f"\n--- signal vs the PRE-FIX noise floor (TRUNK support, "
          f"floor={nf['grad_delta_l2_TRUNK']:.4f}) ---")
    for lev, s in summary.items():
        v = s["vs_pre_fix_noise_floor"]
        print(f"  {lev}: |g_term|={v['term_norm_TRUNK_mean']:.5f}  "
              f"ratio={v['ratio_term_over_noise']:.3f}  "
              f"measurable_before_the_fix={v['measurable_before_the_fix']}")
    print("\n--- pairwise cos(term_i, term_j) on the TRUNK, seed 0 ---")
    m = runs[0]["cos_matrix_trunk"]
    ks = list(LEVERS)
    print("        " + "".join(f"{k:>9}" for k in ks))
    for i in ks:
        print(f"  {i:>5} " + "".join(
            f"{'    n/a' if m[i][j] is None else f'{m[i][j]:+9.4f}'}"
            for j in ks))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HERE.parents[1] / "raw"
                                         / "mask_grad_probe.json"))
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    t0 = time.time()
    pairs = [(s, 100 + s) for s in range(a.seeds)]

    live = _geometry([probe_seed(bs, xs, GRID_LIVE) for bs, xs in pairs],
                     controls(0, 100, GRID_LIVE),
                     "PRIMARY — live readout geometry (--readout-grid 4)")
    degen = _geometry([probe_seed(bs, xs, GRID_DEGENERATE) for bs, xs in pairs],
                      controls(0, 100, GRID_DEGENERATE),
                      "CONTROL — DEGENERATE grid 2x2: the 2x2 block IS the "
                      "whole grid, mask_rate 1.0")

    out = {
        "meta": {
            "what": "paired single-batch gradient probe on S-W, 0 GPU",
            "date": "2026-08-16", "stage": STAGE,
            "estimator": "EXACT by linearity: g(ON) - g(OFF) is the term's own "
                         "gradient. Uncertainty is ACROSS SEEDS only.",
            "evidence_class": "MEASURED (ours)",
            "torch": torch.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "device": "cpu",
            "seed_pairs": pairs, "wall_s": None,
            "build": "SYNTHETIC CPU build — NOT the live v6F S-W checkpoint. "
                     "Readout GRID matches the live run (4x4) because it sets "
                     "O3's mask geometry; d_readout/d_model/window/plan_steps "
                     "are the small CPU fixture and do NOT match the live run.",
            "interpretation": {
                "cos ~ 0": "term adds an INDEPENDENT signal to the trunk",
                "cos > 0": "REDUNDANT with the rest of the objective",
                "cos < 0": "term FIGHTS the rest of the objective",
                "norm == 0": "a true NULL: the lever never reaches the parameter",
                "caveat": "|cos| ~ 1/sqrt(D) arises by CHANCE; every cosine is "
                          "reported against chance_level and as a multiple of it"},
        },
        "geometries": {"live_grid4": live, "degenerate_grid2": degen},
    }
    out["meta"]["wall_s"] = round(time.time() - t0, 2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    _report(live)
    _report(degen)
    print(f"\nwrote {a.out}  ({out['meta']['wall_s']} s)")


if __name__ == "__main__":
    main()
