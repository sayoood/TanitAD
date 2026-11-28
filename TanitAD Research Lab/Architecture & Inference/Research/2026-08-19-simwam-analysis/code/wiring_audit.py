"""Did the ACTION PATH ever leave zero? A weight-level audit of the predictors.

THE TENSION THIS RESOLVES.  Two things were reported that sound contradictory:
  * the residual-scale fix improved the predictor's one-step error ~580x
  * the resulting predictor is ACTION-BLIND (counterfactual divergence 0.000)

Reading `OperativePredictor` (stack/tanitad/models/predictor.py) they may be the
SAME EVENT.  The action path is:

    actions -> act_emb -> cond -> FiLM.to_scale_shift  [ZERO-INIT]
            -> x*(1+scale)+shift -> mlp -> h_last -> heads[k] [x 1e-3] -> delta
    out[k]  = z_t + delta

Two multiplicative bottlenecks in series:
  1. FiLM `to_scale_shift` is EXACTLY zero-init (weight AND bias), so at step 0
     actions have LITERALLY no effect -- `x * (1+0) + 0 == x`.
  2. `heads[k]` is initialised at RESIDUAL_HEAD_INIT_SCALE = 1e-3, and the
     gradient reaching FiLM/act_emb flows BACK THROUGH that head:
     dL/dh_last = W_head^T . dL/dout.  Scaling W by 1e-3 attenuates the learning
     signal into the ONLY path by which an action can matter, by ~1000x.

So "predict no change" is reachable with delta ~ 0 and FiLM still at zero, and
the gradient that would teach action-conditioning is throttled.

⛔ THAT IS AN ARGUMENT, NOT A MEASUREMENT.  This script measures it: on each
trained checkpoint, how far did each stage of the action path actually travel
from its initial value?

    film_rel   = ||FiLM.to_scale_shift.weight||        (init EXACTLY 0)
    head_rel   = ||heads[k].weight|| / (1e-3 * sqrt-scale of a default init)
    act_emb    = ||act_emb weights||                   (init: normal, NOT zero)

CONTROLS that make it readable:
  * `fixed` is action-SENSITIVE (divergence 474.9x) -- its FiLM MUST be non-zero.
    If champ30k's FiLM is ~0 and `fixed`'s is not, the mechanism is identified.
  * act_emb is NOT zero-init, so it is the "did anything train at all" reference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
OUT = SP / "wiring_audit.json"

ARMS = ("champ30k", "lewm", "lewm_o1", "lewm_o1_detach", "fixed")


def main() -> int:
    import v7tiny_g2 as G
    dev = torch.device("cpu")
    rep = {"_evidence_class": "MEASURED (ours; trained checkpoint weights)",
           "what": "how far each stage of the action path travelled from init",
           "arms": {}}

    print("\n  ACTION-PATH WEIGHT AUDIT — FiLM is zero-init, so any non-zero norm is LEARNED\n")
    print(f"  {'arm':<16}{'FiLM |W|':>11}{'FiLM |b|':>11}{'act_emb |W|':>13}"
          f"{'head |W|':>11}{'head/1e-3':>11}   divergence(measured)")
    print("  " + "-" * 92)
    div_known = {"champ30k": "0.0000", "lewm": "0.000", "lewm_o1": "516.6",
                 "lewm_o1_detach": "6.53", "fixed": "474.9"}

    for arm in ARMS:
        try:
            world, step = G.load_arm(arm, dev)
        except Exception as e:
            print(f"  {arm:<16} NOT LOADABLE ({type(e).__name__}) — reported, not skipped")
            rep["arms"][arm] = {"error": type(e).__name__}
            continue
        p = world.predictor
        film_w = film_b = 0.0
        n_film = 0
        for blk in p.blocks:
            film_w += float(blk.film.to_scale_shift.weight.norm())
            film_b += float(blk.film.to_scale_shift.bias.norm())
            n_film += 1
        act_w = sum(float(m.weight.norm()) for m in p.act_emb if hasattr(m, "weight"))
        head_w = sum(float(h.weight.norm()) for h in p.heads.values()) / len(p.heads)
        # a default nn.Linear(d, state_dim) has ||W|| ~ sqrt(d*state_dim/(3*d)) ;
        # report the ratio to the DOWN-SCALED init so "1.0" means "never grew"
        d = p.cfg.d_model
        sd = p.state_dim
        import math
        default_norm = math.sqrt(d * sd / (3.0 * d))       # U(-1/sqrt(d),1/sqrt(d))
        head_ratio = head_w / max(default_norm * 1e-3, 1e-12)
        rep["arms"][arm] = {"step": int(step), "n_blocks": n_film,
                            "film_weight_norm": round(film_w, 6),
                            "film_bias_norm": round(film_b, 6),
                            "act_emb_weight_norm": round(act_w, 4),
                            "head_weight_norm_mean": round(head_w, 6),
                            "head_norm_over_downscaled_init": round(head_ratio, 3),
                            "measured_divergence_x100": div_known.get(arm)}
        print(f"  {arm:<16}{film_w:>11.5f}{film_b:>11.5f}{act_w:>13.3f}"
              f"{head_w:>11.5f}{head_ratio:>11.2f}   {div_known.get(arm, '-'):>8}")
        del world

    a = rep["arms"]
    ch = a.get("champ30k", {})
    fx = a.get("fixed", {})
    if "film_weight_norm" in ch and "film_weight_norm" in fx:
        ratio = fx["film_weight_norm"] / max(ch["film_weight_norm"], 1e-12)
        rep["film_ratio_fixed_over_champ30k"] = round(ratio, 2)
        rep["verdict"] = (
            f"champ30k's FiLM weight norm is {ch['film_weight_norm']:.5f} against `fixed`'s "
            f"{fx['film_weight_norm']:.5f} ({ratio:.1f}x). FiLM is EXACTLY zero at init, so this "
            f"is the entire learned action pathway. Its residual heads sit at "
            f"{ch['head_norm_over_downscaled_init']:.2f}x their DOWN-SCALED init, i.e. they "
            f"{'never left' if ch['head_norm_over_downscaled_init'] < 3 else 'grew past'} the 1e-3 regime.")
        print(f"\n  {rep['verdict']}")
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
