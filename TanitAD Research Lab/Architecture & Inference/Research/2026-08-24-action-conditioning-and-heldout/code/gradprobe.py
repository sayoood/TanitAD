"""E-DEC-37 — WHERE DOES O11's GRADIENT ACTUALLY GO, AND IS w=1.0 INERT?

⛔ THE QUESTION THIS SETTLES BEFORE 8 GPU-HOURS ARE SPENT. `o11p30k` launched at
`--w-o11-cf 1.0` and sat at its no-information floor for the first ~1,800 steps
(`o11_excess` ~1e-5 against a floor of 1.3863). Two readings are available and
they lead to opposite actions: **the term is inert at this weight** (⇒ restart at
a higher weight, and a flat result at 30k would be an artefact of SCALE, not
evidence about the hypothesis), or **the term is working and the predictor is
simply hard to move** (⇒ let it run, and a flat result is informative). Guessing
costs 8 h of the only GPU we have. Measuring costs one forward/backward.

MEASURED (2026-08-24, synthetic batch, V6Config, batch 8, o5_k 8, o11_k 4,
o11_negs 3, CPU, identical batch for both terms):

    term      loss     grad -> predictor   grad -> FiLM   grad -> act_emb
    o5     0.83754        7.2304e-01        7.2023e-05       0.0000e+00
    o11    1.38629        3.5570e-05        3.5570e-05       0.0000e+00

⭐⭐ THE STRUCTURAL FACT, AND IT IS THE POINT OF THE WHOLE MEASUREMENT:
**O11's gradient is ~99.997 % concentrated on the FiLM** — 3.5570e-05 at the
action seam against 1.0351e-09 on the state-path blocks, a ratio of ~34,000x.
⚠️ NOT "exactly equal": the coarse first pass printed the same 4-significant-
figure value in both columns and I read that as exact cancellation. The finer
per-group probe shows the state path receives a small NON-ZERO gradient, which is
what should happen — the cancellation argument below is exact only for a
PERFECTLY action-blind predictor, and this one is merely nearly so. The mechanism
is right; "exactly" was an artefact of print precision. It is a property of the
contrastive:

    L11 = CE(logits, 0),  logits_i = -||zhat(a_i) - z_true||^2 / tau
    dL/dtheta = SUM_i (dL/dd_i) * (dd_i/dtheta),   and   SUM_i dL/dd_i = 0

When the predictor is action-blind, `zhat(a_i)` is the same for every i, so every
`dd_i/dtheta` is IDENTICAL — and multiplying identical terms by softmax gradients
that sum to zero CANCELS EXACTLY. ⇒ **The only parameters that receive any O11
gradient are those whose effect DIFFERS across action variants — i.e. the action
seam.** O11 puts 100 % of its push on the FiLM and nothing anywhere else.

⛔ SO "RATIO INTO THE PREDICTOR = 0.0000" IS THE WRONG QUANTITY, AND MY FIRST
AUTOMATED VERDICT FIRED ON IT ("the weight must be RAISED"). The right quantity
is the fraction at the seam:

    at the FiLM:  O5 = 7.2023e-05, O11 = 3.5570e-05
    => O11 OWNS 33 % OF THE GRADIENT AT THE PARAMETER THAT CARRIES
       ACTION-CONDITIONING.

⚠️ AND AdamW NORMALISES PER PARAMETER, so a consistently-signed small gradient
still produces full-size steps. The absolute magnitude matters far less than the
FRACTION. ⇒ **w = 1.0 is a fair weight; the term is correctly targeted; a flat
`o11_excess` is INFORMATIVE rather than a scale artefact. The arm runs unchanged,
and the pre-registration is not violated.**

⭐ THE GENUINELY NEW MECHANISTIC FINDING, WHICH IS NOT ABOUT O11 AT ALL:
**O5 gives the state-path BLOCKS 1.8023e-03 and the action seam 7.2023e-05 — a
25x starvation.** ⚠️ NOT 10,000x: that figure compared O5's WHOLE-predictor
gradient (7.2e-01, which is dominated by the wide output HEADS, `Linear(d,
state_dim)` at state_dim 2048) against the FiLM, i.e. two groups that are not
comparable. The apples-to-apples comparison is blocks-vs-seam and it is 25x —
meaningful, and far less dramatic than the number I first quoted. That is exactly why E-DEC-34 saw the FiLM open so slowly (|FiLM|/
|act_emb| 0.072 at 2k -> 0.152 at 10k -> 0.205-0.308 at 30k, never plateauing).
The architecture is not blocking the action pathway; it is starving it.

⚠️ SCOPE. Synthetic batch, untrained stack — this measures the gradient GEOMETRY
of the objective, not what a trained model does. The cancellation argument is
exact for an action-blind predictor and weakens as the predictor becomes
action-sensitive, which is precisely the direction training should move it.
"""
from __future__ import annotations

import sys

sys.path[:0] = [r"C:/Users/Admin/tanitad-mirror/stack",
                r"C:/Users/Admin/tanitad-mirror/stack/scripts"]


def main() -> int:
    import json

    import torch
    from tanitad.models.v6 import V6Config, V6Stack
    from train_v6_staged import V6LossWeights, synthetic_train_batch, v6_loss_step

    torch.manual_seed(0)
    stack = V6Stack(V6Config())
    b = synthetic_train_batch(stack, batch=8, k=12, seed=0)
    base = dict(o1_ctrl=0.0, o1_fact=0.0, o1_scene=0.0, o2_nearfield=0.0,
                o3_masked=0.0, o6_sigreg=0.0, t1_latent=0.0, s1_latent=0.0)
    groups = {
        "predictor (all)": lambda n: "predictor_op" in n,
        "FiLM (action seam)": lambda n: "predictor_op" in n and "film.to_scale_shift" in n,
        "act_emb": lambda n: "predictor_op" in n and "act_emb" in n,
        "blocks (state path)": lambda n: "predictor_op" in n and ".blocks." in n
                                         and "film" not in n,
    }
    P = {g: [p for n, p in stack.named_parameters() if f(n) and p.requires_grad]
         for g, f in groups.items()}

    def grads(w, key):
        stack.zero_grad(set_to_none=True)
        L = v6_loss_step(stack, b, stage="S-W",
                         weights=V6LossWeights(**{**base, **w}), o5_k=8,
                         o11_k=4, o11_negs=3)
        t = L[key]
        out = {}
        for g, ps in P.items():
            if not ps:
                out[g] = 0.0
                continue
            gg = torch.autograd.grad(t, ps, retain_graph=True, allow_unused=True)
            out[g] = float(torch.sqrt(sum((x ** 2).sum()
                                          for x in gg if x is not None)))
        return out, float(t.detach())

    g5, v5 = grads({"o5_rollout": 1.0}, "o5")
    g11, v11 = grads({"o5_rollout": 1.0, "o11_cf": 1.0}, "o11")

    print("\n  E-DEC-37 · WHERE DOES O11's GRADIENT GO?\n")
    print(f"  {'group':<24}{'o5':>14}{'o11':>14}{'o11 share':>12}")
    print("  " + "-" * 64)
    for g in groups:
        tot = g5[g] + g11[g]
        print(f"  {g:<24}{g5[g]:>14.4e}{g11[g]:>14.4e}"
              f"{(g11[g] / tot if tot > 0 else 0):>12.1%}")
    share = g11["FiLM (action seam)"] / max(
        g5["FiLM (action seam)"] + g11["FiLM (action seam)"], 1e-30)
    same = abs(g11["predictor (all)"] - g11["FiLM (action seam)"]) < 1e-12
    print(f"\n  O11's whole-predictor gradient == its FiLM gradient: {same}")
    print(f"  ⇒ the shared-path gradients CANCEL: softmax grads sum to zero and,")
    print(f"    for an action-blind predictor, every dd_i/dtheta is identical.")
    print(f"\n  O11's SHARE of the gradient at the action seam: {share:.1%}")
    print(f"  ⚠️ AdamW normalises PER PARAMETER, so the SHARE drives the step,")
    print(f"     not the raw magnitude. The readable verdict uses the share.")
    verdict = ("w=1.0 is a FAIR weight — O11 is a substantial fraction of the "
               "gradient at the action seam, so a flat o11_excess is INFORMATIVE"
               if share > 0.15 else
               "w=1.0 is EFFECTIVELY INERT at the seam — raise --w-o11-cf, or a "
               "flat result is an artefact of scale")
    print(f"\n  VERDICT (computed on the SEAM share, not the whole-predictor "
          f"ratio): {verdict}")
    starve = g5["blocks (state path)"] / max(g5["FiLM (action seam)"], 1e-30)
    print(f"\n  ⭐ AND THE FINDING THAT IS NOT ABOUT O11: O5 gives the STATE PATH")
    print(f"     {starve:,.0f}x the gradient it gives the ACTION SEAM. The seam is")
    print(f"     GRADIENT-STARVED, which is why E-DEC-34 saw the FiLM open so slowly.")
    out = {"_evidence_class": "MEASURED (ours; synthetic batch, untrained stack)",
           "eval_tier": "T0-DIAGNOSTIC", "o5": g5, "o11": g11,
           "o5_loss": v5, "o11_loss": v11,
           "o11_share_at_action_seam": round(share, 4),
           "state_path_over_seam_gradient_ratio": round(starve, 1),
           "shared_path_cancels": bool(same), "verdict": verdict}
    p = (__import__("pathlib").Path(__file__).resolve().parent / "gradprobe.json")
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
