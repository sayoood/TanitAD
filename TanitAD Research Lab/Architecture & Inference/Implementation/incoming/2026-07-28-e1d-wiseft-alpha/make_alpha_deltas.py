#!/usr/bin/env python3
"""E1d — WiSE-FT weight-space interpolation between REF-C base and the E1c CL-SFT.

WHY THIS AND NOT "TRAIN LONGER": E1c's frontier is BOUND — P1/P2 (corridor
departure, overall AND junction, CI-separated LOWER) fire at 15/17 points, but
guardrail Ga (open-loop ADE@2s not separated-higher) holds at 0/17. MEASURED, the
open-loop cost does NOT keep shrinking: it falls 0.5048 -> 0.2197 over steps
500-2250 and then PLATEAUS (0.2083, 0.2158, 0.2133, 0.1893, 0.2026, 0.1969,
0.1947). So more steps cannot close Ga; the trade is real and stable at
-0.43 departure for +0.20 ADE.

Weight-space interpolation traces a DIFFERENT curve than early stopping, and it is
the published remedy for exactly this shape (WiSE-FT, Wortsman et al. 2022:
interpolating a fine-tune with its zero-shot base dominates early stopping on the
robustness/accuracy frontier). PUBLISHED precedent; our application is MEASURED here.

THE TRICK THAT KEEPS THIS HONEST: we change NOTHING in the evaluator. Each alpha is
written as a normal `delta_step{NNNNN}.pt` with the step number encoding
alpha*100, so `e1c_eval.py --ft-dir <this dir> --steps 10,20,...` adjudicates it
with literally the same code, the same paired episode-cluster bootstrap
(taniteval/ci.py, B=2000) and the same P1/P2/Ga/Gb1/Gb2/Gc logic.

BUILT-IN CONTROL: alpha=1.00 -> step 100 is bit-identical to the source delta, so
it MUST reproduce frontier row 4000 (dep_overall -0.4274, ade +0.1947). If it does
not, the harness is wrong and no other row is quotable.
"""
import sys, json, argparse
from pathlib import Path

# EXACTLY e1c_eval.py:38-40 — same roots, same order. Guessing a different root
# is how you load a capture copy from the wrong tree.
for _p in ("/workspace/e1c", "/workspace/e1b"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
# Bind e1a EXACTLY as e1c_eval.py:43 does (`e1a = EB.e1a`), not by a direct
# `import e1a_horizon` — e1b_eval binds a CAPTURE COPY, and importing the module
# by name could resolve a different tree and load the base with different code.
import e1b_eval as EB
e1a = EB.e1a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt",
                    default="/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt")
    ap.add_argument("--src-delta",
                    default="/workspace/e1c/refc-base-e1c-clsft/delta_step04000.pt")
    ap.add_argument("--preset", default="base")
    ap.add_argument("--out-dir", default="/workspace/e1c/alpha_sweep")
    ap.add_argument("--alphas", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.85,1.00")
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cpu"                      # pure weight arithmetic; no GPU needed

    model, base_step, _cfg = e1a.load_refc(a.base_ckpt, a.preset, dev)
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    d = torch.load(a.src_delta, map_location=dev, weights_only=False)
    ft = d["trainable"]
    src_step = int(d["step"])

    # Every trainable key must exist in the base, or the lerp is undefined.
    missing = [k for k in ft if k not in base_sd]
    assert not missing, f"delta keys absent from base: {missing[:5]}"
    # ... and must match in shape AND dtype, or the arithmetic silently upcasts.
    bad = [k for k in ft
           if tuple(ft[k].shape) != tuple(base_sd[k].shape)
           or ft[k].dtype != base_sd[k].dtype]
    assert not bad, f"shape/dtype mismatch on {bad[:5]}"

    n_changed = sum(1 for k in ft if not torch.equal(ft[k], base_sd[k]))
    print(f"[alpha] base_step={base_step} src_step={src_step} "
          f"trainable_keys={len(ft)} actually_changed={n_changed}", flush=True)
    assert n_changed > 0, "the delta is identical to the base — nothing to interpolate"

    manifest = []
    for al in [float(x) for x in a.alphas.split(",")]:
        step_code = int(round(al * 100))
        interp = {}
        for k, v_ft in ft.items():
            v_b = base_sd[k]
            if v_ft.is_floating_point():
                interp[k] = (v_b.to(torch.float64) * (1.0 - al)
                             + v_ft.to(torch.float64) * al).to(v_ft.dtype)
            else:
                # ints/bools cannot be interpolated; take the FT value verbatim and
                # say so, rather than silently rounding a counter into nonsense.
                interp[k] = v_ft.clone()
        p = out / f"delta_step{step_code:05d}.pt"
        torch.save({"trainable": interp, "step": step_code,
                    "_alpha": al, "_src_delta": str(a.src_delta),
                    "_src_step": src_step, "_base_ckpt": str(a.base_ckpt),
                    "_scheme": "WiSE-FT lerp (1-a)*base + a*ft, fp64 accumulate"}, p)
        # alpha=1 must be BIT-IDENTICAL to the source, else the control is void
        if al == 1.0:
            same = all(torch.equal(interp[k], ft[k]) for k in ft)
            assert same, "alpha=1.00 is not bit-identical to the source delta"
            print("[alpha] control OK: alpha=1.00 is bit-identical to the source")
        manifest.append({"alpha": al, "step_code": step_code, "path": str(p)})
        print(f"[alpha] wrote a={al:.2f} -> {p.name}", flush=True)

    (out / "ALPHA_MANIFEST.json").write_text(json.dumps({
        "_experiment": "E1d WiSE-FT alpha sweep over the E1c CL-SFT delta",
        "base_ckpt": a.base_ckpt, "src_delta": a.src_delta,
        "src_step": src_step, "base_step": base_step,
        "trainable_keys": len(ft), "keys_changed_vs_base": n_changed,
        "points": manifest,
        "_steps_arg": ",".join(str(m["step_code"]) for m in manifest),
    }, indent=1))
    print("[alpha] steps arg:", ",".join(str(m["step_code"]) for m in manifest))


if __name__ == "__main__":
    main()
