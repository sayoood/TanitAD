"""Q1b — SPEC §5's phrase *"every decoder layer, EVERY DENOISING PASS"*.

⛔ THE PHRASE HAS NO REFERENT UNLESS THE SAMPLER IS BUILT. With
`decoder.sampler == "none"` the decoder stack runs exactly ONCE per forward, so
a probe on that build can only ever measure "every decoder layer" and would
read as a pass on the whole clause. Q1's arm was such a build (MEASURED: 1 FiLM
call per layer, `cfg_diffusion_steps` 2). This probe builds `sampler="ddim"`
and COUNTS the FiLM invocations per layer, then asserts that the SAME nav-
carrying `cond` object reaches every one of them.

⭐ THE DISCRIMINATOR IS THE COUNT, NOT THE DELTA. Two runs differing in nav
would show a delta on pass 1 alone and look identical to a build that
conditions every pass — so the count of invocations, and the per-invocation
identity of `cond`, are what separate them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_refcv6_probe_model as B            # noqa: E402
from q1_nav_reach_three_consumers import Capture, _install, _diff  # noqa: E402

OUT = HERE.parent / "raw"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    res = {}
    for sampler in ("none", "ddim"):
        model, cfg = B.build(sampler=sampler)
        cap = Capture()
        _install(model, cap)
        # record the python id() of every `cond` the FiLM sees, so "the same
        # condition reached every pass" is an IDENTITY claim, not a similarity.
        ids: dict[str, list[int]] = {}
        handles = []
        for i, ly in enumerate(model.core.decoder.layers):
            def mk(n):
                def h(m, args, kwargs, out):
                    c = args[1] if len(args) > 1 else kwargs.get("cond")
                    ids.setdefault(n, []).append(
                        {"id": id(c), "shape": list(c.shape),
                         "sum": round(float(c.float().sum()), 6)})
                return h
            handles.append(ly.film.register_forward_hook(
                mk(f"opdec.layer{i}.film"), with_kwargs=True))

        f, kw = B.batch(model, cfg, nav=1)
        with torch.no_grad():
            model(f, **kw)
        A = {k: list(v) for k, v in cap.calls.items()}
        idsA = {k: list(v) for k, v in ids.items()}
        cap.clear(); ids.clear()

        f, kw = B.batch(model, cfg, nav=2)
        with torch.no_grad():
            model(f, **kw)
        Bc = {k: list(v) for k, v in cap.calls.items()}
        idsB = {k: list(v) for k, v in ids.items()}

        arrival = _diff(A, Bc, "cond")
        res[sampler] = {
            "fingerprint": B.fingerprint(model, cfg),
            "film_calls_per_layer": {k: len(v) for k, v in A.items()
                                     if "opdec" in k},
            "cond_per_invocation_navA": idsA,
            "cond_per_invocation_navB": idsB,
            # ⭐ the identity claim: did EVERY invocation see a cond whose sum
            # differs between the two nav runs?
            "every_invocation_moves_with_nav": {
                k: all(
                    abs(a["sum"] - b["sum"]) > 0
                    for a, b in zip(idsA.get(k, []), idsB.get(k, [])))
                and len(idsA.get(k, [])) > 0
                for k in sorted(set(idsA) | set(idsB))},
            "ARRIVAL_cond_delta_navA_vs_navB": {
                k: v["max_abs_delta"] for k, v in arrival.items()},
        }
        for h in handles:
            h.remove()
        cap.close()

    p = OUT / "q1b_denoising_passes.json"
    p.write_text(json.dumps(res, indent=1))
    for s, r in res.items():
        print(f"== sampler={s} diffusion_steps={r['fingerprint']['diffusion_steps']}")
        print("   film_calls_per_layer :", r["film_calls_per_layer"])
        print("   every_invocation_moves_with_nav:",
              r["every_invocation_moves_with_nav"])
        print("   cond sums navA layer0:",
              [c["sum"] for c in r["cond_per_invocation_navA"]
               .get("opdec.layer0.film", [])])
        print("   cond ids  navA layer0:",
              [c["id"] for c in r["cond_per_invocation_navA"]
               .get("opdec.layer0.film", [])])
    print(f"\n[banked] {p}")


if __name__ == "__main__":
    main()
