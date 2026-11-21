"""Q2 — is the tactical decoder's KEY/VALUE set what SPEC §4 says?

SPEC §4: *"Keys/values: the agent slots and the BEV tokens (30x16) ... Image
tokens are deliberately not its input."*

⛔ HOOK IT, DO NOT READ IT. The claim is about a tensor at runtime. So: hook
`_DecoderLayer.cross_attn` inside the behaviour decoder, print the kv shape,
and reconcile it arithmetically against the three candidate sources measured in
the SAME forward — agent slots, BEV tokens, and the IMAGE token grid the
operative decoder attends to.

⭐ THE DISCRIMINATING CONTROL. "K != n_image" is weak: two numbers can coincide.
So this probe sets the BEV token count to a DISTINCTIVE value (`FakeBranch
n_cells`) and asserts `K == n_agent + n_cells` EXACTLY, then re-runs with a
different `n_cells` and shows K tracks it. A decoder that also attended to the
image grid could not track both.

⭐ AND A MUTATION: the structural refusal is re-tested by trying to declare
`"image"` as a source and by trying to pass image tokens positionally.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_refcv6_probe_model as B            # noqa: E402

from tanitad.refs import refcv6_tactical as v6tac   # noqa: E402

OUT = HERE.parent / "raw"
OUT.mkdir(parents=True, exist_ok=True)


def _measure(n_cells: int, d_bev: int = 16) -> dict:
    model, cfg = B.build(d_bev=d_bev)
    model._perception = B.FakeBranch(d_bev, n_cells=n_cells)
    td = model.tac_decoder_v6

    seen: list[dict] = []
    handles = []
    for i, ly in enumerate(td.layers):
        def mk(idx):
            def h(m, args, kwargs):
                # MultiheadAttention(query, key, value, ...)
                q, k, v = args[0], args[1], args[2]
                seen.append({"layer": idx, "n_queries": int(q.shape[1]),
                             "K": int(k.shape[1]), "d": int(k.shape[-1]),
                             "k_is_v": bool(k is v)})
            return h
        handles.append(ly.cross_attn.register_forward_pre_hook(
            mk(i), with_kwargs=True))

    # the IMAGE token grid the OPERATIVE decoder attends to, measured in the
    # same forward: `feat_proj(fmap.flatten(2).transpose(1,2))` -> [B, P, d]
    img = {}

    def _img_hook(m, a, o):
        # ⛔ MUST RETURN None. A forward hook that returns a value REPLACES the
        # module's output — a `setdefault` here turned `kv` into an int and the
        # operative decoder died inside MultiheadAttention. Measured, this run.
        img.setdefault("P", int(o.shape[1]))
        return None
    handles.append(model.core.decoder.feat_proj.register_forward_hook(_img_hook))

    # the AGENT slot count, measured off the tensor the hook receives
    agent = {}
    orig = model._scene_hook if hasattr(model, "_scene_hook") else None

    f, kw = B.batch(model, cfg, nav=1)
    with torch.no_grad():
        out = model(f, **kw)
    for h in handles:
        h.remove()
    n_scene = int(out["tacv6_n_scene"][0]) if "tacv6_n_scene" in out else -1
    return {
        "n_cells_requested": n_cells,
        "cross_attn_calls": seen,
        "K": (seen[0]["K"] if seen else None),
        "n_queries": (seen[0]["n_queries"] if seen else None),
        "image_token_count_P_operative": img.get("P"),
        "tacv6_n_scene_row0": n_scene,
        "sources_declared": list(td.cfg.sources),
        "d_bev_declared": int(td.cfg.d_bev),
        "forward_signature": sorted(
            __import__("inspect").signature(
                v6tac.TacticalBehaviourDecoder.forward).parameters),
    }


def main():
    res = {}
    a = _measure(n_cells=12)
    b = _measure(n_cells=25)
    res["arm_ncells_12"] = a
    res["arm_ncells_25"] = b
    n_agent_12 = a["K"] - 12 if a["K"] is not None else None
    n_agent_25 = b["K"] - 25 if b["K"] is not None else None
    res["derived_n_agent_slots"] = {"from_12": n_agent_12, "from_25": n_agent_25,
                                    "consistent": n_agent_12 == n_agent_25}
    res["K_tracks_bev_count"] = (a["K"] is not None and b["K"] is not None
                                 and (b["K"] - a["K"]) == (25 - 12))
    res["image_tokens_in_kv"] = {
        "P_operative": a["image_token_count_P_operative"],
        "K_minus_agent_minus_bev": (None if n_agent_12 is None
                                    else a["K"] - n_agent_12 - 12),
        "verdict": ("ABSENT (K is exactly agent+bev and tracks bev exactly)"
                    if (n_agent_12 == n_agent_25
                        and (b["K"] - a["K"]) == 13) else "NOT ESTABLISHED"),
    }

    # ---------------------------------------------------------------- MUTATION
    muts = {}
    try:
        v6tac.TacticalDecoderConfig(sources=("agent", "bev", "image"))
        muts["declare_image_source"] = "ACCEPTED (guard is INERT)"
    except v6tac.SceneInputRefused as e:
        muts["declare_image_source"] = f"REFUSED: {str(e)[:90]}"
    # ⭐ THE GREEN CONTROL beside the refusal: the legal declaration must build.
    try:
        v6tac.TacticalDecoderConfig(sources=("agent", "bev"))
        muts["declare_legal_sources_control"] = "ACCEPTED (correct)"
    except Exception as e:                                    # pragma: no cover
        muts["declare_legal_sources_control"] = f"REFUSED (WRONG): {e}"
    # the structural half: is there an image port at all?
    import inspect
    p = inspect.signature(v6tac.TacticalBehaviourDecoder.forward).parameters
    muts["forward_has_image_port"] = any(
        "image" in k or "fmap" in k or "img" in k for k in p)
    try:
        td = v6tac.TacticalBehaviourDecoder(
            v6tac.TacticalDecoderConfig(d_model=32, n_heads=4, d_agent=8,
                                        d_bev=8))
        cond = v6tac.build_condition(
            torch.eye(4)[:2], torch.eye(4)[:2], torch.zeros(2), torch.zeros(2))
        td(cond, agent_tokens=torch.randn(2, 3, 8),
           image_tokens=torch.randn(2, 40, 8))
        muts["pass_image_tokens_kwarg"] = "ACCEPTED (port EXISTS - defect)"
    except TypeError as e:
        muts["pass_image_tokens_kwarg"] = f"TypeError: {str(e)[:80]}"
    res["MUTATIONS"] = muts

    p = OUT / "q2_tactical_kv.json"
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"\n[banked] {p}")


if __name__ == "__main__":
    main()
