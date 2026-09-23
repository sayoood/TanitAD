"""Q4b — the 8 -> n_anchors graft EXISTS. Is it FED?

⛔ `D-TLIGHT-1` is the registered defect where a label existed, was documented,
and never reached training. A zero-init `Linear(8, N)` that is built but never
called is bit-identical, in every artifact, to one that is called — because
zero-init makes its contribution 0.0 either way. So "the graft exists and is
zero-init" (Q4/D2) is NOT the same claim as "the tactical posterior reaches the
anchor prior", and only the second one is what SPEC §4(a) promises.

MEASUREMENT: hook `tac8_lat_to_anchor` / `tac8_lon_to_anchor` and record
(1) whether they are CALLED at all in a live forward, and (2) whether the
tensor they receive is bitwise equal to `planner_feeds`' `lat_logprob` /
`lon_logprob` — i.e. the tactical decoder's own detached posterior and not
something else with the same width.

⭐ CONTROL: the mutual-exclusion refusal at `refc.py:2911-2918` must FIRE when
both the 3-wide and the 8-wide prior are fed, so the "only one prior is on the
surface" claim is enforced rather than hoped for. A RED without its GREEN
cannot tell a guard from a brick, so the legal arm is run first.
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


def main():
    res = {}
    model, cfg = B.build(tac8_prior=True)
    dec = model.core.decoder

    seen: dict[str, list] = {}
    feeds: dict[str, torch.Tensor] = {}
    handles = []
    for nm in ("tac8_lat_to_anchor", "tac8_lon_to_anchor",
               "lat_to_anchor", "lon_to_anchor"):
        m = getattr(dec, nm, None)
        if m is None:
            continue

        def mk(n):
            def h(mod, args, kwargs):
                x = args[0] if args else kwargs.get("input")
                seen.setdefault(n, []).append(x.detach().clone())
                return None
            return h
        handles.append(m.register_forward_pre_hook(mk(nm), with_kwargs=True))

    # capture what the TACTICAL decoder actually emitted, through the exact
    # function the model uses to build the feeds
    td = model.tac_decoder_v6
    orig = v6tac.planner_feeds

    def spy(out, **kw):
        r = orig(out, **kw)
        feeds.update({k: v.detach().clone() for k, v in r.items()})
        return r
    v6tac.planner_feeds = spy
    try:
        f, kw = B.batch(model, cfg, nav=1)
        with torch.no_grad():
            out = model(f, **kw)
    finally:
        v6tac.planner_feeds = orig
        for h in handles:
            h.remove()

    res["graft_called"] = {k: len(v) for k, v in seen.items()}
    res["graft_present_but_uncalled"] = sorted(
        nm for nm in ("tac8_lat_to_anchor", "tac8_lon_to_anchor",
                      "lat_to_anchor", "lon_to_anchor")
        if getattr(dec, nm, None) is not None and nm not in seen)
    res["planner_feeds_captured"] = sorted(feeds)
    for a, b in (("tac8_lat_to_anchor", "lat_logprob"),
                 ("tac8_lon_to_anchor", "lon_logprob")):
        if a in seen and b in feeds:
            x, y = seen[a][0], feeds[b]
            res[f"{a}_input_is_{b}"] = {
                "shape_in": list(x.shape), "shape_feed": list(y.shape),
                "bitwise_equal": bool(torch.equal(x, y)),
                "max_abs_delta": float((x - y).abs().max())}
        else:
            res[f"{a}_input_is_{b}"] = "NOT MEASURED (graft not called)"
    res["tac8_input_width"] = (list(seen["tac8_lat_to_anchor"][0].shape)
                               if "tac8_lat_to_anchor" in seen else None)

    # ---------------------------------------------------- the CONTROL/mutation
    # GREEN first: the legal arm above already ran without raising.
    res["GREEN_legal_arm_ran"] = True
    # RED: feed BOTH priors to the decoder and require the refusal. ⛔ The
    # decoder's real arguments are CAPTURED from a live forward rather than
    # invented — a hand-made `fmap` of the wrong channel width dies inside a
    # Linear and reads exactly like "the guard did not fire" (it did that on
    # this instrument's first run, and the row is kept so the correction is
    # legible).
    grab = {}

    def _grab(mod, args, kwargs):
        grab["fmap"], grab["m"] = args[0].detach(), args[1].detach()
        grab["kwargs"] = {k: v for k, v in kwargs.items()}
        return None
    h = dec.register_forward_pre_hook(_grab, with_kwargs=True)
    f2, kw2 = B.batch(model, cfg, nav=1)
    with torch.no_grad():
        model(f2, **kw2)
    h.remove()
    res["decoder_call_kwargs_seen"] = sorted(grab.get("kwargs", {}))
    b_ = grab["fmap"].shape[0]
    legal = dict(grab["kwargs"])
    legal.pop("lat_prior", None)
    legal.pop("lon_prior", None)
    legal["tac_lat_prior"] = torch.log_softmax(torch.rand(b_, 8), -1)
    try:
        with torch.no_grad():
            dec(grab["fmap"], grab["m"], **legal)
        res["GREEN_tac8_only_ran"] = True
    except Exception as e:                                   # pragma: no cover
        res["GREEN_tac8_only_ran"] = f"FAILED: {type(e).__name__} {str(e)[:160]}"
    bad = dict(legal)
    bad["lat_prior"] = torch.log_softmax(torch.rand(b_, 3), -1)
    try:
        with torch.no_grad():
            dec(grab["fmap"], grab["m"], **bad)
        res["RED_both_priors_refused"] = "NOT REFUSED (guard is INERT)"
    except ValueError as e:
        res["RED_both_priors_refused"] = (
            "REFUSED: " + str(e)[:130] if "BOTH" in str(e)
            else f"raised, but NOT the guard: {str(e)[:160]}")
    except Exception as e:
        res["RED_both_priors_refused"] = f"other error: {type(e).__name__} {str(e)[:160]}"

    p = OUT / "q4b_tac8_fed.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    print(f"\n[banked] {p}")


if __name__ == "__main__":
    main()
