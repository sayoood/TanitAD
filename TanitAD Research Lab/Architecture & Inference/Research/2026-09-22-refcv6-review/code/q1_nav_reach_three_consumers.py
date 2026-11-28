"""Q1 — does NAV actually REACH all three consumers SPEC_REFCV6_V2 §5 names?

  (a) the FiLM condition  -> EVERY decoder layer AND EVERY denoising pass
  (b) the tactical decoder
  (c) SELECTION (the nav-compliance term behind a zero-init gate)

⛔ METHOD IS INTERVENTIONAL, NOT A CODE READING. `D-TLIGHT-1` is a registered
defect where a label never reached training while every doc said it did. So:
run the SAME batch twice, changing ONLY `nav_cmd`, and hook the tensors at the
three consumers. A consumer whose captured tensor is bit-identical across the
two runs did NOT receive nav, whatever the source says.

⛔ EVERY POSITIVE IS PAIRED WITH A NEGATIVE CONTROL IN THE SAME BREATH. Two
runs with the SAME nav must be bit-identical at the same hooks; without that
row a "difference" could be forward nondeterminism (dropout, a sampler draw)
rather than nav. A table of only positives cannot tell a live channel from a
noisy one.

⚠️ ARRIVAL AND INFLUENCE ARE TWO QUESTIONS. Several of these grafts are
ZERO-INIT by design, so at step 0 nav can ARRIVE at a module and move its
output by exactly 0.0. That is not a defect — but it is also not "nav reaches
the plan". Both are measured and reported separately:
  ARRIVAL   = the tensor entering the module differs under the intervention
  INFLUENCE = the tensor LEAVING the module differs under the intervention
and INFLUENCE is re-measured after the zero-init grafts are given non-zero
weights (the post-training state), because a graft that is zero-init AND never
receives gradient is dead forever and the two cases must be distinguishable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_refcv6_probe_model as B            # noqa: E402

from tanitad.refs import refc as refcmod        # noqa: E402
from tanitad.refs import refcv6_tactical as v6tac  # noqa: E402

OUT = HERE.parent / "raw"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #
class Capture:
    """Records, per named module, every (input, output) tensor pair it sees."""

    def __init__(self):
        self.calls: dict[str, list[dict]] = {}
        self.handles = []

    def watch(self, name: str, mod, *, cond_arg: int | None = None):
        def hook(m, args, kwargs, out):
            rec = {}
            allargs = list(args) + list(kwargs.values())
            if cond_arg is not None and len(args) > cond_arg:
                rec["cond"] = args[cond_arg].detach().clone()
            elif "cond" in kwargs:
                rec["cond"] = kwargs["cond"].detach().clone()
            elif allargs and torch.is_tensor(allargs[0]):
                rec["cond"] = allargs[0].detach().clone()
            if torch.is_tensor(out):
                rec["out"] = out.detach().clone()
            elif isinstance(out, (tuple, list)) and torch.is_tensor(out[0]):
                rec["out"] = out[0].detach().clone()
            elif isinstance(out, dict):
                rec["out"] = torch.cat(
                    [v.detach().reshape(v.shape[0], -1).float()
                     for v in out.values()
                     if torch.is_tensor(v) and v.dim() >= 1], dim=-1)
            self.calls.setdefault(name, []).append(rec)
        self.handles.append(mod.register_forward_hook(hook, with_kwargs=True))

    def clear(self):
        self.calls = {}

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _install(model, cap: Capture):
    core = model.core
    # (a) the OPERATIVE decoder's per-layer FiLM. `CrossAttnLayer.film(x, cond)`
    #     -> cond is positional arg 1. Counting calls answers "every layer AND
    #     every denoising pass" without reading the loop.
    dec = core.decoder
    for i, ly in enumerate(dec.layers):
        cap.watch(f"opdec.layer{i}.film", ly.film, cond_arg=1)
    # (b) the TACTICAL decoder: the per-layer query FiLM, and the decoder itself
    td = getattr(model, "tac_decoder_v6", None)
    if td is not None:
        for i, ly in enumerate(td.layers):
            cap.watch(f"tacv6.layer{i}.film", ly.film, cond_arg=1)
        cap.watch("tacv6.decoder", td, cond_arg=0)
    # (c) SELECTION: `navc_gate` is an nn.Parameter, NOT a module, so there is
    #     nothing to hook. The measurable surface is the CALL to
    #     `refcv6_selection.nav_compliance_prior` and the telemetry it writes
    #     into `sel_tele` — both are recorded by `NavCompSpy` below.
    #     ⛔ The zero-init gate means its contribution to the score is EXACTLY
    #     0.0 at step 0; that is arrival without influence and is reported so.
    return cap


class NavCompSpy:
    """Wrap `refcv6_selection.nav_compliance_prior` to record every call.

    ⛔ This is a MEASUREMENT of the call, not a code reading: it records the
    nav tensor the selection path actually passed and the compliance vector it
    got back. A selection seam that never fires leaves an EMPTY log, which is
    the honest negative — `D-TLIGHT-1`'s shape.
    """

    def __init__(self):
        self.calls: list[dict] = []
        self._orig = None

    def __enter__(self):
        from tanitad.refs import refcv6_selection as v6sel
        self._mod, self._orig = v6sel, v6sel.nav_compliance_prior

        def spy(cand, nav_cmd, **kw):
            out = self._orig(cand, nav_cmd, **kw)
            self.calls.append({
                "nav_cmd": nav_cmd.detach().clone(),
                "prior": out[0].detach().clone(),
                "kw": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                       for k, v in kw.items()},
            })
            return out
        v6sel.nav_compliance_prior = spy
        # the call site holds a MODULE-LEVEL reference through `v6sel.`, so
        # rebinding the attribute is enough; assert it, do not assume it.
        return self

    def __exit__(self, *a):
        self._mod.nav_compliance_prior = self._orig


def _diff(a: dict, b: dict, key: str) -> dict:
    """Per-call max-abs difference of `key` between two capture runs."""
    rows = []
    for name in sorted(set(a) | set(b)):
        ca, cb = a.get(name, []), b.get(name, [])
        n = min(len(ca), len(cb))
        mx, any_missing = 0.0, False
        for i in range(n):
            ta, tb = ca[i].get(key), cb[i].get(key)
            if ta is None or tb is None:
                any_missing = True
                continue
            mx = max(mx, float((ta.float() - tb.float()).abs().max()))
        rows.append({"module": name, "n_calls_A": len(ca), "n_calls_B": len(cb),
                     "max_abs_delta": mx, "tensor_missing": any_missing})
    return {r["module"]: r for r in rows}


def run(model, cfg, *, nav, seed=0, cap=None):
    cap.clear()
    f, kw = B.batch(model, cfg, nav=nav, seed=seed)
    with torch.no_grad():
        out = model(f, **kw)
    return {k: list(v) for k, v in cap.calls.items()}, out


def main():
    torch.use_deterministic_algorithms(False)
    model, cfg = B.build()
    fp = B.fingerprint(model, cfg)
    cap = Capture()
    _install(model, cap)

    # --------------------------------------------------------------- ARRIVAL
    with NavCompSpy() as spy:
        A, outA = run(model, cfg, nav=1, cap=cap)      # NAV_TURN_L
        nc_A = list(spy.calls); spy.calls.clear()
        Bc, outB = run(model, cfg, nav=2, cap=cap)     # NAV_TURN_R
        nc_B = list(spy.calls); spy.calls.clear()
        A2, _ = run(model, cfg, nav=1, cap=cap)        # same-nav control
        nc_A2 = list(spy.calls)
        # nav=None: the selection seam must NOT fire (`nav_cmd_sel=None`).
        spy.calls.clear()
        _, outNone = run(model, cfg, nav=None, cap=cap)
        nc_none = list(spy.calls)

    def _nc_summary(calls):
        return {"n_calls": len(calls),
                "nav_values": sorted({int(v) for c in calls
                                      for v in c["nav_cmd"].reshape(-1).tolist()}),
                "prior_mean": ([round(float(c["prior"].float().mean()), 6)
                                for c in calls] if calls else []),
                "tau_kw": (calls[0]["kw"] if calls else None)}

    arrival = _diff(A, Bc, "cond")
    control = _diff(A, A2, "cond")
    influence = _diff(A, Bc, "out")
    influence_ctl = _diff(A, A2, "out")

    # ------------------------------------------------- INFLUENCE, grafts live
    # ⭐ THE ZERO-INIT GRAFTS ARE GIVEN NON-ZERO WEIGHTS, so "moves by 0.0"
    # can be told apart from "is wired to nothing". A graft that STILL does
    # not move after its own weights are non-zero is not wired.
    model2, cfg2 = B.build()
    with torch.no_grad():
        td = model2.tac_decoder_v6
        for ly in td.layers:
            ly.film.proj.weight.normal_(0.0, 0.05)
            ly.film.proj.bias.normal_(0.0, 0.05)
        g = getattr(model2.core.decoder, "navc_gate", None)
        if g is not None:            # nn.Parameter, zero-init by design
            g.data.fill_(0.7)
        bg = getattr(model2, "tac_behaviour_gate_v6", None)
        if bg is not None:
            for p in getattr(bg, "parameters", lambda: [])():
                p.data.normal_(0.0, 0.05)
    cap2 = Capture()
    _install(model2, cap2)
    A3, outA3 = run(model2, cfg2, nav=1, cap=cap2)
    B3, outB3 = run(model2, cfg2, nav=2, cap=cap2)
    A4, outA4 = run(model2, cfg2, nav=1, cap=cap2)
    influence_live = _diff(A3, B3, "out")
    influence_live_ctl = _diff(A3, A4, "out")

    # -------------------------------------------- end-to-end plan/selection
    def _e2e(oa, ob, keys):
        r = {}
        for k in keys:
            ta, tb = oa.get(k), ob.get(k)
            if not (torch.is_tensor(ta) and torch.is_tensor(tb)):
                r[k] = None
                continue
            r[k] = float((ta.float() - tb.float()).abs().max())
        return r

    e2e_keys = ["traj", "sel_score", "sel_idx", "sel_score_v3",
                "tacv6_goal_logits", "tacv6_lat_logits", "tacv6_lon_logits",
                "route_logits", "g_str", "g_tac"]
    e2e_zero = _e2e(outA, outB, e2e_keys)
    e2e_live = _e2e(outA3, outB3, e2e_keys)
    e2e_live_ctl = _e2e(outA3, outA4, e2e_keys)   # same-nav control
    sel_tele_A = outA.get("sel_tele")
    sel_tele_B = outB.get("sel_tele")

    # ------------------------------------------------ the pass/layer census
    n_layers = len(model.core.decoder.layers)
    calls_per_layer = {k: len(v) for k, v in A.items() if "opdec" in k}
    diffusion_steps = int(getattr(cfg.core.decoder, "diffusion_steps",
                                  getattr(cfg.core, "diffusion_steps", -1)))

    res = {
        "fingerprint": fp,
        "nav_A": 1, "nav_B": 2,
        "opdec_n_layers": n_layers,
        "opdec_film_calls_per_layer": calls_per_layer,
        "cfg_diffusion_steps": diffusion_steps,
        "ARRIVAL_cond_delta_navA_vs_navB": arrival,
        "ARRIVAL_control_same_nav": control,
        "INFLUENCE_out_delta_zeroinit": influence,
        "INFLUENCE_control_same_nav_zeroinit": influence_ctl,
        "INFLUENCE_out_delta_grafts_live": influence_live,
        "INFLUENCE_control_same_nav_grafts_live": influence_live_ctl,
        "E2E_output_delta_zeroinit": e2e_zero,
        "E2E_output_delta_grafts_live": e2e_live,
        "E2E_control_same_nav_grafts_live": e2e_live_ctl,
        "SELECTION_navcomp_calls_navA": _nc_summary(nc_A),
        "SELECTION_navcomp_calls_navB": _nc_summary(nc_B),
        "SELECTION_navcomp_calls_navA_repeat": _nc_summary(nc_A2),
        "SELECTION_navcomp_calls_nav_NONE": _nc_summary(nc_none),
        "SELECTION_sel_tele_navA": (
            {k: v for k, v in sel_tele_A.items() if "nav" in k or "gate" in k}
            if isinstance(sel_tele_A, dict) else str(type(sel_tele_A))),
        "SELECTION_sel_tele_navB": (
            {k: v for k, v in sel_tele_B.items() if "nav" in k or "gate" in k}
            if isinstance(sel_tele_B, dict) else str(type(sel_tele_B))),
    }
    p = OUT / "q1_nav_reach.json"
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"\n[banked] {p}")
    cap.close(); cap2.close()


if __name__ == "__main__":
    main()
