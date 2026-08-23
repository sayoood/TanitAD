"""S1b/S1c ADMISSIBILITY GATE — the D-SEL standard, applied to the two new flags.

Three things must hold before any S1 number is quotable, and each is the control
for a mistake the programme has already paid for:

  (a) CAPACITY. Both flags cost EXACTLY 0 parameters. (+272,001 once bought what
      +897 delivered; the capacity check is what caught it.)
  (b) BYTE-IDENTICAL WHEN OFF. The all-off state_dict has the same KEYS and the
      same VALUES as the pre-edit build, and the forward is BIT-identical across
      several flag combinations. A "default-off" lever that perturbs the default
      path is a silent re-baselining of every published number.
  (c) THE LEVER IS REAL AND BOUNDED. S1b must change `refined_logits` and must
      leave `anchor_traj` BIT-unchanged - the emitted fan is what the published
      oracle-in-fan (0.1914 / 0.1640) is defined on, and every D-SEL contrast is
      paired against it. S1c's masked CE must give a FINITE loss and FINITE
      gradients (a -inf-masked cross-entropy is the standard way to get NaN).

Compares against the PRE-EDIT file loaded from a backup path under a distinct
module name, so the "before" is real bytes rather than a remembered claim.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path.home() / "TanitAD"
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))

from tanitad.refs import refc as NOW                                # noqa: E402


def load_pre(path: Path):
    spec = importlib.util.spec_from_file_location("refc_pre", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["refc_pre"] = m
    spec.loader.exec_module(m)
    return m


def smoke(mod, **flags):
    cfg = mod.refc_smoke_config()
    for k, v in flags.items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    return mod.RefCModel(cfg).eval()


def main() -> int:
    pre = load_pre(Path.home() / "_s1_backup" / "refc_pre.py")
    R = {"host": "thor6", "torch": torch.__version__}

    # ---- (a) capacity: EXACTLY 0 ----------------------------------------
    with torch.device("meta"):
        n0 = NOW.param_breakdown(NOW.RefCModel(NOW.refc_config()))
        caps = {}
        for name, flags in (("sel_score_emitted", {"sel_score_emitted": True}),
                            ("sel_ce_reach", {"sel_ce_reach": True}),
                            ("both", {"sel_score_emitted": True,
                                      "sel_ce_reach": True}),
                            ("climbout_full", {"sel_refined": True,
                                               "sel_score_emitted": True,
                                               "sel_reach_clamp": True,
                                               "sel_ce_reach": True})):
            cfg = NOW.refc_config()
            for k, v in flags.items():
                setattr(cfg, k, v)
            caps[name] = NOW.param_breakdown(NOW.RefCModel(cfg))["total"] - n0["total"]
        n_dsel = NOW.param_breakdown(NOW.RefCModel(NOW.refc_select_config()))["total"]
    R["capacity"] = {"refc_config_total": n0["total"],
                     "registry_number": 104_191_577,
                     "matches_registry": n0["total"] == 104_191_577,
                     "param_delta_by_flag": caps,
                     "all_zero": all(v == 0 for v in caps.values()),
                     "dsel_preset_still_plus_385": n_dsel - n0["total"] == 385}

    # ---- (b) byte-identical when off, bit-identical in the forward -------
    a, b = smoke(pre), smoke(NOW)
    sa, sb = a.state_dict(), b.state_dict()
    keys_ok = set(sa) == set(sb)
    vals_ok = all(torch.equal(sa[k], sb[k]) for k in sa) if keys_ok else False
    frames = torch.randn(6, 4, 1, 64, 64)
    v0 = torch.tensor([3.0, 10.0, 20.0, 0.5, 7.0, 14.0])
    combos = [{}, {"sel_refined": True}, {"sel_reach_clamp": True},
              {"graft_cons": True}, {"seam_clamp": 1.0},
              {"sel_refined": True, "sel_reach_clamp": True, "seam_clamp": 1.0}]
    fwd = {}
    for c in combos:
        ma, mb = smoke(pre, **c), smoke(NOW, **c)
        mb.load_state_dict(ma.state_dict(), strict=True)
        with torch.no_grad():
            oa, ob = ma(frames, v0=v0, steps=2), mb(frames, v0=v0, steps=2)
        fwd[json.dumps(c, sort_keys=True) or "{}"] = {
            k: bool(torch.equal(oa[k], ob[k]))
            for k in ("anchor_logits", "refined_logits", "anchor_traj",
                      "sel_score", "sel_idx", "traj")}
    R["all_off_identity"] = {"state_dict_keys_equal": keys_ok,
                             "state_dict_values_equal": vals_ok,
                             "forward_bit_identical": fwd,
                             "ALL_PASS": keys_ok and vals_ok
                             and all(all(v.values()) for v in fwd.values())}

    # ---- (c1) S1b is real, and the EMITTED FAN IS UNTOUCHED --------------
    off = smoke(NOW, sel_refined=True)
    on = smoke(NOW, sel_refined=True, sel_score_emitted=True)
    on.load_state_dict(off.state_dict(), strict=True)
    with torch.no_grad():
        o_off = off(frames, v0=v0, steps=2)
        o_on = on(frames, v0=v0, steps=2)
        o_on0 = on(frames[:2], v0=v0[:2], steps=0)
    R["s1b"] = {
        "fan_bit_unchanged": bool(torch.equal(o_off["anchor_traj"],
                                              o_on["anchor_traj"])),
        "offset_bit_unchanged": bool(torch.equal(o_off["offset"], o_on["offset"])),
        "anchor_logits_bit_unchanged": bool(torch.equal(o_off["anchor_logits"],
                                                        o_on["anchor_logits"])),
        "refined_logits_CHANGED": not bool(torch.equal(o_off["refined_logits"],
                                                       o_on["refined_logits"])),
        "prefinal_is_the_old_refined": bool(torch.equal(
            o_on["prefinal_logits"], o_off["refined_logits"])),
        "sel_score_is_the_new_refined": bool(torch.equal(
            o_on["sel_score"], o_on["refined_logits"])),
        "inert_at_steps_0": bool(o_on0["refined_logits"] is o_on0["anchor_logits"]
                                 and "prefinal_logits" not in o_on0),
    }

    # ---- (c2) S1c: the masked CE is FINITE, forward and backward ---------
    m = smoke(NOW, sel_refined=True, sel_reach_clamp=True, sel_ce_reach=True)
    m.train()
    out = m(frames, v0=v0, steps=2)
    keep = out["reach_keep"]
    fan_err = out["anchor_traj"].norm(dim=-1).mean(-1)          # stand-in target
    ce_err = fan_err.masked_fill(~keep, float("inf"))
    neg = torch.finfo(out["sel_score"].dtype).min / 4
    ce_score = out["sel_score"].masked_fill(~keep, neg)
    tgt = ce_err.argmin(1)
    loss = F.cross_entropy(ce_score, tgt)
    loss.backward()
    gsum = sum(float(p.grad.abs().sum()) for p in m.parameters()
               if p.grad is not None)
    R["s1c"] = {
        "reach_keep_exported": True,
        "every_row_has_a_survivor": bool(keep.any(1).all()),
        "target_is_always_a_survivor": bool(keep.gather(1, tgt[:, None]).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "loss": round(float(loss), 6),
        "grad_sum_finite": bool(torch.isfinite(torch.tensor(gsum))),
        "grad_sum": round(gsum, 4),
        "ce_support_frac": round(float(keep.float().mean()), 4),
        "full_fan_support_frac": 1.0,
    }
    R["VERDICT"] = {
        "capacity_zero": R["capacity"]["all_zero"],
        "all_off_identical": R["all_off_identity"]["ALL_PASS"],
        "s1b_bounded": all(R["s1b"].values()),
        "s1c_finite": (R["s1c"]["loss_finite"] and R["s1c"]["grad_sum_finite"]
                       and R["s1c"]["target_is_always_a_survivor"]),
    }
    R["VERDICT"]["ALL_PASS"] = all(R["VERDICT"].values())
    print(json.dumps(R, indent=2))
    out_p = Path.home() / "s1_climbout" / "raw" / "verify_s1_identity.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(R, indent=2))
    return 0 if R["VERDICT"]["ALL_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
