# -*- coding: utf-8 -*-
"""MUTATION PROOF for --no-strategic (D-STRAT-BYPASS-1).

Three assertions, each with a control that MUST be able to fail:

  P1  ON  -> the strategic path contributes NOTHING to the emitted plan.
      Mutate every strategic parameter; the plan must be BIT-IDENTICAL.
  P1c CONTROL -> the SAME mutation on an OFF build MUST change the plan.
      Without this, P1 could pass because the probe cannot see anything.
  P2  the nav command still reaches the OPERATIVE and TACTICAL planners
      under the bypass (that is the PI's directive, not a side effect).

ASCII-only in every print(): this runs on a cp1252 dev box.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stack"))

from tanitad.refs import refc          # noqa: E402
from tanitad.refs import refc_v3 as v3  # noqa: E402

OUT = {}


# --------------------------------------------------------------------------
# the strategic parameter set, named explicitly (never a substring guess)
# --------------------------------------------------------------------------
def strategic_params(model):
    """Every parameter that belongs to the STRATEGIC layer, by module path."""
    roots = [
        "core.strategic",            # StrategicCtx GRU + proj
        "core.decoder.ctx_to_cond",  # S-BYPASS-1 seam
        "core.route_head",           # the strategic aux READOUT
        "str_goal_head",             # E3
        "gstr_embed", "gstr_film",   # E4 (the tactical seam)
        "nav_to_str",                # E13 route -> strategic
        "ego_to_str",                # E11' ego -> strategic
        "gp_head", "gp_cond",        # E15 goal point (off on refcv4b)
    ]
    got = {}
    for name, p in model.named_parameters():
        for r in roots:
            if name == r or name.startswith(r + "."):
                got[name] = p
                break
    return got, roots


#: THE EMITTED PLAN **and every surface between the strategic layer and it**.
#: The first probe of this file listed only the core-side keys and P1 passed
#: while never looking at `z_tac` -- the tactical state the strategic goal
#: FiLMs, and the one thing E4 exists to move. A proof that does not look at
#: the seam it is proving is not a proof.
PLAN_KEYS = [
    # the emitted plan + the selection that produced it
    "traj", "wp_seq", "sel_idx", "sel_idx_base", "sel_score", "sel_score_v3",
    "anchor_logits", "refined_logits", "offset", "anchor_traj", "traj_base",
    # the OPERATIVE decoder's other readouts
    "goal_dist", "goal_gate_value", "goal_score_absmean",
    # the CORE tactical head (image branch)
    "maneuver_logits", "lat_logits", "lon_logits",
    "lat_decision", "lon_decision", "maneuver_decision",
    # the TACTICAL BRAIN -- z_tac is what E4 FiLMs and what E7 feeds the
    # decoder, so it is the load-bearing surface for S-BYPASS-2
    "z_tac", "lat_logits_tac", "lon_logits_tac", "g_tac", "g_tac_delta",
    "goal_point_tac", "goal_point_free", "bank_speed_pred",
]


def plan_of(out):
    """The EMITTED PLAN plus every tactical decision surface downstream."""
    return {k: out[k].detach().clone() for k in PLAN_KEYS if k in out
            and torch.is_tensor(out[k])}


def differs(a, b):
    """Which keys differ BITWISE, and by how much."""
    d = {}
    for k in a:
        if k not in b:
            d[k] = "MISSING"
            continue
        if a[k].shape != b[k].shape:
            d[k] = "SHAPE"
            continue
        if not torch.equal(a[k], b[k]):
            d[k] = float((a[k].float() - b[k].float()).abs().max())
    return d


def build(no_strategic, seed=0):
    torch.manual_seed(seed)
    cfg = v3.refc_v3_smoke_config(True)          # hier arm, tiny
    cfg.core.no_strategic = bool(no_strategic)
    m = v3.RefCV3Model(cfg)
    m.eval()
    return m, cfg


def inputs(cfg, seed=1234, nav_idx=0):
    g = torch.Generator().manual_seed(seed)
    h, w = cfg.core.encoder.image_hw()
    b = 3
    frames = torch.rand(b, cfg.core.window, cfg.core.encoder.in_channels, h, w,
                        generator=g)
    nav = torch.full((b,), int(nav_idx), dtype=torch.long)
    v0 = torch.tensor([3.0, 7.0, 11.0])
    return dict(frames=frames, nav_cmd=nav, v0=v0)


def run(model, kw):
    with torch.no_grad():
        return model(**kw)


def mutate(params, seed=99, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(params.items()):
            p.copy_(torch.randn(p.shape, generator=g) * scale)


# ==========================================================================
print("=" * 74)
print("P1 / P1c  MUTATION PROOF -- does the strategic path reach the plan?")
print("=" * 74)

results = {}
for tag, flag in (("ON__bypassed", True), ("OFF_today", False)):
    model, cfg = build(flag)
    kw = inputs(cfg)
    sp, roots = strategic_params(model)
    before = plan_of(run(model, kw))
    mutate(sp)
    after = plan_of(run(model, kw))
    d = differs(before, after)
    results[tag] = dict(n_strategic_params=len(sp),
                        n_strategic_tensors=len(sp),
                        n_elem=int(sum(p.numel() for p in sp.values())),
                        changed=d, plan_keys=sorted(before))
    print(f"\n[{tag}]  no_strategic={flag}")
    print(f"  strategic tensors mutated : {len(sp)} "
          f"({sum(p.numel() for p in sp.values())} elements)")
    print(f"  plan keys compared        : {sorted(before)}")
    print(f"  keys CHANGED by mutation  : {sorted(d) if d else 'NONE (bit-identical)'}")
    if d:
        for k, vv in sorted(d.items()):
            print(f"      {k:18s} max|delta| = {vv}")

OUT["P1_names"] = sorted(strategic_params(build(True)[0])[0])
OUT["P1"] = results

p1_ok = not results["ON__bypassed"]["changed"]
p1c_ok = bool(results["OFF_today"]["changed"])
print()
print(f"P1   ON  bypasses (plan bit-identical under mutation) : "
      f"{'PASS' if p1_ok else 'FAIL'}")
print(f"P1c  OFF control (same mutation MUST move the plan)   : "
      f"{'PASS' if p1c_ok else 'FAIL -- the probe cannot detect anything'}")

# ==========================================================================
print()
print("=" * 74)
print("P2  under the bypass, does the NAV COMMAND still reach the planners?")
print("=" * 74)

model, cfg = build(True)
kw0 = inputs(cfg, nav_idx=0)          # 'follow'
kw2 = inputs(cfg, nav_idx=2)          # a different command
a = plan_of(run(model, kw0))
b = plan_of(run(model, kw2))
d_nav = differs(a, b)
print(f"\n  nav_cmd 0 -> 2, at INIT")
print(f"  keys changed: {sorted(d_nav) if d_nav else 'NONE'}")
OUT["P2_init"] = {k: v for k, v in d_nav.items()}

# E13's nav->tactical is ZERO-INIT by construction, so it is bit-inert until
# trained. Prove the PATHWAY EXISTS by making that projection non-zero.
with torch.no_grad():
    torch.manual_seed(7)
    model.nav_to_tac.weight.normal_(0, 1.0)
    model.nav_to_tac.bias.normal_(0, 1.0)
a2 = plan_of(run(model, kw0))
b2 = plan_of(run(model, kw2))
d_nav2 = differs(a2, b2)
print(f"\n  same, after making E13 nav_to_tac non-zero (it is ZERO-INIT)")
print(f"  keys changed: {sorted(d_nav2) if d_nav2 else 'NONE'}")
OUT["P2_nav_to_tac_live"] = {k: v for k, v in d_nav2.items()}

TAC_KEYS = {"z_tac", "lat_logits_tac", "lon_logits_tac", "g_tac",
            "g_tac_delta"}
p2_op = bool(d_nav)                     # operative path live at init
p2_tac = bool(TAC_KEYS & set(d_nav2))   # tactical state responds to nav
print()
print(f"P2a  nav -> OPERATIVE (measurement -> decoder cond), live at init : "
      f"{'PASS' if p2_op else 'FAIL'}")
print(f"P2b  nav -> TACTICAL (E13 nav_to_tac), pathway present            : "
      f"{'PASS' if p2_tac else 'FAIL'}")

OUT["verdict"] = dict(P1=p1_ok, P1c=p1c_ok, P2a=p2_op, P2b=p2_tac)
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "MUTATION_PROOF.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=str)
print()
print("ALL: " + ("PASS" if all(OUT["verdict"].values()) else "FAIL"))
sys.exit(0 if all(OUT["verdict"].values()) else 1)
