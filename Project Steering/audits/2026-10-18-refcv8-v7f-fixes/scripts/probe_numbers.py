"""Re-measure the numbers behind AUDIT.md (R3 / R4 / R6 / R2 / R1) with the SAME helpers as the tests; writes
raw/probe_numbers.json. CPU only. Run from a tree that holds stack/ (patched or tip for R3/R4/R6/R2)."""
import json, sys, types
from pathlib import Path
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:/Users/Admin/r8v7f/tree_patched")
sys.path[:0] = [str(ROOT / "stack"), str(ROOT / "stack/scripts"), str(ROOT / "stack/tests")]
import torch
import _refcv8_rig as R
import test_refcv8_v7f_equivalence as E
from tanitad.refs import refc_v3 as v3

T = R.trainer()
res = {}

# ---- R3 ------------------------------------------------------------------------------------------------------ #
cfg, m = E._tac_model(T)
tok = list(T.v7l.TAC_GOAL_TOKENS)
red, green, band = (tok.index("TRAFFIC_LIGHT_REACT_RED"), tok.index("TRAFFIC_LIGHT_REACT_GREEN"), tok.index("SPEED_BAND"))
b0 = E._loss_batch(T, cfg)
setc = lambda c: (lambda b: b["tac_goal_y"].__setitem__((slice(None), c), 1.0))
r = E._with(b0, a=setc(red)); g = E._with(b0, a=setc(green))
w0 = E._with(b0, a=lambda b: b["tac_goal_w"].__setitem__((slice(None), red), 0.0))
w0f = E._with(w0, a=setc(red))
res["R3_loss_deltas"] = {
    "control_same_batch_same_seed": E._delta(T, m, b0, E._with(b0)),
    "TL_RED_cell_0_to_1": E._delta(T, m, b0, r),
    "TL_GREEN_cell_0_to_1": E._delta(T, m, b0, g),
    "TL_RED_vs_GREEN_labelled": E._delta(T, m, r, g),
    "control_masked_class_SPEED_BAND_flipped": E._delta(T, m, b0, E._with(b0, a=setc(band))),
    "control_w0_cell_flipped": E._delta(T, m, w0, w0f),
    "lat_label_0to6_to_6to6": E._delta(T, m, b0, E._with(b0, a=lambda b: b.__setitem__("lat_v7", torch.tensor([6, 6])))),
    "lon_label_to_7": E._delta(T, m, b0, E._with(b0, a=lambda b: b.__setitem__("lon_v7", torch.tensor([7, 7])))),
}
la, lb = E._loss(T, m, r), E._loss(T, m, b0)
params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
gr = torch.autograd.grad(la - lb, [p for _n, p in params], allow_unused=True)
reach = {n: (0.0 if x is None else float(x.abs().sum())) for (n, _p), x in zip(params, gr)}
res["R3_TL_term_gradient_abs_sum"] = {
    "tac_decoder_v6.*": sum(v for n, v in reach.items() if n.startswith("tac_decoder_v6.")),
    "n_tac_decoder_tensors_reached": sum(1 for n, v in reach.items() if n.startswith("tac_decoder_v6.") and v > 0),
    "core.encoder.*": sum(v for n, v in reach.items() if n.startswith("core.encoder.")),
    "strategic_layer_control": sum(v for n, v in reach.items() if any(n.startswith(p) for p in ("core.strategic.", "core.route_head", "str_goal_head", "nav_to_str", "gstr_"))),
}

# ---- R4 ------------------------------------------------------------------------------------------------------ #
class MP:
    def setattr(self, obj, name, val): setattr(obj, name, val)
cfg4, m4 = E._r4_model(T, MP())
bt = R.batch(cfg4)
a = E._plan(T, m4, cfg4, bt, lat=6, lon=3); b = E._plan(T, m4, cfg4, bt, lat=7, lon=3)
d = lambda x, y, k: float((x[k] - y[k]).abs().max())
res["R4_at_init_zero_init_ports"] = {"swap_lat_traj": d(a, b, "traj"), "swap_lat_sel_score_v3": d(a, b, "sel_score_v3")}
E._wake(m4); m4.eval()
base = E._plan(T, m4, cfg4, bt, lat=6, lon=3); again = E._plan(T, m4, cfg4, bt, lat=6, lon=3)
ls = E._plan(T, m4, cfg4, bt, lat=7, lon=3); os_ = E._plan(T, m4, cfg4, bt, lat=6, lon=7)
res["R4_ports_opened"] = {
    "control_same_swap_twice_traj": d(base, again, "traj"),
    "swap_lat_sel_score_v3": d(base, ls, "sel_score_v3"), "swap_lat_anchor_traj": d(base, ls, "anchor_traj"), "swap_lat_traj": d(base, ls, "traj"),
    "swap_lon_sel_score_v3": d(base, os_, "sel_score_v3"), "swap_lon_anchor_traj": d(base, os_, "anchor_traj"), "swap_lon_traj": d(base, os_, "traj"),
    "n_windows": int(base["traj"].shape[0]),
    "sel_idx_changed_lat": int((base["sel_idx"] != ls["sel_idx"]).sum()), "sel_idx_changed_lon": int((base["sel_idx"] != os_["sel_idx"]).sum()),
}
# zero-init ports, emission OFF (steps < --r8-alloc-emit-start): the plan-side loss's gradient on every port
cfg4b, m4b = E._r4_model(T, MP(), alloc_emit=False)
bt4b = R.batch(cfg4b)
o_a = E._plan(T, m4b, cfg4b, bt4b, lat=6, lon=3); o_b = E._plan(T, m4b, cfg4b, bt4b, lat=7, lon=3); o_c = E._plan(T, m4b, cfg4b, bt4b, lat=6, lon=7)
res["R4_before_emission_zero_init"] = {"swap_lat_traj": d(o_a, o_b, "traj"), "swap_lon_traj": d(o_a, o_c, "traj")}
outb = E._plan(T, m4b, cfg4b, bt4b, lat=6, lon=3, train=True)
lossb = torch.nn.functional.cross_entropy(outb["sel_score_v3"], torch.zeros(outb["sel_score_v3"].shape[0], dtype=torch.long)) + outb["traj"].abs().mean()
lossb.backward()
dd = m4b.core.decoder
res["R4_before_emission_port_grad_abs_sum_at_zero_init"] = {n: float(p.grad.abs().sum()) for n, p in (
    ("tac8_lat_to_anchor.weight", dd.tac8_lat_to_anchor.weight), ("tac8_lon_to_anchor.weight", dd.tac8_lon_to_anchor.weight),
    ("r8_sel.beta_lat", dd.r8_sel.beta_lat), ("r8_sel.beta_lon", dd.r8_sel.beta_lon), ("r8_mod.proj[-1].weight", dd.r8_mod.proj[-1].weight))}
# emission ON, ports still zero-init (steps >= 1000): per-axis effect of a swap
cfg4c, m4c = E._r4_model(T, MP(), alloc_emit=True); m4c.eval(); bt4c = R.batch(cfg4c)
bs = E._plan(T, m4c, cfg4c, bt4c, lat=6, lon=3)
res["R4_after_emission_ports_zero_init"] = {}
for nm, kw in (("lat TURN_L->TURN_R", dict(lat=7, lon=3)), ("lon BRAKE_TO->ACCELERATE", dict(lat=6, lon=7)), ("both", dict(lat=7, lon=7))):
    o = E._plan(T, m4c, cfg4c, bt4c, **kw)
    res["R4_after_emission_ports_zero_init"][nm] = {"fan_anchor_traj_max_abs_m": d(bs, o, "anchor_traj"), "emitted_traj_max_abs_m": d(bs, o, "traj"),
                                                    "sel_idx_changed_windows": int((bs["sel_idx"] != o["sel_idx"]).sum()), "n_windows": int(bs["sel_idx"].shape[0])}

# ---- R6 ------------------------------------------------------------------------------------------------------ #
cfg6, m6 = R.build(T, False); bt6 = R.batch(cfg6)
mods = E._strategic_modules(m6)
res["R6_modules_perturbed"] = [p for p, _ in mods]
def pert(bypass):
    cfg6.core.no_strategic = bypass
    base = R.forward(cfg6, m6, bt6)
    saved = [(p, p.detach().clone()) for _n, mod in mods for p in mod.parameters()]
    gen = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for _n, mod in mods:
            for p in mod.parameters(): p.add_(torch.randn(p.shape, generator=gen) * 0.5)
    pe = R.forward(cfg6, m6, bt6)
    with torch.no_grad():
        for p, v in saved: p.copy_(v)
    return {k: float((base[k] - pe[k]).abs().max()) for k in ("traj", "anchor_traj", "sel_score_v3", "g_tac", "z_tac", "ctx") if k in base}
res["R6_perturb_all_strategic_params__no_strategic_ON"] = pert(True)
res["R6_perturb_all_strategic_params__no_strategic_OFF_control"] = pert(False)

# ---- R2 ------------------------------------------------------------------------------------------------------ #
cfg2, m2 = R.build(T, False); bt2 = R.batch(cfg2)
sw = dict(bt2, nav_cmd=torch.tensor([2, 0, 1]))
a2, s2, b2 = R.forward(cfg2, m2, bt2), R.forward(cfg2, m2, bt2), R.forward(cfg2, m2, sw)
res["R2_nav_swap_at_init"] = {"control_same_nav": {k: float((a2[k] - s2[k]).abs().max()) for k in ("anchor_logits", "sel_score_v3", "z_tac", "traj")},
                              "swapped": {k: float((a2[k] - b2[k]).abs().max()) for k in ("anchor_logits", "sel_score_v3", "z_tac", "traj")}}

# ---- R1 (needs the patched tree) ------------------------------------------------------------------------------- #
try:
    import test_refcv8_plan_vmax_clamp as C
    T1, cfg1, m1 = C.rig.__wrapped__() if hasattr(C.rig, "__wrapped__") else (None, None, None)
except Exception as e:  # pragma: no cover
    T1 = None
if T1 is not None:
    rg = (T1, cfg1, m1)
    lim = torch.tensor([k / 3.6 for k in C.MIXED])
    off, vo = C._run(rg, C.MIXED, clamp=False); on, vn = C._run(rg, C.MIXED, clamp=True)
    res["R1_AB_emitted_plan_max_speed_ms"] = {"limit": lim.tolist(), "v0": C.V0_GRID, "filter_only": vo.tolist(), "filter_plus_clamp": vn.tolist(),
                                              "exempt_rows(filter_only>limit)": (vo > lim + 1e-5).tolist(),
                                              "e9_ceil_dead_frac_filter_only": float(off.get("e9_ceil_dead_frac", float("nan"))) if "e9_ceil_dead_frac" in off else None}
outp = ROOT.parent / "out/Project Steering/audits/2026-10-18-refcv8-v7f-fixes/raw/probe_numbers.json"
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
