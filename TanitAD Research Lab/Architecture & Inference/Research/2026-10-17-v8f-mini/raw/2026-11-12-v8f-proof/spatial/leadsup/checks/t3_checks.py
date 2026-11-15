"""LEADSUP checks on the FULL tiny V8FMini (d 32, grid state 640 = 40 x 16, hmax 16) with the synthetic world and a synthetic lead track. CPU.
  l1 labels   the vectorised label rule equals a literal re-implementation of objslot hist_run / ls_slope (and the cases: a jump breaks the run, a clip boundary breaks it, < 3 frames = no closing label, > 5 frames = the last 5);
              planted exact series give the exact slope.
  l2 batch    the dataset carries lead_pres / lead_gapm / lead_clos [B, 16] only with lead_sup=True; the default batch has the same keys as the base tree's.
  l3 reach    lead_sup term ALONE: head (frame, mlp) non-zero; state encoder wm.proj and adapter non-zero (detach off); level-1 predictor, levels 2 and 3, O/T/S heads EXACTLY 0.0 (the term cannot reach them);
              control: the total reaches the predictor.
  l4 detach   detach on: the term's gradient into every parameter outside the head is EXACTLY 0.0 (the head itself non-zero); detach off the same parameters are non-zero (so the zero is the wiring, not the data).
  l5 zero     all labels masked (NaN / frame_ok False): term, parts and every gradient EXACTLY 0.0.
  l6 align    a label at a non-end frame (0 .. hist - 2) leaves the term EXACTLY unchanged; at an end frame it moves it; the head output at end frame t is EXACTLY independent of frames > t and of frame t - hist, and depends on t - hist + 1.
  l7 add-line  total(W) - total(0) = W x term (1e-5 relative, float32 rounding), on the full model; MUTATION: the same check on a copy of v8f_mini.py with the total line deleted goes red; with the detach wiring removed l4 goes red; with the T.update line deleted the loss raises.
  l8 init     with lead_sup on the model's state dict equals G's plus the head keys only (same seed): every tensor of G is bit-identical; param count of the head.
Usage: t3_checks.py <tree_root>   (PYTHONPATH = tree stack;taniteval;stack/scripts)"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

tree = sys.argv[1]
torch.set_num_threads(2)
import tanitad  # noqa: E402

assert os.path.abspath(tanitad.__file__).lower().startswith(os.path.abspath(tree).lower()), tanitad.__file__
sys.path.insert(0, os.path.join(tree, "stack", "scripts"))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

CK = os.path.dirname(os.path.abspath(__file__))
TMP = os.environ.get("LS_TMP", "C:/Users/Admin/v8proof_c/leadsup/tmp/t3")
os.makedirs(TMP, exist_ok=True)
res = {"tanitad_file": tanitad.__file__}
TINY = dict(d=32, ff=64, heads=4, n_q=2, depth_s=1, depth_t=1, depth_o=1, nb=6)
HIST = 5


# ------------------------------------------------------------------ l1 labels
def ref_labels(off, n, valid, gap):
    """literal per-row re-implementation of the objslot rule: hist_run (trailing run of finite labels, step <= 3.0 m, at most 5 frames, never across a clip) + ls_slope (>= 3 frames)."""
    N = int(sum(n))
    clos = np.full(N, np.nan)
    ok = (valid == 1) & np.isfinite(gap) & (gap >= 0) & (gap <= 60)
    for c in range(len(off)):
        for r in range(off[c], off[c] + n[c]):
            if not ok[r]:
                continue
            idx = [r]
            j = r - 1
            while j >= off[c] and len(idx) < 5 and ok[j] and abs(gap[j + 1] - gap[j]) <= 3.0:
                idx.append(j)
                j -= 1
            idx = idx[::-1]
            if len(idx) < 3:
                continue
            t = np.arange(len(idx)) * 0.2
            G = gap[idx]
            clos[r] = ((t - t.mean()) * (G - G.mean())).sum() / ((t - t.mean()) ** 2).sum()
    return np.where(ok, gap, np.nan), clos


rng = np.random.default_rng(5)
n = np.array([60, 45, 7, 80])
off = np.concatenate([[0], np.cumsum(n)[:-1]])
N = int(n.sum())
valid = (rng.random(N) < 0.8).astype(np.int8)
gap = 20 + np.cumsum(rng.normal(0, 1.5, N))
gap[rng.random(N) < 0.05] += 8.0                                   # jumps above 3 m
gap[rng.random(N) < 0.03] = np.nan
gap[rng.random(N) < 0.02] = 75.0                                   # beyond 60 m
gap[rng.random(N) < 0.02] = -1.0                                   # negative
J0 = {"off": off, "n": n, "lead_valid": valid, "lead_gap": gap.astype(np.float32)}
L = D.lead_sup_label_arrays(J0)
rg, rc = ref_labels(off, n, valid, J0["lead_gap"].astype(np.float64))
fin = np.isfinite(rc)
l1 = {"n_rows": N, "n_gap": int(np.isfinite(L["gap"]).sum()), "n_clos": int(np.isfinite(L["clos"]).sum()),
      "gap_equal": bool(np.array_equal(np.isfinite(L["gap"]), np.isfinite(rg)) and np.allclose(L["gap"][np.isfinite(rg)], rg[np.isfinite(rg)])),
      "clos_support_equal": bool(np.array_equal(np.isfinite(L["clos"]), fin)),
      "clos_max_abs_diff": float(np.abs(L["clos"][fin].astype(np.float64) - rc[fin]).max()),
      "pres_equal": bool(np.array_equal(L["pres"], valid.astype(np.float32)))}
# planted exact series: a clip of 12 rows, gap = 30 - 2.0 t (t in s), slope -2.0 exactly; run capped at 5 frames; a clip boundary and a jump
off2, n2 = np.array([0, 12, 18]), np.array([12, 6, 10])
g2 = np.concatenate([30.0 - 2.0 * 0.2 * np.arange(12), 10.0 + 1.0 * 0.2 * np.arange(6), 40.0 - 0.5 * 0.2 * np.arange(5), 50.0 - 0.5 * 0.2 * np.arange(5)])
g2[23] = g2[22] - 2.9          # 2.9 m step (within bound)
g2[24] = g2[23] - 3.1          # 3.1 m step: breaks the run
L2 = D.lead_sup_label_arrays({"off": off2, "n": n2, "lead_valid": np.ones(28, np.int8), "lead_gap": g2.astype(np.float32)})
c = L2["clos"]
l1["planted"] = {"clip0_rows2_to_11_slope": [float(x) for x in c[[0, 1, 2, 3, 4, 11]]],      # rows 0,1 have < 3 frames -> NaN; rows 2.. -> -2.0
                 "clip1_first_rows": [float(x) for x in c[12:16]],                           # clip start breaks the run: 12, 13 NaN; 14 -> +1.0 (3 frames)
                 "clip2_around_the_jump": [float(x) for x in c[18:28]]}
pl = l1["planted"]
l1["planted_ok"] = bool(np.isnan(pl["clip0_rows2_to_11_slope"][:2]).all() and np.allclose(pl["clip0_rows2_to_11_slope"][2:], -2.0, atol=1e-5)
                        and np.isnan(pl["clip1_first_rows"][:2]).all() and np.allclose(pl["clip1_first_rows"][2:], 1.0, atol=1e-5))
# clip 2 rows: gap = 40 - 0.1 t_row for rows 18..22 (steps of 0.1 m per 0.2 s = -0.5 m/s), then g2[23] is 2.9 below g2[22] (within the bound), g2[24] 3.1 below g2[23] (breaks): exact expectations below
exp = {18: np.nan, 19: np.nan, 20: -0.5, 21: -0.5, 22: -0.5}
l1["clip2_expected_first"] = all((np.isnan(c[r]) if np.isnan(v) else abs(c[r] - v) < 1e-5) for r, v in exp.items())
l1["clip2_row24_is_nan_or_short_after_break"] = bool(np.isnan(c[24]))          # step 3.1 m: run = 1 frame -> no closing label
l1["PASS"] = bool(l1["gap_equal"] and l1["clos_support_equal"] and l1["clos_max_abs_diff"] < 1e-4 and l1["pres_equal"] and l1["planted_ok"] and l1["clip2_expected_first"]
                  and l1["clip2_row24_is_nan_or_short_after_break"] and l1["n_clos"] > 50)
res["l1"] = l1
print("l1", json.dumps(l1))

# ------------------------------------------------------------------ world + batch
root = Path(TMP) / "world"
w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
J = D.load_join(root / "join_train.npz")
store = D.ClipStore(root, "train", lru=8)
wins = D.train_windows(store, J, w["names"]["train"])
ds0 = D.V8FMiniWindows(store, J, wins, hmax=16)
J_default_keys = sorted(D.collate([ds0.item(*wins[0])]))
D.synthetic_lead_labels(J, 0)
ds = D.V8FMiniWindows(store, J, wins, hmax=16, lead_sup=True)
idx = np.random.default_rng(0).choice(len(wins), 8, replace=False)
batch = D.collate([ds.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
default_batch = D.collate([ds0.item(wins[i][0], wins[i][1], 1, 3) for i in idx])
l2 = {"default_batch_has_no_lead_keys": not any(k.startswith("lead_") and k != "lead_gap_h" for k in default_batch),
      "lead_batch_extra_keys": sorted(set(batch) - set(default_batch)), "shapes": {k: list(batch[k].shape) for k in ("lead_pres", "lead_gapm", "lead_clos")},
      "n_gap_finite": int(torch.isfinite(batch["lead_gapm"]).sum()), "n_clos_finite": int(torch.isfinite(batch["lead_clos"]).sum()), "n_pres": int(batch["lead_pres"].sum()),
      "labels_equal_join_rows": True}
for b_i in range(8):                                                   # the labels of window b are the rows off + idc of its clip
    ci, t = wins[int(idx[b_i])][:2]
    idc = np.clip(np.arange(t - 15, t + 1), 0, int(J["n"][ci]) - 1)
    rr = int(J["off"][ci]) + idc
    l2["labels_equal_join_rows"] &= bool(np.array_equal(batch["lead_pres"][b_i].numpy(), ds.lead_sup_labels["pres"][rr]) and
                                         np.array_equal(np.isnan(batch["lead_gapm"][b_i].numpy()), np.isnan(ds.lead_sup_labels["gap"][rr])))
l2["PASS"] = bool(l2["default_batch_has_no_lead_keys"] and l2["lead_batch_extra_keys"] == ["lead_clos", "lead_gapm", "lead_pres"] and l2["labels_equal_join_rows"]
                  and l2["n_gap_finite"] > 0 and l2["n_clos_finite"] > 0 and l2["shapes"]["lead_pres"] == [8, 16])
res["l2"] = l2
print("l2", json.dumps(l2))


def build(M_=M, seed=0, **kw):
    cfg = M_.V8FMiniConfig(wm="b", hmax=16, wm_state="grid", wm_cell_dim=16, kin_mode="yaw", consistency=True, **TINY, **kw)
    mu, sd = M_.cons_stats(J)
    torch.manual_seed(seed)
    return cfg, M_.V8FMini(cfg, M_.P10Prior.synthetic(0), mu, sd)


def loss(model, b=None, **over):
    b = dict(batch if b is None else b)
    b.update(over)
    return model.loss(b, p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]


gs = lambda ps: float(sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None))      # noqa: E731


def grad_by_group(model):
    out = {"head": gs(model.wm.lead_head.parameters()), "proj": gs(model.wm.proj.parameters()), "adapter": gs(model.adapter.parameters()),
           "pred_l1": gs(model.wm.stack.predictors[0].parameters()), "idm_l1": gs(model.wm.stack.idm_heads[0].parameters()), "act_embed_l1": gs(model.wm.stack.act_embed[0].parameters()),
           "levels_2_3": gs(p for m in (model.wm.stack.predictors[1], model.wm.stack.predictors[2], model.wm.stack.encoders[0], model.wm.stack.encoders[1]) for p in m.parameters()),
           "heads_TS_O": gs(p for m in (model.read_s, model.head_s, model.read_t, model.trunk_t, model.h_goal, model.read_o, model.head_o) for p in m.parameters())}
    return out


def outside_head(model):
    names = [n for n, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0 and not n.startswith("wm.lead_head.")]
    return names


# ------------------------------------------------------------------ l3 reach + l4 detach + l5 zero
cfg, mod = build(lead_sup="on", w_lead_sup=1.0)
mod.train()
T = loss(mod)
T["lead_sup"].backward()                                              # the term ALONE
g_on = grad_by_group(mod)
nz_on = outside_head(mod)
mod.zero_grad()
T = loss(mod)
T["total"].backward()
g_tot = grad_by_group(mod)
l3 = {"term": float(T["lead_sup"]), "parts": {k: float(T[k]) for k in ("lead_sup_bce", "lead_sup_gap", "lead_sup_clos")},
      "counts": {k: float(T[k]) for k in ("lead_sup_n_pres", "lead_sup_n_gap", "lead_sup_n_clos")}, "term_alone_grad": g_on, "total_grad": g_tot,
      "term_alone_nonzero_outside_head": sorted(nz_on)}
l3["PASS"] = bool(g_on["head"] > 0 and g_on["proj"] > 0 and g_on["adapter"] > 0 and g_on["pred_l1"] == 0.0 and g_on["idm_l1"] == 0.0 and g_on["act_embed_l1"] == 0.0 and g_on["levels_2_3"] == 0.0
                  and g_on["heads_TS_O"] == 0.0 and g_tot["pred_l1"] > 0 and g_tot["levels_2_3"] > 0 and g_tot["head"] > 0
                  and set(nz_on) <= {n for n, _ in mod.named_parameters() if n.startswith(("wm.proj.", "adapter.", "pos", "time"))}
                  and l3["term"] > 0 and all(np.isfinite(v) for v in l3["parts"].values()))
res["l3"] = l3
print("l3", json.dumps(l3))

cfg_d, mod_d = build(lead_sup="on", w_lead_sup=1.0, lead_sup_detach="on")
mod_d.train()
T_d = loss(mod_d)
T_d["lead_sup"].backward()
l4 = {"detach_on_head_grad": gs(mod_d.wm.lead_head.parameters()), "detach_on_nonzero_params_outside_head": outside_head(mod_d), "detach_on_term": float(T_d["lead_sup"]),
      "detach_off_nonzero_params_outside_head": sorted(nz_on), "term_equal_detach_on_off_at_same_init": float(T_d["lead_sup"]) == float(l3["term"])}
mod_d.zero_grad()
T_d = loss(mod_d)
T_d["total"].backward()
l4["detach_on_total_reaches_proj_via_other_terms"] = gs(mod_d.wm.proj.parameters()) > 0
l4["built_levers"] = {k: mod_d.built_levers()[k] for k in ("lead_sup", "lead_hist", "lead_sup_detach", "w_lead_sup")}
l4["PASS"] = bool(l4["detach_on_head_grad"] > 0 and l4["detach_on_nonzero_params_outside_head"] == [] and len(nz_on) > 0 and l4["term_equal_detach_on_off_at_same_init"]
                  and l4["detach_on_total_reaches_proj_via_other_terms"] and l4["built_levers"]["lead_sup_detach"] == "on")
res["l4"] = l4
print("l4", json.dumps(l4))

cfg_z, mod_z = build(lead_sup="on", w_lead_sup=1.0)
nan = torch.full_like(batch["lead_gapm"], float("nan"))
T_z = loss(mod_z, lead_gapm=nan, lead_clos=nan, lead_pres=torch.zeros_like(batch["lead_pres"]), frame_ok=torch.zeros_like(batch["frame_ok"]))
T_z["lead_sup"].backward()
l5 = {"term": float(T_z["lead_sup"]), "parts": [float(T_z[k]) for k in ("lead_sup_bce", "lead_sup_gap", "lead_sup_clos")], "head_grad": gs(mod_z.wm.lead_head.parameters()),
      "n_counted": float(T_z["lead_sup_n_pres"])}
cfg_z2, mod_z2 = build(lead_sup="on", w_lead_sup=1.0)                  # gap and closing masked, presence kept: only the presence part moves, the other two read exactly 0.0
T_z2 = loss(mod_z2, lead_gapm=nan, lead_clos=nan)
l5["gap_and_closing_masked_parts"] = [float(T_z2[k]) for k in ("lead_sup_bce", "lead_sup_gap", "lead_sup_clos")]
l5["PASS"] = bool(l5["term"] == 0.0 and l5["parts"] == [0.0, 0.0, 0.0] and l5["head_grad"] == 0.0 and l5["n_counted"] == 0.0
                  and l5["gap_and_closing_masked_parts"][0] > 0 and l5["gap_and_closing_masked_parts"][1:] == [0.0, 0.0])
res["l5"] = l5
print("l5", json.dumps(l5))

# ------------------------------------------------------------------ l6 alignment
cfg_a, mod_a = build(lead_sup="on", w_lead_sup=1.0)
mod_a.eval()
ref = float(loss(mod_a)["lead_sup"])
l6 = {"ref_term": ref, "label_at_frame": {}}
for f in range(16):
    bl = batch["lead_gapm"].clone()
    bl[:, f] = bl[:, f] + 7.0                                         # shift every window's gap label at frame f (NaN stays NaN: use the presence label too)
    bp = batch["lead_pres"].clone()
    bp[:, f] = 1.0 - bp[:, f]
    l6["label_at_frame"][f] = float(loss(mod_a, lead_gapm=bl, lead_pres=bp)["lead_sup"]) - ref
non_end = [l6["label_at_frame"][f] for f in range(HIST - 1)]
end = [l6["label_at_frame"][f] for f in range(HIST - 1, 16)]
# frames 0 .. 3 are not end frames: EXACTLY 0.0; every end frame 4 .. 15 moves the term unless no window of the batch counts it (frame_ok)
l6["non_end_frames_exact_zero"] = all(x == 0.0 for x in non_end)
l6["end_frames_moved"] = [f for f in range(HIST - 1, 16) if l6["label_at_frame"][f] != 0.0]
adapt = torch.randn(2, 16, cfg_a.n_tok, cfg_a.d, generator=torch.Generator().manual_seed(3))
act = torch.randn(2, 16, 2, generator=torch.Generator().manual_seed(4))
head = mod_a.wm.lead_head


def head_out(a_):
    with torch.no_grad():
        return head(mod_a.wm.z1(a_))


base_out = head_out(adapt)
dep = {}
for f in range(16):
    a2 = adapt.clone()
    a2[:, f] += 1.0
    d = (head_out(a2) - base_out).abs().amax((0, 2))               # per end frame
    dep[f] = [int(i) + HIST - 1 for i in torch.nonzero(d > 0).flatten()]
l6["frame_moves_end_frames"] = dep
l6["causal_window_ok"] = all(dep[f] == [e for e in range(HIST - 1, 16) if e - HIST + 1 <= f <= e] for f in range(16))
l6["PASS"] = bool(l6["non_end_frames_exact_zero"] and len(l6["end_frames_moved"]) >= 8 and l6["causal_window_ok"])
res["l6"] = l6
print("l6", {k: v for k, v in l6.items() if k != "label_at_frame"}, "label_at_frame", {k: round(v, 6) for k, v in l6["label_at_frame"].items()})

# ------------------------------------------------------------------ l7 float64 + mutations
def f64_check(M_, tag, detach="off"):
    cfg7, m7 = build(M_, lead_sup="on", w_lead_sup=1.0, lead_sup_detach=detach)
    b = dict(batch)                                                   # float32: the encoder casts the cached tokens to float32, so a float64 model cannot run end to end; the add-line check is exact to float32 rounding
    m7.eval()
    import copy
    m0 = copy.deepcopy(m7)                                            # a twin made before any call: the stack's SIGReg term draws random projections from the GLOBAL rng, so two calls differ by 1.7e-4 (measured) unless the global seed is reset before each call; the twin carries weight 0
    object.__setattr__(m0.cfg, "w_lead_sup", 0.0)
    w_save = m7.cfg.w_lead_sup
    with torch.no_grad():
        torch.manual_seed(77)
        t1 = m7.loss(dict(b), p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]
        tot_w, term = float(t1["total"]), float(t1["lead_sup"])
        torch.manual_seed(77)
        tot_0 = float(m0.loss(dict(b), p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]["total"])
    return {"tag": tag, "term": term, "total_W": tot_w, "total_0": tot_0, "total_W_minus_total_0": tot_w - tot_0, "abs_diff_to_W_x_term": abs((tot_w - tot_0) - w_save * term),
            "float32_rounding_bar": 8 * 1.1920929e-07 * max(abs(tot_w), abs(tot_0))}


l7 = {"real": f64_check(M, "real")}
src = open(M.__file__, encoding="utf-8").read()
line = [ln for ln in src.splitlines() if ln.lstrip().startswith("total = total + cfg.w_lead_sup * T[")]
assert len(line) == 1, line
muts = {"mut_total_line_deleted": src.replace(line[0], "        pass"),
        "mut_detach_removed": src.replace("res[\"lead_out\"] = self.lead_head(z1.detach() if self.lead_detach else z1)", "res[\"lead_out\"] = self.lead_head(z1)"),
        "mut_term_update_deleted": src.replace('T.update(lead_sup_terms(wm["lead_out"], batch, cfg.lead_hist))', "pass")}
for k_, v_ in muts.items():
    assert v_ != src, k_


def load_mut(tag, text):
    p = Path(TMP) / f"{tag}.py"
    p.write_text(text, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location(f"v8f_mini_{tag}", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


Mm = load_mut("mut_total", muts["mut_total_line_deleted"])
l7["mut_total_line_deleted"] = f64_check(Mm, "mut_total_line_deleted")
l7["real_ok"] = l7["real"]["abs_diff_to_W_x_term"] < l7["real"]["float32_rounding_bar"] and l7["real"]["term"] > 0
l7["mut_total_red"] = l7["mut_total_line_deleted"]["abs_diff_to_W_x_term"] > 100 * l7["real"]["float32_rounding_bar"] / 8
Mm = load_mut("mut_detach", muts["mut_detach_removed"])
cfgm, mm = build(Mm, lead_sup="on", w_lead_sup=1.0, lead_sup_detach="on")
mm.train()
Tm = mm.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))["terms"]
Tm["lead_sup"].backward()
l7["mut_detach_removed_nonzero_outside_head"] = len([n for n, p in mm.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0 and not n.startswith("wm.lead_head.")])
l7["mut_detach_red"] = l7["mut_detach_removed_nonzero_outside_head"] > 0
Mm = load_mut("mut_update", muts["mut_term_update_deleted"])
cfgu, mu_ = build(Mm, lead_sup="on", w_lead_sup=1.0)
try:
    mu_.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    l7["mut_term_update_deleted_raised"] = None
except KeyError as e:
    l7["mut_term_update_deleted_raised"] = f"KeyError {e}"
l7["mut_update_red"] = l7["mut_term_update_deleted_raised"] is not None
l7["PASS"] = bool(l7["real_ok"] and l7["mut_total_red"] and l7["mut_detach_red"] and l7["mut_update_red"])
res["l7"] = l7
print("l7", json.dumps(l7))

# ------------------------------------------------------------------ l8 init identity with G
cfgG, mG = build()
cfgL, mL = build(lead_sup="on", w_lead_sup=1.0)
sG, sL = mG.state_dict(), mL.state_dict()
extra = [k for k in sL if k not in sG]
l8 = {"extra_keys": extra, "keys_of_G_all_present": all(k in sL for k in sG), "G_tensors_bit_identical": all(torch.equal(sG[k], sL[k]) for k in sG),
      "head_params": int(sum(p.numel() for p in mL.wm.lead_head.parameters())), "G_params": int(sum(p.numel() for p in mG.parameters())), "L_params": int(sum(p.numel() for p in mL.parameters())),
      "head_params_expected": 640 * 64 + 64 + 320 * 128 + 128 + 128 * 3 + 3}
l8["PASS"] = bool(l8["keys_of_G_all_present"] and l8["G_tensors_bit_identical"] and all(k.startswith("wm.lead_head.") for k in extra) and l8["head_params"] == l8["head_params_expected"]
                  and l8["L_params"] - l8["G_params"] == l8["head_params"])
res["l8"] = l8
print("l8", json.dumps(l8))

res["PASS"] = all(res[k]["PASS"] for k in ("l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8"))
json.dump(res, open(os.path.join(CK, "t3_checks.json"), "w", encoding="utf-8", newline="\n"), indent=1, default=str)
print("T3", "PASS" if res["PASS"] else "FAIL")
