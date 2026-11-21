#!/usr/bin/env python3
"""Tests for VDAC A5's arm HB (the LoRA trunk copy that feeds only the auxiliary drivable head) and HC's reachability fixes.  CPU only (CUDA_VISIBLE_DEVICES=-1), no pod, no checkpoint, no data, no scipy.
Run:  CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python refe/test_vdac_hb.py   ->   ZZTEST_PASS <n>   (prints the mutation table)

TWO LAYERS (CLAUDE.md "built, tested, and unreachable from its caller"):
  * MODULE layer on the REAL tiny REFe (model.py, the real LoRALinear, the real 2,440,450-parameter head): the copy's ctx reproduces model.py's own visual_ctx bit for bit; the head's loss reaches EVERY copy LoRA tensor
    (non-zero), reaches NO P6 tensor, and at weight 0.0 the copy reads exactly 0.0 (a tensor, not None); no scorer path; storage independence; aux_drv/2 round trip + 2 resumed updates bit-identical; judge_g21 field by field.
  * CALLER layer: the REAL scorer_finetune.main() (test_aux_s1_wiring's harness: a stub trunk with a real LoRALinear backbone, a synthetic label store) -- the flag reaches the copy, the copy's LoRA moves, P6's tensors do not,
    aux_drv.pt is aux_drv/2, G12's tap row reads the copy's ctx, G21 is in the preflight and judged, and each wiring line is DELETED from the source / each runtime switch set and the check that owns it must go RED.
"""
from __future__ import annotations

import copy
import os
import shlex
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TMPROOT = os.environ.get("TMP") or tempfile.gettempdir()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import aux_drivable_head as AUX  # noqa: E402
import aux_hb as HB  # noqa: E402
import gate_checks as C  # noqa: E402
import launch_gate as LG  # noqa: E402
import test_aux_s1_wiring as W  # noqa: E402
import vdac_postgate as PG  # noqa: E402
import vdac_probes as VP  # noqa: E402
from model import LoRALinear, REFe, REFeConfig  # noqa: E402

SCORER = ("score_q_mlp", "score_dec", "score_head")
TINY = dict(backbone="vits16", width=32, depth=1, heads=2, img_h=32, img_w=32, n_cameras=1, cameras=("CAM_F0",), dec_depth=1, n_proposals=4, horizon_steps=4, reg_heads=2, n_registers=2, lora_rank=2, nav_mode="command")
N = [0]
TABLE: list = []


def ok(cond, msg):
    N[0] += 1
    assert cond, msg


def red(reasons, needle, what):
    """a mutation arm: the judge must be NON-EMPTY and name `needle`"""
    N[0] += 1
    assert reasons and any(needle in r for r in reasons), f"MUTATION NOT CAUGHT ({what}): {reasons}"
    TABLE.append((what, "RED", needle))


def env(**kw):
    class _E:
        def __enter__(s):
            s.old = {k: os.environ.get(k) for k in kw}
            os.environ.update(kw)

        def __exit__(s, *a):
            for k, v in s.old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return _E()


# ================================================================================================ the module layer: real tiny REFe
def make_model(seed=7):
    torch.manual_seed(seed)
    model = REFe(REFeConfig(**TINY)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for nm in SCORER:                                                 # the scorer is the trainable part, as in the trainer (so a scorer loss has a graph)
        for p in getattr(model, nm).parameters():
            p.requires_grad_(True)
    g = torch.Generator().manual_seed(3)
    for _n, p in HB.lora_named(model.backbone):
        p.data.copy_(torch.randn(p.shape, generator=g) * 0.05)       # a TRAINED P6 has non-zero A and B (B = 0 would make every A gradient exactly zero)
    return model


def batch_of(B=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    img, ego, goal = torch.rand(B, 1, 3, 32, 32, generator=g), torch.rand(B, 7, generator=g), torch.rand(B, 4, generator=g)
    label = torch.from_numpy(W.synth_label("T0", "a")).unsqueeze(0).repeat(B, 1, 1)
    return img, ego, goal, (label, torch.ones(B, dtype=torch.bool))


class Rig:
    """model + copy + head + the callables g21_fields takes, replicating the trainer's fwd / aux_backward on the real tiny model (the CALLER layer proves the trainer's own copies)"""

    def __init__(self, w_aux=0.1, cell=2.0, bw=0.0):
        self.model = make_model()
        self.outs: list = []
        self.model.scene_proj.register_forward_hook(lambda m, i, o: self.outs.append(o))
        self.cap: dict = {}
        self.model.score_dec[0].register_forward_pre_hook(lambda m, a: self.cap.__setitem__("sctx", a[1]))
        self.hb = HB.HBTrunk(self.model)
        self.aux = AUX.AuxDrivable(AUX.AuxDrivableConfig(w_aux=w_aux, cell=cell, bw=bw))
        self.aux.head.register_forward_pre_hook(lambda m, a: self.cap.__setitem__("aux_ctx", a[0]))
        self.img, self.ego, self.goal, self.lv = batch_of()
        self.cxy = torch.rand(2, 4, 12)
        self.optH = torch.optim.AdamW(list(self.aux.head.parameters()), lr=1e-3, weight_decay=0.01)
        self.optL = HB.build_lora_optimizer(self.hb, 1e-3)

    def fwd(self, batch=None):
        m = self.model
        m(self.img, self.ego, self.goal)
        sctx = self.cap["sctx"]
        actx = HB.select_actx(self.hb, m, self.img, None, sctx, lambda: m.ego_enc(torch.cat([self.ego, m.cmd_embed(self.goal)], -1)).sum(-1))
        logit, sdf = self.aux.predict(actx)
        self.cap["aux"] = (logit, sdf)
        sB = m.score_trajectories(torch.rand(2, 4, 4, 3), sctx)
        return None, None, sB, None

    def aux_backward(self, lv, div=1.0, retain=False):
        logit, sdf = self.cap["aux"]
        l_aux, bce, hub = self.aux.loss(logit, sdf, lv[0], lv[1])
        tot = self.aux.combine(torch.zeros(()), l_aux) / div
        if tot.requires_grad:
            tot.backward(retain_graph=retain)
        return float(bce), float(hub)

    def g21(self):
        def step():
            r = HB.two_update_probe(model=self.model, hb=self.hb, aux=self.aux, scorer_names=SCORER, img=self.img, calib=None, lv=self.lv, dev="cpu",
                                    autocast=lambda: torch.autocast("cpu", enabled=False), lr_head=1e-3, lr_lora=1e-3)
            r.update(HB.step0_equal(self.model, self.hb, self.img, None, self.cap["sctx"]))
            return r
        return HB.g21_fields(model=self.model, hb=self.hb, aux=self.aux, scorer_names=SCORER, fwd=self.fwd, aux_backward=self.aux_backward, batch=None, lv=self.lv, step_probe=step)


def module_layer():
    # ---- T4 step-0 identity vs the REAL model.py forward, and the arithmetic of the LoRA count
    m = make_model()
    outs: list = []
    m.scene_proj.register_forward_hook(lambda mm, i, o: outs.append(o))
    img, ego, goal, _lv = batch_of()
    m(img, ego, goal)
    hb = HB.HBTrunk(m)
    ok(len(outs) == 2, "REFe.forward calls scene_proj twice (scene_ctx, visual_ctx)")
    ok(torch.equal(hb.ctx(m, img, None), outs[1]), "T4: step-0 copy ctx == model.py's own visual_ctx, bit for bit")
    ok(torch.equal(hb.ctx(m, img[:, 0], None), outs[1]), "a 4-dim image ([B,3,H,W]) takes the same unsqueeze path as REFe.forward")
    f = hb.built_facts(m, SCORER)
    ok(f["n_lora_trainable"] == f["lora_expected"] == 1 * 2 * 2 * (32 + 32) and f["trainable_all_lora"] and f["copy_storage_disjoint"] and f["copy_not_in_model_state_dict"] and f["p6_params_requires_grad"] == 0, f"built facts {f}")
    ok(HB.VITL16_LORA_PARAMS == 24 * 2 * 32 * (1024 + 1024) == VP.VITL16_LORA_PARAMS == LG.VDAC_HB_LORA_PARAMS, "the vitl16 LoRA count is arithmetic: 24 x 2 x 32 x 2,048 = 3,145,728")
    ok(HB.P6_LORA_LR == 2e-4 and LG.VDAC_HB_LORA_LR == "0.0002", "HB's LoRA lr is P6's own (train.py single group at --lr 2e-4)")

    # ---- T1 / T2 / T3 / T6: live + exact-zero control + P6 untouched + no scorer path
    r = Rig()
    ok(not any(k in model_keys(r.model) for k in ("hb", "backbone_copy")) and not any(id(p) in {id(q) for q in r.hb.parameters()} for p in r.model.parameters()), "the copy is a sibling: not a model attribute / parameter")
    g = r.g21()
    ok(g["lora_grad_declared"]["none"] == 0 and g["lora_grad_declared"]["min_abs"] > 0.0 and g["lora_grad_declared"]["n"] == 4, f"T1: every copy LoRA tensor has a non-zero gradient {g['lora_grad_declared']}")
    ok(g["lora_grad_w0"]["none"] == 0 and g["lora_grad_w0"]["max_abs"] == 0.0 and g["lora_grad_w0"]["min_abs"] == 0.0, f"T2: weight 0.0 reads EXACTLY 0.0, not None {g['lora_grad_w0']}")
    ok(g["p6_nonscorer_grad_not_none_declared"] == 0 and g["p6_params_requires_grad"] == 0, "T3: no P6 parameter outside the scorer holds a gradient")
    ok(g["scorer_to_copy_grads_all_none"] is True, "T6: autograd.grad(scorer loss, copy LoRA) is None for every tensor")
    ok(g["p6_state_unchanged_2_updates"] and g["lora_moved_in_2_updates"] and g["step0_ctx_equal"] and g["adam_lora_params_with_state"] == g["adam_lora_params"] == 4, f"T3/T4/T11 {g}")
    ok(VP.judge_g21(g) == [], f"judge_g21 accepts the real tiny HB: {VP.judge_g21(g)}")
    # ---- T5 independence: editing the copy moves nothing in P6
    r2 = Rig()
    before = r2.model(r2.img, r2.ego, r2.goal)[0].clone()
    with torch.no_grad():
        for p in r2.hb.lora_parameters():
            p.add_(1.0)
    ok(torch.equal(before, r2.model(r2.img, r2.ego, r2.goal)[0]), "T5: editing a copy LoRA tensor does not change model(...)")
    with env(**{HB.ENV_SHARE: "1"}):
        r3 = Rig()
        shared = r3.hb.built_facts(r3.model, SCORER)
    ok(shared["copy_storage_disjoint"] is False, "T5 mutation: a shared 'copy' reads storage-NOT-disjoint")
    # ---- the mutation arms of G21 (each must be RED; the baseline is green above)
    for envname, needle, what in ((HB.ENV_P6CTX, "NO gradient", "head reads P6's ctx instead of the copy's"), (HB.ENV_DETACH, "NO gradient", "copy output detached"),
                                  (HB.ENV_SHARE, "copy_storage_disjoint", "the copy shares P6's storage"), (HB.ENV_UNFREEZE_SP, "requires_grad True", "P6's scene_proj unfrozen")):
        with env(**{envname: "1"}):
            rm = Rig()
            reasons = VP.judge_g21(rm.g21())
        red(reasons, needle, f"G21 / {envname}")
    # the scorer path mutation: route the copy's ctx into the scorer
    rm = Rig()
    gm = rm.g21()
    sB_bad = rm.model.score_trajectories(torch.rand(2, 4, 4, 3), rm.hb.ctx(rm.model, rm.img, None))
    gs = torch.autograd.grad(sB_bad.sum(), rm.hb.lora_parameters(), allow_unused=True)
    gm = dict(gm, scorer_to_copy_grads_all_none=bool(all(x is None for x in gs)))
    red(VP.judge_g21(gm), "scorer_to_copy_grads_all_none", "scorer reads the copy's ctx")
    # ---- T9 judge, field by field
    for k, bad, needle in (("trunk_copy_built", False, "trunk_copy_built"), ("copy_not_in_model_state_dict", False, "copy_not_in_model_state_dict"), ("step0_ctx_equal", False, "step0_ctx_equal"),
                           ("p6_state_unchanged_2_updates", False, "p6_state_unchanged"), ("n_lora_trainable", 255, "trainable parameters"), ("p6_params_requires_grad", 1, "requires_grad True"),
                           ("p6_nonscorer_grad_not_none_declared", 1, "hold a gradient"), ("mutate", ["X"], "regression switch"), ("adam_lora_params_with_state", 3, "Adam state")):
        red(VP.judge_g21(dict(g, **{k: bad})), needle, f"judge_g21 field {k}")
    red(VP.judge_g21(dict(g, lora_grad_w0={"none": 0, "n": 4, "min_abs": 0.0, "max_abs": 1e-9, "all_finite": True})), "weight-0 control", "judge_g21 dead control")
    red(VP.judge_g21(dict(g, lora_grad_declared={"none": 1, "n": 4, "min_abs": 1.0, "max_abs": 1.0, "all_finite": True})), "NO gradient", "judge_g21 one tensor without gradient")
    red(VP.judge_g21(g, expected_lora=VP.VITL16_LORA_PARAMS), "trainable parameters", "judge_g21 against the registered vitl16 count")

    # ---- T8 the tap row reads the COPY's ctx (outs[2]); the stale index is RED
    r = Rig()
    r.fwd()
    outs3 = r.outs[-3:]
    tf = HB.tap_fields(outs3, r.cap["aux_ctx"], True, AUX.tap_is)
    ok(tf["scene_proj_calls"] == 3 and tf["tap_is_visual_ctx"] and tf["tap_index"] == 2 and tf["tap_is_p6_visual_ctx"] is False and tf["tap_is_scene_ctx"] is False, f"T8: {tf}")
    with env(**{HB.ENV_STALE_TAP: "1"}):
        tf2 = HB.tap_fields(outs3, r.cap["aux_ctx"], True, AUX.tap_is)
    ok(tf2["tap_is_visual_ctx"] is False, "T8 mutation: the stale outs[1] index reads False while HB is on")
    ns_hb = types.SimpleNamespace(w_aux_drv=0.1, aux_drv_feed="off", aux_drv_trunk_lora="on")
    g12 = {"ok": True, "labelled_frames_in_batch": 2, "w_aux": 0.1, "feed": "off", "per_group_declared": {"q": 1.0}, "per_group_w0": {"q": 0.0}, "bce_declared": 0.69, "bce_near_ln2": True,
           "score_to_head_grads_all_none": True, "adam_head_params": 3, "adam_head_params_with_state": 3, "reference_replay_identical": True, "scorer_params_bit_identical_2steps": True,
           "identity_probe_reads_one_ulp": True, "head_retained_gb": None, "mutate": [], **tf}
    ok(C.judge_g12(g12, ns_hb) == [], f"judge_g12 accepts the HB tap: {C.judge_g12(g12, ns_hb)}")
    red(C.judge_g12(dict(g12, **tf2), ns_hb), "visual_ctx", "G12 tap: stale index while HB is on")
    red(C.judge_g12(dict(g12, scene_proj_calls=2, tap_index=1, hb=False), ns_hb), "tap row did not read the copy", "G12 tap: HB declared but HA tap")

    # ---- T7 G19: head disjoint from ego and command on HB; RED under the trunk-side switch
    def run19(e, gl):
        rr = Rig()
        rr.ego, rr.goal = e, gl
        rr.model(rr.img, e, gl)
        actx = HB.select_actx(rr.hb, rr.model, rr.img, None, rr.cap["sctx"], lambda: rr.model.ego_enc(torch.cat([e, rr.model.cmd_embed(gl)], -1)).sum(-1))
        lg, sd = rr.aux.predict(actx)
        tr, _sc = rr.model(rr.img, e, gl)[:2]
        return lg.float().sum() + sd.float().sum(), tr.float().sum()
    g19 = VP.disjoint_probe(run19, batch_of()[1], batch_of()[2])
    ok(VP.judge_g19(g19) == [], f"T7: G19 passes on HB {VP.judge_g19(g19)}")
    with env(**{HB.ENV_EGO_TRUNK: "1"}):
        g19m = VP.disjoint_probe(run19, batch_of()[1], batch_of()[2], mutate=[HB.ENV_EGO_TRUNK])
    ok(g19m["head_grad_ego"] not in (None, 0.0), "the trunk-side mutant gives the head a gradient w.r.t. ego")
    red(VP.judge_g19(g19m), "regression switch", "G19 under REFE_AUX_MUTATE_EGO_TRUNK")
    with env(**{HB.ENV_EGO_TRUNK: "1"}):
        g19n = VP.disjoint_probe(run19, batch_of()[1], batch_of()[2])             # even with the switch NOT declared in `mutate`, the probe reads the leak
    red(VP.judge_g19(g19n), "NOT disjoint", "G19 reads the trunk-side leak by gradient alone")

    # ---- T10 aux_drv/2: round trip, 2 resumed updates bit-identical, strict refusals
    def train(rg, n):
        for _ in range(n):
            rg.fwd()
            rg.aux_backward(rg.lv, 1.0)
            torch.nn.utils.clip_grad_norm_(list(rg.aux.head.parameters()), 1.0)
            rg.optH.step(); rg.optH.zero_grad(set_to_none=True)
            HB.lora_step(rg.hb, rg.optL, 1e-3)
    a1 = Rig()
    train(a1, 3)
    d = tempfile.mkdtemp(prefix="hb_ck_", dir=TMPROOT)
    pth = os.path.join(d, "aux_drv.pt")
    HB.save_v2(pth, aux=a1.aux, hb=a1.hb, optH=a1.optH, optL=a1.optL, update=3, meta={"x": 1}, atomic_save=torch.save)
    a2 = Rig()
    ok(HB.load_v2(pth, aux=a2.aux, hb=a2.hb, optH=a2.optH, optL=a2.optL) == 3, "T10: load_v2 returns the update count")
    same = all(torch.equal(x, y) for x, y in zip(a1.hb.lora_parameters(), a2.hb.lora_parameters())) and all(torch.equal(x, y) for x, y in zip(a1.aux.head.parameters(), a2.aux.head.parameters()))
    ok(same, "T10: head and LoRA round-trip bit for bit")
    train(a1, 2)
    train(a2, 2)
    ok(all(torch.equal(x, y) for x, y in zip(a1.hb.lora_parameters(), a2.hb.lora_parameters())) and all(torch.equal(x, y) for x, y in zip(a1.aux.head.parameters(), a2.aux.head.parameters())),
       "T10: 2 resumed updates are bit-identical to 2 continued updates (both Adam states restored)")
    a3 = Rig()
    train(a3, 3)
    HB.save_v2(pth + ".nolora_opt", aux=a3.aux, hb=a3.hb, optH=a3.optH, optL=a3.optL, update=3, meta={}, atomic_save=torch.save)
    ck = torch.load(pth, weights_only=False)
    v1 = {"format": "aux_drv/1", "head": ck["head"], "feed": None, "opt": ck["opt"], "update": 3, "meta": {}}
    for what, bad, needle in (("a head-only aux_drv/1 file", v1, "head-only"), ("no LoRA optimiser state", {k: v for k, v in ck.items() if k != "opt_lora"}, "opt_lora"),
                              ("a missing LoRA tensor", dict(ck, hb_lora={k: v for k, v in list(ck["hb_lora"].items())[1:]}), "different key set")):
        try:
            HB.load_v2(bad, aux=Rig().aux, hb=Rig().hb, optH=a2.optH, optL=a2.optL)
            ok(False, f"load_v2 accepted {what}")
        except HB.HBCkptRefused as e:
            N[0] += 1
            ok(needle in str(e), str(e))
            TABLE.append((f"strict load: {what}", "RED", needle))
    # ---- T11 the post-training Adam-state check covers the LoRA tensors
    ck3 = torch.load(pth, weights_only=False)
    ck3["update"] = 3
    n_lora = sum(int(v.numel()) for v in ck3["hb_lora"].values())
    ok(PG.judge(ck3, 3, n_params=2440450, n_lora=n_lora) == [], f"T11: judge accepts a trained aux_drv/2: {PG.judge(ck3, 3, 2440450, n_lora)}")
    nostate = copy.deepcopy(ck3)
    pid = nostate["opt_lora"]["param_groups"][0]["params"][1]
    del nostate["opt_lora"]["state"][pid]
    red(PG.judge(nostate, 3, 2440450, n_lora), "NO Adam state", "post-gate: a LoRA tensor that never stepped")
    zero = copy.deepcopy(ck3)
    zero["opt_lora"]["state"][pid]["exp_avg"].zero_()
    red(PG.judge(zero, 3, 2440450, n_lora), "ZERO first moment", "post-gate: a LoRA first moment exactly zero")
    red(PG.judge({k: v for k, v in ck3.items() if k != "opt_lora"}, 3, 2440450, n_lora), "LoRA optimiser", "post-gate: no LoRA optimiser state in aux_drv/2")
    red(PG.judge(ck3, 3, 2440450, n_lora + 1), "LoRA parameters", "post-gate: wrong LoRA count")

    # ---- T12 HC: cell, boundary weight
    h1 = AUX.AuxDrivable(AUX.AuxDrivableConfig(cell=1.0)).built_facts()
    ok((h1["nx"], h1["ny"], h1["n_params"], h1["cell"]) == (80, 64, 3423490, 1.0) and C.AUX_HEAD_PARAMS_BY_CELL[1.0] == 3423490, f"T12: cell 1.0 builds 80 x 64 / 3,423,490 {h1}")
    h2 = AUX.AuxDrivable(AUX.AuxDrivableConfig()).built_facts()
    ok((h2["nx"], h2["ny"], h2["n_params"]) == (40, 32, 2440450 == C.AUX_HEAD_PARAMS_BY_CELL[2.0] and 2440450), "the default head is the S1 head")
    ok(h2["bw_probe"]["inside_over_outside"] == 1.0 and h2["bw"] == 0.0, "bw off: the probe reads multiplier 1.0")
    h3 = AUX.AuxDrivable(AUX.AuxDrivableConfig(cell=1.0, bw=3.0)).built_facts()
    ok(abs(h3["bw_probe"]["inside_over_outside"] - 4.0) < 1e-4, f"bw 3: the built loss applies 4 inside / 1 outside: {h3['bw_probe']}")
    lab = torch.from_numpy(W.synth_label("T0", "a")).unsqueeze(0).repeat(2, 1, 1)
    lg, sd = torch.randn(2, 40, 32), torch.randn(2, 40, 32) * 3
    a_off = AUX.aux_loss(lg, sd, lab, None)
    a_zero = AUX.aux_loss(lg, sd, lab, None, bw=0.0, bw_band_m=3.0)
    ok(all(torch.equal(x, y) for x, y in zip(a_off, a_zero)), "bw 0.0 reproduces the unweighted loss bit for bit")
    a_w = AUX.aux_loss(lg, sd, lab, None, bw=3.0)
    ok(torch.equal(a_w[1], a_off[1]) and float(a_w[2]) > float(a_off[2]), "bw 3 weights the Huber term only (BCE unchanged)")
    src_aux = (HERE / "aux_drivable_head.py").read_text(encoding="utf-8")
    snippet = "        hub = hub * (1.0 + bw * (t.sdf_m.abs() < bw_band_m).to(hub.dtype))\n"
    ok(src_aux.count(snippet) == 1, "the weight line occurs once")
    mod = types.ModuleType("aux_nobw"); mod.__file__ = str(HERE / "aux_drivable_head.py"); sys.modules["aux_nobw"] = mod
    exec(compile(src_aux.replace(snippet, "        pass" + chr(10)), "aux_nobw.py", "exec"), mod.__dict__)
    pr = mod.AuxDrivable(mod.AuxDrivableConfig(cell=1.0, bw=3.0)).built_facts()["bw_probe"]["inside_over_outside"]
    red([f"probe reads {pr}"] if abs(pr - 4.0) > 1e-3 else [], "probe reads 1.0", "HC: bw named, weight line dropped from the loss")


def model_keys(m):
    return set(dict(m.named_children()))


# ================================================================================================ the caller layer: the REAL scorer_finetune.main()
class StubBB(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = LoRALinear(12, 256, 4)
        g = torch.Generator().manual_seed(5)
        self.lin.B.data.copy_(torch.randn(self.lin.B.shape, generator=g) * 0.05)       # a trained P6's B is non-zero

    def forward(self, x):
        return self.lin(x.reshape(x.shape[0], 2, 12)) * 0.05


class StubHBREFe(W.Stub256REFe):
    """Stub256REFe with a REAL-shaped trunk path: backbone (a real LoRALinear), _pos3d, scene_proj called twice in forward, ego_enc / cmd_embed for the trunk-side ego mutant"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.backbone = StubBB()
        self.cmd_embed = nn.Linear(2, 3)
        self.ego_enc = nn.Linear(7, 256)

    def _pos3d(self, img, n_cam=None, calib=None):
        return torch.zeros(1, n_cam * 2, 256)

    def context(self, img):
        if img.dim() == 4:
            img = img.unsqueeze(1)
        B, Nc = img.shape[0], img.shape[1]
        tok = self.backbone(img.reshape(B * Nc, *img.shape[2:]))
        visual = tok.reshape(B, Nc * tok.shape[1], tok.shape[2]) + self._pos3d(img, Nc, None)
        return self.scene_proj(visual[:, :2]), self.scene_proj(visual)


class StubHBTargets(W.StubTargets256):
    def sample_inputs(self, i):
        g = torch.Generator().manual_seed(1000 + i)
        return torch.randn(1, 3, 2, 4, generator=g) * (1.0 if i < 100 else 20.0), torch.randn(8, 20, 3, generator=g) * 2.0


HB_ARGS = W.H1_ARGS + ["--aux-drv-trunk-lora", "on", "--aux-drv-lora-lr", "1e-3"]
SRC = (HERE / "scorer_finetune.py").read_text(encoding="utf-8")
CACHE: dict = {}


def smoke(tag, extra=HB_ARGS, src=SRC, env_=None, **kw):
    key = (tag,)
    if key not in CACHE:
        CACHE[key] = W.smoke_aux(src, tempfile.mkdtemp(prefix="hb_" + tag, dir=TMPROOT), tag=tag, extra=extra, env=env_, refe_cls=StubHBREFe, targets_cls=StubHBTargets, **kw)
    return CACHE[key]


def mutated(old, new):
    assert SRC.count(old) == 1, f"the snippet must occur exactly once: {old[:60]}"
    return SRC.replace(old, new)


def caller_layer():
    # ---- the HB run: preflight PASS, a real run
    r = smoke("hb")
    ok(r["rc_pre"] == 0 and r["pass_json"] is not None, f"HB preflight PASSES (rc {r['rc_pre']}): {[ (k, v.get('ok')) for k, v in (W.gate(r) and {} or {}).items()]}")
    gts = [x for x in r["rows"]["preflight.jsonl"] if x.get("event") == "preflight"][-1]["gates"]
    ok(gts["G21_vdac_hb_copy_isolation"]["ok"] is True and VP.judge_g21(gts["G21_vdac_hb_copy_isolation"]) == [], f"G21 in the REAL preflight and judged: {VP.judge_g21(gts['G21_vdac_hb_copy_isolation'])}")
    ok(gts["G12_aux_drv_live_and_identity"]["scene_proj_calls"] == 3 and gts["G12_aux_drv_live_and_identity"]["tap_index"] == 2 and gts["G12_aux_drv_live_and_identity"]["ok"] is True, "G12's tap row reads the copy's ctx (outs[2]) in the real trainer")
    ns, err = C.parse_scorer_argv(__import__("scorer_finetune").build_parser(), r["argv"] + ["--preflight"])
    ok(not err and ns.aux_drv_trunk_lora == "on" and ns.aux_drv_lora_lr == 1e-3, f"the trainer's parser takes the flags: {err}")
    argv_off = list(r["argv"])
    for tok in ("--aux-drv-trunk-lora", "--aux-drv-lora-lr"):
        i_ = argv_off.index(tok)
        del argv_off[i_:i_ + 2]
    ns_off = C.parse_scorer_argv(__import__("scorer_finetune").build_parser(), argv_off)[0]
    ok("G21" in C.scorer_required_gates(ns) and "G21" not in C.scorer_required_gates(ns_off), "G21 is required iff --aux-drv-trunk-lora on")
    d = W.data_event(r)
    ok(d["aux_hb"]["trunk_copy_built"] and d["aux_hb"]["lora_lr"] == 1e-3 and d["aux_hb"]["n_lora_trainable"] == d["aux_hb"]["lora_params_in_optimiser"] == d["aux_hb"]["lora_expected"] == 4 * (12 + 256), f"data event aux_hb {d['aux_hb']}")
    rows = C.scorer_dvb_rows(ns, d, {})
    bad_rows = [x for x in rows if not x["ok"]]
    ok(not bad_rows and {x["dest"] for x in rows} >= {"aux_drv_trunk_lora", "aux_drv_lora_lr"}, f"G-DVB rows (HB) all OK: {bad_rows}")
    # the run: the copy MOVED, P6 did not
    init = torch.load(os.path.join(os.path.dirname(r["out"]), "ck.pt"), weights_only=False)["model"]
    ok(r["rc_run"] == 0 and r["aux"]["format"] == "aux_drv/2", f"the run finishes and writes aux_drv/2 ({r['rc_run']}, {r['aux'] and r['aux']['format']})")
    lt = r["aux"]["hb_lora"]
    ok(set(lt) == {"lin.A", "lin.B"} and any(not torch.equal(lt[k], init["backbone." + k]) for k in lt), "REACHABLE: the copy's LoRA tensors moved during the run (trained != P6's)")
    ok(all(torch.equal(r["A"]["model"][k], init[k]) for k in init if not k.split(".")[0] in SCORER) and all(torch.equal(r["B"]["model"][k], init[k]) for k in init if k.startswith("backbone.")),
       "P6's trunk tensors in model_sft_A / B are bit-identical to the checkpoint (the planner's strict file never carries the copy)")
    ok(r["aux"]["update"] == 25 and set(r["aux"]["opt_lora"]["state"]) and len(r["aux"]["opt_lora"]["state"]) == 2 and "HB" in r["states"], "both Adam states + the states file carry the copy")
    ok(PG.judge(r["aux"], 25, n_params=2440450, n_lora=4 * (12 + 256)) == [], f"the post-training gate accepts the real run's aux_drv.pt: {PG.judge(r['aux'], 25, 2440450, 4 * (12 + 256))}")
    sft = [x for x in r["rows"]["sft.jsonl"] if x.get("event") == "train"]
    ok(sft and sft[-1]["aux_lr"] >= 0.0, "training rows carry aux_lr")
    # ---- the head-less comparator: the scorer arms are bit-identical to an HA run (the copy is read by the head only)
    ha = smoke("ha", extra=W.H1_ARGS)
    ok(all(torch.equal(r["B"]["model"][k], ha["B"]["model"][k]) for k in r["B"]["model"] if k.split(".")[0] in SCORER) and all(torch.equal(r["A"]["model"][k], ha["A"]["model"][k]) for k in r["A"]["model"]),
       "NO SCORER PATH: arm A and arm B of an HB run equal an HA run bit for bit (the copy cannot reach the scorer)")
    ok(any(not torch.equal(r["aux"]["head"][k], ha["aux"]["head"][k]) for k in r["aux"]["head"]), "the head DOES differ (it reads the copy, which moved)")

    # ---- MUTATION: delete the one wiring line -> the copy's LoRA never gets a gradient -> G21 RED, no PREFLIGHT_PASS
    line = "                    actx = HB.select_actx(hb, model, img, cal if cal.numel() else None, sctx, lambda: model.ego_enc(torch.cat([ego, model.cmd_embed(goal.to(ego.dtype))], -1)).sum(-1))\n"
    m1 = smoke("mut_wire", src=mutated(line, "                    pass\n"), run=False)
    g21 = [x for x in m1["rows"]["preflight.jsonl"] if x.get("event") == "preflight"][-1]["gates"]["G21_vdac_hb_copy_isolation"]
    red(VP.judge_g21(g21), "NO gradient", "trainer: the select_actx wiring line DELETED")
    ok(m1["rc_pre"] != 0 and m1["pass_json"] is None, "a deleted wiring line withholds PREFLIGHT_PASS.json")
    # ... the build line deleted: the flag is named but no copy is built
    m2 = smoke("mut_build", src=mutated("            hb = HB.HBTrunk(model).to(dev)\n            optL = HB.build_lora_optimizer(hb, a.aux_drv_lora_lr)\n", "            pass\n"), run=False)
    g2 = [x for x in m2["rows"]["preflight.jsonl"] if x.get("event") == "preflight"][-1]["gates"]
    ok("G21_vdac_hb_copy_isolation" not in g2 and "G21" in C.scorer_required_gates(ns), "no copy built: the preflight has no G21, which the argv OBLIGES (consume_scorer_preflight refuses a missing required gate)")
    d2 = W.data_event(m2)
    red([x["how"] for x in C.scorer_dvb_rows(ns, d2, {}) if not x["ok"]], "aux_hb", "G-DVB: --aux-drv-trunk-lora on, no copy built")
    # ... the optimiser step deleted: the copy never moves
    step_line = "                HB.lora_step(hb, optL, lr_at(upd) * (a.aux_drv_lora_lr / a.lr))\n"
    m3 = smoke("mut_step", src=mutated(step_line, "                pass\n"))
    lt3 = m3["aux"]["hb_lora"] if m3["aux"] else {}
    N[0] += 1
    ok(m3["aux"] is not None and all(torch.equal(lt3[k], init["backbone." + k]) for k in lt3), "MUTATION control: with the step deleted the copy equals P6's (so the 'moved' check above is able to read a stall)")
    TABLE.append(("trainer: HB.lora_step line deleted", "RED", "the reachability check (copy moved) fails"))
    # ... runtime switches
    for envname, why in ((HB.ENV_P6CTX, "head reads P6's ctx"), (HB.ENV_SHARE, "shared copy"), (HB.ENV_UNFREEZE_SP, "P6 scene_proj unfrozen"), (HB.ENV_DETACH, "copy detached"), (HB.ENV_STALE_TAP, "stale G12 tap index")):
        mm = smoke("env_" + envname[-12:], env_={envname: "1"}, run=False)
        N[0] += 1
        ok(mm["rc_pre"] != 0 and mm["pass_json"] is None, f"{envname}: the preflight must FAIL and withhold PREFLIGHT_PASS ({why}): rc {mm['rc_pre']}")
        TABLE.append((f"preflight under {envname} ({why})", "RED", "rc != 0, no PREFLIGHT_PASS.json"))

    # ---- refusals (flags without the mechanism; defaults disable it)
    for extra, needle in ((W.H1_ARGS + ["--aux-drv-lora-lr", "1e-3"], "--aux-drv-trunk-lora is not on"), (W.H1_ARGS + ["--aux-drv-trunk-lora", "on"], "needs --aux-drv-lora-lr"),
                          (W.H1_ARGS + ["--aux-drv-trunk-lora", "on", "--aux-drv-lora-lr", "1e-3", "--aux-drv-feed", "on"], "H1 arm"), (["--aux-drv-trunk-lora", "on"], "--aux-drv is not")):
        x = smoke("ref_" + str(abs(hash(tuple(extra)))), extra=extra, run=False, preflight=True)
        N[0] += 1
        ok(x["rc_pre"] == 4 and needle in x["out_pre"], f"refused: {needle}: {x['rc_pre']} {x['out_pre'][:160]}")
    ha_ns = C.parse_scorer_argv(__import__("scorer_finetune").build_parser(), ha["argv"])[0]
    ok(ha_ns.aux_drv_trunk_lora == "off" and ha_ns.aux_drv_lora_lr is None and ha_ns.aux_drv_cell == 2.0 and ha_ns.aux_drv_bw == 0.0, "(d) every new default disables its mechanism")

    # ---- HC: the cell flag REACHES the build; the weight reaches the loss; G-DVB reads both
    hc = smoke("hc", extra=W.H1_ARGS + ["--aux-drv-cell", "1.0", "--aux-drv-bw", "3.0", "--aux-drv-bw-band-m", "3.0"], run=False)
    ok(hc["rc_pre"] == 0, f"an HC preflight (cell 1.0, bw 3) passes on the stub: {hc['rc_pre']} {hc['out_pre'][-300:]}")
    dh = W.data_event(hc)
    nsh = C.parse_scorer_argv(__import__("scorer_finetune").build_parser(), hc["argv"] + ["--preflight"])[0]
    ok((dh["aux_drv"]["nx"], dh["aux_drv"]["ny"], dh["aux_drv"]["cell"], dh["aux_drv"]["n_params"]) == (80, 64, 1.0, 3423490), f"T12: --aux-drv-cell 1.0 reaches the build: {dh['aux_drv']}")
    ok(abs(dh["aux_drv"]["bw_probe"]["inside_over_outside"] - 4.0) < 1e-4, "T12: the boundary weight reaches the built loss (4 inside / 1 outside)")
    ok(any(x["dest"] == "aux_drv_cell" for x in C.scorer_dvb_rows(nsh, dh, {})) and any(x["dest"] == "aux_drv_bw" for x in C.scorer_dvb_rows(nsh, dh, {})), "G-DVB observes the cell and the weight under HC")
    ok(all(x["ok"] for x in C.scorer_dvb_rows(nsh, dh, {})), f"G-DVB reads HC OK: {[x for x in C.scorer_dvb_rows(nsh, dh, {}) if not x['ok']]}")
    cellline = "cell=a.aux_drv_cell, bw=a.aux_drv_bw, bw_band_m=a.aux_drv_bw_band_m, "
    hcm = smoke("hc_mut", extra=W.H1_ARGS + ["--aux-drv-cell", "1.0", "--aux-drv-bw", "3.0", "--aux-drv-bw-band-m", "3.0"], src=mutated(cellline, ""), run=False)
    red([x["flag"] + " " + x["how"] for x in C.scorer_dvb_rows(nsh, W.data_event(hcm), {}) if not x["ok"]], "--aux-drv-cell", "trainer: cell / bw dropped from the build (the unreachable-flag class)")
    # ---- the real HC training run: held-out read at the 1 m grid works
    hcr = smoke("hc_run", extra=W.H1_ARGS + ["--aux-drv-cell", "1.0", "--aux-drv-bw", "3.0"])
    ev = [x for x in hcr["rows"]["sft.jsonl"] if x.get("event") == "aux_eval"]
    ok(hcr["rc_run"] == 0 and ev and ev[-1]["n_cells"] > 0, f"an HC run trains and reads its held-out at the 1 m grid ({hcr['rc_run']})")


def launch_layer():
    toks = LG.vdac_argv("HB-s0")
    ns = C.parse_scorer_argv(__import__("scorer_finetune").build_parser(), toks)[0]
    ok(ns.aux_drv_trunk_lora == "on" and ns.aux_drv_lora_lr == 2e-4 and ns.aux_drv_cell == 2.0 and ns.seed == 0 and ns.max_updates == 10000, "(d) vdac_argv HB-s0 NAMES both HB levers (lr = P6's own 2e-4)")
    ok("--aux-drv-trunk-lora" in toks and "--aux-drv-lora-lr" in toks, "both levers are tokens of the argv")
    sh = LG.vdac_argv("HB-SHUF")
    ok(sh[sh.index("--aux-drv-shuffle-seed") + 1] == "0" and sh[sh.index("--max-updates") + 1] == "2000" and "--aux-drv-trunk-lora" in sh, "HB-SHUF: the derangement control with the copy, 2,000 updates")
    hc = LG.vdac_argv("HC-s0")
    nsc = C.parse_scorer_argv(__import__("scorer_finetune").build_parser(), hc)[0]
    ok(nsc.aux_drv_cell == 1.0 and nsc.aux_drv_bw == 3.0 and nsc.aux_drv_bw_band_m == 3.0 and nsc.aux_drv_trunk_lora == "off", "(d) vdac_argv HC-s0 NAMES the cell and the weight")
    ha_old = LG.vdac_argv("HA-s0")
    ok("--aux-drv-trunk-lora" not in ha_old and "--aux-drv-cell" not in ha_old, "HA-s0's argv is unchanged (no new flag)")
    for arm in ("HB-s0", "HB-SHUF", "HC-s0", "HC-SHUF"):
        pr = LG.PROFILES["refe_vdac_p6"]
        ok(arm in pr["vdac_arms"] and pr["arm_class"][arm] in pr["arm_recipe"] and pr["hb_lora_params"] == 3145728, f"profile carries {arm} with a recipe")
    # the gate's own required-flag / dvb table covers every new dest (a new trainer flag without a G-DVB entry is refused)
    for dest in ("aux_drv_trunk_lora", "aux_drv_lora_lr", "aux_drv_cell", "aux_drv_bw", "aux_drv_bw_band_m"):
        N[0] += 1
        ok(any(isinstance(v, dict) and dest in v for v in vars(C).values()), f"{dest} has a G-DVB table entry")


def layer_check():
    import hashlib

    import vdac_a5_layer as A5
    import vdac_layer as VL

    def blob(s):
        b = s.encode("utf-8")
        return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()
    ok(A5.has_layer(SRC) and blob(A5.peel(SRC)) == A5.TIP_TRAINER_BLOB == "4df07e51278303abd9854f4f61f9bc4960b5a98d", "the A5 layer peels scorer_finetune.py back to the a3cc6a77 blob (git blob 4df07e51)")
    ok(A5.peel(A5.peel(SRC)) == A5.peel(SRC) and VL.peel(SRC) == VL.peel(A5.peel(SRC)), "peel is idempotent and vdac_layer.peel calls the A5 layer first")
    try:
        A5.peel(SRC.replace("optL = HB.build_lora_optimizer(hb, a.aux_drv_lora_lr)", "optL = None"))
        ok(False, "a drifted trainer was peeled")
    except AssertionError as e:
        N[0] += 1
        TABLE.append(("layer: a trainer drifted from the A5 text", "RED", "reversal asserts"))
        ok("reversal" in str(e), str(e)[:80])


def main() -> int:
    module_layer()
    launch_layer()
    layer_check()
    caller_layer()
    print(f"{'mutation arm':72s} result  how")
    for what, res, how in TABLE:
        print(f"{what[:72]:72s} {res:7s} {how}")
    print(f"ZZTEST_PASS {N[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
