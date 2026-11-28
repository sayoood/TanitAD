#!/usr/bin/env python3
"""Tests for VDAC A6's arm HBC (HB's LoRA trunk copy + HC's 1 m cells and boundary-weighted Huber, in ONE arm).  CPU only (CUDA_VISIBLE_DEVICES=-1), no pod, no checkpoint, no data.
Run:  CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 python refe/test_vdac_hbc.py   ->   ZZTEST_PASS <n>   (prints the mutation table)

WHY A NEW TEST.  HBC combines two levers that each have their own reachability test (test_vdac_hb.py: HB's copy, HC's cell and weight).  "Each works alone" is not "both reach in the SAME build": the head built at
1 m (80 x 64 = 5,120 queries, 3,423,490 parameters) must read the COPY's ctx, the boundary weight must change the loss that backpropagates INTO the copy's LoRA tensors, and the gate must refuse an arm that declares
one lever and builds the other.  Layers (CLAUDE.md "built, tested, and unreachable from its caller": every flag traced argparse -> config -> built module -> loss -> backward, each hop with a mutation that goes RED):
  LAUNCH   vdac_argv / vdac_pod.sh MODE=argv / the pause stage / the chain table / the profile: HBC-s0 and HBC-SHUF NAME all five levers (defaults disable every one of them), HB's memory fallback, HB's allocator setting.
  MODULE   the REAL tiny REFe (model.py's LoRALinear) with the REAL 1 m head: step-0 equality, non-zero gradient on every LoRA tensor AND on the head's 5,120 query rows, exact 0.0 at weight 0, the weight reaches the LoRA
           gradient, the four HB regression switches stay RED with the 1 m head, a head-side dead gradient is RED.
  CALLER   the REAL scorer_finetune.main() (test_aux_s1_wiring's harness + test_vdac_hb's HB stub): HB + HC flags together; the built head reads 80 x 64 / 3,423,490 and the copy is built in the SAME data event; each wiring line
           DELETED (copy build, select_actx, lora_step, cell, bw, shuffle) must turn the check that owns it RED.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TMPROOT = os.environ.get("TMP") or tempfile.gettempdir()

import torch  # noqa: E402

import aux_drivable_head as AUX  # noqa: E402
import aux_hb as HB  # noqa: E402
import gate_checks as C  # noqa: E402
import launch_gate as LG  # noqa: E402
import test_aux_s1_wiring as W  # noqa: E402
import test_vdac_hb as T  # noqa: E402
import test_vdac_hbmem as M  # noqa: E402
import vdac_postgate as PG  # noqa: E402
import vdac_probes as VP  # noqa: E402

N = [0]
TABLE: list = []
HC_FLAGS = ["--aux-drv-cell", "1.0", "--aux-drv-bw", "3.0", "--aux-drv-bw-band-m", "3.0"]
HBC_ARGS = T.HB_ARGS + HC_FLAGS
HEAD_1M = 3_423_490
LORA_VITL16 = 3_145_728
SCORER = T.SCORER


def ok(cond, msg):
    N[0] += 1
    assert cond, msg


def red(reasons, needle, what):
    """a mutation arm: the judge must be NON-EMPTY and name `needle`"""
    N[0] += 1
    assert reasons and any(needle in r for r in reasons), f"MUTATION NOT CAUGHT ({what}): {reasons}"
    TABLE.append((what, "RED", needle))


def pairs(argv):
    return {(f, tuple(v)) for f, v in LG.GT.flag_pairs(list(argv))}


# ================================================================================================ LAUNCH
def launch_layer():
    hb, hc, hbc = LG.vdac_argv("HB-s0"), LG.vdac_argv("HC-s0"), LG.vdac_argv("HBC-s0")
    ns = C.parse_scorer_argv(__import__("scorer_finetune").build_parser(), hbc)[0]
    ok(ns.aux_drv_trunk_lora == "on" and ns.aux_drv_lora_lr == 2e-4 and ns.aux_drv_cell == 1.0 and ns.aux_drv_bw == 3.0 and ns.aux_drv_bw_band_m == 3.0, "HBC-s0 NAMES all five levers (the defaults off / None / 2.0 / 0.0 disable them)")
    ok(ns.seed == 0 and ns.max_updates == 10000 and ns.eval_every == 5000 and ns.batch == 4 and ns.accum == 2 and ns.aux_drv_shuffle_seed is None and ns.aux_drv and ns.w_aux_drv == 0.1 and ns.aux_drv_lr == 2e-4,
       "HBC-s0: seed 0, 10,000 updates, eval 5,000, the registered fallback batch 4 x accum 2 (effective 8), head recipe unchanged")
    ok(pairs(hbc) - pairs(hb) == {("--aux-drv-cell", ("1.0",)), ("--aux-drv-bw", ("3.0",)), ("--aux-drv-bw-band-m", ("3.0",)), ("--out", ("/workspace/data/refe_vdac/hbc-s0",)), ("--dump-scores", ("/workspace/data/refe_vdac/hbc-s0/scores",))}
       and pairs(hb) - pairs(hbc) == {("--out", ("/workspace/data/refe_vdac/hb-s0",)), ("--dump-scores", ("/workspace/data/refe_vdac/hb-s0/scores",))},
       "HBC-s0's argv == HB-s0's + the three HC flags (and its own --out): the arm is exactly HB plus HC, nothing else moved")
    ok(pairs(hbc) - pairs(hc) >= {("--aux-drv-trunk-lora", ("on",)), ("--aux-drv-lora-lr", ("0.0002",)), ("--batch", ("4",)), ("--accum", ("2",))}, "HBC-s0 differs from HC-s0 by HB's flags and HB's batch fallback")
    sh = LG.vdac_argv("HBC-SHUF")
    nss = C.parse_scorer_argv(__import__("scorer_finetune").build_parser(), sh)[0]
    ok(nss.aux_drv_shuffle_seed == 0 and nss.max_updates == 2000 and nss.eval_every == 1000 and nss.seed == 0 and nss.aux_drv_trunk_lora == "on" and nss.aux_drv_cell == 1.0 and nss.aux_drv_bw == 3.0 and nss.batch == 4 and nss.accum == 2,
       "HBC-SHUF: the same five levers + the derangement (--aux-drv-shuffle-seed 0), 2,000 updates")
    ok(pairs(sh) - pairs(hbc) == {("--aux-drv-shuffle-seed", ("0",)), ("--max-updates", ("2000",)), ("--eval-every", ("1000",)), ("--out", ("/workspace/data/refe_vdac/hbc-shuf",)), ("--dump-scores", ("/workspace/data/refe_vdac/hbc-shuf/scores",))},
       "HBC-SHUF == HBC-s0 + the shuffle, the short budget and its own --out")
    ok(all(a not in LG.vdac_argv("HA-s0") for a in ("--aux-drv-trunk-lora", "--aux-drv-cell", "--aux-drv-bw")) and LG.vdac_argv("HB-s0") == hb and LG.vdac_argv("HC-s0") == hc, "the five older arms' argv are unchanged by the HBC edit")
    pr = LG.PROFILES["refe_vdac_p6"]
    for arm, cls in (("HBC-s0", "H1hbc"), ("HBC-SHUF", "H1hbcshuf")):
        ok(arm in pr["vdac_arms"] and pr["arm_class"][arm] == cls and cls in pr["arm_recipe"] and pr["arm_recipe"][cls]["batch"] == 4 and pr["arm_recipe"][cls]["accum"] == 2, f"the profile carries {arm} (class {cls}, batch 4 / accum 2)")
    ok(pr["hb_lora_params"] == LORA_VITL16 == LG.VDAC_HB_LORA_PARAMS and C.AUX_HEAD_PARAMS_BY_CELL[1.0] == HEAD_1M, "the registered counts: LoRA 3,145,728 and the 1 m head 3,423,490")
    # the gate's pin: every argv lever moved is refused (the pinned_argv_refusals of vdac_arm_refusals)
    for what, mut in (("cell dropped", lambda a: [t for t in a if t not in ("--aux-drv-cell", "1.0")]), ("bw dropped", lambda a: a[:a.index("--aux-drv-bw")] + a[a.index("--aux-drv-bw") + 2:]),
                      ("lora lr dropped", lambda a: a[:a.index("--aux-drv-lora-lr")] + a[a.index("--aux-drv-lora-lr") + 2:]), ("trunk lora off", lambda a: [("off" if t == "on" and a[i - 1] == "--aux-drv-trunk-lora" else t) for i, t in enumerate(a)]),
                      ("batch 8", lambda a: [("8" if a[i - 1] == "--batch" else t) for i, t in enumerate(a)])):
        bad = LG.pinned_argv_refusals({**pr, "pinned_argv": tuple(LG.vdac_argv("HBC-s0"))}, mut(list(LG.vdac_argv("HBC-s0"))))
        N[0] += 1
        ok(bad, f"MUTATION: the pinned HBC argv with {what} must be refused")
        TABLE.append((f"launch_gate pinned argv: HBC-s0 with {what}", "RED", "pinned_argv_refusals non-empty"))
    # declared but not built, at the CLI level (hb_refusals): the one combination HBC adds is that cell / bw / lora lr may all be on together, and each alone without its partner flag is still refused
    class A:
        pass
    a = A()
    a.aux_drv, a.aux_drv_trunk_lora, a.aux_drv_lora_lr, a.aux_drv_cell, a.aux_drv_bw, a.aux_drv_bw_band_m, a.aux_drv_feed, a.aux_drv_tap = True, "on", 2e-4, 1.0, 3.0, 3.0, "off", "visual"
    ok(HB.hb_refusals(a) == [], "hb_refusals accepts the HBC combination")
    for nm, val, needle in (("aux_drv_lora_lr", None, "needs --aux-drv-lora-lr"), ("aux_drv_trunk_lora", "off", "--aux-drv-lora-lr is set but"), ("aux_drv_bw", 0.0, "--aux-drv-bw-band-m is set but --aux-drv-bw is 0")):
        b = A()
        b.__dict__.update(a.__dict__)
        setattr(b, nm, val)
        if nm == "aux_drv_bw":
            b.aux_drv_bw_band_m = 2.0          # the band named while the weight is 0: a lever that would silently do nothing (bw 0 with the default band is the pinned-argv's job, tested above)
        red(HB.hb_refusals(b), needle, f"CLI: HBC with {nm}={val!r} refused (a lever declared, not built)")
    # the pod script, the chain table and the pause stage
    for arm in ("HBC-s0", "HBC-SHUF"):
        r = M.sh((HERE / "vdac_pod.sh").as_posix(), {"ARM": arm, "MODE": "argv", "VDAC_WS": "/workspace"}, HERE.as_posix())
        ok(r.returncode == 0 and r.stdout.split() == LG.vdac_argv(arm), f"{arm}: vdac_pod.sh MODE=argv equals launch_gate.vdac_argv token for token")
    POD = (HERE / "vdac_pod.sh").read_bytes().decode("utf-8")
    for arm in ("HBC-s0", "HBC-SHUF"):
        for mode in ("preflight", "run", "mutate", "mutate_ego"):
            ok(M.pod_env(POD, arm, mode) == M.WANT, f"{arm} {mode}: the trainer's process holds PYTORCH_CUDA_ALLOC_CONF={M.WANT} (HB's allocator setting)")
    ok(M.pod_env(POD.replace('EXTRA="$HBX $HCX"; BATCH=4; ACCUM=2; ALLOC="expandable_segments:True" ;;          # VDAC A6', 'EXTRA="$HBX $HCX"; BATCH=4; ACCUM=2; ALLOC="" ;;          # VDAC A6'), "HBC-s0", "preflight") == "UNSET", "mutation control: HBC-s0 without its ALLOC reads UNSET")
    TABLE.append(("pod script: HBC-s0 ALLOC dropped", "RED", "the trainer's environment reads UNSET"))
    N[0] += 1
    chain = (HERE / "vdac_chain.sh").read_bytes().decode("utf-8")
    pause = (HERE / "vdac_pause_p7.sh").read_bytes().decode("utf-8")
    ok("HBC-s0)  HAUPD=10000; SHUFSEED=none; HEADN=3423490" in chain and "HBC-SHUF) HAUPD=2000;  SHUFSEED=0; HEADN=3423490" in chain, "vdac_chain.sh arm table: HBC-s0 10,000 / HBC-SHUF 2,000 with the shuffle seed 0 and the 1 m head count")
    ok("--updates $HAUPD --n-params $HEADN" in chain and chain.count("HEADN=2440450") == 1 and chain.count("HEADN=3423490") == 4, "the chain's post-training gate is told the arm's BUILT head count (every 1 m arm: HC-s0, HC-SHUF, HBC-s0, HBC-SHUF)")
    ok('A6HBC) ARMS="HBC-s0 HBC-SHUF" ;;' in pause and "HBC-s0) TMO=50400" in pause and "HBC-SHUF) TMO=21600" in pause and "A5HB) ARMS=\"HB-s0 HB-SHUF\"" in pause, "vdac_pause_p7.sh: stage A6HBC runs HBC-s0 then HBC-SHUF (as A5HB), with bounded timeouts")
    ok('A5HCSHUF) ARMS="HC-SHUF" ;;' in pause and "HA-SHUF|HC-SHUF) TMO=7200" in pause and "A5HCSHUF |" in pause, "vdac_pause_p7.sh: stage A5HCSHUF (VDAC A5 A2) runs HC-SHUF alone, under its 7,200 s timeout, on the fixed chain")
    N[0] += 1
    ok(pause.index("HBC-s0 HBC-SHUF") < pause.index("for A in $ARMS") and "break; }" in pause, "the first arm of a stage that does not finish stops the stage (HBC-SHUF is not run after a failed HBC-s0)")
    # the post-training gate: the head count it is told must be the built one.  The chain USED to call it without --n-params (default 2,440,450): a 1 m arm's Adam state (3,423,490) would be refused
    ck = {"format": "aux_drv/2", "update": 3, "head": {"q": torch.zeros(1)}, "hb_lora": {"x.A": torch.zeros(4)}, "opt": None, "opt_lora": None}
    q = torch.ones(HEAD_1M)
    ck["opt"] = {"state": {0: {"exp_avg": q, "exp_avg_sq": q, "step": torch.tensor(3.0)}}, "param_groups": [{"params": [0]}]}
    ql = torch.ones(4)
    ck["opt_lora"] = {"state": {0: {"exp_avg": ql, "exp_avg_sq": ql, "step": torch.tensor(3.0)}}, "param_groups": [{"params": [0]}]}
    ok(PG.judge(ck, 3, n_params=HEAD_1M, n_lora=4) == [], f"postgate: a 1 m head's Adam state passes when told the 1 m count: {PG.judge(ck, 3, n_params=HEAD_1M, n_lora=4)}")
    red(PG.judge(ck, 3, n_lora=4), "head has 2440450", "postgate WITHOUT --n-params on a 1 m arm (the chain before this edit)")
    red(PG.judge(ck, 3, n_params=HEAD_1M + 1, n_lora=4), "head has", "postgate: a head count that is not the built one")
    ok(chain.replace(" --n-params $HEADN", "") != chain, "mutation text: the chain without --n-params differs (and the 1 m arm judge above is RED on it)")
    # the registered counts are ARITHMETIC on the real vitl16 module shapes (meta device: no weights, no memory)
    try:
        from model import REFe, REFeConfig
        with torch.device("meta"):
            big = REFe(REFeConfig(backbone="vitl16"))
        ok(HB.expected_lora_params(big.backbone) == LORA_VITL16, f"vitl16 (meta) LoRA arithmetic == 3,145,728: {HB.expected_lora_params(big.backbone)}")
    except Exception as e:  # noqa: BLE001 -- reported, never silent: the count then rests on the arithmetic test of test_vdac_hb.py
        N[0] += 1
        TABLE.append((f"vitl16 meta-device LoRA count: not built ({type(e).__name__}: {str(e)[:80]})", "UNVERIFIED", "falls back to test_vdac_hb arithmetic"))


# ================================================================================================ MODULE
def head_reach(rig, w0=False):
    """the 1 m head's own gradient fields after the run's own backward at the declared weight (or at weight 0): the query tensor `q` (every one of the 5,120 rows) and every other head tensor"""
    for p in list(rig.aux.head.parameters()) + rig.hb.lora_parameters():
        p.grad = None
    keep = rig.aux.w_aux
    if w0:
        rig.aux.w_aux = 0.0
    try:
        rig.fwd()
        rig.aux_backward(rig.lv, 1.0)
    finally:
        rig.aux.w_aux = keep
    q = rig.aux.head.q
    g = {"q_none": q.grad is None, "q_rows": int(q.shape[1]), "q_rows_zero": (int((q.grad.abs().sum(-1) == 0).sum()) if q.grad is not None else None), "q_max": (float(q.grad.abs().max()) if q.grad is not None else None),
         "head_none": sum(1 for p in rig.aux.head.parameters() if p.grad is None), "head_n": sum(1 for _ in rig.aux.head.parameters()),
         "head_min": min((float(p.grad.abs().max()) for p in rig.aux.head.parameters() if p.grad is not None), default=None),
         "lora_none": sum(1 for p in rig.hb.lora_parameters() if p.grad is None), "lora_min": min((float(p.grad.abs().max()) for p in rig.hb.lora_parameters() if p.grad is not None), default=None),
         "lora_max": max((float(p.grad.abs().max()) for p in rig.hb.lora_parameters() if p.grad is not None), default=None)}
    return g


def judge_head_reach(g, w0=False):
    r = []
    if w0:
        if g["q_none"] or g["q_max"] != 0.0 or g["head_min"] != 0.0 or g["lora_none"] or g["lora_max"] != 0.0:
            r.append(f"weight-0 control does not read EXACTLY 0.0 on the 1 m head and the copy: {g}")
        return r
    if g["q_none"] or g["q_rows"] != 5120 or g["q_rows_zero"] != 0:
        r.append(f"the 1 m queries do not ALL hold a non-zero gradient (rows {g['q_rows']}, zero rows {g['q_rows_zero']}, none {g['q_none']}): NO gradient")
    if g["head_none"] or not g["head_min"] or g["head_min"] <= 0.0:
        r.append(f"a head tensor holds NO gradient ({g['head_none']} None of {g['head_n']}, min {g['head_min']})")
    if g["lora_none"] or not g["lora_min"] or g["lora_min"] <= 0.0:
        r.append(f"a copy LoRA tensor holds NO gradient through the 1 m head ({g['lora_none']} None, min {g['lora_min']})")
    return r


def module_layer():
    rig = T.Rig(cell=1.0, bw=3.0)
    bf = rig.aux.built_facts()
    ok((bf["nx"], bf["ny"], bf["n_params"], bf["cell"]) == (80, 64, HEAD_1M, 1.0) and abs(bf["bw_probe"]["inside_over_outside"] - 4.0) < 1e-4, f"the built 1 m head and its weight: {bf['nx'], bf['ny'], bf['n_params']}")
    hf = rig.hb.built_facts(rig.model, SCORER)
    ok(hf["n_lora_trainable"] == hf["lora_expected"] and hf["trainable_all_lora"] and hf["copy_storage_disjoint"] and hf["p6_params_requires_grad"] == 0 and bf["n_params"] == HEAD_1M,
       "SAME BUILD: the copy (LoRA == its arithmetic, all trainable tensors LoRA, disjoint storage) and the 1 m head (3,423,490) are built together")
    # T-step0: the 1 m head reads a ctx that equals P6's visual_ctx bit for bit at step 0, so its prediction equals the frozen-trunk prediction (HC) at step 0
    rig.fwd()
    sctx = rig.cap["sctx"]
    ok(HB.step0_equal(rig.model, rig.hb, rig.img, None, sctx)["step0_ctx_equal"], "step 0: the copy's ctx == P6's visual_ctx bit for bit")
    with torch.no_grad():
        l_copy, s_copy = rig.aux.predict(rig.hb.ctx(rig.model, rig.img, None))
        l_p6, s_p6 = rig.aux.predict(sctx)
    ok(tuple(l_copy.shape) == (2, 80, 64) and torch.equal(l_copy, l_p6) and torch.equal(s_copy, s_p6), "step 0: the 1 m head's prediction on the copy's ctx == on P6's ctx (HBC starts where HC starts)")
    # live gradient + exact-zero control
    g = head_reach(rig)
    ok(judge_head_reach(g) == [], f"every one of the 5,120 query rows, every head tensor and every LoRA tensor holds a non-zero gradient: {g}")
    g0 = head_reach(rig, w0=True)
    ok(judge_head_reach(g0, w0=True) == [], f"CONTROL at --w-aux-drv 0: the head and the copy read EXACTLY 0.0 (a tensor, not None): {g0}")
    # G21 on the 1 m rig
    g21 = rig.g21()
    ok(VP.judge_g21(g21) == [] and g21["lora_grad_declared"]["min_abs"] > 0.0 and g21["lora_grad_w0"]["max_abs"] == 0.0, f"G21 on the 1 m rig: {VP.judge_g21(g21)}")
    # the boundary weight reaches the LoRA gradient (same init, same batch: bw 0 vs bw 3 must differ), and bw 0 is bit-identical to the unweighted loss through the head's own loss()
    r0 = T.Rig(cell=1.0, bw=0.0)
    g_bw0 = head_reach(r0)
    ok(g["lora_max"] != g_bw0["lora_max"] or g["head_min"] != g_bw0["head_min"], "the boundary weight reaches the copy: the LoRA / head gradients under bw 3 differ from bw 0")
    lg, sd = r0.cap["aux"][0].detach(), r0.cap["aux"][1].detach()
    lv = r0.lv
    base = AUX.aux_loss(lg, sd, lv[0], lv[1], cell=1.0)
    l_w0 = r0.aux.loss(lg, sd, lv[0], lv[1])
    l_w3 = rig.aux.loss(lg, sd, lv[0], lv[1])
    ok(torch.equal(l_w0[1], l_w3[1]) and float(l_w3[2]) > float(l_w0[2]), "bw 3 weights the Huber term only (BCE unchanged), upward")
    ok(all(torch.equal(x, y) for x, y in zip(base, l_w0)), "bw 0.0 through the head's loss() == the unweighted aux_loss, bit for bit")
    # MUTATION: the weight line deleted from the loss -> bw named, weight not applied
    src_aux = (HERE / "aux_drivable_head.py").read_text(encoding="utf-8")
    snippet = "        hub = hub * (1.0 + bw * (t.sdf_m.abs() < bw_band_m).to(hub.dtype))\n"
    ok(src_aux.count(snippet) == 1, "the weight line occurs once")
    mod = types.ModuleType("aux_nobw_hbc")
    mod.__file__ = str(HERE / "aux_drivable_head.py")
    sys.modules["aux_nobw_hbc"] = mod
    exec(compile(src_aux.replace(snippet, "        pass" + chr(10)), "aux_nobw_hbc.py", "exec"), mod.__dict__)
    mh = mod.AuxDrivable(mod.AuxDrivableConfig(cell=1.0, bw=3.0))
    lm = mh.loss(lg, sd, lv[0], lv[1])
    red([] if not torch.equal(lm[2], l_w0[2]) else ["the mutant's weighted Huber equals the unweighted: weight not applied (bw named, dropped from the loss)"], "weight not applied", "module: bw 3 named, weight line deleted from the loss")
    # MUTATIONS on the head's gradient fields
    rq = T.Rig(cell=1.0, bw=3.0)
    rq.aux.head.q.requires_grad_(False)
    red(judge_head_reach(head_reach(rq)), "NO gradient", "module: the 1 m query tensor frozen (the head's 5,120 queries do not train)")
    rd = T.Rig(cell=1.0, bw=3.0)
    rd.aux_backward = lambda lv_, div=1.0, retain=False: None
    red(judge_head_reach(head_reach(rd)), "NO gradient", "module: the head's backward never runs (the loss does not reach the head or the copy)")
    red(judge_head_reach(dict(g0, q_max=1e-9), w0=True), "EXACTLY 0.0", "module: a dead weight-0 control (a probe that cannot read zero)")
    # the four HB regression switches stay RED with the 1 m head
    for envname, needle, what in ((HB.ENV_P6CTX, "NO gradient", "head reads P6's ctx"), (HB.ENV_DETACH, "NO gradient", "copy detached"), (HB.ENV_SHARE, "copy_storage_disjoint", "shared copy"),
                                  (HB.ENV_UNFREEZE_SP, "requires_grad True", "P6 scene_proj unfrozen")):
        with T.env(**{envname: "1"}):
            rm = T.Rig(cell=1.0, bw=3.0)
            reasons = VP.judge_g21(rm.g21())
        red(reasons, needle, f"G21 on the 1 m rig under {envname} ({what})")
    with T.env(**{HB.ENV_P6CTX: "1"}):
        rp = T.Rig(cell=1.0, bw=3.0)
        gp = head_reach(rp)
    red(judge_head_reach(gp), "LoRA tensor holds NO gradient", "module: the 1 m head reads P6's ctx: its queries train, the copy does not")


# ================================================================================================ CALLER
def rows_bad(ns, d):
    return [x for x in C.scorer_dvb_rows(ns, d, {}) if not x["ok"]]


def caller_layer():
    ps = __import__("scorer_finetune").build_parser()
    r = T.smoke("hbc", extra=HBC_ARGS)
    ok(r["rc_pre"] == 0 and r["pass_json"] is not None, f"the HBC preflight PASSES (rc {r['rc_pre']}): {r['out_pre'][-300:]}")
    gts = [x for x in r["rows"]["preflight.jsonl"] if x.get("event") == "preflight"][-1]["gates"]
    ok(gts["G21_vdac_hb_copy_isolation"]["ok"] is True and VP.judge_g21(gts["G21_vdac_hb_copy_isolation"]) == [], "G21 is in the REAL HBC preflight and judged")
    g12 = gts["G12_aux_drv_live_and_identity"]
    ok(g12["scene_proj_calls"] == 3 and g12["tap_index"] == 2 and g12["ok"] is True and g12.get("head_retained_bound_gb") == 4.0, f"G12 reads the copy's ctx (outs[2]) with the 1 m bound 4.0 GB: {g12.get('tap_index'), g12.get('head_retained_bound_gb')}")
    ns = C.parse_scorer_argv(ps, r["argv"] + ["--preflight"])[0]
    ok(ns.aux_drv_trunk_lora == "on" and ns.aux_drv_cell == 1.0 and ns.aux_drv_bw == 3.0 and ns.aux_drv_lora_lr == 1e-3, "the trainer's parser takes HB's and HC's flags together")
    d = W.data_event(r)
    ok((d["aux_drv"]["nx"], d["aux_drv"]["ny"], d["aux_drv"]["cell"], d["aux_drv"]["n_params"]) == (80, 64, 1.0, HEAD_1M) and abs(d["aux_drv"]["bw_probe"]["inside_over_outside"] - 4.0) < 1e-4
       and d["aux_drv"]["n_params"] == d["aux_drv"]["head_params_in_optimiser"], f"SAME DATA EVENT, the head: 80 x 64, 3,423,490, weight 4 / 1: {d['aux_drv']}")
    ok(d["aux_hb"]["trunk_copy_built"] and d["aux_hb"]["n_lora_trainable"] == d["aux_hb"]["lora_params_in_optimiser"] == d["aux_hb"]["lora_expected"] == 4 * (12 + 256) and d["aux_hb"]["trainable_all_lora"], f"SAME DATA EVENT, the copy: {d['aux_hb']}")
    rows = C.scorer_dvb_rows(ns, d, {})
    ok(not rows_bad(ns, d) and {x["dest"] for x in rows} >= {"aux_drv_trunk_lora", "aux_drv_lora_lr", "aux_drv_cell", "aux_drv_bw", "aux_drv_bw_band_m"}, f"G-DVB reads all five levers against ONE build, all OK: {rows_bad(ns, d)}")
    ok("G21" in C.scorer_required_gates(ns), "G21 is required for HBC (the copy)")
    # the run: the copy moved, P6 did not, the 1 m head trained, aux_drv/2
    init = torch.load(os.path.join(os.path.dirname(r["out"]), "ck.pt"), weights_only=False)["model"]
    lt = r["aux"]["hb_lora"]
    ok(r["rc_run"] == 0 and r["aux"]["format"] == "aux_drv/2" and any(not torch.equal(lt[k], init["backbone." + k]) for k in lt), "REACHABLE: the HBC run finishes, writes aux_drv/2 and the copy's LoRA tensors MOVED")
    ok(tuple(r["aux"]["head"]["q"].shape) == (1, 5120, 256), "the saved head is the 1 m head (q is 5,120 x 256)")
    ok(all(torch.equal(r["A"]["model"][k], init[k]) for k in init if k.split(".")[0] not in SCORER) and PG.judge(r["aux"], 25, n_params=HEAD_1M, n_lora=4 * (12 + 256)) == [],
       "P6's trunk tensors are bit-identical; the post-training gate accepts the run's aux_drv.pt at the 1 m head count AND the LoRA count")
    ev = [x for x in r["rows"]["sft.jsonl"] if x.get("event") == "aux_eval"]
    ok(ev and ev[-1]["n_cells"] > 0, "the held-out read runs at the 1 m grid under the copy")
    # the head-less comparator: the scorer arms are bit-identical to an HA run
    ha = T.smoke("ha", extra=W.H1_ARGS)
    ok(all(torch.equal(r["B"]["model"][k], ha["B"]["model"][k]) for k in r["B"]["model"] if k.split(".")[0] in SCORER) and any(not torch.equal(r["aux"]["head"][k], ha["aux"]["head"][k]) for k in r["aux"]["head"] if k in ha["aux"]["head"] and r["aux"]["head"][k].shape == ha["aux"]["head"][k].shape),
       "NO SCORER PATH: an HBC run's scorer arms equal an HA run bit for bit (the copy and the 1 m head cannot reach the scorer)")
    # ---- MUTATIONS, one wiring line each, the OTHER lever still declared and built
    line = "                    actx = HB.select_actx(hb, model, img, cal if cal.numel() else None, sctx, lambda: model.ego_enc(torch.cat([ego, model.cmd_embed(goal.to(ego.dtype))], -1)).sum(-1))\n"
    m1 = T.smoke("hbc_wire", extra=HBC_ARGS, src=T.mutated(line, "                    pass\n"), run=False)
    g21 = [x for x in m1["rows"]["preflight.jsonl"] if x.get("event") == "preflight"][-1]["gates"]["G21_vdac_hb_copy_isolation"]
    red(VP.judge_g21(g21), "NO gradient", "trainer: the select_actx wiring line DELETED (1 m head, copy never trained)")
    ok(m1["rc_pre"] != 0 and m1["pass_json"] is None, "a deleted wiring line withholds PREFLIGHT_PASS.json under HBC")
    cellline = "cell=a.aux_drv_cell, bw=a.aux_drv_bw, bw_band_m=a.aux_drv_bw_band_m, "
    m2 = T.smoke("hbc_cellbw", extra=HBC_ARGS, src=T.mutated(cellline, ""), run=False)
    red([x["flag"] + " " + x["how"] for x in rows_bad(ns, W.data_event(m2))], "--aux-drv-cell", "trainer: cell AND bw dropped from the build while HB is built (declared 1 m, built 2 m)")
    m3 = T.smoke("hbc_bw", extra=HBC_ARGS, src=T.mutated("bw=a.aux_drv_bw, ", ""), run=False)
    d3 = W.data_event(m3)
    red([x["flag"] + " " + x["how"] for x in rows_bad(ns, d3)], "--aux-drv-bw", "trainer: ONLY the boundary weight dropped from the build (1 m head built, copy built, weight 1 / 1)")
    ok(d3["aux_drv"]["n_params"] == HEAD_1M and d3["aux_hb"]["trunk_copy_built"], "control: that mutant still built the 1 m head and the copy (only the weight is missing)")
    m4 = T.smoke("hbc_build", extra=HBC_ARGS, src=T.mutated("            hb = HB.HBTrunk(model).to(dev)\n            optL = HB.build_lora_optimizer(hb, a.aux_drv_lora_lr)\n", "            pass\n"), run=False)
    d4 = W.data_event(m4)
    red([x["flag"] + " " + x["how"] for x in rows_bad(ns, d4)], "aux_hb", "trainer: the copy not built while the 1 m head is (G-DVB aux_drv_trunk_lora)")
    ok(d4["aux_drv"]["n_params"] == HEAD_1M, "control: that mutant still built the 1 m head")
    step_line = "                HB.lora_step(hb, optL, lr_at(upd) * (a.aux_drv_lora_lr / a.lr))\n"
    m5 = T.smoke("hbc_step", extra=HBC_ARGS, src=T.mutated(step_line, "                pass\n"))
    lt5 = m5["aux"]["hb_lora"] if m5["aux"] else {}
    N[0] += 1
    ok(m5["aux"] is not None and all(torch.equal(lt5[k], init["backbone." + k]) for k in lt5), "MUTATION control: with the lora_step line deleted the copy equals P6's, so the 'moved' check above can read a stall")
    TABLE.append(("trainer: HB.lora_step line deleted (HBC)", "RED", "the reachability check (copy moved) fails"))
    for envname, why in ((HB.ENV_P6CTX, "head reads P6's ctx"), (HB.ENV_DETACH, "copy detached")):
        mm = T.smoke("hbc_env_" + envname[-8:], extra=HBC_ARGS, env_={envname: "1"}, run=False)
        N[0] += 1
        ok(mm["rc_pre"] != 0 and mm["pass_json"] is None, f"{envname}: the HBC preflight must FAIL and withhold PREFLIGHT_PASS ({why}): rc {mm['rc_pre']}")
        TABLE.append((f"HBC preflight under {envname} ({why})", "RED", "rc != 0, no PREFLIGHT_PASS.json"))
    # ---- the SHUF arm: the derangement reaches the label lookup (the key map), over ALL store keys, and trains a different head
    sh = T.smoke("hbc_shuf", extra=HBC_ARGS + ["--aux-drv-shuffle-seed", "0"])
    ds = W.data_event(sh)
    ok(ds["aux_drv"]["shuffle_seed"] == 0 and ds["aux_drv"]["derangement_fixed_points"] == 0 and ds["aux_drv"]["shuffle_keys"] == AUX.PINNED_STORE_KEYS and sh["rc_run"] == 0, f"HBC-SHUF: a derangement of ALL {AUX.PINNED_STORE_KEYS} keys, 0 fixed points, trains: {ds['aux_drv'].get('shuffle_seed')}")
    nss = C.parse_scorer_argv(ps, sh["argv"] + ["--preflight"])[0]
    ok(not rows_bad(nss, ds) and any(x["dest"] == "aux_drv_shuffle_seed" for x in C.scorer_dvb_rows(nss, ds, {})), "G-DVB reads the shuffle row against the built key map, all OK")
    ok(any(not torch.equal(sh["aux"]["head"][k], r["aux"]["head"][k]) for k in sh["aux"]["head"]) and sh["aux"]["format"] == "aux_drv/2", "the SHUF arm's head DIFFERS from the real-label arm's (the shuffled labels reach the loss) and is aux_drv/2")
    m6 = T.smoke("hbc_shufmut", extra=HBC_ARGS + ["--aux-drv-shuffle-seed", "0"], src=T.mutated("shuffle_seed=a.aux_drv_shuffle_seed, ", ""), run=False)
    red([x["flag"] + " " + x["how"] for x in rows_bad(nss, W.data_event(m6))], "--aux-drv-shuffle-seed", "trainer: the shuffle seed dropped from the label source (the control would train on TRUE labels)")


def main() -> int:
    launch_layer()
    module_layer()
    caller_layer()
    print(f"{'mutation arm':96s} result  how")
    for what, res, how in TABLE:
        print(f"{what[:96]:96s} {res:7s} {how}")
    print(f"ZZTEST_PASS {N[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
