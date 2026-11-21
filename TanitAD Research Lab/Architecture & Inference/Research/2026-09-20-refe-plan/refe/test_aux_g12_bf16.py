#!/usr/bin/env python3
"""G12's H2 feed probe on the pod's NUMERIC PATH: CUDA, torch.autocast("cuda", bfloat16), the source UNREWRITTEN (Amendment A6).  Needs a CUDA device.
Run:  PYTHONIOENCODING=utf-8 python refe/test_aux_g12_bf16.py   ->   ZZTEST_PASS <n>      (no CUDA: a LOUD skip -- it prints ZZTEST_SKIP_NO_CUDA, never ZZTEST_PASS, exit 0)

WHY THIS FILE EXISTS.  test_aux_s1_wiring.py runs the trainer's real main() on the CPU: it rewrites `dev = "cuda"` to cpu and DISABLES the bf16 autocast.  On the A40 (2026-10-14
21:07Z) the S1 arm H2-s0 failed gate G12_aux_drv_live_and_identity with `feed_probe_reads_difference: false` -- a failure that suite could not see, because the cause is a property
of the autocast region it had switched off:  inside ONE autocast region the first bf16 use of a requires-grad fp32 leaf CACHES the bf16 cast until the outermost region exits, so
an in-place `feed.proj.weight.copy_(1e-2)` made after a call with the zero weight is not seen by the next call in the same region (MEASURED:
raw/2026-10-14-s1-g12-bf16/probe_b_rootcause.json).  The probe re-read the cached zeros, scored exactly the base scorer, and said "the scorer ignores the feed".

SECOND FAILURE, SAME BLIND SPOT (A40, 2026-10-14 23:31Z, arm H1-s0, the head WITHOUT the feed): G12's `reference_replay_identical` and `scorer_params_bit_identical_2steps` read false.  Cause
(MEASURED, raw/2026-10-14-s1-g12-bf16/probe_d_*.json): the identity probe ran its arms under `use_deterministic_algorithms(True, warn_only=True)`, which on CUDA leaves the memory-efficient
SDPA backward (nn.MultiheadAttention at thousands of visual tokens) on its atomics-based NON-deterministic algorithm and only warns -> two identical head-less runs differ in every scorer
tensor.  `warn_only=False` selects the deterministic algorithm: X1 == X2 == Y bit for bit, no tolerance.  The H1 checks below run the real H1 preflight on the GPU with a REAL CrossBlock
scorer (the CPU stub's tanh block has no SDPA and cannot show it).

WHAT IS PROVEN HERE (the REAL main() --preflight of scorer_finetune.py, H2 argv, on this machine's GPU):
  * the fixed probe PASSES on CUDA bf16, and its fields are real: the scorer's output changes (max |diff| > 0, elements differ), on the head's own step-0 field AND on a KNOWN field;
  * it IS the bf16 path: every aux.score call inside the probe ran with CUDA autocast on and returned bf16; the probe really set the fp32 weight to 1e-2; and the weight is RESTORED to
    exactly 0.0 afterwards;
  * the launch gate's consumer accepts the real CUDA H2 preflight (G12 re-judged from raw fields);
  * the OLD probe block, put back into the source, goes RED on CUDA (`feed_probe_reads_difference` False) and stays GREEN on CPU -- the dev-box blind spot reproduced, so a green here is a
    property of the code and not of a blind comparison;
  * the ROOT CAUSE is pinned as a mechanism (the stale cast), and the other candidate causes are refuted: 1e-2 does NOT round away in bf16, and the step-0 feed scalars are NOT degenerate;
  * cutting the feed path (aux.score ignoring the feed) turns G12 RED on CUDA, and the consumer refuses it.
  * H1 (feed off): on CUDA bf16 the head-less replay X1 == X2 and the head arm Y == X1 BIT FOR BIT (no tolerance), the one-ulp control still reads, the consumer accepts; with warn_only=True put
    back the replay breaks (RED) while the CPU run stays green; and a head whose loss DOES reach the scorer reads different (RED) while the replay stays identical.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# TEST-RED-1 (2026-10-17): torch 2.7.1 -- the pod's build and this venv's -- raises its cuBLAS determinism error under STRICT `use_deterministic_algorithms(True)` unless CUBLAS_WORKSPACE_CONFIG is in the
# environment at PROCESS START (a later in-process setdefault is too late for that build: PREREG_AUX_DRIVABLE_S1.md A6 addendum, MEASURED).  The pod gets it from s1aux_pod.sh (the PREFLIGHT command line); without it here every
# H1 check fails (the preflight dies, G12 is never written) -- an ENVIRONMENT failure that read like a code defect.  Set BEFORE `import torch`; setdefault keeps an explicit value.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aux_drivable_head as AUXREAL  # noqa: E402
import test_aux_s1_wiring as W  # noqa: E402  (the stub trunk, the synthetic label store, smoke_aux, gate())
from model import CrossBlock  # noqa: E402

SRC_PATH = HERE / "scorer_finetune.py"
H2 = W.H2_ARGS
H1 = W.H1_ARGS
G12 = "G12_aux_drv_live_and_identity"
N_TOK = 7680                                              # visual tokens per frame in the H1 stub = the pod's count: SDPA then picks its memory-efficient kernel and its atomics show


class Stub256REFeCB(W.Stub256REFe):
    """the 256-wide stub trunk with a REAL model.CrossBlock as the scoring decoder (nn.MultiheadAttention -> SDPA, as on the pod): the tanh stub block cannot show an SDPA effect"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.score_dec = nn.Sequential(CrossBlock(256, 8, 256))


class StubTargetsBig(W.StubTargets256):
    def sample_inputs(self, i):
        g = torch.Generator().manual_seed(1000 + i)
        return torch.randn(N_TOK, 3, generator=g) * (1.0 if i < 100 else 20.0), torch.randn(8, 20, 3, generator=g) * 2.0


def tmp() -> str:
    return tempfile.mkdtemp(prefix="auxg12_", dir=None)


def split_block(src: str):
    """-> (head, new_block, tail): the probe's fixed block (from its A6 comment to just before `armB.zero_grad`), cut by markers that must each occur exactly once"""
    a = src.index("            # ⛔ THE AUTOCAST WEIGHT-CAST CACHE")
    z_marker = "            armB.zero_grad(set_to_none=True)\n            aux.feed.zero_grad(set_to_none=True)\n            losses(sB, sB, tg, a.lam)[1].backward(retain_graph=True)"
    assert src.count(z_marker) == 1 and src.count("# ⛔ THE AUTOCAST WEIGHT-CAST CACHE") == 1, "the probe's markers moved: update split_block"
    z = src.index(z_marker)
    assert a < z
    return src[:a], src[a:z], src[z:]


# the block as it stood before A6 (scorer_finetune.py 1056-1064 at the A40 failure), the weight mutated and re-read INSIDE one autocast region
OLD_BLOCK = '''            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                lg0, sd0 = aux.predict(sctx_d)
                s_plain = armB(cxy_d, sctx_d)
                g["feed_zero_init"] = bool(float(aux.feed.proj.weight.abs().max()) == 0.0)
                g["feed_step0_equal_base"] = bool(torch.equal(aux.score(armB, cxy_d, sctx_d, sd0), s_plain))
                keep_w = aux.feed.proj.weight.detach().clone()
                aux.feed.proj.weight.copy_(torch.full_like(keep_w, 1e-2))
                g["feed_probe_reads_difference"] = bool(not torch.equal(aux.score(armB, cxy_d, sctx_d, sd0), s_plain))
                aux.feed.proj.weight.copy_(keep_w)
'''


# ------------------------------------------------------------------------------------------------ deliberately broken / recording AuxDrivable (the aux module the trainer sees)
REC: dict = {"calls": [], "instances": []}


def patch_record(stub_aux) -> None:
    Base = stub_aux.AuxDrivable

    class Rec(Base):
        def __init__(self, cfg):
            super().__init__(cfg)
            REC["instances"].append(self)

        def score(self, scorer, trajs, sctx, sdf_pred=None):
            out = super().score(scorer, trajs, sctx, sdf_pred)
            try:
                ac = bool(torch.is_autocast_enabled("cuda"))
            except TypeError:                                                   # torch < 2.4
                ac = bool(torch.is_autocast_enabled())
            REC["calls"].append({"autocast": ac, "w_max": float(self.feed.proj.weight.detach().abs().max()) if self.feed is not None else None,
                                 "dtype": str(out.dtype), "cuda": bool(out.is_cuda)})
            return out
    stub_aux.AuxDrivable = Rec


def patch_cut(stub_aux) -> None:
    Base = stub_aux.AuxDrivable

    class Cut(Base):
        def score(self, scorer, trajs, sctx, sdf_pred=None):
            return scorer(trajs, sctx)                                          # the scorer's output ignores the feed entirely
    stub_aux.AuxDrivable = Cut


def pre_cuda(src: str, tag: str, patch=None, cuda=True, extra=None, big=False) -> dict:
    kw = {"refe_cls": Stub256REFeCB, "targets_cls": StubTargetsBig} if big else {}
    return W.smoke_aux(src, tmp(), tag=tag, extra=H2 if extra is None else extra, preflight=True, run=False, cuda=cuda, aux_patch=patch, **kw)


def pre_h1(src: str, tag: str, cuda=True) -> dict:
    return pre_cuda(src, tag, cuda=cuda, extra=H1, big=True)


def consume(res: dict, arm="H2-s0") -> dict:
    import launch_gate as LG
    return LG.consume_scorer_preflight(Path(res["out"]) / "preflight.jsonl", res["argv"] + ["--preflight"], {**LG.PROFILES["refe_aux_s1"], "declared_arm": arm})


# ------------------------------------------------------------------------------------------------ the mechanism, without the trainer
def stale_cast_mechanism() -> dict:
    """the real DrivableFeed on CUDA: one autocast region, a zero weight used once, the fp32 weight then set to 1e-2 in place"""
    feed = AUXREAL.DrivableFeed(256, "zero", 0).cuda()
    sdf = (torch.randn(2, 40, 32) * 2.0).cuda()
    t = torch.arange(1, 21, dtype=torch.float32) * 0.2
    trajs = torch.stack([(8.0 * t).expand(2, 8, 20), torch.zeros(2, 8, 20), torch.zeros(2, 8, 20)], -1).cuda()
    out: dict = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out["w0"] = float(feed(sdf, trajs).float().abs().max())
        feed.proj.weight.copy_(torch.full_like(feed.proj.weight, 1e-2))
        out["same_region_after_copy"] = float(feed(sdf, trajs).float().abs().max())
        torch.clear_autocast_cache()
        out["same_region_after_clear"] = float(feed(sdf, trajs).float().abs().max())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out["fresh_region"] = float(feed(sdf, trajs).float().abs().max())
    out["fp32_weight_is_1e-2"] = bool(torch.allclose(feed.proj.weight, torch.full_like(feed.proj.weight, 1e-2)))
    return out


def sdpa_backward_determinism() -> dict:
    """the real CrossBlock (nn.MultiheadAttention -> SDPA) at pod-like token counts, bf16 autocast, backward 3 times per mode: do the gradients repeat bit for bit?"""
    import warnings
    torch.manual_seed(0)
    blk = CrossBlock(256, 8, 256).cuda()
    g = torch.Generator().manual_seed(1)
    q, ctx = torch.randn(2, 64, 256, generator=g).cuda(), torch.randn(2, 7680, 256, generator=g).cuda()
    out: dict = {}
    was, was_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        for mode, warn_only in (("warn_only", True), ("strict", False)):
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            gs = []
            for _ in range(3):
                blk.zero_grad(set_to_none=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        y = blk(q, ctx)
                    y.float().square().mean().backward()
                gs.append([p_.grad.detach().clone() for p_ in blk.parameters()])
            out[mode + "_all_runs_identical"] = all(torch.equal(a_, b_) for gg in gs[1:] for a_, b_ in zip(gs[0], gg))
    finally:
        torch.use_deterministic_algorithms(was, warn_only=was_warn)
    return out


def step0_scalars_on_cuda() -> dict:
    """the real head at step 0 (random init, seeded) on CUDA under bf16: the three feed scalars of a set of forward proposals are not degenerate"""
    aux = AUXREAL.AuxDrivable(AUXREAL.AuxDrivableConfig(feed="on", w_aux=0.1, seed=0)).cuda()
    g = torch.Generator().manual_seed(0)
    t = torch.arange(1, 21, dtype=torch.float32) * 0.2
    v, lat = 6.0 + 8.0 * torch.rand(2, 64, 1, generator=g), (torch.rand(2, 64, 1, generator=g) - 0.5) * 0.6
    trajs = torch.stack([(v * t).expand(2, 64, 20), (lat * t ** 2 / 2).expand(2, 64, 20), torch.atan2(lat * t, v).expand(2, 64, 20)], -1).cuda()
    ctx = torch.randn(2, 512, 256, generator=g).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, sd0 = aux.predict(ctx)
    feats, _ = AUXREAL.feed_scalars(sd0, trajs)
    rows = feats.reshape(-1, 3)
    return {"std_over_proposals": feats.std(1).mean(0).tolist(), "distinct_rows": int(torch.unique(rows, dim=0).shape[0]), "rows": int(rows.shape[0]),
            "abs_sum_per_row_min": float(rows.sum(-1).abs().min())}


def contribution_survives_bf16() -> dict:
    """1e-2 through the real feed into the real first layer of a scorer-shaped MLP, each read in its own region: how many bf16 elements change"""
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(60, 256), nn.GELU(), nn.Linear(256, 256)).cuda()
    for p in mlp.parameters():
        p.requires_grad_(True)
    feed = AUXREAL.DrivableFeed(256, "zero", 0).cuda()
    sdf = (torch.randn(2, 40, 32) * 2.0).cuda()
    t = torch.arange(1, 21, dtype=torch.float32) * 0.2
    trajs = torch.stack([(10.0 * t).expand(2, 8, 20), torch.zeros(2, 8, 20), torch.zeros(2, 8, 20)], -1).cuda()
    x = trajs.flatten(2)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h0 = mlp[0](x) + feed(sdf, trajs)
        feed.proj.weight.copy_(torch.full_like(feed.proj.weight, 1e-2))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h1 = mlp[0](x) + feed(sdf, trajs)
    return {"elems": int(h0.numel()), "elems_differ": int((h0 != h1).sum()), "dtype": str(h1.dtype)}


# ------------------------------------------------------------------------------------------------ the suite
N_CHECKS = [0]


def suite(src: str, only=None) -> list:
    bad: list = []

    def t(name, fn):
        if only and not any(o in name for o in only):
            return
        if not only:
            N_CHECKS[0] += 1
        try:
            ok = bool(fn())
        except Exception as e:                                                       # noqa: BLE001 -- an exception IS a failed check
            ok = False
            print(f"   [{name}] raised {type(e).__name__}: {str(e)[:300]}")
        if not ok:
            bad.append(name)
    memo: dict = {}

    def mem(k, fn):
        if k not in memo:
            memo[k] = fn()
        return memo[k]

    def real():
        REC["calls"].clear()
        REC["instances"].clear()
        res = pre_cuda(src, "real", patch_record)
        res["calls"], res["inst"] = list(REC["calls"]), list(REC["instances"])
        return res
    r_ok = lambda: mem("real", real)                                                  # noqa: E731
    g_ = lambda: W.gate(r_ok(), G12)                                                  # noqa: E731

    # ---- the fixed probe on the pod's numeric path
    t("cuda/preflight_ran_to_a_G12_on_the_GPU", lambda: g_() is not None and g_().get("feed") == "on" and r_ok()["out_exists"])
    t("cuda/G12_feed_fields_true", lambda: g_()["feed_zero_init"] is True and g_()["feed_step0_equal_base"] is True and g_()["feed_probe_reads_difference"] is True and g_()["feed_proj_grad_abs"] > 0.0)
    t("cuda/G12_ok", lambda: g_()["ok"] is True)
    t("cuda/probe_reads_a_nonzero_difference", lambda: g_()["feed_probe_max_abs_diff"] > 0.0 and 0 < g_()["feed_probe_elems_differ"] <= g_()["feed_probe_elems"])
    t("cuda/probe_reads_a_KNOWN_field_too", lambda: g_()["feed_probe_known_field_reads_difference"] is True)
    t("cuda/step0_scalars_are_not_degenerate_in_the_probe", lambda: all(v > 0.0 for v in g_()["feed_probe_scalar_std_over_proposals"][:2]))
    t("cuda/the_probe_ran_on_the_bf16_autocast_path", lambda: (lambda c: len([x for x in c if abs((x["w_max"] or 0.0) - 1e-2) < 1e-8]) >= 2 and all(
        x["autocast"] and x["dtype"] == "torch.bfloat16" and x["cuda"] for x in c if abs((x["w_max"] or 0.0) - 1e-2) < 1e-8))(r_ok()["calls"]))
    t("cuda/probe_sets_the_fp32_weight_to_1e-2_and_RESTORES_exactly_zero", lambda: any(abs((x["w_max"] or 0.0) - 1e-2) < 1e-8 for x in r_ok()["calls"]) and all(
        float(i.feed.proj.weight.detach().abs().max()) == 0.0 for i in r_ok()["inst"]) and len(r_ok()["inst"]) == 1)
    t("cuda/launch_gate_consumer_accepts_the_real_CUDA_H2_preflight", lambda: consume(r_ok())["status"] == "PASS")

    # ---- the cause, as a mechanism (and the refuted candidates)
    t("cause/stale_bf16_cast_inside_one_autocast_region", lambda: (lambda m: m["w0"] == 0.0 and m["same_region_after_copy"] == 0.0 and m["same_region_after_clear"] > 0.0
                                                                   and m["fresh_region"] > 0.0 and m["fp32_weight_is_1e-2"])(mem("mech", stale_cast_mechanism)))
    t("cause/1e-2_does_not_round_away_in_bf16", lambda: (lambda c: c["dtype"] == "torch.bfloat16" and c["elems_differ"] > c["elems"] // 2)(mem("surv", contribution_survives_bf16)))
    t("cause/step0_feed_scalars_not_degenerate", lambda: (lambda s: s["distinct_rows"] == s["rows"] and min(s["std_over_proposals"][:2]) > 0.0 and s["abs_sum_per_row_min"] > 0.0)(mem("sc", step0_scalars_on_cuda)))

    # ---- the old probe: RED on CUDA, GREEN on CPU (the blind spot), and the fixed one is not the old one
    def old_src():
        h, nb, tl = split_block(src)
        return h + OLD_BLOCK + tl
    t("old/block_is_really_different_from_the_fixed_one", lambda: OLD_BLOCK not in src and "torch.clear_autocast_cache()" in split_block(src)[1])
    t("old/RED_on_CUDA_feed_probe_reads_difference_false", lambda: (lambda g: g is not None and g["feed_probe_reads_difference"] is False and g["feed_zero_init"] is True
                                                                    and g["feed_step0_equal_base"] is True and g["ok"] is False)(W.gate(mem("old_cuda", lambda: pre_cuda(old_src(), "old_cuda")), G12)))
    t("old/GREEN_on_CPU_the_dev_box_blind_spot", lambda: (lambda g: g is not None and g["feed_probe_reads_difference"] is True and g["ok"] is True)(
        W.gate(mem("old_cpu", lambda: pre_cuda(old_src(), "old_cpu", cuda=False)), G12)))
    t("old/consumer_refuses_the_old_probes_CUDA_preflight", lambda: consume(mem("old_cuda", lambda: pre_cuda(old_src(), "old_cuda")))["status"] == "FAIL")

    # ---- the feed path cut
    def cut():
        return pre_cuda(src, "cut", patch_cut)
    t("cut/RED_when_aux_score_ignores_the_feed", lambda: (lambda g: g is not None and g["feed_probe_reads_difference"] is False and g["feed_probe_known_field_reads_difference"] is False
                                                          and g["feed_proj_grad_abs"] == 0.0 and g["ok"] is False)(W.gate(mem("cut", cut), G12)))
    t("cut/consumer_refuses", lambda: consume(mem("cut", cut))["status"] == "FAIL")

    # ---- H1 (feed off): the identity probe on the pod's numeric path
    WARN = ("torch.use_deterministic_algorithms(True, warn_only=False)", "torch.use_deterministic_algorithms(True, warn_only=True)")
    LEAK = ("lg_, sd_ = H.predict(sctx_d)\n", "lg_, sd_ = H.predict(sctx_d + S.score_head.bias.sum())\n")           # the head's loss now reaches the scorer's own parameters through the context it reads (a leaf: no second backward through sx's freed graph)

    def h1_real():
        return mem("h1", lambda: pre_h1(src, "h1"))
    gh = lambda: W.gate(h1_real(), G12)                                              # noqa: E731
    t("h1cuda/preflight_ran_to_a_G12_with_no_feed", lambda: gh() is not None and gh().get("feed") == "off" and "reference_replay_identical" in gh())
    t("h1cuda/replay_X1_equals_X2_bit_for_bit", lambda: gh()["reference_replay_identical"] is True)
    t("h1cuda/head_leaves_scorer_bit_identical_2steps", lambda: gh()["scorer_params_bit_identical_2steps"] is True)
    t("h1cuda/one_ulp_control_still_reads", lambda: gh()["identity_probe_reads_one_ulp"] is True)
    t("h1cuda/G12_ok", lambda: gh()["ok"] is True and gh()["adam_head_params"] > 0 and gh()["adam_head_params_with_state"] == gh()["adam_head_params"])
    t("h1cuda/launch_gate_consumer_accepts_the_real_CUDA_H1_preflight", lambda: consume(h1_real(), "H1-s0")["status"] == "PASS")
    t("cause/sdpa_backward_is_nondeterministic_under_warn_only_and_repeats_under_strict", lambda: (lambda d: d["warn_only_all_runs_identical"] is False and d["strict_all_runs_identical"] is True)(
        mem("sdpa", sdpa_backward_determinism)))

    def warn_src():
        assert src.count(WARN[0]) == 1
        return src.replace(*WARN)

    def leak_src():
        assert src.count(LEAK[0]) == 1, "the H1 head forward moved: update LEAK"
        return src.replace(*LEAK)
    def warn_run():
        """warn_only=True is non-deterministic BY CONSTRUCTION, so a single run can be lucky (X1 == X2 happens); take the first of <= 3 runs in which G12 reads RED"""
        def go():
            for k in range(3):
                r = pre_h1(warn_src(), f"h1w{k}")
                g = W.gate(r, G12)
                if g is not None and not g["ok"]:
                    return r
            return r
        return mem("h1w", go)
    t("h1old/RED_on_CUDA_warn_only_breaks_the_identity_probe", lambda: (lambda g: g is not None and g["ok"] is False and g["identity_probe_reads_one_ulp"] is True and (
        g["reference_replay_identical"] is False or g["scorer_params_bit_identical_2steps"] is False))(W.gate(warn_run(), G12)))
    t("h1old/consumer_refuses_the_warn_only_CUDA_preflight", lambda: consume(warn_run(), "H1-s0")["status"] == "FAIL")
    t("h1old/GREEN_on_CPU_the_dev_box_blind_spot", lambda: (lambda g: g is not None and g["reference_replay_identical"] is True and g["scorer_params_bit_identical_2steps"] is True and g["ok"] is True)(
        W.gate(mem("h1wc", lambda: pre_h1(warn_src(), "h1wc", cuda=False)), G12)))
    t("h1leak/RED_a_head_that_reaches_the_scorer_reads_different_while_the_replay_stays_identical", lambda: (lambda g: g is not None and g["reference_replay_identical"] is True
                                                                                                         and g["scorer_params_bit_identical_2steps"] is False and g["ok"] is False)(
        W.gate(mem("h1l", lambda: pre_h1(leak_src(), "h1l")), G12)))
    t("h1leak/consumer_refuses", lambda: consume(mem("h1l", lambda: pre_h1(leak_src(), "h1l")), "H1-s0")["status"] == "FAIL")
    return bad


# the fixed probe is broken IN THE SOURCE (the snippet must occur exactly once) and the check that owns it must go RED
MUTATIONS = [
    ("probe_never_restores_the_weight", "                    aux.feed.proj.weight.copy_(keep_w)                 # restored, whatever happened", "                    pass",
     ("cuda/probe_sets_the_fp32_weight_to_1e-2_and_RESTORES_exactly_zero",)),
    ("probe_never_sets_the_weight", "                    aux.feed.proj.weight.copy_(torch.full_like(keep_w, 1e-2))\n                torch.clear_autocast_cache()",
     "                    pass\n                torch.clear_autocast_cache()", ("cuda/G12_feed_fields_true", "cuda/probe_reads_a_nonzero_difference")),
    ("h1_probe_warn_only_again", "torch.use_deterministic_algorithms(True, warn_only=False)", "torch.use_deterministic_algorithms(True, warn_only=True)",
     ("h1cuda/G12_ok", "h1cuda/head_leaves_scorer_bit_identical_2steps"), "stochastic"),       # a non-deterministic kernel can be lucky on one run (X1 == X2 == Y): up to 5 tries
    ("h1_head_loss_reaches_the_scorer", "lg_, sd_ = H.predict(sctx_d)\n", "lg_, sd_ = H.predict(sctx_d + S.score_head.bias.sum())\n", ("h1cuda/head_leaves_scorer_bit_identical_2steps",)),
    ("h1_identity_verdict_is_a_constant_True", 'g.update({"reference_replay_identical": bool(same(X1, X2)), "scorer_params_bit_identical_2steps": bool(same(X1, Y)),',
     'g.update({"reference_replay_identical": bool(same(X1, X2)), "scorer_params_bit_identical_2steps": True,', ("h1leak/RED_a_head_that_reaches_the_scorer_reads_different_while_the_replay_stays_identical",)),
    ("probe_verdict_is_a_constant_True", 'g["feed_probe_reads_difference"] = bool(not torch.equal(s_fed, s_plain))', 'g["feed_probe_reads_difference"] = True',
     ("cut/RED_when_aux_score_ignores_the_feed",)),
]


def main() -> int:
    if not torch.cuda.is_available():
        print("=" * 150)
        print("ZZTEST_SKIP_NO_CUDA: no CUDA device -- the G12 feed probe was NOT exercised on the bf16 autocast path.  This is NOT a pass.")
        print("The CPU suite (test_aux_s1_wiring.py) disables autocast and cannot see the A40 failure (PREREG_AUX_DRIVABLE_S1.md A6).  Run this file on a machine with a GPU.")
        print("=" * 150)
        return 0
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    src = SRC_PATH.read_text(encoding="utf-8")
    real = suite(src)
    assert not real, f"the G12 bf16 checks FAIL on the real source: {real}"
    n = 1
    print("unmutated suite green", flush=True)
    print(f"{'mutation':46s} caught-by (failing checks)")
    for name, old, new, owners, *opt in MUTATIONS:
        cnt = src.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times (must be exactly 1) -- a vacuous mutation proves nothing"
        msrc = src.replace(old, new)
        compile(msrc, str(SRC_PATH), "exec")
        failing = []
        for _try in range(5 if opt == ["stochastic"] else 1):
            failing = suite(msrc, only=owners)
            if failing:
                break
        assert failing, f"mutation {name} stayed GREEN on {owners}: the suite cannot see it"
        print(f"{name:46s} {', '.join(failing[:3])}", flush=True)
        n += 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite of {N_CHECKS[0]} checks + {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
