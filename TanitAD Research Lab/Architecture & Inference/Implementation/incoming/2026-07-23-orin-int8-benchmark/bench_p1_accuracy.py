#!/usr/bin/env python3
"""Phase 1 (PyTorch, real weights): per-block INT8 accuracy sensitivity sweep +
downstream rollout ADE proxy, for the mandated FP16-vs-INT8 encoder benchmark.

Deployed flagship-v1 = flagship4b, action_dim=3, ckpt step 29999
(Sayood/tanitad-flagship-4b-speedjerk). Run on pod1 (RTX A6000, SM 8.6).

Data: physicalai-train-e438721ae894 (the parity-locked TRAIN corpus, on pod1
already). NOT the canonical taniteval val set (that lives on tanitad-eval,
off-limits to this stream) -- every accuracy number here is explicitly labeled
a TRAIN-CACHE PROXY, never conflated with registry ADE.

Method:
  - Weight-only INT8: per-output-channel symmetric fake-quant of every
    Linear/Conv2d/MHA weight inside a target block (dequantized back to fp32
    -- simulates a weight-only INT8 GEMM's numerical effect without needing a
    real TRT engine per block).
  - INT8 W+A: weight fake-quant as above + a per-tensor dynamic fake-quant on
    that block's internal activations via forward-pre-hooks.
  - FP16: real half-precision arithmetic (blanket only -- see note in report).
  - ISOLATED sweep: quantize ONE block, leave every other block fp32 -- this
    is what actually attributes error to a specific block (a blanket sweep
    cannot distinguish "block 3 is bad" from "block 9 is bad").
  - BLANKET sweep: quantize ALL blocks of a graph at once (the "naive INT8
    ViT" configuration the trap warns about).
  - Downstream proxy: full 20-step rollout_decode + grounding.step['op']
    (kept fp32, per the deployment recipe) -> SE(2) waypoints -> ADE vs GT,
    on a held-out 40-episode slice disjoint from calibration/sweep episodes.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

STACK = "/workspace/int8_bench/stack_clean"
SCRIPTS = "/workspace/int8_bench/stack_clean_scripts"
sys.path.insert(0, STACK)
sys.path.insert(0, SCRIPTS)

from tanitad.config import flagship4b_config                       # noqa: E402
from tanitad.eval.ckpt_compat import (build_world_from_ckpt,        # noqa: E402
                                      state_dict_of, append_speed_channel,
                                      SPEED_SCALE)
from tanitad.data.mixing import load_episode                        # noqa: E402
from tanitad.models.metric_dynamics import (HierarchicalGrounding,   # noqa: E402
                                            rollout_decode, accumulate_se2)
from tanitad.instruments.numerics import strict_numerics            # noqa: E402
from driving_diagnostic import (WP_STEPS, gt_ego_waypoints,         # noqa: E402
                                baseline_waypoints, de_of, scalar_metrics)

DEVICE = "cuda"
CKPT = "/workspace/int8_bench/ckpt/ckpt.pt"
EPCACHE = Path("/workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894")
OUT = Path("/workspace/int8_bench/orin_int8_benchmark.json")
WINDOW = 8
K_MAX = 20
STRIDE = 8
CALIB_EP_RANGE = range(0, 150)          # calibration / block-sweep episodes
PROXY_EP_RANGE = range(2000, 2040)      # disjoint 40-ep slice for the ADE proxy
N_ACC_SAMPLES = 96                       # frames / (state,action) pairs for the accuracy sweep


def log(msg):
    print(f"[bench-p1] {msg}", flush=True)


def merge_report(key, val):
    rep = json.loads(OUT.read_text()) if OUT.exists() else {}
    rep[key] = val
    rep["_last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.write_text(json.dumps(rep, indent=2, default=str))
    log(f"banked report key={key!r} -> {OUT}")


# --------------------------------------------------------------------------- #
# Model + grounding load (STRICT, real weights)                               #
# --------------------------------------------------------------------------- #
def load_world_and_grounding():
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    cfg = flagship4b_config()
    world, speed_input, src = build_world_from_ckpt(cfg, ck, ckpt_path=CKPT)
    world = world.to(DEVICE).eval()
    grounding = HierarchicalGrounding(world.state_dim).to(DEVICE).eval()
    missing, unexpected = grounding.load_state_dict(ck["grounding"], strict=False)
    step = int(ck.get("step", -1))
    n_params = sum(p.numel() for p in world.parameters())
    meta = {
        "ckpt": CKPT, "ckpt_step": step, "action_dim_source": src,
        "speed_input": speed_input, "world_params": n_params,
        "registry_total_model_expected": 263442838,
        "matches_registry_exactly": n_params == 263442838,
        "grounding_missing_keys": list(missing), "grounding_unexpected_keys": list(unexpected),
        "note": ("built via tanitad.eval.ckpt_compat.build_world_from_ckpt "
                 "(dataclasses.replace on BOTH predictor.action_dim and "
                 "tactical_pred.action_dim before construction, then strict-load) "
                 "-- this is the reproducible path; a naive `cfg.predictor."
                 "action_dim = 3` post-hoc mutation (as in the 2026-07-22 "
                 "export_and_bench.py) under-widens tactical_pred.act_emb by "
                 "512 params (263,442,326 vs the registry's 263,442,838) "
                 "because it never touches tactical_pred -- MEASURED here, a "
                 "correction to the prior intake, harmless to its actual "
                 "exported encoder/predictor graphs since neither touches "
                 "tactical_pred."),
    }
    log(f"world params={n_params} (registry={263442838}, match={meta['matches_registry_exactly']})")
    log(f"grounding load: missing={missing} unexpected={unexpected}")
    step_readout = grounding.step["op"]
    return world, grounding, step_readout, meta


# --------------------------------------------------------------------------- #
# Data harvesting from the real, parity-locked TRAIN cache                    #
# --------------------------------------------------------------------------- #
def _episode_files():
    return sorted(EPCACHE.glob("ep_*.pt"))


@torch.no_grad()
def harvest_encoder_frames(ep_range, n=N_ACC_SAMPLES):
    files = _episode_files()
    chosen = [files[i] for i in ep_range if i < len(files)]
    frames = []
    per_ep = max(1, n // max(1, len(chosen)))
    for f in chosen:
        ep = load_episode(str(f), mmap=True)
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 else ep.frames
        T = fr.shape[0]
        idx = torch.linspace(0, T - 1, steps=min(per_ep, T)).long()
        for i in idx.tolist():
            frames.append(fr[i])
        if len(frames) >= n:
            break
    return torch.stack(frames[:n]).to(DEVICE)


@torch.no_grad()
def harvest_predictor_pairs(world, ep_range, n=N_ACC_SAMPLES):
    files = _episode_files()
    chosen = [files[i] for i in ep_range if i < len(files)]
    states_list, actions_list = [], []
    per_ep = max(1, n // max(1, len(chosen)))
    for f in chosen:
        ep = load_episode(str(f), mmap=True)
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 else ep.frames
        T = fr.shape[0]
        starts = list(range(0, T - WINDOW - K_MAX, STRIDE))
        if not starts:
            continue
        for t in starts[:per_ep]:
            fw = fr[t:t + WINDOW].to(DEVICE)
            aw = ep.actions[t:t + WINDOW].to(DEVICE)
            last = t + WINDOW - 1
            v0 = (ep.poses[last, 3:4] / SPEED_SCALE).to(DEVICE).unsqueeze(0)
            aw3 = append_speed_channel(aw.unsqueeze(0), v0)[0]
            st = world.encode_window(fw.unsqueeze(0))[0]
            states_list.append(st)
            actions_list.append(aw3)
        if len(states_list) >= n:
            break
    return torch.stack(states_list[:n]).to(DEVICE), torch.stack(actions_list[:n]).to(DEVICE)


@torch.no_grad()
def harvest_proxy_episodes(ep_range):
    files = _episode_files()
    return [(i, load_episode(str(files[i]), mmap=True)) for i in ep_range if i < len(files)]


# --------------------------------------------------------------------------- #
# Fake-quantization utilities (weight-only + weight+activation)               #
# --------------------------------------------------------------------------- #
def _fakequant_(w: Tensor, bits=8, dim=0):
    with torch.no_grad():
        orig_dtype = w.dtype
        wf = w.data.float()
        other = [d for d in range(wf.dim()) if d != dim]
        amax = (wf.abs().amax(dim=other, keepdim=True).clamp_min(1e-8)
                if other else wf.abs().amax().clamp_min(1e-8))
        qmax = 2 ** (bits - 1) - 1
        scale = amax / qmax
        wq = torch.clamp(torch.round(wf / scale), -qmax - 1, qmax)
        w.data.copy_((wq * scale).to(orig_dtype))


def quantize_weights_(module: nn.Module, bits=8) -> int:
    n = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            _fakequant_(m.weight, bits=bits, dim=0)
            n += 1
        elif isinstance(m, nn.Conv2d):
            _fakequant_(m.weight, bits=bits, dim=0)
            n += 1
        elif isinstance(m, nn.MultiheadAttention):
            if m.in_proj_weight is not None:
                _fakequant_(m.in_proj_weight, bits=bits, dim=0)
                n += 1
            if m.out_proj.weight is not None:
                _fakequant_(m.out_proj.weight, bits=bits, dim=0)
                n += 1
    return n


def _act_quant_hook(mod, inputs):
    x = inputs[0]
    if not (torch.is_tensor(x) and x.is_floating_point()):
        return None
    with torch.no_grad():
        qmax = 127
        amax = x.abs().amax().clamp_min(1e-8)
        scale = amax / qmax
        xq = torch.clamp(torch.round(x / scale), -qmax - 1, qmax) * scale
    return (xq.to(x.dtype),) + tuple(inputs[1:])


def add_activation_quant_(module: nn.Module) -> list:
    handles = []
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
            handles.append(m.register_forward_pre_hook(_act_quant_hook))
    return handles


def cosine_mse(a: Tensor, b: Tensor) -> dict:
    af, bf = a.float().reshape(a.shape[0], -1), b.float().reshape(b.shape[0], -1)
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=1).mean().item()
    mse = (af - bf).pow(2).mean().item()
    rel_l2 = ((af - bf).norm(dim=1) / bf.norm(dim=1).clamp_min(1e-8)).mean().item()
    return {"cosine": cos, "mse": mse, "rel_l2": rel_l2}


# --------------------------------------------------------------------------- #
# Block enumeration                                                           #
# --------------------------------------------------------------------------- #
def encoder_block_names(n_enc_blocks=12):
    return ["patch_embed"] + [f"enc_block_{i}" for i in range(n_enc_blocks)] + ["readout_head"]


def predictor_block_names(n_pred_blocks=10):
    return ["in_proj", "act_emb"] + [f"pred_block_{i}" for i in range(n_pred_blocks)] + ["pred_heads"]


def get_encoder_submodule(enc, readout, name):
    if name == "patch_embed":
        return enc.patch
    if name.startswith("enc_block_"):
        return enc.blocks[int(name.rsplit("_", 1)[1])]
    if name == "readout_head":
        return readout
    raise KeyError(name)


def get_predictor_submodule(pred, name):
    if name == "in_proj":
        return pred.in_proj
    if name == "act_emb":
        return pred.act_emb
    if name.startswith("pred_block_"):
        return pred.blocks[int(name.rsplit("_", 1)[1])]
    if name == "pred_heads":
        return pred.heads
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# Encoder accuracy sweep (isolated + blanket)                                 #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encoder_sweep(world, frames_real):
    enc_ref = copy.deepcopy(world.encoder).to(DEVICE).eval()
    readout_ref = copy.deepcopy(world.readout).to(DEVICE).eval()
    ref_state = readout_ref(enc_ref(frames_real))

    names = encoder_block_names(len(world.encoder.blocks))
    isolated = {}
    for name in names:
        isolated[name] = {}
        for mode in ("int8_wo", "int8_wa"):
            enc_c = copy.deepcopy(world.encoder).to(DEVICE).eval()
            readout_c = copy.deepcopy(world.readout).to(DEVICE).eval()
            target = get_encoder_submodule(enc_c, readout_c, name)
            n_q = quantize_weights_(target, bits=8)
            handles = add_activation_quant_(target) if mode == "int8_wa" else []
            out_state = readout_c(enc_c(frames_real))
            for h in handles:
                h.remove()
            isolated[name][mode] = {**cosine_mse(ref_state, out_state), "n_weight_tensors_quantized": n_q}
        log(f"encoder isolated sweep: {name} done "
            f"(int8_wo cos={isolated[name]['int8_wo']['cosine']:.5f}, "
            f"int8_wa cos={isolated[name]['int8_wa']['cosine']:.5f})")

    # Blanket: every block at once
    blanket = {}
    for mode in ("fp16", "int8_wo", "int8_wa"):
        enc_c = copy.deepcopy(world.encoder).to(DEVICE).eval()
        readout_c = copy.deepcopy(world.readout).to(DEVICE).eval()
        if mode == "fp16":
            enc_c, readout_c = enc_c.half(), readout_c.half()
            out_state = readout_c(enc_c(frames_real.half())).float()
        else:
            quantize_weights_(enc_c, bits=8)
            quantize_weights_(readout_c, bits=8)
            handles = (add_activation_quant_(enc_c) + add_activation_quant_(readout_c)
                       if mode == "int8_wa" else [])
            out_state = readout_c(enc_c(frames_real))
            for h in handles:
                h.remove()
        blanket[mode] = cosine_mse(ref_state, out_state)
        log(f"encoder blanket sweep: {mode} cos={blanket[mode]['cosine']:.5f} mse={blanket[mode]['mse']:.3e}")

    return {"n_samples": frames_real.shape[0], "isolated_per_block": isolated, "blanket": blanket}


# --------------------------------------------------------------------------- #
# Predictor accuracy sweep (isolated + blanket) -- 1-step head, k=1           #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predictor_sweep(world, states_real, actions_real):
    pred_ref = copy.deepcopy(world.predictor).to(DEVICE).eval()
    ref_out = pred_ref(states_real, actions_real)[1]

    names = predictor_block_names(len(world.predictor.blocks))
    isolated = {}
    for name in names:
        isolated[name] = {}
        for mode in ("int8_wo", "int8_wa"):
            pred_c = copy.deepcopy(world.predictor).to(DEVICE).eval()
            target = get_predictor_submodule(pred_c, name)
            n_q = quantize_weights_(target, bits=8)
            handles = add_activation_quant_(target) if mode == "int8_wa" else []
            out = pred_c(states_real, actions_real)[1]
            for h in handles:
                h.remove()
            isolated[name][mode] = {**cosine_mse(ref_out, out), "n_weight_tensors_quantized": n_q}
        log(f"predictor isolated sweep: {name} done "
            f"(int8_wo cos={isolated[name]['int8_wo']['cosine']:.5f}, "
            f"int8_wa cos={isolated[name]['int8_wa']['cosine']:.5f})")

    blanket = {}
    for mode in ("fp16", "int8_wo", "int8_wa"):
        pred_c = copy.deepcopy(world.predictor).to(DEVICE).eval()
        if mode == "fp16":
            pred_c = pred_c.half()
            out = pred_c(states_real.half(), actions_real.half())[1].float()
        else:
            quantize_weights_(pred_c, bits=8)
            handles = add_activation_quant_(pred_c) if mode == "int8_wa" else []
            out = pred_c(states_real, actions_real)[1]
            for h in handles:
                h.remove()
        blanket[mode] = cosine_mse(ref_out, out)
        log(f"predictor blanket sweep: {mode} cos={blanket[mode]['cosine']:.5f} mse={blanket[mode]['mse']:.3e}")

    return {"n_samples": states_real.shape[0], "isolated_per_block": isolated, "blanket": blanket}


# --------------------------------------------------------------------------- #
# Downstream proxy: full 20-step rollout ADE, TRAIN-CACHE held-out episodes   #
# --------------------------------------------------------------------------- #
class _FP16PredictorAdapter:
    def __init__(self, predictor_module):
        self.predictor = predictor_module.half()

    def __call__(self, states, actions):
        out = self.predictor(states.half(), actions.half())
        return {k: v.float() for k, v in out.items()}


def _make_predictor_variant(world, mode):
    if mode == "fp32":
        return copy.deepcopy(world.predictor).to(DEVICE).eval()
    if mode == "fp16":
        return _FP16PredictorAdapter(copy.deepcopy(world.predictor).to(DEVICE).eval())
    pred_c = copy.deepcopy(world.predictor).to(DEVICE).eval()
    quantize_weights_(pred_c, bits=8)
    if mode == "int8_wa":
        add_activation_quant_(pred_c)          # left attached deliberately (eval-only object)
    elif mode != "int8_wo":
        raise ValueError(mode)
    return pred_c


@torch.no_grad()
def downstream_ade_proxy(world, step_readout, episodes):
    wp_idx = torch.tensor([k - 1 for k in WP_STEPS])
    modes = ("fp32", "fp16", "int8_wo", "int8_wa")
    per_mode = {m: {"pred": [], "gt": [], "cv": [], "step_dpose": []} for m in modes}
    n_windows = 0
    for _eid, ep in episodes:
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 else ep.frames
        T = fr.shape[0]
        starts = list(range(0, T - WINDOW - K_MAX, STRIDE))
        for t in starts:
            last_t = torch.tensor([t + WINDOW - 1])
            fw = fr[t:t + WINDOW].unsqueeze(0).to(DEVICE)
            aw = ep.actions[t:t + WINDOW].unsqueeze(0).to(DEVICE)
            fa = ep.actions[t + WINDOW:t + WINDOW + K_MAX].unsqueeze(0).to(DEVICE)
            v0 = (ep.poses[last_t, 3:4] / SPEED_SCALE).to(DEVICE)
            aw3 = append_speed_channel(aw, v0)
            fa3 = append_speed_channel(fa, v0)
            states = world.encode_window(fw)
            gt_wp = gt_ego_waypoints(ep.poses, last_t)
            cv_wp = baseline_waypoints(ep.poses, last_t)["constant_velocity"]
            for m in modes:
                predv = _make_predictor_variant(world, m)
                wp_full, step_dpose = rollout_decode(predv, states, aw3, fa3, step_readout, K_MAX)
                pred_wp = wp_full.index_select(1, wp_idx.to(DEVICE)).cpu()
                per_mode[m]["pred"].append(pred_wp.float())
                per_mode[m]["gt"].append(gt_wp)
                per_mode[m]["cv"].append(cv_wp)
                per_mode[m]["step_dpose"].append(step_dpose.cpu().float())
            n_windows += 1
        log(f"downstream proxy: episode consumed, cumulative windows={n_windows}")

    report = {"n_windows": n_windows, "note": ("TRAIN-CACHE PROXY on "
              "physicalai-train-e438721ae894 episodes 2000-2039 (NOT the "
              "canonical taniteval val set physicalai-val-0c5f7dac3b11, which "
              "lives on tanitad-eval, off-limits to this stream) -- directional "
              "evidence for the FP16-vs-INT8 accuracy delta, not a registry-"
              "quotable ADE."), "wp_steps": list(WP_STEPS), "per_precision": {}}
    for m in modes:
        pred = torch.cat(per_mode[m]["pred"])
        gt = torch.cat(per_mode[m]["gt"])
        cv = torch.cat(per_mode[m]["cv"])
        sdp = torch.cat(per_mode[m]["step_dpose"])                 # [N, 20, 3]
        de = de_of(pred, gt)
        metrics = scalar_metrics(de)
        cv_de = de_of(cv, gt)
        cv_metrics = scalar_metrics(cv_de)
        per_step_l2 = sdp[..., :2].norm(dim=-1).mean(dim=0).tolist()   # length 20
        report["per_precision"][m] = {
            "ade_de_metrics": metrics, "cv_metrics": cv_metrics,
            "per_step_dpose_l2_mean_m": per_step_l2,
        }
        log(f"downstream proxy [{m}] ade_0_2s={metrics.get('ade_0_2s')}")

    ref = report["per_precision"]["fp32"]["ade_de_metrics"]
    for m in ("fp16", "int8_wo", "int8_wa"):
        cur = report["per_precision"][m]["ade_de_metrics"]
        delta = {k: (cur[k] - ref[k]) for k in ref if isinstance(ref[k], (int, float))
                 and isinstance(cur.get(k), (int, float))}
        report["per_precision"][m]["delta_vs_fp32"] = delta
        # compounding diagnostic: ratio of |delta at 2s| to |delta at 0.5s|
        try:
            d05 = abs(delta.get("de@0.5s", delta.get("ade@0.5s", 0.0)))
            d20 = abs(delta.get("de@2s", 0.0))
            report["per_precision"][m]["degradation_grows_with_horizon"] = (
                None if d05 < 1e-9 else (d20 / d05))
        except Exception:
            pass
    return report


def main():
    log("=== PHASE 1: real-weight load ===")
    world, grounding, step_readout, meta = load_world_and_grounding()
    merge_report("setup", {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                           **meta})

    log("=== PHASE 2: harvest calibration/accuracy data (TRAIN cache) ===")
    frames_real = harvest_encoder_frames(CALIB_EP_RANGE, n=N_ACC_SAMPLES)
    states_real, actions_real = harvest_predictor_pairs(world, CALIB_EP_RANGE, n=N_ACC_SAMPLES)
    log(f"harvested {frames_real.shape[0]} encoder frames, {states_real.shape[0]} predictor pairs")

    log("=== PHASE 3: encoder per-block accuracy sweep ===")
    with strict_numerics():
        enc_report = encoder_sweep(world, frames_real)
    merge_report("encoder_accuracy_sweep", enc_report)

    log("=== PHASE 4: predictor per-block accuracy sweep ===")
    with strict_numerics():
        pred_report = predictor_sweep(world, states_real, actions_real)
    merge_report("predictor_accuracy_sweep", pred_report)

    log("=== PHASE 5: downstream rollout ADE proxy (held-out episodes) ===")
    proxy_episodes = harvest_proxy_episodes(PROXY_EP_RANGE)
    log(f"proxy episodes loaded: {len(proxy_episodes)}")
    with strict_numerics():
        ade_report = downstream_ade_proxy(world, step_readout, proxy_episodes)
    merge_report("downstream_ade_proxy", ade_report)

    log("PHASE 1 COMPLETE")


if __name__ == "__main__":
    main()
