"""PREREG_V8F_WMPOOL controls 1, 2, 4 and the synthetic half of 3 (plus 5's read-back), each with its deliberate-regression arm (a mutation that must make the check go RED).
CPU only (CUDA_VISIBLE_DEVICES=-1, 2 threads). Usage:
    python controls_wmpool.py --tree <patched tree> --base <baseline tree> --out <dir>      (all controls)
    python controls_wmpool.py --dump-c1 <tree> <npz>                                         (child of control 1)
Batch: the REAL val windows of ladder_read/cache/frames_s0.npz (zmean, act of 141 clips) for the mean path and the action channel. That file holds no adapter tokens, so the 40 TOKENS of the on-arm batch are PLANTED:
token = the real zmean of the frame + noise (std = the std of zmean, a different draw per token); the cells are the 2 x 2 block means of those tokens, and the mean arm's input is the token mean of the same tokens.
Control 3 on 8 REAL clips needs box 2's raw feature cache: `zcells.py check3` (not run here).
"""
import argparse, copy, dataclasses, hashlib, json, os, re, subprocess, sys, tempfile
import numpy as np

FRAMES = "F:/TanitAD-scratch/ladder_read/cache/frames_s0.npz"
ARGV0 = "C:/Users/Admin/t2v8f_pulls/ladder_R35H_s0/argv.txt"
F_WIN = 91
HERE = os.path.dirname(os.path.abspath(__file__))


def setup(tree):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, os.path.join(tree, "stack")), sys.path.insert(0, os.path.join(tree, "stack", "scripts"))
    import tanitad
    assert os.path.normcase(os.path.abspath(tanitad.__file__)).startswith(os.path.normcase(os.path.abspath(tree))), tanitad.__file__
    import v8f_mini_train as T
    from tanitad.models import v8f_mini as M
    assert os.path.normcase(os.path.abspath(T.__file__)).startswith(os.path.normcase(os.path.abspath(tree))), T.__file__
    return torch, T, M


def free_gb():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("ml", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
    m = MS(); m.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ap / 2**30


def need_ram(gb=3.0):
    import time
    for _ in range(30):
        if free_gb() >= gb:
            return
        time.sleep(10)
    raise RuntimeError(f"free physical memory {free_gb():.2f} GB < {gb}")


def real_batch(torch, n=2):
    f = np.load(FRAMES)
    offs = np.concatenate([[0], np.cumsum(f["T"])])
    zs, acts = [], []
    for i in range(len(f["T"])):
        T_ = int(f["T"][i])
        if T_ < F_WIN:
            continue
        lo = offs[i] + T_ - F_WIN
        if not f["ok"][lo:lo + F_WIN].all():
            continue
        zs.append(f["zmean"][lo:lo + F_WIN]), acts.append(np.nan_to_num(f["act"][lo:lo + F_WIN]))
        if len(zs) == n:
            break
    z, a = torch.as_tensor(np.stack(zs), dtype=torch.float32), torch.as_tensor(np.stack(acts), dtype=torch.float32)
    g = torch.Generator().manual_seed(3)
    tok = z[:, :, None, :] + float(z.std()) * torch.randn(z.shape[0], z.shape[1], 40, z.shape[2], generator=g)        # PLANTED tokens [B, F, 40, 256]
    return {"zmean": z, "act": a, "tok": tok}


def cfg_for(T, extra):
    argv = open(ARGV0).read().split()
    a = T.build_parser().parse_args(argv + extra)
    return a, T.make_config(a)


def full_model(torch, M, cfg, seed=0):
    torch.manual_seed(seed)
    return M.V8FMini(cfg, M.P10Prior.synthetic(0), None, None)


def on_batch(M, b):
    cells = M.WMAux.cells_of_adapt(b["tok"])
    return {"zcell": cells, "act": b["act"]}, {"zmean": b["tok"].mean(2), "act": b["act"]}


def line_of(tree, rel, needle, nth=0):
    ls = open(os.path.join(tree, rel), encoding="utf-8").read().splitlines()
    hits = [i + 1 for i, l in enumerate(ls) if needle in l]
    return hits[nth] if len(hits) > nth else None


# ------------------------------------------------------------------------------------------ control 1 (child): dump one off step
def dump_c1(tree, out):
    torch, T, M = setup(tree)
    f = np.load(FRAMES)
    b = real_batch(torch)
    _, cfg = cfg_for(T, [])
    m = full_model(torch, M, cfg)
    res = {}
    for k, v in m.state_dict().items():
        res["init/" + k] = v.clone().numpy()
    opt = torch.optim.AdamW([p for n, p in m.named_parameters() if n.startswith("wm.")], lr=1e-3)
    m.train()
    torch.manual_seed(7)
    terms = m.wm_only_loss({"zmean": b["zmean"], "act": b["act"]})["terms"]
    for k, v in terms.items():
        res["term/" + k] = v.detach().numpy()
    terms["total"].backward()
    for n, p in m.named_parameters():
        if n.startswith("wm."):
            res["grad/" + n] = p.grad.detach().numpy().copy()
    opt.step()
    for n, p in m.named_parameters():
        if n.startswith("wm."):
            res["after/" + n] = p.detach().numpy().copy()
    np.savez(out, **res)


def control1_off(tree, base, outdir):
    me = os.path.abspath(__file__)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2")
    paths = {}
    for nm, t in (("base", base), ("patched", tree)):
        need_ram(3.0)
        paths[nm] = os.path.join(outdir, f"_c1_{nm}.npz")
        env2 = dict(env, PYTHONPATH=";".join([os.path.join(t, "stack"), os.path.join(t, "stack", "scripts")]))
        r = subprocess.run([sys.executable, me, "--dump-c1", t, paths[nm]], env=env2, capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            raise RuntimeError(r.stderr[-2000:])
    A, B = np.load(paths["base"]), np.load(paths["patched"])
    keys = sorted(A.files)
    same = [k for k in keys if k in B.files and np.array_equal(A[k], B[k])]
    res = {"n_arrays": len(keys), "n_bit_equal": len(same), "keys_only_in_patched": sorted(set(B.files) - set(A.files)), "differing": sorted(set(keys) - set(same))[:6],
           "pass": len(same) == len(keys) and set(A.files) == set(B.files)}
    A.close(); B.close()
    for p in paths.values():
        os.remove(p)
    return res


# ------------------------------------------------------------------------------------------ control 1 (on, zero query)
def control1_on(torch, T, M, b):
    need_ram(3.0)
    _, c_off = cfg_for(T, [])
    _, c_on = cfg_for(T, ["--wm-pool", "att25"])
    m_off, m_on = full_model(torch, M, c_off), full_model(torch, M, c_on)
    sd_off, sd_on = m_off.state_dict(), m_on.state_dict()
    extra = sorted(set(sd_on) - set(sd_off))
    eq_init = all(torch.equal(sd_off[k], sd_on[k]) for k in sd_off)                 # same seed: every shared tensor bit-equal at init (the query consumes no RNG)
    bon, boff = on_batch(M, b)
    with torch.no_grad():
        w = m_on.wm.pool_weights(bon["zcell"])
        z_on, z_off = m_on.wm.z1_from_cells(bon["zcell"]), m_off.wm.z1_from_mean(boff["zmean"])
        z_on16 = m_on.wm.z1_from_cells(bon["zcell"].half())                          # the cache stores fp16
        z_live = m_on.wm.z1(b["tok"])                                                # the live-token spelling must equal the cached-cell spelling
    d32, d16 = float((z_on - z_off).abs().max()), float((z_on16 - z_off).abs().max())
    rel32 = d32 / float(z_off.abs().max())
    torch.manual_seed(11)                                                           # the loss has random projections (SIGReg): same stream for both
    terms_off = {k: v.detach() for k, v in m_off.wm_only_loss(boff)["terms"].items()}
    torch.manual_seed(11)
    terms_on = {k: v.detach() for k, v in m_on.wm_only_loss(bon)["terms"].items()}
    tdiff = {k: float((terms_on[k] - terms_off[k]).abs()) for k in terms_off}
    tol = 1e-5
    out = {"extra_state_keys": extra, "shared_tensors_bit_equal_at_init": bool(eq_init), "weights_exactly_uniform_0.1": bool(torch.equal(w, torch.full_like(w, 0.1))),
           "z1_on_vs_proj_mean_max_abs": d32, "z1_scale_max_abs": float(z_off.abs().max()), "z1_rel": rel32, "bit_exact": bool(torch.equal(z_on, z_off)),
           "z1_fp16_cells_vs_proj_mean_max_abs": d16, "live_tokens_equal_cached_cells_max_abs": float((z_live - z_on).abs().max()), "loss_terms_max_abs_diff": max(tdiff.values()), "n_terms": len(tdiff), "tolerance": tol,
           "statement": "NOT bit-exact: the sum of ten 0.1 x cell terms differs from the mean of 40 tokens at fp32 round-off; measured gap above"}
    # deliberate regression: a NON-ZERO query init must make the equality go red
    m_bad = full_model(torch, M, c_on)
    torch.nn.init.normal_(m_bad.wm.pool_q, std=0.5)
    with torch.no_grad():
        d_bad = float((m_bad.wm.z1_from_cells(bon["zcell"]) - z_off).abs().max())
    out["mutation_nonzero_query_init"] = {"max_abs_diff": d_bad, "went_red": bool(d_bad > tol)}
    out["pass"] = bool(extra == ["wm.pool_q"] and eq_init and out["weights_exactly_uniform_0.1"] and d32 <= tol and max(tdiff.values()) <= tol and float((z_live - z_on).abs().max()) <= tol and d_bad > tol)
    return out


# ------------------------------------------------------------------------------------------ control 2
def control2(torch, T, M, b, tree):
    need_ram(3.0)
    a, c = cfg_for(T, ["--wm-pool", "att25"])
    bon, _ = on_batch(M, b)
    m = full_model(torch, M, c).cpu()
    live = T.wm_only_live_check(m, bon, c)                                          # the trainer's own G-LIVE on the batch
    mm = copy.deepcopy(m)
    for p in mm.parameters():
        p.requires_grad_(True)
    torch.manual_seed(0)
    mm.wm_only_loss(bon)["terms"]["total"].backward()
    gq = mm.wm.pool_q.grad
    out = {"live_pass": bool(live["pass"]), "dead_wm_groups": live["dead_wm_groups"], "pool_q_grad_abs_sum": float(gq.abs().sum()), "pool_q_grad_nonzero": bool(float(gq.abs().sum()) > 0.0),
           "control_nonwm_grad_sum_must_be_exactly_0.0": live["control_nonwm_grad_sum"], "outside_graph_params_read_exactly_zero": bool(live["control_nonwm_grad_sum"] == 0.0)}
    # the default does not enable it
    a0, c0 = cfg_for(T, [])
    m0 = full_model(torch, M, c0)
    out["default_argv"] = {"wm_pool": a0.wm_pool, "cfg": c0.wm_pool, "has_pool_q": m0.wm.pool_q is None, "built_mode": m0.wm.built_pool()["mode"], "parser_default": T.build_parser().get_default("wm_pool")}
    out["default_off_pass"] = bool(a0.wm_pool == "mean" and c0.wm_pool == "mean" and m0.wm.pool_q is None and m0.wm.built_pool()["mode"] == "mean" and not any(k.endswith("pool_q") for k in m0.state_dict()))
    out["named_in_argv_arm"] = {"argv_has_flag": "--wm-pool" in ["--wm-pool", "att25"], "parsed": a.wm_pool}
    # file:line per hop
    hops = [("argparse --wm-pool", "stack/scripts/v8f_mini_train.py", 'ap.add_argument("--wm-pool"'), ("gate entry FLAG_CHECKS", "stack/scripts/v8f_mini_train.py", '"wm_pool": "lever:wm_pool"'),
            ("namespace -> config (the one line)", "stack/scripts/v8f_mini_train.py", "wm_pool=a.wm_pool,"), ("config field", "stack/tanitad/models/v8f_mini.py", 'wm_pool: str = "mean"'),
            ("model attribute: the query", "stack/tanitad/models/v8f_mini.py", "self.pool_q = nn.Parameter"), ("data: pool kind handed to the dataset", "stack/scripts/v8f_mini_train.py", "zcache=a.wm_zcache, pool=a.wm_pool"),
            ("data: cells served", "stack/tanitad/data/v8f_wm_windows.py", '"zcell" if self.pool == "att25"'), ("batch: collate", "stack/tanitad/data/v8f_wm_windows.py", 'out["zcell"]'),
            ("loss entry: batch -> forward_cells", "stack/tanitad/models/v8f_mini.py", 'wm = self.wm.forward_cells(batch["zcell"]'), ("forward_cells -> z1_from_cells", "stack/tanitad/models/v8f_mini.py", "self.stack.loss(self.z1_from_cells(cells)"),
            ("attention weights (the pool line)", "stack/tanitad/models/v8f_mini.py", "w = self.pool_weights(cells)"), ("pooled -> proj", "stack/tanitad/models/v8f_mini.py", "self._z1_post((w.unsqueeze(-1) * cells.float()).sum(-2))")]
    out["hops"] = [{"hop": h, "file:line": f"{f}:{line_of(tree, f, n)}"} for h, f, n in hops]
    out["all_hops_found"] = all(not h["file:line"].endswith("None") for h in out["hops"])
    # deliberate regression: delete the pool line (the cells bypass the attention: a plain mean) -> pool_q gets no gradient -> G-LIVE must go red
    orig = M.WMAux.z1_from_cells
    M.WMAux.z1_from_cells = lambda self, cells: self._z1_post(cells.float().mean(-2))
    try:
        try:
            T.wm_only_live_check(full_model(torch, M, c).cpu(), bon, c)
            red, why = False, "live check PASSED under the mutation"
        except AssertionError as e:
            red, why = True, str(e)[:240]
    finally:
        M.WMAux.z1_from_cells = orig
    out["mutation_pool_line_deleted"] = {"went_red": red, "why": why}
    # deliberate regression 2: the line that applies --wm-pool is deleted (cfg built without it) -> the built model is mean -> G-DVB must refuse
    cfg_m = dataclasses.replace(c, wm_pool="mean")
    mod_m = full_model(torch, M, cfg_m)
    bad_m = T.bad_wm_pool(a, cfg_m, mod_m, {"wm_windows": {"pool": "att25", "zcell_shape": [F_WIN, 10, 256]}})
    out["mutation_flag_not_applied"] = {"went_red": bool(bad_m), "refusal": bad_m[:2]}
    out["pass"] = bool(out["live_pass"] and out["pool_q_grad_nonzero"] and out["outside_graph_params_read_exactly_zero"] and out["default_off_pass"] and out["all_hops_found"] and red and bool(bad_m))
    return out


# ------------------------------------------------------------------------------------------ control 3 (synthetic half)
def control3_synth(torch, T, M, tree):
    sys.path.insert(0, HERE)
    import zcells as Z
    g = torch.Generator().manual_seed(5)
    d = 256
    tok = torch.randn(6, 40, d, generator=g)
    cells = Z.cells_of_tokens(tok)
    live = M.WMAux.cells_of_adapt(tok[None])[0]
    ident = float((cells.mean(1) - tok.mean(1)).abs().max())
    out = {"cells_of_tokens_equals_WMAux_cells_of_adapt_max_abs": float((cells - live).abs().max()), "mean_of_cells_vs_token_mean_fp32_max_abs": ident}
    # layout: token i = r*10 + c; cell R*5 + C = tokens 20R + 10r' + 2C + c'
    plant = torch.arange(40, dtype=torch.float32)[None, :, None].expand(1, 40, 4).contiguous()
    pc = Z.cells_of_tokens(plant)[0, :, 0].tolist()
    want = [sum(20 * R + 10 * a + 2 * C + b_ for a in (0, 1) for b_ in (0, 1)) / 4.0 for R in (0, 1) for C in range(5)]
    out["layout_cell_map_ok"] = bool(np.allclose(pc, want))
    out["layout_cells"] = {"0": pc[0], "4": pc[4], "5": pc[5], "9": pc[9]}
    # fp16 storage error of cells and of their mean
    c16 = cells.half().float()
    out["fp16_cell_max_abs_err"], out["fp16_cell_max_rel_err_of_max"] = float((c16 - cells).abs().max()), float((c16 - cells).abs().max() / cells.abs().max())
    out["fp16_mean_of_stored_cells_vs_token_mean_max_abs"] = float((c16.mean(1) - tok.mean(1)).abs().max())
    out["token_scale_max_abs"] = float(tok.abs().max())
    # builder on a built model with the real adapter module (random weights): cells' mean == zcache.zmean_of
    need_ram(3.0)
    _, c = cfg_for(T, [])
    m = full_model(torch, M, c)
    feats = (torch.randn(20, 40, 1024, generator=g)).half()
    sys.path.insert(0, os.path.join(HERE, "..", "ladder"))
    os.environ.setdefault("ZC_REPO", tree.replace("\\", "/"))
    import zcache as ZC
    zm = ZC.zmean_of(m, feats)
    zc = Z.zcell_of(m, feats)
    out["builder_cells_mean_vs_zcache_zmean_max_abs"] = float((zc.mean(1) - zm).abs().max())
    out["builder_zmean_scale_max_abs"] = float(zm.abs().max())
    # wrong adapter (another seed) must FAIL the identity
    m2 = full_model(torch, M, c, seed=1)
    out["wrong_seed_adapter_max_abs"] = float((Z.zcell_of(m2, feats).mean(1) - zm).abs().max())
    # mutation: a block partition that drops half the tokens (only row r'=0 of each block) must break the identity
    bad = tok.reshape(6, 2, 2, 5, 2, d)[:, :, 0].mean(dim=3).reshape(6, 10, d)
    out["mutation_block_drops_half_the_tokens"] = {"mean_vs_token_mean_max_abs": float((bad.mean(1) - tok.mean(1)).abs().max())}
    out["mutation_block_drops_half_the_tokens"]["went_red"] = bool(out["mutation_block_drops_half_the_tokens"]["mean_vs_token_mean_max_abs"] > 1e-3)
    out["pass"] = bool(ident < 1e-5 and out["cells_of_tokens_equals_WMAux_cells_of_adapt_max_abs"] == 0.0 and out["layout_cell_map_ok"] and out["builder_cells_mean_vs_zcache_zmean_max_abs"] < 1e-5
                       and out["wrong_seed_adapter_max_abs"] > 1e-5 and out["mutation_block_drops_half_the_tokens"]["went_red"])
    out["note"] = "control 3 on 8 REAL clips (cached cells vs the cached zmean on disk) needs box 2: `zcells.py check3`; this is the synthetic half"
    return out


# ------------------------------------------------------------------------------------------ control 4
def control4(torch, T, M, b):
    need_ram(3.0)
    _, c_off = cfg_for(T, [])
    _, c_on = cfg_for(T, ["--wm-pool", "att25"])
    m_off, m_on = full_model(torch, M, c_off), full_model(torch, M, c_on)
    bon, boff = on_batch(M, b)
    a = m_off.wm.scale_actions(b["act"])[:, :-1]
    with torch.no_grad():
        enc_off = m_off.wm.stack.encode(m_off.wm.z1_from_mean(boff["zmean"]), a)
        enc_on = m_on.wm.stack.encode(m_on.wm.z1_from_cells(bon["zcell"]), a)
        a2 = a + 1.0                                                                  # different actions: the targets must not move
        enc_on_a2 = m_on.wm.stack.encode(m_on.wm.z1_from_cells(bon["zcell"]), a2)
        z1o, z1p = m_off.wm.z1_from_mean(boff["zmean"]), m_on.wm.z1_from_cells(bon["zcell"])
        tgt = [float((enc_off.z[i] - enc_on.z[i]).abs().max()) for i in range(3)]
        tgt_act = [bool(torch.equal(enc_on.z[i], enc_on_a2.z[i])) for i in range(3)]
        goal = [float((m_off.wm.stack.lift(z1o[:, :31], 1, L, strict=False) - m_on.wm.stack.lift(z1p[:, :31], 1, L, strict=False)).abs().max()) for L in (2, 3)]
    import inspect
    sig = list(inspect.signature(M.WMAux.z1_from_cells).parameters)
    wm_pool_params = [n for n, _ in m_on.named_parameters() if "pool" in n]
    tol = 1e-5
    out = {"target_latents_max_abs_off_vs_on_zero_query": tgt, "target_latents_bit_equal_under_changed_actions": tgt_act, "goal_latents_max_abs_off_vs_on_zero_query": goal,
           "z1_from_cells_signature": sig, "only_cells_enter_the_pool": sig == ["self", "cells"], "pool_parameters": wm_pool_params, "tolerance": tol,
           "no_speed_or_action_channel": bool(wm_pool_params == ["wm.pool_q"] and sig == ["self", "cells"])}
    # with a NON-ZERO query the lever acts, and the targets are still a function of the cells only (equal for two different action batches)
    with torch.no_grad():
        m_on.wm.pool_q.copy_(torch.randn(256, generator=torch.Generator().manual_seed(1)) * 0.5)
        e1 = m_on.wm.stack.encode(m_on.wm.z1_from_cells(bon["zcell"]), a)
        e2 = m_on.wm.stack.encode(m_on.wm.z1_from_cells(bon["zcell"]), a2)
        out["nonzero_query_targets_bit_equal_under_changed_actions"] = [bool(torch.equal(e1.z[i], e2.z[i])) for i in range(3)]
        out["nonzero_query_moves_the_target_vs_off"] = [float((e1.z[i] - enc_off.z[i]).abs().max()) for i in range(3)]
    # deliberate regression: an action/speed-derived scalar leaks into the pool -> targets move with the action batch
    orig = M.WMAux.z1_from_cells
    leak = {"v": 0.0}
    M.WMAux.z1_from_cells = lambda self, cells: orig(self, cells) + leak["v"]
    try:
        with torch.no_grad():
            leak["v"] = 0.0
            r1 = m_on.wm.stack.encode(m_on.wm.z1_from_cells(bon["zcell"]), a).z[0]
            leak["v"] = 0.7                                                          # a different "speed" in the pool
            r2 = m_on.wm.stack.encode(m_on.wm.z1_from_cells(bon["zcell"]), a).z[0]
        red = not torch.equal(r1, r2)
    finally:
        M.WMAux.z1_from_cells = orig
    out["mutation_speed_scalar_leaks_into_pool"] = {"went_red": bool(red)}
    out["pass"] = bool(max(tgt) <= tol and max(goal) <= tol and all(tgt_act) and out["only_cells_enter_the_pool"] and out["no_speed_or_action_channel"] and all(out["nonzero_query_targets_bit_equal_under_changed_actions"]) and red)
    return out


# ------------------------------------------------------------------------------------------ control 5 (read-back / refusal)
def control5(torch, T, M, b, outdir, tree):
    import shutil
    need_ram(3.0)
    a, c = cfg_for(T, ["--wm-pool", "att25"])
    torch.manual_seed(0)
    m = full_model(torch, M, c)
    sha = T.adapter_state_sha256(m.state_dict())
    rec = {"wm_windows": {"pool": "att25", "zcell_shape": [c.hmax, 10, 256]}}
    d = os.path.join(outdir, "_cells_fixture")
    os.makedirs(d, exist_ok=True)
    man = {"kind": "zcell", "grid": "pool4x10->2x5", "n_cells": 10, "d": 256, "dtype": "float16", "adapter_state_sha256": sha}
    json.dump(man, open(os.path.join(d, "CELL_MANIFEST.json"), "w"))
    out = {}
    try:
        a.wm_zcache = d
        out["good_cache_bad_list"] = T.bad_wm_pool(a, c, m, rec)
        out["att25_readback"] = m.wm.built_pool()
        # refusals
        a2 = copy.copy(a); a2.wm_zcache = ""
        out["refuse_no_cache"] = T.pool_cache_problems(a2)
        a3 = copy.copy(a); a3.wm_zcache = outdir
        out["refuse_mean_cache_dir_without_manifest"] = T.pool_cache_problems(a3)
        json.dump(dict(man, adapter_state_sha256="0" * 64), open(os.path.join(d, "CELL_MANIFEST.json"), "w"))
        out["refuse_wrong_adapter_sha"] = T.bad_wm_pool(a, c, m, rec)
        a4 = copy.copy(a); a4.wm_only = False
        out["refuse_no_wm_only"] = T.pool_cache_problems(a4)
        # main() refuses before building anything
        r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s'); import v8f_mini_train as T; T.main(%r)" % (os.path.join(tree, "stack"), os.path.join(tree, "stack", "scripts"),
                            open(ARGV0).read().split() + ["--wm-pool", "att25"])], capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
        out["main_refuses_att25_with_mean_cache_argv"] = {"returncode": r.returncode, "message": (r.stderr or r.stdout).strip().splitlines()[-1][:300] if (r.stderr or r.stdout).strip() else ""}
        # dataset refuses att25 without a directory; a mean cache read as cells fails loudly
        sys.path.insert(0, os.path.join(tree, "stack"))
        from tanitad.data import v8f_wm_windows as WW
        try:
            WW.V8FMiniWMWindows(None, {}, [], F=91, zcache="", pool="att25"); out["dataset_refuses_no_dir"] = False
        except ValueError as e:
            out["dataset_refuses_no_dir"] = str(e)[:120]
        np.savez(os.path.join(d, "clipX.npz"), zmean=np.zeros((5, 256), np.float32), ok=np.ones(5, bool), t_s=np.zeros(5, np.float32))
        ds = WW.V8FMiniWMWindows(None, {}, [], F=91, zcache=d, pool="att25")
        try:
            ds._zmean("clipX"); out["mean_cache_read_as_cells_fails"] = False
        except KeyError as e:
            out["mean_cache_read_as_cells_fails"] = f"KeyError {e}"
        # a cell cache read as a mean cache fails too
        np.savez(os.path.join(d, "clipY.npz"), zcell=np.zeros((5, 10, 256), np.float16), ok=np.ones(5, bool), t_s=np.zeros(5, np.float32))
        ds2 = WW.V8FMiniWMWindows(None, {}, [], F=91, zcache=d, pool="mean")
        try:
            ds2._zmean("clipY"); out["cell_cache_read_as_mean_fails"] = False
        except KeyError as e:
            out["cell_cache_read_as_mean_fails"] = f"KeyError {e}"
        # item/collate shapes on a stub store
        class S:
            def get(self, clip):
                return {"a": np.zeros(100, np.float32), "ch1": np.zeros(100, np.float32), "pose": np.zeros((100, 4)), "n": 100}
        np.savez(os.path.join(d, "clipZ.npz"), zcell=np.ones((100, 10, 256), np.float16), ok=np.ones(100, bool), t_s=np.zeros(100, np.float32))
        ds3 = WW.V8FMiniWMWindows(S(), {"clip": np.array(["clipZ"]), "n": np.array([100])}, [(0, 95, 0, 0)], F=91, zcache=d, pool="att25")
        bt = WW.collate_wm([ds3[0], ds3[0]])
        out["batch_keys_and_shapes"] = {k: list(v.shape) + [str(v.dtype)] for k, v in bt.items()}
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # a mean model: the lever off reads nothing built
    a0, c0 = cfg_for(T, [])
    m0 = full_model(torch, M, c0)
    out["control_R35H_argv"] = {"declared": a0.wm_pool, "built": m0.wm.built_pool()["mode"], "bad": T.bad_wm_pool(a0, c0, m0, {"wm_windows": {"pooled": True}})}
    out["pass"] = bool(out["good_cache_bad_list"] == [] and out["refuse_no_cache"] and out["refuse_mean_cache_dir_without_manifest"] and out["refuse_wrong_adapter_sha"] and out["refuse_no_wm_only"]
                       and out["main_refuses_att25_with_mean_cache_argv"]["returncode"] != 0 and "ZZDVB-REFUSEDZZ" in out["main_refuses_att25_with_mean_cache_argv"]["message"] and out["dataset_refuses_no_dir"]
                       and out["mean_cache_read_as_cells_fails"] and out["cell_cache_read_as_mean_fails"] and out["batch_keys_and_shapes"]["zcell"][1:3] == [91, 10] and out["control_R35H_argv"]["bad"] == []
                       and out["att25_readback"]["mode"] == "att25" and out["control_R35H_argv"]["built"] == "mean")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree"), ap.add_argument("--base"), ap.add_argument("--out")
    ap.add_argument("--dump-c1", nargs=2)
    a = ap.parse_args()
    if a.dump_c1:
        return dump_c1(*a.dump_c1)
    os.makedirs(a.out, exist_ok=True)
    torch, T, M = setup(a.tree)
    b = real_batch(torch)
    r = {"tree": a.tree, "base": a.base, "batch": {"source": FRAMES, "windows": int(b["zmean"].shape[0]), "F": F_WIN, "note": "act and the frame-level zmean are REAL val frames; the 40 tokens per frame are PLANTED (zmean + noise); cells = their 2 x 2 block means"}}
    r["control1_off_identity"] = control1_off(a.tree, a.base, a.out)
    r["control1_on_zero_query"] = control1_on(torch, T, M, b)
    r["control2_reachability"] = control2(torch, T, M, b, a.tree)
    r["control3_synthetic_cell_identity"] = control3_synth(torch, T, M, a.tree)
    r["control4_echo_guard"] = control4(torch, T, M, b)
    r["control5_launch_readback_refusal"] = control5(torch, T, M, b, a.out, a.tree)
    r["all_pass"] = bool(r["control1_off_identity"]["pass"] and r["control1_on_zero_query"]["pass"] and r["control2_reachability"]["pass"] and r["control3_synthetic_cell_identity"]["pass"]
                         and r["control4_echo_guard"]["pass"] and r["control5_launch_readback_refusal"]["pass"])
    json.dump(r, open(os.path.join(a.out, "CONTROLS_WMPOOL.json"), "w"), indent=1, default=str)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) and "pass" in v else (v if not isinstance(v, dict) else "...")) for k, v in r.items()}, default=str))


if __name__ == "__main__":
    main()
