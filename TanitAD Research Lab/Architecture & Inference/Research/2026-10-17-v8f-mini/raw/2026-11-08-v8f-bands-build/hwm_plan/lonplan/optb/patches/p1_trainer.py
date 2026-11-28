import sys
p = sys.argv[1]
s = open(p, encoding='utf-8', newline='').read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old))
    s = s.replace(old, new)


# FLAG_CHECKS
rep('    "wm_rsp_w": "lever:wm_rsp_w",\n', '''    "wm_rsp_w": "lever:wm_rsp_w",
    # PREREG_V8F_ACTRSP Amendment 2 (Option B): `lever:wm_rsp_readout` (also `wm_rsp_probe`, `wm_rsp_probe_sha256`) = the readout the BUILT loss reads through (HJEPAStack.built_levers()[1]["rsp_readout"] and WMAux.built_rsp: readout, contrast count 2, s_cal, the sha256 of the live probe weights, each compared with the probe FILE re-read here) -> bad_wm_rsp; `lever:wm_init_from_ckpt_wm` = the loader record (every wm.* tensor loaded, read back equal) -> bad_wm_only; `lever:wm_freeze_proj` = requires_grad of wm.proj.* read back from the BUILT model -> bad_wm_freeze.
    "wm_rsp_readout": "lever:wm_rsp_readout", "wm_rsp_probe": "lever:wm_rsp_readout", "wm_rsp_probe_sha256": "lever:wm_rsp_readout",
    "wm_init_from_ckpt_wm": "lever:wm_init_from_ckpt_wm", "wm_freeze_proj": "lever:wm_freeze_proj",
''')
# argparse
i = s.index('    ap.add_argument("--wm-rollout-k"')
s = s[:i] + '''    ap.add_argument("--wm-rsp-readout", dest="wm_rsp_readout", choices=("idm", "fixed_k"), default="idm",
                    help="PREREG_V8F_ACTRSP Amendment 2: the readout of the response term. idm (default) = the a13f6f16 term, bit for bit. fixed_k = Option B: a FROZEN linear readout of the K = 10 endpoint, acceleration channel only, two contrasts, a constant slope. Needs --wm-rsp-w > 0, --wm-rsp-probe, --wm-rsp-probe-sha256 and --wm-freeze-proj on")
    ap.add_argument("--wm-rsp-probe", dest="wm_rsp_probe", default="", help="Amendment 2: the readout file (readout_K10_R35_s{k}.pt of amend1_kstep.py). Only with --wm-rsp-readout fixed_k")
    ap.add_argument("--wm-rsp-probe-sha256", dest="wm_rsp_probe_sha256", default="", help="Amendment 2: the readout file sha256 (64 hex); the build refuses a mismatch")
    ap.add_argument("--wm-init-from-ckpt-wm", dest="wm_init_from_ckpt_wm", choices=("off", "on"), default="off",
                    help="Amendment 2: on = the WM-only start loads EVERY tensor of --init-ckpt including wm.* (a fine-tune of the checkpoint world model); off (default) = the WM is re-initialised under --seed (every earlier WM-only arm). Needs --wm-only and --init-ckpt")
    ap.add_argument("--wm-freeze-proj", dest="wm_freeze_proj", choices=("off", "on"), default="off",
                    help="Amendment 2: on = wm.proj (LayerNorm + Linear, adapter -> level-1 latent) is frozen, so the latent space the frozen readout lives in cannot move. Needs --wm-only. Default off = every earlier arm")
''' + s[i:]
# make_config
rep("                           wm_rsp_w=float(a.wm_rsp_w),",
    "                           wm_rsp_readout=a.wm_rsp_readout, wm_rsp_probe=a.wm_rsp_probe, wm_rsp_probe_sha256=a.wm_rsp_probe_sha256, wm_freeze_proj=a.wm_freeze_proj,   # Amendment 2: the line that applies the Option B flags (MR11-MR13 unwire them)\n                           wm_rsp_w=float(a.wm_rsp_w),")
# freeze
rep('''def freeze_all_but_wm(model) -> None:
    """PREREG_V8F_MRATE: the WM-only mode trains the world model and nothing else. Called BEFORE the optimiser is built (its decay / nodecay lists filter on requires_grad)."""
    for n, p in model.named_parameters():
        p.requires_grad_(is_wm_param(n))''', '''def is_wm_proj_param(name: str) -> bool:
    return name.startswith("wm.proj.")


def freeze_all_but_wm(model, freeze_proj: bool = False) -> None:
    """PREREG_V8F_MRATE: the WM-only mode trains the world model and nothing else. Called BEFORE the optimiser is built (its decay / nodecay lists filter on requires_grad). ``freeze_proj`` (Amendment 2, ``--wm-freeze-proj on``): ``wm.proj`` stays frozen too."""
    for n, p in model.named_parameters():
        p.requires_grad_(is_wm_param(n) and not (freeze_proj and is_wm_proj_param(n)))''')
# loader
rep("def load_wm_only_init(path, model, cfg, seed: int, apply: bool = True) -> dict:",
    "def load_wm_only_init(path, model, cfg, seed: int, apply: bool = True, init_wm: bool = False) -> dict:")
rep('''    diff = {k: (norm(ck["cfg"][k]), norm(getattr(cfg, k))) for k in INIT_CKPT_MATCH if k not in WM_ONLY_NOT_COMPARED and k in ck["cfg"] and norm(ck["cfg"][k]) != norm(getattr(cfg, k))}
    if diff:
        raise ValueError("config mismatch between the checkpoint and the model this argv builds (checkpoint vs built; " + " and ".join(WM_ONLY_NOT_COMPARED)''',
    '''    skip = ("hmax",) if init_wm else WM_ONLY_NOT_COMPARED                      # Amendment 2: a fine-tune of the checkpoint's own WM compares its strides too (only hmax, which only sizes the sequence, is exempt)
    diff = {k: (norm(ck["cfg"][k]), norm(getattr(cfg, k))) for k in INIT_CKPT_MATCH if k not in skip and k in ck["cfg"] and norm(ck["cfg"][k]) != norm(getattr(cfg, k))}
    if diff:
        raise ValueError("config mismatch between the checkpoint and the model this argv builds (checkpoint vs built; " + " and ".join(skip)''')
rep('''"mode": "wm_only",
           "hmax_ckpt"''', '''"mode": "wm_only_init_wm" if init_wm else "wm_only", "init_wm": bool(init_wm),
           "hmax_ckpt"''')
rep('''    rest = {k: v for k, v in sd.items() if not is_wm_param(k)}
    res = model.load_state_dict(rest, strict=False)''', '''    if init_wm:                                                                # Amendment 2: EVERY tensor of the checkpoint, wm.* included, strictly; read back
        res = model.load_state_dict(sd, strict=False)
        if res.unexpected_keys or res.missing_keys:
            raise ValueError(f"--wm-init-from-ckpt-wm: the checkpoint and the built model disagree on keys: missing {sorted(res.missing_keys)[:6]}, unexpected {sorted(res.unexpected_keys)[:6]}")
        cur = model.state_dict()
        rec["n_mismatch_after_load"] = sum(1 for k in sd if not torch.equal(cur[k].detach().cpu(), sd[k].detach().cpu()))
        old = {k: v for k, v in sd.items() if is_wm_param(k)}
        rec["wm_loaded_from_ckpt"] = len(old)
        rec["wm_params_equal_to_ckpt"] = sum(1 for n, q in model.named_parameters() if is_wm_param(n) and n in old and torch.equal(old[n].detach().cpu(), q.detach().cpu()))
        rec["n_wm_params"] = sum(1 for n, _ in model.named_parameters() if is_wm_param(n))
        rec["proj_sha256"] = hashlib.sha256(b"".join(sd[k].detach().cpu().contiguous().numpy().tobytes() for k in sorted(sd) if is_wm_proj_param(k))).hexdigest()
        return rec
    rest = {k: v for k, v in sd.items() if not is_wm_param(k)}
    res = model.load_state_dict(rest, strict=False)''')
# call sites
rep('info["init_ckpt"] = info["wm_only"] = load_wm_only_init(a.init_ckpt, model, cfg, a.seed, apply=_apply)',
    'info["init_ckpt"] = info["wm_only"] = load_wm_only_init(a.init_ckpt, model, cfg, a.seed, apply=_apply, init_wm=(a.wm_init_from_ckpt_wm == "on"))')
rep("        freeze_all_but_wm(model)                                     # PREREG_V8F_MRATE:",
    "        freeze_all_but_wm(model, a.wm_freeze_proj == \"on\")          # PREREG_V8F_MRATE (Amendment 2: + wm.proj when --wm-freeze-proj on):")
open(p, 'w', encoding='utf-8', newline='').write(s)
print("ok")
