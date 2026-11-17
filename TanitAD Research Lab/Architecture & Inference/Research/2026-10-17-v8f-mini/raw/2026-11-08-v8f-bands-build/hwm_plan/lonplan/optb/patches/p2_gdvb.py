import sys
p = sys.argv[1]
s = open(p, encoding='utf-8', newline='').read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old))
    s = s.replace(old, new)


# --- bad_wm_only: trainable set and the loader record, both modes
rep('''    names = [n for n, p in model.named_parameters() if p.requires_grad]
    wm_names = [n for n, _ in model.named_parameters() if is_wm_param(n)]
    if not wm_names or sorted(names) != sorted(wm_names):''', '''    names = [n for n, p in model.named_parameters() if p.requires_grad]
    wm_names = [n for n, _ in model.named_parameters() if is_wm_param(n) and not (a.wm_freeze_proj == "on" and is_wm_proj_param(n))]      # Amendment 2: --wm-freeze-proj on removes wm.proj from the trainable set
    if not wm_names or sorted(names) != sorted(wm_names):''')
rep('''    if rec.get("applied"):
        if rec.get("wm_loaded_from_ckpt") != 0:''', '''    if bool(rec.get("init_wm")) != (a.wm_init_from_ckpt_wm == "on"):
        bad.append(f"lever wm_init_from_ckpt_wm: declared {a.wm_init_from_ckpt_wm} but the loader record says init_wm {rec.get('init_wm')!r}")
    if rec.get("applied") and a.wm_init_from_ckpt_wm == "on":          # Amendment 2: a fine-tune: every tensor loaded and read back equal, no WM re-initialisation
        n_wm = sum(1 for n, _ in model.named_parameters() if is_wm_param(n))
        if rec.get("n_mismatch_after_load") != 0:
            bad.append(f"lever wm_init_from_ckpt_wm: after the load {rec.get('n_mismatch_after_load')} of {rec.get('n_tensors')} tensors differ from the checkpoint's")
        if not (rec.get("wm_loaded_from_ckpt") == n_wm and rec.get("wm_params_equal_to_ckpt") == n_wm and n_wm > 0):
            bad.append(f"lever wm_init_from_ckpt_wm: {rec.get('wm_loaded_from_ckpt')} WM tensors loaded and {rec.get('wm_params_equal_to_ckpt')} equal the checkpoint's, the built WM has {n_wm} parameters")
        if rec.get("wm_reinit_seed") is not None:
            bad.append("lever wm_init_from_ckpt_wm: on, but the WM was re-initialised under a seed")
        if rec.get("mode") != "wm_only_init_wm":
            bad.append(f"lever wm_init_from_ckpt_wm: the start state was loaded in mode {rec.get('mode')!r}")
    elif rec.get("applied"):
        if rec.get("wm_loaded_from_ckpt") != 0:''')

# --- bad_wm_rsp: readout
rep('''    if lv["wm"] == "off" or model.wm is None:
        return bad
    levers = model.wm.stack.built_levers()''', '''    ro = a.wm_rsp_readout                                                       # Amendment 2: the Option B flags are refused when they would do nothing
    if ro == "fixed_k":
        if not w > 0.0:
            bad.append("lever wm_rsp_readout: fixed_k needs --wm-rsp-w > 0 (a readout with no term builds nothing)")
        if a.wm_freeze_proj != "on":
            bad.append("lever wm_rsp_readout: fixed_k needs --wm-freeze-proj on (the readout is valid in the checkpoint latent space only)")
        if a.wm_init_from_ckpt_wm != "on":
            bad.append("lever wm_rsp_readout: fixed_k needs --wm-init-from-ckpt-wm on (the readout is valid for the checkpoint world model, not a re-initialised one)")
        if not (a.wm_rsp_probe and len(a.wm_rsp_probe_sha256) == 64):
            bad.append("lever wm_rsp_readout: fixed_k needs --wm-rsp-probe and a 64-hex --wm-rsp-probe-sha256")
    elif a.wm_rsp_probe or a.wm_rsp_probe_sha256:
        bad.append("lever wm_rsp_readout: --wm-rsp-probe / --wm-rsp-probe-sha256 are named but the readout is idm: a lever that is declared and does nothing")
    if lv["wm"] == "off" or model.wm is None:
        return bad
    if cfg.wm_rsp_readout != ro:
        bad.append(f"lever wm_rsp_readout: declared {ro} but the built config holds {cfg.wm_rsp_readout}")
    levers = model.wm.stack.built_levers()''')
rep('''    anchors = len(range(0, cfg.hmax - 10, 10))
    for k, want in (("calls", 1), ("contrasts", 4), ("k", 10), ("ctx_passed", 1), ("anchors", anchors)):
        if r.get(k) != want:
            bad.append(f"lever wm_rsp_w: the BUILT term's {k} = {r.get(k)!r}, the card fixes {want} (hmax {cfg.hmax})")
    return bad''', '''    anchors = len(range(0, cfg.hmax - 10, 10))
    for k, want in (("calls", 1), ("contrasts", 2 if ro == "fixed_k" else 4), ("k", 10), ("ctx_passed", 1), ("anchors", anchors)):
        if r.get(k) != want:
            bad.append(f"lever wm_rsp_w: the BUILT term's {k} = {r.get(k)!r}, the card fixes {want} (hmax {cfg.hmax}, readout {ro})")
    built_ro = row.get("rsp_readout", "idm")
    if built_ro != ro or r.get("readout") != ro:
        bad.append(f"lever wm_rsp_readout: declared {ro} but the BUILT LevelConfig holds {built_ro!r} and the BUILT loss reads through {r.get('readout')!r}")
    if ro == "fixed_k":                                                         # the live buffers equal the FILE the argv names (re-read here, sha256 checked by the loader)
        try:
            w_file, s_file = M.load_rsp_probe(a.wm_rsp_probe, a.wm_rsp_probe_sha256, cfg.wm_dim)
        except Exception as e:                                                  # noqa: BLE001
            return bad + [f"lever wm_rsp_readout: the probe file does not load: {type(e).__name__}: {str(e)[:300]}"]
        if r.get("w_sha256") != hashlib.sha256(w_file.numpy().tobytes()).hexdigest():
            bad.append("lever wm_rsp_readout: the BUILT readout weights are not the probe file's")
        if not (isinstance(r.get("s_cal"), float) and abs(r["s_cal"] - s_file) <= 1e-12):
            bad.append(f"lever wm_rsp_readout: the BUILT slope {r.get('s_cal')!r} is not the probe file's {s_file!r}")
    return bad


def bad_wm_freeze(a, cfg, model, data_info) -> list:
    """PREREG_V8F_ACTRSP Amendment 2, G-DVB for ``--wm-freeze-proj``: read the frozen set back from the BUILT model's ``requires_grad``. ``on``: needs ``--wm-only``; every ``wm.proj.*`` tensor is frozen and every other ``wm.*`` tensor trains; with
    ``--wm-init-from-ckpt-wm on`` the live ``wm.proj`` equals the checkpoint's (sha256 of the loader record). ``off``: no ``wm.proj`` tensor is frozen."""
    bad = []
    on = a.wm_freeze_proj == "on"
    if on and not a.wm_only:
        bad.append("lever wm_freeze_proj: on needs --wm-only: a lever that is declared and does nothing")
    if model.wm is None:
        return bad
    proj = [(n, p) for n, p in model.named_parameters() if is_wm_proj_param(n)]
    if not proj:
        return bad + (["lever wm_freeze_proj: the built model has no wm.proj parameter"] if on else [])
    for n, p in proj:
        if p.requires_grad == on:
            bad.append(f"lever wm_freeze_proj: declared {a.wm_freeze_proj} but {n} requires_grad is {p.requires_grad}")
    if on and a.wm_only:
        stray = [n for n, p in model.named_parameters() if is_wm_param(n) and not is_wm_proj_param(n) and not p.requires_grad]
        if stray:
            bad.append(f"lever wm_freeze_proj: WM tensors other than wm.proj are frozen: {stray[:4]}")
        rec = (data_info or {}).get("wm_only") or {}
        if rec.get("applied") and rec.get("proj_sha256"):
            sd = model.state_dict()
            now = hashlib.sha256(b"".join(sd[k].detach().cpu().contiguous().numpy().tobytes() for k in sorted(sd) if is_wm_proj_param(k))).hexdigest()
            if now != rec["proj_sha256"]:
                bad.append("lever wm_freeze_proj: the live wm.proj is not the checkpoint's")
    return bad''')
rep("    bad += bad_wm_rsp(a, cfg, model)    ", "    bad += bad_wm_freeze(a, cfg, model, data_info)                                              # PREREG_V8F_ACTRSP Amendment 2: --wm-freeze-proj read back from requires_grad\n    bad += bad_wm_rsp(a, cfg, model)    ")
# --- readback record
rep('''            "built": None if model.wm is None else model.wm.built_rsp(cfg.hmax), "hmax": cfg.hmax}''', '''            "built": None if model.wm is None else model.wm.built_rsp(cfg.hmax), "hmax": cfg.hmax,
            "declared_readout": a.wm_rsp_readout, "built_readout": None if st is None else st.built_levers()[1].get("rsp_readout", "idm"),
            "freeze_proj_declared": a.wm_freeze_proj, "wm_proj_requires_grad": sorted({bool(p.requires_grad) for n, p in model.named_parameters() if is_wm_proj_param(n)}),
            "init_from_ckpt_wm_declared": a.wm_init_from_ckpt_wm}''')
rep('    if a.wm_rsp_w != 0.0:                                                # PREREG_V8F_ACTRSP: lever-NAMED',
    '    if a.wm_rsp_w != 0.0 or a.wm_freeze_proj != "off" or a.wm_init_from_ckpt_wm != "off":   # Amendment 2: the NULL arms name freeze / init without the term and carry the record too. PREREG_V8F_ACTRSP: lever-NAMED')
open(p, 'w', encoding='utf-8', newline='').write(s)
print("ok")
