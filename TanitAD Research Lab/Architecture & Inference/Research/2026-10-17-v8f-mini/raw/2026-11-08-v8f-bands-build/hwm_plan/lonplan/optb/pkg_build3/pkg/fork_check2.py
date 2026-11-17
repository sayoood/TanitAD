import re
BS = chr(92)
p = "optb_check.py"
s = open(p, encoding="utf-8").read()
assert "STAGE2_APPLIED" not in s


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


# ---- init path: the R35 checkpoint under D:\tanitad\optb\inputs
sub('M2B_DECL + rf"' + BS + 'M2b_s{init}' + BS + 'ckpt_last.pt"', 'INPUTS_DECL + rf"' + BS + 'ladder_R35_s{init}.pt"')
sub("M2B_DECL + rf\"" + BS + "M2b_s{spec['init']}" + BS + "ckpt_last.pt\"", "INPUTS_DECL + rf\"" + BS + "ladder_R35_s{spec['init']}.pt\"")
sub("M2B_DECL + rf\"" + BS + "M2b_s{TABLE[base_tag(tag)]['init']}" + BS + "ckpt_last.pt\"", "INPUTS_DECL + rf\"" + BS + "ladder_R35_s{TABLE[base_tag(tag)]['init']}.pt\"")
sub("start state M2b_s{TABLE[base_tag(tag)]['init']} missing", "start state ladder_R35_s{TABLE[base_tag(tag)]['init']}.pt missing")
sub('t = tokens("RSP_s0")', 't = tokens("FTRSP_s0")')

# ---- readback block: the record is present on every arm (NULL arms name freeze + init)
i0 = s.index('    rr = lv.get("wm_rsp_readback")')
i1 = s.index('    eq("levers.wm_idm_gamma_readback')
blk = '''    rr = lv.get("wm_rsp_readback")
    if not isinstance(rr, dict):
        p.append("levers.wm_rsp_readback is missing: the Option B levers (readout, freeze, init) were not read back from the built objects")
    else:
        eq("wm_rsp_readback.freeze_proj_declared", rr.get("freeze_proj_declared"), "on")
        eq("wm_rsp_readback.wm_proj_requires_grad (wm.proj frozen in the BUILT model)", rr.get("wm_proj_requires_grad"), [False])
        eq("wm_rsp_readback.init_from_ckpt_wm_declared", rr.get("init_from_ckpt_wm_declared"), "on")
        eq("wm_rsp_readback.hmax", rr.get("hmax"), HMAX)
        if lever == "rsp":
            eq("wm_rsp_readback.declared", rr.get("declared"), RSP_W)
            eq("wm_rsp_readback.built_config (the LevelConfig the loss multiplies by)", rr.get("built_config"), RSP_W)
            eq("wm_rsp_readback.declared_readout", rr.get("declared_readout"), "fixed_k")
            eq("wm_rsp_readback.built_readout (the BUILT level-1 config)", rr.get("built_readout"), "fixed_k")
            b = rr.get("built") if isinstance(rr.get("built"), dict) else {}
            w = b.get("w_in_loss_rsp")
            if not (finite(w) and abs(w - RSP_W) <= 1e-6):
                p.append(f"BUILT loss gives the response term the weight {w!r}, wanted {RSP_W} (1e-6)")
            for k, v in (("calls", 1), ("contrasts", 2), ("k", 10), ("ctx_passed", 1), ("anchors", 9), ("readout", "fixed_k")):
                eq(f"wm_rsp_readback.built.{k} (AMEND_OPTB geometry at hmax 91)", b.get(k), v)
            if not (finite(b.get("s_cal")) and abs(b["s_cal"] - S_CAL[init]) <= 1e-5):
                p.append(f"BUILT slope s_cal = {b.get('s_cal')!r}, wanted {S_CAL[init]} (1e-5)")
        else:
            eq("wm_rsp_readback.declared (NULL arm: term off)", rr.get("declared"), 0.0)
            eq("wm_rsp_readback.built_readout (NULL arm)", rr.get("built_readout"), "idm")
            bw = (rr.get("built") or {}).get("w_in_loss_rsp")
            if bw not in (None, 0.0):
                p.append(f"a NULL arm's BUILT loss carries a response weight {bw!r}")
'''
s = s[:i0] + blk + s[i1:]

# ---- trainable parameters: the WM minus wm.proj
old = '''    if not (wo.get("n_wm") == n and wo.get("n_trainable") == n and lv.get("n_params_trainable") == n and pr.get("WM") == n):
        p.append(f"BUILT WM / trainable parameters = {wo.get('n_wm')!r}/{wo.get('n_trainable')!r}/{lv.get('n_params_trainable')!r} (params.WM {pr.get('WM')!r}), wanted {n} for all (R35's count: neither lever adds a parameter)")'''
new = '''    nt = wo.get("n_trainable")
    if not (wo.get("n_wm") == n and finite(nt) and 0 < nt < n and lv.get("n_params_trainable") == nt and pr.get("WM") == nt):
        p.append(f"BUILT WM / trainable parameters = {wo.get('n_wm')!r}/{nt!r}/{lv.get('n_params_trainable')!r} (params.WM {pr.get('WM')!r}); wanted n_wm {n} and a trainable count below it (wm.proj frozen), equal at all three reads. The exact count is UNVERIFIED: pin it from the pilot selfcheck")'''
sub(old, new)

# ---- wm_only record: init_wm True, source = the R35 checkpoint
i0 = s.index('    rec = info.get("wm_only") if isinstance(info.get("wm_only"), dict) else {}\n    for k, v in (("sha256"')
i1 = s.index("    # ---- G-LIVE")
rec = '''    rec = info.get("wm_only") if isinstance(info.get("wm_only"), dict) else {}
    for k, v in (("sha256", init_sha(init)), ("applied", True), ("n_mismatch_after_load", 0), ("mode", "wm_only_init_wm"), ("init_wm", True)):
        eq(f"info.wm_only.{k} (the fine-tune starts from the chassis R35 checkpoint, wm.* included)", rec.get(k), v)
    want_ic = INPUTS_DECL + rf"''' + BS + '''ladder_R35_s{init}.pt"
    if ntpath.normpath(str(rec.get("path", "")).replace("/", "''' + BS + BS + '''")).lower() != ntpath.normpath(want_ic).lower():
        p.append(f"info.wm_only.path {rec.get('path')!r} is not {want_ic}")
    if rec.get("wm_reinit_seed") is not None:
        p.append(f"info.wm_only.wm_reinit_seed = {rec.get('wm_reinit_seed')!r}: the WM must not be re-initialised")
'''
s = s[:i0] + rec + s[i1:]

# ---- weights identity: against the R35 checkpoint
s = s.replace("bit-identical to the M2b start state", "bit-identical to the R35 start checkpoint").replace("differ from M2b's", "differ from R35's")

# ---- C5: 141 of 145 wm.* Adam tensors, wm.proj bit-equal to the init checkpoint
old = '''    ev2, fp2 = adam_state_problems(b)
    ev.update(ev2)
    p += fp2
'''
new = '''    ev2, fp2 = adam_state_problems(b)
    ev.update(ev2)
    p += fp2
    if ev2.get("n_group_params") != N_WM_TENSORS - N_PROJ_TENSORS or ev2.get("n_with_state") != N_WM_TENSORS - N_PROJ_TENSORS:
        p.append(f"C5: {ev2.get('n_with_state')!r} of {ev2.get('n_group_params')!r} optimiser tensors have Adam state, wanted {N_WM_TENSORS - N_PROJ_TENSORS} of {N_WM_TENSORS - N_PROJ_TENSORS} (the {N_WM_TENSORS} wm.* tensors minus the {N_PROJ_TENSORS} frozen wm.proj tensors)")
    pk = sorted(k for k in a["model"] if k.startswith("wm.proj."))
    if len(pk) != N_PROJ_TENSORS:
        p.append(f"the init checkpoint holds {len(pk)} wm.proj tensors, wanted {N_PROJ_TENSORS}")
    pd = [k for k in pk if k not in b["model"] or not same_bits(a["model"][k], b["model"][k])]
    ev["wm_proj_tensors_equal"] = len(pk) - len(pd)
    if pd:
        p.append(f"wm.proj moved although frozen: {len(pd)} of {len(pk)} tensors differ from the init checkpoint (first: {pd[:3]})")
'''
sub(old, new)
s = s.replace("# ---- frozen identities", "# STAGE2_APPLIED\n# ---- frozen identities", 1)
open(p, "w", encoding="utf-8").write(s)
print("stage2 ok")
