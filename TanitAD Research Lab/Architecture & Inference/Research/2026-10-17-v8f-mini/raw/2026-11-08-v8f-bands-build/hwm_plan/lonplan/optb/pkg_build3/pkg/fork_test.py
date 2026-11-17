BS = chr(92)
src = open('F:/TanitAD-scratch/actrsp/launch/pkg/tests/test_actrsp_check.py', encoding='utf-8', newline='').read().replace('\r\n', '\n')


def sub(a, b, c=1):
    global src
    assert src.count(a) == c, (src.count(a), a[:70])
    src = src.replace(a, b)


sub('actrsp_check as CK', 'optb_check as CK')
sub('Unit tests of actrsp_check.py', 'Unit tests of optb_check.py (ACTRSP Option B)')
sub('ic = "D:' + BS * 2 + 'tanitad' + BS * 2 + 'runs' + BS * 2 + 'v8f_mini_full' + BS * 2 + 'M2b_s%d' + BS * 2 + 'ckpt_last.pt" % init',
    'ic = "D:' + BS * 2 + 'tanitad' + BS * 2 + 'optb' + BS * 2 + 'inputs' + BS * 2 + 'ladder_R35_s%d.pt" % init')
sub('n = CK.WM_PARAMS_BASE\n', 'n = CK.WM_PARAMS_BASE\n    nt = n - 4000      # trainable = WM minus wm.proj (exact count UNVERIFIED until the pilot selfcheck)\n')
sub('"n_params_trainable": n,', '"n_params_trainable": nt,')
sub('"params": {"adapter": 0, "cond": 0, "S": 0, "T": 0, "O": 0, "WM": n}', '"params": {"adapter": 0, "cond": 0, "S": 0, "T": 0, "O": 0, "WM": nt}')
sub('"n_wm": n, "n_trainable": n}', '"n_wm": n, "n_trainable": nt}')
sub('"mode": "wm_only", "source_arm": "M2b", "source_seed": init, "wm_reinit_seed": seed}', '"mode": "wm_only_init_wm", "init_wm": True}')
sub('''    if rsp:
        terms["wm/rsp/L1"] = 0.4''', '''    lv["wm_rsp_readback"] = {"declared": 0.0, "built_config": 0.0, "hmax": 91, "declared_readout": "idm", "built_readout": "idm", "built": {"w_in_loss_rsp": 0.0},
                             "freeze_proj_declared": "on", "wm_proj_requires_grad": [False], "init_from_ckpt_wm_declared": "on"}
    if rsp:
        terms["wm/rsp/L1"] = 0.4''')
sub('lv["wm_rsp_readback"] = {"declared": 1.0, "built_config": 1.0, "hmax": 91, "built": {"calls": 1, "contrasts": 4, "anchors": 9, "k": 10, "ctx_passed": 1, "w_in_loss_rsp": 1.0, "rsp_term": 0.4}}',
    'lv["wm_rsp_readback"].update({"declared": 1.0, "built_config": 1.0, "declared_readout": "fixed_k", "built_readout": "fixed_k", "built": {"calls": 1, "contrasts": 2, "anchors": 9, "k": 10, "ctx_passed": 1, "w_in_loss_rsp": 1.0, "rsp_term": 0.4, "readout": "fixed_k", "s_cal": CK.S_CAL[init]}})')
for a, b in (('"RSP_s0"', '"FTRSP_s0"'), ('"RSP_s2"', '"FTRSP_s2"'), ('"NULL_s4"', '"FTNULL_s4"'), ('"RSP_s1"', '"FTRSP_s1"'), ('"RSP_s3"', '"FTRSP_s3"'), ('"NULL_s5"', '"FTNULL_s5"')):
    src = src.replace(a, b)
sub('check("run order: RSP_s0, RSP_s2, NULL_s4, RSP_s1, RSP_s3, NULL_s5", CK.RUN_ORDER == ["FTRSP_s0", "FTRSP_s2", "FTNULL_s4", "FTRSP_s1", "FTRSP_s3", "FTNULL_s5"])',
    'check("run order: chassis 0 (FTRSP_s0, FTNULL_s0, FTRSP_s2, FTNULL_s2, FTNULL_s4) then chassis 1", CK.RUN_ORDER == ["FTRSP_s0", "FTNULL_s0", "FTRSP_s2", "FTNULL_s2", "FTNULL_s4", "FTRSP_s1", "FTNULL_s1", "FTRSP_s3", "FTNULL_s3", "FTNULL_s5"])\n    check("ten arms", len(CK.TABLE) == 10)')
sub('check("a NULL arm carrying wm_rsp_readback is refused", planted("FTNULL_s4", lambda sc: sc["levers"].update(wm_rsp_readback={"declared": 0.0})) != [])',
    'check("a NULL arm WITHOUT wm_rsp_readback is refused (it names freeze + init)", planted("FTNULL_s4", lambda sc: sc["levers"].pop("wm_rsp_readback")) != [])\n    check("a NULL arm whose readback shows the fixed_k readout is refused", planted("FTNULL_s4", lambda sc: sc["levers"]["wm_rsp_readback"].update(built_readout="fixed_k")) != [])')
sub('("anchors", 8), ("k", 5), ("contrasts", 2), ("ctx_passed", 3), ("calls", 2)', '("anchors", 8), ("k", 5), ("contrasts", 4), ("ctx_passed", 3), ("calls", 2), ("readout", "idm")')
sub('''wrong wm parameter count (a lever that adds a parameter) is refused", planted("FTRSP_s0", lambda sc: sc["levers"]["wm_only"].update(n_wm=CK.WM_PARAMS_BASE + 1)) != [])''',
    '''wrong wm parameter count (a lever that adds a parameter) is refused", planted("FTRSP_s0", lambda sc: sc["levers"]["wm_only"].update(n_wm=CK.WM_PARAMS_BASE + 1)) != [])
    # Option B levers
    check("wm.proj trainable in the BUILT model is refused", planted("FTRSP_s0", lambda sc: rb(sc, wm_proj_requires_grad=[True])) != [])
    check("mixed wm.proj requires_grad is refused", planted("FTRSP_s0", lambda sc: rb(sc, wm_proj_requires_grad=[False, True])) != [])
    check("freeze declared off is refused", planted("FTNULL_s0", lambda sc: rb(sc, freeze_proj_declared="off")) != [])
    check("init-from-ckpt-wm declared off is refused", planted("FTRSP_s0", lambda sc: rb(sc, init_from_ckpt_wm_declared="off")) != [])
    check("built readout idm on an RSP arm is refused", planted("FTRSP_s0", lambda sc: rb(sc, built_readout="idm")) != [])
    check("built slope off by 1e-3 is refused", planted("FTRSP_s0", lambda sc: sc["levers"]["wm_rsp_readback"]["built"].update(s_cal=CK.S_CAL[0] + 1e-3)) != [])
    check("trainable count equal to the full WM (nothing frozen) is refused", planted("FTRSP_s0", lambda sc: sc["levers"]["wm_only"].update(n_trainable=CK.WM_PARAMS_BASE)) != [])
    check("a trainable count that differs between the reads is refused", planted("FTRSP_s0", lambda sc: sc["levers"].update(n_params_trainable=CK.WM_PARAMS_BASE - 1)) != [])
    check("wm_only record from the old re-init mode is refused", planted("FTRSP_s0", lambda sc: sc["info"]["wm_only"].update(mode="wm_only", init_wm=False)) != [])
    check("a wm_reinit_seed in the record is refused", planted("FTRSP_s0", lambda sc: sc["info"]["wm_only"].update(wm_reinit_seed=0)) != [])
    check("the init checkpoint path of the wrong chassis is refused", planted("FTRSP_s0", lambda sc: sc["info"]["wm_only"].update(path=sc["info"]["wm_only"]["path"].replace("s0", "s1"))) != [])
    check("a wrong init sha256 is refused", planted("FTRSP_s0", lambda sc: sc["info"]["wm_only"].update(sha256="0" * 64)) != [])
    check("chassis-1 arm accepts its own slope and refuses chassis 0's", planted("FTRSP_s1", lambda sc: None) == [] and planted("FTRSP_s1", lambda sc: sc["levers"]["wm_rsp_readback"]["built"].update(s_cal=CK.S_CAL[0])) != [])''')
sub('check(f"{t}: no ACTSENS lever flag", "--wm-idm-gamma" not in toks and "--wm-sampler" not in toks)',
    'check(f"{t}: no ACTSENS lever flag", "--wm-idm-gamma" not in toks and "--wm-sampler" not in toks)\n        check(f"{t}: freeze + init named on", toks[toks.index("--wm-freeze-proj") + 1] == "on" and toks[toks.index("--wm-init-from-ckpt-wm") + 1] == "on")\n        check(f"{t}: RSP readout flags named only on RSP arms", all((f in toks) == (lev == "rsp") for f in ("--wm-rsp-readout", "--wm-rsp-probe", "--wm-rsp-probe-sha256")))')
sub('"actrsp_check.py"', '"optb_check.py"')
import os
os.makedirs('tests', exist_ok=True)
open('tests/test_optb_check.py', 'w', encoding='utf-8', newline='\n').write(src)
print('ok')
