import sys
p = sys.argv[1]
s = open(p, encoding='utf-8', newline='').read()
assert '\r\n' not in s


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:80], s.count(old))
    s = s.replace(old, new)


# FLAG_CHECKS
rep('''    "wm_idm_gamma": "lever:wm_idm_gamma",
''', '''    "wm_idm_gamma": "lever:wm_idm_gamma",
    # PREREG_V8F_ACTRSP: `lever:wm_rsp_w` = the weight the BUILT loss really gives the level-1 action-response term (WMAux.built_rsp: the loss run once on noise, (level/L1 - every other level-1 term) / rsp/L1, never the config that built it) AND the term's
    # own geometry read off its rollout call (anchors, K, contrasts, context) AND HJEPAStack.built_levers()[1]["rsp_w"] -> bad_wm_rsp -> `selfcheck.json levers.wm_rsp_readback` (written when the argv NAMES the flag).
    "wm_rsp_w": "lever:wm_rsp_w",
''')
# argparse
rep('''    ap.add_argument("--wm-rollout-k", dest="wm_rollout_k",''', '''    ap.add_argument("--wm-rsp-w", dest="wm_rsp_w", type=float, default=0.0,
                    help="PREREG_V8F_ACTRSP: the weight of the level-1 action-response term (the WM's answer to +/- one sd of each action channel, read through the level-1 IDM head and matched to the identity gain). Default 0.0 = the term is not built = every earlier "
                         "arm, bit for bit (no key, no op, no parameter). The card fixes 1.0. Needs --wm b, --kin-mode yaw and --wm-only. Read back from the BUILT loss (HJEPAStack.built_rsp), not from the config")
    ap.add_argument("--wm-rollout-k", dest="wm_rollout_k",''')
# make_config
rep('''                           wm_gamma_levels=declared_wm_idm_gamma(a),                                      # PREREG_V8F_ACTSENS: the ONE line that applies --wm-idm-gamma (mutation M1 deletes it)
''', '''                           wm_gamma_levels=declared_wm_idm_gamma(a),                                      # PREREG_V8F_ACTSENS: the ONE line that applies --wm-idm-gamma (mutation M1 deletes it)
                           wm_rsp_w=float(a.wm_rsp_w),                                                    # PREREG_V8F_ACTRSP: the ONE line that applies --wm-rsp-w (mutation MR1 deletes it)
''')
# G-LIVE expect
rep('''    expect = {"wm", "total"} | {f"wm/{t}/L{l}" for t in ("pred", "sigreg", "idm", "level") for l in (1, 2, 3)} | {f"wm/act_sigreg/L{l}" for l in (2, 3)}
''', '''    expect = {"wm", "total"} | {f"wm/{t}/L{l}" for t in ("pred", "sigreg", "idm", "level") for l in (1, 2, 3)} | {f"wm/act_sigreg/L{l}" for l in (2, 3)}
    if cfg.wm_rsp_w > 0.0:                                            # PREREG_V8F_ACTRSP: the action-response term must be among the live terms when the lever is on
        expect |= {"wm/rsp/L1"}
''')
# readback + bad
rep('''def declared_vs_built(a, cfg, model, opt, data_info,''', '''def wm_rsp_readback(a, cfg, model) -> dict:
    """PREREG_V8F_ACTRSP: the record ``selfcheck.json`` carries (``levers.wm_rsp_readback``) when the argv NAMES ``--wm-rsp-w``: the weight the argv DECLARED, the BUILT LevelConfig's, and what the BUILT loss measures (``WMAux.built_rsp``)."""
    st = model.wm.stack if model.wm is not None else None
    return {"declared": float(a.wm_rsp_w), "built_config": None if st is None else float(st.built_levers()[1].get("rsp_w", 0.0)),
            "built": None if model.wm is None else model.wm.built_rsp(cfg.hmax), "hmax": cfg.hmax}


def bad_wm_rsp(a, cfg, model) -> list:
    """PREREG_V8F_ACTRSP, G-DVB for ``--wm-rsp-w``. Refuses: a non-finite or negative weight; a positive weight without ``--wm b``, ``--kin-mode yaw`` or ``--wm-only``; and any mismatch between the declared weight and what the BUILT objects say:
    ``HJEPAStack.built_levers()[1]["rsp_w"]`` (the LevelConfig the loss multiplies by) AND ``WMAux.built_rsp(hmax)["w_in_loss_rsp"]`` (the loss run once: (level/L1 - every other level-1 term) / rsp/L1) to 1e-6, and the term's own geometry
    (one rollout call, 4 contrasts, K = 10, context 1, ``len(range(0, hmax - 10, 10))`` anchors = 9 at hmax 91). Flag absent: the built stack must carry NO response term (no ``rsp_w`` key)."""
    bad = []
    lv = declared_levers(a)
    w = a.wm_rsp_w
    if not (isinstance(w, float) and w == w and abs(w) != float("inf") and w >= 0.0):
        return [f"lever wm_rsp_w: --wm-rsp-w must be a finite number >= 0, got {w!r}"]
    if w > 0.0:
        if lv["wm"] != "b":
            bad.append(f"lever wm_rsp_w: --wm-rsp-w {w} needs --wm b (got --wm {lv['wm']}): a lever that is declared and builds nothing")
        if a.kin_mode != "yaw":
            bad.append(f"lever wm_rsp_w: --wm-rsp-w {w} needs --kin-mode yaw (got {a.kin_mode}): the contrast sizes are WM units of the yaw-rate channel")
        if not a.wm_only:
            bad.append(f"lever wm_rsp_w: --wm-rsp-w {w} needs --wm-only (the term is a world-model objective on the frozen adapter)")
    if lv["wm"] == "off" or model.wm is None:
        return bad
    levers = model.wm.stack.built_levers()
    row = levers[1]
    if abs(float(cfg.wm_rsp_w) - w) > 1e-12:
        bad.append(f"lever wm_rsp_w: declared {w} but the built config holds {cfg.wm_rsp_w}")
    if w == 0.0:
        if any("rsp_w" in r for r in levers.values()):
            bad.append(f"lever wm_rsp_w: flag absent (0.0) but the BUILT stack carries a response term: {row}")
        return bad
    if abs(float(row.get("rsp_w", 0.0)) - w) > 1e-9 * max(1.0, abs(w)):
        bad.append(f"lever wm_rsp_w: declared {w} but the BUILT LevelConfig (the weight the loss multiplies by) holds {row.get('rsp_w', 0.0)}")
    r = model.wm.built_rsp(cfg.hmax)
    got = r.get("w_in_loss_rsp")
    if not (isinstance(got, float) and got == got and abs(got - w) <= 1e-6 * max(1.0, abs(w))):
        bad.append(f"lever wm_rsp_w: declared {w} but the BUILT loss gives the response term the weight {got!r} (measured: (level/L1 - every other level-1 term) / rsp/L1)")
    anchors = len(range(0, cfg.hmax - 10, 10))
    for k, want in (("calls", 1), ("contrasts", 4), ("k", 10), ("ctx_passed", 1), ("anchors", anchors)):
        if r.get(k) != want:
            bad.append(f"lever wm_rsp_w: the BUILT term's {k} = {r.get(k)!r}, the card fixes {want} (hmax {cfg.hmax})")
    return bad


def declared_vs_built(a, cfg, model, opt, data_info,''')
rep('''    bad += bad_wm_sampler(a, cfg, model, data_info)                                              # PREREG_V8F_DIVRS''', '''    bad += bad_wm_rsp(a, cfg, model)                                                             # PREREG_V8F_ACTRSP: --wm-rsp-w read back from the BUILT loss (HJEPAStack.built_rsp) and the BUILT LevelConfig
    bad += bad_wm_sampler(a, cfg, model, data_info)                                              # PREREG_V8F_DIVRS''')
rep('''    if wm_ladder_named(a):                                              # PREREG_V8F_LADDER: lever-NAMED launches only (a flag-off selfcheck.json keeps its key set): declared vs the BUILT measurements
        levers["wm_ladder_readback"]''', '''    if a.wm_rsp_w != 0.0:                                                # PREREG_V8F_ACTRSP: lever-NAMED launches only (a flag-off selfcheck.json keeps its key set): declared vs the BUILT weight and geometry
        levers["wm_rsp_readback"] = wm_rsp_readback(a, cfg, model)
    if wm_ladder_named(a):                                              # PREREG_V8F_LADDER: lever-NAMED launches only (a flag-off selfcheck.json keeps its key set): declared vs the BUILT measurements
        levers["wm_ladder_readback"]''')
rep('''    if a.wm_only and a.wm_idm_gamma is not None:                         # AMEND_ACTSENS_1: lever-NAMED launches only: G-LIVE step 0, the instrument of control C3''', '''    if a.wm_only and (a.wm_idm_gamma is not None or a.wm_rsp_w != 0.0):  # AMEND_ACTSENS_1: lever-NAMED launches only: G-LIVE step 0, the instrument of control C3 (ACTRSP: the same instrument for its C3)''')
open(p, 'w', encoding='utf-8', newline='').write(s)
print("patched", p)
