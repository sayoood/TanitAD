"""Edits of stack/scripts/v8f_mini_train.py for LEADSUP (run once on tree_patch; argv[1] = tree root). Each edit must match exactly once."""
import sys

root = sys.argv[1]
p = f"{root}/stack/scripts/v8f_mini_train.py"
s = open(p, encoding="utf-8", newline="").read()
assert "\r\n" not in s


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


# 1. FLAG_CHECKS
rep('''    "wm_l1_act": "lever:wm_l1_act", "wm_l1_futtok_k": "lever:wm_l1_futtok_k", "w_wm_futtok": "lever:w_wm_futtok",''',
    '''    "lead_sup": "lever:lead_sup", "w_lead_sup": "lever:w_lead_sup", "lead_hist": "lever:lead_hist", "lead_sup_detach": "lever:lead_sup_detach",     # LEADSUP: read back from the BUILT head (see bad_lead_sup)
    "wm_l1_act": "lever:wm_l1_act", "wm_l1_futtok_k": "lever:wm_l1_futtok_k", "w_wm_futtok": "lever:w_wm_futtok",''')

# 2. argparse
rep('''    ap.add_argument("--w-wm-futtok", dest="w_wm_futtok", type=float, default=0.0,''', '''    ap.add_argument("--lead-sup", dest="lead_sup", choices=M.LEAD_SUP_MODES, default="off",
                    help="LEADSUP: on = a supervised lead-vehicle head (presence logit, gap / 50 m, closing speed / 10 m/s) reads the level-1 grid states of the last --lead-hist frames; trained with --w-lead-sup x "
                         "(BCE + Huber + Huber) from privileged labels built from the label join (training only; the driving path never reads the head). Needs --wm-state grid and --w-lead-sup > 0. off (default) builds nothing")
    ap.add_argument("--w-lead-sup", dest="w_lead_sup", type=float, default=0.0, help="LEADSUP: the weight of the lead_sup term in the total. 0.0 (default) with --lead-sup off; must be > 0 with it on (a head with no loss is refused)")
    ap.add_argument("--lead-hist", dest="lead_hist", type=int, default=5, help="LEADSUP: consecutive frames (ending at t) whose level-1 states the head reads (default 5)")
    ap.add_argument("--lead-sup-detach", dest="lead_sup_detach", choices=M.LEAD_SUP_DETACH_MODES, default="off",
                    help="LEADSUP: on = the head reads state.detach(): the lead_sup term then reaches no parameter outside the head (the zero control / comparison arm). Needs --lead-sup on")
    ap.add_argument("--w-wm-futtok", dest="w_wm_futtok", type=float, default=0.0,''')

# 3. make_config
rep('''wm_l1_futtok_k=a.wm_l1_futtok_k, w_wm_futtok=a.w_wm_futtok, wm_dim=a.wm_dim,''',
    '''wm_l1_futtok_k=a.wm_l1_futtok_k, w_wm_futtok=a.w_wm_futtok, lead_sup=a.lead_sup, w_lead_sup=a.w_lead_sup, lead_hist=a.lead_hist, lead_sup_detach=a.lead_sup_detach, wm_dim=a.wm_dim,''')

# 4. INIT_CKPT_MATCH + load_init_ckpt defaults
rep('''"wm_l1_act", "wm_l1_futtok_k", "w_wm_futtok",        # TARGET (EGOV / FUTTOK, same rule);''',
    '''"wm_l1_act", "wm_l1_futtok_k", "w_wm_futtok", "lead_sup", "w_lead_sup", "lead_hist", "lead_sup_detach",        # TARGET (EGOV / FUTTOK, same rule); LEADSUP (an absent key = the default);''')
rep('''*M.MOTION_CFG_DEFAULTS.items(), *M.TARGET_CFG_DEFAULTS.items()):''', '''*M.MOTION_CFG_DEFAULTS.items(), *M.TARGET_CFG_DEFAULTS.items(), *M.LEADSUP_CFG_DEFAULTS.items()):''')

# 5. checkpoint cfg
rep('''and (k not in M.TARGET_CFG_DEFAULTS or v != M.TARGET_CFG_DEFAULTS[k])}''', '''and (k not in M.TARGET_CFG_DEFAULTS or v != M.TARGET_CFG_DEFAULTS[k]) and (k not in M.LEADSUP_CFG_DEFAULTS or v != M.LEADSUP_CFG_DEFAULTS[k])}''')

# 6. wm-off declared-and-built-nothing list
rep('''"wm_l1_diag", "wm_l1_act", "wm_l1_futtok_k", "w_wm_futtok"):
            if getattr(a, f) != d.get_default(f):''', '''"wm_l1_diag", "wm_l1_act", "wm_l1_futtok_k", "w_wm_futtok", "lead_sup", "w_lead_sup", "lead_hist", "lead_sup_detach"):
            if getattr(a, f) != d.get_default(f):''')

# 7. the gate function (before bad_wm_state)
rep('''def bad_wm_state(a, cfg, model) -> list:''', '''def bad_lead_sup(a, cfg, model, data_info=None) -> list:
    """LEADSUP, G-DVB: --lead-sup / --w-lead-sup / --lead-hist / --lead-sup-detach read back from the BUILT head and the BUILT dataset. Empty = pass. Refused: a head declared on that builds nothing (or built and not
    declared), a head with no loss (weight 0 with it on), a weight / history / detach with the head off, a history width that is not the declared one (read from the head's read-out layer), a detach wiring that
    is not the declared one, a non-grid state, the lora encoder (no cached tokens, no labels path), a dataset that does not carry the labels, and a label file whose hash is not recorded."""
    bad = []
    built = model.built_levers()
    if not (math.isfinite(a.w_lead_sup) and a.w_lead_sup >= 0.0):
        bad.append(f"w_lead_sup {a.w_lead_sup!r} is not a finite number >= 0")
    if a.lead_sup == "off":
        if a.w_lead_sup != 0.0 or a.lead_sup_detach != "off" or a.lead_hist != 5:
            bad.append(f"lead_sup off with w_lead_sup {a.w_lead_sup} / lead_sup_detach {a.lead_sup_detach} / lead_hist {a.lead_hist}: a lever that is declared and does nothing")
        if model.wm is not None and model.wm.lead_head is not None:
            bad.append("lead_sup off declared but a lead head was built")
        return bad
    if not a.w_lead_sup > 0.0:
        bad.append(f"lead_sup on declared but w_lead_sup is {a.w_lead_sup}: a head with no loss -- a lever that is declared and does nothing")
    if model.wm is None or model.wm.lead_head is None:
        bad.append("lead_sup on declared but no lead head was built")
        return bad
    for k, want in (("lead_sup", a.lead_sup), ("lead_hist", a.lead_hist), ("lead_sup_detach", a.lead_sup_detach), ("w_lead_sup", a.w_lead_sup)):
        if built.get(k) != want:
            bad.append(f"lever {k}: declared {want!r} but built {built.get(k)!r}")
    if a.wm_state != "grid":
        bad.append("lead_sup needs --wm-state grid")
    if a.encoder == "lora":
        bad.append("lead_sup is not built for --encoder lora (the labels ride the cached-token dataset)")
    ll = (data_info or {}).get("lead_sup_labels")
    if not (data_info or {}).get("ds_lead_sup") or not ll:
        bad.append("lead_sup on declared but the built dataset carries no lead_sup labels")
    else:
        if not (isinstance(ll.get("sha256"), str) and len(ll["sha256"]) == 64):
            bad.append(f"lead_sup labels: the sha256 of the label arrays is not recorded ({ll.get('sha256')!r})")
        if ll.get("n_pres", 0) <= 0 or ll.get("n_gap", 0) <= 0 or ll.get("n_clos", 0) <= 0:
            bad.append(f"lead_sup labels: presence {ll.get('n_pres')} / gap {ll.get('n_gap')} / closing {ll.get('n_clos')} labelled rows: a part of the head has nothing to learn from")
    return bad


def bad_wm_state(a, cfg, model) -> list:''')

# 8. call it from declared_vs_built
rep('''    bad += bad_lead(a, model, data_info)''', '''    bad += bad_lead_sup(a, cfg, model, data_info)                                                # LEADSUP: --lead-sup / --w-lead-sup / --lead-hist / --lead-sup-detach read back from the BUILT head and dataset
    bad += bad_lead(a, model, data_info)''')

# 9. G-LIVE expected terms (two places)
rep('''    if cfg.wm_l1_futtok_k > 0:
        expect.add("wm_futtok")                                          # TARGET lever FUTTOK: a declared token term that never appears is a launch blocker
''', '''    if cfg.wm_l1_futtok_k > 0:
        expect.add("wm_futtok")                                          # TARGET lever FUTTOK: a declared token term that never appears is a launch blocker
    if cfg.lead_sup == "on":
        expect |= {"lead_sup", "lead_sup_bce", "lead_sup_gap", "lead_sup_clos"}      # LEADSUP: the term and its three parts must appear (finite) in the log
''', count=2)

# 10. dataset
rep('''        ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s,
                              window_start=cfg.window_start)
    info["ds_window_start"] = ds.window_start ''', '''        ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s,
                              window_start=cfg.window_start, lead_sup=cfg.lead_sup == "on")
    info["ds_lead_sup"] = bool(getattr(ds, "lead_sup", False))      # LEADSUP: read back from the BUILT dataset (G-DVB compares it)
    if info["ds_lead_sup"]:
        info["lead_sup_labels"] = {k: ds.lead_sup_labels[k] for k in ("sha256", "n_rows", "n_pres", "n_gap", "n_clos")}
    info["ds_window_start"] = ds.window_start ''')

# 11. selfcheck info: the labelled frames the probe batch saw
rep('''        live = live_check(model.cpu(), probe, cfg)
    else:''', '''        if cfg.lead_sup == "on":                         # LEADSUP: the number of frames with a label that the G-LIVE probe batch carries (counted like the loss counts: all `lead_hist` frames behind the end frame are real)
            okw = probe["frame_ok"].unfold(1, cfg.lead_hist, 1).all(-1)
            sl = slice(cfg.lead_hist - 1, None)
            info["lead_sup_n_labelled"] = int((okw & torch.isfinite(probe["lead_gapm"][:, sl])).sum())
            info["lead_sup_n_presence"] = int(okw.sum())
            info["lead_sup_n_closing"] = int((okw & torch.isfinite(probe["lead_clos"][:, sl])).sum())
        live = live_check(model.cpu(), probe, cfg)
    else:''')

open(p, "w", encoding="utf-8", newline="").write(s)
print("OK", p)
