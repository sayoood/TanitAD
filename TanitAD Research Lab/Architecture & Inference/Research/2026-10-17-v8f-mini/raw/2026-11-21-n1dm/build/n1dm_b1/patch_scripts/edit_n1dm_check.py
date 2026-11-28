"""edit_n1dm_check.py -- anchored, counted edits: strat1_check.py kind N1DM (+ the literal-list census tests). Every edit asserts its anchor count."""
import sys
TREE = sys.argv[1]


def sub(rel, old, new, count=1):
    p = f"{TREE}/{rel}"
    s = open(p, encoding="utf-8", newline="").read()
    n = s.count(old)
    assert n == count, (rel, n, count, old[:100])
    open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))


CHK = "stack/scripts/strat1_check.py"
sub(CHK, '"N1DW4", "N1DW2", "N1DC", "N1DH")\n#: PREREG_V8F_NAVRB2', '"N1DW4", "N1DW2", "N1DC", "N1DH", "N1DM")\n#: PREREG_V8F_NAVRB2')
sub(CHK, '"N1DC": "N1", "N1DH": "N1"}', '"N1DC": "N1", "N1DH": "N1", "N1DM": "N1"}')
sub(CHK, 'H_WEIGHT = "0.5"\nDCHAN_KINDS = D_KINDS + W_KINDS + C_KINDS + H_KINDS\n',
    'H_WEIGHT = "0.5"\n'
    '#: PREREG_V8F_N1DM: the M kind = N1D\'s argv (the real 15-column channel, the same npz + sha) + exactly `--aux-route modes --w-aux-route 0.5` (route head + K = 3 zero-init residual plan branches selected softly by the detached posterior; no\n'
    '#: route_emb). It reads the D channel too, never names `--o-code-skip` or `--rb-window-w`; no existing kind\'s argv moves (N1DH still renders `cond`), and a non-M kind never builds route_br or carries the modes probe.\n'
    'M_KINDS = ("N1DM",)\n'
    'M_MODE = "modes"\n'
    'DCHAN_KINDS = D_KINDS + W_KINDS + C_KINDS + H_KINDS + M_KINDS\n')
sub(CHK, '''        rb += [H_FLAGS[0], "cond", H_FLAGS[1], H_WEIGHT]                       # PREREG_V8F_N1DH: the two flags of the H kind, appended after N1D's (an N1D render is untouched)
''', '''        rb += [H_FLAGS[0], "cond", H_FLAGS[1], H_WEIGHT]                       # PREREG_V8F_N1DH: the two flags of the H kind, appended after N1D's (an N1D render is untouched)
    if kind in M_KINDS:
        rb += [H_FLAGS[0], M_MODE, H_FLAGS[1], H_WEIGHT]                       # PREREG_V8F_N1DM: the two flags of the M kind (same flags as N1DH, the mode differs), appended after N1D's
''')
# an M kind is not read by the H gate
sub(CHK, '''    if kind not in H_KINDS:
        for fl in H_FLAGS:''', '''    if kind in M_KINDS:
        return bad                                                             # PREREG_V8F_N1DM: the M kind is read by m_gate_problems
    if kind not in H_KINDS:
        for fl in H_FLAGS:''')
NEW_FUNCS = '''def m_pairing_problems(kind, toks, ref_toks):
    """PREREG_V8F_N1DM pairing rule: the M kind's argv equals its paired N1D argv (the banked one of the same seed) PLUS exactly ``--aux-route modes --w-aux-route 0.5``, with its own --out and nothing else different: the same seed, steps,
    data, lr, nav flags, channel npz + sha, variant and feat. Neither argv names ``--o-code-skip`` or ``--rb-window-w``; the reference names no route flag. (N1DH's own argv, ``--aux-route cond``, is NOT an M reference.)"""
    if kind not in M_KINDS:
        return [f"{kind} is not an M kind"]
    bad, toks, ref = [], list(toks), list(ref_toks)
    for fl in (*H_FLAGS, C_FLAG, RB_W_FLAG):
        if fl in ref:
            bad.append(f"the reference (N1D) argv names {fl} (N1D's argv is the arm without it and does not)")
    for fl in (C_FLAG, RB_W_FLAG):
        if fl in toks:
            bad.append(f"the {kind} argv names {fl} (N1DM runs at that lever's default, which builds nothing)")
    if flag_value(ref, RB_FEAT_FLAG) != "exitdist":
        bad.append("the reference argv is not an N1D argv (--nav-rb-feat exitdist missing)")
    if toks.count(H_FLAGS[0]) != 1 or flag_value(toks, H_FLAGS[0]) != M_MODE or toks.count(H_FLAGS[1]) != 1 or flag_value(toks, H_FLAGS[1]) != H_WEIGHT:
        return bad + [f"the {kind} argv must name {H_FLAGS[0]} exactly once as {M_MODE} and {H_FLAGS[1]} exactly once as {H_WEIGHT}"]
    for fl in H_FLAGS:
        i = toks.index(fl)
        toks = toks[:i] + toks[i + 2:]
    if len(toks) != len(ref):
        return bad + [f"argv length {len(toks)} (without the two route flags) != the N1D argv's {len(ref)}"]
    diff = [(j, y, x) for j, (x, y) in enumerate(zip(toks, ref)) if x != y and not (j > 0 and toks[j - 1] == "--out" and ref[j - 1] == "--out")]
    if len(diff):
        bad.append(f"the {kind} argv departs from N1D's beyond the two route flags / out: {diff[:4]}")
    return bad


def m_gate_problems(kind, toks, sc):
    """PREREG_V8F_N1DM G-DVB / G-LIVE read of a selfcheck.json: the M kind names ``--aux-route modes`` and ``--w-aux-route 0.5`` once each (a flag whose default disables it must be named), builds them (levers.aux_route / w_aux_route), and
    carries the probe's controls: route_head.weight CE gradient > 0; EVERY one of the K branches a gradient > 0 at the real init AND with seeded weights; the plan moves when the branches are set AND when p is permuted across rows; the identity at
    the zero init (the plan and every loss term but route / total are the model-without-the-head's); and every exact-0.0 control (the decoder term on route_head: the detach; the T/S-side terms; the repeat; the prior; the batch with no own-code row);
    the ROUTE group's G-LIVE gradient and the ``route`` term. A non-M kind is read by the H gate, which refuses any aux_route flag, lever or probe on it."""
    bad = []
    lv, live = sc.get("levers", {}), sc.get("live", {})
    pr = sc.get("aux_route_probe") or {}
    if kind not in M_KINDS:
        return bad                                                             # a non-M kind: the H / D / W / C gates read it (the H gate refuses any aux_route flag, lever or probe on a non-H kind)
    if toks.count(H_FLAGS[0]) != 1 or flag_value(toks, H_FLAGS[0]) != M_MODE:
        bad.append(f"{H_FLAGS[0]}: arm {kind} needs it named once as {M_MODE}, argv {flag_value(toks, H_FLAGS[0])!r} (a flag whose default disables it must be named)")
    if flag_value(toks, H_FLAGS[1]) != H_WEIGHT or toks.count(H_FLAGS[1]) != 1:
        bad.append(f"{H_FLAGS[1]}: arm {kind} needs it named once as {H_WEIGHT} (the card's weight), argv {flag_value(toks, H_FLAGS[1])!r}")
    if lv.get("aux_route") != M_MODE:
        bad.append(f"levers.aux_route: arm {kind} needs {M_MODE}, built {lv.get('aux_route')!r}")
    if float(lv.get("w_aux_route", -1.0)) != float(H_WEIGHT):
        bad.append(f"levers.w_aux_route: arm {kind} needs {H_WEIGHT}, built {lv.get('w_aux_route')!r}")
    if not pr.get("pass"):
        bad.append("aux_route_probe missing or not pass")
    for key in ("head_grad_total", "head_grad_route", "emb_grad_total", "emb_grad_ade", "plan_move_when_emb_set", "plan_move_when_p_permuted", "own_rows_probe"):
        v = pr.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not v > 0.0:
            bad.append(f"aux_route_probe.{key} {v!r} (must be > 0)")
    for key in ("real_init_emb_each", "emb_each"):
        v = pr.get(key)
        if not (isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0.0 for x in v)):
            bad.append(f"aux_route_probe.{key} {v!r} (must be 3 values, every one > 0: each branch reaches the loss)")
    zero_keys = ("control_decoder_grad_on_head", "control_tside_grad_on_head", "control_tside_grad_on_emb", "control_repeat_grad_diff", "control_prior_grad", "control_no_own_row_route_term", "control_no_own_row_head_grad")
    for key in zero_keys:
        if pr.get(key) != 0.0:
            bad.append(f"aux_route_probe.{key} {pr.get(key)!r} (must be exactly 0.0)")
    ident_keys = ("identity_outputs_differing", "identity_terms_differing")
    for key in ident_keys:
        if pr.get(key) != []:
            bad.append(f"aux_route_probe.{key} {pr.get(key)!r} (must be an empty list: at the zero init the model is the parent's)")
    grp = (live.get("groups_grad") or {}).get("ROUTE", 0.0)
    if not grp > 0.0:
        bad.append("G-LIVE: the ROUTE group has no gradient")
    if "route" not in set(live.get("terms") or {}):
        bad.append("G-LIVE: the route loss term is not among the terms the loop logged")
    return bad


def hyg_m():
    """PREREG_V8F_N1DM G-HYG: the config refuses the unbuilt ce mode, an unknown mode, a non-finite or negative weight, modes with a per-unit code mix or a cut interface, and an unknown key."""
    from tanitad.models import v8f_mini as M
    n = 0
    for kw in (dict(aux_route="ce"), dict(aux_route="mode"), dict(aux_route="modes", w_aux_route=float("nan")), dict(aux_route="modes", w_aux_route=-1.0), dict(aux_route="modes", w_aux_route=0.5, code_mix="kind"),
               dict(aux_route="modes", w_aux_route=0.5, o_interface="cut"), dict(aux_routez="modes")):
        try:
            M.V8FMiniConfig(**kw)
        except (ValueError, TypeError):
            n += 1
    return [] if n == 7 else [f"only {n} of 7 bad aux_route (modes) configs were refused"]


def hyg_c():
'''
sub(CHK, "def hyg_c():\n", NEW_FUNCS)
sub(CHK, '''    bad += h_gate_problems(kind, toks, sc)  ''', '''    bad += m_gate_problems(kind, toks, sc)                                   # PREREG_V8F_N1DM: the M kind's modes head (a non-M kind builds no route_br and carries no modes probe)
    bad += h_gate_problems(kind, toks, sc)  ''')
sub(CHK, "+ (hyg_h() if kind in H_KINDS else [])", "+ (hyg_h() if kind in H_KINDS else []) + (hyg_m() if kind in M_KINDS else [])")
sub(CHK, 'choices=D_KINDS + W_KINDS + C_KINDS + H_KINDS)', 'choices=D_KINDS + W_KINDS + C_KINDS + H_KINDS + M_KINDS)')
sub(CHK, "c_pairing_problems if a.kind in C_KINDS else h_pairing_problems if a.kind in H_KINDS else pairing_problems)", "c_pairing_problems if a.kind in C_KINDS else h_pairing_problems if a.kind in H_KINDS else m_pairing_problems if a.kind in M_KINDS else pairing_problems)")
# moved_identity
sub(CHK, '''    rr = [k for k in fin_sd if k.startswith(("route_head.", "route_emb."))]''', '''    rr = [k for k in fin_sd if k.startswith(("route_head.", "route_emb.", "route_br."))]''')
sub(CHK, '''    elif rr:
        bad.append(f"route keys {rr[:2]} in a {kind} checkpoint")''', '''    elif kind in M_KINDS:                                               # PREREG_V8F_N1DM: route_head (seeded default, moved) + K = 3 branches that START at EXACTLY 0.0 and must have moved; NO route_emb
        wh, bh = fin_sd.get("route_head.weight"), fin_sd.get("route_head.bias")
        brs = [fin_sd.get(f"route_br.{k}.weight") for k in range(3)]
        if wh is None or bh is None or any(w is None for w in brs):
            bad.append(f"{kind} checkpoint lacks route_head.weight / route_head.bias / route_br.0..2.weight")
        else:
            for k, w in enumerate(brs):
                if not bool((w != 0).any()):
                    bad.append(f"{kind}: route_br.{k}.weight did not move off its zero init")
            do = int(fin_sd["head_o.2.weight"].shape[0]) if "head_o.2.weight" in fin_sd else None
            if len(tuple(wh.shape)) != 2 or int(wh.shape[0]) != 3 or tuple(bh.shape) != (3,) or any(len(tuple(w.shape)) != 2 or int(w.shape[1]) != int(wh.shape[1]) or (do is not None and int(w.shape[0]) != do) for w in brs) or (
                    "head_o.0.weight" in fin_sd and int(wh.shape[1]) != int(fin_sd["head_o.0.weight"].shape[1])):
                bad.append(f"{kind}: route modules are {tuple(wh.shape)} / {tuple(bh.shape)} / {[tuple(w.shape) for w in brs]} (need [3, n_q * d] / [3] / 3 x [head_o's output width, n_q * d])")
        if any(k.startswith("route_emb.") for k in fin_sd):
            bad.append(f"{kind} checkpoint carries route_emb keys (the modes arm has no conditioning embedding)")
    elif rr:
        bad.append(f"route keys {rr[:2]} in a {kind} checkpoint")''')
# an H kind must not carry route_br
sub(CHK, '''    if kind in H_KINDS:
        wh, bh, we = fin_sd.get("route_head.weight"), fin_sd.get("route_head.bias"), fin_sd.get("route_emb.weight")
''', '''    if kind in H_KINDS:
        if any(k.startswith("route_br.") for k in fin_sd):
            bad.append(f"{kind} checkpoint carries route_br keys (the cond arm has no plan branches)")
        wh, bh, we = fin_sd.get("route_head.weight"), fin_sd.get("route_head.bias"), fin_sd.get("route_emb.weight")
''')
print("check edits ok")
