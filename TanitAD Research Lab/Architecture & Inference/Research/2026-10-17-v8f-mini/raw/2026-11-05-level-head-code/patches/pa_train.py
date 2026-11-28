"""Lever (a) `--level-head {off,on}` + `--w-level`: trainer side. Reads tip -> writes new (LF, exact-anchor replace; each anchor must occur exactly once)."""
import sys
from pathlib import Path

SRC = Path(sys.argv[1]) / "stack/scripts/v8f_mini_train.py"
DST = Path(sys.argv[2]) / "stack/scripts/v8f_mini_train.py"
s = SRC.read_bytes().decode("utf-8")
assert "\r" not in s


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:100])
    s = s.replace(old, new)


# ---- FLAG_CHECKS (G-DVB entries)
rep('''    "init_ckpt": "data:init_ckpt",''', '''    # PREREG_V8F_LEVEL (lever (a) of DECISION_K2): the vision-only speed-level head. `lever:level_head` = equals built_levers().get("level_head", "off") (the BUILT model: lvl_head + e_level exist iff on); `cfg:w_level` = equals the BUILT config's
    # weight: argv -> make_config -> the BUILT model -> declared_vs_built (bad_level: a head with no loss, a weight with no head, a non-zero e_level at init, a head whose code names a non-vision input are REFUSED) -> level_head_probe (G-LIVE:
    # the BUILT loss moves by exactly w_level x the term, the head receives its gradient, the detach control reads EXACTLY 0.0, the token reaches O's output, vision-only guard) -> `level_stage(b)` BEFORE the ego stages in the loop.
    "level_head": "lever:level_head", "w_level": "cfg:w_level",
    "init_ckpt": "data:init_ckpt",''')

# ---- parser
i = s.index("    # ---- PREREG_V8F_HACC_MC section 7 (O-RECOVER): an ego-state perturbation of the kinematic inputs; targets and labels are never touched.")
s = s[:i] + '''    # ---- PREREG_V8F_LEVEL (lever (a) of DECISION_K2). Both flags default to OFF and have their G-DVB entry in FLAG_CHECKS and their clause in bad_level.
    ap.add_argument("--level-head", dest="level_head", choices=M.LEVEL_HEAD_MODES, default="off",
                    help="PREREG_V8F_LEVEL: off (default) = nothing is built = every earlier arm, bit for bit (state dict AND outputs). on = a head on the mean of the CURRENT tick frame's adapted VISION tokens -> MLP(d, d, 1) -> v_hat (m/s), "
                         "trained with --w-level x Huber(v_hat, the LOGGED speed at NOW) on every window; O reads e_level(v_hat.detach() / 10) (zero-initialised, added to O's ego conditioning token) at every tick, train and inference alike. "
                         "The head reads vision only (labels may use ego, inference may not). Needs --w-level > 0")
    ap.add_argument("--w-level", dest="w_level", type=float, default=0.0,
                    help="PREREG_V8F_LEVEL: the weight of the level term in the total. 0.0 (default) with --level-head off. Must be > 0 with --level-head on (a head with no loss is refused) and exactly 0.0 with it off (a weight with no head is refused). "
                         "The value is fixed by the arm's PREREG (the level loss is ~10 % of the total at step 0 on the smoke), written into the argv and never tuned")
''' + s[i:]

# ---- config
rep("w_acc=a.w_acc, launch_intent_w=a.launch_intent_w)", "w_acc=a.w_acc, launch_intent_w=a.launch_intent_w,\n                           level_head=a.level_head, w_level=a.w_level)")

# ---- init ckpt: the declared new keys may be missing, nothing else
rep('''    model.load_state_dict(sd, strict=True)
    cur = model.state_dict()
    rec["n_mismatch_after_load"]''', '''    fresh_ok = sorted(k for k in model.state_dict() if k.startswith(M.LEVEL_KEY_PREFIXES))       # PREREG_V8F_LEVEL: the keys the declared lever adds (none for every earlier argv)
    if not fresh_ok:
        model.load_state_dict(sd, strict=True)                                                   # every earlier argv: the strict load, verbatim
    else:                                                                                        # a lever-ON fine-tune: EVERY key of the checkpoint loads, and EXACTLY the declared new keys are missing (fresh, zero/seeded init)
        res = model.load_state_dict(sd, strict=False)
        if res.unexpected_keys:
            raise ValueError(f"the checkpoint carries keys the built model does not have: {sorted(res.unexpected_keys)[:8]}")
        if set(res.missing_keys) - set(fresh_ok):
            raise ValueError(f"the built model lacks checkpoint values for keys that are not the declared lever's: {sorted(set(res.missing_keys) - set(fresh_ok))[:8]}")
        rec["fresh_keys"], rec["declared_new_keys"] = sorted(res.missing_keys), fresh_ok
    cur = model.state_dict()
    rec["n_mismatch_after_load"]''')
rep('''    elif got.get("skipped") != "resume":
        bad.append(f"init_ckpt: declared but neither applied nor skipped for a resume ({got.get('skipped')!r})")
    return bad''', '''    elif got.get("skipped") != "resume":
        bad.append(f"init_ckpt: declared but neither applied nor skipped for a resume ({got.get('skipped')!r})")
    if got.get("applied"):                                  # PREREG_V8F_LEVEL: only the declared lever's keys may be fresh, and a lever-ON argv must have declared them
        if bool(got.get("declared_new_keys")) != (a.level_head == "on"):
            bad.append(f"init_ckpt: level_head {a.level_head} declared but the load declared new keys {got.get('declared_new_keys')}")
        if set(got.get("fresh_keys", [])) - set(got.get("declared_new_keys", [])):
            bad.append(f"init_ckpt: keys {sorted(set(got.get('fresh_keys', [])) - set(got.get('declared_new_keys', [])))} were left fresh but are not the declared lever's")
    return bad''')

# ---- bad_level + call
rep('''def declared_vs_built(a, cfg, model, opt, data_info, enc=None''', '''def bad_level(a, model, data_info=None) -> list:
    """PREREG_V8F_LEVEL, G-DVB: ``--level-head`` / ``--w-level`` read back from the BUILT model. Empty = pass. A head declared on that builds nothing, a head built that was not declared, a head with no loss (w_level 0),
    a weight with no head, a head whose code names a non-vision input and an e_level that is not zero at a fresh start are each refused."""
    bad = []
    if not (math.isfinite(a.w_level) and a.w_level >= 0.0):
        bad.append(f"w_level {a.w_level!r} is not a finite number >= 0")
    built = model.built_levers()
    got = built.get("level_head", "off")
    if got != a.level_head:
        bad.append(f"lever level_head: declared {a.level_head} but built {got}")
    if model.cfg.w_level != a.w_level:
        bad.append(f"cfg.w_level: declared {a.w_level} but built {model.cfg.w_level}")
    has_keys = [k for k in model.state_dict() if k.startswith(M.LEVEL_KEY_PREFIXES)]
    if a.level_head == "off":
        if a.w_level != 0.0:
            bad.append(f"w_level {a.w_level} declared with --level-head off: a loss weight with no head -- a lever that is declared and does nothing")
        if model.lvl_head is not None or model.e_level is not None or has_keys:
            bad.append(f"level_head off declared but a level head was built ({len(has_keys)} state-dict keys)")
        return bad
    if not a.w_level > 0.0:
        bad.append(f"level_head on declared but w_level is {a.w_level}: a head with no loss -- a lever that is declared and does nothing")
    if model.lvl_head is None or model.e_level is None or not has_keys:
        bad.append("level_head on declared but no head / e_level was built")
        return bad
    if not built.get("level_vision_only_source", False):
        bad.append(f"the level head's code names non-vision inputs: {M.level_head_source_problems()}")
    ic = (data_info or {}).get("init_ckpt") or {}
    carried = bool(ic.get("applied")) and not any(k.startswith("e_level.") for k in ic.get("fresh_keys", []))      # a lever-ON checkpoint carries its own (trained) e_level
    if not built.get("level_token_zero_init", False) and not carried:
        bad.append("e_level is not zero-initialised at a fresh start: O's step-0 forward would not be the off model's")
    return bad


def declared_vs_built(a, cfg, model, opt, data_info, enc=None''')
rep('''    bad += bad_dagger(a, dag, model.cfg, data_info)                                              # PREREG_V8F_DAGGER''', '''    bad += bad_level(a, model, data_info)                                                        # PREREG_V8F_LEVEL: --level-head / --w-level read back from the BUILT model
    bad += bad_dagger(a, dag, model.cfg, data_info)                                              # PREREG_V8F_DAGGER''')

# ---- G-LIVE expects the level term
rep('''    if cfg.wm != "off":
        expect.add("wm")
    missing = sorted(expect - set(terms_seen))''', '''    if cfg.wm != "off":
        expect.add("wm")
    if cfg.level_head == "on":
        expect.add("level")                                              # PREREG_V8F_LEVEL
    missing = sorted(expect - set(terms_seen))''', count=2)

# ---- stage + probe
i = s.index("def ego_perturb_probe(model, batch, cfg, ego, exclude=None) -> dict:")
s = s[:i] + '''def level_stage(batch: dict) -> tuple:
    """PREREG_V8F_LEVEL: the ONE call that turns a COLLATED batch into the level head's label ``(level_y [B] float32, level_ok [B] bool)`` -- the LOGGED speed at NOW, ``kin3[:, 0]``. The training loop calls it, and
    ``level_head_probe`` / the main() probe batch call the same function. It must run BEFORE ``ego.apply`` / ``dag.apply`` / ``comp.apply`` / ``rcv.apply`` (a swapped or perturbed sample's ``kin3`` is the own or perturbed
    state, not the label: the head sees the LOGGED frame and is asked for the LOGGED speed). Labels may use ego; the head's input is vision only."""
    y = batch["kin3"][:, 0].float()
    return y, torch.isfinite(y)


def level_head_probe(model, batch, cfg, rtol: float = 1e-3) -> dict:
    """G-LIVE for PREREG_V8F_LEVEL, read from the BUILT model's own ``loss`` on the real probe batch (a COPY; the zero-init e_level and O's zero-init head are woken on the copy so every path carries signal). With the head off
    nothing runs: {"level": "off"} (the default run is the pre-change run).

      vision only    ``M.assert_level_head_vision_only``: the source names no ego / nav / limit / code / label input, ``_level`` reads only frame_ok, v_hat is bit-identical when every non-vision key changes and moves with the features
      term           finite and > 0
      contribution   total(w_level) - total(0) == w_level x term (to ``rtol`` and float32 rounding). DELETE the total line and this reads 0.0 -> RED
      head gradient  sum |grad(total(w)) - grad(total(0))| over ``lvl_head.*`` > 0
      detach control at w_level = 0 the head's gradient is EXACTLY 0.0 (O reads v_hat.detach(): the plan's loss never reaches the head). Remove the detach and this reads > 0 -> RED
      token reaches  at w_level = 0 e_level receives a non-zero gradient, and zeroing e_level changes O's plan (the token is read)
      controls       the same loss twice at w gives gradients that differ by EXACTLY 0.0; an ALL-FALSE level_ok gives term 0.0, total difference 0.0 and head gradient difference EXACTLY 0.0; the frozen P10 prior's gradient difference is EXACTLY 0.0

    Returns the record selfcheck.json carries; any violation is an AssertionError, which the caller turns into a refusal."""
    if cfg.level_head != "on":
        return {"level": "off", "pass": True, "skipped": "--level-head not named: nothing is built (the default run is the pre-change run)"}
    if "level_y" not in batch or "level_ok" not in batch:
        raise AssertionError("level probe: the probe batch carries no level_y / level_ok (the loop's own call is level_stage)")
    W = float(cfg.w_level)
    eps32 = float(torch.finfo(torch.float32).eps)
    m = copy.deepcopy(model).cpu()
    m.train()
    vis = M.assert_level_head_vision_only(m, batch)
    with torch.no_grad():                    # the number the arm's PREREG needs: W such that the level term is ~10 % of the total at STEP 0 (the copy before it is woken = the model as launched)
        torch.manual_seed(20261105)
        o0 = m.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    t0_total, t0_term = float(o0["terms"]["total"]), float(o0["terms"]["level"])
    step0 = {"step0_total": t0_total, "step0_level_term": t0_term, "step0_total_without_level": t0_total - W * t0_term,
             "w_level_for_10pct_share": ((0.1 / 0.9) * (t0_total - W * t0_term) / t0_term) if t0_term > 0.0 else None}
    g = torch.Generator().manual_seed(17)
    with torch.no_grad():
        m.e_level.weight.copy_(torch.randn(m.e_level.weight.shape, generator=g) * 0.5)
        m.e_level.bias.copy_(torch.randn(m.e_level.bias.shape, generator=g) * 0.5)
        wo = m.head_o[2].weight
        wo.copy_(torch.randn(wo.shape, generator=g) * 0.05)
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    none = torch.zeros_like(batch["level_ok"])

    def run(w_, ok_mask=None):
        m.cfg.w_level = w_
        for _, p in named:
            p.grad = None
        for p in m.prior.parameters():
            p.grad = None
        torch.manual_seed(20261105)
        out = m.loss({**batch, "level_ok": batch["level_ok"] if ok_mask is None else ok_mask}, p_gt=0.5, gen=torch.Generator().manual_seed(11))
        T = out["terms"]
        for k, v in T.items():
            if not torch.isfinite(v):
                raise AssertionError(f"level probe: loss term {k} is not finite at w_level {w_}: {float(v)}")
        T["total"].backward()
        gr = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}
        gr["__prior__"] = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in m.prior.parameters()])
        return float(T["total"].detach()), float(T["level"].detach()), gr, out["traj60"].detach().clone()

    t_w, term_w, g_w, tr_w = run(W)
    t_0, term_0, g_0, tr_0 = run(0.0)
    t_wb, _, g_wb, _ = run(W)
    t_n, term_n, g_n, _ = run(W, none)
    dsum = lambda ga, gb, pre: float(sum((ga[n] - gb[n]).abs().sum() for n in ga if n.startswith(pre)))      # noqa: E731
    asum = lambda ga, pre: float(sum(ga[n].abs().sum() for n in ga if n.startswith(pre)))                       # noqa: E731
    saved = (m.e_level.weight.detach().clone(), m.e_level.bias.detach().clone())
    with torch.no_grad():
        m.e_level.weight.zero_()
        m.e_level.bias.zero_()
    _, _, _, tr_z = run(0.0)
    with torch.no_grad():
        m.e_level.weight.copy_(saved[0])
        m.e_level.bias.copy_(saved[1])
    res = {"w_level_declared": W, "vision_only": vis, **step0, "term": term_w, "total_w": t_w, "total_0": t_0, "contribution": t_w - t_0, "w_level_x_term": W * term_w,
           "head_grad_diff": dsum(g_w, g_0, "lvl_head."), "head_grad_at_w0_detach_control": asum(g_0, "lvl_head."), "e_level_grad_at_w0": asum(g_0, "e_level."),
           "plan_moves_with_token": float((tr_0 - tr_z).abs().max()),
           "control_repeat_total_diff": abs(t_wb - t_w), "control_repeat_grad_diff": dsum(g_w, g_wb, ""),
           "control_allfalse_term": term_n, "control_allfalse_total_diff": abs(t_n - t_0), "control_allfalse_head_grad_diff": dsum(g_n, g_0, "lvl_head."),
           "control_prior_grad_diff": float((g_w["__prior__"] - g_0["__prior__"]).abs().sum())}
    if not math.isfinite(term_w) or not term_w > 0.0:
        raise AssertionError(f"level probe: the level term reads {term_w} (finite and > 0 required)")
    if res["control_repeat_total_diff"] != 0.0 or res["control_repeat_grad_diff"] != 0.0:
        raise AssertionError(f"level probe: the probe is not deterministic (the same loss twice differs by {res['control_repeat_total_diff']} / {res['control_repeat_grad_diff']}): its differences mean nothing")
    if not res["w_level_x_term"] > 64.0 * eps32 * max(abs(t_0), 1.0):
        raise AssertionError(f"level probe: w_level x term = {res['w_level_x_term']} is below float32 visibility against the total {t_0}: the probe cannot see the term")
    if abs(res["contribution"] - res["w_level_x_term"]) > rtol * res["w_level_x_term"] + 4.0 * eps32 * max(abs(t_w), abs(t_0)):
        raise AssertionError(f"level probe: the total moved by {res['contribution']} when the weight went from 0 to {W}, w_level x term is {res['w_level_x_term']}: the term does not reach the loss with its weight")
    if not res["head_grad_diff"] > 0.0:
        raise AssertionError("level probe: the level head receives no gradient from the level term (the sum of |d total / d lvl_head| differences is 0.0)")
    if res["head_grad_at_w0_detach_control"] != 0.0:
        raise AssertionError(f"level probe: at w_level 0 the head's gradient is {res['head_grad_at_w0_detach_control']}, not EXACTLY 0.0: the plan's loss reaches the head (O must read v_hat.detach())")
    if not res["e_level_grad_at_w0"] > 0.0 or not res["plan_moves_with_token"] > 0.0:
        raise AssertionError(f"level probe: the level token does not reach O (e_level grad {res['e_level_grad_at_w0']}, plan moves by {res['plan_moves_with_token']} when e_level is zeroed)")
    if term_n != 0.0 or res["control_allfalse_total_diff"] != 0.0 or res["control_allfalse_head_grad_diff"] != 0.0:
        raise AssertionError(f"level probe: an ALL-FALSE level_ok must read exactly 0.0 (term {term_n}, total diff {res['control_allfalse_total_diff']}, head grad diff {res['control_allfalse_head_grad_diff']}): the mask is not what selects the rows")
    if res["control_prior_grad_diff"] != 0.0:
        raise AssertionError(f"level probe: the frozen P10 prior's gradient moved with the weight ({res['control_prior_grad_diff']}): the control must read exactly 0.0")
    res["pass"] = True
    return res


''' + s[i:]

# ---- main: probe batch, probe call, selfcheck
rep('''            probe["launch"] = launch_stage(probe)
''', '''            probe["launch"] = launch_stage(probe)
        if cfg.level_head == "on":                       # PREREG_V8F_LEVEL: every probe calls model.loss on this batch, and a lever-ON loss reads level_y / level_ok (the loop's own call, on the LOGGED state)
            probe["level_y"], probe["level_ok"] = level_stage(probe)
''')
rep('''            probe_f["launch"] = launch_stage(probe_f)
''', '''            probe_f["launch"] = launch_stage(probe_f)
        if cfg.level_head == "on":                       # PREREG_V8F_LEVEL: as above
            probe_f["level_y"], probe_f["level_ok"] = level_stage(probe_f)
''')
rep('''    levers = model.built_levers()
    levers["ego_perturb"] = ego.built_levers()''', '''    try:
        lvp = level_head_probe(model, probe, cfg)         # PREREG_V8F_LEVEL: G-LIVE on the loop's own label call (level_stage); the term reaches the loss with its weight, the head gets its gradient, the detach control reads EXACTLY 0.0, the token reaches O, the head is vision-only (off: nothing runs)
    except (AssertionError, ValueError) as e:
        raise SystemExit(f"ZZDVB-REFUSEDZZ level-head: {e}")
    levers = model.built_levers()
    levers["ego_perturb"] = ego.built_levers()''')
rep('''    sc["launch_intent_probe"] = lip
''', '''    sc["launch_intent_probe"] = lip
    if cfg.level_head == "on":                                          # lever-ON arms only: a flag-off selfcheck.json keeps its key set
        sc["level_head_probe"] = lvp
''')

# ---- loop
rep('''    launch_on = cfg.launch_intent_w > 0.0                          # PREREG_V8F_LAUNCH_INTENT: off (the default) = the loop runs no new line
''', '''    launch_on = cfg.launch_intent_w > 0.0                          # PREREG_V8F_LAUNCH_INTENT: off (the default) = the loop runs no new line
    level_on = cfg.level_head == "on"                              # PREREG_V8F_LEVEL: off (the default) = the loop runs no new line
''')
rep('''                lmask = launch_stage(b)
''', '''                lmask = launch_stage(b)
            if level_on:                                           # PREREG_V8F_LEVEL: the label is the LOGGED speed at NOW, taken BEFORE any ego stage rebuilds kin3 (the head sees the logged frame)
                lvl_lab = level_stage(b)
''')
rep('''            b = to_dev(b, dev)
            for g in opt.param_groups:''', '''            if level_on:                                           # ... and attached AFTER the stages (they return their own batch dict): the loss reads b["level_y"], b["level_ok"]
                b["level_y"], b["level_ok"] = lvl_lab
            b = to_dev(b, dev)
            for g in opt.param_groups:''')
rep('''    if enc is not None:
        summary["encoder"] = "lora"
''', '''    if level_on:                                                           # PREREG_V8F_LEVEL: the lever-ON arm's run record (a flag-off summary.json keeps its key set)
        summary["level_head"], summary["w_level"] = "on", cfg.w_level
    if enc is not None:
        summary["encoder"] = "lora"
''')

DST.write_bytes(s.encode("utf-8"))
print("wrote", DST, len(s))
