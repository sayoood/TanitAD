"""Patch 3 of 3 (lead head): stack/scripts/v8f_mini_train.py of the tip 219f6c7 -> the lead-head trainer (flags, FLAG_CHECKS, the gate clause bad_lead, init-ckpt fresh keys, labels, G-LIVE probe, the loop).

usage: python -I pb_train.py <path of the tip file> <path of the file to write>
Every anchor must match EXACTLY ONCE (a drifted tip is refused, not guessed at)."""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
t = src.read_bytes().decode("utf-8")
assert "\r\n" not in t


def rep(old, new, count=1):
    global t
    n = t.count(old)
    assert n == count, f"{n} matches (want {count}) for: {old[:100]!r}"
    t = t.replace(old, new)


def line_span(key):
    i = t.find(key)
    assert i >= 0 and t.count(key) == 1, f"{t.count(key)} lines contain {key[:100]!r}"
    a = t.rfind("\n", 0, i) + 1
    b = t.find("\n", i) + 1
    return a, b


def ins_after(key, text):
    global t
    a, b = line_span(key)
    t = t[:b] + text + t[b:]


def ins_before(key, text):
    global t
    a, b = line_span(key)
    t = t[:a] + text + t[a:]


# ---------------------------------------------------------------------------------------------- FLAG_CHECKS + flags
ins_after('"level_head": "lever:level_head", "w_level": "cfg:w_level",', '''    # PREREG_LEVELHEAD_A2 (arm LLR): the lead head + how O reads the heads. `lever:lead_head` = built_levers().get("lead_head", "off"); `cfg:w_lead` = the BUILT config's weight; `lever:aux_cond` = built_levers().get("aux_cond", "add")
    # (the BUILT wiring: e_lead / e_level exist iff add, e_aux iff token and a head is on). argv -> make_config -> the BUILT model -> declared_vs_built (bad_lead) -> lead_head_probe (G-LIVE) -> `lead_stage(b, LL)` BEFORE the ego stages.
    "lead_head": "lever:lead_head", "w_lead": "cfg:w_lead", "aux_cond": "lever:aux_cond",
''')
ins_after('"The value is fixed by the arm\'s PREREG (the level loss is ~10 % of the total at step 0 on the smoke), written into the argv and never tuned")', '''    # ---- PREREG_LEVELHEAD_A2 (arm LLR). All three default to the earlier behaviour and have their G-DVB entry in FLAG_CHECKS and their clause in bad_lead.
    ap.add_argument("--lead-head", dest="lead_head", choices=M.LEAD_HEAD_MODES, default="off",
                    help="PREREG_LEVELHEAD_A2: off (default) = nothing is built = every earlier arm, bit for bit (state dict AND outputs). on = a head on the mean of the CURRENT tick frame's adapted VISION tokens -> MLP(d, d, 2) -> (a presence logit "
                         "for 'a lead within 40 m', gap_hat = sigmoid = gap / 40 in [0, 1]), trained with --w-lead x (BCE(presence) on the frames with a lead label + Huber(gap_hat, min(gap, 40) / 40) on the frames with a lead within 40 m) "
                         "from the lead labels of --lead-train; O reads [sigmoid(logit), gap_hat] DETACHED at every tick, train and inference alike (how: --aux-cond). The head reads vision only. Needs --w-lead > 0")
    ap.add_argument("--w-lead", dest="w_lead", type=float, default=0.0,
                    help="PREREG_LEVELHEAD_A2: the weight of the lead term in the total. 0.0 (default) with --lead-head off. Must be > 0 with --lead-head on (a head with no loss is refused) and exactly 0.0 with it off (a weight with no head is refused). "
                         "The value is fixed by the arm's PREREG (the lead term is ~10 % of the total at step 0: selfcheck.json lead_head_probe.w_lead_for_10pct_share), written into the argv and never tuned")
    ap.add_argument("--aux-cond", dest="aux_cond", choices=M.AUX_COND_MODES, default="add",
                    help="PREREG_LEVELHEAD_A2: how O reads the heads' outputs. add (default) = e_level(v_hat.detach() / 10) and e_lead([sigmoid(logit), gap_hat].detach()) are zero-initialised and ADDED to O's ego conditioning token (the step-0 forward is the "
                         "init's). token = O gets ONE extra conditioning token e_aux(Linear(3, d), default init) of [v_hat / 10 (0 without the level head), sigmoid(logit), gap_hat] (detached), REPLACING both additive paths (the step-0 forward differs "
                         "from the init's by construction). token is refused unless --level-head or --lead-head is on")
''')
rep("level_head=a.level_head, w_level=a.w_level)", "level_head=a.level_head, w_level=a.w_level,\n                           lead_head=a.lead_head, w_lead=a.w_lead, aux_cond=a.aux_cond)")

# ---------------------------------------------------------------------------------------------- init-ckpt: only the declared lever keys may be fresh
rep('''        if bool(got.get("declared_new_keys")) != (a.level_head == "on"):
            bad.append(f"init_ckpt: level_head {a.level_head} declared but the load declared new keys {got.get('declared_new_keys')}")''',
    '''        if bool(got.get("declared_new_keys")) != (a.level_head == "on" or a.lead_head == "on"):          # PREREG_LEVELHEAD_A2: the lead head adds keys too
            bad.append(f"init_ckpt: level_head {a.level_head} / lead_head {a.lead_head} declared but the load declared new keys {got.get('declared_new_keys')}")''')
rep("fresh_ok = sorted(k for k in model.state_dict() if k.startswith(M.LEVEL_KEY_PREFIXES))       # PREREG_V8F_LEVEL: the keys the declared lever adds (none for every earlier argv)",
    "fresh_ok = sorted(k for k in model.state_dict() if k.startswith(M.NEW_LEVER_KEY_PREFIXES))       # PREREG_V8F_LEVEL / PREREG_LEVELHEAD_A2: the keys the declared levers add (level head, lead head, aux token; none for every earlier argv)")

# ---------------------------------------------------------------------------------------------- bad_level: the token wiring builds no e_level
rep('''    if model.lvl_head is None or model.e_level is None or not has_keys:
        bad.append("level_head on declared but no head / e_level was built")
        return bad''', '''    if model.lvl_head is None or (a.aux_cond == "add" and model.e_level is None) or not has_keys:           # PREREG_LEVELHEAD_A2: under --aux-cond token the additive e_level is not built (e_aux replaces it)
        bad.append("level_head on declared but no head / e_level was built")
        return bad''')
rep('''    if not built.get("level_token_zero_init", False) and not carried:
        bad.append("e_level is not zero-initialised at a fresh start: O's step-0 forward would not be the off model's")
    return bad
''', '''    if model.e_level is not None and not built.get("level_token_zero_init", False) and not carried:
        bad.append("e_level is not zero-initialised at a fresh start: O's step-0 forward would not be the off model's")
    return bad


#: PREREG_LEVELHEAD_A2: a --lead-train file that labels fewer than this share of the train clips is the wrong file (the eval file, a truncated copy): the launch is refused, and the share is recorded in selfcheck.json
LEAD_MIN_CLIP_COVERAGE = 0.5


def bad_lead(a, model, data_info=None) -> list:
    """PREREG_LEVELHEAD_A2, G-DVB: ``--lead-head`` / ``--w-lead`` / ``--aux-cond`` read back from the BUILT model and the BUILT label arrays. Empty = pass. Refused: a head declared on that builds nothing, a head built that was not declared,
    a head with no loss (w_lead 0), a weight with no head, ``--aux-cond token`` with no head on, a wiring that is not the declared one (token: e_aux built and BOTH additive paths absent; add: no e_aux), a head or aux assembly whose code
    names a non-vision input, an e_lead that is not zero at a fresh start, and a lead head with no lead labels (missing file, no labelled frame, no frame with a lead within 40 m, or too few labelled clips)."""
    bad = []
    if not (math.isfinite(a.w_lead) and a.w_lead >= 0.0):
        bad.append(f"w_lead {a.w_lead!r} is not a finite number >= 0")
    built = model.built_levers()
    any_head = a.level_head == "on" or a.lead_head == "on"
    got = built.get("lead_head", "off")
    if got != a.lead_head:
        bad.append(f"lever lead_head: declared {a.lead_head} but built {got}")
    if model.cfg.w_lead != a.w_lead:
        bad.append(f"cfg.w_lead: declared {a.w_lead} but built {model.cfg.w_lead}")
    if model.cfg.aux_cond != a.aux_cond:
        bad.append(f"cfg.aux_cond: declared {a.aux_cond} but built {model.cfg.aux_cond}")
    if a.aux_cond == "token" and not any_head:
        bad.append("--aux-cond token declared with --level-head and --lead-head both off: there is no head output for the token to carry -- a lever that is declared and does nothing")
    if built.get("aux_cond", "add") != a.aux_cond:
        bad.append(f"lever aux_cond: declared {a.aux_cond} but built {built.get('aux_cond', 'add')}")
    lead_keys = [k for k in model.state_dict() if k.startswith(M.LEAD_KEY_PREFIXES)]
    aux_keys = [k for k in model.state_dict() if k.startswith(M.AUX_KEY_PREFIXES)]
    if a.aux_cond == "token" and any_head:
        if model.e_aux is None or not aux_keys:
            bad.append("aux_cond token declared but no aux token (e_aux) was built")
        if model.e_level is not None or model.e_lead is not None:
            bad.append("aux_cond token declared but an additive path (e_level / e_lead) was built: the token REPLACES both additive paths")
        if built.get("aux_token_dim") != M.AUX_TOKEN_DIM:
            bad.append(f"aux token input width: expected {M.AUX_TOKEN_DIM} but built {built.get('aux_token_dim')}")
        if built.get("aux_vision_only_source") is not True:
            bad.append(f"the aux token's assembly names non-vision inputs: {M.aux_source_problems()}")
    elif model.e_aux is not None or aux_keys:
        bad.append(f"aux_cond {a.aux_cond} declared but an aux token was built ({len(aux_keys)} state-dict keys)")
    if a.lead_head == "off":
        if a.w_lead != 0.0:
            bad.append(f"w_lead {a.w_lead} declared with --lead-head off: a loss weight with no head -- a lever that is declared and does nothing")
        if model.lead_head is not None or model.e_lead is not None or lead_keys:
            bad.append(f"lead_head off declared but a lead head was built ({len(lead_keys)} state-dict keys)")
        return bad
    if not a.w_lead > 0.0:
        bad.append(f"lead_head on declared but w_lead is {a.w_lead}: a head with no loss -- a lever that is declared and does nothing")
    if model.lead_head is None or not lead_keys or (a.aux_cond == "add" and model.e_lead is None):
        bad.append("lead_head on declared but no head / e_lead was built")
        return bad
    if not built.get("lead_vision_only_source", False):
        bad.append(f"the lead head's code names non-vision inputs: {M.lead_head_source_problems()}")
    ic = (data_info or {}).get("init_ckpt") or {}
    carried = bool(ic.get("applied")) and not any(k.startswith("e_lead.") for k in ic.get("fresh_keys", []))      # a lead-ON checkpoint carries its own (trained) e_lead
    if model.e_lead is not None and not built.get("lead_token_zero_init", False) and not carried:
        bad.append("e_lead is not zero-initialised at a fresh start: O's step-0 forward would not be the init's")
    ll = (data_info or {}).get("lead_labels")
    if not ll:
        bad.append("lead_head on declared but no lead labels were built (--lead-train)")
    else:
        if ll.get("path") != os.path.abspath(a.lead_train):
            bad.append(f"lead labels: declared {os.path.abspath(a.lead_train)!r} but built from {ll.get('path')!r}")
        if not (isinstance(ll.get("sha256"), str) and len(ll["sha256"]) == 64):
            bad.append(f"lead labels: the file's sha256 is not recorded ({ll.get('sha256')!r})")
        if not (ll.get("n_has", 0) > 0 and ll.get("n_near", 0) > 0):
            bad.append(f"lead labels: {ll.get('n_has')} labelled frames and {ll.get('n_near')} with a lead within {M.LEAD_RANGE_M:.0f} m: the head has nothing to learn from")
        if ll.get("n_clips_with_labels", 0) < LEAD_MIN_CLIP_COVERAGE * max(ll.get("n_clips", 0), 1):
            bad.append(f"lead labels: only {ll.get('n_clips_with_labels')} of {ll.get('n_clips')} train clips carry labels (< {LEAD_MIN_CLIP_COVERAGE:.0%}): the wrong file")
    return bad
''')
ins_after("bad += bad_level(a, model, data_info)", '''    bad += bad_lead(a, model, data_info)                                                         # PREREG_LEVELHEAD_A2: --lead-head / --w-lead / --aux-cond read back from the BUILT model and the built label arrays
''')

# ---------------------------------------------------------------------------------------------- G-LIVE: the expected loss terms
rep('''    if cfg.level_head == "on":
        expect.add("level")                                              # PREREG_V8F_LEVEL
''', '''    if cfg.level_head == "on":
        expect.add("level")                                              # PREREG_V8F_LEVEL
    if cfg.lead_head == "on":
        expect.add("lead")                                               # PREREG_LEVELHEAD_A2
''', count=2)

# ---------------------------------------------------------------------------------------------- labels + the stage
ins_before("def level_head_probe(model, batch, cfg, rtol: float = 1e-3) -> dict:", '''def make_lead_labels(a, cfg, J, info):
    """PREREG_LEVELHEAD_A2: the lead head's label arrays, row-aligned to the label join (``D.lead_label_arrays``), or None when the head is off (nothing is read). A missing ``--lead-train`` file is a REFUSAL (the
    default path is a dev-box path: a lead head launched elsewhere must not train on nothing). Records the file's sha256 and the coverage in ``info["lead_labels"]`` for the gate and selfcheck.json."""
    if cfg.lead_head != "on":
        return None
    if not os.path.exists(a.lead_train):
        raise SystemExit(f"ZZDVB-REFUSEDZZ lead-head: --lead-head on needs the lead labels and --lead-train {a.lead_train!r} does not exist")
    LL = D.lead_label_arrays(J, a.lead_train)
    h = hashlib.sha256()
    with open(a.lead_train, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    info["lead_labels"] = {"path": os.path.abspath(a.lead_train), "sha256": h.hexdigest(), **{k: LL[k] for k in ("n_rows", "n_has", "n_near", "n_clips", "n_clips_with_labels")}}
    return LL


def lead_stage(batch: dict, LL: dict) -> tuple:
    """PREREG_LEVELHEAD_A2: the ONE call that turns a COLLATED batch into the lead head's label ``(lead_y [B] float32, lead_has [B] bool, lead_near [B] bool)`` -- looked up by the sample's clip index and cache step
    (``batch["ci"]``, ``batch["t"]``: row ``off[ci] + t`` of ``D.lead_label_arrays``, i.e. the lead file's raw frame ``k = 2 t``). The training loop calls it, and the main() probe batch / ``lead_head_probe`` call the same function. It must
    run BEFORE ``ego.apply`` / ``dag.apply`` / ``comp.apply`` / ``rcv.apply`` (a swapped sample's ``ci`` / ``t`` is the own one, not the label's: the head sees the LOGGED frame and is asked for the LOGGED lead). Labels may use
    anything; the head's input is vision only."""
    rows = LL["off"][batch["ci"].cpu().numpy()] + batch["t"].cpu().numpy()
    return torch.as_tensor(LL["y"][rows]), torch.as_tensor(LL["has"][rows]), torch.as_tensor(LL["near"][rows])


def lead_head_probe(model, batch, cfg, rtol: float = 1e-3) -> dict:
    """G-LIVE for PREREG_LEVELHEAD_A2, read from the BUILT model's own ``loss`` on the real probe batch (a COPY; the zero-init e_lead and O's zero-init head are woken on the copy so every path carries signal). With the head
    off nothing runs: {"lead": "off"}.

      vision only    ``M.assert_lead_head_vision_only`` (and ``M.assert_aux_vision_only`` under --aux-cond token): the source names no ego / nav / limit / code / label input, the head reads only frame_ok, its outputs are bit-identical when every
                     non-vision key changes and move with the features
      labels         the probe batch has frames with a lead label AND frames with a lead within 40 m (else the gap term is invisible)
      term           finite and > 0
      contribution   total(w_lead) - total(0) == w_lead x term (to ``rtol`` and float32 rounding). DELETE the total line and this reads 0.0 -> RED
      head gradient  sum |grad(total(w)) - grad(total(0))| over ``lead_head.*`` > 0, AND over each of the two output rows of ``lead_head.2.weight`` (presence, gap) > 0: a term that reaches one output only is RED
      detach control at w_lead = 0 the head's gradient is EXACTLY 0.0 (O reads the outputs detached: the plan's loss never reaches the head). Remove the detach and this reads > 0 -> RED
      token reaches  at w_lead = 0 e_lead (add) / the lead columns of e_aux (token) receive a non-zero gradient, and zeroing them changes O's plan (the head's output is read)
      controls       the same loss twice at w gives gradients that differ by EXACTLY 0.0; an ALL-FALSE lead_has gives term 0.0, total difference 0.0 and head gradient difference EXACTLY 0.0; the frozen P10 prior's gradient difference is EXACTLY 0.0

    ``w_lead_for_10pct_share`` = (0.1 / 0.9) x (total without the lead term) / (lead term) at STEP 0 (the formula of the level probe; the rest includes the level term when that head is on). Returns the record selfcheck.json carries;
    any violation is an AssertionError, which the caller turns into a refusal."""
    if cfg.lead_head != "on":
        return {"lead": "off", "pass": True, "skipped": "--lead-head not named: nothing is built (the default run is the pre-change run)"}
    if not all(k in batch for k in ("lead_y", "lead_has", "lead_near")):
        raise AssertionError("lead probe: the probe batch carries no lead_y / lead_has / lead_near (the loop's own call is lead_stage)")
    W = float(cfg.w_lead)
    eps32 = float(torch.finfo(torch.float32).eps)
    m = copy.deepcopy(model).cpu()
    m.train()
    vis = M.assert_lead_head_vision_only(m, batch)
    aux_vis = M.assert_aux_vision_only(m, batch) if m.e_aux is not None else None
    fok = batch["frame_ok"][:, -1]
    n_has, n_near = int((batch["lead_has"] & fok).sum()), int((batch["lead_near"] & batch["lead_has"] & fok).sum())
    if n_has == 0 or n_near == 0:
        raise AssertionError(f"lead probe: the probe batch has {n_has} frames with a lead label and {n_near} with a lead within {M.LEAD_RANGE_M:.0f} m: the probe cannot see the lead term (use a larger probe batch or the real corpus)")
    with torch.no_grad():                    # the number the arm's PREREG needs: W such that the lead term is ~10 % of the total at STEP 0 (the copy before it is woken = the model as launched)
        torch.manual_seed(20261105)
        o0 = m.loss(dict(batch), p_gt=0.5, gen=torch.Generator().manual_seed(11))
    T0 = o0["terms"]
    t0_total, t0_term = float(T0["total"]), float(T0["lead"])
    step0 = {"step0_total": t0_total, "step0_lead_term": t0_term, "step0_lead_bce": float(T0["lead_bce"]), "step0_lead_huber": float(T0["lead_huber"]), "step0_total_without_lead": t0_total - W * t0_term,
             "step0_lead_share": W * t0_term / t0_total, "w_lead_for_10pct_share": ((0.1 / 0.9) * (t0_total - W * t0_term) / t0_term) if t0_term > 0.0 else None}
    if "level" in T0:
        step0["step0_level_share"] = float(m.cfg.w_level) * float(T0["level"]) / t0_total
    g = torch.Generator().manual_seed(17)
    with torch.no_grad():
        if m.e_lead is not None:
            m.e_lead.weight.copy_(torch.randn(m.e_lead.weight.shape, generator=g) * 0.5)
            m.e_lead.bias.copy_(torch.randn(m.e_lead.bias.shape, generator=g) * 0.5)
        wo = m.head_o[2].weight
        wo.copy_(torch.randn(wo.shape, generator=g) * 0.05)
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    none = torch.zeros_like(batch["lead_has"])

    def run(w_, has_mask=None):
        m.cfg.w_lead = w_
        for _, p in named:
            p.grad = None
        for p in m.prior.parameters():
            p.grad = None
        torch.manual_seed(20261105)
        out = m.loss({**batch, "lead_has": batch["lead_has"] if has_mask is None else has_mask}, p_gt=0.5, gen=torch.Generator().manual_seed(11))
        T = out["terms"]
        for k, v in T.items():
            if not torch.isfinite(v):
                raise AssertionError(f"lead probe: loss term {k} is not finite at w_lead {w_}: {float(v)}")
        T["total"].backward()
        gr = {n: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for n, p in named}
        gr["__prior__"] = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel())) for p in m.prior.parameters()])
        return float(T["total"].detach()), float(T["lead"].detach()), gr, out["traj60"].detach().clone()

    t_w, term_w, g_w, tr_w = run(W)
    t_0, term_0, g_0, tr_0 = run(0.0)
    t_wb, _, g_wb, _ = run(W)
    t_n, term_n, g_n, _ = run(W, none)
    dsum = lambda ga, gb, pre: float(sum((ga[n] - gb[n]).abs().sum() for n in ga if n.startswith(pre)))      # noqa: E731
    asum = lambda ga, pre: float(sum(ga[n].abs().sum() for n in ga if n.startswith(pre)))                       # noqa: E731
    row = lambda i: float((g_w["lead_head.2.weight"][i] - g_0["lead_head.2.weight"][i]).abs().sum())          # noqa: E731
    if m.e_lead is not None:                                       # --aux-cond add: zero e_lead and read O's plan
        reach_grad = {"e_lead_grad_at_w0": asum(g_0, "e_lead.")}
        saved = (m.e_lead.weight.detach().clone(), m.e_lead.bias.detach().clone())
        with torch.no_grad():
            m.e_lead.weight.zero_()
            m.e_lead.bias.zero_()
        _, _, _, tr_z = run(0.0)
        with torch.no_grad():
            m.e_lead.weight.copy_(saved[0])
            m.e_lead.bias.copy_(saved[1])
    else:                                                          # --aux-cond token: zero the lead columns (1, 2) of e_aux's weight and read O's plan
        reach_grad = {"e_aux_lead_cols_grad_at_w0": float(g_0["e_aux.weight"][:, 1:].abs().sum())}
        saved = m.e_aux.weight.detach().clone()
        with torch.no_grad():
            m.e_aux.weight[:, 1:].zero_()
        _, _, _, tr_z = run(0.0)
        with torch.no_grad():
            m.e_aux.weight.copy_(saved)
    res = {"w_lead_declared": W, "aux_cond": cfg.aux_cond, "vision_only": vis, "aux_vision_only": aux_vis, "probe_n_has": n_has, "probe_n_near": n_near, **step0, "term": term_w, "total_w": t_w, "total_0": t_0,
           "contribution": t_w - t_0, "w_lead_x_term": W * term_w,
           "head_grad_diff": dsum(g_w, g_0, "lead_head."), "head_grad_diff_presence_row": row(0), "head_grad_diff_gap_row": row(1), "head_grad_at_w0_detach_control": asum(g_0, "lead_head."), **reach_grad,
           "plan_moves_with_token": float((tr_0 - tr_z).abs().max()),
           "control_repeat_total_diff": abs(t_wb - t_w), "control_repeat_grad_diff": dsum(g_w, g_wb, ""),
           "control_allfalse_term": term_n, "control_allfalse_total_diff": abs(t_n - t_0), "control_allfalse_head_grad_diff": dsum(g_n, g_0, "lead_head."),
           "control_prior_grad_diff": float((g_w["__prior__"] - g_0["__prior__"]).abs().sum())}
    if not math.isfinite(term_w) or not term_w > 0.0:
        raise AssertionError(f"lead probe: the lead term reads {term_w} (finite and > 0 required)")
    if res["control_repeat_total_diff"] != 0.0 or res["control_repeat_grad_diff"] != 0.0:
        raise AssertionError(f"lead probe: the probe is not deterministic (the same loss twice differs by {res['control_repeat_total_diff']} / {res['control_repeat_grad_diff']}): its differences mean nothing")
    if not res["w_lead_x_term"] > 64.0 * eps32 * max(abs(t_0), 1.0):
        raise AssertionError(f"lead probe: w_lead x term = {res['w_lead_x_term']} is below float32 visibility against the total {t_0}: the probe cannot see the term")
    if abs(res["contribution"] - res["w_lead_x_term"]) > rtol * res["w_lead_x_term"] + 4.0 * eps32 * max(abs(t_w), abs(t_0)):
        raise AssertionError(f"lead probe: the total moved by {res['contribution']} when the weight went from 0 to {W}, w_lead x term is {res['w_lead_x_term']}: the term does not reach the loss with its weight")
    if not res["head_grad_diff"] > 0.0:
        raise AssertionError("lead probe: the lead head receives no gradient from the lead term (the sum of |d total / d lead_head| differences is 0.0)")
    if not (res["head_grad_diff_presence_row"] > 0.0 and res["head_grad_diff_gap_row"] > 0.0):
        raise AssertionError(f"lead probe: the lead term reaches one output of the head only (presence row {res['head_grad_diff_presence_row']}, gap row {res['head_grad_diff_gap_row']}): both the BCE and the Huber must reach it")
    if res["head_grad_at_w0_detach_control"] != 0.0:
        raise AssertionError(f"lead probe: at w_lead 0 the head's gradient is {res['head_grad_at_w0_detach_control']}, not EXACTLY 0.0: the plan's loss reaches the head (O must read the head's outputs detached)")
    if not list(reach_grad.values())[0] > 0.0 or not res["plan_moves_with_token"] > 0.0:
        raise AssertionError(f"lead probe: the lead output does not reach O ({reach_grad}, plan moves by {res['plan_moves_with_token']} when the lead path is zeroed)")
    if term_n != 0.0 or res["control_allfalse_total_diff"] != 0.0 or res["control_allfalse_head_grad_diff"] != 0.0:
        raise AssertionError(f"lead probe: an ALL-FALSE lead_has must read exactly 0.0 (term {term_n}, total diff {res['control_allfalse_total_diff']}, head grad diff {res['control_allfalse_head_grad_diff']}): the mask is not what selects the rows")
    if res["control_prior_grad_diff"] != 0.0:
        raise AssertionError(f"lead probe: the frozen P10 prior's gradient moved with the weight ({res['control_prior_grad_diff']}): the control must read exactly 0.0")
    res["pass"] = True
    return res


''')

# ---------------------------------------------------------------------------------------------- level probe under --aux-cond token (the additive e_level is not built)
rep('''    vis = M.assert_level_head_vision_only(m, batch)
''', '''    vis = M.assert_level_head_vision_only(m, batch)
    aux_vis = M.assert_aux_vision_only(m, batch) if m.e_aux is not None else None                # PREREG_LEVELHEAD_A2: under --aux-cond token the assembly of O's extra token is vision-only too
''')
rep('''    with torch.no_grad():
        m.e_level.weight.copy_(torch.randn(m.e_level.weight.shape, generator=g) * 0.5)
        m.e_level.bias.copy_(torch.randn(m.e_level.bias.shape, generator=g) * 0.5)
        wo = m.head_o[2].weight''', '''    with torch.no_grad():
        if m.e_level is not None:                                    # PREREG_LEVELHEAD_A2: absent under --aux-cond token (e_aux has a default, non-zero init)
            m.e_level.weight.copy_(torch.randn(m.e_level.weight.shape, generator=g) * 0.5)
            m.e_level.bias.copy_(torch.randn(m.e_level.bias.shape, generator=g) * 0.5)
        wo = m.head_o[2].weight''')
rep('''    saved = (m.e_level.weight.detach().clone(), m.e_level.bias.detach().clone())
    with torch.no_grad():
        m.e_level.weight.zero_()
        m.e_level.bias.zero_()
    _, _, _, tr_z = run(0.0)
    with torch.no_grad():
        m.e_level.weight.copy_(saved[0])
        m.e_level.bias.copy_(saved[1])
''', '''    if m.e_level is not None:
        saved = (m.e_level.weight.detach().clone(), m.e_level.bias.detach().clone())
        with torch.no_grad():
            m.e_level.weight.zero_()
            m.e_level.bias.zero_()
        _, _, _, tr_z = run(0.0)
        with torch.no_grad():
            m.e_level.weight.copy_(saved[0])
            m.e_level.bias.copy_(saved[1])
        reach_grad = {"e_level_grad_at_w0": asum(g_0, "e_level.")}
    else:                                                            # PREREG_LEVELHEAD_A2 --aux-cond token: the level enters through column 0 of e_aux
        saved = m.e_aux.weight.detach().clone()
        with torch.no_grad():
            m.e_aux.weight[:, 0].zero_()
        _, _, _, tr_z = run(0.0)
        with torch.no_grad():
            m.e_aux.weight.copy_(saved)
        reach_grad = {"e_aux_level_col_grad_at_w0": float(g_0["e_aux.weight"][:, 0].abs().sum())}
''')
rep('''"e_level_grad_at_w0": asum(g_0, "e_level."),
''', '''**reach_grad, "aux_vision_only": aux_vis,
''')
rep('''    if not res["e_level_grad_at_w0"] > 0.0 or not res["plan_moves_with_token"] > 0.0:
        raise AssertionError(f"level probe: the level token does not reach O (e_level grad {res['e_level_grad_at_w0']}, plan moves by {res['plan_moves_with_token']} when e_level is zeroed)")''',
    '''    if not list(reach_grad.values())[0] > 0.0 or not res["plan_moves_with_token"] > 0.0:
        raise AssertionError(f"level probe: the level token does not reach O ({reach_grad}, plan moves by {res['plan_moves_with_token']} when its path is zeroed)")''')

# ---------------------------------------------------------------------------------------------- main(): labels, probe batches, the probe, selfcheck, loop, summary
ins_after('info["ds_window_start"] = ds.window_start', '''    LL = make_lead_labels(a, cfg, J, info)                          # PREREG_LEVELHEAD_A2: the lead head's label arrays (None with the head off: nothing read, nothing built)
''')
ins_after('probe["level_y"], probe["level_ok"] = level_stage(probe)', '''        if cfg.lead_head == "on":                        # PREREG_LEVELHEAD_A2: as above (the lead labels, looked up by the probe's own clip index and step)
            probe["lead_y"], probe["lead_has"], probe["lead_near"] = lead_stage(probe, LL)
''')
ins_after('probe_f["level_y"], probe_f["level_ok"] = level_stage(probe_f)', '''        if cfg.lead_head == "on":                        # PREREG_LEVELHEAD_A2: as above
            probe_f["lead_y"], probe_f["lead_has"], probe_f["lead_near"] = lead_stage(probe_f, LL)
''')
ins_after('raise SystemExit(f"ZZDVB-REFUSEDZZ level-head: {e}")', '''    try:
        ldp = lead_head_probe(model, probe, cfg)          # PREREG_LEVELHEAD_A2: G-LIVE on the loop's own label call (lead_stage); the term reaches the loss with its weight, BOTH outputs get their gradient, the detach control reads EXACTLY 0.0, the lead output reaches O, the head is vision-only (off: nothing run)
    except (AssertionError, ValueError) as e:
        raise SystemExit(f"ZZDVB-REFUSEDZZ lead-head: {e}")
''')
ins_after('sc["level_head_probe"] = lvp', '''    if cfg.lead_head == "on":                                           # PREREG_LEVELHEAD_A2: lever-ON arms only (a flag-off selfcheck.json keeps its key set)
        sc["lead_head_probe"] = ldp
        sc["lead_labels"] = info["lead_labels"]
''')
ins_after('level_on = cfg.level_head == "on"', '''    lead_on = cfg.lead_head == "on"                                # PREREG_LEVELHEAD_A2: off (the default) = the loop runs no new line
''')
ins_after("lvl_lab = level_stage(b)", '''            if lead_on:                                            # PREREG_LEVELHEAD_A2: the lead label of the LOGGED frame (clip index and step), taken BEFORE any ego stage (a swapped sample's ci / t is the own one)
                lead_lab = lead_stage(b, LL)
''')
ins_after('b["level_y"], b["level_ok"] = lvl_lab', '''            if lead_on:                                            # PREREG_LEVELHEAD_A2: attached AFTER the stages: the loss reads b["lead_y"], b["lead_has"], b["lead_near"]
                b["lead_y"], b["lead_has"], b["lead_near"] = lead_lab
''')
ins_after('summary["level_head"], summary["w_level"] = "on", cfg.w_level', '''    if lead_on or cfg.aux_cond == "token":                                 # PREREG_LEVELHEAD_A2: the lever-ON arm's run record (a flag-off summary.json keeps its key set)
        summary["aux_cond"] = cfg.aux_cond
    if lead_on:
        summary["lead_head"], summary["w_lead"] = "on", cfg.w_lead
        summary["lead_labels"] = info["lead_labels"]
''')

assert "\r\n" not in t
dst.write_bytes(t.encode("utf-8"))
print("wrote", dst, len(t))
