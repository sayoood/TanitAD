"""Reproduces the PREREG_V8F_MRATE edit of stack/scripts/v8f_mini_train.py on a fresh tip: `--wm-k`, `--wm-only`, their G-DVB entries and clauses, the WM-only loader, G-LIVE and the planner-identity probe, and the loop's loss call."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8", newline="").read()
assert "wm_only" not in s, "already patched"


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


# ---- 1. parser
rep('''    ap.add_argument("--w-wm", type=float, default=0.1)
''', '''    ap.add_argument("--w-wm", type=float, default=0.1)
    # ---- PREREG_V8F_MRATE: the multi-rate world model and the WM-only mode. Both default to the earlier behaviour; both have their G-DVB entry in FLAG_CHECKS and their clause in bad_wm_only.
    ap.add_argument("--wm-k", dest="wm_k", type=int, default=1,
                    help="PREREG_V8F_MRATE: how many past states each world-model level reads, spaced at that level's own step. 1 (default) = W0 = every earlier arm, bit for bit (no module and no state-dict key is added). "
                         "K > 1 (W1: 3): L1 reads the adapter tokens of frames t, t-1, .., t-K+1; L2 / L3 read K states of the level below, 0.4 s / 0.8 s apart. Needs --wm b")
    ap.add_argument("--wm-only", dest="wm_only", action="store_true", default=False,
                    help="PREREG_V8F_MRATE: WM-only training. Needs --init-ckpt (a plain M2b s{k}): its adapter, readers and planner load strictly and bit-equal and are FROZEN; its world-model parameters are NOT loaded -- the WM is "
                         "re-initialised under --seed (hmax is not compared: M2b trained at 16, this arm at 28); only the WM loss terms are trained. Every other lever must be at its default (a lever that is declared and does nothing is refused). Default off")
''')
# ---- 2. FLAG_CHECKS
rep('''    "init_ckpt": "data:init_ckpt",''', '''    # PREREG_V8F_MRATE: `lever:wm_k` = equals built_levers().get("wm_k", 1) (the BUILT modules' window / dilation / causal pad, WMAux.built_k); `lever:wm_only` = the record the WM-only loader read back (bad_wm_only: non-WM tensors bit-equal to the checkpoint,
    # NO WM tensor loaded, the WM re-initialised under --seed, trainable == the WM's parameters exactly, every other lever at its default) -> wm_only_live_check (G-LIVE) + planner_identity_probe -> `model.wm_only_loss(b)` in the loop.
    "wm_k": "lever:wm_k", "wm_only": "lever:wm_only",
    "init_ckpt": "data:init_ckpt",''')
# ---- 3. make_config
rep('''p_gt_start=a.p_gt_start, p_gt_end=a.p_gt_end, w_ckpt=a.w_ckpt, w_tac=a.w_tac,
                           w_cons=a.w_cons, w_traj=a.w_traj, w_pen=a.w_pen, w_pen_self=a.w_pen_self, w_wm=a.w_wm,''',
    '''p_gt_start=a.p_gt_start, p_gt_end=a.p_gt_end, w_ckpt=a.w_ckpt, w_tac=a.w_tac,
                           w_cons=a.w_cons, w_traj=a.w_traj, w_pen=a.w_pen, w_pen_self=a.w_pen_self, w_wm=a.w_wm, wm_k=a.wm_k,''')
# ---- 4. loader + clauses + probes, inserted before LRSchedule
NEW_FUNCS = '''#: PREREG_V8F_MRATE: every other lever a WM-only argv must leave at its parser default (the readers, heads and planner are frozen and never run: a lever on them is declared and does nothing)
WM_ONLY_DEFAULT_DESTS = ("navonly", "no_consistency", "cons_absent", "code_mix", "o_interface", "w_acc", "launch_intent_w", "level_head", "w_level", "lead_head", "w_lead", "aux_cond", "detach_upper",
                         "ego_perturb_p", "ego_perturb_yawrate_std", "dagger_p", "dagger_bank", "dagger_bank_sha256", "ego_perturb_with_dagger", "recover_p", "recover_dev_bank", "recover_dev_bank_sha256",
                         "encoder", "lora_lr", "frames_root", "accum", "enc_ckpt", "window_start")


def is_wm_param(name: str) -> bool:
    return name.startswith("wm.")


def freeze_all_but_wm(model) -> None:
    """PREREG_V8F_MRATE: the WM-only mode trains the world model and nothing else. Called BEFORE the optimiser is built (its decay / nodecay lists filter on requires_grad)."""
    for n, p in model.named_parameters():
        p.requires_grad_(is_wm_param(n))


def load_wm_only_init(path, model, cfg, seed: int, apply: bool = True) -> dict:
    """Start state of a WM-only run (PREREG_V8F_MRATE, the Master Mind's reading of 2026-11-07 13:09). Hash the file; REFUSE a checkpoint whose architecture config differs from the BUILT one in any ``INIT_CKPT_MATCH`` key EXCEPT
    ``hmax`` (M2b trained at 16, this arm at 28: hmax only sizes the WM sequence); then, if ``apply``, load EVERY non-``wm.`` tensor strictly (the missing keys are exactly the model's ``wm.*`` keys, nothing is unexpected), READ THEM BACK
    (``n_mismatch_after_load``), and replace the world model by a FRESH ``WMAux`` built right after ``torch.manual_seed(seed)`` (none of the checkpoint's WM tensors is loaded; ``wm_params_equal_to_ckpt`` counts the shape-matching float
    parameters that happen to equal the checkpoint's: it must be 0). ``apply`` is False when ``--out`` already holds a ckpt_last.pt (resume wins)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck or not isinstance(ck.get("cfg"), dict):
        raise ValueError("not a v8f_mini_train checkpoint (no 'model' / 'cfg')")
    norm = lambda v: tuple(v) if isinstance(v, (list, tuple)) else v          # noqa: E731
    diff = {k: (norm(ck["cfg"][k]), norm(getattr(cfg, k))) for k in INIT_CKPT_MATCH if k != "hmax" and k in ck["cfg"] and norm(ck["cfg"][k]) != norm(getattr(cfg, k))}
    if diff:
        raise ValueError("config mismatch between the checkpoint and the model this argv builds (checkpoint vs built; hmax is not compared): " + "; ".join(f"{k}: {v[0]!r} != {v[1]!r}" for k, v in sorted(diff.items())))
    sd = ck["model"]
    rec = {"path": os.path.abspath(path), "sha256": h.hexdigest(), "applied": bool(apply), "skipped": None if apply else "resume", "n_tensors": len(sd), "n_mismatch_after_load": None,
           "source_step": ck.get("step"), "source_arm": ck.get("arm"), "source_seed": ck.get("seed"), "source_argv_sha": ck.get("argv_sha"), "mode": "wm_only",
           "hmax_ckpt": ck["cfg"].get("hmax"), "hmax_built": cfg.hmax, "wm_loaded_from_ckpt": 0, "wm_reinit_seed": None, "wm_params_equal_to_ckpt": None}
    if not apply:
        return rec
    rest = {k: v for k, v in sd.items() if not is_wm_param(k)}
    res = model.load_state_dict(rest, strict=False)
    wm_keys = sorted(k for k in model.state_dict() if is_wm_param(k))
    if res.unexpected_keys:
        raise ValueError(f"the checkpoint carries non-WM keys the built model does not have: {sorted(res.unexpected_keys)[:8]}")
    if sorted(res.missing_keys) != wm_keys:
        raise ValueError(f"the built model lacks checkpoint values for keys that are not its WM's: {sorted(set(res.missing_keys) - set(wm_keys))[:8]}")
    cur = model.state_dict()
    rec["n_mismatch_after_load"] = sum(1 for k in rest if not torch.equal(cur[k].detach().cpu(), rest[k].detach().cpu()))
    dev = next(model.parameters()).device
    torch.manual_seed(seed)                                                    # the WM is a function of (seed, cfg) only: W0 and W1 of one seed share every non-level-encoder parameter bit for bit
    model.wm = M.WMAux(cfg).to(dev)
    rec["wm_reinit_seed"] = seed
    old = {k: v for k, v in sd.items() if is_wm_param(k)}
    rec["wm_params_equal_to_ckpt"] = sum(1 for n, q in model.named_parameters() if is_wm_param(n) and n in old and old[n].shape == q.shape and torch.equal(old[n].detach().cpu(), q.detach().cpu()))
    return rec


def bad_wm_only(a, cfg, model, data_info) -> list:
    """G-DVB for ``--wm-k`` and ``--wm-only`` (PREREG_V8F_MRATE). ``--wm-k`` is read back from the BUILT modules (``WMAux.built_k``: window, dilation, causal pad per level). ``--wm-only`` is read back from the loader's record and the
    parameters' ``requires_grad``: the checkpoint's non-WM tensors are in the model bit-equal, no WM tensor was loaded, the WM was re-initialised under the run seed, the trainable set is exactly the WM's parameters, and every other
    lever is at its default."""
    bad = []
    built = model.built_levers()
    if built.get("wm_k", 1) != a.wm_k:
        bad.append(f"lever wm_k: declared {a.wm_k} but built {built.get('wm_k', 1)}")
    if a.wm_k > 1:
        want = {"wm_k_levels": [a.wm_k] * 3, "wm_dilation_levels": [1, *cfg.wm_strides], "wm_causal_pad_levels": [True] * 3}
        for k, v in want.items():
            if built.get(k) != v:
                bad.append(f"lever wm_k {a.wm_k}: built {k} {built.get(k)} != {v}")
    elif model.wm is not None and (model.wm.enc1 is not None or any(k != 1 for k in model.wm.built_k()["k"])):
        bad.append("wm_k 1 declared but a multi-frame world-model module was built")
    rec = data_info.get("wm_only")
    if not a.wm_only:
        if rec:
            bad.append("wm_only not declared but a WM-only start state was loaded")
        return bad
    if model.wm is None:
        return bad + ["wm_only declared but no world model was built (--wm off)"]
    if not a.init_ckpt:
        bad.append("wm_only declared without --init-ckpt: nothing to freeze the readers and the planner at")
    if not rec:
        return bad + ["wm_only declared but no WM-only start state was loaded"]
    if not cfg.w_wm > 0.0:
        bad.append(f"wm_only declared but w_wm is {cfg.w_wm}: the only loss term has weight 0 -- a lever that is declared and does nothing")
    dfl = build_parser()
    for d in WM_ONLY_DEFAULT_DESTS:
        if getattr(a, d) != dfl.get_default(d):
            bad.append(f"wm_only: --{d.replace('_', '-')} is {getattr(a, d)!r} (default {dfl.get_default(d)!r}): the readers, heads and planner are frozen and never run -- a lever that is declared and does nothing")
    if a.arm != "M2b":
        bad.append(f"wm_only: arm {a.arm} (the card trains the world model of M2b)")
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    wm_names = [n for n, _ in model.named_parameters() if is_wm_param(n)]
    if not wm_names or sorted(names) != sorted(wm_names):
        bad.append(f"wm_only: trainable parameters are not exactly the WM's ({len(names)} trainable, {len(wm_names)} WM; stray: {sorted(set(names) - set(wm_names))[:4]})")
    if rec.get("applied"):
        if rec.get("wm_loaded_from_ckpt") != 0:
            bad.append(f"wm_only: {rec.get('wm_loaded_from_ckpt')} world-model tensors were loaded from the checkpoint (the card re-initialises the WM)")
        if rec.get("wm_reinit_seed") != a.seed:
            bad.append(f"wm_only: the WM was re-initialised under seed {rec.get('wm_reinit_seed')}, not --seed {a.seed}")
        if rec.get("wm_params_equal_to_ckpt") != 0:
            bad.append(f"wm_only: {rec.get('wm_params_equal_to_ckpt')} WM parameters equal the checkpoint's (the WM must be fresh)")
        if rec.get("mode") != "wm_only":
            bad.append(f"wm_only: the start state was loaded in mode {rec.get('mode')!r}")
    return bad


def wm_only_live_check(model, batch, cfg, steps: int = 3, lr: float = 1e-3) -> dict:
    """G-LIVE for the WM-only mode (PREREG_V8F_MRATE control 3): three optimiser steps of ``wm_only_loss`` on a real batch, on a COPY with EVERY parameter thawed (so a missing ``.detach()`` shows up as a gradient on the adapter). Every WM loss term
    finite; every WM parameter receives a non-zero gradient at some step (a=without IDM: the IDM heads are declared dead); the adapter, the readers, the heads and the planner receive EXACTLY 0.0. ``--w-wm 0`` zeroes every WM gradient and FAILS."""
    m = copy.deepcopy(model)
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    wm_params = [p for n, p in m.named_parameters() if is_wm_param(n)]
    opt = torch.optim.AdamW(wm_params, lr=lr)
    acc = {n: 0.0 for n, _ in m.named_parameters()}
    terms_seen = None
    for s in range(steps):
        opt.zero_grad(set_to_none=False)
        for n, p in m.named_parameters():
            if not is_wm_param(n):
                p.grad = None
        torch.manual_seed(s)
        T = m.wm_only_loss(batch)["terms"]
        for k, v in T.items():
            if not torch.isfinite(v):
                raise AssertionError(f"G-LIVE: WM loss term {k} is not finite at step {s}: {float(v)}")
        terms_seen = {k: float(v.detach()) for k, v in T.items()}
        T["total"].backward()
        for n, p in m.named_parameters():
            if p.grad is not None:
                acc[n] += float(p.grad.abs().sum())
        opt.step()
    expect = {"wm", "total"} | {f"wm/{t}/L{l}" for t in ("pred", "sigreg", "idm", "level") for l in (1, 2, 3)} | {f"wm/act_sigreg/L{l}" for l in (2, 3)}
    missing = sorted(expect - set(terms_seen))
    declared_dead = [n for n in acc if cfg.wm == "a" and ".idm_heads." in n]
    dead = sorted(n for n, g in acc.items() if is_wm_param(n) and g == 0.0)
    unexpected_dead = sorted(set(dead) - set(declared_dead))
    groups = {}
    for n, g in acc.items():
        if is_wm_param(n):
            parts = n.split(".")                                           # wm.proj / wm.enc1 / wm.stack.act_embed / wm.stack.<encoders|macro_encoders|predictors|idm_heads>.<level>
            key = ".".join(parts[:4] if parts[1] == "stack" and parts[2] in ("encoders", "macro_encoders", "predictors", "idm_heads") else parts[:3] if parts[1] == "stack" else parts[:2])
            groups[key] = groups.get(key, 0.0) + g
    dead_groups = sorted(k for k, g in groups.items() if g == 0.0 and not (cfg.wm == "a" and ".idm_heads." in k))
    nonwm = {n: g for n, g in acc.items() if not is_wm_param(n)}
    res = {"terms": terms_seen, "missing_terms": missing, "wm_groups_grad": groups, "dead_wm_groups": dead_groups, "dead_params": dead, "declared_dead": declared_dead, "unexpected_dead": unexpected_dead,
           "n_wm_params": len(wm_params), "control_nonwm_grad_sum": float(sum(nonwm.values())), "control_adapter_grad_sum": float(sum(g for n, g in nonwm.items() if n.startswith("adapter.") or n == "pos")),
           "nonwm_with_grad": sorted(n for n, g in nonwm.items() if g != 0.0)[:8], "n_nonwm_params": len(nonwm)}
    ok = (not missing and not dead_groups and not unexpected_dead and res["control_nonwm_grad_sum"] == 0.0 and res["n_wm_params"] > 0 and res["n_nonwm_params"] > 0)
    res["pass"] = bool(ok)
    if not ok:
        raise AssertionError("G-LIVE (wm_only) failed: " + json.dumps({k: v for k, v in res.items() if k != "wm_groups_grad"}, default=str)[:1500])
    return res


def _tensors_of(o, prefix=""):
    if isinstance(o, torch.Tensor):
        yield prefix, o
    elif isinstance(o, dict):
        for k in sorted(o, key=str):
            yield from _tensors_of(o[k], f"{prefix}/{k}")


def planner_identity_probe(model, batch, cfg, steps: int = 3, lr: float = 1e-3) -> dict:
    """PREREG_V8F_MRATE control 1 (planner identity), the in-trainer form: ``forward_infer`` on a fixed batch (eval, CPU, a COPY) is compared tensor by tensor with ``torch.equal`` before and after ``steps`` WM-only optimiser steps. The WM must
    have MOVED (the control: the steps did something) and every planner output must be bit-equal."""
    m = copy.deepcopy(model)
    m.eval()
    with torch.no_grad():
        before = {k: v.clone() for k, v in _tensors_of(m.forward_infer(batch))}
    wm0 = {n: p.detach().clone() for n, p in m.named_parameters() if is_wm_param(n)}
    m.train()
    opt = torch.optim.AdamW([p for n, p in m.named_parameters() if p.requires_grad], lr=lr)
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(s)
        m.wm_only_loss(batch)["terms"]["total"].backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        after = {k: v for k, v in _tensors_of(m.forward_infer(batch))}
    same = sorted(k for k in before if k in after and before[k].shape == after[k].shape and torch.equal(before[k], after[k]))
    moved = sum(1 for n, p in m.named_parameters() if is_wm_param(n) and not torch.equal(p.detach(), wm0[n]))
    res = {"n_tensors": len(before), "n_equal": len(same), "differing": sorted(set(before) - set(same))[:8], "wm_params_moved": moved, "n_wm_params": len(wm0), "steps": steps, "pass": False}
    res["pass"] = bool(len(before) > 0 and len(same) == len(before) and set(before) == set(after) and moved > 0)
    if not res["pass"]:
        raise AssertionError("planner identity failed: " + json.dumps(res, default=str)[:1200])
    return res


class LRSchedule:'''
rep('''class LRSchedule:''', NEW_FUNCS)
# ---- 5. declared_vs_built hook
rep('''    bad += bad_level(a, model, data_info)                                                        # PREREG_V8F_LEVEL: --level-head / --w-level read back from the BUILT model
''', '''    bad += bad_level(a, model, data_info)                                                        # PREREG_V8F_LEVEL: --level-head / --w-level read back from the BUILT model
    bad += bad_wm_only(a, cfg, model, data_info)                                                 # PREREG_V8F_MRATE: --wm-k read back from the BUILT WM modules, --wm-only from the loader's record and the frozen set
''')
# ---- 6. main: loader + freeze
rep('''    if a.init_ckpt:                                                  # O-RECOVER's start state: the MODEL weights of an earlier checkpoint (before the optimiser, the probes and the gate); a resume of THIS run wins
        try:
            info["init_ckpt"] = load_init_ckpt(a.init_ckpt, model, cfg, apply=not (os.path.exists(os.path.join(a.out, "ckpt_last.pt")) and not a.no_resume))
        except Exception as e:''', '''    if a.init_ckpt:                                                  # O-RECOVER's start state: the MODEL weights of an earlier checkpoint (before the optimiser, the probes and the gate); a resume of THIS run wins
        try:
            _apply = not (os.path.exists(os.path.join(a.out, "ckpt_last.pt")) and not a.no_resume)
            if a.wm_only:                                            # PREREG_V8F_MRATE: the non-WM tensors load strictly and bit-equal, hmax is not compared, the WM is re-initialised under --seed
                info["init_ckpt"] = info["wm_only"] = load_wm_only_init(a.init_ckpt, model, cfg, a.seed, apply=_apply)
            else:
                info["init_ckpt"] = load_init_ckpt(a.init_ckpt, model, cfg, apply=_apply)
        except Exception as e:''')
rep('''    dag = make_dagger(a, J, wins, clips, info)                       # PREREG_V8F_DAGGER: off by default''', '''    if a.wm_only:
        freeze_all_but_wm(model)                                     # PREREG_V8F_MRATE: BEFORE the optimiser below (its decay / nodecay lists filter on requires_grad); bad_wm_only reads the frozen set back
    dag = make_dagger(a, J, wins, clips, info)                       # PREREG_V8F_DAGGER: off by default''')
# ---- 7. main: the probes
rep('''        live = live_check(model.cpu(), probe, cfg)
    else:''', '''        if a.wm_only:                                    # PREREG_V8F_MRATE: G-LIVE of the WM-only mode + the planner identity (the other probes need gradients on the frozen readers and are not run)
            live = wm_only_live_check(model.cpu(), probe, cfg)
            pid = planner_identity_probe(model.cpu(), probe, cfg)
        else:
            live = live_check(model.cpu(), probe, cfg)
    else:''')
a0 = s.index("    model.to(dev)\n    try:\n        mix = code_mix_probe(")
a1 = s.index("    levers = model.built_levers()\n    levers[\"ego_perturb\"]")
block = s[a0:a1]
head = "    model.to(dev)\n"
assert block.startswith(head)
body = block[len(head):]
ind = "".join(("    " + ln if ln.strip() else ln) for ln in body.splitlines(True))
new_block = head + '''    if a.wm_only:                                                          # PREREG_V8F_MRATE: the readers, heads and planner are frozen and never run; the probes that need their gradients are skipped (and recorded as skipped)
        mix = ocut = accp = egop = dagp = recp = lvp = ldp = {"skipped": "wm_only"}
        lip = {"skipped": "wm_only", "launch_n": 0, "launch_eligible": 0}
        cmpp = {"skipped": "wm_only", "overlap": 0}
        adm = M.assert_no_situation_classifier_input(model.cpu().eval(), probe)
        model.to(dev).train()
    else:
''' + ind
s = s[:a0] + new_block + s[a1:]
rep('''    sc["launch_intent_probe"] = lip
''', '''    sc["launch_intent_probe"] = lip
    if a.wm_only:                                                       # WM-only arms only: a flag-off selfcheck.json keeps its key set
        sc["planner_identity_probe"] = pid
''')
rep('''    levers["init_ckpt"] = info.get("init_ckpt")
''', '''    levers["init_ckpt"] = info.get("init_ckpt")
    if a.wm_only:                                                       # WM-only arms only (a flag-off run record keeps its key set)
        levers["wm_only"] = {"wm_only": True, "wm_k": a.wm_k, "n_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)), "n_wm": int(sum(p.numel() for n, p in model.named_parameters() if is_wm_param(n))),
                             "init": info.get("wm_only")}
''')
# ---- 8. save / loop / log / summary
rep('''        if launch_on:                                                   # lever-ON arms only: a flag-off checkpoint keeps its key set
            blob["launch_totals"] = dict(launch_tot)
''', '''        if launch_on:                                                   # lever-ON arms only: a flag-off checkpoint keeps its key set
            blob["launch_totals"] = dict(launch_tot)
        if a.wm_only:                                                   # WM-only arms only
            blob["wm_only"] = True
''')
rep('''                    out = model.loss(b, p_gt=p_gt, gen=None)
''', '''                    out = model.wm_only_loss(b) if a.wm_only else model.loss(b, p_gt=p_gt, gen=None)       # PREREG_V8F_MRATE: ONLY the WM terms (the adapter is detached inside, the rest is frozen and not run)
''')
rep('''                print(f"[{step}/{total}] ade {rec['ade']:.3f} total {rec['total']:.3f} p_gt {p_gt:.2f}", flush=True)''',
    '''                print(f"[{step}/{total}] ade {rec.get('ade', float('nan')):.3f} total {rec['total']:.3f} p_gt {p_gt:.2f}", flush=True)''')
rep('''    if enc is not None:
        summary["encoder"] = "lora"
''', '''    if enc is not None:
        summary["encoder"] = "lora"
    if a.wm_only:                                                          # PREREG_V8F_MRATE: the WM-only arm's run record (a flag-off summary.json keeps its key set)
        summary["wm_only"], summary["wm_k"] = True, a.wm_k
''')
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched", p)
