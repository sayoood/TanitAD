"""PREREG_V8F_NAVRB2: patch v8f_mini_train.py (+ built_levers in v8f_mini.py) in the tree export (exact, unique replacements; refuses on any miss)."""
import sys

ROOT = sys.argv[1]


def sub(path, pairs):
    p = ROOT + "/" + path
    s = open(p, "rb").read().decode("utf-8")
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"{path}: {n} matches for {old[:90]!r}")
        s = s.replace(old, new)
    open(p, "wb").write(s.encode("utf-8"))
    print("patched", path, len(pairs))


sub("stack/tanitad/models/v8f_mini.py", [
    ('                "residual_zero_init": bool(float(self.head_o[2].weight.detach().abs().sum()) == 0.0),\n',
     '                "residual_zero_init": bool(float(self.head_o[2].weight.detach().abs().sum()) == 0.0),\n'
     '                "nav_rb": "on" if self.e_nav_rb is not None else "off",       # PREREG_V8F_NAVRB2: the BUILT roundabout channel (e_nav_rb exists iff on)\n'),
])

T = "stack/scripts/v8f_mini_train.py"
NAVRB_FN = '''

#: PREREG_V8F_NAVRB2: the stages that move the ego frame or rebuild the nav with no rule for nav_rb (the recipe uses DAgger only, whose swap holds nav_rb as logged)
NAVRB_REFUSED_WITH = ("ego_perturb_p", "recover_p", "t2_bank_p")


def bad_navrb(a, model, data_info=None, opt=None) -> list:
    """PREREG_V8F_NAVRB2: the five --nav-rb* flags read back from the BUILT model (e_nav_rb iff on, zero at init is the probe's), the LOADED channel array (sha256 re-hashed, variant applied) and the
    optimiser groups (navrb holds lr x nav_rb_lr_mult and exactly e_nav_rb.weight). Off forbids every other nav-rb flag at a non-default value: a flag that does nothing is refused."""
    bad, info = [], (data_info or {})
    on = a.nav_rb == "on"
    built = model.built_levers().get("nav_rb", "off")
    if built != a.nav_rb or (model.e_nav_rb is not None) != on or model.cfg.nav_rb != a.nav_rb:
        bad.append(f"lever nav_rb: declared {a.nav_rb} but built {built} (cfg {model.cfg.nav_rb}, module {'present' if model.e_nav_rb is not None else 'absent'})")
    if not (math.isfinite(a.nav_rb_lr_mult) and a.nav_rb_lr_mult > 0.0):
        bad.append(f"nav_rb_lr_mult {a.nav_rb_lr_mult!r} is not a finite number > 0")
    rec = info.get("nav_rb")
    if on:
        if not a.nav_rb_npz or not os.path.exists(a.nav_rb_npz):
            bad.append(f"nav_rb on but --nav-rb-npz {a.nav_rb_npz!r} does not exist")
        if not rec or rec.get("sha256") != a.nav_rb_sha256 or not a.nav_rb_sha256:
            bad.append(f"nav_rb: declared sha256 {a.nav_rb_sha256!r} but the loaded file is {(rec or {}).get('sha256')!r}")
        if not rec or rec.get("variant") != a.nav_rb_variant:
            bad.append(f"nav_rb_variant: declared {a.nav_rb_variant} but the loaded array is {(rec or {}).get('variant')!r}")
        elif a.nav_rb_variant == "zero" and rec.get("n_on") != 0:
            bad.append(f"nav_rb_variant zero but {rec.get('n_on')} rows are on")
        elif a.nav_rb_variant == "real" and not rec.get("n_on", 0) > 0:
            bad.append("nav_rb_variant real but no row is on: the channel would carry nothing")
        for k in NAVRB_REFUSED_WITH:
            if float(getattr(a, k, 0.0) or 0.0) > 0.0:
                bad.append(f"--nav-rb on with --{k.replace('_', '-')} {getattr(a, k)}: that stage has no rule for nav_rb (PREREG_V8F_NAVRB2 uses DAgger only)")
        if getattr(a, "ego_perturb_with_dagger", False):
            bad.append("--nav-rb on with --ego-perturb-with-dagger: the composed stage has no rule for nav_rb")
        if opt is not None:
            gs = [g for g in opt.param_groups if g.get("name") == "navrb"]
            ids = {id(p) for p in model.e_nav_rb.parameters()} if model.e_nav_rb is not None else set()
            if len(gs) != 1 or {id(p) for p in gs[0]["params"]} != ids:
                bad.append("nav_rb on but the optimiser has no single navrb group holding exactly e_nav_rb's parameters")
            elif abs(gs[0]["lr"] - a.lr * a.nav_rb_lr_mult) > 1e-15 * max(1.0, a.lr) or gs[0].get("lr_mult") != a.nav_rb_lr_mult:
                bad.append(f"nav_rb_lr_mult: declared {a.nav_rb_lr_mult} but the navrb group holds lr {gs[0]['lr']} / lr_mult {gs[0].get('lr_mult')}")
            other = [g for g in opt.param_groups if g.get("name") != "navrb" and ids & {id(p) for p in g["params"]}]
            if other:
                bad.append("e_nav_rb's parameters are also in another optimiser group")
    else:
        if a.nav_rb_npz or a.nav_rb_sha256 or a.nav_rb_variant != "real" or a.nav_rb_lr_mult != 1.0:
            bad.append("--nav-rb off with --nav-rb-npz / --nav-rb-sha256 / --nav-rb-variant / --nav-rb-lr-mult named at a non-default value: the channel is not read, so they are declared and do nothing")
        if rec is not None:
            bad.append("--nav-rb off but a nav_rb array was loaded")
        if opt is not None and any(g.get("name") == "navrb" for g in opt.param_groups):
            bad.append("--nav-rb off but the optimiser carries a navrb group")
    return bad


def navrb_probe(model, batch, cfg) -> dict:
    """PREREG_V8F_NAVRB2 G-LIVE (reachability from the caller): on a COPY of the model, (a) forward_infer reads ``nav_rb``; (b) the step-0 identity: with e_nav_rb at its zero init, every forward output
    is bit-identical under the real channel and the channel zeroed; (c) the total loss gives e_nav_rb.weight a gradient > 0 on the real batch (ones planted on a copy when the batch's read frames carry none,
    recorded); (d) the zero control: the channel all 0.0 gives EXACTLY 0.0. Raises AssertionError on any failure."""
    import copy
    import torch
    if model.e_nav_rb is None:
        raise AssertionError("navrb probe: the model has no e_nav_rb (the probe runs only for --nav-rb on)")
    m = copy.deepcopy(model).cpu()
    b = {k: (v.cpu() if hasattr(v, "cpu") else ({kk: vv.cpu() if hasattr(vv, "cpu") else vv for kk, vv in v.items()} if isinstance(v, dict) else v)) for k, v in batch.items()}
    res = {"init_weight_abs_sum": float(m.e_nav_rb.weight.detach().abs().sum()), "n_on_batch": int(b["nav_rb"].sum())}
    m.eval()
    with torch.no_grad():
        res["keys_read_has_nav_rb"] = "nav_rb" in m.forward_infer(b)["_keys_read"]
        if res["init_weight_abs_sum"] == 0.0:
            o1 = m.forward_infer(b)
            o0 = m.forward_infer({**b, "nav_rb": torch.zeros_like(b["nav_rb"])})
            res["step0_identity"] = all(torch.equal(o1[k], o0[k]) for k in o1 if isinstance(o1.get(k), torch.Tensor))
        else:
            res["step0_identity"] = None                                            # a resumed / trained weight: the identity is a step-0 property only
    m.train()
    planted = False
    if res["n_on_batch"] == 0:
        b = {**b, "nav_rb": torch.ones_like(b["nav_rb"])}
        planted = True
    res["planted"] = planted

    def grad(bb):
        m.zero_grad(set_to_none=True)
        t = m.loss(bb, p_gt=0.5, gen=torch.Generator().manual_seed(0))["terms"]["total"]
        g = torch.autograd.grad(t, [m.e_nav_rb.weight], allow_unused=True)[0]
        return 0.0 if g is None else float(g.abs().sum())
    res["e_nav_rb_grad"] = grad(b)
    res["control_e_nav_rb_grad_zero_channel"] = grad({**b, "nav_rb": torch.zeros_like(b["nav_rb"])})
    if not res["keys_read_has_nav_rb"]:
        raise AssertionError("navrb probe: forward_infer does not read nav_rb")
    if res["step0_identity"] is False:
        raise AssertionError("navrb probe: at the zero init the outputs differ under the real and the zeroed channel")
    if not res["e_nav_rb_grad"] > 0.0:
        raise AssertionError(f"navrb probe: e_nav_rb.weight gets gradient {res['e_nav_rb_grad']!r} from the total loss (must be > 0)")
    if res["control_e_nav_rb_grad_zero_channel"] != 0.0:
        raise AssertionError(f"navrb probe: the zero-channel control reads {res['control_e_nav_rb_grad_zero_channel']!r} (must be exactly 0.0)")
    res["pass"] = True
    return res
'''

sub(T, [
    ('    "ctx_lr_mult": "opt:ctx_lr_mult", "ctx_detach": "lever:ctx_detach",\n',
     '    "ctx_lr_mult": "opt:ctx_lr_mult", "ctx_detach": "lever:ctx_detach",\n'
     '    # PREREG_V8F_NAVRB2: argv -> make_config -> the BUILT model (e_nav_rb iff on) / the navrb optimiser group / the LOADED nav_rb array -> declared_vs_built (bad_navrb) -> navrb_probe (G-LIVE) -> the window item and\n'
     '    # collate -> V8FMini._nav_tok in S, T and O -> the total loss. With --nav-rb off (the default) nothing is built or read.\n'
     '    "nav_rb": "lever:nav_rb", "nav_rb_npz": "path", "nav_rb_sha256": "data:nav_rb_sha256", "nav_rb_variant": "data:nav_rb_variant", "nav_rb_lr_mult": "opt:nav_rb_lr_mult",\n'),
    ('                    help="PREREG_V8F_STRAT1: off (default) = the context loss reaches the shared adapter (a declared training-time coupling); on = the head reads the tokens detached")\n',
     '                    help="PREREG_V8F_STRAT1: off (default) = the context loss reaches the shared adapter (a declared training-time coupling); on = the head reads the tokens detached")\n'
     '    # ---- PREREG_V8F_NAVRB2: the ROUNDABOUT state channel of the route-text nav. Every flag defaults to OFF (bit for bit the earlier trainer) and has its G-DVB entry in FLAG_CHECKS and its clause in bad_navrb.\n'
     '    ap.add_argument("--nav-rb", dest="nav_rb", choices=M.NAV_RB_MODES, default="off",\n'
     '                    help="PREREG_V8F_NAVRB2: off (default) = nothing built or read; on = e_nav_rb (Linear(1, d, bias=False), zero-init) ADDED to the nav token of S, T and O, fed by --nav-rb-npz")\n'
     '    ap.add_argument("--nav-rb-npz", dest="nav_rb_npz", default="", help="PREREG_V8F_NAVRB2: navtext_nav_rb.npz (build_navtext_nav_rb.py). Must exist iff --nav-rb on, must be \'\' iff off")\n'
     '    ap.add_argument("--nav-rb-sha256", dest="nav_rb_sha256", default="", help="PREREG_V8F_NAVRB2: sha256 of --nav-rb-npz (re-hashed at load; a mismatch is a refusal)")\n'
     '    ap.add_argument("--nav-rb-variant", dest="nav_rb_variant", choices=D.NAV_RB_VARIANTS, default="real",\n'
     '                    help="PREREG_V8F_NAVRB2: real (default) = the table; zero = every row 0.0 (the zeroed-channel control arm N1Z)")\n'
     '    ap.add_argument("--nav-rb-lr-mult", dest="nav_rb_lr_mult", type=float, default=1.0,\n'
     '                    help="PREREG_V8F_NAVRB2: e_nav_rb\'s lr as a multiple of --lr (its own optimiser group named navrb). 1.0 (default) with the channel off")\n'),
    ('"ctx_heads", "w_ctx", "ctx_detach", "ctx_lr_mult", "ctx_labels", "ctx_labels_sha256", "nav_source", "nav_text_npz", "nav_text_sha256")\n',
     '"ctx_heads", "w_ctx", "ctx_detach", "ctx_lr_mult", "ctx_labels", "ctx_labels_sha256", "nav_source", "nav_text_npz", "nav_text_sha256",\n'
     '                         "nav_rb", "nav_rb_npz", "nav_rb_sha256", "nav_rb_variant", "nav_rb_lr_mult")\n'),
    ('                           ctx_heads=a.ctx_heads, w_ctx=a.w_ctx, ctx_detach=a.ctx_detach, nav_source=a.nav_source,        # PREREG_V8F_STRAT1: the lines that apply the STRAT1 levers (a mutation deletes them)\n',
     '                           ctx_heads=a.ctx_heads, w_ctx=a.w_ctx, ctx_detach=a.ctx_detach, nav_source=a.nav_source,        # PREREG_V8F_STRAT1: the lines that apply the STRAT1 levers (a mutation deletes them)\n'
     '                           nav_rb=a.nav_rb,                                                                             # PREREG_V8F_NAVRB2: the line that applies --nav-rb (a mutation deletes it)\n'),
    ('def declared_vs_built(a, cfg, model, opt, data_info, enc=None, ego=None, sched=None, dag=None, comp=None, rec=None, t2b=None) -> list:\n',
     NAVRB_FN.lstrip("\n") + '\n\ndef declared_vs_built(a, cfg, model, opt, data_info, enc=None, ego=None, sched=None, dag=None, comp=None, rec=None, t2b=None) -> list:\n'),
    ('    bad += bad_ctx(a, model, data_info, opt)                                                    # PREREG_V8F_STRAT1',
     '    bad += bad_navrb(a, model, data_info, opt)                                                  # PREREG_V8F_NAVRB2: the five --nav-rb* flags read back from the BUILT model / navrb group / loaded channel array\n'
     '    bad += bad_ctx(a, model, data_info, opt)                                                    # PREREG_V8F_STRAT1'),
    ('({a.lr * a.ctx_lr_mult} if a.ctx_heads == "on" else set())\n',
     '({a.lr * a.ctx_lr_mult} if a.ctx_heads == "on" else set()) | ({a.lr * a.nav_rb_lr_mult} if a.nav_rb == "on" else set())\n'),
    ('    stray = sorted(keys - set(DG.REBUILT) - set(DG.KEPT))\n',
     '    stray = sorted(keys - set(DG.REBUILT) - set(DG.KEPT) - M.NAV_RB_KEYS)          # PREREG_V8F_NAVRB2: nav_rb has no distance part, so the swap holds it as logged (kept_moved below asserts the same object)\n'),
    ('        info["nav_source"].update(rec_)\n',
     '        info["nav_source"].update(rec_)\n'
     '    if a.nav_rb == "on":\n'
     '        J["nav_rb"], info["nav_rb"] = D.nav_rb_arrays(J, a.nav_rb_npz, a.nav_rb_sha256, "train", a.nav_rb_variant)      # PREREG_V8F_NAVRB2: ONE line (a mutation deletes it)\n'),
    ('    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2 and id(p) not in ctx_ids]\n    nodecay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2 and id(p) not in ctx_ids]\n',
     '    navrb_ids = {id(p) for p in model.e_nav_rb.parameters()} if model.e_nav_rb is not None else set()      # PREREG_V8F_NAVRB2: e_nav_rb in its own group (empty with the channel off = the tip\'s groups, bit for bit)\n'
     '    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2 and id(p) not in ctx_ids and id(p) not in navrb_ids]\n'
     '    nodecay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2 and id(p) not in ctx_ids and id(p) not in navrb_ids]\n'),
    ('    if a.wm_only:                                                          # PREREG_V8F_MRATE: the readers, heads and planner are frozen and never run; the prob',
     '    nrp = None                                                             # PREREG_V8F_NAVRB2: the navrb probe record (set below for --nav-rb on)\n'
     '    if a.wm_only:                                                          # PREREG_V8F_MRATE: the readers, heads and planner are frozen and never run; the prob'),
    ('    if enc is not None:                                # the LoRA matrices (2-D: the same weight-decay rule as the heads), their own peak lr\n',
     '    if navrb_ids:\n'
     '        groups.append({"params": list(model.e_nav_rb.parameters()), "weight_decay": a.wd, "lr": a.lr * a.nav_rb_lr_mult, "name": "navrb", "lr_mult": a.nav_rb_lr_mult})\n'
     '    if enc is not None:                                # the LoRA matrices (2-D: the same weight-decay rule as the heads), their own peak lr\n'),
    ('            raise SystemExit(f"ZZDVB-REFUSEDZZ nav-source: {e}")\n',
     '            raise SystemExit(f"ZZDVB-REFUSEDZZ nav-source: {e}")\n'
     '        if a.nav_rb == "on":\n'
     '            try:\n'
     '                nrp = navrb_probe(model, probe, cfg)      # PREREG_V8F_NAVRB2: G-LIVE on the loop\'s own batch: the channel is read, the step-0 identity, gradient > 0, the zero-channel control EXACTLY 0.0\n'
     '            except (AssertionError, ValueError, KeyError) as e:\n'
     '                raise SystemExit(f"ZZDVB-REFUSEDZZ nav-rb: {e}")\n'),
    ('    sc["dagger_probe"] = dagp\n',
     '    sc["dagger_probe"] = dagp\n'
     '    if a.nav_rb == "on":\n'
     '        sc["navrb_probe"] = nrp                                       # PREREG_V8F_NAVRB2: lever-ON arms only (an off arm\'s selfcheck keeps its key set)\n'),
    ('        if a.nav_source == "text":                                      # PREREG_V8F_STRAT1: lever-ON arms only\n',
     '        if a.nav_rb == "on":                                            # PREREG_V8F_NAVRB2: lever-ON arms only\n'
     '            blob["nav_rb_sha256"], blob["nav_rb_variant"] = a.nav_rb_sha256, a.nav_rb_variant\n'
     '        if a.nav_source == "text":                                      # PREREG_V8F_STRAT1: lever-ON arms only\n'),
])
