# (old, new) pairs, LF-normalised; each old must occur exactly once.
EDITS = [
# ---- rl_step signature ---------------------------------------------------------------
('''def rl_step(ctx, arm, micro, windows_step, gen, shuf_gen, step, seed, consts, params,
            opt=None, apply=True):
    """One optimiser step over the whole step batch (len(windows_step) windows)."""
''',
'''def rl_step(ctx, arm, micro, windows_step, gen, shuf_gen, step, seed, consts, params,
            opt=None, apply=True, kl_beta=0.0, ref=None):
    """One optimiser step over the whole step batch (len(windows_step) windows).

    ``kl_beta > 0`` (SPEC_RL_A1 sec. 3): adds ``kl_beta * mean_{B,M} KL_i / T`` per denoising step,
    the KL of the step's surrogate density to the REFERENCE policy ``ref`` (name -> tensor, the
    frozen start checkpoint = the trainer's ``base``). The reference x0 of all ``T`` steps is
    computed ONCE per micro-batch under ``no_grad``, BEFORE any theta forward of that micro-batch.
    """
    if kl_beta > 0 and ref is None:
        raise SystemExit("[rl7] --kl-beta > 0 needs the reference parameters")
'''),
# ---- accumulator ----------------------------------------------------------------------
('''    loss_total, il_total, rl_part = 0.0, 0.0, 0.0
    row0 = 0
''',
'''    loss_total, il_total, rl_part = 0.0, 0.0, 0.0
    kl_m = 0.0
    row0 = 0
'''),
# ---- reference x0 before any theta forward -------------------------------------------
('''    for r in recs:
        m = r["chain"].shape[0]
        stages: list = []
        fn = R.make_x0_fn(ctx.dec, r["inp"], stages=stages)
''',
'''    for r in recs:
        m = r["chain"].shape[0]
        x0_refs = None
        if kl_beta > 0:
            # BEFORE fn / any theta forward: the swap writes the parameters in place.
            with R.swapped_params(ctx.named, ref), torch.no_grad():
                fn_ref = R.make_x0_fn(ctx.dec, r["inp"])
                x0_refs = [fn_ref(r["chain"][..., i], int(r["labels"][i])).detach()
                           for i in range(len(r["labels"]))]
        stages: list = []
        fn = R.make_x0_fn(ctx.dec, r["inp"], stages=stages)
'''),
# ---- KL term into the per-step loss --------------------------------------------------
('''            li, rl_i = R.microbatch_step_loss(lp, cr[..., i], w["il_coef"], il_i, frac)
            li.backward()
''',
'''            li, rl_i = R.microbatch_step_loss(lp, cr[..., i], w["il_coef"], il_i, frac)
            if x0_refs is not None:
                kl = R.step_kl_to_reference(_x0, x0_refs[i], r["chain"][..., i], int(r["labels"][i]),
                                            ctx.table, eta=consts.eta, consts=consts)
                li = li + kl_beta * kl.mean() * frac / T
                kl_m += float(kl.mean()) * frac / T
            li.backward()
'''),
# ---- telemetry ------------------------------------------------------------------------
('''    return {"loss": loss_total, "rl_part": rl_part, "il_m": il_total, "il_all_modes_m": il_all,
''',
'''    return {"loss": loss_total, "rl_part": rl_part, "il_m": il_total, "il_all_modes_m": il_all,
            "kl_m": kl_m, "kl_beta": float(kl_beta),
'''),
# ---- lr_at ----------------------------------------------------------------------------
('''def lr_at(step, total, lr, min_lr=1e-6, warmup_frac=0.10):
    """P suppl. sec. 7 'cosine learning rate schedule with a 10 % linear warmup', over STEPS (SPEC A15:
    the release steps per epoch, which makes its warmup a no-op -- flagged)."""
    wu = max(1, int(round(warmup_frac * total)))
''',
'''def lr_at(step, total, lr, min_lr=1e-6, warmup_frac=0.10, warmup_steps=None):
    """P suppl. sec. 7 'cosine learning rate schedule with a 10 % linear warmup', over STEPS (SPEC A15:
    the release steps per epoch, which makes its warmup a no-op -- flagged). ``warmup_steps`` (SPEC_RL_A1
    sec. 2) replaces the 10 % fraction by an absolute step count."""
    wu = max(1, int(round(warmup_frac * total))) if not warmup_steps else max(1, int(warmup_steps))
'''),
# ---- HALT latch at start --------------------------------------------------------------
('''    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.batch % a.micro:
''',
'''    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "HALT.json").exists():
        raise SystemExit(f"[rl7] {out / 'HALT.json'} is LATCHED -- the kill-switch tripped in this out-dir "
                         "(SPEC_RL_A1 sec. 5); the arm is FAILED-DIVERGED and is not resumed")
    kl_beta = float(getattr(a, "kl_beta", 0.0) or 0.0)
    warmup_steps = int(getattr(a, "warmup_steps", 0) or 0)
    halt = {"gn_median": float(getattr(a, "halt_gn_median", 0.0) or 0.0),
            "oob_median": float(getattr(a, "halt_oob_median", 0.0) or 0.0),
            "pdn": float(getattr(a, "halt_pdn", 0.0) or 0.0),
            "window": int(getattr(a, "halt_window", 20) or 20)}
    halt_on = any(halt[k] > 0 for k in ("gn_median", "oob_median", "pdn"))
    if a.batch % a.micro:
'''),
# ---- run record (keys only when non-default: pre-A1 run_hashes are unchanged) ----------
('''    run["trainable"] = {"prefixes": list(R.TRAINABLE_PREFIXES), "n_params": n_train,
                        "n_tensors": len(named)}
''',
'''    run["trainable"] = {"prefixes": list(R.TRAINABLE_PREFIXES), "n_params": n_train,
                        "n_tensors": len(named)}
    if kl_beta > 0:
        run["kl_beta"] = kl_beta
        run["kl_reference"] = "base = the generator parameters at step 0 of this run (SPEC_RL_A1 sec. 3)"
    if warmup_steps > 0:
        run["warmup_steps"] = warmup_steps
        run["schedule"] = f"linear warmup {warmup_steps} steps + cosine to 1e-6 over steps (SPEC_RL_A1)"
    if halt_on:
        run["halt"] = halt
'''),
# ---- kill-switch history seeded from this out-dir's own metrics on resume ---------------
('''    mf = open(out / "metrics.jsonl", "a", encoding="utf-8")

    def save(step_done, tag="latest"):
''',
'''    hist = []
    if halt_on and start > 0 and (out / "metrics.jsonl").exists():
        by_step = {}
        for ln in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                rec = json.loads(ln)
                if int(rec.get("step", -1)) < start:
                    by_step[int(rec["step"])] = {k: rec[k] for k in R.HALT_KEYS}
        hist = [by_step[k] for k in sorted(by_step)][-halt["window"]:]
    mf = open(out / "metrics.jsonl", "a", encoding="utf-8")

    def save(step_done, tag="latest"):
'''),
# ---- lr + rl_step call ------------------------------------------------------------------
('''        lr = lr_at(step, a.steps, a.lr)
''',
'''        lr = lr_at(step, a.steps, a.lr, warmup_steps=warmup_steps or None)
'''),
('''        stats = rl_step(ctx, arm, micro, None, gen, shuf_gen, step, a.seed, consts, params, opt=opt,
                        apply=True)
''',
'''        stats = rl_step(ctx, arm, micro, None, gen, shuf_gen, step, a.seed, consts, params, opt=opt,
                        apply=True, kl_beta=kl_beta, ref=base if kl_beta > 0 else None)
'''),
# ---- kill-switch after the finite check ---------------------------------------------------
('''        if not stats["finite"]:
            save(step, "nonfinite")
            raise SystemExit(f"[train] non-finite at step {step}")
''',
'''        if not stats["finite"]:
            save(step, "nonfinite")
            raise SystemExit(f"[train] non-finite at step {step}")
        if halt_on:
            hist.append({k: stats[k] for k in R.HALT_KEYS})
            del hist[:-halt["window"]]
            reason = R.halt_check(hist, gn_median=halt["gn_median"], oob_median=halt["oob_median"],
                                  pdn=halt["pdn"], window=halt["window"])
            if reason:
                (out / "HALT.json").write_text(json.dumps(
                    {"reason": reason, "step": step, "thresholds": halt, "run_hash": run_hash,
                     "verdict": "FAILED-DIVERGED (SPEC_RL_A1 sec. 5)",
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), encoding="utf-8")
                (out / "STOP").write_text(f"halt: {reason}\\n", encoding="utf-8")
                log(f"[train] KILL-SWITCH at step {step}: {reason}")
                stop_reason = f"halt:{reason}"
                break
'''),
# ---- argparse (train) ---------------------------------------------------------------------
('''    s.add_argument("--grad-clip", type=float, default=100.0)
    s.add_argument("--clip-sample", action="store_true", help="the release's x0 clamp (OFF: SPEC D-4)")
''',
'''    s.add_argument("--grad-clip", type=float, default=100.0)
    s.add_argument("--clip-sample", action="store_true", help="the release's x0 clamp (OFF: SPEC D-4)")
    s.add_argument("--warmup-steps", type=int, default=0,
                   help="absolute linear-warmup steps (0 = the 10 %% fraction; SPEC_RL_A1 sec. 2)")
    s.add_argument("--kl-beta", type=float, default=0.0,
                   help="KL anchor to the start checkpoint (0 = off; SPEC_RL_A1 sec. 3)")
    s.add_argument("--halt-gn-median", type=float, default=0.0, help="K1 (0 = off; SPEC_RL_A1 sec. 5)")
    s.add_argument("--halt-oob-median", type=float, default=0.0, help="K2 (0 = off)")
    s.add_argument("--halt-pdn", type=float, default=0.0, help="K3 (0 = off)")
    s.add_argument("--halt-window", type=int, default=20, help="K1/K2 median window (steps)")
'''),
]
