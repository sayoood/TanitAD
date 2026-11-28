"""F3 (found by the v7f launch gate's G-CKPT rehearsal, 2026-09-27): a v6 resume REPLAYED the data/RNG stream.

MEASURED by the gate (…/2026-09-27-v7f-refav1-state/v7f_gate/RESULT.md, G-CKPT row + F3): `train()` re-seeds
`torch.manual_seed(a.seed)`, `random.seed(a.seed)`, `rng = random.Random(a.seed)` and
`gen = torch.Generator().manual_seed(a.seed + 1)` at EVERY launch, and `_save_ckpt` wrote only
`{stack, opt, step, config}` -- so a resumed S-T run drew the uninterrupted run's FIRST batches instead of its
steps 101-102.

This script patches, anchor by anchor (every anchor must match EXACTLY ONCE or it refuses; EOL preserved per file):
  * stack/scripts/train_v6_staged.py -- the exact-resume payload (`rng_state` / `data_pos` / `loop_state`) is
    captured at the checkpoint boundary and restored immediately before the next step; a pre-F3 checkpoint
    still resumes, LOUDLY, recorded as config.json `resume_rng.mode = "absent-replayed"`;
  * stack/scripts/launch_gate.py -- G-CKPT (v6) runs its resume on an INTERRUPTED run launched with the FINAL
    --steps (see the gate edit's comment: run 1 at --steps n + a resume to n+extra re-derives the cosine LR
    schedule, so it could never equal an uninterrupted n+extra run -- a PROTOCOL confound, not an RNG one), and
    the arm `v6_resume_drops_rng` must FAIL;
  * stack/tests/test_launch_gate_v7f.py -- the G-CKPT pins flipped to positive assertions + the new arm;
  * stack/tests/test_v6_exact_resume.py -- NEW (copied from fix/stack/tests/ next to this script).

Usage:  python apply_f3_exact_resume.py [--stack <dir holding scripts/ and tests/>] [--only trainer,gate,tests]
        (default --stack C:/Users/Admin/v7f_merge/stack). Nothing is written unless EVERY anchor of a file matches.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CR, LF = chr(13), chr(10)
CRLF = CR + LF


def _read(p: Path) -> tuple[str, str]:
    b = p.read_bytes()
    n_crlf, n_lf = b.count(CRLF.encode()), b.count(LF.encode())
    if n_crlf and n_crlf != n_lf:
        sys.exit(f"{p}: MIXED line endings ({n_crlf} CRLF of {n_lf} LF) -- refusing to guess the EOL")
    eol = CRLF if n_crlf else LF
    return b.decode("utf-8").replace(CRLF, LF), eol


def _patch(p: Path, edits: list[tuple[str, str]]) -> None:
    s, eol = _read(p)
    for i, (old, new) in enumerate(edits):
        c = s.count(old)
        if c != 1:
            sys.exit(f"{p.name}: anchor #{i} matched {c} times (want 1) -- refusing, NOTHING written:\n"
                     f"{old[:300]}")
        s = s.replace(old, new)
    p.write_bytes(s.replace(LF, eol).encode("utf-8"))
    print(f"patched {p} ({len(edits)} anchors, EOL {'CRLF' if eol == CRLF else 'LF'})")


# ================================================================================================================
# 1. the trainer
# ================================================================================================================
TRAINER: list[tuple[str, str]] = []

# --- the export list -------------------------------------------------------------------------------------------
TRAINER.append((
    '''    "supersede_init_on_resume",
''',
    '''    "supersede_init_on_resume",
    "RESUME_STATE_SCHEMA", "RESUME_STATE_KEYS", "resume_stream_fingerprint",
    "capture_resume_state", "read_resume_state", "resume_state_record",
    "apply_resume_state", "replay_lr_schedule",
'''))

# --- read the payload right after the weights + optimiser load ------------------------------------------------
TRAINER.append((
    '''    start_step = 0
    if rg["mode"] == "resume":
        start_step = load_resume(stack, opt, rg["from"], stage=a.stage)
        for _ in range(start_step):
            sched.step()                       # replay the LR schedule
''',
    '''    start_step = 0
    # ⭐ F3: the exact-resume payload of the checkpoint being resumed (None on a
    # fresh launch, and on a checkpoint written before F3). READ here, next to
    # the weights; APPLIED immediately before the step loop, after every
    # construction-time consumer of the seeded streams has run.
    resume_state: dict | None = None
    if rg["mode"] == "resume":
        start_step = load_resume(stack, opt, rg["from"], stage=a.stage)
        # ⛔ F3b: NOT `for _ in range(start_step): sched.step()` on top of the
        # LOADED lr -- that applied the first start_step steps of cosine decay
        # twice (see replay_lr_schedule).
        replay_lr_schedule(opt, sched, start_step)
        resume_state = read_resume_state(rg["from"])
        if resume_state is None:
            # ⛔ RESUME, LOUDLY -- do not refuse. Refusing would strand every
            # checkpoint written before F3 (the live and banked v6 runs among
            # them); a replayed stream re-draws in-distribution batches, it does
            # not corrupt the model. The deviation is RECORDED in config.json
            # (`resume_rng.mode = "absent-replayed"`) and in every checkpoint
            # this run writes, so no number from the run can lose it; the NEXT
            # checkpoint carries the payload, so the replay happens at most once.
            print("=" * 72, flush=True)
            print(f"[v6] ⚠️  THIS RESUME REPLAYS THE DATA/RNG STREAM: "
                  f"{rg['from']} predates the exact-resume payload (F3, "
                  f"2026-09-27) -- it carries no rng_state/data_pos. torch, "
                  f"random, the sampler and the shared generator restart from "
                  f"--seed {a.seed}, so steps {start_step + 1}.. re-draw the "
                  f"batches steps 1.. drew. Recorded: config.json resume_rng."
                  f"mode = 'absent-replayed'.", flush=True)
            print("=" * 72, flush=True)
'''))

# --- the stream fingerprint + the resume record, INTO the run row ---------------------------------------------
TRAINER.append((
    '''                                                "launch_mode": rg}
    # ⭐ F-9's provenance stamp travels INTO THE RUN ROW, not just the log.
''',
    '''                                                "launch_mode": rg}
    # ⭐ F3: the data stream's fingerprint rides every checkpoint; a resume
    # compares it with its own and RECORDS any difference (a resume whose
    # stream geometry changed cannot be exact, and must say so). Resume only:
    # a fresh launch's config.json is unchanged.
    stream_fp = resume_stream_fingerprint(a, ds_train, sample, t5_partner,
                                          t3_curr, dmix, device)
    if rg["mode"] == "resume":
        cfg_json["resume_rng"] = resume_state_record(resume_state,
                                                     stream_now=stream_fp)
    # ⭐ F-9's provenance stamp travels INTO THE RUN ROW, not just the log.
'''))

# --- the T3 curriculum: remember the `prog` the live weights were computed at ---------------------------------
TRAINER.append((
    '''                sample.weights = t3_curr.weights_at(t3_scores, prog)
                t3_alpha_applied = al
''',
    '''                sample.weights = t3_curr.weights_at(t3_scores, prog)
                t3_alpha_applied = al
                t3_prog_applied = prog         # F3: what a resume rebuilds
'''))

# --- the spectrum reference IS carried now --------------------------------------------------------------------
TRAINER.append((
    '''                    # ⚠️ NOT carried across a resume: a restarted process takes
                    # a fresh reference, so its retention is measured from the
                    # restart, not from the phase start. The verdict says which
                    # step the reference came from via the record it embeds —
                    # read it, do not assume the phase start.
''',
    '''                    # ⭐ F3: CARRIED across a resume (`loop_state.monitors`).
                    # ⚠️ Only a checkpoint written before F3 restarts it: that
                    # process takes a fresh reference, so its retention is
                    # measured from the restart. The verdict says which step the
                    # reference came from via the record it embeds — read it,
                    # do not assume the phase start.
'''))

# --- restore, LAST, immediately before the first step ---------------------------------------------------------
TRAINER.append((
    '''    t3_alpha_applied = None
    grad_reach: dict | None = None
    for step in range(start_step + 1, a.steps + 1):
''',
    '''    t3_alpha_applied = None
    t3_prog_applied = None                # F3: the `prog` behind sample.weights
    grad_reach: dict | None = None
    # ⭐ F3 -- THE RESTORE, and it must be HERE. Everything above consumed the
    # SEEDED streams at construction (the stack's init, --init-from, the O4/T3
    # samplers, the monitors' generators) exactly as the uninterrupted run did;
    # nothing between this line and the first draw consumes one. So step
    # start_step + 1 starts from precisely the state saved at the end of step
    # start_step -- the RNG streams, the T3 weights, the SIGReg row bank and
    # the spectrum/o6/observer monitors.
    if resume_state is not None:
        _rl = apply_resume_state(resume_state, device=device, rng=rng, gen=gen,
                                 spec_gen=spec_gen, x4_mon=x4_mon,
                                 sigreg_bank=sigreg_bank, spec_acc=spec_acc,
                                 obs_mon=obs_mon)
        t3_alpha_applied = _rl["t3_alpha_applied"]
        t3_prog_applied = _rl["t3_prog_applied"]
        if t3_curr is not None and t3_prog_applied is not None:
            sample.weights = t3_curr.weights_at(t3_scores, t3_prog_applied)
        _mon = _rl["monitors"]
        spectrum_last = _mon.get("spectrum_last", spectrum_last)
        spectrum_pooled_last = _mon.get("spectrum_pooled_last",
                                        spectrum_pooled_last)
        spectrum_ref = _mon.get("spectrum_ref", spectrum_ref)
        x4_last = _mon.get("x4_last", x4_last)
        o6_trend_base = list(_mon.get("o6_trend_base", o6_trend_base))
        o6_trend_cur.extend(_mon.get("o6_trend_cur", ()))
        print(f"[v6] F3 exact resume: restored {json.dumps(_rl['applied'])} "
              f"at the end of step {start_step}", flush=True)
    for step in range(start_step + 1, a.steps + 1):
'''))

# --- capture at the checkpoint boundary, FIRST in the save block ---------------------------------------------
TRAINER.append((
    '''        if step % a.save_every == 0 or step == a.steps:
''',
    '''        if step % a.save_every == 0 or step == a.steps:
            # ⭐ F3: captured FIRST in the block -- the state at the END of step
            # `step`'s training work, i.e. exactly what step + 1 of an
            # uninterrupted run starts from (an uninterrupted run need not save
            # at this step, so nothing the save block does may be inside it).
            resume_payload = capture_resume_state(
                step=step, device=device, rng=rng, gen=gen, spec_gen=spec_gen,
                x4_mon=x4_mon, stream=stream_fp,
                t3_alpha_applied=t3_alpha_applied,
                t3_prog_applied=t3_prog_applied, sigreg_bank=sigreg_bank,
                spec_acc=spec_acc, obs_mon=obs_mon,
                monitors={"spectrum_last": spectrum_last,
                          "spectrum_pooled_last": spectrum_pooled_last,
                          "spectrum_ref": spectrum_ref, "x4_last": x4_last,
                          "o6_trend_base": list(o6_trend_base),
                          "o6_trend_cur": list(o6_trend_cur)})
'''))

TRAINER.append((
    '''            _save_ckpt(out_dir / "ckpt.pt", stack=stack, opt=opt, step=step,
                       cfg_json=cfg_json,
                       keep_step=not bool(getattr(a, "no_step_ckpts", False)))
''',
    '''            _save_ckpt(out_dir / "ckpt.pt", stack=stack, opt=opt, step=step,
                       cfg_json=cfg_json,
                       keep_step=not bool(getattr(a, "no_step_ckpts", False)),
                       extra=resume_payload)
'''))

# --- _save_ckpt: the extra keys, never over the incumbent four ------------------------------------------------
TRAINER.append((
    '''def _save_ckpt(path: Path, *, stack, opt, step: int, cfg_json: dict,
               keep_step: bool = True) -> None:
''',
    '''def _save_ckpt(path: Path, *, stack, opt, step: int, cfg_json: dict,
               keep_step: bool = True, extra: dict | None = None) -> None:
'''))
TRAINER.append((
    '''    payload = {"stack": stack.state_dict(), "opt": opt.state_dict(),
               "step": step, "config": cfg_json}
''',
    '''    payload = {"stack": stack.state_dict(), "opt": opt.state_dict(),
               "step": step, "config": cfg_json}
    if extra:
        # ⭐ F3: the exact-resume keys (`rng_state` / `data_pos` / `loop_state`,
        # :func:`capture_resume_state`). ADDITIVE: every reader of the four
        # incumbent keys is untouched, and a clash is refused, never merged.
        clash = sorted(set(extra) & set(payload))
        if clash:
            raise ValueError(f"_save_ckpt: extra keys {clash} would overwrite "
                             f"the checkpoint's own")
        payload.update(extra)
'''))

# --- the helpers, right before resume_guard -------------------------------------------------------------------
TRAINER.append((
    '''

def resume_guard(out_dir, *, resume: str, force_rerun: bool) -> dict:
''',
    '''

# ============================================================================
# ⭐ F3 (v7f launch gate G-CKPT, 2026-09-27) -- an EXACT resume
# ============================================================================
#: ⛔ WHAT A RESUME USED TO DO. MEASURED by the v7f launch gate's G-CKPT
#: rehearsal (`…/2026-09-27-v7f-refav1-state/v7f_gate/RESULT.md`, F3):
#: ``train()`` re-seeds torch, ``random``, the sampler's ``rng`` and the shared
#: generator ``gen`` from ``--seed`` at EVERY launch, and ``_save_ckpt`` wrote
#: only ``{stack, opt, step, config}`` -- so a resumed S-T run drew
#: ``[[1,5],[1,23]],[[2,10],[2,7]]``, the uninterrupted run's FIRST batches,
#: instead of its steps 101-102. A resume trained the first N batches twice and
#: never saw the ones an uninterrupted run would, and no log line showed it.
#:
#: ⇒ Three keys ride every checkpoint, captured at the END of the saved step:
#:   * ``rng_state`` -- EVERY stream the step loop consumes: torch CPU (global
#:     draws: SIGReg slices, dropout), torch CUDA (per device, when training on
#:     CUDA), python ``random``, the uniform sampler's ``rng``
#:     (``make_sampler``), the shared ``gen`` (O4/T3/F-10 samplers,
#:     ``sample_random_deltas``, ``v6_loss_step``, the O5 target crop, O9),
#:     ``spec_gen`` (seed+7) and X4's generator (seed+11). numpy's global
#:     stream is carried too: nothing on the path draws from it today
#:     (grep), and carrying it means a future consumer cannot silently break
#:     exactness;
#:   * ``data_pos`` -- the step, the stream FINGERPRINT (a resume compares its
#:     own and records any difference) and the T3 curriculum's applied
#:     alpha/prog (the live weights are a function of the UNROUNDED prog of
#:     the step they were recomputed at, so a resume rebuilds them from it);
#:   * ``loop_state`` -- the rolling buffers that carry state across steps: the
#:     SIGReg row bank (it enters the O6 LOSS), the O6/X4 spectrum pools and
#:     references, the o6 trend series, the observer monitor.
#: NOT carried, by construction: the LR schedule (replayed by stepping, a pure
#: function of the step), the EMA ramp (reads the ABSOLUTE step), the O13
#: projection (re-seeded per call from --o13-seed), the X2 seam dump (no RNG).
#: ⚠️ And NOT in scope here, found while auditing: the O7/O8/O9/O10 auxiliary
#: heads are trained (their params are in the optimiser) but are not in
#: ``stack.state_dict()`` -- a resume of such an arm re-initialises them.
RESUME_STATE_SCHEMA = "v6-exact-resume/1"
RESUME_STATE_KEYS = ("rng_state", "data_pos", "loop_state")


def replay_lr_schedule(opt, sched, n_steps: int) -> None:
    """Bring a FRESH scheduler to step ``n_steps`` after ``load_resume``.

    ⛔ F3b -- MEASURED 2026-09-27 on the exact-resume rig, and it hit EVERY v6
    resume on the default schedule. ``opt.load_state_dict`` restores each
    group's ``lr`` AS SAVED (already decayed to step ``n``), and
    ``CosineAnnealingLR`` is RECURSIVE: ``get_lr`` scales the group's CURRENT
    ``lr`` by ``(1+cos(pi*e/T)) / (1+cos(pi*(e-1)/T))``. Replaying ``n`` steps
    on top of the loaded ``lr`` therefore applied the first ``n`` steps of decay
    TWICE, and because every later step is again a ratio of the current value,
    the WHOLE REMAINDER of the run ran at ``(1 + cos(pi*n/T)) / 2`` of its
    schedule -- a resume at 50 % of --steps halved every later LR, a resume at
    90 % left 2.4 %. MEASURED on a 5-step tiny S-W run resumed at step 3: the lr
    logged after step 4 was 3.2991502812526298e-06 resumed vs
    9.549150281252631e-06 uninterrupted, ratio 0.34549 = (1+cos(0.6*pi))/2.
    ⚠️ The launch gate's G-CKPT hid it: its optimiser param-group comparison
    excluded ``lr``.
    ⇒ every group's ``lr`` is reset to its ``initial_lr`` and the schedule is
    replayed from there: the identical sequence of float operations the
    uninterrupted run performed, so the lr of step ``n + 1`` is BIT-equal to
    it. ``LambdaLR`` (the --trunk-lr-factor path) is closed-form and was never
    affected; the reset is a no-op for it.
    """
    for g in opt.param_groups:
        if "initial_lr" in g:
            g["lr"] = g["initial_lr"]
    for _ in range(int(n_steps)):
        sched.step()                       # replay the LR schedule


def resume_stream_fingerprint(a, ds_train, sample, t5_partner, t3_curr, dmix,
                              device) -> dict:
    """What the saved streams are streams OF. Two runs with the same
    fingerprint draw from the same index space with the same consumers; a
    resume whose fingerprint differs cannot be exact and records why."""
    return {"n_windows": len(ds_train.index), "batch": int(a.batch),
            "eps_per_batch": int(a.eps_per_batch), "seed": int(a.seed),
            "sampler": getattr(sample, "__qualname__", type(sample).__name__),
            "t5_pairs": bool(t5_partner),
            "t5_lag": int(getattr(a, "t5_lag", 0) or 0),
            "t3": t3_curr is not None, "domain_mix": dmix is not None,
            "sigreg_accum": int(getattr(a, "sigreg_accum", 1) or 1),
            "spectrum_accum": int(getattr(a, "spectrum_accum", 1) or 1),
            "spectrum_ci_reps": int(getattr(a, "spectrum_ci_reps", 0) or 0),
            "x4_spectrum_layers": str(getattr(a, "x4_spectrum_layers", "")),
            "obs_monitor_every": int(getattr(a, "obs_monitor_every", 0) or 0),
            "device_type": "cuda" if str(device).startswith("cuda") else "cpu"}


def _np_rng_get():
    try:
        import numpy as _np  # noqa: PLC0415
    except Exception:                                    # noqa: BLE001
        return None
    name, keys, pos, has_gauss, cached = _np.random.get_state(legacy=True)
    return {"name": str(name), "keys": [int(k) for k in keys.tolist()],
            "pos": int(pos), "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached)}


def _np_rng_set(st: dict) -> None:
    import numpy as _np  # noqa: PLC0415
    _np.random.set_state((st["name"], _np.asarray(st["keys"], dtype=_np.uint32),
                          int(st["pos"]), int(st["has_gauss"]),
                          float(st["cached_gaussian"])))


def _rows_cpu(buf) -> list:
    return [t.detach().cpu().clone() for t in buf]


def capture_resume_state(*, step: int, device, rng, gen, spec_gen=None,
                         x4_mon=None, stream: dict | None = None,
                         t3_alpha_applied=None, t3_prog_applied=None,
                         sigreg_bank=None, spec_acc=None, obs_mon=None,
                         monitors: dict | None = None) -> dict:
    """-> ``{"rng_state", "data_pos", "loop_state"}`` at the END of step
    ``step``: exactly what step ``step + 1`` of an uninterrupted run starts
    from. Pure reads -- captures consume no stream."""
    import copy  # noqa: PLC0415
    cuda = None
    if str(device).startswith("cuda") and torch.cuda.is_available():
        cuda = [s.clone() for s in torch.cuda.get_rng_state_all()]
    x4g = getattr(x4_mon, "generator", None) if x4_mon is not None else None
    rng_state = {
        "schema": RESUME_STATE_SCHEMA,
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": cuda,
        "python_random": random.getstate(),
        "sampler_rng": rng.getstate() if rng is not None else None,
        "gen": gen.get_state().clone() if gen is not None else None,
        "spec_gen": spec_gen.get_state().clone() if spec_gen is not None else None,
        "x4_gen": x4g.get_state().clone() if x4g is not None else None,
        "numpy": _np_rng_get(),
    }
    data_pos = {"schema": RESUME_STATE_SCHEMA, "step": int(step),
                "next_step": int(step) + 1, "stream": dict(stream or {}),
                "t3_alpha_applied": t3_alpha_applied,
                "t3_prog_applied": t3_prog_applied}
    obs = None
    if obs_mon is not None:
        obs = {"f": _rows_cpu(obs_mon._f), "p": _rows_cpu(obs_mon._p),
               "t": _rows_cpu(obs_mon._t), "rho0": obs_mon.rho0,
               "rho0_step": obs_mon.rho0_step,
               "consecutive_fails": int(obs_mon.consecutive_fails)}
    loop_state = {
        "schema": RESUME_STATE_SCHEMA,
        "sigreg_bank": (_rows_cpu(sigreg_bank._buf)
                        if sigreg_bank is not None else None),
        "spec_acc": _rows_cpu(spec_acc._buf) if spec_acc is not None else None,
        "x4_acc": ({k: _rows_cpu(acc._buf) for k, acc in x4_mon._acc.items()}
                   if x4_mon is not None else None),
        "x4_ref": copy.deepcopy(x4_mon._ref) if x4_mon is not None else None,
        "obs_mon": obs,
        "monitors": copy.deepcopy(monitors or {}),
    }
    return {"rng_state": rng_state, "data_pos": data_pos,
            "loop_state": loop_state}


def read_resume_state(ckpt_path) -> dict | None:
    """The exact-resume payload of ``ckpt_path``, or ``None`` when it carries
    none (a checkpoint written before F3). ``mmap``: the model tensors are
    never materialised -- this is a second read of a file ``load_resume`` has
    just loaded, and it must not double the resume's peak memory. The payload
    is DEEP-COPIED out and the mapping dropped before returning, so the next
    atomic ``os.replace`` onto this path is not blocked by a live mapping
    (Windows)."""
    import copy  # noqa: PLC0415
    import gc  # noqa: PLC0415
    p = Path(ckpt_path)
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, ValueError, TypeError):
        ck = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or not isinstance(ck.get("rng_state"), dict):
        del ck
        gc.collect()
        return None
    out = {k: copy.deepcopy(ck.get(k)) for k in RESUME_STATE_KEYS}
    del ck
    gc.collect()
    return out


def resume_state_record(state: dict | None, *, stream_now: dict) -> dict:
    """What config.json says about the resume's streams (``resume_rng``)."""
    if state is None:
        return {"mode": "absent-replayed",
                "_read": "the checkpoint predates the exact-resume payload "
                         "(F3, 2026-09-27): every stream restarted from --seed, "
                         "so the steps after the resume RE-DREW the batches of "
                         "the run's first steps. Numbers from this segment are "
                         "not those of an uninterrupted run."}
    rs = state.get("rng_state") or {}
    dp = state.get("data_pos") or {}
    saved = dp.get("stream") or {}
    mism = {k: [saved.get(k), stream_now.get(k)]
            for k in sorted(set(saved) | set(stream_now))
            if saved.get(k) != stream_now.get(k)}
    return {"mode": "restored" if not mism else "restored-stream-mismatch",
            "schema": rs.get("schema"), "saved_step": dp.get("step"),
            "streams_saved": sorted(k for k, v in rs.items()
                                    if k != "schema" and v is not None),
            "stream_mismatch": mism,
            "_read": ("every stream, the T3 weights and the rolling buffers "
                      "continue from the end of the saved step: the run is the "
                      "uninterrupted run" if not mism else
                      "RESTORED, but the stream fingerprint differs from the "
                      "saved one (listed): the continuation is deterministic "
                      "and NOT the uninterrupted run's")}


def apply_resume_state(state: dict, *, device, rng, gen, spec_gen=None,
                       x4_mon=None, sigreg_bank=None, spec_acc=None,
                       obs_mon=None) -> dict:
    """Restore every stream and buffer :func:`capture_resume_state` saved into
    the LIVE objects; returns the loop-locals ``train()`` re-binds
    (``t3_alpha_applied`` / ``t3_prog_applied`` / ``monitors``) plus
    ``applied`` (what was restored). A stream saved but not live now (or live
    but not saved) is left as seeded -- the fingerprint mismatch in
    config.json already names why."""
    import copy  # noqa: PLC0415
    rs = state.get("rng_state") or {}
    dp = state.get("data_pos") or {}
    ls = state.get("loop_state") or {}
    applied: list[str] = []
    torch.set_rng_state(rs["torch_cpu"])
    applied.append("torch_cpu")
    random.setstate(tuple(rs["python_random"]))
    applied.append("python_random")
    if rng is not None and rs.get("sampler_rng") is not None:
        rng.setstate(tuple(rs["sampler_rng"]))
        applied.append("sampler_rng")
    if gen is not None and rs.get("gen") is not None:
        gen.set_state(rs["gen"])
        applied.append("gen")
    if spec_gen is not None and rs.get("spec_gen") is not None:
        spec_gen.set_state(rs["spec_gen"])
        applied.append("spec_gen")
    x4g = getattr(x4_mon, "generator", None) if x4_mon is not None else None
    if x4g is not None and rs.get("x4_gen") is not None:
        x4g.set_state(rs["x4_gen"])
        applied.append("x4_gen")
    cuda = rs.get("torch_cuda")
    if (cuda is not None and str(device).startswith("cuda")
            and torch.cuda.is_available()
            and len(cuda) == torch.cuda.device_count()):
        torch.cuda.set_rng_state_all(cuda)
        applied.append("torch_cuda")
    if rs.get("numpy") is not None:
        try:
            _np_rng_set(rs["numpy"])
            applied.append("numpy")
        except Exception as e:                           # noqa: BLE001
            # nothing on the path draws from numpy (grep); say so, never hide it
            applied.append(f"numpy-NOT-restored:{type(e).__name__}")
    if sigreg_bank is not None and ls.get("sigreg_bank") is not None:
        keep = sigreg_bank.capacity - 1
        rows = ls["sigreg_bank"][-keep:] if keep > 0 else []
        sigreg_bank._buf = [t.to(device) for t in rows]
        applied.append("sigreg_bank")
    if spec_acc is not None and ls.get("spec_acc") is not None:
        spec_acc._buf = list(ls["spec_acc"])[-spec_acc.capacity:]
        applied.append("spec_acc")
    if x4_mon is not None and ls.get("x4_acc") is not None:
        for k, acc in x4_mon._acc.items():
            if k in ls["x4_acc"]:
                acc._buf = list(ls["x4_acc"][k])[-acc.capacity:]
        x4_mon._ref = copy.deepcopy(ls.get("x4_ref") or {})
        applied.append("x4_monitor")
    ob = ls.get("obs_mon")
    if obs_mon is not None and ob is not None:
        for key, dq in (("f", obs_mon._f), ("p", obs_mon._p), ("t", obs_mon._t)):
            dq.clear()
            dq.extend(ob[key])
        obs_mon.rho0, obs_mon.rho0_step = ob["rho0"], ob["rho0_step"]
        obs_mon.consecutive_fails = int(ob["consecutive_fails"])
        applied.append("obs_monitor")
    return {"t3_alpha_applied": dp.get("t3_alpha_applied"),
            "t3_prog_applied": dp.get("t3_prog_applied"),
            "monitors": copy.deepcopy(ls.get("monitors") or {}),
            "applied": applied}


def resume_guard(out_dir, *, resume: str, force_rerun: bool) -> dict:
'''))


# ================================================================================================================
# 2. the gate -- G-CKPT (v6): the INTERRUPTED-run protocol, lr compared, the streams required, three arms
# ================================================================================================================
GATE: list[tuple[str, str]] = []

GATE.append((
    '''  decisive) and reports `.grad` population; G-CKPT compares a resume with an UNINTERRUPTED run (the
  v6 checkpoint carries no data position). The dev-box stage is a REHEARSAL (tiny geometry, synthetic
''',
    '''  decisive) and reports `.grad` population; G-CKPT resumes an INTERRUPTED run (launched with the FINAL
  --steps, killed right after its step-n checkpoint) and compares it with the same argv UNINTERRUPTED --
  draws, lr, final model AND optimiser state (F3, 2026-09-27: the v6 checkpoint carries `rng_state` /
  `data_pos` / `loop_state`). The dev-box stage is a REHEARSAL (tiny geometry, synthetic
'''))
GATE.append((
    '''  `v6_drop_term`, `v6_resume_drops_opt`, `v6_loader_drops_vocab_version`). The rehearsal keeps
''',
    '''  `v6_drop_term`, `v6_resume_drops_opt`, `v6_resume_drops_rng`, `v6_resume_lr_compounds`,
  `v6_ckpt_strips_rng`, `v6_loader_drops_vocab_version`). The rehearsal keeps
'''))
GATE.append((
    '''def run_smoke_v6(ctx: Ctx, T, argv: list[str], *, run_label: str, arms: bool = True) -> dict:
''',
    '''class _GateInterrupt(BaseException):
    """G-CKPT (v6): raised by the gate's `_save_ckpt` wrapper right after the checkpoint of the step it was
    asked to interrupt at -- the SIGKILL-right-after-a-checkpoint case, in-process. A BaseException, so no
    `except Exception` in the trainer can swallow it."""


def run_smoke_v6(ctx: Ctx, T, argv: list[str], *, run_label: str, arms: bool = True,
                 interrupt_at: int | None = None) -> dict:
'''))
GATE.append((
    '''      * `load_resume` -- the post-load digests (G-CKPT)."""
''',
    '''      * `load_resume` -- the post-load digests (G-CKPT);
      * `_save_ckpt` (with `interrupt_at`, or the arm `v6_ckpt_strips_rng`) -- the run dies right after
        the checkpoint of step `interrupt_at` (G-CKPT's interrupted run)."""
'''))
GATE.append((
    '''    pat.set(T, "apply_stage_freeze", freeze_wrap)
    pat.set(T, "build_trunk_optimizer", bto_wrap)
''',
    '''    if arms and arm_is(ctx, "v6_resume_drops_rng"):
        # F3 re-introduced: the resume restores NO stream -- every stream restarts from --seed (the pre-F3
        # trainer). A trainer without the restore seam needs no patch: it never restored anything.
        if hasattr(T, "apply_resume_state"):
            pat.set(T, "apply_resume_state", lambda state, **kw: {
                "t3_alpha_applied": None, "t3_prog_applied": None, "monitors": {}, "applied": []})
        rec["arm"] = {"v6_resume_drops_rng": "train_v6_staged.apply_resume_state -> a no-op"}
    if arms and arm_is(ctx, "v6_resume_lr_compounds"):
        # F3b re-introduced: the cosine schedule replayed ON TOP of the optimiser's LOADED lr
        def _replay_on_loaded_lr(opt, sched, n):
            for _ in range(int(n)):
                sched.step()
        if hasattr(T, "replay_lr_schedule"):
            pat.set(T, "replay_lr_schedule", _replay_on_loaded_lr)
        rec["arm"] = {"v6_resume_lr_compounds": "train_v6_staged.replay_lr_schedule -> sched.step() x n on "
                                                 "the LOADED lr (the pre-F3b replay)"}
    strip_rng = bool(arms and arm_is(ctx, "v6_ckpt_strips_rng"))
    if strip_rng:
        rec["arm"] = {"v6_ckpt_strips_rng": "_save_ckpt writes the pre-F3 payload {stack, opt, step, config}"}
    if interrupt_at is not None or strip_rng:
        orig_save = T._save_ckpt

        def save_wrap(path, *aa, _o=orig_save, **kk):
            if strip_rng and "extra" in kk:
                kk = {k: v for k, v in kk.items() if k != "extra"}     # a PRE-F3 checkpoint
            _o(path, *aa, **kk)
            if interrupt_at is not None and int(kk.get("step", -1)) == int(interrupt_at):
                rec["interrupted_at"] = int(interrupt_at)
                raise _GateInterrupt(f"G-CKPT: interrupted right after the step-{interrupt_at} checkpoint")
        pat.set(T, "_save_ckpt", save_wrap)
    pat.set(T, "apply_stage_freeze", freeze_wrap)
    pat.set(T, "build_trunk_optimizer", bto_wrap)
'''))
GATE.append((
    '''    try:
        rc = T.main(list(argv))
        if rc not in (0, None):
            rec["exit"] = f"main() returned {rc}: " + " | ".join(r[:300] for r in refusals[:4])
    except SystemExit as e:
''',
    '''    try:
        rc = T.main(list(argv))
        if rc not in (0, None):
            rec["exit"] = f"main() returned {rc}: " + " | ".join(r[:300] for r in refusals[:4])
    except _GateInterrupt:
        pass                                              # asked for: rec["interrupted_at"]
    except SystemExit as e:
'''))
GATE.append((
    '''    rec["elapsed_s"] = round(time.time() - t0, 1)
    if state["stack"] is not None:
''',
    '''    rec["elapsed_s"] = round(time.time() - t0, 1)
    if interrupt_at is not None and rec.get("interrupted_at") != int(interrupt_at) and not rec["exit"]:
        rec["exit"] = (f"the run was to die right after its step-{interrupt_at} checkpoint and did not (no "
                       f"such checkpoint was written)")
    if state["stack"] is not None:
'''))
GATE.append((
    '''    """G-CKPT (v6): save == in-memory; post-load == saved; the resumed data stream and final state ==
    an UNINTERRUPTED run's (the data position, measured -- the v6 checkpoint carries none)."""
''',
    '''    """G-CKPT (v6): save == in-memory; post-load == saved (optimiser param_groups INCLUDING lr); the
    checkpoint carries the streams; the resumed data stream, lr and final model + optimiser state == an
    UNINTERRUPTED run's.

    ⛔ PROTOCOL (F3, 2026-09-27): `run1` is an INTERRUPTED run launched with the FINAL --steps n+extra and
    killed right after its step-n checkpoint; `run2` resumes it with the SAME argv; `run3` is that argv
    uninterrupted. The previous protocol resumed a run that had FINISHED at --steps n with --steps n+extra.
    MEASURED: that run's first n steps already differ from run3's -- the cosine LR is a function of --steps
    (S-T rehearsal, step 50: lr 5.0000e-05 at T=100 vs 5.1540e-05 at T=102) -- so it FAILED on the final
    state even with an EXACT resume: it could not tell a correct trainer from a broken one."""
'''))
GATE.append((
    '''    det["ckpt_rng_or_data_position_keys"] = rng_keys
    if run2 is None:
''',
    '''    det["ckpt_rng_or_data_position_keys"] = rng_keys
    if not rng_keys:
        reasons.append("ckpt.pt carries no RNG state / data position (no key naming rng, data_pos or "
                       "sampler) -- a resume can only restart the stream from --seed")
    if run2 is None:
'''))
GATE.append((
    '''            g1 = [{k: v for k, v in g.items() if k != "lr"} for g in fin["opt"]["param_groups"]]
            g2 = [{k: v for k, v in g.items() if k != "lr"} for g in pl["opt"]["param_groups"]]
            if g1 != g2:
                reasons.append("resume: optimizer param_groups (lr excepted) differ from the saved")
    if not (run2_summary and run2_summary.get("done") is True
''',
    '''            g1 = [{k: v for k, v in g.items() if k != "lr"} for g in fin["opt"]["param_groups"]]
            g2 = [{k: v for k, v in g.items() if k != "lr"} for g in pl["opt"]["param_groups"]]
            det["resume_lr"] = {"saved": [g.get("lr") for g in fin["opt"]["param_groups"]],
                                "after_replay": [g.get("lr") for g in pl["opt"]["param_groups"]]}
            if g1 != g2:
                reasons.append("resume: optimizer param_groups (lr excepted) differ from the saved")
            elif det["resume_lr"]["saved"] != det["resume_lr"]["after_replay"]:
                # ⛔ F3b: this comparison used to EXCLUDE lr -- and that exclusion hid a defect in every
                # v6 resume: the cosine schedule replayed ON TOP of the loaded (already decayed) lr
                reasons.append(f"resume: the lr after the schedule replay {det['resume_lr']['after_replay']} "
                               f"!= the saved lr {det['resume_lr']['saved']} -- the resumed run trains on a "
                               f"different LR schedule (F3b: the cosine replay compounding on the loaded lr)")
    if not (run2_summary and run2_summary.get("done") is True
'''))
GATE.append((
    '''    if lm != "resume":
        reasons.append(f"the resume run's config.json launch_mode is {lm!r}, not 'resume'")
''',
    '''    if lm != "resume":
        reasons.append(f"the resume run's config.json launch_mode is {lm!r}, not 'resume'")
    rr = ((run2_config or {}).get("resume_rng") or {}).get("mode")
    det["resume_config_resume_rng"] = rr
    if rr != "restored":
        reasons.append(f"the resume run's config.json resume_rng.mode is {rr!r}, not 'restored' -- the "
                       f"trainer itself records that the continuation is not the uninterrupted run's")
'''))
GATE.append((
    '''                           " -- they are the uninterrupted run's FIRST batches: the sampler/RNG "
                           "stream restarts at every launch (train() re-seeds torch, random and the "
                           "sampler generator from --seed -- c36b6ddd:6475-6478 -- and _save_ckpt's "
                           "payload is {stack, opt, step, config}: no RNG state, no data position)"
''',
    '''                           " -- they are the uninterrupted run's FIRST batches: the sampler/RNG "
                           "stream restarts at every launch (train() re-seeds torch, random and the "
                           "sampler generator from --seed, and nothing restored the checkpoint's "
                           "streams -- the pre-F3 trainer, whose payload was {stack, opt, step, config})"
'''))
GATE.append((
    '''    f2, f3 = (run2.get("final") or {}).get("model"), (run3.get("final") or {}).get("model")
    if f2 and f3:
        cmp("resumed_final_vs_uninterrupted_final", f3, f2)
    return reasons, det
''',
    '''    f2, f3 = (run2.get("final") or {}).get("model"), (run3.get("final") or {}).get("model")
    if f2 and f3:
        cmp("resumed_final_vs_uninterrupted_final", f3, f2)
    else:
        reasons.append("the resumed or the uninterrupted run produced no final state to compare")
    o2, o3 = (run2.get("final") or {}).get("opt"), (run3.get("final") or {}).get("opt")
    if o2 and o3:
        cmp("resumed_final_opt_vs_uninterrupted_final", o3["state"], o2["state"])
        if o3["param_groups"] != o2["param_groups"]:
            reasons.append("the resumed run's final optimizer param_groups (incl. lr) differ from the "
                           "uninterrupted run's")
    return reasons, det
'''))
GATE.append((
    '''    def one_run(argv_r: list[str], label: str, arms: bool = True) -> dict:
        if reh:
            with v6_corpus_seam(ctx, corpus):
                return run_smoke_v6(ctx, T, argv_r, run_label=label, arms=arms)
        return run_smoke_v6(ctx, T, argv_r, run_label=label, arms=arms)
''',
    '''    def one_run(argv_r: list[str], label: str, arms: bool = True,
                interrupt_at: int | None = None) -> dict:
        if reh:
            with v6_corpus_seam(ctx, corpus):
                return run_smoke_v6(ctx, T, argv_r, run_label=label, arms=arms,
                                    interrupt_at=interrupt_at)
        return run_smoke_v6(ctx, T, argv_r, run_label=label, arms=arms, interrupt_at=interrupt_at)
'''))
GATE.append((
    '''        run2 = run3 = summ2 = cfg2 = None
        if ckpt is not None and not run1.get("exit"):
            (run_dir / "summary.json").unlink(missing_ok=True)   # run 1's marker, read above
            argv2 = set_flag(argv1, "--steps", [str(steps + extra)])
            run2 = one_run(argv2, "run2-resume")
            summ2 = read_json(run_dir / "summary.json") if (run_dir / "summary.json").is_file() \\
                else None
            cfg2 = read_json(run_dir / "config.json") if (run_dir / "config.json").is_file() else None
            ref_dir = root / "reference"
            argv3 = set_flag(set_flag(base, "--steps", [str(steps + extra)]), "--out", [str(ref_dir)])
            run3 = one_run(argv3, "run3-uninterrupted", arms=False)
        try:
            reasons, det = judge_ckpt_v6(run1, run2, run3, n=steps, extra=extra, ckpt=ckpt,
                                         run2_summary=summ2, run2_config=cfg2)
''',
    '''        run1i = run2 = run3 = summ2 = cfg2 = ckpt_i = None
        # ⛔ F3 PROTOCOL: resume an INTERRUPTED run launched with the FINAL --steps -- never a run that
        # FINISHED at --steps n, whose cosine schedule is a different one (judge_ckpt_v6). The three runs
        # share ONE argv (`--save-every n`, so the interrupted run checkpoints at n); only --out differs.
        intr_dir = root / "interrupted"
        argv_i = set_flag(set_flag(set_flag(base, "--steps", [str(steps + extra)]), "--out",
                                   [str(intr_dir)]), "--save-every", [str(steps)])
        protocol = {"argv": argv_i, "interrupt_after_checkpoint_at": steps, "resume_to": steps + extra,
                    "departures": {"--steps": str(steps + extra), "--save-every": str(steps),
                                   "why": "the interrupted run must checkpoint at n and share the final "
                                          "--steps with the resume and the reference"}}
        if not run1.get("exit"):
            run1i = one_run(argv_i, "run1-interrupted", interrupt_at=steps)
            ck_i = intr_dir / "ckpt.pt"
            ckpt_i = torch.load(ck_i, map_location="cpu", weights_only=False) if ck_i.is_file() else None
            if ckpt_i is not None and not run1i.get("exit"):
                run2 = one_run(argv_i, "run2-resume")
                summ2 = read_json(intr_dir / "summary.json") if (intr_dir / "summary.json").is_file() \\
                    else None
                cfg2 = read_json(intr_dir / "config.json") if (intr_dir / "config.json").is_file() \\
                    else None
                run3 = one_run(set_flag(argv_i, "--out", [str(root / "reference")]),
                               "run3-uninterrupted", arms=False)
        try:
            if run1i is None:
                reasons, det = [f"G-LIVE's smoke failed, so the G-CKPT protocol did not run: "
                                f"{str(run1.get('exit'))[:300]}"], {}
            else:
                reasons, det = judge_ckpt_v6(run1i, run2, run3, n=steps, extra=extra, ckpt=ckpt_i,
                                             run2_summary=summ2, run2_config=cfg2)
            det["protocol"] = protocol
'''))

# ================================================================================================================
# 3. the gate's tests -- the G-CKPT pins flipped, the new arms, a source mutation
#    (⚠️ anchored on the G-CKPT bullet line ALONE and on the G-CKPT test functions: the G-HYG bullet and
#     test_positive_G_HYG_names_the_open_config_classes_and_nothing_else belong to the F2 patch)
# ================================================================================================================
GATE_TESTS: list[tuple[str, str]] = []
GATE_TESTS.append((
    '''  * G-CKPT FAILS: a resume replays the sampler/RNG stream from --seed (no RNG state in ckpt.pt).
''',
    '''  * (F3 + F3b, FIXED 2026-09-27 and flipped below) G-CKPT: a resume replayed the sampler/RNG stream
    from --seed, and the cosine LR replay compounded on the loaded lr -- the rehearsals now PASS G-CKPT on
    the interrupted-run protocol (arms `v6_resume_drops_rng`, `v6_resume_lr_compounds`,
    `v6_ckpt_strips_rng`, `v6_resume_drops_opt` + a source mutation of the restore call).
'''))
GATE_TESTS.append((
    '''def test_positive_G_CKPT_names_the_TIP_defect_and_only_it(st_positive):
    _ctx_, evs = st_positive
    ev = evs["G-CKPT"]
    r = _reasons(ev)
    assert ev["status"] == "FAIL"
    assert "restarts at every launch" in r                      # MEASURED: the RNG stream replays
    for clean in ("save_model_vs_memory", "save_opt_state_vs_memory", "resume_model_vs_saved",
                  "resume_opt_state_vs_saved"):
        assert clean not in r, clean                            # save -> load itself is exact
''',
    '''def test_positive_G_CKPT_PASSES_the_resume_IS_the_uninterrupted_run(st_positive):
    """Was F3 (MEASURED 2026-09-27: the resumed S-T run drew the uninterrupted run's FIRST batches) and F3b
    (the cosine replay compounding on the loaded lr, hidden by an lr-excluding comparison). Flipped in the
    same change as the trainer fix: the interrupted-then-resumed S-T rehearsal draws the uninterrupted
    run's batches at the same steps and ends BIT-identical in model AND optimiser state."""
    _ctx_, evs = st_positive
    ev = evs["G-CKPT"]
    assert ev["status"] == "PASS", _reasons(ev)
    d = ev["details"]
    assert d["ckpt_rng_or_data_position_keys"] == ["data_pos", "rng_state"]
    assert len(d["draws_resumed"]) == 2                                       # resume_extra_steps
    assert d["draws_resumed"] == d["draws_uninterrupted_same_steps"]
    assert d["draws_resumed"] != d["draws_uninterrupted_first_steps"]         # NOT a replay
    for k in ("save_model_vs_memory", "save_opt_state_vs_memory", "resume_model_vs_saved",
              "resume_opt_state_vs_saved", "resumed_final_vs_uninterrupted_final",
              "resumed_final_opt_vs_uninterrupted_final"):
        assert d[k]["n"] > 0 and d[k]["n_differ"] == 0, k
    assert d["resume_lr"]["saved"] == d["resume_lr"]["after_replay"]
    assert d["resume_config_launch_mode"] == "resume" and d["resume_config_resume_rng"] == "restored"
    assert d["protocol"]["interrupt_after_checkpoint_at"] == 4 and d["protocol"]["resume_to"] == 6


def test_positive_S_W_G_CKPT_PASSES(tmp_path):
    ev = LG.job_smoke_v6(_ctx(tmp_path, SW_ARGV), ["G-CKPT"])["G-CKPT"]
    assert ev["status"] == "PASS", _reasons(ev)
    d = ev["details"]
    assert d["draws_resumed"] == d["draws_uninterrupted_same_steps"]
    assert d["draws_resumed"] != d["draws_uninterrupted_first_steps"]
    assert d["resumed_final_vs_uninterrupted_final"]["n_differ"] == 0
    assert d["resumed_final_opt_vs_uninterrupted_final"]["n_differ"] == 0
'''))
GATE_TESTS.append((
    '''def test_ARM_v6_resume_drops_opt_FAILS_G_CKPT_on_the_optimizer(tmp_path):
    ctx = _ctx(tmp_path, ST_ARGV, arm="v6_resume_drops_opt")
    ev = LG.job_smoke_v6(ctx, ["G-CKPT"])["G-CKPT"]
    assert ev["status"] == "FAIL" and "resume_opt_state_vs_saved" in _reasons(ev)
''',
    '''def test_ARM_v6_resume_drops_opt_FAILS_G_CKPT_on_the_optimizer(tmp_path):
    ctx = _ctx(tmp_path, ST_ARGV, arm="v6_resume_drops_opt")
    ev = LG.job_smoke_v6(ctx, ["G-CKPT"])["G-CKPT"]
    assert ev["status"] == "FAIL" and "resume_opt_state_vs_saved" in _reasons(ev)


def test_ARM_v6_resume_drops_rng_REPLAYS_the_first_batches_and_FAILS_G_CKPT(tmp_path):
    """F3 re-introduced through the trainer's own seam: the resume restores no stream."""
    ev = LG.job_smoke_v6(_ctx(tmp_path, ST_ARGV, arm="v6_resume_drops_rng"), ["G-CKPT"])["G-CKPT"]
    r = _reasons(ev)
    assert ev["status"] == "FAIL" and "restarts at every launch" in r, r
    d = ev["details"]
    assert d["draws_resumed"] == d["draws_uninterrupted_first_steps"]         # the MEASURED defect
    assert "resumed_final_vs_uninterrupted_final" in r


def test_ARM_v6_resume_lr_compounds_FAILS_G_CKPT_on_the_lr(tmp_path):
    """F3b re-introduced: the draws stay exact, but the lr after the replay is lr_n x (lr_n / lr_0) --
    the whole remainder of the run on a scaled schedule."""
    ev = LG.job_smoke_v6(_ctx(tmp_path, SW_ARGV, arm="v6_resume_lr_compounds"), ["G-CKPT"])["G-CKPT"]
    r = _reasons(ev)
    assert ev["status"] == "FAIL" and "F3b" in r, r
    d = ev["details"]
    assert d["draws_resumed"] == d["draws_uninterrupted_same_steps"]          # F3 itself holds
    s, a = d["resume_lr"]["saved"][0], d["resume_lr"]["after_replay"][0]
    assert s > 0 and abs(a / s - s / 1e-4) < 1e-9                             # --lr 0.0001 in the argv


def test_ARM_v6_ckpt_strips_rng_a_PRE_F3_checkpoint_FAILS_G_CKPT_and_the_trainer_records_it(tmp_path):
    """A checkpoint in the pre-F3 layout {stack, opt, step, config}: the trainer resumes it LOUDLY
    (`resume_rng.mode = "absent-replayed"`), and G-CKPT FAILS on the missing keys, the replay AND the record."""
    ev = LG.job_smoke_v6(_ctx(tmp_path, SW_ARGV, arm="v6_ckpt_strips_rng"), ["G-CKPT"])["G-CKPT"]
    r = _reasons(ev)
    assert ev["status"] == "FAIL"
    assert "carries no RNG state" in r and "restarts at every launch" in r and "'absent-replayed'" in r, r
    assert ev["details"]["ckpt_rng_or_data_position_keys"] == []


#: the F3 restore call in the trainer (its first line) -- the source mutation below deletes the call
F3_RESTORE_CALL = "_rl = apply_resume_state(resume_state, device=device, rng=rng, gen=gen,"


def test_ARM_SOURCE_deleting_the_resume_restore_call_FAILS_G_CKPT(tmp_path, monkeypatch):
    """Mutation, not inspection: the F3 restore call replaced by a no-op in the trainer SOURCE (compiled
    under its own path) -- G-CKPT must FAIL on the replayed stream."""
    import types
    src_path = TREE / LG.PROFILES["v7f"]["trainer"]
    src = src_path.read_text(encoding="utf-8")
    assert src.count(F3_RESTORE_CALL) == 1                     # the fix is present, exactly once
    src = src.replace(F3_RESTORE_CALL, "_rl = (lambda *_a, **_k: {'t3_alpha_applied': None, "
                      "'t3_prog_applied': None, 'monitors': {}, 'applied': []})(resume_state, "
                      "device=device, rng=rng, gen=gen,")
    mod = types.ModuleType(LG._TRAINER_MOD_V6)
    mod.__file__ = str(src_path)
    monkeypatch.setitem(sys.modules, LG._TRAINER_MOD_V6, mod)  # the gate's loader returns it
    exec(compile(src, str(src_path), "exec"), mod.__dict__)
    ev = LG.job_smoke_v6(_ctx(tmp_path, SW_ARGV), ["G-CKPT"])["G-CKPT"]
    r = _reasons(ev)
    assert ev["status"] == "FAIL" and "restarts at every launch" in r, r
    assert ev["details"]["draws_resumed"] == ev["details"]["draws_uninterrupted_first_steps"]
'''))
NEW_FILES = {"tests/test_v6_exact_resume.py": HERE / "fix" / "stack" / "tests" / "test_v6_exact_resume.py"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stack", default="C:/Users/Admin/v7f_merge/stack")
    ap.add_argument("--only", default="trainer,gate,tests")
    a = ap.parse_args(argv)
    root = Path(a.stack)
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    if "trainer" in only:
        _patch(root / "scripts" / "train_v6_staged.py", TRAINER)
    if "gate" in only and GATE:
        _patch(root / "scripts" / "launch_gate.py", GATE)
    if "tests" in only:
        if GATE_TESTS:
            _patch(root / "tests" / "test_launch_gate_v7f.py", GATE_TESTS)
        for rel, src in NEW_FILES.items():
            dst = root / rel
            if not src.is_file():
                sys.exit(f"new file source missing: {src}")
            data = src.read_bytes()
            if dst.exists() and dst.read_bytes() != data:
                sys.exit(f"{dst} exists with DIFFERENT content -- refusing to overwrite")
            dst.write_bytes(data)
            print(f"wrote {dst} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
