"""refcv9 C6: the learning-rate schedule (``cosine`` | ``wsd``) and the WSD DECAY BRANCH.

REFCV9_DESIGN sec. 2.6 "Schedule (R6)" + sec. 4.4 (E2 / E3). DESIGN IS PINNED: this module implements it.

WHY (MEASURED, REFCV9_DESIGN sec. 2.6): refcv8's cosine left one pre-registered endpoint and a 40k checkpoint that scored
better on the test set -- which could not be chosen without selecting on eval139. A WSD schedule (warm-up, STABLE at the
peak, then a LINEAR decay over the last N steps) makes **decay branches** possible: the early read D1 resumes a stable-
phase checkpoint (step 30,000) and decays it over 6,000 steps on its own, while the main run keeps going at the peak.

THE FACTOR (a PURE function of ``(step, args)`` -- a resume reproduces it, nothing is carried in a checkpoint):

* ``cosine`` (the default; BIT-IDENTICAL to the lambda this module replaced): ``(s+1)/warmup`` for ``s < warmup``, then
  ``0.5 (1 + cos(pi (s - warmup) / (steps - warmup)))``.
* ``wsd`` (``--wsd-decay-steps N``): ``(s+1)/warmup`` for ``s < warmup`` (the same linear warm-up), then ``1.0`` while
  ``s < steps - N``, then ``(steps - s)/N`` -- linear, EXACTLY 1.0 at the decay start ``steps - N`` and EXACTLY 0.0 at
  ``s == steps`` (a step that is never executed: the last update uses ``1/N``, the cosine convention).

THE DECAY BRANCH (``--wsd-branch-from <ckpt> --wsd-branch-decay M``): a NEW run (its own ``--out``) that resumes the FULL
state of a stable-phase checkpoint of step ``s`` (model, optimiser, EMA, data position, the stamps the resume guards
check), keeps the factor at the stable value and decays linearly over ``s .. s+M`` to 0, ending at ``s+M``. Mechanism: the
checkpoint is STAGED (bit-for-bit copy, md5-verified) as the branch's own ``<out>/ckpt.pt``, so the trainer's ONE existing
resume path (and every guard on it) runs unchanged; the branch's factor is then the plain WSD factor of a run with
``steps = s + M`` and ``decay = M``. It REFUSES: a checkpoint in the warm-up or the decay phase of ITS OWN run; a source
whose ``config.json`` is missing or not WSD; a branch argv that differs from the source's in anything but ``--out`` /
``--steps`` / the branch flags / ``--save-at`` (every differing key is named); ``--steps != s + M``; a model-only
checkpoint; a ``--out`` already holding some other run's ``ckpt.pt``.

``--save-at s1,s2,...`` writes a FULL-STATE ``ckpt_full_<s>.pt`` at those steps. (The trainer's existing mechanisms cannot:
``ckpt.pt`` is overwritten every ``--save-every``; ``--snapshot-steps`` / ``MILESTONES`` write MODEL-ONLY files, which
cannot be resumed and so cannot seed a branch.)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

SCHEDULES = ("cosine", "wsd")
#: the full-state file name pattern of ``--save-at`` (NOT ``ckpt_<step>.pt``: that is the MODEL-ONLY milestone name)
FULL_CKPT_FMT = "ckpt_full_%d.pt"
#: written into a branch's ``--out`` beside the staged ``ckpt.pt``
BRANCH_RECORD = "wsd_branch_source.json"
#: argv keys a branch MAY differ from its source in (argparse dests). ``save_at`` is a checkpoint-cadence flag of this module.
BRANCH_EXEMPT = frozenset({"out", "steps", "wsd_branch_from", "wsd_branch_decay", "save_at"})
#: the checkpoint keys a branch needs (``data_pos``: a legacy checkpoint would RESTART the data order)
BRANCH_REQUIRED_KEYS = ("model", "opt", "step", "data_pos")


# --------------------------------------------------------------------------- #
# the factor -- pure                                                          #
# --------------------------------------------------------------------------- #
def cosine_factor(step, warmup, steps):
    """VERBATIM the pre-C6 ``sched`` lambda of ``refc_v3_train.train`` (warm-up, then cosine to 0). Do not 'tidy' it:
    ``tests/test_lr_schedule_wsd.py`` holds it bit-identical to the old expression at every step."""
    return (step + 1) / max(1, warmup) if step < warmup else \
        0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))


def wsd_factor(step, warmup, steps, decay):
    """Warm-up (linear) -> STABLE 1.0 -> linear decay to 0 over the last ``decay`` steps of ``steps``."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if step < steps - decay:
        return 1.0
    return max(0.0, (steps - step) / decay)


def phase_of(step, warmup, steps, decay):
    """``'warmup'`` | ``'stable'`` | ``'decay'`` of a WSD run. ``step`` is the number of updates DONE (a checkpoint's
    ``step``), i.e. the index of the NEXT update, whose factor is ``wsd_factor(step, ...)``. The decay phase starts AT
    ``steps - decay`` (where the factor is still exactly 1.0)."""
    if step < warmup:
        return "warmup"
    if step >= steps - decay:
        return "decay"
    return "stable"


def decay_steps_in_use(args) -> int:
    """The N of the factor. In a branch it is the BRANCH decay M (the run's ``--steps`` is ``s + M``); otherwise
    ``--wsd-decay-steps``."""
    if getattr(args, "wsd_branch_from", None):
        return int(getattr(args, "wsd_branch_decay", 0) or 0)
    return int(getattr(args, "wsd_decay_steps", 0) or 0)


def lr_factor(step, args):
    """The learning-rate factor at ``step`` -- a pure function of ``(step, args)``. An ``args`` without the C6 flags
    (an older namespace) is the cosine."""
    if str(getattr(args, "lr_schedule", "cosine")) == "wsd":
        return wsd_factor(step, args.warmup, args.steps, decay_steps_in_use(args))
    return cosine_factor(step, args.warmup, args.steps)


def make_sched(args):
    """The ``sched`` callable ``apply_lr_schedule`` scales every group by. Reads ``args`` at call time (as the lambda it
    replaces did), so the run's own ``--steps`` / ``--warmup`` are the ones used."""
    def sched(step):
        return lr_factor(step, args)
    return sched


# --------------------------------------------------------------------------- #
# argv refusals -- pure, fire before any data or GPU work                     #
# --------------------------------------------------------------------------- #
def parse_save_at(spec, steps: int) -> frozenset:
    """``--save-at "30000,60000"`` -> the frozenset of steps that get a FULL-STATE ``ckpt_full_<step>.pt``. ``None`` ->
    empty. REFUSES a non-integer, a step outside ``[1, --steps]`` and a duplicate: a save that can never be written is a
    stable-phase checkpoint that silently does not exist."""
    if spec in (None, ""):
        return frozenset()
    try:
        vals = [int(x) for x in str(spec).split(",") if x.strip() != ""]
    except ValueError:
        raise SystemExit(f"[v3] ⛔ --save-at {spec!r}: not a comma list of integers") from None
    if len(set(vals)) != len(vals):
        raise SystemExit(f"[v3] ⛔ --save-at {spec!r}: duplicate step")
    bad = [v for v in vals if v < 1 or v > int(steps)]
    if bad:
        raise SystemExit(f"[v3] ⛔ --save-at {bad}: outside [1, --steps {steps}] -- never written")
    return frozenset(vals)


def check_args(args) -> None:
    """Every pure-argv refusal of the schedule flags. A no-op on the defaults (cosine, no branch, no save-at)."""
    sched = str(getattr(args, "lr_schedule", "cosine"))
    n = int(getattr(args, "wsd_decay_steps", 0) or 0)
    bfrom = getattr(args, "wsd_branch_from", None)
    bdec = int(getattr(args, "wsd_branch_decay", 0) or 0)
    steps, warmup = int(args.steps), int(args.warmup)
    if sched not in SCHEDULES:
        raise SystemExit(f"[v3] ⛔ --lr-schedule {sched!r} not in {SCHEDULES}")
    if sched == "cosine":
        if n != 0:
            raise SystemExit(f"[v3] ⛔ --wsd-decay-steps {n} with --lr-schedule cosine: the flag would be recorded in "
                             "config.json and change nothing. Pass --lr-schedule wsd, or drop the flag.")
        if bfrom or bdec:
            raise SystemExit("[v3] ⛔ --wsd-branch-from / --wsd-branch-decay need --lr-schedule wsd (a branch decays a "
                             "stable-phase checkpoint; the cosine has no stable phase).")
    else:
        if n <= 0:
            raise SystemExit("[v3] ⛔ --lr-schedule wsd needs --wsd-decay-steps N > 0 (the decay phase is the last N steps; "
                             f"got {n}) -- without it the 'decay' would never start and the run would train at the peak "
                             "to the end.")
    if bool(bfrom) != (bdec > 0):
        raise SystemExit("[v3] ⛔ --wsd-branch-from and --wsd-branch-decay M > 0 go together (got "
                         f"from={bfrom!r}, decay={bdec}).")
    eff = bdec if bfrom else n
    if sched == "wsd":
        if bfrom:
            # a branch starts at s = steps - M; whether s is past the warm-up is the CHECKPOINT's phase check (stage_branch)
            if bdec > steps:
                raise SystemExit(f"[v3] ⛔ --wsd-branch-decay {bdec} > --steps {steps}: the branch ends at s + M.")
        else:
            if n > steps:
                raise SystemExit(f"[v3] ⛔ --wsd-decay-steps {n} > --steps {steps}.")
            if warmup > steps - n:
                raise SystemExit(f"[v3] ⛔ --warmup {warmup} > --steps {steps} - decay {n} = {steps - n}: the warm-up would "
                                 "run into the decay and there is no stable phase.")
    parse_save_at(getattr(args, "save_at", None), steps)


# --------------------------------------------------------------------------- #
# the startup guard on the BUILT sched closure (G-DVB, kind "elsewhere")       #
# --------------------------------------------------------------------------- #
def assert_sched_matches_args(sched, args) -> dict:
    """Probe the BUILT ``sched`` closure against closed forms read straight off argv (no shared code with
    ``wsd_factor`` / ``cosine_factor``): warm-up shape, the stable value, the decay start, the last update's value and the
    value at ``steps``. REFUSES on any mismatch. Returns the probe table (stamped into config.json)."""
    steps, warmup = int(args.steps), int(args.warmup)
    wsd = str(getattr(args, "lr_schedule", "cosine")) == "wsd"
    n = decay_steps_in_use(args)
    want = {0: 1.0 / max(1, warmup)}
    if warmup >= 1:
        want[warmup - 1] = 1.0
    if wsd:
        start = steps - n
        if start < warmup:                    # only reachable by a BRANCH (check_args refuses it for a plain run)
            raise SystemExit(f"[v3] ⛔ THE DECAY WOULD START AT STEP {start}, INSIDE THE WARM-UP (--warmup {warmup}): there is no "
                             "stable phase. A branch (--steps s + M, decay M) from a checkpoint in the warm-up is refused.")
        if start > warmup:
            want[(warmup + start) // 2] = 1.0
        want[start] = 1.0
        want[steps - 1] = 1.0 / n
        want[steps] = 0.0
    else:
        want[warmup] = 1.0
        if steps > warmup:          # a run shorter than its warm-up (the CI smokes: --steps 1) never reaches the cosine
            want[steps] = 0.0
    got, bad = {}, []
    for s, w in sorted(want.items()):
        g = float(sched(s))
        got[s] = g
        if abs(g - w) > 1e-12:
            bad.append(f"step {s}: sched {g!r} != argv-derived {w!r}")
    if bad:
        raise SystemExit("[v3] ⛔ THE SCHEDULE THAT WILL RUN IS NOT THE ONE THE ARGV DECLARES (--lr-schedule "
                         f"{getattr(args, 'lr_schedule', 'cosine')}, --warmup {warmup}, --steps {steps}, decay {n}):\n  - "
                         + "\n  - ".join(bad))
    return {str(k): v for k, v in got.items()}


# --------------------------------------------------------------------------- #
# the decay branch                                                            #
# --------------------------------------------------------------------------- #
def file_md5(path, chunk=1 << 23) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _argv_of(cfg: dict):
    """The source run's argv: ``argv_effective`` (what ``main`` was really given) else ``argv`` (``sys.argv[1:]``)."""
    a = cfg.get("argv_effective")
    return list(a) if a else list(cfg.get("argv") or [])


def read_source_namespace(config_path, parser):
    """The SOURCE run's argparse namespace, re-parsed from its ``config.json`` argv with the SAME parser."""
    try:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(f"[v3] ⛔ --wsd-branch-from: cannot read the source run's {config_path} ({e}); the branch "
                         "cannot be checked against the run that made the checkpoint.") from None
    argv = _argv_of(cfg)
    if not argv:
        raise SystemExit(f"[v3] ⛔ --wsd-branch-from: {config_path} carries no argv; the source run cannot be reconstructed.")
    try:
        return parser.parse_args(argv), cfg
    except SystemExit:
        raise SystemExit(f"[v3] ⛔ --wsd-branch-from: the source argv in {config_path} does not parse with this trainer's "
                         "parser.") from None


def argv_differences(args, src_ns) -> list:
    """``[(flag, branch value, source value), ...]`` for every key the branch differs on, OUTSIDE ``BRANCH_EXEMPT`` (and
    private ``_*`` attributes). ``args`` must be the namespace as PARSED (before the trainer pins anything into it)."""
    a, b = vars(args), vars(src_ns)
    out = []
    for k in sorted(set(a) | set(b)):
        if k.startswith("_") or k in BRANCH_EXEMPT:
            continue
        if a.get(k) != b.get(k):
            out.append(("--" + k.replace("_", "-"), a.get(k), b.get(k)))
    return out


def branch_preflight(args, parser):
    """Cheap, before anything is built: ``None`` without ``--wsd-branch-from``; else the source paths + namespace after
    the argv check. REFUSES a missing checkpoint / ``config.json`` and any differing key (all named)."""
    bfrom = getattr(args, "wsd_branch_from", None)
    if not bfrom:
        return None
    ckpt = Path(bfrom)
    if not ckpt.is_file():
        raise SystemExit(f"[v3] ⛔ --wsd-branch-from {bfrom}: no such file.")
    cfg_path = ckpt.parent / "config.json"
    src_ns, _ = read_source_namespace(cfg_path, parser)
    diffs = argv_differences(args, src_ns)
    if diffs:
        raise SystemExit("[v3] ⛔ --wsd-branch-from: the branch's argv differs from the source run's (read from "
                         f"{cfg_path}) in keys other than --out / --steps / the branch flags / --save-at:\n  - "
                         + "\n  - ".join(f"{f}: branch {x!r} vs source {y!r}" for f, x, y in diffs)
                         + "\nA branch is the SAME experiment decayed early; anything else it changes is a different arm.")
    return {"ckpt": ckpt, "config": cfg_path, "src_ns": src_ns}


def _read_ckpt_head(path) -> dict:
    """``{'step', 'keys'}`` of a checkpoint without materialising its tensors (``mmap``)."""
    import gc

    import torch
    st = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    try:
        return {"step": int(st["step"]) if "step" in st else None, "keys": sorted(st.keys())}
    finally:
        del st
        gc.collect()


def check_branch_checkpoint(args, src_ns, step: int, keys=()) -> None:
    """The refusals that need the checkpoint's STEP."""
    keys = set(keys)
    miss = [k for k in BRANCH_REQUIRED_KEYS if keys and k not in keys]
    if miss:
        raise SystemExit(f"[v3] ⛔ --wsd-branch-from: the checkpoint lacks {miss} (keys {sorted(keys)}): a branch resumes "
                         "the FULL state; a model-only snapshot (ckpt_<step>.pt) cannot seed one. Use ckpt_full_<step>.pt "
                         "(--save-at) or a ckpt.pt.")
    if str(getattr(src_ns, "lr_schedule", "cosine")) != "wsd":
        raise SystemExit(f"[v3] ⛔ --wsd-branch-from: the source run used --lr-schedule {src_ns.lr_schedule!r}, not wsd: it "
                         "has no stable phase to branch from.")
    n_src, w_src, st_src = decay_steps_in_use(src_ns), int(src_ns.warmup), int(src_ns.steps)
    ph = phase_of(step, w_src, st_src, n_src)
    if ph != "stable":
        raise SystemExit(f"[v3] ⛔ --wsd-branch-from: the checkpoint is at step {step}, in the {ph.upper()} phase of its own "
                         f"run (warm-up < {w_src}; stable [{w_src}, {st_src - n_src}); decay [{st_src - n_src}, {st_src}]). "
                         "Only a STABLE-phase checkpoint can be decayed early: its factor is the peak.")
    m = int(args.wsd_branch_decay)
    if int(args.steps) != step + m:
        raise SystemExit(f"[v3] ⛔ --steps {args.steps} != checkpoint step {step} + --wsd-branch-decay {m} = {step + m}: "
                         "the branch ends at s + M (pass --steps %d)." % (step + m))
    dead = sorted(s for s in parse_save_at(getattr(args, "save_at", None), int(args.steps)) if s <= step)
    if dead:
        raise SystemExit(f"[v3] ⛔ --save-at {dead} <= the branch's start step {step}: never written.")


def stage_branch(args, out_dir, src):
    """Stage the stable checkpoint as ``<out>/ckpt.pt`` (so the trainer's own resume runs, guards and all) and return the
    record that goes into config.json. ``None`` when there is no branch. A relaunch of a branch whose ``ckpt.pt`` is
    already staged (and is this branch's) stages nothing."""
    if src is None:
        return None
    out_dir = Path(out_dir)
    ck, rec_path = out_dir / "ckpt.pt", out_dir / BRANCH_RECORD
    ckpt, m = src["ckpt"], int(args.wsd_branch_decay)
    head = _read_ckpt_head(ckpt)
    if head["step"] is None:
        raise SystemExit(f"[v3] ⛔ --wsd-branch-from {ckpt}: the checkpoint carries no 'step'.")
    step = head["step"]
    check_branch_checkpoint(args, src["src_ns"], step, head["keys"])
    md5 = file_md5(ckpt)
    sn = src["src_ns"]
    rec = {"path": str(ckpt), "md5": md5, "step": step, "branch_decay": m, "steps": int(args.steps),
           "source_config": str(src["config"]),
           "source_run": {"lr_schedule": sn.lr_schedule, "warmup": int(sn.warmup), "steps": int(sn.steps),
                          "wsd_decay_steps": int(sn.wsd_decay_steps), "phase_of_checkpoint": "stable"}}
    if ck.exists():
        old = None
        if rec_path.exists():
            try:
                old = json.loads(rec_path.read_text(encoding="utf-8"))
            except ValueError:
                old = None
        diff = [k for k in ("md5", "step", "branch_decay") if old is None or old.get(k) != rec[k]]
        if diff:
            raise SystemExit(f"[v3] ⛔ --out {out_dir} already holds a ckpt.pt that is NOT this branch's (" +
                             (f"{BRANCH_RECORD} missing" if old is None else f"{BRANCH_RECORD} differs in {diff}") +
                             "). Use a fresh --out for a branch.")
        rec["staged"] = False
        print(f"[v3] wsd branch: --out already holds this branch's staged ckpt.pt (source step {step}, md5 {md5[:8]}); resuming it",
              flush=True)
        return rec
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")          # record FIRST: ckpt.pt exists => record exists
    tmp = out_dir / "ckpt.pt.staging"
    shutil.copyfile(ckpt, tmp)
    got = file_md5(tmp)
    if got != md5:
        tmp.unlink()
        raise SystemExit(f"[v3] ⛔ wsd branch: the staged copy's md5 {got} != the source's {md5}.")
    os.replace(tmp, ck)
    rec["staged"] = True
    print(f"[v3] wsd branch: staged {ckpt} (step {step}, md5 {md5[:8]}) as {ck}; decay over {step} .. {step + m}", flush=True)
    return rec


def config_stamp(args, branch_rec) -> dict:
    """The keys config.json gains: the schedule, its decay length, the branch's source ``(path, md5, step)`` or ``None``,
    and the argv ``main`` was really given (the branch-of-this-run check reads it)."""
    import sys
    return {"lr_schedule": str(getattr(args, "lr_schedule", "cosine")),
            "wsd_decay_steps": int(getattr(args, "wsd_decay_steps", 0) or 0),
            "wsd_branch": branch_rec,
            "argv_effective": list(getattr(args, "_ew_argv", None) or sys.argv[1:])}


# --------------------------------------------------------------------------- #
# G-DVB                                                                       #
# --------------------------------------------------------------------------- #
def register_dvb_levers(register) -> None:
    """The five flags enter the G-DVB registry (``declared_vs_built.register``). ``lr_schedule`` / ``wsd_decay_steps`` are
    verified on the BUILT ``sched`` closure by ``assert_sched_matches_args`` (kind ``elsewhere``, like ``lr``); the branch
    flags by ``branch_preflight`` / ``stage_branch`` refusals + the config.json stamp; ``save_at`` is a cadence flag
    (``runtime``, like ``--save-every`` / ``--snapshot-steps``)."""
    register("lr_schedule", "elsewhere", reason=(
        "the schedule that will run is checked against argv by `lr_schedule.assert_sched_matches_args` on the BUILT `sched` "
        "closure at startup (warm-up shape, stable value, decay start, last update, step == steps)"))
    register("wsd_decay_steps", "elsewhere", reason=(
        "checked with --lr-schedule by `lr_schedule.assert_sched_matches_args` (decay start = steps - N; last update 1/N) "
        "and by `lr_schedule.check_args` (> 0 iff wsd)"))
    register("wsd_branch_from", "elsewhere", reason=(
        "the branch's source checkpoint is validated and STAGED by `lr_schedule.branch_preflight` / `stage_branch` "
        "(stable phase, argv identical to the source run's, md5 verified) and stamped in config.json `wsd_branch`"))
    register("wsd_branch_decay", "elsewhere", reason=(
        "checked with --wsd-branch-from (`lr_schedule.stage_branch`: --steps == source step + M) and by "
        "`assert_sched_matches_args` (the branch's decay is the factor's N)"))
    register("save_at", "runtime", reason=(
        "the FULL-STATE checkpoint cadence (`ckpt_full_<step>.pt`); a step that can never be written REFUSES "
        "(`lr_schedule.parse_save_at`)"))
