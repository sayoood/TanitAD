"""SLOWWM planner read: ``ladder_plantest/lp_lib.py`` for an arm whose z1 input is a closed-form basis (``--wm-basis slow|pca``) or the adapter token mean (R35H). A THIN layer: lp_lib, lr_lib and the hook
``slowwm/slowwm_hook.py`` are imported and never edited (PREREG_V8F_SLOWWM, frozen sha256 7c2775e3...94ca). Tier T0, CPU.

What differs for a basis arm, and nothing else:
  * the whole-clip level-1 latents (``lp_lib.clip_z1``, called by ``build_ctx``) come from ``SH.clip_z1_basis`` (``wm.z1_from_basis`` of the clip's own ``u``); an adapter arm keeps lp_lib's function. The G-TRUE goal
    latent is the latent at the future knot, i.e. "the arm's own basis applied to the future frame";
  * ``prepare_arm_read`` refuses the arm before any scoring unless (a) the BUILT model's basis route reads back the argv's ``--wm-basis`` (mean and token paths refused, basis path = proj tail), (b) the argv /
    BUILT-strides check of the ladder passes, (c) ``levers.wm_basis_readback`` of the run's ``selfcheck.json`` carries the kind and sha256 of the basis file handed to the reader (``SH.require_arm_basis``), and
    (d) the eval ``ubasis`` file was built with that same basis (``SH.attach_basis_frames``).
Windows, planner, lambda grid and choice rule are the ladder's own (``LP.sel_arrays``, ``LP.unit3``, plantest2_lib)."""
import contextlib
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.normpath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("LADDER_READ_CODE_ROOT", "F:/TanitAD-scratch/slowwm/tree")
os.environ.setdefault("WMACT_CODE_ROOT", os.environ["LADDER_READ_CODE_ROOT"])
sys.path.insert(0, os.path.join(THEORY, "slowwm"))
sys.path.insert(0, os.path.join(THEORY, "ladder_plantest"))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import lp_lib as LP  # noqa: E402  (puts the ladder readers and the gates on sys.path, asserts tanitad is under the code root)
import lr_lib as LR  # noqa: E402
import plantest_lib as P  # noqa: E402
import slowwm_hook as SH  # noqa: E402

FRAMES_SEED = {0: 0, 1: 1, 2: 0}         # the seed-2 replicate keeps the readers and frames of the s0 init (HANDOFF_SLOWWM: "SLOW_rep uses seed 2 with the s0 init")
ARM_PREFIX_KIND = (("SLOW", "slow"), ("PCA-B", "pca"), ("R35H", "adapter"))


def seed_of(name):
    return int(name.rsplit("_s", 1)[1])


def kind_of(name):
    for pre, k in ARM_PREFIX_KIND:
        if name.startswith(pre):
            return k
    raise ValueError(f"arm name {name!r} starts with none of {[p for p, _ in ARM_PREFIX_KIND]}")


@contextlib.contextmanager
def basis_clip_z1():
    """``LP.build_ctx`` looks ``clip_z1`` up as a module global: dispatch to the hook for a basis model, to lp_lib's own function for every other model. Undone on exit."""
    orig = LP.clip_z1

    def dispatch(wm, fr):
        return SH.clip_z1_basis(wm, fr) if SH.is_basis(wm) else orig(wm, fr)
    LP.clip_z1 = dispatch
    try:
        yield
    finally:
        LP.clip_z1 = orig


def built_basis_problems(model, declared):
    """The BUILT module's own measurement of its z1 route against the argv's ``--wm-basis``; [] when they agree."""
    bb = model.wm.built_basis()
    bad = []
    if bb["mode"] != declared:
        bad.append(f"BUILT wm_basis {bb['mode']} vs argv {declared}")
    if declared == "adapter":
        if bb["mean_path_refused"] or bb["token_path_refused"] or not bb["basis_path_refused"]:
            bad.append(f"adapter arm: mean/token/basis refusals read {bb}")
    elif not (bb["mean_path_refused"] and bb["token_path_refused"] and not bb["basis_path_refused"] and bb["basis_path_is_proj_tail"]):
        bad.append(f"basis arm: mean/token path not refused or basis path not the proj tail: {bb}")
    return bad


def load_checked(name, run_dir, basis_npz=None, ubasis_npz=None, frames_dir=None):
    """Model + eval frames of one SLOWWM arm after every refusal (see the module docstring); adapter arms (R35H) skip the basis checks. Returns a dict."""
    kind = kind_of(name)
    model, cfg, argv, ck = LR.load_arm(run_dir)
    declared = (argv.get("wm-basis") or ["adapter"])[0]
    bad = built_basis_problems(model, kind)
    if declared != kind:
        bad.append(f"argv --wm-basis {declared} vs the arm name {name} ({kind})")
    if kind != "adapter" and (not basis_npz or not ubasis_npz):
        bad.append(f"{name}: a {kind} arm needs its basis file and its eval ubasis file")
    if bad:
        raise LR.LadderMismatch("; ".join(bad))
    built = LR.built_values(model)
    checked = LR.check_built_vs_argv(built, argv)
    seed = seed_of(name)
    if int(argv["seed"][0]) != seed:
        raise LR.LadderMismatch(f"{name}: argv --seed {argv['seed']}")
    fseed = FRAMES_SEED[seed]
    fr = LR.load_eval_frames(f"{frames_dir or LP.FRAMES}/frames_s{fseed}.npz")
    sha = LR.adapter_sha(model)
    if sha != fr["adapter_sha"]:
        raise LR.LadderMismatch(f"{name}: adapter hash {sha[:12]} != cached frames {fr['adapter_sha'][:12]}")
    B = None
    if kind != "adapter":
        B = SH.require_arm_basis(os.path.join(run_dir, "selfcheck.json"), basis_npz)        # raises BasisMismatch: wrong kind or sha, before any scoring
        SH.attach_basis_frames(fr, B, ubasis_npz)
    return dict(model=model, cfg=cfg, argv=argv, ck=ck, built=built, checked=checked, fr=fr, adapter_sha=sha, B=B, kind=kind, frames_seed=fseed)


def prepare_arm_read(name, run_dir, basis_npz=None, ubasis_npz=None, frames_dir=None):
    """lp_lib.prepare_arm for one SLOWWM arm (the ONE path every arm takes, R35H included: for an adapter model the clip_z1 dispatch returns lp_lib's own function, so the control that reads R35H here and
    against ``lp_lib.prepare_arm`` compares this whole route, not a delegation). ``run_dir``: the banked run folder (argv.txt, ckpt_last.pt; selfcheck.json for a basis arm)."""
    L = load_checked(name, run_dir, basis_npz, ubasis_npz, frames_dir)
    ev, tr = P.load_latents(LP.EVAL, "M2b_s0"), P.load_latents(LP.TRAIN, "M2b_s0")
    with basis_clip_z1():
        ctx = LP.build_ctx(L["model"].wm, L["fr"], name, ev, tr, LP.td3_defined(L["built"]))
    ctx.built, ctx.checked, ctx.adapter_sha, ctx.td3_defined = L["built"], L["checked"], L["adapter_sha"], LP.td3_defined(L["built"])
    ctx.basis_kind, ctx.basis_sha, ctx.frames_seed = L["kind"], (L["B"]["sha"] if L["B"] else ""), L["frames_seed"]
    return ctx


def meta_of(ctx):
    """The arm-independent window arrays and the arm's collapse scale, for a report that loads no model."""
    return dict(A=np.asarray(ctx.A), v0=np.asarray(ctx.v0), eid=np.asarray([str(e) for e in ctx.eid]), t=np.asarray(ctx.t), tune_idx=np.asarray(ctx.tune_idx), score_idx=np.asarray(ctx.score_idx),
                hist_ok=np.asarray(ctx.hist_ok), std_raw=np.asarray(ctx.std_raw, np.float64), ch1=np.array(float(ctx.ch1)), basis_kind=np.array(ctx.basis_kind), basis_sha=np.array(ctx.basis_sha))
