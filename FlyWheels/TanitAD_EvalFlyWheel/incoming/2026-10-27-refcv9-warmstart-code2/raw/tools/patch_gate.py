"""launch_gate.py (LF working copy): the refcv9-warm profile, the w_map_fp liveness rule, the admitted-dead rule of the
train-only G-LIVE judgement, and the two refcv9 RED arms. Every anchor must match EXACTLY once."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
groups = Path(sys.argv[2]).read_text(encoding="utf-8").rstrip("\n")
s = p.read_text(encoding="utf-8")
EDITS = []


def edit(old, new):
    EDITS.append((old, new))


PROFILE = '''

# --------------------------------------------------------------------------------------------- #
# refcv9-warm: the refcv9 WARM START (PI decisions 2026-10-27)                                    #
# --------------------------------------------------------------------------------------------- #
#: The PI, 2026-10-27: "warm start from refcv8 EMA-48,852; perception FROZEN; planner + tactical heads TRAINED; the map
#: head ALSO trained, with a precision-targeted loss; inputs unchanged from refcv8; a `refcv9-warm` profile may be added
#: to launch_gate.py; Thor cap 48 h". = refcv8's profile (every refcv8 rule still binds: the architecture, the data, the
#: recipe values) with `--r9-train-only planner,map` REQUIRED, the warm start bound by CONTENT to the shipped EMA model
#: (the map-only route's machinery, `map_only` below), the instruments that read frozen gradients refused, the map
#: precision term and the GT-in-pool schedule (H-R9-ALLOC-ORACLE-1) required, R9-PF (a separate stage) forbidden.
_R9W_FLAG = "--r9-train-only"
#: the trainable sets, as the argv literal (map_only_ft.TRAIN_ONLY_SETS: planner = core.decoder, map = the 3 modules)
_R9W_SETS = "planner,map"
#: the trainable modules -- LITERALS (the trainer's map_only_ft.modules_of_sets(("planner", "map")) is checked against
#: them, never read). `core.decoder` = EXACTLY what `_mapft_inject`'s `mapft_planner_trainable` arm un-freezes
_R9W_MODULES = ("core.decoder", "_map_hires.refine", "_map_hires.near", "_map_hires.near_refine")
#: the trainable leaf groups and their sizes on the REAL refcv8-r101-s0 build, MEASURED 2026-10-27 (CPU,
#: `refcv7_loader.build_model` on ckpt_ema_48852.pt, strict load, G-DVB clean): 111 groups = 99 of core.decoder (186
#: tensors, 15,365,801 values) + the 12 map-only groups of `_MAPFT_GROUPS` (18 tensors, 105,032 values)
_R9W_GROUPS = {
''' + groups + '''}
_R9W_NUMEL = 15_470_833
_R9W_SCHED_FLAG = "--r9-alloc-force-schedule"
_R9W_MAPFP_FLAG = "--w-map-fp"
#: R9-PF is a SEPARATE stage-2 fit (REFCV9_INMODEL_DESIGN s1, S2): never in this run
_R9W_PF_FLAGS = ("--r9-progress-head", "--r9-calib-stage", "--w-r9-prog", "--w-r9-regime", "--r9-decision",
                 "--r9-hidden", "--r9-decel-tau", "--r9-seed")
_R9W_OPEN_ITEMS = (
    {"id": "R9W-TACTICAL",
     "what": "the PI's 2026-10-27 decision says 'planner + TACTICAL HEADS trained'; `--r9-train-only planner,map` trains "
             "core.decoder (+ the map) ONLY. MEASURED on the real refcv8 build: the tactical heads are OUTSIDE "
             "core.decoder and stay FROZEN under this argv -- tac_decoder_v6 (72 tensors, 2,272,802), phi_tac (24, "
             "2,101,504), core.tactical_trunk (2, 786,816), tac_latent_proj (2, 262,656), tac_goal_head (2, 6,156), "
             "lat_head_tac / lon_head_tac (2 + 2, 4,104 each), core.lat_head / core.lon_head (2 + 2, 1,155 each), "
             "tac_behaviour_gate_v6 (1, 2,574), nav_to_tac (2, 33,280). The DRAFT prereg's 'planner + tactical heads = "
             "the arm's set' is therefore not what the arm's set is",
     "owner": "the Master Mind (model design: a `tactical` set as a literal module list in map_only_ft.TRAIN_ONLY_SETS "
              "+ this profile's _R9W_MODULES / _R9W_GROUPS / _R9W_NUMEL, or a PI ruling that the decoder alone is meant)",
     "closes_when": "the trained set matches the PI's wording, or the PI rules that `planner,map` is the intended set"},
    {"id": "R9W-BUDGET",
     "what": "--steps / --warmup / --ema-decay / --eval-every are not fixed: refcv8's full-run literals (48,852 / 1,000 / "
             "0.9998 / 500) are NOT inherited. PI 2026-10-27: Thor cap 48 h; N from the 50-step smoke (this gate's "
             "G-LIVE `cost` row) by the MAPFT rule form, N = min(Nmax, floor(0.85 x (cap - smoke - 1800) / s_step)); "
             "DRAFT prereg: --warmup 200, --lr 1e-4, EMA horizon N/10",
     "owner": "the Master Mind (sizing from the measured s/step; the PI's 48 Thor-hour cap binds)",
     "closes_when": "the literals are written here, or closed open-only as MAPFT-BUDGET was (the token's argv sha256 pins N)"},
    {"id": "R9W-SCHEDULE",
     "what": "the --r9-alloc-force-schedule literal: DRAFT prereg 'forced with probability 1.0 -> 0.0 linearly over the "
             "first half of the run, 0.0 after' = `linear:1.0:0.0:0:<N//2>` (or <warmup> as the start) -- needs N",
     "owner": "the Master Mind / PI", "closes_when": "the literal is written into required_values"},
    {"id": "R9W-MAP-FP-WEIGHT",
     "what": "the --w-map-fp literal: DRAFT prereg s5 'the weight w of the map term fixed from the smoke's loss scales "
             "(stated, not tuned on eval)'; this profile requires it present and positive only",
     "owner": "the Master Mind (from this gate's smoke rows: `map_hires` vs `map_fp`)",
     "closes_when": "the literal is written into required_values"},
    {"id": "R9W-TF-RESTART",
     "what": "refcv8's constraint teacher-forcing schedule RESTARTS on a warm start: tf_ratio(step, steps, 1.0, 0.25) "
             "begins at 1.0 again although refcv8 ended at 0.25 (refc_v3_train.py, the per-step r8 block); decide "
             "--r8-tf-start / --r8-tf-end explicitly (e.g. 0.25 / 0.25 = continue where refcv8 ended)",
     "owner": "the Master Mind", "closes_when": "both literals are in required_values"},
    {"id": "R9W-EMIT-START",
     "what": "the inherited --r8-alloc-emit-start 1000 re-runs refcv8's emission-OFF phase for 1,000 steps on a model "
             "that already emits its allocated extras; decide 0, or keep 1000 deliberately",
     "owner": "the Master Mind", "closes_when": "the literal is decided here"},
)
_r9w_drop = set(_MAPFT_B1) | {"--grad-share-every"}
#: G-LIVE's admitted dead groups, DERIVED FROM THE FROZEN SET: refcv8's rows whose group is INSIDE the trainable modules
#: (a frozen group is not in the optimizer, so neither G-LIVE nor the supervisor's zero-gradient gate reads it) + ONE
#: parameter the supervisor reads by name: core.decoder.r8_rc_gate (zero-init scalar applied only when r8['rc'] is not
#: None -- refc.py `_r8t + self.r8_rc_gate * ...`; --r8-no-rc feeds none). MEASURED on NT A1 (2026-10-26): it was the
#: one offender of the supervisor's gate outside the admitted groups. Its G-LIVE leaf group `core.decoder` is live
#: (navc_gate trains), so the row admits nothing there; zerograd_allow_from_gate.py turns it into the exact allow pattern.
_R9W_LIVE_DEAD = tuple(r for r in _R8["live_dead_admitted"]
                       if any(r[0] == m or r[0].startswith(m + ".") for m in _R9W_MODULES)) + (
    ("core.decoder.r8_rc_gate", "--r8-no-rc",
     "the route-checkpoint bearing gate (zero-init); refc.py applies it only when r8['rc'] is not None, and --r8-no-rc "
     "(PI Q-A 2026-10-11) feeds none -- exactly zero gradient by construction"),)
PROFILES["refcv9-warm"] = dict(
    _R8, name="refcv9-warm",
    open_items=_R9W_OPEN_ITEMS,
    #: G-BOX-OVERFIT is NOT required, for refcv8-mapft's reason (MM 2026-10-26 ~21:20): the box head is PERCEPTION and
    #: FROZEN here, so the check has no subject (and its replay does not apply the declared freeze). What this launch
    #: must prove about the box head is that it does NOT move: G-LIVE's bit identity of every frozen tensor (map_only
    #: below), the RED arms and the trainer's FrozenStateGuard. G-MAP-OVERFIT stays required: the map head trains.
    required_checks=tuple(c for c in _R8["required_checks"] if c != "G-BOX-OVERFIT"),
    required_flags=tuple(f for f in _R8["required_flags"] if f not in _r9w_drop) + (
        _R9W_FLAG, "--conflict-detector", _R9W_MAPFP_FLAG, _R9W_SCHED_FLAG, "--r9-alloc-force-seed"),
    required_flags_why=dict({k: v for k, v in (_R8.get("required_flags_why") or {}).items() if k not in _r9w_drop}, **{
        _R9W_FLAG: "PI 2026-10-27: perception FROZEN; the planner (core.decoder) and the map head trained",
        "--init-from": "PI 2026-10-27: the warm start IS refcv8 EMA-48,852 (md5 bound by map_only below)",
        "--conflict-detector": "the detector reads trunk gradients; the trunk is FROZEN (`off`, required by value)",
        _R9W_MAPFP_FLAG: "PI 2026-10-27: the map head is trained WITH a precision-targeted term (value: R9W-MAP-FP-WEIGHT)",
        _R9W_SCHED_FLAG: "H-R9-ALLOC-ORACLE-1: the predicted-only pool inference sees, in training (value: R9W-SCHEDULE)",
        "--r9-alloc-force-seed": "the schedule's PRIVATE per-(seed, step) coin stream, written as a literal"}),
    required_values=dict({k: v for k, v in _R8["required_values"].items() if k not in set(_MAPFT_B1) | set(_MAPFT_BUDGET)},
                         **{"--conflict-detector": "off", _R9W_FLAG: _R9W_SETS, "--r9-alloc-force-seed": "20261027"}),
    required_values_why=dict({k: v for k, v in (_R8.get("required_values_why") or {}).items()
                              if k not in set(_MAPFT_B1) | set(_MAPFT_BUDGET)}, **{
        "--conflict-detector": "PI 2026-10-27: perception frozen, no trunk gradient exists to measure",
        _R9W_FLAG: "PI 2026-10-27; the sets' modules are map_only_ft.TRAIN_ONLY_SETS (planner = core.decoder)",
        "--r9-alloc-force-seed": "refcv9_alloc_force.DEFAULT_SEED (a literal, not re-derived)"}),
    required_positive=tuple(_R8["required_positive"]) + (_R9W_MAPFP_FLAG,),
    forbidden_levers=tuple(_R8["forbidden_levers"]) + (
        (_MAPFT_B1, "the B1 guard cuts the HEAD group's LR on the EMA box3d AP: the box head is FROZEN (perception) and "
                    "there is no head / encoder group to cut"),
        (_MAPFT_GS, "the gradient-share instrument measures the trunk / bev025 / bev_pool groups, all FROZEN"),
        (("--conflict-every",), "the conflict detector is off under the train-only freeze (a dead cadence)"),
        (("--trunk-bn-recalib",), "BN recalibration REWRITES the frozen trunk's statistics"),
        (("--r8-critic-drivable", "--w-r8-drivable"), "not in the refcv8 build the warm start loads (inputs unchanged)"),
        (_R9W_PF_FLAGS, "R9-PF is the SEPARATE stage-2 calibration fit on the fresh split (REFCV9_INMODEL_DESIGN s1, "
                        "S2), with its own freeze")),
    live_dead_admitted=_R9W_LIVE_DEAD,
    #: ga-reach: the map-only parts must be > 0 (they train); the SHARED map parts must read EXACTLY 0.0 (frozen)
    map_hires=dict(_R8["map_hires"], ga_keys=("ga_mh_refine", "ga_mh_near", "ga_mh_near_refine"),
                   ga_keys_zero=("ga_mh_lift", "ga_mh_encoder", "ga_mh_trunk_s8_stage")),
    #: the train-only literals G-DVB (built set) / G-LIVE (gradients + frozen bit identity) / the argv rules (init md5)
    #: read -- the map-only route's machinery, keyed by THIS profile's flag, modules, groups and numel
    map_only={"flag": _R9W_FLAG, "modules": _R9W_MODULES, "groups": dict(_R9W_GROUPS),
              "numel": _R9W_NUMEL, "init_from_md5": _MAPFT_INIT_MD5},
)
'''

edit('''    #: the map-only literals G-DVB (built set) / G-LIVE (gradients + bit identity) / the argv rules (init md5) read
    map_only={"flag": _MAPFT_FLAG, "modules": _MAPFT_MODULES, "groups": dict(_MAPFT_GROUPS),
              "numel": _MAPFT_NUMEL, "init_from_md5": _MAPFT_INIT_MD5},
)
''', '''    #: the map-only literals G-DVB (built set) / G-LIVE (gradients + bit identity) / the argv rules (init md5) read
    map_only={"flag": _MAPFT_FLAG, "modules": _MAPFT_MODULES, "groups": dict(_MAPFT_GROUPS),
              "numel": _MAPFT_NUMEL, "init_from_md5": _MAPFT_INIT_MD5},
)''' + PROFILE)

# the liveness rule of --w-map-fp
edit('''    "w_r9_regime": (lambda a: bool(getattr(a, "r9_progress_head", False)) and bool(getattr(a, "r9_calib_stage", False))
                    and _w(a, "w_r9_regime") > 0, ("r9_l_regime",), 1.0, "refcv9 R9-PF decision CE"),
}''', '''    "w_r9_regime": (lambda a: bool(getattr(a, "r9_progress_head", False)) and bool(getattr(a, "r9_calib_stage", False))
                    and _w(a, "w_r9_regime") > 0, ("r9_l_regime",), 1.0, "refcv9 R9-PF decision CE"),
    # refcv9 warm start (PI 2026-10-27): the map precision term rides the 10 cm map term and is emitted on EVERY training
    # step once live (an all-unlabelled batch writes the attached zero, `map_precision` / `_map_hires_loss`)
    "w_map_fp": (lambda a: _w(a, "w_map_fp") > 0 and str(getattr(a, "map_hires", "off")) == "on"
                 and _w(a, "w_map_hires") > 0, ("map_fp",), 1.0,
                 "refcv9 map precision: own sidewalk + edge mass on GT-drivable near-field corridor cells"),
}''')

# the train-only G-LIVE judgement honours the profile's admitted dead groups (inert when none is inside its set)
edit('''def judge_map_only_live(prof: dict, rec: dict | None, grads: dict) -> tuple[list[str], dict]:''',
     '''def judge_map_only_live(prof: dict, rec: dict | None, grads: dict,
                        admitted_dead: dict[str, str] | None = None) -> tuple[list[str], dict]:''')
edit('''    dead = [g for g in want if g in have and int(have[g]["nz_params"]) == 0]
    if dead:''', '''    # `admitted_dead` (refcv9-warm): the profile's live_dead_admitted rows whose flag is in the argv -- a TRAINABLE group
    # dead BY CONSTRUCTION under that flag (e.g. core.decoder.ctx_to_cond under --no-strategic) is admitted, with its
    # reason in the evidence, exactly as judge_live admits it; every other dead group still FAILS. refcv8-mapft: none of
    # its admitted groups is inside its map-only modules, so its judgement is unchanged
    adm = {g: w for g, w in (admitted_dead or {}).items() if g in want}
    det["dead_admitted"] = {g: adm[g] for g in want if g in have and int(have[g]["nz_params"]) == 0 and g in adm}
    dead = [g for g in want if g in have and int(have[g]["nz_params"]) == 0 and g not in adm]
    if dead:''')
edit('''        mor_, mod_ = judge_map_only_live(ctx.prof, run1.get("map_only"), gt)''',
     '''        mor_, mod_ = judge_map_only_live(ctx.prof, run1.get("map_only"), gt, admitted_dead=admitted)''')

# the two refcv9 RED arms (the apply_mapft_build_arm idiom): at the BUILD (G-DVB) and live (G-LIVE)
edit('''def profile_argv_rules(prof: dict, argv: list[str], options: dict | None = None,''', '''#: the refcv9-warm RED arms (drafted 2026-10-27; run by the kit, graded on G-DVB / G-LIVE)
REFCV9_ARMS = ("refcv9_perception_trainable", "refcv9_planner_frozen")
#: `refcv9_perception_trainable` un-freezes a whole PERCEPTION subtree, the first that exists in this order: the trunk
#: (every training loss -- the planner's and the map head's -- reaches it), then the planner's pooled BEV. A subtree, not
#: one tensor, for `_mapft_inject`'s reason: one tensor could be one no loss reaches, and an arm that moves nothing at
#: G-LIVE would ESCAPE (G-DVB's built-set read catches any trainable perception tensor either way)
_REFCV9_PERCEPTION_PICK = ("core.encoder.", "_perception.bev_pool.")


def apply_refcv9_build_arm(ctx: Any, pat: "_Patcher", T: Any) -> str | None:
    """The refcv9-warm profile's regression arms at the BUILD (G-DVB): the trainer's `--r9-train-only` freeze leaves ONE
    perception tensor trainable (`refcv9_perception_trainable`) or freezes the planner set too
    (`refcv9_planner_frozen`). Each must be caught by G-DVB (the gate's built-set read + the trainer's `r9_train_only`
    lever) with a reason naming the freeze."""
    if not any(arm_is(ctx, a) for a in REFCV9_ARMS):
        return None
    mof = getattr(T, "_mof", None)
    if mof is None or not callable(getattr(mof, "apply_train_only", None)):
        return "arm requested but the trainer has no map_only_ft.apply_train_only (the arm cannot install)"
    orig = mof.apply_train_only
    mods = tuple((ctx.prof.get("map_only") or {}).get("modules") or ())

    def wrapped(model, *a, _o=orig, **k):
        r = _o(model, *a, **k)
        _refcv9_inject(ctx, model, None, mods or tuple(r.get("modules") or ()))
        return r
    pat.set(mof, "apply_train_only", wrapped)
    return f"{ctx.arm}: injected after the trainer's --r9-train-only freeze"


def _refcv9_inject(ctx: Any, model, opt, mods) -> str | None:
    """The defect itself: a PERCEPTION subtree trainable again (and, with an optimizer, IN it), or every parameter of
    the planner set (`core.decoder`) frozen. Inert for any other arm."""
    if arm_is(ctx, "refcv9_perception_trainable"):
        cands = [(n, p) for n, p in model.named_parameters() if not _under_any(n, mods) and not p.requires_grad]
        pick: list = []
        for pre in _REFCV9_PERCEPTION_PICK:
            pick = [(n, p) for n, p in cands if n.startswith(pre)]
            if pick:
                break
        pick = pick or cands[:1]
        if not pick:
            return None
        for _n, p in pick:
            p.requires_grad_(True)
            if opt is not None:
                opt.param_groups[0]["params"].append(p)
        return (f"refcv9_perception_trainable: {len(pick)} perception tensor(s) un-frozen (first {pick[0][0]})"
                + (" and added to the optimizer" if opt is not None else "") + " -- the --r9-train-only freeze broken")
    if arm_is(ctx, "refcv9_planner_frozen"):
        k = 0
        for n, p in model.named_parameters():
            if _under_any(n, ("core.decoder",)):
                p.requires_grad_(False)
                k += 1
        return f"refcv9_planner_frozen: {k} planner (core.decoder) tensors frozen (the --r9-train-only planner set)"
    return None


def profile_argv_rules(prof: dict, argv: list[str], options: dict | None = None,''')
edit('''    _mapft_arm = apply_mapft_build_arm(ctx, pat, T)        # refcv8-mapft's arms act at the trainer's freeze
    if _mapft_arm:
        _say("arm: " + _mapft_arm)
''', '''    _mapft_arm = apply_mapft_build_arm(ctx, pat, T)        # refcv8-mapft's arms act at the trainer's freeze
    if _mapft_arm:
        _say("arm: " + _mapft_arm)
    _r9w_arm = apply_refcv9_build_arm(ctx, pat, T)         # refcv9-warm's arms act at the trainer's --r9-train-only freeze
    if _r9w_arm:
        _say("arm: " + _r9w_arm)
''')
edit('''            note = _mapft_inject(ctx, state["model"], state["opt"], mo_mods)
''', '''            note = (_mapft_inject(ctx, state["model"], state["opt"], mo_mods)
                    or _refcv9_inject(ctx, state["model"], state["opt"], mo_mods))     # refcv9-warm's LIVE arms
''')

for old, new in EDITS:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"anchor matched {n} times:\n{old[:300]}")
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("applied", len(EDITS), "edits")
