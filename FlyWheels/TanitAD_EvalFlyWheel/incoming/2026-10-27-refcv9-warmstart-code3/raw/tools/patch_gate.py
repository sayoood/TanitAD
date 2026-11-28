"""launch_gate.py (LF working copy), round 3 (MM 2026-10-27): refcv9-warm trains planner,tactical,map; R9W-TACTICAL /
-EMIT-START / -TF-RESTART / -SCHEDULE closed as rules; the declared grad-unreachable subtrees inside the sets read as
FROZEN by G-DVB / G-LIVE; refcv8 / refcv8-nt / refcv8-mapft / v8f refuse --r9-train-only."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
groups = Path(sys.argv[2]).read_text(encoding="utf-8").rstrip("\n")
s = p.read_text(encoding="utf-8")
E = []


def edit(old, new):
    E.append((old, new))


# ---- 6. refcv8 (and every profile built from its forbidden levers: refcv8-nt, refcv8-mapft, v8f, v8f_steer) ------ #
edit('''        (("--r8-train-only-map",), "the map-only fine-tune (PI D-R9-MAPHEAD-PRECISION) -- launched under profile "
                                   "refcv8-mapft only"),
    ),''', '''        (("--r8-train-only-map",), "the map-only fine-tune (PI D-R9-MAPHEAD-PRECISION) -- launched under profile "
                                   "refcv8-mapft only"),
        # MM 2026-10-27 (refcv9 round 3): the refcv9 warm-start freeze launches ONLY under its own profile `refcv9-warm`,
        # whose G-DVB / G-LIVE read the freeze (this profile's G-LIVE would not); inherited by refcv8-nt, refcv8-mapft
        # and v8f, which take this tuple
        (("--r9-train-only",), "the refcv9 warm-start freeze (PI 2026-10-27) -- launched under profile refcv9-warm "
                               "only"),
    ),''')

# ---- the profile, rewritten ------------------------------------------------------------------------------------- #
a = s.index("# --------------------------------------------------------------------------------------------- #\n"
            "# refcv9-warm: the refcv9 WARM START")
b = s.index("\n\n\nclass GateError")
PROFILE = '''# --------------------------------------------------------------------------------------------- #
# refcv9-warm: the refcv9 WARM START (PI decisions 2026-10-27)                                    #
# --------------------------------------------------------------------------------------------- #
#: The PI, 2026-10-27: "warm start from refcv8 EMA-48,852; perception FROZEN; planner + tactical heads TRAINED; the map
#: head ALSO trained, with a precision-targeted loss; inputs unchanged from refcv8; a `refcv9-warm` profile may be added
#: to launch_gate.py; Thor cap 48 h". = refcv8's profile (every refcv8 rule still binds: the architecture, the data, the
#: recipe values) with `--r9-train-only planner,tactical,map` REQUIRED, the warm start bound by CONTENT to the shipped
#: EMA model (the map-only route's machinery, `map_only` below), the instruments that read frozen gradients refused,
#: the map precision term and the GT-in-pool schedule (H-R9-ALLOC-ORACLE-1) required, R9-PF (a separate stage)
#: forbidden. MM 2026-10-27 (round 3): R9W-TACTICAL / -EMIT-START / -TF-RESTART / -SCHEDULE CLOSED as rules below.
_R9W_FLAG = "--r9-train-only"
#: the trainable sets, as the argv literal (map_only_ft.TRAIN_ONLY_SETS, canonical order)
_R9W_SETS = "planner,tactical,map"
#: the trainable modules -- LITERALS (the trainer's map_only_ft.modules_of_sets(...) is checked against them, never
#: read): planner = core.decoder (what `_mapft_inject`'s `mapft_planner_trainable` arm un-freezes); tactical = the
#: eleven tactical-branch modules (map_only_ft's docstring says why each is tactical and which neighbours are not);
#: map = the three map-only modules
_R9W_TACTICAL = ("phi_tac", "nav_to_tac", "lat_head_tac", "lon_head_tac", "tac_goal_head", "tac_latent_proj",
                 "tac_decoder_v6", "tac_behaviour_gate_v6", "core.tactical_trunk", "core.lat_head", "core.lon_head")
_R9W_MODULES = ("core.decoder",) + _R9W_TACTICAL + ("_map_hires.refine", "_map_hires.near", "_map_hires.near_refine")
#: the DECLARED grad-unreachable subtrees INSIDE the sets on the refcv8 build (tanitad/models/_gradreach.py): FROZEN by
#: the build, frozen by the freeze, out of the optimizer -- MEASURED 2026-10-27 on the REAL build (8 tensors, 111,710
#: values): the sampler build's offset_head / control_head, --no-strategic's ctx_to_cond, --graft-tac8-prior's
#: lat_to_anchor / lon_to_anchor. G-DVB holds the built model's declarations inside the sets EQUAL to this literal
_R9W_DECLARED_FROZEN = ("core.decoder.control_head", "core.decoder.ctx_to_cond", "core.decoder.lat_to_anchor",
                        "core.decoder.lon_to_anchor", "core.decoder.offset_head")
#: the TRAINED leaf groups and their sizes on the REAL refcv8-r101-s0 build, MEASURED 2026-10-27 (CPU,
#: `refcv7_loader.build_model` on ckpt_ema_48852.pt, strict load, G-DVB clean; F:/refcv9_code3 real_r8_probe_tip.json):
#: 164 groups = 94 of core.decoder (178 tensors, 15,254,091 values; the 5 declared subtrees excluded) + 58 tactical
#: (113 tensors, 5,476,306) + the 12 map-only groups of `_MAPFT_GROUPS` (18 tensors, 105,032)
_R9W_GROUPS = {
''' + groups + '''}
_R9W_NUMEL = 20_835_429
_R9W_SCHED_FLAG = "--r9-alloc-force-schedule"
_R9W_MAPFP_FLAG = "--w-map-fp"
#: R9W-SCHEDULE (MM 2026-10-27): forced with p 1.0 -> 0.0 linearly over the FIRST HALF of the run, 0.0 after; the end
#: step is floor(--steps / 2) of THIS argv (`r9_schedule_reasons`)
_R9W_SCHED_RULE = {"flag": _R9W_SCHED_FLAG, "kind": "linear", "p0": 1.0, "p1": 0.0, "start": 0,
                   "end_of": "--steps", "end_div": 2}
#: R9W-TF-RESTART (MM 2026-10-27): refcv8's GT-constraint teacher-forcing ratio is HELD at its end-of-run value.
#: refcv8-r101-s0's config.json: --r8-tf-start 1.0 --r8-tf-end 0.25 --steps 48852, and refcv8_train.tf_ratio is linear
#: over steps 0 .. steps-1, so the ratio of its LAST step (48,851) is exactly 0.25; start = end = 0.25 holds it there
_R9W_TF_HOLD = "0.25"
#: R9-PF is a SEPARATE stage-2 fit (REFCV9_INMODEL_DESIGN s1, S2): never in this run
_R9W_PF_FLAGS = ("--r9-progress-head", "--r9-calib-stage", "--w-r9-prog", "--w-r9-regime", "--r9-decision",
                 "--r9-hidden", "--r9-decel-tau", "--r9-seed")
_R9W_OPEN_ITEMS = (
    {"id": "R9W-BUDGET",
     "what": "--steps / --warmup / --ema-decay / --eval-every are not fixed: refcv8's full-run literals (48,852 / 1,000 / "
             "0.9998 / 500) are NOT inherited. PI 2026-10-27: Thor cap 48 h; N from the 50-step smoke (this gate's "
             "G-LIVE `cost` row) by the MAPFT rule form, N = min(Nmax, floor(0.85 x (cap - smoke - 1800) / s_step)); "
             "DRAFT prereg: --warmup 200, --lr 1e-4, EMA horizon N/10. The schedule rule then reads its end step "
             "floor(N / 2) off the same --steps",
     "owner": "the operator, from the smoke (the PI's 48 Thor-hour cap binds)",
     "closes_when": "the literals are written here, or closed open-only as MAPFT-BUDGET was (the token's argv sha256 pins N)"},
    {"id": "R9W-MAP-FP-WEIGHT",
     "what": "the --w-map-fp literal, fixed from the smoke's loss scales (stated, not tuned on eval) by the MM's RULE "
             "(2026-10-27): w = 0.1 x map_hires / map_fp at the smoke's FIRST LOGGED step, rounded to ONE significant "
             "figure; this profile requires it present and positive until then",
     "owner": "the operator, from this gate's smoke rows (`map_hires`, `map_fp`)",
     "closes_when": "the literal is written into required_values"},
)
_r9w_drop = set(_MAPFT_B1) | {"--grad-share-every"}
#: G-LIVE's admitted dead groups, RE-DERIVED for planner,tactical,map (MM 2026-10-27): refcv8's rows whose group is
#: TRAINED here -- inside the sets and NOT a declared subtree (a declared group is frozen and out of the optimizer, so
#: neither G-LIVE nor the supervisor's zero-gradient gate reads it: ctx_to_cond / lat_to_anchor / lon_to_anchor drop
#: out) -- + ONE parameter the supervisor reads by name: core.decoder.r8_rc_gate (zero-init scalar applied only when
#: r8['rc'] is not None; --r8-no-rc feeds none; MEASURED on NT A1 2026-10-26 as the one offender outside the admitted
#: groups; its leaf group `core.decoder` is live through navc_gate). refcv8 admits NO tactical group: none is dead.
_R9W_LIVE_DEAD = tuple(r for r in _R8["live_dead_admitted"]
                       if any(r[0] == m or r[0].startswith(m + ".") for m in _R9W_MODULES)
                       and not any(r[0] == d or r[0].startswith(d + ".") for d in _R9W_DECLARED_FROZEN)) + (
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
        _R9W_FLAG, "--conflict-detector", _R9W_MAPFP_FLAG, _R9W_SCHED_FLAG, "--r9-alloc-force-seed",
        "--r8-alloc-emit-start", "--r8-tf-start", "--r8-tf-end", "--steps"),
    required_flags_why=dict({k: v for k, v in (_R8.get("required_flags_why") or {}).items() if k not in _r9w_drop}, **{
        _R9W_FLAG: "PI 2026-10-27: perception FROZEN; the planner (core.decoder), the tactical heads and the map head "
                   "trained",
        "--init-from": "PI 2026-10-27: the warm start IS refcv8 EMA-48,852 (md5 bound by map_only below)",
        "--conflict-detector": "the detector reads trunk gradients; the trunk is FROZEN (`off`, required by value)",
        _R9W_MAPFP_FLAG: "PI 2026-10-27: the map head is trained WITH a precision-targeted term (value: R9W-MAP-FP-WEIGHT)",
        _R9W_SCHED_FLAG: "H-R9-ALLOC-ORACLE-1: the predicted-only pool inference sees, in training (form: "
                         "r9_schedule_reasons)",
        "--r9-alloc-force-seed": "the schedule's PRIVATE per-(seed, step) coin stream, written as a literal",
        "--steps": "the schedule's end step is floor(--steps / 2) (value: R9W-BUDGET)"}),
    required_values=dict({k: v for k, v in _R8["required_values"].items() if k not in set(_MAPFT_B1) | set(_MAPFT_BUDGET)},
                         **{"--conflict-detector": "off", _R9W_FLAG: _R9W_SETS, "--r9-alloc-force-seed": "20261027",
                            "--r8-alloc-emit-start": "0", "--r8-tf-start": _R9W_TF_HOLD,
                            "--r8-tf-end": _R9W_TF_HOLD}),
    required_values_why=dict({k: v for k, v in (_R8.get("required_values_why") or {}).items()
                              if k not in set(_MAPFT_B1) | set(_MAPFT_BUDGET)}, **{
        "--conflict-detector": "PI 2026-10-27: perception frozen, no trunk gradient exists to measure",
        _R9W_FLAG: "PI 2026-10-27 + R9W-TACTICAL closed (MM 2026-10-27): planner + tactical heads + map head trained; "
                   "the sets' modules are map_only_ft.TRAIN_ONLY_SETS",
        "--r9-alloc-force-seed": "refcv9_alloc_force.DEFAULT_SEED (a literal, not re-derived)",
        "--r8-alloc-emit-start": "R9W-EMIT-START closed (MM 2026-10-27): refcv8 emitted its allocated candidates from "
                                 "step 1,000 to its end, so the warm start emits them from step 0 -- a start >= 1 "
                                 "would HIDE them; the trainer admits 0 under --r9-train-only only",
        "--r8-tf-start": "R9W-TF-RESTART closed (MM 2026-10-27): refcv8's ratio at its last step, tf_ratio(48851, 48852, "
                         "1.0, 0.25) = 0.25 exactly, HELD (start = end)",
        "--r8-tf-end": "R9W-TF-RESTART closed: start = end = 0.25 is the constant ratio refcv8 ended at"}),
    required_positive=tuple(_R8["required_positive"]) + (_R9W_MAPFP_FLAG,),
    #: refcv8's own refusal of --r9-train-only is the one entry NOT inherited (this IS that profile)
    forbidden_levers=tuple(e for e in _R8["forbidden_levers"] if _R9W_FLAG not in e[0]) + (
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
    #: the train-only literals G-DVB (built set + declared subtrees) / G-LIVE (gradients + frozen bit identity) / the
    #: argv rules (init md5) read -- the map-only route's machinery, keyed by THIS profile's literals
    map_only={"flag": _R9W_FLAG, "modules": _R9W_MODULES, "groups": dict(_R9W_GROUPS),
              "numel": _R9W_NUMEL, "init_from_md5": _MAPFT_INIT_MD5, "declared_frozen": _R9W_DECLARED_FROZEN},
    #: R9W-SCHEDULE (MM 2026-10-27): the schedule's literal form, its end step read off THIS argv's --steps
    alloc_force_rule=dict(_R9W_SCHED_RULE),
)'''
s = s[:a] + PROFILE + s[b:]

# ---- G-DVB: the declared subtrees inside the modules are frozen BY DECLARATION, and equal to the profile's literal - #
edit('''    mods = tuple(mo["modules"])
    tr = [(n, int(p.numel())) for n, p in model.named_parameters() if p.requires_grad]
    groups: dict[str, int] = {}
    for n, k in tr:
        g = n.rsplit(".", 1)[0] if "." in n else n
        groups[g] = groups.get(g, 0) + k
    outside = [n for n, _ in tr if not _under_any(n, mods)]
    frozen_map = [n for n, p in model.named_parameters() if _under_any(n, mods) and not p.requires_grad]''', '''    mods = tuple(mo["modules"])
    decl = tuple(mo.get("declared_frozen") or ())
    tr = [(n, int(p.numel())) for n, p in model.named_parameters() if p.requires_grad]
    groups: dict[str, int] = {}
    for n, k in tr:
        g = n.rsplit(".", 1)[0] if "." in n else n
        groups[g] = groups.get(g, 0) + k
    outside = [n for n, _ in tr if not _under_any(n, mods)]
    frozen_map = [n for n, p in model.named_parameters()
                  if _under_any(n, mods) and not _under_any(n, decl) and not p.requires_grad]
    # the build's OWN declarations inside the modules (the module attribute `_gradreach.GRAD_UNREACHABLE_FLAG`, read by
    # name: this check is independent of the trainer's map_only_ft) must be EXACTLY the profile's literal, and frozen
    built_decl = sorted(n for n, m in model.named_modules()
                        if getattr(m, _GRAD_UNREACHABLE_ATTR, None) and _under_any(n, mods))
    undead = [n for n, p in model.named_parameters() if _under_any(n, decl) and p.requires_grad]''')
edit('''    det = {"on": True, "trainable_numel": int(sum(k for _, k in tr)), "trainable_groups": groups,
           "outside_sample": outside[:5], "n_outside": len(outside), "frozen_map_only_sample": frozen_map[:5]}
    reasons = []
    if not has_flag(argv, mo["flag"]):
        reasons.append(f"MAP-ONLY: {mo['flag']} is not in the argv")''', '''    det = {"on": True, "trainable_numel": int(sum(k for _, k in tr)), "trainable_groups": groups,
           "outside_sample": outside[:5], "n_outside": len(outside), "frozen_map_only_sample": frozen_map[:5],
           "declared_frozen_built": built_decl}
    reasons = []
    if not has_flag(argv, mo["flag"]):
        reasons.append(f"MAP-ONLY: {mo['flag']} is not in the argv")
    if built_decl != sorted(decl):
        reasons.append(f"MAP-ONLY: the build's declared grad-unreachable subtrees inside the trained modules are "
                       f"{built_decl}; the profile's literal is {sorted(decl)} -- a declaration would hide a trained "
                       f"group, or a dead one would be trained")
    if undead:
        reasons.append(f"MAP-ONLY: {len(undead)} DECLARED grad-unreachable parameter(s) are trainable again, e.g. "
                       f"{undead[:3]}")''')
edit('''def map_only_built_reasons(prof: dict, argv: list[str], model) -> tuple[list[str], dict]:''',
     '''#: `tanitad.models._gradreach.GRAD_UNREACHABLE_FLAG`, read BY NAME (a test holds the two equal)
_GRAD_UNREACHABLE_ATTR = "_tanitad_grad_unreachable"


def map_only_built_reasons(prof: dict, argv: list[str], model) -> tuple[list[str], dict]:''')

# ---- G-LIVE: the declared subtrees inside the modules are tracked and snapshotted as FROZEN ----------------------- #
edit('''    mo_mods = tuple((ctx.prof.get("map_only") or {}).get("modules") or ())
    if mo_mods:''', '''    mo_mods = tuple((ctx.prof.get("map_only") or {}).get("modules") or ())
    # refcv9-warm: the declared grad-unreachable subtrees INSIDE the modules are frozen too (none on refcv8-mapft)
    mo_decl = tuple((ctx.prof.get("map_only") or {}).get("declared_frozen") or ())

    def mo_is_frozen(n_):
        return not _under_any(n_, mo_mods) or _under_any(n_, mo_decl)
    if mo_mods:''')
edit('''            state["mo_frozen"] = [(n_, p_) for n_, p_ in model.named_parameters() if not _under_any(n_, mo_mods)]''',
     '''            state["mo_frozen"] = [(n_, p_) for n_, p_ in model.named_parameters() if mo_is_frozen(n_)]''')
edit('''            state["mo_snap"] = {k_: tensor_digest(v_) for k_, v_ in model.state_dict().items()
                                if not _under_any(k_, mo_mods)}''', '''            state["mo_snap"] = {k_: tensor_digest(v_) for k_, v_ in model.state_dict().items()
                                if mo_is_frozen(k_)}''')
edit('''        now = {k_: tensor_digest(v_) for k_, v_ in state["model"].state_dict().items()
               if not _under_any(k_, mo_mods)}''', '''        now = {k_: tensor_digest(v_) for k_, v_ in state["model"].state_dict().items()
               if mo_is_frozen(k_)}''')

# ---- R9W-SCHEDULE: the argv rule ---------------------------------------------------------------------------------- #
edit('''    refusals += map_only_init_reasons(prof, argv, pmap)[0]''', '''    refusals += map_only_init_reasons(prof, argv, pmap)[0]
    refusals += r9_schedule_reasons(prof, argv)[0]''')
edit('''def open_item_reasons(prof: dict) -> list[str]:''', '''def r9_schedule_reasons(prof: dict, argv: list[str]) -> tuple[list[str], dict]:
    """R9W-SCHEDULE (MM 2026-10-27): the profile's `alloc_force_rule` -- the GT-in-pool schedule is
    `linear:<p0>:<p1>:<start>:<end>` with the profile's p0 / p1 / start and end = floor(<end_of> / end_div) of THIS argv
    (refcv9-warm: linear:1.0:0.0:0:floor(--steps / 2)). Numbers compared as numbers, the form exactly. Inert without
    the rule."""
    ru = prof.get("alloc_force_rule")
    if not ru:
        return [], {"on": False}
    f = ru["flag"]
    det: dict[str, Any] = {"on": True, "rule": dict(ru)}
    v, n = flag_values(argv, f), flag_values(argv, ru["end_of"])
    if not v or not n:
        return [f"R9W-SCHEDULE: {f} and {ru['end_of']} are both required (the end step is "
                f"floor({ru['end_of']} / {ru['end_div']}))"], det
    try:
        end = int(n[0]) // int(ru["end_div"])
    except ValueError:
        return [f"R9W-SCHEDULE: {ru['end_of']} {n[0]!r} is not an integer"], det
    want = f"{ru['kind']}:{ru['p0']}:{ru['p1']}:{ru['start']}:{end}"
    det.update(value=v[0], want=want, end=end)
    parts = str(v[0]).split(":")
    ok = len(parts) == 5 and parts[0] == ru["kind"]
    if ok:
        try:
            ok = (float(parts[1]) == float(ru["p0"]) and float(parts[2]) == float(ru["p1"])
                  and parts[3].lstrip("-").isdigit() and int(parts[3]) == int(ru["start"])
                  and parts[4].lstrip("-").isdigit() and int(parts[4]) == end)
        except ValueError:
            ok = False
    det["ok"] = bool(ok)
    if not ok:
        return [f"R9W-SCHEDULE: {f} {v[0]!r} is not the rule's {want!r} (p {ru['p0']} -> {ru['p1']} from step "
                f"{ru['start']} to floor({ru['end_of']} {n[0]} / {ru['end_div']}) = {end})"], det
    return [], det


def open_item_reasons(prof: dict) -> list[str]:''')

for old, new in E:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"anchor matched {n} times:\n{old[:400]}")
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("applied profile + ", len(E), "edits")
