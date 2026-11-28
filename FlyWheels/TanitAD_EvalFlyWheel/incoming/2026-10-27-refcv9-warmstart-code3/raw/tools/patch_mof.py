"""map_only_ft.py (LF working copy), round 3: the `tactical` set, and the declared grad-unreachable subtrees INSIDE a
trainable set stay frozen (the landed `--r9-train-only` re-enabled them -- check_grad_unreachable's "declared frozen
but trainable again"). Every anchor must match exactly once."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
E = []


def edit(old, new):
    E.append((old, new))


edit('''    "Perception" = every parameter outside the chosen sets.
    ⚠️ STATED, NOT HIDDEN: ``core.decoder`` does NOT hold the TACTICAL heads (``tac_decoder_v6``, ``phi_tac``,
    ``lat_head_tac`` / ``lon_head_tac``, ``tac_goal_head``, ``tac_latent_proj``, ``core.tactical_trunk``,
    ``core.lat_head`` / ``core.lon_head``, ...). Under ``planner,map`` they are FROZEN together with the perception.
    Training them needs a THIRD named set -- a model-design decision, not a default taken here.
"""''', '''      * ``tactical`` = :data:`TACTICAL_MODULES` (PI 2026-10-27, R9W-TACTICAL closed: "planner + tactical heads
                      trained"). ``core.decoder`` does NOT hold the tactical heads; these eleven modules are the
                      tactical branch -- each one's only consumer is the tactical state / decision / goal path
                      (``refc_v3.RefCV3Model``, ``refc.RefCModel``):
                        - ``phi_tac``               E5: the tactical state z_tac (PhiTac over the pooled sequence);
                        - ``nav_to_tac``            E13: nav -> z_tac (zero-init projection of the nav embedding);
                        - ``lat_head_tac`` / ``lon_head_tac``  E6: the factored tactical decision heads on z_tac;
                        - ``tac_goal_head``         E8: the tactical geometric goals off z_tac;
                        - ``tac_latent_proj``       E7: z_tac -> the decoder's target-latent FiLM port;
                        - ``tac_decoder_v6``        refcv6 s4: the DETR behaviour decoder (22 tactical tokens);
                        - ``tac_behaviour_gate_v6`` refcv6: the valid-behaviour set -> selection gate;
                        - ``core.tactical_trunk`` / ``core.lat_head`` / ``core.lon_head``  the core's lat / lon
                          manoeuvre heads (the tactical class decision; their posterior feeds the decoder's priors).
                      NOT tactical, and FROZEN under every set: ``nav_inj`` (the nav embedding, SHARED with the
                      strategic ``nav_to_str``), ``gstr_embed`` / ``gstr_film`` (the STRATEGIC goal's FiLM on z_tac),
                      ``str_goal_head`` / ``core.strategic`` / ``core.route_head`` (strategic), ``scorer`` /
                      ``goal_gate`` (E9: OPERATIVE selection over the decoder's fan -- the tactical goal enters it
                      DETACHED), ``core.law_head`` (the planner's LAW aux), ``core.measurement`` / ``core.ego_hist``
                      (ego input encoders), ``core.agent_*`` / ``_perception.*`` / ``core.encoder`` / the shared map
                      parts (perception).
    "Perception" = every parameter outside the chosen sets.
    ⛔ A DECLARED grad-unreachable subtree INSIDE a set stays FROZEN (``tanitad/models/_gradreach.py``): the build
    froze it because no loss can reach it (e.g. ``core.decoder.offset_head`` / ``control_head`` on a sampler build,
    ``core.decoder.ctx_to_cond`` under ``--no-strategic``, ``lat/lon_to_anchor`` under ``--graft-tac8-prior``), and
    re-enabling it is the stage-freeze defect ``declared_vs_built.check_grad_unreachable`` refuses ("declared frozen
    but trainable again"). MEASURED on the REAL refcv8 build: 5 such subtrees under ``core.decoder``, 8 tensors /
    111,710 values -- so ``planner`` TRAINS 178 tensors / 15,254,091 values (the gate arm's 186 counts the declared
    ones too); ``tactical`` 113 / 5,476,306 (none declared); ``planner,tactical,map`` 309 / 20,835,429. They are
    recorded (``declared_frozen``) and guarded as frozen: the optimizer, the EMA, the source check and
    :class:`FrozenStateGuard` all treat them as frozen.
"""''')

edit('''           "REFCV8_PLANNER_TENSORS", "REFCV8_PLANNER_NUMEL", "REFCV8_MAP_ONLY_TENSORS",
           "is_r8", "is_r9", "r9_spec", "r9_sets", "parse_sets", "modules_of_sets", "trainable_modules",
           "is_trainable_under", "ema_exclude_for", "apply_train_only", "freeze_tag", "dvb_check_r9"]''',
     '''           "REFCV8_PLANNER_TENSORS", "REFCV8_PLANNER_NUMEL", "REFCV8_MAP_ONLY_TENSORS",
           "is_r8", "is_r9", "r9_spec", "r9_sets", "parse_sets", "modules_of_sets", "trainable_modules",
           "is_trainable_under", "ema_exclude_for", "apply_train_only", "freeze_tag", "dvb_check_r9",
           "TACTICAL_MODULES", "REFCV8_TACTICAL_TENSORS", "REFCV8_TACTICAL_NUMEL", "REFCV8_PLANNER_DECLARED_FROZEN",
           "declared_prefixes", "is_trained"]''')

edit('''#: the named sets, in their CANONICAL order (a spec is normalised to it: `map,planner` == `planner,map`)
TRAIN_ONLY_SETS: dict = {"planner": PLANNER_MODULES, "map": MAP_ONLY_MODULES}
#: refcv8-r101-s0 (48,852 config) -- MEASURED 2026-10-27 on the REAL build (the module docstring)
REFCV8_PLANNER_TENSORS = 186
REFCV8_PLANNER_NUMEL = 15_365_801
REFCV8_MAP_ONLY_TENSORS = 18''', '''#: the tactical branch (PI 2026-10-27, R9W-TACTICAL: "planner + tactical heads trained") -- the module docstring says
#: why each one is tactical and which neighbours are NOT
TACTICAL_MODULES: tuple = ("phi_tac", "nav_to_tac", "lat_head_tac", "lon_head_tac", "tac_goal_head", "tac_latent_proj",
                           "tac_decoder_v6", "tac_behaviour_gate_v6", "core.tactical_trunk", "core.lat_head",
                           "core.lon_head")
#: the named sets, in their CANONICAL order (a spec is normalised to it: `map,tactical,planner` == `planner,tactical,map`)
TRAIN_ONLY_SETS: dict = {"planner": PLANNER_MODULES, "tactical": TACTICAL_MODULES, "map": MAP_ONLY_MODULES}
#: refcv8-r101-s0 (48,852 config) -- MEASURED 2026-10-27 on the REAL build (the module docstring): the TRAINED
#: tensors / values of each set (a declared grad-unreachable subtree inside a set stays frozen)
REFCV8_PLANNER_TENSORS = 178
REFCV8_PLANNER_NUMEL = 15_254_091
REFCV8_MAP_ONLY_TENSORS = 18
REFCV8_TACTICAL_TENSORS = 113
REFCV8_TACTICAL_NUMEL = 5_476_306
#: the declared grad-unreachable subtrees INSIDE `planner` on the refcv8 build (8 tensors, 111,710 values)
REFCV8_PLANNER_DECLARED_FROZEN: tuple = ("core.decoder.offset_head", "core.decoder.ctx_to_cond",
                                         "core.decoder.lat_to_anchor", "core.decoder.lon_to_anchor",
                                         "core.decoder.control_head")''')

edit('''def ema_exclude_for(modules: Iterable[str]) -> Callable:
    """``WeightEMA(exclude=...)`` for a trainable module set: every tensor OUTSIDE it is frozen. The map-only set
    returns :func:`ema_exclude` ITSELF (``--r8-train-only-map``'s EMA is the tip's, object for object)."""
    mods = tuple(modules)
    if mods == MAP_ONLY_MODULES:
        return ema_exclude
    return lambda name, _m=mods: not is_trainable_under(name, _m)''', '''def is_trained(name: str, modules: Iterable[str], declared: Iterable[str] = ()) -> bool:
    """True for a tensor the freeze TRAINS: inside one of ``modules`` and NOT inside a declared grad-unreachable
    subtree (``declared``: dot-boundary prefixes)."""
    return is_trainable_under(name, modules) and not is_trainable_under(name, tuple(declared))


def declared_prefixes(model) -> dict:
    """``{module path: reason}`` of every subtree the BUILT model declared grad-unreachable
    (``tanitad/models/_gradreach.py``)."""
    from tanitad.models._gradreach import grad_unreachable_prefixes
    return grad_unreachable_prefixes(model) if callable(getattr(model, "named_modules", None)) else {}


def ema_exclude_for(modules: Iterable[str], declared: Iterable[str] = ()) -> Callable:
    """``WeightEMA(exclude=...)`` for a trainable module set: every tensor OUTSIDE it, or inside a declared
    grad-unreachable subtree, is frozen. The map-only set with nothing declared returns :func:`ema_exclude` ITSELF
    (``--r8-train-only-map``'s EMA is the tip's, object for object)."""
    mods, dec = tuple(modules), tuple(declared)
    if mods == MAP_ONLY_MODULES and not dec:
        return ema_exclude
    return lambda name, _m=mods, _d=dec: not is_trained(name, _m, _d)''')

# apply_train_only: the declared subtrees inside the sets stay frozen, and are recorded
edit('''    tied = [n for n, p in model.named_parameters() if len(owners[id(p)]) > 1]
    if tied:
        raise MapOnlyError(f"[r9-train-only] ⛔ {R9_FLAG}: {len(tied)} parameter(s) are TIED across the trainable / "
                           f"frozen boundary, e.g. {tied[:3]} -- training them would move the frozen side")
    trainable, n_frozen, numel_frozen = [], 0, 0
    per_set = {s: {"modules": list(TRAIN_ONLY_SETS[s]), "n_tensors": 0, "n_params": 0} for s in sets}
    for n, p in model.named_parameters():
        want = is_trainable_under(n, mods)''', '''    tied = [n for n, p in model.named_parameters() if len(owners[id(p)]) > 1]
    if tied:
        raise MapOnlyError(f"[r9-train-only] ⛔ {R9_FLAG}: {len(tied)} parameter(s) are TIED across the trainable / "
                           f"frozen boundary, e.g. {tied[:3]} -- training them would move the frozen side")
    # a DECLARED grad-unreachable subtree inside a set stays frozen (the build froze it: no loss reaches it)
    decl_all = declared_prefixes(model)
    decl = {k: v for k, v in decl_all.items() if is_trainable_under(k, mods)}
    dec = tuple(decl)
    n_decl = sum(1 for n, _ in model.named_parameters() if is_trainable_under(n, dec))
    numel_decl = sum(int(q.numel()) for n, q in model.named_parameters() if is_trainable_under(n, dec))
    trainable, n_frozen, numel_frozen = [], 0, 0
    per_set = {s: {"modules": list(TRAIN_ONLY_SETS[s]), "n_tensors": 0, "n_params": 0} for s in sets}
    for n, p in model.named_parameters():
        want = is_trained(n, mods, dec)''')
edit('''    pinned = pin_frozen_norms(model, mods)
    rec = {"schema": R9_SCHEMA, "flag": R9_FLAG, "sets": list(sets), "modules": list(mods),
           "trainable": [n for n, _ in trainable],''', '''    pinned = pin_frozen_norms(model, mods)
    rec = {"schema": R9_SCHEMA, "flag": R9_FLAG, "sets": list(sets), "modules": list(mods),
           "declared_frozen": dict(sorted(decl.items())),
           "n_declared_frozen_tensors": int(n_decl), "n_declared_frozen_params": int(numel_decl),
           "trainable": [n for n, _ in trainable],''')
edit('''           "ema": "shadows the trainable tensors only (WeightEMA exclude=ema_exclude_for(modules))"}
    setattr(model, MARKER_ATTR, rec)
    return rec''', '''           "ema": "shadows the trainable tensors only (WeightEMA exclude=ema_exclude_for(modules, declared_frozen))"}
    setattr(model, MARKER_ATTR, rec)
    return rec''')

# the optimizer
edit('''    mods = tuple(rec.get("modules") or ())
    params = [p for _n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    stray = [n for n in names if not is_trainable_under(n, mods)]
    if stray or not params or not mods:''', '''    mods = tuple(rec.get("modules") or ())
    dec = tuple(rec.get("declared_frozen") or ())
    params = [p for _n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    stray = [n for n in names if not is_trained(n, mods, dec)]
    if stray or not params or not mods:''')
edit('''    mods = tuple(rec.get("modules") or ())
    bad = []
    if type(opt).__name__ != "AdamW":
        bad.append(f"optimizer is {type(opt).__name__}, the recipe is AdamW")
    if [g.get("name") for g in opt.param_groups] != [R9_GROUP_NAME]:
        bad.append(f"param groups {[g.get('name') for g in opt.param_groups]} != ['{R9_GROUP_NAME}']")
    want = {id(p) for n, p in model.named_parameters() if is_trainable_under(n, mods)}''', '''    mods = tuple(rec.get("modules") or ())
    dec = tuple(rec.get("declared_frozen") or ())
    bad = []
    if type(opt).__name__ != "AdamW":
        bad.append(f"optimizer is {type(opt).__name__}, the recipe is AdamW")
    if [g.get("name") for g in opt.param_groups] != [R9_GROUP_NAME]:
        bad.append(f"param groups {[g.get('name') for g in opt.param_groups]} != ['{R9_GROUP_NAME}']")
    want = {id(p) for n, p in model.named_parameters() if is_trained(n, mods, dec)}''')
edit('''    live = [n for n, p in model.named_parameters() if p.requires_grad and not is_trainable_under(n, mods)]
    if live:
        bad.append(f"{len(live)} parameter(s) OUTSIDE the declared sets {rec.get('sets')} are trainable, e.g. "
                   f"{live[:3]}")''', '''    live = [n for n, p in model.named_parameters() if p.requires_grad and not is_trainable_under(n, mods)]
    if live:
        bad.append(f"{len(live)} parameter(s) OUTSIDE the declared sets {rec.get('sets')} are trainable, e.g. "
                   f"{live[:3]}")
    undead = [n for n, p in model.named_parameters() if p.requires_grad and is_trainable_under(n, dec)]
    if undead:
        bad.append(f"{len(undead)} DECLARED grad-unreachable parameter(s) are trainable again, e.g. {undead[:3]}")''')

# the source check, the guard, the EMA check
edit('''def verify_against_source(model, src: dict, *, fresh: bool, modules: Iterable[str] = MAP_ONLY_MODULES) -> dict:''',
     '''def verify_against_source(model, src: dict, *, fresh: bool, modules: Iterable[str] = MAP_ONLY_MODULES,
                          declared: Iterable[str] = ()) -> dict:''')
edit('''    mods = tuple(modules)
    r9 = mods != MAP_ONLY_MODULES
    tag, what = ("r9-train-only", "frozen perception") if r9 else ("map-only", "frozen planner")''', '''    mods, dec = tuple(modules), tuple(declared)
    r9 = mods != MAP_ONLY_MODULES
    tag, what = ("r9-train-only", "frozen perception") if r9 else ("map-only", "frozen planner")''')
edit('''    diff = [k for k, v in sd.items() if (fresh or not is_trainable_under(k, mods)) and not bit_equal(v, src[k])]''',
     '''    diff = [k for k, v in sd.items() if (fresh or not is_trained(k, mods, dec)) and not bit_equal(v, src[k])]''')
edit('''    n_map = sum(1 for k in sd if is_trainable_under(k, mods))
    if r9:''', '''    n_map = sum(1 for k in sd if is_trained(k, mods, dec))
    if r9:''')
edit('''    def __init__(self, src: dict, modules: Iterable[str] = MAP_ONLY_MODULES):
        self.modules = tuple(modules)
        self.ref = {k: v.detach().to("cpu") for k, v in src.items() if not is_trainable_under(k, self.modules)}''',
     '''    def __init__(self, src: dict, modules: Iterable[str] = MAP_ONLY_MODULES, declared: Iterable[str] = ()):
        self.modules, self.declared = tuple(modules), tuple(declared)
        self.ref = {k: v.detach().to("cpu") for k, v in src.items() if not is_trained(k, self.modules, self.declared)}''')
edit('''def ema_problems(ema, model, src: dict | None = None, *, fresh: bool = False,
                 modules: Iterable[str] = MAP_ONLY_MODULES) -> list:
    """The EMA under the flag ([] = sound): it shadows EXACTLY the trainable tensors (default: the map-only ones); on
    a fresh start every shadow equals the live tensor AND the source (raw == EMA == the shipped model at step 0)."""
    mods = tuple(modules)
    where = "the map-only modules" if mods == MAP_ONLY_MODULES else f"the trainable modules {list(mods)}"
    bad = []
    keys = set(getattr(ema, "shadow", {}) or {})
    outside = sorted(k for k in keys if not is_trainable_under(k, mods))''', '''def ema_problems(ema, model, src: dict | None = None, *, fresh: bool = False,
                 modules: Iterable[str] = MAP_ONLY_MODULES, declared: Iterable[str] = ()) -> list:
    """The EMA under the flag ([] = sound): it shadows EXACTLY the trainable tensors (default: the map-only ones); on
    a fresh start every shadow equals the live tensor AND the source (raw == EMA == the shipped model at step 0)."""
    mods, dec = tuple(modules), tuple(declared)
    where = "the map-only modules" if mods == MAP_ONLY_MODULES else f"the trainable modules {list(mods)}"
    bad = []
    keys = set(getattr(ema, "shadow", {}) or {})
    outside = sorted(k for k in keys if not is_trained(k, mods, dec))''')

# G-DVB: the declared subtrees inside the sets are frozen BY DECLARATION, never "a frozen parameter of a set"
edit('''    frozen_in = [n for n, p in model.named_parameters() if is_trainable_under(n, mods) and not p.requires_grad]
    if frozen_in:''', '''    dec = tuple(rec.get("declared_frozen") or ())
    frozen_in = [n for n, p in model.named_parameters() if is_trained(n, mods, dec) and not p.requires_grad]
    undead = [n for n, p in model.named_parameters() if is_trainable_under(n, dec) and p.requires_grad]
    if undead:
        out.append(Mismatch(R9_FLAG, "declared grad-unreachable subtrees frozen", f"{len(undead)} trainable",
                            "model.named_parameters()", f"a declared subtree is trainable again, e.g. {undead[:3]}"))
    if frozen_in:''')

for old, new in E:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"anchor matched {n} times:\n{old[:400]}")
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("applied", len(E), "edits")
