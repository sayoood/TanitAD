"""Batch 3 -- the refcv7 launch blockers (Master Mind 2026-09-27), built as FULL files on
tip 06380de + batch 2:

  (a) the three modules the launch gate's G-LIVE MEASURED at zero gradient under the refcv7 argv
      (core.decoder.control_head, core.decoder.offset_head, scorer.goal_point) are BYPASSED BY
      CONSTRUCTION -- so they are FROZEN and DECLARED with tanitad/models/_gradreach.py (kept
      built: every checkpoint still loads strictly), and G-DVB holds the declared set against
      ARGV (check_grad_unreachable) and can MEASURE it (probe_grad_unreachable);
  (b) G-HYG: strict_fields on the six config classes the gate's probe found open
      (AgentSeamConfig, DiffusionFlags, EgoHistoryConfig, MaxSpeedConfig, Refcv7HeadConfig,
      TacticalDecoderConfig) + the walker API the probe uses;
  (c) SPEC_REFCV7 section 10 (A5), "G-DVB refuses any other mode for a refcv7 launch":
      check_refcv7_required also requires --residual-prior ha0_ext_pose in argv AND on the built
      decoder, and --ego-history in argv AND built.

    python patch_b3.py            # seeds code/fix3/ from the bases below, then edits it

Bases (40-char asserted before any edit):
  * tip 06380de -- refc.py, config_hygiene.py, the six config modules, test_config_hygiene.py;
  * batch 2 (code/fix2, on the Thor gate) -- refc_v3.py, declared_vs_built.py,
    test_declared_vs_built.py, refc_v3_train.py.
Every edit is anchored; an anchor found != 1 times refuses. Each file keeps its tip EOL.
"""
import subprocess
import sys
from pathlib import Path

PK = Path(__file__).resolve().parents[1]
FIX2 = PK / "code" / "fix2"
OUT = PK / "code" / "fix3"
GIT = ["git", "--git-dir=C:/Users/Admin/tanitad-push/.git"]
TIP = "06380de"

FROM_TIP = {
    "stack/tanitad/refs/refc.py": "6638d443533fb87e228f6bce7acf3e765ff93bed",
    "stack/tanitad/train/config_hygiene.py": "e6559a60874f4049b8c40a083ba68b762c83ac22",
    "stack/tanitad/models/ego_history.py": "f57adc9624b11bcac125e6ab855bcc1f18fec364",
    "stack/tanitad/models/refcv6_diffusion.py": "c1b8816a2a90fb064b75716257cbda3ca992bf62",
    "stack/tanitad/refs/max_speed_input.py": "0c32644a0d08ae243acc8692ba9dc3dd59d3fdce",
    "stack/tanitad/refs/refcv6_tactical.py": "afe1d5a59b122987502bca5b5f94ca69eaba773c",
    "stack/tanitad/refs/refcv7_heads.py": "a0206ea2f87d270ea3667148aa16c958ea88214c",
    "stack/tanitad/refs/refc_agents.py": "e647eace00b7a49da7e599b9393d84e190dbc38d",
    "stack/tests/test_config_hygiene.py": "8022e9777b581b69c5f98224190a04cc225232ae",
}
FROM_B2 = {
    "stack/tanitad/refs/refc_v3.py": "0b855f5334d0adcc36111e0872a8d261940d0098",
    "stack/tanitad/train/declared_vs_built.py": "a5fdddc36864abf7ebdd84dd74949be87b2a6388",
    "stack/tests/test_declared_vs_built.py": "2e9c620ec54295c3adb428afad6e7afffee35039",
    "stack/scripts/refc_v3_train.py": "f1a9c83ebccd040bf934281db970dc25326f6d1f",
}


def blob_of(p: Path) -> str:
    out = subprocess.run(["git", "hash-object", "--no-filters", str(p)], capture_output=True,
                         text=True).stdout.strip()
    assert len(out) == 40, (p, out)
    return out


def edit(s: str, old: str, new: str, tag: str) -> str:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"[patch_b3] {tag}: anchor found {n} times")
    return s.replace(old, new)


# ------------------------------------------------------------------ seed + assert the bases
if OUT.exists() and any(OUT.rglob("*.py")):
    raise SystemExit(f"[patch_b3] {OUT} already holds files -- refusing to overwrite (fresh dir)")
files: dict = {}
for rp, want in FROM_TIP.items():
    raw = subprocess.run(GIT + ["cat-file", "-p", f"{TIP}:{rp}"], capture_output=True,
                         check=True).stdout
    tmp = OUT / rp
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(raw)
    got = blob_of(tmp)
    if got != want:
        raise SystemExit(f"[patch_b3] tip {rp} is {got}, expected {want} -- the tip moved")
    files[rp] = raw
for rp, want in FROM_B2.items():
    got = blob_of(FIX2 / rp)
    if got != want:
        raise SystemExit(f"[patch_b3] batch-2 {rp} is {got}, expected {want}")
    files[rp] = (FIX2 / rp).read_bytes()

EOL = {rp: ("CRLF" if b.count(b"\r\n") == b.count(b"\n") else "LF") for rp, b in files.items()}
assert all((b"\r\n" not in b) or EOL[rp] == "CRLF" for rp, b in files.items()), "mixed EOL"
S = {rp: b.decode("utf-8").replace("\r\n", "\n") for rp, b in files.items()}

# ================================================================== (b) G-HYG: the six classes
HYG_IMPORT = "from tanitad.train.config_hygiene import strict_fields\n"
HYG_NOTE = ("# ⛔ G-HYG (SPEC_REFCV7 §2; batch 3, 2026-09-27): an UNDECLARED attribute RAISES at\n"
            "# assignment -- the launch gate's probe found this class accepting one.\n")
SIX = {
    "stack/tanitad/models/ego_history.py": ("from torch import Tensor, nn\n", "EgoHistoryConfig"),
    "stack/tanitad/models/refcv6_diffusion.py": ("from torch import Tensor, nn\n", "DiffusionFlags"),
    "stack/tanitad/refs/max_speed_input.py":
        ("from tanitad.channel_admissibility import ChannelExclusion\n", "MaxSpeedConfig"),
    "stack/tanitad/refs/refcv6_tactical.py":
        ("from tanitad.refs.refcv6_max_speed import N_SPEED_MAX_BINS_V6\n", "TacticalDecoderConfig"),
    "stack/tanitad/refs/refcv7_heads.py": ("from .refcv7_oracle import SUBSCORES\n", "Refcv7HeadConfig"),
    "stack/tanitad/refs/refc_agents.py": ("from torch import Tensor, nn\n", "AgentSeamConfig"),
}
for rp, (imp_anchor, cls) in SIX.items():
    s = S[rp]
    if imp_anchor.startswith("from torch"):
        s = edit(s, imp_anchor, imp_anchor + "\n" + HYG_IMPORT, f"{cls} import")
    else:
        s = edit(s, imp_anchor, imp_anchor + HYG_IMPORT, f"{cls} import")
    s = edit(s, f"@dataclass\nclass {cls}:", f"{HYG_NOTE}@strict_fields\n@dataclass\nclass {cls}:",
             f"{cls} decorator")
    S[rp] = s

# the walker the gate's probe runs, as the hygiene module's own API
rp = "stack/tanitad/train/config_hygiene.py"
S[rp] = edit(S[rp], '''__all__ = ["UndeclaredConfigAttribute", "strict_fields", "is_strict",
           "undeclared_attributes", "assert_config_hygiene"]
''', '''__all__ = ["UndeclaredConfigAttribute", "strict_fields", "is_strict",
           "undeclared_attributes", "assert_config_hygiene", "config_dataclass_instances",
           "non_strict_instances"]
''', "hyg __all__")
S[rp] = S[rp].rstrip("\n") + '''


def config_dataclass_instances(root: Any, path: str = "cfg") -> list[tuple[str, Any]]:
    """Every ``(path, dataclass instance)`` in the config tree under ``root``, sorted by path --
    the walk the refcv7 launch gate's G-HYG probe makes (``launch_gate._dataclass_instances``):
    dataclass instances through ``vars()``, lists, tuples and dicts; cycles cut by identity.

    ⚠️ Batch 3 (2026-09-27): the probe found SIX config classes in the refcv7 tree that
    accepted an undeclared attribute -- the tree had grown past the twelve batch 1 decorated.
    A strictness check must enumerate the TREE, never a list of modules.
    """
    out: list[tuple[str, Any]] = []
    seen: set[int] = set()
    stack: list[tuple[str, Any]] = [(path, root)]
    while stack:
        p, obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            out.append((p, obj))
        for suffix, child in _children(obj):
            if dataclasses.is_dataclass(child) or isinstance(child, (list, tuple, dict)):
                stack.append((p + suffix, child))
    return sorted(out, key=lambda pv: pv[0])


def non_strict_instances(root: Any, path: str = "cfg") -> list[tuple[str, str]]:
    """``(path, module.qualname)`` of every config dataclass instance under ``root`` whose class
    ACCEPTS an undeclared attribute (neither :func:`strict_fields` nor frozen). ``[]`` == the
    whole tree refuses the D-REFCV6-EQUALIZE-DROPPED mechanism at assignment."""
    return [(p, f"{type(o).__module__}.{type(o).__qualname__}")
            for p, o in config_dataclass_instances(root, path) if not is_strict(o)]
'''

# ================================================================== (a) the declarations
rp = "stack/tanitad/refs/refc.py"
S[rp] = edit(S[rp], "from tanitad.models import refcv6_diffusion as _rv6\n",
             "from tanitad.models import refcv6_diffusion as _rv6\n"
             "from tanitad.models import _gradreach as _gr  # batch 3 (a): the declaration\n",
             "refc import")
S[rp] = edit(S[rp], '''                "a distance, which is exactly the `metre_sigma_m` divisor trap "
                "one field up.")

    def anchor_control_seq(self, batch: int, dtype: torch.dtype) -> Tensor:
''', '''                "a distance, which is exactly the `metre_sigma_m` divisor trap "
                "one field up.")
        # ---- ⛔⛔ BATCH 3 (a), 2026-09-27: BUILT, BYPASSED BY DESIGN -> FROZEN + DECLARED ----- #
        # MEASURED by the refcv7 launch gate's G-LIVE smoke (and by NEW-1, its F-2): these heads
        # receive ZERO gradient on the refcv7 argv. Neither is a wiring defect; each is bypassed
        # BY CONSTRUCTION:
        #  * `offset_head` on a SAMPLER build: the classifier pass writes `x = bank + offset`
        #    (`forward`) and `_sample` then REPLACES `x` -- "the sampler REPLACES this loop, it
        #    does not wrap it"; `out["offset"]` is read by no loss of `compute_losses_v3`.
        #  * `control_head` under F3: `_decode_ctrl` returns the CASCADE's last stage whenever
        #    `self.cascade` is set and calls `control_head(q)` only when it is None; the
        #    cascade's per-layer heads GENERALISE it (`refcv6_diffusion.CascadeHeads`, DD).
        # ⭐ DECLARED, NOT DELETED (`tanitad/models/_gradreach.py`): both stay BUILT, so every
        # banked checkpoint (refcv6@38k, BAR-R7-2's reference) loads STRICTLY -- `requires_grad`
        # is not serialised. They leave the optimizer and carry their reason, so G-LIVE's "every
        # declared-trainable group gets a non-zero gradient" holds for the RIGHT reason, and
        # G-DVB holds the declared set against ARGV (`declared_vs_built.check_grad_unreachable`)
        # and can MEASURE it (`probe_grad_unreachable`), so a freeze cannot hide a live head.
        # ⚠️ The CLASSIFIER RL adapters read `out["offset"]` as their policy mean
        # (`tanitad/rl/refc_adapter.py`, `refcv3_adapter.py`); they are for classifier builds,
        # where nothing is frozen. The sampler's RL chain binds to `_decode_ctrl`.
        if self.control_head is not None:
            _gr.declare_grad_unreachable(
                self.offset_head,
                "sampler build: `_sample` REPLACES the classifier-pass fan `bank + offset`; "
                "out['offset'] is read by no training loss")
            if self.cascade is not None:
                _gr.declare_grad_unreachable(
                    self.control_head,
                    "F3: `_decode_ctrl` emits the CASCADE's last stage (DD's per-layer heads); "
                    "control_head(q) runs only when self.cascade is None")

    def anchor_control_seq(self, batch: int, dtype: torch.dtype) -> Tensor:
''', "refc decoder declarations")

rp = "stack/tanitad/refs/refc_v3.py"
S[rp] = edit(S[rp], '''        self.scorer = GoalDistanceScorer(d_goal_embed=cfg.d_tac,
                                         n_candidates=cfg.core.anchors.n_anchors,
                                         tau_m=cfg.scorer_tau_m)
        self.goal_gate = nn.Parameter(torch.zeros(()))   # zero-init: bit-inert
''', '''        self.scorer = GoalDistanceScorer(d_goal_embed=cfg.d_tac,
                                         n_candidates=cfg.core.anchors.n_anchors,
                                         tau_m=cfg.scorer_tau_m)
        # ⛔⛔ BATCH 3 (a), 2026-09-27: E9 ALWAYS hands the scorer the STRUCTURED goal
        # (`goal_point=g2` in `forward`), so its FREE decode `goal_point` (zero-init,
        # `v6.GoalDistanceScorer`) is bypassed BY CONSTRUCTION: it is returned only as
        # `goal_point_free` (the E-AG2 control) and read by no loss -- the launch gate's G-LIVE
        # MEASURED it at zero gradient on the refcv7 argv. Declared, not deleted
        # (`tanitad/models/_gradreach.py`): it stays built (strict checkpoint loads), leaves the
        # optimizer, and G-DVB holds the declaration against argv.
        # ⚠️ E-AG2's free-vs-structured comparison therefore compares against an UNFITTED (zero)
        # decode on this model; fitting it would need its own loss -- a design decision.
        from tanitad.models._gradreach import declare_grad_unreachable
        declare_grad_unreachable(
            self.scorer.goal_point,
            "E9 always passes goal_point=g2 (the structured goal); the free decode is returned "
            "only as out['goal_point_free'] and read by no training loss")
        self.goal_gate = nn.Parameter(torch.zeros(()))   # zero-init: bit-inert
''', "refc_v3 scorer declaration")

# ================================================================== G-DVB: (a) + (c)
rp = "stack/tanitad/train/declared_vs_built.py"
d = S[rp]
d = edit(d, '''__all__ = ["Mismatch", "Lever", "REGISTRY", "check", "refuse_on_mismatch", "coverage",
           "register", "KINDS", "DRIVORT_DEFAULTS", "drivort_levers_set",
           "REFCV7_REQUIRED_ON", "check_refcv7_required"]
''', '''__all__ = ["Mismatch", "Lever", "REGISTRY", "check", "refuse_on_mismatch", "coverage",
           "register", "KINDS", "DRIVORT_DEFAULTS", "drivort_levers_set",
           "REFCV7_REQUIRED_ON", "REFCV7_RESIDUAL_PRIOR", "check_refcv7_required",
           "check_logged_rows", "GRAD_UNREACHABLE_RULES", "expected_grad_unreachable",
           "declared_grad_unreachable", "check_grad_unreachable", "probe_grad_unreachable"]
''', "dvb __all__")
d = edit(d, '''    ⛔ Order of the report: unregistered flags first (a flag G-DVB cannot see is a hole in the
    guard itself), then every lever's mismatches in registry order, then the selection
    DECLARATION, then G-HYG on the config tree the model holds.
''', '''    ⛔ Order of the report: unregistered flags first (a flag G-DVB cannot see is a hole in the
    guard itself), then every lever's mismatches in registry order, then the selection
    DECLARATION, then the grad-unreachable DECLARATION (batch 3: the frozen modules against
    argv), then G-HYG on the config tree the model holds.
''', "check docstring")
d = edit(d, '''    out += _c_selection_declaration(model, args)
    cfg = getattr(model, "cfg", None)
''', '''    out += _c_selection_declaration(model, args)
    out += check_grad_unreachable(model, args)
    cfg = getattr(model, "cfg", None)
''', "check wiring")
d = edit(d, '''REFCV7_REQUIRED_ON: tuple[str, ...] = ("graft_tac8_prior", "graft_nav_compliance",
                                       "speed_ceiling_filter")
''', '''REFCV7_REQUIRED_ON: tuple[str, ...] = ("graft_tac8_prior", "graft_nav_compliance",
                                       "speed_ceiling_filter")

#: ⛔⛔ SPEC_REFCV7 §10 (A5): refcv7's prior is ``ha0_ext_pose`` -- past poses only -- and "G-DVB
#: refuses any other mode for a refcv7 launch": ``ha0_ext`` reads the recorded STEER at t0 (no PI
#: ruling on that channel at inference; NavSim has none, so BAR-R7-N1 could not be scored),
#: ``cv_yawrate`` starts 0.23 m behind the bar it must beat, ``off`` is refcv6. It requires
#: ``--ego-history``, the window the prior reads. A LITERAL, never read from kinematic_prior.
REFCV7_RESIDUAL_PRIOR: str = "ha0_ext_pose"
''', "REFCV7_RESIDUAL_PRIOR")
d = edit(d, '''    """-> a Mismatch for every REFCV7_REQUIRED_ON lever that is OFF in argv or UNBUILT in the
    model ([] == all three on and built). With ``tau_file`` (the banked
    `raw/nav_compliance_tau_train.json`), argv's `--nav-compliance-tau-rad` must equal its `tau`.
    """
''', '''    """-> a Mismatch for every REFCV7_REQUIRED_ON lever that is OFF in argv or UNBUILT in the
    model, for a residual prior that is not REFCV7_RESIDUAL_PRIOR in argv or on the BUILT
    decoder, and for an ego history OFF in argv or not built ([] == a refcv7 build). With
    ``tau_file`` (the banked `raw/nav_compliance_tau_train.json`), argv's
    `--nav-compliance-tau-rad` must equal its `tau`.
    """
''', "check_refcv7_required docstring")
d = edit(d, '''    if tau_file is not None:
        import json as _json
''', '''    # ---- SPEC_REFCV7 §10 (A5): the prior, in argv AND on the built decoder ---------------- #
    got = str(_a(args, "residual_prior", "off"))
    if got != REFCV7_RESIDUAL_PRIOR:
        out.append(Mismatch("--residual-prior", REFCV7_RESIDUAL_PRIOR, got, "argv",
                            "SPEC_REFCV7 §10 (A5): a refcv7 launch uses ha0_ext_pose only"))
    dec = getattr(_core(model), "decoder", None)
    built = str(getattr(dec, "residual_prior", "<absent>")) if dec is not None else "<absent>"
    if built != REFCV7_RESIDUAL_PRIOR:
        out.append(Mismatch("--residual-prior", REFCV7_RESIDUAL_PRIOR, built,
                            "core.decoder.residual_prior",
                            "SPEC_REFCV7 §10 (A5): the BUILT prior is the one that trains"))
    # ...and the observed pose window it reads
    if not bool(_a(args, "ego_history", False)):
        out.append(Mismatch("--ego-history", "ON (SPEC_REFCV7 §10)", "OFF in argv", "argv",
                            "ha0_ext_pose reads the observed pose window"))
    if getattr(_core(model), "ego_hist", None) is None:
        out.append(Mismatch("--ego-history", "BUILT", "not built", "core.ego_hist",
                            "ha0_ext_pose reads the observed pose window"))
    if tau_file is not None:
        import json as _json
''', "check_refcv7_required body")
d = d.rstrip("\n") + '''


# ============================================================================
# BATCH 3 (a), 2026-09-27: BUILT, BYPASSED BY DESIGN -- the grad-unreachable declaration
# ============================================================================
# MEASURED by the refcv7 launch gate's G-LIVE smoke: three modules take ZERO gradient on the
# refcv7 argv. Each is bypassed BY CONSTRUCTION, so the model FREEZES and DECLARES it
# (`tanitad/models/_gradreach.declare_grad_unreachable`, kept built for strict checkpoint
# loads). A freeze is exactly what could hide a dead group from G-LIVE, so it is held here in
# BOTH directions against a LITERAL rule table read from ARGV -- never from the config or the
# model -- and `probe_grad_unreachable` re-opens the frozen tensors for one backward to MEASURE
# that no loss reaches them.

#: (module path, the argv rule, why) -- the modules a refc_v3 build DECLARES grad-unreachable.
GRAD_UNREACHABLE_RULES: tuple[tuple[str, str, str], ...] = (
    ("core.decoder.offset_head", "--sampler ddim",
     "a sampler build: `_sample` REPLACES the classifier-pass fan bank + offset"),
    ("core.decoder.control_head", "--sampler ddim --f3-per-layer",
     "F3: the cascade's per-layer heads emit the fan (refcv6_diffusion.CascadeHeads)"),
    ("scorer.goal_point", "--arm hier",
     "E9 always passes the structured goal; the free decode is only out['goal_point_free']"),
)

#: the attributes the two freeze declarations set (`_gradreach.GRAD_UNREACHABLE_FLAG` and
#: `v6.FROZEN_EXTERNAL_FLAG`), read BY NAME so this module imports neither torch nor v6
_GRAD_UNREACHABLE_ATTR = "_tanitad_grad_unreachable"
_FROZEN_EXTERNAL_ATTR = "_tanitad_frozen_external"


def expected_grad_unreachable(args) -> dict[str, str]:
    """``{module path: why}`` -- the rule column of GRAD_UNREACHABLE_RULES evaluated on argv."""
    ddim = str(_a(args, "sampler", "none")) == "ddim"
    on = {"core.decoder.offset_head": ddim,
          "core.decoder.control_head": ddim and bool(_a(args, "f3_per_layer", False)),
          "scorer.goal_point": str(_a(args, "arm", "")) == "hier"}
    return {p: why for p, _rule, why in GRAD_UNREACHABLE_RULES if on[p]}


def _freezes(model) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for name, mod in model.named_modules():
        for attr in (_GRAD_UNREACHABLE_ATTR, _FROZEN_EXTERNAL_ATTR):
            why = getattr(mod, attr, None)
            if why:
                out[name] = (attr, str(why))
    return out


def declared_grad_unreachable(model) -> dict[str, str]:
    """``{module path: reason}`` of every subtree the BUILT model declared grad-unreachable
    (`config.json` records it: a frozen tensor that cannot say WHY it is frozen is the defect)."""
    if not callable(getattr(model, "named_modules", None)):
        return {}
    return {p: why for p, (attr, why) in sorted(_freezes(model).items())
            if attr == _GRAD_UNREACHABLE_ATTR}


def _under(name: str, prefix: str) -> bool:
    return prefix == "" or name == prefix or name.startswith(prefix + ".")


def check_grad_unreachable(model, args) -> list[Mismatch]:
    """-> the frozen-by-declaration modules held against argv; ``[]`` == they agree.

    * a module argv says is bypassed and the build did NOT declare (left trainable, it is a
      dead group G-LIVE refuses);
    * a declaration argv does NOT justify (a freeze that hides a live head from G-LIVE);
    * a declared module trainable again (the v6 stage-freeze class: a later pass UNDID the
      constructor's freeze -- `_gradreach.py`'s own measurement);
    * ANY frozen parameter outside a declared subtree (grad-unreachable or frozen-external).
    A hand-built stand-in with no ``named_parameters`` (unit tests) has nothing built and
    nothing declared, and returns ``[]``.
    """
    if not callable(getattr(model, "named_parameters", None)):
        return []
    decl = _freezes(model)
    gr = {p: why for p, (attr, why) in decl.items() if attr == _GRAD_UNREACHABLE_ATTR}
    want = expected_grad_unreachable(args)
    out: list[Mismatch] = []
    for p in sorted(set(want) - set(gr)):
        out.append(Mismatch(f"grad_unreachable[{p}]", "declared (argv: bypassed by design)",
                            "not declared", p,
                            f"{want[p]} -- left trainable it is a dead group G-LIVE refuses"))
    for p in sorted(set(gr) - set(want)):
        out.append(Mismatch(f"grad_unreachable[{p}]", "not declared (argv: reached)",
                            "declared", p,
                            f"a freeze argv does not justify hides a live head from G-LIVE: "
                            f"{gr[p]}"))
    params = list(model.named_parameters())
    for p in sorted(decl):
        live = [n for n, t in params if _under(n, p) and t.requires_grad]
        if live:
            out.append(Mismatch(f"grad_unreachable[{p}]", "requires_grad False",
                                f"{len(live)} trainable tensor(s)", p,
                                f"declared frozen but trainable again: {live[:3]}"))
    stray = [n for n, t in params if not t.requires_grad and not any(_under(n, p) for p in decl)]
    if stray:
        out.append(Mismatch("frozen parameters", "declared (grad-unreachable / frozen-external)",
                            f"{len(stray)} undeclared", "model.named_parameters()",
                            f"a freeze nobody declared hides a dead group from G-LIVE: "
                            f"{stray[:4]}"))
    return out


def probe_grad_unreachable(model, backward: Callable[[], Any]) -> list[Mismatch]:
    """MEASURE the declaration: re-open every declared grad-unreachable tensor, run
    ``backward()`` (the caller's forward + loss + ``.backward()``), and require that NONE of them
    received a gradient -- ``p.grad is None``: never in the graph. ``[]`` == every declaration
    is TRUE on this batch.

    ⛔ A freeze alone cannot prove a module is unreachable -- it makes the question
    unanswerable, since no gradient can arrive at a frozen tensor. This re-opens it for ONE
    backward. A present-but-zero gradient is a module IN the graph (wired, currently zero) and
    is reported: "unreachable" must mean no loss reaches it at all.
    ⛔ A POSITIVE CONTROL: some parameter outside the declared subtrees must receive a non-zero
    gradient, or the closure never ran a real backward and the probe read nothing.
    Restores ``requires_grad=False`` and ``.grad = None`` on every tensor it re-opened, and
    clears every gradient before and after (``zero_grad(set_to_none=True)``).
    """
    decl = declared_grad_unreachable(model)
    params = list(model.named_parameters())
    touched = [(n, t) for n, t in params if any(_under(n, p) for p in decl)]
    model.zero_grad(set_to_none=True)
    for _, t in touched:
        t.requires_grad_(True)
    try:
        backward()
        out: list[Mismatch] = []
        for p in sorted(decl):
            hit = [n for n, t in touched if _under(n, p) and t.grad is not None]
            if hit:
                out.append(Mismatch(f"grad_unreachable[{p}]", "no gradient (declared unreachable)",
                                    f"a gradient on {len(hit)} tensor(s)", p,
                                    f"the declaration is FALSE on this batch -- a loss reaches "
                                    f"{hit[:3]}: {decl[p]}"))
        reached = [n for n, t in params if not any(_under(n, p) for p in decl)
                   and t.grad is not None and bool((t.grad != 0).any())]
        if not reached:
            out.append(Mismatch("grad_unreachable probe", "a backward that reached the model",
                                "no non-zero gradient anywhere", "probe",
                                "the closure computed no gradient -- a probe that read nothing "
                                "certifies nothing"))
        return out
    finally:
        for _, t in touched:
            t.grad = None
            t.requires_grad_(False)
        model.zero_grad(set_to_none=True)
'''
S[rp] = d

# ================================================================== the trainer: the run record
rp = "stack/scripts/refc_v3_train.py"
S[rp] = edit(S[rp], '''                              "mismatches": 0,
                              "module": "tanitad/train/declared_vs_built.py"},
''', '''                              "mismatches": 0,
                              # batch 3 (a): the modules this build froze BY DESIGN, and why
                              "grad_unreachable": _dvb.declared_grad_unreachable(model),
                              "module": "tanitad/train/declared_vs_built.py"},
''', "trainer config.json record")

# ================================================================== tests: (c) in G-DVB's file
rp = "stack/tests/test_declared_vs_built.py"
t = S[rp]
t = edit(t, '''def test_REFCV7_all_three_on_and_built_PASSES():
    assert dvb.check_refcv7_required(_v6_model(**_ALL_ON), _SEL_ARGV) == []
''', '''#: ⛔ SPEC_REFCV7 §10 (A5): a refcv7 build ALSO carries the residual prior `ha0_ext_pose` on the
#: control-space DDIM sampler (v0-conditioned vocabulary) and the ego history the prior reads.
#: NEW-1's `_arm` recipe (`test_residual_prior.py`) through the trainer's OWN pin, then the
#: `_v6_model` tactical setup and the three selection mechanisms on the config.
_R7_PIN = ["--sampler", "ddim", "--anchor-v0-conditioned", "--anchor-control-units", "alat",
           "--n-anchors", "20", "--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln",
           "--f5-focal", "--f5-emitting-conf", "--f6-w-u0-zero"]
_R7_ARGV = argparse.Namespace(**vars(_SEL_ARGV), residual_prior="ha0_ext_pose", ego_history=True)


def _r7_model(residual_prior="ha0_ext_pose", **core_flags):
    from tanitad.refs import refcv6_tactical as v6tac
    from tanitad.refs.refc_agents import AgentSeamConfig
    argv = list(_R7_PIN) + ["--ego-history"]
    if residual_prior is not None:
        argv += ["--residual-prior", residual_prior]
    _a_, cfg = _pin(*argv)
    cfg.tac_vocab_version = "v7.0"
    cfg.tac_decoder_v6 = True
    cfg.max_speed_onehot_v6 = True
    cfg.core.agents = AgentSeamConfig(enable=True)
    cfg.core.decoder.cross_agent = True
    for k, v in {**_ALL_ON, **core_flags}.items():
        setattr(cfg.core, k, v)
    cfg.tac_decoder_cfg = v6tac.TacticalDecoderConfig(d_model=64, n_layers=1, n_heads=4,
                                                      d_bev=0)
    return v3.RefCV3Model(cfg)


@pytest.fixture(scope="module")
def r7():
    return _r7_model()


def test_REFCV7_all_three_on_and_built_PASSES(r7):
    assert dvb.check_refcv7_required(r7, _R7_ARGV) == []
''', "t6 PASSES")
t = edit(t, '''def test_REFCV7_RED_an_argv_missing_one_is_REFUSED(missing):
    args = argparse.Namespace(**{**vars(_SEL_ARGV), missing: False})
    got = [(x.lever, x.built) for x in dvb.check_refcv7_required(_v6_model(**_ALL_ON), args)]
''', '''def test_REFCV7_RED_an_argv_missing_one_is_REFUSED(r7, missing):
    args = argparse.Namespace(**{**vars(_R7_ARGV), missing: False})
    got = [(x.lever, x.built) for x in dvb.check_refcv7_required(r7, args)]
''', "t6 RED missing")
t = edit(t, '''    """refcv6-r101-s0's own build (behaviour set only) under a refcv7 argv."""
    got = sorted((x.lever, x.built) for x in
                 dvb.check_refcv7_required(_v6_model(graft_behaviour_sel=True), _SEL_ARGV))
    assert got == [("--graft-nav-compliance", "not built"), ("--graft-tac8-prior", "not built"),
                   ("--speed-ceiling-filter", "not built")]
''', '''    """refcv6-r101-s0's own build (behaviour set only, no residual prior, no ego history) under a
    refcv7 argv: FIVE refusals."""
    got = sorted((x.lever, x.built) for x in
                 dvb.check_refcv7_required(_v6_model(graft_behaviour_sel=True), _R7_ARGV))
    assert got == [("--ego-history", "not built"), ("--graft-nav-compliance", "not built"),
                   ("--graft-tac8-prior", "not built"), ("--residual-prior", "off"),
                   ("--speed-ceiling-filter", "not built")]
''', "t6 RED unbuilt")
t = edit(t, '''def test_REFCV7_tau_must_EQUAL_the_banked_file(tmp_path):
    p = tmp_path / "nav_compliance_tau_train.json"
    p.write_text(json.dumps({"tau": 0.08}), encoding="utf-8")
    m = _v6_model(**_ALL_ON)
    assert dvb.check_refcv7_required(m, _SEL_ARGV, tau_file=str(p)) == []
    off = argparse.Namespace(**{**vars(_SEL_ARGV), "nav_compliance_tau_rad": 0.09})
    assert [x.lever for x in dvb.check_refcv7_required(m, off, tau_file=str(p))] == \\
        ["--nav-compliance-tau-rad"]
''', '''def test_REFCV7_tau_must_EQUAL_the_banked_file(r7, tmp_path):
    p = tmp_path / "nav_compliance_tau_train.json"
    p.write_text(json.dumps({"tau": 0.08}), encoding="utf-8")
    assert dvb.check_refcv7_required(r7, _R7_ARGV, tau_file=str(p)) == []
    off = argparse.Namespace(**{**vars(_R7_ARGV), "nav_compliance_tau_rad": 0.09})
    assert [x.lever for x in dvb.check_refcv7_required(r7, off, tau_file=str(p))] == \\
        ["--nav-compliance-tau-rad"]


# ---- SPEC_REFCV7 §10 (A5): "G-DVB refuses any other mode for a refcv7 launch" (batch 3 (c)) - #
def test_REFCV7_RESIDUAL_PRIOR_is_the_SPEC_literal():
    assert dvb.REFCV7_RESIDUAL_PRIOR == "ha0_ext_pose"


@pytest.mark.parametrize("mode", ["ha0_ext", "cv_yawrate", "off"])
def test_REFCV7_RED_any_other_residual_prior_in_ARGV_is_REFUSED(r7, mode):
    args = argparse.Namespace(**{**vars(_R7_ARGV), "residual_prior": mode})
    got = [(x.lever, x.declared, x.built, x.read_from)
           for x in dvb.check_refcv7_required(r7, args)]
    assert got == [("--residual-prior", "ha0_ext_pose", mode, "argv")]


@pytest.mark.parametrize("mode", ["ha0_ext", "cv_yawrate", None])
def test_REFCV7_RED_a_decoder_BUILT_with_any_other_prior_is_REFUSED(mode):
    """The argv says `ha0_ext_pose`; the BUILT decoder carries another mode (None = off)."""
    got = [(x.lever, x.declared, x.built, x.read_from)
           for x in dvb.check_refcv7_required(_r7_model(residual_prior=mode), _R7_ARGV)]
    assert got == [("--residual-prior", "ha0_ext_pose", mode or "off",
                    "core.decoder.residual_prior")]


def test_REFCV7_RED_ego_history_OFF_in_argv_or_NOT_BUILT_is_REFUSED(r7):
    args = argparse.Namespace(**{**vars(_R7_ARGV), "ego_history": False})
    assert [(x.lever, x.built) for x in dvb.check_refcv7_required(r7, args)] == \\
        [("--ego-history", "OFF in argv")]
    m = _r7_model()
    m.core.ego_hist = None                      # REGRESSION ARM: lost in the build
    got = [(x.lever, x.built, x.read_from) for x in dvb.check_refcv7_required(m, _R7_ARGV)]
    assert got == [("--ego-history", "not built", "core.ego_hist")]
''', "t6 tau + new arms")
S[rp] = t

# ================================================================== tests: (b) in G-HYG's file
rp = "stack/tests/test_config_hygiene.py"
h = S[rp]
h = edit(h, '''from tanitad.train import config_hygiene as hyg  # noqa: E402
''', '''from tanitad.train import config_hygiene as hyg  # noqa: E402

import copy  # noqa: E402
import importlib  # noqa: E402
''', "hyg test imports")
h = h.rstrip("\n") + '''


# ============================================================================================ #
# Batch 3 (2026-09-27): the six config classes the launch gate's G-HYG probe found OPEN          #
# ============================================================================================ #
#: MEASURED by the gate's probe on the refcv7 argv (evidence on Thor: `.../G-HYG.json`): each of
#: these accepted an undeclared attribute. LITERAL -- module and class name.
STRICT_B3 = {
    "tanitad.refs.refc_agents": ("AgentSeamConfig",),
    "tanitad.models.refcv6_diffusion": ("DiffusionFlags",),
    "tanitad.models.ego_history": ("EgoHistoryConfig",),
    "tanitad.refs.max_speed_input": ("MaxSpeedConfig",),
    "tanitad.refs.refcv7_heads": ("Refcv7HeadConfig",),
    "tanitad.refs.refcv6_tactical": ("TacticalDecoderConfig",),
}
#: the OTHER dataclasses of those modules, each with the reason it is not an open config of the
#: pinned tree -- LITERAL, so a new dataclass there fails BY NAME until it is classified
NOT_A_TREE_CONFIG = {
    ("tanitad.refs.refc_agents", "RigCameraBank"): "frozen=True -- strict by construction",
    ("tanitad.refs.refcv6_tactical", "TacticalLossWeights"):
        "a per-call loss-weight record (`TacticalLossWeights().to_dict()`); RefCV3Config holds none",
}
#: the gate's active-probe attribute (`launch_gate._HYG_PROBE_ATTR`)
PROBE_ATTR = "_g_hyg_probe_undeclared_attribute"
#: a refcv7-shaped argv whose pin instantiates ALL SIX without opening a file (the label / join
#: paths are never opened by the pin: `test_refcv6_tactical_training.py`'s BASE)
R7_TREE_ARGV = ["--arm", "hier", "--out", "z", "--v7-labels", "labels.jsonl.gz",
                "--agents", "head", "--w-agent", "1.0", "--agent-join", "join.jsonl.xz",
                "--agent-join-verify", "off", "--tac-decoder-v6", "--w-tac-v6", "1.0",
                "--sampler", "ddim", "--anchor-v0-conditioned", "--anchor-control-units", "alat",
                "--n-anchors", "20", "--ego-history", "--residual-prior", "ha0_ext_pose",
                "--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln", "--f5-focal",
                "--f5-emitting-conf", "--f6-w-u0-zero"]


def _r7_tree():
    import refc_v3_train as T
    args = T.build_parser().parse_args(R7_TREE_ARGV)
    return T._pin_trainer_cfg(v3.refc_v3_smoke_config(True), args)


@pytest.mark.parametrize("modname,name", [(m, n) for m, ns in STRICT_B3.items() for n in ns])
def test_B3_the_six_classes_the_gate_found_open_are_STRICT(modname, name):
    cls = getattr(importlib.import_module(modname), name)
    assert hyg.is_strict(cls), f"{modname}.{name} accepts ad-hoc attributes"
    obj = cls()                                   # every field has its OFF default
    with pytest.raises(hyg.UndeclaredConfigAttribute, match=PROBE_ATTR):
        setattr(obj, PROBE_ATTR, 1)
    assert PROBE_ATTR not in vars(obj)
    assert dataclasses.replace(obj) == obj         # replace / __init__ / __post_init__ unchanged


def test_B3_no_OTHER_dataclass_in_those_modules_escaped():
    for modname, names in STRICT_B3.items():
        mod = importlib.import_module(modname)
        found = {n for n, o in vars(mod).items()
                 if isinstance(o, type) and dataclasses.is_dataclass(o) and o.__module__ == modname}
        assert set(names) <= found, (modname, sorted(set(names) - found))
        extra = sorted(n for n in found - set(names) if (modname, n) not in NOT_A_TREE_CONFIG)
        assert extra == [], (modname, extra)
    for (modname, name), _why in NOT_A_TREE_CONFIG.items():   # the exemptions are not stale
        assert dataclasses.is_dataclass(getattr(importlib.import_module(modname), name))


def test_B3_the_GATE_PROBE_on_a_refcv7_tree_every_config_object_REFUSES():
    """The launch gate's G-HYG probe, mirrored (`launch_gate.judge_hygiene`): walk EVERY config
    dataclass instance of the pinned refcv7-shaped tree; every class must be strict, and setting
    an undeclared attribute on EVERY object (on a deep copy) must raise."""
    cfg = _r7_tree()
    inst = hyg.config_dataclass_instances(cfg)
    classes = {type(o).__qualname__ for _, o in inst}
    six = {c for ns in STRICT_B3.values() for c in ns}
    assert six <= classes, ("the walk did not reach all six -- a probe that read less certifies "
                            "less", sorted(six - classes))
    assert len(inst) >= 10, len(inst)
    assert hyg.non_strict_instances(cfg) == []
    assert hyg.undeclared_attributes(cfg) == []
    accepted = []
    for p, o in hyg.config_dataclass_instances(copy.deepcopy(cfg)):
        try:
            setattr(o, PROBE_ATTR, 1)
        except (AttributeError, TypeError, dataclasses.FrozenInstanceError):
            continue
        accepted.append(p)
    assert accepted == [], accepted


def test_B3_RED_an_OPEN_class_in_the_tree_is_NAMED_and_ACCEPTS_the_probe():
    """The deliberate regression: the tree holds an instance of an undecorated class (the state
    of all six before this batch) -- the walk names it, and the probe's assignment succeeds on
    it, which is the D-REFCV6-EQUALIZE-DROPPED hole the guard exists to close."""
    @dataclasses.dataclass
    class OpenTwin:
        enable: bool = True
    cfg = _r7_tree()
    cfg.core.agents = OpenTwin()                      # a DECLARED field: the assignment is legal
    assert hyg.non_strict_instances(cfg) == [
        ("cfg.core.agents", f"{OpenTwin.__module__}.{OpenTwin.__qualname__}")]
    setattr(cfg.core.agents, PROBE_ATTR, 1)           # accepted: the hole
    assert vars(cfg.core.agents)[PROBE_ATTR] == 1
'''
S[rp] = h

# ================================================================== write, each with its tip EOL
for rp, txt in S.items():
    p = OUT / rp
    p.parent.mkdir(parents=True, exist_ok=True)
    body = txt.replace("\n", "\r\n") if EOL[rp] == "CRLF" else txt
    p.write_bytes(body.encode("utf-8"))
    print(f"[patch_b3] {rp} ({EOL[rp]}) blob {blob_of(p)}")
