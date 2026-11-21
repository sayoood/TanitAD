"""SPEC_REFCV7 §6.1 (5de9363): the DrivoR-T levers are NOT refcv7. G-DVB lists them as a
`drivort` kind with LITERAL defaults and reads none of their internals; a caller that launches
refcv7 forbids them (`forbid_kinds=("drivort",)`). Exact-match edits on the LF module."""
import sys

P = sys.argv[1]
s = open(P, encoding="utf-8", newline="").read()


def edit(old, new, tag):
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"[patch] {tag}: anchor found {n} times")
    s = s.replace(old, new)


edit('''__all__ = ["Mismatch", "Lever", "REGISTRY", "check", "refuse_on_mismatch", "coverage",
           "register", "KINDS"]

KINDS = ("built", "loss", "elsewhere", "data", "runtime", "record")
''', '''__all__ = ["Mismatch", "Lever", "REGISTRY", "check", "refuse_on_mismatch", "coverage",
           "register", "KINDS", "DRIVORT_DEFAULTS", "drivort_levers_set"]

KINDS = ("built", "loss", "elsewhere", "data", "runtime", "record", "drivort")
''', "kinds")

edit('''@dataclass(frozen=True)
class Lever:
    dest: str
    kind: str
    check: Callable[[Any, Any], list] | None = None
    reason: str = ""
''', '''@dataclass(frozen=True)
class Lever:
    dest: str
    kind: str
    check: Callable[[Any, Any], list] | None = None
    reason: str = ""
    #: ``drivort`` entries only: the argparse default, written as a LITERAL -- the value the
    #: flag must hold on a refcv7 launch (SPEC_REFCV7 §6.1).
    default: Any = None
''', "lever default")

# remove the DrivoR-T readers (they read DrivoR-T internals that are about to be renamed)
start = s.index("def _c_refcv7(m, a):")
end = s.index("def _c_perception(m, a):")
s = s[:start] + s[end:]

edit('''_b("refcv7", _c_refcv7)
for _d in ("r7_queries", "r7_d_model", "r7_layers", "r7_heads", "r7_img_pool", "r7_no_select",
           "r7_scorer_self_attn", "r7_toad"):
    register(_d, "elsewhere", reason="checked with --refcv7 against model.cfg (G-DVB)")
_b("w_r7_wta", _c_loss_weight("w_r7_wta", "_w_r7_wta",
                              lambda m: getattr(m, "refcv7_wta", None) is not None,
                              "model.refcv7_wta is not None"))
_b("w_r7_scorer", _c_loss_weight("w_r7_scorer", "_w_r7_scorer"))
_b("r7_nav_tau_rad", _c_attr("r7_nav_tau_rad", "_r7_nav_tau_rad",
                             lambda a: float(_a(a, "r7_nav_tau_rad", 0.0) or 0.0), tol=1e-12))
_b("r7_n_perturb", _c_attr("r7_n_perturb", "_r7_n_perturb",
                           lambda a: int(_a(a, "r7_n_perturb", 0) or 0)))
''', '''# ⛔⛔ DrivoR-T (SPEC_REFCV7 §6.1, PI 2026-09-26): `--refcv7` / `--w-r7-*` / `--r7-*` are the
# 2026-09-19 DrivoR-T draft (`Project Steering/SPEC_DRIVORT.md`), NOT refcv7, and are pending a
# mechanical rename to `drivort_*`. G-DVB reads NONE of their internals; it lists them with the
# literal default each must hold on a refcv7 launch. `check(..., forbid_kinds=("drivort",))`
# refuses any that is set; `train()` itself keeps DrivoR-T's own model-vs-weight refusal.
DRIVORT_DEFAULTS: dict[str, Any] = {
    "refcv7": False, "w_r7_wta": 0.0, "w_r7_scorer": 0.0, "r7_queries": 64, "r7_d_model": 256,
    "r7_layers": 4, "r7_heads": 8, "r7_img_pool": 2, "r7_no_select": False,
    "r7_scorer_self_attn": False, "r7_nav_tau_rad": 0.0, "r7_n_perturb": 32, "r7_toad": False,
    "r7_toad_iters": 5, "r7_toad_samples": 64, "r7_toad_seed": 0, "r7_w_ttc": 5.0,
    "r7_w_ep": 5.0, "r7_w_comf": 2.0,
}
for _d, _v in DRIVORT_DEFAULTS.items():
    REGISTRY[_d] = Lever(_d, "drivort", None, (
        "DrivoR-T (SPEC_DRIVORT.md), not refcv7: OFF on a refcv7 launch (SPEC_REFCV7 §6.1); "
        "pending the refcv7_* -> drivort_* rename"), _v)
''', "drivort registry")

edit('''for _d, _attr in (("r7_toad_iters", "refcv7_toad_iters"), ("r7_toad_samples", "refcv7_toad_samples"),
                  ("r7_toad_seed", "refcv7_toad_seed"), ("r7_w_ttc", "refcv7_w_ttc"),
                  ("r7_w_ep", "refcv7_w_ep"), ("r7_w_comf", "refcv7_w_comf")):
    register(_d, "elsewhere", reason=(
        f"pinned onto model.cfg.{_attr} only under --refcv7 (and --r7-toad); a READ at inference"))
''', '', "remove r7 toad elsewhere entries")

edit('''def check(model, args, parser=None) -> list[Mismatch]:
    """-> every declared-vs-built mismatch on the BUILT ``model``; ``[]`` == clean.
''', '''def drivort_levers_set(args) -> list[str]:
    """The DrivoR-T flags ``args`` sets away from their literal defaults (SPEC_REFCV7 §6.1)."""
    out = []
    for d, v in DRIVORT_DEFAULTS.items():
        got = getattr(args, d, v)
        if (abs(float(got) - float(v)) > 1e-12 if isinstance(v, float) else got != v):
            out.append(_flag(d))
    return out


def check(model, args, parser=None, *, forbid_kinds: tuple = ()) -> list[Mismatch]:
    """-> every declared-vs-built mismatch on the BUILT ``model``; ``[]`` == clean.

    ``forbid_kinds=("drivort",)`` -- what a refcv7 launch passes -- also refuses every DrivoR-T
    flag set away from its default (SPEC_REFCV7 §6.1). ``train()`` does not forbid them, so a
    DrivoR-T arm still trains under its own refusals until it is renamed.
''', "check signature")

edit('''    out += _c_selection_declaration(model, args)
''', '''    if "drivort" in tuple(forbid_kinds):
        for f in drivort_levers_set(args):
            d = f[2:].replace("-", "_")
            out.append(Mismatch(f, DRIVORT_DEFAULTS[d], getattr(args, d, None), "argv",
                                "a DrivoR-T lever on a refcv7 launch (SPEC_REFCV7 §6.1): "
                                "refcv7 passes none of them"))
    out += _c_selection_declaration(model, args)
''', "forbid drivort")

edit('''def refuse_on_mismatch(model, args, parser=None, where: str = "train") -> None:
    """``SystemExit`` naming EVERY mismatch, or return ``None`` when the build is clean."""
    bad = check(model, args, parser)
''', '''def refuse_on_mismatch(model, args, parser=None, where: str = "train", *,
                       forbid_kinds: tuple = ()) -> None:
    """``SystemExit`` naming EVERY mismatch, or return ``None`` when the build is clean."""
    bad = check(model, args, parser, forbid_kinds=forbid_kinds)
''', "refuse signature")

edit('''    if kind in ("built", "loss") and check is None:
''', '''    if kind == "drivort":
        raise ValueError("`drivort` entries are fixed by DRIVORT_DEFAULTS, not registered")
    if kind in ("built", "loss") and check is None:
''', "register guard")

open(P, "w", encoding="utf-8", newline="").write(s)
print("patched")
