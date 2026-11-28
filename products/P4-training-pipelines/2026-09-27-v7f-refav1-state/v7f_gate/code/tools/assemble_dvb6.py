"""Assemble tree_m/stack/tanitad/train/declared_vs_built_v6.py = the merge package's file VERBATIM +
the gate-time section + check_all. Exact anchors, each must match once; LF output (the package's EOL).
⚠️ HISTORY (2026-09-27 ~17:15): the merge PACKAGE has since adopted the assembled file (it now carries the gate
section), so the union base these anchors expect is gone from PKG and a re-run stops at ANCHOR x0 by design. The
nav-update edits were made in the section file AND the assembled file; the section is verified VERBATIM inside the
banked registry (`dvb6_gate_section.py` text `in` the file: True)."""
import sys
from pathlib import Path

PKG = Path(r"D:/Projects/TanitAD/products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_merge/"
           r"code/fix/stack/tanitad/train/declared_vs_built_v6.py")
OUT = Path(r"C:/Users/Admin/v7f_gate/tree_m/stack/tanitad/train/declared_vs_built_v6.py")
SEC = Path(r"C:/Users/Admin/v7f_gate/work/dvb6_gate_section.py").read_text(encoding="utf-8")
src = PKG.read_bytes().decode("utf-8")
assert "\r\n" not in src
edits = [
    # docstring: the scope statement is superseded by the gate-time section
    ("""⚠️ SCOPE: it covers ONLY the seven dests these two changes add. The ~230 pre-existing v6 flags have no entry here;
``uncovered_v6(parser)`` counts them. A v7F launch-gate profile must register those before a token can mean "every
flag is covered".
""", """⚠️ SCOPE (updated by the launch-gate agent, 2026-09-27): the seven dests the two changes add run in the TRAINER
(``check`` at build, ``check_v6`` on the R3 path); every OTHER trainer dest -- the tip's 231 + R6's ``strategic_off``
-- has a GATE-TIME entry (``register_gate``, the ``GATE_TIME`` set) that only ``check_all`` runs: the v7f launch
gate calls it on the FULLY built and frozen stack, so ``uncovered_v6(parser) == []`` on the merged trainer and the
trainer's own behaviour is unchanged.
"""),
    ("""    dvb6.uncovered_v6(parser)                   # -> every parser dest with no v6 entry (the gate's gap)
""", """    dvb6.uncovered_v6(parser)                   # -> every parser dest with no v6 entry (the gate's gap)
    dvb6.check_all(stack, args, weights_in_force=w_stage, parser=parser)   # EVERY entry (the launch gate)
"""),
    ("""           "V7_GOAL_WIDTH", "V7_LAT_WIDTH", "V7_LON_WIDTH", "SPEED_BAND_ARG_SLOTS", "Mismatch"]
""", """           "V7_GOAL_WIDTH", "V7_LAT_WIDTH", "V7_LON_WIDTH", "SPEED_BAND_ARG_SLOTS", "Mismatch",
           "GATE_TIME", "register_gate", "check_all", "kinds_census", "MODEL_CFG_LEVERS",
           "STAGE_LOSS_TERMS"]
"""),
    ("""NEEDS_WEIGHTS: set[str] = set()
""", """NEEDS_WEIGHTS: set[str] = set()
#: dests checked only at GATE time (``check_all``), never inside the trainer -- see ``register_gate``
GATE_TIME: set[str] = set()
"""),
    ("""# ============================================================================
# the API
# ============================================================================
""", SEC + """# ============================================================================
# the API
# ============================================================================
"""),
    ("""def _run(stack, args, weights_in_force, *, include_weight_levers: bool) -> list[Mismatch]:
    out: list[Mismatch] = []
    for lever in list(REGISTRY_V6.values()):
        if lever.kind not in ("built", "loss") or lever.check is None:
            continue
        if lever.dest in NEEDS_WEIGHTS and not include_weight_levers:
            continue
""", """def _run(stack, args, weights_in_force, *, include_weight_levers: bool,
         include_gate_levers: bool = False) -> list[Mismatch]:
    out: list[Mismatch] = []
    for lever in list(REGISTRY_V6.values()):
        if lever.kind not in ("built", "loss") or lever.check is None:
            continue
        if lever.dest in NEEDS_WEIGHTS and not include_weight_levers:
            continue
        if lever.dest in GATE_TIME and not include_gate_levers:
            continue
"""),
]
for old, new in edits:
    n = src.count(old)
    if n != 1:
        sys.exit(f"ANCHOR x{n}: {old[:90]!r}")
    src = src.replace(old, new)
src = src.rstrip("\n") + '''


def check_all(stack, args, *, weights_in_force=None, parser=None) -> list[Mismatch]:
    """-> EVERY registered lever on the FULLY built stack (build-time, R3 and gate-time entries), plus
    the parser coverage of EVERY dest (``uncovered_v6``) and G-HYG's walk of the config tree the stack
    holds; ``[]`` == clean. The launch gate's v7f G-DVB. With ``weights_in_force=None`` the levers that
    need the loss weights are skipped (say so: pass the stage's weights)."""
    from tanitad.train.config_hygiene import undeclared_attributes  # noqa: PLC0415
    out: list[Mismatch] = []
    if parser is not None:
        for d in uncovered_v6(parser):
            out.append(Mismatch(_flag(d), "a G-DVB(v6) registry entry", "none",
                                "tanitad.train.declared_vs_built_v6.REGISTRY_V6",
                                "a trainer flag with no declared-vs-built entry is refused "
                                "(SPEC_REFCV7 §2): register it with its built reader"))
    out += _run(stack, args, weights_in_force, include_weight_levers=weights_in_force is not None,
                include_gate_levers=True)
    cfg = getattr(stack, "cfg", None)
    if cfg is not None:
        for path, cls, attr in undeclared_attributes(cfg, "stack.cfg"):
            out.append(Mismatch(f"{path}.{attr}", "a declared field", "an ad-hoc attribute",
                                f"{path} ({cls})", "G-HYG: dropped by the next rebuild"))
    return out


def kinds_census(parser) -> dict[str, int]:
    """{kind: n} over the parser's dests (``UNCOVERED`` for a dest with no entry) -- the gate's table."""
    out: dict[str, int] = {}
    for d in sorted(_dests(parser)):
        k = REGISTRY_V6[d].kind if d in REGISTRY_V6 else "UNCOVERED"
        out[k] = out.get(k, 0) + 1
    return out
'''
OUT.write_bytes(src.encode("utf-8"))
print("assembled", OUT, src.count("\n"), "lines")
