"""§12 refusal 1, in BOTH halves — parsed namespaces AND the BUILT config.

`PREREG_REFCV6_V2.ERRATUM-1.md` §E5 records why one half is not enough. MEASURED
2026-09-17: `_pin_trainer_cfg` rebuilt `CNNEncoderConfig` from a **hand-written
field list** — 12 fields in the dataclass, 8 in the list — so under `--image-hw`
four fields silently returned to their defaults:

    --trunk-name resnet34        BUILT resnet101
    --trunk-fuse last            BUILT concat1x1   (the single-frame CONTROL)
    --trunk-fuse-plain-init      BUILT identity    (the deliberate REGRESSION)

⛔ **A parsed-namespace diff cannot see this.** The namespaces differ exactly as
declared — `trunk_name` really is `resnet34` on one side — and the **built configs
do not differ at all**. The defect lives BETWEEN argv and the model, which is
precisely where a checker that stops at argv is blind.

⭐ So the rule has two halves and **both are required**:

1. the parsed namespaces differ in exactly the declared lever (plus allow-listed
   constitutive and bookkeeping keys);
2. the **BUILT config** — the object the model is actually constructed from, after
   every pin and rebuild — differs in exactly the fields that lever is declared to
   move, **and a field that silently returns to its default is a VIOLATION**.

⛔ Expectations are written as LITERALS by the caller (`expected_from` /
`expected_to`), never as an expression over the code under test: an expectation
computed from the thing being checked measures determinism, not correctness.
"""
from __future__ import annotations

import dataclasses as _dc

__all__ = ["OneVariableViolation", "namespace_diff", "config_diff",
           "refuse_more_than_one_variable"]


class OneVariableViolation(RuntimeError):
    """Two arms differ in something other than their declared lever — in argv, or
    in the config they actually build."""


def _as_plain(obj, _depth: int = 0):
    """A dataclass / namespace tree as nested plain dicts, for diffing.

    ⛔ Bounded depth: a config that refers back to a model would otherwise recurse
    forever, and a checker that hangs is a checker that gets removed.
    """
    if _depth > 6:
        return repr(obj)
    if _dc.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _as_plain(getattr(obj, f.name, None), _depth + 1)
                for f in _dc.fields(obj)}
    if isinstance(obj, dict):
        return {k: _as_plain(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_plain(v, _depth + 1) for v in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, type) and obj.__dict__:
        return {k: _as_plain(v, _depth + 1) for k, v in vars(obj).items()}
    return obj


def _flat(tree, prefix: str = "") -> dict:
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(_flat(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = tree
    return out


def namespace_diff(ns_a, ns_b) -> dict:
    """Keys whose PARSED value differs, as ``{key: (a, b)}``."""
    a, b = vars(ns_a), vars(ns_b)
    keys = set(a) | set(b)
    return {k: (a.get(k), b.get(k)) for k in sorted(keys) if a.get(k) != b.get(k)}


def config_diff(cfg_a, cfg_b) -> dict:
    """Dotted paths whose BUILT value differs, as ``{path: (a, b)}``."""
    fa, fb = _flat(_as_plain(cfg_a)), _flat(_as_plain(cfg_b))
    keys = set(fa) | set(fb)
    return {k: (fa.get(k), fb.get(k)) for k in sorted(keys) if fa.get(k) != fb.get(k)}


def refuse_more_than_one_variable(
        argv_a, argv_b, *, lever: str, expected_from, expected_to,
        build_parser, build_config, constitutive=(), bookkeeping=("out", "seed"),
        built_allowed=()) -> dict:
    """Both halves. Returns a report; RAISES :class:`OneVariableViolation`.

    ``build_parser()`` must return the **trainer's own** parser and
    ``build_config(ns)`` the **trainer's own** config construction — ⛔ never a
    copy of either, or this becomes a check that shares the defect it checks for.

    ``built_allowed`` names the BUILT paths the lever is declared to move BESIDES
    the ones carrying the lever's own name — e.g. a channel width that follows a
    backbone. ⭐ Naming them is the point: an undeclared built difference is a
    VIOLATION, and a lever that moves nothing in the built config is one too.
    """
    ap = build_parser()
    ns_a, ns_b = ap.parse_args(list(argv_a)), ap.parse_args(list(argv_b))

    # ---- half 1: the parsed namespaces -------------------------------------
    nd = namespace_diff(ns_a, ns_b)
    if lever not in nd:
        raise OneVariableViolation(
            f"the declared lever {lever!r} does not differ between the arms at "
            f"all. Keys that do: {sorted(nd)}. An arm pair that does not move its "
            f"own lever is not an ablation.")
    got_from, got_to = nd[lever]
    if (got_from, got_to) != (expected_from, expected_to):
        raise OneVariableViolation(
            f"{lever!r} moved {got_from!r} -> {got_to!r}, but the arms declare "
            f"{expected_from!r} -> {expected_to!r}. ⛔ A pair differing in exactly "
            f"one key is NOT evidence the lever is on: it can move the wrong way, "
            f"or to the default.")
    extra = sorted(set(nd) - {lever} - set(constitutive) - set(bookkeeping))
    if extra:
        raise OneVariableViolation(
            f"the arms differ in argv keys beyond the lever: {extra}. Declare each "
            f"as CONSTITUTIVE (the trainer's own guard forces it) or BOOKKEEPING, "
            f"with a reason — or it is a second lever.")

    # ---- half 2: the BUILT config, which is where E5's defect lived ---------
    cd = config_diff(build_config(ns_a), build_config(ns_b))
    if not cd:
        raise OneVariableViolation(
            f"the arms' argv differ in {lever!r} but the BUILT CONFIGS ARE "
            f"IDENTICAL. ⛔ This is the `--image-hw` class: a hand-written field "
            f"list dropped the flag on the way to the model, so both arms build "
            f"the SAME network while config.json records what was asked for. The "
            f"two arms are one arm.")
    allowed = set(built_allowed)
    unexplained = sorted(p for p in cd
                         if p not in allowed and lever.split(".")[-1] not in p)
    if unexplained:
        raise OneVariableViolation(
            f"the BUILT configs differ in paths the lever does not declare: "
            f"{unexplained}. Add each to `built_allowed` with a reason, or the "
            f"arms differ in more than one thing where it actually counts.")
    return {"lever": lever, "from": expected_from, "to": expected_to,
            "argv_keys_differing": sorted(nd),
            "built_paths_differing": sorted(cd),
            "verdict": "ONE-VARIABLE"}
