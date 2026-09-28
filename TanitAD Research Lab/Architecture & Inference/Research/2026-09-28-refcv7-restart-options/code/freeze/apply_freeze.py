#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv7 RESTART item 2 -- the model-side freeze of G-LIVE's 10 ADMITTED dead groups.

⛔ CLOSURE-TOUCHING. Both files this edits are IN BOTH binding closures of the live run
(`gmo_closure.json` and `gbo_closure.json`, pulled md5-verified from Thor 2026-09-28):

    stack/tanitad/refs/refc_v3.py             a5e7156752eecf5ce08831457b649bed27a4034c
    stack/tanitad/train/declared_vs_built.py  87cad54ea10d1a23e8f952fc3bdc86d91ed8711f

Landing either file voids G-MAP-OVERFIT and G-BOX-OVERFIT for any restart (judge_closure: "one
changed blob refuses"). This applier therefore stays PACKAGE-ONLY; it is run against a CLEAN tree
(a `git archive` of the launch commit, core.autocrlf=false) at the restart the PI decides.

What it does (exact-once replacements; it REFUSES any base but the launch blobs above and asserts
the result blobs, so a drifted base can never be half-patched):

* refc_v3.py: a literal table `BYPASS_DECLARATIONS` (module path, the CORE-config flag that
  bypasses it, why) and `declare_bypassed_by_flag(model)`, called LAST in RefCV3Model.__init__.
  Each module is frozen with `tanitad.models._gradreach.declare_grad_unreachable` -- the batch-3
  mechanism the three existing declarations use: the tensors stay in the state dict (strict loads
  of every banked checkpoint), leave the optimiser (`param_groups_dd` skips requires_grad=False),
  and carry their reason into config.json (`declared_vs_built.grad_unreachable`).
* declared_vs_built.py: the SAME ten rows as an independently written ARGV table
  `GRAD_UNREACHABLE_BYPASS_RULES` (keyed on `--no-strategic` / `--graft-tac8-prior`), folded into
  `expected_grad_unreachable`, so G-DVB holds the declaration against argv in BOTH directions. A
  row whose module the build did not construct (e.g. `nav_to_str` under `--goal-point-inject`) is
  not demanded: a module that does not exist cannot be a dead trainable group.

Usage:  python apply_freeze.py <tree-root> [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

BASES = {
    "stack/tanitad/refs/refc_v3.py": "a5e7156752eecf5ce08831457b649bed27a4034c",
    "stack/tanitad/train/declared_vs_built.py": "87cad54ea10d1a23e8f952fc3bdc86d91ed8711f",
}
#: the patched blobs (LF, as git stores both files), verified 2026-09-28 (raw/freeze: 12/12 green,
#: M0 launch code 11/11 RED, M1-M5 each RED).
RESULTS = {
    "stack/tanitad/refs/refc_v3.py": "72b62a0c9341aea19c5e1f9d80188d417078f4d6",
    "stack/tanitad/train/declared_vs_built.py": "d314cd1384e59e10c46a7b5029cb75103d06b5f4",
}

REFC_V3_TABLE_ANCHOR = (
    "# ============================================================================\n"
    "# The model\n"
    "# ============================================================================\n"
    "\n"
    "class RefCV3Model(nn.Module):\n")

REFC_V3_TABLE = '''# ============================================================================
# ⛔⛔ refcv7 RESTART FREEZE -- G-LIVE's ten ADMITTED dead groups, declared model-side
# ============================================================================
# At launch (fec3a0d) the launch gate ADMITTED these ten leaf groups by FLAG
# (`launch_gate.PROFILES["refcv7"]["live_dead_admitted"]`, PI 2026-09-27 ~20:55: "We dont need
# these modules for refcv7") and owed the model-side declaration at the next restart, because it
# touches code the binding runs recorded. Each is BYPASSED BY CONSTRUCTION under a flag of the
# launch argv; none is a wiring defect. MEASURED twice:
#   * G-LIVE's launch smoke (Thor, 100 steps, b16): exactly these ten of 474 leaf groups took
#     ZERO gradient (4,256,324 parameters, 20 tensors);
#   * the live run's own checkpoint at step 1,500 (ckpt.pt md5 c35966f7...): these 20 tensors are
#     the ONLY 20 of 808 optimiser entries with NO AdamW state -- AdamW creates state on the first
#     step a parameter has a `.grad`, so they had `.grad is None` on every one of 1,500 steps and
#     were never moved (not even by the decoupled weight decay, which AdamW applies only to
#     parameters with a gradient).
# ⇒ Freezing them leaves every OTHER parameter's trajectory bit-identical: `clip_grad_norm_` and
# AdamW both skip `.grad is None` tensors, so the frozen and the unfrozen build hand them the SAME
# tensor lists. What changes is the RECORD: they leave the optimiser and stop being counted as
# trained (`_gradreach.py`: a dead trainable tensor is a MEASUREMENT DEFECT).
# ⚠️ DECLARED, NOT DELETED: `requires_grad` is not serialised, so every banked checkpoint loads
# strictly. A resumed OPTIMISER is a different matter -- its param groups shrink by 20 entries;
# `…/2026-09-28-refcv7-restart-options/code/freeze/ckpt_freeze_convert.py` remaps a pre-freeze
# `ckpt.pt['opt']` and REFUSES if any dropped entry carries state.
# ⛔ The ARGV mirror of this table is `declared_vs_built.GRAD_UNREACHABLE_BYPASS_RULES`, written
# independently; G-DVB holds the two against each other.
#: (module path, the CORE-config flag that bypasses it, why)
BYPASS_DECLARATIONS: tuple[tuple[str, str, str], ...] = (
    ("core.strategic.gru", "no_strategic",
     "--no-strategic: the strategic context is computed only as a diagnostic (out['ctx']); "
     "no training loss reads it"),
    ("core.strategic.proj", "no_strategic",
     "--no-strategic: the strategic context projection, bypassed with the layer"),
    ("nav_to_str", "no_strategic",
     "--no-strategic: nav -> strategic ctx feeds only the diagnostic ctx; nav still reaches "
     "tactical (nav_to_tac) and operative (meas_in)"),
    ("str_goal_head", "no_strategic",
     "--no-strategic: the strategic goal head; its hindsight loss is dropped with the layer "
     "(goal_str_loss_applied False)"),
    ("gstr_embed", "no_strategic",
     "--no-strategic (S-BYPASS-2): the strategic goal's FiLM on the tactical latent is skipped"),
    ("gstr_film", "no_strategic",
     "--no-strategic (S-BYPASS-2): the strategic goal's FiLM on the tactical latent is skipped"),
    ("core.decoder.ctx_to_cond", "no_strategic",
     "--no-strategic (S-BYPASS-1): the decoder receives ctx=None, so ctx_to_cond never runs"),
    ("core.route_head", "no_strategic",
     "--no-strategic: the route readout; route_loss_applied is False and the route graft is off"),
    ("core.decoder.lat_to_anchor", "graft_tac8_prior",
     "--graft-tac8-prior (refcv6 sec. 4): the tactical 8x8 posterior REPLACES the image-only "
     "lat3 prior; the call site passes lat_prior=None"),
    ("core.decoder.lon_to_anchor", "graft_tac8_prior",
     "--graft-tac8-prior (refcv6 sec. 4): the tactical 8x8 posterior REPLACES the image-only "
     "lon3 prior; the call site passes lon_prior=None"),
)


def _module_at(root: nn.Module, path: str):
    node = root
    for part in path.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node if isinstance(node, nn.Module) else None


def declare_bypassed_by_flag(model: nn.Module) -> dict[str, str]:
    """Freeze + declare every BYPASS_DECLARATIONS module whose flag is ON in the core config and
    which this build constructed. -> ``{path: why}`` of what was declared (empty with both flags
    off: a build without them is untouched, bit for bit)."""
    from tanitad.models._gradreach import declare_grad_unreachable
    core_cfg = model.cfg.core
    done: dict[str, str] = {}
    for path, flag, why in BYPASS_DECLARATIONS:
        if not bool(getattr(core_cfg, flag, False)):
            continue
        mod = _module_at(model, path)
        if mod is None:
            continue
        declare_grad_unreachable(mod, why)
        done[path] = why
    return done


'''

REFC_V3_CALL_ANCHOR = (
    "            self.refcv7_scorer = r7h.DisentangledScorer(hc, src, slot_t)\n"
    "\n"
    "    # --- provenance (the PI's admissibility ruling, as data + roles) --------\n")
REFC_V3_CALL = (
    "            self.refcv7_scorer = r7h.DisentangledScorer(hc, src, slot_t)\n"
    "        # ⛔⛔ refcv7 RESTART FREEZE: LAST, after every module exists (BYPASS_DECLARATIONS).\n"
    "        # Consumes no RNG, so every parameter's init is exactly what it was before.\n"
    "        self._bypass_declared = declare_bypassed_by_flag(self)\n"
    "\n"
    "    # --- provenance (the PI's admissibility ruling, as data + roles) --------\n")

DVB_ALL_OLD = ('           "declared_grad_unreachable", "check_grad_unreachable", '
               '"probe_grad_unreachable"]\n')
DVB_ALL_NEW = ('           "declared_grad_unreachable", "check_grad_unreachable", '
               '"probe_grad_unreachable",\n'
               '           "GRAD_UNREACHABLE_BYPASS_RULES", "expected_bypass_unreachable"]\n')

DVB_RULES_ANCHOR = (
    '    ("scorer.goal_point", "--arm hier",\n'
    '     "E9 always passes the structured goal; the free decode is only out[\'goal_point_free\']"),\n'
    ')\n')
DVB_RULES_NEW = DVB_RULES_ANCHOR + '''
#: ⛔⛔ refcv7 RESTART FREEZE (the launch gate's G-LIVE ADMITTED these ten leaf groups by flag at
#: launch -- `launch_gate.PROFILES["refcv7"]["live_dead_admitted"]` -- and the model-side
#: declaration was owed at the next restart). MEASURED dead twice: G-LIVE's launch smoke (10 of 474
#: leaf groups) and the live run's step-1,500 checkpoint (the only 20 of 808 optimiser entries with
#: no AdamW state). The MODEL side is `refc_v3.BYPASS_DECLARATIONS` (keyed on the CORE config);
#: this is the ARGV side, written independently. (module path, the argv flag, why)
#: A row whose module the build did not construct is not demanded (see check_grad_unreachable).
GRAD_UNREACHABLE_BYPASS_RULES: tuple[tuple[str, str, str], ...] = (
    ("core.strategic.gru", "--no-strategic", "the strategic ctx is a diagnostic only"),
    ("core.strategic.proj", "--no-strategic", "the strategic ctx projection"),
    ("nav_to_str", "--no-strategic", "nav -> strategic ctx (diagnostic only)"),
    ("str_goal_head", "--no-strategic", "the strategic goal head; its loss is dropped"),
    ("gstr_embed", "--no-strategic", "S-BYPASS-2: the g_str FiLM is skipped"),
    ("gstr_film", "--no-strategic", "S-BYPASS-2: the g_str FiLM is skipped"),
    ("core.decoder.ctx_to_cond", "--no-strategic", "S-BYPASS-1: the decoder gets ctx=None"),
    ("core.route_head", "--no-strategic", "the route readout; route_loss_applied False"),
    ("core.decoder.lat_to_anchor", "--graft-tac8-prior", "the 8x8 posterior replaces lat3"),
    ("core.decoder.lon_to_anchor", "--graft-tac8-prior", "the 8x8 posterior replaces lon3"),
)
_BYPASS_ARGV = {"--no-strategic": "no_strategic", "--graft-tac8-prior": "graft_tac8_prior"}
'''

DVB_EXPECTED_OLD = (
    '    return {p: why for p, _rule, why in GRAD_UNREACHABLE_RULES if on[p]}\n')
DVB_EXPECTED_NEW = (
    '    out = {p: why for p, _rule, why in GRAD_UNREACHABLE_RULES if on[p]}\n'
    '    out.update(expected_bypass_unreachable(args))\n'
    '    return out\n'
    '\n'
    '\n'
    'def expected_bypass_unreachable(args) -> dict[str, str]:\n'
    '    """``{module path: why}`` -- GRAD_UNREACHABLE_BYPASS_RULES evaluated on argv (the refcv7\n'
    '    restart freeze): a row is ON when its flag is in argv."""\n'
    '    return {p: why for p, flag, why in GRAD_UNREACHABLE_BYPASS_RULES\n'
    '            if bool(_a(args, _BYPASS_ARGV[flag], False))}\n'
    '\n'
    '\n'
    'def _built_module(model, path: str) -> bool:\n'
    '    node = model\n'
    '    for part in path.split("."):\n'
    '        node = getattr(node, part, None)\n'
    '        if node is None:\n'
    '            return False\n'
    '    return callable(getattr(node, "named_parameters", None))\n')

DVB_CHECK_OLD = (
    '    decl = _freezes(model)\n'
    '    gr = {p: why for p, (attr, why) in decl.items() if attr == _GRAD_UNREACHABLE_ATTR}\n'
    '    want = expected_grad_unreachable(args)\n')
DVB_CHECK_NEW = (
    '    decl = _freezes(model)\n'
    '    gr = {p: why for p, (attr, why) in decl.items() if attr == _GRAD_UNREACHABLE_ATTR}\n'
    '    want = expected_grad_unreachable(args)\n'
    '    # the refcv7 restart freeze: a bypass row whose module this build did not construct is not\n'
    '    # demanded (it cannot be a dead TRAINABLE group); every built one is.\n'
    '    _bp = {p for p, _f, _w in GRAD_UNREACHABLE_BYPASS_RULES}\n'
    '    want = {p: w for p, w in want.items() if p not in _bp or _built_module(model, p)}\n')

EDITS = {
    "stack/tanitad/refs/refc_v3.py": [
        (REFC_V3_TABLE_ANCHOR, REFC_V3_TABLE + REFC_V3_TABLE_ANCHOR),
        (REFC_V3_CALL_ANCHOR, REFC_V3_CALL),
    ],
    "stack/tanitad/train/declared_vs_built.py": [
        (DVB_ALL_OLD, DVB_ALL_NEW),
        (DVB_RULES_ANCHOR, DVB_RULES_NEW),
        (DVB_EXPECTED_OLD, DVB_EXPECTED_NEW),
        (DVB_CHECK_OLD, DVB_CHECK_NEW),
    ],
}


def git_blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(b) + b).hexdigest()


def patch_bytes(rel: str, raw: bytes) -> bytes:
    if b"\r\n" in raw:
        raise SystemExit(f"[freeze] {rel}: CRLF on disk -- extract the tree with core.autocrlf=false "
                         f"(git stores this file LF; the base blob check would refuse anyway)")
    txt = raw.decode("utf-8")
    for old, new in EDITS[rel]:
        n = txt.count(old)
        if n != 1:
            raise SystemExit(f"[freeze] {rel}: anchor found {n} times, expected exactly 1:\n{old[:200]}")
        txt = txt.replace(old, new)
    return txt.encode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("--check", action="store_true", help="verify only; write nothing")
    ap.add_argument("--out-dir", default=None,
                    help="write the patched files here (mirroring their tree paths) instead of "
                         "in place")
    a = ap.parse_args(argv)
    root = Path(a.tree)
    for rel, base in BASES.items():
        p = root / rel
        raw = p.read_bytes()
        got = git_blob(raw)
        if got == RESULTS.get(rel):
            print(f"[freeze] {rel}: ALREADY PATCHED ({got})")
            continue
        if got != base:
            raise SystemExit(f"[freeze] {rel}: base blob {got} is not the launch blob {base} -- REFUSED")
        new = patch_bytes(rel, raw)
        nb = git_blob(new)
        want = RESULTS.get(rel)
        if want is not None and nb != want:
            raise SystemExit(f"[freeze] {rel}: result blob {nb} != the verified {want} -- REFUSED")
        print(f"[freeze] {rel}: {base[:12]} -> {nb}")
        if a.check:
            continue
        dst = (Path(a.out_dir) / rel) if a.out_dir else p
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
