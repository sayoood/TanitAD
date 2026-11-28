"""PROMOTED, NOT REWRITTEN — pinned (W1, 2026-09-19).

The suite promotes E1/E2's validated NavSim code. "Promoted" is a claim about CONTENT, so it is
checked by content: devkit-side files must be byte-identical to their origins (git blob), and the
promoted functions must be AST-identical to the origin functions. A deliberate-regression arm
shows the AST check has teeth (a one-token change goes RED).
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
E1 = REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms/code"
E2 = REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/code"
BENCH = REPO / "taniteval/taniteval/bench"

#: (origin file, product file, [top-level function / class names that must be AST-identical])
PROMOTED = [
    (E1 / "verify_controls.py", BENCH / "navsim/summarize.py", ["epdms_formula", "c4"]),
    (E2 / "parse_scores.py", BENCH / "navsim/summarize.py", ["s2_group_uniform"]),
    # ⚠ ONLY build_win is still pinnable here. MEASURED 2026-09-20: E1 restructured their
    # build_artifacts.py (mtime 14:09) and it no longer defines short_row / navsim_gates /
    # gate_mutations at all. Those move to ORPHANED_PROMOTIONS below -- see the note there.
    (E1 / "build_artifacts.py", BENCH / "navsim/artifacts.py", ["build_win"]),
    (E2 / "build_artifacts.py", BENCH / "navsim/artifacts.py", ["_set", "_get"]),
    (E2 / "tanitad_navsim_bridge.py", BENCH / "navsim/bridge.py",
     ["declare", "nav_index_from_command", "ego_block", "FrameBank", "frame_tag_check", "slot_sources",
      "pack_frames", "load_refcv4b", "model_inputs", "run_model", "knots_to_navsim", "cv_navsim_poses",
      "sha256_file", "md5_file", "json_dump", "RefusedInput"]),
]
#: ⛔ ORPHANED PROMOTIONS -- promoted from an origin version that NO LONGER EXISTS.
#:
#: MEASURED 2026-09-20: E1 restructured `build_artifacts.py`; it now defines only
#: {build_win, gate_rows_from_criteria_check, main, read_csv_rows}. The version that contained
#: short_row / navsim_gates / gate_mutations is UNRECOVERABLE -- that file is not in HEAD (never
#: committed) and the index already holds the new one (blob 53051383), so there is no blob to
#: compare against. W1's copies are now the ONLY surviving ones.
#:
#: ⭐ The claim therefore CHANGES, and the pin changes with it rather than being quietly deleted:
#: "AST-identical to the live origin" is no longer checkable, so we pin what IS checkable -- that
#: MY copy has not drifted since it was orphaned. Dropping these names from PROMOTED without this
#: would have weakened the guarantee INVISIBLY, which is the failure this whole file exists to stop.
#: (product function name -> sha256(ast.dump)[:16] at the moment of orphaning)
ORPHANED_PROMOTIONS = {
    "short_row": "8f65223d034691ac",
    "navsim_gates": "4aaa8e489d8ca865",
    "gate_mutations": "dfdf5b5c03b04625",
}
ORPHANED_ORIGIN = E1 / "build_artifacts.py"

#: module-level constants that must be value-identical
CONSTANTS = [
    (E1 / "verify_controls.py", BENCH / "navsim/summarize.py", ["M8", "EC", "W", "SUMMARY", "SIGMA2"]),
    (E2 / "parse_scores.py", BENCH / "navsim/summarize.py", ["SUB"]),
    (E1 / "build_artifacts.py", BENCH / "navsim/artifacts.py", ["SHA", "LONG", "SUMMARY"]),
    (E2 / "build_artifacts.py", BENCH / "navsim/artifacts.py", ["NO_GT"]),
    (E2 / "tanitad_navsim_bridge.py", BENCH / "navsim/bridge.py",
     ["KNOT_T_S", "NAVSIM_T_S", "ALAT_V_FLOOR_MS", "KAPPA_CAP_INV_M", "DT_FRAME_S", "N_STACK",
      "NAVSIM_CMD_TO_NAV_NAME", "HEADING_HOLD_SPEED_MS", "ARMS", "FIELDS"]),
]


def _defs(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}


def _consts(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out[n.targets[0].id] = ast.dump(n.value)
    return out


def _dump(node) -> str:
    return ast.dump(node, include_attributes=False)


@pytest.mark.parametrize("origin,product,names", PROMOTED, ids=[f"{o.name}->{p.name}" for o, p, _ in PROMOTED])
def test_promoted_functions_are_ast_identical(origin, product, names):
    assert origin.exists(), f"MISSING origin (historical record moved?): {origin}"
    o, p = _defs(origin), _defs(product)
    for n in names:
        assert n in o, f"{n} not in origin {origin.name}"
        assert n in p, f"{n} not in product {product.name}"
        assert _dump(o[n]) == _dump(p[n]), f"{n}: product {product.name} DIVERGED from origin {origin.name}"


@pytest.mark.parametrize("origin,product,names", CONSTANTS, ids=[f"{o.name}->{p.name}" for o, p, _ in CONSTANTS])
def test_promoted_constants_are_identical(origin, product, names):
    o, p = _consts(origin), _consts(product)
    for n in names:
        assert o.get(n) is not None and o.get(n) == p.get(n), n


def test_bridge_has_exactly_one_promotion_edit():
    """bridge.py is E2's file with a 4-line PROMOTED header and ONE changed line (the REPO depth)."""
    o = (E2 / "tanitad_navsim_bridge.py").read_text(encoding="utf-8").splitlines()
    p = (BENCH / "navsim/bridge.py").read_text(encoding="utf-8").splitlines()
    assert p[0].startswith("# PROMOTED 2026-09-19") and p[4] == o[0], p[:5]
    body = p[4:]
    assert len(o) == len(body), (len(o), len(body))
    diff = [(i, a, b) for i, (a, b) in enumerate(zip(o, body)) if a != b]
    assert len(diff) == 1 and '*[".."] * 5' in diff[0][1] and '*[".."] * 4' in diff[0][2], diff


def _blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def _strip_w1_additions(text: str) -> str:
    out, skip = [], False
    for ln in text.splitlines(keepends=True):
        if "# --- W1 ADDITION" in ln:
            skip = True
            continue
        if "# --- end W1 ADDITION" in ln:
            skip = False
            continue
        if not skip:
            out.append(ln)
    return "".join(out)


def test_devkit_side_files_are_byte_identical_to_origin():
    prov = json.loads((BENCH / "navsim/devkit_side/PROVENANCE.json").read_text(encoding="utf-8"))
    assert set(prov["files"]) == {"navsim_win.py", "tanitad_seam_agent.py", "export_agent_inputs.py"}
    for name, rec in prov["files"].items():
        b = (BENCH / "navsim/devkit_side" / name).read_bytes()
        ob = (REPO / rec["origin"]).read_bytes()
        assert _blob(b) == rec["blob"], f"{name}: PROVENANCE blob is stale"
        assert _blob(ob) == rec["origin_blob"], f"{name}: the ORIGIN moved"
        if "w1_additions" not in rec:
            assert _blob(b) == _blob(ob), f"{name} is not byte-identical to its origin"


def test_navsim_win_is_e1_plus_marked_additions():
    """The wrapper carries ONE W1 addition (the pre-aggregation dump). Removing every line between
    the `# --- W1 ADDITION` / `# --- end W1 ADDITION` markers must reproduce E1's file BYTE-FOR-BYTE."""
    prov = json.loads((BENCH / "navsim/devkit_side/PROVENANCE.json").read_text(encoding="utf-8"))
    rec = prov["files"]["navsim_win.py"]
    assert rec.get("w1_additions"), "the addition must be declared in PROVENANCE.json"
    product = (BENCH / "navsim/devkit_side/navsim_win.py").read_bytes().decode("utf-8")
    origin = (REPO / rec["origin"]).read_bytes().decode("utf-8")
    assert _strip_w1_additions(product) == origin
    assert "--dump-preaggregation" in product and "worker_map_hook" in product
    assert product.count("# --- W1 ADDITION") == product.count("# --- end W1 ADDITION") == 2


def test_deliberate_regression_ast_check_has_teeth():
    """Mutate one token of a promoted function (the EPDMS denominator weight 5.0 -> 4.0 inside W is a
    constant; here: `worst <= 1e-9` -> `worst <= 1e-6` in c4) — the AST comparison must go RED."""
    src = (BENCH / "navsim/summarize.py").read_text(encoding="utf-8")
    mutated = src.replace("\"pass\": bool(n > 0 and worst <= 1e-9)}", "\"pass\": bool(n > 0 and worst <= 1e-6)}", 1)
    assert mutated != src
    m = {n.name: n for n in ast.parse(mutated).body if isinstance(n, ast.FunctionDef)}
    o = _defs(E1 / "verify_controls.py")
    assert _dump(m["c4"]) != _dump(o["c4"])


def test_e1_patch_pin_still_holds_on_the_promoted_wrapper():
    """E1's test_navsim_win_patch literals, on the PROMOTED wrapper (no navsim import needed)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("promoted_navsim_win", BENCH / "navsim/devkit_side/navsim_win.py")
    src = (BENCH / "navsim/devkit_side/navsim_win.py").read_text(encoding="utf-8")
    ns: dict = {}
    fn_src = src[src.index("def token_from_cache_path"):src.index("def _load_metric_cache_paths_patched")]
    exec("import re\n" + fn_src, ns)
    f = ns["token_from_cache_path"]
    WIN = r"C:\Users\Admin\navsim\exp\metric_cache_warmup_two_stage\2021.08.16.14.23.37_veh-45_00015_00132\unknown\5b733b329372ed8c8\metric_cache.pkl"
    LINUX = "/root/exp/metric_cache/2021.08.16.14.23.37_veh-45_00015_00132/unknown/4cac9f6cd85a5b47/metric_cache.pkl"
    assert f(WIN) == "5b733b329372ed8c8" and f(LINUX) == "4cac9f6cd85a5b47"
    with pytest.raises(IndexError):
        WIN.split("/")[-2]                      # the original devkit expression fails (the defect)
    assert spec is not None


def test_orphaned_promotions_have_not_drifted():
    """⛔ These came from an origin version that no longer exists, so AST-identity to the live file
    is unverifiable. What IS verifiable: my copy is byte-for-byte the logic that was promoted."""
    import hashlib
    pr = _defs(BENCH / "navsim/artifacts.py")
    for name, want in ORPHANED_PROMOTIONS.items():
        assert name in pr, f"{name} vanished from the product -- it is the ONLY surviving copy"
        got = hashlib.sha256(_dump(pr[name]).encode()).hexdigest()[:16]
        assert got == want, (f"{name}: AST changed since orphaning ({got} != {want}). If the change is "
                             f"intended, update ORPHANED_PROMOTIONS and say why in PROVENANCE.")


def test_orphaned_names_are_really_absent_from_the_live_origin():
    """⭐ THE DISCRIMINATING CONTROL. If E1 ever restores one of these, this goes RED and tells us to
    move it back under the real AST pin -- otherwise an 'orphaned' claim would quietly outlive the
    condition that justified it, which is a stale absence-claim of exactly the kind CLAUDE.md warns
    about. It also fails if the origin file disappears, rather than passing vacuously."""
    assert ORPHANED_ORIGIN.exists(), f"origin file is gone: {ORPHANED_ORIGIN}"
    o = _defs(ORPHANED_ORIGIN)
    assert o, "control: the origin must parse to a NON-EMPTY set of defs, or this proves nothing"
    back = sorted(n for n in ORPHANED_PROMOTIONS if n in o)
    assert back == [], (f"{back} are defined in the origin again -- move them back into PROMOTED and "
                        f"re-check AST identity instead of leaving them orphaned")
