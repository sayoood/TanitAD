"""hold_r9_reuse_fp (MM 2026-11-04): a REUSE-bound arm's G-MAP / G-BOX-OVERFIT evidence must be USABLE by `finalize`.

MEASURED on Thor 2026-11-04T13:20:30Z, arm T1 (its records = reuse bindings to C0's) at 5761cd2: every check PASS, token INCOMPLETE --
"G-MAP-OVERFIT MISSING: input gate:map-overfit-record read with a different fingerprint, input gate:map-overfit-reuse-binding read with a different
fingerprint" (and the same for G-BOX-OVERFIT).  The job fingerprints the BASE record (through the binding) and the binding; the token's manifest bound the
BINDING under the record key and did not bind the binding key at all.  `overfit_record_gate_inputs` now binds exactly what the job reads.

Executed end to end, nothing re-implemented: the real binding (test_launch_gate_e1rig_reuse.Base -> `write_reuse_binding`), the real job
(`_job_box_overfit` / `_job_map_overfit`), the real manifest (`data_manifest` over `overfit_record_gate_inputs`) and finalize's own per-evidence rule
(`evidence_rejections`).  The job's STATUS is not asserted: off-Thor the closure / class-weight judges FAIL (no real run, no
Thor data) exactly as in test_launch_gate_e1rig_reuse; what this file pins is whether finalize USES the evidence.  RED: the old manifest (the option under the record key, no binding key) reproduces Thor's two rejections word for word.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402

RU = pytest.importorskip("test_launch_gate_e1rig_reuse")

JOB = {"box": LG._job_box_overfit, "map": LG._job_map_overfit}
CHECK = {"box": "G-BOX-OVERFIT", "map": "G-MAP-OVERFIT"}


def _ctx(argv, rec, clo, out, kind, gin):
    opts = {f"{kind}_overfit_record": str(rec), f"{kind}_overfit_closure": str(clo), "gate_inputs": gin}
    return LG.Ctx(profile="refcv9-e1rig", tree=str(RU.TREE), commit=RU.COMMIT, argv=list(argv), out_dir=str(out), path_map=[], tree_sha256="0" * 64,
                  argv_sha256=LG.argv_sha256(argv), options=opts)


def _read(kind, rec, clo, argv, out, gin):
    """(the job's evidence, finalize's rejections of it under the manifest `gin` binds)."""
    ctx = _ctx(argv, rec, clo, out, kind, gin)
    ev = JOB[kind](ctx)[CHECK[kind]]
    data = LG.data_manifest(ctx.argv, ctx.prof, [], gin)
    return ev, LG.evidence_rejections(ev, ctx.binding(), data)


def _gin(kind, rec, clo):
    return {**LG.overfit_record_gate_inputs(kind, str(rec)), f"gate:{kind}-overfit-closure": str(clo)}


@pytest.mark.parametrize("kind", ["box", "map"])
@pytest.mark.parametrize("arm", ["B1", "T1"])
def test_a_reuse_bound_arms_evidence_is_usable_by_finalize(tmp_path, kind, arm):
    base = RU.Base(tmp_path, kind)
    reasons, _, bp = base.bind(tmp_path, arm)
    assert reasons == [], reasons
    clo = bp.parent / f"{kind}_closure.json"
    gin = _gin(kind, bp, clo)
    assert Path(gin[f"gate:{kind}-overfit-record"]).samefile(base.rec) and gin[f"gate:{kind}-overfit-reuse-binding"] == str(bp)
    ev, why = _read(kind, bp, clo, RU._argv(arm), tmp_path / "gate", gin)
    assert set(ev["inputs_read"]) == {f"gate:{kind}-overfit-record", f"gate:{kind}-overfit-closure", f"gate:{kind}-overfit-reuse-binding"}
    assert why == [], why


@pytest.mark.parametrize("kind", ["box", "map"])
def test_RED_the_old_manifest_reproduces_thors_two_rejections(tmp_path, kind):
    base = RU.Base(tmp_path, kind)
    _, _, bp = base.bind(tmp_path, "T1")
    clo = bp.parent / f"{kind}_closure.json"
    old = {f"gate:{kind}-overfit-record": str(bp), f"gate:{kind}-overfit-closure": str(clo)}     # what the gate bound before this fix
    ev, why = _read(kind, bp, clo, RU._argv("T1"), tmp_path / "gate", old)
    assert why == [f"input gate:{kind}-overfit-record read with a different fingerprint",
                   f"input gate:{kind}-overfit-reuse-binding read with a different fingerprint"], why


@pytest.mark.parametrize("kind", ["box", "map"])
def test_a_plain_run_record_binds_only_the_record_and_is_usable(tmp_path, kind):
    base = RU.Base(tmp_path, kind)
    gin = _gin(kind, base.rec, base.clo)
    assert set(gin) == {f"gate:{kind}-overfit-record", f"gate:{kind}-overfit-closure"} and gin[f"gate:{kind}-overfit-record"] == str(base.rec)
    ev, why = _read(kind, base.rec, base.clo, RU._argv("C0"), tmp_path / "gate", gin)
    assert "gate:%s-overfit-reuse-binding" % kind not in ev["inputs_read"] and why == [], (ev["reasons"], why)


def test_an_unresolvable_base_binds_only_the_binding_and_the_judge_refuses(tmp_path):
    import json
    base = RU.Base(tmp_path, "box")
    _, _, bp = base.bind(tmp_path, "B1")
    b = json.loads(bp.read_text(encoding="utf-8"))
    b["base_record"] = dict(b["base_record"], relpath="nope.json", abspath=str(tmp_path / "nope.json"))
    bp.write_text(json.dumps(b), encoding="utf-8")
    gin = LG.overfit_record_gate_inputs("box", str(bp))
    assert gin == {"gate:box-overfit-reuse-binding": str(bp)}
    ev, _ = _read("box", bp, bp.parent / "box_closure.json", RU._argv("B1"), tmp_path / "gate", {**gin, "gate:box-overfit-closure": str(bp.parent / "box_closure.json")})
    assert ev["status"] != "PASS" and any("does not exist here" in r for r in ev["reasons"])


def test_the_judge_and_the_manifest_resolve_the_SAME_base_files(tmp_path):
    base = RU.Base(tmp_path, "box")
    _, _, bp = base.bind(tmp_path, "B1")
    r, det, rec, clo = base.judge(bp, RU._argv("B1"), work=tmp_path / "w")
    paths = LG.reuse_base_paths(bp)
    assert r == [] and Path(rec) == paths["base_record"] and Path(clo) == paths["base_closure"]


def test_finalize_uses_evidence_rejections():
    src = Path(LG.__file__).read_text(encoding="utf-8")
    body = src.split("def finalize(", 1)[1].split("\ndef ", 1)[0]
    assert "why = evidence_rejections(ev, ctx.binding(), data)" in body
    assert 'gin.update(overfit_record_gate_inputs("map", options["map_overfit_record"]))' in src
    assert 'gin.update(overfit_record_gate_inputs("box", options["box_overfit_record"]))' in src
