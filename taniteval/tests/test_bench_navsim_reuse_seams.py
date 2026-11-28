"""``--reuse-seams`` adopts a banked model seam ONLY when its identity matches — MUTATION-tested.

⛔ WHY A MUTATION TEST AND NOT AN INSPECTION (CLAUDE.md, "guards need mutation, not inspection": an
AST census once read 0 suspects on BOTH the fixed and the broken trainer). The thing that must be
true is not *"the code contains a check"* — it is *"a seam from the wrong checkpoint CANNOT be
adopted"*. So each arm below reintroduces one real historical defect and REQUIRES the refusal.

⭐ The positive control matters as much: an identical seam MUST be adopted, and the adopted file
must be byte-identical to the banked one. A guard that refuses everything passes every negative
test and is useless.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "taniteval"))

MA = pytest.importorskip("taniteval.bench.navsim.model_arms")
B = pytest.importorskip("taniteval.bench.navsim.bridge")

MD5 = "0123456789abcdef0123456789abcdef"
TAG = {"bank_frame": [256, 640, 305.577491, "cylindrical"], "model_frame": "256x640f305.5775cyl",
       "match": True}
N1, N2 = 3, 7


def _bank(tmp: Path, *, md5=MD5, tag=TAG, e2_arm="A1_ego_cmd", n1=N1, n2=N2,
          n_model=None, n_standin=None, rows=None) -> Path:
    """Write a minimal banked seam + manifest, with every identity field overridable."""
    src = tmp / "banked"
    src.mkdir(parents=True, exist_ok=True)
    rows = rows if rows is not None else (n1 + n2)
    source = np.asarray(["cv_standin"] * n1 + ["refcv4b"] * (rows - n1))
    np.savez(src / "A1.npz", token=np.asarray([f"t{i}" for i in range(rows)]), source=source,
             poses=np.zeros((rows, 8, 3), np.float32), knots=np.zeros((rows, 8, 2), np.float32),
             sampling=np.asarray([8, 0.5]), arm=np.asarray(e2_arm))
    man = {"arm": "A1", "e2_arm": e2_arm, "n_stage1": n1, "n_stage2": n2,
           "n_model_rows": n_model if n_model is not None else int((source != "cv_standin").sum()),
           "n_cv_standin_rows": n_standin if n_standin is not None else int((source == "cv_standin").sum()),
           "seconds": 11232.6, "device_at_end": "cpu", "model": {"ckpt_md5": md5, "frame_tag": tag}}
    (src / "A1.manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return src


def _adopt(src: Path, out: Path, **kw):
    kw.setdefault("ckpt_md5", MD5)
    kw.setdefault("frame_tag", TAG)
    kw.setdefault("n_stage1", N1)
    kw.setdefault("n_stage2", N2)
    return MA.reuse_seam("A1", kw.pop("e2_arm", "A1_ego_cmd"), src, out, log=lambda *_: None, **kw)


def test_identical_seam_is_adopted_and_copied_byte_for_byte(tmp_path):
    """POSITIVE CONTROL — without it, a guard that refuses everything would pass this file."""
    src = _bank(tmp_path)
    out = tmp_path / "out"
    rec = _adopt(src, out)
    assert rec["reused"] is True
    assert rec["n_model_rows"] == N2 and rec["n_cv_standin_rows"] == N1
    assert (out / "A1.npz").read_bytes() == (src / "A1.npz").read_bytes()
    man = json.loads((out / "A1.manifest.json").read_text(encoding="utf-8"))
    assert man["reused"] is True and man["reused_from"]["dir"].endswith("banked")
    # the adoption must RECORD what it checked — an unrecorded guard is unauditable later
    assert "ckpt_md5" in man["reused_from"]["identity_checked"]


@pytest.mark.parametrize("mut, why", [
    (dict(md5="ffffffffffffffffffffffffffffffff"), "a seam from ANOTHER CHECKPOINT"),
    (dict(e2_arm="A4_blind_ego_cmd"), "a seam from ANOTHER ARM (blind vs camera)"),
    (dict(tag={"bank_frame": [256, 640, 999.0, "cylindrical"]}), "a seam at ANOTHER FRAME GEOMETRY"),
    (dict(n2=99), "a seam over a DIFFERENT TOKEN SET"),
    (dict(rows=N1 + N2 - 1), "a TRUNCATED seam (one row short)"),
    (dict(n_model=999), "a manifest whose counts LIE about the seam file"),
])
def test_mismatched_seam_is_REFUSED(tmp_path, mut, why):
    """Each case reintroduces a real way a stale seam could be adopted; all must go RED."""
    src = _bank(tmp_path, **mut)
    with pytest.raises(B.RefusedInput):
        _adopt(src, tmp_path / "out", e2_arm="A1_ego_cmd")
    assert not (tmp_path / "out" / "A1.npz").exists(), f"{why} left an adopted file behind"


def test_a_tuple_frame_tag_is_the_SAME_geometry_as_its_json_list(tmp_path):
    """⚠️ REGRESSION, MEASURED 2026-09-20 on the real banked seam. ``frame_tag_check`` returns
    ``bank_frame`` as a TUPLE in-process; JSON round-trips it to a LIST. A raw ``!=`` then refuses
    an IDENTICAL geometry, and the first version of this test could not see it because its fixture
    used a list on BOTH sides — a check whose two operands come from ONE serialisation path is not
    testing the comparison that production performs."""
    src = _bank(tmp_path)                                   # manifest holds a JSON list
    rec = _adopt(src, tmp_path / "out",
                 frame_tag={"bank_frame": (256, 640, 305.577491, "cylindrical")})   # caller: tuple
    assert rec["reused"] is True
    # and the control: a genuinely different geometry must still be refused, tuple or not
    with pytest.raises(B.RefusedInput):
        _adopt(src, tmp_path / "out2", frame_tag={"bank_frame": (256, 640, 999.0, "cylindrical")})


def test_missing_files_are_refused_not_silently_recomputed(tmp_path):
    src = _bank(tmp_path)
    (src / "A1.manifest.json").unlink()
    with pytest.raises(B.RefusedInput):
        _adopt(src, tmp_path / "out")


def test_cli_exposes_the_flag_and_benchmark_passes_it_through():
    """⛔ The "built, tested and UNREACHABLE FROM ITS CALLER" class: the flag must exist on the
    parser AND the benchmark must hand it to run_model_arms. Both hops are asserted, and the
    second is a source assertion with a same-breath control that must read non-zero."""
    from taniteval.bench import cli
    a = cli.build_parser().parse_args(
        ["navsim_v2", "--ckpt", "none", "--split", "navhard_two_stage", "--reuse-seams", "X/Y"])
    assert a.reuse_seams == "X/Y"
    src = (REPO / "taniteval" / "taniteval" / "bench" / "navsim" / "benchmark.py").read_text(
        encoding="utf-8")
    assert src.count('reuse=getattr(a, "reuse_seams", None)') == 1
    assert src.count("MA.run_model_arms(") == 1          # the control: the call site still exists
