"""tools/packed_path.py: packed repo paths resolve to the tar that holds them, and --verify catches a tampered,
missing or extra member (Master Mind 2026-10-22, the MAX_PATH packing of 171 over-long repo paths)."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import packed_path as pp  # noqa: E402

FILES = {"wrap/run__navhard_two_stage_final_scores_frame_PRE_AGGREGATION.csv": b"token,score\nabc,0.5\n",
         "wrap/run__navhard_two_stage_manifest.json": b'{"n": 1}\n'}


def _build(repo: Path, files: dict[str, bytes], tamper: str | None = None, drop: str | None = None,
           extra: str | None = None) -> None:
    """A miniature packed repo: pkg/raw/wrap.tar + sidecar + registry, written the way the lander writes them."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:", format=tarfile.PAX_FORMAT) as tf:
        for arc in sorted(files):
            if arc == drop:
                continue
            data = files[arc] + (b"X" if arc == tamper else b"")
            ti = tarfile.TarInfo(arc)
            ti.size, ti.mtime, ti.mode = len(data), 1700000000, 0o644
            tf.addfile(ti, io.BytesIO(data))
        if extra:
            ti = tarfile.TarInfo(extra)
            ti.size = 1
            tf.addfile(ti, io.BytesIO(b"x"))
    raw = buf.getvalue()
    side = "# arcname\tgit_blob\tsha256\tsize\tmode\n" + "".join(
        f"{a}\t{pp.git_blob_id(d)}\t{hashlib.sha256(d).hexdigest()}\t{len(d)}\t100644\n" for a, d in sorted(files.items()))
    (repo / "pkg/raw").mkdir(parents=True)
    (repo / "pkg/raw/wrap.tar").write_bytes(raw)
    (repo / "pkg/raw/wrap.tar.MEMBERS.tsv").write_text(side, encoding="utf-8")
    (repo / "tools").mkdir()
    (repo / "tools/packed_paths.json").write_text(json.dumps({"schema": "tanitad.packed_paths/1", "packs": [{
        "tar": "pkg/raw/wrap.tar", "sidecar": "pkg/raw/wrap.tar.MEMBERS.tsv", "prefix": "pkg/raw",
        "tar_sha256": hashlib.sha256(raw).hexdigest(), "members": sorted(files)}]}), encoding="utf-8")


def test_an_old_path_resolves_to_its_tar_and_member(tmp_path):
    _build(tmp_path, FILES)
    reg = pp.load_registry(tmp_path)
    hit = pp.resolve("pkg/raw/wrap/run__navhard_two_stage_manifest.json", reg)
    assert hit is not None
    assert hit[0]["tar"] == "pkg/raw/wrap.tar"
    assert hit[1] == "wrap/run__navhard_two_stage_manifest.json"
    assert pp.resolve("pkg/raw/wrap/not_there.csv", reg) is None
    assert pp.resolve("pkg/other/wrap/run__navhard_two_stage_manifest.json", reg) is None


def test_extract_returns_the_original_bytes(tmp_path):
    _build(tmp_path, FILES)
    rc = pp.main(["pkg/raw/wrap/run__navhard_two_stage_manifest.json", "--extract", str(tmp_path / "out"),
                  "--repo", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "out/run__navhard_two_stage_manifest.json").read_bytes() == b'{"n": 1}\n'


def test_verify_is_clean_on_an_intact_pack(tmp_path):
    _build(tmp_path, FILES)
    assert pp.verify(tmp_path) == []


def test_verify_catches_a_tampered_member(tmp_path):
    _build(tmp_path, FILES, tamper="wrap/run__navhard_two_stage_manifest.json")
    probs = pp.verify(tmp_path)
    assert any("content differs from its sidecar" in p for p in probs), probs


def test_verify_catches_a_missing_and_an_extra_member(tmp_path):
    _build(tmp_path, FILES, drop="wrap/run__navhard_two_stage_manifest.json", extra="wrap/stray.txt")
    probs = pp.verify(tmp_path)
    assert any("absent from the tar" in p for p in probs), probs
    assert any("not in the sidecar" in p for p in probs), probs


def test_the_real_registry_verifies_when_this_is_a_full_checkout():
    """In a full checkout every pack must verify; in a stack-only tree there is no registry to read."""
    if not (ROOT / pp.REGISTRY).is_file():
        pytest.skip("stack/tools-only tree: tools/packed_paths.json is not in this checkout")
    assert pp.verify(ROOT) == []
