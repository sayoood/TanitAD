"""Source-integrity guard for `tanitad.data.trajrecon`.

WHY THIS EXISTS -- a corrupt file passed both of the checks used while landing
this package (2026-08-08).

`vp_calib.py` was transferred, verified against the source's exact byte count,
verified again with `ast.parse`, and committed. It was CORRUPT: 2883 bytes of
binary garbage (`\\x02\\x8fOOOOO...`) sat where a reStructuredText table belongs.

Both checks passed because:

  - the corruption REPLACED a run of characters with a different run of the SAME
    LENGTH, so the byte count still matched the source exactly; and
  - the damage lay inside a module DOCSTRING, so the file remained syntactically
    valid Python and `ast.parse` was happy.

Byte count catches truncation and insertion. `ast.parse` catches structural
damage. Neither catches same-length substitution inside a string literal. This
module adds the check that does: no control characters, and valid UTF-8.

That is cheap and blunt, and it is deliberately not a checksum against the
upstream source -- these files have no upstream manifest in-repo. It defends
against the observed failure mode (binary noise landing in source) rather than
proving equivalence to an original.
"""

from __future__ import annotations

import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1] / "tanitad" / "data" / "trajrecon"

# Tab, LF, CR are the only control bytes legitimately present in these sources.
_ALLOWED_CONTROL = {0x09, 0x0A, 0x0D}

SOURCES = sorted(PKG.glob("*.py")) + sorted(PKG.glob("*.md"))


def test_package_directory_is_populated() -> None:
    """Guard the guard: an empty glob would make every check below vacuous."""
    assert len(SOURCES) >= 20, f"expected the full package, found {len(SOURCES)}: {SOURCES}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_source_has_no_control_bytes(path: pathlib.Path) -> None:
    """No stray control bytes -- the signature of binary noise in a transfer."""
    raw = path.read_bytes()
    bad = {b for b in raw if b < 0x20 and b not in _ALLOWED_CONTROL} | {b for b in raw if b == 0x7F}
    assert not bad, (
        f"{path.name} contains control bytes {sorted(hex(b) for b in bad)} -- this is the "
        f"corruption signature that byte-count and ast.parse both miss"
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_source_is_valid_utf8(path: pathlib.Path) -> None:
    path.read_bytes().decode("utf-8")   # raises UnicodeDecodeError on binary noise


def test_no_ondisk_module_is_undeclared() -> None:
    """Every module present on disk must be declared in `__all__`.

    This direction is always valid, including while the package is mid-transfer:
    a module that exists but is undeclared is unreachable through the lazy
    `__getattr__` and would be silently dead.
    """
    from tanitad.data import trajrecon

    on_disk = {p.stem for p in PKG.glob("*.py")} - {"__init__"}
    undeclared = on_disk - set(trajrecon.__all__)
    assert not undeclared, f"present on disk but missing from __all__: {sorted(undeclared)}"


def test_package_is_complete() -> None:
    """`__all__` and the directory agree -- i.e. every declared module landed.

    Skips (loudly) rather than fails while the source transfer is still in
    progress, so an incomplete checkout is visible as a skip with the missing
    names rather than as a red test that invites being ignored.
    """
    from tanitad.data import trajrecon

    on_disk = {p.stem for p in PKG.glob("*.py")} - {"__init__"}
    absent = set(trajrecon.__all__) - on_disk
    if absent:
        pytest.skip(f"package incomplete -- declared but not yet on disk: {sorted(absent)}")
    assert set(trajrecon.__all__) == on_disk


def test_import_is_lazy_and_dependency_free() -> None:
    """Importing the package must not drag in opencv/scipy/matplotlib.

    The package is an optional extra (`trajrecon`) and also needs ffmpeg+ffprobe
    on PATH. If this import went eager, a bare `import tanitad.data` would break
    for every consumer without those installed.
    """
    import sys

    for mod in ("cv2", "scipy", "matplotlib", "pandas"):
        sys.modules.pop(mod, None)

    from tanitad.data import trajrecon  # noqa: F401

    heavy = [m for m in ("cv2", "scipy", "matplotlib") if m in sys.modules]
    assert not heavy, f"importing trajrecon eagerly pulled in {heavy}"


def test_geodetic_transform_is_numerically_sane() -> None:
    """A behavioural check on transferred code, not just a syntactic one.

    0.0008993 deg of latitude is ~100 m north; the ENU 'up' component must show
    the small negative earth-curvature term rather than zero or a large value.
    """
    from tanitad.data.trajrecon.geo import LocalENU, wrap_pi

    enu = LocalENU(48.8566, 2.3522, 0.0)          # Paris; the 08-08 drive is in France
    e, n, u = enu.forward(48.8566 + 0.0008993, 2.3522)[0]

    assert abs(e) < 0.01, e
    assert 99.0 < n < 101.0, n
    assert -0.01 < u < 0.0, f"expected a small negative curvature term, got {u}"

    # 3*pi is the same ANGLE as pi, so assert on magnitude, not signed value.
    # MEASURED: this implementation wraps to [-pi, pi) -- wrap_pi(pi) and
    # wrap_pi(3*pi) both return -pi -- while geo.py's own docstring claims
    # "(-pi, pi]". The behaviour is self-consistent and correct as an angle; the
    # docstring's interval is what is wrong. Asserting the signed value here
    # would encode the docstring's error into the test suite.
    pi = 3.141592653589793
    assert abs(abs(float(wrap_pi(3.0 * pi))) - pi) < 1e-9
    assert abs(float(wrap_pi(0.5)) - 0.5) < 1e-12
