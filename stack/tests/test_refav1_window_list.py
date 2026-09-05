"""`refav1_arm.select_windows` -- the panel's own definition.

⭐ WHY THIS FILE EXISTS. `--window-list` was added so a per-direction turn recall
could be resolved at all: the banked stride-16 panel carries 11 GT-left turns, and
a recall on 11 trials moves in steps of 1/11 = 0.0909, which is COARSER than the
0.0750 absolute inference-seed floor it is compared against. A stride cannot
enrich a stratum, so the selection had to become explicit.

⛔ THE THING THAT MUST NOT MOVE is every banked arm. `test_a_*` pins the
stride path bit-for-bit against the pre-2026-09-05 expression, each with a
same-breath control that must differ, so "the default is unchanged" is asserted
rather than asserted-about.
"""
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "taniteval", "tools"),
           os.path.join(_REPO, "stack")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

refav1_arm = pytest.importorskip("refav1_arm")
select_windows = refav1_arm.select_windows


class _Loader:
    """The two attributes `select_windows` reads, and nothing else."""

    def __init__(self, names, per_ep, w=4):
        self.names = list(names)
        self.W = int(w)
        self.windows = [(ei, t) for ei in range(len(names))
                        for t in range(self.W - 1, self.W - 1 + per_ep)]


def _legacy(ld, stride):
    """The pre-2026-09-05 expression, written out here on purpose so the test
    compares against the OLD CODE and not against the new code's own idea of
    itself."""
    return [(wi, ei, t) for wi, (ei, t) in enumerate(ld.windows)
            if (t - (ld.W - 1)) % stride == 0]


# --------------------------------------------------------------------------- #
# a. the default path is bit-identical                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stride", [1, 2, 3, 5, 16, 67])
def test_a_default_is_bit_identical_to_the_legacy_stride(stride):
    ld = _Loader(["e0", "e1", "e2"], per_ep=67)
    sel, meta = select_windows(ld, stride, None)
    assert meta is None
    assert sel == _legacy(ld, stride)
    # same-breath control: the strides must not all agree, or the assertion
    # above is vacuous
    assert _legacy(ld, 1) != _legacy(ld, 16)


def test_a_empty_window_list_string_takes_the_stride_path():
    """`--window-list ""` is an operator typo, not a request for zero windows."""
    ld = _Loader(["e0"], per_ep=20)
    assert select_windows(ld, 4, "")[0] == _legacy(ld, 4)
    assert select_windows(ld, 4, None)[1] is None


# --------------------------------------------------------------------------- #
# b. the explicit path selects exactly what it was asked for                  #
# --------------------------------------------------------------------------- #
def test_b_explicit_list_selects_exactly_those_windows(tmp_path):
    ld = _Loader(["e0", "e1"], per_ep=10)
    want = [["e1", 5], ["e0", 3], ["e1", 9]]
    p = tmp_path / "wl.json"
    p.write_text(json.dumps({"windows": want, "rule": "a stated rule"}),
                 encoding="utf-8")
    sel, meta = select_windows(ld, 16, str(p))
    assert [(ld.names[ei], t) for _, ei, t in sel] == [("e0", 3), ("e1", 5),
                                                       ("e1", 9)]
    assert meta["n_selected"] == 3 and meta["n_requested"] == 3
    assert meta["rule"] == "a stated rule"
    assert len(meta["sha256"]) == 64
    # control: the stride argument really is ignored, i.e. this is not the
    # legacy path wearing a hat
    assert [(ld.names[ei], t) for _, ei, t in sel] != [
        (ld.names[ei], t) for _, ei, t in _legacy(ld, 16)]


def test_b_selection_is_ordered_by_loader_index_not_by_file_order(tmp_path):
    ld = _Loader(["e0", "e1"], per_ep=10)
    p = tmp_path / "wl.json"
    p.write_text(json.dumps({"windows": [["e1", 3], ["e0", 7]]}),
                 encoding="utf-8")
    sel, _ = select_windows(ld, 1, str(p))
    assert [wi for wi, _, _ in sel] == sorted(wi for wi, _, _ in sel)
    assert sel[0][1] == 0 and sel[1][1] == 1


def test_b_a_missing_rule_is_recorded_rather_than_invented(tmp_path):
    ld = _Loader(["e0"], per_ep=10)
    p = tmp_path / "wl.json"
    p.write_text(json.dumps({"windows": [["e0", 3]]}), encoding="utf-8")
    assert "no rule declared" in select_windows(ld, 1, str(p))[1]["rule"]


# --------------------------------------------------------------------------- #
# c. it REFUSES rather than silently thinning the panel                       #
# --------------------------------------------------------------------------- #
def test_c_a_window_off_the_grid_is_refused_by_name(tmp_path):
    ld = _Loader(["e0"], per_ep=10)
    p = tmp_path / "wl.json"
    p.write_text(json.dumps({"windows": [["e0", 3], ["e0", 9999]]}),
                 encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        select_windows(ld, 1, str(p))
    assert "9999" in str(e.value) and "1 of 2" in str(e.value)
    # control: the same list WITHOUT the bad entry is accepted
    p.write_text(json.dumps({"windows": [["e0", 3]]}), encoding="utf-8")
    assert len(select_windows(ld, 1, str(p))[0]) == 1


def test_c_an_unknown_episode_is_refused(tmp_path):
    ld = _Loader(["e0"], per_ep=10)
    p = tmp_path / "wl.json"
    p.write_text(json.dumps({"windows": [["NOT_AN_EPISODE", 3]]}),
                 encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        select_windows(ld, 1, str(p))
    assert "NOT_AN_EPISODE" in str(e.value)


def test_c_duplicate_pairs_are_refused(tmp_path):
    """A duplicated window would be counted twice by every rate in the record
    and would make one episode silently heavier in the cluster bootstrap."""
    ld = _Loader(["e0"], per_ep=10)
    p = tmp_path / "wl.json"
    p.write_text(json.dumps({"windows": [["e0", 3], ["e0", 3]]}),
                 encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        select_windows(ld, 1, str(p))
    assert "duplicate" in str(e.value)


# --------------------------------------------------------------------------- #
# d. the file identifies windows by NAME, so a grid change cannot re-point it #
# --------------------------------------------------------------------------- #
def test_d_pairs_are_by_episode_name_not_by_window_index(tmp_path):
    """Two loaders whose episode ORDER differs must select the same windows."""
    a = _Loader(["e0", "e1"], per_ep=10)
    b = _Loader(["e1", "e0"], per_ep=10)
    p = tmp_path / "wl.json"
    p.write_text(json.dumps({"windows": [["e1", 4]]}), encoding="utf-8")
    ga = [(a.names[ei], t) for _, ei, t in select_windows(a, 1, str(p))[0]]
    gb = [(b.names[ei], t) for _, ei, t in select_windows(b, 1, str(p))[0]]
    assert ga == gb == [("e1", 4)]
    # control: the raw loader index for that pair really does differ, which is
    # what would have broken had the file carried indices
    assert select_windows(a, 1, str(p))[0][0][0] != \
        select_windows(b, 1, str(p))[0][0][0]


def test_d_sha256_changes_with_content_and_pins_the_panel(tmp_path):
    ld = _Loader(["e0"], per_ep=10)
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    p1.write_text(json.dumps({"windows": [["e0", 3]]}), encoding="utf-8")
    p2.write_text(json.dumps({"windows": [["e0", 4]]}), encoding="utf-8")
    h1 = select_windows(ld, 1, str(p1))[1]["sha256"]
    h2 = select_windows(ld, 1, str(p2))[1]["sha256"]
    assert h1 != h2 and len(h1) == 64


# --------------------------------------------------------------------------- #
# e. --help must survive a narrow console (a help string is CODE)             #
# --------------------------------------------------------------------------- #
def test_e_help_exits_zero_on_a_cp1252_console(tmp_path):
    """MEASURED 2026-09-05: `--help` exited 1 with `UnicodeEncodeError` on the
    dev box, because NINE help strings in this file carry a marker and a default
    Windows console is cp1252. Running arms never saw it (the queue scripts set
    PYTHONIOENCODING=utf-8), so it surfaced only when an operator asked for help.
    """
    import subprocess
    tool = os.path.join(_REPO, "taniteval", "tools", "refav1_arm.py")
    if not os.path.isfile(tool):
        pytest.skip("tool not present in this tree")
    env = dict(os.environ, PYTHONIOENCODING="cp1252",
               PYTHONPATH=os.path.join(_REPO, "stack"))
    r = subprocess.run([sys.executable, tool, "--help"], env=env,
                       capture_output=True, text=True, errors="replace",
                       timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "--window-list" in r.stdout
    # same-breath control: the markers really ARE unencodable in cp1252, so the
    # assertion above is not vacuously true of an ASCII-only help text
    with pytest.raises(UnicodeEncodeError):
        "\u26d4".encode("cp1252", errors="strict")


def test_e_a_cli_typo_gives_usage_not_a_traceback(tmp_path):
    import subprocess
    tool = os.path.join(_REPO, "taniteval", "tools", "refav1_arm.py")
    if not os.path.isfile(tool):
        pytest.skip("tool not present in this tree")
    env = dict(os.environ, PYTHONIOENCODING="cp1252",
               PYTHONPATH=os.path.join(_REPO, "stack"))
    r = subprocess.run([sys.executable, tool, "--not-a-flag"], env=env,
                       capture_output=True, text=True, errors="replace",
                       timeout=300)
    assert r.returncode == 2                      # argparse usage error
    assert "Traceback" not in (r.stdout + r.stderr)
