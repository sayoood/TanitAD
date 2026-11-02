"""refcv9 C7 integration pieces (E1 rig through the FULL train() path): the G-DVB rig exemption and the E1 argv builder.
Literal expectations; each guard carries a RED arm (the defect it must catch)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tanitad.train import declared_vs_built as dvb          # noqa: E402
import r9_e1_make_argv as M                                  # noqa: E402

RIG = SimpleNamespace(core=SimpleNamespace(_perc_rig=True))
FULL = SimpleNamespace(core=SimpleNamespace(_perc_rig=False))


def _err(name):
    return AttributeError("'RefCModel' object has no attribute '%s'" % name)


def test_rig_exempts_only_a_skipped_planner_module_absence():
    assert dvb._rig_exempt(RIG, _err("decoder")) is True
    assert dvb._rig_exempt(RIG, _err("agent_embed")) is True
    assert dvb._rig_exempt(RIG, _err("ego_hist")) is True


def test_rig_exemption_RED_arms():
    assert dvb._rig_exempt(FULL, _err("decoder")) is False             # a NON-rig model that lost its decoder still FAILS
    assert dvb._rig_exempt(RIG, _err("encoder")) is False              # a BUILT module that cannot be read still FAILS
    assert dvb._rig_exempt(RIG, KeyError("decoder")) is False          # only an AttributeError is exempt
    assert dvb._rig_exempt(RIG, AttributeError("no such thing")) is False


BASE = ["--arm", "hier", "--size", "small", "--anchors", "a.pt", "--n-anchors", "117", "--w-agent", "1.0", "--steps", "48852",
        "--out", "x", "--seed", "0", "--refcv8", "--w-r8-cons", "0.05", "--map-hires", "on", "--w-map-hires", "1.0"]


def test_argv_builder_drops_the_planner_keeps_perception_and_sets_the_rig_values():
    a = M.build(BASE, seed=1, out="/o", steps=4000)
    toks = " ".join(a)
    for dropped in ("--anchors", "--n-anchors", "--refcv8", "--w-r8-cons"):
        assert dropped not in a
    assert a[:2] == ["--arm", "flat"] and "--w-agent" in a and "--map-hires" in a
    assert a[a.index("--seed") + 1] == "1" and a[a.index("--steps") + 1] == "4000" and a[a.index("--ema-decay") + 1] == "0.999"
    assert "--r9-perc-rig" in a and a[a.index("--out") + 1] == "/o" and "--conflict-detector off" in toks
    assert a.count("--steps") == 1 and a.count("--seed") == 1


def test_argv_builder_refuses_an_unclassified_flag():
    with pytest.raises(SystemExit):
        M.build(BASE + ["--some-new-refcv9-flag", "1"], seed=0, out="/o")       # neither KEPT nor DROPPED: never passed through
