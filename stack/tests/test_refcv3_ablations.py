"""⭐⭐ EVERY PRE-REGISTERED EVAL-TIME ABLATION IS RUNNABLE, AND EVERY FLAG BITES.

``Project Steering/PREREG_REFCV4B_HIERARCHY_EVAL.md`` §3 registers TWELVE
ablation arms for the post-training hierarchy panel. Before Rung A1 (2026-09-05)
three had a flag (``--with-navzero``, ``--no-navshuf``, ``--with-navflip``) and
FULL needs none — the other EIGHT could not be run from the command line at all.

⚠️ The prereg's own §7 escalation lists only SEVEN of the eight. It omits the
frame-blind DELIBERATE REGRESSION, which is the arm the panel's validity rests
on: **a gate that has never been shown to FAIL an image-blind arm certifies
nothing** (H-ECHO-4, where an ADE-scored gate once passed an echoing arm). §3 is
the authoritative list and this file pins the bijection against it.

WHAT IS PINNED

  (1) COMPLETENESS — every §3 row has a live CLI route, and the tool's registry
      is a BIJECTION with §3's twelve. A missing switch is a missing arm.
  (2) THE FLAGS PARSE — asserted by actually parsing, not by reading the source.
  (3) ⭐⭐ EVERY FLAG CHANGES THE FORWARD PATH. "A flag that parses but does
      nothing is worse than a missing one": it produces a table that reads like
      a knockout and is a copy of the FULL arm. Each ablation must move at
      least one banked array, and the test NAMES which.
  (4) WHERE A SWITCH WOULD BE INERT, THE TOOL REFUSES — `sel_refined` at
      diffusion steps 0 (refc.py:1596 leaves `refined is conf` by
      construction), `ego_zero` on a build fed no ego block, `gstr_shuffle`
      without a bank, hier-only edges on a flat build. A refusal is a result;
      a silent no-op is a fabricated one.
  (5) ⭐ THE DELIBERATE REGRESSION'S INTERNAL CONTROL: `ha`, `ha0` and
      `ha0_ext` read no frames, so under `--ablate-frames` they must come back
      BIT-IDENTICAL to the FULL run while `os` moves. If the controls moved,
      the ablation leaked somewhere it should not have.
  (6) THE STAMP TRAVELS. The manifest carries the ablation AND
      ``analyze_refcv3`` copies it onto EVERY per-arm block, so a number can
      never be read without knowing which ablation produced it. A dump whose
      manifest predates the stamp reads UNKNOWN, never "none".
  (7) TWO REGIMES IN ONE ``--dump-dir`` ARE REFUSED.

⛔ CPU only, random-init tiny model, synthetic 3-clip corpus; never touches
Thor, a pod, or a real checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_refcv3_arm as base                            # noqa: E402

rc = base.rc
from tanitad.refs import refc_v3 as v3                    # noqa: E402

_REPO = Path(__file__).resolve().parents[2]

#: ⭐ THE PRE-REGISTERED TWELVE — ``PREREG_REFCV4B_HIERARCHY_EVAL.md`` §3, the
#: "arm" column verbatim. Written HERE, in the test, so the pin survives a
#: reader who cannot reach the prereg file; the file itself is cross-checked
#: below when it is reachable.
PREREG_SECTION_3_ARMS = {
    "FULL", "nav-ZERO", "nav-SHUFFLE", "nav-FLIP", "g_str-ZERO",
    "g_str-SHUFFLE", "E7-OFF", "E9-OFF", "H19-OFF", "EGO-ZERO", "SEL-REFINED",
    "DELIBERATE REGRESSION",
}


# --------------------------------------------------------------------------- #
# fixtures — a tiny build whose seams are LIVE                                  #
# --------------------------------------------------------------------------- #
def _make(root: Path, *, hier: bool = True, v4: bool = False,
          gate: float | None = 0.7, steps: int | None = None):
    """A random-init tiny checkpoint. ``gate`` opens E9 on purpose: ``goal_gate``
    is ZERO-INIT, so on an untouched fixture ``e9_off`` would be a genuine
    no-op and the test would pin nothing."""
    eps = root / "eps"
    eps.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        base._write_v2ep(eps / f"{base.CLIPS[i]}.v2ep.pt", i)
    lp = root / "labels_v72_eval.jsonl"
    lp.write_text("\n".join(json.dumps(base._record(i)) for i in range(3)),
                  encoding="utf-8")
    run = root / "run"
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    cfg = base._tiny_cfg(hier)
    if v4:
        cfg.ego_state_inject = True
        cfg.core.ego_valid_channel = True
    if steps is not None:
        cfg.core.decoder.diffusion_steps = int(steps)
    model = v3.RefCV3Model(cfg)
    if hier and gate is not None:
        with torch.no_grad():
            model.goal_gate.fill_(float(gate))
    torch.save({"step": 11, "model": model.state_dict(), "opt": {}},
               run / "ckpt.pt")
    (run / "config.json").write_text(json.dumps({
        "arm": "hier" if hier else "flat",
        "horizons": list(cfg.core.trajectory.horizons),
        "goal_tau_steps": list(cfg.goal_tau_steps),
        "image_hw": list(cfg.core.encoder.image_hw()),
        "tac_vocab_version": cfg.tac_vocab_version,
        "nav_from_v7": True,
        "nav_cmd_derivation": "v7.2 nav_command token (test fixture)",
        "refcv3_arm_model_cfg": json.loads(
            json.dumps(rc.dataclasses.asdict(cfg))),
    }, indent=1), encoding="utf-8")
    return eps, lp, run / "ckpt.pt"


def _args(root: Path, ck: Path, eps: Path, lp: Path, dump: Path, **over):
    """The FAST roll: one fed conditioning, no oracle arm, stride 4. The
    ablations are model-level, so a smaller grid pins them just as hard."""
    a = vars(base._args(root, ck, eps, lp))
    a.update(dump_dir=str(dump), window_stride=4, no_navshuf=True,
             no_navzero=True, with_oracle_sel=False, with_navflip=False,
             ablate=[], ablate_frames=False, gstr_bank=None,
             gstr_shuffle_seed=0)
    a.update(over)
    return argparse.Namespace(**a)


def _arrays(dump: Path) -> dict:
    """Every banked array, arms AND decisions sidecar, concatenated."""
    out = {}
    for f in sorted(dump.glob("ep*.npz")):
        with np.load(f) as d:
            for k in d.files:
                out.setdefault(f"arm:{k}", []).append(d[k])
    for f in sorted((dump / "decisions").glob("ep*.npz")):
        with np.load(f) as d:
            for k in d.files:
                out.setdefault(f"dec:{k}", []).append(d[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def _moved(full: dict, abl: dict) -> list[str]:
    """The keys whose banked values are NOT bit-identical between two rolls."""
    moved = []
    for k, v in full.items():
        w = abl.get(k)
        if w is None or w.shape != v.shape or not np.array_equal(v, w):
            moved.append(k)
    return sorted(moved)


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    """FULL reference roll on a hier build whose E9 gate is open."""
    root = tmp_path_factory.mktemp("abl_live")
    eps, lp, ck = _make(root)
    dump = root / "dump_full"
    a = _args(root, ck, eps, lp, dump)
    man = rc.run_dump(a)
    return root, eps, lp, ck, dump, man, _arrays(dump)


# =========================================================================== #
# (1) completeness — every pre-registered ablation has a route                 #
# =========================================================================== #
def test_the_registry_is_a_bijection_with_the_prereg_twelve():
    covered = {s["prereg_arm"] for s in rc.ABLATIONS.values()}
    covered |= set(rc.ABLATIONS_PREEXISTING)
    assert covered == PREREG_SECTION_3_ARMS, (
        f"missing a route for {sorted(PREREG_SECTION_3_ARMS - covered)}; "
        f"unregistered extras {sorted(covered - PREREG_SECTION_3_ARMS)}")
    # ⚠️ the eighth is the one the prereg's own §7 escalation forgets
    assert any(s["prereg_arm"] == "DELIBERATE REGRESSION"
               for s in rc.ABLATIONS.values()), (
        "the frame-blind deliberate regression has no flag — the panel cannot "
        "show that its gate is capable of FAILING an image-blind arm")
    for name, spec in rc.ABLATIONS.items():
        for key in ("prereg_arm", "mechanism", "seen_in_training", "tests",
                    "applies_at", "requires"):
            assert spec.get(key), f"{name} carries no {key!r}"


def test_the_prereg_file_when_reachable_names_the_same_twelve():
    """A bonus cross-check: the pin above is the test, this catches a prereg
    edit that adds a thirteenth arm without a switch."""
    p = _REPO / "Project Steering" / "PREREG_REFCV4B_HIERARCHY_EVAL.md"
    try:
        txt = p.read_text(encoding="utf-8")
    except OSError as ex:                    # the G: mount drops files
        pytest.skip(f"prereg unreachable ({type(ex).__name__}); the bijection "
                    f"pin above still ran: {p}")
    assert "PRE-REGISTRATION" in txt, "control read failed"
    for arm in PREREG_SECTION_3_ARMS:
        needle = arm if arm != "DELIBERATE REGRESSION" else "DELIBERATE REGRESSION"
        assert needle in txt, f"§3 row {arm!r} not found in the prereg"


# =========================================================================== #
# (2) the flags parse                                                          #
# =========================================================================== #
@pytest.mark.parametrize("name", sorted(rc.ABLATIONS))
def test_each_ablation_name_is_accepted_by_the_cli(name):
    with pytest.raises(SystemExit) as ei:
        rc.main(["--ablate", name])
    # it got past argparse and died on the NEXT required thing
    assert "--out is required" in str(ei.value), (
        f"--ablate {name} did not parse: {ei.value!r}")


def test_the_frame_blind_regression_has_its_own_named_flag():
    with pytest.raises(SystemExit) as ei:
        rc.main(["--ablate-frames"])
    assert "--out is required" in str(ei.value)
    a = argparse.Namespace(ablate=[], ablate_frames=True)
    assert rc.resolve_ablations(a) == ["frames_blind"]
    a2 = argparse.Namespace(ablate=["frames_blind"], ablate_frames=True)
    assert rc.resolve_ablations(a2) == ["frames_blind"], "not deduplicated"


def test_an_unknown_ablation_is_refused_by_the_cli():
    with pytest.raises(SystemExit) as ei:
        rc.main(["--ablate", "make_it_better"])
    assert ei.value.code == 2                       # argparse's own refusal
    with pytest.raises(SystemExit):
        rc.resolve_ablations(argparse.Namespace(ablate=["nope"],
                                                ablate_frames=False))


# =========================================================================== #
# (3) ⭐⭐ every flag changes the forward path                                  #
# =========================================================================== #
@pytest.mark.parametrize("name", ["gstr_zero", "e7_off", "e9_off", "h19_off",
                                  "sel_refined", "frames_blind"])
def test_each_ablation_moves_the_record(live, name, tmp_path):
    root, eps, lp, ck, _dump, _man, full = live
    dump = tmp_path / f"dump_{name}"
    rc.run_dump(_args(root, ck, eps, lp, dump, ablate=[name]))
    moved = _moved(full, _arrays(dump))
    assert moved, (
        f"ablation {name!r} left EVERY banked array bit-identical to the FULL "
        f"roll. A flag that parses but does nothing is worse than a missing "
        f"one: it produces a table that reads like a knockout and is a copy.")
    print(f"[{name}] moved {len(moved)} arrays: {moved[:8]}")


def test_frames_blind_moves_the_model_but_not_the_model_free_controls(live,
                                                                     tmp_path):
    """⭐ THE INTERNAL CONTROL ON THE DELIBERATE REGRESSION. `ha`, `ha0` and
    `ha0_ext` are integrated from the recorded ego state and read NO frames, so
    blinding the camera must leave them bit-identical. If they moved, the
    ablation leaked into the controls and every margin computed against them
    would be measuring the leak."""
    root, eps, lp, ck, _dump, _man, full = live
    dump = tmp_path / "dump_blind"
    rc.run_dump(_args(root, ck, eps, lp, dump, ablate_frames=True))
    got = _arrays(dump)
    for ctrl in ("arm:ha", "arm:ha0", "arm:ha0_ext", "arm:g", "arm:v0",
                 "arm:ws"):
        assert np.array_equal(full[ctrl], got[ctrl]), (
            f"{ctrl} MOVED under the frame-blind ablation — it reads no "
            f"frames, so the intervention leaked")
    assert not np.array_equal(full["arm:os"], got["arm:os"]), (
        "the model arm did not move when the camera was blinded")


def test_ego_zero_moves_a_v4_build_and_is_refused_on_a_v3_one(tmp_path):
    root = tmp_path / "v4"
    eps, lp, ck = _make(root, v4=True)
    full = tmp_path / "d_full"
    rc.run_dump(_args(root, ck, eps, lp, full))
    abl = tmp_path / "d_egozero"
    rc.run_dump(_args(root, ck, eps, lp, abl, ablate=["ego_zero"]))
    assert _moved(_arrays(full), _arrays(abl)), "ego_zero changed nothing"
    # and the controls, which keep the MEASURED v0, must not have moved
    assert np.array_equal(_arrays(full)["arm:ha0"], _arrays(abl)["arm:ha0"])

    root3 = tmp_path / "v3"
    eps3, lp3, ck3 = _make(root3, v4=False)
    with pytest.raises(SystemExit) as ei:
        rc.run_dump(_args(root3, ck3, eps3, lp3, tmp_path / "d3",
                          ablate=["ego_zero"]))
    assert "ego_state_inject" in str(ei.value)


# =========================================================================== #
# (4) where a switch would be inert, the tool REFUSES                          #
# =========================================================================== #
def test_sel_refined_is_refused_where_it_would_be_a_silent_no_op(tmp_path):
    """refc.py:1596 — at ``steps == 0`` the refined readout IS the confidence by
    construction, and :1630 gates ``score_emitted`` on ``steps > 0``."""
    root = tmp_path / "steps0"
    eps, lp, ck = _make(root, steps=0)
    with pytest.raises(SystemExit) as ei:
        rc.run_dump(_args(root, ck, eps, lp, tmp_path / "d0",
                          ablate=["sel_refined"]))
    assert "steps > 0" in str(ei.value)
    assert "worse than a missing one" in str(ei.value)


def test_gstr_shuffle_refuses_without_a_bank(live, tmp_path):
    root, eps, lp, ck, _dump, _man, _full = live
    with pytest.raises(SystemExit) as ei:
        rc.run_dump(_args(root, ck, eps, lp, tmp_path / "d", ablate=["gstr_shuffle"]))
    msg = str(ei.value)
    assert "--gstr-bank" in msg
    assert "ACROSS WINDOWS" in msg, (
        "the refusal must say WHY a batch permutation would be the wrong "
        "intervention here")


def test_a_hier_only_ablation_is_refused_on_a_flat_build(tmp_path):
    root = tmp_path / "flat"
    eps, lp, ck = _make(root, hier=False, gate=None)
    with pytest.raises(SystemExit) as ei:
        rc.run_dump(_args(root, ck, eps, lp, tmp_path / "df",
                          ablate=["gstr_zero"]))
    assert "HIER build" in str(ei.value)


def test_e9_off_reports_itself_INERT_when_the_gate_never_opened(tmp_path):
    """⭐ ``goal_gate`` is ZERO-INIT. On a checkpoint whose gate never opened,
    knocking E9 out CANNOT change the forward — and that is a finding about the
    checkpoint (Caveat-B), not a licence to report 'no effect, seam inert'."""
    root = tmp_path / "gate0"
    eps, lp, ck = _make(root, gate=0.0)
    dump = tmp_path / "d_e9"
    man = rc.run_dump(_args(root, ck, eps, lp, dump, ablate=["e9_off"]))
    ev = man["ablation"]["per_ablation"]["e9_off"]
    assert ev["goal_gate_before"] == 0.0
    assert "INERT" in ev, (
        "an ablation that cannot change anything must SAY so in the record")
    assert "Caveat-B" in ev["INERT"]


# =========================================================================== #
# (5) the g_str shuffle really injects ANOTHER WINDOW's goal                    #
# =========================================================================== #
def test_gstr_shuffle_injects_the_banked_goal_of_the_permuted_window(live,
                                                                    tmp_path):
    root, eps, lp, ck, dump_full, man_full, _full = live
    dump = tmp_path / "d_shuf"
    man = rc.run_dump(_args(root, ck, eps, lp, dump, ablate=["gstr_shuffle"],
                            gstr_bank=str(dump_full), gstr_shuffle_seed=0))
    st = man["ablation"]["per_ablation"]["gstr_shuffle"]["bank"]
    assert st["n_windows"] == man_full["grid"]["n_windows"]
    assert st["frac_changed"] > 0.0, "the permutation is the identity"
    # the injection was VERIFIED against the model's own emitted g_str
    v = man["ablation"]["verified"]["g_str"]
    assert v["max_abs_err"] < 1e-4
    assert man["ablation"]["n_gstr_windows"] == man["grid"]["n_windows"]
    # and every emitted goal is one the FULL roll produced SOMEWHERE
    bank = {tuple(np.round(g, 4))
            for g in _arrays(dump_full)["dec:gstr_nav_true"]}
    got = _arrays(dump)["dec:gstr_nav_true"]
    hits = sum(1 for g in got if tuple(np.round(g, 4)) in bank)
    assert hits == len(got), (
        f"only {hits}/{len(got)} shuffled goals came from the bank — the "
        f"injection is not reproducing banked values")


def test_gstr_zero_emits_the_straight_ahead_constant(live, tmp_path):
    root, eps, lp, ck, _dump, _man, _full = live
    dump = tmp_path / "d_gz"
    man = rc.run_dump(_args(root, ck, eps, lp, dump, ablate=["gstr_zero"]))
    assert man["ablation"]["verified"]["g_str"]["injected"] == [1.0, 0.0, 0.0]
    g = _arrays(dump)["dec:gstr_nav_true"]
    assert np.allclose(g, np.array([1.0, 0.0, 0.0], dtype=np.float32),
                       atol=1e-4), (
        "the emitted strategic goal is not the straight-ahead constant the "
        "prereg registers for this arm")


# =========================================================================== #
# (6) the stamp travels to every arm block                                     #
# =========================================================================== #
def test_the_ablation_stamp_reaches_the_manifest_and_every_arm_block(live,
                                                                    tmp_path):
    root, eps, lp, ck, _dump, _man, _full = live
    dump = tmp_path / "d_stamp"
    man = rc.run_dump(_args(root, ck, eps, lp, dump,
                            ablate=["h19_off", "e7_off"]))
    assert man["ablation"]["applied"] == ["h19_off", "e7_off"]
    assert (dump / "ABLATION.txt").read_text(encoding="utf-8").strip() == \
        "h19_off, e7_off"
    rec = rc.analyze_refcv3(str(dump), n_boot=40, seed=0)
    assert rec["refcv3"]["ablation"]["applied"] == ["h19_off", "e7_off"]
    assert rec["arms"], "no arms analysed"
    for arm, blk in rec["arms"].items():
        assert blk["ablation"]["applied"] == ["h19_off", "e7_off"], arm


def test_the_FULL_run_is_stamped_as_FULL_and_a_legacy_dump_reads_UNKNOWN(live):
    _root, _eps, _lp, _ck, dump, man, _full = live
    assert man["ablation"]["applied"] == []
    assert "FULL" in man["ablation"]["is"]
    rec = rc.analyze_refcv3(str(dump), n_boot=40, seed=0)
    assert rec["refcv3"]["ablation"]["applied"] == []
    # a manifest with no stamp must read UNKNOWN, never "none"
    man_p = Path(dump) / "manifest.json"
    saved = man_p.read_text(encoding="utf-8")
    try:
        d = json.loads(saved)
        d.pop("ablation")
        man_p.write_text(json.dumps(d, indent=1), encoding="utf-8")
        rec2 = rc.analyze_refcv3(str(dump), n_boot=40, seed=0)
        abl = rec2["refcv3"]["ablation"]
        assert abl["applied"] is None
        assert "UNKNOWN" in abl["⛔ provenance"]
        assert "absence of the record" in abl["⛔ provenance"]
        for blk in rec2["arms"].values():
            assert blk["ablation"]["applied"] is None
    finally:
        man_p.write_text(saved, encoding="utf-8")


# =========================================================================== #
# (7) two regimes in one dump dir                                              #
# =========================================================================== #
def test_a_second_regime_in_one_dump_dir_is_refused(live, tmp_path):
    root, eps, lp, ck, _dump, _man, _full = live
    dump = tmp_path / "d_mix"
    rc.run_dump(_args(root, ck, eps, lp, dump, ablate=["h19_off"]))
    with pytest.raises(SystemExit) as ei:
        rc.run_dump(_args(root, ck, eps, lp, dump, ablate=["e7_off"]))
    assert "unreadable mixture" in str(ei.value)
    # the SAME regime is allowed to be re-rolled in place
    rc.run_dump(_args(root, ck, eps, lp, dump, ablate=["h19_off"]))
