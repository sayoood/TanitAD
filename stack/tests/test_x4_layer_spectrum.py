"""X4 — the per-layer spectrum monitor, PROVED to fire per layer.

⛔ WHY THIS FILE EXISTS. V6_TRAINING_MEASURES.md's X4 row is "per-layer
SIGReg + per-layer spectrum monitors (O6 pattern at T/S scale) | rank
retention per layer". Until 2026-08-16 only z_op was monitored, and the O6
constants CANNOT be copied down:

* z_tac / z_str contribute ONE row per window (the uplink reads only the
  window's last frame), so a per-batch reading has ceiling B-1 = **7** on the
  live geometry — not 47;
* z_op's ceiling_min = 1024 EXCEEDS what a centred covariance over z_str
  (d=256) can EVER reach — copying it would make the strategic layer
  INCONCLUSIVE FOREVER by construction.

The per-layer constants are MEASURED (x4_layer_power.py, artifact
`…/2026-08-16-x4-p9/raw/x4_layer_power.json`): tac (d=512) ceiling_min 256 /
floor 32 (separation 5.89x, pair-FP 0.000 at ceiling 263; floor margins
2.06x / 2.86x), str (d=256) 128 / 32 (5.40x, 0.000; 1.86x / 2.90x) — and the
same selection rules RE-DERIVE z_op's shipped 1024 / 64, which is the check
that the derivation is one lineage, not a fork.

The guard discipline is inherited from ``test_o6_spectrum_power.py``:
every per-layer verdict is watched FIRE on synthetic collapse, watched NOT
fire on health, and watched say INCONCLUSIVE at the live per-batch n. The
no-change proof is CONTENT-anchored (C75), never HEAD.
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.models.v6 import (  # noqa: E402
    O6_ADMISSIBLE_CEILING, O6_RANK_FLOOR, X4_LAYER_POLICY,
    X4_RECOMMENDED_ACCUM, LayerSpectrumMonitor, layer_spectrum_policy,
    sigreg_trend_verdict, spectrum_report, x4_rank_verdict)

#: The live geometry: z_tac/z_str rows per step = --batch (8), drawn from
#: --eps-per-batch (4) episodes, ONE frame per row (last-frame uplink).
LIVE_B, LIVE_EPS = 8, 4
D_TAC, D_STR = 512, 256


def _rows(d: int, steps: int, gen: torch.Generator, keep: int | None = None,
          floor: float = 1e-2) -> torch.Tensor:
    """``[steps*8, d]`` with the tac/str sampler geometry: 8 one-frame rows
    per step over 4 interleaved episodes (variance split 0.5 episode / 0.5
    idiosyncratic — the calibrated regime's rho_ep with the window factor
    merged into the row, since one row IS one window here).

    ``keep=k`` is a SQUEEZE (x ``floor``), not a truncation — a truncation is
    the easiest possible thing to detect and would flatter the instrument."""
    scale = torch.ones(d, dtype=torch.float64)
    if keep is not None:
        scale[keep:] = floor
    a = math.sqrt(0.5)
    out = []
    for _ in range(steps):
        ge = torch.randn(LIVE_EPS, d, generator=gen, dtype=torch.float64)
        gr = torch.randn(LIVE_B, d, generator=gen, dtype=torch.float64)
        idx = torch.arange(LIVE_B) % LIVE_EPS
        out.append(((a * ge[idx] + a * gr) * scale).float())
    return torch.cat(out)


def _reading(d, steps, gen, keep=None, floor=1e-2, ci=8):
    return spectrum_report(_rows(d, steps, gen, keep=keep, floor=floor),
                           ci_reps=ci, block=LIVE_B, generator=gen)


# =========================================================================== #
# 1. the policy IS the measured one — and refuses to invent numbers
# =========================================================================== #
def test_policy_constants_are_the_measured_ones():
    """The numbers in x4_layer_power.json and the code are one thing."""
    assert layer_spectrum_policy("tac", 512)["ceiling_min"] == 256
    assert layer_spectrum_policy("tac", 512)["floor"] == 32.0
    assert layer_spectrum_policy("str", 256)["ceiling_min"] == 128
    assert layer_spectrum_policy("str", 256)["floor"] == 32.0
    # the op anchors are the UNCHANGED O6 constants — one derivation lineage
    assert layer_spectrum_policy("op", 2048)["ceiling_min"] \
        == O6_ADMISSIBLE_CEILING == 1024
    assert layer_spectrum_policy("op", 2048)["floor"] == O6_RANK_FLOOR == 64.0
    for pol in X4_LAYER_POLICY.values():
        assert "basis" in pol       # every constant names its measurement


def test_policy_refuses_to_invent_a_floor_for_an_unmeasured_d():
    """⛔ C76: a threshold ships with the FP rate it achieves, or it does not
    ship. A re-configured d gets the RULE ceiling and NO floor."""
    pol = layer_spectrum_policy("tac", 300)
    assert pol["ceiling_min"] == 128            # largest pow2 <= 150
    assert pol["floor"] is None
    assert "RULE ONLY" in pol["basis"]
    v = x4_rank_verdict("tac", 300, _reading(300, 20,
                                             torch.Generator().manual_seed(3),
                                             ci=0))
    assert v["absolute_floor"] is None
    assert "clause 3 DISABLED" in v["floor_note"]


def test_str_could_never_reach_the_op_ceiling():
    """The reason per-layer constants exist at all, pinned: min(n-1, d) for
    d=256 is bounded by 256 < 1024 however many rows are pooled."""
    assert min(10 ** 9, D_STR) < O6_ADMISSIBLE_CEILING


def test_recommended_accum_makes_all_layers_adjudicable():
    """32*8-1 = 255 is ONE ROW short of tac's 256; 33 clears every layer."""
    assert 32 * LIVE_B - 1 < X4_LAYER_POLICY["tac"]["ceiling_min"]
    assert X4_RECOMMENDED_ACCUM * LIVE_B - 1 \
        >= X4_LAYER_POLICY["tac"]["ceiling_min"]
    assert X4_RECOMMENDED_ACCUM == 33


# =========================================================================== #
# 2. ⭐ each layer's guard FIRES — and does not fire on health
# =========================================================================== #
def test_tac_guard_FIRES_on_clause_2_with_the_floor_genuinely_idle():
    """⭐ THE RETENTION PATH, per layer: a keep-8 squeeze of z_tac reads ~42
    (ABOVE the floor 32, so clause 3 stays idle) and the jackknife retention
    interval sits wholly below 0.8 — the FAIL comes from the interval."""
    gen = torch.Generator().manual_seed(5)
    ref = _reading(D_TAC, 40, gen)
    cur = _reading(D_TAC, 40, gen, keep=8)
    v = x4_rank_verdict("tac", D_TAC, cur, ref)
    assert v["status"] == "FAIL" and "clause 2" in v["reason"]
    assert v["effective_rank"] > v["absolute_floor"], \
        "fixture drifted: the floor must stay idle for this to prove clause 2"
    assert v["retention_ci95"]["hi"] < 0.8
    assert v["layer"] == "tac" and v["d"] == D_TAC


def test_str_guard_FIRES_on_a_collapsed_representation():
    gen = torch.Generator().manual_seed(6)
    ref = _reading(D_STR, 20, gen)
    cur = _reading(D_STR, 20, gen, keep=8)
    v = x4_rank_verdict("str", D_STR, cur, ref)
    assert v["status"] == "FAIL" and v["pass"] is False
    assert ("clause 2" in v["reason"]) or ("clause 3" in v["reason"])


def test_tac_guard_DOES_NOT_fire_on_a_healthy_representation():
    """A real PASS, not merely the absence of a FAIL."""
    gen = torch.Generator().manual_seed(5)
    ref = _reading(D_TAC, 40, gen)
    cur = _reading(D_TAC, 40, gen)
    v = x4_rank_verdict("tac", D_TAC, cur, ref)
    assert v["status"] == "PASS" and v["pass"] is True, v["reason"]
    assert v["retention"] == pytest.approx(1.0, abs=0.12)


def test_per_batch_n8_is_INCONCLUSIVE_by_construction_for_every_layer():
    """⛔ THE HEADLINE, one level up: a single z_tac/z_str batch has ceiling
    B-1 = 7, and the verdict says so instead of producing a number."""
    gen = torch.Generator().manual_seed(7)
    for layer, d in (("tac", D_TAC), ("str", D_STR)):
        r = spectrum_report(_rows(d, 1, gen))
        assert r["rank_ceiling"] == LIVE_B - 1 == 7
        v = x4_rank_verdict(layer, d, r)
        assert v["status"] == "INCONCLUSIVE" and v["pass"] is None
        assert "cannot resolve rank" in v["reason"]


# =========================================================================== #
# 3. the monitor — per-layer rings, references, layer admissibility
# =========================================================================== #
def test_monitor_stamps_LAYER_admissibility_next_to_the_o6_one():
    """⛔ `spectrum_report` stamps `rank_admissible` against the z_op constant
    (1024) unconditionally — for a d=256 layer that is a misreading trap. The
    layer record must carry BOTH, labelled."""
    mon = LayerSpectrumMonitor({"str": D_STR}, accum=20,
                               rows_per_step={"str": LIVE_B})
    gen = torch.Generator().manual_seed(9)
    for _ in range(20):
        mon.push({"str": _rows(D_STR, 1, gen)})
    rec = mon.emit({"str": _rows(D_STR, 1, gen)}, step=200)["str"]
    pooled = rec["spectrum_pooled"]
    assert pooled["rank_ceiling"] == 20 * LIVE_B - 1 == 159
    assert pooled["rank_admissible"] is False          # the O6/z_op constant
    assert pooled["rank_admissible_layer"] is True     # 159 >= 128
    assert "BINDING ADMISSIBILITY FOR THIS LAYER" in \
        pooled["admissibility_note"]


def test_monitor_takes_the_reference_at_first_LAYER_admissible_reading():
    mon = LayerSpectrumMonitor({"str": D_STR}, accum=20,
                               rows_per_step={"str": LIVE_B})
    gen = torch.Generator().manual_seed(10)
    z = _rows(D_STR, 1, gen)
    r1 = mon.emit({"str": z}, step=200)["str"]         # ring empty: no pooled
    assert "spectrum_pooled" not in r1
    assert mon.references == {}
    for _ in range(20):
        mon.push({"str": _rows(D_STR, 1, gen)})
    r2 = mon.emit({"str": _rows(D_STR, 1, gen)}, step=400)["str"]
    assert r2["reference_taken_at_step"] == 400
    assert mon.references["str"]["ref_step"] == 400
    r3 = mon.emit({"str": _rows(D_STR, 1, gen)}, step=600)["str"]
    assert "reference_taken_at_step" not in r3         # taken once
    assert r3["verdict"]["reference_effective_rank"] == pytest.approx(
        mon.references["str"]["effective_rank"])


def test_monitor_reports_an_absent_layer_instead_of_dropping_it():
    """Rule 2: absence is declared, never silent."""
    mon = LayerSpectrumMonitor({"tac": D_TAC, "str": D_STR})
    rec = mon.emit({"tac": torch.randn(8, D_TAC)})
    assert rec["str"]["status"] == "n/a"
    assert "not present" in rec["str"]["reason"]
    assert "spectrum" in rec["tac"]


def test_monitor_accepts_the_raw_forward_output_keys_too():
    """A caller handing ``V6Stack.forward``'s own dict (keys ``z_tac`` /
    ``z_str``) must get real records, not silent n/a rows — the near-miss
    that motivated this was CAUGHT by the n/a record, and this closes the
    class."""
    mon = LayerSpectrumMonitor({"tac": D_TAC, "str": D_STR}, accum=2,
                               rows_per_step={"tac": LIVE_B,
                                              "str": LIVE_B})
    fwd = {"z_tac": torch.randn(8, D_TAC), "z_str": torch.randn(8, D_STR)}
    mon.push(fwd)
    rec = mon.emit(fwd, step=1)
    for k in ("tac", "str"):
        assert "spectrum" in rec[k]
        assert "spectrum_pooled" in rec[k]
        assert rec[k]["spectrum"]["n"] == 8


def test_monitor_refuses_an_empty_layer_set():
    with pytest.raises(ValueError, match="at least one layer"):
        LayerSpectrumMonitor({})


def test_monitor_default_emit_consumes_NO_global_rng():
    """Same contract as spectrum_report: monitoring must not be able to move
    the run's loss."""
    mon = LayerSpectrumMonitor({"tac": D_TAC}, accum=2,
                               rows_per_step={"tac": LIVE_B})
    z = torch.randn(8, D_TAC)
    torch.manual_seed(99)
    before = torch.randn(3)
    torch.manual_seed(99)
    mon.push({"tac": z})
    mon.emit({"tac": z}, step=1)
    assert torch.equal(torch.randn(3), before)


# =========================================================================== #
# 4. the o6 TREND guard — the 42.4-sigma alarm, watched fire and not fire
# =========================================================================== #
#: MEASURED (O6_ABLATION_AND_MASK_PROBE.md §4.1): healthy o6_sigreg
#: 0.4023 +- 0.0195 on the live 48-row batch; the SAME batch 2x-collapsed
#: reads 1.2283 — 42.4 sigma. The tests replay exactly those levels.
O6_HEALTHY_MEAN, O6_HEALTHY_SD, O6_2X_COLLAPSED = 0.4023, 0.0195, 1.2283


def _series(n, gen, mean=O6_HEALTHY_MEAN, sd=O6_HEALTHY_SD):
    return [mean + sd * float(x) for x in torch.randn(n, generator=gen)]


def test_trend_guard_FIRES_at_the_measured_collapse_level():
    gen = torch.Generator().manual_seed(20)
    v = sigreg_trend_verdict(_series(400, gen), [O6_2X_COLLAPSED] * 100)
    assert v["status"] == "FAIL" and v["pass"] is False
    assert v["z"] > 20        # the measured response is ~42 sigma


def test_trend_guard_DOES_NOT_fire_on_healthy_noise():
    gen = torch.Generator().manual_seed(21)
    v = sigreg_trend_verdict(_series(400, gen), _series(100, gen))
    assert v["status"] == "PASS" and abs(v["z"]) < 4


def test_trend_guard_never_fires_on_a_FALLING_o6():
    """A falling o6 is the regulariser training — the healthy direction."""
    gen = torch.Generator().manual_seed(22)
    v = sigreg_trend_verdict(_series(400, gen), [0.25] * 100)
    assert v["status"] == "PASS" and v["z"] < 0


def test_trend_guard_is_INCONCLUSIVE_when_undersampled():
    v = sigreg_trend_verdict([0.4] * 10, [0.4] * 2)
    assert v["status"] == "INCONCLUSIVE" and v["pass"] is None


def test_trend_guard_floors_a_degenerate_zero_variance_baseline():
    """An exactly-constant baseline must not manufacture an infinite z: the
    scale floor (2 % of |median|) binds, is stamped, and a small rise stays
    a PASS while a collapse-sized one still FAILS."""
    small = sigreg_trend_verdict([0.4] * 64, [0.41] * 16)
    assert small["status"] == "PASS" and small["scale_floored"] is True
    big = sigreg_trend_verdict([0.4] * 64, [1.23] * 16)
    assert big["status"] == "FAIL"


# =========================================================================== #
# 5. trainer wiring — flag, per-layer trend block, gate passthrough
# =========================================================================== #
def _trainer():
    import train_v6_staged as T
    return T


def test_x4_flag_default_builds_tac_and_str_and_op_is_refused():
    T = _trainer()
    from tanitad.models.v6 import V6Config
    cfg = V6Config()
    a = T.build_parser().parse_args(["--stage", "S-W", "--out", "x"])
    assert a.x4_spectrum_layers == "tac,str"
    mon = T.x4_monitor_from_args(a, cfg)
    assert mon.layers == {"tac": cfg.d_tac, "str": cfg.d_str}
    assert mon.rows_per_step == {"tac": a.batch, "str": a.batch}
    a2 = T.build_parser().parse_args(["--stage", "S-W", "--out", "x",
                                      "--x4-spectrum-layers", "none"])
    assert T.x4_monitor_from_args(a2, cfg) is None
    a3 = T.build_parser().parse_args(["--stage", "S-W", "--out", "x",
                                      "--x4-spectrum-layers", "op"])
    with pytest.raises(SystemExit, match="incumbent O6 monitor"):
        T.x4_monitor_from_args(a3, cfg)


def test_trend_record_says_why_tac_and_str_have_no_trend_guard():
    """⛔ 'wire the trend guard per layer WHERE A PER-LAYER SIGREG LOSS
    EXISTS; where it does not, say so rather than inventing one.' There is no
    per-layer SigReg for tac/str (v6_loss_step applies stack.sigreg to z_op
    only), so their block is an EXPLICIT not-applicable, never a number."""
    T = _trainer()
    gen = torch.Generator().manual_seed(30)
    rec = T.x4_trend_record(_series(400, gen), _series(100, gen))
    assert rec["op"]["applicable"] is True
    assert rec["op"]["status"] in ("PASS", "FAIL")
    for layer in ("tac", "str"):
        assert rec[layer]["applicable"] is False
        assert "no per-layer SIGReg loss exists" in rec[layer]["reason"]
    empty = T.x4_trend_record([], [])
    assert empty["op"]["applicable"] is False
    assert "nothing to baseline" in empty["op"]["reason"]


def test_S_W_gate_spec_reports_x4_with_the_per_layer_ceilings():
    T = _trainer()
    spec = T.STAGE_GATE_SPEC["S-W"]
    assert "X4_spectrum_layers" in spec["reported"]
    assert "X4_spectrum_layers" not in spec["required"]
    assert spec["owners"]["X4_spectrum_layers"] \
        == "tanitad.models.v6.LayerSpectrumMonitor"
    crit = spec["criteria"]["X4_rank_retention"]
    assert "tac 256" in crit and "str 128" in crit and "1024" in crit


def test_run_stage_gate_carries_the_x4_layers_probe(tmp_path):
    """The gate artifact must carry the per-layer records under a REPORTED
    probe — visibility, never adjudication."""
    T = _trainer()
    from tanitad.config import (EncoderConfig, PredictorConfig,
                                ReadoutConfig)
    from tanitad.models.v6 import V6Config, V6Stack
    tiny = V6Stack(V6Config(
        encoder=EncoderConfig(in_channels=3, image_size=64, patch_size=16,
                              d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=2, d_readout=16),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1,)),
        d_tac=32, d_str=16, d_goal_embed=8, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=16, f_blocks=1, n_candidates=2))
    x4 = {"tac": {"spectrum": {"effective_rank": 5.0, "rank_ceiling": 7}}}
    gate = T.run_stage_gate(tiny, "S-W", out_dir=tmp_path, x4_spectra=x4,
                            dry_run=True)
    probe = gate["probes"]["X4_spectrum_layers"]
    assert probe["status"] == "reported" and probe["pass"] is None
    assert probe["layers"] == x4
    assert "33 recommended" in probe["reason"]
    on_disk = json.loads((tmp_path / "stage_gate.json").read_text())
    assert on_disk["probes"]["X4_spectrum_layers"]["layers"] == x4


def test_x4_gate_probe_absent_is_stubbed_not_silent(tmp_path):
    """Without x4_spectra the probe appears as not-run — rule 2 again."""
    T = _trainer()
    from tanitad.config import (EncoderConfig, PredictorConfig,
                                ReadoutConfig)
    from tanitad.models.v6 import V6Config, V6Stack
    tiny = V6Stack(V6Config(
        encoder=EncoderConfig(in_channels=3, image_size=64, patch_size=16,
                              d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=2, d_readout=16),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1,)),
        d_tac=32, d_str=16, d_goal_embed=8, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=16, f_blocks=1, n_candidates=2))
    gate = T.run_stage_gate(tiny, "S-W", out_dir=tmp_path, dry_run=True)
    assert gate["probes"]["X4_spectrum_layers"]["status"] == "not-run"


# =========================================================================== #
# 6. ⛔ the no-change proof — CONTENT-anchored, never HEAD (C75)
# =========================================================================== #
_PRE_CHANGE_MARKER = "X4_LAYER_POLICY"
_V6_REL = "stack/tanitad/models/v6.py"


def _pre_x4_module():
    """The newest v6.py revision that does NOT yet carry X4 — the code the
    incumbent monitors were running. HEAD would compare the file with itself
    the moment anyone commits (C75)."""
    root = _STACK.parent
    try:
        log = subprocess.run(["git", "log", "--format=%H", "--", _V6_REL],
                             cwd=root, capture_output=True, timeout=180)
        if log.returncode != 0:
            return None
        src = None
        for sha in log.stdout.decode().split():
            r = subprocess.run(["git", "show", f"{sha}:{_V6_REL}"], cwd=root,
                               capture_output=True, timeout=120)
            if r.returncode != 0 or not r.stdout:
                continue
            if _PRE_CHANGE_MARKER.encode() in r.stdout:
                continue
            src = r.stdout
            break
        if src is None:
            return None
    except Exception:
        return None
    tmp = Path(tempfile.mkdtemp()) / "v6_pre_x4.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location("v6_pre_x4", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v6_pre_x4"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def test_spectrum_report_and_o6_verdict_UNCHANGED_vs_the_pre_X4_revision():
    """⛔ X4 must be PURELY ADDITIVE on the O6 path: bit-equal records from
    ``spectrum_report`` (no keys added, none moved) and identical verdicts
    from ``o6_rank_verdict`` — the live run reads both."""
    prev = _pre_x4_module()
    if prev is None:
        pytest.skip("git could not supply a pre-X4 v6.py revision")
    for shape in [(48, 2048), (8, 512), (8, 256), (2, 5)]:
        torch.manual_seed(7)
        z = torch.randn(*shape)
        old = prev.spectrum_report(z)
        new = spectrum_report(z)
        assert old == new, f"spectrum_report moved at {shape}"
        vo = prev.o6_rank_verdict(old)
        vn = __import__("tanitad.models.v6", fromlist=["o6_rank_verdict"]
                        ).o6_rank_verdict(new)
        assert vo == vn, f"o6_rank_verdict moved at {shape}"
