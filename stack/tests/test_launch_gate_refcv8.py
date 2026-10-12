"""The launch gate's refcv8 profile against the REGISTERED SPEC (SPEC_REFCV8 3.2 / 10.1 / 10.3) and the PI's rulings of
2026-10-11 (Q-A no route checkpoint, Q-B the quantized max-speed label is the speed input).

CPU only, no model build, no data. Every expectation is a LITERAL authored here from the SPEC / the rulings, never
computed from the code under test, and every guard carries a DELIBERATE-REGRESSION arm that must go RED:
  * B-RECIPE-FLAGS  --ema-decay 0.9998 / --b1-guard-ref-ap2m 0.21211 / --b1-guard-step 1000 REQUIRED, exactly
  * B-EVAL-GEOMETRY --batch 16 x --eval-batches 8 (the 128-window in-run eval the B1 reference is defined on), --eval-every 500
  * Q-A / Q-B       --r8-no-rc, no RC flag; the speed input is `--r8-speed-input label` with refcv7's two sidecars
  * B-GS-SMOKE      a conformant launch argv must survive the trainer's OWN pins at the smoke's --steps 100
  * step_cost       a gradient-share reading must not enter the projected s/step
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import launch_gate as LG  # noqa: E402

PROF = LG.PROFILES["refcv8"]
SMOKE_JSON = ROOT / "ops" / "runs.d" / "refcv8-wpb-smoke.argv.json"
TRAIN_SIDECAR = "/home/nvidia/data/refcv6_speed_max_v8_train.jsonl"
EVAL_SIDECAR = "/home/nvidia/data/refcv6_speed_max_v8_eval.jsonl"

#: the deltas that turn the registered WP-B smoke argv into a CONFORMANT launch argv (all literals, authored from the
#: SPEC and the rulings). value None = remove the flag; [] = a bare flag.
DELTAS = {
    "--r8-rc-variant": None, "--r8-rc-noise-along-m": None, "--r8-rc-noise-lat-m": None, "--r8-rc-dropout": None,
    "--r8-no-rc": [],
    "--r8-speed-input": ["label"],
    "--speed-max-sidecar-v6": [TRAIN_SIDECAR], "--speed-max-sidecar-v6-eval": [EVAL_SIDECAR],
    "--ema-decay": ["0.9998"], "--b1-guard-ref-ap2m": ["0.21211"], "--b1-guard-step": ["1000"],
    "--warmup": ["1000"], "--steps": ["48852"], "--batch": ["16"], "--eval-batches": ["8"], "--eval-every": ["500"],
    "--grad-share-every": ["250"], "--grad-share-at": ["1000,16250,32500,48750"],
    # SPEC_REFCV8 A2 (PI 2026-10-12, one full pass over the 4,572-clip train view): the data set, as LITERALS
    "--v2-cache": ["/home/nvidia/data/refcv8-b1-416x1024-train-4572"],
    "--agent-rig-extrinsics": ["/home/nvidia/data/refcv8_train4572_eval139_extrinsics.json"],
    "--pose-sync-sidecar": ["/home/nvidia/data/refcv8_pose_sync_sidecar.jsonl"],
    "--clip-clock-sidecar": ["/home/nvidia/data/refcv8_clip_clock_sidecar_ts33.jsonl"],
    "--r8-v9-labels": ["/home/nvidia/refcv8_v9labels/release_train4572/v9_labels_train.npz"], "--r8-v9-md5": ["83e30bd3791f0767707eade5f4671759"],
    "--vis1-sidecar": ["/home/nvidia/data/refcv8/vis1_sidecar_refcv8b1_train4572_eval139.npz"],
    # SPEC_REFCV8 A3: the nav tau re-derived on the 4,572 view
    "--nav-compliance-tau-file": ["/home/nvidia/data/refcv8/nav_compliance_tau_train4572.json"], "--nav-compliance-tau-rad": ["0.1806411302089691"],
}


def _pairs(a):
    out, i = [], 0
    while i < len(a):
        j = i + 1
        while j < len(a) and not a[j].startswith("--"):
            j += 1
        out.append((a[i], list(a[i + 1:j])))
        i = j
    return out


def _with(argv, flag, vals):
    ps, done = [], False
    for f, v in _pairs(argv):
        if f == flag:
            if not done and vals is not None:
                ps.append((f, [str(x) for x in vals]))
            done = True
        else:
            ps.append((f, v))
    if not done and vals is not None:
        ps.append((flag, [str(x) for x in vals]))
    return [t for f, v in ps for t in [f, *v]]


_OPTS: dict = {}


@pytest.fixture(scope="module")
def launch(tmp_path_factory):
    d = json.loads(SMOKE_JSON.read_text(encoding="utf-8"))
    a = d["argv"] if isinstance(d, dict) else d
    for f, v in DELTAS.items():
        a = _with(a, f, v)
    # FIX-4's banked tau record: an in-repo-schema fixture derived from the argv's own split (it is a FIXTURE for the
    # nav-tau rule, which is not under test here; the profile's refcv8 rules are)
    rec = tmp_path_factory.mktemp("tau") / "nav_tau.json"
    rec.write_text(json.dumps({"status": "OK", "tau": float(LG.flag_values(a, "--nav-compliance-tau-rad")[0]),
                               "split_v2_cache": LG.flag_values(a, "--v2-cache")[0],
                               "labels": LG.flag_values(a, "--v7-labels")[0]}), encoding="utf-8")
    _OPTS["nav_tau_record"] = str(rec)
    return a


def _rules(argv):
    return LG.profile_argv_rules(PROF, argv, _OPTS)


# ------------------------------------------------------------------------------------- pregate B1/B2 ---- #
def test_B1_the_model_job_keeps_the_join_defect_masks_real(tmp_path):
    """pregate 2026-10-12 B1: the refcv8 trainer opens --join-defect-masks while building, so the model job must keep
    it real. refcv7's tuple (the DELIBERATE REGRESSION) sends it to the non-existent sentinel."""
    assert PROF["model_input_flags"] == ("--anchors", "--agent-rig-extrinsics", "--nav-compliance-tau-file",
                                         "--map-hires-class-weights", "--join-defect-masks")
    assert "--join-defect-masks" not in LG.PROFILES["refcv7"]["model_input_flags"]        # refcv7 unchanged
    jdm = "/home/nvidia/refcv6_run/x/stack/tanitad/configs/refcv8_join_label_defects_train4572.json"
    argv = ["--join-defect-masks", jdm, "--v2-cache", "/home/nvidia/data/refcv8-b1-416x1024-train-4572"]
    ctx = types.SimpleNamespace(argv=argv, path_map=[], prof=PROF)
    out, rec = LG._sentinel_argv(ctx, PROF["model_input_flags"], tmp_path)
    assert out[1] == jdm and rec["kept"]["--join-defect-masks"] == jdm
    assert "__gate_unread__" in out[3]                                         # every other path: the sentinel
    out, rec = LG._sentinel_argv(ctx, LG.PROFILES["refcv7"]["model_input_flags"], tmp_path)
    assert "__gate_unread__" in out[1] and "--join-defect-masks" in rec["sentinel"]


_EVO = ROOT / "ops" / "refcv8_eval_remap_a2_thor.json"
_EVO_LIT = {"--agent-rig-extrinsics": "/home/nvidia/data/refcv8_train4572_eval139_extrinsics.json",
            "--clip-clock-sidecar": "/home/nvidia/data/refcv8_clip_clock_sidecar_ts33.jsonl",
            "--vis1-sidecar": "/home/nvidia/data/refcv8/vis1_sidecar_refcv8b1_train4572_eval139.npz",
            "--nav-compliance-tau-file": "/home/nvidia/data/refcv8/nav_compliance_tau_train4572.json"}


def _loader_remap(tmp_path, overrides):
    """refcv7_loader.remap_argv in a FRESH process (it reads REFCV6_REMAP_OVERRIDES at import) on an argv that
    carries every remapped flag: the four A2 files as the profile pins them, the rest under their kit names."""
    kit = tmp_path / "kit"
    kit.mkdir(exist_ok=True)
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); from tanitad.eval import refcv7_loader as L\n"
            "argv = []\n"
            "for f, (bn, _) in L.PATH_REMAP_SPEC.items():\n"
            "    argv += [f, json.loads(sys.argv[2]).get(f, '/home/nvidia/data/' + bn)]\n"
            "try:\n"
            "    out, rec = L.remap_argv(argv)\n"
            "    print(json.dumps({'ok': rec['remapped']}))\n"
            "except SystemExit as e:\n"
            "    print(json.dumps({'refused': str(e)}))\n")
    pins = {f: PROF["required_values"][f] for f in _EVO_LIT}
    env = {k: v for k, v in os.environ.items() if not k.startswith(("REFCV6_", "REFCV7_"))}
    env.update(REFCV6_KIT=str(kit), PYTHONIOENCODING="utf-8")
    if overrides:
        env["REFCV6_REMAP_OVERRIDES"] = str(overrides)
    r = subprocess.run([sys.executable, "-c", code, str(ROOT), json.dumps(pins)], env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_B2_the_recorded_eval_override_maps_exactly_the_four_A2_files(tmp_path):
    """pregate 2026-10-12 B2: the eval loader REFUSES the four A2 files by basename (its kit is refcv7's 4,369
    view); the recorded override maps them to the argv's own Thor files and nothing else."""
    o = json.loads(_EVO.read_text(encoding="utf-8"))
    assert o == _EVO_LIT
    assert all(PROF["required_values"][f] == p for f, p in o.items())        # = the launch argv's own files
    got = _loader_remap(tmp_path, _EVO)
    assert "ok" in got, got
    for f, p in _EVO_LIT.items():
        assert got["ok"][f]["source"] == "recorded override" and got["ok"][f]["local"] == p
    assert {f for f, e in got["ok"].items() if e["source"] == "recorded override"} == set(_EVO_LIT)
    # DELIBERATE REGRESSION: without the override the loader refuses, naming exactly these four flags
    bad = _loader_remap(tmp_path, None)
    assert "refused" in bad, bad
    for f in _EVO_LIT:
        assert f + ": the argv names" in bad["refused"]
    assert bad["refused"].count(": the argv names") == 4


def test_B2_the_prelaunch_passes_the_override_and_reads_A2_files_from_the_argv():
    """The gate command carries the override; the v9 / pose-sync preflights name no 4,369-view file."""
    s = (ROOT / "scripts" / "refcv8_prelaunch.sh").read_text(encoding="utf-8")
    assert '--eval-remap-overrides "$EVO")' in s
    assert 'EVO="$CODE/stack/ops/refcv8_eval_remap_a2_thor.json"' in s
    for stale in ("f63ece410b725febb8a5242cf2b01d3c", "refcv6_pose_sync_sidecar.jsonl",
                  "/home/nvidia/refcv8_v9labels/release/${f%%:*}.npz"):
        assert stale not in s, stale
    assert "--r8-v9-labels:83e30bd3791f0767707eade5f4671759" in s


# ------------------------------------------------------------------------------------------------ A4 ---- #
_CR = "stack/tanitad/configs/refcv8_clock_reference"


def test_A4_the_clock_references_ship_in_stack_and_cover_the_ADD_clips():
    """SPEC_REFCV8 A4: G-CLOCK reads THREE independent references, all under stack/ (a Thor tree ships stack/ +
    taniteval/ only). Counts are LITERALS from the measured gate runs: 4,486 / 4,656 / 4,669 referenced clips."""
    assert PROF["clock_reference"] == (f"{_CR}/q4c_grid_vs_egolog_ALLTRAIN.json", f"{_CR}/q4c_grid_vs_egolog_ADD203.json",
                                       f"{_CR}/q4c_grid_vs_egolog_ADD203_slowregime.json")
    assert PROF["nav_tau_record_default"] == "stack/tanitad/configs/refcv8_nav_compliance_tau_train4572.json"
    assert LG.PROFILES["refcv7"]["clock_reference"][0].endswith("2026-09-26-refcv6-frozen-trunk-audit/raw/"
                                                                 "q4c_grid_vs_egolog_ALLTRAIN.json")   # refcv7 unchanged
    paths = [str(ROOT.parent / r) for r in PROF["clock_reference"]]
    assert LG.load_clock_reference(paths)[1]["n_reference_clips"] == 4669
    # DELIBERATE REGRESSIONS: without the slow regime 13 ADD clips are unreferenced (the 55-clip, 1.203 % FAIL);
    # with ALLTRAIN alone all 203 are (the 225-clip FAIL)
    assert LG.load_clock_reference(paths[:2])[1]["n_reference_clips"] == 4656
    assert LG.load_clock_reference(paths[:1])[1]["n_reference_clips"] == 4486
    rec = json.loads((ROOT.parent / PROF["nav_tau_record_default"]).read_text(encoding="utf-8"))
    assert repr(rec["tau"]) == "0.1806411302089691" == PROF["required_values"]["--nav-compliance-tau-rad"]
    assert rec["inputs"]["manifest"]["path"] == "/home/nvidia/data/refcv8-b1-416x1024-train-4572/_v2manifest.pt"


# ------------------------------------------------------------------------------------------------ A2 ---- #
def test_A2_the_full_pass_rows_are_due_in_the_trainers_own_pin():
    """SPEC_REFCV8 A2: --steps 48852 = ONE epoch over 781,635 windows at batch 16 (drop_last). Every launch
    --grad-share-at row must be a multiple of --grad-share-every 250 and <= --steps, or grad_share.parse_at refuses."""
    gs = pytest.importorskip("tanitad.train.grad_share")
    assert 781635 // 16 == 48852
    assert sorted(gs.parse_at("1000,16250,32500,48750", 250, 48852)) == [1000, 16250, 32500, 48750]
    assert sorted(gs.parse_at("1000,48750", 250, 48852)) == [1000, 48750]
    # DELIBERATE REGRESSIONS: the unrounded thirds and the run end itself are not multiples of 250 -> never due
    for bad in ("1000,16284,32568,48852", "1000,48852"):
        with pytest.raises(ValueError):
            gs.parse_at(bad, 250, 48852)


@pytest.mark.parametrize("flag,old", [
    ("--steps", ["30000"]),                                                         # the 0.614-epoch length
    ("--v2-cache", ["/home/nvidia/data/refcv6-b1-416x1024-train"]),                 # the 4,369-clip view
    ("--clip-clock-sidecar", ["/home/nvidia/data/refcv6_clip_clock_sidecar.jsonl"]),
    ("--pose-sync-sidecar", ["/home/nvidia/data/refcv6_pose_sync_sidecar.jsonl"]),
    ("--agent-rig-extrinsics", ["/home/nvidia/data/refcv6_train_eval139_extrinsics.json"]),
    ("--vis1-sidecar", ["/home/nvidia/data/refcv7/vis1_sidecar_refcv6b1_train4369_eval139.npz"]),
    ("--r8-v9-labels", ["/home/nvidia/refcv8_v9labels/release/v9_labels_train.npz"]),
    ("--r8-v9-md5", ["f63ece410b725febb8a5242cf2b01d3c"]),
    ("--nav-compliance-tau-file", ["/home/nvidia/data/refcv7/nav_compliance_tau_train.json"]),                       # A3: the 4,369-view record
    ("--nav-compliance-tau-rad", ["0.18063741505146028"])])
def test_A2_the_4369_clip_data_set_is_REFUSED(launch, flag, old):
    refusals, _p = _rules(_with(launch, flag, old))
    # the REQUIRED-VALUE refusal itself: an unrelated refusal that merely NAMES the flag (the nav-tau record is
    # keyed by --v2-cache) does not count -- MEASURED, it let the unpatched gate pass the --v2-cache mutation
    assert any(r.startswith(flag + " ") and "is REQUIRED for a refcv8 launch" in r for r in refusals), refusals


def test_the_conformant_launch_argv_has_no_refusal_and_no_pi_pending(launch):
    refusals, pending = _rules(launch)
    assert refusals == [], refusals
    assert pending == [], pending            # Q-A and Q-B CLOSED the two PI items: nothing awaits a PI value


@pytest.mark.parametrize("flag", ["--ema-decay", "--b1-guard-ref-ap2m", "--b1-guard-step"])
def test_B_RECIPE_FLAGS_each_flag_is_required(launch, flag):
    refusals, _ = _rules(_with(launch, flag, None))
    assert any(flag in r for r in refusals), refusals            # deliberate regression: omitted -> RED


@pytest.mark.parametrize("flag,bad", [("--ema-decay", "0.999"), ("--ema-decay", "0.9998000001"),
                                      ("--b1-guard-ref-ap2m", "0.248"), ("--b1-guard-ref-ap2m", "0.22867"),
                                      ("--b1-guard-step", "500"), ("--b1-guard-step", "2000")])
def test_B_RECIPE_FLAGS_other_values_are_refused(launch, flag, bad):
    refusals, _ = _rules(_with(launch, flag, [bad]))
    assert any(flag in r for r in refusals), (flag, bad, refusals)


@pytest.mark.parametrize("flag,bad", [("--batch", "20"), ("--batch", "8"), ("--eval-batches", "4"),
                                      ("--eval-every", "1000"), ("--warmup", "2000"), ("--steps", "12000")])
def test_B_EVAL_GEOMETRY_and_the_run_recipe_are_pinned(launch, flag, bad):
    refusals, _ = _rules(_with(launch, flag, [bad]))
    assert any(flag in r for r in refusals), (flag, bad, refusals)


def test_B_EVAL_GEOMETRY_is_the_product_the_trainer_builds():
    # the trainer's eval loader: n_need = eval_batches * batch (refc_v3_train.py); 8 x 16 = 128 windows, LITERAL
    assert int(PROF["required_values"]["--eval-batches"]) * int(PROF["required_values"]["--batch"]) == 128


@pytest.mark.parametrize("flag,vals", [("--r8-rc-variant", ["A50"]), ("--r8-rc-dropout", ["0.3"]),
                                       ("--r8-rc-noise-along-m", ["2.0"]), ("--r8-rc-noise-lat-m", ["0.75"]),
                                       ("--r8-rc-roll", [])])
def test_Q_A_no_route_checkpoint_every_rc_flag_is_refused(launch, flag, vals):
    refusals, _ = _rules(launch + [flag, *vals])
    assert any(flag in r for r in refusals), (flag, refusals)


def test_Q_A_no_rc_is_required_and_rc_variant_is_no_longer_pi_pending(launch):
    refusals, _ = _rules(_with(launch, "--r8-no-rc", None))
    assert any("--r8-no-rc" in r for r in refusals), refusals
    assert "--r8-rc-variant" not in PROF["required_flags"]
    assert "--r8-rc-variant" not in PROF["pi_pending_values"]
    assert "--r8-speed-input" not in PROF["pi_pending_values"]


@pytest.mark.parametrize("src", ["n2", "n3"])
def test_Q_B_the_past_only_proxies_are_refused(launch, src):
    refusals, _ = _rules(_with(launch, "--r8-speed-input", [src]))
    assert any("--r8-speed-input" in r for r in refusals), refusals


def test_Q_B_the_label_sidecars_are_admitted_and_required(launch):
    assert LG.has_flag(launch, "--speed-max-sidecar-v6") and LG.has_flag(launch, "--speed-max-sidecar-v6-eval")
    assert _rules(launch)[0] == []                                 # the lifted forbidden-lever entry stays lifted
    assert not any("--speed-max-sidecar" in str(names) for names, _w in PROF["forbidden_levers"])
    for flag in ("--speed-max-sidecar-v6", "--speed-max-sidecar-v6-eval"):
        refusals, _ = _rules(_with(launch, flag, None))
        assert any(flag in r for r in refusals), (flag, refusals)           # missing -> RED
    refusals, _ = _rules(_with(launch, "--speed-max-sidecar-v6", ["/x/other_train.jsonl"]))
    assert any("--speed-max-sidecar-v6" in r for r in refusals), refusals   # not refcv7's sidecar -> RED


def test_the_refcv7_family_is_untouched():
    # a refcv7 launch still forbids nothing about the sidecar and still pins its own values
    assert "--r8-speed-input" not in LG.PROFILES["refcv7"]["required_values"]
    assert LG.PROFILES["refcv7"]["pi_pending_values"] == {}


# ---------------------------------------------------------------------------------------- B-GS-SMOKE ---- #
def test_B_GS_SMOKE_the_smoke_argv_survives_the_trainers_own_gradshare_pin(launch):
    gs = pytest.importorskip("tanitad.train.grad_share")
    # the MEASURED defect: the launch's rows are past a 100-step smoke -> the trainer's pin REFUSES them
    with pytest.raises(ValueError):
        gs.parse_at("1000,10000,20000,30000", 250, 100)
    old = _with(launch, "--steps", ["100"])                         # the pre-fix rewrite: --steps only
    at = LG.flag_values(old, "--grad-share-at")[0]
    with pytest.raises(ValueError):                                 # deliberate regression: the old rewrite is RED
        gs.parse_at(at, int(LG.flag_values(old, "--grad-share-every")[0]), 100)
    new, dep = LG.smoke_argv(launch, 100, 50)
    assert LG.flag_values(new, "--steps") == ["100"]
    assert LG.flag_values(new, "--grad-share-every") == ["50"] and LG.flag_values(new, "--grad-share-at") == ["50"]
    assert gs.parse_at("50", 50, 100) == frozenset({50})
    assert 100 % 50 == 0 and "--grad-share-every/--grad-share-at" in dep


def test_B_GS_SMOKE_the_b1_guard_step_is_brought_inside_the_smoke(launch):
    new, dep = LG.smoke_argv(launch, 100, 50)
    assert LG.flag_values(launch, "--b1-guard-step") == ["1000"]
    assert LG.flag_values(new, "--b1-guard-step") == ["100"]       # the trainer refuses a guard step past --steps
    assert "--b1-guard-step" in dep
    assert LG.flag_values(new, "--b1-guard-ref-ap2m") == ["0.21211"]   # the guard stays ARMED, with the same reference
    assert LG.flag_values(new, "--ema-decay") == ["0.9998"]


def test_B_GS_SMOKE_the_emission_switch_runs_inside_the_smoke(launch):
    """MM 2026-10-11: the launch's step-1000 emission switch must execute in a 100-step smoke."""
    assert LG.flag_values(launch, "--r8-alloc-emit-start") == ["1000"]
    new, dep = LG.smoke_argv(launch, 100, 50)
    assert LG.flag_values(new, "--r8-alloc-emit-start") == ["50"]
    assert "--r8-alloc-emit-start" in dep
    # DELIBERATE REGRESSION: the pre-fix rewrite (--steps only) leaves emission off for the whole smoke
    old = _with(launch, "--steps", ["100"])
    assert int(LG.flag_values(old, "--r8-alloc-emit-start")[0]) >= 100


def test_B_GS_SMOKE_nothing_is_rewritten_that_is_already_inside_or_off(launch):
    off = _with(_with(launch, "--grad-share-every", None), "--grad-share-at", None)
    off = _with(_with(off, "--grad-share-terms", None), "--b1-guard-step", ["50"])
    off = _with(off, "--r8-alloc-emit-start", ["40"])
    new, dep = LG.smoke_argv(off, 100, 50)
    assert not LG.has_flag(new, "--grad-share-every") and not LG.has_flag(new, "--grad-share-at")
    assert LG.flag_values(new, "--b1-guard-step") == ["50"]            # already inside the smoke: untouched
    assert LG.flag_values(new, "--r8-alloc-emit-start") == ["40"]      # already inside the smoke: untouched
    assert set(dep) == {"--steps"}


@pytest.mark.parametrize("argv,rows", [
    (["--grad-share-every", "250", "--grad-share-at", "1000,10000", "--steps", "30000"], [1000, 10000]),
    (["--grad-share-every", "50", "--grad-share-at", "50", "--steps", "100"], [50]),
    (["--grad-share-every", "50", "--steps", "200"], [50, 100, 150, 200]),
    (["--grad-share-every", "0", "--steps", "200"], []),
    (["--steps", "200"], [])])
def test_gs_reading_rows_literals(argv, rows):
    assert LG.gs_reading_rows(argv) == rows


def test_step_cost_excludes_the_gradient_share_reading():
    # 100 steps at exactly 10.0 s, except that the reading (row 50) sits inside interval 49 and costs 1000 s extra
    t = [0.0]
    for j in range(99):
        t.append(t[-1] + (1010.0 if j == 49 else 10.0))
    polluted = LG.step_cost(t, 10)                                  # deliberate regression: no exclusion -> RED
    assert polluted["s_per_step_mean"] > 20.0
    fixed = LG.step_cost(t, 10, exclude_rows=[50])
    assert fixed["s_per_step_mean"] == 10.0
    assert fixed["s_per_step_max"] == 10.0
    assert fixed["reading_intervals_excluded"] == 3 and fixed["reading_rows"] == [50]
    # the reading on the OTHER side of the step's timestamp (interval 48) is excluded as well
    t2 = [0.0]
    for j in range(99):
        t2.append(t2[-1] + (1010.0 if j == 48 else 10.0))
    assert LG.step_cost(t2, 10, exclude_rows=[50])["s_per_step_mean"] == 10.0
    # without readings the function is the tip's: same keys, no extras
    assert "reading_rows" not in LG.step_cost(t, 10)


def test_b1_reach_judge():
    armed = ["--b1-guard-ref-ap2m", "0.21211", "--b1-guard-step", "100"]
    row = {"step": 100, "b1_guard": "keep", "b1_ema_box3d_ap2m": 0.22, "b1_ref_box3d_ap2m": 0.21211}
    assert LG.judge_b1_reach(armed, [{"step": 50}, row])[0] == []
    assert len(LG.judge_b1_reach(armed, [{"step": 50}, {"step": 100}])[0]) == 1   # armed, never decided -> RED
    assert LG.judge_b1_reach(["--steps", "100"], [{"step": 50}])[0] == []          # not armed -> inert
