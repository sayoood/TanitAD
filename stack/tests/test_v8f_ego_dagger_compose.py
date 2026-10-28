"""AMEND1 (DAG-ORC re-based on DAG-OACC) section 1: ``--ego-perturb-with-dagger`` in ``stack/scripts/v8f_mini_train.py`` -- the DAgger swap and O-RECOVER's ego perturbation in ONE arm.

The flag is default-OFF. THIS file proves the composition is REACHABLE from the real caller (CLAUDE.md, 2026-09-10/11: "built, tested, and unreachable from its caller" is a class), hop by hop:

  argv --ego-perturb-with-dagger  ->  make_compose  ->  the EgoDaggerCompose stage  ->  declared_vs_built (bad_dagger lifts the exclusivity; bad_ego_dagger reads the BUILT stage back)
        ->  ego_dagger_compose_probe (G-LIVE on a COMPOSED batch: ego_perturb_probe on it, both levers reach the loss gradient, the p = 0 control reads EXACTLY 0.0)
        ->  ``b = comp.apply(b, step)`` in the loop (swap first, the perturbation only on the samples the swap did not replace)  ->  the batch the LOSS receives
        ->  summary.json / selfcheck.json / train_log.jsonl / the checkpoint (ego_dagger_overlap_n, ego_perturb_with_dagger, dagger_n / dagger_eligible, ego_perturb_n / ego_perturb_eligible)

and that with the flag OFF every existing argv builds and trains bit-identically (K-BUILD): ``ego_dagger_compose_golden.json`` was written by the UNMODIFIED tip (66b6c80e) on the three argv families
below; this tree must reproduce it key for key and bit for bit. Section 7 (2026-10-28) proves the perturbation draws from a stream INDEPENDENT of the swap's even at equal seeds (the box-2 defect: 0.2492 instead of 0.3). ``mutation.txt`` (the deliverable beside this file) deletes one rule at a time and requires this file to go RED.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np  # noqa: F401
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts", REPO / "stack" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_dagger as DG  # noqa: E402,F401
from tanitad.data import v8f_mini_data as D  # noqa: E402,F401
from tanitad.models import v8f_mini as M  # noqa: E402,F401
import v8f_mini_train as TR  # noqa: E402
from test_v8f_dagger_train import COMMON, DAG, _log, _run, _selfcheck, _summary, gate_parts, pack, spy  # noqa: E402,F401  (the fixtures: ONE trainer-written checkpoint, the DATA generator's bank, the synthetic world; the loss spy)

torch.set_num_threads(1)
GOLDEN = Path(__file__).with_name("ego_dagger_compose_golden.json")
WRITE_GOLDEN = os.environ.get("EGO_DAGGER_COMPOSE_WRITE_GOLDEN") == "1"          # set ONLY on the unmodified tip, to (re)write the golden
COUNTER_KEYS = ("ego_perturb_n", "ego_perturb_eligible", "ego_perturb_rows", "ego_perturb_nav_kept", "dagger_n", "dagger_eligible", "dagger_rows", "dagger_nav_kept")
ORC_LAW = ["--ego-perturb-vscale", "0.5", "1.0", "--ego-perturb-vmax", "8.0", "--ego-perturb-adrop", "1.5", "--ego-perturb-yawrate-std", "2.0"]       # PREREG Amendment 3's law (the seed is the arm's)
FT = ["--lr", "1e-4", "--p-gt-start", "0.2", "--p-gt-end", "0.2"]                  # the fine-tune's own optimiser line, as the DAG / ORC chains write it (warmup is COMMON's 2: a 4-step run)


def argv_family(pk, name):
    """The three argv families whose flag-off behaviour must not move: DAG-OACC-style (bank + swap + the acceleration weight), ORC-style (the perturbation's law flags, no bank), plain M2b."""
    if name == "dag":
        return ["--steps", "4", *FT, *DAG(pk, 0.3, 3), "--w-acc", "49.6969"]
    if name == "orc":
        return ["--steps", "4", *FT, "--init-ckpt", str(pk.ck), "--ego-perturb-p", "0.3", *ORC_LAW, "--ego-perturb-seed", "7"]
    assert name == "m2b"
    return ["--steps", "4"]


def _tensor_digest(h, name, t):
    t = t.detach().cpu().contiguous()
    h.update(name.encode())
    h.update(str(t.dtype).encode())
    h.update(t.numpy().tobytes())


def digest(out):
    """Everything a run's identity is made of: every logged number (minus wall-clock), the final weights, the optimiser state, the counters (summary + checkpoint) and the SETS of keys of each record."""
    out = Path(out)
    ck = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)
    hm, ho = hashlib.sha256(), hashlib.sha256()
    for k in sorted(ck["model"]):
        _tensor_digest(hm, k, ck["model"][k])
    for i in sorted(ck["opt"]["state"]):
        for kk in sorted(ck["opt"]["state"][i]):
            _tensor_digest(ho, f"{i}.{kk}", torch.as_tensor(ck["opt"]["state"][i][kk]))
    s = _summary(out)
    return {"log": [{k: v for k, v in r.items() if k != "sec"} for r in _log(out)], "model_sha256": hm.hexdigest(), "opt_sha256": ho.hexdigest(),
            "ego_totals": ck["ego_totals"], "dagger_totals": ck["dagger_totals"], "counters": {k: s[k] for k in COUNTER_KEYS},
            "summary_keys": sorted(s), "levers_keys": sorted(s["levers"]), "ckpt_keys": sorted(ck), "levers_ego": s["levers"]["ego_perturb"], "levers_dagger_p": s["levers"]["dagger_p"]}


def bank_identity(pk):
    """The fixture bank's identity: the content sha256 of each shard (bank.json's own sha256 embeds the tmp path of the run that wrote it, so it is not a stable identity)."""
    return [s["content_sha256"] for s in json.loads((pk.bank / "bank.json").read_text(encoding="utf-8"))["shards"]]


def test_flag_off_reproduces_the_unmodified_tip_bit_for_bit(pack, tmp_path):
    """K-BUILD: the golden was written by the unmodified tip (66b6c80e) for (a) a DAG-OACC-style argv, (b) an ORC-style argv and (c) plain M2b -- 4 CPU steps each on the trainer's synthetic world. With the flag
    absent this tree must give the same log (every loss, every logged key), the same final weights and optimiser state, the same counters, and the same SET of keys in summary.json and in the checkpoint."""
    got = {}
    for name in ("dag", "orc", "m2b"):
        assert TR.main(_run(tmp_path / name, argv_family(pack, name))) == 0
        got[name] = digest(tmp_path / name)
    got["_fixture"] = {"bank_content_sha256": bank_identity(pack), "torch": torch.__version__}
    if WRITE_GOLDEN:
        GOLDEN.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        pytest.skip("golden written from the unmodified tip")
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert want["_fixture"]["bank_content_sha256"] == bank_identity(pack), "the fixture bank is the golden's bank, record for record (the DATA generator and the trainer's synthetic world are unchanged)"
    for name in ("dag", "orc", "m2b"):
        g, w = got[name], want[name]
        assert g["log"] == w["log"] and len(w["log"]) == 4, f"{name}: every logged number and every logged key"
        assert (g["model_sha256"], g["opt_sha256"]) == (w["model_sha256"], w["opt_sha256"]), f"{name}: final weights and optimiser state, bit for bit (torch {want['_fixture']['torch']} wrote the golden, {torch.__version__} read it)"
        assert (g["ego_totals"], g["dagger_totals"], g["counters"]) == (w["ego_totals"], w["dagger_totals"], w["counters"]), f"{name}: the counters"
        assert g["summary_keys"] == w["summary_keys"] and g["ckpt_keys"] == w["ckpt_keys"], f"{name}: a flag-off run's summary.json / checkpoint carry no new key"
        assert g["levers_ego"] == w["levers_ego"] and g["levers_dagger_p"] == w["levers_dagger_p"]
        assert set(g["levers_keys"]) - set(w["levers_keys"]) <= {"ego_perturb_with_dagger", "ego_dagger_overlap_n", "ego_perturb_stream"} and set(w["levers_keys"]) <= set(g["levers_keys"]), f"{name}: levers only GAIN the three flag fields"
    assert want["dag"]["counters"]["dagger_n"] > 0 and want["orc"]["counters"]["ego_perturb_n"] > 0, "the golden is not vacuous: the DAG family swapped rows and the ORC family perturbed rows"
    assert want["orc"]["counters"]["dagger_n"] == 0 and want["dag"]["counters"]["ego_perturb_n"] == 0 and want["m2b"]["counters"]["dagger_n"] == want["m2b"]["counters"]["ego_perturb_n"] == 0


# ------------------------------------------------------------------------------------------------ helpers
TRAIN_P = REPO / "stack" / "scripts" / "v8f_mini_train.py"
DATA_P = REPO / "stack" / "tanitad" / "data" / "v8f_mini_data.py"
FILES = ("selfcheck.json", "argv.json", "ckpt_last.pt", "summary.json", "train_log.jsonl")


def COMPOSED(pk, p_dag=0.4, p_ego=0.5, dseed=3, eseed=7, flag=True):
    """AMEND1's A-arm argv family on the synthetic world: the DAG argv (init, swap p, bank, seed), the perturbation at the law's flags, and (flag) --ego-perturb-with-dagger."""
    return [*DAG(pk, p_dag, dseed), "--ego-perturb-p", str(p_ego), *ORC_LAW, "--ego-perturb-seed", str(eseed), *(["--ego-perturb-with-dagger"] if flag else [])]


def _nothing_written(out):
    for f in FILES:
        assert not (Path(out) / f).exists(), f"a refused launch writes no {f}"


def _same(x, y):
    """[B] bool: row equal (NaN == NaN)."""
    same = ((x == y) | (torch.isnan(x) & torch.isnan(y))) if x.is_floating_point() else (x == y)
    return same.reshape(x.shape[0], -1).all(1)


def world_batch(pk):
    """(a collated batch of EVERY training window of the trainer's synthetic world at hmax 16, the DAgger bank, the windows)."""
    J = pk.J
    store = D.ClipStore(pk.world, "train", lru=8, kin_mode="yaw", tracks=D.load_native_tracks(pk.world / "native_tracks.pt"))
    wins = D.train_windows(store, J, D.select_train_clips(list(map(str, J["clip"])), 1.0))
    ds = D.V8FMiniWindows(store, J, wins, hmax=16)
    batch = D.collate([ds.item(w[0], w[1], 1, 3) for w in wins])
    return batch, DG.DaggerBank.load(pk.bank, clip_names=J["clip"]), wins


def make_stage(bank, p_dag=0.1, p_ego=0.3, dseed=3, eseed=7, vmax=8.0):
    ego = D.EgoPerturb(p=p_ego, vscale=(0.5, 1.0), vmax=vmax, adrop=1.5, yawrate_std_deg=2.0, seed=eseed, kin_mode="yaw")
    dag = DG.DaggerSwap(bank, p=p_dag, seed=dseed, kin_mode="yaw")
    return TR.EgoDaggerCompose(ego, dag, on=True)


# ------------------------------------------------------------------------------------------------ 1. the refusals
def test_the_flag_parses_defaults_off_and_has_its_gdvb_entry():
    assert TR.build_parser().parse_args(["--out", "x"]).ego_perturb_with_dagger is False
    assert TR.build_parser().parse_args(["--out", "x", "--ego-perturb-with-dagger"]).ego_perturb_with_dagger is True
    assert TR.FLAG_CHECKS["ego_perturb_with_dagger"] == "compose:ego_perturb_with_dagger"


def test_without_the_flag_the_old_exclusivity_still_fires(pack, tmp_path):
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ declared-vs-built: .*never combined"):
        TR.main(_run(tmp_path / "bad", ["--steps", "2", *COMPOSED(pack, flag=False)]))
    _nothing_written(tmp_path / "bad")


@pytest.mark.parametrize("extra, msgs", [
    (lambda pk: ["--init-ckpt", str(pk.ck), "--dagger-p", "0", "--ego-perturb-p", "0.3", "--ego-perturb-with-dagger"], ["BUILT swap is off"]),                  # the swap at p = 0
    (lambda pk: [*DAG(pk, 0.3, 3), "--ego-perturb-p", "0", "--ego-perturb-with-dagger"], ["BUILT perturbation is off"]),                                         # the perturbation at p = 0, named
    (lambda pk: [*DAG(pk, 0.3, 3), "--ego-perturb-with-dagger"], ["BUILT perturbation is off"]),                                                                   # the perturbation not named (its default is 0)
    (lambda pk: ["--ego-perturb-with-dagger"], ["BUILT perturbation is off", "BUILT swap is off"]),                                                                # neither
])
def test_the_flag_with_either_lever_at_p_zero_is_refused_as_a_lever_declared_without_effect(pack, tmp_path, extra, msgs):
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ declared-vs-built: ") as ei:
        TR.main(_run(tmp_path / "bad", ["--steps", "2", *extra(pack)]))
    assert "a lever that is declared and does nothing" in str(ei.value) and all(m in str(ei.value) for m in msgs), str(ei.value)
    _nothing_written(tmp_path / "bad")


# ------------------------------------------------------------------------------------------------ 2. G-DVB reads the flag back from the BUILT stage
def _gate(pk, extra, make=None):
    a, cfg, model, opt, info, dag = gate_parts(pk, extra)[:6]
    ego = TR.make_ego_perturb(a)
    comp = TR.make_compose(a, ego, dag) if make is None else make(a, ego, dag)
    return TR.declared_vs_built(a, cfg, model, opt, info, None, ego, TR.LRSchedule(a.lr, a.warmup, 10), dag, comp)


def test_gdvb_the_control_reads_clean_and_the_flag_is_read_back_from_the_built_stage(pack):
    assert _gate(pack, COMPOSED(pack)) == []
    assert _gate(pack, COMPOSED(pack, flag=False))[0].count("never combined") == 1, "the same argv without the flag: the exclusivity"


@pytest.mark.parametrize("name, flag, make, msg", [
    ("built_off_declared_on", True, lambda a, ego, dag: TR.EgoDaggerCompose(ego, dag, on=False), "declared True but built False"),
    ("built_on_declared_off", False, lambda a, ego, dag: TR.EgoDaggerCompose(ego, dag, on=True), "declared False but built True"),
    ("no_stage_built", True, lambda a, ego, dag: None, "no composed stage was built"),
    ("another_perturber", True, lambda a, ego, dag: TR.EgoDaggerCompose(TR.make_ego_perturb(a), dag, on=True), "does not hold the perturber and the swap"),
])
def test_gdvb_deliberate_regression_a_stage_that_is_not_the_declared_one_is_refused(pack, name, flag, make, msg):
    bad = _gate(pack, COMPOSED(pack, flag=flag), make)
    assert any(msg in m for m in bad), bad


def test_gdvb_a_flag_without_an_entry_is_refused_by_the_gate_itself(pack, monkeypatch):
    monkeypatch.delitem(TR.FLAG_CHECKS, "ego_perturb_with_dagger")
    bad = _gate(pack, COMPOSED(pack))
    assert any("flags without a declared-vs-built entry" in m and "ego_perturb_with_dagger" in m for m in bad), bad


# ------------------------------------------------------------------------------------------------ 3. G-LIVE on a composed batch
def test_glive_a_composed_argv_passes_the_gate_and_the_selfcheck_carries_the_composed_probe(pack, tmp_path):
    assert TR.main(_run(tmp_path / "g", ["--steps", "4", *COMPOSED(pack), "--selfcheck-only"])) == 0
    sc = _selfcheck(tmp_path / "g")
    pr = sc["ego_dagger_compose_probe"]
    assert pr["pass"] is True and pr["overlap"] == pr["overlap_recomputed"] == 0 and pr["n_swapped"] >= 1 and pr["n_perturbed"] >= 1
    assert pr["probe_step"] < 0 and pr["dagger_p_probe"] == 0.5 and pr["ego_p_probe"] == 1.0
    for k in ("control_repeat_loss_diff", "control_repeat_grad_diff", "control_ego_p0_loss_diff", "control_ego_p0_grad_diff", "control_both_p0_loss_diff", "control_both_p0_grad_diff", "control_prior_grad_diff", "control_wm_grad_diff"):
        assert pr[k] == 0.0, f"{k} must read EXACTLY 0.0"
    assert pr["control_both_p0_same_object"] is True
    for k in ("grad_diff_swap_alone", "grad_diff_ego_alone", "grad_diff_swap_inside_composed", "grad_diff_ego_inside_composed"):
        assert pr[k] > 0.0, f"{k}: both levers reach the gradient, alone and inside the composed batch"
    assert pr["loss_diff_swap_inside_composed"] != 0.0 and pr["loss_diff_ego_inside_composed"] != 0.0
    ep = pr["ego_perturb_probe_on_composed"]
    assert ep["n_selected_at_p1"] == pr["n_perturbed"] > 0 and pr["n_swapped"] > 0, "the existing probe ran on the composed selection: the rows it perturbs are the rows the composition perturbs, none of them swapped"
    assert ep["pass"] is True and ep["control_p0_same_object"] is True and ep["control_p0_loss_diff"] == 0.0 and ep["control_p0_grad_diff"] == 0.0 and ep["grad_diff_all"] > 0.0 and ep["loss_diff"] != 0.0, \
        "the EXISTING ego_perturb_probe, run on the composed (swapped) batch with the swap's mask excluded"
    assert sc["ego_perturb_probe"]["pass"] is True and sc["dagger_probe"]["pass"] is True, "the single-lever probes still run on the logged batch"
    assert sc["levers"]["ego_perturb_with_dagger"] is True and sc["levers"]["ego_dagger_overlap_n"] == 0


def test_glive_a_flag_off_gate_runs_no_composed_probe(pack, tmp_path):
    assert TR.main(_run(tmp_path / "g", ["--steps", "2", *DAG(pack, 0.3, 3), "--selfcheck-only"])) == 0
    sc = _selfcheck(tmp_path / "g")
    assert sc["ego_dagger_compose_probe"]["ego_dagger_compose"] == "off" and sc["ego_dagger_compose_probe"]["pass"] is True
    assert sc["levers"]["ego_perturb_with_dagger"] is False and sc["levers"]["ego_dagger_overlap_n"] == 0


@pytest.mark.parametrize("kind", ["no_perturbation", "no_swap", "no_exclusion"])
def test_glive_deliberate_regression_the_gate_refuses_a_composed_path_that_drops_a_rule(pack, tmp_path, monkeypatch, kind):
    orig = TR.EgoDaggerCompose.apply

    def broken(self, batch, step, *, ego_p=None, dag_p=None):
        if kind == "no_perturbation":
            return self.dag.apply(batch, step, p=dag_p)                                                   # the perturbation's apply is deleted from the composed path
        if kind == "no_swap":
            return self.ego.apply(batch, step, p=ego_p)                                                   # the swap's apply is deleted from the composed path
        swapped = self.dag.apply(batch, step, p=dag_p)                                                    # the exclusion is deleted (and the guard with it)
        out = self.ego.apply(swapped, step, p=ego_p)
        self.overlap = 0
        return out
    monkeypatch.setattr(TR.EgoDaggerCompose, "apply", broken)
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ ego-dagger-compose: "):
        TR.main(_run(tmp_path / "bad", ["--steps", "2", *COMPOSED(pack)]))
    assert not (tmp_path / "bad" / "selfcheck.json").exists() and not (tmp_path / "bad" / "summary.json").exists()
    assert orig is not broken


# ------------------------------------------------------------------------------------------------ 4. EgoPerturb.apply(.., exclude=)
def test_ego_exclude_none_is_the_old_path_and_the_mask_changes_nothing_but_the_eligible_set(pack):
    batch, _, _ = world_batch(pack)
    B = int(batch["t"].shape[0])
    ego = D.EgoPerturb(p=0.6, vscale=(0.5, 1.0), vmax=8.0, adrop=1.5, yawrate_std_deg=2.0, seed=5, kin_mode="yaw")
    a = ego.apply(batch, 3)
    la = dict(ego.last)
    for ex in (None, torch.zeros(B, dtype=torch.bool)):
        b = ego.apply(batch, 3, exclude=ex)
        assert all(bool(_same(a[k], b[k]).all()) for k in D.EGO_PERTURB_REBUILT) and ego.last["n_eligible"] == la["n_eligible"] and ego.last["n_selected"] == la["n_selected"], "no exclusion = the old path"
    ex = torch.from_numpy(np.random.default_rng(0).random(B) < 0.5)
    c = ego.apply(batch, 3, exclude=ex)
    lc = dict(ego.last)
    assert bool(lc["sel"].equal(la["sel"] & ~ex)) and not bool((lc["sel"] & ex).any()), "an excluded sample is never selected; every other sample is selected exactly as without the mask (the draws are untouched)"
    assert lc["n_eligible"] == int((ego.eligible(batch) & ~ex).sum()) < la["n_eligible"], "n_eligible counts the NON-excluded eligible samples"
    for k in D.EGO_PERTURB_REBUILT:
        assert bool(_same(c[k], a[k])[~ex].all()) and bool(_same(c[k], batch[k])[ex].all()), f"{k}: non-excluded rows as without the mask, excluded rows as logged"
    allx = ego.apply(batch, 3, exclude=torch.ones(B, dtype=torch.bool))
    assert ego.last["n_selected"] == 0 and ego.last["n_eligible"] == 0 and all(bool(_same(allx[k], batch[k]).all()) for k in D.EGO_PERTURB_REBUILT)
    assert ego.apply(batch, 3, p=0.0, exclude=ex) is batch and ego.last["n_eligible"] == lc["n_eligible"] and ego.last["n_selected"] == 0, "the p = 0 control: the same object, the same denominator"
    with pytest.raises(ValueError, match="exclude has shape"):
        ego.apply(batch, 3, exclude=torch.zeros(B + 1, dtype=torch.bool))


# ------------------------------------------------------------------------------------------------ 5. the composition
def test_composition_overlap_is_exactly_zero_the_shares_are_the_declared_p_and_the_rows_are_each_lever_alone(pack):
    batch, bank, _ = world_batch(pack)
    comp = make_stage(bank, p_dag=0.1, p_ego=0.3)
    ego, dag = comp.ego, comp.dag
    el_logged = ego.eligible(batch)                                                                       # the LOGGED v0 < vmax mask (the swap changes kin3 on swapped rows only)
    tot = dict(sw=0, dag_el=0, pe=0, ego_el=0, protected=0, differs=0)
    for step in range(150):
        out = comp.apply(batch, step)
        sw, pe = dag.last["sel"], ego.last["sel"]
        assert comp.overlap == 0 and int((sw & pe).sum()) == 0, f"step {step}: no sample is both swapped and perturbed"
        assert ego.last["n_eligible"] == int((el_logged & ~sw).sum()), f"step {step}: the perturbation's eligible count = the NON-swapped eligible samples, recomputed from the logged batch"
        assert bool((pe <= (el_logged & ~sw)).all()), f"step {step}: a perturbed sample is an eligible, non-swapped one"
        tot["sw"] += dag.last["n_selected"]
        tot["dag_el"] += dag.last["n_eligible"]
        tot["pe"] += ego.last["n_selected"]
        tot["ego_el"] += ego.last["n_eligible"]
        tot["protected"] += int((sw & el_logged).sum())
        if step < 12:                                                                                      # row-level: swapped rows are the swap alone, the rest is the perturbation alone
            ref_sw = DG.DaggerSwap(bank, p=0.1, seed=3, kin_mode="yaw").apply(batch, step)
            ref_pe = D.EgoPerturb(p=0.3, vscale=(0.5, 1.0), vmax=8.0, adrop=1.5, yawrate_std_deg=2.0, seed=7, kin_mode="yaw", stream=D.EGO_PERTURB_COMPOSED_STREAM).apply(batch, step)   # the composed stage's own stream (independent of the swap's)
            for k in D.EGO_PERTURB_REBUILT:
                assert bool(_same(out[k], ref_sw[k])[sw].all()), f"step {step}: {k} of a swapped row is the bank's, untouched by the perturber"
                assert bool(_same(out[k], ref_pe[k])[~sw].all()), f"step {step}: {k} of a non-swapped row is what the perturbation alone (on the composed stage's stream) gives it"
            assert bool(_same(out["lab_s"]["rc"], ref_sw["lab_s"]["rc"]).all())
            rows = (sw & el_logged).nonzero(as_tuple=True)[0]
            tot["differs"] += int((~_same(out["kin3"], ref_pe["kin3"])[rows]).sum())                       # where the exclusion mattered, the composed row is NOT what the perturbation alone gives
            if int(sw.sum()):
                rec = bank.gather(np.asarray(dag.last["record_idx"]))
                assert bool((out["kin3"][sw.nonzero(as_tuple=True)[0]] == torch.from_numpy(rec["kin3"]).to(out["kin3"].dtype)).all()), "a swapped sample's ego state IS the bank record's"
    assert abs(tot["sw"] / tot["dag_el"] - 0.1) <= 0.02, f"swap share {tot['sw'] / tot['dag_el']:.4f} of {tot['dag_el']} eligible"
    assert abs(tot["pe"] / tot["ego_el"] - 0.3) <= 0.02, f"perturbation share {tot['pe'] / tot['ego_el']:.4f} of {tot['ego_el']} non-swapped eligible"
    assert tot["protected"] > 0 and tot["differs"] > 0, "the exclusion mattered: swapped samples that the perturbation would have taken exist, and they differ from its output"


def test_the_overlap_guard_refuses_a_perturber_that_ignores_the_exclusion(pack):
    batch, bank, _ = world_batch(pack)

    class Blind(D.EgoPerturb):
        def apply(self, batch, step, *, p=None, exclude=None):
            return super().apply(batch, step, p=p, exclude=None)                                           # the mask is dropped on the floor

    comp = make_stage(bank, p_dag=0.5, p_ego=1.0)
    comp.ego = Blind(p=1.0, vscale=(0.5, 1.0), vmax=8.0, adrop=1.5, yawrate_std_deg=2.0, seed=7, kin_mode="yaw")
    with pytest.raises(RuntimeError, match="ZZEGO-DAGGER-OVERLAPZZ"):
        for step in range(20):
            comp.apply(batch, step)


# ------------------------------------------------------------------------------------------------ 6. the real caller: the whole trainer
@pytest.fixture
def stage_calls(monkeypatch):
    """Record every LOOP call (step >= 0) of the composed stage: its input, its output and what its two levers reported."""
    calls = []
    orig = TR.EgoDaggerCompose.apply

    def apply(self, batch, step, *, ego_p=None, dag_p=None):
        out = orig(self, batch, step, ego_p=ego_p, dag_p=dag_p)
        if step >= 0:
            calls.append({"step": step, "in": batch, "out": out, "sw": self.dag.last["sel"].clone(), "pe": self.ego.last["sel"].clone(), "overlap": self.overlap, "ego": self.ego, "dag": self.dag,
                          "dag_last": dict(self.dag.last), "ego_last": dict(self.ego.last)})
        return out
    monkeypatch.setattr(TR.EgoDaggerCompose, "apply", apply)
    return calls


def test_a_real_composed_run_feeds_the_loss_the_composed_batch_and_records_the_counts(pack, tmp_path, spy, stage_calls):
    steps = 6
    assert TR.main(_run(tmp_path / "ref", ["--steps", str(steps), "--init-ckpt", str(pack.ck)])) == 0               # the same data order with both levers off: the LOGGED batches
    c0 = spy[-steps:]
    spy.clear()
    assert TR.main(_run(tmp_path / "c", ["--steps", str(steps), *COMPOSED(pack, 0.5, 0.5)])) == 0
    c1 = spy[-steps:]
    assert len(stage_calls) == steps == len(c1)
    s, log = _summary(tmp_path / "c"), _log(tmp_path / "c")
    ch_sw = ch_pe = 0
    for i, call in enumerate(stage_calls):
        out, sw, pe = call["out"], call["sw"], call["pe"]
        assert call["overlap"] == 0 and not bool((sw & pe).any())
        for k in ("p10f", "base", "kin3", "p10_t", "pose", "pose_t", "pose_s", "nav"):
            assert bool(_same(c1[i][k], out[k]).all()), f"step {i}: the LOSS received the composed batch's {k}"
        rows = lambda m: m.nonzero(as_tuple=True)[0]                                                       # noqa: E731
        rec = call["dag"].bank.gather(np.asarray(call["dag_last"]["record_idx"]))
        assert bool((c1[i]["kin3"][rows(sw)] == torch.from_numpy(rec["kin3"]).to(c1[i]["kin3"].dtype)).all()), f"step {i}: the swapped rows hold the bank's kin3 in the batch the loss received"
        moved = ~_same(c1[i]["kin3"], c0[i]["kin3"])
        assert not bool((moved & ~(sw | pe)).any()), f"step {i}: nothing but a swapped or a perturbed row moved"
        ch_sw += int((moved & sw).sum())
        ch_pe += int((moved & pe).sum())
    assert ch_sw > 0 and ch_pe > 0, "the loss received BOTH levers' tensors (a swapped row and a perturbed row differ from the logged batch)"
    n = lambda key, last: sum(int(c[last][key]) for c in stage_calls)                                      # noqa: E731
    assert (s["dagger_n"], s["dagger_eligible"]) == (n("n_selected", "dag_last"), n("n_eligible", "dag_last")) and s["dagger_n"] > 0
    assert (s["ego_perturb_n"], s["ego_perturb_eligible"]) == (n("n_selected", "ego_last"), n("n_eligible", "ego_last")) and s["ego_perturb_n"] > 0
    assert s["ego_perturb_eligible"] == sum(int((c["ego"].eligible(c["in"]) & ~c["sw"]).sum()) for c in stage_calls), "ego_perturb_eligible = the NON-swapped eligible samples, recounted from the logged batches"
    assert s["dagger_eligible"] == sum(int((c["dag"].bank.candidates(c["in"]["ci"].numpy(), c["in"]["t"].numpy())[0] > 0).sum()) for c in stage_calls)
    assert s["ego_dagger_overlap_n"] == 0 and s["ego_perturb_with_dagger"] is True
    assert s["levers"]["ego_perturb_with_dagger"] is True and s["levers"]["ego_dagger_overlap_n"] == 0 and s["levers"]["ego_perturb"]["ego_perturb_p"] == 0.5 and s["levers"]["dagger_p"] == 0.5
    assert [r["ego_dagger_overlap_n"] for r in log] == [0] * steps
    ck = torch.load(tmp_path / "c" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert ck["ego_perturb_with_dagger"] is True and ck["ego_dagger_overlap"] == {"n": 0} and ck["ego_totals"]["n"] == s["ego_perturb_n"] and ck["dagger_totals"]["n"] == s["dagger_n"]


def test_the_check_that_the_loss_received_composed_tensors_can_see_a_dead_apply(pack, tmp_path, spy, monkeypatch):
    """The deliberate-regression arm of the test above, in-process: a loop call that returned the logged batch reads zero changed rows -- while the counters (apply's own report) still read > 0."""
    steps = 4
    assert TR.main(_run(tmp_path / "ref", ["--steps", str(steps), "--init-ckpt", str(pack.ck)])) == 0
    c0 = spy[-steps:]
    spy.clear()
    orig = TR.EgoDaggerCompose.apply

    def dead(self, batch, step, *, ego_p=None, dag_p=None):
        out = orig(self, batch, step, ego_p=ego_p, dag_p=dag_p)
        return batch if step >= 0 else out
    monkeypatch.setattr(TR.EgoDaggerCompose, "apply", dead)
    assert TR.main(_run(tmp_path / "mut", ["--steps", str(steps), *COMPOSED(pack, 0.5, 0.5)])) == 0
    c1 = spy[-steps:]
    assert sum(int((~_same(x1[k], x0[k])).sum()) for x0, x1 in zip(c0, c1) for k in ("p10f", "base", "kin3")) == 0
    s = _summary(tmp_path / "mut")
    assert s["dagger_n"] > 0 and s["ego_perturb_n"] > 0, "the counts come from apply's own report: only the batch-level predicate can see a deleted application"


def test_a_resumed_composed_run_draws_what_an_uninterrupted_run_drew_and_keeps_counting(pack, tmp_path, monkeypatch):
    common = [*COMMON, "--steps", "4", "--ckpt-every", "1", *COMPOSED(pack, 0.5, 0.5)]
    assert TR.main(["--out", str(tmp_path / "full"), *common]) == 0
    monkeypatch.setenv("V8F_TRAIN_CRASH_AT", "2")
    with pytest.raises(RuntimeError, match="V8F_TRAIN_CRASH_AT"):
        TR.main(["--out", str(tmp_path / "res"), *common])
    monkeypatch.delenv("V8F_TRAIN_CRASH_AT")
    assert TR.main(["--out", str(tmp_path / "res"), *common]) == 0
    a = torch.load(tmp_path / "full" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    b = torch.load(tmp_path / "res" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]), "the final weights are those of the uninterrupted run"
    for key in ("ego_totals", "dagger_totals", "ego_dagger_overlap"):
        assert a[key] == b[key], key
    sa, sb = _summary(tmp_path / "full"), _summary(tmp_path / "res")
    assert sa["dagger_n"] == sb["dagger_n"] > 0 and sa["ego_perturb_n"] == sb["ego_perturb_n"] > 0 and sa["ego_dagger_overlap_n"] == sb["ego_dagger_overlap_n"] == 0


def test_the_log_interval_overlap_counts_add_up_and_a_flag_off_log_has_no_overlap_key(pack, tmp_path):
    common = list(COMMON)
    common[common.index("--log-every") + 1] = "3"
    assert TR.main(_run(tmp_path / "r", ["--steps", "7", *COMPOSED(pack, 0.5, 0.5)], common)) == 0
    log = _log(tmp_path / "r")
    assert [r["step"] for r in log] == [3, 6, 7] and sum(r["ego_dagger_overlap_n"] for r in log) == _summary(tmp_path / "r")["ego_dagger_overlap_n"] == 0
    assert sum(r["dagger_n"] for r in log) == _summary(tmp_path / "r")["dagger_n"] and sum(r["ego_perturb_n"] for r in log) == _summary(tmp_path / "r")["ego_perturb_n"]
    assert TR.main(_run(tmp_path / "off", ["--steps", "2", *DAG(pack, 0.5, 3)])) == 0
    assert not any(k.startswith("ego_dagger") for r in _log(tmp_path / "off") for k in r) and "ego_dagger_overlap_n" not in _summary(tmp_path / "off") and "ego_perturb_with_dagger" not in _summary(tmp_path / "off")


# ------------------------------------------------------------------------------------------------ 7. INDEPENDENT STREAMS (the box-2 defect, MEASURED 2026-10-28)
# The swap and the perturbation seeded their generators from ONE formula; the arm's registered seeds are equal (--dagger-seed 0, --ego-perturb-seed 0); both take their selection uniform from row 0 of
# torch.rand(k, B), so the swap's uniform WAS the perturbation's: a sample eligible for both was swapped iff u < p_swap and perturbed iff p_swap <= u < p_ego, and P(perturbed | not swapped) was
# (p_ego - p_swap) / (1 - p_swap) = 0.222, not 0.3 (box 2: ego_perturb_n / ego_perturb_eligible = 19161 / 76898 = 0.2492). The tests above used UNEQUAL seeds (3 and 7): they could not see it.
def _binom_tol(p, n, k=5.0):
    """k binomial standard deviations of a share p over n independent trials."""
    return k * (p * (1.0 - p) / n) ** 0.5


@pytest.mark.parametrize("p_dag, p_ego, seed, steps", [(0.5, 0.5, 0, 30), (0.1, 0.3, 0, 80), (0.1, 0.3, 3, 80)])
def test_equal_seeds_every_sample_eligible_for_both_the_perturbed_share_of_the_not_swapped_is_its_own_p(pack, p_dag, p_ego, seed, steps):
    """The defect, stated as the probability it broke. EQUAL seeds, every swap-eligible sample also perturbation-eligible (vmax lifted): P(perturbed | not swapped) must be p_ego within a binomial tolerance.
    Shared stream: max(p_ego - p_dag, 0) / (1 - p_dag) -- 0.0 at 0.5 / 0.5 and 0.222 at the registered 0.1 / 0.3. The test asserts its own POWER: the shared-stream value is outside the tolerance."""
    batch, bank, _ = world_batch(pack)
    comp = make_stage(bank, p_dag=p_dag, p_ego=p_ego, dseed=seed, eseed=seed, vmax=1e6)
    ego, dag = comp.ego, comp.dag
    both = dag.eligible(batch) & ego.eligible(batch)
    assert int(both.sum()) >= 100 and bool(both.equal(dag.eligible(batch))), "every swap-eligible sample is also perturbation-eligible, and there are enough of them"
    n = k = n_el = k_sw = 0
    for step in range(steps):
        comp.apply(batch, step)
        sw, pe = dag.last["sel"], ego.last["sel"]
        free = both & ~sw
        assert comp.overlap == 0 and bool((pe <= (ego.eligible(batch) & ~sw)).all()), "a perturbed sample is an eligible, non-swapped one"
        n += int(free.sum())
        k += int((pe & free).sum())
        n_el += int(both.sum())
        k_sw += int((sw & both).sum())
    shared = max(p_ego - p_dag, 0.0) / (1.0 - p_dag)                                                         # what ONE shared stream gives
    assert abs(shared - p_ego) > 2 * _binom_tol(p_ego, n), f"power: the shared-stream value {shared:.4f} must be well outside {p_ego} +- {_binom_tol(p_ego, n):.4f} (n={n})"
    assert abs(k / n - p_ego) <= _binom_tol(p_ego, n), f"P(perturbed | not swapped, eligible for both) = {k}/{n} = {k / n:.4f}, declared {p_ego} (a shared stream gives {shared:.4f})"
    assert abs(k_sw / n_el - p_dag) <= _binom_tol(p_dag, n_el), f"the swap's own share {k_sw}/{n_el}"


def test_the_swaps_and_the_perturbations_selection_uniforms_are_not_one_stream_on_the_composed_path(pack, monkeypatch):
    """Read the uniforms the two levers ACTUALLY draw (a spy on ``torch.rand``: the swap draws rand(2, B), the perturbation rand(3, B), selection = row 0): equal seeds, composed stage -> not equal, and uncorrelated.
    The CONTROL: the same seeds on an UNSEPARATED perturber (stream 0, the pre-fix generator) read EXACTLY equal -- the spy can see the defect."""
    batch, bank, _ = world_batch(pack)
    B = int(batch["t"].shape[0])
    drawn = []
    orig = torch.rand

    def spy_rand(*size, **kw):
        out = orig(*size, **kw)
        if kw.get("generator") is not None and len(size) == 2 and size[1] == B and size[0] in (2, 3):
            drawn.append((size[0], out[0].clone()))
        return out
    monkeypatch.setattr(torch, "rand", spy_rand)
    for seed in (0, 3):
        comp = make_stage(bank, p_dag=0.1, p_ego=0.3, dseed=seed, eseed=seed)
        assert comp.dag.seed == comp.ego.seed == seed
        for step in (0, 1, 7, 1000):
            drawn.clear()
            comp.apply(batch, step)
            assert [d[0] for d in drawn] == [2, 3], "the swap draws first (rand(2, B)), then the perturbation (rand(3, B))"
            u_swap, u_ego = drawn[0][1], drawn[1][1]
            assert not torch.equal(u_swap, u_ego), f"seed {seed} step {step}: the swap's and the perturbation's selection uniforms are the same numbers"
            assert abs(float(np.corrcoef(u_swap.numpy(), u_ego.numpy())[0, 1])) < 0.25, "and they are uncorrelated (B = 342: 0.25 is 4.6 standard deviations)"
            assert comp.dag._seed_of(step) != comp.ego._seed_of(step)
        plain = D.EgoPerturb(p=0.3, vscale=(0.5, 1.0), vmax=8.0, adrop=1.5, yawrate_std_deg=2.0, seed=seed, kin_mode="yaw")             # stream 0: the pre-fix generator
        drawn.clear()
        comp.dag.apply(batch, 7)
        plain.apply(batch, 7)
        assert torch.equal(drawn[0][1], drawn[1][1]) and comp.dag._seed_of(7) == plain._seed_of(7), "CONTROL: at equal seeds an unseparated perturber draws exactly the swap's uniforms"


def test_seed_of_stream_zero_is_the_original_formula_and_a_stream_is_a_deterministic_separate_one():
    """stream 0 (every non-composed path) is the original formula, bit for bit; a stream is a pure function of (stream, seed, step), in [0, 2**62), never the original seed, collision-free over a grid."""
    for seed in (0, 1, 3, 7, 12345):
        e0 = D.EgoPerturb(p=0.3, seed=seed)
        es = D.EgoPerturb(p=0.3, seed=seed, stream=D.EGO_PERTURB_COMPOSED_STREAM)
        dg = DG.DaggerSwap(None, p=0.1, seed=seed)
        assert e0.stream == 0 and es.stream == D.EGO_PERTURB_COMPOSED_STREAM != 0
        for step in (-1, 0, 1, 99, 10 ** 6):
            assert e0._seed_of(step) == (seed * 1_000_003 + step) % (1 << 62) == dg._seed_of(step), "stream 0 is the original formula (and the swap's)"
            assert es._seed_of(step) != e0._seed_of(step) and es._seed_of(step) == es._seed_of(step) and 0 <= es._seed_of(step) < (1 << 62)
    grid = {D.EgoPerturb(p=0.3, seed=s, stream=1)._seed_of(t) for s in range(20) for t in range(500)}
    assert len(grid) == 20 * 500, "no two (seed, step) pairs share a seed within one stream"
    assert D.EgoPerturb(p=0.3, seed=3, stream=1)._seed_of(5) != D.EgoPerturb(p=0.3, seed=3, stream=2)._seed_of(5), "two stream ids are two streams"
    assert D.EgoPerturb(p=0.3, stream=-1).law_problems() and not D.EgoPerturb(p=0.3, stream=0).law_problems() and "ego_perturb_stream" not in D.EgoPerturb(p=0.3).built_levers(), \
        "a negative stream is illegal; a stream-0 lever record is the pre-change record (no new key)"


def test_the_trainer_run_at_the_box2_shape_equal_seeds_overlapping_eligibility_perturbs_its_declared_share(pack, tmp_path):
    """The box-2 defect through the real caller: the registered argv shape (--dagger-p 0.1, --ego-perturb-p 0.3, BOTH seeds 0, the law's vmax 8) on the trainer's synthetic world, whose eligibility overlaps as the real
    corpus does (140 of 204 perturbation-eligible windows are also swap-eligible). summary.json's ego_perturb_n / ego_perturb_eligible must be 0.3 +- 0.02 (box 2 read 0.2492; e7ea20d1 reads ~0.25 here)."""
    common = list(COMMON)
    common[common.index("--batch") + 1] = "64"
    assert TR.main(_run(tmp_path / "box2", ["--steps", "200", *COMPOSED(pack, 0.1, 0.3, dseed=0, eseed=0)], common)) == 0
    s = _summary(tmp_path / "box2")
    assert s["ego_perturb_with_dagger"] is True and s["ego_dagger_overlap_n"] == 0
    n_el = s["ego_perturb_eligible"]
    assert n_el >= 4725, f"power: {n_el} eligible samples is 3 standard deviations of 0.02 or better (0.21 * (3 / 0.02) ** 2 = 4725)"
    share = s["ego_perturb_n"] / n_el
    assert abs(share - 0.3) <= 0.02, f"ego_perturb_n / ego_perturb_eligible = {s['ego_perturb_n']} / {n_el} = {share:.4f}, registered 0.3 +- 0.02"
    sw = s["dagger_n"] / s["dagger_eligible"]
    assert abs(sw - 0.1) <= 0.02, f"dagger_n / dagger_eligible = {s['dagger_n']} / {s['dagger_eligible']} = {sw:.4f}, registered 0.1 +- 0.02"


def test_the_run_record_names_the_perturbation_stream_and_a_flag_off_record_names_stream_zero(pack, tmp_path):
    S = D.EGO_PERTURB_COMPOSED_STREAM
    assert TR.main(_run(tmp_path / "c", ["--steps", "2", "--ckpt-every", "1", *COMPOSED(pack, 0.5, 0.5, dseed=0, eseed=0)])) == 0
    sc, s = _selfcheck(tmp_path / "c"), _summary(tmp_path / "c")
    ck = torch.load(tmp_path / "c" / "ckpt_last.pt", map_location="cpu", weights_only=False)
    assert sc["levers"]["ego_perturb_stream"] == s["levers"]["ego_perturb_stream"] == S != 0, "levers.ego_perturb_stream: the BUILT stream, in the selfcheck and the summary"
    assert sc["levers"]["ego_perturb"]["ego_perturb_stream"] == S and ck["ego_perturb"]["ego_perturb_stream"] == S and ck["ego_perturb_stream"] == S
    assert TR.main(_run(tmp_path / "dag", ["--steps", "2", *DAG(pack, 0.5, 3)])) == 0
    assert TR.main(_run(tmp_path / "orc", ["--steps", "2", *argv_family(pack, "orc")[2:]])) == 0
    for name in ("dag", "orc"):
        sc, s = _selfcheck(tmp_path / name), _summary(tmp_path / name)
        ck = torch.load(tmp_path / name / "ckpt_last.pt", map_location="cpu", weights_only=False)
        assert sc["levers"]["ego_perturb_stream"] == s["levers"]["ego_perturb_stream"] == 0, f"{name}: a flag-off record reads stream 0"
        assert "ego_perturb_stream" not in sc["levers"]["ego_perturb"] and "ego_perturb_stream" not in ck["ego_perturb"] and "ego_perturb_stream" not in ck, f"{name}: no new key in the ego record or the checkpoint"


def _unseparated(ego, dag, on):
    comp = TR.EgoDaggerCompose(ego, dag, on=on)
    comp.ego.stream = 0                                                                                   # the pre-fix stage: composed, perturbation on the original generator
    return comp


def _separated_off(ego, dag):
    ego.stream = D.EGO_PERTURB_COMPOSED_STREAM                                                            # a stage that is NOT composed but whose perturber was left on the composed stream
    return TR.EgoDaggerCompose(ego, dag, on=False)


@pytest.mark.parametrize("name, flag, make, msg", [
    ("composed_without_separation", True, lambda a, ego, dag: _unseparated(ego, dag, on=True), "draws from stream 0 but the declaration needs 1"),
    ("separation_without_the_flag", False, lambda a, ego, dag: _separated_off(ego, dag), "draws from stream 1 but the declaration needs 0"),
])
def test_gdvb_deliberate_regression_a_perturbation_that_is_not_on_the_declared_stream_is_refused(pack, name, flag, make, msg):
    bad = _gate(pack, COMPOSED(pack, flag=flag), make)
    assert any(msg in m for m in bad), bad
    assert _gate(pack, COMPOSED(pack)) == [], "CONTROL: the real stage reads clean"


def test_a_composed_checkpoint_from_before_the_stream_separation_is_not_resumed_on_another_generator(pack, tmp_path, monkeypatch):
    """e7ea20d1's composed checkpoint carries no ``ego_perturb_stream`` (= stream 0). Same argv, so the argv_sha matches -- the resume must still be REFUSED: the perturbation would switch generators mid-run."""
    common = [*COMMON, "--steps", "4", "--ckpt-every", "1", *COMPOSED(pack, 0.5, 0.5)]
    monkeypatch.setenv("V8F_TRAIN_CRASH_AT", "2")
    with pytest.raises(RuntimeError, match="V8F_TRAIN_CRASH_AT"):
        TR.main(["--out", str(tmp_path / "old"), *common])
    monkeypatch.delenv("V8F_TRAIN_CRASH_AT")
    p = tmp_path / "old" / "ckpt_last.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    assert ck["ego_perturb_stream"] == D.EGO_PERTURB_COMPOSED_STREAM
    del ck["ego_perturb_stream"]                                                                           # what the pre-fix trainer wrote
    torch.save(ck, p)
    with pytest.raises(SystemExit, match=r"ZZRESUME-REFUSEDZZ .*stream 0 but this build draws stream 1"):
        TR.main(["--out", str(tmp_path / "old"), *common])


# ------------------------------------------------------------------------------------------------ the mutation anchors
def test_the_single_applying_lines_exist_exactly_once():
    """`mutation.txt`'s mutants delete these; an anchor that is not found exactly once would make a mutation test vacuous. The flag-off path keeps the two ORIGINAL applying lines (the other suites' anchors)."""
    s = TRAIN_P.read_text(encoding="utf-8")
    assert s.count("                b = comp.apply(b, step)") == 1
    assert s.count("        swapped = self.dag.apply(batch, step, p=dag_p) ") == 1
    assert s.count("        out = self.ego.apply(swapped, step, p=ego_p, exclude=sw) ") == 1
    assert s.count("        if self.overlap:") == 1
    assert s.count("    comp = make_compose(a, ego, dag) ") == 1
    assert s.count("        cmpp = ego_dagger_compose_probe(model, probe, cfg, comp) ") == 1
    assert s.count("    bad += bad_ego_dagger(a, comp, ego, dag) ") == 1
    assert s.count("    res[\"ego_perturb_probe_on_composed\"] = ego_perturb_probe(model, swapped, cfg, ego, exclude=sw)") == 1
    assert s.count("            b = ego.apply(b, step) ") == 1 and s.count("            b = dag.apply(b, step) ") == 1, "the flag-off path: the two original lines, once each"
    assert s.count("            self.ego.stream = D.EGO_PERTURB_COMPOSED_STREAM ") == 1
    assert s.count("        if comp.on and int(ck.get(\"ego_perturb_stream\", 0)) != ego.stream: ") == 1
    assert s.count("    if comp.ego.stream != want_stream:") == 1
    assert s.count("    levers[\"ego_perturb_stream\"] = comp.ego.stream ") == 1
    d = DATA_P.read_text(encoding="utf-8")
    assert d.count("        if not self.stream:\n            return base ") == 1
    assert d.count("        return elig & ~exclude.to(device=elig.device, dtype=torch.bool)") == 1
    assert d.count("        elig = self.eligible_free(batch, exclude)\n        sel =(u[0] < p)") == 1
