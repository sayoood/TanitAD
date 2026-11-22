"""test_lookat.py -- tests of the kit of card PREREG_V8F_PROOF_ADD107 (LOOKAT-T0): the arithmetic of target, loss and read-out, the sample builder, a tiny real fit on CPU (the permuted fit differs, the prediction file, the
refusals), the identity control k5, S4 / S8 / S8_out and the fixed reference, the paired seeded bootstrap, the four names and their second derivation, every control red on its planted fault, the controls stage on a tiny pod-like
tree, the names stage refusing, and the queue (DRY, markers, resume, slots) through stubs. Synthetic data only; no file above 5 MB.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest lookat/test_lookat.py -q -p no:cacheprovider --basetemp /f/TanitAD-scratch/v8proof/lookat/tmp_pt/<new name>"""
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
V8 = HERE.parent
KIT_FIT = V8 / "cropread" / "fit"
sys.path.insert(0, str(HERE))
import locator_fit as LF        # noqa: E402
import read_lookat as RL        # noqa: E402

BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
QUEUE, PODREAD, PUSH, PULL = HERE / "pod_lookat_queue.sh", HERE / "podread_lookat.sh", HERE / "push_lookat.sh", HERE / "pull_lookat.sh"
LF.set_cft(str(KIT_FIT))
CFT = LF.CFT


def posix(p):
    return Path(p).as_posix()


def bash_exe():
    return shutil.which("bash")


# ------------------------------------------------------------------------------------------------ arithmetic
def test_targets_and_readout_arithmetic():
    t = LF.targets_of([40.0, -8.0, 16.0, -50.0], [20.0, -2.0, 6.0, -30.0])
    assert np.allclose(t[:, 0], [2.0, -0.5, 1.0, -2.0]) and np.allclose(t[:, 1], [2.0, 0.0, 1.0, -2.0])
    az, el = LF.readout([0.5, -1.0], [1.0, 0.0])
    assert np.allclose(az, [8.0, -16.0]) and np.allclose(el, [6.0, -2.0])
    a, e = np.array([-30.0, 0.0, 12.5, 31.0]), np.array([-17.0, -2.0, 5.0, 13.0])
    t = LF.targets_of(a, e)
    ra, re_ = LF.readout(t[:, 0], t[:, 1])
    assert np.allclose(ra, a, atol=1e-5) and np.allclose(re_, e, atol=1e-5)          # inside the clip range the read-out inverts the target


def test_loss_is_smooth_l1_beta_quarter_mean_over_both_values_and_the_head_has_two_outputs_without_sigmoid():
    import torch
    fn = torch.nn.SmoothL1Loss(beta=LF.BETA, reduction="none")
    out = fn(torch.tensor([[0.1, 1.0]]), torch.zeros(1, 2))
    assert abs(float(out[0, 0]) - 0.5 * 0.01 / 0.25) < 1e-7 and abs(float(out[0, 1]) - (1.0 - 0.125)) < 1e-7
    assert abs(float(out.mean()) - (0.02 + 0.875) / 2) < 1e-7
    m = LF.make_loc_model()
    assert m.mlp[2].out_features == 2 and m.mlp[0].in_features == 5 * CFT.D_MODEL
    with torch.no_grad():
        m.mlp[2].bias.fill_(100.0)
    z = torch.randn(3, 5 * CFT.D_MODEL)
    assert m.from_z(z).shape == (3, 2) and float(m.from_z(z).min()) > 50.0           # no sigmoid: values far above 1
    assert LF.loc_hparams()["beta_m"] == 0.25 and {k: v for k, v in LF.loc_hparams().items() if k != "beta_m"} == {k: v for k, v in CFT.CARD_HPARAMS.items() if k != "beta_m"}


# ------------------------------------------------------------------------------------------------ the tiny world
NFR = 9


def make_tiny(tmp):
    rng = np.random.RandomState(0)
    root = tmp / "run1"
    (root / "tok" / "D").mkdir(parents=True)
    (root / "lists").mkdir()
    box = tmp / "box"
    box.mkdir()
    fit, sel, ev = ["f0", "f1", "f2"], ["s0"], ["e0", "e1"]
    man = {}
    for c in fit + sel + ev:
        tok = rng.standard_normal((NFR, 2, 65, 1024)).astype(np.float16)
        p = root / "tok" / "D" / (c + ".npy")
        np.save(str(p), tok)
        import hashlib
        man[c] = {"n_frames": NFR, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        has = rng.rand(NFR) < 0.8
        has[5:8] = True
        az, el = rng.uniform(-20, 20, NFR), rng.uniform(-8, 4, NFR)
        az[1] = np.nan
        np.savez(str(box / (c + ".npz")), has_lead=has, ctr_az=az, ctr_el=el)
    (root / "tok" / "D" / "MANIFEST.json").write_text(json.dumps({"clips": man}), encoding="utf-8")
    (root / "lists" / "train_final.txt").write_text("".join("%s fit\n" % c for c in fit) + "s0 sel\n", encoding="utf-8")
    (root / "lists" / "eval141.txt").write_text("e0\ne1\n", encoding="utf-8")
    names = fit + sel
    np.savez(str(tmp / "join_train.npz"), clip=np.asarray(names), n=np.full(4, NFR), off=np.arange(4) * NFR, lead_gap=rng.uniform(0, 60, 4 * NFR), lead_valid=np.ones(4 * NFR, np.int8))
    np.savez(str(tmp / "join_eval.npz"), clip=np.asarray(ev), n=np.full(2, NFR), off=np.arange(2) * NFR, lead_gap=rng.uniform(0, 60, 2 * NFR), lead_valid=np.ones(2 * NFR, np.int8))
    fr = np.tile(np.arange(NFR), 2)
    gf = []
    for i in range(5):
        p = tmp / ("gap%d.npz" % i)
        np.savez(str(p), clip_index=np.repeat(np.arange(2), NFR), frame=fr, gap=np.where(fr >= 4, 10.0, np.nan), has_out=fr >= 4, clips=np.asarray(ev))
        gf.append(posix(p))
    return {"root": root, "box": box, "jt": tmp / "join_train.npz", "je": tmp / "join_eval.npz", "gaps": ",".join(gf), "fit": fit, "sel": sel, "ev": ev}


def lf_args(w, out, *extra, seed=0):
    return ["--seed", str(seed), "--root", str(w["root"]), "--box-dir", str(w["box"]), "--out-root", str(out), "--kit-fit", str(KIT_FIT), "--device", "cpu", "--epochs", "2", "--batch", "8"] + list(extra)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lk")
    w = make_tiny(tmp)
    out = tmp / "O"
    for seed in (0, 1, 2):
        assert LF.main(lf_args(w, out, seed=seed)) == 0
    assert LF.main(lf_args(w, out, "--permute-targets")) == 0
    assert CFT.main(["--arm", "D", "--head", "H5", "--seed", "0", "--epochs", "1", "--batch", "8", "--root", str(w["root"]), "--join-train", str(w["jt"]), "--device", "cpu"]) == 0
    assert LF.main(["--identity", "--root", str(w["root"]), "--out-root", str(out), "--kit-fit", str(KIT_FIT), "--join-train", str(w["jt"]), "--device", "cpu"]) == 0
    w["out"], w["tmp"] = out, tmp
    return w


def test_sample_builder_uses_box_frames_with_a_full_history_and_refuses_a_frame_count_mismatch(world):
    store = CFT.TokenStore(str(world["root"]), "D")
    sc, sf, sy = LF.build_samples(["f0", "f1"], store, "box", box_dir=str(world["box"]))
    z = np.load(str(world["box"] / "f0.npz"))
    has = z["has_lead"].astype(bool) & np.isfinite(z["ctr_az"]) & np.isfinite(z["ctr_el"])
    want = [f for f in range(NFR) if has[f] and f >= 4]
    assert list(sf[sc == 0]) == want and sy.shape == (len(sf), 2) and sf.min() >= 4
    k = list(sf[sc == 0]).index(want[0])
    assert np.allclose(sy[sc == 0][k], LF.targets_of(z["ctr_az"][want[0]], z["ctr_el"][want[0]]))
    np.savez(str(world["tmp"] / "bad_box_f0.npz"), has_lead=np.ones(NFR + 1, bool), ctr_az=np.zeros(NFR + 1), ctr_el=np.zeros(NFR + 1))
    (world["tmp"] / "badbox").mkdir(exist_ok=True)
    shutil.copy(str(world["tmp"] / "bad_box_f0.npz"), str(world["tmp"] / "badbox" / "f0.npz"))
    with pytest.raises(SystemExit):
        LF.build_samples(["f0"], store, "box", box_dir=str(world["tmp"] / "badbox"))
    labels = CFT.load_join_labels(str(world["jt"]), ["f0"])
    gc, gf, gy = LF.build_samples(["f0"], store, "gap", labels=labels)
    c2, f2, y2 = CFT.build_samples(["f0"], labels, store, "H5")
    assert np.array_equal(gc, c2) and np.array_equal(gf, f2) and np.array_equal(gy, y2)      # the gap mode is the builder of crop_fit.py for H5


def test_the_fit_writes_record_head_and_prediction_and_refuses_a_second_run(world):
    out = world["out"]
    rec = json.loads((out / "fits" / "LOC_s0" / "FIT_RECORD.json").read_text(encoding="utf-8"))
    for k in ("card", "tier", "seed", "permute_targets", "argv", "tool_sha256", "crop_fit_sha256", "train_list_sha256", "eval_list_sha256", "hparams", "hparams_equal_card", "history", "kept_epoch",
              "kept_sel_loss", "head_pt_sha256", "n_samples", "seconds"):
        assert k in rec, k
    assert rec["card"] == "PREREG_V8F_PROOF_ADD107" and rec["permute_targets"] is False and rec["hparams_equal_card"] is False and len(rec["history"]) == 2
    assert rec["head_pt_sha256"] == CFT.sha256_file(str(out / "fits" / "LOC_s0" / "head.pt")) and rec["crop_fit_sha256"] == CFT.sha256_file(str(KIT_FIT / "crop_fit.py"))
    assert rec["tool_sha256"] == CFT.sha256_file(str(HERE / "locator_fit.py")) and rec["hparams"]["beta_m"] == 0.25
    z = np.load(str(out / "pred" / "LOC_s0.npz"))
    assert sorted(z.files) == ["az_pred", "clip_index", "clips", "el_pred", "frame", "has_out"] and len(z["frame"]) == 2 * NFR
    assert not np.isfinite(z["az_pred"][z["frame"] < 4]).any() and np.isfinite(z["az_pred"][z["frame"] >= 4]).all() and np.array_equal(z["has_out"], np.isfinite(z["az_pred"]))
    meta = json.loads((out / "pred" / "LOC_s0_meta.json").read_text(encoding="utf-8"))
    assert meta["pred_sha256"] == CFT.sha256_file(str(out / "pred" / "LOC_s0.npz"))
    with pytest.raises(SystemExit):
        LF.main(lf_args(world, out))
    with pytest.raises(SystemExit):
        LF.main(lf_args(world, out, "--predict"))                                    # the prediction file exists


def test_the_permuted_fit_differs_and_permutes_pairs_of_targets(world):
    import torch
    a = torch.load(str(world["out"] / "fits" / "LOC_s0" / "head.pt"), map_location="cpu")["state_dict"]
    b = torch.load(str(world["out"] / "fits" / "LOC_s0_perm" / "head.pt"), map_location="cpu")["state_dict"]
    assert any(not torch.equal(a[k], b[k]) for k in a)
    assert json.loads((world["out"] / "fits" / "LOC_s0_perm" / "FIT_RECORD.json").read_text(encoding="utf-8"))["permute_targets"] is True
    y = np.arange(20, dtype=np.float32).reshape(10, 2)
    p = y[np.random.default_rng(LF.PERM_SEED_BASE).permutation(10)]
    assert not np.array_equal(p, y) and sorted(map(tuple, p)) == sorted(map(tuple, y))
    za, zb = np.load(str(world["out"] / "pred" / "LOC_s0.npz")), np.load(str(world["out"] / "pred" / "LOC_s0_perm.npz"))
    assert not np.allclose(np.nan_to_num(za["az_pred"]), np.nan_to_num(zb["az_pred"]))


def test_identity_control_k5_passes_on_the_original_head_and_goes_red_on_tolerance_and_on_a_changed_record(world):
    res = json.loads((world["out"] / "IDENTITY_K5.json").read_text(encoding="utf-8"))
    assert res["PASS"] is True and res["abs_diff"] < 1e-4 and res["n_sel_samples"] == res["n_sel_samples_record"] and RL.ctl_k5(res)["PASS"] is True
    assert RL.ctl_k5(dict(res, crop_fit_sha256="0" * 64))["PASS"] is False
    assert LF.main(["--identity", "--root", str(world["root"]), "--out-root", str(world["tmp"] / "O2"), "--kit-fit", str(KIT_FIT), "--join-train", str(world["jt"]), "--device", "cpu", "--identity-tol", "0"]) == 1
    rp = world["root"] / "fits" / "D_H5_s0" / "FIT_RECORD.json"
    orig = rp.read_text(encoding="utf-8")
    try:
        d = json.loads(orig)
        d["kept_sel_loss"] += 0.01
        rp.write_text(json.dumps(d), encoding="utf-8")
        assert LF.main(["--identity", "--root", str(world["root"]), "--out-root", str(world["tmp"] / "O3"), "--kit-fit", str(KIT_FIT), "--join-train", str(world["jt"]), "--device", "cpu"]) == 1
        assert json.loads((world["tmp"] / "O3" / "IDENTITY_K5.json").read_text(encoding="utf-8"))["PASS"] is False
    finally:
        rp.write_text(orig, encoding="utf-8")
    with pytest.raises(SystemExit):
        LF.main(["--identity", "--root", str(world["root"]), "--out-root", str(world["out"]), "--kit-fit", str(KIT_FIT), "--join-train", str(world["jt"]), "--device", "cpu"])   # exists


# ------------------------------------------------------------------------------------------------ the read: values
def hand_world():
    """two clips of 5 rows; every number checked by hand."""
    J = {"clip": np.asarray(["a", "b"]), "n": np.array([5, 5]), "off": np.array([0, 5]), "lead_gap": np.array([5.0, 15.0, 25.0, 45.0, 55.0, 5.0, 15.0, 25.0, 45.0, 55.0]), "lead_valid": np.ones(10, np.int8)}
    az = np.array([0.0, 3.0, 9.0, -12.0, 20.0, 1.0, -3.5, 7.0, 10.0, -30.0])
    el = np.array([-2.0, 0.0, -2.0, 1.0, -2.0, -10.0, -2.0, 2.0, -2.0, -2.0])
    pa = az + np.array([1.0, 5.0, 7.0, 3.0, 9.0, 0.0, 0.0, 4.5, 0.0, 12.0])
    pe = el + np.array([1.0, 1.0, 4.5, 3.9, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    box = {"has": np.ones(10, bool), "az": az, "el": el}
    perm = {"az": np.zeros(10), "el": np.full(10, -2.0)}
    preds = {"s0": {"az": pa, "el": pe}, "s1": {"az": pa, "el": pe}, "s2": {"az": pa, "el": pe}, "perm": perm}
    return J, np.ones(10, bool), box, preds


def test_s4_s8_s8_out_and_the_fixed_reference_by_hand():
    J, sel, box, preds = hand_world()
    P = RL.compute_values(J, sel, box, preds, 200)
    # |az|: 0 3 9 12 20 1 3.5 7 10 30 ; |el+2|: 0 2 0 3 0 8 0 4 0 0 ; outside rows (|az| > 8): 9, 12, 20, 10, 30 -> 5
    assert P["n_rows"] == 10 and P["n_outside"] == 5 and P["n_el_off8"] == 0
    # fixed S4: rows with |az| <= 4 and |el+2| <= 4: rows 0, 1, 6 (|el+2| = 0), row 5 has |el+2| = 8 -> no; row 7 |az| 7 -> no => 3 of 10
    assert abs(P["fixed"]["S4"] - 0.3) < 1e-12
    # fixed S8: |az| <= 8 and |el+2| <= 8: rows 0, 1, 5, 6, 7 -> 5 of 10 = (10 - 5) / 10
    assert abs(P["fixed"]["S8"] - 0.5) < 1e-12 and abs(P["fixed"]["S8"] - P["S8_fixed_closed_form"]) < 1e-12
    # fit: d_az = 1 5 7 3 9 0 0 4.5 0 12 ; d_el = 1 1 4.5 3.9 9 0 0 0 0 0
    # S4 (both <= 4): rows 0, 3, 5, 6, 8 -> 5 ; the row 7 has d_az 4.5 -> no
    assert abs(P["fits"]["s0"]["S4"] - 0.5) < 1e-12
    # S8 (both <= 8): rows 0, 1, 2, 3, 5, 6, 7, 8 -> 8 (row 4 d_az 9, row 9 d_az 12)
    assert abs(P["fits"]["s0"]["S8"] - 0.8) < 1e-12
    # S8_out: outside rows 2, 3, 4, 8, 9: d <= 8 for rows 2, 3, 8 -> 3 of 5
    assert abs(P["fits"]["s0"]["S8_out"] - 0.6) < 1e-12
    assert P["fits"]["s0"]["by_gap"][0]["n"] == 2 and P["fits"]["s0"]["by_az"][3]["n"] == 2          # gaps 5, 5 in 0-10; |az| above 16: rows 4 and 9
    assert [b["n"] for b in P["fits"]["s0"]["by_az"]] == [4, 1, 3, 2] and [b["n"] for b in P["fits"]["s0"]["by_gap"]] == [2, 2, 2, 4]
    assert abs(P["fits"]["s0"]["med_d_az"] - np.median([1, 5, 7, 3, 9, 0, 0, 4.5, 0, 12])) < 1e-12


def test_bootstrap_is_paired_seeded_and_on_the_clip_sums():
    rng = np.random.RandomState(5)
    ha, hb, r = rng.randint(0, 5, 141).astype(float), rng.randint(0, 5, 141).astype(float), np.full(141, 6.0)
    iv = RL.boot_diff(ha, hb, r)
    assert iv == RL.boot_diff(ha, hb, r) and iv != RL.boot_diff(ha, hb, r, seed=108) and RL.BOOT_SEED == 107 and RL.N_BOOT == 2000
    idx = np.random.RandomState(107).randint(0, 141, size=(2000, 141))
    d = (ha[idx].sum(1) - hb[idx].sum(1)) / r[idx].sum(1)
    assert np.allclose(iv, [np.percentile(d, 2.5), np.percentile(d, 97.5)])               # one draw of clips for both arms
    assert RL.boot_diff(ha, ha, r) == [0.0, 0.0]                                          # paired: identical arms give an interval of zero width
    assert RL.name_second(RL.compute_values(*RL.synth_world("good")[:4], 100))[1]["s0"][0] > 0


@pytest.mark.parametrize("kind", ["good", "centres", "bad_seed1", "same"])
def test_each_of_the_four_names_and_the_second_derivation_agrees(kind):
    J, sel, box, preds, want = RL.synth_world(kind)
    P = RL.compute_values(J, sel, box, preds, 300)
    n1, n2, agree = RL.derive_name(P)
    assert n1 == want == n2 and agree


def test_name_rules_on_the_edges():
    base = {"fits": {k: {"S8_out": 0.5} for k in ("s0", "s1", "s2")}, "intervals": {k: {"iv": [0.01, 0.1]} for k in ("s0", "s1", "s2")}}
    assert RL.name_of(base) == "LOOKAT-T0-LOCATES"                                       # S8_out exactly 0.50 holds
    base["fits"]["s1"]["S8_out"] = 0.4999
    assert RL.name_of(base) == "LOOKAT-T0-CENTRES-ONLY"
    base["intervals"]["s2"]["iv"] = [0.0, 0.1]                                            # a lower end of exactly 0 is not above 0
    assert RL.name_of(base) == "LOOKAT-T0-NOT-SEPARATED"
    base["intervals"]["s0"]["iv"] = [-0.2, -0.0001]
    assert RL.name_of(base) == "LOOKAT-T0-DOES-NOT-LOCATE"


def test_selftest_runs_and_every_control_goes_red_on_its_fault(capsys):
    assert RL.selftest() == 0
    out = capsys.readouterr().out
    assert "all red: True" in out and "ZZ SELFTEST-READ-LOOKAT ok: True" in out and "all red: False" not in out
    for k in ("k0", "k1", "k2", "k3", "k4", "k5", "k6"):
        assert "fault %s:" % k in out


def test_k3_goes_red_when_the_permuted_fit_is_as_good_as_seed_0():
    J, sel, box, preds, _ = RL.synth_world("good")
    assert RL.ctl_k3(RL.compute_values(J, sel, box, preds, 200))["PASS"] is True
    preds["perm"] = preds["s0"]
    assert RL.ctl_k3(RL.compute_values(J, sel, box, preds, 200))["PASS"] is False


# ------------------------------------------------------------------------------------------------ the controls stage on the tiny tree, the names stage
def rl_args(w, out_file, stage="controls", extra=()):
    return ["--stage", stage, "--root", str(w["root"]), "--out-root", str(w["out"]), "--box-dir", str(w["box"]), "--eval-join", str(w["je"]), "--gapfiles", w["gaps"], "--out", str(out_file),
            "--n-boot", "50", "--expect-fit", "3", "--expect-sel", "1", "--expect-eval", "2"] + list(extra)


def test_controls_stage_on_the_tiny_tree_prints_only_control_lines_and_withholds_values(world, capsys):
    (world["out"] / "kit").mkdir(exist_ok=True)
    (world["out"] / "kit" / "lead_pytest.out").write_text("== 12 passed in 3s ==\n", encoding="utf-8")
    outf = world["tmp"] / "CONTROLS.json"
    assert RL.main(rl_args(world, outf)) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [ln.split()[0] for ln in lines if ln.startswith("k")][:7] == ["k0", "k1", "k2", "k3", "k4", "k5", "k6"] and lines[-1].startswith("ZZ READ-LOOKAT controls rc 0 controls_all_pass False")
    for ln in lines[:-1]:
        assert re.match(r"^k[0-6] (PASS|FAIL)$|^k4 note: ", ln), ln
    res = json.loads(outf.read_text(encoding="utf-8"))
    c = res["controls"]
    assert c["k0"]["PASS"] and c["k1"]["PASS"] and c["k6"]["PASS"] and not c["k2"]["PASS"]           # the tiny fits do not carry the settings of the card
    assert "payload" not in res and "withheld" in res and res["controls_all_pass"] is False
    assert RL.main(rl_args(world, outf)) == 2                                                       # a second read into the same file is refused
    assert RL.main(["--stage", "names", "--controls", str(outf), "--out", str(world["tmp"] / "RES.json")]) == 3
    assert not (world["tmp"] / "RES.json").exists()


def test_controls_stage_sees_a_missing_box_file_and_a_changed_token_file(world, capsys):
    shutil.copy(str(world["box"] / "e0.npz"), str(world["tmp"] / "e0.keep"))
    os.rename(str(world["box"] / "e0.npz"), str(world["tmp"] / "e0.moved"))
    try:
        outf = world["tmp"] / "CONTROLS_b.json"
        RL.main(rl_args(world, outf))
        assert json.loads(outf.read_text(encoding="utf-8"))["controls"]["k0"]["PASS"] is False
    finally:
        os.rename(str(world["tmp"] / "e0.moved"), str(world["box"] / "e0.npz"))
    capsys.readouterr()
    man = world["root"] / "tok" / "D" / "MANIFEST.json"
    orig = man.read_text(encoding="utf-8")
    try:
        d = json.loads(orig)
        d["clips"]["f1"]["sha256"] = "0" * 64
        man.write_text(json.dumps(d), encoding="utf-8")
        outf = world["tmp"] / "CONTROLS_c.json"
        RL.main(rl_args(world, outf))
        assert json.loads(outf.read_text(encoding="utf-8"))["controls"]["k1"]["PASS"] is False
    finally:
        man.write_text(orig, encoding="utf-8")


def good_controls_file(tmp, name="CONTROLS_ok.json"):
    J, sel, box, preds, _ = RL.synth_world("good")
    P = RL.compute_values(J, sel, box, preds, 300)
    res = RL.assemble({k: {"PASS": True} for k in RL.CONTROLS}, P)
    p = tmp / name
    p.write_text(json.dumps(res), encoding="utf-8")
    return p, res


def test_names_stage_refuses_an_incomplete_or_failed_controls_file_and_writes_the_name_otherwise(tmp_path, capsys):
    p, res = good_controls_file(tmp_path)
    for name, mut in (("inc.json", lambda r: r["controls"].pop("k6")), ("fail.json", lambda r: r["controls"]["k3"].update(PASS=False)), ("flag.json", lambda r: r.update(controls_all_pass=False)),
                      ("nopayload.json", lambda r: r.pop("payload"))):
        r = json.loads(json.dumps(res))
        mut(r)
        q = tmp_path / name
        q.write_text(json.dumps(r), encoding="utf-8")
        assert RL.main(["--stage", "names", "--controls", str(q), "--out", str(tmp_path / ("R_" + name))]) == 3
        assert not (tmp_path / ("R_" + name)).exists()
    capsys.readouterr()
    outp = tmp_path / "RESULT.json"
    assert RL.main(["--stage", "names", "--controls", str(p), "--out", str(outp)]) == 0
    out = capsys.readouterr().out
    assert "name: LOOKAT-T0-LOCATES" in out and out.strip().splitlines()[-1] == "ZZ READ-LOOKAT names rc 0 controls_all_pass True"
    r = json.loads(outp.read_text(encoding="utf-8"))
    assert r["name"] == "LOOKAT-T0-LOCATES" and r["second_derivation_agrees"] is True and r["controls_sha256"] == CFT.sha256_file(str(p))
    log = (tmp_path / "RESULT_names.log").read_text(encoding="utf-8")
    for key in ("S4", "S8", "S4_fixed", "S8_fixed", "S8_out", "interval", "median d_az", "p90", "by gap", "by |az|", "not read"):
        assert key in log, key
    assert RL.main(["--stage", "names", "--controls", str(p), "--out", str(outp)]) == 2          # never written twice


def test_names_stage_stops_when_the_second_derivation_disagrees(tmp_path, monkeypatch, capsys):
    p, _ = good_controls_file(tmp_path)
    monkeypatch.setattr(RL, "name_second", lambda P: ("LOOKAT-T0-NOT-SEPARATED", {k: [0.0, 0.0] for k in ("s0", "s1", "s2")}))
    assert RL.main(["--stage", "names", "--controls", str(p), "--out", str(tmp_path / "R.json")]) == 4
    assert not (tmp_path / "R.json").exists() and "disagree" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ the queue (stubs)
STUB = r'''import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
for n in ("seed", "root", "box-dir", "out-root", "kit-fit", "join-train", "device"):
    ap.add_argument("--" + n)
for n in ("permute-targets", "predict", "identity"):
    ap.add_argument("--" + n, action="store_true")
a = ap.parse_args()
name = "ident" if a.identity else "s%s%s" % (a.seed, "p" if a.permute_targets else "")
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write("S %s %.3f %s\n" % (name, time.time(), " ".join(sys.argv[1:])))
time.sleep(float(os.environ.get("STUB_SLEEP", "0.4")))
rc = 0
if a.identity:
    ok = os.environ.get("STUB_IDENT_PASS", "1") == "1"
    os.makedirs(a.out_root, exist_ok=True)
    json.dump({"control": "k5", "PASS": ok}, open(os.path.join(a.out_root, "IDENTITY_K5.json"), "w"))
    rc = 0 if ok else 1
elif os.environ.get("STUB_FAIL") == name:
    rc = 7
else:
    nm = "LOC_s%s%s" % (a.seed, "_perm" if a.permute_targets else "")
    if not a.predict:
        os.makedirs(os.path.join(a.out_root, "fits", nm))
        json.dump({}, open(os.path.join(a.out_root, "fits", nm, "FIT_RECORD.json"), "w"))
    os.makedirs(os.path.join(a.out_root, "pred"), exist_ok=True)
    json.dump({}, open(os.path.join(a.out_root, "pred", nm + "_meta.json"), "w"))
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write("E %s %.3f\n" % (name, time.time()))
sys.exit(rc)
'''


def queue_world(tmp, gpu=0, mem_free_gb=100):
    r1, o, kit, kf = tmp / "run1", tmp / "O", tmp / "kit", tmp / "kitfit"
    (r1 / "lists").mkdir(parents=True)
    (r1 / "fits" / "D_H5_s0").mkdir(parents=True)
    kit.mkdir()
    kf.mkdir()
    (tmp / "box").mkdir()
    for f in (r1 / "lists" / "train_final.txt", r1 / "lists" / "eval141.txt", r1 / "fits" / "D_H5_s0" / "head.pt", r1 / "fits" / "D_H5_s0" / "FIT_RECORD.json", kf / "crop_fit.py", tmp / "join.npz"):
        f.write_text("x", encoding="utf-8")
    (kit / "locator_fit.py").write_text(STUB, encoding="utf-8")
    cg, shim = tmp / "cg", tmp / "shim"
    cg.mkdir()
    shim.mkdir()
    (cg / "memory.max").write_text(str((mem_free_gb + 1) * 1073741824) + "\n", encoding="utf-8")
    (cg / "memory.stat").write_text("anon 1073741824\nfile 5\n", encoding="utf-8")
    (shim / "nvidia-smi").write_bytes(("#!/bin/bash\n" + "".join("echo %d\n" % (100 + i) for i in range(gpu))).encode())
    os.chmod(str(shim / "nvidia-smi"), 0o755)
    env = {"LK_RUN1": posix(r1), "LK_ROOT": posix(o), "LK_KIT": posix(kit), "LK_KIT_FIT": posix(kf), "LK_BOX": posix(tmp / "box"), "LK_JOIN_TRAIN": posix(tmp / "join.npz"), "LK_PY": posix(sys.executable),
           "LK_CG": posix(cg), "LK_ULIMIT_SOFT": "1", "LK_POLL_S": "0", "LK_WAIT_S": "0", "LK_SPACING_S": "0", "STUB_LOG": posix(tmp / "stub.log"), "PATH": posix(shim) + os.pathsep + os.environ["PATH"]}
    return o, env


def run_q(env, extra=None, script=QUEUE):
    r = subprocess.run([bash_exe(), str(script)], env=dict(os.environ, **env, **(extra or {})), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=240)
    return r.returncode, r.stdout


def overlap(log):
    ev = []
    for ln in Path(log).read_text(encoding="utf-8").splitlines():
        p = ln.split()
        ev.append((float(p[2]), 1 if p[0] == "S" else -1))
    cur = mx = 0
    for _, d in sorted(ev):
        cur += d
        mx = max(mx, cur)
    return mx


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_dry_run_prints_every_command_and_makes_nothing(tmp_path):
    for f in (QUEUE, PODREAD, PUSH, PULL):
        assert subprocess.run([bash_exe(), "-n", str(f)]).returncode == 0, f
    gp = tmp_path / "lk"
    rc, out = run_q({"LK_ROOT": posix(gp)}, {"DRY": "1"})
    assert rc == 0 and not gp.exists(), out[-500:]
    cmds = [x.strip() for x in out.splitlines() if x.strip().startswith("CMD: ")]
    ident = [c for c in cmds if "--identity" in c]
    fits = [c for c in cmds if "--seed" in c and "--predict" not in c.split("(")[0]]
    assert len(ident) == 1 and "--join-train /workspace/v8proof/data/inputs/join_train.npz" in ident[0] and "--root /workspace/v8proof/cropread/run1" in ident[0]
    assert out.index("ident_k5") < out.index("JOB fit_s0 ")
    assert len(fits) == 4 and [("--seed %d" % s) in c for s, c in zip((0, 1, 2, 0), fits)] == [True] * 4 and ["--permute-targets" in c for c in fits] == [False, False, False, True]
    for c in fits:
        assert "--box-dir /workspace/v8proof/cropread/enc_out/box" in c and "--out-root %s" % posix(gp) in c and "--kit-fit /workspace/v8proof/cropread/kit/fit" in c and "--device cuda" in c
    for m in ("START_LOOKAT.txt", "DONE_LOOKAT.txt", "STOP_LOOKAT.txt", "ulimit -n 16384", "memory free > 28 GB", "fewer than 3 GPU compute processes", "at most 2 at a time"):
        assert m in out, m


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_runs_identity_first_then_four_fits_with_at_most_the_slots_and_writes_the_markers(tmp_path):
    o, env = queue_world(tmp_path)
    rc, out = run_q(env, {"LK_MAX_JOBS": "2"})
    assert rc == 0 and "ZZ LOOKAT-FITS DONE" in out, out[-800:]
    assert (o / "DONE_LOOKAT.txt").is_file() and (o / "START_LOOKAT.txt").is_file() and not (o / "STOP_LOOKAT.txt").exists()
    log = (tmp_path / "stub.log").read_text(encoding="utf-8").splitlines()
    assert log[0].startswith("S ident") and log[1].startswith("E ident")                 # the identity ran first and alone
    assert sorted(x.split()[1] for x in log if x.startswith("S ")) == ["ident", "s0", "s0p", "s1", "s2"]
    assert overlap(tmp_path / "stub.log") <= 2
    for j in ("ident_k5", "fit_s0", "fit_s1", "fit_s2", "fit_s0_perm"):
        assert (o / "queue" / "jobs" / (j + ".DONE")).is_file()
    for nm in ("LOC_s0", "LOC_s1", "LOC_s2", "LOC_s0_perm"):
        assert (o / "fits" / nm / "FIT_RECORD.json").is_file() and (o / "pred" / (nm + "_meta.json")).is_file()
    rc2, out2 = run_q(env)
    assert rc2 == 0 and "ALREADY-DONE" in out2


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_with_one_slot_never_overlaps(tmp_path):
    o, env = queue_world(tmp_path)
    rc, out = run_q(env, {"LK_MAX_JOBS": "1"})
    assert rc == 0 and overlap(tmp_path / "stub.log") == 1


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_failed_job_writes_the_stop_marker_and_resume_continues(tmp_path):
    o, env = queue_world(tmp_path)
    rc, out = run_q(env, {"STUB_FAIL": "s1", "LK_MAX_JOBS": "1"})
    assert rc == 3 and (o / "STOP_LOOKAT.txt").is_file() and not (o / "DONE_LOOKAT.txt").exists() and (o / "queue" / "jobs" / "fit_s1.STOP").is_file(), out[-600:]
    assert (o / "queue" / "jobs" / "fit_s0.DONE").is_file() and not (tmp_path / "O" / "fits" / "LOC_s2").exists()          # nothing starts after the failure
    rc, out = run_q(env)
    assert rc == 2 and "REFUSED" in out
    (tmp_path / "stub.log").write_text("", encoding="utf-8")
    rc, out = run_q(env, {"RESUME": "1"})
    assert rc == 0 and (o / "DONE_LOOKAT.txt").is_file(), out[-600:]
    started = [x.split()[1] for x in (tmp_path / "stub.log").read_text(encoding="utf-8").splitlines() if x.startswith("S ")]
    assert "ident" not in started and "s0" not in started and "s1" in started and "s2" in started        # finished jobs are skipped


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_identity_that_does_not_pass_stops_before_any_fit(tmp_path):
    o, env = queue_world(tmp_path)
    rc, out = run_q(env, {"STUB_IDENT_PASS": "0"})
    assert rc == 3 and (o / "STOP_LOOKAT.txt").is_file() and not (o / "fits" / "LOC_s0").exists() and "k5" in out


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_waits_for_a_slot_and_stops_when_gpu_is_full_or_memory_is_low(tmp_path):
    (tmp_path / "g").mkdir()
    o, env = queue_world(tmp_path / "g", gpu=3)
    rc, out = run_q(env)
    assert rc == 3 and "NO-SLOT" in out and not any((o / "fits").iterdir())
    (tmp_path / "m").mkdir()
    o2, env2 = queue_world(tmp_path / "m", mem_free_gb=10)
    rc, out = run_q(env2)
    assert rc == 3 and "NO-SLOT" in out
    (tmp_path / "ok").mkdir()
    o3, env3 = queue_world(tmp_path / "ok", gpu=2)                                       # two GPU processes: still a slot (fewer than three)
    rc, out = run_q(env3)
    assert rc == 0, out[-400:]


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_refuses_missing_input_with_a_stop_marker(tmp_path):
    o, env = queue_world(tmp_path)
    os.remove(str(tmp_path / "join.npz"))
    rc, out = run_q(env)
    assert rc == 3 and "MISSING-INPUT" in out and (o / "STOP_LOOKAT.txt").is_file()


def test_scripts_never_delete_move_or_stop_and_push_pull_are_as_the_model():
    for f in (QUEUE, PODREAD, PUSH, PULL):
        t = f.read_text(encoding="utf-8")
        assert not re.search(r"(^|[\s;&|(])(rm|mv|kill|pkill|rmdir)\s", t, re.M), f.name
    push, pull = PUSH.read_text(encoding="utf-8"), PULL.read_text(encoding="utf-8")
    assert "set -C" in push and "/workspace/v8proof/lookat" in push and "kit" in push and "lead_pytest.out" in push
    assert "pod_add107" in pull and '--exclude="*.pt"' in pull and '--exclude="*.npz"' in pull and 'if [ -e "$f" ]' in pull
    assert "podread" in PODREAD.name and "LOOKAT_CONTROLS.json" in PODREAD.read_text(encoding="utf-8")


def test_files_follow_the_wording_path_and_line_end_rules():
    for f in HERE.iterdir():
        if not f.is_file() or f.suffix not in (".py", ".sh", ".md"):
            continue
        t = f.read_text(encoding="utf-8")
        low = t.lower()
        for b in BANNED:
            assert b not in low, (f.name, b)
        if f.suffix == ".sh":
            assert b"\r" not in f.read_bytes(), f.name
        assert len(str(f)) <= 200 and f.stem not in ("aux", "con", "prn", "nul")
        assert ("hf" + "_") not in t and ("Keys" + ".txt") not in t
