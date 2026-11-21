"""test_fit.py -- tests of the kit of card ADD99 (CROPREAD): fit, dump, wrapper, name rules, controls, stage order. Synthetic data only; nothing runs on the pod.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest cropread/fit/test_fit.py -q -p no:cacheprovider --basetemp /f/TanitAD-scratch/v8proof/cropread/fit/tmp_pt/<new name>"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crop_dump as CD          # noqa: E402
import crop_fit as CFT          # noqa: E402
import read_cropread as RC      # noqa: E402
import make_part_joins as MMJ   # noqa: E402

K2, K4 = RC.K2, RC.K4
K2.load_tools(next(p for p in (str(HERE.parent.parent / "closeread"), RC.CR_TOOLS) if os.path.isfile(os.path.join(p, "closeread_fit.py"))))


# ------------------------------------------------------------------------------------------------ a small token world
def make_root(base, n_fit=16, n_sel=3, n_held=3, n=24, seed=0, signal=True):
    """tok/D and tok/B for train and held clips: 16 token channels carry the gap (signal) plus noise; lists, join, lists record."""
    rng = np.random.default_rng(seed)
    root = Path(base)
    names = ["tr-%02d" % i for i in range(n_fit + n_sel)] + ["ho-%02d" % i for i in range(n_held)]
    gaps = {}
    for c in names:
        gaps[c] = np.clip(rng.uniform(8, 50) + rng.uniform(-0.6, 0.6) * np.arange(n) + rng.normal(0, 0.3, n), 4.0, 56.0)
    for arm in ("D", "B"):
        d = root / "tok" / arm
        d.mkdir(parents=True, exist_ok=True)
        for c in names:
            x = rng.normal(0, 0.3, (n, 2, 65, 1024)).astype(np.float32)
            if signal:
                x[..., :16] += ((gaps[c] / 30.0 - 1.0) * 3.0)[:, None, None, None]
            np.save(d / (c + ".npy"), x.astype(np.float16))
    (root / "lists").mkdir(exist_ok=True)
    tr = names[:n_fit + n_sel]
    (root / "lists" / "train_final.txt").write_text("".join("%s %s\n" % (c, "fit" if i < n_fit else "sel") for i, c in enumerate(tr)))
    (root / "lists" / "eval141.txt").write_text("".join(c + "\n" for c in names[n_fit + n_sel:]))
    off = np.arange(len(names)) * n
    np.savez(root / "join_train.npz", clip=np.array(names), off=off.astype(np.int64), n=np.full(len(names), n, np.int64),
             lead_gap=np.concatenate([gaps[c] for c in names]).astype(np.float32), lead_valid=np.ones(len(names) * n, np.int8))
    return root, names, gaps


def fit_args(root, arm="D", head="H1", seed=0, perm=False, epochs=30):
    return argparse.Namespace(root=str(root), arm=arm, head=head, seed=seed, permute_labels=perm, join_train=str(root / "join_train.npz"), device="cpu", epochs=epochs, batch=32, lr=1e-3,
                              weight_decay=0.05, warmup_epochs=1)


def held_r2(root, arm, head, seed, perm, names):
    import torch
    fdir = root / "fits" / CFT.fit_name(arm, head, seed, perm)
    ck = torch.load(fdir / "head.pt", map_location="cpu")
    model = CFT.make_model(head)
    model.load_state_dict(ck["state_dict"])
    store = CFT.TokenStore(str(root), arm)
    J = np.load(root / "join_train.npz", allow_pickle=True)
    allc = [str(c) for c in J["clip"]]
    ys, ps = [], []
    for c in names:
        out = CD.head_outputs(model, store.get(c), torch.device("cpu"))
        i = allc.index(c)
        y = J["lead_gap"][J["off"][i]:J["off"][i] + J["n"][i]].astype(np.float64)
        m = np.isfinite(out)
        ys.append(y[m])
        ps.append(out[m])
    y, q = np.concatenate(ys), np.concatenate(ps)
    return float(1.0 - np.mean((q - y) ** 2) / np.var(y))          # R squared on held-out clips


@pytest.fixture(scope="module")
def tokworld(tmp_path_factory):
    root, names, gaps = make_root(tmp_path_factory.mktemp("tw"))
    return {"root": root, "names": names, "held": names[-3:]}


# ------------------------------------------------------------------------------------------------ fit: recovers the gap, a permuted fit does not, refusals
@pytest.mark.parametrize("head", ["H1", "H5"])
def test_fit_recovers_gap_and_permuted_fit_does_not(tokworld, head):
    root = tokworld["root"]
    assert CFT.run_fit(fit_args(root, head=head), log=lambda s: None) == 0
    assert CFT.run_fit(fit_args(root, head=head, perm=True), log=lambda s: None) == 0
    good = held_r2(root, "D", head, 0, False, tokworld["held"])
    perm = held_r2(root, "D", head, 0, True, tokworld["held"])
    assert good > 0.9, good
    assert perm < 0.3, perm
    rec = json.load(open(root / "fits" / ("D_%s_s0" % head) / "FIT_RECORD.json"))
    rp = json.load(open(root / "fits" / ("D_%s_s0_perm" % head) / "FIT_RECORD.json"))
    assert rec["kept_sel_loss"] < rp["kept_sel_loss"]
    assert rec["permute_labels"] is False and rp["permute_labels"] is True
    assert len(rec["history"]) == 30 and 0 <= rec["kept_epoch"] < 30
    assert rec["kept_sel_loss"] == min(h["sel_loss"] for h in rec["history"])
    assert set(rec["clips_read"]["fit"]).isdisjoint(set(tokworld["held"])) and set(rec["clips_read"]["sel"]).isdisjoint(set(tokworld["held"]))
    assert rec["train_list_sha256"] == CFT.sha256_file(str(root / "lists" / "train_final.txt"))
    assert rec["tool_sha256"] == CFT.sha256_file(str(HERE / "crop_fit.py"))
    assert rec["hparams_equal_card"] is False          # the test uses 30 epochs and batch 64, not the card's 40 and 256


def test_fit_refuses_an_evaluation_clip_in_the_train_list(tmp_path):
    root, names, _ = make_root(tmp_path / "r", n_fit=4, n_sel=2, n_held=2, n=8)
    ev = root / "lists" / "eval141.txt"
    ev.write_text(ev.read_text() + "tr-01\n")           # planted fault
    with pytest.raises(SystemExit) as e:
        CFT.run_fit(fit_args(root, epochs=1), log=lambda s: None)
    assert "evaluation clip" in str(e.value)
    assert not (root / "fits").exists()


def test_fit_refuses_a_finished_fit_and_a_frame_count_that_differs_from_the_join(tmp_path):
    root, names, _ = make_root(tmp_path / "r", n_fit=3, n_sel=2, n_held=1, n=8)
    CFT.run_fit(fit_args(root, epochs=1), log=lambda s: None)
    with pytest.raises(SystemExit):
        CFT.run_fit(fit_args(root, epochs=1), log=lambda s: None)
    x = np.load(root / "tok" / "B" / "tr-00.npy")
    np.save(root / "tok" / "B" / "tr-00.npy", x[:-1])
    with pytest.raises(SystemExit) as e:
        CFT.run_fit(fit_args(root, arm="B", epochs=1), log=lambda s: None)
    assert "join frames" in str(e.value)


def test_train_list_roles_and_token_store(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("a fit\nb other\n")
    with pytest.raises(SystemExit):
        CFT.read_train_list(str(p))
    p.write_text("a fit\na sel\n")
    with pytest.raises(SystemExit):
        CFT.read_train_list(str(p))
    root, names, _ = make_root(tmp_path / "r", n_fit=1, n_sel=1, n_held=1, n=6)
    a = CFT.TokenStore(str(root), "D").get("tr-00")
    assert isinstance(a, np.memmap) and a.shape == (6, 2, 65, 1024) and a.dtype == np.float16


def test_schedule_warmup_then_cosine_to_zero():
    spe, ep = 10, 40
    lrs = [CFT.lr_at(s, spe, ep, 1e-3) for s in range(spe * ep)]
    assert lrs[0] < lrs[spe - 1] <= 1e-3 + 1e-12 and abs(lrs[spe - 1] - 1e-3) < 1e-9
    assert lrs[spe] <= 1e-3 and all(b <= a + 1e-15 for a, b in zip(lrs[spe:], lrs[spe + 1:]))
    assert lrs[-1] < 1e-6


def test_model_shapes_and_dropout_only_in_training():
    import torch
    for head, nf in (("H1", 1), ("H5", 5)):
        m = CFT.make_model(head)
        x = torch.randn(3, nf, CFT.N_TOK, CFT.WIDTH)
        m.eval()
        a, b = m(x), m(x)
        assert a.shape == (3,) and torch.equal(a, b) and bool((a >= 0).all() and (a <= 60).all())
        m.train()
        assert not torch.equal(m(x), m(x))
    assert sum(p.numel() for p in CFT.make_model("H1").parameters()) < sum(p.numel() for p in CFT.make_model("H5").parameters())


# ------------------------------------------------------------------------------------------------ dump round trip
def test_dump_round_trip_against_a_present_dump(tmp_path):
    import torch
    dumps, eps = RC.make_world(n_clips=4, T=30)
    pres = tmp_path / "present"
    RC.write_dumps(str(pres), dumps)
    root = tmp_path / "root"
    rng = np.random.default_rng(1)
    for c in dumps[0]["clips"]:
        (root / "tok" / "D").mkdir(parents=True, exist_ok=True)
        np.save(root / "tok" / "D" / (str(c) + ".npy"), rng.normal(0, 0.5, (30, 2, 65, 1024)).astype(np.float16))
    for head in ("H1", "H5"):
        torch.manual_seed(3)
        m = CFT.make_model(head)
        fd = root / "fits" / ("D_%s_s0" % head)
        fd.mkdir(parents=True)
        torch.save({"state_dict": m.state_dict(), "head": head, "arm": "D", "seed": 0}, fd / "head.pt")
        (fd / "FIT_RECORD.json").write_text("{}")
        a = argparse.Namespace(arm="D", head=head, seed=0, permute_labels=False, root=str(root), present_dump=str(pres), device="cpu")
        assert CD.run_dump(a, log=lambda s: None) == 0
        out = np.load(root / "dump" / ("D_%s" % head) / "reader_s0_score.npz", allow_pickle=False)
        D0 = dumps[0]
        for k in CD.COPIED:
            assert np.array_equal(out[k], np.asarray(D0[k]), equal_nan=(np.asarray(D0[k]).dtype.kind == 'f')), k
        assert out["gap"].dtype == np.float64 and int(out["seed"]) == 0
        assert (out["p"] == 1.0).all() and (out["clos"] == 0.0).all()
        store = CFT.TokenStore(str(root), "D")
        for i, c in enumerate(D0["clips"]):
            sel = D0["cid"] == i
            direct = CD.head_outputs(m, store.get(str(c)), torch.device("cpu"))
            assert np.allclose(out["gap"][sel], direct[D0["frame"][sel]], equal_nan=True, atol=1e-9)
        if head == "H5":
            assert np.isnan(out["gap"][D0["frame"] < 4]).all() and np.isfinite(out["gap"][D0["frame"] >= 4]).all()
        else:
            assert np.isfinite(out["gap"]).all()
        assert np.array_equal(out["valid"], np.asarray(D0["valid"], bool) & np.isfinite(out["gap"]))
        assert (out["gap"][np.isfinite(out["gap"])] <= 60).all()
        Dl = K2.CF.load_dump(str(root / "dump" / ("D_%s" % head) / "reader_s0_score.npz"))        # the reader's own loader accepts it
        assert len(Dl["frame"]) == len(D0["frame"])
        with pytest.raises(SystemExit):
            CD.run_dump(a, log=lambda s: None)                                                       # an existing dump is refused


def test_dump_refuses_a_clip_with_fewer_token_frames_than_the_present_dump():
    dumps, _ = RC.make_world(n_clips=2, T=30)
    per = {str(c): np.zeros(20) for c in dumps[0]["clips"]}
    with pytest.raises(SystemExit):
        CD.dump_from_present(dumps[0], per, 0)


# ------------------------------------------------------------------------------------------------ the wrapper is the tool
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    dumps, eps = RC.make_world()
    base = tmp_path_factory.mktemp("wd")
    RC.write_dumps(str(base / "present"), dumps)
    K2.write_json(str(base / "epm.json"), eps)
    n = [len(K2.CF.build_rows(dumps[k], 25, eps)["y"]) for k in (0, 1)]
    K2.write_json(str(base / "c81.json"), {"controls": {"t0": {"frames": {"H25_seed%d" % k: {"n_tool": n[k]} for k in (0, 1)}}}})
    a = argparse.Namespace(dump_dir=str(base / "present"), episode_map=str(base / "epm.json"), add81_controls=str(base / "c81.json"), out_dir=str(base / "gr"), boot=100, seeds="0,1")
    assert K4.run_read(a, log=lambda s: None) == 0
    return {"dumps": dumps, "eps": eps, "base": base, "gr": str(base / "gr" / "GAPRANGE_RESULT.json")}


def test_wrapper_equals_kinclose4_called_directly(world):
    ref = K2.read_json(world["gr"])
    P = RC.measure_dumps(RC.load_ctxs(str(world["base"] / "present"), world["eps"]), 100)
    assert abs(P["arms"]["ALL"]["RG10"]["r_pooled"] - ref["arms"]["ALL"]["RG10"]["r_pooled"]) <= 1e-12
    assert np.allclose(P["arms"]["ALL"]["RG10"]["r_ci"], ref["arms"]["ALL"]["RG10"]["r_ci"], atol=1e-12)
    assert abs(P["gap_error"]["ALL"]["sd"] - ref["gap_error"]["ALL"]["sd"]) <= 1e-12
    assert np.allclose(P["gap_error"]["ALL"]["sd_ci"], ref["gap_error"]["ALL"]["sd_ci"], atol=1e-12)
    ctl = RC.control_c2(str(world["base"] / "present"), world["eps"], world["gr"], 100, check_shas=False)
    assert ctl["pass"] and ctl["detail"]["max_abs_diff"] <= 1e-12
    bad = json.load(open(world["gr"]))
    bad["arms"]["ALL"]["RG10"]["r_ci"][0] += 1e-4                       # planted: a different lower end
    bp = world["base"] / "gr_bad.json"
    bp.write_text(json.dumps(bad))
    assert not RC.control_c2(str(world["base"] / "present"), world["eps"], str(bp), 100, check_shas=False)["pass"]
    assert not RC.control_c2(str(world["base"] / "present"), world["eps"], world["gr"], 100, check_shas=True)["pass"]    # synthetic files do not carry the card's hashes


def test_tool_and_gaprange_hashes_of_the_card_match_the_files_here():
    assert RC.sha16(K4.__file__) == RC.TOOL_SHA16["kinclose4.py"]
    assert RC.sha16(RC.K6.__file__) == RC.TOOL_SHA16["kinclose6.py"]


# ------------------------------------------------------------------------------------------------ name rules and the cut estimator with planted values
def fake_P(r_ci, r=0.5):
    return {"arms": {"ALL": {"RG10": {"r_ci": r_ci, "r_pooled": r}}}}


@pytest.mark.parametrize("ci,card,form", [([0.9, 0.99], "PASS", "REACHES"), ([0.95, 0.99], "PASS", "REACHES"), ([0.5, 0.8999], "FAIL", "BELOW"), ([0.8, 0.9], "NOT-DECIDED", "NOT-DECIDED"),
                                          ([0.2, 0.95], "NOT-DECIDED", "NOT-DECIDED"), ([float("nan"), float("nan")], "NOT-DECIDED", "NOT-DECIDED"), (None, "NOT-DECIDED", "NOT-DECIDED")])
def test_card_and_form_name_rules(ci, card, form):
    P = fake_P(ci)
    assert RC.card_name(P) == "CROPREAD-PILOT-" + card
    assert RC.form_name(P, "CROPREAD-D-H5") == "CROPREAD-D-H5-" + form
    assert RC.card_name({"arms": {"ALL": {}}}) == "CROPREAD-PILOT-NOT-DECIDED"


@pytest.mark.parametrize("ci,name", [([0.3, 0.5], "REACHED"), ([0.1, 0.49], "REACHED"), ([0.51, 0.9], "NOT-REACHED"), ([0.4, 0.7], "NOT-DECIDED"), ([0.5, 0.6], "NOT-DECIDED"),
                                     ([float("nan"), 0.4], "NOT-DECIDED"), (None, "NOT-DECIDED")])
def test_cut_name_rules(ci, name):
    assert RC.cut_name({"ci": ci}, "CROPREAD-CUT-BY-2") == "CROPREAD-CUT-BY-2-" + name
    assert RC.cut_name({"ci": ci}, "CROPREAD-BOUND-CUT-BY-2") == "CROPREAD-BOUND-CUT-BY-2-" + name


def test_cut_estimator_with_planted_scales_and_a_noisy_pilot(world):
    dumps, eps = world["dumps"], world["eps"]
    cr = [K4.make_ctx(D, eps) for D in dumps.values()]
    for s, want in ((0.4, "REACHED"), (0.9, "NOT-REACHED"), (0.0, "REACHED")):
        cp = [K4.make_ctx(RC.scaled_dump(D, s), eps) for D in dumps.values()]
        c = RC.cut_ratio(cp, cr, B=200)
        assert abs(c["ratio"] - s) < 1e-9 and RC.cut_name(c, "X") == "X-" + want
    # a pilot with independent noise of the same size as the present reader's: ratio near 1, name NOT-REACHED
    rng = np.random.default_rng(0)
    def indep(D):
        lg = np.asarray(D["lab_gap"], np.float64)
        pres = np.asarray(D["lab_present"], bool) & np.isfinite(lg)
        sd = (np.asarray(D["gap"])[pres] - lg[pres]).std()
        g = np.asarray(D["gap"], np.float64).copy()
        g[pres] = lg[pres] + rng.normal(0, sd, int(pres.sum()))
        return dict(D, gap=g)
    c = RC.cut_ratio([K4.make_ctx(indep(D), eps) for D in dumps.values()], cr, B=300)
    assert 0.9 < c["ratio"] < 1.1 and RC.cut_name(c, "X") == "X-NOT-REACHED" and c["ci"][0] < c["ratio"] < c["ci"][1]


def test_cut_uses_only_the_frames_both_measure(world):
    dumps, eps = world["dumps"], world["eps"]
    cr = [K4.make_ctx(D, eps) for D in dumps.values()]
    def drop(D):                                         # the pilot has no value on the first 20 frames of every clip: the same frames drop out of the ratio's pooled set
        v = np.asarray(D["valid"], bool) & (np.asarray(D["frame"]) >= 20)
        return dict(D, valid=v)
    cp = [K4.make_ctx(drop(RC.scaled_dump(D, 0.4)), eps) for D in dumps.values()]
    c = RC.cut_ratio(cp, cr, B=100)
    assert abs(c["ratio"] - 0.4) < 1e-9 and c["n"] < sum(int(K4.meas(x).sum()) for x in cr)


# ------------------------------------------------------------------------------------------------ parts of card ADD98 on the raw gap, against the routine of make_part_joins
def test_parts_raw_equal_plain_loops_and_the_second_routine_of_make_part_joins():
    J, dumps, qmaps = MMJ.synthetic_all()
    D = dumps[0]
    st, pa = RC.parts_raw(D)
    plain = RC.parts_plain(D)
    ix = pa["idx"]
    assert len(ix) == len(plain) == st["n_mix_frames"]
    for i in ix:
        m, s, f = plain[int(i)]
        assert abs(pa["m"][i] - m) < 1e-9 and abs(pa["s"][i] - s) < 1e-9 and abs(pa["f"][i] - f) < 1e-9
    idq = {"xs": [0.0, 1000.0], "ys": [0.0, 1000.0]}                                         # min(g, Q(g)) = g: the raw gap
    second, runs = MMJ.second_parts(J, D, idq, 0)
    tab = MMJ.frame_table(J, D)
    by_row = {int(tab["row"][i]): int(i) for i in ix}
    assert set(second) == set(by_row)
    for row, v in second.items():
        i = by_row[row]
        assert abs(pa["m"][i] - v["m"]) < 1e-9 and abs(pa["s"][i] - v["s"]) < 1e-9 and abs(pa["f"][i] - v["f"]) < 1e-9
    assert sorted(runs) == sorted(int(x) for x in pa["length"])
    assert abs(st["m"]["rms"] - np.sqrt(np.mean(pa["m"][ix] ** 2))) < 1e-12


# ------------------------------------------------------------------------------------------------ controls and the stage order
def build_pod_like_root(base, world, p1=True):
    root = Path(base)
    for d in ("lists", "ctl", "fits", "dump", "read"):
        (root / d).mkdir(parents=True, exist_ok=True)
    tr = ["tr-%02d" % i for i in range(5)]
    (root / "lists" / "train_final.txt").write_text("".join("%s %s\n" % (c, "fit" if i < 4 else "sel") for i, c in enumerate(tr)))
    ev = [str(c) for c in world["dumps"][0]["clips"]]
    (root / "lists" / "eval141.txt").write_text("".join(c + "\n" for c in ev))
    sha = K2.sha256_file(str(root / "lists" / "train_final.txt"))
    (root / "lists" / "LISTS_RECORD.json").write_text(json.dumps({"train_final": {"sha256": sha}}))
    for n in RC.expected_fits(p1):
        d = root / "fits" / n
        d.mkdir()
        (d / "FIT_RECORD.json").write_text(json.dumps({"clips_read": {"fit": tr[:4], "sel": tr[4:]}, "train_list_sha256": sha, "eval_list_sha256": K2.sha256_file(str(root / "lists" / "eval141.txt")),
                                                      "hparams_equal_card": True, "kept_epoch": 3, "kept_sel_loss": 1.5, "history": [{}] * 40}))
    for n in ("C1", "C7", "P1"):
        (root / "ctl" / (n + ".json")).write_text(json.dumps({"pass": True}))
    dumps = world["dumps"]
    def shuffled(D):
        lg = np.asarray(D["lab_gap"], np.float64)
        pres = np.asarray(D["lab_present"], bool) & np.isfinite(lg)
        g = np.asarray(D["gap"], np.float64).copy()
        g[pres] = lg[pres][np.random.default_rng(1).permutation(int(pres.sum()))]
        return dict(D, gap=g)
    RC.write_dumps(str(root / "dump" / "D_H5_perm"), {0: shuffled(dumps[0])})
    (root / "pytest.out").write_text("....\n12 passed in 3.2s\n")
    return root


def controls_args(root, world, **kw):
    return argparse.Namespace(root=str(root), present_dump=str(world["base"] / "present"), episode_map=str(world["base"] / "epm.json"), closeread_tools=None, gaprange_result=world["gr"],
                              c6_pytest_out=str(root / "pytest.out"), out=None, boot=100, test_mode=True, **kw)


def test_controls_assemble_and_each_planted_fault_turns_its_control_red(world, tmp_path):
    root = build_pod_like_root(tmp_path / "ok", world)
    out = []
    assert RC.run_controls(controls_args(root, world), log=out.append) == 0
    C = json.load(open(root / "read" / "CROPREAD_CONTROLS.json"))
    assert C["controls_all_pass"] and [x for x in out if x[:2] in ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "p1")] == ["%s PASS" % n for n in RC.CTL_ORDER]
    assert not any("D-H5" in x for x in out)
    faults = {}
    r = build_pod_like_root(tmp_path / "f_eval", world)
    (r / "lists" / "eval141.txt").write_text((r / "lists" / "eval141.txt").read_text() + "tr-02\n")
    faults["c4"] = r
    r = build_pod_like_root(tmp_path / "f_c6", world)
    (r / "pytest.out").write_text("....F\n1 failed, 11 passed\n")
    faults["c6"] = r
    r = build_pod_like_root(tmp_path / "f_c1", world)
    (r / "ctl" / "C1.json").write_text(json.dumps({"pass": False}))
    faults["c1"] = r
    r = build_pod_like_root(tmp_path / "f_c7", world)
    (r / "ctl" / "C7.json").write_text(json.dumps({"pass": "yes"}))
    faults["c7"] = r
    r = build_pod_like_root(tmp_path / "f_c5", world)
    RC.write_dumps(str(r / "dump" / "D_H5_perm"), {0: RC.planted_dump(world["dumps"][0], 0.0)})      # a "permuted" fit that reads the label: REACHES
    faults["c5"] = r
    r = build_pod_like_root(tmp_path / "f_rec", world)
    rec = r / "fits" / "D_H1_s1" / "FIT_RECORD.json"
    j = json.load(open(rec))
    j["clips_read"]["fit"] = j["clips_read"]["fit"] + ["somewhere-else"]
    rec.write_text(json.dumps(j))
    faults["c4b"] = r
    for n, r in faults.items():
        assert RC.run_controls(controls_args(r, world), log=lambda s: None) == 1, n
        C = json.load(open(r / "read" / "CROPREAD_CONTROLS.json"))
        want = n[:2]
        assert not C["controls"][want]["pass"] and not C["controls_all_pass"], n
        assert all(C["controls"][k]["pass"] for k in ("c2", "c3") if k != want), n
    r = build_pod_like_root(tmp_path / "f_p1", world, p1=False)
    (r / "ctl" / "P1.json").write_text(json.dumps({"pass": False}))
    assert RC.run_controls(controls_args(r, world), log=lambda s: None) == 0         # p1 does not gate c1..c7; B fits are not expected then
    assert json.load(open(r / "read" / "CROPREAD_CONTROLS.json"))["p1_pass"] is False


def test_stage_order_names_before_controls_is_refused_and_arm_map(world, tmp_path):
    root = build_pod_like_root(tmp_path / "so", world)
    a = argparse.Namespace(root=str(root), controls=None, out=None, present_dump=str(world["base"] / "present"), episode_map=str(world["base"] / "epm.json"), closeread_tools=None, boot=100)
    out = []
    assert RC.run_names(a, log=out.append) == 2 and "no controls file" in out[0]
    assert not (root / "read" / "CROPREAD_RESULT.json").exists()
    assert RC.FORMS == (("D", "H5"), ("D", "H1"), ("B", "H5"), ("B", "H1"))
    assert [CFT.dump_name(x, y) for x, y in RC.FORMS] == ["D_H5", "D_H1", "B_H5", "B_H1"] and CFT.dump_name("D", "H5", True) == "D_H5_perm"
    assert CFT.fit_name("B", "H1", 1) == "B_H1_s1" and CFT.fit_name("D", "H5", 0, True) == "D_H5_s0_perm"
    assert RC.check_controls({"controls": {n: {"pass": True} for n in RC.CTL_ORDER}, "controls_all_pass": True})[0]
    assert not RC.check_controls({"controls": {n: {"pass": n != "c4"} for n in RC.CTL_ORDER}, "controls_all_pass": True})[0]
    assert not RC.check_controls({"controls": {n: {"pass": True} for n in RC.CTL_ORDER}, "controls_all_pass": False})[0]


def test_read_selftest_runs_green():
    r = subprocess.run([sys.executable, str(HERE / "read_cropread.py"), "--selftest"], capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0 and "ZZ CROPREAD-SELFTEST rc 0" in r.stdout, r.stdout[-800:] + r.stderr[-800:]


def test_no_banned_words_in_the_kit():
    banned = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop"]
    for f in HERE.glob("*"):
        if f.suffix in (".py", ".sh", ".md") and f.name != "test_fit.py":
            t = f.read_text(encoding="utf-8").lower()
            assert not [w for w in banned if w in t], (f.name, [w for w in banned if w in t])
