"""test_steer_fit.py -- tests of the kit of card ADD100 (STEERREAD): localiser, folds and the refusal to steer a clip a localiser read, S_H5d sees the direction, dump round trip,
name rules with planted values, controls and stage order. Synthetic data only; nothing runs on the pod.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest cropread/steer_fit/test_steer_fit.py -q -p no:cacheprovider --basetemp /f/TanitAD-scratch/v8proof/cropread/steer_fit/tmp_pt/<new name>"""
import argparse
import hashlib
import json
import re
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import steer_common as SC       # noqa: E402
import loc_fit as LF            # noqa: E402
import loc_steer as LS          # noqa: E402
import crop_fit_s as CS         # noqa: E402
import crop_dump_s as CDS       # noqa: E402
import read_steerread as RS     # noqa: E402

CFT, RC = SC.CFT, RS.RC
K2, K4 = RC.K2, RC.K4
K2.load_tools(next(p for p in (str(HERE.parent.parent / "closeread"), RC.CR_TOOLS) if os.path.isfile(os.path.join(p, "closeread_fit.py"))))


def torch_threads():
    import torch
    torch.set_num_threads(2)


torch_threads()


# ------------------------------------------------------------------------------------------------ small worlds
def make_run1(base, n_fit=20, n_sel=3, n_eval=3, n=16, seed=0, signal=True, tokens=True):
    """run1 with lists, tok/D (two channels carry the direction of the box centre), box files for every clip, a join; returns names and the per-clip truth."""
    rng = np.random.default_rng(seed)
    root = Path(base)
    fit = ["tr-%02d" % i for i in range(n_fit)]
    sel = ["se-%02d" % i for i in range(n_sel)]
    ev = ["ev-%02d" % i for i in range(n_eval)]
    truth = {}
    (root / "tok" / "D").mkdir(parents=True, exist_ok=True)
    (root / "box").mkdir(exist_ok=True)
    for c in fit + sel + ev:
        az, el = rng.uniform(-10, 10, n), rng.uniform(-8, 8, n)
        has = rng.random(n) < 0.8
        truth[c] = (az, el, has)
        np.savez(root / "box" / (c + ".npz"), has_lead=has, ctr_az=np.where(has, az, 99.0), ctr_el=np.where(has, el, np.nan))
        if tokens:
            x = rng.normal(0, 0.3, (n, 2, 65, 1024)).astype(np.float32)
            if signal:
                x[..., 0] += (az / 3.0)[:, None, None]
                x[..., 1] += (el / 3.0)[:, None, None]
            np.save(root / "tok" / "D" / (c + ".npy"), x.astype(np.float16))
    (root / "lists").mkdir(exist_ok=True)
    (root / "lists" / "train_final.txt").write_text("".join("%s %s\n" % (c, "fit") for c in fit) + "".join("%s sel\n" % c for c in sel))
    (root / "lists" / "eval141.txt").write_text("".join(c + "\n" for c in ev))
    return root, fit, sel, ev, truth


def loc_args(root, s_root, name, epochs=30):
    return argparse.Namespace(set=name, run1=str(root), box_dir=str(root / "box"), s_root=str(s_root), device="cpu", seed=0, epochs=epochs, batch=32, lr=1e-3, weight_decay=0.05, warmup_epochs=1)


def fake_localisers(s_root, run1, seed=0):
    """six localisers with random weights and a correct record (clips read as the name defines) -- for the steering and control tests."""
    import torch
    rows, ev = SC.read_lists(str(run1))
    fit = SC.fit_clip_ids(rows)
    for k, n in enumerate(SC.LOC_NAMES):
        torch.manual_seed(seed + k)
        m = SC.make_loc_model()
        d = Path(s_root) / "loc" / n
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": m.state_dict(), "set": n}, d / "head.pt")
        SC.write_json(str(d / "LOC_RECORD.json"), {"set": n, "clips_read": SC.loc_train_clips(n, fit), "head_pt_sha256": SC.sha256_file(str(d / "head.pt"))})


@pytest.fixture(scope="module")
def locworld(tmp_path_factory):
    base = tmp_path_factory.mktemp("lw")
    root, fit, sel, ev, truth = make_run1(base / "run1")
    return {"root": root, "s_root": base / "s", "fit": fit, "sel": sel, "ev": ev, "truth": truth}


# ------------------------------------------------------------------------------------------------ localiser
def test_localiser_recovers_the_direction_and_reads_only_lead_frames(locworld):
    import torch
    root, s_root = locworld["root"], locworld["s_root"]
    assert LF.run_loc_fit(loc_args(root, s_root, "LA"), log=lambda s: None) == 0
    rec = SC.read_json(str(s_root / "loc" / "LA" / "LOC_RECORD.json"))
    assert rec["clips_read"] == locworld["fit"] and rec["set"] == "LA" and len(rec["history"]) == 30 and rec["kept_epoch"] == 29
    want_n = int(sum(locworld["truth"][c][2].sum() for c in locworld["fit"]))
    assert rec["n_samples"] == want_n                                    # frames with no lead (garbage angles planted there) are not samples
    assert rec["hparams_equal_card"] is False and rec["tool_sha256"] == SC.sha256_file(str(HERE / "loc_fit.py"))
    assert rec["train_list_sha256"] == SC.sha256_file(str(root / "lists" / "train_final.txt")) and set(rec["box_files_sha256"]) == set(locworld["fit"])
    assert rec["history"][-1]["train_loss"] < 0.5 * rec["history"][0]["train_loss"]
    model, _ = SC.load_localiser(str(s_root / "loc" / "LA"), torch.device("cpu"))
    store = CFT.TokenStore(str(root), "D")
    errs = []
    for c in locworld["ev"]:
        az, el, has = locworld["truth"][c]
        st = SC.steer_clip(model, store.get(c), torch.device("cpu"))
        assert st.dtype == np.float32 and st.shape == (len(az), 2) and np.isfinite(st).all()
        errs.append(np.abs(st[has] - np.stack([az, el], 1)[has]))
    e = np.concatenate(errs)
    assert np.median(e[:, 0]) < 2.5 and np.median(e[:, 1]) < 2.5, np.median(e, 0)       # targets spread over +-10 and +-8 degrees (a constant guess misses by about 5 and 4)
    with pytest.raises(SystemExit) as x:
        LF.run_loc_fit(loc_args(root, s_root, "LA", epochs=1), log=lambda s: None)
    assert "finished" in str(x.value)


def test_localiser_output_is_bounded_by_16_degrees_and_refuses_bad_sets(locworld):
    import torch
    m = SC.make_loc_model()
    out = m(torch.randn(4, CFT.N_TOK, CFT.WIDTH) * 50)
    assert out.shape == (4, 2) and float(out.abs().max()) <= 16.0
    with pytest.raises(SystemExit):
        LF.run_loc_fit(loc_args(locworld["root"], locworld["s_root"], "LX"), log=lambda s: None)
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in CFT.make_model("H1").parameters()) - (256 * 256 + 256 + 256 + 1) + (256 * 256 + 256 + 256 * 2 + 2)


def test_localiser_refuses_an_evaluation_clip_in_the_train_list(tmp_path):
    root, fit, sel, ev, _ = make_run1(tmp_path / "run1", n_fit=5, n_sel=1, n_eval=1, n=6)
    (root / "lists" / "eval141.txt").write_text("ev-00\ntr-02\n")
    with pytest.raises(SystemExit) as x:
        LF.run_loc_fit(loc_args(root, tmp_path / "s", "LA", epochs=1), log=lambda s: None)
    assert "evaluation clip" in str(x.value) and not (tmp_path / "s").exists()


# ------------------------------------------------------------------------------------------------ folds, audit, steering
def test_fold_logic_and_assignment():
    rows = [("c%03d" % i, "fit") for i in range(450)] + [("s%02d" % i, "sel") for i in range(50)]
    ev = ["e%03d" % i for i in range(141)]
    fit = SC.fit_clip_ids(rows)
    assert len(fit) == 450
    assign = SC.steer_assignment(rows, ev)
    assert len(assign) == 641 and len({c for c, _r, _l in assign}) == 641
    for k in range(5):
        tr = SC.loc_train_clips("L%d" % k, fit)
        assert len(tr) == 360 and all(fit.index(c) % 5 != k for c in tr)
        steered = [c for c, r, l in assign if l == "L%d" % k]
        assert len(steered) == 90 and all(fit.index(c) % 5 == k for c in steered) and not set(steered) & set(tr)
    assert SC.loc_train_clips("LA", fit) == fit
    assert all(l == "LA" for c, r, l in assign if r in ("sel", "eval")) and sum(1 for _c, _r, l in assign if l == "LA") == 191
    recs = {n: {"clips_read": SC.loc_train_clips(n, fit)} for n in SC.LOC_NAMES}
    assert SC.audit_assignment(assign, recs, rows, ev) == []
    r2 = json.loads(json.dumps(recs))
    r2["L2"]["clips_read"].append(fit[2])                      # planted: L2 read a clip of its own fold
    bad = SC.audit_assignment(assign, r2, rows, ev)
    assert any("steered by L2 which read it" in b for b in bad) and any("differs from the clips" in b for b in bad)
    r3 = json.loads(json.dumps(recs))
    r3["LA"]["clips_read"].append(ev[0])                       # planted: LA read an evaluation clip
    bad = SC.audit_assignment(assign, r3, rows, ev)
    assert any("read 1 evaluation" in b for b in bad) and any("steered by LA which read it" in b for b in bad)
    r4 = json.loads(json.dumps(recs))
    r4["L0"]["clips_read"].append("s00")                       # planted: a localiser read a selection clip
    assert SC.audit_assignment(assign, r4, rows, ev)
    del recs["L3"]
    assert any("no record for localiser L3" in b for b in SC.audit_assignment(assign, recs, rows, ev))


def test_sha_order_of_the_six_evaluation_clips_is_by_hex_of_the_clip_id():
    ev = ["ev-%03d" % i for i in range(20)]
    want = sorted(ev, key=lambda c: hashlib.sha256(c.encode()).hexdigest())[:6]
    assert SC.first_eval_by_sha(ev) == want and len(want) == 6
    assert SC.first_eval_by_sha(ev[:3]) == sorted(ev[:3], key=lambda c: hashlib.sha256(c.encode()).hexdigest())


@pytest.fixture(scope="module")
def steerworld(tmp_path_factory):
    base = tmp_path_factory.mktemp("sw")
    root, fit, sel, ev, truth = make_run1(base / "run1", n_fit=10, n_sel=2, n_eval=8, n=8, signal=False)
    s_root = base / "s"
    fake_localisers(s_root, root)
    a = argparse.Namespace(run1=str(root), s_root=str(s_root), device="cpu", resume=False)
    assert LS.run_steer(a, log=lambda s: None) == 0
    return {"root": root, "s_root": s_root, "fit": fit, "sel": sel, "ev": ev, "base": base}


def test_steering_files_are_the_assigned_localisers_output_on_every_frame(steerworld):
    import torch
    w = steerworld
    rows, ev = SC.read_lists(str(w["root"]))
    store, steer = CFT.TokenStore(str(w["root"]), "D"), SC.SteerStore(str(w["s_root"]))
    for clip, role, loc in SC.steer_assignment(rows, ev):
        tok = store.get(clip)
        st = steer.get(clip, tok.shape[0])
        m, _ = SC.load_localiser(str(w["s_root"] / "loc" / loc), torch.device("cpu"))
        direct = SC.steer_clip(m, tok, torch.device("cpu"), chunk=3)
        assert st.dtype == np.float32 and np.allclose(st, direct, atol=1e-4), (clip, loc)
    other, _ = SC.load_localiser(str(w["s_root"] / "loc" / "L4"), torch.device("cpu"))       # a clip of fold 0 is not the output of another localiser
    c0 = SC.fit_clip_ids(rows)[0]
    assert not np.allclose(steer.get(c0), SC.steer_clip(other, store.get(c0), torch.device("cpu")), atol=1e-3)
    s1 = SC.read_json(str(w["s_root"] / "ctl" / "S1.json"))
    s2 = SC.read_json(str(w["s_root"] / "ctl" / "S2.json"))
    assert s1["pass"] is True and s2["pass"] is True
    assert len(s1["detail"]["steer_sha256"]) == 10 + 2 + 8 and sum(s1["detail"]["n_by_localiser"].values()) == 20
    assert s2["clips"] == SC.first_eval_by_sha(w["ev"]) and len(s2["clips"]) == 6 and all(v["max_abs_diff_deg"] <= 1e-3 for v in s2["per_clip"].values())
    for c, h in s1["detail"]["steer_sha256"].items():
        assert SC.sha256_file(steer.path(c)) == h


def test_steering_refuses_a_localiser_that_read_the_clip_and_writes_nothing(steerworld, tmp_path):
    w = steerworld
    for tag in ("own_fold", "eval", "sel"):
        s_root = tmp_path / tag
        fake_localisers(s_root, w["root"])
        rows, ev = SC.read_lists(str(w["root"]))
        fit = SC.fit_clip_ids(rows)
        p = s_root / "loc" / ("L1" if tag == "own_fold" else "LA" if tag == "eval" else "L0") / "LOC_RECORD.json"
        r = SC.read_json(str(p))
        r["clips_read"].append(fit[1] if tag == "own_fold" else ev[0] if tag == "eval" else "se-00")
        SC.write_json(str(p), r)
        a = argparse.Namespace(run1=str(w["root"]), s_root=str(s_root), device="cpu", resume=False)
        with pytest.raises(SystemExit) as x:
            LS.run_steer(a, log=lambda s: None)
        assert "cross-fit audit failed" in str(x.value), tag
        assert not (s_root / "steer").exists() and not (s_root / "ctl").exists(), tag
    s_root = tmp_path / "headpt"                                             # a head.pt that differs from its record is refused as well
    fake_localisers(s_root, w["root"])
    with open(s_root / "loc" / "L2" / "head.pt", "ab") as fh:
        fh.write(b"x")
    with pytest.raises(SystemExit) as x:
        LS.run_steer(argparse.Namespace(run1=str(w["root"]), s_root=str(s_root), device="cpu", resume=False), log=lambda s: None)
    assert "sha256 differs" in str(x.value)


def test_steering_refuses_existing_files_and_a_second_run(steerworld):
    w = steerworld
    a = argparse.Namespace(run1=str(w["root"]), s_root=str(w["s_root"]), device="cpu", resume=False)
    with pytest.raises(SystemExit) as x:
        LS.run_steer(a, log=lambda s: None)
    assert "exists" in str(x.value)


def test_steering_resume_keeps_finished_files_and_checks_them(steerworld, tmp_path):
    w = steerworld
    s_root = tmp_path / "res"
    fake_localisers(s_root, w["root"])
    (s_root / "steer").mkdir()
    rows, ev = SC.read_lists(str(w["root"]))
    c0 = SC.fit_clip_ids(rows)[0]
    bad = np.full((3, 2), np.nan, np.float32)                                   # a half-written earlier file: wrong frame count and non-finite
    np.save(s_root / "steer" / (c0 + ".npy"), bad)
    a = argparse.Namespace(run1=str(w["root"]), s_root=str(s_root), device="cpu", resume=True)
    with pytest.raises(SystemExit):
        LS.run_steer(a, log=lambda s: None)


# ------------------------------------------------------------------------------------------------ controls s1 and s2 on the steering world
def head_fit_records(s_root, run1, tamper=None):
    rows, ev = SC.read_lists(str(run1))
    sha = SC.sha256_file(str(Path(run1) / "lists" / "train_final.txt"))
    for n in RS.EXPECTED_FITS:
        d = Path(s_root) / "fits" / n
        d.mkdir(parents=True, exist_ok=True)
        SC.write_json(str(d / "FIT_RECORD.json"), {"clips_read": {"fit": [c for c, r in rows if r == "fit"], "sel": [c for c, r in rows if r == "sel"]}, "train_list_sha256": sha, "hparams_equal_card": True})
    if tamper:
        p = Path(s_root) / "fits" / "S_H5d_s0" / "FIT_RECORD.json"
        r = SC.read_json(str(p))
        r["clips_read"]["sel"].append(ev[0])
        SC.write_json(str(p), r)


def test_control_s1_and_s2_pass_and_each_planted_fault_turns_red(steerworld, tmp_path):
    w = steerworld
    s_root = tmp_path / "s"
    shutil.copytree(w["s_root"], s_root)
    head_fit_records(s_root, w["root"])
    s1 = RS.control_s1(str(s_root), str(w["root"]))
    assert s1["pass"], s1["detail"]
    rows, ev = SC.read_lists(str(w["root"]))
    clips = [c for c, _r, _l in SC.steer_assignment(rows, ev)]
    shas = {c: SC.sha256_file(str(s_root / "steer" / (c + ".npy"))) for c in clips}
    SC.write_json(str(s_root / "ctl" / "S8.json"), {"pass": True, "steer_sha256": shas})
    s2 = RS.control_s2(str(s_root), str(w["root"]))
    assert s2["pass"], s2["detail"]
    # manifest form: clip -> {steer_sha256}
    SC.write_json(str(s_root / "ctl" / "S8.json"), {"pass": True})
    (s_root / "tok" / "S").mkdir(parents=True)
    SC.write_json(str(s_root / "tok" / "S" / "MANIFEST.json"), {"clips": {c: {"n_frames": 8, "sha256": "0" * 64, "steer_sha256": h} for c, h in shas.items()}})
    assert RS.control_s2(str(s_root), str(w["root"]))["pass"]
    SC.write_json(str(s_root / "tok" / "S" / "MANIFEST.json"), {"clips": {c: {"n_frames": 8} for c in clips}})            # no steering sha anywhere
    r = RS.control_s2(str(s_root), str(w["root"]))
    assert not r["pass"] and any("neither" in f for f in r["detail"]["failures"])
    SC.write_json(str(s_root / "tok" / "S" / "MANIFEST.json"), {"steer_sha256": dict(shas, **{clips[3]: "1" * 64})})        # one wrong sha
    r = RS.control_s2(str(s_root), str(w["root"]))
    assert not r["pass"] and any("differs from the file" in f for f in r["detail"]["failures"])
    # a steering file changed after the localiser wrote it: S1's sha and the carried sha differ from the file
    SC.write_json(str(s_root / "tok" / "S" / "MANIFEST.json"), {"steer_sha256": shas})
    p = s_root / "steer" / (clips[0] + ".npy")
    a = np.load(p)
    a[0, 0] += 0.5
    np.save(p, a)
    assert not RS.control_s2(str(s_root), str(w["root"]))["pass"]
    # s2 first half failing
    j = SC.read_json(str(s_root / "ctl" / "S2.json"))
    j["pass"] = False
    SC.write_json(str(s_root / "ctl" / "S2.json"), j)
    assert not RS.control_s2(str(s_root), str(w["root"]))["pass"]
    # s1: an evaluation clip in a head fit record; a missing head fit record; S1.json not pass
    head_fit_records(tmp_path / "s_t", w["root"], tamper=True)
    shutil.copytree(w["s_root"] / "loc", tmp_path / "s_t" / "loc")
    shutil.copytree(w["s_root"] / "ctl", tmp_path / "s_t" / "ctl")
    r = RS.control_s1(str(tmp_path / "s_t"), str(w["root"]))
    assert not r["pass"] and any("read an evaluation clip" in f for f in r["detail"]["failures"])
    os.remove(s_root / "fits" / "S_H5_s1" / "FIT_RECORD.json")
    assert not RS.control_s1(str(s_root), str(w["root"]))["pass"]
    j = SC.read_json(str(s_root / "ctl" / "S1.json"))
    j["pass"] = False
    SC.write_json(str(s_root / "ctl" / "S1.json"), j)
    head_fit_records(s_root, w["root"])
    assert not RS.control_s1(str(s_root), str(w["root"]))["pass"]


def test_find_steer_shas_accepts_the_three_forms_and_nothing_else():
    h = {c: ("%02d" % i) * 32 for i, c in enumerate(["a", "b", "c"])}
    assert SC.find_steer_shas({"x": {"steer_sha256": h}}, list(h)) == h
    assert SC.find_steer_shas({"clips": {c: {"steer_sha256": v, "sha256": "f" * 64} for c, v in h.items()}}, list(h)) == h
    assert SC.find_steer_shas({"rows": [{"clip": c, "steer_sha256": v} for c, v in h.items()]}, list(h)) == h
    assert SC.find_steer_shas({"sha256": h}, list(h)) == {}                               # token sha256 is not a steering sha
    assert SC.find_steer_shas({"steer_sha256": {"zz": "0" * 64}}, list(h)) == {}


# ------------------------------------------------------------------------------------------------ heads: S_H5d sees the direction, S_H5 does not
def make_dir_world(base, n_fit=12, n_sel=2, n_eval=3, n=12, seed=0):
    """tokens are one constant array (they carry nothing about the gap); the gap of frame f is 30 + 2 * azimuth of the steering of frame f. Held clips are the evaluation clips."""
    rng = np.random.default_rng(seed)
    root, s_root = Path(base) / "run1", Path(base) / "s"
    fit = ["tr-%02d" % i for i in range(n_fit)]
    sel = ["se-%02d" % i for i in range(n_sel)]
    ev = ["ev-%02d" % i for i in range(n_eval)]
    (root / "lists").mkdir(parents=True)
    (root / "lists" / "train_final.txt").write_text("".join("%s fit\n" % c for c in fit) + "".join("%s sel\n" % c for c in sel))
    (root / "lists" / "eval141.txt").write_text("".join(c + "\n" for c in ev))
    for d in ("tok/S", "steer", "ctl"):
        (s_root / d).mkdir(parents=True)
    gaps = []
    const_tok = rng.normal(0, 0.3, (1, 2, 65, 1024)).astype(np.float16)          # the same tokens on every frame of every clip: they carry nothing
    for c in fit + sel + ev:
        az = rng.uniform(-10, 10, n)
        el = rng.uniform(-5, 5, n)
        np.save(s_root / "steer" / (c + ".npy"), np.stack([az, el], 1).astype(np.float32))
        np.save(s_root / "tok" / "S" / (c + ".npy"), np.repeat(const_tok, n, 0))
        gaps.append(30.0 + 2.0 * az)
    names = fit + sel + ev
    np.savez(base / "join_train.npz", clip=np.array(names), off=(np.arange(len(names)) * n).astype(np.int64), n=np.full(len(names), n, np.int64),
             lead_gap=np.concatenate(gaps).astype(np.float32), lead_valid=np.ones(len(names) * n, np.int8))
    SC.write_json(str(s_root / "ctl" / "S8.json"), {"pass": True})
    SC.write_json(str(s_root / "tok" / "S" / "MANIFEST.json"), {"n": len(names)})
    return root, s_root, names, ev, n


def sfit_args(root, s_root, base, head, seed=0, perm=False, epochs=50):
    return argparse.Namespace(head=head, seed=seed, permute_labels=perm, run1=str(root), s_root=str(s_root), join_train=str(base / "join_train.npz"), device="cpu", epochs=epochs, batch=16,
                              lr=3e-3, weight_decay=0.05, warmup_epochs=1)


def held_r2_s(root, s_root, base, head, seed, perm, ev):
    import torch
    fdir = s_root / "fits" / CFT.fit_name("S", head, seed, perm)
    ck = torch.load(fdir / "head.pt", map_location="cpu")
    m = CS.make_model_s(head)
    m.load_state_dict(ck["state_dict"])
    store, steer = CFT.TokenStore(str(s_root), "S"), SC.SteerStore(str(s_root))
    J = np.load(base / "join_train.npz", allow_pickle=True)
    allc = [str(c) for c in J["clip"]]
    ys, qs = [], []
    for c in ev:
        out = CDS.head_outputs_s(m, store.get(c), steer.get(c), torch.device("cpu"))
        i = allc.index(c)
        y = J["lead_gap"][J["off"][i]:J["off"][i] + J["n"][i]].astype(np.float64)
        ok = np.isfinite(out)
        ys.append(y[ok])
        qs.append(out[ok])
    y, q = np.concatenate(ys), np.concatenate(qs)
    return float(1.0 - np.mean((q - y) ** 2) / np.var(y))


@pytest.fixture(scope="module")
def dirworld(tmp_path_factory):
    base = tmp_path_factory.mktemp("dw")
    root, s_root, names, ev, n = make_dir_world(base)
    return {"base": base, "root": root, "s_root": s_root, "ev": ev}


def test_s_h5d_fits_a_gap_carried_only_by_the_direction_and_s_h5_does_not(dirworld):
    w = dirworld
    assert CS.run_fit_s(sfit_args(w["root"], w["s_root"], w["base"], "H5d"), log=lambda s: None) == 0
    assert CS.run_fit_s(sfit_args(w["root"], w["s_root"], w["base"], "H5"), log=lambda s: None) == 0
    assert CS.run_fit_s(sfit_args(w["root"], w["s_root"], w["base"], "H5d", perm=True), log=lambda s: None) == 0
    r_d = held_r2_s(w["root"], w["s_root"], w["base"], "H5d", 0, False, w["ev"])
    r_t = held_r2_s(w["root"], w["s_root"], w["base"], "H5", 0, False, w["ev"])
    r_p = held_r2_s(w["root"], w["s_root"], w["base"], "H5d", 0, True, w["ev"])
    assert r_d > 0.8, r_d
    assert r_t < 0.3, r_t
    assert r_p < 0.3, r_p
    rec = SC.read_json(str(w["s_root"] / "fits" / "S_H5d_s0" / "FIT_RECORD.json"))
    assert rec["arm"] == "S" and rec["head"] == "H5d" and rec["permute_labels"] is False and rec["hparams_equal_card"] is False
    assert set(rec["clips_read"]["fit"]).isdisjoint(set(w["ev"])) and set(rec["clips_read"]["sel"]).isdisjoint(set(w["ev"]))
    assert rec["kept_sel_loss"] == min(h["sel_loss"] for h in rec["history"]) and len(rec["steer_sha256"]) == 14
    assert rec["train_list_sha256"] == SC.sha256_file(str(w["root"] / "lists" / "train_final.txt"))
    with pytest.raises(SystemExit):
        CS.run_fit_s(sfit_args(w["root"], w["s_root"], w["base"], "H5d", epochs=1), log=lambda s: None)               # a finished fit is refused


def test_head_shapes_the_mlp_input_is_1290_and_the_direction_enters_divided_by_16():
    import torch
    m = CS.make_model_s("H5d")
    assert m.mlp[0].in_features == 1290 and CS.DIM_FRAME == 258
    assert CS.make_model_s("H5").mlp[0].in_features == 1280
    x = torch.randn(2, 5, CFT.N_TOK, CFT.WIDTH)
    d = torch.zeros(2, 5, 2)
    m.eval()
    a, b = m(x, d), m(x, d + 8.0)
    assert a.shape == (2,) and not torch.allclose(a, b) and bool((a >= 0).all() and (a <= 60).all())
    # the vector the MLP reads has the direction / 16 at positions 256, 257 of every frame
    seen = {}
    h = m.mlp[0].register_forward_hook(lambda mod, inp, out: seen.update(v=inp[0].detach().clone()))
    m(x, torch.full((2, 5, 2), 16.0))
    h.remove()
    v = seen["v"].reshape(2, 5, 258)
    assert torch.allclose(v[:, :, 256:], torch.ones(2, 5, 2))


def test_fit_s_refuses_without_s8_and_with_a_wrong_frame_count_and_an_evaluation_clip(tmp_path):
    root, s_root, names, ev, n = make_dir_world(tmp_path, n_fit=4, n_sel=2, n_eval=1, n=8)
    os.remove(s_root / "ctl" / "S8.json")
    with pytest.raises(SystemExit) as x:
        CS.run_fit_s(sfit_args(root, s_root, tmp_path, "H5", epochs=1), log=lambda s: None)
    assert "S8.json" in str(x.value)
    SC.write_json(str(s_root / "ctl" / "S8.json"), {"pass": False})
    with pytest.raises(SystemExit):
        CS.run_fit_s(sfit_args(root, s_root, tmp_path, "H5", epochs=1), log=lambda s: None)
    SC.write_json(str(s_root / "ctl" / "S8.json"), {"pass": True})
    st = np.load(s_root / "steer" / "tr-00.npy")
    np.save(s_root / "steer" / "tr-00.npy", st[:-1])                                       # steering with one frame too few
    with pytest.raises(SystemExit) as x:
        CS.run_fit_s(sfit_args(root, s_root, tmp_path, "H5d", epochs=1), log=lambda s: None)
    assert "frames" in str(x.value)
    np.save(s_root / "steer" / "tr-00.npy", np.where(np.arange(8)[:, None] == 3, np.nan, st).astype(np.float32))      # a non-finite frame
    with pytest.raises(SystemExit) as x:
        CS.run_fit_s(sfit_args(root, s_root, tmp_path, "H5d", epochs=1), log=lambda s: None)
    assert "non-finite" in str(x.value)
    (root / "lists" / "eval141.txt").write_text("ev-00\ntr-01\n")
    with pytest.raises(SystemExit) as x:
        CS.run_fit_s(sfit_args(root, s_root, tmp_path, "H5", epochs=1), log=lambda s: None)
    assert "evaluation clip" in str(x.value)
    assert not (s_root / "fits").exists()


# ------------------------------------------------------------------------------------------------ dump round trip
def test_dump_round_trip_and_the_direction_reaches_only_s_h5d(tmp_path):
    import torch
    dumps, eps = RC.make_world(n_clips=4, T=30)
    pres = tmp_path / "present"
    RC.write_dumps(str(pres), dumps)
    s_root = tmp_path / "s"
    rng = np.random.default_rng(1)
    for c in dumps[0]["clips"]:
        (s_root / "tok" / "S").mkdir(parents=True, exist_ok=True)
        (s_root / "steer").mkdir(exist_ok=True)
        np.save(s_root / "tok" / "S" / (str(c) + ".npy"), rng.normal(0, 0.5, (30, 2, 65, 1024)).astype(np.float16))
        np.save(s_root / "steer" / (str(c) + ".npy"), rng.uniform(-10, 10, (30, 2)).astype(np.float32))
    outs = {}
    for head in ("H5", "H5d"):
        torch.manual_seed(3)
        m = CS.make_model_s(head)
        fd = s_root / "fits" / ("S_%s_s0" % head)
        fd.mkdir(parents=True)
        torch.save({"state_dict": m.state_dict(), "head": head, "arm": "S", "seed": 0}, fd / "head.pt")
        (fd / "FIT_RECORD.json").write_text("{}")
        a = argparse.Namespace(head=head, seed=0, permute_labels=False, s_root=str(s_root), present_dump=str(pres), device="cpu")
        assert CDS.run_dump_s(a, log=lambda s: None) == 0
        out = np.load(s_root / "dump" / ("S_%s" % head) / "reader_s0_score.npz", allow_pickle=False)
        outs[head] = out
        D0 = dumps[0]
        for k in CDS.CD.COPIED:
            assert np.array_equal(out[k], np.asarray(D0[k]), equal_nan=(np.asarray(D0[k]).dtype.kind == "f")), k
        store, steer = CFT.TokenStore(str(s_root), "S"), SC.SteerStore(str(s_root))
        for i, c in enumerate(D0["clips"]):
            sel = D0["cid"] == i
            direct = CDS.head_outputs_s(m, store.get(str(c)), steer.get(str(c)), torch.device("cpu"))
            assert np.allclose(out["gap"][sel], direct[D0["frame"][sel]], equal_nan=True, atol=1e-9)
        assert np.isnan(out["gap"][D0["frame"] < 4]).all() and np.isfinite(out["gap"][D0["frame"] >= 4]).all()
        assert (out["p"] == 1.0).all() and (out["clos"] == 0.0).all() and int(out["seed"]) == 0
        assert np.array_equal(out["valid"], np.asarray(D0["valid"], bool) & np.isfinite(out["gap"]))
        meta = SC.read_json(str(s_root / "dump" / ("S_%s" % head) / "reader_s0_meta.json"))
        assert len(meta["steer_sha256"]) == 4 and meta["dump_sha256"] == SC.sha256_file(str(s_root / "dump" / ("S_%s" % head) / "reader_s0_score.npz"))
        assert len(K2.CF.load_dump(str(s_root / "dump" / ("S_%s" % head) / "reader_s0_score.npz"))["frame"]) == len(D0["frame"])     # the reader's own loader accepts it
        with pytest.raises(SystemExit):
            CDS.run_dump_s(a, log=lambda s: None)
    # change one steering file: S_H5 output unchanged, S_H5d output changes
    c = str(dumps[0]["clips"][0])
    st = np.load(s_root / "steer" / (c + ".npy"))
    np.save(s_root / "steer" / (c + ".npy"), st + 4.0)
    for head in ("H5", "H5d"):
        m = CS.make_model_s(head)
        m.load_state_dict(torch.load(s_root / "fits" / ("S_%s_s0" % head) / "head.pt", map_location="cpu")["state_dict"])
        store = CFT.TokenStore(str(s_root), "S")
        new = CDS.head_outputs_s(m, store.get(c), np.load(s_root / "steer" / (c + ".npy")), torch.device("cpu"))
        old = CDS.head_outputs_s(m, store.get(c), st, torch.device("cpu"))
        same = np.allclose(new, old, equal_nan=True, atol=1e-9)
        assert same if head == "H5" else not same


def test_dump_refuses_steering_with_the_wrong_frame_count(tmp_path):
    import torch
    m = CS.make_model_s("H5d")
    with pytest.raises(SystemExit):
        CDS.head_outputs_s(m, np.zeros((10, 2, 65, 1024), np.float16), np.zeros((9, 2), np.float32), torch.device("cpu"))


# ------------------------------------------------------------------------------------------------ name rules with planted values
@pytest.mark.parametrize("ci,name", [([0.3, 0.5], "REACHED"), ([0.51, 0.9], "NOT-REACHED"), ([0.4, 0.7], "NOT-DECIDED"), (None, "NOT-DECIDED"), ([float("nan"), 0.4], "NOT-DECIDED")])
def test_cut_name_rules_for_both_stems(ci, name):
    assert RC.cut_name({"ci": ci}, RS.CUT_STEM) == "STEERREAD-CUT-BY-2-" + name
    assert RC.cut_name({"ci": ci}, RS.CUT_STEM_TOK) == "STEERREAD-TOKENS-ONLY-CUT-BY-2-" + name


@pytest.mark.parametrize("ci,fixed,bound", [([0.5, 0.99], "BEATS-FIXED", "NOT-SEPARATED-FROM-BOUND"), ([0.5, 1.0], "NOT-SEPARATED-FROM-FIXED", "NOT-SEPARATED-FROM-BOUND"),
                                            ([1.0, 1.5], "NOT-SEPARATED-FROM-FIXED", "NOT-SEPARATED-FROM-BOUND"), ([1.01, 1.5], "WORSE-THAN-FIXED", "BELOW-BOUND"),
                                            ([0.8, 1.2], "NOT-SEPARATED-FROM-FIXED", "NOT-SEPARATED-FROM-BOUND"), (None, "NOT-SEPARATED-FROM-FIXED", "NOT-SEPARATED-FROM-BOUND"),
                                            ([float("nan"), 2.0], "NOT-SEPARATED-FROM-FIXED", "NOT-SEPARATED-FROM-BOUND")])
def test_ratio_names_against_fixed_and_bound(ci, fixed, bound):
    assert RS.vs_fixed_name({"ci": ci}) == "STEERREAD-" + fixed
    assert RS.vs_bound_name({"ci": ci}) == "STEERREAD-" + bound


@pytest.mark.parametrize("ci,form", [([0.9, 0.99], "REACHES"), ([0.5, 0.8999], "BELOW"), ([0.8, 0.9], "NOT-DECIDED"), (None, "NOT-DECIDED")])
def test_per_form_names_by_the_tools_rule(ci, form):
    P = {"arms": {"ALL": {"RG10": {"r_ci": ci, "r_pooled": 0.5}}}}
    assert RC.form_name(P, "STEERREAD-S-H5d") == "STEERREAD-S-H5d-" + form and RC.form_name(P, "STEERREAD-S-H5") == "STEERREAD-S-H5-" + form


def test_localiser_stats_with_planted_errors(tmp_path):
    dumps, eps = RC.make_world(n_clips=6, T=40)
    pres = tmp_path / "present"
    RC.write_dumps(str(pres), dumps)
    s_root, box = tmp_path / "s", tmp_path / "box"
    (s_root / "steer").mkdir(parents=True)
    box.mkdir()
    D0 = dumps[0]
    errs = {0: (9.0, -9.0), 1: (7.9, 7.9), 2: (8.1, 0.0), 3: (0.0, 7.96), 4: (1.0, 0.5), 5: (2.0, 1.0)}      # inside W16: clips 1, 4, 5; outside: 0 (az 9), 2 (az 8.1), 3 (el 7.96)
    T = 40
    exp_in = exp_n = 0
    for i, c in enumerate(D0["clips"]):
        has = np.zeros(T, bool)
        has[np.asarray(D0["frame"])[(np.asarray(D0["cid"]) == i) & np.asarray(D0["lab_present"], bool)]] = True
        az, el = np.where(has, 5.0, np.nan), np.where(has, 2.0, np.nan)
        np.savez(box / (str(c) + ".npz"), has_lead=has, ctr_az=az, ctr_el=el)
        np.save(s_root / "steer" / (str(c) + ".npy"), np.stack([np.where(has, 5.0 + errs[i][0], 0.0), np.where(has, 2.0 + errs[i][1], 0.0)], 1).astype(np.float32))
        sc = np.zeros(T, bool)
        sc[np.asarray(D0["frame"])[(np.asarray(D0["cid"]) == i) & np.asarray(D0["valid"], bool)]] = True
        k = int((has & sc).sum())
        exp_n += k
        exp_in += k if i in (1, 4, 5) else 0
    r = RS.localiser_stats(str(box), str(s_root), str(pres))
    assert r["scored"]["n"] == exp_n and abs(r["scored"]["share_box_centre_inside_w16"] - exp_in / exp_n) < 1e-12
    assert r["all_frames"]["n"] >= r["scored"]["n"]
    pool = np.concatenate([np.full(int(((np.asarray(D0["cid"]) == i) & np.asarray(D0["lab_present"], bool) & np.asarray(D0["valid"], bool)).sum()), abs(errs[i][0])) for i in range(6)])
    assert abs(r["scored"]["az_abs"]["median"] - float(np.median(pool))) < 1e-5 and abs(r["scored"]["az_abs"]["p90"] - float(np.percentile(pool, 90))) < 1e-5


# ------------------------------------------------------------------------------------------------ controls assembled, stage order
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("rw")
    root, run1, box = base / "s", base / "run1", base / "box"
    dumps, eps, exp = RS.write_world(str(root), str(run1), str(box), T=60, n_clips=12)
    n = [len(K2.CF.build_rows(dumps[k], 25, eps)["y"]) for k in (0, 1)]
    K2.write_json(str(base / "c81.json"), {"controls": {"t0": {"frames": {"H25_seed%d" % k: {"n_tool": n[k]} for k in (0, 1)}}}})
    a = argparse.Namespace(dump_dir=str(root / "present"), episode_map=str(root / "epm.json"), add81_controls=str(base / "c81.json"), out_dir=str(base / "gr"), boot=100, seeds="0,1")
    assert K4.run_read(a, log=lambda s: None) == 0
    return {"base": base, "root": root, "run1": run1, "box": box, "gr": str(base / "gr" / "GAPRANGE_RESULT.json"), "eps": eps, "dumps": dumps}


def names_args(w, ctl_path, out):
    return argparse.Namespace(root=str(w["root"]), run1=str(w["run1"]), box_dir=str(w["box"]), controls=ctl_path, out=out, present_dump=str(w["root"] / "present"),
                              episode_map=str(w["root"] / "epm.json"), closeread_tools=None, boot=100)


def test_names_with_planted_dumps_and_every_control_blocks_it(world, tmp_path):
    w = world
    good = str(tmp_path / "good.json")
    RS.write_controls(good, {})
    o = []
    assert RS.run_names(names_args(w, good, str(tmp_path / "r.json")), log=o.append) == 0
    R = SC.read_json(str(tmp_path / "r.json"))
    assert R["names"]["CARD"] == "STEERREAD-CUT-BY-2-REACHED" and R["names"]["S-H5-cut"] == "STEERREAD-TOKENS-ONLY-CUT-BY-2-NOT-REACHED"
    assert R["names"]["vs-fixed"] == "STEERREAD-BEATS-FIXED" and R["names"]["vs-bound"] == "STEERREAD-BELOW-BOUND"
    assert o[0] == "NAME STEERREAD-CUT-BY-2-REACHED" and o[-1].startswith("ZZ READ-STEERREAD names rc 0")
    assert any(x.startswith("LOCALISER scored") for x in o) and any(x.startswith("GAPERR S-H5d NEAR") for x in o) and any(x.startswith("SPLIT S-H5d") for x in o) and any(x.startswith("PARTS-ADD98 S-H5d") for x in o)
    assert any(x.startswith("RG10 S-H5d") for x in o) and any("per-seed" in x for x in o)
    for n in RS.CTL_ORDER:
        bp = str(tmp_path / ("bad_%s.json" % n))
        RS.write_controls(bp, {n: False})
        ob = []
        assert RS.run_names(names_args(w, bp, str(tmp_path / ("rb_%s.json" % n))), log=ob.append) == 2
        assert not os.path.exists(tmp_path / ("rb_%s.json" % n)) and all(("NAME" not in x and "GAPERR" not in x) for x in ob)
    with pytest.raises(SystemExit):                                                     # a second read into the same file is refused
        RS.run_names(names_args(w, good, str(tmp_path / "r.json")), log=lambda s: None)


def test_check_controls_needs_all_eight_and_the_file_flag():
    allp = {"controls": {n: {"pass": True} for n in RS.CTL_ORDER}, "controls_all_pass": True}
    assert RS.check_controls(allp)[0] and RS.CTL_ORDER == ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8")
    for n in RS.CTL_ORDER:
        c = json.loads(json.dumps(allp))
        c["controls"][n]["pass"] = False
        assert not RS.check_controls(c)[0]
        c = json.loads(json.dumps(allp))
        del c["controls"][n]
        assert not RS.check_controls(c)[0]
    c = json.loads(json.dumps(allp))
    c["controls"]["s3"]["pass"] = "yes"
    assert not RS.check_controls(c)[0]
    c = json.loads(json.dumps(allp))
    c["controls_all_pass"] = False
    assert not RS.check_controls(c)[0]


def test_controls_stage_assembles_s3_s8_from_the_data_side_and_each_fault_is_red(world, steerworld, tmp_path):
    w, sw = world, steerworld
    root = tmp_path / "s"
    shutil.copytree(sw["s_root"], root)
    head_fit_records(root, sw["root"])
    rows, ev = SC.read_lists(str(sw["root"]))
    clips = [c for c, _r, _l in SC.steer_assignment(rows, ev)]
    SC.write_json(str(root / "ctl" / "S8.json"), {"pass": True, "steer_sha256": {c: SC.sha256_file(str(root / "steer" / (c + ".npy"))) for c in clips}})
    SC.write_json(str(root / "ctl" / "S3.json"), {"pass": True})
    shuffled = lambda D: dict(D, gap=_shuffle_gap(D))             # noqa: E731
    RC.write_dumps(str(root / "dump" / "S_H5d_perm"), {0: shuffled(w["dumps"][0])})
    (tmp_path / "py.out").write_text("....\n20 passed in 3s\n")

    def args(r, **kw):
        return argparse.Namespace(root=str(r), run1=str(sw["root"]), box_dir=str(w["box"]), present_dump=str(w["root"] / "present"), episode_map=str(w["root"] / "epm.json"), closeread_tools=None,
                                  gaprange_result=w["gr"], s7_pytest_out=str(tmp_path / "py.out"), out=None, boot=100, test_mode=True, **kw)
    out = []
    assert RS.run_controls(args(root), log=out.append) == 0
    assert [x for x in out if x[:1] == "s" and x[1].isdigit()] == ["%s PASS" % n for n in RS.CTL_ORDER]
    C = SC.read_json(str(root / "read" / "STEERREAD_CONTROLS.json"))
    assert C["controls_all_pass"] is True and set(C["controls"]) == set(RS.CTL_ORDER)
    faults = {}
    for n, mut in (("s3", lambda r: SC.write_json(str(r / "ctl" / "S3.json"), {"pass": False})), ("s8", lambda r: SC.write_json(str(r / "ctl" / "S8.json"), {"pass": "yes"})),
                   ("s7", lambda r: (tmp_path / "py.out").write_text("1 failed, 19 passed\n")),
                   ("s6", lambda r: RC.write_dumps(str(r / "dump" / "S_H5d_perm"), {0: RC.planted_dump(w["dumps"][0], 0.0)}))):
        r = tmp_path / ("f_" + n)
        shutil.copytree(root, r, ignore=shutil.ignore_patterns("read"))
        (tmp_path / "py.out").write_text("....\n20 passed in 3s\n")
        mut(r)
        assert RS.run_controls(args(r), log=lambda s: None) == 1, n
        C = SC.read_json(str(r / "read" / "STEERREAD_CONTROLS.json"))
        assert C["controls"][n]["pass"] is False and not C["controls_all_pass"], n
        assert all(C["controls"][k]["pass"] for k in ("s4", "s5") if k != n), n
    (tmp_path / "py.out").write_text("....\n20 passed in 3s\n")
    with pytest.raises(SystemExit):
        RS.run_controls(args(root), log=lambda s: None)                                 # the controls file exists: a second read is refused
    with pytest.raises(SystemExit):
        RS.run_controls(argparse.Namespace(**dict(vars(args(root)), boot=50, test_mode=False)), log=lambda s: None)    # s4 needs the 2,000-draw bootstrap


def _shuffle_gap(D):
    lg = np.asarray(D["lab_gap"], np.float64)
    pres = np.asarray(D["lab_present"], bool) & np.isfinite(lg)
    g = np.asarray(D["gap"], np.float64).copy()
    g[pres] = lg[pres][np.random.default_rng(1).permutation(int(pres.sum()))]
    return g


def test_stage_order_names_without_controls_is_refused(world, tmp_path):
    out = []
    assert RS.run_names(names_args(world, None, str(tmp_path / "r.json")), log=out.append) == 2 and "no controls file" in out[0]
    assert not os.path.exists(tmp_path / "r.json")


def bash_run(cmd):
    b = shutil.which("bash")
    if not b:
        pytest.skip("no bash")
    return subprocess.run([b, "-c", cmd], cwd=str(HERE), capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))


def test_queue_scripts_stage_order_dry_run_and_syntax():
    r = bash_run("DRY=1 bash pod_steer_fit_queue.sh loc")
    assert r.returncode == 0, r.stderr[-400:]
    jobs = [ln for ln in r.stdout.splitlines() if ln.startswith("JOB ")]
    assert jobs == ["JOB L0:", "JOB L1:", "JOB L2:", "JOB L3:", "JOB L4:", "JOB LA:", "JOB steer (after all six localisers):"]
    assert r.stdout.count("loc_fit.py --set") == 6 and r.stdout.count("loc_steer.py") == 1 and "crop_fit_s.py" not in r.stdout
    h = bash_run("DRY=1 bash pod_steer_fit_queue.sh heads")
    assert h.returncode == 0
    hj = [ln for ln in h.stdout.splitlines() if ln.startswith("JOB ")]
    assert hj == ["JOB S_H5_s0:", "JOB S_H5d_s0:", "JOB S_H5d_s0_perm:", "JOB S_H5_s1:", "JOB S_H5d_s1:"]
    assert h.stdout.count("--permute-labels") == 2 and "ctl/S8.json" in h.stdout and h.stdout.count("crop_dump_s.py") == 5 and "28 GB" in h.stdout
    perm_lines = [ln for ln in h.stdout.splitlines() if "--permute-labels" in ln]
    assert all("--head H5d --seed 0" in ln for ln in perm_lines)
    assert "at most 3 jobs" in h.stdout
    bad = bash_run("bash pod_steer_fit_queue.sh")
    assert bad.returncode == 2 and "usage" in bad.stdout
    for f in ("pod_steer_fit_queue.sh", "podread_steerread.sh", "push_steer_fit.sh"):
        assert bash_run("bash -n %s" % f).returncode == 0, f
    t = (HERE / "podread_steerread.sh").read_text(encoding="utf-8")
    assert "export KINCLOSE_CRTOOLS=/workspace/v8proof/closeread/tools" in t and "S7_OUT" in t and "STEERREAD_CONTROLS.json" in t
    q = (HERE / "pod_steer_fit_queue.sh").read_text(encoding="utf-8")
    assert not re.search(r"(^|[;&|\s])rm\s", q) and " mv " not in q and "kill" not in q.lower().replace("never kills", "")
    assert "MINGB=${CR_MIN_FREE_GB:-28}" in q


def test_read_selftest_runs_green():
    r = subprocess.run([sys.executable, str(HERE / "read_steerread.py"), "--selftest"], capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0 and "ZZ STEERREAD-SELFTEST rc 0" in r.stdout, r.stdout[-800:] + r.stderr[-800:]


def test_no_banned_words_in_the_kit():
    banned = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop"]
    for f in HERE.glob("*"):
        if f.suffix in (".py", ".sh", ".md") and f.name != "test_steer_fit.py":
            t = f.read_text(encoding="utf-8").lower()
            assert not [x for x in banned if x in t], (f.name, [x for x in banned if x in t])
