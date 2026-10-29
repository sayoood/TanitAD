"""soup_read.py's HEAVY layer (SoupHeavy, make_heavy, main) on STUBS: no dump, no rig, no soup, no checkpoint of a model.  The real data run of the same layer is soup_dry_run.py."""
import json
import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("SOUP_READ_DIR", HERE))
import soup_read as R  # noqa: E402

W = os.path.dirname(HERE)
LI = R.load_li(os.path.join(HERE, "ref", "launch_intent_read.py"))
BUILDER = os.environ.get("SOUP_BUILD_FILE", os.path.join(W, "soup_build.py"))
BUILDER_SHA = R.sha256_file(BUILDER)
N = 2399


def row(d=0.0, sep=False):
    w = abs(d) / 4 if sep else abs(d) + 0.1
    return {"delta": d, "lo": d - w, "hi": d + w, "separated": sep, "n": "2399/141", "mean_arm": 1.0 + d, "mean_ref": 1.0}


def table(over=None):
    t = {l: row() for l in LI.LABELS}
    t.update(over or {})
    return t


def fake_pb(a, b, eid, n_boot=2000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    ue = np.unique(eid)
    rng = np.random.default_rng(seed)
    s = np.array([d[eid == e].sum() for e in ue])
    n = np.array([(eid == e).sum() for e in ue])
    bs = [s[p].sum() / n[p].sum() for p in (rng.integers(0, len(ue), len(ue)) for _ in range(n_boot))]
    return {"delta": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


class StubDR:
    class Refusal(Exception):
        pass

    @staticmethod
    def need(p, what):
        if not p or not os.path.exists(p):
            raise StubDR.Refusal(f"{what}: {p!r} missing")
        return p

    @staticmethod
    def jload(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def tail2(p):
        return "/".join(str(p).replace("\\", "/").rstrip("/").split("/")[-2:])

    @staticmethod
    def t0_arrays(t0e, ckpt_tail):
        js = StubDR.jload(StubDR.need(t0e["json"], "T0 json"))
        z = np.load(StubDR.need(t0e["npz"], "T0 npz"))
        return js, {"ade6": np.asarray(z["ade6"], np.float64), "eid": np.asarray(z["eid"])}, StubDR.tail2(js.get("ckpt", "")) == ckpt_tail

    @staticmethod
    def g_t1cfg(label, hdr, ckpt_tail, want_limit, meta):
        return hdr.get("t1v8f", {}).get("limit") == want_limit and StubDR.tail2(hdr["t1v8f"].get("ckpt", "")) == ckpt_tail, {"limit": hdr.get("t1v8f", {}).get("limit")}

    DUMP = {}

    @staticmethod
    def load_dump(d):
        return StubDR.DUMP[d]


class Pack:
    def __init__(self, controls=None, n=N):
        self.controls = controls if controls is not None else {"SELF": (True, "ok")}
        self.meta = {"ckpt": "x/ckpt_last.pt", "tag": "t"}
        self.N = n
        self.ci = np.arange(n) % 141
        self.t0 = np.arange(n)


class StubBackend:
    def __init__(self, n=N):
        self.n = n
        self.k4_ok = True

    def analyze_t1(self, t1, tag, s):
        return Pack({"PACK": (True, "ok")}, self.n)

    def analyze_t1_light(self, lg, tag, s):
        return Pack({"LIGHT": (True, "ok")}, self.n)

    def k4(self, lg, pk, t0f, s):
        return self.k4_ok, "k4"


class StubHeavy:
    def __init__(self, orc_dir=None, work_dir=None, n_boot=2000):
        self.orc_dir, self.work_dir, self.n_boot = orc_dir, work_dir, n_boot
        self.recs, self.calls, self.tables, self.seen = {}, [], {}, []
        self.DR = StubDR
        self.K = types.SimpleNamespace(comps=lambda cl, g: {"ade6_m": np.asarray(cl, np.float64)})
        self.T = types.SimpleNamespace(PB=fake_pb)
        self.rig = types.SimpleNamespace(backend=StubBackend())

    def load_slot(self, ctl, role, seed, e, standin=False):
        self.calls.append(("load_slot", role, seed, standin))
        rec = {"role": role, "seed": seed, "ckpt_tail": e["ckpt_tail"], "t0arr": {"ade6": np.full(5, 1.0), "eid": np.arange(5)}, "t0cons": (np.zeros((5, 26)), np.arange(5))}
        self.recs.setdefault(role, {})[seed] = rec
        return rec

    def contrast(self, x, y):
        self.calls.append(("contrast", x, y))
        return self.tables[(x, y)]

    def guard_rows(self, ctl, x, y, s, tab_xy):
        self.seen.append(("guard", x, y, s, self.recs[x][s]["t0arr"]["ade6"].copy(), self.recs[y][s]["t0arr"]["ade6"].copy(), self.recs[y][s]["t0cons"][0].copy()))
        if getattr(self, "boom", False):
            raise RuntimeError("boom")
        return {"g": 1}

    def speed(self, x, y, s, rng=None):
        self.seen.append(("speed", x, y, s, self.recs[x][s]["t0cons"][0].copy(), self.recs[y][s]["t0cons"][0].copy()))
        if getattr(self, "boom", False):
            raise RuntimeError("boom")
        return {"cells": 1}

    def static_problems(self):
        return []

    def rf0(self, files):
        return [], {l: 0.0 for l in LI.LABELS}, {"n": len(files)}


STUB_LI = types.SimpleNamespace(**{**vars(LI), "Heavy": StubHeavy})


def heavy(tmp_path, standin=False, spec=None):
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec or json.load(open(os.path.join(W, "soups.json"), encoding="utf-8"))), encoding="utf-8")
    H = R.make_heavy(STUB_LI, "ORC", str(sp), BUILDER, BUILDER_SHA, str(tmp_path / "banked.json"), n_boot=123, work_dir="WORK", standin=standin)
    return H


def t0_files(tmp_path, name, ckpt, ade, eid=None):
    js, npz = tmp_path / f"{name}.json", tmp_path / f"{name}.npz"
    js.write_text(json.dumps({"ckpt": ckpt}), encoding="utf-8")
    np.savez(npz, ade6=np.asarray(ade, np.float64), eid=np.asarray(eid if eid is not None else np.arange(len(ade))), cons_pred=np.full((len(ade), 26), 2.0))
    return {"json": str(js), "npz": str(npz)}


# ---------------------------------------------------------------- make_heavy
def test_make_heavy_forwards_its_parameters_and_defaults_to_a_real_read(tmp_path):
    H = heavy(tmp_path)
    assert H.standin is False and H.orc_dir == "ORC" and H.work_dir == "WORK" and H.n_boot == 123 and H.na == [] and H.dev == {}
    assert heavy(tmp_path, standin=True).standin is True
    assert H.banked_path == str(tmp_path / "banked.json")


def test_make_heavy_refuses_a_builder_that_is_not_the_pinned_one(tmp_path):
    with pytest.raises(SystemExit, match="REFUSED"):
        R.make_heavy(STUB_LI, "ORC", os.path.join(W, "soups.json"), BUILDER, "0" * 64, "banked.json")


def test_make_heavy_refuses_a_builder_module_imported_from_elsewhere(tmp_path):
    other = tmp_path / "soup_build.py"
    other.write_text(open(BUILDER, encoding="utf-8").read(), encoding="utf-8")
    with pytest.raises(SystemExit, match="REFUSED"):
        R.make_heavy(STUB_LI, "ORC", os.path.join(W, "soups.json"), str(other), R.sha256_file(str(other)), "banked.json")


# ---------------------------------------------------------------- member slots and the dev-box T0
def entry(tmp_path, tail="a/ckpt_last.pt", dev=True, dev_ckpt=None, dev_ade=(3.0, 3.0, 3.0, 3.0, 3.0)):
    e = {"ckpt_tail": tail}
    if dev:
        e["t0_dev"] = t0_files(tmp_path, "dev", dev_ckpt or f"X/{tail}", dev_ade)
    return e


def test_ctrl_and_a_carry_the_dev_box_t0_and_the_others_do_not(tmp_path):
    H = heavy(tmp_path)
    ctl = LI.Ctl()
    for role in ("CTRL", "A"):
        rec = H.load_slot(ctl, role, 1, entry(tmp_path), False)
        assert (role, 1) in H.dev and np.array_equal(H.dev[(role, 1)]["t0arr"]["ade6"], np.full(5, 3.0)) and H.dev[(role, 1)]["t0cons"][0].shape == (5, 26)
        assert ctl.items[f"T0DEV-CKPT:{role}_s1"]["PASS"] is True
        assert ctl.items[f"K-T0DEV-BESIDE:{role}_s1"]["detail"] == {"dev_minus_box2_ade6_mean": 2.0} and rec["role"] == role
    for role in ("REP", "Arep"):
        H.load_slot(ctl, role, 0, {"ckpt_tail": "b/ckpt_last.pt"}, False)
        assert (role, 0) not in H.dev and f"T0DEV-CKPT:{role}_s0" not in ctl.items
    assert ("load_slot", "CTRL", 1, False) in H.calls


def test_a_missing_dev_t0_refuses_and_a_wrong_checkpoint_fails_the_control(tmp_path):
    H = heavy(tmp_path)
    ctl = LI.Ctl()
    for role in ("CTRL", "A"):
        with pytest.raises(StubDR.Refusal, match="t0_dev"):
            H.load_slot(ctl, role, 0, entry(tmp_path, dev=False), False)
    H.load_slot(ctl, "CTRL", 0, entry(tmp_path, dev_ckpt="X/other/ckpt_last.pt"), False)
    assert ctl.items["T0DEV-CKPT:CTRL_s0"]["PASS"] is False


def test_standin_flag_reaches_the_parent_slot_loader(tmp_path):
    H = heavy(tmp_path)
    H.load_slot(LI.Ctl(), "CTRL", 0, entry(tmp_path), True)
    assert ("load_slot", "CTRL", 0, True) in H.calls


# ---------------------------------------------------------------- the dev-box swap
@pytest.mark.parametrize("fn", ["guard_rows", "speed"])
def test_the_controls_box2_t0_is_swapped_for_the_dev_box_one_only_inside_the_call(tmp_path, fn):
    H = heavy(tmp_path)
    ctl = LI.Ctl()
    H.load_slot(ctl, "CTRL", 0, entry(tmp_path), False)
    H.recs["SOUP"] = {0: {"t0arr": {"ade6": np.full(5, 7.0), "eid": np.arange(5)}, "t0cons": (np.full((5, 26), 8.0), np.arange(5))}}
    before = H.recs["CTRL"][0]
    if fn == "guard_rows":
        assert H.guard_rows(ctl, "SOUP", "CTRL", 0, {}) == {"g": 1}
        _, x, y, s, ax, ay, cy = H.seen[-1]
        assert np.array_equal(ax, np.full(5, 7.0)) and np.array_equal(ay, np.full(5, 3.0)) and np.array_equal(cy, np.full((5, 26), 2.0))      # x: its own; y: the DEV arrays
    else:
        H.speed("SOUP", "CTRL", 0)
        _, x, y, s, cx, cy = H.seen[-1]
        assert np.array_equal(cx, np.full((5, 26), 8.0)) and np.array_equal(cy, np.full((5, 26), 2.0))
    assert H.recs["CTRL"][0] is before and np.array_equal(before["t0arr"]["ade6"], np.full(5, 1.0)) and np.array_equal(before["t0cons"][0], np.zeros((5, 26)))        # restored
    H.boom = True
    with pytest.raises(RuntimeError):
        getattr(H, fn)(*(({}, ) if False else ()), *( (ctl, "SOUP", "CTRL", 0, {}) if fn == "guard_rows" else ("SOUP", "CTRL", 0)))
    assert H.recs["CTRL"][0] is before                                  # restored even when the parent raises


def test_the_swap_uses_the_replicate_pairs_control_a(tmp_path):
    H = heavy(tmp_path)
    ctl = LI.Ctl()
    H.load_slot(ctl, "A", 1, entry(tmp_path, dev_ade=(9.0,) * 5), False)
    H.recs["SOUPR"] = {1: {"t0arr": {"ade6": np.full(5, 7.0), "eid": np.arange(5)}, "t0cons": (np.full((5, 26), 8.0), np.arange(5))}}
    H.guard_rows(ctl, "SOUPR", "A", 1, {})
    assert np.array_equal(H.seen[-1][5], np.full(5, 9.0))


# ---------------------------------------------------------------- contrast memo, K-RC
def test_contrast_is_computed_once_per_pair(tmp_path):
    H = heavy(tmp_path)
    H.tables[("A", "CTRL")] = {0: table(), 1: table()}
    H.tables[("CTRL", "A")] = {0: table({"ADE6": row(1.0)}), 1: table()}
    a, b = H.contrast("A", "CTRL"), H.contrast("A", "CTRL")
    assert a is b and H.contrast("CTRL", "A") is not a and [c for c in H.calls if c[0] == "contrast"] == [("contrast", "A", "CTRL"), ("contrast", "CTRL", "A")]


def krc_world(tmp_path, over=None, rf_over=None):
    tabs = {}
    k = 0
    for name, (x, y) in R.K_RC_TABLES.items():
        tabs[(x, y)] = {s: table({"ADE6": row(0.01 * (k + 1) * (1 if s == 0 else -1), True)}) for s in R.SEEDS}
        k += 1
    banked = {"read_a": {name: {f"s{s}": LI.trim(tabs[xy][s]) for s in R.SEEDS} for name, xy in R.K_RC_TABLES.items()},
              "floors": {"RF": LI.floors_from({s: tabs[("REP", "CTRL")][s] for s in R.SEEDS})}}
    if over:
        t, sd, lab, key, val = over
        banked["read_a"][t][sd][lab][key] = val
    if rf_over:
        banked["floors"]["RF"]["ADE6"] = rf_over
    (tmp_path / "banked.json").write_text(json.dumps(banked), encoding="utf-8")
    H = heavy(tmp_path)
    H.tables.update(tabs)
    return H


def test_k_rc_passes_on_the_banked_tables_and_calls_the_seven_pairs(tmp_path):
    H = krc_world(tmp_path)
    ctl = LI.Ctl()
    H.k_rc(ctl)
    assert sorted(ctl.items) == sorted([f"K-RC:{n}" for n in R.K_RC_TABLES] + ["K-RC:RF"]) and not ctl.failed()
    assert sorted((c[1], c[2]) for c in H.calls if c[0] == "contrast") == sorted(R.K_RC_TABLES.values())


@pytest.mark.parametrize("tab,sd,lab,key,val", [("A_CTRL", "s1", "ADE6", "delta", 0.5), ("Arep_REP", "s0", "ADE6", "lo", -9.0), ("CTRL_P10", "s0", "speed MAE", "hi", 9.0),
                                                ("REP_P10", "s1", "yaw-rate MAE", "separated", True), ("REP_CTRL", "s0", "ADE6", "mean_arm", 99.0), ("A_P10", "s1", "M1 crawl share", "n", "1/1"),
                                                ("Arep_P10", "s0", "TAC-LAT", "mean_ref", 5.0)])
def test_k_rc_fails_on_one_changed_key_of_any_one_banked_table(tmp_path, tab, sd, lab, key, val):
    H = krc_world(tmp_path, over=(tab, sd, lab, key, val))
    ctl = LI.Ctl()
    H.k_rc(ctl)
    assert sorted(ctl.failed()) == [f"K-RC:{tab}"] or (tab == "REP_CTRL" and f"K-RC:{tab}" in ctl.failed())


def test_k_rc_also_checks_rf_against_the_banked_floors(tmp_path):
    H = krc_world(tmp_path, rf_over=123.0)
    ctl = LI.Ctl()
    H.k_rc(ctl)
    assert sorted(ctl.failed()) == ["K-RC:RF"]


def test_k_rc_is_exact_not_rounded(tmp_path):
    H = krc_world(tmp_path, over=("A_CTRL", "s0", "ADE6", "delta", 0.01 + 1e-9))
    ctl = LI.Ctl()
    H.k_rc(ctl)
    assert "K-RC:A_CTRL" in ctl.failed()


# ---------------------------------------------------------------- SG
def test_sg_is_the_window_wise_gap_to_the_members_mean_and_requires_one_window_list(tmp_path):
    H = heavy(tmp_path)
    rng = np.random.default_rng(1)
    eid = np.repeat(np.arange(40), 10)
    a, b = rng.normal(2, 0.3, 400), rng.normal(2.5, 0.3, 400)
    for role, arr in (("SOUP", (a + b) / 2 + 0.2), ("CTRL", a), ("REP", b)):
        H.recs[role] = {0: {"dump": {"cl": arr, "g": None, "eid": eid}}}
    z = H.sg("SOUP", ("CTRL", "REP"), 0)
    assert abs(z["delta"] - 0.2) < 1e-9 and z["barrier"] and z["n_windows"] == 400
    assert z == R.soup_gain((a + b) / 2 + 0.2, [a, b], eid, fake_pb, 123)
    H.recs["REP"][0]["dump"]["eid"] = eid[::-1].copy()
    with pytest.raises(ValueError, match="windows differ"):
        H.sg("SOUP", ("CTRL", "REP"), 0)


# ---------------------------------------------------------------- weights-only global controls
def fake_basin(passes):
    def fn(spec, name, cache=None, clauses=("B1", "B2")):
        key = "full" if tuple(clauses) == ("B1", "B2") else ("B2" if tuple(clauses) == ("B2",) else "B1")
        return {"PASS": passes[key], "spread_ratio": 17.0}
    return fn


@pytest.mark.parametrize("passes,ok", [({"full": False, "B2": False, "B1": False}, True), ({"full": True, "B2": False, "B1": False}, False), ({"full": False, "B2": True, "B1": False}, False),
                                       ({"full": False, "B2": False, "B1": True}, False)])
def test_k_xinit_needs_the_cross_init_soup_to_fail_with_each_clause_alone(tmp_path, monkeypatch, passes, ok):
    import soup_build as SB
    monkeypatch.setattr(SB, "basin_for", fake_basin(passes))
    monkeypatch.setattr(SB, "load_ckpt", lambda p, want_sha=None: ({"model": {}}, "h"))
    monkeypatch.setattr(SB, "k_zero", lambda ck: {"PASS": True, "per_n": {"2": True}})
    H = heavy(tmp_path)
    ctl = LI.Ctl()
    H.global_controls(ctl)
    assert sorted(ctl.items) == ["K-XINIT:s0", "K-XINIT:s1", "K-ZERO-W"]
    assert ctl.items["K-XINIT:s0"]["PASS"] is ok and ctl.items["K-XINIT:s1"]["PASS"] is ok and ctl.items["K-ZERO-W"]["PASS"] is True and H.na == []
    assert {k: ctl.items["K-XINIT:s1"]["detail"][k] for k in ("full", "B2_only", "B1_only")} == {"full": passes["full"], "B2_only": passes["B2"], "B1_only": passes["B1"]}


def test_k_zero_w_follows_the_builders_k_zero_and_an_exception_is_a_fail(tmp_path, monkeypatch):
    import soup_build as SB
    monkeypatch.setattr(SB, "basin_for", fake_basin({"full": False, "B2": False, "B1": False}))
    seen = {}
    monkeypatch.setattr(SB, "load_ckpt", lambda p, want_sha=None: (seen.update(p=p, sha=want_sha) or {"model": {}}, "h"))
    monkeypatch.setattr(SB, "k_zero", lambda ck: {"PASS": False, "per_n": {"2": False}})
    H = heavy(tmp_path)
    ctl = LI.Ctl()
    H.global_controls(ctl)
    m = H.spec["members"]["DAGOACC_0"]
    assert ctl.items["K-ZERO-W"]["PASS"] is False and seen == {"p": m["path"], "sha": m["sha256"]}
    monkeypatch.setattr(SB, "basin_for", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no file")))
    ctl = LI.Ctl()
    H.global_controls(ctl)
    assert ctl.items["K-XINIT:s0"]["PASS"] is False and "RuntimeError" in ctl.items["K-XINIT:s0"]["detail"]


def test_standin_global_controls_are_named_not_passed(tmp_path):
    H = heavy(tmp_path, standin=True)
    ctl = LI.Ctl()
    H.global_controls(ctl)
    assert ctl.items == {} and H.na == ["K-XINIT:s0", "K-XINIT:s1", "K-ZERO-W"]


# ---------------------------------------------------------------- soup slot dispatch
def test_load_soup_slot_real_reads_the_soup_record_and_the_soup_checkpoint(tmp_path):
    H = heavy(tmp_path)
    calls = []
    H.soup_record_controls = lambda *a: calls.append(("rec",) + a[1:])
    H.load_soup_run = lambda ctl, role, s, e, tag, ck: calls.append(("run", role, s, tag, ck)) or "REC"
    ctl = LI.Ctl()
    out = H.load_soup_slot(ctl, "SOUPR", 1, {"soup_dir": "D/M2b_s1_soupr", "train_dir": "T"})
    assert out == "REC" and calls == [("rec", "SOUPR", 1, {"soup_dir": "D/M2b_s1_soupr", "train_dir": "T"}, "SOUPR_s1", "SOUPR_s1"), ("run", "SOUPR", 1, "SOUPR_s1", "D/M2b_s1_soupr/ckpt_last.pt")] and H.na == []


def test_load_soup_slot_standin_skips_the_soup_file_controls_and_names_them(tmp_path):
    H = heavy(tmp_path, standin=True)
    calls = []
    H.soup_record_controls = lambda *a: calls.append("rec")
    H.load_soup_run = lambda ctl, role, s, e, tag, ck: calls.append(("run", role, s, tag, ck))
    H.load_soup_slot(LI.Ctl(), "SOUP4", 0, {"soup_dir": "no", "train_dir": "E:/a/M2b_s0_launchintent"})
    assert calls == [("run", "SOUP4", 0, "SOUP4_s0", "E:/a/M2b_s0_launchintent/ckpt_last.pt")] and H.na == ["K-SOUP-REC:SOUP4_s0", "K-MID:SOUP4_s0", "K-BASIN:SOUP4_s0"]


def make_real_spec_and_soup(tmp_path, torch, opt=None, cfg_soup=None):
    donor = {"cfg": {"a": 1}, "model": {"w": torch.ones(3)}, "opt": None, "arm": "M2b"}
    dp = tmp_path / "donor.pt"
    torch.save(donor, dp)
    sd = tmp_path / "ckpt" / "M2b_s0_soup"
    sd.mkdir(parents=True)
    soup = {**donor, "cfg": cfg_soup or {"a": 1}, "opt": opt}
    torch.save(soup, sd / "ckpt_last.pt")
    spec = json.load(open(os.path.join(W, "soups.json"), encoding="utf-8"))
    spec["members"]["DAGOACC_0"]["path"] = str(dp)
    members = spec["soups"]["SOUP_s0"]["members"]
    rec = {"name": "SOUP_s0", "output_sha256": R.sha256_file(str(sd / "ckpt_last.pt")), "members": members, "donor": "DAGOACC_0", "member_sha256": {m: spec["members"][m]["sha256"] for m in members},
           "builder_sha256": BUILDER_SHA, "K-BASIN": {"PASS": True, "B1": True, "B2": True}, "K-MID": {"PASS": True}, "weights": [0.5, 0.5]}
    (sd / "soup.json").write_text(json.dumps(rec), encoding="utf-8")
    return spec, str(sd)


def test_soup_record_controls_run_k_soup_rec_k_mid_and_k_basin_from_the_files(tmp_path, monkeypatch):
    import soup_build as SB
    import torch
    seen = {}
    monkeypatch.setattr(SB, "verify", lambda spec, name, root: seen.update(v=(name, root)) or {"PASS": True, "problems": []})
    monkeypatch.setattr(SB, "basin_for", lambda spec, name, *a, **k: seen.update(b=name) or {"PASS": True, "spread_ratio": 1.2, "B1": True, "B2": True})
    spec, sd = make_real_spec_and_soup(tmp_path, torch)
    H = heavy(tmp_path, spec=spec)
    ctl = LI.Ctl()
    H.soup_record_controls(ctl, "SOUP", 0, {"soup_dir": sd}, "SOUP_s0", "SOUP_s0")
    assert not ctl.failed() and sorted(ctl.items) == ["K-BASIN:SOUP_s0", "K-MID:SOUP_s0", "K-SOUP-REC:SOUP_s0"]
    assert seen["v"] == ("SOUP_s0", os.path.dirname(sd)) and seen["b"] == "SOUP_s0"
    # a soup whose cfg is not its donor's, or that kept an optimizer state, fails K-SOUP-REC
    for kw in ({"cfg_soup": {"a": 2}}, {"opt": {"state": 1}}):
        spec, sd = make_real_spec_and_soup(tmp_path / "x", torch, **kw) if (tmp_path / "x").mkdir(exist_ok=True) is None else (None, None)
        H = heavy(tmp_path, spec=spec)
        ctl = LI.Ctl()
        H.soup_record_controls(ctl, "SOUP", 0, {"soup_dir": sd}, "SOUP_s0", "SOUP_s0")
        assert "K-SOUP-REC:SOUP_s0" in ctl.failed(), kw
        for f in os.listdir(tmp_path / "x"):
            pass
        import shutil
        shutil.rmtree(tmp_path / "x")
    # K-MID and K-BASIN failures are carried through
    monkeypatch.setattr(SB, "verify", lambda *a: {"PASS": False, "problems": ["bad"]})
    monkeypatch.setattr(SB, "basin_for", lambda *a, **k: {"PASS": False, "spread_ratio": 9.0, "B1": False, "B2": True})
    spec, sd = make_real_spec_and_soup(tmp_path / "y", torch) if (tmp_path / "y").mkdir() is None else (None, None)
    H = heavy(tmp_path, spec=spec)
    ctl = LI.Ctl()
    H.soup_record_controls(ctl, "SOUP", 0, {"soup_dir": sd}, "SOUP_s0", "SOUP_s0")
    assert set(ctl.failed()) == {"K-MID:SOUP_s0", "K-BASIN:SOUP_s0"} and ctl.items["K-MID:SOUP_s0"]["detail"] == ["bad"]


# ---------------------------------------------------------------- the per-run pipeline on a soup's rollouts
class RunWorld:
    def __init__(self, tmp_path, tail="M2b_s0_soup/ckpt_last.pt"):
        self.p, self.tail = tmp_path, tail
        ck = tmp_path / "ck" / "ckpt_last.pt"
        ck.parent.mkdir(exist_ok=True)
        ck.write_bytes(b"checkpoint bytes")
        self.ck = str(ck)
        self.sha = R.sha256_file(self.ck)
        self.eid = np.arange(N) % 141
        arr = np.random.default_rng(0).normal(2.0, 0.3, N)
        self.dump = {"ws": np.arange(N), "clip_index": self.eid, "eid": self.eid, "cl": arr, "fl": arr + 0.4, "g": None}
        StubDR.DUMP = {"dump_t1": self.dump}
        delta = fake_pb(arr, arr + 0.4, self.eid, 123, 0)["delta"]
        self.t1json = {"t1v8f": {"limit": "causal", "ckpt": f"X/{tail}"}, "paired_decision_grade": {"paired_model_minus_floor_t1": {"ade_m": {"delta": delta}}}}
        self.lgjson = {"t1v8f": {"limit": "logged", "ckpt": f"X/{tail}"}}
        self.entry = {"ckpt_tail": tail, "t1": {"tag": "T", "json": self.f("t1.json", self.t1json), "dump_dir": "dump_t1", "done": self.done("done_t1.txt")},
                      "t1_logged": {"tag": "L", "json": self.f("lg.json", self.lgjson), "dump_dir": "dump_lg", "done": self.done("done_lg.txt")},
                      "t0": t0_files(tmp_path, "t0", f"X/{tail}", np.ones(5)), "t0_fresh": {"json": self.f("t0f.json", {"ckpt": f"X/{tail}"}), "npz": ""}}

    def f(self, name, obj):
        pth = self.p / name
        pth.write_text(json.dumps(obj), encoding="utf-8")
        return str(pth)

    def done(self, name, **over):
        d = {"ckpt_sha256": self.sha, "ckpt_sha256_before": self.sha, "ckpt_sha256_after": self.sha, "tool_digest": R.TOOL_DIGEST, "smoke": False, "max_windows": None}
        d.update(over)
        return self.f(name, d)


def run_world(tmp_path, mutate=None, backend=None):
    w = RunWorld(tmp_path)
    if mutate:
        mutate(w)
    H = heavy(tmp_path)
    if backend:
        H.rig.backend = backend
    ctl = LI.Ctl()
    rec = H.load_soup_run(ctl, "SOUP", 0, w.entry, "SOUP_s0", w.ck)
    return H, ctl, rec, w


ALL = ["PACK:SOUP_s0", "T1-CONFIG:SOUP_s0", "LIGHT-logged:SOUP_s0", "T1-CONFIG-logged:SOUP_s0", "K4:SOUP_s0", "T0F-CKPT:SOUP_s0", "T0-CKPT:SOUP_s0", "DUMP-WINDOWS:SOUP_s0", "GRID-COUNTS:SOUP_s0",
       "R2-ADE6:SOUP_s0", "K-TREE:SOUP_s0"]


def test_the_per_run_pipeline_adds_every_control_and_stores_the_record(tmp_path):
    H, ctl, rec, w = run_world(tmp_path)
    assert sorted(ctl.items) == sorted(ALL) and not ctl.failed(), ctl.failed()
    assert H.recs["SOUP"][0] is rec and rec["role"] == "SOUP" and rec["seed"] == 0 and rec["tag"] == "SOUP_s0" and rec["ckpt_tail"] == w.tail
    assert rec["dump"] is w.dump and set(rec) >= {"t1json", "dump", "pack", "t0json", "t0arr", "t0cons", "entry"}
    assert ctl.items["K-TREE:SOUP_s0"]["detail"] == {"ckpt_sha256": w.sha, "tool_digest": R.TOOL_DIGEST}


def test_t0_checkpoint_tails_are_checked_for_the_fresh_and_the_standard_eval(tmp_path):
    def m1(w):
        w.entry["t0_fresh"]["json"] = w.f("t0f2.json", {"ckpt": "X/other/ckpt_last.pt"})
    assert set(run_world(tmp_path / "a" if (tmp_path / "a").mkdir() is None else tmp_path, m1)[1].failed()) == {"T0F-CKPT:SOUP_s0"}

    def m2(w):
        w.entry["t0"] = t0_files(w.p, "t0b", "X/other/ckpt_last.pt", np.ones(5))
    (tmp_path / "b").mkdir()
    assert set(run_world(tmp_path / "b", m2)[1].failed()) == {"T0-CKPT:SOUP_s0"}


def test_dump_windows_and_grid_counts_are_checked(tmp_path):
    def m1(w):
        w.dump["ws"] = w.dump["ws"][::-1].copy()
    assert set(run_world(tmp_path, m1)[1].failed()) == {"DUMP-WINDOWS:SOUP_s0"}
    (tmp_path / "d").mkdir()

    def m0(w):
        w.dump["clip_index"] = w.dump["clip_index"][::-1].copy()
    assert set(run_world(tmp_path / "d", m0)[1].failed()) == {"DUMP-WINDOWS:SOUP_s0"}
    (tmp_path / "b").mkdir()
    assert set(run_world(tmp_path / "b", None, StubBackend(n=N - 1))[1].failed()) >= {"GRID-COUNTS:SOUP_s0"}
    (tmp_path / "c").mkdir()

    def m3(w):
        w.dump["eid"] = np.arange(N) % 140
    assert "GRID-COUNTS:SOUP_s0" in run_world(tmp_path / "c", m3)[1].failed()


def test_t1_config_runs_on_both_lanes_with_their_own_limit(tmp_path):
    def m1(w):
        w.t1json["t1v8f"]["limit"] = "logged"
        w.entry["t1"]["json"] = w.f("t1b.json", w.t1json)
    assert set(run_world(tmp_path, m1)[1].failed()) >= {"T1-CONFIG:SOUP_s0"}
    (tmp_path / "b").mkdir()

    def m2(w):
        w.lgjson["t1v8f"]["limit"] = "causal"
        w.entry["t1_logged"]["json"] = w.f("lgb.json", w.lgjson)
    assert set(run_world(tmp_path / "b", m2)[1].failed()) == {"T1-CONFIG-logged:SOUP_s0"}


def test_k4_and_pack_controls_are_carried(tmp_path):
    be = StubBackend()
    be.k4_ok = False
    assert set(run_world(tmp_path, None, be)[1].failed()) == {"K4:SOUP_s0"}
    (tmp_path / "b").mkdir()
    be = StubBackend()
    be.analyze_t1 = lambda t1, tag, s: Pack({"BAD": (False, "x")})
    assert "BAD:SOUP_s0" in run_world(tmp_path / "b", None, be)[1].failed()
    (tmp_path / "c").mkdir()
    be = StubBackend()
    be.analyze_t1_light = lambda lg, tag, s: Pack({"BADL": (False, "x")})
    assert "BADL-logged:SOUP_s0" in run_world(tmp_path / "c", None, be)[1].failed()


def test_r2_ade6_reproduces_the_run_jsons_paired_delta(tmp_path):
    def m1(w):
        w.t1json["paired_decision_grade"]["paired_model_minus_floor_t1"]["ade_m"]["delta"] += 2e-3
        w.entry["t1"]["json"] = w.f("t1c.json", w.t1json)
    assert set(run_world(tmp_path, m1)[1].failed()) == {"R2-ADE6:SOUP_s0"}
    (tmp_path / "b").mkdir()

    def m2(w):
        del w.t1json["paired_decision_grade"]
        w.entry["t1"]["json"] = w.f("t1d.json", w.t1json)
    assert set(run_world(tmp_path / "b", m2)[1].failed()) == {"R2-ADE6:SOUP_s0"}
    (tmp_path / "c").mkdir()

    def m3(w):
        w.t1json["paired_decision_grade"]["paired_model_minus_floor_t1"]["ade_m"]["delta"] += 5e-7      # inside the 1e-6 tolerance
        w.entry["t1"]["json"] = w.f("t1e.json", w.t1json)
    assert not run_world(tmp_path / "c", m3)[1].failed()


@pytest.mark.parametrize("lane", ["t1", "t1_logged"])
@pytest.mark.parametrize("what", ["other_ckpt", "other_tree", "smoke", "absent"])
def test_k_tree_reads_both_lanes_done_records(tmp_path, lane, what):
    def m(w):
        name = f"d_{lane}.txt"
        if what == "other_ckpt":
            w.entry[lane]["done"] = w.done(name, ckpt_sha256="0" * 64)
        elif what == "other_tree":
            w.entry[lane]["done"] = w.done(name, tool_digest="0" * 64)
        elif what == "smoke":
            w.entry[lane]["done"] = w.done(name, smoke=True)
        else:
            w.entry[lane]["done"] = str(w.p / "nope.txt")
    H, ctl, rec, w = run_world(tmp_path, m)
    assert set(ctl.failed()) == {"K-TREE:SOUP_s0"} and lane in ctl.items["K-TREE:SOUP_s0"]["detail"]


def test_k_tree_binds_the_checkpoint_file_that_is_passed(tmp_path):
    H, ctl, rec, w = run_world(tmp_path)
    other = tmp_path / "other.pt"
    other.write_bytes(b"another file")
    ctl2 = LI.Ctl()
    H.load_soup_run(ctl2, "SOUP", 0, w.entry, "SOUP_s0", str(other))
    assert set(ctl2.failed()) == {"K-TREE:SOUP_s0"}


# ---------------------------------------------------------------- main
class MainHeavy:
    """read_core's FakeHeavy plus the three members main() calls on the heavy layer."""
    def __init__(self, rf0_problems=(), static=(), boom=None):
        sys.path.insert(0, HERE)
        import test_soup_read as T
        self.T = T
        self.inner = T.FakeHeavy(T.dom_scen())
        self.rf0_problems, self.static, self.boom = list(rf0_problems), list(static), boom
        self.DR = StubDR

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def static_problems(self):
        return self.static

    def rf0(self, files):
        self.files = files
        if self.rf0_problems:
            return self.rf0_problems, None, {}
        return [], {l: 0.0 for l in LI.LABELS}, {"n": len(files)}

    def load_soup_slot(self, ctl, role, s, e):
        if self.boom:
            raise self.boom
        return self.inner.load_soup_slot(ctl, role, s, e)


def main_world(tmp_path, monkeypatch, heavy_obj, prereg=None, prereg_sha=None, outcome=None):
    import test_soup_read as T
    pr = prereg or os.path.join(W, "PREREG_V8F_SOUP.md")
    sha = prereg_sha or R.sha256_file(pr)
    man = tmp_path / "man.json"
    man.write_text(json.dumps(T.MAN), encoding="utf-8")
    spy = {}
    monkeypatch.setattr(LI, "load_train_records", lambda e: {"fake": True})
    monkeypatch.setattr(LI, "run_record_controls", lambda ctl, trs, standin=False: ctl.add("K-MEMBERS", set(trs) == set(R.MEMBER_ROLES), sorted(trs)))
    monkeypatch.setattr(R, "load_li", lambda path, pin=None: spy.update(li=(path, pin)) or LI)
    monkeypatch.setattr(R, "make_heavy", lambda *a, **k: spy.update(args=a, kwargs=k) or heavy_obj)
    if outcome:
        monkeypatch.setattr(R, "decide", lambda ctrl, b: {"OUTCOME": outcome, "controls_pass": ctrl, "adopts": False, "matching_rows_in_table_order": [], "per_seed": {}})
    out = tmp_path / "out" / "SOUP_READ.json"
    argv = ["--manifest", str(man), "--out", str(out), "--prereg", pr, "--prereg-sha", sha, "--li-reader", "li.py", "--orc-dir", "ORC", "--banked-li-read", "banked.json",
            "--soup-build", BUILDER, "--soup-build-sha", BUILDER_SHA, "--spec", "spec.json", "--work-dir", "WD"]
    return argv, out, spy


def test_main_writes_the_outcome_and_returns_0(tmp_path, monkeypatch, capsys):
    argv, out, spy = main_world(tmp_path, monkeypatch, MainHeavy())
    assert R.main(argv) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["OUTCOME"]["OUTCOME"] == "DOMINANT / ADOPTED" and "read_a" in rec and rec["provenance"]["prereg_sha256"] == argv[argv.index("--prereg-sha") + 1]
    assert rec["provenance"]["reader_sha256"] == R.sha256_file(R.__file__) and rec["provenance"]["claim_row"] == "H-V8F-SOUP-1" and rec["provenance"]["li_reader_sha256"] == R.LI_READER_SHA
    assert rec["provenance"]["manifest_sha256"] == R.sha256_file(argv[1])
    assert "SOUP READ DOMINANT / ADOPTED" in capsys.readouterr().out
    # a real read: not a stand-in; the work dir and the pins reach the heavy layer
    assert spy["kwargs"].get("standin", False) is False and spy["kwargs"]["work_dir"] == "WD" and spy["args"][3] == BUILDER and spy["args"][4] == BUILDER_SHA
    assert spy["li"] == ("li.py", R.LI_READER_SHA) and spy["args"] == (LI, "ORC", "spec.json", BUILDER, BUILDER_SHA, "banked.json")


def test_main_work_dir_is_optional(tmp_path, monkeypatch):
    argv, out, spy = main_world(tmp_path, monkeypatch, MainHeavy())
    i = argv.index("--work-dir")
    assert R.main(argv[:i] + argv[i + 2:]) == 0 and spy["kwargs"]["work_dir"] is None


def test_main_refused_writes_failed_controls_and_no_row(tmp_path, monkeypatch, capsys):
    argv, out, _ = main_world(tmp_path, monkeypatch, MainHeavy(static=["a pin differs"]))
    assert R.main(argv) == 2
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["OUTCOME"]["OUTCOME"] == "REFUSED" and "STATIC" in rec["failed"] and rec["failed"]["STATIC"] == ["a pin differs"] and "read_a" not in rec and "tables" not in rec
    assert "REFUSED" in capsys.readouterr().err


def test_main_k_p0_failure_refuses_before_any_slot_is_read(tmp_path, monkeypatch):
    h = MainHeavy(rf0_problems=["RF0 is newer than a rollout"])
    argv, out, _ = main_world(tmp_path, monkeypatch, h)
    assert R.main(argv) == 2 and not h.inner.loaded
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["OUTCOME"]["OUTCOME"] == "REFUSED" and rec["failed"]["K-P0"] == ["RF0 is newer than a rollout"] and "read_a" not in rec


def test_main_passes_the_rollout_files_of_the_new_roles_to_k_p0(tmp_path, monkeypatch):
    h = MainHeavy()
    argv, out, _ = main_world(tmp_path, monkeypatch, h)
    monkeypatch.setattr(R, "rollout_files", lambda man: ["f1", "f2", "f3"])
    assert R.main(argv) == 0 and h.files == ["f1", "f2", "f3"]


@pytest.mark.parametrize("exc", [ValueError("x"), KeyError("k"), OSError("o"), StubDR.Refusal("r")])
def test_main_turns_a_read_error_into_a_refusal_without_writing(tmp_path, monkeypatch, exc, capsys):
    argv, out, _ = main_world(tmp_path, monkeypatch, MainHeavy(boom=exc))
    assert R.main(argv) == 2 and not out.exists() and "REFUSED" in capsys.readouterr().err


def test_main_row_not_computed_is_a_refusal(tmp_path, monkeypatch):
    argv, out, _ = main_world(tmp_path, monkeypatch, MainHeavy(boom=LI.RowNotComputed("ADE6")))
    assert R.main(argv) == 2 and not out.exists()


def test_main_refuses_a_prereg_that_is_not_the_pinned_one_before_loading_anything(tmp_path, monkeypatch, capsys):
    called = {}
    argv, out, spy = main_world(tmp_path, monkeypatch, MainHeavy(), prereg_sha="0" * 64)
    monkeypatch.setattr(R, "load_li", lambda *a, **k: called.update(li=True))
    assert R.main(argv) == 2 and not called and not spy and not out.exists() and "PREREG-TEXT" in capsys.readouterr().out


def test_main_unplaced_returns_3(tmp_path, monkeypatch):
    argv, out, _ = main_world(tmp_path, monkeypatch, MainHeavy(), outcome="UNPLACED")
    assert R.main(argv) == 3 and json.loads(out.read_text(encoding="utf-8"))["OUTCOME"]["OUTCOME"] == "UNPLACED"


def test_main_requires_the_prereg_and_its_sha_as_arguments(tmp_path, monkeypatch):
    argv, _, _ = main_world(tmp_path, monkeypatch, MainHeavy())
    for flag in ("--prereg", "--prereg-sha", "--manifest", "--out", "--soup-build-sha", "--spec"):
        i = argv.index(flag)
        with pytest.raises(SystemExit) as e:
            R.main(argv[:i] + argv[i + 2:])
        assert e.value.code == 2


def test_soup_record_controls_hash_the_soup_file_and_the_builder_themselves(tmp_path, monkeypatch):
    import soup_build as SB
    import torch
    monkeypatch.setattr(SB, "verify", lambda *a: {"PASS": True, "problems": []})
    monkeypatch.setattr(SB, "basin_for", lambda *a, **k: {"PASS": True, "spread_ratio": 1.2, "B1": True, "B2": True})
    spec, sd = make_real_spec_and_soup(tmp_path, torch)
    H = heavy(tmp_path, spec=spec)
    ctl = LI.Ctl()
    H.soup_record_controls(ctl, "SOUP", 0, {"soup_dir": sd}, "SOUP_s0", "SOUP_s0")
    assert not ctl.failed()
    rec = json.load(open(os.path.join(sd, "soup.json"), encoding="utf-8"))
    for key in ("output_sha256", "builder_sha256"):
        json.dump({**rec, key: "0" * 64}, open(os.path.join(sd, "soup.json"), "w", encoding="utf-8"))
        ctl = LI.Ctl()
        H.soup_record_controls(ctl, "SOUP", 0, {"soup_dir": sd}, "SOUP_s0", "SOUP_s0")
        assert set(ctl.failed()) == {"K-SOUP-REC:SOUP_s0"}, key


def test_sg_passes_the_heavy_layers_n_boot_and_seed_0_to_the_bootstrap(tmp_path):
    H = heavy(tmp_path)                                              # n_boot=123
    seen = {}

    def spy(a, b, eid, n_boot=None, seed=None):
        seen.update(n_boot=n_boot, seed=seed)
        return {"delta": 0.0, "lo": -1.0, "hi": 1.0}
    H.T = types.SimpleNamespace(PB=spy)
    rng = np.random.default_rng(3)
    eid = np.repeat(np.arange(40), 10)
    for role in ("SOUP", "CTRL", "REP"):
        H.recs[role] = {0: {"dump": {"cl": rng.normal(2, 0.3, 400), "g": None, "eid": eid}}}
    H.sg("SOUP", ("CTRL", "REP"), 0)
    assert seen == {"n_boot": 123, "seed": 0}


def test_sg_on_noisy_windows_equals_the_reference_at_the_heavy_layers_n_boot(tmp_path):
    H = heavy(tmp_path)
    rng = np.random.default_rng(5)
    eid = np.repeat(np.arange(40), 10)
    a, b, x = rng.normal(2, 0.3, 400), rng.normal(2.5, 0.3, 400), rng.normal(2.4, 0.4, 400)
    for role, arr in (("SOUP", x), ("CTRL", a), ("REP", b)):
        H.recs[role] = {0: {"dump": {"cl": arr, "g": None, "eid": eid}}}
    z = H.sg("SOUP", ("CTRL", "REP"), 0)
    assert z == R.soup_gain(x, [a, b], eid, fake_pb, 123) and z != R.soup_gain(x, [a, b], eid, fake_pb, 2000)


def test_the_dev_swap_forwards_the_speed_rng(tmp_path, monkeypatch):
    H = heavy(tmp_path)
    got = {}
    monkeypatch.setattr(StubHeavy, "speed", lambda self, x, y, s, rng=None: got.update(rng=rng) or {"cells": 1})
    ctl = LI.Ctl()
    H.load_slot(ctl, "CTRL", 0, entry(tmp_path), False)
    H.recs["SOUP"] = {0: {"t0arr": {"ade6": np.full(5, 7.0), "eid": np.arange(5)}, "t0cons": (np.full((5, 26), 8.0), np.arange(5))}}
    sentinel = object()
    H.speed("SOUP", "CTRL", 0, sentinel)
    assert got["rng"] is sentinel


def test_k_basin_is_asked_for_the_soup_being_read_not_for_seed_0(tmp_path, monkeypatch):
    import soup_build as SB
    import torch
    seen = {}
    monkeypatch.setattr(SB, "verify", lambda spec, name, root: seen.update(v=name) or {"PASS": True, "problems": []})
    monkeypatch.setattr(SB, "basin_for", lambda spec, name, *a, **k: seen.update(b=name) or {"PASS": True, "spread_ratio": 1.2, "B1": True, "B2": True})
    spec, sd = make_real_spec_and_soup(tmp_path, torch)
    spec["members"]["DAGOACC_1"]["path"] = spec["members"]["DAGOACC_0"]["path"]
    H = heavy(tmp_path, spec=spec)
    H.soup_record_controls(LI.Ctl(), "SOUP", 1, {"soup_dir": sd}, "SOUP_s1", "SOUP_s1")
    assert seen == {"v": "SOUP_s1", "b": "SOUP_s1"}
