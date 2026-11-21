"""Unit tests for the C0 same-seed replicate read (PREREG_ADDENDUM_C0REP.md): F_rep / F2, the re-read under F2, the three labels, K-ARGV, the pinned prefixes, the arm classification and commit
widening of c0rep_run (the two replicate tags only), and the pod uploader's distinct-tag guard (the real script run against a fake hub).  Planted rows only; one mutation per rule."""
import copy
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c0rep_read as C  # noqa: E402
import c0rep_run as CRUN  # noqa: E402

SEEDS = C.SEEDS
CELLS = C.CELLS
LAT = C.DECIDING


def row(delta, sep, status="OK"):
    return {"status": status, "delta": delta, "lo": delta - 0.1 if sep else -0.1, "hi": delta + 0.1 if sep else 0.1, "separated": sep}


def pair_all(delta=0.0, sep=False):
    if delta == 0.0:                                   # a bit-identical pair reads delta, lo and hi exactly 0.0
        return {c: {"status": "OK", "delta": 0.0, "lo": 0.0, "hi": 0.0, "separated": False} for c in CELLS}
    return {c: row(delta, sep) for c in CELLS}


def seed_rows(lat_delta, sep=True):
    """one seed of a pair: TAC-LAT (higher is better) with the given delta, every other cell zero and not separated"""
    d = pair_all()
    d[LAT] = row(lat_delta, sep)
    return d


F0 = {c: 0.0 for c in CELLS}


def S1C0(deltas):
    return {s: seed_rows(deltas[s]) for s in SEEDS}


# ---- reflag: the card rule under another floor
def test_reflag_cost_gain_and_floor():
    r = row(-0.03, True)
    assert C.reflag(r, True, 0.01)["cost"] is True and C.reflag(r, True, 0.01)["gain"] is False      # higher-is-better cell, negative delta = cost
    assert C.reflag(r, True, 0.03)["cost"] is False                                                  # |delta| == floor is not above it
    assert C.reflag(r, True, 0.05)["cost"] is False and C.reflag(r, True, 0.05)["above_floor"] is False
    assert C.reflag(row(0.03, True), True, 0.01)["gain"] is True
    assert C.reflag(r, False, 0.01)["gain"] is True and C.reflag(r, False, 0.01)["cost"] is False   # lower-is-better cell flips the sides
    assert C.reflag(row(-0.03, False), True, 0.0)["cost"] is False                                   # not separated: never a cost
    assert C.reflag(r, True, None)["cost"] is False                                                  # no floor: not scorable
    assert C.reflag(row(0, False, status="NOT-APPLICABLE"), True, 0.0)["cost"] is False


def test_reflag_reproduces_the_frozen_cell_delta_flags():
    """the pure re-derivation equals obd_read.cell_delta's own flags on a planted dump"""
    import numpy as np
    CI = C.L.load_ci()
    rng = np.random.default_rng(1)
    eid = np.repeat(np.arange(30), 10)
    b = rng.normal(size=300)
    a = b + 0.2 + rng.normal(scale=0.1, size=300)
    for higher in (False, True):
        r = C.S.OR.cell_delta(CI, a, b, eid, higher, 0.05, 200)
        again = C.reflag(r, higher, 0.05)
        assert (again["cost"], again["gain"]) == (r["cost"], r["gain"]) and (r["cost"] or r["gain"])


# ---- F_rep and F2
def test_f_rep_is_the_max_abs_point_delta_over_pairs():
    p1, p2 = pair_all(), pair_all()
    p1[LAT], p2[LAT] = row(-0.04, True), row(0.01, False)
    fr = C.f_rep([p1, p2])
    assert fr[LAT] == 0.04 and fr[CELLS[0]] == 0.0
    assert C.f_rep([p2])[LAT] == 0.01
    p2[LAT] = row(0.0, False, status="NOT-APPLICABLE")
    assert C.f_rep([p1, p2])[LAT] == 0.04


def test_f2_is_max_of_f_and_f_rep_and_keeps_missing_floors():
    F = dict(F0, **{LAT: 0.0067})
    assert C.floor2(F, dict(F0, **{LAT: 0.0475}))[LAT] == 0.0475
    assert C.floor2(F, dict(F0, **{LAT: 0.001}))[LAT] == 0.0067
    F[CELLS[0]] = None
    assert C.floor2(F, dict(F0, **{CELLS[0]: 1.0}))[CELLS[0]] is None


# ---- the re-read under F2 with the card count rule
def test_cost4_holds_or_drops_with_the_floor():
    rows = S1C0({0: -0.03, 1: -0.03, 10: -0.03, 11: 0.0})                     # costs at 0, 1, 10 (11 not separated enough: delta 0)
    rows[11][LAT] = row(0.0, False)
    under_f, _ = C.read_under(rows, dict(F0, **{LAT: 0.01}))
    assert under_f["COST4"] == {LAT: [0, 1, 10]}
    under_f2, _ = C.read_under(rows, dict(F0, **{LAT: 0.05}))                 # F2 above every |delta|: no cost
    assert under_f2["COST4"] == {}


def test_gain4_is_read_too():
    rows = {s: seed_rows(0.03) for s in SEEDS}
    out, _ = C.read_under(rows, dict(F0, **{LAT: 0.01}))
    assert out["GAIN4"] == {LAT: [0, 1, 10, 11]} and out["COST4"] == {}


def mut_k4(per_seed, k=3):
    return C.SS.count_rule(per_seed, k=4)


def mut_k2(per_seed, k=3):
    return C.SS.count_rule(per_seed, k=2)


def test_mutation_threshold_3_to_4_and_3_to_2_flip():
    rows = S1C0({0: -0.03, 1: -0.03, 10: -0.03, 11: 0.0})
    rows[11][LAT] = row(0.0, False)
    fl = dict(F0, **{LAT: 0.01})
    assert C.read_under(rows, fl)[0]["COST4"] == {LAT: [0, 1, 10]}
    assert C.read_under(rows, fl, rule=mut_k4)[0]["COST4"] == {}
    two = S1C0({0: -0.03, 1: -0.03, 10: 0.0, 11: 0.0})
    two[10][LAT], two[11][LAT] = row(0.0, False), row(0.0, False)
    assert C.read_under(two, fl)[0]["COST4"] == {}
    assert C.read_under(two, fl, rule=mut_k2)[0]["COST4"] == {LAT: [0, 1]}


def test_mutation_one_init_only_never_counts():
    only_init0 = {0: -0.03, 10: -0.03, 1: 0.0, 11: 0.0}
    rows = S1C0(only_init0)
    rows[1][LAT], rows[11][LAT] = row(0.0, False), row(0.0, False)
    assert C.read_under(rows, dict(F0, **{LAT: 0.01}))[0]["COST4"] == {}


# ---- the labels
def costing():
    return {LAT: [0, 1, 10]}


def test_label_cost_holds():
    p = [pair_all(0.001, True), pair_all(0.002, False)]
    assert C.c0rep_label(p, costing(), costing())[0] == "COST-HOLDS"


def test_label_cost_within_floor():
    p = [pair_all(0.001, True), pair_all(0.002, False)]
    assert C.c0rep_label(p, costing(), {})[0] == "COST-WITHIN-FLOOR"


def test_label_deterministic_needs_every_cell_zero_in_both_pairs():
    z = [pair_all(0.0), pair_all(0.0)]
    assert C.c0rep_label(z, costing(), costing())[0] == "DETERMINISTIC-AT-SEED"
    one_off = [pair_all(0.0), pair_all(0.0)]
    one_off[1][CELLS[5]] = row(1e-9, False)
    assert C.c0rep_label(one_off, costing(), costing())[0] == "COST-HOLDS"
    ci_off = [pair_all(0.0), pair_all(0.0)]
    ci_off[0][CELLS[3]] = dict(row(0.0, False), lo=-1e-9)                     # a point delta 0.0 with a non-degenerate interval is not a bit-identical read
    assert C.c0rep_label(ci_off, costing(), costing())[0] != "DETERMINISTIC-AT-SEED"


def test_label_deterministic_wins_over_the_cost_branches_and_branches_are_distinct():
    z = [pair_all(0.0), pair_all(0.0)]
    assert C.c0rep_label(z, {}, {})[0] == "DETERMINISTIC-AT-SEED"
    nz = [pair_all(0.001), pair_all(0.001)]
    labs = {C.c0rep_label(nz, costing(), costing())[0], C.c0rep_label(nz, costing(), {})[0], C.c0rep_label(nz, {}, {})[0], C.c0rep_label(z, costing(), costing())[0], C.c0rep_label(nz[:1], {}, {})[0]}
    assert labs == {"COST-HOLDS", "COST-WITHIN-FLOOR", "PREMISE-ABSENT", "DETERMINISTIC-AT-SEED", "NOT-COMPUTABLE"}
    assert set(C.NEXT) == labs and len(set(C.NEXT.values())) == 5


def test_end_to_end_label_from_rows_swaps_with_the_floor():
    """the full chain: planted S1 - C0 rows, F and F_rep -> F2 -> COST4 -> label"""
    rows = S1C0({0: -0.0225, 1: -0.0150, 10: -0.0150, 11: -0.0250})
    F = dict(F0, **{LAT: 0.0067})
    for fr_lat, want in ((0.001, "COST-HOLDS"), (0.0475, "COST-WITHIN-FLOOR")):
        p = [pair_all(fr_lat, True), pair_all(0.0, False)]
        F2 = C.floor2(F, C.f_rep(p))
        lab, _ = C.c0rep_label(p, C.read_under(rows, F)[0]["COST4"], C.read_under(rows, F2)[0]["COST4"])
        assert lab == want, fr_lat


# ---- K-ARGV
ARGV = ["python", "train.py", "--out", "/root/strat1/runs/C0_s10", "--seed", "10", "--steps", "3000", "--w-tac", "1.0"]


def rep_of(argv):
    return json.loads(json.dumps(argv).replace("/root/strat1/runs/", "/root/strat1/runs_c0rep/"))


def test_k_argv_passes_when_only_the_run_root_differs():
    assert C.k_argv(rep_of(ARGV), ARGV) == (True, [])


def test_k_argv_fails_on_a_changed_lever_a_changed_value_and_a_length_change():
    for mut in (lambda a: [x if x != "3000" else "3001" for x in a], lambda a: a + ["--ctx-detach", "on"], lambda a: [x if x != "1.0" else "0.0" for x in a]):
        r = rep_of(mut(ARGV))
        ok, why = C.k_argv(r, ARGV)
        assert not ok and why
    ok, why = C.k_argv(rep_of(ARGV)[:-2], ARGV)
    assert not ok and any("length" in w for w in why)


def test_k_argv_fails_when_a_replicate_still_points_at_the_original_run_root_elsewhere():
    r = rep_of(ARGV)
    r[3] = "/root/strat1/runs_other/C0_s10"
    assert not C.k_argv(r, ARGV)[0]


def test_k_argv_dict_form():
    a = {"out": "/root/strat1/runs/C0_s10", "seed": 10}
    assert C.k_argv({"out": "/root/strat1/runs_c0rep/C0_s10", "seed": 10}, a)[0]
    ok, why = C.k_argv({"out": "/root/strat1/runs_c0rep/C0_s10", "seed": 11}, a)
    assert not ok and "seed" in why[0]


def test_k_argv_on_the_banked_arms():
    """the real banked argv files: the original arms pass against themselves and the replicates' rerooted argv"""
    for o in ("C0_s10", "C0_s11"):
        p = f"{C.L.CK}/{o}/argv.json"
        if os.path.exists(p):
            orig = json.load(open(p, encoding="utf-8"))
            assert C.k_argv(rep_of(orig), orig)[0]
            assert not C.k_argv(rep_of(orig), json.loads(json.dumps(orig).replace("3000", "3001")))[0]


# ---- pins
def test_pins_match_the_frozen_readers_and_the_addendum():
    assert C.pin_problems(C.R, addendum=os.path.join(HERE, "PREREG_ADDENDUM_C0REP.md")) == []


def test_pins_fail_on_a_changed_prefix(tmp_path):
    bad = dict(C.PINS, **{"strat1_read.py": "00000000"})
    assert any("strat1_read.py" in p for p in C.pin_problems(C.R, pins=bad))
    t = tmp_path / "a.md"
    t.write_text("changed", encoding="utf-8")
    assert any("addendum" in p for p in C.pin_problems(C.R, pins={}, addendum=str(t)))


# ---- c0rep_run: arm classification and commit widening
def test_replicate_arms_are_v9_nav_without_ctx_heads_like_c0_s10():
    L = CRUN.patch_lib()
    for t in CRUN.REP:
        assert t in L.ARMS and t not in L.TEXT_ARMS and t not in L.CTX_ARMS
    assert "C0_s10" in L.ARMS and "C0_s10" not in L.TEXT_ARMS and "S1d_s0" in L.CTX_ARMS


def test_commit_widening_only_for_the_two_replicate_tags(tmp_path):
    RDS = CRUN.patch_rds()
    ck = tmp_path / "ckpt_last.pt"
    ck.write_bytes(b"x")
    want = RDS.RD.sha(str(ck))
    saved = dict(RDS.ARMS)
    L = CRUN.L
    try:
        cases = (("STRAT1-C0_s10r", CRUN.REP_COMMIT, True), ("STRAT1-C0_s11r", CRUN.REP_COMMIT, True), ("STRAT1-C0_s10r", L.COMMIT, False),
                 ("STRAT1-C0_s0", CRUN.REP_COMMIT, False), ("STRAT1-C0_s0", L.COMMIT, True), ("STRAT1-C0_s10", CRUN.REP_COMMIT, True), ("STRAT1-S1d_s0", CRUN.SD.S1D_COMMIT, True))
        for tag, commit, ok in cases:
            man = tmp_path / f"m_{tag}_{commit[:6]}.json"
            man.write_text(json.dumps({"tag": tag[7:], "commit": commit, "files": {"ckpt_last.pt": {"sha256": want}}}), encoding="utf-8")
            RDS.ARMS[tag] = (str(ck), ("bank", str(man)))
            why = RDS.refusal(tag)
            assert (why is None) == ok, (tag, commit, why)
    finally:
        RDS.ARMS.clear()
        RDS.ARMS.update(saved)


def test_a_manifest_naming_the_original_tag_is_refused_for_the_replicate(tmp_path):
    """distinct-tag guard on the read side: STRAT1-C0_s10r with a manifest that says C0_s10 is refused"""
    RDS = CRUN.patch_rds()
    ck = tmp_path / "ckpt_last.pt"
    ck.write_bytes(b"x")
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"tag": "C0_s10", "commit": CRUN.REP_COMMIT, "files": {"ckpt_last.pt": {"sha256": RDS.RD.sha(str(ck))}}}), encoding="utf-8")
    saved = dict(RDS.ARMS)
    try:
        RDS.ARMS["STRAT1-C0_s10r"] = (str(ck), ("bank", str(man)))
        assert "names tag" in RDS.refusal("STRAT1-C0_s10r")
    finally:
        RDS.ARMS.clear()
        RDS.ARMS.update(saved)


# ---- the pod uploader against a fake hub: the real script text is executed
UPLOADER = os.path.join(HERE, "strat1_bank_pod_c0rep.py")


class FakeHub:
    def __init__(self, files):
        self.files = dict(files)                       # path -> (size, sha or None)
        self.uploads = []
        self.private = True
        self.sha = "fakerev"

    def api(self):
        hub = self

        class Api:
            def __init__(self, token=None):
                pass

            def whoami(self):
                return {"name": "tester"}

            def repo_info(self, repo, repo_type=None, files_metadata=False):
                sib = [types.SimpleNamespace(rfilename=k, size=v[0], lfs=(types.SimpleNamespace(sha256=v[1]) if v[1] else None)) for k, v in hub.files.items()]
                return types.SimpleNamespace(private=hub.private, siblings=sib, sha=hub.sha)

            def upload_file(self, path_or_fileobj=None, path_in_repo=None, **kw):
                data = open(path_or_fileobj, "rb").read()
                import hashlib
                hub.uploads.append(path_in_repo)
                hub.files[path_in_repo] = (len(data), hashlib.sha256(data).hexdigest() if path_in_repo.endswith("ckpt_last.pt") else None)
        return Api


def run_uploader(tmp_path, hub, mutate=lambda s: s):
    root = tmp_path / "runs"
    for src in ("C0_s10", "C0_s11"):
        for rel, body in (("train/ckpt_last.pt", src.encode() * 50), ("argv.json", b"[]"), ("train/argv.json", b"[]"), ("gate/selfcheck.json", b"{}"), ("gate/gate_token.json", b'{"commit": "a4357e79"}'),
                          ("train/summary.json", b"{}"), ("train/train_log.jsonl", b"{}"), ("ARM_DONE", b"done")):
            p = root / src / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(body)
    tmpd = tmp_path / "tmp"
    tmpd.mkdir()
    tokf = tmp_path / "tok"
    tokf.write_text("hf_fake", encoding="utf-8")
    src_text = open(UPLOADER, encoding="utf-8").read().replace('"/root/strat1/runs_c0rep"', repr(str(root))).replace("/root/.hf_token", str(tokf).replace("\\", "/")).replace("/root/strat1/tmp", str(tmpd).replace("\\", "/"))
    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = hub.api()
    saved = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = fake
    try:
        code = compile(mutate(src_text), "uploader", "exec")
        with pytest.raises(SystemExit) as e:
            exec(code, {"__name__": "__main__"})
        return e.value.code
    finally:
        if saved is None:
            del sys.modules["huggingface_hub"]
        else:
            sys.modules["huggingface_hub"] = saved


ORIG_FILES = {f"{t}/{f}": (10, "a" * 64 if f == "ckpt_last.pt" else None) for t in ("C0_s10", "C0_s11") for f in ("ckpt_last.pt", "argv.json", "BANK_MANIFEST.json")}


def test_uploader_banks_under_distinct_tags_and_leaves_the_originals_alone(tmp_path):
    hub = FakeHub(ORIG_FILES)
    assert run_uploader(tmp_path, hub) == 0
    assert hub.uploads and all(u.split("/")[0] in ("C0_s10r", "C0_s11r") for u in hub.uploads)
    assert all(hub.files[k] == v for k, v in ORIG_FILES.items())
    assert {"C0_s10r/ckpt_last.pt", "C0_s11r/BANK_MANIFEST.json"} <= set(hub.files)


def test_uploader_refuses_when_a_destination_prefix_is_not_empty(tmp_path):
    hub = FakeHub(dict(ORIG_FILES, **{"C0_s10r/ckpt_last.pt": (5, "b" * 64)}))
    with pytest.raises(AssertionError, match="not empty"):
        run_uploader(tmp_path, hub)
    assert hub.uploads == []


def test_uploader_refuses_when_the_originals_are_absent_wrong_repo(tmp_path):
    hub = FakeHub({})
    with pytest.raises(AssertionError, match="originals"):
        run_uploader(tmp_path, hub)
    assert hub.uploads == []


def test_mutation_uploader_with_the_destination_equal_to_the_source_is_refused(tmp_path):
    hub = FakeHub(ORIG_FILES)
    with pytest.raises(AssertionError):
        run_uploader(tmp_path, hub, mutate=lambda s: s.replace('"C0_s10": "C0_s10r"', '"C0_s10": "C0_s10"'))
    assert hub.uploads == []


def test_mutation_uploader_that_overwrote_an_original_is_detected_by_the_unchanged_check(tmp_path):
    """a destination that maps into an original's prefix cannot pass the empty-prefix guard; with the guard removed the before / after comparison exits non-zero"""
    hub = FakeHub(ORIG_FILES)
    mut = lambda s: s.replace('"C0_s10": "C0_s10r"', '"C0_s10": "C0_s10r"').replace('assert not any(k.startswith(dst + "/") for k in before), f"destination {dst}/ is not empty in the repo: refuse"', "pass")  # noqa: E731
    class Evil(FakeHub):
        pass
    hub2 = Evil(ORIG_FILES)
    api_cls = hub2.api()
    orig_upload = api_cls.upload_file

    def evil_upload(self, path_or_fileobj=None, path_in_repo=None, **kw):
        orig_upload(self, path_or_fileobj=path_or_fileobj, path_in_repo=path_in_repo, **kw)
        hub2.files["C0_s10/ckpt_last.pt"] = (10, "c" * 64)               # simulates an overwrite of an original
    api_cls.upload_file = evil_upload
    hub2.api = lambda: api_cls
    assert run_uploader(tmp_path, hub2, mutate=mut) == 1
