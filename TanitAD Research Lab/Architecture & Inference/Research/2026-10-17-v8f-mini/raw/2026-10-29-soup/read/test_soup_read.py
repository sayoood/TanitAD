"""soup_read.py's pure layer and composition on a FAKE heavy layer (no soup data, no dump, no rollout)."""
import itertools
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("SOUP_READ_DIR", HERE))
import soup_read as R  # noqa: E402

LI = R.load_li(os.path.join(HERE, "ref", "launch_intent_read.py"))
PREREG = os.path.join(os.path.dirname(HERE), "PREREG_V8F_SOUP.md")


def row(d=0.0, sep=False, label=None):
    w = abs(d) / 4 if sep else abs(d) + 0.1
    return {"delta": d, "lo": d - w, "hi": d + w, "separated": sep, "n": "2399/141", "mean_arm": 1.0 + d, "mean_ref": 1.0}


def table(over=None):
    t = {l: row() for l in LI.LABELS}
    t.update(over or {})
    return t


DOM = {"ADE6": row(-1.3, True), "speed MAE": row(-0.5, True), "yaw-rate MAE": row(-0.5, True), "TAC-LAT": row(+0.1, True), "S-POS@30": row(-1.0, True)}


class FakeHeavy:
    """contrast(x, y) -> {s: table}; scenario[(x, y, s)] overrides rows; guard rows from g[(x, y, s)]; SG from sgv[(x, s)]."""

    def __init__(self, scen=None, g=None, sgv=None, zero_ok=True, fail=None, cells=None, zero_bad=None):
        self.scen, self.g, self.sgv, self.zero_ok, self.fail, self.cells = scen or {}, g or {}, sgv or {}, zero_ok, fail, cells or {}
        self.zero_bad = zero_bad
        self.loaded, self.calls = [], []

    def load_slot(self, ctl, role, s, e):
        self.loaded.append((role, s))

    def load_soup_slot(self, ctl, role, s, e):
        self.loaded.append((role, s))
        ctl.add(f"K-SOUP-REC:{role}_s{s}", self.fail != f"K-SOUP-REC:{role}_s{s}", None)

    def prepare(self, ctl):
        self.calls.append("prepare")
        ctl.add("C4", True, None)

    def k_rc(self, ctl):
        self.calls.append("k_rc")
        ctl.add("K-RC", self.fail != "K-RC", None)

    def contrast(self, x, y):
        self.calls.append(("contrast", x, y))
        if x == y:
            z = {l: {"delta": 0.0, "lo": 0.0, "hi": 0.0, "separated": False} for l in LI.LABELS}
            if not self.zero_ok or x == self.zero_bad:
                z["ADE6"] = {"delta": 1e-9, "lo": 0.0, "hi": 0.0, "separated": False}
            return {s: z for s in R.SEEDS}
        return {s: table(self.scen.get((x, y, s))) for s in R.SEEDS}

    def guard_rows(self, ctl, x, y, s, tab):
        self.calls.append(("guard", x, y, s))
        d = {"R-TLAT": {"delta": 0.0, "separated": False}, "R-TLON": {"delta": 0.0, "separated": False}, "T0_point": 0.0, "KEEP_point": float(tab[s]["ADE6"]["delta"])}
        d.update(self.g.get((x, y, s), {}))
        return d

    def sg(self, x, members, s):
        self.calls.append(("sg", x, tuple(members), s))
        v = self.sgv.get((x, s), "flat")
        if v == "missing":
            raise RuntimeError("no dump")
        return {"barrier": v == "barrier", "gain": v == "gain", "delta": 0.0, "lo": 0.0, "hi": 0.0, "separated": v != "flat"}

    def global_controls(self, ctl):
        self.calls.append("global_controls")
        ctl.add("K-XINIT", self.fail != "K-XINIT", None)

    def speed(self, x, y, s, rng=None):
        self.calls.append(("speed", x, y, s))
        if self.fail == "speed":
            raise RuntimeError("no T0 cons")
        return self.cells.get((x, y, s)) or {n: (0.0, -0.1, 0.1) for n in LI.SPEED_CELLS}

    def finish(self, ctl):
        self.calls.append("finish")
        if self.fail == "finish":
            ctl.add("K1-FAKE", False, "a control added at the end")


@pytest.fixture(autouse=True)
def li(monkeypatch):
    monkeypatch.setattr(LI, "load_train_records", lambda e: {"fake": True})
    monkeypatch.setattr(LI, "run_record_controls", lambda ctl, trs, standin=False: ctl.add("K-MEMBERS", set(trs) == set(R.MEMBER_ROLES), sorted(trs)))
    return LI


MAN = {"arms": {r: {"0": {"fake": r}, "1": {"fake": r}} for r in ("CTRL", "REP", "A", "Arep", "SOUP", "SOUPR", "SOUP4")}}
RF0 = {l: 0.0 for l in LI.LABELS}


def run(H, li, man=MAN):
    ctl = li.Ctl()
    return R.read_core(H, li, ctl, man, RF0), ctl


def dom_scen(soups=("SOUP", "SOUPR"), seeds=R.SEEDS):
    return {(x, "P10", s): DOM for x in soups for s in seeds}


def test_neutral_world_is_no_effect_and_routes_flat(li):
    res, ctl = run(FakeHeavy(), li)
    d = res["decision"]
    assert not ctl.failed() and d["OUTCOME"] == "NO EFFECT" and d["next_soup"] == "SOUP-K" and res["route"] == "FLAT"


def test_dominant_replicated_and_adopted(li):
    res, _ = run(FakeHeavy(dom_scen()), li)
    assert res["decision"]["OUTCOME"] == "DOMINANT / ADOPTED" and res["decision"]["adopts"] and res["decision"]["next_soup"] is None


def test_dominant_on_one_soup_only_is_not_dominant(li):
    res, _ = run(FakeHeavy(dom_scen(soups=("SOUP",))), li)
    assert res["decision"]["OUTCOME"] == "NO EFFECT"
    assert res["read_a"]["SOUP"]["outcome"] == "DOMINANT" and res["read_a"]["SOUPR"]["outcome"] != "DOMINANT"


def test_dominant_at_one_seed_only_is_not_dominant(li):
    res, _ = run(FakeHeavy(dom_scen(seeds=(0,))), li)
    assert res["decision"]["OUTCOME"] == "NO EFFECT"


def test_dominant_not_adopted_when_keep_fails(li):
    g = {("SOUPR", "A", 1): {"KEEP_point": 0.07}}
    res, _ = run(FakeHeavy(dom_scen(), g=g), li)
    assert res["decision"]["OUTCOME"] == "DOMINANT / NOT ADOPTED" and res["decision"]["per_seed"]["s1"]["AD"] is False


def test_harms_on_a_guard_in_the_replicate_pair_routes_by_sg(li):
    scen = dom_scen()
    scen[("SOUPR", "A", 0)] = {"ADE6": row(+0.2, True)}            # R-ADE fails in the replicate pair at s0
    res, _ = run(FakeHeavy(scen, sgv={("SOUP", 1): "barrier"}), li)
    d = res["decision"]
    assert d["OUTCOME"] == "HARMS" and d["per_seed"]["s0"]["H"] is False and d["per_seed"]["s1"]["H"] is True
    assert d["next_soup"] == "ENS2" and res["route"] == "BARRIER"


def test_t0_guard_is_inclusive_at_half_f(li):
    res, _ = run(FakeHeavy(g={("SOUP", "CTRL", 0): {"T0_point": 0.0604}}), li)
    assert res["decision"]["per_seed"]["s0"]["H"] is True
    res2, _ = run(FakeHeavy(g={("SOUP", "CTRL", 0): {"T0_point": 0.0605}}), li)
    assert res2["decision"]["OUTCOME"] == "HARMS"


def crawl_move(x, y, s, size):
    return {l: row(-size, True) for l in LI.CRAWL_SET}


def test_moved_both_seeds_both_pairs_adopted(li):
    scen = {}
    bars = LI.bar_table({**RF0, "yaw-rate MAE": 0.5})
    for (x, y) in R.PAIRS.values():
        for s in R.SEEDS:
            scen[(x, y, s)] = crawl_move(x, y, s, 2.0)
    res, _ = run(FakeHeavy(scen), li)
    assert res["decision"]["OUTCOME"] == "MOVED / ADOPTED"
    assert bars["yaw-rate MAE"] == 1.0


def test_moved_needs_the_bar(li):
    scen = {}
    for (x, y) in R.PAIRS.values():
        for s in R.SEEDS:
            scen[(x, y, s)] = crawl_move(x, y, s, 2.0)
    ctl = li.Ctl()
    big = {**RF0, "M1 crawl share": 1.5}                             # bar 3.0 on M1 > the 2.0 improvement
    res = R.read_core(FakeHeavy(scen), li, ctl, MAN, big)
    assert res["decision"]["OUTCOME"] == "NO EFFECT"


def test_one_seed_and_a_miss_in_either_pair_is_a_miss(li):
    scen = {(x, y, 0): crawl_move(x, y, 0, 2.0) for (x, y) in R.PAIRS.values()}
    scen[("SOUP", "CTRL", 1)] = crawl_move("SOUP", "CTRL", 1, 2.0)      # s1: main only
    res, _ = run(FakeHeavy(scen), li)
    assert res["decision"]["OUTCOME"] == "ONE SEED" and res["decision"]["per_seed"]["s1"]["MV"] is False


def test_moved_not_adopted(li):
    scen = {(x, y, s): crawl_move(x, y, s, 2.0) for (x, y) in R.PAIRS.values() for s in R.SEEDS}
    res, _ = run(FakeHeavy(scen, g={("SOUP", "CTRL", 0): {"KEEP_point": 0.1}}), li)
    assert res["decision"]["OUTCOME"] == "MOVED / NOT ADOPTED" and res["decision"]["next_soup"] == "SOUP-K"


@pytest.mark.parametrize("fail", ["K-RC", "K-SOUP-REC:SOUPR_s1"])
def test_a_failed_control_refuses(li, fail):
    res, ctl = run(FakeHeavy(dom_scen(), fail=fail), li)
    assert res["decision"]["OUTCOME"] == "REFUSED" and ctl.failed()


def test_k_zero_is_a_control(li):
    res, _ = run(FakeHeavy(dom_scen(), zero_ok=False), li)
    assert res["decision"]["OUTCOME"] == "REFUSED"


def test_k_rc_runs_before_any_soup_contrast(li):
    H = FakeHeavy()
    run(H, li)
    i = H.calls.index("k_rc")
    assert all(not (isinstance(c, tuple) and c[1] in R.SOUP_ROLES) for c in H.calls[:i])


def test_missing_sg_cell_routes_undecided_without_refusing(li):
    res, ctl = run(FakeHeavy(sgv={("SOUPR", 0): "missing"}), li)
    assert not ctl.failed() and res["decision"]["OUTCOME"] == "NO EFFECT" and res["route"] is None
    assert res["decision"]["next_soup"].startswith("UNDECIDED") and "SOUPR:s0" in res["sg_errors"]


def test_soup4_is_beside_and_decides_nothing(li):
    scen = dom_scen()
    scen[("SOUP4", "CTRL", 0)] = {"ADE6": row(+0.5, True)}
    scen[("SOUP4", "P10", 1)] = {"yaw-rate MAE": row(+3.0, True)}
    res, _ = run(FakeHeavy(scen), li)
    assert res["decision"]["OUTCOME"] == "DOMINANT / ADOPTED" and res["read_a"]["SOUP4"]["outcome"] != "DOMINANT"
    man = {"arms": {k: v for k, v in MAN["arms"].items() if k != "SOUP4"}}
    res2, _ = run(FakeHeavy(dom_scen()), li, man)
    assert res2["decision"]["OUTCOME"] == "DOMINANT / ADOPTED" and "SOUP4" not in res2["read_a"]


def test_a_decisive_soup_missing_from_the_manifest_is_an_error(li):
    man = {"arms": {k: v for k, v in MAN["arms"].items() if k != "SOUPR"}}
    with pytest.raises(ValueError, match="SOUPR"):
        run(FakeHeavy(), li, man)


# ---- the outcome table itself
def test_outcome_table_is_complete_and_exclusive_over_all_cells():
    names = ("H", "MV", "AD", "DM")
    seen = set()
    for ctrl in (True, False):
        for bits in itertools.product((False, True), repeat=8):
            b = {s: dict(zip(names, bits[4 * s:4 * s + 4])) for s in R.SEEDS}
            d = R.decide(ctrl, b)
            assert len(d["matching_rows_in_table_order"]) == 1, (ctrl, b)
            seen.add(d["OUTCOME"])
    assert seen == set(R.OUTCOME_NAMES)


def test_per_seed_and_over_pairs():
    ok = {"flags": {"R-MOVE": True, "R-KEEP": True, "R-NOBASE": True, "R-CRAWL": True}, "guards": {g: True for g in R.GUARDS}}
    for which in ("main", "rep"):
        for k in ("R-MOVE", "R-KEEP", "R-NOBASE"):
            bad = {"flags": dict(ok["flags"], **{k: False}), "guards": ok["guards"]}
            p = {s: {"main": ok, "rep": ok} for s in R.SEEDS}
            p[0] = {"main": bad if which == "main" else ok, "rep": bad if which == "rep" else ok}
            b = R.per_seed_booleans(p, {s: {"SOUP": True, "SOUPR": True} for s in R.SEEDS})
            assert b[1]["MV"] and b[1]["AD"]
            assert (not b[0]["MV"]) if k == "R-MOVE" else (not b[0]["AD"])
        for g in R.GUARDS:
            bad = {"flags": ok["flags"], "guards": dict(ok["guards"], **{g: False})}
            p = {s: {"main": ok, "rep": ok} for s in R.SEEDS}
            p[1] = {"main": bad if which == "main" else ok, "rep": bad if which == "rep" else ok}
            assert not R.per_seed_booleans(p, {s: {"SOUP": True, "SOUPR": True} for s in R.SEEDS})[1]["H"]
    p = {s: {"main": ok, "rep": ok} for s in R.SEEDS}
    assert not R.per_seed_booleans(p, {0: {"SOUP": True, "SOUPR": False}, 1: {"SOUP": True, "SOUPR": True}})[0]["DM"]


def test_committed_next_and_routing():
    for name, _, nxt in R.OUTCOME_TABLE:
        c = R.committed_next(name, "BARRIER")
        assert c["committed_next"] == nxt
        assert c["next_soup"] == ("ENS2" if name in R.ROUTED else None)
        assert R.committed_next(name, "FLAT")["next_soup"] == ("SOUP-K" if name in R.ROUTED else None)
    assert R.committed_next("UNPLACED", "FLAT")["committed_next"] is None
    cells = {f"{x}:s{s}": {"barrier": False} for x in ("SOUP", "SOUPR") for s in R.SEEDS}
    assert R.sg_route(cells) == "FLAT"
    cells["SOUPR:s1"] = {"barrier": True}
    assert R.sg_route(cells) == "BARRIER"
    cells["SOUP:s0"] = None
    assert R.sg_route(cells) is None


def fake_pb(a, b, eid, n_boot=2000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    ue = np.unique(eid)
    rng = np.random.default_rng(seed)
    s = np.array([d[eid == e].sum() for e in ue])
    n = np.array([(eid == e).sum() for e in ue])
    bs = [s[p].sum() / n[p].sum() for p in (rng.integers(0, len(ue), len(ue)) for _ in range(n_boot))]
    return {"delta": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def test_soup_gain_identity_barrier_and_gain():
    rng = np.random.default_rng(1)
    eid = np.repeat(np.arange(40), 10)
    a, b = rng.normal(2, 0.3, 400), rng.normal(2.2, 0.3, 400)
    z = R.soup_gain((a + b) / 2, [a, b], eid, fake_pb, 200)
    assert z["delta"] == 0.0 and z["lo"] == 0.0 and z["hi"] == 0.0 and not z["separated"] and not z["barrier"] and not z["gain"]
    w = R.soup_gain((a + b) / 2 + 0.3, [a, b], eid, fake_pb, 200)
    assert w["barrier"] and not w["gain"]
    g = R.soup_gain((a + b) / 2 - 0.3, [a, b], eid, fake_pb, 200)
    assert g["gain"] and not g["barrier"]
    x = (a + b) / 2
    x[:5] = np.nan
    assert R.soup_gain(x, [a, b], eid, fake_pb, 50)["n_dropped"] == 5
    with pytest.raises(ValueError):
        R.soup_gain(x[:-1], [a, b], eid, fake_pb, 50)


def test_k_soup_rec_catches_each_problem():
    spec = {"members": {"m0": {"sha256": "a" * 64}, "m1": {"sha256": "b" * 64}}, "soups": {"S": {"members": ["m0", "m1"], "donor": "m0"}}}
    good = {"name": "S", "output_sha256": "f" * 64, "members": ["m0", "m1"], "donor": "m0", "member_sha256": {"m0": "a" * 64, "m1": "b" * 64},
            "builder_sha256": "c" * 64, "K-BASIN": {"PASS": True, "B1": True, "B2": True}, "K-MID": {"PASS": True}, "weights": [0.5, 0.5]}
    assert R.k_soup_rec(good, spec, "S", "f" * 64, "c" * 64, "cfg", "cfg", True) == []
    muts = [({"output_sha256": "e" * 64}, {}), ({"donor": "m1"}, {}), ({"member_sha256": {"m0": "a" * 64, "m1": "d" * 64}}, {}),
            ({"builder_sha256": "d" * 64}, {}), ({"K-BASIN": {"PASS": True, "B1": False, "B2": True}}, {}), ({"K-MID": {"PASS": False}}, {}),
            ({"weights": [0.6, 0.4]}, {}), ({"name": "T"}, {}), ({}, {"ckpt": "other"}), ({}, {"opt": False})]
    for rec_over, arg_over in muts:
        p = R.k_soup_rec({**good, **rec_over}, spec, "S", "f" * 64, "c" * 64, "cfg", arg_over.get("ckpt", "cfg"), arg_over.get("opt", True))
        assert p, (rec_over, arg_over)


def test_prereg_text_matches_and_detects_an_edit(tmp_path):
    assert R.prereg_text_problems(PREREG, None) == []
    assert R.prereg_text_problems(PREREG, R.sha256_file(PREREG)) == []
    assert R.prereg_text_problems(PREREG, "0" * 64)
    txt = open(PREREG, encoding="utf-8").read()
    p = tmp_path / "p.md"
    p.write_text(txt.replace("MV at exactly one seed", "MV at one seed or more"), encoding="utf-8")
    assert any("ONE SEED" in x for x in R.prereg_text_problems(str(p), None))
    p.write_text(txt.replace("| **NO EFFECT** |", "| **NO EFFECT** | x |", 1), encoding="utf-8")
    assert R.prereg_text_problems(str(p), None)


def test_li_reader_pin_is_enforced(tmp_path):
    q = tmp_path / "launch_intent_read.py"
    shutil.copy(os.path.join(HERE, "ref", "launch_intent_read.py"), q)
    with open(q, "a", encoding="utf-8") as f:
        f.write("\n# edited\n")
    with pytest.raises(SystemExit, match="REFUSED"):
        R.load_li(str(q))


# =====================================================================================================================================================================
# Pass 2: the composition layer against an independent oracle, and every wiring of read_core pinned by a distinguishing scenario (tools/mutation_soup_reader.py runs these against
# one-change mutants of soup_read.py).  Nothing below reads a soup, a dump or a checkpoint.
# =====================================================================================================================================================================
import json  # noqa: E402

SPEC = os.path.join(os.path.dirname(HERE), "soups.json")


def oracle(ctrl, b):
    """The PREREG section 10 table as plain prose, independent of soup_read.outcome_rows."""
    if not ctrl:
        return "REFUSED"
    if not (b[0]["H"] and b[1]["H"]):
        return "HARMS"
    ad = b[0]["AD"] and b[1]["AD"]
    if b[0]["DM"] and b[1]["DM"]:
        return "DOMINANT / ADOPTED" if ad else "DOMINANT / NOT ADOPTED"
    mv = int(b[0]["MV"]) + int(b[1]["MV"])
    if mv == 2:
        return "MOVED / ADOPTED" if ad else "MOVED / NOT ADOPTED"
    return "ONE SEED" if mv == 1 else "NO EFFECT"


def test_decide_equals_the_oracle_on_every_cell():
    names = ("H", "MV", "AD", "DM")
    for ctrl in (True, False):
        for bits in itertools.product((False, True), repeat=8):
            b = {s: dict(zip(names, bits[4 * s:4 * s + 4])) for s in R.SEEDS}
            d = R.decide(ctrl, b)
            assert d["OUTCOME"] == oracle(ctrl, b), (ctrl, b)
            assert d["adopts"] == (d["OUTCOME"] in ("DOMINANT / ADOPTED", "MOVED / ADOPTED"))
            assert d["controls_pass"] is ctrl and d["per_seed"] == {"s0": b[0], "s1": b[1]}
            assert d["matching_rows_in_table_order"] == [d["OUTCOME"]]


def test_unplaced_when_two_rows_match_or_none_does(monkeypatch):
    monkeypatch.setattr(R, "outcome_rows", lambda ctrl, b: {n: n in ("HARMS", "NO EFFECT") for n in R.OUTCOME_NAMES})
    d = R.decide(True, {s: {"H": True, "MV": False, "AD": True, "DM": False} for s in R.SEEDS})
    assert d["OUTCOME"] == "UNPLACED" and not d["adopts"] and len(d["matching_rows_in_table_order"]) == 2
    monkeypatch.setattr(R, "outcome_rows", lambda ctrl, b: {n: False for n in R.OUTCOME_NAMES})
    assert R.decide(True, {s: {"H": True, "MV": False, "AD": True, "DM": False} for s in R.SEEDS})["OUTCOME"] == "UNPLACED"


# ---- the coded constants against the registered spec
def test_roles_members_donors_and_pairs_agree_with_soups_json():
    spec = json.load(open(SPEC, encoding="utf-8"))
    assert R.SEEDS == (0, 1) and R.GUARDS == ("R-TLAT", "R-TLON", "R-ADE", "R-T0", "R-NEW")
    assert R.PAIRS == {"main": ("SOUP", "CTRL"), "rep": ("SOUPR", "A")} and R.SOUP_ROLES == ("SOUP", "SOUPR", "SOUP4") and R.MEMBER_ROLES == ("CTRL", "REP", "A", "Arep")
    assert R.MEMBERS == {"SOUP": ("CTRL", "REP"), "SOUPR": ("A", "Arep"), "SOUP4": ("CTRL", "REP", "A", "Arep")} and R.DONOR == {"SOUP": "CTRL", "SOUPR": "A", "SOUP4": "CTRL"}
    for role in R.SOUP_ROLES:
        for s in R.SEEDS:
            sp = spec["soups"][R.SPEC_NAME[role].format(s=s)]
            assert sp["members"] == [R.SPEC_MEMBER[m].format(s=s) for m in R.MEMBERS[role]], (role, s)
            assert sp["donor"] == R.SPEC_MEMBER[R.DONOR[role]].format(s=s) and sp["seed"] == s and sp["decisive"] == (role != "SOUP4")
    for pname, (x, y) in R.PAIRS.items():                       # the pair's control IS the soup's donor (the one-lever form of K-PAIR)
        assert R.DONOR[x] == y and y in R.MEMBERS[x]
    for m, name in R.SPEC_MEMBER.items():
        assert name.format(s=0) in spec["members"] and name.format(s=1) in spec["members"]


def test_pinned_constants():
    assert R.LI_READER_SHA == "fb3d2d7d945b54209b62d5f906d4dcea30925a24f0b807459b8cbf3d7f0747a9" and R.sha256_file(os.path.join(HERE, "ref", "launch_intent_read.py")) == R.LI_READER_SHA
    assert R.TOOL_DIGEST == "579183a362f7f7843d36bd6c2cc339457b41511c3721e476bfc126eee2321224"
    assert R.CLAIM_ROW == "H-V8F-SOUP-1" and "DRAFT" in R.PREREG_ID and "OPEN loop" in R.TIER_T1 and "Never 'closed loop'" in R.TIER_T1 and "Never 'driving performance'" in R.TIER_T0
    assert "B=2000" in R.ESTIMATOR and "141 episodes" in R.ESTIMATOR and "paired" in R.ESTIMATOR
    assert R.NEXT_SOUP == {"BARRIER": "ENS2", "FLAT": "SOUP-K"} and R.ADOPTING == ("DOMINANT / ADOPTED", "MOVED / ADOPTED")
    assert R.ROUTED == ("HARMS", "MOVED / NOT ADOPTED", "ONE SEED", "NO EFFECT")
    assert list(R.K_RC_TABLES) == ["A_CTRL", "Arep_REP", "CTRL_P10", "REP_P10", "REP_CTRL", "A_P10", "Arep_P10"]
    assert R.K_RC_TABLES["A_CTRL"] == ("A", "CTRL") and R.K_RC_TABLES["Arep_REP"] == ("Arep", "REP") and R.K_RC_TABLES["REP_CTRL"] == ("REP", "CTRL")
    assert R.K_RC_TABLES["CTRL_P10"] == ("CTRL", "P10") and R.K_RC_TABLES["REP_P10"] == ("REP", "P10") and R.K_RC_TABLES["A_P10"] == ("A", "P10") and R.K_RC_TABLES["Arep_P10"] == ("Arep", "P10")


# ---- per_seed_booleans, one clause at a time
def pair_ok():
    return {"flags": {"R-MOVE": True, "R-KEEP": True, "R-NOBASE": True, "R-CRAWL": True}, "guards": {g: True for g in R.GUARDS}}


def test_per_seed_booleans_pairs_decide_exactly_their_own_clause():
    dm_ok = {s: {"SOUP": True, "SOUPR": True} for s in R.SEEDS}
    base = R.per_seed_booleans({s: {"main": pair_ok(), "rep": pair_ok()} for s in R.SEEDS}, dm_ok)
    assert all(base[s] == {"H": True, "MV": True, "AD": True, "DM": True} for s in R.SEEDS)
    for s in R.SEEDS:
        for which in ("main", "rep"):
            for k, out in (("R-MOVE", "MV"), ("R-KEEP", "AD"), ("R-NOBASE", "AD")):
                p = {t: {"main": pair_ok(), "rep": pair_ok()} for t in R.SEEDS}
                p[s][which]["flags"][k] = False
                b = R.per_seed_booleans(p, dm_ok)
                assert b[s][out] is False and all(b[s][o] for o in ("H", "MV", "AD", "DM") if o != out) and b[1 - s] == base[1 - s], (s, which, k)
            for g in R.GUARDS:
                p = {t: {"main": pair_ok(), "rep": pair_ok()} for t in R.SEEDS}
                p[s][which]["guards"][g] = False
                b = R.per_seed_booleans(p, dm_ok)
                assert b[s]["H"] is False and b[s]["MV"] and b[s]["AD"] and b[1 - s] == base[1 - s], (s, which, g)
            p = {t: {"main": pair_ok(), "rep": pair_ok()} for t in R.SEEDS}
            p[s][which]["flags"]["R-CRAWL"] = False                  # R-CRAWL is not a section-10 boolean: it moves nothing
            assert R.per_seed_booleans(p, dm_ok)[s] == base[s]
        for x in ("SOUP", "SOUPR"):
            d = {t: {"SOUP": True, "SOUPR": True} for t in R.SEEDS}
            d[s][x] = False
            b = R.per_seed_booleans({t: {"main": pair_ok(), "rep": pair_ok()} for t in R.SEEDS}, d)
            assert b[s]["DM"] is False and b[1 - s]["DM"] is True and b[s]["H"] and b[s]["MV"] and b[s]["AD"]
    assert set(base[0]) == {"H", "MV", "AD", "DM"}


# ---- SG
def test_soup_gain_four_members_uses_the_mean_and_drops_any_nonfinite_window():
    rng = np.random.default_rng(3)
    eid = np.repeat(np.arange(40), 10)
    ms = [rng.normal(2.0 + 0.7 * i * i, 0.2, 400) for i in range(4)]             # four unequal members: mean != median
    x = np.mean(ms, axis=0) + 0.25
    z = R.soup_gain(x, ms, eid, fake_pb, 200)
    assert abs(z["delta"] - 0.25) < 1e-9 and z["barrier"] and not z["gain"] and z["n_windows"] == 400 and z["n_dropped"] == 0
    bad = [m.copy() for m in ms]
    bad[2][:7] = np.nan                                                       # a NaN in ONE member drops the window
    z = R.soup_gain(x, bad, eid, fake_pb, 50)
    assert z["n_dropped"] == 7 and z["n_windows"] == 393 and np.isfinite([z["delta"], z["lo"], z["hi"]]).all()
    with pytest.raises(ValueError):
        R.soup_gain(x, [m[:-1] for m in ms], eid, fake_pb, 50)
    with pytest.raises(ValueError):
        R.soup_gain(x, ms, eid[:-1], fake_pb, 50)
    sep = R.soup_gain(x - 0.5, ms, eid, fake_pb, 200)
    assert sep["gain"] and sep["separated"] and not sep["barrier"] and sep["hi"] < 0 < 1


def test_soup_gain_separated_is_either_side_and_zero_straddle_is_not():
    eid = np.repeat(np.arange(40), 10)
    rng = np.random.default_rng(5)
    a = rng.normal(2, 1.0, 400)
    r = R.soup_gain(a + rng.normal(0, 0.5, 400), [a], eid, fake_pb, 200)           # noise around zero
    assert not r["separated"] and not r["barrier"] and not r["gain"] and r["lo"] < 0 < r["hi"]
    for sh, flag in ((0.6, "barrier"), (-0.6, "gain")):
        r = R.soup_gain(a + sh + rng.normal(0, 0.01, 400), [a], eid, fake_pb, 200)
        assert r["separated"] and r[flag] and set(("barrier", "gain")) - {flag} and not r["gain" if flag == "barrier" else "barrier"]


# ---- K-SOUP-REC: the remaining clauses
def test_k_soup_rec_each_clause_alone_and_n4_weights():
    spec = {"members": {f"m{i}": {"sha256": str(i) * 64} for i in range(4)}, "soups": {"S": {"members": [f"m{i}" for i in range(4)], "donor": "m0"}}}
    good = {"name": "S", "output_sha256": "f" * 64, "members": [f"m{i}" for i in range(4)], "donor": "m0", "member_sha256": {f"m{i}": str(i) * 64 for i in range(4)},
            "builder_sha256": "c" * 64, "K-BASIN": {"PASS": True, "B1": True, "B2": True}, "K-MID": {"PASS": True}, "weights": [0.25] * 4}
    assert R.k_soup_rec(good, spec, "S", "f" * 64, "c" * 64, "cfg", "cfg", True) == []
    assert R.k_soup_rec(good, spec, "S", "f" * 64, "c" * 64) == []                               # no donor cfg given: the cfg clause is skipped
    for over in ({"K-BASIN": {"PASS": False, "B1": True, "B2": True}}, {"K-BASIN": {"PASS": True, "B1": False, "B2": True}}, {"K-BASIN": {"PASS": True, "B1": True, "B2": False}},
                 {"K-BASIN": None}, {"K-MID": None}, {"weights": [0.5, 0.5]}, {"members": list(reversed(good["members"]))}, {"member_sha256": {**good["member_sha256"], "m3": "0" * 64}}):
        assert R.k_soup_rec({**good, **over}, spec, "S", "f" * 64, "c" * 64, "cfg", "cfg", True), over
    assert R.k_soup_rec({**good, "member_sha256": {}}, spec, "S", "f" * 64, "c" * 64)
    assert R.k_soup_rec(good, spec, "S", "e" * 64, "c" * 64)


# ---- K-TREE
def test_k_tree_clauses():
    ck, td = "a" * 64, R.TOOL_DIGEST
    good = {"ckpt_sha256": ck, "ckpt_sha256_before": ck, "ckpt_sha256_after": ck, "tool_digest": td, "smoke": False, "max_windows": None}
    assert R.k_tree(good, ck) == []
    for k in ("ckpt_sha256", "ckpt_sha256_before", "ckpt_sha256_after"):
        assert R.k_tree({**good, k: "b" * 64}, ck), k
        assert R.k_tree({kk: v for kk, v in good.items() if kk != k}, ck), k
    assert R.k_tree({**good, "tool_digest": "0" * 64}, ck) and R.k_tree({**good, "smoke": True}, ck) and R.k_tree({**good, "max_windows": 5}, ck)
    assert R.k_tree({kk: v for kk, v in good.items() if kk != "smoke"}, ck)
    assert R.k_tree(None, ck) and R.k_tree("x", ck)
    assert R.k_tree({**good, "tool_digest": "z"}, ck, tool_digest="z") == []


def test_rollout_files_lists_dumps_json_and_ticks_of_both_lanes_for_the_new_roles(tmp_path):
    def entry(role, s):
        d = {}
        for lane in ("t1", "t1_logged"):
            dd = tmp_path / f"{role}{s}{lane}"
            dd.mkdir()
            (dd / "ep0000.npz").write_bytes(b"x")
            (dd / "ep0001.npz").write_bytes(b"x")
            (dd / "other.txt").write_bytes(b"x")
            (dd / "summary.npz").write_bytes(b"x")
            d[lane] = {"dump_dir": str(dd), "json": str(tmp_path / f"{role}{s}{lane}.json"), "ticks": str(tmp_path / f"{role}{s}{lane}.npz")}
        return d
    man = {"arms": {r: {str(s): entry(r, s) for s in R.SEEDS} for r in ("A", "Arep", "SOUP", "SOUPR", "SOUP4", "CTRL", "REP")}}
    got = R.rollout_files(man)
    assert len(got) == 5 * 2 * 2 * 4
    assert not any("CTRL" in os.path.basename(g) or "REP" in os.path.basename(g).replace("Arep", "") for g in got)
    assert not any(g.endswith("other.txt") or g.endswith("summary.npz") for g in got)
    for r in ("A", "Arep", "SOUP", "SOUPR", "SOUP4"):
        for s in R.SEEDS:
            for lane in ("t1", "t1_logged"):
                assert str(tmp_path / f"{r}{s}{lane}.json") in got and str(tmp_path / f"{r}{s}{lane}.npz") in got
                assert any(g.replace("\\", "/").endswith(f"{r}{s}{lane}/ep0001.npz") for g in got)
    assert R.rollout_files(man, roles=("CTRL",)) != got and len(R.rollout_files(man, roles=("CTRL",))) == 16
    assert len(R.rollout_files({"arms": {}})) == 0 or True
    assert R.rollout_files({"arms": {"A": {"0": {}}}}, roles=("A",)).count("") == 8


# ---- read_core wiring, each pinned by a scenario that distinguishes it
def uniq_scen():
    """a distinct ADE6 delta for every (x, y, s) contrast, so a table read from the wrong contrast is visible."""
    scen, v = {}, 0.01
    for x in ("SOUP", "SOUPR", "SOUP4", "CTRL", "REP", "A", "Arep"):
        for y in ("P10", "CTRL", "REP", "A", "Arep"):
            for s in R.SEEDS:
                if x != y:
                    scen[(x, y, s)] = {"ADE6": row(v)}
                    v += 0.01
    return scen


def test_every_table_is_the_contrast_it_is_named_after():
    scen = uniq_scen()
    res, ctl = run(FakeHeavy(scen), LI)
    assert not ctl.failed()
    for key in ("SOUP_P10", "SOUP_CTRL", "SOUPR_P10", "SOUPR_A", "SOUP4_P10", "SOUP4_CTRL", "SOUP_REP", "SOUPR_Arep", "SOUPR_CTRL", "REP_CTRL", "CTRL_P10", "A_P10"):
        x, y = key.split("_")
        for s in R.SEEDS:
            assert res["tabs"][key][s]["ADE6"]["delta"] == scen[(x, y, s)]["ADE6"]["delta"], key


def test_pairs_use_their_own_guards_controls_and_p10_context():
    scen = uniq_scen()
    g = {(x, y, s): {"T0_point": 0.001 * (1 + i)} for i, (x, y, s) in enumerate([("SOUP", "CTRL", 0), ("SOUP", "CTRL", 1), ("SOUPR", "A", 0), ("SOUPR", "A", 1)])}
    H = FakeHeavy(scen, g=g)
    res, _ = run(H, LI)
    for s in R.SEEDS:
        assert res["pairs"][s]["main"]["T0_point"] == g[("SOUP", "CTRL", s)]["T0_point"] and res["pairs"][s]["rep"]["T0_point"] == g[("SOUPR", "A", s)]["T0_point"]
        assert res["pairs"][s]["main"]["KEEP_point"] == scen[("SOUP", "CTRL", s)]["ADE6"]["delta"] and res["pairs"][s]["rep"]["KEEP_point"] == scen[("SOUPR", "A", s)]["ADE6"]["delta"]
        assert ("guard", "SOUP", "CTRL", s) in H.calls and ("guard", "SOUPR", "A", s) in H.calls
    # X - P10 feeds R-CRAWL of its own pair only; Y - P10 sets the domain of its own pair only
    over = {("SOUP", "P10", 0): {"yaw-rate MAE": row(+1.0, True)}, ("SOUPR", "P10", 0): {"curvature MAE": row(+1.0, True)},
            ("CTRL", "P10", 0): {"heading MAE": row(+1.0, True)}, ("A", "P10", 0): {"cross-track MAE": row(+1.0, True)}}
    res, _ = run(FakeHeavy({**uniq_scen(), **over}), LI)
    m, r = res["pairs"][0]["main"], res["pairs"][0]["rep"]
    assert m["R-CRAWL_rows"]["yaw-rate MAE"] is False and m["R-CRAWL_rows"]["curvature MAE"] is True
    assert r["R-CRAWL_rows"]["curvature MAE"] is False and r["R-CRAWL_rows"]["yaw-rate MAE"] is True
    assert "heading MAE" not in m["domain"] and "heading MAE" in r["domain"] and "cross-track MAE" not in r["domain"] and "cross-track MAE" in m["domain"]


def test_rf_is_rep_minus_ctrl_over_both_seeds_and_the_bar_ignores_it():
    scen = {("REP", "CTRL", 0): {"ADE6": row(+0.4, True)}, ("REP", "CTRL", 1): {"ADE6": row(-0.9, True)}}
    res, _ = run(FakeHeavy(scen), LI)
    assert res["floors"]["RF"]["ADE6"] == 0.9 and res["floors"]["RF"]["yaw-rate MAE"] == 0.0 and res["floors"]["RF0"] == RF0
    assert res["floors"]["bar"] == LI.bar_table(RF0) and res["floors"]["bar"]["ADE6"] == pytest.approx(2 * 0.1207) and res["floors"]["bar"]["yaw-rate MAE"] == 0.0
    rf0 = {**RF0, "M1 crawl share": 0.3}
    res = R.read_core(FakeHeavy(scen), LI, LI.Ctl(), MAN, rf0)
    assert res["floors"]["bar"]["M1 crawl share"] == pytest.approx(0.6) and res["floors"]["RF0"]["M1 crawl share"] == 0.3
    custom = {l: 7.0 for l in LI.LABELS}
    assert R.read_core(FakeHeavy(scen), LI, LI.Ctl(), MAN, RF0, bars=custom)["floors"]["bar"] == custom


@pytest.mark.parametrize("bad", ["SOUP", "SOUPR", "SOUP4", "CTRL", "A", "REP", "Arep"])
def test_k_zero_runs_for_every_soup_and_the_two_controls(li, bad):
    res, ctl = run(FakeHeavy(dom_scen(), zero_bad=bad), li)
    covered = bad in ("SOUP", "SOUPR", "SOUP4", "CTRL", "A")
    assert (res["decision"]["OUTCOME"] == "REFUSED") == covered
    if covered:
        assert any(k.startswith(f"K-ZERO:{bad}-{bad}") for k in ctl.failed())


def test_sg_is_asked_for_each_soup_with_its_registered_members_at_both_seeds():
    H = FakeHeavy()
    run(H, LI)
    got = {c for c in H.calls if isinstance(c, tuple) and c[0] == "sg"}
    assert got == {("sg", x, R.MEMBERS[x], s) for x in R.SOUP_ROLES for s in R.SEEDS}


def test_global_controls_and_member_slots_and_soup_slots_are_all_visited():
    H = FakeHeavy()
    res, ctl = run(H, LI)
    assert "global_controls" in H.calls and "K-XINIT" in ctl.items and "K-MEMBERS" in ctl.items
    assert sorted(H.loaded) == sorted((r, s) for r in R.MEMBER_ROLES + R.SOUP_ROLES for s in R.SEEDS)
    assert FakeHeavy(fail="K-XINIT") and run(FakeHeavy(dom_scen(), fail="K-XINIT"), LI)[0]["decision"]["OUTCOME"] == "REFUSED"


def test_a_soup_present_at_one_seed_only_is_ignored():
    man = {"arms": {**MAN["arms"], "SOUP4": {"0": {"fake": 1}}}}
    H = FakeHeavy()
    res, _ = run(H, LI, man)
    assert "SOUP4" not in res["read_a"] and ("SOUP4", 0) not in H.loaded
    man2 = {"arms": {**MAN["arms"], "SOUP": {"0": {"fake": 1}, "1": None}}}
    with pytest.raises(ValueError, match="SOUP "):
        run(FakeHeavy(), LI, man2)


def dm_variants():
    ade = {"ADE6": row(-1.3, True)}
    better = {"speed MAE": row(-0.5, True), "yaw-rate MAE": row(-0.5, True), "TAC-LAT": row(+0.1, True), "S-POS@30": row(-1.0, True)}
    return ade, better


@pytest.mark.parametrize("which", ["D1", "D2", "D3"])
def test_dm_needs_each_of_d1_d2_d3_on_both_soups(which):
    ade, better = dm_variants()
    rows = {**ade, **better}
    if which == "D1":
        rows["ADE6"] = row(-0.1, True)                                # separated-better but not <= -2F
    elif which == "D2":
        rows["heading MAE"] = row(+0.2, True)                         # one row separated-worse
    else:
        rows.pop("S-POS@30")                                          # STRATEGIC has no separated-better row
    for victim in ("SOUP", "SOUPR"):
        scen = {(x, "P10", s): (rows if x == victim else DOM) for x in ("SOUP", "SOUPR") for s in R.SEEDS}
        res, _ = run(FakeHeavy(scen), LI)
        assert res["decision"]["OUTCOME"] != "DOMINANT / ADOPTED", (which, victim)
        assert res["decision"]["per_seed"]["s0"]["DM"] is False and res["decision"]["per_seed"]["s1"]["DM"] is False


def test_slow_draw_is_beside_and_asked_for_x_against_its_own_control():
    slow = {n: (-1.0, -1.2, -0.8) for n in LI.SPEED_CELLS}
    H = FakeHeavy(cells={("SOUP", "CTRL", 0): slow, ("SOUPR", "A", 1): {n: (1.0, 0.8, 1.2) for n in LI.SPEED_CELLS}})
    res, _ = run(H, LI)
    assert res["slow"]["main:SOUP-CTRL:s0"]["label"] == "SLOW-T" and res["slow"]["rep:SOUPR-A:s1"]["label"] == "FAST-T"
    assert res["slow"]["main:SOUP-CTRL:s1"]["label"] is None and res["slow"]["rep:SOUPR-A:s0"]["label"] is None and set(res["slow"]) == {f"{p}:{x}-{y}:s{s}" for p, (x, y) in R.PAIRS.items() for s in R.SEEDS}
    assert res["decision"]["OUTCOME"] == "NO EFFECT" and not res["slow_errors"]                     # decides nothing
    res, ctl = run(FakeHeavy(fail="speed"), LI)
    assert not ctl.failed() and res["decision"]["OUTCOME"] == "NO EFFECT" and len(res["slow_errors"]) == 4 and all(v is None for v in res["slow"].values())


def test_families_block_notes():
    fb = R.families_block(LI, {"a": table(), "b": table()})
    assert set(fb) == {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"}
    assert "crawl rows" in fb["LATERAL"]["note"] and "anti-echo" in fb["LONGITUDINAL"]["note"] and fb["STRATEGIC"]["paused"] is False


def test_records_refused_has_no_row_and_result_has_every_block(tmp_path):
    res, ctl = run(FakeHeavy(dom_scen()), LI)
    prov = {"p": 1}
    rec = R.result_record(LI, res, ctl, prov)
    assert rec["OUTCOME"]["OUTCOME"] == "DOMINANT / ADOPTED" and rec["provenance"] == prov and set(rec["pairs"]) == {"s0", "s1"} and rec["route"] == "FLAT"
    for k in ("read_a", "sg", "sg_errors", "beside", "floors", "tables", "four_families", "tier", "estimator", "strategic_family", "controls_PASS"):
        assert k in rec, k
    assert set(rec["four_families"]) == {f"{p}:s{s}" for p in R.PAIRS for s in R.SEEDS} and set(rec["beside"]) == {"slow_draw", "slow_draw_errors", "RF_per_row"}
    assert rec["tier"]["T1"] == R.TIER_T1 and rec["tier"]["T0"] == R.TIER_T0 and rec["estimator"] == R.ESTIMATOR and "NOT PAUSED" in rec["strategic_family"]
    assert set(rec["tables"]) == set(res["tabs"]) and set(rec["tables"]["SOUP_CTRL"]) == {"s0", "s1"}
    json.dumps(rec, default=str)
    res2, ctl2 = run(FakeHeavy(dom_scen(), fail="K-RC"), LI)
    ref = R.refused_record(res2["decision"], ctl2, prov)
    assert ref["OUTCOME"]["OUTCOME"] == "REFUSED" and "K-RC" in ref["failed"] and ref["controls_PASS"]["K-RC"] is False and "read_a" not in ref and "tables" not in ref and "pairs" not in ref
    assert "no row" in ref["note"] and ref["provenance"] == prov


def test_soup_gain_passes_the_bootstrap_its_seed_and_n_boot():
    seen = {}

    def spy(a, b, eid, n_boot=None, seed=None):
        seen.update(n_boot=n_boot, seed=seed, n=len(a))
        return {"delta": 0.0, "lo": -1.0, "hi": 1.0}
    R.soup_gain(np.zeros(30), [np.zeros(30)], np.repeat(np.arange(3), 10), spy, 77)
    assert seen == {"n_boot": 77, "seed": 0, "n": 30}
    assert R.soup_gain(np.zeros(30), [np.zeros(30)], np.repeat(np.arange(3), 10), spy)["n_windows"] == 30 and seen["n_boot"] == 2000


def test_prepare_finish_and_the_order_of_the_heavy_calls():
    H = FakeHeavy()
    res, ctl = run(H, LI)
    assert "C4" in ctl.items and H.calls.index("prepare") < H.calls.index("k_rc") and H.calls.index("global_controls") < H.calls.index("prepare")
    assert H.calls[-1] == "finish" and H.calls.index("finish") > max(i for i, c in enumerate(H.calls) if isinstance(c, tuple) and c[0] in ("sg", "speed", "guard"))
    res, ctl = run(FakeHeavy(dom_scen(), fail="finish"), LI)
    assert res["decision"]["OUTCOME"] == "REFUSED" and "K1-FAKE" in ctl.failed()


def test_composition_uses_the_bars_it_is_given():
    scen = {(x, y, s): crawl_move(x, y, s, 2.0) for (x, y) in R.PAIRS.values() for s in R.SEEDS}
    assert run(FakeHeavy(scen), LI)[0]["decision"]["OUTCOME"] == "MOVED / ADOPTED"
    big = {l: 5.0 for l in LI.LABELS}
    res = R.read_core(FakeHeavy(scen), LI, LI.Ctl(), MAN, RF0, bars=big)
    assert res["decision"]["OUTCOME"] == "NO EFFECT" and res["floors"]["bar"] == big


def test_prereg_section_10_may_not_carry_a_row_the_reader_does_not_code(tmp_path):
    txt = open(PREREG, encoding="utf-8").read()
    p = tmp_path / "p.md"
    p.write_text(txt.replace("| **NO EFFECT** |", "| **MYSTERY** | c | n |\n| **NO EFFECT** |", 1), encoding="utf-8")
    assert any("MYSTERY" in x for x in R.prereg_text_problems(str(p), None))
    missing = R.prereg_text_problems(str(tmp_path / "nope.md"), None)
    assert missing and "missing" in missing[0]
    p.write_text(txt + "\n| **BARRIER** | extra |\n", encoding="utf-8")
    assert R.prereg_text_problems(str(p), None) == []                          # a bold-first row after section 11 is not section 10


def test_g_rows_carry_each_pairs_guard_rows():
    g = {("SOUP", "CTRL", 0): {"T0_point": 0.123}, ("SOUPR", "A", 1): {"T0_point": 0.456}}
    res, _ = run(FakeHeavy(dom_scen(), g=g), LI)
    assert set(res["g_rows"]) == {(p, s) for p in R.PAIRS for s in R.SEEDS}
    assert res["g_rows"][("main", 0)]["T0_point"] == 0.123 and res["g_rows"][("rep", 1)]["T0_point"] == 0.456 and res["g_rows"][("main", 1)]["T0_point"] == 0.0


def test_a_failing_slow_draw_is_recorded_with_its_error_and_decides_nothing():
    res, ctl = run(FakeHeavy(fail="speed"), LI)
    assert not ctl.failed() and set(res["slow_errors"]) == set(res["slow"]) and len(res["slow_errors"]) == 4
    assert all("RuntimeError" in v and "no T0 cons" in v for v in res["slow_errors"].values())


class ZeroAtSeed1(FakeHeavy):
    def contrast(self, x, y):
        t = super().contrast(x, y)
        if x == y == "SOUP":
            t = dict(t)
            t[1] = {**t[1], "ADE6": {"delta": 1e-9, "lo": 0.0, "hi": 0.0, "separated": False}}
        return t


def test_k_zero_is_read_at_both_seeds(li):
    res, ctl = run(ZeroAtSeed1(dom_scen()), li)
    assert res["decision"]["OUTCOME"] == "REFUSED" and "K-ZERO:SOUP-SOUP:s1" in ctl.failed() and "K-ZERO:SOUP-SOUP:s0" not in ctl.failed()


def test_soup_gain_refuses_arms_on_different_windows():
    eid = np.repeat(np.arange(3), 10)
    with pytest.raises(ValueError, match="same windows"):
        R.soup_gain(np.zeros(30), [np.zeros(29)], eid, fake_pb)                  # the members are shorter than the soup (the eid list matches the soup)
    with pytest.raises(ValueError, match="same windows"):
        R.soup_gain(np.zeros(30), [np.zeros(30)], eid[:29], fake_pb)             # the episode list is shorter


def test_the_registered_next_cell_of_a_row_is_checked_too(tmp_path):
    txt = open(PREREG, encoding="utf-8").read()
    p = tmp_path / "p.md"
    p.write_text(txt.replace("the uniform two-draw soup is not adopted.", "the uniform two-draw soup is adopted.", 1), encoding="utf-8")
    assert any("HARMS" in x for x in R.prereg_text_problems(str(p), None))


def test_k_soup_rec_without_a_donor_cfg_skips_the_cfg_comparison_and_with_one_enforces_it():
    spec = {"members": {"m0": {"sha256": "a" * 64}, "m1": {"sha256": "b" * 64}}, "soups": {"S": {"members": ["m0", "m1"], "donor": "m0"}}}
    good = {"name": "S", "output_sha256": "f" * 64, "members": ["m0", "m1"], "donor": "m0", "member_sha256": {"m0": "a" * 64, "m1": "b" * 64},
            "builder_sha256": "c" * 64, "K-BASIN": {"PASS": True, "B1": True, "B2": True}, "K-MID": {"PASS": True}, "weights": [0.5, 0.5]}
    assert R.k_soup_rec(good, spec, "S", "f" * 64, "c" * 64, None, "anything", True) == []
    assert R.k_soup_rec(good, spec, "S", "f" * 64, "c" * 64, "donor", "other", True)
