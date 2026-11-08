"""refcv9 E1 arm T1v (``--r9-hist-supervise-geom``) -- the GATE, ARGV, G-EVAL and VERDICT side (the lever itself is ``test_r9_e1_t1v.py``).

  1  FLAG PINS      the flag is the SEVENTH member of ``E1_T1_FLAGS`` == the trainer's ``BWF.GEOM_FLAG`` == both harnesses' ``T1_FLAGS``; the FULL family's copy of the
                    E1-BOX-OWN-RECORD item keeps the six-flag literal (so the e2 / full / full-d1 fingerprints do not move)
  2  ARGV           ``r9e1_T1v.argv.json`` == the TOOL's output (byte-equal, via the CLI) == C0 + the flag + ``--out`` and nothing else; the trainer's parser reads it as geom on /
                    warp off; the emitter and ``check_t1_arm_pins`` refuse a base that already carries the flag and T1v + ``--r9-bev-warp``
  3  e1rig          ADMITS it (static rules, forbidden levers, no open item, no own-record row); T1v + warp is refused by the lever's own ``pin_args`` (the build), not by a static rule
  4  FULL FAMILY    e2 / full / full-d1 hit R9F-UNCLASSIFIED-LEVER on it (fail closed; T1v enters K2 only by a FLIP after its read); the admitted K2 argv itself is untouched
  5  G-LIVE         T1v is NEVER entered into ``BEV_WARP_LIVE`` (no gate, no fuse); the T1h argv IS (positive control)
  6  RECORDS        every overfit judge requires the stamp ``bev_warp_not_certified == [--r9-hist-supervise-geom]``; the harnesses strip the flag from what they build
  7  G-EVAL         ``rig_eval_t1_build_reasons`` (trainer vs loader HistGeom), ``rig_eval_probe_dataset`` (the pose-only attach), and a REAL tiny rig: both builds forward alike
                    on a probe batch and the forward REFUSES when the attach is gone
  8  VERDICT        ``V.ARMS["T1v"]`` (ref C0, T1h's primary and keep rule), the REPLACEMENT READ vs T1h (R1 / R2) and the PREREG's four committed outcomes
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402
import r9_e1_make_argv as MK  # noqa: E402
import r9_e1_verdict as V  # noqa: E402
import refc_v3_train as T  # noqa: E402
import test_bev_warp as TB  # noqa: E402
import test_launch_gate_e1rig_t1 as E1T  # noqa: E402
import test_launch_gate_geval_t1 as GT  # noqa: E402
import test_r9_e1_verdict as TV  # noqa: E402
from tanitad.train import bev_warp_flag as BWF  # noqa: E402

G = pytest.importorskip("g_box_overfit")
O = pytest.importorskip("map_hires_overfit")

GEOM = "--r9-hist-supervise-geom"
RIG = LG.PROFILES["refcv9-e1rig"]
RUNS = ROOT / "ops" / "runs.d"
FULL_FAMILY = ("refcv9-e2", "refcv9-full", "refcv9-full-d1")
PRE_T1V = ("--r9-bev-warp", "--r9-bev-warp-lags", "--r9-bev-warp-shuffle", "--r9-bev-warp-supervise-hist", "--r9-bev-warp-fuse-init", "--r9-bev-warp-fill-only")


def _argv(fname):
    return json.loads((RUNS / fname).read_text(encoding="utf-8"))["argv"]


C0, T1, T1H, T1V = (_argv("r9e1_%s.argv.json" % n) for n in ("C0", "T1", "T1h", "T1v"))
K0, K2 = _argv("r9e2_K0.argv.json"), _argv("r9e2_K2.argv.json")
WARP = "--r9-bev-warp"


def _diff(a, b):
    da, db = dict(LG.flag_pairs(a)), dict(LG.flag_pairs(b))
    return {f for f in set(da) | set(db) if da.get(f) != db.get(f)}


# ============================================================================================== 1. flag pins ==
def test_the_flag_is_the_seventh_member_of_ONE_fact_and_the_full_family_exemption_keeps_the_six_flag_literal():
    assert LG.E1_T1_FLAGS[-1] == BWF.GEOM_FLAG == G.T1_FLAGS[-1] == O.T1_FLAGS[-1] == GEOM and len(LG.E1_T1_FLAGS) == 7
    assert tuple(G.T1_FLAGS) == tuple(O.T1_FLAGS) == LG.E1_T1_FLAGS and BWF.GEOM_DEST == "r9_hist_supervise_geom"
    assert GEOM in T.build_parser()._option_string_actions
    assert MK.T1_ARMS["T1v"] == (GEOM,) and MK.T1_GEOM_ARMS == ("T1v",)
    assert LG._E1_T1_FLAGS_PRE_T1V == PRE_T1V == LG.E1_T1_FLAGS[:6] and GEOM not in LG._E1_T1_FLAGS_PRE_T1V
    assert GEOM not in LG._R9F_T1H_ADMITTED and GEOM not in LG._R9F_T1_REFUSED                         # classified in NEITHER: fail closed
    assert set(LG._R9F_T1H_ADMITTED + LG._R9F_T1_REFUSED) == set(PRE_T1V)
    for name in FULL_FAMILY:
        box = next(i for i in LG.PROFILES[name]["open_items"] if i["id"] == "E1-BOX-OWN-RECORD")
        assert GEOM not in box["when_except_exact"] and tuple(box["when_except_exact"]) == tuple(LG.E1_BOX_AUX_FLAGS) + PRE_T1V, name   # the fingerprint-preserving literal
        assert box["when_prefixes"] == ("--r9-box-",) and LG.open_item_hits(box, K0 + [GEOM]) == [], name   # the exemption list is INERT for any --r9-* T1 flag
    rig_box = next(i for i in RIG["open_items"] if i["id"] == "E1-BOX-OWN-RECORD")
    assert LG.open_item_hits(rig_box, T1V) == []                                                       # e1rig: the flag is exempt there, as the other six are


# ===================================================================================================== 2. argv ==
def test_the_committed_T1v_argv_is_the_TOOLS_output_and_is_C0_plus_the_flag_plus_out_only(tmp_path):
    assert MK.t1_arm_argv(C0, "T1v") == T1V
    out = tmp_path / "t1v.argv.json"
    MK.main(["--base", str(RUNS / "r9e1_C0.argv.json"), "--t1-arm", "T1v", "--write", str(out)])
    assert out.read_bytes() == (RUNS / "r9e1_T1v.argv.json").read_bytes()                              # the file was GENERATED, never hand-written
    assert _diff(C0, T1V) == {GEOM, "--out"} and dict(LG.flag_pairs(T1V))[GEOM] == []
    assert [f for f, _ in LG.flag_pairs(C0)] == [f for f, _ in LG.flag_pairs(T1V) if f != GEOM]        # same flags, same order
    assert not any(f.startswith("--r9-bev-warp") for f, _ in LG.flag_pairs(T1V))                      # none of the four T1h flags, no lags, no shuffle
    i = T1V.index(GEOM)
    assert T1V[i - 2:i] == ["--map-hires-near-refine-blocks", "1"] and T1V[i + 1] == "--bev-source"
    assert dict(LG.flag_pairs(T1V))["--out"] == ["/home/nvidia/r8ns/r9e1/runs/r9e1-T1v"]
    raw = (RUNS / "r9e1_T1v.argv.json").read_bytes()
    rec = json.loads(raw)
    assert b"\r" not in raw and rec["arm"] == "r9e1_T1v" and rec["seed"] == 0 and rec["derived_from"] == "r9e1_C0" and rec["steps"] == 4000
    assert rec["lever"]["flags"] == [GEOM] and rec["base_sha256"] == hashlib.sha256((RUNS / "r9e1_C0.argv.json").read_bytes()).hexdigest()
    assert LG.argv_sha256(T1V) not in {LG.argv_sha256(C0), LG.argv_sha256(T1), LG.argv_sha256(T1H)}
    a = T.build_parser().parse_args(T1V)                                                               # the TRAINER'S parser: geom on, the warp bundle off
    assert BWF.geom_on(a) is True and BWF.is_on(a) is False and a.r9_hist_supervise_geom is True
    assert not (a.r9_bev_warp or a.r9_bev_warp_shuffle or a.r9_bev_warp_supervise_hist or a.r9_bev_warp_fill_only)
    BWF.pin_args(a)                                                                                    # and the lever's own pin accepts it


def test_the_emitter_and_the_pins_REFUSE_what_they_must_for_T1v():
    for base, arm in ((T1V, "T1v"), (T1V, "T1"), (LG.set_flag(C0, GEOM, []), "T1h")):                # derive from C0 only: a base carrying the flag is refused
        with pytest.raises(SystemExit, match="already carries"):
            MK.t1_arm_argv(base, arm)
    MK.check_t1_arm_pins(T1V, "T1v")                                                                   # control: the real argv passes
    with pytest.raises(SystemExit, match="ALTERNATIVE to the T1h bundle"):
        MK.check_t1_arm_pins(LG.set_flag(T1V, WARP, []), "T1v")                                        # T1v + --r9-bev-warp
    with pytest.raises(SystemExit, match="did not parse as on"):
        MK.check_t1_arm_pins(T1, "T1v")                                                                # the wrong arm name for the argv, both ways
    with pytest.raises(SystemExit, match="did not parse as on"):
        MK.check_t1_arm_pins(T1V, "T1")
    with pytest.raises(SystemExit, match="only T1v carries it"):
        MK.check_t1_arm_pins(LG.set_flag(T1, GEOM, []), "T1")
    with pytest.raises(SystemExit):
        MK.check_t1_arm_pins(LG.set_flag(T1V, "--map-hires", ["off"]), "T1v")                          # the lever's pin: needs --map-hires on
    a = T.build_parser().parse_args(LG.set_flag(T1V, WARP, []))
    with pytest.raises(SystemExit, match="ALTERNATIVE to the T1h bundle"):
        BWF.pin_args(a)                                                                                # what the trainer's build runs (refc_v3_train._pin_refcv8)
    assert inspect.getsource(T._pin_refcv8).count("_bw.pin_args(args)") == 1


# ===================================================================================================== 3. e1rig ==
def test_e1rig_ADMITS_T1v_exactly_as_it_admits_C0(tmp_path):
    init = tmp_path / "ckpt_ema_48852.pt"
    init.write_bytes(b"stand-in for the shipped refcv8 EMA-48,852")
    prof = dict(RIG, init_pin=dict(RIG["init_pin"], init_from_md5=hashlib.md5(init.read_bytes()).hexdigest()))
    pmap = [("/home/nvidia/r8ns/ckpt", str(tmp_path))]
    assert LG.profile_argv_rules(prof, C0, {}, pmap) == ([], [])
    assert LG.profile_argv_rules(prof, T1V, {}, pmap) == ([], [])
    assert LG.forbidden_lever_reasons(RIG, T1V)[0] == LG.forbidden_lever_reasons(RIG, C0)[0] == []
    assert LG.required_flag_reasons(prof, T1V) == LG.required_flag_reasons(prof, C0)
    assert LG.open_item_reasons(RIG, T1V) == [] and LG.open_item_reasons(RIG, LG.set_flag(C0, GEOM, [])) == []
    assert LG.own_record_row(RIG, T1V, "G-MAP-OVERFIT") is None and LG.own_record_row(RIG, T1V, "G-BOX-OVERFIT") is None
    ids = [x.split(":")[0] for x in LG.open_item_reasons(RIG, LG.set_flag(T1V, "--r9-hist-supervise-geom-v2", []))]   # the exemption is EXACT: a sibling flag is still unclassified
    assert ids == ["OPEN ITEM E1-BOX-OWN-RECORD"], ids


# ============================================================================================ 4. full family ==
@pytest.mark.parametrize("name", FULL_FAMILY)
def test_RED_the_full_family_REFUSES_T1v_through_R9F_UNCLASSIFIED_LEVER_and_the_admitted_K2_argv_is_untouched(name):
    p = LG.PROFILES[name]
    it = next(i for i in p["open_items"] if i["id"] == "R9F-UNCLASSIFIED-LEVER")
    assert LG.open_item_hits(it, K0) == [] and LG.open_item_hits(it, K2) == []                       # controls: the admitted argvs hit nothing
    assert LG.open_item_hits(it, K0 + [GEOM]) == [GEOM] and LG.open_item_hits(it, K2 + [GEOM]) == [GEOM]
    assert LG.open_item_reasons(p, K2) == []
    for argv in (K2 + [GEOM], T1V):
        r = LG.open_item_reasons(p, argv)
        assert len(r) >= 1 and any(x.startswith("OPEN ITEM R9F-UNCLASSIFIED-LEVER") and GEOM in x for x in r), (name, r)
    assert LG.open_item_reasons(p, K2 + [GEOM]).__len__() == 1                                         # exactly the one item, not a forbidden lever
    assert LG.forbidden_lever_reasons(p, K2 + [GEOM])[0] == LG.forbidden_lever_reasons(p, K2)[0] == []


# ================================================================================================ 5. G-LIVE ==
def test_T1v_never_enters_BEV_WARP_LIVE_and_the_T1h_argv_does():
    assert LG.BEV_WARP_LIVE["flag"] == WARP != GEOM and not LG.has_flag(T1V, LG.BEV_WARP_LIVE["flag"])
    assert LG.judge_bev_warp_live(T1V, {}, None, expect_steps=10) == ([], {})
    junk = {LG.BEV_WARP_LIVE["gate"]: {"numel": 4, "steps": 3, "nz_steps": 0, "nonfinite": False}}
    assert LG.judge_bev_warp_live(T1V, junk, None, expect_steps=10) == ([], {})                        # whatever the smoke tracked: not read for T1v
    assert LG.judge_bev_warp_live(T1H, {}, None, expect_steps=10)[0] != []                              # positive control: the T1h argv IS entered (gate / fuse missing)
    assert LG.judge_bev_warp_live(LG.set_flag(T1V, WARP, []), {}, None, expect_steps=10)[0] != []      # and it is the WARP flag that is the switch
    src = inspect.getsource(LG.judge_bev_warp_live)
    assert 'has_flag(argv, BEV_WARP_LIVE["flag"])' in src and "E1_T1_FLAGS" not in src and "r9_hist" not in src


# ================================================================================================ 6. records ==
JUDGES = {"map1": (lambda a, **o: E1T._map1(a, **o), E1T._j_map1, T1V),
          "box1": (lambda a, **o: E1T._box1(a, **o), E1T._j_box1, T1V),
          "banded": (lambda a, **o: E1T.OR._record(a, **{**O.t1_stamp_fields({"argv": a}), **o}), E1T._j_banded, LG.set_flag(E1T.M1, GEOM, []))}


def test_the_build_stamp_and_the_harness_strip_for_T1v():
    assert G.b3_record_fields(T1V) == {G.T1_STAMP: [GEOM], G.T1_STAMP_WHY: G.T1_WHY} and O.t1_flags_of(T1V) == [GEOM]
    assert LG.bev_warp_stamp_reasons("X", {G.T1_STAMP: [GEOM], G.T1_STAMP_WHY: "w"}, T1V) == ([], {"argv_t1_set": [GEOM], "record_stamp": [GEOM]})
    assert "stamps no `bev_warp_not_certified`" in LG.bev_warp_stamp_reasons("X", {}, T1V)[0][0]
    assert "T1 set is ['--r9-hist-supervise-geom']" in LG.bev_warp_stamp_reasons("X", {G.T1_STAMP: [WARP], G.T1_STAMP_WHY: "w"}, T1V)[0][0]
    assert G.strip_t1(T1V) == (LG.set_flag(C0, "--out", LG.flag_values(T1V, "--out")), [GEOM])           # the harness builds EXACTLY C0
    build, rec = G.harness_build_argv(T1V)
    assert GEOM not in build and rec["bev_warp_flags_not_built"] == [GEOM]
    assert BWF.geom_on(T.build_parser().parse_args(build)) is False and BWF.is_on(T.build_parser().parse_args(build)) is False


@pytest.mark.parametrize("jn", sorted(JUDGES))
def test_the_overfit_judges_REQUIRE_the_T1v_stamp_and_it_must_equal_the_argvs_set(jn, tmp_path):
    mk, judge, argv = JUDGES[jn]
    good = mk(argv)
    assert good[G.T1_STAMP] == [GEOM] and good[G.T1_STAMP_WHY]
    reasons, det = judge(tmp_path, good, argv)
    assert reasons == [] and det["bev_warp_not_certified"] == {"argv_t1_set": [GEOM], "record_stamp": [GEOM]}
    no_stamp = {k: v for k, v in good.items() if k not in (G.T1_STAMP, G.T1_STAMP_WHY)}
    for rec, want in ((no_stamp, "stamps no `bev_warp_not_certified`"),
                      (dict(good, bev_warp_not_certified=[WARP]), "T1 set is ['--r9-hist-supervise-geom']"),                  # T1's stamp offered to T1v
                      (dict(good, bev_warp_not_certified=list(PRE_T1V)), "T1 set is ['--r9-hist-supervise-geom']"),            # T1h-with-everything's stamp
                      (dict(good, bev_warp_not_certified=[GEOM, WARP]), "T1 set is ['--r9-hist-supervise-geom']"),
                      (dict(good, bev_warp_not_certified_why=" "), G.T1_STAMP_WHY)):
        r, _ = judge(tmp_path, rec, argv)
        assert r and want in " | ".join(r), (jn, want, r)
    other = LG.set_flag(argv, GEOM, None)                                                                 # the record made for T1v offered to the argv WITHOUT the flag
    r, _ = judge(tmp_path, good, other)
    assert r and "T1 set is []" in " | ".join(r)


# ================================================================================================ 7. G-EVAL ==
def _hg(rows):
    return types.SimpleNamespace(lag_rows=tuple(rows))


def _geom_model(rows=(5, 10), branch_rows=None, lags=(0.5, 1.0), state=True, branch=True):
    st = types.SimpleNamespace(lags_s=tuple(lags), lag_rows=tuple(rows)) if state else None
    mh = types.SimpleNamespace(hist_geom=_hg(rows if branch_rows is None else branch_rows)) if branch else types.SimpleNamespace()
    return types.SimpleNamespace(_map_hires=mh, _r9_hist_geom=st)


def _gargs(declared=True, warp=False):
    return types.SimpleNamespace(r9_hist_supervise_geom=declared, r9_bev_warp=warp)


def test_G_EVAL_build_reasons_T1_dict_is_unchanged_and_T1v_adds_one_key_and_reads_the_two_builds():
    plain = types.SimpleNamespace()
    assert LG.rig_eval_t1_build_reasons(_gargs(False), plain, _gargs(False), plain) == ([], {"on": False})        # neither declares / builds: exactly the tip's answer
    r, d = LG.rig_eval_t1_build_reasons(_gargs(), _geom_model(), _gargs(), _geom_model())
    assert r == [] and d["on"] is False and set(d) == {"on", "hist_geom"} and d["hist_geom"]["trainer"] == d["hist_geom"]["loader"]
    assert d["hist_geom"]["trainer"] == {"declared": True, "state_built": True, "branch_built": True, "lags_s": [0.5, 1.0], "lag_rows": [5, 10], "branch_lag_rows": [5, 10]}
    r, _ = LG.rig_eval_t1_build_reasons(_gargs(), _geom_model(), _gargs(), _geom_model(rows=(5, 8)))               # other lag rows
    assert any("lag_rows differs between the builds: trainer [5, 10] vs loader [5, 8]" in x for x in r), r
    r, _ = LG.rig_eval_t1_build_reasons(_gargs(), _geom_model(), _gargs(), _geom_model(branch_rows=(5, 9)))       # the 10 cm branch holds other rows than the model state
    assert any("branch_lag_rows differs" in x for x in r) or any("declares" in x for x in r), r
    assert LG.rig_eval_t1_build_reasons(_gargs(), _geom_model(), _gargs(), _geom_model(lags=(0.5, 2.0)))[0] != []  # other lags (seconds)
    r, _ = LG.rig_eval_t1_build_reasons(_gargs(), _geom_model(), _gargs(), plain)                                   # the loader declares T1v but built nothing
    assert any("the loader's argv declares True but its build carries the model state False / the 10 cm HistGeom False" in x for x in r), r
    r, _ = LG.rig_eval_t1_build_reasons(_gargs(), _geom_model(), _gargs(), _geom_model(state=True, branch=False))
    assert any("HistGeom False" in x for x in r), r
    r, _ = LG.rig_eval_t1_build_reasons(_gargs(False), _geom_model(), _gargs(False), _geom_model())               # built but NOT declared, both sides
    assert sum("declares False but its build carries" in x for x in r) == 2, r
    r, d = LG.rig_eval_t1_build_reasons(_gargs(False, warp=True), plain, _gargs(False, warp=True), plain)          # the T1 (warp) answer carries no hist_geom key
    assert d["on"] is True and "hist_geom" not in d and set(d) == {"on", "trainer", "loader"}


class _FakeBw:
    def __init__(self):
        self.calls = []

    def attach_dataset(self, ds, model, args):
        self.calls.append((ds, model, args))
        return {"fake_attach": True}


class _FakeDS:
    def __init__(self, eps, **kw):
        self.eps, self.kw = eps, kw

    def __len__(self):
        return 3


def _fake_probe(monkeypatch, args, model=None, with_bw=True, bw=None):
    import tanitad.data.v2_dataset as V2
    monkeypatch.setattr(V2, "build_v2_providers", lambda caches, lru_size: [object(), object()])
    tr = types.SimpleNamespace(V3Dataset=_FakeDS, **({"_bw": bw} if with_bw else {}))
    cfg = types.SimpleNamespace(core=types.SimpleNamespace(window=4, encoder=types.SimpleNamespace(in_channels=3)))
    a = types.SimpleNamespace(eval_labels=None, eval_cache="/x", v2_lru=2, u8_batches=False, pose_sync_sidecar=None, r9_bev_warp=False, r9_hist_supervise_geom=False)
    for k, v in args.items():
        setattr(a, k, v)
    return LG.rig_eval_probe_dataset(types.SimpleNamespace(trainer=lambda: tr), cfg, a, model=model), a


def test_G_EVAL_the_probe_dataset_attaches_the_pose_history_for_T1v_only(monkeypatch):
    bw = _FakeBw()
    model = object()
    (ds, _eps, rec), a = _fake_probe(monkeypatch, {"r9_hist_supervise_geom": True}, model=model, bw=bw)
    assert len(bw.calls) == 1 and bw.calls[0][0] is ds and bw.calls[0][1] is model and bw.calls[0][2] is a      # attached ONCE, to the probe dataset, with the LOADER'S model
    assert rec["hist_geom"] == {"fake_attach": True} and "bev_warp" not in rec
    bw0 = _FakeBw()
    (_ds, _e, rec0), _ = _fake_probe(monkeypatch, {}, model=model, bw=bw0)                                       # C0: not attached, record is the tip's
    assert bw0.calls == [] and set(rec0) == {"rig", "labels", "n_episodes", "n_windows", "newest_row_only", "label_clock"}
    bwt = _FakeBw()
    (_ds, _e, rect), _ = _fake_probe(monkeypatch, {"r9_bev_warp": True}, model=model, bw=bwt)                    # T1: the warp attach, unchanged
    assert len(bwt.calls) == 1 and "bev_warp" in rect and "hist_geom" not in rect
    with pytest.raises(LG.GateError, match="--r9-hist-supervise-geom but the probe dataset was built without the loader's model"):
        _fake_probe(monkeypatch, {"r9_hist_supervise_geom": True}, model=None, bw=_FakeBw())
    with pytest.raises(LG.GateError, match="the trainer module has no `_bw`"):
        _fake_probe(monkeypatch, {"r9_hist_supervise_geom": True}, model=model, with_bw=False)
    src = inspect.getsource(LG.eval_identity)
    assert "rig_eval_probe_dataset(L, cfg_l, args_l, model=model_l)" in src and 'det["hist_geom"] = t1d["hist_geom"]' in src


@pytest.fixture(scope="module")
def t1v_rig():
    """A trainer-built and a LOADER-built T1v rig model (no fuse), both in G-EVAL's eval-time flags -- as ``test_launch_gate_geval_t1.rig`` for T1."""
    with pytest.MonkeyPatch.context() as mp:
        over = dict(r9_bev_warp=False, r9_hist_supervise_geom=True)
        mt, cfg, args_t = TB._rig(mp, on=True, **over)
        ml, _, args_l = TB._rig(mp, on=True, **over)
        ml.load_state_dict(mt.state_dict(), strict=True)
        for m in (mt, ml):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        yield types.SimpleNamespace(mt=mt, ml=ml, cfg=cfg, args_t=args_t, args_l=args_l)


def test_G_EVAL_a_REAL_T1v_rig_forwards_alike_on_the_probe_batch_and_the_forward_REFUSES_without_the_attach(t1v_rig, monkeypatch):
    rig = t1v_rig
    assert not hasattr(rig.mt._map_hires, "bev_warp") and rig.mt._map_hires.hist_geom is not None           # no fuse; a HistGeom on the 10 cm branch
    _ds, rec, batch = GT._probe(rig, monkeypatch)
    assert rec["hist_geom"]["lag_rows"] == list(rig.ml._r9_hist_geom.lag_rows) and rec["hist_geom"]["with_frames"] is False
    assert sorted(k for k in batch if k.startswith("bw_")) == sorted(BWF.GEOM_BATCH_KEYS)                    # POSE-ONLY: no bw_frames
    ot, ol = GT._out(rig.mt, batch, rig.args_t), GT._out(rig.ml, batch, rig.args_l)
    assert GT._out(rig.mt, batch, rig.args_t) == ot                                                          # G-EVAL's determinism control
    assert sorted(k for k in set(ot) | set(ol) if ot.get(k) != ol.get(k)) == [] and any(k.startswith("out.perception.map_hires_logits") for k in ot)
    r, d = LG.rig_eval_t1_build_reasons(rig.args_t, rig.mt, rig.args_l, rig.ml)
    assert r == [] and d["hist_geom"]["trainer"]["state_built"] is True
    monkeypatch.setattr(BWF, "attach_dataset", lambda ds, model, args: {})                                  # RED: the attach gone -> no pose history -> the forward REFUSES, by name
    _ds2, _rec2, batch2 = GT._probe(rig, monkeypatch)
    assert not [k for k in batch2 if k.startswith("bw_")]
    for m, a in ((rig.mt, rig.args_t), (rig.ml, rig.args_l)):
        with pytest.raises(SystemExit, match=r"--r9-hist-supervise-geom but the batch carries no \['bw_pose_now', 'bw_pose_past', 'bw_has'\]"):
            GT._out(m, batch2, a)


# ================================================================================================= 8. verdict ==
FL = TV.base_floor()
NEU = TV.neutral(FL)
PRIM = "nf_prec_0_10"


def _vs_c0(delta=None, sep=True, **others):
    p = NEU if delta is None else TV.with_delta(NEU, PRIM, delta, sep)
    for k, d in others.items():
        p = TV.with_delta(p, k, d)
    return V.judge("T1v", "C0", FL, p)


def _vs_t1h(delta=None, sep=True, **others):
    p = NEU if delta is None else TV.with_delta(NEU, PRIM, delta, sep)
    for k, d in others.items():
        p = TV.with_delta(p, k, d)
    return V.judge("T1v", "T1h", FL, p)


def test_the_arm_row_resolve_ref_and_the_vs_C0_read_is_T1hs_keep_rule_UNRELAXED():
    a, h = V.ARMS["T1v"], V.ARMS["T1h"]
    assert a["kind"] == "lever" and a["ref"] == "C0" and a["allowed"] == ("C0", "T1h") and a["argv_flags"] == (GEOM,)
    assert [m["keys"] for m in a["metrics"]] == [m["keys"] for m in h["metrics"]] == [(PRIM,)] and a["metrics"][0]["better"] == h["metrics"][0]["better"] == "higher"
    assert a["metrics"][0]["clause3"] is True and a["next_lever"] == V.T1V_NEXT_LEVER and V.REPLACEMENT_READS == {("T1v", "T1h")}
    assert V.resolve_ref("T1v", None) == "C0" and V.resolve_ref("T1v", "C0") == "C0" and V.resolve_ref("T1v", "T1h") == "T1h"
    for arm, ref in (("T1v", "M1"), ("T1v", "B1"), ("T1v", "T1hs"), ("T1h", "T1v"), ("T1hs", "T1v")):
        with pytest.raises(V.Refused):
            V.resolve_ref(arm, ref)
    for d, sep, others in ((None, True, {}), (0.75, True, {}), (0.4, True, {}), (0.75, False, {}), (-0.75, True, {}), (0.75, True, {"ap2m_all": -0.625}), (0.75, True, {"ap2m_all": -0.375})):
        v = _vs_c0(d, sep, **others)                                                                       # the SAME rows, T1h and T1v: the same verdict and clauses
        h = V.judge("T1h", "C0", FL, NEU if d is None else _chain(d, sep, others))
        assert v["verdict"] == h["verdict"] and TV.clauses(v) == TV.clauses(h) and v["verdict"] in ("KEEP", "DROP") and "replacement_outcome" not in v, (d, sep, others)
    assert _vs_c0().get("verdict") == "DROP" and TV.clauses(_vs_c0()) == [1, 2]
    assert _vs_c0(0.75)["verdict"] == "KEEP" and _vs_c0(0.5)["verdict"] == "DROP" and TV.clauses(_vs_c0(0.4)) == [2]       # exactly 2 x floor is not beyond it
    v = _vs_c0(0.75, ap2m_all=-0.625)
    assert v["verdict"] == "DROP" and TV.clauses(v) == [3] and [x["metric"] for x in v["clause3"]["vetoes"]] == ["ap2m_all"]


def _chain(d, sep, others):
    p = TV.with_delta(NEU, PRIM, d, sep)
    for k, x in others.items():
        p = TV.with_delta(p, k, x)
    return p


@pytest.mark.parametrize("delta,sep,r1", [(-0.6, True, False), (-0.5000001, True, False), (-0.5, True, True), (-0.4, True, True), (-0.6, False, True), (0.6, True, True), (0.0, False, True)])
def test_R1_nf_prec_0_10_T1v_minus_T1h_fails_ONLY_when_separated_worse_beyond_2x_floor(delta, sep, r1):
    v = _vs_t1h(delta, sep)
    ro = v["replacement_outcome"]
    assert v["verdict"] == "REPLACEMENT-READ" and v["failing_clauses"] == [] and ro["ref"] == "T1h" and ro["R1"]["pass"] is r1, ro["R1"]
    assert ro["R1"]["key"] == PRIM and ro["R1"]["two_x_floor"] == 0.5 and ro["R2"]["pass"] is True and ro["replaces_if_kept_vs_c0"] is r1


@pytest.mark.parametrize("delta,sep,r2", [(-0.625, True, False), (-0.375, True, True), (-0.625, False, True), (0.625, True, True)])
def test_R2_no_other_perception_primary_separated_worse_beyond_2x_its_floor(delta, sep, r2):
    ro = V.judge("T1v", "T1h", FL, TV.with_delta(NEU, "ap2m_all", delta, sep))["replacement_outcome"]
    assert ro["R1"]["pass"] is True and ro["R2"]["pass"] is r2 and ro["replaces_if_kept_vs_c0"] is r2 and ro["R2"]["complete"] is True
    assert ro["R2"]["vetoes"] == ([] if r2 else ["ap2m_all"])


def test_the_replacement_read_and_the_keep_read_are_DIFFERENT_reads_never_swapped():
    t1h_read, c0_read = _vs_t1h(0.75), _vs_c0(0.75)
    assert t1h_read["verdict"] == "REPLACEMENT-READ" and c0_read["verdict"] == "KEEP"                          # same rows, two refs, two kinds of verdict
    assert "replacement_outcome" in t1h_read and "replacement_outcome" not in c0_read
    assert _vs_c0(-0.6)["verdict"] == "DROP" and _vs_t1h(-0.6)["replacement_outcome"]["R1"]["pass"] is False  # a row that fails R1 is also not a keep vs C0 here, by its own rule
    src = inspect.getsource(V.judge)
    assert "(arm, ref) in REPLACEMENT_READS" in src


def test_the_four_committed_outcomes_are_the_PREREGS_and_the_join_returns_each():
    assert len(V.T1V_OUTCOMES) == 4 and V.T1V_OUTCOMES[0].startswith("KEPT and replaces (1-3, R1-R2)") and V.T1V_OUTCOMES[1].startswith("KEPT, fails R1 or R2")
    assert V.T1V_OUTCOMES[2].startswith("FAILS 1-2 vs C0") and V.T1V_OUTCOMES[3].startswith("FAILS 3")
    assert all(o in V.T1V_NEXT_LEVER for o in V.T1V_OUTCOMES)
    ok = lambda v: dict(v, status="OK")                                         # `build_verdict` stamps status OK on what `judge` returns   # noqa: E731
    j = lambda kv, rv: V.t1v_committed_outcome(None if kv is None else (kv if "status" in kv else ok(kv)), None if rv is None else (rv if "status" in rv else ok(rv)))   # noqa: E731
    keep, repl_ok = _vs_c0(0.75), _vs_t1h(-0.1, False)
    assert j(keep, repl_ok)["index"] == 0 and j(keep, repl_ok)["outcome"] == V.T1V_OUTCOMES[0]                      # KEPT and replaces
    assert j(keep, _vs_t1h(-0.6))["index"] == 1 and j(keep, V.judge("T1v", "T1h", FL, TV.with_delta(NEU, "ap2m_all", -0.625)))["index"] == 1   # fails R1 / fails R2
    assert j(_vs_c0(), repl_ok)["index"] == 2 and j(_vs_c0(0.4), repl_ok)["index"] == 2 and j(_vs_c0(0.75, False), repl_ok)["index"] == 2       # FAILS 1-2
    drop3 = _vs_c0(0.75, ap2m_all=-0.625)
    assert TV.clauses(drop3) == [3] and j(drop3, repl_ok)["index"] == 3 and j(drop3, repl_ok)["outcome"] == V.T1V_OUTCOMES[3]                     # FAILS 3 alone
    both = _vs_c0(0.4, ap2m_all=-0.625)
    assert TV.clauses(both) == [2, 3] and j(both, repl_ok)["index"] == 2                                          # clause 2 AND 3 fail: it is the FAILS 1-2 branch
    for kv, rv in ((None, repl_ok), ({"status": "REFUSED"}, repl_ok), (keep, None), (keep, {"status": "OK"}), (keep, dict(repl_ok, status="REFUSED"))):  # a REFUSED / missing read  # not decidable: None, never a guess
        assert j(kv, rv)["index"] is None and j(kv, rv)["outcome"] is None
    nc = V.judge("T1v", "T1h", TV.base_floor(omit=(PRIM,)), {m: r for m, r in NEU.items() if m != PRIM})
    assert nc["verdict"] == "NOT-COMPUTABLE" and j(keep, nc)["index"] is None
