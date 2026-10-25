"""PREREG_V8F_KINDX, the SEED-1 CONFIRMATION of CANDIDATE (speed): the box-2 instrument built, not launched (chain_v8f_kindx_s1.sh).

What this pins, and what each part's deliberate-regression arm is:
  (A) the s1 outcome table on synthetic arrays   CARRIER / NOT A CARRIER / REFUSED; mutations: `clears` ignores the CI, the bar moves, a carrier need not have been an s0 candidate
  (B) the s0 mode is unchanged                    the deciding block of the banked s0 evals reproduces the banked READ.json exactly (the reader was parametrised, not rewritten)
  (C) the controls at s1                          own / union / full-GT vs the standard evals, the identity oracle EXACTLY 0.0, the fallback windows, the checkpoint sha256
  (D) the predecessor predicate                   KSCORE's chain: every stop text in its SOURCE is classified (a new one fails the test); mutations: unknown proceeds, evidence ignored,
                                                  the refresh job's rc proceeds, the run-level regex matches any job
  (E) the argv diff, the gate and the run assertions    "MIXO's argv, --seed 1": the seed (and the output directory that carries it) is the ONLY difference; MIXO's banked s0 gate and the
                                                  new tip's o_interface = full record pass; mutations: a second flag moves, a probe is dropped, the model differs
  (F) the box-2 driver                            the ten evals are KIND's queue lines (flags and values), `none` is the identity patch, the guard's regex matches the driver
  (G) the chain                                   syntax, --dry-run (stage order, the lines printed), load-bearing lines and their order, pins against the reader / the MANIFEST / the PREREG
  (H) reachability on the new tip (d3d314c7)      the real trainer (synthetic data) accepts MIXO's argv with `--seed 1` and WITHOUT `--o-interface`, builds o_interface = full, passes G-DVB and
                                                  both probes, and builds the same model as the seed-0 run
No test launches anything on box 2, writes into box2_ship or touches a process it did not start.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
RAWX = PKG / "raw" / "2026-10-25-kindx"
RAWM = PKG / "raw" / "2026-10-25-mixo" / "run"
CODE = PKG / "code"
CHAIN = CODE / "box2_ops" / "chain_v8f_kindx_s1.sh"
KS_CHAIN = REPO / "stack" / "scripts" / "chain_v8f_kscore.sh"
MIXO_CHAIN = CODE / "box2_ops" / "chain_v8f_mixo.sh"
GUARD = CODE / "box2_ops" / "v8f_guard.sh"
QUEUE = CODE / "v8f_mini_box2_queue.cmd"
DRIVER = REPO / "stack" / "scripts" / "v8f_mini_eval_kindx.py"
READER = RAWX / "kindx_read.py"
CONTROLS = RAWX / "kindx_controls.py"
PREREG = REPO / "Project Steering" / "PREREG_V8F_KINDX.md"
MANIFEST = PKG / "raw" / "full" / "MANIFEST.md"
BOX2_SHIP = Path("C:/Users/Admin/box2_ship")
KINDS = ("speed", "angle", "tte", "timegap", "geom", "leadgap")
X0_SHA = "c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d"
N_EP, N_WIN, N_FB = 141, 2399, 34
CNT = itertools.count()
win = pytest.mark.skipif(sys.platform != "win32", reason="the queue is a Windows batch file")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


D = _load("v8f_mini_eval_kindx_under_test", DRIVER)


def _bash():
    """A GNU bash that can run the chain (a bare ``bash`` on Windows may be System32's WSL launcher)."""
    import shutil
    for c in (os.environ.get("KINDX_BASH"), shutil.which("bash"), "C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe", "bash"):
        if not c:
            continue
        try:
            r = subprocess.run([c, "--version"], capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and "GNU bash" in r.stdout:
                return c
        except Exception:
            pass
    return None


BASH = _bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="bash is needed to run the chain")


# =============================================================================================================================================
# the reader / controls under test: loaded fresh with the environment that configures them
# =============================================================================================================================================
def _env(monkeypatch, out, seed, s0_read=None, ev=None, be=None):
    for k, v in {"KINDX_OUT": Path(out).as_posix(), "KINDX_CODE": REPO.as_posix(), "KINDX_EV": Path(ev or Path(out) / "eval").as_posix(), "KINDX_BE": Path(be or Path(out) / "b_kind" / "eval").as_posix(),
                 "KINDX_SEED": str(seed), "KINDX_S0_READ": Path(s0_read or RAWX / "READ.json").as_posix()}.items():
        monkeypatch.setenv(k, v)


def load_reader(monkeypatch, out, seed=1, src=None, **kw):
    _env(monkeypatch, out, seed, **kw)
    m = _load(f"kindx_read_t{next(CNT)}", src or READER)
    m.B_BOOT = 200                                                       # the estimator's draws: fewer in the tests (the instrument is the same function)
    return m


def load_controls(monkeypatch, out, seed=1, src=None, **kw):
    _env(monkeypatch, out, seed, **kw)
    return _load(f"kindx_controls_t{next(CNT)}", src or CONTROLS)


def _mutant_src(tmp_path, base, old, new, name):
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    src = Path(base).read_text(encoding="utf-8")
    assert src.count(old) == 1, f"the mutation target is not unique / not there: {old!r}"
    p = tmp_path / name
    p.write_bytes(src.replace(old, new).encode("utf-8"))
    return p


# =============================================================================================================================================
# the synthetic world: 2,399 windows over 141 episodes, X_s<seed>'s ten KINDX evals with the gains asked for, in the files and the json shapes the reader and the controls read
# =============================================================================================================================================
def _eid():
    counts = np.full(N_EP, N_WIN // N_EP)
    counts[: N_WIN - counts.sum()] += 1
    return np.repeat(np.arange(N_EP), counts)


DEFAULT_GAINS = {"speed": 0.70, "angle": 0.05, "tte": 0.05, "timegap": 0.0, "geom": 0.05, "leadgap": 0.0}


def _man(spec):
    from tanitad.models import v8f_mini as M
    return {"mode": "oracle", "groups": spec, "n_code_dims_replaced": len(M.code_group_index(spec)) if spec != "none" else 0, "alpha": 1.0}


def make_world(root, seed=1, gains=None, ep_sd=None, noise=0.15, rng=7, shift_eid=False, std_dev=0.0, drop=(), extra_json=None):
    """-> root. `gains` = the per-kind true gain; `ep_sd` = {kind: sd of an episode-level random effect on that gain}. `std_dev` perturbs the standard (`evalmixo`) evals the controls compare with."""
    g = {**DEFAULT_GAINS, **(gains or {})}
    ep_sd = ep_sd or {}
    root = Path(root)
    ev = root / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    r = np.random.default_rng(rng)
    eid = _eid()
    own = 2.2 + r.normal(0, 0.5, N_WIN)
    fb = np.zeros(N_WIN, bool)
    fb[-N_FB:] = True
    z = r.normal(0, 1, (len(KINDS), N_EP))
    pred = np.zeros((N_WIN, 3, 2), np.float32)
    ckpt = f"D:\\tanitad\\runs\\v8f_mini_full\\M2b_s{seed}_mixo\\ckpt_last.pt"
    T = f"M2b-mixo_s{seed}"
    fam = {"tactical": {"lat_macro_f1": 0.8}, "strategic": {"status": "PAUSED"}}

    def arm(gk, k=None):
        a = own - gk + r.normal(0, noise, N_WIN)
        if k is not None and ep_sd.get(k):
            a = a - ep_sd[k] * z[KINDS.index(k)][eid]
        a[fb] = own[fb]
        return a

    arms = {"own": own, "none": own.copy(), "all": arm(0.83), "union": arm(0.80), **{f"only-{k}": arm(g[k], k) for k in KINDS}}
    for nm, ade in arms.items():
        if nm in drop:
            continue
        e = (eid + 1) % N_EP if (shift_eid and nm == "only-tte") else eid
        d = {"eid": e, "ade6": ade, "pred": pred}
        if nm != "own":
            d["tac_oracle_fallback"] = fb
        np.savez(ev / f"eval_{T}-kindx-{nm}.npz", **d)
        spec = None if nm == "own" else ("constraints" if nm == "union" else nm.replace("only-", ""))
        j = {"tag": f"{T}-kindx-{nm}", "ckpt": ckpt, "tac": "pred", "raw": copy.deepcopy(fam)}
        if spec is not None:
            j["tac"] = "oracle"
            j["tac_manipulation"] = _man(spec)
        j.update((extra_json or {}).get(nm, {}))
        (ev / f"eval_{T}-kindx-{nm}.json").write_text(json.dumps(j), encoding="utf-8")
    for suf, src in (("", "own"), ("-tacpred", "own"), ("-tacorc-constraints", "union"), ("-tacorc", "all")):             # the standard evals (`evalmixo`): the controls' "banked" files at s1
        np.savez(ev / f"eval_{T}{suf}.npz", eid=eid, ade6=arms[src] + std_dev, pred=pred)
    return root


def write_controls_ok(root, seed=1, sha="ab" * 32):
    (Path(root) / f"controls_s{seed}.json").write_text(json.dumps({"phase1": {"PASS": True, "checkpoint_sha256": {"observed": sha}}, "phase2": {"PASS": True}}), encoding="utf-8")


def s0_read_variant(tmp_path, name="s0.json", **changes):
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    r = json.loads((RAWX / "READ.json").read_text(encoding="utf-8"))
    r.pop("beside", None)
    r.update(changes)
    p = tmp_path / name
    p.write_text(json.dumps(r), encoding="utf-8")
    return p


def run_reader(R, tmp_path, argv=("--seed", "1", "--no-beside"), out="READ_s1.json"):
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    rc = R.main([*argv, "--out", str(tmp_path / out)])
    return rc, json.loads((tmp_path / out).read_text(encoding="utf-8"))


def reader_case(monkeypatch, tmp_path, gains=None, s0=None, ep_sd=None, **wk):
    w = make_world(tmp_path / "w", 1, gains, ep_sd, **wk)
    write_controls_ok(w)
    R = load_reader(monkeypatch, w, 1, s0_read=s0 or s0_read_variant(tmp_path))
    return R, run_reader(R, tmp_path)[1], w


# =============================================================================================================================================
# (A) THE SEED-1 OUTCOME TABLE
# =============================================================================================================================================
def test_the_banked_s0_read_is_the_candidate_this_confirmation_is_for():
    r = json.loads((RAWX / "READ.json").read_text(encoding="utf-8"))
    assert r["OUTCOME"] == "CANDIDATE" and r["candidates_ranked"] == ["speed"] and r["checkpoint_sha256"] == X0_SHA and r["bar_2F_m"] == 0.2415
    assert r["controls"] == {"phase1": True, "phase2": True} and abs(r["gain"]["speed"]["point"] - 0.7164) < 1e-4


def test_carrier_when_the_s0_candidate_clears_2F_at_s1_too(monkeypatch, tmp_path):
    R, res, _ = reader_case(monkeypatch, tmp_path, {"speed": 0.70})
    assert res["schema"] == "v8f-kindx-read-s1-v1" and res["OUTCOME"] == "CARRIER" and res["carriers"] == ["speed"]
    assert res["clears_s1"]["speed"] is True and not any(res["clears_s1"][k] for k in KINDS if k != "speed")
    assert abs(res["gain"]["speed"]["point"] - 0.70) < 0.02 and res["gain"]["speed"]["separated"] is True
    assert res["arm"] == "X = M2b-mixo_s1" and res["s0"]["candidates_ranked"] == ["speed"] and res["bar_2F_m"] == 0.2415
    assert res["committed_rule_quoted"] == R.NEXT["CANDIDATE"] and res["committed_next"] == R.NEXT["CANDIDATE"], "a carrier gets exactly the PREREG's text"
    assert res["carrier_routes"] == {"speed": "T's speed variance (two-seed code average / EMA)"}
    assert res["reading"].startswith("PREREG_V8F_KINDX seed-1 confirmation: CARRIER (speed)") and "ns" in res["reading"]


def test_not_a_carrier_when_the_candidate_misses_2F_at_s1(monkeypatch, tmp_path):
    R, res, _ = reader_case(monkeypatch, tmp_path, {"speed": 0.10})
    assert res["OUTCOME"] == "NOT A CARRIER" and res["carriers"] == [] and res["candidates_not_confirmed"] == ["speed"]
    assert res["committed_next"] == R.NEXT_S1["NOT A CARRIER"] and "commits no step for this case" in res["committed_next"]
    assert "carrier_routes" in res and res["carrier_routes"] == {}


def test_not_a_carrier_when_the_point_clears_but_the_interval_does_not_exclude_zero(monkeypatch, tmp_path):
    R, res, _ = reader_case(monkeypatch, tmp_path, {"speed": 0.30}, ep_sd={"speed": 3.0})
    g = res["gain"]["speed"]
    assert g["point"] >= 0.2415 and g["separated"] is False, g
    assert res["OUTCOME"] == "NOT A CARRIER", "a separated interval is NECESSARY: a big point with an interval through 0 is not a clearance"


def test_a_kind_that_clears_at_s1_only_is_reported_and_is_a_carrier_at_neither_seed(monkeypatch, tmp_path):
    R, res, _ = reader_case(monkeypatch, tmp_path, {"speed": 0.05, "angle": 0.60})
    assert res["clears_s1"]["angle"] is True and res["OUTCOME"] == "NOT A CARRIER" and res["carriers"] == []
    assert res["clears_at_s1_only"] == ["angle"] and "clears at s1 only" in res["reading"]


def test_two_s0_candidates_keep_their_s0_rank_and_only_the_confirmed_one_is_a_carrier(monkeypatch, tmp_path):
    s0 = s0_read_variant(tmp_path, candidates_ranked=["geom", "speed"])
    R, res, _ = reader_case(monkeypatch, tmp_path, {"speed": 0.70, "geom": 0.05}, s0=s0)
    assert res["carriers"] == ["speed"] and res["candidates_not_confirmed"] == ["geom"] and res["OUTCOME"] == "CARRIER"
    s0b = s0_read_variant(tmp_path, "s0b.json", candidates_ranked=["geom", "speed"])
    R2, res2, _ = reader_case(monkeypatch, tmp_path / "b", {"speed": 0.70, "geom": 0.50}, s0=s0b)
    assert res2["carriers"] == ["geom", "speed"], "the carriers keep the s0 rank (gain_k(X_s0)), not the s1 order"


def test_clears_is_inclusive_at_2F_and_needs_the_interval(monkeypatch, tmp_path):
    R, _, _ = reader_case(monkeypatch, tmp_path, {"speed": 0.70})
    base = {"separated": True}
    assert R.K.clears({**base, "point": 0.2415}, R.BAR_2F) and not R.K.clears({**base, "point": 0.24149}, R.BAR_2F)
    assert not R.K.clears({"separated": False, "point": 0.9}, R.BAR_2F) and not R.K.clears({**base, "point": -0.5}, R.BAR_2F)


REFUSALS = {
    "s0 read is NO CANDIDATE": (lambda tp: {"s0": s0_read_variant(tp, OUTCOME="NO CANDIDATE", candidates_ranked=[])}, "not the CANDIDATE"),
    "s0 read is another checkpoint": (lambda tp: {"s0": s0_read_variant(tp, checkpoint_sha256="00" * 32)}, "is not X_s0's"),
    "s0 read has a candidate that is not a kind": (lambda tp: {"s0": s0_read_variant(tp, candidates_ranked=["torque"])}, "candidates_ranked"),
    "s0 read has another bar": (lambda tp: {"s0": s0_read_variant(tp, bar_2F_m=0.2)}, "bar_2F_m"),
    "s0 read's controls did not pass": (lambda tp: {"s0": s0_read_variant(tp, controls={"phase1": True, "phase2": False})}, "controls"),
    "an eval scored X_s0's checkpoint": (lambda tp: {"extra_json": {"only-speed": {"ckpt": "D:\\x\\M2b_s0_mixo\\ckpt_last.pt"}}}, "not X = M2b_s1_mixo/ckpt_last.pt"),
    "an oracle declared another group": (lambda tp: {"extra_json": {"union": {"tac_manipulation": {"mode": "oracle", "groups": "all", "n_code_dims_replaced": 60, "alpha": 1.0}}}}, "manipulation"),
    "an oracle declared alpha 0.5": (lambda tp: {"extra_json": {"only-tte": {"tac_manipulation": {**_man("tte"), "alpha": 0.5}}}}, "alpha 1"),
    "own is not T's own code": (lambda tp: {"extra_json": {"own": {"tac": "oracle"}}}, "not T's own code"),
    "the strategic family is missing": (lambda tp: {"extra_json": {"only-geom": {"raw": {"tactical": {"a": 1}}}}}, "no strategic family"),
    "the tactical family is missing": (lambda tp: {"extra_json": {"all": {"raw": {"strategic": {"a": 1}}}}}, "no tactical family"),
    "the arms are on another window list": (lambda tp: {"shift_eid": True}, "not on one window list"),
    "an eval is missing": (lambda tp: {"drop": ("only-leadgap",)}, "is missing"),
}


@pytest.mark.parametrize("name", list(REFUSALS))
def test_reader_refuses_rc2_with_the_reason_and_writes_nothing(monkeypatch, tmp_path, capsys, name):
    kw, frag = REFUSALS[name][0](tmp_path), REFUSALS[name][1]
    s0 = kw.pop("s0", None)
    w = make_world(tmp_path / "w", 1, None, None, **kw)
    write_controls_ok(w)
    R = load_reader(monkeypatch, w, 1, s0_read=s0 or s0_read_variant(tmp_path))
    with pytest.raises(SystemExit) as e:
        R.main(["--seed", "1", "--no-beside", "--out", str(tmp_path / "READ_s1.json")])
    assert e.value.code == 2 and frag in capsys.readouterr().err, name
    assert not (tmp_path / "READ_s1.json").exists(), "a refusal writes no result"


@pytest.mark.parametrize("ctl, frag", [({"phase1": {"PASS": False}, "phase2": {"PASS": True}}, "phase1 is not PASS"), ({"phase1": {"PASS": True}, "phase2": {"PASS": False}}, "phase2 is not PASS"),
                                       ({"phase1": {"PASS": True}}, "phase2 is not PASS")])
def test_reader_reads_no_gain_while_a_control_phase_is_not_pass(monkeypatch, tmp_path, capsys, ctl, frag):
    w = make_world(tmp_path / "w", 1)
    (w / "controls_s1.json").write_text(json.dumps(ctl), encoding="utf-8")
    R = load_reader(monkeypatch, w, 1, s0_read=s0_read_variant(tmp_path))
    with pytest.raises(SystemExit) as e:
        R.main(["--seed", "1", "--no-beside", "--out", str(tmp_path / "o.json")])
    assert e.value.code == 2 and frag in capsys.readouterr().err and not (tmp_path / "o.json").exists()


def test_reader_refuses_a_seed_argument_that_disagrees_with_its_configuration(monkeypatch, tmp_path, capsys):
    w = make_world(tmp_path / "w", 1)
    R = load_reader(monkeypatch, w, 1)
    with pytest.raises(SystemExit) as e:
        R.main(["--seed", "0", "--no-beside"])
    assert e.value.code == 2 and "one of them is wrong" in capsys.readouterr().err
    monkeypatch.setenv("KINDX_SEED", "2")
    with pytest.raises(SystemExit):
        _load("kindx_read_bad_seed", READER)


def test_the_s1_reader_writes_the_deciding_block_before_it_computes_the_beside_block(monkeypatch, tmp_path):
    w = make_world(tmp_path / "w", 1)
    write_controls_ok(w)
    R = load_reader(monkeypatch, w, 1, s0_read=s0_read_variant(tmp_path))
    seen = {}

    def boom(*a, **k):
        seen["on_disk"] = json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))
        raise RuntimeError("beside failed")
    R.beside_block = boom
    with pytest.raises(RuntimeError):
        R.main(["--seed", "1", "--out", str(tmp_path / "o.json")])
    assert seen["on_disk"]["OUTCOME"] == "CARRIER" and "beside" not in seen["on_disk"], "the outcome is on disk before any beside number exists"


def test_the_quoted_rule_is_the_preregs_candidate_row_verbatim(monkeypatch, tmp_path):
    if not PREREG.exists():
        pytest.skip("PREREG_V8F_KINDX.md is not in this checkout")
    R = load_reader(monkeypatch, tmp_path / "w", 1)
    prereg = PREREG.read_text(encoding="utf-8").replace("→", "->").replace("\u2192", "->")
    for piece in R.NEXT["CANDIDATE"].replace("`", "").split(". "):
        assert piece.replace("`", "").rstrip(".") in prereg.replace("`", ""), piece
    assert "2F = 0.2415 m" in prereg and "gain_k(X) ≥ 2F = 0.2415 m" in prereg.replace("\u2265", "≥")


# ---- the mutations of (A): each must turn a row above RED
def _mut_outcome(monkeypatch, tmp_path, old, new, gains, ep_sd=None, name="mut.py"):
    mp = _mutant_src(tmp_path, READER, old, new, name)
    w = make_world(tmp_path / "w", 1, gains, ep_sd)
    write_controls_ok(w)
    R = load_reader(monkeypatch, w, 1, src=mp, s0_read=s0_read_variant(tmp_path))
    return run_reader(R, tmp_path)[1]["OUTCOME"]


def test_mutation_clears_that_ignores_the_interval_turns_the_not_separated_row_red(monkeypatch, tmp_path):
    ep = {"speed": 3.0}
    assert reader_case(monkeypatch, tmp_path / "a", {"speed": 0.30}, ep_sd=ep)[1]["OUTCOME"] == "NOT A CARRIER"
    # the reader's `clears` is KIND's: the mutation is made on the function the outcome calls
    R, _, _ = reader_case(monkeypatch, tmp_path / "b", {"speed": 0.30}, ep_sd=ep)
    R.K.clears = lambda g, bar=R.BAR_2F: bool(g["point"] >= bar and g["point"] > 0)
    assert run_reader(R, tmp_path / "b")[1]["OUTCOME"] == "CARRIER", "the mutation (the interval is ignored) must change the outcome: the row is not blind"


def test_mutation_a_lower_bar_turns_the_miss_row_red(monkeypatch, tmp_path):
    assert reader_case(monkeypatch, tmp_path / "a", {"speed": 0.20})[1]["OUTCOME"] == "NOT A CARRIER"
    mp = _mutant_src(tmp_path, READER, "BAR_2F = 0.2415 ", "BAR_2F = 0.15 ", "mut_bar.py")
    w = make_world(tmp_path / "w", 1, {"speed": 0.20})
    write_controls_ok(w)
    with pytest.raises(AssertionError):                              # the reader cross-checks its bar with KIND's helpers at load: a moved bar cannot even be imported
        load_reader(monkeypatch, w, 1, src=mp, s0_read=s0_read_variant(tmp_path))


def test_mutation_a_carrier_that_need_not_have_been_an_s0_candidate_turns_the_s1_only_row_red(monkeypatch, tmp_path):
    g = {"speed": 0.05, "angle": 0.60}
    assert reader_case(monkeypatch, tmp_path / "a", g)[1]["OUTCOME"] == "NOT A CARRIER"
    out = _mut_outcome(monkeypatch, tmp_path / "b", "carriers = [k for k in cand0 if clears1[k]]", "carriers = [k for k in KINDS if clears1[k]]", g)
    assert out == "CARRIER", "the mutation (clearing at s1 alone makes a carrier) must change the outcome"


def test_mutation_swapping_the_two_outcomes_turns_the_carrier_row_red(monkeypatch, tmp_path):
    out = _mut_outcome(monkeypatch, tmp_path, '"CARRIER" if carriers else "NOT A CARRIER"', '"NOT A CARRIER" if carriers else "CARRIER"', {"speed": 0.70})
    assert out == "NOT A CARRIER"


# =============================================================================================================================================
# (B) THE s0 MODE IS UNCHANGED
# =============================================================================================================================================
def test_s0_mode_reproduces_the_banked_deciding_block_exactly(monkeypatch, tmp_path):
    """The reader was PARAMETRISED, not rewritten: on the banked s0 evals and controls it writes the banked READ.json's deciding block bit for bit (B = 2,000)."""
    if not (RAWX / "eval" / "eval_M2b-mixo_s0-kindx-own.npz").exists() or not (RAWX / "controls.json").exists():
        pytest.skip("the banked s0 evals are not in this checkout")
    (tmp_path / "controls.json").write_bytes((RAWX / "controls.json").read_bytes())
    _env(monkeypatch, tmp_path, 0, ev=RAWX / "eval")
    R = _load(f"kindx_read_s0_{next(CNT)}", READER)
    assert R.SEED == 0 and R.T == "M2b-mixo_s0" and R.B_BOOT == 2000
    rc = R.main(["--no-beside", "--out", str(tmp_path / "READ.json")])
    got, want = json.loads((tmp_path / "READ.json").read_text(encoding="utf-8")), json.loads((RAWX / "READ.json").read_text(encoding="utf-8"))
    assert rc == 0 and got["schema"] == "v8f-kindx-read-v1"
    assert {k: v for k, v in got.items()} == {k: v for k, v in want.items() if k != "beside"}, "the s0 deciding block is the banked one, field for field"


def test_s0_mode_still_names_candidate_and_no_candidate_and_never_the_s1_outcomes(monkeypatch, tmp_path):
    for gains, want in (({"speed": 0.70}, "CANDIDATE"), ({"speed": 0.10}, "NO CANDIDATE")):
        w = make_world(tmp_path / want.replace(" ", "_") / "w", 0, gains)
        (w / "controls.json").write_text(json.dumps({"phase1": {"PASS": True, "checkpoint_sha256": {"observed": "ab" * 32}}, "phase2": {"PASS": True}}), encoding="utf-8")
        R = load_reader(monkeypatch, w, 0)
        rc = R.main(["--no-beside", "--out", str(w / "READ.json")])
        res = json.loads((w / "READ.json").read_text(encoding="utf-8"))
        assert rc == 0 and res["OUTCOME"] == want and res["committed_next"] == R.NEXT[want] and "carriers" not in res


# =============================================================================================================================================
# (C) THE CONTROLS AT s1
# =============================================================================================================================================
def _controls_world(tmp_path, seed=1, sha_trained=None, sha_after=None, ckpt_bytes=b"x" * 5000, **wk):
    w = make_world(tmp_path / "w", seed, **wk)
    (w / "runs" / f"M2b_s{seed}_mixo").mkdir(parents=True, exist_ok=True)
    (w / "runs" / f"M2b_s{seed}_mixo" / "ckpt_last.pt").write_bytes(ckpt_bytes)
    real = hashlib.sha256(ckpt_bytes).hexdigest()
    (w / "ckpt_sha256.json").write_text(json.dumps({f"M2b-mixo_s{seed}_trained": sha_trained or real, f"M2b-mixo_s{seed}_after_evals": sha_after or sha_trained or real}), encoding="utf-8")
    return w


def _run_controls(C, phase):
    return C.phase1() if phase == 1 else C.phase2()


def test_controls_pass_on_a_consistent_s1_world(monkeypatch, tmp_path):
    w = _controls_world(tmp_path)
    C = load_controls(monkeypatch, w, 1)
    r1, r2 = _run_controls(C, 1), _run_controls(C, 2)
    assert r1["PASS"] and r2["PASS"], (r1, r2)
    assert r1["structure"]["manipulations"]["none"]["n_code_dims_replaced"] == 0 and r1["structure"]["manipulations"]["union"]["n_code_dims_replaced"] == 26
    assert r1["control_3_identity_oracle"]["gain_exactly_zero_per_window"] is True and r1["checkpoint_sha256"]["differs_from_X_s0"] is True
    assert r1["control_1_own_vs_banked"]["ade6_max_abs_dev_vs_eval_M2b-mixo_s1_m"] == 0.0
    assert all(v["max_abs_gain_on_fallback_windows_m"] == 0.0 for v in r2["per_arm"].values()) and r2["per_arm"]["only-speed"]["n_fallback_windows"] == N_FB


CONTROL_MUTANTS = {
    "own differs from the standard eval by 1e-3 m": (dict(std_dev=1e-3), 1, "control_1_own_vs_banked"),
    "the identity oracle gains something": (None, 1, "control_3_identity_oracle"),
    "trained and after-eval sha256 disagree": (dict(sha_after="cd" * 32), 1, "checkpoint_sha256"),
    "the sha256 on the record is not the file's": (dict(sha_trained="ef" * 32, sha_after="ef" * 32), 1, "checkpoint_sha256"),
    "an oracle arm moves a window with no v9 row": (None, 2, "per_arm"),
}


@pytest.mark.parametrize("name", list(CONTROL_MUTANTS))
def test_controls_fail_when_the_s1_world_is_not_the_preregs(monkeypatch, tmp_path, name):
    kw, phase, key = CONTROL_MUTANTS[name]
    kw = dict(kw or {})
    w = _controls_world(tmp_path, **kw)
    ev = w / "eval"
    if name == "the identity oracle gains something":
        z = dict(np.load(ev / "eval_M2b-mixo_s1-kindx-none.npz"))
        z["ade6"] = z["ade6"] - 1e-9
        np.savez(ev / "eval_M2b-mixo_s1-kindx-none.npz", **z)
    if name == "an oracle arm moves a window with no v9 row":
        z = dict(np.load(ev / "eval_M2b-mixo_s1-kindx-only-angle.npz"))
        z["ade6"] = z["ade6"].copy()
        z["ade6"][-1] += 0.5
        np.savez(ev / "eval_M2b-mixo_s1-kindx-only-angle.npz", **z)
    C = load_controls(monkeypatch, w, 1)
    r = _run_controls(C, phase)
    assert r["PASS"] is False, name
    if phase == 1:
        assert r[key]["PASS"] is False


def test_the_checkpoint_of_x_s1_must_not_be_x_s0s(monkeypatch, tmp_path):
    w = _controls_world(tmp_path)
    C = load_controls(monkeypatch, w, 1)
    monkeypatch.setattr(C, "sha256", lambda p: X0_SHA)
    (w / "ckpt_sha256.json").write_text(json.dumps({"M2b-mixo_s1_trained": X0_SHA, "M2b-mixo_s1_after_evals": X0_SHA}), encoding="utf-8")
    r = C.phase1()
    assert r["checkpoint_sha256"]["observed"] == r["checkpoint_sha256"]["expected"] == X0_SHA and r["checkpoint_sha256"]["PASS"] is False and r["PASS"] is False


def test_controls_at_s0_are_untouched(monkeypatch, tmp_path):
    C = load_controls(monkeypatch, tmp_path, 0)
    assert C.SEED == 0 and C.T == "M2b-mixo_s0" and C.XDIR == "M2b_s0_mixo" and C.expected_sha() == (X0_SHA, None)
    src = CONTROLS.read_text(encoding="utf-8")
    assert 'path = f"{OUT}/controls.json" if SEED == 0 else' in src and 'BK = os.environ.get("KINDX_BK", "C:/Users/Admin/box2_ship/logs/v8f_mixo/eval" if SEED == 0 else EV)' in src


# =============================================================================================================================================
# (D) THE PREDECESSOR PREDICATE: KSCORE's chain
# =============================================================================================================================================
PROCEED, HOLDING, WAIT = 0, 1, 2
P, H = PROCEED, HOLDING
JOB = "v8fkscore10261904_"
KRUN = "D:/tanitad/runs/v8f_kscore"
RCTAIL = f"(the read's rc 2 = a failed stop-control or an UNMAPPED state: {KRUN}/READ.json on box 2)"
WAIT_MSG = "(2 = CRASH/reboot, 3 = unreachable, 4 = timeout)"
DONE_LINE = "ZZV8FMINI-DONE 10-26 04:10 tip d3d314c7: scorer READ outcome X"
KS_STAGES = ["hyps_eval_0", "hyps_train_0", "hyps_eval_1", "hyps_train_1", "gate_s0", "gate_s0_perm", "gate_s1", "gate_s1_perm", "fit_s0", "fit_s0_perm", "fit_s1", "fit_s1_perm", "eval", "read"]
PKS = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
# EVERY text KSCORE's chain can print after ZZV8FMINI-STOP, as its source spells it, rendered with plausible values, each with the decision this chain commits for it
KS_STOPS = {
    "$1 wait=$w (2 = CRASH/reboot, 3 = unreachable, 4 = timeout)": [(f"{JOB}hyps_train_0 wait=2 {WAIT_MSG}", H), (f"{JOB}gate_s1_perm wait=3 {WAIT_MSG}", H), (f"{JOB}read wait=4 {WAIT_MSG}", H),
                                                                    (f"{JOB}refresh wait=4 {WAIT_MSG}", H)],
    "$1 rc=$rc (the read's rc 2 = a failed stop-control or an UNMAPPED state: $KRUN/READ.json on box 2)":
        [(f"{JOB}{st} rc=1 {RCTAIL}", P) for st in KS_STAGES] + [(f"{JOB}read rc=2 {RCTAIL}", P), (f"{JOB}hyps_train_1 rc=137 {RCTAIL}", P), (f"{JOB}fit_s1_perm rc=3 {RCTAIL}", P),
                                                                  (f"{JOB}refresh rc=1 {RCTAIL}", H), (f"{JOB}refresh rc=3 {RCTAIL}", H),              # the refresh job runs no model code
                                                                  (f"{JOB}hyps_train_0 rc= {RCTAIL}", H), (f"{JOB}hyps_train_0 rc=0 {RCTAIL}", H), (f"{JOB}hyps_train_0 rc=-1073741819 {RCTAIL}", H),
                                                                  (f"{JOB}hyps_train_0 rc=1", H), (f"{JOB}hyps_train_0 rc=1 (the read's rc 2 = something else)", H)],
    "no such stage $st": [("no such stage hyps_eval_9", P)],
    "ship $j": [(f"ship {JOB}hyps_eval_0", H)],
    "launch $j": [(f"launch {JOB}gate_s0", H)],
    "pull $1": [(f"pull {KRUN}/fit/s0/summary.json", H)],
    "$1: cannot hash box 2's M2b_s$s checkpoint": [("before anything runs: cannot hash box 2's M2b_s0 checkpoint", H)],
    "$1: box 2's M2b_s$s checkpoint sha256 $g != the banked run's ${CKPT_SHA[$s]}": [("before anything runs: box 2's M2b_s1 checkpoint sha256 aa != the banked run's bb", H)],
    "box 2 unreachable": [("box 2 unreachable", H)],
    "remote tip $T is not in the mirror": [("remote tip d3d314c7fd13e1c3a941720d9417d188cc2effc9 is not in the mirror", P)],
    "the tip has no $f": [(f"the tip has no {PKS}/code/v8f_dk_read.py", P), ("the tip has no stack/scripts/v8f_kscore.py", P)],
    "the tip's v8f_kscore.py has no FLAG_CHECKS (the gate would refuse every flag)": [("the tip's v8f_kscore.py has no FLAG_CHECKS (the gate would refuse every flag)", P)],
    "the tip's v8f_guard.sh does not know v8f_kscore (WIRING.patch not applied): the temperature guard could not stop our python":
        [("the tip's v8f_guard.sh does not know v8f_kscore (WIRING.patch not applied): the temperature guard could not stop our python", P)],
    "archive": [("archive", P)],
    "ship the code archive": [("ship the code archive", H)],
    "ship refresh job": [("ship refresh job", H)],
    "launch refresh": [("launch refresh", H)],
    "box 2 $f != tip $T": [("box 2 stack/scripts/v8f_kscore.py != tip d3d314c7", H)],
    "box 2 has no $R/join_train.npz (built by the first trainer run on the full corpus)": [("box 2 has no D:/tanitad/runs/v8f_mini_full/join_train.npz (built by the first trainer run on the full corpus)", H)],
    "no fresh telemetry ($t1 / $t2): the temperature guard would be blind": [("no fresh telemetry (a / b): the temperature guard would be blind", H)],
    "v8f_guard.sh does not know this chain's markers (anchored): it would never exit": [("v8f_guard.sh does not know this chain's markers (anchored): it would never exit", H)],
    "v8f_guard.sh (box2_ship) does not kill v8f_kscore: copy the tip's v8f_guard.sh over it first": [("v8f_guard.sh (box2_ship) does not kill v8f_kscore: copy the tip's v8f_guard.sh over it first", H)],
    "guard stop": [("guard stop", H)],
    "gate s$s$p assertions": [("gate s0 assertions", P), ("gate s0_perm assertions", P), ("gate s1 assertions", P), ("gate s1_perm assertions", P), ("gate s2 assertions", H), ("gate s0_permx assertions", H)],
}
# the direct `echo ZZV8FMINI-STOP <msg>` lines of KSCORE's chain (not through stop())
KS_DIRECT = {
    "the predecessor's end does not clear the box: $d": [("the predecessor's end does not clear the box: HOLD a stop that is not recognised as verdict-, gate- or run-level (fail closed): x", H)],
    "busy marker present: $(cat $B/BOX2_BUSY.txt)": [("busy marker present: v8F-mini PREREG_V8F_FLAT since 10-25 12:00", H)],
    "busy marker taken while arming: $(cat $B/BOX2_BUSY.txt 2>/dev/null)": [("busy marker taken while arming: v8F-mini PREREG_V8F_KIND", H), ("busy marker taken while arming: ", H)],
}
# texts that are NOT KSCORE's (another chain's, a stranger's, a malformed one): every one HOLDS (fail closed)
NOT_KSCORES = [
    ("v8fflat10261904_t_M2b_s0_flat rc=1", H), (f"v8fflat10261904_t_M2b_s0_flat rc=1 {RCTAIL}", H), ("v8fmixo10261904_t_M2b_s0_mixo rc=1", H), ("flat read", H), ("gate FLAT assertions", H),
    ("gate X assertions", H), ("mixo read final", H), ("rc=1", H), (f"v8fkscore1026190_hyps_train_0 rc=1 {RCTAIL}", H), (f"v8fkscore10261904_hyps_train_2 rc=1 {RCTAIL}", H),
    (f"{JOB}telemetry_long rc=1 {RCTAIL}", H), (f"{JOB}v8fkindx_x rc=1 {RCTAIL}", H), ("a stop text this predicate has never seen", H), ("", H), ("gate s0 assertions extra", H),
    ("the predecessor stopped: ZZV8FMINI-STOP hyps_train_0 rc=1", H),
]
UNKNOWN = [m for m, _ in NOT_KSCORES if m in ("a stop text this predicate has never seen", "", "gate X assertions", "mixo read final")]
STOP_TABLE = [x for d in (KS_STOPS, KS_DIRECT) for rows in d.values() for x in rows] + NOT_KSCORES
PREFLIGHT_KEYS = ["no such stage $st", "remote tip $T is not in the mirror", "the tip has no $f", "the tip's v8f_kscore.py has no FLAG_CHECKS (the gate would refuse every flag)",
                  "the tip's v8f_guard.sh does not know v8f_kscore (WIRING.patch not applied): the temperature guard could not stop our python", "archive"]
EVIDENCE = ["ZZGUARD-STOP TEMP 10-25 13:00 USB HDD: 12 of the last 20 samples >= 61 C, max 63 C", "ZZGUARD-STOP TIME 10-25 13:00 M2b_s0_mixo projects 11.2 h (> 10 h) at 2.1 s/step",
            "ZZCRASH box 2 rebooted: 2026-10-24 07:00:00 -> 2026-10-25 13:01:00", "ZZUNREACHABLE 12 min", "ZZTIMEOUT 1320 polls"]
PROCEEDING_STOPS = [f"{JOB}hyps_train_0 rc=1 {RCTAIL}", f"{JOB}read rc=2 {RCTAIL}", "gate s0_perm assertions", "archive", "the tip has no B1 lead block"]


def _classify(tmp_path, out_lines, guard_lines=None, name="c", chain=None, raw=None):
    o = tmp_path / f"{name}.out"
    o.write_bytes(raw if raw is not None else (("\n".join(out_lines) + "\n").encode("utf-8") if out_lines is not None else b""))
    args = [BASH, str(chain or CHAIN), "--classify-predecessor", o.as_posix()]
    if guard_lines is not None:
        g = tmp_path / f"{name}.guard"
        g.write_bytes(("\n".join(guard_lines) + "\n").encode("utf-8"))
        args.append(g.as_posix())
    r = subprocess.run(args, capture_output=True, text=True, timeout=60, encoding="utf-8", stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout.strip(), r.stderr


def _stop(msg):
    return ["10-25 12:00 boot0=2026-10-24 07:00:00", f"ZZV8FMINI-STOP {msg}".rstrip()]


def _row_violations(tmp_path, chain, rows, tag):
    bad = []
    for i, (msg, want) in enumerate(rows):
        rc, out, err = _classify(tmp_path, _stop(msg), name=f"{tag}{i}", chain=chain)
        if rc != want or out.split()[:1] != [("PROCEED" if want == P else "HOLD")]:
            bad.append((msg, want, rc, out, err[:200]))
    return bad


def _evidence_violations(tmp_path, chain, tag, evs=EVIDENCE):
    bad = []
    for e, ev in enumerate(evs):
        for s, stop in enumerate(PROCEEDING_STOPS):
            for where in ("chain", "guard"):
                rows = _stop(stop)
                lines, guard = (rows[:1] + [ev] + rows[1:], None) if where == "chain" else (rows, [ev])
                rc, out, _ = _classify(tmp_path, lines, guard, name=f"{tag}{e}{s}{where}", chain=chain)
                if rc != H or not out.startswith("HOLD box-level evidence"):
                    bad.append((ev, stop, where, rc, out))
    return bad


def _mutant(tmp_path, old, new, name):
    return _mutant_src(tmp_path, CHAIN, old, new, name)


@needs_bash
def test_predecessor_done_proceeds_and_no_end_marker_waits(tmp_path):
    rc, out, err = _classify(tmp_path, ["10-25 12:00 boot0=2026-10-24 07:00:00", f"10-25 12:30 {JOB}eval rc=0", DONE_LINE])
    assert rc == PROCEED and out.startswith("PROCEED the predecessor ended: ZZV8FMINI-DONE"), (rc, out, err)
    rc, out, _ = _classify(tmp_path, ["10-25 12:00 boot0=2026-10-24 07:00:00", f"10-25 12:30 {JOB}hyps_train_0 rc=0"], name="wait")
    assert rc == WAIT and out.startswith("WAIT no ZZV8FMINI-DONE"), (rc, out)
    assert _classify(tmp_path, None, name="empty")[0] == WAIT, "an empty log: the predecessor has not started"
    assert _classify(tmp_path, None, name="crlf", raw=(DONE_LINE + "\r\n").encode())[0] == PROCEED, "a CRLF log (the chain's output file) is read the same"
    rc, out, _ = _classify(tmp_path, None, name="crlfstop", raw=(f"ZZV8FMINI-STOP {JOB}hyps_train_0 rc=1 {RCTAIL}\r\n").encode())
    assert rc == PROCEED and out.endswith("on box 2)"), "a CRLF stop line is read the same (the CR is not part of the message)"
    assert _classify(tmp_path, [f"ZZV8FMINI-STOP {JOB}hyps_train_0 rc=1 {RCTAIL}", DONE_LINE], name="both")[0] == PROCEED, "a DONE anywhere in the log wins"
    assert _classify(tmp_path, ["ZZV8FMINI-STOP guard stop", f"ZZV8FMINI-STOP {JOB}hyps_train_0 rc=1 {RCTAIL}"], name="last")[0] == PROCEED, "the LAST stop line decides"
    assert _classify(tmp_path, [f"ZZV8FMINI-STOP {JOB}hyps_train_0 rc=1 {RCTAIL}", "ZZV8FMINI-STOP guard stop"], name="last2")[0] == HOLDING, "a later box-level stop holds"


@needs_bash
@pytest.mark.parametrize("msg, want", STOP_TABLE, ids=[f"{i:02d}" for i in range(len(STOP_TABLE))])
def test_predecessor_stop_is_classified_run_verdict_gate_proceed_and_everything_else_holds(tmp_path, msg, want):
    rc, out, err = _classify(tmp_path, _stop(msg))
    assert rc == want, (msg, rc, out, err)
    assert out.split()[0] == ("PROCEED" if want == PROCEED else "HOLD")


@needs_bash
def test_the_read_job_stop_is_named_verdict_level_and_the_gate_stop_gate_level(tmp_path):
    rc, out, _ = _classify(tmp_path, _stop(f"{JOB}read rc=2 {RCTAIL}"))
    assert rc == P and out.startswith("PROCEED verdict-level stop"), out
    rc, out, _ = _classify(tmp_path, _stop(f"{JOB}fit_s1 rc=1 {RCTAIL}"))
    assert rc == P and out.startswith("PROCEED run-level stop"), out
    rc, out, _ = _classify(tmp_path, _stop("gate s1_perm assertions"))
    assert rc == P and out.startswith("PROCEED gate-level stop"), out
    rc, out, _ = _classify(tmp_path, _stop("archive"))
    assert rc == P and out.startswith("PROCEED pre-flight stop"), out


def _ks_texts():
    src = KS_CHAIN.read_text(encoding="utf-8")
    code = "\n".join(x for x in src.splitlines() if not x.lstrip().startswith("#"))
    stops = set(re.findall(r'stop "([^"]*)"', code))
    direct = set(re.findall(r'echo "ZZV8FMINI-STOP ([^"]*)"', code)) - {"$*"}
    return src, code, stops, direct


def test_the_stop_table_covers_every_text_of_kscores_chain_source_so_a_new_one_cannot_slip_through_unclassified():
    """The table is checked against KSCORE's chain SOURCE (stack/scripts/chain_v8f_kscore.sh): a stop text added there later is not in the table, and this test names it. (The predicate holds on anything
    it does not recognise, so the default is the safe one, but a NEW text deserves a decision made on purpose: it may be one that should proceed, and holding it re-creates the idle box.)"""
    src, code, stops, direct = _ks_texts()
    assert sorted(stops - set(KS_STOPS)) == [] and sorted(set(KS_STOPS) - stops) == [], "KSCORE's `stop \"<msg>\"` texts and the table differ"
    assert sorted(direct - set(KS_DIRECT)) == [] and sorted(set(KS_DIRECT) - direct) == [], "KSCORE's direct ZZV8FMINI-STOP lines and the table differ"
    assert len(KS_STOPS) == 24 and len(KS_DIRECT) == 3
    proceeding = sorted(k for d in (KS_STOPS, KS_DIRECT) for k, rows in d.items() if any(w == P for _, w in rows))
    assert proceeding == sorted(["$1 rc=$rc (the read's rc 2 = a failed stop-control or an UNMAPPED state: $KRUN/READ.json on box 2)", "gate s$s$p assertions"] + PREFLIGHT_KEYS)
    assert all(w == P for k in PREFLIGHT_KEYS for _, w in KS_STOPS[k] if k != "the tip has no $f" or True)
    assert all(w == H for _, w in NOT_KSCORES), "no text outside KSCORE's chain may proceed"
    # what the regexes are built from: KSCORE's own job names, tag format, stage table, KRUN and the rc text
    assert "TAG=$(date '+%m%d%H%M')" in src and 'j="v8fkscore${TAG}_$st"' in src, "KSCORE's job name is v8fkscore<MMDDHHMM>_<stage>"
    stage_table = re.search(r"stage_table\(\) \{\n  cat <<'EOF'\n(.*?)\nEOF", src, re.S).group(1)
    assert [x.split()[0] for x in stage_table.splitlines()] == KS_STAGES, "the stages the run-level regex names are KSCORE's stage table"
    assert "KRUN=D:/tanitad/runs/v8f_kscore" in src
    assert 'stop "$1 rc=$rc (the read\'s rc 2 = a failed stop-control or an UNMAPPED state: $KRUN/READ.json on box 2)"' in code
    assert "launch v8fkscore${TAG}_refresh" in src and "waitrc v8fkscore${TAG}_refresh" in src and 'stop "gate s$s$p assertions"' in src
    assert "V8F_CHAIN_OUT=$L/chain_v8f_kscore.out" in src and "> $L/v8fkscore_guard.out" in src, "the logs this chain waits on are KSCORE's own"
    mine = CHAIN.read_text(encoding="utf-8")
    assert "AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_kscore.out}" in mine and "AFTER_GUARD=${AFTER_GUARD:-$L/v8fkscore_guard.out}" in mine
    m = re.search(r"^RUN_LEVEL_RE='\^v8fkscore\[0-9\]\{8\}_\((.*)\) rc=\[1-9\]\[0-9\]\*", mine, re.M)
    assert m and m.group(1) == "hyps_eval_[01]|hyps_train_[01]|gate_s[01](_perm)?|fit_s[01](_perm)?|eval|read", "RUN_LEVEL_RE must name exactly KSCORE's jobs"


def test_the_running_kscore_chain_is_the_one_in_the_tree():
    """The chain on box 2 runs from a tip export (run_chain_v8f_kscore.cmd). If that export's chain is not the tree's, the table above classifies the wrong texts."""
    cmd = BOX2_SHIP / "run_chain_v8f_kscore.cmd"
    if not cmd.exists():
        pytest.skip("box2_ship is not on this machine")
    m = re.search(r"cd (/c/Users/Admin/box2_ship/[^ ]+/stack/scripts)", cmd.read_text(encoding="utf-8", errors="replace"))
    assert m, "the launch command names no tip export"
    running = Path(re.sub(r"^/([a-z])/", lambda x: x.group(1).upper() + ":/", m.group(1))) / "chain_v8f_kscore.sh"      # an MSYS path -> a Windows path
    if not running.exists():
        pytest.skip("the export the running chain was launched from is gone")
    norm = lambda p: p.read_bytes().replace(b"\r\n", b"\n")                          # noqa: E731
    assert norm(running) == norm(KS_CHAIN), "the running KSCORE chain differs from the tree's: re-derive the stop table from the running one"


@needs_bash
@pytest.mark.parametrize("ev", EVIDENCE)
def test_predecessor_stop_with_box_level_evidence_in_its_logs_holds(tmp_path, ev):
    assert _evidence_violations(tmp_path, None, "ev", [ev]) == []
    rc, out, _ = _classify(tmp_path, _stop(f"{JOB}hyps_train_0 rc=1 {RCTAIL}"), ["10-25 13:00 chain ended; guard off (max USB 52 C)"], name="clean")
    assert rc == PROCEED, "a guard log that merely ended is no evidence"
    assert _classify(tmp_path, ["10-25 12:00 boot0=x", ev, DONE_LINE], name="done_wins")[0] == PROCEED, "ZZV8FMINI-DONE proceeds whatever the log says before it"


@needs_bash
def test_hold_texts_name_their_reason(tmp_path):
    rc, out, _ = _classify(tmp_path, _stop(f"{JOB}refresh rc=1 {RCTAIL}"))
    assert rc == HOLDING and out.startswith("HOLD box-level: KSCORE's code-refresh job"), out
    rc, out, _ = _classify(tmp_path, _stop("something new"))
    assert rc == HOLDING and out.startswith("HOLD a stop that is not recognised") and out.endswith("something new"), out
    rc, out, _ = _classify(tmp_path, _stop(f"{JOB}gate_s0 rc=1 {RCTAIL}"), ["ZZGUARD-STOP TEMP 10-25 13:00 x"])
    assert rc == HOLDING and f"its stop: {JOB}gate_s0 rc=1" in out, out


@needs_bash
def test_mutation_an_unknown_stop_text_that_proceeds_turns_the_unknown_rows_red(tmp_path):
    rows = [(m, H) for m in UNKNOWN]
    assert _row_violations(tmp_path, None, rows, "u0") == [], "the real predicate holds on every unknown text"
    mutant = _mutant(tmp_path, '(fail closed): $msg"; return 1', '(fail closed): $msg"; return 0', "mut_unknown_proceeds.sh")
    bad = _row_violations(tmp_path, mutant, rows, "u1")
    assert len(bad) == len(rows) and all(b[2] == PROCEED for b in bad), f"the mutation (an unrecognised stop PROCEEDs) was not detected: {bad}"


@needs_bash
def test_mutation_ignoring_the_box_level_evidence_turns_the_evidence_check_red(tmp_path):
    evs = [EVIDENCE[0], EVIDENCE[2]]
    assert _evidence_violations(tmp_path, None, "e0", evs) == []
    mutant = _mutant(tmp_path, '[ -z "$ev" ] || { echo "HOLD box-level evidence in the predecessor\'s logs: $ev (its stop: $msg)"; return 1; }', ":", "mut_no_evidence.sh")
    bad = _evidence_violations(tmp_path, mutant, "e1", evs)
    assert len(bad) == len(evs) * len(PROCEEDING_STOPS) * 2, f"the mutation (box-level evidence ignored) was not detected on every case: {len(bad)}"


@needs_bash
def test_mutation_letting_the_refresh_job_rc_through_turns_its_rows_red(tmp_path):
    rows = [(m, w) for m, w in KS_STOPS["$1 rc=$rc (the read's rc 2 = a failed stop-control or an UNMAPPED state: $KRUN/READ.json on box 2)"] if "refresh" in m]
    assert len(rows) == 2 and _row_violations(tmp_path, None, rows, "r0") == []
    mutant = _mutant(tmp_path, "_(hyps_eval_[01]|hyps_train_[01]|gate_s[01](_perm)?|fit_s[01](_perm)?|eval|read) rc=[1-9][0-9]* \\(the read.s rc 2 = a failed stop-control or an UNMAPPED state: D:/tanitad/runs/v8f_kscore/READ\\.json on box 2\\)$'\nGATE_LEVEL_RE",
                     "_(refresh|hyps_eval_[01]|hyps_train_[01]|gate_s[01](_perm)?|fit_s[01](_perm)?|eval|read) rc=[1-9][0-9]* \\(the read.s rc 2 = a failed stop-control or an UNMAPPED state: D:/tanitad/runs/v8f_kscore/READ\\.json on box 2\\)$'\nGATE_LEVEL_RE", "mut_refresh.sh")
    assert len(_row_violations(tmp_path, mutant, rows, "r1")) == len(rows), "the mutation (the refresh job's rc proceeds) was not detected"


@needs_bash
def test_mutation_widening_the_run_level_regex_to_any_job_name_turns_the_wrong_chain_rows_red(tmp_path):
    rows = [(m, w) for m, w in NOT_KSCORES if re.fullmatch(r"[A-Za-z0-9_.-]+ rc=[1-9][0-9]*( \(.*\))?", m)]
    assert len(rows) >= 5 and _row_violations(tmp_path, None, rows, "w0") == []
    src = CHAIN.read_text(encoding="utf-8")
    old = re.search(r"^RUN_LEVEL_RE='.*'$", src, re.M).group(0)
    mutant = _mutant(tmp_path, old, "RUN_LEVEL_RE='^[A-Za-z0-9_.-]+ rc=[1-9][0-9]*( \\(.*\\))?$'", "mut_anyjob.sh")
    bad = _row_violations(tmp_path, mutant, rows, "w1")
    assert len(bad) == len(rows), f"the mutation (RUN_LEVEL_RE matches any job name) was not detected on every wrong-chain row: {bad}"


@needs_bash
def test_the_wait_loop_calls_the_same_predicate_and_the_chain_stops_on_a_hold():
    raw = CHAIN.read_bytes()
    assert b"\r\n" not in raw, "a bash script with CRLF endings does not run"
    c = raw.decode("utf-8")
    assert 'd=$(after_decision "$AFTER_OUT" "$AFTER_GUARD"); r=$?' in c and 'if [ $r = 1 ]; then echo "ZZV8FMINI-STOP the predecessor\'s end does not clear the box: $d"; exit 1; fi' in c
    assert 'if [ $r = 0 ]; then echo "$(date \'+%m-%d %H:%M\') predecessor: $d"; break; fi' in c
    assert 'if [ "${1:-}" = "--classify-predecessor" ]; then after_decision' in c
    # the predicate and the classify mode come before EVERY statement that touches box state (the mkdir of $E, the ssh / scp definitions) and before the dry-run: classify has no side effect
    assert c.index("after_decision() {") < c.index('if [ "${1:-}" = "--classify-predecessor" ]') < c.index('if [ "${1:-}" = "--dry-run" ]') < c.index("mkdir -p $E/runs") < c.index("waiting for the predecessor chain")
    assert c.index("AFTER_DONE=${AFTER_DONE:-ZZV8FMINI-DONE}; AFTER_STOP=${AFTER_STOP:-ZZV8FMINI-STOP}") < c.index("after_decision() {")
    assert "sleep 300" in c and "until grep -q" not in c and "the predecessor stopped:" not in c, "the old 'any STOP stops me' loop is gone"


# =============================================================================================================================================
# (E) THE ARGV DIFF, THE GATE AND THE RUN ASSERTIONS
# =============================================================================================================================================
def _banked(name):
    return json.loads((RAWM / name).read_text(encoding="utf-8"))


S0_ARGV = RAWM / "runs" / "M2b_s0_mixo" / "argv.json"
S0_GATE = RAWM / "gate_M2b-mixo" / "selfcheck.json"
B_S0 = RAWM / "runs" / "M2b_s0" / "summary.json"


def _s1_argv(**changes):
    a = list(D.load_argv(str(S0_ARGV)))
    a[a.index("--seed") + 1] = "1"
    a[a.index("--out") + 1] = a[a.index("--out") + 1].replace("_s0_mixo", "_s1_mixo")
    for k, v in changes.items():
        a[a.index(k) + 1] = v
    return a


def test_the_banked_mixo_s0_argv_is_the_preregs_argv():
    a = D.load_argv(str(S0_ARGV))
    assert a[:6] == ["--arm", "M2b", "--seed", "0", "--grid", "pool4x10"] and a.count("--code-mix") == 1 and a[a.index("--code-mix") + 1] == "kind"
    assert "--o-interface" not in a, "MIXO's argv predates the lever; X's s1 argv is MIXO's with the seed changed and does not name it (the default is full)"


def test_seed_1_argv_is_mixos_with_the_seed_the_only_difference():
    assert D.seed_only_violations(D.load_argv(str(S0_ARGV)), _s1_argv()) == []


@pytest.mark.parametrize("name, mutate", [
    ("another batch size", lambda a: _s1_argv(**{"--batch": "32"})),
    ("another steps multiplier", lambda a: _s1_argv(**{"--steps-mult": "2"})),
    ("code-mix whole", lambda a: _s1_argv(**{"--code-mix": "whole"})),
    ("another arm", lambda a: _s1_argv(**{"--arm": "M2a"})),
    ("another grid", lambda a: _s1_argv(**{"--grid": "pool8x20"})),
    ("another exclude", lambda a: _s1_argv(**{"--exclude": "0:60"})),
    ("the seed is still 0", lambda a: _s1_argv(**{"--seed": "0"})),
    ("the seed is 2", lambda a: _s1_argv(**{"--seed": "2"})),
    ("the output directory is X_s0's", lambda a: _s1_argv(**{"--out": a[a.index("--out") + 1]})),
    ("the output directory is B's", lambda a: _s1_argv(**{"--out": "D:\\tanitad\\runs\\v8f_mini_full\\M2b_s1"})),
    ("a lever flag is added", lambda a: _s1_argv() + ["--o-interface", "cut"]),
    ("a flag is dropped", lambda a: [x for i, x in enumerate(_s1_argv()) if not (x == "--device" or (i > 0 and _s1_argv()[i - 1] == "--device"))]),
])
def test_seed_1_argv_that_differs_in_anything_else_is_refused(name, mutate):
    base = D.load_argv(str(S0_ARGV))
    assert D.seed_only_violations(base, mutate(base)) != [], name


def test_mutation_an_argv_check_that_compares_nothing_turns_the_rows_red():
    src = DRIVER.read_text(encoding="utf-8")
    old = '        bad.append(f"token {i} differs: {x!r} vs {y!r} (the argv may differ in the seed only)")'
    assert src.count(old) == 1
    M = {}
    exec(compile(src.replace(old, "        pass"), "mut_argv.py", "exec"), M)
    base = D.load_argv(str(S0_ARGV))
    assert M["seed_only_violations"](base, _s1_argv(**{"--batch": "32"})) == [], "the mutation hides a moved lever"
    assert D.seed_only_violations(base, _s1_argv(**{"--batch": "32"})) != [], "the real check does not"


@win
def test_the_tip_queues_own_trainmixo_lines_differ_in_the_seed_and_the_output_directory_only(tmp_path):
    env = dict(os.environ, V8F_DRYRUN="1")
    lines = {}
    for s in (0, 1):
        r = subprocess.run(f'cmd /c ""{QUEUE}" trainmixo {s}"', env=env, capture_output=True, text=True, timeout=60, encoding="utf-8")
        ln = [x for x in r.stdout.splitlines() if ".py" in x]
        assert r.returncode == 0 and len(ln) == 1, (r.stdout, r.stderr)
        lines[s] = ln[0]
        (tmp_path / f"l{s}.txt").write_text(ln[0], encoding="utf-8")
    a, b = D.tokens(lines[0]), D.tokens(lines[1])
    assert D.seed_only_violations(a, b) == [] and a[a.index("--seed") + 1] == "0" and b[b.index("--seed") + 1] == "1" and "--o-interface" not in b
    assert subprocess.run([sys.executable, str(DRIVER), "assert-argv", "--base", str(tmp_path / "l0.txt"), "--other", str(tmp_path / "l1.txt")], capture_output=True, text=True).returncode == 0
    # and the QUEUE's line is the argv MIXO's seed-0 run was banked with (the seed-0 form of the line, flag for flag, up to the python paths)
    banked = D.load_argv(str(S0_ARGV))
    assert a[2:] == banked, "the tip queue's `trainmixo 0` line is exactly the argv MIXO's seed-0 run recorded"


B_SUM = json.loads(B_S0.read_text(encoding="utf-8")) if B_S0.exists() else None
needs_bank = pytest.mark.skipif(B_SUM is None or not S0_GATE.exists(), reason="MIXO's banked gate / B summary are not in this checkout")


@needs_bank
def test_mixos_banked_s0_gate_passes_the_ported_assertions_without_the_flat_record_only():
    sc = _banked("gate_M2b-mixo/selfcheck.json")
    assert D.gate_violations(sc, B_SUM, require_o_interface=False) == []
    bad = D.gate_violations(sc, B_SUM)
    assert len(bad) == 2 and all("o_interface" in x for x in bad), "a selfcheck that predates FLAT has no o_interface record: the new tip's must"


def _new_tip_selfcheck():
    sc = copy.deepcopy(_banked("gate_M2b-mixo/selfcheck.json"))
    sc["levers"]["o_interface"] = "full"
    sc["o_interface_probe"] = {"o_interface": "full", "shift": 2.5, "n_windows": 64, "perturbations": {}, "pass": True}
    return sc


@needs_bank
def test_the_new_tips_gate_record_passes_and_builds_mixos_model_apart_from_o_interface_full():
    sc, base = _new_tip_selfcheck(), _banked("gate_M2b-mixo/selfcheck.json")
    assert D.gate_violations(sc, B_SUM) == [] and D.same_model_violations(sc, base) == []


GATE_MUTANTS = {
    "declared_vs_built missing": lambda sc: sc.__setitem__("declared_vs_built", "FAIL"),
    "a dead control moved": lambda sc: sc["live"].__setitem__("control_prior_grad_sum", 1e-9),
    "the code_mix probe saw no partly-GT code": lambda sc: sc["code_mix_probe"].__setitem__("n_partial", 0),
    "the code_mix probe is for whole": lambda sc: sc["code_mix_probe"].__setitem__("code_mix", "whole"),
    "the model built code_mix whole": lambda sc: sc["levers"].__setitem__("code_mix", "whole"),
    "the model built o_interface cut": lambda sc: sc["levers"].__setitem__("o_interface", "cut"),
    "the o_interface probe is absent": lambda sc: sc.pop("o_interface_probe"),
    "the o_interface probe ran for cut": lambda sc: sc["o_interface_probe"].__setitem__("o_interface", "cut"),
    "the holdout is not RA-1's": lambda sc: sc["info"].__setitem__("excluded_sha256", "00" * 32),
    "the train list is not B's": lambda sc: sc["info"].__setitem__("train_sha256", "11" * 32),
    "the p_gt schedule moved": lambda sc: sc["levers"].__setitem__("p_gt_end", 0.0),
    "a second lever moved (w_tac)": lambda sc: sc["levers"]["w"].__setitem__("w_tac", 2.0),
    "the window set is not B's": lambda sc: sc["info"].__setitem__("n_windows_all_clips", 306565),
    "the step count moved": lambda sc: sc.__setitem__("steps_total", 19000),
    "hist moved": lambda sc: sc["levers"].__setitem__("hist", 2),
    "the lever added a parameter": lambda sc: sc["levers"]["params"].__setitem__("o", sc["levers"]["params"].get("o", 0) + 1),
    "lru is not recorded": lambda sc: sc["info"].__setitem__("lru", 0),
}


@needs_bank
@pytest.mark.parametrize("name", list(GATE_MUTANTS))
def test_gate_assertions_catch_a_gate_that_is_not_mixos_x(name):
    sc = _new_tip_selfcheck()
    GATE_MUTANTS[name](sc)
    assert D.gate_violations(sc, B_SUM) != [], name


@needs_bank
def test_the_queue_line_is_checked_for_one_lever_only():
    sc = _new_tip_selfcheck()
    ok = '"py" "t.py" --arm M2b --code-mix kind --out "x" --selfcheck-only'
    assert D.gate_violations(sc, B_SUM, line=ok) == []
    for bad in ('"py" "t.py" --arm M2b --out "x"', '"py" "t.py" --arm M2b --code-mix kind --code-mix kind', '"py" "t.py" --arm M1 --code-mix kind', '"py" "t.py" --arm M2b --code-mix kind --hist 2',
                '"py" "t.py" --arm M2b --code-mix kind --w-cons 1.0', '"py" "t.py" --arm M2b --code-mix whole'):
        assert D.gate_violations(sc, B_SUM, line=bad) != [], bad


@needs_bank
def test_the_new_tip_must_build_mixos_model_the_same_levers_data_and_steps():
    base = _banked("gate_M2b-mixo/selfcheck.json")
    for name, f in {"a lever changed": lambda s: s["levers"].__setitem__("hmax", 10), "a parameter group changed": lambda s: s["levers"]["params"].__setitem__("O", 1),
                    "a lever disappeared": lambda s: s["levers"].pop("wm"), "an unknown lever appeared": lambda s: s["levers"].__setitem__("surprise", 1),
                    "o_interface is cut": lambda s: s["levers"].__setitem__("o_interface", "cut"), "the data changed": lambda s: s["info"].__setitem__("train_sha256", "22" * 32),
                    "the steps changed": lambda s: s.__setitem__("steps_total", 1)}.items():
        sc = _new_tip_selfcheck()
        f(sc)
        assert D.same_model_violations(sc, base) != [], name


def _run_dir(tmp_path, mutate=None, argv_mutate=None):
    d = tmp_path / "M2b_s1_mixo"
    d.mkdir(parents=True, exist_ok=True)
    sc = _new_tip_selfcheck()
    sc["argv_sha"] = "5a5a5a5a5a5a5a5a"
    sm = copy.deepcopy(_banked("runs/M2b_s0_mixo/summary.json"))
    sm.update({"seed": 1, "argv_sha": "5a5a5a5a5a5a5a5a"})
    ar = {"argv": _s1_argv(), "sha": "5a5a5a5a5a5a5a5a"}
    if mutate:
        mutate(sc, sm)
    if argv_mutate:
        argv_mutate(ar)
    for n, o in (("selfcheck.json", sc), ("summary.json", sm), ("argv.json", ar)):
        (d / n).write_text(json.dumps(o), encoding="utf-8")
    return d


@needs_bank
def test_the_runs_own_records_pass_assert_run_and_every_departure_is_refused(tmp_path):
    base_sc = _banked("gate_M2b-mixo/selfcheck.json")
    d = _run_dir(tmp_path / "ok")
    assert D.run_violations(str(d), B_SUM, str(S0_ARGV), 1, base_sc) == []
    cases = {
        "the summary is seed 0": lambda sc, sm: sm.__setitem__("seed", 0), "the run stopped early": lambda sc, sm: sm.__setitem__("steps", 5000), "the run did not finish": lambda sc, sm: sm.__setitem__("done", False),
        "the arm is M1": lambda sc, sm: sm.__setitem__("arm", "M1"), "the train list is not B's": lambda sc, sm: sm.__setitem__("train_sha256", "33" * 32),
        "the argv hash disagrees": lambda sc, sm: sm.__setitem__("argv_sha", "6b6b6b6b6b6b6b6b"), "the in-process gate recorded a cut": lambda sc, sm: sc["levers"].__setitem__("o_interface", "cut"),
        "the in-process probe is missing": lambda sc, sm: sc.pop("o_interface_probe"), "the model differs from MIXO's": lambda sc, sm: sc["levers"].__setitem__("d", 512)}
    for name, f in cases.items():
        assert D.run_violations(str(_run_dir(tmp_path / f"m{len(name)}{abs(hash(name)) % 997}", f)), B_SUM, str(S0_ARGV), 1, base_sc) != [], name
    bad_argv = _run_dir(tmp_path / "ba", argv_mutate=lambda ar: ar["argv"].__setitem__(ar["argv"].index("--batch") + 1, "32"))
    assert any("argv" in x for x in D.run_violations(str(bad_argv), B_SUM, str(S0_ARGV), 1, base_sc))
    assert D.run_violations(str(tmp_path / "nowhere"), B_SUM, str(S0_ARGV), 1, base_sc)[0].startswith("cannot read the run's records")


@needs_bank
def test_assert_gate_cli_exits_0_on_a_pass_and_2_on_a_refusal(tmp_path):
    sc = _new_tip_selfcheck()
    p = tmp_path / "sc.json"
    p.write_text(json.dumps(sc), encoding="utf-8")
    cmd = [sys.executable, str(DRIVER), "assert-gate", "--selfcheck", str(p), "--b-summary", str(B_S0), "--base-selfcheck", str(S0_GATE)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0 and r.stdout.startswith("X gate: PASS") and "built o_interface full" in r.stdout, (r.stdout, r.stderr)
    sc["levers"]["code_mix"] = "whole"
    p.write_text(json.dumps(sc), encoding="utf-8")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 2 and "built code_mix 'whole'" in r.stdout


@needs_bank
def test_mutation_a_gate_that_drops_the_partial_code_check_turns_its_row_red():
    src = DRIVER.read_text(encoding="utf-8")
    old = 'if not (isinstance(pr.get("n_partial"), int) and pr["n_partial"] > 0):'
    assert src.count(old) == 1
    M = {}
    exec(compile(src.replace(old, "if False:"), "mut_gate.py", "exec"), M)
    sc = _new_tip_selfcheck()
    GATE_MUTANTS["the code_mix probe saw no partly-GT code"](sc)
    assert M["gate_violations"](sc, B_SUM) == [], "the mutation hides a mix that never reached the code O reads"
    assert D.gate_violations(sc, B_SUM) != []


def test_write_token_records_the_commit_the_queue_and_the_selfcheck_hash(tmp_path):
    if not S0_GATE.exists():
        pytest.skip("MIXO's banked gate is not in this checkout")
    out = tmp_path / "t.json"
    tok = D.make_token(str(S0_GATE), str(out), "d3d314c7", "ab" * 32, {"stack/x.py": "cd" * 32})
    j = json.loads(out.read_text(encoding="utf-8"))
    assert j == tok and j["gate"] == "PASS" and j["commit"] == "d3d314c7" and j["selfcheck_sha256"] == hashlib.sha256(S0_GATE.read_bytes()).hexdigest() and j["argv_sha"] == _banked("gate_M2b-mixo/selfcheck.json")["argv_sha"]


# =============================================================================================================================================
# (F) THE BOX-2 DRIVER
# =============================================================================================================================================
BOXP = D.paths_from("D:\\tanitad\\runs\\v8f_mini_full", "D:\\tanitad\\cache_full", "D:\\tanitad\\data\\v9", "D:\\tanitad\\runs\\v8f_mini_full\\p10\\p10_prior.pt",
                    "D:\\tanitad\\cache_full\\native_tracks.pt", "D:\\tanitad\\runs\\v8f_mini_full\\eval")
CK1 = "D:\\tanitad\\runs\\v8f_mini_full\\M2b_s1_mixo\\ckpt_last.pt"


def _flagmap(argv):
    """argv -> {flag: value or True}: order-free (the queue puts --tac before the eval context, kindx_eval after it)."""
    out, i = {}, 0
    while i < len(argv):
        t = argv[i]
        if t.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[t] = argv[i + 1]
            i += 2
        else:
            out[t] = True
            i += 1
    return out


def test_the_ten_kindx_evals_are_named_and_ordered_as_the_prereg_lists_them():
    assert D.EVAL_NAMES == ("own", "none", "all", "union", "speed", "angle", "tte", "timegap", "geom", "leadgap") and len(D.EVAL_NAMES) == 10
    assert [D.tag_of(n, 1) for n in D.EVAL_NAMES] == [f"M2b-mixo_s1-kindx-{x}" for x in ("own", "none", "all", "union", *[f"only-{k}" for k in KINDS])]
    assert D.tag_of("speed", 0) == "M2b-mixo_s0-kindx-only-speed", "the tags are the s0 tags with the seed changed (the reader's names)"


@win
def test_the_kindx_eval_flags_are_kinds_own_queue_lines_with_x_s_checkpoint_and_tag(tmp_path):
    """KIND's `kind` stage prints `kindown` / `kindorc` for M2b: the same lines for X differ only in the checkpoint and the tag. `none` is `all` with an empty selection."""
    env = dict(os.environ, V8F_DRYRUN="1", REPO="D:\\tanitad\\repo", PY="D:\\tanitad\\venv\\Scripts\\python.exe", CACHEFULL="D:\\tanitad\\cache_full", V9="D:\\tanitad\\data\\v9",
               RUNSM="D:\\tanitad\\runs\\v8f_mini_full")
    r = subprocess.run(f'cmd /c ""{QUEUE}" kind"', env=env, capture_output=True, text=True, timeout=60, encoding="utf-8")
    lines = [x for x in r.stdout.splitlines() if "v8f_mini_eval.py" in x]
    assert r.returncode == 0 and len(lines) == 30, (r.returncode, len(lines))
    assert re.search(r'add_argument\("--tac", choices=\("pred", "shuffle", "oracle", "absmask"\), default="pred"', (REPO / "stack" / "scripts" / "v8f_mini_eval.py").read_text(encoding="utf-8"))
    by_tag = {}
    for ln in lines:
        t = D.tokens(ln)
        by_tag[_flagmap(t)["--tag"]] = _flagmap(t[2:])
    for name in D.EVAL_NAMES:
        kname = {"own": "kind-own", "all": "kind-all", "union": "kind-union", "none": "kind-all"}.get(name, f"kind-only-{name}")
        q = dict(by_tag[f"M2b_s1-{kname}"])
        q.setdefault("--tac", "pred")                                                   # the queue's own line omits it: v8f_mini_eval's default (asserted below)
        mine = _flagmap(D.eval_argv(name, CK1, D.tag_of(name, 1), BOXP))
        for k in ("--ckpt", "--tag"):
            q.pop(k)
            mine.pop(k)
        if name == "none":
            assert mine.pop("--tac-groups") == "none" and q.pop("--tac-groups") == "all"
        assert mine == q, (name, mine, q)


def test_the_ckpt_of_every_kindx_eval_is_x_s1s_and_never_bs():
    for n in D.EVAL_NAMES:
        a = D.eval_argv(n, CK1, D.tag_of(n, 1), BOXP)
        assert a[a.index("--ckpt") + 1].endswith("M2b_s1_mixo\\ckpt_last.pt") and "M2b_s1\\" not in a[a.index("--ckpt") + 1]


def test_the_identity_patch_selects_nothing_and_leaves_every_other_spec_alone():
    from tanitad.models import v8f_mini as M
    orig = M.code_group_index
    try:
        D.install_none(M)
        assert M.code_group_index("none") == [] and M.code_group_index("constraints") == orig("constraints") and len(M.code_group_index("all")) == 60
        assert len(M.code_group_index("speed")) == 5
    finally:
        M.code_group_index = orig


class _FakeE:
    """Stands in for v8f_mini_eval: records each run's argv and the code indices its --tac-groups selects, and writes the two files an eval writes."""

    def __init__(self, M, eval_dir, fail_on=None):
        self.M, self.eval_dir, self.calls, self.fail_on = M, Path(eval_dir), [], fail_on

    def main(self, argv):
        f = _flagmap(argv)
        sel = self.M.code_group_index(f["--tac-groups"]) if "--tac-groups" in f else None
        self.calls.append((f["--tag"], sel))
        if self.fail_on == f["--tag"]:
            return 1
        (self.eval_dir / f"eval_{f['--tag']}.npz").write_bytes(b"x")
        (self.eval_dir / f"eval_{f['--tag']}.json").write_text("{}", encoding="utf-8")
        return 0


def _driver_ns(tmp_path, sha):
    ns = D.build_parser().parse_args(["evals", "--seed", "1", "--expect-sha", sha, "--run-dir", str(tmp_path / "run"), "--runs-dir", str(tmp_path), "--eval-dir", str(tmp_path / "ev"),
                                      "--cache-dir", str(tmp_path / "c"), "--v9-dir", str(tmp_path / "v9"), "--p10", str(tmp_path / "p10.pt"), "--tracks", str(tmp_path / "t.pt")])
    (tmp_path / "run").mkdir(exist_ok=True)
    (tmp_path / "ev").mkdir(exist_ok=True)
    return ns


def test_the_driver_runs_the_ten_evals_in_order_restores_the_patch_skips_what_exists_and_checks_the_sha(tmp_path):
    from tanitad.models import v8f_mini as M
    ck = tmp_path / "run" / "ckpt_last.pt"
    (tmp_path / "run").mkdir()
    ck.write_bytes(b"fake checkpoint " * 100)
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    ns = _driver_ns(tmp_path, sha)
    orig = M.code_group_index
    E = _FakeE(M, tmp_path / "ev")
    log = []
    assert D.run_evals(ns, M, E, log.append) == 0 and M.code_group_index is orig, "the patch is removed afterwards"
    assert [t for t, _ in E.calls] == [D.tag_of(n, 1) for n in D.EVAL_NAMES]
    sel = dict(E.calls)
    assert sel[D.tag_of("none", 1)] == [] and len(sel[D.tag_of("all", 1)]) == 60 and len(sel[D.tag_of("union", 1)]) == 26 and sel[D.tag_of("own", 1)] is None
    assert [len(sel[D.tag_of(k, 1)]) for k in KINDS] == [5, 3, 5, 2, 9, 2], "the single-kind arms select KIND's cells, not the identity patch's empty set"
    E2 = _FakeE(M, tmp_path / "ev")
    D.run_evals(ns, M, E2, log.append)
    assert E2.calls == [] and any(x.startswith("[skip]") for x in log), "a finished eval is skipped (resume)"
    (tmp_path / "ev" / f"eval_{D.tag_of('geom', 1)}.json").unlink()
    E3 = _FakeE(M, tmp_path / "ev")
    D.run_evals(ns, M, E3, log.append)
    assert [t for t, _ in E3.calls] == [D.tag_of("geom", 1)], "only the eval whose npz AND json are not both there is re-run"
    with pytest.raises(SystemExit, match="checkpoint sha256"):
        D.run_evals(_driver_ns(tmp_path, "0" * 64), M, _FakeE(M, tmp_path / "ev"), log.append)
    bad = tmp_path / "ev2"
    bad.mkdir()
    ns2 = _driver_ns(tmp_path, sha)
    ns2.eval_dir = str(bad)
    with pytest.raises(SystemExit, match="FAIL: eval"):
        D.run_evals(ns2, M, _FakeE(M, bad, fail_on=D.tag_of("union", 1)), log.append)
    assert M.code_group_index is orig, "the patch is removed even when an eval fails"


def test_the_driver_refuses_a_missing_checkpoint_and_one_that_changes_during_the_evals(tmp_path):
    from tanitad.models import v8f_mini as M
    ns = _driver_ns(tmp_path, "0" * 64)
    with pytest.raises(SystemExit, match="does not exist"):
        D.run_evals(ns, M, _FakeE(M, tmp_path / "ev"), lambda *_: None)
    ck = tmp_path / "run" / "ckpt_last.pt"
    ck.write_bytes(b"a" * 100)
    sha = hashlib.sha256(b"a" * 100).hexdigest()

    class Tamper(_FakeE):
        def main(self, argv):
            ck.write_bytes(b"b" * 100)
            return super().main(argv)
    with pytest.raises(SystemExit, match="changed during the evals"):
        D.run_evals(_driver_ns(tmp_path, sha), M, Tamper(M, tmp_path / "ev"), lambda *_: None)


def test_mutation_an_identity_patch_that_selects_everything_turns_the_none_row_red():
    from tanitad.models import v8f_mini as M
    src = DRIVER.read_text(encoding="utf-8")
    old = 'M.code_group_index = lambda spec: [] if spec == "none" else orig(spec)'
    assert src.count(old) == 1
    mut = {}
    exec(compile(src.replace(old, 'M.code_group_index = lambda spec: list(range(60)) if spec == "none" else orig(spec)'), "mut_none.py", "exec"), mut)
    orig = M.code_group_index
    try:
        mut["install_none"](M)
        assert M.code_group_index("none") != [], "the mutant's `none` selects cells: it is not the identity oracle"
    finally:
        M.code_group_index = orig
    try:
        D.install_none(M)
        assert M.code_group_index("none") == []
    finally:
        M.code_group_index = orig


def test_the_job_line_the_dry_run_prints_is_the_line_the_job_embeds():
    args = ["evals", "--seed", "1", "--expect-sha", "ab" * 32]
    line = D.box2_cmdline(args)
    job = D.mkjob_text("v8fkindx10261904_k_M2b_s1_mixo", args)
    assert "\r\n" in job and job.count("\n") == job.count("\r\n") and ("%RUNSAFE% " + line) in job and "rc=!RC!" in job and "ZZJOB-END" in job
    assert line.startswith('"D:\\tanitad\\venv\\Scripts\\python.exe" "D:\\tanitad\\repo\\stack\\scripts\\v8f_mini_eval_kindx.py" evals --seed 1 --expect-sha ' + "ab" * 32)
    assert '--run-dir "D:\\tanitad\\runs\\v8f_mini_full\\M2b_s1_mixo"' in line and '--eval-dir "D:\\tanitad\\runs\\v8f_mini_full\\eval"' in line and "--device" not in line
    assert 'set "PYTHONPATH=%REPO%\\stack;%REPO%\\taniteval"' in job and "call D:\\tanitad\\env.cmd" in job, "the queue's PYTHONPATH and the SAFE MODE env"
    for bad in (["evals", "--seed", "1"], ["gate"], []):
        with pytest.raises(SystemExit):
            D.box2_cmdline(bad)
    out = subprocess.run([sys.executable, str(DRIVER), "mkjob", "j1", "evals", "--seed", "1", "--expect-sha", "ab" * 32], capture_output=True).stdout
    assert out.decode("utf-8") == D.mkjob_text("j1", args), "the CLI writes the CRLF job bytes unchanged"
    toks = D.tokens(line)[2:]
    parsed = D.build_parser().parse_args(toks)
    assert parsed.seed == 1 and parsed.run_dir.endswith("M2b_s1_mixo") and parsed.tracks.endswith("native_tracks.pt")


@win
def test_the_drivers_eval_context_is_the_queues_eval_context():
    """EC / ER of the queue at its defaults == the paths the job hands the driver (v9 labels, lead, speed sidecar, p10, tracks, cache, out dir)."""
    q = QUEUE.read_text(encoding="utf-8").replace("\r\n", "\n")
    for frag in ('set "EC=--v9-eval %V9%\\v9_labels_eval139.npz --lead-eval %V9%\\lead_eval139.npz --speed-eval %V9%\\refcv6_speed_max_v8_eval.jsonl --kin-mode yaw --p10 %P10% --out-dir %EV%"',
                 'set "ER=--eval-cache %CACHEFULL% --tracks %TRACKS%"', 'set "P10=%RUNSM%\\p10\\p10_prior.pt"', 'set "EV=%RUNSM%\\eval"', 'set "TRACKS=%CACHEFULL%\\native_tracks.pt"',
                 'if not defined RUNSM set "RUNSM=D:\\tanitad\\runs\\v8f_mini_full"', 'if not defined CACHEFULL set "CACHEFULL=D:\\tanitad\\cache_full"', 'if not defined V9 set "V9=D:\\tanitad\\data\\v9"',
                 'if not defined REPO set "REPO=D:\\tanitad\\repo"', 'if not defined PY set "PY=D:\\tanitad\\venv\\Scripts\\python.exe"', 'set "PYTHONPATH=%REPO%\\stack;%REPO%\\taniteval"'):
        assert frag in q, frag
    assert D.BOX2 == {"REPO": "D:\\tanitad\\repo", "PY": "D:\\tanitad\\venv\\Scripts\\python.exe", "RUNSM": "D:\\tanitad\\runs\\v8f_mini_full", "CACHE": "D:\\tanitad\\cache_full", "V9": "D:\\tanitad\\data\\v9"}


def test_the_guard_that_is_already_running_matches_the_driver_without_a_change():
    """v8f_guard.sh stops python.exe whose CommandLine -match 'v8f_mini_train|v8f_mini_eval|...'. The driver's command line must match it, or a thermal stop would leave the evals running."""
    g = GUARD.read_text(encoding="utf-8")
    m = re.search(r"CommandLine -match '([^']*)'", g)
    assert m, "the guard's kill pattern was not found"
    assert re.search(m.group(1), D.box2_cmdline(["evals", "--seed", "1", "--expect-sha", "ab" * 32])), "the driver's command line is not matched by the guard's kill pattern"
    assert re.search(m.group(1), "python.exe v8f_mini_train.py --arm M2b") and not re.search(m.group(1), "python.exe some_other_script.py")


# =============================================================================================================================================
# (G) THE CHAIN
# =============================================================================================================================================
@needs_bash
def test_chain_syntax_and_text_have_the_load_bearing_lines_in_the_right_order():
    r = subprocess.run([BASH, "-n", str(CHAIN)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    raw = CHAIN.read_bytes()
    assert b"\r\n" not in raw
    c = raw.decode("utf-8")
    for s in ("( set -C; echo", "BOX2_BUSY.txt", "ZZV8FMINI-STOP", "ZZV8FMINI-DONE", "chain_v8f_kindx_s1.out", "v8f_guard.sh", "assert-gate", "assert-run", "assert-argv", "write-token", "TIP_V8FKINDX.txt",
              "ckptsha_b", "waitjob.sh", "launch.ps1", 'stop "gate X assertions"', 'stop "run assertions"', 'stop "argv assertion"', 'stop "kindx check-inputs"', 'stop "kindx controls 1"',
              'stop "kindx controls 2"', 'stop "kindx read s1"', "KINDX_SEED=1", "--check-inputs", 'queue "t_$g" $TRAIN_POLLS trainmixo 1', 'queue "e_$g" $EVAL_POLLS evalmixo 1',
              'kjob "k_$g" $KINDX_POLLS evals --seed 1 --expect-sha "$XSHA"', "stack/scripts/v8f_mini_eval_kindx.py"):
        assert s in c, s
    code = "\n".join(x for x in c.splitlines() if not x.lstrip().startswith("#"))
    assert "rm -rf" not in code and "--allow-" not in c, "the chain deletes nothing (the one `rm -rf` in its text is a comment saying it never uses it)"
    assert "trainmixo 0" in c and 'queue "t_$g" $TRAIN_POLLS trainmixo 0' not in c, "the chain never trains seed 0 (it prints the line to diff against)"
    assert "trainmixo 1" in c and "--o-interface" in c, "the tip's o_interface wiring is pre-flighted"
    order = ["after_decision() {", "waiting for the predecessor chain", "[ -f $B/BOX2_BUSY.txt ]", "git -C $MIRROR fetch", 'waitrc v8fkindx${TAG}_refresh 40', "ckptsha_b \"before the gate\"", 'stop "argv assertion"',
             'stop "kindx check-inputs"', "launch telemetry_long", "queue gatemixo 120 gatemixo", 'stop "gate X assertions"', 'stop "write the gate token"', 'queue "t_$g" $TRAIN_POLLS trainmixo 1',
             "ckptsha_x trained", 'stop "run assertions"', 'queue "e_$g" $EVAL_POLLS evalmixo 1', 'kjob "k_$g"', "ckptsha_x after_evals", 'stop "kindx controls 1"', 'stop "kindx controls 2"',
             'stop "kindx read s1"', 'finish "$(jget READ_s1.json reading)']
    pos = [c.index(x) for x in order]
    assert pos == sorted(pos), [o for o, p, q in zip(order, pos, pos[1:] + [10**9]) if p > q]
    assert c.count("trainmixo 1") >= 1 and c.index("queue gatemixo") < c.index('queue "t_$g"'), "no training before the launch gate"
    assert c.index('stop "argv assertion"') < c.index("queue gatemixo"), "the argv diff is asserted before anything runs on box 2"
    assert c.index('stop "run assertions"') < c.index('queue "e_$g"'), "the run's records are asserted before the evals are spent"
    # the pull of the large checkpoint comes after the evals read it on box 2, and the reader never runs before the evals are pulled
    assert c.index("ckptsha_x after_evals") < c.index('pull "$g/ckpt_last.pt"') < c.index('stop "kindx controls 1"')
    assert "X0_SHA" in c and "no second seed was trained" in c


@needs_bash
def test_chain_dry_run_prints_the_stages_the_lines_and_the_argv_diff_and_writes_nothing_to_box2_ship(tmp_path):
    if not QUEUE.exists():
        pytest.skip("the queue is not in this checkout")
    before = sorted((p.name, p.stat().st_mtime) for p in BOX2_SHIP.glob("*")) if BOX2_SHIP.exists() else None
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    env = dict(os.environ, V8F_QUEUE=str(QUEUE), KINDX_PYX=sys.executable, TMPDIR=tmp.as_posix(), KINDX_DRIVER=str(DRIVER))
    r = subprocess.run([BASH, str(CHAIN), "--dry-run"], env=env, capture_output=True, text=True, timeout=240, stdin=subprocess.DEVNULL, encoding="utf-8")
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    out = r.stdout.splitlines()
    heads = [x for x in out if x.startswith("== ")]
    assert [h.split(" (")[0] for h in heads] == ["== predecessor", "== checkpoint sha256", "== gatemixo", "== trainmixo 0", "== trainmixo 1", "== evalmixo 1", "== argv diff", "== kindx evals", "== kindx evals",
                                                   "== read", "== stages"], heads
    assert "wait for <logs>/chain_v8f_kscore.out: ^ZZV8FMINI-DONE" in r.stdout and f"M2b_s0 D:/tanitad/runs/v8f_mini_full/M2b_s0/ckpt_last.pt 601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8" in r.stdout
    assert "argv diff: exactly the seed (and the output directory that carries it)" in r.stdout
    g = next(x for x in out if "--selfcheck-only" in x)
    assert g.count("--code-mix") == 1 and "--seed" not in g and "--o-interface" not in g
    t1 = next(x for x in out[out.index("== trainmixo 1"):] if "v8f_mini_train.py" in x)
    assert "--seed 1" in t1 and t1.count("--code-mix kind") == 1 and "M2b_s1_mixo" in t1 and "--o-interface" not in t1
    assert sum(1 for x in out if re.search(r'v8f_mini_eval\.py"? run ', x) and "--tag M2b-mixo_s1" in x and "kindx" not in x) == 7, "MIXO's seven evals"
    assert sum(1 for x in out if x.startswith("v8f_mini_eval.py run") and "-kindx-" in x) == 10, "the ten KINDX evals"
    jobline = next(x for x in out if "v8f_mini_eval_kindx.py" in x and "evals --seed 1" in x)
    assert jobline == D.box2_cmdline(["evals", "--seed", "1", "--expect-sha", "X_s1_sha256"]), "the printed job line IS the line the job embeds"
    assert out[-1].startswith("wait(KSCORE) refresh(tip) ckpt-sha(B)") and "kindx_read.py --seed 1" in r.stdout
    if before is not None:
        assert sorted((p.name, p.stat().st_mtime) for p in BOX2_SHIP.glob("*")) == before, "the dry run wrote into box2_ship"


def test_chain_pins_agree_with_the_reader_the_manifest_the_prereg_and_the_mixo_chain():
    c = CHAIN.read_text(encoding="utf-8")
    shas = re.search(r"^CKPT_SHA=\((\S+) (\S+)\)$", c, re.M).groups()
    m = MIXO_CHAIN.read_text(encoding="utf-8")
    assert shas == tuple(re.search(r"^CKPT_SHA=\((\S+) (\S+)\)$", m, re.M).groups()), "the same two banked checkpoints as MIXO's chain"
    if MANIFEST.exists():
        mt = MANIFEST.read_text(encoding="utf-8")
        assert all(s in mt for s in shas), "raw/full/MANIFEST.md names both"
    assert re.search(r"^X0_SHA=(\S+)", c, re.M).group(1) == X0_SHA == json.loads((RAWX / "READ.json").read_text(encoding="utf-8"))["checkpoint_sha256"]
    for key in ("HOLDOUT_SHA", "M2B_STEPS", "M2B_WINDOWS"):
        assert re.search(rf"^{key}=(\S+)", c, re.M).group(1) == re.search(rf"^{key}=(\S+)", m, re.M).group(1), key
    assert D.HOLDOUT_SHA == re.search(r"^HOLDOUT_SHA=(\S+)", m, re.M).group(1)
    assert "TRAIN_POLLS=1320" in c and "TRAIN_POLLS=1320" in m
    assert re.search(r"^KINDX_NAMES=\"([^\"]*)\"", c, re.M).group(1).split() == [("only-" + n if n in KINDS else n) for n in D.EVAL_NAMES]
    assert re.search(r'^X_SUFFIXES="([^"]*)"', c, re.M).group(1) == re.search(r'^X_SUFFIXES="([^"]*)"', m, re.M).group(1), "the seven evals pulled are MIXO's seven"
    r_text = READER.read_text(encoding="utf-8")
    assert shas[0] == "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8" and "X0_CKPT_SHA = \"" + X0_SHA + "\"" in r_text and "X0_CKPT_SHA = \"" + X0_SHA + "\"" in CONTROLS.read_text(encoding="utf-8")
    assert re.search(r"BAR_2F = 0\.2415 ", r_text)


# =============================================================================================================================================
# (H) REACHABILITY ON THE NEW TIP: the real trainer, synthetic data
# =============================================================================================================================================
@pytest.fixture(scope="module")
def trainer_runs(tmp_path_factory):
    import torch  # noqa: F401
    sys.path.insert(0, str(REPO / "stack" / "scripts"))
    TR = _load("v8f_mini_train_for_kindx", REPO / "stack" / "scripts" / "v8f_mini_train.py")
    base = tmp_path_factory.mktemp("kindx_trainer")
    common = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--log-every", "1", "--warmup", "2", "--ckpt-every", "100", "--threads", "1", "--kin-mode", "yaw", "--arm", "M2b", "--steps", "4",
              "--code-mix", "kind"]
    out = {}
    for s in (0, 1):
        assert TR.main(["--out", str(base / f"s{s}"), *common, "--seed", str(s)]) == 0
        out[s] = {"sc": json.loads((base / f"s{s}" / "selfcheck.json").read_text(encoding="utf-8")), "argv": json.loads((base / f"s{s}" / "argv.json").read_text(encoding="utf-8")),
                  "sm": json.loads((base / f"s{s}" / "summary.json").read_text(encoding="utf-8"))}
    return {"runs": out, "TR": TR}


def test_mixos_argv_with_seed_1_passes_the_trainers_gate_on_the_new_tip_without_naming_o_interface(trainer_runs):
    for s in (0, 1):
        r = trainer_runs["runs"][s]
        assert "--o-interface" not in r["argv"]["argv"], "the lever's default (full) is what X is: the gate does not require the flag to be named"
        assert r["sc"]["declared_vs_built"] == "PASS" and r["sc"]["levers"]["o_interface"] == "full" and r["sc"]["levers"]["code_mix"] == "kind"
        assert r["sc"]["o_interface_probe"]["pass"] is True and r["sc"]["o_interface_probe"]["o_interface"] == "full"
        assert r["sc"]["code_mix_probe"]["pass"] is True and r["sc"]["code_mix_probe"]["n_partial"] > 0, "the mix reaches the code O reads"
        assert r["sm"]["done"] is True and r["sm"]["seed"] == s
    ar0, ar1 = trainer_runs["runs"][0]["argv"]["argv"], trainer_runs["runs"][1]["argv"]["argv"]
    diff = [(x, y) for x, y in zip(ar0, ar1) if x != y]
    assert [d for d in diff if not d[0].replace("\\", "/").endswith(("/s0", "s0"))] == [("0", "1")], diff


def test_seed_1_builds_the_same_model_and_data_as_seed_0_apart_from_the_seed(trainer_runs):
    r0, r1 = trainer_runs["runs"][0]["sc"], trainer_runs["runs"][1]["sc"]
    assert r0["levers"] == r1["levers"], "the built levers (every lever, every parameter group) are seed-independent"
    strip = lambda i: {k: v for k, v in i.items() if k not in ("lru", "argv_sha")}                      # noqa: E731
    assert strip(r0["info"]) == strip(r1["info"]) and r0["steps_total"] == r1["steps_total"] and r0["argv_sha"] != r1["argv_sha"]
    assert D.same_model_violations(r1, r0) == []


def test_the_trainers_own_gate_refuses_a_declared_cut_that_was_not_built_and_the_new_tip_names_its_flag(trainer_runs):
    TR = trainer_runs["TR"]
    assert TR.FLAG_CHECKS["o_interface"] == "lever:o_interface" and TR.FLAG_CHECKS["code_mix"] == "lever:code_mix"
    a = TR.build_parser().parse_args(["--out", "x", "--arm", "M2b", "--code-mix", "kind"])
    assert a.o_interface == "full" and a.code_mix == "kind" and a.seed == 0
    assert TR.build_parser().parse_args(["--out", "x", "--arm", "M2b", "--o-interface", "full"]).o_interface == "full"
