"""run_v1_fix1: three instrument fixes for W3's navtest driver, each proven on synthetic tmp dirs with NO
devkit, each with a RED mutation arm and a CONTROL showing the unfixed ``run_v1.py`` really has the defect.

    python -m pytest -q tests/test_run_v1_fix1.py                    # py3.9 (NAVSIM venv) or py3.13 (tanitad venv)
    FIX1_MUTANT=A python -m pytest -q tests/test_run_v1_fix1.py      # the SAME tests against a mutant of
    FIX1_MUTANT=B ...  /  FIX1_MUTANT=C ...                          # run_v1_fix1.py: that fix's tests go RED

The devkit is replaced by a fake ``run_wrapped`` that writes what the real wrapper writes (log lines the
harness greps, the streamed rows.jsonl, a devkit-format csv in ``<EXP>/runs/<label>/``). Everything else is
W3's own ``cmd_score``, unmodified, exercised end to end.

MEASURED origin (2026-10-18, RL-1 V1 navtest): (A) ``os.path.relpath`` across drives crashed the run after the
devkit finished; (B) a 4,433-row reconstructed partial csv sat in the run dir and the glob fallback re-copied it;
(C) a partial run's average (88.44, 88.47) was printed and read before the full read.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
import sys
import types
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
CODE = PKG / "code"
FIX_SRC = CODE / "run_v1_fix1.py"
ORIG = CODE / "run_v1.py"
PIN = "125670597d99b243eaf68bf77973753b314e41f3ed060d0073ad627da601914e"      # literal, on purpose
N = 6
TOKS = [f"tok{i}" for i in range(N)]
_uniq = itertools.count()


def _import(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


REAL = _import(FIX_SRC, "run_v1_fix1_under_test")

# (exact source line of run_v1_fix1.py, what that line becomes when the fix is REVERTED)
MUTATIONS = {
    "A": ('            return os.path.abspath(path).replace("\\\\", "/")', "            raise"),
    "B": ("        quarantined = quarantine_stale_csvs(run_out, out_dir, a.label, stamp)",
          "        quarantined = []"),
    "C": ("                rep.pop(k, None)", "                pass"),
}


def build_mutant(which, dirpath):
    src = FIX_SRC.read_text(encoding="utf-8")
    old, new = MUTATIONS[which]
    assert src.count(old) == 1, f"mutation {which}: the fix line is not present exactly once -- a mutant " \
                                f"that does not apply is not a RED arm"
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    p = Path(dirpath) / f"run_v1_fix1_mut{which}.py"
    with open(str(p), "w", encoding="utf-8", newline="\n") as fh:      # py3.9: write_text has no newline=
        fh.write(src.replace(old, new))
    return _import(p, f"run_v1_fix1_mut{which}")


@pytest.fixture(scope="session")
def FX(tmp_path_factory):
    which = os.environ.get("FIX1_MUTANT", "")
    return build_mutant(which, tmp_path_factory.mktemp("mut")) if which else REAL


# --------------------------------------------------------------------------- #
class Rig:
    """One module (patched via ``fx.load`` or the raw original) pointed at tmp dirs, with a fake devkit."""

    def __init__(self, tmp, mod):
        self.tmp = Path(tmp)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.m = mod
        self.exp, self.pkg = self.tmp / "exp", self.tmp / "pkg"
        self.raw = self.pkg / "raw"
        mod.EXP, mod.RAW, mod.PKG = self.exp.as_posix(), str(self.raw), str(self.pkg)
        self.tokfile = self.tmp / "tokens.json"
        self.tokfile.write_text(json.dumps({"tokens": TOKS, "token_log": {t: "log0" for t in TOKS}}),
                                encoding="utf-8")
        self.plan = {}
        self.plan_default()
        mod.run_wrapped = self.fake

    def plan_default(self, **kw):
        self.plan = dict(rc=0, tokens=list(TOKS), write_csv=True, log_ok=True)
        self.plan.update(kw)

    def run_out(self, label="refe_t"):
        return self.exp / "runs" / label

    def rec(self, tok, i):
        r = {"no_at_fault_collisions": 1.0, "drivable_area_compliance": 1.0,
             "ego_progress": round(0.2 + 0.1 * i, 10), "time_to_collision_within_bound": 1.0, "comfort": 1.0,
             "driving_direction_compliance": 1.0}
        r["score"] = self.m.formula(r)
        return {"token": tok, "row": r}

    def write_jsonl(self, path, toks):
        with open(path, "w", encoding="utf-8") as fh:
            for i, t in enumerate(toks):
                fh.write(json.dumps(self.rec(t, i)) + "\n")

    def fake(self, script, label, overrides, out_dir, *, patch, hashseed="1", record_poses=False):
        p = self.plan
        os.makedirs(out_dir, exist_ok=True)
        ro = self.run_out(label)
        ro.mkdir(parents=True, exist_ok=True)
        jsonl = Path(out_dir) / f"{label}_rows.jsonl"
        self.write_jsonl(jsonl, p["tokens"])                       # the real wrapper opens it with "w"
        lines = ["CMD: fake"]
        if p["write_csv"]:
            dst = ro / "2026.10.19.12.00.00.csv"
            self.m.reconstruct_csv(str(jsonl), str(dst))
            if p["log_ok"]:
                lines += [f"Results are stored in: {dst}"]
        if p["log_ok"]:
            lines += [f"Number of successful scenarios: {len(p['tokens'])}", "Number of failed scenarios: 0"]
        log = Path(out_dir) / f"{label}.log"
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p["rc"], str(log), 0.01

    def plant_stale(self, label="refe_t", n=3):
        """A partial reconstruction from an earlier attempt: one in the devkit run dir, one raw copy."""
        ro, od = self.run_out(label), self.raw / label
        ro.mkdir(parents=True, exist_ok=True)
        od.mkdir(parents=True, exist_ok=True)
        jl = self.tmp / "stale_rows.jsonl"
        self.write_jsonl(jl, TOKS[:n])
        a, b = ro / "2026.10.18.partial.csv", od / f"{label}.csv"
        self.m.reconstruct_csv(str(jl), str(a))
        self.m.reconstruct_csv(str(jl), str(b))
        return a, b

    def run(self, label="refe_t", arm="CV"):
        ns = types.SimpleNamespace(label=label, arm=arm, tokens=str(self.tokfile), cache_name="navtest",
                                   worker="sequential", hashseed="1", record_poses=False, patch_loader=False)
        rc = self.m.cmd_score(ns)
        return rc, json.load(open(self.raw / label / f"{label}.counts.json", encoding="utf-8"))


def fixed_rig(fx, tmp):
    return Rig(tmp, fx.load(orig_path=str(ORIG), name=f"fx_{next(_uniq)}"))


def orig_rig(tmp):
    return Rig(tmp, _import(ORIG, f"orig_{next(_uniq)}"))


def last_json(text):
    return json.loads([ln for ln in text.splitlines() if ln.startswith("{")][-1])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def same_file(a, b):
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def _boom(path, start=os.curdir):
    raise ValueError("path is on mount 'C:', start on mount 'E:'")


def _other_drive(tmp):
    here = os.path.splitdrive(str(tmp))[0].upper()
    return next(f"{d}:" for d in "QZYX" if f"{d}:" != here)


# ------------------------------- scenarios: each returns normally iff its fix works --------------------- #
def scn_a_sim(fx, tmp, capsys, monkeypatch):
    capsys.readouterr()
    rig = fixed_rig(fx, tmp)
    with monkeypatch.context() as mp:
        mp.setattr(os.path, "relpath", _boom)                      # what ntpath raises across drives
        rc, rep = rig.run()
    assert rc == 0 and rep["status"] == "PASS", rep.get("failures")
    assert same_file(rep["csv"], rig.raw / "refe_t" / "refe_t.csv") and "\\" not in rep["csv"]
    assert rep["csv_path_kind"] == "abs"
    assert rep["csv_token_rows"] == N and rep["summary_x100_4dp"]["PDMS"] == 77.0833


def scn_a_real(fx, tmp, capsys, monkeypatch):
    capsys.readouterr()
    if os.name != "nt":
        pytest.skip("real cross-drive relpath failure exists on Windows only")
    rig = fixed_rig(fx, tmp)
    rig.m.PKG = f"{_other_drive(tmp)}/elsewhere_pkg"               # a different drive letter, no monkeypatch
    rc, rep = rig.run()
    assert rc == 0 and rep["status"] == "PASS", rep.get("failures")
    assert same_file(rep["csv"], rig.raw / "refe_t" / "refe_t.csv") and rep["csv_path_kind"] == "abs"


def scn_b_stale_never_scored(fx, tmp, capsys, monkeypatch):
    """The earlier attempt left partials; the new attempt dies with nothing. Nothing stale may be scored."""
    capsys.readouterr()
    rig = fixed_rig(fx, tmp)
    a, b = rig.plant_stale()
    sa, sb = sha(a), sha(b)
    rig.plan_default(rc=3, tokens=[], write_csv=False, log_ok=False)    # RAM-guard abort before any row
    rc, rep = rig.run()
    out = capsys.readouterr().out
    assert '"fix1_quarantine": 2' in out and last_json(out)["status"] == "FAIL"   # in the driver's log too
    assert rc != 0 and rep["status"] == "FAIL"
    assert rep["csv_src"] is None and "csv" not in rep, "a stale partial was scored"
    assert "no result CSV found" in rep["failures"]
    assert not a.exists() and not b.exists(), "stale csv left where the harness can find it"
    q = rep["stale_csv_quarantine"]
    assert q["n_moved"] == 2 and {os.path.basename(x["from"]) for x in q["moved"]} == {a.name, b.name}
    for x in q["moved"]:
        assert Path(x["to"]).is_file() and os.path.basename(os.path.dirname(x["to"])).startswith("quarantine_")
        assert x["token_rows"] == 3 and x["sha256"] in (sa, sb) and sha(x["to"]) == x["sha256"]


def scn_b_stale_beside_a_good_run(fx, tmp, capsys, monkeypatch):
    capsys.readouterr()
    rig = fixed_rig(fx, tmp)
    a, b = rig.plant_stale()
    rc, rep = rig.run()
    assert rc == 0 and rep["status"] == "PASS" and rep["csv_token_rows"] == N, rep.get("failures")
    assert os.path.basename(rep["csv_src"]) == "2026.10.19.12.00.00.csv", "scored the stale file"
    assert rep["stale_csv_quarantine"]["n_moved"] == 2


def scn_b_nothing_to_move(fx, tmp, capsys, monkeypatch):
    capsys.readouterr()
    rig = fixed_rig(fx, tmp)
    rc, rep = rig.run()
    assert rc == 0 and rep["stale_csv_quarantine"] == {"n_moved": 0, "moved": []}
    assert not [p for p in rig.run_out().iterdir() if p.name.startswith("quarantine_")]


def scn_c_partial_prints_no_average(fx, tmp, capsys, monkeypatch):
    capsys.readouterr()
    rig = fixed_rig(fx, tmp)
    rig.plan_default(rc=3, tokens=TOKS[:4], write_csv=False, log_ok=False)   # devkit died, 4 of 6 streamed
    rc, rep = rig.run()
    out = capsys.readouterr().out
    assert rc != 0 and rep["status"] == "FAIL" and rep["csv_token_rows"] == 4 and rep["n_expected"] == N
    assert "summary" not in rep and "summary_x100_4dp" not in rep, "partial average left in counts.json"
    assert "72.9167" not in out and "summary_x100_4dp" not in out, "partial average printed"
    assert rep["summary_suppressed"]["csv_token_rows"] == 4 and "csv_token_rows=4 != 6" in rep["summary_suppressed"]["reason"]
    assert last_json(out)["status"] == "FAIL" and "summary_suppressed" in last_json(out)


def scn_c_wrong_token_set(fx, tmp, capsys, monkeypatch):
    capsys.readouterr()
    rig = fixed_rig(fx, tmp)
    rig.plan_default(tokens=TOKS[:5] + ["intruder"])                          # right COUNT, wrong SET
    rc, rep = rig.run()
    out = capsys.readouterr().out
    assert rc != 0 and rep["status"] == "FAIL" and rep["csv_token_rows"] == N and rep["C3_unexpected"] == 1
    assert "summary" not in rep and "summary_x100_4dp" not in rep and "summary_x100_4dp" not in out


def scn_c_good_run_keeps_its_summary(fx, tmp, capsys, monkeypatch):
    capsys.readouterr()
    rig = fixed_rig(fx, tmp)
    rc, rep = rig.run()
    out = capsys.readouterr().out
    assert rc == 0 and rep["status"] == "PASS" and "summary_suppressed" not in rep
    assert rep["summary_x100_4dp"]["PDMS"] == 77.0833 and rep["summary"]["n"] == N
    assert last_json(out)["summary_x100_4dp"]["PDMS"] == 77.0833


RED = {       # scenarios that MUST fail once that fix's line is reverted
    "A": [scn_a_sim, scn_a_real],
    "B": [scn_b_stale_never_scored, scn_b_stale_beside_a_good_run],
    "C": [scn_c_partial_prints_no_average, scn_c_wrong_token_set],
}
CONTROLS = [scn_b_nothing_to_move, scn_c_good_run_keeps_its_summary]      # green on every mutant


# ------------------------------------ the fixed module: every scenario green --------------------------- #
@pytest.mark.parametrize("scn", [s for v in RED.values() for s in v] + CONTROLS, ids=lambda s: s.__name__)
def test_fix(FX, tmp_path, capsys, monkeypatch, scn):
    scn(FX, tmp_path, capsys, monkeypatch)


# --------------------------- CONTROLS: the unfixed run_v1.py really has each defect -------------------- #
def test_control_a_original_crashes_across_drives(tmp_path, monkeypatch):
    rig = orig_rig(tmp_path)
    with monkeypatch.context() as mp:
        mp.setattr(os.path, "relpath", _boom)
        with pytest.raises(ValueError, match="mount"):
            rig.run()


@pytest.mark.skipif(os.name != "nt", reason="real cross-drive failure is Windows-only")
def test_control_a_original_crashes_on_a_real_second_drive(tmp_path):
    rig = orig_rig(tmp_path)
    rig.m.PKG = f"{_other_drive(tmp_path)}/elsewhere_pkg"
    with pytest.raises(ValueError, match="mount"):
        rig.run()
    # ... and the crash is AFTER the devkit finished and BEFORE any report was written
    assert not (rig.raw / "refe_t" / "refe_t.counts.json").exists()


def test_control_b_original_rescores_the_stale_partial(tmp_path):
    rig = orig_rig(tmp_path)
    a, _ = rig.plant_stale()
    rig.plan_default(rc=3, tokens=[], write_csv=False, log_ok=False)
    rc, rep = rig.run()
    assert same_file(rep["csv_src"], a) and rep["csv_token_rows"] == 3, "glob fallback no longer re-copies it?"


def test_control_c_original_prints_the_partial_average(tmp_path, capsys):
    rig = orig_rig(tmp_path)
    rig.plan_default(rc=3, tokens=TOKS[:4], write_csv=False, log_ok=False)
    rc, rep = rig.run()
    out = capsys.readouterr().out
    assert rc != 0 and rep["status"] == "FAIL"                     # C2 already fails it ...
    assert rep["summary_x100_4dp"]["PDMS"] == 72.9167              # ... but still reports the partial mean
    assert "72.9167" in out


# --------------------------------------- non-regression and plumbing --------------------------------- #
def test_good_run_is_unchanged_apart_from_the_added_fields(FX, tmp_path, capsys):
    o = orig_rig(tmp_path / "o")
    rc_o, rep_o = o.run()
    out_o = last_json(capsys.readouterr().out)
    f = fixed_rig(FX, tmp_path / "f")
    rc_f, rep_f = f.run()
    out_f = last_json(capsys.readouterr().out)
    assert rc_o == rc_f == 0
    for k in ("status", "failures", "log_successful", "log_failed", "csv_token_rows", "csv_valid_rows",
              "C1_max_abs_delta", "summary", "summary_x100_4dp", "n_expected", "arm", "rc"):
        assert rep_o[k] == rep_f[k], k
    assert rep_o["csv"] == rep_f["csv"] == "raw/refe_t/refe_t.csv" and rep_f["csv_path_kind"] == "rel"
    out_o.pop("wall_s"), out_f.pop("wall_s")
    assert out_o == out_f
    assert rep_f["harness_fix1"]["run_v1_sha256"] == PIN and rep_f["harness_fix1"]["version"] == 1
    assert rep_f["harness_fix1"]["run_v1_matches_pinned"] is True


def test_cli_main_dispatches_the_patched_cmd_score(FX, tmp_path, capsys):
    rig = fixed_rig(FX, tmp_path)
    rc = rig.m.main(["score", "--label", "refe_t", "--arm", "CV", "--tokens", str(rig.tokfile),
                     "--cache-name", "navtest"])
    assert rc == 0
    rep = json.load(open(rig.raw / "refe_t" / "refe_t.counts.json", encoding="utf-8"))
    assert rep["harness_fix1"]["fixes"] == list(REAL.FIXES)


def test_quarantine_never_overwrites_and_leaves_other_things_alone(tmp_path):
    d = tmp_path / "[run]"                                         # glob metacharacters in the path
    (d / "quarantine_old").mkdir(parents=True)
    (d / "quarantine_old" / "kept.csv").write_text("x", encoding="utf-8")
    (d / "notes.txt").write_text("n", encoding="utf-8")
    (d / "a.csv").write_text(",token,valid\n0,t,True\n1,average,True\n", encoding="utf-8")
    m1 = REAL.quarantine_stale_csvs(str(d), str(tmp_path / "no_such_out"), "L", "20261019T000000")
    (d / "b.csv").write_text(",token,valid\n", encoding="utf-8")
    m2 = REAL.quarantine_stale_csvs(str(d), str(tmp_path / "no_such_out"), "L", "20261019T000000")
    assert [x["token_rows"] for x in m1] == [1] and [x["token_rows"] for x in m2] == [0]
    assert os.path.dirname(m1[0]["to"]).endswith("quarantine_20261019T000000")
    assert os.path.dirname(m2[0]["to"]).endswith("quarantine_20261019T000000_2")
    assert (d / "quarantine_old" / "kept.csv").exists() and (d / "notes.txt").exists()
    assert not (d / "a.csv").exists() and not (d / "b.csv").exists()


def test_the_original_is_still_the_validated_bytes():
    """run_v1.py is bound by the Thor admission record, the Colab code manifest and 24 thor.json files."""
    raw = ORIG.read_bytes()
    assert PIN in (hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest())
    assert REAL.ORIGINAL_SHA256 == PIN


def test_drift_is_refused_and_crlf_is_accepted(tmp_path):
    raw = ORIG.read_bytes().replace(b"\r\n", b"\n")
    crlf = tmp_path / "run_v1.py"
    crlf.write_bytes(raw.replace(b"\n", b"\r\n"))                  # an autocrlf checkout: the same program
    assert REAL.load(orig_path=str(crlf), name=f"crlf_{next(_uniq)}").FIX1["original"]["matches_pinned"]
    bad = tmp_path / "drifted" / "run_v1.py"
    bad.parent.mkdir()
    bad.write_bytes(raw + b"# one byte of drift\n")
    with pytest.raises(REAL.Fix1DriftError):
        REAL.load(orig_path=str(bad), name=f"bad_{next(_uniq)}")
    m = REAL.load(orig_path=str(bad), name=f"badok_{next(_uniq)}", allow_drift=True)
    assert m.FIX1["original"]["matches_pinned"] is False


# ------------------------------- RED arms: revert one fix line, ITS tests must fail ------------------- #
@pytest.mark.parametrize("which", sorted(MUTATIONS))
def test_mutant_goes_red_only_where_it_should(which, tmp_path, capsys, monkeypatch):
    mut = build_mutant(which, tmp_path / "mut")
    ran = 0
    for scn in RED[which]:
        if scn is scn_a_real and os.name != "nt":
            continue
        ran += 1
        with pytest.raises((AssertionError, ValueError)):
            scn(mut, tmp_path / f"{which}_{scn.__name__}", capsys, monkeypatch)
    assert ran >= 1
    green = [s for w, ss in RED.items() if w != which for s in ss if s is not scn_a_real] + CONTROLS
    for scn in green:                                              # every OTHER fix is still intact
        scn(mut, tmp_path / f"g_{which}_{scn.__name__}", capsys, monkeypatch)
