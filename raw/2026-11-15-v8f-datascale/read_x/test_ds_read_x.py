"""test_ds_read_x.py -- unit tests of the DSCALE10K X_s0 read (rules: blockx/MM_RULINGS_X_READ.md, sha256 e6e3f24c...): the label rule R2 reached by PLANTED deltas (both signs, the boundaries, the floor and NaN guards),
STAGE-SENSITIVE, the precedence BLOCKED > NOT-YET > label > CONTROL FAILED, K-VLM-ZERO pass / fail, K1, K-EVALX over both stages, C-JOIN-EQ, K-RE of the goal-F1 tool, the ruling file's sha256, the block-X constants / checkpoint
roots / route jobs, make_banker_x (ledger, hub, chain sentinel), and the launcher.
Run (the whole file, never -k alone):  env -u DS_BLOCK CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -p no:cacheprovider --basetemp=F:/TanitAD-scratch/datascale/readx/bt/t test_ds_read_x.py"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ds_lib as L  # noqa: E402
import ds_pull as PU  # noqa: E402
import ds_read_x as X  # noqa: E402
import make_banker_x as MB  # noqa: E402

PYX = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def in_block(n, code):
    """run `code` in a fresh interpreter with DS_BLOCK=n (the lib reads it at import) -> stdout"""
    r = subprocess.run([PYX, "-c", "import sys; sys.path.insert(0, " + repr(HERE) + "); " + code], env=dict(os.environ, DS_BLOCK=str(n), CUDA_VISIBLE_DEVICES="-1"), capture_output=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-400:]
    return r.stdout.strip()


# ---------------------------------------------------------------------------------------------------------------------------------------- R2: the label
F = 0.0767


@pytest.mark.parametrize("delta,sep,want", [
    (-F, True, "X-CONTENT"),                  # separated and exactly -F: "worse by at least the floor" is inclusive
    (-0.2, True, "X-CONTENT"),
    (-F + 1e-9, True, "X-NO-COST"),           # separated but |Delta| < F
    (0.0, True, "X-NO-COST"),
    (F - 1e-9, True, "X-NO-COST"),
    (F, True, "X-GAIN"),                      # separated and exactly +F: inclusive
    (0.3, True, "X-GAIN"),
    (-0.3, False, "X-NO-COST"),               # not separated: NO-COST whatever the size, both signs
    (0.3, False, "X-NO-COST"),
    (-F, False, "X-NO-COST"),
])
def test_label_boundaries_and_signs(delta, sep, want):
    assert X.label_x(delta, sep, F) == want


@pytest.mark.parametrize("f", [None, 0.0, -0.01, float("nan"), float("inf")])
def test_a_floor_that_is_not_positive_and_finite_is_not_a_floor(f):
    assert X.label_x(-1.0, True, f) == "NOT-YET" and X.label_x(1.0, True, f) == "NOT-YET"


def test_a_non_finite_delta_or_unread_separation_is_not_yet_never_no_cost():
    assert X.label_x(float("nan"), True, F) == "NOT-YET" and X.label_x(None, True, F) == "NOT-YET" and X.label_x(-0.3, None, F) == "NOT-YET"


def test_the_three_labels_are_mutually_exclusive_over_a_grid():
    import itertools
    for d, s in itertools.product([x / 100.0 for x in range(-30, 31)], (True, False)):
        got = X.label_x(d, s, F)
        assert got in X.LABELS
        assert (got == "X-CONTENT") == (s and d <= -F) and (got == "X-GAIN") == (s and d >= F)


def _row(d, sep):
    return {"delta": d, "separated": sep}


def test_stage_a_decides_and_stage_sensitive_prints_both_labels():
    same = X.stage_labels(_row(-0.2, True), _row(-0.2, True), F, 0.03)
    assert same["label"] == "X-CONTENT" and not same["stage_sensitive"] and X.label_text(same) == "X-CONTENT (one seed)"
    sens = X.stage_labels(_row(0.0, False), _row(-0.2, True), F, 0.03)               # A: NO-COST, B: CONTENT
    assert sens["label"] == "X-NO-COST" and sens["label_stage_B"] == "X-CONTENT" and sens["stage_sensitive"]
    t = X.label_text(sens)
    assert "STAGE-SENSITIVE" in t and "stage A X-NO-COST" in t and "stage B X-CONTENT" in t and t.startswith("X-NO-COST (one seed)")
    rev = X.stage_labels(_row(-0.2, True), _row(0.0, False), F, 0.03)
    assert rev["label"] == "X-CONTENT" and rev["stage_sensitive"]                    # stage A still names the lever
    assert X.stage_labels(_row(-0.2, True), _row(-0.2, True), F, 0.03)["F_cell"] == {"A": F, "B": 0.03}


def test_each_stage_uses_its_own_floor():
    lab = X.stage_labels(_row(-0.05, True), _row(-0.05, True), F, 0.03)              # -0.05: inside the stage-A floor, beyond the stage-B floor
    assert lab["label_stage_A"] == "X-NO-COST" and lab["label_stage_B"] == "X-CONTENT" and lab["stage_sensitive"]


def test_a_stage_b_not_yet_is_reported_and_is_not_stage_sensitive():
    lab = X.stage_labels(_row(-0.2, True), _row(float("nan"), True), F, 0.03)
    assert lab["label"] == "X-CONTENT" and lab["label_stage_B"] == "NOT-YET" and not lab["stage_sensitive"]
    assert X.stage_labels(_row(float("nan"), True), _row(-0.2, True), F, 0.03)["label"] == "NOT-YET"
    assert X.stage_labels(None, None, F, 0.03)["label"] == "NOT-YET"


def test_every_label_has_its_committed_next_lever():
    assert set(X.NEXT) == set(X.LABELS)
    assert "Zero-training audit" in X.NEXT["X-CONTENT"] and "UM_s0" in X.NEXT["X-CONTENT"] and "UM_s0" in X.NEXT["X-NO-COST"] and "UM_s0 first" in X.NEXT["X-GAIN"]


def test_pairs_are_x_against_b_s0_deciding_and_x_against_u_s0_beside():
    assert X.PAIRS == {"X-B_s0": ("X_s0", "B_s0"), "X-U_s0": ("X_s0", "U_s0")} and X.CELL == "TAC-LAT" and X.STAGES == ("A", "B")


# ---------------------------------------------------------------------------------------------------------------------------------------- precedence
def test_outcome_precedence_blocked_notyet_label_controls():
    lab = X.stage_labels(_row(-0.2, True), _row(-0.2, True), F, 0.03)
    assert X.outcome_x(lab, ["X_s0.A"], True, []) == ["BLOCKED"] and X.outcome_x(lab, ["X_s0.A"], False, ["x"]) == ["BLOCKED"]           # BLOCKED beats everything
    assert X.outcome_x(None, None, False, []) == ["NOT-YET"] and X.outcome_x(lab, None, False, []) == ["NOT-YET"] and X.outcome_x(None, None, True, ["x"]) == ["NOT-YET"]
    nl = X.stage_labels(_row(float("nan"), True), _row(0.0, False), F, 0.03)
    assert X.outcome_x(nl, None, True, []) == ["NOT-YET"]                                                          # a NOT-YET label is never printed as a label
    assert X.outcome_x(lab, None, True, []) == ["X-CONTENT (one seed)"]
    cf = X.outcome_x(lab, None, True, ["K-VLM-ZERO: x"])
    assert len(cf) == 1 and "X-CONTENT" in cf[0] and "CONTROL FAILED" in cf[0] and "not quotable" in cf[0]
    assert "STAGE-SENSITIVE" in X.outcome_x(X.stage_labels(_row(0.0, False), _row(-0.2, True), F, 0.03), None, True, [])[0]


# ---------------------------------------------------------------------------------------------------------------------------------------- R4 controls
@pytest.mark.parametrize("x,b,fl,want", [
    (0.0983, 0.12574, 0.02737, True),         # just below B_s0 minus the floor: passes
    (0.09, 0.12574, 0.02737, True),
    (0.0501, 0.06, 0.02, False),              # above 0.05 and above b - floor (0.04): fails
    (0.05, 0.06, 0.02, True),                 # at most 0.05 passes through the absolute bar although above b - floor
    (0.0499, 0.06, 0.02, True),
    (0.12, 0.12574, 0.02737, False),          # X scores VLM classes like B_s0: the mask did not hold
    (0.12574, 0.12574, 0.02737, False),
    (0.16, 0.12574, 0.02737, False),
    (0.0, 0.12574, 0.02737, True),
])
def test_k_vlm_zero_pass_and_fail(x, b, fl, want):
    assert X.k_vlm_zero(x, b, fl)["pass"] is want


def test_k_vlm_zero_boundary_is_inclusive_and_a_missing_number_is_a_fail():
    assert X.k_vlm_zero(0.1 - 0.02, 0.1, 0.02)["pass"] and X.k_vlm_zero(0.05, 0.5, 0.01)["pass"]
    for bad in ((float("nan"), 0.1, 0.02), (0.01, float("nan"), 0.02), (0.01, 0.1, float("nan")), (None, 0.1, 0.02), (0.01, 0.1, None)):
        assert X.k_vlm_zero(*bad)["pass"] is False
    r = X.k_vlm_zero(0.0501, 0.06, 0.02)
    assert not r["le_abs"] and not r["le_b_minus_floor"] and r["bar_abs"] == 0.05 and r["bar_b_minus_floor"] == pytest.approx(0.04)


def _z(d=0.0, lo=0.0, hi=0.0, status="OK"):
    return {"status": status, "delta": d, "lo": lo, "hi": hi}


def test_k1_self_pair_needs_exact_zero_on_every_ok_cell():
    assert X.self_pair_problems({"a": _z(), "b": _z(), "c": _z(status="NOT-APPLICABLE")}) == []
    assert X.self_pair_problems({"a": _z(), "b": _z(d=1e-12)}) and X.self_pair_problems({"a": _z(lo=-0.01)}) and X.self_pair_problems({"a": _z(hi=0.01)})
    assert X.self_pair_problems({}) and X.self_pair_problems({"a": _z(status="NOT-APPLICABLE")})            # nothing read is not a pass


def _kx(path, arms, stages, ok=True, drop=None, extra=None):
    items = {f"{a}.{s} {v}": {"identical": ok} for a in arms for s in stages for v in ("real", "donor")}
    if drop:
        items.pop(drop)
    if extra:
        items[extra] = {"identical": True}
    path.write_text(json.dumps({"pass": ok, "items": items}), encoding="utf-8")


def test_k_evalx_on_x_covers_both_stages(tmp_path):
    p = tmp_path / "KEVALX.json"
    assert "missing" in X.kevalx_problem_x(str(p), ("X_s0",))
    _kx(p, ("X_s0",), ("A", "B"))
    assert X.kevalx_problem_x(str(p), ("X_s0",)) is None
    _kx(p, ("X_s0",), ("A", "B"), drop="X_s0.B donor")
    assert "not a PASS" in X.kevalx_problem_x(str(p), ("X_s0",))
    _kx(p, ("X_s0",), ("A",))
    assert "not a PASS" in X.kevalx_problem_x(str(p), ("X_s0",))                                             # stage A only is not both stages
    _kx(p, ("X_s0",), ("A", "B"), ok=False)
    assert "not a PASS" in X.kevalx_problem_x(str(p), ("X_s0",))
    _kx(p, ("X_s0",), ("A", "B"), extra="B_s0.A real")
    assert "not a PASS" in X.kevalx_problem_x(str(p), ("X_s0",))                                             # an unexpected item set is refused


def _route_dir(d, items, variants=("real", "donor", "zero", "realx", "donorx")):
    d.mkdir(parents=True, exist_ok=True)
    for a, s in items:
        for v in variants:
            (d / f"eval_DS-{a}-{s}_{v}.npz").write_bytes(b"x")
    return str(d)


def test_route_problem_needs_every_pass_in_every_folder_and_a_kevalx_pass_in_each(tmp_path):
    refs = ("B_s0", "U_s0", "B_rep")
    xi, r3, r1 = [("X_s0", "A"), ("X_s0", "B")], [(a, "A") for a in refs], [(a, "B") for a in refs]
    dirs = {"X": _route_dir(tmp_path / "x", xi), "R3": _route_dir(tmp_path / "r3", r3), "R1": _route_dir(tmp_path / "r1", r1)}
    abd = {"X": xi, "R3": r3, "R1": r1}
    _kx(tmp_path / "x" / "KEVALX.json", ("X_s0",), ("A", "B"))
    _kx(tmp_path / "r3" / "KEVALX.json", refs + ("UE_s0", "BL_s0"), ("A",))                                  # the reference records are larger item sets: fine
    _kx(tmp_path / "r1" / "KEVALX.json", refs, ("B",))
    assert X.route_problem_x(dirs, abd) is None
    _kx(tmp_path / "r1" / "KEVALX.json", refs, ("B",), ok=False)
    assert "K-EVALX" in X.route_problem_x(dirs, abd)
    _kx(tmp_path / "r1" / "KEVALX.json", refs, ("B",))
    _kx(tmp_path / "x" / "KEVALX.json", ("X_s0",), ("A",))
    assert "K-EVALX" in X.route_problem_x(dirs, abd)                                                         # X must pass at both stages
    _kx(tmp_path / "x" / "KEVALX.json", ("X_s0",), ("A", "B"))
    os.remove(tmp_path / "x" / "eval_DS-X_s0-B_donorx.npz")
    assert "T0 route passes missing" in X.route_problem_x(dirs, abd) and "X_s0.B donorx" in X.route_problem_x(dirs, abd)


def test_c_join_eq_is_quoted_from_the_gate_record():
    ok = {"C-JOIN-EQ": {"pass": True, "problems": [], "rule": "X: the arm join against the union join on the X clips"}}
    assert X.cjoin_problem(ok) is None
    assert X.cjoin_problem({"C-JOIN-EQ": {"pass": False, "problems": ["array k: 3 differing elements"]}}) and X.cjoin_problem({"C-JOIN-EQ": {"pass": True, "problems": ["x"]}})
    assert X.cjoin_problem({}) and X.cjoin_problem(None) and X.cjoin_problem({"C-JOIN-EQ": {"pass": "yes", "problems": []}})
    rec = json.load(open(HERE + "/evidence/X_s0.A.checks.json", encoding="utf-8")) if os.path.isfile(HERE + "/evidence/X_s0.A.checks.json") else None
    if rec is not None:                                                                                       # the banked record of the real chain (sha256-verified by make_banker_x.py)
        assert X.cjoin_problem(rec) is None


def _f1(shift=0.0):
    return {st: {w: {"VLM": 0.1 + shift, "GEO": 0.5 + shift} for w in ("B_s0", "B_rep")} for st in X.STAGES}


def _diag(shift=0.0):
    return {"pairs": {st: {"floor_Brep-B_s0": {"VLM": {"a": 0.1 + shift, "b": 0.1 + shift}, "GEO": {"a": 0.5 + shift, "b": 0.5 + shift}}} for st in X.STAGES}}


def test_k_re_of_the_goal_f1_tool_is_exact():
    assert X.kre_f1(_f1(), _diag())["pass"] and X.kre_f1(_f1(), _diag())["rows_compared"] == 8
    assert not X.kre_f1(_f1(1e-6), _diag())["pass"]
    d = _f1()
    d["B"]["B_rep"]["GEO"] = float("nan")
    assert not X.kre_f1(d, _diag())["pass"]


def test_the_ruling_file_is_checked_by_sha256(tmp_path):
    assert X.rulings_problem() is None and L.sha256_file(X.RULINGS) == X.RULINGS_SHA256 == "e6e3f24cc79fb65098af3e6b166ebb9375d2c65ca939e5c4e937e989a58aecca"
    t = tmp_path / "r.md"
    t.write_bytes(open(X.RULINGS, "rb").read() + b" ")
    assert "hashes to" in X.rulings_problem(str(t))
    assert "missing" in X.rulings_problem(str(tmp_path / "nope.md"))


def test_the_committed_rules_in_the_docstring_carry_the_ruling_text():
    d = X.__doc__
    for s in ("X-CONTENT  : separated AND Delta <= -F_cell", "X-NO-COST : not separated OR |Delta| < F_cell", "X-GAIN : separated AND Delta >= +F_cell", "B 2,000, seed 0", "STAGE-SENSITIVE", "K-VLM-ZERO", "<= 0.05", "C-JOIN-EQ",
              "BLOCKED > NOT-YET > label", "CONTROL FAILED, not quotable", "fog unscored"):
        assert s in d, s
    ruling = open(X.RULINGS, encoding="utf-8").read()
    for s in ("X-CONTENT", "X-NO-COST", "X-GAIN", "STAGE-SENSITIVE", "K-VLM-ZERO", "K1", "K-RE", "K-EVALX", "C-JOIN-EQ", "0.0767", "B 2,000, seed 0"):
        assert s in ruling, s


# ---------------------------------------------------------------------------------------------------------------------------------------- the four families per pair (never pooled)
def _frow(gain=False, cost=False):
    return {"status": "OK", "gain": gain, "cost": cost, "delta": 0.1, "n_windows": 5}


def test_pair_family_status_reads_each_family_from_its_own_headline_cells():
    rows = {c: _frow() for f in X.FAMILIES for c in X.HEADLINE[f]}
    rows[X.HEADLINE["LATERAL"][0]] = _frow(gain=True)
    rows[X.HEADLINE["TACTICAL"][1]] = _frow(cost=True)
    st = X.pair_family_status(rows)
    assert set(st) == {"LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"}
    assert st["LATERAL"]["status"] == "gain" and st["TACTICAL"]["status"] == "cost" and st["LONGITUDINAL"]["status"] == "none" and st["STRATEGIC"]["status"] == "none"
    del rows[X.HEADLINE["STRATEGIC"][0]], rows[X.HEADLINE["STRATEGIC"][1]]
    assert X.pair_family_status(rows)["STRATEGIC"]["status"] == "NOT-YET"                                    # STRATEGIC unread is NOT-YET, never "no gain"


def test_fam_rows_x_adds_strategic_cells_only_when_read():
    t1 = {"a": _frow()}
    assert "S1:pos_err_m@80" in X.fam_rows_x(t1, {"S1:pos_err_m@80": _frow()}) and set(X.fam_rows_x(t1, None)) == {"a"}


# ---------------------------------------------------------------------------------------------------------------------------------------- block-X constants, roots, jobs
def test_block_x_constants_and_defaults():
    assert L.X_ITEMS == (("X_s0", "A"), ("X_s0", "B")) and L.ARMS["X_s0"] == 0 and L.STAGES_OF["X_s0"] == ("A", "B") and L.REF_ARMS_X == ("B_s0", "U_s0", "B_rep")
    assert L.steps_of("X_s0", "A") == 19164 and L.steps_of("X_s0", "B") == 3000
    out = in_block(4, "import ds_lib as L; print(L.DEFAULT_ITEMS); print(L.ck_root('X_s0')); print(L.ck_root('B_s0')); print(L.ck_root('U_s0')); print(L.ck_root('B_rep'))").splitlines()
    assert out[0] == "(('X_s0', 'A'), ('X_s0', 'B'))"
    assert out[1].replace("\\", "/") == os.path.normpath(HERE + "/ckpts").replace("\\", "/") and all(o.replace("\\", "/") == L.B1_ROOT + "/ckpts" for o in out[2:])
    other = subprocess.run([PYX, "-c", "import sys; sys.path.insert(0, " + repr(HERE) + "); import ds_lib as L; print(len(L.DEFAULT_ITEMS)); print(L.ck_root('B_s0'))"], env={k: v for k, v in os.environ.items() if k != "DS_BLOCK"},
                           capture_output=True, encoding="utf-8").stdout.splitlines()
    assert other[0] == "7" and other[1].replace("\\", "/") == os.path.normpath(HERE + "/ckpts").replace("\\", "/")        # the block-1 default is unchanged


def _fake_ck(tmp_path, arm, stage, steps, sha=None):
    d = tmp_path / L.item(arm, stage)
    d.mkdir(parents=True)
    (d / "ckpt_last.pt").write_bytes(b"ckpt-bytes-" + arm.encode() + stage.encode())
    sha = sha or L.sha256_file(str(d / "ckpt_last.pt"))
    (d / "BANK_RECORD.json").write_text(json.dumps({"arm": arm, "stage": stage, "sha256": sha, "source": "hf", "commit": L.COMMIT, "steps": steps, "seed": 0}), encoding="utf-8")
    return str(tmp_path)


def test_bank_problem_wants_19164_steps_at_stage_a_and_3000_at_stage_b_for_x(tmp_path):
    assert L.bank_problem("X_s0", "A", _fake_ck(tmp_path / "a", "X_s0", "A", 19164)) is None and L.bank_problem("X_s0", "B", _fake_ck(tmp_path / "b", "X_s0", "B", 3000)) is None
    assert "3000" in L.bank_problem("X_s0", "B", _fake_ck(tmp_path / "b2", "X_s0", "B", 19164)) and "19164" in L.bank_problem("X_s0", "A", _fake_ck(tmp_path / "a2", "X_s0", "A", 3000))
    assert "sha256" in L.bank_problem("X_s0", "A", _fake_ck(tmp_path / "a3", "X_s0", "A", 19164, sha="0" * 64))


def test_route_jobs_cover_x_at_both_stages_and_five_variants():
    out = in_block(4, "import ds_lib as L, ds_run as D; j = D.route_jobs(L.ROUTE_ITEMS_X, 'o'); print(len(j)); print(sorted({x[0] for x in j})); print(D.ROUTE_STAGES); "
                      "z = [x for x in j if x[0] == 'DS-X_s0-B_zero'][0]; print(z[2] == L.TREE_ZERO and '--nav-withheld' in z[1]); r = [x for x in j if x[0] == 'DS-X_s0-A_real'][0]; print(r[2] == L.TREE); "
                      "import ds_run; print(len(D.parse_items([], route=True)), D.parse_items([]))").splitlines()
    assert out[0] == "10" and "DS-X_s0-A_zero" in out[1] and "DS-X_s0-B_donorx" in out[1] and out[2] == "('A', 'B')" and out[3] == "True" and out[4] == "True"
    assert out[5].startswith("2 [('X_s0', 'A'), ('X_s0', 'B')]")
    blk3 = subprocess.run([PYX, "-c", "import sys; sys.path.insert(0, " + repr(HERE) + "); import ds_run as D; print(D.ROUTE_STAGES)"], env=dict(os.environ, DS_BLOCK="3"), capture_output=True, encoding="utf-8").stdout.strip()
    blk1 = subprocess.run([PYX, "-c", "import sys; sys.path.insert(0, " + repr(HERE) + "); import ds_run as D; print(D.ROUTE_STAGES)"], env={k: v for k, v in os.environ.items() if k != "DS_BLOCK"}, capture_output=True, encoding="utf-8").stdout.strip()
    assert blk3 == "('A',)" and blk1 == "('B',)"                                                             # the older blocks keep their stage sets


def test_ds_pull_uses_the_git_bash_not_the_wsl_stub():
    assert PU.GIT_BASH == "C:/Program Files/Git/usr/bin/bash.exe" and os.path.isfile(PU.GIT_BASH)
    assert 'subprocess.run(["bash"' not in open(HERE + "/ds_pull.py", encoding="utf-8").read()


# ---------------------------------------------------------------------------------------------------------------------------------------- make_banker_x
def _rec(stage="A", arm="X_s0", sha="a" * 64, drop=None, status="BANKED", rev="f" * 40, repo=MB.REPO):
    files = {f"{arm}/{stage}/ckpt_last.pt": [86970901, sha], f"{arm}/{stage}/summary.json": [3697, "b" * 64], f"{arm}/{stage}/gate/checks.json": [2206, "c" * 64]}
    if drop:
        files.pop(f"{arm}/{stage}/{drop}")
    return {"tag": f"{arm}.{stage}", "status": status, "repo": repo, "revision": rev, "sha256": sha, "files": files, "t": "2026-11-18T09:00:55Z"}


def test_parse_ledger_takes_the_last_banked_line_per_tag():
    a1, a2, b = _rec("A", rev="1" * 40), _rec("A", rev="2" * 40), _rec("B")
    txt = "\n".join([json.dumps(a1), "garbage", json.dumps({**b, "status": "FAILED"}), json.dumps(a2), ""])
    led = MB.parse_ledger(txt)
    assert led["X_s0.A"]["revision"] == "2" * 40 and "X_s0.B" not in led and MB.parse_ledger("") == {} and MB.parse_ledger(None) == {}


def test_record_problem_refuses_each_defect():
    assert MB.record_problem(_rec("A"), "X_s0", "A") is None and MB.record_problem(_rec("B"), "X_s0", "B") is None
    assert "not banked" in MB.record_problem(None, "X_s0", "B")
    assert "ledger line is for" in MB.record_problem(_rec("A"), "X_s0", "B")                                  # the stage-A line is not stage B
    assert "ledger line is for" in MB.record_problem(_rec("A", status="FAILED"), "X_s0", "A")
    assert "repo" in MB.record_problem(_rec("A", repo="someone/else"), "X_s0", "A") and "repo" in MB.record_problem({**_rec("A"), "revision": ""}, "X_s0", "A")
    for f in ("summary.json", "gate/checks.json", "ckpt_last.pt"):
        assert "lists no" in MB.record_problem(_rec("A", drop=f), "X_s0", "A")
    r = _rec("A")
    r["files"]["X_s0/A/ckpt_last.pt"][1] = "d" * 64
    assert "differs from its own file entry" in MB.record_problem(r, "X_s0", "A")
    assert MB.record_problem({**_rec("A"), "sha256": "xyz"}, "X_s0", "A")


def test_hub_problem_needs_private_lfs_sha_equal_and_summary_present():
    r = _rec("A")
    hub = {"X_s0/A/ckpt_last.pt": "a" * 64, "X_s0/A/summary.json": None}
    assert MB.hub_problem(hub, True, r, "X_s0", "A") is None
    assert "not private" in MB.hub_problem(hub, False, r, "X_s0", "A")
    assert "differs" in MB.hub_problem({**hub, "X_s0/A/ckpt_last.pt": "e" * 64}, True, r, "X_s0", "A")
    assert "lacks" in MB.hub_problem({"X_s0/A/ckpt_last.pt": "a" * 64}, True, r, "X_s0", "A") and "lacks" in MB.hub_problem({"X_s0/A/summary.json": None}, True, r, "X_s0", "A")
    assert "differs" in MB.hub_problem({**hub, "X_s0/A/ckpt_last.pt": None}, True, r, "X_s0", "A")            # a non-LFS file is not a verified checkpoint


def test_banker_json_is_in_the_format_ds_pull_reads(tmp_path):
    r = _rec("B", sha="9" * 64)
    d = MB.banker_json(r, "X_s0", "B")
    (tmp_path / "X_s0.B.json").write_text(json.dumps(d), encoding="utf-8")
    sha, src, info = PU.expected_sha("X_s0", "B", str(tmp_path), lambda p: None)
    assert sha == "9" * 64 and src == "hf" and info["hf_path"] == "X_s0/B/ckpt_last.pt" and info["hf_summary_path"] == "X_s0/B/summary.json" and info["hf_rev"] == "f" * 40 and info["hf_type"] == "dataset"
    assert d["gate_checks_sha256"] == "c" * 64


def test_the_chain_must_end_in_done():
    done, fail = "ZZ DSCALE10K BLOCKX DONE ZZ", "ZZ DSCALE10K BLOCKX FAIL ZZ stage B"
    assert MB.chain_done_problem("noise\n" + done + "\n") is None
    assert "ended in" in MB.chain_done_problem(fail) and "ended in" in MB.chain_done_problem(done + "\n" + fail)                # a later FAIL stops the read
    assert MB.chain_done_problem(fail + "\n" + done) is None                                                  # a re-run that finished is DONE
    assert "no ZZ DSCALE10K BLOCKX sentinel" in MB.chain_done_problem("09:00 bank X_s0\n") and MB.chain_done_problem("") and MB.chain_done_problem(None)
    assert "no ZZ DSCALE10K BLOCKX sentinel" in MB.chain_done_problem("noise  ZZ DSCALE10K BLOCKX DONE ZZ mid-line")        # only a sentinel at the start of a line counts


def test_make_banker_x_main_writes_only_banked_stages(tmp_path):
    led = tmp_path / "led.jsonl"
    led.write_text(json.dumps(_rec("A")) + "\n", encoding="utf-8")
    rc = MB.main(["--ledger", str(led), "--no-chain-check", "--no-hub", "--bank-dir", str(tmp_path / "bank"), "--evidence", str(tmp_path / "ev")])
    assert rc == 1 and os.path.isfile(tmp_path / "bank" / "X_s0.A.json") and not os.path.exists(tmp_path / "bank" / "X_s0.B.json")        # stage B is not banked: nothing written for it
    led.write_text(json.dumps(_rec("A")) + "\n" + json.dumps(_rec("B")) + "\n", encoding="utf-8")
    assert MB.main(["--ledger", str(led), "--no-chain-check", "--no-hub", "--bank-dir", str(tmp_path / "bank2"), "--evidence", str(tmp_path / "ev2")]) == 0
    log = tmp_path / "log.txt"
    log.write_text("running\n", encoding="utf-8")
    assert MB.main(["--ledger", str(led), "--chain-log", str(log), "--no-hub", "--bank-dir", str(tmp_path / "bank3"), "--evidence", str(tmp_path / "ev3")]) == 1 and not os.path.exists(tmp_path / "bank3" / "X_s0.A.json")


# ---------------------------------------------------------------------------------------------------------------------------------------- the launcher
def test_launcher_keeps_the_bars_the_order_and_the_sentinels():
    sh = chr(10).join(ln for ln in open(HERE + "/launch_dscale10k_x.sh", encoding="utf-8").read().splitlines() if not ln.lstrip().startswith("#"))
    assert '[ "$FREE" -lt 25 ]' in sh and "ZZ DSCALE10K X READ DONE ZZ" in sh and "ZZ DSCALE10K X READ FAIL ZZ" in sh and "DS_BLOCK=4" in sh
    assert "$WC 5.5 30" in sh and "WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py" in sh and sh.count("$WC 5.5 30") == 2
    assert 'ds_read_x.py --out "$R/out/RESULT_DSCALE10K_X.json"' in sh
    steps = ("rulings_problem", "$WC 5.5 30", "pytest", "make_banker_x.py", "ds_pull.py >>", "ds_pull.py --check-only", "ds_run.py dumps", "ds_run.py route", "ds_read_x.py --out")
    order = [sh.index(x) for x in steps]
    assert order == sorted(order)
    assert "env -u DS_BLOCK" in sh and "test_ds_read_x.py test_ds_read.py" in sh                              # the tests run whole, and do not inherit the block env var
    assert "ds_pull.py --check-only --items X_s0.A X_s0.B B_s0.A B_s0.B U_s0.A U_s0.B B_rep.A B_rep.B" in sh
    import re
    assert "e6e3f24c" in sh and not re.search(r"pytest.* -k ", sh)
    assert "CreateFlags = 0x08000200" in open(HERE + "/launch_wmi_x.ps1", encoding="utf-8").read()
    cmd = open(HERE + "/job_dscale10k_x.cmd", encoding="utf-8").read()
    assert "Git" + chr(92) + "usr" + chr(92) + "bin" + chr(92) + "bash.exe" in cmd and "launch_dscale10k_x.sh" in cmd
    assert "RESULT_DSCALE10K_B1.json" in sh and "RESULT_DSCALE10K_B3.json" in sh and "RESULT_DIAG.json" in sh and sh.count("KEVALX.json") >= 2
    assert sh.count("rm -rf") == 1 and 'rm -rf "F:/TanitAD-scratch/datascale/readx/bt"' in sh                    # the one delete is a literal absolute path


def test_launcher_refuses_a_failed_ruling_check(tmp_path):
    """the sha256 gate is the first step: with the check made to fail, the launcher writes FAIL and runs nothing else"""
    sh = chr(10).join(ln for ln in open(HERE + "/launch_dscale10k_x.sh", encoding="utf-8").read().splitlines() if not ln.lstrip().startswith("#"))
    first = sh.index("rulings_problem")
    assert first < sh.index("make_banker_x.py") and first < sh.index("ds_run.py")
    lines = sh.splitlines()
    for key in ("rulings_problem", "make_banker_x.py", "ds_pull.py >>", "ds_run.py dumps", "ds_run.py route", "ds_read_x.py --out", "-m pytest"):
        assert all("|| fail" in ln for ln in lines if key in ln), key                                           # every step stops the launcher on failure


def test_kevalx_for_x_runs_at_both_stages(tmp_path):
    """cmd_kevalx in block 4 writes a record over X_s0's real and donor passes at BOTH stages (tiny identical fake passes)"""
    import numpy as np
    for st in ("A", "B"):
        for v, vx in (("real", "realx"), ("donor", "donorx")):
            for nm in (v, vx):
                np.savez(tmp_path / f"eval_DS-X_s0-{st}_{nm}.npz", a=np.arange(4.0))
                (tmp_path / f"eval_DS-X_s0-{st}_{nm}.json").write_text(json.dumps({"x": 1.0, "tag": nm}), encoding="utf-8")
    out = in_block(4, "import ds_lib as L, ds_run as D; print(D.cmd_kevalx(L.X_ITEMS, " + repr(str(tmp_path).replace(chr(92), "/")) + "))")
    k = json.load(open(tmp_path / "KEVALX.json", encoding="utf-8"))
    assert k["pass"] and set(k["items"]) == {f"X_s0.{s} {v}" for s in "AB" for v in ("real", "donor")} and X.kevalx_problem_x(str(tmp_path / "KEVALX.json"), ("X_s0",)) is None, out
