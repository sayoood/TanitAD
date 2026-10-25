"""H-NT-DECIDE-1 (PREREG_NT_DECIDE) -- the trainer-side guards and the launch-gate profile ``refcv8-nt``.

Every guard is tested on its REAL function, and each carries a deliberate-regression arm (the defect it exists for,
reintroduced) that must go RED. Expectations are LITERALS, never expressions over the code under test.

* K3 held-out guard (``tanitad.train.nt_guard``): selection log / navtest log / unknown clip / sha mismatch /
  partial flags / selection-sample token -> REFUSED; the in-tree guard file is the pinned one (38 + 136 logs).
* R2 window filter (``refc_v3_train.restrict_window_starts``): one window per episode, t = 0.
* the speed all-unknown declaration (``read_speed_max_sidecar_v6(allow_all_unknown=True)``).
* ``parse_snapshot_steps``.
* the gate profile: the A1 argv (``stack/ops/runs.d/nt-a1-s0.argv.json``) is refused ONLY for the PENDING v9 md5 and
  the two open items; every NT departure refuses when reverted; the inherited nav tau; the must-be-zero reach keys.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

STACK = Path(__file__).resolve().parents[1]
for _p in (str(STACK), str(STACK / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.train import nt_guard as G  # noqa: E402

GUARD_FILE = STACK / "tanitad/configs/refcv8_nt/nt_heldout.json"
GUARD_SHA = "4b5b1aee2be780026c3df58ffc7924f8220459e4d9aa603ef801d76d8af95652"
TAU_RECORD = STACK / "tanitad/configs/refcv8_nav_compliance_tau_train4572.json"
A1_ARGV = STACK / "ops/runs.d/nt-a1-s0.argv.json"
A1R_ARGV = STACK / "ops/runs.d/nt-a1r-s1.argv.json"
#: the PI's DATA-line ACK (PREREG sec. 4.1): the in-tree byte copy, its git blob as a LITERAL (computed by
#: `git hash-object --no-filters` on the MM's file, 2026-10-21), and the Thor inputs dir the profile names
ACK_FILE = STACK / "tanitad/configs/refcv8_nt/AMENDMENT_ACK_NT.txt"
ACK_BLOB = "38175992b524e29353a8e666c37a2813e0912080"
ACK_PMAP = [("/home/nvidia/r8ns/nt/inputs", str(ACK_FILE.parent))]


# ------------------------------------------------------------------------------------------------- K3 guard
def _guard(tmp: Path, sel=("logS1", "logS2"), nt=("logN1",), tokens=("tokS",)) -> tuple[Path, str]:
    d = {"schema": G.SCHEMA, "selection_scan_logs": list(sel), "navtest_logs": list(nt),
         "heldout_logs": sorted(set(sel) | set(nt)), "selection_sample_tokens": list(tokens),
         "counts": {"selection_scan_logs": len(sel), "navtest_logs": len(nt)}}
    p = tmp / "guard.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def _cache(tmp: Path, rows: list) -> tuple[Path, Path]:
    c = tmp / "cache"
    c.mkdir()
    for r in rows:
        (c / f"{r['clip_id']}.v2ep.pt").write_bytes(b"")          # the guard reads the LISTING only
    t = tmp / "token_table.json"
    t.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return c, t


ROWS = [{"clip_id": "navtrain-aaaa", "token": "aaaa", "log_name": "logT1"},
        {"clip_id": "navtrain-bbbb", "token": "bbbb", "log_name": "logT2"}]


def _args(g, sha, t):
    return SimpleNamespace(nt_heldout=str(g), nt_heldout_sha256=sha, nt_token_table=str(t))


def test_guard_clean_table_passes(tmp_path):
    g, sha = _guard(tmp_path)
    c, t = _cache(tmp_path, ROWS)
    cen = G.refuse_heldout_in_train(str(c), _args(g, sha, t))
    assert cen["n_heldout_hits"] == 0 and cen["n_train_clips"] == 2 and cen["n_heldout_logs"] == 3


@pytest.mark.parametrize("bad_log", ["logS1", "logN1"])
def test_guard_mutation_heldout_log_refused(tmp_path, bad_log):
    """DELIBERATE REGRESSION: one selection-scan (or navtest) log in the train table -> REFUSED."""
    g, sha = _guard(tmp_path)
    rows = ROWS + [{"clip_id": "navtrain-cccc", "token": "cccc", "log_name": bad_log}]
    c, t = _cache(tmp_path, rows)
    with pytest.raises(SystemExit, match="HELD-OUT"):
        G.refuse_heldout_in_train(str(c), _args(g, sha, t))


def test_guard_unknown_clip_refused(tmp_path):
    g, sha = _guard(tmp_path)
    c, t = _cache(tmp_path, ROWS)
    (c / "navtrain-zzzz.v2ep.pt").write_bytes(b"")                 # a clip the table does not name
    with pytest.raises(SystemExit, match="NOT in the token table"):
        G.refuse_heldout_in_train(str(c), _args(g, sha, t))


def test_guard_sha_mismatch_refused(tmp_path):
    g, sha = _guard(tmp_path)
    c, t = _cache(tmp_path, ROWS)
    with pytest.raises(SystemExit, match="sha256"):
        G.refuse_heldout_in_train(str(c), _args(g, "0" * 64, t))


def test_guard_partial_flags_refused(tmp_path):
    g, sha = _guard(tmp_path)
    c, t = _cache(tmp_path, ROWS)
    with pytest.raises(SystemExit, match="go together"):
        G.refuse_heldout_in_train(str(c), SimpleNamespace(nt_heldout=str(g), nt_heldout_sha256=None,
                                                          nt_token_table=str(t)))
    assert G.refuse_heldout_in_train(str(c), SimpleNamespace()) == {}      # not requested: inert


def test_guard_selection_sample_token_refused(tmp_path):
    g, sha = _guard(tmp_path, tokens=("bbbb",))
    c, t = _cache(tmp_path, ROWS)
    with pytest.raises(SystemExit, match="selection-sample"):
        G.refuse_heldout_in_train(str(c), _args(g, sha, t))


def test_guard_internal_checks(tmp_path):
    d = {"schema": G.SCHEMA, "selection_scan_logs": ["a"], "navtest_logs": ["b"], "heldout_logs": ["a"]}
    p = tmp_path / "g.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(SystemExit, match="heldout_logs"):
        G.load_guard(p)
    d["heldout_logs"] = ["a", "b"]
    d["counts"] = {"selection_scan_logs": 38}
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(SystemExit, match="counts"):
        G.load_guard(p)


def test_in_tree_guard_file_is_the_pinned_one():
    """The shipped guard IS the file the profile pins, and holds the PREREG's literal counts."""
    assert hashlib.sha256(GUARD_FILE.read_bytes()).hexdigest() == GUARD_SHA
    g = G.load_guard(GUARD_FILE, expect_sha256=GUARD_SHA)
    assert len(g["selection_scan_logs"]) == 38
    assert len(g["navtest_logs"]) == 136
    assert len(g["heldout_logs"]) == 174
    assert len(g.get("selection_sample_tokens") or ()) == 2089


# ------------------------------------------------------------------------------------------------- trainer helpers
@pytest.fixture(scope="module")
def T():
    import refc_v3_train as _T
    return _T


def test_window_starts_first_keeps_t0(T):
    ds = SimpleNamespace(episodes=[0, 1, 2], index=[(0, 0), (0, 1), (1, 0), (1, 5), (2, 0)])
    st = T.restrict_window_starts(ds, "first")
    assert ds.index == [(0, 0), (1, 0), (2, 0)]
    assert st == {"mode": "first", "n_windows_before": 5, "n_windows": 3, "n_episodes": 3}
    ds2 = SimpleNamespace(episodes=[0, 1], index=[(0, 0), (0, 1)])
    assert T.restrict_window_starts(ds2, "all") == {} and ds2.index == [(0, 0), (0, 1)]


def test_window_starts_mutation_missing_t0_refused(T):
    """DELIBERATE REGRESSION: an episode without a t = 0 window (a too-short token-episode) -> REFUSED."""
    ds = SimpleNamespace(episodes=[0, 1], index=[(0, 0), (1, 1)])
    with pytest.raises(SystemExit, match="exactly ONE window"):
        T.restrict_window_starts(ds, "first")
    with pytest.raises(SystemExit, match="choices"):
        T.restrict_window_starts(ds, "last")


def test_parse_snapshot_steps(T):
    assert T.parse_snapshot_steps("0,1560,3120,6239,9359,12478", 12478) == frozenset(
        {0, 1560, 3120, 6239, 9359, 12478})
    assert T.parse_snapshot_steps(None, 10) == frozenset()
    for bad in ("0,12479", "-1", "5,5", "a,b"):
        with pytest.raises(SystemExit):
            T.parse_snapshot_steps(bad, 12478)


def test_nt_flags_are_registered_for_g_dvb(T):
    """G-DVB: every trainer flag has a REGISTRY entry; the six NT flags are among them."""
    from tanitad.train import declared_vs_built as D
    parser = T.build_parser()
    assert D.coverage(parser) == []
    dests = {a.dest for a in parser._actions}
    for f in ("window_starts", "nt_heldout", "nt_heldout_sha256", "nt_token_table", "r8_speed_unknown_always",
              "snapshot_steps"):
        assert f in dests


# ------------------------------------------------------------------------------------------------- speed sidecar
def _sidecar(tmp: Path, rows: list) -> str:
    p = tmp / "speed.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (tmp / "speed.jsonl.meta.json").write_text(json.dumps({"source_md5": "feedface"}), encoding="utf-8")
    return str(p)


UNKNOWN = {"v_hi_ms": None, "bin": None, "limit_kmh": None, "one_hot": [0, 0, 0, 0], "valid": 0}


def test_speed_all_unknown_declared(tmp_path):
    from tanitad.refs.refcv6_max_speed import read_speed_max_sidecar_v6
    p = _sidecar(tmp_path, [dict(UNKNOWN, sid=1), dict(UNKNOWN, sid=2)])
    by_sid, rep = read_speed_max_sidecar_v6(p, label_md5="feedface", allow_all_unknown=True)
    assert by_sid == {1: (0.0, 0.0), 2: (0.0, 0.0)}
    with pytest.raises(SystemExit, match="NOT ONE"):                   # undeclared: the old refusal stands
        read_speed_max_sidecar_v6(p, label_md5="feedface")


def test_speed_all_unknown_mutation_known_row_refused(tmp_path):
    """DELIBERATE REGRESSION: one KNOWN ceiling in a sidecar declared all-unknown -> REFUSED."""
    from tanitad.refs.refcv6_max_speed import read_speed_max_sidecar_v6
    p = _sidecar(tmp_path, [dict(UNKNOWN, sid=1), {"sid": 2, "v_hi_ms": 13.9, "valid": 1}])
    with pytest.raises(SystemExit, match="allow_all_unknown"):
        read_speed_max_sidecar_v6(p, label_md5="feedface", allow_all_unknown=True)


# ------------------------------------------------------------------------------------------------- the gate profile
@pytest.fixture(scope="module")
def LG():
    import launch_gate as _LG
    return _LG


def _argv(path=A1_ARGV) -> list:
    return list(json.loads(Path(path).read_text(encoding="utf-8"))["argv"])


def _rules(LG, prof, argv):
    return LG.profile_argv_rules(prof, argv, {"nav_tau_record": str(TAU_RECORD)}, ACK_PMAP)[0]


def _set(argv, flag, val):
    out = list(argv)
    i = out.index(flag)
    j = i + 1
    while j < len(out) and not out[j].startswith("--"):
        j += 1
    return out[:i + 1] + ([str(val)] if val is not None else []) + out[j:]


def _drop(argv, flag):
    out = list(argv)
    i = out.index(flag)
    j = i + 1
    while j < len(out) and not out[j].startswith("--"):
        j += 1
    return out[:i] + out[j:]


def test_a1_argv_refused_only_for_the_pending_md5(LG):
    """OPEN (NT-CORPUS pending): the committed A1 argvs are refused on EXACTLY the PENDING md5. CLOSED (nt_close.py wrote
    the md5 literal into the profile AND the argvs): refused on NOTHING. The open items follow the two constants."""
    prof = LG.PROFILES["refcv8-nt"]
    closed = LG._NT_V9_MD5 != LG._NT_PENDING
    for path in (A1_ARGV, A1R_ARGV):
        ref = _rules(LG, prof, _argv(path))
        if closed:
            assert ref == [], ref
        else:
            assert len(ref) == 1 and "--r8-v9-md5" in ref[0] and "PENDING-NT-CORPUS" in ref[0]
    want = ([] if closed else ["NT-CORPUS"]) + (["NT-B1-GUARD"] if LG._NT_B1_GUARD is None else [])
    assert [it["id"] for it in prof["open_items"]] == want


def _closed(LG, md5="0123456789abcdef0123456789abcdef"):
    """The NT-CORPUS closing procedure applied to a COPY of the profile: the md5 literal written, PENDING unforbidden."""
    p = copy.deepcopy(LG.PROFILES["refcv8-nt"])
    p["required_values"]["--r8-v9-md5"] = md5
    p["forbidden_values"] = {k: v for k, v in p["forbidden_values"].items() if k != "--r8-v9-md5"}
    return p


def test_closing_nt_corpus_admits_the_a1_argv(LG):
    p = _closed(LG)
    argv = _set(_argv(), "--r8-v9-md5", "0123456789abcdef0123456789abcdef")
    assert _rules(LG, p, argv) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda a: a + ["--pose-sync-sidecar", "/home/nvidia/data/refcv8_pose_sync_sidecar.jsonl"], "--pose-sync-sidecar"),
    (lambda a: _set(a, "--steps", "48852"), "--steps"),
    (lambda a: _set(a, "--window-starts", "all"), "--window-starts"),
    (lambda a: _drop(a, "--nt-heldout"), "--nt-heldout"),
    (lambda a: _set(a, "--nt-heldout-sha256", "0" * 64), "--nt-heldout-sha256"),
    (lambda a: _drop(a, "--r8-speed-unknown-always"), "--r8-speed-unknown-always"),
    (lambda a: _set(a, "--map-min-coverage", "0.9"), "--map-min-coverage"),
    (lambda a: _set(a, "--snapshot-steps", "1560,3120,6239,9359,12478"), "--snapshot-steps"),
    (lambda a: _set(a, "--v2-cache", "/home/nvidia/data/refcv8-b1-416x1024-train-4572"), "--v2-cache"),
    (lambda a: _set(a, "--init-from", "/home/nvidia/refcv8_run/runs/refcv8-r101-s0/ckpt_48852.pt"), "--init-from"),
    (lambda a: _set(a, "--w-map-hires", "0.0"), "--w-map-hires"),
])
def test_profile_mutations_refused(LG, mutate, needle):
    """DELIBERATE REGRESSIONS: each NT departure reverted (or a refcv8 lever moved) -> REFUSED by name."""
    p = _closed(LG)
    base = _set(_argv(), "--r8-v9-md5", "0123456789abcdef0123456789abcdef")
    ref = _rules(LG, p, mutate(base))
    assert ref and any(needle in r for r in ref), ref


def test_b1_guard_decision_switch(LG, monkeypatch):
    """NT-B1-GUARD: open -> refcv8's literals required; 'drop' -> both flags refused and the item closes."""
    p = LG.PROFILES["refcv8-nt"]
    if LG._NT_B1_GUARD == "drop":         # CLOSED to drop (nt_close.py): both flags refused by the built profile
        assert "--b1-guard-ref-ap2m" not in p["required_values"]
        assert any(("--b1-guard-ref-ap2m" in str(k)) for k in (p.get("forbidden_levers") or ()))
    else:                                  # OPEN or CLOSED to keep: refcv8's literal stays required
        assert p["required_values"]["--b1-guard-ref-ap2m"] == "0.21211"
    monkeypatch.setattr(LG, "_NT_B1_GUARD", "drop")
    fl, va, fb = LG._nt_b1(tuple(p["required_flags"]), dict(p["required_values"]))
    assert "--b1-guard-ref-ap2m" not in fl and "--b1-guard-step" not in va
    assert fb and fb[0][0] == ("--b1-guard-ref-ap2m", "--b1-guard-step")


def test_nav_tau_inherited(LG, tmp_path):
    p = LG.PROFILES["refcv8-nt"]
    argv = _argv()
    reasons, det = LG.nav_tau_reasons(p, argv, str(TAU_RECORD), [])
    assert reasons == [] and det["inherited"]["sha256"].startswith("9fa7ca38")
    # DELIBERATE REGRESSION 1: a tau moved in the argv
    r1, _ = LG.nav_tau_reasons(p, _set(argv, "--nav-compliance-tau-rad", "0.2"), str(TAU_RECORD), [])
    assert r1
    # DELIBERATE REGRESSION 2: another record (a navtrain re-derivation) with the same tau -> refused by digest
    d = json.loads(TAU_RECORD.read_text(encoding="utf-8"))
    d["note"] = "re-derived elsewhere"
    other = tmp_path / "tau.json"
    other.write_text(json.dumps(d), encoding="utf-8")
    r2, _ = LG.nav_tau_reasons(p, argv, str(other), [])
    assert r2 and "inherited record" in r2[0]
    # refcv8's own profile still binds the split (the inheritance is NT-only)
    r3, _ = LG.nav_tau_reasons(LG.PROFILES["refcv8"], argv, str(TAU_RECORD), [])
    assert r3


def _rows(**over):
    base = {"ga_mh_lift": 2491.1, "ga_mh_encoder": 6117.3, "ga_mh_trunk_s8_stage": 30953.1,
            "ga_mh_refine": 0.0, "ga_mh_near": 0.0, "ga_mh_near_refine": 0.0}
    return [dict(base, step=s, **over) for s in (1, 2)]


def test_ga_reach_zero_keys(LG):
    p = LG.PROFILES["refcv8-nt"]
    r, d = LG.judge_ga_reach(p, _rows(), log_every=1, final_step=2, groups={}, ga_on=True, hires=True)
    assert r == [] and d["required_zero_keys"] == ["ga_mh_refine", "ga_mh_near", "ga_mh_near_refine"]
    # DELIBERATE REGRESSION: map supervision leaking into the map-only part
    r2, _ = LG.judge_ga_reach(p, _rows(ga_mh_refine=1e-9), log_every=1, final_step=2, groups={}, ga_on=True,
                              hires=True)
    assert r2 and "must-be-ZERO" in r2[0]
    # refcv8 itself still REQUIRES the refine reach > 0 (the zero rule is NT-only)
    r3, _ = LG.judge_ga_reach(LG.PROFILES["refcv8"], _rows(), log_every=1, final_step=2, groups={}, ga_on=True,
                              hires=True)
    assert r3


def test_admitted_dead_groups_are_the_map_only_parts(LG):
    adm = [g for g, flag, _w in LG.PROFILES["refcv8-nt"]["live_dead_admitted"] if flag == "--map-min-coverage"]
    assert len(adm) == 12
    assert all(g.startswith(("_map_hires.refine.", "_map_hires.near.", "_map_hires.near_refine.")) for g in adm)
    assert "_map_hires.refine.cls" in adm
    assert not any(g.startswith(("_map_hires.lift", "_map_hires.encoder")) for g in adm)


# ------------------------------------------------------------------------------- PREREG sec. 4.1: the DATA-line ACK
def test_git_blob_id_analytic_targets(LG):
    """git's own blob ids of two fixed byte strings (independent of the code under test)."""
    assert LG.git_blob_id(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert LG.git_blob_id(b"hello" + bytes([10])) == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_amendment_ack_is_pinned_and_the_in_tree_copy_is_it(LG):
    ack = LG.PROFILES["refcv8-nt"]["amendment_ack"]
    assert ack["git_blob"] == ACK_BLOB and ack["data_line"] == 84.5 and ack["fallback_line"] == 85.0
    assert ack["path"] == "/home/nvidia/r8ns/nt/inputs/AMENDMENT_ACK_NT.txt"
    reasons, det = LG.amendment_ack_reasons(LG.PROFILES["refcv8-nt"], ACK_PMAP)
    assert reasons == [] and det["git_blob"] == ACK_BLOB
    assert det["data_line_read"] == 84.5 and det["names_prereg_blob"] is True


def test_amendment_ack_missing_refused(LG, tmp_path):
    """Absent the ACK the DATA line is 85.0: the refusal says so."""
    reasons, _ = LG.amendment_ack_reasons(LG.PROFILES["refcv8-nt"], [("/home/nvidia/r8ns/nt/inputs", str(tmp_path))])
    assert len(reasons) == 1 and "MISSING" in reasons[0] and "85.0" in reasons[0]
    # and the A1 argv is refused on the ACK through the ONE argv-rule path (run AND finalize)
    p = _closed(LG)
    argv = _set(_argv(), "--r8-v9-md5", "0123456789abcdef0123456789abcdef")
    ref = LG.profile_argv_rules(p, argv, {"nav_tau_record": str(TAU_RECORD)},
                                [("/home/nvidia/r8ns/nt/inputs", str(tmp_path))])[0]
    assert len(ref) == 1 and "amendment ACK" in ref[0]


@pytest.mark.parametrize("edit", [
    lambda b: b.replace(b"DATA_LINE L_D = 84.5", b"DATA_LINE L_D = 84.0"),     # the line moved
    lambda b: b + bytes([10]),                                              # one byte appended
    lambda b: b.replace(bytes([10]), bytes([13, 10])),                      # a CRLF checkout of the same text
])
def test_amendment_ack_mutation_refused(LG, tmp_path, edit):
    (tmp_path / "AMENDMENT_ACK_NT.txt").write_bytes(edit(ACK_FILE.read_bytes()))
    reasons, det = LG.amendment_ack_reasons(LG.PROFILES["refcv8-nt"],
                                            [("/home/nvidia/r8ns/nt/inputs", str(tmp_path))])
    assert len(reasons) == 1 and "pins " + ACK_BLOB in reasons[0] and det["git_blob"] != ACK_BLOB


# ---------------------------------------------------------------- G-EVAL: the eval loader's recorded path override
def test_nt_eval_remap_override_names_exactly_each_arms_own_files():
    """refcv8's override (stack/ops/refcv8_eval_remap_a2_thor.json) names 1 of the NT argv's 4 files (MEASURED
    2026-10-21: --agent-rig-extrinsics / --clip-clock-sidecar / --vis1-sidecar point at the NT merged release), and
    refcv8's own preflight demands 4 of 4. The NT override must be the identity on EACH committed arm's values."""
    o = json.loads((STACK / "ops/refcv8nt_eval_remap_thor.json").read_text(encoding="utf-8"))
    r8 = json.loads((STACK / "ops/refcv8_eval_remap_a2_thor.json").read_text(encoding="utf-8"))
    # MM 2026-10-22: + --agent-join / --join3d, the two flags the loader REFUSED at the f5262ea gate (see the test below)
    assert sorted(o) == sorted(["--agent-rig-extrinsics", "--clip-clock-sidecar", "--vis1-sidecar",
                                "--nav-compliance-tau-file", "--agent-join", "--join3d"])
    assert set(r8) < set(o)
    for path in (A1_ARGV, A1R_ARGV):
        a = _argv(path)
        assert {f: a[a.index(f) + 1] for f in o} == o, path.name
    assert o["--vis1-sidecar"] == "/home/nvidia/r8ns/nt/labels/a1/nt_train_plus_eval139_vis1_sidecar.npz"


# ------------------------------------------------------------------------------ the NT pre-launch kit's literals
def test_nt_prelaunch_kit_literals():
    """scripts/nt_prelaunch.sh carries three literals the gate also binds: the ACK md5 (must be the md5 of the in-tree
    ACK copy, computed HERE), theta_DEC's md5 (the K0 record's), and the profile name; and it never fires on its own."""
    import hashlib
    import re as _re
    src = (STACK / "scripts/nt_prelaunch.sh").read_text(encoding="utf-8")
    assert _re.search(r"^ACK_MD5=([0-9a-f]{32})", src, _re.M).group(1) == hashlib.md5(ACK_FILE.read_bytes()).hexdigest()
    assert "THETA_DEC_MD5=dad4cb5b5fb316922b4e9a7e85cee8f8" in src
    assert "run --profile refcv8-nt --stage thor" in src
    assert "refcv8nt_eval_remap_thor.json" in src and "refcv8_eval_remap_a2_thor.json" not in src.split("case \"$MODE\"")[1]
    assert src.count("gpu_take") >= 4                    # defined once, taken by bind, gate and fire
    # MM 2026-10-22 00:45: the L7 yield marker is touched BEFORE the lock is taken, and the kit never removes it
    i0 = src.index("gpu_take() {")
    body = src[i0:src.index("\n}\n", i0)]
    assert "NT=/home/nvidia/r8ns/nt" in src and "NT_PENDING=$NT/NT_LAUNCH_PENDING" in src
    assert body.index('touch "$NT_PENDING"') < body.index('exec 9>"$LOCKF"') < body.index("flock -w 7200 9")
    assert "NT_LAUNCH_PENDING" in src and not any(w in src for w in ('rm -f "$NT_PENDING"', 'rm "$NT_PENDING"'))


# ---------------------------------- the f5262ea gate refusal (MM 2026-10-22 23:31 Berlin): three launch-kit defects
def test_G_LIVE_the_smoke_rewrites_the_snapshot_steps_inside_the_smoke(T, LG):
    """MEASURED: G-LIVE and G-CKPT FAILED because the 100-step smoke inherited `--snapshot-steps 0,...,12478` and the
    trainer's own pin refused it before step 1. The smoke keeps 0, adds its last step, and the REAL trainer function
    accepts the result. RED arm: the pre-fix rewrite (`--steps` only) is refused by that same function. The gate's own
    resume-run record still matches the smoke rule (G-EVAL's identity check), and a launch that already fits is untouched."""
    for path in (A1_ARGV, A1R_ARGV):
        launch = _argv(path)
        new, dep = LG.smoke_argv(launch, 100, 50)
        assert LG.flag_values(new, "--snapshot-steps") == ["0,100"], path.name
        assert "--snapshot-steps" in dep
        assert T.parse_snapshot_steps(LG.flag_values(new, "--snapshot-steps")[0],
                                      int(LG.flag_values(new, "--steps")[0])) == frozenset({0, 100})
        old = LG.set_flag(list(launch), "--steps", ["100"])
        with pytest.raises(SystemExit, match="never written"):
            T.parse_snapshot_steps(LG.flag_values(old, "--snapshot-steps")[0], 100)
        rec = LG.set_flag(LG.set_flag(new, "--out", ["/tmp/smoke"]), "--steps", ["102"])
        assert LG.stamps_written_for(rec, launch, 100, 50)[0]
    fit = LG.set_flag(_argv(A1_ARGV), "--snapshot-steps", ["0,50"])
    new, dep = LG.smoke_argv(fit, 100, 50)
    assert LG.flag_values(new, "--snapshot-steps") == ["0,50"] and "--snapshot-steps" not in dep


def _private_loader(tmp_path, monkeypatch, overrides: dict):
    """A PRIVATE instance of refcv7_loader with `overrides` as its recorded override (the module reads the env at import)."""
    import importlib.util
    f = tmp_path / "remap.json"
    f.write_text(json.dumps(overrides), encoding="utf-8")
    for k in ("REMAP_OVERRIDES", "KIT"):
        monkeypatch.delenv("REFCV7_" + k, raising=False)
        monkeypatch.delenv("REFCV6_" + k, raising=False)
    monkeypatch.setenv("REFCV6_REMAP_OVERRIDES", str(f))
    monkeypatch.setenv("REFCV6_KIT", str(tmp_path))
    spec = importlib.util.spec_from_file_location("refcv7_loader_nt_remap_test",
                                                  STACK / "tanitad/eval/refcv7_loader.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_G_EVAL_the_nt_override_clears_the_loaders_basename_refusal(tmp_path, monkeypatch):
    """MEASURED: G-EVAL FAILED -- the loader refused --agent-join and --join3d by basename (the NT argv names the NT-merged
    joins; the kit's files are the b1 ones) and the override carried only 4 flags. The REAL remap_argv on each committed
    arm under the shipped override: no refusal, both flags taken from the recorded override. RED arm: the 4-flag override
    the f5262ea gate ran with refuses, naming exactly those two flags."""
    o = json.loads((STACK / "ops/refcv8nt_eval_remap_thor.json").read_text(encoding="utf-8"))
    for path in (A1_ARGV, A1R_ARGV):
        L = _private_loader(tmp_path, monkeypatch, o)
        _, rec = L.remap_argv(_argv(path))
        for f in ("--agent-join", "--join3d"):
            assert rec["remapped"][f]["source"] == "recorded override", (path.name, f)
    four = {k: v for k, v in o.items() if k not in ("--agent-join", "--join3d")}
    L = _private_loader(tmp_path, monkeypatch, four)
    with pytest.raises(SystemExit) as e:
        L.remap_argv(_argv(A1_ARGV))
    assert "--agent-join" in str(e.value) and "--join3d" in str(e.value)


def test_G_SUITE_PINNED_import_reads_the_dir_that_holds_the_file(tmp_path, LG):
    """MEASURED: G-SUITE-PINNED read MISSING with ZERO rejected candidates -- `_collect_evidence` globs G-*.json directly
    in each --import-evidence dir, and nt_prelaunch.sh passed the dev-box out-dir ROOT (the file is under evidence/).
    RED arm: the root imports nothing. The kit now passes evidence/ and refuses when the file is absent."""
    root = tmp_path / "devbox_evidence_x"
    (root / "evidence").mkdir(parents=True)
    (root / "evidence" / "G-SUITE-PINNED.json").write_text(
        json.dumps({"kind": "evidence", "check": "G-SUITE-PINNED", "status": "PASS"}), encoding="utf-8")
    assert "G-SUITE-PINNED" not in LG._collect_evidence(tmp_path / "gate", [str(root)])
    assert "G-SUITE-PINNED" in LG._collect_evidence(tmp_path / "gate", [str(root / "evidence")])
    src = (STACK / "scripts/nt_prelaunch.sh").read_text(encoding="utf-8")
    assert '[ -d "$DEVEV/evidence" ] && DEVEV="$DEVEV/evidence"' in src
    assert '[ -s "$DEVEV/G-SUITE-PINNED.json" ] || die' in src
    assert '[ "$NEV" = "6 6" ]' in src and '[ "$NEV" = "4 4" ]' not in src


def test_G_MAP_K4_the_nt_train_rows_are_judged_by_K4_not_the_contribution_sum(LG):
    """MEASURED 2026-10-25 01:18-02:06 (gate on 54c6028, Thor): G-LIVE FAIL "2 of 2 training rows lack the per-class loss
    contributions `map_hires_lc_{cls}_{band}`" -- NavSim has no map GT, so every NT train batch is all-IGNORE
    (smoke: map_hires_n_labelled 0 of 48 windows) and the trainer writes the counted zero with no contributions.
    PI 2026-10-25 ("Profile-scoped K4 check"): on refcv8-nt the rows are judged by K4 itself. RED arms: a labelled
    window, a non-zero loss, no loss row at all -- and the SAME unlabelled row under plain refcv8 still FAILS."""
    nt, r8 = LG.PROFILES["refcv8-nt"], LG.PROFILES["refcv8"]
    assert nt["map_hires"]["train_labels"] == "absent_k4"
    assert "train_labels" not in r8["map_hires"]
    row = {"step": 50, "map_hires": -0.0, "map_hires_n_labelled": 0.0, "map_hires_n_windows": 16.0}
    reasons, det = LG.map_train_row_reasons(nt, [row, dict(row, step=100)])
    assert reasons == [] and det["mode"] == "absent_k4" and det["n_rows"] == 2
    lab = dict(row, step=100, map_hires=0.37, map_hires_n_labelled=3.0)
    reasons, det = LG.map_train_row_reasons(nt, [row, lab])
    assert len(reasons) == 2 and "step 100" in reasons[0] and "K4" in reasons[1]
    assert det["rows_labelled"] == [(100, 3.0)] and det["rows_nonzero_loss"] == [(100, 0.37)]
    reasons, _ = LG.map_train_row_reasons(nt, [dict(row, map_hires=1e-6)])
    assert len(reasons) == 1 and "not exactly 0.0" in reasons[0]
    reasons, _ = LG.map_train_row_reasons(nt, [{"step": 50}])
    assert len(reasons) == 1 and "no training row carries `map_hires`" in reasons[0]
    reasons, _ = LG.map_train_row_reasons(r8, [row])
    assert len(reasons) == 1 and "lack the per-class loss contributions" in reasons[0]
