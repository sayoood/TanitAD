"""test_visens.py -- tests of the kit of card ADD105 (VISENS10K): the gap file VGAPE5 with control e0, the rollout queue, the reader controls e0 to e5 and names, with planted faults.
Synthetic data only; nothing runs on the pod.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest visens/test_visens.py -q -p no:cacheprovider --basetemp /f/TanitAD-scratch/v8proof/visens/tmp_pt/<new name>"""
import copy
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_ens_gapfile as MEG                # noqa: E402
import read_visens10k as RE                   # noqa: E402

QUEUE = HERE / "pod_visens10k_queue.sh"
Q104 = HERE.parent / "visseed" / "pod_visseed10k_queue.sh"
Q102 = HERE.parent / "visrule" / "pod_visrule10k_queue.sh"
BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
W16 = "28bc31f21ecb4ccb"
FITS = MEG.FITS


def bash_exe():
    return shutil.which("bash")


# ------------------------------------------------------------------------------------------------ the gap file and control e0
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("ens")
    J, dumps, shas, r102, r104 = MEG.synthetic_world(str(base / "dumps"))
    jp = base / "join.npz"
    np.savez(str(jp), **J)
    p102, p104 = base / "rec102.json", base / "rec104.json"
    p102.write_text(json.dumps(r102), encoding="utf-8")
    p104.write_text(json.dumps(r104), encoding="utf-8")
    dumps_, shas_, rr102, rr104, meta = MEG.load_inputs(str(base / "dumps" / "old"), str(base / "dumps" / "new"), str(p102), str(p104))
    assert shas_ == shas
    rec = MEG.build(J, dumps_, shas_, rr102, rr104, str(base / "clean"), eval_join_sha=MEG.sha_file(str(jp)), meta=meta)
    return {"base": base, "J": J, "dumps": dumps, "shas": shas, "r102": r102, "r104": r104, "rec": rec, "join": jp, "p102": p102, "p104": p104, "meta": meta}


def test_clean_build_passes_e0_has_the_layout_of_the_gap_files_of_card_add104_and_the_t0_diagnostic(world):
    rec = world["rec"]
    assert rec["e0"]["PASS"] and all(rec["e0"]["clauses"].values()), rec["e0"]
    z = MEG.load_npz(str(world["base"] / "clean" / "VGAPE5.npz"))
    assert sorted(z) == sorted(["clip_index", "frame", "gap", "has_out", "clips", "seed"]) and z["gap"].dtype == np.float64 and z["has_out"].dtype == bool
    assert z["clip_index"].dtype == np.int64 and z["frame"].dtype == np.int64 and int(z["seed"]) == -1
    n = int(np.asarray(world["J"]["n"]).sum())
    assert len(z["gap"]) == n and np.array_equal(np.isfinite(z["gap"]), z["has_out"])
    assert [str(c) for c in z["clips"]] == [str(c) for c in world["J"]["clip"]]
    # the layout equals the layout of the other gap files of the family (same keys and dtypes as the build of card ADD102)
    sys.path.insert(0, str(HERE.parent / "visrule"))
    import make_vis_gapfile as MV
    arr1 = MV.build_arrays(world["J"], world["dumps"][0], MV.frame_table(world["J"], world["dumps"][0]), 0)
    assert sorted(arr1) == sorted(z) and all(np.asarray(arr1[k]).dtype == z[k].dtype for k in arr1)
    t0 = rec["t0_gap_error_diagnostic"]
    assert t0["n_rows_with_label_lead_and_output_of_all_five"] > 0 and set(t0["mean_abs_gap_error_m"]) == {"mean_of_five", "fit0", "fit1", "fit2", "fit3", "fit4"}


def test_the_gap_is_the_arithmetic_mean_of_the_five_dumped_gaps_where_all_five_have_one(world):
    z = MEG.load_npz(str(world["base"] / "clean" / "VGAPE5.npz"))
    J, D = world["J"], world["dumps"]
    off, n = np.asarray(J["off"]), np.asarray(J["n"])
    n_checked = n_none = 0
    for c in range(len(n)):
        for f in range(int(n[c])):
            r = int(off[c]) + f
            vals = []
            for s in FITS:
                m = (np.asarray(D[s]["cid"]) == c) & (np.asarray(D[s]["frame"]) == f)
                vals.append(float(np.asarray(D[s]["gap"])[m][0]))
            if all(np.isfinite(vals)):
                assert z["has_out"][r] and abs(z["gap"][r] - sum(vals) / 5.0) < 1e-12
                n_checked += 1
            else:
                assert not z["has_out"][r] and np.isnan(z["gap"][r])
                n_none += 1
    assert n_checked > 20 and n_none >= 16
    # a head output above 60 m is kept (no clipping, no map, no minimum)
    assert float(z["gap"][z["has_out"]].max()) > 60.0


def test_each_planted_fault_of_each_clause_turns_e0_red(world):
    pf = MEG.planted_faults(world["J"], world["dumps"], world["shas"], world["r102"], world["r104"])
    assert len(pf) >= 20 and all(pf.values()), {k: v for k, v in pf.items() if not v}


def test_e0_clause_a_names_the_dump_hashes_of_card_add102_for_fit_seeds_0_and_1(world):
    J, D, sh = world["J"], world["dumps"], world["shas"]
    arr = MEG.build_arrays(J, D)
    f = MEG.facts_of(J, D, sh, world["r102"], world["r104"], arr)
    assert MEG.e0_check(f)["PASS"] and f["hash_named"]["fit0"]["record_of_card"] == "ADD102" and f["hash_named"]["fit2"]["record_of_card"] == "ADD104"
    r = copy.deepcopy(world["r102"])
    r["files"]["VGAP_s1"]["dump_sha256"] = "ab" * 32
    c = MEG.e0_check(MEG.facts_of(J, D, sh, r, world["r104"], arr))
    assert not c["PASS"] and not c["clauses"]["a_dumps_of_fit_seeds_0_1_have_the_hashes_of_the_record_of_card_ADD102"] and c["clauses"]["b_dumps_of_fit_seeds_2_3_4_have_the_hashes_of_the_record_of_card_ADD104"]


def test_e0_clause_b_hashes_of_card_add104_and_its_x0_x1(world):
    J, D, sh = world["J"], world["dumps"], world["shas"]
    arr = MEG.build_arrays(J, D)
    r = copy.deepcopy(world["r104"])
    r["files"]["VGAPF_s3"]["dump_sha256"] = "cd" * 32
    c = MEG.e0_check(MEG.facts_of(J, D, sh, world["r102"], r, arr))
    assert not c["PASS"] and not c["clauses"]["b_dumps_of_fit_seeds_2_3_4_have_the_hashes_of_the_record_of_card_ADD104"]
    for k in ("x0", "x1"):
        r = copy.deepcopy(world["r104"])
        r[k] = {"PASS": False}
        c = MEG.e0_check(MEG.facts_of(J, D, sh, world["r102"], r, arr))
        assert not c["PASS"] and not c["clauses"]["b_record_of_card_ADD104_reads_x0_and_x1_PASS"]


def test_e0_clause_c_five_dumps_on_the_same_rows(world):
    J, D, sh = world["J"], world["dumps"], world["shas"]
    d2 = {s: dict(D[s]) for s in FITS}
    d2[2]["frame"] = np.where(np.arange(len(d2[2]["frame"])) == 8, d2[2]["frame"][9], d2[2]["frame"])           # one frame twice, one frame absent
    arr = MEG.build_arrays(J, d2)
    c = MEG.e0_check(MEG.facts_of(J, d2, sh, world["r102"], world["r104"], arr))
    assert not c["PASS"] and not c["clauses"]["c_the_five_dumps_are_on_the_same_rows"]


def test_e0_clause_d_tolerance_is_1e6_m_on_every_row(world):
    J, D, sh = world["J"], world["dumps"], world["shas"]
    arr = MEG.build_arrays(J, D)
    for delta, want in ((5e-7, True), (2e-6, False)):
        a2 = dict(arr)
        g = arr["gap"].copy()
        i = int(np.flatnonzero(arr["has_out"])[-1])
        g[i] += delta
        a2["gap"] = g
        c = MEG.e0_check(MEG.facts_of(J, D, sh, world["r102"], world["r104"], a2))
        assert c["PASS"] is want, (delta, c)
    # the deviation on the very first row with an output counts too (every row, not a sample)
    a2 = dict(arr)
    g = arr["gap"].copy()
    g[int(np.flatnonzero(arr["has_out"])[0])] += 2e-6
    a2["gap"] = g
    assert not MEG.e0_check(MEG.facts_of(J, D, sh, world["r102"], world["r104"], a2))["PASS"]


def test_e0_clause_e_output_flag_exactly_when_all_five_have_one(world):
    J, D, sh = world["J"], world["dumps"], world["shas"]
    arr = MEG.build_arrays(J, D)
    a2 = dict(arr)
    h = arr["has_out"].copy()
    h[int(np.flatnonzero(~arr["has_out"])[0])] = True                                     # a flag where one dump has none (the gap there is NaN)
    a2["has_out"] = h
    c = MEG.e0_check(MEG.facts_of(J, D, sh, world["r102"], world["r104"], a2))
    assert not c["PASS"] and not c["clauses"]["e_output_flag_exactly_when_all_five_have_one"]
    a3 = dict(arr)
    g = arr["gap"].copy()
    i = int(np.flatnonzero(~arr["has_out"])[0])
    g[i] = 10.0                                                                              # an output where one dump has none
    a3["gap"] = g
    a3["has_out"] = arr["has_out"].copy()
    a3["has_out"][i] = True
    assert not MEG.e0_check(MEG.facts_of(J, D, sh, world["r102"], world["r104"], a3))["PASS"]


def test_the_record_of_card_add102_names_the_dump_hashes_in_its_files_entries():
    """section 0 of the DEFS: read from the CODE of the tool of card ADD102 (not from a result file): files.VGAP_s<k>.dump_sha256 is written by measure_file"""
    t = (HERE.parent / "visrule" / "make_vis_gapfile.py").read_text(encoding="utf-8")
    assert '"dump_sha256": sha_file(dump_path)' in t and 'rec["files"][nm] = f' in t and 'nm = "VGAP_s%d" % s' in t
    t4 = (HERE.parent / "visseed" / "make_seed_gapfile.py").read_text(encoding="utf-8")
    assert 'nm = "VGAPF_s%d" % s' in t4 and '"dump_sha256": sha_file(dump_path)' in t4 and 'rec["x0"]' in t4 and 'rec["x1"]' in t4


def test_main_through_files_refusals_and_arguments(world, tmp_path):
    b = world["base"]
    out = tmp_path / "o1"
    args = ["--eval-join", str(world["join"]), "--out-dir", str(out), "--old-dump-root", str(b / "dumps" / "old"), "--new-dump-root", str(b / "dumps" / "new"),
            "--rec102", str(world["p102"]), "--rec104", str(world["p104"]), "--expect-clips", "4"]
    assert MEG.main(args) == 0
    rec = json.loads((out / "VISENS_GAPFILES_RECORD.json").read_text(encoding="utf-8"))
    assert rec["e0"]["PASS"] and rec["files"]["VGAPE5"]["file_sha256"] == MEG.sha_file(str(out / "VGAPE5.npz")) and rec["eval_join_sha256"] == MEG.sha_file(str(world["join"]))
    assert rec["inputs"]["record_add102"]["sha256"] == MEG.sha_file(str(world["p102"])) and sorted(rec["files"]["VGAPE5"]["dump_sha256"]) == ["fit0", "fit1", "fit2", "fit3", "fit4"]
    with pytest.raises(SystemExit):
        MEG.main(args)                                                                       # an existing out dir is refused
    assert MEG.main(["--out-dir", str(tmp_path / "o2")]) == 2
    assert MEG.main(args[:-1] + ["141"]) == 2                                                  # the join has 4 clips, 141 expected
    with pytest.raises(SystemExit):
        MEG.main(["--eval-join", str(world["join"]), "--out-dir", str(tmp_path / "o3"), "--old-dump-root", str(tmp_path / "nowhere"), "--new-dump-root", str(b / "dumps" / "new"),
                  "--rec102", str(world["p102"]), "--rec104", str(world["p104"]), "--expect-clips", "4"])


def test_a_failed_e0_writes_no_t0_diagnostic_and_the_queue_refuses_it(world, tmp_path):
    b = world["base"]
    r104 = copy.deepcopy(world["r104"])
    r104["x1"] = {"PASS": False}
    rec = MEG.build(world["J"], world["dumps"], world["shas"], world["r102"], r104, str(tmp_path / "bad"), eval_join_sha="1" * 64)
    assert not rec["e0"]["PASS"] and "t0_gap_error_diagnostic" not in rec
    assert (tmp_path / "bad" / "VISENS_GAPFILES_RECORD.json").is_file()


# ------------------------------------------------------------------------------------------------ the rollout queue
def fn_text(t):
    i0 = t.index("rollout() {\n")
    return t[i0:t.index("\n}\n", i0) + 3]


def dry_commands(q, seeds="ALL"):
    env = dict(os.environ, DRY="1", GR_SEED=seeds)
    out = subprocess.run([bash_exe(), str(q)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=100).stdout
    return [x for x in out.splitlines() if x.startswith("CMD: ")], out


def test_queue_rollout_function_is_the_one_of_cards_add104_and_add102_and_the_header_facts_hold():
    q, q4, q2 = QUEUE.read_text(encoding="utf-8"), Q104.read_text(encoding="utf-8"), Q102.read_text(encoding="utf-8")
    assert hashlib.sha256(q4.encode("utf-8")).hexdigest()[:16] == "ba9c4bf35430e42f" and hashlib.sha256(q2.encode("utf-8")).hexdigest()[:16] == "6321d4e6e12c0f47"
    assert fn_text(q) == fn_text(q4) == fn_text(q2)
    assert "\r" not in q and not [w for w in BANNED if w in q]
    for bad in ("VISSEED", "visseed", "VISREP", "visrep", "VISRULE10K_", "queue_visrule", "$G2", "VGAP1", "VGAPF", "pKD5", "rollout I10RQ50KF5P30Hb20w", "rollout I10RQ50KF5P30Hb20v", "rollout I10RQ50KF5P30Hb20f"):
        assert bad not in q, bad
    for need in ("GP=${GR_ROOT:-/workspace/v8proof/visens10k}", "CHK=${GR_CHECK:-/workspace/v8proof/visrule10k/check_v}", '"$GP/VISENS10K_STOP$PS.txt"', "START_VISENS10K$PS.txt", "VISENS10K_DONE$PS.txt",
                 "queue_visens$PS.log", '[ "$got" = "%s" ]' % W16, 'grep -q "%s" "$CHK/CHECK_WRAPPER_V.hashes"' % W16, "ZZ CHECK-WRAPPER-V PASS",
                 '"VGAPE5" "e0")', "VISENS_GAPFILES_RECORD.json", "I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20vVD5c I10RQ50KF5P30Hb20wVD5cx", 'NAMES="I10RQ50KF5P30Hb20eE5c"',
                 'if [ "$DRY" = 1 ]; then echo "DRY DONE"; exit 0; fi'):
        assert need in q, need


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_dry_run_has_two_commands_with_the_right_arguments_and_syntax_is_valid():
    assert subprocess.run([bash_exe(), "-n", str(QUEUE)]).returncode == 0
    cmds, out = dry_commands(QUEUE)
    assert len(cmds) == 2, out[-800:]
    for k, c in zip((0, 1), cmds):
        assert "v8f_t1_eval_reader_rule_rzgfhqetjv.py" in c and "--form RR" in c and "--rule-join" not in c and "--vis-closing" in c
        assert "--rule-every 5" in c and "--preroll-steps 30" in c and "--kappa-max 0.2" in c and "--readout causal_g" in c and "--eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz" in c
        assert ("/I10RQ50KF5P30Hb20eE5c_s%d " % k) in c and ("--reader /workspace/v8proof/integ10k/lr16/TK16_s%d/" % k) in c and ("QMAP_LR16_s%d_q50" % k) in c
        assert "--vis-gap-file /workspace/v8proof/visens10k/gapfiles/VGAPE5.npz" in c and c.count("--vis-gap-file") == 1
    for s in ("0", "1"):
        one, _ = dry_commands(QUEUE, s)
        assert len(one) == 1 and ("_s%s " % s) in one[0] and ("TK16_s%s" % s) in one[0] and ("TK16_s%s" % ("1" if s == "0" else "0")) not in one[0]


def test_the_arguments_of_the_rollouts_are_those_of_the_vd5c_call_of_the_add102_queue():
    q0 = Q102.read_text(encoding="utf-8")
    base = [x for x in q0.splitlines() if "rollout I10RQ50KF5P30Hb20vVD5c RR" in x][0]
    env0 = base.split(" VGAP=")[0].strip()
    mine = [x for x in QUEUE.read_text(encoding="utf-8").splitlines() if "rollout I10RQ50KF5P30Hb20eE5c" in x]
    assert len(mine) == 1 and mine[0].strip().startswith(env0) and ' VGAP="$GF/VGAPE5.npz" VCLOSE=1 rollout I10RQ50KF5P30Hb20eE5c RR "$k" "$k"' in mine[0]
    assert 'VGAP="$GF/VGAP_s$k.npz" VCLOSE=1' in base


def test_the_queue_admits_the_gap_file_only_with_the_e0_pass_and_the_hashes():
    """the admit function of the queue run on synthetic records: e0 PASS, the file hash and the join hash are what it reads"""
    t = QUEUE.read_text(encoding="utf-8")
    i0 = t.index("admit() {")
    body = t[i0:t.index("\n}\n", i0) + 3]
    py = body.split("<<'PYEOF'\n")[1].split("\nPYEOF")[0]
    import tempfile
    with tempfile.TemporaryDirectory(dir=str(HERE / "tmp_pt")) as td:
        td = Path(td)
        gf = td / "VGAPE5.npz"
        gf.write_bytes(b"gapfile")
        ej = td / "join.npz"
        ej.write_bytes(b"join")
        rec = {"e0": {"PASS": True}, "eval_join_sha256": hashlib.sha256(b"join").hexdigest(), "files": {"VGAPE5": {"file_sha256": hashlib.sha256(b"gapfile").hexdigest()}}}

        def run(r):
            rp = td / "rec.json"
            rp.write_text(json.dumps(r), encoding="utf-8")
            p = subprocess.run([sys.executable, "-c", py, str(rp), str(td), str(ej), "VGAPE5", "e0"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")
            return p.returncode, p.stdout.strip()
        assert run(rec) == (0, "ADMIT-OK joins 1")
        assert run(dict(rec, e0={"PASS": False}))[0] == 1
        assert run(dict(rec, eval_join_sha256="0" * 64))[1] == "RECORD-IS-FOR-ANOTHER-EVAL-JOIN"
        assert run(dict(rec, files={"VGAPE5": {"file_sha256": "0" * 64}}))[1] == "JOIN-HASH-DIFFERS VGAPE5"
        r2 = dict(rec)
        del r2["e0"]
        assert run(r2)[1] == "E0-NOT-PASS"


# ------------------------------------------------------------------------------------------------ names
def iv(lo, hi):
    return {"interval": [lo, hi], "point": 0.5 * (lo + hi)}


def payload(rr=0.0954, rlc=0.0, c_rr=(-0.05, -0.01), c_rlc=(0.0, 0.004), c_vd=(-0.03, 0.001), f_rr=(-0.02, 0.03)):
    pr = {RE.ARM + "-VS-RR": {"C": iv(*c_rr), "U": iv(-0.01, 0.01), "F": iv(*f_rr)}, RE.ARM + "-VS-RLc": {"C": iv(*c_rlc), "U": iv(-0.01, 0.01), "F": iv(0.0, 0.02)},
          RE.ARM + "-VS-VD5c": {"C": iv(*c_vd), "U": iv(-0.01, 0.01), "F": iv(-0.01, 0.01)}}
    lv = {k: {"C": c} for k, c in (("RR", rr), ("RLc", rlc), ("VD5c", 0.0169), ("E5c", 0.02))}
    return {"levels": lv, "levels_per_seed": {k: {"C": [v["C"], v["C"]], "U": [0.0, 0.0], "F": [0.0, 0.0]} for k, v in lv.items()}, "pairs": pr, "families": {}, "strategic": {"pairs": {}},
            "single_fit_contact": {"note": "x"}, "t0_gap_error_diagnostic": {"n_rows_with_label_lead_and_output_of_all_five": 7, "mean_abs_gap_error_m": {"mean_of_five": 1.25}},
            "falling_behind_against_RLc": {RE.ARM: iv(0.0, 0.02)["interval"]}, "rollout_shares": {}}


def test_contact_names_against_rr_with_the_edges():
    nm = lambda c: RE.names_of("E5c", list(c), [0.0, 0.004], [-0.03, 0.001], [-0.02, 0.03], 0.0954, 0.0)["E5c_CONTACT"]       # noqa: E731
    assert nm((-0.05, -0.01)) == "VISENS10K-E5c-CUTS-CONTACT"
    assert nm((-0.05, 0.0)) == "VISENS10K-E5c-CONTACT-NOT-SEPARATED-FROM-RR"                 # an upper end of exactly 0 is not below 0
    assert nm((0.0, 0.05)) == "VISENS10K-E5c-CONTACT-NOT-SEPARATED-FROM-RR"                  # a lower end of exactly 0 is not above 0
    assert nm((0.0001, 0.05)) == "VISENS10K-E5c-MORE-CONTACT"
    assert nm((-0.05, 0.05)) == "VISENS10K-E5c-CONTACT-NOT-SEPARATED-FROM-RR"


def test_half_the_gap_names_against_rlc_with_the_edges():
    nm = lambda c, rr=0.0954: RE.names_of("E5c", [-0.05, -0.01], list(c), [-0.03, 0.001], [-0.02, 0.03], rr, 0.0)["E5c_HALF"]       # noqa: E731
    assert nm((0.0, 0.0477)) == "VISENS10K-E5c-WITHIN-HALF-THE-GAP"                          # upper end at H: within
    assert nm((0.0, 0.0478)) == "VISENS10K-E5c-HALF-THE-GAP-NOT-DECIDED"
    assert nm((0.0477, 0.09)) == "VISENS10K-E5c-HALF-THE-GAP-NOT-DECIDED"                    # lower end at H: not above H
    assert nm((0.0478, 0.09)) == "VISENS10K-E5c-HALF-THE-GAP-NOT-REACHED"
    assert nm((0.0, 0.025), rr=0.05) == "VISENS10K-E5c-WITHIN-HALF-THE-GAP" and nm((0.0, 0.026), rr=0.05) == "VISENS10K-E5c-HALF-THE-GAP-NOT-DECIDED"   # H follows the point values


def test_names_against_vd5c_and_falling_behind():
    n = lambda vd, f: RE.names_of("E5c", [-0.05, -0.01], [0.0, 0.004], list(vd), list(f), 0.0954, 0.0)       # noqa: E731
    assert n((-0.03, -0.001), (0, 0.01))["E5c_VD5C"].endswith("-LESS-CONTACT-THAN-VD5c") and n((0.001, 0.03), (0, 0.01))["E5c_VD5C"].endswith("-MORE-CONTACT-THAN-VD5c")
    assert n((-0.03, 0.0), (0, 0.01))["E5c_VD5C"].endswith("-CONTACT-NOT-SEPARATED-FROM-VD5c") and n((0.0, 0.03), (0, 0.01))["E5c_VD5C"].endswith("-CONTACT-NOT-SEPARATED-FROM-VD5c")
    assert n((0, 0), (0.001, 0.1))["E5c_PRICE"] == "VISENS10K-E5c-FALLS-BEHIND-MORE" and n((0, 0), (-0.1, -0.001))["E5c_PRICE"] == "VISENS10K-E5c-FALLS-BEHIND-LESS"
    assert n((0, 0), (0.0, 0.1))["E5c_PRICE"] == "VISENS10K-E5c-FALLING-BEHIND-NOT-SEPARATED" and n((0, 0), (-0.1, 0.0))["E5c_PRICE"] == "VISENS10K-E5c-FALLING-BEHIND-NOT-SEPARATED"


def test_the_card_name_is_the_first_that_applies_in_the_order_of_the_card():
    mk = lambda c, h: {"E5c_CONTACT": "VISENS10K-E5c-" + c, "E5c_HALF": "VISENS10K-E5c-" + h}                # noqa: E731
    W_, N_, D_ = "WITHIN-HALF-THE-GAP", "HALF-THE-GAP-NOT-REACHED", "HALF-THE-GAP-NOT-DECIDED"
    cut = "CUTS-CONTACT"
    assert RE.card_name(mk(cut, W_)) == "VISENS10K-WITHIN-HALF-THE-GAP"
    assert RE.card_name(mk(cut, N_)) == "VISENS10K-HALF-THE-GAP-NOT-REACHED"
    assert RE.card_name(mk(cut, D_)) == "VISENS10K-HALF-THE-GAP-NOT-DECIDED"
    for notcut in ("CONTACT-NOT-SEPARATED-FROM-RR", "MORE-CONTACT"):                     # not cutting comes first whatever the half-the-gap name is
        for h in (W_, N_, D_):
            assert RE.card_name(mk(notcut, h)) == "VISENS10K-DOES-NOT-CUT"
    assert RE.card_name(mk(cut, W_)) != RE.card_name(mk(cut, D_))


def test_all_names_card_name_and_the_second_derivation_from_the_intervals():
    for kw, want in ((dict(), "VISENS10K-WITHIN-HALF-THE-GAP"), (dict(c_rr=(-0.02, 0.01)), "VISENS10K-DOES-NOT-CUT"), (dict(c_rlc=(0.05, 0.08)), "VISENS10K-HALF-THE-GAP-NOT-REACHED"),
                     (dict(c_rlc=(0.0, 0.06)), "VISENS10K-HALF-THE-GAP-NOT-DECIDED")):
        nm = RE.all_names(payload(**kw))
        assert nm["CARD"] == want and RE.card_check(payload(**kw), nm)["PASS"]
    nm = RE.all_names(payload())
    assert sorted(nm) == ["CARD", "E5c_CONTACT", "E5c_HALF", "E5c_PRICE", "E5c_VD5C"]
    assert not RE.card_check(payload(), dict(nm, CARD="VISENS10K-DOES-NOT-CUT"))["PASS"]
    h = RE.h_sensitivity(payload())
    assert abs(h["H_point_values"] - 0.0477) < 1e-12 and h["differs"] is False


def test_single_fit_table_includes_f_arms_only_when_told_and_says_so():
    ls = {k: {"C": [0.02, 0.04]} for k in ("E5c", "VD5c", "VD5cx", "F2", "F3", "F4")}
    t = RE.single_fit_table(ls, False)
    assert sorted(t["arms"]) == ["E5c", "VD5c", "VD5cx"] and "left out" in t["note"] and abs(t["arms"]["VD5c"]["pooled"] - 0.03) < 1e-12
    t = RE.single_fit_table(ls, True)
    assert sorted(t["arms"]) == ["E5c", "F2", "F3", "F4", "VD5c", "VD5cx"] and "included" in t["note"]


def test_add104_controls_file_is_read_only_for_its_pass_flags(tmp_path):
    ok = {"card": "PREREG_V8F_PROOF_ADD104", "controls_all_pass": True, "controls": {k: {"PASS": True} for k in RE.ADD104_CONTROLS}}

    def w(j, nm):
        p = tmp_path / nm
        p.write_text(json.dumps(j), encoding="utf-8")
        return str(p)
    assert RE.add104_ok(w(ok, "a.json"))[0] is True
    assert RE.add104_ok("")[0] is False and RE.add104_ok(str(tmp_path / "nope.json"))[0] is False
    bad = copy.deepcopy(ok)
    bad["controls"]["x3"]["PASS"] = False
    assert RE.add104_ok(w(bad, "b.json"))[0] is False
    miss = copy.deepcopy(ok)
    del miss["controls"]["x6"]
    assert RE.add104_ok(w(miss, "c.json"))[0] is False
    other = dict(ok, card="PREREG_V8F_PROOF_ADD103")
    assert RE.add104_ok(w(other, "d.json"))[0] is False
    flag = dict(ok, controls_all_pass=False)
    assert RE.add104_ok(w(flag, "e.json"))[0] is False
    (tmp_path / "f.json").write_text("{not json", encoding="utf-8")
    assert RE.add104_ok(str(tmp_path / "f.json"))[0] is False


# ------------------------------------------------------------------------------------------------ controls e1 to e5
def test_control_e1_e3_e4_e5():
    shas = {"E5c_s0": W16 + "0" * 48, "E5c_s1": W16 + "1" * 48}
    assert RE.ctl_e1(W16, "x\nZZ CHECK-WRAPPER-V PASS\n", "..." + W16, shas)["PASS"]
    assert not RE.ctl_e1("0" * 16, "ZZ CHECK-WRAPPER-V PASS\n", W16, shas)["PASS"]
    assert not RE.ctl_e1(W16, "ZZ CHECK-WRAPPER-V FAIL\n", W16, shas)["PASS"]
    assert not RE.ctl_e1(W16, "ZZ CHECK-WRAPPER-V PASS\n", "no hash here", shas)["PASS"]
    assert not RE.ctl_e1(W16, "ZZ CHECK-WRAPPER-V PASS\n", W16, dict(shas, E5c_s0="1" * 64))["PASS"]            # a run text that names another wrapper
    assert not RE.ctl_e1(W16, "ZZ CHECK-WRAPPER-V PASS\n", W16, dict(shas, E5c_s1=None))["PASS"]              # a run text that names none
    assert RE.ctl_e3({"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0169})["PASS"] and RE.ctl_e3({"RR": 0.09544, "RLc": 0.00004, "VD5c": 0.01694})["PASS"]
    for bad in ({"RR": 0.0955, "RLc": 0.0, "VD5c": 0.0169}, {"RR": 0.0954, "RLc": 0.0001, "VD5c": 0.0169}, {"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0170}, {"RR": 0.0954, "RLc": 0.0}):
        assert not RE.ctl_e3(bad)["PASS"]
    assert RE.ctl_e4("25 passed in 3.0s")["PASS"] and not RE.ctl_e4("2 failed, 25 passed")["PASS"] and not RE.ctl_e4("")["PASS"] and not RE.ctl_e4("1 error")["PASS"] and not RE.ctl_e4("0 passed")["PASS"]
    assert RE.ctl_e5(True, True, True)["PASS"] and not RE.ctl_e5(True, True, None)["PASS"] and not RE.ctl_e5(False, True, True)["PASS"] and not RE.ctl_e5(True, False, True)["PASS"]
    assert RE.CONTROLS == ("e0", "e1", "e2", "e3", "e4", "e5") and set(RE.C_EXPECTED) == {"RR", "RLc", "VD5c"}


def rows(ci, j, pres, g, c=None):
    ci, j, pres, g = np.asarray(ci), np.asarray(j), np.asarray(pres, bool), np.asarray(g, float)
    return (ci.astype(np.int64), j.astype(np.int64), pres, g, np.full(len(g), -0.4) if c is None else np.asarray(c, float))


def test_control_e2_on_planted_rows():
    ref = {(0, 4): 10.0, (0, 5): 11.0, (0, 3): None, (1, 4): 20.0}
    v0 = {(0, 3): [7.0], (0, 4): [9.0], (0, 5): [9.5, 9.5004], (1, 4): [19.0]}
    K = ([0, 0, 0, 1], [3, 4, 5, 4], [1, 1, 1, 1])
    good = rows(*K, [7.0, 10.0, 11.0005, 20.0])
    r = RE.w2_one(good, ref, v0)
    assert r["PASS"] and r["n_lead_rows_with_head_output"] == 3 and r["n_lead_rows_without_head_output"] == 1
    assert not RE.w2_one(rows(*K, [7.0, 10.0, 11.002, 20.0]), ref, v0)["PASS"]                  # a gap that differs from the file by 2e-3 m
    assert not RE.w2_one(rows(*K, [7.1, 10.0, 11.0, 20.0]), ref, v0)["PASS"]                    # no output of the file: not V0's gap
    assert not RE.w2_one(rows(*K, [7.0, 9.0, 11.0, 20.0]), ref, v0)["PASS"]                     # V0's gap where the file has an output
    assert not RE.w2_one(rows(*K, [7.0, 10.0, 11.0, 20.0], c=[0, 0, 0, 0]), ref, v0)["PASS"]     # the closing input is 0 everywhere
    assert RE.w2_one(rows(*K, [7.0, 10.0, 11.0, 20.0], c=[0, 0, 0, 0.5]), ref, v0)["PASS"]
    assert not RE.w2_one(rows([0, 0], [3, 4], [1, 1], [7.0, 10.0]), ref, {(0, 4): [9.0]})["PASS"]  # no V0 row for a lead step with no output
    assert not RE.w2_one(rows([0, 0], [3, 4], [0, 0], [7.0, 10.0]), ref, v0)["PASS"]            # no lead row at all
    exp = {"vis_gap_file_sha256": "a" * 64, "vis_closing": 1}
    assert RE.w2_one(good, ref, v0, exp, {"vis_gap_file_sha256": "a" * 64, "vis_closing": 1})["PASS"]
    assert not RE.w2_one(good, ref, v0, exp, {"vis_gap_file_sha256": "b" * 64, "vis_closing": 1})["PASS"]
    assert not RE.w2_one(good, ref, v0, exp, {"vis_gap_file_sha256": "a" * 64, "vis_closing": 0})["PASS"]
    assert not RE.w2_one(good, ref, v0, exp, {})["PASS"]


def write_logs(pull, run, m, tick, mid, meta):
    d = Path(pull) / ("I10%s_s%d" % (run, m)) / "ticks"
    d.mkdir(parents=True, exist_ok=True)
    for nm, r in (("rule_log_I10%s_s%d.npz" % (run, m), tick), ("rule_log_mid_I10%s_s%d.npz" % (run, m), mid)):
        np.savez(str(d / nm), ci=r[0], j=r[1], present=r[2], g_src=r[3], c_src=r[4], **{k: np.asarray(v) for k, v in meta.items()})
    (d / ("run_I10%s_s%d.txt" % (run, m))).write_text("arm x wrapper sha256 %s started t\n" % (W16 + "0" * 48), encoding="utf-8")


class Ns:
    pass


def synth_pull(tmp_path, world, spoil=None, closing_zero=None, meta_bad=None, spoil_noout=None):
    """the rule logs of V0 and E5c for both model seeds, made from the gap file the build wrote: the rule gets the file's gap where it has an output, the reader's own gap (= V0's) elsewhere"""
    pull = tmp_path / "pull"
    n = np.asarray(world["J"]["n"])
    ref = RE.gapfile_ref(str(world["base"] / "clean" / "VGAPE5.npz"))
    for m in (0, 1):
        for arm in ("V0", "E5c"):
            run = RE.V0_RUN if arm == "V0" else RE.RUN[arm]
            cols = {0: [[], [], [], []], 1: [[], [], [], []]}
            for c in range(len(n)):
                for j in range(int(n[c])):
                    own = np.float32(20.0 + 0.1 * j + c)
                    h = ref.get((c, j)) if arm == "E5c" else None
                    g = np.float32(h) if h is not None else own
                    t = cols[j % 2]
                    t[0].append(c), t[1].append(j), t[2].append(g)
                    t[3].append(0.0 if arm == "V0" else -0.5)
                    if arm == "V0" and j % 2 == 0:                                   # a frame logged twice with the float noise of the reader
                        t[0].append(c), t[1].append(j), t[2].append(np.float32(own + 0.0004)), t[3].append(0.0)
            if spoil == (arm, m):
                cols[0][2][5] = cols[0][2][5] + 0.01
            if spoil_noout == (arm, m):
                cols[0][2][0] = cols[0][2][0] + 0.01                              # frame 0 of clip 0: the file has no output there
            if closing_zero == (arm, m):
                cols[0][3] = [0.0] * len(cols[0][3])
                cols[1][3] = [0.0] * len(cols[1][3])
            exp = "" if arm == "V0" else world["rec"]["files"]["VGAPE5"]["file_sha256"]
            if meta_bad == (arm, m):
                exp = "f" * 64
            meta = {"vis_gap_file_sha256": exp, "vis_closing": 0 if arm == "V0" else 1}
            tick, mid = [rows(cols[k][0], cols[k][1], [True] * len(cols[k][0]), cols[k][2], cols[k][3]) for k in (0, 1)]
            write_logs(pull, run, m, tick, mid, meta)
    return pull


def test_control_e2_end_to_end_on_the_gap_file_on_disk_with_planted_faults(world, tmp_path):
    a = Ns()
    a.gap_dir = str(world["base"] / "clean")
    a.pull = str(synth_pull(tmp_path / "a", world))
    c2 = RE.e2_from_files(a, world["rec"])
    assert c2["PASS"], c2
    assert sorted(c2["per_run"]) == ["E5c_s0", "E5c_s1"]
    assert all(v["n_lead_rows_with_head_output"] > 0 and v["n_lead_rows_without_head_output"] > 0 and v["share_of_lead_steps_with_nonzero_closing"] > 0 for v in c2["per_run"].values())
    for label, kw in (("spoil", dict(spoil=("E5c", 1))), ("closing_zero", dict(closing_zero=("E5c", 0))), ("meta", dict(meta_bad=("E5c", 0))), ("spoil_noout", dict(spoil_noout=("E5c", 0)))):
        a.pull = str(synth_pull(tmp_path / label, world, **kw))
        assert not RE.e2_from_files(a, world["rec"])["PASS"], label
    a.pull = str(synth_pull(tmp_path / "recwrong", world))
    wrong = copy.deepcopy(world["rec"])
    wrong["files"]["VGAPE5"]["file_sha256"] = "e" * 64                                       # the gap file the rule log names is not the one of the record
    assert not RE.e2_from_files(a, wrong)["PASS"]


def test_control_e0_in_the_reader_end_to_end_and_planted_faults(world, tmp_path):
    b = world["base"]
    gdir = str(b / "clean")
    ejs = MEG.sha_file(str(world["join"]))
    now = MEG.check_now(str(world["join"]), gdir + "/VGAPE5.npz", str(b / "dumps" / "old"), str(b / "dumps" / "new"), str(world["p102"]), str(world["p104"]))
    assert RE.ctl_e0(world["rec"], gdir, ejs, now)["PASS"]
    assert not RE.ctl_e0(world["rec"], gdir, "8" * 64, now)["PASS"]                          # another evaluation join
    assert not RE.ctl_e0(None, gdir, ejs, now)["PASS"] and not RE.ctl_e0(world["rec"], gdir, ejs, None)["PASS"]
    r2 = copy.deepcopy(world["rec"])
    r2["e0"]["PASS"] = False
    assert not RE.ctl_e0(r2, gdir, ejs, now)["PASS"]
    r3 = copy.deepcopy(world["rec"])
    r3["files"]["VGAPE5"]["file_sha256"] = "7" * 64                                          # the gap file on disk is not the record's
    assert not RE.ctl_e0(r3, gdir, ejs, now)["PASS"]
    # a dump changed after the record was built: the hash the record of card ADD104 names does not match any more
    w2 = tmp_path / "w2"
    shutil.copytree(str(b / "dumps"), str(w2 / "dumps"))
    D3 = dict(world["dumps"][3])
    D3["gap"] = np.asarray(D3["gap"]) + 0.5
    np.savez(str(w2 / "dumps" / "new" / "D_H5" / "reader_s3_score.npz"), **D3)
    now2 = MEG.check_now(str(world["join"]), gdir + "/VGAPE5.npz", str(w2 / "dumps" / "old"), str(w2 / "dumps" / "new"), str(world["p102"]), str(world["p104"]))
    assert not now2[0]["PASS"] and not RE.ctl_e0(world["rec"], gdir, ejs, now2)["PASS"]
    # a record of card ADD104 that was changed after the build (x1 now reads not PASS)
    p104b = tmp_path / "rec104b.json"
    p104b.write_text(json.dumps(dict(world["r104"], x1={"PASS": False})), encoding="utf-8")
    now3 = MEG.check_now(str(world["join"]), gdir + "/VGAPE5.npz", str(b / "dumps" / "old"), str(b / "dumps" / "new"), str(world["p102"]), str(p104b))
    assert not RE.ctl_e0(world["rec"], gdir, ejs, now3)["PASS"]


def test_a_withheld_payload_and_the_names_stage_refuses_with_a_control_failed(tmp_path, capsys):
    ok = {k: {"PASS": True} for k in RE.CONTROLS}
    bad = dict(ok, e2={"PASS": False})
    p = payload()
    r = RE.assemble_controls(bad, p)
    assert r["controls_all_pass"] is False and "payload" not in r and "withheld" in r
    cf = tmp_path / "c.json"
    cf.write_text(json.dumps(r), encoding="utf-8")
    out = tmp_path / "res.json"
    assert RE.main(["--stage", "names", "--controls", str(cf), "--out", str(out)]) == 3 and not out.exists()
    assert "T0 diagnostic" not in capsys.readouterr().out
    cf2 = tmp_path / "c2.json"
    cf2.write_text(json.dumps(RE.assemble_controls(ok, p)), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(cf2), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "T0 diagnostic" in printed and "VISENS10K-WITHIN-HALF-THE-GAP" in printed and "VISENS10K-E5c-CUTS-CONTACT" in printed
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["controls_sha256"] == RE.sha_file(str(cf2)) and res["names"]["CARD"] == "VISENS10K-WITHIN-HALF-THE-GAP" and "E5c_VD5C" in res["names"]
    for drop in RE.CONTROLS:                                                                 # with any one control missing nothing is opened
        cf3 = tmp_path / ("c3_%s.json" % drop)
        cf3.write_text(json.dumps(RE.assemble_controls({k: v for k, v in ok.items() if k != drop}, p)), encoding="utf-8")
        assert RE.main(["--stage", "names", "--controls", str(cf3), "--out", str(tmp_path / ("res3_%s.json" % drop))]) == 3
    old = {"x%d" % i: {"PASS": True} for i in range(7)}                                       # a controls file of card ADD104 is not one of this card
    cf4 = tmp_path / "c4.json"
    cf4.write_text(json.dumps(RE.assemble_controls(old, p)), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(cf4), "--out", str(tmp_path / "res4.json")]) == 3
    cf5 = tmp_path / "c5.json"
    cf5.write_text(json.dumps(dict(RE.assemble_controls(ok, p), card="PREREG_V8F_PROOF_ADD104")), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(cf5), "--out", str(tmp_path / "res5.json")]) == 3


def test_reader_source_carries_the_strategic_family_the_references_the_bootstrap_setting_and_keeps_the_t0_line_for_the_names_stage():
    t = (HERE / "read_visens10k.py").read_text(encoding="utf-8")
    assert "a.n_boot_control = 300" in t and t.index("a = ap.parse_args(argv)") < t.index("a.n_boot_control = 300")
    assert "R.strategic_grid(" in t and "S.pair_read(" in t and '"-VS-VD5c"' in t and 'ARM + "-VS-VD5c"' in t
    assert 'REF_RUNS = {"RR": BANKED["RR"], "RLc": BANKED["RLc"], "VD5c": B102["VD5c"]}' in t and "for ref, rn in REF_RUNS.items()" in t
    assert 'RUN = {ARM: "RQ50KF5P30Hb20eE5c"}' in t and 'B103 = {"VD5cx": "RQ50KF5P30Hb20wVD5cx"}' in t and "C_EXPECTED" in t and "--add104-controls" in t
    body = inspect.getsource(RE.run_controls)
    key = '(rec or {}).get("t0_gap_error_diagnostic")'
    assert body.count(key) == 1 and "log(" not in body.split(key)[1].split("\n")[0] and "print(" not in body.replace("log = lambda s: print(s, flush=True)", "")
    assert "T0 diagnostic" in inspect.getsource(RE.print_sections)
    assert "include_f" in t and "left out" in t


def test_podread_text():
    t = (HERE / "podread_visens10k.sh").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=-1" in t and "VISENS10K_CONTROLS.json" in t and "VISENS10K_RESULT.json" in t and "ALREADY-READ" in t and "QUEUE-STOPPED" in t
    for x in ("eE5c", "fF2", "vV0", "vVD5c", "wVD5cx", "D_H5", "--old-dump-root", "--new-dump-root", "--rec102", "--rec104", "--wrapper-hashes", "--wrapper-file", "VISENS_GAPFILES_RECORD.json", "VGAPE5.npz", "--add104-controls",
              "VISSEED10K_CONTROLS.json", "reader_s0_score.npz", "reader_s4_score.npz"):
        assert x in t, x
    for bad in (" rm ", " mv ", "kill", "pkill", "/workspace/v8proof/visrep10k", "visseed10k/kit", "VISSEED10K_DONE"):
        assert bad not in t, bad
    if bash_exe():
        assert subprocess.run([bash_exe(), "-n", str(HERE / "podread_visens10k.sh")]).returncode == 0


def test_push_and_pull_scripts(tmp_path):
    pu, pl = (HERE / "push_visens.sh").read_text(encoding="utf-8"), (HERE / "pull_visens.sh").read_text(encoding="utf-8")
    assert "set -C" in pu and "lead_pytest.out" in pu and "/workspace/v8proof/visens10k" in pu and "cat $P" not in pu and "ZZ PUSH-VISENS" in pu
    for f in ("make_ens_gapfile.py", "pod_visens10k_queue.sh", "read_visens10k.py", "podread_visens10k.sh"):
        assert f in pu and (HERE / f).is_file()
    assert "pull_add105.tar" in pl and "dry_all.out" in pl and "pod_add105" in pl and "VISENS_GAPFILES_RECORD.json" in pl and "queue_visens_s0.log" in pl and "read" in pl
    assert "head.pt" not in pl and "score.npz" not in pl and "VGAPE5.npz" not in pl
    if bash_exe():
        for f in ("push_visens.sh", "pull_visens.sh"):
            assert subprocess.run([bash_exe(), "-n", str(HERE / f)]).returncode == 0
        # the push refuses without lead_pytest.out; run in a copy of the folder layout so the real one is not touched
        if not (HERE / "lead_pytest.out").exists():
            r = subprocess.run([bash_exe(), str(HERE / "push_visens.sh")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=60)
            assert r.returncode == 2 and "REFUSED" in r.stdout


def test_texts_hold_no_banned_words_and_paths_are_short():
    for f in ("make_ens_gapfile.py", "make_visens_queue.py", "pod_visens10k_queue.sh", "read_visens10k.py", "podread_visens10k.sh", "push_visens.sh", "pull_visens.sh", "VISENS_DEFS.md", "test_visens.py"):
        p = HERE / f
        if p.is_file():
            t = p.read_text(encoding="utf-8")
            if f != "test_visens.py":
                assert not [w for w in BANNED if w in t], f
            assert "\r" not in t
        rel = "F:/TanitAD-scratch/v8proof/visens/" + f
        assert len(rel) < 200


def test_selftests_run():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for args, last in (([str(HERE / "read_visens10k.py"), "--selftest"], "ZZ SELFTEST-READ-VISENS10K ok: True"), ([str(HERE / "make_ens_gapfile.py"), "--selftest"], "ZZ MAKE-ENS-GAPFILE-SELFTEST PASS")):
        r = subprocess.run([sys.executable] + args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=300)
        assert r.stdout.strip().splitlines()[-1] == last, (args, r.stdout[-400:])
