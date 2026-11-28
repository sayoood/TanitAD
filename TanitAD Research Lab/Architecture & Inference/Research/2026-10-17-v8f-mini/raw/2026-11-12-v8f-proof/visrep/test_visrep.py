"""test_visrep.py -- tests of the kit of card ADD103 (VISREP10K): the gap files of the head on one frame and the check of the ADD102 gap files, the queue, the reader controls and names, with planted faults.
Synthetic data only; nothing runs on the pod.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest visrep/test_visrep.py -q -p no:cacheprovider --basetemp /f/TanitAD-scratch/v8proof/visrep/tmp_pt/<new name>"""
import copy
import hashlib
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
import make_rep_gapfile as MRG                # noqa: E402
import read_visrep10k as RE                   # noqa: E402

QUEUE = HERE / "pod_visrep10k_queue.sh"
Q102 = HERE.parent / "visrule" / "pod_visrule10k_queue.sh"
BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
W16 = "28bc31f21ecb4ccb"


def bash_exe():
    return shutil.which("bash")


# ------------------------------------------------------------------------------------------------ the gap files
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("gf")
    J, dumps = MRG.synthetic_all()
    meta = MRG.synthetic_meta(dumps, str(base / "dumps"))
    a102 = MRG.synthetic_add102(str(base / "a102"))
    rec = MRG.build(J, dumps, str(base / "clean"), meta=meta, add102_dir=a102)
    return {"J": J, "dumps": dumps, "meta": meta, "rec": rec, "base": base, "a102": a102}


def test_generator_source_hash_is_the_file_on_disk_and_the_derivation_is_visible():
    src = HERE.parent / "visrule" / "make_vis_gapfile.py"
    gen = (HERE / "gen_rep_gapfile.py").read_text(encoding="utf-8")
    assert hashlib.sha256(src.read_bytes()).hexdigest()[:16] == "97a90ec432bcfc5c" and "97a90ec432bcfc5c" in gen
    t = (HERE / "make_rep_gapfile.py").read_text(encoding="utf-8")
    assert 'DUMP_DIR = "D_H1"' in t and '"VGAP_s%d"' not in t and "VISREP_GAPFILES_RECORD.json" in t and 'CARD = "PREREG_V8F_PROOF_ADD103"' in t


def test_clean_build_passes_every_part_of_w0(world):
    r = world["rec"]
    assert r["v0"]["PASS"] and r["build"]["PASS"] and r["add102"]["PASS"] and r["w0"]["PASS"], r["w0"]
    assert sorted(r["files"]) == ["VGAP1_s0", "VGAP1_s1"] and r["card"] == "PREREG_V8F_PROOF_ADD103"
    assert (world["base"] / "clean" / "VISREP_GAPFILES_RECORD.json").is_file()
    for a in r["files"].values():
        assert a["dump_dir_name"] == "D_H1" and a["max_abs_dev_saved_minus_second_routine"] <= 1e-9 and a["n_rows"] == a["n_join_rows"]


def test_head_on_one_frame_has_an_output_on_every_frame_and_still_builds(world, tmp_path):
    J = world["J"]
    d0 = {s: MRG.synthetic_dump(J, s, first_out=0) for s in MRG.SEEDS}
    meta = MRG.synthetic_meta(d0, str(tmp_path / "dumps"))
    rec = MRG.build(J, d0, str(tmp_path / "c"), meta=meta, add102_dir=world["a102"])
    assert rec["w0"]["PASS"] and all(a["n_without_output"] == 0 for a in rec["files"].values())


def test_each_planted_fault_turns_its_check_red(world, tmp_path):
    vf = MRG.v0_faults(world["J"], world["dumps"])
    assert vf and all(vf.values()), vf
    bf = MRG.build_faults(world["J"], world["dumps"], str(tmp_path / "bf"), world["meta"])
    assert sorted(bf) == ["clip60", "min_label", "scale", "shift", "zero_nooutput"] and all(bf.values()), bf
    os.makedirs(str(tmp_path / "af"))
    af = MRG.add102_faults(str(tmp_path / "af"), world["a102"])
    assert sorted(af) == ["file_changed", "file_missing", "record_hash_changed", "record_missing", "record_wrong_card"] and all(af.values()), af
    rec = MRG.build(world["J"], world["dumps"], str(tmp_path / "noadd"), meta=world["meta"])            # no folder of card ADD102 given: w0 is not PASS
    assert rec["v0"]["PASS"] and rec["build"]["PASS"] and not rec["w0"]["PASS"]


def test_the_model_seed_reads_its_own_fit_seed_and_the_folder_d_h1(world):
    rec = copy.deepcopy(world["rec"])
    rec["files"]["VGAP1_s0"]["dump_file"] = rec["files"]["VGAP1_s0"]["dump_file"].replace("reader_s0", "reader_s1")
    assert not MRG.build_check(rec)["PASS"]
    rec = copy.deepcopy(world["rec"])
    rec["files"]["VGAP1_s1"]["dump_dir_name"] = "D_H5"
    assert not MRG.build_check(rec)["PASS"]
    rec = copy.deepcopy(world["rec"])
    rec["dumps"]["VGAP1_s1"]["e0"]["dump_seed"] = 0
    assert not MRG.v0_check(rec)["PASS"]


def test_v0_wants_141_clips_by_default_and_the_same_order(world):
    rec = copy.deepcopy(world["rec"])
    rec["dumps"]["VGAP1_s0"]["e0"]["expected_clips"] = 141
    assert not MRG.v0_check(rec)["PASS"]
    rec = copy.deepcopy(world["rec"])
    rec["dumps"]["VGAP1_s0"]["e0"]["clips_same_order_as_join"] = False
    assert not MRG.v0_check(rec)["PASS"]


def test_main_through_files_refusals_and_arguments(world, tmp_path):
    ej = tmp_path / "join.npz"
    np.savez(str(ej), **world["J"])
    root = tmp_path / "dump"
    MRG.synthetic_meta(world["dumps"], str(root))
    args = ["--eval-join", str(ej), "--dump-root", str(root), "--out-dir", str(tmp_path / "out"), "--expect-clips", str(len(world["J"]["clip"])), "--add102-gap-dir", world["a102"]]
    assert MRG.main(args) == 0
    rec = json.load(open(str(tmp_path / "out" / "VISREP_GAPFILES_RECORD.json"), encoding="utf-8"))
    assert rec["w0"]["PASS"] and rec["eval_join_sha256"] == MRG.sha_file(str(ej))
    with pytest.raises(SystemExit):
        MRG.main(args)                                                          # the out dir exists
    assert MRG.main([]) == 2 and MRG.main(args[:-2]) == 2                       # --add102-gap-dir is required
    bad = tmp_path / "a102bad"
    shutil.copytree(world["a102"], str(bad))
    np.savez(str(bad / "VGAP_s0.npz"), gap=np.zeros(3))
    assert MRG.main(args[:2] + args[2:4] + ["--out-dir", str(tmp_path / "out2"), "--expect-clips", str(len(world["J"]["clip"])), "--add102-gap-dir", str(bad)]) == 1


# ------------------------------------------------------------------------------------------------ the queue
def fn_text(t):
    i0 = t.index("rollout() {\n")
    return t[i0:t.index("\n}\n", i0) + 3]


def dry_commands(q, seeds="ALL"):
    env = dict(os.environ, DRY="1", GR_SEED=seeds)
    out = subprocess.run([bash_exe(), str(q)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=100).stdout
    return [x for x in out.splitlines() if x.startswith("CMD: ")], out


def test_queue_rollout_function_is_the_one_of_card_add102_and_the_header_facts_hold():
    q, q0 = QUEUE.read_text(encoding="utf-8"), Q102.read_text(encoding="utf-8")
    assert hashlib.sha256(q0.encode("utf-8")).hexdigest()[:16] == "6321d4e6e12c0f47"
    assert fn_text(q) == fn_text(q0)
    assert "\r" not in q and not [w for w in BANNED if w in q]
    for bad in ("START_VISRULE10K", "VISRULE10K_STOP", "VISRULE10K_DONE", "queue_visrule", "small_vr", "rollout I10RQ50KF5P30Hb20v"):
        assert bad not in q, bad
    for need in ("GP=${GR_ROOT:-/workspace/v8proof/visrep10k}", "G2=${GR_GAP102:-/workspace/v8proof/visrule10k/gapfiles}", "CHK=${GR_CHECK:-/workspace/v8proof/visrule10k/check_v}", '"$GP/VISREP10K_STOP$PS.txt"',
                 "START_VISREP10K$PS.txt", "VISREP10K_DONE$PS.txt", "queue_visrep$PS.log", '[ "$got" = "%s" ]' % W16, 'grep -q "%s" "$CHK/CHECK_WRAPPER_V.hashes"' % W16, "ZZ CHECK-WRAPPER-V PASS",
                 '"VGAP_s0,VGAP_s1" "v0,build")', '"VGAP1_s0,VGAP1_s1" "v0,build,add102,w0")', "I10RQ50KF5P30Hb20vV0 I10RQ50KF5P30Hb20vVD5 I10RQ50KF5P30Hb20vVD5c", "I10RLcKF5P30Hb20pKD5_s$k/T1_DONE"):
        assert need in q, need


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_dry_run_has_six_commands_with_the_right_arguments_and_syntax_is_valid():
    assert subprocess.run([bash_exe(), "-n", str(QUEUE)]).returncode == 0
    cmds, out = dry_commands(QUEUE)
    assert len(cmds) == 6, out[-800:]
    seen = {}
    for c in cmds:
        assert "v8f_t1_eval_reader_rule_rzgfhqetjv.py" in c and "--form RR" in c and "--rule-join" not in c and "--vis-closing" in c
        assert "--rule-every 5" in c and "--preroll-steps 30" in c and "--kappa-max 0.2" in c and "--readout causal_g" in c and "--eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz" in c
        for arm in ("VD5cx", "VRc", "VD1c"):
            for k in (0, 1):
                if ("/I10RQ50KF5P30Hb20w%s_s%d " % (arm, k)) in c:
                    seen[(arm, k)] = 1
                    assert ("--reader /workspace/v8proof/integ10k/lr16/TK16_s%d/" % k) in c and ("QMAP_LR16_s%d_q50" % k) in c
                    want = {"VD5cx": "--vis-gap-file /workspace/v8proof/visrule10k/gapfiles/VGAP_s%d.npz" % (1 - k), "VD1c": "--vis-gap-file /workspace/v8proof/visrep10k/gapfiles/VGAP1_s%d.npz" % k}.get(arm)
                    assert (want in c) if want else ("--vis-gap-file" not in c), (arm, k)
    assert sorted(seen) == [(a, k) for a in ("VD1c", "VD5cx", "VRc") for k in (0, 1)]
    one, _ = dry_commands(QUEUE, "0")
    assert len(one) == 3 and all("_s0" in c and "TK16_s0" in c and "TK16_s1" not in c for c in one)
    assert any("VGAP_s1.npz" in c for c in one)                                 # model seed 0 reads fit seed 1 in VD5cx


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_the_arguments_of_the_rollouts_are_those_of_the_vd5c_call_of_the_add102_queue():
    q0 = Q102.read_text(encoding="utf-8")
    base = [x for x in q0.splitlines() if "rollout I10RQ50KF5P30Hb20vVD5c RR" in x][0]
    env0 = base.split(" VGAP=")[0].strip()
    for x in [x for x in QUEUE.read_text(encoding="utf-8").splitlines() if "rollout I10RQ50KF5P30Hb20w" in x]:
        assert x.strip().startswith(env0) and " VCLOSE=1 rollout " in x
    assert 'VGAP="$GF/VGAP_s$k.npz" VCLOSE=1' in base


# ------------------------------------------------------------------------------------------------ the reader
def iv(lo, hi):
    return {"interval": [lo, hi], "pooled": 0.5 * (lo + hi)}


def payload(rr=0.0954, rlc=0.0, vd5cx=(0.0, 0.004), vrc=(0.06, 0.07), vd1c=(0.0, 0.004), gap=(-0.07, -0.05), slope=(-0.07, -0.05)):
    pairs = {}
    for a, r in (("VD5cx", vd5cx), ("VRc", vrc), ("VD1c", vd1c)):
        pairs[a + "-VS-RR"] = {"C": iv(r[0] - rr, r[1] - rr), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
        pairs[a + "-VS-RLc"] = {"C": iv(*r), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
        pairs[a + "-VS-VD5c"] = {"C": iv(r[0] - 0.0169, r[1] - 0.0169), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
        pairs[a + "-VS-KD5"] = {"C": iv(-0.01, 0.01), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
    pairs["VD5c-VS-VRc"] = {"C": iv(*gap), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
    pairs["VD5c-VS-VD5"] = {"C": iv(*slope), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
    return {"levels": {"RR": {"C": rr}, "RLc": {"C": rlc}}, "pairs": pairs}


def test_name_rules_on_planted_intervals_and_their_edges():
    n = RE.names_of("VD5cx", [-0.05, -0.01], [0.0, 0.004], [0.001, 0.02], [0.001, 0.03], 0.0954, 0.0)
    assert n["VD5cx_CONTACT"] == "VISREP10K-VD5cx-CUTS-CONTACT" and n["VD5cx_HALF"] == "VISREP10K-VD5cx-WITHIN-HALF-THE-GAP" and n["VD5cx_VD5C"] == "VISREP10K-VD5cx-MORE-CONTACT-THAN-VD5c" and n["VD5cx_PRICE"] == "VISREP10K-VD5cx-FALLS-BEHIND-MORE"
    n = RE.names_of("VRc", [0.001, 0.01], [0.05, 0.06], [-0.02, -0.001], [-0.03, -0.01], 0.0954, 0.0)
    assert n["VRc_CONTACT"].endswith("-VRc-MORE-CONTACT") and n["VRc_HALF"].endswith("-VRc-HALF-THE-GAP-NOT-REACHED") and n["VRc_VD5C"].endswith("-VRc-LESS-CONTACT-THAN-VD5c") and n["VRc_PRICE"].endswith("-VRc-FALLS-BEHIND-LESS")
    n = RE.names_of("VD1c", [-0.04, 0.01], [0.01, 0.06], [-0.01, 0.01], [-0.02, 0.01], 0.0954, 0.0)
    assert n["VD1c_CONTACT"].endswith("CONTACT-NOT-SEPARATED-FROM-RR") and n["VD1c_HALF"].endswith("HALF-THE-GAP-NOT-DECIDED") and n["VD1c_VD5C"].endswith("CONTACT-NOT-SEPARATED-FROM-VD5c") and n["VD1c_PRICE"].endswith("FALLING-BEHIND-NOT-SEPARATED")
    assert RE.names_of("VRc", [0, 0], [0.0, 0.0477], [0, 0], [0, 0], 0.0954, 0.0)["VRc_HALF"].endswith("WITHIN-HALF-THE-GAP")       # an upper end exactly at H is WITHIN
    assert RE.names_of("VRc", [0, 0], [0.0477, 0.06], [0, 0], [0, 0], 0.0954, 0.0)["VRc_HALF"].endswith("NOT-DECIDED")             # a lower end exactly at H is not NOT-REACHED
    assert RE.names_of("VRc", [0, 0], [0, 0], [0.0, 0.01], [0, 0], 0.0954, 0.0)["VRc_VD5C"].endswith("CONTACT-NOT-SEPARATED-FROM-VD5c")


def test_the_replicate_name_follows_the_card_word_for_word():
    P = RE.PRE
    cases = [("CUTS-CONTACT", "WITHIN-HALF-THE-GAP", "-REPLICATED"), ("CUTS-CONTACT", "HALF-THE-GAP-NOT-REACHED", "-NOT-REPLICATED"), ("CONTACT-NOT-SEPARATED-FROM-RR", "WITHIN-HALF-THE-GAP", "-NOT-REPLICATED"),
             ("MORE-CONTACT", "HALF-THE-GAP-NOT-DECIDED", "-NOT-REPLICATED"), ("CUTS-CONTACT", "HALF-THE-GAP-NOT-DECIDED", "-REPLICATION-NOT-DECIDED")]
    for c, h, want in cases:
        assert RE.replicate_name(P + "-VD5cx-" + c, P + "-VD5cx-" + h) == P + want, (c, h)


def test_all_names_card_names_and_the_second_derivation():
    names = RE.all_names(payload())
    assert names["CARD_REPLICATE"] == "VISREP10K-REPLICATED" and names["CARD_GAP"] == "VISREP10K-CROP-GAP-CUTS-CONTACT" and names["CARD_SLOPE"] == "VISREP10K-SLOPE-CUTS-CONTACT"
    assert len(names) == 3 * 4 + 3
    n2 = RE.all_names(payload(vd5cx=(0.05, 0.06), gap=(-0.01, 0.02), slope=(0.01, 0.03)))
    assert n2["CARD_REPLICATE"] == "VISREP10K-NOT-REPLICATED" and n2["CARD_GAP"] == "VISREP10K-CROP-GAP-NOT-SEPARATED-FROM-READER-GAP" and n2["CARD_SLOPE"] == "VISREP10K-SLOPE-MORE-CONTACT"
    n3 = RE.all_names(payload(vd5cx=(0.01, 0.0475)))                       # lower end above 0 of C(X)-C(RR)? C - RR = [-0.0854, -0.0479] cuts; half: [0.01, 0.0475] has upper end at or below H
    assert n3["CARD_REPLICATE"] == "VISREP10K-REPLICATED"
    n4 = RE.all_names(payload(vd5cx=(0.01, 0.06)))
    assert n4["CARD_REPLICATE"] == "VISREP10K-REPLICATION-NOT-DECIDED"
    p = payload()
    assert RE.card_check(p, dict(n2))["PASS"] is False and RE.card_check(p, RE.all_names(p))["PASS"]
    h = RE.h_sensitivity(p)
    assert h["H_card"] == 0.0477 and abs(h["H_point_values"] - 0.0477) < 1e-12 and set(RE.ARMS) <= set(h)


def test_control_w3_w4_w5_and_w1():
    lv = {"RR": 0.0954, "RLc": 0.0, "KD5": 0.0169, "VD5c": 0.0169, "VD5": 0.0769}
    assert RE.ctl_w3(lv)["PASS"] and RE.ctl_w3(dict(lv, RR=0.09544, VD5=0.07694))["PASS"]
    for k, v in (("RR", 0.0953), ("VD5c", 0.0177), ("VD5", 0.0777), ("KD5", None), ("RLc", 0.0005)):
        assert not RE.ctl_w3(dict(lv, **{k: v}))["PASS"], k
    assert not RE.ctl_w3({k: v for k, v in lv.items() if k != "VD5"})["PASS"]
    assert RE.ctl_w4("12 passed in 3s")["PASS"] and not RE.ctl_w4("11 passed, 1 failed in 3s")["PASS"] and not RE.ctl_w4("")["PASS"] and not RE.ctl_w4("1 passed, 2 error in 1s")["PASS"]
    assert RE.ctl_w5(True, True, True)["PASS"] and not RE.ctl_w5(True, True, False)["PASS"] and not RE.ctl_w5(False, True, True)["PASS"] and not RE.ctl_w5(None, True, True)["PASS"]
    shas = {"%s_s%d" % (a, s): W16 + "0" * 48 for a in RE.ARMS for s in (0, 1)}
    ok = RE.ctl_w1(W16, "x\nZZ CHECK-WRAPPER-V PASS\n", "abc " + W16 + " v8f\n", shas)
    assert ok["PASS"], ok
    assert not RE.ctl_w1("0" * 16, "ZZ CHECK-WRAPPER-V PASS\n", W16, shas)["PASS"]
    assert not RE.ctl_w1(W16, "ZZ CHECK-WRAPPER-V FAIL\n", W16, shas)["PASS"]
    assert not RE.ctl_w1(W16, "ZZ CHECK-WRAPPER-V PASS\n", "no hash here", shas)["PASS"]
    assert not RE.ctl_w1(W16, "ZZ CHECK-WRAPPER-V PASS\n", W16, dict(shas, VRc_s0="1" * 64))["PASS"]
    assert not RE.ctl_w1(W16, "ZZ CHECK-WRAPPER-V PASS\n", W16, dict(shas, VRc_s0=None))["PASS"]


def rows(ci, j, pres, g, c=None):
    ci, j, pres, g = np.asarray(ci), np.asarray(j), np.asarray(pres, bool), np.asarray(g, float)
    return (ci.astype(np.int64), j.astype(np.int64), pres, g, np.full(len(g), -0.4) if c is None else np.asarray(c, float))


def test_control_w2_on_planted_rows():
    ref = {(0, 4): 10.0, (0, 5): 11.0, (0, 3): None, (1, 4): 20.0}
    v0 = {(0, 3): [7.0], (0, 4): [9.0], (0, 5): [9.5, 9.5004], (1, 4): [19.0]}
    K = ([0, 0, 0, 1], [3, 4, 5, 4], [1, 1, 1, 1])
    good = rows(*K, [7.0, 10.0, 11.0005, 20.0])
    r = RE.w2_one(good, ref, v0)
    assert r["PASS"] and r["n_lead_rows_with_head_output"] == 3 and r["n_lead_rows_without_head_output"] == 1
    assert not RE.w2_one(rows(*K, [7.0, 10.0, 11.002, 20.0]), ref, v0)["PASS"]                       # 2e-3 m off the dumped gap
    assert not RE.w2_one(rows(*K, [7.1, 10.0, 11.0, 20.0]), ref, v0)["PASS"]                          # not a V0 value where the head has no output
    assert not RE.w2_one(rows(*K, [7.0, 9.0, 11.0, 20.0]), ref, v0)["PASS"]                           # the reader's own gap where the head has an output
    assert not RE.w2_one(rows(*K, [7.0, 10.0, 11.0, 20.0], c=[0, 0, 0, 0]), ref, v0)["PASS"]          # the closing input is 0 on every lead row
    assert RE.w2_one(rows(*K, [7.0, 10.0, 11.0, 20.0], c=[0, 0, 0, 0.5]), ref, v0)["PASS"]            # non-zero on a positive share is enough
    assert not RE.w2_one(rows([0, 0], [3, 4], [1, 1], [7.0, 10.0]), ref, {(0, 4): [9.0]})["PASS"]      # no V0 row for a lead row without a head output
    assert not RE.w2_one(rows([0, 0], [3, 4], [0, 0], [7.0, 10.0]), ref, v0)["PASS"]                  # no lead rows at all
    # VRc: no head; every lead row against ANY V0 value at the same clip and frame within 1e-3 m
    vr = RE.w2_one(rows([0, 0, 1], [4, 5, 4], [1, 1, 1], [9.0, 9.5004, 19.0003]), None, v0)
    assert vr["PASS"] and vr["n_lead_rows_with_head_output"] == 0 and vr["n_lead_rows_without_head_output"] == 3
    assert not RE.w2_one(rows([0, 0, 1], [4, 5, 4], [1, 1, 1], [9.0, 9.52, 19.0]), None, v0)["PASS"]
    exp = {"vis_gap_file_sha256": "a" * 64, "vis_closing": 1}
    assert RE.w2_one(good, ref, v0, exp, {"vis_gap_file_sha256": "a" * 64, "vis_closing": 1})["PASS"]
    assert not RE.w2_one(good, ref, v0, exp, {"vis_gap_file_sha256": "b" * 64, "vis_closing": 1})["PASS"]
    assert not RE.w2_one(good, ref, v0, exp, {"vis_gap_file_sha256": "a" * 64, "vis_closing": 0})["PASS"]
    vrr = rows([0, 0], [4, 5], [1, 1], [9.0, 9.5])
    assert RE.w2_one(vrr, None, v0, {"vis_gap_file_sha256": "", "vis_closing": 1}, {"vis_gap_file_sha256": "", "vis_closing": 1})["PASS"]
    assert not RE.w2_one(vrr, None, v0, {"vis_gap_file_sha256": "", "vis_closing": 1}, {"vis_gap_file_sha256": "a" * 64, "vis_closing": 1})["PASS"]   # VRc with a gap file in its meta


def write_logs(pull, run, m, tick, mid, meta):
    d = Path(pull) / ("I10%s_s%d" % (run, m)) / "ticks"
    d.mkdir(parents=True, exist_ok=True)
    for nm, r in (("rule_log_I10%s_s%d.npz" % (run, m), tick), ("rule_log_mid_I10%s_s%d.npz" % (run, m), mid)):
        np.savez(str(d / nm), ci=r[0], j=r[1], present=r[2], g_src=r[3], c_src=r[4], **{k: np.asarray(v) for k, v in meta.items()})
    (d / ("run_I10%s_s%d.txt" % (run, m))).write_text("arm x wrapper sha256 %s started t\n" % (W16 + "0" * 48), encoding="utf-8")


def reader_gap(ci, j):
    return np.float32(20.0 + 0.1 * j + ci)


@pytest.fixture(scope="module")
def pulled(world, tmp_path_factory):
    tp = tmp_path_factory.mktemp("pull")
    ej = tp / "join.npz"
    np.savez(str(ej), **world["J"])
    root = tp / "dump"
    MRG.synthetic_meta(world["dumps"], str(root))                                      # D_H1 (the dumps of the fixture)
    d5 = {s: MRG.synthetic_dump(world["J"], s, rng_seed=99) for s in MRG.SEEDS}
    os.makedirs(str(root / "D_H5"))
    for s in MRG.SEEDS:
        np.savez(str(root / "D_H5" / ("reader_s%d_score.npz" % s)), **d5[s])
    rec102 = json.load(open(str(Path(world["a102"]) / "VISRULE_GAPFILES_RECORD.json"), encoding="utf-8"))
    return {"tp": tp, "ej": str(ej), "root": str(root), "d5": d5, "rec102": rec102}


def synth_pull(tmp_path, world, pulled, spoil=None, closing_zero=None, v0_noise=True, meta_bad=None):
    pull = tmp_path / "pull"
    J, n = world["J"], np.asarray(world["J"]["n"])
    refs5 = {m: MRG.second_routine(J, pulled["d5"][m]) for m in (0, 1)}
    refs1 = {m: MRG.second_routine(J, world["dumps"][m]) for m in (0, 1)}
    for m in (0, 1):
        for arm in ("V0",) + RE.ARMS:
            run = RE.V0_RUN if arm == "V0" else RE.RUN[arm]
            cols = {0: [[], [], [], []], 1: [[], [], [], []]}
            for c in range(len(n)):
                for j in range(int(n[c])):
                    own = reader_gap(c, j)
                    h = {"VD5cx": refs5[1 - m].get((c, j)), "VD1c": refs1[m].get((c, j))}.get(arm)
                    g = np.float32(h) if h is not None else own
                    t = cols[j % 2]
                    t[0].append(c), t[1].append(j), t[2].append(g)
                    t[3].append(0.0 if arm == "V0" else -0.5)
                    if arm == "V0" and v0_noise and j % 2 == 0:                      # a frame logged twice with the float noise of the reader
                        t[0].append(c), t[1].append(j), t[2].append(np.float32(own + 0.0004)), t[3].append(0.0)
            if spoil and spoil[0] == (arm, m):
                cols[0][2][spoil[1]] = cols[0][2][spoil[1]] + 0.01
            if closing_zero == (arm, m):
                cols[0][3] = [0.0] * len(cols[0][3]); cols[1][3] = [0.0] * len(cols[1][3])
            exp = {"V0": "", "VD5cx": pulled["rec102"]["files"]["VGAP_s%d" % (1 - m)]["file_sha256"], "VRc": "", "VD1c": world["rec"]["files"]["VGAP1_s%d" % m]["file_sha256"]}[arm]
            if meta_bad == (arm, m):
                exp = "f" * 64
            meta = {"vis_gap_file_sha256": exp, "vis_closing": 0 if arm == "V0" else 1}
            tick, mid = [rows(cols[k][0], cols[k][1], [True] * len(cols[k][0]), cols[k][2], cols[k][3]) for k in (0, 1)]
            write_logs(pull, run, m, tick, mid, meta)
    return pull


class Ns:
    pass


def test_control_w2_end_to_end_with_the_three_arms_and_planted_faults(world, pulled, tmp_path):
    a = Ns()
    a.eval_join, a.dump_root = pulled["ej"], pulled["root"]
    a.pull = str(synth_pull(tmp_path / "a", world, pulled))
    c2, J = RE.w2_from_files(a, world["rec"], pulled["rec102"])
    assert c2["PASS"], c2
    assert sorted(c2["per_run"]) == sorted("%s_s%d" % (x, s) for x in RE.ARMS for s in (0, 1))
    assert c2["per_run"]["VRc_s0"]["n_lead_rows_with_head_output"] == 0 and c2["per_run"]["VD5cx_s1"]["n_lead_rows_with_head_output"] > 0 and c2["per_run"]["VD1c_s0"]["n_lead_rows_closing_nonzero"] > 0
    assert all(v["share_of_lead_steps_with_nonzero_closing"] > 0 for v in c2["per_run"].values())
    for label, kw in (("spoil_vd5cx", dict(spoil=(("VD5cx", 1), 5))), ("spoil_vrc", dict(spoil=(("VRc", 0), 5))), ("spoil_vd1c", dict(spoil=(("VD1c", 1), 9))), ("closing_zero", dict(closing_zero=("VRc", 1))),
                      ("meta", dict(meta_bad=("VD1c", 0)))):
        a.pull = str(synth_pull(tmp_path / label, world, pulled, **kw))
        assert not RE.w2_from_files(a, world["rec"], pulled["rec102"])[0]["PASS"], label
    a.pull = str(synth_pull(tmp_path / "wrongfit", world, pulled))
    wrong = copy.deepcopy(pulled["rec102"])
    wrong["files"]["VGAP_s0"]["file_sha256"] = "e" * 64                                # the VD5cx hash of model seed 1 is the one of fit seed 0
    assert not RE.w2_from_files(a, world["rec"], wrong)[0]["PASS"]


def test_vd5cx_reads_the_other_fit_seed_not_its_own(world, pulled, tmp_path):
    """model seed 0 fed fit seed 0's dump (the own seed) must fail: the dumps of the two seeds differ"""
    J = world["J"]
    pull = synth_pull(tmp_path / "p", world, pulled)
    own = MRG.second_routine(J, pulled["d5"][0])
    rows_ = RE.rows_of(RE.load_logs(str(pull), RE.RUN["VD5cx"], 0))
    ci, j, g = rows_[0], rows_[1], rows_[3]
    diff = sum(1 for i in range(len(ci)) if own.get((int(ci[i]), int(j[i]))) is not None and abs(g[i] - own[(int(ci[i]), int(j[i]))]) > 1e-3)
    assert diff > 0


def test_control_w0_on_the_record_and_the_files(world):
    gdir = str(world["base"] / "clean")
    ejs = "x" * 64
    assert not RE.ctl_w0(world["rec"], gdir, world["a102"], ejs)["PASS"]                  # the record is for another evaluation join
    rec = copy.deepcopy(world["rec"])
    rec["eval_join_sha256"] = ejs
    assert RE.ctl_w0(rec, gdir, world["a102"], ejs)["PASS"]
    assert not RE.ctl_w0(rec, str(world["base"]), world["a102"], ejs)["PASS"]             # the gap files are not where the record says
    assert not RE.ctl_w0(None, gdir, world["a102"], ejs)["PASS"]
    assert not RE.ctl_w0(rec, gdir, str(world["base"]), ejs)["PASS"]                      # no record of card ADD102 there
    rec2 = copy.deepcopy(rec)
    rec2["w0"]["PASS"] = False
    assert not RE.ctl_w0(rec2, gdir, world["a102"], ejs)["PASS"]


def test_a_withheld_payload_and_the_names_stage_refuses_with_a_control_failed(tmp_path):
    ok = {k: {"PASS": True} for k in RE.CONTROLS}
    assert RE.CONTROLS == ("w0", "w1", "w2", "w3", "w4", "w5")
    bad = dict(ok, w2={"PASS": False})
    p = payload()
    r = RE.assemble_controls(bad, p)
    assert r["controls_all_pass"] is False and "payload" not in r and "withheld" in r
    cf = tmp_path / "c.json"
    cf.write_text(json.dumps(r), encoding="utf-8")
    out = tmp_path / "res.json"
    assert RE.main(["--stage", "names", "--controls", str(cf), "--out", str(out)]) == 3 and not out.exists()
    cf2 = tmp_path / "c2.json"
    cf2.write_text(json.dumps(RE.assemble_controls(ok, p)), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(cf2), "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["controls_sha256"] == RE.sha_file(str(cf2)) and res["names"]["CARD_REPLICATE"].startswith("VISREP10K-") and "VD1c_VD5C" in res["names"] and "CARD_SLOPE" in res["names"]
    cf3 = tmp_path / "c3.json"
    cf3.write_text(json.dumps(RE.assemble_controls({k: v for k, v in ok.items() if k != "w5"}, p)), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(cf3), "--out", str(tmp_path / "res3.json")]) == 3


def test_reader_source_carries_the_strategic_family_the_references_and_the_engine_bootstrap_setting():
    t = (HERE / "read_visrep10k.py").read_text(encoding="utf-8")
    assert "a.n_boot_control = 300" in t and t.index("a = ap.parse_args(argv)") < t.index("a.n_boot_control = 300")
    assert "R.strategic_grid(" in t and "S.pair_read(" in t and 'pairs_def["VD5c-VS-VRc"]' in t and 'pairs_def["VD5c-VS-VD5"]' in t and '"-VS-VD5c"' in t and '"-VS-KD5"' in t
    assert 'REF_RUNS = {"RR": BANKED["RR"], "RLc": BANKED["RLc"], "VD5c": B102["VD5c"]}' in t and "for ref, rn in REF_RUNS.items()" in t
    assert 'RUN = {"VD5cx": "RQ50KF5P30Hb20wVD5cx", "VRc": "RQ50KF5P30Hb20wVRc", "VD1c": "RQ50KF5P30Hb20wVD1c"}' in t


def test_podread_text():
    t = (HERE / "podread_visrep10k.sh").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=-1" in t and "VISREP10K_CONTROLS.json" in t and "VISREP10K_RESULT.json" in t and "ALREADY-READ" in t and "QUEUE-STOPPED" in t
    for x in ("wVD5cx", "wVRc", "wVD1c", "vV0", "vVD5", "vVD5c", "pKD5", "D_H1", "D_H5", "--add102-gap-dir", "--wrapper-hashes", "--wrapper-file", "VISRULE_GAPFILES_RECORD.json", "VISREP_GAPFILES_RECORD.json"):
        assert x in t, x
    for bad in (" rm ", " mv ", "kill", "pkill", "visrule10k/read"):
        assert bad not in t, bad
    if bash_exe():
        assert subprocess.run([bash_exe(), "-n", str(HERE / "podread_visrep10k.sh")]).returncode == 0


def test_push_and_pull_scripts():
    pu, pl = (HERE / "push_visrep.sh").read_text(encoding="utf-8"), (HERE / "pull_visrep.sh").read_text(encoding="utf-8")
    assert "set -C" in pu and "lead_pytest.out" in pu and "/workspace/v8proof/visrep10k" in pu and "cat $P" not in pu and "ZZ PUSH-VISREP" in pu
    assert "pull_add103.tar" in pl and "dry_all.out" in pl and "pod_add103" in pl and "VISREP_GAPFILES_RECORD.json" in pl
    if bash_exe():
        for f in ("push_visrep.sh", "pull_visrep.sh"):
            assert subprocess.run([bash_exe(), "-n", str(HERE / f)]).returncode == 0


def test_texts_hold_no_banned_words():
    for f in ("gen_rep_gapfile.py", "make_rep_gapfile.py", "make_visrep_queue.py", "pod_visrep10k_queue.sh", "read_visrep10k.py", "podread_visrep10k.sh", "push_visrep.sh", "pull_visrep.sh", "VISREP_DEFS.md"):
        p = HERE / f
        if p.is_file():
            t = p.read_text(encoding="utf-8")
            assert not [w for w in BANNED if w in t], f
            assert len(str(p)) < 200


def test_selftests_run():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for args, last in (([str(HERE / "read_visrep10k.py"), "--selftest"], "ZZ SELFTEST-READ-VISREP10K ok: True"), ([str(HERE / "make_rep_gapfile.py"), "--selftest"], "ZZ MAKE-REP-GAPFILE-SELFTEST PASS")):
        r = subprocess.run([sys.executable] + args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=300)
        assert r.stdout.strip().splitlines()[-1] == last, (args, r.stdout[-400:])
