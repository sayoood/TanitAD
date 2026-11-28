"""test_visrule.py -- tests of the kit of card ADD102 (VISRULE10K): the wrapper variant, the gap files, the queue, the reader controls and names, with planted faults. Synthetic data only; nothing runs on the pod.
Run (whole file): cd /f/TanitAD-scratch/v8proof && PYCAP_MB=3000 bash ops/pycap.sh -m pytest visrule/test_visrule.py -q -p no:cacheprovider --basetemp /f/TanitAD-scratch/v8proof/visrule/tmp_pt/<new name>"""
import copy
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
os.environ.setdefault("DISTRULE_DIR", str(HERE.parent / "distrule"))
import check_wrapper_v as CW                  # noqa: E402
import make_vis_gapfile as MVG                # noqa: E402
import make_wrapper_v as MWV                  # noqa: E402
import read_visrule10k as RE                  # noqa: E402

WRAPPER = HERE / "v8f_t1_eval_reader_rule_rzgfhqetjv.py"
PARENT = HERE.parent / "gaprule" / "v8f_t1_eval_reader_rule_rzgfhqetj.py"
QUEUE = HERE / "pod_visrule10k_queue.sh"
CR_QUEUE = HERE.parent / "croprule" / "pod_croprule10k_queue.sh"
BANNED = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
needs_dist = pytest.mark.skipif(not (Path(os.environ["DISTRULE_DIR"]) / "v8f_t1_eval_rule_mix.py").is_file(), reason="the distrule folder is not there")


# ------------------------------------------------------------------------------------------------ the wrapper variant
def test_generated_text_is_what_the_maker_builds_and_the_parent_hash_is_asserted():
    ptext = PARENT.read_bytes().decode("utf-8")
    import hashlib
    assert hashlib.sha256(ptext.encode("utf-8")).hexdigest()[:16] == "cbf7063d867c9f4e"
    t, anchors, plines = MWV.build(ptext)
    assert t == WRAPPER.read_bytes().decode("utf-8")
    assert len(anchors) == 10 and all(n.startswith("V") for n, _ in anchors)
    assert t.count("\n") > ptext.count("\n")


def test_each_edit_text_must_occur_exactly_once_in_the_parent():
    ptext = PARENT.read_bytes().decode("utf-8")
    with pytest.raises(AssertionError):
        MWV.build(ptext.replace('ap.add_argument("--rule-join", default="")', 'ap.add_argument("--rule-join", default="")\n    ap.add_argument("--rule-join", default="")'))
    with pytest.raises(AssertionError):
        MWV.build(ptext.replace('present, used = feed(STATE["form"], p, g, ok, STATE["xs"], STATE["ys"])', "pass"))


def test_static_facts_hold_and_each_planted_fault_turns_one_red():
    src = WRAPPER.read_text(encoding="utf-8")
    assert all(c for _, c in CW.static_facts(src))
    faults = {"A1": ('ap.add_argument("--vis-gap-file", default="")', 'ap.add_argument("--vis-gap-fileX", default="")'),
              "A2": ('ap.add_argument("--vis-closing", action="store_true")', 'ap.add_argument("--vis-closing", action="store_false")'),
              "A3": ('(mine.vis_gap_file or mine.vis_closing) and mine.form != "RR"', '(mine.vis_gap_file or mine.vis_closing) and mine.form != "RRc"'),
              "A6": ("used, vis_hc = vis_look(st, rd, q, p, g, used, okr)", "pass"),
              "A8": ("def build_rule_label(", "a.eval_join = ''\ndef build_rule_label(")}
    for k, (old, new) in faults.items():
        assert src.count(old) == 1, k
        res = dict((n.split(" ")[0], c) for n, c in CW.static_facts(src.replace(old, new)))
        assert res[k] is False, k
    # A5: the replacement after the log columns
    a = "        used, vis_c = vis_tick(p, g, used, ok)\n"
    b5 = src.replace(a, "")
    b5 = b5.replace('    return p, used.astype(np.float32)', "    used, vis_c = vis_tick(p, g, used, ok)\n    return p, used.astype(np.float32)")
    assert dict((n.split(" ")[0], c) for n, c in CW.static_facts(b5))["A5"] is False


@needs_dist
def test_identity_checks_pass_on_the_generated_wrapper():
    res = CW.identity_checks(str(PARENT), str(WRAPPER))
    assert all(ok for _, ok, _ in res), [(n, d) for n, ok, d in res if not ok]
    assert len(res) == 7


def mutant(tmp_path, name, old, new):
    t = WRAPPER.read_text(encoding="utf-8")
    assert t.count(old) == 1, old
    p = tmp_path / ("mut_%s.py" % name)
    p.write_text(t.replace(old, new), encoding="utf-8", newline="\n")
    return p


@needs_dist
@pytest.mark.parametrize("name,old,new,red", [
    ("step", "VIS_GAP_MAX_M, VIS_STEP_MAX_M, VIS_RUN_MAX, VIS_RUN_MIN = 60.0, 3.0, 5, 3", "VIS_GAP_MAX_M, VIS_STEP_MAX_M, VIS_RUN_MAX, VIS_RUN_MIN = 60.0, 30.0, 5, 3", "I5"),
    ("runmin", "VIS_GAP_MAX_M, VIS_STEP_MAX_M, VIS_RUN_MAX, VIS_RUN_MIN = 60.0, 3.0, 5, 3", "VIS_GAP_MAX_M, VIS_STEP_MAX_M, VIS_RUN_MAX, VIS_RUN_MIN = 60.0, 3.0, 5, 2", "I5"),
    ("runmax", "VIS_GAP_MAX_M, VIS_STEP_MAX_M, VIS_RUN_MAX, VIS_RUN_MIN = 60.0, 3.0, 5, 3", "VIS_GAP_MAX_M, VIS_STEP_MAX_M, VIS_RUN_MAX, VIS_RUN_MIN = 60.0, 3.0, 6, 3", "I5"),
    ("range", "VIS_GAP_MAX_M, VIS_STEP_MAX_M, VIS_RUN_MAX, VIS_RUN_MIN = 60.0, 3.0, 5, 3", "VIS_GAP_MAX_M, VIS_STEP_MAX_M, VIS_RUN_MAX, VIS_RUN_MIN = 90.0, 3.0, 5, 3", "I5"),
    ("presence", "    sel = present & has\n", "    sel = has\n", "I3"),
    ("nofile", "    used[sel] = g[sel]\n", "    used[sel] = used[sel]\n", "I3"),
])
def test_planted_faults_in_the_variant_turn_the_identity_checks_red(tmp_path, name, old, new, red):
    p = mutant(tmp_path, name, old, new)
    res = CW.identity_checks(str(PARENT), str(p))
    failed = [n.split(" ")[0] for n, ok, _ in res if not ok]
    assert red in failed, (name, failed)


@needs_dist
def test_closing_run_rule_on_hand_made_series():
    m = CW.load_module(str(WRAPPER), "wv_hand")
    g = np.array([[10, 9.8, 9.6, 9.4, 9.2],           # a clean run of 5: slope -1.0 m/s
                  [10, 9.8, 50, 9.4, 9.2],            # a step above 3 m ends the run after 2 frames: 0
                  [0, 0, 0, 5, 4.8],                  # the invalid frames end the run after 2 frames: 0
                  [10, 9.8, 9.6, 9.4, 9.2],           # the frame of the look is not in the run: 0
                  [20, 19.0, 18.0, 17.0, 16.0]], float)
    ok = np.ones((5, 5), bool)
    ok[2, :3] = False
    ok[3, 4] = False
    out = m.vis_run_closing(g, ok)
    assert np.allclose(out, [-1.0, 0.0, 0.0, 0.0, -5.0], atol=1e-5)
    g3 = np.array([[40.0, 41.0, 9.6, 9.4, 9.2]])      # run of 3 frames after a step of more than 3: slope -1.0
    assert abs(m.vis_run_closing(g3, np.ones((1, 5), bool))[0] + 1.0) < 1e-5


def test_the_wrapper_text_holds_no_banned_words_and_no_carriage_return():
    t = WRAPPER.read_text(encoding="utf-8")
    assert not [w for w in BANNED if w in t] and "\r" not in t


# ------------------------------------------------------------------------------------------------ the gap files
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("gf")
    J, dumps = MVG.synthetic_all()
    meta = MVG.synthetic_meta(dumps, str(base / "dumps"))
    rec = MVG.build(J, dumps, str(base / "clean"), meta=meta)
    return {"J": J, "dumps": dumps, "meta": meta, "rec": rec, "base": base}


def test_clean_build_passes_v0_and_the_build_check(world):
    r = world["rec"]
    assert r["v0"]["PASS"], r["v0"]
    assert r["build"]["PASS"], r["build"]
    assert sorted(r["files"]) == ["VGAP_s0", "VGAP_s1"] and r["card"] == "PREREG_V8F_PROOF_ADD102"
    for nm, a in r["files"].items():
        assert a["max_abs_dev_saved_minus_second_routine"] <= 1e-9 and a["n_rows"] == a["n_join_rows"] and a["n_without_output"] >= 4 * len(world["J"]["clip"])
        assert a["n_out_above_60"] > 0 and a["n_output_flag_mismatch"] == 0


def test_the_file_carries_the_head_gap_as_dumped_by_hand(world):
    J, D = world["J"], world["dumps"][0]
    z = MVG.load_npz(str(world["base"] / "clean" / "VGAP_s0.npz"))
    names = [str(c) for c in J["clip"]]
    for c, f in ((0, 5), (2, 7), (3, 0), (1, 3)):
        r = int(J["off"][c]) + f
        dr = [i for i in range(len(D["cid"])) if int(D["cid"][i]) == c and int(D["frame"][i]) == f][0]
        want = float(D["gap"][dr])
        if np.isfinite(want):
            assert z["has_out"][r] and z["gap"][r] == want and z["clip_index"][r] == c and z["frame"][r] == f
        else:
            assert not z["has_out"][r] and np.isnan(z["gap"][r])
    assert z["gap"][z["has_out"]].max() > 60.0            # not clipped to 0 to 60 m
    assert [str(c) for c in z["clips"]] == names


def test_each_planted_fault_of_the_dumps_turns_v0_red_and_of_the_build_turns_the_build_check_red(world, tmp_path):
    vf = MVG.v0_faults(world["J"], world["dumps"])
    assert vf and all(vf.values()), vf
    bf = MVG.build_faults(world["J"], world["dumps"], str(tmp_path / "bf"), world["meta"])
    assert sorted(bf) == ["clip60", "min_label", "scale", "shift", "zero_nooutput"] and all(bf.values()), bf


def test_the_model_seed_reads_its_own_fit_seed_and_the_folder_d_h5(world):
    rec = copy.deepcopy(world["rec"])
    rec["files"]["VGAP_s0"]["dump_file"] = rec["files"]["VGAP_s0"]["dump_file"].replace("reader_s0", "reader_s1")
    assert not MVG.build_check(rec)["PASS"]
    rec = copy.deepcopy(world["rec"])
    rec["files"]["VGAP_s1"]["dump_dir_name"] = "B_H5"
    assert not MVG.build_check(rec)["PASS"]
    rec = copy.deepcopy(world["rec"])
    rec["dumps"]["VGAP_s1"]["e0"]["dump_seed"] = 0
    assert not MVG.v0_check(rec)["PASS"]


def test_v0_wants_141_clips_by_default_and_the_same_order(world):
    rec = copy.deepcopy(world["rec"])
    rec["dumps"]["VGAP_s0"]["e0"]["expected_clips"] = 141
    assert not MVG.v0_check(rec)["PASS"]
    rec = copy.deepcopy(world["rec"])
    rec["dumps"]["VGAP_s0"]["e0"]["clips_same_order_as_join"] = False
    assert not MVG.v0_check(rec)["PASS"]


def test_out_dir_refused_and_arguments_required(world, tmp_path):
    with pytest.raises(SystemExit):
        MVG.build(world["J"], world["dumps"], str(world["base"] / "clean"))
    assert MVG.main([]) == 2


def test_main_through_files(world, tmp_path):
    ej = tmp_path / "join.npz"
    np.savez(str(ej), **world["J"])
    root = tmp_path / "dump"
    MVG.synthetic_meta(world["dumps"], str(root))
    rc = MVG.main(["--eval-join", str(ej), "--dump-root", str(root), "--out-dir", str(tmp_path / "out"), "--expect-clips", str(len(world["J"]["clip"]))])
    assert rc == 0
    rec = json.load(open(str(tmp_path / "out" / "VISRULE_GAPFILES_RECORD.json"), encoding="utf-8"))
    assert rec["v0"]["PASS"] and rec["build"]["PASS"] and rec["eval_join_sha256"] == MVG.sha_file(str(ej))
    with pytest.raises(SystemExit):
        MVG.main(["--eval-join", str(ej), "--dump-root", str(root), "--out-dir", str(tmp_path / "out")])


# ------------------------------------------------------------------------------------------------ the queue
def fn_text(t):
    i0 = t.index("rollout() {\n")
    return t[i0:t.index("\n}\n", i0) + 3]


def bash_exe():
    return shutil.which("bash")


def dry_commands(q, seeds="ALL"):
    env = dict(os.environ, DRY="1", GR_SEED=seeds)
    out = subprocess.run([bash_exe(), str(q)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=100).stdout
    return [x for x in out.splitlines() if x.startswith("CMD: ")], out


def test_queue_is_what_the_maker_writes_and_the_rollout_function_differs_only_by_the_new_arguments():
    q = QUEUE.read_text(encoding="utf-8")
    c = CR_QUEUE.read_text(encoding="utf-8")
    ins = '  if [ -n "${VGAP:-}" ]; then\n    [ "$DRY" = 1 ] || [ -f "$VGAP" ] || stop "NO-GAP-FILE $VGAP $arm"\n    extra+=(--vis-gap-file "$VGAP")\n  fi\n  [ "${VCLOSE:-0}" = 1 ] && extra+=(--vis-closing)\n'
    run_txt = '${VGAP:+ vis-gap-file $VGAP gap file sha256 $(sha256sum "$VGAP" | cut -c1-64) vis-closing ${VCLOSE:-0}}'
    f = fn_text(q)
    assert f.count(ins) == 1 and f.count(run_txt) == 1
    assert f.replace(ins, "").replace(run_txt, "") == fn_text(c)
    assert "\r" not in q and not [w for w in BANNED if w in q]
    for bad in ("CROPRULE", "croprule", "KB5", "KB1", "KD1", "K100", "RJOIN=", "$PJ"):
        assert bad not in q, bad
    import hashlib
    assert hashlib.sha256(WRAPPER.read_bytes()).hexdigest()[:16] in q


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_queue_dry_run_has_six_commands_with_the_right_arguments():
    cmds, out = dry_commands(QUEUE)
    assert len(cmds) == 6, out[-800:]
    seen = {}
    for c in cmds:
        assert "v8f_t1_eval_reader_rule_rzgfhqetjv.py" in c and "--form RR" in c and "--rule-join" not in c
        assert "--reader /workspace/v8proof/integ10k/lr16/TK16_s" in c and "--qmap /workspace/v8proof/integ10k/qmap_sweep/QMAP_LR16_s" in c
        assert "--rule-every 5" in c and "--preroll-steps 30" in c and "--kappa-max 0.2" in c and "--readout causal_g" in c and "--eval-join /workspace/v8proof/hierrep/pins/join_eval141.npz" in c
        for arm in ("V0", "VD5c", "VD5"):
            if ("/I10RQ50KF5P30Hb20v%s_s" % arm) in c:
                seen[arm] = seen.get(arm, 0) + 1
                has_g, has_c = "--vis-gap-file /workspace/v8proof/visrule10k/gapfiles/VGAP_s" in c, "--vis-closing" in c
                assert (has_g, has_c) == {"V0": (False, False), "VD5": (True, False), "VD5c": (True, True)}[arm], (arm, c[:300])
    assert seen == {"V0": 2, "VD5c": 2, "VD5": 2}
    one, _ = dry_commands(QUEUE, "1")
    assert len(one) == 3 and all("_s1" in c and "TK16_s1" in c and "QMAP_LR16_s1_q50" in c and "VGAP_s0" not in c for c in one)


@pytest.mark.skipif(not bash_exe(), reason="no bash")
def test_the_arguments_of_v0_are_those_of_the_banked_rr_call_in_the_longroll_queue():
    """the banked RR runs: RGRID=$G30 PREROLL=30 READOUT=causal_g ACCGAIN=0 PATHEXEC=1 KAPPA=0.2 EVERY=5 QMAP_FILE=<q50 map of the seed> rollout I10RQ50KF5P30Hb20 RR k k (integ10k/pod_longroll10k_queue.sh:254)"""
    lr = (HERE.parent / "integ10k" / "pod_longroll10k_queue.sh")
    if not lr.is_file():
        pytest.skip("no longroll queue")
    t = lr.read_text(encoding="utf-8")
    line = [x for x in t.splitlines() if "rollout I10RQ50KF5P30Hb20 RR" in x][0]
    for tok in ("RGRID=$G30", "PREROLL=30", "READOUT=causal_g", "ACCGAIN=0", "PATHEXEC=1", "KAPPA=0.2", "EVERY=5"):
        assert tok in line, tok
    q = QUEUE.read_text(encoding="utf-8")
    v0 = [x for x in q.splitlines() if "rollout I10RQ50KF5P30Hb20vV0 RR" in x][0]
    for tok in ("RGRID=$G30", "PREROLL=30", "READOUT=causal_g", "ACCGAIN=0", "PATHEXEC=1", "KAPPA=0.2", "EVERY=5", "QMAP_FILE="):
        assert tok in v0, tok
    assert "VGAP" not in v0 and "VCLOSE" not in v0


# ------------------------------------------------------------------------------------------------ the reader
def iv(lo, hi):
    return {"interval": [lo, hi], "pooled": 0.5 * (lo + hi)}


def payload(rr=0.0954, rlc=0.0, vd5c=(0.0, 0.004), vd5=(0.0, 0.004), kd5=(-0.01, 0.01)):
    pairs = {}
    for a, r in (("VD5c", vd5c), ("VD5", vd5)):
        pairs[a + "-VS-RR"] = {"C": iv(r[0] - rr, r[1] - rr), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
        pairs[a + "-VS-RLc"] = {"C": iv(*r), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
        pairs[a + "-VS-KD5"] = {"C": iv(*kd5), "F": iv(-0.02, 0.03), "U": iv(-0.01, 0.01)}
    return {"levels": {"RR": {"C": rr}, "RLc": {"C": rlc}}, "pairs": pairs}


def test_name_rules_on_planted_intervals():
    n = RE.names_of("VD5c", [-0.05, -0.01], [0.0, 0.004], [0.001, 0.02], [0.001, 0.03], 0.0954, 0.0)
    assert n["VD5c_CONTACT"].endswith("-VD5c-CUTS-CONTACT") and n["VD5c_HALF"].endswith("-VD5c-WITHIN-HALF-THE-GAP") and n["VD5c_KD5"].endswith("-VD5c-PRESENCE-COSTS-CONTACT") and n["VD5c_PRICE"].endswith("-VD5c-FALLS-BEHIND-MORE")
    n = RE.names_of("VD5", [0.001, 0.01], [0.05, 0.06], [-0.02, -0.001], [-0.03, -0.01], 0.0954, 0.0)
    assert n["VD5_CONTACT"].endswith("-VD5-MORE-CONTACT") and n["VD5_HALF"].endswith("-VD5-HALF-THE-GAP-NOT-REACHED") and n["VD5_KD5"].endswith("-VD5-LESS-CONTACT-THAN-KD5") and n["VD5_PRICE"].endswith("-VD5-FALLS-BEHIND-LESS")
    n = RE.names_of("VD5", [-0.04, 0.01], [0.01, 0.06], [-0.01, 0.01], [-0.02, 0.01], 0.0954, 0.0)
    assert n["VD5_CONTACT"].endswith("CONTACT-NOT-SEPARATED-FROM-RR") and n["VD5_HALF"].endswith("HALF-THE-GAP-NOT-DECIDED") and n["VD5_KD5"].endswith("CONTACT-NOT-SEPARATED-FROM-KD5") and n["VD5_PRICE"].endswith("FALLING-BEHIND-NOT-SEPARATED")
    # the edges: an upper end exactly at H is WITHIN; a lower end exactly at H is NOT-DECIDED
    assert RE.names_of("VD5", [0, 0], [0.0, 0.0477], [0, 0], [0, 0], 0.0954, 0.0)["VD5_HALF"].endswith("WITHIN-HALF-THE-GAP")
    assert RE.names_of("VD5", [0, 0], [0.0477, 0.06], [0, 0], [0, 0], 0.0954, 0.0)["VD5_HALF"].endswith("NOT-DECIDED")


def test_the_card_name_comes_from_vd5c_only_and_is_checked_a_second_time():
    for vd5c, vd5, want in (((0.0, 0.004), (0.06, 0.07), "WITHIN-HALF-THE-GAP"), ((0.05, 0.06), (0.0, 0.001), "HALF-THE-GAP-NOT-REACHED"), ((0.01, 0.06), (0.0, 0.001), "HALF-THE-GAP-NOT-DECIDED")):
        names = RE.all_names(payload(vd5c=vd5c, vd5=vd5))
        assert names["CARD"] == "VISRULE10K-" + want
    p = payload()
    assert RE.card_check(p, {"CARD": "VISRULE10K-HALF-THE-GAP-NOT-REACHED"})["PASS"] is False


def test_h_note_is_reported_not_used():
    p = payload(vd5c=(0.0, 0.0478))                        # between the point-value H (0.0477) and a card H of 0.0477: the same here
    h = RE.h_sensitivity(p)
    assert h["H_card"] == 0.0477 and abs(h["H_point_values"] - 0.0477) < 1e-12 and "VD5c" in h and "VD5" in h


def lv(c, u=0.05, f=0.1):
    return {"C": list(c), "U": [u, u], "F": [f, f]}


def test_control_v1_two_windows_of_650_per_pooled_rate():
    # note N1: the 650 windows are the two model seeds together; a per-seed rate is over 325 windows, so one window in one seed moves the pooled rate by 1/650
    base = {"V0": lv([0.0092, 0.0123]), "RR": lv([0.0092, 0.0123])}
    assert RE.ctl_v1(base)["PASS"]
    one_each = {"V0": lv([0.0092 + 1.0 / 325, 0.0123 + 1.0 / 325]), "RR": lv([0.0092, 0.0123])}       # two windows of the 650
    r = RE.ctl_v1(one_each)
    assert r["PASS"] and abs(r["rows"]["C"]["windows_of_650_pooled"] - 2.0) < 1e-6
    two_in_one = {"V0": lv([0.0092 + 2.0 / 325, 0.0123]), "RR": lv([0.0092, 0.0123])}                 # two windows of the 650, both in one seed
    assert RE.ctl_v1(two_in_one)["PASS"]
    three = {"V0": lv([0.0092 + 2.0 / 325, 0.0123 + 1.0 / 325]), "RR": lv([0.0092, 0.0123])}          # three windows of the 650
    assert not RE.ctl_v1(three)["PASS"]
    cancel = {"V0": lv([0.0092 + 3.0 / 325, 0.0123 - 3.0 / 325]), "RR": lv([0.0092, 0.0123])}         # the pooled rate is the control; the per-seed differences are printed
    rc = RE.ctl_v1(cancel)
    assert rc["PASS"] and max(rc["rows"]["C"]["windows_of_650_per_seed"]) > 5.9
    for c in ("U", "F"):
        b = {"V0": lv([0.0092, 0.0123]), "RR": lv([0.0092, 0.0123])}
        b["V0"][c][1] += 3.0 / 325
        assert not RE.ctl_v1(b)["PASS"], c
    short = {"V0": {c: [0.01] for c in ("C", "U", "F")}, "RR": lv([0.0092, 0.0123])}                  # one seed missing
    assert not RE.ctl_v1(short)["PASS"]


def rows(ci, j, pres, g, c=None):
    ci, j, pres, g = np.asarray(ci), np.asarray(j), np.asarray(pres, bool), np.asarray(g, float)
    return (ci.astype(np.int64), j.astype(np.int64), pres, g, np.zeros(len(g)) if c is None else np.asarray(c, float))


def test_control_v2_on_planted_rows():
    ref = {(0, 4): 10.0, (0, 5): 11.0, (0, 3): None, (1, 4): 20.0}
    v0 = {(0, 3): 7.0, (0, 4): 9.0, (0, 5): 9.5, (1, 4): 19.0}
    good = rows([0, 0, 0, 1], [3, 4, 5, 4], [1, 1, 1, 1], [7.0, 10.0, 11.0005, 20.0])
    r = RE.v2_one(good, ref, v0, zero_closing=True)
    assert r["PASS"] and r["n_lead_rows_with_head_output"] == 3 and r["n_lead_rows_without_head_output"] == 1
    assert not RE.v2_one(rows([0, 0, 0, 1], [3, 4, 5, 4], [1, 1, 1, 1], [7.0, 10.0, 11.002, 20.0]), ref, v0, True)["PASS"]            # 2e-3 m off the dumped gap
    assert not RE.v2_one(rows([0, 0, 0, 1], [3, 4, 5, 4], [1, 1, 1, 1], [7.1, 10.0, 11.0, 20.0]), ref, v0, True)["PASS"]               # a gap that is not the reader's own where the head has no output
    assert not RE.v2_one(rows([0, 0, 0, 1], [3, 4, 5, 4], [1, 1, 1, 1], [7.0, 9.0, 11.0, 20.0]), ref, v0, True)["PASS"]                # the reader's own gap where the head has an output
    assert not RE.v2_one(rows([0, 0, 0, 1], [3, 4, 5, 4], [1, 1, 1, 1], [7.0, 10.0, 11.0, 20.0], c=[0, 0, 0, 0.5]), ref, v0, True)["PASS"]   # a closing input in VD5
    assert RE.v2_one(rows([0, 0, 0, 1], [3, 4, 5, 4], [1, 1, 1, 1], [7.0, 10.0, 11.0, 20.0], c=[0, 0, 0, 0.5]), ref, v0, False)["PASS"]    # allowed in VD5c
    assert not RE.v2_one(rows([0, 0], [3, 4], [1, 1], [7.0, 10.0]), ref, {(0, 4): 9.0}, True)["PASS"]                                    # no V0 row for a lead row without a head output
    assert not RE.v2_one(rows([0, 0], [3, 4], [0, 0], [7.0, 10.0]), ref, v0, True)["PASS"]                                              # no lead rows at all
    exp = {"vis_gap_file_sha256": "a" * 64, "vis_closing": 0}
    assert RE.v2_one(good, ref, v0, True, meta_expect=exp, meta={"vis_gap_file_sha256": "a" * 64, "vis_closing": 0})["PASS"]
    assert not RE.v2_one(good, ref, v0, True, meta_expect=exp, meta={"vis_gap_file_sha256": "b" * 64, "vis_closing": 0})["PASS"]
    assert not RE.v2_one(good, ref, v0, True, meta_expect=exp, meta={"vis_gap_file_sha256": "a" * 64, "vis_closing": 1})["PASS"]


def write_logs(pull, run, m, tick, mid, meta):
    d = Path(pull) / ("I10%s_s%d" % (run, m)) / "ticks"
    d.mkdir(parents=True, exist_ok=True)
    for nm, r in (("rule_log_I10%s_s%d.npz" % (run, m), tick), ("rule_log_mid_I10%s_s%d.npz" % (run, m), mid)):
        np.savez(str(d / nm), ci=r[0], j=r[1], present=r[2], g_src=r[3], c_src=r[4], **{k: np.asarray(v) for k, v in meta.items()})


def reader_gap(ci, j):
    return np.float32(20.0 + 0.1 * j + ci)


def synth_pull(tmp_path, J, dumps, rec, closing=False, spoil=None, skip_rr=False):
    pull = tmp_path / "pull"
    refs = {m: MVG.second_routine(J, dumps[m]) for m in (0, 1)}
    n = np.asarray(J["n"])
    for m in (0, 1):
        for arm, run in RE.RUN.items():
            tick_c, tick_j, tick_g, tick_p, mid_c, mid_j, mid_g, mid_p = [], [], [], [], [], [], [], []
            for c in range(len(n)):
                for j in range(int(n[c])):
                    hg = refs[m].get((c, j))
                    g = reader_gap(c, j) if (arm == "V0" or hg is None) else np.float32(hg)
                    (tick_c if j % 2 == 0 else mid_c).append(c)
                    (tick_j if j % 2 == 0 else mid_j).append(j)
                    (tick_g if j % 2 == 0 else mid_g).append(g)
                    (tick_p if j % 2 == 0 else mid_p).append(True)
            cl = [0.0] * len(tick_c), [0.0] * len(mid_c)
            if arm == "VD5c":
                cl = [-0.5] * len(tick_c), [-0.5] * len(mid_c)
            if spoil and spoil[0] == (arm, m):
                tick_g[spoil[1]] = tick_g[spoil[1]] + 0.01
            meta = {"vis_gap_file_sha256": "" if arm == "V0" else rec["files"]["VGAP_s%d" % m]["file_sha256"], "vis_closing": 1 if arm == "VD5c" else 0}
            write_logs(pull, run, m, rows(tick_c, tick_j, tick_p, tick_g, cl[0]), rows(mid_c, mid_j, mid_p, mid_g, cl[1]), meta)
        if skip_rr:
            continue
        cc = [c for c in range(len(n)) for j in range(int(n[c]))]
        jj = [j for c in range(len(n)) for j in range(int(n[c]))]
        rr = rows(cc, jj, [True] * len(cc), [reader_gap(c, j) for c, j in zip(cc, jj)])
        write_logs(pull, RE.BANKED["RR"], m, rr, rr, {})
    return pull


class Ns:
    pass


@pytest.fixture(scope="module")
def pulled(world, tmp_path_factory):
    tp = tmp_path_factory.mktemp("pull")
    ej = tp / "join.npz"
    np.savez(str(ej), **world["J"])
    root = tp / "dump"
    MVG.synthetic_meta(world["dumps"], str(root))
    return {"tp": tp, "ej": str(ej), "root": str(root)}


def test_control_v2_from_files_end_to_end_and_a_planted_fault(world, pulled, tmp_path):
    pull = synth_pull(tmp_path, world["J"], world["dumps"], world["rec"])
    a = Ns()
    a.eval_join, a.dump_root, a.pull = pulled["ej"], pulled["root"], str(pull)
    c2, rows_all, J = RE.v2_from_files(a, world["rec"])
    assert c2["PASS"], c2
    assert sorted(c2["per_run"]) == ["VD5_s0", "VD5_s1", "VD5c_s0", "VD5c_s1"]
    assert c2["per_run"]["VD5c_s0"]["n_lead_rows_closing_nonzero"] > 0 and c2["per_run"]["VD5_s0"]["n_rows_closing_nonzero"] == 0
    sh = RE.lead_shares({k: tuple(np.concatenate([rows_all["%s_s%d" % (k, s)][i] for s in (0, 1)]) for i in range(5)) for k in ("V0", "VD5c", "VD5", "RR")}, J)
    assert sh["share_of_rule_steps_with_a_lead"]["RR"] == 1.0 and 0.0 < sh["share_of_valid_label_lead_frames_the_reader_marks"] <= 1.0 and 0.0 <= sh["share_of_marked_frames_with_no_label_lead"] < 1.0
    pull2 = synth_pull(tmp_path / "b", world["J"], world["dumps"], world["rec"], spoil=(("VD5", 1), 5))
    a.pull = str(pull2)
    assert not RE.v2_from_files(a, world["rec"])[0]["PASS"]


def test_printed_shares_fall_back_to_v0_when_the_banked_rr_logs_are_not_in_the_pull_folder(world, pulled, tmp_path):
    pull = synth_pull(tmp_path, world["J"], world["dumps"], world["rec"], skip_rr=True)
    a = Ns()
    a.eval_join, a.dump_root, a.pull = pulled["ej"], pulled["root"], str(pull)
    c2, rows_all, J = RE.v2_from_files(a, world["rec"])
    assert c2["PASS"] and rows_all["rr_source"].startswith("run V0")


def test_controls_v3_v4_v5():
    assert RE.ctl_v3(0.0954, 0.0, 0.0169)["PASS"] and RE.ctl_v3(0.09544, 0.00001, 0.01694)["PASS"]
    assert not RE.ctl_v3(0.0954, 0.0, 0.0175)["PASS"] and not RE.ctl_v3(0.0953, 0.0, 0.0169)["PASS"] and not RE.ctl_v3(None, 0.0, 0.0169)["PASS"]
    assert RE.ctl_v4("12 passed in 3s")["PASS"] and not RE.ctl_v4("11 passed, 1 failed in 3s")["PASS"] and not RE.ctl_v4("")["PASS"] and not RE.ctl_v4("1 passed, 2 error in 1s")["PASS"]
    assert RE.ctl_v5(True, True, True)["PASS"] and not RE.ctl_v5(True, True, False)["PASS"] and not RE.ctl_v5(False, True, True)["PASS"] and not RE.ctl_v5(None, True, True)["PASS"]


def test_control_v0_on_the_record_and_the_files(world):
    gdir = str(world["base"] / "clean")
    ejs = "x" * 64
    assert not RE.ctl_v0(world["rec"], gdir, ejs)["PASS"]                       # the record is for another evaluation join
    rec = copy.deepcopy(world["rec"])
    rec["eval_join_sha256"] = ejs
    assert RE.ctl_v0(rec, gdir, ejs)["PASS"]
    assert not RE.ctl_v0(rec, str(world["base"]), ejs)["PASS"]                   # the files are not where the record says
    assert not RE.ctl_v0(None, gdir, ejs)["PASS"]


def test_a_withheld_payload_and_the_names_stage_refuses_with_a_control_failed(tmp_path):
    ok = {k: {"PASS": True} for k in RE.CONTROLS}
    bad = dict(ok, v2={"PASS": False})
    p = payload()
    r = RE.assemble_controls(bad, p)
    assert r["controls_all_pass"] is False and "payload" not in r and "withheld" in r
    cf = tmp_path / "c.json"
    cf.write_text(json.dumps(r), encoding="utf-8")
    out = tmp_path / "res.json"
    assert RE.main(["--stage", "names", "--controls", str(cf), "--out", str(out)]) == 3 and not out.exists()
    r2 = RE.assemble_controls(ok, p)
    cf2 = tmp_path / "c2.json"
    cf2.write_text(json.dumps(r2), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(cf2), "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["controls_sha256"] == RE.sha_file(str(cf2)) and res["names"]["CARD"].startswith("VISRULE10K-") and "VD5c_KD5" in res["names"]
    miss = {k: v for k, v in ok.items() if k != "v5"}
    cf3 = tmp_path / "c3.json"
    cf3.write_text(json.dumps(RE.assemble_controls(miss, p)), encoding="utf-8")
    assert RE.main(["--stage", "names", "--controls", str(cf3), "--out", str(tmp_path / "res3.json")]) == 3


def test_reader_source_carries_the_strategic_family_the_ref_kd5_and_the_engine_bootstrap_setting():
    t = (HERE / "read_visrule10k.py").read_text(encoding="utf-8")
    assert "a.n_boot_control = 300" in t and t.index("a = ap.parse_args(argv)") < t.index("a.n_boot_control = 300")
    assert "R.strategic_grid(" in t and "S.pair_read(" in t and 'pairs_def["V0-VS-RR"]' in t and "-VS-KD5" in t
    assert "I10RLcKF5P30Hb20pKD5" in (QUEUE.read_text(encoding="utf-8")) and "RLcKF5P30Hb20pKD5" in t


def test_podread_text():
    t = (HERE / "podread_visrule10k.sh").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=-1" in t and "VISRULE10K_CONTROLS.json" in t and "VISRULE10K_RESULT.json" in t
    for x in ("V0", "VD5c", "VD5"):
        assert x in t
    for bad in (" rm ", " mv ", "kill", "pkill", "CROPRULE"):
        assert bad not in t, bad
    assert "ALREADY-READ" in t and "QUEUE-STOPPED" in t


def test_texts_hold_no_banned_words():
    for f in ("make_wrapper_v.py", "check_wrapper_v.py", "pod_check_wrapper_v.sh", "make_vis_gapfile.py", "make_visrule_queue.py", "pod_visrule10k_queue.sh", "read_visrule10k.py", "podread_visrule10k.sh",
              "push_visrule.sh", "pull_visrule.sh", "VISRULE_DEFS.md", "test_visrule.py"):
        p = HERE / f
        if not p.is_file():
            continue
        t = p.read_text(encoding="utf-8")
        assert f == "test_visrule.py" or not [w for w in BANNED if w in t], f


def test_selftests_run():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for args, last in (([str(HERE / "read_visrule10k.py"), "--selftest"], "ZZ SELFTEST-READ-VISRULE10K ok: True"), ([str(HERE / "make_vis_gapfile.py"), "--selftest"], "ZZ MAKE-VIS-GAPFILE-SELFTEST PASS"),
                       ([str(HERE / "check_wrapper_v.py"), "--selftest"], "ZZ CHECK-WRAPPER-V-SELFTEST PASS")):
        r = subprocess.run([sys.executable] + args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=300)
        assert r.stdout.strip().splitlines()[-1] == last, (args, r.stdout[-400:])
