"""clr_test_edits.py -- the edits make_clr_tests.py applies to the (role-renamed) CL tests.  ("all", name, old, new) = replace every occurrence (>= 1); ("edit", name, old, new) = exact once;
("span", name, start, end, new) = replace from the exactly-once start up to (not including) the exactly-once end."""

READ_EDITS, GATES_EDITS = [], []

# =============================================================================================== test_clr_read.py
R = READ_EDITS.append
R(("edit", "src", 'SRC = f"{PK}/code/cl_read.py"', 'SRC = f"{PK}/code/clr_read.py"'))
R(("edit", "doc", "of code/cl_read.py (the CLR read: PREREG_LEVELHEAD_CL.md sections 3-5).", "of code/clr_read.py (the CLR read: PREREG_LEVELHEAD_A2.md; the rules by reference to PREREG_LEVELHEAD_CL.md sections 4-5)."))
R(("span", "decision", "def test_decision_table_section_5():", '@pytest.mark.parametrize("mode", ["ka-null", "ka-ri"])', r'''def test_decision_table_a2_outcomes():
    D = C.decide
    r = D(True, True, True, 2, False)
    assert r["label"] == "D-HOLDS" and "signal the PI for scale" in r["next"] and r["counts_as_result"] is True
    r = D(True, True, False, 2, False)
    assert r["label"] == "P-HOLDS-G-OR-D-FAILS" and "CLR-W2" in r["next"] and "(W / 2)" in r["next"] and "G holds, D fails" in r["notes"]
    r = D(True, False, False, 0, False)
    assert r["label"] == "P-HOLDS-G-OR-D-FAILS" and "G fails" in r["notes"]
    r = D(False, True, False, 0, False)
    assert r["label"] == "P-FAILS-ATTR-NOT-SEPARATED-BETTER" and "LLR" in r["next"] and "O-sensitivity" in r["next"] and r["counts_as_result"] is True
    r = D(False, True, False, 2, False)
    assert r["label"] == "P-FAILS-ATTR-SEPARATED-BETTER" and "UNCOMMITTED" in r["next"] and r["counts_as_result"] is True
    r = D(False, True, False, 1, False)
    assert r["label"] == "P-FAILS-ATTR-SEPARATED-BETTER" and any("ONE seed" in n for n in r["notes"])
    r = D(False, False, False, 0, False)
    assert r["label"] == "P-FAILS-ATTR-NOT-SEPARATED-BETTER" and any("G fails too" in n for n in r["notes"])


def test_leak_borne_overrides_the_label_whatever_else_it_reads():
    r = C.decide(True, True, True, 2, True)                                     # a DOMINANT-looking row that is camera-clock borne
    assert r["label"] == "LEAK-BORNE" and r["underlying"] == "D-HOLDS" and r["counts_as_result"] is False
    assert "camera-clock" in r["next"] and "never a result" in r["next"]
    r = C.decide(False, True, False, 0, True)
    assert r["label"] == "LEAK-BORNE" and r["underlying"] == "P-FAILS-ATTR-NOT-SEPARATED-BETTER" and "LLR" in r["next"]       # next as for 'P fails'
    r = C.decide(False, True, False, 2, True)
    assert "UNCOMMITTED" in r["next"]


def test_attribution_rule_counts_separated_better_seeds_and_the_logged_scene_does_not_decide():
    ri = {"s0": cell(True, 0.4), "s1": cell(False, 0.0)}
    lg = {"s0": cell(False, 0.0), "s1": cell(False, 0.0)}
    a = C.rule_attr(ri, lg)
    assert a["n_seeds_separated_better_reindexed"] == 1 and a["SEPARATED_BETTER_AT_ANY_SEED"] is True and a["n_seeds_separated_better_logged_beside"] == 0
    a = C.rule_attr(lg, ri)                                                     # the same cells in the other slot: the reindexed scene decides, the logged one is beside
    assert a["SEPARATED_BETTER_AT_ANY_SEED"] is False and a["n_seeds_separated_better_reindexed"] == 0 and a["n_seeds_separated_better_logged_beside"] == 1


''' ))
R(("span", "ka", '@pytest.mark.parametrize("mode", ["ka-null", "ka-ri"])', "def blend(z, label, alpha):", r'''@pytest.mark.parametrize("mode", ["ka-null", "ka-ri", "ka-cl"])
def test_reader_known_answers(tmp_path, mode):
    assert C.main(["--mode", mode, "--raw", str(tmp_path)]) == 0
    res = json.load(open(tmp_path / f"KA_READER_{mode}.json", encoding="utf-8"))
    assert res["KA_PASS"] is True and all(res["ka_checks"].values())
    assert res["outcome"]["label"] == "P-FAILS-ATTR-NOT-SEPARATED-BETTER"       # the arm is its base (null) / the reindexed slots are null / CL's own arrays with a null attribution base: nothing moved
    if mode in ("ka-ri", "ka-cl"):
        assert len(res["ka_checks"]) >= 40                                      # ka-ri: every one of the 19 rows x 2 seeds against RI_C0_READ.json; ka-cl: every contrast of CL_READ.json
    if mode == "ka-cl":
        assert res["ka_checks"]["rows_compared_at_least_500"] and any(k.endswith("equals_CL_READ[RICL_minus_RIC0]") for k in res["ka_checks"])


# ------------------------------------------------------------------------------------------------ (3) end to end on real arrays with DESIGNED arms
''' ))
R(("span", "designed", "def designed(cl=0.0, ri=0.0, riclc=None, ri_edit=None):", "def test_designed_arm_closes_exactly_alpha(reads):", r'''def designed(cl=0.0, ri=0.0, riclc=None, ri_edit=None, rap=0.0):
    """CLR's arrays as designed arms (``cl`` logged, ``ri`` reindexed share closed against C0 / RIC0); the attribution base RAPR-P01 closes ``rap`` of its own excess on both clocks (rap = 0: the base IS C0 / RIC0)."""
    out = {}
    for s in C.SEEDS:
        z, _ = C.reused_arrays(s)
        z["CLR"] = blend(z["C0"], C.M1, cl) if cl else z["C0"]
        z["CLRC"], z["FZCLR"] = z["C0C"], z["FZC0"]
        z["RICLR"] = blend(z["RIC0"], C.M1, ri) if ri else z["RIC0"]
        if ri_edit:
            z["RICLR"] = ri_edit(z["RICLR"])
        z["RICLRC"] = riclc(z["RIC0C"]) if riclc else z["RIC0C"]
        z["RIRAPRP01"] = blend(z["RIC0"], C.M1, rap) if rap else z["RIC0"]
        z["RAPRP01"] = blend(z["C0"], C.M1, rap) if rap else z["C0"]
        out[s] = z
    return out


@pytest.fixture(scope="module")
def reads():
    R = {}
    R["p_holds"] = C.read_all(designed(cl=0.8, ri=0.8), 2000, log=lambda *a: None)
    R["moved"] = C.read_all(designed(cl=0.3, ri=0.3), 2000, log=lambda *a: None)
    R["moved_base_moved"] = C.read_all(designed(cl=0.3, ri=0.3, rap=0.3), 2000, log=lambda *a: None)
    R["ri_only"] = C.read_all(designed(cl=0.0, ri=0.3), 2000, log=lambda *a: None)
    R["leak"] = C.read_all(designed(cl=0.8, ri=0.0), 2000, log=lambda *a: None)
    R["d_holds"] = C.read_all(designed(cl=0.8, ri=0.8, riclc=dominant), 2000, log=lambda *a: None)
    R["g_fails"] = C.read_all(designed(cl=0.8, ri=0.8, ri_edit=lambda z: worsen(z, "ADE6", 0.5)), 2000, log=lambda *a: None)
    return R


''' ))
R(("span", "p_fails_moved", "def test_branch_P_fails_M1_moved(reads):", "def test_branch_leak_borne_when_only_the_logged_clock_moves(reads):", r'''def test_branch_P_fails_attribution_separated_better_is_the_uncommitted_cell(reads):
    r = reads["moved"]                                                          # the base IS RIC0 here, so the attribution contrast equals the C0 contrast
    assert r["rules"]["P"]["PASS"] is False and r["rules"]["P"]["n_seeds_separated_better"] == 2
    assert r["rules"]["ATTRIBUTION"]["n_seeds_separated_better_reindexed"] == 2
    assert r["outcome"]["label"] == "P-FAILS-ATTR-SEPARATED-BETTER" and "UNCOMMITTED" in r["outcome"]["next"]
    for k in ("s0", "s1"):
        assert r["seeds"][k]["RICLR_minus_RIRAPRP01"][C.M1]["delta"] == r["seeds"][k]["RICLR_minus_RIC0"][C.M1]["delta"]
        assert r["seeds"][k]["CLR_minus_RAPRP01"][C.M1]["delta"] == r["seeds"][k]["CLR_minus_C0"][C.M1]["delta"]


def test_branch_P_fails_attribution_not_separated_better_when_the_base_moved_as_much(reads):
    r = reads["moved_base_moved"]                                               # RAPR-P01 closes the same 30 % of its excess: CLR adds nothing over it, though CLR - C0 moved
    for k in ("s0", "s1"):
        assert r["seeds"][k]["RICLR_minus_RIC0"][C.M1]["separated"] is True and r["seeds"][k]["RICLR_minus_RIRAPRP01"][C.M1]["delta"] == 0.0
        assert r["seeds"][k]["m1_cell_attribution_reindexed"]["better"] is False
    assert r["rules"]["P"]["PASS"] is False and r["rules"]["ATTRIBUTION"]["SEPARATED_BETTER_AT_ANY_SEED"] is False
    assert r["outcome"]["label"] == "P-FAILS-ATTR-NOT-SEPARATED-BETTER" and "LLR" in r["outcome"]["next"]


def test_the_reindexed_attribution_decides_not_the_logged_one(reads):
    r = reads["ri_only"]                                                        # moves on the reindexed clock only: the logged attribution is null
    assert r["rules"]["ATTRIBUTION"]["n_seeds_separated_better_reindexed"] == 2 and r["rules"]["ATTRIBUTION"]["n_seeds_separated_better_logged_beside"] == 0
    assert r["outcome"]["label"] == "P-FAILS-ATTR-SEPARATED-BETTER"


''' ))
R(("edit", "leak_label", 'r["outcome"]["underlying"] == "P-FAILS-M1-NOT-MOVED" and r["outcome"]["counts_as_result"] is False', 'r["outcome"]["underlying"] == "P-FAILS-ATTR-NOT-SEPARATED-BETTER" and r["outcome"]["counts_as_result"] is False'))
R(("edit", "contrasts", 'for c in ("RICLR_minus_RIC0", "CLR_minus_C0", "RICLRC_minus_P10", "CLRC_minus_P10", "FZCLR_minus_FZC0", "C0REP_minus_C0", "RIC0_minus_P10"):',
   'for c in ("RICLR_minus_RIC0", "CLR_minus_C0", "RICLRC_minus_P10", "CLRC_minus_P10", "FZCLR_minus_FZC0", "C0REP_minus_C0", "RIC0_minus_P10", "RICLR_minus_RIRAPRP01", "CLR_minus_RAPRP01", "RIRAPRP01_minus_P10"):'))
R(("edit", "md", 'assert "**Outcome: D-HOLDS**" in md and "## 19 rows: RICLR - RIC0" in md and "## Families" in md and "STRATEGIC" in md',
   'assert "**Outcome: D-HOLDS**" in md and "## 19 rows: RICLR - RIC0" in md and "## 19 rows: RICLR - RIRAPRP01" in md and "## Families" in md and "STRATEGIC" in md and "ATTRIBUTION" in md'))
R(("edit", "refusal_names", '("rollout_RICLR_s0", "rollout_FZCLR_s1", "KA_NEW", "CLR_PRECHECK")', '("rollout_RICLR_s0", "rollout_FZCLR_s1", "rollout_RIRAPRP01_s1", "CLR_PRECHECK")'))
R(("edit", "card_rules_ctl", 'assert ctl["card_rules"]["PASS"] is True and ctl["KA_NEW"]["PASS"] is False and ctl["CLR_PRECHECK"]["PASS"] is False',
   'assert ctl["card_rules"]["PASS"] is True and ctl["card_a2"]["PASS"] is True and ctl["KA_NEW"]["PASS"] is True and ctl["CLR_PRECHECK"]["PASS"] is False'))
R(("edit", "pins", "def test_pins_hold():\n    assert C.card_rules_sha() == C.CARD_RULES_SHA\n",
   r'''def test_a2_card_control_fails_on_a_changed_card(tmp_path, monkeypatch):
    empty = tmp_path / "out"
    empty.mkdir()
    assert C.run_controls(str(empty), str(tmp_path))["card_a2"]["PASS"] is True
    monkeypatch.setattr(C, "A2_CARD_SHA", "0" * 64)                             # the frozen card was edited after the pin
    assert C.run_controls(str(empty), str(tmp_path))["card_a2"]["PASS"] is False


def test_pins_hold():
    assert C.card_rules_sha() == C.CARD_RULES_SHA
    assert C.sha256_file(C.CARD) == C.A2_CARD_SHA == open(C.A2_FROZEN, encoding="utf-8").readline().split()[0]
'''))
R(("edit", "pins_len", "assert all(v[\"PASS\"] for v in ctl.values()) and len(ctl) == 6", "assert all(v[\"PASS\"] for v in ctl.values()) and len(ctl) == 7"))
R(("edit", "flags_tail", '    assert C.expected_flags("RICLRC") == "--smooth sg72 --scene reindexed --feasible kbox"',
   '    assert C.expected_flags("RICLRC") == "--smooth sg72 --scene reindexed --feasible kbox"\n    assert C.expected_flags("RIRAPRP01") == "--smooth sg72 --scene reindexed"                  # the attribution base: RAPR-P01\'s own checkpoint, same flags as RICLR\n'
   '    assert C.RUN_OF["RIRAPRP01"] == "RAPRP01" and all(C.RUN_OF[r] == "CLR" for r in C.NEW_ROLES if r != "RIRAPRP01") and len(C.NEW_ROLES) == 6'))

# =============================================================================================== test_clr_chain_gates.py
G = GATES_EDITS.append
G(("all", "vacuous_marker", 'json.dumps({"evidence": {"ckpt_sha256": digest}})', 'json.dumps({"evidence": {"ckpt_sha256": digest, "recover_n": 123, "level_head": "on", "w_level": 0.1}})'))
G(("edit", "import_sys", "import subprocess\n", "import subprocess\nimport sys\n"))
G(("edit", "rdb_test", 'RDB_TEST = "F:/TanitAD-scratch/motion_c/cl_read/pt/rdb_gates"', 'RDB_TEST = "F:/TanitAD-scratch/motion_c/clr_read/pt/rdb_gates"\nA1 = "E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_readout/code"'))
G(("all", "load_pre", 'f"{PK}/code/cl_precheck.py"', 'f"{PK}/code/clr_precheck.py"'))
G(("all", "load_read", 'f"{PK}/code/cl_read.py"', 'f"{PK}/code/clr_read.py"'))
G(("all", "markers_dir", "cl_markers", "clr_markers"))
G(("all", "chain_name", "cl_read_chain.sh", "clr_read_chain.sh"))
G(("all", "ro_marker", "CLRO.", "CLRRO."))
G(("span", "fixture", '@pytest.fixture(scope="module")\ndef rdb():', "def pre(rdb, tmp_path, w=None):", r'''@pytest.fixture(scope="module")
def rdb():
    """A synthetic pulled-run tree: both seeds carry the level-ON smoke checkpoint as CLR (RAPR-P01's real argv + the two level tokens, a SHA256SUMS line, the box-2 marker with the recovery evidence) and a level-OFF
    copy of it as RAPRP01 (the attribution base).  It is NOT a CLR checkpoint and nothing here reads, writes or fakes a file in E:/Projects/TanitAD-artifacts/v8f-recover/."""
    import torch
    assert os.path.isfile(NZ), "run cl_read/code/smoke_level_on.py first (it builds the level-ON smoke checkpoints)"
    shutil.rmtree(RDB_TEST, ignore_errors=True)
    os.makedirs(f"{RDB_TEST}/clr_markers")
    digest = sha(NZ)
    off = f"{RDB_TEST}/_base_level_off.pt"
    ck = torch.load(NZ, map_location="cpu", weights_only=False)
    ck["cfg"] = dict(ck["cfg"], level_head="off")
    torch.save(ck, off)
    del ck
    off_digest = sha(off)
    for s in (0, 1):
        base = json.load(open(f"{ART}/v8f-recover/M2b_s{s}_recft_RAPRP01/argv.json", encoding="utf-8"))
        b = f"{RDB_TEST}/M2b_s{s}_recft_RAPRP01"
        os.makedirs(b)
        json.dump(base, open(f"{b}/argv.json", "w", encoding="utf-8"))
        shutil.copyfile(off, f"{b}/ckpt_last.pt")
        open(f"{b}/SHA256SUMS.txt", "w", encoding="utf-8", newline="\n").write(f"{off_digest} *ckpt_last.pt\n")
        d = f"{RDB_TEST}/M2b_s{s}_recft_CLR"
        os.makedirs(d)
        shutil.copyfile(NZ, f"{d}/ckpt_last.pt")
        open(f"{d}/SHA256SUMS.txt", "w", encoding="utf-8", newline="\n").write(f"{digest} *ckpt_last.pt\n")
        json.dump({"argv": base["argv"] + ["--level-head", "on", "--w-level", "0.1"], "sha": "x"}, open(f"{d}/argv.json", "w", encoding="utf-8"))
        json.dump({"tag": f"CLR_{s}", "evidence": {"ckpt_sha256": digest, "recover_n": 123, "level_head": "on", "w_level": 0.1}}, open(f"{RDB_TEST}/clr_markers/RUN_CLR_{s}.done", "w", encoding="utf-8"))
    open(f"{RDB_TEST}/FROZEN_W.txt", "w", encoding="utf-8", newline="\n").write("# the W frozen before launch\nW=0.1\n")
    return RDB_TEST


''' ))
G(("edit", "new_precheck_tests", 'def fake_outputs(root, ckpt=("a" * 64, "b" * 64)):', r'''def test_precheck_refuses_a_marker_without_recovery_evidence(rdb, tmp_path):
    mk = f"{rdb}/clr_markers/RUN_CLR_1.done"
    for ev in ({"recover_n": 0}, {"recover_n": None}, {"recover_n": True}, {"level_head": "off"}, {"w_level": 0.2}):
        with edit(mk, lambda t, ev=ev: json.dumps({**json.loads(t), "evidence": {**json.loads(t)["evidence"], **ev}})):
            rc, r = pre(rdb, tmp_path)
        assert rc == 8 and any("marker evidence" in f for f in r["failures"]), ev


def test_precheck_refuses_an_argv_without_the_recovery_lever(rdb, tmp_path):
    """The base is C0's argv (no --recover-* token at all) and the CLR argv = it + the level tokens: every other audit check holds, the recovery check alone refuses."""
    c0 = json.load(open(f"{ART}/v8f-recover/M2b_s0_recft_C0/argv.json", encoding="utf-8"))
    with edit(f"{rdb}/M2b_s0_recft_RAPRP01/argv.json", lambda t: json.dumps(c0)), edit(f"{rdb}/M2b_s0_recft_CLR/argv.json", lambda t: json.dumps({"argv": c0["argv"] + ["--level-head", "on", "--w-level", "0.1"]})):
        rc, r = pre(rdb, tmp_path)
    a = r["seeds"]["s0"]["argv_audit"]
    assert rc == 8 and a["recovery_lever_is_RAPRP01s"] is False and a["nothing_changed"] and a["nothing_removed"] and a["added_exactly_the_two_level_tokens"]


def test_precheck_checks_the_attribution_base_checkpoint(rdb, tmp_path):
    rc, r = pre(rdb, tmp_path)
    s0 = r["seeds"]["s0"]
    assert rc == 0 and s0["raprp01_ckpt_sha256"] == s0["raprp01_sha256sums_line"] and s0["raprp01_cfg_level_head"] == "off"
    with edit(f"{rdb}/M2b_s0_recft_RAPRP01/SHA256SUMS.txt", lambda t: "0" * 64 + " *ckpt_last.pt\n"):
        rc, r = pre(rdb, tmp_path)
    assert rc == 8 and any("RAPRP01 ckpt sha256" in f for f in r["failures"])
    d = f"{rdb}/M2b_s1_recft_RAPRP01"
    os.replace(f"{d}/ckpt_last.pt", f"{d}/ckpt_last.pt.keep")
    try:
        shutil.copyfile(NZ, f"{d}/ckpt_last.pt")                                # a level-ON checkpoint as the base: refused
        with edit(f"{d}/SHA256SUMS.txt", lambda t: f"{sha(NZ)} *ckpt_last.pt\n"):
            rc, r = pre(rdb, tmp_path)
        assert rc == 8 and any("level_head on" in f for f in r["failures"])
        os.remove(f"{d}/ckpt_last.pt")                                          # an absent base: exit 5
        rc, r = pre(rdb, tmp_path)
        assert rc == 5 and any("attribution base" in f for f in r["failures"])
    finally:
        if os.path.exists(f"{d}/ckpt_last.pt"):
            os.remove(f"{d}/ckpt_last.pt")
        os.replace(f"{d}/ckpt_last.pt.keep", f"{d}/ckpt_last.pt")


def test_generated_files_equal_their_generators():
    for name in ("make_clr_from_cl.py", "make_clr_precheck.py", "make_clr_readout.py", "make_clr_tests.py"):
        r = subprocess.run([sys.executable, "-I", f"{PK}/code/{name}", "--check"], capture_output=True, text=True)
        assert r.returncode == 0, (name, r.stdout, r.stderr)


def test_readout_clone_differs_from_A1_only_in_the_five_path_lines():
    a = open(f"{A1}/cl_readout.py", encoding="utf-8", newline="").read().splitlines()
    b = open(f"{PK}/code/clr_readout.py", encoding="utf-8", newline="").read().splitlines()
    changed = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(a) == len(b) and len(changed) == 5 and all(("CLR" in y or "clr_read" in y) for _, y in changed)


def fake_outputs(root, ckpt=("a" * 64, "b" * 64), base_ckpt=("e" * 64, "f" * 64)):'''))
G(("edit", "fake_ckpt", '"ckpt": f"E:/x/v8f-recover/M2b_s{s}_recft_CLR/ckpt_last.pt", "ckpt_sha256": ckpt[s],',
   '"ckpt": f"E:/x/v8f-recover/M2b_s{s}_recft_{C.RUN_OF[role]}/ckpt_last.pt", "ckpt_sha256": ckpt[s] if C.RUN_OF[role] == "CLR" else base_ckpt[s],'))
G(("edit", "fake_precheck", '"seeds": {f"s{s}": {"ckpt_sha256": ckpt[s], "e_level_abs_sum": 1.0} for s in (0, 1)}}',
   '"seeds": {f"s{s}": {"ckpt_sha256": ckpt[s], "raprp01_ckpt_sha256": base_ckpt[s], "e_level_abs_sum": 1.0} for s in (0, 1)}}'))
G(("edit", "pass_test", "def test_reader_controls_pass_on_consistent_outputs(tmp_path):\n    out, raw = fake_outputs(tmp_path)\n",
   "def test_reader_controls_pass_on_consistent_outputs(tmp_path, monkeypatch):\n    out, raw = fake_outputs(tmp_path)\n    monkeypatch.setattr(C, \"CL_RAW\", raw)\n"))
G(("edit", "n_rollouts", '== 10', '== 12'))
G(("edit", "param", '"precheck_fail", "missing_marker"])', '"precheck_fail", "missing_marker", "cl_not_clr_run", "attribution_base_ckpt", "attribution_ckpt_vs_precheck"])'))
G(("edit", "refuse_sig", "def test_reader_controls_refuse_each_inconsistency(tmp_path, what):\n    out, raw = fake_outputs(tmp_path)\n",
   "def test_reader_controls_refuse_each_inconsistency(tmp_path, what, monkeypatch):\n    out, raw = fake_outputs(tmp_path)\n    monkeypatch.setattr(C, \"CL_RAW\", raw)\n"))
G(("edit", "refuse_branches", '    else:\n        os.remove(o("done_RICLRC_s1.txt")); bad = "rollout_RICLRC_s1"',
   '''    elif what == "cl_not_clr_run":                  # CL's checkpoint path contains CLR's as a prefix: the control must still refuse it
        d = rj(o("done_CLR_s1.txt")); d["ckpt"] = "E:/x/v8f-recover/M2b_s1_recft_CL/ckpt_last.pt"; wj(o("done_CLR_s1.txt"), d); bad = "rollout_CLR_s1"
    elif what == "attribution_base_ckpt":           # RIRAPRP01 rolled from the CLR checkpoint instead of RAPR-P01's
        d = rj(o("done_RIRAPRP01_s0.txt")); d["ckpt"] = "E:/x/v8f-recover/M2b_s0_recft_CLR/ckpt_last.pt"; wj(o("done_RIRAPRP01_s0.txt"), d); bad = "rollout_RIRAPRP01_s0"
    elif what == "attribution_ckpt_vs_precheck":    # the rolled RAPR-P01 checkpoint is not the one the pre-check hashed
        d = rj(os.path.join(raw, "CLR_PRECHECK.json")); d["seeds"]["s0"]["raprp01_ckpt_sha256"] = "9" * 64; wj(os.path.join(raw, "CLR_PRECHECK.json"), d); bad = "CLR_PRECHECK"
    else:
        os.remove(o("done_RICLRC_s1.txt")); bad = "rollout_RICLRC_s1"'''))
G(("span", "chain_table", "def test_chain_table_matches_the_readers_flags_and_priority():", "def test_chain_refuses_without_pins_or_checkpoints(tmp_path):", r'''def test_production_main_runs_end_to_end_on_pre_extracted_arrays(tmp_path, monkeypatch):
    """The production branch of main (controls -> extract -> read_all -> CLR_READ.json + RESULTS.md) on FAKE rollout outputs whose arrays are pre-placed (the extractor returns an existing file): every new role is a copy
    of the base it must equal (RICLR := RIC0 ..., the attribution base RIRAPRP01 := RIC0), so the result is the null read."""
    out, raw = fake_outputs(tmp_path)
    monkeypatch.setattr(C, "CL_RAW", raw)
    arr = tmp_path / "arrays"
    arr.mkdir()
    base = {"CLR": "C0", "CLRC": "C0C", "FZCLR": "FZC0", "RICLR": "RIC0", "RICLRC": "RIC0C", "RIRAPRP01": "RIC0"}
    for s in (0, 1):
        for role, b in base.items():
            shutil.copyfile(f"{C.REUSED[b]}/{b}_s{s}.npz", arr / f"{role}_s{s}.npz")
    md = tmp_path / "RESULTS.md"
    rc = C.main(["--mode", "production", "--out-root", out, "--raw", raw, "--arr-dir", str(arr), "--results-md", str(md), "--work-dir", str(tmp_path / "w")])
    assert rc == 0
    res = json.load(open(f"{raw}/CLR_READ.json", encoding="utf-8"))
    assert "refused" not in res and all(v["PASS"] for v in res["controls"].values()) and res["outcome"]["label"] == "P-FAILS-ATTR-NOT-SEPARATED-BETTER"
    assert md.exists() and "**Outcome: P-FAILS-ATTR-NOT-SEPARATED-BETTER**" in open(md, encoding="utf-8").read()
    for k in ("s0", "s1"):
        assert all(r["delta"] == 0.0 for r in res["seeds"][k]["RICLR_minus_RIC0"].values()) and all(r["delta"] == 0.0 for r in res["seeds"][k]["RICLR_minus_RIRAPRP01"].values())


def test_chain_table_matches_the_readers_flags_and_priority():
    r = subprocess.run([BASH, f"{PK}/clr_read_chain.sh", "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0
    table = [l.split("\t") for l in r.stdout.splitlines() if l.count("\t") == 4]
    assert [t[0] for t in table] == ["RICLR_s0", "RICLR_s1", "RICLRC_s0", "RICLRC_s1", "RIRAPRP01_s0", "RIRAPRP01_s1", "CLR_s0", "CLR_s1", "CLRC_s0", "CLRC_s1", "FZCLR_s0", "FZCLR_s1"]     # the card's priority order
    for tag, seed, scene, feas, run in table:
        role = tag.rsplit("_s", 1)[0]
        assert seed == tag[-1] and (scene, feas) == C.NEW_ROLES[role] and run == C.RUN_OF[role], tag
    src = open(f"{PK}/clr_read_chain.sh", encoding="utf-8").read()
    assert 'local flags="--smooth sg72 --scene $scene"' in src and '[ "$feas" != "off" ] && flags="$flags --feasible $feas"' in src and "--freeze-card-ref" in src
    assert "--replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks" in src and "--dump-heads --device cpu" in src
    assert '[ "$scene" = "reindexed" ] && th=$TH_RI' in src and "TH_LOGGED=${CLR_TH_LOGGED:-16}" in src and "TH_RI=${CLR_TH_RI:-12}" in src     # reindexed (incl. RIRAPRP01) at 12, logged / hold_now at 16
    assert "ckpt_last.pt" in src and "M2b_s${2}_recft_${5}" in src and "clr_markers" in open(f"{PK}/code/clr_precheck.py", encoding="utf-8").read()
    assert "CLRRO.done" in src and "CLRRO.fail" in src and "CLREAD.done" in src and "CLREAD.fail" in src and "CLRREADOUT.done" in src and "clr_readout.py" in src and "clr_wiring.py" in src


''' ))

READ = {"sha": "b9db7ea85abc9c2223e5be817a7f1565c93bc80632c01e01cc047e08108c91b9", "edits": READ_EDITS}
GATES = {"sha": "797663fb965cb544987100f1fdfd8b9fe5011f4ca5bc7315d93cfa4c895a352d", "edits": GATES_EDITS}
