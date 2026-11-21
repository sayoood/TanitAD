

# ================================================================================================================================ ROLLHIST P03: the RHp3 arm (Amendment 7 DRAFT): p = 0.3 on the SAME bank
# The chain checker pins the RH flag set against the template (p = 0.1). RHp3 = the RH argv with EXACTLY --dagger-p 0.3 and its own --out. Everything else is the same argv, bank, init, steps, seed.
def p3_toks(r, p="0.3"):
    t = r.setf(list(r.rh_toks), "--dagger-p", p)
    return r.setf(t, "--out", os.path.join(r.tmp, "runs", f"RHp3_s{r.seed}", "train"))


def mint_p3(r, toks=None, token="causal_g1x", sc=None, pair_toks=None, mod=None, tag="", template=None):
    mod = mod or NEW
    r.patch_heavy(mod)
    toks = toks or p3_toks(r)
    rp, bp = r.argv_file(f"RHp3_s{r.seed}{tag}", toks), r.argv_file(f"B0_s{r.seed}{tag}_p3", pair_toks or r.b0_toks)
    a = r.mint_ns(f"RHp3_s{r.seed}", rp, f"RHp3_s{r.seed}{tag}", pair=bp, token=token, selfcheck=sc or r.sc_rh, template=template)
    return mod.mint(a), a.gate_dir


def test_p3_the_arms_exist_and_the_original_kinds_are_unchanged():
    assert NEW.RH_ARMS["RHp3_s0"] == ("RHp3", 0) and NEW.RH_ARMS["RHp3_s1"] == ("RHp3", 1)
    assert NEW.RH_ARMS["RH_s0"] == ("RH", 0) and NEW.RH_ARMS["B0_s1"] == ("B0", 1)
    assert NEW.required_checks("RHp3_s0", "B") == NEW.required_checks("RH_s0", "B")
    assert NEW.ARMS["RHp3_s1"] == NEW.ARMS["RH_s1"]                              # same set, stage, seed, dagger-seed: a different dose and nothing else


@pytest.mark.parametrize("seed", [0, 1])
def test_p3_rhp3_mints_pass_and_binds_the_pair(tmp_path, seed):
    r = Rig(tmp_path / f"p3s{seed}", seed=seed)
    toks = p3_toks(r)
    # the argv differs from the RH argv in EXACTLY --dagger-p and --out (the card's one lever for this arm)
    a, b = NEW.parse_argv(toks), NEW.parse_argv(r.rh_toks)
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"--dagger-p", "--out"}
    assert a["--dagger-p"] == ["0.3"] and b["--dagger-p"] == ["0.1"]
    pr, gd = mint_p3(r, toks)
    assert pr == [], pr
    tok = json.load(open(os.path.join(gd, "gate_token.json")))
    assert tok["arm"] == f"RHp3_s{seed}" and tok["stage"] == "B" and tok["argv_sha256"] == NEW.argv_sha(toks)
    assert tok["checks"]["C-RH-PAIR"]["readout_token"] == "causal_g1x"
    assert set(NEW.required_checks(f"RHp3_s{seed}", "B")) <= {c for c, v in tok["checks"].items() if v["pass"]}
    assert tok["checks"]["C-RH-BANKGEN"]["pass"] and tok["checks"]["C-BANK"]["pass"] and tok["checks"]["G-DAGGER"]["pass"]
    assert tok["data"]["bank/bank.json"]["sha256"] == r.bank_sha                 # the SAME bank the RH arm trained on
    # the existing RH and B0 arms of the same rig still mint (the new kind did not move them)
    assert r.mint("RH")[0] == [] and r.mint("B0")[0] == []


@pytest.mark.parametrize("p", ["0.5", "0.1", "0.25", "1.0", "0.30", "0", "0.0", "-0.3", "abc"])
def test_p3_any_other_dose_is_refused(rig, p):                      # mutation arm: p = 0.5 (and every other value) is refused; the dose is the literal 0.3
    pr, _ = mint_p3(rig, p3_toks(rig, p))
    assert pr and refused_by(pr, "C-ARGV"), (p, pr)


@pytest.mark.parametrize("how", ["lr", "steps", "added", "removed", "batch", "data_path", "kin", "seed", "wacc", "dagger_seed", "p_gt", "ego_perturb"])
def test_p3_a_second_changed_flag_is_refused(rig, how):             # mutation arm: p = 0.3 plus ONE more changed flag is refused (the "fifth changed flag" rule, at the RHp3 kind)
    t = p3_toks(rig)
    t = {"lr": lambda: rig.setf(t, "--lr", "2e-4"), "steps": lambda: rig.setf(t, "--steps", "3001"), "added": lambda: t + ["--foo", "1"], "removed": lambda: rig.dropf(t, "--detach-upper"),
         "batch": lambda: rig.setf(t, "--batch", "32"), "data_path": lambda: rig.setf(t, "--tracks", rig.files["lead"]), "kin": lambda: rig.setf(t, "--kin-mode", "xy"),
         "seed": lambda: rig.setf(t, "--seed", "7"), "wacc": lambda: rig.setf(t, "--w-acc", "10"), "dagger_seed": lambda: rig.setf(t, "--dagger-seed", "5"),
         "p_gt": lambda: rig.setf(t, "--p-gt-end", "0.5"), "ego_perturb": lambda: rig.setf(t, "--ego-perturb-p", "0.1")}[how]()
    pr, _ = mint_p3(rig, t)
    assert pr and refused_by(pr, "C-ARGV"), (how, pr)


def test_p3_the_rh_arm_still_refuses_a_changed_dose(rig):           # the new kind did not loosen RH: an RH_s0 argv at p = 0.3 is refused
    t = rig.setf(list(rig.rh_toks), "--dagger-p", "0.3")
    pr, _ = rig.mint("RH", rh_toks=t)
    assert pr and refused_by(pr, "C-ARGV"), pr


def test_p3_rh_dose_under_the_rhp3_name_is_refused(rig):            # RH's p = 0.1 under the RHp3 arm name: refused (the dose is the arm's identity)
    pr, _ = mint_p3(rig, rig.setf(list(rig.rh_toks), "--out", os.path.join(rig.tmp, "runs", "RHp3_s0", "train")))
    assert pr and refused_by(pr, "C-ARGV"), pr


def test_p3_out_must_name_the_arm(rig):
    pr, _ = mint_p3(rig, rig.setf(p3_toks(rig), "--out", os.path.join(rig.tmp, "runs", "RH_s0", "train")))
    assert pr and refused_by(pr, "C-ARGV"), pr


def test_p3_the_pair_is_the_existing_b0_and_a_moved_b0_is_refused(rig):
    pr, _ = mint_p3(rig, pair_toks=rig.setf(rig.b0_toks, "--lr", "2e-4"))
    assert pr and refused_by(pr, "C-RH-PAIR"), pr
    pr, _ = mint_p3(rig, pair_toks=rig.setf(rig.b0_toks, "--dagger-p", "0.1"), tag="b")
    assert pr and refused_by(pr, "C-RH-PAIR"), pr


def test_p3_needs_the_declaration_token_and_the_right_one(rig):
    pr, _ = mint_p3(rig, token="")
    assert pr, pr
    pr, _ = mint_p3(rig, token="causal_g1", tag="b")                                  # the stock pair check hard-codes causal_g1x
    assert pr, pr


def test_p3_bank_checks_apply_to_the_new_kind(rig):
    t = rig.setf(p3_toks(rig), "--dagger-bank-sha256", "0" * 64)
    pr, _ = mint_p3(rig, t)
    assert pr, pr                                                                     # a declared sha that is not the bank's
    rig.make_bank("causal_g1x", with_generator=False)                                  # a bank whose header carries no generator record reads pinned: C-RH-BANKGEN refuses
    t2 = rig.setf(p3_toks(rig), "--dagger-bank-sha256", rig.bank_sha)
    pr, _ = mint_p3(rig, t2, tag="nogen")
    assert pr and refused_by(pr, "C-RH-BANKGEN"), pr


def test_p3_the_selfcheck_must_carry_the_declaration_and_pass_the_probe(rig):
    sc = json.load(open(rig.sc_rh))
    sc["levers"]["dagger"]["dagger_bank_readout"] = "pinned"
    pr, _ = mint_p3(rig, sc=put_json(os.path.join(rig.tmp, "sc3_bad.json"), sc))
    assert pr and refused_by(pr, "G-DAGGER"), pr
    sc = json.load(open(rig.sc_rh))
    sc["dagger_probe"]["control_p0_grad_diff"] = 0.1                                   # a control that must read exactly 0.0
    pr, _ = mint_p3(rig, sc=put_json(os.path.join(rig.tmp, "sc3_bad2.json"), sc), tag="b")
    assert pr and refused_by(pr, "G-DAGGER"), pr


def test_p3_stage_a_refused(rig):
    rig.patch_heavy(NEW)
    a = rig.mint_ns("RHp3_s0", rig.argv_file("RHp3_s0", p3_toks(rig)), "RHp3_A", pair=rig.argv_file("B0_s0", rig.b0_toks), token="causal_g1x", selfcheck=rig.sc_rh, stage="A")
    assert NEW.mint(a)


def test_p3_cli_mint_then_run_only_on_the_token(rig, capsys):
    rig.patch_heavy(NEW)
    open(os.path.join(rig.root, "stack", "scripts", "v8f_mini_train.py"), "w").write("import sys, os\nopen(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'STARTED'), 'w').write(' '.join(sys.argv[1:4]))\n")
    toks = p3_toks(rig)
    rp, bp = rig.argv_file("RHp3_cli", toks), rig.argv_file("B0_cli3", rig.b0_toks)
    gate = os.path.join(rig.tmp, "gate", "RHp3_cli")
    started = os.path.join(rig.root, "stack", "scripts", "STARTED")
    assert NEW.main(_cli(rig, "mint", "RHp3_s0", rp, bp, gate, "causal_g1x", rig.sc_rh)) == 0
    assert "TOKEN-WRITTEN RHp3_s0" in capsys.readouterr().out
    assert NEW.main(_cli(rig, "run", "RHp3_s0", rp, bp, gate, "causal_g1x", rig.sc_rh)) == 0 and os.path.isfile(started)
    os.remove(started)
    # an argv edited after the mint (p 0.3 -> 0.5), a wrong token and a missing token each REFUSE with rc 2 and start nothing
    edited = rig.argv_file("RHp3_edit", rig.setf(toks, "--dagger-p", "0.5"))
    assert NEW.main(_cli(rig, "run", "RHp3_s0", edited, bp, gate, "causal_g1x", rig.sc_rh)) == 2
    assert NEW.main(_cli(rig, "run", "RHp3_s0", rp, bp, gate, "causal_g1", rig.sc_rh)) == 2
    assert NEW.main(_cli(rig, "run", "RHp3_s0", rp, bp, os.path.join(rig.tmp, "gate", "none"), "causal_g1x", rig.sc_rh)) == 2
    assert not os.path.isfile(started)
