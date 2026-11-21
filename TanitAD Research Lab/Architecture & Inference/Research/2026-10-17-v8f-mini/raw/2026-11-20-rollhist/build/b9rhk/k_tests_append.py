

# ================================================================================================================================ ROLLHIST RHK (Amendment 8 DRAFT): RH's dose and flags, a bank generated under readout K at bank_k/
# RHK argv = the RH argv with the bank flags (--dagger-bank, --dagger-bank-sha256, --dagger-bank-readout) and --out changed; --dagger-p stays 0.1.
K = "causal_g0x"


def krig(tmp, seed=0, gen=K, bank_sub="bank_k"):
    return Rig(tmp, seed=seed, gen=gen, bank_sub=bank_sub)


def k_toks(r, tok=K, p=None):
    t = r.setf(list(r.rh_toks), "--out", os.path.join(r.tmp, "runs", f"RHK_s{r.seed}", "train"))
    t = r.setf(t, "--dagger-bank-readout", tok)
    return t if p is None else r.setf(t, "--dagger-p", p)


def mint_k(r, toks=None, token=K, sc=None, pair_toks=None, mod=None, tag="", template=None):
    mod = mod or NEW
    r.patch_heavy(mod)
    toks = toks or k_toks(r)
    rp, bp = r.argv_file(f"RHK_s{r.seed}{tag}", toks), r.argv_file(f"B0_s{r.seed}{tag}_k", pair_toks or r.b0_toks)
    a = r.mint_ns(f"RHK_s{r.seed}", rp, f"RHK_s{r.seed}{tag}", pair=bp, token=token, selfcheck=sc or r.sc_rh, template=template)
    return mod.mint(a), a.gate_dir


def test_k_the_arms_exist_and_the_original_kinds_are_unchanged():
    assert NEW.RH_ARMS["RHK_s0"] == ("RHK", 0) and NEW.RH_ARMS["RHK_s1"] == ("RHK", 1)
    assert NEW.RH_ARMS["RH_s0"] == ("RH", 0) and NEW.RH_ARMS["RHp3_s1"] == ("RHp3", 1) and NEW.RH_ARMS["B0_s1"] == ("B0", 1)
    assert NEW.required_checks("RHK_s0", "B") == NEW.required_checks("RH_s0", "B")
    assert NEW.ARMS["RHK_s1"] == NEW.ARMS["RH_s1"]
    assert NEW.RHK_P == "0.1" and NEW.RHK_TOKENS == ("causal_g0x",) and NEW.RH_TOKENS == ("causal_g1x", "causal_g1")      # K is NOT admitted to RH / RHp3's token set


@pytest.mark.parametrize("seed", [0, 1])
def test_k_rhk_mints_pass_and_binds_the_k_bank(tmp_path, seed):
    r = krig(tmp_path / f"ks{seed}", seed=seed)
    toks = k_toks(r)
    a, b = NEW.parse_argv(toks), NEW.parse_argv(r.rh_toks)
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"--out"}      # in this rig the bank path and sha are already the K bank's; against the real RH argv the bank flags move too
    assert a["--dagger-p"] == ["0.1"] and a["--dagger-bank-readout"] == [K]
    pr, gd = mint_k(r, toks)
    assert pr == [], pr
    tok = json.load(open(os.path.join(gd, "gate_token.json")))
    assert tok["arm"] == f"RHK_s{seed}" and tok["stage"] == "B" and tok["argv_sha256"] == NEW.argv_sha(toks)
    assert tok["checks"]["C-RH-PAIR"]["readout_token"] == K and tok["checks"]["C-RH-BANKGEN"]["readout_token"] == K
    assert set(NEW.required_checks(f"RHK_s{seed}", "B")) <= {c for c, v in tok["checks"].items() if v["pass"]}
    assert tok["data"]["bank/bank.json"]["sha256"] == r.bank_sha
    r2 = Rig(tmp_path / f"ks{seed}_rh", seed=seed)                                                          # the RH and B0 kinds of an RH rig still mint (the new kind did not move them)
    assert r2.mint("RH")[0] == [] and r2.mint("B0")[0] == []
    r3 = Rig(tmp_path / f"ks{seed}_p3", seed=seed)                                                          # ... and so does RHp3 at 0.3
    t3 = r3.setf(r3.setf(list(r3.rh_toks), "--dagger-p", "0.3"), "--out", os.path.join(r3.tmp, "runs", f"RHp3_s{seed}", "train"))
    r3.patch_heavy(NEW)
    a3 = r3.mint_ns(f"RHp3_s{seed}", r3.argv_file("p3", t3), "p3", pair=r3.argv_file("b0p3", r3.b0_toks), token="causal_g1x", selfcheck=r3.sc_rh)
    assert NEW.mint(a3) == []


def test_k_the_stock_pair_check_is_not_patched():
    assert hashlib.sha256(open(PAIR_CHECK, "rb").read()).hexdigest() == "0dc30d9ef773be761a9b36a07d0fd73ef770e87d3384826230ba798db560869f"
    m = _load("stock_pair_k", PAIR_CHECK)
    assert m.RH_READOUT == "causal_g1x" and m.DECL["--dagger-bank-readout"] == "causal_g1x"


@pytest.mark.parametrize("p", ["0.3", "0.5", "0.25", "1.0", "0.10", "0", "0.0", "-0.1", "abc"])
def test_k_any_dose_but_0_1_is_refused(tmp_path, p):                       # mutation arm: a dose other than the literal 0.1
    r = krig(tmp_path / "kp")
    pr, _ = mint_k(r, k_toks(r, p=p))
    assert pr and refused_by(pr, "C-ARGV"), (p, pr)


def test_k_the_rh_bank_under_the_rhk_name_is_refused(tmp_path):            # mutation arm: the RH bank (causal_g1x, under bank/) behind the RHK name
    r = Rig(tmp_path / "kr")                                               # a real RH world: bank under .../bank/s0, generator causal_g1x
    t = r.setf(list(r.rh_toks), "--out", os.path.join(r.tmp, "runs", "RHK_s0", "train"))
    pr, _ = mint_k(r, t, token="causal_g1x")                               # RH's own token: not in RHK's token set, and the path is not bank_k
    assert pr and refused_by(pr, "C-ARGV"), pr
    pr, _ = mint_k(r, r.setf(t, "--dagger-bank-readout", K), token=K, tag="b")      # declares K over the RH bank: the generator record reads causal_g1x, and the path is bank/
    assert pr and refused_by(pr, "C-ARGV", "C-RH-BANKGEN"), pr
    assert "bank_k" in "\n".join(pr) and "generator record reads readout 'causal_g1x'" in "\n".join(pr)


def test_k_a_cstar_bank_under_the_rhk_name_is_refused(tmp_path):           # mutation arm: a C* bank placed AT bank_k/ and named RHK
    r = krig(tmp_path / "kc", gen="causal_g1x")                            # generator causal_g1x under bank_k/s0
    pr, _ = mint_k(r, k_toks(r), token=K)                                  # the argv declares K: the bank header says C*
    assert pr and refused_by(pr, "C-RH-BANKGEN"), pr
    pr, _ = mint_k(r, k_toks(r, tok="causal_g1x"), token="causal_g1x", tag="b")     # ... and declaring C* is refused by the kind's token set
    assert pr and refused_by(pr, "C-ARGV"), pr


@pytest.mark.parametrize("how", ["lr", "steps", "added", "removed", "batch", "data_path", "kin", "seed", "wacc", "dagger_seed", "p_gt", "ego_perturb", "init"])
def test_k_a_second_changed_flag_is_refused(tmp_path, how):                # mutation arm: the K bank flags plus ONE more changed flag
    r = krig(tmp_path / "ks2")
    t = k_toks(r)
    t = {"lr": lambda: r.setf(t, "--lr", "2e-4"), "steps": lambda: r.setf(t, "--steps", "3001"), "added": lambda: t + ["--foo", "1"], "removed": lambda: r.dropf(t, "--detach-upper"),
         "batch": lambda: r.setf(t, "--batch", "32"), "data_path": lambda: r.setf(t, "--tracks", r.files["lead"]), "kin": lambda: r.setf(t, "--kin-mode", "xy"),
         "seed": lambda: r.setf(t, "--seed", "7"), "wacc": lambda: r.setf(t, "--w-acc", "10"), "dagger_seed": lambda: r.setf(t, "--dagger-seed", "5"),
         "p_gt": lambda: r.setf(t, "--p-gt-end", "0.5"), "ego_perturb": lambda: r.setf(t, "--ego-perturb-p", "0.1"), "init": lambda: r.setf(t, "--init-ckpt", r.files["ckpt1"])}[how]()
    pr, _ = mint_k(r, t)
    assert pr and refused_by(pr, "C-ARGV"), (how, pr)


def test_k_the_bank_must_be_under_bank_k(tmp_path):
    r = krig(tmp_path / "kb", bank_sub="bank")                             # a valid K bank, but at .../bank/s0: the arm must name bank_k
    pr, _ = mint_k(r)
    assert pr and refused_by(pr, "C-ARGV") and "bank_k" in "\n".join(pr), pr
    r = krig(tmp_path / "kb2", seed=1, bank_sub="bank_k")
    pr, _ = mint_k(r, r.setf(k_toks(r), "--dagger-bank", os.path.join(r.tmp, "bank_k", "s0")))      # seed 1's arm naming seed 0's K bank
    assert pr and refused_by(pr, "C-ARGV"), pr


def test_k_the_rh_and_rhp3_kinds_refuse_the_k_token(tmp_path):             # the new token did not loosen the old kinds
    r = krig(tmp_path / "kt")
    pr, _ = r.mint("RH", token=K)
    assert pr and refused_by(pr, "C-ARGV"), pr
    t3 = r.setf(r.setf(list(r.rh_toks), "--dagger-p", "0.3"), "--out", os.path.join(r.tmp, "runs", "RHp3_s0", "train"))
    r.patch_heavy(NEW)
    a3 = r.mint_ns("RHp3_s0", r.argv_file("p3k", t3), "p3k", pair=r.argv_file("b0p3k", r.b0_toks), token=K, selfcheck=r.sc_rh)
    pr = NEW.mint(a3)
    assert pr and refused_by(pr, "C-ARGV"), pr


def test_k_the_pair_is_the_existing_b0_and_a_moved_b0_is_refused(tmp_path):
    r = krig(tmp_path / "kpair")
    pr, _ = mint_k(r, pair_toks=r.setf(r.b0_toks, "--lr", "2e-4"))
    assert pr and refused_by(pr, "C-RH-PAIR"), pr
    pr, _ = mint_k(r, pair_toks=r.setf(r.b0_toks, "--dagger-p", "0.1"), tag="b")
    assert pr and refused_by(pr, "C-RH-PAIR"), pr


def test_k_needs_the_declaration_token_and_the_right_one(tmp_path):
    r = krig(tmp_path / "kd")
    pr, _ = mint_k(r, token="")
    assert pr, pr
    pr, _ = mint_k(r, token="causal_g1x", tag="b")                         # K argv, C* token on the CLI
    assert pr and refused_by(pr, "C-ARGV"), pr
    pr, _ = mint_k(r, token="causal_g0", tag="c")
    assert pr, pr
    pr, _ = mint_k(r, k_toks(r, tok="causal_g1x"), token=K, tag="d")       # the argv says C*, the CLI says K
    assert pr and refused_by(pr, "C-ARGV"), pr


def test_k_the_swap_for_the_stock_pair_check_is_exactly_one_token():
    toks = ["--dagger-p", "0.1", "--dagger-bank-readout", K, "--dagger-bank-kappa", "0.2"]
    assert NEW.rh_pair_problems(toks, [], "causal_g1x", None, PAIR_CHECK, True) == ["RHK readout token 'causal_g1x' is not one of ('causal_g0x',)"]
    assert "does not declare" in NEW.rh_pair_problems(["--dagger-p", "0.1"], [], K, None, PAIR_CHECK, True)[0]
    assert NEW.rh_pair_problems(toks, [], K, None, PAIR_CHECK, False)[0].startswith("readout 'causal_g0x': stack/scripts/rollhist_pair_check.py hard-codes")      # no swap unless the kind is RHK


def test_k_the_selfcheck_must_carry_the_k_declaration_and_pass_the_probe(tmp_path):
    r = krig(tmp_path / "ksc")
    sc = json.load(open(r.sc_rh))
    sc["levers"]["dagger"]["dagger_bank_readout"] = "causal_g1x"
    pr, _ = mint_k(r, sc=put_json(os.path.join(r.tmp, "sck_bad.json"), sc))
    assert pr and refused_by(pr, "G-DAGGER"), pr
    sc = json.load(open(r.sc_rh))
    sc["dagger_probe"]["control_p0_grad_diff"] = 0.1
    pr, _ = mint_k(r, sc=put_json(os.path.join(r.tmp, "sck_bad2.json"), sc), tag="b")
    assert pr and refused_by(pr, "G-DAGGER"), pr


def test_k_bank_checks_apply_to_the_new_kind(tmp_path):
    r = krig(tmp_path / "kbc")
    pr, _ = mint_k(r, r.setf(k_toks(r), "--dagger-bank-sha256", "0" * 64))
    assert pr, pr
    r.make_bank(K, with_generator=False)
    pr, _ = mint_k(r, r.setf(k_toks(r), "--dagger-bank-sha256", r.bank_sha), tag="nogen")
    assert pr and refused_by(pr, "C-RH-BANKGEN"), pr


def test_k_out_must_name_the_arm_and_stage_a_is_refused(tmp_path):
    r = krig(tmp_path / "ko")
    pr, _ = mint_k(r, r.setf(k_toks(r), "--out", os.path.join(r.tmp, "runs", "RH_s0", "train")))
    assert pr and refused_by(pr, "C-ARGV"), pr
    r.patch_heavy(NEW)
    a = r.mint_ns("RHK_s0", r.argv_file("RHK_s0a", k_toks(r)), "RHK_A", pair=r.argv_file("B0a", r.b0_toks), token=K, selfcheck=r.sc_rh, stage="A")
    assert NEW.mint(a)


def test_k_cli_mint_then_run_only_on_the_token(tmp_path, capsys):
    rig = krig(tmp_path / "kcli")
    rig.patch_heavy(NEW)
    open(os.path.join(rig.root, "stack", "scripts", "v8f_mini_train.py"), "w").write("import sys, os\nopen(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'STARTED'), 'w').write(' '.join(sys.argv[1:4]))\n")
    toks = k_toks(rig)
    rp, bp = rig.argv_file("RHK_cli", toks), rig.argv_file("B0_clik", rig.b0_toks)
    gate = os.path.join(rig.tmp, "gate", "RHK_cli")
    started = os.path.join(rig.root, "stack", "scripts", "STARTED")
    assert NEW.main(_cli(rig, "mint", "RHK_s0", rp, bp, gate, K, rig.sc_rh)) == 0
    assert "TOKEN-WRITTEN RHK_s0" in capsys.readouterr().out
    assert NEW.main(_cli(rig, "run", "RHK_s0", rp, bp, gate, K, rig.sc_rh)) == 0 and os.path.isfile(started)
    os.remove(started)
    for edited_t in (rig.setf(toks, "--dagger-p", "0.3"), rig.setf(toks, "--dagger-bank-readout", "causal_g1x")):          # an argv edited after the mint refuses with rc 2 and starts nothing
        assert NEW.main(_cli(rig, "run", "RHK_s0", rig.argv_file("RHK_edit", edited_t), bp, gate, K, rig.sc_rh)) == 2
    assert NEW.main(_cli(rig, "run", "RHK_s0", rp, bp, gate, "causal_g1x", rig.sc_rh)) == 2
    assert NEW.main(_cli(rig, "run", "RHK_s0", rp, bp, os.path.join(rig.tmp, "gate", "none"), K, rig.sc_rh)) == 2
    assert not os.path.isfile(started)
