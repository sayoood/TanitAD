"""PREREG_V8F_ORECOVER Amendment 3 (Master Mind, pre-data): ORC-0 names ONLY `--ego-perturb-p 0` from the ego-perturbation family (no vscale / vmax / adrop / yawrate-std / seed); ORC keeps all five law flags.
Applies the matching edits to stack/tests/test_v8f_orc_chain.py (every replacement must match exactly once). Usage: apply_amendment3_tests.py <path of test_v8f_orc_chain.py>"""
import sys

p = sys.argv[1]
with open(p, "rb") as fh:
    raw = fh.read()
assert b"\r" not in raw
s = raw.decode("utf-8")
n_edits = 0


def rep(old, new):
    global s, n_edits
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)
    n_edits += 1


# ---- docstring
rep('''"""PREREG_V8F_ORECOVER on box 2 (Amendments 1 and 2):''', '''"""PREREG_V8F_ORECOVER on box 2 (Amendments 1, 2 and 3):''')
rep('''ORC and ORC-0 are ONE argv except
                  `--ego-perturb-p` and `--out`, at both seeds; every lever of Amendment 1 NAMED once with its registered value;''',
    '''ORC-0 is ORC's argv MINUS the five
                  ego-perturbation LAW flags (Amendment 3), with `--ego-perturb-p 0` and its own `--out`, at both seeds; every lever of Amendment 1 NAMED once with its registered value (ORC-0: none of the five);''')
rep('''A BLOCKER is pinned, not hidden: see `test_blocker_*` (the trainer's own gate REFUSES ORC-0's registered argv)."""''',
    '''AMENDMENT 3 closes the blocker this build found (the trainer's gate refused ORC-0 with the law flags): ORC-0's gates PASS on the REAL trainer, unpatched, at both seeds (`test_orc0_gates_pass_*`), and a law flag
named on ORC-0 is refused (`test_a_law_flag_*`: by the trainer's gate when its value is not the default, by the chain's assertions whenever it is named at all)."""''')

# ---- constants + helpers
rep('''ORC_STAGES = ("gateorc", "trainorc", "evalorc")
''', '''ORC_STAGES = ("gateorc", "trainorc", "evalorc")
LAW = ("--ego-perturb-vscale", "--ego-perturb-vmax", "--ego-perturb-adrop", "--ego-perturb-yawrate-std", "--ego-perturb-seed")     # PREREG Amendment 3: named by ORC, by NONE of ORC-0
''')
rep('''def _flags(tokens):''', '''def _drop_flags(tokens, drop):
    """tokens without the flags in ``drop`` AND all the values that follow each of them (up to the next `--flag`): vscale carries two."""
    out, k = [], 0
    while k < len(tokens):
        if tokens[k] in drop:
            k += 1
            while k < len(tokens) and not tokens[k].startswith("--"):
                k += 1
            continue
        out.append(tokens[k])
        k += 1
    return out


def _flags(tokens):''')

# ---- _registered: ORC-0 names none of the law flags
rep('''    """The fine-tune of Amendment 1 as the flags that follow M2b's own (%A%), in the order the queue writes them."""
    return ["--init-ckpt", rf"D:\\tanitad\\runs\\v8f_mini_full\\M2b_s{seed}\\ckpt_last.pt", "--steps", "3000", "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2",
            "--ego-perturb-p", "0.3" if arm == "orc" else "0", "--ego-perturb-vscale", "0.5", "1.0", "--ego-perturb-vmax", "8.0", "--ego-perturb-adrop", "1.5", "--ego-perturb-yawrate-std", "2.0",
            "--ego-perturb-seed", str(seed)]
''', '''    """The fine-tune of Amendments 1 and 3 as the flags that follow M2b's own (%A%), in the order the queue writes them: ORC names the five law flags after --ego-perturb-p, ORC-0 names NONE of them."""
    return ["--init-ckpt", rf"D:\\tanitad\\runs\\v8f_mini_full\\M2b_s{seed}\\ckpt_last.pt", "--steps", "3000", "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2",
            "--ego-perturb-p", "0.3" if arm == "orc" else "0",
            *(["--ego-perturb-vscale", "0.5", "1.0", "--ego-perturb-vmax", "8.0", "--ego-perturb-adrop", "1.5", "--ego-perturb-yawrate-std", "2.0", "--ego-perturb-seed", str(seed)] if arm == "orc" else [])]
''')

# ---- (A) static: the helper's text
rep('''    assert f == ('set "ORCF=--init-ckpt "%RUNSM%\\\\M2b_s%~2\\\\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %ORCP% --ego-perturb-vscale 0.5 1.0 '
                 '--ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed %~2"'), "the fine-tune's levers are written ONCE, here"
''', '''    assert f == ('set "ORCF=--init-ckpt "%RUNSM%\\\\M2b_s%~2\\\\ckpt_last.pt" --steps 3000 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --ego-perturb-p %ORCP%%ORCL%"'), \\
        "the fine-tune's levers are written ONCE, here: --ego-perturb-p, then %ORCL% (the five law flags for ORC, NOTHING for ORC-0: Amendment 3)"
    lw = 'if /i "%~1"=="orc" set "ORCL= --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed %~2"'
    assert arm.count('set "ORCL="') == 1 and arm.count(lw) == 1 and arm.index('set "ORCL="') < arm.index(lw) < arm.index(f), "ORCL is emptied, then set for ORC ONLY, then read by ORCF"
    assert all(sum(x.count(fl) for x in _code(_queue_text().split("\\n:orcarm\\n")[1].split("\\n"))) == 1 for fl in LAW), "each law flag is written once in the whole new region, inside ORC's ORCL"
    assert not any(fl in x for fl in LAW for x in arm if x != lw), "no other line of the helper names a law flag (ORC-0 reaches none)"
''')

# ---- (A) dry run: ORC-0 = ORC minus the law flags
rep('''def test_dry_run_orc_and_orc0_are_one_argv_except_the_perturbation_probability_and_the_output_directory(seed):
    o, o0 = _train_argv(_dry1(f"trainorc orc {seed}")), _train_argv(_dry1(f"trainorc orc0 {seed}"))
    assert _without(o, {"--out", "--ego-perturb-p"}) == _without(o0, {"--out", "--ego-perturb-p"}), "ORC and ORC-0 differ in nothing but --ego-perturb-p and --out (orc_read.py refuses otherwise)"
    fo, f0 = _flags(o), _flags(o0)
    assert fo["--ego-perturb-p"] == [["0.3"]] and f0["--ego-perturb-p"] == [["0"]], "always NAMED (the flag's default disables the lever), 0.3 for ORC and 0 for ORC-0"
    assert fo["--out"][0][0].endswith(f"M2b_s{seed}_orc") and f0["--out"][0][0].endswith(f"M2b_s{seed}_orc0") and o != o0
    gate_o, gate_0 = _train_argv(_dry1(f"gateorc orc {seed}")), _train_argv(_dry1(f"gateorc orc0 {seed}"))
    assert _without(gate_o, {"--out", "--ego-perturb-p"}) == _without(gate_0, {"--out", "--ego-perturb-p"}) and _flags(gate_o)["--ego-perturb-p"] == [["0.3"]] and _flags(gate_0)["--ego-perturb-p"] == [["0"]]
''', '''def test_dry_run_orc_argv_is_the_orc0_argv_plus_the_five_law_flags_with_p_0_3_vs_0_and_out_differing(seed):
    """PREREG Amendment 3: ORC argv == ORC-0 argv + the five law flags; the ONLY other differences are --ego-perturb-p (0.3 vs 0) and --out."""
    o, o0 = _train_argv(_dry1(f"trainorc orc {seed}")), _train_argv(_dry1(f"trainorc orc0 {seed}"))
    fo, f0 = _flags(o), _flags(o0)
    assert _drop_flags(o, {"--out", "--ego-perturb-p", *LAW}) == _drop_flags(o0, {"--out", "--ego-perturb-p"}), "ORC minus (law flags, p, out) == ORC-0 minus (p, out): nothing else differs (a second lever is a confound)"
    assert [k for k in LAW if k in f0] == [] and all(fo[k] for k in LAW) and all(len(fo[k]) == 1 for k in LAW), "ORC names each law flag once; ORC-0 names none of them"
    assert fo["--ego-perturb-p"] == [["0.3"]] and f0["--ego-perturb-p"] == [["0"]], "always NAMED (the flag's default disables the lever), 0.3 for ORC and 0 for ORC-0"
    assert [k for k in fo if k.startswith("--ego-perturb")] == ["--ego-perturb-p", *LAW] and [k for k in f0 if k.startswith("--ego-perturb")] == ["--ego-perturb-p"]
    assert o0[-2:] == ["--ego-perturb-p", "0"], "ORC-0's argv ENDS with --ego-perturb-p 0: nothing of the family follows"
    assert fo["--out"][0][0].endswith(f"M2b_s{seed}_orc") and f0["--out"][0][0].endswith(f"M2b_s{seed}_orc0") and o != o0
    gate_o, gate_0 = _train_argv(_dry1(f"gateorc orc {seed}")), _train_argv(_dry1(f"gateorc orc0 {seed}"))
    assert _drop_flags(gate_o, {"--out", "--ego-perturb-p", *LAW}) == _drop_flags(gate_0, {"--out", "--ego-perturb-p"}) and _flags(gate_o)["--ego-perturb-p"] == [["0.3"]] and _flags(gate_0)["--ego-perturb-p"] == [["0"]]
    assert [k for k in LAW if k in _flags(gate_0)] == [] and all(k in _flags(gate_o) for k in LAW), "the gates carry the same flags as their runs"
''')

# ---- (A) every lever named
rep('''    assert fl["--arm"] == [["M2b"]] and fl["--seed"] == [[str(seed)]] and fl["--grid"] == [["pool4x10"]], "M2b's training argv, at this seed: the --seed is the seed of the M2b checkpoint being fine-tuned"
''', '''    assert fl["--arm"] == [["M2b"]] and fl["--seed"] == [[str(seed)]] and fl["--grid"] == [["pool4x10"]], "M2b's training argv, at this seed: the --seed is the seed of the M2b checkpoint being fine-tuned"
    assert all((k in fl) == (arm == "orc") for k in LAW), "ORC names all five law flags, ORC-0 NONE (Amendment 3: the trainer's gate refuses a law flag set while p is 0)"
''')

# ---- (C) the synthetic fixture: the REAL trainer, unpatched, for all four runs
rep('''def _ft(ck, arm, seed, steps="3000"):
    return ["--init-ckpt", str(ck), "--steps", steps, "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", "0.3" if arm == "orc" else "0",
            "--ego-perturb-vscale", "0.5", "1.0", "--ego-perturb-vmax", "8.0", "--ego-perturb-adrop", "1.5", "--ego-perturb-yawrate-std", "2.0", "--ego-perturb-seed", str(seed)]
''', '''def _ft(ck, arm, seed, steps="3000"):
    return ["--init-ckpt", str(ck), "--steps", steps, "--lr", "1e-4", "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", "0.3" if arm == "orc" else "0",
            *(["--ego-perturb-vscale", "0.5", "1.0", "--ego-perturb-vmax", "8.0", "--ego-perturb-adrop", "1.5", "--ego-perturb-yawrate-std", "2.0", "--ego-perturb-seed", str(seed)] if arm == "orc" else [])]
''')
rep('''    GATE (`--selfcheck-only`, steps 3000) and, for seed 0, a 6-step TRAINING run. ORC-0 runs with the ONE clause of EgoPerturb.problems that the registered argv trips patched out IN PROCESS (the BLOCKER test below
    shows the unpatched trainer refuses it); the file on disk is never touched."""''',
    '''    GATE (`--selfcheck-only`, steps 3000) and, for seed 0, a 6-step TRAINING run. NOTHING is patched: ORC-0 is the Amendment 3 argv (`--ego-perturb-p 0`, none of the five law flags), which the trainer's own gate
    ACCEPTS; `test_orc0_gates_pass_*` pins that, and `test_a_law_flag_*` pins that it refuses the old shape."""''')
rep('''    orig = D.EgoPerturb.problems
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(D.EgoPerturb, "problems", lambda self: [m for m in orig(self) if "are set while ego_perturb_p is 0" not in m])
        for seed in (0, 1):
            src = root / "runsm" / f"M2b_s{seed}"
            assert TR.main(["--arm", "M2b", "--seed", str(seed), "--grid", "pool4x10", "--out", str(src), *SYN_A, "--steps", "1", "--ckpt-every", "1"]) == 0
            ck = src / "ckpt_last.pt"
            out["ck"][seed], out["sha"][seed] = ck, hashlib.sha256(ck.read_bytes()).hexdigest()
            for arm in ("orc", "orc0"):
                gate = root / "runsm" / f"gate_M2b-{arm}_s{seed}"
                assert TR.main(_argv(gate, ck, arm, seed, gate=True)) == 0
                out["gate"][(arm, seed)] = json.loads((gate / "selfcheck.json").read_text(encoding="utf-8"))
                if seed == 0:
                    run = root / "runsm" / f"M2b_s{seed}_{arm}"
                    assert TR.main(_argv(run, ck, arm, seed, steps="6")) == 0
                    out["run"][(arm, seed)] = run
    return out
''', '''    for seed in (0, 1):
        src = root / "runsm" / f"M2b_s{seed}"
        assert TR.main(["--arm", "M2b", "--seed", str(seed), "--grid", "pool4x10", "--out", str(src), *SYN_A, "--steps", "1", "--ckpt-every", "1"]) == 0
        ck = src / "ckpt_last.pt"
        out["ck"][seed], out["sha"][seed] = ck, hashlib.sha256(ck.read_bytes()).hexdigest()
        for arm in ("orc", "orc0"):
            gate = root / "runsm" / f"gate_M2b-{arm}_s{seed}"
            assert TR.main(_argv(gate, ck, arm, seed, gate=True)) == 0, f"the trainer's own gate must ACCEPT the registered {arm} argv at seed {seed}"
            out["gate"][(arm, seed)] = json.loads((gate / "selfcheck.json").read_text(encoding="utf-8"))
            if seed == 0:
                run = root / "runsm" / f"M2b_s{seed}_{arm}"
                assert TR.main(_argv(run, ck, arm, seed, steps="6")) == 0
                out["run"][(arm, seed)] = run
    return out
''')

# ---- (C) ORC-0 gate: the built law values are the trainer's defaults; a law flag named on ORC-0 is refused by the chain
rep('''    sc = good_selfcheck(synth, "orc0", 0)
    rc, out = _gate(tmp_path, sc, "orc0", 0, tl=_lines("orc", 0)[1], gl=_lines("orc", 0)[0])
    assert rc == 1 and "does not carry --ego-perturb-p 0 exactly once" in out, "ORC-0's gate record with ORC's lines (the arms swapped) is refused"
''', '''    sc = good_selfcheck(synth, "orc0", 0)
    rc, out = _gate(tmp_path, sc, "orc0", 0, tl=_lines("orc", 0)[1], gl=_lines("orc", 0)[0])
    assert rc == 1 and "does not carry --ego-perturb-p 0 exactly once" in out and "names --ego-perturb-yawrate-std on ORC-0" in out, "ORC-0's gate record with ORC's lines (the arms swapped) is refused"
    # Amendment 3: the BUILT law values of ORC-0 are the trainer's DEFAULTS (vmax 8.0 = ORC's, so the eligible denominator is the same); a built value moved is refused, each of the five
    for seed in (0, 1):
        sc = good_selfcheck(synth, "orc0", seed)
        ep = sc["levers"]["ego_perturb"]
        assert (ep["ego_perturb_vscale"], ep["ego_perturb_vmax"], ep["ego_perturb_adrop"], ep["ego_perturb_yawrate_std"], ep["ego_perturb_seed"]) == ([0.5, 1.0], 8.0, 1.5, 0.0, 0), "the REAL built values are the defaults"
        for k, v in (("ego_perturb_vscale", [0.6, 1.0]), ("ego_perturb_vmax", 6.0), ("ego_perturb_adrop", 1.0), ("ego_perturb_yawrate_std", 2.0), ("ego_perturb_seed", seed + 1)):
            bad = copy.deepcopy(sc)
            bad["levers"]["ego_perturb"][k] = v
            rc, out = _gate(tmp_path, bad, "orc0", seed)
            assert rc == 1 and f"built ego_perturb {k}" in out, (k, out)
    # ... and a law flag NAMED on either ORC-0 line is refused whatever its value (the trainer's guard refuses only non-default values; the chain closes the rest)
    g0, t0 = _lines("orc0", 0)
    for fl, val in (("--ego-perturb-vscale", "0.5 1.0"), ("--ego-perturb-vmax", "8.0"), ("--ego-perturb-adrop", "1.5"), ("--ego-perturb-yawrate-std", "2.0"), ("--ego-perturb-seed", "0")):
        sc = good_selfcheck(synth, "orc0", 0)
        rc, out = _gate(tmp_path, sc, "orc0", 0, gl=g0.replace(" --selfcheck-only", f" {fl} {val} --selfcheck-only"), tl=t0 + f" {fl} {val}")
        assert rc == 1 and f"names {fl} on ORC-0" in out, (fl, out[-500:])
        rc, out = _gate(tmp_path, sc, "orc0", 0, tl=t0 + f" {fl} {val}")
        assert rc == 1 and f"names {fl} on ORC-0" in out
''')

# ---- (C) a law flag missing on ORC: refused
rep('''             ("yaw std other", g, t.replace("--ego-perturb-yawrate-std 2.0", "--ego-perturb-yawrate-std 3.0"), "--ego-perturb-yawrate-std"),
''', '''             ("yaw std other", g, t.replace("--ego-perturb-yawrate-std 2.0", "--ego-perturb-yawrate-std 3.0"), "--ego-perturb-yawrate-std"),
             ("a law flag missing on ORC: yaw std", g, _drop(t, "--ego-perturb-yawrate-std"), "does not carry --ego-perturb-yawrate-std 2.0 exactly once"),
             ("a law flag missing on ORC: vscale", g, _drop(t, "--ego-perturb-vscale", 2), "does not carry --ego-perturb-vscale 0.5 1.0 exactly once"),
             ("a law flag missing on ORC: vmax", g, _drop(t, "--ego-perturb-vmax"), "does not carry --ego-perturb-vmax 8.0 exactly once"),
             ("a law flag missing on ORC: adrop", g, _drop(t, "--ego-perturb-adrop"), "does not carry --ego-perturb-adrop 1.5 exactly once"),
             ("a law flag missing on ORC: seed", g, _drop(t, "--ego-perturb-seed"), "does not carry --ego-perturb-seed 0 exactly once"),
''')

# ---- the blocker test -> Amendment 3 tests
i = s.index('''# =============================================================================================================================================
# THE BLOCKER, pinned and self-expiring''')
s = s[:i] + '''# =============================================================================================================================================
# AMENDMENT 3: the trainer's OWN gate accepts ORC-0 (it refused the earlier shape) and refuses a law flag on it
# =============================================================================================================================================
@pytest.mark.parametrize("seed", [0, 1])
def test_orc0_gates_pass_on_the_real_trainer_unpatched_at_both_seeds_with_the_probe_off_and_the_law_at_its_defaults(synth, seed):
    """The trainer is NOT patched anywhere in this file (the fixture ran `TR.main` on the Amendment 3 argv): the gate record it wrote is a PASS, the ego probe reads `off`, nothing of the law is moved from its default."""
    TR = synth["TR"]
    out = synth["root"] / f"orc0_gate_again_s{seed}"
    assert TR.main(_argv(out, synth["ck"][seed], "orc0", seed, gate=True)) == 0, "the trainer's own gate ACCEPTS ORC-0's registered argv"
    sc = json.loads((out / "selfcheck.json").read_text(encoding="utf-8"))
    assert sc["declared_vs_built"] == "PASS" and sc["ego_perturb_probe"]["pass"] is True and sc["ego_perturb_probe"]["ego_perturb"] == "off"
    ep = sc["levers"]["ego_perturb"]
    assert ep["ego_perturb_p"] == 0.0 and ep["kin_mode"] == "yaw"
    D = synth["D"]
    d = D.EGO_PERTURB_DEFAULTS
    assert (ep["ego_perturb_vscale"], ep["ego_perturb_vmax"], ep["ego_perturb_adrop"], ep["ego_perturb_yawrate_std"], ep["ego_perturb_seed"]) == (list(d["vscale"]), d["vmax"], d["adrop"], d["yawrate_std_deg"], d["seed"])
    assert d["vmax"] == 8.0 and ep["ego_perturb_vmax"] == synth["gate"][("orc", seed)]["levers"]["ego_perturb"]["ego_perturb_vmax"], "vmax's default IS ORC's registered 8.0: the eligible denominator is identical"
    orc = synth["gate"][("orc", seed)]
    assert orc["ego_perturb_probe"]["ego_perturb"] != "off" and orc["levers"]["ego_perturb"]["ego_perturb_p"] == 0.3, "the control is ORC's gate with the perturbation OFF, not the same gate"


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("flag,value", [("--ego-perturb-vscale", ["0.6", "1.0"]), ("--ego-perturb-vmax", ["6.0"]), ("--ego-perturb-adrop", ["1.0"]), ("--ego-perturb-yawrate-std", ["2.0"]), ("--ego-perturb-seed", ["1"])])
def test_a_law_flag_named_on_orc0_with_a_non_default_value_is_refused_by_the_trainers_own_gate(synth, seed, flag, value):
    """The guard Amendment 3 keeps unchanged: a law flag set (to a value other than the default) while --ego-perturb-p is 0 is 'a lever that is declared and does nothing' and the trainer REFUSES it. (yawrate-std 2.0 and
    seed 1 are the two registered values that tripped it before Amendment 3.)"""
    TR = synth["TR"]
    out = synth["root"] / f"orc0_law_{flag.replace('--ego-perturb-', '')}_s{seed}"
    with pytest.raises(SystemExit) as e:
        TR.main(_argv(out, synth["ck"][seed], "orc0", seed, gate=True) + [flag, *value])
    assert "are set while ego_perturb_p is 0" in str(e.value) and flag[2:].replace("-", "_") in str(e.value), str(e.value)[:300]


def test_a_law_flag_named_on_orc0_at_its_default_value_passes_the_trainers_gate_which_is_why_the_chain_refuses_the_naming(synth):
    """HONEST SCOPE of the previous test: the trainer's guard compares VALUES with the defaults, so `--ego-perturb-vscale 0.5 1.0`, `--ego-perturb-vmax 8.0`, `--ego-perturb-adrop 1.5` and `--ego-perturb-seed 0` named on ORC-0
    pass it. The queue never writes them on ORC-0 and the chain's gate assertions (test_gate_assertions_for_orc0_*) refuse the NAMING itself, so ORC-0 stays 'p = 0 and nothing else of the family'."""
    TR = synth["TR"]
    for k, (flag, value) in enumerate((("--ego-perturb-vscale", ["0.5", "1.0"]), ("--ego-perturb-vmax", ["8.0"]), ("--ego-perturb-adrop", ["1.5"]), ("--ego-perturb-seed", ["0"]))):
        out = synth["root"] / f"orc0_default_law_{k}"
        assert TR.main(_argv(out, synth["ck"][0], "orc0", 0, gate=True) + [flag, *value]) == 0, flag
''' + "\n"
# the xfail block's test body ended the file: nothing else follows it
with open(p, "wb") as fh:
    fh.write(s.encode("utf-8"))
print("edits", n_edits, "len", len(s))
