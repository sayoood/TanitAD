import pathlib
D = pathlib.Path('C:/Users/Admin/dagread')


def edit(fn, pairs):
    p = D / fn
    s = p.read_text(encoding='utf-8')
    for old, new in pairs:
        assert s.count(old) == 1, (fn, old[:80], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding='utf-8', newline='')


# ---- reader: argv --dagger-bank-sha256 == the recorded levers.dagger_bank_sha256 == sha256(bank.json)
edit('dag_read.py', [(
    '''    """The DAG arm trained on THIS bank: the argv's --dagger-bank-sha256 and the trainer's built record equal the bank's sha256 (bank.json, hashed here: BANK-FILES ties it to PROVENANCE);
    K-B2 on the arm: the bank's generating checkpoint is the arm's --init-ckpt."""
    bsha, bad = B["man_sha256"], []
    ok, declared = one_value(r["flags"], "--dagger-bank-sha256")
    if not ok or declared != bsha:
        bad.append(f"argv --dagger-bank-sha256 {str(declared)[:16]} != the bank's sha256 {bsha[:16]} (bank.json hashed here)")
    built = find_key(r["sources"], key_bank)
    if not built:
        bad.append(f"no built bank sha256 '{key_bank}' recorded (the lever's built_levers entry)")
    for w, v in built:
        if str(v) != bsha:
            bad.append(f"{w} = {str(v)[:16]} != the bank's sha256 {bsha[:16]}")''',
    '''    """The DAG arm trained on THIS bank (PREREG section 4: DAG names `--dagger-bank-sha256 <hex>`): (A) the argv value equals the trainer's RECORDED `levers.dagger_bank_sha256`, and (B) that record is
    the sha256 of the bank's bank.json (the trainer's definition of the bank's identity; hashed here, BANK-FILES ties it to PROVENANCE). Each comparison is its own message.
    K-B2 on the arm: the bank's generating checkpoint is the arm's --init-ckpt."""
    bsha, bad = B["man_sha256"], []
    ok, declared = one_value(r["flags"], "--dagger-bank-sha256")
    if not ok:
        bad.append(f"--dagger-bank-sha256 is not named exactly once with one value (argv: {r['flags'].get('--dagger-bank-sha256')})")
    built = find_key(r["sources"], key_bank)
    if not built:
        bad.append(f"no recorded bank sha256 '{key_bank}' (the lever's built_levers entry)")
    for w, v in built:
        if ok and str(v) != declared:
            bad.append(f"argv --dagger-bank-sha256 {str(declared)[:16]} != the recorded {w} = {str(v)[:16]}")
        if str(v) != bsha:
            bad.append(f"the recorded {w} = {str(v)[:16]} != the sha256 of the bank's bank.json {bsha[:16]} (the trainer's definition of the bank's identity)")''')])

# ---- tests: update needles, extend the unit test, add the section-4 test
edit('test_dag_read.py', [
    ('''("BANK-DECLARED:DAG_s0", "!= the bank's sha256"), ("BANK-DECLARED:DAG_s1",)),''',
     '''("BANK-DECLARED:DAG_s0", "argv --dagger-bank-sha256", "!= the recorded"), ("BANK-DECLARED:DAG_s1",)),'''),
    ('''     ("BANK-DECLARED:DAG_s1", "the bank's sha256"), ("BANK-DECLARED:DAG_s0",)),''',
     '''     ("BANK-DECLARED:DAG_s1", "the sha256 of the bank's bank.json"), ("BANK-DECLARED:DAG_s0",)),'''),
    ('''    assert not f(fake_rec(["--init-ckpt", "/n", "--dagger-bank-sha256", "11" * 32], sums))[0]                         # argv declares another bank
    assert not f(fake_rec(argv, {**sums, "levers": {**sums["levers"], "dagger_bank_sha256": "11" * 32}}))[0]          # the trainer BUILT another bank
    assert not f(fake_rec(argv, {"levers": {"init_ckpt": {"sha256": msha}}}))[0]                                      # no built record''',
     '''    ok, det = f(fake_rec(["--init-ckpt", "/n", "--dagger-bank-sha256", "11" * 32], sums))                              # (A) argv declares another bank than the trainer recorded
    assert not ok and any("argv --dagger-bank-sha256" in d and "!= the recorded" in d for d in det) and not any("bank.json" in d for d in det)
    ok, det = f(fake_rec(argv, {**sums, "levers": {**sums["levers"], "dagger_bank_sha256": "11" * 32}}))             # the trainer BUILT another bank: A and B both fire
    assert not ok and any("!= the recorded" in d for d in det) and any("bank.json" in d for d in det)
    ok, det = f(fake_rec(["--init-ckpt", "/n", "--dagger-bank-sha256", "11" * 32], {**sums, "levers": {**sums["levers"], "dagger_bank_sha256": "11" * 32}}))   # (B) argv == recorded, both name a bank that is not on disk
    assert not ok and len(det) == 1 and "the sha256 of the bank's bank.json" in det[0]
    assert not f(fake_rec(argv + ["--dagger-bank-sha256", bsha], sums))[0]                                            # --dagger-bank-sha256 named twice
    assert not f(fake_rec(["--init-ckpt", "/n"], sums))[0]                                                            # not named
    ok, det = f(fake_rec(argv, {**sums, "dagger_bank_sha256": "11" * 32}))                                            # the same key recorded at two places, one of them another bank
    assert not ok and any("bank.json" in d for d in det)
    assert not f(fake_rec(argv, {"levers": {"init_ckpt": {"sha256": msha}}}))[0]                                      # no built record'''),
    ('''def test_g_argv_the_ego_perturbation_levers_are_never_combined():''',
     '''def test_section_4_names_FOUR_dagger_flags_and_the_reader_follows_it_exactly():
    t = PREREG.read_text(encoding="utf-8").replace("**", "")
    assert "`--dagger-p`, `--dagger-bank`, `--dagger-bank-sha256`,\\n  `--dagger-seed`" in t and "DAG names all four." in t and "DAG-0 names only `--dagger-p 0`." in t
    assert "identical except `--out`, `--dagger-p` and the three DAG-only flags" in t
    four = ("--dagger-p", "--dagger-bank", "--dagger-bank-sha256", "--dagger-seed")
    assert dag.DAG_ONLY_FLAGS == four[1:] and len(dag.DAG_ONLY_FLAGS) == 3 and set(dag.DAG_ONLY_FLAGS) | {"--dagger-p"} == set(four)
    assert dag.g_argv("t", "DAG", 0, fake_rec(GOOD))[0] and all(f in GOOD for f in four)                              # DAG names all four
    assert dag.g_argv("t", "DAG0", 0, fake_rec(GOOD0))[0] and "--dagger-p" in GOOD0 and not any(f in GOOD0 for f in dag.DAG_ONLY_FLAGS)   # DAG-0 names only --dagger-p 0
    for f in four:                                                                                                    # DAG: each of the four is required
        a = list(GOOD)
        del a[a.index(f):a.index(f) + 2]
        ok, det = dag.g_argv("t", "DAG", 0, fake_rec(a))
        assert not ok and any(f in d for d in det), f
    a = list(GOOD0)
    del a[a.index("--dagger-p"):a.index("--dagger-p") + 2]
    assert not dag.g_argv("t", "DAG0", 0, fake_rec(a))[0]                                                             # DAG-0 must NAME --dagger-p (a default is not a test of the lever)
    a = list(GOOD0)
    a[a.index("--dagger-p") + 1] = "0.1"
    assert not dag.g_argv("t", "DAG0", 0, fake_rec(a))[0]                                                             # ... and name it 0
    for f in dag.DAG_ONLY_FLAGS:                                                                                      # DAG-0: none of the three
        assert not dag.g_argv("t", "DAG0", 0, fake_rec(GOOD0 + [f, "0" if f == "--dagger-seed" else "x"]))[0], f
    # the pair rule: identical except --out, --dagger-p and the three DAG-only flags (any one of them may differ or be absent; nothing else may)
    rd, r0 = fake_rec(GOOD, SUM), fake_rec(GOOD0, SUM)
    assert dag.g_pair(0, rd, r0, "init_ckpt.sha256")[0] is True
    for f in dag.DAG_ONLY_FLAGS:
        a = list(GOOD0) + [f, "0" if f == "--dagger-seed" else "x"]
        assert dag.g_pair(0, rd, fake_rec(a, SUM), "init_ckpt.sha256")[0] is True, f                                  # the pair rule sets the DAG-only flags aside (ARGV is what refuses them on DAG-0)
    a = list(GOOD0)
    a[a.index("--out") + 1] = "/elsewhere"
    assert dag.g_pair(0, rd, fake_rec(a, SUM), "init_ckpt.sha256")[0] is True                                         # --out may differ
    a = list(GOOD0)
    a[a.index("--dagger-p") + 1] = "0.05"
    assert dag.g_pair(0, rd, fake_rec(a, SUM), "init_ckpt.sha256")[0] is True                                         # --dagger-p may differ
    for f in ("--steps", "--lr", "--warmup", "--p-gt-start", "--p-gt-end", "--ego-perturb-p", "--init-ckpt", "--seed", "--arm"):   # nothing else may
        a = list(GOOD0)
        a[a.index(f) + 1] = "7" if f not in ("--init-ckpt", "--arm") else "/x/other"
        ok, det = dag.g_pair(0, rd, fake_rec(a, SUM), "init_ckpt.sha256")
        assert not ok and "not one lever" in str(det), f


def test_g_argv_the_ego_perturbation_levers_are_never_combined():'''),
])

# ---- mutants
p = D / 'mutate_dag.py'
s = p.read_text(encoding='utf-8')
a = s.index('    ("N54"')
b = s.index('    ("N56"')
new = '''    ("N54", "BANK-DECLARED ignores that --dagger-bank-sha256 is named exactly once with one value", sub('    if not ok:\\n        bad.append(f"--dagger-bank-sha256 is not named exactly once', '    if False:\\n        bad.append(f"--dagger-bank-sha256 is not named exactly once'), "bank_declared"),
    ("N55", "BANK-DECLARED: the recorded bank sha256 is not held to the sha256 of bank.json (comparison B)", sub("        if str(v) != bsha:", "        if False:"), "bank_declared"),
    ("N139", "BANK-DECLARED: the argv's --dagger-bank-sha256 is not held to the recorded levers.dagger_bank_sha256 (comparison A)", sub("        if ok and str(v) != declared:", "        if False:"), "bank_declared"),
    ("N140", "BANK-DECLARED: comparison A reads a doubly-named argv flag as absent", sub("        if ok and str(v) != declared:", "        if str(v) != declared:"), "bank_declared"),
'''
s = s[:a] + new + s[b:]
anchor = '    ("N64", "the PREREG id names another commit"'
extra = '''    ("N141", "the pair rule does not set --dagger-p aside (DAG at p 0.1 vs DAG-0 at 0 is then 'another lever')", sub('drop = {"--out", "--dagger-p"} | set(DAG_ONLY_FLAGS)', 'drop = {"--out"} | set(DAG_ONLY_FLAGS)'), "section_4 or world_ROBUST or g_pair"),
    ("N142", "the DAG-only flag set is two flags (--dagger-seed is not one of them)", sub('DAG_ONLY_FLAGS = ("--dagger-bank", "--dagger-bank-sha256", "--dagger-seed")', 'DAG_ONLY_FLAGS = ("--dagger-bank", "--dagger-bank-sha256")'), "section_4 or dagger_flags_belong"),
    ("N143", "the DAG-only flag set also holds --dagger-p (DAG-0 may not name it)", sub('DAG_ONLY_FLAGS = ("--dagger-bank", "--dagger-bank-sha256", "--dagger-seed")', 'DAG_ONLY_FLAGS = ("--dagger-p", "--dagger-bank", "--dagger-bank-sha256", "--dagger-seed")'), "section_4 or world_ROBUST"),
    ("N144", "DAG-0 is not required to NAME --dagger-p (the registered value is not checked on DAG-0)", sub('"--ego-perturb-p": EGO_P, "--dagger-p": ARM_P[arm]}', '"--ego-perturb-p": EGO_P, **({"--dagger-p": ARM_P[arm]} if arm == "DAG" else {})}'), "section_4"),
'''
assert s.count(anchor) == 1
s = s.replace(anchor, extra + anchor)
p.write_text(s, encoding='utf-8', newline='')
print("applied sec4")
