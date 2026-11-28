p = 'C:/Users/Admin/dagread/test_dag_read.py'
s = open(p, encoding='utf-8', newline='').read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:100], s.count(old))
    s = s.replace(old, new)


def splice(start, end, new):
    global s
    a, b = s.index(start), s.index(end)
    b = s.rindex(chr(10), 0, b) + 1
    assert a < b
    s = s[:a] + new + s[b:]


# BLOCKED test
rep('''assert set(b["coverage"]["s1"]["cuts"]) == {"off_route", "heading", "end_edge"} and b["coverage_min"] == 0.5''',
    '''assert set(b["coverage"]["s1"]["cuts"]) == {"off_lat", "off_head", "end_edge", "nonfinite"} and b["coverage_min"] == 0.5''')
rep('''    assert res["bank"]["s1"]["header"]["generator_ckpt_sha256"] == fx.m2b_sha[1]''',
    '''    assert res["bank"]["s1"]["header"]["ckpt_sha256"] == fx.m2b_sha[1] and res["bank"]["s1"]["header"]["bank_sha256"] == fx.bank_sha[1] and res["bank"]["s1"]["header"]["n_bank_clips"] == 4512''')
# REFUSED beats BLOCKED
rep('''    def e(fx):
        jedit(fx.root / "bank" / "report_s0.json", lambda o: o["controls"]["K-B0"].update(PASS=False))
    rc, _, out, _ = run(tmp_path, edit=e, fxkw={"elig": 1000}, **STRONG)
    refused(capsys, rc, out, "K-B0:s0")''',
    '''    rc, _, out, _ = run(tmp_path, fxkw={"elig": 1000, "bank_mods": {0: lambda B: B["shard_provs"][0]["kb0_tick0_identity"].update(mismatch=1)}}, **STRONG)
    refused(capsys, rc, out, "K-B0:s0")''')
# every failed bank control, and missing inputs
splice('def test_bank_stage_names_every_failed_bank_control(tmp_path, capsys):', '# ======================================================================================================================== 9. a refusal per control',
       '''def test_bank_stage_names_every_failed_bank_control(tmp_path, capsys):
    def s0(B):
        B["shard_provs"][0]["kb0_tick0_identity"].update(mismatch=1)                                       # K-B0
        B["shards"][0]["clips"]["4"] = EVAL_IDS[9]                                                         # an eval clip inside the bank (and not a TRAIN clip): K-B3.3 and K-B3.1
        B["man"]["common"]["rollout"].update(limit="logged")                                               # BANK-CONFIG

    def s1(B):
        B["shard_provs"][0]["kb1_replay"].update(windows=1)                                                # K-B1
        B["prov"]["controls"]["K-B4"]["cases"].update(loop=False)                                          # K-B4
        B["prov"]["controls"]["K-B5"].update(max_rel_transform_err_m=1e-3)                                 # K-B5
        both(B, lambda d: d["reasons_by_tick"]["2"].update(end_edge=d["reasons_by_tick"]["2"]["end_edge"] - 1))   # BANK-ACCOUNTING
        B["man"]["common"]["ckpt_sha256"] = "11" * 32                                                      # K-B2
        B["shards"][1]["clips"]["1"] = "clip-not-in-train"                                                 # K-B3.1 only (not an eval clip)
        for sp in B["shard_provs"]:
            sp["inputs"]["ckpt"]["sha256"] = "11" * 32
    rc, _, out, _ = run(tmp_path, fxkw={"bank_mods": {0: s0, 1: s1}}, extra={"--banked-sha": "1=" + "11" * 32}, **STRONG)
    refused(capsys, rc, out, "K-B0:s0", "K-B3.3:s0", "K-B3.1:s0", "overlap", "BANK-CONFIG:s0", "common.rollout", "K-B1:s1", "K-B4:s1", "loop", "K-B5:s1", "BANK-ACCOUNTING:s1",
            "K-B2:s1", "K-B3.1:s1", "not in M2b's TRAIN clip list", "M2B-CKPT:s1",
            absent=("K-B1:s0", "K-B2:s0", "K-B4:s0", "K-B5:s0", "BANK-ACCOUNTING:s0", "BANK-CONFIG:s1", "K-B0:s1", "K-B3.3:s1", "M2B-CKPT:s0", "BANK-FILES", "K-B3.2", "TRAIN-LIST", "ARGV:"))


def test_bank_stage_refuses_missing_inputs(tmp_path, capsys):
    rc, _, out, _ = run(tmp_path / "a", edit=lambda fx: fx.manifest["bank"].pop("1"), **STRONG)
    refused(capsys, rc, out, "no bank for seed 1")
    rc, _, out, _ = run(tmp_path / "b", edit=lambda fx: (fx.root / "bank_s0" / "PROVENANCE.json").unlink(), **STRONG)
    refused(capsys, rc, out, "bank seed 0 report", "is missing")
    rc, _, out, _ = run(tmp_path / "c", edit=lambda fx: fx.manifest.pop("eval141_clip_ids"), **STRONG)
    refused(capsys, rc, out, "eval141_clip_ids")
    rc, _, out, _ = run(tmp_path / "d", edit=lambda fx: jedit(fx.root / "bank_s1" / "PROVENANCE.json", lambda o: o.update(kind="other")), **STRONG)
    refused(capsys, rc, out, "bank seed 1 report", "not the generator's")
    rc, _, out, _ = run(tmp_path / "e", edit=lambda fx: jdump(fx.root / "eval141_clip_ids.json", []), **STRONG)
    refused(capsys, rc, out, "K-B3.3:s0", "empty")
    rc, _, out, _ = run(tmp_path / "f", edit=lambda fx: fx.manifest.pop("train_clip_ids"), **STRONG)
    refused(capsys, rc, out, "train_clip_ids")
    rc, _, out, _ = run(tmp_path / "g", edit=lambda fx: jdump(fx.root / "train_clip_ids.json", []), **STRONG)
    refused(capsys, rc, out, "K-B3.1:s0", "TRAIN clip list is empty")
    rc, _, out, _ = run(tmp_path / "h", edit=lambda fx: (fx.root / "bank_s1" / "shard_001of002.npz").unlink(), **STRONG)
    refused(capsys, rc, out, "shard_001of002.npz", "is missing")


def test_world_ROBUST_with_a_bank_that_holds_fewer_clips_than_TRAIN(tmp_path, capsys):
    """Amendment 1 section 3: the bank may hold fewer clips than TRAIN (equality is NOT required) -- through the whole pipeline."""
    def shrink(B):
        for k in list(B["shards"][0]["clips"])[:100]:
            B["shards"][0]["clips"].pop(k)
    rc, res, *_ = run(tmp_path, fxkw={"bank_mods": {0: shrink, 1: shrink}}, **STRONG)
    assert rc == 0, capsys.readouterr().err
    assert res["OUTCOME"]["OUTCOME"] == "ROBUST" and all(res["controls_PASS"].values())
    assert res["controls"]["K-B3.1:s0"]["detail"]["n_bank_clips"] == 4412 and res["controls"]["K-B3.1:s0"]["detail"]["n_train_clips_without_bank_records"] == 100


def test_the_three_kb3_controls_are_separate_entries_in_the_output(tmp_path, capsys):
    rc, res, *_ = run(tmp_path, **STRONG)
    assert rc == 0, capsys.readouterr().err
    for s in ("s0", "s1"):
        for cid in ("K-B3.1", "K-B3.2", "K-B3.3"):
            assert res["controls"][f"{cid}:{s}"]["PASS"] is True, cid
        assert "K-B3:" + s not in res["controls"]
    assert res["controls"]["K-B3.3:s0"]["detail"]["deliberate_regression"] == "refused" and res["controls"]["K-B3.1:s0"]["detail"]["deliberate_regression"] == "refused"
    assert res["controls"]["K-B4:s0"]["detail"]["verified_by_reader"] is False and res["controls"]["K-B2:s0"]["detail"]["verified_by_reader"] != False   # noqa: E712


''')
# wiring: absent list
rep('''"K4:DAG_s0", "BANK-WINDOWS", "K-B0:", "K-B1:", "K-B3:", "K-B4:", "K-B5:", "BANK-CONFIG", "BANK-ACCOUNTING",''',
    '''"K4:DAG_s0", "K-B3.2", "TRAIN-LIST", "BANK-FILES", "K-B0:", "K-B1:", "K-B3.1:", "K-B3.3:", "K-B4:", "K-B5:", "BANK-CONFIG", "BANK-ACCOUNTING",''')
# parametrized list
rep('''("BANK-DECLARED:DAG_s0", "!= the bank report's bank_sha256"), ("BANK-DECLARED:DAG_s1",)),''', '''("BANK-DECLARED:DAG_s0", "!= the bank's sha256"), ("BANK-DECLARED:DAG_s1",)),''')
rep('''     ("BANK-DECLARED:DAG_s1", "bank report's bank_sha256"), ("BANK-DECLARED:DAG_s0",)),''', '''     ("BANK-DECLARED:DAG_s1", "the bank's sha256"), ("BANK-DECLARED:DAG_s0",)),''')
rep('''    ("bank_file_corrupt", lambda fx: (fx.root / "bank" / "bank_s0.bin").write_bytes(b"corrupted"), ("BANK-DECLARED:DAG_s0", "hashes to"), ("BANK-DECLARED:DAG_s1",)),''',
    '''    ("bank_manifest_altered", lambda fx: jedit(fx.root / "bank_s0" / "bank.json", lambda o: o.update(note="x")), ("BANK-FILES:s0", "bank.json hashes to", "BANK-DECLARED:DAG_s0"),
     ("BANK-FILES:s1", "BANK-DECLARED:DAG_s1")),
    ("bank_shard_altered", lambda fx: (fx.root / "bank_s1" / "shard_000of002.npz").write_bytes((fx.root / "bank_s1" / "shard_000of002.npz").read_bytes() + b"x"),
     ("BANK-FILES:s1", "shard_000of002.npz hashes to"), ("BANK-FILES:s0",)),''')
a = s.index('    ("trainer_window_sha", lambda fx:')
b = s.index('    ("orc0_names_dagger_p"')
s = s[:a] + '''    ("trainer_clip_list_sha", lambda fx: jedit(fx.root / "runs" / "DAG0_s1" / "summary.json", lambda o: o.update(train_sha256="33" * 32)),
     ("TRAIN-LIST:s1", "PAIR:s1", "ORC0-REPLICATE:s1"), ("TRAIN-LIST:s0", "K-B3.2:s1")),
    ("trainer_window_list_sha_differs_between_the_arms", lambda fx: jedit(fx.root / "runs" / "DAG0_s1" / "summary.json", lambda o: o.update(train_window_list_sha256="33" * 32)),
     ("K-B3.2:s1", "differs between DAG and DAG-0", "!= the trainer's (DAG-0)"), ("K-B3.2:s0", "TRAIN-LIST", "PAIR:s1")),
    ("trainer_window_list_sha_not_recorded", lambda fx: jedit(fx.root / "runs" / "DAG_s0" / "summary.json", lambda o: o.pop("train_window_list_sha256")),
     ("K-B3.2:s0", "DAG: the trainer's 'train_window_list_sha256' is not recorded", "needs the trainer to add it"), ("K-B3.2:s1", "DAG-0: the trainer's")),
''' + s[b:]
rep('''    rc, _, out, _ = run(tmp_path, edit=edit, **STRONG)
    refused(capsys, rc, out, *needles, absent=absent)''', '''    rc, _, out, _ = run(tmp_path, edit=edit, **STRONG)
    refused(capsys, rc, out, *needles, absent=absent)


def test_refuse_arm_stage_when_the_banks_window_list_is_not_the_trainers(tmp_path, capsys):
    def mod(B):
        B["man"]["common"]["window_list_sha256"] = "44" * 32
        for sp in B["shard_provs"]:
            sp["plan"]["window_list_sha256"] = "44" * 32
    rc, _, out, _ = run(tmp_path, fxkw={"bank_mods": {0: mod}}, **STRONG)
    refused(capsys, rc, out, "K-B3.2:s0", "common.window_list_sha256 4444444444444444 != the trainer's (DAG)", "(DAG-0)", absent=("K-B3.2:s1", "BANK-FILES", "TRAIN-LIST"))


def test_refuse_arm_stage_when_the_generator_planned_over_another_TRAIN_set(tmp_path, capsys):
    def mod(B):
        B["man"]["common"]["train"]["train_sha256"] = "55" * 32
    rc, _, out, _ = run(tmp_path, fxkw={"bank_mods": {1: mod}}, **STRONG)
    refused(capsys, rc, out, "TRAIN-LIST:s1", "planned over", absent=("TRAIN-LIST:s0", "K-B3.2", "BANK-CONFIG"))''')
# key override test
rep('''                jedit(fx.root / "runs" / f"{arm}_s{s}" / "summary.json", lambda o: o.update(clip_list_sha=o.pop("train_sha256")))
    rc, res, *_ = run(tmp_path, edit=e, extra={"--key-dagger-n": "rep_n", "--key-dagger-eligible": "rep_elig", "--key-dagger-bank-sha": "bank_sha", "--key-window-sha": "clip_list_sha"}, **STRONG)''',
    '''                jedit(fx.root / "runs" / f"{arm}_s{s}" / "summary.json", lambda o: o.update(clip_list_sha=o.pop("train_sha256"), **({"win_list_sha": o.pop("train_window_list_sha256")} if "train_window_list_sha256" in o else {})))
    rc, res, *_ = run(tmp_path, edit=e, extra={"--key-dagger-n": "rep_n", "--key-dagger-eligible": "rep_elig", "--key-dagger-bank-sha": "bank_sha", "--key-train-sha": "clip_list_sha",
                                                "--key-window-sha": "win_list_sha"}, **STRONG)''')
open(p, 'w', encoding='utf-8', newline='').write(s)
print("applied part 2")
