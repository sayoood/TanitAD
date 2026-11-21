import pathlib
P = pathlib.Path("C:/Users/Admin/dagread/test_dag_read.py")
s = P.read_text(encoding="utf-8")
def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)

# comment of the audit test: the 'MISSING' branch is now the OLDER generator's output
sub('''    B2["prov"].pop("controls")                                                # what the generator writes TODAY''',
    '''    B2["prov"].pop("controls")                                                # what the generator wrote BEFORE the K-B4/K-B5 build (fixtures/dagger_smoke/*.pre_kb45)''')
sub('''lambda B: B["prov"]["controls"].pop("K-B4")):    # what the generator writes TODAY''',
    '''lambda B: B["prov"]["controls"].pop("K-B4")):    # what the generator wrote BEFORE the K-B4/K-B5 build''')
sub('''    ("trainer_window_list_sha_not_recorded", lambda fx: jedit(fx.root / "runs" / "DAG_s0" / "summary.json", lambda o: o.pop("train_window_list_sha256")),''',
    '''    ("trainer_window_list_sha_not_recorded", lambda fx: jedit(fx.root / "runs" / "DAG_s0" / "summary.json", lambda o: o.pop("train_window_list_sha256")),   # selfcheck.json["info"] still holds a copy: the reader reads summary.json (the run's LAST write), not the early copy''')

NEW = '''

FIXTURE_SHAS = {"PROVENANCE.json": "6c780d62261a53254867ebca2e2c100dfb31ebd73ef2f2ba820ce7195020ad45", "bank.json": "c23c35b5607360400b1d779584ec613ae70d48ea10ac91ca61fc9ac3ab8b5d13",
                "shard_000of001.provenance.json": "d9915ee1d69f4d282d2d6b4377a564096d6183b58340463277f21b76e13163c4", "shard_000of001.npz": "8250fe281e88ff39e21b4a102706e4ea15b1612cc5871822ca078703571785e9",
                "PROVENANCE.json.pre_kb45": "2fe315f0011ec7d89ecd62345f222f32d70faa7b0fa053ca1595a80b1eec91b6", "shard_000of001.provenance.json.pre_kb45": "45264478c9d89903ba8053ee5cbd4f2e7340d2cbcfeacf839e50c5ff8e553b5b"}


def test_fixtures_are_the_kb45_generator_output_and_the_pre_kb45_documents_are_kept():
    """The refresh changed what it meant to change: PROVENANCE.json and the shard provenance gained `controls` (K-B4 / K-B5); bank.json and the shard npz did not move by a byte."""
    for nm, want in FIXTURE_SHAS.items():
        assert hashlib.sha256((FIXTURES / nm).read_bytes()).hexdigest() == want, nm
    new, old = real_doc("PROVENANCE.json"), real_doc("PROVENANCE.json.pre_kb45")
    assert "controls" in new and "controls" not in old
    assert {k: v for k, v in new.items() if k not in ("controls", "created_unix")} == {k: v for k, v in old.items() if k != "created_unix"}
    assert new["bank_sha256"] == old["bank_sha256"] == FIXTURE_SHAS["bank.json"]
    sn, so = real_doc("shard_000of001.provenance.json"), real_doc("shard_000of001.provenance.json.pre_kb45")
    assert "controls" in sn and "controls" not in so
    same = ("schema", "kind", "plan", "n_records", "reasons_by_tick", "coverage", "kb0_tick0_identity", "kb1_replay", "kb3_eval_disjoint", "causal_bucket_stats", "file_sha256", "content_sha256", "warnings")
    assert {k: sn[k] for k in same} == {k: so[k] for k in same}                   # every field the reader reads is the old smoke's; only the controls (and the code / timing / path fields) moved
    assert sn["file_sha256"] == FIXTURE_SHAS["shard_000of001.npz"]
    assert new["controls"]["K-B4"]["cases"] == sn["controls"]["K-B4"]["cases"]


TRAINER_SITES = HERE / "fixtures" / "trainer_kb45" / "WRITE_SITES.txt"


def test_the_trainer_records_train_window_list_sha256_where_the_readers_default_key_looks(tmp_path):
    """The reader's `--key-window-sha` default is the key the REAL trainer writes, at the place the reader looks (summary.json, top level, beside train_sha256), and the synthetic runs carry it THERE.
    WRITE_SITES.txt is a mechanical extract (scratch/extract_trainer_sites.py) of v8f_mini_train.py :1398 (info), :1542 (selfcheck.json carries info) and :1662-1663 (summary.json)."""
    rows = [ln.split("\t", 1) for ln in TRAINER_SITES.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    sites = {int(n): t for n, t in rows}
    assert sorted(sites) == [1398, 1542, 1662, 1663]
    ap = dag.build_parser()
    kw, kt = ap.get_default("key_window_sha"), ap.get_default("key_train_sha")
    assert (kw, kt) == ("train_window_list_sha256", "train_sha256")
    assert sites[1398].strip().startswith(f'info["{kw}"] = DG.windows_sha256(J["clip"], wins)')            # the value is windows_sha256 over the TRAIN window list, in build_data's info
    assert f'"{kt}": info.get("{kt}")' in sites[1662] and sites[1663].strip() == f'"{kw}": info.get("{kw}"),'   # summary.json, top level, the two keys side by side
    assert '"info": info' in sites[1542]                                                                  # selfcheck.json carries the same info record
    rc, res, out, fx = run(tmp_path, **STRONG)
    assert rc == 0
    for arm in ("DAG", "DAG0"):
        for s in (0, 1):
            sm = json.loads((fx.root / "runs" / f"{arm}_s{s}" / "summary.json").read_text(encoding="utf-8"))
            sc = json.loads((fx.root / "runs" / f"{arm}_s{s}" / "selfcheck.json").read_text(encoding="utf-8"))
            assert sm[kw] == sc["info"][kw] == WIN_SHA and sm[kt] == sc["info"][kt] == TRAIN_SHA
    assert kw not in json.loads((fx.root / "runs" / "ORC0_s0" / "summary.json").read_text(encoding="utf-8"))      # ORC-0 is the older build: no window-list sha
    assert res["controls"]["K-B3.2:s0"]["PASS"] and res["controls"]["K-B3.2:s1"]["PASS"] and res["controls"]["TRAIN-LIST:s0"]["PASS"]


@pytest.mark.skipif(not SMOKE_WORLD.exists(), reason="the smoke's synthetic TRAIN list (dg/smoke_world/join_train.npz) is not on this machine")
def test_bank_audit_on_the_real_kb45_smoke_passes_k_b4_and_k_b5_and_fails_only_the_partial_bank(tmp_path, capsys):
    """bank_audit.py on the generator's real documents: K-B4 / K-B5 PASS, BANK-CONFIG is the one FAIL (a 12-clip smoke is never a bank), the trainer-side controls are NOT RUN."""
    A = load_audit()
    rc = A.main(["--report", str(SMOKE / "PROVENANCE.json"), "--m2b-sha", M2B_S0, "--eval-grid", str(RAW / "eval_grid_windows.json"), "--train-clips", str(SMOKE_WORLD), "--out", str(tmp_path / "a.json")])
    out = capsys.readouterr().out
    assert rc == 1
    tag = {ln.split()[1]: ln.split()[0] for ln in out.splitlines() if ln.split() and ln.split()[0] in ("PASS", "FAIL", "MISSING")}
    assert tag == {"BANK-FILES:s0": "PASS", "BANK-CONFIG:s0": "FAIL", "K-B0:s0": "PASS", "K-B1:s0": "PASS", "K-B2:s0": "PASS", "K-B3.1:s0": "PASS", "K-B3.3:s0": "PASS",
                   "K-B4:s0": "PASS", "K-B5:s0": "PASS", "BANK-ACCOUNTING:s0": "PASS"}, out
    assert "MISSING" not in out and "NOT RUN  K-B3.2:s0" in out and "recorded on every run" in out
'''
anchor = '''    assert B["man"]["common"]["thresholds_file_sha256"].startswith("438366d5") and B["man"]["common"]["constants_sha256"].startswith("8f94b3b5")   # the PREREG's sha is the FILE's, not the constants dict's'''
sub(anchor, anchor + NEW)
P.write_text(s, encoding="utf-8", newline="\n")
print("ok")
