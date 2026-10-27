import pathlib, sys
P = pathlib.Path("C:/Users/Admin/dagread/test_dag_read.py")
s = P.read_text(encoding="utf-8")
def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)

# 1. KB45 constant -> derived from the real PROVENANCE.json controls block
sub('''KB45 = {"K-B4": {"PASS": True, "cases": {k: True for k in ("straight", "stationary_tie", "lateral_cut", "heading_cut", "end_edge", "loop")}},
        "K-B5": {"PASS": True, "max_pose_now_err_m": 0.0, "max_rel_transform_err_m": 2e-7}}              # the `controls` block the generator does NOT write yet (needs the bank generator to add)


def real_doc(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))
''', '''

def real_doc(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def kb45_block(n_shards, n_records):
    """The `controls` block (K-B4 / K-B5) of the bank generator's REAL PROVENANCE.json (fixtures/dagger_smoke, the K-B4/K-B5 build), re-shaped to a bank of `n_shards` shards and `n_records` records.
    Every key and every number except those four fields is the generator's own."""
    c = json.loads(json.dumps(real_doc("PROVENANCE.json")["controls"]))
    c["K-B4"]["n_shards_checked"] = n_shards
    c["K-B5"].update(n_records=n_records, n_shards=n_shards, log_pose_sha256_by_shard=[f"{k + 1:02x}" * 32 for k in range(n_shards)])
    return c
''')
sub('''shard_provenance=[f"shard_{k:03d}of002.provenance.json" for k in range(2)], controls=json.loads(json.dumps(KB45)))''',
    '''shard_provenance=[f"shard_{k:03d}of002.provenance.json" for k in range(2)], controls=kb45_block(2, 2 * N_WIN))''')
# the shard documents carry their own `controls` (K-B4 / K-B5 over that shard): keep the key structure, set the shard's own record count
sub('''                  kb3_eval_disjoint={"n_eval_clips": 141, "overlap": 0})
        shard_provs.append(sp) ''', '''                  kb3_eval_disjoint={"n_eval_clips": 141, "overlap": 0})
        sp["controls"]["K-B5"]["n_records"] = 2 * half                                                 # the shard's own K-B5 sample (the reader reads the merged block in PROVENANCE.json)
        shard_provs.append(sp) ''')

# 2. trainer summary / info: synthesise exactly what the real trainer writes (v8f_mini_train.py :1398 info, :1542 selfcheck.json "info", :1663 summary.json top level)
sub('''            jdump(td / "selfcheck.json", {"declared_vs_built": "PASS", "argv_sha": sha, "levers": lev})''',
    '''            info = {"n_clips": 4512, "n_windows": N_WIN, "train_sha256": TRAIN_SHA,
                    **({"train_window_list_sha256": WIN_SHA} if arm != "ORC0" else {})}      # build_data's `info` (trainer :1398), which selfcheck.json carries under "info" (:1542); the reader reads summary.json's top-level copy (:1663)
            jdump(td / "selfcheck.json", {"declared_vs_built": "PASS", "argv_sha": sha, "levers": lev, "info": info})''')

# 3. the generator's key structure now includes `controls`
sub('''    assert shape({k: v for k, v in B["prov"].items() if k != "controls"}) == shape(real_prov)
''', '''    assert shape(B["prov"]) == shape(real_prov)                      # the K-B4/K-B5 build: the merged `controls` block is part of the generator's key structure now
''')
sub('''    assert real_man["common"]["thresholds_file_sha256"].startswith(dag.MATCH_CONST_SHA_PREFIX)
    rp = write_bank(B, tmp_path / "b")''', '''    assert real_man["common"]["thresholds_file_sha256"].startswith(dag.MATCH_CONST_SHA_PREFIX)
    kc = real_prov["controls"]                                       # the reader's registered K-B4 cases and K-B5 tolerance ARE the generator's
    assert set(kc["K-B4"]["cases"]) == set(dag.KB4_CASES) and all(v is True for v in kc["K-B4"]["cases"].values()) and kc["K-B4"]["PASS"] is True
    assert kc["K-B5"]["tol_m"] == dag.KB5_TOL_M and kc["K-B5"]["PASS"] is True and kc["K-B5"]["n_records"] == real_prov["n_records"]
    assert set(real_sp["controls"]) == {"K-B4", "K-B5"} and real_sp["controls"]["K-B4"]["cases"] == kc["K-B4"]["cases"]
    rp = write_bank(B, tmp_path / "b")''')

# 4. the smoke test reads the byte copy in fixtures and K-B4 / K-B5 now PASS on the real documents
sub('''SMOKE = TREE.parent.parent / "dg" / "tree" / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "raw" / "2026-10-26-dagger-build" / "smoke"
''', '''SMOKE = FIXTURES                                                   # the generator's K-B4/K-B5 smoke (di/.../2026-10-26-dagger-build/smoke_kb45): PROVENANCE.json, bank.json, the shard provenance and the shard npz, byte copies
''')
sub('''@pytest.mark.skipif(not (SMOKE / "PROVENANCE.json").exists() or not SMOKE_WORLD.exists(), reason="the bank generator's smoke output (dg/) is not on this machine")
def test_the_real_generator_smoke_reads_exactly_as_NOTES_md_reports():
    """The 12-clip SYNTHETIC smoke the bank generator wrote (read-only here): what the reader's bank controls make of the generator's REAL provenance."""''',
'''@pytest.mark.skipif(not SMOKE_WORLD.exists(), reason="the smoke's synthetic TRAIN list (dg/smoke_world/join_train.npz) is not on this machine")
def test_the_real_generator_smoke_reads_exactly_as_NOTES_md_reports():
    """The 12-clip SYNTHETIC smoke the bank generator wrote (byte copies under fixtures/dagger_smoke): what the reader's bank controls make of the generator's REAL provenance.
    K-B4 and K-B5 PASS on it now (the generator records them); BANK-CONFIG is the only failure and it is the partial-bank refusal."""''')
sub('''    assert ok == {"BANK-FILES:s0", "K-B0:s0", "K-B1:s0", "K-B2:s0", "K-B3.1:s0", "K-B3.3:s0", "BANK-ACCOUNTING:s0"}
    assert set(ctl.failed()) == {"BANK-CONFIG:s0", "K-B4:s0", "K-B5:s0"}
    assert "PARTIAL" in ctl.items["BANK-CONFIG:s0"]["detail"][0] and "4512" in str(ctl.items["BANK-CONFIG:s0"]["detail"]) and len(ctl.items["BANK-CONFIG:s0"]["detail"]) == 2
    assert all("needs the bank generator to add" in ctl.items[k]["detail"] for k in ("K-B4:s0", "K-B5:s0"))
''', '''    assert ok == {"BANK-FILES:s0", "K-B0:s0", "K-B1:s0", "K-B2:s0", "K-B3.1:s0", "K-B3.3:s0", "K-B4:s0", "K-B5:s0", "BANK-ACCOUNTING:s0"}
    assert set(ctl.failed()) == {"BANK-CONFIG:s0"}
    assert "PARTIAL" in ctl.items["BANK-CONFIG:s0"]["detail"][0] and "4512" in str(ctl.items["BANK-CONFIG:s0"]["detail"]) and len(ctl.items["BANK-CONFIG:s0"]["detail"]) == 2
    d4, d5 = ctl.items["K-B4:s0"]["detail"], ctl.items["K-B5:s0"]["detail"]                        # relayed, not recomputed: the reader says so
    assert d4["cases"] == list(dag.KB4_CASES) and d4["verified_by_reader"] is False
    assert d5 == {"max_pose_now_err_m": 0.0, "max_rel_transform_err_m": 1.504511633759493e-05, "verified_by_reader": False}
''')
P.write_text(s, encoding="utf-8", newline="\n")
print("ok")
