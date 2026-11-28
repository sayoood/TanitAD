SMOKE = TREE.parent.parent / "dg" / "tree" / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini" / "raw" / "2026-10-26-dagger-build" / "smoke"
SMOKE_WORLD = TREE.parent.parent / "dg" / "smoke_world" / "join_train.npz"
M2B_S0 = "601cb435267317862539c2ba8bcce78b04eb0c31fe08a183ecb1bafc71dcc6f8"


def load_audit():
    spec = importlib.util.spec_from_file_location("bank_audit_under_test", str(HERE / "bank_audit.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_bank_audit_script_runs_the_readers_bank_controls_and_marks_what_the_generator_does_not_write(tmp_path, capsys):
    A = load_audit()
    B = good_bank()
    rp = write_bank(B, tmp_path / "b")
    jdump(tmp_path / "eval.json", EVAL_IDS)
    jdump(tmp_path / "train.json", CLIPS)
    args = ["--report", rp, "--m2b-sha", "ab" * 32, "--eval-grid", str(tmp_path / "eval.json"), "--train-clips", str(tmp_path / "train.json"), "--out", str(tmp_path / "audit.json")]
    assert A.main(args) == 0
    out = capsys.readouterr().out
    assert "PASS     K-B3.1:s0" in out and "PASS     K-B3.3:s0" in out and "NOT RUN  K-B3.2:s0" in out and "COVERAGE launches" in out and "FAIL" not in out and "MISSING" not in out
    res = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert res["controls"]["BANK-FILES:s0"]["PASS"] and res["bank_sha256"] == B["man_sha256"]
    B2 = good_bank()
    B2["prov"].pop("controls")                                                # what the generator writes TODAY
    rp2 = write_bank(B2, tmp_path / "b2")
    assert A.main([*args[:1], rp2, *args[2:]]) == 1
    out = capsys.readouterr().out
    assert "MISSING  K-B4:s0" in out and "MISSING  K-B5:s0" in out and "needs the bank generator to add" in out and "PASS     K-B0:s0" in out


@pytest.mark.skipif(not (SMOKE / "PROVENANCE.json").exists() or not SMOKE_WORLD.exists(), reason="the bank generator's smoke output (dg/) is not on this machine")
def test_the_real_generator_smoke_reads_exactly_as_NOTES_md_reports():
    """The 12-clip SYNTHETIC smoke the bank generator wrote (read-only here): what the reader's bank controls make of the generator's REAL provenance."""
    B = dag.load_bank(str(SMOKE / "PROVENANCE.json"))
    ev = json.loads((RAW / "eval_grid_windows.json").read_text(encoding="utf-8"))["order"]
    tr = [str(x) for x in np.load(SMOKE_WORLD, allow_pickle=False)["clip"].tolist()]
    ctl = dag.Ctl()
    dag.bank_stage(ctl, 0, B, M2B_S0, ev, tr)
    ok = {k for k, v in ctl.items.items() if v["PASS"]}
    assert ok == {"BANK-FILES:s0", "K-B0:s0", "K-B1:s0", "K-B2:s0", "K-B3.1:s0", "K-B3.3:s0", "BANK-ACCOUNTING:s0"}
    assert set(ctl.failed()) == {"BANK-CONFIG:s0", "K-B4:s0", "K-B5:s0"}
    assert "PARTIAL" in ctl.items["BANK-CONFIG:s0"]["detail"][0] and "4512" in str(ctl.items["BANK-CONFIG:s0"]["detail"]) and len(ctl.items["BANK-CONFIG:s0"]["detail"]) == 2
    assert all("needs the bank generator to add" in ctl.items[k]["detail"] for k in ("K-B4:s0", "K-B5:s0"))
    cv = dag.coverage_of(B)
    assert cv["n_windows"] == 804 and cv["n_eligible_windows"] == 580 and cv["n_states_before_cuts"] == 1608 and cv["n_states_kept"] == 1120 and cv["cuts"]["nonfinite"] == 0
    assert dag.bank_clip_names(B) == sorted(tr) and B["man_sha256"] == "c23c35b5607360400b1d779584ec613ae70d48ea10ac91ca61fc9ac3ab8b5d13"
    assert B["man"]["common"]["thresholds_file_sha256"].startswith("438366d5") and B["man"]["common"]["constants_sha256"].startswith("8f94b3b5")   # the PREREG's sha is the FILE's, not the constants dict's
    assert not dag.g_bank_config(B, ev)[0] and dag.g_bank_config({**B, "man": {**B["man"], "common": {**B["man"]["common"], "partial": False, "limit_clips": None,
                                                                                                  "train": {**B["man"]["common"]["train"], "n_clips": 4512, "n_windows": 306566}}}}, ev)[0]


