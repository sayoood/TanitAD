def good_bank(**kw):
    return make_bank("ab" * 32, **kw)


def both(B, fn):
    """Apply an edit to PROVENANCE.json and bank.json alike (they carry the same coverage / reasons / n_records)."""
    fn(B["prov"])
    fn(B["man"])
    return B


def raises(fn, *a, match=None):
    with pytest.raises(dag.Refusal, match=match):
        fn(*a)


def test_clip_overlap_by_id_and_by_sid_and_empty_eval():
    assert dag.clip_overlap(CLIPS, EVAL_IDS) == []
    assert dag.clip_overlap(CLIPS + [EVAL_IDS[7]], EVAL_IDS)[0].startswith("bank and eval-141 clip ids overlap")
    assert dag.clip_overlap(CLIPS, []) and "empty" in dag.clip_overlap(CLIPS, [])[0]
    assert dag.clip_overlap(CLIPS[:5], EVAL_IDS[:2] + [CLIPS[5]]) == []                  # not in this bank
    orig = dag.stable_sid
    dag.stable_sid = lambda c: 7                                                         # every sid collides: the sid half of the check must fire on its own
    try:
        r = dag.clip_overlap(["a"], ["b"])
    finally:
        dag.stable_sid = orig
    assert r == ["bank and eval-141 stable sids overlap"]


def shape(x, pre=""):
    """The set of key paths of a JSON document (lists contribute their first element's paths under '[]')."""
    out = set()
    if isinstance(x, dict):
        for k, v in x.items():
            out.add(pre + "." + k)
            out |= shape(v, pre + "." + k)
    elif isinstance(x, list) and x and isinstance(x[0], (dict, list)):
        out |= shape(x[0], pre + "[]")
    return out


def test_the_synthetic_bank_has_the_generators_real_key_structure_and_the_loader_reads_it(tmp_path):
    B = good_bank()
    real_prov, real_man, real_sp = real_doc("PROVENANCE.json"), real_doc("bank.json"), real_doc("shard_000of001.provenance.json")
    assert shape({k: v for k, v in B["prov"].items() if k != "controls"}) == shape(real_prov)
    assert shape(B["man"]) == shape(real_man) and shape(B["shard_provs"][0]) == shape(real_sp)
    assert real_prov["kind"] == dag.BANK_KIND and real_prov["schema"] == dag.BANK_SCHEMA == real_man["schema"] == real_man["common"]["schema"]
    assert real_man["common"]["rollout"] == {**dag.BANK_ROLLOUT} and real_man["common"]["constants"]["REASONS"] == list(dag.REASONS)       # the reader's registered spelling IS the generator's
    assert {k: real_man["common"]["constants"][k] for k in dag.MATCH_CONSTANTS} == dag.MATCH_CONSTANTS
    assert set(real_prov["reasons_by_tick"]["1"]) == set(dag.REASONS) and list(real_prov["reasons_by_tick"]) == [str(t) for t in dag.TICKS]
    assert real_man["common"]["thresholds_file_sha256"].startswith(dag.MATCH_CONST_SHA_PREFIX)
    rp = write_bank(B, tmp_path / "b")
    L = dag.load_bank(rp)
    assert L["prov"] == B["prov"] and L["man"] == B["man"] and L["shard_provs"] == B["shard_provs"] and L["man_sha256"] == B["man_sha256"] == B["prov"]["bank_sha256"]
    assert [(s["file"], s["sha256"], s["clips"]) for s in L["shards"]] == [(s["file"], s["sha256"], s["clips"]) for s in B["shards"]]
    assert dag.bank_clip_names(L) == sorted(CLIPS) and dag.g_bank_files(L)[0] is True
    assert dag.load_bank(rp, bank_file=str(tmp_path / "b" / "bank.json"))["man_sha256"] == B["man_sha256"]


def test_loader_refuses_unreadable_banks(tmp_path):
    B = good_bank()
    rp = write_bank(B, tmp_path / "b")
    raises(dag.load_bank, str(tmp_path / "nope.json"), match="is missing")
    jedit(rp, lambda o: o.update(kind="dagger-bank-report"))
    raises(dag.load_bank, rp, match="not the generator's")
    jedit(rp, lambda o: o.update(kind="bank", schema="v9"))
    raises(dag.load_bank, rp, match="not the generator's")
    jedit(rp, lambda o: o.update(schema="v8f-dagger-bank-v1", manifest="other.json"))
    raises(dag.load_bank, rp, match="bank manifest")
    jedit(rp, lambda o: o.update(manifest="bank.json"))
    (tmp_path / "b" / "shard_001of002.npz").unlink()
    raises(dag.load_bank, rp, match="shard_001of002.npz")
    (tmp_path / "b" / "shard_001of002.npz").write_bytes(b"not an npz")
    raises(dag.load_bank, rp, match="cannot read the clip names")
    (tmp_path / "b" / "shard_001of002.provenance.json").unlink()
    raises(dag.load_bank, rp, match="shard provenance")
    jedit(rp, lambda o: o.update(shard_provenance=[]))
    (tmp_path / "b" / "bank.json").write_text("[]", encoding="utf-8")
    raises(dag.load_bank, rp, match="no 'shards'")


def test_bank_files_hashes_and_counts():
    ok, det = dag.g_bank_files(good_bank())
    assert ok and det["n_shards"] == 2 and det["n_records"] == 2 * N_WIN and det["n_bank_clips"] == 4512
    edits = {
        "man_sha": lambda B: B.update(man_sha256="00" * 32),
        "prov_sha_not_hex": lambda B: B["prov"].update(bank_sha256="xyz"),
        "shard_file": lambda B: B["shards"][1].update(sha256="11" * 32),
        "n_shards": lambda B: B["man"].update(n_shards=3),
        "shard_provs_count": lambda B: (B["shard_provs"].pop(), B["prov"]["shard_provenance"].pop()),
        "records_man": lambda B: B["man"].update(n_records=B["man"]["n_records"] + 1),
        "records_prov": lambda B: B["prov"].update(n_records=B["prov"]["n_records"] + 1),
        "records_shard": lambda B: B["man"]["shards"][0].update(n_records=B["man"]["shards"][0]["n_records"] + 1),
        "partial_prov": lambda B: B["prov"].update(partial=True),
        "partial_common": lambda B: B["man"]["common"].update(partial=True),
        "dup_clip": lambda B: B["shards"][1]["clips"].update({"99999": B["shards"][0]["clips"]["0"]}),
        "dup_ci": lambda B: B["shards"][1]["clips"].update({"0": "clip-other"}),
        "shard_prov_k": lambda B: B["shard_provs"][1]["plan"].update(shard=[0, 2]),
        "shard_prov_sha": lambda B: B["shard_provs"][0].update(file_sha256="22" * 32),
        "shard_prov_records": lambda B: B["shard_provs"][0].update(n_records=5),
        "shard_prov_windows": lambda B: B["shard_provs"][0]["plan"].update(window_list_sha256="33" * 32),
    }
    for nm, fn in edits.items():
        B = good_bank()
        fn(B)
        assert not dag.g_bank_files(B)[0], nm


def test_bank_config_is_section_1_and_2():
    B = good_bank()
    assert dag.g_bank_config(B, EVAL_IDS)[0] is True
    c = lambda B: B["man"]["common"]
    edits = {"partial": lambda B: c(B).update(partial=True), "limit_clips": lambda B: c(B).update(limit_clips=12), "limit": lambda B: c(B)["rollout"].update(limit="logged"),
             "replan": lambda B: c(B)["rollout"].update(replan_steps=5), "executor": lambda B: c(B)["rollout"].update(executor="gt"), "nav": lambda B: c(B)["rollout"].update(nav="logged"),
             "post_map": lambda B: c(B)["rollout"].update(post_map="learned"), "scene": lambda B: c(B)["rollout"].update(scene="rendered"),
             "schedule": lambda B: c(B)["rollout"].update(schedule="every"), "extra_rollout_key": lambda B: c(B)["rollout"].update(ticks=[1, 2]),
             "reasons": lambda B: c(B)["constants"].update(REASONS=["ok", "off_lat"]), "thresholds": lambda B: c(B).update(thresholds_file_sha256="00" * 32),
             "thresholds_none": lambda B: c(B).update(thresholds_file_sha256=None),
             "hmax": lambda B: c(B)["cfg"].update(hmax=8), "train_windows": lambda B: c(B)["train"].update(n_windows=1000), "train_clips": lambda B: c(B)["train"].update(n_clips=10),
             "ckpt_hex": lambda B: c(B).update(ckpt_sha256="xyz"), "window_hex": lambda B: c(B).update(window_list_sha256="xyz"), "train_hex": lambda B: c(B)["train"].update(train_sha256="xyz"),
             "eval_overlap": lambda B: c(B)["eval"].update(overlap=1), "eval_clips": lambda B: c(B)["eval"].update(clips=c(B)["eval"]["clips"][:-1] + ["eval-other"]),
             "schema": lambda B: c(B).update(schema="v9")}
    for k in dag.MATCH_CONSTANTS:
        edits["const_" + k] = (lambda k: lambda B: c(B)["constants"].update({k: [1, 3] if k == "TICKS" else c(B)["constants"][k] + 1}))(k)
    for nm, fn in edits.items():
        B = good_bank()
        fn(B)
        assert not dag.g_bank_config(B, EVAL_IDS)[0], nm
    assert not dag.g_bank_config(good_bank(), [])[0]                                      # an empty eval list proves nothing
    assert not dag.g_bank_config(good_bank(), EVAL_IDS[:-1])[0]                           # the eval list given is not the one the generator checked


def test_kb0_checks_every_window_and_sums_the_shards():
    assert dag.g_kb0(good_bank())[0]
    for nm, fn in {"short": lambda B: B["shard_provs"][0]["kb0_tick0_identity"].update(checked=B["shard_provs"][0]["kb0_tick0_identity"]["checked"] - 1),
                   "mismatch": lambda B: B["shard_provs"][1]["kb0_tick0_identity"].update(mismatch=1),
                   "none_checked": lambda B: [sp["kb0_tick0_identity"].update(checked=0) for sp in B["shard_provs"]],
                   "bool": lambda B: B["shard_provs"][0]["kb0_tick0_identity"].update(checked=True),
                   "missing": lambda B: B["shard_provs"][1].pop("kb0_tick0_identity"), "no_shards": lambda B: B.update(shard_provs=[])}.items():
        B = good_bank()
        fn(B)
        raises(dag.g_kb0, B)
    B = good_bank()
    B["shard_provs"][0]["kb0_tick0_identity"]["checked"], B["shard_provs"][1]["kb0_tick0_identity"]["checked"] = N_WIN, 0       # only the SUM must be every window
    assert dag.g_kb0(B)[0]


def test_kb1_replays_half_a_percent_of_the_rollouts_bit_for_bit():
    assert dag.g_kb1(good_bank())[1]["required"] == 1532 and dag.g_kb1(good_bank())[1]["windows"] == 1536
    for w1, ok in ((766, True), (765, False), (0, False)):                                  # floor(0.005 x 306,566) = 1532 windows in all: 766 + 766 is the bar
        B = good_bank()
        B["shard_provs"][0]["kb1_replay"]["windows"], B["shard_provs"][1]["kb1_replay"]["windows"] = 766, w1
        (dag.g_kb1(B) if ok else raises(dag.g_kb1, B, match="windows >= 1532"))
    for nm, fn in {"mismatch": lambda B: B["shard_provs"][1]["kb1_replay"].update(mismatch_chunks=1), "frac": lambda B: B["shard_provs"][0]["kb1_replay"].update(frac=0.004),
                   "chunks0": lambda B: [sp["kb1_replay"].update(chunks=0) for sp in B["shard_provs"]], "bool": lambda B: B["shard_provs"][0]["kb1_replay"].update(chunks=True),
                   "missing": lambda B: B["shard_provs"][0].pop("kb1_replay"), "frac_str": lambda B: B["shard_provs"][0]["kb1_replay"].update(frac="0.5")}.items():
        B = good_bank()
        fn(B)
        raises(dag.g_kb1, B)
    B = good_bank()
    B["shard_provs"][0]["kb1_replay"]["frac"] = 0.005                                       # the fraction is inclusive
    assert dag.g_kb1(B)[0]


def test_kb2_generator_checkpoint_must_be_the_m2b_checkpoint():
    assert dag.g_kb2(good_bank(), "ab" * 32)[0]
    raises(dag.g_kb2, good_bank(), "ef" * 32, match="on-policy")
    B = good_bank()
    B["shard_provs"][1]["inputs"]["ckpt"]["sha256"] = "11" * 32                              # a shard generated by another checkpoint
    raises(dag.g_kb2, B, "ab" * 32, match="shard's inputs.ckpt.sha256")
    B = good_bank()
    B["man"]["common"]["ckpt_sha256"] = "11" * 32
    raises(dag.g_kb2, B, "ab" * 32, match="on-policy")


# ---- K-B3, Amendment 1 section 3: THREE controls, each failing on its own
def test_kb31_bank_clips_are_a_subset_of_TRAIN_by_name_and_fewer_is_fine():
    ok, det = dag.g_kb31(good_bank(), CLIPS)
    assert ok and det["n_bank_clips"] == 4512 and det["n_train_clips_without_bank_records"] == 0 and det["deliberate_regression"] == "refused"
    ok, det = dag.g_kb31(good_bank(clips=CLIPS[:4000]), CLIPS)                               # equality is NOT required
    assert ok and det["n_bank_clips"] == 4000 and det["n_train_clips_without_bank_records"] == 512
    raises(dag.g_kb31, good_bank(clips=CLIPS[:-1] + ["clip-not-in-train"]), CLIPS, match="not in M2b's TRAIN clip list")
    raises(dag.g_kb31, good_bank(), CLIPS[:-1], match="not in M2b's TRAIN")                                                       # the other direction: TRAIN lacks a clip the bank holds
    raises(dag.g_kb31, good_bank(), [], match="TRAIN clip list is empty")
    raises(dag.g_kb31, good_bank(clips=[]), CLIPS, match="holds no clips")
    orig = dag.train_subset_problems                                                        # a BLIND subset check is itself refused by the reader's own regression
    dag.train_subset_problems = lambda a, b: []
    try:
        raises(dag.g_kb31, good_bank(), CLIPS, match="blind")
    finally:
        dag.train_subset_problems = orig


def test_kb33_bank_clips_and_eval141_are_disjoint_and_the_regression_fires():
    ok, det = dag.g_kb33(good_bank(), EVAL_IDS)
    assert ok and det["overlap"] == 0 and det["deliberate_regression"] == "refused"
    raises(dag.g_kb33, good_bank(clips=CLIPS[:-1] + [EVAL_IDS[3]]), EVAL_IDS, match="overlap")      # one eval clip in the bank
    raises(dag.g_kb33, good_bank(), [], match="empty")
    B = good_bank()
    B["shard_provs"][0]["kb3_eval_disjoint"]["overlap"] = 1                                 # the generator's own record
    raises(dag.g_kb33, B, EVAL_IDS, match="kb3_eval_disjoint")
    B = good_bank()
    B["shard_provs"][1]["kb3_eval_disjoint"]["n_eval_clips"] = 140
    raises(dag.g_kb33, B, EVAL_IDS, match="kb3_eval_disjoint")
    B = good_bank()
    B["shard_provs"][1].pop("kb3_eval_disjoint")
    raises(dag.g_kb33, B, EVAL_IDS, match="kb3_eval_disjoint")
    orig = dag.clip_overlap
    dag.clip_overlap = lambda a, b: []
    try:
        raises(dag.g_kb33, good_bank(), EVAL_IDS, match="blind")
    finally:
        dag.clip_overlap = orig


def test_the_three_kb3_checks_each_own_their_failure():
    """The same defect must fail exactly the check it belongs to: a non-TRAIN clip fails 3.1 only, a contaminated TRAIN list fails 3.3 only, a window-list mismatch fails 3.2 only."""
    not_train = good_bank(clips=CLIPS[:-1] + ["clip-not-in-train"])
    raises(dag.g_kb31, not_train, CLIPS)
    assert dag.g_kb33(not_train, EVAL_IDS)[0]
    dirty = CLIPS + [EVAL_IDS[5]]                                                           # an eval clip in M2b's TRAIN list AND in the bank: it IS a subset, and it is NOT disjoint
    B = good_bank(clips=CLIPS[:-1] + [EVAL_IDS[5]])
    assert dag.g_kb31(B, dirty)[0]
    raises(dag.g_kb33, B, EVAL_IDS, match="overlap")
    k = rec_with({"train_window_list_sha256": WIN_SHA})
    bad = rec_with({"train_window_list_sha256": "00" * 32})
    assert dag.g_kb31(good_bank(), CLIPS)[0] and dag.g_kb33(good_bank(), EVAL_IDS)[0] and not dag.g_kb32(good_bank(), bad, k, "train_window_list_sha256")[0]


def rec_with(sums):
    return fake_rec([], sums)


def test_kb32_trainer_window_list_sha_is_the_banks_on_both_arms():
    B, key = good_bank(), "train_window_list_sha256"
    k = rec_with({key: WIN_SHA})
    ok, det = dag.g_kb32(B, k, k, key)
    assert ok is True and det["window_list_sha256"] == WIN_SHA
    bad = rec_with({key: "00" * 32})
    assert not dag.g_kb32(B, bad, k, key)[0] and not dag.g_kb32(B, k, bad, key)[0]                     # either arm alone
    ok, det = dag.g_kb32(B, bad, bad, key)                                                            # both arms agree with each other and disagree with the bank
    assert not ok and "!= the trainer's" in str(det)
    assert not dag.g_kb32(B, rec_with({}), k, key)[0] and not dag.g_kb32(B, k, rec_with({}), key)[0]  # not recorded
    assert "needs the trainer to add it" in str(dag.g_kb32(B, rec_with({}), k, key)[1])
    other = rec_with({key: "11" * 32})
    ok, det = dag.g_kb32(B, k, other, key)                                                            # the arms differ from each other (one equals the bank)
    assert not ok and "differs between DAG and DAG-0" in str(det)
    conflict = rec_with({key: WIN_SHA, "levers": {key: "11" * 32}})
    assert not dag.g_kb32(B, conflict, k, key)[0] and "conflicting" in str(dag.g_kb32(B, conflict, k, key)[1])
    B2 = good_bank()
    B2["man"]["common"]["window_list_sha256"] = "44" * 32                                             # the bank's own sha moved
    assert not dag.g_kb32(B2, k, k, key)[0]
    assert dag.g_kb32(B, rec_with({"tr": WIN_SHA}), rec_with({"tr": WIN_SHA}), "tr")[0]               # the key is overridable
    assert not dag.g_kb32(B, rec_with({"train_sha256": TRAIN_SHA}), rec_with({"train_sha256": TRAIN_SHA}), key)[0]   # the CLIP-list sha is not a window-list sha


def test_train_list_binds_the_clip_list_to_the_trainer_and_the_generators_plan():
    B, key = good_bank(), "train_sha256"
    k = rec_with({key: TRAIN_SHA})
    ok, det = dag.g_train_list(B, CLIPS, k, k, key)
    assert ok and det["train_sha256"] == TRAIN_SHA
    assert not dag.g_train_list(B, CLIPS[:-1], k, k, key)[0]                                         # a list that is not the trainer's
    bad = rec_with({key: "00" * 32})
    assert not dag.g_train_list(B, CLIPS, bad, k, key)[0] and not dag.g_train_list(B, CLIPS, k, bad, key)[0]
    assert not dag.g_train_list(B, CLIPS, rec_with({}), k, key)[0]
    B2 = good_bank()
    B2["man"]["common"]["train"]["train_sha256"] = "55" * 32                                          # the generator planned over another TRAIN set
    ok, det = dag.g_train_list(B2, CLIPS, k, k, key)
    assert not ok and "planned over" in str(det)


def test_kb4_every_matcher_case_and_kb5_tolerance_come_from_the_generators_controls_block():
    assert dag.g_kb4(good_bank())[0] and dag.g_kb5(good_bank())[0]
    for case in dag.KB4_CASES:
        for bad in (False, None):
            B = good_bank()
            if bad is None:
                del B["prov"]["controls"]["K-B4"]["cases"][case]
            else:
                B["prov"]["controls"]["K-B4"]["cases"][case] = bad
            raises(dag.g_kb4, B, match=case)
    for k in ("max_pose_now_err_m", "max_rel_transform_err_m"):
        for v, ok in ((1e-4, True), (1.0001e-4, False), (float("nan"), False), (None, False), ("0", False)):
            B = good_bank()
            B["prov"]["controls"]["K-B5"][k] = v
            (dag.g_kb5(B) if ok else raises(dag.g_kb5, B))
    for kid, fn in (("K-B4", dag.g_kb4), ("K-B5", dag.g_kb5)):
        for bad in (False, None, "yes"):
            B = good_bank()
            B["prov"]["controls"][kid]["PASS"] = bad
            raises(fn, B, match="PASS")
    for edit in (lambda B: B["prov"].pop("controls"), lambda B: B["prov"].update(controls=[]), lambda B: B["prov"]["controls"].pop("K-B4")):    # what the generator writes TODAY
        B = good_bank()
        edit(B)
        raises(dag.g_kb4, B, match="needs the bank generator to add")
    B = good_bank()
    B["prov"]["controls"].pop("K-B5")
    raises(dag.g_kb5, B, match="needs the bank generator to add")


def test_bank_accounting_and_coverage_arithmetic():
    ok, det = dag.g_bank_accounting(good_bank())
    assert ok and det["n_states_before_cuts"] == 2 * 306566 and det["eligible_fraction"] == pytest.approx(250000 / 306566)
    assert det["states_by_tick"] == {"1": N_WIN, "2": N_WIN} and det["n_states_kept"] == 430000 and det["cuts"] == {"off_lat": 103000, "off_head": 21000, "end_edge": 59127, "nonfinite": 5}
    edits = {
        "tick2_short_n_records_consistent": lambda B: both(B, lambda d: (d["reasons_by_tick"]["2"].update(end_edge=d["reasons_by_tick"]["2"]["end_edge"] - 1), d.update(n_records=2 * N_WIN - 1))),
        "tick1_over_tick2_under_total_right": lambda B: both(B, lambda d: (d["reasons_by_tick"]["1"].update(end_edge=d["reasons_by_tick"]["1"]["end_edge"] + 1),
                                                                          d["reasons_by_tick"]["2"].update(end_edge=d["reasons_by_tick"]["2"]["end_edge"] - 1))),
        "total_2n_plus_1_n_records_consistent": lambda B: both(B, lambda d: (d["reasons_by_tick"]["1"].update(end_edge=d["reasons_by_tick"]["1"]["end_edge"] + 1), d.update(n_records=2 * N_WIN + 1))),
        "extra_tick": lambda B: both(B, lambda d: d["reasons_by_tick"].update({"3": {r: 0 for r in dag.REASONS}})),
        "one_tick": lambda B: both(B, lambda d: d["reasons_by_tick"].pop("2")),
        "n_records_prov": lambda B: B["prov"].update(n_records=B["prov"]["n_records"] + 1),
        "n_records_man": lambda B: B["man"].update(n_records=B["man"]["n_records"] + 1),
        "elig_gt_kept": lambda B: both(B, lambda d: d["coverage"].update(n_eligible=430001, fraction=430001 / N_WIN)),
        "frac_stated": lambda B: both(B, lambda d: d["coverage"].update(fraction=0.9)),
        "frac_missing": lambda B: both(B, lambda d: d["coverage"].pop("fraction")),
        "blocked_flag": lambda B: both(B, lambda d: d["coverage"].update(blocked=True)),
        "min": lambda B: both(B, lambda d: d["coverage"].update(min=0.4)),
        "man_coverage_differs": lambda B: B["man"]["coverage"].update(n_eligible=B["man"]["coverage"]["n_eligible"] - 1, fraction=(B["man"]["coverage"]["n_eligible"] - 1) / N_WIN),
        "man_reasons_differ": lambda B: B["man"]["reasons_by_tick"]["1"].update(ok=B["man"]["reasons_by_tick"]["1"]["ok"] + 1),
        "elig_gt_kept_small": lambda B: both(B, lambda d: [d["reasons_by_tick"][t].update(ok=100000, end_edge=d["reasons_by_tick"][t]["end_edge"] + d["reasons_by_tick"][t]["ok"] - 100000) for t in "12"]),
        "elig_gt_windows": lambda B: both(B, lambda d: d["coverage"].update(n_eligible=N_WIN + 1, fraction=(N_WIN + 1) / N_WIN)),
        "nwin_not_train_everything_else_consistent": lambda B: both(B, lambda d: (d["coverage"].update(n_windows=N_WIN - 1, fraction=250000 / (N_WIN - 1)),
                                                                                  [d["reasons_by_tick"][t].update(end_edge=d["reasons_by_tick"][t]["end_edge"] - 1) for t in "12"], d.update(n_records=2 * (N_WIN - 1)))),
    }
    for nm, fn in edits.items():
        B = good_bank()
        fn(B)
        assert not dag.g_bank_accounting(B)[0], nm
    B = good_bank()                                                                          # a PARTIAL bank's window count is its own (BANK-CONFIG refuses partial banks)
    B["man"]["common"]["partial"] = True
    B["man"]["common"]["train"]["n_windows"] = 999
    assert dag.g_bank_accounting(B)[0]
    for nm, fn in {"reason": lambda B: both(B, lambda d: d["reasons_by_tick"]["1"].pop("off_head")), "neg": lambda B: both(B, lambda d: d["reasons_by_tick"]["1"].update(off_lat=-1)),
                   "none": lambda B: B["prov"].pop("coverage"), "no_rbt": lambda B: B["prov"].pop("reasons_by_tick"), "nwin0": lambda B: B["prov"]["coverage"].update(n_windows=0),
                   "elig_float": lambda B: B["prov"]["coverage"].update(n_eligible=1.5), "bool": lambda B: B["prov"]["coverage"].update(n_windows=True)}.items():
        B = good_bank()
        fn(B)
        raises(dag.coverage_of, B)
    assert dag.coverage_of(good_bank(n_elig=153283))["eligible_fraction"] == 0.5 and dag.coverage_of(good_bank(n_elig=153282))["eligible_fraction"] < 0.5


def test_bank_declared_argv_trainer_record_init_and_bank_hash():
    bsha, msha = good_bank()["man_sha256"], "ab" * 32
    B = good_bank()
    argv = ["--init-ckpt", "/nope/M2b_s0/ckpt_last.pt", "--dagger-bank-sha256", bsha]
    sums = {"levers": {"dagger_bank_sha256": bsha, "init_ckpt": {"sha256": msha}}}
    f = lambda r, B=B: dag.g_bank_declared("t", 0, B, r, msha, "init_ckpt.sha256", "dagger_bank_sha256")
    ok, det = f(fake_rec(argv, sums))
    assert ok is True and det["bank_manifest_hashed_here"] is True and det["bank_sha256"] == bsha
    assert not f(fake_rec(["--init-ckpt", "/n", "--dagger-bank-sha256", "11" * 32], sums))[0]                         # argv declares another bank
    assert not f(fake_rec(argv, {**sums, "levers": {**sums["levers"], "dagger_bank_sha256": "11" * 32}}))[0]          # the trainer BUILT another bank
    assert not f(fake_rec(argv, {"levers": {"init_ckpt": {"sha256": msha}}}))[0]                                      # no built record
    assert not f(fake_rec(argv, {"levers": {"dagger_bank_sha256": bsha}}))[0]                                         # no init record
    assert not f(fake_rec(argv, {"levers": {"dagger_bank_sha256": bsha, "init_ckpt": {"sha256": "11" * 32}}}))[0]     # K-B2 on the arm: init is not the generator
    B2 = good_bank()
    B2["man"]["common"]["ckpt_sha256"] = "11" * 32
    assert not f(fake_rec(argv, sums), B=B2)[0]
    B3 = good_bank()
    B3["man_sha256"] = "11" * 32                                                                                       # bank.json hashes to something else than the arm declared
    assert not f(fake_rec(argv, sums), B=B3)[0]


