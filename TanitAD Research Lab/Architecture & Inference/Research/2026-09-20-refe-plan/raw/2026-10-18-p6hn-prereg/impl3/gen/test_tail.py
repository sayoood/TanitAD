

# ------------------------------------------------------------------------------------------------ the judges of the RAW G14 / G15 fields (never the trainer's own `ok`)
def g14_bad(g) -> list:
    b: list = []

    def c(n, f):
        check(b, "G14_" + n, f)
    gs = g["grad_abs_sum"]
    c("n_hn_positive", lambda: g["n_hn"] > 0)
    c("grads_finite", lambda: g["grads_finite"] is True)
    c("value_on_positive", lambda: g["value_on"] > 0.0)
    c("grad_q_mlp_nonzero", lambda: gs["score_q_mlp"] > 0.0)
    c("grad_dec_nonzero", lambda: gs["score_dec"] > 0.0)
    c("grad_head_rows_0_5_nonzero", lambda: gs["score_head_rows_0_5"] > 0.0)
    c("lane_head_present", lambda: g["lane_head"] is True)                                    # the exact-zero control reads zero for a reason only if there IS a lane row
    c("lane_row_exactly_zero", lambda: gs["score_head_row_6_lane"] == 0.0)
    c("armA_exactly_zero", lambda: gs["armA_all"] == 0.0)
    c("base_and_frozen_no_grad", lambda: gs["base_params_with_grad"] == 0 and gs["frozen_params_with_grad"] == 0 and g["trunk_requires_grad"] is False)
    c("armA_bce_control_nonzero", lambda: len(g["A_bce_control"]) == 3 and all(v > 0.0 for v in g["A_bce_control"].values()))
    c("controls_exactly_zero", lambda: set(g["controls"]) == {"off", "empty"} and all(v["value"] == 0.0 and v["B_all"] == 0.0 and v["A_all"] == 0.0 for v in g["controls"].values()))
    c("two_step_identities", lambda: g["two_step"]["off_equal_A"] and g["two_step"]["empty_equal_A"] and g["two_step"]["on_differs_from_A"] and g["two_step"]["probe_reads_one_ulp"])
    c("dup_identity", lambda: g["dup_identity"]["share"] > 0.0 and g["dup_identity"]["abs_diff"] <= 1e-5 * max(1.0, g["dup_identity"]["share"]))
    c("pad_mask_equals_bank", lambda: g["pad"]["served_mask_equals_bank"] and g["pad"]["pad_slots"] > 0)
    c("pad_perturbation_bit_identical", lambda: g["pad"]["perturbed_loss_equal"])
    c("fan_call_bit_identical", lambda: g["fan_call_max_abs_diff"] == 0.0)
    c("fan_rows_isolated_from_hn", lambda: g["fan_rows_hn_isolation_max_abs"] == 0.0)
    c("hn_row_independent_of_other_hn_rows", lambda: g["hn_row_independence_max_abs"] == 0.0)
    c("d1_rows", lambda: g["d1"]["rows_in_term"] == g["d1"]["rows_expected"] and g["d1"]["rows_expected"] > 0)
    c("eval_reads_no_bank", lambda: g["eval_reads_bank"]["held_items_checked"] > 0 and g["eval_reads_bank"]["held_items_with_hn_tail"] == 0)
    c("mutate_list_empty", lambda: g["mutate"] == [])
    return b


def g15_bad(g) -> list:
    b: list = []

    def c(n, f):
        check(b, "G15_" + n, f)
    c("ctx_bit_identical", lambda: g["ctx_max_abs_diff"] == 0.0)
    c("logits_bit_identical", lambda: set(g["logit_max_abs_diff"]) == {"base", "A", "B"} and all(v == 0.0 for v in g["logit_max_abs_diff"].values()))
    c("shuffled_cache_far", lambda: g["shuffled"]["ctx_max_abs_diff"] > 0.0 and all(g["shuffled"]["logit_max_abs_diff"][k] > 10.0 * g["logit_max_abs_diff"][k] and g["shuffled"]["logit_max_abs_diff"][k] > 0.0
                                                                                      for k in ("base", "A", "B")))
    c("trunk_fingerprint_bound", lambda: g["trunk_fingerprint_manifest"] == g["trunk_fingerprint_checkpoint"] and len(g["trunk_fingerprint_checkpoint"]) == 64)
    c("covers_run", lambda: g["covers_run"] == {"missing_train": 0, "missing_val": 0})
    c("dtype_bf16", lambda: g["dtype"] == "bf16")
    c("calib_bound", lambda: g["calib_sha256_run"] is not None and g["calib_sha256_manifest"] == g["calib_sha256_run"])
    c("scenes_compared", lambda: g["n_train"] == min(64, NT) and g["n_val"] == min(64, NH) and g["micro_batch"] == 2)
    c("mutate_list_empty", lambda: g["mutate"] == [])
    return b


# ------------------------------------------------------------------------------------------------ the suite
def evals(res):
    return [r for r in res["rows"]["sft.jsonl"] if r.get("event") == "eval"]


def trains(res):
    return [r for r in res["rows"]["sft.jsonl"] if r.get("event") == "train"]


def expected_micro_hn_total(n_micro: int, counts) -> int:
    """the hard-negative rows the loop MUST have fed arm B over its first `n_micro` micro-batches: replay the loader's shuffle (a Generator seeded 0, one randperm per epoch, batches of 2 consecutive, drop_last) and sum
    the bank's per-scene counts -- computed here from the fixture's construction, not from the trainer's own mask"""
    g = torch.Generator().manual_seed(0)
    tot, done = 0, 0
    while done < n_micro:
        perm = torch.randperm(NT, generator=g).tolist()
        for j in range(NT // 2):
            if done >= n_micro:
                break
            tot += counts[perm[2 * j]] + counts[perm[2 * j + 1]]
            done += 1
    return tot


def suite(rigs, srcs, groups, tag="x") -> list:
    """-> the list of FAILING check names.  srcs = {"sf": trainer source, "hn": hn_bank source | None, "cc": ctx_cache source | None}"""
    bad: list = []
    SF, HNS, CCS = srcs["sf"], srcs.get("hn"), srcs.get("cc")
    PRE = read_text(PRE_PATH)
    r7, rp = rigs["n7"], rigs["plain"]

    def run(src, rig, name, **kw):
        return wrun(src, rig, f"{tag}_{name}_{rig.name}", **kw)

    if "PIN" in groups:
        mod = load_mod(SF, "sft_wire_pin_new", SRC_PATH)
        pre_mod = load_mod(PRE, "sft_wire_pin_pre", SRC_PATH)
        import rls1_checks as RC
        check(bad, "PIN_peel_equals_tip_blob", lambda: blob_sha1(HL.peel(SF)) == TIP_BLOB == HL.TIP_TRAINER_BLOB)
        check(bad, "PIN_carvefeed_chain_equals_pre_B8_blob", lambda: blob_sha1(CL.peel(SF)) == PRE_B8_BLOB)
        check(bad, "PIN_fixture_sha256", lambda: hashlib.sha256(PRE.encode("utf-8")).hexdigest() == PRE_SHA256 and blob_sha1(PRE) == TIP_BLOB)
        check(bad, "PIN_fixture_is_the_peeled_trainer", lambda: HL.peel(SF) == PRE)
        check(bad, "PIN_layer_marker", lambda: HL.has_layer(SF) and not HL.has_layer(PRE) and HL.peel(PRE) == PRE)
        check(bad, "PIN_partition_and_label_version", lambda: mod.HN_PARTITION_ID == RC.PARTITION["partition_id"] and mod.HN_LABEL_VERSION == 5 and mod.HN_ARM == "B")
        check(bad, "PIN_new_dests_are_exactly_the_two_flags_default_empty", lambda: (
            {a_.dest for a_ in mod.build_parser()._actions} - {a_.dest for a_ in pre_mod.build_parser()._actions} == {"hn_bank", "ctx_cache"}
            and {a_.dest for a_ in pre_mod.build_parser()._actions} <= {a_.dest for a_ in mod.build_parser()._actions}
            and mod.build_parser().parse_args(["--ckpt", "x"]).hn_bank == "" and mod.build_parser().parse_args(["--ckpt", "x"]).ctx_cache == ""))

        def nested_identical():
            def nest(src):
                m = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main")
                return {n.name: ast.dump(n) for n in m.body if isinstance(n, ast.FunctionDef)}
            a_, b_ = nest(PRE), nest(SF)
            return set(a_) <= set(b_) and all(a_[k] == b_[k] for k in a_) and len(a_) >= 5
        check(bad, "PIN_every_nested_function_of_main_is_AST_identical", nested_identical)
        check(bad, "PIN_loop_rebinding_only_inside_if", lambda: SF.count("\n    def fwd(batch):") == 1 and SF.count("\n        def fwd(batch):") == 1)

    if "I3" in groups:
        for rg in (rp, r7):
            a_, b_ = run(PRE, rg, "pre"), run(SF, rg, "new")
            a2 = run(PRE, rg, "pre", nonce="again")
            n = "I3_" + rg.name
            check(bad, n + "_runs_complete", lambda: a_["rc_pre"] == 0 and a_["rc_run"] == 0 and b_["rc_pre"] == 0 and b_["rc_run"] == 0 and a_["pass_json"] is not None)
            check(bad, n + "_preflight_gates_equal", lambda: a_["gates"] is not None and rows_equal(a_["gates"], b_["gates"]))
            check(bad, n + "_every_logged_row_equal", lambda: len(a_["rows"]["sft.jsonl"]) >= 6 and rows_equal(a_["rows"]["sft.jsonl"], b_["rows"]["sft.jsonl"]))
            check(bad, n + "_has_a_train_row_and_evals", lambda: len(trains(b_)) == 1 and len(evals(b_)) >= 3 and "hn_rows" not in trains(b_)[0])
            check(bad, n + "_saved_tensors_A_B_equal", lambda: same_sd(a_["A"], b_["A"]) and same_sd(a_["B"], b_["B"]))
            check(bad, n + "_data_event_has_no_hn_or_cache_block", lambda: "hn" not in b_["data"] and "ctx_cache" not in b_["data"])
            check(bad, n + "_control_pre_edit_twice_is_deterministic", lambda: rows_equal(a_["rows"]["sft.jsonl"], a2["rows"]["sft.jsonl"]) and same_sd(a_["A"], a2["A"]) and same_sd(a_["B"], a2["B"]))
            c_ = run(SF, rg, "newlr", lr="1.1e-3")
            check(bad, n + "_control_comparison_sees_a_changed_lr", lambda: not same_sd(b_["A"], c_["A"]) and not rows_equal(b_["rows"]["sft.jsonl"], c_["rows"]["sft.jsonl"]))
        check(bad, "I3_B_equals_A_without_the_bank_lam0", lambda: same_sd(run(SF, r7, "new")["A"], run(SF, r7, "new")["B"]))

    if "ON" in groups:
        counts = expected_hn_counts()
        RL = run(SF, r7, "new")                                                                         # flags absent
        RC_ = run(SF, r7, "cache", extra=("--ctx-cache", r7.cache), hn_src=HNS, cc_src=CCS)
        RCB = run(SF, r7, "cb", extra=("--ctx-cache", r7.cache, "--hn-bank", r7.bank), hn_src=HNS, cc_src=CCS)
        RB = run(SF, r7, "bank", extra=("--hn-bank", r7.bank), hn_src=HNS, cc_src=CCS)
        RE = run(SF, r7, "cbempty", extra=("--ctx-cache", r7.cache, "--hn-bank", r7.bank), hn_src=HNS, cc_src=CCS, empty_term=True)
        for nm, rr in (("cache", RC_), ("cb", RCB), ("bank", RB), ("cbempty", RE)):
            check(bad, f"ON_{nm}_preflight_pass_and_run_complete", lambda rr=rr: rr["rc_pre"] == 0 and rr["rc_run"] == 0 and rr["pass_json"] is not None and rr["A"] is not None and rr["B"] is not None)
        # the cache alone is the live trunk, bit for bit
        check(bad, "ON_cache_only_equals_live_tensors", lambda: same_sd(RL["A"], RC_["A"]) and same_sd(RL["B"], RC_["B"]))
        check(bad, "ON_cache_only_equals_live_eval_and_train_rows", lambda: len(evals(RL)) >= 3 and rows_equal(evals(RL), evals(RC_)) and rows_equal(trains(RL), trains(RC_)))
        check(bad, "ON_cache_only_B_equals_A", lambda: same_sd(RC_["A"], RC_["B"]))
        # the bank moves arm B only
        check(bad, "ON_bank_leaves_arm_A_bit_identical", lambda: same_sd(RC_["A"], RCB["A"]))
        check(bad, "ON_bank_moves_arm_B", lambda: not same_sd(RC_["B"], RCB["B"]) and not same_sd(RCB["A"], RCB["B"]))
        check(bad, "ON_bank_arm_A_and_base_eval_numbers_equal_the_cache_only_run", lambda: all(x["A"] == y["A"] and x["base"] == y["base"] for x, y in zip(evals(RC_), evals(RCB))) and len(evals(RCB)) >= 3)
        check(bad, "ON_bank_step0_eval_row_fully_equal_I4", lambda: rows_equal(evals(RC_)[0], evals(RCB)[0]) and evals(RCB)[0]["tag"] == "step0")
        check(bad, "ON_bank_arm_B_eval_moves_after_training", lambda: any(x["B"] != y["B"] for x, y in zip(evals(RC_)[1:], evals(RCB)[1:])))
        check(bad, "ON_live_trunk_bank_run_equals_cached_bank_run", lambda: same_sd(RB["A"], RCB["A"]) and same_sd(RB["B"], RCB["B"]) and rows_equal(evals(RB), evals(RCB)))
        check(bad, "ON_empty_term_B_equals_A_equals_cache_only", lambda: same_sd(RE["A"], RE["B"]) and same_sd(RE["A"], RC_["A"]))
        # D1: the rows that entered L_B equal the rows the bank holds for the scenes the loop served (independent replay of the loader)
        check(bad, "ON_D1_train_row_counts", lambda: len(trains(RCB)) == 1 and trains(RCB)[0]["hn_rows"] == trains(RCB)[0]["hn_rows_expected"] > 0 and trains(RCB)[0]["hn_micro"] == 50
              and trains(RCB)[0]["hn_micro_nonempty"] > 0 and trains(RCB)[0]["hn_loss"] > 0.0)
        check(bad, "ON_D1_rows_equal_an_independent_replay_of_the_loader", lambda: trains(RCB)[0]["hn_rows"] == expected_micro_hn_total(50, counts))
        check(bad, "ON_D1_no_bank_no_hn_fields", lambda: all("hn_rows" not in t for t in trains(RC_) + trains(RL)))
        # the data event
        bk = HN0.load_bank(r7.bank)
        check(bad, "ON_data_event_hn_block_is_the_banks", lambda: RCB["data"]["hn"] == bk.hn_block("B") and RCB["data"]["hn"]["rows"] == sum(counts) and RCB["data"]["hn"]["arm"] == "B" and "ctx_cache" not in RB["data"])
        check(bad, "ON_data_event_cache_block", lambda: RCB["data"]["ctx_cache"] == {"manifest_sha256": CC0.sha256_file(os.path.join(r7.cache, "manifest.json")), "trunk_fingerprint": r7.trunk_fp,
                                                                                       "n_scenes": NT + NH, "dtype": "bf16"} and "hn" not in RC_["data"])
        # the raw gate fields
        g14 = RCB["gates"]["G14_hn_live_and_identity"] if RCB["gates"] else None
        g15 = RCB["gates"]["G15_ctx_cache_identity"] if RCB["gates"] else None
        for nm in (g14_bad(g14) if g14 else ["G14_missing"]):
            bad.append(nm)
        for nm in (g15_bad(g15) if g15 else ["G15_missing"]):
            bad.append(nm)
        check(bad, "ON_G14_absent_without_bank_G15_absent_without_cache", lambda: "G14_hn_live_and_identity" not in RC_["gates"] and "G15_ctx_cache_identity" not in RB["gates"] and "G14_hn_live_and_identity" in RB["gates"]
              and "G14_hn_live_and_identity" not in RL["gates"] and "G15_ctx_cache_identity" not in RL["gates"])
        check(bad, "ON_bank_only_G14_lane_row_zero_and_pass", lambda: g14_bad(RB["gates"]["G14_hn_live_and_identity"]) == [])
        # the gate rows the launch gate reads: ok is the trainer's, the judges above are ours -- both must agree
        check(bad, "ON_trainer_ok_agrees_with_the_judges", lambda: g14["ok"] is True and g15["ok"] is True and RCB["pass_json"]["verdict"] == "PASS")

    if "REF" in groups:
        good_cache, good_bank = r7.cache, r7.bank

        def ref(name, text, extra, **kw):
            res = run(SF, r7, "ref_" + name, extra=extra, phases=("pre",), hn_src=HNS, cc_src=CCS, **kw)
            check(bad, "REF_" + name, lambda: refused(res, text) and not os.path.exists(os.path.join(r7.td, f"run_{tag}_ref_{name}_{r7.name}", "out", "PREFLIGHT_PASS.json")))
        ref("hn_with_epo", "--hn-bank with --epo", ("--hn-bank", good_bank, "--epo"))
        ref("hn_with_lam", "--hn-bank needs --lam 0", ("--hn-bank", good_bank, "--lam", "0.5"))
        ref("hn_with_bmode", "--hn-bank needs --b-mode listnet", ("--hn-bank", good_bank, "--b-mode", "compw"))
        ref("hn_with_aux", "--hn-bank with --aux-drv", ("--hn-bank", good_bank, "--aux-drv"))
        ref("cache_with_aux", "--ctx-cache with --aux-drv", ("--ctx-cache", good_cache, "--aux-drv"))
        ref("bank_other_partition", "bank partition_id", ("--hn-bank", r7.mint("partition", partition_id="0" * 64)))
        ref("bank_other_label_version", "bank label_version", ("--hn-bank", r7.mint("lv4", label_version=4)))
        ref("bank_other_fan_fingerprint", "bank fan_fingerprint", ("--hn-bank", r7.mint("fanfp", fan_fp="1" * 64)))
        flipped = os.path.join(r7.td, "bank_flipped.hnb")
        with open(good_bank, "rb") as f_:
            data_ = bytearray(f_.read())
        data_[-1] ^= 0xFF
        with open(flipped, "wb") as f_:
            f_.write(bytes(data_))
        with open(good_bank + ".manifest.json", "rb") as f_, open(flipped + ".manifest.json", "wb") as g_:
            g_.write(f_.read())
        ref("bank_bytes_changed", "changed after minting", ("--hn-bank", flipped))
        stat = anchors_for(r7.train_keys)
        for a_ in stat:
            a_["traj"] = (np.array(a_["traj"]) * 0.2).tolist()                                           # every path < 2.0 m: STATIONARY, no row is labelled, 0 hard negatives
        ref("bank_zero_hard_negatives", "0 hard-negative rows", ("--hn-bank", r7.mint("empty", anchors=stat)))
        WStubBank.LV = 4
        try:
            ref("training_sets_other_label_version", "label versions", ("--hn-bank", good_bank))
        finally:
            WStubBank.LV = 5
        ref("cache_other_trunk", "cache trunk", ("--ctx-cache", r7.build_cache("trunk2", perturb_trunk=True)))
        ref("cache_lacks_a_scene", "holds no context", ("--ctx-cache", r7.build_cache("drop", drop_log="T3")))
    return bad


# ------------------------------------------------------------------------------------------------ the 7 mutations of PREREG section 4.3 (each edits SOURCE and must turn the NAMED checks RED)
def mutations(SF: str, HNS: str, CCS: str):
    """-> [(id, what, srcs, groups, expected-RED check names)]"""
    return [
        ("M1", "the hn term is computed but never added to arm B's loss (`lB = lB + hn_out.loss` removed)",
         {"sf": once(SF, "            lB = lB + hn_out.loss\n", "            pass\n"), "hn": None, "cc": None}, ("ON",),
         ["ON_bank_moves_arm_B", "G14_value_on_positive", "G14_grad_q_mlp_nonzero", "G14_grad_dec_nonzero", "G14_grad_head_rows_0_5_nonzero"]),
        ("M2", "the term is applied to arm A's scorer (`HN.hn_term(armB,` -> `HN.hn_term(model,`)",
         {"sf": once(SF, "hn_out = HN.hn_term(armB,", "hn_out = HN.hn_term(model,"), "hn": None, "cc": None}, ("ON",),
         ["ON_bank_leaves_arm_A_bit_identical", "G14_armA_exactly_zero"]),
        ("M3", "pad slots are served as valid (the bank's mask replaced by all-True)",
         {"sf": once(SF, 'b["hn_m"][0]', 'torch.ones_like(b["hn_m"][0])'), "hn": None, "cc": None}, ("ON",),
         ["G14_pad_mask_equals_bank", "G14_d1_rows", "ON_D1_train_row_counts"]),
        ("M4", "the fan rows can see the hn slots (hn_attn_mask no longer blocks fan -> hn)",
         {"sf": SF, "hn": once(HNS, "blocked = (j >= K) & (j != i)\n", "blocked = (j >= K) & (j != i) & (i >= K)\n"), "cc": None}, ("ON",),
         ["G14_fan_rows_isolated_from_hn"]),
        ("M5", "the hn rows train the LANE output (a loss term on logits[..., 6:])",
         {"sf": SF, "hn": once(HNS, 'logits[..., :N_COMP], tg[..., :N_COMP], reduction="none")', 'logits[..., :N_COMP], tg[..., :N_COMP], reduction="none") + 1e-3 * logits[..., N_COMP:].sum(-1, keepdim=True)'),
          "cc": None}, ("ON",),
         ["G14_lane_row_exactly_zero"]),
        ("M6", "the bank's partition / label_version / fan-fingerprint identity is not checked (`expect=hn_expect` removed)",
         {"sf": once(SF, "expect=hn_expect", "expect=None"), "hn": None, "cc": None}, ("REF",),
         ["REF_bank_other_partition", "REF_bank_other_label_version", "REF_bank_other_fan_fingerprint"]),
        ("M7", "the cache's trunk fingerprint is not checked against the checkpoint's (`expect_trunk_fingerprint=trunk_fp` removed)",
         {"sf": once(SF, "expect_trunk_fingerprint=trunk_fp", "expect_trunk_fingerprint=None"), "hn": None, "cc": None}, ("REF",),
         ["REF_cache_other_trunk"]),
    ]


def main() -> int:
    SF, HNS, CCS = read_text(SRC_PATH), read_text(HN_PATH), read_text(CC_PATH)
    HNW["partition"] = load_mod(patch_src(SF), "sft_wire_probe", SRC_PATH).HN_PARTITION_ID
    n_fail = 0
    with tempfile.TemporaryDirectory(prefix="hnwire_") as td:
        rigs = {"plain": Rig(td, False), "n7": Rig(td, True)}
        bad = suite(rigs, {"sf": SF, "hn": None, "cc": None}, ("PIN", "I3", "ON", "REF"), tag="base")
        print("-- checks:", "ALL GREEN" if not bad else f"{len(bad)} RED: {bad}")
        n_fail += len(bad)
        print("-- mutation table (each must be RED on the named checks)")
        for mid, what, srcs, groups, expect in mutations(SF, HNS, CCS):
            red = suite(rigs, srcs, groups, tag=mid)
            hit = [e for e in expect if e in red]
            ok = bool(red) and len(hit) == len(expect)
            print(f"   {mid} {'RED as required' if ok else 'NOT RED'}  {what}\n      named: {expect}\n      red:   {red}")
            if not ok:
                n_fail += 1
    if n_fail:
        print(f"ZZTEST_FAIL {n_fail}")
        return 1
    print("ZZTEST_PASS", 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
