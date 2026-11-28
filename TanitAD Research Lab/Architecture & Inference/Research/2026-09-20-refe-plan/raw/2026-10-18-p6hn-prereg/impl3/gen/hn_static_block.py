# ====================================================================================================== P6-HN-S: the refe_hnscreen profile
# PREREG_P6HN_SCREEN 4.4: the profile is refe_rls1 + its own arm table + the pinned argv; DVB_KINDS_SCORER classifies --hn-bank / --ctx-cache `data`; the data event carries the `hn` and `ctx_cache` BUILT
# blocks; the preflight adds G14 / G15, re-judged from raw fields by judge_g14 / judge_g15; and TWELVE deliberate-regression arms must each FAIL.  Every refusal has its positive twin (the clean argv).
HN_ARM_FRAGMENTS = {                       # arm -> (where it acts, a fragment its failure must name)
    "lever_ignored": ("dvb", "--hn-bank"), "hn_to_armA": ("pf", "arm A received a gradient"), "hn_in_selection": ("pf", "held-out items carry a hard-negative tail"),
    "hn_lane_target": ("pf", "trained the LANE output"), "unmasked_pad": ("pf", "junk written into the pad slots"), "fan_sees_hn": ("pf", "the fan can see a hard negative"),
    "stale_bank": ("dvb", "hn.sha256"), "empty_bank_in_launch": ("dvb", "hn.rows > 0"), "moved_lever": ("argv", "MOVED"), "stale_cache": ("pf", "is not the checkpoint's"),
    "shuffled_cache": ("pf", "differs from the live trunk"), "epo_with_hn": ("argv", "--hn-bank with --epo")}


def hn_g14(**mut) -> dict:
    """a CLEAN G14 in the REAL shape (scorer_finetune.g14_hn_live_and_identity), numbers of the order the CPU rig measured; `mut` replaces top-level keys"""
    z = {"value": 0.0, "B_all": 0.0, "A_all": 0.0}
    g = {"mutate": [], "scorer": list(C.SCORER_GROUPS), "lane_head": True, "n_hn": 9,
         "grad_abs_sum": {"score_q_mlp": 1.96, "score_dec": 9.4, "score_head_rows_0_5": 2.18, "score_head_row_6_lane": 0.0, "armA_all": 0.0, "base_params_with_grad": 0, "frozen_params_with_grad": 0},
         "grads_finite": True, "A_bce_control": {m: 2.0 for m in C.SCORER_GROUPS}, "value_on": 0.2067, "controls": {"off": dict(z), "empty": dict(z)}, "trunk_requires_grad": False,
         "two_step": {"off_equal_A": True, "empty_equal_A": True, "on_differs_from_A": True, "probe_reads_one_ulp": True},
         "d1": {"rows_in_term": 9, "rows_expected": 9, "batch": 8, "K": 64}, "pad": {"served_mask_equals_bank": True, "pad_slots": 55, "perturbed_loss_equal": True},
         "fan_call_max_abs_diff": 0.0, "fan_rows_hn_isolation_max_abs": 0.0, "fan_rows_vs_one_call_max_abs": 1.8e-07, "fan_rows_scale": 1.25, "hn_row_independence_max_abs": 0.0,
         "dup_identity": {"loss": 0.0932, "share": 0.0932, "abs_diff": 1.5e-08, "j": 3}, "eval_reads_bank": {"held_items_checked": 3, "held_items_with_hn_tail": 0}, "ok": True}
    g.update(mut)
    return g


def hn_g15(manifest_sha: str, **mut) -> dict:
    g = {"n_train": 64, "n_val": 64, "micro_batch": 8, "ctx_max_abs_diff": 0.0, "logit_max_abs_diff": {"base": 0.0, "A": 0.0, "B": 0.0},
         "shuffled": {"ctx_max_abs_diff": 0.953, "logit_max_abs_diff": {"base": 0.092, "A": 0.092, "B": 0.092}},
         "trunk_fingerprint_manifest": "a" * 64, "trunk_fingerprint_checkpoint": "a" * 64, "dtype": "bf16", "n_scenes_manifest": 19000, "manifest_sha256": manifest_sha,
         "covers_run": {"missing_train": 0, "missing_val": 0}, "calib_sha256_manifest": "b" * 64, "calib_sha256_run": "b" * 64, "mutate": [], "ok": True}
    g.update(mut)
    return g


def hnscreen_static() -> None:
    import hashlib
    import scorer_finetune as S
    P = L.PROFILES["refe_hnscreen"]
    td = Path(tempfile.mkdtemp())
    bank, cache = td / "bank.hnb", td / "cache"
    bank.write_bytes(b"HNBANK1\n" + bytes(range(200)))
    cache.mkdir()
    (cache / "manifest.json").write_text('{"tool": "ctx_cache/1"}', encoding="utf-8")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()                  # noqa: E731
    arms_json = json.loads((PKG / "eval" / "rls1b_arms.json").read_text(encoding="utf-8"))
    c1 = list(arms_json["arms"]["C1"]["argv"])

    def argv_for(seed):
        return C._set(c1, "--seed", str(seed)) + ["--hn-bank", str(bank), "--ctx-cache", str(cache)]
    a1 = argv_for(1)
    ns1, err = C.parse_scorer_argv(S.build_parser(), a1)
    ok(err is None, err)

    def refs(argv, arm="S1", prof=None):
        return L.profile_refusals({**(prof or P), "declared_arm": arm}, argv)

    def clean(argv, arm="S1", prof=None):
        rs = refs(argv, arm, prof)
        ok(rs == [], (arm, rs))

    def red(argv, frag, arm="S1", prof=None):
        rs = refs(argv, arm, prof)
        ok(any(frag in r for r in rs), (frag, arm, rs))
    # ------------------------------------------------------------------ H1 the profile: refe_rls1's runner and checks, its OWN arm table, the screen's forbidden set, the registry
    R = L.PROFILES["refe_rls1"]
    ok(P["name"] == "refe_hnscreen" and P["runner"] == "scorer" and P["parser_module"] == "scorer_finetune" and P["trainer_src"] == "scorer_finetune.py" and P["hn_rules"] is True)
    ok(all(P[k] == R[k] for k in ("checks", "na_ok", "required_flags", "data_flags", "live_terms", "output_flags", "namespace_funcs")) and P["mintable"] and P["require_commit"] and not P["rehearsal"])
    ok(P["arm_class"] == {"S0": "C", "S1": "C", "S2": "C"} and P["arm_seeds"] == {"S0": 0, "S1": 1, "S2": 2} and set(P["arms_need_pi"]) == {"S3", "S4"} and P["schedule_reach"] is True)
    ok(all(f in P["data_flags"] for f in L.HNSCREEN_FLAGS) and not any(f in P["forbidden_flags"] for f in L.HNSCREEN_FLAGS) and set(P["forbidden_flags"]) == set(L.RLS1_FORBIDDEN) - set(L.HNSCREEN_FLAGS))
    for nm in ("refe_rls1", "refe_rls1b", "refe_rls1c", "refe_rls1d", "refe_aux_s1", "refe_aux_s1c", "refe_sft4r"):          # every other scorer profile fingerprints the inputs (registry) and REFUSES them
        ok(all(f in L.PROFILES[nm]["data_flags"] for f in L.HNSCREEN_FLAGS), nm)
        if nm != "refe_sft4r":
            ok(all(f in L.PROFILES[nm]["forbidden_flags"] for f in L.HNSCREEN_FLAGS), nm)
    ok(C.DVB_KINDS_SCORER["hn_bank"] == "data" and C.DVB_KINDS_SCORER["ctx_cache"] == "data" and set(C.DVB_KINDS_SCORER) == {x.dest for x in S.build_parser()._actions if x.dest != "help"})
    ok(C.SCORER_GATES["G14"] == "hn_live_and_identity" and C.SCORER_GATES["G15"] == "ctx_cache_identity")
    ok(C.SCORER_HN_KEYS == ("sha256", "rows", "scenes_with_hn", "max_per_sample", "arm", "comps", "weight", "mask") and C.SCORER_CTX_KEYS == ("manifest_sha256", "trunk_fingerprint", "n_scenes", "dtype"))
    for e in L.HNSCREEN_SUITE:
        ok((HERE / e["script"]).is_file() and e["token"] in (HERE / e["script"]).read_text(encoding="utf-8"), e)
    ok(L.HNSCREEN_SUITE[:len(L.RLS1_SUITE)] == L.RLS1_SUITE)                                         # the existing refe_rls1 suite entries stay (0 regressions)
    ok(set(HN_ARM_FRAGMENTS) == set(C.HN_ARMS) and len(C.HN_ARMS) == 12)                                # exactly the twelve arms of PREREG 4.4, each tested below
    ok(C.scorer_required_gates(ns1) == ["G1", "G2", "G3", "G4", "G5", "G7", "G9", "G10", "G14", "G15"])   # G14 iff --hn-bank, G15 iff --ctx-cache
    ns_only_bank, _ = C.parse_scorer_argv(S.build_parser(), a1[:-2])
    ok("G14" in C.scorer_required_gates(ns_only_bank) and "G15" not in C.scorer_required_gates(ns_only_bank))
    # ------------------------------------------------------------------ H2 the clean argvs (the three passes) and the arm table
    for s_ in (0, 1, 2):
        clean(argv_for(s_), f"S{s_}")
    red(argv_for(1), "registered with --seed 0", "S0")                                                # naming an arm asserts its seed
    red(argv_for(0), "registered with --seed 1", "S1")
    for arm in ("S3", "S4"):
        try:
            L._profile_for(SimpleNamespace(profile="refe_hnscreen", arm_name=arm), True)
            ok(False, arm)
        except SystemExit as e:
            ok("rule E" in str(e), str(e))
    try:
        L._profile_for(SimpleNamespace(profile="refe_hnscreen", arm_name=None), True)
        ok(False, "an arm-less mint")
    except SystemExit as e:
        ok("mints for ONE named arm" in str(e), str(e))
    ok(L._profile_for(SimpleNamespace(profile="refe_hnscreen", arm_name="S2"), True)["declared_arm"] == "S2")
    # ------------------------------------------------------------------ H3 any moved / added / dropped / doubled token FAILS (each with its twin above)
    for f, nv in [(f, len(v)) for f, v in L.HNSCREEN_PIN.items()] + [(f, 1) for f in L.HNSCREEN_PATHS + ("--seed",)]:
        red(C._drop(a1, f, nv), f"{f} is DROPPED")
        red(a1 + [f] + (["1"] * nv), "times (each token exactly once)") if nv else red(a1 + [f], "times (each token exactly once)")
    for f, v in (("--lr", "2e-4"), ("--epochs", "2"), ("--max-updates", "500"), ("--workers", "4"), ("--eval-every", "100"), ("--b-mode", "compw"), ("--lam", "0.3"), ("--backbone", "vitb16")):
        red(C._set(a1, f, v), f"{f} MOVED to {v}")
    red(L.hnscreen_regression_argv(a1, "moved_lever"), "--lr MOVED to 2e-4")
    for extra in (["--warmup", "50"], ["--batch", "8"], ["--lane-weight", "1.0"], ["--tau-t", "0.3"], ["--seed-extra", "1"]):
        rs = refs(a1 + extra)
        ok(any("ADDS" in r or "FORBIDS" in r or "rejects" in r for r in rs), (extra, rs))
    red(a1 + ["--preflight"], "FORBIDS")                                                             # the generic forbidden set still applies
    clean(C._set(a1, "--lr", "0.0001"))                                                                # the same VALUE spelled differently is not a moved token
    clean(C._set(C._set(a1, "--out", "/other/out"), "--dump-scores", "/other/scores"))                 # the allowed differences: --out, --dump-scores, re-rooted paths
    clean(C._set(a1, "--ckpt", "/re/rooted/model.pt"))
    # the trainer's OWN refusals are surfaced, never re-derived
    red(L.hnscreen_regression_argv(a1, "epo_with_hn"), "--hn-bank with --epo")
    red(L.hnscreen_regression_argv(a1, "epo_with_hn"), "must NOT carry")
    red(C._set(a1, "--lam", "0.5"), "--hn-bank needs --lam 0")
    red(a1 + ["--aux-drv"], "--hn-bank with --aux-drv")
    red(a1 + ["--aux-drv"], "FORBIDS")
    red(C._set(a1, "--b-mode", "compw"), "--hn-bank needs --b-mode listnet")
    # the other profiles refuse the flags (positive twin: their clean argvs are untouched, S2 of rls1_static)
    red(a1, "refe_rls1 FORBIDS", "C1", R)
    red(a1, "refe_aux_s1 FORBIDS", "N-s0", L.PROFILES["refe_aux_s1"])
    clean(L.rls1_control_argv(C._set(c1, "--seed", "1")), "C1", R)                                      # RL-S1's C1 argv is still clean under refe_rls1
    # content fingerprints of the re-rooted data (the optional registered hook)
    fk = L.GT.fp_key(L.GT.fingerprint(str(bank)))
    clean(a1, "S1", {**P, "fingerprints_expected": {"--hn-bank": fk}})
    red(a1, "content fingerprint of --hn-bank", "S1", {**P, "fingerprints_expected": {"--hn-bank": "0" * 64}})
    # the CLI
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = L.main(["refusals", "--profile", "refe_hnscreen", "--arm-name", "S1", "--args", " ".join(a1)])
    ok(rc == 0 and "ZZGATE-CLEANZZ" in buf.getvalue(), buf.getvalue())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = L.main(["refusals", "--profile", "refe_hnscreen", "--arm-name", "S1", "--args", " ".join(L.hnscreen_regression_argv(a1, "moved_lever"))])
    ok(rc == 5 and "ZZGATE-REFUSEDZZ" in buf.getvalue(), buf.getvalue())
    # ------------------------------------------------------------------ H4 G-DVB: the built blocks against the argv and the bytes it names
    k14, k15 = (next(k for k in rls1_gate_keys() if k.startswith(p)) for p in ("G14_", "G15_"))
    man_sha = sha(cache / "manifest.json")

    def data_ev(**mut):
        d = {"train_sets": 16000, "heldout_sets": 3000,
             "hn": {"sha256": sha(bank), "rows": 24000, "scenes_with_hn": 9000, "max_per_sample": 8, "arm": "B", "comps": 6, "weight": "1/(B*64*6) per element", "mask": "hn->fan+self"},
             "ctx_cache": {"manifest_sha256": man_sha, "trunk_fingerprint": "a" * 64, "n_scenes": 19000, "dtype": "bf16"}}
        d.update(mut)
        return d

    def pf_of(argv, *, g14=None, g15=None, drop=(), data=None, **kw):
        rows = rls1_pf_rows(argv, drop=drop, add={k14: g14 if g14 is not None else hn_g14(), k15: g15 if g15 is not None else hn_g15(man_sha)}, data_mut=data if data is not None else data_ev(), **kw)
        return C.load_scorer_preflight(rls1_write(Path(tempfile.mkdtemp()) / "preflight.jsonl", rows))

    def dvb(argv=None, pf=None, arm=None, prof=None):
        return C.check_dvb_scorer(argv or a1, pf or pf_of(argv or a1), {**(prof or P), "declared_arm": "S1"}, arm=arm)
    ev = dvb()
    rows_ = ev["details"]["rows"]
    ok(ev["status"] == "PASS" and ev["details"]["unknown_flags"] == [] and ev["details"]["stale_registry"] == [], ev["reasons"])
    ok(sum(1 for r in rows_ if r["dest"] == "hn_bank") == 10 and sum(1 for r in rows_ if r["dest"] == "ctx_cache") == 6, [(r["dest"], r["ok"]) for r in rows_ if r["dest"] in ("hn_bank", "ctx_cache")])
    ok(any(r["flag"] == "--max-updates" and r["declared"] == 1000 == r["built"] for r in rows_))          # schedule_reach: ceil(16000 / 8 x 4 / 8) = 1000 (PREREG 4.4's G-CLOCK)
    ok(C.check_dvb_scorer(L.rls1_control_argv(C._set(c1, "--seed", "1")), C.load_scorer_preflight(rls1_write(Path(tempfile.mkdtemp()) / "p.jsonl", rls1_pf_rows(L.rls1_control_argv(C._set(c1, "--seed", "1"))))),
                          {**R, "declared_arm": "C1"})["status"] == "PASS")                                   # no hn flags, no hn rows: refe_rls1's own G-DVB is unchanged
    for what, bad_data, frag in (
            ("block absent with the flag named", {k: v for k, v in data_ev().items() if k != "hn"}, "declared True != built False"),
            ("cache block absent with the flag named", {k: v for k, v in data_ev().items() if k != "ctx_cache"}, "declared True != built False"),
            ("another arm built", data_ev(hn=dict(data_ev()["hn"], arm="A")), "hn.arm"),
            ("another weight", data_ev(hn=dict(data_ev()["hn"], weight="1/(B*6) per element")), "hn.weight"),
            ("another mask", data_ev(hn=dict(data_ev()["hn"], mask="hn->all")), "hn.mask"),
            ("another comps", data_ev(hn=dict(data_ev()["hn"], comps=7)), "hn.comps"),
            ("a key missing", data_ev(hn={k: v for k, v in data_ev()["hn"].items() if k != "mask"}), "the `hn` block's keys"),
            ("more scenes than trained on", data_ev(hn=dict(data_ev()["hn"], scenes_with_hn=16001)), "no more scenes / rows"),
            ("the bank's bytes are not the file's", data_ev(hn=dict(data_ev()["hn"], sha256="1" * 64)), "hn.sha256"),
            ("the manifest is not the cache's", data_ev(ctx_cache=dict(data_ev()["ctx_cache"], manifest_sha256="2" * 64)), "ctx_cache.manifest_sha256"),
            ("a cache that is not bf16", data_ev(ctx_cache=dict(data_ev()["ctx_cache"], dtype="fp16")), "ctx_cache.dtype"),
            ("a cache of too few scenes", data_ev(ctx_cache=dict(data_ev()["ctx_cache"], n_scenes=18999)), "at least every train + held-out scene"),
            ("a trunk fingerprint that is not a sha256", data_ev(ctx_cache=dict(data_ev()["ctx_cache"], trunk_fingerprint="abc")), "trunk_fingerprint")):
        ev = dvb(pf=pf_of(a1, data=bad_data))
        ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]), (what, ev["reasons"]))
    ev = dvb(pf=pf_of(a1, data=data_ev(train_sets=100)))                                                  # schedule_reach: too few training sets ends the run before 1,000 updates
    ok(ev["status"] == "FAIL" and "--max-updates" in " ".join(ev["reasons"]))
    c1_argv = L.rls1_control_argv(C._set(c1, "--seed", "1"))                                              # a block in the data event WITHOUT the flag: the flag was dropped, the trainer built it anyway
    ev = C.check_dvb_scorer(c1_argv, C.load_scorer_preflight(rls1_write(Path(tempfile.mkdtemp()) / "p.jsonl", rls1_pf_rows(c1_argv, data_mut=data_ev()))), {**R, "declared_arm": "C1"})
    ok(ev["status"] == "FAIL" and "iff --hn-bank" in " ".join(ev["reasons"]), ev["reasons"])
    # the bank / cache file the argv names is missing or changed after the evidence was minted
    ev = dvb(argv=C._set(a1, "--hn-bank", str(td / "nope.hnb")))
    ok(ev["status"] == "FAIL" and "not readable" in " ".join(ev["reasons"]), ev["reasons"])
    (cache / "manifest.json").write_text('{"tool": "ctx_cache/1", "n": 1}', encoding="utf-8")            # a stale cache: the manifest on disk is not the one the evidence was built from
    ev = dvb()
    ok(ev["status"] == "FAIL" and "ctx_cache.manifest_sha256" in " ".join(ev["reasons"]), ev["reasons"])
    (cache / "manifest.json").write_text('{"tool": "ctx_cache/1"}', encoding="utf-8")
    ok(dvb()["status"] == "PASS")
    # ------------------------------------------------------------------ H5 G-PREFLIGHT: G14 / G15 are REQUIRED by the argv and re-judged from raw fields; the 12 regression arms
    def pre(argv=None, pf_kw=None, arm=None):
        argv = argv or a1
        rows = rls1_pf_rows(argv, add={k14: (pf_kw or {}).get("g14", hn_g14()), k15: (pf_kw or {}).get("g15", hn_g15(man_sha))}, data_mut=data_ev(), **{k: v for k, v in (pf_kw or {}).items() if k not in ("g14", "g15")})
        p = rls1_write(Path(tempfile.mkdtemp()) / "preflight.jsonl", rows)
        return L.consume_scorer_preflight(p, argv, {**P, "declared_arm": "S1"}, arm=arm)
    ev = pre()
    ok(ev["status"] == "PASS" and ev["details"]["required"][-2:] == ["G14", "G15"], ev["reasons"])
    for g_, frag in (("G14", "required preflight gate(s) ABSENT: ['G14']"), ("G15", "required preflight gate(s) ABSENT: ['G15']")):
        ev = pre(pf_kw={"drop": (g_,)})
        ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]), (g_, ev["reasons"]))
    ev = pre(pf_kw={"verdict": "FAIL"})
    ok(ev["status"] == "FAIL")
    ev = pre(argv=L.rls1_control_argv(C._set(c1, "--seed", "1")), pf_kw={})                                # a preflight WITH G14 / G15 for an argv without the flags: another configuration
    ok(ev["status"] == "FAIL" and "do not oblige" in " ".join(ev["reasons"]) or "do not oblige" in " ".join(ev["reasons"]) or ev["status"] == "FAIL", ev["reasons"])
    for arm, (where, frag) in HN_ARM_FRAGMENTS.items():
        if where == "pf":
            ev = pre(arm=arm)
            ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]), (arm, ev["reasons"]))
        elif where == "dvb":
            ev = dvb(arm=arm)
            ok(ev["status"] == "FAIL" and frag in " ".join(ev["reasons"]), (arm, ev["reasons"]))
        else:
            rs = L.profile_refusals({**P, "declared_arm": "S1"}, L.hnscreen_regression_argv(a1, arm))
            ok(any(frag in r for r in rs), (arm, rs))
    ok(pre()["status"] == "PASS" and dvb()["status"] == "PASS")                                           # the arms left no residue
    for arm in C.HN_ARMS:                                                                                 # an arm never mutates the clean evidence in place
        g14_0, g15_0 = hn_g14(), hn_g15(man_sha)
        a14, a15 = C.apply_hn_arm(g14_0, g15_0, arm)
        ok(g14_0 == hn_g14() and g15_0 == hn_g15(man_sha))
        if arm in ("lever_ignored", "stale_bank", "empty_bank_in_launch", "moved_lever", "epo_with_hn"):
            ok(a14 == g14_0 and a15 == g15_0, arm)                                                       # those arms act elsewhere
        else:
            ok(a14 != g14_0 or a15 != g15_0, arm)
    # the trainer's `ok` is a claim, the raw fields are the evidence: ok True with a defect in the fields still FAILS
    ev = pre(pf_kw={"g14": hn_g14(grad_abs_sum={**hn_g14()["grad_abs_sum"], "score_head_row_6_lane": 0.5})})
    ok(ev["status"] == "FAIL" and "LANE output" in " ".join(ev["reasons"]))
    ev = pre(pf_kw={"g15": hn_g15(man_sha, ctx_max_abs_diff=1e-3)})
    ok(ev["status"] == "FAIL" and "bit-identical" in " ".join(ev["reasons"]))
    # G15 is bound to the data event's built block
    ev = pre(pf_kw={"g15": hn_g15(man_sha, trunk_fingerprint_checkpoint="c" * 64, trunk_fingerprint_manifest="c" * 64)})
    ok(ev["status"] == "FAIL" and "data event's ctx_cache.trunk_fingerprint" in " ".join(ev["reasons"]))
    ev = pre(pf_kw={"g15": hn_g15("3" * 64)})
    ok(ev["status"] == "FAIL" and "manifest_sha256" in " ".join(ev["reasons"]))
    # ------------------------------------------------------------------ H6 the judges, field by field (each corruption alone must turn it RED; the clean one is [])
    ok(C.judge_g14(hn_g14(), ns1) == [] and C.judge_g15(hn_g15(man_sha), ns1) == [])
    ok(C.judge_g14(None, ns1) and C.judge_g15([], ns1))
    g = hn_g14()
    gs = g["grad_abs_sum"]
    for what, mut, frag in (
            ("ok false", dict(ok=False), "ok != true"), ("regression switch", dict(mutate=["x"]), "regression switch"), ("no hn rows", dict(n_hn=0), "valid hard-negative rows"),
            ("non-finite", dict(grads_finite=False), "not finite"), ("value zero", dict(value_on=0.0), "not part of arm B's loss"),
            ("q_mlp zero", dict(grad_abs_sum={**gs, "score_q_mlp": 0.0}), "score_q_mlp"), ("dec zero", dict(grad_abs_sum={**gs, "score_dec": 0.0}), "score_dec"),
            ("head rows 0-5 zero", dict(grad_abs_sum={**gs, "score_head_rows_0_5": 0.0}), "score_head_rows_0_5"),
            ("lane probed on another model", dict(lane_head=False), "lane-row control"), ("lane row non-zero", dict(grad_abs_sum={**gs, "score_head_row_6_lane": 1e-9}), "LANE output"),
            ("arm A gradient", dict(grad_abs_sum={**gs, "armA_all": 1e-9}), "arm A received"), ("base on the path", dict(grad_abs_sum={**gs, "base_params_with_grad": 3}), "on the gradient path"),
            ("trunk trainable", dict(trunk_requires_grad=True), "on the gradient path"), ("A's own BCE control zero", dict(A_bce_control={m: 0.0 for m in C.SCORER_GROUPS}), "arm A's own BCE gradient"),
            ("off control non-zero", dict(controls={"off": {"value": 1e-9, "B_all": 0.0, "A_all": 0.0}, "empty": g["controls"]["empty"]}), "controls do not read EXACTLY 0.0"),
            ("empty control missing", dict(controls={"off": g["controls"]["off"]}), "controls do not read EXACTLY 0.0"),
            ("B != A with the bank off", dict(two_step={**g["two_step"], "off_equal_A": False}), "two_step.off_equal_A"),
            ("B != A with the bank empty", dict(two_step={**g["two_step"], "empty_equal_A": False}), "two_step.empty_equal_A"),
            ("the bank does nothing", dict(two_step={**g["two_step"], "on_differs_from_A": False}), "two_step.on_differs_from_A"),
            ("the equality probe is blind", dict(two_step={**g["two_step"], "probe_reads_one_ulp": False}), "two_step.probe_reads_one_ulp"),
            ("duplicate identity", dict(dup_identity={"loss": 0.2, "share": 0.1, "abs_diff": 0.1, "j": 3}), "analytic identity"),
            ("no pad slot", dict(pad={"served_mask_equals_bank": True, "pad_slots": 0, "perturbed_loss_equal": True}), "no pad slot"),
            ("pad moves the term", dict(pad={"served_mask_equals_bank": True, "pad_slots": 55, "perturbed_loss_equal": False}), "pad slots"),
            ("fan call differs", dict(fan_call_max_abs_diff=1e-7), "one-call path"), ("fan sees hn", dict(fan_rows_hn_isolation_max_abs=1e-7), "can see a hard negative"),
            ("hn rows depend on each other", dict(hn_row_independence_max_abs=1e-7), "OTHER hard-negative rows"), ("no scene with two rows", dict(hn_row_independence_max_abs=None), "OTHER hard-negative rows"),
            ("D1 rows differ", dict(d1={"rows_in_term": 10, "rows_expected": 9, "batch": 8, "K": 64}), "(D1)"), ("D1 nothing expected", dict(d1={"rows_in_term": 0, "rows_expected": 0}), "(D1)"),
            ("eval reads the bank", dict(eval_reads_bank={"held_items_checked": 3, "held_items_with_hn_tail": 1}), "(I4)"), ("no eval item checked", dict(eval_reads_bank={"held_items_checked": 0, "held_items_with_hn_tail": 0}), "(I4)")):
        rs = C.judge_g14(hn_g14(**mut), ns1)
        ok(any(frag in r for r in rs), (what, rs))
    for what, mut, frag in (
            ("ok false", dict(ok=False), "ok != true"), ("regression switch", dict(mutate=["x"]), "regression switch"), ("too few train scenes", dict(n_train=63), "needs >= 64"),
            ("too few val scenes", dict(n_val=10), "needs >= 64"), ("micro-batch is not the run's", dict(micro_batch=2), "--batch"), ("ctx differs", dict(ctx_max_abs_diff=2 ** -9), "bit-identical"),
            ("A logits differ", dict(logit_max_abs_diff={"base": 0.0, "A": 1e-9, "B": 0.0}), "logits read from the cache"), ("B logits missing", dict(logit_max_abs_diff={"base": 0.0, "A": 0.0}), "logits read from the cache"),
            ("the shuffle probe is blind", dict(shuffled={"ctx_max_abs_diff": 0.0, "logit_max_abs_diff": {"base": 0.0, "A": 0.0, "B": 0.0}}), "row-shuffled cache"),
            ("the shuffle reads < 10x", dict(logit_max_abs_diff={"base": 0.05, "A": 0.0, "B": 0.0}), "logits read from the cache"),
            ("another trunk", dict(trunk_fingerprint_manifest="d" * 64), "not the checkpoint's"), ("a cache that misses scenes", dict(covers_run={"missing_train": 1, "missing_val": 0}), "does not cover"),
            ("fp16 cache", dict(dtype="fp16"), "dtype"), ("calib mismatch", dict(calib_sha256_manifest="e" * 64), "calib sha256"), ("no calib", dict(calib_sha256_run=None, calib_sha256_manifest=None), "calib sha256"),
            ("no manifest sha", dict(manifest_sha256=None), "manifest's sha256 is absent")):
        rs = C.judge_g15(hn_g15(man_sha, **mut), ns1)
        ok(any(frag in r for r in rs), (what, rs))
    ok(C.judge_g15(hn_g15(man_sha, n_train=24, n_val=12, micro_batch=8), ns1, min_scenes=12) == [])      # a side that holds fewer scenes than 64 compares them all: the floor is a parameter, named
    # ------------------------------------------------------------------ H7 the judges against REAL evidence: the preflight raw fields of the CPU rig's trainer (test_hn_screen_wiring.py), clean and with
    # each of its mutations (M1 term never added, M2 term on arm A, M3 pad served, M4 fan sees hn, M5 lane trained).  Skipped loudly (not counted) when the fixture is absent.
    fx = HERE / "fixtures" / "hn_wiring_evidence.json"
    if fx.is_file():
        evd = json.loads(fx.read_text(encoding="utf-8"))
        ns2 = SimpleNamespace(**{**vars(ns1), "batch": 2})                                              # the rig trains in micro-batches of 2
        ok(C.judge_g14(evd["base"]["g14"], ns2) == [] and C.judge_g15(evd["base"]["g15"], ns2, min_scenes=12) == [], (C.judge_g14(evd["base"]["g14"], ns2), C.judge_g15(evd["base"]["g15"], ns2, min_scenes=12)))
        for mid, frag in (("M1", "not part of arm B's loss"), ("M2", "arm A received"), ("M3", "served mask"), ("M4", "can see a hard negative"), ("M5", "LANE output")):
            rs = C.judge_g14(evd[mid]["g14"], ns2)
            ok(any(frag in r for r in rs), (mid, rs))
    else:
        print("ZZTEST_SKIP hn_wiring_evidence.json absent (not counted)")
