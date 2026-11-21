# ---------------------------------------------------------------------------------------------------------------- the bank (section 1, 2, 6; Amendment 1 section 3)
# The bank is what `stack/scripts/v8f_dagger_bank.py` WRITES (reconciled with the built generator and its smoke, A1-era; schema "v8f-dagger-bank-v1"). One bank directory:
#   PROVENANCE.json            {schema, kind "bank", bank_sha256, manifest "bank.json", shard_provenance [..], n_records, coverage{n_windows, n_eligible, fraction, min, blocked},
#                               reasons_by_tick{"1","2": {ok, off_lat, off_head, end_edge, nonfinite}}, partial, warnings}          <- the manifest's bank.{s}.report
#   bank.json                  {schema, n_shards, shards[{file, k, n_records, file_sha256, content_sha256}], common{ckpt_sha256, rollout, constants, thresholds_file_sha256, cfg{hmax},
#                               train{n_clips, n_windows, train_sha256}, window_list_sha256, eval{n_clips, clips, overlap}, partial, limit_clips, ..}, partial, n_records, coverage, reasons_by_tick}
#                               its sha256 IS the bank's sha256
#   shard_KKKofNNN.provenance.json   per shard: {plan{shard [k, n], window_list_sha256, ckpt_sha256 (None in gen mode), ..}, inputs{ckpt{sha256}}, kb0_tick0_identity{checked, mismatch},
#                               kb1_replay{chunks, windows, mismatch_chunks, frac, seed}, kb3_eval_disjoint{n_eval_clips, overlap}, coverage, reasons_by_tick, file_sha256, ..}
#   shard_KKKofNNN.npz         the records; its `header_json` member holds {"clips": {ci: clip NAME}} -- the ONLY place the bank's clips are listed by name
# NOT written by the generator (the reader REFUSES until it is; see NOTES.md "needs the bank generator to add"): PROVENANCE.json `controls` {K-B4{PASS, cases{..}}, K-B5{PASS, max_pose_now_err_m,
# max_rel_transform_err_m}} (those two are unit tests of the build, not per-bank records), and, on the TRAINER side, a TRAIN window-list sha256 (windows_sha256(J["clip"], wins)).
def sha12(c):
    return hashlib.sha256(c.encode("utf-8")).hexdigest()[:12]


def stable_sid(c):
    """v8f_mini_data.stable_episode_id (a test compares it with the tip's function)."""
    return int.from_bytes(hashlib.blake2b(c.encode("utf-8"), digest_size=8).digest(), "big") >> 1


def train_list_sha256(ids):
    """The trainer's own TRAIN CLIP-list hash (v8f_mini_train.py: sha256 of the clip ids sorted by sha12, newline-joined, recorded as `train_sha256`); a test compares it with the tip's source."""
    return hashlib.sha256(chr(10).join(sorted(ids, key=sha12)).encode("utf-8")).hexdigest()


def clip_overlap(bank_ids, eval_ids):
    """K-B3.3's disjointness, by clip id AND by stable sid (v8f_mini_data.assert_disjoint). Returns the problems found."""
    t, e = set(bank_ids), set(eval_ids)
    bad = []
    if not e:
        bad.append("the eval-141 clip-id list is empty: disjointness proves nothing")
    if t & e:
        bad.append(f"bank and eval-141 clip ids overlap: {sorted(t & e)[:3]} ({len(t & e)})")
    if {stable_sid(c) for c in t} & {stable_sid(c) for c in e}:
        bad.append("bank and eval-141 stable sids overlap")
    return bad


def train_subset_problems(bank_names, train_ids):
    """K-B3.1: every clip the bank holds is a clip of M2b's TRAIN list, matched BY NAME. The bank may hold FEWER (Amendment 1 section 3: equality is not required)."""
    t, bad = set(train_ids), []
    if not t:
        bad.append("the TRAIN clip list is empty: the subset proves nothing")
    if not bank_names:
        bad.append("the bank holds no clips")
    out = sorted(set(bank_names) - t)
    if out:
        bad.append(f"{len(out)} bank clip(s) are not in M2b's TRAIN clip list, e.g. {out[:3]}")
    return bad


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def is_count(x):
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _shard_clip_names(path, what):
    """{ci: clip name} from the shard npz's `header_json` member (only that member is read)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            h = json.loads(bytes(z["header_json"]).decode("utf-8"))
        clips = h["clips"]
        if not isinstance(clips, dict) or not clips:
            raise ValueError("header_json['clips'] is not a non-empty mapping")
        return {str(k): str(v) for k, v in clips.items()}
    except Refusal:
        raise
    except Exception as e:      # noqa: BLE001
        raise Refusal(f"{what}: cannot read the clip names from {path}: {type(e).__name__}: {e}")


def load_bank(report_path, bank_file=None, what="bank report"):
    """Reads one bank directory as the generator wrote it. Raises Refusal only when a file is missing or unreadable; every consistency check is a CONTROL (g_bank_files / g_bank_config / ...)."""
    prov = jload(need(report_path, f"{what} (PROVENANCE.json)"))
    if not isinstance(prov, dict) or prov.get("kind") != BANK_KIND or prov.get("schema") != BANK_SCHEMA:
        raise Refusal(f"{what}: kind {prov.get('kind') if isinstance(prov, dict) else None!r} / schema {prov.get('schema') if isinstance(prov, dict) else None!r}, "
                      f"not the generator's {BANK_KIND!r} / {BANK_SCHEMA!r}")
    d = os.path.dirname(os.path.abspath(report_path))
    mp = need(bank_file or os.path.join(d, str(prov.get("manifest") or "bank.json")), f"{what}: the bank manifest (bank.json)")
    man = jload(mp)
    if not isinstance(man, dict) or not isinstance(man.get("shards"), list) or not man["shards"] or not isinstance(man.get("common"), dict):
        raise Refusal(f"{what}: bank.json has no 'shards' / 'common'")
    provs = [jload(need(os.path.join(d, nm), f"{what}: shard provenance {nm}")) for nm in (prov.get("shard_provenance") or [])]
    shards = []
    for s in man["shards"]:
        f = need(os.path.join(d, str(s.get("file"))), f"{what}: shard {s.get('file')}")
        shards.append({"file": s.get("file"), "sha256": sha256_file(f), "clips": _shard_clip_names(f, f"{what}: shard {s.get('file')}")})
    return {"dir": d, "prov": prov, "man": man, "man_sha256": sha256_file(mp), "shard_provs": provs, "shards": shards}


def common_of(B):
    return B["man"]["common"]


def bank_clip_names(B):
    return sorted({n for s in B["shards"] for n in s["clips"].values()})


def kb_record(B, kid):
    c = ((B["prov"].get("controls") or {}) if isinstance(B["prov"].get("controls"), dict) else {}).get(kid)
    if not isinstance(c, dict):
        raise Refusal(f"the bank's PROVENANCE.json carries no 'controls.{kid}' record (the generator does not write it: needs the bank generator to add it)")
    if c.get("PASS") is not True:
        raise Refusal(f"{kid} is recorded PASS = {c.get('PASS')!r}")
    return c


def g_bank_files(B):
    """The files are the ones the report names: bank.json hashes to the report's bank_sha256; every shard file to the manifest's; the shard provenance files belong to those shards; the counts add up."""
    prov, man, c, bad = B["prov"], B["man"], common_of(B), []
    if not is_hex(prov.get("bank_sha256")):
        bad.append("PROVENANCE.bank_sha256 is not a 64-hex sha256")
    if B["man_sha256"] != prov.get("bank_sha256"):
        bad.append(f"bank.json hashes to {B['man_sha256'][:16]}, not PROVENANCE.bank_sha256 {str(prov.get('bank_sha256'))[:16]}")
    ms = man["shards"]
    if man.get("n_shards") != len(ms) or len(B["shard_provs"]) != len(ms):
        bad.append(f"n_shards {man.get('n_shards')!r} / manifest shards {len(ms)} / shard provenance files {len(B['shard_provs'])} disagree")
    for m_, sf in zip(ms, B["shards"]):
        if sf["sha256"] != m_.get("file_sha256"):
            bad.append(f"shard {m_.get('file')} hashes to {sf['sha256'][:16]}, not the manifest's {str(m_.get('file_sha256'))[:16]}")
    by_k = {(sp.get("plan") or {}).get("shard", [None])[0]: sp for sp in B["shard_provs"]}
    for m_ in ms:
        sp = by_k.get(m_.get("k"))
        if sp is None:
            bad.append(f"no shard provenance for shard k={m_.get('k')}")
            continue
        if sp.get("file_sha256") != m_.get("file_sha256") or sp.get("n_records") != m_.get("n_records"):
            bad.append(f"shard k={m_.get('k')}: its provenance (file_sha256 {str(sp.get('file_sha256'))[:16]}, n_records {sp.get('n_records')}) is not the manifest's")
        if (sp.get("plan") or {}).get("window_list_sha256") != c.get("window_list_sha256"):
            bad.append(f"shard k={m_.get('k')}: plan.window_list_sha256 differs from the manifest's common.window_list_sha256")
    tot = sum(m_.get("n_records") or 0 for m_ in ms)
    if not (tot == man.get("n_records") == prov.get("n_records")):
        bad.append(f"n_records: shards sum {tot}, bank.json {man.get('n_records')}, PROVENANCE {prov.get('n_records')}")
    if not (prov.get("partial") == man.get("partial") == c.get("partial")):
        bad.append(f"partial flag differs: PROVENANCE {prov.get('partial')!r}, bank.json {man.get('partial')!r}, common {c.get('partial')!r}")
    names = [n for s in B["shards"] for n in s["clips"].values()]
    cis = [k for s in B["shards"] for k in s["clips"]]
    if len(set(names)) != len(names) or len(set(cis)) != len(cis):
        bad.append("a clip (or a clip index) appears in more than one shard")
    return not bad, bad or {"bank_sha256": prov.get("bank_sha256"), "n_shards": len(ms), "n_records": tot, "n_bank_clips": len(names)}


def g_bank_config(B, eval_ids):
    """The bank's configuration equals section 1 / 2: complete (not --limit-clips), the rollout, the matching constants, hmax, the TRAIN plan size, the eval list the generator checked."""
    c, bad = common_of(B), []
    cst = c.get("constants") if isinstance(c.get("constants"), dict) else {}
    if c.get("schema") != BANK_SCHEMA:
        bad.append(f"common.schema {c.get('schema')!r}, expected {BANK_SCHEMA!r}")
    if c.get("partial") is not False or c.get("limit_clips") is not None:
        bad.append(f"the bank is PARTIAL (partial {c.get('partial')!r}, limit_clips {c.get('limit_clips')!r}): a --limit-clips bank is a smoke, never the bank an arm trains on")
    if c.get("rollout") != BANK_ROLLOUT:
        bad.append(f"common.rollout {c.get('rollout')!r}, registered {BANK_ROLLOUT}")
    for k, v in MATCH_CONSTANTS.items():
        if cst.get(k) != v:
            bad.append(f"common.constants.{k} {cst.get(k)!r}, registered {v!r} (section 2)")
    if cst.get("REASONS") != list(REASONS):
        bad.append(f"common.constants.REASONS {cst.get('REASONS')!r}, expected {list(REASONS)}")
    if not str(c.get("thresholds_file_sha256") or "").lower().startswith(MATCH_CONST_SHA_PREFIX):
        bad.append(f"common.thresholds_file_sha256 {str(c.get('thresholds_file_sha256'))[:16]!r} does not start with the PREREG's {MATCH_CONST_SHA_PREFIX} (probe/THRESHOLDS_FIXED.txt, section 2)")
    if (c.get("cfg") or {}).get("hmax") != HMAX:
        bad.append(f"common.cfg.hmax {(c.get('cfg') or {}).get('hmax')!r}, registered {HMAX}")
    tr = c.get("train") if isinstance(c.get("train"), dict) else {}
    if tr.get("n_clips") != N_TRAIN_CLIPS or tr.get("n_windows") != N_TRAIN_WINDOWS:
        bad.append(f"common.train n_clips {tr.get('n_clips')!r} / n_windows {tr.get('n_windows')!r}, registered {N_TRAIN_CLIPS} / {N_TRAIN_WINDOWS}")
    for k, v in (("ckpt_sha256", c.get("ckpt_sha256")), ("window_list_sha256", c.get("window_list_sha256")), ("train.train_sha256", tr.get("train_sha256"))):
        if not is_hex(v):
            bad.append(f"common.{k} is not a 64-hex sha256")
    ev = c.get("eval") if isinstance(c.get("eval"), dict) else {}
    if not eval_ids or set(ev.get("clips") or []) != set(eval_ids) or ev.get("overlap") != 0:
        bad.append(f"the eval clips the generator checked (n {len(ev.get('clips') or [])}, overlap {ev.get('overlap')!r}) are not the eval-141 list given to the reader (n {len(set(eval_ids))})")
    return not bad, bad or {"rollout": c.get("rollout"), "thresholds_file_sha256": c.get("thresholds_file_sha256"), "train": tr, "hmax": HMAX}


def g_kb0(B):
    nw = (B["prov"].get("coverage") or {}).get("n_windows")
    recs = [sp.get("kb0_tick0_identity") for sp in B["shard_provs"]]
    if not recs or any(not (isinstance(r, dict) and is_count(r.get("checked")) and is_count(r.get("mismatch"))) for r in recs):
        raise Refusal(f"K-B0 (tick-0 identity): a shard provenance carries no kb0_tick0_identity {{checked, mismatch}}: {recs}")
    chk, mm = sum(r["checked"] for r in recs), sum(r["mismatch"] for r in recs)
    if not (chk > 0 and chk == nw and mm == 0):
        raise Refusal(f"K-B0 (tick-0 identity) must have checked == every window ({nw}) and mismatch == 0: checked {chk}, mismatch {mm}")
    return True, {"checked": chk, "n_windows": nw, "mismatch": 0, "verified_by_reader": False}


def g_kb1(B):
    nw = (B["prov"].get("coverage") or {}).get("n_windows")
    recs = [sp.get("kb1_replay") for sp in B["shard_provs"]]
    if not recs or any(not (isinstance(r, dict) and is_count(r.get("chunks")) and is_count(r.get("windows")) and is_count(r.get("mismatch_chunks")) and num(r.get("frac"))) for r in recs):
        raise Refusal(f"K-B1 (replay): a shard provenance carries no kb1_replay {{chunks, windows, mismatch_chunks, frac}}: {recs}")
    need_n = max(1, math.floor(REPLAY_FRAC * nw)) if isinstance(nw, int) else None
    w, ch, mm = sum(r["windows"] for r in recs), sum(r["chunks"] for r in recs), sum(r["mismatch_chunks"] for r in recs)
    low = [r["frac"] for r in recs if r["frac"] < REPLAY_FRAC]
    if not (need_n is not None and ch >= 1 and w >= need_n and mm == 0 and not low):
        raise Refusal(f"K-B1 (replay) must have mismatch_chunks == 0, a replay fraction >= {REPLAY_FRAC:.1%} in every shard (got {low or 'ok'}) and windows >= {need_n} "
                      f"(a seeded {REPLAY_FRAC:.1%} of {nw} rollouts): chunks {ch}, windows {w}, mismatch_chunks {mm}")
    return True, {"chunks": ch, "windows": w, "required": need_n, "mismatch_chunks": 0, "verified_by_reader": False}


def g_kb2(B, m2b_sha):
    """K-B2 (derived by the reader; the generator records the checkpoint sha256, not a PASS): the bank's generating checkpoint is the M2b checkpoint passed on the CLI."""
    g = common_of(B).get("ckpt_sha256")
    if g != m2b_sha:
        raise Refusal(f"K-B2 (on-policy): the bank was generated by checkpoint {str(g)[:16]}, not the M2b checkpoint passed on the CLI ({str(m2b_sha)[:16]})")
    odd = [((sp.get("inputs") or {}).get("ckpt") or {}).get("sha256") for sp in B["shard_provs"]]
    if any(o != g for o in odd):
        raise Refusal(f"K-B2: a shard's inputs.ckpt.sha256 {[str(o)[:12] for o in odd]} is not the manifest's common.ckpt_sha256 {g[:12]}")
    return True, {"generator_ckpt_sha256": g, "verified_by_reader": "common.ckpt_sha256 and every shard's inputs.ckpt.sha256 vs the CLI's M2b sha256"}


def g_kb31(B, train_ids):
    """K-B3.1 (Amendment 1): the bank's clips are a SUBSET of M2b's TRAIN clips, matched by name; fewer is allowed. Own deliberate regression: a bank with one non-TRAIN clip must be refused."""
    names = bank_clip_names(B)
    bad = train_subset_problems(names, train_ids)
    if not bad and not train_subset_problems(names + ["\x00not-a-train-clip"], train_ids):
        bad.append("the reader's subset check did not refuse a bank holding a clip that is not in TRAIN (the check is blind)")
    if bad:
        raise Refusal(bad)
    return True, {"n_bank_clips": len(names), "n_train_clips": len(set(train_ids)), "n_train_clips_without_bank_records": len(set(train_ids) - set(names)),
                  "deliberate_regression": "refused", "verified_by_reader": "clip names from the shard npz headers vs the TRAIN list"}


def g_kb33(B, eval_ids):
    """K-B3.3 (Amendment 1): bank clips ∩ eval-141 = ∅ (ids AND stable sids). Own deliberate regression: one eval clip injected must be refused. Also the generator's own record."""
    names = bank_clip_names(B)
    bad = clip_overlap(names, eval_ids)
    for sp in B["shard_provs"]:
        k3 = sp.get("kb3_eval_disjoint")
        if not (isinstance(k3, dict) and k3.get("overlap") == 0 and k3.get("n_eval_clips") == len(set(eval_ids))):
            bad.append(f"a shard's kb3_eval_disjoint {k3!r} is not {{overlap 0, n_eval_clips {len(set(eval_ids))}}}")
    if bad == [] and not clip_overlap(names + [sorted(eval_ids)[0]], eval_ids):
        bad.append("the reader's disjointness check did not refuse a bank holding one eval clip (the check is blind)")
    if bad:
        raise Refusal(bad)
    return True, {"n_bank_clips": len(names), "n_eval_clips": len(set(eval_ids)), "overlap": 0, "deliberate_regression": "refused", "verified_by_reader": "ids and stable sids vs the eval-141 list"}


def g_kb4(B):
    c = kb_record(B, "K-B4")
    cases = c.get("cases") if isinstance(c.get("cases"), dict) else {}
    miss = [k for k in KB4_CASES if cases.get(k) is not True]
    if miss:
        raise Refusal(f"K-B4 (the matcher's synthetic cases) must record every case True; not True / absent: {miss}")
    return True, {"cases": list(KB4_CASES), "verified_by_reader": False}


def g_kb5(B):
    c = kb_record(B, "K-B5")
    for k in ("max_pose_now_err_m", "max_rel_transform_err_m"):
        if not (num(c.get(k)) and c[k] <= KB5_TOL_M):
            raise Refusal(f"K-B5 (re-anchoring) must record {k} <= {KB5_TOL_M}: {c.get(k)!r}")
    return True, {"max_pose_now_err_m": c["max_pose_now_err_m"], "max_rel_transform_err_m": c["max_rel_transform_err_m"], "verified_by_reader": False}


def coverage_of(B):
    """The coverage, validated and RECOMPUTED from the counts: eligible fraction = n_eligible / n_windows; the states are summed from reasons_by_tick (ok = kept, the rest = cuts by reason)."""
    cv, rbt = B["prov"].get("coverage"), B["prov"].get("reasons_by_tick")
    if not isinstance(cv, dict):
        raise Refusal("the bank's PROVENANCE.json has no 'coverage' block")
    for k in ("n_windows", "n_eligible"):
        if not is_count(cv.get(k)):
            raise Refusal(f"coverage.{k} is not a non-negative integer: {cv.get(k)!r}")
    if cv["n_windows"] <= 0:
        raise Refusal("coverage.n_windows is 0")
    if not isinstance(rbt, dict) or any(not isinstance(v, dict) or any(not is_count(v.get(r)) for r in REASONS) for v in rbt.values()):
        raise Refusal(f"reasons_by_tick must count every reason {list(REASONS)} per tick (section 2: every cut is counted by reason): {rbt!r}")
    by_tick = {t: sum(v[r] for r in REASONS) for t, v in rbt.items()}
    cuts = {r: sum(v[r] for v in rbt.values()) for r in CUT_REASONS}
    kept = sum(v[REASON_OK] for v in rbt.values())
    return {"n_windows": cv["n_windows"], "n_eligible_windows": cv["n_eligible"], "eligible_fraction": cv["n_eligible"] / cv["n_windows"],
            "n_states_before_cuts": kept + sum(cuts.values()), "n_states_kept": kept, "cuts": cuts, "states_by_tick": by_tick, "stated_fraction": cv.get("fraction")}


def g_bank_accounting(B):
    """The arithmetic of section 1 / 2: every window has one own state at EACH of ticks 1 and 2 before cuts (2 x windows in all); cut records are stored (A11), so the states equal n_records;
    eligible windows <= kept states; the stated fraction / blocked / min are the recomputed ones."""
    cv, bad = coverage_of(B), []
    prov, man, c = B["prov"], B["man"], common_of(B)
    if c.get("partial") is False and cv["n_windows"] != (c.get("train") or {}).get("n_windows"):
        bad.append(f"coverage.n_windows {cv['n_windows']} != common.train.n_windows {(c.get('train') or {}).get('n_windows')}")
    if sorted(cv["states_by_tick"]) != [str(t) for t in TICKS]:
        bad.append(f"reasons_by_tick ticks {sorted(cv['states_by_tick'])}, expected {[str(t) for t in TICKS]}")
    for t, n in cv["states_by_tick"].items():
        if n != cv["n_windows"]:
            bad.append(f"tick {t}: {n} own states before cuts != n_windows {cv['n_windows']} (2 x n_windows = {2 * cv['n_windows']} in all: ticks 1 and 2 of every window)")
    if cv["n_states_before_cuts"] != prov.get("n_records") or cv["n_states_before_cuts"] != man.get("n_records"):
        bad.append(f"states before cuts {cv['n_states_before_cuts']} != n_records (PROVENANCE {prov.get('n_records')}, bank.json {man.get('n_records')}): a cut record is stored, never dropped")
    if cv["n_eligible_windows"] > min(cv["n_windows"], cv["n_states_kept"]):
        bad.append(f"n_eligible {cv['n_eligible_windows']} exceeds the windows / kept states")
    sf = cv["stated_fraction"]
    if not (num(sf) and abs(sf - cv["eligible_fraction"]) <= 1e-9):
        bad.append(f"the stated coverage.fraction {sf!r} != n_eligible / n_windows = {cv['eligible_fraction']}")
    if prov["coverage"].get("min") != COVERAGE_MIN or prov["coverage"].get("blocked") is not (cv["eligible_fraction"] < COVERAGE_MIN):
        bad.append(f"coverage.min {prov['coverage'].get('min')!r} / blocked {prov['coverage'].get('blocked')!r} vs the registered {COVERAGE_MIN} / {cv['eligible_fraction'] < COVERAGE_MIN}")
    if man.get("coverage") != prov.get("coverage") or man.get("reasons_by_tick") != prov.get("reasons_by_tick"):
        bad.append("bank.json's coverage / reasons_by_tick differ from PROVENANCE.json's")
    return not bad, bad or {k: cv[k] for k in ("n_windows", "n_eligible_windows", "eligible_fraction", "n_states_before_cuts", "n_states_kept", "cuts", "states_by_tick")}


def bank_stage(ctl, s, B, m2b_sha, eval_ids, train_ids):
    """Every bank control that needs no arm run (all of them can run before any training)."""
    ctl.run(f"BANK-FILES:s{s}", lambda: g_bank_files(B))
    ctl.run(f"BANK-CONFIG:s{s}", lambda: g_bank_config(B, eval_ids))
    ctl.run(f"K-B0:s{s}", lambda: g_kb0(B))
    ctl.run(f"K-B1:s{s}", lambda: g_kb1(B))
    ctl.run(f"K-B2:s{s}", lambda: g_kb2(B, m2b_sha))
    ctl.run(f"K-B3.1:s{s}", lambda: g_kb31(B, train_ids))
    ctl.run(f"K-B3.3:s{s}", lambda: g_kb33(B, eval_ids))
    ctl.run(f"K-B4:s{s}", lambda: g_kb4(B))
    ctl.run(f"K-B5:s{s}", lambda: g_kb5(B))
    ctl.run(f"BANK-ACCOUNTING:s{s}", lambda: g_bank_accounting(B))


def trainer_values(rd, r0, key, what):
    """The trainer's recorded value of `key` on DAG and DAG-0: ({arm: value}, problems). Missing, conflicting inside a run, or different between the arms are problems."""
    tw, bad = {}, []
    for nm, r in (("DAG", rd), ("DAG-0", r0)):
        vals = find_key(r["sources"], key)
        if not vals:
            bad.append(f"{nm}: the trainer's '{key}' is not recorded (needs the trainer to add it)")
        elif len({json.dumps(v) for _, v in vals}) > 1:
            bad.append(f"{nm}: '{key}' recorded with conflicting values {vals}")
        else:
            tw[nm] = vals[0][1]
    if len(set(tw.values())) > 1:
        bad.append(f"the trainer's '{key}' differs between DAG and DAG-0: {tw}")
    return tw, bad


def g_kb32(B, rd, r0, key_w):
    """K-B3.2 (Amendment 1): the TRAIN window-list sha256 the bank records (common.window_list_sha256 = windows_sha256(J["clip"], wins)) equals the trainer's, on BOTH arms."""
    bank = common_of(B).get("window_list_sha256")
    tw, bad = trainer_values(rd, r0, key_w, "window list")
    for nm, v in tw.items():
        if v != bank:
            bad.append(f"common.window_list_sha256 {str(bank)[:16]} != the trainer's ({nm}) {str(v)[:16]}")
    return not bad, bad or {"window_list_sha256": bank, "trainer": tw}


def g_train_list(B, train_ids, rd, r0, key_t):
    """What makes K-B3.1's TRAIN list M2b's: it hashes (the trainer's own formula) to the trainer's recorded clip-list sha256 on both arms, and to the TRAIN plan the generator recorded."""
    got = train_list_sha256(train_ids)
    tw, bad = trainer_values(rd, r0, key_t, "clip list")
    for nm, v in tw.items():
        if v != got:
            bad.append(f"the TRAIN clip list given to the reader hashes to {got[:16]}, not the trainer's ({nm}) '{key_t}' {str(v)[:16]}")
    gen = (common_of(B).get("train") or {}).get("train_sha256")
    if gen != got:
        bad.append(f"the generator planned over a TRAIN list hashing to common.train.train_sha256 {str(gen)[:16]}, not {got[:16]}")
    return not bad, bad or {"train_sha256": got, "trainer": tw}


def g_bank_declared(tag, seed, B, r, m2b_sha, key_init, key_bank):
    """The DAG arm trained on THIS bank: the argv's --dagger-bank-sha256 and the trainer's built record equal the bank's sha256 (bank.json, hashed here: BANK-FILES ties it to PROVENANCE);
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
            bad.append(f"{w} = {str(v)[:16]} != the bank's sha256 {bsha[:16]}")
    inits = init_sha_values(r, key_init)
    if not inits:
        bad.append("no record of the arm's init checkpoint sha256: K-B2 cannot be shown on the arm")
    gen = common_of(B).get("ckpt_sha256")
    for w, v in inits:
        if str(v) != gen or str(v) != m2b_sha:
            bad.append(f"K-B2: {w} = {str(v)[:16]} but the bank was generated by {str(gen)[:16]} (M2b {m2b_sha[:16]})")
    return not bad, bad or {"bank_sha256": bsha, "built_records": [w for w, _ in built], "bank_manifest_hashed_here": True}


def header_summary(B):
    c = common_of(B)
    return {"bank_sha256": B["man_sha256"], "ckpt_sha256": c.get("ckpt_sha256"), "window_list_sha256": c.get("window_list_sha256"), "thresholds_file_sha256": c.get("thresholds_file_sha256"),
            "constants_sha256_generator_own": c.get("constants_sha256"), "rollout": c.get("rollout"), "hmax": (c.get("cfg") or {}).get("hmax"), "train": c.get("train"),
            "partial": c.get("partial"), "limit_clips": c.get("limit_clips"), "n_shards": B["man"].get("n_shards"), "n_bank_clips": len(bank_clip_names(B)),
            "n_records": B["man"].get("n_records"), "generator_warnings": B["prov"].get("warnings")}


