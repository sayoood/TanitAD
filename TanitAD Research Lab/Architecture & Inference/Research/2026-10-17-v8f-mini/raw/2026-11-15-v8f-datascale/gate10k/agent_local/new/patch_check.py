import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)


# (1) reach_problems: batch kinds (a B-set arm has no 10k windows)
rep('def reach_problems(rec: dict, stage_b: bool = False) -> list:', 'def reach_problems(rec: dict, stage_b: bool = False, kinds: tuple = ("corpus", "tenk", "mixed")) -> list:')
rep('    for kind in ("corpus", "tenk", "mixed"):\n        k = rec.get(kind)',
    '    for kind in kinds:                                                        # a B-set arm has no 10k windows: kinds = ("corpus",) (gate patch 2026-11-17)\n        k = rec.get(kind)')
# (2) required checks by stage
rep('REQUIRED_UNION = ("C-MERGE", "C-JOIN-EQ", "C-MASK")',
    'REQUIRED_UNION = ("C-MERGE", "C-JOIN-EQ", "C-MASK")\nREQUIRED_STAGE_B = ("C-BANK", "G-DAGGER")             # gate patch 2026-11-17: the bank (n_shards 4, complete, controls) and the trainer\'s own stage-B G-LIVE (acc + dagger probes)')
rep('def required_checks(arm: str) -> tuple:\n    return REQUIRED_DATA + (REQUIRED_UNION if ARMS[arm]["set"] in ("U", "R", "X") else ()) + REQUIRED_TRAINER',
    'def required_checks(arm: str, stage: str = "A") -> tuple:\n    return REQUIRED_DATA + (REQUIRED_UNION if ARMS[arm]["set"] in ("U", "R", "X") else ()) + REQUIRED_TRAINER + (REQUIRED_STAGE_B if stage == "B" else ())')
rep('    miss = [c for c in required_checks(arm) if not (checks.get(c) or {}).get("pass")]\n    if miss:\n        raise ValueError',
    '    miss = [c for c in required_checks(arm, b.get("stage", "A")) if not (checks.get(c) or {}).get("pass")]\n    if miss:\n        raise ValueError')
rep('    miss = [c for c in required_checks(arm) if not (tok.get("checks", {}).get(c) or {}).get("pass")]',
    '    miss = [c for c in required_checks(arm, stage) if not (tok.get("checks", {}).get(c) or {}).get("pass")]')
rep('fails = [f"check {c} did not pass: {json.dumps(checks[c])[:500]}" for c in required_checks(arm) if not checks.get(c, {}).get("pass")]',
    'fails = [f"check {c} did not pass: {json.dumps(checks.get(c))[:500]}" for c in required_checks(arm, stage) if not checks.get(c, {}).get("pass")]')
# (3) mint: kinds for G-LIVE; C-BANK and G-DAGGER on stage B
rep('    put("G-LIVE", reach_problems(ev["reach"], spec["stage_b"] and stage == "B") if ev.get("reach") else ["no reach-table evidence (pod-side real-data smoke)"])',
    '    kinds = ("corpus",) if spec["set"] == "B" else ("corpus", "tenk", "mixed")\n'
    '    put("G-LIVE", reach_problems(ev["reach"], False, kinds) if ev.get("reach") else ["no reach-table evidence (pod-side real-data smoke)"])\n'
    '    if stage == "B":\n'
    '        bank_dir = (parse_argv(toks).get("--dagger-bank") or [""])[0]\n'
    '        init = (parse_argv(toks).get("--init-ckpt") or [""])[0]\n'
    '        put("C-BANK", bank_problems(bank_dir, init, dry), n_shards=BANK_SHARDS)\n'
    '        sc = json.load(open(a.selfcheck, encoding="utf-8")) if getattr(a, "selfcheck", "") and os.path.isfile(a.selfcheck) else None\n'
    '        put("G-DAGGER", dagger_problems(sc, toks) if sc else ["no --selfcheck: the trainer\'s --selfcheck-only record of THIS stage-B argv (real bank, real init checkpoint)"],\n'
    '            selfcheck_sha256=sha_file(a.selfcheck) if sc else None)')
rep('            p.add_argument("--holdout-manifest", default="")', '            p.add_argument("--selfcheck", default=""); p.add_argument("--holdout-manifest", default="")')
# the stage-B bank / dagger checks
new = open(sys.argv[2], encoding="utf-8").read()
rep("# ------------------------------------------------------------------------------------------------ the token\n", new + "\n# ------------------------------------------------------------------------------------------------ the token\n")
open(p, "w", encoding="utf-8").write(s)
print("patched", p)
