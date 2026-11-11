#!/usr/bin/env python3
"""DIV layer of refe/scorer_finetune.py (PREREG_DIV_P6): the reversal table that undoes the --split-half edits, so the trainer peels back to the tip blob d665ed995d064f9e1cae956c3c48867cd348cd3d (commit ee1ba225 = the trainer BEFORE DIV).
It sits OUTERMOST: s1zsm_layer.peel calls this layer first (then it peels S1ZSM, s1z_layer peels S1Z, scorer1_layer peels SCORER1), so every older pin that peels the layers (test_s1zsm_layer, test_s1z_layer, test_scorer1_layer, ...
via hnscreen_layer.peel -> carvefeed_layer.peel -> scorer1_layer.peel) keeps working unchanged.  A trainer without the layer (the marker below is absent) is returned unchanged.  refe/test_div_gate.py proves peel(trainer) == the tip blob.
"""
TIP_TRAINER_BLOB = "d665ed995d064f9e1cae956c3c48867cd348cd3d"          # refe/scorer_finetune.py at ee1ba225 = the trainer BEFORE DIV
MARKER = "SPLIT_MIN_SETS = 25000"
# <<<GEN DIV_FILE_REVERSALS
REVERSALS = (
    ('SPLIT_MIN_SETS = 25000                                       # PREREG_DIV_P6 section 2: a half below this many training sets is a launch blocker (refused, never trained)\nSPLIT_RULE = "int(sha256(log_name utf-8).hexdigest(), 16) % 2"\nSPLIT_MUTATE_ENV = "REFE_SPLIT_MUTATE_NOFILTER"              # the deliberate regression: the split filter is REMOVED; the audit below must still see the other half\'s rows and the preflight must REFUSE\n\n\ndef split_half_of(log_name) -> int:\n    """PREREG_DIV_P6: the half a training LOG belongs to, h = int(sha256(log name), 16) % 2; every set of a log follows its log"""\n    if not isinstance(log_name, str) or not log_name:\n        raise ValueError(f"split_half_of: a training row\'s log_name must be a non-empty string, got {log_name!r}")\n    return int(hashlib.sha256(log_name.encode("utf-8")).hexdigest(), 16) % 2\n\n\ndef split_training_index(log_of, tr_idx, half, no_filter=False):\n    """(kept index, facts).  Keeps ONLY the training rows whose log hashes to `half` (log_of(i) = row i\'s log name).  The facts are MEASURED on the index that results (rows_off_half re-hashes every kept row), never echoed\n    from the flag.  no_filter=True is the deliberate regression (REFE_SPLIT_MUTATE_NOFILTER): nothing is filtered, so rows_off_half is > 0 and split_refusals refuses."""\n    by = {0: [0, set()], 1: [0, set()]}\n    for i in tr_idx:\n        h_ = split_half_of(log_of(i))\n        by[h_][0] += 1\n        by[h_][1].add(log_of(i))\n    kept = list(tr_idx) if no_filter else [i for i in tr_idx if split_half_of(log_of(i)) == half]\n    kept_logs = {log_of(i) for i in kept}\n    return kept, {"half": int(half), "rule": SPLIT_RULE, "min_sets": SPLIT_MIN_SETS, "filter_built": not no_filter, "sets": len(kept), "logs": len(kept_logs),\n                  "rows_off_half": sum(1 for i in kept if split_half_of(log_of(i)) != half), "pre_split_sets": len(tr_idx),\n                  "half_sets": {"0": by[0][0], "1": by[1][0]}, "half_logs": {"0": len(by[0][1]), "1": len(by[1][1])}}\n\n\ndef split_refusals(facts):\n    """PREREG_DIV_P6: every kept row\'s log hashes to the half; the kept sets ARE the half\'s sets; BOTH halves hold >= SPLIT_MIN_SETS sets (a half below that is a launch blocker)"""\n    out, h = [], str(facts["half"])\n    if facts["rows_off_half"] != 0:\n        out.append(f"{facts[\'rows_off_half\']} kept training rows have a log that hashes to the other half (the split filter is missing or wrong)")\n    if facts["sets"] != facts["half_sets"][h] or facts["logs"] != facts["half_logs"][h]:\n        out.append(f"the kept {facts[\'sets\']} sets / {facts[\'logs\']} logs are not half {h}\'s {facts[\'half_sets\'][h]} sets / {facts[\'half_logs\'][h]} logs")\n    if min(facts["half_sets"].values()) < facts["min_sets"]:\n        out.append(f"a half holds fewer than {facts[\'min_sets\']} training sets (half 0: {facts[\'half_sets\'][\'0\']}, half 1: {facts[\'half_sets\'][\'1\']}): a launch blocker")\n    return out\n\n\ndef key(r):\n    return (r.get("log_name", ""), r.get("token", ""), int(r.get("step", 0)), int(r.get("rank", 0)))\n',
     'def key(r):\n    return (r.get("log_name", ""), r.get("token", ""), int(r.get("step", 0)), int(r.get("rank", 0)))\n'),
    ('    ap.add_argument("--split-half", type=int, default=-1, choices=(-1, 0, 1), help="DIV (PREREG_DIV_P6 section 2): train ONLY on the training sets whose LOG hashes to this half, h = int(sha256(log name), 16) % 2 (every set of a log follows its log); "\n                    "the held-out monitor is unchanged.  A half below 25,000 sets is refused.  -1 = OFF = this trainer exactly, so a DV launch argv must NAME it")\n    return ap\n\n\ndef main() -> int:\n',
     '    return ap\n\n\ndef main() -> int:\n'),
    ('        print(f"REFUSED: {len(tr_idx)} training sets and {len(ho_idx)} held-out sets survive the filters -- an empty side would train or evaluate on nothing"); return 4\n    split_info = None\n    if a.split_half >= 0:                                    # DIV (PREREG_DIV_P6): the split applies to the TRAINING index only; the held-out index above is untouched\n        try:\n            tr_idx, split_info = split_training_index(lambda i_: ds.rows[i_].get("log_name"), tr_idx, a.split_half, os.environ.get(SPLIT_MUTATE_ENV) == "1")\n        except ValueError as e_:\n            print(f"REFUSED: split: {e_}"); return 4\n        print("ZZSPLIT " + json.dumps(split_info), flush=True)\n        bad_split = split_refusals(split_info)\n        if bad_split:\n            for m_ in bad_split:\n                print("REFUSED: split: " + m_)\n            return 4\n',
     '        print(f"REFUSED: {len(tr_idx)} training sets and {len(ho_idx)} held-out sets survive the filters -- an empty side would train or evaluate on nothing"); return 4\n'),
    ('    if split_info is not None:                               # DIV: the BUILT facts of the split (absent without --split-half: every pre-DIV data event is byte-identical)\n        info["split"] = split_info\n    log(info)\n    g = torch.Generator().manual_seed(a.seed)\n',
     '    log(info)\n    g = torch.Generator().manual_seed(a.seed)\n'),
)
# >>>GEN DIV_FILE_REVERSALS


def has_layer(src: str) -> bool:
    return MARKER in src


def peel(src: str) -> str:
    """undo the DIV edits (each `new` text must occur exactly once); a trainer without the layer is returned unchanged.  The VDAC layer (refe/vdac_layer.py, PREREG_VDAC_P6) sits ON TOP of this one and is peeled first"""
    import vdac_layer as VDAC
    src = VDAC.peel(src)
    if not has_layer(src):
        return src
    for new, old in REVERSALS:
        assert src.count(new) == 1, f"DIV reversal: the text occurs {src.count(new)} times (must be exactly 1): {new[:70]!r}"
        src = src.replace(new, old)
    return src
