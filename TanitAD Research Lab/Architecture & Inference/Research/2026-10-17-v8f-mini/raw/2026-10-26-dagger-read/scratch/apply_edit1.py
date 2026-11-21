p = 'C:/Users/Admin/dagread/dag_read.py'
s = open(p, encoding='utf-8', newline='').read()


def rep(old, newt, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, newt)


rep('''    kn, ke, kp, kb, ki, kw = (cfg["keys"][k] for k in ("n", "eligible", "built_p", "bank_sha", "init_sha", "window_sha"))''',
    '''    kn, ke, kp, kb, ki, kw, kt = (cfg["keys"][k] for k in ("n", "eligible", "built_p", "bank_sha", "init_sha", "window_sha", "train_sha"))''')
rep('''    banks = {}
    for s in SEEDS:
        be = (m.get("bank") or {}).get(str(s))
        if be is None:
            raise Refusal(f"manifest has no bank for seed {s}: COVERAGE and the bank controls are read for BOTH seeds")
        rep = jload(need(be.get("report"), f"bank seed {s} report"))
        if not isinstance(rep, dict) or rep.get("kind") != BANK_KIND:
            raise Refusal(f"bank seed {s}: the report's kind is {rep.get('kind') if isinstance(rep, dict) else None!r}, not {BANK_KIND!r}")
        banks[s] = {"report": rep, "entry": be}
        bank_stage(ctl, s, rep, cfg["m2b_sha"].get(s, ""), eval_ids)''',
    '''    train_ids = jload(need(m.get("train_clip_ids"), "train_clip_ids (M2b's TRAIN clip list K-B3.1 is checked against; it must hash to the trainer's recorded clip-list sha256)"))
    if not isinstance(train_ids, list) or not all(isinstance(x, str) for x in train_ids):
        raise Refusal("train_clip_ids is not a JSON list of clip-id strings")
    banks = {}
    for s in SEEDS:
        be = (m.get("bank") or {}).get(str(s))
        if be is None:
            raise Refusal(f"manifest has no bank for seed {s}: COVERAGE and the bank controls are read for BOTH seeds")
        banks[s] = {"B": load_bank(be.get("report"), be.get("bank_file"), what=f"bank seed {s} report"), "entry": be}
        bank_stage(ctl, s, banks[s]["B"], cfg["m2b_sha"].get(s, ""), eval_ids, train_ids)''')
rep('''    cov = {s: coverage_of(banks[s]["report"]) for s in SEEDS}''', '''    cov = {s: coverage_of(banks[s]["B"]) for s in SEEDS}''')
rep('''        blk["unverified"] = ["window-list sha256 vs the trainer's (K-B3, second half): read from the arm runs, none may exist under a BLOCK"]''',
    '''        blk["unverified"] = ["K-B3.2 (the bank's window-list sha256 vs the trainer's) and the TRAIN-LIST binding: read from the arm runs, none may exist under a BLOCK"]''')
rep('''"bank": {f"s{s}": {"header": hdr(banks[s]["report"]), "coverage": cov[s]} for s in SEEDS}, "runs": {}}''',
    '''"bank": {f"s{s}": {"header": header_summary(banks[s]["B"]), "coverage": cov[s]} for s in SEEDS}, "runs": {}}''')
rep('''g_bank_declared(tag, s, banks[s]["report"], tr, cfg["m2b_sha"].get(s, ""), ki, kb,
                                                                                                         banks[s]["entry"].get("bank_file")))''',
    '''g_bank_declared(tag, s, banks[s]["B"], tr, cfg["m2b_sha"].get(s, ""), ki, kb))''')
rep('''        ctl.run(f"BANK-WINDOWS:s{s}", lambda rd=rd, r0=r0, s=s: g_bank_windows(s, banks[s]["report"], rd["train"], r0["train"], kw))''',
    '''        ctl.run(f"K-B3.2:s{s}", lambda rd=rd, r0=r0, s=s: g_kb32(banks[s]["B"], rd["train"], r0["train"], kw))
        ctl.run(f"TRAIN-LIST:s{s}", lambda rd=rd, r0=r0, s=s: g_train_list(banks[s]["B"], train_ids, rd["train"], r0["train"], kt))''')
rep('''            "bank": {f"s{s}": {"header": hdr(banks[s]["report"]), "coverage": cov[s]} for s in SEEDS},''',
    '''            "bank": {f"s{s}": {"header": header_summary(banks[s]["B"]), "coverage": cov[s]} for s in SEEDS},''')
rep('''"bank.{0,1}: {report, [bank_file]}; eval141_clip_ids: path to a JSON list")''',
    '''"bank.{0,1}: {report = the bank directory's PROVENANCE.json (bank.json and the shards are read beside it), [bank_file = bank.json elsewhere]}; "
                    "eval141_clip_ids, train_clip_ids: paths to JSON lists of clip-id strings (M2b's TRAIN list must hash to the trainer's train_sha256)")''')
rep('''    ap.add_argument("--key-window-sha", default="train_sha256", help="PROVISIONAL: the trainer's TRAIN-list sha256 in the summary (v8f_mini_train.py hashes the CLIP list under this key)")''',
    '''    ap.add_argument("--key-window-sha", default="train_window_list_sha256", help="PROVISIONAL: the trainer's TRAIN WINDOW-list sha256 (G.windows_sha256(J['clip'], wins)); the tip's trainer does not record one yet")
    ap.add_argument("--key-train-sha", default="train_sha256", help="the trainer's TRAIN CLIP-list sha256 (v8f_mini_train.py records it under this key)")''')
rep('''"init_sha": a.key_init_sha, "window_sha": a.key_window_sha}, "prereg": a.prereg}''',
    '''"init_sha": a.key_init_sha, "window_sha": a.key_window_sha, "train_sha": a.key_train_sha}, "prereg": a.prereg}''')
rep('''PREREG_ID = ("TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-26-dagger-draft/PREREG_V8F_DAGGER.md @ commit 07dcb513 "
             "(DRAFT 2026-10-26, no amendment; sha256 719e5f153070e9c6986c11dac9250ea3f2a2363968c9ea65225d8b582b029374)")''',
    '''PREREG_DRAFT_SHA256 = "719e5f153070e9c6986c11dac9250ea3f2a2363968c9ea65225d8b582b029374"                       # the draft as banked at ac139f71 / 07dcb513
PREREG_LANDED_SHA256 = "a4df4b8f44d9a322a1907338c990dee09d03fb35e1e136e6c60eb72a790f60f9"                      # the landed file (2a78f059): the draft's bytes + `## Amendment 1`
PREREG_ID = ("Project Steering/PREREG_V8F_DAGGER.md @ commit 2a78f059 (landed file sha256 " + PREREG_LANDED_SHA256 + " = the DRAFT 2026-10-26 (sha256 " + PREREG_DRAFT_SHA256
             + ") + Amendment 1 (A1-2026-10-26))")''')
rep('''"""dag_read.py -- the reader for PREREG_V8F_DAGGER (DRAFT 2026-10-26, commit 07dcb513, claim row H-V8F-DAGGER-1), adapted from the landed orc_read.py (Amendment 4).''',
    '''"""dag_read.py -- the reader for PREREG_V8F_DAGGER (landed 2a78f059: the 2026-10-26 draft + Amendment 1 `A1-2026-10-26`, claim row H-V8F-DAGGER-1), adapted from the landed orc_read.py (Amendment 4).''')
rep('''    PROVISIONAL (to be reconciled with the trainer / bank worker; each is a CLI override or a documented schema, see NOTES.md): the bank report schema
    (BANK_KIND), the built-record key names dagger_p / dagger_bank_sha256, the summary key holding the trainer's TRAIN-list sha, and the eval-141 clip-id list.''',
    '''    The bank is read as `v8f_dagger_bank.py` writes it (PROVENANCE.json + bank.json + shard provenance + the shard npz headers; schema in the comment above `sha12`). K-B3 is THREE controls
    (Amendment 1 section 3): K-B3.1 bank clips subset of M2b's TRAIN clips by name, K-B3.2 the TRAIN window-list sha256 equals the trainer's, K-B3.3 bank clips and eval-141 are disjoint.
    PROVISIONAL (not yet written by the trainer / generator; each is a CLI override or listed in NOTES.md under "needs ... to add"): the built-record key names dagger_p / dagger_bank_sha256,
    the trainer's TRAIN window-list sha key (train_window_list_sha256), and the generator's `controls` K-B4 / K-B5 records.''')
open(p, 'w', encoding='utf-8', newline='').write(s)
print("applied")
