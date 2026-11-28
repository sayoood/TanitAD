import re
p = 'C:/Users/Admin/dagread/test_dag_read.py'
s = open(p, encoding='utf-8', newline='').read()
fx = open('C:/Users/Admin/dagread/scratch/t_fixture.py', encoding='utf-8', newline='').read()
bt = open('C:/Users/Admin/dagread/scratch/t_bank_tests.py', encoding='utf-8', newline='').read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


def splice(start, end, new):
    """Replace s[start_marker : start of the line holding end_marker) (that line kept)."""
    global s
    a, b = s.index(start), s.index(end)
    b = s.rindex(chr(10), 0, b) + 1
    assert a < b
    s = s[:a] + new + s[b:]


# 1. docstring + ROLLOUT
rep('''the T1 tool's ep*.npz dumps and t1v8f json, the T0 eval's json + npz, and a bank report + eval-141 clip list in the (provisional) schema documented in dag_read.py.''',
    '''the T1 tool's ep*.npz dumps and t1v8f json, the T0 eval's json + npz, and a DAgger bank in the bank generator's REAL layout (PROVENANCE.json + bank.json + shard provenance + shard npz,
built from the generator's real smoke documents under fixtures/dagger_smoke/ as templates) with the eval-141 and M2b TRAIN clip lists.''')
rep('''ROLLOUT = {"replan_steps": 10, "limit": "causal", "nav": "deadreckon", "scene": "logged", "schedule": "periodic", "executor": "plan", "ticks": [1, 2]}''',
    '''ROLLOUT = {"replan_steps": 10, "limit": "causal", "nav": "deadreckon", "scene": "logged", "schedule": "periodic", "executor": "plan", "post_map": "none (PLAIN)"}''')
# 2. bank_report -> fixture
splice('def bank_report(s, m2b_sha, bank_sha, n_elig=250000, clips=None):', 'class Fx:', fx + '\n\n')
# 3. Fx
rep('''"bank": {}, "eval141_clip_ids": str(self.root / "eval141_clip_ids.json")}''', '''"bank": {}, "eval141_clip_ids": str(self.root / "eval141_clip_ids.json"),
                         "train_clip_ids": str(self.root / "train_clip_ids.json")}''')
rep('''    def __init__(self, root, world, m2b=False, orc=False, logged=False, nb=None, elig=250000):''',
    '''    def __init__(self, root, world, m2b=False, orc=False, logged=False, nb=None, elig=250000, bank_mods=None):''')
rep('''        jdump(self.root / "eval141_clip_ids.json", EVAL_IDS)
''', '''        jdump(self.root / "eval141_clip_ids.json", EVAL_IDS)
        jdump(self.root / "train_clip_ids.json", CLIPS)
        self.banks = {}
''')
splice('            bf = self.root / "bank" / f"bank_s{s}.bin"', '            for arm in self.arms:\n                self.manifest["arms"][arm][str(s)] = self.build_run(arm, s, logged)',
       '''            n_e = elig[s] if isinstance(elig, dict) else elig
            B = make_bank(self.m2b_sha[s], n_e)
            if bank_mods and s in bank_mods:                              # an edit is made BEFORE the arms' records are built, so only the intended control sees it
                bank_mods[s](B)
                finalize(B)
            self.banks[s], self.bank_sha[s] = B, B["man_sha256"]
            self.manifest["bank"][str(s)] = {"report": write_bank(B, self.root / f"bank_s{s}")}
''')
rep('''"--dagger-bank", f"{root}/bank/bank_s{s}.bin",''', '''"--dagger-bank", f"{root}/bank_s{s}/bank.json",''')
rep('''"train_sha256": TRAIN_SHA,
                                        "excluded_sha256": "bb" * 32, "argv_sha": sha, "levers": lev, **tot})''',
    '''"train_sha256": TRAIN_SHA,
                                        "excluded_sha256": "bb" * 32, "argv_sha": sha, "levers": lev, **tot,
                                        **({"train_window_list_sha256": WIN_SHA} if arm != "ORC0" else {})})        # ORC-0 is the older build: it records no window-list sha''')
# 5. bank_edit
splice('def bank_edit(s, fn):', '# ======================================================================================================================== 1. the outcome table is the PREREG\'s', '')
# 6. prereg tests
splice('def test_prereg_file_is_the_07dcb513_draft():', 'def test_committed_numbers_are_the_preregs():',
       '''DRAFT_BYTES, DRAFT_SHA = 21728, "719e5f153070e9c6986c11dac9250ea3f2a2363968c9ea65225d8b582b029374"                 # the draft as banked at ac139f71 / 07dcb513
LANDED_SHA = "a4df4b8f44d9a322a1907338c990dee09d03fb35e1e136e6c60eb72a790f60f9"                                      # git show 2a78f059:"Project Steering/PREREG_V8F_DAGGER.md"


def test_prereg_file_is_the_landed_draft_plus_amendment_1():
    b = PREREG.read_bytes()
    assert len(b) > DRAFT_BYTES and hashlib.sha256(b[:DRAFT_BYTES]).hexdigest() == DRAFT_SHA                         # it STARTS WITH the draft's bytes: nothing above the amendment moved
    t = b.decode("utf-8")
    assert t.count("## Amendment 1 (`A1-2026-10-26`)") == 1 and b.index(b"## Amendment 1 (`A1-2026-10-26`)") >= DRAFT_BYTES    # carried, and appended after the draft
    assert hashlib.sha256(b).hexdigest() == LANDED_SHA                                                               # the whole landed file (a later amendment re-pins this on purpose)
    assert "2a78f059" in dag.PREREG_ID and LANDED_SHA in dag.PREREG_ID and DRAFT_SHA in dag.PREREG_ID and "PREREG_V8F_DAGGER" in dag.PREREG_ID
    assert dag.PREREG_ID.endswith("+ Amendment 1 (A1-2026-10-26))") and dag.PREREG_LANDED_SHA256 == LANDED_SHA and dag.PREREG_DRAFT_SHA256 == DRAFT_SHA and "07dcb513" not in dag.PREREG_ID


def test_amendment_1_says_what_the_reader_implements():
    t = PREREG.read_text(encoding="utf-8").replace("**", "")
    a = t[t.index("## Amendment 1"):]
    assert "so this arm runs at **p = 0.1**".replace("**", "") in a and dag.DAG_P == 0.1                           # section 0: p by the mapping (the reader's one constant)
    assert "K-B3 is three checks, each read on its own" in a
    assert "the bank's clips ⊆ M2b's TRAIN clips, matched BY NAME" in a and "Equality is not required" in a                # 3.1, and the bank may hold fewer
    assert "the TRAIN window-list sha256 that the bank records equals the trainer's" in a                                 # 3.2
    assert "the bank's clips ∩ eval-141 = ∅" in a and "The deliberate regression (one eval clip injected) must fail check 3" in a   # 3.3
    assert "Each is its own control, so a failure names which one" in a
    src = open(DAG_PATH, encoding="utf-8").read()
    for cid in ("K-B3.1", "K-B3.2", "K-B3.3"):                                                                        # the reader names each of the three
        assert f'f"{cid}:s{{s}}"' in src, cid
    assert "BANK-WINDOWS" not in src                                                                                  # the pre-amendment single control is gone


''')
# 7. committed numbers
rep('''    assert dag.BANK_ROLLOUT == ROLLOUT
''', '''    assert dag.BANK_ROLLOUT == ROLLOUT and dag.TICKS == (1, 2) and "ticks 1 and 2 only" in t
''')
rep('''and dag.CUT_REASONS == ("off_route", "heading", "end_edge")''', '''and dag.CUT_REASONS == ("off_lat", "off_head", "end_edge", "nonfinite") and dag.REASONS == ("ok",) + dag.CUT_REASONS''')
# 8. ROBUST controls
rep('''    for k in ("K-B0", "K-B1", "K-B2", "K-B3", "K-B4", "K-B5", "BANK-CONFIG", "BANK-ACCOUNTING"):''',
    '''    for k in ("K-B0", "K-B1", "K-B2", "K-B3.1", "K-B3.3", "K-B4", "K-B5", "BANK-FILES", "BANK-CONFIG", "BANK-ACCOUNTING"):''')
rep('''"ORC0-REPLICATE:s0", "BANK-WINDOWS:s1",''', '''"ORC0-REPLICATE:s0", "K-B3.2:s1", "TRAIN-LIST:s0", "TRAIN-LIST:s1", "K-B3.2:s0",''')
# 9. bank unit tests
splice('def good_bank(**kw):', '# ======================================================================================================================== 8. COVERAGE -> BLOCKED', bt + '\n')
open(p, 'w', encoding='utf-8', newline='').write(s)
print("applied part 1")
