#!/usr/bin/env python3
"""Re-base dag_orc_read.py (DAG-ORC's reader, sha256 a4907f4e...) from DAG onto DAG-OACC exactly as Amendment 1 (A1-2026-10-27) specifies.

    python rebase_reader.py        reads the PRISTINE reader E:/Projects/TanitAD-artifacts/dag_orc_read/dag_orc_read.py, applies every exact-count replacement below,
                                   writes ./read/dag_orc_read.py (LF).  Every sha256 / commit the new reader pins is READ from a banked artifact here (SHA256SUMS.txt, the
                                   launch-gate token, the pulled argv.json, DAGOACC_READ.json), never retyped: a disagreement between two banked copies stops this script.
The replacements are the auditable record of "where DAG became DAG-OACC" (NOTES.md section R lists them).
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path("E:/Projects/TanitAD-artifacts/dag_orc_read/dag_orc_read.py")
OUT = HERE / "read" / "dag_orc_read.py"
ART = Path("E:/Projects/TanitAD-artifacts")
OACC = ART / "v8f-dagoacc"


def sums(p):
    o = {}
    for ln in Path(p).read_text(encoding="utf-8").splitlines():
        h, n = ln.split(" ", 1)
        o[n.strip().lstrip("*")] = h
    return o


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- the pins, READ from the banked artifacts
pin = {"ckpt": {}, "m2b": {}, "bank": {}, "argv_sha": {}, "commit": set(), "queue_sha": set()}
for s in (0, 1):
    d = OACC / f"M2b_s{s}_dagoacc"
    sm = sums(d / "SHA256SUMS.txt")
    assert sha(d / "argv.json") == sm["argv.json"] and sha(d / "summary.json") == sm["summary.json"], f"s{s}: banked files != SHA256SUMS.txt"
    tok = json.loads((d / "launch_gate_token.json").read_text(encoding="utf-8"))
    assert sha(d / "launch_gate_token.json") == sm["launch_gate_token.json"]
    argv = json.loads((d / "argv.json").read_text(encoding="utf-8"))["argv"]
    summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    pin["ckpt"][s] = sm["ckpt_last.pt"]
    pin["m2b"][s] = tok["init_ckpt_sha256"]
    assert summ["levers"]["init_ckpt"]["sha256"] == tok["init_ckpt_sha256"], "summary vs token disagree on the M2b init"
    pin["bank"][s] = tok["bank_sha256"]
    assert argv[argv.index("--dagger-bank-sha256") + 1] == tok["bank_sha256"], "argv vs token disagree on the bank"
    pin["argv_sha"][s] = summ["argv_sha"]
    pin["commit"].add(tok["binding"]["commit"])
    w = argv[argv.index("--w-acc") + 1]
    assert w == "49.6969" and tok["w_acc"] == 49.6969
assert len(pin["commit"]) == 1
COMMIT = next(iter(pin["commit"]))
AMEND1_SHA = sha(ART / "dagland/dagorc_amend1/AMEND1.md")
OACC_READ_SHA = sha(ART / "v8f-dagoacc-t1/read/DAGOACC_READ.json")
assert OACC_READ_SHA == "9e6bd4f6940fb3e868d4d8aede50cf47c082a3833f014c34bb87ff201c99a894", OACC_READ_SHA      # the sha the amendment and the task name
assert sha(HERE / "read" / "AMEND1.md") == AMEND1_SHA

t = SRC.read_text(encoding="utf-8")
assert "\r" not in t
N = [0]


def rep(old, new, n=1):
    global t
    assert t.count(old) == n, (t.count(old), old[:90])
    t = t.replace(old, new)
    N[0] += 1


# ================================================================ header
rep('"""dag_orc_read.py -- the dev-box T1-kin read of PREREG_V8F_DAG_ORC (claim row H-V8F-DAG-ORC-1).  BUILT AND SELF-TESTED, NOT RUN ON DAG-ORC OUTPUTS (none exist).',
    '"""dag_orc_read.py -- the dev-box T1-kin read of PREREG_V8F_DAG_ORC (claim row H-V8F-DAG-ORC-1), RE-BASED ON DAG-OACC by Amendment 1 (A1-2026-10-27).  BUILT AND SELF-TESTED, NOT RUN ON DAG-ORC OUTPUTS (none exist).\n'
    '\nAMENDMENT 1 (binding; AMEND1.md, raw-byte sha256 AMEND1_SHA256 below): "DAG" and "CTRL" both mean DAG-OACC_s (the banked run v8f-dagoacc/M2b_s<s>_dagoacc).  So the reader has THREE live slots\n'
    '(A = DAG-OACC + the perturbation composed with the swap; CTRL = DAG-OACC, banked; REP = DAG-OACC\'s recipe at --seed 10 / 11) and B (read only on A\'s HARMS, not queued).  X - CTRL and X - DAG are ONE\n'
    'contrast (A - DAG-OACC); R-ADE and R-KEEP share it, so do R-NEW and R-NOBASE.  C_s = {yaw-rate MAE, curvature MAE, L2, M1} at both seeds; D_s = the 15 other rows, NO exempt rows.  RF0 is unchanged\n'
    '(P0_RF0.json); RF_r = max over s of |d_r,s(REP - DAG-OACC)|.  K-RC has TWO legs: DAG2 as before (RC_DAG2.json) and DAG-OACC - P10 against DAGOACC_READ.json, 0 differences (RC_OACC.json).\n'
    'The ORIGINAL PREREG text (sha 38bc688b...) is still pinned verbatim; every place the amendment changes a coded value is marked "A1" in a comment.\n')
rep("""    python dag_orc_read.py --rc-dag2 --out RC_DAG2.json                         K-RC: this reader's analysis on DAG2 / DAG2-0, floors 0, adoption off, vs the banked DAG2_READ.json
""", """    python dag_orc_read.py --rc-dag2 --out RC_DAG2.json                         K-RC leg 1: this reader's analysis on DAG2 / DAG2-0, floors 0, adoption off, vs the banked DAG2_READ.json
    python dag_orc_read.py --rc-oacc --out RC_OACC.json                         K-RC leg 2 (A1): DAG-OACC - P10 through this reader's slot / pair machinery vs DAGOACC_READ.json's Read-A table, 0 differences
""")
rep("""Manifest (runs.json): {"seeds": [0, 1], "arms": {"A"|"CTRL"|"REP"|"B"|"DAG": {"0": ENTRY, "1": ENTRY}}, "build": K-BUILD record}.  ENTRY = {ckpt_tail, train_dir, token,
t1: {tag, ticks, dump_dir, json}, t1_logged: {..}, t0: {json, npz} (box 2's standard eval), t0_fresh: {json, npz}} exactly as dag2t1_manifest.py writes them.  CTRL and DAG are the banked runs.""",
    """Manifest (runs.json): {"seeds": [0, 1], "arms": {"A"|"CTRL"|"REP"|"B": {"0": ENTRY, "1": ENTRY}}, "build": K-BUILD record}.  ENTRY = {ckpt_tail, train_dir, token,
t1: {tag, ticks, dump_dir, json}, t1_logged: {..}, t0: {json, npz} (box 2's standard eval), t0_fresh: {json, npz}} exactly as dag2t1_manifest.py writes them.  A1: CTRL is DAG-OACC, the banked run
(t1 + t0 only: runs.json's arms.DAGOACC of v8f-dagoacc-t1; it has no --limit logged lane and no fresh T0, and needs none).  The rc-DAG2 leg still uses A / CTRL / DAG slots of the DAG2 manifests.""")

# ================================================================ constants (A1)
rep('PREREG_ID = "Project Steering/PREREG_V8F_DAG_ORC.md @ 3b7b404e"',
    'PREREG_ID = "Project Steering/PREREG_V8F_DAG_ORC.md @ 3b7b404e + Amendment 1 (A1-2026-10-27, re-based on DAG-OACC)"')
rep('RF0_SHA256 = "1608d70b21b6b6a45136a6d96828f70b5bbabd9c0ceb84db136927593c79ad0f"',
    f'AMEND1_SHA256 = "{AMEND1_SHA}"       # A1: raw bytes of AMEND1.md (E:/Projects/TanitAD-artifacts/dagland/dagorc_amend1/AMEND1.md)\n'
    f'OACC_READ_SHA256 = "{OACC_READ_SHA}"       # A1: v8f-dagoacc-t1/read/DAGOACC_READ.json (K-RC leg 2\'s banked table)\n'
    'RF0_SHA256 = "1608d70b21b6b6a45136a6d96828f70b5bbabd9c0ceb84db136927593c79ad0f"')
rep('DAG_CKPT_SHA = {0: "d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1", 1: "1d12ec2eb825990d57dcc45dda50b5178d1d733961b47ab0e3d50e6155fa39ef"}    # section 1: DAG_s',
    'DAG_CKPT_SHA = {0: "d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1", 1: "1d12ec2eb825990d57dcc45dda50b5178d1d733961b47ab0e3d50e6155fa39ef"}    # section 1: DAG_s (rc-DAG2 leg ONLY after A1)\n'
    f'M2B_CKPT_SHA = {{0: "{pin["m2b"][0]}", 1: "{pin["m2b"][1]}"}}      # A1 K-INIT: M2b_s, as recorded in DAG-OACC\'s launch-gate token (init_ckpt_sha256)\n'
    f'OACC_CKPT_SHA = {{0: "{pin["ckpt"][0]}", 1: "{pin["ckpt"][1]}"}}      # A1: CTRL = DAG-OACC_s ckpt_last.pt (v8f-dagoacc/M2b_s<s>_dagoacc/SHA256SUMS.txt)\n'
    f'OACC_BANK_SHA = {{0: "{pin["bank"][0]}", 1: "{pin["bank"][1]}"}}      # A1 K-PAIR: "the bank sha256 is DAG-OACC\'s" (the iteration-1 bank.json, from the launch-gate token)\n'
    f'OACC_ARGV_SHA16 = {{0: "{pin["argv_sha"][0]}", 1: "{pin["argv_sha"][1]}"}}      # DAG-OACC_s summary.json argv_sha\n'
    'W_ACC = "49.6969"      # A1: DAG-OACC\'s --w-acc, verbatim in A and REP\n'
    'SWAP_P = 0.1      # A1 K-REACH: dagger_n / dagger_eligible in 0.1 +- 0.02 on A, B and REP\n'
    'COMPOSE_FLAG = "--ego-perturb-with-dagger"      # A1: the new default-off trainer flag; named on A and B, absent on REP and CTRL')
rep('BUILD_BASE_COMMIT = "1d4f4fbb"                                                           # section 6, K-BUILD',
    f'BUILD_BASE_COMMIT = "{COMMIT[:8]}"                                                           # A1 K-BUILD: the commit DAG-OACC trained at (launch-gate token binding.commit {COMMIT})')
rep('BUILD_FILES = ("stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/models/v8f_mini.py")',
    'BUILD_FILES = ("stack/scripts/v8f_mini_train.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_dagger.py")      # A1: + v8f_dagger.py (a key file of DAG-OACC\'s gate token; the composition may touch the swap\'s mask)')
rep('ARMS = ("A", "CTRL", "REP", "B", "DAG")\nREQUIRED_SLOTS = ("A", "CTRL", "REP", "DAG")             # B is read only on A\'s HARMS (section 7)',
    'ARMS = ("A", "CTRL", "REP", "B")                         # A1: DAG := CTRL := DAG-OACC, one slot; B is read only on A\'s HARMS (section 7) and is NOT queued (A1 section 4)\nRC_ARMS = ("A", "CTRL", "REP", "B", "DAG")      # the rc-DAG2 leg keeps its three slots\nREQUIRED_SLOTS = ("A", "CTRL", "REP")')
rep('''CRAWL_SET = {0: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"),                    # section 4: the rows separated-worse for DAG - P10 (RESULT_V8F_DAG_REREAD section 1)
             1: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share", "yaw-rate MAE")}''',
    '''CRAWL_SET = {0: ("yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"),         # A1 section 2: the rows separated-worse for DAG-OACC - P10, both seeds (DAGOACC_READ.json)
             1: ("yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")}
CRAWL_SET_PRE_A1 = {0: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"), 1: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share", "yaw-rate MAE")}      # the ORIGINAL section 4 (DAG - P10); the rc-DAG2 leg composes with THIS''')
rep('''EXEMPT = ("accel MAE", "DK-VIOL")                                                                       # section 4: belong to DAG-OACC; reported, exempt from R-NEW and R-NOBASE
DOMAIN_N = {0: 14, 1: 13}                                                                               # section 4: |D_s|''',
    '''EXEMPT = ()                                                                                             # A1 section 2: D_s has NO exempt rows (DAG-OACC has cleared accel MAE and DK-VIOL, so they are guarded)
EXEMPT_PRE_A1 = ("accel MAE", "DK-VIOL")                                                                # the ORIGINAL section 4 exemption; the rc-DAG2 leg composes with THIS
DOMAIN_N = {0: 15, 1: 15}                                                                               # A1: |D_s| = 19 rows - |C_s| = 15 at both seeds
DOMAIN_N_PRE_A1 = {0: 14, 1: 13}                                                                        # the ORIGINAL section 4 (rc-DAG2 leg)''')

# ================================================================ the amendment pin + the amended committed-next texts
rep('''OUTCOME_TABLE = tuple(tuple(r) for r in PREREG_CELLS["7."]["rows"][1:])          # (outcome, condition, committed next), the PREREG's order''',
    '''AMEND_PHRASES = (      # A1: verbatim phrases of AMEND1.md that carry the values this reader codes (markdown bold stripped, whitespace normalised)
    "argv(A_s) equals argv(DAG-OACC_s) except `--out`, the six perturbation flags at §1's values and `--ego-perturb-with-dagger`.",
    "argv(REP_s) equals argv(DAG-OACC_s) except `--out` and `--seed`.", "The bank sha256 is DAG-OACC's.", "K-INIT:** M2b_s, as recorded in DAG-OACC's launch-gate token.",
    "against `aeb54468`, the commit DAG-OACC trained at", "RF_r = max over s of |d_r,s(REP − DAG-OACC)|.", "RF0 is unchanged: P-0 is frozen at `raw/2026-10-27-dag-orc-read/P0_RF0.json`, sha256 `1608d70b…`.",
    "The crawl set C** is the set of rows separated-worse for DAG-OACC − P10. At both seeds that is yaw-rate MAE, curvature MAE, L2 and M1.",
    "every Read-A row not separated-worse for DAG-OACC − P10, with no exempt rows.", "R-ADE and R-KEEP now share one contrast**, A − DAG-OACC.", "R-KEEP keeps its point bar (≤ +F/2 = 0.0604 m)",
    "reproduces `DAGOACC_READ.json`'s Read-A table for DAG-OACC − P10 with 0 differences", "the swap is drawn first;", "no sample is ever both swapped and perturbed.",
    "an overlap count of exactly 0. REP reads ego 0 exactly and swap 0.1 ± 0.02.", "Next is B (p 0.1) under this table", "B is not queued.", "HARMS** of A ⇒ B, as written. HARMS of B ⇒ NEXT-CRAWL.",
    "So its branch is LAUNCH-INTENT, with DAG-OACC's recipe in place of \\"DAG init\\".", "524,288-cell enumeration stays complete and exclusive")


def amend_problems(path, pin=None):
    """A1: the file is the pinned AMEND1.md and every coded phrase is verbatim in it.  -> [problems]"""
    pin = AMEND1_SHA256 if pin is None else pin
    if not os.path.isfile(path):
        return [f"{path} is missing"]
    raw = open(path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != pin:
        return [f"{path} is not the pinned AMEND1.md (sha256 {hashlib.sha256(raw).hexdigest()[:12]}..., pinned {pin[:12]}...)"]
    text = re.sub(r"\\s+", " ", raw.decode("utf-8").replace("**", ""))
    return [f"AMEND1: the phrase {ph!r} is not verbatim" for ph in AMEND_PHRASES if re.sub(r"\\s+", " ", ph.replace("**", "")) not in text]


OUTCOME_TABLE_PRE_A1 = tuple(tuple(r) for r in PREREG_CELLS["7."]["rows"][1:])          # (outcome, condition, committed next) as the ORIGINAL PREREG prints them
AMENDED_NEXT = {      # A1 section 3: the committed next of each row, re-based (the conditions are unchanged; "Every other row is unchanged")
    "REFUSED": None,
    "HARMS": "A: not adopted; read B (p 0.1) under this same table (B is NOT queued: it needs its own PI go, A1 section 4). B: not adopted; the perturbation cannot be dosed under the guards; NEXT-CRAWL on DAG-OACC's states (section 8, branch LAUNCH-INTENT with DAG-OACC's recipe)",
    "FIXED / ADOPTED": "adopt X as the base. Re-run Read A on X: DOMINANT there is the dominance claim. (Then X with DAG-OACC's lever is already done.)",
    "FIXED / NOT ADOPTED": "the lever is attributed, but X cannot replace DAG-OACC. DAG-ORC-SWAP cannot apply, because A already keeps the swap on. Next: B (p 0.1) under this table, a lower dose that may keep the base. It needs its own PI go",
    "PARTIAL / ADOPTED": "adopt X. The crawl rows that still fail R-CRAWL go to NEXT-CRAWL on X's states (branch LAUNCH-INTENT with DAG-OACC's recipe)",
    "PARTIAL / NOT ADOPTED": "as FIXED / NOT ADOPTED: B (p 0.1) under this table; DAG-ORC-SWAP cannot apply. It needs its own PI go",
    "PARTIAL / ONE SEED": "not adopted, and never called an effect of the lever (`H-ESTIM-SEED-1`). NEXT-CRAWL on DAG-OACC's states (branch LAUNCH-INTENT with DAG-OACC's recipe)",
    "NO EFFECT": "own-state levers have now failed twice (DAgger iteration 2, then the perturbation composed with the swap). The state channel is eliminated as the crawl's lever. NEXT-CRAWL on DAG-OACC's states (branch LAUNCH-INTENT with DAG-OACC's recipe)",
}
OUTCOME_TABLE = tuple((nm, cond, AMENDED_NEXT[nm] if AMENDED_NEXT[nm] is not None else nxt) for nm, cond, nxt in OUTCOME_TABLE_PRE_A1)      # (outcome, condition, committed next), A1-amended text''')
rep('''def table_row(name):
    return next((c, n) for nm, c, n in OUTCOME_TABLE if nm == name)''',
    '''def table_row(name):
    return next((c, n) for nm, c, n in OUTCOME_TABLE if nm == name)


def table_row_original(name):
    return next(n for nm, c, n in OUTCOME_TABLE_PRE_A1 if nm == name)''')
rep('''    cond, _ = table_row(out)
    return {**d, "arm": arm, "condition": cond, "committed_next": next_for_arm(out, arm)}''',
    '''    cond, _ = table_row(out)
    return {**d, "arm": arm, "condition": cond, "committed_next": next_for_arm(out, arm), "committed_next_original_prereg": table_row_original(out), "amendment": "A1-2026-10-27 (re-based on DAG-OACC)"}''')

# ================================================================ run-record controls (A1)
rep('''def a_flags(ctrl_flags, seed, p=P_A):
    """The flags A must carry: DAG2-0's, plus --ego-perturb-p <p> (replacing its 0) and the five law flags (section 1)."""
    fl = {k: v for k, v in ctrl_flags.items()}
    fl["--ego-perturb-p"] = [[repr(p)]]
    for k, v in LAW_A.items():
        fl[k] = [list(v)]
    fl["--ego-perturb-seed"] = [[str(seed)]]
    return fl''',
    '''def a_flags(ctrl_flags, seed, p=P_A):
    """The flags A must carry (A1): DAG-OACC_s's argv, plus --ego-perturb-p <p> (replacing its 0), the five law flags (section 1) and the composition flag --ego-perturb-with-dagger (a bare flag)."""
    fl = {k: v for k, v in ctrl_flags.items()}
    fl["--ego-perturb-p"] = [[repr(p)]]
    for k, v in LAW_A.items():
        fl[k] = [list(v)]
    fl["--ego-perturb-seed"] = [[str(seed)]]
    fl[COMPOSE_FLAG] = [[]]
    return fl''')
rep('''    """K-PAIR (section 6).  A_s: DAG2-0_s's argv except --out, --ego-perturb-p = 0.3 and the five A-only law flags (values in section 1).  REP_s: except --out and --seed (10 / 11).
    B_s: A_s's argv except --out and --ego-perturb-p = 0.1.  -> (ok, problems)."""''',
    '''    """K-PAIR (section 6, A1).  A_s: DAG-OACC_s's argv except --out, the six perturbation flags at section 1's values (--ego-perturb-p 0.3 + the five law flags) and --ego-perturb-with-dagger.
    REP_s: DAG-OACC_s's argv except --out and --seed (10 / 11).  B_s: A_s's argv except --out and --ego-perturb-p = 0.1.  The bank sha256 is DAG-OACC's.  -> (ok, problems)."""''')
rep('''[f"A_s{seed}'s argv differs from DAG2-0's + the registered law in {d}"] if d else "argv(A) = argv(DAG2-0) + --ego-perturb-p 0.3 + the five law flags")''',
    '''[f"A_s{seed}'s argv differs from DAG-OACC_s{seed}'s + the registered law + {COMPOSE_FLAG} in {d}"] if d else "argv(A) = argv(DAG-OACC) + --ego-perturb-p 0.3 + the five law flags + --ego-perturb-with-dagger")''')
rep('''        want = {k: v for k, v in ctrl_flags.items()}
        want["--seed"] = [[str(REP_SEED[seed])]]
        d = flag_diff(flags, want)
        return not d, ([f"REP_s{seed}'s argv differs from DAG2-0's with --seed {REP_SEED[seed]} in {d}"] if d else f"argv(REP) = argv(DAG2-0) with --seed {REP_SEED[seed]}")''',
    '''        want = {k: v for k, v in ctrl_flags.items()}
        want["--seed"] = [[str(REP_SEED[seed])]]
        d = flag_diff(flags, want)
        return not d, ([f"REP_s{seed}'s argv differs from DAG-OACC_s{seed}'s with --seed {REP_SEED[seed]} in {d}"] if d else f"argv(REP) = argv(DAG-OACC) with --seed {REP_SEED[seed]}")''')
rep('''        bad = [f"B_s{seed}'s argv differs from DAG2-0's + the registered law at p = 0.1 in {d}"] if d else []''',
    '''        bad = [f"B_s{seed}'s argv differs from DAG-OACC_s{seed}'s + the registered law at p = 0.1 + {COMPOSE_FLAG} in {d}"] if d else []''')
rep('''def k_registered(arm, seed, flags):
    """The fine-tune is NAMED on the argv (a default would not be a test): steps, lr, warmup, p-gt, --ego-perturb-p, --dagger-p 0, --arm M2b, --seed, --init-ckpt tail."""
    bad = []
    want_seed = REP_SEED[seed] if arm == "REP" else seed
    p = {"A": P_A, "B": P_B, "REP": 0.0, "CTRL": 0.0}[arm]
    for name, w in (("--steps", STEPS), ("--lr", LR), ("--warmup", WARMUP), ("--p-gt-start", PGT), ("--p-gt-end", PGT), ("--ego-perturb-p", p), ("--dagger-p", 0.0), ("--seed", want_seed)):''',
    '''def k_registered(arm, seed, flags):
    """The fine-tune is NAMED on the argv (a default would not be a test) (A1: DAG-OACC's recipe): steps, lr, warmup, p-gt, --ego-perturb-p, --dagger-p 0.1, --w-acc 49.6969, --dagger-bank-sha256 = DAG-OACC's,
    --arm M2b, --seed, --init-ckpt tail M2b_s<s>/ckpt_last.pt, and the composition flag: named once on A and B, absent on REP and CTRL."""
    bad = []
    want_seed = REP_SEED[seed] if arm == "REP" else seed
    p = {"A": P_A, "B": P_B, "REP": 0.0, "CTRL": 0.0}[arm]
    ok_w, v_w = one_value(flags, "--w-acc")
    if not ok_w or v_w != W_ACC:
        bad.append(f"--w-acc {v_w!r}, registered {W_ACC} (DAG-OACC's, verbatim)")
    ok_b, v_b = one_value(flags, "--dagger-bank-sha256")
    if not ok_b or v_b != OACC_BANK_SHA[seed]:
        bad.append(f"--dagger-bank-sha256 {v_b!r} is not DAG-OACC's {OACC_BANK_SHA[seed][:16]}...")
    ok_d, v_d = one_value(flags, "--dagger-seed")
    if not ok_d or v_d != str(seed):
        bad.append(f"--dagger-seed {v_d!r}, registered {seed} (DAG-OACC's)")
    has_flag = COMPOSE_FLAG in flags
    if arm in ("A", "B"):
        if not (has_flag and flags[COMPOSE_FLAG] == [[]]):
            bad.append(f"{COMPOSE_FLAG} is not named exactly once as a bare flag on {arm} (it has {flags.get(COMPOSE_FLAG)})")
    elif has_flag:
        bad.append(f"{COMPOSE_FLAG} is named on {arm}: it must be absent")
    for name, w in (("--steps", STEPS), ("--lr", LR), ("--warmup", WARMUP), ("--p-gt-start", PGT), ("--p-gt-end", PGT), ("--ego-perturb-p", p), ("--dagger-p", SWAP_P), ("--seed", want_seed)):''')
rep('''    if not ok or not str(v).replace("\\\\", "/").endswith(f"M2b_s{seed}_dag/ckpt_last.pt"):
        bad.append(f"--init-ckpt {v!r} is not .../M2b_s{seed}_dag/ckpt_last.pt (the init is DAG_s{seed})")''',
    '''    if not ok or not str(v).replace("\\\\", "/").endswith(f"/M2b_s{seed}/ckpt_last.pt"):
        bad.append(f"--init-ckpt {v!r} is not .../M2b_s{seed}/ckpt_last.pt (the init is M2b_s{seed}, DAG-OACC's)")''')
rep('''def k_init(arm, seed, init_values):
    """K-INIT: every recorded init-checkpoint sha256 equals DAG_s's.  ``init_values`` = [(where, sha)] from dag_read.init_sha_values."""
    if not init_values:
        return False, "no record of the init checkpoint's sha256: the init is unproven"
    bad = [f"{w} = {str(v)[:16]}" for w, v in init_values if str(v) != DAG_CKPT_SHA[seed]]
    return not bad, (f"init sha256 differs from DAG_s{seed} ({DAG_CKPT_SHA[seed][:16]}...): " + "; ".join(bad)) if bad else {"init_sha256": DAG_CKPT_SHA[seed][:16] + "...", "sources": [w for w, _ in init_values]}''',
    '''def k_init(arm, seed, init_values, want=None):
    """K-INIT (A1): every recorded init-checkpoint sha256 equals M2b_s's, as recorded in DAG-OACC's launch-gate token (``want`` = the rc-DAG2 leg's DAG_CKPT_SHA).  ``init_values`` = [(where, sha)] from dag_read.init_sha_values."""
    want = M2B_CKPT_SHA if want is None else want
    if not init_values:
        return False, "no record of the init checkpoint's sha256: the init is unproven"
    bad = [f"{w} = {str(v)[:16]}" for w, v in init_values if str(v) != want[seed]]
    return not bad, (f"init sha256 differs from the registered init of s{seed} ({want[seed][:16]}...): " + "; ".join(bad)) if bad else {"init_sha256": want[seed][:16] + "...", "sources": [w for w, _ in init_values]}''')
rep('''def k_reach(arm, built_p, n, eligible, nav_kept=None):
    """K-REACH: the lever REACHED training.  ego_perturb_n / ego_perturb_eligible lies in 0.3 +- 0.02 for A, 0.1 +- 0.02 for B, and is exactly 0 for REP (n == 0); the built p equals the argv p."""
    want = {"A": P_A, "B": P_B, "REP": 0.0}[arm]
    bad, det = [], {"ego_perturb_n": n, "ego_perturb_eligible": eligible, "ego_perturb_nav_kept": nav_kept}''',
    '''def k_reach(arm, built_p, n, eligible, nav_kept=None, dag=None):
    """K-REACH (A1): BOTH levers REACHED training.  ``dag`` = {"dagger_n", "dagger_eligible", "ego_dagger_overlap_n", "ego_perturb_with_dagger"} (the trainer's run record).
      A: swap dagger_n / dagger_eligible in 0.1 +- 0.02 AND ego_perturb_n / ego_perturb_eligible (non-swapped eligible) in 0.3 +- 0.02 AND overlap exactly 0 AND the built composition flag True.
      B: the same with ego 0.1.   REP: swap in 0.1 +- 0.02 AND ego_perturb_n exactly 0 AND the composition flag absent (built False / None).   The built p equals the argv p."""
    want = {"A": P_A, "B": P_B, "REP": 0.0}[arm]
    bad, det = [], {"ego_perturb_n": n, "ego_perturb_eligible": eligible, "ego_perturb_nav_kept": nav_kept}
    dg = dag or {}
    det["dag"] = dict(dg)
    dn, de = dg.get("dagger_n"), dg.get("dagger_eligible")
    if not (isinstance(dn, (int, float)) and not isinstance(dn, bool) and isinstance(de, (int, float)) and not isinstance(de, bool) and de > 0 and 0 <= dn <= de):
        bad.append(f"dagger_n {dn!r} / dagger_eligible {de!r}: not a count pair with 0 <= n <= eligible, eligible > 0")
    else:
        sr = dn / de
        det["swap_ratio"] = sr
        if not (SWAP_P - REACH_TOL - 1e-12 <= sr <= SWAP_P + REACH_TOL + 1e-12):
            bad.append(f"{arm} swapped fraction {dn}/{de} = {sr:.4f} outside {SWAP_P} +- {REACH_TOL}")
    ov, wf = dg.get("ego_dagger_overlap_n"), dg.get("ego_perturb_with_dagger")
    if arm in ("A", "B"):
        if not (isinstance(ov, int) and not isinstance(ov, bool) and ov == 0):
            bad.append(f"ego_dagger_overlap_n {ov!r} must be exactly 0 (no sample is both swapped and perturbed)")
        if wf is not True:
            bad.append(f"built ego_perturb_with_dagger {wf!r} must be True on {arm}")
    else:
        if wf not in (False, None):
            bad.append(f"built ego_perturb_with_dagger {wf!r} on REP: the flag must be absent (False / None)")
        if ov not in (0, None):
            bad.append(f"ego_dagger_overlap_n {ov!r} on REP must be 0 / absent")''')
rep('''    """K-BUILD: the launch commit's v8f_mini_train.py, v8f_mini_data.py and v8f_mini.py are byte-identical to 1d4f4fbb's, or every changed hunk sits behind a default-off flag whose bit-identity
    test passes on the launch commit.''',
    '''    """K-BUILD (A1): the launch commit's v8f_mini_train.py, v8f_mini_data.py, v8f_mini.py (and v8f_dagger.py) are byte-identical to aeb54468's (the commit DAG-OACC trained at), or every changed hunk sits behind a
    default-off flag (--ego-perturb-with-dagger) whose bit-identity test (flag OFF) passes on the launch commit.''')
rep('''def floors_from(tables):
    """RF_r = max over seeds of |d_r,s(REP - CTRL)|; tables[s] = {label: row}."""''',
    '''def floors_from(tables):
    """RF_r = max over seeds of |d_r,s(REP - CTRL)|; tables[s] = {label: row}.  A1: CTRL = DAG-OACC, so this is the amendment's RF_r = max over s of |d_r,s(REP - DAG-OACC)|."""''')

# ================================================================ slot discovery (A1: CTRL is the banked DAG-OACC run: t1 + t0, no logged lane, no fresh T0)
rep('''def slot_needs(arm, e):
    """The files one slot entry must name and have on disk.  CTRL and DAG are banked runs; A / REP / B are new."""
    need = []
    if arm != "DAG":''',
    '''def slot_needs(arm, e, light=("CTRL",)):
    """The files one slot entry must name and have on disk.  CTRL (= DAG-OACC, A1) is a banked run: its T1-kin lane, box 2's T0 eval and its train records; it has no --limit logged lane and no fresh T0 (K4 is for the
    new arms).  ``light`` = the slots read that way; the rc-DAG2 leg passes () and the old DAG slot keeps its train_dir + t1 only."""
    need = []
    if arm in light:
        td = e.get("train_dir")
        need += [("train_dir", td)] + [(f"train_dir/{f}", os.path.join(td, f) if td else None) for f in ("argv.json", "summary.json", "selfcheck.json")]
        need.append(("token", e.get("token")))
        for f in ("json", "npz"):
            need.append((f"t0.{f}", (e.get("t0") or {}).get(f)))
        lanes = ("t1",)
    elif arm != "DAG":''')
rep('''def discover(manifest, arms=ARMS, seeds=SEEDS):''', '''def discover(manifest, arms=ARMS, seeds=SEEDS, light=("CTRL",)):''')
rep('''            for what, p in slot_needs(arm, e):''', '''            for what, p in slot_needs(arm, e, light):''')

# ================================================================ the PREREG + amendment + banked pins are static (A1)
rep('''    bad += [f"PREREG: {p}" for p in prereg_problems(a.prereg)]
    pr, _ = rf0_problems(a.rf0)
    bad += [f"RF0: {p}" for p in pr]
    return bad''',
    '''    bad += [f"PREREG: {p}" for p in prereg_problems(a.prereg)]
    bad += amend_problems(a.amend1)
    pr, _ = rf0_problems(a.rf0)
    bad += [f"RF0: {p}" for p in pr]
    bad += banked_pin_problems()
    return bad


def banked_pin_problems():
    """A1: the checkpoint / init / bank / argv pins coded above equal what the BANKED DAG-OACC artifacts say (SHA256SUMS.txt, the launch-gate token, summary.json) and DAGOACC_READ.json is the pinned file."""
    bad = []
    oacc = f"{ART}/v8f-dagoacc"
    for s in SEEDS:
        d = f"{oacc}/M2b_s{s}_dagoacc"
        try:
            sums = {ln.split(" ", 1)[1].strip().lstrip("*"): ln.split(" ", 1)[0] for ln in open(f"{d}/SHA256SUMS.txt", encoding="utf-8").read().splitlines() if ln.strip()}
            tok = json.load(open(f"{d}/launch_gate_token.json", encoding="utf-8"))
            sm = json.load(open(f"{d}/summary.json", encoding="utf-8"))
        except (OSError, ValueError, IndexError) as e:
            bad.append(f"DAG-OACC_s{s} banked artifacts unreadable: {e}")
            continue
        for what, got, want in (("ckpt_last.pt", sums.get("ckpt_last.pt"), OACC_CKPT_SHA[s]), ("token init_ckpt_sha256", tok.get("init_ckpt_sha256"), M2B_CKPT_SHA[s]), ("token bank_sha256", tok.get("bank_sha256"), OACC_BANK_SHA[s]),
                                ("summary argv_sha", sm.get("argv_sha"), OACC_ARGV_SHA16[s]), ("token commit[:8]", str(tok.get("binding", {}).get("commit"))[:8], BUILD_BASE_COMMIT)):
            if got != want:
                bad.append(f"DAG-OACC_s{s} {what}: banked {str(got)[:16]}, the reader pins {str(want)[:16]}")
        if sums.get("summary.json") and sha256_file(f"{d}/summary.json") != sums["summary.json"]:
            bad.append(f"DAG-OACC_s{s} summary.json != its SHA256SUMS.txt entry")
    p = f"{ART}/v8f-dagoacc-t1/read/DAGOACC_READ.json"
    if not os.path.isfile(p) or sha256_file(p) != OACC_READ_SHA256:
        bad.append(f"{p} is missing or is not the pinned DAGOACC_READ.json ({OACC_READ_SHA256[:12]}...)")
    return bad''')
rep('''            t1kf_banked=f"{ART}/dagreread/raw/t1kf_DAG/T1KF_READ_A1_DAG.json", prereg=f"{HERE}/PREREG_V8F_DAG_ORC.md", rf0=f"{HERE}/P0_RF0.json", work_dir=None, n_boot=N_BOOT)''',
    '''            t1kf_banked=f"{ART}/dagreread/raw/t1kf_DAG/T1KF_READ_A1_DAG.json", prereg=f"{HERE}/PREREG_V8F_DAG_ORC.md", amend1=f"{HERE}/AMEND1.md", rf0=f"{HERE}/P0_RF0.json", work_dir=None, n_boot=N_BOOT)''')
rep('''    print(json.dumps({"problems": bad, "prereg_sha256": PREREG_SHA256, "rf0_sha256": RF0_SHA256}, indent=1))''',
    '''    print(json.dumps({"problems": bad, "prereg_sha256": PREREG_SHA256, "amend1_sha256": AMEND1_SHA256, "rf0_sha256": RF0_SHA256}, indent=1))''')

# ================================================================ the dry run (A1: CTRL = the banked DAG-OACC)
rep('''def banked_ctrl_dag_manifest():
    """The dry-run manifest: CTRL = DAG2-0 (v8f-dagger2 + v8f-dagger2-t1), DAG = DAG (v8f-dagger + v8f-dagger-t1), both banked; A / REP / B absent (they do not exist yet)."""
    m2 = json.load(open(f"{ART}/v8f-dagger2-t1/runs.json", encoding="utf-8"))
    m1 = json.load(open(f"{ART}/v8f-dagger-t1/runs.json", encoding="utf-8"))
    return {"seeds": [0, 1], "arms": {"CTRL": m2["arms"]["DAG0"], "DAG": m1["arms"]["DAG"]}, "provenance": "dry-run manifest: CTRL := DAG2-0 (banked), DAG := DAG (banked); A / REP / B absent"}''',
    '''def banked_ctrl_dag_manifest():
    """The dry-run manifest (A1): CTRL = DAG-OACC (v8f-dagoacc + v8f-dagoacc-t1, banked); A / REP / B absent (they do not exist yet)."""
    m = json.load(open(f"{ART}/v8f-dagoacc-t1/runs.json", encoding="utf-8"))
    return {"seeds": [0, 1], "arms": {"CTRL": m["arms"]["DAGOACC"]}, "provenance": "dry-run manifest: CTRL := DAG-OACC (banked, v8f-dagoacc-t1/runs.json arms.DAGOACC); A / REP / B absent"}


def rc_oacc_manifest():
    """K-RC leg 2's slots (A1): CTRL := DAG-OACC (the banked run)."""
    return banked_ctrl_dag_manifest()''')
rep('''    ctl.add("SLOTS-BANKED", present["CTRL"] and present["DAG"], {"CTRL": slot_summary(slots)["CTRL"], "DAG": slot_summary(slots)["DAG"]})
    # CTRL's run records through the same controls a new run gets (summary, init, gate token, registered fine-tune)''',
    '''    ctl.add("SLOTS-BANKED", present["CTRL"], {"CTRL": slot_summary(slots)["CTRL"]})
    # CTRL's (= DAG-OACC's) run records through the same controls a new run gets (summary, init, gate token, registered fine-tune)''')
rep('''        ctl.add(f"CTRL-CKPT:{tag}", bool(sha and sha.startswith(CTRL_CKPT_PREFIX[s])), {"ckpt_sha256": (sha or "")[:16], "PREREG": CTRL_CKPT_PREFIX[s]})''',
    '''        ctl.add(f"CTRL-CKPT:{tag}", bool(sha and sha == OACC_CKPT_SHA[s]), {"ckpt_sha256": (sha or "")[:16], "A1 (DAG-OACC)": OACC_CKPT_SHA[s][:16]})''')
rep('''                "A / REP / B argv built from DAG2-0's banked argv pass K-PAIR; DAG2-0's own argv does not pass as A")''',
    '''                "A / REP / B argv built from DAG-OACC's banked argv pass K-PAIR; DAG-OACC's own argv does not pass as A")
        ctl.add(f"K-REGISTERED-selfcheck:{tag}", k_registered("CTRL", s, rec["flags"])[0] and k_registered("A", s, a_flags(rec["flags"], s, P_A))[0]
                and k_registered("REP", s, {**rec["flags"], "--seed": [[str(REP_SEED[s])]]})[0] and not k_registered("A", s, rec["flags"])[0]
                and not k_registered("REP", s, a_flags(rec["flags"], s, P_A))[0], "the registered-argv control passes DAG-OACC / A / REP built from its banked argv and refuses A without the composition flag and REP with it")''')
rep('''"prereg": PREREG_ID, "prereg_sha256": PREREG_SHA256, "rf0_sha256": RF0_SHA256, "reader_sha256": sha256_file(os.path.abspath(__file__))}
    ok = not ctl.failed() and all(''',
    '''"prereg": PREREG_ID, "prereg_sha256": PREREG_SHA256, "amend1_sha256": AMEND1_SHA256, "rf0_sha256": RF0_SHA256, "reader_sha256": sha256_file(os.path.abspath(__file__))}
    ok = not ctl.failed() and all(''')

# ================================================================ analyze_slot: the banked DAG-OACC base is read light; the init map is a parameter
rep('''def analyze_slot(rig, ctl, arm, seed, e, records):
    """One run through the readers' own controls.  ``records`` = 'full' (A / REP / B: K-GATE, K-INIT, the registered argv) | 'basic' (summary, init, gate token only) | 'none' (the banked DAG run).
    -> rec {tag, entry, pack, dump (T1-HEADS loader), t0 (arrays | None)}; adds every control to ``ctl``."""''',
    '''def analyze_slot(rig, ctl, arm, seed, e, records, light=False):
    """One run through the readers' own controls.  ``records`` = 'full' (A / REP / B: K-GATE, K-INIT, the registered argv) | 'basic' (summary, init, gate token only) | 'none' (the banked DAG run).
    A1: ``light`` = the banked DAG-OACC base (CTRL): basic records + the registered-argv control, K-INIT against M2b_s, the T1-kin lane and box 2's T0 eval; no logged lane, no K4, no fresh T0 (none exist for it).
    -> rec {tag, entry, pack, dump (T1-HEADS loader), t0 (arrays | None)}; adds every control to ``ctl``."""''')
rep('''        ctl.run(f"K-INIT:{tag}", lambda: k_init(arm, seed, DR.init_sha_values(tr, KEY_INIT)))
        ctl.run(f"K-GATE:{tag}", lambda: DR.g_token(tag, tr, DR.read_token(e.get("token"))))
        if records == "full":''',
    '''        ctl.run(f"K-INIT:{tag}", lambda: k_init(arm, seed, DR.init_sha_values(tr, KEY_INIT), None if (light or records == "full") else DAG_CKPT_SHA))
        ctl.run(f"K-GATE:{tag}", lambda: DR.g_token(tag, tr, DR.read_token(e.get("token"))))
        if light:
            ctl.run(f"ARGV:{tag}", lambda: k_registered(arm, seed, tr["flags"]))
        if records == "full":''')
rep('''                    ns, es, nk = DR.find_key(tr["sources"], "ego_perturb_n"), DR.find_key(tr["sources"], "ego_perturb_eligible"), DR.find_key(tr["sources"], "ego_perturb_nav_kept")
                    for what, found in (("ego_perturb_p", bp), ("ego_perturb_n", ns), ("ego_perturb_eligible", es)):
                        if not found:
                            return False, f"no built value {what!r} recorded by the trainer"
                        if len({json.dumps(v) for _, v in found}) > 1:
                            return False, f"{what!r} recorded with conflicting values {found}"
                    return k_reach(arm, bp[0][1], ns[0][1], es[0][1], nk[0][1] if nk else None)''',
    '''                    ns, es, nk = DR.find_key(tr["sources"], "ego_perturb_n"), DR.find_key(tr["sources"], "ego_perturb_eligible"), DR.find_key(tr["sources"], "ego_perturb_nav_kept")
                    for what, found in (("ego_perturb_p", bp), ("ego_perturb_n", ns), ("ego_perturb_eligible", es)):
                        if not found:
                            return False, f"no built value {what!r} recorded by the trainer"
                        if len({json.dumps(v) for _, v in found}) > 1:
                            return False, f"{what!r} recorded with conflicting values {found}"
                    dg = {}
                    for k in ("dagger_n", "dagger_eligible", "ego_dagger_overlap_n", "ego_perturb_with_dagger"):      # A1: the composed run record (top level or levers / built_levers / totals, DR.find_key)
                        found = DR.find_key(tr["sources"], k)
                        if len({json.dumps(v) for _, v in found}) > 1:
                            return False, f"{k!r} recorded with conflicting values {found}"
                        dg[k] = found[0][1] if found else None
                    return k_reach(arm, bp[0][1], ns[0][1], es[0][1], nk[0][1] if nk else None, dg)''')
rep('''    if arm != "DAG":
        lg = e.get("t1_logged")''', '''    if arm != "DAG" and not light:
        lg = e.get("t1_logged")''')
rep('''    d = rec["dump"]
    ctl.add(f"DUMP-WINDOWS:{tag}"''', '''    if light:      # box 2's standard T0 eval (R-T0's comparator); no fresh-age eval exists for the banked base
        t0e = e.get("t0")
        if not t0e:
            raise DR.Refusal(f"{tag}: manifest has no 't0' (box 2's standard T0 eval of the banked base)")
        rec["t0json"], rec["t0arr"], t0_ok = DR.t0_arrays(t0e, ckpt_tail)
        ctl.add(f"T0-CKPT:{tag}", t0_ok, f"T0 eval json ckpt {rec['t0json'].get('ckpt')!r} vs .../{ckpt_tail}")
    d = rec["dump"]
    ctl.add(f"DUMP-WINDOWS:{tag}"''')

# ================================================================ read_core
rep('''    """The whole analysis.  mode 'read': arms A, CTRL, REP, DAG (+ B when present); mode 'rc': A := DAG2, CTRL := DAG2-0, DAG := DAG, floors 0, no REP.
    -> dict of tables, G rows, floors, per-arm compose and decisions (nothing is written here)."""
    DR, A, T, K, np, nb = rig.DR, rig.A, rig.T, rig.K, rig.np, rig.a.n_boot
    rc = mode == "rc"
    xs = ["A"] + (["B"] if (not rc and all(((man["arms"].get("B") or {}).get(str(s))) for s in SEEDS)) else [])
    arms = xs + ["CTRL", "DAG"] + ([] if rc else ["REP"])
    recs = {arm: {} for arm in arms}
    for arm in arms:
        for s in SEEDS:
            e = (man["arms"].get(arm) or {}).get(str(s))
            if not e:
                raise DR.Refusal(f"manifest has no {arm} seed {s}")
            recs[arm][s] = analyze_slot(rig, ctl, arm, s, e, "none" if arm == "DAG" else ("basic" if rc else "full"))''',
    '''    """The whole analysis.  mode 'read' (A1): arms A, CTRL (= DAG-OACC, banked, light), REP (+ B when present); X - DAG is X - CTRL (one slot, aliased).  mode 'rc': A := DAG2, CTRL := DAG2-0, DAG := DAG, floors 0,
    no REP (the DAG2 leg of K-RC, unchanged).  mode 'rc-oacc': CTRL := DAG-OACC only, returns DAG-OACC - P10 (K-RC leg 2).
    -> dict of tables, G rows, floors, per-arm compose and decisions (nothing is written here)."""
    DR, A, T, K, np, nb = rig.DR, rig.A, rig.T, rig.K, rig.np, rig.a.n_boot
    rc = mode == "rc"
    ro = mode == "rc-oacc"
    xs = [] if ro else (["A"] + (["B"] if (not rc and all(((man["arms"].get("B") or {}).get(str(s))) for s in SEEDS)) else []))
    arms = ["CTRL"] if ro else (xs + ["CTRL", "DAG"] if rc else xs + ["CTRL", "REP"])
    recs = {arm: {} for arm in arms}
    for arm in arms:
        for s in SEEDS:
            e = (man["arms"].get(arm) or {}).get(str(s))
            if not e:
                raise DR.Refusal(f"manifest has no {arm} seed {s}")
            recs[arm][s] = analyze_slot(rig, ctl, arm, s, e, "none" if arm == "DAG" else ("basic" if (rc or ro or arm == "CTRL") else "full"), light=(not rc and arm == "CTRL"))''')
rep('''        ctl.add(f"CTRL-CKPT:s{s}", bool(sha and (rc or sha.startswith(CTRL_CKPT_PREFIX[s]))), {"ckpt_sha256": (sha or "")[:16], "PREREG": CTRL_CKPT_PREFIX[s]})''',
    '''        ctl.add(f"CTRL-CKPT:s{s}", bool(sha and (rc or sha == OACC_CKPT_SHA[s])), {"ckpt_sha256": (sha or "")[:16], "A1 (DAG-OACC)": OACC_CKPT_SHA[s][:16]})''')
rep('''        if not rc:
            ctrl_fl = recs["CTRL"][s]["train"]["flags"]''', '''        if not rc and not ro:
            ctrl_fl = recs["CTRL"][s]["train"]["flags"]''')
rep('''    for s in SEEDS:                                     # P10's dump = CTRL's `fl`
        dumps[s]["P10"] = dumps[s]["CTRL"]
''', '''    for s in SEEDS:                                     # P10's dump = CTRL's `fl`
        dumps[s]["P10"] = dumps[s]["CTRL"]
        if not rc:                                      # A1: DAG := CTRL := DAG-OACC, ONE slot
            dumps[s]["DAG"] = dumps[s]["CTRL"]
''')
rep('''    # ---- PHASE 1, the floors (section 3: RF "after REP's rollouts, before any A or B row is read"): REP - CTRL, then the file, BEFORE any X contrast is computed
    rf0z = {l: 0.0 for l in LABELS}''', '''    if ro:      # K-RC leg 2: DAG-OACC - P10 through the same pair machinery
        return {"tabs": {"CTRL_P10": pair_tab("CTRL", "P10")}, "recs": recs, "g": {}, "xs": [], "floors": {}}
    # ---- PHASE 1, the floors (section 3: RF "after REP's rollouts, before any A or B row is read"): REP - CTRL, then the file, BEFORE any X contrast is computed
    rf0z = {l: 0.0 for l in LABELS}''')
rep('''    for x in xs:
        for k, (p, q) in ((f"{x}_P10", (x, "P10")), (f"{x}_CTRL", (x, "CTRL")), (f"{x}_DAG", (x, "DAG"))):
            tabs[k] = pair_tab(p, q)
    tabs["DAG_P10"], tabs["CTRL_P10"] = pair_tab("DAG", "P10"), pair_tab("CTRL", "P10")
    d_ = [domain(tabs["DAG_P10"][s_]) for s_ in SEEDS]
    ctl.add("DOMAIN-COUNT", (len(d_[0]), len(d_[1])) == (DOMAIN_N[0], DOMAIN_N[1]), {"n_s0": len(d_[0]), "n_s1": len(d_[1]), "PREREG": DOMAIN_N})''',
    '''    for x in xs:
        for k, (p, q) in ((f"{x}_P10", (x, "P10")), (f"{x}_CTRL", (x, "CTRL"))) + (((f"{x}_DAG", (x, "DAG")),) if rc else ()):
            tabs[k] = pair_tab(p, q)
        if not rc:                                      # A1: X - DAG-OACC IS X - CTRL (R-ADE / R-KEEP and R-NEW / R-NOBASE share the contrast): one table, two names
            tabs[f"{x}_DAG"] = tabs[f"{x}_CTRL"]
    tabs["CTRL_P10"] = pair_tab("CTRL", "P10")
    tabs["DAG_P10"] = pair_tab("DAG", "P10") if rc else tabs["CTRL_P10"]
    d_ = [domain(tabs["DAG_P10"][s_]) for s_ in SEEDS]
    dn_ = DOMAIN_N_PRE_A1 if rc else DOMAIN_N
    ctl.add("DOMAIN-COUNT", (len(d_[0]), len(d_[1])) == (dn_[0], dn_[1]), {"n_s0": len(d_[0]), "n_s1": len(d_[1]), "registered": dn_})
    if not rc:      # A1: C_s is the set of rows separated-worse for DAG-OACC - P10 (computed here from the live table, compared with the coded set)
        cs_ = {s_: tuple(l for l in LABELS if sep_worse(l, tabs["CTRL_P10"][s_][l])) for s_ in SEEDS}
        ctl.add("CRAWL-SET", all(set(cs_[s_]) == set(CRAWL_SET[s_]) for s_ in SEEDS), {"computed": {f"s{s_}": list(cs_[s_]) for s_ in SEEDS}, "registered (A1)": {f"s{s_}": list(CRAWL_SET[s_]) for s_ in SEEDS}})''')

# compose_seed must take the crawl set / exemption of its leg
rep('''def compose_seed(s, t, g, rf0, rf):
    """The PREREG's per-seed composition from the readers' rows.''', '''def compose_seed(s, t, g, rf0, rf, pre_a1=False):
    """The PREREG's per-seed composition from the readers' rows (A1: C_s, the domain and the exemption are DAG-OACC's; ``pre_a1`` = the rc-DAG2 leg, composed with the ORIGINAL section 4).''')
rep('''    c = CRAWL_SET[s]
    crawl_rows = {l: not sep_worse(l, t["X_P10"][l]) for l in c}''', '''    c = (CRAWL_SET_PRE_A1 if pre_a1 else CRAWL_SET)[s]
    exempt_ = EXEMPT_PRE_A1 if pre_a1 else EXEMPT
    crawl_rows = {l: not sep_worse(l, t["X_P10"][l]) for l in c}''')
rep('''    d_s = domain(t["DAG_P10"])
    fire_rows''', '''    d_s = domain(t["DAG_P10"], exempt_)
    fire_rows''')
rep('''def domain(dag_p10):
    """D_s: every Read-A row not separated-worse for DAG - P10 at s, minus the exempt rows."""
    return [l for l in LABELS if l not in EXEMPT and not sep_worse(l, dag_p10[l])]''',
    '''def domain(dag_p10, exempt=None):
    """D_s: every Read-A row not separated-worse for DAG-OACC - P10 at s (A1: ``exempt`` = () , no exempt rows; the rc-DAG2 leg passes the original pair)."""
    exempt = EXEMPT if exempt is None else exempt
    return [l for l in LABELS if l not in exempt and not sep_worse(l, dag_p10[l])]''')
rep('''                  "X_minus_CTRL_separated_worse": sep_worse(l, t["X_CTRL"][l])} for l in EXEMPT}''', '''                  "X_minus_CTRL_separated_worse": sep_worse(l, t["X_CTRL"][l])} for l in exempt_}''')
rep('''        cs = compose_seed(s, t, res["g"]["A"][s], {l: 0.0 for l in LABELS}, {l: 0.0 for l in LABELS})''', '''        cs = compose_seed(s, t, res["g"]["A"][s], {l: 0.0 for l in LABELS}, {l: 0.0 for l in LABELS}, pre_a1=True)''')
rep('''        nw_mine = sorted(l for l in cs["domain"] if sep_worse(l, t["X_P10"][l]))''', '''        nw_mine = sorted(l for l in cs["domain"] if sep_worse(l, t["X_P10"][l]))      # (the DAG2 leg: ORIGINAL domain)''')

# families + provenance + output
rep('''    return {f"s{s}": families_report({"X-CTRL": res["tabs"][f"{x}_CTRL"][s], "X-P10": res["tabs"][f"{x}_P10"][s], "X-DAG": res["tabs"][f"{x}_DAG"][s]}) for s in SEEDS}''',
    '''    return {f"s{s}": families_report({"X-CTRL (= X-DAG-OACC, A1)": res["tabs"][f"{x}_CTRL"][s], "X-P10": res["tabs"][f"{x}_P10"][s]}) for s in SEEDS}''')
rep('''    return {"prereg": PREREG_ID, "prereg_sha256": PREREG_SHA256, "claim_row": CLAIM_ROW, "rf0_sha256": RF0_SHA256,''', '''    return {"prereg": PREREG_ID, "prereg_sha256": PREREG_SHA256, "amend1_sha256": AMEND1_SHA256, "dagoacc_read_sha256": OACC_READ_SHA256, "claim_row": CLAIM_ROW, "rf0_sha256": RF0_SHA256,''')
rep('''    ctl.run("K-RC", lambda: rc_evidence_ok(rc_evidence))''', '''    ctl.run("K-RC-DAG2", lambda: rc_evidence_ok(rc_evidence))
    ctl.run("K-RC-OACC", lambda: rc_evidence_ok(rc_oacc_evidence, oacc=True))''')
rep('''def main_read(a, manifest_path, out_path, rc_evidence):''', '''def main_read(a, manifest_path, out_path, rc_evidence, rc_oacc_evidence=None):''')
rep('''           "perturbation_counts": {f"{arm}_s{s}": {k: (DR.find_key(res["recs"][arm][s]["train"]["sources"], k) or [(None, None)])[0][1] for k in ("ego_perturb_n", "ego_perturb_eligible", "ego_perturb_nav_kept")}''',
    '''           "perturbation_counts": {f"{arm}_s{s}": {k: (DR.find_key(res["recs"][arm][s]["train"]["sources"], k) or [(None, None)])[0][1] for k in ("ego_perturb_n", "ego_perturb_eligible", "ego_perturb_nav_kept", "dagger_n", "dagger_eligible", "ego_dagger_overlap_n", "ego_perturb_with_dagger")}''')

# ================================================================ K-RC: two legs
rep('''def rc_evidence_ok(path):
    """K-RC: the evidence file of a PASSing `--rc-dag2` run exists AND was produced by THIS reader (its recorded reader_sha256 equals this file's)."""
    if not os.path.isfile(path):
        return False, f"{path} is missing: run `dag_orc_read.py --rc-dag2` first"''',
    '''def rc_evidence_ok(path, oacc=False):
    """K-RC (two legs, A1): the evidence file of a PASSing `--rc-dag2` (oacc=False) or `--rc-oacc` (oacc=True) run exists AND was produced by THIS reader (its recorded reader_sha256 equals this file's)."""
    if not path or not os.path.isfile(path):
        return False, f"{path} is missing: run `dag_orc_read.py {'--rc-oacc' if oacc else '--rc-dag2'}` first"''')
rep('''def main_rc(a, out_path):''', '''def main_rc_oacc(a, out_path):
    """K-RC leg 2 (A1): this reader's slot / pair machinery with DAG-OACC in the base slot (CTRL) and the floors at 0 reproduces DAGOACC_READ.json's Read-A table for DAG-OACC - P10 (all 19 rows, both seeds) with 0
    differences, and C_s / |D_s| as the amendment states them."""
    t_start = time.time()
    D2, DR = mods()
    bad = static_problems(a)
    if bad:
        print("K-RC (oacc) REFUSED:", bad, file=sys.stderr)
        return 2
    pth = f"{ART}/v8f-dagoacc-t1/read/DAGOACC_READ.json"
    banked = json.load(open(pth, encoding="utf-8"))
    ctl = DR.Ctl()
    rig = Rig(a)
    res = read_core(rig, ctl, rc_oacc_manifest(), "rc-oacc", None)
    diffs, detail = [], {}
    for s in SEEDS:
        ref = banked["rows"][f"s{s}"]["READ_A"]["table_DAGOACC_minus_P10"]
        d = rc_diffs(res["tabs"]["CTRL_P10"][s], ref)
        detail[f"DAGOACC_minus_P10.s{s}"] = {"rows": len(ref), "differences": d}
        diffs += [f"DAGOACC_minus_P10.s{s}: {x}" for x in d]
        sw = [l for l in LABELS if sep_worse(l, res["tabs"]["CTRL_P10"][s][l])]
        bsw = banked["rows"][f"s{s}"]["READ_A"]["lateral_or_crawl_rows_separated_worse"]
        d2 = ([f"C_s (separated-worse rows) {sw} vs the banked {sorted(bsw)}"] if sorted(sw) != sorted(bsw) else []) + ([f"C_s {sw} vs the amendment's {list(CRAWL_SET[s])}"] if sorted(sw) != sorted(CRAWL_SET[s]) else [])
        dn = len(domain(res["tabs"]["CTRL_P10"][s]))
        d2 += [f"|D_s| {dn} vs the amendment's {DOMAIN_N[s]}"] if dn != DOMAIN_N[s] else []
        detail[f"crawl_and_domain.s{s}"] = {"C_s": sw, "domain_n": dn, "differences": d2}
        diffs += [f"s{s} {x}" for x in d2]
    out = {"K-RC": "A1 leg 2: this reader's analysis with DAG-OACC in the base slot, floors 0, vs DAGOACC_READ.json's DAG-OACC - P10 Read-A table", "PASS": not diffs and not ctl.failed(), "differences": len(diffs),
           "difference_list": diffs[:20], "detail": detail, "controls_failed": {k: str(v["detail"])[:200] for k, v in ctl.failed().items()}, "controls_n": len(ctl.items), "reader_sha256": sha256_file(os.path.abspath(__file__)),
           "banked_sha256": sha256_file(pth), "banked_path": pth, "wall_s": round(time.time() - t_start, 1)}
    with open(out_path, "w", encoding="utf-8", newline="\\n") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: out[k] for k in ("PASS", "differences", "difference_list", "controls_failed", "controls_n")}, indent=1))
    print(f"wrote {out_path}  sha256 {sha256_file(out_path)}")
    return 0 if out["PASS"] else 1


def main_rc(a, out_path):''')
rep('''    man["arms"]["REP"], man["arms"]["B"] = man["arms"]["CTRL"], man["arms"]["A"]
    pr, rf0, _ = k_p0(a.rf0, [], check_inputs=False)
    ctl = DR.Ctl()
    rig = Rig(a)
    fdir = os.path.join(os.path.dirname(os.path.abspath(out_path)), "smoke_floors")
    res = read_core(rig, ctl, man, "read", rf0, floors_dir=fdir)''',
    '''    d20 = man["arms"]["CTRL"]      # DAG2-0: a FULL slot (it has the logged tick-0 run that REP's full analysis reads)
    man["arms"]["CTRL"] = rc_oacc_manifest()["arms"]["CTRL"]      # A1: CTRL := DAG-OACC (banked, light); A := DAG2, B := DAG2, REP := DAG2-0 as stand-ins
    man["arms"]["REP"], man["arms"]["B"] = d20, man["arms"]["A"]
    pr, rf0, _ = k_p0(a.rf0, [], check_inputs=False)
    ctl = DR.Ctl()
    rig = Rig(a)
    fdir = os.path.join(os.path.dirname(os.path.abspath(out_path)), "smoke_floors")
    res = read_core(rig, ctl, man, "read", rf0, floors_dir=fdir)''')
rep('''    out = {"SMOKE": "stand-in slots (A := DAG2, B := DAG2, REP := DAG2-0): NOT A READ"''', '''    out = {"SMOKE": "stand-in slots (A := DAG2, B := DAG2, REP := DAG2-0, CTRL := DAG-OACC): NOT A READ"''')
rep('''    ap.add_argument("--rc-dag2", action="store_true", help="K-RC: the analysis on DAG2 / DAG2-0, floors 0, adoption off, against the banked DAG2_READ.json")''',
    '''    ap.add_argument("--rc-dag2", action="store_true", help="K-RC leg 1: the analysis on DAG2 / DAG2-0, floors 0, adoption off, against the banked DAG2_READ.json")
    ap.add_argument("--rc-oacc", action="store_true", help="K-RC leg 2 (A1): DAG-OACC - P10 vs DAGOACC_READ.json's Read-A table, 0 differences")
    ap.add_argument("--rc-oacc-evidence", default=f"{HERE}/RC_OACC.json", help="K-RC leg 2's evidence file (must carry this reader's sha256)")
    ap.add_argument("--amend1", default=f"{HERE}/AMEND1.md")''')
rep('''    a = default_args(prereg=ns.prereg, rf0=ns.rf0, work_dir=ns.work_dir, n_boot=ns.n_boot)''', '''    a = default_args(prereg=ns.prereg, amend1=ns.amend1, rf0=ns.rf0, work_dir=ns.work_dir, n_boot=ns.n_boot)''')
rep('''    if ns.rc_dag2:
        return main_rc(a, ns.out or f"{HERE}/RC_DAG2.json")''', '''    if ns.rc_dag2:
        return main_rc(a, ns.out or f"{HERE}/RC_DAG2.json")
    if ns.rc_oacc:
        return main_rc_oacc(a, ns.out or f"{HERE}/RC_OACC.json")''')
rep('''    return main_read(a, ns.manifest, ns.out, ns.rc_evidence)''', '''    return main_read(a, ns.manifest, ns.out, ns.rc_evidence, ns.rc_oacc_evidence)''')

OUT.write_text(t, encoding="utf-8", newline="\n")
print(f"{N[0]} replacements; wrote {OUT} sha256 {hashlib.sha256(t.encode()).hexdigest()} ({len(t)} bytes)")
