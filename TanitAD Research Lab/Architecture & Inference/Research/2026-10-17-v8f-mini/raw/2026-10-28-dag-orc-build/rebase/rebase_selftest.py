#!/usr/bin/env python3
"""Re-base dag_orc_selftest.py and dag_orc_mutation.py (the reader's self-test and its mutation proof) for Amendment 1.
    python rebase_selftest.py     reads the PRISTINE files in E:/Projects/TanitAD-artifacts/dag_orc_read/, writes ./read/dag_orc_selftest.py and ./read/dag_orc_mutation.py
Every replacement is exact-count (a stale anchor stops the script).  The checks that encoded DAG-based values are rewritten for DAG-OACC; checks of unchanged logic are untouched."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path("E:/Projects/TanitAD-artifacts/dag_orc_read")
COMMIT = json.loads(Path("E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s0_dagoacc/launch_gate_token.json").read_text(encoding="utf-8"))["binding"]["commit"]


class P:
    def __init__(self, name):
        self.name = name
        self.t = (SRC / name).read_text(encoding="utf-8")
        assert "\r" not in self.t
        self.n = 0

    def rep(self, old, new, n=1):
        assert self.t.count(old) == n, (self.name, self.t.count(old), old[:80])
        self.t = self.t.replace(old, new)
        self.n += 1

    def block(self, start, end, new):
        i = self.t.index(start)
        assert self.t.count(start) == 1, (self.name, start[:60])
        j = self.t.index(end, i)
        j = self.t.rfind("\n", 0, j) + 1
        self.t = self.t[:i] + new + self.t[j:]
        self.n += 1

    def save(self):
        (HERE / "read" / self.name).write_text(self.t, encoding="utf-8", newline="\n")
        print(f"{self.name}: {self.n} edits, sha256 {hashlib.sha256(self.t.encode()).hexdigest()}")


# ================================================================================================ the self-test
S = P("dag_orc_selftest.py")
S.block('check("L16 the constants', "# ---------------------------------------------------------------------------------------------------------------- B: the bar", '''check("L16 the constants (A1): F 0.1207, F/2 0.0604, p 0.3 / 0.1, tolerance 0.02, REP seeds 10 / 11, M2b_s / DAG-OACC shas, crawl sets (4 rows, both seeds), NO exempt rows, |D_s| = 15",
      R.F_ADE6 == 0.1207 and R.HALF_F == 0.0604 and R.P_A == 0.3 and R.P_B == 0.1 and R.REACH_TOL == 0.02 and R.REP_SEED == {0: 10, 1: 11}
      and R.M2B_CKPT_SHA[0].startswith("601cb435") and R.M2B_CKPT_SHA[1].startswith("6104fb15") and R.OACC_CKPT_SHA[0].startswith("67265ec1") and R.OACC_CKPT_SHA[1].startswith("90b01474")
      and R.OACC_BANK_SHA[0].startswith("197a68bb") and R.OACC_BANK_SHA[1].startswith("36c7aea9") and R.W_ACC == "49.6969" and R.SWAP_P == 0.1 and R.COMPOSE_FLAG == "--ego-perturb-with-dagger"
      and R.CRAWL_SET[0] == R.CRAWL_SET[1] == ("yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")
      and R.EXEMPT == () and R.DOMAIN_N == {0: 15, 1: 15} and R.SEEDS == (0, 1) and R.BUILD_BASE_COMMIT == "%s" and R.ARMS == ("A", "CTRL", "REP", "B") and R.REQUIRED_SLOTS == ("A", "CTRL", "REP"))
check("L16b the ORIGINAL section 4 values survive for the rc-DAG2 leg only", R.CRAWL_SET_PRE_A1[1] == R.CRAWL_SET_PRE_A1[0] + ("yaw-rate MAE",) and R.EXEMPT_PRE_A1 == ("accel MAE", "DK-VIOL") and R.DOMAIN_N_PRE_A1 == {0: 14, 1: 13}
      and R.DAG_CKPT_SHA[0].startswith("d186fd0bcfa7ee9f"))
check("L17 every pin the reader codes equals the banked DAG-OACC artifacts (SHA256SUMS.txt, launch-gate token, summary.json, DAGOACC_READ.json)", R.banked_pin_problems() == [], R.banked_pin_problems())
_keep = (dict(R.M2B_CKPT_SHA), dict(R.OACC_BANK_SHA), dict(R.OACC_CKPT_SHA), dict(R.OACC_ARGV_SHA16), R.BUILD_BASE_COMMIT, R.OACC_READ_SHA256)
try:
    R.M2B_CKPT_SHA[0] = "0" * 64
    p_m2b = R.banked_pin_problems()
    R.M2B_CKPT_SHA.update(_keep[0]); R.OACC_BANK_SHA[1] = "0" * 64
    p_bank = R.banked_pin_problems()
    R.OACC_BANK_SHA.update(_keep[1]); R.OACC_CKPT_SHA[1] = "0" * 64
    p_ck = R.banked_pin_problems()
    R.OACC_CKPT_SHA.update(_keep[2]); R.OACC_ARGV_SHA16[0] = "0" * 16
    p_av = R.banked_pin_problems()
    R.OACC_ARGV_SHA16.update(_keep[3]); R.BUILD_BASE_COMMIT = "1d4f4fbb"
    p_cm = R.banked_pin_problems()
    R.BUILD_BASE_COMMIT = _keep[4]; R.OACC_READ_SHA256 = "0" * 64
    p_rd = R.banked_pin_problems()
finally:
    R.M2B_CKPT_SHA.update(_keep[0]); R.OACC_BANK_SHA.update(_keep[1]); R.OACC_CKPT_SHA.update(_keep[2]); R.OACC_ARGV_SHA16.update(_keep[3]); R.BUILD_BASE_COMMIT = _keep[4]; R.OACC_READ_SHA256 = _keep[5]
check("L17b each banked pin, doctored one at a time (M2b init, bank, DAG-OACC ckpt, argv sha, build commit, DAGOACC_READ sha), is caught by banked_pin_problems against the banked artifacts",
      all(bool(x) for x in (p_m2b, p_bank, p_ck, p_av, p_cm, p_rd)) and R.banked_pin_problems() == [], (p_m2b, p_bank, p_ck, p_av, p_cm, p_rd))
AM = f"{HERE}/AMEND1.md"
am_text = open(AM, encoding="utf-8").read()
check("L18 the real AMEND1.md passes amend_problems and hashes to AMEND1_SHA256", R.amend_problems(AM) == [] and hashlib.sha256(open(AM, "rb").read()).hexdigest() == R.AMEND1_SHA256)
am_bad = os.path.join(TMP, "AMEND1_changed.md")
open(am_bad, "w", encoding="utf-8", newline="\\n").write(am_text.replace("0.0604 m", "0.0605 m", 1))
check("L19 an AMEND1.md with one changed number is refused by the raw-byte pin", any("not the pinned" in p for p in R.amend_problems(am_bad)))
check("L20 a missing AMEND1.md is refused", R.amend_problems(os.path.join(TMP, "nope_am.md")) != [])
miss_am = 0
for k_, ph_ in enumerate(R.AMEND_PHRASES):
    t_ = re.sub(r"\\s+", " ", am_text.replace("**", ""))
    p_ = re.sub(r"\\s+", " ", ph_.replace("**", ""))
    t2 = t_.replace(p_, p_[:-1] + "X", 1)
    pth = os.path.join(TMP, f"AM_ph{k_}.md")
    open(pth, "w", encoding="utf-8", newline="\\n").write(t2)
    own = hashlib.sha256(open(pth, "rb").read()).hexdigest()
    miss_am += bool(R.amend_problems(pth, pin=own)) and not any("not the pinned" in p for p in R.amend_problems(pth, pin=own))
check(f"L21 each of the {len(R.AMEND_PHRASES)} coded amendment phrases, edited one at a time (pin = the copy's own sha256), is caught by the phrase step alone", miss_am == len(R.AMEND_PHRASES), miss_am)
check("L22 AMENDED_NEXT covers every outcome row (REFUSED keeps the original) and the original text stays reachable", set(R.AMENDED_NEXT) == set(R.OUTCOME_NAMES) and R.AMENDED_NEXT["REFUSED"] is None
      and all(R.table_row_original(n) == next(r[2] for r in R.PREREG_CELLS["7."]["rows"][1:] if r[0] == n) for n in R.OUTCOME_NAMES))

''' % COMMIT[:8])
S.rep('''open(ev, "w").write(json.dumps''', '''open(ev, "w").write(json.dumps''', n=S.t.count('''open(ev, "w").write(json.dumps'''))      # (anchor sanity only)
S.rep('''check("B20 domain drops DAG-P10 separated-worse rows (error and agreement) and the exempt rows, keeps improved/neutral ones",
      "curvature MAE" not in dom and "accel MAE" not in dom and "TAC-LAT" not in dom and "speed MAE" in dom and "DK-VIOL" not in dom and "ADE6" in dom and len(dom) == 19 - 4)''',
      '''check("B20 domain drops DAG-OACC-P10 separated-worse rows (error and agreement), keeps improved/neutral ones and, after A1, accel MAE / DK-VIOL when they are not worse",
      "curvature MAE" not in dom and "accel MAE" not in dom and "TAC-LAT" not in dom and "speed MAE" in dom and "DK-VIOL" in dom and "ADE6" in dom and len(dom) == 19 - 3)''')
S.rep('''check("B21 exempt rows are excluded even when DAG-P10 is NOT worse on them", "accel MAE" not in R.domain(dp2) and "DK-VIOL" not in R.domain(dp2) and len(R.domain(dp2)) == 17)''',
      '''check("B21 A1: there are NO exempt rows: accel MAE and DK-VIOL are in the domain when DAG-OACC-P10 is not worse on them (all 19 rows of a neutral table)", "accel MAE" in R.domain(dp2) and "DK-VIOL" in R.domain(dp2) and len(R.domain(dp2)) == 19
      and "accel MAE" not in R.domain(dp2, R.EXEMPT_PRE_A1) and len(R.domain(dp2, R.EXEMPT_PRE_A1)) == 17)''')
S.rep('''    want = next(r[2] for r in R.PREREG_CELLS["7."]["rows"][1:] if r[0] == name)
    check(f"O14 the committed next of {name} is the PREREG's cell", nxt == want and R.next_for_arm(name, "A").startswith(nxt[:10]) or name == "HARMS")''',
      '''    want = next(r[2] for r in R.PREREG_CELLS["7."]["rows"][1:] if r[0] == name)
    want_am = R.AMENDED_NEXT[name] or want
    check(f"O14 the committed next of {name} is the AMENDED cell (A1) and the original cell stays reachable", nxt == want_am and R.table_row_original(name) == want and R.next_for_arm(name, "A").startswith(nxt[:10]) or name == "HARMS")''')
S.rep('''check("O15 HARMS' committed next splits: A reads B, B goes to NEXT-CRAWL on DAG states", na.startswith("A: not adopted; read B") and "NEXT-CRAWL" not in na and nb_.startswith("B: not adopted") and "NEXT-CRAWL on DAG states" in nb_, (na, nb_))''',
      '''check("O15 HARMS' committed next splits (A1): A reads B (needs its own PI go), B goes to NEXT-CRAWL on DAG-OACC's states, LAUNCH-INTENT branch", na.startswith("A: not adopted; read B") and "NEXT-CRAWL" not in na and "PI go" in na and nb_.startswith("B: not adopted") and "NEXT-CRAWL on DAG-OACC's states" in nb_ and "LAUNCH-INTENT" in nb_, (na, nb_))
for nm_ in ("FIXED / NOT ADOPTED", "PARTIAL / NOT ADOPTED"):
    check(f"O15b A1: {nm_} no longer routes to DAG-ORC-SWAP (it cannot apply: A keeps the swap on); the next is B under this table with its own PI go", "B (p 0.1)" in R.table_row(nm_)[1] and "PI go" in R.table_row(nm_)[1] and "cannot apply" in R.table_row(nm_)[1])
check("O15c A1: every NEXT-CRAWL route names DAG-OACC's states and the LAUNCH-INTENT branch", all("DAG-OACC's states" in R.table_row(n)[1] and "LAUNCH-INTENT" in R.table_row(n)[1] for n in ("PARTIAL / ONE SEED", "NO EFFECT")) and "X's states" in R.table_row("PARTIAL / ADOPTED")[1] and "LAUNCH-INTENT" in R.table_row("PARTIAL / ADOPTED")[1])''')
S.rep('''ps0 = exempt_case(lambda cs: [cs[0]["t"]["X_P10"].update({"accel MAE": row(+0.9, True)}), cs[0]["t"]["X_CTRL"].update({"accel MAE": row(+0.9, True)}), cs[0]["t"]["DAG_P10"].update({"accel MAE": row(0.0, False)})])
check("O33 accel MAE is exempt from R-NEW (not in the domain) but REPORTED", ps0["guards"]["R-NEW"] is True and ps0["exempt_reported"]["accel MAE"]["X_minus_CTRL_separated_worse"] is True and "accel MAE" not in ps0["domain"])
ps0 = exempt_case(lambda cs: [cs[0]["t"]["X_P10"].update({"DK-VIOL": row(+0.9, True)}), cs[0]["t"]["X_DAG"].update({"DK-VIOL": row(+0.9, True)})])
check("O34 DK-VIOL is exempt from R-NOBASE but REPORTED", ps0["flags"]["R-NOBASE"] is True and ps0["exempt_reported"]["DK-VIOL"]["X_minus_P10_separated_worse"] is True)''',
      '''ps0 = exempt_case(lambda cs: [cs[0]["t"]["X_P10"].update({"accel MAE": row(+0.9, True)}), cs[0]["t"]["X_CTRL"].update({"accel MAE": row(+0.9, True)}), cs[0]["t"]["DAG_P10"].update({"accel MAE": row(0.0, False)})])
check("O33 A1: accel MAE is GUARDED by R-NEW (it is in the domain; no exempt rows): X - P10 worse, X - CTRL worse 0.9, DAG-OACC - P10 not worse -> R-NEW fires", ps0["guards"]["R-NEW"] is False and "accel MAE" in ps0["domain"] and ps0["exempt_reported"] == {})
ps0 = exempt_case(lambda cs: [cs[0]["t"]["X_P10"].update({"DK-VIOL": row(+0.9, True)}), cs[0]["t"]["X_DAG"].update({"DK-VIOL": row(+0.9, True)})])
check("O34 A1: DK-VIOL is GUARDED by R-NOBASE: X - P10 worse and X - DAG-OACC worse by 0.9 -> R-NOBASE fails", ps0["flags"]["R-NOBASE"] is False and "DK-VIOL" in ps0["domain"])
ps_pre = R.compose_seed(0, base_seed(0)["t"], base_seed(0)["g"], RF0_SMALL, RF_SMALL, pre_a1=True)
check("O34b the rc-DAG2 leg composes with the ORIGINAL section 4 (pre_a1): the exempt rows are exempt and the original crawl set applies", set(ps_pre["exempt_reported"]) == {"accel MAE", "DK-VIOL"} and tuple(ps_pre["crawl_set"]) == R.CRAWL_SET_PRE_A1[0]
      and "accel MAE" not in ps_pre["domain"] and "DK-VIOL" not in ps_pre["domain"])
_blk = R.outcome_block(R.decide_arm(True, compose_pair()), "A")
check("O16b the outcome block carries the amended next AND the original PREREG cell verbatim, and names the amendment", _blk["committed_next_original_prereg"] == R.table_row_original("FIXED / ADOPTED") and _blk["committed_next"] != _blk["committed_next_original_prereg"]
      and "A1" in _blk["amendment"])''')
S.rep('''check("O40 R-MOVE needs EVERY row of C_s (one weak row fails it); seed 1 has the yaw-rate row, seed 0 does not",
      compose_pair(lambda cs: cs[1]["t"]["X_CTRL"].update({"yaw-rate MAE": row(-0.001, True)}))[1]["flags"]["R-MOVE"] is False
      and compose_pair(lambda cs: cs[0]["t"]["X_CTRL"].update({"yaw-rate MAE": row(+9.0, True)}))[0]["flags"]["R-MOVE"] is True)''',
      '''check("O40 A1: R-MOVE needs EVERY row of C_s (one weak row fails it); the yaw-rate row is in C at BOTH seeds",
      compose_pair(lambda cs: cs[1]["t"]["X_CTRL"].update({"yaw-rate MAE": row(-0.001, True)}))[1]["flags"]["R-MOVE"] is False
      and compose_pair(lambda cs: cs[0]["t"]["X_CTRL"].update({"yaw-rate MAE": row(-0.001, True)}))[0]["flags"]["R-MOVE"] is False
      and compose_pair(lambda cs: cs[0]["t"]["X_CTRL"].update({"heading MAE": row(+9.0, True)}))[0]["flags"]["R-MOVE"] is True)''')
S.rep('''check("O41 R-CRAWL reads X - P10 on C_s only: a worse non-C row does not fail it; the yaw-rate row fails it at seed 1 only",
      compose_pair(lambda cs: cs[0]["t"]["X_P10"].update({"heading MAE": row(+0.4, True)}))[0]["flags"]["R-CRAWL"] is True
      and compose_pair(lambda cs: [cs[s]["t"]["X_P10"].update({"yaw-rate MAE": row(+0.4, True)}) for s in R.SEEDS])[1]["flags"]["R-CRAWL"] is False
      and compose_pair(lambda cs: [cs[s]["t"]["X_P10"].update({"yaw-rate MAE": row(+0.4, True)}) for s in R.SEEDS])[0]["flags"]["R-CRAWL"] is True)''',
      '''check("O41 A1: R-CRAWL reads X - P10 on C_s only: a worse non-C row does not fail it; the yaw-rate row fails it at BOTH seeds",
      compose_pair(lambda cs: cs[0]["t"]["X_P10"].update({"heading MAE": row(+0.4, True)}))[0]["flags"]["R-CRAWL"] is True
      and compose_pair(lambda cs: [cs[s]["t"]["X_P10"].update({"yaw-rate MAE": row(+0.4, True)}) for s in R.SEEDS])[1]["flags"]["R-CRAWL"] is False
      and compose_pair(lambda cs: [cs[s]["t"]["X_P10"].update({"yaw-rate MAE": row(+0.4, True)}) for s in R.SEEDS])[0]["flags"]["R-CRAWL"] is False)''')
# ---- K: the run-record controls on DAG-OACC's banked argv
S.rep('''ctrl_argv = ["--arm", "M2b", "--seed", "0", "--grid", "pool4x10", "--out", "D:\\\\runs\\\\M2b_s0_dag20", "--init-ckpt", "D:\\\\runs\\\\M2b_s0_dag\\\\ckpt_last.pt", "--steps", "3000", "--lr", "1e-4",
             "--warmup", "100", "--p-gt-start", "0.2", "--p-gt-end", "0.2", "--ego-perturb-p", "0", "--dagger-p", "0"]
real_argv_path = "E:/Projects/TanitAD-artifacts/v8f-dagger2/M2b_s0_dag20/argv.json"''',
      '''real_argv_path = "E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s0_dagoacc/argv.json"
assert os.path.isfile(real_argv_path), "DAG-OACC's banked argv.json is the K-section's fixture (A1)"
ctrl_argv = None''')
S.rep('''f["--out"] = [["D:\\\\runs\\\\M2b_s%d_dagorc" % s]]''', '''f["--out"] = [["D:\\\\runs\\\\M2b_s%d_dagoaccorc" % s]]''')
S.rep('''    cf = R.parse_flags(json.load(open(f"E:/Projects/TanitAD-artifacts/v8f-dagger2/M2b_s{s}_dag20/argv.json", encoding="utf-8"))["argv"]) if os.path.isfile(real_argv_path) else CF''',
      '''    cf = R.parse_flags(json.load(open(f"E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s{s}_dagoacc/argv.json", encoding="utf-8"))["argv"])''')
S.rep('''    check(f"K1 A_s{s} = DAG2-0 + p 0.3 + the five law flags passes K-PAIR", ok and True)
    check(f"K2 DAG2-0's own argv does not pass as A (no lever)", not R.k_pair("A", s, cf, cf)[0])''',
      '''    check(f"K1 A_s{s} = DAG-OACC + p 0.3 + the five law flags + --ego-perturb-with-dagger passes K-PAIR", ok and True)
    check(f"K2 DAG-OACC's own argv does not pass as A (no lever)", not R.k_pair("A", s, cf, cf)[0])
    wc = flags_A(s, cf)
    del wc[R.COMPOSE_FLAG]
    check("K2b A WITHOUT --ego-perturb-with-dagger fails K-PAIR (the composition is the point)", not R.k_pair("A", s, wc, cf)[0])
    check("K2b2 ... and fails the registered-argv check too; REP carrying the flag fails it; B (p 0.1) needs it as well", not R.k_registered("A", s, wc)[0]
          and not R.k_registered("REP", s, {**cf, "--seed": [[str(R.REP_SEED[s])]], R.COMPOSE_FLAG: [[]]})[0] and R.k_registered("B", s, flags_A(s, cf, p=0.1))[0] and not R.k_registered("B", s, dict(wc, **{"--ego-perturb-p": [["0.1"]]}))[0]
          and not R.k_registered("A", s, {**flags_A(s, cf), R.COMPOSE_FLAG: [["x"]]})[0] and not R.k_registered("CTRL", s, flags_A(s, cf))[0])
    wc = {k: v for k, v in cf.items()}
    wc["--seed"] = [[str(R.REP_SEED[s])]]
    check("K2c REP = DAG-OACC with --seed 10/11 passes K-PAIR; REP carrying the composition flag fails", R.k_pair("REP", s, wc, cf)[0] and not R.k_pair("REP", s, {**wc, R.COMPOSE_FLAG: [[]]}, cf)[0])
    wb = flags_A(s, cf)
    wb["--dagger-bank-sha256"] = [["0" * 64]]
    check("K2d A with another bank sha256 fails K-PAIR (the bank is DAG-OACC's)", not R.k_pair("A", s, wb, cf)[0])
    wb = flags_A(s, cf)
    wb["--w-acc"] = [["49.0"]]
    check("K2e A with another --w-acc fails K-PAIR", not R.k_pair("A", s, wb, cf)[0])''')
S.rep('''    check("K21 the registered fine-tune check passes on DAG2-0's banked argv", reg[0], reg[1])
    for nm, val in (("--steps", "2999"), ("--lr", "2e-4"), ("--warmup", "99"), ("--p-gt-start", "0.3"), ("--p-gt-end", "0.1"), ("--dagger-p", "0.1"), ("--seed", str(1 - s))):''',
      '''    check("K21 the registered fine-tune check passes on DAG-OACC's banked argv", reg[0], reg[1])
    for nm, val in (("--steps", "2999"), ("--lr", "2e-4"), ("--warmup", "99"), ("--p-gt-start", "0.3"), ("--p-gt-end", "0.1"), ("--dagger-p", "0.2"), ("--dagger-p", "0"), ("--seed", str(1 - s)), ("--w-acc", "49.0"),
                    ("--dagger-seed", str(1 - s)), ("--dagger-bank-sha256", "0" * 64)):''')
S.rep('''    w["--init-ckpt"] = [[f"D:\\\\runs\\\\M2b_s{s}\\\\ckpt_last.pt"]]
    check("K23 an init that is M2b_s (not DAG_s) fails", not R.k_registered("CTRL", s, w)[0])''',
      '''    w["--init-ckpt"] = [[f"D:\\\\runs\\\\M2b_s{s}_dag\\\\ckpt_last.pt"]]
    check("K23 an init that is DAG_s (not M2b_s) fails (A1: the init is M2b_s, DAG-OACC's)", not R.k_registered("CTRL", s, w)[0])''')
S.rep('''    good = [("summary:levers.init_ckpt.sha256", R.DAG_CKPT_SHA[s]), ("selfcheck:levers.init_ckpt.sha256", R.DAG_CKPT_SHA[s])]
    check("K26 K-INIT: every recorded init sha256 equal to DAG_s passes", R.k_init("A", s, good)[0])''',
      '''    good = [("summary:levers.init_ckpt.sha256", R.M2B_CKPT_SHA[s]), ("selfcheck:levers.init_ckpt.sha256", R.M2B_CKPT_SHA[s])]
    check("K26 K-INIT (A1): every recorded init sha256 equal to M2b_s (DAG-OACC's token) passes; DAG_s's checkpoint no longer does", R.k_init("A", s, good)[0] and not R.k_init("A", s, [("x", R.DAG_CKPT_SHA[s])])[0]
          and R.k_init("A", s, [("x", R.DAG_CKPT_SHA[s])], want=R.DAG_CKPT_SHA)[0])''')
S.rep('''    check("K28 K-INIT: the other seed's DAG sha fails", not R.k_init("A", s, [("x", R.DAG_CKPT_SHA[1 - s])])[0])''', '''    check("K28 K-INIT: the other seed's M2b sha fails", not R.k_init("A", s, [("x", R.M2B_CKPT_SHA[1 - s])])[0])''')
S.block('check("K30 K-REACH A:', "# K-BUILD", '''DG = {"dagger_n": 8200, "dagger_eligible": 82000, "ego_dagger_overlap_n": 0, "ego_perturb_with_dagger": True}      # a composed run record (swap 0.1, no overlap, flag built)
DGR = {"dagger_n": 8200, "dagger_eligible": 82000, "ego_dagger_overlap_n": 0, "ego_perturb_with_dagger": False}      # REP: swap on, flag not built
check("K30 K-REACH A (A1): ego 24,720 / 82,457 (0.2998) AND swap 8,200 / 82,000 AND overlap 0 AND the flag built passes", R.k_reach("A", 0.3, 24720, 82457, None, DG)[0])
check("K31 K-REACH A: the 0.02 ego band is inclusive at 0.32 and fails at 0.3201", R.k_reach("A", 0.3, 32000, 100000, None, DG)[0] and not R.k_reach("A", 0.3, 32010, 100000, None, DG)[0] and R.k_reach("A", 0.3, 28000, 100000, None, DG)[0] and not R.k_reach("A", 0.3, 27990, 100000, None, DG)[0])
check("K32 K-REACH B: ego 0.1 +- 0.02 (swap 0.1 too)", R.k_reach("B", 0.1, 8200, 82457, None, DG)[0] and not R.k_reach("B", 0.1, 24720, 82457, None, DG)[0] and not R.k_reach("B", 0.1, 4000, 82457, None, DG)[0])
check("K33 K-REACH REP: ego exactly 0 + swap in band + flag absent passes; ego 1 fails", R.k_reach("REP", 0.0, 0, 82457, None, DGR)[0] and not R.k_reach("REP", 0.0, 1, 82457, None, DGR)[0]
      and R.k_reach("REP", 0.0, 0, 82457, None, {**DGR, "ego_perturb_with_dagger": None})[0])
check("K34 K-REACH: built p must equal the argv p (A built at 0.0 fails: the lever never reached training)", not R.k_reach("A", 0.0, 24720, 82457, None, DG)[0] and not R.k_reach("A", None, 24720, 82457, None, DG)[0])
check("K35 K-REACH: eligible 0 / n > eligible / non-numbers fail", not R.k_reach("A", 0.3, 0, 0, None, DG)[0] and not R.k_reach("A", 0.3, 5, 4, None, DG)[0] and not R.k_reach("A", 0.3, None, 82457, None, DG)[0])
check("K36 K-REACH: A perturbing NOTHING (n = 0) fails (the lever never fired)", not R.k_reach("A", 0.3, 0, 82457, None, DG)[0])
check("K36a K-REACH (A1): the SWAP must reach too: swap fraction 0.14 (outside 0.1 +- 0.02) fails on A and on REP; swap 0 fails", not R.k_reach("A", 0.3, 24720, 82457, None, {**DG, "dagger_n": 11500})[0]
      and not R.k_reach("REP", 0.0, 0, 82457, None, {**DGR, "dagger_n": 11500})[0] and not R.k_reach("A", 0.3, 24720, 82457, None, {**DG, "dagger_n": 0})[0])
check("K36b K-REACH (A1): the swap band is inclusive at 0.12 and fails at 0.1201", R.k_reach("A", 0.3, 24720, 82457, None, {**DG, "dagger_n": 12000, "dagger_eligible": 100000})[0]
      and not R.k_reach("A", 0.3, 24720, 82457, None, {**DG, "dagger_n": 12010, "dagger_eligible": 100000})[0] and R.k_reach("A", 0.3, 24720, 82457, None, {**DG, "dagger_n": 8000, "dagger_eligible": 100000})[0]
      and not R.k_reach("A", 0.3, 24720, 82457, None, {**DG, "dagger_n": 7990, "dagger_eligible": 100000})[0])
check("K36c K-REACH (A1): the overlap must be EXACTLY 0: 1, None, 0.0-as-float-missing and a bool fail on A", all(not R.k_reach("A", 0.3, 24720, 82457, None, {**DG, "ego_dagger_overlap_n": v})[0] for v in (1, None, True)))
check("K36d K-REACH (A1): the composition flag must be built on A (False / None fail) and absent on REP (True fails)", not R.k_reach("A", 0.3, 24720, 82457, None, {**DG, "ego_perturb_with_dagger": False})[0]
      and not R.k_reach("A", 0.3, 24720, 82457, None, {**DG, "ego_perturb_with_dagger": None})[0] and not R.k_reach("REP", 0.0, 0, 82457, None, {**DGR, "ego_perturb_with_dagger": True})[0])
check("K36e K-REACH (A1): no run-record counts at all fails (a lever declared and unrecorded)", not R.k_reach("A", 0.3, 24720, 82457)[0] and not R.k_reach("REP", 0.0, 0, 82457)[0])
''')
S.rep('''"base_commit": "1d4f4fbb11922f4dd62b51871d898c03e0c88635"''', '''"base_commit": "%s"''' % COMMIT)
S.rep('''check("K42 K-BUILD: a base commit other than 1d4f4fbb fails"''', '''check("K42 K-BUILD: a base commit other than aeb54468 (DAG-OACC's) fails, 1d4f4fbb (DAG2's) included"''')
S.rep('''b3["base_commit"] = "deadbeef"
check(''', '''b3["base_commit"] = "1d4f4fbb11922f4dd62b51871d898c03e0c88635"
check(''')
S.rep('''check("K43 K-BUILD: a missing file record fails; so does no record at all", not R.k_build(b3)[0] and not R.k_build(None)[0])''',
      '''check("K43 K-BUILD: a missing file record fails; so does no record at all", not R.k_build(b3)[0] and not R.k_build(None)[0])
b3 = copy.deepcopy(bld)
del b3["files"]["stack/tanitad/data/v8f_dagger.py"]
check("K43b K-BUILD (A1): v8f_dagger.py is a build file too: a missing record for it fails, and a changed one needs its listed default-off hunk", not R.k_build(b3)[0] and "stack/tanitad/data/v8f_dagger.py" in R.BUILD_FILES
      and not R.k_build({**bld, "files": {**bld["files"], "stack/tanitad/data/v8f_dagger.py": {"sha256_launch": sha_b, "sha256_base": sha_a}}})[0])''')
# ---- S: slots (A1: no DAG slot; CTRL is the banked DAG-OACC, t1 + t0 only)
S.rep('''man = {"arms": {"A": {"0": mk_entry(f"{slot_dir}/A0", "A"), "1": mk_entry(f"{slot_dir}/A1", "A", n_dump=140)}, "CTRL": {"0": mk_entry(f"{slot_dir}/C0", "CTRL")},
                "DAG": {"0": mk_entry(f"{slot_dir}/D0", "DAG")}}}''',
      '''man = {"arms": {"A": {"0": mk_entry(f"{slot_dir}/A0", "A"), "1": mk_entry(f"{slot_dir}/A1", "A", n_dump=140)}, "CTRL": {"0": mk_entry(f"{slot_dir}/C0", "CTRL")}}}''')
S.rep('''sl["CTRL"][0]["state"] == "PRESENT" and sl["DAG"][0]["state"] == "PRESENT", sl["A"][0])''', '''sl["CTRL"][0]["state"] == "PRESENT" and "DAG" not in sl, sl["A"][0])''')
S.rep('''man2["arms"]["DAG"]["0"].pop("t1_logged")
check("S6 discover: the banked DAG slot needs no logged lane and no T0", R.discover(man2)["DAG"][0]["state"] == "PRESENT")''',
      '''man2["arms"]["CTRL"]["0"].pop("t1_logged")
man2["arms"]["CTRL"]["0"].pop("t0_fresh")
check("S6 discover (A1): the banked CTRL slot (= DAG-OACC) needs no logged lane and no fresh T0", R.discover(man2)["CTRL"][0]["state"] == "PRESENT")
os.remove(man2["arms"]["CTRL"]["0"]["t0"]["npz"])
check("S6b discover (A1): ... but it DOES need box 2's standard T0 eval (R-T0's comparator) and its train records: a missing t0.npz makes it PARTIAL", R.discover(man2)["CTRL"][0]["state"] == "PARTIAL")
man2 = copy.deepcopy(man)
os.remove(os.path.join(man2["arms"]["CTRL"]["0"]["train_dir"], "selfcheck.json"))
check("S6c discover (A1): the banked CTRL slot without its selfcheck.json is PARTIAL", R.discover(man2)["CTRL"][0]["state"] == "PARTIAL")
man2 = copy.deepcopy(man)
check("S6d discover with light = () (the rc-DAG2 leg's reading) wants the logged lane of CTRL again", R.discover({"arms": {"CTRL": {"0": {k: v for k, v in man2["arms"]["CTRL"]["0"].items() if k != "t1_logged"}}}}, light=())["CTRL"][0]["state"] == "PARTIAL")''')
S.rep('''set(R.a_flags({}, 0)) == {"--ego-perturb-p", "--ego-perturb-seed", *R.LAW_A})''', '''set(R.a_flags({}, 0)) == {"--ego-perturb-p", "--ego-perturb-seed", R.COMPOSE_FLAG, *R.LAW_A})''')
S.rep('''    pth = f"E:/Projects/TanitAD-artifacts/v8f-dagger2/M2b_s{s}_dag20/argv.json"
    return R.parse_flags(json.load(open(pth, encoding="utf-8"))["argv"]) if os.path.isfile(pth) else CF''',
      '''    pth = f"E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s{s}_dagoacc/argv.json"
    return R.parse_flags(json.load(open(pth, encoding="utf-8"))["argv"])''')
S.block('banked =json.load(open("E:/Projects/TanitAD-artifacts/dagreread/raw/READ_A.json"', "# ---------------------------------------------------------------------------------------------------------------- report", '''banked_oacc = json.load(open("E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1/read/DAGOACC_READ.json", encoding="utf-8"))
for s in R.SEEDS:
    rows_s = banked_oacc["rows"][f"s{s}"]["READ_A"]["table_DAGOACC_minus_P10"]
    dom_s = R.domain({l: rows_s[l] for l in R.LABELS})
    check(f"R1 the banked DAG-OACC - P10 table (DAGOACC_READ.json) gives |D_{s}| = {R.DOMAIN_N[s]} (Amendment 1, no exempt rows)", len(dom_s) == R.DOMAIN_N[s], (len(dom_s), dom_s))
    derived = sorted(l for l, r in rows_s.items() if r.get("separated_worse"))
    check(f"R2 the banked separated-worse rows are the amendment's crawl set C_{s} (yaw-rate, curvature, L2, M1)", derived == sorted(R.CRAWL_SET[s]), derived)
    check(f"R3 every Read-A label of the banked table has the reader's direction", all(R.DIRECTION[l] == rows_s[l].get("direction") for l in R.LABELS))
    check(f"R4 the reader's sep_worse over the banked table agrees with the banked flag on all 19 rows", all(R.sep_worse(l, rows_s[l]) == bool(rows_s[l]["separated_worse"]) for l in R.LABELS))

''')
# MM 2026-10-28: the mutation harness found K-REACH's REP overlap clause untested (a1-reach-rep-overlap SURVIVED): pin it
S.rep('''check("K36e K-REACH (A1): no run-record counts at all fails (a lever declared and unrecorded)", not R.k_reach("A", 0.3, 24720, 82457)[0] and not R.k_reach("REP", 0.0, 0, 82457)[0])''',
      '''check("K36e K-REACH (A1): no run-record counts at all fails (a lever declared and unrecorded)", not R.k_reach("A", 0.3, 24720, 82457)[0] and not R.k_reach("REP", 0.0, 0, 82457)[0])
check("K36f K-REACH (A1): REP carrying an overlap fails (1 and 5); overlap 0 and an absent overlap pass", not R.k_reach("REP", 0.0, 0, 82457, None, {**DGR, "ego_dagger_overlap_n": 1})[0]
      and not R.k_reach("REP", 0.0, 0, 82457, None, {**DGR, "ego_dagger_overlap_n": 5})[0] and R.k_reach("REP", 0.0, 0, 82457, None, {**DGR, "ego_dagger_overlap_n": 0})[0]
      and R.k_reach("REP", 0.0, 0, 82457, None, {k: v for k, v in DGR.items() if k != "ego_dagger_overlap_n"})[0])''')
S.save()
