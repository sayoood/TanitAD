#!/usr/bin/env python3
"""Re-base dag_orc_mutation.py (the reader's mutation proof) for Amendment 1.
    python rebase_mutation.py    reads the PRISTINE E:/Projects/TanitAD-artifacts/dag_orc_read/dag_orc_mutation.py, rewrites the anchors that A1 moved, adds the A1 mutants, writes ./read/dag_orc_mutation.py
Exact-count replacements (a stale anchor stops this script)."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
t = (Path("E:/Projects/TanitAD-artifacts/dag_orc_read") / "dag_orc_mutation.py").read_text(encoding="utf-8")
n = [0]


def rep(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:90])
    t = t.replace(old, new)
    n[0] += 1


rep('SCR = "F:/TanitAD-scratch/dag_orc_read_tmp/mut"', 'SCR = "F:/TanitAD-scratch/dagorc_rebase_tmp/mut"')
rep('DAGORC_TMP="F:/TanitAD-scratch/dag_orc_read_tmp"', 'DAGORC_TMP="F:/TanitAD-scratch/dagorc_rebase_tmp"')
rep('python dag_orc_mutation.py [--workers 2]', 'python dag_orc_mutation.py [--workers 2]')
rep('main_rc: the readers run on real dumps) is exercised on REAL banked data by `--rc-dag2` (K-RC) and `--dry-run`, not by mutation.',
    'main_rc / main_rc_oacc: the readers run on real dumps) is exercised on REAL banked data by `--rc-dag2` and `--rc-oacc` (the two legs of K-RC), `--dry-run` and `--smoke-standins`, not by mutation.\nA1: the anchors that Amendment 1 moved are rewritten and the amendment\'s own logic (the AMEND1 pin, the swap + overlap + flag K-REACH, the composition flag in K-PAIR / the registered argv, the light CTRL slot, no exempt rows, C_s at both seeds,\nthe amended next texts) carries its own mutants (the "a1-" ids).')
# ---- anchors A1 moved
rep('''("domain-no-exempt", "domain keeps the exempt rows", 'if l not in EXEMPT and not sep_worse(l, dag_p10[l])]', 'if not sep_worse(l, dag_p10[l])]'),''',
    '''("domain-no-exempt", "domain ignores its exempt argument", 'if l not in exempt and not sep_worse(l, dag_p10[l])]', 'if not sep_worse(l, dag_p10[l])]'),''')
rep('''("domain-inverted", "domain keeps the DAG-worse rows", 'if l not in EXEMPT and not sep_worse(l, dag_p10[l])]', 'if l not in EXEMPT and sep_worse(l, dag_p10[l])]'),''',
    '''("domain-inverted", "domain keeps the DAG-OACC-worse rows", 'if l not in exempt and not sep_worse(l, dag_p10[l])]', 'if l not in exempt and sep_worse(l, dag_p10[l])]'),''')
rep('''("crawlset-s1-yaw", "C_1 drops yaw-rate", '"M1 crawl share", "yaw-rate MAE")}', '"M1 crawl share")}'),''',
    '''("crawlset-s1-yaw", "C_1 drops yaw-rate (A1: both seeds carry it)", '             1: ("yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")}\\nCRAWL_SET_PRE_A1', '             1: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")}\\nCRAWL_SET_PRE_A1'),
    ("crawlset-pre-a1", "the original C_1 loses yaw-rate (rc-DAG2 leg)", '1: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share", "yaw-rate MAE")}      # the ORIGINAL', '1: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")}      # the ORIGINAL'),''')
rep('''("crawlset-s0-m1", "C_0 drops M1", 'CRAWL_SET = {0: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"),', 'CRAWL_SET = {0: ("curvature MAE", "L2 curvature MAE (GT>=5.2)"),'),''',
    '''("crawlset-s0-m1", "C_0 drops M1", 'CRAWL_SET = {0: ("yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"),', 'CRAWL_SET = {0: ("yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)"),'),
    ("crawlset-s0-yaw", "C_0 drops yaw-rate (A1: both seeds carry it)", 'CRAWL_SET = {0: ("yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"),', 'CRAWL_SET = {0: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"),'),''')
rep('''("exempt-accel", "EXEMPT drops accel", 'EXEMPT = ("accel MAE", "DK-VIOL")', 'EXEMPT = ("DK-VIOL",)'),''',
    '''("exempt-a1", "A1: EXEMPT is no longer empty (accel MAE exempt again)", 'EXEMPT = ()    ', 'EXEMPT = ("accel MAE",)    '),
    ("exempt-pre-a1", "the original exemption loses accel MAE (rc-DAG2 leg)", 'EXEMPT_PRE_A1 = ("accel MAE", "DK-VIOL")', 'EXEMPT_PRE_A1 = ("DK-VIOL",)'),''')
rep('''("domain-n", "|D_1| 13 -> 14", 'DOMAIN_N = {0: 14, 1: 13}', 'DOMAIN_N = {0: 14, 1: 14}'),''',
    '''("domain-n", "|D_1| 15 -> 14", 'DOMAIN_N = {0: 15, 1: 15}', 'DOMAIN_N = {0: 15, 1: 14}'),
    ("domain-n-pre", "the original |D_1| 13 -> 14", 'DOMAIN_N_PRE_A1 = {0: 14, 1: 13}', 'DOMAIN_N_PRE_A1 = {0: 14, 1: 14}'),''')
rep('''("reg-no-dagger", "registered argv skips --dagger-p", '("--ego-perturb-p", p), ("--dagger-p", 0.0), ("--seed", want_seed))', '("--ego-perturb-p", p), ("--seed", want_seed))'),''',
    '''("reg-no-dagger", "registered argv skips --dagger-p", '("--ego-perturb-p", p), ("--dagger-p", SWAP_P), ("--seed", want_seed))', '("--ego-perturb-p", p), ("--seed", want_seed))'),
    ("reg-dagger-zero", "registered --dagger-p is 0 (the DAG-ORC value), not DAG-OACC's 0.1", '("--ego-perturb-p", p), ("--dagger-p", SWAP_P), ("--seed", want_seed))', '("--ego-perturb-p", p), ("--dagger-p", 0.0), ("--seed", want_seed))'),''')
rep('''endswith(f"M2b_s{seed}_dag/ckpt_last.pt")', 'endswith("ckpt_last.pt")'),''', '''endswith(f"/M2b_s{seed}/ckpt_last.pt")', 'endswith("ckpt_last.pt")'),''')
rep('''("init-any", "K-INIT accepts any sha", 'if str(v) != DAG_CKPT_SHA[seed]]', 'if False]'),''', '''("init-any", "K-INIT accepts any sha", 'if str(v) != want[seed]]', 'if False]'),''')
rep('''("init-other-seed", "K-INIT compares with the other seed's DAG", 'if str(v) != DAG_CKPT_SHA[seed]]', 'if str(v) != DAG_CKPT_SHA[1 - seed]]'),''',
    '''("init-other-seed", "K-INIT compares with the other seed's M2b", 'if str(v) != want[seed]]', 'if str(v) != want[1 - seed]]'),
    ("init-default-dag", "K-INIT's default map is DAG_s (the pre-A1 init), not M2b_s", 'want = M2B_CKPT_SHA if want is None else want', 'want = DAG_CKPT_SHA if want is None else want'),''')
rep('''("disc-dag", "discover: DAG needs the T0 files too", '    if arm != "DAG":\\n        td = e.get("train_dir")', '    if True:\\n        td = e.get("train_dir")'),''',
    '''("disc-light-off", "discover: the banked CTRL slot is read like a new arm (logged lane, fresh T0 demanded)", '    if arm in light:\\n        td = e.get("train_dir")', '    if False:\\n        td = e.get("train_dir")'),
    ("disc-light-default", "discover: the default light set is empty", 'def discover(manifest, arms=ARMS, seeds=SEEDS, light=("CTRL",)):', 'def discover(manifest, arms=ARMS, seeds=SEEDS, light=()):'),
    ("disc-light-t0", "discover: the light slot needs no T0 eval", '        for f in ("json", "npz"):\\n            need.append((f"t0.{f}", (e.get("t0") or {}).get(f)))', '        for f in ():\\n            need.append((f"t0.{f}", (e.get("t0") or {}).get(f)))'),''')

# ---- the A1 mutants
NEW = r'''
    # ---- A1: the amendment's own logic
    ("a1-amend-no-sha", "amend_problems skips the raw sha256", 'if hashlib.sha256(raw).hexdigest() != pin:\n        return [f"{path} is not the pinned AMEND1.md', 'if False:\n        return [f"{path} is not the pinned AMEND1.md'),
    ("a1-amend-no-phrases", "amend_problems skips the phrases", 'for ph in AMEND_PHRASES if re.sub(', 'for ph in () if re.sub('),
    ("a1-amend-pin-default", "amend_problems default pin is empty", 'pin = AMEND1_SHA256 if pin is None else pin', 'pin = "" if pin is None else pin'),
    ("a1-amend-const", "AMEND1_SHA256 changed", 'AMEND1_SHA256 = "f2f96157', 'AMEND1_SHA256 = "f2f96158'),
    ("a1-oaccread-const", "OACC_READ_SHA256 changed", 'OACC_READ_SHA256 = "9e6bd4f6', 'OACC_READ_SHA256 = "9e6bd4f7'),
    ("a1-m2b-const", "M2B_CKPT_SHA s0 changed", 'M2B_CKPT_SHA = {0: "601cb435', 'M2B_CKPT_SHA = {0: "601cb436'),
    ("a1-bank-const", "OACC_BANK_SHA s1 changed", 'OACC_BANK_SHA = {0: "197a68bb540be5a7d157942e9157e1c8e60ca60d82b1ca4fb4fcb68edccfaa8f", 1: "36c7aea9', 'OACC_BANK_SHA = {0: "197a68bb540be5a7d157942e9157e1c8e60ca60d82b1ca4fb4fcb68edccfaa8f", 1: "36c7aeaa'),
    ("a1-ckpt-const", "OACC_CKPT_SHA s0 changed", 'OACC_CKPT_SHA = {0: "67265ec1', 'OACC_CKPT_SHA = {0: "67265ec2'),
    ("a1-build-const", "BUILD_BASE_COMMIT is DAG2's 1d4f4fbb", 'BUILD_BASE_COMMIT = "aeb54468"', 'BUILD_BASE_COMMIT = "1d4f4fbb"'),
    ("a1-build-files", "BUILD_FILES drops v8f_dagger.py", ', "stack/tanitad/data/v8f_dagger.py")      # A1', ')      # A1'),
    ("a1-wacc-const", "W_ACC changed", 'W_ACC = "49.6969"', 'W_ACC = "49.697"'),
    ("a1-swap-const", "SWAP_P 0.1 -> 0.2", 'SWAP_P = 0.1 ', 'SWAP_P = 0.2 '),
    ("a1-flag-const", "COMPOSE_FLAG renamed", 'COMPOSE_FLAG = "--ego-perturb-with-dagger"', 'COMPOSE_FLAG = "--ego-perturb-with-daggers"'),
    ("a1-bankpin-skip", "banked_pin_problems never compares", '            if got != want:\n                bad.append(f"DAG-OACC_s{s} {what}', '            if False:\n                bad.append(f"DAG-OACC_s{s} {what}'),
    ("a1-bankpin-read", "banked_pin_problems ignores DAGOACC_READ.json", 'if not os.path.isfile(p) or sha256_file(p) != OACC_READ_SHA256:', 'if False:'),
    ("a1-aflags-compose", "a_flags drops the composition flag", '    fl[COMPOSE_FLAG] = [[]]\n    return fl', '    return fl'),
    ("a1-reg-flag-A", "registered argv does not require the flag on A / B", 'if arm in ("A", "B"):\n        if not (has_flag', 'if False:\n        if not (has_flag'),
    ("a1-reg-flag-REP", "registered argv lets REP / CTRL carry the flag", '    elif has_flag:\n        bad.append(f"{COMPOSE_FLAG} is named on {arm}', '    elif False:\n        bad.append(f"{COMPOSE_FLAG} is named on {arm}'),
    ("a1-reg-wacc", "registered argv skips --w-acc", 'if not ok_w or v_w != W_ACC:', 'if False:'),
    ("a1-reg-bank", "registered argv skips the bank sha256", 'if not ok_b or v_b != OACC_BANK_SHA[seed]:', 'if False:'),
    ("a1-reg-dseed", "registered argv skips --dagger-seed", 'if not ok_d or v_d != str(seed):', 'if False:'),
    ("a1-reach-swap-off", "K-REACH skips the swap band", 'if not (SWAP_P - REACH_TOL - 1e-12 <= sr <= SWAP_P + REACH_TOL + 1e-12):', 'if False:'),
    ("a1-reach-swap-wide", "K-REACH swap band widened", 'sr <= SWAP_P + REACH_TOL + 1e-12):', 'sr <= SWAP_P + 0.2):'),
    ("a1-reach-swap-low", "K-REACH swap lower band dropped", 'if not (SWAP_P - REACH_TOL - 1e-12 <= sr <= SWAP_P + REACH_TOL + 1e-12):', 'if not (sr <= SWAP_P + REACH_TOL + 1e-12):'),
    ("a1-reach-swap-fake", "K-REACH takes the swap ratio from the constant", 'sr = dn / de\n', 'sr = SWAP_P\n'),
    ("a1-reach-nodag", "K-REACH accepts a missing run record", 'if not (isinstance(dn, (int, float)) and not isinstance(dn, bool) and isinstance(de, (int, float)) and not isinstance(de, bool) and de > 0 and 0 <= dn <= de):', 'if False:'),
    ("a1-reach-overlap-off", "K-REACH skips the overlap == 0", 'if not (isinstance(ov, int) and not isinstance(ov, bool) and ov == 0):', 'if False:'),
    ("a1-reach-overlap-ge", "K-REACH overlap: >= 0 instead of == 0", 'and not isinstance(ov, bool) and ov == 0):', 'and not isinstance(ov, bool) and ov >= 0):'),
    ("a1-reach-flag-A", "K-REACH does not require the built flag on A", 'if wf is not True:', 'if False:'),
    ("a1-reach-flag-REP", "K-REACH lets REP build the flag", 'if wf not in (False, None):', 'if False:'),
    ("a1-reach-rep-overlap", "K-REACH lets REP carry an overlap", 'if ov not in (0, None):', 'if False:'),
    ("a1-next-swap", "FIXED / NOT ADOPTED routes to DAG-ORC-SWAP again", 'DAG-ORC-SWAP cannot apply, because A already keeps the swap on. Next: B (p 0.1)', 'Next: DAG-ORC-SWAP, under its own PREREG. Next: B (p 0.1)'),
    ("a1-next-crawl", "HARMS (B) routes to NEXT-CRAWL on DAG states", "NEXT-CRAWL on DAG-OACC's states (section 8, branch LAUNCH-INTENT with DAG-OACC's recipe)", "NEXT-CRAWL on DAG states (section 8)"),
    ("a1-next-pi-go", "B needs no PI go in the amended HARMS text", "(B is NOT queued: it needs its own PI go, A1 section 4)", "(B is queued)"),
    ("a1-next-orig-lost", "outcome_block drops the original PREREG cell", '"committed_next_original_prereg": table_row_original(out)', '"committed_next_original_prereg": next_for_arm(out, arm)'),
    ("a1-next-none", "AMENDED_NEXT is ignored", 'AMENDED_NEXT[nm] if AMENDED_NEXT[nm] is not None else nxt', 'nxt'),
    ("a1-compose-pre", "compose_seed ignores pre_a1 for the crawl set", 'c = (CRAWL_SET_PRE_A1 if pre_a1 else CRAWL_SET)[s]', 'c = CRAWL_SET[s]'),
    ("a1-compose-exempt", "compose_seed ignores pre_a1 for the exemption", 'exempt_ = EXEMPT_PRE_A1 if pre_a1 else EXEMPT', 'exempt_ = EXEMPT'),
    ("a1-compose-domain", "compose_seed does not pass the exemption to domain()", 'd_s = domain(t["DAG_P10"], exempt_)', 'd_s = domain(t["DAG_P10"])'),
    ("a1-init-default-map", "k_init default (None) ignored", 'want = M2B_CKPT_SHA if want is None else want', 'want = M2B_CKPT_SHA'),
'''
anchor = '''    ("parse-flags", "parse_flags drops multi-value flags", 'while j < len(argv) and not argv[j].startswith("--"):', 'while j < i + 2 and j < len(argv) and not argv[j].startswith("--"):'),
]'''
rep(anchor, anchor[:-1].rstrip() + NEW + "]")
out = HERE / "read" / "dag_orc_mutation.py"
out.write_text(t, encoding="utf-8", newline="\n")
print(f"dag_orc_mutation.py: {n[0]} edits, sha256 {hashlib.sha256(t.encode()).hexdigest()}")
