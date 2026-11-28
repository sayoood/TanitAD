#!/usr/bin/env python3
"""Render box2_ship/chain_v8f_dagoaccorc_r2.sh: the DAG-ORC (DAG-OACC base) chain, RE-RUN (r2) after the r1 chain stopped on K-REACH (2026-10-28).

Same method as r1's make_chain.py (E:/Projects/TanitAD-artifacts/dagorc_rebase/chain/make_chain.py, which is the model of this file and is NOT touched): every pinned value is COMPUTED from a banked artifact, never retyped.
The r2 TEMPLATE is r1's chain_v8f_dagoaccorc.sh.in with EXACT-COUNT replacements (listed in ``R2_EDITS`` below and nowhere else; a replacement whose anchor does not occur exactly the stated number of times aborts):
the template it derives is also written to box2_ship/chain_v8f_dagoaccorc_r2.sh.in so the difference is reviewable with `diff`.

  r2 differs from r1 in exactly these ways (NOTES.md lists each with file:line):
   1. r2_aside_all (oaf_dagoaccorc_r2_lib.sh) is the FIRST action after BOX2_BUSY is taken and the boot probe answered: every r1 artifact on both sides is RENAMED to <name>_r1_kreach_stop (never deleted, never overwritten)
   2. wait_predecessor (overridden in that lib) GOes only after the r1 chain's log ENDED in exactly  ZZV8FMINI-STOP reached-control fail M2b_s0_dagoaccorc
   3. the lever grep additionally requires the trainer fix's stream separation: ONE variable, STREAM_PATTERN ("<file>|<grep -E pattern>")
   4. the run-record check additionally requires the separated stream on A (absent on REP): ONE variable, STREAM_RECORD (NONE | <dotted.path> | <dotted.path>=<json>)
   5. its own names (log, guard out, current file, tar, TIP file, E dir) so r1's evidence is never overwritten; the helper pins (r2 lib sha, r2 checks sha)
   K-REACH bands, the registered argv, the one-lever rule, the guard, the pulls and the tokens are r1's, line for line.

    python make_chain_r2.py [--stream-pattern 'FILE|ERE'] [--stream-record SPEC]     writes box2_ship/chain_v8f_dagoaccorc_r2.sh (LF) + chain_v8f_dagoaccorc_r2.sh.in
    python make_chain_r2.py --check                                                  regenerates in memory (the STREAM_* lines are READ from the file on disk, so setting them by this tool or by hand are both fine) and compares
    The two STREAM_* values default to DEFAULT_PATTERN / DEFAULT_RECORD below (the Master Mind's registered values for the landed trainer fix); a placeholder (PH_*) makes --check-tree and a real run REFUSE (fail closed).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIP2 = HERE.parent / "box2_ship"
R1DIR = Path("E:/Projects/TanitAD-artifacts/dagorc_rebase/chain")
R1_SHIP = R1DIR / "box2_ship"
OACC = Path("E:/Projects/TanitAD-artifacts/v8f-dagoacc")
INSTALLED = Path("C:/Users/Admin/box2_ship")
MIRROR = "C:/Users/Admin/tanitad-push"
PH_PATTERN = '<<UNSET: "<file>|<grep -E pattern>" that only the landed trainer fix satisfies, e.g. "stack/scripts/v8f_mini_train.py|<ERE>">>'
PH_RECORD = '<<UNSET: NONE | <dotted.path> | <dotted.path>=<A json> | <dotted.path>=<A json>;<REP json>, read in summary.json and selfcheck.json>>'
# THE REGISTERED VALUES (Master Mind, 2026-10-28, from the staged trainer fix E:/Projects/TanitAD-artifacts/dagland/compose_fix/stage/): stack/tanitad/data/v8f_mini_data.py:595 `EGO_PERTURB_COMPOSED_STREAM = 1` (absent from e7ea20d1, the r1 tip:
# `git show e7ea20d1:...v8f_mini_data.py | grep -c EGO_PERTURB_COMPOSED_STREAM` = 0); the trainer records levers["ego_perturb_stream"] = comp.ego.stream (v8f_mini_train.py:1739) in selfcheck.json and summary.json: EXACTLY 1 on A, EXACTLY 0 on REP.
DEFAULT_PATTERN = 'stack/tanitad/data/v8f_mini_data.py|^EGO_PERTURB_COMPOSED_STREAM = 1[[:space:]]*$'
DEFAULT_RECORD = 'levers.ego_perturb_stream=1;0'


def sums(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        h, n = line.split(" ", 1)
        out[n.strip().lstrip("*")] = h
    return out


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rep(t: str, old: str, new: str, n: int = 1) -> str:
    assert t.count(old) == n, (old[:90], t.count(old), n)
    return t.replace(old, new)


R2_HEADER = '''# ============================================================================================================================================================================================================
# r2 -- THE RE-RUN of PREREG_V8F_DAG_ORC Amendment 1 (DAG-OACC base), ALL FOUR RUNS FROM THEIR GATES (2026-10-28). The r1 chain (chain_v8f_dagoaccorc.sh) passed its four gates, trained and evaluated A s0 (M2b_s0_dagoaccorc), pulled it,
#   and STOPPED on K-REACH: ego 19161/76898 = 0.2492, must lie in 0.3 +- 0.02 (swap 14688/145037 = 0.1013 in range, overlap 0). The swap and the perturbation shared ONE RNG stream; the trainer fix gives the perturbation a domain-separated stream
#   on the composed path. REP s0, A s1 and REP s1 never ran. This chain is r1 with ONLY these differences (everything else -- the K-REACH bands, the registered argv, the one-lever rule, the guard, the pulls, the tokens -- is r1's, line for line):
#   1. r2_aside_all (oaf_dagoaccorc_r2_lib.sh) is the FIRST action after BOX2_BUSY is taken and the boot probe answered, before anything else touches box 2 or E:: every r1 artifact is RENAMED to <name>_r1_kreach_stop (never deleted, never
#      overwritten) -- the dev box's $ART, then (one cmd job) box 2's run dirs, gate dirs + logs, pinned argv (+ .new / .chk) and T0 eval files of ALL FOUR runs (all four gates re-run at the new commit; the queue SKIPS a stage whose summary.json /
#      eval npz exists). A relaunch of r2 keeps r2's own progress (an aside marker on each side).
#   2. AFTER_OUT = the r1 chain's log, and the predecessor check is r2's: GO ONLY when that log ENDED in exactly "ZZV8FMINI-STOP reached-control fail M2b_s0_dagoaccorc" (v8f_dagoaccorc_r2_chain_checks.py pred-verdict-r2); a DONE, any other STOP,
#      a ZZGUARD-STOP or a disk-floor stop HALTs. (oaf_lib.sh's own wait_predecessor, which GOes after ANY predecessor STOP, is overridden by the r2 lib.)
#   3. STREAM_PATTERN (below): the lever grep ALSO requires the landed fix's stream separation. While it is the placeholder, --check-tree and a real run REFUSE.
#   4. STREAM_RECORD (below): the run-record check ALSO requires the separated stream recorded on A and absent on REP (NONE = the fix records no field). While it is the placeholder, a real run REFUSES.
#   5. its own names, so r1's evidence is never overwritten: chain_v8f_dagoaccorc_r2.out, v8fdagoaccorc_r2_guard.out, v8fdagoaccorc_r2_current.txt, logs/v8f_dagoaccorc_r2/, repo_v8fdagoaccorc_r2_tip.tar, TIP_V8FDAGOACCORC_R2.txt; the r2 lib and r2
#      checks (v8f_dagoaccorc_r2_chain_checks.py) are sha256-pinned like the r1 helpers. The dev-box pull dir stays $ART (the reader and the T1-kin rollouts look there); r1's is $ART_r1_kreach_stop.
# ============================================================================================================================================================================================================
'''

STREAM_BLOCK = '''# ################################################################################################################################################################################################################
# r2: THE TWO VARIABLES THE MASTER MIND SETS FROM THE LANDED TRAINER FIX (nothing else pins them; `python make_chain_r2.py --stream-pattern ... --stream-record ...` rewrites exactly these two lines, or edit them here).
#   STREAM_PATTERN = "<file>|<grep -E pattern>", exactly like a LEVER record below: a line that only the trainer WITH the domain-separated perturbation stream contains (--check-tree and the real run's lever grep require it).
#   STREAM_RECORD  = NONE | <dotted.path> | <dotted.path>=<A json> | <dotted.path>=<A json>;<REP json>: where the fix records the stream in the run's summary.json AND selfcheck.json. NONE = the fix records no field (then K-REACH's perturbed share is the only evidence).
#   SET 2026-10-28 by the Master Mind from the staged trainer fix (dagland/compose_fix/stage): v8f_mini_data.py:595 `EGO_PERTURB_COMPOSED_STREAM = 1` (absent from e7ea20d1, the r1 tip); the trainer records levers.ego_perturb_stream (v8f_mini_train.py:1739):
#   EXACTLY the int 1 on A, EXACTLY the int 0 on REP (absent fails; the gate asserter and the run-record check both enforce it).
#   While either still contains "<<UNSET", --check-tree (STREAM_PATTERN) and a real run (both) REFUSE before anything is waited for or shipped.
STREAM_PATTERN='@STREAM_PATTERN@'
STREAM_RECORD='@STREAM_RECORD@'
# ################################################################################################################################################################################################################
'''

STREAM_FUNCS = '''stream_pattern_problem() {   # rc 0 = STREAM_PATTERN cannot be used (prints why)
  case $STREAM_PATTERN in
    *'<<'*|'') echo "STREAM_PATTERN is still the placeholder: the Master Mind sets it from the landed trainer fix (a '<file>|<grep -E pattern>' record)"; return 0 ;;
    *'|'*) return 1 ;;
    *) echo "STREAM_PATTERN is not a '<file>|<grep -E pattern>' record: $STREAM_PATTERN"; return 0 ;;
  esac
}
stream_record_problem() {   # rc 0 = STREAM_RECORD cannot be used (prints why)
  case $STREAM_RECORD in
    *'<<'*|'') echo "STREAM_RECORD is still the placeholder: the Master Mind sets it from the landed trainer fix (NONE | <dotted.path> | <dotted.path>=<json>)"; return 0 ;;
    *) return 1 ;;
  esac
}
stream_vars_check() {   # the real run's early refusal: nothing is waited for, taken or shipped while either variable is unset
  local w
  w=$(stream_pattern_problem) && { echo "ZZV8FMINI-STOP $w"; return 1; }
  w=$(stream_record_problem) && { echo "ZZV8FMINI-STOP $w"; return 1; }
  return 0
}
'''

DRY_PRED_OLD = ('  echo "== predecessor"; echo "wait for ${AFTER_OUT:-<UNSET: a real run REFUSES to start without AFTER_OUT>}: GO after ^ZZV8FMINI-DONE or a predecessor ^ZZV8FMINI-STOP of its own with BOX2_BUSY released; '
                'HALT on a ZZGUARD-STOP or a held BOX2_BUSY"\n')
DRY_PRED_NEW = ('  echo "== predecessor (r2)"; echo "wait for ${AFTER_OUT:-<UNSET: a real run REFUSES to start without AFTER_OUT>}: GO ONLY after the r1 DAG-ORC chain ENDED in exactly ^ZZV8FMINI-STOP reached-control fail M2b_s0_dagoaccorc '
                '(pred-verdict-r2) with BOX2_BUSY released; HALT on a DONE, any other STOP, a ZZGUARD-STOP, a disk-floor stop or a held BOX2_BUSY"\n'
                '  echo "== r2 aside (the FIRST action after BOX2_BUSY is taken and the boot probe answered; renames only, never a delete or an overwrite)"\n'
                '  echo "dev box: $ART -> ${ART}${R2_SUF} (ASIDE-KEPT if ${ART}${R2_SUF} already exists: the earlier launch moved r1 aside and $ART is r2 progress), then $ART is created empty"\n'
                '  echo "box 2 ($(r2_box_items | wc -l) items, runs root D:\\\\tanitad\\\\runs\\\\v8f_mini_full; marker r2_aside_done_dagoaccorc.txt -> ASIDE-ALREADY-DONE on a relaunch): each -> <item>${R2_SUF}"; r2_box_items | sed \'s/^/  /\'\n'
                '  echo "-- the box-2 job text:"; r2_aside_job_text \'D:\\tanitad\\runs\\v8f_mini_full\' | tr -d \'\\r\' | sed \'s/^/  | /\'\n'
                '  echo "== r2 stream separation (a real run REFUSES while either is the placeholder)"\n'
                '  echo "STREAM_PATTERN=$STREAM_PATTERN"; echo "STREAM_RECORD=$STREAM_RECORD"\n'
                '  w=$(stream_pattern_problem) && echo "!! $w"; w=$(stream_record_problem) && echo "!! $w"\n')

R2_EDITS = [
    # (name, old, new, count)
    ("header-predecessor",
     "#   1. wait for the predecessor (AFTER_OUT, REQUIRED, no default: the DAG-OACC r2 chain's log): GO after ZZV8FMINI-DONE, or after a ZZV8FMINI-STOP of the predecessor's OWN once its BOX2_BUSY is released; HALT on a ZZGUARD-STOP or a BOX2_BUSY\n#      that stays held.",
     "#   1. (r2) FIRST wait for the predecessor (AFTER_OUT, REQUIRED, no default: the r1 DAG-ORC chain's log, chain_v8f_dagoaccorc.out): GO ONLY after it ENDED in exactly ZZV8FMINI-STOP reached-control fail M2b_s0_dagoaccorc (pred-verdict-r2) with its BOX2_BUSY\n#      released; HALT on a DONE, any other STOP, a ZZGUARD-STOP or a BOX2_BUSY that stays held. Then the FIRST action is r2_aside_all (every r1 artifact renamed aside on both sides).",
     1),
    ("header-refresh-tar", "(repo_v8fdagoaccorc_tip.tar, <= 300 MB asserted)", "(repo_v8fdagoaccorc_r2_tip.tar, <= 300 MB asserted)", 1),
    ("usage-dry-run", "#   chain_v8f_dagoaccorc.sh --dry-run ", "#   chain_v8f_dagoaccorc_r2.sh --dry-run ", 1),
    ("usage-check-tree", "#   chain_v8f_dagoaccorc.sh --check-tree ROOT        OFFLINE: the tip-carries-the-lever greps applied to a directory (REFUSES a tree without add_argument(\"--ego-perturb-with-dagger\")",
     "#   chain_v8f_dagoaccorc_r2.sh --check-tree ROOT     OFFLINE: the tip-carries-the-lever greps applied to a directory (REFUSES a tree without add_argument(\"--ego-perturb-with-dagger\") or without STREAM_PATTERN, and while STREAM_PATTERN is the placeholder)", 1),
    ("usage-wait-only", "#   chain_v8f_dagoaccorc.sh --wait-only              OFFLINE: only the predecessor wait (AFTER_OUT, BOX2_BUSY.txt in $B): prints ZZOAF-WAIT-GO or the STOP marker",
     "#   chain_v8f_dagoaccorc_r2.sh --wait-only           OFFLINE: only the r2 predecessor wait (AFTER_OUT = the r1 log, BOX2_BUSY.txt in $B): prints ZZOAF-WAIT-GO or the STOP marker", 1),
    ("usage-real", "#   AFTER_OUT=<log> chain_v8f_dagoaccorc.sh          the real run", "#   AFTER_OUT=<the r1 log> chain_v8f_dagoaccorc_r2.sh   the real run", 1),
    ("markers-log", "# Markers in ITS OWN log (chain_v8f_dagoaccorc.out)", "# Markers in ITS OWN log (chain_v8f_dagoaccorc_r2.out)", 1),
    ("e-dir", "E=$L/v8f_dagoaccorc\n", "E=$L/v8f_dagoaccorc_r2\n", 1),
    ("checks-line", "CHECKS=$HERE_SHIP/v8f_oaf_chain_checks.py; DCHECKS=$HERE_SHIP/v8f_dagoaccorc_chain_checks.py; ASSERT=$HERE_SHIP/gate_dagoaccorc_assert.py\n",
     "CHECKS=$HERE_SHIP/v8f_oaf_chain_checks.py; DCHECKS=$HERE_SHIP/v8f_dagoaccorc_r2_chain_checks.py; ASSERT=$HERE_SHIP/gate_dagoaccorc_r2_assert.py; R2LIB=$HERE_SHIP/oaf_dagoaccorc_r2_lib.sh\n", 1),
    ("assert-sha-comment", "ASSERT_SHA=@ASSERT_SHA@      # gate_dagoaccorc_assert.py as reviewed: a changed asserter is a changed chain\n",
     "ASSERT_SHA=@ASSERT_SHA@      # gate_dagoaccorc_r2_assert.py as reviewed (r1's asserter + ego_stream_problems: levers.ego_perturb_stream EXACTLY 1 on A / 0 on REP; gen/build_r2_asserter.py): a changed asserter is a changed chain\n", 1),
    ("header-asserter", "(gate_dagoaccorc_assert.py: G-DVB,", "(gate_dagoaccorc_r2_assert.py: G-DVB, the built perturbation stream,", 1),
    ("guard-out", "GUARD_OUT=$L/v8fdagoaccorc_guard.out\n", "GUARD_OUT=$L/v8fdagoaccorc_r2_guard.out\n", 1),
    ("current", "CUR=$L/v8fdagoaccorc_current.txt\n", "CUR=$L/v8fdagoaccorc_r2_current.txt\n", 1),
    ("guard-chain-out", "V8F_CHAIN_OUT=$L/chain_v8f_dagoaccorc.out ", "V8F_CHAIN_OUT=$L/chain_v8f_dagoaccorc_r2.out ", 1),
    ("tar-name", "repo_v8fdagoaccorc_tip.tar", "repo_v8fdagoaccorc_r2_tip.tar", 6),
    ("tip-file", "TIP_V8FDAGOACCORC.txt", "TIP_V8FDAGOACCORC_R2.txt", 1),
    ("dchecks-sha", "DCHECKS_SHA=@DCHECKS_SHA@      # v8f_dagoaccorc_chain_checks.py as reviewed\n",
     "DCHECKS_SHA=@DCHECKS_SHA@      # v8f_dagoaccorc_r2_chain_checks.py as reviewed (r1's file + pred-verdict-r2 + the stream-record clause; gen/build_r2_checks.py)\n"
     "R2LIB_SHA=@R2LIB_SHA@      # oaf_dagoaccorc_r2_lib.sh as reviewed (r2_aside_all / wait_predecessor override; gen/build_r2_lib.py): a changed lib is a changed chain\n", 1),
    ("stream-block", "# the lever-in-the-tip checks: \"<file>|<grep -E pattern>\". `.` = a backslash in the queue patterns.\nLEVER=(\n",
     STREAM_BLOCK + "# the lever-in-the-tip checks: \"<file>|<grep -E pattern>\". `.` = a backslash in the queue patterns.\nLEVER=(\n", 1),
    ("stream-funcs", "lever_check() {   # lever_check <git tip | tree ROOT>: 0 iff every LEVER pattern is in its file\n",
     STREAM_FUNCS + "lever_check() {   # lever_check <git tip | tree ROOT>: 0 iff every LEVER pattern AND the r2 STREAM_PATTERN is in its file\n", 1),
    ("lever-loop", "  for rec in \"${LEVER[@]}\"; do\n    f=${rec%%|*}; pat=${rec#*|}\n",
     "  local w; w=$(stream_pattern_problem) && { echo \"MISSING $w\"; miss=$((miss+1)); }\n  for rec in \"${LEVER[@]}\" \"$STREAM_PATTERN\"; do\n    case $rec in *'<<'*|'') continue ;; esac      # the placeholder is reported above\n    f=${rec%%|*}; pat=${rec#*|}\n", 1),
    ("check-tree-count", "echo \"lever check: every one of ${#LEVER[@]} patterns is in $2\"", "echo \"lever check: every one of $((${#LEVER[@]}+1)) patterns (the ${#LEVER[@]} lever greps + STREAM_PATTERN) is in $2\"", 1),
    ("lib-check-r2", "[ \"$(sha256sum \"$DCHECKS\" 2>/dev/null | cut -c1-64)\" = \"$DCHECKS_SHA\" ] || { echo \"ZZV8FMINI-STOP $DCHECKS is not the reviewed checks file ($DCHECKS_SHA): nothing was touched\"; return 1; }\n",
     "[ \"$(sha256sum \"$DCHECKS\" 2>/dev/null | cut -c1-64)\" = \"$DCHECKS_SHA\" ] || { echo \"ZZV8FMINI-STOP $DCHECKS is not the reviewed checks file ($DCHECKS_SHA): nothing was touched\"; return 1; }\n"
     "  [ \"$(sha256sum \"$R2LIB\" 2>/dev/null | cut -c1-64)\" = \"$R2LIB_SHA\" ] || { echo \"ZZV8FMINI-STOP $R2LIB is not the reviewed r2 lib ($R2LIB_SHA): nothing was touched\"; return 1; }\n", 1),
    ("source-r2lib", "source \"$HERE_SHIP/oaf_lib.sh\"\n",
     "source \"$HERE_SHIP/oaf_lib.sh\"\nsource \"$R2LIB\"      # r2: r2_aside_all / r2_aside_job_text / r2_aside_local and the OVERRIDING wait_predecessor (a NEW file: oaf_lib.sh and oaf_r2_lib.sh are not touched)\n", 1),
    ("dry-predecessor", DRY_PRED_OLD, DRY_PRED_NEW, 1),
    ("real-stream-check", "[ -d \"$(dirname \"$ART\")\" ] || { echo \"ZZV8FMINI-STOP $(dirname \"$ART\") does not exist: the artifacts have nowhere to go\"; exit 1; }\n",
     "stream_vars_check || exit 1      # r2: STREAM_PATTERN / STREAM_RECORD must be set from the landed fix BEFORE anything is waited for, taken or shipped\n"
     "[ -d \"$(dirname \"$ART\")\" ] || { echo \"ZZV8FMINI-STOP $(dirname \"$ART\") does not exist: the artifacts have nowhere to go\"; exit 1; }\n", 1),
    ("busy-text", "echo \"v8F-mini PREREG_V8F_DAG_ORC Amendment 1 (re-based on DAG-OACC: A = ", "echo \"v8F-mini PREREG_V8F_DAG_ORC Amendment 1 r2 (the re-run after the r1 K-REACH stop; re-based on DAG-OACC: A = ", 1),
    ("aside-call", "echo \"$(date '+%m-%d %H:%M') boot0=$BOOT0\"\n",
     "echo \"$(date '+%m-%d %H:%M') boot0=$BOOT0\"\n# ---- r2: the FIRST action on box 2 and on E: after BOX2_BUSY and the boot probe: every r1 artifact is moved ASIDE (renamed, never deleted), the dev box's $ART and box 2's runs / gates / pins / evals (oaf_dagoaccorc_r2_lib.sh)\nr2_aside_all\n", 1),
    ("run-record-stream", "--w-acc $WACC --steps $DAG_STEPS || stop \"run DAG-ORC (DAG-OACC base) record $g\"", "--w-acc $WACC --steps $DAG_STEPS --stream-record \"$STREAM_RECORD\" || stop \"run DAG-ORC (DAG-OACC base) record $g\"", 1),
    ("finish-text", "finish \"PREREG_V8F_DAG_ORC Amendment 1 (DAG-OACC base): ", "finish \"PREREG_V8F_DAG_ORC Amendment 1 (DAG-OACC base) r2: ", 1),
]


def derive_template() -> str:
    t = (R1_SHIP / "chain_v8f_dagoaccorc.sh.in").read_bytes().decode("utf-8")
    assert "\r" not in t
    for name, old, new, n in R2_EDITS:
        try:
            t = rep(t, old, new, n)
        except AssertionError as e:
            raise AssertionError(f"r2 edit {name!r}: {e}")
    # the r2 header goes right after the shebang line
    first, rest = t.split("\n", 1)
    assert first.startswith("#!/usr/bin/env bash")
    t = first + "\n" + R2_HEADER + rest
    # no r1-only name may survive in a LIVE position (comments excepted): the only chain_v8f_dagoaccorc.sh / .out mentions left must be the r1-log references
    return t


def render(stream_pattern: str, stream_record: str) -> str:
    for v in (stream_pattern, stream_record):
        assert "'" not in v and "\n" not in v, "the STREAM_* values are single-quoted shell words: no ' and no newline"
    t = derive_template()
    sub = {}
    r2 = (INSTALLED / "chain_v8f_dagoacc_r2.sh").read_text(encoding="utf-8")

    def r2arr(var):
        m = re.search(rf"^{var}=\(([0-9a-f]{{64}}) ([0-9a-f]{{64}})\)", r2, re.M)
        assert m, f"{var} not in the DAG-OACC r2 chain"
        return m.group(1), m.group(2)
    r2_m2b, r2_bank = r2arr("M2B_SHA"), r2arr("BANK1_SHA")
    wacc_txt = (OACC / "wacc" / "w_acc.txt").read_bytes()
    wacc = wacc_txt.decode("ascii").strip()
    assert re.fullmatch(r"[0-9]+(\.[0-9]+)?", wacc), wacc
    for s in (0, 1):
        d = OACC / f"M2b_s{s}_dagoacc"
        sm = sums(d / "SHA256SUMS.txt")
        sub[f"OACC_CKPT_SHA{s}"], sub[f"OACC_ARGV_SHA{s}"], sub[f"OACC_SUMM_SHA{s}"] = sm["ckpt_last.pt"], sm["argv.json"], sm["summary.json"]
        assert sha(d / "argv.json") == sm["argv.json"] and sha(d / "summary.json") == sm["summary.json"], f"DAG-OACC_s{s}: banked files != SHA256SUMS.txt"
        tok = json.loads((d / "launch_gate_token.json").read_text(encoding="utf-8"))
        summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        argv = json.loads((d / "argv.json").read_text(encoding="utf-8"))["argv"]
        m2b = tok["init_ckpt_sha256"]
        assert m2b == summ["levers"]["init_ckpt"]["sha256"] == r2_m2b[s], f"M2b_s{s}: token / summary / r2 chain disagree"
        bank = argv[argv.index("--dagger-bank-sha256") + 1]
        assert bank == r2_bank[s] == tok.get("bank_sha256", bank), f"bank s{s}: argv.json / r2 chain disagree"
        assert argv[argv.index("--w-acc") + 1] == wacc, "DAG-OACC's argv.json --w-acc != w_acc.txt"
        sub[f"M2B_SHA{s}"], sub[f"BANK1_SHA{s}"] = m2b, bank
        commit = tok["binding"]["commit"]
        assert re.fullmatch(r"[0-9a-f]{40}", commit) and commit.startswith("aeb54468"), commit
        sub["BUILD_REF"] = commit
    sub["WACC"], sub["WACC_TXT_SHA"] = wacc, hashlib.sha256(wacc_txt).hexdigest()
    mir = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "rev-parse", "aeb54468^{commit}"], capture_output=True, text=True, check=True).stdout.strip()
    assert mir == sub["BUILD_REF"], "the mirror's aeb54468 != DAG-OACC's token commit"
    # the reviewed helpers: r1's asserter must be the INSTALLED one (the record of r1, untouched); the r2 asserter / checks / lib are this directory's generated files
    assert sha(R1_SHIP / "gate_dagoaccorc_assert.py") == sha(INSTALLED / "gate_dagoaccorc_assert.py"), "r1's asserter is not the installed one"
    asr = SHIP2 / "gate_dagoaccorc_r2_assert.py"
    sub["ASSERT_SHA"] = sha(asr)
    sub["DCHECKS_SHA"], sub["R2LIB_SHA"] = sha(SHIP2 / "v8f_dagoaccorc_r2_chain_checks.py"), sha(SHIP2 / "oaf_dagoaccorc_r2_lib.sh")
    sub["OAF_LIB_SHA"], sub["OAF_CHECKS_SHA"] = sha(INSTALLED / "oaf_lib.sh"), sha(INSTALLED / "v8f_oaf_chain_checks.py")
    m = re.search(r"^EXPORT=\(\n(.*?)\n\)\n", r2, re.M | re.S)
    assert m, "no EXPORT block in the DAG-OACC r2 chain"
    lines = [x for x in m.group(1).split("\n") if x.strip()]
    assert all(re.fullmatch(r"  '[^']+'", x) for x in lines), "EXPORT lines must be quoted paths"
    for need in ("stack/scripts/v8f_mini_train.py", "stack/scripts/v8f_mini_eval.py", "stack/tanitad/data/v8f_dagger.py", "stack/tanitad/data/v8f_mini_data.py", "stack/tanitad/models/v8f_mini.py"):
        assert any(f"'{need}'" in x for x in lines), need
    sub["EXPORT"] = "\n".join(lines)
    sub["STREAM_PATTERN"], sub["STREAM_RECORD"] = stream_pattern, stream_record
    for k, v in sub.items():
        assert t.count(f"@{k}@") >= 1, f"placeholder @{k}@ not in the template"
        t = t.replace(f"@{k}@", v)
    left = re.findall(r"@[A-Z0-9_]+@", t)
    assert not left, f"unfilled placeholders {left}"
    return t


def read_stream_from_disk(path: Path):
    """The two STREAM_* lines of an existing chain file (so --check works however they were set)."""
    txt = path.read_text(encoding="utf-8")
    mp, mr = re.search(r"^STREAM_PATTERN='([^'\n]*)'$", txt, re.M), re.search(r"^STREAM_RECORD='([^'\n]*)'$", txt, re.M)
    assert mp and mr, "no STREAM_PATTERN / STREAM_RECORD line in " + str(path)
    return mp.group(1), mr.group(1)


def main() -> int:
    a = sys.argv[1:]
    out = SHIP2 / "chain_v8f_dagoaccorc_r2.sh"

    def opt(name, default):
        return a[a.index(name) + 1] if name in a else default
    if "--check" in a:
        sp, sr = read_stream_from_disk(out)
        text = render(sp, sr)
        ok = out.exists() and out.read_bytes() == text.encode("utf-8") and (SHIP2 / "chain_v8f_dagoaccorc_r2.sh.in").read_bytes() == derive_template().encode("utf-8")
        print("chain_v8f_dagoaccorc_r2.sh == a fresh regeneration:", ok)
        return 0 if ok else 1
    if out.exists() and "--stream-pattern" not in a and "--stream-record" not in a:      # keep the values already set (a placeholder on disk gets the registered default)
        dp, dr = read_stream_from_disk(out)
        dp, dr = (DEFAULT_PATTERN if "<<" in dp else dp), (DEFAULT_RECORD if "<<" in dr else dr)
    else:
        dp, dr = DEFAULT_PATTERN, DEFAULT_RECORD
    sp, sr = opt("--stream-pattern", dp), opt("--stream-record", dr)
    text = render(sp, sr)
    (SHIP2 / "chain_v8f_dagoaccorc_r2.sh.in").write_bytes(derive_template().encode("utf-8"))
    out.write_bytes(text.encode("utf-8"))
    print(f"wrote {out} sha256 {hashlib.sha256(text.encode()).hexdigest()} ({len(text)} bytes)")
    print("STREAM_PATTERN =", sp)
    print("STREAM_RECORD  =", sr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
