"""Derive chain_v8f_tacgrp2.sh (PREREG_V8F_TAC Amendment 2) from the FIXED chain_v8f_tacgrp.sh; every replacement must match exactly once."""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
assert "\r" not in s


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old, s.count(old))
    s = s.replace(old, new)


rep("""# PREREG_V8F_TAC Amendment 1 on box 2 (EVAL ONLY, zero training): WHICH part of v8F-mini's tactical code carries the oracle's headroom?
# One job at a time under SAFE MODE. ARMED ONLY ON THE PI's GO. Derived from chain_v8f_tac.sh; the differences are the predecessor,
# the stage (tacgrp: 4 checkpoints x 8 group oracles = 32 evals), the pulls and the reader. Its markers are ZZV8FTAC-DONE / -STOP in
# ITS OWN log (chain_v8f_tacgrp.out), so v8f_guard.sh's anchored end test already knows them.
#   0. wait for the predecessor (default: PREREG_V8F_TAC's chain, ZZV8FTAC-DONE; stop on its ZZV8FTAC-STOP); AFTER_* override it
""", """# PREREG_V8F_TAC Amendment 2 on box 2 (EVAL ONLY, zero training): the joint lon pair, leave-one-out, and the constraints dose-response.
# One job at a time under SAFE MODE. ARMED ONLY ON THE PI's GO. Derived from the FIXED chain_v8f_tacgrp.sh (mk_chain_grp2.py); the
# differences are the predecessor, the stage (tacgrp2: 4 checkpoints x 10 oracles = 40 evals), the pulls and the reader. Its markers are
# ZZV8FTAC-DONE / -STOP in ITS OWN log (chain_v8f_tacgrp2.out), so v8f_guard.sh's anchored end test already knows them.
#   0. wait for the predecessor (default: Amendment 1's finisher, ZZV8FTAC-DONE; stop on its ZZV8FTAC-STOP); AFTER_* override it
""")
rep("#   2. `v8f_mini_box2_queue.cmd tacgrp`: M1 / M2b x seeds 0, 1 x 8 group oracles = 32 evals of ~25 s each\n"
    "#   3. pull them + PREREG_V8F_TAC's tacpred / tacorc runs; code/v8f_tac_groups_read.py -> tac_groups_read.json\n",
    "#   2. `v8f_mini_box2_queue.cmd tacgrp2`: M1 / M2b x seeds 0, 1 x 10 oracles = 40 evals of ~13 s each\n"
    "#   3. pull them + tacpred / tacorc / Amendment 1's tacorc-constraints; code/v8f_tac_groups2_read.py -> tac_groups2_read.json\n")
rep("B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_tacgrp; mkdir -p $E/eval\n",
    "B=/c/Users/Admin/box2_ship; L=$B/logs; E=$L/v8f_tacgrp2; mkdir -p $E/eval\n")
rep('TAC_GROUPS="goal latdec londec latcons lonspeed stoplead decisions constraints"\n',
    '# not GROUPS: a bash builtin that ignores assignment (TRAPS.md; stack/tests/test_bash_specials.py)\n'
    'TAC_SPECS="lonspeed+londec all-goal all-latdec all-londec all-latcons all-lonspeed all-stoplead constraints-a25 constraints-a50 constraints-a75"\n')
rep("AFTER_OUT=${AFTER_OUT:-$L/chain_v8f_tac.out}", "AFTER_OUT=${AFTER_OUT:-$L/finish_v8f_tacgrp.out}")
rep('echo "v8F-mini per-group oracle (PREREG_V8F_TAC Amendment 1, eval only) since $(date \'+%m-%d %H:%M\'), ~30 min.',
    'echo "v8F-mini tactical oracle reads (PREREG_V8F_TAC Amendment 2, eval only) since $(date \'+%m-%d %H:%M\'), ~15 min.')
rep("grep -q '^:tacgrp' || stop \"the tip's queue has no tacgrp stage\"", "grep -q '^:tacgrp2' || stop \"the tip's queue has no tacgrp2 stage\"")
rep('git -C $MIRROR cat-file -e "$T:$PK/code/v8f_tac_groups_read.py" || stop "the tip has no v8f_tac_groups_read.py"',
    'git -C $MIRROR cat-file -e "$T:$PK/code/v8f_tac_groups2_read.py" || stop "the tip has no v8f_tac_groups2_read.py"')
s = s.replace("repo_v8ftacgrp_tip.tar", "repo_v8ftacgrp2_tip.tar").replace("v8ftacgrp_refresh", "v8ftacgrp2_refresh")
s = s.replace("TIP_V8FTACGRP.txt", "TIP_V8FTACGRP2.txt").replace("v8ftacgrp_tip_", "v8ftacgrp2_tip_")
rep('[ -f "$RD/v8f_tac_groups_read.py" ] || stop "the extract has no v8f_tac_groups_read.py"',
    '[ -f "$RD/v8f_tac_groups2_read.py" ] || stop "the extract has no v8f_tac_groups2_read.py"')
rep("[ -f $L/v8f_tac/tac_read.json ] || stop \"PREREG_V8F_TAC's tac_read.json is missing locally\"",
    "[ -f $L/v8f_tacgrp/tac_groups_read.json ] || stop \"Amendment 1's tac_groups_read.json is missing locally\"")
s = s.replace("v8ftacgrp_guard.out", "v8ftacgrp2_guard.out").replace("v8ftacgrp_current.txt", "v8ftacgrp2_current.txt")
rep("V8F_CHAIN_OUT=$L/chain_v8f_tacgrp.out", "V8F_CHAIN_OUT=$L/chain_v8f_tacgrp2.out")
rep("# ---- 2. the 32 group-oracle evals\n", "# ---- 2. the 40 Amendment 2 evals\n")
s = s.replace("v8ftacgrp_run", "v8ftacgrp2_run")
rep('/mk_v8f_job.py" v8ftacgrp2_run tacgrp >/dev/null', '/mk_v8f_job.py" v8ftacgrp2_run tacgrp2 >/dev/null')
rep("""  for s in tacpred tacorc; do $SCP "tanitad-box2:$EVR/eval_$c-$s.npz" $E/eval/ || stop "pull eval_$c-$s.npz"; done
  for g in $TAC_GROUPS; do s=tacorc-$g
""", """  for s in tacpred tacorc tacorc-constraints; do $SCP "tanitad-box2:$EVR/eval_$c-$s.npz" $E/eval/ || stop "pull eval_$c-$s.npz"; done
  $SCP "tanitad-box2:$EVR/eval_$c-tacorc-constraints.json" $E/eval/ || stop "pull eval_$c-tacorc-constraints.json"
  for g in $TAC_SPECS; do s=tacorc-$g
""")
rep('''PYTHONIOENCODING=utf-8 $PY "$RD/v8f_tac_groups_read.py" --eval-dir "$(cygpath -m $E/eval)" --tac-read "$(cygpath -m $L/v8f_tac/tac_read.json)" \\
  --out "$(cygpath -m $E/tac_groups_read.json)" || stop "tac groups read"''',
    '''PYTHONIOENCODING=utf-8 $PY "$RD/v8f_tac_groups2_read.py" --eval-dir "$(cygpath -m $E/eval)" \\
  --groups-read "$(cygpath -m $L/v8f_tacgrp/tac_groups_read.json)" --out "$(cygpath -m $E/tac_groups2_read.json)" || stop "tac groups2 read"''')
s = s.replace("telemetry_v8ftacgrp.csv", "telemetry_v8ftacgrp2.csv")
rep("""print(json.load(open(sys.argv[1],encoding='utf-8'))['reading'])" "$(cygpath -m $E/tac_groups_read.json)")""",
    """print(json.load(open(sys.argv[1],encoding='utf-8'))['reading'])" "$(cygpath -m $E/tac_groups2_read.json)")""")
assert "tacgrp\"" not in s and "tac_groups_read.py" not in s and "GROUPS=" not in s.replace("# not GROUPS", ""), "a leftover"
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
