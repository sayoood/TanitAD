"""Builds rh_arm.sh (patched copy) from ../b2/rh_arm.sh. Every replacement asserts it matches exactly once."""
s = open("../b2/rh_arm.sh", encoding="utf-8", newline="").read()
assert "\r" not in s


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


rep("""# Order, as STAGEB (chk_arm.sh / block2.sh) and N1DC (n1dc_arm.sh) do it: idle GPU -> pair check (on the filled argv) -> trainer --selfcheck-only on THIS argv (real bank for RH, real init) -> mint the chain's gate record
#   -> verify it -> train (3,000 steps) -> summary check""", """# Order, as STAGEB (chk_arm.sh / block2.sh) and N1DC (n1dc_arm.sh) do it: idle GPU -> pair check (on the filled argv) -> trainer --selfcheck-only on THIS argv (real bank for RH, real init) -> mint the chain's gate record
#   -> verify it -> MINT THE STUDY-SIDE GATE TOKEN (chk_arm_rh.sh mint -> dscale10k_check_rh.py, kinds RH / B0) -> train (3,000 steps) THROUGH THE TOKEN (chk_arm_rh.sh run: rc 2 without it) -> summary check""")

# 1. the pair check: the STOCK committed rollhist_pair_check.py (never patched in process); causal_g1 is refused by it, and by the checker
rep("""  $PY $HERE/render_argv_rh.py pair $D/argv.json $R/runs/B0_s$seed/argv.json --bank-sha256 $sha > $D/pair_$tag.log 2>&1 || afail "pair check: $(tail -c 300 $D/pair_$tag.log)\"""",
    """  $PY $TREE/stack/scripts/rollhist_pair_check.py --rh $D/argv.json --b0 $R/runs/B0_s$seed/argv.json --bank-sha256 $sha > $D/pair_$tag.log 2>&1 || afail "pair check (stock rollhist_pair_check.py; it hard-codes causal_g1x): $(tail -c 300 $D/pair_$tag.log)\"""")
rep("""  $PY $HERE/render_argv_rh.py pair $R/runs/RH_s$seed/argv.json $D/argv.json ${sha:+--bank-sha256 $sha} > $D/pair_$tag.log 2>&1 || afail "pair check: $(tail -c 300 $D/pair_$tag.log)\"""",
    """  $PY $TREE/stack/scripts/rollhist_pair_check.py --rh $R/runs/RH_s$seed/argv.json --b0 $D/argv.json ${sha:+--bank-sha256 $sha} > $D/pair_$tag.log 2>&1 || afail "pair check (stock rollhist_pair_check.py; it hard-codes causal_g1x): $(tail -c 300 $D/pair_$tag.log)\"""")

# 2. mint the study-side token after the chain-local record
rep("""$PY $HERE/rh_gate_record.py verify --tag $tag > $D/verify.log 2>&1 || afail "gate record verify: $(tail -c 300 $D/verify.log | tr '\\n' ' ')"
""", """$PY $HERE/rh_gate_record.py verify --tag $tag > $D/verify.log 2>&1 || afail "gate record verify: $(tail -c 300 $D/verify.log | tr '\\n' ' ')"
# the study-side gate token (dscale10k_check_rh.py mint: every check the stage-B mint runs + the RH / B0 argv, bank generator record and pair checks); bound to the commit, argv sha256 and data sha256s
bash $HERE/chk_arm_rh.sh mint $tag > $D/mint_token.log 2>&1 || afail "gate token mint refused: $(tail -c 500 $D/mint_token.log | tr '\\n' ' ')"
grep -q "^TOKEN-WRITTEN $tag" $D/mint_token.log && [ -s $D/gate/gate_token.json ] || afail "no gate token written: $(tail -c 300 $D/mint_token.log | tr '\\n' ' ')"
""")

# 3. train through the token
old = """CUDA_VISIBLE_DEVICES=0 $PY - $D/argv.json $TREE/stack/scripts/v8f_mini_train.py $PY > $D/train.log 2>&1 <<'PYE'
import json, subprocess, sys
toks = json.load(open(sys.argv[1]))["argv"]
sys.exit(subprocess.run([sys.argv[3], sys.argv[2]] + [str(t) for t in toks]).returncode)
PYE
rc=$?
[ $rc -eq 0 ] || afail "trainer rc=$rc: $(tail -c 400 $D/train.log | tr '\\n' ' ')\""""
new = """bash $HERE/chk_arm_rh.sh run $tag > $D/train.log 2>&1      # the checker's runner: REFUSED (rc 2, nothing started) unless the token binds exactly this argv, commit, data, tree, trainer, checker, template and pair
rc=$?
[ $rc -eq 0 ] || afail "trainer / gate runner rc=$rc: $(tail -c 400 $D/train.log | tr '\\n' ' ')\""""
rep(old, new)
open("rh_arm.sh", "w", encoding="utf-8", newline="").write(s)
print("rh_arm.sh written")
