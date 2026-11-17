"""fork_x.py -- forks the block-3 chain files into /root/dscale10k/blockx/ for the conditional arm X_s0.  Every replacement is asserted to match the stated number of times (-1 = at least once),
so a changed block-3 file fails loudly.  The block-3 files (MM-ruled, running now) are only READ.  Pod, CPU, a few KB of text."""
import os

D = "/root/dscale10k"
B3, BX = D + "/block3", D + "/blockx"
os.makedirs(BX, exist_ok=True)


def fork(src, dst, reps, header=None, mode=None):
    s = open(f"{B3}/{src}", encoding="utf-8").read()
    for a, b, n in reps:
        c = s.count(a)
        assert (c >= 1) if n == -1 else (c == n), (src, a[:80], c, n)
        s = s.replace(a, b)
    if header:
        lines = s.split("\n")
        k = 1 if lines[0].startswith("#!") else 0
        lines[k:k] = [header]
        s = "\n".join(lines)
    open(f"{BX}/{dst}", "w", encoding="utf-8", newline="\n").write(s)
    if mode:
        os.chmod(f"{BX}/{dst}", mode)
    print("wrote", dst, len(s))


FORK = "# FORK of block3/%s for the conditional arm X_s0 (the 10,000 data-scale clips alone; card 2.3 row X_s0, section 6 DATA-HURTS). Changes: blockx/FORK_NOTES.md. NOT launched: the MM asks the PI first."

# ---- init_x.py
fork("init_b3.py", "init_x.py", [
    ('B3 = D + "/block3"\nsys.path.insert(0, B3 + "/study")', 'B3 = D + "/blockx"\nsys.path.insert(0, B3 + "/study")', 1),
    ('for arm in ("UE_s0", "BL_s0"):', 'for arm in ("X_s0",):', 1),
    ('"BL_s0_vs_B_s0_params_equal": out["BL_s0"]["params_only_sha256"] == base["params_only_sha256"],', '"X_s0_vs_B_s0_params_equal": out["X_s0"]["params_only_sha256"] == base["params_only_sha256"],', 1),
    ('"BL_s0_vs_UE_s0_params_equal": out["BL_s0"]["params_only_sha256"] == out["UE_s0"]["params_only_sha256"],', '"X_s0_vs_B_s0_state_equal": out["X_s0"]["state_sha256"] == base["state_sha256"],', 1),
    ('"BL_s0_vs_UE_s0_state_equal": out["BL_s0"]["state_sha256"] == out["UE_s0"]["state_sha256"],', '"X_s0_vs_B_s0_buffers_equal": out["X_s0"]["buffers_sha256"] == base["buffers_sha256"],', 1),
    ('                        "BL_s0_vs_UE_s0_buffers_equal": out["BL_s0"]["buffers_sha256"] == out["UE_s0"]["buffers_sha256"],\n', '', 1),
    ('bad = copy.deepcopy(out["UE_s0"])', 'bad = copy.deepcopy(out["X_s0"])', 1),
    ('GI.save("init_b3", out, probs', 'GI.save("init_x", out, probs', 1),
], FORK % "init_b3.py")

# ---- assemble_evidence_x.py: X_s0 reads init_x; reach table = UE_s0's tenk record + control (same code and release; INHERITED); fixed step count
fork("assemble_evidence_b3.py", "assemble_evidence_x.py", [
    ('init_r = item("init_b3")', 'init_r = item("init_x")', 1),
    ('probe = f"{G}/probe2/{arm}.A.json"', 'probe = f"{G}/probe2/UE_s0.A.json"      # X_s0: the union reach table of UE_s0 (real data, same tree and release); only its tenk record and the corpus control enter. INHERITED, not re-measured on the X view', 1),
    ('kinds = ("corpus", "tenk", "mixed") if arm == "UE_s0" else ("corpus",)', 'kinds = ("tenk",)', 1),
    ('if arm == "UE_s0":', 'if arm == "X_s0":', 1),
    ('        out["ue_steps"] = 4 * int(L(udry)["S_U"])\n', '', 1),
    ('        out["ue_steps_source"] = {"path": udry, "sha256": sha(udry), "S_U": int(L(udry)["S_U"])}\n', '        out["reach_source"] = "INHERITED from gate/probe2/UE_s0.A.json (tenk record + control)"\n', 1),
], FORK % "assemble_evidence_b3.py")

# ---- verify_token_x.py
fork("verify_token3.py", "verify_token_x.py", [
    ('D + "/study", D + "/block3/study"):', 'D + "/study", D + "/blockx/study"):', 1),
    ('os.path.realpath(D + "/block3/study/dscale10k_check.py")', 'os.path.realpath(D + "/blockx/study/dscale10k_check.py")', 1),
    ('not {D}/block3/study/dscale10k_check.py', 'not {D}/blockx/study/dscale10k_check.py', 1),
    ('if arm in ("U_s0", "U_s1", "UE_s0", "R_s0"):', 'if arm in ("U_s0", "U_s1", "UE_s0", "R_s0", "X_s0"):', 1),
], FORK % "verify_token3.py")

# ---- chk_armx.sh
fork("chk_arm3.sh", "chk_armx.sh", [
    ('B3=${DS_B3:-$D/block3}', 'B3=${DS_BX:-$D/blockx}', 1),
    ('  U_s0|U_s1|UE_s0) FILES=', '  U_s0|U_s1|UE_s0|X_s0) FILES=', 1),
    ('  case $ARM in UE_s0|BL_s0) MORE="$MORE --ue-dry $D/dry/U_s0.A.dry.json";; esac\n', '  [ "$ARM" = X_s0 ] && MORE="--corpus-join $D/merged/join_union.npz --union-join $D/merged/join_X.npz"      # X: the UNION join sits in the corpus slot, the X join in the union slot (blockx study fork)\n', 1),
], FORK % "chk_arm3.sh")

# ---- blockx_prep.sh
fork("block3_prep.sh", "blockx_prep.sh", [
    ('B3=${DS_B3:-$D/block3}', 'B3=${DS_BX:-$D/blockx}', 1),
    ('ZZ BLOCK3-PREP', 'ZZ BLOCKX-PREP', -1),
    ('for f in gen_argv_b3.py init_b3.py assemble_evidence_b3.py verify_token3.py chk_arm3.sh study/dscale10k_check.py;', 'for f in xview.py gen_argv_x.py init_x.py assemble_evidence_x.py verify_token_x.py chk_armx.sh study/dscale10k_check.py;', 1),
    ('ARMS="UE_s0 BL_s0"', 'ARMS="X_s0"', 1),
    ('[ -s argv/UE_s0.A.json ] && [ -s argv/BL_s0.A.json ] || { DS_ROOT=$D $PY $B3/gen_argv_b3.py > $G/gen_argv_b3.log 2>&1 || fail "gen_argv_b3: $(tail -c 300 $G/gen_argv_b3.log)"; }',
     '[ -d /root/data/cache_x_dscale/train/clips ] && [ -s /root/data/cache_x_dscale/train/manifest.jsonl ] || { $PY $B3/xview.py > $G/xview.log 2>&1 || fail "xview: $(tail -c 300 $G/xview.log)"; }\n[ -s argv/X_s0.A.json ] && [ -s argv/X_s0.B.tmpl.json ] || { DS_ROOT=$D $PY $B3/gen_argv_x.py > $G/gen_argv_x.log 2>&1 || fail "gen_argv_x: $(tail -c 300 $G/gen_argv_x.log)"; }', 1),
    ('if [ ! -s $G/init_b3.json ]; then CUDA_VISIBLE_DEVICES= $PY $B3/init_b3.py > $G/init_b3.log 2>&1 || fail "init_b3: $(tail -c 400 $G/init_b3.log)"; fi\necho "$(date -u +%T) init_b3: $(tail -n 1 $G/init_b3.log | cut -c1-200)"',
     'if [ ! -s $G/init_x.json ]; then CUDA_VISIBLE_DEVICES= $PY $B3/init_x.py > $G/init_x.log 2>&1 || fail "init_x: $(tail -c 400 $G/init_x.log)"; fi\necho "$(date -u +%T) init_x: $(tail -n 1 $G/init_x.log | cut -c1-200)"', 1),
    ('mkdir -p $G/probe2\nfor a in $ARMS; do\n  [ -s $G/probe2/$a.A.json ] && continue\n  n=0; until gpu_idle; do n=$((n+1)); [ $n -gt 360 ] && fail "GPU not idle before the $a reach probe"; sleep 10; done\n  $PY $G/reach_probe2.py $D/argv/$a.A.json $G/probe2/$a.A.json $D/merged/v9_labels_union.manifest.json --reach > $G/probe2/$a.A.log 2>&1 || fail "reach probe $a: $(tail -c 400 $G/probe2/$a.A.log)"\n  echo "$(date -u +%T) reach probe $a: $(tail -n 1 $G/probe2/$a.A.log | cut -c1-200)"\ndone\n',
     '[ -s $G/probe2/UE_s0.A.json ] || fail "no gate/probe2/UE_s0.A.json: the X evidence takes its tenk record and control from UE_s0 (no GPU probe is run for X)"\n', 1),
    ('$PY $B3/assemble_evidence_b3.py $ARMS > $G/assemble_b3.log 2>&1 || fail "assemble_evidence_b3: $(tail -c 400 $G/assemble_b3.log)"', '$PY $B3/assemble_evidence_x.py $ARMS > $G/assemble_x.log 2>&1 || fail "assemble_evidence_x: $(tail -c 400 $G/assemble_x.log)"', 1),
    ('bash $B3/chk_arm3.sh mint $a A > $G/mint_$a.A.log', 'bash $B3/chk_armx.sh mint $a A > $G/mint_$a.A.log', 1),
    ('$PY $B3/verify_token3.py $a A > $G/verify_$a.A.log', '$PY $B3/verify_token_x.py $a A > $G/verify_$a.A.log', 1),
], FORK % "block3_prep.sh", 0o755)

# ---- blockx.sh
fork("block3.sh", "blockx.sh", [
    ('B3=${DS_B3:-$D/block3}', 'B3=${DS_BX:-$D/blockx}', 1),
    ('STEPS_A=61568; STEPS_B=3000;', 'STEPS_A=19164; STEPS_B=3000;', 1),
    ('ZZ DSCALE10K BLOCK3', 'ZZ DSCALE10K BLOCKX', -1),
    ('for a in B_s0 U_s0 B_rep B_s1 U_s1; do [ -e $D/runs/$a/ARM_DONE ] && $LIB sweep $a; done;', 'for a in B_s0 U_s0 B_rep B_s1 U_s1 X_s0; do [ -e $D/runs/$a/ARM_DONE ] && $LIB sweep $a; done;', 1),
    ('DS_ROOT=$D bash $B3/chk_arm3.sh run $2 $3', 'DS_ROOT=$D bash $B3/chk_armx.sh run $2 $3', 1),
    ('DS_ROOT=$D bash $B3/chk_arm3.sh mint $2 $3', 'DS_ROOT=$D bash $B3/chk_armx.sh mint $2 $3', 1),
    ('DS_ROOT=$D bash $B3/block3_prep.sh', 'DS_ROOT=$D bash $B3/blockx_prep.sh', 1),
    ('for f in chainlib.py dscale_banker.py block3/chk_arm3.sh block3/block3_prep.sh block3/study/dscale10k_check.py; do', 'for f in chainlib.py dscale_banker.py blockx/chk_armx.sh blockx/blockx_prep.sh blockx/study/dscale10k_check.py blockx/xview.py; do', 1),
    ('[ -n "${DS_FAKE:-}" ] || $PY - $D <<\'E\' || fail "S_U check: 4 x S_U of dry/U_s0.A.dry.json is not $STEPS_A"\nimport json, sys\nassert 4 * json.load(open(sys.argv[1] + "/dry/U_s0.A.dry.json"))["S_U"] == 61568\nE\n', '', 1),
    ('for a in UE_s0 BL_s0; do', 'for a in X_s0; do', 2),
    ('do_arm UE_s0 no\ndo_arm BL_s0 no\n', 'do_arm X_s0 yes\n', 1),
    ('block 3 prep', 'block X prep', -1),
    ('block3_prep.out', 'blockx_prep.out', -1),
    ('ZZ BLOCK3-PREP DONE ZZ', 'ZZ BLOCKX-PREP DONE ZZ', -1),
    ('block 3 start', 'block X start', 1),
], FORK % "block3.sh", 0o755)

# ---- queue_blockx.sh: waits for the NAVRB2 chain's end line, then takes the GPU lock
fork("queue_block3.sh", "queue_blockx.sh", [
    ('L=${QUEUE_LOG:-/root/strat1/logs/c0rep.log}', 'L=${QUEUE_LOG:-/root/strat1/logs/navrb2_queue.log}', 1),
    ('CHAIN=${QUEUE_CHAIN:-/root/dscale10k/block3/block3.sh}', 'CHAIN=${QUEUE_CHAIN:-/root/dscale10k/blockx/blockx.sh}', 1),
    ('BLOCK3 queued behind the C0REP chain', 'BLOCKX queued behind the NAVRB2 chain', 1),
    ("'^ZZ STRAT1 C0REP (DONE|FAIL) ZZ'", "'^ZZ STRAT1 NAVRB2 (DONE|FAIL) ZZ'", 2),
    ('ZZ DSCALE10K BLOCK3 FAIL ZZ queue: no C0REP end line after 48 h', 'ZZ DSCALE10K BLOCKX FAIL ZZ queue: no NAVRB2 end line after 48 h', 1),
    ('C0REP ended:', 'NAVRB2 ended:', 1),
], FORK % "queue_block3.sh", 0o755)

# ---- fake leaf for the dry run
fork("fake_leaf3.sh", "fake_leaf_x.sh", [
    ('    n=61568\n', '    n=19164; [ "$ST" = B ] && n=3000\n', 1),
    ('for x in UE_s0 BL_s0; do', 'for x in X_s0; do', 1),
    ('ZZ BLOCK3-PREP DONE ZZ', 'ZZ BLOCKX-PREP DONE ZZ', 1),
], FORK % "fake_leaf3.sh", 0o755)
print("ZZ FORK-X DONE ZZ")
