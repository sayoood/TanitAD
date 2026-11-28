# pod_ruledose_setup.sh : runs ON THE POD (bash -s). Note PREREG_V8F_PROOF_ADD32_N1, stage 1: the six fine-tunes of 6,000 steps (two chains, one per base).
# Derives the chain tools of 6,000 steps from the pinned originals by fixed, checked edits, writes the two arms files from arms_ruletrain_a.txt, dry-gates GRT6_s0
# and GFT6_s1, checks the token kinds. It trains nothing; the chains are started by the lead in a second call.
set -u
W=/workspace/v8proof; cd $W/chain2 || exit 1
for f in arms_ruledose_a.txt arms_ruledose_b.txt; do [ -e $f ] && { echo "REFUSED: $f exists"; exit 1; }; done
[ -e $W/chains/ruledose_a ] || [ -e $W/chains/ruledose_b ] && { echo "REFUSED: a ruledose chain folder exists"; exit 1; }
# gate token tool: kind HIERFT6 = 6,000 steps, init checkpoint applied
sed -e 's|^EXPECT = {"HIERFT": ("full", 2000, 0.0),|EXPECT = {"HIERFT6": ("full", 6000, 0.0), "HIERFT": ("full", 2000, 0.0),|' \
    -e 's|    if a.kind == "HIERFT" and |    if a.kind in ("HIERFT", "HIERFT6") and |' gate_token3.py > gate_token4.py
[ "$(grep -c HIERFT6 gate_token4.py)" = 2 ] && [ "$(diff gate_token3.py gate_token4.py | grep -c '^[<>]')" = 4 ] && $W/../venv/bin/python -c 'import ast; ast.parse(open("gate_token4.py").read())' || { echo "REFUSED: gate_token4.py is not the expected edit of gate_token3.py"; exit 1; }
sed -e 's|19164|6000|g' -e 's|gate_token2\.py|gate_token4.py|g' chain_arms.sh > chain_arms_ft6.sh
[ "$(grep -c 19164 chain_arms_ft6.sh)" = 0 ] && [ "$(grep -c gate_token2 chain_arms_ft6.sh)" = 0 ] && [ "$(diff chain_arms.sh chain_arms_ft6.sh | grep -c '^>')" = "$(grep -c '19164\|gate_token2\.py' chain_arms.sh)" ] && bash -n chain_arms_ft6.sh || { echo "REFUSED: chain_arms_ft6.sh is not the expected edit of chain_arms.sh"; exit 1; }
echo "chain tools: gate_token4.py $(sha256sum gate_token4.py | cut -c1-64); chain_arms_ft6.sh $(sha256sum chain_arms_ft6.sh | cut -c1-64)"; diff gate_token3.py gate_token4.py | cut -c1-200
# arms files: the lines of arms_ruletrain_a.txt with the name, the kind and the step count changed; nothing else
mk() { # mk <out> <seed suffix of the base: 0 or 1>
  { echo "# RULETRAIN note PREREG_V8F_PROOF_ADD32_N1, stage 1: fine-tunes of 6,000 steps from G_s$2; GFT6 control, GRT6 with --rule-relabel on, GFT6b replicate of the control.";
    grep '^PIN|' arms_ruletrain_a.txt;
    for n in GFT GRT GFTb; do
      grep "^ARM|${n}_s$2|" arms_ruletrain_a.txt | sed -e "s/^ARM|${n}_s$2|/ARM|$(echo $n | sed 's/^\(G[FR]T\)/\16/')_s$2|/" -e 's/|HIERFT|/|HIERFT6|/' -e 's/ --steps 2000 / --steps 6000 /'
    done; } > "$1"
}
mk arms_ruledose_a.txt 0; mk arms_ruledose_b.txt 1
for f in arms_ruledose_a.txt arms_ruledose_b.txt; do
  echo "== $f sha256 $(sha256sum $f | cut -c1-64) ; PIN $(grep -c '^PIN|' $f) ; ARM $(grep -c '^ARM|' $f) ; steps 6000 lines $(grep -c -- '--steps 6000 --lr 1e-4 --warmup 100' $f) ; relabel lines $(grep -c -- '--rule-relabel on' $f)"
  grep '^ARM|' $f | cut -d'|' -f2,4,5
done
# the argv of each new arm equals that of its 2,000-step arm apart from name, out folder and step count
C2_CARD=PREREG_V8F_PROOF_ADD32 bash chain_arms_ft.sh dryx arms_ruletrain_a.txt --dry-run | grep '^TRAIN' | sed -e 's|/runs/[a-z_0-9]*/G[FR]T6\{0,1\}\(b\{0,1\}\)_s|/runs/X/GXT\1_s|g' -e 's|^TRAIN G[FR]T6\{0,1\}\(b\{0,1\}\)_s|TRAIN GXT\1_s|' -e 's|--steps [0-9]*|--steps N|' | sort > /tmp/argv_2k.txt
{ C2_CARD=PREREG_V8F_PROOF_ADD32 bash chain_arms_ft6.sh dryx arms_ruledose_a.txt --dry-run; C2_CARD=PREREG_V8F_PROOF_ADD32 bash chain_arms_ft6.sh dryx arms_ruledose_b.txt --dry-run; } | grep '^TRAIN' | sed -e 's|/runs/[a-z_0-9]*/G[FR]T6\{0,1\}\(b\{0,1\}\)_s|/runs/X/GXT\1_s|g' -e 's|^TRAIN G[FR]T6\{0,1\}\(b\{0,1\}\)_s|TRAIN GXT\1_s|' -e 's|--steps [0-9]*|--steps N|' | sort > /tmp/argv_6k.txt
echo "argv lines 2k $(grep -c . /tmp/argv_2k.txt) 6k $(grep -c . /tmp/argv_6k.txt)"; cmp -s /tmp/argv_2k.txt /tmp/argv_6k.txt && echo "argv of the six new arms equals that of the six arms of the card apart from name, out folder and step count" || { echo "ARGV DIFFERS"; diff /tmp/argv_2k.txt /tmp/argv_6k.txt | cut -c1-260 | head -n 8; }
{ C2_CARD=PREREG_V8F_PROOF_ADD32 bash chain_arms_ft6.sh dryx arms_ruledose_a.txt --dry-run; } | grep '^TRAIN GRT6_s0' | grep -o -- '--steps [0-9]* --lr [0-9e.-]* --warmup [0-9]*\|--rule-relabel on\|--seed [0-9]*' | tr '\n' ' '; echo
echo "memory.max $(cat /sys/fs/cgroup/memory.max) ; $(grep -E '^anon ' /sys/fs/cgroup/memory.stat)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';'; echo
ls dry_gate.sh >/dev/null && grep -n "chain_arms\|gate_token" dry_gate.sh | cut -c1-200 | head -n 6
