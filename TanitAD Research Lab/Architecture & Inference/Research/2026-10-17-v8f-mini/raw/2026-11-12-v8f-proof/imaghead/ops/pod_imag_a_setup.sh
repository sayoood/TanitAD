# pod_imag_a_setup.sh : runs ON THE POD (bash -s). Card PREREG_V8F_PROOF_ADD36: arms file of chain imag_a (RL_s0, RL_s1) derived from the LS lines of
# arms_leadsup_a.txt (same pinned tree, same PIN lines): the name, the four added flags and their four expectations. It starts nothing.
set -u
cd /workspace/v8proof/chain2 || exit 1
[ -e arms_imag_a.txt ] && { echo "REFUSED: arms_imag_a.txt exists"; exit 1; }
ADD_ARGV='--wm-l1-residual on --wm-l1-roll-k 10 --w-wm-roll 1.0 --wm-l1-diag on'
ADD_EXP='levers.wm_l1_residual=on levers.wm_l1_roll_k=10 levers.w_wm_roll=1 levers.wm_l1_diag=on'
{ echo "# IMAGLEAD card PREREG_V8F_PROOF_ADD36: RL_s0 and RL_s1 = the lines of LS_s0 and LS_s1 (arms_leadsup_a.txt) with the residual form and the rollout loss added.";
  grep '^PIN|' arms_leadsup_a.txt;
  for s in 0 1; do grep "^ARM|LS_s$s|" arms_leadsup_a.txt | tr -d '\r' | awk -F'|' -v OFS='|' -v s=$s -v a="$ADD_ARGV" -v e="$ADD_EXP" '{ $2="RL_s" s; $8=$8 " " a; n=split($9, p, " info."); $9=p[1] " " e " info." p[2]; print }'; done; } > arms_imag_a.txt
echo "arms_imag_a.txt sha256 $(sha256sum arms_imag_a.txt | cut -c1-64) ; PIN lines $(grep -c '^PIN|' arms_imag_a.txt) ; ARM lines $(grep -c '^ARM|' arms_imag_a.txt)"
cmp -s <(grep '^PIN|' arms_imag_a.txt) <(grep '^PIN|' arms_leadsup_a.txt) && echo "PIN lines equal those of arms_leadsup_a.txt" || echo "PIN LINES DIFFER"
grep '^ARM|' arms_imag_a.txt | cut -d'|' -f2-6
grep '^ARM|RL_s0|' arms_imag_a.txt | cut -d'|' -f8
grep '^ARM|RL_s0|' arms_imag_a.txt | cut -d'|' -f9
norm() { sed -e 's|/runs/[a-z_0-9]*/[A-Z]*_s[0-9]|/runs/X/ARM|g' -e 's|^TRAIN [A-Z]*_s[0-9]|TRAIN ARM|' -e 's|--seed [0-9]|--seed K|'; }
C2_CARD=PREREG_V8F_PROOF_ADD24 bash chain_arms.sh dry_ls arms_leadsup_a.txt --dry-run | grep '^TRAIN LS_s0' | norm > /tmp/imag_ls0.txt
C2_CARD=PREREG_V8F_PROOF_ADD36 bash chain_arms.sh dry_rl arms_imag_a.txt --dry-run | grep '^TRAIN RL_s0' | norm > /tmp/imag_rl0.txt
C2_CARD=PREREG_V8F_PROOF_ADD36 bash chain_arms.sh dry_rl arms_imag_a.txt --dry-run | grep '^TRAIN RL_s1' | norm > /tmp/imag_rl1.txt
echo "TRAIN lines: LS_s0 $(wc -l < /tmp/imag_ls0.txt) RL_s0 $(wc -l < /tmp/imag_rl0.txt) RL_s1 $(wc -l < /tmp/imag_rl1.txt)"
[ "$(sed "s| $ADD_ARGV||" /tmp/imag_rl0.txt)" = "$(cat /tmp/imag_ls0.txt)" ] && echo "argv of RL_s0 equals that of LS_s0 plus the four added flags" || { echo "ARGV DIFFERS BEYOND THE ADDED FLAGS"; diff <(tr ' ' '\n' < /tmp/imag_ls0.txt) <(tr ' ' '\n' < /tmp/imag_rl0.txt) | head -20; }
cmp -s /tmp/imag_rl0.txt /tmp/imag_rl1.txt && echo "argv of RL_s1 equals that of RL_s0 apart from name, seed and out folder" || echo "RL_s1 DIFFERS FROM RL_s0"
C2_CARD=PREREG_V8F_PROOF_ADD36 bash chain_arms.sh dry_rl arms_imag_a.txt --dry-run | grep '^TRAIN RL_s1' | grep -o -- '--seed [0-9]*'
ls ../chains | grep "dry_ls\|dry_rl" | tr '\n' ' '
