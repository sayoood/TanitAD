# pod_motion_e_setup.sh : runs ON THE POD (bash -s). Note PREREG_V8F_PROOF_ADD18_N2: arms file of chain motion_e (RR_s2, RR_s3) derived from the RR_s1 line of
# arms_motion_d2.txt (name and seed field only), dry run compared with RR_s1. It starts nothing.
set -u
cd /workspace/v8proof/chain2 || exit 1
[ -e arms_motion_e.txt ] && { echo "REFUSED: arms_motion_e.txt exists"; exit 1; }
{ echo "# MOTION note PREREG_V8F_PROOF_ADD18_N2: RR_s2 and RR_s3, the line of RR_s1 (arms_motion_d2.txt) with the name and the seed changed.";
  grep '^PIN|' arms_motion_d2.txt;
  for s in 2 3; do grep '^ARM|RR_s1|' arms_motion_d2.txt | awk -F'|' -v OFS='|' -v s=$s '{ $2="RR_s" s; $5=s; print }'; done; } > arms_motion_e.txt
echo "arms_motion_e.txt sha256 $(sha256sum arms_motion_e.txt | cut -c1-64) ; PIN lines $(grep -c '^PIN|' arms_motion_e.txt) ; ARM lines $(grep -c '^ARM|' arms_motion_e.txt)"
grep '^ARM|' arms_motion_e.txt | cut -d'|' -f2-6
C2_CARD=PREREG_V8F_PROOF_ADD18 bash chain_arms.sh dry_e arms_motion_e.txt --dry-run | grep '^TRAIN' | sed -e 's|/runs/[a-z_]*/RR_s[0-9]|/runs/X/RR_sK|g' -e 's|^TRAIN RR_s[0-9]|TRAIN RR_sK|' -e 's|--seed [0-9]|--seed K|' | sort -u | wc -l
C2_CARD=PREREG_V8F_PROOF_ADD18 bash chain_arms.sh dry_d arms_motion_d2.txt --dry-run | grep '^TRAIN RR_s1' | sed -e 's|/runs/[a-z_]*/RR_s[0-9]|/runs/X/RR_sK|g' -e 's|^TRAIN RR_s[0-9]|TRAIN RR_sK|' -e 's|--seed [0-9]|--seed K|' > /tmp/rr1.txt
C2_CARD=PREREG_V8F_PROOF_ADD18 bash chain_arms.sh dry_e arms_motion_e.txt --dry-run | grep '^TRAIN RR_s2' | sed -e 's|/runs/[a-z_]*/RR_s[0-9]|/runs/X/RR_sK|g' -e 's|^TRAIN RR_s[0-9]|TRAIN RR_sK|' -e 's|--seed [0-9]|--seed K|' > /tmp/rr2.txt
cmp -s /tmp/rr1.txt /tmp/rr2.txt && echo "argv of RR_s2 equals that of RR_s1 apart from name, seed and out folder" || { echo "ARGV DIFFERS"; diff /tmp/rr1.txt /tmp/rr2.txt | cut -c1-300; }
C2_CARD=PREREG_V8F_PROOF_ADD18 bash chain_arms.sh dry_e arms_motion_e.txt --dry-run | grep '^TRAIN RR_s3' | grep -o -- '--seed [0-9]*'
sed -n 70,96p chain_arms.sh | cut -c1-230
ls ../chains | grep "dry_e\|dry_d" | tr '\n' ' '
