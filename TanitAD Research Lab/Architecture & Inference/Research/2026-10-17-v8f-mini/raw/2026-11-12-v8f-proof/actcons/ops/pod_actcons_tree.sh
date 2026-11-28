# pod_actcons_tree.sh : runs ON THE POD (bash -s). Card PREREG_V8F_PROOF_ADD37 with note N1: the tree of the lever --w-act-cons = tree_leadcond plus the three
# patched files staged in actcons_in (sha256 checked), every PIN line verified, the arms file of chain actcons_a (LA_s0, LA_s1, then the dose arms LA01_s0, LA01_s1)
# with kind HIERFT6, and the argv of LA_s0 compared with that of LFT_s0. No GPU, trains nothing.
set -u
W=/workspace/v8proof; IN=$W/actcons_in; T=$W/tree_actcons; C2=$W/chain2
[ -e $T ] && { echo "REFUSED: tree_actcons exists"; exit 1; }
[ -e $C2/arms_actcons_a.txt ] && { echo "REFUSED: arms_actcons_a.txt exists"; exit 1; }
chk() { [ "$(sha256sum $IN/$1 | cut -c1-64)" = "$2" ] || { echo "STAGED FILE DIFFERS: $1"; exit 1; }; }
chk v8f_mini_train.py 529e6a0d57d3e7d4036ac259d21d8f5025839ac8d9a71367c0e3ef3144e10a80
chk act_cons.py ad537369e02a82e41ccc0f76753437f02af5bc66dc130f863470558399e69797
chk test_v8f_act_cons.py 356e84ba694c450e0d9ba4d7024889682f532d4758d3e676de6d452f4dce4357
chk ARMS_ACTCONS.txt e9cd3c78656cd3deaaf15f9ad35a4c76ae20193b90ef8ae9f1430b5698b65208
echo "four staged files: sha256 equal"
[ "$(sha256sum $W/tree_leadcond/stack/scripts/v8f_mini_train.py | cut -c1-64)" = "85035ca4a3e9545659e1e6ccaa24c74d3386b9b83a889bb7df490373b721d519" ] || { echo "BASE TRAINER DIFFERS"; exit 1; }
cp -a $W/tree_leadcond $T || exit 1
find $T -name __pycache__ -type d -prune -exec true {} + 2>/dev/null
cp $IN/v8f_mini_train.py $T/stack/scripts/v8f_mini_train.py && cp $IN/act_cons.py $T/stack/tanitad/models/act_cons.py && cp $IN/test_v8f_act_cons.py $T/stack/tests/test_v8f_act_cons.py || exit 1
n=0; bad=0
while IFS='|' read -r tag tree rel sha; do
  [ "$tag" = PIN ] || continue
  n=$((n+1)); [ "$(sha256sum $tree/$rel | cut -c1-64)" = "$sha" ] || { bad=$((bad+1)); echo "PIN MISMATCH $rel"; }
done < <(tr -d '\r' < $IN/ARMS_ACTCONS.txt)
echo "PIN lines $n, mismatches $bad"; [ $bad = 0 ] && [ $n -ge 12 ] || exit 1
echo "files that differ from tree_leadcond (expected 3: one changed, two new): $(diff -rq -x __pycache__ $W/tree_leadcond $T | wc -l)"
diff -rq -x __pycache__ $W/tree_leadcond $T | sed "s|$W/||g" | cut -c1-160
cd $C2 || exit 1
mk() { { echo "# IMAGACT card PREREG_V8F_PROOF_ADD37: $2 (fine-tunes of 6,000 steps from LS_s0 / LS_s1, kind HIERFT6), lines of actcons/ARMS_ACTCONS.txt.";
         tr -d '\r' < $IN/ARMS_ACTCONS.txt | grep '^PIN|'; tr -d '\r' < $IN/ARMS_ACTCONS.txt | grep -E "^ARM\|($3)_s[01]\|" | sed 's/|HIER|/|HIERFT6|/'; } > $1;
       echo "$1 sha256 $(sha256sum $1 | cut -c1-64) ; PIN $(grep -c '^PIN|' $1) ; ARM $(grep -c '^ARM|' $1) : $(grep '^ARM|' $1 | cut -d'|' -f2,4 | tr '\n' ' ')"; }
mk arms_actcons_a.txt "LA_s0, LA_s1 at weight 1.0, then LA01_s0, LA01_s1 at weight 0.1" "LA|LA01"
dr() { C2_CARD=$1 bash chain_arms_ft6.sh dryx $2 --dry-run | grep "^TRAIN $3" | sed -e "s|/runs/[a-z_0-9]*/$3|/runs/X/A|g" -e "s|^TRAIN $3|TRAIN A|" -e "s|tree_actcons|TREE|g" -e "s|tree_leadcond|TREE|g"; }
dr PREREG_V8F_PROOF_ADD35 arms_leadcond_a.txt LFT_s0 > /tmp/ac_lft.txt
dr PREREG_V8F_PROOF_ADD37 arms_actcons_a.txt LA_s0 | sed -e 's| --w-act-cons 1.0||' > /tmp/ac_la.txt
dr PREREG_V8F_PROOF_ADD37 arms_actcons_a.txt LA01_s0 | sed -e 's| --w-act-cons 0.1||' > /tmp/ac_la01.txt
echo "TRAIN lines: LFT_s0 $(wc -l < /tmp/ac_lft.txt) LA_s0 $(wc -l < /tmp/ac_la.txt) LA01_s0 $(wc -l < /tmp/ac_la01.txt)"
cmp -s /tmp/ac_lft.txt /tmp/ac_la.txt && echo "argv of LA_s0 equals that of LFT_s0 plus --w-act-cons 1.0 (tree and out folder apart)" || { echo "ARGV OF LA_s0 DIFFERS"; diff <(tr ' ' '\n' < /tmp/ac_lft.txt) <(tr ' ' '\n' < /tmp/ac_la.txt) | head -12; }
cmp -s /tmp/ac_lft.txt /tmp/ac_la01.txt && echo "argv of LA01_s0 equals that of LFT_s0 plus --w-act-cons 0.1" || echo "ARGV OF LA01_s0 DIFFERS"
C2_CARD=PREREG_V8F_PROOF_ADD37 bash chain_arms_ft6.sh dryx arms_actcons_a.txt --dry-run | grep '^TRAIN LA_s1' | grep -o -- '--seed [0-9]*\|--init-ckpt [^ ]*' | tr '\n' ' '; echo
echo "memory.max $(cat /sys/fs/cgroup/memory.max) ; $(grep -E '^anon ' /sys/fs/cgroup/memory.stat) ; GPU jobs $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
