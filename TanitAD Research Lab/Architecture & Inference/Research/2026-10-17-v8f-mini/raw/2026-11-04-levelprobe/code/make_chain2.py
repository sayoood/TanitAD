#!/usr/bin/env python3
"""make_chain2.py -- derive levelprobe2_chain.sh (PREREG_LEVELPROBE2.md) from levelprobe_chain.sh by exact-match edits; each must match as stated.  python -I code/make_chain2.py"""
PK = "E:/Projects/TanitAD-artifacts/dagland/levelprobe"
s = open(f"{PK}/levelprobe_chain.sh", encoding="utf-8").read()
head, _ = s.split("nfail=0; nskip=0\n")
head = head.split("# LEVELPROBE Stage-2 rollout chain")[0] + "".join([
    "# LEVELPROBE2 rollout chain (H-LEVELPROBE-2, PREREG_LEVELPROBE2.md).  Dev-box CPU, zero training, 16 threads, ONE run at a time.\n",
    "#   LP2KA_DAGOACC_s0 (L_T0 level file; must equal HT0_DAGOACC_s0 bit for bit), then M6_* (MEAN6), F3_* (FUT3), F1_* (FUT1) on the four cells, all `--smooth sg72 --histswap cv_pred --level-file`.\n",
    "# Deadline: every rollout must have FINISHED by 05:00 on 2026-11-04; a run that would START after 04:45 is skipped and the chain fails saying so.  RESUME = launch again (finished runs are skipped).\n",
    "# Markers: ZZLP2-* in logs/chain2.out; raw/LP2.done on success (raw/LEVELPROBE2.json) / raw/LP2.fail (reason inside).\n"]) + "set -u\n" + head.split("set -u\n", 1)[1]
for a, b in (("ZZLP-", "ZZLP2-"), ("chain.out", "chain2.out"), ("raw/LP.done", "raw/LP2.done"), ("raw/LP.fail", "raw/LP2.fail"), ("LP.done LP.fail", "LP2.done LP2.fail"), ("_old_${m}_", "_old_${m}_"),
             ("chain.launch.out", "chain2.launch.out")):
    head = head.replace(a, b)
head = head.replace("PREREG_LEVELPROBE.md", "PREREG_LEVELPROBE2.md")
a = """want=$(sed -n 's/^PREREG_LEVELPROBE2.md sha256 \\([0-9a-f]*\\)$/\\1/p' "$PK/FROZEN_SHA.txt")"""
assert a in head, "card-hash line not found"
head = head.replace(a, a.replace("FROZEN_SHA.txt", "FROZEN_SHA2.txt"))
assert 'prereg does not hash to FROZEN_SHA.txt' in head
head = head.replace("prereg does not hash to FROZEN_SHA.txt", "prereg does not hash to FROZEN_SHA2.txt")
tail = r'''lvl() { sed -n "s#^level_file \(.*/L_$1.npz\) sha256 [0-9a-f]*\$#\1#p" "$PK/FROZEN_SHA2.txt" | tail -1; }
lvs() { sed -n "s#^level_file .*/L_$1.npz sha256 \([0-9a-f]*\)\$#\1#p" "$PK/FROZEN_SHA2.txt" | tail -1; }
for k in T0 MEAN6 FUT1 FUT3; do
  f=$(lvl $k); [ -n "$f" ] && [ -f "$f" ] && [ "$(sha256sum "$f" | cut -c1-64)" = "$(lvs $k)" ] || fail "level file L_$k missing or does not hash as pinned in FROZEN_SHA2.txt" 2
done
nfail=0; nskip=0
step() { run_one "$@"; local rc=$?; if [ $rc -eq 9 ]; then nskip=$((nskip+1)); elif [ $rc -ne 0 ]; then nfail=$((nfail+1)); fi; }
step LP2KA_DAGOACC_s0 "$DAG" "${CK[DAGOACC_s0]}" --smooth sg72 --histswap cv_pred --level-file "$(lvl T0)"
for pair in "M6 MEAN6" "F3 FUT3" "F1 FUT1"; do
  set -- $pair
  for as in DAGOACC_s0 DAGOACC_s1 SOUP_s0 SOUP_s1; do
    step "$1_$as" "${TREE[$as]}" "${CK[$as]}" --smooth sg72 --histswap cv_pred --level-file "$(lvl $2)"
  done
done
log "ZZLP2-ROLLOUTS-DONE nfail=$nfail nskip=$nskip"
[ "$nskip" -eq 0 ] || fail "$nskip rollouts skipped by the 04:45 start deadline" 9
missing=""
for t in LP2KA_DAGOACC_s0 LPKA1_DAGOACC_s0 HT0_DAGOACC_s0; do [ -f "$OUT/done_$t.txt" ] || missing="$missing $t"; done
for p in M6 F3 F1; do for as in DAGOACC_s0 DAGOACC_s1 SOUP_s0 SOUP_s1; do [ -f "$OUT/done_${p}_$as.txt" ] || missing="$missing ${p}_$as"; done; done
[ -z "$missing" ] || fail "missing done markers:$missing" 3
log "ZZLP2-READ start"
"$PY" -I "$PK/code/levelprobe2_read.py" --out-dir "$OUT" --raw "$PK/raw" > "$PK/logs/read2.out" 2>&1; rc=$?
log "ZZLP2-READ rc=$rc"
[ $rc -eq 0 ] || fail "read rc=$rc (see logs/read2.out)" "$rc"
echo "$(date +%FT%T) LEVELPROBE2.json written" > "$PK/raw/LP2.done"
log "ZZLP2-CHAIN-DONE"
'''
open(f"{PK}/levelprobe2_chain.sh", "w", encoding="utf-8", newline="\n").write(head + "DAG=$S/tree_dag; LI=$S/tree_li\n" if "DAG=$S/tree_dag" not in head else head)
# the head cut above stops before the DAG/LI + CK/TREE declarations (they sit after run_one in the source); re-append them
decl = s.split("nfail=0; nskip=0\n")[0].split("DAG=$S/tree_dag; LI=$S/tree_li\n")[1]
txt = open(f"{PK}/levelprobe2_chain.sh", encoding="utf-8").read()
if "declare -A CK TREE" not in txt:
    txt = txt.rstrip("\n") + "\n" + decl
open(f"{PK}/levelprobe2_chain.sh", "w", encoding="utf-8", newline="\n").write(txt + tail)
open(f"{PK}/levelprobe2_chain.cmd", "w", newline="").write('@echo off\r\nrem LEVELPROBE2 chain launcher via WMI Win32_Process.Create.  RESUME = launch again.\r\n"C:\\Program Files\\Git\\bin\\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagland/levelprobe/levelprobe2_chain.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/levelprobe/logs/chain2.launch.out 2>&1" < NUL\r\n')
print("wrote levelprobe2_chain.sh / .cmd")
