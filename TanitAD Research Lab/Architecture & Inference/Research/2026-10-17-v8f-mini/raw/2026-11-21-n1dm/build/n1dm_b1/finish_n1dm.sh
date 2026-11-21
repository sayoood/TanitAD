#!/bin/bash
# finish_n1dm.sh -- detached finisher of the N1DM build (nohup). ONE python at a time, each step commit-gated by the scripts it calls. Resumable (each step leaves logs/fin_<step>.done). Done when logs/finish.log ends with FINISHED.
B=F:/TanitAD-scratch/n1dc/build/n1dm_b1; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; LOG=$B/logs/finish.log
cd $B || exit 1
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
step() { n=$1; shift; [ -e logs/fin_$n.done ] && { say "skip $n"; return; }; say "step $n"; "$@" >> $LOG 2>&1; rc=$?; say "$n rc=$rc"; echo $rc > logs/fin_$n.done; }
say "start"
until grep -q ALLDONE regress_new/_summary.txt; do sleep 30; done; say "regress ALLDONE"
step check_retest bash runpt.sh t_check tests/test_strat1_n1dm_check.py -rfE
mkdir -p tmp/mut
step mutants $PY -I -u run_mutants_n1dm.py "$B/tree" "$B/mutation_table_n1dm.json" "$B/tmp/mut"
mkdir -p tmp/tiny
step tiny_arm bash envrun.sh "$B/tree" tiny_arm "$B/mk_tiny_arm.py" "$B/tmp/tiny/w"
step post_lines bash dry_post_lines.sh "$B/tmp/tiny/w"
step read_dry bash F:/TanitAD-scratch/n1dc/read_n1dm/dry_check_n1dm.sh
step regress_cmp $PY -I regress_cmp.py
# hashes, patch, delta tar, manifest
git_hash() { : > newhash.txt; while read f; do echo "$(git hash-object "tree/$f") $f" >> newhash.txt; done < changed_files.txt
  awk 'NR==FNR{h[$2]=$1; next} {p=substr($0,42); if (p in h) print h[p], p; else print $0}' newhash.txt pod_ref/all_blobs.txt > all_blobs_n1dm.txt
  awk 'NR==FNR{s[substr($0,42)]=1; next} !($2 in s){print $1, $2}' pod_ref/all_blobs.txt newhash.txt >> all_blobs_n1dm.txt; }
step hashes git_hash
step patch $PY -I mkpatch_n1dm.py
cd $B && ( cd tree && tar -cf ../n1dm_delta.tar $(cat ../changed_files.txt) ) && sha256sum $(sed 's#^#tree/#' changed_files.txt) | sed 's# tree/# *#' > n1dm_tree.sha256
say FINISHED
