. C:/Users/Admin/par18/n1sim/work/args.sh
echo "ZZCHAIN1_START $(date -Iseconds)"
for j in C-GS117-A X-GS41-A C-GS41-P; do runjob $j; done
echo "ZZCHAIN1_DONE $(date -Iseconds)"
echo done > $N1/work/logs/chain1.done
