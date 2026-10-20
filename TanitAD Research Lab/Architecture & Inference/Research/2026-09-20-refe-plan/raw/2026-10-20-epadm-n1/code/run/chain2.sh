. C:/Users/Admin/par18/n1sim/work/args.sh
echo "ZZCHAIN2_START $(date -Iseconds)"
for j in X-GS117-A C-GS41-A C-GS117-P; do runjob $j; done
echo "ZZCHAIN2_DONE $(date -Iseconds)"
while [ ! -f $N1/work/logs/chain1.done ]; do sleep 5; done
bash $N1/work/finish.sh
