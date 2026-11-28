. C:/Users/Admin/par18/n1sim/work/env.sh
ARGS="--pool C:/Users/Admin/par18/epadm_search/work/carveB_bank2 --stats $N1/work/stats_r2confirm.npz --basex $N1/work/sim/basex_n1.npz --geometry $N1/geometry_s0.json --workdir $N1/work/sim"
runjob() { "$PY" -I "$N1/code/epadm_sim_n1.py" run --job "$1" $ARGS; "$PY" -I "$N1/code/epadm_sim_n1.py" repro --job "$1" $ARGS; }
