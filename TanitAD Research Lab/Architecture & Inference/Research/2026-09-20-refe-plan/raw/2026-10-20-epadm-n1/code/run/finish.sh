. C:/Users/Admin/par18/n1sim/work/args.sh
cd "$N1/code"
echo "ZZFINISH_START $(date -Iseconds)"
"$PY" -I "$N1/code/epadm_sim_s0.py" pc8 --stats "$N1/work/stats_r2confirm.npz" --banked "$N1/work/a2_power_g25_g05_replicates.npz" --geometry "$N1/geometry_s0.json" --workdir "$N1/work/sim"
rm -rf "$N1/work/pc3_tmp"; mkdir -p "$N1/work/pc3_tmp"
"$PY" -I "$N1/code/epadm_sim_n1.py" pc3 --workdir "$N1/work/pc3_tmp"
cp "$N1/work/pc3_tmp/pc3.json" "$N1/work/sim/pc3.json"
rm -rf "$N1/work/pc3_tmp"
"$PY" -I "$N1/code/epadm_sim_n1.py" report --workdir "$N1/work/sim"
"$PY" -I "$N1/code/make_tables_n1.py" "$N1/work/sim/sim_results.json" > "$N1/work/tables.md"
rm -rf "$N1/work/pt"; mkdir -p "$N1/work/pt" "$N1/work/mut"
"$PY" -I -m pytest test_n1.py -q -p no:cacheprovider --basetemp="$N1/work/pt/a" > "$N1/work/logs/pytest_n1.log" 2>&1
"$PY" -I -m pytest test_sim_s0.py -q -p no:cacheprovider --basetemp="$N1/work/pt/b" > "$N1/work/logs/pytest_sim_s0_copy.log" 2>&1
MUT_TMP="$N1/work/mut" "$PY" -I mutation_check_n1.py "$PY" "$N1/work/mutation.json" > "$N1/work/logs/mutation_n1.log" 2>&1
rm -rf "$N1/work/pt" "$N1/work/mut"
"$PY" -I "$N1/code/make_result_n1.py" > "$N1/RESULT.md"
"$PY" -I "$N1/code/stage.py"
echo "ZZFINISH_DONE $(date -Iseconds)"
echo done > $N1/work/logs/finish.done
