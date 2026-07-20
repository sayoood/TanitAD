#!/bin/bash
cd /root/taniteval
export PYTHONPATH=/root/taniteval:/root/TanitAD/stack
export TANITEVAL_STACK_OVERRIDE=/root/models/assess-20260719/stack-v2b
LOG=results/refbv2_30k_suite.log
: > $LOG
say(){ echo "=== $* ($(date -u +%T)) ==="; }
say "SUITE START" | tee -a $LOG
say "core run refb-v2-30k" | tee -a $LOG
python3 -m taniteval.runner run --model refb-v2-30k --episodes 40 2>&1 | tee -a $LOG
say "core run refb-v2-20k (prior; refresh windows for AB)" | tee -a $LOG
python3 -m taniteval.runner run --model refb-v2-20k --episodes 40 2>&1 | tee -a $LOG
say "generalize physicalai (in-dist ref)" | tee -a $LOG
python3 -m taniteval.runner generalize --model refb-v2-30k --corpus physicalai --episodes 40 2>&1 | tee -a $LOG
say "generalize comma2k19 (OOD)" | tee -a $LOG
python3 -m taniteval.runner generalize --model refb-v2-30k --corpus comma --episodes 40 2>&1 | tee -a $LOG
say "generalize cosmos (OOD; provisional)" | tee -a $LOG
python3 -m taniteval.runner generalize --model refb-v2-30k --corpus cosmos --episodes 40 2>&1 | tee -a $LOG
say "imagination (planner -> expect skip)" | tee -a $LOG
python3 -m taniteval.runner imagination --model refb-v2-30k 2>&1 | tee -a $LOG
say "hierarchy (planner -> expect skip)" | tee -a $LOG
python3 -m taniteval.runner hierarchy --model refb-v2-30k 2>&1 | tee -a $LOG
say "AB refb-v2-30k vs flagship-30k" | tee -a $LOG
python3 -m taniteval.runner ab --a refb-v2-30k --b flagship-30k 2>&1 | tee -a $LOG
say "AB refb-v2-30k vs refb-v2-20k (prior)" | tee -a $LOG
python3 -m taniteval.runner ab --a refb-v2-30k --b refb-v2-20k 2>&1 | tee -a $LOG
say "SUITE DONE" | tee -a $LOG
