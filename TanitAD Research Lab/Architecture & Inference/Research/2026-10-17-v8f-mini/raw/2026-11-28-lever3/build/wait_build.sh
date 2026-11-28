#!/bin/bash
# Wait (script-side) until both build jobs have written their rc files, or 12 h pass. Prints a one-line state.
B=/f/TanitAD-scratch/rollhist/lever3/build
for i in $(seq 1 720); do
  if [ -f "$B/tenk.rc" ] && [ -f "$B/corpus.rc" ]; then break; fi
  sleep 60
done
echo "tenk.rc: $(cat $B/tenk.rc 2>/dev/null) | corpus.rc: $(cat $B/corpus.rc 2>/dev/null) | TENK.done: $(cat $B/state_tenk/TENK.done 2>/dev/null) | CORPUS.done: $(cat $B/state_corpus/CORPUS.done 2>/dev/null)"
ls $B/state_tenk/*.done.json 2>/dev/null | wc -l
tail -2 $B/tenk.log
