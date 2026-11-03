#!/usr/bin/env bash
# r9e1_bind.sh -- THOR-SIDE producer of the two BINDING overfit records for the refcv9 E1 perception rig
# (launch-gate profile `refcv9-e1rig`; C12-E1 blocker 2).
#
#   r9e1_bind.sh [--dry-run] <map|box|map-banded|box-b3> <ARM> --commit SHA40 [--root DIR] [--code DIR] [--inputs DIR]
#
# ARM (E1-REC2) = any `stack/ops/runs.d/r9e1_<ARM>.argv.json` of the shipped tree (C0 C1 M1 M2a M2b B3 B3p M3 ...). G-MAP-OVERFIT and G-BOX-OVERFIT bind
# the WHOLE argv sha256, so EVERY arm needs its OWN record of BOTH kinds, and the KIND must match what the argv builds -- refused otherwise:
#     map         the inherited /1 8-code head       REFUSED on an argv carrying --r9-map-banded   ("use map-banded")
#     map-banded  the BANDED head's /2-banded record REFUSED on an argv WITHOUT --r9-map-banded
#     box         the inherited box record           REFUSED on an argv carrying an OWN-HEAD --r9-box-* lever (E1_BOX_OWN_FLAGS: "use box-b3")
#     box-b3      the B3 head's /2-b3 record         REFUSED on an argv WITHOUT any own-head --r9-box-* lever
#   => M1/M2a/M2b: `box` + `map-banded`;  M3: `box` + `map-banded` (the record builds the own encoder from the argv);  B3/B3p: `map` + `box-b3`.
#   E1-REC3 (MM 207ebd, pinned): the TRAINING-ONLY aux heads (E1_BOX_AUX_FLAGS: B1 --r9-box-depth-aux, B2 --r9-box-persp-aux, B2c --r9-box-bev2-aux)
#   are NOT own-head levers: B1/B2/B2c take `map` + `box` (the inherited /1 record bound to their OWN argv; the harness does not build the aux
#   head and the record STAMPS `aux_heads_not_certified` == the argv's aux set -- the gate's /1 judge reads it, and this script asserts it after the
#   run). A `--r9-box-*` flag in NEITHER tuple is REFUSED ("unknown box lever: classify it" in launch_gate.py). Both tuples are READ from the gate.
#   E1-REC4 (MM 207ebd, pinned): T1 (`--r9-bev-warp`, `--r9-bev-warp-lags`, `--r9-bev-warp-shuffle` = launch_gate.E1_T1_FLAGS, read from the gate) changes the
#   heads' INPUT, not their structure: T1 / T1s take `map` + `box` (the inherited /1 record bound to their OWN argv; a banded base would take
#   map-banded / box-b3 the same way). The harness does NOT build T1 (frozen frames, no staged history: the build argv strips it) and the record STAMPS
#   `bev_warp_not_certified` == the argv's T1 set -- the gate's judges read it, and this script asserts it after the run (T1JUDGE-OK). Kind `map` hands the /1 map
#   harness the launch argv (`--launch-argv`, appended for a T1 argv only: every other arm's `map` args are byte-identical to what they were).
#
# box-b3 (E1-BOX-OWN-RECORD): the SAME lock discipline and closure_run --binding as `box`, with the template `stack/scripts/r9e1_bind_box-b3.args.tpl`
#   (g_box_overfit.py, arms main,memory_zeros,presence_w0 -- the /2-b3 record needs RESULT PASS, which the harness writes only when BOTH must-fail
#   arms ran and failed; the A19 MAIN-only `box` record is RESULT FAIL by construction) and the record `BD/run_box-b3/gbo_binding.json` (schema
#   tanitad.g_box_overfit_record/2-b3), read back by the GATE'S OWN judges (`judge_box_overfit` re-judge + `judge_box_own_record` via
#   `own_record_closes`) instead of the /1 assert. THREE arms train on the Thor GPU (MEASURED for `box`: MAIN alone 5,065.7 s) -- a detached call of its own.
#
# map-banded (E1-MAP-OWN-RECORD): the SAME lock discipline and closure_run --binding as `map`, with the template
#   `stack/scripts/r9e1_bind_map-banded.args.tpl` (map_hires_overfit.py --banded: the A18 spec, 3 arms healthy,lane_w0,s8_zeros, near lift 10)
#   and the record `BD/run_map-banded/g_map_overfit.json` (schema tanitad.g_map_overfit_record/2-banded) read back by the GATE'S OWN
#   judge (`judge_map_overfit_banded` on the arm's argv) instead of the /1 assert. Needs {INPUTS}/gmo_spec_A18.json (sha256 checked)
#   next to the frameset. ONE record per arm (the argv sha256 differs): M1, M2a, M2b.
#
# WHY IT EXISTS. G-MAP-OVERFIT and G-BOX-OVERFIT bind the LAUNCH ARGV sha256 (launch_gate.py _overfit_binding_reasons:
# schema + `launch_argv_sha256`). The sha is over the WHOLE argv -- `--seed` and `--out` included -- so C0 and C1 need ONE
# RECORD EACH (their shas differ). The binding was NOT redesigned (MM decision): the records are produced under the rig argv
# `stack/ops/runs.d/r9e1_<ARM>.argv.json` of the shipped tree, through the SAME harness, templates and bars as refcv8's
# (`stack/scripts/refcv8_bind_{map,box}.args.tpl`, `closure_run.py --binding`); the argv sha is the only thing that changes.
#
# Mirrors the `bind` mode of r9w_prelaunch.sh (refcv9-warm, the precedent that worked on this Thor), with:
#   * the argv read from the SHIPPED TREE (the committed file, never a copy);
#   * ONE kind x ONE arm per call, so a long job is a detached call of its own (CLAUDE.md token rule 1);
#   * the GPU protocol of r9e1_run.sh: the Thor lock (blocking, <= 2 h) THEN an EMPTY compute-apps list;
#   * every verdict read from an ARTIFACT (closure json, record json, the assert-* markers), never from $?.
#
# Outputs (BD = $ROOT/ovf_<ARM>_<C12>):  BD/run_<kind>/{g_map_overfit.json | gbo_binding.json}, BD/<kind>_closure.json,
#   BD/<kind>.{out,render.txt,closure.txt,record.txt}, BD/<kind>.TIMING (start/end epoch, wall s).
# Markers:  ZZR9E1BIND-<KIND>-<ARM>-OKZZ  ZZR9E1BIND-REFUSEDZZ  ZZR9E1BIND-DRYOKZZ
set -u
DRY=0; if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
KIND="${1:-}"; ARM="${2:-}"; shift 2 || true
ROOT=/home/nvidia/r8ns/r9e1; CODE=""; COMMIT=""; INPUTS=""
LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock
_VENV_PY=/home/nvidia/venvs/tanitad-train/bin/python
PY="${PYTHON:-$( [ -x "$_VENV_PY" ] && echo "$_VENV_PY" || echo python3 )}"
export MSYS_NO_PATHCONV=1
while [ $# -gt 0 ]; do case "$1" in
  --commit) COMMIT="$2"; shift 2;; --root) ROOT="$2"; shift 2;; --code) CODE="$2"; shift 2;; --inputs) INPUTS="$2"; shift 2;;
  *) echo "REFUSED: unknown argument $1"; echo "ZZR9E1BIND-REFUSEDZZ"; exit 2;; esac; done
CODE="${CODE:-$ROOT/launch_tree}"
INPUTS="${INPUTS:-/home/nvidia/refcv7_bind/run_map/inputs}"      # the A19 MAIN spec + frameset (refcv8's measured location)
say() { echo "[r9e1-bind:$KIND:$ARM] $*"; }
die() { say "REFUSED: $*"; echo "ZZR9E1BIND-REFUSEDZZ"; exit 10; }
pyval() { "$PY" -c "$1" "${@:2}" 9>&- 200>&- | tr -d '\r'; }
case "$KIND" in map|box|map-banded|box-b3) ;; *) die "kind must be map|box|map-banded|box-b3";; esac
# E1-REC2: the arm is any r9e1_<ARM>.argv.json of the shipped tree (checked below) -- only the NAME is validated here (it builds a path)
case "$ARM" in ""|*[!A-Za-z0-9_]*) die "arm must be the NAME of a stack/ops/runs.d/r9e1_<ARM>.argv.json (letters, digits, _)";; esac
[ "${#COMMIT}" = 40 ] || die "--commit must be 40 chars"
C12="${COMMIT:0:12}"; BD="$ROOT/ovf_${ARM}_$C12"; HERE="$CODE/stack/scripts"
ARGV="$CODE/stack/ops/runs.d/r9e1_$ARM.argv.json"
[ -f "$ARGV" ] || die "no rig argv $ARGV in the shipped tree"
# E1-REC2: the KIND must match what THIS argv builds (the gate demands the lever's OWN record, bound to the whole argv sha256). The flag list is read
# by the GATE'S OWN loader + `flag_pairs` (no second parser); refused BEFORE any directory, lock or GPU is touched.
AFLAGS="$(pyval 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L; print(" ".join(sorted({f for f, _ in L.flag_pairs(L.load_argv_file(sys.argv[2]))})))' "$CODE" "$ARGV")"
[ -n "$AFLAGS" ] || die "could not read the flags of the launch argv $ARGV"
a_has() { case " $AFLAGS " in *" $1 "*) return 0;; esac; return 1; }
a_has_prefix() { case " $AFLAGS " in *" $1"*) return 0;; esac; return 1; }
# E1-REC3: the argv's --r9-box-* flags CLASSIFIED by the GATE'S OWN tuples (no second list): own-head levers ; training-only aux heads ; unknown
BCLS="$(pyval 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L
f = [x for x, _ in L.flag_pairs(L.load_argv_file(sys.argv[2])) if x.startswith("--r9-box-")]
print(" ".join(x for x in f if x in L.E1_BOX_OWN_FLAGS) + ";" + " ".join(x for x in f if x in L.E1_BOX_AUX_FLAGS) + ";" + " ".join(x for x in f if x not in L.E1_BOX_OWN_FLAGS + L.E1_BOX_AUX_FLAGS))' "$CODE" "$ARGV")"
case "$BCLS" in *";"*";"*) ;; *) die "could not classify the --r9-box-* flags of $ARGV (the gate's E1_BOX_OWN_FLAGS / E1_BOX_AUX_FLAGS)";; esac
BOWN="${BCLS%%;*}"; _r="${BCLS#*;}"; BAUX="${_r%%;*}"; BUNK="${_r#*;}"
# E1-REC4: the argv's temporal-BEV-warp flags, read from the GATE'S OWN tuple (no second list). Empty for every arm but T1 / T1s.
T1F="$(pyval 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L
print(" ".join(sorted({x for x, _ in L.flag_pairs(L.load_argv_file(sys.argv[2])) if x in L.E1_T1_FLAGS})))' "$CODE" "$ARGV")"
case "$KIND" in
  map)        a_has --r9-map-banded && die "kind map (the inherited /1 8-code head) on an argv carrying --r9-map-banded: use map-banded";;
  map-banded) a_has --r9-map-banded || die "kind map-banded on an argv WITHOUT --r9-map-banded: it builds the 8-code head, use map";;
  box)        [ -n "$BUNK" ] && die "kind box on an argv carrying an unknown box lever ($BUNK): classify it in launch_gate.py (E1_BOX_OWN_FLAGS or E1_BOX_AUX_FLAGS)"
              [ -n "$BOWN" ] && die "kind box (the inherited record) on an argv carrying an own-head --r9-box-* lever ($BOWN): use box-b3";;
  box-b3)     [ -n "$BUNK" ] && die "kind box-b3 on an argv carrying an unknown box lever ($BUNK): classify it in launch_gate.py (E1_BOX_OWN_FLAGS or E1_BOX_AUX_FLAGS)"
              [ -n "$BOWN" ] || die "kind box-b3 on an argv WITHOUT any --r9-box-* own-head lever (B3's four; an aux-only arm is kind box + the aux stamp): it builds the inherited box head, use box";;
esac
[ -n "$BAUX" ] && case "$KIND" in box|box-b3) say "training-only aux head(s) $BAUX: not built by the box harness; the record will stamp aux_heads_not_certified (read by the gate's /1 judge)";; esac
[ -n "$T1F" ] && say "temporal BEV warp flag(s) $T1F: not built by the overfit harness (frozen frames, no staged history); the record will stamp bev_warp_not_certified (read by the gate's judges)"
[ -f "$HERE/refcv8_assemble_argv.py" ] && [ -f "$HERE/closure_run.py" ] || die "no refcv8_assemble_argv.py / closure_run.py under $HERE"
# the rig's own template where one exists (box: the --candidate text and the data roots; map: refcv8's, unchanged)
TPL="$HERE/r9e1_bind_$KIND.args.tpl"; [ -f "$TPL" ] || TPL="$HERE/refcv8_bind_$KIND.args.tpl"; [ -f "$TPL" ] || die "no harness template $TPL"
mkdir -p "$BD" 2>/dev/null
if [ "$KIND" = map-banded ]; then   # the A18 spec the record binds (its sha256 is the gate's literal), staged next to the frameset
  A18SHA=5cb4f6fc4a8c15660237f541f077bfb334337dd287f674f8de02b7d414077c58
  [ -f "$INPUTS/gmo_spec_A18.json" ] || die "no $INPUTS/gmo_spec_A18.json (stage TanitAD Research Lab/.../2026-09-26-refcv7-map-hires/raw/gmo_spec_A18.json)"
  echo "$A18SHA  $INPUTS/gmo_spec_A18.json" | sha256sum -c - > /dev/null 2>&1 || die "gmo_spec_A18.json sha256 is not the registered $A18SHA"
fi
# the argv sha: the GATE'S OWN function on the GATE'S OWN loader (no second implementation)
ASHA="$(pyval 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L; print(L.argv_sha256(L.load_argv_file(sys.argv[2])))' "$CODE" "$ARGV")"
[ "${#ASHA}" = 64 ] || die "could not hash the launch argv $ARGV"
say "argv $ARGV sha256 $ASHA"
# the BOX harness READS its prereg + frame set from {OUT}/audit/raw (refcv8_prelaunch.sh bind): copies, md5 LITERALS (blob md5s)
if { [ "$KIND" = box ] || [ "$KIND" = box-b3 ]; } && [ "$DRY" != 1 ]; then
  BAUD="$BD/run_$KIND/audit/raw"; BSRC="$CODE/stack/ops/refcv8_box_audit/raw"
  mkdir -p "$BAUD/visibility" 2>/dev/null
  cp -p "$BSRC/PREREG_G_BOX_OVERFIT.md" "$BAUD/" && cp -p "$BSRC/visibility/gbo_frameset.json" "$BAUD/visibility/" || die "could not prepare the box audit inputs from $BSRC"
  echo "594c71196cc5bbd527fee40b2cb0e3f1  $BAUD/PREREG_G_BOX_OVERFIT.md" | md5sum -c - > /dev/null 2>&1 || die "box audit prereg md5"
  echo "b291404c36f83c3e397e3b90367e8e7b  $BAUD/visibility/gbo_frameset.json" | md5sum -c - > /dev/null 2>&1 || die "box audit frame-set md5"
fi
"$PY" "$HERE/refcv8_assemble_argv.py" --render-template "$TPL" --sub "CODE=$CODE" --sub "COMMIT=$COMMIT" --sub "ARGV=$ARGV" \
    --sub "ARGV_SHA=$ASHA" --sub "OUT=$BD/run_$KIND" --sub "INPUTS=$INPUTS" \
    --args-out "$BD/$KIND.args" --data-roots-out "$BD/$KIND.roots" > "$BD/$KIND.render.txt" 2>&1 9>&- 200>&-
[ "$(grep -c '^RENDERED' "$BD/$KIND.render.txt")" = 1 ] || die "template $TPL did not render ($BD/$KIND.render.txt)"
if [ "$KIND" = map ] && [ -n "$T1F" ]; then   # E1-REC4: the /1 map harness learns the launch argv (its T1 flags -> the stamp) only for a T1 argv
  [ -z "$(tail -c1 "$BD/$KIND.args")" ] || echo >> "$BD/$KIND.args"
  printf '%s\n' --launch-argv "$ARGV" >> "$BD/$KIND.args"
fi
if [ "$DRY" = 1 ]; then
  say "DRY: would hold $LOCKF (blocking <= 2 h), assert compute-apps EMPTY, then closure_run.py --binding --commit $C12 --tree $CODE -- $(wc -l < "$BD/$KIND.args") harness tokens (argv sha ${ASHA:0:16})"
  echo "ZZR9E1BIND-DRYOKZZ"; exit 0
fi
# ---- GPU protocol (r9e1_run.sh): the blocking lock, then an EMPTY GPU
exec 9>"$LOCKF"; say "waiting for the Thor GPU lock (blocking, <= 2 h)"; flock -w 7200 9 || die "the Thor GPU lock stayed held for 2 h"
NAPPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')"
[ "$NAPPS" = 0 ] || die "the GPU is not empty under the lock ($NAPPS compute app(s)): never share the GPU"
mapfile -t HARGS < "$BD/$KIND.args"; mapfile -t ROOTS < "$BD/$KIND.roots"
DR=(); for r in "${ROOTS[@]}"; do [ -n "$r" ] && DR+=(--data-root "$r"); done
mkdir -p "$BD/run_$KIND" 2>/dev/null
if [ "$KIND" = map ]; then REC="$BD/run_map/g_map_overfit.json"; elif [ "$KIND" = map-banded ]; then REC="$BD/run_map-banded/g_map_overfit.json"
elif [ "$KIND" = box-b3 ]; then REC="$BD/run_box-b3/gbo_binding.json"; else REC="$BD/run_box/gbo_binding.json"; fi
T0=$(date +%s); echo "$T0" > "$BD/$KIND.TIMING"; say "lock held, GPU empty; closure_run --binding starts at $(date -u +%T)"
( cd "$CODE" && export PYTHONPATH="$CODE/stack" OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 &&
  "$PY" "$CODE/stack/scripts/closure_run.py" --binding --commit "$COMMIT" --tree "$CODE" \
    --out "$BD/${KIND}_closure.json" --result "$REC" "${DR[@]}" -- "${HARGS[@]}" > "$BD/$KIND.out" 2>&1 9>&- 200>&- )
T1=$(date +%s); echo "$T0 $T1 $((T1 - T0))" > "$BD/$KIND.TIMING"; say "harness finished, wall $((T1 - T0)) s"
"$PY" "$HERE/refcv8_assemble_argv.py" --assert-closure "$BD/${KIND}_closure.json" > "$BD/$KIND.closure.txt" 2>&1 9>&- 200>&-
[ "$(grep -c '^ASSERT-OK' "$BD/$KIND.closure.txt")" = 1 ] || die "$KIND closure refused: $(tr -d '\r' < "$BD/$KIND.closure.txt" | head -3 | tr '\n' ' ')"
if [ "$KIND" = map-banded ]; then   # the GATE'S OWN judge on THIS arm's argv (no second implementation); ASSERT-OK only when it has no reason
  "$PY" -c 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L
a = L.load_argv_file(sys.argv[3]); r, d = L.judge_map_overfit_banded(L.PROFILES["refcv9-e1rig"], sys.argv[2], sys.argv[4], argv_sha=L.argv_sha256(a), argv=a)
print(chr(10).join("REFUSED: " + x for x in r) if r else "ASSERT-OK map-banded record (" + str(d.get("sha256"))[:16] + ")")' "$CODE" "$REC" "$ARGV" "$COMMIT" > "$BD/$KIND.record.txt" 2>&1 9>&- 200>&-
elif [ "$KIND" = box-b3 ]; then   # the GATE'S OWN judges on THIS arm's argv: the full G-BOX-OVERFIT re-judge AND the E1-BOX-OWN-RECORD closure
  "$PY" -c 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L
a = L.load_argv_file(sys.argv[3]); sha = L.argv_sha256(a); P = L.PROFILES["refcv9-e1rig"]
r, d = L.judge_box_overfit(P, sys.argv[2], sys.argv[4], argv_sha=sha, argv=a)
ok, why = L.own_record_closes(P, "E1-BOX-OWN-RECORD", a, {"box_overfit_record": sys.argv[2]}, sys.argv[4], sha)
r = list(r) + ([] if ok else list(why))
print(chr(10).join("REFUSED: " + x for x in r) if r else "ASSERT-OK box-b3 record (" + str(d.get("sha256"))[:16] + ")")' "$CODE" "$REC" "$ARGV" "$COMMIT" > "$BD/$KIND.record.txt" 2>&1 9>&- 200>&-
else
"$PY" "$HERE/refcv8_assemble_argv.py" --assert-record "$REC" --kind "$KIND" > "$BD/$KIND.record.txt" 2>&1 9>&- 200>&-
if [ "$KIND" = box ] && [ -n "$BAUX" ]; then   # E1-REC3: an aux-carrying argv: ALSO the GATE'S OWN /1 judge on THIS argv (it reads aux_heads_not_certified)
  "$PY" -c 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L
a = L.load_argv_file(sys.argv[3]); r, d = L.judge_box_overfit(L.PROFILES["refcv9-e1rig"], sys.argv[2], sys.argv[4], argv_sha=L.argv_sha256(a), argv=a)
print(chr(10).join("REFUSED: " + x for x in r) if r else "AUXJUDGE-OK box record with aux stamp " + str(d.get("aux_heads_not_certified")))' "$CODE" "$REC" "$ARGV" "$COMMIT" >> "$BD/$KIND.record.txt" 2>&1 9>&- 200>&-
  [ "$(grep -c '^AUXJUDGE-OK' "$BD/$KIND.record.txt")" = 1 ] || die "$KIND record refused by the gate's /1 judge (aux stamp): $(tr -d '\r' < "$BD/$KIND.record.txt" | grep '^REFUSED' | head -3 | tr '\n' ' ')"
fi
if [ -n "$T1F" ] && { [ "$KIND" = box ] || [ "$KIND" = map ]; }; then   # E1-REC4: a T1 argv: the GATE'S OWN stamp clause on THIS argv (box: the whole /1 judge, which calls it)
  "$PY" -c 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L
a = L.load_argv_file(sys.argv[3]); sha = L.argv_sha256(a); P = L.PROFILES["refcv9-e1rig"]
if sys.argv[5] == "box":
    r, d = L.judge_box_overfit(P, sys.argv[2], sys.argv[4], argv_sha=sha, argv=a); s = d.get("bev_warp_not_certified")
else:
    rec = L.read_json(sys.argv[2]); r, s = L.bev_warp_stamp_reasons("G-MAP-OVERFIT", rec, a)
    r = list(r) + ([] if rec.get("launch_argv_sha256") == sha else ["G-MAP-OVERFIT: the record is bound to argv " + str(rec.get("launch_argv_sha256"))[:12] + ", not " + sha[:12]])
print(chr(10).join("REFUSED: " + x for x in r) if r else "T1JUDGE-OK " + sys.argv[5] + " record with the T1 stamp " + str(s))' "$CODE" "$REC" "$ARGV" "$COMMIT" "$KIND" >> "$BD/$KIND.record.txt" 2>&1 9>&- 200>&-
  [ "$(grep -c '^T1JUDGE-OK' "$BD/$KIND.record.txt")" = 1 ] || die "$KIND record refused by the gate's judge (T1 stamp): $(tr -d '\r' < "$BD/$KIND.record.txt" | grep '^REFUSED' | head -3 | tr '\n' ' ')"
fi
fi
[ "$(grep -c '^ASSERT-OK' "$BD/$KIND.record.txt")" = 1 ] || die "$KIND record refused: $(tr -d '\r' < "$BD/$KIND.record.txt" | head -3 | tr '\n' ' ')"
say "$KIND binding run asserted: closure $BD/${KIND}_closure.json, record $REC (argv sha ${ASHA:0:16})"
echo "ZZR9E1BIND-${KIND}-${ARM}-OKZZ"
