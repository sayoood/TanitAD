#!/usr/bin/env bash
# r9e1_bind.sh -- THOR-SIDE producer of the two BINDING overfit records for the refcv9 E1 perception rig
# (launch-gate profile `refcv9-e1rig`; C12-E1 blocker 2).
#
#   r9e1_bind.sh [--dry-run] <map|box|map-banded|box-b3> <ARM> --commit SHA40 [--root DIR] [--code DIR] [--inputs DIR] [--reuse-from BASE_ARM]
#                [--argv-prefix r9e1|r9e2] [--reuse-argv-prefix r9e1|r9e2] [--reuse-root DIR]
#
#   --argv-prefix P (default r9e1 = the E1 perception rig, EXACTLY as before): the study this arm belongs to. ARM names `stack/ops/runs.d/<P>_<ARM>.argv.json` (r9e2 resolves the full-model
#   arm `r9e2_K0.argv.json`); the study -- root (--root's default), the gate profile the python assertions judge under (refcv9-e1rig / refcv9-e2) -- comes from
#   `r9e1_gate3/r9gate3_study.sh` next to this script, the SAME definition the gate3 chain reads. A prefix no study owns REFUSES before any directory, lock or GPU is touched.
#   The study is read from THIS flag only, never from R9_STUDY: a binding record is bound to a whole argv, and an ambient variable must not retarget it.
#   --reuse-argv-prefix / --reuse-root: where the BASE arm of --reuse-from lives when it belongs to ANOTHER study (default: this arm's own prefix and --root). K0 (r9e2) reuses C0's
#   map record: `--argv-prefix r9e2 map K0 --reuse-from C0 --reuse-argv-prefix r9e1 --reuse-root /home/nvidia/r8ns/r9e1` (the base record must exist at THIS --commit's c12).
#
#   --reuse-from BASE_ARM (E1-REUSE, MM 207ebd, PINNED; kinds map and box only): take a REUSE binding instead of a 2,000 s Thor run. NO lock, NO GPU. The harness's OWN
#   `--identity-only` runs on THIS arm's rendered invocation AND on the base's saved one ($ROOT/ovf_<BASE>_<C12>/<kind>.args: what the base ACTUALLY ran); the two
#   identity_sha256s must be EQUAL (effective build argv after the harness's own strip, data roots, every other harness argument -- only the launch-argv references and
#   the per-arm stamps may differ), the base record + closure must exist at the SAME C12 and pass the GATE'S OWN judge for the base argv, and the binding
#   BD/run_<kind>/reuse_binding.json (schema tanitad.overfit_reuse/1: base arm, sha256 of the base record + closure, both identity shas, THIS argv sha256, THIS arm's
#   stamps, commit) is written and re-judged by `launch_gate.write_reuse_binding` (the gate's `judge_overfit_reuse`). The base closure is copied to BD/<kind>_closure.json.
#   Markers: ZZR9E1BIND-REUSE-<kind>-<ARM>-OKZZ  ZZR9E1BIND-DRYREUSE-OKZZ (--dry-run: identity only)  ZZR9E1BIND-REFUSEDZZ (every refusal names its reason).
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
AP=r9e1; if [ "${1:-}" = "--argv-prefix" ]; then AP="${2:-}"; shift 2 || true; fi   # --argv-prefix is also accepted BEFORE <kind> (every other flag follows <ARM>)
KIND="${1:-}"; ARM="${2:-}"; shift 2 || true
ROOT=""; CODE=""; COMMIT=""; INPUTS=""; REUSE=""; RAP=""; RROOT=""   # ROOT "" = the study's root (resolved below; r9e1 for the default prefix)
LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock
_VENV_PY=/home/nvidia/venvs/tanitad-train/bin/python
PY="${PYTHON:-$( [ -x "$_VENV_PY" ] && echo "$_VENV_PY" || echo python3 )}"
export MSYS_NO_PATHCONV=1
while [ $# -gt 0 ]; do case "$1" in
  --commit) COMMIT="$2"; shift 2;; --root) ROOT="$2"; shift 2;; --code) CODE="$2"; shift 2;; --inputs) INPUTS="$2"; shift 2;; --reuse-from) REUSE="$2"; shift 2;;
  --argv-prefix) AP="$2"; shift 2;; --reuse-argv-prefix) RAP="$2"; shift 2;; --reuse-root) RROOT="$2"; shift 2;;
  *) echo "REFUSED: unknown argument $1"; echo "ZZR9E1BIND-REFUSEDZZ"; exit 2;; esac; done
say() { echo "[r9e1-bind:$KIND:$ARM] $*"; }
die() { say "REFUSED: $*"; echo "ZZR9E1BIND-REFUSEDZZ"; exit 10; }
pyval() { "$PY" -c "$1" "${@:2}" 9>&- 200>&- | tr -d '\r'; }
# the study of --argv-prefix: ONE definition (r9gate3_study.sh, next to this script), read from the flag and never from R9_STUDY
R9_STUDY_NOAUTOLOAD=1
. "$(dirname "${BASH_SOURCE[0]}")/r9e1_gate3/r9gate3_study.sh" || die "no r9e1_gate3/r9gate3_study.sh next to this script (the study definition)"
STUDY="$(r9_study_of_prefix "$AP")" || die "--argv-prefix '$AP' is not the argv prefix of any study (r9gate3_study.sh: $R9_STUDIES -> r9e1 / r9e2)"
r9_study_load "$STUDY"; PREFIX="$R9_PREFIX"; export R9_PROFILE   # R9_PROFILE: the gate profile the python assertions below judge under (they default to refcv9-e1rig when it is unset)
ROOT="${ROOT:-$R9_ROOT}"; RAP="${RAP:-$PREFIX}"
r9_study_of_prefix "$RAP" > /dev/null || die "--reuse-argv-prefix '$RAP' is not the argv prefix of any study"
RROOT="${RROOT:-$ROOT}"
CODE="${CODE:-$ROOT/launch_tree}"
INPUTS="${INPUTS:-/home/nvidia/refcv7_bind/run_map/inputs}"      # the A19 MAIN spec + frameset (refcv8's measured location)
case "$KIND" in map|box|map-banded|box-b3) ;; *) die "kind must be map|box|map-banded|box-b3";; esac
# E1-REC2: the arm is any r9e1_<ARM>.argv.json of the shipped tree (checked below) -- only the NAME is validated here (it builds a path)
case "$ARM" in ""|*[!A-Za-z0-9_]*) die "arm must be the NAME of a stack/ops/runs.d/r9e1_<ARM>.argv.json (letters, digits, _)";; esac
[ "${#COMMIT}" = 40 ] || die "--commit must be 40 chars"
C12="${COMMIT:0:12}"; BD="$ROOT/ovf_${ARM}_$C12"; HERE="$CODE/stack/scripts"
ARGV="$CODE/stack/ops/runs.d/${PREFIX}_$ARM.argv.json"
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
if { [ "$KIND" = box ] || [ "$KIND" = box-b3 ]; } && { [ "$DRY" != 1 ] || [ -n "$REUSE" ]; }; then
  BAUD="$BD/run_$KIND/audit/raw"; BSRC="$CODE/stack/ops/refcv8_box_audit/raw"
  mkdir -p "$BAUD/visibility" 2>/dev/null
  cp -p "$BSRC/PREREG_G_BOX_OVERFIT.md" "$BAUD/" && cp -p "$BSRC/visibility/gbo_frameset.json" "$BAUD/visibility/" || die "could not prepare the box audit inputs from $BSRC"
  echo "594c71196cc5bbd527fee40b2cb0e3f1  $BAUD/PREREG_G_BOX_OVERFIT.md" | md5sum -c - > /dev/null 2>&1 || die "box audit prereg md5"
  echo "b291404c36f83c3e397e3b90367e8e7b  $BAUD/visibility/gbo_frameset.json" | md5sum -c - > /dev/null 2>&1 || die "box audit frame-set md5"
fi
# E1-NEAR20: the banded record's near band IS the argv's near-band end (--r9-map-banded-near-m; absent = 10), read by the GATE'S OWN
# banded_near_m_of (no second parser) into the template's {NEAR_LIFT}. MM 2026-11-07: the template's literal 10 would have built M1N20's
# record at 10 m and bound it to its 20 m argv (the gate refuses that record, after ~74 GPU-min). The other kinds' templates have no {NEAR_LIFT}.
NL=""
if [ "$KIND" = map-banded ]; then
  NL="$(pyval 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L; print("%g" % L.banded_near_m_of(L.load_argv_file(sys.argv[2])))' "$CODE" "$ARGV")"
  case "$NL" in 10|20) say "banded near band 0-$NL m (the argv's near-band end = the record's --near-lift-m)";; *) die "could not read the banded near-band end of $ARGV (got '$NL')";; esac
fi
"$PY" "$HERE/refcv8_assemble_argv.py" --render-template "$TPL" --sub "CODE=$CODE" --sub "COMMIT=$COMMIT" --sub "ARGV=$ARGV" \
    --sub "ARGV_SHA=$ASHA" --sub "OUT=$BD/run_$KIND" --sub "INPUTS=$INPUTS" --sub "NEAR_LIFT=$NL" \
    --args-out "$BD/$KIND.args" --data-roots-out "$BD/$KIND.roots" > "$BD/$KIND.render.txt" 2>&1 9>&- 200>&-
[ "$(grep -c '^RENDERED' "$BD/$KIND.render.txt")" = 1 ] || die "template $TPL did not render ($BD/$KIND.render.txt)"
if [ "$KIND" = map ] && [ -n "$T1F" ]; then   # E1-REC4: the /1 map harness learns the launch argv (its T1 flags -> the stamp) only for a T1 argv
  [ -z "$(tail -c1 "$BD/$KIND.args")" ] || echo >> "$BD/$KIND.args"
  printf '%s\n' --launch-argv "$ARGV" >> "$BD/$KIND.args"
fi
# ---- E1-REUSE: bind THIS arm to a BASE arm's PASSING record when the harness invocation is IDENTICAL (computed by the harness's own --identity-only). No lock, no GPU.
if [ -n "$REUSE" ]; then
  case "$REUSE" in ""|*[!A-Za-z0-9_]*) die "--reuse-from must be the NAME of a base arm (letters, digits, _)";; esac
  { [ "$REUSE" != "$ARM" ] || [ "$RAP" != "$PREFIX" ]; } || die "--reuse-from names the arm itself"
  { [ "$KIND" = map ] || [ "$KIND" = box ]; } || die "--reuse-from covers the kinds map and box only: a banded / own-head argv ($KIND) needs its OWN record"
  BASE_ARGV="$CODE/stack/ops/runs.d/${RAP}_$REUSE.argv.json"; [ -f "$BASE_ARGV" ] || die "no base rig argv $BASE_ARGV in the shipped tree"
  BBD="$RROOT/ovf_${REUSE}_$C12"; CHECK=G-BOX-OVERFIT; BREC="$BBD/run_box/gbo_binding.json"
  if [ "$KIND" = map ]; then CHECK=G-MAP-OVERFIT; BREC="$BBD/run_map/g_map_overfit.json"; fi
  BCLO="$BBD/${KIND}_closure.json"
  BASHA="$(pyval 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L; print(L.argv_sha256(L.load_argv_file(sys.argv[2])))' "$CODE" "$BASE_ARGV")"
  [ "${#BASHA}" = 64 ] || die "could not hash the base argv $BASE_ARGV"
  idrun() {   # idrun <args-file> <out-json>: the harness's own --identity-only on EXACTLY those tokens (CPU, exits before any model / data / GPU)
    local TK=(); mapfile -t TK < "$1"
    ( cd "$CODE" && export PYTHONPATH="$CODE/stack:$CODE/stack/scripts" CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 &&
      "$PY" "$CODE/${TK[0]}" "${TK[@]:1}" --identity-only "$2" ) > "$2.log" 2>&1 9>&- 200>&-
    [ -s "$2" ] || die "the harness's --identity-only wrote nothing for $1 ($2.log: $(tail -2 "$2.log" | tr -d '\r' | tr '\n' ' '))"
  }
  if [ -f "$BBD/$KIND.args" ]; then BARGS="$BBD/$KIND.args"; BROOTS="$BBD/$KIND.roots"; BOUT="$BBD/run_$KIND"     # what the base ACTUALLY ran
  elif [ "$DRY" = 1 ]; then   # a dry plan on a tree with no base record dir: re-render the base's invocation with the SAME template (its OUT under THIS BD)
    BOUT="$BD/base_run_$KIND"; BARGS="$BD/$KIND.base.args"; BROOTS="$BD/$KIND.base.roots"
    if [ "$KIND" = box ]; then mkdir -p "$BOUT/audit/raw/visibility" && cp -p "$CODE/stack/ops/refcv8_box_audit/raw/PREREG_G_BOX_OVERFIT.md" "$BOUT/audit/raw/" &&
      cp -p "$CODE/stack/ops/refcv8_box_audit/raw/visibility/gbo_frameset.json" "$BOUT/audit/raw/visibility/" || die "could not prepare the base's box audit inputs"; fi
    "$PY" "$HERE/refcv8_assemble_argv.py" --render-template "$TPL" --sub "CODE=$CODE" --sub "COMMIT=$COMMIT" --sub "ARGV=$BASE_ARGV" --sub "ARGV_SHA=$BASHA" \
        --sub "OUT=$BOUT" --sub "INPUTS=$INPUTS" --args-out "$BARGS" --data-roots-out "$BROOTS" > "$BD/$KIND.base.render.txt" 2>&1 9>&- 200>&-
    [ "$(grep -c '^RENDERED' "$BD/$KIND.base.render.txt")" = 1 ] || die "template $TPL did not render for the base ($BD/$KIND.base.render.txt)"
    BT1F="$(pyval 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L
print(" ".join(sorted({x for x, _ in L.flag_pairs(L.load_argv_file(sys.argv[2])) if x in L.E1_T1_FLAGS})))' "$CODE" "$BASE_ARGV")"
    if [ "$KIND" = map ] && [ -n "$BT1F" ]; then [ -z "$(tail -c1 "$BARGS")" ] || echo >> "$BARGS"; printf '%s\n' --launch-argv "$BASE_ARGV" >> "$BARGS"; fi
  else die "no base record directory $BBD (the base arm's $KIND record must exist at THIS commit: the SAME C12)"; fi
  [ -f "$BARGS" ] || die "no base harness invocation $BARGS"
  idrun "$BD/$KIND.args" "$BD/$KIND.identity.json"; idrun "$BARGS" "$BD/$KIND.base.identity.json"
  EQ="$(pyval 'import json, sys
a = json.load(open(sys.argv[2])); b = json.load(open(sys.argv[3]))
K = ("schema", "tool", "effective_build_argv", "build_facts", "data_roots", "other_args", "audit_inputs"); d = [k for k in K if a.get(k) != b.get(k)]
print("EQUAL " + a["identity_sha256"] if a["identity_sha256"] == b["identity_sha256"] and not d else "DIFFER " + ",".join(d))' "$CODE" "$BD/$KIND.identity.json" "$BD/$KIND.base.identity.json")"
  case "$EQ" in "EQUAL "*) say "harness invocation IDENTICAL to $REUSE's (identity ${EQ#EQUAL })";;
    *) die "THIS argv's $KIND harness invocation is NOT identical to $REUSE's ($EQ): the base record does not describe it -- run its OWN record";; esac
  sed "s|$BD/run_$KIND|<OUT>|" "$BD/$KIND.roots" > "$BD/$KIND.roots.norm"; sed "s|$BOUT|<OUT>|" "$BROOTS" > "$BD/$KIND.base.roots.norm" 2>/dev/null
  cmp -s "$BD/$KIND.roots.norm" "$BD/$KIND.base.roots.norm" || die "the closure data roots differ from the base's ($BD/$KIND.roots.norm vs $BD/$KIND.base.roots.norm)"
  if [ "$DRY" = 1 ]; then say "DRY REUSE: $KIND of $ARM would bind to $REUSE's record (identity ${EQ#EQUAL }); no record / closure read"; echo "ZZR9E1BIND-DRYREUSE-OKZZ"; exit 0; fi
  [ -s "$BREC" ] && [ -s "$BCLO" ] || die "the base arm $REUSE has no PASSING $KIND record + closure at this commit ($BREC, $BCLO): run it first"
  mkdir -p "$BD/run_$KIND" 2>/dev/null; cp -p "$BCLO" "$BD/${KIND}_closure.json" || die "could not copy the base closure"
  "$PY" -c 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L
r, d = L.write_reuse_binding(__import__("os").environ.get("R9_PROFILE") or "refcv9-e1rig", sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[1], closure_option=sys.argv[13])
print(chr(10).join("REFUSED: " + x for x in r) if r else "ASSERT-OK reuse binding (" + str(d.get("sha256"))[:16] + ")")' "$CODE" "$CHECK" "$ARM" "$REUSE" "$ARGV" "$BASE_ARGV" "$BREC" "$BCLO" \
      "$BD/$KIND.identity.json" "$BD/$KIND.base.identity.json" "$COMMIT" "$BD/run_$KIND/reuse_binding.json" "$BD/${KIND}_closure.json" > "$BD/$KIND.reuse.txt" 2>&1 9>&- 200>&-
  [ "$(grep -c '^ASSERT-OK' "$BD/$KIND.reuse.txt")" = 1 ] || die "reuse binding refused by the gate's judge: $(tr -d '\r' < "$BD/$KIND.reuse.txt" | grep '^REFUSED' | head -3 | tr '\n' ' ')"
  say "reuse binding written and re-judged: $BD/run_$KIND/reuse_binding.json (this argv sha ${ASHA:0:16}, base $REUSE)"
  echo "ZZR9E1BIND-REUSE-${KIND}-${ARM}-OKZZ"; exit 0
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
a = L.load_argv_file(sys.argv[3]); r, d = L.judge_map_overfit_banded(L.PROFILES[__import__("os").environ.get("R9_PROFILE") or "refcv9-e1rig"], sys.argv[2], sys.argv[4], argv_sha=L.argv_sha256(a), argv=a)
print(chr(10).join("REFUSED: " + x for x in r) if r else "ASSERT-OK map-banded record (" + str(d.get("sha256"))[:16] + ")")' "$CODE" "$REC" "$ARGV" "$COMMIT" > "$BD/$KIND.record.txt" 2>&1 9>&- 200>&-
elif [ "$KIND" = box-b3 ]; then   # the GATE'S OWN judges on THIS arm's argv: the full G-BOX-OVERFIT re-judge AND the E1-BOX-OWN-RECORD closure
  "$PY" -c 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L
a = L.load_argv_file(sys.argv[3]); sha = L.argv_sha256(a); P = L.PROFILES[__import__("os").environ.get("R9_PROFILE") or "refcv9-e1rig"]
r, d = L.judge_box_overfit(P, sys.argv[2], sys.argv[4], argv_sha=sha, argv=a)
ok, why = L.own_record_closes(P, "E1-BOX-OWN-RECORD", a, {"box_overfit_record": sys.argv[2]}, sys.argv[4], sha)
r = list(r) + ([] if ok else list(why))
print(chr(10).join("REFUSED: " + x for x in r) if r else "ASSERT-OK box-b3 record (" + str(d.get("sha256"))[:16] + ")")' "$CODE" "$REC" "$ARGV" "$COMMIT" > "$BD/$KIND.record.txt" 2>&1 9>&- 200>&-
else
"$PY" "$HERE/refcv8_assemble_argv.py" --assert-record "$REC" --kind "$KIND" > "$BD/$KIND.record.txt" 2>&1 9>&- 200>&-
if [ "$KIND" = box ] && [ -n "$BAUX" ]; then   # E1-REC3: an aux-carrying argv: ALSO the GATE'S OWN /1 judge on THIS argv (it reads aux_heads_not_certified)
  "$PY" -c 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L
a = L.load_argv_file(sys.argv[3]); r, d = L.judge_box_overfit(L.PROFILES[__import__("os").environ.get("R9_PROFILE") or "refcv9-e1rig"], sys.argv[2], sys.argv[4], argv_sha=L.argv_sha256(a), argv=a)
print(chr(10).join("REFUSED: " + x for x in r) if r else "AUXJUDGE-OK box record with aux stamp " + str(d.get("aux_heads_not_certified")))' "$CODE" "$REC" "$ARGV" "$COMMIT" >> "$BD/$KIND.record.txt" 2>&1 9>&- 200>&-
  [ "$(grep -c '^AUXJUDGE-OK' "$BD/$KIND.record.txt")" = 1 ] || die "$KIND record refused by the gate's /1 judge (aux stamp): $(tr -d '\r' < "$BD/$KIND.record.txt" | grep '^REFUSED' | head -3 | tr '\n' ' ')"
fi
if [ -n "$T1F" ] && { [ "$KIND" = box ] || [ "$KIND" = map ]; }; then   # E1-REC4: a T1 argv: the GATE'S OWN stamp clause on THIS argv (box: the whole /1 judge, which calls it)
  "$PY" -c 'import sys; sys.path.insert(0, sys.argv[1] + "/stack/scripts"); import launch_gate as L
a = L.load_argv_file(sys.argv[3]); sha = L.argv_sha256(a); P = L.PROFILES[__import__("os").environ.get("R9_PROFILE") or "refcv9-e1rig"]
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
