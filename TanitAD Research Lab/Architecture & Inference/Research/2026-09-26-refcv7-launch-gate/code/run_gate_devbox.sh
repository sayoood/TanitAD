#!/usr/bin/env bash
# refcv7 launch gate -- the DEV-BOX stage (SPEC_REFCV7 sec. 4): G-HYG, G-DVB, G-EVAL, G-SUITE
# [+ G-MAP-OVERFIT], CPU only, behind the brief's 8 GB RAM floor. Its evidence/ dir is then
# copied to Thor and imported by run_gate_thor.sh (only evidence bound to the same commit, tree,
# argv and input fingerprints is ever used).
#
#   COMMIT=<40-char sha of the launch commit>  ARGV=<refcv7 launch argv .json>
#   GATE_DIR=<fresh dir, short path>  [TAU_RECORD=<json>] [OVERFIT_RECORD=<json>]
#   ./run_gate_devbox.sh
#
# ⛔ MSYS rewrites POSIX-looking args for a native python (`/home/nvidia/...=D:/...` became
#    `C:\Program Files\Git\home\...` on 2026-09-26): MSYS_NO_PATHCONV=1 is not optional.
set -u
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' PYTHONIOENCODING=utf-8
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
GITDIR=C:/Users/Admin/tanitad-push/.git
: "${COMMIT:?}" "${ARGV:?}" "${GATE_DIR:?}"
TREE="C:/lgt/${COMMIT:0:10}"          # a CLEAN tree of the commit (git archive, never a worktree)
if [ ! -d "$TREE" ]; then
  mkdir -p "$TREE"
  git --git-dir="$GITDIR" -c core.autocrlf=false archive "$COMMIT" stack taniteval tools \
    > "$TREE.tar" && tar -xf "$TREE.tar" -C "$TREE" && rm -f "$TREE.tar"
fi
IN=C:/Users/Admin/lg0926/inputs       # the kit path map, the clock reference, the loader override
PM=()
while IFS= read -r l; do [ -n "$l" ] && PM+=(--path-map "$l"); done < "$IN/pathmap.txt"
EXTRA=()
if [ -n "${TAU_RECORD:-}" ]; then EXTRA+=(--nav-tau-record "$TAU_RECORD"); fi
if [ -n "${OVERFIT_RECORD:-}" ]; then EXTRA+=(--map-overfit-record "$OVERFIT_RECORD"); fi
"$PY" "$TREE/stack/scripts/launch_gate_refcv7.py" run --stage devbox --tree "$TREE" \
  --commit "$COMMIT" --argv-file "$ARGV" --out-dir "$GATE_DIR" "${PM[@]}" \
  --git-dir "$GITDIR" --baseline "${BASELINE:-agent/arch-inf-20260803}" --suite-work C:/lgs \
  --eval-loader "${EVAL_LOADER:-C:/Users/Admin/ev6_battery/code/refcv6_loader.py}" \
  --eval-kit D:/refcv6_eval_kit --eval-stamps D:/refcv6_eval_kit/ckpt_final/config.json \
  --eval-remap-overrides "$IN/eval_remap_overrides.json" \
  --clock-reference "$IN/q4c_grid_vs_egolog_ALLTRAIN.json" \
  --cpu-only --omp 4 --min-free-gb 8 --ram-wait-s 21600 ${EXTRA[@]+"${EXTRA[@]}"} \
  >> "$GATE_DIR.log" 2>&1
echo "devbox stage rc=$? (not the verdict: read $GATE_DIR/evidence/*.json)" >> "$GATE_DIR.log"
