#!/usr/bin/env bash
# optb_inputs.sh: put the Option B INPUT files on box 2 under D:\tanitad\optb\inputs and prove them by sha256 (MM ruling 5).
#   ladder_R35_s{0,1}.pt   the R35 start checkpoints (45,266,310 B each), from $PULLS/ladder_R35_s{k}/ckpt_last.pt on the dev box
#   readout_K10_R35_s{0,1}.pt  the two frozen response probes, from the package (frozen/inputs/)
# Each LOCAL file must hash to the constant pinned in optb_check.py (INIT_SHA / PROBE_SHA) BEFORE the put; each put is re-hashed on box 2 (certutil) and must equal it. Any mismatch refuses (rc 16 / 12). Nothing is deleted. Idempotent: a file already on box 2 with the right sha256 is not sent again.
source "$(dirname "${BASH_SOURCE[0]}")/optb_common.sh"; OPTB_STAGE="inputs"
need_commit; need_pkg optb_check.py frozen/inputs/readout_K10_R35_s0.pt frozen/inputs/readout_K10_R35_s1.pt
need_free_gb 1
INP="$B2_ROOT/inputs"; b2_mkdir "$INP"
HEX64='^[0-9a-f]{64}$'
for k in 0 1; do
  for kind in ladder probe; do
    if [ "$kind" = ladder ]; then SRC="$PULLS/ladder_R35_s$k/ckpt_last.pt"; DST="$INP/ladder_R35_s$k.pt"; WANT="$("$PY" -I -c "import sys, os, json; sys.path.insert(0, sys.argv[1]); import optb_check as C; f = os.environ.get('OPTB_FAKE_INIT_SHA_FILE') if os.environ.get('OPTB_DRY') == '1' else None; print(json.load(open(f))['$k'] if f else C.INIT_SHA[$k])" "$PKG")"
    else SRC="$PKG/frozen/inputs/readout_K10_R35_s$k.pt"; DST="$INP/readout_K10_R35_s$k.pt"; WANT="$("$PY" -I -c "import sys; sys.path.insert(0, sys.argv[1]); import optb_check as C; print(C.PROBE_SHA[$k])" "$PKG")"; fi
    [[ "$WANT" =~ $HEX64 ]] || die 16 "the checker holds no sha256 constant for $kind s$k"
    [ -f "$SRC" ] || die 16 "$SRC is missing on the dev box"
    GOT="$(sha256sum "$SRC" | cut -d' ' -f1)"
    [ "$GOT" = "$WANT" ] || die 16 "$SRC hashes to $GOT, the checker pins $WANT: refused, nothing sent"
    if b2_exists "$DST" && [ "$(b2_sha256 "$DST")" = "$WANT" ]; then say "$kind s$k already on box 2 with the pinned sha256"; continue; fi
    put_verified "$SRC" "$DST"
    [ "$(b2_sha256 "$DST")" = "$WANT" ] || die 12 "$DST on box 2 does not hash to the pinned sha256 after the put"
    say "$kind s$k on box 2, sha256 $WANT"
  done
done
say "ok"
