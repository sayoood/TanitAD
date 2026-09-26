#!/usr/bin/env bash
# Fetch the MINIMAL nuScenes file set — ONLY after a human has accepted the Terms of Use.
#
# ⛔ NO AGENT MAY RUN THIS. Registration and acceptance of https://www.nuscenes.org/terms-of-use
#    are human-only acts (stack/tanitad/data/nuscenes.py docstring). Reachability of the official
#    AWS Open Data bucket `motional-nuscenes` is NOT permission: the Terms are the operative
#    instrument. This script therefore REFUSES to run without the acknowledgement flag, and it
#    records who acknowledged, when, in a receipt beside the archives.
#
# Usage (the PI, or an operator the PI names):
#   bash fetch_nuscenes_after_tou.sh --i-accepted-the-nuscenes-terms-of-use "<name>" [tier]
#     tier = pilot     (4.57 GB)  v1.0-mini + map expansion v1.3      — smoke + SAM3 pilot
#          | planning  (45.36 GB) trainval metadata + 10 keyframe parts — the full-val eval
#          | maps      (0.40 GB)  map expansion v1.3 only             — SAM3 paint reference
#          | sweeps    (177.3 GB) 10 camera-sweep parts               — ONLY if the scored arm
#                                                                        needs 10 Hz history
#   (sizes are MEASURED from the bucket LIST/HEAD on 2026-09-19; see raw/bucket_listing.md)
#
# Everything lands PACKED in $ROOT/archives (D: is exFAT with 1 MiB clusters — 410k extracted
# keyframes would allocate ~430 GB for 45 GB of data). Extract selectively afterwards with
#   python -m adapters.nuscenes_planning extract-list --nuscenes-root $ROOT/data --split val \
#          --channels CAM_FRONT --out $ROOT/val_cam_front.txt
#   tar -xzf $ROOT/archives/v1.0-trainval01_keyframes.tgz -C $ROOT/data -T $ROOT/val_cam_front.txt
set -euo pipefail

ROOT="${TANITAD_NUSCENES_HOME:-D:/Archive/devbox-C/nuscenes}"
BASE="https://motional-nuscenes.s3.amazonaws.com"
CURL=(curl -sS --fail --ssl-no-revoke -L -C -)          # --ssl-no-revoke: dev-box TLS proxy
# ⭐ CONTENT verification against each S3 object's OWN ETag (multipart-aware), added 2026-09-26.
# NOT md5.checksum: MEASURED, the bucket's md5.checksum does not match the served trainval_meta object
# (537d3954... received vs 3eee6988... listed) while the recomputed multipart ETag matched the live one.
REPO="${TANITAD_REPO:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"
VERIFY="$REPO/tools/verify_s3_etag.py"
PY="${TANITAD_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
[ -x "$PY" ] || PY=python

if [ "${1:-}" != "--i-accepted-the-nuscenes-terms-of-use" ] || [ -z "${2:-}" ]; then
  echo "REFUSED: this script only runs for a human who has accepted the nuScenes Terms of Use." >&2
  echo "  bash $0 --i-accepted-the-nuscenes-terms-of-use \"<your name>\" [meta|pilot|planning|maps|sweeps]" >&2
  exit 2
fi
WHO="$2"; TIER="${3:-planning}"
# refuse BEFORE any byte is fetched if the verifier is unreachable -- a download that cannot be
# content-checked must not be allowed to look checked
[ -f "$VERIFY" ] || { echo "REFUSED: content verifier missing at $VERIFY" >&2; exit 2; }

# key<TAB>expected bytes (MEASURED 2026-09-19 by bucket LIST, cross-checked by HEAD)
PILOT="public/v1.0/v1.0-mini.tgz	4168148189
public/v1.0/nuScenes-map-expansion-v1.3.zip	398535531"
MAPS="public/v1.0/nuScenes-map-expansion-v1.3.zip	398535531"
# META: the 0.46 GB metadata ALONE. Added 2026-09-26 so the harness can be validated end to end
# (GT-collision floor, STOP + CV floors, sample counts 6,019/5,119/4,819, the VAD category audit)
# BEFORE the 44.9 GB keyframe pull -- "planning" bundles both and offered no cheaper first step.
META="public/v1.0/v1.0-trainval_meta.tgz	461678030"

PLANNING="public/v1.0/v1.0-trainval_meta.tgz	461678030
public/v1.0/v1.0-trainval01_keyframes.tgz	4529954279
public/v1.0/v1.0-trainval02_keyframes.tgz	4272745444
public/v1.0/v1.0-trainval03_keyframes.tgz	4249901872
public/v1.0/v1.0-trainval04_keyframes.tgz	4562843980
public/v1.0/v1.0-trainval05_keyframes.tgz	3957488692
public/v1.0/v1.0-trainval06_keyframes.tgz	3856821591
public/v1.0/v1.0-trainval07_keyframes.tgz	4177675764
public/v1.0/v1.0-trainval08_keyframes.tgz	4282633755
public/v1.0/v1.0-trainval09_keyframes.tgz	4814177310
public/v1.0/v1.0-trainval10_keyframes.tgz	6198448085"
SWEEPS="public/v1.0/v1.0-trainval01_blobs_camera.tgz	17590765342
public/v1.0/v1.0-trainval02_blobs_camera.tgz	16512734591
public/v1.0/v1.0-trainval03_blobs_camera.tgz	16599534206
public/v1.0/v1.0-trainval04_blobs_camera.tgz	17820165067
public/v1.0/v1.0-trainval05_blobs_camera.tgz	14852318156
public/v1.0/v1.0-trainval06_blobs_camera.tgz	14477516576
public/v1.0/v1.0-trainval07_blobs_camera.tgz	16065659931
public/v1.0/v1.0-trainval08_blobs_camera.tgz	16743348010
public/v1.0/v1.0-trainval09_blobs_camera.tgz	19302674331
public/v1.0/v1.0-trainval10_blobs_camera.tgz	27323394106"

case "$TIER" in
  pilot)    LIST="$PILOT" ;;
  maps)     LIST="$MAPS" ;;
  meta)     LIST="$META" ;;
  planning) LIST="$PLANNING" ;;
  sweeps)   LIST="$SWEEPS" ;;
  *) echo "unknown tier $TIER" >&2; exit 2 ;;
esac

mkdir -p "$ROOT/archives" "$ROOT/data"
RECEIPT="$ROOT/archives/RECEIPT_${TIER}.json"
printf '{"tier": "%s", "accepted_terms_by": "%s", "utc": "%s", "source": "%s", "files": [\n' \
       "$TIER" "$WHO" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$BASE" > "$RECEIPT"

first=1
while IFS=$'\t' read -r KEY WANT; do
  [ -z "$KEY" ] && continue
  OUT="$ROOT/archives/$(basename "$KEY")"
  echo "== $KEY -> $OUT (expect $WANT bytes)"
  "${CURL[@]}" -o "$OUT" "$BASE/$KEY"
  GOT=$(wc -c < "$OUT" | tr -d ' ')
  if [ "$GOT" != "$WANT" ]; then
    echo "SIZE MISMATCH for $KEY: got $GOT, listing says $WANT — REFUSING to mark it verified" >&2
    exit 1
  fi
  MD5=$(md5sum "$OUT" | cut -d' ' -f1)
  ETAG=$(curl -sSI --ssl-no-revoke "$BASE/$KEY" | tr -d '\r' | awk 'tolower($1)=="etag:"{print $2}') || ETAG=""
  if [ -z "$ETAG" ]; then
    ETAG='""'; VRC=3; VERDICT="INCONCLUSIVE: no ETag from HEAD"
  else
    VRC=0; VERDICT=$("$PY" "$VERIFY" "$OUT" "$ETAG") || VRC=$?
  fi
  echo "   etag $ETAG -> $VERDICT"
  if [ "$VRC" -eq 1 ]; then
    echo "CONTENT MISMATCH for $KEY against its own S3 ETag — REFUSING to mark it verified" >&2
    exit 1
  fi
  [ $first -eq 1 ] || printf ',\n' >> "$RECEIPT"
  printf '  {"key": "%s", "bytes": %s, "md5": "%s", "s3_etag": %s, "etag_verdict": "%s"}' \
         "$KEY" "$GOT" "$MD5" "$ETAG" "${VERDICT%%:*}" >> "$RECEIPT"
  first=0
done <<< "$LIST"
printf '\n],\n' >> "$RECEIPT"

# the bucket ships its own md5 list. ⚠️ KEPT FOR REFERENCE ONLY -- it is NOT the verification: MEASURED
# 2026-09-26 it does not describe the served trainval_meta object. The per-file s3_etag verdicts above are.
"${CURL[@]}" -o "$ROOT/archives/md5.checksum" "$BASE/public/v1.0/md5.checksum" || true
printf ' "md5_checksum_file": "archives/md5.checksum",\n' >> "$RECEIPT"
printf ' "licence": "CC BY-NC-SA 4.0 (research-only, share-alike; derivatives inherit NC+SA). The\\n' >> "$RECEIPT"
printf '   AWS Open Data registry lists the licence field as Commercial - a CONFLICT recorded in\\n' >> "$RECEIPT"
printf '   CRITERIA_REGISTRY.json benchmarks.nuscenes.licence_status; treat as research-only."\n}\n' >> "$RECEIPT"
echo "receipt: $RECEIPT"
echo "NEXT: extract ONLY what is needed (exFAT clusters are 1 MiB) — see the header of this file."
