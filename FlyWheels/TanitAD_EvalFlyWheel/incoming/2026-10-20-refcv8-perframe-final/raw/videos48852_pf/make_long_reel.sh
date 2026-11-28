#!/bin/sh
# Long video of refcv8 EMA-48,852 (per-frame feed, RT layer ON): waits for ZZP48RENDEREND, then concatenates a RULE-chosen
# episode list (every 10th eval139 episode index 0,10,...,130 + the three earlier rule picks 114 / 13 / 103, never chosen by
# how the plan looks) into one mp4 (stream copy, full quality) and a smaller share copy (CRF 26). Markers ZZREEL<NAME>ZZ.
VID=E:/refcv8_eval_kit/videos/final48852_pf; OUT=C:/Users/Admin/refcv8_pf/reel48852; LOG=$OUT/reel.log
P48=/c/Users/Admin/refcv8_pf/post48852/p48.log
mark() { echo "ZZREEL$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
mark START "pid $$ waits for ZZP48RENDEREND"
until grep -qE 'ZZP48(RENDEREND|REFUSE)ZZ' "$P48" 2>/dev/null; do sleep 30; done
EPS="000 010 013 020 030 040 050 060 070 080 090 100 103 110 114 120 130"
: > "$OUT/list.txt"; n=0
for e in $EPS; do f="$VID/ep$e.mp4"; if [ -s "$f" ]; then echo "file '$f'" >> "$OUT/list.txt"; n=$((n+1)); else mark MISSING "ep$e"; fi; done
mark LIST "n=$n"
ffmpeg -loglevel error -y -f concat -safe 0 -i "$OUT/list.txt" -c copy "$OUT/refcv8_48852_perframe_long_full.mp4"
mark FULL "rc=$? bytes=$(stat -c %s "$OUT/refcv8_48852_perframe_long_full.mp4" 2>/dev/null)"
ffmpeg -loglevel error -y -i "$OUT/refcv8_48852_perframe_long_full.mp4" -c:v libx264 -preset medium -crf 26 -pix_fmt yuv420p -an "$OUT/refcv8_48852_perframe_long_share.mp4"
mark SHARE "rc=$? bytes=$(stat -c %s "$OUT/refcv8_48852_perframe_long_share.mp4" 2>/dev/null) dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/refcv8_48852_perframe_long_share.mp4" 2>/dev/null)"
mark END ""
