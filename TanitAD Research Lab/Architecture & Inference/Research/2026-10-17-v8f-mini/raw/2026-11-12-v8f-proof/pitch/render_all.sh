#!/usr/bin/env bash
# render_all.sh : four parallel segments of the film (headless Chrome, no GPU), then a stream-copy concat. Output C:/Users/Admin/v8proof_c/pitch/TanitAD_pitch_film.mp4
SP=/c/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/d1b9de00-7d76-4030-8bc5-507afd36195b/scratchpad
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
FF=/c/Users/Admin/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1.1-full_build/bin/ffmpeg
OUT=C:/Users/Admin/v8proof_c/pitch
cd $SP/pitch || exit 1
SPW=$(cygpath -m $SP)
i=0
for seg in "0 117" "117 234" "234 351" "351 468"; do
  set -- $seg
  RPORT=$((9381 + i)) RFMT=jpeg $PY render.py video $SPW/pitch/tanitad_pitch_film.html $OUT/seg_$i.mp4 30 $1 $2 > seg_$i.log 2>&1 &
  i=$((i + 1))
done
wait
tail -q -n 1 seg_0.log seg_1.log seg_2.log seg_3.log
grep -c "ZZ VIDEO DONE" seg_0.log seg_1.log seg_2.log seg_3.log | tr "\n" " "; echo
printf "file 'seg_0.mp4'\nfile 'seg_1.mp4'\nfile 'seg_2.mp4'\nfile 'seg_3.mp4'\n" > $OUT/concat.txt
(cd $OUT && $FF -y -loglevel error -f concat -safe 0 -i concat.txt -c copy -movflags +faststart -metadata title="TanitAD pitch film" TanitAD_pitch_film.mp4; echo "concat rc $?")
${FF}probe -v error -show_entries format=duration,size -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames -of default=nw=1 $OUT/TanitAD_pitch_film.mp4
echo "ZZ RENDER ALL DONE $(date -u +%H:%M:%SZ)"
