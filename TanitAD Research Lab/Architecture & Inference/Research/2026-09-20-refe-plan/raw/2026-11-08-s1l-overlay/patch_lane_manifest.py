"""Adds the Amendment 4 lane-draw check to the S1L overlay's scorer1_chain.sh and regenerates its diff against the base blob."""
import difflib, re, subprocess
S = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/s1l/"
F = S + "files/scorer1_chain.sh"
BASE = "2317a935c5c7e89ed5a3a2175689759648e9134c"
b = open(F, "rb").read().decode("utf-8")
assert "\r\n" not in b and "lane_check" not in b
pod = open(S + "files/scorer1_pod.sh", "rb").read().decode("utf-8")
m = re.findall(r"--teacher-lane-train (\S+)", pod)
assert m and set(m) == {"$D/train_teacher_lane"}, m
def sub(old, new):
    global b
    assert b.count(old) == 1, (old[:70], b.count(old))
    b = b.replace(old, new)
BLOCK = r'''# >>> lane manifest (PREREG_SCORER1 AMENDMENT 4): every arm trains on ONE lane draw.  File names + sha256, pinned once before the first arm and re-checked at the start of every arm; the row count in the inputs stage cannot see a different draw.
LANE_DIR=$D/train_teacher_lane
LANE_MAN=$ROOT/train_teacher_lane.manifest.sha256
lane_list() { (cd "$LANE_DIR" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum); }
lane_pin() {
  [ -s "$LANE_MAN" ] && return 0
  lane_list > "$LANE_MAN.tmp"
  [ -s "$LANE_MAN.tmp" ] || refuse "$LANE_DIR holds no files: no lane draw to pin"
  mv "$LANE_MAN.tmp" "$LANE_MAN"
}
lane_check() {          # uses $ARGS (arm_argv)
  [ -s "$LANE_MAN" ] || refuse "no lane manifest at $LANE_MAN: the inputs stage writes it before the first arm"
  lane_list | cmp -s - "$LANE_MAN" || refuse "$LANE_DIR differs from the manifest pinned before the first arm: the arms would not share one lane draw"
  case " $ARGS " in *" --teacher-lane-train $LANE_DIR "*) ;; *) refuse "this arm's argv does not read --teacher-lane-train $LANE_DIR";; esac
}
# <<< lane manifest
'''
A = 'prof_of() { if [ "$1" = "S1L" ]; then echo refe_scorer1l; else echo refe_scorer1; fi; }'
i = b.index(A)
assert b.count(A) == 1
b = b[:i] + BLOCK + b[i:]
sub('log "ZZSCORER1_INPUTS_OK"; stop_after inputs',
    'lane_pin\nlog "lane manifest sha256 $(sha256sum "$LANE_MAN" | cut -c1-64)"; sed \'s/^/lane_manifest /\' "$LANE_MAN" | head -n 200 >> $L\nlog "ZZSCORER1_INPUTS_OK"; stop_after inputs')
sub('  arm_argv "$ARMN"\n  if ! stage_done "preflight_$LA"; then',
    '  arm_argv "$ARMN"\n  lane_check; log "ZZSCORER1_LANE_DRAW_OK $ARMN"\n  if ! stage_done "preflight_$LA"; then')
open(F, "wb").write(b.encode("utf-8"))
o = subprocess.run(["git", "-C", "C:/Users/Admin/tanitad-push", "cat-file", "blob", BASE], check=True, capture_output=True).stdout.decode("utf-8")
d = difflib.unified_diff(o.splitlines(), b.splitlines(), "a/refe/scorer1_chain.sh", "b/refe/scorer1_chain.sh", lineterm="")
open(S + "diffs/scorer1_chain.sh.diff", "w", encoding="utf-8", newline="\n").write("\n".join(d) + "\n")
print("ZZPATCH_OK")
