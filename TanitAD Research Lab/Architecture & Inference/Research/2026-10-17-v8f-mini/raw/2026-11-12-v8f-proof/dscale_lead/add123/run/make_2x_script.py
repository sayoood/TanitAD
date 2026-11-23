"""make_2x_script.py -- writes pod_dscale_lead_2x.sh (card PREREG_V8F_PROOF_ADD123, DSCALE-LEAD-2X) from the banked pod_dscale_lead_run.sh by ASSERTED text replaces.
Every old text must occur exactly once in the text it is applied to, else the generator stops (exit 3) and writes nothing.

  python make_2x_script.py [--src <banked pod_dscale_lead_run.sh>] [--out <pod_dscale_lead_2x.sh>]
Default source: ../../pod_dscale_lead_run.sh (the banked kit folder above this one); default output: beside this file.
Nothing is deleted or moved.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))

CK_SHA_LINE = 'echo "tok_probe_u_ck.py sha256: $(sha256sum "$TPCK" | cut -c1-64)"'

REPLACES = [
    # --- header comments
    (r"""# pod_dscale_lead_run.sh -- run kit of card PREREG_V8F_PROOF_ADD45 (DSCALE-LEAD) with note N1, for the pod of the other v8F session. Tier T0.""",
     r"""# pod_dscale_lead_2x.sh -- run kit of card PREREG_V8F_PROOF_ADD123 (DSCALE-LEAD-2X), generated from pod_dscale_lead_run.sh by make_2x_script.py, for the pod of the other v8F session. Tier T0; no model is trained."""),
    (r"""#   usage: bash pod_dscale_lead_run.sh <stage>""", r"""#   usage: bash pod_dscale_lead_2x.sh <stage>"""),
    (r"""#   smoke    : 200 steps of the four arms of folder runC form on the union (no clip list), under the flock; prints the projected GPU hours of all five folders; exit 4 ESTIMATE-ABOVE-6H if above $MAX_H""",
     r"""#   smoke    : 200 steps of the four arms of folder runQ form (banked tool), under the flock; prints the projected GPU hours of the ONE training process (seconds/200 x 38328); exit 4 ESTIMATE-ABOVE-0.9H if above $MAX_H (0.9)"""),
    (r"""#   run      : runC, runU, runU2, runCSH, runUSH one after another, each ONE GPU process under the flock (threads <= 4), exactly --steps 19164 (runU2: 38328) = the same step count and schedule""",
     r"""#   run      : runQ = the arms TK16_s0 to TK16_s3 at 38328 steps, ONE GPU process under the flock (threads <= 4) through tok_probe_u_ck.py (--ck-steps 19164,28746), under timeout TRAIN_LIMIT_S (3300); rc 124 = stop TRAIN-ABOVE-TIME-LIMIT"""),
    (r"""#   extract  : evaluation split only, tokens from $EVAL_CACHE, per folder, under the flock""",
     r"""#   extract  : evaluation split only, tokens from $EVAL_CACHE, per arm THREE extracts under the flock (the last state; steps 19164 and 28746 through --ckpt into states/runQ_s19164 and states/runQ_s28746)"""),
    (r"""writes $OUT/DSCALE_LEAD_STOP.txt (with the reason) on a stop, $OUT/DSCALE_LEAD_DONE.txt after checklog.""",
     r"""writes $OUT/DSCALE_2X_STOP.txt (with the reason) on a stop, $OUT/DSCALE_2X_DONE.txt after checklog."""),
    # --- variables
    (r"""PINS=${PINS:-$KIT/TREE_PINS.txt}""", r"""PINS=${PINS:-$KIT/TREE_PINS_2X.txt}"""),
    (r"""MAX_H=${MAX_H:-6}""", r"""MAX_H=${MAX_H:-0.9}"""),
    (r"""STEPS=${STEPS:-19164}; STEPS_U2=${STEPS_U2:-38328}; SMOKE_STEPS=${SMOKE_STEPS:-200}""",
     r"""STEPS=${STEPS:-19164}; STEPS_U2=${STEPS_U2:-38328}; SMOKE_STEPS=${SMOKE_STEPS:-200}; TRAIN_LIMIT_S=${TRAIN_LIMIT_S:-3300}"""),
    (r"""TP=$KIT/tok_probe_u.py""", "TP=$KIT/tok_probe_u.py\nTPCK=$KIT/tok_probe_u_ck.py"),
    # --- folders
    (r"""# folder -> arms, steps, extra flags (the card's arm names: P16C_sK = runC/TK16_sK, P16U_sK = runU/TK16_sK, P16U2_s0 = runU2/TK16_s0, P16CSH_s0 = runCSH/TK16_s0, P16USH_s0 = runUSH/TK16_s0)""",
     r"""# folder -> arms, steps, extra flags (the card's arm names: P16Q_sK = runQ/TK16_sK)"""),
    (r'''FOLDERS="runC runU runU2 runCSH runUSH"''', r'''FOLDERS="runQ"'''),
    (r"""case $1 in runC|runU) echo""", r"""case $1 in runQ) echo"""),
    (r"""case $1 in runU2) echo "$STEPS_U2";;""", r"""case $1 in runQ) echo "$STEPS_U2";;"""),
    (r"""case $1 in runC) echo "--clip-list $BSET";; runCSH) echo "--clip-list $BSET --shuffle-gap-labels";; runUSH) echo "--shuffle-gap-labels";; *) echo "";; esac""",
     r"""case $1 in *) echo "";; esac"""),
    # --- log, stop and done names
    (r'''] dscale_lead: $*"''', r'''] dscale_2x: $*"'''),
    (r'''"$OUT/dscale_lead.log"''', r'''"$OUT/dscale_2x.log"'''),
    (r"""printf 'DSCALE_LEAD_STOP %s""", r"""printf 'DSCALE_2X_STOP %s"""),
    (r"""> "$OUT/DSCALE_LEAD_STOP.txt"; fi; exit""", r"""> "$OUT/DSCALE_2X_STOP.txt"; fi; exit"""),
    (r"""[ -f "$OUT/DSCALE_LEAD_DONE.txt" ] && { echo "REFUSED: $OUT/DSCALE_LEAD_DONE.txt exists"; exit 3; }""",
     r"""[ -f "$OUT/DSCALE_2X_DONE.txt" ] && { echo "REFUSED: $OUT/DSCALE_2X_DONE.txt exists"; exit 3; }"""),
    (r"""[ -f "$OUT/DSCALE_LEAD_STOP.txt" ] && { echo "REFUSED: $OUT/DSCALE_LEAD_STOP.txt exists (read it, then use a new OUT)"; exit 3; }""",
     r"""[ -f "$OUT/DSCALE_2X_STOP.txt" ] && { echo "REFUSED: $OUT/DSCALE_2X_STOP.txt exists (read it, then use a new OUT)"; exit 3; }"""),
    # --- smoke: the one training process, bar 0.9 h
    (r'''build_train "$SMOKE_STEPS" "$(folder_arms runC)" "$d"''', r'''build_train "$SMOKE_STEPS" "$(folder_arms runQ)" "$d"'''),
    (r"""projection = seconds/$SMOKE_STEPS x (2 x $STEPS + $STEPS_U2 + 2 x $STEPS) steps, bar ${MAX_H} h""",
     r"""projection = seconds/$SMOKE_STEPS x $STEPS_U2 steps (the ONE training process), bar ${MAX_H} h"""),
    (r"""(s/n)*(2*a+b+2*a)/3600""", r"""(s/n)*b/3600"""),
    (r"""projected GPU hours of the five folders: ${proj} h (upper bound: the three one-arm folders priced as four-arm; data build and extracts not included)""",
     r"""projected GPU hours of the ONE training process of $STEPS_U2 steps (four arms): ${proj} h (data build and extracts not included)"""),
    (r"""stop "ESTIMATE-ABOVE-6H: projected ${proj} h is above ${MAX_H} h; the run goes back to the PI" 4""",
     r"""stop "ESTIMATE-ABOVE-0.9H: projected ${proj} h is above ${MAX_H} h; the run goes back to the PI" 4"""),
    # --- run: the saving tool under the time limit
    (r'''    gpu "train $f" "${CMD[@]}"''',
     "    CMD[1]=\"$TPCK\"; CMD+=(--ck-steps 19164,28746); CMD=(timeout \"$TRAIN_LIMIT_S\" \"${CMD[@]}\")   # the saving tool, under the time limit (the wait for the lock is not counted)\n"
     "    gpu \"train $f\" \"${CMD[@]}\""),
    (r"""      [ $rc -eq 0 ] || [ $rc -eq 2 ] || stop "train $f exited $rc" 1""",
     "      [ $rc -eq 124 ] && stop \"TRAIN-ABOVE-TIME-LIMIT: train $f ran past ${TRAIN_LIMIT_S} s and was stopped (rc 124); the run goes back to the PI\" 1\n"
     r"""      [ $rc -eq 0 ] || [ $rc -eq 2 ] || stop "train $f exited $rc" 1"""),
    (r"""      [ -f "$d/TRAIN_DONE.json" ] || stop "train $f wrote no TRAIN_DONE.json (rc $rc)" 1""",
     r"""      [ -f "$d/TRAIN_DONE.json" ] || stop "train $f wrote no TRAIN_DONE.json (rc $rc)" 1""" + "\n"
     r"""      [ -f "$d/CK_RECORD.json" ] || stop "train $f wrote no CK_RECORD.json (rc $rc): the saving tool did not run to its end" 1"""),
    # --- extract: three extracts per arm
    (r"""  local f a rc
""", "  local f a rc tag xsrc xdir\n"),
    (r"""[ "$DRY" != 1 ] && [ -d "$sd" ] && stop "folder $sd exists without a marker: not overwritten" 1""",
     r"""[ "$DRY" != 1 ] && [ -d "$sd" -o -d "${sd}_s19164" -o -d "${sd}_s28746" ] && stop "a states folder of $f ($sd, ${sd}_s19164 or ${sd}_s28746) exists without a marker: not overwritten" 1"""),
    ("""      gpu "extract $f $a" "$PY" "$TP" extract --grid pool4x10 --tree "$TREE" --run-dir "$OUT/$f" --arm "$a" --cache "$EVAL_CACHE" --states-dir "$sd" --splits eval --eval-range "$EVAL_RANGE" --device "$DEVICE" --threads "$THREADS"
      rc=$?
      [ "$DRY" = 1 ] || [ $rc -eq 0 ] || stop "extract $f $a failed (rc $rc)" 1
""",
     """      for tag in last s19164 s28746; do
        if [ "$tag" = last ]; then xsrc=(--run-dir "$OUT/$f"); xdir=$sd; else xsrc=(--ckpt "$OUT/$f/$a/ckpt_step${tag#s}.pt"); xdir=${sd}_$tag; fi
        gpu "extract $f $a $tag" "$PY" "$TP" extract --grid pool4x10 --tree "$TREE" "${xsrc[@]}" --arm "$a" --cache "$EVAL_CACHE" --states-dir "$xdir" --splits eval --eval-range "$EVAL_RANGE" --device "$DEVICE" --threads "$THREADS"
        rc=$?
        [ "$DRY" = 1 ] || [ $rc -eq 0 ] || stop "extract $f $a $tag failed (rc $rc)" 1
      done
"""),
    # --- checklog: the done file
    (r"""{ echo "DSCALE_LEAD_DONE $(date -u""", r"""{ echo "DSCALE_2X_DONE $(date -u"""),
    (r'''echo "dscale_precheck.py sha256: $(sha256sum "$PC" | cut -c1-64)"''',
     CK_SHA_LINE + "\n" + r'''      echo "dscale_precheck.py sha256: $(sha256sum "$PC" | cut -c1-64)"'''),
    (r"""echo "ckpt $f/$a $(sha256sum "$OUT/$f/$a/ckpt_last.pt" | cut -c1-64)"; done; done""",
     r"""for c in ckpt_last ckpt_step19164 ckpt_step28746; do echo "ckpt $f/$a/$c $(sha256sum "$OUT/$f/$a/$c.pt" | cut -c1-64)"; done; done; done"""),
    (r'''} > "$OUT/DSCALE_LEAD_DONE.txt"''', r'''} > "$OUT/DSCALE_2X_DONE.txt"'''),
    (r'''say "DONE: $OUT/DSCALE_LEAD_DONE.txt"''', r'''say "DONE: $OUT/DSCALE_2X_DONE.txt"'''),
]

# words that must not appear in the generated text (built from parts so that this file does not hold them as one word)
BANNED = ["vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no lon" + "ger", "closed " + "loop"]


def apply_replaces(text, pairs=None):
    """Apply every (old, new) in order; each old must occur exactly once in the text as it is at that point, else SystemExit(3)."""
    for i, (old, new) in enumerate(pairs if pairs is not None else REPLACES):
        n = text.count(old)
        if n != 1:
            sys.stderr.write("REFUSED: replace %d: the old text occurs %d times (needs exactly 1): %r\n" % (i + 1, n, old[:120]))
            raise SystemExit(3)
        text = text.replace(old, new)
    return text


def final_checks(text):
    bad = []
    if "DSCALE_LEAD_" in text:
        bad.append("a banked marker name DSCALE_LEAD_ is left")
    if "ESTIMATE-ABOVE-6H" in text:
        bad.append("ESTIMATE-ABOVE-6H is left")
    if re.search(r"run(C|U|U2|CSH|USH)\b", text):
        bad.append("a banked folder name is left")
    low = text.lower()
    for w in BANNED:
        if w in low:
            bad.append("banned word %r" % w)
    if re.search(r"(^|[;&|(]|\s)(rm|mv)\s", text.replace("\n", " \n ")):
        bad.append("an rm or mv command")
    if text.count('FOLDERS="runQ"') != 1:
        bad.append("FOLDERS line")
    if bad:
        sys.stderr.write("REFUSED: the generated text fails its checks: %s\n" % "; ".join(bad))
        raise SystemExit(3)
    return text


def build_text(src_text):
    if "\r" in src_text:
        sys.stderr.write("REFUSED: the source has CR characters\n")
        raise SystemExit(3)
    return final_checks(apply_replaces(src_text))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(HERE), "..", "pod_dscale_lead_run.sh"))
    ap.add_argument("--out", default=os.path.join(HERE, "pod_dscale_lead_2x.sh"))
    a = ap.parse_args(argv)
    with open(a.src, "rb") as fh:
        raw = fh.read()
    text = build_text(raw.decode("utf-8"))
    if os.path.exists(a.out):
        sys.stderr.write("REFUSED: %s exists (nothing is overwritten; give a new --out)\n" % a.out)
        return 3
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote %s  source sha256 %s  output sha256 %s" % (a.out, hashlib.sha256(raw).hexdigest()[:16], hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
