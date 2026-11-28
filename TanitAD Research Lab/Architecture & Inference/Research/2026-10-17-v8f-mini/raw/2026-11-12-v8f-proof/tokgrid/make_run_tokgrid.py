"""make_run_tokgrid.py -- writes run_tokgrid.sh (card PREREG_V8F_PROOF_ADD87, TOKGRID) from ../tokclose/run_tokclose.sh (card ADD81, sha256 first 16 4def827fd897ce17) by counted edits.
Each old text has to occur exactly once in the parent, else the generator stops with REFUSED.
Pod layout: V=/workspace/v8proof, root W=$V/tokgrid (exists before the start), tools $W/tools/tokclose_fit_g8.py and $W/tools/pair_tokgrid.py, the other tool files where the parent expects them.
Steps in order: (i) refuse unless $V/tokclose2/TOKCLOSE2_DONE.txt exists; (ii) self-test of the generated tool; (iii) control g0: score of the generated tool on the fit folder of card ADD86;
(iv) the four E fits of card ADD81 copied with a sha256 equality check and the lists c0_copies / c0_sources; (v) five fits with TOKCLOSE_T_GRID=pool8x20 and the ADD86 fit setting;
(vi) score with --allow-hp-override; then pair_tokgrid.py. Markers START_TOKGRID.txt, TOKGRID_STOP.txt, TOKGRID_DONE.txt, log queue_tokgrid.log.
Nothing else of the parent changes (slot waiting before every fit, DRY=1, no removal, no move, no stopping, no process listing)."""
import hashlib, io, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.join(HERE, "..", "tokclose", "run_tokclose.sh")
PARENT_SHA = "4def827fd897ce17"

G = "tokclose_fit_g8.py"
EDITS = [
    # 1 roots
    ("W=${TC_W:-$V/tokclose}\n", "W=${TC_W:-$V/tokgrid}\nSRC1=${TC_SRC1:-$V/tokclose}\nSRC2=${TC_SRC2:-$V/tokclose2}\n"),
    # 2 logs and markers
    ("\"$W/queue_tokclose.log\"", "\"$W/queue_tokgrid.log\""),
    ("\"$W/TOKCLOSE_STOP.txt\"", "\"$W/TOKGRID_STOP.txt\""),
    ("\"$W/TOKCLOSE_DONE.txt\"", "\"$W/TOKGRID_DONE.txt\""),
    ("> \"$W/START_TOKCLOSE.txt\" ) 2>/dev/null || { echo \"REFUSED: $W/START_TOKCLOSE.txt exists (a chain was started)\"; exit 2; }\n",
     "> \"$W/START_TOKGRID.txt\" ) 2>/dev/null || { echo \"REFUSED: $W/START_TOKGRID.txt exists (a chain was started)\"; exit 2; }\n"),
    # 3 step (i): refuse unless card ADD86 is done (before the start marker is written)
    ("  [ -d \"$W\" ] || { echo \"REFUSED: no $W\"; exit 2; }\n",
     "  [ -d \"$W\" ] || { echo \"REFUSED: no $W\"; exit 2; }\n"
     "  [ -f \"$SRC2/TOKCLOSE2_DONE.txt\" ] || { echo \"REFUSED: card ADD86 is not done ($SRC2/TOKCLOSE2_DONE.txt is missing)\"; exit 2; }\n"),
    ("log \"chain start (DRY=$DRY): card ADD81 (TOKCLOSE): 8 fits + 1 planted fit, then score\"\n",
     "log \"chain start (DRY=$DRY): card ADD87 (TOKGRID): self-test, control g0, 4 E fits copied, 5 fits on pool8x20, score, pair\"\n"
     "[ \"$DRY\" != 1 ] || echo \"CHECK: test -f $SRC2/TOKCLOSE2_DONE.txt (the chain is refused when it is missing)\"\n"),
    # 4 tool files
    ("for f in \"$TOOLS/tokclose_fit.py\" \"$CT/closeread_dump.py\"", "for f in \"$TOOLS/%s\" \"$TOOLS/pair_tokgrid.py\" \"$CT/closeread_dump.py\"" % G),
    ("\"$PY\" \"$TOOLS/tokclose_fit.py\" --selftest > \"$LOGD/selftest.log\"", "\"$PY\" \"$TOOLS/%s\" --selftest > \"$LOGD/selftest.log\"" % G),
    ("echo \"CMD: $PY $TOOLS/tokclose_fit.py --selftest\"", "echo \"CMD: $PY $TOOLS/%s --selftest\"" % G),
    ("local cmd=(\"$PY\" \"$TOOLS/tokclose_fit.py\" fit", "local cmd=(\"$PY\" \"$TOOLS/%s\" fit" % G),
    # 5 steps (iii) to (v): control g0, the four E fits copied, the five fits on the fine grid
    ("for k in 0 1; do for arm in E10 E25; do fit \"$arm\" \"$k\"; done; done\n"
     "for k in 0 1; do for arm in T10 T25; do fit \"$arm\" \"$k\"; done; done\n"
     "fit T10 0 shuf\n",
     "SCBASE=(--dump-dir \"$DUMP\" --est-dir \"$EST\" --add78-result \"$ADD78\" --episode-map \"$EPMAP\" --eval-cache \"$EVALC\" --join-eval \"$JE\" --build-dir \"$B\")\n"
     "# step (iii), control g0: the scorer of the generated tool on the fit folder of card ADD86 (the reader compares it with the result file of card ADD86)\n"
     "G0=(\"$PY\" \"$TOOLS/%s\" score --fit-dir \"$SRC2/fits\" \"${SCBASE[@]}\" --out-dir \"$W/g0_score\" --allow-hp-override)\n"
     "if [ \"$DRY\" = 1 ]; then echo \"CMD: ${G0[*]}\"; else\n"
     "  done_step g0_score || { \"${G0[@]}\" > \"$LOGD/g0_score.log\" 2>&1; g0rc=$?; log \"control g0 score rc $g0rc\"; mark g0_score; }\n"
     "fi\n"
     "# step (iv): the four E fits of card ADD81 are taken over as copies (control c0 compares their sha256)\n"
     "for tag in E10_s0 E25_s0 E10_s1 E25_s1; do for sfx in .pt _log.json _table.npz; do\n"
     "  src=\"$SRC1/fits/tokclose_$tag$sfx\"; dst=\"$FITS/tokclose_$tag$sfx\"\n"
     "  if [ \"$DRY\" = 1 ]; then echo \"CMD: cp $src $dst\"; continue; fi\n"
     "  [ -f \"$src\" ] || stop \"SOURCE-FIT-MISSING $tag$sfx\"\n"
     "  [ -e \"$dst\" ] || cp \"$src\" \"$dst\" || stop \"COPY-FAILED $tag$sfx\"\n"
     "  [ \"$(sha256sum < \"$src\")\" = \"$(sha256sum < \"$dst\")\" ] || stop \"COPY-NOT-EQUAL $tag$sfx\"\n"
     "done; done\n"
     "[ \"$DRY\" = 1 ] || ( cd \"$FITS\" && sha256sum tokclose_E*_s[01]* ) > \"$LOGD/c0_copies.sha256\"\n"
     "[ \"$DRY\" = 1 ] || ( cd \"$SRC1/fits\" && sha256sum tokclose_E*_s[01]* ) > \"$LOGD/c0_sources.sha256\"\n"
     "# step (v): the five fits on the fine grid, with the fit setting of card ADD86\n"
     "export TOKCLOSE_T_GRID=pool8x20\n"
     "export TOKCLOSE_HP_OVERRIDE='{\"max_evals\": 24, \"patience\": 4}'\n"
     "[ \"$DRY\" != 1 ] || echo \"ENV: TOKCLOSE_T_GRID=$TOKCLOSE_T_GRID TOKCLOSE_HP_OVERRIDE=$TOKCLOSE_HP_OVERRIDE\"\n"
     "for k in 0 1; do for arm in T10 T25; do fit \"$arm\" \"$k\"; done; done\n"
     "fit T10 0 shuf\n"
     "unset TOKCLOSE_T_GRID TOKCLOSE_HP_OVERRIDE\n" % G),
    # 6 step (vi) and the pair
    ("SC=(\"$PY\" \"$TOOLS/tokclose_fit.py\" score --fit-dir \"$FITS\" --dump-dir \"$DUMP\" --est-dir \"$EST\" --add78-result \"$ADD78\" --episode-map \"$EPMAP\" --eval-cache \"$EVALC\" --join-eval \"$JE\" --build-dir \"$B\" --out-dir \"$OUT\")\n",
     "SC=(\"$PY\" \"$TOOLS/%s\" score --fit-dir \"$FITS\" \"${SCBASE[@]}\" --out-dir \"$OUT\" --allow-hp-override)\n"
     "PC=(\"$PY\" \"$TOOLS/pair_tokgrid.py\" --fit-dir-fine \"$FITS\" --fit-dir-coarse \"$SRC2/fits\" \"${SCBASE[@]}\" --out-dir \"$W/pair\")\n" % G),
    ("if [ \"$DRY\" = 1 ]; then echo \"CMD: ${SC[*]}\"; echo \"DRY run complete\"; exit 0; fi\n",
     "if [ \"$DRY\" = 1 ]; then echo \"CMD: ${SC[*]}\"; echo \"CMD: ${PC[*]}\"; echo \"DRY run complete\"; exit 0; fi\n"),
    ("rc=$?\necho \"$(ts) score rc $rc (0 = controls pass and names written; 1 = a control failed, no name)\"",
     "rc=$?\n\"${PC[@]}\" > \"$LOGD/pair.log\" 2>&1\nprc=$?\n"
     "echo \"$(ts) score rc $rc pair rc $prc (score rc 0 = controls pass and names written; 1 = a control failed, no name)\""),
    # 7 comments
    ("# run_tokclose.sh -- pod chain of card PREREG_V8F_PROOF_ADD81 (TOKCLOSE), tier T0:",
     "# run_tokgrid.sh -- pod chain of card PREREG_V8F_PROOF_ADD87 (TOKGRID), tier T0, written by make_run_tokgrid.py from run_tokclose.sh 4def827fd897ce17. Refused unless card ADD86 is done.\n"
     "# Steps: self-test of tokclose_fit_g8.py; control g0 (score of the generated tool on the fits of card ADD86); the four E fits of card ADD81 copied; five fits on the grid pool8x20 with at most 24\n"
     "# evaluations and patience 4; score with --allow-hp-override; pair_tokgrid.py. Markers START_TOKGRID.txt / TOKGRID_DONE.txt / TOKGRID_STOP.txt, log queue_tokgrid.log. Text of the parent follows.\n"
     "# run_tokclose.sh -- pod chain of card PREREG_V8F_PROOF_ADD81 (TOKCLOSE), tier T0:"),
]


def main():
    raw = io.open(PARENT, "rb").read()
    if hashlib.sha256(raw).hexdigest()[:16] != PARENT_SHA:
        sys.exit("REFUSED: parent hash is not %s" % PARENT_SHA)
    s = raw.decode("utf-8").replace("\r\n", "\n")
    for i, (old, new) in enumerate(EDITS, 1):
        n = s.count(old)
        if n != 1:
            sys.exit("REFUSED: edit %d: %d matches of %r" % (i, n, old[:70]))
        s = s.replace(old, new)
    out = os.path.join(HERE, "run_tokgrid.sh")
    io.open(out, "w", encoding="utf-8", newline="\n").write(s)
    print("wrote", out, "%d edits" % len(EDITS), hashlib.sha256(s.encode("utf-8")).hexdigest()[:16])


if __name__ == "__main__":
    main()
