"""make_run_tokclose2.py -- writes run_tokclose2.sh (card PREREG_V8F_PROOF_ADD86, TOKCLOSE2) from ../tokclose/run_tokclose.sh (card ADD81) by counted edits.
Changes: the root is /workspace/v8proof/tokclose2 with the tools of the first root; the four E fits and the planted fit are taken over as file copies from the first root (control c0 compares
their sha256); the four T fits run with TOKCLOSE_HP_OVERRIDE = 24 evaluations, patience 4; the score runs with --allow-hp-override; the done marker is TOKCLOSE2_DONE.txt.
Nothing else of the parent changes (slot waiting, markers, no removal, no process listing, no stopping)."""
import hashlib, io, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.join(HERE, "..", "tokclose", "run_tokclose.sh")
PARENT_SHA = "4def827fd897ce17"


def main():
    raw = io.open(PARENT, "rb").read()
    if hashlib.sha256(raw).hexdigest()[:16] != PARENT_SHA:
        sys.exit("REFUSED: parent hash is not %s" % PARENT_SHA)
    s = raw.decode("utf-8").replace("\r\n", "\n")
    edits = [
        ("W=${TC_W:-$V/tokclose}\n", "W=${TC_W:-$V/tokclose2}\nSRCW=${TC_SRCW:-$V/tokclose}\n"),
        ("TOOLS=$W/tools\n", "TOOLS=$SRCW/tools\n"),
        ("for k in 0 1; do for arm in E10 E25; do fit \"$arm\" \"$k\"; done; done\n"
         "for k in 0 1; do for arm in T10 T25; do fit \"$arm\" \"$k\"; done; done\n"
         "fit T10 0 shuf\n",
         "# card ADD86: the E fits and the planted fit of card ADD81 are taken over as copies; only the four T fits run, with the longer cap\n"
         "for tag in E10_s0 E25_s0 E10_s1 E25_s1 T10shuf_s0; do for sfx in .pt _log.json _table.npz; do\n"
         "  src=\"$SRCW/fits/tokclose_$tag$sfx\"; dst=\"$FITS/tokclose_$tag$sfx\"\n"
         "  if [ \"$DRY\" = 1 ]; then echo \"CMD: cp $src $dst\"; continue; fi\n"
         "  [ -f \"$src\" ] || stop \"SOURCE-FIT-MISSING $tag$sfx\"\n"
         "  [ -e \"$dst\" ] || cp \"$src\" \"$dst\" || stop \"COPY-FAILED $tag$sfx\"\n"
         "  [ \"$(sha256sum < \"$src\")\" = \"$(sha256sum < \"$dst\")\" ] || stop \"COPY-NOT-EQUAL $tag$sfx\"\n"
         "done; done\n"
         "[ \"$DRY\" = 1 ] || ( cd \"$FITS\" && sha256sum tokclose_E*_s[01]* tokclose_T10shuf_s0* ) > \"$LOGD/c0_copies.sha256\"\n"
         "[ \"$DRY\" = 1 ] || ( cd \"$SRCW/fits\" && sha256sum tokclose_E*_s[01]* tokclose_T10shuf_s0* ) > \"$LOGD/c0_sources.sha256\"\n"
         "export TOKCLOSE_HP_OVERRIDE='{\"max_evals\": 24, \"patience\": 4}'\n"
         "for k in 0 1; do for arm in T10 T25; do fit \"$arm\" \"$k\"; done; done\n"
         "unset TOKCLOSE_HP_OVERRIDE\n"),
        ("--build-dir \"$B\" --out-dir \"$OUT\")\n", "--build-dir \"$B\" --out-dir \"$OUT\" --allow-hp-override)\n"),
        ("\"$W/TOKCLOSE_DONE.txt\"", "\"$W/TOKCLOSE2_DONE.txt\""),
        ("card ADD81 (TOKCLOSE): 8 fits + 1 planted fit, then score\"",
         "card ADD86 (TOKCLOSE2): 5 fits of card ADD81 copied, 4 T fits with at most 24 evaluations and patience 4, then score\""),
        ("# run_tokclose.sh -- pod chain of card PREREG_V8F_PROOF_ADD81 (TOKCLOSE), tier T0:",
         "# run_tokclose2.sh -- pod chain of card PREREG_V8F_PROOF_ADD86 (TOKCLOSE2), tier T0, written by make_run_tokclose2.py from run_tokclose.sh 4def827fd897ce17: the four T fits with at most 24\n"
         "# evaluations and patience 4 (TOKCLOSE_HP_OVERRIDE), the E fits and the planted fit of card ADD81 taken over as copies, then the score with --allow-hp-override. Text of the parent follows.\n"
         "# run_tokclose.sh -- pod chain of card PREREG_V8F_PROOF_ADD81 (TOKCLOSE), tier T0:"),
    ]
    for old, new in edits:
        if s.count(old) != 1:
            sys.exit("REFUSED: %d matches of %r" % (s.count(old), old[:60]))
        s = s.replace(old, new)
    out = os.path.join(HERE, "run_tokclose2.sh")
    io.open(out, "w", encoding="utf-8", newline="\n").write(s)
    print("wrote", out, hashlib.sha256(s.encode("utf-8")).hexdigest()[:16])


if __name__ == "__main__":
    main()
