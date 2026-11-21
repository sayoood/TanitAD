"""make_tok_probe_lr.py -- writes tok_probe_lr.py from ../finegrid_r/tok_probe_gr.py by asserted text edits (card PREREG_V8F_PROOF_ADD46, INTEG10K-PRE, part 2).
The original is never edited. Every old text must occur exactly once in the original, else the generator stops.  Run: python make_tok_probe_lr.py

The ONLY change of behaviour: the clips whose held-out value is divisible by 10 (int of the first 8 hex digits of sha256 of the clip id) are left out of training
(the window list is rebuilt by the trainer's own call `D.train_windows` on the remaining clips). The rest is a record: the held-out count and the sha256 of the held-out id list go into train_meta.json.
Arms: the arm names of the original are kept (TK16_s0, TK16_s1); the reader arms LR16_s0 / LR16_s1 of the card are these two arms trained by this tool into the folder lr16/ (the queue does this).
Use the original arm names so the checkpoint format, the extract and the train-log check (p5) stay exactly the ones of card ADD43.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("TOKPROBE_GR_SRC") or os.path.join(HERE, "..", "finegrid_r", "tok_probe_gr.py")
DST = os.path.join(HERE, "tok_probe_lr.py")
EDITS = []


def edit(old, new):
    EDITS.append((old, new))


edit('"""tok_probe_gr.py -- made from tok_probe_g.py',
     '"""tok_probe_lr.py -- made from tok_probe_gr.py by make_tok_probe_lr.py (card PREREG_V8F_PROOF_ADD46): the clips with held-out value divisible by 10 are NOT in the training windows. Original text follows.\n'
     'tok_probe_gr.py -- made from tok_probe_g.py')

edit('_TREE = {}\n',
     '_TREE = {}\n\n\n'
     'def heldout_value(clip_id):\n'
     '    """card ADD46: integer value of the first 8 hex digits of sha256(clip id); a clip is held out when it is divisible by 10."""\n'
     '    return int(hashlib.sha256(str(clip_id).encode("utf-8")).hexdigest()[:8], 16)\n\n\n'
     'def is_heldout(clip_id):\n'
     '    return heldout_value(clip_id) % 10 == 0\n')

edit('    store, J, wins, clips, info = TR.build_data(a, cfg)\n',
     '    store, J, wins, clips, info = TR.build_data(a, cfg)\n'
     '    held = sorted(c for c in clips if is_heldout(c))\n'
     '    clips = [c for c in clips if not is_heldout(c)]\n'
     '    wins = D.train_windows(store, J, clips, hist=cfg.hist, start=cfg.window_start)        # the trainer\'s own call, on the clips that are not held out\n'
     '    if not held or not clips or any(is_heldout(c) for c in clips):\n'
     '        raise SystemExit("REFUSED: held-out split: %d held out, %d trained" % (len(held), len(clips)))\n'
     '    info["n_clips"], info["n_windows"] = len(clips), len(wins)\n'
     '    info["heldout"] = {"rule": "int(sha256(clip id)[:8], 16) % 10 == 0", "n_heldout": len(held), "n_trained": len(clips), "heldout_sha256": hashlib.sha256(chr(10).join(held).encode("utf-8")).hexdigest()}\n')

edit('"grid": args.grid, "n_tok": GRIDS[args.grid], "host_est": info["host_est"],',
     '"heldout": info["heldout"], "grid": args.grid, "n_tok": GRIDS[args.grid], "host_est": info["host_est"],')


def build_text(src=SRC):
    text = open(src, encoding="utf-8", newline="").read()
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            raise SystemExit("STOP: old text occurs %d times (need 1): %r" % (n, old[:70]))
        text = text.replace(old, new)
    return text


def main():
    text = build_text()
    with open(DST, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote", DST, len(EDITS), "edits")


if __name__ == "__main__":
    main()
