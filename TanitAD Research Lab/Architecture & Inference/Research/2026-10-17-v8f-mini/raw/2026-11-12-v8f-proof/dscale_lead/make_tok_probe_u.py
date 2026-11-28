"""make_tok_probe_u.py -- writes tok_probe_u.py from ../finegrid_r/tok_probe_gr.py by asserted text edits (card PREREG_V8F_PROOF_ADD45 DSCALE-LEAD with note N1).
The original is never edited. Every old text must occur exactly once in the original, else the generator stops.  Run: python make_tok_probe_u.py   (pure text work: no numpy, no torch)
Style of the precedent integ10k/make_tok_probe_lr.py.

New levers, all default-off (without them the tool is the source tool, text for text on every path the source runs; the test `default_path_bit_identical` checks it on a pod):
  train --clip-list FILE       keep only the training clips whose id is in FILE (windows rebuilt with the trainer's own D.train_windows); needs --steps (matched compute)
  train --eval-guard FILE      refuse to train when a clip id of FILE is among the clips the trainer built (control d1)
  train --w-clos 1.0|0.0       0.0: the closing label of every batch is set to NaN, so the closing mask of lead_sup_terms is empty and its part is exactly 0.0 with exactly 0.0 gradient (N1)
  train --shuffle-gap-labels   permute the finite gap labels among the frames that have one (fixed rng); presence intact; excludes --shuffle-closing-labels; needs --w-clos 0.0 (N1 control arms)
  precheck ...                 delegates to dscale_precheck.py (numpy only; the kit calls that file directly so that torch is not imported)
check_log (control p5 / d4): an arm trained with --shuffle-gap-labels is read on the presence part of the lead term only.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("TOKPROBE_GR_SRC") or os.path.join(HERE, "..", "finegrid_r", "tok_probe_gr.py")
DST = os.path.join(HERE, "tok_probe_u.py")
EDITS = []


def edit(old, new):
    EDITS.append((old, new))


edit('"""tok_probe_gr.py -- made from tok_probe_g.py',
     '"""tok_probe_u.py -- made from tok_probe_gr.py by make_tok_probe_u.py (card PREREG_V8F_PROOF_ADD45 + N1): --clip-list, --eval-guard, --w-clos, --shuffle-gap-labels, precheck. Original text follows.\n'
     'tok_probe_gr.py -- made from tok_probe_g.py')

HELPERS = '''_TREE = {}
_OPTS = {"clip_list": None, "eval_guard": None, "w_clos": 1.0}     # DSCALE-LEAD levers; the defaults are the source tool's behaviour


def read_id_list(path):
    """One clip id per line; blank lines and lines starting with # are skipped; duplicates are refused. Returns (ids in file order, sha256 of the file bytes)."""
    ids = [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip() and not ln.lstrip().startswith("#")]
    if not ids or len(set(ids)) != len(ids):
        raise SystemExit("REFUSED: id list %s holds %d ids, %d distinct" % (path, len(ids), len(set(ids))))
    return ids, sha256_file(path)


def set_opts(args):
    """Read the DSCALE-LEAD levers of ``train`` into _OPTS and refuse the combinations the card does not allow."""
    w = float(args.w_clos)
    if w not in (0.0, 1.0):
        raise SystemExit("REFUSED: --w-clos %r: only 1.0 (the source loss) or 0.0 (closing part exactly absent) are built" % w)
    if args.shuffle_gap_labels and args.shuffle_closing_labels:
        raise SystemExit("REFUSED: --shuffle-gap-labels and --shuffle-closing-labels are mutually exclusive")
    if args.shuffle_gap_labels and w != 0.0:
        raise SystemExit("REFUSED: --shuffle-gap-labels is the control of note N1 and needs --w-clos 0.0")
    if args.clip_list and not args.steps:
        raise SystemExit("REFUSED: --clip-list changes the window count, so the step count must be given (--steps N): matched compute")
    _OPTS["w_clos"] = w
    _OPTS["clip_list"] = None
    _OPTS["eval_guard"] = None
    if args.clip_list:
        ids, sha = read_id_list(args.clip_list)
        _OPTS["clip_list"] = {"path": os.path.abspath(args.clip_list), "ids": ids, "sha256": sha}
    if args.eval_guard:
        ids, sha = read_id_list(args.eval_guard)
        _OPTS["eval_guard"] = {"path": os.path.abspath(args.eval_guard), "ids": ids, "sha256": sha}
'''
edit('_TREE = {}\n', HELPERS)

FILTER = '''    store, J, wins, clips, info = TR.build_data(a, cfg)
    guard = _OPTS.get("eval_guard")
    if guard is not None:                                                         # CONTROL-D1: no evaluation clip among the clips the trainer built
        shared = sorted(set(str(c) for c in clips) & set(guard["ids"]))
        info["eval_guard"] = {"path": guard["path"], "sha256": guard["sha256"], "n_eval_ids": len(guard["ids"]), "n_shared": len(shared)}
        if shared:
            raise SystemExit("REFUSED: %d evaluation clip ids are among the training clips (first: %s)" % (len(shared), shared[0]))
    cl = _OPTS.get("clip_list")
    if cl is not None:                                                            # CLIP-FILTER
        have = set(str(c) for c in clips)
        missing = sorted(i for i in cl["ids"] if i not in have)
        if missing:
            raise SystemExit("REFUSED: %d listed clip ids are not among the built clips (first: %s)" % (len(missing), missing[0]))
        keep = set(cl["ids"])
        clips = [c for c in clips if str(c) in keep]
        wins = D.train_windows(store, J, clips, hist=cfg.hist, start=cfg.window_start)    # the trainer's own call, on the listed clips
        if not clips or not wins:
            raise SystemExit("REFUSED: clip list: %d clips, %d windows left" % (len(clips), len(wins)))
        info["n_clips"], info["n_windows"] = len(clips), len(wins)
        info["train_sha256"] = hashlib.sha256(chr(10).join(sorted(clips, key=D.sha12)).encode("utf-8")).hexdigest()     # the trainer's own digest of the training clips (v8f_mini_train.py build_data)
        info["clip_list"] = {"path": cl["path"], "sha256": cl["sha256"], "n_listed": len(cl["ids"]), "n_kept": len(clips)}
'''
edit('    store, J, wins, clips, info = TR.build_data(a, cfg)\n', FILTER)

edit('    T = M.lead_sup_terms(out, batch, hist)\n',
     '    if _OPTS.get("w_clos", 1.0) == 0.0:                                       # W-CLOS-ZERO: no closing label -> empty mask -> that part is exactly 0.0 with exactly 0.0 gradient (N1)\n'
     '        batch = dict(batch)\n'
     '        batch["lead_clos"] = torch.full_like(batch["lead_clos"], float("nan"))\n'
     '    T = M.lead_sup_terms(out, batch, hist)\n')

edit('def trainer_lru(nmax, n_tok):',
     'def shuffle_gap(ds, seed=SHUFFLE_RNG_SEED):\n'
     '    """Control of note N1: permute the finite gap labels among the rows that have one (presence labels and the gap mask, hence the counts, stay)."""\n'
     '    LS = ds.lead_sup_labels\n'
     '    g = np.array(LS["gap"], copy=True)\n'
     '    idx = np.flatnonzero(np.isfinite(g))\n'
     '    g[idx] = g[idx][np.random.default_rng(seed).permutation(len(idx))]\n'
     '    LS["gap"] = g\n'
     '    return {"n_shuffled": int(len(idx)), "rng_seed": seed}\n\n\n'
     'def trainer_lru(nmax, n_tok):')

edit('    T = use_tree(args.tree)\n', '    T = use_tree(args.tree)\n    set_opts(args)\n')

edit('    shuf = shuffle_closing(ds) if args.shuffle_closing_labels else None\n',
     '    shuf = shuffle_closing(ds) if args.shuffle_closing_labels else None\n'
     '    shuf_gap = shuffle_gap(ds) if args.shuffle_gap_labels else None\n')

edit('"shuffle_closing": shuf, "trainer_argv": argv,',
     '"shuffle_closing": shuf, "shuffle_gap": shuf_gap, "w_clos": _OPTS["w_clos"], "clip_list": info.get("clip_list"), "eval_guard": info.get("eval_guard"), "trainer_argv": argv,')

edit('    shuffled = bool(re.match(r"^(TK|F)\\d+SH_s\\d+$", str(hdr["arm"])))\n',
     '    shuffled = bool(re.match(r"^(TK|F)\\d+SH_s\\d+$", str(hdr["arm"]))) or bool(hdr.get("shuffle_gap"))\n'
     '    parts_keys = ("lead_sup_bce_mean",) if hdr.get("shuffle_gap") else ("lead_sup_bce_mean", "lead_sup_gap_mean")      # PRESENCE-ONLY-FORM: a gap-permuted arm is read on its presence part\n')

edit('        fi, la = part(("lead_sup_bce_mean", "lead_sup_gap_mean"), lambda s: s <= window), part(("lead_sup_bce_mean", "lead_sup_gap_mean"), lambda s: s > total - window)\n',
     '        fi, la = part(parts_keys, lambda s: s <= window), part(parts_keys, lambda s: s > total - window)\n')

edit('    t.add_argument("--sampler-chunk-like", default="", help="take the sampler chunk the trainer would take for this grid (control r1); default: the run\'s own grid")\n',
     '    t.add_argument("--sampler-chunk-like", default="", help="take the sampler chunk the trainer would take for this grid (control r1); default: the run\'s own grid")\n'
     '    t.add_argument("--clip-list", default="", help="DSCALE-LEAD: file of clip ids, one per line; only these training clips are used (needs --steps)")\n'
     '    t.add_argument("--eval-guard", default="", help="DSCALE-LEAD control d1: file of evaluation clip ids; refuse to train if one is among the built training clips")\n'
     '    t.add_argument("--w-clos", type=float, default=1.0, help="DSCALE-LEAD note N1: 1.0 = the source loss, 0.0 = closing part exactly absent")\n'
     '    t.add_argument("--shuffle-gap-labels", action="store_true", help="DSCALE-LEAD note N1 control: gap labels permuted, presence intact (needs --w-clos 0.0)")\n')

edit('    r.add_argument("--out", required=True)\n    a = ap.parse_args(argv)\n',
     '    r.add_argument("--out", required=True)\n'
     '    pc = sub.add_parser("precheck", help="numpy-only checks d1, d2, counts, input hashes (dscale_precheck.py)")\n'
     '    pc.add_argument("rest", nargs=argparse.REMAINDER)\n'
     '    a = ap.parse_args(argv)\n'
     '    if a.cmd == "precheck":\n'
     '        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n'
     '        import dscale_precheck\n'
     '        return dscale_precheck.main(a.rest)\n')


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
