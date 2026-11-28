"""make_tok_probe_gr.py -- writes tok_probe_gr.py from ../finegrid/tok_probe_g.py by asserted text edits (card PREREG_V8F_PROOF_ADD43). The original is never edited.
Every old text must occur exactly once in the original, else the generator stops. Run: python make_tok_probe_gr.py  (writes tok_probe_gr.py next to this file).

Changes (all logging, one lever, one check form; the default path is bit-identical to tok_probe_g.py, control g1):
  E-r1   control r1: the window indices of the first 200 batches are hashed per batch and written to <out>/r1_batches.json (logging only), plus a subcommand `r1` that compares two such files.
  E-chk  lever --sampler-chunk-like GRID: the sampler's chunk (clips per pool) is the one the trainer would take for that grid. Without it the chunk follows the run's own grid (the default).
         Why: the trainer sizes its clip store by the token count (lru 174 at 40 tokens, 43 at 160 on the pilot's data: 105 frames at most), so the sampler chunk is 48 at 4 by 10 and 41 at 8 by 20,
         and the two processes would NOT see the same windows in the same order (control r1 would fail by construction).
  E-p5   check_log: for an arm with permuted closing labels (SH in the name) p5 is read on the intact parts of the lead term (presence + gap); the permuted part is reported beside.
         The parts are already in the train log of tok_probe_g.py (lead_sup_bce_mean, lead_sup_gap_mean, lead_sup_clos_mean): no change to the log format.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "finegrid", "tok_probe_g.py")
DST = os.path.join(HERE, "tok_probe_gr.py")

EDITS = []


def edit(old, new):
    EDITS.append((old, new))


edit('"""tok_probe_g.py -- TOKPROBE with a grid lever, the probe tool of the frozen card PREREG_V8F_PROOF_ADD41 (FINEGRID; definitions: FINEGRID_DEFS.md).',
     '"""tok_probe_gr.py -- made from tok_probe_g.py by make_tok_probe_gr.py (card PREREG_V8F_PROOF_ADD43, FINEGRID-R; definitions: FINEGRID_R_DEFS.md): adds the r1 batch log, the lever\n'
     '--sampler-chunk-like and the p5 form for permuted-label arms; the default path is the one of tok_probe_g.py (control g1).\n'
     'TOKPROBE with a grid lever, the probe tool of the frozen card PREREG_V8F_PROOF_ADD41 (FINEGRID; definitions: FINEGRID_DEFS.md).')

edit('SHUFFLE_RNG_SEED = 12345\n',
     'SHUFFLE_RNG_SEED = 12345\nR1_BATCHES = 200                                            # control r1: the window indices of the first 200 batches are hashed\n')

edit('def trainer_argv(text, out, grid=GRID_DEFAULT):\n',
     '''def trainer_lru(nmax, n_tok):
    """The trainer's clip-store size when --lru is 0 (v8f_mini_train.py:2736): max(8, 1.5e9 // (longest clip * tokens * 1024 * 2))."""
    return max(8, int(1.5e9 // (int(nmax) * int(n_tok) * 1024 * 2)))


def sampler_chunk(lru):
    """The trainer's sampler chunk (v8f_mini_train.py:2851): clips per pool, at most 48, at least 4, two below the clip-store size."""
    return max(4, min(48, int(lru) - 2))


def batch_hash(batch):
    """Hash of the window indices of one batch: the list of (clip, t, age_t, age_s) in order."""
    return hashlib.sha256(json.dumps([[int(x) for x in w] for w in batch]).encode()).hexdigest()[:16]


class R1Log:
    """Control r1 (card ADD43 section 4): records a hash per batch, in order, for the first ``n`` batches the sampler yields (logging only; the batches pass through unchanged)
    and writes <out>/r1_batches.json. ``idx`` hashes (clip, t) only, ``full`` hashes the whole item including the two age draws."""

    def __init__(self, path, n, meta=None):
        self.path, self.n, self.meta, self.idx, self.full, self.done = path, int(n), dict(meta or {}), [], [], False

    def watch(self, batches):
        for b in batches:
            if len(self.idx) < self.n:
                self.idx.append(batch_hash([w[:2] for w in b]))
                self.full.append(batch_hash(b))
                if len(self.idx) == self.n:
                    self.flush()
            yield b

    def flush(self):
        if self.done:
            return
        self.done = True
        rec = {"n_batches": len(self.idx), "idx_hashes": self.idx, "full_hashes": self.full, "sequence_sha256": hashlib.sha256("".join(self.full).encode()).hexdigest(), **self.meta}
        with open(self.path, "w", encoding="utf-8", newline="\\n") as fh:
            json.dump(rec, fh, indent=1)


def r1_compare(path_a, path_b):
    """Control r1: the two batch logs hold the same sequence of window indices (both the (clip, t) hashes and the full-item hashes), over at least one batch."""
    a, b = json.load(open(path_a, encoding="utf-8")), json.load(open(path_b, encoding="utf-8"))
    same_idx = a["idx_hashes"] == b["idx_hashes"]
    same_full = a["full_hashes"] == b["full_hashes"]
    mism = next((i for i, (x, y) in enumerate(zip(a["idx_hashes"], b["idx_hashes"])) if x != y), None)
    return {"pass": bool(same_idx and same_full and len(a["idx_hashes"]) > 0 and a["n_batches"] == b["n_batches"]), "n_a": a["n_batches"], "n_b": b["n_batches"], "idx_equal": same_idx, "full_equal": same_full,
            "first_mismatch_batch": mism, "sampler_chunk_a": a.get("sampler_chunk"), "sampler_chunk_b": b.get("sampler_chunk"), "sequence_sha256_a": a.get("sequence_sha256"), "sequence_sha256_b": b.get("sequence_sha256"),
            "files": [path_a, path_b]}


def cmd_r1(args):
    res = r1_compare(os.path.join(args.a, "r1_batches.json"), os.path.join(args.b, "r1_batches.json"))
    res["card"] = "PREREG_V8F_PROOF_ADD43 control r1"
    jdump(res, args.out)
    print("r1: pass %s (%d batches, chunks %s / %s) -> %s" % (res["pass"], res["n_a"], res["sampler_chunk_a"], res["sampler_chunk_b"], args.out))
    return 0 if res["pass"] else 2


def trainer_argv(text, out, grid=GRID_DEFAULT):
''')

edit('    sampler = D.ChunkedSampler(wins, a.batch, seed=a.seed, chunk=max(4, min(48, info["lru"] - 2)), max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)\n',
     '''    like = args.sampler_chunk_like
    if like and like not in GRIDS:
        raise SystemExit("REFUSED: --sampler-chunk-like %r is not one of %s" % (like, sorted(GRIDS)))
    lru_like = (a.lru or trainer_lru(info["host_est"]["clip_frames_max"], GRIDS[like])) if like else info["lru"]
    chunk = sampler_chunk(lru_like)                                            # the default (no flag) is the original expression max(4, min(48, lru - 2))
    sampler = D.ChunkedSampler(wins, a.batch, seed=a.seed, chunk=chunk, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    r1 = R1Log(os.path.join(args.out, "r1_batches.json"), min(R1_BATCHES, total), {"grid": args.grid, "seed": a.seed, "batch": a.batch, "sampler_chunk": chunk, "sampler_chunk_like": like or None,
                                                                              "lru_trainer": info["lru"], "lru_used_for_chunk": lru_like, "total_steps": total})  # R1-LOG
''')

edit('            "grid": args.grid, "n_tok": GRIDS[args.grid], "host_est": info["host_est"]}\n',
     '            "grid": args.grid, "n_tok": GRIDS[args.grid], "host_est": info["host_est"], "sampler_chunk": chunk, "sampler_chunk_like": like or None, "lru_trainer": info["lru"]}\n')

edit('            pf = TR.Prefetch(sampler.epoch(ep), lambda idx: D.collate([ds.item(*w) for w in idx]), depth=3)\n',
     '            pf = TR.Prefetch(r1.watch(sampler.epoch(ep)), lambda idx: D.collate([ds.item(*w) for w in idx]), depth=3)\n')

edit('    res = {n: check_log(os.path.join(args.out, n)) for n in names}\n',
     '    r1.flush()\n    res = {n: check_log(os.path.join(args.out, n)) for n in names}\n')

edit('"p5": res, "p5_all_pass": all(r["pass"] for r in res.values()), "input_norm": args.input_norm,',
     '"p5": res, "p5_all_pass": all(r["pass"] for r in res.values()), "input_norm": args.input_norm, "sampler_chunk": chunk, "r1_sequence_sha256": hashlib.sha256("".join(r1.full).encode()).hexdigest(),')

edit('    out["pass_lead_term_falls"] = bool(first and last and out["last_1000_mean"] < out["first_1000_mean"])\n',
     '''    out["pass_lead_term_falls"] = bool(first and last and out["last_1000_mean"] < out["first_1000_mean"])
    shuffled = bool(re.match(r"^(TK|F)\\d+SH_s\\d+$", str(hdr["arm"])))
    if all(k in r for r in body for k in ("lead_sup_bce_mean", "lead_sup_gap_mean", "lead_sup_clos_mean")):  # P5-PARTS: the parts of the lead term are in the log of tok_probe_g.py
        def part(keys, lo_hi):
            sel = [sum(r[k] for k in keys) for r in body if lo_hi(r["step"])]
            return float(np.mean(sel)) if sel else None
        fi, la = part(("lead_sup_bce_mean", "lead_sup_gap_mean"), lambda s: s <= window), part(("lead_sup_bce_mean", "lead_sup_gap_mean"), lambda s: s > total - window)
        out["intact_parts_first_mean"], out["intact_parts_last_mean"] = fi, la
        out["permuted_part_clos_first_mean"], out["permuted_part_clos_last_mean"] = part(("lead_sup_clos_mean",), lambda s: s <= window), part(("lead_sup_clos_mean",), lambda s: s > total - window)
        out["pass_intact_parts_fall"] = bool(fi is not None and la is not None and la < fi)
    else:
        out["pass_intact_parts_fall"] = None
    out["shuffled_closing_arm"] = shuffled
''')

edit('    out["pass"] = bool(out["pass_lead_term_falls"] and out["pass_gap_labels_every_step"] and out["pass_complete"])\n',
     '''    falls = out["pass_intact_parts_fall"] if shuffled else out["pass_lead_term_falls"]          # P5-FORM: an arm with permuted closing labels is read on its intact parts
    out["pass"] = bool(falls and out["pass_gap_labels_every_step"] and out["pass_complete"])
''')

edit('    t.add_argument("--threads", type=int, default=0)\n',
     '    t.add_argument("--threads", type=int, default=0)\n    t.add_argument("--sampler-chunk-like", default="", help="take the sampler chunk the trainer would take for this grid (control r1); default: the run\'s own grid")\n')

edit('    a = ap.parse_args(argv)\n    if a.cmd == "g2":\n        return cmd_g2(a)\n',
     '''    r = sub.add_parser("r1")
    r.add_argument("--a", required=True)
    r.add_argument("--b", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "r1":
        return cmd_r1(a)
    if a.cmd == "g2":
        return cmd_g2(a)
''')


def main():
    txt = open(SRC, encoding="utf-8", newline="").read()
    assert "\r" not in txt
    for i, (old, new) in enumerate(EDITS):
        n = txt.count(old)
        if n != 1:
            raise SystemExit("EDIT %d: old text occurs %d times (needs exactly 1): %r" % (i, n, old[:80]))
        txt = txt.replace(old, new)
    with open(DST, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print("wrote", DST, len(EDITS), "edits")


if __name__ == "__main__":
    main()
