"""Scratch-only: visible progress, crash-safe partial saves and a settable batch for SCORER1 Stage 0's dump.
Everything is behind --dump-step0-only.  Never lands into refe/ from here; the diff is kept beside the MM note."""
import difflib
F = "F:/tanitad_scratch/stage0_20261108/code/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe/scorer_finetune.py"
b = open(F, "rb").read()
orig = b
eol = b"\r\n" if b"\r\n" in b[:4000] else b"\n"
def L(*lines):
    return eol.join(lines)
def sub(old, new):
    global b
    assert b.count(old) == 1, (old[:70], b.count(old))
    b = b.replace(old, new)

A = b'    ap.add_argument("--dump-step0-only", action="store_true",'
sub(A, b'    ap.add_argument("--dump-batch", type=int, default=16, help="--dump-step0-only: held-out batch size of the dump (evaluate() uses 16 otherwise)")' + eol + A)

A = L(b'            out[arm] = path', b'        return out', b'')
sub(A, A + L(
    b'    def save_partial(self, path):',
    b'        """Scratch Stage 0: a crash-safe snapshot of the rows so far (written beside, then renamed over)."""',
    b'        tmp = path + ".tmp.npz"',
    b'        np.savez(tmp, log=np.asarray([k[0] for k in self.keys]), token=np.asarray([k[1] for k in self.keys]),',
    b'                 step=np.asarray([k[2] for k in self.keys], np.int64), rank=np.asarray([k[3] for k in self.keys], np.int64),',
    b'                 ckpt_step=np.asarray(self.ckpts, np.int64), **{"scores_" + arm: np.concatenate(self.chunks[arm], 0) for arm in self.ARMS})',
    b'        os.replace(tmp, path)',
    b'',
    b'    def load_partial(self, path):',
    b'        """-> rows restored; the inverse of save_partial."""',
    b'        with np.load(path, allow_pickle=False) as z:',
    b'            self.keys = [(str(l), str(t), int(s), int(r)) for l, t, s, r in zip(z["log"], z["token"], z["step"], z["rank"])]',
    b'            self.ckpts = [int(c) for c in z["ckpt_step"]]',
    b'            self.chunks = {arm: [z["scores_" + arm].astype(np.float64)] for arm in self.ARMS}',
    b'        return len(self.keys)',
    b'', b''))

A = L(b'        idx = ho_idx[:limit] if limit else ho_idx',
      b'        dlh = DataLoader(Subset(ds_ho, idx), batch_size=16, shuffle=False, num_workers=2)')
sub(A, L(
    b'        idx = ho_idx[:limit] if limit else ho_idx',
    b'        _s0 = bool(a.dump_step0_only and a.dump_scores)                      # Stage 0 (scratch): scores only, progress lines, partial saves, resume',
    b'        _part = os.path.join(a.dump_scores, "partial_step0.npz") if _s0 else None',
    b'        _pre, _n0, _nb, _t0 = None, 0, 0, time.time()',
    b'        if _part and os.path.exists(_part):',
    b'            _pre = ScoreDump(); _n0 = _pre.load_partial(_part)',
    b'            if [key(ds.rows[i]) for i in idx[:_n0]] != _pre.keys:',
    b'                raise RuntimeError("partial dump does not match the held-out row order: " + _part)',
    b'            log({"event": "dump_resume", "rows_restored": _n0})',
    b'        _n_tot = len(idx); idx = idx[_n0:]',
    b'        dlh = DataLoader(Subset(ds_ho, idx), batch_size=(a.dump_batch if _s0 else 16), shuffle=False, num_workers=2)'))

sub(b'        sdump = ScoreDump() if a.dump_scores else None', b'        sdump = _pre if _pre is not None else (ScoreDump() if a.dump_scores else None)')

A = L(b'                sdump.add(bkeys, [ho.by[k_][0] for k_ in bkeys], sx0, sxA, sxB)', b'')
sub(A, A + L(
    b'            if _s0:',
    b'                j += tg.shape[0]; _nb += 1',
    b'                if _nb == 1 or _nb % 5 == 0:',
    b'                    log({"event": "dump_progress", "done": len(sdump.keys), "of": _n_tot, "elapsed_s": round(time.time() - _t0, 1),',
    b'                         "vram_alloc_mib": int(torch.cuda.memory_allocated() / 2**20), "vram_reserved_mib": int(torch.cuda.memory_reserved() / 2**20)})',
    b'                if _nb % 25 == 0:',
    b'                    os.makedirs(a.dump_scores, exist_ok=True); sdump.save_partial(_part)',
    b'                continue',
    b''))

A = b'        out = {"event": "eval", "tag": tag, "n": len(toks)}'
sub(A, L(
    b'        if _s0:',
    b'            log({"event": "score_dump", "tag": tag, "n_sets": len(sdump.keys), "files": sdump.write(a.dump_scores, tag)})',
    b'            return {"event": "eval", "tag": tag, "n": len(sdump.keys), "dump_only": True}',
    A))

open(F, "wb").write(b)
d = difflib.unified_diff(orig.decode("utf-8").splitlines(), b.decode("utf-8").splitlines(), "a/scorer_finetune.py", "b/scorer_finetune.py", lineterm="", n=1)
open("E:/Projects/TanitAD-artifacts/scorer1_impl_20261108/stage0_pull/dump_resumable.diff", "w", encoding="utf-8", newline="\n").write("\n".join(d) + "\n")

P = "F:/tanitad_scratch/stage0_20261108/common_args.py"
s = open(P, "rb").read()
old = b'"--dump-step0-only"]'
assert s.count(old) == 1
open(P, "wb").write(s.replace(old, b'"--dump-step0-only", "--dump-batch", "8"]'))
print("ZZPATCH_OK")
