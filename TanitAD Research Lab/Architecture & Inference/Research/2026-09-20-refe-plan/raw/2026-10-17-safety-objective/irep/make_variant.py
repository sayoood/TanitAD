#!/usr/bin/env python3
"""make_variant.py: build `fp_forward_irep.py` (the I-REP variant of the navtest forward) from the ORIGINAL `navtestE/fp/fp_forward.py` by exact-string patches.

    python make_variant.py            -> writes fp_forward_irep.py and fp_forward_irep.diff next to this file
    python make_variant.py --check    -> regenerates in memory and exits 1 unless it equals the shipped fp_forward_irep.py byte for byte

WHY A PATCH SCRIPT AND NOT A HAND-EDITED COPY: the claim "with both options at their defaults the variant is byte-identical to the original" is only as good as the claim that the
variant IS the original plus these hunks.  Each hunk below is anchored on an `old` string that must occur EXACTLY ONCE in the original (a changed original makes this script refuse,
it does not guess), the original is pinned by md5 (`fp_COPIES.md5`, the free-point worker's own file), and the shipped variant is re-derived from it by `test_irep.py`.

WHAT THE VARIANT ADDS (PREREG_SAFEOBJ section 7, "inference": the I-rep of PREREG_RL_S1 section 3, eval batch 8 vs 32, shuffled order, non-deterministic kernels):
  --eval-batch N          run the trunk + the scorers on N tokens at a time (default 1 = the planner's own batch 1)
  --token-order-seed S    process the tokens in the order of a seeded permutation (default: none, i.e. the sorted order)
Every output row is written back to the token's OWN position, so the npz is in the order the original writes it.  At the defaults NOTHING in the original path runs differently: the
only edit on that path is one `if irep_active(a)` that is False.  `meta_json` gains an `irep` record only when a run used a non-default option.
"""
from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "navtestE" / "fp" / "fp_forward.py"
ORIGINAL_MD5 = "82bf21514f919ee4eafab72690004e34"
OUT = HERE / "fp_forward_irep.py"
DIFF = HERE / "fp_forward_irep.diff"

# ------------------------------------------------------------------------------------------------ the hunks (name, old, new); `old` occurs exactly once in the original (LF-normalised)
H_GLOBALS = (
    "globals",
    "EXPECT_UPDATE = 1000\n",
    '''EXPECT_UPDATE = 1000
IREP_VARIANT_OF = "navtestE/fp/fp_forward.py md5 82bf21514f919ee4eafab72690004e34 (irep/make_variant.py)"
IREP_RUNS: dict = {}          # n tokens -> the record of the I-rep run that produced them; copied into meta_json as `irep` ONLY when a run used a non-default option
''',
)

H_FUNCS = (
    "irep_functions_and_dispatch",
    "def run_tokens(a, model, cfg, states, tokens: list, inputs: dict, calib, zc: ZipCache, log=print):\n    import torch\n    dev = a.device\n",
    '''def irep_active(a) -> bool:
    """True iff this run asks for anything but the planner's own batch 1 in sorted order"""
    return int(getattr(a, "eval_batch", 1)) != 1 or getattr(a, "token_order_seed", None) is not None


def run_tokens_irep(a, model, cfg, states, tokens: list, inputs: dict, calib, zc: ZipCache, log=print):
    """THE I-REP PATH (PREREG_SAFEOBJ section 7): the model call of `run_tokens`, on `--eval-batch` tokens at a time, visiting the tokens in the order of a seeded permutation
    (`--token-order-seed`).  Every output row goes back to the token's OWN position (`traj_o[idx] = ...`), so the returned arrays are in the order of `tokens`, exactly as
    `run_tokens` returns them.  fp32, no autocast, and NO determinism flag is set anywhere in this file: the kernels are the library's own, and they depend on the batch shape.
    IC2 (the replicated BASE scorer path against the model's own score) is checked PER TOKEN, as in the original."""
    import torch
    dev = a.device
    taps = Taps(model)
    n = len(tokens)
    M, C = cfg.n_proposals, cfg.n_score_components
    names = ["BASE"] + list(states)
    traj_o = np.zeros((n, M, 20, 3), np.float32)
    logits_o = {nm: np.zeros((n, M, C), np.float32) for nm in names}
    ic2 = np.zeros(n, np.float64)
    bs = max(1, int(a.eval_batch))
    seed = a.token_order_seed
    if seed is not None and not 0 <= int(seed) < 2 ** 32:
        raise SystemExit(f"REFUSED: --token-order-seed {seed} is outside [0, 2**32)")
    order = np.arange(n, dtype=np.int64) if seed is None else np.random.RandomState(int(seed)).permutation(n).astype(np.int64)    # RandomState: the frozen legacy stream
    sizes: dict = {}
    t0 = time.time()
    done, ic2_max = 0, 0.0
    for s in range(0, n, bs):
        idx = [int(j) for j in order[s:s + bs]]                          # positions in `tokens` of the rows of this batch, in processing order
        imgs, egos, goals, cals = [], [], [], []
        for j in idx:
            r = inputs[tokens[j]]
            img = build_img(r["frames"], zc, cfg)
            if a.mutate_swap_cams and j < 8:                              # D3, a DELIBERATE REGRESSION: the same first 8 tokens whatever the order
                img = img.clone()
                img[:, [1, 2]] = img[:, [2, 1]]
            imgs.append(img)
            egos.append(r["ego"])
            goals.append(r["goal"])
            cals.append(torch.tensor(calib[r["log"]], dtype=torch.float64))   # [1, n_cam, 16] per token
        img_b = torch.cat(imgs, dim=0)                                    # [b, n_cam, 3, H, W]
        ego = torch.tensor(egos, dtype=torch.float32, device=dev)
        goal = torch.tensor(goals, dtype=torch.float32).to(dev)
        cal = torch.cat(cals, dim=0)                                      # [b, n_cam, 16], float64, CPU: as the planner passes it
        taps.xflat = taps.sctx = None
        with torch.no_grad():
            traj, score = model(img_b.to(dev), ego, goal, calib=cal)
            if taps.xflat is None or taps.sctx is None:
                raise RuntimeError("the scoring decoder never ran: the taps saw nothing")
            base_mods = {m: getattr(model, m) for m in SCORER}
            rep = score_with(base_mods, taps.xflat, taps.sctx)
            d = (rep - score).abs().flatten(1).max(dim=1).values.double().cpu().numpy()    # IC2 per token: must be EXACTLY 0.0
            outs = {"BASE": score}
            for nm, mods in states.items():
                outs[nm] = score_with(mods, taps.xflat, taps.sctx)
        ix = np.asarray(idx, dtype=np.int64)
        ic2[ix] = d
        ic2_max = max(ic2_max, float(d.max()))
        traj_o[ix] = traj.float().cpu().numpy()                           # <- the restore: each row to its token's own position
        for nm in names:
            logits_o[nm][ix] = outs[nm].float().cpu().numpy()
        sizes[len(idx)] = sizes.get(len(idx), 0) + 1
        prev, done = done, done + len(idx)
        if done // 25 != prev // 25 or done == n:
            log(f"  {done}/{n} tokens (I-rep batch {bs})  {(time.time() - t0) / done:.2f} s/token  ic2 max so far {ic2_max:.3g}", flush=True)
    taps.h1.remove()
    taps.h2.remove()
    IREP_RUNS[n] = {"eval_batch": bs, "token_order_seed": (None if seed is None else int(seed)), "n": n, "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
                    "batch_size_counts": {str(k): v for k, v in sorted(sizes.items())}, "variant_of": IREP_VARIANT_OF}
    return traj_o, logits_o, ic2, time.time() - t0


def run_tokens(a, model, cfg, states, tokens: list, inputs: dict, calib, zc: ZipCache, log=print):
    if irep_active(a):
        return run_tokens_irep(a, model, cfg, states, tokens, inputs, calib, zc, log=log)
    import torch
    dev = a.device
''',
)

H_REPEAT = (
    "repeat_check_skips_in_irep_runs",
    "    n = min(n, len(tokens))\n    t2, l2, _ic2, _s = run_tokens(",
    '''    if irep_active(a):          # IC5 re-runs the first n tokens, which in an I-rep run would be a DIFFERENT batch composition: it would measure the very thing this run varies
        return {"n": 0, "skipped": "I-rep run: a repeat would re-batch the tokens", "bit_identical": None}
    n = min(n, len(tokens))
    t2, l2, _ic2, _s = run_tokens(''',
)

H_WRITE = (
    "write_out_records_irep",
    "    info = dict(info, seconds=round(secs, 1), n=len(tokens))\n",
    '''    info = dict(info, seconds=round(secs, 1), n=len(tokens))
    if len(tokens) in IREP_RUNS:
        info = dict(info, irep=IREP_RUNS[len(tokens)])
''',
)

H_ARGS = (
    "argparse_two_options",
    '    ap.add_argument("--mutate-swap-cams", action="store_true")\n',
    '''    ap.add_argument("--mutate-swap-cams", action="store_true")
    ap.add_argument("--eval-batch", type=int, default=1, help="I-rep: tokens per forward (default 1 = the planner's own batch 1)")
    ap.add_argument("--token-order-seed", type=int, default=None, help="I-rep: visit the tokens in the order of np.random.RandomState(S).permutation(n) (default: sorted order)")
''',
)

HUNKS = [H_GLOBALS, H_FUNCS, H_REPEAT, H_WRITE, H_ARGS]


def read_original() -> str:
    raw = ORIGINAL.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != ORIGINAL_MD5:
        raise SystemExit(f"REFUSED: {ORIGINAL} md5 {md5} is not the pinned {ORIGINAL_MD5}")
    txt = raw.decode("utf-8")
    if txt.count("\r\n") != txt.count("\n"):
        raise SystemExit("REFUSED: the original is not uniformly CRLF")
    return txt.replace("\r\n", "\n")


def apply(lf: str, hunks=HUNKS) -> str:
    for name, old, new in hunks:
        if lf.count(old) != 1:
            raise SystemExit(f"REFUSED: hunk {name!r}: the anchor occurs {lf.count(old)} times in the original, not exactly once")
        lf = lf.replace(old, new)
    return lf


def build() -> tuple[str, str]:
    lf = read_original()
    out = apply(lf)
    diff = "".join(difflib.unified_diff(lf.splitlines(True), out.splitlines(True), "a/fp_forward.py", "b/fp_forward_irep.py", n=2))
    return out.replace("\n", "\r\n"), diff


def main() -> int:
    out, diff = build()
    if "--check" in sys.argv:
        shipped = OUT.read_bytes().decode("utf-8") if OUT.exists() else None
        ok = shipped == out
        print(f"{'ZZVARIANT_CHECK_OK' if ok else 'ZZVARIANT_CHECK_FAILED'} {OUT.name} sha256 {hashlib.sha256(out.encode('utf-8')).hexdigest()}")
        return 0 if ok else 1
    OUT.write_bytes(out.encode("utf-8"))
    DIFF.write_text(diff, encoding="utf-8", newline="\n")
    added = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    print(f"wrote {OUT.name} (sha256 {hashlib.sha256(out.encode('utf-8')).hexdigest()}) and {DIFF.name}: {added} lines added, {removed} removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
