#!/usr/bin/env python3
"""FOUND WHILE BUILDING P1 - `readout()`'s R1/R2 were computed on a B x B OUTER PRODUCT.

MEASURED 2026-09-05 (this WP, `raw/probe_ctx_rank.json`), reproduced in three lines:

    fan2 [B, N, 5, 2]  x  reward_ctx()'s lead_path [B, 1, 1, 5, 2]
      -> _collision / _headway return [B, B, N]        (not [B, N])
    fan2 [B, N, 5, 2]  x  reward_ctx()'s v0 [B, 1, 1]
      -> _progress returns [B, B, N]                   (not [B, N])

`reward_ctx` is shaped for the TRAINING trajectory `[B, N, G, S, 2]` (three leading
axes). `readout()` hands it a `[B, N, S, 2]` fan (two leading axes), so every scene
fact broadcasts against the WRONG axis and each window's fan is scored against every
window's lead. The driver then indexes `r1[j]` / `coll[j]`, i.e. window j's row of a
B x B slab, and means over it -- so the banked `R1` and `R2` readouts are a batch-mixed
statistic, not "the composed reward on this window's fan".

WHAT IS AND IS NOT AFFECTED -- scoped, not blanket-voided:
  * AFFECTED: `readout()`'s `R1` and `R2` only (and their paired deltas), for any
    readout batch > 1. The default is `batch=4`.
  * NOT affected: TRAINING. `make_sample_fn` hands `traj2 [B, N, G, 5, 2]` -- three
    leading axes -- which is exactly what `reward_ctx` builds for. Verified by shape.
  * NOT affected: the PRIMARY fan-safety endpoint. `FS.score_paths` is called with
    `lead5 = b["lead_track"].reshape(-1, 1, 5, 2)` and `v0 = b["v0"]` ([B]), which
    broadcast correctly against `[B, N, ...]`; `score_paths` unsqueezes v0 itself.

THE FIX: `reward_ctx` takes the candidate rank explicitly (`cand_dims`), and `readout`
asserts the returned reward's shape against the fan's -- a positive assertion, so a
future rank mismatch fails instead of silently averaging the wrong windows.
Same family as the CLAUDE.md units trap: a correct quantity computed in the wrong
SCOPE reads exactly like an answer.
"""
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
CR, LF = chr(13), chr(10)
CRLF = CR + LF
NE, ST, WARN = chr(9940), chr(11088), chr(9888)


def rw(path, old, new, *, count=1):
    p = os.path.join(REPO, path)
    raw = open(p, "rb").read()
    is_crlf = CRLF.encode() in raw
    s = raw.decode("utf-8").replace(CRLF, LF)
    if new in s and old not in s:
        print("  [skip] %s: already patched" % path)
        return
    n = s.count(old)
    if n != count:
        raise SystemExit("[patch] %s: expected %d occurrence(s), found %d" % (path, count, n))
    s = s.replace(old, new)
    open(p, "wb").write((s.replace(LF, CRLF) if is_crlf else s).encode("utf-8"))
    print("  [ok]   %s (%s)" % (path, "CRLF" if is_crlf else "LF"))


OLD = '''def reward_ctx(batch: dict, *, S5: int, extras: dict | None = None) -> dict:
    """{NE} THE ONLY READER OF SCENE FACTS FOR THE REWARD. Reads v0 (t0) and the lead's
    first sample. Nothing here touches a future_* field — proved by the preflight."""
    B = batch["v0"].shape[0]
    lead = batch["lead_xy"]                                             # [B, 2]
    ctx = {"dt": DT_REWARD_S,
           "v0": batch["v0"].reshape(B, 1, 1),                          # [B, 1, 1]
           "lead_len_m": LEAD_LEN_DEFAULT_M}
    if LEAD_MODE == "track":
        if S5 != len(GRID_S):
            raise ValueError(f"track mode scores the {len(GRID_S)}-point prefix, got S5={S5}")
        # MOVING lead, time-aligned; NO static `obstacles` key, so `collision`
        # takes rewards._collision's lead_path branch (per-step contact).
        ctx["lead_path"] = batch["lead_track"].reshape(B, 1, 1, S5, 2)
    else:
        ctx["obstacles"] = lead.reshape(B, 1, 1, 1, 2)                   # [B,1,1,K=1,2]
        ctx["lead_path"] = lead.reshape(B, 1, 1, 1, 2).expand(B, 1, 1, S5, 2)  # static
    if extras:
        ctx.update(extras)
    return ctx
'''.replace("{NE}", NE)

NEW = '''def reward_ctx(batch: dict, *, S5: int, extras: dict | None = None,
               cand_dims: int = 2) -> dict:
    """{NE} THE ONLY READER OF SCENE FACTS FOR THE REWARD. Reads v0 (t0) and the lead's
    first sample. Nothing here touches a future_* field — proved by the preflight.

    ``cand_dims`` is the number of CANDIDATE axes in the trajectory this context will
    be scored against: **2** for the training tensor ``[B, N, G, S, 2]`` (the default,
    unchanged) and **1** for a readout fan ``[B, N, S, 2]``.

    {NE}{NE} WHY THIS ARGUMENT EXISTS AND IS NOT COSMETIC. Every scene fact here is
    reshaped to broadcast against the candidate axes. Hand a ``[B, N, S, 2]`` fan a
    context built for ``[B, N, G, S, 2]`` and the leading ``B`` no longer lines up:
    ``lead_path [B,1,1,S,2]`` against ``traj [B,N,S,2]`` right-aligns to
    ``[B, B, N, S, 2]``, so **every window's fan is scored against every window's
    lead**, and ``v0 [B,1,1]`` against ``along [B,N]`` does the same. The result has
    the right dtype, no error, and one more axis than anybody looks at — MEASURED in
    this repo: ``readout()``'s ``R1``/``R2`` were a B x B outer product for the whole
    2026-09-05 RL panel (readout batch = 4). Training was never affected; its tensor
    really does have three leading axes.
    {W} A caller that gets this wrong gets a NUMBER, not an exception. That is why
    ``readout`` now asserts the returned reward's shape against the fan's.
    """
    B = batch["v0"].shape[0]
    if cand_dims < 1:
        raise ValueError(f"cand_dims must be >= 1, got {cand_dims}")
    ones = (1,) * cand_dims
    lead = batch["lead_xy"]                                             # [B, 2]
    ctx = {"dt": DT_REWARD_S,
           "v0": batch["v0"].reshape(B, *ones),                         # [B, 1(, 1)]
           "lead_len_m": LEAD_LEN_DEFAULT_M}
    if LEAD_MODE == "track":
        if S5 != len(GRID_S):
            raise ValueError(f"track mode scores the {len(GRID_S)}-point prefix, got S5={S5}")
        # MOVING lead, time-aligned; NO static `obstacles` key, so `collision`
        # takes rewards._collision's lead_path branch (per-step contact).
        ctx["lead_path"] = batch["lead_track"].reshape(B, *ones, S5, 2)
    else:
        ctx["obstacles"] = lead.reshape(B, *ones, 1, 2)                  # [B,1(,1),K=1,2]
        ctx["lead_path"] = lead.reshape(B, *ones, 1, 2).expand(B, *ones, S5, 2)  # static
    if extras:
        ctx.update(extras)
    return ctx
'''.replace("{NE}", NE).replace("{W}", WARN)

rw("stack/scripts/rl_refcv3_min.py", OLD, NEW)

rw("stack/scripts/rl_refcv3_min.py",
   '''        fan2 = with_origin(fan[..., :N_REWARD_SLOTS, :])                # [B, N, 5, 2]
        ctx = reward_ctx(b, S5=fan2.shape[-2])
        r1 = spec(fan2, ctx)                                            # [B, N]
        coll = RW.COMPONENTS["collision"](fan2, ctx) < 0                # [B, N]
''',
   '''        fan2 = with_origin(fan[..., :N_REWARD_SLOTS, :])                # [B, N, 5, 2]
        # {ST} cand_dims=1: this is a [B, N, S, 2] FAN, not the [B, N, G, S, 2]
        # training tensor. With the training rank the scene facts broadcast to a
        # B x B outer product and every window is scored against every window's
        # lead — MEASURED, it is what the 2026-09-05 panel's R1/R2 actually were.
        ctx = reward_ctx(b, S5=fan2.shape[-2], cand_dims=1)
        r1 = spec(fan2, ctx)                                            # [B, N]
        coll = RW.COMPONENTS["collision"](fan2, ctx) < 0                # [B, N]
        # {NE} POSITIVE ASSERTION, not a comment. A rank mismatch here returns a
        # NUMBER rather than raising, so the shape is checked rather than assumed.
        if tuple(r1.shape) != tuple(fan.shape[:2]) or tuple(coll.shape) != tuple(fan.shape[:2]):
            raise RuntimeError(
                f"reward rank mismatch: r1 {tuple(r1.shape)} / collision "
                f"{tuple(coll.shape)} vs fan {tuple(fan.shape[:2])} — the reward "
                "context was built for a different candidate rank and the readout "
                "would be a batch-mixed outer product")
'''.replace("{ST}", ST).replace("{NE}", NE))

print("[patch] P1a readout ctx rank applied")
