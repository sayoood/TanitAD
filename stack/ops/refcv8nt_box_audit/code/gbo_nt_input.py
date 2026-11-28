#!/usr/bin/env python3
"""gbo_nt_input.py -- the INPUT ADAPTER that lets the FROZEN G-BOX-OVERFIT selector run on the NT (navtrain) TRAIN corpus.

The selector (``…/2026-09-26-box-head-audit/code/gbo_select.py``, md5 f9f93f92b60a979b581b8f78be2ec6db, + ``vis_rules.py``
md5 99cd007c0fd4e539ca38fa299a948fbf) is run UNCHANGED on this adapter's output. This file only builds the selector's
INPUT (a ``vis_train_boxes.json``-shaped list of per-GT-row dicts) for the NT launch, the way the audit built it for
refcv8 -- and it builds it from LABELS ONLY.

Where the refcv8 input came from (all MEASURED in the audit package):
  * the POOL is ``bha_dump.py --set train``: the train manifest's clip ids sorted by sha12, every k-th
    (k = len // 64), the first 64 (``bha_dump.py:195-199``), then per clip ``--per-clip 4`` windows evenly spaced over
    the clip's AGENT-LABELLED windows (join lookup at NOW = t + W - 1 is not None; ``bha_dump.py:76-79, 244-252``);
  * the rows are every GT row of those windows with a 3-D label, ``target`` = inside the trainer's target filter, and
    ``vis_frac`` / ``n_vis`` from the exact z-buffer of the GT cuboids in the clip's own camera (``vis_zbuf.py``).

What this adapter does for NT (the SAME rule; the only differences are forced by the launch configuration):
  * POOL: ``bha_dump``'s rule VERBATIM on the NT launch's ``--v2-cache`` manifest (``pool_rule`` / ``evenly`` below are
    copies of ``bha_dump.py:76-79, 197-199``). The NT launch carries ``--window-starts first`` -- ONE window per
    token-episode, t = 0 (``refc_v3_train.restrict_window_starts``) -- so the only window of the launch's TRAIN
    dataset per clip is t = 0, and ``evenly(1, 4) = [0]``: the pool is <= 64 windows (refcv8: 256).
  * ROWS: the HARNESS's OWN data path -- ``g_box_overfit.TrainerAdapter`` (data only: its ``__init__`` builds the
    TRAIN dataset through the trainer's ``V3Dataset._agent_item`` with the launch argv and joins the launch's VIS-1
    sidecar; NO model is built, NO checkpoint is opened) -- and ``vis1.vis1_split`` on each window's target block.
    ``target`` = the trainer's ``visible_target_filter``; ``vis_frac = n_vis / n_full`` (float64, the sidecar's
    integer pixel counts from the audit's own z-buffer, ``vis_zbuf.py:39-144``, run on the NT stitch camera's
    observed mask: the sidecar's stamp names it); an UNKNOWN visibility row reads ``vis_frac`` NaN, which the frozen
    ``vis_rules.label`` calls DROPPED and the harness (A10) counts as IGNORE -- the same row on both sides.
  * The held-out guard is re-run on the pool: every pool clip's log is checked against ``--nt-heldout`` (pinned
    sha256) through ``--nt-token-table``, and every token against the selection-scan sample. The pool is drawn from
    the launch's ``--v2-cache`` only, so no eval139 clip (a separate ``--eval-cache``) can enter it.

PRE-COMMITTED (written before any NT count was read): if the frozen rule on this pool yields fewer than 16 frames, the
result is reported as INFEASIBLE and NO rule parameter (pool size, per-clip, eligibility band) is changed.

MEMORY SCOPE (this process only): the loader indexes the 3-D join over EVERY clip of the cache's clip table
(``_clip_table_for_caches`` -> ``open_join3d(clips=<all 99,400>)``), which was OOM-killed at an 8 GB cap (MEASURED
2026-10-22, ``8.0G memory peak``). ``_scope_clip_table`` narrows that table to the clips being read; the rows of
those clips are the same rows either way (the join is a per-(clip, frame) lookup), so the labels this adapter emits
are the ones the full harness reads. And ``TrainerAdapter`` holds every window's decoded item, so the pool is read in
CHUNKS of 16 windows (the harness's own frame-set size): a 64-window adapter peaked past 7 GB. The real binding run
is NOT scoped.

``--check-frameset`` (the DRY check): the HARNESS's ``TrainerAdapter`` on the written frame set -- its manifest
lookup, its provider filter, its ``(sha12, t)`` window lookup and its VIS-1 recount -- then ``control_c3`` against the
profile's literals (``g_box_overfit.FRAMESETS["refcv8-nt"]``). Nothing trains; no model is built.

Writes ``--out`` (the selector input; sha12 only, no raw clip id) and ``--record`` (provenance: the pool rule and its
sha12 list, the guard census, per-window counts, the sidecar stamp, and every file this process OPENED, from a
``sys.addaudithook`` hook -- the evidence that no model output was read).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

POOL_CLIPS, PER_CLIP = 64, 4          # bha_dump.py --set train defaults (--train-clips 64, --per-clip 4)

_OPENED: list = []


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if p not in _OPENED:
            _OPENED.append(p)


def sha12(s) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


def opened_files() -> list:
    """The files this process opened, outside the interpreter; a ``<clip id>.v2ep.pt`` is written as its sha12
    (no raw clip id in any written file)."""
    out = []
    for p in _OPENED:
        if p.startswith(sys.prefix) or p.startswith(sys.base_prefix) or "/site-packages/" in p or p.endswith(".pyc"):
            continue
        if p.endswith(".v2ep.pt"):
            d, b = os.path.split(p)
            p = os.path.join(d, f"sha12:{sha12(b[: -len('.v2ep.pt')])}.v2ep.pt")
        out.append(p)
    return out


def evenly(n: int, k: int) -> list:
    """``bha_dump.py:76-79`` verbatim."""
    if n <= k:
        return list(range(n))
    return sorted({int(round((j + 0.5) * n / k - 0.5)) for j in range(k)})


def pool_rule(clip_ids, n_clips: int = POOL_CLIPS) -> tuple:
    """``bha_dump.py:197-199`` verbatim: clip ids sorted by sha12, every ``len // n_clips``-th, the first ``n_clips``."""
    cids = sorted((str(c) for c in clip_ids), key=sha12)
    step = max(1, len(cids) // int(n_clips))
    return cids[::step][: int(n_clips)], step, len(cids)


def _scope_clip_table(L, keep: set) -> None:
    """Narrow the LOADER's trainer ``_clip_table_for_caches`` to ``keep`` (raw clip ids): memory only, see above."""
    ltr = L.trainer()
    if not hasattr(ltr, "_gbo_nt_orig_clip_table"):
        ltr._gbo_nt_orig_clip_table = ltr._clip_table_for_caches
    orig = ltr._gbo_nt_orig_clip_table

    def scoped(cache_dirs, *aa, **kk):
        table, n_stack = orig(cache_dirs, *aa, **kk)
        return {e: c for e, c in table.items() if str(c) in keep}, n_stack
    ltr._clip_table_for_caches = scoped


def rows_of_window(it: dict, s12: str, t: int, V, classes) -> tuple:
    """-> (selector rows, vis1_split counts) for ONE window, from the harness's own target block."""
    import torch
    b = torch.utils.data.default_collate([it])
    tgt = {"box": b["agent_box"], "valid": b["agent_valid"]}
    sp = V.vis1_split(tgt, n_full=b["agent_vis_full"], n_vis=b["agent_vis_px"], vis_known=b["agent_vis_known"])
    valid = b["agent_valid"][0].to(torch.bool)
    inf, known = sp["in_filter"][0], b["agent_vis_known"][0].to(torch.bool)
    nf, nv, cls, box = b["agent_vis_full"][0], b["agent_vis_px"][0], b["agent_cls"][0], b["agent_box"][0]
    rows = []
    for j in range(int(valid.shape[0])):
        if not bool(valid[j]):
            continue
        f, v, c = int(nf[j]), int(nv[j]), int(cls[j])
        rows.append({"sha12": s12, "w": int(t), "cls": classes[c] if 0 <= c < len(classes) else "unknown",
                     "range": math.hypot(float(box[j, 0]), float(box[j, 1])), "target": bool(inf[j]),
                     "known": bool(known[j]), "n_full": f, "n_vis": v,
                     "vis_frac": (float(v) / float(f)) if (bool(known[j]) and f > 0) else float("nan")})
    return rows, dict(sp["counts"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", required=True, help="the launch tree (holds stack/scripts/g_box_overfit.py)")
    ap.add_argument("--launch-argv", required=True, help="stack/ops/runs.d/<run>.argv.json of the NT launch")
    ap.add_argument("--out", required=True, help="the selector input (vis_train_boxes-shaped JSON list)")
    ap.add_argument("--record", required=True, help="the provenance record (JSON)")
    ap.add_argument("--check-frameset", default=None, help="DRY: the harness's data path on this frame set + C3")
    a = ap.parse_args(argv)
    sys.addaudithook(_audit)
    t0 = time.time()
    scripts = Path(a.tree).resolve() / "stack" / "scripts"
    for p in (str(scripts.parent), str(scripts)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    import g_box_overfit as G
    tr = G._load_by_path("refc_v3_train_for_gbo", G.SCRIPTS / "refc_v3_train.py")
    L, loader_rec = G.load_vendored_loader()
    launch_argv, argv_rec = G.load_launch_argv(a.launch_argv)
    build_argv, _tc = G.harness_build_argv(launch_argv)
    args = tr.build_parser().parse_args(build_argv)
    from tanitad.data import v2_dataset as v2d
    from tanitad.data import vis1 as V
    from tanitad.refs.refc_agents import AGENT_CLASSES
    from tanitad.train import nt_guard as NG
    if getattr(args, "window_starts", "all") != "first":
        raise SystemExit("[gbo-nt] the launch argv does not carry --window-starts first: this adapter's pool is "
                         "derived for one window per token-episode")
    cache = args.v2_cache[0] if isinstance(args.v2_cache, (list, tuple)) else args.v2_cache
    man = v2d.load_or_build_manifest(cache, verbose=False)
    pool, step, n_man = pool_rule(man["clip_id"])
    guard = NG.load_guard(args.nt_heldout, expect_sha256=args.nt_heldout_sha256)
    table = NG.load_token_table(args.nt_token_table)
    miss_tt = [sha12(c) for c in pool if c not in table]
    if miss_tt:
        raise SystemExit(f"[gbo-nt] {len(miss_tt)} pool clips are not in the token table: {miss_tt[:4]}")
    census = NG.assert_disjoint([table[c]["log_name"] for c in pool], guard, where="G-BOX-OVERFIT NT pool")
    tok_hits = sorted({str(table[c]["token"]) for c in pool} & set(guard.get("selection_sample_tokens") or ()))
    if tok_hits:
        raise SystemExit(f"[gbo-nt] {len(tok_hits)} pool tokens are selection-scan sample tokens")
    del table
    if a.check_frameset:
        return check_frameset(a, G, tr, L, build_argv, man, argv_rec, census)
    # ---- the HARNESS's own data path: TrainerAdapter.__init__ = parse + _build_items (data only, no model) ---- #
    import gc
    rows, per_window, ad = [], [], None
    for c0 in range(0, len(pool), 16):
        chunk = pool[c0:c0 + 16]
        _scope_clip_table(L, set(chunk))
        pseudo = {"frames": [{"sha12": sha12(c), "t": 0} for c in chunk]}
        ad = G.TrainerAdapter(tr, L, build_argv, pseudo, "cpu")
        for (s12, t, n_pos, n_ign), it in zip(ad.per_frame, ad.items):
            labelled = bool(it["agent_label"])
            rw, cnt = rows_of_window(it, s12, t, V, AGENT_CLASSES)
            tg = [r for r in rw if r["target"]]
            per_window.append({"sha12": s12, "t": int(t), "agent_label": labelled, "n_real": len(rw),
                               "n_target": len(tg), "harness_pos": int(n_pos), "harness_ign": int(n_ign),
                               "split_counts": cnt})
            if labelled:                                 # bha_dump: only AGENT-LABELLED windows enter the pool
                rows += rw
        if c0 + 16 < len(pool):
            ad.items = None
            ad = None
            gc.collect()
    Path(a.out).write_text(json.dumps(rows), encoding="utf-8")
    out_md5 = hashlib.md5(Path(a.out).read_bytes()).hexdigest()
    rec = {"tool": "gbo_nt_input.py", "started_s": round(t0, 1), "wall_s": round(time.time() - t0, 1),
           "launch_argv": argv_rec, "loader": loader_rec, "tree": str(Path(a.tree).resolve()),
           "cache": cache, "pool_rule": {"rule": f"bha_dump.py --set train: manifest clip ids sorted by sha12, every "
                                                 f"{step}-th, first {POOL_CLIPS}; per clip evenly({PER_CLIP}) over the "
                                                 f"AGENT-LABELLED windows of the launch's TRAIN dataset "
                                                 f"(--window-starts first: t = 0 only)",
                                         "n_manifest": n_man, "step": step, "n_pool": len(pool),
                                         "pool_sha12": [sha12(c) for c in pool]},
           "heldout_guard": census, "selection_token_hits": 0,
           "sidecar": ad.sidecar_stamp, "vis1_stats": ad.vis1_stats,
           "dataset_record": {k: v for k, v in (ad.dataset_record or {}).items() if k != "label_clock_g3"},
           "per_window": per_window, "n_labelled_windows": sum(1 for w in per_window if w["agent_label"]),
           "n_rows": len(rows), "n_target_rows": sum(1 for r in rows if r["target"]),
           "out": a.out, "out_md5": out_md5,
           "model_built": False,
           "opened_files": opened_files()}
    Path(a.record).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(f"ZZGBONT-INPUT n_pool={len(pool)} labelled={rec['n_labelled_windows']} rows={len(rows)} "
          f"targets={rec['n_target_rows']} md5={out_md5}ZZ", flush=True)
    return 0


def check_frameset(a, G, tr, L, build_argv, man, argv_rec, census) -> int:
    """The DRY check: every frame-set clip in the cache manifest, the harness's own ``_build_items`` finds all 16
    windows, and C3 (per-frame POS, IGN + DROP; totals; md5) reads PASS against the registered NT literals."""
    fs_bytes = Path(a.check_frameset).read_bytes()
    fs = json.loads(fs_bytes.decode("utf-8"))
    by12 = {sha12(c): str(c) for c in man["clip_id"]}
    in_man = [f["sha12"] in by12 for f in fs["frames"]]
    _scope_clip_table(L, {by12[f["sha12"]] for f in fs["frames"] if f["sha12"] in by12})
    ad = G.TrainerAdapter(tr, L, build_argv, fs, "cpu")
    lit = G.FRAMESETS["refcv8-nt"]
    c3 = G.control_c3(ad.per_frame, fs, hashlib.md5(fs_bytes).hexdigest(), lit)
    rec = {"tool": "gbo_nt_input.py --check-frameset", "frameset": a.check_frameset,
           "frameset_md5": hashlib.md5(fs_bytes).hexdigest(), "launch_argv": argv_rec, "heldout_guard": census,
           "n_frames": len(fs["frames"]), "n_in_manifest": int(sum(in_man)), "n_items_built": len(ad.items),
           "per_frame": ad.per_frame, "literals": lit, "C3": c3, "sidecar": ad.sidecar_stamp,
           "dataset_record": {k: v for k, v in (ad.dataset_record or {}).items() if k != "label_clock_g3"},
           "model_built": False,
           "opened_files": opened_files()}
    Path(a.record).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(f"ZZGBONT-DRY in_manifest={rec['n_in_manifest']}/{rec['n_frames']} items={len(ad.items)} "
          f"C3={'PASS' if c3['pass'] else 'FAIL'} totals={c3['totals']} want={c3['want_totals']}ZZ", flush=True)
    return 0 if c3["pass"] and rec["n_in_manifest"] == rec["n_frames"] == len(ad.items) else 1


if __name__ == "__main__":
    sys.exit(main())
