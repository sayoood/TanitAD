"""navfeed_overlay.py -- OPT-IN per-frame nav feed for the refcv8 eval roll (EvalFlyWheel 2026-10-20).  OVERLAY ONLY.

THE DEFECT (NAV_TRACE.md, MEASURED): refcv8 was TRAINED with the per-frame v9 nav token (``refc_v3_train.py:4665-4668``, reader
``refcv8_train.py:895-898``) and ``r8_nav`` [B,6] = ``scale_nav(nav_args, known)`` (``refcv8_train.py:367-378``, ``known`` dropped with
p 0.5 while TRAINING only, ``refcv8_train.py:560-567``).  The eval roll feeds the CLIP-LEVEL v7 token instead
(``taniteval/tools/refcv3_arm.py:2064-2071``) and never passes ``r8_nav`` (``refcv3_arm.py:2358``, ``r7_roll.py:152-163``).

WHAT THIS MODULE DOES.  It touches NO file in the launch tree (``C:/lgs/Lfb47a58376``) or the live battery package.  It builds a PATCHED
COPY of ``refcv3_arm.py`` IN MEMORY by exact-anchor source edits (each anchor must occur EXACTLY once, else it refuses -- the
``r8_battery_patch.patched_source`` convention), executes it with the ORIGINAL ``__file__`` (so every sibling load by path still resolves),
and registers it where ``r7_roll.ra3()`` caches it.  ``--nav-feed`` is the only new flag:

    clip            (DEFAULT) today's behaviour, byte-identical: no patched statement executes; manifest and sidecar gain no key.
    perframe_token  ``nav_true[i]`` = the item's per-frame v9 token (``R8_V9_NAV_TO_NAV_CMD[r8_nav_token]``), i.e. what
                    ``V3Dataset.__getitem__`` puts in ``item["nav_cmd"]`` when ``r8_nav_from_v9`` is on (``refc_v3_train.py:4665-4668``).
    perframe_full   perframe_token AND ``r8_nav = scale_nav(item raw args, item known)`` on the ``nav_true`` row, exactly as the
                    trainer builds it (``refcv8_train.py:572-575``: ``fwd["r8_nav"] = scale_nav(nav, known)``).

The other nav conditions (shuffled / flipped / zero) are DERIVED FROM THE SAME ``nav_true`` vector (``ra.shuffle_nav(nav_true, ...)``
and the flip, both of which read ``nav_true`` AFTER the patched assignment), so they stay paired with the per-frame token.
``r8_nav`` rows of a perturbed condition (shuffled / flipped) are the UNKNOWN row (zeros): the args describe the TRUE maneuver, and
feeding them next to a perturbed token would leak the truth through the args and build a (token, args) pair the model never saw.
The nav-zero call (``nav_cmd=None``) carries no ``r8_nav`` (None == the unknown row, ``refc_v3.py:1968-1973``).
"""
from __future__ import annotations

import difflib
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

NAV_FEEDS = ("clip", "perframe_token", "perframe_full")
PERFRAME = ("perframe_token", "perframe_full")
R8_NAV_DIMS = 6
SPEC = "PREREG_NAV_FEED.md"


# =============================================================================================================== pure helpers
def perframe_nav(ds, sel, mode, nav_on, *, nav_map, scale_nav):
    """The per-frame nav of every selected window, WITHOUT decoding a frame.

    Mirrors the dataset item path, line for line:
      * ``refc_v3_train.py:4657-4658``  ``_j8 = self.r8_join.item(int(ep.episode_id), int(t) + int(w) - 1 + self._raw_offset(ep),
                                                                    self._now_s(ep, t), ...)``
      * ``refc_v3_train.py:4665-4668``  ``item["nav_cmd"] = R8_V9_NAV_TO_NAV_CMD[int(item.pop("r8_nav_token"))]``; ``nav_valid = True``
      * ``refcv8_train.py:895-898``     the ``r8_nav_raw`` (v9 ``nav_args`` [5]) / ``r8_nav_known`` (``nav_args_valid``) the item carries.
    ``sel`` is refcv3_arm's list of ``(window_index, episode_index, t)``.  Returns ``(nav_true int64[n], nav_valid bool[n], pf)``;
    ``pf`` carries ``raw`` [n,5], ``known`` [n], ``fed_row`` [n,6] (= ``scale_nav(raw, known)``, the row ``perframe_full`` feeds) and
    ``token`` [n].  REFUSES (SystemExit) rather than falling back to the clip token."""
    if mode not in PERFRAME:
        raise SystemExit(f"[navfeed] nav_feed {mode!r} is not a per-frame mode {PERFRAME}")
    if not nav_on:
        raise SystemExit(f"[navfeed] --nav-feed {mode} needs a nav source: with nav_source 'none' the forward already passes "
                         f"nav_cmd=None and there is no token to replace")
    join = getattr(ds, "r8_join", None)
    if join is None:
        raise SystemExit(f"[navfeed] --nav-feed {mode}: the eval dataset has no v9 join (ds.r8_join is None) -- refusing to fall "
                         f"back to the clip token")
    if not bool(getattr(ds, "r8_nav_from_v9", False)):
        raise SystemExit(f"[navfeed] --nav-feed {mode}: ds.r8_nav_from_v9 is off, so the dataset items carry the CLIP token, not the "
                         f"per-frame one the run trained on (refc_v3_train.py:4666) -- refusing")
    w = int(ds.window)
    n = len(sel)
    token = np.zeros(n, dtype=np.int64)
    raw = np.zeros((n, 5), dtype=np.float32)
    known = np.zeros(n, dtype=bool)
    for i, (_wi, e_i, t) in enumerate(sel):
        ep = ds.episodes[e_i]
        j8 = join.item(int(ep.episode_id), int(t) + w - 1 + int(ds._raw_offset(ep)), float(ds._now_s(ep, t)))
        tok8 = int(j8["r8_nav_token"])
        if tok8 not in nav_map:
            raise SystemExit(f"[navfeed] v9 nav token {tok8} is not in the trainer's map {sorted(nav_map)}")
        token[i] = int(nav_map[tok8])
        raw[i] = np.asarray(j8["r8_nav_raw"], dtype=np.float32).reshape(5)
        known[i] = bool(j8["r8_nav_known"])
    import torch
    fed_row = scale_nav(torch.from_numpy(raw), torch.from_numpy(known)).numpy().astype(np.float32)
    if not np.isfinite(fed_row).all():
        raise SystemExit("[navfeed] scale_nav produced a non-finite r8_nav row (a NaN in the v9 args)")
    if not np.all(fed_row[~known] == 0.0):
        raise SystemExit("[navfeed] an UNKNOWN row (known=0) is not exactly zeros -- scale_nav's contract "
                         "(refcv8_train.py:367-378) is broken or the release carries NaN args")
    return token.copy(), np.ones(n, dtype=bool), {"mode": mode, "token": token, "raw": raw, "known": known, "fed_row": fed_row}


def assert_item_matches(item, i, nav_true, pf):
    """Per-window POSITIVE assertion, run at the moment the item is actually built: the dataset's own item must carry the token and
    the raw args the pre-pass derived (a drift between the two is the failure this experiment cannot afford to read through)."""
    tok = int(item["nav_cmd"])
    if tok != int(nav_true[i]):
        raise SystemExit(f"[navfeed] window {i}: item nav_cmd {tok} != pre-pass per-frame token {int(nav_true[i])}")
    r = np.asarray(item["r8_nav_raw"], dtype=np.float32).reshape(5)
    if not np.array_equal(r, pf["raw"][i]):
        raise SystemExit(f"[navfeed] window {i}: item r8_nav_raw {r.tolist()} != pre-pass {pf['raw'][i].tolist()}")
    if bool(item["r8_nav_known"]) != bool(pf["known"][i]):
        raise SystemExit(f"[navfeed] window {i}: item r8_nav_known {bool(item['r8_nav_known'])} != pre-pass {bool(pf['known'][i])}")


def model_kw(mode, pf, i, conds_fed, dev):
    """Extra keyword arguments for the BATCHED fed forward of window ``i`` (row r == ``conds_fed[r]``).  ``{}`` unless perframe_full."""
    if mode != "perframe_full":
        return {}
    import torch
    rows = torch.zeros(len(conds_fed), R8_NAV_DIMS, dtype=torch.float32)
    rows[list(conds_fed).index("nav_true")] = torch.from_numpy(pf["fed_row"][i])
    return {"r8_nav": rows.to(dev)}


def manifest_record(mode, pf, nav_clip, nav_true):
    return {"mode": mode, "spec": SPEC,
            "n_windows": int(len(nav_true)),
            "n_token_differs_from_clip": int((np.asarray(nav_true) != np.asarray(nav_clip)).sum()),
            "n_args_known": int(pf["known"].sum()),
            "token_source": "v9 per-frame nav_token via ds.r8_join.item (refc_v3_train.py:4657-4668, refcv8_train.py:895-898)",
            "args_fed": mode == "perframe_full",
            "args_rule": ("r8_nav = refcv8_train.scale_nav(item r8_nav_raw, item r8_nav_known) on the nav_true row; unknown "
                          "(known=0) -> exactly zeros; shuffled / flipped rows carry the unknown row; nav-zero call carries none"),
            "other_conditions": "nav_shuffled / nav_flipped are derived from the SAME per-frame nav_true vector"}


# =============================================================================================================== source patches
def _p(pid, desc, anchor, replacement):
    return {"id": pid, "desc": desc, "anchor": anchor, "replacement": replacement}


#: exact-anchor edits of taniteval/tools/refcv3_arm.py (launch tree fb47a58).  Each anchor must occur exactly once.
PATCHES = [
    _p("P1-argparse",
       "new flag --nav-feed (default clip)",
       '    ap.add_argument("--with-navpred", action="store_true",\n',
       '    ap.add_argument("--nav-feed", choices=("clip", "perframe_token", "perframe_full"), default="clip",\n'
       '                    help="which nav the true-nav forward is fed. clip (DEFAULT, byte-identical to the banked battery): the '
       'clip-level v7 token. perframe_token: the v9 per-frame token the run TRAINED on. perframe_full: that token AND r8_nav = '
       'scale_nav(args, known) exactly as the trainer builds it. See PREREG_NAV_FEED.md.")\n'
       '    ap.add_argument("--with-navpred", action="store_true",\n'),
    _p("P2-nav_true",
       "per-frame token (and args) replace the clip token BEFORE the shuffle/flip derive from nav_true",
       '    nav_shuf, shuf_stats = ra.shuffle_nav(nav_true, nav_valid, a.nav_shuffle_seed)\n',
       '    nav_feed = str(getattr(a, "nav_feed", "clip") or "clip")\n'
       '    nav_clip = nav_true.copy()\n'
       '    pf_args = None\n'
       '    if nav_feed != "clip":\n'
       '        import navfeed_overlay as _navfeed\n'
       '        nav_true, nav_valid, pf_args = _navfeed.perframe_nav(\n'
       '            ds, sel, nav_feed, nav_on, nav_map=tr.R8_V9_NAV_TO_NAV_CMD, scale_nav=tr.r8train.scale_nav)\n'
       '        _p(f"[nav-feed] {nav_feed}: nav_true = the v9 PER-FRAME token on {len(nav_true)} windows; it differs from the clip "\n'
       '           f"token on {int((nav_true != nav_clip).sum())}; r8_nav args known on {int(pf_args[\'known\'].sum())} "\n'
       '           f"({\'FED\' if nav_feed == \'perframe_full\' else \'NOT fed\'})")\n'
       '    nav_shuf, shuf_stats = ra.shuffle_nav(nav_true, nav_valid, a.nav_shuffle_seed)\n'),
    _p("P3-item-assert",
       "positive per-window check that the dataset item agrees with the pre-pass",
       '            item = ds[wi]\n            fv = item["future_valid_ext"]\n',
       '            item = ds[wi]\n'
       '            if pf_args is not None:\n'
       '                _navfeed.assert_item_matches(item, i, nav_true, pf_args)\n'
       '            fv = item["future_valid_ext"]\n'),
    _p("P4-model-call",
       "r8_nav rides the batched fed forward (perframe_full only)",
       '                out = model(fr_b, nav_cmd=nav_t if nav_on else None,\n'
       '                            v0=_v0_fed, steps=steps, ego_state=ego_b,\n'
       '                            **_geom(int(fr_b.shape[0])))\n',
       '                out = model(fr_b, nav_cmd=nav_t if nav_on else None,\n'
       '                            v0=_v0_fed, steps=steps, ego_state=ego_b,\n'
       '                            **_geom(int(fr_b.shape[0])),\n'
       '                            **({} if pf_args is None else\n'
       '                               _navfeed.model_kw(nav_feed, pf_args, i, conds_fed, dev)))\n'),
    _p("P5-sidecar",
       "decisions sidecar: clip token and the fed args beside the (per-frame) nav_cmd",
       '            dec.setdefault("nav_cmd", []).append(int(nav_true[i]))\n',
       '            dec.setdefault("nav_cmd", []).append(int(nav_true[i]))\n'
       '            if pf_args is not None:\n'
       '                dec.setdefault("nav_cmd_clip", []).append(int(nav_clip[i]))\n'
       '                dec.setdefault("r8_nav_known", []).append(bool(pf_args["known"][i]))\n'
       '                dec.setdefault("r8_nav_fed", []).append(\n'
       '                    pf_args["fed_row"][i:i + 1] if nav_feed == "perframe_full"\n'
       '                    else np.zeros((1, 6), dtype=np.float32))\n'),
    _p("P6-manifest",
       "manifest records the feed (only when not clip)",
       '        "nav_shuffle": shuf_stats,\n',
       '        "nav_shuffle": shuf_stats,\n'
       '        **({} if pf_args is None else {"nav_feed": _navfeed.manifest_record(nav_feed, pf_args, nav_clip, nav_true)}),\n'),
]


def patched_source(src: str, patches=None) -> str:
    """Apply every edit; each anchor must occur exactly once (else refuse -- the upstream file changed)."""
    for p in (PATCHES if patches is None else patches):
        n = src.count(p["anchor"])
        if n != 1:
            raise SystemExit(f"[navfeed_overlay] anchor {p['id']} occurs {n} times in refcv3_arm.py (need exactly 1); the upstream "
                             f"file changed -- refusing to patch")
        src = src.replace(p["anchor"], p["replacement"])
    return src


def refcv3_arm_path() -> Path:
    import os
    tree = Path(os.environ.get("R8_LAUNCH_TREE", "C:/lgs/Lfb47a58376"))
    return tree / "taniteval" / "tools" / "refcv3_arm.py"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def unified_diff(src: str, new: str, name="taniteval/tools/refcv3_arm.py") -> str:
    return "".join(difflib.unified_diff(src.splitlines(keepends=True), new.splitlines(keepends=True),
                                        fromfile=f"a/{name}", tofile=f"b/{name} (navfeed overlay)", n=3))


def load_patched_refcv3_arm(path: Path | None = None, *, register: bool = True):
    """Execute the PATCHED refcv3_arm as module ``refcv3_arm`` with the ORIGINAL ``__file__`` (the module resolves its sibling files
    -- refav1_arm.py, the scripts dir -- from ``__file__``).  Same contract as ``r7_roll.ra3()``, which it replaces for this process."""
    p = Path(path) if path else refcv3_arm_path()
    src = p.read_text(encoding="utf-8")
    new = patched_source(src)
    spec = importlib.util.spec_from_file_location("refcv3_arm", str(p))
    mod = importlib.util.module_from_spec(spec)
    mod.__navfeed__ = {"src_sha256": sha256_text(src), "patched_sha256": sha256_text(new),
                       "patches": [x["id"] for x in PATCHES]}
    if register:
        sys.modules["refcv3_arm"] = mod
    exec(compile(new, str(p), "exec", dont_inherit=True), mod.__dict__)
    return mod


def install_into_r7_roll(RR):
    """Make ``r7_roll.ra3()`` return the patched module (it caches in ``RR._RA3``)."""
    mod = load_patched_refcv3_arm()
    RR._RA3 = mod
    return mod


if __name__ == "__main__":      # python navfeed_overlay.py --write-diff <path>
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-diff", required=True)
    a = ap.parse_args()
    p = refcv3_arm_path()
    s = p.read_text(encoding="utf-8")
    n = patched_source(s)
    Path(a.write_diff).write_text(unified_diff(s, n), encoding="utf-8")
    print(f"src_sha256={sha256_text(s)} patched_sha256={sha256_text(n)} diff={a.write_diff}")
