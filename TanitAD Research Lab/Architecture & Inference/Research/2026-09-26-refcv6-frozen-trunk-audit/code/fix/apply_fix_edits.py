"""Apply the A16 batch-2 edits to a CLEAN copy of the tip (never to D: or cfull_tip).

    python apply_fix_edits.py <tree-root>        # e.g. a `git archive` of agent/arch-inf-20260803

Every edit is an exact-string replacement that must match EXACTLY ONCE (asserted), and every
anchor of every file is matched BEFORE any file is written, so a drifted base refuses instead of
producing a half-applied tree.

⚠️ MEASURED 2026-09-26: these blobs are committed CRLF (refc.py 4,643 of 4,643 lines). Anchors are
matched on LF text and each file is written back in its OWN convention; a mixed file is refused.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _load(root: Path, rel: str) -> tuple[str, bool]:
    raw = (root / rel).read_bytes().decode("utf-8")
    n_crlf, n_lf = raw.count("\r\n"), raw.count("\n")
    if n_crlf not in (0, n_lf):
        raise SystemExit(f"{rel}: mixed line endings ({n_crlf} CRLF of {n_lf}) -- refusing")
    return raw.replace("\r\n", "\n"), n_crlf > 0


def _check(root: Path, rel: str, pairs) -> None:
    s, _ = _load(root, rel)
    for old, _new in pairs:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"{rel}: anchor matched {n} times (want 1): {old[:90]!r}")


def edit(root: Path, rel: str, pairs) -> None:
    s, crlf = _load(root, rel)
    for old, new in pairs:
        if s.count(old) != 1:
            raise SystemExit(f"{rel}: anchor matched {s.count(old)} times (want 1): {old[:90]!r}")
        s = s.replace(old, new)
    (root / rel).write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))


REFC_PAIRS = [(
    '        for _k in ("prefinal_logits", "reach_keep"):\n',
    '        # ⛔⛔ A16 2026-09-26: refcv6 F3\'s per-stage outputs MUST pass through. The trainer\'s\n'
    '        # per-stage loss reads `layer_u0_hat` / `layer_logits` from THIS dict; the whitelist that\n'
    '        # omitted them skipped F3\'s loss for the whole of refcv6-r101-s0 (0 of 668 training rows\n'
    '        # carried `cascade`; stage-0..2 heads bit-identical across 29,000 steps). Pinned by\n'
    '        # tests/test_refcv6_f3_cascade_reaches_loss.py on the REAL train().\n'
    '        for _k in ("prefinal_logits", "reach_keep", "layer_u0_hat", "layer_logits"):\n')]

CLOCK_METHODS = '''    # ======================================================================= #
    # ⭐⭐ A16 2026-09-26 -- THE LABEL CLOCK (`tanitad/data/clip_clock.py`).     #
    # ======================================================================= #
    #: ``sid -> (grid_start_s, dt_s, source)``: the true RAW-timeline clock of each clip. Filled
    #: by :meth:`enable_clip_clock`; a clip it does not hold is clocked on first use from its OWN
    #: poses (``clip_clock.pose_dt``) with ``grid_start_s = 0.0``.
    _label_clock: dict | None = None
    label_clock_report: dict | None = None
    #: ⛔ THE DELIBERATE-REGRESSION SWITCH, for tests ONLY: True restores the historical
    #: ``(t + w - 1) * v7_dt`` so a guard can prove it goes RED on the defect it exists for.
    legacy_label_clock: bool = False

    @staticmethod
    def _raw_offset(ep) -> int:
        """PROVIDER row -> RAW clip row: the provider drops the first ``n_stack - 1`` frames
        (``v2_dataset.py:36``) and the labels live on the RAW timeline -- the conversion
        ``s2_labels.py:738-740`` already applies."""
        return max(int(ep.frames.shape[1]) // 3 - 1, 0)

    def _clock_for(self, ep) -> tuple:
        from tanitad.data import clip_clock as _cc
        if self._label_clock is None:
            self._label_clock = {}
        sid = int(ep.episode_id)
        clk = self._label_clock.get(sid)
        if clk is None:
            dt = _cc.pose_dt(ep.poses)
            clk = ((0.0, float(dt), "pose_dt") if dt is not None
                   else (0.0, float(_cc.NOMINAL_DT_S), "nominal_dt"))
            self._label_clock[sid] = clk
        return clk

    def _now_s(self, ep, t: int) -> float:
        """The window NOW (row ``t + w - 1``) on the label's RAW timeline, in seconds."""
        r = int(t) + int(self.window) - 1
        if self.legacy_label_clock:              # the historical defect, tests only
            return r * float(self.v7_dt)
        g0, dt, _src = self._clock_for(ep)
        return float(g0) + (r + self._raw_offset(ep)) * float(dt)

    def enable_clip_clock(self, sidecar_path: str | None = None) -> dict:
        """Resolve every episode's clock ONCE and return the census config.json stamps.

        ``sidecar_path`` (``--clip-clock-sidecar``): the clock MEASURED from each clip's 100 Hz
        egomotion log (``scripts/build_clip_clock_sidecar.py``). Without it -- or for a clip it
        does not cover -- ``dt`` comes from the clip's own poses and ``grid_start_s`` is 0.0, and
        the census COUNTS it: the measured corpus median is +0.113 s, so that residual is stated,
        never hidden. ⛔ A sidecar that covers NONE of this split's clips REFUSES: that is a
        sidecar for another corpus, not a partial one."""
        from tanitad.data import clip_clock as _cc
        side = _cc.read_clip_clock_sidecar(str(sidecar_path)) if sidecar_path else None
        self._label_clock = {}
        n_side = n_pose = n_nom = 0
        dts, offs = [], set()
        for ep in self.episodes:
            sid = int(ep.episode_id)
            if sid in self._label_clock:
                continue
            if side is not None and sid in side.table:
                g0, dt = side.table[sid]
                self._label_clock[sid] = (float(g0), float(dt), "sidecar")
                n_side += 1
            else:
                clk = self._clock_for(ep)
                n_pose += int(clk[2] == "pose_dt")
                n_nom += int(clk[2] == "nominal_dt")
            dts.append(self._label_clock[sid][1])
            offs.add(self._raw_offset(ep))
        if side is not None and n_side == 0:
            raise SystemExit(
                f"[v3] ⛔ --clip-clock-sidecar {sidecar_path}: it covers NONE of this split's "
                f"{len(self._label_clock)} clips -- a sidecar for another corpus. Refusing rather "
                f"than clocking every clip on the fallback while config.json names the sidecar.")
        dts_s = sorted(dts)
        self.label_clock_report = {
            "rule": "t_now = grid_start_s + (t + w - 1 + n_stack - 1) * dt_s "
                    "(tanitad/data/clip_clock.py; A16 2026-09-26)",
            "sidecar": None if side is None else side.path,
            "sidecar_rows": None if side is None else side.n_rows,
            "n_clips": len(self._label_clock), "n_from_sidecar": n_side,
            "n_pose_dt_grid_start_0": n_pose, "n_nominal_dt_grid_start_0": n_nom,
            "raw_offsets": sorted(offs),
            "dt_s_median": dts_s[len(dts_s) // 2] if dts_s else None,
            "dt_s_min": dts_s[0] if dts_s else None,
            "dt_s_max": dts_s[-1] if dts_s else None}
        return self.label_clock_report

'''

TRAIN_PAIRS = [
    # (1) F3: refuse, do not skip
    ('    if _rv6f.f3_per_layer and "layer_u0_hat" in out:\n',
     '    # ⛔⛔ A16 2026-09-26: REFUSE, DO NOT SKIP. This block used to read `if f3 and "layer_u0_hat"\n'
     '    # in out:` -- a silent skip -- and `RefCModel.forward` never passed the key through, so the\n'
     '    # live refcv6 run trained with F3 STAMPED and F3\'s loss ABSENT (0/668 rows carried `cascade`).\n'
     '    if _rv6f.f3_per_layer and "layer_u0_hat" not in out:\n'
     '        raise SystemExit(\n'
     '            "[v3] ⛔ this build is --f3-per-layer but the forward carries no `layer_u0_hat`: the "\n'
     '            "per-stage (cascade) loss would be SILENTLY SKIPPED while config.json stamps F3. "\n'
     '            "RefCModel.forward must pass the decoder\'s per-stage outputs through.")\n'
     '    if _rv6f.f3_per_layer and "layer_u0_hat" in out:\n'),
    # (2) the two label lookups read the TRUE clock
    ('                lat_v7, lon_v7 = v7l.tactical_class_ids(\n'
     '                    lab, (t + w - 1) * self.v7_dt)\n',
     '                lat_v7, lon_v7 = v7l.tactical_class_ids(\n'
     '                    lab, self._now_s(ep, t))\n'),
    ('                    _tg_y, _tg_w = v7l.tactical_goal_targets(\n'
     '                        lab, (t + w - 1) * self.v7_dt,\n',
     '                    _tg_y, _tg_w = v7l.tactical_goal_targets(\n'
     '                        lab, self._now_s(ep, t),\n'),
    # (3) the clock itself, on V3Dataset
    ('    def __getitem__(self, i: int):\n'
     '        item = super().__getitem__(i)\n'
     '        e_i, t = self.index[i]\n',
     CLOCK_METHODS +
     '    def __getitem__(self, i: int):\n'
     '        item = super().__getitem__(i)\n'
     '        e_i, t = self.index[i]\n'),
    # (4) wiring: train split
    ('        ds.v7_by_sid = by_sid\n'
     '        ds.v7_dt = 0.1\n',
     '        ds.v7_by_sid = by_sid\n'
     '        ds.v7_dt = 0.1\n'
     '        # ⭐⭐ A16 2026-09-26: the label clock (tanitad/data/clip_clock.py), resolved ONCE.\n'
     '        clip_clock_stats = ds.enable_clip_clock(getattr(args, "clip_clock_sidecar", None))\n'),
    # (5) wiring: eval split
    ('            e_ds.v7_by_sid = {stable_episode_id(l.clip_id): l for l in e_lab}\n'
     '            e_ds.v7_dt = 0.1\n',
     '            e_ds.v7_by_sid = {stable_episode_id(l.clip_id): l for l in e_lab}\n'
     '            e_ds.v7_dt = 0.1\n'
     '            eval_clip_clock_stats = e_ds.enable_clip_clock(\n'
     '                getattr(args, "clip_clock_sidecar", None))\n'),
    # (6) stats defaults
    ('    max_speed_v6_stats = eval_max_speed_v6_stats = None\n'
     '    tac_v6_stats = None\n',
     '    max_speed_v6_stats = eval_max_speed_v6_stats = None\n'
     '    clip_clock_stats = eval_clip_clock_stats = None       # A16 label clock\n'
     '    tac_v6_stats = None\n'),
    # (7) the stamp
    ('        "refcv6_max_speed": ({"train": max_speed_v6_stats,\n',
     '        # ⭐⭐ A16 2026-09-26: WHICH clock read the labels, and how many clips fell back.\n'
     '        "label_clock": ({"train": clip_clock_stats, "eval": eval_clip_clock_stats}\n'
     '                        if clip_clock_stats is not None else None),\n'
     '        "refcv6_max_speed": ({"train": max_speed_v6_stats,\n'),
    # (8) the flag
    ('    ap.add_argument("--v7-labels", default=None,\n',
     '    ap.add_argument("--clip-clock-sidecar", default=None,\n'
     '                    help="A16 2026-09-26: JSON-lines {sid, grid_start_s, dt_s} built by "\n'
     '                         "`scripts/build_clip_clock_sidecar.py` from each clip\'s 100 Hz "\n'
     '                         "egomotion log -- the TRUE clock the v7/v8 labels are read on. "\n'
     '                         "Without it dt comes from the clip\'s own poses and grid_start is "\n'
     '                         "0.0 (counted in config.json `label_clock`).")\n'
     '    ap.add_argument("--v7-labels", default=None,\n'),
]


def main():
    root = Path(sys.argv[1])
    plan = (("stack/tanitad/refs/refc.py", REFC_PAIRS),
            ("stack/scripts/refc_v3_train.py", TRAIN_PAIRS))
    for rel, pairs in plan:            # phase 1: every anchor, every file, nothing written
        _check(root, rel, pairs)
    for rel, pairs in plan:            # phase 2: write
        edit(root, rel, pairs)
    print("applied")


if __name__ == "__main__":
    main()
