"""E2(b): G3's per-family EVAL mode -- TACTICAL is scored on clock-verified clips only; the
unverified clips are EXCLUDED from the tactical targets (n + reason stamped) and the other three
families keep every clip. CRLF-aware exact-match edits on code/fix/stack/scripts/refc_v3_train.py."""
import sys

P = sys.argv[1]
d = open(P, "rb").read()
assert d.count(b"\r\n") == d.count(b"\n"), "expected an all-CRLF file"
s = d.decode("utf-8").replace("\r\n", "\n")


def edit(old, new, tag):
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"[patch] {tag}: anchor found {n} times")
    s = s.replace(old, new)


edit('''    legacy_label_clock: bool = False

    def assert_label_clock_true(self, sidecar_path: str | None, *, split: str,
                                synthetic: bool = False,
                                tol_s: float | None = None,
                                max_unverified_frac: float | None = None) -> dict:
''', '''    legacy_label_clock: bool = False
    #: ⛔ G3's EVAL mode (E2(b), 2026-09-26): clips whose label clock could not be MEASURED (the
    #: sidecar refused them). Their TACTICAL targets read as "clip has no record" (IGNORE), so
    #: the tactical family is scored on verified clips only; nav, ego and every other input and
    #: target are untouched, so the other three families keep every clip.
    tactical_excluded_sids: frozenset = frozenset()

    def assert_label_clock_true(self, sidecar_path: str | None, *, split: str,
                                synthetic: bool = False,
                                tol_s: float | None = None,
                                max_unverified_frac: float | None = None,
                                exclude_unverified_tactical: bool = False) -> dict:
''', "attr + signature")

edit('''        * clips the sidecar does not cover are UNVERIFIED; more than ``max_unverified_frac`` of
          the split REFUSES, fewer are counted and stamped by name-count.
        """
''', '''        * clips the sidecar does not cover are UNVERIFIED; more than ``max_unverified_frac`` of
          the split REFUSES, fewer are counted and stamped by name-count;
        * ``exclude_unverified_tactical=True`` (the EVAL split): instead of refusing, the
          unverified clips' TACTICAL targets are excluded and the family is scored on the
          verified clips, with n and the reason stamped -- the four-families rule's per-family n,
          never a raised cap (MEASURED: eval-139 has 3 unverified clips = 2.16 %, all refused
          by the sidecar for "fewer than 20 moving rows").
        """
''', "docstring")

edit('''        n_ep = len(by_ep)
        frac = len(unverified) / max(n_ep, 1)
        if frac > cap:
''', '''        n_ep = len(by_ep)
        frac = len(unverified) / max(n_ep, 1)
        if exclude_unverified_tactical:
            self.tactical_excluded_sids = frozenset(int(x) for x in unverified)
            return {"g3": "PASS", "split": split, "tol_s": tol,
                    "n_clips": n_ep, "n_reads_checked": n_checked,
                    "worst_abs_err_s": round(worst, 6),
                    "n_unverified_clips": len(unverified),
                    "tactical_scored_on_clips": n_ep - len(unverified),
                    "tactical_excluded_clips": len(unverified),
                    "tactical_excluded_sids": sorted(int(x) for x in unverified),
                    "tactical_excluded_reason": (
                        "no MEASURED label clock (absent from the clip-clock sidecar): their "
                        "tactical targets cannot be shown within tol_s of the truth"),
                    "families_on_all_clips": ["longitudinal", "lateral", "strategic"],
                    "mode": "per-family: TACTICAL on verified clips only (E2(b))",
                    "rule": "t_true = grid_start_s + (t + w - 1 + n_stack - 1) * dt_s (sidecar)"}
        if frac > cap:
''', "eval-mode branch")

edit('''            lab = self.v7_by_sid.get(int(ep.episode_id))
''', '''            lab = self.v7_by_sid.get(int(ep.episode_id))
            if int(ep.episode_id) in self.tactical_excluded_sids:
                lab = None            # G3 eval mode: no measured clock -> tactical EXCLUDED
''', "getitem exclusion")

edit('''            eval_clip_clock_stats["g3"] = e_ds.assert_label_clock_true(
                getattr(args, "clip_clock_sidecar", None), split="eval",
                synthetic=bool(getattr(args, "synth_episodes", 0)),
                max_unverified_frac=getattr(args, "label_clock_max_unverified", None))
''', '''            # ⛔ E2(b): the EVAL split never raises the cap -- its unverified clips are EXCLUDED
            # from the TACTICAL family (n + reason in config.json); the other families keep them.
            eval_clip_clock_stats["g3"] = e_ds.assert_label_clock_true(
                getattr(args, "clip_clock_sidecar", None), split="eval",
                synthetic=bool(getattr(args, "synth_episodes", 0)),
                max_unverified_frac=getattr(args, "label_clock_max_unverified", None),
                exclude_unverified_tactical=True)
''', "eval call")

out = s.replace("\n", "\r\n").encode("utf-8")
open(P, "wb").write(out)
print("patched (CRLF kept)")
