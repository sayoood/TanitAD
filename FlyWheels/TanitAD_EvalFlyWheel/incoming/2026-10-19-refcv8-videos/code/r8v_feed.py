"""r8v_feed.py -- the ONE place the refcv8 VIDEO path decides which nav the model was FED (EvalFlyWheel 2026-10-20, hold_r8vidpf).  NEW file.

WHY (R8-EVAL-NAV-FEED-1, retraction R57, MEASURED by the nav-feed hold): training feeds the PER-FRAME v9 nav token + ``r8_nav = scale_nav(args, known)``;
the old eval roll fed the CLIP-LEVEL v7 token (constant over 139/139 eval clips) and no ``r8_nav``.  PI decision (a) 2026-10-20: the 48,852 bars and
SHIP-R8 read on the per-frame feed (tree ``step48852ema_pf``).  The videos must show THAT model, and their NAV label must show what was FED.

Three entry points, each REUSING the pf battery's own code (read-only imports, never re-implemented):
  install_feed(RR, mode)      dump_heads_r8: the pf battery's wiring (``roll_seed_r8_pf.py``): the REGISTERED ``navfeed_overlay`` (sha256 checked by
                              ``roll_seed_r8_train_pf.check_overlay``) installed into ``r7_roll`` BEFORE ``install_patches`` / ``ra3()``, and
                              ``--nav-feed perframe_full`` appended to the argv ``refcv3_arm.main`` receives.  The token and ``r8_nav`` are then built by
                              ``navfeed_overlay.perframe_nav`` / ``model_kw`` inside the roll -- the SAME function the binding battery runs.
  read_dump_feed(dump_dir)    renderer + checks: the fed token / clip token / r8_nav / known PER WINDOW exactly as the roll WROTE them (decisions sidecar,
                              mode from the dump manifest).  ⛔ It never re-derives the fed token by a separate lookup (that is how R57 happened).
  check_feed_parity(rows, pf_dump, v9)
                              a video artifact's per-window feed == the pf battery dump's sidecar (``feed_parity_pf.load_dump``) AND == the INDEPENDENT
                              v9 release lookup + numpy scaling (``feed_parity_pf.V9``, numpy only, shares no code with the overlay).  PASS needs >= 1
                              DISCRIMINATING window (per-frame token != clip token): a set where the two agree cannot tell the feeds apart.

MUTATION ARM: ``R8V_MUTATE_FEED=clip`` makes every entry point behave like the R57 defect (no overlay, no ``--nav-feed``; the clip token reported as the
fed token).  ``test_r8v_feed_parity.py`` must go RED under it.

    python r8v_feed.py check --heads-dir <dir> --pf-dump <tree>/dump_s0 --sha12-tree <tree> [--v9 ...] --out <json>
    python r8v_feed.py check --tree-root <arch>/<tree> --pf-dump <tree>/dump_s0 --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PF_CODE = Path(os.environ.get("R8V_PF_CODE", "C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                             "2026-10-20-refcv8-perframe-final/code"))
NAV_FEED_PF = "perframe_full"                          # LITERAL: the only per-frame mode the run was TRAINED on (token + r8_nav)
FEEDS = ("perframe_full", "clip")
OVERLAY_SHA256 = "98293dd7411270160cb47d070bb1538babe1ff3a0f8ab85da61dcc84129ba72f"   # LITERAL (PREREG_NAV_FEED.md; == roll_seed_r8_train_pf.OVERLAY_SHA256)
SIDE_KEYS = ("nav_cmd", "nav_cmd_clip", "r8_nav_known", "r8_nav_fed")
V9_EVAL = "E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz"
V9_EVAL_MD5 = "6b5c7f207cffc3b7eb3cd527fd433599"       # LITERAL (chain_final_pf_r8.sh EV9MD5)
R8_NAV_DIMS = 6
NAV_NAMES = ("FOLLOW_ROAD", "TURN_L", "TURN_R")         # NAV_COMMAND_TOKENS order (vocab_v7); == r8v_data.NAV_NAMES


def mutated() -> bool:
    return os.environ.get("R8V_MUTATE_FEED", "").strip().lower() == "clip"


def sha12(s) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


def _pf_mod(name):
    if str(PF_CODE) not in sys.path:
        sys.path.insert(0, str(PF_CODE))
    return __import__(name)


# ============================================================================================================ (1) the roll wiring (dump_heads_r8)
def install_feed(RR, mode: str) -> dict:
    """Wire the nav feed into ``r7_roll`` (RR) BEFORE ``run_battery_r7.roll_one`` runs.  Returns the record for keys.json.

    perframe_full: exactly what ``roll_seed_r8_pf.py`` does at import (READ-ONLY reuse of its two dependencies): ``TP.check_overlay()`` (the overlay file
    is the REGISTERED one), ``navfeed_overlay.install_into_r7_roll(RR)`` (``RR.ra3()`` now returns the PATCHED refcv3_arm), and ``refcv3_arm.main``'s argv
    gains ``--nav-feed perframe_full``.  clip: nothing is installed (the banked clip-fed behaviour, byte-identical)."""
    if mode not in FEEDS:
        raise SystemExit(f"[r8v_feed] --feed {mode!r} is not one of {FEEDS}")
    asked = mode
    if mutated():
        mode = "clip"                                   # MUTATION: the R57 defect (overlay never installed, no --nav-feed)
    if mode == "clip":
        return {"nav_feed": "clip", "asked": asked, "mutated": mutated(), "overlay": None, "argv_tail": [],
                "note": "clip-level v7 token, no r8_nav (the banked *_unc behaviour; NOT how refcv8 was trained)"}
    TP = _pf_mod("roll_seed_r8_train_pf")
    if TP.NAV_FEED != NAV_FEED_PF or TP.OVERLAY_SHA256 != OVERLAY_SHA256:
        raise SystemExit(f"[r8v_feed] the pf package's NAV_FEED {TP.NAV_FEED!r} / overlay {TP.OVERLAY_SHA256} != this module's literals")
    ov = TP.check_overlay()
    if str(TP.NF_OVERLAY) not in sys.path:
        sys.path.insert(0, str(TP.NF_OVERLAY))          # the patched refcv3_arm does `import navfeed_overlay`
    import navfeed_overlay as NO
    R = NO.install_into_r7_roll(RR)
    tail = ["--nav-feed", TP.NAV_FEED]
    R._r8v_orig_main = R.main

    def _main_with_feed(argv_=None):                    # == roll_seed_r8_pf._main_with_feed (same refusal, same tail)
        av = list(argv_ or [])
        if "--nav-feed" in av:
            raise SystemExit("[r8v_feed] the parent argv already carries --nav-feed")
        return R._r8v_orig_main(av + tail)              # looked up at CALL time (the test re-points it; the roll never does)
    R.main = _main_with_feed
    return {"nav_feed": TP.NAV_FEED, "asked": asked, "mutated": False, "overlay": ov, "patched_refcv3_arm": dict(R.__navfeed__),
            "argv_tail": tail, "pf_code": str(PF_CODE), "wiring": "roll_seed_r8_pf.py (overlay into r7_roll before install_patches; --nav-feed appended)",
            "navfeed_overlay_file": str(Path(NO.__file__).resolve())}


# ============================================================================================================ (2) what the roll WROTE (renderer, checks)
def feed_from_decisions(z, mode: str) -> dict:
    """Per-window feed of ONE decisions file (an open npz / dict).  mode = the dump manifest's ``nav_feed.mode`` ('clip' when the block is absent)."""
    files = set(z.files) if hasattr(z, "files") else set(z)
    n = len(z["ws"])
    if mode == NAV_FEED_PF:
        miss = [k for k in SIDE_KEYS if k not in files or np.asarray(z[k]).shape[0] != n]
        if miss:
            raise SystemExit(f"[r8v_feed] a {mode} dump lacks per-window sidecar key(s) {miss}: refusing to show a nav that was not recorded")
        fed = np.asarray(z["nav_cmd"]).astype(np.int64)
        clip = np.asarray(z["nav_cmd_clip"]).astype(np.int64)
        known = np.asarray(z["r8_nav_known"]).astype(bool)
        r8 = np.asarray(z["r8_nav_fed"], np.float32).reshape(n, R8_NAV_DIMS)
    elif mode == "clip":
        extra = [k for k in SIDE_KEYS[1:] if k in files]
        if extra:
            raise SystemExit(f"[r8v_feed] the manifest says the CLIP feed but the sidecar carries per-frame key(s) {extra}: inconsistent artifact")
        fed = np.asarray(z["nav_cmd"]).astype(np.int64)
        clip = fed.copy()
        known = np.zeros(n, bool)
        r8 = np.zeros((n, R8_NAV_DIMS), np.float32)
    else:
        raise SystemExit(f"[r8v_feed] nav_feed mode {mode!r}: the videos render only {FEEDS} ('perframe_token' is not the trained regime)")
    if mutated():
        fed = clip.copy()                               # MUTATION: the clip token reported as the fed token (the R57 display defect)
    return {"mode": mode, "ws": np.asarray(z["ws"]).astype(np.int64), "nav_cmd_fed": fed, "nav_cmd_clip": clip,
            "r8_nav_known": known, "r8_nav": r8, "args_fed": mode == NAV_FEED_PF}


def dump_mode(dump_dir) -> tuple[str, dict]:
    man = json.load(open(Path(dump_dir) / "manifest.json", encoding="utf-8"))
    nf = man.get("nav_feed") or {}
    mode = nf.get("mode") or "clip"                    # overlay P6 writes the block only when the feed is not clip
    if mode == NAV_FEED_PF and nf.get("args_fed") is not True:
        raise SystemExit(f"[r8v_feed] {dump_dir}: manifest mode {mode} but args_fed={nf.get('args_fed')!r}")
    return mode, man


def read_dump_feed(dump_dir) -> dict:
    """{'mode', 'rows': [ {sha12, episode_index, file_index, ws, nav_cmd_fed, nav_cmd_clip, r8_nav_known, r8_nav} ]} for every window of the dump."""
    d = Path(dump_dir)
    mode, man = dump_mode(d)
    rows = []
    for e in man["episodes"]:
        fi = int(e["file_index"])
        z = np.load(d / "decisions" / f"ep{fi:03d}.npz", allow_pickle=False)
        f = feed_from_decisions(z, mode)
        s12 = sha12(e["clip_id"])                       # the raw clip id never leaves this function
        for j, w in enumerate(f["ws"]):
            rows.append({"sha12": s12, "episode_index": int(e["episode_index"]), "file_index": fi, "ws": int(w),
                         "nav_cmd_fed": int(f["nav_cmd_fed"][j]), "nav_cmd_clip": int(f["nav_cmd_clip"][j]),
                         "r8_nav_known": bool(f["r8_nav_known"][j]), "r8_nav": f["r8_nav"][j].astype(np.float32)})
    return {"mode": mode, "manifest_nav_feed": man.get("nav_feed"), "rows": rows, "n_windows_manifest": int(man["grid"]["n_windows"])}


def tree_feed(tree_root, tree_name: str | None = None) -> dict:
    """The feed of a battery TREE, asserted on its artifacts: dump_s0 manifest mode + (for a pf tree) battery_summary.pf.all_seeds_fed.
    A tree NAMED *_pf must be per-frame fed and a per-frame tree must be named *_pf (run_battery_r8_pf.py's own rule)."""
    root = Path(tree_root)
    name = tree_name or root.name
    mode, _man = dump_mode(root / "dump_s0")
    rec = {"tree": name, "mode": mode}
    if name.endswith("_pf") != (mode == NAV_FEED_PF):
        raise SystemExit(f"[r8v_feed] tree {name!r}: name says {'per-frame' if name.endswith('_pf') else 'per-clip'} but dump_s0/manifest.json "
                         f"says nav_feed mode {mode!r} -- refusing to render a tree whose feed is not what its name says")
    if mode == NAV_FEED_PF:
        sp = root / "battery_summary.json"
        pf = (json.load(open(sp, encoding="utf-8")).get("pf") or {}) if sp.exists() else {}
        if pf.get("all_seeds_fed") is not True or pf.get("nav_feed") != NAV_FEED_PF:
            raise SystemExit(f"[r8v_feed] {sp}: no pf block with all_seeds_fed=True / nav_feed={NAV_FEED_PF} (the battery did not assert its feed)")
        rec["battery_pf"] = {"all_seeds_fed": True, "overlay": pf.get("overlay"), "wrapper_sha256": pf.get("wrapper_sha256")}
    return rec


def label(fed: int, clip: int, known: bool, mode: str) -> tuple[str, str]:
    """(NAV FED text, clip-nav text) for the on-screen tactical panel; values come from the ARTIFACT (``feed_from_decisions``)."""
    def nm(t):
        return NAV_NAMES[int(t)] if 0 <= int(t) < len(NAV_NAMES) else f"?{int(t)}"
    if mode == NAV_FEED_PF and not mutated():
        return f"{nm(fed)}  (per-frame v9, args {'known' if known else 'UNKNOWN row'})", f"{nm(clip)}" + ("" if fed == clip else "  (differs)")
    return f"{nm(fed)}  (CLIP-level token, no r8_nav -- NOT the trained feed)", f"{nm(clip)}"


# ============================================================================================================ (3) parity vs the pf battery + independent v9
def rows_from_heads(heads_dir, sha12_by_index: dict) -> list:
    """Per-window feed rows of a dump_heads_r8 output (``heads_ep###.npz`` carry nav_cmd_fed / nav_cmd_clip / r8_nav / r8_nav_known / nav_feed)."""
    rows = []
    for p in sorted(Path(heads_dir).glob("heads_ep*.npz")):
        ci = int(p.stem.replace("heads_ep", ""))
        z = np.load(p, allow_pickle=False)
        need = ("ws", "nav_cmd_fed", "nav_cmd_clip", "r8_nav", "r8_nav_known", "nav_feed")
        miss = [k for k in need if k not in z.files]
        if miss:
            raise SystemExit(f"[r8v_feed] {p.name} lacks {miss}: a pre-2026-10-20 (clip-fed, unrecorded) heads dump")
        for j, w in enumerate(z["ws"]):
            rows.append({"sha12": sha12_by_index[ci], "episode_index": ci, "ws": int(w), "nav_cmd_fed": int(z["nav_cmd_fed"][j]),
                         "nav_cmd_clip": int(z["nav_cmd_clip"][j]), "r8_nav_known": bool(z["r8_nav_known"][j]),
                         "r8_nav": np.asarray(z["r8_nav"][j], np.float32), "nav_feed": str(z["nav_feed"])})
    return rows


def check_feed_parity(rows: list, pf_dump, v9_path=V9_EVAL, v9_md5=V9_EVAL_MD5, label_="") -> dict:
    """Every video row == the pf battery dump's sidecar at (sha12, ws) [PF] AND == the independent v9 lookup [V9].  INCONCLUSIVE on any unreadable
    input, an empty set, or a set with no window where the per-frame token differs from the clip token (it could not see the R57 defect)."""
    FP = _pf_mod("feed_parity_pf")
    r = {"tool": "r8v_feed.check_feed_parity", "label": label_, "status": "INCONCLUSIVE", "pf_dump": str(pf_dump), "v9": str(v9_path),
         "mutated": mutated(), "checks": {}, "counts": {}}
    try:
        if not rows:
            r["error"] = "no video windows"
            return r
        if v9_md5:
            got = FP.md5_file(v9_path)
            if got != v9_md5:
                r["error"] = f"v9 md5 {got} != {v9_md5}"
                return r
        man, P, ids = FP.load_dump(pf_dump)
        off = FP.raw_offset_of(man)
        if (man.get("nav_feed") or {}).get("mode") != NAV_FEED_PF:
            r["error"] = f"the reference dump {pf_dump} is not a {NAV_FEED_PF} dump"
            return r
        v9 = FP.V9(v9_path)
        bad = {"PF": [], "V9": []}
        n_disc = 0
        for x in rows:
            k = (x["sha12"], int(x["ws"]))
            ref = P.get(k)
            fed, ctok, kn = int(x["nav_cmd_fed"]), int(x["nav_cmd_clip"]), bool(x["r8_nav_known"])
            r8 = np.asarray(x["r8_nav"], np.float64).reshape(-1)
            if ref is None or any(ref[q] is None for q in ("nav_cmd", "nav_cmd_clip", "r8_nav_known", "r8_nav_fed")):
                bad["PF"].append({"episode_index": x["episode_index"], "ws": x["ws"], "why": "window absent from the pf battery dump"})
            else:
                rf = np.asarray(ref["r8_nav_fed"], np.float64).reshape(-1)
                if (fed != int(ref["nav_cmd"]) or ctok != int(ref["nav_cmd_clip"]) or kn != bool(ref["r8_nav_known"])
                        or r8.shape != rf.shape or float(np.max(np.abs(r8 - rf))) > 1e-6):
                    bad["PF"].append({"episode_index": x["episode_index"], "ws": x["ws"], "video": [fed, ctok, kn], "pf": [int(ref["nav_cmd"]),
                                      int(ref["nav_cmd_clip"]), bool(ref["r8_nav_known"])],
                                      "r8_nav_max_abs": float(np.max(np.abs(r8 - rf))) if r8.shape == rf.shape else None})
            sid = ids[x["sha12"]][1] if x["sha12"] in ids else None
            if sid is None:
                bad["V9"].append({"episode_index": x["episode_index"], "ws": x["ws"], "why": "clip not in the pf dump manifest (no sid)"})
                continue
            e = v9.expected(sid, int(x["ws"]) + off)
            nz = bool(np.abs(r8).sum() > 0)
            if fed != e["token"] or kn != e["known"] or nz != kn or r8.shape != (6,) or float(np.max(np.abs(r8 - e["fed"]))) > FP.ARGS_TOL:
                bad["V9"].append({"episode_index": x["episode_index"], "ws": x["ws"], "fed": fed, "v9_token": e["token"], "known": kn,
                                  "v9_known": e["known"], "max_abs_vs_independent": float(np.max(np.abs(r8 - e["fed"]))) if r8.shape == (6,) else None})
            n_disc += int(e["token"] != ctok)
        r["checks"] = {"PF_video_feed_equals_pf_battery_sidecar": not bad["PF"], "V9_video_feed_equals_independent_v9_lookup": not bad["V9"]}
        r["counts"] = {"n_windows": len(rows), "n_discriminating_windows": n_disc, "n_fail": {k: len(v) for k, v in bad.items()},
                       "raw_offset_from_manifest": off}
        r["first_failures"] = {k: v[:5] for k, v in bad.items() if v}
        if n_disc == 0:
            r["status"] = "INCONCLUSIVE"
            r["error"] = "no window where the per-frame token differs from the clip token: this set cannot distinguish the feeds"
        else:
            r["status"] = "PASS" if all(r["checks"].values()) else "FAIL"
    except SystemExit as exc:
        r["error"] = f"SystemExit: {exc}"
    except Exception as exc:      # noqa: BLE001 -- an unreadable artifact is INCONCLUSIVE, never PASS
        r["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check",))
    ap.add_argument("--heads-dir")
    ap.add_argument("--sha12-tree", help="battery tree root whose battery_summary maps episode index -> sha12 (with --heads-dir)")
    ap.add_argument("--tree-root", help="renderer path: read the feed exactly as r8v_data.EvalBank does (dump_s0 sidecar)")
    ap.add_argument("--pf-dump", required=True)
    ap.add_argument("--v9", default=V9_EVAL)
    ap.add_argument("--v9-md5", default=V9_EVAL_MD5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if bool(a.heads_dir) == bool(a.tree_root):
        ap.error("exactly one of --heads-dir / --tree-root")
    if a.heads_dir:
        if not a.sha12_tree:
            ap.error("--heads-dir needs --sha12-tree")
        summ = json.load(open(Path(a.sha12_tree) / "battery_summary.json", encoding="utf-8"))
        by = {int(k): v for k, v in summ["stages"]["roll_s0"]["extras"]["clip_sha12_by_index"].items()}
        rows, lab = rows_from_heads(a.heads_dir, by), f"heads {a.heads_dir}"
    else:
        tree_feed(a.tree_root)
        rows, lab = read_dump_feed(Path(a.tree_root) / "dump_s0")["rows"], f"tree {a.tree_root}"
    r = check_feed_parity(rows, a.pf_dump, a.v9, a.v9_md5, lab)
    Path(a.out).write_text(json.dumps(r, indent=1, default=str), encoding="utf-8", newline="\n")
    print(json.dumps({"status": r["status"], "counts": r.get("counts"), "error": r.get("error")}),
          "ZZVIDFEEDPASSZZ" if r["status"] == "PASS" else "ZZVIDFEEDNOTPASSZZ", flush=True)
    return 0 if r["status"] == "PASS" else 5


if __name__ == "__main__":
    sys.exit(main())
