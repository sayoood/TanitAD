"""trainfit_pf.py -- SHIP-R8 AMENDMENT 3 (ii): the UNCHANGED H-R8-RETIME-TRAINFIT-1 procedure on PER-FRAME-fed train rolls, frozen into a NEW dir.

NEW file (EvalFlyWheel 2026-10-20, hold_r8pf).  It imports ``retime_trainfit_r8.py`` (battery package, READ-ONLY, sha256 pinned below) and calls its OWN
``cmd_panels`` / ``cmd_freeze`` with exactly four module attributes re-pointed -- nothing in the fit is touched:
    TRAIN_ROOT    -> the per-frame roll root (default C:/Users/Admin/refcv8_pf/trainfit)
    TRAIN_TREE    -> step30000ema_train139_pf (dump_s0 / dump_s1 written by roll_seed_r8_train_pf.py)
    TF            -> the NEW output dir (default <hold>/raw/retime_trainfit_pf): the clip-fed table b208c9c1... is never overwritten
    FROZEN_TRAIN  -> TF / FROZEN_GAINS_RETIME_train_pf.json
``check_prereg`` still hashes the registered PREREG_TRAINFIT.md (the module constant is resolved at import, never re-pointed): the gains are produced under the
registered text.  The fit guard (train-only, the registered A6 sample fingerprint 6086d9cc...) runs inside ``freeze_arrays`` unchanged; this wrapper REFUSES to
write the pf sidecar files unless that guard reads ``is_registered_A6_sample: true`` and both seeds were pooled.

    python trainfit_pf.py panels                      # dump_s{0,1} -> panel_s{0,1}, s6_s{0,1}   (retime_trainfit_r8.cmd_panels)
    python trainfit_pf.py freeze  --parity-json <s0.json> <s1.json>    # FEED PARITY must read PASS on both seeds, then retime_trainfit_r8.cmd_freeze
    python trainfit_pf.py control --out-dir <scratch> # KNOWN-VALUE CONTROL: the SAME redirected path on the CLIP-FED tree must reproduce the registered
                                                      # clip-fed table b208c9c1... gain for gain (bit-equal floats); a no-op is not a pass
Env: PF_TRAIN_ROOT, PF_TRAIN_TREE, PF_TF_OUT override the defaults (used by the tests / the control).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = Path(__file__).resolve().parent
HOLD = HERE.parent
R8_CODE = Path(os.environ.get("PF_R8_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-15-refcv8-milestone-battery/code"))
TRAIN_ROOT_PF = os.environ.get("PF_TRAIN_ROOT", "C:/Users/Admin/refcv8_pf/trainfit")
TRAIN_TREE_PF = os.environ.get("PF_TRAIN_TREE", "step30000ema_train139_pf")
TF_OUT = Path(os.environ.get("PF_TF_OUT", str(HOLD / "raw" / "retime_trainfit_pf")))
FROZEN_NAME = "FROZEN_GAINS_RETIME_train_pf.json"
#: the fit code this wrapper is allowed to call: retime_trainfit_r8.py as it stands since 2026-10-19 15:00:38 (sha256 17b05b7e...).  MEASURED 2026-10-20: the
#: version that froze b208c9c1... (its record: script_sha256 9494d803..., recovered byte-exact from E:/refcv8_eval_kit/pkg_backup_20261019T0802.tgz) differs from
#: it ONLY in cmd_forward's `role` string (3 lines, diff in raw/retime_trainfit_pf/FIT_CODE_DIFF.txt); `cmd_control` re-proves the fit path bit-equal.
FIT_SCRIPT_SHA256 = "17b05b7ee48f1fe477f95f920e472caecd902e917f324a24aa7d96ab3c2526d4"
FIT_SCRIPT_SHA256_AT_CLIPFED_FREEZE = "9494d8032681a8ffe73005a253a349db5ada305d129c8ae458b353db5212c055"
CLIPFED_TABLE = R8_CODE.parent / "raw" / "retime_trainfit" / "FROZEN_GAINS_RETIME_train.json"
CLIPFED_SHA256 = "b208c9c1e4236075de107ef49eaef39c1dddddba257f6e594bf6b84f8a24b1ee"
CLIPFED_ROOT, CLIPFED_TREE = "E:/refcv8_eval_kit/trainfit", "step30000ema_train139"
A6_FINGERPRINT = "6086d9cc0c90281619eb0bb63736a73e012322b48dddf5b4ed0363008a0ee3ed"


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_fit_module():
    got = sha256_file(R8_CODE / "retime_trainfit_r8.py")
    if got != FIT_SCRIPT_SHA256:
        raise SystemExit(f"[trainfit_pf] retime_trainfit_r8.py sha256 {got} != {FIT_SCRIPT_SHA256}: not the fit code that produced the clip-fed table")
    sys.path.insert(0, str(R8_CODE))
    import retime_trainfit_r8 as TFM
    return TFM


def configure(TFM, root: str, tree: str, out_dir: Path, frozen_name: str = FROZEN_NAME) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.resolve() == Path(TFM.TF).resolve():
        raise SystemExit("[trainfit_pf] the output dir is the clip-fed table's dir -- refusing (it would overwrite b208c9c1...)")
    TFM.TRAIN_ROOT, TFM.TRAIN_TREE, TFM.TF, TFM.FROZEN_TRAIN = root, tree, out_dir, out_dir / frozen_name
    return {"TRAIN_ROOT": root, "TRAIN_TREE": tree, "TF": str(out_dir), "FROZEN_TRAIN": str(out_dir / frozen_name)}


def gains(fz: dict) -> dict:
    return {s: {"instants_s": fz["surfaces"][s]["instants_s"], "RETIME": fz["surfaces"][s]["all"]["RETIME"]} for s in ("S2", "S6")}


def cmd_panels() -> int:
    TFM = load_fit_module()
    configure(TFM, TRAIN_ROOT_PF, TRAIN_TREE_PF, TF_OUT)
    return TFM.cmd_panels()


def parity_ok(paths) -> tuple[bool, list]:
    rows = []
    for p in paths:
        try:
            j = json.load(open(p, encoding="utf-8"))
            rows.append({"file": str(p), "sha256": sha256_file(p), "status": j.get("status"), "label": j.get("label"), "counts": j.get("counts")})
        except Exception as exc:      # noqa: BLE001
            rows.append({"file": str(p), "status": "INCONCLUSIVE", "error": str(exc)[:200]})
    return (len(rows) == 2 and all(r["status"] == "PASS" for r in rows)), rows


def cmd_freeze(parity_paths) -> int:
    ok, prow = parity_ok(parity_paths)
    if not ok:
        raise SystemExit(f"[trainfit_pf] FEED PARITY did not read PASS on both train seeds -- no freeze: {prow}")
    TFM = load_fit_module()
    cfg = configure(TFM, TRAIN_ROOT_PF, TRAIN_TREE_PF, TF_OUT)
    frozen = Path(cfg["FROZEN_TRAIN"])
    if frozen.exists():
        raise SystemExit(f"[trainfit_pf] {frozen} already exists -- a frozen table is never re-frozen in place (move it aside by hand if that is intended)")
    rc = TFM.cmd_freeze()
    fz = json.load(open(frozen, encoding="utf-8"))
    sh = sha256_file(frozen)
    rec_sha = (TF_OUT / "FROZEN_GAINS_RETIME_train.sha256").read_text().strip()     # written by the UNCHANGED cmd_freeze, in the NEW dir
    g = fz.get("fit_guard") or {}
    problems = []
    if rec_sha != sh:
        problems.append(f"cmd_freeze's sha256 record {rec_sha} != the file {sh}")
    if not (g.get("is_registered_A6_sample") is True and g.get("fit_clips_fingerprint") == A6_FINGERPRINT and g.get("n_overlap_with_eval139") == 0):
        problems.append(f"fit guard is not the registered A6 train139 sample: {g}")
    if fz.get("seeds_pooled") != [0, 1] or fz.get("fallback") is not None:
        problems.append(f"seeds pooled {fz.get('seeds_pooled')} fallback {fz.get('fallback')}: both inference seeds are required")
    if fz.get("tree") != f"{TRAIN_ROOT_PF}/{TRAIN_TREE_PF}":
        problems.append(f"frozen tree {fz.get('tree')} is not the per-frame tree")
    if problems:
        (TF_OUT / "FREEZE_PF_REFUSED.json").write_text(json.dumps({"problems": problems, "sha256": sh}, indent=1))
        raise SystemExit("[trainfit_pf] REFUSED: " + "; ".join(problems))
    (TF_OUT / "FROZEN_GAINS_RETIME_train_pf.sha256").write_text(sh + "\n")
    clip = json.load(open(CLIPFED_TABLE, encoding="utf-8"))
    prov = {"tool": "trainfit_pf.py freeze", "generated": time.strftime("%FT%T%z"), "frozen_table": str(frozen), "sha256": sh,
            "spec": "SHIP-R8 AMENDMENT 3 (ii): the UNCHANGED H-R8-RETIME-TRAINFIT-1 procedure on per-frame-fed (perframe_full) train rolls",
            "fit_code": {"file": str(R8_CODE / "retime_trainfit_r8.py"), "sha256": FIT_SCRIPT_SHA256, "prereg_sha256": TFM.PREREG_SHA256,
                         "sha256_at_clip_fed_freeze": FIT_SCRIPT_SHA256_AT_CLIPFED_FREEZE, "diff_vs_clip_fed_freeze": "cmd_forward role string only (raw/retime_trainfit_pf/FIT_CODE_DIFF.txt)"},
            "redirect": cfg, "wrapper_sha256": sha256_file(__file__), "fit_guard": g, "seeds_pooled": fz["seeds_pooled"],
            "feed_parity": prow, "gains_pf": gains(fz),
            "beside_clip_fed_table": {"file": str(CLIPFED_TABLE), "sha256": sha256_file(CLIPFED_TABLE), "registered_sha256": CLIPFED_SHA256,
                                      "gains": gains(clip), "role": "BESIDE ONLY (AMENDMENT 3 (ii)); never the scored table"},
            "n_windows_by_surface": fz.get("n_windows_by_surface"), "n_compliant_by_surface": fz.get("n_compliant_by_surface"),
            "input_files": fz.get("input_files")}
    roll_recs = {}
    for s in (0, 1):
        p = Path(TRAIN_ROOT_PF) / TRAIN_TREE_PF / f"roll_s{s}.json"
        if p.exists():
            r = json.load(open(p, encoding="utf-8"))
            roll_recs[f"s{s}"] = {"file": str(p), "sha256": sha256_file(p), "n_windows": r.get("n_windows"), "n_episodes": r.get("n_episodes"),
                                  "device": r.get("device"), "wall_s": r.get("wall_s"), "pf_ok": (r.get("pf") or {}).get("ok"),
                                  "overlay_sha256": ((r.get("pf") or {}).get("overlay") or {}).get("navfeed_overlay_sha256")}
    prov["rolls"] = roll_recs
    (TF_OUT / "FROZEN_GAINS_RETIME_train_pf.provenance.json").write_text(json.dumps(prov, indent=1, default=str), encoding="utf-8")
    print(f"FROZEN pf table {frozen} sha256 {sh} ZZTFPFFROZENZZ", flush=True)
    return rc


def cmd_control(out_dir: str) -> int:
    """KNOWN-VALUE CONTROL of the redirect: the clip-fed tree through THIS wrapper's path must give the registered clip-fed gains exactly."""
    out = Path(out_dir)
    if (out / "FROZEN_GAINS_RETIME_train_control.json").exists():
        (out / "FROZEN_GAINS_RETIME_train_control.json").unlink()
    TFM = load_fit_module()
    configure(TFM, CLIPFED_ROOT, CLIPFED_TREE, out, "FROZEN_GAINS_RETIME_train_control.json")
    TFM.cmd_freeze()
    got = json.load(open(out / "FROZEN_GAINS_RETIME_train_control.json", encoding="utf-8"))
    ref = json.load(open(CLIPFED_TABLE, encoding="utf-8"))
    eq = {s: (got["surfaces"][s]["all"]["RETIME"] == ref["surfaces"][s]["all"]["RETIME"]
              and got["surfaces"][s]["all"]["GATED_RETIME"] == ref["surfaces"][s]["all"]["GATED_RETIME"]) for s in ("S2", "S6")}
    res = {"control": "redirected wrapper on the CLIP-FED tree reproduces the registered clip-fed table", "registered_sha256": CLIPFED_SHA256,
           "registered_file_sha256": sha256_file(CLIPFED_TABLE), "bit_equal_by_surface": eq, "fit_guard_equal": got["fit_guard"] == ref["fit_guard"],
           "ok": bool(all(eq.values()) and got["fit_guard"] == ref["fit_guard"] and sha256_file(CLIPFED_TABLE) == CLIPFED_SHA256)}
    (out / "CONTROL_REDIRECT.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), "ZZTFPFCONTROLOKZZ" if res["ok"] else "ZZTFPFCONTROLFAILZZ", flush=True)
    return 0 if res["ok"] else 4


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("panels")
    f = sub.add_parser("freeze")
    f.add_argument("--parity-json", nargs=2, required=True)
    c = sub.add_parser("control")
    c.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "panels":
        return cmd_panels()
    if a.cmd == "freeze":
        return cmd_freeze(a.parity_json)
    return cmd_control(a.out_dir)


if __name__ == "__main__":
    sys.exit(main())
