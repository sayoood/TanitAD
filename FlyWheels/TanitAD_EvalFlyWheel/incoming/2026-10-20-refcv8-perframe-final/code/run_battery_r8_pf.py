"""run_battery_r8_pf.py -- the refcv8 four-family battery with the PER-FRAME nav feed (SHIP-R8 AMENDMENT 3 (i)).

NEW file (EvalFlyWheel 2026-10-20, hold_r8pf).  It runs ``run_battery_r8.py`` (battery package, READ-ONLY import, UNCHANGED: same refusals,
same windows -- the refcv4b S2 list --, same panels / families / bootstrap, same argv) with ONE substitution: the roll CHILD.
``r8_battery_patch.rewrite_child_cmd`` swaps ``roll_seed_r7.py`` -> ``roll_seed_r8.py``; this wrapper then swaps ``roll_seed_r8.py`` ->
``roll_seed_r8_pf.py`` (same argv), so every seed's dump is rolled with ``--nav-feed perframe_full``.  Nothing else in the parent changes.

    R8_NO_POSE_SYNC=1 python run_battery_r8_pf.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json \
        --no-g0 --tag step48852ema_pf --infer-seeds 0,1 --lock-held --skip-roll --out-root C:/Users/Admin/refcv8_eval_archive/battery

REFUSES a ``--tag`` that does not end in ``_pf`` (a per-frame tree must never land under a per-clip name).  After the parent returns,
``battery_summary.json`` gains a ``pf`` block: the child rewrites and every seed's ``roll_s*.json`` ``pf`` census (asserted on the artifact).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import roll_seed_r8_train_pf as TP  # noqa: E402

PF_CHILD = HERE / "roll_seed_r8_pf.py"
REWRITES = []


def pf_rewrite_factory(orig):
    def pf_rewrite(cmd, **kw):
        out = orig(cmd, **kw)
        for i, c in enumerate(out):
            if str(c).replace("\\", "/").endswith("/roll_seed_r8.py"):
                out[i] = str(PF_CHILD)
                REWRITES.append({"from": c, "to": out[i], "argv_tail": [str(x) for x in out[i + 1:]]})
                break
        else:
            if any(str(c).replace("\\", "/").endswith(("/roll_seed_r7.py",)) for c in out):
                raise SystemExit("[run_battery_r8_pf] a roll child escaped the r8 rewrite -- refusing to roll it un-fed")
        return out
    return pf_rewrite


def _val(flag, argv):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def main() -> int:
    argv = sys.argv[1:]
    tag = _val("--tag", argv)
    if not tag or not tag.endswith("_pf"):
        raise SystemExit(f"[run_battery_r8_pf] refusing --tag {tag!r}: a per-frame tree must be tagged *_pf")
    ov = TP.check_overlay()
    sys.path.insert(0, str(TP.R8_CODE))
    import r8_battery_patch as BP
    import run_battery_r8 as RB8
    BP.rewrite_child_cmd = pf_rewrite_factory(BP.rewrite_child_cmd)     # SubprocessShim.run looks the name up at CALL time
    rc = 0
    try:
        RB8.main()
    except SystemExit as e:                                              # the parent's own refusals / a missing roll record
        rc = e.code if isinstance(e.code, int) else 1
        print(f"[run_battery_r8_pf] parent exited: {e}", flush=True)
    out_root = _val("--out-root", argv) or "E:/refcv8_eval_kit/battery"
    work = Path(out_root) / tag
    sp = work / "battery_summary.json"
    seeds = [s for s in (_val("--infer-seeds", argv) or "0,1").split(",") if s.strip()]
    per_seed = {}
    for s in seeds:
        rj = work / f"roll_s{s}.json"
        if rj.exists():
            r = json.load(open(rj, encoding="utf-8"))
            per_seed[s] = {"pf_ok": bool((r.get("pf") or {}).get("ok")), "census": (r.get("pf") or {}).get("census")}
        else:
            per_seed[s] = {"pf_ok": False, "census": None, "note": "no roll record"}
    pf_ok = bool(per_seed) and all(v["pf_ok"] for v in per_seed.values())
    if sp.exists():
        j = json.load(open(sp, encoding="utf-8"))
        j["pf"] = {"wrapper": "run_battery_r8_pf.py", "wrapper_sha256": TP.sha256_file(__file__), "child": str(PF_CHILD),
                   "child_sha256": TP.sha256_file(PF_CHILD), "child_rewrites": REWRITES, "overlay": ov, "nav_feed": TP.NAV_FEED,
                   "per_seed": per_seed, "all_seeds_fed": pf_ok,
                   "spec": "SHIP-R8 AMENDMENT 3 (i): every clause read on rolls fed perframe_full",
                   "pose_sync_env": os.environ.get("R8_NO_POSE_SYNC")}
        json.dump(j, open(sp, "w", encoding="utf-8"), indent=1, default=str)
    print(f"[run_battery_r8_pf] tag={tag} all_seeds_fed={pf_ok} rewrites={len(REWRITES)} {'ZZBATTPFFEDZZ' if pf_ok else 'ZZBATTPFNOTFEDZZ'}", flush=True)
    return rc if rc else (0 if pf_ok else 3)


if __name__ == "__main__":
    sys.exit(main())
