"""restamp_banked_r8.py -- re-stamp the BANKED refcv8 milestone reads: label, read stamp and G0 stamp only.

    python restamp_banked_r8.py --pkg-raw <package raw/> --work-root C:/Users/Admin/refcv8_eval_archive/battery \
        --g0-root E:/refcv8_eval_kit/battery --tags step5000,step5000ema,step10500,...  [--dry-run]

Why (2026-10-18): `pair_read_r8.py` and `bank_battery_r8.py` hard-coded "step 5,000" into every title / stamp, and read G0 AS
CODED only -- so the step-10,500 / 15,000 / 20,000 reads were titled step 5,000 and stamped PROVISIONAL although their G0 is
PASS under the registered amendments (A7 + W-R8-1, and + W-R8-2 at 20,000; PI 2026-10-17).  Both tools are fixed; this applies
the fix to what is already banked.  EVERY NUMBER IS COPIED UNCHANGED: per pass it re-stamps the work copy of PAIRED_READ_*.json
(+ .md) through `pair_read_r8.py`'s own functions, asserts bars / surfaces / controls / inference_floor are byte-identical
before and after, then re-banks it through the bank's sanitiser, and patches BATTERY_SUMMARY_<tag>.json (tier, stamps,
passes.<p>.paired_read.g0) with a `restamped` record.  EMA tags carry no G0 of their own (NO_G0_REASON_<tag>.json): their G0
stamp is left as is and only the label / read stamp change.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bank_battery_r8 as B  # noqa: E402  (also puts the refcv7 battery code dir on sys.path)
import pair_read_r8 as PR  # noqa: E402
import sanitize_for_bank as S  # noqa: E402

KEEP = ("bars", "surfaces", "controls", "inference_floor")


def restamp_read(res: dict, g0_json, wrapfix_json, fp64_json, config_json) -> dict:
    before = {k: json.dumps(res.get(k), sort_keys=True) for k in KEEP}
    lab = PR.read_label(res.get("root") or "", config_json)
    res["label"] = lab
    res["stamps"] = [PR.read_stamp(lab) if str(x).startswith(("EARLY READ", "FINAL checkpoint", "step unknown")) else x
                     for x in res.get("stamps") or []]
    if g0_json:
        res["g0"] = PR.g0_stamp(g0_json, wrapfix_json, fp64_json)
    res["restamped"] = {"at": time.strftime("%FT%T"), "by": "restamp_banked_r8.py",
                        "what": "label, read stamp and G0 stamp only; every number unchanged"}
    after = {k: json.dumps(res.get(k), sort_keys=True) for k in KEEP}
    assert before == after, "a number changed under a restamp"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg-raw", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--g0-root", required=True)
    ap.add_argument("--config", default="E:/refcv8_eval_kit/ckpt/config.json")
    ap.add_argument("--tags", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    raw, work, g0r = Path(a.pkg_raw), Path(a.work_root), Path(a.g0_root)
    report = {}
    for tag in [t for t in a.tags.split(",") if t]:
        m = re.match(r"step(\d+)(ema)?$", tag)
        assert m, tag
        base = f"step{m.group(1)}"
        ema = bool(m.group(2))
        g0 = None if ema else str(g0r / base / "g0.json")
        wf, f64 = raw / base / "wrapper_clause_r8fix_gpu.json", raw / base / "wrapper_fp64" / "result.json"
        sp = raw / tag / f"BATTERY_SUMMARY_{tag}.json"
        summ = json.load(open(sp, encoding="utf-8"))
        rep = {"passes": {}}
        for ps, d in (summ.get("passes") or {}).items():
            pr = (d or {}).get("paired_read") or {}
            if not pr.get("file"):
                continue
            src = work / f"{tag}_{ps}" / pr["file"]
            if not src.exists():
                rep["passes"][ps] = "WORK COPY ABSENT: " + str(src)
                continue
            res = restamp_read(json.load(open(src, encoding="utf-8")), g0, str(wf), str(f64), a.config)
            g0s = res["g0"]
            rep["passes"][ps] = {"label": res["label"], "G0": g0s.get("G0"), "G0_amended": g0s.get("G0_amended"),
                                 "amendments": g0s.get("amendments"), "provisional": g0s.get("provisional")}
            if a.dry_run:
                continue
            json.dump(res, open(src, "w", encoding="utf-8"), indent=1, default=str)
            src.with_suffix(".md").write_text(PR.render_md(res), encoding="utf-8")
            dst = raw / tag / f"battery_{ps}"
            for f in (src, src.with_suffix(".md")):
                S.sanitize_file(f, dst / f.name)
            d["paired_read"]["g0"] = S._walk(g0s)
        tier, stamps = B.bank_stamps(tag)
        summ["tier"], summ["stamps"] = tier, stamps
        summ["restamped"] = {"at": time.strftime("%FT%T"), "by": "restamp_banked_r8.py",
                             "what": "tier, stamps and passes.<p>.paired_read.g0 only (the PAIRED_READ files were re-stamped "
                                     "the same way); every number unchanged"}
        rep["tier"] = tier
        if not a.dry_run:
            json.dump(summ, open(sp, "w", encoding="utf-8"), indent=1, default=str)
        report[tag] = rep
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
