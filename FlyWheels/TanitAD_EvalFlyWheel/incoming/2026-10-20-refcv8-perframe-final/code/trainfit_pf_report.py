"""trainfit_pf_report.py -- writes TRAINFIT_PF_FROZEN.md (the registration sheet of the per-frame RETIME table) from the frozen file + its provenance.

NEW file (EvalFlyWheel 2026-10-20, hold_r8pf).  Reads only; the sha256 is RE-COMPUTED from the frozen file (never copied from a record) and must equal the
.sha256 record, else the sheet says INCONSISTENT and carries no hash to register.  No raw clip id is read or written.

    python trainfit_pf_report.py --tf-dir <hold>/raw/retime_trainfit_pf --out <hold>/TRAINFIT_PF_FROZEN.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fmt(v) -> str:
    return "[" + ", ".join(f"{float(x):.4f}" for x in v) + "]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = Path(a.tf_dir)
    fz_p, rec_p, prov_p = d / "FROZEN_GAINS_RETIME_train_pf.json", d / "FROZEN_GAINS_RETIME_train_pf.sha256", d / "FROZEN_GAINS_RETIME_train_pf.provenance.json"
    sh = sha256_file(fz_p)
    rec = rec_p.read_text().split()[0].strip()
    prov = json.load(open(prov_p, encoding="utf-8"))
    fz = json.load(open(fz_p, encoding="utf-8"))
    ok = len(sh) == 64 and sh == rec and prov.get("sha256") == sh
    L = ["# TRAINFIT_PF_FROZEN -- the per-frame RETIME table (SHIP-R8 AMENDMENT 3 (ii))", ""]
    if not ok:
        L += ["**INCONSISTENT** -- the file, its .sha256 record and its provenance do not agree; NOTHING here may be registered.",
              f"file {sh} | record {rec} | provenance {prov.get('sha256')}"]
    else:
        L += [f"**sha256 to register: `{sh}`**", "",
              f"* file: `{fz_p}` ({fz_p.stat().st_size} B); record `{rec_p.name}` equals the file (re-computed at report time)",
              "* evidence class: MEASURED (fit on our rolls; T1 open loop; single training seed; inference seeds 0 and 1 pooled)",
              f"* procedure: H-R8-RETIME-TRAINFIT-1, UNCHANGED fit code `retime_trainfit_r8.py` sha256 `{prov['fit_code']['sha256']}` "
              f"(the clip-fed freeze ran `{prov['fit_code'].get('sha256_at_clip_fed_freeze')}`; diff = {prov['fit_code'].get('diff_vs_clip_fed_freeze')}); "
              f"PREREG_TRAINFIT.md sha256 `{prov['fit_code']['prereg_sha256']}`",
              f"* tree: `{fz.get('tree')}` (EMA-30k, `--nav-feed perframe_full`), seeds pooled {fz.get('seeds_pooled')}, prior `{fz.get('prior')}`",
              f"* fit guard: {json.dumps(prov.get('fit_guard'))}",
              f"* windows by surface: {fz.get('n_windows_by_surface')}; ceiling-compliant: {fz.get('n_compliant_by_surface')}",
              "", "## Feed parity of the train rolls (clause (1b) form, asserted on the roll artifacts; independent of the overlay)", ""]
        for r in prov.get("feed_parity", []):
            L.append(f"* {r.get('label')}: **{r.get('status')}** -- {json.dumps(r.get('counts'))} (file sha256 `{r.get('sha256')}`)")
        L += ["", "## Rolls", ""]
        for s, r in sorted((prov.get("rolls") or {}).items()):
            L.append(f"* {s}: {r.get('n_windows')} windows / {r.get('n_episodes')} episodes, device {r.get('device')}, wall {r.get('wall_s')} s, "
                     f"pf.ok {r.get('pf_ok')}, overlay sha256 `{r.get('overlay_sha256')}`, record sha256 `{r.get('sha256')}`")
        L += ["", "## Gains (all windows, `RETIME`) -- per-frame (SCORED once registered) vs clip-fed (BESIDE ONLY)", "",
              "| surface | instants s | per-frame (this table) | clip-fed b208c9c1... (beside) |", "|---|---|---|---|"]
        for s in ("S2", "S6"):
            g, c = prov["gains_pf"][s], prov["beside_clip_fed_table"]["gains"][s]
            L.append(f"| {s} | {g['instants_s']} | {fmt(g['RETIME'])} | {fmt(c['RETIME'])} |")
        L += ["", "Amendment 1's deployed form (knots S2 0.5-2 s, S6 3-6 s, piecewise-linear, held outside) is DERIVED from this file at forward time by "
              "`deploy_gains_pf.py`; no second fit exists.", "",
              "## Gate", "", "The 48,852 per-frame eval rolls REFUSE to start unless `C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256` exists "
              "and equals the sha256 above (the Master Mind writes it after registering this hash in GOALS_AND_CLAIMS)."]
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out} consistent={ok} sha256={sh}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
