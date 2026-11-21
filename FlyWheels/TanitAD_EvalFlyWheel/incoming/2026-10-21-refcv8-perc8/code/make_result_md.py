"""make_result_md.py -- compose RESULT.md from PERC8_EVAL.json, the table files and a hand-written READING.md.
python make_result_md.py --root <perc8 dir>"""
import argparse, json, os, sys


def rd(p, default=""):
    return open(p, encoding="utf-8").read() if os.path.exists(p) else default


def jd(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    R = a.root
    T = f"{R}/raw/tables"
    E = jd(f"{R}/PERC8_EVAL.json")
    parts = [rd(f"{R}/code/RESULT_HEAD.md"), "\n## 1. Controls (each must read its known value)\n", rd(f"{T}/controls.md"),
             "\n## 2. Box read: refcv8 EMA-48,852 vs refcv7@50,400, declared protocol (S2 windows, unshifted pose clock)\n", rd(f"{T}/tbl_box_ap.md"),
             "\n### Census at the DECLARED gate (sigma >= 0.5)\n", rd(f"{T}/tbl_census_declared_gate_0.5.md"),
             "\n## 3. Map read: 10 cm IoU per class x distance band\n", rd(f"{T}/tbl_map.md"),
             "\n## 4. Zero-training inference fixes, fitted on TRAIN-DIAG only, frozen, re-scored on EVAL\n", rd(f"{T}/refit.md"),
             "\n### Census at the TRAIN P=R gate, no NMS\n", rd(f"{T}/tbl_census_train_PR_gate_noNMS.md"),
             "\n### Census with the TRAIN NMS + its re-fitted gate\n", rd(f"{T}/tbl_census_train_NMS_plus_gate.md"),
             "\n### Within-arm effect of the fixes (same arm, same draws)\n", rd(f"{T}/within.md"),
             "\n## 5. WiSE decoder-only blend (DEC) as a control\n", rd(f"{T}/wise.md"),
             "\n## 6. Reading\n", rd(f"{R}/code/READING.md"),
             "\n## 7. Deliverable manifest\n", rd(f"{R}/code/MANIFEST.md")]
    open(f"{R}/RESULT.md", "w", encoding="utf-8").write("\n".join(parts))
    print("wrote", f"{R}/RESULT.md")


if __name__ == "__main__":
    sys.exit(main())
