"""The Master Mind's W-freezing helper (stdlib only, no network). Reads cl_prereg/W_PROBE.json and proposes W = level_head_probe.w_level_for_10pct_share rounded to 4 significant figures.
  python -I freeze_w.py              print the proposal and the freeze line to paste into PREREG_LEVELHEAD_CL.md (writes nothing)
  python -I freeze_w.py --write      write cl_prereg/FROZEN_W.txt (refuses to overwrite a different frozen value)
After --write: paste the freeze line into the card, then `python -I mk_cl_package.py` (regenerates argv, pins and the checker with this W and the card's new sha256), then `python -I mk_cl_package.py check`."""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
PRE = H.parent / "cl_prereg"


def main(a):
    wp = PRE / "W_PROBE.json"
    if not wp.exists():
        print(f"{wp} is absent: run the probe and pull it first"); return 1
    rec = json.loads(wp.read_text(encoding="utf-8"))
    raw = float(rec["w_level_for_10pct_share"])
    if not raw > 0:
        print(f"w_level_for_10pct_share {raw} is not > 0"); return 1
    w = f"{raw:.4g}"
    line = (f"**FROZEN W = {w}** (2026-11-05 launch: `cl_prereg/FROZEN_W.txt`; the W probe's level_head_probe.w_level_for_10pct_share = {raw!r}, selfcheck sha256 {rec['selfcheck_sha256']}, "
            f"step-0 L_without_level {rec.get('step0_total_without_level')!r}, L_level {rec.get('step0_level_term')!r}; the level term is a tenth of the total at step 0)")
    print(f"proposal: W = {w}  (raw {raw!r})\nfreeze line for the card:\n{line}")
    if "--write" in a:
        fw = PRE / "FROZEN_W.txt"
        if fw.exists():
            old = [l for l in fw.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
            if old and old[0].replace("W=", "").strip() != w:
                print(f"{fw} already freezes {old[0]!r}: not overwritten (move it aside by hand to re-freeze)"); return 1
        fw.write_text(f"W={w}\n# from W_PROBE.json: w_level_for_10pct_share {raw!r}, selfcheck sha256 {rec['selfcheck_sha256']}\n", encoding="utf-8", newline="\n")
        print(f"wrote {fw}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
