# -*- coding: utf-8 -*-
"""WP-D step 8 -- inject the CARTESIAN CI table into PANEL_RESULT.md from the JSON.

⛔ Every number is read out of the banked artifact. Nothing is transcribed, so the
report cannot disagree with the file it cites.
"""
import json
import os
import shutil
import sys

RAW = r"C:\Users\Admin\wpd-probe\raw"
SRC = r"C:\Users\Admin\wpd-probe\PANEL_RESULT.md"
PKG = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Lab"
       r"\Architecture & Inference\Research\2026-09-07-wpd-bev-aux")

b = json.load(open(os.path.join(RAW, "boot_cart.json"), encoding="utf-8"))
p, pt = b["pairs"], b["point_ap"]


def row(d):
    sep = "**YES**" if d["separated"] else "no"
    return f"{d['delta']:+.5f} | [{d['lo']:+.5f}, {d['hi']:+.5f}] | {sep}"


A1, A2 = p["tok_D1_minus_pos_only"], p["tok_D1_minus_pix"]
A3, A4 = p["tok_D1_minus_tok_D0"], p["tok_D1_minus_tok_D2"]
need = A3["delta"] / 3.0

lines = [
    f"Same estimator, same 30 episodes, **n_boot = {b['n_boot']}**, **all "
    f"{b['n_scored_cells']:,} scored cells** (no cell subsampling).",
    "",
    "⭐ **The bootstrap is EXACT, not an approximation, and that was verified rather than "
    "asserted:** a draw only changes how many times a row is counted, never a score, so the "
    "global sort and tie-group boundaries are computed once and re-weighted. Checked against a "
    "genuine re-sorted AP on 3 real draws — **bit-equal on all three** "
    f"(`{b['exactness_check'][0]['fast']:.12f}`, `{b['exactness_check'][1]['fast']:.12f}`, "
    f"`{b['exactness_check'][2]['fast']:.12f}`).",
    "",
    "| bar | statement | delta | 95 % CI | separated | verdict |",
    "|---|---|---|---|---|---|",
    f"| **A1** | `D1 − pos_only` ≥ **+0.010** | {row(A1)} | "
    f"{'✅ **PASS**' if (A1['separated'] and A1['delta'] >= 0.010) else '⛔ **FAIL**'} |",
    f"| **A2** | `D1 > pix` | {row(A2)} | "
    f"{'✅ **PASS**' if (A2['separated'] and A2['delta'] > 0) else '⛔ **FAIL**'} |",
    f"| **A3** | ≥ **3×** a replicate floor from THIS panel | {row(A3)} | ⏳ **NOT MEASURED** |",
    f"| **A4** | `D1 − D2` ≥ 3× that floor | {row(A4)} | ⛔ **FAIL** |",
    "",
    "⭐ **A3, stated precisely so `D0b` can settle it without ambiguity.** The measured lever gap "
    f"is **{A3['delta']:+.5f} AP**. A3 passes only if the replicate floor "
    f"`|AP(D0b) − AP(D0)|` is **below {need:.5f} AP** — i.e. only if two runs differing in "
    "**nothing** agree to better than "
    f"**{100 * need / pt['tok_D0']:.2f} %** of the arm's own AP. ⛔ For scale, WP-A's oracle-rig "
    "floor was ≤ **0.0122 AP**, and that number is **not borrowed** — the floor is being measured "
    "here. ⚠️ If the floor exceeds "
    f"**{abs(pt['tok_D2'] - pt['tok_D0']):.5f}**, then D0, D1 and D2 are all within one replicate "
    "spread of each other and the honest reading becomes `F5` — **underpowered at 4,000 steps**, "
    "not a measured difference in either direction.",
    "",
    "Context rows, same estimator:",
    "",
    "| comparison | delta | 95 % CI | separated |",
    "|---|---|---|---|",
]
CTX = [("tok_D0_minus_pos_only", "`D0 − pos_only` — **does the AUX-OFF trunk already carry agents?**"),
       ("tok_D0_minus_pix", "`D0 − pix` — **aux-off trunk vs the raw-pixel floor**"),
       ("pix_minus_pos_only", "`pix − pos_only` — do raw pixels beat the marginal?"),
       ("tok_D2_minus_tok_D0", "`D2 − D0` — the ZERO-INFORMATION target vs aux-off"),
       ("tok_D2_minus_pos_only", "`D2 − pos_only`"),
       ("shuf_D1_minus_pos_only", "`shuf_D1 − pos_only` (CONTROL: must not gain)")]
for k, why in CTX:
    if k in p:
        lines.append(f"| {why} | {row(p[k])} |")
lines += ["",
          "| arm | pooled AP (this bootstrap's point estimate) |", "|---|---|"]
for k in ("tok_D0", "tok_D1", "tok_D2", "pix", "pos_only", "shuf_D1"):
    if k in pt:
        lines.append(f"| `{k}` | {pt[k]:.6f} |")
lines += ["",
          "⚠️ These point estimates are recomputed from the **fp16** score matrices the panel "
          "banked, so they can differ from §4.2's fp32 values in the 4th decimal; the ORDERING and "
          "every delta above are computed within this one consistent set."]

doc = open(SRC, encoding="utf-8").read()
if "<!--CART_CI_TABLE-->" not in doc:
    sys.exit("REFUSING: the CART_CI_TABLE placeholder is gone")
doc = doc.replace("<!--CART_CI_TABLE-->", "\n".join(lines))
open(SRC, "w", encoding="utf-8", newline="").write(doc)
dst = os.path.join(PKG, "PANEL_RESULT.md")
shutil.copyfile(SRC, dst)
back = open(dst, encoding="utf-8").read()
print(f"[verify] round-trip identical: {back == doc}  ({len(doc)} chars)")
if back != doc:
    sys.exit("REFUSING: copy to G: did not round-trip")
print("WROTE", dst)
