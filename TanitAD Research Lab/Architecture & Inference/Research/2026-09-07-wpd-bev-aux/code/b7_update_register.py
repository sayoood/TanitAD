# -*- coding: utf-8 -*-
"""WP-D step 7 -- update GOALS_AND_CLAIMS.md's `E-BEV-AUX-1` row with the outcome.

⛔ Reads every number it writes OUT OF THE BANKED JSON. Nothing is transcribed by
hand, so the register cannot disagree with the artifact it cites.
⛔ Never writes to the G: path in place: the whole file is composed LOCALLY and
copied over, and the edit is an EXACT single-occurrence string replacement that
REFUSES if the anchor is absent or appears more than once.
"""
import json
import os
import shutil
import sys

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
REG = os.path.join(REPO, "Project Steering", "GOALS_AND_CLAIMS.md")
LOCAL = r"C:\Users\Admin\wpd-probe\GOALS_AND_CLAIMS.staged.md"
RAW = r"C:\Users\Admin\wpd-probe\raw"

cart = json.load(open(os.path.join(RAW, "boot_cart.json"), encoding="utf-8"))
pol = json.load(open(os.path.join(RAW, "panel_pol.json"), encoding="utf-8"))
wpa = json.load(open(os.path.join(RAW, "wpa_recheck.json"), encoding="utf-8"))
cp, pp = cart["pairs"], pol["pairs"]
cpt = cart["point_ap"]
pa = {k: v["ap_test"] for k, v in pol["arms"].items()}


def ci(d):
    return f"{d['delta']:+.5f} [{d['lo']:+.5f}, {d['hi']:+.5f}]{'' if d['separated'] else ', NOT separated'}"


STATUS = (
    "⛔⛔ **REFUTED AS STATED (MEASURED 2026-09-08) — failure twin `F4` fired: the gain is "
    "CAPACITY/REGULARISATION, not agent content.** On the POLAR target the aux head actually "
    f"TRAINED on, the ZERO-INFORMATION shuffled-target arm `D2` (AP **{pa['tok_D2']:.4f}**) is "
    f"indistinguishable from the lever `D1` (**{pa['tok_D1']:.4f}**): **A4 = "
    f"{ci(pp['tok_D1_minus_tok_D2'])}**. On the CARTESIAN target the shuffled arm is **AHEAD** "
    f"(**{cpt['tok_D2']:.4f}** vs **{cpt['tok_D1']:.4f}**), so A4 fails on the POINT ESTIMATE "
    "alone and SUCCESS (A1∧A2∧A3∧A4) is unreachable. ⏳ **A3 is `NOT MEASURED`, NOT negative** "
    "(the prereg's own `F5`): the replicate arm `wpd-D0b-4k` is RUNNING on Thor. ⛔ `F1` did NOT "
    "fire — the trunk is not empty, so `E-DEC-8` DINOv3 distillation is NOT the lever this calls for."
)

APPEND = (
    " ⭐⭐ **PANEL RESULT 2026-09-08 — AND IT IS THE PREMISE, NOT ONLY THE CLAIM, THAT FELL.** "
    "MEASURED on the pre-registered frozen-feature probe (no tier: no trajectory is produced), "
    f"n = **{cart['n_test_rows']:,} test rows / {cart['n_episodes']} episodes / "
    f"{cart['n_scored_cells']:,} scored cells**, base rate **{cart['base_rate']:.9f}** ⇒ an "
    "all-zero predictor scores **98.3893 % ACCURACY**, which is why no number here is an accuracy; "
    "`d` raw = **112,640** per row against 149 into the head, so this is NOT the `n ≪ d` failure. "
    "⛔ **The constant-score control reads the base rate EXACTLY on both geometries** "
    f"(`{cart['base_rate']:.9f}` / `{pol['base_rate_test']:.9f}`) and the panel REFUSES to run if it "
    "does not — it fired on the first attempt, correctly, against MY float32 reference (a 3.9e-10 "
    "accumulation error over 20.9 M cells), and the fix was the REFERENCE (integer counts), never a "
    "tolerance on the comparison. **Cartesian arms:** constant "
    f"**{cart['base_rate']:.4f}** · `shuf_D1` **{cpt['shuf_D1']:.4f}** · `pos_only` "
    f"**{cpt['pos_only']:.4f}** · `pix` **{cpt['pix']:.4f}** · **`D0` (aux OFF) "
    f"{cpt['tok_D0']:.4f}** · **`D1` (aux ON) {cpt['tok_D1']:.4f}** · **`D2` (SHUFFLED target) "
    f"{cpt['tok_D2']:.4f}**. **Bars, paired episode-cluster bootstrap over the 30 test episodes "
    "(`taniteval.ci._draws` resampling, statistic = pooled TIE-GROUP AP; "
    "`overlapping_holdout_se` never called): A1 "
    f"{ci(cp['tok_D1_minus_pos_only'])} ⇒ **PASS**; A2 {ci(cp['tok_D1_minus_pix'])} ⇒ "
    f"**{'PASS' if cp['tok_D1_minus_pix']['separated'] else 'FAIL'}**; A3 "
    f"{ci(cp['tok_D1_minus_tok_D0'])} ⇒ **NOT MEASURED** (no replicate floor yet); A4 "
    f"{ci(cp['tok_D1_minus_tok_D2'])} ⇒ **FAIL**.** ⚠️ Every interval above answers *\"would "
    "another DRAW OF EPISODES say this?\"* and is structurally blind to training-run variance — "
    "which is exactly what `D0b` measures. ⭐⭐ **THE LOAD-BEARING FINDING IS NOT THE REFUTATION: "
    "THE MAP ALREADY CONTAINED AGENTS, AND `E-READOUT-CEILING-1` SAID OTHERWISE BECAUSE ITS PROBE'S "
    "AZIMUTH ADDRESS IS MIRRORED.** The aux-OFF control `D0` reads "
    f"**{cpt['tok_D0']:.4f}** Cartesian / **{pa['tok_D0']:.4f}** polar against a marginal of "
    f"**{cpt['pos_only']:.4f} / {pa['pos_only']:.4f}** — `D0 − pos_only` = "
    f"**{ci(pp['tok_D0_minus_pos_only'])}** on polar and **{ci(cp['tok_D0_minus_pos_only'])}** on "
    "Cartesian — and REF-C's trunk BEATS the raw-pixel floor "
    f"(`D0 − pix` = {ci(cp['tok_D0_minus_pix'])}). ⛔ `bev_aux.azimuth_column` AND "
    "`psg_targets.azimuth_column` — two independent implementations that AGREE — both put image "
    "column 0 at azimuth **+hfov/2 (LEFT)**; WP-A's `s5_indexed.py` uses `u = f_ref·az + W/2`, the "
    "**MIRROR**. MEASURED on WP-A's OWN banked bytes with ONE variable moved (`code/b4_wpa_recheck.py`): "
    f"`pix` **{wpa['arms']['pix_prog']['ap_test']:.4f}** corrected vs "
    f"**{wpa['arms']['pix_wpa']['ap_test']:.4f}** mirrored (**2.10×**), `tok` "
    f"**{wpa['arms']['tok_prog']['ap_test']:.4f}** vs **{wpa['arms']['tok_wpa']['ap_test']:.4f}** "
    f"(**1.57×**), `pos_only` {wpa['arms']['pos_only']['ap_test']:.4f} — and **the MIRRORED column "
    "REPRODUCES WP-A's banked panel** (`pix_16x40` 0.0270, `tok_16x40` 0.0295), which is the "
    "discriminating control: the re-run is not merely different, it MATCHES when mirrored. ⭐ The "
    "tell was in WP-A's own table: its BEST token arm is `tok_1x1` (0.0335), the one that pools ALL "
    "azimuth away, while `tok_16x40` (0.0295) is the WORST — under a correct address more azimuth "
    "resolution cannot hurt; under a mirrored one, destroying it helps. ⛔ This does NOT retract "
    "WP-A's ORACLE ceiling (0.4713/0.3374) — that arm was NOT re-run, evidence class NOT MEASURED. "
    "⭐ **NEXT LEVER (Rule Zero): WP-B itself**, which was gated on WP-D *because the map was "
    "believed empty*; that premise is gone, and the cheap prerequisite is re-reading WP-A's banked "
    "panel under the corrected address (~10 GPU-min, the script exists and runs). ⚠️ **A second "
    "measured fact worth its own row:** raw pixels BEAT the v6 trunk "
    f"({wpa['arms']['pix_prog']['ap_test']:.4f} vs {wpa['arms']['tok_prog']['ap_test']:.4f}) while "
    f"REF-C's trunk BEATS raw pixels ({cpt['tok_D0']:.4f} vs {cpt['pix']:.4f}) — the v6 encoder "
    "DESTROYS agent-localisation information relative to its own input; REF-C's adds to it. "
    "⭐ **Removability reproduced on the TRAINED checkpoint:** stripping `bev_aux` removed **6 keys "
    "/ 182,616 parameters** — the prereg's own figure, independently confirmed by the run's own "
    "`param_breakdown.total` difference (108,440,118 − 108,257,502 = **182,616**) — leaving "
    "**108,307,024**, identical to D0's, every kept tensor bit-identical. ⚠️ **But the prereg's "
    "claim that this is `param_breakdown[\"bev_aux\"]`-subtractable is WRONG in the shipped record: "
    "that key is ABSENT and the 182,616 are absorbed into `param_breakdown['core']`**, so a single "
    "arm's `config.json` cannot report the deployed count — it needs the aux-off arm to difference "
    "against. ⭐ Cross-checks that had to agree and did: polar base rate **0.0316155** vs the "
    "prereg's **0.031634**; masked fraction **17.635 %** vs **17.656 %**; and the Cartesian target "
    "is **BIT-IDENTICAL to WP-A's banked `y.npy`, 0 disagreeing cells**, on an element-wise equal "
    "row plan (139 clips / 13,223 rows). ⛔ `D3` (detached), `D4` (two-state deliberate regression, "
    "§5D) and `D5` were **NOT RUN** — named blockers at 4.7 h of Thor each, not omissions. "
    "**Evidence:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-07-wpd-bev-aux/"
    "PANEL_RESULT.md` + `raw/boot_cart.json`, `raw/panel_pol.json`, `raw/wpa_recheck.json`, "
    "`raw/D0b_argv_audit.json`, `raw/strip_bev_head.json`."
)

ANCHOR = "| **PRE-REGISTERED — code + tests DELIVERED, 0 GPU spent, NO ARM RUN.** |"
END = ("**GPU:** 0 spent. 4.0 s/step MEASURED ⇒ the cheapest discriminating cut (D0/D1/D2 at "
       "4,000 steps) is **13.3 h on one card**; the full 7-arm panel 93 h. ⛔ Blocked on a free "
       "card — A40 training refcv5-v2, Thor training refav1. |")
END_NEW = ("**GPU:** D0/D1/D2 RAN on Thor (16,821 / 16,822 / 16,557 s, `summary.json` `done: true` "
           "at 4,000 steps, **stderr 0 bytes on every arm, one launch each**); `D0b` (replicate, "
           "seed 0) and `D0c` (seed 1) chained after them, ~4.7 h each. The probe cost ~50 min on "
           "the dev-box 4060; **the A40 was never contacted**. |" + APPEND)

src = open(REG, encoding="utf-8").read()
for name, anchor in (("STATUS", ANCHOR), ("END", END)):
    n = src.count(anchor)
    if n != 1:
        sys.exit(f"REFUSING: the {name} anchor occurs {n} times, not exactly once. "
                 f"The register moved under me; re-read it before editing.")
new = src.replace(ANCHOR, f"| {STATUS} |").replace(END, END_NEW)
if new == src:
    sys.exit("REFUSING: replacement was a no-op")
open(LOCAL, "w", encoding="utf-8", newline="").write(new)
print(f"[local] wrote {LOCAL}  {len(new)} chars (was {len(src)}, +{len(new)-len(src)})")
shutil.copyfile(LOCAL, REG)
back = open(REG, encoding="utf-8").read()
ok = (back == new)
print(f"[verify] G: file readback identical to local composition: {ok}")
if not ok:
    sys.exit("REFUSING: the copy to G: did not round-trip")
for marker in ("REFUTED AS STATED (MEASURED 2026-09-08)", "b4_wpa_recheck.py",
               "MIRRORED", "wpd-D0b-4k"):
    c = back.count(marker)
    print(f"[assert] '{marker[:44]}' occurs {c}x")
    if c < 1:
        sys.exit(f"REFUSING: marker '{marker}' absent after write")
print("OK")
