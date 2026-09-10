<title>LEDGER A3 - vision encoders and visual representation learning</title>

# LEDGER A3 — Vision encoders / visual representation learning

`Frontier Scan track A3. APPEND-ONLY (charter §3). The running state of the art in this track, WITH OUR`
`POSITION IN IT. Never rewrite history — correct with a dated correction entry.`
`Why this track is Band A: it is our trunk, and our MEASURED ceiling.`

**Track opened 2026-09-05.** ⛔ **This ledger should have existed on 2026-08-31.** A3 is a Band-A track and
went **five days with no ledger and no DEEP read**, because every probe returned empty and an empty was
treated as an end state rather than as a query defect. Entry 1 is the correction.

---

## Entry 2026-09-05-01 — ⭐⭐ EMPTY **E1** IS RESOLVED AT THE FOURTH PROBE

`Revisiting Birds Eye View Perception Models with Frozen Foundation Models: DINOv2 and Metric3Dv2,`
`arXiv 2501.08118. PUBLISHED. Banked this pass. Cited by:`
`Frontier Scan/Daily/2026-09-05/RESULT.md F5.`

### The history this entry closes

| date | probe | outcome |
|---|---|---|
| 2026-08-31 | E1, first probe | EMPTY — returned medical/dental transfer benchmarks |
| 2026-09-01 | E1, second probe | frozen-DINOv3 dense-prediction SOTA found; **DINOv3 × BEV-driving still empty** |
| 2026-09-02 | E1, third probe | EMPTY. `TRACKS.md`: *"E1 now at THREE probes, still empty. Promote to the standing empty-search register"* |
| 2026-09-02 | next-rotation item 6 | *"A3 fourth probe **or formal retirement of E1** — three probes is enough to stop guessing"* |
| **2026-09-05** | **E1, FOURTH probe** | ⭐⭐ **RESOLVED** |

⛔ **It was one pass away from being retired as a real absence — and it was never absence.** All three
earlier probes led on the token **`DINOv3`**. The evidence lives under **`DINOv2` + `Metric3Dv2`**.

⭐ **The lesson is sharper than the two-probe rule that saved it: a term-specific empty is evidence about
the TERM, not about the world.** Same family as the `w120` identifier misread — *a token read as a
different quantity than it encodes* — and the same family as the standing empty-search register's purpose.
⇒ **Before an empty is promoted to standing absence, one probe must vary the TERM, not the phrasing.**
Proposed as row **FS5-6**.

### MEASURED (theirs, PUBLISHED — not our units)

| result | value |
|---|---|
| **frozen DINOv2** features in **Lift-Splat-Shoot** vs baseline | ⭐ **+7.4 IoU, using HALF the training data and HALF the iterations** |
| Metric3Dv2 depth as pseudo-LiDAR into Simple-BEV vs camera-only | **+3 IoU** |

Second hit, untriaged: `2409.10228` — *Robust Bird's Eye View Segmentation by Adapting DINOv2*.
Third: RAP-DINO `2510.04333`, 36.9 EPDMS on NAVSIM v2 (see `LEDGER_A5_benchmarks.md` 2026-09-05-01).

### ⭐ CONSEQUENCE — it moves the suspect off the trunk

Gate problem **P-5** records that L2 requires beating frozen DINOv3 on five targets (`n_agents`,
`n_free_cols`, `occ_{l,c,r}`) and **we beat it on ONE**. The natural reading was that our representation is
behind. **This result says the opposite is possible: a FROZEN DINO trunk with a proper BEV lift head beats
a trained baseline by a wide margin on half the data.**

⇒ **The trunk is not the suspect; the READOUT is.** That converges with two things we already hold:
- our own MEASURED v6 readout-geometry ceiling — 16×40 tokens pooled to 4×4 = **four azimuth bins over
  120° (30°/bin)**, 2.1–7.8× too coarse for BEV localisation;
- **GS-4** (explicit geometry loss on the v7 readout), whose committed read is *"if it does not move
  azimuth-bin decodability, the ceiling is RESOLUTION-bound and the readout must be re-architected"*.

⭐ **GS-4's priority should rise on this evidence, and P-5's framing should be re-read**: "behind on 4 of 5"
may be a **head** result, not a **representation** result. ⛔ It does not settle P-5 — it says the
experiment that would settle it is a readout experiment.

### CHANCES / RISKS

*Chance:* the cheapest possible resolution of our largest measured ceiling — a head change, not a trunk
change, on a trunk we are already committed to freezing.
*Risks:* ⛔ **(a) do NOT quote +7.4 IoU as a TanitAD-expected delta** — different architecture (LSS),
different corpus, different task (**BEV segmentation, not trajectory**). It is a **direction**, not a
magnitude. **(b)** LSS lifts with explicit depth supervision; our readout does not, so part of their gain
may be depth, not the frozen trunk — **their own +3 IoU Metric3Dv2 result is evidence that depth carries
real weight**. **(c)** abstract/summary-level read — full text not yet read.

### EXPERIMENT (0 GPU to design)

Read `2501.08118` §method for the **lift head** specification and compare it, component by component,
against our `readout.py` pooling. **Committed in advance:** if their head carries explicit geometry or
depth supervision that ours lacks, **GS-4 is the right next arm and P-5's "behind on 4 of 5" is
re-scoped as a head finding**; if the heads are comparable, the representation reading of P-5 stands and
GS-4 does not gain priority.

`Track debts: full text of 2501.08118 unread; 2409.10228 and 2510.04333 unbanked and untriaged.`
`⛔ E1 is RESOLVED and must be removed from the standing empty-search register's candidate list.`

## 2026-09-10-01 — E1 at a FIFTH probe, term varied again: still empty

`SCAN only.` Query: `arxiv 2026 vision encoder DINOv3 frozen BEV occupancy driving benchmark September` — term varied to `BEV occupancy`, replacing `dense prediction`. 8 hits, **none on target**: rainfall nowcasting `2511.10894`, diffusion policy `2509.17684`, Occ3D `2304.14365`, two occupancy surveys, DVGT `2512.16919`, multi-camera encoding `2512.10947`.

⛔ **E1 is now empty at FIVE probes across four passes, with the term varied twice** (FS5-6's requirement satisfied). This is approaching a genuine absence rather than a term artefact: **there is no published benchmark of frozen DINOv3 against BEV / occupancy driving tasks.**

⭐ **If that holds it is a positioning asset, not a gap in our reading.** Our measured readout-geometry ceiling — 16×40 tokens pooled to 4×4, i.e. **4 azimuth bins over 120°, 30°/bin** — is a number nobody else has published for this encoder class.

⚠️ **Not promoted to standing absence yet.** One more probe should vary the ENCODER term (drop `DINOv3` for `frozen self-supervised backbone`), per FS5-6's own lesson that the fourth probe succeeded by dropping the model name.
