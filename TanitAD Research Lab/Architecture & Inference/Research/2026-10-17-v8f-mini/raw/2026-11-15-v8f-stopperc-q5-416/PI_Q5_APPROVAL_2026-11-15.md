# PI answer to STOPPERC Q5 (2026-11-15)

**Question (card `PREREG_V8F_STOPPERC.md` section 6, Q5; put to the PI by the Master Mind on 2026-11-15 after S0-L read NO-GAIN, outcome PERC-NULL):** *"Approve compute for the 416x1024 re-encode (Q5), the card's next stop-cue lever?"*

**PI, verbatim:** *"Approve reencoding"*

**Context the PI had:**
* S0-L read NO-GAIN, STOPCUE dAUC +0.0095 [-0.0289, +0.0525] (landed 22edb8a6).
* The MM recommended declining, citing S0-P FLAT, FULLRES PROOF-14 and S0-L.
* The PI approved anyway.

**What this approves:** the compute for a 416x1024 re-encode: frames at 416x1024 and DINOv3-L tokens from them, on the pod and/or box 2.

**What it does not approve by itself:**
* The design of what is trained and read on the new source. That needs its own pre-registration (an amendment to the STOPPERC card) frozen before any 416x1024 data is scored.
* Any spend beyond the re-encode. A training arm's GPU time is named in that amendment with its cost; if it is large, it goes back to the PI.
