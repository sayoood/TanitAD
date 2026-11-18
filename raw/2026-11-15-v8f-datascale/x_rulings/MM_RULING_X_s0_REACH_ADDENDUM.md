# MM addendum to MM_RULINGS_X_s0.md item 2 (2026-11-18 ~08:20Z, X_s0 stage A running; no X_s0 outcome read)

**What happened.** Ruling 2 (2026-11-18 ~00:20Z) required FRESH G-LIVE reach evidence for X_s0: "a probe borrowed from another arm is INHERITED evidence and may not open a GPU run". The ruling was written in this folder and never wired into the chain.
* The pod's `blockx_prep.sh:32` still takes the tenk record and control from `gate/probe2/UE_s0.A.json` (2026-11-17 21:57).
* The prep ran 07:56:03 to 07:58:58Z: dry build, `init_x` PASS with its regression arm RED, evidence, token. Stage A started at 07:59:32Z.
* This is the "escalate integration, don't write it into a doc" failure class: a ruling that lived in a markdown file and not in the code that launches.

**What fresh reach evidence exists, MEASURED on X_s0's own built model and real data** (`/root/dscale10k/runs/X_s0/A/selfcheck.json`, written by the trainer at launch):
* `declared_vs_built` PASS.
* `live.pass` true. Every declared term is present and finite, with no missing and no unexpected term: goal 0.477, lat 0.338, lon 1.447, cons 0.460, ade, fde, acc, pens, wm.
* All 7 trainable groups have a non-zero gradient: adapter, S, T 803.7, O, head_s, head_o, wm.
* `unexpected_dead` [], `declared_dead_but_alive` [], `n_trainable` 252 (B_s0's count).

**Ruling.**
* The trainer's own launch selfcheck on X_s0's real data is FRESH reach evidence at the term and group level. That is the question ruling 2 asked ("reach is the first question for this arm"). X_s0 continues.
* What stays INHERITED: probe2's per-source record (the 15 VLM goal bits zero-gradient on 10k rows). It is a property of the code and the release, both identical to UE_s0's (same commit, same tenk release md5 `395221440e3a3c20f17dd381297af4f0`). X has no corpus batch for the probe's control.
* It is reported as INHERITED wherever X_s0 is quoted.
* **Process fix, binding for the next arm (UM_s0):** a ruling that changes a launch condition is implemented in the chain or the gate before the arm is queued, and the queue step greps for it. A ruling in a markdown file is not a gate.
