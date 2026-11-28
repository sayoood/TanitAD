# PREREG_V8F_PROOF_ADD36_N1 : IMAGLEAD, what the build of the read tool fixed, before any data

Note to card `PREREG_V8F_PROOF_ADD36.md`. Frozen after the tool `imaghead` was built and tested on the dev box (CPU, tiny random models and synthetic dumps only) and before it has run on any real checkpoint: no head read of any imagined state exists. The card, its names and its controls stay as written; this note makes five points precise and states one part of control c2 that cannot be evaluated as written.

## 1. The tool (MEASURED on the dev box)

* `imaghead/imag_dump.py` sha256 f9fa166c24a3d793df06ccf84d3dc43505575a90cada4faaa58ad4452b1d20e9 (pod side), `imag_read.py` sha256 cc9b12ae0c680cb110dd690969905000e141393ea32d168df6c0f432e5fa0198 (dev side), `test_imaghead.py` sha256 3c628c6efdd2e7721790b631f9c012fd0278f51dc58b35fe532c27c733959894 (31 tests pass, run twice: by the builder and by the lead); definitions in `imaghead/IMAGHEAD_DEFS.md`.
* Every control c1 to c5 has a deliberate regression that makes it fail (offset path that adds a constant at 0; shifted actions and an off-by-one head window; wrong sign, wrong time scaling and wrong units of the offset, which read a slope of 2.0; an un-zeroed copy; identity permutation, no substitution, an uninformative true future).
* The 11 pinned files of the source tree equal `spatial/leadsup/PINS.txt`.
* Measured on labels and banked files only (no model): 534 of the 1,948 windows have a lead at t; 441 have a lead at t and at t + 2.0 s and equal the windows of the r-b read (label difference 0.0); 356 pass the H2 rule (gap 5 to 50 m, ego speed at least 5 m/s), of them 146 with a gap below 20 m. The ego-only floor read is bit-identical across arms, so the file of base arm G of the same seed is used.

## 2. Points made precise

* **Start of the rollout.** The card says "from the true states up to frame t". The tool calls the stack's own rollout from the state of frame t alone, the context growing by its own predictions: the form the rollout loss trains (card ADD18) and the r-b read uses. No true history is given to the predictor.
* **Inputs.** The dump reads the eval state file of `extract_states.py` (the states of the checkpoint on the eval clips), the window list of the r-b read and the label pack of the LEADSUP read; it does not rebuild states from the cache.
* **Units of the offset.** The acceleration channel enters the predictor divided by 2.0 (`v8f_mini.py:201` and `:947`; the banked scaled actions equal the raw ones over 2.0 with difference 0.0); an offset d in m/s2 is added as d / 2.0 at every step.
* **Readings of the rules.** "The two-level interval lies above 0" = the lower bound of the pooled two-level interval is above 0. In control c5, "H1 lies above 0" with the true future read given as the rollout = above 1e-9 at every seed (a numerical zero is not above 0). The head reads are compared in metres (gap = output x 50).
* **Where the dump runs.** On the pod CPU while three GPU jobs train (no fourth GPU job is started). Controls c1 and c4 need exactly 0.0; a miss at any size is reported as a miss.

## 3. Control c2, as it can be evaluated

The card: "on the logged actions the rollout of the tool equals the rollout of the stack (the banked rollout files of the r-b read) and the head on true states equals the trainer's own head output, each to 1e-5."

* (a) At dump time, on the real checkpoint: the tool's rollout against the trainer's own rollout core (`roll_pairs`, the function inside the rollout loss), and the head on the tool's five-frame slices against the head on the whole window sequence as the trainer calls it; each at most 1e-5 in the trainer's units.
* (b) Against a banked file: the head read on the true states at frame t against the banked head outputs of the LEADSUP read (`<ARM>_eval_lead.npz`, written by `lead_dump.py` for card ADD24; for RL written by the same script once the arm is trained), matched by clip and frame; at most 1e-5 in the trainer's units (5e-4 m on the gap, 1e-4 m/s on the closing speed, 1e-5 on the presence logit).
* (c) The banked rollout files of the r-b read do not exist for LS or RL (the r-b extraction was never run on a lead-head arm). This part is reported as not evaluated. Where an r-b extraction of an arm exists later, the tool's option `--banked-roll` is run on it and the value is reported.
* The names of the card are given for an arm only if (a) and (b) hold for it, together with c1, c3, c4, c5 (and c6, c7 where they apply); (c) is stated beside every name as not evaluated.

## 4. State of the arms at this freeze

LS_s0 and LS_s1 exist (their eval state files and head outputs are banked on the pod). The dry gate of RL_s0 passed (declared equals built for the levers of both sets, live step finite); chain `imag_a` waits behind chain `motion_e`; no RL arm has trained a step.
