# 2026-07-07 — comma2k19 real-data validation, H7 pipeline deltas, contract reconciliation

**Agent:** Data Engineering (Tuesday). **Run:** W1→W2, iteration 1 of 3.
**Budget used:** ~4 web searches / ~1.5 h wall-clock. **Quality:** full (G-A…G-F, G-D1, G-D2 met).
**Consumed:** Monday's Tools&DevEnv note (2026-07-06), DECISIONS D-009, my own `_contract` refactor.

Every claim below carries a source link or a repo-path / measured-command reference (G-A).

---

## 1. Situation on entry (honest state note — P8)

The backlog top item is "comma2k19 ingestion module". On entry I found it **already delivered and
committed by Sayed** as **D-009** (`stack/tanitad/data/comma2k19.py` + `tests/test_comma2k19.py`),
landed mid-session (repo advanced from `8994220` to `e1bf4d5` during my run). D-009 deliberately **forks
the frames contract** to `base250cam` = 6-channel 2-frame RGB stacks @ 256 px (vs the toy/MetaDrive
single-channel `[T,1,H,W]` BEV). I therefore did **not** re-implement or overwrite it. My work this week:
(a) **validate** D-009 on real data, (b) **reconcile** the forked contract into the shared contract home,
(c) close integration/test gaps, (d) research the H7 pipeline deltas. Backlog item #1's required **data
card** (which was missing) is now written: `2026-07-07-comma2k19-data-card.md`.

## 2. Real-data validation of the D-009 loader (implementation increment)

Using the OS-trust TLS fix + secret loader from Monday (`tanitad.keys.enable_tls/load_keys`) and the
git-ignored `Keys.txt` HF token, I pulled ONE real segment's `video.hevc` (37.5 MB) and drove the
loader's **own** decode path on it (not a mock):

- `_decode_video` → `[200,3,256,256]` uint8 in **1.9 s (~105 fps)** via `av` on py3.13/Windows;
  `stack_two_frames` → `[199,6,256,256]`. The `base250cam` pipeline is real-byte-correct end to end.
- **A8 (real highway) `frame_change_fraction` = 0.053 @0.05 / 0.012 @0.10** — see §4.
- **Windows `|` path** independently confirmed (`WinError 123` on a raw-layout `hf_hub_download`): route
  dirs `dongle|date` are illegal on Win32. This is **already handled** by D-009's
  `stack/scripts/extract_comma2k19.py` (`|`→`_`); the lesson is "always go through that extractor, never
  `extractall`/raw-path download". Data card §6.1.

No secrets were logged or committed (`Keys.txt` is git-ignored; verified out of the commit). This
de-risks D-009 before the A40 spend: the previously-deferred, codec-dependent decode path now has a
green real-data run behind it.

## 3. H7 pipeline deltas (research focus)

H7 = 1000× data leverage from action-free video via inverse-dynamics (IDM) pseudo-labels + focal
canonicalization. New/confirmed since kickoff:

- **LAOF — latent actions from optical flow** (Bu et al., 20 Nov 2025). Uses optical-flow
  pseudo-supervision to bias latent actions toward *true agent motion* and away from appearance
  distractors; strong gains in **label-sparse regimes** — exactly our comma2k19→BDD100K/OpenDV transfer.
  Actionable: when we pseudo-label action-free video, add a flow-consistency term to the IDM loss.
  [LAM overview](https://www.emergentmind.com/topics/latent-action-model).
- **Sensorimotor World Models: Perception for Action via Inverse Dynamics** (arXiv
  [2606.20104](https://arxiv.org/pdf/2606.20104)). Frames IDM as the perception objective for a world
  model — same seed-IDM role our inverse-dynamics head plays. Supports treating comma2k19's *real*
  (frame, action) pairs as the IDM's calibration/supervision anchor before unleashing it on unlabeled
  video.
- **LAPA** confirmed as the reference recipe: VQ latent-action codebook between adjacent frames, then a
  light decoder to real controls with minimal labels ([LAPA arXiv](https://arxiv.org/html/2410.11758v1)).
- **Focal canonicalization (VLM3, f≈1000)** unchanged from kickoff (no 2026 delta found this run); stays
  the Phase-1 heterogeneous-video unifier. comma2k19 is single-camera/single-intrinsic, so canonicalize
  is a **no-op for Chunk-1** and only bites when GoPro/YouTube/BDD enter — deprioritize until then.

**H7 concrete path (updated).** comma2k19 gives real (steering, accel) at 20 Hz → train the seed IDM on
these real pairs, **log the steering-ratio calibration residual** (loader constant 15.3 is v0), then
pseudo-label action-free corpora with an added **optical-flow-consistency term (LAOF)** and epistemic
top-15 %-uncertain discard. The steering-ratio log is the single cheapest H7 calibration artifact and is
now a named backlog item.

## 4. Finding: A8 is weak on raw highway RGB → change-weighting is empirically justified

Measured real-highway `frame_change_fraction ≈ 0.053` (@0.05) is ~1.7× the toy BEV floor and far below
the "tens of %" the ego-centric BEV toy hits by construction. Interpretation: on a forward-facing highway
camera, most pixels (sky, road surface, distant background) are low-texture and near-static per 10 Hz
step; the consequence of an action lives in a **small, high-information image region**. Implications:

- The world-model reconstruction/prediction loss should be **change-weighted** (weight residual pixels),
  not flat MSE — otherwise the loss is dominated by trivially-predictable static background. This directly
  informs the **W2 Stage-0 bake-off** (residual + change-weighted vs plain MSE) already on the plan.
- The D-009 **2-frame stack** is well-motivated: it puts motion inside a single input tensor, raising the
  effective consequence signal. Kept.
- Recommend a **per-dataset A8 statistics harness** (backlog item #4) to report `frame_change_fraction`
  distribution per corpus, so the change-weighting schedule is set from data, not guessed. (H3/A8.)

## 5. Reconciliation: the forked frames contract is now explicit, not drift (increment)

D-009's `[T,6,S,S]` diverged from the shared `[T,1,H,W]` contract. Rather than silently allow two
contracts, I generalized the single shared assertion
`tanitad.data._contract.assert_contract(ep, channels=1|6|None)` (additive; default `1` keeps
toy/MetaDrive unchanged) and added `tests/test_comma2k19_contract.py` asserting the D-009 loader validates
at `channels=6` **and** clears the A8 floor on structured motion. So the fork is a *typed, tested*
contract variant with one home, not an untracked deviation. Full suite: **40 passed, 1 skipped** (the
MetaDrive live test). `comma2k19` is now exported from `tanitad.data.__init__`.

## 6. Actionable recommendations (tied to hypotheses / gates)

- **[H7, new backlog]** Log the **steering-ratio calibration residual** per segment when the seed IDM
  trains on comma2k19 — cheapest H7 calibration artifact; blocks the headline data-efficiency claim.
- **[W2 bake-off, H3/A8]** Use **change-weighted** reconstruction loss; the real A8≈0.05 makes flat MSE a
  weak baseline on highway RGB. Feed §4 numbers into the Stage-0 bake-off design.
- **[Ops, before A40 spend]** Pull Chunk_1 and extract **via `scripts/extract_comma2k19.py`** (never raw
  `extractall` — the `|`→`_` rewrite is mandatory on Windows and harmless on Linux); add `av` to the
  `[real]` extra; smoke `build_episode` on 3 segments. ~1–2 engineer-hours, zero new code.
- **[Backlog #4 next]** Build the per-dataset A8 statistics harness (`frame_change_fraction` distribution)
  so change-weighting is data-driven.
- **[H4, my hypothesis]** Frozen-encoder arm B is still open and cheap — schedule it against D1–D3 once
  Chunk_1 is on the pod (real data makes the frozen-vs-trained encoder comparison meaningful).

## 7. Sources

- comma2k19: [arXiv 1812.05752](https://arxiv.org/abs/1812.05752) ·
  [commaai/comma2k19 (MIT)](https://github.com/commaai/comma2k19) · HF `commaai/comma2k19` (measured, ungated)
- Steering ratio: [comma-steering-control](https://github.com/commaai/comma-steering-control) ·
  [openpilot #599](https://github.com/commaai/openpilot/issues/599)
- H7: [LAPA](https://arxiv.org/html/2410.11758v1) ·
  [Sensorimotor World Models](https://arxiv.org/pdf/2606.20104) ·
  [Latent Action Model overview (LAOF)](https://www.emergentmind.com/topics/latent-action-model)
- Repo: `stack/tanitad/data/comma2k19.py` (D-009), `stack/tanitad/data/_contract.py`,
  `stack/tests/test_comma2k19_contract.py`, `2026-07-07-comma2k19-data-card.md`
