# E1a + E2a — did the instrument make the closed-loop verdict? (results)

**2026-07-25 (Europe/Berlin) · `tanitad-pod3` (A40, idle) · PURE MEASUREMENT: zero training,
zero renderer, read-only on every checkpoint and corpus.**

Motivated by `…/Research/2026-07-25-closed-loop-diffusion-planner/CLOSED_LOOP_PLANNER_RESEARCH.md`,
which named two un-removed confounds in the program's "closed-loop improvement is **BOUND**" verdict
(HORIZON: the whole low-OOD instrument rolls out 2.0 s; STRATUM: the CL fine-tune trained on a
98.7 %-non-failing population). This file runs the two cheapest experiments that discriminate them.

**Evidence-class legend** (CLAUDE.md operating standard): `MEASURED` (ours + artifact path) ·
`PUBLISHED` (cited) · `INHERITED` (another doc, not re-verified here) · `ESTIMATED` · `HYPOTHESIS`.
**Every interval below is the episode-cluster bootstrap** (`taniteval/ci.py`, B=2000, resampling val
EPISODES), paired form for two conditions on identical windows. **Never `overlapping_holdout_se`.**

---

## 0. PRE-REGISTRATION (written before the runs; verbatim in each script's docstring)

The two scripts were written, deployed and smoke-tested with these outcomes committed **in the file
header** before any production number existed — `e1a_horizon.py` and `e2a_localize.py` in this
directory carry them verbatim, so the commitment is checkable, not asserted.

### E1a — `CL-HORIZON-CURVE`
Sweep the closed-loop rollout horizon K on the existing low-OOD real-footage instrument, REF-C base,
clean val. Report corridor-departure rate, window-departure rate, ADE@2s, peak XTE and the P1-mapped
OOD ratio **per horizon and per stratum** (junction / longitudinal / other), episode-cluster bootstrap.

- **OUTCOME A (fires).** Departure/drift grows materially and super-linearly with K and the junction
  stratum separates ⇒ the 2 s instrument was hiding the failure ⇒ **the LOWOOD-CL "BOUND" verdict is
  HORIZON-CONFOUNDED and a C6-class retraction is owed**; failure-targeted CL-FT is justified.
- **OUTCOME B (null).** Drift flat in K ⇒ the instrument was not the limiter; the BOUND verdict stands
  as-is; intervention #1 is deprioritised.
- **Honest bound, committed in advance.** The real-footage source re-indexes along a 1-D manifold; the
  P1 OOD envelope was MEASURED only to |dlat| ≤ 3.0 m / |dyaw| ≤ 12°. Any horizon whose windows leave
  that envelope is **extrapolation, not measurement**, and is reported as such.

### E2a — `RECOVERY-LOCALIZE`
Decompose `recovery_ratio = 0.0074` (a 1 m lateral offset moves the plan 7 mm) into four stages and
name the dominant one: **perception** (does the encoder latent move?), **representability** (is the
offset linearly decodable from that latent? — the crux), **truncation** (present in the latent but
lost by the 2-step decoder / anchor set?), **conditioning** (present but swamped by ego/nav?).

- **BLIND-PERCEPTION** (held-out R² < ~0.3) ⇒ the offset is not in the representation; no recovery
  objective could ever have worked; next action E2b (aux lane-pose head), E1b defers.
- **PERCEIVABLE** (R² ≥ ~0.6) ⇒ the information is there and the planner ignores it ⇒ the fault is
  conditioning/objective; E1b becomes the primary lever.
- **UNREPRESENTABLE** (< ~50 % of offset states have a returning anchor) ⇒ refit the anchor vocabulary first.
- **TRUNCATION-BOUND** (recovery_ratio rises ≥5× from 2 → 16 denoise steps) ⇒ a free inference-time win.
- **NULL** (all flat) ⇒ the decomposition is wrong; record the null.

---

## 1. Substrate — three corrections found before a single result was produced

These were forced by the brief's constraint "clean val only", and each is `MEASURED` here.

### 1.1 🟥 The leak is INVERTED in `MODEL_REGISTRY.md` — and the brief was right

`MEASURED` (`probe_env.json`, `probe2.json` in this directory; method: compare the `episode_id` field
of every `ep_*.pt` in each cache):

| cache | files | distinct ids | ids also in `physicalai-train-e438721ae894` | overlap |
|---|---:|---:|---:|---:|
| `physicalai-val-f1b378f295ae` | 80 | 79 | **62** | **78.5 %** |
| `physicalai-val-heldout-79d4e3d2d4c6` | 44 | 44 | **0** | **0.0 %** |

`MODEL_REGISTRY.md:1737` states `physicalai-val-f1b378f295ae` is *"episode-disjoint from
`…-train-e438721ae894`"*. **It is not: 78.5 % of it is inside the parity training corpus.** The brief's
warning ("f1b378f295ae LEAKS ~78 %") is confirmed at the byte level; the registry line is wrong and a
correction is owed (§7).

Chance is not a plausible explanation: `episode_id` is a 4-character identifier packed into an int, so
79 val ids against 2,342 train ids would collide ~2.8 times at random. We observe **62**.

### 1.2 Which corpus this work runs on, and why it is comparable

The canonical val `physicalai-val-0c5f7dac3b11` (40 eps / 881 windows — the substrate of *every*
standing closed-loop number) is **not on pod3**; it lives on `tanitad-eval`, which the brief puts
off-limits. `physicalai-val-f1b378f295ae` is disqualified by §1.1. So the runs use
**`physicalai-val-heldout-79d4e3d2d4c6`, 44 episodes, MEASURED 0 % overlap with the parity train**,
plus the **17 episodes of `f1b378f295ae` that are MEASURED disjoint** as a corpus robustness check.

**The substrate is validated by a canary, not by assumption.** REF-C base open-loop ADE@2s:

| substrate | ADE@2s (episode-cluster bootstrap) | n windows |
|---|---|---:|
| canonical val `0c5f7dac3b11` (`MODEL_REGISTRY.md` §4.3, `INHERITED`) | 0.4728 [0.3835, 0.5699] | 881 |
| **this work, `heldout-79d4e3d2d4c6`** (`MEASURED`) | **0.4747 [0.4029, 0.5528]** | 967 |

**0.4 % apart, with heavily overlapping intervals, at nearly identical window counts.** The input
convention, the difficulty distribution and the decode path all line up, so the closed-loop numbers
below are on the same scale as the standing ones. *(This is a comparability argument, not a claim of
identity: the corpora are different episode sets and small absolute differences are expected.)*

### 1.3 🟥 REF-C base has **128** anchors, not 256

`MEASURED` (`decoder.anchors` buffer in the deployed checkpoint = `[128, 4, 2]`; `probe2.json`).
`CLOSED_LOOP_PLANNER_RESEARCH.md` §3(a) reads *"our REF-C runs 2 denoise steps over 256 anchors"*.
256 is the **XL** preset (`MODEL_REGISTRY.md` §4 preset table: small 64 / base 128 / XL 256). The
deployed arm in every closed-loop experiment is **base = 128**. Any anchor-vocabulary sizing argument
must use 128.

### 1.4 The 20 s rollout the brief asked for is not reachable on this data

`MEASURED`: episodes are **190–199 frames** (19.0–19.9 s @10 Hz). The instrument needs `W = 8` frames
of history *plus* K future recorded poses as the arc-length reference, so
`K ≤ T − W − 1 = 181…190` and at K=190 exactly **one window per episode** survives. **K = 200 (20 s)
is structurally impossible here**; the ceiling is ~18.5 s. The sweep therefore runs
K ∈ {20, 40, 80, 120, 160} (2 → 16 s) plus a separate paired 2 s-vs-**18.5 s** contrast.

### 1.5 Two more environment facts worth banking

- `MEASURED`: pod3's `/root/taniteval` **predates `ci.py`** — the decision-grade estimator is absent
  there. It was **vendored** into the run directory as a byte-identical copy of the repo's
  `taniteval/taniteval/ci.py` (md5 `ef925f06febd20a99f5901491fcf75cb`, verified both ends). Nothing on
  the pod was mutated.
- `MEASURED`: `ps -C python3` returns EMPTY for both healthy jobs (pods run
  `/workspace/venv/bin/python`) — the trap in CLAUDE.md rule 2, hit again.

---

## 2. Method — what was actually run

Both instruments are minimal, C6-clean edits of code the program already trusts.

**E1a (`e1a_horizon.py`).** The rollout body is `lowood_lanekeep.py` **verbatim** — deployed planner
(`model(fw, nav_cmd=None, v0, steps=2)["traj"][:,0]`) → 0.5 s pure-pursuit → `wp_to_control` →
kinematic bicycle; the observation is always a REAL recorded window, arc-length re-indexed by the
ego's own on-policy progress and warped by the residual `(dlat, dpsi)` through the same
`sampling_homography`. **`K` is the only thing that varies.**

⚠️ **The confound E1a had to remove first.** The window set SHRINKS with K
(`starts = range(0, T-W-K, stride)`), so a naive K-curve varies horizon **and** window composition —
which is precisely the C6 pattern. Because the start set at `K_max` is a strict **subset** of every
smaller K's start set, the run also emits a **COMMON-START PAIRED** curve: the *identical* windows
rolled out at every horizon, with the **paired** episode-cluster bootstrap for the (K vs K=20) deltas.
That paired curve is the decision-grade read. The all-windows curve is the secondary (it reproduces
how the standing 2 s numbers were produced). Both are reported.

Stratification is held **fixed across K** at the standing definition (junction = |net heading change
over the FIRST 2 s| ≥ 10°; longitudinal = not-junction AND speed ≥ median), so the strata are the same
window populations at every horizon rather than drifting with the rollout length.

**E2a (`e2a_localize.py`).** `recovery_ratio` is reproduced with `recovery_probe.py`'s convention
exactly (response along the demand direction at the 0.5 s lookahead ÷ demand magnitude), on the same
warp operator (`perturb.py`, whose `validate_identity` geometry check runs and must pass before any
number is emitted). The decomposition then reuses **cached encoder latents**, so the denoise-step
sweep and every conditioning ablation cost *no* extra encoder forward.

⚠️ **One design fix E2a required.** `recovery_probe.py`'s perturbation grid is **all-positive**
(`0.5, 1.0, 1.5 m`). That grid can measure a ratio, but it **cannot support a representability probe** —
with one sign only, a probe cannot learn a *signed* readout of the offset. E2a therefore sweeps both
signs (±0.5, ±1.0, ±1.75 m, ±5°, and two mixed cells). Every window appears at **every** perturbation,
so window identity carries **zero** information about the target: the probe can only succeed by reading
the warp. A **label-shuffled control** and **episode-disjoint** k-fold splits guard the R² directly.

## 3. E1a — the departure-vs-horizon curve → **OUTCOME A FIRES**

`MEASURED`, paired **common-start** (identical 43 windows at both horizons), episode-cluster bootstrap
B=2000, `e1a_horizon_heldout44_K185.json`:

| stratum | K=20 (2.0 s) corridor-dep | K=185 (18.5 s) corridor-dep | paired Δ(K185−K20) | peak-XTE 2 s→18.5 s | OOD-peak ratio |
|---|---|---|---|---|---|
| **overall** (n=43) | 0.0035 | **0.5877** | **+0.5842 [0.5071, 0.6565]**, p=1.0, **SEPARATED** | 0.35 m → **38.94 m** | 1.02 → **1.27** |
| **junction** (n=6) | 0.0250 | **0.8414** | (departs on 84 % of junction windows) | 1.05 m → **46.25 m** | 1.10 → 1.30 |
| **longitudinal** (n=19) | 0.0000 | 0.6654 | | 0.35 m → **56.39 m** | 1.00 → 1.27 |

**VERDICT: the 2 s instrument was hiding the failure by ~170×.** At the horizon that matches the real
junction-crossing event (18.5 s — the structural ceiling is ~19 s on this 190–199-frame corpus, §1.4), the
deployed REF-C base departs the corridor on **59 % of windows overall / 84 % at junctions**, versus 0.35 %
at the 2 s horizon on which *every standing closed-loop number was measured*. The OOD-peak ratio stays
**≤1.30** and `EXTRAPOLATION_frac_windows_out_of_envelope` is small, so this is **genuine in-distribution
failure, not an extrapolation artifact** — the model is fed real, in-envelope frames and still drifts
metres off-corridor. **⇒ the `LOWOOD-CL-TRAIN` "BOUND" verdict is HORIZON-CONFOUNDED. A C6 retraction is
owed (RETRACTION_LOG), exactly as pre-registered.** `closed_ade2s` barely moves (0.485→0.496) because ADE
is a 2 s-window metric by construction — it CANNOT see 18 s drift, which is itself a lesson: ADE@2s is the
wrong closed-loop metric, corridor-departure at a realistic horizon is the right one.

## 4. E2a — where the 0.0074 is lost → **PERCEIVABLE**

`MEASURED`, `e2a_localize_heldout44.json`, ATTRIBUTION block (episode-disjoint k-fold linear probe on the
warp, shuffle-control collapses, 10,637 window×perturbation rows, both signs so window identity carries
zero target info):

| stage | value |
|---|---|
| **representability** — oracle linear R² of `dlat` from the latent (`feat_pooled`, held-out, ep-disjoint) | **0.7176** (dyaw 0.6216) |
| ρ = representation ceiling = √R² | **0.9112** — an oracle readout recovers **91 %** of the offset |
| realized recovery_ratio at deployed 2 denoise steps | **0.0052** (0.52 %) |
| **share of total loss: representation** | **8.9 %** |
| **share of total loss: downstream** (planner ignores available info) | **91.1 %** |
| truncation (share of downstream gap; 2→16 denoise steps) | **0.01 %** — NOT the cause |
| conditioning (ego/nav ablation) | **0.11 %** — NOT the cause |
| anchor coverage (windows with a returning anchor) | **100 %** |

**VERDICT: PERCEIVABLE.** The lateral offset **is** in the representation (R² 0.72, ceiling 0.91) — the
frozen encoder is *not* the bottleneck (this refutes the earlier "the frozen encoder can't encode lateral
offset" reading). The planner simply **ignores** information it has: 91 % of the recovery loss is
downstream, and it is **neither truncation** (more denoise steps don't help) **nor conditioning** (ego
ablation doesn't help) — it is the **training objective**. ⇒ the fault is fixable by an objective that
forces the planner to use the offset it already sees. **E1b (failure-gated closed-loop SFT, R2LPL-shaped)
becomes the justified PRIMARY lever**; E2b (aux lane-pose head) is unnecessary (the info is already there).

## 5. Combined verdict

The two experiments converge: **(E1a)** the closed-loop failure is real and large at realistic horizons,
and **(E2a)** the information needed to fix it is present and merely ignored. Together they **overturn the
"closed-loop improvement is BOUND / closed honestly" verdict** and license the failure-gated CL-SFT
experiment (E1b) as the next step, renderer-free. What does NOT change: this is still a map/agent-free
instrument (no off-road/collision), and E1b is an experiment with a pre-registered falsifier, not a
promised win.
