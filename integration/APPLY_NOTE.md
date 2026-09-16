# `refc_wiring.patch` — how to apply it, and what it must not break

**Author:** refcv6 perception branch. **Cut 2026-09-16, REBASED 2026-09-17.**
**Target:** `stack/tanitad/refs/refc.py` at **`8c7d215`** (tip of
`agent/arch-inf-20260803`) — *"refcv6 core: ImageNet timm trunk with frame + ego
history at 256x1024, and all of F1-F9"*.
**Owner note:** ⛔ `refc.py`, `refc_sampler.py` and `refc_v3_train.py` belong to
another agent. This patch was built by editing a **copy** and diffing; the repo
file was never modified. The Master Mind applies it.

```
cd <worktree>
git apply --check integration/refc_wiring.patch      # verified clean at 8c7d215
git apply         integration/refc_wiring.patch
python integration/verify_patch.py                   # 5 checks, exit 0
```

⚠️ **The first cut was against `9a782fa` and does not apply.** `8c7d215` took
refc.py from 3,576 to 4,035 lines and moved three of the eleven anchors; this
file is the rebase. What changed, and why each mattered:

| the landed change | what it did to the patch |
|---|---|
| `RefCModel.forward` grew `ego_poses` / `ego_n_past` | the signature anchor moved — re-cut |
| the decoder call grew `ego_hist=ego_vec` | the threading anchor moved — re-cut |
| **F3/F4 split `_decode_ctrl` into TWO layer loops** | ⛔ the fast path (`cascade` and `adaln` both off) *and* the per-layer cascade path both need the coupling. Patching only the first would have left coupling (1) **silently dead in every F3/F4 arm** — which is the arm refcv6 is for |
| the timm trunk factory + `CNNEncoderConfig.s16_dim` | `d_bev` is now a required argument, and a **new edit 11** exposes the stride-16 map (below) |

## What it wires

`SPEC_REFCV6_V2.md` §1 lists three per-layer decoder reads. Only **(1)** is new.

| # | coupling | status |
|---|---|---|
| (1) | **BEV sampled at the candidate's own waypoints** | ⛔ did not exist — this patch |
| (2) | agent slots addressed by waypoint (**WP-B**) | ✅ already complete end to end |
| (3) | image tokens by content | ✅ already the decoder's `kv` |

⭐ **Coupling (2) needed no code.** `agent_pos` is already derived from
`agent_slots["box"][..., :2]` and threaded through `forward` → `_decode` /
`_decode_ctrl` → `_agent_index` → `CrossAttnLayer._agent_bias`, with
`WaypointIndexBias`'s output layer zero-init and `attach_wp_index` refusing
without the agent seam. WP-B is *"never run"* because no arm switched it on,
**not** because a wire is missing. One asymmetry is pre-existing and documented
in place: the `diffusion_steps` refinement pass is agent-free on purpose, so
WP-B does not reach it. This patch does not change that.

### ⭐ Correction C5 is applied

Three comments in `refc.py` called **WP-B** *"DiffusionDrive coupling (1)"*. The
spec numbers it **(2)**; (1) is the BEV sampler. A reader who trusted the old
wording concluded the BEV seam already existed — it did not. All three sites are
corrected by this patch (no behaviour change), so the code and the programme
record agree.

### ⭐ New in the rebase: the stride-16 perception seam (edit 11)

`TimmResNetTrunk.forward_features` returns `(s16, s32, pooled)` and
`TimmResNetTrunk.forward` **throws `s16` away**. refcv6's whole perception
branch reads stride 16 — an oracle on stride 32 caps at **AP 0.3341 vs 0.4713**
(spec §2) — so without this seam a caller must run the 21.8–45 M backbone a
second time for a map the trunk already built.

Edit 11 calls `forward_features` when the trunk offers it and emits the result
as `out["fmap_s16"]`, on **both** forward branches. ⚠️ An earlier cut wired only
the last-frame branch and called the hierarchy branch "reported rather than
half-wired" — but `refc_smoke_config()` and every hierarchy arm set
`hierarchy = True`, so the seam would have been **dead on the path actually
taken**. MEASURED after the fix:

| trunk | `hierarchy=False` | `hierarchy=True` |
|---|---|---|
| `resnet34.a1_in1k` | `(2, 256, 16, 64)` | `(2, 256, 16, 64)` |
| `resnet101.a1_in1k` | `(2, 1024, 16, 64)` | `(2, 1024, 16, 64)` |

— the PI's 16×64 geometry, with the channel count from `feature_info` via
`cfg.encoder.s16_dim`, never a literal. `None` on the in-repo REF-C trunk, whose
`s16_dim` raises rather than inventing a width.

## The anchors, by line number in `8c7d215`

| # | anchor | line | edit |
|---|---|---|---|
| 1 | `from tanitad.refs import refc_wp_index as wpi` | **167** | add the `refc_bev_coupling` import |
| C5 | the three "coupling (1)" comments | **164**, **2080**, **3331** | correct to (2) |
| 2 | `wp_index: "object \| None" = None` (`DecoderConfig`) | **490** | add `bev_coupling` + `bev_coupling_d_bev` |
| 3 | `self.wp_index: nn.Module \| None = None` | **1355** | add `self.bev_wp` |
| 4 | `CrossAttnLayer.forward` + its `_attend_agents` block | **1357**, **1369** | add `bev`, `waypoints`; the gated call |
| 5 | `def wp_index_params` | **1940** | insert `attach_bev_coupling` / `bev_coupling_params` / `bev_coupling_provenance` **before** it |
| 6 | `_decode` signature + its layer loop | **2130**, **2163** | thread `bev`, `x_est` |
| 7 | `_decode_ctrl` signature + **BOTH** layer loops | **2171**, **2202**, **2212** | thread `bev`, `x_path` |
| 8 | `_sample` signature + its `_decode_ctrl` call | — | thread `bev` |
| 9 | `AnchoredDiffusionDecoder.forward` + its `_decode` call | — | thread `bev` |
| 10 | `attach_wp_index(_wp)` (`RefCModel.__init__`) | **3331** | attach the BEV coupling **immediately after** |
| 11 | `b, w = frames.shape[:2]` + the `out = {...}` dict | **3580**, **3878** | expose `fmap_s16` |
| 12 | `RefCModel.forward` signature + the decoder call | **3545**, **3871** | accept and pass `bev` |

## ⛔ The one ordering rule that must not be broken

`refc.py` states that `attach_wp_index` is the **last statement of
`RefCModel.__init__`**, and that this *is* WP-B's removability proof: every
module above draws from the global RNG in a fixed order, so an `index on`
build's shared parameters are bit-identical to an `index off` build's.

This patch inserts `attach_bev_coupling` **after** it — the two attachments are
now the last two statements, **in that order**. Deliberately:

* everything above both is still bit-identical, so WP-B's proof survives;
* **and** WP-B's heads are drawn *before* the BEV sampler, so turning the BEV
  coupling on leaves every WP-B head bit-identical too — which is what makes the
  2×2 ablation (WP-B on/off) × (BEV on/off) clean.

⛔ **Nothing may be added below these two lines.**

## What was MEASURED on the patched file (CPU, at `8c7d215`)

`python integration/verify_patch.py` — it copies `refc.py` into a temp tree,
applies the patch there, and imports both modules in one process. It embeds no
machine- or session-specific path; `--repo` / `--patch` / `--tmp` override.

| check | result |
|---|---|
| `git apply --check` at `8c7d215` | **clean** |
| A. coupling off: parameter set + values vs the UNPATCHED module | 136 tensors, **0 differ** |
| B. forward on **64 fixed windows**, coupling off | **64/64 BIT-IDENTICAL** |
| C. coupling on | +3,466 params / 2 layers; `provenance = ddv2-faithful`; `n_points = 4 = n_steps`; shared params still bit-identical |
| D. gates at init | `[0.0, 0.0]`; gate 0 → exact identity, gate 1 → output changes |
| E. WP-B heads across BEV on/off (`cross_agent=True`) | 8 tensors, **0 differ** |

### The repo's own suites, against the APPLIED patch

Run in a throwaway copy of `stack/` with the patch applied:

| suite | result |
|---|---|
| `test_wp_index.py` (incl. `test_shared_params_bit_identical`) | **35 passed** |
| `test_refcv6_diffusion.py` + `test_refcv6_trunk.py` | included in **108 passed, 1 skipped** |
| the broad REF-C set (`test_refc`, `_sampler`, `_agents`, `_select`, `_selector`, `_tactical`, `refcv6_tactical`, `wp_index`, `refcv6_diffusion`, `refcv6_trunk`) | see RESULT §5 |

### Both sampler loops reach the coupling

Gate 0 vs gate 1 through `_decode_ctrl`, all four F3/F4 settings:

| `f3_per_layer` | `f4_adaln` | loop | max\|Δ\| conf |
|---|---|---|---|
| 0 | 0 | FAST | 1.0406e-01 |
| 1 | 0 | CASCADE | 4.4307e-01 |
| 0 | 1 | CASCADE | 1.0763e-01 |
| 1 | 1 | CASCADE | 3.8681e-01 |

⚠️ `max|Δ| du` is **0.0000e+00** in every row and that is correct, not a null
result: `control_head` is zero-init, so the first pass predicts exactly the
current state whatever the queries say. The coupling shows up in **conf**.
Reporting only `du` would read as "live, delta 0" — true, and wrong for the
reader.

## What the applier still has to supply

The patch adds the **seam**, not the data flow into it. To run an arm:

1. build the BEV features from `out["fmap_s16"]` —
   `bev_lift.build_lift_geometry` → `BEVLift` → `bev_encoder.BEVMapBranch`
   (`out["bev_feats"]`, `[B, d_bev, 120, 64]`);
2. set `cfg.decoder.bev_coupling = BEVCouplingConfig(d_bev=<that width>)` **and**
   `cfg.decoder.bev_coupling_d_bev = <that width>`;
3. pass `bev=out["bev_feats"]` to `RefCModel.forward`.

⚠️ `d_bev` must equal the BEV encoder's `d_out` and the BEV grid must be 120×64;
both are asserted in `BEVWaypointSampler.forward`, not assumed. ⛔ Neither is a
backbone constant — the stride-16 width upstream of them is 1024 on `resnet101`
and 256 on `resnet34`, read from `feature_info`.

⚠️ `learned_offsets` defaults to **False**, which is the released DiffusionDrive
(`…/ddv2_src/blocks.py:80-108`: the sample positions are the waypoints
themselves; only the softmax over them is learned). Switching it on makes the
arm **our extension** and `bev_coupling_provenance()` says
`"tanitad-extension"` — stamp it into `config.json` and label the arm so.

## Line endings: nothing to do

The repo runs `core.autocrlf=true` with no `.gitattributes`. All three forms
were tested against the real `refc.py` and each applies cleanly: as generated,
as the staged (LF) blob, and as the post-checkout (CRLF) form. No
`--ignore-whitespace` is needed. If it is ever rejected, that is a real content
conflict, not an EOL artefact — do not force past it.
