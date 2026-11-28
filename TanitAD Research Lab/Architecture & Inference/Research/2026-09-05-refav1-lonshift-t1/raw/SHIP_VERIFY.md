# P1 — ship-and-verify: the refav1 LON rig, reproduced on Thor

MEASURED 2026-09-05, all checks POSITIVE-ASSERTION (content, never presence).
Host `thor6` (NVIDIA Thor), venv `/home/nvidia/venvs/tanitad-edge`, torch 2.13.0+cu130.

## 0. ⛔⛔ THE BRIEF'S PREMISE WAS FALSE — the checkpoint on Thor is NOT ckpt 21109

The task brief stated Thor "already holds
`/home/nvidia/experiments/refav1-b1-v72-1ep-21109/ckpt.pt` (2,093,802,833 B — ckpt 21,109,
the exact checkpoint the longitudinal work used)". **It is not.**

| probe | Thor `…-1ep-21109/ckpt.pt` | dev box `refav1_eval_slice/ckpt_ep3/ckpt.pt` |
|---|---|---|
| size | 2,093,802,833 B | **2,122,997,633 B** |
| `ckpt["step"]` | **1000** | **21109** |
| state-dict keys | **387** | **407** |
| param numel | **175,166,517** | **182,459,701** |
| shape digest | `225d7e5b62e6a4618e18ea9cc7c584c8` | `9162549b602779e147e2e524a2d861e7` |
| content digest | `65ef540e5c4c90c90ddc99157616992c` | `f2ff324f440860e040ac1ff0089104e4` |

Corroborated independently by that run's own record: `train_log.jsonl` runs **step 1 → step 1000**
and stops, and `DRIFT_ALARM` fires at **step 950** ("tactical share OSCILLATING"). So the directory
holds a **halted 1,000-step run of a DIFFERENT architecture** (387 vs 407 param keys), and the
`21109` in its NAME is not its step.

⇒ **Launching from it would have produced a full four-family T1 panel of a different, barely-trained
model, and every number would have looked plausible.** The 29 MB size gap was the only tell, and it
is why a size check is run as a *lead*, never as a verification.

⭐ Same family as `MODEL_REGISTRY.md`'s founding case — *"`flagship4b-phase0-30k` is the no-speed
ablation control, NOT the deployed v1 … the HF repo name invites this inversion"* — and as the
`anchors.pt` units trap: **an artifact's NAME is not evidence about its CONTENT.**

## 1. What was shipped, and its md5 on BOTH sides

| artifact | dev box | Thor | verdict |
|---|---|---|---|
| `ckpt.pt` (step 21109, 2,122,997,633 B) | `1189bc020018c2c67ce03d566c390285` | `1189bc020018c2c67ce03d566c390285` | ✅ MATCH |
| `config.json` | `bd3ebed78eec2236156b176017745e2b` | `bd3ebed78eec2236156b176017745e2b` | ✅ MATCH |
| `code.tgz` (`stack/tanitad` + `taniteval`) | `16cdf3cd8e578a12d7cca95dac213225` | `16cdf3cd8e578a12d7cca95dac213225` | ✅ MATCH |
| labels `s2_labels_v7.2_eval.jsonl.gz` | `aa12c948f062181c3297265b51526ec5` | `aa12c948f062181c3297265b51526ec5` | ✅ MATCH (already on Thor) |
| `lon_emitted.py` | `43b685970fe0e2203916716d15602a24` | same | ✅ |
| `lon_oracle.py` | `f85edaf405aebaf858d0165bf6b25c8e` | same | ✅ |
| `lon_attribution.py` | `7abe3fcdd05ef0c1f1c7d943c25c1681` | same | ✅ |
| `queueTHOR.sh` | `bd4201dcc68d811c58aec61611c9dfa8` | same | ✅ |

⛔ **No git was used to move any of this.** Thor's checkout sits at `30d6d601` (weeks stale) and
Thor has no git credentials — a `fetch` hangs, and a failed `fetch` followed by `checkout -B` would
RESET the tree over the shipped files. Everything above arrived by `scp` and was md5-verified.

## 2. The 16 episode artifacts — size-identical on both hosts, 16/16

The eval slice is **8 episodes**. Thor already held both required forms; they were symlinked into a
Thor-local rig at `/home/nvidia/refav1_lon/p4/{fp8,eps}` rather than copied.

* fp8 tokens ← `/home/nvidia/data/dinov3-b1-fp8-w120-256x640cyl/<uuid>.pt`
* v2ep episodes ← `/home/nvidia/data/physicalai-b1-w120-256x640cyl/<uuid>.v2ep.pt`

| episode | fp8 B | v2ep B |
|---|---|---|
| `18b72291-84f4-45e0-a736-908fc303020e` | 66,193,268 | 37,799,055 |
| `24fee8a5-9ccc-4610-b27b-560f08fd46c8` | 66,193,268 | 39,263,951 |
| `af6f5964-2634-46e3-bbb1-ac2b194aec5f` | 66,193,268 | 45,227,535 |
| `dbad28c2-dfc1-4860-8c45-bd51760bc969` | 65,537,908 | 32,589,519 |
| `ebd51f52-29f6-4422-b9da-1a90cbbb48f2` | 66,193,268 | 37,817,039 |
| `f1e1536b-1e23-4d73-bbf9-492a54358d10` | 66,193,268 | 39,434,255 |
| `fe764229-15df-4506-b7e4-2977fd78e834` | 66,193,268 | 59,900,559 |
| `fe83e06e-3c2c-46a6-8b08-9f1a547cba1f` | 66,193,268 | 37,025,103 |

**Every one equals the dev box's byte count — 16/16.** (`dbad28c2`'s fp8 is legitimately smaller on
both hosts; an identical *anomaly* on both sides is corroboration, not a defect.)

## 3. Grep-verify the LEVER code is actually on Thor — before any launch

| probe | dev box `tanitad-wt` | repo `agent/arch-inf-20260803` | Thor shipped tree |
|---|---|---|---|
| `refa_v1.py` chars | 173,247 | 173,247 | **173,247** |
| `refav1_arm.py` chars | 162,595 | 162,595 | **162,595** |
| `refa_v1.py` hits `a_sustain|a_shift|jerk_seam_a0` | 39 | — | **39** |
| `refav1_arm.py` hits `a_sustain_mode|a0_shift|jerk_seam` | 16 | — | **16** |
| ⭐ SAME-BREATH CONTROL `grep -c 'def '` (must be NON-ZERO) | 58 | — | **58** |

The control is there because a `grep -c` of **0** from a file that could not be READ is
indistinguishable from a genuine absence.

## 4. Real import + CUDA, not `git log`

```
tanitad      -> /home/nvidia/refav1_lon/code/stack/tanitad/__init__.py
taniteval    -> /home/nvidia/refav1_lon/code/taniteval/taniteval/__init__.py
refa_v1      -> /home/nvidia/refav1_lon/code/stack/tanitad/refs/refa_v1.py
taniteval.ci -> /home/nvidia/refav1_lon/code/taniteval/taniteval/ci.py
```
— every import resolves into the **SHIPPED** tree, not Thor's stale `30d6d601` checkout.

```
RefAV1.plan has a_sustain    : True
RefAV1.plan has a_shift      : True
RefAV1.plan has jerk_seam_a0 : True
CONTROL (must be True) v0    : True
W_KAPPA = 0.05   GOAL_REACH_S = 2.0   GOAL_A_MAX = 1.5
conv2d CUDA OK: (1, 4, 30, 30)      matmul CUDA OK: (64, 64)
device: NVIDIA Thor
```
⛔ cuDNN **conv** was exercised, not merely `import torch` — cuBLAS can pass while conv is broken.
Device memory is read only via `torch.cuda.max_memory_allocated()` (8.58 MB at smoke); on Thor
`mem_get_info` / `free` / `tegrastats` / `VmRSS` are all inadmissible.

CLI surface confirmed present:
`--a-sustain-mode {none,a0,a0_shift}` · `--jerk-seam {off,a0}` · `--plan-seed` · `--cost-weights`.

## 5. ⭐ THE RIG CONTROL — Thor's arm header is line-for-line identical to the banked `wk15`

```
[preflight] 11 analysis imports OK
[refav1-labels] v7.2 join: labels 8/8 eps (0 missing), 168/535 windows in-band +-[2.0]s,
                md5=aa12c948f062181c3297265b51526ec5 | nav 8/8 eps ... allow_oracle_nav=True
[model] .../ckpt.pt step=21109 params=0 a_dim=2 target_space=frozen vocab=v7.0
[loader] 535 windows over 8 episodes (W=4, K_loader=30, grid 0.2 s); labels=ON nav=ON
[cost] metric=ccos weights={'W_JERK': 0.0, 'W_KAPPA': 15.11245, 'W_VEND': 64.2971504241507}
[grid] K=10 (2.0 s) K_wm=30 stride=16 windows=40 arms=['cl','ha','ha0','ha0_ext','ol']
       plan={samples 300, iters 30, elites 30, seed 0} nav_shuffle=23/40 changed
```
Every field — including `168/535 windows in-band`, `step=21109`, and `nav_shuffle=23/40` — matches
the banked dev-box `wk15` log exactly. Only the file PATHS differ.

## 6. ⛔ Why a Thor-local baseline is run anyway

The ckpt, labels, episodes and code are md5- or byte-identical, but **CEM on a different GPU is not
bit-reproducible**, so pairing a Thor arm against the dev box's `wk15` would be a CROSS-RIG
comparison. `T_wk15` is therefore arm 1; every lever is paired against **it**, window-for-window,
inside one rig — and `T_wk15` vs the dev box's `wk15` is read as a **RIG** control, never as a lever
result.

## 7. Parity

Episodes are the **8-episode eval slice**, unchanged from the banked baseline; nothing re-selected
episodes. `--window-stride 16` → **40 windows**, identical to `wk15`.
