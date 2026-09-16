# refcv6 — the specification the PI approved on 2026-09-16

**Status: BINDING for implementation.** It supersedes `REFCV6_DESIGN_GROUNDED.md` wherever they disagree.
Every implementation agent builds against this file; nothing else is authority.

## 0. The PI's directives of 2026-09-16, verbatim

> * *"Let plan to train jointly the resnet-trunk, the bev map (based on the sam2 maps as gt) and a head for 3d bounding boxes extracted from the resnet trunk (based on the gt agent bounding boxes included in our training corpus from the av data set). I would also recommand to give additionally access to the truink for the diffusion planner"*
> * *"Confirm using nav command as mandatory input for tactical and operative planning. The selection of the tactical plan and the planing and selection of the trajectory must use the nav command and follow it. We dont need any head to estimate the route. Regarding the startegic layer, we will deacitivate it in the next training and eval experiment"*
> * *"Let review the max speed logic, it should be simple and models a max set speed included as input parameter. It should few discrete values: 30 kph, 50 kph, 100 kph, 120 kph. I correct my statement sayibng the tactical layer must learn max speed. Let use it as input in our next experiment. I'm not excluding to train the tactical layer in the future to set the max speed depending on the environmnet All other tactical vocabs must be learned. The tactical layer must learn to emitt the valid tactical behaviors, choose the best architecture for it. It shoudl learn them from the scene embeddings, for the agent and the map."*
> * *"Implement f1 to f9"*
> * *"Init our resent trunk with timm + imagenet and train it as described in the papers"*
> * *"Do what is necessary to prepare the rl post training, implement D9"*
> * *"run the d3 proof package"*
> * *"We won't use lidar for bev GT, we will stick to the sam3 maps and wait until the augmentation finishes. Let prepare the archotecture, the implementation and the trainer for it"*

⭐ **The correction the PI made:** max speed is an **INPUT**, not a tactical target. This reverses the 09-15 wording and restores the 09-01 / 09-10 rulings. **D1 is settled: none of R1/R2/R3 — max speed is a 4-value input parameter.** Every *other* tactical behaviour is learned.

## 1. The model

```
frame 256x640 ─► TRUNK: timm resnet34, ImageNet a1_in1k  (§2)
   ├─ stride-16 map 16x40x256 ─► BEV LIFT (parameter-free geometry) ─► BEV encoder ─► BEV feats 120x64
   │                                 ├─► MAP head  (9-class SAM3 soft CE, seen cells only)
   │                                 └─► BOX head  (3-D cuboids, Hungarian)
   └─ stride-32 map 8x20x512 ─► 160 image tokens
                                     │
 nav(4) + max-speed(4) + v0/ego ─► CONDITION ─FiLM─► every decoder layer, every denoising pass
                                     │
 TACTICAL layer (§4): behaviour queries cross-attend {agent slots, BEV tokens} ─► valid behaviours + lat/lon
                                     │ detached
 117 v0-rolled anchors ─► DIFFUSION DECODER, per layer:
      (1) BEV sampled at the candidate's own waypoints        [DiffusionDrive coupling (1)]
      (2) agent slots addressed by waypoint                   [WP-B]
      (3) image tokens by content                             [direct trunk access — the PI's request]
      ─► DD-faithful training (F1–F9, §3)
      ─► fan ─► selection: confidence + tactical prior + nav-compliance term + speed mask
```

⛔ **Deactivated for this experiment:** the whole strategic layer — route head, `g_str`, strategic GRU. No head estimates the route (PI). The flags remain but default OFF and the heads are not built.

## 2. Trunk — timm ResNet-34, ImageNet, trained as the papers do

| item | value | source |
|---|---|---|
| backbone | `timm` **`resnet34.a1_in1k`**, ImageNet-pretrained | DiffusionDrive `rl_config.py:16-18`; paper p.12 |
| input | **single frame, 3 channels**, 256×640. DD is single-frame, and H6 measured our 3-frame stack is not load-bearing. ⚠️ A 9-channel stem-inflated variant (ImageNet weights repeated ÷3) is the registered knockout, not the default | DD p.7 |
| features | stride-16 `layer3` → 16×40×256 for **perception**; stride-32 `layer4` → 8×20×512 for the **planner** | an oracle on 8×20 tops out at AP 0.3341 vs 0.4713 on 16×40 |
| optimiser | **AdamW**, weight decay 1e-4, **encoder lr ×0.5** of the heads, warm-up then cosine | DD `rl_config.py:124-128`, paper §4.2 |
| normalisation | ImageNet mean/std on the RGB input — without it the prior is wasted (measured trap in the seed campaign) | `E-SEED-2` |
| params | ≈21.8 M trunk, down from 90.5 M. Trunk size was MEASURED not to matter for planning (base vs XL, not separated) | `MODEL_REGISTRY:2563` |

## 3. Diffusion — F1…F9, all implemented

| # | change |
|---|---|
| F1 | random-t training: ONE decoder call at t ~ U[0,50), every cascade layer supervised |
| F2 | DD step semantics t → t−1 (keeps 95 % of the residual) instead of our 10 → 0 |
| F3 | per-layer offset heads + per-layer loss + **detach** between layers |
| F4 | per-layer AdaLN timestep modulation |
| F5 | score from the emitting pass's own confidence + **focal** loss |
| F6 | `--w-u0 0` — DD has ONE reconstruction loss; ours duplicated it |
| F7 | several noise samples per anchor (widen the three indexing sites first) |
| F8 | DD-style flat waypoint-space noise as an explicit ARM against our control-space noise |
| F9 | keep the v0-conditioned 117-anchor vocabulary |

## 4. Tactical layer — behaviour queries over the scene embedding

**Architecture (chosen; the PI left it to us): a DETR-style behaviour decoder**, 2 layers, d = 256.

- **Queries:** one per vocabulary behaviour (22) + 8 lateral + 8 longitudinal action queries.
- **Keys/values:** the **agent slots** and the **BEV tokens** (30×16) — *"the scene embeddings, for the agent and the map"*. Image tokens are deliberately **not** its input: behaviours are about the scene, not the pixels.
- **Conditioning:** FiLM with [nav one-hot (4), max-speed one-hot (4), v0, a0].
- **Outputs:** per-behaviour **validity** as multi-label **BCE** — several behaviours are valid at once, which is exactly why this is not a softmax — plus per-behaviour confidence, and lat/lon action posteriors.
- **Losses:** v8.1 GT labels; BCE 0.05 on goal tokens, CE 0.025 per action head, inside the existing `MANEUVER_WEIGHT` budget.
- **Feeds the planner, detached:** (a) the lat/lon posterior replaces the image-only lat3/lon3 as the **anchor prior** (new zero-init 8→117); (b) the valid-behaviour set gates **selection**.

⚠️ 17 of 22 tokens are trainable (5 masked for lack of negatives); 10 sit under the n = 200 floor. Reported per class, never pooled.

## 5. Nav and max speed

**Nav (4-way one-hot) is MANDATORY** and reaches: the condition (FiLM in every decoder layer, every denoising pass), the tactical decoder, and **selection** via a parameter-free nav-compliance term behind a zero-init gate.

**The plan must FOLLOW nav.** Acceptance: **T-FLIP** — with the junction command flipped, the plan follows the fed command on ≥ 0.50 of windows (today 0.205), and true-minus-shuffled compliance ≥ 0.38.

**Max speed is an INPUT: 4 discrete values {30, 50, 100, 120} km/h**, one-hot, embedded into the same condition.
- Training value = the **smallest of the four that is ≥ the window's realised maximum speed**. ⚠️ Declared **ego-future derived (~2 bits)** and stamped in `config.json` like every other oracle channel.
- **Obedience test (acceptance):** force 30 km/h on windows whose GT exceeds 40 km/h; the planned maximum must stay ≤ the limit on ≥ 99 % of them, with the ADE cost reported.
- The tactical layer does **not** predict it in this experiment. A future arm may.

## 6. Perception heads

| head | reads | ground truth | loss |
|---|---|---|---|
| **MAP** | BEV features from the stride-16 lift | **SAM3 semantic maps only** — ⛔ no LiDAR GT (PI) | soft cross-entropy, 9 classes, **seen cells only** |
| **BOX (3-D)** | stride-16 tokens (+ BEV features) | `obstacle.offline` **cuboids** (87,481 measured, ground-standing bottom faces) | Hungarian matching; (x, y, z, l, w, h, yaw) + class. Today's seam emits (cx, cy, l, w, yaw): **add z and h** |

Both losses reach the trunk. The gradient-conflict detector (+1 / 0 / −1 analytic controls, 30× mutation) runs in every arm.

## 7. Data

- **SAM3 maps:** all clips once production finishes (~22 Sep). Dev-box preparation uses eval-clip maps copied from Thor.
- **Boxes:** the existing obstacle join (96.83 % of train clips).
- ⛔ **LiDAR BEV is not a training target.** It may still be quoted as an independent evaluation reference.

## 8. Before any pod hour

The D3 proof package and the D9 RL lever L1, both on the dev box.

## 9. Open risks, named

1. **"Follow nav" versus learning from vision.** 42.7 % of the TURN label's entropy is already in the nav token, so a tactical layer can satisfy turn labels by copying the command. **T-ZERO stays a reported diagnostic on the non-turn classes**, so we can still see what was learned beyond the command.
2. **Max speed remains ego-future derived**, even at 4 values. The obedience test is what makes it defensible.
3. **The trunk shrinks 90.5 M → 21.8 M.** Planning capacity was measured not to matter; perception capacity at 21.8 M is untested here.
4. **No strategic layer** means no route output at all in this experiment, by the PI's instruction.
