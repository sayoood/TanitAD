# SPEC — H-REFAV1-MOTION: cross-channel motion injection on real DINOv3 features

**Written BEFORE any arm ran** · 2026-08-31 · Master Mind · Tier: **T0 mechanism probe**
(feature-prediction quality; nothing here is a driving claim)

```yaml
hypothesis: H-REFAV1-MOTION          # registered in GOALS_AND_CLAIMS.md first
one_variable: the initial-state motion channel (off / on / SHUFFLED control)
held_constant: [seed 0, episodes, window starts, steps 600, bs 2, lr 3e-4,
# bs AMENDED 4 -> 2 before any scored run: MEASURED 7.8 GiB of the 4060's
# 8.2 GiB at bs 2 (full-chain K=30 rollout over 640 tokens) -- bs 4 OOMs.
                clip 1.0, op_layers 2, d_state 1024, n_tokens 640,
                target_space FROZEN, no_hierarchy (brains off), w_cf 0]
# AMENDED before any arm ran: target_space=frozen for ALL arms, because
# each arm's adapter space is arm-specific -- cross-arm MSE is comparable
# only in the FIXED std(DINOv3) space. Also removes the collapse confound.
success: "motion arm beats base on held-out feature MSE at k in [6,30]
          (paired per-window, sign-consistent across >=4 of 5 val episodes)
          AND the shuffled-diff arm does NOT show the same gain"
failure: "no separation at any k, or the shuffle matches the true diff —
          then the channel is regularisation, not motion, and H-REFAV1-MOTION
          is refuted for this injection form"
controls:
  - copy_last_floor        # predict z_{t+k} = z_t — the no-dynamics floor
  - shuffled_diff_arm      # diff vs a NON-adjacent frame, same capacity
  - n_and_scale_printed    # 19 train / 5 val episodes — probe scale, stated
splits: {train: "eps 1..19 of the 24-spread", val: "eps 20..24, episode-disjoint"}
```

## Data (deterministic, recorded)

24 episodes from the PARITY corpus `physicalai-train-e438721ae894-w120-256x640cyl`,
sorted filename order, **every 100th** (NR%100==1) — pulled from Thor 2026-08-31,
list banked as `eplist.txt`. ⚠️ **NON-parity subset**: this is a mechanism probe,
its numbers are never comparable to any registry arm.

Features: **real DINOv3 ViT-L/16** (`facebook/dinov3-vitl16-pretrain-lvd1689m`,
access verified), 256×640 cylindrical frames at the v1 geometry, patch tokens only
(CLS + 4 registers discarded), every 2nd frame (0.2 s grid, `op_dt`), fp16 cache.

## Why the shuffle control is the load-bearing one

The injected channel adds parameters and input variance. A gain over base alone is
therefore ambiguous (capacity/regularisation vs motion). The SHUFFLED arm has the
same parameters and the same marginal input statistics but a temporally WRONG
difference — only a gain that the shuffle does NOT reproduce is evidence the
predictor used *motion*. (Same logic as the time-shuffled control rule in the
probe doctrine.)

## What this cannot show

Driving (T1), action-conditioning (that is O11's floor), or anything at scale.
A positive here gates `--motion-inject` INTO the real refav1 launch config as a
pre-registered arm; a negative kills the flag before it costs a real run.
