# Option (c), motion from vision: where it stands after three measurements (MM, 2026-11-05 13:2x)

| step | result (MEASURED, landed) | consequence |
|---|---|---|
| RI-C0 (`raw/2026-11-05-motion-c-ri/`) | the base reads CLOCK-INSENSITIVE (M1 excess ratio reindexed / logged 1.002 / 1.003) | `reindexed` is the deciding read for any motion arm; logged-frame reads of frozen-feature arms stand |
| MOTIONPROBE (`raw/2026-11-05-motion-c-probe/`) | no usable motion in the cached frozen tokens (v WEAK, not motion-borne; dv ABSENT) | the cheap form (motion tokens over cached features) is dropped |
| CRAWL-LEAD (this folder) | PARTIAL at both bank seeds: L40(CRAWL) 0.4639 / 0.4956 against MOVING-OK 0.2409 / 0.2687 (difference +0.2230 / +0.2269, clip-cluster bootstrap); where a lead exists it is faster than the crawling model in about 88 % of records (ESTIMATED from the gap sequence) | the admissible content of a pixel-motion arm (other agents' motion) has a target in about half the crawl |

## Why (c) stops here on the allocated resources (named blockers, Rule Zero item 3)

1. **The pixel form cannot be read faithfully on any provisioned tier.** Its one admissible cue in the crawl is a visible lead moving
   away from a model that lags. On `logged` frames the lead sits at the human's gap at time t; on `reindexed` frames it sits at the
   gap the human had at the model's own position, i.e. closer than the lead really is. Neither renders the lead pulling away, and
   the model's own speed read from pixels duplicates the measured history. A pixel-motion arm read on T1-kin would therefore give a
   number that cannot count. **Unblocks with:** a re-perception tier (T2: AlpaSim / CARLA / a neural re-renderer), not provisioned.
2. **Encoder adaptation (LoRA on DINOv3-L)**, which would also need T2 to be read for this cue, is projected at 470 h per seed on box
   2 (§7.9). **Unblocks with:** the A40 or other compute the PI allocates.
3. Building the pixel frame cache (a frame store exists for 6 clips) and the network is days of box-2 and dev-box work whose read is
   blocked by item 1, so it is not started.

## What carries forward
* The lead association is a pointer for (a) and for the next longitudinal design: crawl states sit behind a lead about 1.9 times
  as often as normally moving states, and the lead is usually faster than the crawling model, so the missing behaviour is "keep
  pace with a lead you can see", not "yield". A single frame shows the lead's presence and distance (a static, admissible cue).
* Next, in the PI's order: (a), the level-from-image head, read on `logged`, `hold_now` and `reindexed` (deciding).
