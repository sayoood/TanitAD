# RESULT — refb-derived nav vs the v7.2 `nav_command` token: 65.5 % agreement, and refcv3's nav is near-null on B1

`E-ARCH-NAVSRC-1` · 2026-09-02 · Master Mind · dev box (0 load on any training box) · **MEASURED**

## Question
The PI asked (2026-09-02) for refav1 to use *"the nav command from our v7.2 vocabulary currently used in
training refcv3"*, then asked for confirmation that refcv3 uses it. Source reading found that **refav1 already
reads the v7.2 token** (`refav1_loader.py:264-328`) while **refcv3 does not** — the ONLY assignment of
`item["nav_cmd"]` in its dataset chain (`V3Dataset` → `RouteV21Dataset` → `FailLoudWindowDataset`) is
`refb_train.py:220-222` → `refb_labels.nav_command(poses, t_last)`. Confirmed on `tanitad-refcv3`
(`refc_v3_train.py`/`refc_train.py` md5-identical to the repo); `refc_v3_train.py` has no nav-source
argparse option, and its `--v7-labels` feeds only the tactical `lat_v7`/`lon_v7` join (`:170-184`).
What WAS aligned on 2026-09-02 for refcv3: E13 nav injection to all layers (done) and the v7.2 tactical
labels (done). The nav SOURCE was never changed. Same 3-way vocabulary; different source. **How different, per window?**

## The two rules (both from source)
| | refb `nav_command` (`refb_labels.py:138`) | v7.2 `nav_command` (`s2_geom_emit_v7.py:692`) |
|---|---|---|
| input | net yaw change from the window's last frame to `min(25 s, episode end)` | the NEXT manoeuvre in a 30 s lookahead on the full recording |
| turn criterion | abs(Δyaw) > 45° | `is_turn`: dyaw ≥ 15°, arc radius ≤ 140 m, v_min ≤ 8 m/s (`turn_rule.json`); sweeping curves → `follow` by design |
| validity | needs ≥ 15 s of future, else `follow` + invalid | per-clip constant (`t0_constant`), oracle (`provenance: ego-future`) on all records |

## Method
For every clip in the v7.2 release (train 4,572 + eval 147 = 4,719; all present in the labeler's own
`egomotion_alpamayo.tar`, 4,719 parquets, columns `timestamp, qx..qw, x, y, z, vx..az, curvature`):
yaw from the quaternion, unwrapped, interpolated to 10 Hz over [0, 19.8] s (T = 199, the MEASURED B1
episode length, D-EPISODE-LENGTH); refb's rule re-implemented with its own constants (H = 250,
MIN = 150, ±π/4) at every window end `t_last ∈ [7, 198]` (W = 8, the live refcv3 batch shape);
compared with the clip's token. Script: `nav_agreement.py`; raw: `raw/nav_agreement_refb_vs_v72.json`.

## Results (MEASURED, n = 4,719 clips, 906,048 windows, 0 skipped, 14 s)
| read | value |
|---|---|
| v7.2 token per clip | follow 2,993 (63.4 %) · left 824 (17.5 %) · right 902 (19.1 %) |
| **fed command agrees, all windows** | **65.5 %** (train 65.5 % over 877,824 windows; eval 66.1 % over 28,224) |
| fed distribution, refb | follow **94.6 %** · left 2.7 % · right 2.7 % |
| fed distribution, v7.2 | follow 63.4 % · left 17.5 % · right 19.1 % |
| refb VALID windows | 198,198 = 21.9 % of windows; agreement on them 72.8 % |
| at the s2 anchor (8.0 s, frame 80) | refb is `follow` + INVALID on **every** clip (11.8 s of future < 15 s) |
| clips where v7.2 commands a turn | 1,726; refb has ANY turn window on 888 of them (51.4 %) |
| refb-turn but v7.2-follow | 423 clips (sweeping curves the v7.2 rule excludes) |
| v7.2 turn `args.time_s` | n = 1,726, median 7.4 s, p10 0.0, p90 26.3; 79.6 % ≤ 19.9 s if clip-timeline, 62.3 % if relative to t0 |

Confusion over refb-valid windows (refb / v7.2): follow/follow 111,618 · follow/left 17,510 ·
follow/right 19,995 · left/left 16,101 · left/follow 7,345 · left/right 1,354 · right/right 16,535 ·
right/follow 6,743 · right/left 997.

## What it means
1. **refcv3's E13 nav injection is fed a near-constant signal on B1** (`follow` 94.6 %). The comment in
   `refc_v3.py:433` ("~75–79 % follow") was measured on the parity corpus, whose episodes are long enough
   for the 15 s rule; on 19.9 s B1 episodes only the first 4.8 s of windows can ever carry a turn.
2. The two arms are therefore **not conditioned on the same information**, and "finish and document" would
   leave the refcv3-vs-refav1 comparison confounded on the programme's own thesis input.
3. The v7.2 rule is the semantically right nav (a *route command* ahead of a *turn*, with `distance_m`/`time_s`);
   refb's is a horizon-yaw oracle that conflates curves with turns.

## Caveats (stated, not hidden)
* Yaw comes from the egomotion quaternion, not from a `v2ep` `poses` file. INHERITED: the v2ep poses were
  built from this same egomotion — not re-verified against a v2ep here. The agreement rate is insensitive to
  small yaw differences (45° threshold).
* `t0_s = 8.0` is read as clip-timeline time (frame 80). The emitter's timeline for `nxt[0]` (`time_s`) was
  not pinned here — both readings are reported.
* This is a LABEL/INPUT statistic, not a model result — no tier stamp applies.

## Decision this feeds
`GOALS_AND_CLAIMS.md` C-NAV-SOURCE-DIVERGENCE — recommended **(d)**: switch refcv3's nav source at its next
500-step checkpoint via `--resume` + `--nav-from-v7` (in implementation; default OFF; train AND eval datasets;
`config.json` stamps `nav_cmd_derivation`), recording the switch step as a confound. Nav-shuffle control stays
mandatory for both arms.
