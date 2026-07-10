# YouTube / Heterogeneous Dashcam Strategy (H7 — the 1000× data thesis)

**Directed by Sayed 2026-07-09 night. Status: strategy v1 (concept-complete); pilot experiments
queued; deep-research validation via the Sonnet literature workflow.**

## 1. Why this matters strategically

OpenDV-YouTube-class corpora (~2,000 h; the full YouTube dashcam space is orders of magnitude
larger) are the only data source that scales like the thesis demands: free, globally diverse
(countries, weather, chaos levels no curated corpus reaches), and continuously growing. Every
competitor trains on fleet data they own; a stack that can *drink from the open web* has a
structurally different data economics — that IS H7. The obstacles are exactly the three Sayed
names: unknown cameras (intrinsics), unknown mounting (extrinsics), and non-automotive character
(overlays, dashcam artifacts, non-driving segments).

## 2. The pipeline (staged, each stage with a falsifier)

### Stage Y0 — Canonicalization without calibration (the hard core)
Our D-016 focal canonicalization assumes a KNOWN focal length. For YouTube we must ESTIMATE it:
1. **Geometric self-calibration:** vanishing-point + horizon-line estimation over many frames of
   the same video (driving scenes are calibration patterns: lane parallels → VP; VP trajectory →
   focal + pitch/roll). Deliverable: per-VIDEO (f_px, horizon, roll) with confidence.
2. **Learned focal regressor as cross-check:** small net trained on our calibrated corpora
   (comma/PhysicalAI/cosmos at various synthetic crops → known f) predicting f from single frames;
   agreement between (1) and (2) becomes the acceptance test.
3. Then the EXISTING `focal_crop_resize` path canonicalizes to F_REF=266 px — same contract,
   I7-fingerprint-compatible.
- **Falsifier Y0:** on held-out comma frames with SPOOFED unknown focal (random crops), the
  estimation pipeline must recover f within ±10%; worse ⇒ per-video canonicalization is unreliable
  ⇒ restrict to videos with detectable lane geometry (self-selecting subset is still huge).

### Stage Y1 — Content & quality gating
- Automatic filters: driving-segment detection (ego-motion present via optical-flow signature),
  overlay/watermark detection (static-pixel maps — comma overlay precedent), night/weather tags,
  minimum resolution/bitrate, dedup (per-channel + perceptual hashes).
- Non-automotive character is partly a FEATURE: bus/truck/motorcycle mounts widen the embodiment
  distribution — tag, don't discard (embodiment tag joins the domain embedding).

### Stage Y2 — Actions via inverse dynamics (H7 core)
- No CAN. Two complementary label sources:
  a. **Our trained inv-dyn head** (A5, trained on real CAN corpora) pseudo-labels actions from
     consecutive canonicalized frames — the flywheel: model labels new data that trains the model.
  b. **Visual odometry** (monocular VO on canonicalized frames) → yaw-rate/speed → derived
     steer/accel exactly like the physicalai/cosmos pose path (`poses_to_signals` reuse).
- Cross-agreement (a)↔(b) is the per-clip action-quality score; low agreement ⇒ video-only use
  (representation learning without action conditioning — still valuable for the encoder).
- **Falsifier Y2:** pseudo-labels on held-out comma (pretending no CAN) must correlate with real
  CAN at r>0.8 for steer; below ⇒ pseudo-labeled data enters at reduced loss weight or video-only.

### Stage Y3 — Curriculum entry
- YouTube data enters the mix ONLY through the D-010 bake-off discipline: real-anchored gates
  (D1/D2 on comma val) must not regress as the YouTube share grows (10% → 30% → ...).
- Domain tag `data:youtube-<channel-class>`; PUBLIC-CLAIM note: source-derived license is gray —
  training-internal only pending legal review (same firewall machinery as physicalai).

## 3. What we reuse (this is mostly built)
Focal machinery (D-016), the 9-ch episode contract, epcache, MixedWindowDataset domain tags,
inv-dyn head (training now), the A8/consequence stats for corpus QA, I7 fingerprints.

## 4. Pilot (1 week, cheap): "Y-pilot-50"
50 hand-picked dashcam videos (diverse cameras) → Y0 calibration estimates → canonicalize →
Y1 filters → Y2 pseudo-labels → measure: focal-recovery spread, action agreement r, A8 stats,
probe-fit on 20 clips vs comma baseline. Hardware: local 4060 + Colab. Owner: DataEng agent +
loop. Success = go/no-go evidence for the Phase-1 scale-up (OpenDV-2K full ingest).

## 5. Open research questions (→ Sonnet literature workflow)
- Best published monocular self-calibration for dashcam video (GeoCalib-class single-image
  calibration nets — production-ready?); driving-VO robustness at low fps; OpenDV-YouTube's own
  preprocessing choices (what did GenAD/Vista do about intrinsics — replicate or improve?);
  legal posture precedents for research training on YouTube-sourced video.
