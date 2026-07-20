"""TanitEval — model registry: the checkpoints under evaluation on the eval pod.
Single source of truth for loaders/runner/report. feat_kind names the frozen
encoder the data layer must apply for feature-input (REF-A) models."""

MODELS = [
    dict(key="flagship-speed", name="Flagship 4B (speed)", family="TanitAD",
         arch="flagship-worldmodel",
         ckpt="/root/models/flagship-speed/ckpt.pt",
         config="flagship4b", encoder="trained ViT-12 (9ch, 256px)",
         encoder_frozen=False, speed_input=True, action_dim=3,
         hf="Sayood/tanitad-flagship-4b-phase0 (pending re-push)",
         anti_collapse="SigReg-64",
         note="THE definitive flagship (action_dim 3); mid-training relay."),
    dict(key="flagship-30k", name="Flagship 4B (speed, 30k FINAL)",
         family="TanitAD", arch="flagship-worldmodel",
         ckpt="/root/models/flagship-30k/ckpt.pt",
         config="flagship4b", encoder="trained ViT-12 (9ch, 256px)",
         encoder_frozen=False, speed_input=True, action_dim=3,
         hf="Sayood/tanitad-flagship-4b-phase0 (pending re-push)",
         anti_collapse="SigReg-64",
         note="v1 FINAL (step 29999, flagship4b-speedjerk-30k). "
              "A/B vs flagship-speed (19k relay) = more-training read."),
    dict(key="flagship-nospeed", name="Flagship 4B (no-speed, ref)",
         family="TanitAD", arch="flagship-worldmodel",
         ckpt="/root/models/tanitad-flagship-4b-phase0/ckpt.pt",
         config="flagship4b", encoder="trained ViT-12 (9ch, 256px)",
         encoder_frozen=False, speed_input=False, action_dim=2,
         hf="Sayood/tanitad-flagship-4b-phase0", anti_collapse="SigReg-64",
         note="No-speed ablation baseline (step 22k). Comparison only."),
    dict(key="refa-dinov2", name="REF-A DINOv2 4B", family="TanitAD",
         arch="refa-plus", ckpt="/root/models/tanitad-refa-dinov2-4b/ckpt.pt",
         config="flagship4b", d_dino=768, adapter="temporal",
         feat_kind="dinov2", encoder="frozen DINOv2-B/14", encoder_frozen=True,
         speed_input=True, action_dim=3, four_brain=True,
         hf="Sayood/tanitad-refa-dinov2-4b", anti_collapse="frozen encoder",
         note="Canonical 30k (2,376-ep training)."),
    dict(key="refa-ijepa", name="REF-A I-JEPA 4B (best=7k)", family="TanitAD",
         arch="refa-plus", ckpt="/root/models/tanitad-refa-ijepa-4b/ckpt.pt",
         config="flagship4b", d_dino=1280, adapter="temporal",
         feat_kind="ijepa", encoder="frozen I-JEPA ViT-H/14",
         encoder_frozen=True, speed_input=True, action_dim=3, four_brain=True,
         hf="Sayood/tanitad-refa-ijepa-4b", anti_collapse="frozen encoder",
         train_ids="/root/taniteval/train_ids_320.txt",
         note="320-ep variant; 7k ckpt (beats its own 15k = overfit). "
              "Canonical val 80% LEAKED into its train set -> guard excludes; "
              "clean number lives on the f1b378 val (pod3 gates)."),
    dict(key="refa-dynin", name="REF-A dyn-in 4B (snapshot)", family="TanitAD",
         arch="refa-plus", ckpt="/root/models/refa-dynin-snap/ckpt.pt",
         config="flagship4b", d_dino=768, adapter="temporal",
         feat_kind="dinov2", encoder="frozen DINOv2-B/14", encoder_frozen=True,
         speed_input=True, dyn_input=True, action_dim=4, four_brain=True,
         hf=None, anti_collapse="frozen encoder",
         note="--dyn-input (ego [v0, yr0], action_dim 4) --four-brain, "
              "ego_dropout 0.25. Rolling ckpt on pod3 "
              "(/workspace/experiments/refa-dynin-4brain-30k) snapshotted aside "
              "at the target step. H26 intent->operative read in the FROZEN-"
              "encoder regime (REF-A trains WITH intent on, unlike flagship's "
              "intent-free deploy)."),
    dict(key="refa-dynin-30k", name="REF-A dyn-in 4B (30k FINAL)",
         family="TanitAD",
         arch="refa-plus", ckpt="/root/models/refa-dynin-30k/ckpt.pt",
         config="flagship4b", d_dino=768, adapter="temporal",
         feat_kind="dinov2", encoder="frozen DINOv2-B/14", encoder_frozen=True,
         speed_input=True, dyn_input=True, action_dim=4, four_brain=True,
         hf=None, anti_collapse="frozen encoder",
         note="FINAL 30k (step 29999) of --dyn-input (ego [v0,yr0], "
              "action_dim 4) --four-brain REF-A, ego_dropout 0.25, frozen "
              "DINOv2-B/14. scp from pod3 refa-dynin-4brain-30k. Milestone "
              "A/B vs flagship-30k (v1); H26 intent->operative, frozen-"
              "encoder regime."),
    dict(key="refb", name="REF-B (speed)", family="TanitAD", arch="refb",
         ckpt="/root/models/tanitad-refb-speed/ckpt.pt", config="refb",
         encoder="trained ViT-25 (9ch, 256px)", encoder_frozen=False,
         speed_input=True, action_dim=2, hf="Sayood/tanitad-refb-speed",
         anti_collapse="trained encoder",
         note="Hierarchical planner — profiling + (planner metrics TBD); "
              "no grounded rollout head."),
    dict(key="refb-10k", name="REF-B (speed) step-10k last", family="TanitAD",
         arch="refb", ckpt="/root/models/refb-speed-10k/ckpt.pt",
         config="refb", encoder="trained ViT-25 (9ch, 256px)",
         encoder_frozen=False, speed_input=True, action_dim=2, hf=None,
         anti_collapse="trained encoder",
         note="Last ckpt of the refb-speed lineage (step 10000; run rotated "
              "to refbpatch before 30k). NOTE: pod1 ckpt_prepatch_step8500.pt "
              "is a byte-identical copy of this file (misnamed). "
              "arch-v2 milestones need the v2 loader (not on eval yet)."),
    # ---- arch-v2 --refbpatch milestones (2026-07-19 refb-v2-30k eval) ----
    dict(key="refb-v2-20k", name="REF-B v2 (@20k milestone)",
         family="TanitAD", arch="refb", strict=True,
         ckpt="/root/models/assess-20260719/refb-v2/ckpt_step20000.pt",
         config="refb", encoder="trained ViT-25 (9ch, 256px)",
         encoder_frozen=False, speed_input=True, yaw_input=True,
         action_dim=2, hf=None, anti_collapse="trained encoder",
         note="--arch-v2 --refbpatch (B1 TIME-anchored tactical decoder, B2 "
              "[v0,yr0] ego, aux-yaw, path heads), step-20000 milestone of "
              "refb-refbpatch-v2-30k (tanitad-pod). STRICT load. Prior "
              "milestone for the 30k A/B. Needs TANITEVAL_STACK_OVERRIDE="
              "/root/models/assess-20260719/stack-v2b."),
    dict(key="refb-v2-30k", name="REF-B v2 (30k FINAL)",
         family="TanitAD", arch="refb", strict=True,
         ckpt="/root/models/refb-v2-30k/ckpt.pt",
         config="refb", encoder="trained ViT-25 (9ch, 256px)",
         encoder_frozen=False, speed_input=True, yaw_input=True,
         action_dim=2, hf="Sayood/tanitad-refb-speed", anti_collapse="trained encoder",
         note="FINAL 30k (step 29999) of --arch-v2 --refbpatch REF-B "
              "(B1 TIME-anchored tactical decoder, B2 [v0,yr0] ego, aux-yaw, "
              "path heads). STRICT load from run's own config.json. "
              "scp from tanitad-pod refb-refbpatch-v2-30k. Needs "
              "TANITEVAL_STACK_OVERRIDE=/root/models/assess-20260719/stack-v2b."),
    # ---- REF-C anchored-diffusion arm (DiffusionDrive-style) -----------------
    dict(key="refc-xl", name="REF-C-XL (anchored-diffusion, snapshot)",
         family="TanitAD", arch="refc", config_preset="xl", mode="diffusion",
         ckpt="/root/models/refc-xl-snap/ckpt.pt", config="refc-xl",
         encoder="trained ResNet-L (9ch, 256px, base_width 124)",
         encoder_frozen=False, speed_input=True, action_dim=2, hf=None,
         anti_collapse="trained encoder",
         note="REF-C-XL (~252M): a V2-99-class ResNet trunk + a d=512/6-layer/"
              "256-anchor DiffusionDrive-style truncated-diffusion decoder (2 "
              "denoise steps) + H15 imagination graft + hierarchy/maneuver->"
              "anchor grafts. DIRECT anchored-diffusion trajectory head (own "
              "decoder; NO grounded operative rollout, step_readout=None -> "
              "evaluated via taniteval.refc_eval). Read-only snapshot pulled "
              "from pod3 refc-diffusion-xl-30k (mid-training, ~16k). v0 is fed "
              "through the measurement encoder (nav+v0); REF-C's trajectory "
              "head consumes NO actions (action_dim nominal). refc1=False -> "
              "horizons ARE 0.5/1/1.5/2 s time waypoints, comparable to "
              "gt_ego_waypoints. NOTE: this is the XL scale arm, ~252M — NOT "
              "the 54.7M `small`/DiffusionDrive-scale preset."),
    dict(key="refc-xl-live", name="REF-C-XL (anchored-diffusion, step 28000)",
         family="TanitAD", arch="refc", config_preset="xl", mode="diffusion",
         ckpt="/root/models/refc-xl-live/ckpt.pt", config="refc-xl",
         encoder="trained ResNet-L (9ch, 256px, base_width 124)",
         encoder_frozen=False, speed_input=True, action_dim=2, hf=None,
         anti_collapse="trained encoder",
         note="REF-C-XL (~252M) LATEST read-only pull from pod3 "
              "refc-diffusion-xl-30k, direct pod3->eval scp (18.2 MB/s, md5 "
              "531fd19cc13f411cd0bf2ef49c72ec26 verified against source, "
              "source mtime unchanged during copy -> untorn). PROVISIONAL: "
              "step 28000 of a 30000-step run that was STILL TRAINING at pull "
              "time (resumed 28001, ~3.4 s/step, 30k ETA ~09:40 UTC "
              "2026-07-20) — this is NOT the final ckpt and the number must "
              "be reported as step-28000, not '30k'. refc1=False -> horizons "
              "ARE the 0.5/1/1.5/2 s time waypoints, so the row is comparable "
              "to gt_ego_waypoints / every other arm. Same XL scale as "
              "refc-xl (that entry is the stale step-16000 snapshot)."),
    dict(key="refc-xl-30k", name="REF-C-XL (anchored-diffusion, 30k FINAL)",
         family="TanitAD", arch="refc", config_preset="xl", mode="diffusion",
         ckpt="/root/models/refc-xl-30k/ckpt.pt", config="refc-xl",
         encoder="trained ResNet-L (9ch, 256px, base_width 124)",
         encoder_frozen=False, speed_input=True, action_dim=2, hf=None,
         anti_collapse="trained encoder",
         note="FINAL of refc-diffusion-xl-30k: ckpt `step` field reads 29999 "
              "(the last of the 30000-step schedule; metrics.json final.step "
              "29999, steps 30000 — report it as step-29999). Pulled 2026-07-20 "
              "09:41 UTC pod3 -> eval over the direct agent-forwarded path, md5 "
              "966d4eff1ea5ddf86efba01b8344e198 verified identical on both "
              "sides; pod3 training process already exited (GPU idle) so the "
              "source file was quiescent. Supersedes refc-xl-live (step 28000, "
              "pulled mid-training) and refc-xl (step ~16000). Same XL config: "
              "256 FPS anchors (externally built, refc_anchors_full.pt, carried "
              "in the decoder.anchors buffer), d=512/6-layer decoder, 2 "
              "truncated-denoise steps, H15 imagination + hierarchy + "
              "maneuver->anchor grafts ON, grounded_selector OFF, refc1=False "
              "-> the horizons ARE the 0.5/1/1.5/2 s time waypoints."),
]


EXTERNAL = [
    dict(name="PDM-Closed (NAVSIM v2)", metric="EPDMS", value=51.3,
         kind="closed-loop", bench="navhard", date="2026-03",
         src="arXiv:2506.04218"),
    dict(name="TF++ (VLAAD-MIL)", metric="DrivingScore", value=86.97, sr=71.97,
         kind="closed-loop", bench="Bench2Drive", date="2026",
         src="Bench2Drive LB"),
    dict(name="ADT", metric="DrivingScore", value=77.90, sr=55.0,
         kind="closed-loop", bench="Bench2Drive", date="2026",
         src="Bench2Drive LB"),
]

# CORPORA — evaluation-corpus inventory for the generalization panel (test E).
# Added 2026-07-18 (tools-devenv agent), MINIMAL additive entry. physicalai is
# the pre-existing in-distribution default; each root is a dir of ep_*.pt in the
# canonical epcache contract (frames_u8 [T,9,256,256] uint8, actions [T,2],
# poses [T,4]=x,y,yaw,v). Back-compat: absence of --corpus keeps physicalai.
CORPORA = [
    dict(key="physicalai", name="PhysicalAI-AV val (in-distribution)",
         root="/root/valdata/physicalai-val-0c5f7dac3b11",
         kind="in_distribution", feed="frames",
         note=("held-out episodes of the TRAINING corpus (disjoint eps) — "
               "the in-dist reference for the OOD gap.")),
    dict(key="comma", name="comma2k19 highway commute (UNSEEN OOD)",
         root="/root/valdata/comma2k19-val-76b6e94a97a1",
         kind="ood_cross_corpus", feed="frames", license="MIT",
         note=("real comma.ai highway/commute, EON road cam. 64 segments "
               "across 21 routes from raw_data/Chunk_1 (HF commaai/comma2k19). "
               "f-theta-canonical 256 crop (f_eff 266 via COMMA2K19_FOCAL_PX, "
               "same canonical FOV as physicalai). Flagship trained on "
               "PhysicalAI-AV ONLY -> every episode is out-of-distribution.")),
    dict(key="cosmos",
         name="Cosmos-Drive-Dreams synthetic (UNSEEN OOD, CC-BY-4.0)",
         root="/root/valdata/cosmos-val-e8f3cef4976b",
         kind="ood_cross_corpus_synthetic", feed="frames",
         license="CC-BY-4.0",
         note=("NVIDIA Cosmos-Drive-Dreams rendered pixels "
               "(front_wide_120fov), 46 eps (weather variants: "
               "clear/foggy/rainy/snowy/night) from "
               "generation.tar.gz.part-000. Same [T,9,256,256] contract. "
               "REBUILT 2026-07-19 (build_cosmos_v2.py): crop uses the "
               "TRUE per-clip source-rig intrinsics "
               "(pinhole_intrinsic.camera_front_wide_120fov, fx~944 native) "
               "centered on the per-clip PRINCIPAL POINT with the rig-B "
               "bottom overflow (~90-98 px) replicate-padded -> real "
               "f_eff 265.9-266.2, matching comma/physicalai. The previous "
               "cache (cosmos-val-a7a8527ba14e) used a nominal-120deg focal "
               "+ geometric-center crop and was ~1.70x ZOOMED (true f_eff "
               "~452) -> every cosmos result before 2026-07-19 is on the "
               "zoomed geometry. Pose pairing (121*chunk + frame) was "
               "already correct in both caches (verified per-ep). Clip "
               "e4ae6dee excluded: no pinhole_intrinsic upstream. Exact "
               "per-ep calibration in results/cosmos_calib.json; as-built "
               "crop geometry in the cache's build_manifest.json. Paired "
               "manifest at /root/cosmos_data/pairs/pairs_manifest.json "
               "for a future clear-vs-degraded counterfactual (test F).")),
]
