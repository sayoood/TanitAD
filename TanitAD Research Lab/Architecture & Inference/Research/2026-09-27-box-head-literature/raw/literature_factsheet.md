# Literature fact sheet: the published detection heads and protocols compared in RESULT.md

Read online on 2026-09-27 from arXiv HTML renderings (ar5iv / arxiv.org/html) and the official code repositories.
No PDF was downloaded and nothing was banked. Findings are PARAPHRASED; numbers, settings and code defaults are
copied as facts. Every source carries its key (S#), the URL that was read, and where in it the fact sits.

Evidence classes used below:
- `[LIB:<arXiv id>]` means the paper is already banked in `TanitAD Research Lab/Library/library.json` (it was still
  read online for this pass).
- `NOT BANKED` means PUBLISHED (arXiv:<id>, read online, NOT BANKED).
- `CODE` means an official repository file (config or source) read online.
- `2x` means the number was read twice with independent prompts and agreed. `1x` means one read only; treat the
  number as published but not cross-checked by this pass. One first read was WRONG and corrected by a second
  (S27), which is why the distinction is kept.

---

## S1. DETR, arXiv:2005.12872 [LIB:2005.12872]
URLs: https://ar5iv.labs.arxiv.org/html/2005.12872 ; CODE https://raw.githubusercontent.com/facebookresearch/detr/main/main.py ; CODE https://raw.githubusercontent.com/facebookresearch/detr/main/models/detr.py
- Queries: N = 100, chosen to be much larger than the typical object count per image (§3.1; appendix A.4). (1x)
- COCO has 7.7 instances per image on average (from S30).
- Appendix A.5 (2x, read verbatim): with 100 query slots, DETR finds every instance up to about 50 visible
  instances of one class, then saturates; with 100 instances it finds about 30 on average. The authors note the
  test is out of distribution by design.
- No-object handling: the log-probability of the no-object class is down-weighted by a factor of 10 (§3.1);
  code: `empty_weight[-1] = eos_coef`, `eos_coef` default 0.1, loss = `F.cross_entropy(..., empty_weight)`
  (a WEIGHTED MEAN, i.e. normalised by the sum of the weights). CODE
- Box loss: L1 weight 5, GIoU weight 2, both summed over matched boxes and divided by the number of GT boxes. CODE
- Matching cost: class probability (weight 1) + L1 (5) + GIoU (2). CODE (`set_cost_*` defaults)
- Auxiliary decoding losses: prediction FFNs and the Hungarian loss after EVERY decoder layer, FFN parameters
  shared; the authors found it helps the model output the correct number of objects of each class (§3.2). CODE:
  aux_loss on by default.
- Decoder depth: AP and AP50 improve after every layer, +8.2 / +9.5 AP between the first and last layer (§4.2). (1x)
- Positional encoding ablation (Table 3): no spatial PE 32.8 AP vs baseline 40.6 (−7.8); learned PE at attention
  39.6. (1x)
- Architecture defaults (CODE main.py): 6 encoder / 6 decoder layers, hidden 256, FFN 2048, 8 heads, dropout 0.1,
  lr 1e-4, backbone lr 1e-5, weight decay 1e-4, 300 epochs (lr drop at 200), batch 2 per GPU (64 total in the paper).
- Schedule: 300 epochs for ablations, 500 for the final comparison, COCO 118k training images (§4).
- Evaluation: COCO AP over ranked detections; PostProcess applies no threshold (CODE). The panoptic demo filters
  detections below 85 % confidence (§4.4).
- A loss-component ablation (Table 4) exists but my read of it was garbled; NOT used.

## S2. Deformable DETR, arXiv:2010.04159 (NOT BANKED)
URLs: https://ar5iv.labs.arxiv.org/html/2010.04159 ; CODE main.py and models/deformable_detr.py (raw.githubusercontent.com/fundamentalvision/Deformable-DETR/main/...) ; CODE https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/matcher.py
- Queries raised from 100 to 300 (§5). Focal loss with loss weight 2 for classification (§5).
- Table 1 (1x, matches the widely cited values): DETR 500 ep 42.0 AP; DETR-DC5 500 ep 43.3; DETR-DC5 50 ep 35.3;
  DETR-DC5+ (= DC5 with focal loss AND 300 queries) 50 ep 36.2; Deformable DETR 50 ep 43.8; + iterative bounding
  box refinement 45.4; ++ two-stage 46.2. Abstract: better than DETR with 10x fewer epochs.
- Deformable attention: M = 8 heads, K = 4 sampling points, L = 4 feature levels (§4-5).
- Iterative refinement: each decoder layer refines the boxes predicted by the previous layer (§4.2).
- CODE defaults: dec_layers 6, hidden 256, FFN 1024, 8 heads, 300 queries, 50 epochs (lr drop 40), lr 2e-4,
  backbone 2e-5, aux_loss on; matching cost class 2 / L1 5 / GIoU 2; loss coefficients cls 2 / L1 5 / GIoU 2;
  focal_alpha 0.25.
- CODE loss: sigmoid focal loss normalised by `num_boxes` (the number of GT boxes), times the number of queries.
- CODE matcher: classification cost = alpha(1-p)^gamma(-log p) - (1-alpha)p^gamma(-log(1-p)), alpha 0.25, gamma 2.
- CODE init: class bias = -log((1-0.01)/0.01), i.e. a 0.01 foreground prior.
- CODE PostProcess: top-100 over sigmoid scores, no threshold.
- Multi-scale ablation (Table 2): +1.7 AP for multi-scale inputs and +1.5 AP for multi-scale attention (1x; a
  second number in the same read was garbled, so treat these as single-read).

## S3. DN-DETR, arXiv:2203.01305 (NOT BANKED)
URL: https://ar5iv.labs.arxiv.org/html/2203.01305
- Abstract: +1.9 AP under the same setting; 46.0 AP (12 epochs) and 49.5 AP (50 epochs) with ResNet-50; comparable
  performance with 50 % of the training epochs. (1x, abstract)
- Cause named: bipartite-matching instability gives inconsistent optimisation targets early in training (§1, §3-A).
- Method: noised copies of the GT boxes and labels are fed as extra queries, with an attention mask against
  leakage (§4-C, §4-D).
- Table II (1x): 12-epoch DC5-R50, DAB-DETR 38.0 vs DN-DETR 41.7 AP.

## S4. DINO, arXiv:2203.03605 (NOT BANKED)
URL: https://ar5iv.labs.arxiv.org/html/2203.03605
- Abstract: 49.4 AP in 12 epochs and 51.3 AP in 24 epochs (ResNet-50), +6.0 / +2.7 AP over DN-DETR. (1x)
- 900 queries (300 x 3), 6-layer encoder and decoder; focal loss alpha 0.25, gamma 2; loss weights cls 1.0,
  L1 5.0, GIoU 2.0 (appendix D.3). (1x)
- Contrastive denoising: "negative" noised queries farther from the GT are trained to predict no object, which
  teaches the model to reject anchors near but not on an object and removes duplicate predictions (§3.3). (1x)
- Table 4 (1x): DN-DETR 43.4 -> optimised 44.9 -> + pure query selection 46.5 -> + mixed 47.0 -> + look forward
  twice 47.4 -> + contrastive DN 47.9.

## S5. DETR3D, arXiv:2110.06922 (NOT BANKED)
URLs: https://ar5iv.labs.arxiv.org/html/2110.06922 ; CODE https://raw.githubusercontent.com/WangYueFt/detr3d/main/projects/configs/detr3d/detr3d_res101_gridmask.py
- Head: 6 layers, hidden 256 (§4.1). Each layer predicts a 3-D reference point per query, projects it into all
  camera feature maps with the camera matrices, samples features bilinearly, then applies self-attention
  (§3.3). FPN levels at 1/8, 1/16, 1/32, 1/64.
- Loss after EVERY layer; focal for classes, L1 for boxes; Hungarian matching (§3.4).
- Table 6 (2x): queries 30 / 100 / 300 / 600 / 900 / 1200 / 1500 -> mAP 0.201 / 0.313 / 0.338 / 0.347 / 0.346 /
  0.340 / 0.346, NDS 0.331 / 0.408 / 0.415 / 0.420 / 0.425 / 0.415 / 0.420; the authors read it as improving until
  saturation at 900.
- Table 5 (2x): NDS by layer 0..5 = 0.380, 0.410, 0.420, 0.420, 0.424, 0.425; mAP 0.302 -> 0.346.
- No NMS or other post-processing (§1/§3).
- Paper text: 12 epochs, batch 1 per GPU on 8 GPUs (1x). CONFIG: total_epochs 24.
- CONFIG: num_query 900; num_layers 6; embed 256; FFN 512; cross-attention num_points 1; with_box_refine True;
  focal (sigmoid, gamma 2, alpha 0.25, weight 2.0); L1 weight 0.25; GIoU 0.0; assigner FocalLossCost 2.0 +
  BBox3DL1Cost 0.25 + IoU 0.0; bbox coder max_num 300, no score threshold; point_cloud_range +-51.2 m;
  ObjectRangeFilter + ObjectNameFilter; use_valid_flag True; lr 2e-4 with backbone lr_mult 0.1;
  load_from fcos3d.pth.
- CODE (https://raw.githubusercontent.com/WangYueFt/detr3d/main/projects/mmdet3d_plugin/models/dense_heads/detr3d_head.py):
  `Detr3DHead(DETRHead)`; classification `avg_factor` = positives + negatives x `bg_cls_weight`; losses of every
  decoder layer logged as `d0.loss_cls` ... ; classification bias initialised with `bias_init_with_prob(0.01)`
  when sigmoid is used.

## S6. PETR, arXiv:2203.05625 [LIB:2203.05625]
URLs: https://ar5iv.labs.arxiv.org/html/2203.05625 ; CODE https://raw.githubusercontent.com/megvii-research/PETR/main/projects/configs/petr/petr_r50dcn_gridmask_p4.py
- 3-D position embedding: camera-frustum grid points (pixel x depth bins) are mapped to 3-D world coordinates with
  each camera's parameters and encoded into the image features (§3.2); depth 64 bins, linear-increasing.
- Table 3 (2x): 2-D PE only NDS 0.208 / mAP 0.069; 2-D + multi-view 0.224 / 0.089; 3-D PE only 0.356 / 0.305;
  all three 0.359 / 0.309. The authors attribute the main gain to the 3-D PE.
- Table 5(d) (2x): 600 / 900 / 1200 / 1500 anchor points -> NDS 0.339 / 0.351 / 0.354 / 0.359; paper default 1500.
- Loss: focal for classification (lambda_cls 2.0) and L1 regression; Hungarian (§3.5). 24 epochs (1x).
- CONFIG: num_query 900; 6 layers; embed 256; FFN 2048; standard multi-head cross-attention; LID, 3-D position
  range +-61.2 m; focal (gamma 2, alpha 0.25, weight 2.0); L1 0.25; GIoU 0.0; assigner FocalLossCost 2.0 +
  BBox3DL1Cost 0.25; NMSFreeCoder max_num 300, no threshold; use_valid_flag True; 24 epochs; final_dim 512 x 1408.
  The `_p4` config reads a single stride-16 feature level (INFERRED from the config name `p4`).
- CODE (https://raw.githubusercontent.com/megvii-research/PETR/main/projects/mmdet3d_plugin/models/dense_heads/petr_head.py):
  `PETRHead(AnchorFreeHead)`; classification `avg_factor` = positives + negatives x `bg_cls_weight` (the DETRHead
  pattern; `bg_cls_weight` is only set from a class_weight, none in the config, so 0: INFERRED); losses of every
  decoder layer logged as `d0.loss_cls` ...; the reference points are NOT refined between decoder layers.

## S7. PETRv2, arXiv:2206.01256 [LIB:2206.01256]
URL: https://ar5iv.labs.arxiv.org/html/2206.01256
- Extends DN-DETR query denoising to 3-D detection to accelerate convergence (§4.2); no isolated ablation
  number found. 1500 detection queries on the test set. 24 epochs. Extrinsic-noise robustness: at Rmax = 8,
  -4.12 % mAP and -2.85 % NDS (Table 6). (1x)

## S8. BEVFormer, arXiv:2203.17270 [LIB:2203.17270]
URLs: https://ar5iv.labs.arxiv.org/html/2203.17270 ; CODE https://raw.githubusercontent.com/fundamentalvision/BEVFormer/master/projects/configs/bevformer/bevformer_base.py
- Detection head = the Deformable DETR head on single-scale BEV features; 900 queries, top-300 kept at inference
  (§3.5, appendix A.3); BEV 200 x 200 over +-51.2 m; 24 epochs; ResNet101-DCN initialised from FCOS3D; 1600 x 900.
- Joint detection + segmentation (Table 4, 1x): detection NDS 0.520 joint vs 0.517 alone; road and lane
  segmentation worse when trained jointly.
- CONFIG: num_query 900; decoder 6 layers; embed 256; FFN 512; CustomMSDeformableAttention num_levels 1;
  with_box_refine True; focal (gamma 2, alpha 0.25, weight 2.0); L1 0.25; GIoU 0.0; assigner FocalLossCost 2.0 +
  BBox3DL1Cost 0.25; max_num 300; use_valid_flag True; ObjectRangeFilter + ObjectNameFilter; 24 epochs;
  load_from r101_dcn_fcos3d_pretrain.pth.

## S9. StreamPETR, arXiv:2303.11926 (NOT BANKED)
URLs: https://ar5iv.labs.arxiv.org/html/2303.11926 ; CODE https://raw.githubusercontent.com/exiawsh/StreamPETR/main/projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_24e.py
- 644 random queries + 256 propagated from a memory of 4 frames x 256 objects; query denoising as in PETRv2;
  24 / 60 epochs; batch 16; 256 x 704 input; ImageNet / nuImages pre-training (§4-5, appendix). (1x)
- CONFIG: num_query 644; num_propagated 256; memory_len 1024; topk_proposals 256; DN scalar 10, noise_scale 1.0,
  dn_weight 1.0; 6 layers; embed 256; FFN 2048; focal (gamma 2, alpha 0.25, weight 2.0); L1 0.25; code_weights
  2.0 on the two centre terms; assigner FocalLossCost 2.0 + BBox3DL1Cost 0.25; max_num 300; use_valid_flag True;
  24 epochs.

## S10. Sparse4D, arXiv:2211.10581 (NOT BANKED) and Sparse4D v3, arXiv:2311.11722 (NOT BANKED)
URLs: https://ar5iv.labs.arxiv.org/html/2211.10581 ; https://arxiv.org/html/2311.11722
- v1: 900 anchors; 6 cascade refinement layers; 13 keypoints per anchor (7 fixed + 6 learnable) sampled across
  4 scales, all views and recent frames; focal + L1 + depth BCE; a depth-reweight module scales each instance
  feature by the predicted depth confidence; 24 epochs (48 for test), FCOS3D initialisation, 640 x 1600. (1x)
- v3 (Table 5, 2x): Sparse4Dv2 0.439 mAP / 0.539 NDS; + single-frame denoising 0.447 / 0.548; + decoupled
  attention 0.458 / 0.551; + temporal denoising 0.462 / 0.557; + centerness 0.463 / 0.554 (mATE 0.581 -> 0.563);
  + yawness 0.469 / 0.561. Stated motivation for the quality heads: classification confidence is a poor proxy for
  box quality. 900 + 600 temporal instances, 6 layers, 256-d, 100 epochs (1x).

## S11. UniAD, arXiv:2212.10156 [LIB:2212.10156]
URLs: https://ar5iv.labs.arxiv.org/html/2212.10156 ; CODE https://raw.githubusercontent.com/OpenDriveLab/UniAD/main/projects/configs/stage1_track_map/base_track_map.py
- Training: perception jointly for 6 epochs, then end-to-end for 20 epochs (§2.5), starting from BEVFormer
  weights (appendix E). Track thresholds 0.4 (detection / new-born) and 0.35 (track filter) (appendix E.1). (1x)
- CONFIG: BEVFormerTrackHead num_query 900; 6 decoder layers; focal (gamma 2, alpha 0.25, weight 2.0); L1 0.25;
  assigner FocalLossCost 2.0 + BBox3DL1Cost 0.25; max_num 300; score_thresh 0.4; filter_score_thresh 0.35;
  stage-1 total_epochs 6; load_from bevformer_r101_dcn_24ep.pth; use_valid_flag True.

## S12. FCOS3D, arXiv:2104.10956 (NOT BANKED)
URL: https://ar5iv.labs.arxiv.org/html/2104.10956
- Dense per-location head on FPN P3-P7, objects assigned to levels by 2-D regression ranges
  (0, 48, 96, 192, 384, inf); 3-D centre-ness as a 2-D Gaussian; focal alpha 0.25, gamma 2; regression weights
  1 (offsets, size, angle), 0.2 (depth), 0.05 (velocity). No GT visibility rule stated in the paper. (1x)
- See S16 for the mono GT conversion used in practice.

## S13. MonoDETR, arXiv:2203.13310 (NOT BANKED)
URLs: https://ar5iv.labs.arxiv.org/html/2203.13310 ; CODE https://raw.githubusercontent.com/ZrrSkywalker/MonoDETR/main/configs/monodetr.yaml
- 50 queries; 3 decoder blocks; 256-d (§3.2, §4.1). Foreground depth map supervised from object boxes only
  (pixels in a box get that object's depth; nearest object wins where boxes overlap) (§3.1).
- §4.1 (2x): training DISCARDS objects with depth labels above 65 m or below 2 m, "for training stability";
  inference drops queries with category confidence below 0.2, no NMS.
- 195 epochs, batch 16, KITTI (7,481 train+val images, 3,769 val -> 3,712 train); AP40.
- Table 5 (1x): full 28.84 / 20.61 / 16.38 (Easy/Mod/Hard) vs without the depth-guided transformer
  19.69 / 15.15 / 13.93.
- CONFIG: classes ['Car']; 50 queries; 3 decoder layers; hidden 256; FFN 256; aux_loss True; focal alpha 0.25;
  loss coefficients cls 2, bbox 5, GIoU 2, 3-D centre 10, dim 1, angle 1, depth 1, depth map 1; matcher costs
  cls 2, bbox 5, GIoU 2, 3-D centre 10; 195 epochs; LID depth range to 60 m.

## S14. CenterPoint, arXiv:2006.11275 (NOT BANKED)
URLs: https://ar5iv.labs.arxiv.org/html/2006.11275 ; CODE https://raw.githubusercontent.com/tianweiy/CenterPoint/master/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py
- Dense centre heatmap: Gaussian per object with radius from object size, minimum radius 2; focal-type loss;
  L1 on offset, height, size, sin/cos yaw and velocity; a second stage scores boxes with an IoU-guided target so
  the score reflects box quality (§4, §4.1). 20 epochs on nuScenes (1x). Centre vs anchor on nuScenes:
  52.6 -> 56.4 mAP (Table 6, 1x).
- CONFIG: 6 class-group tasks; code_weights with 0.2 on velocity; regression weight 0.25; max_objs 500;
  min_radius 2; test score_threshold 0.1, max_per_img 500, NMS 0.2; class-balanced GT-paste sampler; 20 epochs.
  The training info file name carries `filter_True` (zero-point boxes filtered: INFERRED from the name, not read).

## S15. nuScenes, arXiv:1903.11027 [LIB:1903.11027], and its devkit
URLs: https://ar5iv.labs.arxiv.org/html/1903.11027 ; CODE https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/eval/detection/configs/detection_cvpr_2019.json ; CODE https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/eval/detection/README.md
- 1000 scenes x 20 s; 40k keyframes at 2 Hz; 1.4M 3-D boxes, i.e. about 35 annotated boxes per keyframe before
  any range or class filter (ANALYTIC: 1.4M / 40k). Objects are annotated only if at least one LiDAR or radar
  point covers them (§2). 10 detection classes (§3.1).
- The paper's OFT camera baseline filtered annotations with visibility below 40 % (appendix B). (1x)
- CONFIG (devkit): class_range car/truck/bus/trailer/construction_vehicle 50 m, pedestrian/motorcycle/bicycle 40 m,
  traffic_cone/barrier 30 m; centre-distance matching at 0.5 / 1 / 2 / 4 m; min_recall 0.1, min_precision 0.1;
  max_boxes_per_sample 500; mean_ap_weight 5.
- README: GT and predictions beyond the class range are removed; bikes in bike racks are removed; GT boxes with
  no LiDAR or radar points are removed; at most 500 boxes per sample; AP integrates the PR curve above 10 %
  recall and precision; NDS weights mAP 5 and each TP metric 1. The detection score is used for ranking; no
  threshold is prescribed.

## S16. mmdetection3d / mmdetection (the training pipeline of S5, S6, S8, S9, S11)
URLs: CODE https://raw.githubusercontent.com/open-mmlab/mmdetection3d/v0.17.1/mmdet3d/datasets/nuscenes_dataset.py ; CODE https://raw.githubusercontent.com/open-mmlab/mmdetection3d/v0.17.1/tools/data_converter/nuscenes_converter.py ; CODE https://raw.githubusercontent.com/open-mmlab/mmdetection/v2.25.0/mmdet/models/dense_heads/detr_head.py
- `get_ann_info`: with `use_valid_flag` the GT mask is `valid_flag` (= num_lidar_pts + num_radar_pts > 0);
  otherwise `num_lidar_pts > 0`. Either way, TRAINING GT without sensor points is dropped. Default False; every
  camera-3D config above sets True.
- Mono conversion (`get_2d_boxes`) keeps all four visibility bins (`visibilities=['', '1', '2', '3', '4']`), drops
  corners behind the camera and boxes whose projection misses the image.
- DETRHead: `bg_cls_weight` 0 unless the class is exactly DETRHead; classification `avg_factor` = number of
  positives (+ negatives x bg_cls_weight, i.e. 0); box and IoU losses normalised by the number of positives;
  losses computed and logged for EVERY decoder layer (`d0.loss_cls`, ...).

## S17. Waymo Open Dataset, arXiv:1912.04838 [LIB:1912.04838]
URL: https://ar5iv.labs.arxiv.org/html/1912.04838
- 3-D labels with no LiDAR point are ignored; LEVEL_2 = labeller-marked hard or at most 5 LiDAR points, the rest
  LEVEL_1 (§3.3). AP and heading-weighted APH (§4.1.1). Camera 2-D boxes are labelled separately. (1x)

## S18. KITTI object benchmark (Geiger et al., CVPR 2012; not on arXiv)
URL: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
- Easy: box height >= 40 px, fully visible, truncation <= 15 %. Moderate: >= 25 px, partly occluded, <= 30 %.
  Hard: >= 25 px, difficult to see, <= 50 %. Detections in DontCare areas are not false positives. 3-D IoU 0.7
  (car) / 0.5 (pedestrian, cyclist). AP over 40 recall positions since 2019-10-08. (1x)

## S19. Focal loss / RetinaNet, arXiv:1708.02002 (NOT BANKED)
URL: https://ar5iv.labs.arxiv.org/html/1708.02002
- FL = -alpha_t (1 - p_t)^gamma log p_t; gamma 2 with alpha 0.25 works best (§4.1, Table 1b).
- Table 1 (1x, matches the widely cited values): best alpha-balanced CE (gamma 0, alpha 0.75) 31.1 AP vs focal
  (gamma 2, alpha 0.25) 34.0 AP, +2.9. Best OHEM 32.8 vs FL 36.0 in the larger setting (Table 1d).
- Easy negatives dominate the loss and gradient in dense detection; alpha alone balances positives/negatives
  but not easy/hard (§3.2).
- Prior initialisation pi = 0.01 on the classification bias; plain CE without it diverged (§4.1, §5.1).

## S20. MonoDLE, arXiv:2103.16237 (NOT BANKED)
URL: https://ar5iv.labs.arxiv.org/html/2103.16237
- Localisation error is the main bottleneck of monocular 3-D detection (Table 1: 11.12 AP -> 78.84 AP with GT
  3-D location). (1x)
- Table 8 (2x), KITTI car AP40 Easy / Mod / Hard: baseline 16.12 / 12.97 / 10.99; drop training objects beyond
  40 m: 14.25 / 11.25 / 9.63 (WORSE); beyond 60 m: 17.45 / 13.66 / 11.68 (better); soft weighting c = 60,
  T = 1: 17.50 / 13.54 / 11.32. Hard coding: weight 1 if depth <= s, else 0; s = 60 m in their implementation.

## S21. LET-3D-AP, arXiv:2206.07705 (NOT BANKED)
URL: https://ar5iv.labs.arxiv.org/html/2206.07705
- Camera-only detectors localise objects laterally but err in depth; IoU-based 3-D AP counts such boxes as both
  a false positive and a false negative. LET-3D-AP tolerates longitudinal error up to a percentage of range
  (e.g. 10 %, so 5 m at 50 m) with a minimum in metres (§4.2). Primary and secondary metrics of the 2022 Waymo
  camera-only 3-D detection challenge (§5.4). Table 1 (1x): 3.1 % standard 3-D AP vs 23.0 % LET-3D-AP for the
  same camera-only detector at IoU 0.7.

## S22. CityPersons, arXiv:1702.05693 (NOT BANKED)
URL: https://ar5iv.labs.arxiv.org/html/1702.05693
- (2x) The "Reasonable" setup is pedestrians with height 50 px or more and occlusion ratio at most 0.35 (§3.3).
  The baseline is TRAINED only on that reasonable subset; other persons and ignore regions are excluded from
  negative sampling (§3.4).
- Up-sampling the input 2x gives 3.74 MR points (§2, M2). Proper ignore-region handling about 1.2 MR points
  (appendix, 1x).

## S23. How Far are We from Solving Pedestrian Detection?, arXiv:1602.01237 [LIB:1602.01237]
URL: https://ar5iv.labs.arxiv.org/html/1602.01237
- The small number of pixels is identified as the real difficulty for small pedestrians (§3.2.1). Removing
  annotation errors from TRAINING improves MR-2 from 28.63 % to 23.87 % for one detector (Table 4); the largest
  gain comes from removing annotation errors (§4.1). (1x)

## S24. CBGS, arXiv:1908.09492 (NOT BANKED)
URL: https://ar5iv.labs.arxiv.org/html/1908.09492
- nuScenes: car is 43.7 % of annotations, about 40x bicycle; class-balanced resampling grows 28,130 training
  samples to about 128k (§3). No isolated resampling ablation number found. (1x)

## S25. Class-Balanced Loss, arXiv:1901.05555 (NOT BANKED)
URL: https://ar5iv.labs.arxiv.org/html/1901.05555
- Re-weighting by inverse class frequency usually performs poorly on real long-tailed data with high imbalance
  (§1); proposes weights proportional to 1 / effective number (1 - beta^n) / (1 - beta) (§3-4). (1x)

## S26. Calibrating Deep Neural Networks using Focal Loss, arXiv:2002.09437 (NOT BANKED)
URL: https://ar5iv.labs.arxiv.org/html/2002.09437
- Focal loss upper-bounds a KL term minus gamma times the prediction entropy, i.e. it acts as a maximum-entropy
  regulariser and avoids over-confidence relative to cross-entropy (§4, appendix B). (1x)

## S27. Efficient DETR, arXiv:2104.01318 (NOT BANKED)
URL: https://ar5iv.labs.arxiv.org/html/2104.01318
- Table 1 (READ TWICE; the first read conflated rows and was WRONG; the second, verbatim reproduction is used):
  Res50 Deformable DETR, 100 proposals, 3x schedule. Encoder / decoder / decoding (auxiliary) loss / AP:
  3 / 3 / yes / 41.5; 3 / 1 / yes / 32.2; 1 / 3 / yes / 39.8; 1 / 3 / NO / 28.3.
  So at a 3-layer decoder, removing the per-layer decoding loss costs 11.5 AP. The authors attribute the
  decoder's sensitivity to its depth to that auxiliary loss (§3.1).

## S28. Progressive End-to-End Object Detection in Crowded Scenes, arXiv:2203.07669 (NOT BANKED)
URL: https://ar5iv.labs.arxiv.org/html/2203.07669
- Query-based detectors produce duplicates in crowds. CrowdHuman: 22.64 persons and 2.40 IoU>0.5 overlaps per
  image (Table 1). Sparse R-CNN baseline with 500 queries; Deformable DETR experiments with 1000 queries.
  Sparse R-CNN 90.7 AP / 44.7 MR-2 -> 92.0 / 41.4 with their method (Table 2). (1x)

## S29. OneNet, "What Makes for End-to-End Object Detection?", arXiv:2012.05780 (NOT BANKED)
URL: https://ar5iv.labs.arxiv.org/html/2012.05780
- The classification cost in the assignment is what makes a one-to-one detector NMS-free; with location cost
  only, background-like positives keep medium scores and duplicates remain (§3.3-3.4).
- Table 2 (1x): RetinaNet one-to-one, location cost only 33.6 AP (36.8 with NMS, +3.2); location + class 37.5
  (NMS +0.0). FCOS one-to-one 34.9 (37.7 with NMS) vs 38.9 (NMS +0.0).

## S30. Microsoft COCO, arXiv:1405.0312 (NOT BANKED)
URL: https://ar5iv.labs.arxiv.org/html/1405.0312
- 7.7 instances and 3.5 categories per image on average, vs ImageNet 3.0 and PASCAL 2.3 instances (§5). (1x)
