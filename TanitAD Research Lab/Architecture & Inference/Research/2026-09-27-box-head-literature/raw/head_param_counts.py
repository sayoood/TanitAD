"""Head parameter counts from PUBLISHED config shapes (arithmetic only; no framework, no download).

Every count below is the decoder of the detection head plus its query table and output branches, NOT the
image backbone / neck / BEV encoder. Layer layouts:
  * DETR / Deformable DETR: official repo modules (detr/models/transformer.py, deformable_transformer.py,
    ms_deform_attn.py) -- MultiheadAttention = in_proj 3d^2+3d + out_proj d^2+d; MSDeformAttn =
    sampling_offsets d->H*L*P*2, attention_weights d->H*L*P, value_proj d->d, output_proj d->d.
  * DETR3D / BEVFormer / PETR: mmdet DeformableDETRHead branch layout with num_reg_fcs=2 and
    with_box_refine=True (one cls + one reg branch PER decoder layer). DETR3D cross-attention =
    Detr3DCrossAtten(attention_weights d->cams*levels*points, output_proj d->d, position_encoder
    MLP 3->d->d with LayerNorms). ⚠️ These layouts are recalled from the mmdet / project code, not re-read
    online in this pass: the TRANSFORMER-LAYER counts (from config num_layers / embed_dims /
    feedforward_channels) are the reliable part; branch/PE totals are UNVERIFIED to +-10 %.
OURS is read from the checkpoint / config.json, not computed here.
"""
import json, sys

def lin(i, o, bias=True):
    return i * o + (o if bias else 0)

def ln(d):
    return 2 * d

def mha(d):
    return lin(d, 3 * d) + lin(d, d)

def ffn(d, f):
    return lin(d, f) + lin(f, d)

def msdeform(d, heads, levels, points):
    return lin(d, heads * levels * points * 2) + lin(d, heads * levels * points) + lin(d, d) + lin(d, d)

def mlp(dims):
    return sum(lin(a, b) for a, b in zip(dims[:-1], dims[1:]))

out = {}
d = 256
# DETR: 6 layers, FFN 2048, 100 queries, 91+1 classes, bbox MLP 3 layers, final decoder LayerNorm
layer = mha(d) + mha(d) + ffn(d, 2048) + 3 * ln(d)
out["DETR (R50, COCO)"] = dict(decoder_layers=6 * layer + ln(d), queries=100 * d,
                               heads=lin(d, 92) + mlp([d, d, d, 4]))
# Deformable DETR (single-stage, no refine): 6 layers, FFN 1024, 300 queries (query_embed 2d), M8 L4 K4
layer = mha(d) + msdeform(d, 8, 4, 4) + ffn(d, 1024) + 3 * ln(d)
out["Deformable DETR"] = dict(decoder_layers=6 * layer, queries=300 * 2 * d, ref_points=lin(d, 2),
                              heads=lin(d, 91) + mlp([d, d, d, 4]))
# DETR3D: 6 layers, FFN 512, 900 queries (2d), Detr3DCrossAtten cams 6, levels 4, points 1, refine
xattn = lin(d, 6 * 4 * 1) + lin(d, d) + (lin(3, d) + ln(d) + lin(d, d) + ln(d))
layer = mha(d) + xattn + ffn(d, 512) + 3 * ln(d)
cls_b = 2 * (lin(d, d) + ln(d)) + lin(d, 10)
reg_b = 2 * lin(d, d) + lin(d, 10)
out["DETR3D"] = dict(decoder_layers=6 * layer, queries=900 * 2 * d, ref_points=lin(d, 3),
                     heads=6 * (cls_b + reg_b))
# BEVFormer detection decoder: 6 layers, FFN 512, CustomMSDeformableAttention levels 1, heads 8, points 4
layer = mha(d) + msdeform(d, 8, 1, 4) + ffn(d, 512) + 3 * ln(d)
out["BEVFormer (head only)"] = dict(decoder_layers=6 * layer, queries=900 * 2 * d, ref_points=lin(d, 3),
                                    heads=6 * (cls_b + reg_b))
# PETR: 6 layers, FFN 2048, standard MHA cross-attention, 900 queries from 3-D anchor points via MLP
layer = mha(d) + mha(d) + ffn(d, 2048) + 3 * ln(d)
out["PETR (p4 config)"] = dict(decoder_layers=6 * layer, queries=900 * 3 + mlp([3 * d // 2, d, d]),
                               heads=6 * (cls_b + reg_b))
# MonoDETR: 3 layers, FFN 256, deformable cross-attn (4 levels assumed) + depth cross-attn (MHA), 50 queries
layer = mha(d) + msdeform(d, 8, 4, 4) + mha(d) + ffn(d, 256) + 4 * ln(d)
out["MonoDETR (KITTI)"] = dict(decoder_layers=3 * layer, queries=50 * 2 * d, notes="levels=4 assumed (UNVERIFIED)")
for k, v in out.items():
    v["total"] = sum(x for x in v.values() if isinstance(x, int))
# OURS -- measured from the checkpoint (mmap read of ckpt_step38000.pt) and config.json
our_layer = mha(d) + mha(d) + ffn(d, 1024) + 3 * ln(d)
out["OURS box3d head (refcv6)"] = dict(decoder_layers=3 * our_layer, mem_pos_learned_PE=2144 * d,
                                       mem_proj=lin(d, d), queries=100 * d, final_norm=ln(d),
                                       head=lin(d, 23), total_decoder_from_config=3806999,
                                       box_memory_from_config=287744, total=3806999 + 287744)
out["OURS agent head (refcv6)"] = dict(decoder_layers=3 * our_layer, mem_pos_learned_PE=416 * d,
                                       mem_proj=lin(2048, d), queries=100 * d, final_norm=ln(d),
                                       head=lin(d, 21), total_from_ckpt=3822869, total=3822869)
chk = (3 * our_layer + 2144 * d + lin(d, d) + 100 * d + ln(d) + lin(d, 23))
out["_self_check_box3d_decoder_recomputed"] = chk
chk2 = (3 * our_layer + 416 * d + lin(2048, d) + 100 * d + ln(d) + lin(d, 21))
out["_self_check_agent_head_recomputed"] = chk2
json.dump(out, sys.stdout, indent=1)
