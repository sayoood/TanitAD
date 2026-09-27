#!/usr/bin/env python3
"""refcv7 A14 (SPEC_REFCV7 §19): heatmap query selection (HQS) for the BOX head -- the SHARED-FILE edits, applied ON TOP
of the A9 box-head patch (``apply_box_head_edits.py``). A SEPARATE landing variant (MAIN + HQS); the MAIN variant stays
intact.

Default ``--slot-query-select learned``: the branch builds no heat head and no position embed and calls
``box_dec(mem)`` exactly as the A9 build does -- bit-identical (a digest test pins it).

usage:  python apply_hqs_edits.py --base-dir <tree already patched by apply_box_head_edits.py> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

PB = "stack/tanitad/models/refcv6_perception_branch.py"
TR = "stack/scripts/refc_v3_train.py"
DVB = "stack/tanitad/train/declared_vs_built.py"
DVB_T = "stack/tests/test_declared_vs_built.py"
OCC_T = "stack/tests/test_occ_knob_is_stamped.py"
FILES = (PB, TR, DVB, DVB_T, OCC_T)
N_NEW_DVB = 1
E: list = []


def edit(path, eid, old, new):
    E.append((path, eid, old, new))


# -- the perception branch: the declared field, the heat head + position embed, the anchored forward, the loss ---- #
edit(PB, "pb.qs_field",
     "    vis1: bool = False\n",
     "    vis1: bool = False\n"
     "    #: ⭐ refcv7 A14 (SPEC_REFCV7 §19): the BOX head's query selection. \"learned\" (DEFAULT) = the learned query\n"
     "    #: table, the A9 build bit-identical; \"heatmap\" = HQS (``slot_query_select``): each query anchored at a cell of\n"
     "    #: the BEV centre heatmap's top-K (3x3 NMS, K = n_queries), box centre = anchor + tanh(raw) x 4 m.\n"
     "    query_select: str = \"learned\"\n")
edit(PB, "pb.qs_validate",
     "        if str(self.bev_source) not in BEV_SOURCES:\n",
     "        from tanitad.models.slot_query_select import QUERY_SELECT as _QS\n"
     "        if str(self.query_select) not in _QS:\n"
     "            raise ValueError(f\"query_select {self.query_select!r} not in {_QS}\")\n"
     "        if str(self.bev_source) not in BEV_SOURCES:\n")
edit(PB, "pb.qs_as_dict",
     "                \"vis1\": bool(self.vis1),\n",
     "                \"vis1\": bool(self.vis1),\n"
     "                \"query_select\": str(self.query_select),\n")
edit(PB, "pb.qs_none",
     "        self.box_dec: Box3DSlotDecoder | None = None\n",
     "        self.box_dec: Box3DSlotDecoder | None = None\n"
     "        #: refcv7 A14: the centre heatmap (heatmap) / the learned reference points (learned_ref) + the anchor\n"
     "        #: position embed (both anchored modes); all None under the default \"learned\"\n"
     "        self.box_heat = None\n"
     "        self.box_refpts = None\n"
     "        self.box_qpos = None\n")
edit(PB, "pb.qs_build",
     "            self.box_dec.deep_supervision = bool(cfg.deep_supervision)\n",
     "            self.box_dec.deep_supervision = bool(cfg.deep_supervision)\n"
     "            if str(cfg.query_select) == \"heatmap\":\n"
     "                if not cfg.use_bev:\n"
     "                    raise ValueError(\"[perception] ⛔ query_select 'heatmap' needs BEV features (w_map > 0 or \"\n"
     "                                     \"bev_source 'map_hires_pool'): the heatmap would read nothing\")\n"
     "                from tanitad.models.slot_query_select import AnchorPosEmbed, BEVHeatHead\n"
     "                self.box_heat = BEVHeatHead(int(cfg.bev_cfg.d_out))\n"
     "                self.box_qpos = AnchorPosEmbed(int(cfg.d_model), cfg.planner_grid)\n"
     "            elif str(cfg.query_select) == \"learned_ref\":\n"
     "                from tanitad.models.slot_query_select import AnchorPosEmbed, LearnedRefPoints\n"
     "                self.box_refpts = LearnedRefPoints(int(cfg.n_queries), cfg.planner_grid)\n"
     "                self.box_qpos = AnchorPosEmbed(int(cfg.d_model), cfg.planner_grid)\n")
edit(PB, "pb.qs_forward",
     "            out[\"box_slots\"] = self.box_dec(mem)\n",
     "            if self.box_heat is None and self.box_refpts is None:\n"
     "                out[\"box_slots\"] = self.box_dec(mem)\n"
     "            elif self.box_refpts is not None:\n"
     "                # refcv7 A14 (learned_ref): one trainable anchor per query (DAB-DETR static anchors)\n"
     "                from tanitad.models import slot_query_select as _sqs\n"
     "                _anc = self.box_refpts(int(mem.shape[0]))\n"
     "                out[\"box_slots\"] = _sqs.anchored_forward(self.box_dec, mem, _anc, self.box_qpos(_anc))\n"
     "            else:\n"
     "                # ⭐ refcv7 A14 (HQS): anchors from the BEV centre heatmap's top-K (DETACHED), the query POSITION\n"
     "                # from the anchor at every layer, the box centre anchor-relative (slot_query_select).\n"
     "                from tanitad.models import slot_query_select as _sqs\n"
     "                _heat = self.box_heat(bev_feats)\n"
     "                _anc, _sc = _sqs.choose_anchors(self, _heat)\n"
     "                out[\"box_slots\"] = _sqs.anchored_forward(self.box_dec, mem, _anc, self.box_qpos(_anc))\n"
     "                out[\"box_slots\"][\"heat_logits\"] = _heat\n"
     "                out[\"box_slots\"][\"anchor_scores\"] = _sc\n"
     "                out[\"box_slots\"][\"heat_grid\"] = _sqs.heat_grid_of(self.cfg.planner_grid, _heat)\n")
edit(PB, "pb.qs_params",
     "                 \"bev_pool\": n(self.bev_pool), \"total\": d[\"total\"]}\n"
     "        return d\n",
     "                 \"bev_pool\": n(self.bev_pool), \"total\": d[\"total\"]}\n"
     "        if getattr(self, \"box_qpos\", None) is not None:\n"
     "            # refcv7 A14 (anchored modes) only, as bev_pool: the keys appear only when built\n"
     "            _x = {\"box_qpos\": n(self.box_qpos)}\n"
     "            if getattr(self, \"box_heat\", None) is not None:\n"
     "                _x[\"box_heat\"] = n(self.box_heat)\n"
     "            if getattr(self, \"box_refpts\", None) is not None:\n"
     "                _x[\"box_refpts\"] = n(self.box_refpts)\n"
     "            d = {**{k: v for k, v in d.items() if k != \"total\"}, **_x, \"total\": d[\"total\"]}\n"
     "        return d\n")
edit(PB, "pb.qs_loss",
     "    row = {\"loss\": r[\"total\"]}\n",
     "    # ---- refcv7 A14 (HQS): the heatmap's loss, INSIDE the box term, whichever presence path ran ---------- #\n"
     "    if slots.get(\"heat_logits\") is not None:\n"
     "        from tanitad.models import slot_query_select as _sqs\n"
     "        if vis1:\n"
     "            from tanitad.data.vis1 import vis1_split as _v1s\n"
     "            _spl = _v1s(tgt, n_full=vis[\"n_full\"], n_vis=vis[\"n_vis\"], vis_known=vis[\"known\"])\n"
     "            _hpos, _hign = _spl[\"pos\"][\"valid\"].to(torch.bool), _spl[\"ignore\"]\n"
     "        else:\n"
     "            from tanitad.refs.refc_agents import visible_target_filter as _vtf\n"
     "            _hpos, _hign = _vtf(tgt)[\"valid\"].to(torch.bool), None\n"
     "        _ht = _sqs.heat_term(slots[\"heat_logits\"], tgt[\"box\"], _hpos, _hign, slots[\"heat_grid\"])\n"
     "        r = {**r, \"total\": r[\"total\"] + _sqs.HEAT_LOSS_W * _ht[\"loss\"], \"loss_heat\": _ht[\"loss\"],\n"
     "             \"n\": {**dict(r.get(\"n\") or {}), \"heat_pos\": _ht[\"n_pos\"],\n"
     "                   \"heat_pos_outside\": _ht[\"n_pos_outside\"], \"heat_ignore_cells\": _ht[\"n_ignore_cells\"]}}\n"
     "    row = {\"loss\": r[\"total\"]}\n")

# -- the trainer: the flag, the config field, the pin ------------------------------------------------------------- #
edit(TR, "tr.qs_flag",
     "    g5.add_argument(\"--vis1-sidecar\", default=None,\n",
     "    g5.add_argument(\"--slot-query-select\", choices=(\"learned\", \"heatmap\", \"learned_ref\"),\n"
     "                    default=\"learned\",\n"
     "                    help=\"refcv7 A14 (SPEC_REFCV7 §19): the BOX head's query selection. learned (default) = the \"\n"
     "                         \"learned query table, bit-identical; heatmap = HQS: each query anchored at a cell of a \"\n"
     "                         \"BEV centre heatmap's top-K (DINO mixed query selection), box centre = anchor + \"\n"
     "                         \"tanh(raw) x 4 m. Needs BEV features (--w-map > 0 or --bev-source map_hires_pool). \"\n"
     "                         \"learned_ref = one TRAINABLE anchor per query (DAB-DETR static anchors), same \"\n"
     "                         \"position embed and anchor-relative centre, no heatmap.\")\n"
     "    g5.add_argument(\"--vis1-sidecar\", default=None,\n")
edit(TR, "tr.qs_pcfg",
     "        _pcfg = _dc.replace(_pcfg, **_slot_refine_kwargs(args))\n",
     "        _pcfg = _dc.replace(_pcfg, **_slot_refine_kwargs(args),\n"
     "                            query_select=str(getattr(args, \"slot_query_select\", \"learned\") or \"learned\"))\n")
edit(TR, "tr.qs_pin",
     "    if not _slot_refine_active(args):\n"
     "        return\n"
     "    k = _slot_refine_kwargs(args)\n",
     "    if str(getattr(args, \"slot_query_select\", \"learned\") or \"learned\") == \"heatmap\":\n"
     "        if not float(getattr(args, \"w_box3d\", 0.0) or 0.0) > 0.0:\n"
     "            raise SystemExit(\"[v3] ⛔ --slot-query-select heatmap with --w-box3d 0: HQS selects the BOX head's \"\n"
     "                             \"queries; with no box head it would be stamped and train nothing (M18).\")\n"
     "        if not (float(getattr(args, \"w_map\", 0.0) or 0.0) > 0.0\n"
     "                or str(getattr(args, \"bev_source\", \"s16_lift\") or \"s16_lift\") == \"map_hires_pool\"):\n"
     "            raise SystemExit(\"[v3] ⛔ --slot-query-select heatmap needs BEV features (--w-map > 0 or \"\n"
     "                             \"--bev-source map_hires_pool): the heatmap would read nothing.\")\n"
     "    if str(getattr(args, \"slot_query_select\", \"learned\") or \"learned\") == \"learned_ref\" \\\n"
     "            and not float(getattr(args, \"w_box3d\", 0.0) or 0.0) > 0.0:\n"
     "        raise SystemExit(\"[v3] ⛔ --slot-query-select learned_ref with --w-box3d 0: the anchors belong to the \"\n"
     "                         \"BOX head; with no box head they would be stamped and train nothing (M18).\")\n"
     "    if not _slot_refine_active(args):\n"
     "        return\n"
     "    k = _slot_refine_kwargs(args)\n")

# -- G-DVB ----------------------------------------------------------------------------------------------------------- #
edit(DVB, "dvb.qs_entry",
     "_b(\"slot_vis1\", _c_slot_vis1)\n",
     "_b(\"slot_vis1\", _c_slot_vis1)\n"
     "\n"
     "\n"
     "def _c_slot_query_select(m, a):\n"
     "    want = str(_a(a, \"slot_query_select\", \"learned\") or \"learned\")\n"
     "    br = getattr(m, \"_perception\", None)\n"
     "    out = _eq(\"slot_query_select\", want == \"heatmap\", getattr(br, \"box_heat\", None) is not None,\n"
     "              \"model._perception.box_heat is not None\", \"HQS's heatmap is built iff heatmap is declared\")\n"
     "    out += _eq(\"slot_query_select\", want == \"learned_ref\", getattr(br, \"box_refpts\", None) is not None,\n"
     "               \"model._perception.box_refpts is not None\", \"the reference points are built iff learned_ref\")\n"
     "    if br is not None:\n"
     "        out += _eq(\"slot_query_select\", want, str(getattr(br.cfg, \"query_select\", \"learned\")),\n"
     "                   \"model._perception.cfg.query_select (read by the forward)\")\n"
     "    return out\n"
     "\n"
     "\n"
     "_b(\"slot_query_select\", _c_slot_query_select)\n")
edit(DVB_T, "dvbt.count", "__COUNT__", "__COUNT__")
edit(OCC_T, "occt.qs_key",
     "PERCEPTION_STAMP_KEYS = {\n",
     "PERCEPTION_STAMP_KEYS = {\n"
     "    \"query_select\",   # refcv7 A14 (HQS)\n")


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def apply(base: bytes, path: str):
    crlf, lf = base.count(b"\r\n"), base.count(b"\n")
    if crlf and crlf != lf:
        raise SystemExit(f"{path}: MIXED line endings")
    eol = "\r\n" if crlf else "\n"
    text = base.decode("utf-8").replace("\r\n", "\n")
    done = []
    for f, eid, old, new in E:
        if f != path:
            continue
        if eid == "dvbt.count":
            pat = re.compile(r"(    assert len\(dvb\.REGISTRY\) == )(\d+)([^\n]*)\n")
            m = list(pat.finditer(text))
            if len(m) != 1:
                raise SystemExit(f"ANCHOR FAILED {path} [{eid}]")
            k = int(m[0].group(2))
            text = (text[:m[0].start()] + f"{m[0].group(1)}{k + N_NEW_DVB}{m[0].group(3)}\n"
                    f"    # refcv7 A14 (HQS): +1 = --slot-query-select ({k} -> {k + N_NEW_DVB})\n" + text[m[0].end():])
            done.append(f"{eid} ({k} -> {k + N_NEW_DVB})")
            continue
        if text.count(old) != 1:
            raise SystemExit(f"ANCHOR FAILED {path} [{eid}]: found {text.count(old)}, need 1:\n{old[:300]!r}")
        text = text.replace(old, new)
        done.append(eid)
    return text.replace("\n", eol).encode("utf-8"), done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = {"files": {}}
    for path in FILES:
        base = open(os.path.join(a.base_dir, path), "rb").read()
        new, done = apply(base, path)
        dst = os.path.join(a.out, path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").write(new)
        man["files"][path] = {"base_blob_sha1": git_blob_sha1(base), "out_blob_sha1": git_blob_sha1(new),
                              "edits": done}
        print(f"{path}: {len(done)} edits")
    json.dump(man, open(os.path.join(a.out, "HQS_EDIT_MANIFEST.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
