#!/usr/bin/env python3
"""+R6 (SPEC_REFCV7 §15.5 A10.1): denoising queries for the box head -- the SHARED-FILE edits, applied ON TOP of the
A9 box-head patch (``apply_box_head_edits.py``). ⛔ BANKED, NOT IN THE LANDING PACKAGE: it lands only if G-BOX-OVERFIT's
MAIN arm fails criterion 2 or 3 with presence at its base rate (A10.1's trigger).

Default OFF: ``--slot-dn-groups 0`` builds no embedder, emits no memory, adds no loss -- bit-identical to the A9 build.

usage:  python apply_r6_edits.py --base-dir <tree already patched by apply_box_head_edits.py> --out <dir>
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
CW_T = "stack/tests/test_cls_weight_stamp.py"         # NOT an A9 file: its base is the TIP blob
FILES = (PB, TR, DVB, DVB_T, OCC_T, CW_T)
N_NEW_DVB = 1
E: list = []


def edit(path, eid, old, new):
    E.append((path, eid, old, new))


# -- the perception branch: the declared field, the embedder, the memory for the DN pass ------------------- #
edit(PB, "pb.dn_field",
     "    deep_supervision: bool = False\n"
     "    vis1: bool = False\n",
     "    deep_supervision: bool = False\n"
     "    vis1: bool = False\n"
     "    #: ⭐ +R6 (SPEC_REFCV7 §15.5 A10.1): denoising-query GROUPS for the box head, TRAINING ONLY. 0 = off (the\n"
     "    #: A9 build, bit-identical). A10.1's arm uses 5 (``slot_denoise.DN_GROUPS_ARM``).\n"
     "    dn_groups: int = 0\n")
edit(PB, "pb.dn_as_dict",
     "                \"vis1\": bool(self.vis1),\n",
     "                \"vis1\": bool(self.vis1),\n"
     "                \"dn_groups\": int(self.dn_groups),\n")
edit(PB, "pb.dn_none",
     "        self.box_mem: Box3DMemory | None = None\n"
     "        self.box_dec: Box3DSlotDecoder | None = None\n",
     "        self.box_mem: Box3DMemory | None = None\n"
     "        self.box_dec: Box3DSlotDecoder | None = None\n"
     "        #: +R6: the denoising-query embedder (training only; outside the decoder's §6 band)\n"
     "        self.box_dn = None\n")
edit(PB, "pb.dn_build",
     "            self.box_dec.deep_supervision = bool(cfg.deep_supervision)\n",
     "            self.box_dec.deep_supervision = bool(cfg.deep_supervision)\n"
     "            if int(cfg.dn_groups) > 0:\n"
     "                from tanitad.models.agent_slots import AGENT_CLASSES as _AC\n"
     "                from tanitad.models.slot_denoise import DenoiseQueryEmbed\n"
     "                self.box_dn = DenoiseQueryEmbed(\n"
     "                    int(cfg.d_model), len(_AC), x_range=float(self.box_dec.ranges.x_fwd_m),\n"
     "                    y_range=float(self.box_dec.ranges.y_half_m), z_range=float(self.box_dec.z_range_m))\n")
edit(PB, "pb.dn_memory",
     "            out[\"box_slots\"] = self.box_dec(mem)\n",
     "            out[\"box_slots\"] = self.box_dec(mem)\n"
     "            if self.box_dn is not None and self.training:\n"
     "                out[\"box_memory\"] = mem          # +R6: the DN pass reads the SAME memory\n")
edit(PB, "pb.dn_params",
     "                 \"bev_pool\": n(self.bev_pool), \"total\": d[\"total\"]}\n"
     "        return d\n",
     "                 \"bev_pool\": n(self.bev_pool), \"total\": d[\"total\"]}\n"
     "        if getattr(self, \"box_dn\", None) is not None:\n"
     "            # +R6 (A10.1) only, as bev_pool: the DN embedder's key appears only when it is built\n"
     "            d = {**{k: v for k, v in d.items() if k != \"total\"},\n"
     "                 \"box_dn\": n(self.box_dn), \"total\": d[\"total\"]}\n"
     "        return d\n")

# -- the trainer: the flag, the pin, the config field, the loss --------------------------------------------- #
edit(TR, "tr.dn_flag",
     "    g5.add_argument(\"--vis1-sidecar\", default=None,\n",
     "    g5.add_argument(\"--slot-dn-groups\", type=int, default=0,\n"
     "                    help=\"+R6 (SPEC_REFCV7 A10.1): denoising-query groups for the BOX head, training only \"\n"
     "                         \"(DN-DETR/DINO-style: noised GT positives + a no-object negative group, masked per \"\n"
     "                         \"group, reconstructed at every layer). 0 (default) = off, bit-identical.\")\n"
     "    g5.add_argument(\"--vis1-sidecar\", default=None,\n")
edit(TR, "tr.dn_pcfg",
     "        _pcfg = _dc.replace(_pcfg, **_slot_refine_kwargs(args))\n",
     "        _pcfg = _dc.replace(_pcfg, **_slot_refine_kwargs(args),\n"
     "                            dn_groups=int(getattr(args, \"slot_dn_groups\", 0) or 0))\n"
     "        model._dn_gen = torch.Generator().manual_seed(int(getattr(args, \"seed\", 0)) + 7919)\n")
edit(TR, "tr.dn_pin",
     "    if not _slot_refine_active(args):\n"
     "        return\n"
     "    k = _slot_refine_kwargs(args)\n",
     "    _dng = int(getattr(args, \"slot_dn_groups\", 0) or 0)\n"
     "    if _dng < 0:\n"
     "        raise SystemExit(\"[v3] ⛔ --slot-dn-groups must be >= 0.\")\n"
     "    if _dng > 0 and not float(getattr(args, \"w_box3d\", 0.0) or 0.0) > 0.0:\n"
     "        raise SystemExit(\"[v3] ⛔ --slot-dn-groups > 0 with --w-box3d 0: the denoising queries feed the \"\n"
     "                         \"BOX head only; with no box head they would be stamped and train nothing (M18).\")\n"
     "    if not _slot_refine_active(args):\n"
     "        return\n"
     "    k = _slot_refine_kwargs(args)\n")
edit(TR, "tr.dn_loss",
     "                loss = loss + _w_b3d * _brow[\"loss\"]\n"
     "                if _vis3 is not None:\n",
     "                # ⭐ +R6 (A10.1): the denoising pass, TRAINING only, added INTO the box term so every\n"
     "                # reader of `box3d` (the loss, the log, G-BOX-OVERFIT) sees the same number.\n"
     "                if getattr(_br, \"box_dn\", None) is not None and model.training \\\n"
     "                        and \"box_memory\" in _pout:\n"
     "                    from tanitad.models import slot_denoise as _sdn\n"
     "                    _mem3 = (_pout[\"box_memory\"].index_select(0, _sel3)\n"
     "                             if _b3_keep is not None else _pout[\"box_memory\"])\n"
     "                    if _vis3 is not None:\n"
     "                        _tp3 = _vis1.vis1_split(_t3, n_full=_vis3[\"n_full\"], n_vis=_vis3[\"n_vis\"],\n"
     "                                                vis_known=_vis3[\"known\"])[\"pos\"]\n"
     "                    else:\n"
     "                        _tp3 = _refc_agents.visible_target_filter(_t3)\n"
     "                    _dnq = _sdn.make_dn_queries(_br.box_dn, _tp3, groups=int(_br.cfg.dn_groups),\n"
     "                                                gen=model._dn_gen)\n"
     "                    if _dnq[\"q\"] is not None:\n"
     "                        _dnl = _sdn.dn_losses(_sdn.dn_forward(_br.box_dec, _mem3, _dnq), _tp3, _dnq,\n"
     "                                              cls_class_weight=getattr(model, \"_cls_class_weight\", None))\n"
     "                        loss = loss + _w_b3d * _dnl[\"total\"]\n"
     "                        _brow[\"loss\"] = _brow[\"loss\"] + _dnl[\"total\"]\n"
     "                        for _kd, _vd in _dnl[\"parts\"].items():\n"
     "                            extra[f\"box3d_{_kd}\"] = _vd\n"
     "                if _vis3 is not None:\n")

# -- G-DVB ---------------------------------------------------------------------------------------------------- #
edit(DVB, "dvb.dn_entry",
     "_b(\"slot_vis1\", _c_slot_vis1)\n",
     "_b(\"slot_vis1\", _c_slot_vis1)\n"
     "\n"
     "\n"
     "def _c_slot_dn(m, a):\n"
     "    want = int(_a(a, \"slot_dn_groups\", 0) or 0)\n"
     "    br = getattr(m, \"_perception\", None)\n"
     "    out = _eq(\"slot_dn_groups\", want > 0, getattr(br, \"box_dn\", None) is not None,\n"
     "              \"model._perception.box_dn is not None\", \"+R6's embedder is built iff the groups are > 0\")\n"
     "    if br is not None:\n"
     "        out += _eq(\"slot_dn_groups\", want, int(getattr(br.cfg, \"dn_groups\", 0)),\n"
     "                   \"model._perception.cfg.dn_groups (read by the loss)\")\n"
     "    return out\n"
     "\n"
     "\n"
     "_b(\"slot_dn_groups\", _c_slot_dn)\n")
edit(DVB_T, "dvbt.count", "__COUNT__", "__COUNT__")
edit(OCC_T, "occt.dn_key",
     "PERCEPTION_STAMP_KEYS = {\n",
     "PERCEPTION_STAMP_KEYS = {\n"
     "    \"dn_groups\",   # +R6 (A10.1), banked unlanded\n")


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
                    f"    # +R6 (A10.1, banked): +1 = --slot-dn-groups ({k} -> {k + N_NEW_DVB})\n" + text[m[0].end():])
            done.append(f"{eid} ({k} -> {k + N_NEW_DVB})")
            continue
        if text.count(old) != 1:
            raise SystemExit(f"ANCHOR FAILED {path} [{eid}]: found {text.count(old)}, need 1:\n{old[:300]!r}")
        text = text.replace(old, new)
        done.append(eid)
    return text.replace("\n", eol).encode("utf-8"), done


# -- test_cls_weight_stamp.py: the DN loss is a THIRD box-loss call site passing the same class weight ------- #
edit(CW_T, "cwt.sites",
     "    n = src.count('cls_class_weight=getattr(model, \"_cls_class_weight\", None)')\n"
     "    assert n == 2, f\"expected the weight at BOTH loss sites, found {n}\"\n",
     "    n = src.count('cls_class_weight=getattr(model, \"_cls_class_weight\", None)')\n"
     "    # +R6 (SPEC_REFCV7 A10.1): the box head's DN (denoising-query) loss is a THIRD call site of the SAME\n"
     "    # weight -- the agent head, the box head's matched loss, and its DN loss.\n"
     "    assert n == 3, f\"expected the weight at all THREE loss sites (+R6), found {n}\"\n")


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
    json.dump(man, open(os.path.join(a.out, "R6_EDIT_MANIFEST.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
