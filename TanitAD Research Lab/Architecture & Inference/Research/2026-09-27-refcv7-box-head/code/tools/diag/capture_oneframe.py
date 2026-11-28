#!/usr/bin/env python3
"""capture_oneframe.py -- the DECODER BENCH's input, captured once on the run host (CPU is enough).

The one-frame ladder showed the same failure with the trunk FROZEN (only the box memory + decoder train): the
assignment never stabilises. So the box memory's INPUTS for that frame are a fixed tensor pair, and the failure can
be studied on the box head alone, in seconds instead of minutes. This builds the model and the 16 frames through the
harness's own adapter (the corrected A13 config: the refcv7 canonical argv + the A9 flags), runs ONE eval-mode forward
of the ladder's frame, and saves:
  * the box memory's inputs (the stride-16 map and the pooled BEV) exactly as the forward passed them;
  * ``box3d_loss_row``'s targets and keyword arguments (the VIS-1 block, the class weight, the loss switches);
  * the box memory and box decoder MODULES at their seed-0 init (pickled whole, so the bench rebuilds nothing).

usage: python capture_oneframe.py <tree> <argv.json> <audit dir> <out.pt>
"""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    tree, argv_file, audit, out = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    p = tree / "stack" / "scripts" / "g_box_overfit.py"
    spec = importlib.util.spec_from_file_location("g_box_overfit_cap", str(p))
    G = importlib.util.module_from_spec(spec)
    sys.modules["g_box_overfit_cap"] = G
    spec.loader.exec_module(G)
    launch_argv, argv_rec = G.load_launch_argv(argv_file)
    build_argv, _ = G.harness_build_argv(launch_argv)
    tr = G._load_by_path("refc_v3_train_for_gbo", G.SCRIPTS / "refc_v3_train.py")
    L, loader_rec = G.load_vendored_loader()
    frameset = json.loads((audit / "raw" / "visibility" / "gbo_frameset.json").read_text(encoding="utf-8"))
    ad = G.TrainerAdapter(tr, L, build_argv, frameset, "cpu")
    fi = int(np.argmax([int(x[2]) for x in ad.per_frame]))
    ad.setup("main", G.SEED)
    br = ad.model._perception
    cap = {}

    def mem_hook(_m, args, outp):
        cap["mem_in"] = [a.detach().clone() if torch.is_tensor(a) else a for a in args]
        cap["mem_out_eval"] = outp.detach().clone()
    h = br.box_mem.register_forward_hook(mem_hook)
    orig_row = tr._perc.box3d_loss_row

    def row(slots, tgt, **kw):
        cap["tgt"] = {k: v.detach().clone() for k, v in tgt.items()}
        kk = {}
        for k, v in kw.items():
            if torch.is_tensor(v):
                kk[k] = v.detach().clone()
            elif isinstance(v, dict):
                kk[k] = {a: (b.detach().clone() if torch.is_tensor(b) else b) for a, b in v.items()}
            else:
                kk[k] = v
        cap["kw"] = kk
        return orig_row(slots, tgt, **kw)
    tr._perc.box3d_loss_row = row
    ad.model.eval()
    try:
        with torch.no_grad():
            r = ad._loss_row(ad._batch([fi]))
    finally:
        tr._perc.box3d_loss_row = orig_row
        h.remove()
    if "mem_in" not in cap or "tgt" not in cap:
        raise SystemExit("[capture] the box memory or box3d_loss_row was not reached")
    rec = {"frame_index": fi, "frame": {"sha12": ad.per_frame[fi][0], "t": int(ad.per_frame[fi][1]),
                                        "n_pos": int(ad.per_frame[fi][2]), "n_ign": int(ad.per_frame[fi][3])},
           "mem_in": cap["mem_in"], "mem_out_eval": cap["mem_out_eval"], "tgt": cap["tgt"], "kw": cap["kw"],
           "box_mem": br.box_mem.cpu(), "box_dec": br.box_dec.cpu(), "pb_cfg": br.cfg.as_dict(),
           "box3d_eval_row": {k: float(v) for k, v in r.items() if torch.is_tensor(v) and v.ndim == 0
                              and k.startswith("box3d")},
           "argv": argv_rec, "loader": loader_rec, "tree": str(tree),
           "harness_md5": hashlib.md5(p.read_bytes()).hexdigest(),
           "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    torch.save(rec, out)
    print(f"[capture] frame {rec['frame']} -> {out}; mem_in shapes "
          f"{[tuple(a.shape) if torch.is_tensor(a) else a for a in rec['mem_in']]}; targets "
          f"{int(rec['tgt']['valid'].sum())} valid rows; eval box3d {rec['box3d_eval_row'].get('box3d_presence')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
