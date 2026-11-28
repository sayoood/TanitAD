#!/usr/bin/env python
"""loc_fit.py -- one localiser fit of card PREREG_V8F_PROOF_ADD100 (STEERREAD), section 2. One fit per call. Tier T0.

  python loc_fit.py --set L0|L1|L2|L3|L4|LA [--run1 R1] [--box-dir D] [--s-root S] [--device cuda]

Reads <run1>/tok/D/<clip>.npy (arm D tokens, float16 [n, 2, 65, 1024]) of the clips the set defines (steer_common.loc_train_clips: Lk = the fit clips of
<run1>/lists/train_final.txt whose list position modulo 5 is not k; LA = all fit clips) and the box files <box-dir>/<clip>.npz (has_lead, ctr_az, ctr_el).
Target: the direction of the centre of the lead's box on frames with has_lead. Model: the token reader of card ADD99 (crop_fit.make_model embed) then an MLP 256-256-2,
azimuth = 16 tanh, elevation = 16 tanh. Smooth L1 (beta 0.5 degrees), AdamW lr 1e-3, weight decay 0.05, batch 256, 20 epochs, one warm-up epoch then cosine to 0, the last
epoch is kept, seed 0. Writes <s-root>/loc/<set>/head.pt and LOC_RECORD.json (clips read, argv, hashes, loss per epoch). Refuses a finished set and an evaluation clip in the train list.
The selection clips and the evaluation clips are never opened here.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import steer_common as SC  # noqa: E402

CFT = SC.CFT


def run_loc_fit(a, log=print):
    import torch
    t0 = time.time()
    name = a.set
    if name not in SC.LOC_NAMES:
        raise SystemExit("REFUSED: set %r is not one of %s" % (name, list(SC.LOC_NAMES)))
    rows, eval_ids = SC.read_lists(a.run1)
    fit_ids = SC.fit_clip_ids(rows)
    train = SC.loc_train_clips(name, fit_ids)
    if set(train) & set(eval_ids) or not set(train) <= set(fit_ids):
        raise SystemExit("REFUSED: the clip set of %s is not inside the fit clips" % name)
    out_dir = os.path.join(a.s_root, "loc", name)
    if os.path.exists(os.path.join(out_dir, "LOC_RECORD.json")):
        raise SystemExit("REFUSED: %s exists (a finished localiser)" % out_dir)
    az_lab, el_lab = {}, {}
    box_sha = {}
    for c in train:
        az, el, ok = SC.load_box(a.box_dir, c)
        az_lab[c], el_lab[c] = (az, ok), (el, ok)
        box_sha[c] = SC.sha256_file(os.path.join(a.box_dir, c + ".npz"))
    store = CFT.TokenStore(a.run1, "D")
    sc, sf, y_az = CFT.build_samples(train, az_lab, store, "H1")        # frames with a lead box; checks the token frame count against the box frame count
    sc2, sf2, y_el = CFT.build_samples(train, el_lab, store, "H1")
    if not (np.array_equal(sc, sc2) and np.array_equal(sf, sf2)):
        raise SystemExit("REFUSED: azimuth and elevation samples differ")
    if len(y_az) < 2:
        raise SystemExit("REFUSED: %d samples with a lead box" % len(y_az))
    Y = np.stack([y_az, y_el], 1).astype(np.float32)
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = SC.make_loc_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    N = len(Y)
    spe = int(math.ceil(N / a.batch))
    loss_fn = torch.nn.SmoothL1Loss(beta=SC.LOC_HPARAMS["beta_deg"], reduction="none")
    hist, step = [], 0
    for ep in range(a.epochs):
        model.train()
        order = rng.permutation(N)
        batches = [order[i:i + a.batch] for i in range(0, N, a.batch)]
        tl, tn = 0.0, 0
        make = lambda b: (CFT.gather(store, train, sc, sf, b, "H1"), Y[b])          # noqa: E731
        for x, y in CFT.prefetch(batches, make):
            x = torch.from_numpy(x).to(dev).float().reshape(-1, CFT.N_TOK, CFT.WIDTH)
            y = torch.from_numpy(y).to(dev)
            for g in opt.param_groups:
                g["lr"] = CFT.lr_at(step, spe, a.epochs, a.lr, a.warmup_epochs)
            loss = loss_fn(model(x), y).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += float(loss.detach()) * len(y)
            tn += len(y)
            step += 1
        hist.append({"epoch": ep, "train_loss": tl / tn})
        log("epoch %d train %.4f" % (ep, tl / tn))
    os.makedirs(out_dir, exist_ok=True)
    pt = os.path.join(out_dir, "head.pt")
    torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "set": name, "seed": a.seed, "epoch": a.epochs - 1, "card": SC.CARD}, pt)
    hp = {"lr": a.lr, "weight_decay": a.weight_decay, "batch": a.batch, "epochs": a.epochs, "warmup_epochs": a.warmup_epochs, "beta_deg": SC.LOC_HPARAMS["beta_deg"], "seed": a.seed}
    rec = {"card": SC.CARD, "tier": "T0", "set": name, "argv": list(sys.argv), "clips_read": train, "n_samples": int(N), "hparams": hp, "hparams_equal_card": bool(hp == SC.LOC_HPARAMS),
           "tool_sha256": SC.sha256_file(os.path.abspath(__file__)), "steer_common_sha256": SC.sha256_file(os.path.join(HERE, "steer_common.py")),
           "crop_fit_sha256": SC.sha256_file(CFT.__file__), "train_list_sha256": SC.sha256_file(SC.paths(a.run1)["train_list"]), "eval_list_sha256": SC.sha256_file(SC.paths(a.run1)["eval_list"]),
           "box_files_sha256": box_sha, "history": hist, "kept_epoch": a.epochs - 1, "head_pt_sha256": SC.sha256_file(pt), "device": str(dev), "torch": torch.__version__,
           "seconds": float(time.time() - t0), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    SC.write_json(os.path.join(out_dir, "LOC_RECORD.json"), rec)
    log("ZZ LOC-FIT %s samples %d last train loss %.4f" % (name, N, hist[-1]["train_loss"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--set", required=True)
    ap.add_argument("--run1", default=SC.RUN1)
    ap.add_argument("--box-dir", default=SC.BOX_DIR)
    ap.add_argument("--s-root", default=SC.S_ROOT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=SC.LOC_HPARAMS["seed"])
    ap.add_argument("--epochs", type=int, default=SC.LOC_HPARAMS["epochs"])
    ap.add_argument("--batch", type=int, default=SC.LOC_HPARAMS["batch"])
    ap.add_argument("--lr", type=float, default=SC.LOC_HPARAMS["lr"])
    ap.add_argument("--weight-decay", type=float, default=SC.LOC_HPARAMS["weight_decay"])
    ap.add_argument("--warmup-epochs", type=int, default=SC.LOC_HPARAMS["warmup_epochs"])
    return run_loc_fit(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
