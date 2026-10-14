#!/usr/bin/env python3
"""READ-ONLY checks for the P2 training-seed replicate chain (refe/p2rep_pod.sh, eval/P2REP_POD_RUNBOOK.md).

The replicate trains on inputs that P2 (refe/p2r_pod.sh) already built on this pod and that RL-S1 and S1 read beside it: the carved on-policy bank, the PDM labels, the
pixels, the init checkpoint, the DINOv3 weights.  It must NEVER rebuild, re-carve, re-download or write into them (other chains depend on them), so this helper only VERIFIES:
every open() below is a read, every path written is the caller's `--json`.  The definitions are P2's own, imported or re-derived from the same source, never retyped:

    served count      onpolicy_relabel_pdm.served_keys   (the relabeller's definition; p2r_pod.sh `served_count`)
    held-out rows     p2r_carve.log_name_of              (the carve's own row parse) over every row of every set file, against splits/sft_heldout_logs.txt
    init checkpoint   p2r_sync.PINS["model_final"]        (md5 + bytes), at p2r_sync.INIT_REL
    DINOv3 weights    the sha256 pinned in code/pod_train.sh `W_SHA[vitl16]` (parsed from the shipped file, so a pin cannot drift from the stage that fetched it)
    pixels            the `.done_1 .. .done_32` markers pod_dataprep.sh writes (the marker `ZZOK pix 32/32 shards ZZ` counts exactly these) + 32 navtrain_current_* shard dirs
    P2's own markers  weights.log `ZZOK weights ZZ`, carve.log `ZZCARVE_OK <served> <dropped>`, pix.log `ZZOK pix 32/32 shards ZZ`, chain.log `ZZP2R_DONE`

    python p2rep_checks.py tree   --pkg <pkg>                                          -> ZZP2REP_TREE_OK | ZZP2REP_TREE_REFUSED <why>
    python p2rep_checks.py inputs --data <data> --pkg <pkg> --p2-dir <refe_p2> [--json out.json] [--waive-p2-logs]
                                                                                       -> ZZP2REP_INPUTS_OK <summary> | ZZP2REP_INPUTS_REFUSED <why; why; ...>

Exit 0 / 1.  The PDM labels' completeness is NOT re-implemented here: the chain runs P2's own `pdm_label_complete.py` on them (p2r_pod.sh stage 4).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EXPECT_SERVED = 174912            # PREREG_P2: the 174,912 served TRAINING sets (the restored bank serves 177,836; the 24 held-out logs' 2,924 are carved out)
EXPECT_PIX = 32
WEIGHTS_RX = re.compile(r"W_SHA=\(\[vitl16\]=([0-9a-f]{64})\b")


def _hash(path, algo, chunk=1 << 22):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def tree_check(pkg):
    """The shipped tree is the manifest's, file for file (the launch gate re-measures the same tree: a drift is caught HERE, before the wait, not after the gate)."""
    import gate_token as GT
    mp = Path(pkg) / GT.SHIP_MANIFEST
    if not mp.is_file():
        return False, f"{GT.SHIP_MANIFEST} is missing under {pkg}"
    man = GT.read_json(mp)
    got = GT.tree_manifest(pkg)
    want = man.get("files") or {}
    if got != want:
        return False, f"the shipped tree differs from {GT.SHIP_MANIFEST} (commit {str(man.get('commit'))[:12]}): {json.dumps(GT.tree_diff(want, got))[:600]}"
    if GT.tree_digest(got) != man.get("tree_sha256"):
        return False, "tree_sha256 of the manifest does not hash its own files map"
    return True, f"{len(got)} files, commit {str(man.get('commit'))[:12]}, tree {str(man.get('tree_sha256'))[:12]}"


def weights_pin(pkg):
    """The vitl16 sha256 that code/pod_train.sh's weights stage verified on P2's pod, parsed from the SHIPPED file."""
    m = WEIGHTS_RX.search((Path(pkg) / "code" / "pod_train.sh").read_text(encoding="utf-8"))
    return m.group(1) if m else None


def _last_line(path, rx):
    last = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            m = rx.match(ln.rstrip("\n"))
            if m:
                last = m
    return last


def inputs_check(data, pkg, p2_dir, expect_served=EXPECT_SERVED, waive_p2_logs=False):
    """(reasons, details).  reasons == [] means every shared input is what P2 built and RL-S1 / S1 left in place.  Reads only."""
    data, pkg, p2 = Path(data), Path(pkg), Path(p2_dir)
    reasons, det = [], {"expect_served": expect_served, "waive_p2_logs": bool(waive_p2_logs)}

    def bad(msg):
        reasons.append(msg)

    # ---- the init checkpoint: P2's restore pin (md5 + bytes) at P2's path
    try:
        import p2r_sync as S
        md5, nbytes, _ext = S.PINS["model_final"]
        init = data / S.INIT_REL
        det["init"] = str(init)
        if not init.is_file():
            bad(f"the init checkpoint is missing: {init}")
        elif init.stat().st_size != nbytes:
            bad(f"the init checkpoint has {init.stat().st_size} bytes, P2's pin is {nbytes}: {init}")
        else:
            got = _hash(init, "md5")
            det["init_md5"] = got
            if got != md5:
                bad(f"the init checkpoint md5 {got} is not P2's pin {md5}: {init}")
    except Exception as e:                                                    # noqa: BLE001 -- a refusal, never a pass
        bad(f"init checkpoint check failed: {type(e).__name__}: {str(e)[:160]}")
    # ---- the DINOv3 ViT-L weights the stage 'weights' verified (sha256 pinned in the shipped pod_train.sh)
    try:
        want = weights_pin(pkg)
        wf = data / "backbones" / "dinov3-vitl16" / "model.safetensors"
        det["weights"] = str(wf)
        if want is None:
            bad("code/pod_train.sh carries no W_SHA[vitl16] pin to verify the backbone weights against")
        elif not wf.is_file():
            bad(f"the DINOv3 ViT-L weights are missing: {wf}")
        else:
            got = _hash(wf, "sha256")
            det["weights_sha256"] = got
            if got != want:
                bad(f"the DINOv3 ViT-L weights sha256 {got} is not the pin {want}: {wf}")
    except Exception as e:                                                    # noqa: BLE001
        bad(f"backbone weights check failed: {type(e).__name__}: {str(e)[:160]}")
    # ---- train_grow layout (p2r_sync.restore's own layout test)
    tg = data / "refe_navtrain" / "train_grow"
    lay = {"calib_table": (tg / "calib_table.json").is_file(), "scorer_targets": bool(glob.glob(str(tg / "scorer_targets*.jsonl"))),
           "targets_rank": bool(glob.glob(str(tg / "targets_rank*.jsonl")))}
    det["train_grow"] = lay
    if not all(lay.values()):
        bad(f"train_grow is incomplete under {tg}: {lay}")
    # ---- the carved on-policy bank: P2's served-count definition == 174,912, and not one row of the 24 held-out logs
    sets = data / "refe_navtrain" / "onpolicy" / "sets"
    files = sorted(glob.glob(str(sets / "onpolicy_*.jsonl")))
    det["set_files"] = len(files)
    if not files:
        bad(f"no onpolicy_*.jsonl under {sets}")
    else:
        try:
            from onpolicy_relabel_pdm import served_keys
            import p2r_carve as CV
            ho_file = pkg / "splits" / "sft_heldout_logs.txt"
            ho = set(ho_file.read_text(encoding="utf-8").split())
            det["heldout_logs"] = len(ho)
            if not ho:
                bad(f"{ho_file} lists no held-out log")
            best = served_keys(files)
            det["served"] = len(best)
            if len(best) != expect_served:
                bad(f"the on-policy bank serves {len(best)} sets, P2 trained on {expect_served} (the carve was undone, or a chain changed {sets})")
            leaked_keys = sum(1 for k in best if k[0] in ho)
            n_rows = n_ho = 0
            for fp in files:
                with open(fp, encoding="utf-8") as fh:
                    for line in fh:
                        n_rows += 1
                        if CV.log_name_of(line) in ho:
                            n_ho += 1
            det.update(rows=n_rows, heldout_served_keys=leaked_keys, heldout_rows=n_ho)
            if leaked_keys or n_ho:
                bad(f"the bank holds {n_ho} rows ({leaked_keys} served sets) of the 24 SFT held-out logs: it is not the carved bank P2 trained on")
        except Exception as e:                                                # noqa: BLE001
            bad(f"on-policy bank check failed: {type(e).__name__}: {str(e)[:200]}")
    # ---- the PDM label files exist (completeness is P2's own pdm_label_complete.py, run by the chain)
    pdm = data / "refe_sft2" / "train_pdm"
    det["pdm_files"] = len(glob.glob(str(pdm / "pdm_*.jsonl")))
    if not det["pdm_files"]:
        bad(f"no pdm_*.jsonl under {pdm}")
    # ---- the pixels: pod_dataprep.sh's marker counts `.done_*` under navtrain_pixels
    pix = data / "navtrain_pixels"
    done = [n for n in range(1, EXPECT_PIX + 1) if (pix / f".done_{n}").is_file()]
    dirs = [d for d in glob.glob(str(pix / "navtrain_current_*")) if os.path.isdir(d)]
    det.update(pix_done=len(done), pix_dirs=len(dirs))
    if len(done) != EXPECT_PIX:
        bad(f"the pixel shards are incomplete: .done_1..{EXPECT_PIX} present for {len(done)} ({pix})")
    if len(dirs) < EXPECT_PIX:
        bad(f"fewer than {EXPECT_PIX} navtrain_current_* shard dirs under {pix} ({len(dirs)})")
    # ---- the navtrain DBs: INFORMATIONAL.  train.py never reads them (only the PDM relabeller did, in P2); a shortage cannot move the replicate.
    det["navtrain_dbs"] = len(glob.glob(str(data / "navtrain_dbs" / "*.db")))
    # ---- P2's own markers (provenance): the chain that built these inputs said so
    if waive_p2_logs:
        det["p2_markers"] = "WAIVED (P2REP_P2_PROVENANCE_WAIVED=1; the content checks above stay)"
    else:
        mk = {}

        def has(path, needle):
            try:
                return any(ln.startswith(needle) for ln in open(path, encoding="utf-8", errors="replace"))
            except OSError:
                return False
        mk["weights_log"] = has(p2 / "weights.log", "ZZOK weights ZZ")
        mk["p2r_done"] = any("ZZP2R_DONE" in ln for ln in _lines(p2 / "chain.log"))          # chain.log lines carry a timestamp prefix (p2r_pod.sh `log`); rls1_pod.sh greps the same token
        try:
            m = _last_line(p2 / "carve.log", re.compile(r"^ZZCARVE_OK (\d+) (\d+)$"))
            mk["carve_ok"] = bool(m) and int(m.group(1)) == expect_served
        except OSError:
            mk["carve_ok"] = False
        try:
            m = _last_line(p2 / "pix.log", re.compile(r"^ZZOK pix (\d+)/32 shards ZZ$"))
            mk["pix_ok"] = bool(m) and int(m.group(1)) == EXPECT_PIX
        except OSError:
            mk["pix_ok"] = False
        det["p2_markers"] = mk
        for k, v in mk.items():
            if not v:
                bad(f"P2's own marker is absent: {k} (in {p2}); P2REP_P2_PROVENANCE_WAIVED=1 only on the Master Mind's word")
    return reasons, det


def _lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines()
    except OSError:
        return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tree")
    t.add_argument("--pkg", required=True)
    i = sub.add_parser("inputs")
    i.add_argument("--data", required=True)
    i.add_argument("--pkg", required=True)
    i.add_argument("--p2-dir", required=True)
    i.add_argument("--json")
    i.add_argument("--expect-served", type=int, default=EXPECT_SERVED)
    i.add_argument("--waive-p2-logs", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "tree":
        ok, why = tree_check(a.pkg)
        print(("ZZP2REP_TREE_OK " if ok else "ZZP2REP_TREE_REFUSED ") + why, flush=True)
        return 0 if ok else 1
    reasons, det = inputs_check(a.data, a.pkg, a.p2_dir, a.expect_served, a.waive_p2_logs)
    det["verdict"], det["reasons"] = ("OK" if not reasons else "REFUSED"), reasons
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(det, indent=1, sort_keys=True), encoding="utf-8", newline="\n")
    if reasons:
        print("ZZP2REP_INPUTS_REFUSED " + "; ".join(reasons)[:900], flush=True)
        return 1
    print(f"ZZP2REP_INPUTS_OK served={det['served']} rows={det['rows']} heldout_rows=0 pdm_files={det['pdm_files']} pix={det['pix_done']}/{EXPECT_PIX} "
          f"init_md5={det['init_md5'][:12]} weights_sha256={det['weights_sha256'][:12]} dbs={det['navtrain_dbs']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
