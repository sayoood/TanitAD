"""pair_tokgrid.py -- pod side of card PREREG_V8F_PROOF_ADD87 (TOKGRID). Writes PAIR_TOKGRID.json into --out-dir (refuses if the folder exists) and prints no value, only ZZ PAIR-TOKGRID DONE.
For H 25 and H 10: the paired difference of the absolute error, the fine-grid fit minus the card-ADD86 fit of the same arm and seed, both read from their saved _table.npz on the scored frames
(rows of card ADD78, reader seeds 0 and 1), two-level bootstrap as in the scorer (reader seeds, then evaluation episodes; B draws, percentile 2.5 / 97.5), through the routines of the generated tool.
Also, per seed, the standard deviation and the lag-1 autocorrelation (along frames within a clip; pairs of rows of one clip whose frames differ by 1) of the signed error (estimate minus label) of T25 on the fine grid.
The scorer's data arguments are accepted; those this file does not need (est-dir, add78-result, eval-cache, join-eval, build-dir) are ignored."""
import argparse, io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import tokclose_fit_g8 as TG  # noqa: E402

CF = TG.CF


def seed_stats(err, clip, frame):
    """standard deviation of the error and lag-1 autocorrelation over pairs (same clip, frame + 1)."""
    clip, frame = np.asarray(clip), np.asarray(frame, np.int64)
    key = {(str(c), int(f)): i for i, (c, f) in enumerate(zip(clip, frame))}
    a, b = [], []
    for (c, f), i in key.items():
        j = key.get((c, f + 1))
        if j is not None:
            a.append(err[i]), b.append(err[j])
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ac = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")
    return {"sd_error": float(np.std(err)), "lag1_autocorr": ac, "n_pairs": int(len(a)), "n_rows": int(len(err))}


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--fit-dir-fine", required=True)
    p.add_argument("--fit-dir-coarse", required=True)
    p.add_argument("--dump-dir", required=True)
    p.add_argument("--episode-map", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--boot", type=int, default=2000)
    for n in ("est-dir", "add78-result", "eval-cache", "join-eval", "build-dir"):
        p.add_argument("--" + n, default="")
    a = p.parse_args()
    if os.path.exists(a.out_dir):
        print("REFUSED: output folder exists: " + a.out_dir)
        return 2
    ks = [int(x) for x in a.seeds.split(",")]
    ep_map = json.load(io.open(a.episode_map, encoding="utf-8")) if a.episode_map else None
    dumps = {k: CF.load_dump(os.path.join(a.dump_dir, "reader_s%d_score.npz" % k)) for k in ks}
    out = {"card": "PREREG_V8F_PROOF_ADD87", "tier": "T0", "reader_seeds": ks,
           "bootstrap": {"B": a.boot, "levels": "reader seeds, then evaluation episodes", "interval": "percentile 2.5 / 97.5"},
           "difference": "fine-grid fit minus card-ADD86 fit, paired absolute error on the scored frames", "grids": {}}
    for H in (25, 10):
        arm = "T%d" % H
        rows_list = [CF.build_rows(dumps[k], H, ep_map) for k in ks]
        fine, coarse = [], []
        for k, rows in zip(ks, rows_list):
            ci = TG.row_clip_index(dumps[k], rows)
            for d, dst in ((a.fit_dir_fine, fine), (a.fit_dir_coarse, coarse)):
                e = TG.table_lookup(TG.load_table(os.path.join(d, "tokclose_%s_table.npz" % TG.tag_of(arm, k))), ci, rows["frame"])
                if not np.isfinite(e).all():
                    print("REFUSED: the table of %s seed %d in %s lacks scored rows" % (arm, k, d))
                    return 2
                dst.append(e)
            for d, key in ((a.fit_dir_fine, "fine"), (a.fit_dir_coarse, "coarse")):
                lg = json.load(io.open(os.path.join(d, "tokclose_%s_log.json" % TG.tag_of(arm, k)), encoding="utf-8"))
                out["grids"]["%s_s%d_%s" % (arm, k, key)] = {"grid": lg.get("grid", "pool4x10"), "n_tok": lg["dims"]["n_tok"]}
        d = TG.pair_vs_cell(fine, coarse, rows_list, a.boot, 1000 + H)
        out["H%d" % H] = {"arm": arm, "paired_diff_abs_err_fine_minus_coarse": d[0], "ci": [d[1], d[2]], "n_frames": int(sum(len(r["y"]) for r in rows_list)),
                          "mae_fine": float(np.mean(np.concatenate([np.abs(e - r["y"]) for e, r in zip(fine, rows_list)]))),
                          "mae_coarse": float(np.mean(np.concatenate([np.abs(e - r["y"]) for e, r in zip(coarse, rows_list)])))}
        if H == 25:
            out["T25_fine_error_per_seed"] = {str(k): seed_stats(e - r["y"], r["clip"], r["frame"]) for k, e, r in zip(ks, fine, rows_list)}
    os.makedirs(a.out_dir)
    with io.open(os.path.join(a.out_dir, "PAIR_TOKGRID.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=float)
    print("ZZ PAIR-TOKGRID DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
