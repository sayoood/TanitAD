"""AMEND_HWMPLAN_6: the clip selection for RA6's fit set (section 3 of the card). CPU, reads two local copies of box 2's manifests; nothing is pulled.
Rule (committed in the card before this ran):
  eligible = the 4,512 clips M2b trained on (box 2 manifest sorted by sha12, positions 540:600 excluded; sha256 of the excluded list f6a1dee0...)
             minus the 141 clips of RA's current fit set (hwm_plan/ts/clips141.json)
             minus every clip of box 2's EVAL manifest (141 eval episodes; this contains the 71 SCORE episodes) and every SCORE / TRAIN-selection window eid.
  order    = np.random.RandomState(SEED).permutation(eligible sorted by clip id); the first N_NEW are the new fit clips; every prefix is a seeded random sample (nested sets).
Writes clips6.json (clips, md5, n_frames, bytes, nested checkpoints, sha256s).  Refuses on any overlap."""
import hashlib, json, os, sys
import numpy as np

SEED = 20261117
N_NEW = 859
CHECKPOINTS_NEW = (141, 423, 859)          # nested prefixes -> total fit clips 282, 564, 1000
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAN_TRAIN = ROOT + "/work/manifest_box2.jsonl"
MAN_EVAL = ROOT + "/work/manifest_eval_box2.jsonl"
CLIPS141 = "F:/TanitAD-scratch/hwm_plan/ts/clips141.json"
EXCLUDED_SHA = "f6a1dee02af0"


class Refused(Exception):
    pass


def select(man_train, man_eval, clips141, score_eids=(), seed=SEED, n_new=N_NEW):
    """-> dict. man_*: lists of manifest rows; clips141: the dict of clips141.json; score_eids: eids of SCORE and TRAIN-selection windows."""
    S = sorted(man_train, key=lambda m: m["sha12"])
    ex = [m["clip"] for m in S[540:600]]
    if hashlib.sha256("\n".join(ex).encode()).hexdigest()[:12] != EXCLUDED_SHA:
        raise Refused("the excluded-60 list does not hash to f6a1dee0: the manifest order is not the trainer's")
    ex_set = set(ex)
    c141 = list(clips141["clips"])
    if hashlib.sha256("\n".join(c141).encode()).hexdigest() != clips141["clips_sha256"]:
        raise Refused("clips141 sha256 mismatch")
    bad = set(c141) & ex_set
    if bad:
        raise Refused(f"{len(bad)} of the 141 are in the excluded 60")
    ev = {m["clip"] for m in man_eval} | set(score_eids)
    elig = sorted(m["clip"] for m in S if m["clip"] not in ex_set and m["clip"] not in set(c141) and m["clip"] not in ev)
    if len(elig) < n_new:
        raise Refused(f"only {len(elig)} eligible clips")
    perm = np.random.RandomState(seed).permutation(len(elig))
    new = [elig[i] for i in perm[:n_new]]
    by = {m["clip"]: m for m in man_train}
    assert not (set(new) & (set(c141) | ex_set | ev)), "overlap"
    return dict(seed=seed, n_new=n_new, n_eligible=len(elig), clips=new, md5={c: by[c]["md5"] for c in new}, n_frames={c: by[c]["n_frames"] for c in new}, bytes={c: by[c]["bytes"] for c in new},
                clips_sha256=hashlib.sha256("\n".join(new).encode()).hexdigest(), clips141_sha256=clips141["clips_sha256"], excluded_sha12=EXCLUDED_SHA,
                checkpoints_new=list(CHECKPOINTS_NEW), checkpoints_total=[141 + c for c in CHECKPOINTS_NEW], n_eval_ids_refused=len(ev),
                total_bytes=int(sum(by[c]["bytes"] for c in new)))


def main():
    import numpy as _np
    mt = [json.loads(l) for l in open(MAN_TRAIN, encoding="utf-8")]
    me = [json.loads(l) for l in open(MAN_EVAL, encoding="utf-8")]
    c141 = json.load(open(CLIPS141, encoding="utf-8"))
    eids = set()
    for p in ("F:/TanitAD-scratch/actrsp/lonplan/score/A0_s0/windows.npz", "F:/TanitAD-scratch/actrsp/lonplan/score/A0_s1/windows.npz",
              "F:/TanitAD-scratch/actrsp/lonplan/sel_s0/selection_windows.npz", "F:/TanitAD-scratch/actrsp/lonplan/sel_s1/selection_windows.npz"):
        eids |= set(_np.load(p, allow_pickle=False)["eid"].tolist())
    out = select(mt, me, c141, eids)
    out["window_eid_ids_read"] = len(eids)
    json.dump(out, open(ROOT + "/clips6.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print("n_new", len(out["clips"]), "eligible", out["n_eligible"], "sha256", out["clips_sha256"], "GB", round(out["total_bytes"] / 1e9, 2))


if __name__ == "__main__":
    main()
