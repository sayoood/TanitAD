r"""The banked first-dose NULL arms the second dose is read against (BAL-NULL_s0 and BAL-NULL_s1: the reference of every delta; BAL-NULL_rep: the floor).
They are reused, never retrained (AMENDMENT_3_DOSE2.md). This module re-asserts, from the bank folders alone, that each of them is whole and unchanged:
  BANK.done present; ckpt_last.pt present; the bytes of ckpt_last.pt hash to the sha256 in BANK_MANIFEST.json AND in RUN.done (evidence.ckpt_sha256) AND to the value the manifest lists for it;
  selfcheck.json reads the NULL levers (goal_pos_weight_stop 1.0, lon_class_weight_stop 1.0). A missing bank, a missing file, a changed byte or a lever other than 1.0 is a problem line.
Used by dose2_chain.sh (refuses to start without them) and by read/stopbal_read_dose2.py. CLI:  python -I null_banks.py [--pulls C:/Users/Admin/t2v8f_pulls]  -> rc 0 clean, rc 1 problems (one per line on stdout)."""
import hashlib
import json
import os
import sys

NULL_TAGS = ("BAL-NULL_s0", "BAL-NULL_s1", "BAL-NULL_rep")
DEFAULT_PULLS = "C:/Users/Admin/t2v8f_pulls"


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def bank_dir(tag, pulls=DEFAULT_PULLS):
    return os.path.join(pulls, "stopbal_" + tag)


def problems(pulls=DEFAULT_PULLS, tags=NULL_TAGS):
    """[] iff every NULL bank is whole; else one line per problem, each naming the tag and the file"""
    out = []
    for t in tags:
        d = bank_dir(t, pulls)
        if not os.path.isdir(d):
            out.append(f"{t}: the bank folder {d} is missing")
            continue
        for f in ("BANK.done", "BANK_MANIFEST.json", "RUN.done", "selfcheck.json", "ckpt_last.pt"):
            if not os.path.isfile(os.path.join(d, f)):
                out.append(f"{t}: {f} is missing in {d}")
        if any(x.startswith(t + ": ") and "is missing in" in x for x in out):
            continue
        try:
            got = _sha(os.path.join(d, "ckpt_last.pt"))
            man = _json(os.path.join(d, "BANK_MANIFEST.json"))
            run = _json(os.path.join(d, "RUN.done"))
            sc = _json(os.path.join(d, "selfcheck.json"))
        except (OSError, ValueError) as e:
            out.append(f"{t}: a bank file is unreadable ({type(e).__name__})")
            continue
        m_sha = (man.get("files", {}).get("ckpt_last.pt") or {}).get("sha256")
        r_sha = (run.get("evidence") or {}).get("ckpt_sha256")
        if man.get("tag") != t or run.get("tag") != t:
            out.append(f"{t}: the manifest / RUN.done name another tag ({man.get('tag')!r} / {run.get('tag')!r})")
        if got != m_sha:
            out.append(f"{t}: ckpt_last.pt hashes to {got[:12]}, BANK_MANIFEST.json says {str(m_sha)[:12]} (the checkpoint changed)")
        if got != r_sha:
            out.append(f"{t}: ckpt_last.pt hashes to {got[:12]}, RUN.done says {str(r_sha)[:12]} (the checkpoint changed)")
        lv = sc.get("levers") if isinstance(sc.get("levers"), dict) else {}
        for k in ("goal_pos_weight_stop", "lon_class_weight_stop"):
            v = lv.get(k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or abs(v - 1.0) > 1e-12:
                out.append(f"{t}: selfcheck levers.{k} = {v!r}, a NULL arm reads 1.0")
    return out


def checkpoint_shas(pulls=DEFAULT_PULLS, tags=NULL_TAGS):
    """{tag: ckpt sha256} of the banks as they are now (call problems() first)"""
    return {t: _sha(os.path.join(bank_dir(t, pulls), "ckpt_last.pt")) for t in tags}


if __name__ == "__main__":
    pulls = DEFAULT_PULLS
    if "--pulls" in sys.argv:
        pulls = sys.argv[sys.argv.index("--pulls") + 1]
    pr = problems(pulls)
    for x in pr:
        print("NULL-BANK-REFUSED " + x)
    if not pr:
        print("NULL-BANK-OK " + " ".join(f"{t}={s[:12]}" for t, s in checkpoint_shas(pulls).items()))
    sys.exit(1 if pr else 0)
