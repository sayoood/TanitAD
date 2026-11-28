#!/usr/bin/env python
"""make_pres_fit.py -- card PREREG_V8F_PROOF_ADD118 (PRESHEAD-T0), section 2: makes the fit tool of the presence head, pres_fit.py, from crop_fit.py of card ADD99
(used unchanged, sha256 begins 61bf2f4e8e7ec61f) by the substitutions listed in SUBS and nothing else, and writes the record of control p0.

  python make_pres_fit.py --source crop_fit.py --out pres_fit.py --record PRES_FIT_GEN_RECORD.json

Every substitution has to match the source exactly once, else nothing is written. The record holds the hashes of the source and of the variant, the list of
substitutions with their counts, and the numbers of the source lines that differ. An output file that exists is refused.
"""
import argparse
import difflib
import hashlib
import io
import json
import sys

SOURCE_SHA16 = "61bf2f4e8e7ec61f"
SUBS = (
    ("card name",
     'CARD = "PREREG_V8F_PROOF_ADD99"',
     'CARD = "PREREG_V8F_PROOF_ADD118"'),
    ("8 epochs in the place of 40",
     '"batch": 256, "epochs": 40, "warmup_epochs": 1,',
     '"batch": 256, "epochs": 8, "warmup_epochs": 1,'),
    ("label = presence (1 for a lead with a finite gap from 0 to 60 m, else 0) on every frame of the clip",
     "        out[c] = (gap[off[i]:off[i] + n[i]].copy(), ok[off[i]:off[i] + n[i]].copy())",
     "        out[c] = (ok[off[i]:off[i] + n[i]].astype(np.float64), np.ones(int(n[i]), bool))"),
    ("the head's output is one logit",
     "            return GAP_SCALE_M * torch.sigmoid(self.mlp(z)[:, 0])",
     "            return self.mlp(z)[:, 0]"),
    ("binary cross-entropy on the logit, weight 1, no class weight",
     '    loss_fn = torch.nn.SmoothL1Loss(beta=BETA_M, reduction="none")',
     '    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")'),
    ("fits are written under the card's own root",
     '    out_dir = os.path.join(root, "fits", fit_name(arm, head, seed, a.permute_labels))',
     '    out_dir = os.path.join(a.out_root, "fits", fit_name(arm, head, seed, a.permute_labels))'),
    ("the option of the card's own root",
     '    ap.add_argument("--root", default=DEFAULT_ROOT)\n',
     '    ap.add_argument("--root", default=DEFAULT_ROOT)\n    ap.add_argument("--out-root", required=True)\n'),
    ("the record names the label and the count of frames with a lead",
     '"n_samples": {"fit": int(N), "sel": int(len(sy))}, ',
     '"n_samples": {"fit": int(N), "sel": int(len(sy))}, "label": "presence", "n_positive": {"fit": int((fy > 0.5).sum()), "sel": int((sy > 0.5).sum())}, '),
    ("the last log line names the presence fit",
     '    log("ZZ CROP-FIT %s kept epoch %d sel %.4f"',
     '    log("ZZ PRES-FIT %s kept epoch %d sel %.4f"'),
)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def make(src_text):
    """-> (variant text, list of {what, count}); raises SystemExit when a substitution does not match exactly once"""
    out, rec = src_text, []
    for what, old, new in SUBS:
        n = src_text.count(old)
        if n != 1 or out.count(old) != 1:
            raise SystemExit("REFUSED: substitution %r matches the source %d times (has to be 1)" % (what, n))
        out = out.replace(old, new)
        rec.append({"what": what, "count": n})
    return out, rec


def changed_lines(a, b):
    """numbers (from 1) of the lines of a that the variant b replaces, or after which it inserts"""
    sm = difflib.SequenceMatcher(None, a.split("\n"), b.split("\n"), autojunk=False)
    out = []
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        out += list(range(i1 + 1, i2 + 1)) if i2 > i1 else [i1]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", required=True)
    a = ap.parse_args(argv)
    with io.open(a.source, "rb") as fh:
        raw = fh.read()
    src = raw.replace(b"\r\n", b"\n")
    sha = sha_bytes(src)
    if not sha.startswith(SOURCE_SHA16):
        raise SystemExit("REFUSED: the source has sha256 %s, the card names %s" % (sha[:16], SOURCE_SHA16))
    text = src.decode("utf-8")
    var, rec = make(text)
    vb = var.encode("utf-8")
    for p in (a.out, a.record):
        try:
            io.open(p, "rb").close()
            raise SystemExit("REFUSED: %s exists" % p)
        except IOError:
            pass
    with io.open(a.out, "wb") as fh:
        fh.write(vb)
    lines = changed_lines(text, var)
    record = {"card": "PREREG_V8F_PROOF_ADD118", "source_sha256": sha, "source_sha16_of_the_card": SOURCE_SHA16, "variant_sha256": sha_bytes(vb), "substitutions": rec,
              "n_substitutions": len(rec), "every_substitution_once": all(r["count"] == 1 for r in rec), "source_lines_changed": lines, "n_source_lines_changed": len(lines),
              "pass_p0": bool(all(r["count"] == 1 for r in rec) and len(rec) == len(SUBS) and len(lines) == len(SUBS))}
    with io.open(a.record, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, indent=1)
    print("ZZ MAKE-PRES-FIT variant %s substitutions %d lines %d pass_p0 %s" % (record["variant_sha256"][:16], len(rec), len(lines), record["pass_p0"]))
    return 0 if record["pass_p0"] else 1


if __name__ == "__main__":
    sys.exit(main())
