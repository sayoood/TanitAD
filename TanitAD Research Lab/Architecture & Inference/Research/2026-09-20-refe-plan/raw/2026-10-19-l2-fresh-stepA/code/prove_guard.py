#!/usr/bin/env python3
"""Proof that `analyse_fresh.py --seal-check-only` ACCEPTS the sealed amendment and REFUSES mutated copies, through the real CLI, WITHOUT reading any fresh label or score (the dry-run mode reads the prereg file, the
carve's token-list file, the head files, the reference P2 head, the fold-heads file and the pinned code files only).

Cases (each a real subprocess; the refusal must be exit 5 with a 'SEAL REFUSED (5)' message):
  ACCEPT            the sealed file with its own sha256                                   -> exit 0, ZZSEAL_ACCEPTED
  RED one-byte      the sealed file with ONE byte changed, the ORIGINAL sha256            -> exit 5 (sha256 mismatch)
  RED one-byte+sha  the same edit, sha256 recomputed (a forger who re-hashes)             -> must STILL be refused when the edit hits a pin / a marker / a code pin; a prose-only edit with a recomputed
                    sha is, by design, a different file whose hash the operator chose to pass: reported as such, not claimed as a refusal
  RED draft         "Status: SEALED" -> "Status: DRAFT", sha256 recomputed                -> exit 5 (draft marker)
  RED no-marker     the SEALED marker removed, sha256 recomputed                          -> exit 5
  RED pin           one hex digit of the final-head-s0 pin (EVERY occurrence) / of the carve pin changed, sha256 recomputed -> exit 5 (pin not named)
  RED code pin      one hex digit of the analyse_fresh.py / l2lib.py code pin changed, sha recomputed -> exit 5 (stale code pin)
"""
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
PY = sys.executable
SEALED = os.path.join(PKG, "PREREG_L2_FRESH.md")
SAMPLE = os.path.join(PKG, "raw", "fresh_sample_token_disjoint.json")


def sha_lf(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def run(prereg: str, sha: str) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([PY, "-I", os.path.join(HERE, "analyse_fresh.py"), "--seal-check-only", "--prereg", prereg, "--prereg-sha256", sha, "--sample", SAMPLE], capture_output=True, text=True, env=env, timeout=300)
    msg = (p.stdout + p.stderr).strip().splitlines()
    return {"exit": p.returncode, "last_line": msg[-1][:300] if msg else ""}


def flip_char(h: str) -> str:
    """the same 64-hex string with ONE digit changed"""
    return h[:5] + ("0" if h[5] != "0" else "1") + h[6:]


def flip_all_occurrences(text: str, hexval: str) -> str:
    """the sealed text must stop NAMING `hexval`: change one digit of EVERY occurrence (a pin may legitimately appear in several places; editing one of them leaves the text still naming it)"""
    assert hexval in text
    return text.replace(hexval, flip_char(hexval))


def flip_code_pin(text: str, name: str) -> str:
    """one digit of the code pin on the line that names `name` (the line `  * `name` <hex>`)"""
    m = re.search(r"^  \* `" + re.escape(name) + r"` ([0-9a-f]{64})$", text, re.M)
    assert m, name
    return text[:m.start(1)] + flip_char(m.group(1)) + text[m.end(1):]


RED_KEYS = ("RED_one_byte_original_sha", "RED_draft_status", "RED_no_seal_marker", "RED_head_s0_pin_changed", "RED_carve_pin_changed", "RED_code_pin_one_hex_digit", "RED_code_pin_l2lib_one_hex_digit")


def cases() -> dict:
    raw = open(SEALED, "rb").read()
    sha = sha_lf(raw)
    txt = raw.decode("utf-8")
    out = {"sealed_sha256": sha}
    pin_s0 = re.search(r"^\* final head s0 sha256: `([0-9a-f]{64})`$", txt, re.M).group(1)
    pin_carve = re.search(r"^\* carve token-list sha256 .*?`([0-9a-f]{64})`", txt, re.M).group(1)
    with tempfile.TemporaryDirectory() as td:
        def put(name, body):
            p = os.path.join(td, name)
            open(p, "wb").write(body.encode("utf-8"))
            return p, sha_lf(body.encode("utf-8"))
        out["ACCEPT"] = run(SEALED, sha)
        edited = txt[:200] + ("X" if txt[200] != "X" else "Y") + txt[201:]
        p, h = put("onebyte.md", edited)
        out["RED_one_byte_original_sha"] = run(p, sha)
        out["one_byte_recomputed_sha_prose_edit_(informational)"] = run(p, h)
        p, h = put("draft.md", txt.replace("Status: SEALED", "Status: DRAFT", 1))
        out["RED_draft_status"] = run(p, h)
        p, h = put("nomarker.md", txt.replace("Status: SEALED", "Status: -", 1).replace("Status: SEALED", "Status: -"))
        out["RED_no_seal_marker"] = run(p, h)
        p, h = put("pin_head.md", flip_all_occurrences(txt, pin_s0))
        out["RED_head_s0_pin_changed"] = run(p, h)
        p, h = put("pin_carve.md", flip_all_occurrences(txt, pin_carve))
        out["RED_carve_pin_changed"] = run(p, h)
        p, h = put("codepin.md", flip_code_pin(txt, "analyse_fresh.py"))
        out["RED_code_pin_one_hex_digit"] = run(p, h)
        p, h = put("codepin2.md", flip_code_pin(txt, "l2lib.py"))
        out["RED_code_pin_l2lib_one_hex_digit"] = run(p, h)
    out["pass"] = bool(out["ACCEPT"]["exit"] == 0 and "ZZSEAL_ACCEPTED" in out["ACCEPT"]["last_line"]
                       and all(out[k]["exit"] == 5 and "SEAL REFUSED (5)" in out[k]["last_line"] for k in RED_KEYS))
    return out


if __name__ == "__main__":
    o = cases()
    json.dump(o, open(os.path.join(PKG, "raw", "guard_proof.json"), "w"), indent=1)
    print(json.dumps(o, indent=1))
    sys.exit(0 if o["pass"] else 1)
