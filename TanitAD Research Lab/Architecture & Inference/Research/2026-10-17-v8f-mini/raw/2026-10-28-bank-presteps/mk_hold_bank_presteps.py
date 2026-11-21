"""Hold: the DAgger-bank zero-training pre-steps (H-ST-REGRET pre-step, H-ST-RECOVERYLABEL step 1, H-DD-FITGAP on M2b s0 / s1;
cards frozen before data), banked as raw/2026-10-28-bank-presteps/, plus three register rows appended to GOALS_AND_CLAIMS.md.
Derived tables over 3 MB, re-extracted eval dumps, copies of tip source and caches stay on E: and are listed with sha256 in
NOT_LANDED.txt.  python mk_hold_bank_presteps.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
H = "C:/Users/Admin/qland/hold_bank_presteps/"
S = "E:/Projects/TanitAD-artifacts/dagland/bank_presteps/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-bank-presteps/"
assert not os.path.exists(H), H
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")
M = "\u2212"


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(rel, b):
    assert len(b) > 0 and len(rel) <= 220, (rel, len(rel))
    assert not re.search(rb"hf_[A-Za-z0-9]{8,}", b), rel
    if rel.endswith((".py", ".sh", ".md")):
        assert b"\r" not in b, rel
    assert len(b) < 3_000_000 or rel.startswith("mods"), rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


# the agent's own checksum list: every listed file must still carry its sha
pins = {}
for line in open(S + "SHA256SUMS.txt", encoding="utf-8"):
    h, f = line.split(None, 1)
    pins[f.strip().lstrip("*")] = h


def excluded(rel):
    if "__pycache__" in rel or rel.startswith("raw/eval_dumps/"):
        return "re-extractable from v8f-mini-hacc/t1kin_hacc_dumps_f64_2026-10-26.tar (E:), 282 small files" if "eval_dumps" in rel else "cache"
    if rel in ("raw/bank_table_s0.npz", "raw/bank_table_s1.npz", "raw/logged_cache.npz"):
        return "derived table, rebuilt by scripts/build_tables.py / build_logged_cache.py from the DAgger banks (shard shas in their PROVENANCE)"
    if rel == "raw/real_panel_v2_unused.npz":
        return "unused by-product of the post-hoc V2 runner"
    if rel.startswith("src_tip/") and rel.endswith(".py"):
        return "copy of a tip file; src_tip/MANIFEST.json and the tip commit identify it"
    return None


n, notl = 0, []
for root, dirs, fs in os.walk(S):
    for f in fs:
        full = os.path.join(root, f).replace("\\", "/")
        rel = full[len(S):]
        b = open(full, "rb").read()
        if rel in pins:
            assert sha(b) == pins[rel], rel
        why = excluded(rel)
        if why:
            notl.append(f"{rel} | {len(b)} | {sha(b)} | {why}")
            continue
        if not b:
            continue
        if rel.endswith((".py", ".sh", ".md")) and b"\r" in b:
            b = b.replace(b"\r\n", b"\n")
            assert b"\r" not in b, rel
        assert blob(PK + rel) is None, rel
        put("stage/" + PK + rel, b)
        n += 1
hdr = ("# NOT_LANDED: files of this package that stay on the dev box (E:/Projects/TanitAD-artifacts/dagland/bank_presteps/)\n"
       "# path | bytes | sha256 | why\n")
put("stage/" + PK + "NOT_LANDED.txt", (hdr + "\n".join(sorted(notl)) + "\n").encode("utf-8"))
put("stage/" + PK + "mk_hold_bank_presteps.py", open(__file__.replace("\\", "/"), "rb").read().replace(b"\r\n", b"\n"))
n += 2

SRC = "`" + PK + "MANIFEST.md`"
FZ = ("cards frozen at 22:05 before data (FROZEN_CARDS.md sha256 3202a62f\u2026); Amendment 1 (labeller construction, FITGAP statistic) at"
      " 22:17, before the first bank read at 22:21")
ROWS = [
    ("**H-ST-REGRET-PRE** (2026-10-28)",
     "Zero-training pre-step of H-ST-REGRET (outcome-weighted DAgger draws): crawl states (own speed below half the logged speed at the"
     " matched route point, logged speed at least 3 m/s) are rare enough in the DAgger bank that deficit-weighted draws concentrate the"
     " p = 0.1 own-state budget on them by at least 5x (card branch: share < 3 %); share >= 15 % kills the card.",
     "**MEASURED 2026-10-28, T1-kin rollout states on TRAIN (DAgger banks of M2b s0 / s1), clip-cluster bootstrap B = 2,000 over 4,512"
     " clips (another draw of clips only); " + FZ + ".** Pooled crawl share of usable records: s0 5.752 % [5.439, 6.062], s1 2.608 %"
     " [2.443, 2.782]; tick 2 carries it (9.575 % / 5.071 %), tick 1 3.147 % / 0.937 %. 50 : 50 deficit-weighted multiplier 2.68 [2.60,"
     " 2.77] (s0) and 3.08 [2.99, 3.18] (s1), below the 5x branch; ceiling 9.2x / 19.7x. Expected crawl samples in the 14,688 / 14,839"
     " own-state draws: 845 [799, 890] / 387 [363, 413]. Reading: MIDDLE BAND, the seeds straddle the 3 % line; the card is neither killed"
     " nor confirmed, and weighting buys about 3x.",
     "MEASURED (middle band; multiplier 2.7x / 3.1x)"),
    ("**H-ST-RECOVERYLABEL-1** (2026-10-28)",
     "Step 1 of H-ST-RECOVERYLABEL: a kinematically feasible recovery label onto the human path (geometric blend at a lookahead, human"
     " speed profile, envelope-clipped) can be given to the off-route own states the DAgger bank drops, lifting usable tick-2 states from"
     " 53.9 % to 75-81 % (frozen band), with zero feasibility violations and an exact identity on on-route states.",
     "**MEASURED 2026-10-28, census of both full banks (306,566 windows per tick), labels offline; " + FZ + "; a post-hoc V2 (Amendment 2,"
     " 22:27, after a 3,000-record smoke of V1) is reported beside, never in the verdict.** Gates PASS: identity bit-exact on 1,000 / 1,000"
     " real windows; the offset-ignoring regression arm fails the gate on 304 / 305 real off-route records and the no-clip arm on 257 / 257;"
     " 0 violations among 18,707 (s0) and 24,436 (s1) emitted labels; 27 tests pass. Kept fraction at tick 2: 53.85 % to 59.51 % (s0),"
     " 54.69 % to 61.95 % (s1): the 75-81 % prediction is NOT met (V2: 58.20 % / 60.07 %). Of s0's 74,368 tick-2 off-route records 25.8 %"
     " have no logged future to label (the card's 78 % counted them) and 47.8 % are refused as infeasible, mostly below 3 m/s own speed or"
     " above 60 deg heading error. Envelope constants from `feasible_decode.py` (A_MAX 4.0, KAPPA_MAX 0.2); P10Prior holds no such limit."
     " Named next, not run: a curvature-bounded (Dubins / clothoid) recovery and loss-masked partial-horizon labels.",
     "MEASURED (labeller gates PASS; kept-fraction prediction not met: 59.5 % / 62.0 %)"),
    ("**H-DD-FITGAP-M2B** (2026-10-28)",
     "H-DD-FITGAP on the two checkpoints the DAgger banks hold (M2b s0 / s1, 2 of the card's 6): the launch-speed deficit on the model's own"
     " TRAIN GT-launch windows is at least half its size on eval-141 (r >= 0.5: FIT-LIMITED, so no data-amount or data-selection lever can"
     " clear a crawl bar); r <= 0.25 would be a generalisation gap.",
     "**MEASURED 2026-10-28, TRAIN side = M2b's own T1-kin rollout states (49,618 GT-launch windows / 1,891 clips), eval side = the banked"
     " T1-kin dump of plain M2b on the 327 windows / 49 episodes (reference only); cluster bootstrap B = 2,000; " + FZ + "; the eval-side"
     " statistic reproduces the M-LAUNCH instrument to < 1e-9.** C_train " + M + "1.2540 [" + M + "1.3033, " + M + "1.2042] / " + M
     + "0.9105 [" + M + "0.9607, " + M + "0.8591] m/s, C_eval " + M + "1.0202 [" + M + "1.2791, " + M + "0.7408] / " + M + "0.5649 [" + M
     + "0.8669, " + M + "0.2104]; r = 1.229 [0.976, 1.680] (s0), 1.612 [1.023, 4.416] (s1): both FIT-LIMITED. Disclosed mismatch: the"
     " plain-M2b dumps ran `--limit logged`, the bank `--limit causal`; shifted by the DAG arm's measured logged-to-causal change (+0.299 /"
     " +0.305 m/s) r reads 0.950 / 1.047 (ESTIMATED, beside). GEN-GAP is now unreachable under the card's 5-of-6 rule; FIT-LIMITED needs 3"
     " of DAG-OACC s0 / s1 and REP s0 / s1 (about 0.17 h each on a 10 % clip subset, ESTIMATED). Reading: the model crawls on its own"
     " training windows as much as on eval, so the crawl is not a data-coverage gap.",
     "PARTIAL (2 of 6 checkpoints, both FIT-LIMITED; GEN-GAP unreachable)"),
]
GC = "Project Steering/GOALS_AND_CLAIMS.md"
g = blob(GC)
assert g.endswith(b"\n") and b"**H-ST-REGRET-PRE**" not in g and b"\r" not in g
add = "".join("| " + " | ".join((i, c, e, s, SRC)) + " |\n" for i, c, e, s in ROWS).encode("utf-8")
for w in BANNED:
    assert w not in add.lower(), w
assert add.count(b" | ") == 4 * len(ROWS) and g.rstrip(b"\n").split(b"\n")[-1].count(b" | ") == 4
put("mods/" + GC, g + add)
put("msg.txt", ("DAgger-bank zero-training pre-steps (cards frozen before data): H-ST-REGRET pre-step MIDDLE BAND (crawl share 5.75 %"
                " / 2.61 %, deficit weighting 2.7x / 3.1x); H-ST-RECOVERYLABEL step 1 gates PASS (identity bit-exact, regression arms"
                " fail, 0 violations) but the kept-fraction prediction is not met (59.5 % / 62.0 % vs 75-81 %); H-DD-FITGAP on M2b s0 /"
                " s1 FIT-LIMITED (r 1.23 / 1.61; launch-mode mismatch disclosed), GEN-GAP unreachable. Package"
                " raw/2026-10-28-bank-presteps/ (" + str(n) + " files; NOT_LANDED.txt lists " + str(len(notl)) + " files kept on E:"
                " with sha256); three register rows appended.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n").encode("utf-8"))
print("hold written", H, n, "not landed", len(notl))
