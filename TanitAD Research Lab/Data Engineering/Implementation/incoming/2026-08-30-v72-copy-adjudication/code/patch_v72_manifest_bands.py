"""Add the BAND COVERAGE note to V72_MANIFEST.json (MM-E15).

⛔ A NOTE, NOT A CODE CHANGE. v7.2 is released and the trainer is being wired to
it; altering band assignment now would silently change labels under a consumer
already reading them. The same judgement that left the [6,8) hole visible rather
than absorbing it into a neighbour.

⭐ THE NOTE EXISTS BECAUSE "HOW MANY MANOEUVRES ARE UNASSIGNED" IS NOT ONE
QUESTION. Three conventions gave 54, 332 and 453 here, and all three were right
about the thing they measured.

⛔⛔ v2 — EVERY NUMBER IS READ FROM `band_census.json` AND CARRIES ITS SCOPE.
v1 of this note stated train+eval figures with no scope stamp while MM-E15 stated
train-only ones. 53 vs 54 and 440 vs 453 would have sat in two documents as the
same quantity, and a reader reconciling them without checking scope manufactures
a fifth and sixth "right answer" on top of the three we already have. ⭐ THE NOTE
INSTRUCTED READERS TO NAME THEIR CONVENTION WHILE NOT NAMING ITS OWN CORPUS —
the instrument breaking its own rule, which is the class this whole investigation
has been about, one level down.

⇒ Nothing here is typed by hand. Hand-typed numbers are exactly how the scope was
lost; numbers carried from the measurement cannot lose it.

⚠️ THE OPERATIVE WORDING IS DELIBERATE AND WAS CORRECTED ONCE. I first wrote
"declared but never populated", which is true of the tokens and MISLEADING as
guidance — it would send a reader hunting for labels that were never meant to
exist. The operative layer IS supervised, continuously, by trajectory regression
(`ade_dense_m` / `fde_last_m` at T1 — verified present in the harness with a
`separated` verdict, not assumed). `operative_s` states the layer's temporal
SCOPE. The structural point survives: a structure asserting a property it does
not provide.
"""
import hashlib
import json
from pathlib import Path

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release")
MAN = REL / "v72" / "V72_MANIFEST.json"

cen = json.loads((REL / "band_census.json").read_text(encoding="utf-8"))["by_scope"]
TR, EV, BOTH = cen["train"], cen["eval"], cen["train+eval"]


def pct(n, d):
    return round(100.0 * n / d, 1)


def scoped(key):
    """Every figure as {train, eval, train+eval} — scope is never optional."""
    return {"train": TR[key], "eval": EV[key], "train+eval": BOTH[key]}


o30 = ">30 out of horizon"
n_start_outside = {s: cen[s]["starts_by_band"].get("[6,8) gap", 0)
                   + cen[s]["starts_by_band"].get(o30, 0)
                   for s in ("train", "eval", "train+eval")}

man = json.loads(MAN.read_text(encoding="utf-8"))
man.pop("band_coverage", None)          # v2 deliberately REPLACES v1

man["band_coverage"] = {
    "_doc": ("How the hierarchy's bands claim manoeuvres, and the three ways of "
             "counting the ones they do not. Added 2026-08-30 after MM-E15. "
             "DOCUMENTATION ONLY — no label byte changed; the file md5s above are "
             "asserted unaffected by the patch that wrote this."),

    "⛔ EVERY NUMBER HERE IS STAMPED train / eval / train+eval": (
        "train-only and train+eval both appear in programme documents and BOTH are "
        "correct. MM-E15 is train-only (3,464 manoeuvres, 53 unassigned); this "
        "block reports all three scopes. A count that names its convention but not "
        "its corpus is still unreadable — that mistake was made in v1 of this very "
        "note. Reproduce with code/band_coverage_census.py, which asserts "
        "train + eval == train+eval on every figure."),

    "⛔ THE OPERATOR AT t = 30.0, STATED": (
        "Every 'over-horizon' figure in this block uses STRICT `t_start_s > 30.0`. "
        "The %d manoeuvres starting at EXACTLY 30.0 (train+eval) are therefore "
        "counted as STRATEGIC here, not as over-horizon. ⚠️ `>=` is equally "
        "defensible and gives a number larger by exactly that many — MM-E15 quoted "
        "264 and 256 for the same train-only quantity ten minutes apart for exactly "
        "this reason, and the 8 separating them is the t == 30.0 count. ⇒ ANY "
        "over-horizon count must state its operator as well as its convention and "
        "its corpus. Three things, not one."
        % BOTH["t_start_exactly_30_0"]),

    "bands_s": {"operative": [0.0, 2.0], "tactical": [2.0, 6.0],
                "gap": [6.0, 8.0], "strategic": [8.0, 30.0]},

    "census": {
        "_reproduce": "code/band_coverage_census.py -> band_census.json",
        "n_records": scoped("n_records"),
        "n_manoeuvres": scoped("n_manoeuvres"),
        "starts_by_band": {"train": TR["starts_by_band"], "eval": EV["starts_by_band"],
                           "train+eval": BOTH["starts_by_band"]},
        "overlap_by_band": {"train": TR["overlap_by_band"], "eval": EV["overlap_by_band"],
                            "train+eval": BOTH["overlap_by_band"]},
        "unassigned_manoeuvres_entries": scoped("unassigned_manoeuvres_entries"),
        "overlap_operative_only": scoped("overlap_operative_only"),
        "t_start_exactly_30_0": scoped("t_start_exactly_30_0"),
    },

    "⭐ COUNTING CONVENTIONS — NAME YOURS, AND NAME YOUR CORPUS": {
        "_why": ("'How many manoeuvres have no layer' has three right answers. Two "
                 "agents got 53 and 440 and each briefly thought the other had found "
                 "a defect; neither had — the counts differed by CONVENTION and by "
                 "CORPUS SCOPE at the same time."),
        "A_reported_by_the_record": {
            "n": scoped("unassigned_manoeuvres_entries"),
            "means": ("`bands.unassigned_manoeuvres`: FRAGMENTS falling in [6,8) that "
                      "reach MIN_TURN_DEG in NEITHER neighbouring band. Manoeuvres are "
                      "split at band edges, so this counts fragments, not manoeuvres.")},
        "B_no_overlap_with_any_assignable_band": {
            "n": {s: cen[s]["overlap_by_band"].get("no assignable band", 0)
                  for s in ("train", "eval", "train+eval")},
            "means": ("whole manoeuvres intersecting none of tactical/gap/strategic. "
                      "This is the condition the emitter applies, so it answers "
                      "'what can never be assigned'.")},
        "C_start_time_outside_every_band": {
            "n": n_start_outside,
            "operator": "t_start_s in [6,8)  OR  t_start_s > 30.0  (STRICT)",
            "means": ("whole manoeuvres whose t_start_s falls in [6,8) or beyond 30 s. "
                      "Simplest to compute and easiest to misread: a manoeuvre "
                      "STARTING outside a band may still OVERLAP one. ⚠️ Under `>=` "
                      "this figure grows by the t == 30.0 count.")},
    },

    "⚠️ THREE THINGS A PER-LAYER SUPERVISION COUNT WILL GET WRONG": {
        "1_beyond_30s_is_out_of_horizon_BY_DESIGN": {
            "n_starts": {s: cen[s]["starts_by_band"].get(o30, 0)
                         for s in ("train", "eval", "train+eval")},
            "pct_of_manoeuvres": {s: pct(cen[s]["starts_by_band"].get(o30, 0),
                                         cen[s]["n_manoeuvres"])
                                  for s in ("train", "eval", "train+eval")},
            "why": ("These are correctly NOT in `unassigned_manoeuvres`. They exist at "
                    "all because `emit_one` loads `RAW_T0_S + LOOKAHEAD_S + 5.0` = 35 s: "
                    "the +5 s margin is what lets a manoeuvre STRADDLING 30 s be clipped "
                    "at the edge — `dyaw_between(a, 30.0)` would otherwise run off the "
                    "track and the boundary would be ill-defined. ⇒ starts inside that "
                    "margin are a BY-PRODUCT of defining the edge, not content the label "
                    "claims. [6,8) is a hole INSIDE the horizon that no layer owns; "
                    ">30 s is OUTSIDE it. Only the first is a gap in the hierarchy."),
            "⚠️ eval_is_proportionally_worse": (
                "%.1f%% of eval manoeuvres start over-horizon against %.1f%% of train. "
                "An eval-side supervision count will trip on this harder than a train-"
                "side one." % (pct(EV["starts_by_band"].get(o30, 0), EV["n_manoeuvres"]),
                               pct(TR["starts_by_band"].get(o30, 0), TR["n_manoeuvres"]))),
        },
        "2_operative_s_is_SCOPE_not_token_coverage": {
            "starts_in_0_2": {s: cen[s]["starts_by_band"].get("[0,2) operative", 0)
                              for s in ("train", "eval", "train+eval")},
            "overlap_operative_only": scoped("overlap_operative_only"),
            "why": ("⛔ DO NOT read `operative_s` as a promise of token labels. No "
                    "manoeuvre is assigned to it — `split_by_band()` returns only "
                    "tactical/gap/strategic — and no `a_op`/`g_op` field exists in the "
                    "schema. THAT IS CORRECT, NOT A GAP: the operative layer is "
                    "supervised CONTINUOUSLY by trajectory regression (`ade_dense_m` / "
                    "`fde_last_m`, scored at T1), and a layer whose output is a "
                    "trajectory has nothing to hang a discrete token on. `operative_s` "
                    "states the layer's temporal SCOPE. ⚠️ But a count taken from "
                    "`bands` will read operative as ZERO supervision. It is not; it is "
                    "DIFFERENTLY supervised."),
        },
        "3_t_start_exactly_30_0_is_claimed_by_nothing": {
            "n": scoped("t_start_exactly_30_0"),
            "why": ("A start-time census calls these strategic (t <= 30); band overlap "
                    "is strict (min(end,30) > max(start,8)), so a zero-width "
                    "intersection at the closing edge is empty and the emitter assigns "
                    "them nowhere. The two conventions disagree ONLY here."),
        },
    },

    "⛔ READINGS ALREADY TRIED AND WITHDRAWN — DO NOT REDISCOVER THEM": {
        "_why": ("Removing a wrong reading is not enough. Each of these is TRUE on "
                 "its face, which is why it will be reached for again by the next "
                 "person who rediscovers the same fact. A record saying 'this was "
                 "tried, here is why it misleads' is what stops the loop — the same "
                 "reason the superseded blobs are named DO-NOT-USE rather than "
                 "deleted: 'this one is dead, use that one' only becomes more true."),
        "'operative_s is declared but never populated'": (
            "TRUE OF THE TOKENS — no manoeuvre is assigned and no `a_op`/`g_op` "
            "exists. WITHDRAWN because it reads as 'labels are missing' and sends "
            "the reader hunting for labels that were never meant to exist. The layer "
            "is supervised continuously by trajectory regression. Say SCOPE, not "
            "coverage."),
        "'the [6,8) accounting disagrees ~8x with the band arithmetic'": (
            "WITHDRAWN — 53 fragments vs 440 whole manoeuvres are DIFFERENT "
            "QUANTITIES, not a disagreement. Manoeuvres are split at band edges and "
            "`unassigned_manoeuvres` records the straddling FRAGMENT that lands in "
            "the hole. See the counting conventions above."),
        "'the pre-fix v7.2 blobs are unparseable JSON'": (
            "WITHDRAWN — they parse cleanly (4,572 / 147 records). What refused them "
            "was the LOADER'S SCHEMA CHECK (`EXPECTED_SCHEMA = 's2-geom-v7'`). A "
            "parse failure and a schema refusal point at different fixes."),
    },

    "status": ("DOCUMENTED, NOT FIXED — deliberately. Whether the operative layer "
               "should own manoeuvres, and whether [6,8) should stay a hole, are "
               "HIERARCHY questions for the PI. The emitter must not answer them by "
               "silently reassigning labels under a live consumer."),
}

MAN.write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")

# read back and re-assert ON THE WRITTEN FILE
back = json.loads(MAN.read_text(encoding="utf-8"))
bc = back["band_coverage"]
assert bc["census"]["unassigned_manoeuvres_entries"] == {"train": 53, "eval": 1,
                                                        "train+eval": 54}
assert back["files"]["train"]["md5"] == "0ff902130ce76886b8a925eceed9e3a5", \
    "the label md5s must be untouched — this is a documentation patch"
assert back["files"]["eval"]["md5"] == "aa12c948f062181c3297265b51526ec5"
# ⭐ every figure block must carry all three scopes, or the note breaks its own rule
for k, v in bc["census"].items():
    if isinstance(v, dict):
        assert set(v) == {"train", "eval", "train+eval"}, "unscoped figure: %s" % k
print("patched %s  (%d B)" % (MAN.name, MAN.stat().st_size))
print("  every census figure carries train / eval / train+eval (asserted)")
print("  label md5s UNCHANGED (asserted): train 0ff90213..  eval aa12c948..")
print("  manifest sha256 = %s" % hashlib.sha256(MAN.read_bytes()).hexdigest()[:16])
