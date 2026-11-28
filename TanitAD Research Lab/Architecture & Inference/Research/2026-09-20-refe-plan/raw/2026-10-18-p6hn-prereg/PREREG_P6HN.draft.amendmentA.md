# P6-HN full-arm DRAFT, amendment A (2026-10-18): masked two-call set scoring for the hard-negative term

Applies to the DRAFT `PREREG_P6HN.draft.md` (sha256 `4a9ea2086584fdddf9c1b51f34efc0f027a3d096bdae61e2295714103aee0f93`),
which stays a draft. Written before any P6-HN or screen data exists.

**The defect.** The draft says hard negatives are "appended to that sample's scorer candidates" and that "the fan term is
unchanged bit for bit". Both cannot hold: REFe's scoring decoder self-attends across the candidate set
(`refe/model.py` `CrossBlock.self_attn`; `score_trajectories` scores "a trajectory IN ITS SET"). A plain append moves every
fan row's score and trains the head on 72-candidate sets that are never deployed, and the draft's own I3 / G-LIVE identity
would fail for a reason unrelated to the lever.

**The amendment.** The draft adopts the screen's form (`PREREG_P6HN_SCREEN.md` §2.4):
1. the fan call is unchanged (bit-identical to P6's when the bank is absent);
2. a second call scores `[fan ; hn]`, each hard negative attending to the 64 fan rows and to itself; only the hard-negative
   rows are kept and enter the term;
3. the term lives in ONE function (`refe/hn_bank.py`) imported by both trainers, so a screen answer can carry over.
The second call costs one extra scorer forward per sample; the draft's G-LIVE-MEM row must be re-estimated with it before
the draft can be registered.
