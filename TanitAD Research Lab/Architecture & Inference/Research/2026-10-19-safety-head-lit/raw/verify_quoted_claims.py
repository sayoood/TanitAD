#!/usr/bin/env python3
"""Verify every prose claim / quoted phrase that the report cites, against the banked PDF text.

Each claim is (id, arxiv key, phrase(s)).  Pages are NORMALISED (case folded, whitespace collapsed, hyphen-newline joined,
ligatures/odd dashes unified) and the phrase must occur on a page; the script reports every PDF page (1-based) where it does.
A claim with no hit is FAIL and must not be quoted.
Usage: python verify_quoted_claims.py <library papers dir> <out json>
"""
import json, re, sys, os, glob
import fitz

CLAIMS = [
    ("toad-generalizing-scorer", "2606.07170", ["they only appear with a scorer that remains accurate off", "the proposal distribution"]),
    ("toad-fixed-vocab-no-supervision", "2606.07170", ["so they receive no supervision away from that set"]),
    ("toad-drivor-free-decoded", "2606.07170", ["forcing it to score freely decoded trajectories rather than a fixed grid"]),
    ("toad-trust-region", "2606.07170", ["the learned scorer of sec. 3.1 is more reliable nearby", "rates highly but cannot truly judge"]),
    ("toad-build-scorer-in-planner", "2606.07170", ["build well-generalizing scorers directly into the planner"]),
    ("clover-enrichment-s095", "2605.15120", ["among proposals with predicted score", "0.9753", "96.69%", "69.74%"]),
    ("clover-pairwise", "2605.15120", ["94.72% pairwise ranking accuracy"]),
    ("clover-full-score-share", "2605.15120", ["35.42% full-score proposals"]),
    ("clover-bank-families", "2605.15120", ["acceleration profiles", "approach-brake behaviors", "stop-go behaviors", "off-road boundary cases", "total 261"]),
    ("clover-180", "2605.15120", ["at most 180 candidates per scene are sent to the full evaluator"]),
    ("clover-prefilter", "2605.15120", ["we prioritize feasible and near-feasible candidates for full pdm scoring", "retaining a small subset of infeasible candidates to preserve score coverage"]),
    ("w4s-quota", "2609.36438", ["10.7% of scenes lack enough trajectories in the required safety categories to fill the quota"]),
    ("clover-backbone-gap", "2605.15120", ["train-test gap of wan2.2-5b"]),
    ("versatile-abstract", "2609.01799", ["laterally toward the drivable boundary and longitudinally toward a leading", "supervision near decision boundaries"]),
    ("versatile-either-alone", "2609.01799", ["either design alone already improves on the plain scorer"]),
    ("versatile-lat-dac-lon-nc", "2609.01799", ["the lateral design alone already recovers the full dac gain", "longitudinal design alone does the same for nc"]),
    ("safedrive-nine-points", "2602.18887", ["nine key points of each future ego box", "multiplying all sampled probabilities across timesteps and sampled points"]),
    ("safedrive-pwnc-product", "2602.18887", ["multiplying across all n surrounding instances and timesteps"]),
    ("safedrive-traj-deform", "2602.18887", ["trajectory-guided deformable attention that samples fbev along each anchor trajectory"]),
    ("flowr2a-ttc-binary", "2606.24231", ["the official ttc subscore in navsim is binary and gives no gradient near the decision boundary", "tbound = 2 seconds"]),
    ("flowr2a-aux-heads", "2606.24231", ["increase the resolution of the selector's awareness"]),
    ("flowr2a-grid-sample", "2606.24231", ["add a grid-sample-based cross-attention module that attends to the bev features"]),
    ("flowr2a-knob", "2606.24231", ["free inference-time knob to balance ep and ttc"]),
    ("ztrs-7-features", "2510.24108", ["40 timesteps and 7 per-step", "motion-difference features"]),
    ("hydra-weights", "2406.06978", ["typically fall within the following ranges", "indicating the necessity to prioritize rule-based costs over imitation"]),
    ("hydra-pdm-score-distil", "2406.06978", ["the irregular distribution of the pdm score incurs performance degradation"]),
    ("hydra++-aux-perception", "2503.12820", ["auxiliary training of perception worsens the planning performance"]),
    ("hydra++-dac-neural", "2503.12820", ["neural planners score low on drivable area compliance"]),
    ("drivesuprim-easy-neg", "2506.06659", ["easy-to-reject options dominate the training process and gradient"]),
    ("drivesuprim-self-distill-soft", "2506.06659", ["teacher-generated soft labels"]),
    ("gtrs-superdense", "2506.06664", ["super-dense trajectory vocabulary", "16,384"]),
    ("gtrs-vocab-dropout", "2506.06664", ["vocabulary dropout"]),
    ("w4s-bank", "2609.36438", ["scene-matched bank of perturbed trajectories with precomputed", "following clover"]),
    ("w4s-avg-bank", "2609.36438", ["each scene’s bank holds 54.5 valid entries on average"]),
    ("w4s-permute", "2609.36438", ["shuffling bank outcome labels reduces ordering accuracy from", "64.2% to 55.6%"]),
    ("w4s-realised-future", "2609.36438", ["raises pdms from 93.01 to 93.96", "current-frame target gives 92.70"]),
    ("clover-refit-evaluator", "2605.15120", ["repeatedly refit the scorer to evaluator-provided sub-scores"]),
    ("hydra-pp-weaker", "2406.06978", ["non-differentiable post-processing yields fewer performance gains than our framework"]),
    ("dawam-hardneg", "2608.19085", ["safety-critical hard negatives further sharpen the scorer’s discrimination near planning boundaries"]),
    ("dawam-shared-global", "2608.19085", ["reduces ep from 91.36 to 88.68"]),
    ("sparsedrive-scaling", "2603.29163", ["performance consistently improves as trajectory anchors become denser"]),
    ("sparsedrive-dfa", "2603.29163", ["mha denotes multi-head cross-attention and dfa denotes deformable aggregation"]),
    ("drivor-subscores", "2601.05083", ["predicting multiple sub-scores not only enables behavior control but also improves accuracy"]),
    ("drivor-camera-by-traj", "2601.05083", ["scoring draws on left- or rear-camera features depending on trajectory sharpness or collision risk"]),
    ("mmfuture-pair", "2609.20377", ["pairing improves nc, dac, and ttc by 0.5, 0.1, and 1.2 points"]),
    ("wote-3-rows", "2504.01941", ["trajectory evaluation using imitation rewards performs better in nc and ttc", "simulation rewards excels in dac and ep"]),
    ("ipad-proposal-aware", "2505.15111", ["proposal-aware spatial cross-", "attention mechanism"]),
    ("flowr2a-stage1-vocab-gap", "2606.24231", ["the stage-1 selector sees only vocabulary trajectories, which differ from decoder proposals at inference"]),
    ("flowr2a-stage2-calibrated", "2606.24231", ["we mix in random vocabulary trajectories to keep the selector calibrated across the full quality range"]),
    ("flowr2a-stage2-32-invdens", "2606.24231", ["32 random vocabulary trajectories drawn under inverse-density weighting"]),
    ("flowr2a-selector-bottleneck", "2606.24231", ["the selector is the more actionable bottleneck"]),
    ("flowr2a-tab4a-generator", "2606.24231", ["ablation studies on reward conditioning. evaluated on single proposal"]),
]


def norm(t, join=True):
    t = t.replace("­", "").replace("-\n", "" if join else "-")
    t = t.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff").replace("’", "'").replace("’", "'")
    t = re.sub(r"\s+", " ", t).lower()
    return t


def main(lib, out):
    res = []
    cache = {}
    for cid, key, phrases in CLAIMS:
        f = glob.glob(os.path.join(lib, key + "_*.pdf"))[0]
        if f not in cache:
            doc = fitz.open(f)
            raw = [p.get_text("text") for p in doc]
            cache[f] = [(norm(t, True), norm(t, False)) for t in raw]
        pages = cache[f]
        hits = {}
        for ph in phrases:
            p2 = norm(ph)
            hits[ph] = [i + 1 for i, (ta, tb) in enumerate(pages) if p2 in ta or p2 in tb]
        ok = all(hits[ph] for ph in phrases)
        res.append(dict(id=cid, arxiv=key, phrases=phrases, pages=hits, status="PASS" if ok else "FAIL"))
    json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
    print(sum(r["status"] == "PASS" for r in res), "/", len(res), "claims PASS")
    for r in res:
        print(r["status"], r["id"], {k[:40]: v for k, v in r["pages"].items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
