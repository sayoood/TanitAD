# VERDICT_PLANLEAD (note PREREG_V8F_PROOF_ADD39_N1)

Tier: T0 (first tick of each window, logged state, logged frames; a diagnostic, never driving performance). STRATEGIC: PAUSED (PI item 25). LONGITUDINAL: the plan's along-track error against the logged driver; LATERAL, TACTICAL: not measured by this read.

## Controls (first)

* d1_permuted: PASS {"GL": {"per_seed": [0.0012478391916118435, -0.021509705545402522], "pooled": -0.010130933176895339, "ci": [-0.02936514518945691, 0.011678440300725986]}, "GV": {"per_seed": [-0.004115799581645467, -0.005683435395178301], "pooled": -0.004899617488411884, "ci": [-0.014872308399787332, 0.005286212464047671]}}
* d2_planted: PASS {"GL": {"per_seed": [0.424132462866071, 0.43442615864929157], "pooled": 0.4292793107576813, "ci": [0.3478068581139576, 0.48461910765902394]}}
* d3_counts: PASS {"n_lead": [642, 642], "n_used": [642, 642], "n_ep": [56, 56], "expected": [642, 56], "ticks_match_episode_dumps": [true, true]}
* d4_frame_alignment_as_written: FAIL {"corr": [{"-5": 0.7948844042697072, "0": 0.8265883013992315, "5": 0.8292250991617348}, {"-5": 0.8239705312031803, "0": 0.8599081226243643, "5": 0.8698268203433854}]}
* d4b_note_N2: PASS {"k": [0, 2, 4, 5, 6, 8, 10], "corr": [[0.8265883013992315, 0.818676694694128, 0.8072858758587294, 0.7997080528624402, 0.7910137666730805, 0.7714457370307505, 0.7487776938882073], [0.8599081226243643, 0.8523185657319594, 0.8399440585814505, 0.8325455377660107, 0.8235973187585006, 0.8029372606469107, 0.7794568707744324]], "n": [601, 601]}
* d5_one_frame: PASS {"share_within_0.5_mps": [1.0, 1.0]}

## Name

PLAN-ERROR-NOT-LEAD

## Target at 2.0s (plan minus logged along-track position, m)

* windows [642, 642], episodes [56, 56]; mean of y ['0.204', '0.297'] m, spread ['0.723', '0.696'] m; R2 of the ego-speed base ['0.056', '0.042']
* GL (label lead added to the base): per seed ['+0.0045', '+0.0219'], pooled +0.0132 [-0.0079, +0.0343]
* GV (the lead head's camera read added to the base): per seed ['-0.0114', '-0.0022'], pooled -0.0068 [-0.0234, +0.0089]

## Target at 4.0s (plan minus logged along-track position, m)

* windows [642, 642], episodes [56, 56]; mean of y ['-1.271', '-0.553'] m, spread ['3.360', '2.966'] m; R2 of the ego-speed base ['0.082', '-0.007']
* GL (label lead added to the base): per seed ['-0.0119', '+0.0259'], pooled +0.0070 [-0.0340, +0.0440]
* GV (the lead head's camera read added to the base): per seed ['+0.0012', '+0.0131'], pooled +0.0071 [-0.0234, +0.0360]

Limits: linear fits on a few inputs; the first tick only; two seeds, one corpus.
