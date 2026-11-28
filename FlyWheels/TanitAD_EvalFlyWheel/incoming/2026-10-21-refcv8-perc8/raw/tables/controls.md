| control | known value | read | verdict |
|---|---|---|---|
| GT as the box prediction (programme `summarise`, 300 windows, 947 positives) | every defined AP cell = 1.0 | 156/156 cells = 1.0 | PASS |
| GT as the map prediction (structured 1000x600 grid) | pooled IoU = 1.0 in every (class, band) with GT | 30/30 cells = 1.0 | PASS |
| all-one-class map prediction (negative control) | IoU of every other class = 0 | max 0.0 | PASS |
| re-score refcv7's banked dump with THIS instrument (AP@2 m pooled) | banked 0.24906 (within 1e-3) | 0.249064; max abs diff over 23 AP/AUROC cells = 1.0e-12 | PASS |
| re-summarise refcv7's banked map WindowTable | banked IoU cells | 40 IoU cells, max abs diff 0.0 | PASS |
| weighted-AP kernel == `detection_metrics._ap` (4 thr x 2 arms, 1e-9) | equal | see `raw/box_read_declared.json` controls | PASS |
| bootstrap draw stream == `ci.paired_episode_cluster_bootstrap` (recall paired CI) | equal | see `raw/box_read_declared.json` controls | PASS |
| weighted AUROC == `detection_metrics.auroc` (1e-9) | equal | see `raw/box_read_declared.json` controls | PASS |
| LAW-frame control (perc pass, 2 windows: zero LAW frame == real LAW frame for every perception tensor) | equal | all equal, law term differs | PASS |
| checkpoint md5 | b4364029b68f29a33eeb4aebbbbd002b (battery record `E:/refcv8_eval_kit/ckpt/ckpt_48852.md5`) | b4364029b68f29a33eeb4aebbbbd002b | PASS |