# Note N1 to card ADD81 (TOKCLOSE): two readings of the controls, fixed before any fit on data

Written after the tool kit was built and tested on synthetic data only and before any fit or read on the real caches. No value of the card exists. The card was landed in hold 100 (c958319e).

## 1. Control t2 and the planted fit

The card says in t2 that for every fit the held-out loss at the stopping point is below the held-out loss before the first step, and it counts nine fits (eight of the arms and the planted fit of control t1). The planted fit is trained on closing labels permuted among the labelled rows: it has nothing to learn, so its loss on the stopping set has no reason to fall and can end above its start by chance. On the synthetic world of the tool's self-test it did (0.7296 before the first step, 0.7394 at the stopping point). A control that asks the planted fit to learn would contradict the control it is planted for.

Reading fixed here: the loss clause of t2 is judged on the eight fits of the arms. For the planted fit the counts of clips and labelled frames are judged as for the others (4,080 training clips, 432 stopping clips, none of the 141), and its loss pair is written into the control record and not judged. The tool is run without its switch `--judge-planted-loss`.

## 2. Control t1 and "pooled r"

The planted arm is one fit (T10, seed 0). Its r in t1 is the r of that fit on the scored frames of H 10 of the evaluation clips (the rows of reader seed 0); there is nothing to pool. The band [-0.10, +0.10] stays.

## 3. Stated beside, no clause of the card

* The T arms read the 4,080 training clips whole from the token cache, eight clips per step with every eligible frame of those clips as a row; they do not reuse the window stream of the reader's own trainer.
* The fixed hyperparameters are those of `tokclose/TOKCLOSE_DEFS.md` section 2 (48fb198c74dca151), written before any fit: one recurrent layer of 128 units over the H embeddings, Huber loss with delta 1 m/s, Adam at 1e-3, stopping by the stopping set with the best state kept (E arms: at most 100 passes over 345 clips, patience 10; T arms: at most 8 passes over 4,080 clips, patience 2).
* The estimate table of a fit holds every frame from H-1 on; the scored frames are the subset of card ADD78.
