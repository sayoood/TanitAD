# R9PF_LH_VERDICT -- VOID

**tier:** T1 self-action OPEN LOOP on logged eval139 frames, not driving performance  
**estimator:** paired episode-cluster bootstrap over EPISODES, B 2000 seed 0 -- blind to training and inference variance; fit seeds 0/1/2 = training-variance read  
**prereg:** PREREG_R9PF_LH.md sha256 6de4e4bb0e8abfe0ada03e3e0e401faea9996857ed80744547b923c4f9f2bd59

**committed outcome:** VOID -- PREREG_R9PF_LH s2: 'R9-PF's own K-EQ / K-OFF must have PASSED in its final verdict, or this arm is VOID.' R9PF_PA_VERDICT = VOID; K-OFF PASS, K-EQ FAIL (seed 0: 28 / 515 windows over 2 ulp, seed 1: 24 / 515; decision agreement 100 %).

No R9LH-ARG / R9LH-SOFT / R9DER-ARG composition was run: the precondition is checked first, and a VOID arm is not scored.

**Unblock:** R9-PF K-EQ resolved (CPU roll discriminating CUDA-vs-CPU head arithmetic + a Master-Mind ruling on the tolerance), then rerun this prereg.
