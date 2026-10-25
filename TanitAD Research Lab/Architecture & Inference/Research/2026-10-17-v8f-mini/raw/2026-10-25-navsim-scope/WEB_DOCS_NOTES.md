# NAVSIM GitHub docs read on 2026-10-25 (WebFetch summariser, so PUBLISHED-SECONDARY)

Sources: https://github.com/autonomousvision/navsim/blob/main/docs/{install,splits,metrics,agents}.md

## splits.md (table as returned; "scenes/tokens" are not stated on that page)
| split | logs | sensors | note |
|---|---|---|---|
| trainval (OpenScene) | 14 GB | >2000 GB | |
| test (OpenScene) | 1 GB | 217 GB | |
| mini | 1 GB | 151 GB | |
| navtrain | 14 GB | 445 GB (300 GB without history) | sensor data is downloaded separately; still needs the trainval logs |
| navtest | 983 MB | 223 GB | filter on the test split |
| navhard_two_stage | 892 MB | 31 GB | synthetic frames via download_navhard_two_stage.sh |
| warmup_two_stage | 27 MB | 1.2 GB | |
| private_test_hard_two_stage | 14 MB | 11 GB | |
Usage restriction: test, navtest, navhard_two_stage, warmup_two_stage and private_test splits may not be used for training challenge submissions.

## agents.md
* agent = get_sensor_config() + compute_trajectory(AgentInput) -> Trajectory; learned agents return a dict with "trajectory" [B, T, 3] (x, y, heading).
* "The PDM score is evaluated for a horizon of 4 seconds at a frequency of 10Hz"; a TrajectorySampling config interpolates other output rates.
* AgentInput: ego pose / velocity / acceleration in local coordinates, a discrete driving command (left, straight, right, plus an "unknown" class used to filter training data), 8 cameras + merged lidar; history at 2 Hz, "4 frames" ("2 seconds" in the page's words; the paper says 1.5 s).
* No speed-limit field is listed.

## metrics.md
* PDMS: multipliers NC, DAC; weighted TTC, EP (and C in v1).
* EPDMS: multipliers NC, DAC, DDC, TLC; weights TTC 5, EP 5, HC 2, LK 2, EC 2; "human also fails" filter.
* Two-stage: stage 1 original scene (4 s), stage 2 pre-computed follow-up scenes weighted by a Gaussian kernel on the stage-1 endpoint; final = product of the two.

## install.md
* Downloads: maps, mini, trainval, test, warmup_two_stage, navhard_two_stage, private_test_hard_two_stage; download_navtrain fetches a small portion of trainval for navtrain.
