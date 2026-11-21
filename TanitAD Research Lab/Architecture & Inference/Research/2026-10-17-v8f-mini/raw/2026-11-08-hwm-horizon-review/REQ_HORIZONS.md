# REQ_HORIZONS: the PI's requirements for the three levels (Master Mind, 2026-11-08)

Every design, label, loss, trainer flag and eval row in the review is checked against this file.

## PI, verbatim (chat, 2026-11-08)
> "We need to implement clearly my definition of operative, tactical and strategic time horizons, those are clearly defined (op up to 2 s), (tac up to 6 sec iccl our tactical vocab), (6se to 30 sec incl. av checkpointsssss, detectedstrategic informtio nd derived goals from it) trajectory is a combo of op and tac and thus 6s . It makes no sense to consider 0.8 s as long term abstract level. So we need to fix and retrain as we need it"
>
> "The idea of estimating time is just to estimate rougly the poeriod of day and linked illumanation from pure images and past frames no additional informaio --> yes. We need to review the whole design to meet my requrirements. If necessary augment our data set by the additionl strategic information that means for each frame extract the missing stra. information and save the data set in hf"

## Requirements
| id | requirement |
|---|---|
| R-OP | The operative level covers **0 to 2 s**. Its prediction and its output reach 2 s. |
| R-TAC | The tactical level covers **up to 6 s** and **includes the tactical vocabulary**. Its prediction reaches 6 s. |
| R-STR | The strategic level covers **6 to 30 s** and includes **AV checkpoints**, **detected strategic information** and **goals derived from that information**. |
| R-TRAJ | The trajectory is the **combination of operative and tactical**, and is therefore **6 s** long. |
| R-ABS | A level's horizon must match its band. A 0.8 s step or reach is not a long-term abstract level. |
| R-VIS | Period of day and the linked illumination are estimated **from images and past frames only**, with no additional information at inference. |
| R-DATA | Missing strategic information is **extracted for each frame**, and the augmented dataset is saved on **HF** (private, on the PI's account, per the programme constitution). |
| R-RETRAIN | Whatever must change is fixed and retrained. Every training run still needs a launch-gate PASS. |

## Strategic information named by the PI so far
Country of the drive, estimated period of day, illumination (night, morning, and so on), weather. The PI calls these strategic because they are linked to long-term planning and adaptation.

## Standing rules that still apply
* Inference is vision-only; labels may use ego state, metadata and other privileged signals (PI, 2026-08-03). Nav is an input at training and at inference.
* The goal path stays information-disjoint from the **situation classifier** (PI, 2026-08-03).
* Every eval reports the four metric families, each with its tier and estimator. STRATEGIC is reported PAUSED (PI item 25) until the PI lifts it; this file does not lift it.
* Parity: the canonical corpus and its episode selection are not re-selected without saying so.

## Readings of the Master Mind, open to PI correction
1. "AV checkpoints" are read as the ego vehicle's route checkpoints across the 6 to 30 s band. Today's S head outputs checkpoints at 30, 50 and 80 m.
2. Country, period of day, illumination and weather are read as **strategic context**, distinct from the ego-dynamics **situation classifier** of the 2026-08-03 ruling, because the PI now asks for goals to be derived from them. The situation classifier's output stays out of the goal path.
3. Metadata (`country`, `hour_of_day`, `month`) may build or check labels. It is never an inference input.

## Added 2026-11-08, PI verbatim
> "yes use multi hierarchacy world model based planning as baseline and compare it to other approaches"

| id | requirement |
|---|---|
| R-PLAN | The **multi-hierarchy world-model planner is the baseline** of the redesign. It is compared with other approaches (the reactive v8F planner, a flat world-model planner, the kinematic floor P10). |
