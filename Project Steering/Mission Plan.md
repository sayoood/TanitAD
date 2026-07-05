
This document represents the initial vision setting version of the document describing my most important project TanitAD: Beat the best AD companies in three months. It will be changed only by me and will act as a constitution for this world class important challenge. If the agents aim to change the plan, they must generate proposals in separate documents. I will then decide to update this plan if necessary.

# Goals and Vision

The vision of this project is to create a AI driven autonomous driving stack with clearly outperforms and disrupts existing solutions within three months. This is the most challenging project you can have and I never had. Our declared Opponents and competitors to beat are clearly:
- Wayve located in UK
- Waymo
- Pony AI
- Momenta
- Autobrains
And any startup or established company which have a powerful autonmous driving stack or system.
Our goal ist to beat all these opponents with clear theoretic and practical proofs without fluffs and paper only work.

To be very clear: We will combine my 19 years experience in the field autonomous driving with AI driven research agents, all my resources even they are limited and all my ideas. I saw in my experience what is success and fail in the space. I saw different initiatives failing and doing the wrong thins. I'm seeing why the successful companies have so high financial value and what' matters in the current phase of the autonomous driving industry: A high performant running driving AI centered system. We will combine this thought with additional ideas and our system/safety driven moat and experience to create a running system disrupting the rest and showing this practically.

In a word this project Im starting a project which will create the next and most valuable European unicorn in the autonomous driving space which will scale to autonomous systems in  a second step.
The goal is not claim this vision and success in a theoretic manner describing the concepts, no we will show practically that we will disrupt all existing approaches in different phases and by practical and theoretic evidence which is recognizable (no own defined claims). We will control the scaling and the different phases of the program. We can achieve this because we have the most valuable experience in the space and also we don't have the legacy of large organizations, lacking of speed and consistency in the strategy and execution. We will clearly address L3 and L4 driving incl. lot2lot. L2++ is only a degradation of L4. No focus on low cost driver assistance systems (L0 - L2+) which will only lead to time loss and defocus.

In order to disrupt the existing autonomous driving systems we need to declare two different goals:
## Goal1
Show a clearly running system solving the autonomous driving problem with real and synthetic data in closed loop and based on embedded. We will clearly focus on the ORIN SOC and also the Thor SOC which I already have as Jetson thor and Jetson Orin prototyping boards. We will focus on inference outside the vehicle until we achieved the best possible results.

## Goal2
In addition to the running system, we need to leverage additional competitive disruptive advantages:
- (Prio goal) Show the generalization capability using far less data (magnitude less e.g. 1000x less)
- (Prio goal) No compromise in safety by design. Design an architecture which inherently safe and showing no compromise between safe modular ai architecture and E2E architecture. We will show that efficient End2End can be efficient and safe
- (Prio goal) Inference Efficiency: We will show that our approach will require less data and compute in inference and achieving better quality at the same time
- We will build an architecture which is inherently compliant to the new UN recent ADS regulation adopted recently
- Adaptability of the stack (driving style), generalization over different regions and markets

# The core ideas, Hypotheses and competitive edges

In order to to achieve Goal1 and Goal2, we need to formulate hypotheses which can create disruptive competitive edges.

**H0 there is no doubt that end2end systems have proven their superiority in comparison to rule based systems. We won't attack this fact. We will increase the effectiveness and efficiency of such systems and completely compensate their weakness but designing them differently and combine them with our ideas

**H1:  The 4B architecture is the core innovation of our program in combination of the LeCuns pattern of world models. The idea is based on having End2End three abstractions layers which have different tasks and act together as autonomous systems. These three brains are strategic layer, tactical layer and operative layers. All of them are end2end and are processing the input data (sensor data, ego motion, commands, parameters, etc...). The fourth brain is the fallback which is responsible to bring the system to mitigation risk condition (MRC) in case of collapse of the operative part or the tactical part of the system. The fallback can be also triggered by the strategic layer which is supervising the whole system and focuses on high level strategic steering of the whole autonomous system. The 4 brain architecture is an efficient and powerful one for autonomous allowing to achieve high performance by separating the abstraction levels and assure high transparency of decision. One of big weaknesses of SOTA end2end autonomous driving models. One of the most important tasks is to finalize the exact tasks assigned to the different layers. We will refine over the project. Nevertheless we will start by the following orientation defined by me.
- Operative layer: Is responsible for the driving task up to 0.5 seconds in consistency to the tactical subgoals and constraints. It can do short terms highly accurate predictions and is also able to drive autonomously if even tactical goals are missing. Its main goals are driving safely, avoiding accidents, be efficient, drive human like and don't break the rules (except of accident avoidance)
- Tactical layer: responsible for maneuver selection, planning up to 3 to 5 seconds, selection of the driving expert, sensor modality, side or camera views, lighting, signalling, gesturing and also managing interactions with other road users. Tactical functions generally occur over a period of seconds. Parts of it also is the tactical diagnosis as base for functional degradation, e.g. driving slower in case of heavy rain. Its able to perform tactical planning and prediction as base for the dynamic behavior of the vehicles 
- Strategic layer: responsible for engaging, deactivating, ending or blocking the usage of autonomous driving, diagnosis of the overall state of the system. Generating strategic commands like navigation, usage of the appropriate part of the road and also outputting strategic explanations and setting strategic goals regarding place to be, efficiency goals and overall strategy.  The strategic layer is responsible for strategic planning, strategic thinking, it has deep knowledge about driving strategies, autonomous driving in general and can perform strategic planning and prediction and evaluating different strategic scenarios. Its time horizon will ragen from 10 sec to minutes.
Its important that the different layers are able to work in both independent mode (can work standalone, process the input data and produce outputs) or together as combined intelligent system. In the second mode, its mandatory that the layers are working together and consistently. The higher abstraction layers are setting abstract goals and constraints which must be followed by the lower layers. E.g. the tactical layer is setting tactical subgoals and constraints/action which are followed by the operative layers. We will deeply investigate on possible conflicts and contradictions. This could be considered as a critic of the 4B architecture, at the same time, it could be also a strategic advantage since it is source of diversity, explainability, transparency and generalization capability. This results in the possibility to train the different layers together (end2end) and also standalone as pretraining or as postraining. The exact strategy will be set later. The end2end training is mandatory.

Its obvious that the different brains have . It matches perfectly to the "thinking slow", "thinking fast" paradigms and making the system very efficient in inference, since not all the parts must work with the same frequency e.g. 100 ms vs 500 ms vs. 1 second (subject of investigation)

I did different interesting experiments with 4B architecture focusing every time on different aspect like efficient recursive reasoning, self monitoring with mathematical guarantees, world models with anti collapse regularization measures, latent space and concept based thinking, latent rag approaches for continuous learning. The experiments are well documented in separate repos and be taken as inspiration without changing the goals of our projetc.

G:\Meine Ablage\SayBouBase\raw\Projects\AIFrontTierChallenge\Synthese\FormulatioofEdgeHypotheses&Evidences\Evidences\4B-HRM-Architecture

G:\Meine Ablage\SayBouBase\raw\Projects\AIFrontTierChallenge\Synthese\FormulatioofEdgeHypotheses&Evidences\Evidences\ACRE

G:\Meine Ablage\SayBouBase\raw\Projects\AIFrontTierChallenge\Synthese\FormulatioofEdgeHypotheses&Evidences\Evidences\ALPS-4B   (world model repo and main source of experiments)

G:\Meine Ablage\SayBouBase\raw\Projects\AIFrontTierChallenge\Synthese\FormulatioofEdgeHypotheses&Evidences\Evidences\RSRA-4B

**H2: The continuous usage of all sense modalities and camera data the whole time is a stupid approach. The choice of the right camera and sensor modality based on the internal assessment of the situation is a tactical capability assigned to the tactical layer. The AI driver can This will reduce  dramatically compute and enable autonomous driving on low compute platform. Think of the usage of radar sensors or side cameras as additional tools used in dependency of the current situation. I call this approach  Attention based Modality Steering and will be one of our core USPs in addition to the 4B architecture

**H3 Following the approach of World models where autonomous agents can predict the world around them through imagination and thus can derive the consequences of their actions and other agent's actions is the main direction we will explore at the begin. This fundamental approach has big advantages regarding the simple generalization, the explainability of behavior and also the amount of data we need. Recent LeWM publications using SigReg to stabilize the latent space prediction and avoid representation collapse + the succssful designs I developed in the 4B-ALPS (\SayBouBase\raw\Projects\AIFrontTierChallenge\Synthese\FormulatioofEdgeHypotheses&Evidences\Evidences\ALPS-4B) are encouraging to continue in this direction. I put the main learnings from this experiment and the transfer thoughts in this document SayBouBase\raw\Projects\AIFrontTierChallenge\Synthese\FormulatioofEdgeHypotheses&Evidences\Evidences\ALPS-4B\docs\AD_TRANSFER_RESEARCH

**H4 Frozen pretrained encoders to reduce training effort: Here we still follow the same approach as H3 which is the main direction, but we want to compare it to the usage of a largely pretrained encoder backbone which is then combined by the LeCun world model and prediction paradigm to achieve high quality by keeping the training very simple. Im interested in comparing this approach with the unpretrained one. We can also combine the approaches by taking the pretrained encoders ans put them in the unsupervised pipeline. Generally I want to investigate on the fact of relying on existing architectural parts, freeze parts of them and focus on fine tuning, specific training or post training.

**H5 transfer efficient inference and decoding techniques like speculative decoding, sparse attention, DSPARK, MTP, continuous flow matching, etc... from the large model  research to automated driving. There must be an initial resarch and then an iterative research work performed by the TanitAD research hub to leverage these possibilities

**H6 In order to create a instantaneous advantage moat include all the known dump and weak situations of Waymo, ponyai etc... in the solution space of TanitAD. e.g. by creating training data for these situations, by generating test cases for these scenarios and situations and also design the architecture addressing these challenges

**H7 in order to prove the data efficiency, we should eventually leverage different source of data and cameras.  Following this direction, we can combine youtube videos, dash cams, smartphone videos, etc.. We need eventually an inverse dynamic model which is trained to extract the action from driving videos, this will enable us to use unlabeled videos as considered in recent world action models. One important direction to achieve this is to leverage the main findings of the VLM3 work from Meta, which transforms all images to one fictive local length improving the 3d spacial reasoning of the models. We need to investigate on this and define an implementation strategy.

**H8 in combination with H2, one interessant research and implementation direction could be to investigate a MOE architecture not only to select the right sensor, camera view as experts but other capabilities depending on the situation and other factors. This is prio 2 consideration, bu we should keep it in mind.

**H9 in order to increase the explainability and the generalization of the TanitAD stack, we need to find way to inherently include the compliance to traffic rules in the design of the AI model and also the training workflow and post training improvements to assure traffic rules compliance or to change behavior as post training in context of compliance to traffic rules. We need to find a way to incrementally add traffic rules without retraining the whole system and without loosing he end2end character of the system

**H10 We want to leverage the latent RAG approach or similar new approaches allowing the TanitAD stack to continuously improve its capabilities and learn by logging its experiences, successes and mistakes and also derive autonomously measure. It should also process external feedback by the user e.g. or any other  to indicate a bad or criticized behavior. This is a very important point, which must included in our approach because it's disruptive

**H11 Self monitoring is what differentiates real autonomous systems from assistance systems. The TanitAD stack must have this capability which reflects the ability of detecting its own limitations and initiate the appropriate actions based on this monitoring. the monitoring must also detect incidents, critical situations and automatically generate the required reports as requested by the the new ADS regulations. As for now this task will be assigned to the strategic part which should be generative and able to process text. this matches to the idea of a slow thinking heavy brain which is able to think strategically and which is able to generate incident monitoring reports. I think we will need different self monitoring at each abstraction level. This must be elaborated and designed desperately.

**H12 Text as part of the architecture: While its stupid to drive with language, our system must process language as additional part and not as the core of the autonomous system I don't believe on the impact of VLAs despite they showed some good results. So we will stick to the unsupervised world model as main direction, but we will extend it by a text processing part (could be an adapted llm backbone). It must process e.g. strategic navigation commands, adapt the behavior of the stack and output interpretable reasoning traces as text. The challenge is to find an effcient way to design a common latent space which includes all

**H13 We need extraction heads, we need to include extraction heads to assure the usability of the TanitAD stack and also increase the explainability without loosing the End2End character of the AD system. We will finalize the exact heads we need to extract the required information from the latent space. I think we will need an environment model exaction heads to visualize the vehicle environment as part of the HMI, a behavior extraction head extracting the chosen behavior and the considered alternatives...

**H14 safe and performant real driving intelligent models must be grounded in real physical world. We need to find a way to inject wide knowledge relevant to driving taks like relevant physical laws, code of behavior, efficient driving, cultural differences in the different regions of the world, general ethics principles, etc.. I don't know now how to do it but this will be very interesting direction, because it will disrupt all the current approaches. eventually, we need to elaborate on the last research works and results 

**H15 One of the important hypotheses related to the prediction capability of the intended world models and related evaluation of the consequences of its actions, is the capability of imagination about the world state even in unobserved areas based on e.g.  single sensor imagination capability. The analogy to humans is very important here. Humans are able to keep an internal representation of other traffic participants even they are not observing them actively. Exactly this capability must be achieved by our system. It will enable H2 and thus the efficient inference for autonomous driving disrupting existing approaches.
# The structure of the projects

The project is structured in different subparts. Every part has its own folder. Each folder acts as sub repo for the related part. The main folder TanitAD is the main repo of the project, including all the others. 

The main repo is already created in github in this adress: https://github.com/sayoood/TanitAD

In addition to the subprojects, the main repo includes the Tanit AD research hub, the backbon eresearch hub of our project.
## Project Steering
Is the part of the project responsible for defining and refining the goals, measuring performance and progress, generate high quality progress reports and steer the project in terms of defining the focus of the next phase, updated every week. It also measure the ressources consumption and decides about investments and resources related decisions.
## Data engineering
The part of the project dealing with data preparation, curation, filtering, augmentation and the selection of suitable datasets for the different phases of the project. This part will also deal with the training strategy and workflow. Even it is not the same as data engineering, we will keep them together to reduce complexity and increase efficiency. The declared goal here is to define a disruptive training and data flying wheel which disrupt the existing ones in the field of autonomous driving. we will combine pretraining, continual training with post training, fine tuning and if necessary reinforcement learning for behavior adaptation and alignment, etc...
The data and training approach must start simple and compatible to scale in later phases of the project to more professional, large scale, production like pipeline.
One of the most important task at the begin to define a rich, simple to use and access dat set suitable to quickly finish the first phases and milestones of the project.
## Architecture & Inference
This is the core of the AD system including the architecture design of the system and all measures taken to make inference highly efficient fitting to edge devices and more importantly fit to mid cost/low cost automotive HW as one of the most important goals of this project. We will stick to the orin and the thor as target SOCs, but that does not mean, we should use all of the compute they are offering. Efficiency is one of the most important moat hypotheses. All measures regarding architecture design like e.g. efficient and tricky encoding strategies of high resolution images, effcient decoding of trajectories or any other output from rich latent space, quantization techniques like TurboQuant or other techniques, must be considered, augmented and new ones must be leveraged from latest impactful research works. We should think in this stream of how to prepare the model for deployment on embedded by leveraging optimization frameworks like ONNX or graph based scheduler for efficient deployment on automotive ECUs.

## Tools&DevEnv
This is the part of the project responsible for incrementally develop and enhance the development and deployment tools and environment including mainly a replay and visualization environment like AlpaSim or ROS, a closed loop simulation environment with real data augmentation like AlpaSim or CARLA, a self play environment for policies and heads, reinforcement learning environment like AplaGym and evaluation framework to show progress etc..
We will here begin in simple way and increase scaling complexity step by step. We should leverage any high quality proven open source
One of the declared goal is to use and adapt AlpaSim to our own system, since Alpasim is particularly developed for the testing of end2end AD systems. We need here a tools and Dev environment strategy defined in a precise way.
## Benchmarks & Eval
In this part we will develop well defined benchmarks and tests to prove clearly the advantages of our system. Considering well recognized autonomous driving benchmarks is a must. We will in addition develop additional high quality own metrics to prove additional edges which are not covered by current benchmarks and KPIs. In this part we will run final and independent tests are part of our releases and also develop and maintain a leader board for the different known systems and also checkpoints or intermediate releases of the TanitAD stack.
The benchmarks and Evals must include a validation strategy to prove the edges and also an eval strategy in order to run efficiently tests without loosing too much resources. The benchmarks must cover generalization capabilities, control and behavior correctness and precision, correctness of reasoning, efficiency of inference interns of compute and memory and any other KPIs/metrics missing.

## TanitAD Research Hub
The research hub will contain all the subparts of the projects as folders (already established). Every folder will include two sub folders one Theoretic Research (called research), the other one contains real MV implementation (called implementation). 

Each agent must do an intensive post doc quality research work by search one day in the week over all possible sources of information like news, publications, YouTube videos, technical reports. It must keep an incrementally increasing knowledge base for its discipline and focus on new updates with high impact on our projects. it must be autonomous and with highest quality. The theoretic research work mus be as post doc, senior engineer and senior strategic advisor for each discipline. It must analyses the identified facts, last news, last impactful research and engineering results with clear relevance and impact on our project and our defined goals. The agents must be able to do research for all my hypotheses described in this document, in order to refin ethem, validatethem improve them and generate eventually new onmes, same for the plan.  

For each sub parts there will be an gent doing the theoretic work and implementation work. All the agents must run once in the week. Every agent will be assigned to a defined day. On this day, the agent must do the theoretic and  the practical work. Here is the plan

**Monday: Tools&DevEnv Agent
Tuesday: The Data Engineering agent
Wednesday: Architecture & Inference
Thursday: Benchmarks & Eval
Friday: The Opponent Analyzer agent and Project steering agent: Orchestrator and Syntheizer. This agent will monitor the health state of the different agents, monitor them and also do the synthesis work to aggregate all results to generate a complete and rich weekly report reflecting both directions of the project: First, the research hub including the theoretic and practical work, Second the Core development including the MVP implementation which will conducted and steered manually by me every day. The orchestrator should control the progress of both streams of the project and identify concrete proposals to our main stream which is the MVP implementation conducted by my self. One important of the report must be the documentation of the current benchmark and KPI driven progress of the TanitAD stack.  

The idea is to organize sequentially the agents. So first the tools&DeveEnv agent updates its findings, knowledge base about the development environment. The data and training engineering agent makes its research and consider the results form the previuos agents. Same for the later agents like Architecture & Inference etc...

In addition to the theoretic and practical research work related to each subpart of the project, we will add a research agent dealing with the continuous analysis of our opponents I specified earlier (Opponents Analyzer agent). This includes the analysis of their results, business models, news, strategies, strengths and weaknesses, analysis of technical and scientific publications. The goal is to identify areas of attacks and derive ideas to strengthen the moat of our system and make it more robust so we can improve our story telling for our vision.

It's important to quickly establish the research hub working in appropriate way in order not to loose time in correcting the hub, since speed is decisive in this challenging program.


# Strategy and Execution Plan

## Overall Strategy and Constitution
This is the most important project of my career, it will require discipline hard working and patience.
It must follow my constitution principles:
### P1: Be fast, flexible and take decisions with determination
### P2: Stick to our principles without being dogmatic, be open for a new direction without loosing our core believes

### P3: No contradiction between research and production driven engineering. We will do every thing what even it requires and takes, both scientific work and engineering work

### P4: Don't forget our ultimative goal: Beat the best as production ready system for autonomous driving

### P5: Our limited resources is our strength. If you don't have resources, you must be efficient. If you are efficient, you will have an edge and competitive advantage

### P6: Our willing to make this project a success is endless. Giving up is not an option. I don't care about possible failures. They will make us stronger and we will learn from them.

### P7: First final evaluation on 05.10.2026

### P8: Be honest, admit failures problems and not working edges

The overall strategy will include different phases: Phase 0, 1 and 2. 
The minimal goal of the program is to achieve at least very successful phase 0. Ideally we will achieve phase 1 and 2. 
## Resources and accounts

I will provide and use different resources in this project in oder to achieve all our goals. The resources include Hardware, Software, AI driven agentic systems, cloud compute and different acesses

Hardware:
- A development PC with 32 GB of RAM with RTX 4060 with 4B of VRAM, which will be themain local dev machine. We will heavily leverage the local gpu for any tiny model tests, smoke test, validation steps or any activities avoidding the abuse of the usage of cloud GPU compute
- A jetson ORIN dev board
- A jetson Thor dev Board
- A runpod account to acess A40 GPU, A100 and H100 for heavy training and tests
- Leverage the usage of google colab environment, in partcular the agents must use the recently published usage of colab as cli tool. This can be used as skill for the research agent

Software:
- A gemini pro account: fambouzouraa@googlemail.com
- The max plan of claude code sayedbouzouraa@googlemail.com
- An account for Huggingface, my user account ist: https://huggingface.co/Sayood
- A github account: sayedbouzouraa@googlemail.com   https://github.com/sayoood
- A google AI studio account with free API key for limited and free usage of gemini model
- An antigravity based dev environment

Data:
- Own small amount of data based on smartphone sensors and ground truth reconstruction developed by my self
- Own GoPro 13 data including GPS and IMU data

Documents:
- A first analysis and transfer plan from my first experiments with the 4B architecture
- A bundle of analysis on the approaches of AD companies 
- The ADS UN regulation adopted end of June 26 as the most important regulation document
- Deep think anaylsis taks for the gemini deep think model delivering research work as input
- A bunch of analysis from google deep think model, consider this als additional input not to replace own thoughts
## Execution Plan

The most important goal and plan is to achieve phase 0 as fast as possible.
Phase 0 includes a running architecture withe most important hypotheses like the 4B architecture, world model based approach, working with real existing data set ideally both in open loop and closed loop. We can begin the poof with one single front camera achieving good performance including reasoning and the MOE tactical to steer the usage of side, rear cameras and additional sensor data like radar and lidar data.

Phase 1: includes the rest of the edges and a boost of the performance. In this phase, I want to leverage the usage of vast amount of unlabeled data including my own data from my smartphone, youtube videos, go pro video. Also scale the model and the number of benchmarks. In this phase, we will professionalize more the develeopment environment and scale the used resources

Phase 3: The real scaling phase achieving the final results, proving all edges and hypotheses with outstanding results and the system ready to be deployed on the vehicle.


