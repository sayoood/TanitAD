---
datasets:
- nvidia/PhysicalAI-Autonomous-Vehicles
- nvidia/PhysicalAI-Autonomous-Vehicles-NuRec
pipeline_tag: robotics
license: openmdw-1.1
language:
- en
base_model:
- nvidia/Cosmos3-Super
tags:
- alpamayo
---
# Alpamayo 2 Super

[**Code**](https://github.com/NVlabs/alpamayo2) | [**Alpamayo Overview**](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/)

## Model Overview:

### Description:

Alpamayo 2 Super is a 34B-parameter foundation model designed to tackle multiple autonomous vehicle (AV) development tasks. It combines a 32B VLM backbone with a 2B diffusion expert.

Alpamayo 2 Super was developed by NVIDIA as a part of the broader [Alpamayo Open Platform](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/).

### License/Terms of Use:

**Model weights:** The model weights are released under the [OpenMDW-1.1](https://openmdw.ai/license/1-1/) license.

**Source code:** Apache License 2.0, as provided in the Alpamayo 2 Super source repository.

### Deployment Geography:

Global

### Use Case:

Developers and researchers working on autonomous vehicle systems who need a foundation model for perception, planning, and decision-making tasks. Alpamayo 2 Super supports multiple AV development tasks such as trajectory prediction, visual question answering, 2D grounding, and auto-labeling. It enables enterprises to accelerate AV software development with a unified model that combines vision-language and diffusion expertise.

### Release Date:

**Hugging Face:** 08/04/2026 via https://huggingface.co/nvidia/Alpamayo2-Super

## Model Architecture:

**Architecture Type:** Transformer

**Network Architecture:** Vision-Language-Action (VLA) model with a VLM backbone and diffusion-based action decoder.

**This model was developed based on:** Cosmos 3 Super Reasoner with a diffusion-based action decoder.

**Number of model parameters:**

- Backbone: 32B parameters
- Action Expert: 2.3B parameters

## Input:

**Input Type(s):** Image/Video, Text, Egomotion History

**Input Format(s):**

- Image: Red, Green, Blue (RGB)
- Text: String
- Egomotion History: Floating-point values `(x, y, z), R_rot`

**Input Parameters:**

- Image: Two-dimensional (2D), multi-camera, multi-timestep
- Text: One-dimensional (1D)
- Egomotion History: Three-dimensional (3D) translation and nine-dimensional (9D, 3x3) rotation, multi-timestep

**Other Properties Related to Input:** The validated public notebook profiles use six cameras and four historical frames per camera. VQA uses camera IDs `[0, 1, 2, 3, 4, 5]` (cross left, front wide, cross right, rear left, rear tele, and rear right). Trajectory, meta-action, auto-labeling, and grounding use camera IDs `[0, 1, 2, 3, 5, 6]` (cross left, front wide, cross right, rear left, rear right, and front tele). The four frames are synchronized context frames ending near `t0`; their exact timestamps come from the input sample. Images and ego-motion history require associated timestamps. Image resizing and normalization are performed by the packaged processor rather than by a fixed public raw-image resolution contract.

## Output:

**Output Type(s):** Text, Trajectory

**Output Format(s):**

- Text: String (Chain-of-Causation reasoning traces or visual question answers or meta-actions)
- Trajectory: Floating-point values `(x, y, z), R_rot`

**Output Parameters:**

- Text: One-dimensional (1D)
- Trajectory: Three-dimensional (3D) translation and nine-dimensional (9D, 3x3) rotation, multi-timestep
-
- **Other Properties Related to Output:** The trajectory API returns 64 waypoints spanning 0.1 through 6.4 seconds at 0.1-second intervals. Each trajectory contains ego-frame XYZ positions and 3x3 rotation matrices. Text outputs include variable-length Chain-of-Causation reasoning, meta-actions, visual question answers, grounding coordinates, and structured auto-labeling fields.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:

**Runtime Engine(s):**

- PyTorch (minimum version: 2.8)
- Hugging Face Transformers (minimum version: 4.57.1)
- DeepSpeed (minimum version: 0.17.4)

**Supported Hardware Microarchitecture Compatibility:**

- Tested: NVIDIA H100 80GB HBM3
- Other GPU architectures have not yet been validated.

**Preferred/Supported Operating System(s):** Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):

Alpamayo 2 Super 34B trained

Can be integrated into autonomous driving software in the cloud for advanced end-to-end perception, reasoning, and motion planning.

## Training, Testing, and Evaluation Datasets:

## Training Dataset:

**Data Modality:**

- Image
- Text
- Video
- Actions

**Image Training Data Size:** More than 1 Billion Images

**Text Training Data Size:** Less than a Billion Tokens

**Video Training Data Size:** 10,000 to 1 Million Hours

**Data Collection Method by dataset:** Automatic/Sensors

**Labeling Method by dataset:** Hybrid: Automated/Manually-Labelled

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**
The dataset comprises roughly 115,000 hours of multi-camera driving video with corresponding egomotion and trajectory annotations.
It includes roughly 3,700,000 Chain-of-Causation (CoC) reasoning traces that provide decision-grounded, causally linked explanations of driving behaviors.
Content includes machine-generated data from vehicle sensors (cameras, IMUs, and GPS) and synthetic reasoning traces.
CoC annotations are in English and use a structured format that links driving decisions to causal factors.
Sensors include RGB cameras, inertial measurement units, and GPS.

### Evaluation Dataset:

**Quantitative Evaluation Benchmarks:** 

- Reasoning Evaluation using [LingoQA](https://github.com/wayveai/LingoQA): Lingo-Judge Score of 79.2.
- Closed-Loop Evaluation using [AlpaSim](https://github.com/NVlabs/alpasim) on 910 scenarios from the [PhysicalAI-AV-NuRec Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec): AlpaSim Score of 1.50 ± 0.13.
- Open-Loop Evaluation on 1434 challenging samples from the [PhysicalAI-AV Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles): minADE_6 at 6.4s of 0.911m.

Additional evaluations and comparisons to prior models are visualized below:

![Multi-task benchmark results](images/Alpamayo-2-Super_Benchmark-Results_A.png)

![LingoQA benchmark results](images/Alpamayo-2-Super_Benchmark-Results_B.png)

**Data Collection Method by dataset:** Automatic/Sensors

**Labeling Method by dataset:** Hybrid: Automatic/Sensors/Manually-Labelled

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** This dataset covers multi-camera driving scenarios with a particular focus on safety-critical, long-tail events. It includes challenging cases such as complex intersections, cut-ins, pedestrian interactions, and adverse weather conditions. Data are collected from RGB cameras and vehicle sensors.

## Inference:

**Acceleration Engine:** Other: PyTorch, Hugging Face Transformers

**Hardware Requirements (GPU Architecture, Model):**

- Tested on: 1 NVIDIA H100 80GB HBM3 GPU
- Measured configuration: seven cameras, four frames per camera, batch size 1, one trajectory sample, BF16, PyTorch SDPA, classifier-free guidance disabled, and 10 diffusion steps.
- Measured peak: 72,115 MiB device memory, with 69.316 GiB peak PyTorch allocation and 69.463 GiB peak PyTorch reservation.
- The six-camera public notebook profiles have not yet been separately memory-profiled. Memory use also varies with camera and frame counts, trajectory sample count, classifier-free guidance, attention backend, and device placement.

## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.
Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.
For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards.
Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Support

📣 **Usage questions**: post on the [Alpamayo NV Developer Forum](https://forums.developer.nvidia.com/c/autonomous-vehicles/alpamayo/766).

🐛 **Code bugs / documentation issues / feature requests**: file a GitHub issue using the appropriate template (Bug report, Documentation request, or Feature request) at https://github.com/NVlabs/alpamayo2/issues/new/choose . The relevant NVIDIA responder is auto-assigned.

🚨 **Security vulnerabilities**: please use [NVIDIA's Vulnerability Disclosure Program](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). Do not file public issues.