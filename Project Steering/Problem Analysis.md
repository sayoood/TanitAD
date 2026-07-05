



### Option 1: Direct & Technical (Best for engineering discussions)

> "In autonomous vehicle development, there are several fundamental unsolved problems: building perception and planning models that are truly grounded in the physical reality of a moving two-ton vehicle, models that understand the causality of traffic behavior rather than just statistical pixel correlations, and models that can keep learning safely after the car rolls off the lot. That last one is especially hard because an individual vehicle needs to adapt to its local driving environment—like regional weather or specific road quirks—while still benefiting from the global knowledge expansion of the entire fleet. We still do not have a clean solution for achieving this continuous, over-the-air fleet learning under the strict safety and compute constraints of real-world driving."

### Option 2: Executive Summary (Best for pitches or high-level overviews)

> "Achieving true Level 5 autonomy still faces core AI hurdles. First, our driving models must move beyond processing pixels to actually understanding physical reality and the causality behind human behavior—knowing _why_ a pedestrian stopped, not just that they did. Second, we have an ongoing deployment challenge: safely updating the AI on the road. A deployed self-driving car must learn the nuances of its specific daily route while continuously integrating updates from the wider global fleet. Executing this kind of local-to-global continual learning, while maintaining absolute safety and operating on edge compute, remains one of the industry's biggest unsolved problems."

### Option 3: Problem-Statement Breakdown (Best for slide decks or strategy docs)

The path to fully autonomous driving is currently blocked by three fundamental AI challenges:

- **Lack of physical grounding:** Our models must understand the actual physics of driving (mass, friction, momentum), rather than just recognizing visual patterns.
    
- **Correlation over Causality:** Vehicles need to understand _why_ events happen on the road (e.g., a rolling ball means a child might follow) rather than just reacting to statistical correlations.
    
- **Safe Continual Deployment:** Once a car is on the road, its AI needs to adapt to local environments (like a snowy climate) while safely integrating over-the-air learnings from the global fleet. Doing this without causing "catastrophic forgetting" or violating strict safety constraints remains an unsolved problem in edge AI.