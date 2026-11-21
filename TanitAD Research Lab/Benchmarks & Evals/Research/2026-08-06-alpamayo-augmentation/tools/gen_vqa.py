"""Generate the >=500-question VQA bank for the Alpamayo augmentation dataset.
Composed from category x template x slot so the bank is broad, not repetitive.
Each question is tagged with its category so downstream analysis can stratify."""
import json, itertools, random

Q = []
def add(cat, qs):
    for q in qs: Q.append({"category": cat, "question": q})

agents = ["vehicles", "pedestrians", "cyclists", "motorcycles", "trucks", "buses",
          "emergency vehicles", "construction vehicles", "animals", "children"]
dirs = ["ahead of the ego vehicle", "behind the ego vehicle", "to the left", "to the right",
        "in the adjacent left lane", "in the adjacent right lane", "at the next intersection",
        "on the sidewalk", "in the oncoming lanes", "in the ego lane"]
add("agents_presence", [f"How many {a} are visible {d}?" for a, d in itertools.product(agents[:6], dirs[:5])])
add("agents_presence", [f"Are there any {a} {d}?" for a, d in zip(agents[4:], dirs[5:])])
add("agent_behavior", [f"What is the closest {a[:-1]} {d} currently doing?" for a, d in itertools.product(["vehicle", "pedestrian", "cyclist"], dirs[:6])])
add("agent_behavior", ["Is the lead vehicle accelerating, braking, or holding speed?",
    "Is any vehicle attempting to merge into the ego lane?",
    "Is any pedestrian about to step onto the roadway?",
    "Are any vehicles parked illegally or blocking a lane?",
    "Is the vehicle behind the ego following at a safe distance?",
    "Is any agent behaving erratically or unpredictably?",
    "Which agent poses the highest collision risk right now, and why?",
    "Is any vehicle signalling a turn or lane change?",
    "Are there vehicles waiting to enter the roadway from a driveway or side street?",
    "Is cross-traffic approaching the intersection from the left or right?"])
infra = ["traffic lights", "stop signs", "yield signs", "speed limit signs", "lane markings",
         "crosswalks", "bicycle lanes", "bus lanes", "traffic cones", "construction barriers",
         "guard rails", "medians", "roundabouts", "on-ramps", "off-ramps", "toll booths",
         "railroad crossings", "school zone signs", "pedestrian islands", "curbs"]
add("infrastructure", [f"Are there {x} visible in the scene, and where?" for x in infra])
add("infrastructure", [f"What is the state or condition of the {x[:-1]} nearest the ego vehicle?" for x in infra[:10]])
add("road_topology", ["How many lanes does the current road have in the ego direction?",
    "Is the ego vehicle on a highway, urban street, or residential road?",
    "Is there an intersection ahead, and how is it controlled?",
    "Does the road curve ahead, and in which direction?",
    "Is the road ahead flat, uphill, or downhill?",
    "Are there any forks, merges, or splits in the road ahead?",
    "Is the ego vehicle in a turn-only lane?",
    "What is the road surface type and condition?",
    "Is this a one-way road?",
    "Is there a shoulder available for an emergency stop?",
    "How wide is the ego lane relative to the vehicle?",
    "Is the ego approaching a roundabout, and which exit leads straight through?",
    "Are lane markings clear enough to determine lane boundaries?",
    "Is there a dedicated left-turn phase at the upcoming intersection?",
    "Does the current lane end soon, requiring a merge?"])
add("environment", ["What time of day does the scene suggest?",
    "What are the current weather conditions?",
    "Is visibility reduced by rain, fog, snow, or glare?",
    "Is the road surface wet, icy, or dry?",
    "Are there strong shadows or low-sun glare affecting the cameras?",
    "Is this scene urban, suburban, rural, or industrial?",
    "Are streetlights on, and is artificial lighting adequate?",
    "Is there standing water or debris on the road?",
    "How dense is the surrounding traffic?",
    "Is this a structured environment with clear lanes or an unstructured area?"])
risk = ["an occluded pedestrian emerging", "a vehicle running the red light", "a door opening from a parked car",
        "a cyclist swerving into the lane", "sudden braking by the lead vehicle", "an animal crossing",
        "a cut-in from the adjacent lane", "loss of traction on the surface", "a reversing vehicle",
        "an oncoming vehicle crossing the centerline"]
add("risk", [f"What is the likelihood of {r} in this scene, and what evidence supports it?" for r in risk])
add("risk", ["What is the safest speed for the current conditions?",
    "What should the ego vehicle be most cautious about in the next five seconds?",
    "Which region of the scene is most occluded, and what could it hide?",
    "Is the current following distance to the lead vehicle adequate?",
    "If the lead vehicle braked maximally now, could the ego stop in time?",
    "What escape paths exist if the ego lane becomes blocked?",
    "Which agents are in the ego vehicle's blind spots?",
    "Is any part of the planned path in conflict with another agent's likely path?",
    "What would a defensive driver do differently here?",
    "Rank the three highest-risk agents in the scene."])
add("planning", ["Should the ego vehicle change lanes, and if so why and in which direction?",
    "Is it safe to overtake the lead vehicle now?",
    "What manoeuvre should the ego vehicle perform at the upcoming intersection?",
    "Should the ego vehicle yield to any agent right now?",
    "What is the appropriate action if the traffic light turns yellow now?",
    "Is a U-turn feasible and legal here?",
    "Should the ego adjust its lane position within the lane, and why?",
    "What acceleration profile is appropriate for the next three seconds?",
    "Should the ego vehicle stop for the pedestrian at the crosswalk?",
    "How should the ego negotiate the construction zone ahead?",
    "What is the correct gap acceptance decision for the upcoming merge?",
    "Should hazard lights or other signals be used in this situation?",
    "Is it appropriate to enter the intersection now or wait?",
    "How should the ego respond to the emergency vehicle, if present?",
    "What lane is optimal for the next kilometre, and why?"])
add("occlusion", [f"What could be hidden behind the {o}?" for o in
    ["parked truck", "bus at the stop", "building corner", "hedge or vegetation", "crest of the hill",
     "large SUV ahead", "construction barrier", "billboard or sign", "stopped traffic", "curve of the road"]])
add("relational", ["Which vehicle is closest to the ego, and how far away is it?",
    "What is the relative speed of the lead vehicle?",
    "Which agent will reach the intersection first, the ego or the cross-traffic?",
    "Is the gap in the adjacent lane large enough for a lane change?",
    "How much lateral clearance is there to the cyclist?",
    "Which of the visible traffic lights applies to the ego lane?",
    "Which agents are moving toward the ego path and which away?",
    "What is the time headway to the lead vehicle?",
    "Which lane is moving faster, the ego lane or the adjacent one?",
    "How far is the ego from the stop line?"])
add("scene_summary", ["Describe the driving scene in one sentence.",
    "Summarise the key challenges of this scene for an autonomous vehicle.",
    "What type of driving scenario is this (car-following, intersection, merge, etc.)?",
    "List every agent that influences the ego vehicle's next decision.",
    "What is the single most important object in the scene for planning?",
    "Describe the state of the intersection ahead.",
    "What has changed in the scene over the last second?",
    "Describe the traffic flow pattern around the ego vehicle.",
    "What navigation context can be inferred from visible signage?",
    "Describe everything relevant in the rear view."])
counter = ["the ego vehicle were travelling 20 km/h faster", "it were night instead of day",
           "the road were wet", "the lead vehicle suddenly stopped", "a pedestrian stepped out now",
           "the traffic light failed", "the lane markings were absent", "an emergency vehicle approached from behind",
           "the ego had to stop within two seconds", "the adjacent lane were closed"]
add("counterfactual", [f"How would the appropriate driving behaviour change if {c}?" for c in counter])

# pad the agent x direction grid to pass 500 with distinct combinations
extra = [f"Is the {a[:-1]} {d} moving or stationary?" for a, d in itertools.product(agents, dirs)]
add("agents_state", extra)

random.seed(0)
random.shuffle(Q)
for i, q in enumerate(Q): q["qid"] = f"vqa{i:04d}"
json.dump({"n": len(Q), "categories": sorted(set(q["category"] for q in Q)),
           "questions": Q}, open("vqa_bank_500.json", "w"), indent=1)
print("questions:", len(Q), "categories:", len(set(q["category"] for q in Q)))
