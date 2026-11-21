#!/usr/bin/env python3
"""Post-hoc nuPlan closed-loop non-reactive aggregation for an E3 run (2026-11-12).

Why: run_simulation was given output_dir=<run>/out, so no metric path contains the challenge name and the
MetricAggregatorCallback filter (metric_aggregator_callback.py:46, `metric_aggregator.challenge in str(path)`) selected no
file ("No metric files found for aggregation!"). This script reproduces that callback's body verbatim (lines 31-59: load every
<out>/metrics/*.parquet via MetricStatisticsDataFrame.load_parquet, key by metric_statistic_name, call the aggregator), with
the aggregator built from the SAME config values as
nuplan/planning/script/config/simulation/metric_aggregator/closed_loop_nonreactive_agents_weighted_average.yaml.
The challenge filter is the only step skipped, because every metric file in <out>/metrics belongs to this one run.
usage: aggregate_posthoc.py <run_out_dir>
"""
import sys
from pathlib import Path

from nuplan.planning.metrics.aggregator.weighted_average_metric_aggregator import WeightedAverageMetricAggregator
from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame

out = Path(sys.argv[1])
save = out / "aggregator_metric"
save.mkdir(parents=True, exist_ok=True)
agg = WeightedAverageMetricAggregator(
    name="closed_loop_nonreactive_agents_weighted_average",
    metric_weights={"ego_progress_along_expert_route": 5.0, "time_to_collision_within_bound": 5.0,
                    "speed_limit_compliance": 4.0, "ego_is_comfortable": 2.0, "default": 1.0},
    file_name="closed_loop_nonreactive_agents_weighted_average_metrics_posthoc",
    aggregator_save_path=save,
    multiple_metrics=["no_ego_at_fault_collisions", "drivable_area_compliance", "ego_is_making_progress",
                      "driving_direction_compliance"],
    challenge_name="closed_loop_nonreactive_agents",
)
dfs = {}
for f in sorted((out / "metrics").rglob("*.parquet")):
    d = MetricStatisticsDataFrame.load_parquet(f)
    dfs[d.metric_statistic_name] = d
print("metric files", len(dfs), sorted(dfs))
agg(metric_dataframes=dfs)
print("written", sorted(p.name for p in save.iterdir()))
