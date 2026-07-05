from tanitad.data.toy_driving import (ToyDrivingDataset, ToyEpisode,
                                      frame_change_fraction, generate_episode)
# MetaDrive adapter helpers import cleanly (MetaDrive itself is imported lazily,
# only inside generate_metadrive_episode), so this is safe with no sim deps.
from tanitad.data.metadrive_env import (MetaDriveDataset,
                                        generate_metadrive_episode)
# comma2k19 (D-009 primary corpus). Video decode (`av`) is imported lazily inside
# the decode path only, so importing this is safe with no codec/data present.
from tanitad.data.comma2k19 import (Comma2k19Dataset, build_episode,
                                    discover_segments, split_by_route)
# Consequence-dominance (A8) statistics harness — measures per-corpus change
# fraction for change-weighting (H3/A8) and the D-010 real-vs-sim mix decision.
from tanitad.data.stats import (consequence_dominance_stats, stats_by_label)

__all__ = ["ToyDrivingDataset", "ToyEpisode", "generate_episode",
           "frame_change_fraction", "MetaDriveDataset",
           "generate_metadrive_episode", "Comma2k19Dataset", "build_episode",
           "discover_segments", "split_by_route",
           "consequence_dominance_stats", "stats_by_label"]
