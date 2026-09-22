from types import SimpleNamespace

from mlagents_envs.side_channel.stats_side_channel import StatsAggregationMethod

from finssim_rl.envs.unity_stats import drain_unity_stats


def test_drain_unity_stats_accepts_metrics_from_multiple_tasks():
    class StatsChannel:
        def get_and_reset_stats(self):
            return {
                "FinsROV/hold/position_error_mean_m": [(0.8, 1), (0.2, 1)],
                "FinsROV/trajectory_tracking/tracking_error_p95_m": [(0.9, 1), (0.4, 1)],
                "FinsROV/unrelated": [(9.0, 1)],
            }

    env = SimpleNamespace(_finssim_stats_channel=StatsChannel())
    assert drain_unity_stats(env) == {
        "FinsROV/hold/position_error_mean_m": 0.2,
        "FinsROV/trajectory_tracking/tracking_error_p95_m": 0.4,
        "FinsROV/unrelated": 9.0,
    }


def test_drain_unity_stats_supports_mlagents_aggregation_enums():
    class StatsChannel:
        def get_and_reset_stats(self):
            return {
                "FinsROV/trajectory_tracking/most_recent": [
                    (0.8, StatsAggregationMethod.MOST_RECENT),
                    (0.2, StatsAggregationMethod.MOST_RECENT),
                ],
                "FinsROV/trajectory_tracking/sum": [
                    (0.8, StatsAggregationMethod.SUM),
                    (0.2, StatsAggregationMethod.SUM),
                ],
                "FinsROV/trajectory_tracking/average": [
                    (0.8, StatsAggregationMethod.AVERAGE),
                    (0.2, StatsAggregationMethod.AVERAGE),
                ],
                "FinsROV/trajectory_tracking/histogram": [
                    (0.8, StatsAggregationMethod.HISTOGRAM),
                    (0.2, StatsAggregationMethod.HISTOGRAM),
                ],
            }

    env = SimpleNamespace(_finssim_stats_channel=StatsChannel())
    assert drain_unity_stats(env) == {
        "FinsROV/trajectory_tracking/most_recent": 0.2,
        "FinsROV/trajectory_tracking/sum": 1.0,
        "FinsROV/trajectory_tracking/average": 0.5,
        "FinsROV/trajectory_tracking/histogram": 0.5,
    }
