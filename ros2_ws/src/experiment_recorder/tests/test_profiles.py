from experiment_recorder.profiles import EXPERIMENTS, PROFILES, topics_for_profile


def test_all_paper_experiments_have_a_profile():
    assert set(EXPERIMENTS) == {"E1", "E2", "E4", "E5", "E6", "E7", "E9", "E10"}
    for experiment in EXPERIMENTS.values():
        assert experiment["profile"] in PROFILES


def test_profiles_are_unique_and_nonempty():
    for name in PROFILES:
        topics = topics_for_profile(name)
        assert topics
        assert len(topics) == len(set(topics))
        assert all(topic.startswith("/") for topic in topics)


def test_apriltag_profile_is_horizontal_only():
    topics = set(topics_for_profile("apriltag"))
    assert "/finsrov/pose" in topics
    assert "/finsrov/vision/status" in topics
    assert "/finsrov/hardware/depth_raw" not in topics
    assert "/finsrov/hardware/imu_raw" not in topics
    assert "/finsrov/hardware/motor_rpm_raw" not in topics
    assert "/finsrov/hardware/telemetry" not in topics


def test_t2_profile_records_the_controller_state_audited_by_completion():
    topics = set(topics_for_profile("t2"))
    assert "/finsrov/controller/state/status" in topics
    assert "/motion_controller/debug/trajectory_reference" in topics
