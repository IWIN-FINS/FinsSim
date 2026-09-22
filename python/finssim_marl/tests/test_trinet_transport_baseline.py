import numpy as np

from finssim_marl.algorithms.trinet_transport_baseline import (
    PHASE_FINAL_ASCENT,
    PHASE_FIXED_INITIAL_FORMATION,
    PHASE_HOLD_STILL,
    PHASE_RAISE_NET,
    PHASE_TOW_TO_GOAL,
    TriNetTransportStateMachineBaseline,
    TriNetTransportStateMachineBaselineConfig,
)
from finssim_marl.envs.unity.wrapper import UnityTriNetCaptureEnvWrapper


def _team_observation(vertices, target, goal=None) -> np.ndarray:
    values = np.zeros((1, 3, 34), dtype=np.float32)
    goal = target if goal is None else np.asarray(goal, dtype=np.float32)
    for agent, position in enumerate(vertices):
        teammates = [other for index, other in enumerate(vertices) if index != agent]
        values[0, agent, 6:9] = target - position
        values[0, agent, 12] = position[1]
        values[0, agent, 14:17] = teammates[0] - position
        values[0, agent, 20:23] = teammates[1] - position
        values[0, agent, 26:29] = goal - position
    return values


def _baseline():
    return TriNetTransportStateMachineBaseline(
        n_agents=3,
        role_ids=np.zeros(3, dtype=np.int64),
        config=TriNetTransportStateMachineBaselineConfig(controller_backend="zero"),
    )


def test_fixed_initial_formation_uses_the_three_requested_target_relative_points():
    target = np.asarray((1.2, -0.4, 0.0), dtype=np.float32)
    vertices = np.asarray(
        ((-1.35, -0.2, 0.0), (-1.35, -0.9361216, -0.425), (-1.35, -0.9361216, 0.425)),
        dtype=np.float32,
    )
    baseline = _baseline()

    baseline.select_action(_team_observation(vertices, target), np.zeros(3, dtype=np.int64))

    expected_positions = target + np.asarray(baseline.config.fixed_initial_target_offsets, dtype=np.float32)
    expected_body_subgoals = expected_positions - vertices
    assert np.all(baseline.last_diagnostics["phase"] == 0)
    assert np.allclose(baseline.last_diagnostics["formation_waypoint_error_m"][0], expected_body_subgoals)
    assert np.allclose(baseline.last_diagnostics["formation_correction_m"], 0.0)


def test_all_rovs_settle_then_lift_tow_and_rise_together():
    target = np.asarray((1.2, -0.4, 0.0), dtype=np.float32)
    baseline = _baseline()
    vertices = target + np.asarray(baseline.config.fixed_initial_target_offsets, dtype=np.float32)
    goal = np.asarray((-1.25, -0.24, 0.0), dtype=np.float32)
    observation = _team_observation(vertices, target, goal)

    # Phase 1 cannot lift early even when every ROV is already at its own
    # requested point.
    for _ in range(29):
        baseline.select_action(observation, np.zeros(3, dtype=np.int64))
    assert np.all(baseline.last_diagnostics["phase"] == 0)

    baseline.select_action(observation, np.zeros(3, dtype=np.int64))

    assert np.all(baseline.last_diagnostics["phase"] == 2)
    assert np.allclose(baseline.last_diagnostics["body_position_error_m"][0, :, (0, 2)], 0.0)
    expected_lift = np.full(3, baseline.config.net_lift_height_m, dtype=np.float32)
    expected_lift[1] += baseline.config.top_extra_lift_height_m
    assert np.allclose(baseline.last_diagnostics["body_position_error_m"][0, :, 1], expected_lift)
    # Phase 2 measures the actual pool-local Y displacement from the height
    # recorded at entry; it does not integrate velocity.
    risen_observation = observation.copy()
    risen_observation[0, :, 12] += baseline.config.net_lift_height_m
    baseline.select_action(risen_observation, np.zeros(3, dtype=np.int64))
    assert np.allclose(
        baseline.last_diagnostics["body_position_error_m"][0, :, 1],
        (0.0, baseline.config.top_extra_lift_height_m, 0.0),
    )

    # Elapsed time does not authorize a retreat while the actual recorded-Y
    # lift targets have not been reached.
    for _ in range(100):
        baseline.select_action(observation, np.zeros(3, dtype=np.int64))
    assert np.all(baseline.last_diagnostics["phase"] == 2)

    reached_lift_observation = observation.copy()
    reached_lift_observation[0, :, 12] += expected_lift
    baseline.select_action(reached_lift_observation, np.zeros(3, dtype=np.int64))

    assert np.all(baseline.last_diagnostics["phase"] == 3)
    # Phase 3 is a full Goal retreat, not a partial 45% translation.
    assert np.all(baseline.last_diagnostics["body_position_error_m"][0, :, 0] == -1.0)

    # Target reaching Goal alone does not authorize the ascent: every ROV
    # must first reach its own configured Goal + offset Phase-3 waypoint.
    target_at_goal_but_rovs_away = _team_observation(vertices, goal, goal)
    baseline.select_action(target_at_goal_but_rovs_away, np.zeros(3, dtype=np.int64))
    assert np.all(baseline.last_diagnostics["phase"] == 3)

    # When all three are at those given Phase-3 points, transition directly to
    # Phase 4. All ROVs receive a pure equal relative ascent; no horizontal
    # return command is mixed in.
    tow_vertices = goal + np.asarray(baseline.config.tow_goal_target_offsets, dtype=np.float32)
    tow_observation = _team_observation(tow_vertices, goal, goal)
    baseline.select_action(tow_observation, np.zeros(3, dtype=np.int64))
    assert np.all(baseline.last_diagnostics["phase"] == 4)
    assert np.allclose(baseline.last_diagnostics["body_position_error_m"][0, :, (0, 2)], 0.0)
    assert np.allclose(
        baseline.last_diagnostics["body_position_error_m"][0, :, 1],
        baseline.config.final_lift_height_m,
    )

    # Reaching the recorded Phase-4 ascent height enters the final position
    # hold. It keeps the Goal + offset horizontal pose and neither continues
    # to ascend nor issues another tow command.
    final_height_observation = tow_observation.copy()
    final_height_observation[0, :, 12] += baseline.config.final_lift_height_m
    baseline.select_action(final_height_observation, np.zeros(3, dtype=np.int64))
    assert np.all(baseline.last_diagnostics["phase"] == PHASE_HOLD_STILL)
    assert np.allclose(baseline.last_diagnostics["body_position_error_m"][0], 0.0, atol=1e-6)


def test_task_frame_offsets_do_not_rotate_with_an_individual_rov_yaw():
    # Target->Goal observed in a ROV body frame after a +90 degree yaw. The
    # desired world Left offset (-0.05, -0.40, -0.30) must become the body
    # vector (+0.30, -0.40, -0.05), not remain a fixed body vector.
    body_offset = TriNetTransportStateMachineBaseline._task_offset_to_body(
        np.asarray((-0.05, -0.40, -0.30), dtype=np.float32),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
    )

    assert np.allclose(body_offset, (0.30, -0.40, -0.05), atol=1e-6)


def test_phase_three_goal_offsets_do_not_depend_on_target_position():
    """A Target crossing Goal must not flip the fixed recovery endpoints."""
    baseline = _baseline()
    goal = np.asarray((-1.25, -0.415, 0.0), dtype=np.float32)
    vertices = goal + np.asarray(baseline.config.tow_goal_target_offsets, dtype=np.float32)

    before_goal = _team_observation(vertices, np.asarray((-1.20, -0.415, 0.0), dtype=np.float32), goal)
    after_goal = _team_observation(vertices, np.asarray((-1.30, -0.415, 0.0), dtype=np.float32), goal)

    expected = np.asarray(baseline.config.tow_goal_target_offsets, dtype=np.float32)
    assert np.allclose(baseline._phase_offsets_in_body(before_goal, PHASE_TOW_TO_GOAL)[0], expected)
    assert np.allclose(baseline._phase_offsets_in_body(after_goal, PHASE_TOW_TO_GOAL)[0], expected)


def test_state_machine_holds_pool_yaw_zero_in_every_phase():
    baseline = TriNetTransportStateMachineBaseline(
        n_agents=3,
        role_ids=np.zeros(3, dtype=np.int64),
        config=TriNetTransportStateMachineBaselineConfig(
            controller_backend="zero",
            use_heading_control=True,
        ),
    )
    observation = _team_observation(
        np.asarray(((0.0, -0.4, -0.5), (0.0, -0.2, 0.0), (0.0, -0.4, 0.5)), dtype=np.float32),
        np.asarray((1.2, -0.4, 0.0), dtype=np.float32),
        np.asarray((-1.25, -0.415, 0.0), dtype=np.float32),
    )
    # Actor slot 13 is Unity's signed bearing of pool +X. The body PID uses
    # the opposite sign because body +Z is left while positive yaw turns
    # right, so its output still corrects toward pool/world yaw 0 degrees.
    observation[0, :, 13] = (15.0, -10.0, 5.0)
    baseline._ensure_state(1)

    for phase in (PHASE_FIXED_INITIAL_FORMATION, PHASE_RAISE_NET, PHASE_TOW_TO_GOAL, PHASE_FINAL_ASCENT, PHASE_HOLD_STILL):
        baseline._phase_by_env[:] = phase
        baseline.select_action(observation, np.zeros(3, dtype=np.int64))
        assert np.allclose(baseline.last_diagnostics["yaw_error_deg"][0], (-15.0, 10.0, -5.0))


def test_first_phase_waypoint_check_uses_each_rovs_yawed_body_frame():
    # The requested points remain correct when one ROV has a different yaw.
    target = np.asarray((1.2, -0.4, 0.0), dtype=np.float32)
    baseline = _baseline()
    vertices = target + np.asarray(baseline.config.fixed_initial_target_offsets, dtype=np.float32)
    goal = np.asarray((-1.25, -0.24, 0.0), dtype=np.float32)
    observation = _team_observation(vertices, target, goal)
    # Rotate every world-vector observation of agent 1 by 90 degrees.
    observation[0, 1, (6, 8)] = observation[0, 1, (8, 6)] * np.asarray((1.0, -1.0), dtype=np.float32)
    observation[0, 1, (14, 16)] = observation[0, 1, (16, 14)] * np.asarray((1.0, -1.0), dtype=np.float32)
    observation[0, 1, (20, 22)] = observation[0, 1, (22, 20)] * np.asarray((1.0, -1.0), dtype=np.float32)
    observation[0, 1, (26, 28)] = observation[0, 1, (28, 26)] * np.asarray((1.0, -1.0), dtype=np.float32)

    for _ in range(30):
        baseline.select_action(observation, np.zeros(3, dtype=np.int64))

    assert np.all(baseline.last_diagnostics["phase"] == 2)


def test_named_fins_rovs_have_the_fixed_left_top_right_action_order():
    agents = [
        "FinsROV_Fossen_Right?team=0&agent_id=1",
        "FinsROV_Fossen_Left?team=0&agent_id=9",
        "FinsROV_Fossen_Top?team=0&agent_id=3",
    ]

    ordered = sorted(agents, key=UnityTriNetCaptureEnvWrapper._trinet_agent_sort_key)

    assert ordered == [
        "FinsROV_Fossen_Left?team=0&agent_id=9",
        "FinsROV_Fossen_Top?team=0&agent_id=3",
        "FinsROV_Fossen_Right?team=0&agent_id=1",
    ]
