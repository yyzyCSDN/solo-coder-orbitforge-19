import math

from orbitforge.core.vector import Vec3
from orbitforge.core.state import TimeWindow
from orbitforge.conjunction.encounter import encounter_frame
from orbitforge.conjunction.avoidance_planner import (
    AvoidanceConstraints,
    ConjunctionScenario,
    apply_constraints,
    default_directions,
    evaluate_maneuver,
    is_plan_current,
    plan_avoidance,
    position_velocity_covariance,
    revalidate_plan,
)

EPOCH = 100000.0
REL_SPEED = 7.5  # km/s, secondary approaches along -z
TIME_TO_TCA = 600.0  # s
BURNS = (100100.0, 100400.0)


def make_scenario(pos_sigma_km=0.05, execution_sigma_m_s=0.0,
                  window_start=BURNS[0], window_end=BURNS[1],
                  hard_radius_km=0.05, epoch=EPOCH,
                  range_to_go_km=None):
    if range_to_go_km is None:
        range_to_go_km = REL_SPEED * TIME_TO_TCA
    rel_r = Vec3(0.02, -0.03, range_to_go_km)
    rel_v = Vec3(0.0, 0.0, -REL_SPEED)
    cov = position_velocity_covariance((pos_sigma_km,) * 3)
    return ConjunctionScenario(
        epoch_tai_s=epoch,
        rel_position_km=rel_r,
        rel_velocity_km_s=rel_v,
        covariance=cov,
        hard_body_radius_km=hard_radius_km,
        burn_window=TimeWindow(window_start, window_end),
        execution_sigma_m_s=execution_sigma_m_s,
    )


def test_nominal_tca_and_geometry():
    scenario = make_scenario()
    assert abs(scenario.nominal_tca_tai_s - (EPOCH + TIME_TO_TCA)) < 1e-9
    baseline = evaluate_maneuver(scenario, None, Vec3(0, 0, 0), 'no_burn')
    assert abs(baseline.miss_distance_km - math.hypot(0.02, 0.03)) < 1e-9
    assert baseline.delta_v_magnitude_m_s == 0.0
    assert baseline.propellant_fraction == 0.0
    assert baseline.mission_loss_s == 0.0
    assert baseline.tca_tai_s == scenario.nominal_tca_tai_s


def test_invalid_covariance_and_window_rejected():
    try:
        ConjunctionScenario(
            epoch_tai_s=0.0,
            rel_position_km=Vec3(0, 0, 10),
            rel_velocity_km_s=Vec3(0, 0, -1),
            covariance=tuple(tuple(range(4)) for _ in range(4)),
            hard_body_radius_km=0.05,
            burn_window=TimeWindow(0, 10),
        )
    except ValueError:
        pass
    else:
        raise AssertionError('4x4 covariance must be rejected')

    scenario = make_scenario()
    try:
        evaluate_maneuver(scenario, scenario.burn_window.end_tai_s + 1.0,
                          Vec3(1, 0, 0))
    except ValueError:
        pass
    else:
        raise AssertionError('burn outside window must raise')


def test_bplane_pulse_reduces_probability():
    scenario = make_scenario()
    baseline = evaluate_maneuver(scenario, None, Vec3(0, 0, 0), 'no_burn')
    _, e_y, e_z = encounter_frame(scenario.rel_position_km, scenario.rel_velocity_km_s)
    # push away from the current miss direction
    push = (e_y * baseline.miss_bt_km + e_z * baseline.miss_br_km).unit() * 2.0
    option = evaluate_maneuver(scenario, EPOCH + 300.0, push)
    assert option.miss_distance_km > baseline.miss_distance_km
    assert option.collision_probability < baseline.collision_probability
    # 2 m/s applied 300 s before TCA -> about 0.6 km b-plane displacement
    assert option.miss_distance_km > 0.5
    # a purely b-plane pulse does not delay the encounter meaningfully
    assert abs(option.tca_tai_s - scenario.nominal_tca_tai_s) < 1.0
    assert option.mission_loss_s > 0.0
    assert option.propellant_fraction > 0.0


def test_along_track_pulse_shifts_tca():
    scenario = make_scenario()
    baseline = evaluate_maneuver(scenario, None, Vec3(0, 0, 0), 'no_burn')
    option = evaluate_maneuver(scenario, EPOCH + 300.0, Vec3(0, 0, 2.0))
    # +2 m/s along +z (against the approach) delays the encounter
    assert option.tca_tai_s > baseline.tca_tai_s
    assert option.mission_loss_s > 0.0


def test_planner_finds_feasible_recommendation():
    scenario = make_scenario()
    plan = plan_avoidance(scenario, magnitudes_m_s=(0.5, 1.0, 2.0, 3.0),
                          epoch_count=3,
                          constraints=AvoidanceConstraints(max_probability=1e-4,
                                                           min_miss_km=0.3))
    assert plan.baseline.collision_probability > 1e-4
    assert plan.recommended is not None
    assert plan.recommended.feasible
    assert plan.recommended.collision_probability <= 1e-4
    assert plan.recommended.miss_distance_km >= 0.3
    # the cheapest adequate pulse is preferred over stronger pulses
    feasible_pulses = [c for c in plan.feasible if c.delta_v_magnitude_m_s > 0]
    assert feasible_pulses
    assert plan.recommended.delta_v_magnitude_m_s <= feasible_pulses[-1].delta_v_magnitude_m_s + 1e-12
    # the Pareto front never contains a dominated candidate
    for i, a in enumerate(plan.pareto):
        for j, b in enumerate(plan.pareto):
            if i == j:
                continue
            no_worse = (b.collision_probability <= a.collision_probability
                        and b.delta_v_magnitude_m_s <= a.delta_v_magnitude_m_s
                        and b.mission_loss_s <= a.mission_loss_s)
            strictly = (b.collision_probability < a.collision_probability
                        or b.delta_v_magnitude_m_s < a.delta_v_magnitude_m_s
                        or b.mission_loss_s < a.mission_loss_s)
            assert not (no_worse and strictly)


def test_no_feasible_option_when_budget_too_tight():
    scenario = make_scenario()
    plan = plan_avoidance(scenario, magnitudes_m_s=(0.01, 0.05), epoch_count=2,
                          constraints=AvoidanceConstraints(max_probability=1e-6,
                                                           min_miss_km=5.0,
                                                           max_delta_v_m_s=0.1))
    assert plan.recommended is None
    assert plan.feasible == ()
    assert all(not c.feasible for c in plan.candidates)


def test_constraint_violations_are_listed():
    scenario = make_scenario()
    option = evaluate_maneuver(scenario, EPOCH + 300.0, Vec3(0, 0, 0.01), 'tiny')
    checked = apply_constraints(option, AvoidanceConstraints(
        max_probability=1e-6, min_miss_km=10.0, max_delta_v_m_s=0.001))
    assert not checked.feasible
    assert 'probability' in checked.violated
    assert 'miss_distance' in checked.violated
    assert 'delta_v' in checked.violated


def test_covariance_update_invalidates_old_plan():
    scenario = make_scenario()
    plan = plan_avoidance(scenario, magnitudes_m_s=(1.0, 2.0), epoch_count=2)
    assert is_plan_current(plan, scenario)

    updated = make_scenario(pos_sigma_km=0.08)  # new CDM covariance
    assert not is_plan_current(plan, updated)

    result = revalidate_plan(plan, updated)
    assert result['status'] == 'stale'
    assert result['option'] is not None
    # feasibility is recomputed against the updated covariance, never assumed
    assert isinstance(result['still_feasible'], bool)
    assert abs(result['probability_change']) > 0.0


def test_tca_update_repropagates_option():
    scenario = make_scenario()
    plan = plan_avoidance(scenario, magnitudes_m_s=(2.0,), epoch_count=2)
    # CDM update: both epoch and range-to-go change so the TCA moves 50 s earlier
    shifted = make_scenario(epoch=EPOCH + 50.0,
                            range_to_go_km=REL_SPEED * (TIME_TO_TCA - 100.0))
    assert abs(shifted.nominal_tca_tai_s - (EPOCH + TIME_TO_TCA - 50.0)) < 1e-9
    assert not is_plan_current(plan, shifted)
    result = revalidate_plan(plan, shifted)
    assert result['status'] == 'stale'
    assert abs(result['tca_change_s'] - (-50.0)) < 1e-3
    # the same absolute burn epoch now has less lead time: miss distance and
    # feasibility are recomputed rather than carried over from the old plan
    assert result['miss_change_km'] < 0.0
    assert result['still_feasible'] is False
    assert 'probability_change' in result


def test_burn_epoch_past_window_closes_revalidation():
    scenario = make_scenario()
    plan = plan_avoidance(scenario, magnitudes_m_s=(3.0,), epoch_count=2)
    recommended = plan.recommended
    assert recommended is not None
    assert recommended.burn_epoch_tai_s <= BURNS[1]
    # new maneuver window opens after the previously planned burn
    later_window = make_scenario(window_start=100500.0, window_end=100800.0)
    result = revalidate_plan(plan, later_window, recommended)
    assert result['status'] == 'window_closed'
    assert result['option'] is None


def test_execution_noise_raises_probability():
    ideal = make_scenario(execution_sigma_m_s=0.0)
    noisy = make_scenario(execution_sigma_m_s=0.5)
    _, e_y, _ = encounter_frame(ideal.rel_position_km, ideal.rel_velocity_km_s)
    a = evaluate_maneuver(ideal, EPOCH + 300.0, e_y * 2.0)
    b = evaluate_maneuver(noisy, EPOCH + 300.0, e_y * 2.0)
    assert b.collision_probability >= a.collision_probability


def test_default_directions_cover_bplane_both_sides():
    scenario = make_scenario()
    dirs = default_directions(scenario)
    _, e_y, e_z = encounter_frame(scenario.rel_position_km, scenario.rel_velocity_km_s)
    for axis in (e_y, e_y * -1.0, e_z, e_z * -1.0):
        assert any((d - axis).norm() < 1e-9 for d in dirs)
    # no duplicated rays
    for i, d in enumerate(dirs):
        for o in dirs[i + 1:]:
            assert d.dot(o) < 1.0 - 1e-9
