import math
from dataclasses import replace

from orbitforge.core.constants import MU_EARTH_KM3_S2
from orbitforge.core.state import CartesianState, TimeWindow
from orbitforge.core.vector import Vec3
from orbitforge.orbits.two_body import propagate_two_body
from orbitforge.conjunction.avoidance import (
    AvoidanceConstraints,
    ConjunctionAssessment,
    MissionLossModel,
    assessment_fingerprint,
    comparison_rows,
    make_plan,
    search_avoidance,
    validate_plan,
)


def _scenario(offset_z_km=0.02):
    # Build a real two-body close encounter: define the crossing geometry at
    # TCA (out-of-plane offset, 1.5 km/s crossing velocity) and back-propagate
    # the secondary to the planning epoch.
    r = 7000.0
    tca = 600.0
    v = math.sqrt(MU_EARTH_KM3_S2 / r)
    primary = CartesianState(0.0, Vec3(r, 0.0, 0.0), Vec3(0.0, v, 0.0))
    p_at_tca = propagate_two_body(primary, tca)
    secondary_at_tca = CartesianState(
        tca,
        p_at_tca.position_km + Vec3(0.0, 0.0, offset_z_km),
        p_at_tca.velocity_km_s + Vec3(1.5, 0.0, 0.0),
    )
    secondary = propagate_two_body(secondary_at_tca, -tca)
    assessment = ConjunctionAssessment(
        primary=primary,
        secondary=secondary,
        tca_tai_s=tca,
        plane_covariance=((0.09, 0.0), (0.0, 0.09)),
        hard_body_radius_km=0.05,
    )
    constraints = AvoidanceConstraints(
        window=TimeWindow(0.0, 300.0),
        max_delta_v_m_s=5.0,
        n_epoch_steps=2,
        n_direction_steps=4,
        cant_angles_deg=(0.0, 90.0),
        n_magnitude_steps=3,
        probability_samples=256,
    )
    return assessment, constraints


def test_planner_searches_and_recommends():
    assessment, constraints = _scenario()
    result = search_avoidance(assessment, constraints)

    # grid: 2 epochs x (4 in-plane + 1 normal) directions x 3 magnitudes
    assert len(result.candidates) == 2 * 5 * 3
    assert all(c.delta_v_m_s <= constraints.max_delta_v_m_s for c in result.candidates)
    assert all(abs(sum(x * x for x in c.direction_rtn) - 1.0) < 1e-9
               for c in result.candidates)

    # baseline is high risk, so the planner must find a burn, not just score
    assert result.baseline.collision_probability > constraints.max_probability
    assert result.status == 'maneuver_recommended'
    rec = result.recommended
    assert rec is not None and rec.feasible
    assert rec.collision_probability <= constraints.max_probability
    assert rec.collision_probability < result.baseline.collision_probability
    assert rec.miss_distance_km > result.baseline.miss_distance_km
    assert rec.propellant_fraction > 0.0


def test_maneuver_updates_tca_and_covariance():
    growth = ((1e-5, 0.0), (0.0, 1e-5))
    assessment, constraints = _scenario()
    growing = replace(assessment, covariance_growth_per_s=growth)

    # covariance follows the shifted TCA instead of being reused frozen
    base = assessment.covariance_at(assessment.tca_tai_s)
    moved = growing.covariance_at(assessment.tca_tai_s + 100.0)
    assert base == assessment.plane_covariance
    assert abs(moved[0][0] - (base[0][0] + 1e-3)) < 1e-12

    result = search_avoidance(growing, constraints)
    burned = [c for c in result.candidates if c.direction_rtn is not None]
    assert any(abs(c.tca_shift_s) > 1e-6 for c in burned)
    # every candidate was re-propagated to its own refined TCA
    assert all(abs(c.tca_tai_s - assessment.tca_tai_s - c.tca_shift_s) < 1e-9
               for c in burned)


def test_comparison_rows_rank_feasible_first():
    assessment, constraints = _scenario()
    result = search_avoidance(assessment, constraints)
    rows = comparison_rows(result)

    assert len(rows) == 1 + len(result.candidates)  # baseline + all burns
    ranked = [row for row in rows if row['rank'] is not None]
    assert ranked == rows[:len(ranked)]  # ranked block comes first
    assert ranked[0]['rank'] == 1
    scores = [row['rank_score'] for row in ranked]
    assert scores == sorted(scores)
    assert any(row['is_baseline'] for row in rows)
    for key in ('delta_v_m_s', 'collision_probability', 'miss_distance_km',
                'mission_cost', 'tca_shift_s', 'propellant_fraction'):
        assert all(key in row for row in rows)


def test_old_plan_invalidated_by_updated_assessment():
    assessment, constraints = _scenario()
    result = search_avoidance(assessment, constraints)
    plan = make_plan(result, assessment, generated_tai_s=100.0)

    ok = validate_plan(plan, assessment)
    assert ok.valid and ok.reasons == ()

    # a new CDM moves the TCA: plan is stale
    later_tca = replace(assessment, tca_tai_s=assessment.tca_tai_s + 5.0)
    stale = validate_plan(plan, later_tca)
    assert not stale.valid and 'tca_updated' in stale.reasons
    assert stale.details['tca_shift_s'] == 5.0

    # covariance update alone also invalidates
    new_cov = replace(assessment, plane_covariance=((0.12, 0.0), (0.0, 0.09)))
    stale_cov = validate_plan(plan, new_cov)
    assert not stale_cov.valid and 'covariance_updated' in stale_cov.reasons

    # a freshly published state vector (new epoch) invalidates too
    new_state = replace(assessment, primary=replace(assessment.primary, epoch_tai_s=60.0))
    stale_state = validate_plan(plan, new_state)
    assert not stale_state.valid and 'primary_state_epoch_updated' in stale_state.reasons

    # a refreshed state vector at the SAME epoch is still an update
    moved = replace(assessment, primary=replace(
        assessment.primary, position_km=Vec3(7000.001, 0.0, 0.0)))
    stale_moved = validate_plan(plan, moved)
    assert not stale_moved.valid and 'primary_state_updated' in stale_moved.reasons
    nudged = replace(assessment, secondary=replace(
        assessment.secondary, velocity_km_s=assessment.secondary.velocity_km_s + Vec3(1e-6, 0.0, 0.0)))
    stale_vel = validate_plan(plan, nudged)
    assert not stale_vel.valid and 'secondary_velocity_updated' in stale_vel.reasons

    # explicit tolerance can accept small TCA noise, but zero is the default
    assert validate_plan(plan, later_tca, tca_tolerance_s=10.0).valid
    assert assessment_fingerprint(assessment) != assessment_fingerprint(later_tca)
    assert assessment_fingerprint(assessment) == assessment_fingerprint(
        replace(assessment))


def test_window_after_tca_yields_no_maneuver():
    assessment, constraints = _scenario()
    late = replace(constraints, window=TimeWindow(700.0, 800.0))
    result = search_avoidance(assessment, late)

    assert result.candidates == ()
    assert result.recommended is None
    assert result.status == 'no_feasible_maneuver'
    assert any('no maneuver epoch' in r['reason'] for r in result.rejected)
    rows = comparison_rows(result)
    assert len(rows) == 1 and rows[0]['is_baseline']


def test_safe_baseline_means_no_maneuver():
    far_assessment, constraints = _scenario(offset_z_km=5.0)
    result = search_avoidance(far_assessment, constraints, MissionLossModel())

    assert result.baseline.feasible
    assert result.status == 'no_maneuver_required'
    assert result.recommended is None
