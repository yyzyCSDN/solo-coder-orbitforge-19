"""Collision avoidance maneuver planner.

Unlike ``maneuver_trade`` (which only scores externally supplied candidates),
this module searches for avoidance maneuvers itself: it samples maneuver
epochs inside an allowed execution window, searches impulse direction in the
RTN frame and impulse magnitude under the delta-v budget, propagates both
objects with two-body dynamics to the *updated* time of closest approach, and
re-evaluates miss distance, collision probability, fuel and mission loss.

The conjunction data (covariance and TCA) are not static: when an updated CDM
arrives, an existing plan must be revalidated. ``make_plan`` snapshots the
assessment fingerprint and ``validate_plan`` reports exactly why an old plan
can no longer be treated as valid.
"""
from __future__ import annotations
import hashlib
import json
import math
from dataclasses import dataclass, replace

from orbitforge.core.state import CartesianState, TimeWindow
from orbitforge.core.vector import Vec3
from orbitforge.maneuvers.impulsive import Impulse, apply_impulse, propellant_fraction
from orbitforge.orbits.two_body import propagate_two_body
from orbitforge.orbits.relative import hill_frame
from orbitforge.conjunction import maneuver_trade
from orbitforge.conjunction.bplane import bplane_coordinates, projected_miss_distance
from orbitforge.conjunction.covariance import determinant
from orbitforge.conjunction.encounter import time_of_closest_approach
from orbitforge.conjunction.probability import collision_probability_2d


@dataclass(frozen=True)
class ConjunctionAssessment:
    """Current conjunction picture (typically parsed from the latest CDM).

    ``plane_covariance`` is the combined 2x2 covariance (km^2) of the
    secondary-relative position projected onto the encounter plane at
    ``tca_tai_s``. ``covariance_growth_per_s`` (km^2/s) propagates that
    covariance to a shifted TCA, so an updated approach epoch also updates the
    covariance used for Pc instead of silently reusing the old one.
    """

    primary: CartesianState
    secondary: CartesianState
    tca_tai_s: float
    plane_covariance: tuple[tuple[float, float], tuple[float, float]]
    hard_body_radius_km: float
    covariance_growth_per_s: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0))

    def __post_init__(self):
        c = self.plane_covariance
        if len(c) != 2 or len(c[0]) != 2 or len(c[1]) != 2:
            raise ValueError('plane_covariance must be 2x2')
        if determinant(c) <= 0.0:
            raise ValueError('plane_covariance must be positive definite')
        if abs(c[0][1] - c[1][0]) > 1e-12:
            raise ValueError('plane_covariance must be symmetric')
        g = self.covariance_growth_per_s
        if len(g) != 2 or any(len(row) != 2 for row in g) or abs(g[0][1] - g[1][0]) > 1e-12:
            raise ValueError('covariance_growth_per_s must be a symmetric 2x2 matrix')
        if self.hard_body_radius_km <= 0.0:
            raise ValueError('hard_body_radius_km must be positive')

    def covariance_at(self, tca_tai_s: float):
        dt = tca_tai_s - self.tca_tai_s
        c = self.plane_covariance
        g = self.covariance_growth_per_s
        return (
            (c[0][0] + g[0][0] * dt, c[0][1] + g[0][1] * dt),
            (c[1][0] + g[1][0] * dt, c[1][1] + g[1][1] * dt),
        )


@dataclass(frozen=True)
class AvoidanceConstraints:
    """Search box and acceptance thresholds for an avoidance burn."""

    window: TimeWindow
    max_delta_v_m_s: float = 5.0
    min_lead_time_s: float = 0.0
    n_epoch_steps: int = 4
    n_direction_steps: int = 8
    cant_angles_deg: tuple[float, ...] = (-90.0, -45.0, 0.0, 45.0, 90.0)
    n_magnitude_steps: int = 5
    isp_s: float = 300.0
    probability_samples: int = 1024
    # acceptance gates
    max_probability: float = 1e-4
    min_miss_km: float = 0.0

    def __post_init__(self):
        if self.max_delta_v_m_s <= 0.0:
            raise ValueError('max_delta_v_m_s must be positive')
        if self.n_epoch_steps < 1 or self.n_direction_steps < 1 or self.n_magnitude_steps < 1:
            raise ValueError('grid step counts must be >= 1')
        if self.min_lead_time_s < 0.0:
            raise ValueError('min_lead_time_s must be non-negative')


@dataclass(frozen=True)
class MissionLossModel:
    """Linear mission-loss model (science/operations cost of a burn)."""

    cost_per_m_s: float = 0.1
    cost_per_maneuver: float = 1.0

    def evaluate(self, delta_v_m_s: float, is_baseline: bool):
        if is_baseline:
            return 0.0
        return self.cost_per_m_s * delta_v_m_s + self.cost_per_maneuver


@dataclass(frozen=True)
class EvaluatedCandidate:
    name: str
    is_baseline: bool
    maneuver_epoch_tai_s: float | None
    lead_time_s: float | None
    direction_rtn: tuple[float, float, float] | None
    delta_v_m_s: float
    propellant_fraction: float
    tca_tai_s: float
    tca_shift_s: float
    miss_distance_km: float
    bplane_km: tuple[float, float]
    collision_probability: float
    mission_cost: float
    feasible: bool
    rank_score: float | None = None


@dataclass(frozen=True)
class AvoidanceSearchResult:
    baseline: EvaluatedCandidate
    candidates: tuple[EvaluatedCandidate, ...]
    rejected: tuple[dict, ...]
    ranked: tuple[EvaluatedCandidate, ...]
    recommended: EvaluatedCandidate | None
    status: str


@dataclass(frozen=True)
class PlanValidity:
    valid: bool
    reasons: tuple[str, ...]
    details: dict


@dataclass(frozen=True)
class AvoidancePlan:
    recommended: EvaluatedCandidate | None
    baseline: EvaluatedCandidate
    status: str
    fingerprint: str
    generated_tai_s: float | None
    # assessment snapshot used for staleness checks
    tca_tai_s: float
    plane_covariance: tuple[tuple[float, float], tuple[float, float]]
    hard_body_radius_km: float
    primary_epoch_tai_s: float
    primary_position_km: tuple[float, float, float]
    primary_velocity_km_s: tuple[float, float, float]
    secondary_epoch_tai_s: float
    secondary_position_km: tuple[float, float, float]
    secondary_velocity_km_s: tuple[float, float, float]


# ---------------------------------------------------------------------------
# search space
# ---------------------------------------------------------------------------

def _epoch_grid(assessment: ConjunctionAssessment, constraints: AvoidanceConstraints):
    latest_epoch = assessment.tca_tai_s - constraints.min_lead_time_s
    start = max(constraints.window.start_tai_s,
                max(assessment.primary.epoch_tai_s, assessment.secondary.epoch_tai_s))
    end = min(constraints.window.end_tai_s, latest_epoch)
    if end < start:
        return []
    if constraints.n_epoch_steps == 1 or end == start:
        return [start]
    step = (end - start) / (constraints.n_epoch_steps - 1)
    return [start + i * step for i in range(constraints.n_epoch_steps)]


def _direction_rtn_grid(constraints: AvoidanceConstraints):
    directions: list[tuple[float, float, float]] = []
    for cant in constraints.cant_angles_deg:
        phi = math.radians(cant)
        cp, sp = math.cos(phi), math.sin(phi)
        # pure +/- orbit normal: in-plane angle is irrelevant, add once
        angles = (0.0,) if abs(abs(phi) - math.pi / 2) < 1e-12 else \
            [2.0 * math.pi * i / constraints.n_direction_steps
             for i in range(constraints.n_direction_steps)]
        for theta in angles:
            directions.append((cp * math.cos(theta), cp * math.sin(theta), sp))
    return directions


def _magnitude_grid(constraints: AvoidanceConstraints):
    n = constraints.n_magnitude_steps
    hi = constraints.max_delta_v_m_s
    return [hi * (i + 1) / n for i in range(n)]


# ---------------------------------------------------------------------------
# propagation / evaluation
# ---------------------------------------------------------------------------

def _refine_tca(primary: CartesianState, secondary: CartesianState, t_guess: float,
                tol_s: float = 1e-3, max_iters: int = 5):
    """Refine TCA by linear correction on two-body propagated states."""
    t = t_guess
    p1 = s1 = None
    for _ in range(max_iters):
        p1 = propagate_two_body(primary, t - primary.epoch_tai_s)
        s1 = propagate_two_body(secondary, t - secondary.epoch_tai_s)
        rel_r = p1.position_km - s1.position_km
        rel_v = p1.velocity_km_s - s1.velocity_km_s
        dt = time_of_closest_approach(rel_r, rel_v)
        t += dt
        if abs(dt) < tol_s:
            break
    return t, p1, s1


def _evaluate(assessment: ConjunctionAssessment, name: str,
              maneuver_epoch_tai_s: float | None, direction_rtn,
              delta_v_m_s: float, constraints: AvoidanceConstraints,
              loss_model: MissionLossModel, is_baseline: bool) -> EvaluatedCandidate:
    primary = assessment.primary
    secondary = assessment.secondary

    if is_baseline:
        post_burn_primary = primary
        lead = None
    else:
        p_at = propagate_two_body(primary, maneuver_epoch_tai_s - primary.epoch_tai_s)
        radial, along, normal = hill_frame(p_at.position_km, p_at.velocity_km_s)
        d_rad, d_along, d_norm = direction_rtn
        direction_eci = radial * d_rad + along * d_along + normal * d_norm
        impulse = Impulse(maneuver_epoch_tai_s,
                          direction_eci * (delta_v_m_s / 1000.0))  # m/s -> km/s
        post_burn_primary = apply_impulse(p_at, impulse)
        lead = assessment.tca_tai_s - maneuver_epoch_tai_s

    # TCA moves because the primary trajectory changed; re-solve it.
    new_tca, p_tca, s_tca = _refine_tca(post_burn_primary, secondary, assessment.tca_tai_s)
    rel_r = p_tca.position_km - s_tca.position_km
    rel_v = p_tca.velocity_km_s - s_tca.velocity_km_s
    bt, br = bplane_coordinates(rel_r, rel_v)
    miss = projected_miss_distance(rel_r, rel_v)

    # covariance is propagated to the *updated* TCA, not reused from nominal.
    cov = assessment.covariance_at(new_tca)
    if determinant(cov) <= 0.0:
        raise ValueError('covariance at updated TCA is not positive definite')

    pc = collision_probability_2d(bt, br, cov, assessment.hard_body_radius_km,
                                  samples=constraints.probability_samples)
    mission_cost = loss_model.evaluate(delta_v_m_s, is_baseline)
    prop_frac = 0.0 if is_baseline else propellant_fraction(delta_v_m_s, constraints.isp_s)
    feasible = (
        delta_v_m_s <= constraints.max_delta_v_m_s
        and pc <= constraints.max_probability
        and miss >= constraints.min_miss_km
    )

    return EvaluatedCandidate(
        name=name,
        is_baseline=is_baseline,
        maneuver_epoch_tai_s=maneuver_epoch_tai_s,
        lead_time_s=lead,
        direction_rtn=direction_rtn,
        delta_v_m_s=delta_v_m_s,
        propellant_fraction=prop_frac,
        tca_tai_s=new_tca,
        tca_shift_s=new_tca - assessment.tca_tai_s,
        miss_distance_km=miss,
        bplane_km=(bt, br),
        collision_probability=pc,
        mission_cost=mission_cost,
        feasible=feasible,
    )


def _rank_score(c: EvaluatedCandidate, probability_weight: float,
                fuel_weight: float, mission_weight: float) -> float:
    # same expression maneuver_trade.rank uses, kept numeric for comparison
    return (probability_weight * c.collision_probability * 1e6
            + fuel_weight * c.delta_v_m_s
            + mission_weight * c.mission_cost)


def search_avoidance(assessment: ConjunctionAssessment,
                     constraints: AvoidanceConstraints,
                     loss_model: MissionLossModel | None = None,
                     probability_weight: float = 1.0,
                     fuel_weight: float = 0.1,
                     mission_weight: float = 0.5) -> AvoidanceSearchResult:
    """Search the maneuver window/direction/magnitude space and rank results."""
    loss_model = loss_model or MissionLossModel()

    baseline = _evaluate(assessment, 'baseline', None, None, 0.0,
                         constraints, loss_model, is_baseline=True)

    candidates: list[EvaluatedCandidate] = []
    rejected: list[dict] = []
    epochs = _epoch_grid(assessment, constraints)
    directions = _direction_rtn_grid(constraints)
    magnitudes = _magnitude_grid(constraints)

    if not epochs:
        rejected.append({'reason': 'no maneuver epoch in window before TCA',
                         'window_start': constraints.window.start_tai_s,
                         'window_end': constraints.window.end_tai_s,
                         'tca_tai_s': assessment.tca_tai_s})

    for ei, epoch in enumerate(epochs):
        for di, direction in enumerate(directions):
            for mi, mag in enumerate(magnitudes):
                name = f'E{ei:02d}_D{di:02d}_M{mi:02d}'
                try:
                    candidates.append(_evaluate(
                        assessment, name, epoch, direction, mag,
                        constraints, loss_model, is_baseline=False))
                except (ValueError, ArithmeticError) as exc:
                    rejected.append({'name': name, 'epoch_tai_s': epoch,
                                     'direction_rtn': direction, 'delta_v_m_s': mag,
                                     'reason': str(exc)})

    # reuse the existing scorer for feasibility gates and ordering
    def to_trade(c: EvaluatedCandidate):
        return maneuver_trade.AvoidanceCandidate(
            name=c.name, delta_v_m_s=c.delta_v_m_s,
            miss_distance_km=c.miss_distance_km,
            collision_probability=c.collision_probability,
            mission_cost=c.mission_cost)

    scored = {c.name: c for c in [baseline, *candidates]}
    trades = maneuver_trade.feasible(
        [to_trade(c) for c in scored.values()],
        constraints.max_delta_v_m_s, constraints.max_probability,
        constraints.min_miss_km)
    ranked_trades = maneuver_trade.rank(
        trades, probability_weight=probability_weight,
        fuel_weight=fuel_weight, mission_weight=mission_weight)

    ranked: list[EvaluatedCandidate] = []
    for tr in ranked_trades:
        ranked.append(replace(scored[tr.name],
                              rank_score=_rank_score(scored[tr.name], probability_weight,
                                                     fuel_weight, mission_weight)))
    ranked_tuple = tuple(ranked)

    if baseline.feasible:
        status = 'no_maneuver_required'
        recommended = None
    else:
        maneuver_ranked = [c for c in ranked if not c.is_baseline]
        if maneuver_ranked:
            status = 'maneuver_recommended'
            recommended = maneuver_ranked[0]
        else:
            status = 'no_feasible_maneuver'
            recommended = None

    return AvoidanceSearchResult(
        baseline=baseline,
        candidates=tuple(candidates),
        rejected=tuple(rejected),
        ranked=ranked_tuple,
        recommended=recommended,
        status=status,
    )


def comparison_rows(result: AvoidanceSearchResult) -> list[dict]:
    """Tabular candidate comparison: ranked feasible first, then the rest."""
    rows: list[dict] = []
    ranked_names: set[str] = set()
    for rank, c in enumerate(result.ranked, start=1):
        ranked_names.add(c.name)
        rows.append(_row(c, rank))
    rest = [result.baseline, *result.candidates]
    rest = [c for c in rest if c.name not in ranked_names]
    rest.sort(key=lambda c: (c.collision_probability, c.delta_v_m_s))
    for c in rest:
        rows.append(_row(c, None))
    return rows


def _row(c: EvaluatedCandidate, rank: int | None) -> dict:
    return {
        'rank': rank,
        'name': c.name,
        'is_baseline': c.is_baseline,
        'maneuver_epoch_tai_s': c.maneuver_epoch_tai_s,
        'lead_time_s': c.lead_time_s,
        'direction_rtn': c.direction_rtn,
        'delta_v_m_s': c.delta_v_m_s,
        'propellant_fraction': c.propellant_fraction,
        'tca_tai_s': c.tca_tai_s,
        'tca_shift_s': c.tca_shift_s,
        'miss_distance_km': c.miss_distance_km,
        'collision_probability': c.collision_probability,
        'mission_cost': c.mission_cost,
        'feasible': c.feasible,
        'rank_score': c.rank_score,
    }


# ---------------------------------------------------------------------------
# plan binding and revalidation
# ---------------------------------------------------------------------------

def assessment_fingerprint(assessment: ConjunctionAssessment) -> str:
    payload = {
        'tca_tai_s': assessment.tca_tai_s,
        'plane_covariance': assessment.plane_covariance,
        'covariance_growth_per_s': assessment.covariance_growth_per_s,
        'hard_body_radius_km': assessment.hard_body_radius_km,
        'primary_epoch_tai_s': assessment.primary.epoch_tai_s,
        'primary_position_km': assessment.primary.position_km.as_tuple(),
        'primary_velocity_km_s': assessment.primary.velocity_km_s.as_tuple(),
        'secondary_epoch_tai_s': assessment.secondary.epoch_tai_s,
        'secondary_position_km': assessment.secondary.position_km.as_tuple(),
        'secondary_velocity_km_s': assessment.secondary.velocity_km_s.as_tuple(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def make_plan(result: AvoidanceSearchResult, assessment: ConjunctionAssessment,
              generated_tai_s: float | None = None) -> AvoidancePlan:
    return AvoidancePlan(
        recommended=result.recommended,
        baseline=result.baseline,
        status=result.status,
        fingerprint=assessment_fingerprint(assessment),
        generated_tai_s=generated_tai_s,
        tca_tai_s=assessment.tca_tai_s,
        plane_covariance=assessment.plane_covariance,
        hard_body_radius_km=assessment.hard_body_radius_km,
        primary_epoch_tai_s=assessment.primary.epoch_tai_s,
        primary_position_km=assessment.primary.position_km.as_tuple(),
        primary_velocity_km_s=assessment.primary.velocity_km_s.as_tuple(),
        secondary_epoch_tai_s=assessment.secondary.epoch_tai_s,
        secondary_position_km=assessment.secondary.position_km.as_tuple(),
        secondary_velocity_km_s=assessment.secondary.velocity_km_s.as_tuple(),
    )


def validate_plan(plan: AvoidancePlan, assessment: ConjunctionAssessment,
                  tca_tolerance_s: float = 0.0,
                  covariance_tolerance_km2: float = 0.0,
                  epoch_tolerance_s: float = 0.0,
                  state_tolerance_km: float = 0.0,
                  velocity_tolerance_km_s: float = 0.0,
                  radius_tolerance_km: float = 0.0) -> PlanValidity:
    """Check whether a plan still applies to the latest assessment.

    Default tolerances are zero on purpose: any CDM update that moves the TCA,
    the covariance or a state vector invalidates the plan, and the maneuver
    search must be re-run.
    """
    reasons: list[str] = []
    details: dict = {}

    tca_shift = abs(assessment.tca_tai_s - plan.tca_tai_s)
    details['tca_shift_s'] = tca_shift
    if tca_shift > tca_tolerance_s:
        reasons.append('tca_updated')

    cov_delta = max(abs(assessment.plane_covariance[i][j] - plan.plane_covariance[i][j])
                    for i in range(2) for j in range(2))
    details['covariance_max_delta_km2'] = cov_delta
    if cov_delta > covariance_tolerance_km2:
        reasons.append('covariance_updated')

    radius_delta = abs(assessment.hard_body_radius_km - plan.hard_body_radius_km)
    details['hard_body_radius_delta_km'] = radius_delta
    if radius_delta > radius_tolerance_km:
        reasons.append('hard_body_radius_updated')

    for label, state, plan_epoch, plan_pos, plan_vel in (
        ('primary', assessment.primary, plan.primary_epoch_tai_s,
         plan.primary_position_km, plan.primary_velocity_km_s),
        ('secondary', assessment.secondary, plan.secondary_epoch_tai_s,
         plan.secondary_position_km, plan.secondary_velocity_km_s),
    ):
        # an epoch change means a new state vector was published
        epoch_delta = abs(state.epoch_tai_s - plan_epoch)
        details[f'{label}_epoch_delta_s'] = epoch_delta
        if epoch_delta > epoch_tolerance_s:
            reasons.append(f'{label}_state_epoch_updated')
        # a refreshed estimate at the same epoch is still an update
        pos_delta = (state.position_km - Vec3(*plan_pos)).norm()
        details[f'{label}_position_delta_km'] = pos_delta
        if pos_delta > state_tolerance_km:
            reasons.append(f'{label}_state_updated')
        vel_delta = (state.velocity_km_s - Vec3(*plan_vel)).norm()
        details[f'{label}_velocity_delta_km_s'] = vel_delta
        if vel_delta > velocity_tolerance_km_s:
            reasons.append(f'{label}_velocity_updated')

    return PlanValidity(valid=not reasons, reasons=tuple(reasons), details=details)
