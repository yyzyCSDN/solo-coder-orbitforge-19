from __future__ import annotations
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

from orbitforge.core.vector import Vec3
from orbitforge.core.state import TimeWindow
from orbitforge.conjunction.encounter import encounter_frame, time_of_closest_approach
from orbitforge.conjunction.probability import collision_probability_2d
from orbitforge.maneuvers.impulsive import propellant_fraction

# A relative 6x6 covariance orders axes (x,y,z,vx,vy,vz), position in km,
# velocity in km/s. Position-only 3x3 covariances are accepted too.
KM_S_PER_M_S = 1.0 / 1000.0


@dataclass(frozen=True)
class ConjunctionScenario:
    """Relative geometry and uncertainty at ``epoch_tai_s``.

    ``rel_position_km`` / ``rel_velocity_km_s`` point from the secondary to the
    primary. ``covariance`` is the combined (primary + secondary) covariance.
    """

    epoch_tai_s: float
    rel_position_km: Vec3
    rel_velocity_km_s: Vec3
    covariance: tuple
    hard_body_radius_km: float
    burn_window: TimeWindow
    spacecraft_mass_kg: float = 1000.0
    isp_s: float = 220.0
    # 1-sigma execution error of a pulse, per axis (m/s)
    execution_sigma_m_s: float = 0.0
    pc_samples: int = 1600

    def __post_init__(self):
        n = len(self.covariance)
        if n not in (3, 6) or any(len(row) != n for row in self.covariance):
            raise ValueError('covariance must be a 3x3 or 6x6 square matrix')
        if self.hard_body_radius_km <= 0.0:
            raise ValueError('hard_body_radius_km must be positive')
        if self.rel_velocity_km_s.norm2() < 1e-18:
            raise ValueError('relative velocity too small to define encounter geometry')

    @property
    def nominal_tca_tai_s(self) -> float:
        tau = time_of_closest_approach(self.rel_position_km, self.rel_velocity_km_s)
        return self.epoch_tai_s + max(0.0, tau)


@dataclass(frozen=True)
class AvoidanceOption:
    name: str
    burn_epoch_tai_s: float | None
    delta_v_m_s: Vec3
    delta_v_magnitude_m_s: float
    tca_tai_s: float
    miss_distance_km: float
    miss_bt_km: float
    miss_br_km: float
    collision_probability: float
    propellant_fraction: float
    mission_loss_s: float
    along_track_offset_km: float
    covariance_2d: tuple
    feasible: bool
    violated: tuple = ()


@dataclass(frozen=True)
class AvoidanceConstraints:
    max_probability: float = 1e-4
    min_miss_km: float = 1.0
    max_delta_v_m_s: float = 10.0
    max_propellant_fraction: float = 1.0
    max_mission_loss_s: float = math.inf


@dataclass(frozen=True)
class AvoidancePlan:
    scenario_fingerprint: str
    baseline: AvoidanceOption
    candidates: tuple[AvoidanceOption, ...]
    feasible: tuple[AvoidanceOption, ...]
    pareto: tuple[AvoidanceOption, ...]
    recommended: AvoidanceOption | None
    constraints: AvoidanceConstraints
    weights: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Covariance helpers
# ---------------------------------------------------------------------------

def position_velocity_covariance(position_sigma_km, velocity_sigma_km_s=0.0):
    """Build a diagonal 6x6 covariance from axis-wise 1-sigma values."""
    ps = list(position_sigma_km)
    vs = list(velocity_sigma_km_s) if velocity_sigma_km_s else [0.0, 0.0, 0.0]
    if len(ps) != 3 or len(vs) != 3:
        raise ValueError('sigma vectors must have three components')
    matrix = [[0.0] * 6 for _ in range(6)]
    for i, sigma in enumerate(ps):
        matrix[i][i] = sigma * sigma
    for i, sigma in enumerate(vs):
        matrix[3 + i][3 + i] = sigma * sigma
    return tuple(tuple(row) for row in matrix)


def _ensure_6x6(cov):
    if len(cov) == 6:
        return cov
    out = [[0.0] * 6 for _ in range(6)]
    for i in range(3):
        for j in range(3):
            out[i][j] = cov[i][j]
    return tuple(tuple(row) for row in out)


def _propagate_relative_covariance(cov6, dt_s, execution_sigma_m_s):
    """Linear constant-velocity propagation; add burn execution noise."""
    c = [list(row) for row in cov6]
    out = [[0.0] * 6 for _ in range(6)]

    def block(rows, cols):
        return [[c[i][j] for j in cols] for i in rows]

    pp = block(range(3), range(3))
    pv = block(range(3), range(3, 6))
    vp = block(range(3, 6), range(3))
    vv = block(range(3, 6), range(3, 6))
    # P(t) = P + dt*(PV^T handled symmetrically) + dt^2 V ; cross terms
    # C_pp' = P + dt*(Pv + vP) + dt^2 V ; C_pv' = Pv + dt*V ; C_vv' = V
    for i in range(3):
        for j in range(3):
            out[i][j] = pp[i][j] + dt_s * (pv[i][j] + vp[j][i]) + dt_s * dt_s * vv[i][j]
            out[i][3 + j] = pv[i][j] + dt_s * vv[i][j]
            out[3 + i][j] = vp[i][j] + dt_s * vv[i][j]
            out[3 + i][3 + j] = vv[i][j]
    if execution_sigma_m_s > 0.0:
        s = (execution_sigma_m_s * KM_S_PER_M_S) ** 2
        for i in range(3):
            out[3 + i][3 + i] += s
    # symmetrize numerical noise
    for i in range(6):
        for j in range(i, 6):
            value = 0.5 * (out[i][j] + out[j][i])
            out[i][j] = value
            out[j][i] = value
    return tuple(tuple(row) for row in out)


def _project_covariance(cov6, e_y: Vec3, e_z: Vec3):
    basis = (e_y.as_tuple(), e_z.as_tuple())
    out = [[0.0, 0.0], [0.0, 0.0]]
    for a in range(2):
        for b in range(2):
            total = 0.0
            for i in range(3):
                for j in range(3):
                    total += basis[a][i] * cov6[i][j] * basis[b][j]
            out[a][b] = total
    return ((out[0][0], out[0][1]), (out[1][0], out[1][1]))


# ---------------------------------------------------------------------------
# Single-maneuver evaluation
# ---------------------------------------------------------------------------

def _propagate_relative(rel_r: Vec3, rel_v: Vec3, dt_s: float):
    return rel_r + rel_v * dt_s


def evaluate_maneuver(scenario: ConjunctionScenario, burn_epoch_tai_s,
                      delta_v_m_s: Vec3, name: str = 'maneuver') -> AvoidanceOption:
    """Apply a pulse in the allowed window, propagate to the new TCA and
    re-evaluate miss distance, collision probability, fuel and mission loss."""
    if delta_v_m_s is None:
        delta_v_m_s = Vec3(0.0, 0.0, 0.0)
    magnitude = delta_v_m_s.norm()

    if burn_epoch_tai_s is None or magnitude <= 0.0:
        burn_epoch_tai_s = None
        applied = Vec3(0.0, 0.0, 0.0)
    else:
        if not (scenario.burn_window.start_tai_s <= burn_epoch_tai_s
                <= scenario.burn_window.end_tai_s):
            raise ValueError('burn epoch outside allowed window')
        applied = delta_v_m_s * KM_S_PER_M_S

    cov6 = _ensure_6x6(scenario.covariance)
    t0 = scenario.epoch_tai_s
    rel_v0 = scenario.rel_velocity_km_s
    rel_r0 = scenario.rel_position_km

    if burn_epoch_tai_s is None:
        # No burn: geometry is the nominal one, but covariance is still
        # propagated to the current TCA so updates take effect.
        tau = time_of_closest_approach(rel_r0, rel_v0)
        rel_r_tca = rel_r0
        rel_v_after = rel_v0
        tca = t0 + max(0.0, tau)
        cov_tca = _propagate_relative_covariance(cov6, max(0.0, tau), 0.0)
    else:
        t_burn = burn_epoch_tai_s - t0
        r_at_burn = _propagate_relative(rel_r0, rel_v0, t_burn)
        cov_before = _propagate_relative_covariance(cov6, t_burn, 0.0)
        rel_v_after = rel_v0 + applied
        tau = time_of_closest_approach(r_at_burn, rel_v_after)
        tau = max(0.0, tau)
        rel_r_tca = _propagate_relative(r_at_burn, rel_v_after, tau)
        tca = t0 + t_burn + tau
        cov_tca = _propagate_relative_covariance(
            cov_before, tau, scenario.execution_sigma_m_s if magnitude > 0.0 else 0.0)

    _, e_y, e_z = encounter_frame(rel_r_tca, rel_v_after)
    miss_y = rel_r_tca.dot(e_y)
    miss_z = rel_r_tca.dot(e_z)
    miss = math.hypot(miss_y, miss_z)
    cov2 = _project_covariance(cov_tca, e_y, e_z)
    pc = collision_probability_2d(miss_y, miss_z, cov2,
                                  scenario.hard_body_radius_km, scenario.pc_samples)

    # Mission loss: timing error introduced at the original TCA, measured as
    # the along-track phase offset divided by the relative speed.
    nominal_tca = scenario.nominal_tca_tai_s
    if burn_epoch_tai_s is None:
        offset_km = 0.0
        mission_loss_s = 0.0
    else:
        t_burn = burn_epoch_tai_s - t0
        dt_orig = nominal_tca - t0
        r_at_burn = _propagate_relative(rel_r0, rel_v0, t_burn)
        r_perturbed = _propagate_relative(r_at_burn, rel_v_after, dt_orig - t_burn)
        r_nominal = _propagate_relative(rel_r0, rel_v0, dt_orig)
        residual = r_perturbed - r_nominal
        s_hat = rel_v_after.unit()
        along = abs(residual.dot(s_hat))
        radial2 = max(0.0, residual.dot(residual) - along * along)
        offset_km = math.sqrt(along * along + radial2)
        # offset km / relative speed (km/s) gives the timing error in seconds
        mission_loss_s = offset_km / rel_v_after.norm()

    propellant = (propellant_fraction(magnitude, scenario.isp_s)
                  if magnitude > 0.0 else 0.0)

    return AvoidanceOption(
        name=name,
        burn_epoch_tai_s=burn_epoch_tai_s,
        delta_v_m_s=delta_v_m_s,
        delta_v_magnitude_m_s=magnitude,
        tca_tai_s=tca,
        miss_distance_km=miss,
        miss_bt_km=miss_y,
        miss_br_km=miss_z,
        collision_probability=pc,
        propellant_fraction=propellant,
        mission_loss_s=mission_loss_s,
        along_track_offset_km=offset_km,
        covariance_2d=cov2,
        feasible=False,
    )


def apply_constraints(option: AvoidanceOption,
                      constraints: AvoidanceConstraints) -> AvoidanceOption:
    violated = []
    if option.collision_probability > constraints.max_probability:
        violated.append('probability')
    if option.miss_distance_km < constraints.min_miss_km:
        violated.append('miss_distance')
    if option.delta_v_magnitude_m_s > constraints.max_delta_v_m_s:
        violated.append('delta_v')
    if option.propellant_fraction > constraints.max_propellant_fraction:
        violated.append('propellant')
    if option.mission_loss_s > constraints.max_mission_loss_s:
        violated.append('mission_loss')
    return replace(option, feasible=not violated, violated=tuple(violated))


# ---------------------------------------------------------------------------
# Search over pulse directions, magnitudes and burn epochs
# ---------------------------------------------------------------------------

def default_directions(scenario: ConjunctionScenario) -> tuple[Vec3, ...]:
    """Unit search directions: b-plane axes, their diagonals, and inertial axes."""
    _, e_y, e_z = encounter_frame(scenario.rel_position_km, scenario.rel_velocity_km_s)
    diag1 = (e_y + e_z).unit()
    diag2 = (e_y - e_z).unit()
    axes = [
        e_y, e_y * -1.0, e_z, e_z * -1.0, diag1, diag1 * -1.0,
        diag2, diag2 * -1.0,
        Vec3(1, 0, 0), Vec3(-1, 0, 0),
        Vec3(0, 1, 0), Vec3(0, -1, 0),
        Vec3(0, 0, 1), Vec3(0, 0, -1),
    ]
    # de-duplicate identical rays, but keep antipodes (they push to opposite
    # sides of the b-plane)
    unique = []
    for axis in axes:
        if all(axis.dot(other) < 1.0 - 1e-9 for other in unique):
            unique.append(axis)
    return tuple(unique)


def _sample_epochs(window: TimeWindow, count: int):
    if count <= 1:
        return (window.start_tai_s,)
    step = window.duration_s / (count - 1)
    return tuple(window.start_tai_s + step * i for i in range(count))


def _pareto(candidates: Sequence[AvoidanceOption]) -> tuple[AvoidanceOption, ...]:
    """Pareto front minimizing pc, delta-v and mission loss."""
    front = []
    for i, a in enumerate(candidates):
        dominated = False
        for j, b in enumerate(candidates):
            if i == j:
                continue
            no_worse = (b.collision_probability <= a.collision_probability
                        and b.delta_v_magnitude_m_s <= a.delta_v_magnitude_m_s
                        and b.mission_loss_s <= a.mission_loss_s)
            strictly = (b.collision_probability < a.collision_probability
                        or b.delta_v_magnitude_m_s < a.delta_v_magnitude_m_s
                        or b.mission_loss_s < a.mission_loss_s)
            if no_worse and strictly:
                dominated = True
                break
        if not dominated:
            front.append(a)
    return tuple(front)


def scenario_fingerprint(scenario: ConjunctionScenario) -> str:
    """Hash of geometry, covariance and window. Any CDM update changes it."""
    payload = {
        'epoch': scenario.epoch_tai_s,
        'rel_r': scenario.rel_position_km.as_tuple(),
        'rel_v': scenario.rel_velocity_km_s.as_tuple(),
        'cov': [list(row) for row in scenario.covariance],
        'hard_radius': scenario.hard_body_radius_km,
        'window': [scenario.burn_window.start_tai_s, scenario.burn_window.end_tai_s],
        'execution_sigma': scenario.execution_sigma_m_s,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def plan_avoidance(scenario: ConjunctionScenario,
                   directions: Iterable[Vec3] | None = None,
                   magnitudes_m_s: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
                   epoch_count: int = 5,
                   constraints: AvoidanceConstraints | None = None,
                   probability_weight: float = 1.0,
                   fuel_weight: float = 1.0,
                   mission_weight: float = 0.01) -> AvoidancePlan:
    """Search pulse direction/magnitude/epoch inside the allowed window.

    Returns the no-burn baseline, all evaluated candidates, the feasible and
    Pareto-optimal subsets and a recommended option.
    """
    constraints = constraints or AvoidanceConstraints()
    dirs = tuple(directions) if directions is not None else default_directions(scenario)
    epochs = _sample_epochs(scenario.burn_window, max(1, epoch_count))

    baseline = apply_constraints(
        evaluate_maneuver(scenario, None, Vec3(0.0, 0.0, 0.0), 'no_burn'),
        constraints)

    candidates = [baseline]
    seen = set()
    for di, direction in enumerate(dirs):
        if direction.norm2() < 1e-18:
            raise ValueError('zero search direction')
        d = direction.unit()
        for magnitude in magnitudes_m_s:
            if magnitude <= 0.0:
                continue
            dv = d * magnitude
            for index, epoch in enumerate(epochs):
                name = f'burn_d{di}_m{magnitude:g}_t{index}'
                option = evaluate_maneuver(scenario, epoch, dv, name)
                option = apply_constraints(option, constraints)
                key = (round(option.tca_tai_s, 6),
                       round(option.miss_bt_km, 9),
                       round(option.miss_br_km, 9),
                       round(option.delta_v_magnitude_m_s, 9))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(option)

    feasible = [c for c in candidates if c.feasible]

    def score(c: AvoidanceOption) -> float:
        return (probability_weight * c.collision_probability * 1e6
                + fuel_weight * c.delta_v_magnitude_m_s
                + mission_weight * c.mission_loss_s)

    feasible.sort(key=lambda c: (score(c), -c.miss_distance_km))
    all_sorted = sorted(candidates, key=lambda c: (score(c), -c.miss_distance_km))
    pareto = tuple(sorted(_pareto(all_sorted), key=lambda c: (score(c), -c.miss_distance_km)))

    return AvoidancePlan(
        scenario_fingerprint=scenario_fingerprint(scenario),
        baseline=baseline,
        candidates=tuple(all_sorted),
        feasible=tuple(feasible),
        pareto=pareto,
        recommended=feasible[0] if feasible else None,
        constraints=constraints,
        weights={'probability': probability_weight,
                 'fuel': fuel_weight, 'mission': mission_weight},
    )


# ---------------------------------------------------------------------------
# Revalidation against updated covariance / TCA
# ---------------------------------------------------------------------------

def is_plan_current(plan: AvoidancePlan, scenario: ConjunctionScenario) -> bool:
    """Old plans are never assumed valid: fingerprints must match exactly."""
    return plan.scenario_fingerprint == scenario_fingerprint(scenario)


def revalidate_plan(plan: AvoidancePlan, scenario: ConjunctionScenario,
                    option: AvoidanceOption | None = None) -> dict:
    """Re-evaluate a stored option against the latest scenario.

    Returns ``status`` of ``unchanged`` (same fingerprint, option re-checked),
    ``window_closed`` (burn time no longer reachable) or ``stale`` (CDM update,
    option re-propagated through the new covariance/TCA and re-constrained).
    """
    option = option or plan.recommended
    if option is None:
        return {'status': 'no_option', 'option': None}

    fingerprint = scenario_fingerprint(scenario)
    unchanged = fingerprint == plan.scenario_fingerprint

    if option.burn_epoch_tai_s is not None and option.burn_epoch_tai_s > scenario.burn_window.end_tai_s:
        return {'status': 'window_closed', 'option': None,
                'scenario_fingerprint': fingerprint,
                'previous_fingerprint': plan.scenario_fingerprint}

    dv = option.delta_v_m_s
    epoch = option.burn_epoch_tai_s
    if epoch is not None and not (scenario.burn_window.start_tai_s <= epoch
                                  <= scenario.burn_window.end_tai_s):
        return {'status': 'window_closed', 'option': None,
                'scenario_fingerprint': fingerprint,
                'previous_fingerprint': plan.scenario_fingerprint}

    updated = apply_constraints(
        evaluate_maneuver(scenario, epoch, dv, option.name),
        plan.constraints)
    return {
        'status': 'unchanged' if unchanged else 'stale',
        'option': updated,
        'scenario_fingerprint': fingerprint,
        'previous_fingerprint': plan.scenario_fingerprint,
        'still_feasible': updated.feasible,
        'probability_change': updated.collision_probability - option.collision_probability,
        'miss_change_km': updated.miss_distance_km - option.miss_distance_km,
        'tca_change_s': updated.tca_tai_s - option.tca_tai_s,
    }
