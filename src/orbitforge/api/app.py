from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from orbitforge.core.vector import Vec3
from orbitforge.core.state import CartesianState, TimeWindow
from orbitforge.orbits.kepler import solve_kepler_elliptic
from orbitforge.maneuvers.hohmann import hohmann
from orbitforge.link.budget import free_space_loss_db
from orbitforge.environment.eclipse import eclipse_state
from orbitforge.attitude.quaternion import Quaternion
from orbitforge.conjunction.avoidance import (
    AvoidanceConstraints, ConjunctionAssessment, MissionLossModel,
    assessment_fingerprint, comparison_rows, search_avoidance)
from orbitforge.storage.sqlite import Store
app = FastAPI(title='OrbitForge Mission Lab', version='1.0.0')
store = Store(':memory:')

class KeplerReq(BaseModel):
    mean_anomaly: float
    eccentricity: float

class HohmannReq(BaseModel):
    r1_km: float
    r2_km: float

class LinkReq(BaseModel):
    range_km: float
    freq_hz: float

class EclipseReq(BaseModel):
    sat: list[float]
    sun: list[float]

class RotateReq(BaseModel):
    q: list[float]
    v: list[float]

class AvoidanceReq(BaseModel):
    primary: list[float]      # [x, y, z, vx, vy, vz] km, km/s
    secondary: list[float]
    epoch_tai_s: float
    tca_tai_s: float
    plane_covariance: list[list[float]]
    hard_body_radius_km: float
    covariance_growth_per_s: list[list[float]] | None = None
    window: list[float]       # [start_tai_s, end_tai_s]
    max_delta_v_m_s: float = 5.0
    min_lead_time_s: float = 0.0
    n_epoch_steps: int = 3
    n_direction_steps: int = 8
    n_magnitude_steps: int = 4
    probability_samples: int = 512
    max_probability: float = 1e-4
    min_miss_km: float = 0.0

@app.get('/live')
def live():
    return {'status': 'live'}

@app.get('/ready')
def ready():
    return {'status': 'ready'}

@app.post('/v1/orbit/kepler')
def kepler(r: KeplerReq):
    return {'eccentric_anomaly': solve_kepler_elliptic(r.mean_anomaly, r.eccentricity)}

@app.post('/v1/maneuver/hohmann')
def h(r: HohmannReq):
    return hohmann(r.r1_km, r.r2_km)

@app.post('/v1/link/fspl')
def l(r: LinkReq):
    return {'loss_db': free_space_loss_db(r.range_km, r.freq_hz)}

@app.post('/v1/environment/eclipse')
def e(r: EclipseReq):
    return {'state': eclipse_state(Vec3(*r.sat), Vec3(*r.sun))}

@app.post('/v1/attitude/rotate')
def rotate(r: RotateReq):
    q = Quaternion(*r.q)
    v = q.rotate(Vec3(*r.v))
    return {'v': v.as_tuple()}

@app.post('/v1/conjunction/avoidance')
def plan_avoidance(r: AvoidanceReq):
    if len(r.primary) != 6 or len(r.secondary) != 6:
        raise HTTPException(400, 'primary/secondary must be [x,y,z,vx,vy,vz]')
    if len(r.window) != 2:
        raise HTTPException(400, 'window must be [start_tai_s, end_tai_s]')
    # default cant angles give 2 pure-normal + 4 * n_direction_steps directions
    grid_size = r.n_epoch_steps * (4 * r.n_direction_steps + 2) * r.n_magnitude_steps
    if grid_size > 20000:
        raise HTTPException(400, 'search grid too large')
    try:
        assessment = ConjunctionAssessment(
            primary=CartesianState(r.epoch_tai_s, Vec3(*r.primary[:3]), Vec3(*r.primary[3:])),
            secondary=CartesianState(r.epoch_tai_s, Vec3(*r.secondary[:3]), Vec3(*r.secondary[3:])),
            tca_tai_s=r.tca_tai_s,
            plane_covariance=tuple(tuple(row) for row in r.plane_covariance),
            hard_body_radius_km=r.hard_body_radius_km,
            covariance_growth_per_s=(tuple(tuple(row) for row in r.covariance_growth_per_s)
                                     if r.covariance_growth_per_s else ((0.0, 0.0), (0.0, 0.0))),
        )
        constraints = AvoidanceConstraints(
            window=TimeWindow(*r.window),
            max_delta_v_m_s=r.max_delta_v_m_s,
            min_lead_time_s=r.min_lead_time_s,
            n_epoch_steps=r.n_epoch_steps,
            n_direction_steps=r.n_direction_steps,
            n_magnitude_steps=r.n_magnitude_steps,
            probability_samples=r.probability_samples,
            max_probability=r.max_probability,
            min_miss_km=r.min_miss_km,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    result = search_avoidance(assessment, constraints, MissionLossModel())
    rows = comparison_rows(result)
    return {
        'status': result.status,
        'fingerprint': assessment_fingerprint(assessment),
        'recommended': next((row for row in rows
                             if result.recommended and row['name'] == result.recommended.name), None),
        'baseline': next(row for row in rows if row['is_baseline']),
        'candidates': rows,
        'rejected': list(result.rejected),
    }

@app.get('/v1/system/audit')
def audit():
    return store.audit_chain()

def main():
    import uvicorn
    uvicorn.run('orbitforge.api.app:app', host='127.0.0.1', port=8080)
