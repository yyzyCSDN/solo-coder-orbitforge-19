import math
from fastapi.testclient import TestClient
from orbitforge.api.app import app
from orbitforge.core.vector import Vec3
from orbitforge.core.state import CartesianState
from orbitforge.core.constants import MU_EARTH_KM3_S2
from orbitforge.orbits.elements import KeplerianElements,elements_to_state,state_to_elements
from orbitforge.orbits.two_body import propagate_two_body
from orbitforge.maneuvers.hohmann import hohmann
from orbitforge.coverage.footprint import footprint_radius_km
from orbitforge.power.battery import BatteryState
from orbitforge.link.modulation import choose_modcod
from orbitforge.mission.timeline import Timeline,Activity
from orbitforge.core.state import TimeWindow

def test_api_live_and_kepler():
    c=TestClient(app); assert c.get('/live').status_code==200; r=c.post('/v1/orbit/kepler',json={'mean_anomaly':1.0,'eccentricity':0.2}); assert r.status_code==200

def test_api_conjunction_avoidance():
    r=7000.0; v=math.sqrt(MU_EARTH_KM3_S2/r); tca=600.0
    p=CartesianState(0.0,Vec3(r,0,0),Vec3(0,v,0))
    pt=propagate_two_body(p,tca)
    s=propagate_two_body(CartesianState(tca,pt.position_km+Vec3(0,0,0.02),pt.velocity_km_s+Vec3(1.5,0,0)),-tca)
    req={'primary':[r,0,0,0,v,0],
         'secondary':list(s.position_km.as_tuple())+list(s.velocity_km_s.as_tuple()),
         'epoch_tai_s':0.0,'tca_tai_s':tca,
         'plane_covariance':[[0.09,0],[0,0.09]],'hard_body_radius_km':0.05,
         'window':[0,300],'n_epoch_steps':2,'n_direction_steps':4,
         'n_magnitude_steps':3,'probability_samples':256}
    c=TestClient(app); resp=c.post('/v1/conjunction/avoidance',json=req)
    assert resp.status_code==200
    data=resp.json()
    assert data['status']=='maneuver_recommended'
    assert data['recommended']['delta_v_m_s']<=5.0
    assert data['recommended']['collision_probability']<data['baseline']['collision_probability']
    assert data['fingerprint'] and len(data['candidates'])>1

def test_elements_roundtrip():
    el=KeplerianElements(7000,.01,.3,.4,.5,.6); r,v=elements_to_state(el); out=state_to_elements(r,v); assert abs(out.a_km-7000)<1e-6 and abs(out.e-.01)<1e-9

def test_hohmann_energy(): assert hohmann(7000,42164)['total_dv_km_s']>3

def test_coverage_positive(): assert footprint_radius_km(500,math.radians(10))>0

def test_battery_bounds():
    b=BatteryState(100,50); b.step(0,1000,3600); assert b.soc==0; b.step(1000,0,3600); assert b.soc==1

def test_modcod_selection(): assert choose_modcod(12)=='16QAM_3_4'

def test_timeline_conflict():
    t=Timeline(); t.add(Activity('a',TimeWindow(0,10),'cam',1))
    try: t.add(Activity('b',TimeWindow(9,11),'cam',2)); assert False
    except ValueError: pass
