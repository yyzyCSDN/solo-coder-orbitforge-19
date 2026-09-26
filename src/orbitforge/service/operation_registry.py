from __future__ import annotations

OPERATIONS = {
    'time.convert': 'Convert between UTC, TAI, TT and GPS time scales',
    'frames.transform': 'Transform vectors between inertial and Earth-fixed frames',
    'orbit.propagate': 'Propagate Cartesian orbital states',
    'maneuver.lambert': 'Solve two-point transfer boundary conditions',
    'visibility.passes': 'Find station visibility passes',
    'environment.eclipse': 'Classify sunlight and eclipse state',
    'link.budget': 'Evaluate radio-frequency link margin',
    'attitude.pointing': 'Evaluate spacecraft pointing geometry',
    'conjunction.screen': 'Screen close approaches and collision risk',
    'conjunction.avoid': 'Search and rank collision avoidance maneuvers',
    'ephemeris.interpolate': 'Interpolate position and velocity ephemerides',
    'mission.windows': 'Combine mission opportunity windows',
    'coverage.evaluate': 'Evaluate constellation coverage and revisit',
    'power.simulate': 'Propagate energy storage through mission modes',
}

def describe():
    return [{'name': name, 'description': description} for name, description in sorted(OPERATIONS.items())]

def exists(name):
    return name in OPERATIONS

def search(text):
    needle = text.lower()
    return [row for row in describe() if needle in row['name'].lower() or needle in row['description'].lower()]
