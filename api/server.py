"""
Nahshon backend API.

Bridges the React frontend to the drift simulation + search planner so that
every run is computed LIVE from the operator's inputs (LKP lon/lat, victim
profile, time window) instead of reading a precomputed file.

Endpoints
---------
POST /api/simulate          body = incident JSON (the Incident Report form).
                            Starts the drift simulation in a background thread
                            and returns {"job_id": ...} immediately.
GET  /api/progress/<job_id> -> {"percent", "stage", "done", "error"} for the
                            loading bar (poll this).
GET  /api/drift_data        -> the latest drift_data.json (frames + lkp +
                            search_plan), produced by the most recent run.
GET  /api/plan?hour=H       -> recompute the coordinated search plan for the
                            heatmap hour currently shown in the UI (teams launch
                            from shore). Returns the search_plan dict.

Run:  python api/server.py     (defaults to http://localhost:5000)
"""

import os
import sys
import json
import uuid
import tempfile
import subprocess

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# make the simulation module importable (it lives in ../test)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'test'))

import sim_drowned_body as sim   # noqa: E402  (used for plan endpoint + paths)

SIM_SCRIPT = os.path.join(ROOT, 'test', 'sim_drowned_body.py')
JOB_DIR = os.path.join(tempfile.gettempdir(), 'nahshon_jobs')
os.makedirs(JOB_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

# job_id -> {"proc": Popen, "prog": progress-file path}
JOBS = {}


@app.post('/api/simulate')
def simulate():
    """Start the simulation as an ISOLATED SUBPROCESS (so Copernicus runs in a
    clean main thread and all NetCDF file handles are released when it exits)
    and stream progress through a per-job JSON file."""
    incident = request.get_json(silent=True) or {}
    job_id = uuid.uuid4().hex
    inc_path = os.path.join(JOB_DIR, f'inc_{job_id}.json')
    prog_path = os.path.join(JOB_DIR, f'prog_{job_id}.json')
    with open(inc_path, 'w', encoding='utf-8') as fh:
        json.dump(incident, fh)
    with open(prog_path, 'w', encoding='utf-8') as fh:
        json.dump({'percent': 0.0, 'stage': 'Queued', 'done': False,
                   'error': None}, fh)

    env = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    proc = subprocess.Popen(
        [sys.executable, '-u', SIM_SCRIPT,
         '--incident', inc_path, '--progress', prog_path],
        cwd=os.path.join(ROOT, 'test'), env=env)
    JOBS[job_id] = {'proc': proc, 'prog': prog_path}
    return jsonify(job_id=job_id)


@app.get('/api/progress/<job_id>')
def progress(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify(error='unknown job'), 404
    try:
        with open(job['prog'], encoding='utf-8') as fh:
            st = json.load(fh)
    except (OSError, json.JSONDecodeError):
        st = {'percent': 0.0, 'stage': 'Starting', 'done': False, 'error': None}
    # if the worker process died without reporting done, surface that
    if not st.get('done') and job['proc'].poll() is not None:
        code = job['proc'].returncode
        st = {**st, 'done': True,
              'error': st.get('error') or f'simulation process exited (code {code})'}
    return jsonify(st)


@app.post('/api/cancel/<job_id>')
def cancel(job_id):
    """Terminate a running simulation subprocess and mark the job cancelled."""
    job = JOBS.get(job_id)
    if job is None:
        return jsonify(error='unknown job'), 404
    proc = job['proc']
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    try:
        with open(job['prog'], 'w', encoding='utf-8') as fh:
            json.dump({'percent': 0.0, 'stage': 'Cancelled', 'done': True,
                       'error': 'cancelled'}, fh)
    except OSError:
        pass
    return jsonify(status='cancelled', job_id=job_id)


@app.get('/api/drift_data')
def drift_data():
    path = os.path.normpath(sim.APP_JSON)
    if not os.path.exists(path):
        return jsonify(error='no simulation has been run yet'), 404
    return send_file(path, mimetype='application/json')


@app.get('/api/plan')
def plan():
    """Recompute the search plan for a given forecast hour (UI slider)."""
    if not os.path.exists(sim.NCFILE):
        return jsonify(error='no simulation output; run a simulation first'), 404
    try:
        hour = int(request.args.get('hour', sim.PLAN_HOUR))
    except (TypeError, ValueError):
        hour = sim.PLAN_HOUR
    plan_dict, _res, _prob, _xe, _ye = sim.plan_search(sim.NCFILE, hour=hour)
    return jsonify(plan_dict)


@app.get('/api/health')
def health():
    return jsonify(status='ok', has_output=os.path.exists(sim.NCFILE))


if __name__ == '__main__':
    # threaded so progress polling works while a job runs in the background
    app.run(host='127.0.0.1', port=5000, threaded=True, debug=False)
