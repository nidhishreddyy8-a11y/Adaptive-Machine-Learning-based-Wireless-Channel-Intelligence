#!/usr/bin/env python3
"""
Adaptive Machine Learning Based Wireless Channel Intelligence
Full-Stack Flask Application Backend
Serves real-time inference, polar beam radar, 2D spatial coverage map,
and continuous mobility trajectory tracking.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = PROJECT_ROOT / 'models'
SCENARIO_PATH = PROJECT_ROOT / 'deepmimo_scenarios' / 'asu_campus_3p5'

app = Flask(__name__, template_folder='templates', static_folder='static')

# In-memory storage for models & metadata
MODELS = {}
METADATA = {}
BS_POS = np.array([166.0, 104.0, 22.0])  # Default ASU BS position

def load_system_artifacts():
    global MODELS, METADATA, BS_POS
    print("Loading ML models and metadata from:", MODELS_DIR)
    
    # Load metadata
    meta_path = MODELS_DIR / 'metadata.json'
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            METADATA = json.load(f)
            BS_POS = np.array(METADATA.get('bs_position', [166.0, 104.0, 22.0]))
            print(f"  Loaded metadata. BS position: {BS_POS}")
    else:
        print("  [WARN] metadata.json not found!")

    def load_model(name):
        path = MODELS_DIR / name
        if path.exists():
            return joblib.load(path)
        print(f"  [WARN] Model artifact not found: {name}")
        return None

    MODELS['beam_spatial'] = load_model('beam_spatial.pkl')
    MODELS['beam_rf']      = load_model('beam_rf.pkl')
    MODELS['rss_spatial']  = load_model('rss_spatial.pkl')
    MODELS['rss_gbm']      = load_model('rss_gbm.pkl')
    MODELS['los_spatial']  = load_model('los_spatial.pkl')
    MODELS['los_gbm']      = load_model('los_gbm.pkl')
    MODELS['ae']           = load_model('ae.pkl')
    
    print("[OK] All models successfully initialized in memory.")

# Load on startup
load_system_artifacts()

@app.route('/')
def index():
    return render_template('index.html', metadata=METADATA)

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'online',
        'models_loaded': {k: (v is not None) for k, v in MODELS.items()},
        'scenario': METADATA.get('scenario', 'ASU Campus 3.5 GHz'),
        'bs_position': BS_POS.tolist(),
        'n_codebook_beams': METADATA.get('n_codebook_beams', 64)
    })

@app.route('/api/dataset-stats')
def api_dataset_stats():
    return jsonify({
        'scenario': METADATA.get('scenario', 'ASU Campus 3.5 GHz (DeepMIMO)'),
        'carrier_frequency_ghz': METADATA.get('frequency_ghz', 3.5),
        'bandwidth_mhz': METADATA.get('bandwidth_mhz', 100.0),
        'bs_position': BS_POS.tolist(),
        'spatial_bounds': METADATA.get('bounds', {}),
        'codebook_size': METADATA.get('n_codebook_beams', 64),
        'training_samples': METADATA.get('n_training_samples', 15000),
        'feature_names': METADATA.get('feature_names', []),
        'feature_importances': METADATA.get('feature_importances', {})
    })

@app.route('/api/model-metrics')
def api_model_metrics():
    return jsonify(METADATA.get('metrics', {}))

@app.route('/api/map-data')
def api_map_data():
    points = METADATA.get('sample_coverage_points', [])
    bounds = METADATA.get('bounds', {})
    return jsonify({
        'bs_pos': BS_POS.tolist(),
        'bounds': bounds,
        'count': len(points),
        'points': points
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Run multi-task wireless channel inference.
    Input JSON: { x: float, y: float, z: float, [model_type: 'spatial'|'full'] }
    """
    try:
        data = request.get_json(force=True)
        x = float(data.get('x', 140.0))
        y = float(data.get('y', 90.0))
        z = float(data.get('z', 1.5))
        pos = np.array([x, y, z])

        # Geometry calculations
        dist_3d = float(np.linalg.norm(pos - BS_POS))
        dist_2d = float(np.linalg.norm(pos[:2] - BS_POS[:2]))
        height_diff = float(pos[2] - BS_POS[2])

        # Directional angles from BS to UE
        dx = pos[0] - BS_POS[0]
        dy = pos[1] - BS_POS[1]
        dz = pos[2] - BS_POS[2]
        aod_az_deg = float(np.rad2deg(np.arctan2(dy, dx)))
        aod_el_deg = float(np.rad2deg(np.arctan2(dz, dist_2d + 1e-9)))

        # 1. TASK 1: Beam Prediction
        # Scale coordinates for spatial model
        scaler_sp = MODELS['beam_spatial']['scaler']
        pos_scaled = scaler_sp.transform([[x, y, z]])
        
        beam_model = MODELS['beam_spatial']['model']
        pred_beam = int(beam_model.predict(pos_scaled)[0])
        
        # Get top-k beam candidates with likelihood
        proba = beam_model.predict_proba(pos_scaled)[0]
        classes = beam_model.classes_
        top3_indices = np.argsort(proba)[::-1][:3]
        top3_beams = [
            {'beam_id': int(classes[idx]), 'probability': round(float(proba[idx]), 4)}
            for idx in top3_indices
        ]

        # Calculate beam azimuth steering angle corresponding to predicted beam
        # Codebook maps azimuth -180 to 180 to 64 beams: beam = (aod + 180)/360 * 63
        beam_azimuth_deg = round(-180.0 + (pred_beam / 63.0) * 360.0, 2)

        # 2. TASK 2: RSS Estimation
        rss_model = MODELS['rss_spatial']['model']
        pred_rss_db = float(rss_model.predict(pos_scaled)[0])
        path_loss_db = round(-pred_rss_db, 2)

        # Estimate link quality / RSRP classification
        if pred_rss_db > -85.0:
            link_quality = 'Excellent'
            quality_color = '#00f0ff'
        elif pred_rss_db > -100.0:
            link_quality = 'Good'
            quality_color = '#22c55e'
        elif pred_rss_db > -115.0:
            link_quality = 'Moderate'
            quality_color = '#f59e0b'
        else:
            link_quality = 'Degraded / Cell-Edge'
            quality_color = '#ef4444'

        # 3. TASK 3: LOS / NLOS Detection
        los_model = MODELS['los_spatial']['model']
        pred_los = int(los_model.predict(pos_scaled)[0])
        los_proba = los_model.predict_proba(pos_scaled)[0]
        los_confidence = round(float(los_proba[1]) if len(los_proba) > 1 else float(pred_los), 4)

        # 4. TASK 4: Autoencoder Latent Projection & Feature Vector Reconstruction
        # Construct synthetic feature vector to pass through full autoencoder
        ae_scaler = MODELS['ae']['scaler']
        encoder   = MODELS['ae']['encoder']
        decoder   = MODELS['ae']['decoder']

        # Estimate synthetic 18 features from position & predicted values
        dom_delay_ns = round(dist_3d / 0.3, 2) # ~3.33 ns/meter propagation speed
        rms_del_ns   = round(12.0 + 8.0 * (1 - pred_los), 2)
        ang_sp_az    = round(0.4 + 0.3 * (1 - pred_los), 3)
        ang_sp_el    = round(0.15 + 0.1 * (1 - pred_los), 3)
        n_paths      = 7.0 if pred_los == 0 else 3.0

        feat_vector = np.array([[
            x, y, z, dist_3d, dist_2d, height_diff,
            pred_rss_db, pred_rss_db - 3.0, path_loss_db,
            rms_del_ns, ang_sp_az, ang_sp_el, n_paths,
            -aod_az_deg, -aod_el_deg, aod_az_deg, aod_el_deg, dom_delay_ns
        ]])

        feat_scaled = ae_scaler.transform(feat_vector)
        latent_4d = encoder.predict(feat_scaled)[0].tolist()
        recon_scaled = decoder.predict(np.array([latent_4d]))[0]
        recon_mse = float(np.mean((feat_scaled[0] - recon_scaled)**2))
        fidelity_snr = round(float(10 * np.log10(np.mean(feat_scaled[0]**2) / (recon_mse + 1e-12))), 2)

        return jsonify({
            'status': 'success',
            'ue_position': {'x': x, 'y': y, 'z': z},
            'bs_position': {'x': float(BS_POS[0]), 'y': float(BS_POS[1]), 'z': float(BS_POS[2])},
            'geometry': {
                'distance_3d_m': round(dist_3d, 2),
                'distance_2d_m': round(dist_2d, 2),
                'height_diff_m': round(height_diff, 2),
                'aod_azimuth_deg': round(aod_az_deg, 2),
                'aod_elevation_deg': round(aod_el_deg, 2)
            },
            'beam_prediction': {
                'optimal_beam_index': pred_beam,
                'beam_azimuth_deg': beam_azimuth_deg,
                'top_candidates': top3_beams,
                'total_beams': 64
            },
            'rss_estimation': {
                'estimated_rss_db': round(pred_rss_db, 2),
                'path_loss_db': path_loss_db,
                'link_quality': link_quality,
                'quality_color': quality_color
            },
            'los_detection': {
                'is_los': pred_los,
                'label': 'LOS' if pred_los == 1 else 'NLOS',
                'los_probability': los_confidence,
                'nlos_probability': round(1.0 - los_confidence, 4)
            },
            'autoencoder': {
                'latent_vector_4d': [round(v, 4) for v in latent_4d],
                'compression_ratio': '4.5x (18D -> 4D)',
                'reconstruction_mse': round(recon_mse, 4),
                'compression_snr_db': fidelity_snr
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/trajectory', methods=['POST'])
def api_trajectory():
    """
    Simulates a dynamic UE moving trajectory across the ASU campus,
    computing beam handovers, RSS fading variations, and LOS shadow transitions.
    """
    try:
        data = request.get_json(force=True) if request.is_json else {}
        route_type = data.get('route_type', 'campus_plaza')
        num_steps = int(data.get('steps', 40))

        # Define route endpoints based on ASU campus bounds
        # X: ~10 to ~300, Y: ~10 to ~250
        if route_type == 'courtyard_transit':
            p_start = np.array([50.0, 30.0, 1.5])
            p_end   = np.array([240.0, 180.0, 1.5])
            curve_amp = 30.0
        elif route_type == 'building_shadow':
            p_start = np.array([120.0, 40.0, 1.5])
            p_end   = np.array([210.0, 160.0, 1.5])
            curve_amp = -20.0
        else: # campus_plaza
            p_start = np.array([60.0, 150.0, 1.5])
            p_end   = np.array([270.0, 70.0, 1.5])
            curve_amp = 25.0

        # Generate smooth trajectory with Bezier / sin curve
        t_vals = np.linspace(0, 1, num_steps)
        waypoints = []
        
        scaler_sp = MODELS['beam_spatial']['scaler']
        beam_model = MODELS['beam_spatial']['model']
        rss_model = MODELS['rss_spatial']['model']
        los_model = MODELS['los_spatial']['model']

        handover_count = 0
        last_beam = None

        for idx, t in enumerate(t_vals):
            # Curved trajectory
            wx = float(p_start[0] + t * (p_end[0] - p_start[0]))
            wy = float(p_start[1] + t * (p_end[1] - p_start[1]) + curve_amp * np.sin(np.pi * t))
            wz = 1.5

            w_scaled = scaler_sp.transform([[wx, wy, wz]])
            beam_id = int(beam_model.predict(w_scaled)[0])
            rss_val = float(rss_model.predict(w_scaled)[0])
            los_val = int(los_model.predict(w_scaled)[0])

            is_handover = False
            if last_beam is not None and beam_id != last_beam:
                handover_count += 1
                is_handover = True
            last_beam = beam_id

            dist = float(np.linalg.norm(np.array([wx, wy, wz]) - BS_POS))
            beam_azimuth = round(-180.0 + (beam_id / 63.0) * 360.0, 1)

            waypoints.append({
                'step': idx,
                'x': round(wx, 2),
                'y': round(wy, 2),
                'z': wz,
                'distance_m': round(dist, 1),
                'beam_id': beam_id,
                'beam_azimuth_deg': beam_azimuth,
                'rss_db': round(rss_val, 2),
                'is_los': los_val,
                'is_handover': is_handover
            })

        return jsonify({
            'status': 'success',
            'route_type': route_type,
            'total_steps': num_steps,
            'total_handovers': handover_count,
            'waypoints': waypoints
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Adaptive Wireless Channel Intelligence server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
