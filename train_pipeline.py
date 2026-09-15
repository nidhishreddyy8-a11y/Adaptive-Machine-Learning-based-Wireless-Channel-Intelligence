#!/usr/bin/env python3
"""
Adaptive Machine Learning Based Wireless Channel Intelligence
End-to-End Training Pipeline & Model Serialization
Dataset: ASU Campus 3.5 GHz (DeepMIMO v3)
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA
import joblib

# Set plot style
plt.rcParams.update({
    'figure.facecolor': '#0a0d1a',
    'axes.facecolor': '#111827',
    'axes.edgecolor': '#2d3748',
    'axes.labelcolor': '#e2e8f0',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'text.color': '#e2e8f0',
    'grid.color': '#1e293b',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'font.family': 'sans-serif',
    'font.size': 11,
})

CYAN   = '#00d4ff'
PURPLE = '#a855f7'
GREEN  = '#22c55e'
ORANGE = '#f97316'
RED    = '#ef4444'

def get_scenario_path():
    candidates = [
        Path('deepmimo_scenarios/asu_campus_3p5'),
        Path('../deepmimo_scenarios/asu_campus_3p5'),
        Path('c:/Users/HP/OneDrive/Desktop/Adaptive ML/deepmimo_scenarios/asu_campus_3p5'),
    ]
    for c in candidates:
        if c.exists() and (c / 'params.json').exists():
            return c.resolve()
    raise FileNotFoundError("Could not locate ASU Campus DeepMIMO scenario directory.")

def get_output_dirs():
    root = Path(__file__).resolve().parent
    models_dir = root / 'models'
    static_img_dir = root / 'app' / 'static' / 'img'
    models_dir.mkdir(parents=True, exist_ok=True)
    static_img_dir.mkdir(parents=True, exist_ok=True)
    return root, models_dir, static_img_dir

def load_mat(scenario_path, filename):
    path = scenario_path / filename
    if not path.exists():
        print(f"  [WARN] File not found: {filename}")
        return None
    mat = sio.loadmat(str(path))
    keys = [k for k in mat.keys() if not k.startswith('_')]
    return mat[keys[0]] if keys else None

def rms_delay_spread(delay_arr, power_lin_arr):
    eps = 1e-12
    total_p = power_lin_arr.sum(axis=1, keepdims=True) + eps
    mean_d = (delay_arr * power_lin_arr / total_p).sum(axis=1)
    rms = np.sqrt(((delay_arr - mean_d[:, None])**2 * power_lin_arr / total_p).sum(axis=1))
    return np.nan_to_num(rms, nan=0.0)

def angular_spread(ang_deg, power_lin_arr):
    ang_rad = np.deg2rad(np.nan_to_num(ang_deg, nan=0.0))
    eps = 1e-12
    total_p = power_lin_arr.sum(axis=1, keepdims=True) + eps
    ms = (np.sin(ang_rad) * power_lin_arr / total_p).sum(axis=1)
    mc = (np.cos(ang_rad) * power_lin_arr / total_p).sum(axis=1)
    val = np.sqrt(-2 * np.log(np.clip(np.sqrt(ms**2 + mc**2), eps, 1.0)))
    return np.nan_to_num(val, nan=0.0)

def compute_beam_index(aod_az_arr, n_beams=64):
    dom = np.nan_to_num(aod_az_arr[:, 0], nan=0.0)
    beam = np.round((dom + 180.0) / 360.0 * (n_beams - 1)).astype(int)
    return np.clip(beam, 0, n_beams - 1)

def top_k_acc(model, Xts, y_true, k=3):
    proba = model.predict_proba(Xts)
    classes = model.classes_
    topk_idx = np.argsort(proba, axis=1)[:, -k:]
    topk_classes = classes[topk_idx]
    return float(np.mean([y_true[i] in topk_classes[i] for i in range(len(y_true))]))

def main():
    start_total = time.time()
    print("=" * 65)
    print("  ADAPTIVE WIRELESS CHANNEL INTELLIGENCE — TRAINING PIPELINE")
    print("=" * 65)

    scenario_path = get_scenario_path()
    root_dir, models_dir, static_img_dir = get_output_dirs()
    print(f"Scenario Path: {scenario_path}")
    print(f"Models Dir:    {models_dir}")
    print(f"Static Img:    {static_img_dir}")

    # Load parameters
    with open(scenario_path / 'params.json') as f:
        scenario_params = json.load(f)

    print("\n--- [Step 1/6] Ingesting DeepMIMO Scenario Data ---")
    rx_pos = load_mat(scenario_path, 'rx_pos_t001_tx000_r000.mat')
    tx_pos = load_mat(scenario_path, 'tx_pos_t001_tx000_r000.mat')
    power  = load_mat(scenario_path, 'power_t001_tx000_r000.mat')
    phase  = load_mat(scenario_path, 'phase_t001_tx000_r000.mat')
    delay  = load_mat(scenario_path, 'delay_t001_tx000_r000.mat')
    aoa_az = load_mat(scenario_path, 'aoa_az_t001_tx000_r000.mat')
    aoa_el = load_mat(scenario_path, 'aoa_el_t001_tx000_r000.mat')
    aod_az = load_mat(scenario_path, 'aod_az_t001_tx000_r000.mat')
    aod_el = load_mat(scenario_path, 'aod_el_t001_tx000_r000.mat')
    inter  = load_mat(scenario_path, 'inter_t001_tx000_r000.mat')

    if tx_pos.ndim == 2:
        tx_pos = tx_pos[0]

    N_total = rx_pos.shape[0]
    print(f"  Raw UE Points:    {N_total:,}")
    print(f"  Base Station Pos: x={tx_pos[0]:.2f}, y={tx_pos[1]:.2f}, z={tx_pos[2]:.2f} m")

    noise_floor = -250.0
    valid_mask = (power[:, 0] != 0) & (~np.isnan(power[:, 0])) & (power[:, 0] > noise_floor + 10)
    valid_indices = np.where(valid_mask)[0]
    print(f"  Active points with channel data: {len(valid_indices):,} ({(len(valid_indices)/N_total)*100:.1f}%)")

    # Sample representative subset for robust training and fast inference
    N_SAMPLE = 15000
    rng = np.random.RandomState(42)
    selected_idx = np.sort(rng.choice(valid_indices, min(N_SAMPLE, len(valid_indices)), replace=False))
    
    rx_pos = rx_pos[selected_idx]
    power  = power[selected_idx]
    phase  = phase[selected_idx]
    delay  = delay[selected_idx]
    aoa_az = aoa_az[selected_idx]
    aoa_el = aoa_el[selected_idx]
    aod_az = aod_az[selected_idx]
    aod_el = aod_el[selected_idx]
    inter  = inter[selected_idx]
    N = len(selected_idx)
    print(f"  Training dataset size: {N:,} UE coordinates")

    print("\n--- [Step 2/6] Feature Engineering & Channel Characterization ---")
    valid_paths = (power != 0) & (~np.isnan(power))
    power_db = np.where(valid_paths, power, noise_floor)
    power_lin = 10 ** (power_db / 10.0)

    delay_clean = np.where(valid_paths, np.nan_to_num(delay, nan=0.0), 0.0)
    aoa_az_clean = np.where(valid_paths, np.nan_to_num(aoa_az, nan=0.0), 0.0)
    aoa_el_clean = np.where(valid_paths, np.nan_to_num(aoa_el, nan=0.0), 0.0)
    aod_az_clean = np.where(valid_paths, np.nan_to_num(aod_az, nan=0.0), 0.0)
    aod_el_clean = np.where(valid_paths, np.nan_to_num(aod_el, nan=0.0), 0.0)

    dist_3d = np.linalg.norm(rx_pos - tx_pos, axis=1)
    dist_2d = np.linalg.norm(rx_pos[:, :2] - tx_pos[:2], axis=1)
    height_diff = rx_pos[:, 2] - tx_pos[2]
    total_power_db = 10 * np.log10(power_lin.sum(axis=1) + 1e-15)
    dom_power_db = power_db[:, 0]
    path_loss_db = -dom_power_db
    rms_del_ns = rms_delay_spread(delay_clean, power_lin) * 1e9
    ang_sp_az = angular_spread(aoa_az_clean, power_lin)
    ang_sp_el = angular_spread(aoa_el_clean, power_lin)
    n_sig_paths = (power_db > noise_floor + 10).sum(axis=1).astype(float)
    dom_aoa_az = aoa_az_clean[:, 0]
    dom_aoa_el = aoa_el_clean[:, 0]
    dom_aod_az = aod_az_clean[:, 0]
    dom_aod_el = aod_el_clean[:, 0]
    dom_delay_ns = delay_clean[:, 0] * 1e9

    feature_names = [
        'x_pos', 'y_pos', 'z_pos', 'dist_3d', 'dist_2d', 'height_diff',
        'total_power_db', 'dom_power_db', 'path_loss_db',
        'rms_delay_ns', 'ang_spread_az', 'ang_spread_el', 'n_paths',
        'dom_aoa_az', 'dom_aoa_el', 'dom_aod_az', 'dom_aod_el', 'dom_delay_ns'
    ]

    X = np.column_stack([
        rx_pos[:, 0], rx_pos[:, 1], rx_pos[:, 2],
        dist_3d, dist_2d, height_diff,
        total_power_db, dom_power_db, path_loss_db,
        rms_del_ns, ang_sp_az, ang_sp_el, n_sig_paths,
        dom_aoa_az, dom_aoa_el, dom_aod_az, dom_aod_el, dom_delay_ns
    ])
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)

    N_BEAMS = 64
    y_beam = compute_beam_index(aod_az_clean, n_beams=N_BEAMS)
    y_rss = total_power_db
    y_los = (np.nan_to_num(inter[:, 0], nan=1) == 0).astype(int)

    print(f"  Feature Matrix X shape: {X.shape}")
    print(f"  Beams present:         {len(np.unique(y_beam))} unique DFT beams")
    print(f"  RSS range:             [{y_rss.min():.1f}, {y_rss.max():.1f}] dB")
    print(f"  LOS link ratio:        {y_los.mean()*100:.1f}%")

    # Generate EDA Overview Plot
    print("\n  Generating EDA Overview Plot...")
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle('EDA — ASU Campus 3.5 GHz Wireless Channel Statistics', fontsize=14, fontweight='bold', color='#e2e8f0', y=0.99)
    
    # 1. Coverage Map
    sc1 = axes[0, 0].scatter(rx_pos[:, 0], rx_pos[:, 1], c=dom_power_db, cmap='plasma', s=3, alpha=0.7)
    axes[0, 0].scatter(tx_pos[0], tx_pos[1], c='red', s=200, marker='^', label='BS (Tx)')
    axes[0, 0].set_title('Spatial Coverage Map (Dominant Power)')
    axes[0, 0].set_xlabel('X (m)'); axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].legend(facecolor='#111827', labelcolor='#e2e8f0')
    plt.colorbar(sc1, ax=axes[0, 0], label='Power (dB)')

    # 2. Path Count Distribution
    axes[0, 1].hist(n_sig_paths, bins=10, color=PURPLE, alpha=0.85)
    axes[0, 1].axvline(n_sig_paths.mean(), color=CYAN, lw=2, ls='--', label=f'Mean={n_sig_paths.mean():.1f}')
    axes[0, 1].set_title('Multipath Richness per UE')
    axes[0, 1].set_xlabel('Active Multipath Rays'); axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend(facecolor='#111827', labelcolor='#e2e8f0')

    # 3. Path Loss vs 3D Distance
    sc3 = axes[0, 2].scatter(dist_3d, path_loss_db, c=y_los, cmap='coolwarm', s=3, alpha=0.6)
    axes[0, 2].set_title('Path Loss vs. 3D Distance (LOS vs NLOS)')
    axes[0, 2].set_xlabel('3D Distance (m)'); axes[0, 2].set_ylabel('Path Loss (dB)')
    plt.colorbar(sc3, ax=axes[0, 2], label='1=LOS, 0=NLOS')

    # 4. AoA Azimuth Distribution
    axes[1, 0].hist(dom_aoa_az, bins=50, color=CYAN, alpha=0.85)
    axes[1, 0].set_title('Dominant AoA Azimuth Distribution')
    axes[1, 0].set_xlabel('AoA Azimuth (deg)'); axes[1, 0].set_ylabel('Count')

    # 5. AoD vs AoA
    sc5 = axes[1, 1].scatter(dom_aod_az, dom_aoa_az, c=dist_3d, cmap='viridis', s=3, alpha=0.6)
    axes[1, 1].set_title('AoD Azimuth vs AoA Azimuth')
    axes[1, 1].set_xlabel('AoD Azimuth (deg)'); axes[1, 1].set_ylabel('AoA Azimuth (deg)')
    plt.colorbar(sc5, ax=axes[1, 1], label='Distance (m)')

    # 6. RMS Delay Spread
    axes[1, 2].hist(rms_del_ns, bins=50, color=GREEN, alpha=0.85)
    axes[1, 2].set_title('RMS Delay Spread Distribution')
    axes[1, 2].set_xlabel('RMS Delay Spread (ns)'); axes[1, 2].set_ylabel('Count')

    plt.tight_layout()
    fig.savefig(root_dir / 'eda_overview.png', dpi=130, facecolor='#0a0d1a')
    fig.savefig(static_img_dir / 'eda_overview.png', dpi=130, facecolor='#0a0d1a')
    plt.close(fig)

    # Train / Test Splitting
    X_train, X_test, yb_train, yb_test, yr_train, yr_test, yl_train, yl_test = train_test_split(
        X, y_beam, y_rss, y_los, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Spatial-only scaler for fast location-only inference
    scaler_spatial = StandardScaler()
    X_train_pos = X_train[:, :3]
    X_test_pos  = X_test[:, :3]
    X_train_pos_s = scaler_spatial.fit_transform(X_train_pos)
    X_test_pos_s  = scaler_spatial.transform(X_test_pos)

    print(f"  Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    metrics_summary = {}

    # ==============================================================
    # TASK 1: BEAM INDEX PREDICTION
    # ==============================================================
    print("\n--- [Step 3/6] Task 1: Optimal Beam Index Prediction (64-Beam Codebook) ---")
    print("  Training Random Forest Beam Classifier (n_estimators=100, max_depth=16)...")
    rf_beam = RandomForestClassifier(n_estimators=100, max_depth=16, n_jobs=-1, random_state=42)
    rf_beam.fit(X_train_s, yb_train)
    yb_pred_rf = rf_beam.predict(X_test_s)
    acc_rf_top1 = accuracy_score(yb_test, yb_pred_rf)
    acc_rf_top3 = top_k_acc(rf_beam, X_test_s, yb_test, k=3)
    print(f"    RF  Top-1 Acc: {acc_rf_top1*100:.2f}% | Top-3 Acc: {acc_rf_top3*100:.2f}%")

    print("  Training MLP Beam Classifier (hidden=(128, 64), max_iter=150)...")
    mlp_beam = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=150, early_stopping=True, random_state=42)
    mlp_beam.fit(X_train_s, yb_train)
    yb_pred_mlp = mlp_beam.predict(X_test_s)
    acc_mlp_top1 = accuracy_score(yb_test, yb_pred_mlp)
    acc_mlp_top3 = top_k_acc(mlp_beam, X_test_s, yb_test, k=3)
    print(f"    MLP Top-1 Acc: {acc_mlp_top1*100:.2f}% | Top-3 Acc: {acc_mlp_top3*100:.2f}%")

    print("  Training Spatial RF Beam Predictor (x,y,z input)...")
    rf_beam_pos = RandomForestClassifier(n_estimators=80, max_depth=14, n_jobs=-1, random_state=42)
    rf_beam_pos.fit(X_train_pos_s, yb_train)
    acc_rf_pos = accuracy_score(yb_test, rf_beam_pos.predict(X_test_pos_s))
    print(f"    Spatial RF Accuracy: {acc_rf_pos*100:.2f}%")

    metrics_summary['task1_beam'] = {
        'rf_top1': float(acc_rf_top1),
        'rf_top3': float(acc_rf_top3),
        'mlp_top1': float(acc_mlp_top1),
        'mlp_top3': float(acc_mlp_top3),
        'rf_spatial': float(acc_rf_pos),
        'best_model': 'Random Forest' if acc_rf_top1 >= acc_mlp_top1 else 'MLP'
    }

    # Save Task 1 Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Task 1 — Optimal Beam Index Prediction', fontsize=13, fontweight='bold', color='#e2e8f0')
    
    bars = axes[0].bar(['RF (Top-1)', 'RF (Top-3)', 'MLP (Top-1)', 'MLP (Top-3)'],
                       [acc_rf_top1*100, acc_rf_top3*100, acc_mlp_top1*100, acc_mlp_top3*100],
                       color=[CYAN, PURPLE, GREEN, ORANGE], edgecolor='none')
    for b in bars:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{b.get_height():.1f}%", ha='center', color='#e2e8f0', fontweight='bold')
    axes[0].set_ylim(0, 105); axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Beam Codebook Accuracy')

    u_b, c_b = np.unique(y_beam, return_counts=True)
    axes[1].bar(u_b, c_b, color=CYAN, edgecolor='none')
    axes[1].set_title('DFT Codebook Beam Usage Frequency')
    axes[1].set_xlabel('Beam Index (0-63)'); axes[1].set_ylabel('Count')

    axes[2].scatter(yb_test[:500], yb_pred_rf[:500], alpha=0.4, s=12, color=CYAN, linewidths=0)
    axes[2].plot([0, N_BEAMS-1], [0, N_BEAMS-1], '--', color=ORANGE, lw=1.5, label='Perfect')
    axes[2].set_title('Predicted vs True Beam Index (500 Test Points)')
    axes[2].set_xlabel('True Beam'); axes[2].set_ylabel('Predicted Beam')
    axes[2].legend(facecolor='#111827', labelcolor='#e2e8f0')

    plt.tight_layout()
    fig.savefig(root_dir / 'beam_results.png', dpi=130, facecolor='#0a0d1a')
    fig.savefig(static_img_dir / 'beam_results.png', dpi=130, facecolor='#0a0d1a')
    plt.close(fig)

    # ==============================================================
    # TASK 2: RECEIVED SIGNAL STRENGTH (RSS) ESTIMATION
    # ==============================================================
    print("\n--- [Step 4/6] Task 2: Received Signal Strength (RSS) Estimation ---")
    print("  Training Gradient Boosting Regressor...")
    gbm_rss = GradientBoostingRegressor(n_estimators=120, max_depth=5, learning_rate=0.1, random_state=42)
    gbm_rss.fit(X_train_s, yr_train)
    yr_pred_gbm = gbm_rss.predict(X_test_s)
    rmse_gbm = np.sqrt(mean_squared_error(yr_test, yr_pred_gbm))
    mae_gbm  = mean_absolute_error(yr_test, yr_pred_gbm)
    r2_gbm   = r2_score(yr_test, yr_pred_gbm)
    print(f"    GBM  RMSE: {rmse_gbm:.3f} dB | MAE: {mae_gbm:.3f} dB | R²: {r2_gbm:.4f}")

    print("  Training MLP Regressor...")
    mlp_rss = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', max_iter=200, early_stopping=True, random_state=42)
    mlp_rss.fit(X_train_s, yr_train)
    yr_pred_mlp = mlp_rss.predict(X_test_s)
    rmse_mlp = np.sqrt(mean_squared_error(yr_test, yr_pred_mlp))
    mae_mlp  = mean_absolute_error(yr_test, yr_pred_mlp)
    r2_mlp   = r2_score(yr_test, yr_pred_mlp)
    print(f"    MLP  RMSE: {rmse_mlp:.3f} dB | MAE: {mae_mlp:.3f} dB | R²: {r2_mlp:.4f}")

    gbm_rss_pos = GradientBoostingRegressor(n_estimators=80, max_depth=5, random_state=42)
    gbm_rss_pos.fit(X_train_pos_s, yr_train)

    metrics_summary['task2_rss'] = {
        'gbm_rmse': float(rmse_gbm), 'gbm_mae': float(mae_gbm), 'gbm_r2': float(r2_gbm),
        'mlp_rmse': float(rmse_mlp), 'mlp_mae': float(mae_mlp), 'mlp_r2': float(r2_mlp),
        'best_model': 'GBM' if rmse_gbm <= rmse_mlp else 'MLP'
    }

    # Save Task 2 Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Task 2 — RSS Estimation Results', fontsize=13, fontweight='bold', color='#e2e8f0')

    axes[0].scatter(yr_test, yr_pred_gbm, alpha=0.3, s=8, color=PURPLE, linewidths=0)
    lo, hi = yr_test.min(), yr_test.max()
    axes[0].plot([lo, hi], [lo, hi], '--', color=ORANGE, lw=2, label='Ideal 1:1')
    axes[0].set_title(f'Predicted vs True RSS (R² = {r2_gbm:.3f})')
    axes[0].set_xlabel('True Total Power (dB)'); axes[0].set_ylabel('Predicted Power (dB)')
    axes[0].legend(facecolor='#111827', labelcolor='#e2e8f0')

    residuals = yr_test - yr_pred_gbm
    axes[1].hist(residuals, bins=50, color=CYAN, alpha=0.85)
    axes[1].axvline(0, color=ORANGE, lw=2, ls='--')
    axes[1].axvline(residuals.mean(), color=GREEN, lw=1.5, ls='--', label=f'Mean={residuals.mean():.2f} dB')
    axes[1].set_title('Estimation Residual Distribution')
    axes[1].set_xlabel('Error (dB)'); axes[1].set_ylabel('Count')
    axes[1].legend(facecolor='#111827', labelcolor='#e2e8f0')

    axes[2].bar(['GBM', 'MLP'], [rmse_gbm, rmse_mlp], color=[CYAN, GREEN], width=0.35)
    axes[2].set_title('Model RMSE Comparison')
    axes[2].set_ylabel('RMSE (dB)')
    for idx_b, val in enumerate([rmse_gbm, rmse_mlp]):
        axes[2].text(idx_b, val + 0.05, f"{val:.3f} dB", ha='center', color='#e2e8f0', fontweight='bold')

    plt.tight_layout()
    fig.savefig(root_dir / 'rss_results.png', dpi=130, facecolor='#0a0d1a')
    fig.savefig(static_img_dir / 'rss_results.png', dpi=130, facecolor='#0a0d1a')
    plt.close(fig)

    # ==============================================================
    # TASK 3: LOS / NLOS CLASSIFICATION
    # ==============================================================
    print("\n--- [Step 5/6] Task 3: LOS / NLOS Channel Detection ---")
    los_idx  = np.where(yl_train == 1)[0]
    nlos_idx = np.where(yl_train == 0)[0]
    n_bal = min(len(los_idx), len(nlos_idx))
    rng_los = np.random.RandomState(42)
    bal_sel = np.concatenate([
        rng_los.choice(los_idx, n_bal, replace=False),
        rng_los.choice(nlos_idx, n_bal, replace=False)
    ])
    X_bal_s = X_train_s[bal_sel]
    yl_bal  = yl_train[bal_sel]
    print(f"  Balanced Training Set: {len(X_bal_s):,} samples (50% LOS, 50% NLOS)")

    print("  Training Logistic Regression...")
    lr_los = LogisticRegression(max_iter=1000, random_state=42)
    lr_los.fit(X_bal_s, yl_bal)
    yl_pred_lr = lr_los.predict(X_test_s)
    acc_lr = accuracy_score(yl_test, yl_pred_lr)
    print(f"    LR  Accuracy: {acc_lr*100:.2f}%")

    print("  Training Gradient Boosting Classifier...")
    gbm_los = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    gbm_los.fit(X_bal_s, yl_bal)
    yl_pred_gbm = gbm_los.predict(X_test_s)
    acc_gbm = accuracy_score(yl_test, yl_pred_gbm)
    print(f"    GBM Accuracy: {acc_gbm*100:.2f}%")

    gbm_los_pos = GradientBoostingClassifier(n_estimators=70, max_depth=4, random_state=42)
    gbm_los_pos.fit(X_train_pos_s[bal_sel], yl_bal)

    cm = confusion_matrix(yl_test, yl_pred_gbm)
    metrics_summary['task3_los'] = {
        'lr_acc': float(acc_lr),
        'gbm_acc': float(acc_gbm),
        'confusion_matrix': cm.tolist(),
        'best_model': 'GBM' if acc_gbm >= acc_lr else 'Logistic Regression'
    }

    # Save Task 3 Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Task 3 — LOS / NLOS Detection Results', fontsize=13, fontweight='bold', color='#e2e8f0')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NLOS', 'LOS'])
    disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title(f'Confusion Matrix (GBM Acc: {acc_gbm*100:.1f}%)')

    is_los = (yl_test == 1)
    axes[1].scatter(X_test[~is_los, 0], X_test[~is_los, 1], c=ORANGE, s=5, alpha=0.4, label='NLOS')
    axes[1].scatter(X_test[is_los, 0], X_test[is_los, 1], c=CYAN, s=5, alpha=0.6, label='LOS')
    axes[1].set_title('Spatial Distribution of LOS vs NLOS')
    axes[1].set_xlabel('X Position (m)'); axes[1].set_ylabel('Y Position (m)')
    axes[1].legend(facecolor='#111827', labelcolor='#e2e8f0')

    axes[2].bar(['Logistic Regression', 'Gradient Boosting'], [acc_lr*100, acc_gbm*100], color=[CYAN, GREEN], width=0.4)
    axes[2].set_title('Model Accuracy')
    axes[2].set_ylim(0, 105); axes[2].set_ylabel('Accuracy (%)')
    for idx_b, val in enumerate([acc_lr*100, acc_gbm*100]):
        axes[2].text(idx_b, val + 1, f"{val:.2f}%", ha='center', color='#e2e8f0', fontweight='bold')

    plt.tight_layout()
    fig.savefig(root_dir / 'los_results.png', dpi=130, facecolor='#0a0d1a')
    fig.savefig(static_img_dir / 'los_results.png', dpi=130, facecolor='#0a0d1a')
    plt.close(fig)

    # ==============================================================
    # TASK 4: CHANNEL FEATURE AUTOENCODER
    # ==============================================================
    print("\n--- [Step 6/6] Task 4: Channel Feature Autoencoder (18D -> 4D -> 18D) ---")
    LATENT_DIM = 4
    rng_ae = np.random.RandomState(42)
    X_ae_train = X_train_s + 0.05 * rng_ae.randn(*X_train_s.shape)
    X_ae_test  = X_test_s  + 0.05 * rng_ae.randn(*X_test_s.shape)

    print(f"  Training Encoder (18D -> 64 -> 32 -> {LATENT_DIM}D)...")
    encoder = MLPRegressor(hidden_layer_sizes=(64, 32, 16, LATENT_DIM), activation='relu', max_iter=200, early_stopping=True, random_state=42)
    pca = PCA(n_components=LATENT_DIM, random_state=42)
    pca_latent_train = pca.fit_transform(X_train_s)
    encoder.fit(X_ae_train, pca_latent_train)

    latent_train = encoder.predict(X_ae_train)
    latent_test  = encoder.predict(X_ae_test)

    print(f"  Training Decoder ({LATENT_DIM}D -> 16 -> 32 -> 64 -> 18D)...")
    decoder = MLPRegressor(hidden_layer_sizes=(16, 32, 64), activation='relu', max_iter=200, early_stopping=True, random_state=42)
    decoder.fit(latent_train, X_train_s)

    X_recon = decoder.predict(latent_test)
    mse_ae = float(mean_squared_error(X_test_s, X_recon))
    mae_ae = float(mean_absolute_error(X_test_s, X_recon))
    snr_ae = float(10 * np.log10(np.mean(X_test_s**2) / (mse_ae + 1e-12)))
    comp_ratio = float(X.shape[1] / LATENT_DIM)

    print(f"    Reconstruction MSE: {mse_ae:.4f}")
    print(f"    Reconstruction MAE: {mae_ae:.4f}")
    print(f"    Reconstruction SNR: {snr_ae:.2f} dB")
    print(f"    Compression Ratio:  {comp_ratio:.1f}x (18D -> {LATENT_DIM}D)")

    metrics_summary['task4_ae'] = {
        'mse': mse_ae,
        'mae': mae_ae,
        'snr_db': snr_ae,
        'compression_ratio': comp_ratio,
        'latent_dim': LATENT_DIM
    }

    # Save Task 4 Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Task 4 — Channel Feature Autoencoder (18D to 4D Latent)', fontsize=13, fontweight='bold', color='#e2e8f0')

    feat_mse = np.mean((X_test_s - X_recon)**2, axis=0)
    cols = plt.cm.plasma(feat_mse / (feat_mse.max() + 1e-6))
    axes[0].bar(range(len(feature_names)), feat_mse, color=cols)
    axes[0].set_xticks(range(len(feature_names)))
    axes[0].set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    axes[0].set_title('Per-Feature Reconstruction Error (MSE)')
    axes[0].set_ylabel('MSE')

    sample_id = 10
    axes[1].plot(X_test_s[sample_id], color=CYAN, marker='o', label='Original 18D')
    axes[1].plot(X_recon[sample_id], '--', color=PURPLE, marker='s', label='Reconstructed')
    axes[1].set_title('Feature Profile Comparison (Sample UE)')
    axes[1].set_xlabel('Feature Index'); axes[1].set_ylabel('Normalized Value')
    axes[1].legend(facecolor='#111827', labelcolor='#e2e8f0')

    sc_ae = axes[2].scatter(latent_test[:, 0], latent_test[:, 1], c=yr_test, cmap='viridis', s=6, alpha=0.6)
    axes[2].set_title('2D Latent Space Projection (Colored by RSS)')
    axes[2].set_xlabel('Latent Dim 1'); axes[2].set_ylabel('Latent Dim 2')
    plt.colorbar(sc_ae, ax=axes[2], label='RSS (dB)')

    plt.tight_layout()
    fig.savefig(root_dir / 'ae_results.png', dpi=130, facecolor='#0a0d1a')
    fig.savefig(static_img_dir / 'ae_results.png', dpi=130, facecolor='#0a0d1a')
    plt.close(fig)

    # Feature Importance Plot
    fi = rf_beam.feature_importances_
    sorted_fi = np.argsort(fi)[::-1]
    top_k = min(12, len(fi))

    fig, ax = plt.subplots(figsize=(10, 5))
    cols_fi = plt.cm.plasma(np.linspace(0.2, 0.9, top_k))
    ax.barh([feature_names[i] for i in sorted_fi[:top_k]][::-1],
            fi[sorted_fi[:top_k]][::-1], color=cols_fi, edgecolor='none')
    ax.set_title('Top Feature Importances — Beam Predictor (RF)', color='#e2e8f0', fontsize=12, fontweight='bold')
    ax.set_xlabel('Relative Importance')
    plt.tight_layout()
    fig.savefig(root_dir / 'feature_importance.png', dpi=130, facecolor='#0a0d1a')
    fig.savefig(static_img_dir / 'feature_importance.png', dpi=130, facecolor='#0a0d1a')
    plt.close(fig)

    # ==============================================================
    # SERIALIZE MODELS & EXPORT METADATA
    # ==============================================================
    print("\n--- Serializing Models to models/ ---")
    joblib.dump({'model': rf_beam, 'scaler': scaler, 'feature_names': feature_names}, models_dir / 'beam_rf.pkl')
    joblib.dump({'model': mlp_beam, 'scaler': scaler, 'feature_names': feature_names}, models_dir / 'beam_mlp.pkl')
    joblib.dump({'model': rf_beam_pos, 'scaler': scaler_spatial}, models_dir / 'beam_spatial.pkl')

    joblib.dump({'model': gbm_rss, 'scaler': scaler, 'feature_names': feature_names}, models_dir / 'rss_gbm.pkl')
    joblib.dump({'model': mlp_rss, 'scaler': scaler, 'feature_names': feature_names}, models_dir / 'rss_mlp.pkl')
    joblib.dump({'model': gbm_rss_pos, 'scaler': scaler_spatial}, models_dir / 'rss_spatial.pkl')

    joblib.dump({'model': gbm_los, 'scaler': scaler, 'feature_names': feature_names}, models_dir / 'los_gbm.pkl')
    joblib.dump({'model': lr_los, 'scaler': scaler, 'feature_names': feature_names}, models_dir / 'los_lr.pkl')
    joblib.dump({'model': gbm_los_pos, 'scaler': scaler_spatial}, models_dir / 'los_spatial.pkl')

    joblib.dump({'encoder': encoder, 'decoder': decoder, 'pca': pca, 'scaler': scaler}, models_dir / 'ae.pkl')

    # Save representative sample points for spatial coverage map in web app
    map_sample_size = 2500
    map_indices = np.random.RandomState(42).choice(N, map_sample_size, replace=False)
    coverage_data = [
        {
            'x': round(float(rx_pos[i, 0]), 2),
            'y': round(float(rx_pos[i, 1]), 2),
            'z': round(float(rx_pos[i, 2]), 2),
            'dist': round(float(dist_3d[i]), 1),
            'power_db': round(float(y_rss[i]), 2),
            'beam_id': int(y_beam[i]),
            'is_los': int(y_los[i]),
            'delay_ns': round(float(dom_delay_ns[i]), 2)
        }
        for i in map_indices
    ]

    metadata = {
        'scenario': 'ASU Campus 3.5 GHz (DeepMIMO)',
        'frequency_ghz': 3.5,
        'bandwidth_mhz': 100.0,
        'bs_position': [float(tx_pos[0]), float(tx_pos[1]), float(tx_pos[2])],
        'bounds': {
            'x_min': float(rx_pos[:, 0].min()), 'x_max': float(rx_pos[:, 0].max()),
            'y_min': float(rx_pos[:, 1].min()), 'y_max': float(rx_pos[:, 1].max()),
            'z_min': float(rx_pos[:, 2].min()), 'z_max': float(rx_pos[:, 2].max())
        },
        'feature_names': feature_names,
        'feature_importances': {feature_names[i]: float(fi[i]) for i in sorted_fi},
        'n_training_samples': N,
        'n_codebook_beams': N_BEAMS,
        'metrics': metrics_summary,
        'sample_coverage_points': coverage_data
    }

    with open(models_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_total
    print(f"\nTraining pipeline completed in {elapsed:.2f} seconds!")
    print(f"All models and assets saved to: {models_dir}")
    print("=" * 65)

if __name__ == '__main__':
    main()
