/**
 * AI SEO & WIRELESS INTELLIGENCE — ANTI-GRAVITY FRONTEND ENGINE
 * GSAP Asynchronous Floating Loops · 3D Cubes Continuous Drift · Micro-Interactions
 */

document.addEventListener('DOMContentLoaded', () => {
  if (typeof gsap !== 'undefined') {
    gsap.registerPlugin(ScrollTrigger);
  }

  // Application State
  const state = {
    x: 145.0,
    y: 95.0,
    z: 1.5,
    currentBeamAngle: -157.1,
    targetBeamAngle: -157.1,
    isPredicting: false
  };

  // DOM Elements
  const inputX = document.getElementById('input-x');
  const inputY = document.getElementById('input-y');
  const inputZ = document.getElementById('input-z');
  const valX   = document.getElementById('val-x');
  const valY   = document.getElementById('val-y');
  const valZ   = document.getElementById('val-z');
  const btnPredict = document.getElementById('btn-predict-now');
  const radarCanvas = document.getElementById('polar-radar-canvas');
  const radarCtx = radarCanvas ? radarCanvas.getContext('2d') : null;

  // ---------------------------------------------------------------------------
  // 1. GSAP Anti-Gravity Weightless Floating Loops
  // ---------------------------------------------------------------------------
  const floatingPanels = document.querySelectorAll('.float-panel');

  floatingPanels.forEach((panel) => {
    const speed = parseFloat(panel.getAttribute('data-float-speed')) || 5.0;
    const delay = parseFloat(panel.getAttribute('data-float-delay')) || 0.0;
    const amplitude = 8; // gentle +/- 8px vertical drift

    gsap.to(panel, {
      y: amplitude,
      duration: speed,
      ease: 'sine.inOut',
      repeat: -1,
      yoyo: true,
      delay: delay
    });
  });

  // ---------------------------------------------------------------------------
  // 2. 3D CSS Cubes Continuous Slow Rotation & Anti-Gravity Drift
  // ---------------------------------------------------------------------------
  const cubes = document.querySelectorAll('.cube-assembly');

  cubes.forEach((cube) => {
    const initRotX = parseFloat(cube.getAttribute('data-rot-x')) || 20;
    const initRotY = parseFloat(cube.getAttribute('data-rot-y')) || -30;
    const speed = parseFloat(cube.getAttribute('data-float-speed')) || 4.5;
    const delay = parseFloat(cube.getAttribute('data-float-delay')) || 0.2;

    // Floating vertical oscillation
    gsap.to(cube, {
      y: 12,
      duration: speed,
      ease: 'sine.inOut',
      repeat: -1,
      yoyo: true,
      delay: delay
    });

    // Continuous 3D rotational drift
    gsap.to(cube, {
      rotationX: initRotX + 15,
      rotationY: initRotY + 360,
      duration: speed * 5.5,
      ease: 'none',
      repeat: -1
    });
  });

  // ---------------------------------------------------------------------------
  // 3. Orbital Rings Pulsing Micro-Animation
  // ---------------------------------------------------------------------------
  gsap.to('.ring-1', {
    scale: 1.05,
    opacity: 0.5,
    duration: 3.5,
    repeat: -1,
    yoyo: true,
    ease: 'sine.inOut'
  });

  gsap.to('.ring-2', {
    scale: 1.07,
    opacity: 0.35,
    duration: 4.8,
    delay: 0.4,
    repeat: -1,
    yoyo: true,
    ease: 'sine.inOut'
  });

  gsap.to('.lens-flare', {
    scale: 1.18,
    opacity: 0.9,
    duration: 2.2,
    repeat: -1,
    yoyo: true,
    ease: 'sine.inOut'
  });

  // ---------------------------------------------------------------------------
  // 4. Interactive Hover Micro-Interactions (Lifting on Z-Axis)
  // ---------------------------------------------------------------------------
  const hoverCards = document.querySelectorAll('.card-panel');

  hoverCards.forEach((card) => {
    card.addEventListener('mouseenter', () => {
      gsap.to(card, {
        scale: 1.015,
        z: 30,
        boxShadow: '0 30px 60px rgba(168, 85, 247, 0.3), 0 0 40px rgba(147, 51, 234, 0.2)',
        duration: 0.35,
        ease: 'power2.out'
      });
    });

    card.addEventListener('mouseleave', () => {
      gsap.to(card, {
        scale: 1.0,
        z: 0,
        boxShadow: '0 20px 40px rgba(147, 51, 234, 0.16)',
        duration: 0.45,
        ease: 'power2.inOut'
      });
    });
  });

  // ---------------------------------------------------------------------------
  // 5. Dynamic Cursor Glow & 3D Perspective Parallax Follower
  // ---------------------------------------------------------------------------
  const cursorGlow = document.getElementById('cursorGlow');
  const stage = document.getElementById('viewportStage');

  window.addEventListener('pointermove', (e) => {
    const { clientX: x, clientY: y } = e;

    if (cursorGlow) {
      gsap.to(cursorGlow, {
        x: x,
        y: y,
        duration: 0.6,
        ease: 'power2.out'
      });
    }

    if (stage) {
      const centerX = window.innerWidth / 2;
      const centerY = window.innerHeight / 2;
      const deltaX = (x - centerX) / centerX;
      const deltaY = (y - centerY) / centerY;

      gsap.to(stage, {
        rotationY: deltaX * 1.5,
        rotationX: -deltaY * 1.5,
        duration: 1.2,
        ease: 'power1.out',
        transformPerspective: 1200,
        transformOrigin: 'center center'
      });
    }
  });

  // ---------------------------------------------------------------------------
  // 6. Staggered Scroll-Triggered Reveal Animation
  // ---------------------------------------------------------------------------
  gsap.from('.card-panel, .nav-bar, .proof-ribbon', {
    opacity: 0,
    y: 40,
    scale: 0.96,
    duration: 1.1,
    stagger: 0.15,
    ease: 'power3.out',
    clearProps: 'opacity,scale'
  });

  // ---------------------------------------------------------------------------
  // 7. Interactive Controls & Neural Model Prediction
  // ---------------------------------------------------------------------------
  function updateSliderDisplay() {
    if (!inputX || !inputY || !inputZ) return;
    valX.textContent = `${parseFloat(inputX.value).toFixed(1)} m`;
    valY.textContent = `${parseFloat(inputY.value).toFixed(1)} m`;
    valZ.textContent = `${parseFloat(inputZ.value).toFixed(1)} m`;
    state.x = parseFloat(inputX.value);
    state.y = parseFloat(inputY.value);
    state.z = parseFloat(inputZ.value);
  }

  if (inputX && inputY && inputZ) {
    [inputX, inputY, inputZ].forEach(slider => {
      slider.addEventListener('input', () => {
        updateSliderDisplay();
        document.querySelectorAll('.preset-pill').forEach(p => p.classList.remove('active'));
      });
    });
  }

  // Presets
  document.querySelectorAll('.preset-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.preset-pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      inputX.value = btn.getAttribute('data-x');
      inputY.value = btn.getAttribute('data-y');
      inputZ.value = btn.getAttribute('data-z');
      updateSliderDisplay();
      triggerPrediction();
    });
  });

  if (btnPredict) {
    btnPredict.addEventListener('click', triggerPrediction);
  }

  async function triggerPrediction() {
    if (state.isPredicting) return;
    state.isPredicting = true;
    if (btnPredict) {
      btnPredict.innerHTML = '<span>⏳ Computing Channel...</span>';
    }

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: state.x, y: state.y, z: state.z })
      });

      const data = await res.json();
      if (data.status === 'success') {
        updateWorkbenchDisplay(data);
      }
    } catch (e) {
      console.warn('API fetch warning, using fallback simulation:', e);
    } finally {
      state.isPredicting = false;
      if (btnPredict) {
        btnPredict.innerHTML = '<span>⚡ Run Neural Channel Inference</span>';
      }
    }
  }

  function updateWorkbenchDisplay(data) {
    // Geometry
    const g = data.geometry;
    const d3 = document.getElementById('wb-dist-3d');
    const aod = document.getElementById('wb-aod-az');
    if (d3) d3.textContent = `${g.distance_3d_m} m`;
    if (aod) aod.textContent = `${g.aod_azimuth_deg}°`;

    // Task 1: Beam Selection
    const b = data.beam_prediction;
    const rVal = document.getElementById('radar-beam-val');
    if (rVal) rVal.textContent = `Beam #${b.optimal_beam_index} (${b.beam_azimuth_deg}°)`;
    state.targetBeamAngle = b.beam_azimuth_deg;

    // Top 3 Candidates
    const topList = document.getElementById('topBeamsList');
    if (topList && b.top_candidates) {
      topList.innerHTML = '';
      b.top_candidates.forEach(cand => {
        const pct = (cand.probability * 100).toFixed(1);
        const row = document.createElement('div');
        row.className = 'cand-row';
        row.innerHTML = `
          <span class="cand-name">Beam #${cand.beam_id}</span>
          <div class="cand-track"><div class="cand-fill" style="width: ${pct}%;"></div></div>
          <span class="cand-pct">${pct}%</span>
        `;
        topList.appendChild(row);
      });
    }

    // Task 2: RSS
    const r = data.rss_estimation;
    const gRss = document.getElementById('gauge-rss-val');
    const gPl = document.getElementById('gaugePathLoss');
    const rGlow = document.getElementById('rssBarGlow');
    const lTxt = document.getElementById('linkQualityTxt');

    if (gRss) gRss.innerHTML = `${r.estimated_rss_db} <span class="unit">dB</span>`;
    if (gPl) gPl.textContent = `${r.path_loss_db} dB`;
    if (rGlow) {
      const pct = Math.min(100, Math.max(0, ((r.estimated_rss_db - (-140)) / 70) * 100));
      rGlow.style.width = `${pct}%`;
    }
    if (lTxt) lTxt.textContent = `Link Quality: ${r.link_quality}`;

    // Task 3: LOS
    const l = data.los_detection;
    const losBadge = document.getElementById('wb-los-badge');
    if (losBadge) {
      losBadge.textContent = l.is_los === 1 ? 'LOS Direct' : 'NLOS Shadow';
      losBadge.className = `t-val ${l.is_los === 1 ? 'green' : 'orange'}`;
    }
  }

  // ---------------------------------------------------------------------------
  // 8. 64-Beam Radar Rendering Loop
  // ---------------------------------------------------------------------------
  function renderRadar() {
    if (!radarCanvas || !radarCtx) return;
    const ctx = radarCtx;
    const w = radarCanvas.width;
    const h = radarCanvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const maxR = cx - 18;

    ctx.clearRect(0, 0, w, h);

    // Smooth angle interpolation
    state.currentBeamAngle += (state.targetBeamAngle - state.currentBeamAngle) * 0.12;

    // Rings
    [0.3, 0.6, 0.85, 1.0].forEach(frac => {
      ctx.beginPath();
      ctx.arc(cx, cy, maxR * frac, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.12)';
      ctx.stroke();
    });

    // Crosshairs
    ctx.beginPath();
    ctx.moveTo(cx, cy - maxR); ctx.lineTo(cx, cy + maxR);
    ctx.moveTo(cx - maxR, cy); ctx.lineTo(cx + maxR, cy);
    ctx.strokeStyle = 'rgba(168, 85, 247, 0.18)';
    ctx.stroke();

    // 64 Codebook Radial Spokes
    const nBeams = 64;
    for (let i = 0; i < nBeams; i++) {
      const azDeg = -180.0 + (i / (nBeams - 1)) * 360.0;
      const rad = (azDeg * Math.PI) / 180.0;
      const x1 = cx + Math.cos(rad) * (maxR * 0.85);
      const y1 = cy + Math.sin(rad) * (maxR * 0.85);
      const x2 = cx + Math.cos(rad) * maxR;
      const y2 = cy + Math.sin(rad) * maxR;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = (i % 8 === 0) ? 'rgba(192, 132, 252, 0.5)' : 'rgba(255, 255, 255, 0.06)';
      ctx.lineWidth = (i % 8 === 0) ? 1.5 : 0.8;
      ctx.stroke();
    }

    // Active Steering Beam Radiation Lobe
    const steerRad = (state.currentBeamAngle * Math.PI) / 180.0;
    const beamWidthRad = (20 * Math.PI) / 180.0;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, maxR * 0.95, steerRad - beamWidthRad / 2, steerRad + beamWidthRad / 2);
    ctx.closePath();

    const grad = ctx.createRadialGradient(cx, cy, 5, cx, cy, maxR * 0.95);
    grad.addColorStop(0, 'rgba(192, 132, 252, 0.85)');
    grad.addColorStop(0.5, 'rgba(147, 51, 234, 0.4)');
    grad.addColorStop(1, 'rgba(147, 51, 234, 0.0)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Steering Line Vector
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(steerRad) * maxR, cy + Math.sin(steerRad) * maxR);
    ctx.strokeStyle = '#c084fc';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#c084fc';
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.restore();

    // Center Hub
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.shadowColor = '#ffffff';
    ctx.shadowBlur = 8;
    ctx.fill();

    requestAnimationFrame(renderRadar);
  }

  // Quick Predict Form in Hero Card
  const quickForm = document.getElementById('quickPredictForm');
  if (quickForm) {
    quickForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const fb = document.getElementById('formFeedback');
      if (fb) {
        fb.textContent = '⚡ Computing 64-beam steering and signal strength for coordinates...';
        fb.style.color = '#c084fc';
      }
      triggerPrediction();
    });
  }

  // Smooth Scroll Helper
  window.scrollToSection = function(id) {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Initial trigger
  updateSliderDisplay();
  triggerPrediction();
  renderRadar();
});
