/* ═══════════════════════════════════════════════════════════
   dashboard.js — Core JS (particles, clock, health, detection)
   ═══════════════════════════════════════════════════════════ */
"use strict";

// ── State ────────────────────────────────────────────────────
let sessionStats = { total: 0, passed: 0, failed: 0, totalConf: 0 };
let defectFreqMap = {};
let recentHistory = [];
let lastReport = null;
let particleSystem = null;

// ── Particle System ──────────────────────────────────────────
class ParticleSystem {
     constructor(canvas) {
          this.canvas = canvas;
          this.ctx = canvas.getContext('2d');
          this.particles = [];
          this.running = true;
          this.resize();
          this.init();
          window.addEventListener('resize', () => this.resize());
          this.animate();
     }
     resize() {
          this.canvas.width = window.innerWidth;
          this.canvas.height = window.innerHeight;
     }
     init() {
          this.particles = [];
          const count = Math.min(80, Math.floor(window.innerWidth / 18));
          for (let i = 0; i < count; i++) this.particles.push(this.spawn());
     }
     spawn() {
          return {
               x: Math.random() * this.canvas.width,
               y: Math.random() * this.canvas.height,
               vx: (Math.random() - 0.5) * 0.28,
               vy: (Math.random() - 0.5) * 0.28,
               r: Math.random() * 1.8 + 0.4,
               a: Math.random() * 0.35 + 0.08,
               color: Math.random() > 0.65 ? '124,58,237' : '6,182,212',
          };
     }
     animate() {
          if (!this.running) return;
          const { ctx, canvas, particles } = this;
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          particles.forEach(p => {
               p.x += p.vx; p.y += p.vy;
               if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
               if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
               ctx.beginPath();
               ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
               ctx.fillStyle = `rgba(${p.color},${p.a})`;
               ctx.fill();
          });

          // Draw connections
          for (let i = 0; i < particles.length; i++) {
               for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const d = Math.sqrt(dx * dx + dy * dy);
                    if (d < 110) {
                         ctx.beginPath();
                         ctx.moveTo(particles[i].x, particles[i].y);
                         ctx.lineTo(particles[j].x, particles[j].y);
                         ctx.strokeStyle = `rgba(6,182,212,${0.07 * (1 - d / 110)})`;
                         ctx.lineWidth = 0.5;
                         ctx.stroke();
                    }
               }
          }
          requestAnimationFrame(() => this.animate());
     }
}

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
     // Particles
     const canvas = document.getElementById('particles-canvas');
     if (canvas) particleSystem = new ParticleSystem(canvas);

     // Clock
     updateClock();
     setInterval(updateClock, 1000);

     // Health
     fetchHealth();
     setInterval(fetchHealth, 30000);

     // Sidebar
     setupSidebar();

     // Ripple
     setupRipple();

     // Drop zone (if on dashboard)
     setupDropZone();

     // Pending badge
     fetchPendingBadge();
});

// ── Real-time Clock ──────────────────────────────────────────
function updateClock() {
     const el = document.getElementById('navClock');
     if (!el) return;
     const now = new Date();
     el.textContent = now.toLocaleTimeString('en-IN', { hour12: false });
}

// ── Health Check ─────────────────────────────────────────────
async function fetchHealth() {
     const dot = document.getElementById('navStatusDot');
     const text = document.getElementById('navStatusText');
     try {
          const r = await fetch('/health');
          const d = await r.json();
          if (dot) { dot.className = 'nav-status-dot online'; }
          if (text) text.textContent = `YOLOv8 ${d.yolo_model === 'custom trained' ? '✓' : '(fallback)'} · CNN ${d.cnn_lstm_model === 'trained' ? '✓' : '(init)'}`;

          // Sidebar widget
          const mhwY = document.getElementById('mhwYolo');
          const mhwC = document.getElementById('mhwCnn');
          const mhwA = document.getElementById('mhwAnomaly');
          if (mhwY) { mhwY.textContent = d.yolo_model === 'custom trained' ? 'Trained ✓' : 'Fallback'; mhwY.className = `mhw-val ${d.yolo_model === 'custom trained' ? 'ok' : 'warn'}`; }
          if (mhwC) { mhwC.textContent = d.cnn_lstm_model === 'trained' ? 'Trained ✓' : 'Untrained'; mhwC.className = `mhw-val ${d.cnn_lstm_model === 'trained' ? 'ok' : 'warn'}`; }
          if (mhwA) { mhwA.textContent = d.anomaly_model === 'loaded' ? 'Loaded ✓' : 'Not Loaded'; mhwA.className = `mhw-val ${d.anomaly_model === 'loaded' ? 'ok' : 'warn'}`; }
     } catch {
          if (dot) dot.className = 'nav-status-dot offline';
          if (text) text.textContent = 'Server offline';
     }
}

// ── Pending Badge ────────────────────────────────────────────
async function fetchPendingBadge() {
     try {
          const r = await fetch('/api/pending-images');
          const d = await r.json();
          const badge = document.getElementById('sidebarPendingBadge');
          if (badge && d.pending && d.pending.length > 0) {
               badge.textContent = d.pending.length;
               badge.style.display = 'inline-block';
          }
     } catch { }
}

// ── Sidebar ──────────────────────────────────────────────────
function setupSidebar() {
     const sidebar = document.getElementById('sidebar');
     const colBtn = document.getElementById('sidebarCollapseBtn');
     const mobileBtn = document.getElementById('sidebarToggleMobile');

     colBtn?.addEventListener('click', () => sidebar.classList.remove('open'));
     mobileBtn?.addEventListener('click', () => sidebar.classList.toggle('open'));

     // Close on outside click (mobile)
     document.addEventListener('click', e => {
          if (window.innerWidth <= 900 && sidebar?.classList.contains('open') &&
               !sidebar.contains(e.target) && !mobileBtn?.contains(e.target)) {
               sidebar.classList.remove('open');
          }
     });
}

// ── Ripple ───────────────────────────────────────────────────
function setupRipple() {
     document.querySelectorAll('.btn, .btn-primary, .btn-secondary, .chat-send-btn').forEach(btn => {
          btn.addEventListener('click', e => {
               const rect = btn.getBoundingClientRect();
               const ripple = document.createElement('span');
               ripple.className = 'ripple';
               ripple.style.left = (e.clientX - rect.left) + 'px';
               ripple.style.top = (e.clientY - rect.top) + 'px';
               btn.appendChild(ripple);
               setTimeout(() => ripple.remove(), 600);
          });
     });
}

// ── Drop Zone (Dashboard) ────────────────────────────────────
function setupDropZone() {
     const dz = document.getElementById('dropZone');
     const inp = document.getElementById('imageInput');
     if (!dz || !inp) return;

     dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
     dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
     dz.addEventListener('drop', e => {
          e.preventDefault(); dz.classList.remove('drag-over');
          const f = e.dataTransfer?.files?.[0];
          if (f) handleFile(f);
     });
     dz.addEventListener('click', e => {
          if (!e.target.closest('button') && !e.target.closest('img') && !e.target.closest('input'))
               inp.click();
     });
     inp.addEventListener('change', () => { if (inp.files[0]) handleFile(inp.files[0]); });
}

function handleFile(file) {
     if (!file.type.startsWith('image/')) { showToast('Please upload an image file', 'fail'); return; }
     const reader = new FileReader();
     reader.onload = ev => {
          document.getElementById('uploadIdle').style.display = 'none';
          const preview = document.getElementById('uploadPreview');
          preview.style.display = 'flex';
          const img = document.getElementById('previewImg');
          img.src = ev.target.result;
          const detectBtn = document.getElementById('detectBtn');
          if (detectBtn) detectBtn.disabled = false;
     };
     reader.readAsDataURL(file);
}

// ── Detection ────────────────────────────────────────────────
async function runDetection() {
     const inp = document.getElementById('imageInput');
     if (!inp?.files[0]) return;

     const detectBtn = document.getElementById('detectBtn');
     detectBtn.disabled = true;
     detectBtn.textContent = '⏳ Analysing…';

     // Show progress
     document.getElementById('uploadPreview').style.display = 'none';
     document.getElementById('uploadIdle').style.display = 'none';
     const prog = document.getElementById('uploadProgress');
     prog.style.display = 'flex';
     setStep(1, 'active');

     const formData = new FormData();
     formData.append('image', inp.files[0]);

     try {
          // Step 1 — uploading
          await delay(400);
          setStep(1, 'done'); setStep(2, 'active');

          const resp = await fetch('/api/pipeline', { method: 'POST', body: formData });
          if (!resp.ok) throw new Error((await resp.json()).error || resp.statusText);
          const data = await resp.json();

          setStep(2, 'done'); setStep(3, 'active');

          // Step 2 — generate report
          let report = null;
          try {
               const rr = await fetch('/api/report', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
               });
               if (rr.ok) report = (await rr.json()).report;
          } catch { }
          lastReport = report;

          await delay(300);
          setStep(3, 'done');
          await delay(400);

          // Render
          prog.style.display = 'none';
          document.getElementById('uploadPreview').style.display = 'flex';
          renderResults(data);
          updateSessionStats(data);
          updateKPIs();
          updateChartsFromResult(data);
          addToRecentTable(data, inp.files[0].name);

          const isPass = (data.status === 'PASS');
          showToast(isPass ? '✅ PASS — Component cleared' : '❌ FAIL — Defects detected', isPass ? 'pass' : 'fail');

     } catch (err) {
          prog.style.display = 'none';
          document.getElementById('uploadIdle').style.display = 'flex';
          showToast('Detection failed: ' + err.message, 'fail');
          console.error(err);
     } finally {
          detectBtn.disabled = false;
          detectBtn.innerHTML = '⚡ Inspect Component';
     }
}

function setStep(n, state) {
     const el = document.getElementById('step' + n);
     if (!el) return;
     el.className = 'progress-step ' + state;
     if (state === 'done') el.innerHTML = '<div class="step-check">✅</div> ' + el.textContent.trim().replace(/^(⏳|✅)?/, '');
     if (state === 'active') {
          const labels = { 1: 'Uploading image…', 2: 'Running AI pipeline…', 3: 'Generating report…' };
          el.innerHTML = '<div class="step-spinner"></div> ' + labels[n];
     }
}

// ── Render Results ───────────────────────────────────────────
function renderResults(data) {
     const isPass = data.status === 'PASS';
     const conf = data.confidence ?? 0;
     const resultArea = document.getElementById('resultArea');
     if (resultArea) resultArea.style.display = 'block';

     // Verdict badge
     const vw = document.getElementById('verdictBadgeWrap');
     if (vw) {
          vw.innerHTML = `
      <div class="verdict-badge ${isPass ? 'pass' : 'fail'}">
        <div class="verdict-pulse"></div>
        ${isPass ? '✅ PASS' : '❌ FAIL'}
      </div>`;
     }

     // Confidence ring
     const circumference = 213;
     const ring = document.getElementById('confRing');
     const rl = document.getElementById('confRingLabel');
     if (ring) {
          ring.style.strokeDashoffset = circumference - (circumference * conf);
          ring.style.stroke = isPass ? '#10B981' : '#EF4444';
     }
     if (rl) { rl.textContent = Math.round(conf * 100) + '%'; rl.style.color = isPass ? '#10B981' : '#EF4444'; }

     const cv = document.getElementById('confValue');
     const mu = document.getElementById('modelUsed');
     if (cv) cv.textContent = (conf * 100).toFixed(1) + '%';
     if (mu) mu.textContent = data.model_info?.type || '—';

     // Annotated image
     const ri = document.getElementById('resultImage');
     if (ri) {
          const b64 = data.annotated_image || data.heatmap_image;
          if (b64) {
               ri.src = `data:image/png;base64,${b64}`;
               ri.style.opacity = '0';
               ri.onload = () => { ri.style.opacity = '1'; };
          }
     }

     // Defect chips
     const da = document.getElementById('defectsArea');
     if (da) {
          const defects = data.defects || [];
          const discarded = data.discarded_defects || [];
          if (defects.length || discarded.length) {
               da.innerHTML = `<div class="defects-grid">` +
                    defects.map(d => `<div class="defect-chip">${d.class} <span style="opacity:.7">${(d.confidence * 100).toFixed(0)}%</span></div>`).join('') +
                    discarded.map(d => `<div class="defect-chip fp">${d.class} (FP)</div>`).join('') +
                    '</div>';
          } else {
               da.innerHTML = '<div class="text-muted text-sm" style="margin-top:.5rem">No defects detected</div>';
          }
     }

     // AI Explanation
     const ae = document.getElementById('aiExplanation');
     const aet = document.getElementById('aiExplanationText');
     if (ae && aet) {
          const defects = data.defects || [];
          const anomaly = data.anomaly_info?.status || 'N/A';
          const classes = [...new Set(defects.map(d => d.class))];
          let txt = '';
          if (isPass) {
               txt = `✅ No defects were detected. Component appears healthy (confidence: ${(conf * 100).toFixed(1)}%). Anomaly scan: ${anomaly}. Cleared for assembly.`;
          } else {
               const summary = classes.map(c => `${defects.filter(d => d.class === c).length}× ${c}`).join(', ');
               txt = `⚠️ Detected ${defects.length} defect(s): ${summary || 'Unknown'}. Highest confidence: ${(conf * 100).toFixed(1)}%. Anomaly: ${anomaly}. ${data.active_learning_flagged ? '🎓 Flagged for active learning review.' : ''} Recommended: manual inspection required.`;
          }
          aet.textContent = txt;
          ae.style.display = 'block';
     }
}

// ── Batch Upload ───────────────────────────────────────────
async function uploadBatch() {
     const inp = document.getElementById('batchImageInput');
     if (!inp?.files?.length) { showToast('Select images first', 'fail'); return; }

     const btn = document.getElementById('batchUploadBtn');
     btn.disabled = true; btn.textContent = '⏳ Uploading…';

     const fd = new FormData();
     for (const f of inp.files) fd.append('images', f);

     try {
          const r = await fetch('/api/batch-detect', { method: 'POST', body: fd });
          if (!r.ok) throw new Error((await r.json()).error || r.statusText);
          const d = await r.json();
          const area = document.getElementById('batchResults');
          area.style.display = 'block';
          area.innerHTML = `<div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:center;margin-bottom:.5rem">
    <span class="tag tag-green">${d.passed}/${d.total} Passed</span>
    <span class="tag tag-red">${d.failed} Failed</span>
  </div>
  <div style="max-height:200px;overflow-y:auto;font-size:.82rem">` +
    d.results.map(r => `<div style="display:flex;align-items:center;gap:.5rem;padding:3px 0;border-bottom:1px solid rgba(255,255,255,.04)">
    <span class="status-badge ${r.status === 'PASS' ? 'pass' : 'fail'}" style="font-size:.7rem">${r.status}</span>
    <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${r.filename}</span>
    <span class="mono text-xs">${(r.confidence*100).toFixed(1)}%</span>
    <span class="text-muted text-xs">${r.defects} defects</span>
  </div>`).join('') +
  '</div>';
          showToast(`Batch complete: ${d.passed}/${d.total} passed`, d.failed === 0 ? 'pass' : 'fail');
          loadHistory();
     } catch (err) {
          showToast('Batch upload failed: ' + err.message, 'fail');
     } finally {
          btn.disabled = false; btn.innerHTML = '📤 Upload & Inspect All';
     }
}

// ── PDF Report ────────────────────────────────────────────
async function downloadPDFReport() {
     if (!lastReport) { showToast('Run an inspection first', 'fail'); return; }
     try {
          const r = await fetch('/api/report/pdf', {
               method: 'POST', headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify(lastReport)
          });
          if (!r.ok) throw new Error('PDF generation failed');
          const blob = await r.blob();
          const a = document.createElement('a');
          a.href = URL.createObjectURL(blob);
          a.download = `report_${Date.now()}.pdf`;
          a.click();
          showToast('PDF report downloaded', 'pass');
     } catch (err) {
          showToast(err.message, 'fail');
     }
}

// ── History CSV ───────────────────────────────────────────
function downloadCSVExport() {
     const a = document.createElement('a');
     a.href = '/api/history/export';
     a.download = 'inspections.csv';
     a.click();
     showToast('CSV exported', 'pass');
}

// ── Continuous Capture ────────────────────────────────────
let continuousCaptureInterval = null;

function toggleContinuousCapture() {
     const btn = document.getElementById('continuousBtn');
     if (continuousCaptureInterval) {
          clearInterval(continuousCaptureInterval);
          continuousCaptureInterval = null;
          btn.textContent = '🔁 Start Continuous Capture';
          btn.className = 'btn btn-accent';
          showToast('Continuous capture stopped', 'info');
          return;
     }

     // Check if camera is active
     const isIp = activeCamTab === 'ipcam';
     if ((isIp && !document.getElementById('ipCamFeed').src) || (!isIp && !webcamStream)) {
          showToast('Start a camera feed first', 'fail');
          return;
     }

     btn.textContent = '⏹ Stop Continuous Capture';
     btn.className = 'btn btn-danger';

     continuousCaptureInterval = setInterval(async () => {
          if (isIp) await captureIpFrame();
          else await captureFrame();
     }, 5000);
     showToast('Continuous capture every 5s started', 'pass');
}

// ── Session Stats ────────────────────────────────────────────
function updateSessionStats(data) {
     const isPass = data.status === 'PASS';
     sessionStats.total++;
     if (isPass) sessionStats.passed++; else sessionStats.failed++;
     sessionStats.totalConf += (data.confidence || 0);
     recentHistory.unshift({ data, name: document.getElementById('imageInput')?.files[0]?.name || 'image', time: new Date() });
     if (recentHistory.length > 50) recentHistory.pop();

     // Defect freq map
     if (data.defects?.length) {
          data.defects.forEach(d => { defectFreqMap[d.class] = (defectFreqMap[d.class] || 0) + 1; });
     } else {
          defectFreqMap['No Defect'] = (defectFreqMap['No Defect'] || 0) + 1;
     }
}

function updateKPIs() {
     const { total, passed, failed, totalConf } = sessionStats;
     const passRate = total ? Math.round(passed / total * 100) : 0;
     const avgConf = total ? Math.round(totalConf / total * 100) : 0;

     animCounter('kpiTotal', total);
     animCounter('kpiDefects', failed);
     const kpiPass = document.getElementById('kpiPassRate');
     if (kpiPass) kpiPass.textContent = total ? passRate + '%' : '—%';
     const kpiConf = document.getElementById('kpiConf');
     if (kpiConf) kpiConf.textContent = total ? avgConf + '%' : '—%';

     // Bars
     setBarWidth('kpiTotalBar', Math.min(100, total * 5));
     setBarWidth('kpiDefectsBar', total ? Math.round(failed / total * 100) : 0);
     setBarWidth('kpiPassBar', passRate);
     setBarWidth('kpiConfBar', avgConf);

     // Deltas
     setText('kpiTotalDelta', `${passed} passed · ${failed} failed`);
     setText('kpiDefectsDelta', failed > 0 ? `${Math.round(failed / total * 100)}% defect rate` : 'No defects ✓');
     setText('kpiPassDelta', passed > 0 ? `${passed} of ${total} cleared` : 'No passes yet');
     setText('kpiConfDelta', avgConf > 70 ? '⬆ High confidence' : avgConf > 40 ? 'Moderate confidence' : 'Low confidence');

     // Recent count
     setText('recentCount', total + ' scans');
}

// ── Recent Table ─────────────────────────────────────────────
function addToRecentTable(data, name) {
     const tbody = document.getElementById('recentTable');
     if (!tbody) return;
     const isPass = data.status === 'PASS';
     const conf = ((data.confidence || 0) * 100).toFixed(1);
     const entry = document.createElement('div');
     entry.className = 'slide-in';
     entry.style.cssText = 'display:flex;align-items:center;gap:.75rem;padding:.5rem .75rem;border-bottom:1px solid rgba(255,255,255,.04);font-size:.82rem';
     entry.innerHTML = `
    <span class="status-badge ${isPass ? 'pass' : 'fail'}">${isPass ? '✅ PASS' : '❌ FAIL'}</span>
    <span class="text-muted" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${name || 'image'}</span>
    <span class="mono text-xs" style="color:${isPass ? 'var(--green)' : 'var(--red)'}">${conf}%</span>
    <span class="text-muted text-xs">${new Date().toLocaleTimeString()}</span>`;

     // Remove empty state
     const empty = tbody.querySelector('.empty-state');
     if (empty) tbody.innerHTML = '';
     tbody.prepend(entry);

     // Keep max 10
     const items = tbody.children;
     if (items.length > 10) tbody.removeChild(items[items.length - 1]);
}

async function loadHistory() {
     try {
          const r = await fetch('/api/history');
          const d = await r.json();
          if (d.history?.length) {
               d.history.forEach(h => recentHistory.push({ data: h, name: h.filename, time: new Date(h.timestamp) }));
               const tbody = document.getElementById('recentTable');
               if (tbody && d.history.length) {
                    tbody.innerHTML = d.history.slice(0, 10).map(h => {
                         const isPass = h.verdict !== 'FAIL';
                         return `<div style="display:flex;align-items:center;gap:.75rem;padding:.5rem .75rem;border-bottom:1px solid rgba(255,255,255,.04);font-size:.82rem">
            <span class="status-badge ${isPass ? 'pass' : 'fail'}">${isPass ? '✅ PASS' : '❌ FAIL'}</span>
            <span class="text-muted" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${h.filename || 'image'}</span>
            <span class="mono text-xs" style="color:${isPass ? 'var(--green)' : 'var(--red)'}">${((h.confidence || 0) * 100).toFixed(1)}%</span>
            <span class="text-muted text-xs">${h.timestamp || ''}</span>
          </div>`;
                    }).join('');
                    setText('recentCount', d.total + ' total');
               }
          }
     } catch { }
}

// ── Report Download ──────────────────────────────────────────
function downloadReport() {
     if (!lastReport) return showToast('No report available', 'fail');
     const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: 'application/json' });
     const a = document.createElement('a');
     a.href = URL.createObjectURL(blob); a.download = `${lastReport.report_id || 'report'}.json`; a.click();
}

// ── Utilities ────────────────────────────────────────────────
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

function animCounter(id, target, duration = 800) {
     const el = document.getElementById(id);
     if (!el) return;
     const start = parseInt(el.textContent) || 0;
     const t0 = performance.now();
     const tick = now => {
          const p = Math.min((now - t0) / duration, 1);
          const e = 1 - Math.pow(1 - p, 3);
          el.textContent = Math.round(start + (target - start) * e);
          if (p < 1) requestAnimationFrame(tick);
     };
     requestAnimationFrame(tick);
}

function setBarWidth(id, pct) {
     const el = document.getElementById(id);
     if (el) el.style.width = pct + '%';
}
function setText(id, val) {
     const el = document.getElementById(id);
     if (el) el.textContent = val;
}

function showToast(msg, type = 'info') {
     let container = document.getElementById('toast-container');
     if (!container) { container = document.createElement('div'); container.id = 'toast-container'; document.body.appendChild(container); }
     const t = document.createElement('div');
     t.className = `toast toast-${type}`;
     t.innerHTML = `<span>${msg}</span>`;
     container.appendChild(t);
     setTimeout(() => t.remove(), 3600);
}

function resetCharts() {
     defectFreqMap = {};
     sessionStats = { total: 0, passed: 0, failed: 0, totalConf: 0 };
     recentHistory = [];
     updateKPIs();
     if (typeof dashCharts !== 'undefined') {
          Object.values(dashCharts).forEach(c => { if (c) { c.data.labels = []; c.data.datasets.forEach(d => d.data = []); c.update(); } });
     }
     showToast('Session data reset', 'info');
}

// Charts bridge (called by charts.js)
function updateChartsFromResult(data) {
     if (typeof updateDashCharts === 'function') updateDashCharts(data, defectFreqMap, sessionStats);
}
