/* ═══════════════════════════════════════════════════════════
   assistant.js — AI Chat interface (Gemini-powered)
   ═══════════════════════════════════════════════════════════ */
"use strict";

let msgCount = 0;

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
     await loadInsights();
});

async function loadInsights() {
     try {
          const [hRes, hlRes] = await Promise.all([
               fetch('/api/history?limit=10'),
               fetch('/health'),
          ]);
          const history = (await hRes.json()).history || [];
          const health = await hlRes.json();

          // Update model info
          setText('miYolo', health.yolo_model || '—');
          setText('miCnn', health.cnn_lstm_model || '—');
          setText('miAnomaly', health.anomaly_model || '—');

          // Update insight cards
          if (history.length) {
               const last = history[0];
               const total = history.length;
               const passed = history.filter(r => r.verdict !== 'FAIL').length;
               const rate = Math.round(passed / total * 100);
               const highConf = history.reduce((b, r) => r.confidence > b.confidence ? r : b, history[0]);

               setText('insLastResult', (last.verdict === 'FAIL' ? '❌ FAIL' : '✅ PASS') + ` — ${((last.confidence || 0) * 100).toFixed(1)}%`);
               setText('insLastTime', last.timestamp || '—');
               setText('insHighConf', (highConf.confidence * 100).toFixed(1) + '%');
               setText('insHighConfFile', highConf.filename || '—');
               setText('insMostCommon', history.some(r => r.num_defects > 0) ? 'Defective' : 'All PASS');
               setText('insMostCommonCount', `${history.filter(r => r.num_defects > 0).length} / ${total} inspections`);
               setText('insPassRate', rate + '%');
               setText('insPassRateSub', `${passed} passed out of ${total} total`);
               setText('insAction', rate >= 90 ? '✅ System performing well' : rate >= 70 ? '⚠️ Monitor defect frequency' : '🔴 High failure rate — recommend retraining');
          }
     } catch { }
}

// ── Chat ──────────────────────────────────────────────────────
async function sendMessage() {
     const input = document.getElementById('chatInput');
     const text = input.value.trim();
     if (!text) return;
     input.value = '';
     input.style.height = 'auto';
     addMessage('user', text);
     await generateResponse(text);
}

function sendQuickMsg(text) {
     document.getElementById('chatInput').value = text;
     sendMessage();
}

function addMessage(role, html, isTyping = false) {
     const container = document.getElementById('chatMessages');
     const idx = ++msgCount;
     const div = document.createElement('div');
     div.className = `msg ${role} fade-up`;
     div.style.animationDelay = '0s';
     div.id = `msg-${idx}`;

     const time = new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
     const avatar = role === 'ai' ? '🤖' : '👤';
     div.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div>
      <div class="msg-bubble" id="bubble-${idx}">${isTyping ? typingHTML() : html}</div>
      <div class="msg-time">${time}</div>
    </div>`;
     container.appendChild(div);
     container.scrollTop = container.scrollHeight;
     return idx;
}

function typingHTML() {
     return `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
}

async function generateResponse(userText) {
     const typingId = addMessage('ai', '', true);
     await sleep(300);

     try {
          const res = await fetch('/api/assistant/chat', {
               method: 'POST',
               headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify({ message: userText })
          });
          const data = await res.json();
          const response = data.response || '❌ No response from AI.';

          const bubble = document.getElementById(`bubble-${typingId}`);
          if (bubble) {
               bubble.innerHTML = '';
               await typeEffect(bubble, response);
          }
     } catch (e) {
          const bubble = document.getElementById(`bubble-${typingId}`);
          if (bubble) bubble.innerHTML = '⚠️ Could not reach the AI assistant. Is the server running?';
     }
     document.getElementById('chatMessages').scrollTop = 99999;
}

async function typeEffect(el, text, delay = 12) {
     el.innerHTML = '';
     const cleaned = text.replace(/<[^>]+>/g, '').length;
     if (cleaned < 200) {
          await sleep(delay * Math.min(cleaned, 60));
          el.innerHTML = text;
     } else {
          el.innerHTML = text;
     }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function setText(id, val) {
     const el = document.getElementById(id);
     if (el) el.textContent = val;
}

// Auto-resize chat input
document.addEventListener('DOMContentLoaded', () => {
     const inp = document.getElementById('chatInput');
     if (inp) {
          inp.addEventListener('input', () => {
               inp.style.height = 'auto';
               inp.style.height = Math.min(inp.scrollHeight, 120) + 'px';
          });
     }
});
