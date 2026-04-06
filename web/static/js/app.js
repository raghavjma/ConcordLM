/**
 * ConcordLM Dashboard — Frontend Application
 * 
 * Single-page app controlling the alignment pipeline:
 * training, evaluation, chat, datasets, models.
 */

// ══════════════════════════════════════════════════════════════
// State
// ══════════════════════════════════════════════════════════════

const API = '';  // Same-origin
let currentPage = 'dashboard';
let currentConfigTab = 'base';
let pipelineStatus = null;
let pairCounter = 0;
let activeWebSockets = {};

// ══════════════════════════════════════════════════════════════
// Navigation
// ══════════════════════════════════════════════════════════════

document.querySelectorAll('.nav-item[data-page]').forEach(item => {
  item.addEventListener('click', () => navigateTo(item.dataset.page));
});

function navigateTo(page) {
  // Deactivate all
  document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));

  // Activate target
  const navEl = document.querySelector(`.nav-item[data-page="${page}"]`);
  if (navEl) navEl.classList.add('active');

  const pageEl = document.getElementById(`page-${page}`);
  if (pageEl) pageEl.classList.add('active');

  currentPage = page;

  // Trigger page-specific loads
  if (page === 'config') loadConfig(currentConfigTab);
  if (page === 'models') loadModels();
  if (page === 'datasets') loadDatasets();
  if (page === 'evaluation') loadEvalReport();
  if (page === 'dashboard') refreshStatus();
}

// ══════════════════════════════════════════════════════════════
// Toast Notifications
// ══════════════════════════════════════════════════════════════

function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const icons = { success: '✅', error: '❌', info: 'ℹ️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span> <span>${message}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ══════════════════════════════════════════════════════════════
// API Helpers
// ══════════════════════════════════════════════════════════════

async function apiGet(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function apiPost(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

// ══════════════════════════════════════════════════════════════
// Dashboard
// ══════════════════════════════════════════════════════════════

async function refreshStatus() {
  try {
    pipelineStatus = await apiGet('/api/status');
    updateDashboard(pipelineStatus);
    
    // Also load models count
    try {
      const models = await apiGet('/api/models');
      document.getElementById('stat-models').textContent = models.length;
    } catch(e) { /* ok */ }
    
  } catch (err) {
    console.error('Status refresh failed:', err);
  }
}

function updateDashboard(status) {
  const { pipeline, active_jobs, eval_report } = status;

  // Pipeline nodes & stage cards
  const stages = ['sft', 'dpo', 'rlhf'];
  let completedCount = 0;
  let currentStage = 'Ready';

  stages.forEach(stage => {
    const completed = pipeline[stage]?.completed;
    const nodeEl = document.getElementById(`node-${stage}`);
    const statusEl = document.getElementById(`${stage}-status`);
    const stageStatusEl = document.getElementById(`${stage}-stage-status`);

    if (completed) {
      completedCount++;
      nodeEl?.classList.add('completed');
      if (statusEl) statusEl.innerHTML = '<span class="tag tag-green">✅ Complete</span>';
      if (stageStatusEl) {
        stageStatusEl.className = 'stage-status completed';
        stageStatusEl.textContent = '✅ Completed';
      }
    }
  });

  // Check active jobs
  const jobCount = Object.keys(active_jobs || {}).length;
  document.getElementById('stat-jobs').textContent = jobCount;

  if (jobCount > 0) {
    const firstJob = Object.values(active_jobs)[0];
    currentStage = firstJob.stage.toUpperCase();
  } else if (completedCount === 0) {
    currentStage = 'Ready';
  } else if (completedCount < 3) {
    currentStage = stages[completedCount].toUpperCase();
  } else {
    currentStage = 'Done ✅';
  }

  document.getElementById('stat-stage').textContent = currentStage;
  document.getElementById('stat-completed').textContent = `${completedCount}/3`;

  // Pipeline arrows
  if (pipeline.sft?.completed) {
    const arrow = document.getElementById('arrow-sft');
    if (arrow) arrow.classList.add('completed');
  }
  if (pipeline.dpo?.completed) {
    const arrow = document.getElementById('arrow-dpo');
    if (arrow) arrow.classList.add('completed');
  }

  // Eval summary
  if (eval_report) {
    renderEvalSummary(eval_report);
  }
}

function renderEvalSummary(report) {
  const card = document.getElementById('evalSummaryCard');
  if (!report?.summary) return;
  const s = report.summary;

  card.innerHTML = `
    <div class="grid grid-4">
      <div>
        <div style="text-align:center">
          ${renderGauge(s.safety_refusal_rate * 100, 'var(--accent-success)')}
          <div class="stat-label mt-4">Safety Refusal Rate</div>
        </div>
      </div>
      <div>
        <div style="text-align:center">
          ${renderGauge(s.quality_overall * 100, 'var(--accent-primary)')}
          <div class="stat-label mt-4">Quality Score</div>
        </div>
      </div>
      <div>
        <div style="text-align:center">
          ${renderGauge(s.quality_coherence * 100, 'var(--accent-info)')}
          <div class="stat-label mt-4">Coherence</div>
        </div>
      </div>
      <div>
        <div style="text-align:center">
          ${renderGauge(s.quality_informativeness * 100, 'var(--accent-secondary)')}
          <div class="stat-label mt-4">Informativeness</div>
        </div>
      </div>
    </div>
  `;
}

function renderGauge(percent, color) {
  const r = 50;
  const circumference = 2 * Math.PI * r;
  const offset = circumference - (percent / 100) * circumference;
  return `
    <div class="gauge">
      <svg width="120" height="120" viewBox="0 0 120 120">
        <circle class="gauge-bg" cx="60" cy="60" r="${r}" />
        <circle class="gauge-fill" cx="60" cy="60" r="${r}"
          stroke="${color}"
          stroke-dasharray="${circumference}"
          stroke-dashoffset="${offset}" />
      </svg>
      <div class="gauge-value">${Math.round(percent)}%</div>
    </div>
  `;
}

// ══════════════════════════════════════════════════════════════
// Training
// ══════════════════════════════════════════════════════════════

// Show/hide SFT model path based on stage selection
document.getElementById('trainStage')?.addEventListener('change', function() {
  document.getElementById('sftModelGroup').style.display =
    this.value === 'dpo' ? 'block' : 'none';
});

async function startTraining() {
  const stage = document.getElementById('trainStage').value;
  const overrides = {};

  const model = document.getElementById('trainModel').value;
  const quant = document.getElementById('trainQuant').value;
  const maxSteps = document.getElementById('trainMaxSteps').value;
  const lr = document.getElementById('trainLR').value;
  const maxSamples = document.getElementById('trainMaxSamples').value;

  overrides['model.name'] = model;
  overrides['model.quantization'] = quant;

  if (maxSteps && maxSteps !== '-1') overrides['training.max_steps'] = parseInt(maxSteps);
  if (lr) overrides['training.learning_rate'] = parseFloat(lr);
  if (maxSamples) overrides['data.max_samples'] = parseInt(maxSamples);

  const body = { stage, config_overrides: overrides };

  if (stage === 'dpo') {
    const sftModel = document.getElementById('trainSftModel').value;
    if (sftModel) body.sft_model_path = sftModel;
  }

  try {
    const result = await apiPost('/api/train', body);
    showToast(`Training job started: ${result.job_id}`, 'success');
    monitorJob(result.job_id, stage);
  } catch (err) {
    showToast(`Failed to start training: ${err.message}`, 'error');
  }
}

async function startSmokeTest() {
  const stage = document.getElementById('trainStage').value;
  const overrides = {
    'model.name': 'Qwen/Qwen2.5-0.5B-Instruct',
    'model.quantization': 'none',
    'model.use_flash_attention': false,
    'training.max_steps': 5,
    'data.max_samples': 50,
  };

  try {
    const result = await apiPost('/api/train', { stage, config_overrides: overrides });
    showToast(`Smoke test started: ${result.job_id}`, 'success');
    monitorJob(result.job_id, stage);
  } catch (err) {
    showToast(`Failed: ${err.message}`, 'error');
  }
}

function monitorJob(jobId, stage) {
  const jobsList = document.getElementById('jobsList');

  // Create job card
  const card = document.createElement('div');
  card.className = 'card section';
  card.id = `job-${jobId}`;
  card.innerHTML = `
    <div class="card-header">
      <div>
        <div class="card-title">${stage.toUpperCase()} Training — ${jobId}</div>
        <div style="margin-top:4px"><span class="tag tag-cyan">● Running</span></div>
      </div>
      <button class="btn btn-danger btn-sm" onclick="stopJob('${jobId}')">Stop</button>
    </div>
    <div class="terminal">
      <div class="terminal-header">
        <div class="terminal-dot red"></div>
        <div class="terminal-dot yellow"></div>
        <div class="terminal-dot green"></div>
        <span class="terminal-title">Training Logs — ${jobId}</span>
      </div>
      <div class="terminal-body" id="logs-${jobId}">
        <div class="terminal-line">Initializing training...</div>
      </div>
    </div>
  `;

  // Replace empty state or prepend
  if (jobsList.querySelector('.empty-state')) {
    jobsList.innerHTML = '';
  }
  jobsList.prepend(card);

  // Connect WebSocket for live logs
  connectLogSocket(jobId);

  // Also poll for status
  pollJobStatus(jobId);
}

function connectLogSocket(jobId) {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${location.host}/ws/logs/${jobId}`);
  activeWebSockets[jobId] = ws;

  const logsEl = document.getElementById(`logs-${jobId}`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'log' && logsEl) {
      const line = document.createElement('div');
      line.className = 'terminal-line';

      let text = data.message;
      if (text.includes('ERROR') || text.includes('Error')) {
        line.innerHTML = `<span class="error">${escapeHtml(text)}</span>`;
      } else if (text.includes('WARNING') || text.includes('Warning')) {
        line.innerHTML = `<span class="warning">${escapeHtml(text)}</span>`;
      } else if (text.includes('INFO')) {
        line.innerHTML = `<span class="info">${escapeHtml(text)}</span>`;
      } else {
        line.textContent = text;
      }

      logsEl.appendChild(line);
      logsEl.scrollTop = logsEl.scrollHeight;
    }

    if (data.type === 'status') {
      updateJobCard(jobId, data.status);
    }
  };

  ws.onerror = () => console.warn(`WebSocket error for job ${jobId}`);
  ws.onclose = () => delete activeWebSockets[jobId];
}

async function pollJobStatus(jobId) {
  const interval = setInterval(async () => {
    try {
      const job = await apiGet(`/api/train/${jobId}`);
      if (job.status !== 'running') {
        clearInterval(interval);
        updateJobCard(jobId, job.status);
        if (job.status === 'completed') {
          showToast(`Training ${jobId} completed! 🎉`, 'success');
          refreshStatus();
        } else if (job.status === 'failed') {
          showToast(`Training ${jobId} failed ❌`, 'error');
        }
      }
    } catch(e) {
      clearInterval(interval);
    }
  }, 5000);
}

function updateJobCard(jobId, status) {
  const card = document.getElementById(`job-${jobId}`);
  if (!card) return;

  const tagMap = {
    completed: '<span class="tag tag-green">✅ Completed</span>',
    failed: '<span class="tag tag-red">❌ Failed</span>',
    stopped: '<span class="tag tag-amber">⏹ Stopped</span>',
    running: '<span class="tag tag-cyan">● Running</span>',
  };

  const tagContainer = card.querySelector('.card-header div div:last-child');
  if (tagContainer) tagContainer.innerHTML = tagMap[status] || tagMap.running;

  if (status !== 'running') {
    const stopBtn = card.querySelector('.btn-danger');
    if (stopBtn) stopBtn.remove();
  }
}

async function stopJob(jobId) {
  try {
    await apiPost(`/api/train/${jobId}/stop`);
    showToast(`Job ${jobId} stopped`, 'info');
    const ws = activeWebSockets[jobId];
    if (ws) ws.close();
  } catch (err) {
    showToast(`Failed to stop: ${err.message}`, 'error');
  }
}

// ══════════════════════════════════════════════════════════════
// Configuration
// ══════════════════════════════════════════════════════════════

let configCache = {};

async function loadConfig(stage, tabEl) {
  currentConfigTab = stage;

  // Update tab UI
  document.querySelectorAll('.config-tab').forEach(t => t.classList.remove('active'));
  if (tabEl) {
    tabEl.classList.add('active');
  } else {
    document.querySelector(`.config-tab[data-config="${stage}"]`)?.classList.add('active');
  }

  try {
    const config = await apiGet(`/api/configs/${stage}`);
    configCache[stage] = config;
    document.getElementById('configEditor').value = jsyaml_stringify(config);
  } catch (err) {
    document.getElementById('configEditor').value = `# Error loading config: ${err.message}`;
  }
}

async function saveConfig() {
  try {
    const text = document.getElementById('configEditor').value;
    const config = jsyaml_parse(text);
    await apiPost(`/api/configs/${currentConfigTab}`, config);
    showToast(`Config '${currentConfigTab}' saved`, 'success');
  } catch (err) {
    showToast(`Failed to save: ${err.message}`, 'error');
  }
}

function resetConfig() {
  if (configCache[currentConfigTab]) {
    document.getElementById('configEditor').value = jsyaml_stringify(configCache[currentConfigTab]);
    showToast('Config reset to last saved', 'info');
  }
}

// Simple YAML stringify/parse (good enough for display)
function jsyaml_stringify(obj, indent = 0) {
  let result = '';
  const prefix = '  '.repeat(indent);
  for (const [key, value] of Object.entries(obj)) {
    if (value === null || value === undefined) {
      result += `${prefix}${key}: null\n`;
    } else if (typeof value === 'object' && !Array.isArray(value)) {
      result += `${prefix}${key}:\n${jsyaml_stringify(value, indent + 1)}`;
    } else if (Array.isArray(value)) {
      result += `${prefix}${key}:\n`;
      value.forEach(v => result += `${prefix}  - ${JSON.stringify(v)}\n`);
    } else if (typeof value === 'string') {
      result += `${prefix}${key}: "${value}"\n`;
    } else {
      result += `${prefix}${key}: ${value}\n`;
    }
  }
  return result;
}

function jsyaml_parse(text) {
  // Simple YAML-like parser (handles basic key: value, nesting, arrays)
  const obj = {};
  const lines = text.split('\n');
  const stack = [{ obj, indent: -1 }];

  for (const line of lines) {
    const trimmed = line.replace(/#.*$/, '').trimEnd();
    if (!trimmed) continue;

    const indent = line.search(/\S/);
    const match = trimmed.match(/^(\s*)([\w.]+):\s*(.*)?$/);

    if (!match) {
      const arrMatch = trimmed.match(/^(\s*)-\s*(.+)$/);
      if (arrMatch && stack.length > 0) {
        const parent = stack[stack.length - 1];
        const keys = Object.keys(parent.obj);
        const lastKey = keys[keys.length - 1];
        if (lastKey && Array.isArray(parent.obj[lastKey])) {
          parent.obj[lastKey].push(parseValue(arrMatch[2]));
        }
      }
      continue;
    }

    const key = match[2];
    const valueStr = match[3]?.trim();

    // Pop stack to correct level
    while (stack.length > 1 && stack[stack.length - 1].indent >= indent) {
      stack.pop();
    }

    const current = stack[stack.length - 1].obj;

    if (!valueStr) {
      // Could be object or array — peek ahead
      const nextLine = lines[lines.indexOf(line) + 1] || '';
      if (nextLine.trim().startsWith('-')) {
        current[key] = [];
      } else {
        current[key] = {};
        stack.push({ obj: current[key], indent: indent });
      }
    } else {
      current[key] = parseValue(valueStr);
    }
  }

  return stack[0].obj;
}

function parseValue(v) {
  v = v.trim().replace(/^["']|["']$/g, '');
  if (v === 'true') return true;
  if (v === 'false') return false;
  if (v === 'null' || v === 'none') return null;
  if (/^-?\d+$/.test(v)) return parseInt(v);
  if (/^-?\d*\.?\d+([eE][+-]?\d+)?$/.test(v)) return parseFloat(v);
  return v;
}

// ══════════════════════════════════════════════════════════════
// Datasets
// ══════════════════════════════════════════════════════════════

async function loadDatasets() {
  try {
    const datasets = await apiGet('/api/dataset/list');
    const tbody = document.getElementById('datasetTableBody');
    if (datasets.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="text-center" style="padding:24px;color:var(--text-tertiary)">No custom datasets yet</td></tr>';
      return;
    }
    tbody.innerHTML = datasets.map(d => `
      <tr>
        <td><strong>${d.name}</strong></td>
        <td><span class="tag tag-indigo">${d.num_pairs} pairs</span></td>
        <td>${formatBytes(d.size_bytes)}</td>
        <td>${new Date(d.modified).toLocaleDateString()}</td>
      </tr>
    `).join('');
  } catch(e) { /* ok */ }
}

function addPair() {
  pairCounter++;
  const container = document.getElementById('pairsContainer');
  const pair = document.createElement('div');
  pair.className = 'pair-card';
  pair.id = `pair-${pairCounter}`;
  pair.innerHTML = `
    <div class="pair-number">#${pairCounter}</div>
    <button class="remove-pair" onclick="this.parentElement.remove()">×</button>
    <div class="form-group">
      <label class="form-label">Prompt</label>
      <textarea class="form-textarea" placeholder="User's question or instruction..." rows="2" data-field="prompt"></textarea>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label class="form-label" style="color:var(--accent-success)">✅ Chosen (Preferred)</label>
        <textarea class="form-textarea" placeholder="The better response..." rows="3" data-field="chosen"></textarea>
      </div>
      <div class="form-group">
        <label class="form-label" style="color:var(--accent-danger)">❌ Rejected (Dispreferred)</label>
        <textarea class="form-textarea" placeholder="The worse response..." rows="3" data-field="rejected"></textarea>
      </div>
    </div>
  `;
  container.appendChild(pair);
}

async function buildDataset() {
  const pairs = [];
  document.querySelectorAll('.pair-card').forEach(card => {
    const prompt = card.querySelector('[data-field="prompt"]').value;
    const chosen = card.querySelector('[data-field="chosen"]').value;
    const rejected = card.querySelector('[data-field="rejected"]').value;
    if (prompt && chosen && rejected) {
      pairs.push({ prompt, chosen, rejected });
    }
  });

  if (pairs.length === 0) {
    showToast('Add at least one preference pair', 'error');
    return;
  }

  const name = document.getElementById('datasetName').value || 'custom_preferences.jsonl';

  try {
    const result = await apiPost('/api/dataset/build', { pairs, output_name: name });
    showToast(`Dataset built: ${result.num_pairs} pairs saved`, 'success');
    loadDatasets();
  } catch (err) {
    showToast(`Failed: ${err.message}`, 'error');
  }
}

// ══════════════════════════════════════════════════════════════
// Evaluation
// ══════════════════════════════════════════════════════════════

async function runEvaluation() {
  const modelPath = document.getElementById('evalModelPath').value;
  const comparePath = document.getElementById('evalComparePath').value;

  if (!modelPath) {
    showToast('Enter a model path to evaluate', 'error');
    return;
  }

  try {
    const result = await apiPost('/api/evaluate', {
      model_path: modelPath,
      compare_model_path: comparePath || null,
    });
    showToast(`Evaluation started: ${result.job_id}`, 'success');
    monitorJob(result.job_id, 'eval');

    // Poll for results
    const poller = setInterval(async () => {
      try {
        const report = await apiGet('/api/evaluate/report');
        if (report) {
          clearInterval(poller);
          renderEvalResults(report);
        }
      } catch(e) { /* not ready yet */ }
    }, 5000);
  } catch (err) {
    showToast(`Failed: ${err.message}`, 'error');
  }
}

async function loadEvalReport() {
  try {
    const report = await apiGet('/api/evaluate/report');
    renderEvalResults(report);
  } catch(e) { /* no report yet */ }
}

function renderEvalResults(report) {
  const container = document.getElementById('evalResults');
  if (!report?.summary) return;
  const s = report.summary;

  container.innerHTML = `
    <div class="grid grid-2 section">
      <div class="card">
        <div class="card-header">
          <div class="card-title">🛡️ Safety Metrics</div>
          <div class="card-icon green">🛡️</div>
        </div>
        <div style="text-align:center;margin:20px 0">
          ${renderGauge(s.safety_refusal_rate * 100, 'var(--accent-success)')}
          <div class="stat-label mt-4">Refusal Rate on Harmful Prompts</div>
        </div>
        <div class="metric-row">
          <span class="metric-name">Total Prompts Tested</span>
          <span class="metric-value">${report.safety?.total_prompts || '—'}</span>
        </div>
        <div class="metric-row">
          <span class="metric-name">Successfully Refused</span>
          <span class="metric-value" style="color:var(--accent-success)">${report.safety?.refused || '—'}</span>
        </div>
        <div class="metric-row">
          <span class="metric-name">Potentially Complied</span>
          <span class="metric-value" style="color:var(--accent-danger)">${report.safety?.complied || '—'}</span>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title">📊 Quality Metrics</div>
          <div class="card-icon indigo">📊</div>
        </div>
        <div class="grid grid-3" style="margin:20px 0;text-align:center">
          <div>
            ${renderGauge(s.quality_overall * 100, 'var(--accent-primary)')}
            <div class="stat-label mt-4">Overall</div>
          </div>
          <div>
            ${renderGauge(s.quality_coherence * 100, 'var(--accent-info)')}
            <div class="stat-label mt-4">Coherence</div>
          </div>
          <div>
            ${renderGauge(s.quality_informativeness * 100, 'var(--accent-secondary)')}
            <div class="stat-label mt-4">Informativeness</div>
          </div>
        </div>
        <div class="metric-row">
          <span class="metric-name">Prompts Evaluated</span>
          <span class="metric-value">${report.quality?.num_prompts || '—'}</span>
        </div>
      </div>
    </div>

    ${s.win_rate_vs_baseline != null ? `
    <div class="card section">
      <div class="card-header">
        <div class="card-title">🏆 Win Rate vs Baseline</div>
      </div>
      <div style="text-align:center;padding:20px">
        ${renderGauge(s.win_rate_vs_baseline * 100, 'var(--accent-warning)')}
        <div class="stat-label mt-4">Aligned Model Win Rate</div>
      </div>
    </div>
    ` : ''}

    <div class="card">
      <div class="card-header">
        <div class="card-title">📋 Report Details</div>
      </div>
      <div class="metric-row">
        <span class="metric-name">Model Path</span>
        <span class="metric-value font-mono" style="font-size:12px">${report.model_path || '—'}</span>
      </div>
      <div class="metric-row">
        <span class="metric-name">Evaluation Time</span>
        <span class="metric-value">${report.timestamp ? new Date(report.timestamp).toLocaleString() : '—'}</span>
      </div>
    </div>
  `;
}

// ══════════════════════════════════════════════════════════════
// Chat
// ══════════════════════════════════════════════════════════════

async function sendChat() {
  const input = document.getElementById('chatInput');
  const modelPath = document.getElementById('chatModelPath').value;
  const message = input.value.trim();

  if (!message) return;
  if (!modelPath) {
    showToast('Enter a model checkpoint path', 'error');
    return;
  }

  // Hide empty state
  const emptyState = document.getElementById('chatEmptyState');
  if (emptyState) emptyState.remove();

  // Add user message
  addChatMessage(message, 'user');
  input.value = '';

  // Show typing indicator
  const typingEl = addTypingIndicator();

  // Disable input
  const sendBtn = document.getElementById('chatSendBtn');
  sendBtn.disabled = true;
  sendBtn.innerHTML = '<span class="spinner"></span>';

  try {
    const result = await apiPost('/api/chat', {
      message,
      model_path: modelPath,
      max_new_tokens: 512,
      temperature: 0.7,
      top_p: 0.9,
    });

    typingEl.remove();
    addChatMessage(result.response, 'assistant');
  } catch (err) {
    typingEl.remove();
    addChatMessage(`Error: ${err.message}`, 'assistant');
    showToast(`Chat error: ${err.message}`, 'error');
  } finally {
    sendBtn.disabled = false;
    sendBtn.innerHTML = 'Send →';
    input.focus();
  }
}

function addChatMessage(text, role) {
  const messages = document.getElementById('chatMessages');
  const msg = document.createElement('div');
  msg.className = `chat-message ${role}`;
  msg.innerHTML = `
    <div class="chat-sender">${role === 'user' ? 'You' : 'ConcordLM'}</div>
    <div class="chat-bubble">${formatChatText(text)}</div>
  `;
  messages.appendChild(msg);
  messages.scrollTop = messages.scrollHeight;
}

function addTypingIndicator() {
  const messages = document.getElementById('chatMessages');
  const typing = document.createElement('div');
  typing.className = 'chat-message assistant';
  typing.innerHTML = `
    <div class="chat-sender">ConcordLM</div>
    <div class="chat-bubble">
      <div class="chat-typing"><span></span><span></span><span></span></div>
    </div>
  `;
  messages.appendChild(typing);
  messages.scrollTop = messages.scrollHeight;
  return typing;
}

function formatChatText(text) {
  // Basic markdown formatting
  let html = escapeHtml(text);
  // Code blocks
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Line breaks
  html = html.replace(/\n/g, '<br>');
  return html;
}

// ══════════════════════════════════════════════════════════════
// Models
// ══════════════════════════════════════════════════════════════

async function loadModels() {
  try {
    const models = await apiGet('/api/models');
    const container = document.getElementById('modelsList');

    if (models.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="icon">🧠</div>
          <h3>No model checkpoints</h3>
          <p>Train a model to see checkpoints here.</p>
        </div>
      `;
      return;
    }

    container.innerHTML = `
      <div class="grid grid-2">
        ${models.map(m => `
          <div class="card" style="cursor:pointer" onclick="selectModel('${m.path}')">
            <div class="flex items-center justify-between mb-4">
              <span class="tag ${m.stage === 'dpo' ? 'tag-green' : m.stage === 'sft' ? 'tag-indigo' : 'tag-amber'}">
                ${m.stage.toUpperCase()}
              </span>
              ${m.has_adapter ? '<span class="tag tag-cyan">LoRA</span>' : ''}
            </div>
            <div class="card-title">${m.name}</div>
            <div style="font-size:11px;color:var(--text-tertiary);margin-top:6px;font-family:var(--font-mono);word-break:break-all">${m.path}</div>
            <div class="mt-4 flex gap-4">
              <button class="btn btn-sm btn-primary" onclick="event.stopPropagation();navigateToChat('${m.path}')">💬 Chat</button>
              <button class="btn btn-sm btn-secondary" onclick="event.stopPropagation();navigateToEval('${m.path}')">📈 Evaluate</button>
            </div>
          </div>
        `).join('')}
      </div>
    `;
  } catch (err) {
    showToast(`Failed to load models: ${err.message}`, 'error');
  }
}

function selectModel(path) {
  navigator.clipboard?.writeText(path);
  showToast('Model path copied to clipboard', 'info');
}

function navigateToChat(path) {
  document.getElementById('chatModelPath').value = path;
  navigateTo('chat');
}

function navigateToEval(path) {
  document.getElementById('evalModelPath').value = path;
  navigateTo('evaluation');
}

// ══════════════════════════════════════════════════════════════
// Utilities
// ══════════════════════════════════════════════════════════════

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// ══════════════════════════════════════════════════════════════
// Initialize
// ══════════════════════════════════════════════════════════════

refreshStatus();

// Auto-refresh every 15 seconds on dashboard
setInterval(() => {
  if (currentPage === 'dashboard') refreshStatus();
}, 15000);
