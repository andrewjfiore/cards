/* ── Card Crop Tuner — app.js ── */
/* global fetch, EventSource */

const API = '';  // same origin

// ── State ───────────────────────────────────────────────────────────────────
const state = {
    page: 'setup',        // setup | experiment | leaderboard | history
    dataset: null,        // {configured, input_dir, output_root, image_count}
    batch: null,          // current batch being reviewed
    batchItems: [],       // enriched items from GET /api/batches/{id}
    processing: false,
    processProgress: 0,
    configs: [],
    batches: [],
    exploration: 1.0,
    epsilon: 0.0,
    pairwiseMode: false,
    votes: {},            // batch_item_id -> {vote, confidence, reason_tags, pairwise_winner}
};

// ── Helpers ─────────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
    const url = API + path;
    const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        ...opts,
    });
    if (!res.ok) {
        const text = await res.text();
        throw new Error(`API ${res.status}: ${text}`);
    }
    return res.json();
}

function $(sel, el) { return (el || document).querySelector(sel); }
function $$(sel, el) { return [...(el || document).querySelectorAll(sel)]; }

function imgUrl(path) {
    if (!path) return '';
    return API + '/api/image?path=' + encodeURIComponent(path);
}

function esc(s) {
    const d = document.createElement('div');
    d.textContent = s || '';
    return d.innerHTML;
}

function detectorClass(det) {
    return 'config-badge detector-' + (det || 'auto');
}

function configSummary(cfg) {
    const parts = [cfg.detector || 'auto'];
    if (cfg.ocr_refine) parts.push('OCR');
    if (cfg.ml_refine) parts.push('ML');
    if (cfg.padding > 0) parts.push('pad=' + cfg.padding);
    if (cfg.no_resize) parts.push('noresize');
    parts.push('conf=' + cfg.detector_conf);
    if (cfg.ml_refine) parts.push('mlw=' + cfg.ml_weight);
    return parts.join(' | ');
}

// ── Navigation ──────────────────────────────────────────────────────────────
function navigate(page) {
    state.page = page;
    $$('.nav-tab').forEach(t => t.classList.toggle('active', t.dataset.page === page));
    $$('.page-section').forEach(s => s.classList.toggle('hidden', s.id !== 'page-' + page));

    if (page === 'leaderboard') loadLeaderboard();
    if (page === 'history') loadHistory();
    if (page === 'setup') loadDatasetInfo();
}

// ── Setup page ──────────────────────────────────────────────────────────────
async function loadDatasetInfo() {
    try {
        const info = await api('/api/datasets/info');
        state.dataset = info;
        renderSetup();
    } catch (e) {
        console.error(e);
    }
}

async function selectDataset() {
    const input_dir = $('#input-dir').value.trim();
    const output_root = $('#output-dir').value.trim();
    if (!input_dir) return alert('Input directory is required');

    try {
        const res = await api('/api/datasets/select', {
            method: 'POST',
            body: JSON.stringify({ input_dir, output_root }),
        });
        state.dataset = { configured: true, ...res };
        renderSetup();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

async function loadSettings() {
    try {
        const s = await api('/api/settings');
        state.exploration = s.exploration;
        state.epsilon = s.epsilon;
    } catch (e) { /* ignore */ }
}

async function saveSettings() {
    try {
        await api('/api/settings', {
            method: 'POST',
            body: JSON.stringify({
                exploration: state.exploration,
                epsilon: state.epsilon,
            }),
        });
    } catch (e) {
        console.error(e);
    }
}

async function resetLearning() {
    if (!confirm('Reset all arms and votes? Configs will be preserved.')) return;
    try {
        await api('/api/reset', { method: 'POST' });
        alert('Learning state reset.');
        loadLeaderboard();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

function renderSetup() {
    const ds = state.dataset;
    const statusEl = $('#dataset-status');

    if (ds && ds.configured) {
        statusEl.innerHTML = `
            <span class="dot ok"></span>
            <span>Dataset: <strong>${esc(ds.input_dir)}</strong> — ${ds.image_count} images</span>
        `;
        $('#start-batch-btn').disabled = false;
    } else {
        statusEl.innerHTML = '<span class="dot"></span><span>No dataset selected</span>';
        $('#start-batch-btn').disabled = true;
    }
}

// ── Batch experiment ────────────────────────────────────────────────────────
async function startBatch() {
    if (state.processing) return;
    state.votes = {};

    const mode = state.pairwiseMode ? 'pairwise' : 'single';

    try {
        const batch = await api('/api/batches/start', {
            method: 'POST',
            body: JSON.stringify({ mode }),
        });
        state.batch = batch;
        state.batchItems = batch.images.map(img => ({
            ...img,
            output_path: '',
            debug_path: '',
            status: 'pending',
            strategy: '',
            output_path_b: '',
            debug_path_b: '',
            status_b: null,
            strategy_b: null,
        }));
        state.processing = true;
        state.processProgress = 0;

        navigate('experiment');
        renderExperiment();
        processBatch(batch.batch_id);
    } catch (e) {
        alert('Error starting batch: ' + e.message);
    }
}

function processBatch(batchId) {
    const progressEl = $('#batch-progress');
    const progressBar = $('#batch-progress-fill');
    const progressText = $('#batch-progress-text');

    fetch(API + `/api/batches/${batchId}/process`, { method: 'POST' })
        .then(response => {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            function pump() {
                return reader.read().then(({ done, value }) => {
                    if (done) {
                        state.processing = false;
                        renderExperiment();
                        return;
                    }
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.done) {
                                state.processing = false;
                                renderExperiment();
                                return;
                            }
                            handleProcessEvent(data);
                        } catch (e) { /* parse error, skip */ }
                    }
                    return pump();
                });
            }
            return pump();
        })
        .catch(e => {
            console.error('Processing error:', e);
            state.processing = false;
            renderExperiment();
        });
}

function handleProcessEvent(data) {
    const idx = data.index;
    const total = data.total;
    state.processProgress = (idx + 1) / total;

    // Update the item
    if (state.batchItems[idx]) {
        const item = state.batchItems[idx];
        const ra = data.result_a;
        item.output_path = ra.output_path;
        item.debug_path = ra.debug_path;
        item.status = ra.status;
        item.strategy = ra.strategy;

        if (data.result_b) {
            const rb = data.result_b;
            item.output_path_b = rb.output_path;
            item.debug_path_b = rb.debug_path;
            item.status_b = rb.status;
            item.strategy_b = rb.strategy;
        }
    }

    renderProgressBar();
    renderImageCard(idx);
}

function renderProgressBar() {
    const pct = Math.round(state.processProgress * 100);
    const fill = $('#batch-progress-fill');
    const text = $('#batch-progress-text');
    if (fill) fill.style.width = pct + '%';
    if (text) text.textContent = state.processing
        ? `Processing... ${pct}%`
        : `Complete — ${state.batchItems.length} images`;
}

function renderExperiment() {
    const container = $('#experiment-content');
    if (!state.batch) {
        container.innerHTML = '<p class="dim">No active batch. Start one from Setup.</p>';
        return;
    }

    const b = state.batch;
    const cfg = b.config;
    const mode = b.mode || 'single';

    let headerHtml = `
        <div class="batch-header">
            <div>
                <h2>Batch #${b.batch_id}</h2>
                <div class="flex-row mt-8">
                    <span class="${detectorClass(cfg.detector)}">${esc(cfg.detector)}</span>
                    <span class="config-badge">${esc(configSummary(cfg))}</span>
                    ${mode === 'pairwise' && b.config_b
                        ? `<span class="dim">vs</span>
                           <span class="${detectorClass(b.config_b.detector)}">${esc(b.config_b.detector)}</span>
                           <span class="config-badge">${esc(configSummary(b.config_b))}</span>`
                        : ''}
                </div>
            </div>
            <div class="flex-row">
                <button class="btn btn-success" onclick="finalizeBatch()" id="finalize-btn">Finalize Batch</button>
            </div>
        </div>
        <div class="progress-bar"><div class="fill" id="batch-progress-fill" style="width:0%"></div></div>
        <div class="progress-text" id="batch-progress-text">Waiting...</div>
    `;

    let itemsHtml = '<div id="batch-items">';
    for (let i = 0; i < state.batchItems.length; i++) {
        itemsHtml += `<div id="item-${i}">${renderImageCardHtml(i)}</div>`;
    }
    itemsHtml += '</div>';

    container.innerHTML = headerHtml + itemsHtml;
    renderProgressBar();
}

function renderImageCard(idx) {
    const el = $(`#item-${idx}`);
    if (el) el.innerHTML = renderImageCardHtml(idx);
}

function renderImageCardHtml(idx) {
    const item = state.batchItems[idx];
    if (!item) return '';

    const vote = state.votes[item.id];
    const voteClass = vote ? `voted voted-${vote.vote}` : '';
    const filename = item.filename.split('/').pop().split('\\').pop();
    const isPairwise = state.batch && state.batch.mode === 'pairwise';

    let imagesHtml;
    if (isPairwise) {
        imagesHtml = `
            <div class="image-pair triple">
                <div class="image-slot">
                    <div class="slot-label">Original</div>
                    ${item.filename
                        ? `<img src="${imgUrl(item.filename)}" onclick="openLightbox(this.src)" alt="original">`
                        : '<div class="fail-placeholder">Pending...</div>'}
                </div>
                <div class="image-slot">
                    <div class="slot-label">Config A (${state.batch.config_id})</div>
                    ${item.output_path
                        ? `<img src="${imgUrl(item.output_path)}" onclick="openLightbox(this.src)" alt="config A">`
                        : item.status !== 'pending'
                            ? `<div class="fail-placeholder">${esc(item.status)}</div>`
                            : '<div class="fail-placeholder">Processing...</div>'}
                </div>
                <div class="image-slot">
                    <div class="slot-label">Config B (${state.batch.config_id_b})</div>
                    ${item.output_path_b
                        ? `<img src="${imgUrl(item.output_path_b)}" onclick="openLightbox(this.src)" alt="config B">`
                        : item.status_b
                            ? `<div class="fail-placeholder">${esc(item.status_b)}</div>`
                            : '<div class="fail-placeholder">Processing...</div>'}
                </div>
            </div>
        `;
    } else {
        const hasDebug = item.debug_path;
        imagesHtml = `
            <div class="image-pair${hasDebug ? ' triple' : ''}">
                <div class="image-slot">
                    <div class="slot-label">Before</div>
                    ${item.filename
                        ? `<img src="${imgUrl(item.filename)}" onclick="openLightbox(this.src)" alt="before">`
                        : '<div class="fail-placeholder">No image</div>'}
                </div>
                <div class="image-slot">
                    <div class="slot-label">After</div>
                    ${item.output_path
                        ? `<img src="${imgUrl(item.output_path)}" onclick="openLightbox(this.src)" alt="after">`
                        : item.status !== 'pending'
                            ? `<div class="fail-placeholder">${esc(item.status)}</div>`
                            : '<div class="fail-placeholder">Processing...</div>'}
                </div>
                ${hasDebug ? `
                <div class="image-slot">
                    <div class="slot-label">Debug Overlay</div>
                    <img src="${imgUrl(item.debug_path)}" onclick="openLightbox(this.src)" alt="debug">
                </div>` : ''}
            </div>
        `;
    }

    let voteHtml;
    if (isPairwise) {
        voteHtml = `
            <div class="vote-controls" data-item-id="${item.id}">
                <button class="vote-btn pick-a ${vote && vote.pairwise_winner === 'a' ? 'selected' : ''}"
                        onclick="submitPairwiseVote(${item.id}, 'a')">A is better</button>
                <button class="vote-btn pick-b ${vote && vote.pairwise_winner === 'b' ? 'selected' : ''}"
                        onclick="submitPairwiseVote(${item.id}, 'b')">B is better</button>
                <button class="vote-btn tie ${vote && vote.pairwise_winner === 'tie' ? 'selected' : ''}"
                        onclick="submitPairwiseVote(${item.id}, 'tie')">Tie</button>
                <button class="vote-btn skip ${vote && vote.vote === 'skip' ? 'selected' : ''}"
                        onclick="submitVote(${item.id}, 'skip')">Skip</button>
                ${renderConfidenceSelector(item.id, vote)}
            </div>
        `;
    } else {
        voteHtml = `
            <div class="vote-controls" data-item-id="${item.id}">
                <button class="vote-btn up ${vote && vote.vote === 'up' ? 'selected' : ''}"
                        onclick="submitVote(${item.id}, 'up')">Thumbs Up</button>
                <button class="vote-btn down ${vote && vote.vote === 'down' ? 'selected' : ''}"
                        onclick="submitVote(${item.id}, 'down')">Thumbs Down</button>
                <button class="vote-btn uncertain ${vote && vote.vote === 'uncertain' ? 'selected' : ''}"
                        onclick="submitVote(${item.id}, 'uncertain')">Uncertain</button>
                <button class="vote-btn failure ${vote && vote.vote === 'failure' ? 'selected' : ''}"
                        onclick="submitVote(${item.id}, 'failure')">Failure</button>
                <button class="vote-btn skip ${vote && vote.vote === 'skip' ? 'selected' : ''}"
                        onclick="submitVote(${item.id}, 'skip')">Skip</button>
                ${renderConfidenceSelector(item.id, vote)}
                ${vote && vote.vote === 'down' ? renderReasonTags(item.id, vote) : ''}
            </div>
        `;
    }

    const strategyInfo = item.strategy
        ? `<span class="mono dim ml-8">${esc(item.strategy)}</span>` : '';

    return `
        <div class="image-review-card ${voteClass}">
            <div class="flex-between mb-8">
                <span class="image-filename">${idx + 1}/${state.batchItems.length} — ${esc(filename)}</span>
                ${strategyInfo}
            </div>
            ${imagesHtml}
            ${item.status !== 'pending' ? voteHtml : ''}
        </div>
    `;
}

function renderConfidenceSelector(itemId, vote) {
    const current = vote ? vote.confidence : 'sure';
    return `
        <div class="confidence-selector">
            <span class="text-xs dim">Conf:</span>
            ${['sure', 'maybe', 'unsure'].map(c => `
                <button class="conf-btn ${current === c ? 'selected' : ''}"
                        onclick="setConfidence(${itemId}, '${c}')">${c}</button>
            `).join('')}
        </div>
    `;
}

const REASON_TAGS = ['cut off edge', 'too much background', 'wrong object',
                     'bad perspective', 'wrong orientation', 'blurred output', 'other'];

function renderReasonTags(itemId, vote) {
    const selected = vote ? (vote.reason_tags || []) : [];
    return `
        <div class="reason-tags mt-8">
            <span class="text-xs dim">Issues:</span>
            ${REASON_TAGS.map(tag => `
                <button class="tag-btn ${selected.includes(tag) ? 'selected' : ''}"
                        onclick="toggleReasonTag(${itemId}, '${tag}')">${tag}</button>
            `).join('')}
        </div>
    `;
}

// ── Vote actions ────────────────────────────────────────────────────────────
async function submitVote(itemId, voteType) {
    const existing = state.votes[itemId] || { confidence: 'sure', reason_tags: [] };
    const voteData = {
        ...existing,
        vote: voteType,
        batch_item_id: itemId,
    };
    state.votes[itemId] = voteData;

    try {
        await api('/api/votes', {
            method: 'POST',
            body: JSON.stringify(voteData),
        });
    } catch (e) {
        console.error('Vote error:', e);
    }

    // Re-render the card
    const idx = state.batchItems.findIndex(i => i.id === itemId);
    if (idx >= 0) renderImageCard(idx);
}

async function submitPairwiseVote(itemId, winner) {
    const existing = state.votes[itemId] || { confidence: 'sure', reason_tags: [] };
    const voteData = {
        ...existing,
        vote: winner === 'tie' ? 'uncertain' : 'up',
        pairwise_winner: winner,
        batch_item_id: itemId,
    };
    state.votes[itemId] = voteData;

    try {
        await api('/api/votes', {
            method: 'POST',
            body: JSON.stringify(voteData),
        });
    } catch (e) {
        console.error('Vote error:', e);
    }

    const idx = state.batchItems.findIndex(i => i.id === itemId);
    if (idx >= 0) renderImageCard(idx);
}

function setConfidence(itemId, conf) {
    const existing = state.votes[itemId];
    if (!existing) return;
    existing.confidence = conf;
    // Re-submit with updated confidence
    submitVote(itemId, existing.vote);
}

function toggleReasonTag(itemId, tag) {
    const existing = state.votes[itemId];
    if (!existing) return;
    const tags = existing.reason_tags || [];
    const idx = tags.indexOf(tag);
    if (idx >= 0) tags.splice(idx, 1);
    else tags.push(tag);
    existing.reason_tags = tags;
    submitVote(itemId, existing.vote);
}

async function finalizeBatch() {
    if (!state.batch) return;

    try {
        const res = await api(`/api/batches/${state.batch.batch_id}/finalize`, {
            method: 'POST',
        });
        alert(`Batch #${state.batch.batch_id} finalized.\n` +
              `Ups: ${res.summary.ups}, Downs: ${res.summary.downs}, ` +
              `Failures: ${res.summary.failures}`);
        state.batch = null;
        state.batchItems = [];
        state.votes = {};
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// ── Leaderboard ─────────────────────────────────────────────────────────────
async function loadLeaderboard() {
    try {
        state.configs = await api('/api/configs');
        renderLeaderboard();
    } catch (e) {
        console.error(e);
    }
}

function renderLeaderboard() {
    const container = $('#leaderboard-content');
    const configs = state.configs;

    if (!configs.length) {
        container.innerHTML = '<p class="dim">No configs yet.</p>';
        return;
    }

    // Slicing filters
    const detectors = [...new Set(configs.map(c => c.config.detector))];
    const ocrOptions = [true, false];
    const mlOptions = [true, false];

    let filterHtml = `
        <div class="slice-filters">
            <div>
                <label>Detector</label>
                <select id="filter-detector" onchange="filterLeaderboard()">
                    <option value="">All</option>
                    ${detectors.map(d => `<option value="${d}">${d}</option>`).join('')}
                </select>
            </div>
            <div>
                <label>OCR</label>
                <select id="filter-ocr" onchange="filterLeaderboard()">
                    <option value="">All</option>
                    <option value="true">On</option>
                    <option value="false">Off</option>
                </select>
            </div>
            <div>
                <label>ML/CLIP</label>
                <select id="filter-ml" onchange="filterLeaderboard()">
                    <option value="">All</option>
                    <option value="true">On</option>
                    <option value="false">Off</option>
                </select>
            </div>
        </div>
    `;

    // Slice summary cards
    let sliceHtml = '<div class="flex-row-wrap mb-16" id="slice-cards"></div>';

    container.innerHTML = filterHtml + sliceHtml +
        '<div id="leaderboard-table-wrapper"></div>';

    filterLeaderboard();
}

function filterLeaderboard() {
    const det = ($('#filter-detector') || {}).value || '';
    const ocr = ($('#filter-ocr') || {}).value || '';
    const ml = ($('#filter-ml') || {}).value || '';

    let filtered = state.configs;
    if (det) filtered = filtered.filter(c => c.config.detector === det);
    if (ocr) filtered = filtered.filter(c => String(c.config.ocr_refine) === ocr);
    if (ml) filtered = filtered.filter(c => String(c.config.ml_refine) === ml);

    // Slice cards
    const totalVotes = filtered.reduce((s, c) => s + c.total_votes, 0);
    const avgMean = filtered.length
        ? (filtered.reduce((s, c) => s + c.mean, 0) / filtered.length)
        : 0;
    const totalFailures = filtered.reduce((s, c) => s + c.failures, 0);
    const failRate = totalVotes > 0 ? (totalFailures / totalVotes) : 0;

    const sliceEl = $('#slice-cards');
    if (sliceEl) {
        sliceEl.innerHTML = `
            <div class="slice-card">
                <span class="slice-label">Configs</span>
                <span class="slice-value">${filtered.length}</span>
            </div>
            <div class="slice-card">
                <span class="slice-label">Total Votes</span>
                <span class="slice-value">${totalVotes}</span>
            </div>
            <div class="slice-card">
                <span class="slice-label">Avg Win Rate</span>
                <span class="slice-value">${(avgMean * 100).toFixed(1)}%</span>
            </div>
            <div class="slice-card">
                <span class="slice-label">Failure Rate</span>
                <span class="slice-value ${failRate > 0.2 ? 'failure-rate' : ''}">${(failRate * 100).toFixed(1)}%</span>
            </div>
        `;
    }

    // Table
    const tableEl = $('#leaderboard-table-wrapper');
    if (!tableEl) return;

    let rows = filtered.map((c, i) => {
        const failRate = c.total_votes > 0
            ? (c.failures / c.total_votes * 100).toFixed(1) + '%'
            : '-';
        const barWidth = Math.max(2, Math.round(c.mean * 200));

        return `
            <tr>
                <td>${i + 1}</td>
                <td><span class="${detectorClass(c.config.detector)}">${c.config.detector}</span></td>
                <td class="mono text-xs">${esc(configSummary(c.config))}</td>
                <td>
                    <span class="mean-bar" style="width:${barWidth}px"></span>
                    <span class="mono">${(c.mean * 100).toFixed(1)}%</span>
                </td>
                <td>${c.total_votes || '<span class="no-data">0</span>'}</td>
                <td>${c.ups}</td>
                <td>${c.downs}</td>
                <td class="${c.failures > 0 ? 'failure-rate' : ''}">${failRate}</td>
                <td class="mono text-xs">${c.last_tested_at ? c.last_tested_at.slice(0, 16) : '-'}</td>
                <td class="mono text-xs">a=${c.alpha.toFixed(1)} b=${c.beta.toFixed(1)}</td>
            </tr>
        `;
    }).join('');

    tableEl.innerHTML = `
        <table class="leaderboard-table">
            <thead>
                <tr>
                    <th>#</th><th>Detector</th><th>Config</th><th>Win Rate</th>
                    <th>Votes</th><th>Up</th><th>Down</th><th>Fail%</th>
                    <th>Last Tested</th><th>Arms</th>
                </tr>
            </thead>
            <tbody>${rows || '<tr><td colspan="10" class="dim">No data yet</td></tr>'}</tbody>
        </table>
    `;
}

// ── History ─────────────────────────────────────────────────────────────────
async function loadHistory() {
    try {
        state.batches = await api('/api/batches');
        renderHistory();
    } catch (e) {
        console.error(e);
    }
}

function renderHistory() {
    const container = $('#history-content');
    const batches = state.batches;

    if (!batches.length) {
        container.innerHTML = '<p class="dim">No batches yet.</p>';
        return;
    }

    const items = batches.map(b => {
        const vs = b.vote_summary || {};
        return `
            <div class="history-item">
                <div class="batch-num">#${b.id}</div>
                <div class="batch-meta">
                    <div>
                        <span class="${detectorClass(b.config.detector)}">${b.config.detector}</span>
                        <span class="mono dim">${esc(configSummary(b.config))}</span>
                    </div>
                    <div class="text-xs dim">
                        ${b.started_at ? b.started_at.slice(0, 19) : ''} — ${b.image_count} images
                        ${b.finished_at ? ' — finalized' : ' — in progress'}
                    </div>
                </div>
                <div class="batch-votes">
                    ${vs.ups ? `<span class="vote-pill up">${vs.ups} up</span>` : ''}
                    ${vs.downs ? `<span class="vote-pill down">${vs.downs} dn</span>` : ''}
                    ${vs.failures ? `<span class="vote-pill failure">${vs.failures} fail</span>` : ''}
                    ${vs.total ? '' : '<span class="dim text-xs">no votes</span>'}
                </div>
                <button class="btn btn-sm" onclick="viewBatch(${b.id})">View</button>
            </div>
        `;
    }).join('');

    container.innerHTML = items;
}

async function viewBatch(batchId) {
    try {
        const data = await api(`/api/batches/${batchId}`);
        state.batch = {
            batch_id: data.id,
            config_id: data.config_id,
            config_id_b: data.config_id_b,
            config: data.config,
            config_b: data.config_b,
            mode: data.mode,
        };
        state.batchItems = data.items.map(item => ({
            ...item,
            id: item.id,
        }));
        // Restore votes
        state.votes = {};
        for (const item of data.items) {
            if (item.votes && item.votes.length > 0) {
                const lastVote = item.votes[item.votes.length - 1];
                state.votes[item.id] = {
                    vote: lastVote.vote,
                    confidence: lastVote.confidence,
                    reason_tags: JSON.parse(lastVote.reason_tags || '[]'),
                    pairwise_winner: lastVote.pairwise_winner,
                };
            }
        }
        state.processing = false;
        state.processProgress = 1.0;
        navigate('experiment');
        renderExperiment();
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

// ── Lightbox ────────────────────────────────────────────────────────────────
function openLightbox(src) {
    const overlay = document.createElement('div');
    overlay.className = 'lightbox-overlay';
    overlay.onclick = () => overlay.remove();
    overlay.innerHTML = `<img src="${src}" alt="zoomed">`;
    document.body.appendChild(overlay);
}

// ── Export ───────────────────────────────────────────────────────────────────
function exportLeaderboard() {
    window.open(API + '/api/exports/leaderboard.json', '_blank');
}

function exportVotes() {
    window.open(API + '/api/exports/votes.csv', '_blank');
}

// ── Init ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Nav tabs
    $$('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => navigate(tab.dataset.page));
    });

    // Setup button
    $('#btn-select-dataset').addEventListener('click', selectDataset);
    $('#start-batch-btn').addEventListener('click', startBatch);
    $('#btn-reset').addEventListener('click', resetLearning);
    $('#btn-export-lb').addEventListener('click', exportLeaderboard);
    $('#btn-export-csv').addEventListener('click', exportVotes);

    // Pairwise toggle
    $('#pairwise-toggle').addEventListener('change', function() {
        state.pairwiseMode = this.checked;
    });

    // Exploration slider
    const explorationSlider = $('#exploration-slider');
    const explorationValue = $('#exploration-value');
    explorationSlider.addEventListener('input', function() {
        state.exploration = parseFloat(this.value);
        explorationValue.textContent = this.value;
    });
    explorationSlider.addEventListener('change', () => saveSettings());

    // Epsilon slider
    const epsilonSlider = $('#epsilon-slider');
    const epsilonValue = $('#epsilon-value');
    epsilonSlider.addEventListener('input', function() {
        state.epsilon = parseFloat(this.value);
        epsilonValue.textContent = this.value;
    });
    epsilonSlider.addEventListener('change', () => saveSettings());

    // Load initial data
    loadDatasetInfo();
    loadSettings().then(() => {
        if (explorationSlider) {
            explorationSlider.value = state.exploration;
            explorationValue.textContent = state.exploration;
        }
        if (epsilonSlider) {
            epsilonSlider.value = state.epsilon;
            epsilonValue.textContent = state.epsilon;
        }
    });

    navigate('setup');
});
