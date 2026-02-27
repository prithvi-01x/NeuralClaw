/**
 * NeuralClaw Web UI — Full Admin Panel Application
 * Handles chat, config editing, skills management, env vars, and system info.
 */

class NeuralClawApp {
    constructor() {
        this.ws = null;
        this.sessionId = null;
        this.gatewayUrl = null;
        this.reconnectTimer = null;
        this.reconnectDelay = 1000;
        this.currentTab = 'chat';
        this.configData = {};
        this.envData = [];
        this.pendingConfirm = null;

        this._bindDom();
        this._bindEvents();
        this._autoConnect();
    }

    // ── DOM Bindings ──────────────────────────────────────────────────────

    _bindDom() {
        this.els = {
            messages: document.getElementById('messages'),
            welcomeScreen: document.getElementById('welcome-screen'),
            input: document.getElementById('user-input'),
            sendBtn: document.getElementById('send-btn'),
            connectionBadge: document.getElementById('connection-badge'),
            connectionText: document.getElementById('connection-text'),
            topbarStatus: document.getElementById('topbar-status'),
            topbarTitle: document.getElementById('topbar-title'),
            sessionId: document.getElementById('session-id'),
            statsDisplay: document.getElementById('stats-display'),
            capsDisplay: document.getElementById('caps-display'),
            toastContainer: document.getElementById('toast-container'),
            configEditor: document.getElementById('config-editor'),
            configStatus: document.getElementById('config-status'),
            skillsList: document.getElementById('skills-list'),
            envList: document.getElementById('env-list'),
            systemInfo: document.getElementById('system-info'),
            confirmDialog: document.getElementById('confirm-dialog'),
            confirmSkill: document.getElementById('confirm-skill'),
            confirmReason: document.getElementById('confirm-reason'),
            confirmArgs: document.getElementById('confirm-args'),
        };
    }

    _bindEvents() {
        // Send message
        this.els.sendBtn.addEventListener('click', () => this._send());
        this.els.input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this._send(); }
        });
        this.els.input.addEventListener('input', () => {
            this.els.input.style.height = 'auto';
            this.els.input.style.height = Math.min(this.els.input.scrollHeight, 120) + 'px';
        });

        // Tab navigation
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', () => this._switchTab(tab.dataset.tab));
        });

        // Quick action chips
        document.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', () => {
                this.els.input.value = chip.dataset.msg;
                this._send();
            });
        });

        // Trust level buttons
        document.querySelectorAll('.trust-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.trust-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this._sendMsg('session.trust', { level: btn.dataset.level });
            });
        });

        // Sidebar buttons
        document.getElementById('btn-clear-chat').addEventListener('click', () => this._clearChat());
        document.getElementById('btn-reconnect').addEventListener('click', () => this.connect());

        // Config save
        document.getElementById('btn-save-config').addEventListener('click', () => this._saveConfig());

        // Skills reload
        document.getElementById('btn-reload-skills').addEventListener('click', () => this._reloadSkills());

        // Add env var
        document.getElementById('btn-add-env').addEventListener('click', () => this._showEnvModal());

        // Confirm dialog
        document.getElementById('btn-approve').addEventListener('click', () => this._respond_confirm(true));
        document.getElementById('btn-deny').addEventListener('click', () => this._respond_confirm(false));

        // Hamburger menu
        document.getElementById('hamburger').addEventListener('click', () => {
            document.getElementById('sidebar').classList.toggle('open');
        });
    }

    // ── Tab Navigation ────────────────────────────────────────────────────

    _switchTab(tabName) {
        this.currentTab = tabName;

        // Update nav
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update content
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(`tab-${tabName}`).classList.add('active');

        // Update title
        const titles = { chat: 'NeuralClaw', settings: 'Settings', skills: 'Skills', env: 'Environment', system: 'System' };
        this.els.topbarTitle.textContent = titles[tabName] || 'NeuralClaw';

        // Toggle session info visibility
        document.getElementById('session-info').style.display = tabName === 'chat' ? 'flex' : 'none';

        // Load data for admin tabs
        if (tabName === 'settings') this._loadConfig();
        if (tabName === 'skills') this._loadSkills();
        if (tabName === 'env') this._loadEnv();
        if (tabName === 'system') this._loadSystemInfo();

        // Close mobile sidebar
        document.getElementById('sidebar').classList.remove('open');
    }

    // ── WebSocket Connection ──────────────────────────────────────────────

    _autoConnect() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = location.hostname || '127.0.0.1';
        const wsPort = '9090';
        this.gatewayUrl = `${protocol}//${host}:${wsPort}`;
        this.connect();
    }

    connect() {
        this._updateStatus('connecting');
        try {
            this.ws = new WebSocket(this.gatewayUrl);
            this.ws.onopen = () => this._onOpen();
            this.ws.onclose = () => this._onClose();
            this.ws.onerror = () => { };
            this.ws.onmessage = (e) => this._onMessage(e);
        } catch (err) {
            this._updateStatus('disconnected');
        }
    }

    _onOpen() {
        this._updateStatus('connected');
        this.reconnectDelay = 1000;
        this._toast('Connected to gateway', 'success');
        // Create session
        this._sendMsg('session.create', { trust_level: 'low' });
    }

    _onClose() {
        this._updateStatus('disconnected');
        this.ws = null;
        // Auto-reconnect
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = setTimeout(() => {
            this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, 10000);
            this.connect();
        }, this.reconnectDelay);
    }

    _onMessage(event) {
        let msg;
        try { msg = JSON.parse(event.data); } catch { return; }

        const kind = msg.data?.kind;

        switch (msg.type) {
            case 'session.created':
                this.sessionId = msg.session_id;
                this.els.sessionId.textContent = this.sessionId?.slice(0, 12) || '—';
                break;

            case 'response':
                if (kind === 'config') this._renderConfig(msg.data.config);
                else if (kind === 'config_saved') { this._renderConfig(msg.data.config); this._toast('Config saved!', 'success'); this.els.configStatus.textContent = '✓ Saved'; }
                else if (kind === 'env_list') this._renderEnv(msg.data.env_vars);
                else if (kind === 'env_saved') { this._toast(`Updated ${msg.data.key}`, 'success'); this._loadEnv(); }
                else if (kind === 'skills_reloaded') { this._toast(`Skills reloaded: ${msg.data.new_count} active`, 'success'); this._loadSkills(); }
                else if (kind === 'system_info') this._renderSystemInfo(msg.data);
                else if (kind === 'status' && msg.data.skills) this._renderSkills(msg.data.skills);
                else this._handleChatResponse(msg);
                break;

            case 'confirm_request':
                this._showConfirmDialog(msg.data);
                break;

            case 'session.updated':
                this._toast('Session updated', 'info');
                break;

            case 'error':
                this._toast(msg.data?.message || 'Error', 'error');
                this._removeThinking();
                break;
        }
    }

    // ── Chat ──────────────────────────────────────────────────────────────

    _send() {
        const text = this.els.input.value.trim();
        if (!text || !this.ws) return;

        this.els.input.value = '';
        this.els.input.style.height = 'auto';
        this.els.welcomeScreen?.remove();

        this._addMessage('user', text);

        if (text.startsWith('/run ')) {
            this._sendMsg('run', { goal: text.slice(5) });
        } else {
            this._sendMsg('ask', { message: text });
        }
        this._addThinking();
    }

    _handleChatResponse(msg) {
        this._removeThinking();
        const text = msg.data?.text || '';
        if (!text) return;

        const isFinal = msg.data?.is_final !== false;
        const kind = msg.data?.kind || 'text';

        if (kind === 'tool_result' || kind === 'progress') {
            this._addMessage('assistant', text, 'tool');
        } else {
            this._addMessage('assistant', text);
        }

        // Update stats
        if (msg.data?.steps_taken !== undefined) {
            this.els.statsDisplay.textContent = `Steps: ${msg.data.steps_taken} · ${Math.round(msg.data.duration_ms || 0)}ms`;
        }
    }

    _addMessage(role, text, type = '') {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        const avatar = role === 'user' ? '👤' : '🐾';
        const rendered = role === 'assistant' ? (typeof marked !== 'undefined' ? marked.parse(text) : text) : this._escapeHtml(text);

        div.innerHTML = `
      <div class="message-avatar">${avatar}</div>
      <div class="message-content ${type}">${rendered}</div>
    `;
        this.els.messages.appendChild(div);
        this.els.messages.scrollTop = this.els.messages.scrollHeight;
    }

    _addThinking() {
        this._removeThinking();
        const div = document.createElement('div');
        div.className = 'message assistant';
        div.id = 'thinking';
        div.innerHTML = `<div class="message-avatar">🐾</div><div class="thinking"><span></span><span></span><span></span></div>`;
        this.els.messages.appendChild(div);
        this.els.messages.scrollTop = this.els.messages.scrollHeight;
    }

    _removeThinking() {
        document.getElementById('thinking')?.remove();
    }

    _clearChat() {
        this.els.messages.innerHTML = '';
        this._toast('Chat cleared', 'info');
    }

    // ── Confirmation Dialog ───────────────────────────────────────────────

    _showConfirmDialog(data) {
        this.pendingConfirm = data;
        this.els.confirmSkill.textContent = `Skill: ${data.skill_name} (${data.risk_level})`;
        this.els.confirmReason.textContent = data.reason || data.text || '';
        this.els.confirmArgs.textContent = JSON.stringify(data.arguments || {}, null, 2);
        this.els.confirmDialog.classList.remove('hidden');
    }

    _respond_confirm(approved) {
        if (this.pendingConfirm) {
            this._sendMsg('confirm', {
                tool_call_id: this.pendingConfirm.tool_call_id,
                approved: approved,
            });
        }
        this.els.confirmDialog.classList.add('hidden');
        this.pendingConfirm = null;
    }

    // ── Config Editor ─────────────────────────────────────────────────────

    _loadConfig() {
        this._sendMsg('config.get', {});
    }

    _renderConfig(config) {
        this.configData = config;
        const editor = this.els.configEditor;
        editor.innerHTML = '';

        for (const [section, values] of Object.entries(config)) {
            const sectionDiv = document.createElement('div');
            sectionDiv.className = 'config-section';
            sectionDiv.innerHTML = `<div class="config-section-title">${section}</div>`;

            if (typeof values === 'object' && values !== null && !Array.isArray(values)) {
                for (const [key, val] of Object.entries(values)) {
                    sectionDiv.appendChild(this._createConfigField(section, key, val));
                }
            } else {
                sectionDiv.appendChild(this._createConfigField('', section, values));
            }

            editor.appendChild(sectionDiv);
        }

        this.els.configStatus.textContent = '';
    }

    _createConfigField(section, key, value) {
        const field = document.createElement('div');
        field.className = 'config-field';

        const label = document.createElement('label');
        label.textContent = section ? `${key}` : key;

        let input;
        if (typeof value === 'boolean') {
            input = document.createElement('select');
            input.innerHTML = `<option value="true" ${value ? 'selected' : ''}>true</option><option value="false" ${!value ? 'selected' : ''}>false</option>`;
        } else if (typeof value === 'object' && value !== null) {
            input = document.createElement('input');
            input.type = 'text';
            input.value = JSON.stringify(value);
            input.title = 'JSON value';
        } else {
            input = document.createElement('input');
            input.type = typeof value === 'number' ? 'number' : 'text';
            input.value = value ?? '';
        }

        input.dataset.section = section;
        input.dataset.key = key;
        input.dataset.originalType = typeof value;
        input.className = 'config-input';

        field.appendChild(label);
        field.appendChild(input);
        return field;
    }

    _saveConfig() {
        const inputs = document.querySelectorAll('.config-input');
        const config = {};

        inputs.forEach(input => {
            const section = input.dataset.section;
            const key = input.dataset.key;
            let value = input.value;

            // Type coercion
            if (input.dataset.originalType === 'number') value = Number(value);
            else if (input.dataset.originalType === 'boolean' || input.tagName === 'SELECT') value = value === 'true';
            else if (input.dataset.originalType === 'object') {
                try { value = JSON.parse(value); } catch { /* keep as string */ }
            }

            if (section) {
                if (!config[section]) config[section] = {};
                config[section][key] = value;
            } else {
                config[key] = value;
            }
        });

        this._sendMsg('config.set', { config });
        this.els.configStatus.textContent = 'Saving...';
    }

    // ── Skills Manager ────────────────────────────────────────────────────

    _loadSkills() {
        this._sendMsg('skills.list', {});
    }

    _renderSkills(skills) {
        const container = this.els.skillsList;
        container.innerHTML = '';

        if (!skills.length) {
            container.innerHTML = '<div class="loading-state">No skills loaded</div>';
            return;
        }

        skills.forEach(skill => {
            const card = document.createElement('div');
            card.className = 'skill-card';

            const riskClass = (skill.risk_level || 'low').toLowerCase();
            const icon = { builtin: '⚡', general: '🔧', cyber: '🛡️', developer: '💻', personal: '👤', system: '🖥️', meta: '🧠' }[skill.category] || '🔧';

            card.innerHTML = `
        <div class="skill-icon">${icon}</div>
        <div class="skill-info">
          <div class="skill-name">${this._escapeHtml(skill.name)}</div>
          <div class="skill-desc">${this._escapeHtml(skill.description || '')}</div>
        </div>
        <div class="skill-meta">
          <span class="skill-badge category">${skill.category || 'general'}</span>
          <span class="skill-badge ${riskClass}">${skill.risk_level || 'LOW'}</span>
        </div>
      `;
            container.appendChild(card);
        });
    }

    _reloadSkills() {
        this._sendMsg('skills.reload', {});
        this._toast('Reloading skills...', 'info');
    }

    // ── Env Vars Manager ──────────────────────────────────────────────────

    _loadEnv() {
        this._sendMsg('env.list', {});
    }

    _renderEnv(envVars) {
        this.envData = envVars;
        const container = this.els.envList;
        container.innerHTML = '';

        if (!envVars.length) {
            container.innerHTML = '<div class="loading-state">No environment variables found</div>';
            return;
        }

        envVars.forEach(env => {
            const row = document.createElement('div');
            row.className = 'env-row';
            row.innerHTML = `
        <div class="env-key">${this._escapeHtml(env.key)}</div>
        <div class="env-value">${this._escapeHtml(env.masked)}</div>
        <div class="env-actions">
          <button class="env-edit-btn" data-key="${this._escapeHtml(env.key)}" data-value="${this._escapeHtml(env.value)}">Edit</button>
        </div>
      `;
            row.querySelector('.env-edit-btn').addEventListener('click', (e) => {
                this._showEnvModal(e.target.dataset.key, e.target.dataset.value);
            });
            container.appendChild(row);
        });
    }

    _showEnvModal(key = '', value = '') {
        // Remove existing modal
        document.querySelector('.modal-overlay')?.remove();

        const overlay = document.createElement('div');
        overlay.className = 'modal-overlay';
        overlay.innerHTML = `
      <div class="modal">
        <h3>${key ? '✏️ Edit Variable' : '➕ Add Variable'}</h3>
        <div class="modal-field">
          <label>Key</label>
          <input type="text" id="modal-env-key" value="${this._escapeHtml(key)}" placeholder="MY_API_KEY" ${key ? 'readonly' : ''}>
        </div>
        <div class="modal-field">
          <label>Value</label>
          <input type="text" id="modal-env-value" value="${this._escapeHtml(value)}" placeholder="your-secret-value">
        </div>
        <div class="modal-actions">
          <button class="btn btn-secondary" id="modal-cancel">Cancel</button>
          <button class="btn btn-primary" id="modal-save">Save</button>
        </div>
      </div>
    `;

        document.body.appendChild(overlay);

        overlay.querySelector('#modal-cancel').addEventListener('click', () => overlay.remove());
        overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });

        overlay.querySelector('#modal-save').addEventListener('click', () => {
            const newKey = document.getElementById('modal-env-key').value.trim();
            const newValue = document.getElementById('modal-env-value').value.trim();
            if (newKey) {
                this._sendMsg('env.set', { key: newKey, value: newValue });
                overlay.remove();
            }
        });

        // Focus the right field
        setTimeout(() => document.getElementById(key ? 'modal-env-value' : 'modal-env-key').focus(), 100);
    }

    // ── System Info ───────────────────────────────────────────────────────

    _loadSystemInfo() {
        this._sendMsg('system.info', {});
    }

    _renderSystemInfo(data) {
        const container = this.els.systemInfo;
        container.innerHTML = `
      <div class="system-grid">
        <div class="system-card">
          <span class="card-icon">🏷️</span>
          <div class="card-label">Version</div>
          <div class="card-value">${data.version || '—'}</div>
        </div>
        <div class="system-card">
          <span class="card-icon">🐍</span>
          <div class="card-label">Python</div>
          <div class="card-value">${data.python || '—'}</div>
        </div>
        <div class="system-card">
          <span class="card-icon">🔧</span>
          <div class="card-label">Skills</div>
          <div class="card-value">${data.skills_count ?? '—'}</div>
        </div>
        <div class="system-card">
          <span class="card-icon">📡</span>
          <div class="card-label">Sessions</div>
          <div class="card-value">${data.sessions_count ?? '—'}</div>
        </div>
        <div class="system-card">
          <span class="card-icon">🔌</span>
          <div class="card-label">Connections</div>
          <div class="card-value">${data.connections_count ?? '—'}</div>
        </div>
        <div class="system-card">
          <span class="card-icon">💻</span>
          <div class="card-label">Platform</div>
          <div class="card-value">${data.platform || '—'}</div>
        </div>
      </div>
    `;
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    _sendMsg(type, data = {}) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        const msg = {
            type,
            id: Math.random().toString(36).slice(2, 10),
            session_id: this.sessionId,
            data,
        };
        this.ws.send(JSON.stringify(msg));
    }

    _updateStatus(status) {
        const dot = this.els.connectionBadge.querySelector('.status-dot');
        dot.className = `status-dot ${status}`;
        this.els.connectionText.textContent = status === 'connected' ? 'Connected' : status === 'connecting' ? 'Connecting…' : 'Disconnected';
        this.els.topbarStatus.className = `connection-indicator ${status === 'connected' ? 'status-dot connected' : 'status-dot disconnected'}`;
    }

    _toast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        this.els.toastContainer.appendChild(toast);
        setTimeout(() => toast.remove(), 4000);
    }

    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// ── Initialize ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    window.app = new NeuralClawApp();
});
