'use strict';

// AgentChat: natural-language assistant bridge for DiagramState editing.

const AgentChat = (() => {
  const history = [];

  function getApiBase() {
    const configured = typeof window.__API_BASE__ === 'string' ? window.__API_BASE__.trim() : '';
    if (configured) return configured.replace(/\/+$/, '');
    if (window.location && /^https?:$/i.test(window.location.protocol) && window.location.origin && window.location.origin !== 'null') {
      return window.location.origin.replace(/\/+$/, '');
    }
    return 'http://127.0.0.1:5000';
  }

  function apiUrl(path) {
    const cleanPath = String(path || '').startsWith('/') ? path : `/${String(path || '')}`;
    return `${getApiBase()}${cleanPath}`;
  }

  function selectedModel() {
    const input = document.querySelector('input[name="agentModel"]:checked');
    return input ? String(input.value) : 'gemini';
  }

  function appendMessage(text, role) {
    const container = document.getElementById('chatMessages');
    if (!container) return;

    const item = document.createElement('div');
    item.className = `chat-msg chat-msg-${role}`;
    item.textContent = String(text || '');
    container.appendChild(item);
    container.scrollTop = container.scrollHeight;
  }

  function buildRequest(command) {
    const exported = window.DiagramState.export();
    const meta = window.DiagramState.getMeta();
    return {
      command,
      model: selectedModel(),
      state: {
        equipment: exported.equipment,
        pipes: exported.pipes,
        image_width: Number(meta.image_width) || 0,
        image_height: Number(meta.image_height) || 0,
      },
      history: history.slice(-3),
    };
  }

  function applyActions(actions) {
    if (!Array.isArray(actions)) return;

    actions.forEach((action) => {
      if (!action || typeof action !== 'object') return;

      const actionType = String(action.type || '').trim();
      if (!actionType) return;

      switch (actionType) {
        case 'MOVE_EQUIPMENT':
          window.DiagramState.moveEquipment(action.id, Number(action.dx) || 0, Number(action.dy) || 0);
          break;

        case 'SET_EQUIPMENT_POSITION':
          window.DiagramState.setEquipmentPosition(action.id, Number(action.x), Number(action.y));
          break;

        case 'DELETE_EQUIPMENT':
          window.DiagramState.deleteEquipment(action.id);
          break;

        case 'UPDATE_LABEL':
          window.DiagramState.updateLabel(action.id, action.label || '');
          break;

        case 'CHANGE_TYPE':
          window.DiagramState.changeType(action.id, action.new_type || action.symbol_type || action.value);
          break;

        case 'ADD_EQUIPMENT': {
          const symbolType = action.new_type || action.symbol_type || action.eq_type || 'gate_valve';
          window.DiagramState.addEquipment({
            id: window.DiagramState.generateEqId(),
            type: String(symbolType),
            position: [Number(action.x) || 0, Number(action.y) || 0],
            label: action.label || '',
            symbol_size: 36,
          });
          break;
        }

        case 'ADD_PIPE': {
          const fromEq = window.DiagramState.getById(action.from_id);
          const toEq = window.DiagramState.getById(action.to_id);
          if (!fromEq || !toEq) break;
          window.DiagramState.addPipe({
            id: window.DiagramState.generatePipeId(),
            from_id: String(fromEq.id),
            to_id: String(toEq.id),
            points: [
              [fromEq.position[0], fromEq.position[1]],
              [toEq.position[0], toEq.position[1]],
            ],
            line_style: String(action.line_style || 'solid').toLowerCase() === 'dashed' ? 'dashed' : 'solid',
          });
          break;
        }

        case 'DELETE_PIPE':
          window.DiagramState.deletePipe(action.id);
          break;

        case 'SET_LINE_STYLE':
          window.DiagramState.setLineStyle(action.id, action.style || action.line_style || 'solid');
          break;

        case 'REROUTE_PIPE':
          if (Array.isArray(action.new_points)) {
            window.DiagramState.reroutePipe(action.id, action.new_points);
          }
          break;

        default:
          break;
      }
    });
  }

  async function sendCommand(commandText) {
    const command = String(commandText || '').trim();
    if (!command) return;

    appendMessage(command, 'user');

    const thinking = document.createElement('div');
    thinking.className = 'chat-msg chat-msg-thinking';
    thinking.textContent = 'Thinking...';
    const container = document.getElementById('chatMessages');
    if (container) {
      container.appendChild(thinking);
      container.scrollTop = container.scrollHeight;
    }

    try {
      const resp = await fetch(apiUrl('/agent_command'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildRequest(command)),
      });
      const data = await resp.json();

      thinking.remove();

      if (!resp.ok || !data.success) {
        const msg = data && data.error ? data.error : `Request failed (${resp.status})`;
        appendMessage(msg, 'agent-error');
        return;
      }

      const reply = String(data.reply || 'Done.');
      appendMessage(reply, 'agent');
      history.push({ user: command, reply });
      if (history.length > 20) history.shift();

      if (Array.isArray(data.actions) && data.actions.length) {
        applyActions(data.actions);
      }
    } catch (error) {
      thinking.remove();
      appendMessage(`Network error: ${error.message}`, 'agent-error');
    }
  }

  function init() {
    const openBtn = document.getElementById('toggleAgentBtn');
    const closeBtn = document.getElementById('closeAgentBtn');
    const panel = document.getElementById('agentPanel');
    const sendBtn = document.getElementById('chatSendBtn');
    const input = document.getElementById('chatInput');

    if (openBtn && panel) {
      openBtn.addEventListener('click', () => panel.classList.toggle('hidden'));
    }

    if (closeBtn && panel) {
      closeBtn.addEventListener('click', () => panel.classList.add('hidden'));
    }

    if (sendBtn && input) {
      sendBtn.addEventListener('click', () => {
        const value = input.value;
        input.value = '';
        sendCommand(value);
      });
      input.addEventListener('keydown', (evt) => {
        if (evt.key === 'Enter') {
          evt.preventDefault();
          sendBtn.click();
        }
      });
    }
  }

  return {
    init,
    sendCommand,
    applyActions,
  };
})();

window.AgentChat = AgentChat;
