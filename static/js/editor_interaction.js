'use strict';

// EditorInteraction: selection, drag, placement, pipe drawing, rerouting, undo/redo.
// Works in image pixel coordinate space only.

const EditorInteraction = (() => {
  let svgEl = null;
  let tool = 'select'; // select | place | draw-pipe
  let placingType = null;

  let dragCtx = null; // equipment drag context
  let rerouteCtx = null; // pipe midpoint drag context
  let drawPipeStartId = null;

  function stateSnapshot() {
    return window.DiagramState ? window.DiagramState.getState() : { equipment: [], pipes: [], meta: {} };
  }

  function findEquipment(id) {
    return window.DiagramState.getById(id);
  }

  function findPipe(id) {
    return window.DiagramState.getPipeById(id);
  }

  function setTool(nextTool, nextPlacingType) {
    tool = nextTool || 'select';
    placingType = nextPlacingType || null;
    drawPipeStartId = null;
    dragCtx = null;
    rerouteCtx = null;
    if (window.PidRenderer) window.PidRenderer.clearDrawLayer();

    document.querySelectorAll('.editor-tool[data-tool]').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.tool === tool);
    });

    if (svgEl) {
      if (tool === 'place') svgEl.style.cursor = 'crosshair';
      else if (tool === 'draw-pipe') svgEl.style.cursor = 'cell';
      else svgEl.style.cursor = 'default';
    }
  }

  function showPropertyPanel(selected) {
    const panel = document.getElementById('propertyPanel');
    const content = document.getElementById('propertyContent');
    if (!panel || !content) return;

    if (!selected || !selected.id || !selected.type) {
      panel.classList.add('hidden');
      content.innerHTML = '';
      return;
    }

    panel.classList.remove('hidden');

    if (selected.type === 'equipment') {
      const eq = findEquipment(selected.id);
      if (!eq) {
        panel.classList.add('hidden');
        return;
      }

      const allTypes = Object.keys(window.symbolTemplates || {}).sort();
      const typeOptions = allTypes
        .map((typeName) => `<option value="${typeName}"${typeName === eq.type ? ' selected' : ''}>${typeName}</option>`)
        .join('');

      content.innerHTML = [
        '<div class="prop-row"><label class="prop-label">ID</label><input class="input" id="propEqId" readonly></div>',
        '<div class="prop-row"><label class="prop-label">Type</label><select class="input" id="propEqType"></select></div>',
        '<div class="prop-row"><label class="prop-label">Label</label><input class="input" id="propEqLabel"></div>',
        '<div class="prop-row"><label class="prop-label">Position X</label><input class="input" id="propEqX" type="number" step="1"></div>',
        '<div class="prop-row"><label class="prop-label">Position Y</label><input class="input" id="propEqY" type="number" step="1"></div>',
        '<button class="btn btn-secondary" id="propEqDelete" type="button">Delete Symbol</button>',
      ].join('');

      document.getElementById('propEqId').value = String(eq.id);
      const typeSelect = document.getElementById('propEqType');
      typeSelect.innerHTML = typeOptions;
      if (!allTypes.includes(eq.type)) {
        const customOption = document.createElement('option');
        customOption.value = eq.type;
        customOption.textContent = eq.type;
        customOption.selected = true;
        typeSelect.appendChild(customOption);
      }
      document.getElementById('propEqLabel').value = eq.label || '';
      document.getElementById('propEqX').value = Math.round(eq.position[0]);
      document.getElementById('propEqY').value = Math.round(eq.position[1]);

      typeSelect.addEventListener('change', (e) => window.DiagramState.changeType(eq.id, e.target.value));
      document.getElementById('propEqLabel').addEventListener('change', (e) => window.DiagramState.updateLabel(eq.id, e.target.value));
      document.getElementById('propEqX').addEventListener('change', (e) => {
        const nx = Number(e.target.value);
        if (Number.isFinite(nx)) window.DiagramState.setEquipmentPosition(eq.id, nx, eq.position[1]);
      });
      document.getElementById('propEqY').addEventListener('change', (e) => {
        const ny = Number(e.target.value);
        if (Number.isFinite(ny)) window.DiagramState.setEquipmentPosition(eq.id, eq.position[0], ny);
      });
      document.getElementById('propEqDelete').addEventListener('click', () => {
        if (confirm(`Delete ${eq.type} \"${eq.label || eq.id}\"?`)) {
          window.DiagramState.deleteEquipment(eq.id);
          window.PidRenderer.clearSelection();
          showPropertyPanel(null);
        }
      });
      return;
    }

    if (selected.type === 'pipe') {
      const pipe = findPipe(selected.id);
      if (!pipe) {
        panel.classList.add('hidden');
        return;
      }

      content.innerHTML = [
        '<div class="prop-row"><label class="prop-label">ID</label><input class="input" id="propPipeId" readonly></div>',
        '<div class="prop-row"><label class="prop-label">Line Style</label>',
        '<select class="input" id="propPipeStyle"><option value="solid">solid</option><option value="dashed">dashed</option></select></div>',
        '<button class="btn btn-secondary" id="propPipeDelete" type="button">Delete Pipe</button>',
      ].join('');

      document.getElementById('propPipeId').value = String(pipe.id);
      const styleSelect = document.getElementById('propPipeStyle');
      styleSelect.value = pipe.line_style || 'solid';
      styleSelect.addEventListener('change', (e) => window.DiagramState.setLineStyle(pipe.id, e.target.value));
      document.getElementById('propPipeDelete').addEventListener('click', () => {
        if (confirm(`Delete pipe \"${pipe.id}\"?`)) {
          window.DiagramState.deletePipe(pipe.id);
          window.PidRenderer.clearSelection();
          showPropertyPanel(null);
        }
      });
    }
  }

  function selectEquipment(eqId) {
    window.PidRenderer.setSelection(eqId, 'equipment');
    showPropertyPanel({ id: eqId, type: 'equipment' });
  }

  function selectPipe(pipeId) {
    window.PidRenderer.setSelection(pipeId, 'pipe');
    showPropertyPanel({ id: pipeId, type: 'pipe' });
  }

  function clearSelection() {
    window.PidRenderer.clearSelection();
    showPropertyPanel(null);
  }

  function beginEquipmentDrag(eqId, clientX, clientY) {
    const eq = findEquipment(eqId);
    if (!eq) return;

    const p = window.PidRenderer.clientToImage(clientX, clientY);
    window.DiagramState.pushUndoSnapshot();
    dragCtx = {
      id: eqId,
      offsetX: p.x - eq.position[0],
      offsetY: p.y - eq.position[1],
    };
  }

  function updateEquipmentDrag(clientX, clientY) {
    if (!dragCtx) return;
    const p = window.PidRenderer.clientToImage(clientX, clientY);
    const nx = p.x - dragCtx.offsetX;
    const ny = p.y - dragCtx.offsetY;
    window.DiagramState.setEquipmentPosition(dragCtx.id, nx, ny, { skipUndo: true });
  }

  function endEquipmentDrag() {
    dragCtx = null;
  }

  function beginPipeReroute(pipeId) {
    const pipe = findPipe(pipeId);
    if (!pipe || !Array.isArray(pipe.points) || pipe.points.length < 2) return;

    window.DiagramState.pushUndoSnapshot();
    rerouteCtx = {
      pipeId,
      start: [...pipe.points[0]],
      end: [...pipe.points[pipe.points.length - 1]],
    };
  }

  function updatePipeReroute(clientX, clientY) {
    if (!rerouteCtx) return;
    const p = window.PidRenderer.clientToImage(clientX, clientY);
    const points = [
      [rerouteCtx.start[0], rerouteCtx.start[1]],
      [p.x, p.y],
      [rerouteCtx.end[0], rerouteCtx.end[1]],
    ];
    window.DiagramState.reroutePipe(rerouteCtx.pipeId, points, { skipUndo: true });
  }

  function endPipeReroute() {
    rerouteCtx = null;
  }

  function placeEquipment(clientX, clientY) {
    if (!placingType) return;
    const p = window.PidRenderer.clientToImage(clientX, clientY);
    const newEq = window.DiagramState.addEquipment({
      id: window.DiagramState.generateEqId(),
      type: placingType,
      position: [p.x, p.y],
      label: '',
      symbol_size: 36,
    });

    if (newEq) selectEquipment(newEq.id);
  }

  function handlePipeDrawClick(eqId, clientX, clientY) {
    const eq = findEquipment(eqId);
    if (!eq) return;

    if (!drawPipeStartId) {
      drawPipeStartId = eqId;
      window.PidRenderer.drawDraftLine(eq.position, window.PidRenderer.clientToImage(clientX, clientY));
      return;
    }

    if (String(drawPipeStartId) === String(eqId)) {
      drawPipeStartId = null;
      window.PidRenderer.clearDrawLayer();
      return;
    }

    const startEq = findEquipment(drawPipeStartId);
    const endEq = findEquipment(eqId);
    if (!startEq || !endEq) {
      drawPipeStartId = null;
      window.PidRenderer.clearDrawLayer();
      return;
    }

    const newPipe = window.DiagramState.addPipe({
      id: window.DiagramState.generatePipeId(),
      from_id: startEq.id,
      to_id: endEq.id,
      points: [
        [startEq.position[0], startEq.position[1]],
        [endEq.position[0], endEq.position[1]],
      ],
      line_style: 'solid',
    });

    drawPipeStartId = null;
    window.PidRenderer.clearDrawLayer();

    if (newPipe) {
      selectPipe(newPipe.id);
      setTool('select');
    }
  }

  function updatePipeDraft(clientX, clientY) {
    if (tool !== 'draw-pipe' || !drawPipeStartId) return;
    const startEq = findEquipment(drawPipeStartId);
    if (!startEq) return;
    const p = window.PidRenderer.clientToImage(clientX, clientY);
    window.PidRenderer.drawDraftLine(startEq.position, [p.x, p.y]);
  }

  function onMouseDown(evt) {
    if (evt.button !== 0) return;

    const handleTarget = evt.target.closest('[data-pipe-mid-handle]');
    if (handleTarget) {
      evt.preventDefault();
      const pipeId = handleTarget.getAttribute('data-pipe-mid-handle');
      if (pipeId) beginPipeReroute(pipeId);
      return;
    }

    const eqTarget = evt.target.closest('[data-eq-id]');
    const pipeTarget = evt.target.closest('[data-pipe-id]');

    if (tool === 'draw-pipe') {
      if (eqTarget) {
        evt.preventDefault();
        handlePipeDrawClick(eqTarget.getAttribute('data-eq-id'), evt.clientX, evt.clientY);
      }
      return;
    }

    if (tool === 'place') {
      evt.preventDefault();
      placeEquipment(evt.clientX, evt.clientY);
      return;
    }

    if (eqTarget) {
      evt.preventDefault();
      const eqId = eqTarget.getAttribute('data-eq-id');
      selectEquipment(eqId);
      beginEquipmentDrag(eqId, evt.clientX, evt.clientY);
      return;
    }

    if (pipeTarget) {
      evt.preventDefault();
      selectPipe(pipeTarget.getAttribute('data-pipe-id'));
      return;
    }

    clearSelection();
  }

  function onMouseMove(evt) {
    if (dragCtx) {
      evt.preventDefault();
      updateEquipmentDrag(evt.clientX, evt.clientY);
      return;
    }

    if (rerouteCtx) {
      evt.preventDefault();
      updatePipeReroute(evt.clientX, evt.clientY);
      return;
    }

    updatePipeDraft(evt.clientX, evt.clientY);
  }

  function onMouseUp() {
    if (dragCtx) endEquipmentDrag();
    if (rerouteCtx) endPipeReroute();
  }

  function onDoubleClick(evt) {
    if (evt.target === svgEl) {
      clearSelection();
    }
  }

  function onKeyDown(evt) {
    const editorSection = document.getElementById('editorSection');
    if (!editorSection || editorSection.classList.contains('hidden')) return;

    const target = evt.target;
    const editingInput = target && (target.matches('input') || target.matches('textarea') || target.matches('select'));

    if ((evt.ctrlKey || evt.metaKey) && !evt.shiftKey && evt.key.toLowerCase() === 'z') {
      evt.preventDefault();
      window.DiagramState.undo();
      return;
    }

    if ((evt.ctrlKey || evt.metaKey) && (evt.key.toLowerCase() === 'y' || (evt.shiftKey && evt.key.toLowerCase() === 'z'))) {
      evt.preventDefault();
      window.DiagramState.redo();
      return;
    }

    if (editingInput) return;

    if (evt.key === 'Delete' || evt.key === 'Backspace') {
      const sel = window.PidRenderer.getSelection();
      if (!sel || !sel.id) return;
      evt.preventDefault();

      if (sel.type === 'equipment') {
        const eq = findEquipment(sel.id);
        if (!eq) return;
        if (confirm(`Delete ${eq.type} \"${eq.label || eq.id}\"?`)) {
          window.DiagramState.deleteEquipment(eq.id);
          clearSelection();
        }
        return;
      }

      if (sel.type === 'pipe') {
        const pipe = findPipe(sel.id);
        if (!pipe) return;
        if (confirm(`Delete pipe \"${pipe.id}\"?`)) {
          window.DiagramState.deletePipe(pipe.id);
          clearSelection();
        }
      }
    }
  }

  function populatePalette() {
    const container = document.getElementById('paletteItems');
    if (!container) return;
    container.innerHTML = '';

    const templates = window.symbolTemplates || {};
    Object.keys(templates).sort().forEach((symbolType) => {
      const item = document.createElement('button');
      item.type = 'button';
      item.className = 'palette-item';
      item.setAttribute('aria-label', `Place ${symbolType}`);
      item.title = symbolType;

      const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      icon.classList.add('palette-icon');
      window.PidRenderer.renderMiniSymbol(icon, symbolType, 40);

      const label = document.createElement('div');
      label.className = 'palette-label';
      label.textContent = symbolType;

      item.appendChild(icon);
      item.appendChild(label);

      item.addEventListener('click', () => {
        document.querySelectorAll('.palette-item').forEach((el) => el.classList.remove('active'));
        item.classList.add('active');
        setTool('place', symbolType);
      });

      container.appendChild(item);
    });
  }

  function bindToolbar() {
    document.querySelectorAll('.editor-tool[data-tool]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const next = btn.dataset.tool || 'select';
        if (next !== 'place') {
          document.querySelectorAll('.palette-item').forEach((el) => el.classList.remove('active'));
        }
        setTool(next);
      });
    });

    const zoomOutBtn = document.getElementById('editorZoomOutBtn');
    const zoomInBtn = document.getElementById('editorZoomInBtn');
    const zoomResetBtn = document.getElementById('editorZoomResetBtn');
    if (zoomOutBtn) zoomOutBtn.addEventListener('click', () => window.PidRenderer.zoomBy(1 / 1.2));
    if (zoomInBtn) zoomInBtn.addEventListener('click', () => window.PidRenderer.zoomBy(1.2));
    if (zoomResetBtn) zoomResetBtn.addEventListener('click', () => window.PidRenderer.setZoom(1));

    const bgToggle = document.getElementById('toggleBackgroundBtn');
    if (bgToggle) {
      let visible = true;
      bgToggle.addEventListener('click', () => {
        visible = !visible;
        window.PidRenderer.setBackgroundVisible(visible);
        bgToggle.textContent = visible ? 'Hide Background' : 'Show Background';
      });
    }

    const editorFrame = document.getElementById('editorFrame');
    if (editorFrame) {
      editorFrame.addEventListener('wheel', (evt) => {
        evt.preventDefault();
        const factor = evt.deltaY < 0 ? 1.12 : 1 / 1.12;
        window.PidRenderer.zoomBy(factor);
      }, { passive: false });
    }
  }

  function init(svgId) {
    svgEl = document.getElementById(svgId);
    if (!svgEl) return;

    svgEl.addEventListener('mousedown', onMouseDown);
    svgEl.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    svgEl.addEventListener('dblclick', onDoubleClick);
    window.addEventListener('keydown', onKeyDown);

    bindToolbar();

    document.addEventListener('stateChanged', () => {
      const sel = window.PidRenderer.getSelection();
      if (!sel || !sel.id) {
        showPropertyPanel(null);
        return;
      }

      if (sel.type === 'equipment' && !findEquipment(sel.id)) {
        clearSelection();
        return;
      }

      if (sel.type === 'pipe' && !findPipe(sel.id)) {
        clearSelection();
        return;
      }

      showPropertyPanel(sel);
    });

    setTool('select');
  }

  return {
    init,
    setTool,
    populatePalette,
    clearSelection,
  };
})();

window.EditorInteraction = EditorInteraction;
