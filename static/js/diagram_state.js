'use strict';

// DiagramState: canonical source of truth for editor data.
// All coordinates are stored in image pixel space (origin top-left).

(function attachDiagramState(globalScope) {
  const STATE_CHANGED_EVENT = 'stateChanged';
  const MAX_UNDO = 50;

  const state = {
    equipment: [],
    pipes: [],
    meta: {},
  };

  const undoStack = [];
  const redoStack = [];

  let eqCounter = 0;
  let pipeCounter = 0;

  function deepClone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function toNumber(value, fallback) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function emitStateChanged(reason) {
    const detail = { state: deepClone(state), reason: reason || 'update' };
    document.dispatchEvent(new CustomEvent(STATE_CHANGED_EVENT, { detail }));
    // Backward-compatible alias for any older listeners.
    document.dispatchEvent(new CustomEvent('diagramStateChanged', { detail }));
  }

  function trimStack(stack) {
    if (stack.length > MAX_UNDO) {
      stack.splice(0, stack.length - MAX_UNDO);
    }
  }

  function normalizeBBox(bbox, fallbackPosition, fallbackSize) {
    if (Array.isArray(bbox) && bbox.length === 4) {
      const x1 = toNumber(bbox[0], null);
      const y1 = toNumber(bbox[1], null);
      const x2 = toNumber(bbox[2], null);
      const y2 = toNumber(bbox[3], null);
      if (x1 != null && y1 != null && x2 != null && y2 != null) {
        return [x1, y1, x2, y2];
      }
    }

    if (Array.isArray(fallbackPosition) && fallbackPosition.length === 2) {
      const cx = toNumber(fallbackPosition[0], 0);
      const cy = toNumber(fallbackPosition[1], 0);
      const size = Math.max(8, toNumber(fallbackSize, 36));
      const h = size * 0.5;
      return [cx - h, cy - h, cx + h, cy + h];
    }
    return null;
  }

  function normalizeEquipmentItem(item) {
    if (!item || typeof item !== 'object') return null;

    const id = String(item.id != null ? item.id : generateEqId()).trim();
    if (!id) return null;

    const posRaw = Array.isArray(item.position) && item.position.length >= 2 ? item.position : [0, 0];
    const position = [toNumber(posRaw[0], 0), toNumber(posRaw[1], 0)];

    const symbolSize = Math.max(6, toNumber(item.symbol_size, 36));

    return {
      id,
      type: String(item.type || 'unknown').trim() || 'unknown',
      position,
      label: item.label == null ? '' : String(item.label),
      bbox: normalizeBBox(item.bbox, position, symbolSize),
      symbol_size: symbolSize,
      attached_line: item.attached_line == null ? null : String(item.attached_line),
    };
  }

  function normalizePipePoints(points) {
    if (!Array.isArray(points)) return [];
    const clean = [];
    for (const pt of points) {
      if (!Array.isArray(pt) || pt.length < 2) continue;
      const x = toNumber(pt[0], null);
      const y = toNumber(pt[1], null);
      if (x == null || y == null) continue;
      if (clean.length && clean[clean.length - 1][0] === x && clean[clean.length - 1][1] === y) continue;
      clean.push([x, y]);
    }
    return clean;
  }

  function normalizePipeItem(item) {
    if (!item || typeof item !== 'object') return null;

    const id = String(item.id != null ? item.id : generatePipeId()).trim();
    if (!id) return null;

    const points = normalizePipePoints(item.points);
    if (points.length < 2) return null;

    const lineStyle = String(item.line_style || 'solid').trim().toLowerCase() === 'dashed' ? 'dashed' : 'solid';

    return {
      id,
      from_id: item.from_id == null ? null : String(item.from_id),
      to_id: item.to_id == null ? null : String(item.to_id),
      points,
      line_style: lineStyle,
    };
  }

  function setState(nextState, reason) {
    state.equipment = nextState.equipment;
    state.pipes = nextState.pipes;
    state.meta = nextState.meta;
    emitStateChanged(reason);
  }

  function pushUndoSnapshot() {
    undoStack.push(deepClone(state));
    trimStack(undoStack);
    redoStack.length = 0;
  }

  function getEquipmentByIdRef(id) {
    const key = String(id);
    return state.equipment.find((eq) => String(eq.id) === key) || null;
  }

  function getPipeByIdRef(id) {
    const key = String(id);
    return state.pipes.find((p) => String(p.id) === key) || null;
  }

  function pointInsideBox(pt, bbox, padding) {
    if (!Array.isArray(pt) || pt.length < 2 || !Array.isArray(bbox) || bbox.length !== 4) return false;
    const x = pt[0];
    const y = pt[1];
    const x1 = Math.min(bbox[0], bbox[2]) - padding;
    const y1 = Math.min(bbox[1], bbox[3]) - padding;
    const x2 = Math.max(bbox[0], bbox[2]) + padding;
    const y2 = Math.max(bbox[1], bbox[3]) + padding;
    return x >= x1 && x <= x2 && y >= y1 && y <= y2;
  }

  function shiftAttachedPipeEndpoints(eq, dx, dy, oldBBox) {
    if (!Number.isFinite(dx) || !Number.isFinite(dy)) return;
    if (Math.abs(dx) < 1e-9 && Math.abs(dy) < 1e-9) return;

    const eqId = String(eq.id);
    const fallbackBBox = normalizeBBox(eq.bbox, eq.position, eq.symbol_size);
    const referenceBox = oldBBox || fallbackBBox;

    state.pipes.forEach((pipe) => {
      if (!Array.isArray(pipe.points) || pipe.points.length < 2) return;

      let movedStart = false;
      let movedEnd = false;

      if (pipe.from_id != null && String(pipe.from_id) === eqId) {
        pipe.points[0] = [pipe.points[0][0] + dx, pipe.points[0][1] + dy];
        movedStart = true;
      }
      if (pipe.to_id != null && String(pipe.to_id) === eqId) {
        const lastIdx = pipe.points.length - 1;
        pipe.points[lastIdx] = [pipe.points[lastIdx][0] + dx, pipe.points[lastIdx][1] + dy];
        movedEnd = true;
      }

      if (!movedStart && referenceBox && pointInsideBox(pipe.points[0], referenceBox, 10)) {
        pipe.points[0] = [pipe.points[0][0] + dx, pipe.points[0][1] + dy];
      }
      if (!movedEnd && referenceBox && pointInsideBox(pipe.points[pipe.points.length - 1], referenceBox, 10)) {
        const last = pipe.points.length - 1;
        pipe.points[last] = [pipe.points[last][0] + dx, pipe.points[last][1] + dy];
      }
    });
  }

  function mutate(reason, options, mutateFn) {
    const opts = options || {};
    if (!opts.skipUndo) {
      pushUndoSnapshot();
    }
    mutateFn();
    if (!opts.silent) {
      emitStateChanged(reason);
    }
  }

  function load(geometryPayload, meta) {
    const geometry = geometryPayload && typeof geometryPayload === 'object' ? geometryPayload : {};
    const nextEquipment = [];
    const nextPipes = [];

    if (Array.isArray(geometry.equipment)) {
      for (const item of geometry.equipment) {
        const normalized = normalizeEquipmentItem(item);
        if (normalized) nextEquipment.push(normalized);
      }
    }

    if (Array.isArray(geometry.pipes)) {
      for (const item of geometry.pipes) {
        const normalized = normalizePipeItem(item);
        if (normalized) nextPipes.push(normalized);
      }
    }

    const nextMeta = meta && typeof meta === 'object' ? deepClone(meta) : {};

    undoStack.length = 0;
    redoStack.length = 0;

    setState({ equipment: nextEquipment, pipes: nextPipes, meta: nextMeta }, 'load');
  }

  function addEquipment(eq) {
    const normalized = normalizeEquipmentItem(eq || {});
    if (!normalized) return null;

    mutate('addEquipment', null, () => {
      state.equipment.push(normalized);
    });
    return deepClone(normalized);
  }

  function setEquipmentPosition(id, x, y, options) {
    const eq = getEquipmentByIdRef(id);
    if (!eq) return false;

    const nx = toNumber(x, null);
    const ny = toNumber(y, null);
    if (nx == null || ny == null) return false;

    mutate('setEquipmentPosition', options, () => {
      const oldX = eq.position[0];
      const oldY = eq.position[1];
      const dx = nx - oldX;
      const dy = ny - oldY;
      const oldBBox = Array.isArray(eq.bbox) ? [...eq.bbox] : null;

      eq.position[0] = nx;
      eq.position[1] = ny;

      if (Array.isArray(eq.bbox) && eq.bbox.length === 4) {
        eq.bbox = [eq.bbox[0] + dx, eq.bbox[1] + dy, eq.bbox[2] + dx, eq.bbox[3] + dy];
      }

      shiftAttachedPipeEndpoints(eq, dx, dy, oldBBox);
    });

    return true;
  }

  function moveEquipment(id, dx, dy, options) {
    const eq = getEquipmentByIdRef(id);
    if (!eq) return false;

    const ddx = toNumber(dx, 0);
    const ddy = toNumber(dy, 0);
    return setEquipmentPosition(id, eq.position[0] + ddx, eq.position[1] + ddy, options);
  }

  function deleteEquipment(id) {
    const eq = getEquipmentByIdRef(id);
    if (!eq) return false;

    mutate('deleteEquipment', null, () => {
      const eqId = String(eq.id);
      const eqBBox = normalizeBBox(eq.bbox, eq.position, eq.symbol_size);

      state.equipment = state.equipment.filter((item) => String(item.id) !== eqId);
      state.pipes = state.pipes.filter((pipe) => {
        if (pipe.from_id != null && String(pipe.from_id) === eqId) return false;
        if (pipe.to_id != null && String(pipe.to_id) === eqId) return false;
        if (!Array.isArray(pipe.points) || pipe.points.length < 2) return true;
        const startInside = eqBBox ? pointInsideBox(pipe.points[0], eqBBox, 10) : false;
        const endInside = eqBBox ? pointInsideBox(pipe.points[pipe.points.length - 1], eqBBox, 10) : false;
        return !(startInside || endInside);
      });
    });

    return true;
  }

  function updateLabel(id, label) {
    const eq = getEquipmentByIdRef(id);
    if (!eq) return false;

    mutate('updateLabel', null, () => {
      eq.label = label == null ? '' : String(label);
    });

    return true;
  }

  function changeType(id, newType) {
    const eq = getEquipmentByIdRef(id);
    if (!eq) return false;

    const clean = String(newType || '').trim();
    if (!clean) return false;

    mutate('changeType', null, () => {
      eq.type = clean;
    });

    return true;
  }

  function addPipe(pipe) {
    const normalized = normalizePipeItem(pipe || {});
    if (!normalized) return null;

    mutate('addPipe', null, () => {
      state.pipes.push(normalized);
    });
    return deepClone(normalized);
  }

  function deletePipe(id) {
    const pipe = getPipeByIdRef(id);
    if (!pipe) return false;

    mutate('deletePipe', null, () => {
      state.pipes = state.pipes.filter((p) => String(p.id) !== String(id));
    });
    return true;
  }

  function reroutePipe(id, newPoints, options) {
    const pipe = getPipeByIdRef(id);
    if (!pipe) return false;

    const cleanPoints = normalizePipePoints(newPoints);
    if (cleanPoints.length < 2) return false;

    mutate('reroutePipe', options, () => {
      pipe.points = cleanPoints;
    });
    return true;
  }

  function setLineStyle(id, style, options) {
    const pipe = getPipeByIdRef(id);
    if (!pipe) return false;

    const clean = String(style || '').trim().toLowerCase() === 'dashed' ? 'dashed' : 'solid';

    mutate('setLineStyle', options, () => {
      pipe.line_style = clean;
    });

    return true;
  }

  function undo() {
    if (!undoStack.length) return false;
    redoStack.push(deepClone(state));
    trimStack(redoStack);

    const prev = undoStack.pop();
    state.equipment = Array.isArray(prev.equipment) ? prev.equipment : [];
    state.pipes = Array.isArray(prev.pipes) ? prev.pipes : [];
    state.meta = prev.meta && typeof prev.meta === 'object' ? prev.meta : {};
    emitStateChanged('undo');
    return true;
  }

  function redo() {
    if (!redoStack.length) return false;
    undoStack.push(deepClone(state));
    trimStack(undoStack);

    const next = redoStack.pop();
    state.equipment = Array.isArray(next.equipment) ? next.equipment : [];
    state.pipes = Array.isArray(next.pipes) ? next.pipes : [];
    state.meta = next.meta && typeof next.meta === 'object' ? next.meta : {};
    emitStateChanged('redo');
    return true;
  }

  function getState() {
    return deepClone(state);
  }

  function getMeta() {
    return deepClone(state.meta);
  }

  function getById(id) {
    const eq = getEquipmentByIdRef(id);
    return eq ? deepClone(eq) : null;
  }

  function getPipeById(id) {
    const pipe = getPipeByIdRef(id);
    return pipe ? deepClone(pipe) : null;
  }

  function exportState() {
    // Keep exact geometry schema compatibility: only equipment + pipes.
    return {
      equipment: deepClone(state.equipment),
      pipes: deepClone(state.pipes),
    };
  }

  function generateEqId() {
    eqCounter += 1;
    return `EQ_${Date.now()}_${eqCounter}`;
  }

  function generatePipeId() {
    pipeCounter += 1;
    return `LP_${Date.now()}_${pipeCounter}`;
  }

  globalScope.DiagramState = {
    state,
    undoStack,
    redoStack,
    events: {
      STATE_CHANGED: STATE_CHANGED_EVENT,
    },
    load,
    pushUndoSnapshot,
    addEquipment,
    moveEquipment,
    setEquipmentPosition,
    deleteEquipment,
    updateLabel,
    changeType,
    addPipe,
    deletePipe,
    reroutePipe,
    setLineStyle,
    undo,
    redo,
    getState,
    getMeta,
    getById,
    getPipeById,
    export: exportState,
    generateEqId,
    generatePipeId,
  };
})(window);
