'use strict';

// PidRenderer: SVG renderer for DiagramState (image pixel coordinate space).
// Rendering applies display scaling only through SVG viewBox + CSS sizing.

const PidRenderer = (() => {
  const NS = 'http://www.w3.org/2000/svg';

  let svgEl = null;
  let bgLayer = null;
  let pipeLayer = null;
  let symbolLayer = null;
  let labelLayer = null;
  let selectionLayer = null;
  let drawLayer = null;

  let selected = { id: null, type: null };
  let showBackground = true;
  let zoom = 1;

  function svgNode(tag, attrs) {
    const el = document.createElementNS(NS, tag);
    if (attrs && typeof attrs === 'object') {
      Object.entries(attrs).forEach(([k, v]) => {
        if (v === undefined || v === null || v === '') return;
        el.setAttribute(k, String(v));
      });
    }
    return el;
  }

  function clearLayer(layer) {
    if (!layer) return;
    while (layer.firstChild) layer.removeChild(layer.firstChild);
  }

  function normalizePointToLocal(x, y) {
    const nx = Number(x);
    const ny = Number(y);
    if (!Number.isFinite(nx) || !Number.isFinite(ny)) return [0, 0];

    if (nx >= 0 && nx <= 1 && ny >= 0 && ny <= 1) {
      return [(nx - 0.5) * 100.0, (0.5 - ny) * 100.0];
    }

    if (nx >= 0 && nx <= 150 && ny >= 0 && ny <= 150) {
      return [((nx / 128.0) - 0.5) * 100.0, (0.5 - (ny / 128.0)) * 100.0];
    }

    return [nx, ny];
  }

  function normalizeCircleToLocal(x, y, r) {
    const cx = Number(x);
    const cy = Number(y);
    const cr = Number(r);
    if (!Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(cr)) return [0, 0, 0];

    if (cx >= 0 && cx <= 1 && cy >= 0 && cy <= 1 && cr >= 0 && cr <= 1) {
      const [lx, ly] = normalizePointToLocal(cx, cy);
      return [lx, ly, cr * 100.0];
    }

    if (cx >= 0 && cx <= 150 && cy >= 0 && cy <= 150 && cr >= 0 && cr <= 80) {
      const [lx, ly] = normalizePointToLocal(cx, cy);
      return [lx, ly, (cr / 128.0) * 100.0];
    }

    return [cx, cy, cr];
  }

  function localToGlobal(localX, localY, centerX, centerY, scale) {
    return [centerX + localX * scale, centerY - localY * scale];
  }

  function fitSvgToImage(meta) {
    if (!svgEl) return;
    const imageWidth = Math.max(1, Number(meta.image_width) || 1);
    const imageHeight = Math.max(1, Number(meta.image_height) || 1);
    svgEl.setAttribute('viewBox', `0 0 ${imageWidth} ${imageHeight}`);
    svgEl.style.width = `${Math.round(imageWidth * zoom)}px`;
    svgEl.style.height = `${Math.round(imageHeight * zoom)}px`;
  }

  function renderBackground(meta) {
    if (!bgLayer) return;
    const imageWidth = Math.max(1, Number(meta.image_width) || 1);
    const imageHeight = Math.max(1, Number(meta.image_height) || 1);
    bgLayer.setAttribute('x', '0');
    bgLayer.setAttribute('y', '0');
    bgLayer.setAttribute('width', String(imageWidth));
    bgLayer.setAttribute('height', String(imageHeight));
    bgLayer.setAttribute('opacity', showBackground ? '0.4' : '0');
    if (meta.original_image_b64) {
      bgLayer.setAttributeNS('http://www.w3.org/1999/xlink', 'href', meta.original_image_b64);
      bgLayer.setAttribute('href', meta.original_image_b64);
    }
  }

  function renderPipes(pipes) {
    clearLayer(pipeLayer);
    if (!Array.isArray(pipes)) return;

    pipes.forEach((pipe) => {
      if (!pipe || !Array.isArray(pipe.points) || pipe.points.length < 2) return;
      const pointsString = pipe.points
        .map((pt) => `${Number(pt[0])},${Number(pt[1])}`)
        .join(' ');

      const isSelected = selected.type === 'pipe' && String(selected.id) === String(pipe.id);
      const dashed = String(pipe.line_style || 'solid').toLowerCase() === 'dashed';

      const poly = svgNode('polyline', {
        points: pointsString,
        fill: 'none',
        stroke: isSelected ? '#2563eb' : (dashed ? '#b45309' : '#111827'),
        'stroke-width': isSelected ? 3 : 1.7,
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
        'stroke-dasharray': dashed ? '7,5' : '',
        'data-pipe-id': String(pipe.id),
      });
      pipeLayer.appendChild(poly);
    });
  }

  function renderTemplateSymbol(group, template, centerX, centerY, symbolSize) {
    const prim = (template && template.primitives) || {};
    const fills = (template && template.fill) || {};
    const scale = symbolSize / 100.0;

    const circleFills = Array.isArray(fills.circles) ? fills.circles : [];
    const contourFills = Array.isArray(fills.contours) ? fills.contours : [];

    const circles = Array.isArray(prim.circles) ? prim.circles : [];
    circles.forEach((circle, idx) => {
      if (!Array.isArray(circle) || circle.length < 3) return;
      const [lx, ly, lr] = normalizeCircleToLocal(circle[0], circle[1], circle[2]);
      const [gx, gy] = localToGlobal(lx, ly, centerX, centerY, scale);
      const fillMode = circleFills[idx] === 'filled' ? '#111827' : 'none';
      group.appendChild(svgNode('circle', {
        cx: gx,
        cy: gy,
        r: Math.max(0.4, lr * scale),
        fill: fillMode,
        stroke: '#111827',
        'stroke-width': 1.5,
      }));
    });

    const contours = Array.isArray(prim.contours) ? prim.contours : [];
    contours.forEach((contour, idx) => {
      if (!Array.isArray(contour) || contour.length < 2) return;
      const points = contour
        .map((pt) => {
          if (!Array.isArray(pt) || pt.length < 2) return null;
          const [lx, ly] = normalizePointToLocal(pt[0], pt[1]);
          const [gx, gy] = localToGlobal(lx, ly, centerX, centerY, scale);
          return `${gx},${gy}`;
        })
        .filter(Boolean)
        .join(' ');
      if (!points) return;
      const fillMode = contourFills[idx] === 'filled' ? '#111827' : 'none';
      group.appendChild(svgNode('polygon', {
        points,
        fill: fillMode,
        stroke: '#111827',
        'stroke-width': 1.5,
        'stroke-linejoin': 'round',
      }));
    });

    const lines = Array.isArray(prim.lines) ? prim.lines : [];
    lines.forEach((line) => {
      if (!Array.isArray(line) || line.length < 4) return;
      const [l1x, l1y] = normalizePointToLocal(line[0], line[1]);
      const [l2x, l2y] = normalizePointToLocal(line[2], line[3]);
      const [x1, y1] = localToGlobal(l1x, l1y, centerX, centerY, scale);
      const [x2, y2] = localToGlobal(l2x, l2y, centerX, centerY, scale);
      group.appendChild(svgNode('line', {
        x1,
        y1,
        x2,
        y2,
        stroke: '#111827',
        'stroke-width': 1.5,
        'stroke-linecap': 'round',
      }));
    });

    const texts = Array.isArray(prim.texts) ? prim.texts : (Array.isArray(template.texts) ? template.texts : []);
    texts.forEach((txt) => {
      if (!Array.isArray(txt) || txt.length < 3) return;
      const [rawX, rawY, content, rawScale] = txt;
      const [lx, ly] = normalizePointToLocal(rawX, rawY);
      const [gx, gy] = localToGlobal(lx, ly, centerX, centerY, scale);
      const textScale = Number.isFinite(Number(rawScale)) ? Number(rawScale) : 0.08;
      const fontSize = Math.max(5, symbolSize * textScale * 0.85);
      const t = svgNode('text', {
        x: gx,
        y: gy,
        'text-anchor': 'middle',
        'dominant-baseline': 'middle',
        fill: '#111827',
        'font-family': 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
        'font-size': fontSize,
        'font-weight': '600',
        'pointer-events': 'none',
      });
      t.textContent = String(content);
      group.appendChild(t);
    });
  }

  function renderFallbackSymbol(group, centerX, centerY, symbolSize) {
    const h = symbolSize * 0.5;
    group.appendChild(svgNode('rect', {
      x: centerX - h,
      y: centerY - h,
      width: symbolSize,
      height: symbolSize,
      fill: 'none',
      stroke: '#64748b',
      'stroke-width': 1.4,
    }));
  }

  function renderSymbols(equipment) {
    clearLayer(symbolLayer);
    if (!Array.isArray(equipment)) return;

    equipment.forEach((eq) => {
      if (!eq || !Array.isArray(eq.position) || eq.position.length < 2) return;

      const cx = Number(eq.position[0]);
      const cy = Number(eq.position[1]);
      if (!Number.isFinite(cx) || !Number.isFinite(cy)) return;

      const symbolSize = Math.max(8, Number(eq.symbol_size) || 36);
      const group = svgNode('g', {
        'data-eq-id': String(eq.id),
        'data-eq-type': String(eq.type || 'unknown'),
        class: 'pid-symbol',
      });

      const template = window.symbolTemplates && window.symbolTemplates[String(eq.type || '').trim()];
      if (template) {
        renderTemplateSymbol(group, template, cx, cy, symbolSize);
      } else {
        renderFallbackSymbol(group, cx, cy, symbolSize);
      }

      const hitSize = Math.max(symbolSize, 24);
      group.appendChild(svgNode('rect', {
        x: cx - hitSize * 0.5,
        y: cy - hitSize * 0.5,
        width: hitSize,
        height: hitSize,
        fill: 'transparent',
        stroke: 'none',
        'data-hit': '1',
      }));

      symbolLayer.appendChild(group);
    });
  }

  function renderLabels(equipment) {
    clearLayer(labelLayer);
    if (!Array.isArray(equipment)) return;

    equipment.forEach((eq) => {
      if (!eq || !eq.label || !Array.isArray(eq.position) || eq.position.length < 2) return;
      const cx = Number(eq.position[0]);
      const cy = Number(eq.position[1]);
      const symbolSize = Math.max(8, Number(eq.symbol_size) || 36);
      const fontSize = Math.max(8, Math.min(13, symbolSize * 0.36));

      const textEl = svgNode('text', {
        x: cx,
        y: cy - symbolSize * 0.65,
        'text-anchor': 'middle',
        fill: '#1e3a5f',
        'font-family': 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
        'font-size': fontSize,
        'font-weight': '600',
        'pointer-events': 'none',
      });
      textEl.textContent = String(eq.label);
      labelLayer.appendChild(textEl);
    });
  }

  function renderSelection(state) {
    clearLayer(selectionLayer);
    if (!selected.id || !selected.type) return;

    if (selected.type === 'equipment') {
      const eq = Array.isArray(state.equipment)
        ? state.equipment.find((item) => String(item.id) === String(selected.id))
        : null;
      if (!eq || !Array.isArray(eq.position)) return;

      const cx = Number(eq.position[0]);
      const cy = Number(eq.position[1]);
      let width = Math.max(14, Number(eq.symbol_size) || 36);
      let height = width;

      if (Array.isArray(eq.bbox) && eq.bbox.length === 4) {
        const bw = Math.abs(Number(eq.bbox[2]) - Number(eq.bbox[0]));
        const bh = Math.abs(Number(eq.bbox[3]) - Number(eq.bbox[1]));
        if (Number.isFinite(bw) && bw > 2) width = bw;
        if (Number.isFinite(bh) && bh > 2) height = bh;
      }

      const pad = 4;
      selectionLayer.appendChild(svgNode('rect', {
        x: cx - width * 0.5 - pad,
        y: cy - height * 0.5 - pad,
        width: width + pad * 2,
        height: height + pad * 2,
        fill: 'none',
        stroke: '#2563eb',
        'stroke-width': 2,
        'stroke-dasharray': '5,3',
        rx: 3,
      }));
      return;
    }

    if (selected.type === 'pipe') {
      const pipe = Array.isArray(state.pipes)
        ? state.pipes.find((item) => String(item.id) === String(selected.id))
        : null;
      if (!pipe || !Array.isArray(pipe.points) || pipe.points.length < 2) return;

      const pointsString = pipe.points.map((pt) => `${pt[0]},${pt[1]}`).join(' ');
      selectionLayer.appendChild(svgNode('polyline', {
        points: pointsString,
        fill: 'none',
        stroke: '#2563eb',
        'stroke-width': 3,
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
      }));

      const start = pipe.points[0];
      const end = pipe.points[pipe.points.length - 1];
      const midIndex = pipe.points.length >= 3 ? Math.floor(pipe.points.length / 2) : null;
      const mid = midIndex != null ? pipe.points[midIndex] : [(start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5];

      selectionLayer.appendChild(svgNode('circle', {
        cx: Number(mid[0]),
        cy: Number(mid[1]),
        r: 5,
        fill: '#ffffff',
        stroke: '#2563eb',
        'stroke-width': 2,
        'data-pipe-mid-handle': String(pipe.id),
      }));
    }
  }

  function renderAll() {
    if (!svgEl || !window.DiagramState) return;
    const state = window.DiagramState.getState();
    const meta = state.meta || {};

    fitSvgToImage(meta);
    renderBackground(meta);
    renderPipes(state.pipes || []);
    renderSymbols(state.equipment || []);
    renderLabels(state.equipment || []);
    renderSelection(state);
  }

  function init(svgId) {
    svgEl = document.getElementById(svgId);
    if (!svgEl) return;

    bgLayer = document.getElementById('bg-layer');
    pipeLayer = document.getElementById('pipe-layer');
    symbolLayer = document.getElementById('symbol-layer');
    labelLayer = document.getElementById('label-layer');
    selectionLayer = document.getElementById('selection-layer');
    drawLayer = document.getElementById('draw-layer');

    document.addEventListener('stateChanged', renderAll);
    window.addEventListener('resize', renderAll);
  }

  function setSelection(id, type) {
    selected = { id: id == null ? null : String(id), type: type || null };
    renderAll();
  }

  function clearSelection() {
    selected = { id: null, type: null };
    renderAll();
  }

  function getSelection() {
    return { ...selected };
  }

  function setZoom(nextZoom) {
    const nz = Number(nextZoom);
    if (!Number.isFinite(nz)) return;
    zoom = Math.max(0.25, Math.min(8, nz));
    renderAll();
    const lbl = document.getElementById('editorZoomLabel');
    if (lbl) lbl.textContent = `${Math.round(zoom * 100)}%`;
  }

  function zoomBy(factor) {
    setZoom(zoom * Number(factor || 1));
  }

  function getZoom() {
    return zoom;
  }

  function setBackgroundVisible(visible) {
    showBackground = !!visible;
    renderAll();
  }

  function clearDrawLayer() {
    clearLayer(drawLayer);
  }

  function drawDraftLine(startPoint, endPoint) {
    if (!drawLayer || !Array.isArray(startPoint) || !Array.isArray(endPoint)) return;
    clearLayer(drawLayer);
    drawLayer.appendChild(svgNode('line', {
      x1: Number(startPoint[0]),
      y1: Number(startPoint[1]),
      x2: Number(endPoint[0]),
      y2: Number(endPoint[1]),
      stroke: '#2563eb',
      'stroke-width': 2,
      'stroke-dasharray': '6,4',
      'pointer-events': 'none',
    }));
  }

  function clientToImage(clientX, clientY) {
    if (!svgEl) return { x: 0, y: 0 };
    const rect = svgEl.getBoundingClientRect();
    const vb = svgEl.viewBox.baseVal;
    const rx = rect.width > 0 ? (clientX - rect.left) / rect.width : 0;
    const ry = rect.height > 0 ? (clientY - rect.top) / rect.height : 0;
    return {
      x: vb.x + rx * vb.width,
      y: vb.y + ry * vb.height,
    };
  }

  function renderMiniSymbol(targetSvgEl, symbolType, size) {
    if (!targetSvgEl) return;
    const iconSize = Math.max(16, Number(size) || 44);
    targetSvgEl.setAttribute('viewBox', `0 0 ${iconSize} ${iconSize}`);
    targetSvgEl.setAttribute('width', String(iconSize));
    targetSvgEl.setAttribute('height', String(iconSize));
    while (targetSvgEl.firstChild) targetSvgEl.removeChild(targetSvgEl.firstChild);

    const center = iconSize * 0.5;
    const symbolSize = iconSize * 0.85;
    const template = window.symbolTemplates && window.symbolTemplates[symbolType];

    const group = svgNode('g', {});
    if (template) {
      renderTemplateSymbol(group, template, center, center, symbolSize);
    } else {
      renderFallbackSymbol(group, center, center, symbolSize);
    }
    targetSvgEl.appendChild(group);
  }

  return {
    init,
    renderAll,
    setSelection,
    clearSelection,
    getSelection,
    setZoom,
    zoomBy,
    getZoom,
    setBackgroundVisible,
    clearDrawLayer,
    drawDraftLine,
    clientToImage,
    renderMiniSymbol,
  };
})();

window.PidRenderer = PidRenderer;
