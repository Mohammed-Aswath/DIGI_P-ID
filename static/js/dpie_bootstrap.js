'use strict';

// DPIE Bootstrap: module wiring + view toggles + symbol template loading.

const DPIE = (() => {
  let initialized = false;
  let symbolTemplatesLoaded = false;

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

  async function loadSymbolTemplates(forceReload) {
    if (symbolTemplatesLoaded && !forceReload) return;
    try {
      const resp = await fetch(apiUrl('/symbol_templates'));
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }
      window.symbolTemplates = data || {};
      symbolTemplatesLoaded = true;
    } catch (error) {
      console.error('[DPIE] Failed to load symbol templates:', error);
      window.symbolTemplates = window.symbolTemplates || {};
      symbolTemplatesLoaded = false;
    }
  }

  function showEditorView() {
    const resultsMain = document.getElementById('resultsMainView');
    const editorSection = document.getElementById('editorSection');
    const switchToEditorBtn = document.getElementById('switchToEditorBtn');
    const switchToResultsBtn = document.getElementById('switchToResultsBtn');

    if (resultsMain) resultsMain.classList.add('hidden');
    if (editorSection) editorSection.classList.remove('hidden');
    if (switchToEditorBtn) switchToEditorBtn.classList.add('hidden');
    if (switchToResultsBtn) switchToResultsBtn.classList.remove('hidden');

    if (window.PidRenderer) {
      window.PidRenderer.renderAll();
      window.PidRenderer.setZoom(1);
    }
  }

  function showResultsView() {
    const resultsMain = document.getElementById('resultsMainView');
    const editorSection = document.getElementById('editorSection');
    const switchToEditorBtn = document.getElementById('switchToEditorBtn');
    const switchToResultsBtn = document.getElementById('switchToResultsBtn');

    if (resultsMain) resultsMain.classList.remove('hidden');
    if (editorSection) editorSection.classList.add('hidden');
    if (switchToEditorBtn) switchToEditorBtn.classList.remove('hidden');
    if (switchToResultsBtn) switchToResultsBtn.classList.add('hidden');
  }

  function bindUiButtons() {
    const switchToEditorBtn = document.getElementById('switchToEditorBtn');
    const switchToResultsBtn = document.getElementById('switchToResultsBtn');

    if (switchToEditorBtn) {
      switchToEditorBtn.addEventListener('click', showEditorView);
    }
    if (switchToResultsBtn) {
      switchToResultsBtn.addEventListener('click', showResultsView);
    }
  }

  function onDigitizeSuccess(payload) {
    const switchToEditorBtn = document.getElementById('switchToEditorBtn');
    if (switchToEditorBtn) {
      switchToEditorBtn.disabled = false;
      switchToEditorBtn.classList.remove('hidden');
    }

    const templates = window.symbolTemplates || {};
    if (!Object.keys(templates).length) {
      loadSymbolTemplates(true).then(() => {
        if (window.EditorInteraction && typeof window.EditorInteraction.populatePalette === 'function') {
          window.EditorInteraction.populatePalette();
        }
      });
    } else if (window.EditorInteraction && typeof window.EditorInteraction.populatePalette === 'function') {
      window.EditorInteraction.populatePalette();
    }

    if (window.PidRenderer) {
      window.PidRenderer.clearSelection();
      window.PidRenderer.renderAll();
    }

    if (payload && payload.analysis_id) {
      const analysisIdBadge = document.getElementById('editorAnalysisId');
      if (analysisIdBadge) analysisIdBadge.textContent = payload.analysis_id;
    }
  }

  async function init() {
    if (initialized) return;

    await loadSymbolTemplates();

    if (window.PidRenderer) {
      window.PidRenderer.init('pid-editor-svg');
    }

    if (window.EditorInteraction) {
      window.EditorInteraction.init('pid-editor-svg');
      window.EditorInteraction.populatePalette();
    }

    if (window.AgentChat) {
      window.AgentChat.init();
    }

    if (window.SaveManager) {
      window.SaveManager.init();
    }

    bindUiButtons();
    showResultsView();

    initialized = true;
  }

  document.addEventListener('digitizeSuccess', (event) => {
    onDigitizeSuccess(event.detail || {});
  });

  return {
    init,
    onDigitizeSuccess,
    showEditorView,
    showResultsView,
  };
})();

window.DPIE = DPIE;

document.addEventListener('DOMContentLoaded', () => {
  DPIE.init();
});
