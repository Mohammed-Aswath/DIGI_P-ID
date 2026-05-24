'use strict';

// SaveManager: cascaded save call (/save_diagram) + status toast + optional DXF download.

const SaveManager = (() => {
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

  function boolBadge(label, isOk) {
    return `<span class=\"save-badge ${isOk ? 'save-ok' : 'save-fail'}\">${label} ${isOk ? 'OK' : 'FAIL'}</span>`;
  }

  function showToast(title, outputs, isError) {
    const toast = document.createElement('div');
    toast.className = 'save-toast';

    if (isError) {
      toast.innerHTML = `<strong>${title}</strong>`;
    } else {
      const badges = [
        boolBadge('JSON', !!(outputs && outputs.json && outputs.json.saved)),
        boolBadge('XML', !!(outputs && outputs.xml && outputs.xml.saved)),
        boolBadge('ISO 15926', !!(outputs && outputs.iso15926 && outputs.iso15926.saved)),
        boolBadge('DXF', !!(outputs && outputs.dxf && outputs.dxf.available)),
      ].join('');
      toast.innerHTML = `<strong>${title}</strong><div class=\"save-badges\">${badges}</div>`;
    }

    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), isError ? 6500 : 5000);
  }

  function triggerDownload(url, filename) {
    if (!url) return;
    const a = document.createElement('a');
    a.href = String(url).startsWith('/') ? apiUrl(url) : url;
    a.download = filename || '';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  async function save() {
    const btn = document.getElementById('saveDiagramBtn');
    if (btn) {
      btn.disabled = true;
      btn.textContent = 'Saving...';
    }

    try {
      const geometry = window.DiagramState.export();
      const meta = window.DiagramState.getMeta();
      const payload = {
        analysis_id: String(meta.analysis_id || ''),
        geometry,
        image_height: Number(meta.image_height) || 0,
        image_width: Number(meta.image_width) || 0,
      };

      const resp = await fetch(apiUrl('/save_diagram'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();

      if (!resp.ok || !data.success) {
        showToast(`Save failed: ${data.error || `HTTP ${resp.status}`}`, null, true);
        return;
      }

      const outputs = data.outputs || {};
      showToast('Diagram saved', outputs, false);

      if (outputs.dxf && outputs.dxf.available && outputs.dxf.download_url) {
        triggerDownload(outputs.dxf.download_url, outputs.dxf.filename || 'diagram.dxf');
      }
    } catch (error) {
      showToast(`Save failed: ${error.message}`, null, true);
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = 'Save Diagram';
      }
    }
  }

  function init() {
    const btn = document.getElementById('saveDiagramBtn');
    if (btn) {
      btn.addEventListener('click', save);
    }
  }

  return {
    init,
    save,
  };
})();

window.SaveManager = SaveManager;
