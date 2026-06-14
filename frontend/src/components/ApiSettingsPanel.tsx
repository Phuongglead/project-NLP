import React, { useEffect, useState } from "react";
import {
  clearApiSettingsOverride,
  DEFAULT_API_HOST,
  DEFAULT_API_PORT,
  getResolvedApiSettings,
  loadAppConfig,
  saveApiSettings,
} from "../config/apiConfig";
import { resetApiClient } from "../services/api";

const ApiSettingsPanel: React.FC = () => {
  const [open, setOpen] = useState(false);
  const [apiHost, setApiHost] = useState(DEFAULT_API_HOST);
  const [apiPort, setApiPort] = useState(DEFAULT_API_PORT);
  const [useRelativeApi, setUseRelativeApi] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    loadAppConfig().then(() =>
      getResolvedApiSettings().then((s) => {
        setApiHost(s.apiHost);
        setApiPort(s.apiPort);
        setUseRelativeApi(s.useRelativeApi);
      })
    );
  }, []);

  const handleSave = () => {
    saveApiSettings({ apiHost: apiHost.trim(), apiPort: Number(apiPort), useRelativeApi });
    resetApiClient();
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    clearApiSettingsOverride();
    resetApiClient();
    loadAppConfig().then(() =>
      getResolvedApiSettings().then((s) => {
        setApiHost(s.apiHost);
        setApiPort(s.apiPort);
        setUseRelativeApi(s.useRelativeApi);
      })
    );
  };

  return (
    <section className="api-settings">
      <button
        type="button"
        className="secondary api-settings-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        API server settings
      </button>
      {open && (
        <div className="api-settings-panel card">
          <p className="info">
            Point the UI at a remote FastAPI host (default <code>{DEFAULT_API_HOST}</code>).
            Server-side fallbacks (NER keywords / Gemini cache) are silent — check browser console and API logs.
          </p>
          <div className="grid-2">
            <div className="field">
              <label htmlFor="api-host">API host (IP or hostname)</label>
              <input
                id="api-host"
                type="text"
                value={apiHost}
                onChange={(e) => setApiHost(e.target.value)}
                placeholder={DEFAULT_API_HOST}
              />
            </div>
            <div className="field">
              <label htmlFor="api-port">API port</label>
              <input
                id="api-port"
                type="number"
                min={1}
                max={65535}
                value={apiPort}
                onChange={(e) => setApiPort(Number(e.target.value))}
              />
            </div>
          </div>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={useRelativeApi}
              onChange={(e) => setUseRelativeApi(e.target.checked)}
            />
            Use relative <code>/api</code> (Docker / Nginx proxy)
          </label>
          <div className="actions">
            <button type="button" className="primary" onClick={handleSave}>
              Apply
            </button>
            <button type="button" className="secondary" onClick={handleReset}>
              Reset to file defaults
            </button>
            {saved && <span className="feedback-saved">Applied</span>}
          </div>
        </div>
      )}
    </section>
  );
};

export default ApiSettingsPanel;
