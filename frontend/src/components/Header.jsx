import React from 'react';

export default function Header({ apiStatus, serviceName }) {
  const isOnline = apiStatus === 'ok';

  return (
    <header className="app-header">
      <div className="header-brand">
        <div className="brand-badge">MT</div>
        <div>
          <h1 className="app-title">MediaTrace</h1>
          <p className="app-subtitle">Digital Media Forensics &amp; Provenance Analysis</p>
        </div>
      </div>

      <div className={`api-status-pill ${isOnline ? 'status-online' : 'status-offline'}`}>
        <span className="status-dot" />
        <span>
          {isOnline ? `${serviceName} Connected` : 'Backend Offline (Start FastAPI on :8000)'}
        </span>
      </div>
    </header>
  );
}
