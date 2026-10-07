import React from 'react';

const RESULT_SECTIONS = [
  {
    label: 'Authenticity & Manipulation Summary',
    detail: 'Confidence indicators and observed forensic anomalies will populate here.',
  },
  {
    label: 'Region & Segment Localization',
    detail: 'Spatial heatmaps, bounding boxes, or timeline markers for suspicious segments.',
  },
  {
    label: 'Version Comparison & Editing History',
    detail: 'Step-by-step reconstruction of plausible transformations and modifications.',
  },
  {
    label: 'Forensic Report Generation',
    detail: 'Exportable structured evidence summary stored under storage/reports/.',
  },
];

export default function ResultsPlaceholder({ activeCase, recentCases = [], onSelectCase }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Forensic Workspace &amp; Case Registry</h2>
        <span className="badge">
          {activeCase ? `Active: ${activeCase.case_id}` : 'Awaiting Analysis'}
        </span>
      </div>
      <p className="panel-description">
        {activeCase
          ? `Selected case ${activeCase.case_id} (${activeCase.original_filename}) is registered and ready for future forensic pipelines.`
          : 'No media analysis has been executed yet. Upload an image to initialize a forensic case.'}
      </p>

      {recentCases.length > 0 && (
        <div className="recent-cases-box">
          <h3 className="sub-heading">Registered Cases ({recentCases.length})</h3>
          <div className="case-pills">
            {recentCases.map((item) => (
              <button
                key={item.case_id}
                type="button"
                className={`case-pill ${activeCase?.case_id === item.case_id ? 'active' : ''}`}
                onClick={() => onSelectCase && onSelectCase(item.case_id)}
              >
                <span className="mono">{item.case_id}</span>
                <span className="case-pill-name">{item.original_filename}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="results-list">
        {RESULT_SECTIONS.map((item) => (
          <div key={item.label} className="result-placeholder-row">
            <h4>{item.label}</h4>
            <p>{item.detail}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
