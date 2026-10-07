import React from 'react';

const PIPELINES = [
  {
    id: 'image',
    title: 'Image Forensics',
    description: 'Pixel-level tamper detection, noise inconsistency, and synthetic generation artifacts.',
    status: 'Planned Pipeline',
  },
  {
    id: 'video',
    title: 'Video Forensics',
    description: 'Temporal consistency verification, frame manipulation, and deepfake region tracking.',
    status: 'Planned Pipeline',
  },
  {
    id: 'audio',
    title: 'Audio Forensics',
    description: 'Spectral splice detection, voice synthesis identification, and acoustic provenance.',
    status: 'Planned Pipeline',
  },
  {
    id: 'document',
    title: 'Document Forensics',
    description: 'Metadata inspection, font/layout alteration detection, and structural integrity.',
    status: 'Planned Pipeline',
  },
];

export default function DashboardPlaceholder() {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Forensic Overview Dashboard</h2>
        <span className="badge">Step 1: Foundation</span>
      </div>
      <p className="panel-description">
        Modular architecture initialized for multi-modal media intake, evidence localization,
        version comparison, and editing history reconstruction.
      </p>

      <div className="pipeline-grid">
        {PIPELINES.map((pipeline) => (
          <div key={pipeline.id} className="pipeline-card">
            <div className="pipeline-card-top">
              <h3>{pipeline.title}</h3>
              <span className="pipeline-tag">{pipeline.status}</span>
            </div>
            <p>{pipeline.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
