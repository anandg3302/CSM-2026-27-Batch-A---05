import { API_BASE_URL, parseErrorResponse } from './api.js';

/**
 * Trigger Step 11 audio forensic analysis for an audio case via POST /api/analysis/audio/{case_id}.
 */
export async function analyzeAudio(caseId) {
  const response = await fetch(
    `${API_BASE_URL}/analysis/audio/${encodeURIComponent(caseId)}`,
    {
      method: 'POST',
    }
  );
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  return response.json();
}

/**
 * Return URL for a derived audio forensic artifact ('waveform.png', 'spectrogram.png', 'discontinuity_<ms>.png').
 */
export function getAudioArtifactUrl(caseId, artifactName, cacheBust = '') {
  const suffix = cacheBust ? `?t=${encodeURIComponent(cacheBust)}` : '';
  return `${API_BASE_URL}/analysis/audio/${encodeURIComponent(
    caseId
  )}/artifacts/${encodeURIComponent(artifactName)}${suffix}`;
}
