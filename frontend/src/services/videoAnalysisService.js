import { API_BASE_URL, parseErrorResponse } from './api.js';

/**
 * Trigger Step 10 video forensic analysis for a video case via POST /api/analysis/video/{case_id}.
 */
export async function analyzeVideo(caseId) {
  const response = await fetch(
    `${API_BASE_URL}/analysis/video/${encodeURIComponent(caseId)}`,
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
 * Return URL for an extracted video frame artifact ('frame_0001.jpg', 'suspicious_frame_0004.jpg', etc.).
 */
export function getVideoArtifactUrl(caseId, artifactName, cacheBust = '') {
  const suffix = cacheBust ? `?t=${encodeURIComponent(cacheBust)}` : '';
  return `${API_BASE_URL}/analysis/video/${encodeURIComponent(
    caseId
  )}/artifacts/${encodeURIComponent(artifactName)}${suffix}`;
}
