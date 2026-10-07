import { API_BASE_URL, parseErrorResponse } from './api.js';

/**
 * Trigger Step 12 document forensic analysis for a document case via POST /api/analysis/document/{case_id}.
 */
export async function analyzeDocument(caseId) {
  const response = await fetch(
    `${API_BASE_URL}/analysis/document/${encodeURIComponent(caseId)}`,
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
 * Return URL for a derived document preview artifact ('page_001.png', etc.).
 */
export function getDocumentArtifactUrl(caseId, artifactName, cacheBust = '') {
  const suffix = cacheBust ? `?t=${encodeURIComponent(cacheBust)}` : '';
  return `${API_BASE_URL}/analysis/document/${encodeURIComponent(
    caseId
  )}/artifacts/${encodeURIComponent(artifactName)}${suffix}`;
}
