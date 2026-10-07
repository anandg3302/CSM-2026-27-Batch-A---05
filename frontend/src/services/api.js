function resolveApiBaseUrl() {
  const configured = (import.meta.env.VITE_API_BASE_URL || '').trim();
  if (!configured) {
    return 'http://127.0.0.1:8000/api';
  }
  return configured
    .replace(/^http:\/\/localhost(?=[:/]|$)/i, 'http://127.0.0.1')
    .replace(/\/+$/, '');
}

export const API_BASE_URL = resolveApiBaseUrl();

/**
 * Helper to extract a human-readable error message from a failed HTTP response.
 */
export async function parseErrorResponse(response) {
  try {
    const payload = await response.json();
    if (payload && typeof payload.detail === 'string') {
      return payload.detail;
    }
  } catch {
    // Ignore JSON parse error
  }
  return `Request failed with HTTP status ${response.status}`;
}

/**
 * Fetch backend health status from GET /api/health.
 */
export async function fetchHealthStatus() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  return response.json();
}

/**
 * Upload a media file to POST /api/media/upload using FormData.
 * Explicitly treats HTTP 201 Created as a successful upload.
 */
export async function uploadMedia(file) {
  const formData = new FormData();
  formData.append('file', file);

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/media/upload`, {
      method: 'POST',
      body: formData,
    });
  } catch (networkError) {
    throw new Error(
      `Network error connecting to ${API_BASE_URL}/media/upload (${networkError?.message || 'Failed to fetch'}). Verify FastAPI is running on http://127.0.0.1:8000.`
    );
  }

  if (response.status !== 201 && !response.ok) {
    throw new Error(await parseErrorResponse(response));
  }

  let caseData;
  try {
    caseData = await response.json();
  } catch {
    throw new Error('Upload succeeded (HTTP 201), but the server returned an unreadable JSON body.');
  }

  if (!caseData || !caseData.case_id) {
    throw new Error('Upload succeeded (HTTP 201), but the response did not include a case_id.');
  }

  return caseData;
}

/**
 * Fetch stored case metadata for a single case ID from GET /api/media/{case_id}.
 */
export async function getMediaCase(caseId) {
  const response = await fetch(`${API_BASE_URL}/media/${encodeURIComponent(caseId)}`);
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  return response.json();
}

/**
 * Fetch all previously uploaded cases from GET /api/media.
 */
export async function getMediaCases() {
  const response = await fetch(`${API_BASE_URL}/media`);
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  return response.json();
}

/**
 * Trigger foundational image forensic analysis for a case via POST /api/analysis/image/{case_id}.
 */
export async function analyzeImage(caseId) {
  const response = await fetch(`${API_BASE_URL}/analysis/image/${encodeURIComponent(caseId)}`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  return response.json();
}

/**
 * Trigger classical image manipulation and AI-generation detection via POST /api/analysis/image/{case_id}/detect.
 */
export async function analyzeImageManipulation(caseId) {
  const response = await fetch(
    `${API_BASE_URL}/analysis/image/${encodeURIComponent(caseId)}/detect`,
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
 * Fetch stored AI-generated image detection result via GET /api/analysis/image/{case_id}/ai-detection.
 */
export async function getImageAiDetection(caseId) {
  const response = await fetch(
    `${API_BASE_URL}/analysis/image/${encodeURIComponent(caseId)}/ai-detection`
  );
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  return response.json();
}

/**
 * Trigger spatial manipulation localization via POST /api/analysis/image/{case_id}/localize.
 */
export async function localizeImage(caseId) {
  const response = await fetch(
    `${API_BASE_URL}/analysis/image/${encodeURIComponent(caseId)}/localize`,
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
 * Return URL for a generated localization image artifact ('heatmap.png' or 'mask.png').
 */
export function getLocalizationArtifactUrl(caseId, artifactName = 'heatmap.png', cacheBust = '') {
  const suffix = cacheBust ? `?t=${encodeURIComponent(cacheBust)}` : '';
  return `${API_BASE_URL}/analysis/image/${encodeURIComponent(caseId)}/localization/${encodeURIComponent(
    artifactName
  )}${suffix}`;
}

/**
 * Upload a reference/original image for comparison via POST /api/comparison/image/{case_id}/reference.
 */
export async function uploadReferenceImage(caseId, file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(
    `${API_BASE_URL}/comparison/image/${encodeURIComponent(caseId)}/reference`,
    {
      method: 'POST',
      body: formData,
    }
  );

  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }

  return response.json();
}

/**
 * Execute Original vs Edited comparison via POST /api/comparison/image/{case_id}.
 */
export async function compareImages(caseId) {
  const response = await fetch(
    `${API_BASE_URL}/comparison/image/${encodeURIComponent(caseId)}`,
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
 * Return URL for a generated comparison artifact ('difference.png' or 'difference_mask.png').
 */
export function getComparisonArtifactUrl(caseId, artifactName = 'difference.png', cacheBust = '') {
  const suffix = cacheBust ? `?t=${encodeURIComponent(cacheBust)}` : '';
  return `${API_BASE_URL}/comparison/image/${encodeURIComponent(caseId)}/${encodeURIComponent(
    artifactName
  )}${suffix}`;
}

/**
 * Generate a correlated forensic EvidenceReport via POST /api/evidence/{case_id}/generate.
 */
export async function generateEvidenceReport(caseId) {
  const response = await fetch(
    `${API_BASE_URL}/evidence/${encodeURIComponent(caseId)}/generate`,
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
 * Fetch the stored EvidenceReport for a case via GET /api/evidence/{case_id}.
 */
export async function getEvidenceReport(caseId) {
  const response = await fetch(`${API_BASE_URL}/evidence/${encodeURIComponent(caseId)}`);
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  return response.json();
}

/**
 * Execute the conservative multi-signal forensic decision & interpretation layer
 * via POST /api/forensics/{case_id}/interpret.
 */
export async function interpretForensics(caseId) {
  const response = await fetch(
    `${API_BASE_URL}/forensics/${encodeURIComponent(caseId)}/interpret`,
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
 * Fetch the stored ForensicDecisionResult for a case via GET /api/forensics/{case_id}/interpret.
 */
export async function getForensicInterpretation(caseId) {
  const response = await fetch(
    `${API_BASE_URL}/forensics/${encodeURIComponent(caseId)}/interpret`
  );
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  return response.json();
}

/**
 * Reconstruct a conservative possible edit-history sequence via POST /api/history/{case_id}/reconstruct.
 */
export async function reconstructEditHistory(caseId) {
  const response = await fetch(
    `${API_BASE_URL}/history/${encodeURIComponent(caseId)}/reconstruct`,
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
 * Fetch the stored EditHistoryReport for a case via GET /api/history/{case_id}.
 */
export async function getEditHistory(caseId) {
  const response = await fetch(`${API_BASE_URL}/history/${encodeURIComponent(caseId)}`);
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  return response.json();
}

/**
 * Generate a structured JSON ForensicReport for a case via POST /api/reports/{case_id}/generate.
 */
export async function generateForensicReport(caseId) {
  const response = await fetch(
    `${API_BASE_URL}/reports/${encodeURIComponent(caseId)}/generate`,
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
 * Fetch the stored JSON ForensicReport for a case via GET /api/reports/{case_id}.
 */
export async function getForensicReport(caseId) {
  const response = await fetch(`${API_BASE_URL}/reports/${encodeURIComponent(caseId)}`);
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  return response.json();
}

/**
 * Generate and download the PDF ForensicReport via POST /api/reports/{case_id}/generate-pdf.
 */
export async function generateForensicReportPdf(caseId) {
  const response = await fetch(
    `${API_BASE_URL}/reports/${encodeURIComponent(caseId)}/generate-pdf`,
    {
      method: 'POST',
    }
  );
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  return response.blob();
}

/**
 * Execute a non-destructive forensic reproducibility check via POST /api/forensics/{case_id}/reproducibility/check.
 */
export async function checkCaseReproducibility(caseId) {
  const response = await fetch(
    `${API_BASE_URL}/forensics/${encodeURIComponent(caseId)}/reproducibility/check`,
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
 * Fetch the immutable ForensicAnalysisSnapshot and audit trail via GET /api/forensics/{case_id}/reproducibility.
 */
export async function fetchCaseReproducibility(caseId) {
  const response = await fetch(
    `${API_BASE_URL}/forensics/${encodeURIComponent(caseId)}/reproducibility`
  );
  if (!response.ok) {
    throw new Error(await parseErrorResponse(response));
  }
  return response.json();
}

export { analyzeVideo, getVideoArtifactUrl } from './videoAnalysisService.js';
export { analyzeAudio, getAudioArtifactUrl } from './audioAnalysisService.js';
export { analyzeDocument, getDocumentArtifactUrl } from './documentAnalysisService.js';





