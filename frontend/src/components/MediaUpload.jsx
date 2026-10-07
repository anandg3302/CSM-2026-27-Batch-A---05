import React, { useEffect, useRef, useState } from 'react';
import {
  analyzeImage,
  analyzeImageManipulation,
  checkCaseReproducibility,
  compareImages,
  fetchCaseReproducibility,
  generateEvidenceReport,
  generateForensicReport,
  generateForensicReportPdf,
  getComparisonArtifactUrl,
  getForensicReport,
  getLocalizationArtifactUrl,
  getMediaCase,
  interpretForensics,
  localizeImage,
  reconstructEditHistory,
  uploadMedia,
  uploadReferenceImage,
} from '../services/api.js';
import { analyzeVideo, getVideoArtifactUrl } from '../services/videoAnalysisService.js';
import { analyzeAudio, getAudioArtifactUrl } from '../services/audioAnalysisService.js';
import { analyzeDocument, getDocumentArtifactUrl } from '../services/documentAnalysisService.js';

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  if (!bytes) return '-';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / Math.pow(1024, index);
  return `${value.toFixed(index === 0 ? 0 : 2)} ${units[index]} (${bytes.toLocaleString()} bytes)`;
}

function formatAssessmentText(assessmentCode, fallbackLabel) {
  if (fallbackLabel) return fallbackLabel;
  switch (assessmentCode) {
    case 'no_strong_indicators':
      return 'No strong manipulation indicators detected';
    case 'potential_manipulation':
      return 'Potential manipulation indicators detected';
    case 'insufficient_evidence':
    default:
      return 'Insufficient evidence';
  }
}

function formatComparisonAssessment(assessmentCode, fallbackLabel) {
  if (fallbackLabel) return fallbackLabel;
  switch (assessmentCode) {
    case 'no_significant_difference':
      return 'No significant difference';
    case 'visual_differences_detected':
      return 'Visual differences detected';
    case 'localized_visual_differences_detected':
      return 'Localized visual differences detected';
    case 'different_dimensions':
      return 'Different dimensions';
    case 'insufficient_data':
    default:
      return 'Insufficient data';
  }
}

function formatEvidenceAssessment(assessmentCode, fallbackLabel) {
  if (fallbackLabel) return fallbackLabel;
  switch (assessmentCode) {
    case 'multiple_forensic_signals':
      return 'Multiple forensic signals';
    case 'localized_visual_differences':
      return 'Localized visual differences';
    case 'no_significant_evidence':
      return 'No significant evidence';
    case 'insufficient_evidence':
    default:
      return 'Insufficient evidence';
  }
}

function formatEvidenceValue(val) {
  if (val === null || val === undefined) return 'N/A';
  if (typeof val === 'boolean') return val ? 'True' : 'False';
  if (typeof val === 'object') return JSON.stringify(val);
  return String(val);
}

export default function MediaUpload({ activeCase, onUploadSuccess, onAnalysisSuccess }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedRefFile, setSelectedRefFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isLocalizing, setIsLocalizing] = useState(false);
  const [isUploadingRef, setIsUploadingRef] = useState(false);
  const [isComparing, setIsComparing] = useState(false);
  const [isGeneratingEvidence, setIsGeneratingEvidence] = useState(false);
  const [isInterpretingForensics, setIsInterpretingForensics] = useState(false);
  const [isReconstructingHistory, setIsReconstructingHistory] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [isViewingReport, setIsViewingReport] = useState(false);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);
  const [isAnalyzingVideo, setIsAnalyzingVideo] = useState(false);
  const [isAnalyzingAudio, setIsAnalyzingAudio] = useState(false);
  const [isAnalyzingDocument, setIsAnalyzingDocument] = useState(false);
  const [forensicReport, setForensicReport] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [followUpWarning, setFollowUpWarning] = useState('');
  const [uploadedCase, setUploadedCase] = useState(null);
  const uploadInFlightRef = useRef(false);

  useEffect(() => {
    if (activeCase && activeCase.case_id && activeCase.case_id !== uploadedCase?.case_id) {
      setUploadedCase(activeCase);
      setErrorMessage('');
      setFollowUpWarning('');
    }
  }, [activeCase, uploadedCase?.case_id]);

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
    setErrorMessage('');
    setFollowUpWarning('');
  };

  const handleRefFileChange = (event) => {
    const file = event.target.files?.[0] || null;
    setSelectedRefFile(file);
    setErrorMessage('');
  };

  const handleUpload = async (event) => {
    event.preventDefault();
    if (isUploading || uploadInFlightRef.current) {
      return;
    }
    if (!selectedFile) {
      setErrorMessage('Please select a media file before uploading.');
      return;
    }

    uploadInFlightRef.current = true;
    setIsUploading(true);
    setErrorMessage('');
    setFollowUpWarning('');

    let caseData = null;
    try {
      caseData = await uploadMedia(selectedFile);
      setErrorMessage('');
      setUploadedCase(caseData);
      setForensicReport(null);
      setSelectedRefFile(null);
    } catch (error) {
      setErrorMessage(error.message || 'Failed to upload media.');
      uploadInFlightRef.current = false;
      setIsUploading(false);
      return;
    }

    if (onUploadSuccess) {
      try {
        await onUploadSuccess(caseData);
      } catch (refreshError) {
        setFollowUpWarning(
          `Case ${caseData.case_id} was uploaded successfully, but refreshing GET /api/media failed: ${
            refreshError.message || 'Could not refresh case registry.'
          }`
        );
      }
    }

    uploadInFlightRef.current = false;
    setIsUploading(false);
  };

  const handleAnalyzeImage = async () => {
    if (!uploadedCase?.case_id) return;

    setIsAnalyzing(true);
    setErrorMessage('');

    try {
      const analyzedCase = await analyzeImage(uploadedCase.case_id);
      setUploadedCase(analyzedCase);
      if (onAnalysisSuccess) {
        onAnalysisSuccess(analyzedCase);
      }
    } catch (error) {
      setErrorMessage(error.message || 'Failed to analyze image.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleRunManipulationDetection = async () => {
    if (!uploadedCase?.case_id) return;

    setIsDetecting(true);
    setErrorMessage('');

    try {
      const detectedCase = await analyzeImageManipulation(uploadedCase.case_id);
      setUploadedCase(detectedCase);
      if (onAnalysisSuccess) {
        onAnalysisSuccess(detectedCase);
      }
    } catch (error) {
      setErrorMessage(error.message || 'Failed to run manipulation detection.');
    } finally {
      setIsDetecting(false);
    }
  };

  const handleLocalizeRegions = async () => {
    if (!uploadedCase?.case_id) return;

    setIsLocalizing(true);
    setErrorMessage('');

    try {
      const localizedCase = await localizeImage(uploadedCase.case_id);
      setUploadedCase(localizedCase);
      if (onAnalysisSuccess) {
        onAnalysisSuccess(localizedCase);
      }
    } catch (error) {
      setErrorMessage(error.message || 'Failed to localize suspicious regions.');
    } finally {
      setIsLocalizing(false);
    }
  };

  const handleUploadReference = async () => {
    if (!uploadedCase?.case_id || !selectedRefFile) return;

    setIsUploadingRef(true);
    setErrorMessage('');

    try {
      const updatedCase = await uploadReferenceImage(uploadedCase.case_id, selectedRefFile);
      setUploadedCase(updatedCase);
      if (onAnalysisSuccess) {
        onAnalysisSuccess(updatedCase);
      }
    } catch (error) {
      setErrorMessage(error.message || 'Failed to upload reference image.');
    } finally {
      setIsUploadingRef(false);
    }
  };

  const handleCompareImages = async () => {
    if (!uploadedCase?.case_id || !uploadedCase?.reference_image) return;

    setIsComparing(true);
    setErrorMessage('');

    try {
      const comparedCase = await compareImages(uploadedCase.case_id);
      setUploadedCase(comparedCase);
      if (onAnalysisSuccess) {
        onAnalysisSuccess(comparedCase);
      }
    } catch (error) {
      setErrorMessage(error.message || 'Failed to compare images.');
    } finally {
      setIsComparing(false);
    }
  };

  const handleGenerateEvidence = async () => {
    if (!uploadedCase?.case_id) return;

    setIsGeneratingEvidence(true);
    setErrorMessage('');

    try {
      const updatedCase = await generateEvidenceReport(uploadedCase.case_id);
      setUploadedCase(updatedCase);
      if (onAnalysisSuccess) {
        onAnalysisSuccess(updatedCase);
      }
    } catch (error) {
      setErrorMessage(error.message || 'Failed to generate evidence report.');
    } finally {
      setIsGeneratingEvidence(false);
    }
  };

  const [isCheckingReproducibility, setIsCheckingReproducibility] = useState(false);

  const handleInterpretForensics = async () => {
    if (!uploadedCase?.case_id) return;

    setIsInterpretingForensics(true);
    setErrorMessage('');

    try {
      await interpretForensics(uploadedCase.case_id);
      const updatedCase = await getMediaCase(uploadedCase.case_id);
      setUploadedCase(updatedCase);
      if (onAnalysisSuccess) {
        onAnalysisSuccess(updatedCase);
      }
    } catch (error) {
      setErrorMessage(error.message || 'Failed to run multi-signal forensic interpretation.');
    } finally {
      setIsInterpretingForensics(false);
    }
  };

  const handleCheckReproducibility = async () => {
    if (!uploadedCase?.case_id) return;

    setIsCheckingReproducibility(true);
    setErrorMessage('');

    try {
      await checkCaseReproducibility(uploadedCase.case_id);
      const snapshot = await fetchCaseReproducibility(uploadedCase.case_id);
      const updatedCase = await getMediaCase(uploadedCase.case_id);
      updatedCase.latest_snapshot = snapshot;
      setUploadedCase(updatedCase);
      if (onAnalysisSuccess) {
        onAnalysisSuccess(updatedCase);
      }
    } catch (error) {
      setErrorMessage(error.message || 'Failed to verify case reproducibility.');
    } finally {
      setIsCheckingReproducibility(false);
    }
  };

  const handleReconstructHistory = async () => {
    if (!uploadedCase?.case_id) return;

    setIsReconstructingHistory(true);
    setErrorMessage('');

    try {
      const updatedCase = await reconstructEditHistory(uploadedCase.case_id);
      setUploadedCase(updatedCase);
      if (onAnalysisSuccess) {
        onAnalysisSuccess(updatedCase);
      }
    } catch (error) {
      setErrorMessage(error.message || 'Failed to reconstruct possible edit history.');
    } finally {
      setIsReconstructingHistory(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!uploadedCase?.case_id) return;

    setIsGeneratingReport(true);
    setErrorMessage('');

    try {
      const reportData = await generateForensicReport(uploadedCase.case_id);
      setForensicReport(reportData);
    } catch (error) {
      setErrorMessage(error.message || 'Failed to generate forensic report.');
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const handleViewReport = async () => {
    if (!uploadedCase?.case_id) return;

    setIsViewingReport(true);
    setErrorMessage('');

    try {
      const reportData = await getForensicReport(uploadedCase.case_id);
      setForensicReport(reportData);
    } catch (error) {
      setErrorMessage(error.message || 'Failed to retrieve forensic report.');
    } finally {
      setIsViewingReport(false);
    }
  };

  const handleDownloadPdfReport = async () => {
    if (!uploadedCase?.case_id) return;

    setIsDownloadingPdf(true);
    setErrorMessage('');

    try {
      const pdfBlob = await generateForensicReportPdf(uploadedCase.case_id);
      const reportData = await getForensicReport(uploadedCase.case_id);
      setForensicReport(reportData);

      const url = window.URL.createObjectURL(pdfBlob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `MediaTrace_Forensic_Report_${uploadedCase.case_id}.pdf`;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      setErrorMessage(error.message || 'Failed to generate or download PDF forensic report.');
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  const handleAnalyzeVideo = async () => {
    if (!uploadedCase?.case_id) return;

    setIsAnalyzingVideo(true);
    setErrorMessage('');

    try {
      const updatedCase = await analyzeVideo(uploadedCase.case_id);
      setUploadedCase(updatedCase);
      if (onAnalysisSuccess) {
        onAnalysisSuccess(updatedCase);
      }
    } catch (error) {
      setErrorMessage(error.message || 'Failed to run video forensic analysis.');
    } finally {
      setIsAnalyzingVideo(false);
    }
  };

  const handleAnalyzeAudio = async () => {
    if (!uploadedCase?.case_id) return;

    setIsAnalyzingAudio(true);
    setErrorMessage('');

    try {
      const updatedCase = await analyzeAudio(uploadedCase.case_id);
      setUploadedCase(updatedCase);
      if (onAnalysisSuccess) {
        onAnalysisSuccess(updatedCase);
      }
    } catch (error) {
      setErrorMessage(error.message || 'Failed to run audio forensic analysis.');
    } finally {
      setIsAnalyzingAudio(false);
    }
  };

  const handleAnalyzeDocument = async () => {
    if (!uploadedCase?.case_id) return;

    setIsAnalyzingDocument(true);
    setErrorMessage('');

    try {
      const updatedCase = await analyzeDocument(uploadedCase.case_id);
      setUploadedCase(updatedCase);
      if (onAnalysisSuccess) {
        onAnalysisSuccess(updatedCase);
      }
    } catch (error) {
      setErrorMessage(error.message || 'Failed to run document forensic analysis.');
    } finally {
      setIsAnalyzingDocument(false);
    }
  };

  const analysis = uploadedCase?.analysis;
  const detection = uploadedCase?.manipulation_detection;
  const localization = uploadedCase?.localization;
  const refImage = uploadedCase?.reference_image;
  const comparison = uploadedCase?.comparison;
  const evidence = uploadedCase?.evidence;
  const forensicDecision = uploadedCase?.forensic_decision;
  const history = uploadedCase?.history;
  const videoAnalysis = uploadedCase?.video_analysis;
  const audioAnalysis = uploadedCase?.audio_analysis;
  const documentAnalysis = uploadedCase?.document_analysis;
  const isVideoCase = uploadedCase?.media_type === 'video';
  const isAudioCase = uploadedCase?.media_type === 'audio';
  const isDocumentCase = uploadedCase?.media_type === 'document';
  const crossOverlapItems =
    evidence?.items?.filter((item) => item.type === 'cross_analysis_overlap') || [];

  const isBusy =
    isUploading ||
    isAnalyzing ||
    isDetecting ||
    isLocalizing ||
    isUploadingRef ||
    isComparing ||
    isGeneratingEvidence ||
    isInterpretingForensics ||
    isReconstructingHistory ||
    isGeneratingReport ||
    isViewingReport ||
    isDownloadingPdf ||
    isAnalyzingVideo ||
    isAnalyzingAudio ||
    isAnalyzingDocument;

  const cameraMakeModel = [analysis?.metadata?.fields?.make, analysis?.metadata?.fields?.model]
    .filter(Boolean)
    .join(' ');
  const originalDate =
    analysis?.metadata?.fields?.datetime_original || analysis?.metadata?.fields?.datetime || null;

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Forensic Media Intake, Multi-Modal Analysis &amp; Reporting</h2>
        <span className="badge">Step 12: Document Analysis</span>
      </div>
      <p className="panel-description">
        Upload an image (<code>.jpg</code>, <code>.jpeg</code>, <code>.png</code>, <code>.webp</code>), video (
        <code>.mp4</code>, <code>.mov</code>, <code>.avi</code>, <code>.mkv</code>, <code>.webm</code>, <code>.m4v</code>
        ), audio recording (
        <code>.mp3</code>, <code>.wav</code>, <code>.flac</code>, <code>.m4a</code>, <code>.aac</code>, <code>.ogg</code>,{' '}
        <code>.opus</code>, <code>.wma</code>), or document (
        <code>.pdf</code>, <code>.docx</code>, <code>.doc</code>, <code>.txt</code>, <code>.rtf</code>, <code>.odt</code>
        ) to compute its SHA-256 fingerprint and run conservative forensic inspection.
      </p>

      <form onSubmit={handleUpload} className="upload-form">
        <label className="upload-dropzone">
          <input
            type="file"
            accept=".jpg,.jpeg,.png,.webp,.mp4,.mov,.avi,.mkv,.webm,.m4v,.mp3,.wav,.flac,.m4a,.aac,.ogg,.opus,.wma,.pdf,.docx,.doc,.txt,.rtf,.odt,image/*,video/*,audio/*,application/pdf"
            onChange={handleFileChange}
            className="file-input-hidden"
            disabled={isBusy}
          />
          <div className="dropzone-content">
            <p className="dropzone-title">
              {selectedFile
                ? 'Media staged for forensic intake'
                : 'Select an image, video, audio, or document file from your workstation'}
            </p>
            <p className="dropzone-hint">
              Supported formats: JPG, JPEG, PNG, WEBP, MP4, MOV, AVI, MKV, WEBM, M4V, MP3, WAV, FLAC, M4A, AAC, OGG, OPUS, WMA, PDF, DOCX, DOC, TXT, RTF, ODT
            </p>
          </div>
        </label>

        {selectedFile && (
          <div className="selected-file-card">
            <div className="file-meta-item">
              <span className="meta-label">Selected Filename:</span>
              <span className="meta-value">{selectedFile.name}</span>
            </div>
            <div className="file-meta-item">
              <span className="meta-label">File Size:</span>
              <span className="meta-value">{formatBytes(selectedFile.size)}</span>
            </div>
          </div>
        )}

        <div className="upload-actions">
          <button
            type="submit"
            className="primary-button"
            disabled={!selectedFile || isBusy}
          >
            {isUploading ? 'Uploading & Hashing...' : 'Upload'}
          </button>
        </div>
      </form>

      {isUploading && (
        <div className="alert alert-loading" role="status">
          Uploading file bytes, calculating SHA-256 digest, and creating case record...
        </div>
      )}

      {isAnalyzing && (
        <div className="alert alert-loading" role="status">
          Extracting image structure, container headers, and EXIF metadata without altering original bytes...
        </div>
      )}

      {isDetecting && (
        <div className="alert alert-loading" role="status">
          Evaluating classical consistency, JPEG compression, and metadata signals...
        </div>
      )}

      {isLocalizing && (
        <div className="alert alert-loading" role="status">
          Computing block-level noise residuals, JPEG recompression differences, and generating heatmap overlay...
        </div>
      )}

      {isUploadingRef && (
        <div className="alert alert-loading" role="status">
          Uploading reference image and computing SHA-256 hash...
        </div>
      )}

      {isComparing && (
        <div className="alert alert-loading" role="status">
          Comparing case and reference images, calculating pixel differences, and generating difference masks...
        </div>
      )}

      {isGeneratingEvidence && (
        <div className="alert alert-loading" role="status">
          Correlating forensic observations and spatial overlaps across completed stages...
        </div>
      )}

      {errorMessage && (
        <div className="alert alert-error" role="alert">
          <strong>Error:</strong> {errorMessage}
        </div>
      )}

      {followUpWarning && (
        <div className="alert alert-loading" role="status">
          <strong>Follow-up Case Registry Notice:</strong> {followUpWarning}
        </div>
      )}

      {uploadedCase && !isUploading && (
        <div className="case-receipt">
          <div className="alert alert-success">
            {evidence
              ? 'Correlated Forensic Evidence Report generated and saved.'
              : uploadedCase.status === 'compared'
                ? 'Original vs Edited image comparison completed and saved.'
                : uploadedCase.status === 'localized'
                  ? 'Spatial manipulation localization completed and artifacts saved.'
                  : uploadedCase.status === 'detected'
                    ? 'Manipulation detection completed and persisted to forensic case JSON.'
                    : uploadedCase.status === 'analyzed'
                      ? 'Basic image forensic analysis completed and saved to case record.'
                      : 'Forensic case initialized and stored successfully.'}
          </div>

          <div className="receipt-header-row">
            <h3 className="receipt-title">Created Case Metadata ({uploadedCase.media_type})</h3>
            <div className="action-button-group">
              {isVideoCase ? (
                <button
                  type="button"
                  className="primary-button"
                  onClick={handleAnalyzeVideo}
                  disabled={isBusy}
                >
                  {isAnalyzingVideo
                    ? 'Analyzing Video...'
                    : videoAnalysis
                      ? 'Re-Analyze Video'
                      : 'Analyze Video'}
                </button>
              ) : isAudioCase ? (
                <button
                  type="button"
                  className="primary-button"
                  onClick={handleAnalyzeAudio}
                  disabled={isBusy}
                >
                  {isAnalyzingAudio
                    ? 'Analyzing Audio...'
                    : audioAnalysis
                      ? 'Re-Analyze Audio'
                      : 'Analyze Audio'}
                </button>
              ) : isDocumentCase ? (
                <button
                  type="button"
                  className="primary-button"
                  onClick={handleAnalyzeDocument}
                  disabled={isBusy}
                >
                  {isAnalyzingDocument
                    ? 'Analyzing Document...'
                    : documentAnalysis
                      ? 'Re-Analyze Document'
                      : 'Analyze Document'}
                </button>
              ) : (
                <>
                  <button
                    type="button"
                    className="secondary-action-button"
                    onClick={handleAnalyzeImage}
                    disabled={isBusy}
                  >
                    {isAnalyzing
                      ? 'Analyzing...'
                      : analysis
                        ? 'Re-Analyze Image'
                        : 'Analyze Image'}
                  </button>

                  {analysis && (
                    <button
                      type="button"
                      className="secondary-action-button"
                      onClick={handleRunManipulationDetection}
                      disabled={isBusy}
                    >
                      {isDetecting
                        ? 'Running Detection...'
                        : detection
                          ? 'Re-Run Manipulation Detection'
                          : 'Run Manipulation Detection'}
                    </button>
                  )}

                  {detection && (
                    <button
                      type="button"
                      className="primary-button"
                      onClick={handleLocalizeRegions}
                      disabled={isBusy}
                    >
                      {isLocalizing
                        ? 'Localizing...'
                        : localization
                          ? 'Re-Localize Suspicious Regions'
                          : 'Localize Suspicious Regions'}
                    </button>
                  )}
                </>
              )}
            </div>
          </div>

          <dl className="metadata-grid">
            <div className="metadata-row">
              <dt>Case ID</dt>
              <dd className="mono">{uploadedCase.case_id}</dd>
            </div>
            <div className="metadata-row">
              <dt>Original Filename</dt>
              <dd>{uploadedCase.original_filename}</dd>
            </div>
            <div className="metadata-row">
              <dt>Media Type</dt>
              <dd className="mono">{uploadedCase.media_type}</dd>
            </div>
            <div className="metadata-row">
              <dt>File Size</dt>
              <dd>{formatBytes(uploadedCase.file_size)}</dd>
            </div>
            <div className="metadata-row">
              <dt>SHA-256</dt>
              <dd className="mono hash-text">{uploadedCase.sha256}</dd>
            </div>
            <div className="metadata-row">
              <dt>Status</dt>
              <dd>
                <span
                  className={`status-tag ${
                    uploadedCase.status !== 'uploaded' ? 'status-analyzed' : ''
                  }`}
                >
                  {uploadedCase.status}
                </span>
              </dd>
            </div>
          </dl>

          {videoAnalysis && (
            <div className="forensic-report-block">
              <div className="forensic-section-banner">
                <h3>Video Forensic Analysis</h3>
                <span className="neutral-note">
                  Step 10: Container, Frame Sampling, Temporal &amp; Recompression Inspection
                </span>
              </div>

              <div className="forensic-subpanel">
                <h4>Video Overview</h4>
                <dl className="compact-kv-list">
                  <div>
                    <dt>Filename</dt>
                    <dd>{videoAnalysis.filename}</dd>
                  </div>
                  <div>
                    <dt>Format / Container</dt>
                    <dd className="mono">
                      {videoAnalysis.format} ({videoAnalysis.container || 'N/A'})
                    </dd>
                  </div>
                  <div>
                    <dt>Resolution</dt>
                    <dd className="mono">
                      {videoAnalysis.width && videoAnalysis.height
                        ? `${videoAnalysis.width} x ${videoAnalysis.height} px`
                        : 'Not available'}
                    </dd>
                  </div>
                  <div>
                    <dt>Duration</dt>
                    <dd className="mono">
                      {videoAnalysis.duration_seconds !== null &&
                      videoAnalysis.duration_seconds !== undefined
                        ? `${videoAnalysis.duration_seconds} s`
                        : 'Not available'}
                    </dd>
                  </div>
                  <div>
                    <dt>Frame Rate</dt>
                    <dd className="mono">
                      {videoAnalysis.frame_rate ? `${videoAnalysis.frame_rate} fps` : 'Not available'}
                    </dd>
                  </div>
                  <div>
                    <dt>Frame Count</dt>
                    <dd className="mono">
                      {videoAnalysis.frame_count ?? 'Not available'}
                    </dd>
                  </div>
                  <div>
                    <dt>Video Codec</dt>
                    <dd className="mono">{videoAnalysis.codec || 'Not available'}</dd>
                  </div>
                  <div>
                    <dt>Audio Presence</dt>
                    <dd className="mono">
                      {videoAnalysis.has_audio
                        ? `Present (${videoAnalysis.audio_codec || 'Audio'})`
                        : 'No audio stream'}
                    </dd>
                  </div>
                </dl>
              </div>

              <div className="forensic-subpanel">
                <h4>Metadata</h4>
                <dl className="compact-kv-list">
                  <div>
                    <dt>Codec Profile</dt>
                    <dd className="mono">{videoAnalysis.metadata?.codec_profile || 'N/A'}</dd>
                  </div>
                  <div>
                    <dt>Bitrate</dt>
                    <dd className="mono">
                      {videoAnalysis.metadata?.bitrate
                        ? `${videoAnalysis.metadata.bitrate.toLocaleString()} bps`
                        : 'N/A'}
                    </dd>
                  </div>
                  <div>
                    <dt>Pixel Format / Color Space</dt>
                    <dd className="mono">
                      {videoAnalysis.metadata?.pixel_format || 'N/A'} /{' '}
                      {videoAnalysis.metadata?.color_space || 'N/A'}
                    </dd>
                  </div>
                  <div>
                    <dt>Encoder / Software</dt>
                    <dd className="mono">
                      {videoAnalysis.metadata?.encoder_software || 'Not recorded'}
                    </dd>
                  </div>
                  <div>
                    <dt>Creation Timestamp</dt>
                    <dd className="mono">
                      {videoAnalysis.metadata?.creation_timestamp || 'Not recorded'}
                    </dd>
                  </div>
                  <div>
                    <dt>GPS Metadata</dt>
                    <dd className="mono">
                      {videoAnalysis.metadata?.gps_present
                        ? 'GPS metadata present (coordinates redacted)'
                        : 'Not present'}
                    </dd>
                  </div>
                </dl>
              </div>

              <div className="forensic-subpanel">
                <h4>
                  Sampled Frames ({videoAnalysis.frame_sampling?.sample_count || 0} / max{' '}
                  {videoAnalysis.frame_sampling?.max_sample_frames || 12})
                </h4>
                <div className="heatmap-grid">
                  {videoAnalysis.frame_sampling?.frames?.map((frame) => (
                    <div key={frame.frame_number} className="heatmap-card">
                      <span className="heatmap-caption">
                        Frame #{frame.frame_index} (t = {frame.timestamp_seconds}s) —{' '}
                        {frame.artifact_filename}
                      </span>
                      <img
                        src={getVideoArtifactUrl(
                          uploadedCase.case_id,
                          frame.artifact_filename,
                          videoAnalysis.analyzed_at
                        )}
                        alt={`Sampled frame ${frame.frame_index}`}
                        className="localization-image"
                      />
                      <p className="finding-value mono" style={{ marginTop: '0.35rem' }}>
                        Brightness: {frame.mean_brightness} | Noise: {frame.noise_level} | Blockiness:{' '}
                        {frame.compression_blockiness}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="forensic-subpanel">
                <h4>Temporal Analysis</h4>
                <p className="finding-title">
                  {videoAnalysis.temporal_analysis?.status === 'potential_signal'
                    ? 'Potential temporal inconsistency detected'
                    : 'No strong temporal inconsistency signal'}
                </p>
                <p className="finding-value" style={{ marginBottom: '0.5rem' }}>
                  {videoAnalysis.temporal_analysis?.description}
                </p>
                {videoAnalysis.temporal_analysis?.signals?.length > 0 && (
                  <div className="findings-list">
                    {videoAnalysis.temporal_analysis.signals.map((sig, idx) => (
                      <div key={idx} className="finding-card">
                        <div className="finding-card-header">
                          <span className="finding-category-tag">
                            Frames {sig.from_frame_index} &rarr; {sig.to_frame_index} (t ={' '}
                            {sig.timestamp_seconds}s)
                          </span>
                          <span className={`signal-strength-tag strength-${sig.strength}`}>
                            {sig.type} ({sig.strength})
                          </span>
                        </div>
                        <p className="finding-value">{sig.description}</p>
                      </div>
                    ))}
                  </div>
                )}
                {videoAnalysis.temporal_analysis?.suspicious_frames?.length > 0 && (
                  <div style={{ marginTop: '0.75rem' }}>
                    <h5>Localized Frame Anomalies (Potentially Suspicious)</h5>
                    <div className="findings-list">
                      {videoAnalysis.temporal_analysis.suspicious_frames.map((sf, idx) => (
                        <div key={idx} className="finding-card">
                          <div className="finding-card-header">
                            <span className="finding-category-tag">
                              Frame {sf.frame_index} @ {sf.timestamp_seconds}s
                            </span>
                            <span className="signal-strength-tag strength-weak">
                              {sf.status} ({sf.finding})
                            </span>
                          </div>
                          <p className="finding-value">{sf.description}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="forensic-subpanel">
                <h4>Recompression</h4>
                <p className="finding-title">
                  Status: <span className="mono">{videoAnalysis.recompression_analysis?.status}</span>
                </p>
                <p className="finding-value" style={{ marginBottom: '0.4rem' }}>
                  {videoAnalysis.recompression_analysis?.description}
                </p>
                {videoAnalysis.recompression_analysis?.indicators?.length > 0 && (
                  <ul className="forensic-notes-list">
                    {videoAnalysis.recompression_analysis.indicators.map((ind, idx) => (
                      <li key={idx}>{ind}</li>
                    ))}
                  </ul>
                )}
                <p className="finding-value" style={{ marginTop: '0.4rem' }}>
                  <strong>Note:</strong> {videoAnalysis.recompression_analysis?.limitation}
                </p>
              </div>

              <div className="forensic-subpanel">
                <h4>Scene Analysis</h4>
                <p className="finding-value">
                  Status: <span className="mono">{videoAnalysis.scene_analysis?.status}</span>
                </p>
                {videoAnalysis.scene_analysis?.scene_changes?.length > 0 ? (
                  <div className="findings-list" style={{ marginTop: '0.4rem' }}>
                    {videoAnalysis.scene_analysis.scene_changes.map((sc, idx) => (
                      <div key={idx} className="finding-card">
                        <div className="finding-card-header">
                          <span className="finding-category-tag">
                            {sc.type} @ {sc.timestamp_seconds}s (Frame {sc.frame_index})
                          </span>
                          <span className="signal-strength-tag strength-informational">
                            Delta: {sc.difference_score}
                          </span>
                        </div>
                        <p className="finding-value">{sc.description}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="empty-state">No abrupt scene/cut visual transitions detected.</p>
                )}
              </div>

              <div className="forensic-subpanel">
                <h4>AI Detection</h4>
                <p className="finding-title">
                  AI-generated video detection: Not available — no trained model configured.
                </p>
                <p className="finding-value">{videoAnalysis.ai_generation?.reason}</p>
              </div>

              <div className="forensic-subpanel">
                <h4>Limitations</h4>
                <ul className="forensic-notes-list">
                  {videoAnalysis.limitations?.map((lim, idx) => (
                    <li key={idx}>{lim}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {isAudioCase && audioAnalysis && (
            <div className="forensic-report-block">
              <div className="forensic-section-banner">
                <h3>Audio Forensic Analysis</h3>
                <span className="neutral-note">Step 11: Conservative Audio Inspection</span>
              </div>

              <div className="forensic-subpanel">
                <h4>Assessment</h4>
                <div className={`assessment-banner assessment-${audioAnalysis.assessment?.status}`}>
                  <span className="assessment-title">{audioAnalysis.assessment?.label}</span>
                  <span className="assessment-code mono">({audioAnalysis.assessment?.status})</span>
                </div>
                <p className="finding-value" style={{ marginTop: '0.4rem' }}>
                  {audioAnalysis.assessment?.description}
                </p>
              </div>

              <div className="forensic-subpanel">
                <h4>Audio Overview</h4>
                <dl className="compact-kv-list">
                  <div>
                    <dt>Filename</dt>
                    <dd className="mono">{audioAnalysis.filename}</dd>
                  </div>
                  <div>
                    <dt>Format / Container</dt>
                    <dd className="mono">
                      {audioAnalysis.format} ({audioAnalysis.container || '-'})
                    </dd>
                  </div>
                  <div>
                    <dt>Duration</dt>
                    <dd className="mono">
                      {audioAnalysis.duration_seconds !== null && audioAnalysis.duration_seconds !== undefined
                        ? `${audioAnalysis.duration_seconds}s`
                        : '-'}
                    </dd>
                  </div>
                  <div>
                    <dt>Sample Rate</dt>
                    <dd className="mono">
                      {audioAnalysis.sample_rate ? `${audioAnalysis.sample_rate} Hz` : '-'}
                    </dd>
                  </div>
                  <div>
                    <dt>Channels</dt>
                    <dd className="mono">
                      {audioAnalysis.channels ?? '-'} ({audioAnalysis.metadata?.channel_layout || '-'})
                    </dd>
                  </div>
                  <div>
                    <dt>Bit Depth</dt>
                    <dd className="mono">
                      {audioAnalysis.bit_depth ? `${audioAnalysis.bit_depth}-bit` : 'Lossy / N/A'}
                    </dd>
                  </div>
                  <div>
                    <dt>Codec</dt>
                    <dd className="mono">{audioAnalysis.codec || '-'}</dd>
                  </div>
                  <div>
                    <dt>Bitrate</dt>
                    <dd className="mono">
                      {audioAnalysis.bitrate ? `${Math.round(audioAnalysis.bitrate / 1000)} kbps` : '-'}
                    </dd>
                  </div>
                </dl>
              </div>

              <div className="forensic-subpanel">
                <h4>Metadata</h4>
                <dl className="compact-kv-list">
                  <div>
                    <dt>Encoder / Software</dt>
                    <dd className="mono">
                      {audioAnalysis.metadata?.encoder || audioAnalysis.metadata?.software || 'Not recorded'}
                    </dd>
                  </div>
                  <div>
                    <dt>Creation Time</dt>
                    <dd className="mono">{audioAnalysis.metadata?.creation_time || 'Not recorded'}</dd>
                  </div>
                  <div>
                    <dt>Title / Artist</dt>
                    <dd className="mono">
                      {[audioAnalysis.metadata?.title, audioAnalysis.metadata?.artist]
                        .filter(Boolean)
                        .join(' — ') || 'Not recorded'}
                    </dd>
                  </div>
                  <div>
                    <dt>GPS Metadata</dt>
                    <dd className="mono">
                      {audioAnalysis.metadata?.gps_present
                        ? audioAnalysis.metadata.gps_notice || 'GPS metadata present'
                        : 'None detected'}
                    </dd>
                  </div>
                </dl>
              </div>

              <div className="forensic-subpanel">
                <h4>Waveform</h4>
                <dl className="compact-kv-list" style={{ marginBottom: '0.6rem' }}>
                  <div>
                    <dt>Peak Amplitude</dt>
                    <dd className="mono">{audioAnalysis.waveform_analysis?.peak_amplitude ?? '-'}</dd>
                  </div>
                  <div>
                    <dt>RMS Level</dt>
                    <dd className="mono">{audioAnalysis.waveform_analysis?.rms_level ?? '-'}</dd>
                  </div>
                  <div>
                    <dt>Dynamic Range</dt>
                    <dd className="mono">
                      {audioAnalysis.waveform_analysis?.dynamic_range_db !== null &&
                      audioAnalysis.waveform_analysis?.dynamic_range_db !== undefined
                        ? `${audioAnalysis.waveform_analysis.dynamic_range_db} dB`
                        : '-'}
                    </dd>
                  </div>
                  <div>
                    <dt>Min / Max / Mean Level</dt>
                    <dd className="mono">
                      {audioAnalysis.waveform_analysis?.minimum_level} /{' '}
                      {audioAnalysis.waveform_analysis?.maximum_level} /{' '}
                      {audioAnalysis.waveform_analysis?.mean_level}
                    </dd>
                  </div>
                  <div>
                    <dt>Clipping Samples</dt>
                    <dd className="mono">
                      {audioAnalysis.waveform_analysis?.clipping_sample_count ?? 0}
                    </dd>
                  </div>
                </dl>
                {audioAnalysis.waveform_analysis?.artifact_filename && (
                  <div className="heatmap-card">
                    <span className="heatmap-caption">
                      Derived Waveform Overview ({audioAnalysis.waveform_analysis.artifact_filename})
                    </span>
                    <img
                      src={getAudioArtifactUrl(
                        uploadedCase.case_id,
                        audioAnalysis.waveform_analysis.artifact_filename,
                        audioAnalysis.analyzed_at
                      )}
                      alt="Audio Waveform"
                      className="localization-image"
                    />
                  </div>
                )}
              </div>

              <div className="forensic-subpanel">
                <h4>Spectrogram &amp; Spectral Analysis</h4>
                <p className="finding-title">
                  Status: <span className="mono">{audioAnalysis.spectral_analysis?.status}</span>
                  {audioAnalysis.spectral_analysis?.finding
                    ? ` (${audioAnalysis.spectral_analysis.finding})`
                    : ''}
                </p>
                <p className="finding-value" style={{ marginBottom: '0.5rem' }}>
                  {audioAnalysis.spectral_analysis?.description}
                </p>
                <dl className="compact-kv-list" style={{ marginBottom: '0.6rem' }}>
                  <div>
                    <dt>Mean Spectral Centroid</dt>
                    <dd className="mono">
                      {audioAnalysis.spectral_analysis?.spectral_centroid_mean_hz} Hz
                    </dd>
                  </div>
                  <div>
                    <dt>Mean Spectral Rolloff (85%)</dt>
                    <dd className="mono">
                      {audioAnalysis.spectral_analysis?.spectral_rolloff_mean_hz} Hz
                    </dd>
                  </div>
                  <div>
                    <dt>Mean Spectral Flux</dt>
                    <dd className="mono">{audioAnalysis.spectral_analysis?.spectral_flux_mean}</dd>
                  </div>
                  <div>
                    <dt>High-Frequency Energy Ratio</dt>
                    <dd className="mono">
                      {audioAnalysis.spectral_analysis?.high_frequency_energy_ratio}
                    </dd>
                  </div>
                </dl>
                {audioAnalysis.spectral_analysis?.artifact_filename && (
                  <div className="heatmap-card">
                    <span className="heatmap-caption">
                      Derived STFT Spectrogram ({audioAnalysis.spectral_analysis.artifact_filename})
                    </span>
                    <img
                      src={getAudioArtifactUrl(
                        uploadedCase.case_id,
                        audioAnalysis.spectral_analysis.artifact_filename,
                        audioAnalysis.analyzed_at
                      )}
                      alt="Audio Spectrogram"
                      className="localization-image"
                    />
                  </div>
                )}
              </div>

              <div className="forensic-subpanel">
                <h4>
                  Silence / Low-Amplitude Regions ({audioAnalysis.silence_analysis?.region_count ?? 0})
                </h4>
                {audioAnalysis.silence_analysis?.regions?.length > 0 ? (
                  <div className="findings-list">
                    {audioAnalysis.silence_analysis.regions.map((reg, idx) => (
                      <div key={idx} className="finding-card">
                        <div className="finding-card-header">
                          <span className="finding-category-tag">Low-amplitude region</span>
                          <span className="signal-strength-tag strength-informational">
                            {reg.start_seconds}s &ndash; {reg.end_seconds}s (Duration:{' '}
                            {reg.duration_seconds}s)
                          </span>
                        </div>
                        <p className="finding-value">{reg.description}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="empty-state">
                    No extended low-amplitude / silence regions detected above minimum duration threshold.
                  </p>
                )}
              </div>

              <div className="forensic-subpanel">
                <h4>
                  Possible Discontinuities ({audioAnalysis.discontinuity_analysis?.candidate_count ?? 0})
                </h4>
                <p className="finding-value" style={{ marginBottom: '0.4rem' }}>
                  {audioAnalysis.discontinuity_analysis?.description}
                </p>
                {audioAnalysis.discontinuity_analysis?.candidates?.length > 0 ? (
                  <div className="findings-list">
                    {audioAnalysis.discontinuity_analysis.candidates.map((cand, idx) => (
                      <div key={idx} className="finding-card">
                        <div className="finding-card-header">
                          <span className="finding-category-tag">
                            Potential audio discontinuity @ {cand.start_seconds}s &ndash;{' '}
                            {cand.end_seconds}s (Duration: {cand.duration_seconds}s)
                          </span>
                          <span className="signal-strength-tag strength-weak">
                            {cand.status} ({cand.finding})
                          </span>
                        </div>
                        <p className="finding-value">{cand.description}</p>
                        {cand.artifact_filename && (
                          <div style={{ marginTop: '0.4rem' }}>
                            <img
                              src={getAudioArtifactUrl(
                                uploadedCase.case_id,
                                cand.artifact_filename,
                                audioAnalysis.analyzed_at
                              )}
                              alt={`Discontinuity at ${cand.start_seconds}s`}
                              style={{ maxWidth: '320px', borderRadius: '6px' }}
                            />
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="empty-state">
                    No abrupt waveform or spectral discontinuity candidates detected.
                  </p>
                )}
              </div>

              <div className="forensic-subpanel">
                <h4>Recompression</h4>
                <p className="finding-title">
                  Status:{' '}
                  <span className="mono">{audioAnalysis.recompression_analysis?.status}</span>
                </p>
                <p className="finding-value" style={{ marginBottom: '0.4rem' }}>
                  {audioAnalysis.recompression_analysis?.description}
                </p>
                {audioAnalysis.recompression_analysis?.indicators?.length > 0 && (
                  <ul className="forensic-notes-list">
                    {audioAnalysis.recompression_analysis.indicators.map((ind, idx) => (
                      <li key={idx}>{ind}</li>
                    ))}
                  </ul>
                )}
                <p className="finding-value" style={{ marginTop: '0.4rem' }}>
                  <strong>Note:</strong> {audioAnalysis.recompression_analysis?.limitation}
                </p>
              </div>

              <div className="forensic-subpanel">
                <h4>AI Detection</h4>
                <p className="finding-title">
                  AI-generated audio detection: Not available — no trained model configured.
                </p>
                <p className="finding-value">{audioAnalysis.ai_generation?.reason}</p>
              </div>

              <div className="forensic-subpanel">
                <h4>Limitations</h4>
                <ul className="forensic-notes-list">
                  {audioAnalysis.limitations?.map((lim, idx) => (
                    <li key={idx}>{lim}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {isDocumentCase && documentAnalysis && (
            <div className="forensic-report-block">
              <div className="forensic-section-banner">
                <h3>Document Forensic Analysis</h3>
                <span className="neutral-note">Step 12: Conservative Document Inspection</span>
              </div>

              <div className="forensic-subpanel">
                <h4>Assessment</h4>
                <div className={`assessment-banner assessment-${documentAnalysis.assessment?.status}`}>
                  <span className="assessment-title">{documentAnalysis.assessment?.label}</span>
                  <span className="assessment-code mono">({documentAnalysis.assessment?.status})</span>
                </div>
                <p className="finding-value" style={{ marginTop: '0.4rem' }}>
                  {documentAnalysis.assessment?.description}
                </p>
              </div>

              <div className="forensic-subpanel">
                <h4>Document Overview</h4>
                <dl className="compact-kv-list">
                  <div>
                    <dt>Filename</dt>
                    <dd className="mono">{documentAnalysis.filename}</dd>
                  </div>
                  <div>
                    <dt>Format</dt>
                    <dd className="mono">{documentAnalysis.format}</dd>
                  </div>
                  <div>
                    <dt>MIME Type</dt>
                    <dd className="mono">{documentAnalysis.mime_type}</dd>
                  </div>
                  <div>
                    <dt>File Size</dt>
                    <dd className="mono">{formatBytes(documentAnalysis.file_size)}</dd>
                  </div>
                  <div>
                    <dt>Page Count</dt>
                    <dd className="mono">{documentAnalysis.structure?.page_count ?? 'N/A'}</dd>
                  </div>
                  <div>
                    <dt>Parser Status</dt>
                    <dd className="mono">
                      {documentAnalysis.parser_status?.status}
                      {documentAnalysis.parser_status?.reason
                        ? ` (${documentAnalysis.parser_status.reason})`
                        : ''}
                    </dd>
                  </div>
                </dl>
              </div>

              <div className="forensic-subpanel">
                <h4>Metadata</h4>
                <dl className="compact-kv-list">
                  <div>
                    <dt>Title</dt>
                    <dd className="mono">{documentAnalysis.metadata?.title || 'Not recorded'}</dd>
                  </div>
                  <div>
                    <dt>Author / Creator</dt>
                    <dd className="mono">
                      {documentAnalysis.metadata?.author ||
                        documentAnalysis.metadata?.creator ||
                        'Not recorded'}
                    </dd>
                  </div>
                  <div>
                    <dt>Producer / Application</dt>
                    <dd className="mono">
                      {documentAnalysis.metadata?.producer ||
                        documentAnalysis.metadata?.application ||
                        documentAnalysis.metadata?.generator ||
                        'Not recorded'}
                    </dd>
                  </div>
                  <div>
                    <dt>Last Modified By</dt>
                    <dd className="mono">
                      {documentAnalysis.metadata?.last_modified_by || 'Not recorded'}
                    </dd>
                  </div>
                  <div>
                    <dt>Creation Timestamp</dt>
                    <dd className="mono">
                      {documentAnalysis.metadata?.creation_timestamp || 'Not recorded'}
                    </dd>
                  </div>
                  <div>
                    <dt>Modification Timestamp</dt>
                    <dd className="mono">
                      {documentAnalysis.metadata?.modification_timestamp || 'Not recorded'}
                    </dd>
                  </div>
                  <div>
                    <dt>GPS Metadata</dt>
                    <dd className="mono">
                      {documentAnalysis.metadata?.gps_present
                        ? documentAnalysis.metadata.gps_notice || 'GPS metadata present'
                        : 'None detected'}
                    </dd>
                  </div>
                </dl>
              </div>

              <div className="forensic-subpanel">
                <h4>Structure</h4>
                <dl className="compact-kv-list">
                  <div>
                    <dt>Format / Package Version</dt>
                    <dd className="mono">
                      {documentAnalysis.structure?.format_version || 'N/A'}
                    </dd>
                  </div>
                  <div>
                    <dt>Object / Relationship Count</dt>
                    <dd className="mono">
                      {documentAnalysis.structure?.object_count ??
                        documentAnalysis.structure?.relationship_count ??
                        'N/A'}
                    </dd>
                  </div>
                  <div>
                    <dt>Font Count</dt>
                    <dd className="mono">{documentAnalysis.structure?.font_count ?? 'N/A'}</dd>
                  </div>
                  <div>
                    <dt>Paragraphs / Tables</dt>
                    <dd className="mono">
                      {documentAnalysis.structure?.paragraph_count ?? 'N/A'} /{' '}
                      {documentAnalysis.structure?.table_count ?? 'N/A'}
                    </dd>
                  </div>
                  <div>
                    <dt>JavaScript / Macros Present</dt>
                    <dd className="mono">
                      JS: {documentAnalysis.structure?.javascript_present ? 'Yes' : 'No'} | Macros:{' '}
                      {documentAnalysis.structure?.macro_present ? 'Yes' : 'No'}
                    </dd>
                  </div>
                </dl>
              </div>

              <div className="forensic-subpanel">
                <h4>Text Statistics</h4>
                <dl className="compact-kv-list">
                  <div>
                    <dt>Character Count</dt>
                    <dd className="mono">
                      {documentAnalysis.text_analysis?.character_count ?? 'N/A'}
                    </dd>
                  </div>
                  <div>
                    <dt>Word Count</dt>
                    <dd className="mono">
                      {documentAnalysis.text_analysis?.word_count ?? 'N/A'}
                    </dd>
                  </div>
                  <div>
                    <dt>Paragraph Count</dt>
                    <dd className="mono">
                      {documentAnalysis.text_analysis?.paragraph_count ?? 'N/A'}
                    </dd>
                  </div>
                  <div>
                    <dt>Line Count / Encoding</dt>
                    <dd className="mono">
                      {documentAnalysis.text_analysis?.line_count ?? 'N/A'} (
                      {documentAnalysis.text_analysis?.encoding || 'N/A'})
                    </dd>
                  </div>
                </dl>
              </div>

              <div className="forensic-subpanel">
                <h4>Embedded Content</h4>
                <dl className="compact-kv-list">
                  <div>
                    <dt>Embedded Images</dt>
                    <dd className="mono">
                      {documentAnalysis.embedded_content?.images?.count ?? 0}
                    </dd>
                  </div>
                  <div>
                    <dt>Embedded Files / Attachments</dt>
                    <dd className="mono">
                      {documentAnalysis.embedded_content?.embedded_files?.count ?? 0}
                    </dd>
                  </div>
                  <div>
                    <dt>Hyperlinks</dt>
                    <dd className="mono">
                      {documentAnalysis.embedded_content?.hyperlinks?.count ?? 0}
                    </dd>
                  </div>
                  <div>
                    <dt>Annotations / Comments</dt>
                    <dd className="mono">
                      {documentAnalysis.embedded_content?.annotations?.count ?? 0}
                    </dd>
                  </div>
                  <div>
                    <dt>Embedded Objects</dt>
                    <dd className="mono">
                      {documentAnalysis.embedded_content?.objects?.count ?? 0}
                    </dd>
                  </div>
                </dl>
              </div>

              <div className="forensic-subpanel">
                <h4>Revision Analysis</h4>
                <p className="finding-title">
                  Status:{' '}
                  <span className="mono">{documentAnalysis.revision_analysis?.status}</span>
                </p>
                <p className="finding-value" style={{ marginBottom: '0.4rem' }}>
                  {documentAnalysis.revision_analysis?.description}
                </p>
                {documentAnalysis.revision_analysis?.indicators?.length > 0 && (
                  <ul className="forensic-notes-list">
                    {documentAnalysis.revision_analysis.indicators.map((ind, idx) => (
                      <li key={idx}>{ind}</li>
                    ))}
                  </ul>
                )}
              </div>

              <div className="forensic-subpanel">
                <h4>Timestamp Analysis</h4>
                <dl className="compact-kv-list" style={{ marginBottom: '0.4rem' }}>
                  <div>
                    <dt>Document Created</dt>
                    <dd className="mono">
                      {documentAnalysis.timestamp_analysis?.document_created_at || 'Not recorded'}
                    </dd>
                  </div>
                  <div>
                    <dt>Document Modified</dt>
                    <dd className="mono">
                      {documentAnalysis.timestamp_analysis?.document_modified_at || 'Not recorded'}
                    </dd>
                  </div>
                  <div>
                    <dt>Filesystem Modified</dt>
                    <dd className="mono">
                      {documentAnalysis.timestamp_analysis?.filesystem_modified_at || '-'}
                    </dd>
                  </div>
                </dl>
                {documentAnalysis.timestamp_analysis?.signals?.map((sig, idx) => (
                  <p key={idx} className="finding-value">
                    <strong>[{sig.status}]:</strong> {sig.description}
                  </p>
                ))}
                <p className="finding-value" style={{ marginTop: '0.35rem' }}>
                  {documentAnalysis.timestamp_analysis?.note}
                </p>
              </div>

              <div className="forensic-subpanel">
                <h4>Consistency Analysis</h4>
                {documentAnalysis.consistency_analysis?.signals?.length > 0 ? (
                  <div className="findings-list">
                    {documentAnalysis.consistency_analysis.signals.map((sig, idx) => (
                      <div key={idx} className="finding-card">
                        <div className="finding-card-header">
                          <span className="finding-category-tag">
                            Potential {sig.category} inconsistency ({sig.finding})
                          </span>
                          <span className="signal-strength-tag strength-weak">
                            {sig.status}
                          </span>
                        </div>
                        <p className="finding-value">{sig.description}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="empty-state">
                    No internal metadata or structural inconsistencies detected.
                  </p>
                )}
              </div>

              {documentAnalysis.artifacts?.length > 0 && (
                <div className="forensic-subpanel">
                  <h4>Artifacts ({documentAnalysis.artifacts.length} Page Preview(s))</h4>
                  <div className="heatmap-preview-grid">
                    {documentAnalysis.artifacts.map((art) => (
                      <div key={art.artifact_filename} className="heatmap-card">
                        <span className="heatmap-caption">
                          Page {art.page_number} ({art.artifact_filename})
                        </span>
                        <img
                          src={getDocumentArtifactUrl(
                            uploadedCase.case_id,
                            art.artifact_filename,
                            documentAnalysis.analyzed_at
                          )}
                          alt={`Document Page ${art.page_number}`}
                          className="localization-image"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="forensic-subpanel">
                <h4>AI Detection</h4>
                <p className="finding-title">
                  AI-generated document detection: Not available — no trained model configured.
                </p>
                <p className="finding-value">{documentAnalysis.ai_generation?.reason}</p>
              </div>

              <div className="forensic-subpanel">
                <h4>Limitations</h4>
                <ul className="forensic-notes-list">
                  {documentAnalysis.limitations?.map((lim, idx) => (
                    <li key={idx}>{lim}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {analysis && (
            <div className="forensic-report-block">
              <div className="forensic-section-banner">
                <h3>Basic Forensic Findings</h3>
                <span className="neutral-note">Step 3: Objective Container &amp; Metadata Inspection</span>
              </div>

              <div className="forensic-subpanel">
                <h4>Image Information</h4>
                <dl className="compact-kv-list">
                  <div>
                    <dt>Format</dt>
                    <dd>{analysis.image.format || 'Unknown'}</dd>
                  </div>
                  <div>
                    <dt>Width</dt>
                    <dd>{analysis.image.width} px</dd>
                  </div>
                  <div>
                    <dt>Height</dt>
                    <dd>{analysis.image.height} px</dd>
                  </div>
                  <div>
                    <dt>Color Mode</dt>
                    <dd>
                      {analysis.image.mode}
                      {analysis.image.channels ? ` (${analysis.image.channels} channels)` : ''}
                    </dd>
                  </div>
                  <div>
                    <dt>File Size</dt>
                    <dd>{formatBytes(uploadedCase.file_size)}</dd>
                  </div>
                </dl>
              </div>

              <div className="forensic-subpanel">
                <h4>Metadata</h4>
                <dl className="compact-kv-list">
                  <div>
                    <dt>EXIF Status</dt>
                    <dd>{analysis.metadata.exif_present ? 'EXIF available' : 'EXIF not available'}</dd>
                  </div>
                  <div>
                    <dt>Camera Make / Model</dt>
                    <dd>{cameraMakeModel || 'Not available'}</dd>
                  </div>
                  <div>
                    <dt>Software</dt>
                    <dd>{analysis.metadata.fields?.software || 'Not available'}</dd>
                  </div>
                  <div>
                    <dt>Original Date</dt>
                    <dd>{originalDate || 'Not available'}</dd>
                  </div>
                  <div>
                    <dt>GPS Metadata</dt>
                    <dd>
                      {analysis.metadata.gps_present
                        ? 'GPS metadata present'
                        : 'GPS metadata not present'}
                    </dd>
                  </div>
                </dl>
              </div>

              <div className="forensic-subpanel">
                <h4>Findings ({analysis.findings?.length || 0})</h4>
                <div className="findings-list">
                  {analysis.findings?.map((item, idx) => (
                    <div key={`${item.category}-${idx}`} className="finding-card">
                      <div className="finding-card-header">
                        <span className="finding-category-tag">{item.category}</span>
                        <span className="finding-severity-tag">{item.severity}</span>
                      </div>
                      <p className="finding-title">{item.finding}</p>
                      <p className="finding-value mono">{item.value}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {detection && (
            <div className="forensic-report-block">
              <div className="forensic-section-banner">
                <h3>Manipulation Detection</h3>
                <span className="neutral-note">Step 4: Heuristic Indicators (Not Proof of Tampering)</span>
              </div>

              <div className="forensic-subpanel">
                <h4>Overall Assessment</h4>
                <div className={`assessment-banner assessment-${detection.overall_assessment}`}>
                  <span className="assessment-title">
                    {formatAssessmentText(detection.overall_assessment, detection.assessment_label)}
                  </span>
                  <span className="assessment-code mono">({detection.overall_assessment})</span>
                </div>
              </div>

              <div className="forensic-subpanel">
                <h4>Forensic Signals ({detection.signals?.length || 0})</h4>
                <div className="findings-list">
                  {detection.signals?.map((sig, idx) => (
                    <div key={`${sig.name}-${idx}`} className="finding-card">
                      <div className="finding-card-header">
                        <span className="finding-category-tag">{sig.name}</span>
                        <span className={`signal-strength-tag strength-${sig.strength}`}>
                          Strength: {sig.strength}
                        </span>
                      </div>
                      <p className="finding-title">{sig.description}</p>
                      <p className="finding-value mono">Value: {sig.value}</p>
                    </div>
                  ))}
                </div>
              </div>

              {detection.assessment && (
                <div className="forensic-subpanel">
                  <h4>Separated Forensic Interpretation</h4>
                  <dl className="compact-kv-list">
                    <div>
                      <dt>Classical Forensic Observations</dt>
                      <dd>{detection.assessment.classical_forensic_observations}</dd>
                    </div>
                    <div>
                      <dt>AI Detector Output</dt>
                      <dd>{detection.assessment.ai_detector_output}</dd>
                    </div>
                    {detection.assessment.cross_analysis_interpretation && (
                      <div>
                        <dt>Cross-Analysis Interpretation</dt>
                        <dd>{detection.assessment.cross_analysis_interpretation}</dd>
                      </div>
                    )}
                    <div>
                      <dt>Combined Interpretation</dt>
                      <dd>{detection.assessment.combined_interpretation}</dd>
                    </div>
                  </dl>
                </div>
              )}
            </div>
          )}

          {detection && (() => {
            const aiDet = detection.ai_generation_detection || detection.ai_generation;
            const isAvailable = aiDet?.status === 'available' && aiDet?.prediction && aiDet?.detector;
            const pinnedRev = aiDet?.detector?.pinned_revision || aiDet?.detector?.version;
            const interp = aiDet?.ai_detector_interpretation;
            const modelOut = interp?.model_output || {};
            const txCtx = aiDet?.transformation_context || interp?.transformation_context;
            const selectedClass =
              modelOut.selected_class || aiDet?.selected_class || aiDet?.prediction?.raw_class;
            const artificialScore =
              typeof modelOut.artificial_score === 'number'
                ? modelOut.artificial_score
                : typeof aiDet?.artificial_score === 'number'
                  ? aiDet.artificial_score
                  : aiDet?.prediction?.score;
            const humanScore =
              typeof modelOut.human_score === 'number'
                ? modelOut.human_score
                : typeof aiDet?.human_score === 'number'
                  ? aiDet.human_score
                  : typeof artificialScore === 'number'
                    ? Math.max(0, 1 - artificialScore)
                    : null;

            return (
              <div className="forensic-report-block">
                <div className="forensic-section-banner">
                  <h3>AI-Generated Image Detection</h3>
                  <span className="neutral-note">
                    Step 14.1.6: Hardened Model Output, Forensic Interpretation &amp; Limitations
                  </span>
                </div>

                <div className="forensic-subpanel">
                  <h4>A. Model Output</h4>
                  <dl className="compact-kv-list">
                    <div>
                      <dt>Status</dt>
                      <dd>{isAvailable ? 'Available' : 'Not Available'}</dd>
                    </div>
                    <div>
                      <dt>AI detector model</dt>
                      <dd className="mono">
                        {isAvailable
                          ? aiDet.detector.architecture || 'SwinForImageClassification'
                          : 'Not configured'}
                      </dd>
                    </div>
                    <div>
                      <dt>Model repository</dt>
                      <dd className="mono">
                        {isAvailable ? aiDet.detector.name : 'Not configured'}
                      </dd>
                    </div>
                    {isAvailable && (
                      <>
                        <div>
                          <dt>Pinned revision</dt>
                          <dd className="mono">{pinnedRev}</dd>
                        </div>
                        <div>
                          <dt>License</dt>
                          <dd>{aiDet.detector.license}</dd>
                        </div>
                        <div>
                          <dt>Model-selected class</dt>
                          <dd className="mono">
                            <strong>{selectedClass}</strong>{' '}
                            {typeof modelOut.selected_class_id === 'number'
                              ? `(class_id: ${modelOut.selected_class_id})`
                              : ''}
                          </dd>
                        </div>
                        {typeof artificialScore === 'number' && (
                          <div>
                            <dt>Artificial model score</dt>
                            <dd className="mono">
                              {artificialScore.toFixed(4)} (Uncalibrated model score)
                            </dd>
                          </div>
                        )}
                        {typeof humanScore === 'number' && (
                          <div>
                            <dt>Human model score</dt>
                            <dd className="mono">
                              {humanScore.toFixed(4)} (Uncalibrated model score)
                            </dd>
                          </div>
                        )}
                        <div>
                          <dt>Score type</dt>
                          <dd className="mono">
                            {aiDet.prediction.score_type || 'uncalibrated_model_score'}
                          </dd>
                        </div>
                        <div>
                          <dt>Decision rule</dt>
                          <dd className="mono">
                            {aiDet.prediction.decision_rule || 'model_class_selection'}
                          </dd>
                        </div>
                        <div>
                          <dt>Threshold</dt>
                          <dd className="mono">
                            {typeof aiDet.prediction.threshold === 'number'
                              ? aiDet.prediction.threshold.toFixed(4)
                              : 'not used (null)'}
                          </dd>
                        </div>
                      </>
                    )}
                  </dl>
                </div>

                <div className="forensic-subpanel">
                  <h4>B. Forensic Interpretation</h4>
                  <dl className="compact-kv-list">
                    <div>
                      <dt>Forensic Meaning</dt>
                      <dd>
                        {interp?.forensic_meaning ||
                          (isAvailable
                            ? aiDet.explanation
                            : aiDet?.explanation ||
                              'AI-generated image detection is currently unavailable because no compatible trained detector is configured.')}
                      </dd>
                    </div>
                    {interp?.recommended_use && (
                      <div>
                        <dt>Recommended Use</dt>
                        <dd>{interp.recommended_use}</dd>
                      </div>
                    )}
                    {txCtx && (
                      <div>
                        <dt>Transformation Sensitivity &amp; Context</dt>
                        <dd>
                          {txCtx.documented
                            ? `Documented signals (${txCtx.signals.join(', ')}): ${txCtx.note}`
                            : txCtx.note}
                        </dd>
                      </div>
                    )}
                    {interp?.interpretation_flags?.length > 0 && (
                      <div>
                        <dt>Interpretation Flags</dt>
                        <dd className="mono">{interp.interpretation_flags.join(', ')}</dd>
                      </div>
                    )}
                  </dl>
                  {interp?.prohibited_inference?.length > 0 && (
                    <div style={{ marginTop: '0.5rem' }}>
                      <p className="finding-title">Prohibited Inferences (Not Provenance or Authenticity Proof):</p>
                      <ul className="forensic-notes-list">
                        {interp.prohibited_inference.map((item, idx) => (
                          <li key={idx}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                {aiDet?.limitations?.length > 0 && (
                  <div className="forensic-subpanel">
                    <h4>C. Detector Limitations</h4>
                    <ul className="forensic-notes-list">
                      {aiDet.limitations.map((lim, idx) => (
                        <li key={idx}>{lim}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            );
          })()}



          {localization && (
            <div className="forensic-report-block">
              <div className="forensic-section-banner">
                <h3>Manipulation Localization</h3>
                <span className="neutral-note">Step 5: Spatial Anomaly Mapping</span>
              </div>

              <div className="forensic-subpanel">
                <h4>Localization Overview</h4>
                <dl className="compact-kv-list">
                  <div>
                    <dt>Status</dt>
                    <dd>{localization.status === 'completed' ? 'Completed' : localization.status}</dd>
                  </div>
                  <div>
                    <dt>Localized Anomaly</dt>
                    <dd>{localization.localized_anomaly ? 'Yes' : 'No'}</dd>
                  </div>
                  <div>
                    <dt>Method</dt>
                    <dd>
                      {localization.method === 'classical_localization_baseline'
                        ? 'Classical localization baseline'
                        : localization.method}
                    </dd>
                  </div>
                  <div>
                    <dt>Anomaly Threshold</dt>
                    <dd>{localization.threshold}</dd>
                  </div>
                </dl>
              </div>

              <div className="forensic-subpanel">
                <h4>Suspicious Regions ({localization.regions?.length || 0})</h4>
                {localization.regions?.length > 0 ? (
                  <div className="findings-list">
                    {localization.regions.map((region) => (
                      <div key={region.id} className="finding-card">
                        <div className="finding-card-header">
                          <span className="finding-category-tag">Region ID: {region.id}</span>
                          <span className="signal-strength-tag strength-moderate">
                            Candidate Region
                          </span>
                        </div>
                        <dl className="region-metrics-grid">
                          <div>
                            <dt>X</dt>
                            <dd className="mono">{region.x} px</dd>
                          </div>
                          <div>
                            <dt>Y</dt>
                            <dd className="mono">{region.y} px</dd>
                          </div>
                          <div>
                            <dt>Width</dt>
                            <dd className="mono">{region.width} px</dd>
                          </div>
                          <div>
                            <dt>Height</dt>
                            <dd className="mono">{region.height} px</dd>
                          </div>
                          <div>
                            <dt>Area</dt>
                            <dd className="mono">{region.area.toLocaleString()} px²</dd>
                          </div>
                          <div>
                            <dt>Mean Anomaly Score</dt>
                            <dd className="mono">{region.mean_anomaly_score}</dd>
                          </div>
                          <div>
                            <dt>Maximum Anomaly Score</dt>
                            <dd className="mono">{region.max_anomaly_score}</dd>
                          </div>
                        </dl>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="no-anomaly-text">
                    No localized anomaly detected by the baseline forensic method.
                  </p>
                )}
              </div>

              {localization.visualization && (
                <div className="forensic-subpanel">
                  <h4>Localization Heatmap</h4>
                  <div className="heatmap-preview-grid">
                    <div className="heatmap-card">
                      <span className="heatmap-caption">Heatmap Overlay ({localization.visualization.heatmap})</span>
                      <img
                        src={getLocalizationArtifactUrl(
                          uploadedCase.case_id,
                          'heatmap.png',
                          localization.localized_at
                        )}
                        alt="Spatial anomaly heatmap overlay"
                        className="localization-image"
                      />
                    </div>
                    <div className="heatmap-card">
                      <span className="heatmap-caption">Anomaly Mask ({localization.visualization.mask})</span>
                      <img
                        src={getLocalizationArtifactUrl(
                          uploadedCase.case_id,
                          'mask.png',
                          localization.localized_at
                        )}
                        alt="Spatial anomaly threshold mask"
                        className="localization-image"
                      />
                    </div>
                  </div>
                </div>
              )}

              <div className="forensic-disclaimer-box">
                <strong>Forensic note:</strong> Localized anomalies are signals for further investigation and do not by
                themselves prove manipulation.
              </div>
            </div>
          )}

          {/* Step 6: Original vs Edited Comparison */}
          <div className="forensic-report-block">
            <div className="forensic-section-banner">
              <h3>Original vs Edited Comparison</h3>
              <span className="neutral-note">Step 6: Reference Image Comparison</span>
            </div>

            <div className="forensic-subpanel">
              <h4>Upload Reference Image</h4>
              <div className="reference-upload-controls">
                <label className="reference-file-picker">
                  <input
                    type="file"
                    accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
                    onChange={handleRefFileChange}
                    disabled={isBusy}
                  />
                </label>
                <button
                  type="button"
                  className="secondary-action-button"
                  onClick={handleUploadReference}
                  disabled={!selectedRefFile || isBusy}
                >
                  {isUploadingRef ? 'Uploading Reference...' : 'Upload Reference Image'}
                </button>
              </div>

              {refImage && (
                <div className="reference-receipt">
                  <dl className="compact-kv-list">
                    <div>
                      <dt>Reference Filename</dt>
                      <dd>{refImage.original_filename}</dd>
                    </div>
                    <div>
                      <dt>File Size</dt>
                      <dd>{formatBytes(refImage.file_size)}</dd>
                    </div>
                    <div>
                      <dt>Dimensions</dt>
                      <dd>
                        {refImage.width} x {refImage.height} px ({refImage.format})
                      </dd>
                    </div>
                    <div>
                      <dt>Reference SHA-256</dt>
                      <dd className="mono hash-text">{refImage.sha256}</dd>
                    </div>
                  </dl>

                  <div className="compare-action-row">
                    <button
                      type="button"
                      className="primary-button"
                      onClick={handleCompareImages}
                      disabled={isBusy}
                    >
                      {isComparing
                        ? 'Comparing Images...'
                        : comparison
                          ? 'Re-Compare Images'
                          : 'Compare Images'}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {comparison && (
              <>
                <div className="forensic-subpanel">
                  <h4>Comparison Summary</h4>
                  <div className="assessment-banner assessment-insufficient_evidence">
                    <span className="assessment-title">
                      Assessment: {formatComparisonAssessment(comparison.assessment, comparison.assessment_label)}
                    </span>
                    <span className="assessment-code mono">({comparison.assessment})</span>
                  </div>

                  <dl className="compact-kv-list" style={{ marginTop: '0.75rem' }}>
                    <div>
                      <dt>Case Image Dimensions</dt>
                      <dd>
                        {comparison.dimensions.case_dimensions.width} x{' '}
                        {comparison.dimensions.case_dimensions.height} px
                      </dd>
                    </div>
                    <div>
                      <dt>Reference Image Dimensions</dt>
                      <dd>
                        {comparison.dimensions.reference_dimensions.width} x{' '}
                        {comparison.dimensions.reference_dimensions.height} px
                      </dd>
                    </div>
                    <div>
                      <dt>Mean Absolute Difference</dt>
                      <dd className="mono">{comparison.pixel_difference.mean_absolute_difference}</dd>
                    </div>
                    <div>
                      <dt>Maximum Pixel Difference</dt>
                      <dd className="mono">{comparison.pixel_difference.max_pixel_difference}</dd>
                    </div>
                    <div>
                      <dt>Changed Pixel Percentage</dt>
                      <dd className="mono">{comparison.pixel_difference.changed_pixel_percentage}%</dd>
                    </div>
                    <div>
                      <dt>Comparison Mode</dt>
                      <dd className="mono">{comparison.comparison_mode}</dd>
                    </div>
                  </dl>
                </div>

                <div className="forensic-subpanel">
                  <h4>Changed Regions ({comparison.changed_regions?.length || 0})</h4>
                  {comparison.changed_regions?.length > 0 ? (
                    <div className="findings-list">
                      {comparison.changed_regions.map((region) => (
                        <div key={region.id} className="finding-card">
                          <div className="finding-card-header">
                            <span className="finding-category-tag">Changed Region: {region.id}</span>
                            <span className="signal-strength-tag strength-weak">
                              Visual Difference
                            </span>
                          </div>
                          <dl className="region-metrics-grid">
                            <div>
                              <dt>Region ID</dt>
                              <dd className="mono">{region.id}</dd>
                            </div>
                            <div>
                              <dt>X</dt>
                              <dd className="mono">{region.x} px</dd>
                            </div>
                            <div>
                              <dt>Y</dt>
                              <dd className="mono">{region.y} px</dd>
                            </div>
                            <div>
                              <dt>Width</dt>
                              <dd className="mono">{region.width} px</dd>
                            </div>
                            <div>
                              <dt>Height</dt>
                              <dd className="mono">{region.height} px</dd>
                            </div>
                            <div>
                              <dt>Area</dt>
                              <dd className="mono">{region.area.toLocaleString()} px²</dd>
                            </div>
                            <div>
                              <dt>Changed Pixel Count</dt>
                              <dd className="mono">{region.changed_pixel_count.toLocaleString()}</dd>
                            </div>
                            <div>
                              <dt>Changed Pixel Percentage</dt>
                              <dd className="mono">{region.changed_pixel_percentage}%</dd>
                            </div>
                          </dl>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="no-anomaly-text">
                      No changed regions exceeded the minimum difference threshold ({comparison.pixel_difference.difference_threshold}).
                    </p>
                  )}
                </div>

                {comparison.visualization && (
                  <div className="forensic-subpanel">
                    <h4>Image Difference Visualization &amp; Difference Mask</h4>
                    <div className="heatmap-preview-grid">
                      <div className="heatmap-card">
                        <span className="heatmap-caption">
                          Image Difference Visualization ({comparison.visualization.difference})
                        </span>
                        <img
                          src={getComparisonArtifactUrl(
                            uploadedCase.case_id,
                            'difference.png',
                            comparison.compared_at
                          )}
                          alt="Image Difference Visualization"
                          className="localization-image"
                        />
                      </div>
                      <div className="heatmap-card">
                        <span className="heatmap-caption">
                          Difference Mask ({comparison.visualization.mask})
                        </span>
                        <img
                          src={getComparisonArtifactUrl(
                            uploadedCase.case_id,
                            'difference_mask.png',
                            comparison.compared_at
                          )}
                          alt="Difference Mask"
                          className="localization-image"
                        />
                      </div>
                    </div>
                  </div>
                )}

                <div className="forensic-disclaimer-box">
                  <strong>Comparison note:</strong> Image differences identify visual changes between the supplied
                  images. They do not by themselves prove intentional manipulation.
                </div>
              </>
            )}
          </div>

          {/* Step 7: Evidence Correlation */}
          <div className="forensic-report-block">
            <div className="forensic-section-banner">
              <h3>Evidence Correlation</h3>
              <button
                type="button"
                className="primary-button"
                onClick={handleGenerateEvidence}
                disabled={isBusy}
              >
                {isGeneratingEvidence
                  ? 'Correlating Evidence...'
                  : evidence
                    ? 'Regenerate Evidence Report'
                    : 'Generate Evidence Report'}
              </button>
            </div>

            {evidence && (
              <>
                <div className="forensic-subpanel">
                  <h4>Forensic Assessment</h4>
                  <div className="assessment-banner assessment-potential_manipulation">
                    <span className="assessment-title">
                      {formatEvidenceAssessment(evidence.assessment, evidence.assessment_label)}
                    </span>
                    <span className="assessment-code mono">({evidence.assessment})</span>
                  </div>
                </div>

                <div className="forensic-subpanel">
                  <h4>Evidence Summary</h4>
                  <dl className="compact-kv-list">
                    <div>
                      <dt>Total Evidence Items</dt>
                      <dd className="mono">{evidence.summary.total_evidence_items}</dd>
                    </div>
                    <div>
                      <dt>Informational Count</dt>
                      <dd className="mono">{evidence.summary.informational}</dd>
                    </div>
                    <div>
                      <dt>Weak Count</dt>
                      <dd className="mono">{evidence.summary.weak}</dd>
                    </div>
                    <div>
                      <dt>Moderate Count</dt>
                      <dd className="mono">{evidence.summary.moderate}</dd>
                    </div>
                    <div>
                      <dt>Analysis Stages Available</dt>
                      <dd>
                        {evidence.summary.analysis_stages_available?.length > 0
                          ? evidence.summary.analysis_stages_available.join(', ')
                          : 'integrity_only'}
                      </dd>
                    </div>
                  </dl>
                </div>

                {crossOverlapItems.length > 0 && (
                  <div className="forensic-subpanel">
                    <h4>Cross-Analysis Evidence ({crossOverlapItems.length})</h4>
                    <div className="findings-list">
                      {crossOverlapItems.map((item) => (
                        <div key={item.id} className="finding-card">
                          <div className="finding-card-header">
                            <span className="finding-category-tag">
                              {item.id} — Cross-Analysis Overlap
                            </span>
                            <span className="signal-strength-tag strength-moderate">
                              Strength: {item.strength}
                            </span>
                          </div>
                          <p className="finding-title">
                            Independent analyses identified overlapping spatial regions.
                          </p>
                          <dl className="region-metrics-grid">
                            <div>
                              <dt>Localization Region</dt>
                              <dd className="mono">{item.value?.localization_region}</dd>
                            </div>
                            <div>
                              <dt>Comparison Region</dt>
                              <dd className="mono">{item.value?.comparison_region}</dd>
                            </div>
                            <div>
                              <dt>Spatial Overlap</dt>
                              <dd className="mono">{item.value?.overlap_percentage}%</dd>
                            </div>
                          </dl>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="forensic-subpanel">
                  <h4>Evidence Items ({evidence.items?.length || 0})</h4>
                  <div className="findings-list">
                    {evidence.items?.map((item) => (
                      <div key={item.id} className="finding-card">
                        <div className="finding-card-header">
                          <span className="finding-category-tag">
                            {item.id} | {item.category}
                          </span>
                          <span className={`signal-strength-tag strength-${item.strength}`}>
                            Strength: {item.strength} | Source: {item.source}
                          </span>
                        </div>
                        <p className="finding-title">{item.title}</p>
                        <p className="finding-value" style={{ marginBottom: '0.25rem' }}>
                          {item.description}
                        </p>
                        <p className="finding-value mono">
                          Value: {formatEvidenceValue(item.value)}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="forensic-disclaimer-box">
                  <strong>Correlation note:</strong> {evidence.disclaimer}
                </div>
              </>
            )}
          </div>

          {/* Step 14.2: Multi-Signal Forensic Interpretation */}
          <div className="forensic-report-block">
            <div className="forensic-section-banner">
              <h3>Multi-Signal Forensic Interpretation</h3>
              <button
                type="button"
                className="primary-button"
                onClick={handleInterpretForensics}
                disabled={isBusy}
              >
                {isInterpretingForensics
                  ? 'Interpreting Forensic Signals...'
                  : forensicDecision
                    ? 'Refresh Multi-Signal Interpretation'
                    : 'Run Multi-Signal Forensic Interpretation'}
              </button>
            </div>

            {forensicDecision && (
              <>
                <div className="forensic-subpanel">
                  <h4>Overall Qualitative Assessment</h4>
                  <dl className="compact-kv-list">
                    <div>
                      <dt>Interpretation Status</dt>
                      <dd className="mono">{forensicDecision.status}</dd>
                    </div>
                    <div>
                      <dt>Qualitative Assessment</dt>
                      <dd>
                        <strong>{forensicDecision.overall_assessment_label}</strong>{' '}
                        <span className="mono">({forensicDecision.overall_assessment})</span>
                      </dd>
                    </div>
                    <div>
                      <dt>Consistency Classification</dt>
                      <dd className="mono">{forensicDecision.consistency_classification}</dd>
                    </div>
                    <div>
                      <dt>Conservative Interpretation</dt>
                      <dd>{forensicDecision.interpretation}</dd>
                    </div>
                    {forensicDecision.transformation_context && (
                      <div>
                        <dt>Transformation Context</dt>
                        <dd>{forensicDecision.transformation_context.statement}</dd>
                      </div>
                    )}
                  </dl>
                </div>

                <div className="forensic-subpanel">
                  <h4>Signal Inventory ({forensicDecision.signal_inventory?.length || 0})</h4>
                  <div className="findings-list">
                    {forensicDecision.signal_inventory?.map((sig) => (
                      <div key={sig.signal_id} className="finding-card">
                        <div className="finding-card-header">
                          <span className="finding-category-tag">
                            {sig.signal_id} | {sig.signal_category} ({sig.independence_group})
                          </span>
                          <span className="finding-severity-tag mono">
                            Availability: {sig.availability} | Strength: {sig.strength}
                          </span>
                        </div>
                        <p className="finding-title">
                          <strong>Observation:</strong> {sig.observation}
                        </p>
                        <p className="finding-value">
                          <strong>Forensic Interpretation:</strong> {sig.forensic_interpretation}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="forensic-subpanel">
                  <h4>
                    Consistency &amp; Conflict Relationships (
                    {forensicDecision.signal_relationships?.length || 0})
                  </h4>
                  <div className="findings-list">
                    {forensicDecision.signal_relationships?.map((rel, idx) => (
                      <div key={`${rel.signal_a}-${rel.signal_b}-${idx}`} className="finding-card">
                        <div className="finding-card-header">
                          <span className="finding-category-tag">
                            {rel.signal_a} &harr; {rel.signal_b}
                          </span>
                          <span className="finding-severity-tag mono">
                            {rel.relationship} ({rel.consistency})
                          </span>
                        </div>
                        <p className="finding-value">{rel.reason}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="forensic-subpanel">
                  <h4>Supported Conclusions</h4>
                  <ul className="forensic-notes-list">
                    {forensicDecision.supported_conclusions?.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>

                <div className="forensic-subpanel">
                  <h4>Unsupported Conclusions (What Cannot Be Concluded)</h4>
                  <ul className="forensic-notes-list">
                    {forensicDecision.unsupported_conclusions?.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>

                <div className="forensic-subpanel">
                  <h4>Alternative Non-Malicious Explanations</h4>
                  <ul className="forensic-notes-list">
                    {forensicDecision.alternative_explanations?.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>

                <div className="forensic-subpanel">
                  <h4>Additional Evidence Recommendations</h4>
                  <ul className="forensic-notes-list">
                    {forensicDecision.additional_evidence_recommendations?.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>

                <div className="forensic-subpanel">
                  <h4>Framework Limitations</h4>
                  <ul className="forensic-notes-list">
                    {forensicDecision.limitations?.map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>
              </>
            )}
          </div>

          {/* Step 14.3: Forensic Reproducibility & Audit Trail */}
          <div className="forensic-report-block">
            <div className="forensic-section-banner">
              <h3>Forensic Reproducibility &amp; Audit Trail</h3>
              <button
                type="button"
                className="primary-button"
                onClick={handleCheckReproducibility}
                disabled={isBusy || isCheckingReproducibility}
              >
                {isCheckingReproducibility
                  ? 'Verifying Reproducibility...'
                  : uploadedCase.latest_snapshot
                    ? 'Re-Verify Reproducibility & Audit Trail'
                    : 'Verify Reproducibility & Audit Trail'}
              </button>
            </div>

            {uploadedCase.latest_snapshot && (
              <>
                <div className="forensic-subpanel">
                  <h4>1. Snapshot Identity &amp; Reproducibility Status</h4>
                  <dl className="compact-kv-list">
                    <div>
                      <dt>Snapshot ID</dt>
                      <dd className="mono">{uploadedCase.latest_snapshot.snapshot_id}</dd>
                    </div>
                    <div>
                      <dt>Snapshot Schema Version</dt>
                      <dd className="mono">
                        v{uploadedCase.latest_snapshot.snapshot_schema_version}
                      </dd>
                    </div>
                    <div>
                      <dt>Canonical Snapshot SHA-256</dt>
                      <dd className="mono">{uploadedCase.latest_snapshot.snapshot_sha256}</dd>
                    </div>
                    <div>
                      <dt>Reproducibility Status</dt>
                      <dd className="mono">
                        <strong>
                          {uploadedCase.latest_snapshot.reproducibility_check?.status?.toUpperCase()}
                        </strong>
                      </dd>
                    </div>
                    <div>
                      <dt>Historical Analysis Registry</dt>
                      <dd className="mono">
                        {uploadedCase.latest_snapshot.reproducibility_check
                          ?.historical_analysis_registry || 'available'}
                      </dd>
                    </div>
                  </dl>
                  {uploadedCase.latest_snapshot.reproducibility_check?.reasons?.length > 0 && (
                    <ul className="forensic-notes-list" style={{ marginTop: '0.4rem' }}>
                      {uploadedCase.latest_snapshot.reproducibility_check.reasons.map(
                        (reason, idx) => (
                          <li key={idx}>{reason}</li>
                        )
                      )}
                    </ul>
                  )}
                </div>

                <div className="forensic-subpanel">
                  <h4>2. Input File Provenance &amp; Integrity Verification</h4>
                  <dl className="compact-kv-list">
                    <div>
                      <dt>Original Filename</dt>
                      <dd className="mono">
                        {uploadedCase.latest_snapshot.input_file_provenance?.original_filename}
                      </dd>
                    </div>
                    <div>
                      <dt>Relative Storage Path</dt>
                      <dd className="mono">
                        {uploadedCase.latest_snapshot.input_file_provenance?.relative_storage_path}
                      </dd>
                    </div>
                    <div>
                      <dt>Original Upload SHA-256</dt>
                      <dd className="mono">
                        {uploadedCase.latest_snapshot.input_integrity?.original_sha256}
                      </dd>
                    </div>
                    <div>
                      <dt>Recomputed File SHA-256</dt>
                      <dd className="mono">
                        {uploadedCase.latest_snapshot.input_integrity?.recomputed_sha256 ||
                          'unavailable'}
                      </dd>
                    </div>
                    <div>
                      <dt>Integrity Status</dt>
                      <dd className="mono">
                        {uploadedCase.latest_snapshot.input_integrity?.status?.toUpperCase()} (Hash
                        Match:{' '}
                        {uploadedCase.latest_snapshot.input_integrity?.hash_match ? 'Yes' : 'No'})
                      </dd>
                    </div>
                  </dl>
                </div>

                <div className="forensic-subpanel">
                  <h4>
                    3. Executed Analyses &amp; Versions (
                    {uploadedCase.latest_snapshot.analyses_executed?.length || 0})
                  </h4>
                  <div className="findings-list">
                    {uploadedCase.latest_snapshot.analyses_executed?.map((anl) => (
                      <div key={anl.analysis_id} className="finding-card">
                        <div className="finding-card-header">
                          <span className="finding-category-tag">
                            {anl.analysis_id} | {anl.service_name} (v{anl.service_version})
                          </span>
                          <span className="finding-severity-tag mono">
                            Status: {anl.status} | Config SHA-256:{' '}
                            {anl.configuration_hash?.slice(0, 12)}...
                          </span>
                        </div>
                        <p className="finding-value">{anl.output_summary}</p>
                        {(anl.rerun_of_analysis_id || anl.reused_analysis_id) && (
                          <p className="finding-value mono">
                            {anl.rerun_of_analysis_id
                              ? `Rerun of: ${anl.rerun_of_analysis_id} (config_changed=${anl.configuration_changed})`
                              : `Reused from: ${anl.reused_analysis_id}`}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="forensic-subpanel">
                  <h4>4. AI Model Provenance</h4>
                  <dl className="compact-kv-list">
                    <div>
                      <dt>Repository</dt>
                      <dd className="mono">
                        {uploadedCase.latest_snapshot.ai_model_provenance?.model_name}
                      </dd>
                    </div>
                    <div>
                      <dt>Pinned Revision</dt>
                      <dd className="mono">
                        {uploadedCase.latest_snapshot.ai_model_provenance?.pinned_revision}
                      </dd>
                    </div>
                    <div>
                      <dt>Architecture</dt>
                      <dd className="mono">
                        {uploadedCase.latest_snapshot.ai_model_provenance?.architecture}
                      </dd>
                    </div>
                    <div>
                      <dt>Score Type &amp; Decision Rule</dt>
                      <dd className="mono">
                        {uploadedCase.latest_snapshot.ai_model_provenance?.score_type} /{' '}
                        {uploadedCase.latest_snapshot.ai_model_provenance?.decision_rule} (Threshold:
                        None)
                      </dd>
                    </div>
                    <div>
                      <dt>ONNX Artifact SHA-256</dt>
                      <dd className="mono">
                        {uploadedCase.latest_snapshot.ai_model_provenance?.onnx_sha256 || 'N/A'}
                      </dd>
                    </div>
                  </dl>
                </div>

                <div className="forensic-subpanel">
                  <h4>
                    5. Derived Artifacts &amp; SHA-256 Digests (
                    {uploadedCase.latest_snapshot.artifacts_produced?.length || 0})
                  </h4>
                  {uploadedCase.latest_snapshot.artifacts_produced?.length > 0 ? (
                    <div className="findings-list">
                      {uploadedCase.latest_snapshot.artifacts_produced.map((art) => (
                        <div key={art.artifact_id} className="finding-card">
                          <div className="finding-card-header">
                            <span className="finding-category-tag">
                              {art.artifact_id} | {art.artifact_type}
                            </span>
                            <span className="finding-severity-tag mono">
                              {formatBytes(art.byte_size)}
                              {art.superseded_by_artifact_id
                                ? ` | Superseded by ${art.superseded_by_artifact_id}`
                                : ' | Active'}
                            </span>
                          </div>
                          <p className="finding-value mono">Path: {art.relative_path}</p>
                          <p className="finding-value mono">SHA-256: {art.sha256}</p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="empty-state">No derived artifacts registered yet.</p>
                  )}
                </div>

                <div className="forensic-subpanel">
                  <h4>
                    6. System Audit Event Log ({uploadedCase.audit_events?.length || 0} Events —
                    Strictly Separate from Media Edit History)
                  </h4>
                  {uploadedCase.audit_events?.length > 0 ? (
                    <div className="findings-list">
                      {uploadedCase.audit_events.slice(-10).map((ev) => (
                        <div key={ev.event_id} className="finding-card">
                          <div className="finding-card-header">
                            <span className="finding-category-tag">
                              {ev.event_id} | {ev.event_type} ({ev.event_category})
                            </span>
                            <span className="finding-severity-tag mono">{ev.timestamp}</span>
                          </div>
                          <p className="finding-value">{ev.summary}</p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="empty-state">No system audit events recorded yet.</p>
                  )}
                </div>

                <div className="forensic-subpanel">
                  <h4>7. Reproducibility Limitations</h4>
                  <ul className="forensic-notes-list">
                    {uploadedCase.latest_snapshot.reproducibility_limitations?.map((lim, idx) => (
                      <li key={idx}>{lim}</li>
                    ))}
                  </ul>
                </div>
              </>
            )}
          </div>

          {/* Step 8: Possible Edit History Reconstruction */}
          <div className="forensic-report-block">
            <div className="forensic-section-banner">
              <h3>Possible Edit History</h3>
              <button
                type="button"
                className="primary-button"
                onClick={handleReconstructHistory}
                disabled={isBusy}
              >
                {isReconstructingHistory
                  ? 'Reconstructing Edit History...'
                  : history
                    ? 'Regenerate Edit History'
                    : 'Reconstruct Edit History'}
              </button>
            </div>

            {history && (
              <>
                <div className="forensic-subpanel">
                  <h4>Timeline Status</h4>
                  <dl className="compact-kv-list">
                    <div>
                      <dt>Sequence Type</dt>
                      <dd className="mono">
                        Evidence-Ordered Hypothesis ({history.sequence_type})
                      </dd>
                    </div>
                    <div>
                      <dt>Historical Accuracy</dt>
                      <dd className="mono">
                        {history.historical_accuracy === 'partially_supported'
                          ? 'Partially Supported'
                          : 'Not Determined'}{' '}
                        ({history.historical_accuracy})
                      </dd>
                    </div>
                    <div>
                      <dt>Integrity Preserved</dt>
                      <dd className="mono">
                        {history.integrity?.hash_preserved ? 'Yes' : 'No'} (SHA-256:{' '}
                        {history.integrity?.original_sha256})
                      </dd>
                    </div>
                  </dl>
                </div>

                <div className="forensic-subpanel">
                  <h4>Timeline Events ({history.events?.length || 0})</h4>
                  <div className="findings-list">
                    {history.events?.map((ev) => (
                      <div key={ev.id} className="finding-card">
                        <div className="finding-card-header">
                          <span className="finding-category-tag">
                            #{ev.sequence} | {ev.id} | {ev.event_type}
                          </span>
                          <span className={`signal-strength-tag strength-${ev.confidence}`}>
                            Status: {ev.status} | Confidence: {ev.confidence} | Temporal:{' '}
                            {ev.temporal_certainty}
                          </span>
                        </div>
                        <p className="finding-title">{ev.title}</p>
                        <p className="finding-value" style={{ marginBottom: '0.35rem' }}>
                          {ev.description}
                        </p>
                        <p className="finding-value mono" style={{ marginBottom: '0.35rem' }}>
                          Source: {ev.source} | Evidence IDs:{' '}
                          {ev.evidence_ids?.length > 0 ? ev.evidence_ids.join(', ') : 'None'}
                        </p>
                        <p className="finding-value" style={{ marginBottom: '0.25rem' }}>
                          <strong>Reasoning:</strong> {ev.reasoning}
                        </p>
                        {ev.limitations?.length > 0 && (
                          <div className="finding-value">
                            <strong>Limitations:</strong>
                            <ul className="forensic-notes-list" style={{ marginTop: '0.25rem' }}>
                              {ev.limitations.map((lim, idx) => (
                                <li key={idx}>{lim}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="forensic-subpanel">
                  <h4>Known Facts ({history.known_facts?.length || 0})</h4>
                  {history.known_facts?.length > 0 ? (
                    <div className="findings-list">
                      {history.known_facts.map((item, idx) => (
                        <div key={idx} className="finding-card">
                          <div className="finding-card-header">
                            <span className="finding-category-tag">Source: {item.source}</span>
                            <span className="signal-strength-tag strength-informational">
                              Evidence: {item.evidence_ids?.join(', ') || 'N/A'}
                            </span>
                          </div>
                          <p className="finding-title">{item.fact}</p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="empty-state">No known facts recorded.</p>
                  )}
                </div>

                <div className="forensic-subpanel">
                  <h4>Possible Inferred Events ({history.inferred_events?.length || 0})</h4>
                  {history.inferred_events?.length > 0 ? (
                    <div className="findings-list">
                      {history.inferred_events.map((inf, idx) => (
                        <div key={idx} className="finding-card">
                          <div className="finding-card-header">
                            <span className="finding-category-tag">{inf.event_type}</span>
                            <span className={`signal-strength-tag strength-${inf.confidence}`}>
                              Confidence: {inf.confidence} | Evidence:{' '}
                              {inf.evidence_ids?.join(', ') || 'N/A'}
                            </span>
                          </div>
                          <p className="finding-title">{inf.title}</p>
                          <p className="finding-value" style={{ marginBottom: '0.25rem' }}>
                            <strong>Reasoning:</strong> {inf.reasoning}
                          </p>
                          {inf.limitations?.length > 0 && (
                            <ul className="forensic-notes-list">
                              {inf.limitations.map((lim, lIdx) => (
                                <li key={lIdx}>{lim}</li>
                              ))}
                            </ul>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="empty-state">
                      No inferred processing or manipulation events identified from available evidence.
                    </p>
                  )}
                </div>

                <div className="forensic-subpanel">
                  <h4>What Cannot Be Determined</h4>
                  <ul className="forensic-notes-list">
                    {history.unknowns?.map((unk, idx) => (
                      <li key={idx}>{unk}</li>
                    ))}
                  </ul>
                </div>

                <div className="forensic-subpanel">
                  <h4>Reconstruction Limitations</h4>
                  <ul className="forensic-notes-list">
                    {history.limitations?.map((lim, idx) => (
                      <li key={idx}>{lim}</li>
                    ))}
                  </ul>
                </div>
              </>
            )}
          </div>

          {/* Step 9: Forensic Report Generation */}
          <div className="forensic-report-block">
            <div className="forensic-section-banner" style={{ flexWrap: 'wrap', gap: '0.5rem' }}>
              <h3>Forensic Report</h3>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                <button
                  type="button"
                  className="primary-button"
                  onClick={handleGenerateReport}
                  disabled={isBusy}
                >
                  {isGeneratingReport ? 'Generating Report...' : 'Generate Forensic Report'}
                </button>
                <button
                  type="button"
                  className="primary-button"
                  onClick={handleViewReport}
                  disabled={isBusy}
                >
                  {isViewingReport ? 'Loading Report...' : 'View Report'}
                </button>
                <button
                  type="button"
                  className="primary-button"
                  onClick={handleDownloadPdfReport}
                  disabled={isBusy}
                >
                  {isDownloadingPdf ? 'Building PDF...' : 'Download PDF Report'}
                </button>
              </div>
            </div>

            <div className="forensic-subpanel">
              <h4>Report Status</h4>
              <dl className="compact-kv-list">
                <div>
                  <dt>Status</dt>
                  <dd className="mono">
                    {forensicReport ? 'Report generated' : 'Not generated'}
                  </dd>
                </div>
                <div>
                  <dt>Report ID</dt>
                  <dd className="mono">{forensicReport?.report_id || '-'}</dd>
                </div>
                <div>
                  <dt>Report Version</dt>
                  <dd className="mono">{forensicReport?.report_version || '1.0'}</dd>
                </div>
                <div>
                  <dt>Generated Timestamp</dt>
                  <dd className="mono">{forensicReport?.generated_at || '-'}</dd>
                </div>
              </dl>
            </div>

            {forensicReport && (
              <>
                <div className="forensic-subpanel">
                  <h4>Assessment</h4>
                  <div className="assessment-banner assessment-potential_manipulation">
                    <span className="assessment-title">
                      {forensicReport.assessment?.assessment_label}
                    </span>
                    <span className="assessment-code mono">
                      ({forensicReport.assessment?.assessment})
                    </span>
                  </div>
                  <p className="finding-value" style={{ marginTop: '0.4rem' }}>
                    {forensicReport.assessment?.note}
                  </p>
                </div>

                <div className="forensic-subpanel">
                  <h4>Integrity</h4>
                  <dl className="compact-kv-list">
                    <div>
                      <dt>SHA-256</dt>
                      <dd className="mono">{forensicReport.integrity?.sha256}</dd>
                    </div>
                    <div>
                      <dt>Hash Preserved</dt>
                      <dd className="mono">
                        {forensicReport.integrity?.hash_preserved ? 'Yes' : 'No'}
                      </dd>
                    </div>
                  </dl>
                  <p className="finding-value" style={{ marginTop: '0.4rem' }}>
                    {forensicReport.integrity?.explanation} {forensicReport.integrity?.scope_notice}
                  </p>
                </div>

                <div className="forensic-subpanel">
                  <h4>Evidence Overview</h4>
                  <dl className="compact-kv-list">
                    <div>
                      <dt>Total Evidence</dt>
                      <dd className="mono">
                        {forensicReport.evidence?.summary?.total_evidence_items ?? 0}
                      </dd>
                    </div>
                    <div>
                      <dt>Informational</dt>
                      <dd className="mono">
                        {forensicReport.evidence?.summary?.informational ?? 0}
                      </dd>
                    </div>
                    <div>
                      <dt>Weak</dt>
                      <dd className="mono">{forensicReport.evidence?.summary?.weak ?? 0}</dd>
                    </div>
                    <div>
                      <dt>Moderate</dt>
                      <dd className="mono">{forensicReport.evidence?.summary?.moderate ?? 0}</dd>
                    </div>
                  </dl>
                </div>

                <div className="forensic-subpanel">
                  <h4>Possible Edit History ({forensicReport.history?.sequence_label})</h4>
                  <p className="finding-value" style={{ marginBottom: '0.4rem' }}>
                    <strong>{forensicReport.history?.historical_accuracy_label}</strong> —{' '}
                    {forensicReport.history?.timeline_disclaimer}
                  </p>
                  <div className="findings-list">
                    {forensicReport.history?.events?.map((ev) => (
                      <div key={ev.id} className="finding-card">
                        <div className="finding-card-header">
                          <span className="finding-category-tag">
                            #{ev.sequence} | {ev.id} | {ev.event_type}
                          </span>
                          <span className={`signal-strength-tag strength-${ev.confidence}`}>
                            {ev.status.toUpperCase().replace('_', ' ')} | Confidence:{' '}
                            {ev.confidence} | Temporal: {ev.temporal_certainty}
                          </span>
                        </div>
                        <p className="finding-title">{ev.title}</p>
                        <p className="finding-value">{ev.reasoning}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="forensic-subpanel">
                  <h4>Limitations</h4>
                  <ul className="forensic-notes-list">
                    {forensicReport.limitations?.map((lim, idx) => (
                      <li key={idx}>{lim}</li>
                    ))}
                  </ul>
                </div>

                <div className="forensic-disclaimer-box">
                  <strong>Report Notice:</strong> {forensicReport.final_notice}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
