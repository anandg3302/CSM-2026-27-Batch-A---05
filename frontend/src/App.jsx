import React, { useEffect, useState } from 'react';
import './App.css';
import Header from './components/Header.jsx';
import DashboardPlaceholder from './components/DashboardPlaceholder.jsx';
import MediaUpload from './components/MediaUpload.jsx';
import ResultsPlaceholder from './components/ResultsPlaceholder.jsx';
import { fetchHealthStatus, getMediaCase, getMediaCases } from './services/api.js';

export default function App() {
  const [health, setHealth] = useState({ status: 'offline', service: 'MediaTrace API' });
  const [recentCases, setRecentCases] = useState([]);
  const [activeCase, setActiveCase] = useState(null);
  const [registryWarning, setRegistryWarning] = useState('');

  useEffect(() => {
    let isMounted = true;

    fetchHealthStatus()
      .then((data) => {
        if (isMounted) {
          setHealth(data);
        }
      })
      .catch(() => {
        if (isMounted) {
          setHealth({ status: 'offline', service: 'MediaTrace API' });
        }
      });

    getMediaCases()
      .then((cases) => {
        if (isMounted && Array.isArray(cases)) {
          setHealth((prev) => ({ ...prev, status: 'ok' }));
          setRecentCases(cases);
          if (cases.length > 0) {
            setActiveCase(cases[0]);
          }
        }
      })
      .catch(() => {
        // Ignore if backend is offline initially
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const refreshCaseList = async (updatedCase) => {
    if (updatedCase && updatedCase.case_id) {
      setHealth((prev) => ({ ...prev, status: 'ok' }));
      setActiveCase(updatedCase);
      setRecentCases((prev) => [
        updatedCase,
        ...prev.filter((item) => item.case_id !== updatedCase.case_id),
      ]);
    }
    setRegistryWarning('');

    const updatedList = await getMediaCases();
    if (Array.isArray(updatedList)) {
      const hasUpdated =
        updatedCase?.case_id &&
        updatedList.some((item) => item.case_id === updatedCase.case_id);
      setRecentCases(hasUpdated ? updatedList : [updatedCase, ...updatedList]);
    }
  };

  const handleSelectCase = async (caseId) => {
    setRegistryWarning('');
    try {
      const detailedCase = await getMediaCase(caseId);
      setHealth((prev) => ({ ...prev, status: 'ok' }));
      setActiveCase(detailedCase);
    } catch (err) {
      const found = recentCases.find((item) => item.case_id === caseId);
      if (found) {
        setActiveCase(found);
      }
      setRegistryWarning(
        `Follow-up request GET /api/media/${caseId} failed: ${err.message || 'Unable to load case details.'}`
      );
    }
  };

  return (
    <div className="app-shell">
      <Header apiStatus={health.status} serviceName={health.service} />

      <main className="main-container">
        <DashboardPlaceholder />

        {registryWarning && (
          <div className="alert alert-error" role="alert" style={{ marginBottom: '1rem' }}>
            <strong>Follow-up Case Fetch Notice:</strong> {registryWarning}
          </div>
        )}

        <div className="workspace-split">
          <MediaUpload
            activeCase={activeCase}
            onUploadSuccess={refreshCaseList}
            onAnalysisSuccess={refreshCaseList}
          />
          <ResultsPlaceholder
            activeCase={activeCase}
            recentCases={recentCases}
            onSelectCase={handleSelectCase}
          />
        </div>
      </main>
    </div>
  );
}
