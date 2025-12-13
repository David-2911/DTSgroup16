import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import LoadingSpinner from './components/LoadingSpinner';
import ClassificationResult from './components/ClassificationResult';
import HeatmapVisualization from './components/HeatmapVisualization';
import MedicalAnalysis from './components/MedicalAnalysis';
import ErrorMessage from './components/ErrorMessage';
import { classifyImage, checkHealth } from './utils/api';
import './styles/App.css';

function App() {
  // Application state
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isBackendConnected, setIsBackendConnected] = useState(false);

  // Check backend health on mount
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        await checkHealth();
        setIsBackendConnected(true);
      } catch (err) {
        setIsBackendConnected(false);
        console.error('Backend not available:', err);
      }
    };

    checkBackendHealth();
    // Check every 30 seconds
    const interval = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Handle image selection
  const handleImageSelect = (file) => {
    setSelectedImage(file);
    setResult(null);
    setError(null);

    // Create preview URL
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(file);
  };

  // Handle classification
  const handleClassify = async () => {
    if (!selectedImage) {
      setError({ type: 'warning', message: 'Please select an image first.' });
      return;
    }

    if (!isBackendConnected) {
      setError({ type: 'error', message: 'Backend server is not available. Please ensure the Flask server is running on port 5000.' });
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await classifyImage(selectedImage);
      setResult(response);
    } catch (err) {
      setError({
        type: 'error',
        message: err.message || 'An error occurred during classification. Please try again.'
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Handle reset/new analysis
  const handleReset = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setResult(null);
    setError(null);
  };

  // Clear error
  const handleClearError = () => {
    setError(null);
  };

  return (
    <div className="app">
      <Header isConnected={isBackendConnected} />
      
      <main className="main-content">
        {/* Error Message */}
        {error && (
          <ErrorMessage
            type={error.type}
            message={error.message}
            onClose={handleClearError}
          />
        )}

        {/* Loading State */}
        {isLoading && <LoadingSpinner />}

        {/* Image Upload (show when not loading and no result) */}
        {!isLoading && !result && (
          <>
            <ImageUpload
              onImageSelect={handleImageSelect}
              selectedImage={selectedImage}
              imagePreview={imagePreview}
            />
            
            {selectedImage && (
              <div className="action-buttons">
                <button className="btn-primary" onClick={handleClassify}>
                  Analyze MRI Scan
                </button>
                <button className="btn-secondary" onClick={handleReset}>
                  Clear
                </button>
              </div>
            )}
          </>
        )}

        {/* Results Display */}
        {!isLoading && result && (
          <>
            <div className="results-container">
              <div className="results-left">
                <ClassificationResult
                  prediction={result.prediction}
                  probabilities={result.probabilities}
                />
                <HeatmapVisualization
                  originalImage={result.visualization?.original_image}
                  heatmapOverlay={result.visualization?.heatmap_overlay}
                />
              </div>
              <div className="results-right">
                <MedicalAnalysis analysis={result.analysis} />
              </div>
            </div>
            
            <div className="action-buttons">
              <button className="btn-primary" onClick={handleReset}>
                Analyze New Image
              </button>
            </div>
          </>
        )}
      </main>

      <footer className="app-footer">
        <p>
          Brain MRI Classification System | Distributed Deep Learning Project
          <br />
          <small>For educational and research purposes only. Not for clinical diagnosis.</small>
        </p>
      </footer>
    </div>
  );
}

export default App;
