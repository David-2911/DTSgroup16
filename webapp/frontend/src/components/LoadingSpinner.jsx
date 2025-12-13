/**
 * Loading Spinner Component
 * Shows during image classification
 */
import React from 'react';
import '../styles/LoadingSpinner.css';

const LoadingSpinner = ({ message = 'Loading...' }) => {
  return (
    <div className="loading-container">
      <div className="spinner-wrapper">
        <div className="spinner">
          <div className="spinner-circle"></div>
          <div className="spinner-circle"></div>
          <div className="spinner-circle"></div>
        </div>
        <p className="loading-message">{message}</p>
        <p className="loading-submessage">
          This may take a few seconds...
        </p>
      </div>
    </div>
  );
};

export default LoadingSpinner;
