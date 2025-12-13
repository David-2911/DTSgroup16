/**
 * Header Component
 * Displays app title, description, and backend status
 */
import React from 'react';
import '../styles/Header.css';

const Header = ({ backendStatus }) => {
  return (
    <header className="app-header">
      <div className="header-content">
        <div className="title-section">
          <h1 className="app-title">Brain MRI Tumor Classification</h1>
          <p className="app-subtitle">
            AI-Powered Medical Imaging Analysis using Deep Learning
          </p>
        </div>
        
        <div className="status-section">
          <div className={`status-indicator status-${backendStatus}`}>
            <span className="status-dot"></span>
            <span className="status-text">
              {backendStatus === 'online' && 'System Online'}
              {backendStatus === 'offline' && 'System Offline'}
              {backendStatus === 'checking' && 'Checking...'}
            </span>
          </div>
        </div>
      </div>
      
      <div className="header-info">
        <div className="info-item">
          <span className="info-label">Model:</span>
          <span className="info-value">ResNet50 (Transfer Learning)</span>
        </div>
        <div className="info-item">
          <span className="info-label">Accuracy:</span>
          <span className="info-value">~92%</span>
        </div>
        <div className="info-item">
          <span className="info-label">Classes:</span>
          <span className="info-value">Glioma, Meningioma, No Tumor, Pituitary</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
