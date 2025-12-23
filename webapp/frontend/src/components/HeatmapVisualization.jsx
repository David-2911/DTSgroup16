/**
 * Heatmap Visualization Component
 * Displays original image and Grad-CAM overlay
 */
import React, { useState } from 'react';
import '../styles/HeatmapVisualization.css';

const HeatmapVisualization = ({ originalImage, heatmapOverlay }) => {
  const [activeView, setActiveView] = useState('heatmap'); // 'original' or 'heatmap'
  
  return (
    <div className="heatmap-visualization">
      <h3 className="visualization-title">Model Attention Visualization</h3>
      <p className="visualization-description">
        The heatmap shows which regions of the MRI the AI model focused on during classification.
        Red areas indicate high attention, blue areas indicate low attention.
      </p>
      
      <div className="view-toggle">
        <button 
          className={`toggle-btn ${activeView === 'original' ? 'active' : ''}`}
          onClick={() => setActiveView('original')}
        >
          Original Image
        </button>
        <button 
          className={`toggle-btn ${activeView === 'heatmap' ? 'active' : ''}`}
          onClick={() => setActiveView('heatmap')}
        >
          Attention Heatmap
        </button>
      </div>
      
      <div className="image-container">
        {activeView === 'original' ? (
          <div className="image-wrapper">
            <img 
              src={`data:image/png;base64,${originalImage}`}
              alt="Original MRI scan"
              className="visualization-image"
            />
            <p className="image-label">Original MRI Scan</p>
          </div>
        ) : (
          <div className="image-wrapper">
            <img 
              src={`data:image/png;base64,${heatmapOverlay}`}
              alt="Grad-CAM heatmap overlay"
              className="visualization-image"
            />
            <p className="image-label">Grad-CAM Attention Map</p>
          </div>
        )}
      </div>
      
      <div className="heatmap-legend">
        <div className="legend-item">
          <div className="legend-color high"></div>
          <span>High Attention</span>
        </div>
        <div className="legend-item">
          <div className="legend-color medium"></div>
          <span>Medium Attention</span>
        </div>
        <div className="legend-item">
          <div className="legend-color low"></div>
          <span>Low Attention</span>
        </div>
      </div>
    </div>
  );
};

export default HeatmapVisualization;
