/**
 * Medical Analysis Component
 * Displays detailed medical analysis and explanations
 */
import React, { useState } from 'react';
import '../styles/MedicalAnalysis.css';

const MedicalAnalysis = ({ analysis }) => {
  const [expandedSections, setExpandedSections] = useState({
    description: true,
    interpretation: true,
    characteristics: false,
    differential: false
  });
  
  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };
  
  return (
    <div className="medical-analysis">
      <h2 className="analysis-title">Detailed Medical Analysis</h2>
      
      {/* Confidence Level */}
      <div className="analysis-section confidence-section">
        <h4>Confidence Level</h4>
        <p className="confidence-interpretation">{analysis.confidence_level}</p>
      </div>
      
      {/* Description */}
      <div className="analysis-section">
        <div 
          className="section-header"
          onClick={() => toggleSection('description')}
        >
          <h4>About {analysis.classification}</h4>
          <span className="toggle-icon">
            {expandedSections.description ? '▼' : '▶'}
          </span>
        </div>
        {expandedSections.description && (
          <div className="section-content">
            <p className="short-desc">{analysis.description}</p>
            <p className="detailed-desc">{analysis.detailed_info}</p>
          </div>
        )}
      </div>
      
      {/* Model Interpretation */}
      <div className="analysis-section highlight-section">
        <div 
          className="section-header"
          onClick={() => toggleSection('interpretation')}
        >
          <h4>How the AI Model Analyzed This Scan</h4>
          <span className="toggle-icon">
            {expandedSections.interpretation ? '▼' : '▶'}
          </span>
        </div>
        {expandedSections.interpretation && (
          <div className="section-content">
            <p>{analysis.model_interpretation}</p>
          </div>
        )}
      </div>
      
      {/* Characteristics */}
      {analysis.characteristics && analysis.characteristics.length > 0 && (
        <div className="analysis-section">
          <div 
            className="section-header"
            onClick={() => toggleSection('characteristics')}
          >
            <h4>Typical Characteristics</h4>
            <span className="toggle-icon">
              {expandedSections.characteristics ? '▼' : '▶'}
            </span>
          </div>
          {expandedSections.characteristics && (
            <div className="section-content">
              <ul className="characteristics-list">
                {analysis.characteristics.map((char, index) => (
                  <li key={index}>{char}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
      
      {/* Differential Considerations */}
      {analysis.differential_considerations && analysis.differential_considerations.length > 0 && (
        <div className="analysis-section warning-section">
          <div 
            className="section-header"
            onClick={() => toggleSection('differential')}
          >
            <h4>Alternative Possibilities</h4>
            <span className="toggle-icon">
              {expandedSections.differential ? '▼' : '▶'}
            </span>
          </div>
          {expandedSections.differential && (
            <div className="section-content">
              <p className="differential-note">
                Due to moderate confidence, consider these alternatives:
              </p>
              <ul className="differential-list">
                {analysis.differential_considerations.map((diff, index) => (
                  <li key={index}>
                    <strong>{diff.class}:</strong> {diff.probability.toFixed(1)}% probability
                    <br />
                    <span className="differential-detail">{diff.note}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
      
      {/* Educational Context */}
      <div className="analysis-section info-section">
        <h4>Educational Context</h4>
        <p>{analysis.educational_context}</p>
      </div>
      
      {/* Disclaimer */}
      <div className="disclaimer-section">
        <h4>Important Disclaimer</h4>
        <p>{analysis.disclaimer}</p>
      </div>
    </div>
  );
};

export default MedicalAnalysis;
