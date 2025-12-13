/**
 * Classification Result Component
 * Displays prediction, confidence, and probability breakdown
 */
import React from 'react';
import '../styles/ClassificationResult.css';

const ClassificationResult = ({ prediction }) => {
  const { class: tumorClass, confidence, probabilities } = prediction;
  
  // Sort probabilities for display
  const sortedProbs = Object.entries(probabilities)
    .sort(([, a], [, b]) => b - a);
  
  // Determine confidence level color
  const getConfidenceColor = (conf) => {
    if (conf >= 90) return 'confidence-high';
    if (conf >= 75) return 'confidence-medium';
    return 'confidence-low';
  };
  
  return (
    <div className="classification-result">
      <h2 className="result-title">Classification Result</h2>
      
      <div className="result-main">
        <div className="prediction-box">
          <p className="prediction-label">Detected Tumor Type:</p>
          <h3 className="prediction-value">{tumorClass}</h3>
        </div>
        
        <div className={`confidence-box ${getConfidenceColor(confidence)}`}>
          <p className="confidence-label">Confidence</p>
          <div className="confidence-value">{confidence.toFixed(1)}%</div>
          <div className="confidence-bar">
            <div 
              className="confidence-fill"
              style={{ width: `${confidence}%` }}
            ></div>
          </div>
        </div>
      </div>
      
      <div className="probabilities-section">
        <h4 className="probabilities-title">Probability Breakdown</h4>
        <div className="probabilities-list">
          {sortedProbs.map(([cls, prob]) => (
            <div key={cls} className="probability-item">
              <div className="probability-header">
                <span className="probability-class">{cls}</span>
                <span className="probability-value">{prob.toFixed(1)}%</span>
              </div>
              <div className="probability-bar">
                <div 
                  className="probability-fill"
                  style={{ width: `${prob}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ClassificationResult;
