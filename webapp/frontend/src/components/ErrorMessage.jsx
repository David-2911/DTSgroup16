/**
 * Error Message Component
 * Displays error messages with dismiss option
 */
import React from 'react';
import '../styles/ErrorMessage.css';

const ErrorMessage = ({ message, type = 'error', onClose }) => {
  return (
    <div className={`error-message ${type}`}>
      <div className="error-content">
        <span className="error-icon">
          {type === 'error' && '!'}
          {type === 'warning' && '!'}
          {type === 'info' && 'i'}
        </span>
        <p className="error-text">{message}</p>
      </div>
      {onClose && (
        <button className="error-close" onClick={onClose}>
          ✕
        </button>
      )}
    </div>
  );
};

export default ErrorMessage;
