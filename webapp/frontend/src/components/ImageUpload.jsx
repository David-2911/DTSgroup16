/**
 * Image Upload Component
 * Drag-and-drop or click to upload MRI images
 */
import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import '../styles/ImageUpload.css';

const ImageUpload = ({ onImageSelect, preview }) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      onImageSelect(acceptedFiles[0]);
    }
  }, [onImageSelect]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png']
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024 // 10MB
  });
  
  return (
    <div className="upload-container">
      <h2 className="section-title">Upload Brain MRI Scan</h2>
      
      <div 
        {...getRootProps()} 
        className={`dropzone ${isDragActive ? 'dropzone-active' : ''} ${preview ? 'has-preview' : ''}`}
      >
        <input {...getInputProps()} />
        
        {!preview ? (
          <div className="dropzone-content">
            <div className="upload-icon"></div>
            <p className="upload-text-primary">
              {isDragActive ? 'Drop the image here' : 'Drag & drop an MRI image here'}
            </p>
            <p className="upload-text-secondary">or click to select a file</p>
            <div className="file-requirements">
              <p>Accepted formats: JPG, PNG</p>
              <p>Maximum size: 10MB</p>
            </div>
          </div>
        ) : (
          <div className="preview-container">
            <img 
              src={preview} 
              alt="Uploaded MRI preview" 
              className="preview-image"
            />
            <div className="preview-overlay">
              <p>Click to change image</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUpload;
