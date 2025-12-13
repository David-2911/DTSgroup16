# Brain MRI Tumor Classification Web Application

A full-stack web application for classifying brain MRI scans using deep learning, built with Flask backend and React frontend.

## Features

- **AI-Powered Classification**: ResNet50 model trained on 5,712 brain MRI images
- **Real-Time Analysis**: Upload and classify MRI scans in seconds
- **Grad-CAM Visualization**: See which regions the AI focuses on
- **Detailed Medical Analysis**: Comprehensive explanations and context
- **Professional UI**: Clean, medical-grade interface with pastel design
- **Privacy-First**: Images discarded after classification

## Model Performance

- **Architecture**: ResNet50 with transfer learning from ImageNet
- **Classes**: Glioma, Meningioma, No Tumor, Pituitary
- **Accuracy**: ~92% on test set
- **Technology Stack**: Spark/Hadoop for distributed processing

## Architecture

```
Frontend (React)          Backend (Flask)           Infrastructure
     |                         |                           |
  Port 3000    <---HTTP--->  Port 5000    <--->      Spark/Hadoop
     |                         |                           |
 User Upload  →  API Request  →  Model Inference  →  HDFS Storage
     ↓                         ↓                           ↓
 Display Results ← JSON Response ← Grad-CAM + Analysis ← Preprocessing
```

## Prerequisites

### System Requirements
- **RAM**: 8GB minimum
- **Storage**: 5GB free space
- **OS**: Linux, macOS, or Windows

### Software Requirements
- Python 3.8+
- Node.js 14+ and npm
- Existing Spark/Hadoop installation (from main project)
- Trained model: `best_model_stage1.keras`

## Quick Start

### 1. Start Hadoop Services
```bash
# Ensure Hadoop is running
jps  # Should show NameNode and DataNode

# If not running, start services
start-dfs.sh  # Linux/Mac
# or
start-dfs.cmd  # Windows
```

### 2. Start Backend
```bash
cd DTSgroup16/webapp/backend/

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure paths in config.py if needed
# - MODEL_PATH: Path to best_model_stage1.keras
# - HDFS_TEMP_PATH: Your HDFS temporary directory

# Start Flask API
python app.py

# Backend will be available at: http://localhost:5000
```

### 3. Start Frontend
```bash
# Open new terminal
cd DTSgroup16/webapp/frontend/

# Install dependencies (first time only)
npm install

# Start React app
npm start

# Frontend will open automatically at: http://localhost:3000
```

### 4. Use the Application

1. Open browser to `http://localhost:3000`
2. Drag and drop an MRI image or click to upload
3. Click "Analyze MRI Scan"
4. Wait 3-8 seconds for analysis
5. View results, heatmap, and detailed analysis

## Project Structure

```
webapp/
├── backend/                 # Flask API
│   ├── app.py              # Main API server
│   ├── model_loader.py     # Model management
│   ├── preprocessing.py    # Spark preprocessing
│   ├── gradcam.py          # Grad-CAM visualization
│   ├── analysis_generator.py  # Medical analysis
│   ├── config.py           # Configuration
│   ├── utils.py            # Helper functions
│   ├── requirements.txt    # Python dependencies
│   └── README.md           # Backend docs
│
├── frontend/               # React application
│   ├── public/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── styles/         # CSS styles
│   │   ├── utils/          # API utilities
│   │   └── App.jsx         # Main app
│   ├── package.json        # Node dependencies
│   └── README.md           # Frontend docs
│
├── shared/                 # Shared resources
│   └── tumor_descriptions.json
│
└── README.md              # This file
```

## Configuration

### Backend Configuration (backend/config.py)
```python
# Key settings to customize:

# Model path (relative to backend directory)
MODEL_PATH = '../../best_model_stage1.keras'

# HDFS temporary storage
HDFS_TEMP_PATH = '/user/YOUR_USERNAME/webapp_temp/'

# File upload limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Spark memory (adjust based on your RAM)
SPARK_DRIVER_MEMORY = '2g'
SPARK_EXECUTOR_MEMORY = '2g'
```

### Frontend Configuration (frontend/src/utils/api.js)
```javascript
// Backend API URL
const API_BASE_URL = 'http://localhost:5000/api';

// Adjust if backend runs on different port
```

## Testing

### Backend Testing
```bash
cd webapp/backend/

# Test health endpoint
curl http://localhost:5000/api/health

# Test classification (replace with your image path)
curl -X POST -F "image=@../../brain_Tumor_Types/glioma/Tr-gl_0010.jpg" \
     http://localhost:5000/api/classify

# Expected: JSON response with prediction, visualization, and analysis
```

### Frontend Testing

1. Open `http://localhost:3000`
2. Check that "System Online" indicator is green
3. Upload test image from training data
4. Verify:
   - Image preview displays
   - "Analyze MRI Scan" button is enabled
   - Classification completes without errors
   - Results display correctly
   - Heatmap shows properly
   - Analysis text is readable

### Integration Testing

Run the integration test suite:
```bash
cd webapp/
python test_integration.py
```

## Troubleshooting

### Backend Issues

**Problem: "Backend server is not responding"**
- Check if Flask is running: `curl http://localhost:5000/api/health`
- Check for errors in Flask terminal
- Verify port 5000 is not used by another application

**Problem: "Model failed to load"**
- Verify `MODEL_PATH` in `config.py` points to correct location
- Check if `best_model_stage1.keras` file exists
- Ensure file is ~94MB (not corrupted)

**Problem: "Hadoop connection refused"**
- Check Hadoop services: `jps`
- Start services if not running: `start-dfs.sh`
- Verify HDFS is accessible: `hdfs dfs -ls /`

**Problem: "Out of memory error"**
- Reduce Spark memory in `config.py`:
  ```python
  SPARK_DRIVER_MEMORY = '1g'
  SPARK_EXECUTOR_MEMORY = '1g'
  ```
- Close other applications
- Reduce batch size if applicable

### Frontend Issues

**Problem: "Cannot connect to backend"**
- Verify backend is running on port 5000
- Check CORS is enabled in Flask (should be by default)
- Verify `API_BASE_URL` in `api.js`

**Problem: "Image won't upload"**
- Check file type (only JPG, PNG allowed)
- Check file size (max 10MB)
- Try different browser
- Check browser console for errors (F12)

**Problem: "Heatmap not displaying"**
- Check browser console for base64 decode errors
- Verify backend response includes `visualization` object
- Try refreshing page

### Port Conflicts

**Backend Port 5000 in use:**
```python
# In backend/app.py, change port:
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

# Update frontend API_BASE_URL accordingly
```

**Frontend Port 3000 in use:**
```bash
# Frontend will prompt to use different port automatically
# Or set PORT environment variable:
PORT=3001 npm start
```

## Performance

### Expected Response Times

| Operation | Time | Notes |
|-----------|------|-------|
| Image Upload | < 1s | Local file system |
| Preprocessing | 2-3s | Spark + HDFS |
| Model Inference | 1-2s | ResNet50 forward pass |
| Grad-CAM Generation | 1-2s | Gradient computation |
| Total | 5-8s | First request may be slower |

### Optimization Tips

1. **Keep backend running**: Avoid restarting Flask (model loads once)
2. **Cache Spark session**: Already implemented in `preprocessing.py`
3. **Use smaller images**: Resize before upload if > 1MB
4. **Close unnecessary apps**: Free up RAM for Spark

## Security & Privacy

### Implemented Security Measures

- File type validation (whitelist only JPG, PNG)
- File size limits (max 10MB)
- Filename sanitization
- Input sanitization
- No persistent storage of user images
- HDFS cleanup after classification

### Privacy Guarantees

- Images are **not stored** permanently
- HDFS temporary storage is **deleted after classification**
- No user tracking or analytics
- No logging of uploaded images

### Disclaimers

This application is for **educational and demonstration purposes only**:
- NOT approved for clinical use
- NOT a substitute for professional medical diagnosis
- Model accuracy is ~92%, not 100%
- Always consult qualified healthcare professionals

## API Reference

### POST /api/classify

Classify uploaded MRI image.

**Request:**
```http
POST /api/classify
Content-Type: multipart/form-data

image: [binary file]
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class": "Glioma",
    "class_id": 0,
    "confidence": 94.5,
    "probabilities": {
      "Glioma": 94.5,
      "Meningioma": 3.2,
      "No Tumor": 1.8,
      "Pituitary": 0.5
    }
  },
  "visualization": {
    "heatmap_overlay": "base64_string...",
    "original_image": "base64_string..."
  },
  "analysis": {
    "classification": "Glioma",
    "confidence_level": "Very High",
    "description": "...",
    "detailed_info": "...",
    "characteristics": [...],
    "model_interpretation": "...",
    "educational_context": "...",
    "disclaimer": "..."
  }
}
```

### GET /api/health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "model": {"status": "up", "loaded": true},
    "spark": {"status": "up", "active": true}
  }
}
```

### GET /api/classes

Get available tumor classes.

**Response:**
```json
{
  "success": true,
  "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
}
```

## Design System

### Color Palette

- **Primary**: `#A8D5E2` (Light Blue - Calming)
- **Secondary**: `#B8E5D2` (Soft Green - Reassuring)
- **Accent**: `#D4C5E0` (Pale Lavender - Professional)
- **Background**: `#F8F9FA` (Off-White - Clean)
- **Success**: `#9BC4BC` (Sage)
- **Warning**: `#FFB6B9` (Coral)
- **Error**: `#E8A0A0` (Soft Red)

### Typography

- **Headings**: Inter, sans-serif
- **Body**: Inter, Roboto, sans-serif
- **Base Size**: 16px

## Additional Documentation

- **Backend Details**: See `backend/README.md`
- **Frontend Details**: See `frontend/README.md`
- **Model Training**: See main project documentation
- **Deployment Checklist**: See `DEPLOYMENT_CHECKLIST.md`

## License

Educational use only. Not licensed for clinical or commercial use.

## Acknowledgments

- Dataset: Brain MRI tumor classification dataset
- Model: ResNet50 (Keras Applications)
- Grad-CAM: Implementation based on original paper
- Technologies: Flask, React, TensorFlow, Spark, Hadoop

---

**Built for medical imaging education**
