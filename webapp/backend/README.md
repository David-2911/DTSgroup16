# Brain Tumor Classification Backend

Flask-based REST API for brain tumor classification using the trained ResNet50 model.

## Overview

This backend provides a REST API for classifying brain MRI images into four categories:
- **Glioma** - Tumor arising from glial cells
- **Meningioma** - Tumor arising from meninges
- **No Tumor** - No tumor detected
- **Pituitary** - Tumor in the pituitary gland

The backend uses:
- **Flask** for the REST API
- **TensorFlow/Keras** for model inference
- **Apache Spark** for image preprocessing (consistency with training pipeline)

## Directory Structure

```
webapp/backend/
├── app.py              # Main Flask application
├── config.py           # Configuration settings
├── model_loader.py     # Model loading and inference
├── preprocessing.py    # Spark-based image preprocessing
├── gradcam.py          # Grad-CAM visualization module
├── utils.py            # Helper utilities
├── requirements.txt    # Python dependencies
├── test_gradcam.py     # Grad-CAM test script
├── README.md           # This file
├── temp_uploads/       # Temporary file storage
└── test_outputs/       # Test output images (created by tests)
```

## Quick Start

### Prerequisites

1. **Python 3.10+** with the project virtual environment activated
2. **Trained model** at `../../best_model_stage1.keras`
3. **HDFS running** (if using HDFS mode) - optional for local development

### Installation

```bash
# Navigate to backend directory
cd webapp/backend

# Install dependencies (if not using main project venv)
pip install -r requirements.txt
```

### Running the Server

```bash
# From the backend directory
python app.py

# Or from project root
python webapp/backend/app.py
```

The server will start at `http://localhost:5000`.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `development` | Environment mode |
| `FLASK_DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level |

## API Endpoints

### 1. POST `/api/classify`

Classify a brain MRI image with Grad-CAM visualization.

**Request:**
```bash
curl -X POST \
  http://localhost:5000/api/classify \
  -F "file=@brain_scan.jpg"
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
    "heatmap_overlay": "base64_encoded_image...",
    "original_image": "base64_encoded_image...",
    "heatmap_only": "base64_encoded_image..."
  },
  "processing_time_ms": 1523.45,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "message": "File type not allowed. Allowed: png, jpg, jpeg, gif, bmp, webp",
    "code": "INVALID_FILE_TYPE",
    "status": 415
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### 2. GET `/api/health`

Health check endpoint.

**Request:**
```bash
curl http://localhost:5000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "services": {
    "model": {
      "status": "up",
      "loaded": true
    },
    "spark": {
      "status": "up",
      "active": true
    }
  },
  "details": {
    "model_path": "/path/to/best_model_stage1.keras",
    "num_classes": 4,
    "image_size": [224, 224],
    "spark_version": "3.5.0"
  }
}
```

### 3. GET `/api/classes`

Get available classification classes.

**Request:**
```bash
curl http://localhost:5000/api/classes
```

**Response:**
```json
{
  "success": true,
  "classes": [
    {
      "id": 0,
      "name": "Glioma",
      "description": "A type of tumor that occurs in the brain and spinal cord..."
    },
    {
      "id": 1,
      "name": "Meningioma",
      "description": "A tumor that arises from the meninges..."
    },
    {
      "id": 2,
      "name": "No Tumor",
      "description": "No tumor detected in the MRI scan."
    },
    {
      "id": 3,
      "name": "Pituitary",
      "description": "A tumor that forms in the pituitary gland..."
    }
  ],
  "total_classes": 4
}
```

### 4. GET `/`

API information.

**Request:**
```bash
curl http://localhost:5000/
```

**Response:**
```json
{
  "name": "Brain Tumor Classification API",
  "version": "1.0.0",
  "endpoints": {
    "/api/classify": {
      "method": "POST",
      "description": "Upload and classify a brain MRI image"
    },
    "/api/health": {
      "method": "GET",
      "description": "Health check endpoint"
    },
    "/api/classes": {
      "method": "GET",
      "description": "Get available classification classes"
    }
  }
}
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `NO_FILE` | 400 | No file provided in request |
| `EMPTY_FILENAME` | 400 | Empty filename |
| `INVALID_FILE_TYPE` | 415 | File extension not allowed |
| `INVALID_IMAGE` | 400 | Image validation failed |
| `FILE_TOO_LARGE` | 413 | File exceeds 16MB limit |
| `CLASSIFICATION_ERROR` | 500 | Model inference failed |
| `NOT_FOUND` | 404 | Endpoint not found |
| `INTERNAL_ERROR` | 500 | Internal server error |

## Supported Image Formats

- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)
- GIF (`.gif`)
- BMP (`.bmp`)
- WebP (`.webp`)

**Maximum file size:** 16MB

## Testing

### Test with curl

```bash
# Health check
curl http://localhost:5000/api/health

# Classify an image
curl -X POST \
  http://localhost:5000/api/classify \
  -F "file=@/path/to/brain_mri.jpg"

# Get classes
curl http://localhost:5000/api/classes
```

### Test with Python

```python
import requests

# Health check
response = requests.get("http://localhost:5000/api/health")
print(response.json())

# Classify image
with open("brain_mri.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:5000/api/classify",
        files={"file": f}
    )
print(response.json())
```

### Test individual modules

```bash
# Test configuration
python config.py

# Test model loader
python model_loader.py

# Test preprocessing
python preprocessing.py

# Test utilities
python utils.py
```

## Configuration

All configuration is in `config.py`. Key settings:

### Model Configuration
```python
MODEL_PATH = "../../best_model_stage1.keras"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}
```

### Flask Configuration
```python
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
CORS_ORIGINS = ["http://localhost:3000"]
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
```

### Spark Configuration
```python
SPARK_APP_NAME = "BrainTumorClassification-Backend"
SPARK_MASTER = "local[*]"
SPARK_EXECUTOR_MEMORY = "2g"
SPARK_DRIVER_MEMORY = "2g"
```

## Grad-CAM Visualization

The backend generates Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps 
showing which regions of the MRI the model focuses on during classification.

### How Grad-CAM Works

1. **Forward pass**: Image goes through the model to get predictions
2. **Gradient computation**: Calculate gradients of predicted class w.r.t. last conv layer
3. **Importance weighting**: Weight each feature map by its gradient importance
4. **Heatmap generation**: Average weighted feature maps to create attention heatmap
5. **Overlay**: Apply colormap and blend with original image

### Visualization Outputs

The API returns three base64-encoded images:

| Key | Description |
|-----|-------------|
| `heatmap_overlay` | Original image with colored heatmap overlay (alpha=0.4) |
| `original_image` | Original uploaded image |
| `heatmap_only` | Jet-colored heatmap showing attention intensity |

### Interpreting Heatmaps

**Color Interpretation (Jet colormap):**
- **Red/Hot**: High activation - model strongly focuses here
- **Yellow**: Moderate-high activation
- **Green**: Moderate activation  
- **Blue/Cool**: Low activation - model ignores this area

**Quality Indicators:**
- Highlights tumor regions (for tumor classes)
- Focuses on brain anatomy, not edges/artifacts
- Relatively smooth, not extremely noisy
- If highlighting corners/edges, may indicate issues

### Testing Grad-CAM

```bash
# Test module directly
python test_gradcam.py --module

# Test via API (requires server running)
python test_gradcam.py --api

# Test with specific image
python test_gradcam.py /path/to/brain_mri.jpg

# Run all tests
python test_gradcam.py --all
```

### Decoding Visualization in Frontend

```javascript
// Decode base64 to display in <img> tag
const response = await fetch('/api/classify', {
  method: 'POST',
  body: formData
});
const result = await response.json();

// Display heatmap overlay
const heatmapSrc = `data:image/png;base64,${result.visualization.heatmap_overlay}`;
document.getElementById('heatmap').src = heatmapSrc;
```

### Performance Notes

Grad-CAM adds ~1-2 seconds to processing time:
- Gradient computation: ~0.5-1 second
- Heatmap generation: ~0.2-0.5 seconds
- Image encoding: ~0.1-0.2 seconds

Total expected response time: 1.5-3 seconds

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│                   (React Application)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP/REST
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Flask Backend                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   app.py    │  │ utils.py     │  │    config.py     │   │
│  │  (Routes)   │  │ (Helpers)    │  │  (Settings)      │   │
│  └──────┬──────┘  └──────────────┘  └──────────────────┘   │
│         │                                                    │
│  ┌──────▼──────┐  ┌──────────────┐                         │
│  │preprocessing│  │model_loader  │                         │
│  │    .py      │  │    .py       │                         │
│  │  (Spark)    │  │  (Keras)     │                         │
│  └──────┬──────┘  └──────┬───────┘                         │
└─────────┼────────────────┼──────────────────────────────────┘
          │                │
          ▼                ▼
┌─────────────────┐  ┌─────────────────┐
│  Apache Spark   │  │   TensorFlow    │
│  (Preprocess)   │  │   (Inference)   │
└─────────────────┘  └─────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Keras Model    │
                   │ (ResNet50 .keras)│
                   └─────────────────┘
```

## Troubleshooting

### Common Issues

**1. Model not found**
```
FileNotFoundError: Model file not found: /path/to/model.keras
```
- Ensure `best_model_stage1.keras` exists in the project root
- Check the `MODEL_PATH` in `config.py`

**2. Spark initialization fails**
```
Exception: Failed to create Spark session
```
- Ensure Java 8 or 11 is installed
- Check `JAVA_HOME` environment variable
- Verify Spark installation

**3. Port already in use**
```
OSError: [Errno 98] Address already in use
```
- Another process is using port 5000
- Change `FLASK_PORT` in `config.py` or kill the other process

**4. CORS errors in browser**
```
Access to fetch at 'http://localhost:5000' has been blocked by CORS policy
```
- Add your frontend URL to `CORS_ORIGINS` in `config.py`

### Debug Mode

Enable debug mode for detailed error messages:
```bash
FLASK_DEBUG=true python app.py
```

### Logging

Increase log verbosity:
```bash
LOG_LEVEL=DEBUG python app.py
```

## Production Deployment

For production, use a proper WSGI server:

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Production checklist:**
- [ ] Set `FLASK_DEBUG=false`
- [ ] Set `FLASK_ENV=production`
- [ ] Use HTTPS (reverse proxy)
- [ ] Configure proper CORS origins
- [ ] Set up logging to file
- [ ] Monitor with health endpoint
- [ ] Use process manager (systemd, supervisor)

## Next Steps

- **Part 2:** Add Grad-CAM visualization endpoint
- **Part 3:** Frontend development (React)
- **Part 4:** Integration and deployment
