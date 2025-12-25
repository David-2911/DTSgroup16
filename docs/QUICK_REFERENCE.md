# Brain MRI Classification System - Quick Reference Card
## DTS 301 - Big Data Computing | Group 16

---

## Project Overview

| Item | Description |
|------|-------------|
| **Purpose** | Classify brain MRI scans into 4 tumor types |
| **Classes** | Glioma, Meningioma, Pituitary, No Tumor |
| **Accuracy** | ~92% on test dataset |
| **Dataset** | 5,712 MRI images |
| **Model** | ResNet50 with transfer learning |

---

## Quick Start Commands

```bash
# 1. Navigate to project
cd /home/dave/Work/DTSgroup16

# 2. Activate environment
source dtsvenv/bin/activate

# 3. Start backend (Terminal 1)
cd webapp/backend && python app.py

# 4. Start frontend (Terminal 2)
cd webapp/frontend && npm start

# Or use quick start script:
cd webapp && ./start.sh
```

---

## URLs

| Service | URL |
|---------|-----|
| Frontend (Web UI) | http://localhost:3000 |
| Backend API | http://localhost:5000 |
| Health Check | http://localhost:5000/api/health |
| Classes Endpoint | http://localhost:5000/api/classes |

---

## File Structure

```
DTSgroup16/
├── Brain_MRI_Distributed_DL.ipynb  # Training notebook
├── brain_Tumor_Types/         # Dataset (137 MB)
├── models/                    # Trained models
│   └── best_model_extended.keras  # Best model (~296 MB)
├── outputs/                   # Training outputs
│
├── webapp/
│   ├── backend/               # Flask API
│   │   ├── app.py             # Main server
│   │   ├── config.py          # Configuration
│   │   ├── model_loader.py    # Model management
│   │   ├── preprocessing.py   # Spark processing
│   │   ├── gradcam.py         # Visualization
│   │   └── analysis_generator.py  # Medical text
│   │
│   └── frontend/              # React UI
│       └── src/
│           ├── App.jsx        # Main component
│           └── components/    # 7 UI components
```

---

## API Reference

### Health Check
```bash
curl http://localhost:5000/api/health
```
Response:
```json
{"status": "healthy", "model_loaded": true, "spark_active": true}
```

### Get Classes
```bash
curl http://localhost:5000/api/classes
```
Response:
```json
{"classes": ["glioma", "meningioma", "notumor", "pituitary"]}
```

### Classify Image
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/classify
```
Response: JSON with prediction, probabilities, visualization, analysis

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18, JavaScript |
| Backend | Flask 3.1, Python 3.12 |
| ML Framework | TensorFlow/Keras |
| Distributed Processing | Apache Spark |
| Distributed Storage | Apache Hadoop HDFS |

---

## Key Concepts

### Transfer Learning
- Use pre-trained ResNet50 (ImageNet)
- Freeze base layers, train only top layers
- Faster training, better accuracy with less data

### Grad-CAM
- Gradient-weighted Class Activation Mapping
- Shows which image regions influenced prediction
- Helps interpret AI decisions

### Spark Processing
- Distributed image preprocessing
- Consistent with training pipeline
- Scalable to large datasets

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Backend not available | Start Flask: `python app.py` |
| Module not found | Activate venv: `source dtsvenv/bin/activate` |
| Port 5000 in use | Kill process: `lsof -i :5000` then `kill <PID>` |
| Model not found | Check path in `config.py` |
| Spark error | Check Java: `java -version` |

---

## Testing Commands

```bash
# Run integration tests
python webapp/test_integration.py

# Verify file structure
python webapp/verify_structure.py

# Test API directly
curl http://localhost:5000/api/health
```

---

## Dataset Statistics

| Class | Count | Percentage |
|-------|-------|------------|
| Glioma | 1,426 | 25.0% |
| Meningioma | 1,339 | 23.4% |
| No Tumor | 1,595 | 27.9% |
| Pituitary | 1,352 | 23.7% |
| **Total** | **5,712** | **100%** |

---

## Model Architecture

```
Input: 224×224×3 RGB image
          ↓
ResNet50 (pre-trained, frozen)
          ↓
Global Average Pooling → 2048 features
          ↓
Dense(256, ReLU)
          ↓
Dropout(0.5)
          ↓
Dense(4, Softmax) → 4 class probabilities
```

---

## Stop Commands

```bash
# Stop servers
Ctrl+C in each terminal

# Deactivate environment
deactivate

# Kill by port
kill $(lsof -t -i:5000)
kill $(lsof -t -i:3000)
```

---

## Documentation Files

| File | Contents |
|------|----------|
| `TECHNICAL_DOCUMENTATION.md` | Full technical details |
| `SETUP_AND_USAGE_GUIDE.md` | Installation & usage |
| `CODE_WALKTHROUGH.md` | Code explanations |
| `QUICK_REFERENCE.md` | This file |
| `SUBMISSION_CLEANUP.md` | Cleanup for submission |

---

## Contact & Credits

- **Course:** DTS 301 - Big Data Computing
- **Group:** 16
- **Model:** ResNet50 with transfer learning
- **Dataset:** Karthik_BrainTypesdata_mri dataset from Kaggle (https://www.kaggle.com/datasets/skarthik112/karthik-braintypesdata-mri)

---
