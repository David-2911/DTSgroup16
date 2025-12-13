# Submission Cleanup Guide - DTS 301 Big Data Computing

## Directory Size Analysis

| Directory/File | Size | Action | Reason |
|---------------|------|--------|--------|
| **dtsvenv/** | 3.3 GB | REMOVE | Virtual environment - reinstallable via requirements.txt |
| **webapp/frontend/node_modules/** | 418 MB | REMOVE | Node modules - reinstallable via npm install |
| **.git/** | ~10 MB | REMOVE | Git history not needed for submission |
| **webapp/backend/__pycache__/** | 120 KB | REMOVE | Python bytecode cache - auto-generated |
| **webapp/backend/server.log** | 11 KB | REMOVE | Runtime log file |
| **webapp/backend/temp_uploads/** | Empty | KEEP | Required directory structure (keep empty) |
| **best_model_stage1.keras** | 105 MB | KEEP | Trained model - essential |
| **brain_Tumor_Types/** | 137 MB | KEEP | Dataset - essential |
| **webapp/frontend/src/** | 116 KB | KEEP | React source code |
| **webapp/backend/*.py** | 150 KB | KEEP | Flask backend code |
| **docs/** | 96 KB | KEEP | Documentation |
| **All .md, .txt, .json, .ipynb files** | ~300 KB | KEEP | Documentation and config |

## Current Total: 4.1 GB
## After Cleanup: ~350 MB (with dataset and model)

---

## Cleanup Commands

### Step 1: Remove Virtual Environment
```bash
cd /home/dave/Work/DTSgroup16
rm -rf dtsvenv/
```

### Step 2: Remove Node Modules
```bash
rm -rf webapp/frontend/node_modules/
```

### Step 3: Remove Git History
```bash
rm -rf .git/
rm .gitignore
```

### Step 4: Remove Python Cache and Logs
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
rm -f webapp/backend/server.log
rm -f webapp/backend/nohup.out 2>/dev/null
```

### Step 5: Clear temp but keep directory
```bash
rm -f webapp/backend/temp_uploads/* 2>/dev/null
touch webapp/backend/temp_uploads/.gitkeep
```

### All-in-One Cleanup Script
```bash
#!/bin/bash
# cleanup_for_submission.sh
cd /home/dave/Work/DTSgroup16

echo "Removing virtual environment (3.3 GB)..."
rm -rf dtsvenv/

echo "Removing node_modules (418 MB)..."
rm -rf webapp/frontend/node_modules/

echo "Removing git history..."
rm -rf .git/
rm -f .gitignore

echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

echo "Removing log files..."
rm -f webapp/backend/server.log
rm -f webapp/backend/nohup.out 2>/dev/null

echo "Clearing temp uploads..."
rm -f webapp/backend/temp_uploads/* 2>/dev/null

echo "Cleanup complete!"
du -sh .
```

---

## Final Directory Structure After Cleanup

```
DTSgroup16/                          (~350 MB total)
|
├── best_model_stage1.keras          (105 MB) - Trained ResNet50 model
├── Brain_MRI_Distributed_DL.ipynb   (167 KB) - Jupyter notebook with training
├── requirements.txt                  (5 KB)  - Python dependencies
├── README.md                         (17 KB) - Main project documentation
├── SETUP.md                          (26 KB) - Setup instructions
├── OUTPUTS.md                        (33 KB) - Output documentation
├── SUBMISSION_CLEANUP.md             (this file)
│
├── brain_Tumor_Types/                (137 MB) - MRI Dataset
│   ├── glioma/                       - Glioma tumor images
│   ├── meningioma/                   - Meningioma tumor images
│   ├── notumor/                      - Healthy brain images
│   └── pituitary/                    - Pituitary tumor images
│
├── docs/                             (96 KB) - Additional documentation
│   ├── ARCHITECTURE.md
│   ├── CHECKLIST.md
│   ├── CONCEPTS.md
│   └── REPORT_TEMPLATE.md
│
└── webapp/                           (500 KB) - Web Application
    ├── README.md                     - Webapp documentation
    ├── PROJECT_SUMMARY.md            - Project summary
    ├── DEPLOYMENT_CHECKLIST.md       - Deployment guide
    ├── test_integration.py           - Integration tests
    ├── verify_structure.py           - Structure verification
    ├── start.sh / start.bat          - Quick start scripts
    │
    ├── backend/                      - Flask API Server
    │   ├── app.py                    - Main Flask application
    │   ├── model_loader.py           - TensorFlow model management
    │   ├── preprocessing.py          - Spark image preprocessing
    │   ├── gradcam.py                - Grad-CAM visualization
    │   ├── analysis_generator.py     - Medical analysis generation
    │   ├── utils.py                  - Utility functions
    │   ├── config.py                 - Configuration settings
    │   ├── requirements.txt          - Backend dependencies
    │   ├── README.md                 - Backend documentation
    │   ├── test_gradcam.py           - Grad-CAM tests
    │   └── temp_uploads/             - Temporary upload directory
    │
    ├── frontend/                     - React Frontend
    │   ├── package.json              - Node.js dependencies
    │   ├── README.md                 - Frontend documentation
    │   ├── public/                   - Static assets
    │   │   ├── index.html
    │   │   ├── manifest.json
    │   │   └── robots.txt
    │   └── src/                      - React source code
    │       ├── App.jsx               - Main React component
    │       ├── index.js              - Entry point
    │       ├── index.css             - Global styles
    │       ├── components/           - React components
    │       │   ├── Header.jsx
    │       │   ├── ImageUpload.jsx
    │       │   ├── LoadingSpinner.jsx
    │       │   ├── ClassificationResult.jsx
    │       │   ├── HeatmapVisualization.jsx
    │       │   ├── MedicalAnalysis.jsx
    │       │   └── ErrorMessage.jsx
    │       ├── styles/               - Component styles
    │       │   ├── Header.css
    │       │   ├── ImageUpload.css
    │       │   ├── LoadingSpinner.css
    │       │   ├── ClassificationResult.css
    │       │   ├── HeatmapVisualization.css
    │       │   ├── MedicalAnalysis.css
    │       │   ├── ErrorMessage.css
    │       │   └── Results.css
    │       └── utils/
    │           └── api.js            - API communication
    │
    └── shared/
        └── tumor_descriptions.json   - Medical descriptions
```

---

## What Gets Included in ZIP Submission

| Category | Approximate Size |
|----------|-----------------|
| Trained Model (best_model_stage1.keras) | 105 MB |
| Dataset (brain_Tumor_Types/) | 137 MB |
| Jupyter Notebook | 167 KB |
| Documentation (.md files) | 200 KB |
| Backend Source Code | 150 KB |
| Frontend Source Code | 160 KB |
| Configuration Files | 20 KB |
| **TOTAL** | **~245 MB** |

---

## Verification After Cleanup

Run this command to verify cleanup was successful:
```bash
cd /home/dave/Work/DTSgroup16
python webapp/verify_structure.py
```

Output: All 34 required files present.

---

## Creating the ZIP File

```bash
cd /home/dave/Work
zip -r DTSgroup16_submission.zip DTSgroup16/
```

Or with progress indicator:
```bash
cd /home/dave/Work
zip -rv DTSgroup16_submission.zip DTSgroup16/ | tail -20
```
