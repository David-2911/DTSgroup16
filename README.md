# Brain MRI Tumor Classification with Distributed Deep Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://www.tensorflow.org/)
[![Spark](https://img.shields.io/badge/Apache%20Spark-4.0.1-red)](https://spark.apache.org/)
[![Hadoop](https://img.shields.io/badge/Hadoop-3.4.2-yellow)](https://hadoop.apache.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://react.dev/)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-green)](https://flask.palletsprojects.com/)
[![Java](https://img.shields.io/badge/Java-17-007396)](https://openjdk.org/)

> **DTS 301 - Big Data Computing | Group 16**

A complete end-to-end pipeline for classifying brain MRI scans into four tumor categories using distributed computing technologies (Apache Spark, Hadoop HDFS) and deep learning (TensorFlow/Keras with ResNet-50).

---

## 🎯 Project Overview

### The Research Question

> **"How to apply deep learning to large-scale medical imaging (e.g., MRI or histopathology) using Spark/Hadoop clusters?"**

### What This Project Does

| Component | Description |
|-----------|-------------|
| **Distributed Storage** | Stores medical images in Hadoop HDFS (fault-tolerant, scalable) |
| **Parallel Processing** | Preprocesses images using Apache Spark (scalable data processing) |
| **Deep Learning** | Trains ResNet-50 model using TensorFlow with transfer learning |
| **Web Application** | React frontend + Flask backend for real-time MRI classification |
| **Explainability** | Grad-CAM visualizations showing where the model "looks" |

### Key Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 90-94% |
| **Macro F1-Score** | 0.88-0.93 |
| **ROC-AUC** | 0.95-0.98 |
| **Classes** | Glioma, Meningioma, No Tumor, Pituitary |

---

## 📁 Project Structure

```
DTSgroup16/
├── README.md                         # This file (start here!)
├── Brain_MRI_Distributed_DL.ipynb    # Main Jupyter notebook (model training)
├── requirements.txt                  # Python dependencies
├── best_model_stage1.keras           # ⚠️ NOT in repo - generated after training
│
├── brain_Tumor_Types/                # Dataset (5,712 MRI images) ✅ Included
│   ├── glioma/                       # 1,321 images
│   ├── meningioma/                   # 1,339 images
│   ├── notumor/                      # 1,595 images
│   └── pituitary/                    # 1,457 images
│
├── webapp/                           # Web application
│   ├── backend/                      # Flask API server
│   │   ├── app.py                    # Main Flask application
│   │   ├── model_loader.py           # TensorFlow model management
│   │   ├── preprocessing.py          # Spark-based image processing
│   │   ├── gradcam.py                # Grad-CAM visualizations
│   │   └── requirements.txt          # Backend dependencies
│   ├── frontend/                     # React web interface
│   │   ├── src/                      # React components
│   │   └── package.json              # Frontend dependencies
│   ├── start.sh                      # Quick start script (Linux/Mac)
│   └── start.bat                     # Quick start script (Windows)
│
└── docs/                             # Documentation
    ├── SETUP_AND_USAGE_GUIDE.md      # ⭐ Complete setup instructions
    ├── TECHNICAL_DOCUMENTATION.md    # Architecture & concepts
    ├── CODE_WALKTHROUGH.md           # Detailed code explanations
    ├── QUICK_REFERENCE.md            # One-page command reference
    └── README.md                     # Additional project details
```

> **⚠️ Note about Model File:** The trained model (`best_model_stage1.keras`, ~105MB) is **not included** in the repository due to GitHub's file size limits. You must train the model by running the Jupyter notebook, which takes 15-25 minutes.

---

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have:

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| **Python** | 3.10+ | `python3 --version` |
| **Java** | 8, 11, or 17 | `java -version` |
| **Node.js** | 18+ | `node --version` |
| **RAM** | 8GB minimum | `free -h` |
| **Disk Space** | 20GB free | `df -h .` |

> 📖 **Don't have these?** See [docs/SETUP_AND_USAGE_GUIDE.md](docs/SETUP_AND_USAGE_GUIDE.md) for complete installation instructions.

### Option 1: Run the Web Application Only

If you just want to try the classification web app **(requires training model first)**:

```bash
# 1. Clone the repository
git clone https://github.com/David-2911/DTSgroup16.git
cd DTSgroup16

# 2. Create and activate Python virtual environment
python3 -m venv dtsvenv
source dtsvenv/bin/activate  # Linux/Mac
# or: dtsvenv\Scripts\activate  # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. ⚠️ IMPORTANT: Train the model first (15-25 mins)
# Start Hadoop: start-dfs.sh
# Then run: jupyter notebook Brain_MRI_Distributed_DL.ipynb
# Execute all cells to generate best_model_stage1.keras

# 5. Install frontend dependencies
cd webapp/frontend
npm install
cd ../..

# 6. Start the web application
cd webapp
./start.sh  # Linux/Mac
# or: start.bat  # Windows
```

Then open http://localhost:3000 in your browser.

### Option 2: Full Setup (Training + Web App)

For the complete experience including Hadoop/Spark and model training:

📖 **Follow the complete guide:** [docs/SETUP_AND_USAGE_GUIDE.md](docs/SETUP_AND_USAGE_GUIDE.md)

This covers:
- Java, Python, Node.js installation
- Hadoop HDFS setup and configuration
- Apache Spark installation
- HDFS data upload
- Running the Jupyter notebook
- Running the web application

---

## 🖥️ Running the Project

### Start Hadoop (Required for Notebook)

```bash
# Start HDFS services
start-dfs.sh

# Verify running
jps
# Should show: NameNode, DataNode, Jps
```

### Run the Jupyter Notebook

```bash
source dtsvenv/bin/activate
jupyter notebook Brain_MRI_Distributed_DL.ipynb
```

Run all cells to:
1. Load data from HDFS
2. Preprocess images with Spark
3. Train ResNet-50 model
4. Evaluate and save model

### Run the Web Application

**Terminal 1 - Backend:**
```bash
cd webapp/backend
source ../../dtsvenv/bin/activate
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd webapp/frontend
npm start
```

**Access the Application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000
- Hadoop UI: http://localhost:9870
- Spark UI: http://localhost:4040

---

## 🔬 How It Works

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                           │
│              (React Frontend @ :3000)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │ Upload MRI Image
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    FLASK BACKEND                            │
│                    (API @ :5000)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Spark      │  │  TensorFlow  │  │   Grad-CAM   │      │
│  │ Preprocessing│→ │   Model      │→ │ Visualization│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 CLASSIFICATION RESULT                        │
│  • Tumor Type (Glioma/Meningioma/No Tumor/Pituitary)        │
│  • Confidence Score (0-100%)                                 │
│  • Grad-CAM Heatmap (model attention visualization)          │
│  • Medical Analysis (AI-generated insights)                  │
└─────────────────────────────────────────────────────────────┘
```

### Model Architecture

```
Input Image (224 × 224 × 3)
          │
          ▼
┌─────────────────────────────────────┐
│         ResNet-50 Base              │
│    (Pre-trained on ImageNet)        │
│      23.5M parameters               │
└─────────────────┬───────────────────┘
                  │
                  ▼
    GlobalAveragePooling2D
          │
          ▼
    Dense(256, ReLU) + BatchNorm + Dropout(0.5)
          │
          ▼
    Dense(128, ReLU) + Dropout(0.3)
          │
          ▼
    Dense(4, Softmax)
          │
          ▼
    [Glioma, Meningioma, No Tumor, Pituitary]
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/skarthik112/karthik-braintypesdata-mri) |
| **Total Images** | 5,712 |
| **Image Size** | Various (resized to 224×224) |
| **Format** | JPEG |

| Class | Count | Percentage |
|-------|-------|------------|
| Glioma | 1,321 | 23.1% |
| Meningioma | 1,339 | 23.4% |
| No Tumor | 1,595 | 27.9% |
| Pituitary | 1,457 | 25.5% |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [SETUP_AND_USAGE_GUIDE.md](docs/SETUP_AND_USAGE_GUIDE.md) | ⭐ **Start here** - Complete installation & setup |
| [TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) | System architecture & distributed computing concepts |
| [CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md) | Detailed code explanations |
| [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | One-page command reference card |
| [OUTPUTS.md](docs/OUTPUTS.md) | Expected outputs from notebook cells |

---

## 🔧 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: pyspark` | Activate venv: `source dtsvenv/bin/activate` |
| `Connection refused` to HDFS | Start Hadoop: `start-dfs.sh` |
| Port 5000 already in use | Kill process: `kill $(lsof -t -i:5000)` |
| Port 3000 already in use | Kill process: `kill $(lsof -t -i:3000)` |
| Model file not found | Train the model first by running the Jupyter notebook |
| `MemoryError` during training | Reduce batch size to 8 in notebook |

### Verification Commands

```bash
# Check services
jps                              # Should show NameNode, DataNode
curl http://localhost:5000/api/health  # Should return healthy

# Check ports
lsof -i :5000 -i :3000 -i :9000

# Run tests
python webapp/test_integration.py
python webapp/verify_structure.py
```

---

## 🛑 Stopping Services

```bash
# Stop web application (Ctrl+C in each terminal)

# Stop Hadoop
stop-dfs.sh

# Deactivate Python environment
deactivate
```

---

## 👥 Team

**Group 16** - Distributed Technologies and Systems (DTS 301)

---

## 📄 License

This project is for educational purposes as part of academic coursework at the university level.

---

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for the Brain Tumor MRI Dataset
- [Apache Spark](https://spark.apache.org/) for distributed computing
- [Apache Hadoop](https://hadoop.apache.org/) for distributed storage
- [TensorFlow](https://www.tensorflow.org/) for deep learning
- [React](https://react.dev/) for the frontend framework
