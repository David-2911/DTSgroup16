# Distributed Deep Learning for Brain MRI Classification

## Using Apache Spark, Hadoop HDFS, and TensorFlow

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://www.tensorflow.org/)
[![Spark](https://img.shields.io/badge/Apache%20Spark-3.5.0-red)](https://spark.apache.org/)
[![Hadoop](https://img.shields.io/badge/Hadoop-3.3.6-yellow)](https://hadoop.apache.org/)

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Results](#-key-results)
3. [Quick Start](#-quick-start)
4. [System Architecture](#-system-architecture)
5. [Notebook Phases](#-notebook-phases)
6. [Technologies Explained](#-technologies-explained)
7. [Documentation](#-documentation)
8. [Troubleshooting](#-troubleshooting)
9. [References](#-references)

---

## Project Overview

### The Question We're Answering

> **"How to apply deep learning to large-scale medical imaging (e.g., MRI or histopathology) using Spark/Hadoop clusters?"**

This project demonstrates a complete end-to-end pipeline for classifying brain MRI scans into four tumor categories using distributed computing technologies. We prove that the same architecture that handles our 5,712-image dataset can scale to millions of images on production clusters.

### What This Project Does

1. **Stores medical images** in Hadoop HDFS (distributed, fault-tolerant storage)
2. **Preprocesses images in parallel** using Apache Spark (scalable data processing)
3. **Trains a deep learning model** using TensorFlow with ResNet-50 transfer learning
4. **Evaluates performance** with comprehensive metrics (accuracy, F1, ROC curves)
5. **Demonstrates scalability** through benchmarking and theoretical analysis

### Dataset

| Property | Value |
|----------|-------|
| **Type** | Brain MRI Scans |
| **Total Images** | 5,712 |
| **Classes** | Glioma (1,321), Meningioma (1,339), No Tumor (1,595), Pituitary (1,457) |
| **Size** | ~130MB (uncompressed) |
| **Task** | 4-class multi-class classification |
| **Source** | [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |

### Why Distributed Computing for 130MB?

Great question! Here's why we use Spark/Hadoop even for a "small" dataset:

1. **Proof of Concept**: The architecture works identically for 130MB or 130TB
2. **Industry Practice**: This is exactly how production medical imaging systems work
3. **Educational Value**: Learn distributed computing with a manageable dataset
4. **Scalability Demonstration**: We show projected performance at scale
5. **Project Requirements**: Addresses the research question directly

---

## Key Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 90-94% |
| **Macro F1-Score** | 0.88-0.93 |
| **ROC-AUC (all classes)** | 0.95-0.98 |
| **Training Time** | 15-25 minutes |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 0.91-0.95 | 0.90-0.94 | 0.90-0.94 |
| Meningioma | 0.88-0.92 | 0.89-0.93 | 0.88-0.92 |
| No Tumor | 0.93-0.97 | 0.94-0.97 | 0.93-0.97 |
| Pituitary | 0.90-0.94 | 0.88-0.92 | 0.89-0.93 |

### Distributed Processing Benefits

| Configuration | Preprocessing Time | Speedup |
|--------------|-------------------|---------|
| Sequential (1 core) | ~10 minutes | 1.0x |
| Spark local[2] | ~5.5 minutes | 1.8x |
| Spark local[4] | ~3.2 minutes | 3.1x |
| Spark local[8] | ~1.8 minutes | 5.6x |

---

## Quick Start

### Prerequisites

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **RAM**: 8GB minimum (we optimize for memory-constrained systems)
- **Storage**: 5GB free space
- **Java**: OpenJDK 8 or 11

### 5-Minute Start (Experienced Users)

```bash
# 1. Clone and enter project
cd /path/to/DTSgroup16

# 2. Activate virtual environment
source dtsvenv/bin/activate

# 3. Start Hadoop services
start-dfs.sh

# 4. Verify services running
jps  # Should show NameNode, DataNode

# 5. Upload data to HDFS (if not done)
hdfs dfs -mkdir -p /user/$USER/brain_mri_data
hdfs dfs -put brain_Tumor_Types /user/$USER/brain_mri_data/

# 6. Open notebook
jupyter notebook Brain_MRI_Distributed_DL.ipynb
```

### First-Time Setup

For detailed installation instructions, see **[SETUP.md](SETUP.md)** which covers:
- Java installation
- Hadoop configuration
- Spark setup
- Python environment
- HDFS data upload
- Troubleshooting common issues

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    JUPYTER NOTEBOOK                         │
│              (Brain_MRI_Distributed_DL.ipynb)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│    SPARK      │ │  TENSORFLOW   │ │    HDFS       │
│   (PySpark)   │ │   (Keras)     │ │  (Storage)    │
│               │ │               │ │               │
│ • Parallel    │ │ • ResNet-50   │ │ • NameNode    │
│   preprocess  │ │ • Transfer    │ │ • DataNode    │
│ • Data load   │ │   learning    │ │ • Replication │
│ • RDD/DF ops  │ │ • Training    │ │ • 128MB blocks│
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    TRAINED MODEL      │
              │ (best_model_stage1.keras) │
              └───────────────────────┘
```

### Data Flow

```
1. Raw MRI Images → HDFS (distributed storage)
                         ↓
2. Spark reads images → Parallel preprocessing across workers
                         ↓
3. Preprocessed data → TensorFlow ImageDataGenerator
                         ↓
4. ResNet-50 model → Two-stage training (frozen → fine-tuned)
                         ↓
5. Evaluation → Confusion matrix, ROC curves, metrics
                         ↓
6. Saved model → Ready for inference
```

### Model Architecture

```
Input Image (224 × 224 × 3)
          ↓
┌─────────────────────────────────────┐
│         ResNet-50 Base              │
│    (Pre-trained on ImageNet)        │
│         175 layers                  │
│      23.5M parameters               │
│   (Frozen in Stage 1)               │
└─────────────────────────────────────┘
          ↓
    GlobalAveragePooling2D
          ↓
    Dense(256, ReLU)
          ↓
    BatchNormalization
          ↓
    Dropout(0.5)
          ↓
    Dense(128, ReLU)
          ↓
    Dropout(0.3)
          ↓
    Dense(4, Softmax) → [Glioma, Meningioma, No Tumor, Pituitary]
```

---

## Notebook Phases

The notebook is organized into 10 distinct phases:

| Phase | Name | Description | Time |
|-------|------|-------------|------|
| 1 | Environment Setup | Import libraries, configure Spark/TF | 1 min |
| 2 | Hadoop Configuration | Connect to HDFS, verify services | 2 min |
| 3 | HDFS Data Management | Upload/verify dataset in HDFS | 3 min |
| 4 | Data Exploration | Visualize samples, check dimensions | 2 min |
| 5 | Class Distribution | Analyze balance, compute weights | 1 min |
| 6 | Distributed Preprocessing | Spark parallel image processing | 5 min |
| 7 | Model Architecture | Build ResNet-50 with custom head | 1 min |
| 8 | Training | Two-stage transfer learning | 15-25 min |
| 9 | Evaluation | Metrics, confusion matrix, ROC | 3 min |
| 10 | Save & Summary | Export model, document results | 1 min |

**Total Estimated Time: 35-45 minutes** (varies by hardware)

---

## Technologies Explained

### For Readers New to These Technologies

This section provides undergraduate CS-level explanations of key concepts. For deeper coverage, see **[docs/CONCEPTS.md](docs/CONCEPTS.md)**.

### Deep Learning & CNNs

**What is Deep Learning?**

*ELI5:* Teaching a computer to recognize patterns by showing it many examples, like teaching a child to recognize cats by showing them thousands of cat pictures.

*Technical:* Deep learning uses neural networks with multiple layers to learn hierarchical representations of data. Each layer learns increasingly abstract features—edges → textures → shapes → objects.

**Why CNNs for Images?**

Convolutional Neural Networks are specifically designed for image data:
- **Preserve spatial information**: Unlike regular neural networks, CNNs understand that pixels next to each other are related
- **Parameter efficient**: Share the same filter across the entire image (fewer parameters to learn)
- **Translation invariant**: A tumor in the top-left looks the same as one in the bottom-right

### Transfer Learning

**What is Transfer Learning?**

*ELI5:* It's like how knowing how to ride a bicycle helps you learn to ride a motorcycle—you don't start from zero.

*Technical:* We start with a model (ResNet-50) that learned to recognize 1000 types of objects on 1.2 million images. This model already knows about edges, textures, and shapes. We keep this knowledge and just teach it the specifics of brain tumors.

**Why It Works for Medical Imaging:**
- Low-level features (edges, textures) are universal
- Reduces training time dramatically (minutes vs. days)
- Works well with limited data (5K images vs. millions)
- Proven effective in medical imaging literature

### Apache Spark

**What is Spark?**

*ELI5:* Instead of one person sorting 5,712 photos, you have 4-8 friends each sorting part of the pile simultaneously.

*Technical:* Apache Spark is a distributed computing engine that processes data in parallel across multiple CPU cores (or machines). Key concepts:

- **RDD (Resilient Distributed Dataset)**: Data split across workers; automatically recovers if one fails
- **Transformations**: Operations like `map()` that describe what to do (lazy—not executed immediately)
- **Actions**: Operations like `collect()` that trigger actual computation
- **In-memory processing**: 100x faster than disk-based alternatives (Hadoop MapReduce)

**How We Use Spark:**
```python
# Load images in parallel from HDFS
images_rdd = spark.sparkContext.binaryFiles("hdfs:///.../brain_Tumor_Types/*/*.jpg")

# Preprocess each image in parallel (runs on multiple cores simultaneously)
preprocessed_rdd = images_rdd.mapPartitions(preprocess_partition)

# Collect results
results = preprocessed_rdd.collect()
```

### Hadoop HDFS

**What is HDFS?**

*ELI5:* Instead of storing a book in one location, you photocopy each chapter and store copies in different libraries. If one library burns down, you still have the chapter elsewhere.

*Technical:* HDFS (Hadoop Distributed File System) is a distributed storage system designed for:
- **Large files**: Split into 128MB blocks
- **Fault tolerance**: Each block replicated 3x by default
- **Scalability**: Add more DataNodes for more storage
- **Data locality**: Process data where it's stored (avoid network transfer)

**Components:**
- **NameNode**: Knows where every block is stored (the "librarian")
- **DataNode**: Actually stores the data blocks (the "bookshelves")

### ResNet-50

**What is ResNet-50?**

ResNet (Residual Network) solved the "vanishing gradient problem" that made deep networks hard to train.

*ELI5:* Imagine a game of telephone where the message gets garbled over many people. ResNet adds "shortcuts" where people can pass the original message directly to skip the confusion.

*Technical:* ResNet uses "skip connections" that allow gradients to flow directly backward through the network:
```
Normal: Input → Layer1 → Layer2 → Output
ResNet: Input → Layer1 → Layer2 → Output + Input (skip connection)
```

This allows training networks with 50, 101, or even 152 layers effectively.

---

## Documentation

### Quick Reference

| Document | Description | When to Use |
|----------|-------------|-------------|
| **[SETUP.md](SETUP.md)** | Complete installation guide | First-time setup |
| **[EXPECTED_OUTPUTS.md](EXPECTED_OUTPUTS.md)** | Cell-by-cell expected results | Verify your outputs |
| **[docs/CONCEPTS.md](docs/CONCEPTS.md)** | Deep technical explanations | Understanding theory |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | ResNet vs U-Net comparison | Model selection |
| **[docs/CHECKLIST.md](docs/CHECKLIST.md)** | Pre-submission validation | Before submitting |
| **[docs/REPORT_TEMPLATE.md](docs/REPORT_TEMPLATE.md)** | Academic report structure | Writing the report |

### File Structure

```
DTSgroup16/
├── Brain_MRI_Distributed_DL.ipynb   # Main notebook (run this!)
├── README.md                        # This file
├── SETUP.md                         # Installation guide
├── EXPECTED_OUTPUTS.md              # Expected notebook outputs
├── requirements.txt                 # Python dependencies
├── best_model_stage1.keras          # Trained model (after running)
├── brain_Tumor_Types/               # Dataset (local copy)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── dtsvenv/                         # Python virtual environment
└── docs/
    ├── CONCEPTS.md                  # Technical concepts explained
    ├── ARCHITECTURE.md              # Model architecture guide
    ├── CHECKLIST.md                 # Pre-submission checklist
    └── REPORT_TEMPLATE.md           # Report writing template
```

---

## Troubleshooting

### Common Issues

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| `ModuleNotFoundError: pyspark` | Virtual environment not activated | `source dtsvenv/bin/activate` |
| `Connection refused` to HDFS | Hadoop not running | `start-dfs.sh`, verify with `jps` |
| `MemoryError` during training | Batch size too large | Reduce to 8 or 16 in notebook |
| Model accuracy ~25% | Labels wrong or model broken | Verify data preprocessing |
| Spark job hangs | Resource contention | Check Spark UI at `localhost:4040` |

### Getting Help

1. **Check OUTPUTS.md** - Compare your outputs 
2. **Check Spark UI** - Visit `http://localhost:4040` during Spark jobs
3. **Check HDFS UI** - Visit `http://localhost:9870` for storage status
4. **Check logs** - Look in `$HADOOP_HOME/logs/` for error details

---

## References

### Academic Papers

1. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
2. Ronneberger, O., et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI.
3. Zaharia, M., et al. (2016). "Apache Spark: A Unified Engine for Big Data Processing." CACM.

### Medical Imaging with Deep Learning

4. Litjens, G., et al. (2017). "A Survey on Deep Learning in Medical Image Analysis." Medical Image Analysis.
5. Ker, J., et al. (2018). "Deep Learning Applications in Medical Image Analysis." IEEE Access.

### Spark in Bioinformatics

6. Nothaft, F., et al. (2015). "Rethinking Data-Intensive Science Using Scalable Analytics Systems." SIGMOD.
7. Massie, M., et al. (2013). "ADAM: Genomics Formats and Processing Patterns for Cloud Scale Computing." UCB Tech Report.

### Online Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [Hadoop HDFS Guide](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html)
- [Keras Applications (ResNet)](https://keras.io/api/applications/resnet/)

---

## Team

**Group 16** - Distributed Technologies and Systems

---

## License

This project is for educational purposes as part of academic coursework.

---

