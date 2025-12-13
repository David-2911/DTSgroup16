#  Outputs Guide

## Complete Cell-by-Cell Reference for Brain_MRI_Distributed_DL.ipynb

---

## Table of Contents

1. [Introduction](#introduction)
2. [Notebook Overview](#notebook-overview)
3. [Phase 1: Environment Setup](#phase-1-environment-setup)
4. [Phase 2: Hadoop Configuration](#phase-2-hadoop-configuration)
5. [Phase 3: HDFS Data Management](#phase-3-hdfs-data-management)
6. [Phase 4: Data Exploration](#phase-4-data-exploration)
7. [Phase 5: Class Distribution](#phase-5-class-distribution)
8. [Phase 6: Distributed Preprocessing](#phase-6-distributed-preprocessing)
9. [Phase 7: Model Architecture](#phase-7-model-architecture)
10. [Phase 8: Training](#phase-8-training)
11. [Phase 9: Evaluation](#phase-9-evaluation)
12. [Phase 10: Save & Summary](#phase-10-save--summary)
13. [Final Results Summary](#final-results-summary)
14. [Troubleshooting Discrepancies](#troubleshooting-discrepancies)

---

## Introduction

This document provides detailed  outputs for each cell in the `Brain_MRI_Distributed_DL.ipynb` notebook. Use this to verify your setup is working correctly and identify issues early.

### How to Use This Guide

1. **Run each notebook cell sequentially** (don't skip cells)
2. **Compare your output** with the  output below
3. **Minor variations are normal** (exact numbers may differ slightly)
4. **Major discrepancies indicate problems** (see Troubleshooting section)

---

## Notebook Overview

| Property | Value |
|----------|-------|
| **Total Cells** | 30 (code + markdown) |
| **Estimated Runtime** | 35-45 minutes |
| **Memory Required** | 4-6 GB RAM |
| **Key Checkpoint** | Cell 17 (Training starts) |

### Critical Checkpoints

| Cell | Phase | What to Verify |
|------|-------|----------------|
| 2 | 1 | TensorFlow/Spark versions correct |
| 7 | 3 | HDFS connection working |
| 15 | 6 | Spark preprocessing completes |
| 17 | 8 | Training starts, loss decreases |
| 23 | 9 | Accuracy > 85% |

---

## Phase 1: Environment Setup

### Cell 1: Title (Markdown)

**What it shows:** Project title and description

```
No output (markdown cell)
```

---

### Cell 2: Import Libraries

**What it does:** Imports all required Python libraries

** Output:**
```
   ============================================================
   DISTRIBUTED DEEP LEARNING FOR MEDICAL IMAGING
   Brain MRI Tumor Classification using Spark and Hadoop
   ============================================================
   
   Environment Setup:
   ✓ TensorFlow version: 2.15.0
   ✓ PySpark version: 3.5.0
   ✓ NumPy version: 1.26.x
   ✓ PIL (Pillow) available
   ✓ Matplotlib available
   ✓ Scikit-learn available
```

**Possible Warnings (OK to ignore):**
```
 "Could not load dynamic library 'libcudart.so.11.0'"
   → This is fine if running on CPU (no GPU)

 "TensorFlow binary is optimized to use available CPU instructions..."
   → Informational only, can be ignored

 Spark log4j warnings
   → Normal during initialization
```

**Errors to Watch For:**
```
 ModuleNotFoundError: No module named 'pyspark'
   → Solution: Activate virtual environment
   → Run: source dtsvenv/bin/activate

 ModuleNotFoundError: No module named 'tensorflow'
   → Solution: pip install -r requirements.txt
```

**Execution Time:** 5-15 seconds

---

### Cell 3: Markdown - Hadoop Configuration Header

```
 No output (markdown cell)
```

---

## Phase 2: Hadoop Configuration

### Cell 4: Initialize Spark Session

**What it does:** Creates SparkSession and connects to Hadoop

** Output:**
```
   ============================================================
   PHASE 2: HADOOP/SPARK CONFIGURATION
   ============================================================
   
   Spark Session Configuration:
   ✓ Application Name: BrainMRI_Distributed_DL
   ✓ Spark Version: 3.5.0
   ✓ Master: local[*]
   ✓ Driver Memory: 2g
   ✓ Executor Memory: 2g
   
   Hadoop Configuration:
   ✓ HDFS Default FS: hdfs://localhost:9000
   ✓ Hadoop Home: /path/to/hadoop
```

**Acceptable Variations:**
```
 Master: local[4] or local[8] (depends on CPU cores)
 Different paths for HADOOP_HOME
 Memory settings may vary based on system
```

**Errors to Watch For:**
```
 "Cannot run multiple SparkContexts at once"
   → Solution: Restart kernel, run from beginning

 "Java gateway process exited before sending port number"
   → Solution: Check JAVA_HOME is set correctly
   → Run: echo $JAVA_HOME
```

**Execution Time:** 10-30 seconds (first Spark init is slow)

---

### Cell 5: Verify HDFS Connection

**What it does:** Tests connection to HDFS filesystem

** Output:**
```
   HDFS Connection Test:
   ✓ NameNode reachable at localhost:9000
   ✓ HDFS filesystem accessible
   ✓ User home directory: /user/[your-username]
   
   HDFS Status:
   ✓ Safe mode: OFF
   ✓ Live datanodes: 1
```

**Errors to Watch For:**
```
 "Connection refused"
   → Solution: Start Hadoop services
   → Run: start-dfs.sh
   → Verify: jps (should show NameNode, DataNode)

 "No such file or directory"
   → Solution: Create user directory
   → Run: hdfs dfs -mkdir -p /user/$USER
```

**Execution Time:** 2-5 seconds

---

## Phase 3: HDFS Data Management

### Cell 6: Markdown - Data Management Header

```
No output (markdown cell)
```

---

### Cell 7: Upload/Verify Data in HDFS

**What it does:** Uploads dataset to HDFS or verifies existing data

** Output (if data already uploaded):**
```
   ============================================================
   PHASE 3: HDFS DATA MANAGEMENT
   ============================================================
   
   Checking HDFS for existing data...
   ✓ Data found at: /user/[username]/brain_mri_data/brain_Tumor_Types
   
   Class Directories:
   ✓ glioma: 1321 images
   ✓ meningioma: 1339 images  
   ✓ notumor: 1595 images
   ✓ pituitary: 1457 images
   
   Total: 5712 images
```

** Output (if uploading):**
```
 Uploading data to HDFS...
   [Progress bar or status updates]
   ✓ Upload complete: 5712 files transferred
   ✓ Total size: ~130 MB
```

**Acceptable Variations:**
```
 Exact image counts may differ by ±5 if dataset version differs
 Upload time varies (2-10 minutes depending on disk speed)
```

**Errors to Watch For:**
```
 "No space left on device"
   → Solution: Check HDFS capacity
   → Run: hdfs dfs -df -h

 "Permission denied"
   → Solution: Check HDFS permissions
   → Run: hdfs dfs -chmod -R 755 /user/$USER
```

**Execution Time:** 1-10 minutes (depends on upload needed)

---

## Phase 4: Data Exploration

### Cell 8: Markdown - Exploration Header

```
No output (markdown cell)
```

---

### Cell 9: Load and Display Sample Images

**What it does:** Loads sample images from HDFS and displays them

** Output:**
```
   ============================================================
   PHASE 4: DATA EXPLORATION
   ============================================================
   
   Loading sample images from HDFS...
   ✓ Loaded 8 sample images
   
   [Visual: 2x4 grid of grayscale brain MRI images]
   [Each image labeled with class name: glioma, meningioma, etc.]
```

**What to Look For in Visualization:**
-  Images are grayscale (not color)
-  Brain anatomy clearly visible
-  Different tumor appearances visible
-  Labels match image content

**Red Flags:**
```
 Completely black/white images → Corrupted data
 Color images → Wrong preprocessing
 Non-medical images → Wrong dataset
```

**Execution Time:** 5-15 seconds

---

### Cell 10: Markdown - Analysis Header

```
 No output (markdown cell)
```

---

### Cell 11: Analyze Image Properties

**What it does:** Checks dimensions, formats, and properties of images

** Output:**
```
 Image Analysis:
   
   Dimension Distribution:
   ├── (512, 512): 3214 images (56.3%)
   ├── (256, 256): 1876 images (32.8%)
   └── (Other): 622 images (10.9%)
   
   Image Properties:
   ├── Format: JPEG
   ├── Mode: RGB (will convert to grayscale)
   └── Bit Depth: 8-bit
   
   Note: All images will be resized to 224x224 for model input
```

**Acceptable Variations:**
```
 Exact dimension counts will vary
 Percentages may differ
```

**Execution Time:** 15-45 seconds

---

## Phase 5: Class Distribution

### Cell 12: Markdown - Class Distribution Header

```
No output (markdown cell)
```

---

### Cell 13: Analyze and Compute Class Weights

**What it does:** Analyzes class balance and computes weights for training

** Output:**
```
   ============================================================
   PHASE 5: CLASS DISTRIBUTION ANALYSIS
   ============================================================
   
   Class Distribution:
   ┌─────────────┬────────┬─────────┐
   │ Class       │ Count  │ Percent │
   ├─────────────┼────────┼─────────┤
   │ glioma      │  1321  │  23.1%  │
   │ meningioma  │  1339  │  23.4%  │
   │ notumor     │  1595  │  27.9%  │
   │ pituitary   │  1457  │  25.5%  │
   └─────────────┴────────┴─────────┘
   Total: 5712 images
   
   Class Weights (for balanced training):
   {0: 1.08, 1: 1.07, 2: 0.90, 3: 0.98}
   
   [Visual: Bar chart showing class distribution]
```

**What This Means:**
- Dataset is relatively balanced (all classes 23-28%)
- Class weights will help prevent bias toward majority class
- "notumor" is largest class, will be slightly downweighted

**Execution Time:** 2-5 seconds

---

## Phase 6: Distributed Preprocessing

### Cell 14: Markdown - Preprocessing Header

```
 No output (markdown cell)
```

---

### Cell 15: Spark Distributed Preprocessing

**What it does:** Preprocesses all images in parallel using Spark

** Output:**
```
   ============================================================
   PHASE 6: DISTRIBUTED PREPROCESSING WITH SPARK
   ============================================================
   
   Configuration:
   ├── Target Size: 224 x 224
   ├── Normalization: [0, 1] range
   ├── Partitions: 4 (parallel workers)
   └── Processing: mapPartitions for efficiency
   
   Starting parallel preprocessing...
   ├── Worker 0: Processing partition 0 (1428 images)
   ├── Worker 1: Processing partition 1 (1428 images)
   ├── Worker 2: Processing partition 2 (1428 images)
   └── Worker 3: Processing partition 3 (1428 images)
   
   ✓ Preprocessing complete!
   ├── Total images processed: 5712
   ├── Time elapsed: 142.35 seconds
   ├── Throughput: 40.1 images/second
   └── Effective speedup: 3.2x vs sequential
   
   Sample verification:
   ├── Image shape: (224, 224, 3)
   ├── Data type: float32
   ├── Value range: [0.000, 1.000]
   └── Labels encoded: [0, 1, 2, 3]
```

**Acceptable Time Ranges:**
```
 2-3 minutes: Fast machine (8+ cores, SSD)
 3-5 minutes: Average machine (4 cores)
 5-10 minutes: Slower hardware (2 cores, HDD)
```

**Monitoring Tip:**
```
 Open http://localhost:4040 in browser during execution
 You'll see Spark job progress, active tasks, memory usage
 This helps identify if processing is stuck
```

**Errors to Watch For:**
```
 "Java heap space" or "OutOfMemoryError"
   → Solution: Reduce executor memory or process in batches
   
 "Task not serializable"
   → Solution: Check preprocessing function doesn't use globals

 Job hangs indefinitely
   → Solution: Check Spark UI for failed tasks
```

**Execution Time:** 2-10 minutes

---

## Phase 7: Model Architecture

### Cell 16: Markdown - Model Architecture Header

```
 No output (markdown cell)
```

---

### Cell 17: Build ResNet-50 Model

**What it does:** Creates ResNet-50 model with custom classification head

** Output:**
```
   ============================================================
   PHASE 7: MODEL ARCHITECTURE
   ============================================================
   
   Loading ResNet-50 base model...
   ✓ Pre-trained weights: ImageNet
   ✓ Include top: False
   ✓ Input shape: (224, 224, 3)
   
   [First time only: Downloading weights...]
   Downloading data from https://storage.googleapis.com/.../resnet50_weights...
   94765736/94765736 [==============================] - 15s
   
   Building custom classification head...
   ├── GlobalAveragePooling2D
   ├── Dense(256, ReLU)
   ├── BatchNormalization
   ├── Dropout(0.5)
   ├── Dense(128, ReLU)
   ├── Dropout(0.3)
   └── Dense(4, Softmax)
   
   Model Summary:
   ═══════════════════════════════════════════════════════════
   Total params: 24,637,508
   Trainable params: 1,051,140 (Stage 1 - frozen base)
   Non-trainable params: 23,586,368
   ═══════════════════════════════════════════════════════════
   
   ✓ Model compiled with Adam optimizer
   ✓ Loss: Sparse Categorical Crossentropy
   ✓ Metrics: Accuracy
```

**Key Numbers to Verify:**
-  Total params: ~24-25 million
-  Trainable params: ~1 million (only custom head)
-  Non-trainable: ~23-24 million (frozen ResNet)

**First Run vs. Subsequent:**
```
 First run: Downloads ~95MB weights (15-45 seconds)
 Later runs: Uses cached weights (2-5 seconds)
```

**Execution Time:** 5-45 seconds (depends on cache)

---

## Phase 8: Training

### Cell 18: Markdown - Training Header

```
 No output (markdown cell)
```

---

### Cell 19: Two-Stage Training

**What it does:** Trains model in two stages (frozen, then fine-tuned)

** Output (Stage 1 - Frozen Base):**
```
   ============================================================
   PHASE 8: TWO-STAGE TRAINING
   ============================================================
   
    Training Configuration:
   ├── Training samples: 4,569 (80%)
   ├── Validation samples: 1,143 (20%)
   ├── Batch size: 16
   ├── Image size: 224 x 224
   └── Data augmentation: Enabled
   
   ══════════════════════════════════════════════════════════
   STAGE 1: Training with frozen ResNet-50 base
   ══════════════════════════════════════════════════════════
   Epochs: 5
   Trainable parameters: 1,051,140
   
   Epoch 1/5
   286/286 [==============================] - 85s 297ms/step 
   loss: 0.8234 - accuracy: 0.6812 - val_loss: 0.5621 - val_accuracy: 0.7923
   
   Epoch 2/5
   286/286 [==============================] - 82s 287ms/step
   loss: 0.4521 - accuracy: 0.8334 - val_loss: 0.3812 - val_accuracy: 0.8623
   
   Epoch 3/5
   286/286 [==============================] - 81s 283ms/step
   loss: 0.3234 - accuracy: 0.8823 - val_loss: 0.3121 - val_accuracy: 0.8834
   
   Epoch 4/5
   286/286 [==============================] - 82s 287ms/step
   loss: 0.2654 - accuracy: 0.9012 - val_loss: 0.2845 - val_accuracy: 0.8956
   
   Epoch 5/5
   286/286 [==============================] - 81s 283ms/step
   loss: 0.2234 - accuracy: 0.9156 - val_loss: 0.2634 - val_accuracy: 0.9034
   
   ✓ Stage 1 complete! Best val_accuracy: 0.9034
   ✓ Model saved: best_model_stage1.keras
```

** Output (Stage 2 - Fine-tuning):**
```
   ══════════════════════════════════════════════════════════
   STAGE 2: Fine-tuning last 30 layers
   ══════════════════════════════════════════════════════════
   Epochs: 10
   Trainable parameters: 12,845,572
   Learning rate: 0.00001 (reduced for fine-tuning)
   
   Epoch 1/10
   286/286 [==============================] - 125s 437ms/step
   loss: 0.1923 - accuracy: 0.9278 - val_loss: 0.2234 - val_accuracy: 0.9156
   
   Epoch 2/10
   286/286 [==============================] - 122s 427ms/step
   loss: 0.1654 - accuracy: 0.9389 - val_loss: 0.2045 - val_accuracy: 0.9234
   
   ... [epochs 3-9] ...
   
   Epoch 10/10
   286/286 [==============================] - 121s 423ms/step
   loss: 0.0823 - accuracy: 0.9712 - val_loss: 0.1845 - val_accuracy: 0.9389
   
   ✓ Stage 2 complete! Best val_accuracy: 0.9389
   ✓ Final model saved
   
   ══════════════════════════════════════════════════════════
   TRAINING SUMMARY
   ══════════════════════════════════════════════════════════
   Total training time: 23.5 minutes
   Final training accuracy: 97.1%
   Final validation accuracy: 93.9%
   Best model saved at epoch: 8 (val_accuracy: 0.9412)
```

**What to Look For:**

 **Good Signs:**
- Loss decreases over epochs
- Accuracy increases over epochs
- Validation accuracy stays close to training accuracy
- No "nan" or "inf" values

 **Acceptable Variations:**
- Exact numbers differ each run (random initialization)
- Time varies by hardware (10-40 minutes)
- Some epoch-to-epoch fluctuation normal

 **Red Flags:**
```
 Loss stays high and flat → Model not learning
   → Check: Learning rate, data preprocessing

 Accuracy stuck at ~25% → Random guessing
   → Check: Labels are correct

 Training accuracy high, validation low → Overfitting
   → Solution: More dropout, early stopping

 "MemoryError" during training
   → Solution: Reduce batch_size to 8
```

**Execution Time:** 15-30 minutes

---

### Cell 20: Training History Plots

**What it does:** Visualizes training metrics over epochs

** Output:**
```
 [Visual: Two side-by-side plots]
   
   Left plot (Loss):
   ├── Title: "Model Loss"
   ├── X-axis: Epoch
   ├── Y-axis: Loss
   ├── Blue line: Training loss (trends downward)
   └── Orange line: Validation loss (trends downward, slightly above training)
   
   Right plot (Accuracy):
   ├── Title: "Model Accuracy"
   ├── X-axis: Epoch
   ├── Y-axis: Accuracy
   ├── Blue line: Training accuracy (trends upward to ~97%)
   └── Orange line: Validation accuracy (trends upward to ~93%)
```

**Ideal Pattern:**
-  Smooth curves (not jagged)
-  Both lines converging
-  Small gap between training/validation
-  No validation diverging upward (overfitting)

**Execution Time:** 2-3 seconds

---

## Phase 9: Evaluation

### Cell 21: Markdown - Evaluation Header

```
 No output (markdown cell)
```

---

### Cell 22: Test Set Evaluation

**What it does:** Evaluates model on held-out test data

** Output:**
```
   ============================================================
   PHASE 9: MODEL EVALUATION
   ============================================================
   
   Loading best model...
   ✓ Model loaded: best_model_stage1.keras
   
   Evaluating on test set (1,143 samples)...
   72/72 [==============================] - 12s 167ms/step
   
   ══════════════════════════════════════════════════════════
   TEST SET RESULTS
   ══════════════════════════════════════════════════════════
   
   Overall Metrics:
   ├── Test Accuracy: 0.9203 (92.03%)
   ├── Test Loss: 0.2134
   └── Macro F1-Score: 0.9156
```

**Acceptable Ranges:**
```
 Test Accuracy: 88-95% is excellent
 85-88% is good
 80-85% is acceptable
 Below 80% indicates issues
```

**Execution Time:** 10-20 seconds

---

### Cell 23: Comprehensive Metrics

**What it does:** Calculates detailed per-class metrics

** Output:**
```
 Per-Class Metrics:
   
   Classification Report:
   ═══════════════════════════════════════════════════════════
                  precision    recall  f1-score   support
   
        glioma       0.93      0.91      0.92       264
    meningioma       0.89      0.91      0.90       268
      notumor        0.95      0.94      0.95       319
     pituitary       0.91      0.90      0.91       292
   
      accuracy                           0.92      1143
     macro avg       0.92      0.92      0.92      1143
   weighted avg      0.92      0.92      0.92      1143
   ═══════════════════════════════════════════════════════════
```

**Understanding Metrics:**
- **Precision**: Of predicted X, how many actually were X?
- **Recall**: Of actual X, how many did we find?
- **F1-Score**: Balanced combination of precision and recall
- **Support**: Number of samples in each class

**Execution Time:** 2-5 seconds

---

### Cell 24: Confusion Matrix

**What it does:** Shows which classes are confused with each other

** Output:**
```
 [Visual: 4x4 heatmap with color intensity]
   
   Confusion Matrix:
   ═══════════════════════════════════════════════════════════
                    Predicted
                 Gli  Men  NoT  Pit
   Actual  Gli [ 240   12    5    7 ]
           Men [  14  244    4    6 ]
           NoT [   3    8  300    8 ]
           Pit [   5    9   10  268 ]
   ═══════════════════════════════════════════════════════════
   
   Interpretation:
   ├── Diagonal: Correct predictions (should be darkest)
   ├── Off-diagonal: Misclassifications
   ├── Most confusion: Meningioma ↔ Glioma (similar appearance)
   └── Best distinguished: No Tumor (most distinctive)
```

**What Good Looks Like:**
-  Dark diagonal (high correct predictions)
-  Light off-diagonal (few errors)
-  No single class with many errors

**Execution Time:** 3-5 seconds

---

### Cell 25: ROC Curves

**What it does:** Plots ROC curves for each class

** Output:**
```
 [Visual: Plot with 4 colored curves + diagonal reference line]
   
   ROC-AUC Scores:
   ├── Glioma:     AUC = 0.97
   ├── Meningioma: AUC = 0.95
   ├── No Tumor:   AUC = 0.98
   └── Pituitary:  AUC = 0.96
   
   Mean AUC: 0.965
```

**Understanding ROC/AUC:**
- AUC = 1.0: Perfect classifier
- AUC > 0.95: Excellent
- AUC > 0.90: Good
- AUC = 0.5: Random guessing

**Execution Time:** 3-5 seconds

---

### Cell 26: Sample Predictions

**What it does:** Shows predictions on individual test images

** Output:**
```
 [Visual: 3x4 grid of brain MRI images]
   
   Each image shows:
   ├── The MRI scan
   ├── True label (actual diagnosis)
   ├── Predicted label (model's guess)
   ├── Confidence score (0.0-1.0)
   └── Color: Green=correct, Red=incorrect
   
   : ~10-11 correct (green), 1-2 incorrect (red) out of 12
```

**What to Look For:**
- Most titles green (correct)
- Confidence scores generally > 0.80
- Incorrect predictions might have lower confidence

**Execution Time:** 3-5 seconds

---

## Phase 10: Save & Summary

### Cell 27: Markdown - Summary Header

```
 No output (markdown cell)
```

---

### Cell 28: Save Artifacts

**What it does:** Saves model and training artifacts

** Output:**
```
   ============================================================
   PHASE 10: SAVE ARTIFACTS
   ============================================================
   
   Saving trained model...
   ✓ Model saved: best_model_stage1.keras
   ✓ File size: 94.23 MB
   
   Saving training history...
   ✓ History saved: training_history.json
   
   Artifacts saved successfully!
```

**Verification:**
```bash
# Check file exists and size
ls -lh best_model_stage1.keras
# Should show ~90-100 MB
```

**Execution Time:** 5-10 seconds

---

### Cell 29: Performance Benchmarking

**What it does:** Compares Spark vs sequential preprocessing

** Output:**
```
   ============================================================
   PERFORMANCE BENCHMARKING
   ============================================================
   
   Preprocessing Comparison (100 image sample):
   ├── Sequential (no Spark): 45.32 seconds
   ├── Spark local[2]: 24.56 seconds (1.85x speedup)
   ├── Spark local[4]: 14.23 seconds (3.18x speedup)
   └── Spark local[*]: 12.15 seconds (3.73x speedup)
   
   Projected Performance at Scale:
   ┌──────────────┬────────────┬────────────┬──────────┐
   │ Dataset Size │ Sequential │ Spark [4]  │ Speedup  │
   ├──────────────┼────────────┼────────────┼──────────┤
   │ 5K (ours)    │ ~10 min    │ ~3 min     │ 3.3x     │
   │ 50K          │ ~100 min   │ ~25 min    │ 4.0x     │
   │ 500K         │ ~17 hours  │ ~3.5 hours │ 4.8x     │
   │ 5M           │ ~7 days    │ ~1.2 days  │ 5.5x     │
   └──────────────┴────────────┴────────────┴──────────┘
```

**Execution Time:** 1-2 minutes

---

### Cell 30: Final Summary (Markdown)

**What it shows:** Complete project summary and results

```
 No output (markdown cell - displays formatted summary)
```

---

## Final Results Summary

###  Final Metrics

```
╔══════════════════════════════════════════════════════════════╗
║                    FINAL MODEL PERFORMANCE                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Training Metrics (after 15 epochs):                         ║
║  ├── Training Accuracy:    95-98%                            ║
║  ├── Training Loss:        0.08-0.15                         ║
║  ├── Validation Accuracy:  91-95%                            ║
║  └── Validation Loss:      0.18-0.28                         ║
║                                                              ║
║  Test Set Performance:                                       ║
║  ├── Test Accuracy:        88-94%                            ║
║  ├── Test Loss:            0.20-0.35                         ║
║  └── Macro F1-Score:       0.88-0.93                         ║
║                                                              ║
║  Per-Class Performance:                                      ║
║  ├── Glioma:      Precision ~0.92, Recall ~0.91, F1 ~0.92   ║
║  ├── Meningioma:  Precision ~0.89, Recall ~0.90, F1 ~0.90   ║
║  ├── No Tumor:    Precision ~0.95, Recall ~0.94, F1 ~0.95   ║
║  └── Pituitary:   Precision ~0.91, Recall ~0.90, F1 ~0.91   ║
║                                                              ║
║  ROC-AUC Scores:                                             ║
║  └── All classes: 0.95-0.98                                  ║
║                                                              ║
║  Performance Benchmarks:                                     ║
║  ├── Preprocessing Speedup: 2.5-4x (vs sequential)           ║
║  ├── Total Training Time: 15-30 minutes                      ║
║  └── Total Notebook Runtime: 35-45 minutes                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Files Generated

| File | Size | Description |
|------|------|-------------|
| `best_model_stage1.keras` | ~94 MB | Trained model |
| Training plots | In notebook | Loss/accuracy curves |
| Confusion matrix | In notebook | Classification analysis |
| ROC curves | In notebook | Per-class performance |

---

## Troubleshooting Discrepancies

### If Your Outputs Differ Significantly

#### Problem: Accuracy Much Lower Than  (< 80%)

**Possible Causes & Solutions:**

1. **Data not uploaded correctly to HDFS**
   ```bash
   # Verify data
   hdfs dfs -ls /user/$USER/brain_mri_data/brain_Tumor_Types/
   hdfs dfs -count /user/$USER/brain_mri_data/brain_Tumor_Types/*
   # Should show 4 directories, 5712 total files
   ```

2. **Preprocessing errors**
   - Check normalized range is [0, 1]
   - Verify image shape is (224, 224, 3)
   - Confirm labels are 0, 1, 2, 3 (not strings)

3. **Model not loading ImageNet weights**
   - First run should download weights
   - Check internet connection
   - Verify `weights='imagenet'` in code

#### Problem: Training Extremely Slow (> 1 hour)

**Possible Causes & Solutions:**

1. **Running on CPU ( but slower)**
   ```python
   # Check GPU availability
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   # Empty list = CPU only
   ```

2. **Batch size too small**
   - Increase from 16 to 32 if memory allows

3. **Too many epochs**
   - 15 total epochs should be sufficient

4. **System resources low**
   - Close other applications
   - Check RAM usage

#### Problem: Memory Errors

**Possible Causes & Solutions:**

1. **Batch size too large**
   - Reduce `batch_size` to 8

2. **Spark memory misconfigured**
   - Reduce `spark.driver.memory` to 1g
   - Reduce `spark.executor.memory` to 1g

3. **Too many images in memory**
   - Process data in smaller batches

#### Problem: HDFS Connection Issues

**Possible Causes & Solutions:**

1. **Hadoop services not running**
   ```bash
   jps
   # Should show: NameNode, DataNode, SecondaryNameNode
   
   # If missing, start services:
   start-dfs.sh
   ```

2. **Wrong HDFS path**
   - Verify path matches your username
   - Check: `echo $USER`

3. **Port conflicts**
   - Check if port 9000 is in use
   - Verify in `core-site.xml`

#### Problem: Model Not Learning (Accuracy Stuck at ~25%)

**Possible Causes & Solutions:**

1. **Labels encoded incorrectly**
   ```python
   # Print sample labels to verify
   print(y_train[:20])
   # Should be: [0, 1, 2, 3, 0, 2, ...] (integers)
   ```

2. **Learning rate wrong**
   - Try 0.0001 or 0.01

3. **Data not shuffled**
   - Ensure shuffling during training

---

## Getting Help

### Diagnostic Steps

1. **Compare outputs line-by-line** with this guide
2. **Check Spark UI** at `http://localhost:4040`
3. **Check HDFS UI** at `http://localhost:9870`
4. **Review logs** in `$HADOOP_HOME/logs/`

### Common Error Patterns

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing package | `pip install -r requirements.txt` |
| `MemoryError` | RAM exhausted | Reduce batch size |
| `Connection refused` | Service not running | `start-dfs.sh` |
| `Shape mismatch` | Preprocessing issue | Verify image dimensions |
| `nan` loss | Learning rate too high | Reduce to 0.0001 |

---

**END OF OUTPUTS.md**


