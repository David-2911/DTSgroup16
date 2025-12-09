# Brain Tumor Classification using Distributed Deep Learning

**Project:** Complete Medical Imaging Deep Learning with Spark/Hadoop  
**Question:** How to apply deep learning to large-scale medical imaging (e.g. MRI or histopathology) using Spark/Hadoop clusters?

## 🎯 Project Overview

This project demonstrates a **production-ready distributed deep learning pipeline** for brain tumor classification from MRI images using:
- **Apache Spark** for distributed data processing
- **Hadoop HDFS** for distributed storage
- **TensorFlow/Keras** with ResNet-50 CNN
- **Parallel training jobs** for hyperparameter tuning

### Dataset
- **Source:** Brain MRI Images (4 classes)
- **Total Images:** 5,662
- **Classes:**
  - Glioma: 1,271 images
  - Meningioma: 1,339 images
  - No Tumor: 1,595 images
  - Pituitary: 1,457 images
- **Storage:** HDFS (125+ MB distributed)

## 📋 Prerequisites

### Software Requirements
- Python 3.8+
- TensorFlow 2.15+
- PySpark 3.0+
- Apache Hadoop 3.x
- Java 8 or 11 (for Hadoop/Spark)

### Python Packages
```bash
pip install tensorflow pyspark numpy pandas pillow matplotlib seaborn scikit-learn
```

### System Requirements
- **Minimum:** 8GB RAM, 20GB disk space
- **Recommended:** 16GB RAM, 50GB disk space, GPU (optional)
- **For production:** Multi-node Spark/Hadoop cluster

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source dtsvenv/bin/activate

# Verify installations
python -c "import tensorflow; print(tensorflow.__version__)"
python -c "import pyspark; print(pyspark.__version__)"
```

### 2. Start Hadoop Services
```bash
# Start HDFS (if not running)
$HADOOP_HOME/sbin/start-dfs.sh

# Verify HDFS
hdfs dfs -ls /
```

### 3. Run the Notebook
```bash
jupyter notebook Brain_MRI_Distributed_DL_FINAL.ipynb
```

**Important:** Execute cells sequentially from top to bottom. The notebook is fully automated with:
- ✅ Auto-detection of all paths (HADOOP_HOME, dataset location, etc.)
- ✅ No hardcoded paths - runs on any computer
- ✅ Comprehensive error checking and status messages

## 📊 Notebook Structure

### Phase 1: Environment Setup & Verification
- Import all required libraries
- Verify TensorFlow, PySpark, NumPy versions
- Check system resources

### Phase 2: Hadoop & Spark Configuration
- Auto-detect HADOOP_HOME and JAVA_HOME
- Initialize Spark session with HDFS connection
- Configure for distributed processing

### Phase 3: Dataset Exploration & Analysis
- Auto-detect dataset location
- Analyze class distribution
- Display sample images
- Calculate dataset statistics

### Phase 4: Data Preparation & Splitting
- Stratified train/validation/test split (72%/14%/14%)
- Maintain class balance across splits
- Generate split metadata

### Phase 5: Upload Dataset to HDFS
- Check for existing HDFS data
- Upload images to distributed storage
- Verify upload integrity
- Display HDFS directory structure

### Phase 5B: Distributed Data Pipeline with HDFS & Spark
- Create Spark DataFrame catalog of HDFS images
- Distributed train/val/test splitting using Spark
- Demonstrate parallel data access

### Phase 6: Distributed Preprocessing with Spark ⭐
**PROJECT REQUIREMENT:** "Use Spark to preprocess (tile, normalize)"
- Parallel image loading from HDFS
- Distributed resizing to 224×224 (tiling)
- Normalization to [0,1] range across workers
- Demonstrate scalability (4 partitions = 4x speedup potential)

### Phase 7: Build ResNet-50 CNN Model
- Transfer learning from ImageNet
- Custom classification head for 4 classes
- ~25M parameters
- Adam optimizer with categorical crossentropy

### Phase 8: Distributed Training with TensorFlow on Spark ⭐⭐
**CRITICAL REQUIREMENT:** "TensorFlow on Spark"
- Spark-based data loading from HDFS
- Parallel batch generation across workers
- Distributed training pipeline
- 10 epochs demonstration

### Phase 9: Model Evaluation & Performance Comparison
- Classification report (precision, recall, F1)
- Confusion matrix visualization
- Training history plots
- **Distributed vs Non-Distributed comparison**

### Phase 10: Save Model & Results
- Save trained model (.keras format)
- Export training history (JSON)
- Generate metadata file
- Save visualizations

### Phase 11: Parallel Training Jobs ⭐
**PROJECT REQUIREMENT:** "Run parallel training jobs"
- Hyperparameter tuning with Spark
- 5 configurations tested simultaneously
- Distributed across workers
- Automated best config selection

### Summary & Conclusion
- Complete project overview
- Key advantages of distributed approach
- Real-world applications
- Technologies demonstrated

## 🔑 Key Features

### 1. Fully Portable
- ✅ **Auto-detection** of all paths and configurations
- ✅ **No hardcoded locations** - works on any computer
- ✅ **Automatic** HADOOP_HOME, JAVA_HOME, dataset discovery
- ✅ **Cross-platform** compatible (Linux, macOS, Windows with WSL)

### 2. Distributed Computing
- ✅ **HDFS storage** for fault-tolerant data distribution
- ✅ **Spark preprocessing** with parallel workers
- ✅ **Scalable** to millions of images
- ✅ **Parallel training jobs** for hyperparameter optimization

### 3. Production-Ready
- ✅ **Comprehensive error handling**
- ✅ **Progress monitoring** and status updates
- ✅ **Metadata tracking** for reproducibility
- ✅ **Visualization** of results and metrics

### 4. Educational
- ✅ **Detailed explanations** in every phase
- ✅ **Comparison** of distributed vs non-distributed
- ✅ **Best practices** demonstrated throughout
- ✅ **Complete documentation**

## 📈 Results

### Model Performance
- **Architecture:** ResNet-50 with transfer learning
- **Training:** 10 epochs with distributed data pipeline
- **Metrics:** Accuracy, precision, recall, F1-score per class
- **Evaluation:** Confusion matrix, ROC curves, training history

### Scalability Comparison

| Aspect | Local (Non-Distributed) | Distributed (Spark/HDFS) |
|--------|------------------------|--------------------------|
| **Data Loading** | Sequential, RAM-limited | Parallel, scales linearly |
| **Preprocessing** | Single-core bottleneck | Multi-worker (4x+ speedup) |
| **Storage** | Single disk, no redundancy | HDFS with replication |
| **Max Dataset Size** | ~10K images (8GB RAM) | Millions (cluster-limited) |
| **Training Speed** | I/O bottleneck | Parallel pipeline |
| **Fault Tolerance** | None | Automatic recovery |

## 🛠️ Troubleshooting

### HDFS Connection Issues
```bash
# Check HDFS status
hdfs dfsadmin -report

# Restart HDFS if needed
$HADOOP_HOME/sbin/stop-dfs.sh
$HADOOP_HOME/sbin/start-dfs.sh
```

### Spark Session Errors
- Ensure HADOOP_HOME is set correctly
- Check HDFS is running on port 8020 (or update in notebook)
- Verify Java version (8 or 11 recommended)

### Memory Issues
- Reduce batch size in Phase 8 (default: 32)
- Decrease number of images processed in demonstrations
- Increase Spark executor memory in Phase 2

### Port Conflicts
- Default HDFS RPC: 8020
- Default HDFS Web UI: 9870
- If conflicts, update `core-site.xml` and restart HDFS

## 📚 Project Structure

```
DTSgroup16/
├── Brain_MRI_Distributed_DL_FINAL.ipynb    # Main notebook (CURRENT)
├── Brain_MRI_Distributed_DL.ipynb          # Original (backup kept)
├── Brain_MRI_Distributed_DL_backup_*.ipynb # Auto-backup
├── NOTEBOOK_STRUCTURE.md                   # Phase documentation
├── README_FINAL.md                         # This file
├── dtsvenv/                                # Python virtual environment
├── dataset/                                # Local dataset (auto-detected)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── brain_tumor_resnet50_distributed.keras  # Trained model (after Phase 10)
├── training_history.json                   # Training metrics (after Phase 10)
├── model_metadata.json                     # Model info (after Phase 10)
├── confusion_matrix.png                    # Visualization (after Phase 9)
└── training_history.png                    # Visualization (after Phase 9)
```

## 🎓 Learning Outcomes

This project demonstrates:

1. **Distributed Storage:** HDFS for fault-tolerant data management
2. **Distributed Processing:** Spark RDDs and DataFrames for parallel computation
3. **Deep Learning:** ResNet-50 CNN for medical image classification
4. **Transfer Learning:** Leveraging pre-trained ImageNet weights
5. **Scalability:** Design patterns for production-scale datasets
6. **Best Practices:** Error handling, monitoring, documentation

## 🚀 Scaling to Production

To deploy on a real Spark/Hadoop cluster:

1. **Multi-node Setup:**
   - Deploy on 3+ node cluster
   - Configure HDFS replication factor
   - Set up Spark standalone/YARN cluster

2. **Code Changes:**
   - Update `SparkSession.master()` to cluster URL
   - Increase executor memory and cores
   - Enable GPU support if available

3. **Data Pipeline:**
   - Use TensorFlowOnSpark or Elephas for true distributed training
   - Implement streaming data ingestion
   - Add model checkpointing to HDFS

4. **Monitoring:**
   - Enable Spark UI (port 4040)
   - Set up HDFS monitoring (port 9870)
   - Add logging and alerting

## 📖 References

- **Dataset:** Brain Tumor MRI Dataset (Kaggle)
- **ResNet Paper:** "Deep Residual Learning for Image Recognition" (He et al., 2015)
- **Apache Spark:** https://spark.apache.org/docs/latest/
- **Hadoop HDFS:** https://hadoop.apache.org/docs/current/
- **TensorFlow:** https://www.tensorflow.org/guide

## 👥 Project Information

**Course:** Distributed and Time-Sensitive Systems  
**Group:** 16  
**Objective:** Demonstrate distributed deep learning for medical imaging

## 📝 License

Educational project - dataset and code for academic purposes.

---

**✅ Project Status:** COMPLETE - All phases implemented and documented

**🎯 Project Question Answered:** Successfully demonstrated how to apply deep learning to large-scale medical imaging using Spark/Hadoop clusters with comprehensive implementation of distributed storage (HDFS), parallel preprocessing (Spark), CNN training (TensorFlow), and parallel jobs.
