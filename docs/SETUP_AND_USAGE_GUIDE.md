# Complete Setup and Usage Guide
## Brain MRI Classification with Distributed Deep Learning
### DTS 301 - Big Data Computing | Group 16

This guide takes you from a fresh machine to running the complete project, including both the Jupyter notebook for model training and the web application for classification.

**Estimated Time:** 1-2 hours (depending on download speeds)

---

## Table of Contents

**Part A: Prerequisites & Environment**
1. [System Requirements](#1-system-requirements)
2. [Prerequisites Checklist](#2-prerequisites-checklist)
3. [Installing Prerequisites](#3-installing-prerequisites)

**Part B: Infrastructure Setup**
4. [Hadoop Installation & Configuration](#4-hadoop-installation--configuration)
5. [Apache Spark Installation](#5-apache-spark-installation)
6. [HDFS Data Upload](#6-hdfs-data-upload)

**Part C: Project Setup**
7. [Python Environment Setup](#7-python-environment-setup)
8. [Frontend Setup](#8-frontend-setup)
9. [Pre-Flight Verification](#9-pre-flight-verification)

**Part D: Running the Project**
10. [Running the Jupyter Notebook](#10-running-the-jupyter-notebook)
11. [Running the Web Application](#11-running-the-web-application)
12. [Using the Web Application](#12-using-the-web-application)

**Part E: Reference**
13. [Troubleshooting](#13-troubleshooting)
14. [Testing](#14-testing)
15. [Stopping Services](#15-stopping-services)
16. [Quick Reference Card](#16-quick-reference-card)

---

# Part A: Prerequisites & Environment

## 1. System Requirements

### 1.1 Hardware Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **RAM** | 8GB | 16GB |
| **Disk Space** | 20GB free | 50GB free |
| **CPU** | 4 cores | 8+ cores |
| **GPU** | Not required | NVIDIA GPU with 2GB+ VRAM (CUDA 12.0 compatible) |

> **Note:** This guide was tested on Ubuntu 22.04 with 16GB RAM, NVIDIA MX150 (2GB), Python 3.12, TensorFlow 2.16.2, Spark 4.1.0, and Hadoop 3.4.2.

### 1.2 Required Ports

| Port | Service | Description |
|------|---------|-------------|
| 3000 | React Frontend | Web interface |
| 5000 | Flask Backend | API server |
| 8888 | Jupyter Notebook | Training interface |
| 9000 | HDFS NameNode | Hadoop storage |
| 9870 | Hadoop Web UI | HDFS monitoring |
| 4040 | Spark Web UI | Job monitoring |

---

## 2. Prerequisites Checklist

Before starting, ensure you have or will install:

```
[ ] Operating System: Ubuntu 20.04+, Windows 10/11, or macOS 10.15+
[ ] Python 3.12
[ ] Java 17 (required for Hadoop/Spark)
[ ] Node.js 18.x or higher (for web frontend)
[ ] 20GB free disk space
[ ] Git (optional, for cloning)
```

### Verification Commands

```bash
# Check Python version (should be 3.12+)
python3 --version

# Check Java version (should be 17)
java -version

# Check Node.js version (should be 18+)
node --version

# Check available disk space
df -h .

# Check available RAM
free -h          # Linux
vm_stat          # macOS
systeminfo       # Windows
```

---

## 3. Installing Prerequisites

### 3.1 Install Java

Java 17 is required for Hadoop and Spark.

<details>
<summary><b>Linux (Ubuntu/Debian)</b></summary>

```bash
sudo apt update
sudo apt install openjdk-17-jdk -y

# Verify
java -version

# Set JAVA_HOME
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
source ~/.bashrc
```
</details>

<details>
<summary><b>macOS</b></summary>

```bash
# Using Homebrew
brew install openjdk@17

# Add to PATH
echo 'export PATH="/usr/local/opt/openjdk@17/bin:$PATH"' >> ~/.zshrc
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 17)' >> ~/.zshrc
source ~/.zshrc
```
</details>

<details>
<summary><b>Windows</b></summary>

1. Download OpenJDK 17 from [adoptium.net](https://adoptium.net/)
2. Run installer, note installation path
3. Set environment variables:
   - `JAVA_HOME` = `C:\Program Files\Eclipse Adoptium\jdk-17`
   - Add `%JAVA_HOME%\bin` to `PATH`
4. Restart terminal and verify: `java -version`
</details>

### 3.2 Install Python

<details>
<summary><b>Linux (Ubuntu/Debian)</b></summary>

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y

# Verify
python3.12 --version
```
</details>

<details>
<summary><b>macOS</b></summary>

```bash
brew install python@3.12
```
</details>

<details>
<summary><b>Windows</b></summary>

1. Download from [python.org](https://www.python.org/downloads/)
2. **Important:** Check "Add Python to PATH" during installation
3. Verify: `python --version`
</details>

### 3.3 Install Node.js

<details>
<summary><b>Linux (Ubuntu/Debian)</b></summary>

```bash
# Using NodeSource repository
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Verify
node --version   # Should show v18.x.x
npm --version    # Should show 9.x.x or higher
```
</details>

<details>
<summary><b>macOS</b></summary>

```bash
brew install node@18
```
</details>

<details>
<summary><b>Windows</b></summary>

1. Download from [nodejs.org](https://nodejs.org/)
2. Run installer
3. Verify: `node --version`
</details>

---

# Part B: Infrastructure Setup

## 4. Hadoop Installation & Configuration

> **What is Hadoop?**
> 
> Hadoop is like a **super-powered filing cabinet for big data**. Instead of storing files on one computer, it splits them into chunks and stores them across multiple machines.
> 
> **Key Components:**
> - **NameNode** = The librarian who knows where every file is stored
> - **DataNode** = The shelves that hold the actual data
> - **HDFS** = The Hadoop Distributed File System

### 4.1 Download Hadoop

```bash
# Download Hadoop 3.4.2
cd ~
wget https://dlcdn.apache.org/hadoop/common/hadoop-3.4.2/hadoop-3.4.2.tar.gz

# Extract
tar -xzvf hadoop-3.4.2.tar.gz

# Rename for convenience
mv hadoop-3.4.2 hadoop

# Verify
ls ~/hadoop/bin/hdfs
```

### 4.2 Set Environment Variables

<details>
<summary><b>Linux (~/.bashrc)</b></summary>

```bash
cat >> ~/.bashrc << 'EOF'

# Hadoop Configuration
export HADOOP_HOME=~/hadoop
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
EOF

source ~/.bashrc
```
</details>

<details>
<summary><b>macOS (~/.zshrc)</b></summary>

```bash
cat >> ~/.zshrc << 'EOF'

# Hadoop Configuration
export HADOOP_HOME=~/hadoop
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
EOF

source ~/.zshrc
```
</details>

### 4.3 Configure Hadoop

Navigate to configuration directory:
```bash
cd $HADOOP_HOME/etc/hadoop
```

#### File 1: `core-site.xml`

```bash
nano core-site.xml
```

Replace content with:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/tmp/hadoop-${user.name}</value>
    </property>
</configuration>
```

#### File 2: `hdfs-site.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file:///tmp/hadoop/namenode</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file:///tmp/hadoop/datanode</value>
    </property>
    <property>
        <name>dfs.permissions.enabled</name>
        <value>false</value>
    </property>
</configuration>
```

#### File 3: `hadoop-env.sh`

```bash
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> hadoop-env.sh
```

### 4.4 Enable SSH (Linux/Mac Only)

```bash
# Install SSH if needed
sudo apt install ssh pdsh -y

# Generate SSH key
ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa

# Allow passwordless SSH
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# Test SSH
ssh localhost
# Type 'exit' to return
```

### 4.5 Format HDFS (First Time Only)

> **WARNING:** Only run this ONCE! Formatting erases all HDFS data.

```bash
hdfs namenode -format
```

### 4.6 Start Hadoop

```bash
start-dfs.sh
```

### 4.7 Verify Hadoop

```bash
# Check processes
jps
# Should show: NameNode, DataNode, SecondaryNameNode

# Web UI
# Open browser: http://localhost:9870
```

---

## 5. Apache Spark Installation

> **What is Spark?**
> 
> Spark is like **Excel for massive datasets**. While Excel crashes with millions of rows, Spark processes them by splitting data across workers and processing in parallel.

### 5.1 Download Spark

```bash
cd ~
wget https://dlcdn.apache.org/spark/spark-4.0.0/spark-4.0.0-bin-hadoop3.tgz

# Extract
tar -xzvf spark-4.0.0-bin-hadoop3.tgz
mv spark-4.0.0-bin-hadoop3 spark
```

### 5.2 Set Environment Variables

```bash
cat >> ~/.bashrc << 'EOF'

# Spark Configuration
export SPARK_HOME=~/spark
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3
EOF

source ~/.bashrc
```

### 5.3 Configure Spark for 16GB RAM

```bash
cd $SPARK_HOME/conf
cp spark-defaults.conf.template spark-defaults.conf
nano spark-defaults.conf
```

Add these settings:
```properties
spark.master                     local[*]
spark.driver.memory              4g
spark.executor.memory            4g
spark.sql.shuffle.partitions     4
spark.default.parallelism        4
spark.serializer                 org.apache.spark.serializer.KryoSerializer
spark.ui.port                    4040
```

> **Note:** For 8GB RAM systems, reduce driver.memory and executor.memory to 2g each.

### 5.4 Verify Spark

```bash
# Test Spark Shell
spark-shell
# Type :quit to exit

# Test PySpark
pyspark
# Type exit() to quit
```

---

## 6. HDFS Data Upload

### 6.1 Verify Dataset

```bash
ls -la brain_Tumor_Types/
# Should show: glioma, meningioma, notumor, pituitary

find brain_Tumor_Types -name "*.jpg" | wc -l
# Should show ~5712
```

### 6.2 Upload to HDFS

```bash
# Create directory
hdfs dfs -mkdir -p /medical_imaging/brain_tumor

# Upload dataset
hdfs dfs -put brain_Tumor_Types/ /medical_imaging/brain_tumor/

# Verify
hdfs dfs -ls /medical_imaging/brain_tumor/brain_Tumor_Types/
```

---

# Part C: Project Setup

## 7. Python Environment Setup

### 7.1 Create Virtual Environment

> **Why Virtual Environments?**
> A virtual environment is like a "sandbox" for your project. It keeps all packages isolated from system Python, preventing conflicts.

```bash
# Navigate to project
cd /path/to/DTSgroup16

# Create virtual environment
python3.12 -m venv dtsvenv

# Activate
source dtsvenv/bin/activate  # Linux/Mac
# dtsvenv\Scripts\activate   # Windows

# Verify - should show path inside dtsvenv
which python
```

### 7.2 Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
pip install -r webapp/backend/requirements.txt
```

**What's being installed?**

| Package | Purpose |
|---------|---------|
| `tensorflow` | Deep learning framework |
| `pyspark` | Apache Spark Python API |
| `flask` | Web server framework |
| `flask-cors` | Cross-origin resource sharing |
| `Pillow` | Image processing |
| `numpy`, `pandas` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `jupyter` | Notebook environment |

**Common Issues:**

| Error | Solution |
|-------|----------|
| `Could not build wheels` | `pip install --upgrade setuptools wheel` |
| `MemoryError` | `pip install --no-cache-dir -r requirements.txt` |
| `Permission denied` | Don't use `sudo` with virtual environment |

---

## 8. Frontend Setup

```bash
# Navigate to frontend
cd webapp/frontend

# Install Node.js packages
npm install
```

**Installation time:** 2-5 minutes

---

## 9. Pre-Flight Verification

### 9.1 Verification Checklist

```
PREREQUISITES
  [ ] Python 3.10+: python3 --version
  [ ] Java 11: java -version
  [ ] Node.js 18+: node --version
  [ ] Virtual environment active: (dtsvenv) in prompt

HADOOP
  [ ] Environment set: echo $HADOOP_HOME
  [ ] Services running: jps shows NameNode, DataNode
  [ ] Web UI: http://localhost:9870

SPARK
  [ ] Environment set: echo $SPARK_HOME
  [ ] spark-shell works

DATA
  [ ] Dataset in HDFS: hdfs dfs -ls /medical_imaging/brain_tumor/

PYTHON
  [ ] TensorFlow: python -c "import tensorflow"
  [ ] PySpark: python -c "import pyspark"
  [ ] Flask: python -c "import flask"
```

### 9.2 Run Structure Verification

```bash
cd /path/to/DTSgroup16
python webapp/verify_structure.py
```

Output:
```
Checking project structure...
[OK] 34/34 required files found
All checks passed!
```

---

# Part D: Running the Project

## 10. Running the Jupyter Notebook

### 10.1 Start Required Services

```bash
# 1. Start Hadoop
start-dfs.sh

# 2. Verify
jps  # Should show NameNode, DataNode

# 3. Activate virtual environment
source dtsvenv/bin/activate

# 4. Install Jupyter kernel
python -m ipykernel install --user --name=dtsvenv --display-name="Python (Brain MRI Project)"

# 5. Launch Jupyter
jupyter notebook
```

### 10.2 Execute Notebook

1. Open `Brain_MRI_Distributed_DL.ipynb`
2. Select kernel: `Kernel` -> `Change kernel` -> `Python (Brain MRI Project)`
3. Run cells sequentially with `Shift+Enter`

### 10.3 Runtime

| Phase | Description | Time |
|-------|-------------|------|
| 1-5 | Environment & Data Setup | 2-5 min |
| 6 | Spark Preprocessing Demo | 1-2 min |
| 7 | Model Building | 30 sec |
| 8 | Training (15 epochs) | 20-40 min |
| 9 | Evaluation | 3-5 min |
| **Total** | | **30-60 min** |

### 10.4 Monitoring

- **Spark UI:** http://localhost:4040 (during Spark jobs)
- **Hadoop UI:** http://localhost:9870

---

## 11. Running the Web Application

### 11.1 Quick Start (Recommended)

```bash
cd /path/to/DTSgroup16/webapp
chmod +x start.sh
./start.sh
```

This script:
1. Activates the virtual environment
2. Starts Flask backend on port 5000
3. Starts React frontend on port 3000
4. Opens your browser

### 11.2 Manual Start

**Terminal 1 - Backend:**
```bash
cd /path/to/DTSgroup16
source dtsvenv/bin/activate
cd webapp/backend
python app.py
```

Output:
```
============================================================
Initializing Brain Tumor Classification Backend
============================================================
Initializing Spark session...
Spark session created: local-xxxxx
Loading classification model...
Model loaded successfully
============================================================
Backend ready! Running on http://localhost:5000
============================================================
```

**Terminal 2 - Frontend:**
```bash
cd /path/to/DTSgroup16/webapp/frontend
npm start
```

Output:
```
Compiled successfully!
Local: http://localhost:3000
```

### 11.3 Verify Application

```bash
curl http://localhost:5000/api/health
```

Expected:
```json
{"status": "healthy", "model_loaded": true, "spark_active": true}
```

---

## 12. Using the Web Application

### 12.1 Main Interface

```
+----------------------------------------------------------------+
|  Brain MRI Classification System          [Backend Connected]  |
+----------------------------------------------------------------+
|                                                                 |
|          +------------------------------------+                 |
|          |                                    |                 |
|          |     Drag and drop an MRI image     |                 |
|          |          or click to upload        |                 |
|          |                                    |                 |
|          |     Supported: PNG, JPG, JPEG      |                 |
|          |     Max size: 16MB                 |                 |
|          |                                    |                 |
|          +------------------------------------+                 |
|                                                                 |
|  [ Analyze MRI Scan ]  [ Clear ]                                |
|                                                                 |
+----------------------------------------------------------------+
```

### 12.2 Uploading an Image

**Method 1: Drag and Drop**
1. Open file manager
2. Navigate to any MRI image (or sample from `brain_Tumor_Types/`)
3. Drag onto upload area

**Method 2: Click to Browse**
1. Click on upload area
2. Browse and select file

### 12.3 Running Classification

1. Click **"Analyze MRI Scan"**
2. Wait 1-3 seconds
3. View results

### 12.4 Understanding Results

**Section 1: Classification Result**
```
+-----------------------------------------+
|  Predicted Class: GLIOMA                |
|  Confidence: 92.5%                      |
|                                         |
|  All Probabilities:                     |
|  ================== Glioma       92.5%  |
|  ==                 Meningioma    4.2%  |
|  =                  Pituitary     2.1%  |
|                     No Tumor      1.2%  |
+-----------------------------------------+
```

**Section 2: Heatmap Visualization**
- Original MRI image alongside Grad-CAM heatmap
- Red/Yellow = Important regions (model focused here)
- Blue/Purple = Less important regions

**Section 3: Medical Analysis**
- Educational information about the detected condition
- **NOT a clinical diagnosis**

### 12.5 Analyzing Another Image

Click **"Analyze New Image"** to reset.

---

# Part E: Reference

## 13. Troubleshooting

### Issue 1: Out of Memory Error

**Symptoms:** `java.lang.OutOfMemoryError` or kernel crashes

**Solutions:**
1. Reduce batch size: `BATCH_SIZE = 8`
2. Close other applications
3. Reduce Spark memory in `spark-defaults.conf`:
   ```properties
   spark.driver.memory    1g
   spark.executor.memory  1g
   ```

### Issue 2: HDFS Connection Refused

**Symptoms:** `Connection refused: localhost:9000`

**Solution:**
```bash
jps  # Check if NameNode/DataNode running
stop-dfs.sh
start-dfs.sh
sleep 30
jps
```

### Issue 3: Backend Not Available

**Symptoms:** Red indicator in header

**Solution:**
```bash
curl http://localhost:5000/api/health

# If not running:
cd webapp/backend
source ../dtsvenv/bin/activate
python app.py
```

### Issue 4: Module Not Found

**Symptoms:** `ModuleNotFoundError`

**Solution:**
```bash
# Verify kernel/environment
source dtsvenv/bin/activate
pip install -r requirements.txt
```

### Issue 5: Port Already in Use

**Symptoms:** `Address already in use`

**Solution:**
```bash
lsof -i :5000  # or :3000, :9000
kill -9 <PID>
```

### Issue 6: Spark Job Hangs

**Solutions:**
1. Check Spark UI: http://localhost:4040
2. Restart kernel
3. `spark.stop()` then re-initialize

### Issue 7: Java Version Mismatch

**Symptoms:** `Unsupported class file major version`

**Solution:** Use Java 17 (recommended for Spark 4.x and Hadoop 3.4.x)

### Issue 8: Model File Not Found

**Solution:**
```bash
ls -la models/best_model_extended.keras
# If missing, retrain using Jupyter notebook
```

### Issue 9: Permission Denied in HDFS

**Solution:**
```bash
hdfs dfs -chmod -R 777 /medical_imaging/
```

### Issue 10: "No such file or directory" in HDFS

**Solution:**
```bash
# Check if data exists
hdfs dfs -ls /medical_imaging/

# If not, upload again
hdfs dfs -mkdir -p /medical_imaging/brain_tumor
hdfs dfs -put brain_Tumor_Types/ /medical_imaging/brain_tumor/
```

### Issue 11: GPU Out of Memory (OOM)

**Symptoms:** `ResourceExhaustedError` or `OOM when allocating tensor`

**Solutions:**
1. Reduce batch size to 4 in the notebook:
   ```python
   BATCH_SIZE = 4
   ```
2. Enable GPU memory growth:
   ```python
   gpus = tf.config.list_physical_devices('GPU')
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
   ```

### Issue 12: libdevice.10.bc Not Found

**Symptoms:** `Could not find libdevice.10.bc` during TensorFlow GPU operations

**Solution:** Set XLA flags before importing TensorFlow:
```python
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit'
```

### Issue 13: TensorFlow + CUDA Version Mismatch

**Symptoms:** TensorFlow crashes or GPU not detected

**Solution:** Use compatible versions:
- CUDA 12.0 + cuDNN 8.9 → TensorFlow 2.16.2
- Install: `pip install tensorflow==2.16.2`

### Issue 14: Webapp Exhausts GPU Memory

**Symptoms:** GPU OOM when running both notebook and webapp

**Solution:** Force webapp to use CPU only by setting in `webapp/backend/config.py`:
```python
os.environ['TF_FORCE_CPU_ONLY'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

---

## 14. Testing

### 14.1 Integration Tests

```bash
cd /path/to/DTSgroup16
source dtsvenv/bin/activate
python webapp/test_integration.py
```

: `9 tests passed, 0 failed`

### 14.2 Structure Verification

```bash
python webapp/verify_structure.py
```

: `34/34 required files found`

### 14.3 API Test

```bash
curl -X POST -F "file=@brain_Tumor_Types/glioma/image(1).jpg" \
     http://localhost:5000/api/classify
```

### 14.4 Sample Images

```bash
ls brain_Tumor_Types/glioma/ | head -5
ls brain_Tumor_Types/meningioma/ | head -5
ls brain_Tumor_Types/notumor/ | head -5
ls brain_Tumor_Types/pituitary/ | head -5
```

### 14.5 TensorFlow + Spark Integration Test

```python
python3 << 'EOF'
from pyspark.sql import SparkSession
import tensorflow as tf

spark = SparkSession.builder.appName("Test").master("local[*]").getOrCreate()
data = [1, 2, 3, 4, 5]
rdd = spark.sparkContext.parallelize(data)
print(f"Spark sum: {rdd.sum()}")

tensor = tf.constant([1, 2, 3, 4, 5])
print(f"TensorFlow sum: {tf.reduce_sum(tensor).numpy()}")

print("Integration working!")
spark.stop()
EOF
```

---

## 15. Stopping Services

### 15.1 Stop Web Application

```bash
# Frontend: Ctrl+C in npm terminal
# Backend: Ctrl+C in Flask terminal
```

### 15.2 Stop Hadoop

```bash
stop-dfs.sh
jps  # Should only show 'Jps'
```

> **Important:** Always stop Hadoop properly with `stop-dfs.sh`. Force-killing can corrupt HDFS data.

### 15.3 Deactivate Environment

```bash
deactivate
```

### 15.4 Kill Background Processes

```bash
pkill -f "python app.py"
kill $(lsof -t -i:5000)
kill $(lsof -t -i:3000)
```

---

## 16. Quick Reference Card

```
+----------------------------------------------------------------+
|                    QUICK REFERENCE                              |
+----------------------------------------------------------------+
|                                                                 |
|  START HADOOP                                                   |
|  -------------                                                  |
|  start-dfs.sh                                                   |
|  jps  # Verify NameNode, DataNode running                       |
|                                                                 |
|  START WEB APPLICATION                                          |
|  ---------------------                                          |
|  cd webapp && ./start.sh                                        |
|                                                                 |
|  MANUAL START                                                   |
|  ------------                                                   |
|  Backend:  cd webapp/backend && python app.py                   |
|  Frontend: cd webapp/frontend && npm start                      |
|                                                                 |
|  START JUPYTER                                                  |
|  -------------                                                  |
|  source dtsvenv/bin/activate                                    |
|  jupyter notebook                                               |
|                                                                 |
|  URLS                                                           |
|  ----                                                           |
|  Frontend:    http://localhost:3000                             |
|  Backend:     http://localhost:5000                             |
|  Hadoop UI:   http://localhost:9870                             |
|  Spark UI:    http://localhost:4040                             |
|  Jupyter:     http://localhost:8888                             |
|                                                                 |
|  TESTING                                                        |
|  -------                                                        |
|  python webapp/test_integration.py                              |
|  python webapp/verify_structure.py                              |
|                                                                 |
|  STOP ALL                                                       |
|  --------                                                       |
|  Ctrl+C in terminals                                            |
|  stop-dfs.sh                                                    |
|  deactivate                                                     |
|                                                                 |
|  TROUBLESHOOTING                                                |
|  ---------------                                                |
|  Check logs:  cat webapp/backend/server.log                     |
|  Check ports: lsof -i :5000 -i :3000 -i :9000                   |
|  Check HDFS:  hdfs dfs -ls /medical_imaging/                    |
|                                                                 |
|  DIAGNOSTIC COMMANDS                                            |
|  -------------------                                            |
|  jps                          # Check Java processes            |
|  hdfs dfsadmin -report        # HDFS health                     |
|  pip list | grep tensorflow   # Package versions                |
|                                                                 |
+----------------------------------------------------------------+
```

---

## Useful Resources

| Resource | URL |
|----------|-----|
| Apache Spark Docs | https://spark.apache.org/docs/latest/ |
| Apache Hadoop Docs | https://hadoop.apache.org/docs/stable/ |
| TensorFlow Docs | https://www.tensorflow.org/api_docs |
| React Docs | https://react.dev/ |
| Flask Docs | https://flask.palletsprojects.com/ |

---

## Setup Complete!

You've successfully set up:
- Python virtual environment with all dependencies
- Apache Hadoop with HDFS
- Apache Spark configured for 8GB RAM
- TensorFlow integrated with Spark
- Dataset uploaded to HDFS
- Jupyter Notebook ready to run
- Web application ready to deploy

**Next Steps:**
1. Train model: Open `Brain_MRI_Distributed_DL.ipynb` and run
2. Run webapp: `cd webapp && ./start.sh`
3. Classify images at http://localhost:3000

---

*Document prepared for DTS 301 - Big Data Computing*
*Group 16*
