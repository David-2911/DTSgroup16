# Brain MRI Classification System - Code Walkthrough
## DTS 301 - Big Data Computing | Group 16

---

# Table of Contents

1. [Project Structure Overview](#1-project-structure-overview)
2. [Jupyter Notebook - Model Training](#2-jupyter-notebook---model-training)
3. [Backend Code Walkthrough](#3-backend-code-walkthrough)
4. [Frontend Code Walkthrough](#4-frontend-code-walkthrough)
5. [Data Flow End-to-End](#5-data-flow-end-to-end)
6. [Key Algorithms Explained](#6-key-algorithms-explained)
7. [Configuration Files](#7-configuration-files)

---

# 1. Project Structure Overview

```
DTSgroup16/
│
├── best_model_stage1.keras          # Trained TensorFlow model (105 MB)
├── Brain_MRI_Distributed_DL.ipynb   # Jupyter notebook for training
├── requirements.txt                  # Python dependencies
│
├── brain_Tumor_Types/                # MRI Dataset (137 MB)
│   ├── glioma/                       # 1426 images
│   ├── meningioma/                   # 1339 images
│   ├── notumor/                      # 1595 images
│   └── pituitary/                    # 1352 images
│
├── docs/                             # Documentation files
│
└── webapp/                           # Web application
    ├── backend/                      # Flask API (Python)
    │   ├── app.py                    # Main server
    │   ├── model_loader.py           # TensorFlow model management
    │   ├── preprocessing.py          # Spark image processing
    │   ├── gradcam.py                # Visualization
    │   ├── analysis_generator.py     # Medical text generation
    │   ├── config.py                 # Settings
    │   └── utils.py                  # Helper functions
    │
    ├── frontend/                     # React application
    │   └── src/
    │       ├── App.jsx               # Main component
    │       ├── components/           # 7 React components
    │       ├── styles/               # 8 CSS files
    │       └── utils/api.js          # Backend communication
    │
    └── shared/
        └── tumor_descriptions.json   # Medical descriptions
```

---

# 2. Jupyter Notebook - Model Training

**File:** `Brain_MRI_Distributed_DL.ipynb`

The notebook contains the complete training pipeline. Here's a walkthrough of each phase:

## Phase 1: Environment Setup

```python
# Import all required libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from pyspark.sql import SparkSession

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
```

**What this does:**
- Imports libraries for deep learning (TensorFlow), data processing (NumPy), and distributed computing (Spark)
- Checks system configuration

## Phase 2: Hadoop & Spark Configuration

```python
# Create Spark session for distributed processing
spark = SparkSession.builder \
    .appName("Brain_MRI_Classification") \
    .master("local[*]") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Verify HDFS connection (optional)
hdfs_available = check_hdfs_connection()
```

**What this does:**
- Creates a Spark session that uses all available CPU cores (`local[*]`)
- Configures memory allocation for Spark executors
- Optionally connects to HDFS for distributed storage

## Phase 3: Data Loading with Spark

```python
def load_brain_mri_dataset(data_dir, spark):
    """
    Load MRI images using Spark for distributed processing.
    """
    data = []
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            data.append((img_path, class_idx, class_name))
    
    # Create Spark DataFrame for distributed processing
    df = spark.createDataFrame(data, ['path', 'label', 'class_name'])
    return df

# Load dataset
dataset_df = load_brain_mri_dataset('./brain_Tumor_Types', spark)
print(f"Total images: {dataset_df.count()}")
```

**What this does:**
- Iterates through each tumor class directory
- Creates (path, label, class_name) tuples for each image
- Converts to Spark DataFrame for distributed operations

## Phase 4: Image Preprocessing

```python
from PIL import Image
import numpy as np

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for the model.
    
    Steps:
    1. Load image from disk
    2. Convert to RGB (in case of grayscale)
    3. Resize to 224x224 (ResNet50 input size)
    4. Convert to numpy array
    5. Normalize pixel values to [0, 1]
    """
    # Load and convert
    img = Image.open(image_path)
    img = img.convert('RGB')  # Ensure 3 channels
    
    # Resize
    img = img.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0  # Scale from [0,255] to [0,1]
    
    return img_array

# Process all images in parallel using Spark RDD
def process_partition(partition):
    """Process a partition of images."""
    for row in partition:
        img = preprocess_image(row.path)
        yield (img, row.label)

# Apply preprocessing across Spark partitions
processed_rdd = dataset_df.rdd.mapPartitions(process_partition)
```

**What this does:**
- Loads each image using PIL
- Converts to RGB (some MRI images might be grayscale)
- Resizes to 224x224 pixels (required by ResNet50)
- Normalizes pixel values from 0-255 to 0-1

## Phase 5: Model Building

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

def create_model(num_classes=4):
    """
    Create a transfer learning model based on ResNet50.
    
    Architecture:
    1. ResNet50 base (pretrained on ImageNet)
    2. Global Average Pooling
    3. Dense layer (256 units, ReLU activation)
    4. Dropout (50% rate)
    5. Output layer (4 units, softmax activation)
    """
    # Load ResNet50 without the top classification layer
    base_model = ResNet50(
        weights='imagenet',      # Use ImageNet pretrained weights
        include_top=False,       # Don't include classification layer
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers (don't train them)
    base_model.trainable = False
    
    # Create the full model
    model = Sequential([
        base_model,                           # Feature extraction
        GlobalAveragePooling2D(),             # Reduce spatial dimensions
        Dense(256, activation='relu'),        # Learn tumor-specific patterns
        Dropout(0.5),                         # Prevent overfitting
        Dense(num_classes, activation='softmax')  # Output probabilities
    ])
    
    return model

model = create_model(num_classes=4)
model.summary()
```

**What each layer does:**
- `ResNet50`: Pre-trained CNN that extracts visual features
- `GlobalAveragePooling2D`: Reduces 7x7x2048 feature maps to 2048 values
- `Dense(256)`: Learns tumor-specific patterns from features
- `Dropout(0.5)`: Randomly disables 50% of neurons during training (prevents overfitting)
- `Dense(4, softmax)`: Outputs probability for each of 4 classes

## Phase 6: Model Training

```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Compile model
model.compile(
    optimizer='adam',                    # Adaptive learning rate optimizer
    loss='sparse_categorical_crossentropy',  # Loss for integer labels
    metrics=['accuracy']
)

# Define callbacks
callbacks = [
    # Save best model during training
    ModelCheckpoint(
        'best_model_stage1.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    # Stop early if no improvement
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,              # Wait 5 epochs before stopping
        restore_best_weights=True
    ),
    # Reduce learning rate if stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,              # Multiply LR by 0.2
        patience=3
    )
]

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=callbacks
)
```

**What this does:**
- Compiles model with Adam optimizer and cross-entropy loss
- Sets up callbacks for model saving, early stopping, and learning rate adjustment
- Trains for up to 20 epochs with validation monitoring

---

# 3. Backend Code Walkthrough

## 3.1 Main Application (app.py)

```python
"""
Flask Application for Brain Tumor Classification.
Entry point for the backend API.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

# Create Flask app
app = Flask(__name__)

# Enable CORS for frontend access
CORS(app, origins=["http://localhost:3000"])

# Global service instances
model_loader = None
preprocessor = None
gradcam = None
analysis_generator = None

def initialize_services():
    """
    Initialize all services at startup.
    Called once when server starts.
    """
    global model_loader, preprocessor, gradcam, analysis_generator
    
    # Initialize Spark session
    spark_manager = SparkSessionManager()
    spark_manager.get_or_create_session()
    
    # Initialize image preprocessor
    preprocessor = ImagePreprocessor()
    
    # Load TensorFlow model
    model_loader = ModelLoader()
    model_loader.load_model()
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model_loader.model)
    
    # Initialize analysis generator
    analysis_generator = AnalysisGenerator()
```

**What this does:**
- Creates Flask application
- Enables Cross-Origin Resource Sharing for the React frontend
- Initializes all services (Spark, model, Grad-CAM) at startup

### Classification Endpoint

```python
@app.route('/api/classify', methods=['POST'])
def classify_image():
    """
    Main endpoint for MRI classification.
    
    Input: POST request with 'file' containing image
    Output: JSON with prediction, probabilities, visualization, analysis
    """
    # Step 1: Validate the request has a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Step 2: Validate file type
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Step 3: Read image bytes
    image_bytes = file.read()
    
    # Step 4: Preprocess with Spark
    preprocessed = preprocessor.preprocess_single(image_bytes)
    
    # Step 5: Run model inference
    predictions = model_loader.predict(preprocessed)
    
    # Step 6: Get top prediction
    class_idx = np.argmax(predictions)
    confidence = float(predictions[class_idx]) * 100
    class_name = CLASS_NAMES[class_idx]
    
    # Step 7: Generate Grad-CAM visualization
    heatmap = gradcam.generate_heatmap(preprocessed, class_idx)
    overlay = gradcam.create_overlay(image_bytes, heatmap)
    
    # Step 8: Generate medical analysis
    analysis = analysis_generator.generate_analysis(
        class_name, confidence, 
        dict(zip(CLASS_NAMES, (predictions * 100).tolist()))
    )
    
    # Step 9: Build and return response
    return jsonify({
        'status': 'success',
        'prediction': {
            'class': class_name,
            'confidence': confidence
        },
        'probabilities': dict(zip(CLASS_NAMES, (predictions * 100).tolist())),
        'visualization': {
            'original_image': encode_image_base64(image_bytes),
            'heatmap_overlay': encode_image_base64(overlay)
        },
        'analysis': analysis
    })
```

## 3.2 Model Loader (model_loader.py)

```python
"""
Manages TensorFlow model loading and inference.
Uses singleton pattern - only one model instance in memory.
"""

class ModelLoader:
    _instance = None  # Singleton instance
    _model = None     # Loaded model
    _is_loaded = False
    
    def __new__(cls):
        """
        Singleton pattern - ensures only one model in memory.
        First call creates instance, subsequent calls return same instance.
        """
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self):
        """
        Load the trained Keras model from disk.
        Only loads once - subsequent calls skip loading.
        """
        if self._is_loaded:
            return True  # Already loaded
        
        # Load model without compilation (inference only)
        self._model = tf.keras.models.load_model(
            'best_model_stage1.keras',
            compile=False
        )
        self._is_loaded = True
        return True
    
    def predict(self, image_array):
        """
        Run inference on preprocessed image.
        
        Args:
            image_array: Numpy array of shape (224, 224, 3)
        
        Returns:
            Numpy array of probabilities for each class
        """
        # Add batch dimension: (224,224,3) -> (1,224,224,3)
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        # Run prediction
        predictions = self._model.predict(image_array, verbose=0)
        
        # Return probabilities (remove batch dimension)
        return predictions[0]
    
    def warmup(self):
        """
        Run a dummy prediction to warm up the model.
        First prediction is always slower - this ensures fast response
        for actual user requests.
        """
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        self._model.predict(dummy, verbose=0)
```

## 3.3 Image Preprocessor (preprocessing.py)

```python
"""
Spark-based image preprocessing.
Ensures consistency with training pipeline.
"""

class SparkSessionManager:
    """
    Singleton for Spark session management.
    Only one Spark session can exist per JVM.
    """
    _instance = None
    _spark = None
    
    def get_or_create_session(self):
        """Create or return existing Spark session."""
        if self._spark is None:
            self._spark = SparkSession.builder \
                .appName("Brain_MRI_Classification") \
                .master("local[*]") \
                .config("spark.executor.memory", "4g") \
                .getOrCreate()
            
            # Reduce logging verbosity
            self._spark.sparkContext.setLogLevel("WARN")
        
        return self._spark


class ImagePreprocessor:
    """
    Preprocesses images using Spark for consistency with training.
    """
    
    def __init__(self):
        """Initialize with Spark session."""
        manager = SparkSessionManager()
        self.spark = manager.get_or_create_session()
        self.image_size = (224, 224)
    
    def preprocess_single(self, image_bytes):
        """
        Preprocess a single image.
        
        Steps:
        1. Load bytes into PIL Image
        2. Convert to RGB
        3. Resize to 224x224
        4. Convert to numpy array
        5. Normalize to [0, 1]
        
        Args:
            image_bytes: Raw image bytes from upload
            
        Returns:
            Numpy array ready for model input
        """
        # Load image from bytes
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB (handles grayscale/RGBA)
        img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize(self.image_size)
        
        # Convert to numpy and normalize
        img_array = np.array(img) / 255.0
        
        return img_array.astype(np.float32)
```

## 3.4 Grad-CAM Visualization (gradcam.py)

```python
"""
Gradient-weighted Class Activation Mapping.
Shows which regions of the image influenced the prediction.
"""

class GradCAM:
    """
    Generate heatmaps showing model attention.
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize with trained model.
        
        Args:
            model: Keras model
            layer_name: Name of conv layer to visualize (auto-detected if None)
        """
        self.model = model
        
        # Find last convolutional layer
        if layer_name is None:
            self.layer_name = self._find_last_conv_layer()
        else:
            self.layer_name = layer_name
        
        # Create gradient model
        self.grad_model = self._create_grad_model()
    
    def _find_last_conv_layer(self):
        """
        Find the last convolutional layer in the model.
        This layer contains the most semantically rich features.
        """
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        
        # For Sequential models with nested Functional models
        # (like ResNet50 inside Sequential)
        for layer in self.model.layers:
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if 'conv' in sublayer.name.lower():
                        return sublayer.name
        
        raise ValueError("No convolutional layer found")
    
    def _create_grad_model(self):
        """
        Create a model that outputs both conv features and predictions.
        This allows computing gradients of predictions w.r.t. features.
        """
        # Get last conv layer output
        conv_layer = self._get_conv_layer()
        
        # Create model: input -> [conv_output, prediction]
        return tf.keras.Model(
            inputs=self.model.input,
            outputs=[conv_layer.output, self.model.output]
        )
    
    def generate_heatmap(self, image_array, class_idx):
        """
        Generate Grad-CAM heatmap for specified class.
        
        Algorithm:
        1. Forward pass to get conv features and predictions
        2. Compute gradients of class score w.r.t. conv features
        3. Weight each feature map by average gradient
        4. Sum weighted feature maps
        5. Apply ReLU (keep only positive influences)
        """
        # Add batch dimension
        img = np.expand_dims(image_array, axis=0)
        
        # Record operations for gradient computation
        with tf.GradientTape() as tape:
            # Forward pass
            conv_outputs, predictions = self.grad_model(img)
            
            # Get score for target class
            class_score = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(class_score, conv_outputs)
        
        # Global average pooling of gradients
        # Shape: (batch, height, width, channels) -> (channels,)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradient importance
        conv_outputs = conv_outputs[0]  # Remove batch dimension
        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads,
            axis=-1
        )
        
        # Apply ReLU
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize to [0, 1]
        heatmap = heatmap / tf.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def create_overlay(self, original_image_bytes, heatmap):
        """
        Overlay heatmap on original image.
        
        Args:
            original_image_bytes: Raw image bytes
            heatmap: Numpy array from generate_heatmap
            
        Returns:
            PIL Image with colored heatmap overlay
        """
        # Load original image
        original = Image.open(BytesIO(original_image_bytes))
        original = original.convert('RGB')
        original = original.resize((224, 224))
        
        # Resize heatmap to image size
        heatmap_resized = Image.fromarray(
            (heatmap * 255).astype(np.uint8)
        )
        heatmap_resized = heatmap_resized.resize((224, 224))
        
        # Apply colormap (blue->green->red)
        import matplotlib.cm as cm
        colormap = cm.get_cmap('jet')
        heatmap_colored = colormap(np.array(heatmap_resized) / 255.0)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap_colored)
        
        # Blend original and heatmap
        overlay = Image.blend(original, heatmap_pil, alpha=0.4)
        
        return overlay
```

## 3.5 Analysis Generator (analysis_generator.py)

```python
"""
Generate educational medical analysis for predictions.
NOT for clinical diagnosis - educational purposes only.
"""

class AnalysisGenerator:
    """
    Generate comprehensive explanations for classifications.
    """
    
    def __init__(self, descriptions_path=None):
        """Load tumor descriptions from JSON file."""
        if descriptions_path is None:
            descriptions_path = '../shared/tumor_descriptions.json'
        
        with open(descriptions_path) as f:
            self.descriptions = json.load(f)
    
    def generate_analysis(self, predicted_class, confidence, probabilities):
        """
        Generate detailed analysis for a prediction.
        
        Args:
            predicted_class: Predicted tumor type
            confidence: Confidence percentage (0-100)
            probabilities: Dict of all class probabilities
            
        Returns:
            dict with description, interpretation, recommendations
        """
        tumor_info = self.descriptions.get(predicted_class, {})
        
        analysis = {
            'classification': predicted_class,
            'confidence_level': self._interpret_confidence(confidence),
            'description': tumor_info.get('short_description', ''),
            'characteristics': tumor_info.get('characteristics', []),
            'model_interpretation': self._generate_interpretation(
                predicted_class, confidence, probabilities
            ),
            'disclaimer': (
                "This is an educational tool with ~92% accuracy. "
                "NOT a substitute for professional medical diagnosis."
            )
        }
        
        # Add differential diagnosis for uncertain predictions
        if confidence < 90:
            analysis['alternatives'] = [
                {'class': cls, 'probability': prob}
                for cls, prob in probabilities.items()
                if prob > 10 and cls != predicted_class
            ]
        
        return analysis
    
    def _interpret_confidence(self, confidence):
        """Convert confidence score to human-readable interpretation."""
        if confidence >= 95:
            return "Very High - Model is very confident"
        elif confidence >= 85:
            return "High - Strong confidence"
        elif confidence >= 75:
            return "Moderate - Reasonably confident"
        elif confidence >= 60:
            return "Low - Some uncertainty"
        else:
            return "Very Low - Significant uncertainty"
```

---

# 4. Frontend Code Walkthrough

## 4.1 Main Application (App.jsx)

```jsx
/**
 * Main React Application Component
 * Manages state and coordinates all child components
 */
import React, { useState, useEffect } from 'react';

function App() {
  // State management
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isBackendConnected, setIsBackendConnected] = useState(false);

  // Check backend health on mount and every 30 seconds
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await checkHealth();
        setIsBackendConnected(true);
      } catch (err) {
        setIsBackendConnected(false);
      }
    };
    
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Handle image selection
  const handleImageSelect = (file) => {
    setSelectedImage(file);
    setResult(null);
    setError(null);
    
    // Create preview URL
    const reader = new FileReader();
    reader.onloadend = () => setImagePreview(reader.result);
    reader.readAsDataURL(file);
  };

  // Handle classification
  const handleClassify = async () => {
    if (!selectedImage) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await classifyImage(selectedImage);
      setResult(response);
    } catch (err) {
      setError({ type: 'error', message: err.message });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <Header isConnected={isBackendConnected} />
      
      <main>
        {error && <ErrorMessage {...error} />}
        
        {isLoading && <LoadingSpinner />}
        
        {!isLoading && !result && (
          <>
            <ImageUpload 
              onImageSelect={handleImageSelect}
              preview={imagePreview}
            />
            {selectedImage && (
              <button onClick={handleClassify}>
                Analyze MRI Scan
              </button>
            )}
          </>
        )}
        
        {!isLoading && result && (
          <>
            <ClassificationResult prediction={result.prediction} />
            <HeatmapVisualization visualization={result.visualization} />
            <MedicalAnalysis analysis={result.analysis} />
            <button onClick={() => setResult(null)}>
              Analyze New Image
            </button>
          </>
        )}
      </main>
    </div>
  );
}
```

**State Variables Explained:**
| State | Purpose |
|-------|---------|
| `selectedImage` | File object of uploaded image |
| `imagePreview` | Base64 data URL for preview |
| `isLoading` | Boolean for loading spinner |
| `result` | Classification result from API |
| `error` | Error object for error display |
| `isBackendConnected` | Backend health status |

## 4.2 Image Upload Component

```jsx
/**
 * Drag-and-drop image upload with preview
 */
import { useDropzone } from 'react-dropzone';

const ImageUpload = ({ onImageSelect, preview }) => {
  // react-dropzone hook for drag-and-drop
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (files) => onImageSelect(files[0]),
    accept: { 'image/*': ['.jpg', '.jpeg', '.png'] },
    multiple: false,
    maxSize: 10 * 1024 * 1024  // 10MB
  });
  
  return (
    <div {...getRootProps()} className="dropzone">
      <input {...getInputProps()} />
      
      {!preview ? (
        <div>
          <p>{isDragActive ? 'Drop here' : 'Drag & drop or click'}</p>
          <p>Accepted: JPG, PNG (max 10MB)</p>
        </div>
      ) : (
        <img src={preview} alt="Preview" />
      )}
    </div>
  );
};
```

## 4.3 Classification Result Component

```jsx
/**
 * Displays prediction and confidence breakdown
 */
const ClassificationResult = ({ prediction }) => {
  const { class: tumorClass, confidence, probabilities } = prediction;
  
  // Sort probabilities highest to lowest
  const sortedProbs = Object.entries(probabilities)
    .sort(([,a], [,b]) => b - a);
  
  return (
    <div className="classification-result">
      <h2>Classification Result</h2>
      
      <div className="prediction">
        <p>Detected: <strong>{tumorClass}</strong></p>
        <p>Confidence: {confidence.toFixed(1)}%</p>
      </div>
      
      <div className="probabilities">
        <h4>All Probabilities:</h4>
        {sortedProbs.map(([cls, prob]) => (
          <div key={cls} className="prob-bar">
            <span>{cls}</span>
            <div style={{ width: `${prob}%` }}></div>
            <span>{prob.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
};
```

## 4.4 API Communication (api.js)

```javascript
/**
 * Backend API communication utilities
 */

const API_BASE = 'http://localhost:5000/api';

/**
 * Check backend health status
 */
export const checkHealth = async () => {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error('Backend unavailable');
  return response.json();
};

/**
 * Classify an MRI image
 * 
 * @param {File} imageFile - Image file to classify
 * @returns {Object} Classification result with prediction, probabilities, visualization
 */
export const classifyImage = async (imageFile) => {
  // Create FormData for file upload
  const formData = new FormData();
  formData.append('file', imageFile);
  
  // Send POST request
  const response = await fetch(`${API_BASE}/classify`, {
    method: 'POST',
    body: formData,
    // Note: Don't set Content-Type header - browser sets it automatically
    // with correct boundary for multipart/form-data
  });
  
  // Parse response
  const data = await response.json();
  
  if (!response.ok) {
    throw new Error(data.error || 'Classification failed');
  }
  
  return data;
};
```

---

# 5. Data Flow End-to-End

## Complete Request Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. USER ACTION                                                       │
│    User drags MRI image onto upload area                            │
│    └─> FileReader creates preview (base64)                          │
│    └─> File stored in state                                         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 2. CLICK "ANALYZE"                                                   │
│    App.handleClassify() called                                      │
│    └─> isLoading = true (shows spinner)                             │
│    └─> api.classifyImage(selectedImage)                             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 3. API REQUEST                                                       │
│    FormData with file sent to localhost:5000/api/classify           │
│    Headers: multipart/form-data                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 4. FLASK RECEIVES REQUEST                                            │
│    app.py: classify_image()                                         │
│    └─> Validates file exists                                        │
│    └─> Validates file type (jpg, png)                               │
│    └─> Reads file bytes                                             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 5. SPARK PREPROCESSING                                               │
│    preprocessing.py: ImagePreprocessor.preprocess_single()          │
│    └─> PIL.Image.open(BytesIO(bytes))                               │
│    └─> Convert to RGB                                               │
│    └─> Resize to (224, 224)                                         │
│    └─> np.array() / 255.0                                           │
│    Output: numpy array shape (224, 224, 3), values 0-1              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 6. MODEL INFERENCE                                                   │
│    model_loader.py: ModelLoader.predict()                           │
│    └─> Add batch dimension: (1, 224, 224, 3)                        │
│    └─> model.predict(image)                                         │
│    └─> Remove batch dimension                                       │
│    Output: [0.925, 0.042, 0.021, 0.012] (probabilities)             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 7. GRAD-CAM VISUALIZATION                                            │
│    gradcam.py: GradCAM.generate_heatmap()                           │
│    └─> Forward pass through grad_model                              │
│    └─> Compute gradients of class score                             │
│    └─> Weight and sum feature maps                                  │
│    └─> Apply ReLU and normalize                                     │
│    └─> Overlay on original image                                    │
│    Output: PIL Image with heatmap overlay                           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 8. ANALYSIS GENERATION                                               │
│    analysis_generator.py: generate_analysis()                       │
│    └─> Look up tumor info in JSON                                   │
│    └─> Interpret confidence level                                   │
│    └─> Generate model interpretation text                           │
│    └─> Add disclaimer                                               │
│    Output: dict with description, interpretation, disclaimer        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 9. RESPONSE ASSEMBLY                                                 │
│    app.py: jsonify({...})                                           │
│    └─> Convert images to base64                                     │
│    └─> Build JSON response                                          │
│    Output: HTTP 200 with JSON body                                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 10. FRONTEND RECEIVES RESPONSE                                       │
│     api.js: response.json()                                         │
│     └─> setResult(data)                                             │
│     └─> isLoading = false                                           │
│     └─> Result components render                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

# 6. Key Algorithms Explained

## 6.1 Softmax Function

Converts raw model outputs (logits) to probabilities that sum to 1.

```
Formula: softmax(x_i) = e^x_i / Σ(e^x_j)

Example:
Logits: [3.5, 1.2, 0.5, 0.1]

e^3.5 = 33.12    → 33.12 / 34.89 = 0.949 (94.9%)
e^1.2 = 3.32     → 3.32 / 34.89 = 0.095 (9.5%)
e^0.5 = 1.65     → 1.65 / 34.89 = 0.047 (4.7%)
e^0.1 = 1.11     → 1.11 / 34.89 = 0.032 (3.2%)
Sum = 34.89      Total: 100%
```

## 6.2 Cross-Entropy Loss

Measures how well predicted probabilities match true labels.

```
Formula: L = -Σ(y_true * log(y_pred))

For correct class:
- If prediction = 1.0 (perfect): loss = -log(1) = 0
- If prediction = 0.5: loss = -log(0.5) = 0.69
- If prediction = 0.1: loss = -log(0.1) = 2.30

Lower loss = better predictions
```

## 6.3 Gradient Descent

How the model learns by adjusting weights to reduce loss.

```
Algorithm:
1. Forward pass: compute predictions
2. Compute loss
3. Backward pass: compute gradients (∂loss/∂weights)
4. Update weights: w = w - learning_rate * gradient
5. Repeat

Adam optimizer adds momentum and adaptive learning rates
for faster, more stable training.
```

---

# 7. Configuration Files

## 7.1 Backend Configuration (config.py)

```python
"""
Central configuration for the backend.
All settings in one place for easy modification.
"""
import os

# Server settings
FLASK_HOST = '0.0.0.0'  # Listen on all interfaces
FLASK_PORT = 5000
FLASK_DEBUG = False     # Disable in production

# CORS settings
CORS_ORIGINS = ['http://localhost:3000']  # Allow frontend

# Model settings
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 
    '..', '..', 
    'best_model_stage1.keras'
)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = 4
IMAGE_SIZE = (224, 224)

# Spark settings
SPARK_APP_NAME = 'Brain_MRI_Classification'
SPARK_MASTER = 'local[*]'  # Use all cores
SPARK_EXECUTOR_MEMORY = '4g'
SPARK_DRIVER_MEMORY = '4g'

# HDFS settings (optional)
HDFS_URL = 'hdfs://localhost:9000'
HDFS_TEMP_DIR = '/tmp/brain_mri_uploads'

# Upload settings
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
TEMP_UPLOAD_DIR = os.path.join(
    os.path.dirname(__file__), 
    'temp_uploads'
)
```

## 7.2 Frontend Configuration (package.json)

```json
{
  "name": "brain-mri-classifier",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-dropzone": "^14.2.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test"
  },
  "proxy": "http://localhost:5000"
}
```

---

*Document prepared for DTS 301 - Big Data Computing*
*Group 16*
