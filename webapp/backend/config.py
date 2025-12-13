"""
Configuration settings for Brain Tumor Classification Backend.

This module contains all configuration constants used throughout the backend,
including model paths, Spark settings, HDFS configuration, and class mappings.
"""

import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Get the absolute path to the project root (two levels up from this file)
BACKEND_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = BACKEND_DIR.parent.parent
MODEL_DIR = PROJECT_ROOT

# Model file path
MODEL_PATH = str(MODEL_DIR / "best_model_stage1.keras")

# Temporary upload directory for processing images
TEMP_UPLOAD_DIR = str(BACKEND_DIR / "temp_uploads")

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Input image dimensions (must match training configuration)
IMAGE_SIZE = (224, 224)

# Number of color channels (RGB = 3)
NUM_CHANNELS = 3

# Class mappings (index to name) - MUST match training order
CLASS_NAMES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

# Reverse mapping (name to index)
CLASS_INDICES = {name: idx for idx, name in CLASS_NAMES.items()}

# Number of classes
NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# SPARK CONFIGURATION
# =============================================================================

# Spark application name
SPARK_APP_NAME = "BrainTumorClassification-Backend"

# Spark master URL (local mode with all available cores)
SPARK_MASTER = "local[*]"

# Spark executor memory (adjust based on available system RAM)
SPARK_EXECUTOR_MEMORY = "2g"

# Spark driver memory
SPARK_DRIVER_MEMORY = "2g"

# =============================================================================
# HDFS CONFIGURATION
# =============================================================================

# HDFS namenode URL
HDFS_NAMENODE = "localhost"
HDFS_PORT = 9000
HDFS_URL = f"hdfs://{HDFS_NAMENODE}:{HDFS_PORT}"

# HDFS temporary directory for image processing
HDFS_TEMP_DIR = "/tmp/brain_tumor_webapp"

# =============================================================================
# FLASK CONFIGURATION
# =============================================================================

# Flask server settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

# CORS settings (allowed origins for frontend)
CORS_ORIGINS = [
    "http://localhost:3000",      # React development server
    "http://127.0.0.1:3000",
    "http://localhost:5000",      # Same-origin requests
]

# =============================================================================
# FILE UPLOAD CONFIGURATION
# =============================================================================

# Maximum allowed file size (16 MB)
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB in bytes

# Allowed image extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

# Allowed MIME types
ALLOWED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/bmp",
    "image/webp"
}

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

# Confidence threshold for predictions (optional, for filtering low-confidence results)
CONFIDENCE_THRESHOLD = 0.0  # 0.0 means no filtering

# Whether to return all class probabilities or just the top prediction
RETURN_ALL_PROBABILITIES = True

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# ENVIRONMENT DETECTION
# =============================================================================

def get_environment():
    """Detect the current running environment."""
    env = os.environ.get("FLASK_ENV", "development")
    return env

def is_production():
    """Check if running in production environment."""
    return get_environment() == "production"

def is_development():
    """Check if running in development environment."""
    return get_environment() == "development"

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config():
    """
    Validate that all required configuration is present and correct.
    
    Returns:
        tuple: (is_valid, list of error messages)
    """
    errors = []
    
    # Check model file exists
    if not os.path.exists(MODEL_PATH):
        errors.append(f"Model file not found: {MODEL_PATH}")
    
    # Check temp directory exists or can be created
    if not os.path.exists(TEMP_UPLOAD_DIR):
        try:
            os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create temp directory: {e}")
    
    # Validate class mappings
    if len(CLASS_NAMES) != NUM_CLASSES:
        errors.append(f"Class names count mismatch: {len(CLASS_NAMES)} vs {NUM_CLASSES}")
    
    return len(errors) == 0, errors


# Print configuration summary when module is loaded directly
if __name__ == "__main__":
    print("=" * 60)
    print("Brain Tumor Classification Backend Configuration")
    print("=" * 60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Model Exists: {os.path.exists(MODEL_PATH)}")
    print(f"\nImage Size: {IMAGE_SIZE}")
    print(f"Classes: {list(CLASS_NAMES.values())}")
    print(f"\nSpark Master: {SPARK_MASTER}")
    print(f"HDFS URL: {HDFS_URL}")
    print(f"\nFlask: {FLASK_HOST}:{FLASK_PORT}")
    print(f"Debug Mode: {FLASK_DEBUG}")
    print(f"Environment: {get_environment()}")
    
    # Validate configuration
    is_valid, errors = validate_config()
    print(f"\nConfiguration Valid: {is_valid}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
