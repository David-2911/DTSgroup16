"""
Image Preprocessing for Brain Tumor Classification.

This module handles Spark-based image preprocessing to maintain consistency
with the training pipeline. Images are processed using the same transformations
applied during model training.
"""

import logging
import os
import uuid
import numpy as np
from typing import Optional, Tuple, Union
from pathlib import Path
from io import BytesIO

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BinaryType

from config import (
    SPARK_APP_NAME, SPARK_MASTER, SPARK_EXECUTOR_MEMORY, SPARK_DRIVER_MEMORY,
    HDFS_URL, HDFS_TEMP_DIR, IMAGE_SIZE, TEMP_UPLOAD_DIR
)

# Configure logging
logger = logging.getLogger(__name__)


class SparkSessionManager:
    """
    Singleton class for managing the Spark session.
    
    Ensures only one Spark session exists throughout the application lifetime.
    """
    
    _instance = None
    _spark = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SparkSessionManager, cls).__new__(cls)
        return cls._instance
    
    def get_or_create_session(self) -> SparkSession:
        """
        Get existing Spark session or create a new one.
        
        Returns:
            SparkSession: Active Spark session
        """
        if self._spark is None or self._spark._jsc is None:
            logger.info("Creating new Spark session...")
            
            self._spark = SparkSession.builder \
                .appName(SPARK_APP_NAME) \
                .master(SPARK_MASTER) \
                .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY) \
                .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
                .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
                .getOrCreate()
            
            # Set log level to reduce verbosity
            self._spark.sparkContext.setLogLevel("WARN")
            
            logger.info(f"Spark session created: {self._spark.sparkContext.applicationId}")
            
        return self._spark
    
    def stop_session(self):
        """Stop the Spark session."""
        if self._spark is not None:
            logger.info("Stopping Spark session...")
            self._spark.stop()
            self._spark = None
            logger.info("Spark session stopped")
    
    @property
    def is_active(self) -> bool:
        """Check if Spark session is active."""
        return self._spark is not None and self._spark._jsc is not None


class ImagePreprocessor:
    """
    Spark-based image preprocessor for brain tumor classification.
    
    This class processes images using Apache Spark to maintain consistency
    with the training pipeline. It handles:
    - Image loading and validation
    - Resizing to model input dimensions (224x224)
    - Normalization to [0, 1] range
    - Optional HDFS storage for distributed processing
    
    Attributes:
        spark: SparkSession instance
        image_size: Target image dimensions (width, height)
    """
    
    def __init__(self, use_hdfs: bool = False):
        """
        Initialize the image preprocessor.
        
        Args:
            use_hdfs: Whether to use HDFS for temporary storage (default: False)
                     Set to True in production distributed environment
        """
        self.spark_manager = SparkSessionManager()
        self.image_size = IMAGE_SIZE
        self.use_hdfs = use_hdfs
        self.hdfs_url = HDFS_URL
        self.hdfs_temp_dir = HDFS_TEMP_DIR
        self.local_temp_dir = TEMP_UPLOAD_DIR
        
        # Ensure local temp directory exists
        os.makedirs(self.local_temp_dir, exist_ok=True)
    
    @property
    def spark(self) -> SparkSession:
        """Get the Spark session."""
        return self.spark_manager.get_or_create_session()
    
    def preprocess_image(self, image_data: bytes, filename: str = None) -> np.ndarray:
        """
        Preprocess a single image for model inference.
        
        This method replicates the preprocessing done during training:
        1. Load image from bytes
        2. Resize to 224x224
        3. Normalize pixel values to [0, 1]
        
        Args:
            image_data: Raw image bytes
            filename: Original filename (optional, for logging)
            
        Returns:
            np.ndarray: Preprocessed image array of shape (224, 224, 3)
                       with values in [0, 1] range
                       
        Raises:
            ValueError: If image cannot be processed
        """
        try:
            logger.info(f"Preprocessing image: {filename or 'unknown'}")
            
            if self.use_hdfs:
                # Use HDFS-based preprocessing (distributed)
                return self._preprocess_with_hdfs(image_data, filename)
            else:
                # Use local Spark preprocessing
                return self._preprocess_local(image_data, filename)
                
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise ValueError(f"Failed to preprocess image: {e}")
    
    def _preprocess_local(self, image_data: bytes, filename: str = None) -> np.ndarray:
        """
        Preprocess image using local Spark processing.
        
        Args:
            image_data: Raw image bytes
            filename: Original filename (optional)
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        # Use PIL for image processing (same as training)
        from PIL import Image
        
        # Load image from bytes
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
        if image.mode != 'RGB':
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Log original size
        original_size = image.size
        logger.debug(f"Original image size: {original_size}")
        
        # Resize to target dimensions (224x224)
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        logger.debug(f"Resized to: {image.size}")
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1] range (same as training: rescale=1./255)
        image_array = image_array / 255.0
        
        logger.info(f"Preprocessing complete. Shape: {image_array.shape}, Range: [{image_array.min():.3f}, {image_array.max():.3f}]")
        
        return image_array
    
    def _preprocess_with_hdfs(self, image_data: bytes, filename: str = None) -> np.ndarray:
        """
        Preprocess image using HDFS for temporary storage.
        
        This method is used in distributed environments where the image
        needs to be accessible across Spark workers.
        
        Args:
            image_data: Raw image bytes
            filename: Original filename (optional)
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        import subprocess
        from PIL import Image
        
        # Generate unique filename for temporary storage
        unique_id = str(uuid.uuid4())
        ext = Path(filename).suffix if filename else '.jpg'
        temp_filename = f"temp_image_{unique_id}{ext}"
        
        # Paths
        local_path = os.path.join(self.local_temp_dir, temp_filename)
        hdfs_path = f"{self.hdfs_temp_dir}/{temp_filename}"
        
        try:
            # Save to local temp directory
            with open(local_path, 'wb') as f:
                f.write(image_data)
            logger.debug(f"Saved to local temp: {local_path}")
            
            # Upload to HDFS
            hdfs_full_path = f"{self.hdfs_url}{hdfs_path}"
            
            # Create HDFS directory if it doesn't exist
            subprocess.run(
                ["hdfs", "dfs", "-mkdir", "-p", self.hdfs_temp_dir],
                capture_output=True, check=False
            )
            
            # Upload file to HDFS
            result = subprocess.run(
                ["hdfs", "dfs", "-put", "-f", local_path, hdfs_path],
                capture_output=True, check=True
            )
            logger.debug(f"Uploaded to HDFS: {hdfs_path}")
            
            # Use Spark to read the image
            # Note: This demonstrates Spark-based processing, but for single images,
            # direct PIL processing is more efficient
            df = self.spark.read.format("binaryFile") \
                .load(hdfs_full_path) \
                .select("content")
            
            # Get the image bytes back
            row = df.first()
            image_bytes = row.content
            
            # Process with PIL (same as local)
            image = Image.open(BytesIO(image_bytes))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0
            
            logger.info(f"HDFS preprocessing complete. Shape: {image_array.shape}")
            
            return image_array
            
        finally:
            # Cleanup: Remove temporary files
            self._cleanup_temp_files(local_path, hdfs_path)
    
    def _cleanup_temp_files(self, local_path: str = None, hdfs_path: str = None):
        """
        Clean up temporary files after processing.
        
        Args:
            local_path: Path to local temporary file
            hdfs_path: Path to HDFS temporary file
        """
        import subprocess
        
        # Remove local file
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
                logger.debug(f"Removed local temp file: {local_path}")
            except Exception as e:
                logger.warning(f"Failed to remove local temp file: {e}")
        
        # Remove HDFS file
        if hdfs_path:
            try:
                subprocess.run(
                    ["hdfs", "dfs", "-rm", "-f", hdfs_path],
                    capture_output=True, check=False
                )
                logger.debug(f"Removed HDFS temp file: {hdfs_path}")
            except Exception as e:
                logger.warning(f"Failed to remove HDFS temp file: {e}")
    
    def validate_image_format(self, image_data: bytes) -> Tuple[bool, str]:
        """
        Validate that the image data is a valid image format.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        from PIL import Image
        
        try:
            image = Image.open(BytesIO(image_data))
            image.verify()  # Verify image integrity
            
            # Reopen for format check (verify() makes image unusable)
            image = Image.open(BytesIO(image_data))
            
            return True, f"Valid {image.format} image ({image.size[0]}x{image.size[1]})"
            
        except Exception as e:
            return False, f"Invalid image format: {e}"
    
    def get_image_info(self, image_data: bytes) -> dict:
        """
        Get information about an image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            dict: Image information including format, size, mode
        """
        from PIL import Image
        
        try:
            image = Image.open(BytesIO(image_data))
            
            return {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.size[0],
                "height": image.size[1],
                "bytes": len(image_data)
            }
        except Exception as e:
            return {"error": str(e)}


# Create global preprocessor instance
_preprocessor = None


def get_preprocessor(use_hdfs: bool = False) -> ImagePreprocessor:
    """
    Get the global image preprocessor instance.
    
    Args:
        use_hdfs: Whether to use HDFS for temporary storage
        
    Returns:
        ImagePreprocessor: The preprocessor instance
    """
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = ImagePreprocessor(use_hdfs=use_hdfs)
    return _preprocessor


def preprocess_image(image_data: bytes, filename: str = None) -> np.ndarray:
    """
    Convenience function to preprocess an image.
    
    Args:
        image_data: Raw image bytes
        filename: Original filename (optional)
        
    Returns:
        np.ndarray: Preprocessed image array
    """
    preprocessor = get_preprocessor()
    return preprocessor.preprocess_image(image_data, filename)


def get_spark_session() -> SparkSession:
    """
    Get the active Spark session.
    
    Returns:
        SparkSession: The active Spark session
    """
    manager = SparkSessionManager()
    return manager.get_or_create_session()


def stop_spark_session():
    """Stop the Spark session."""
    manager = SparkSessionManager()
    manager.stop_session()


# Test preprocessing when run directly
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("=" * 60)
    print("Testing Image Preprocessor")
    print("=" * 60)
    
    try:
        # Initialize preprocessor
        print("\n1. Initializing preprocessor...")
        preprocessor = ImagePreprocessor(use_hdfs=False)
        print(f"   Preprocessor initialized")
        print(f"   Image size: {preprocessor.image_size}")
        print(f"   Use HDFS: {preprocessor.use_hdfs}")
        
        # Test Spark session
        print("\n2. Testing Spark session...")
        spark = preprocessor.spark
        print(f"   Spark version: {spark.version}")
        print(f"   App ID: {spark.sparkContext.applicationId}")
        
        # Create a test image
        print("\n3. Creating test image...")
        from PIL import Image
        
        # Create a simple test image (random colors)
        test_image = Image.new('RGB', (512, 512), color='red')
        
        # Save to bytes
        img_buffer = BytesIO()
        test_image.save(img_buffer, format='JPEG')
        test_bytes = img_buffer.getvalue()
        print(f"   Test image size: {len(test_bytes)} bytes")
        
        # Validate image
        print("\n4. Validating image...")
        is_valid, message = preprocessor.validate_image_format(test_bytes)
        print(f"   Valid: {is_valid}")
        print(f"   Message: {message}")
        
        # Get image info
        print("\n5. Getting image info...")
        info = preprocessor.get_image_info(test_bytes)
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Preprocess image
        print("\n6. Preprocessing image...")
        result = preprocessor.preprocess_image(test_bytes, "test_image.jpg")
        print(f"   Output shape: {result.shape}")
        print(f"   Output dtype: {result.dtype}")
        print(f"   Value range: [{result.min():.3f}, {result.max():.3f}]")
        
        print("\n" + "=" * 60)
        print("Image Preprocessor Test: SUCCESS")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("Image Preprocessor Test: FAILED")
        print("=" * 60)
        sys.exit(1)
    finally:
        # Stop Spark session
        try:
            stop_spark_session()
        except:
            pass
