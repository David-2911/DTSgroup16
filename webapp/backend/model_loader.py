"""
Model Loader for Brain Tumor Classification.

This module handles loading the trained Keras model and running inference.
The model is loaded once at application startup and reused for all predictions.
"""

import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional, Union
from pathlib import Path

from config import MODEL_PATH, CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE

# Configure logging
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton class for loading and managing the trained brain tumor classification model.
    
    The model is loaded once when the class is instantiated and kept in memory
    for efficient inference. This avoids the overhead of loading the model for
    each prediction request.
    
    Attributes:
        model: The loaded Keras model
        is_loaded: Boolean indicating if model is successfully loaded
        model_path: Path to the model file
    """
    
    _instance = None
    _model = None
    _is_loaded = False
    
    def __new__(cls):
        """Implement singleton pattern - only one model instance in memory."""
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the model loader."""
        self.model_path = MODEL_PATH
        self.class_names = CLASS_NAMES
        self.num_classes = NUM_CLASSES
        self.image_size = IMAGE_SIZE
        
    @property
    def model(self):
        """Get the loaded model, loading it if necessary."""
        if not self._is_loaded:
            self.load_model()
        return self._model
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def load_model(self) -> bool:
        """
        Load the trained Keras model from disk.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails for any other reason
        """
        if self._is_loaded:
            logger.info("Model already loaded, skipping reload")
            return True
            
        try:
            # Check if model file exists
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from: {self.model_path}")
            
            # Load the Keras model
            # Using compile=False since we only need inference, not training
            self._model = tf.keras.models.load_model(
                self.model_path,
                compile=False
            )
            
            self._is_loaded = True
            logger.info("Model loaded successfully!")
            
            # Log model summary
            logger.info(f"Model input shape: {self._model.input_shape}")
            logger.info(f"Model output shape: {self._model.output_shape}")
            
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, preprocessed_image: np.ndarray) -> Dict:
        """
        Run inference on a preprocessed image.
        
        Args:
            preprocessed_image: Numpy array of shape (1, 224, 224, 3) or (224, 224, 3)
                               with values normalized to [0, 1]
        
        Returns:
            dict: Prediction results containing:
                - class: Predicted class name (str)
                - class_id: Predicted class index (int)
                - confidence: Confidence percentage (float)
                - probabilities: Dict of all class probabilities
        
        Raises:
            ValueError: If input shape is incorrect
            Exception: If inference fails
        """
        try:
            # Ensure model is loaded
            if not self._is_loaded:
                self.load_model()
            
            # Validate input shape
            image = self._validate_input(preprocessed_image)
            
            # Run inference
            logger.debug("Running model inference...")
            predictions = self._model.predict(image, verbose=0)
            
            # Process predictions
            result = self._process_predictions(predictions[0])
            
            logger.info(f"Prediction: {result['class']} ({result['confidence']:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _validate_input(self, image: np.ndarray) -> np.ndarray:
        """
        Validate and reshape input image for model inference.
        
        Args:
            image: Input numpy array
            
        Returns:
            np.ndarray: Properly shaped array (1, 224, 224, 3)
            
        Raises:
            ValueError: If input shape is invalid
        """
        # Convert to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Handle different input shapes
        if len(image.shape) == 3:
            # Single image (H, W, C) - add batch dimension
            image = np.expand_dims(image, axis=0)
        elif len(image.shape) == 4:
            # Already has batch dimension (B, H, W, C)
            pass
        else:
            raise ValueError(f"Invalid input shape: {image.shape}. Expected (224, 224, 3) or (1, 224, 224, 3)")
        
        # Validate dimensions
        expected_shape = (1, self.image_size[0], self.image_size[1], 3)
        if image.shape != expected_shape:
            raise ValueError(f"Invalid input shape: {image.shape}. Expected {expected_shape}")
        
        # Validate value range (should be normalized to [0, 1])
        if image.max() > 1.0 or image.min() < 0.0:
            logger.warning("Input values outside [0, 1] range. Consider normalizing.")
        
        return image
    
    def _process_predictions(self, predictions: np.ndarray) -> Dict:
        """
        Process raw model predictions into a structured response.
        
        Args:
            predictions: Raw model output array of shape (num_classes,)
            
        Returns:
            dict: Structured prediction results
        """
        # Get predicted class (highest probability)
        predicted_class_id = int(np.argmax(predictions))
        predicted_class_name = self.class_names[predicted_class_id]
        confidence = float(predictions[predicted_class_id]) * 100
        
        # Build probabilities dictionary for all classes
        probabilities = {}
        for class_id, class_name in self.class_names.items():
            prob = float(predictions[class_id]) * 100
            probabilities[class_name] = round(prob, 2)
        
        return {
            "class": predicted_class_name,
            "class_id": predicted_class_id,
            "confidence": round(confidence, 2),
            "probabilities": probabilities
        }
    
    def get_model_for_gradcam(self):
        """
        Return the model instance for Grad-CAM computation.
        
        Grad-CAM requires direct access to the model to compute gradients
        and access intermediate layer outputs.
        
        Returns:
            tf.keras.Model: The loaded Keras model
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including architecture details
        """
        if not self._is_loaded:
            return {
                "loaded": False,
                "model_path": self.model_path,
                "error": "Model not loaded"
            }
        
        return {
            "loaded": True,
            "model_path": self.model_path,
            "input_shape": list(self._model.input_shape),
            "output_shape": list(self._model.output_shape),
            "num_classes": self.num_classes,
            "class_names": list(self.class_names.values()),
            "num_parameters": self._model.count_params()
        }
    
    def warmup(self) -> bool:
        """
        Warm up the model by running a dummy prediction.
        
        This helps reduce latency on the first real prediction by
        initializing TensorFlow graph execution.
        
        Returns:
            bool: True if warmup successful
        """
        try:
            logger.info("Warming up model with dummy prediction...")
            
            # Create a dummy image (zeros)
            dummy_image = np.zeros((1, self.image_size[0], self.image_size[1], 3), dtype=np.float32)
            
            # Run prediction (result doesn't matter)
            _ = self._model.predict(dummy_image, verbose=0)
            
            logger.info("Model warmup complete")
            return True
            
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
            return False


# Create a global model loader instance
model_loader = ModelLoader()


def get_model_loader() -> ModelLoader:
    """
    Get the global model loader instance.
    
    Returns:
        ModelLoader: The singleton model loader instance
    """
    return model_loader


# Module-level functions for convenience
def load_model() -> bool:
    """Load the model (convenience function)."""
    return model_loader.load_model()


def predict(preprocessed_image: np.ndarray) -> Dict:
    """Run prediction (convenience function)."""
    return model_loader.predict(preprocessed_image)


def get_model_info() -> Dict:
    """Get model info (convenience function)."""
    return model_loader.get_model_info()


# Test the model loader when run directly
if __name__ == "__main__":
    import sys
    
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("=" * 60)
    print("Testing Model Loader")
    print("=" * 60)
    
    try:
        # Load model
        print("\n1. Loading model...")
        loader = ModelLoader()
        success = loader.load_model()
        print(f"   Model loaded: {success}")
        
        # Get model info
        print("\n2. Model information:")
        info = loader.get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Warmup
        print("\n3. Running warmup...")
        loader.warmup()
        
        # Test prediction with dummy image
        print("\n4. Testing prediction with dummy image...")
        dummy_image = np.random.rand(224, 224, 3).astype(np.float32)
        result = loader.predict(dummy_image)
        print(f"   Predicted class: {result['class']}")
        print(f"   Confidence: {result['confidence']:.2f}%")
        print("   Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"     - {class_name}: {prob:.2f}%")
        
        print("\n" + "=" * 60)
        print("Model Loader Test: SUCCESS")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\n" + "=" * 60)
        print("Model Loader Test: FAILED")
        print("=" * 60)
        sys.exit(1)
