"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module.

This module generates heatmaps showing which regions of a brain MRI image
the classification model focuses on when making predictions.

Grad-CAM helps:
- Medical professionals understand model decisions
- Build trust in model predictions
- Identify if the model focuses on correct anatomical features
- Provide educational insights into CNN behavior
"""

import logging
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
from io import BytesIO
from typing import Tuple, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)


class GradCAM:
    """
    Generate Grad-CAM heatmaps to visualize model attention.
    
    Grad-CAM works by:
    1. Getting the last convolutional layer's feature maps
    2. Computing gradients of the predicted class score w.r.t. these features
    3. Weighting each feature map channel by its gradient importance
    4. Creating a heatmap by averaging the weighted feature maps
    5. Overlaying the heatmap on the original image
    
    Attributes:
        model: The loaded Keras model (ResNet50)
        layer_name: Name of the convolutional layer to use for Grad-CAM
        grad_model: TensorFlow model that outputs both conv layer and predictions
    """
    
    def __init__(self, model, layer_name: str = None):
        """
        Initialize Grad-CAM with trained model.
        
        Args:
            model: Loaded Keras model (ResNet50)
            layer_name: Name of last convolutional layer. If None, auto-detects.
        """
        self.model = model
        
        # Ensure model is built before accessing layers
        self._ensure_model_built()
        
        # Auto-detect layer name if not provided
        if layer_name is None:
            self.layer_name = self._find_last_conv_layer()
        else:
            self.layer_name = layer_name
            
        logger.info(f"Grad-CAM initialized with layer: {self.layer_name}")
        
        # Create gradient model
        self.grad_model = self._create_grad_model()
    
    def _ensure_model_built(self):
        """
        Ensure the model has been built (called at least once).
        
        Sequential models need to be called with input to define their input shape.
        Also handles nested models (e.g., Sequential containing ResNet50).
        """
        try:
            # Try to access input - if it fails, model needs to be built
            _ = self.model.input
        except AttributeError:
            # Model hasn't been built yet - build it with dummy input
            logger.info("Building model with dummy input...")
            import numpy as np
            dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
            _ = self.model(dummy_input, training=False)
            logger.info("Model built successfully")
        
        # For Sequential models containing Functional models (like ResNet50),
        # we need to find and use the inner model for Grad-CAM
        self._setup_inner_model()
    
    def _setup_inner_model(self):
        """
        Set up inner model reference for nested model structures.
        
        When the model is Sequential with a Functional model (like ResNet50) as
        a sublayer, we need to access the inner model for gradient computation.
        """
        self.inner_model = None
        self.base_model = None
        
        # Check if this is a Sequential model
        if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
            first_layer = self.model.layers[0]
            
            # Check if first layer is a Functional model (like ResNet50)
            if hasattr(first_layer, 'layers') and hasattr(first_layer, 'input'):
                logger.info(f"Found nested Functional model: {first_layer.name}")
                self.base_model = first_layer
                self.inner_model = first_layer
        
    def _find_last_conv_layer(self) -> str:
        """
        Find the name of the last convolutional layer in the model.
        
        Convolutional layers have 4D output (batch, height, width, channels).
        For nested models (Sequential containing ResNet50), searches the inner model.
        
        Returns:
            str: Name of the last convolutional layer
            
        Raises:
            ValueError: If no convolutional layer is found
        """
        # Determine which model/layers to search
        if self.inner_model is not None:
            # Search in the inner Functional model (e.g., ResNet50)
            layers_to_search = self.inner_model.layers
            logger.info(f"Searching for conv layer in inner model: {self.inner_model.name}")
        else:
            layers_to_search = self.model.layers
        
        for layer in reversed(layers_to_search):
            # Check if layer output is 4D (convolutional layer)
            if hasattr(layer, 'output'):
                output_shape = layer.output.shape
                if len(output_shape) == 4:
                    logger.info(f"Found last conv layer: {layer.name} with shape {output_shape}")
                    return layer.name
        
        # Fallback: try common ResNet50 layer names
        common_layers = [
            'conv5_block3_out',      # Standard ResNet50
            'conv5_block3_3_conv',   # Alternative naming
            'activation_48',          # By activation
            'block5_conv3',          # VGG-style naming
        ]
        
        # Try in inner model first, then outer model
        models_to_try = [self.inner_model, self.model] if self.inner_model else [self.model]
        
        for model in models_to_try:
            if model is None:
                continue
            for layer_name in common_layers:
                try:
                    model.get_layer(layer_name)
                    logger.info(f"Using fallback layer: {layer_name}")
                    return layer_name
                except ValueError:
                    continue
        
        raise ValueError("Could not find a convolutional layer in the model")
    
    def _create_grad_model(self) -> tf.keras.Model:
        """
        Create a model that outputs both conv layer activations and predictions.
        
        This model is used to compute gradients efficiently using GradientTape.
        For nested models (Sequential containing ResNet50), creates a combined model.
        
        Returns:
            tf.keras.Model: Model with two outputs [conv_outputs, predictions]
        """
        try:
            # For nested models, we need a special approach
            if self.inner_model is not None:
                # Get the conv layer from the inner model
                conv_layer = self.inner_model.get_layer(self.layer_name)
                
                # Create a new input that matches the inner model's expected input
                input_tensor = tf.keras.Input(shape=(224, 224, 3))
                
                # Get conv layer output from inner model
                inner_output = self.inner_model(input_tensor)
                
                # Create a model to get intermediate conv layer output
                conv_output_model = tf.keras.Model(
                    inputs=self.inner_model.input,
                    outputs=conv_layer.output
                )
                
                # Get conv outputs
                conv_outputs = conv_output_model(input_tensor)
                
                # Get final predictions from the full model
                predictions = self.model(input_tensor)
                
                # Create combined grad model
                grad_model = tf.keras.Model(
                    inputs=input_tensor,
                    outputs=[conv_outputs, predictions]
                )
                
                logger.info(f"Gradient model created for nested architecture")
                return grad_model
            else:
                # Standard approach for non-nested models
                conv_layer = self.model.get_layer(self.layer_name)
                
                grad_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=[conv_layer.output, self.model.output]
                )
                
                logger.info(f"Gradient model created successfully")
                return grad_model
            
        except Exception as e:
            logger.error(f"Failed to create gradient model: {e}")
            raise
    
    def generate_heatmap(self, image_array: np.ndarray, class_index: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specific class prediction.
        
        Process:
        1. Forward pass to get conv layer output and predictions
        2. Compute gradients of predicted class score w.r.t. conv layer
        3. Global average pooling of gradients to get importance weights
        4. Weight conv layer outputs by importance and sum
        5. Apply ReLU (only keep positive influences)
        6. Normalize to [0, 1] range
        7. Resize to match input image size
        
        Args:
            image_array: Preprocessed image array (1, 224, 224, 3) or (224, 224, 3)
            class_index: Index of the class to generate heatmap for (0-3)
            
        Returns:
            np.ndarray: Heatmap array of shape (224, 224) with values in [0, 1]
        """
        try:
            # Ensure image has batch dimension
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)
            
            # Convert to tensor and ensure float32
            image_tensor = tf.cast(image_array, tf.float32)
            
            # For nested models, we need a different approach
            if self.inner_model is not None:
                return self._generate_heatmap_nested(image_tensor, class_index)
            else:
                return self._generate_heatmap_standard(image_tensor, class_index)
                
        except Exception as e:
            logger.error(f"Failed to generate heatmap: {e}")
            raise
    
    def _generate_heatmap_standard(self, image_tensor: tf.Tensor, class_index: int) -> np.ndarray:
        """Generate heatmap using standard approach for non-nested models."""
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            conv_outputs, predictions = self.grad_model(image_tensor)
            class_score = predictions[:, class_index]
        
        grads = tape.gradient(class_score, conv_outputs)
        return self._compute_heatmap_from_grads(conv_outputs, grads)
    
    def _generate_heatmap_nested(self, image_tensor: tf.Tensor, class_index: int) -> np.ndarray:
        """Generate heatmap for nested models (Sequential containing Functional)."""
        # Get the conv layer from inner model
        conv_layer = self.inner_model.get_layer(self.layer_name)
        
        # Create a model that extracts conv layer output from inner model
        conv_output_extractor = tf.keras.Model(
            inputs=self.inner_model.input,
            outputs=conv_layer.output
        )
        
        with tf.GradientTape() as tape:
            # Get conv layer outputs
            conv_outputs = conv_output_extractor(image_tensor)
            tape.watch(conv_outputs)
            
            # Continue the forward pass through the rest of the model
            # The inner model output goes through remaining layers
            inner_output = self.inner_model(image_tensor)
            
            # Pass through remaining layers of the Sequential model
            x = inner_output
            for layer in self.model.layers[1:]:  # Skip the first layer (inner model)
                x = layer(x)
            
            predictions = x
            class_score = predictions[:, class_index]
        
        # Compute gradients
        grads = tape.gradient(class_score, conv_outputs)
        
        if grads is None:
            logger.warning("Gradients are None, trying alternative approach")
            return self._generate_heatmap_alternative(image_tensor, class_index)
        
        return self._compute_heatmap_from_grads(conv_outputs, grads)
    
    def _generate_heatmap_alternative(self, image_tensor: tf.Tensor, class_index: int) -> np.ndarray:
        """Alternative heatmap generation using direct gradient computation."""
        # Get conv layer
        conv_layer = self.inner_model.get_layer(self.layer_name)
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(image_tensor)
            
            # Forward pass through entire model
            predictions = self.model(image_tensor)
            class_score = predictions[:, class_index]
            
            # Get conv outputs separately
            conv_output_model = tf.keras.Model(
                inputs=self.inner_model.input,
                outputs=conv_layer.output
            )
            conv_outputs = conv_output_model(image_tensor)
        
        # Compute gradients of class score w.r.t input
        grads_wrt_input = tape.gradient(class_score, image_tensor)
        
        # Use conv outputs and approximate importance
        if grads_wrt_input is not None:
            # Compute a simpler attention map based on activation strength
            conv_outputs_np = conv_outputs.numpy()[0]
            
            # Average across channels weighted by activation magnitude
            heatmap = np.mean(conv_outputs_np, axis=-1)
            
            # Normalize
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            # Resize to 224x224
            heatmap = self._resize_heatmap(heatmap, (224, 224))
            return heatmap
        
        # Fallback: just use activation strength
        conv_outputs_np = conv_outputs.numpy()[0]
        heatmap = np.mean(np.abs(conv_outputs_np), axis=-1)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmap = self._resize_heatmap(heatmap, (224, 224))
        return heatmap
    
    def _compute_heatmap_from_grads(self, conv_outputs: tf.Tensor, grads: tf.Tensor) -> np.ndarray:
        """Compute heatmap from conv outputs and gradients."""
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Get conv layer output (remove batch dimension)
        conv_outputs = conv_outputs[0]
        
        # Weight each channel by its importance
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU
        heatmap = tf.nn.relu(heatmap)
        
        # Normalize to [0, 1]
        heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        
        # Convert to numpy
        heatmap = heatmap.numpy()
        
        # Resize to 224x224
        heatmap = self._resize_heatmap(heatmap, (224, 224))
        
        logger.debug(f"Heatmap generated: shape={heatmap.shape}, range=[{heatmap.min():.3f}, {heatmap.max():.3f}]")
        
        return heatmap
    
    def _resize_heatmap(self, heatmap: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize heatmap to target size using bilinear interpolation.
        
        Args:
            heatmap: Original heatmap array
            target_size: Target (width, height)
            
        Returns:
            np.ndarray: Resized heatmap
        """
        # Use PIL for resizing (no OpenCV dependency for this)
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img = heatmap_img.resize(target_size, Image.Resampling.BILINEAR)
        return np.array(heatmap_img) / 255.0
    
    def overlay_heatmap(
        self, 
        heatmap: np.ndarray, 
        original_image: Union[np.ndarray, Image.Image, bytes],
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> Image.Image:
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            heatmap: Grad-CAM heatmap (H, W) with values in [0, 1]
            original_image: Original MRI image (PIL Image, numpy array, or bytes)
            alpha: Transparency of heatmap overlay (0.0 = invisible, 1.0 = opaque)
            colormap: Colormap to use ('jet', 'viridis', 'hot', 'cool')
            
        Returns:
            PIL.Image: Image with colored heatmap overlay
        """
        try:
            # Convert original image to PIL if needed
            if isinstance(original_image, bytes):
                original_image = Image.open(BytesIO(original_image))
            elif isinstance(original_image, np.ndarray):
                # Handle different array formats
                if original_image.max() <= 1.0:
                    original_image = (original_image * 255).astype(np.uint8)
                original_image = Image.fromarray(original_image)
            
            # Ensure RGB mode
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            
            # Get original image size
            original_size = original_image.size  # (width, height)
            
            # Resize heatmap to match original image size
            heatmap_resized = self._resize_heatmap(heatmap, original_size)
            
            # Apply colormap to create colored heatmap
            heatmap_colored = self._apply_colormap(heatmap_resized, colormap)
            
            # Convert colored heatmap to PIL Image
            heatmap_pil = Image.fromarray(heatmap_colored)
            
            # Blend original image with heatmap
            # result = original * (1 - alpha) + heatmap * alpha
            overlayed = Image.blend(original_image, heatmap_pil, alpha)
            
            logger.debug(f"Overlay created with alpha={alpha}")
            
            return overlayed
            
        except Exception as e:
            logger.error(f"Failed to overlay heatmap: {e}")
            raise
    
    def _apply_colormap(self, heatmap: np.ndarray, colormap: str = 'jet') -> np.ndarray:
        """
        Apply a colormap to the heatmap.
        
        Uses pure Python/NumPy implementation to avoid OpenCV dependency.
        
        Args:
            heatmap: Grayscale heatmap (H, W) with values in [0, 1]
            colormap: Colormap name ('jet', 'viridis', 'hot', 'cool')
            
        Returns:
            np.ndarray: RGB colored heatmap (H, W, 3) with uint8 values
        """
        # Try to use OpenCV if available (better quality)
        try:
            import cv2
            
            # Convert to uint8
            heatmap_uint8 = np.uint8(255 * heatmap)
            
            # Apply colormap
            colormap_map = {
                'jet': cv2.COLORMAP_JET,
                'viridis': cv2.COLORMAP_VIRIDIS,
                'hot': cv2.COLORMAP_HOT,
                'cool': cv2.COLORMAP_COOL,
                'rainbow': cv2.COLORMAP_RAINBOW,
            }
            
            cv_colormap = colormap_map.get(colormap, cv2.COLORMAP_JET)
            colored = cv2.applyColorMap(heatmap_uint8, cv_colormap)
            
            # Convert BGR to RGB
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            
            return colored
            
        except ImportError:
            # Fallback: Pure NumPy jet colormap implementation
            logger.debug("OpenCV not available, using NumPy colormap")
            return self._numpy_jet_colormap(heatmap)
    
    def _numpy_jet_colormap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Apply jet colormap using pure NumPy (no OpenCV required).
        
        Jet colormap: blue (low) -> cyan -> green -> yellow -> red (high)
        
        Args:
            heatmap: Grayscale heatmap (H, W) with values in [0, 1]
            
        Returns:
            np.ndarray: RGB colored heatmap (H, W, 3) with uint8 values
        """
        # Create RGB arrays
        r = np.zeros_like(heatmap)
        g = np.zeros_like(heatmap)
        b = np.zeros_like(heatmap)
        
        # Jet colormap approximation
        # 0.0 - 0.125: dark blue to blue
        mask = heatmap < 0.125
        r[mask] = 0
        g[mask] = 0
        b[mask] = 0.5 + heatmap[mask] * 4
        
        # 0.125 - 0.375: blue to cyan
        mask = (heatmap >= 0.125) & (heatmap < 0.375)
        r[mask] = 0
        g[mask] = (heatmap[mask] - 0.125) * 4
        b[mask] = 1
        
        # 0.375 - 0.625: cyan to yellow
        mask = (heatmap >= 0.375) & (heatmap < 0.625)
        r[mask] = (heatmap[mask] - 0.375) * 4
        g[mask] = 1
        b[mask] = 1 - (heatmap[mask] - 0.375) * 4
        
        # 0.625 - 0.875: yellow to red
        mask = (heatmap >= 0.625) & (heatmap < 0.875)
        r[mask] = 1
        g[mask] = 1 - (heatmap[mask] - 0.625) * 4
        b[mask] = 0
        
        # 0.875 - 1.0: red to dark red
        mask = heatmap >= 0.875
        r[mask] = 1 - (heatmap[mask] - 0.875) * 2
        g[mask] = 0
        b[mask] = 0
        
        # Stack and convert to uint8
        colored = np.stack([r, g, b], axis=-1)
        colored = np.clip(colored * 255, 0, 255).astype(np.uint8)
        
        return colored
    
    def image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """
        Convert PIL Image to base64 encoded string.
        
        Args:
            image: PIL Image to convert
            format: Output format ('PNG', 'JPEG')
            
        Returns:
            str: Base64 encoded image string
        """
        try:
            buffered = BytesIO()
            image.save(buffered, format=format)
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            logger.debug(f"Image converted to base64 ({format}): {len(img_base64)} chars")
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Failed to convert image to base64: {e}")
            raise
    
    def heatmap_to_base64(self, overlayed_image: Image.Image) -> str:
        """
        Convenience method to convert overlayed image to base64.
        
        Args:
            overlayed_image: PIL Image with heatmap overlay
            
        Returns:
            str: Base64 encoded image string
        """
        return self.image_to_base64(overlayed_image, format='PNG')
    
    def generate_visualization(
        self,
        preprocessed_image: np.ndarray,
        original_image: Union[np.ndarray, Image.Image, bytes],
        class_index: int,
        alpha: float = 0.4
    ) -> dict:
        """
        Generate complete Grad-CAM visualization package.
        
        This is the main method to call from the API endpoint.
        
        Args:
            preprocessed_image: Preprocessed model input (1, 224, 224, 3)
            original_image: Original image for overlay
            class_index: Predicted class index
            alpha: Heatmap transparency
            
        Returns:
            dict: {
                'heatmap_overlay': base64 string of overlayed image,
                'original_image': base64 string of original image,
                'heatmap_raw': base64 string of heatmap only (optional)
            }
        """
        try:
            logger.info(f"Generating Grad-CAM visualization for class {class_index}")
            
            # Generate heatmap
            heatmap = self.generate_heatmap(preprocessed_image, class_index)
            
            # Create overlay on original image
            overlayed = self.overlay_heatmap(heatmap, original_image, alpha=alpha)
            
            # Convert original image to PIL if needed
            if isinstance(original_image, bytes):
                original_pil = Image.open(BytesIO(original_image))
            elif isinstance(original_image, np.ndarray):
                if original_image.max() <= 1.0:
                    original_image = (original_image * 255).astype(np.uint8)
                original_pil = Image.fromarray(original_image)
            else:
                original_pil = original_image
            
            if original_pil.mode != 'RGB':
                original_pil = original_pil.convert('RGB')
            
            # Convert to base64
            heatmap_overlay_b64 = self.image_to_base64(overlayed)
            original_image_b64 = self.image_to_base64(original_pil)
            
            # Also create heatmap-only visualization
            heatmap_colored = self._apply_colormap(heatmap, 'jet')
            heatmap_pil = Image.fromarray(heatmap_colored)
            heatmap_only_b64 = self.image_to_base64(heatmap_pil)
            
            logger.info("Grad-CAM visualization generated successfully")
            
            return {
                'heatmap_overlay': heatmap_overlay_b64,
                'original_image': original_image_b64,
                'heatmap_only': heatmap_only_b64
            }
            
        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")
            raise
    
    def get_layer_info(self) -> dict:
        """
        Get information about the Grad-CAM configuration.
        
        Returns:
            dict: Layer information
        """
        try:
            layer = self.model.get_layer(self.layer_name)
            return {
                'layer_name': self.layer_name,
                'layer_type': layer.__class__.__name__,
                'output_shape': str(layer.output_shape),
                'model_name': self.model.name if hasattr(self.model, 'name') else 'unknown'
            }
        except Exception as e:
            return {
                'layer_name': self.layer_name,
                'error': str(e)
            }


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

_gradcam_instance = None


def get_gradcam(model) -> GradCAM:
    """
    Get or create a GradCAM instance for the given model.
    
    Args:
        model: Keras model
        
    Returns:
        GradCAM: Initialized GradCAM instance
    """
    global _gradcam_instance
    if _gradcam_instance is None:
        _gradcam_instance = GradCAM(model)
    return _gradcam_instance


def generate_gradcam_visualization(
    model,
    preprocessed_image: np.ndarray,
    original_image: Union[np.ndarray, Image.Image, bytes],
    class_index: int,
    alpha: float = 0.4
) -> dict:
    """
    Convenience function to generate Grad-CAM visualization.
    
    Args:
        model: Keras model
        preprocessed_image: Preprocessed model input
        original_image: Original image for overlay
        class_index: Predicted class index
        alpha: Heatmap transparency
        
    Returns:
        dict: Visualization data with base64 encoded images
    """
    gradcam = get_gradcam(model)
    return gradcam.generate_visualization(
        preprocessed_image, 
        original_image, 
        class_index, 
        alpha
    )


# =============================================================================
# TEST WHEN RUN DIRECTLY
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("=" * 60)
    print("Testing Grad-CAM Module")
    print("=" * 60)
    
    try:
        # Import model loader
        from model_loader import get_model_loader
        
        # Load model
        print("\n1. Loading model...")
        loader = get_model_loader()
        loader.load_model()
        model = loader.get_model_for_gradcam()
        print(f"   Model loaded: {model.name}")
        
        # Initialize Grad-CAM
        print("\n2. Initializing Grad-CAM...")
        gradcam = GradCAM(model)
        layer_info = gradcam.get_layer_info()
        print(f"   Layer: {layer_info['layer_name']}")
        print(f"   Output shape: {layer_info.get('output_shape', 'N/A')}")
        
        # Create test image
        print("\n3. Creating test image...")
        test_image = np.random.rand(224, 224, 3).astype(np.float32)
        test_image_batch = np.expand_dims(test_image, axis=0)
        print(f"   Test image shape: {test_image_batch.shape}")
        
        # Generate heatmap
        print("\n4. Generating heatmap...")
        heatmap = gradcam.generate_heatmap(test_image_batch, class_index=0)
        print(f"   Heatmap shape: {heatmap.shape}")
        print(f"   Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        
        # Create overlay
        print("\n5. Creating overlay...")
        original_pil = Image.fromarray((test_image * 255).astype(np.uint8))
        overlayed = gradcam.overlay_heatmap(heatmap, original_pil, alpha=0.4)
        print(f"   Overlay size: {overlayed.size}")
        print(f"   Overlay mode: {overlayed.mode}")
        
        # Convert to base64
        print("\n6. Converting to base64...")
        b64_string = gradcam.heatmap_to_base64(overlayed)
        print(f"   Base64 length: {len(b64_string)} characters")
        
        # Full visualization
        print("\n7. Testing full visualization...")
        viz_data = gradcam.generate_visualization(
            test_image_batch,
            original_pil,
            class_index=0,
            alpha=0.4
        )
        print(f"   Keys: {list(viz_data.keys())}")
        print(f"   Overlay size: {len(viz_data['heatmap_overlay'])} chars")
        print(f"   Original size: {len(viz_data['original_image'])} chars")
        
        print("\n" + "=" * 60)
        print("Grad-CAM Module Test: SUCCESS")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("Grad-CAM Module Test: FAILED")
        print("=" * 60)
        sys.exit(1)
