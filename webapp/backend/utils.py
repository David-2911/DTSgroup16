"""
Utility functions for Brain Tumor Classification Backend.

This module contains helper functions for file validation, response formatting,
error handling, and other common operations used throughout the backend.
"""

import logging
import os
import time
import functools
from typing import Dict, Any, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime

from config import ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, MAX_CONTENT_LENGTH, CLASS_NAMES

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# FILE VALIDATION FUNCTIONS
# =============================================================================

def allowed_file(filename: str) -> bool:
    """
    Check if a filename has an allowed extension.
    
    Args:
        filename: The filename to check
        
    Returns:
        bool: True if the file extension is allowed, False otherwise
    """
    if not filename:
        return False
    
    # Check if filename has an extension
    if '.' not in filename:
        return False
    
    # Get extension (lowercase) and check against allowed list
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def validate_file_extension(filename: str) -> Tuple[bool, str]:
    """
    Validate file extension with detailed error message.
    
    Args:
        filename: The filename to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    if not filename:
        return False, "No filename provided"
    
    if '.' not in filename:
        return False, "Filename must have an extension"
    
    extension = filename.rsplit('.', 1)[1].lower()
    
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ', '.join(sorted(ALLOWED_EXTENSIONS))
        return False, f"File extension '.{extension}' not allowed. Allowed: {allowed}"
    
    return True, f"Valid file extension: .{extension}"


def validate_mime_type(mime_type: str) -> Tuple[bool, str]:
    """
    Validate MIME type of uploaded file.
    
    Args:
        mime_type: The MIME type to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    if not mime_type:
        return False, "No MIME type provided"
    
    if mime_type not in ALLOWED_MIME_TYPES:
        allowed = ', '.join(sorted(ALLOWED_MIME_TYPES))
        return False, f"MIME type '{mime_type}' not allowed. Allowed: {allowed}"
    
    return True, f"Valid MIME type: {mime_type}"


def validate_file_size(file_size: int) -> Tuple[bool, str]:
    """
    Validate file size against maximum limit.
    
    Args:
        file_size: Size of the file in bytes
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    if file_size <= 0:
        return False, "File is empty"
    
    if file_size > MAX_CONTENT_LENGTH:
        max_mb = MAX_CONTENT_LENGTH / (1024 * 1024)
        file_mb = file_size / (1024 * 1024)
        return False, f"File too large: {file_mb:.2f}MB. Maximum: {max_mb:.0f}MB"
    
    return True, f"Valid file size: {file_size} bytes"


def validate_image(file_data: bytes, filename: str = None, mime_type: str = None) -> Tuple[bool, str]:
    """
    Comprehensive image validation.
    
    Validates:
    - File extension (if filename provided)
    - MIME type (if provided)
    - File size
    - Image format and integrity
    
    Args:
        file_data: Raw file bytes
        filename: Original filename (optional)
        mime_type: MIME type from upload (optional)
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message or success_message)
    """
    try:
        # Validate file size
        is_valid, message = validate_file_size(len(file_data))
        if not is_valid:
            return False, message
        
        # Validate extension if filename provided
        if filename:
            is_valid, message = validate_file_extension(filename)
            if not is_valid:
                return False, message
        
        # Validate MIME type if provided
        if mime_type:
            is_valid, message = validate_mime_type(mime_type)
            if not is_valid:
                return False, message
        
        # Validate actual image content
        from PIL import Image
        from io import BytesIO
        
        try:
            image = Image.open(BytesIO(file_data))
            image.verify()  # Verify image integrity
            
            # Reopen to get format info (verify() makes image unusable)
            image = Image.open(BytesIO(file_data))
            
            logger.info(f"Image validation passed: {image.format} {image.size[0]}x{image.size[1]}")
            return True, f"Valid {image.format} image ({image.size[0]}x{image.size[1]})"
            
        except Exception as e:
            return False, f"Invalid or corrupted image file: {e}"
            
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        return False, f"Validation error: {e}"


# =============================================================================
# IMAGE HANDLING FUNCTIONS (for Grad-CAM visualization)
# =============================================================================

def load_original_image(image_source):
    """
    Load original image for Grad-CAM overlay.
    
    Args:
        image_source: Can be:
            - str: File path to image
            - bytes: Raw image bytes
            - PIL.Image: Already loaded image
            
    Returns:
        PIL.Image: Loaded image in RGB mode
    """
    from PIL import Image
    from io import BytesIO
    
    if isinstance(image_source, str):
        # Load from file path
        image = Image.open(image_source)
    elif isinstance(image_source, bytes):
        # Load from bytes
        image = Image.open(BytesIO(image_source))
    elif hasattr(image_source, 'mode'):
        # Already a PIL Image
        image = image_source
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def resize_for_display(image, max_width: int = 800, max_height: int = 600):
    """
    Resize image for frontend display while maintaining aspect ratio.
    
    Args:
        image: PIL Image to resize
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels
        
    Returns:
        PIL.Image: Resized image (or original if already smaller)
    """
    from PIL import Image as PILImage
    
    # Create a copy to avoid modifying original
    resized = image.copy()
    resized.thumbnail((max_width, max_height), PILImage.Resampling.LANCZOS)
    
    return resized


def image_to_base64(image, format: str = 'PNG') -> str:
    """
    Convert PIL Image to base64 encoded string.
    
    Args:
        image: PIL Image to convert
        format: Output format ('PNG', 'JPEG')
        
    Returns:
        str: Base64 encoded image string
    """
    import base64
    from io import BytesIO
    
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    
    return base64.b64encode(img_bytes).decode('utf-8')


def base64_to_image(base64_string: str):
    """
    Convert base64 string back to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL.Image: Decoded image
    """
    import base64
    from io import BytesIO
    from PIL import Image
    
    img_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_bytes))


def format_visualization_response(prediction: Dict, visualization: Dict, processing_time: float = None, analysis: Dict = None) -> Dict:
    """
    Format a classification response with Grad-CAM visualization and medical analysis.
    
    Args:
        prediction: Prediction results from model
        visualization: Visualization data with base64 images
        processing_time: Time taken for processing (optional)
        analysis: Medical analysis data (optional)
        
    Returns:
        dict: Formatted API response with visualization and analysis
    """
    response = {
        "success": True,
        "prediction": {
            "class": prediction["class"],
            "class_id": prediction["class_id"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"]
        },
        "visualization": {
            "heatmap_overlay": visualization.get("heatmap_overlay"),
            "original_image": visualization.get("original_image"),
            "heatmap_only": visualization.get("heatmap_only")
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    if processing_time is not None:
        response["processing_time_ms"] = round(processing_time * 1000, 2)
    
    if analysis is not None:
        response["analysis"] = analysis
    
    return response


# =============================================================================
# RESPONSE FORMATTING FUNCTIONS
# =============================================================================

def format_success_response(prediction: Dict, processing_time: float = None) -> Dict:
    """
    Format a successful classification response.
    
    Args:
        prediction: Prediction results from model
        processing_time: Time taken for processing (optional)
        
    Returns:
        dict: Formatted API response
    """
    response = {
        "success": True,
        "prediction": {
            "class": prediction["class"],
            "class_id": prediction["class_id"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"]
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    if processing_time is not None:
        response["processing_time_ms"] = round(processing_time * 1000, 2)
    
    return response


def format_error_response(error_message: str, error_code: str = None, status_code: int = 400) -> Tuple[Dict, int]:
    """
    Format an error response.
    
    Args:
        error_message: Human-readable error message
        error_code: Machine-readable error code (optional)
        status_code: HTTP status code
        
    Returns:
        Tuple[dict, int]: (response_dict, status_code)
    """
    response = {
        "success": False,
        "error": {
            "message": error_message,
            "code": error_code or get_error_code(status_code),
            "status": status_code
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    return response, status_code


def get_error_code(status_code: int) -> str:
    """
    Get a standard error code for an HTTP status code.
    
    Args:
        status_code: HTTP status code
        
    Returns:
        str: Error code string
    """
    error_codes = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        413: "PAYLOAD_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA_TYPE",
        422: "UNPROCESSABLE_ENTITY",
        429: "TOO_MANY_REQUESTS",
        500: "INTERNAL_ERROR",
        503: "SERVICE_UNAVAILABLE"
    }
    
    return error_codes.get(status_code, "UNKNOWN_ERROR")


def format_health_response(model_loaded: bool, spark_active: bool, details: Dict = None) -> Dict:
    """
    Format a health check response.
    
    Args:
        model_loaded: Whether the model is loaded
        spark_active: Whether Spark session is active
        details: Additional health details
        
    Returns:
        dict: Health check response
    """
    status = "healthy" if (model_loaded and spark_active) else "unhealthy"
    
    response = {
        "status": status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "services": {
            "model": {
                "status": "up" if model_loaded else "down",
                "loaded": model_loaded
            },
            "spark": {
                "status": "up" if spark_active else "down",
                "active": spark_active
            }
        }
    }
    
    if details:
        response["details"] = details
    
    return response


def format_classes_response() -> Dict:
    """
    Format response for available classification classes.
    
    Returns:
        dict: Classes information response
    """
    return {
        "success": True,
        "classes": [
            {
                "id": class_id,
                "name": class_name,
                "description": get_class_description(class_name)
            }
            for class_id, class_name in CLASS_NAMES.items()
        ],
        "total_classes": len(CLASS_NAMES)
    }


def get_class_description(class_name: str) -> str:
    """
    Get a description for a tumor class.
    
    Args:
        class_name: Name of the tumor class
        
    Returns:
        str: Description of the class
    """
    descriptions = {
        "Glioma": "A type of tumor that occurs in the brain and spinal cord, arising from glial cells.",
        "Meningioma": "A tumor that arises from the meninges, the membranes surrounding the brain and spinal cord.",
        "No Tumor": "No tumor detected in the MRI scan.",
        "Pituitary": "A tumor that forms in the pituitary gland, located at the base of the brain."
    }
    
    return descriptions.get(class_name, "Brain tumor classification category.")


# =============================================================================
# TIMING AND PERFORMANCE UTILITIES
# =============================================================================

def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.
    
    Args:
        func: Function to wrap
        
    Returns:
        Callable: Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed_time:.3f}s")
        return result
    return wrapper


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        logger.debug(f"{self.name} completed in {self.elapsed:.3f}s")


# =============================================================================
# FILE HANDLING UTILITIES
# =============================================================================

def secure_filename(filename: str) -> str:
    """
    Sanitize a filename for safe storage.
    
    Removes path separators and potentially dangerous characters.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove path separators
    filename = os.path.basename(filename)
    
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    
    # Keep only alphanumeric, dots, underscores, and hyphens
    safe_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-')
    filename = ''.join(c for c in filename if c in safe_chars)
    
    # Ensure filename is not empty
    if not filename:
        filename = 'unnamed'
    
    return filename


def generate_unique_filename(original_filename: str) -> str:
    """
    Generate a unique filename based on the original.
    
    Args:
        original_filename: Original filename
        
    Returns:
        str: Unique filename with timestamp
    """
    import uuid
    
    # Get extension
    if '.' in original_filename:
        name, ext = original_filename.rsplit('.', 1)
        ext = '.' + ext.lower()
    else:
        name = original_filename
        ext = ''
    
    # Generate unique ID
    unique_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    return f"{secure_filename(name)}_{timestamp}_{unique_id}{ext}"


def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """
    Remove files older than specified age from a directory.
    
    Args:
        directory: Path to directory to clean
        max_age_hours: Maximum age in hours before deletion
    """
    if not os.path.exists(directory):
        return
    
    max_age_seconds = max_age_hours * 3600
    current_time = time.time()
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            
            if file_age > max_age_seconds:
                try:
                    os.remove(filepath)
                    logger.info(f"Removed old file: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to remove {filename}: {e}")


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_logging(log_level: str = "INFO", log_format: str = None):
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
    """
    from config import LOG_LEVEL, LOG_FORMAT
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = log_format or LOG_FORMAT
    
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Reduce verbosity of some libraries
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def log_request_info(request, logger_instance=None):
    """
    Log information about an incoming request.
    
    Args:
        request: Flask request object
        logger_instance: Logger to use (optional)
    """
    log = logger_instance or logger
    
    log.info(f"Request: {request.method} {request.path}")
    log.debug(f"  Remote: {request.remote_addr}")
    log.debug(f"  Content-Type: {request.content_type}")
    log.debug(f"  Content-Length: {request.content_length}")


# =============================================================================
# TEST WHEN RUN DIRECTLY
# =============================================================================

if __name__ == "__main__":
    # Configure logging
    setup_logging("DEBUG")
    
    print("=" * 60)
    print("Testing Utility Functions")
    print("=" * 60)
    
    # Test file validation
    print("\n1. Testing file validation...")
    print(f"   allowed_file('image.jpg'): {allowed_file('image.jpg')}")
    print(f"   allowed_file('image.txt'): {allowed_file('image.txt')}")
    print(f"   allowed_file('noext'): {allowed_file('noext')}")
    
    # Test response formatting
    print("\n2. Testing response formatting...")
    
    mock_prediction = {
        "class": "Glioma",
        "class_id": 0,
        "confidence": 94.5,
        "probabilities": {
            "Glioma": 94.5,
            "Meningioma": 3.2,
            "No Tumor": 1.8,
            "Pituitary": 0.5
        }
    }
    
    success_response = format_success_response(mock_prediction, 0.523)
    print(f"   Success response: {success_response}")
    
    error_response, status = format_error_response("File too large", status_code=413)
    print(f"   Error response: {error_response}")
    
    # Test health response
    print("\n3. Testing health response...")
    health = format_health_response(True, True)
    print(f"   Health: {health}")
    
    # Test classes response
    print("\n4. Testing classes response...")
    classes = format_classes_response()
    print(f"   Classes: {classes}")
    
    # Test filename utilities
    print("\n5. Testing filename utilities...")
    print(f"   secure_filename('../../../etc/passwd'): {secure_filename('../../../etc/passwd')}")
    print(f"   secure_filename('my image.jpg'): {secure_filename('my image.jpg')}")
    print(f"   generate_unique_filename('brain_scan.jpg'): {generate_unique_filename('brain_scan.jpg')}")
    
    # Test timer
    print("\n6. Testing timer...")
    with Timer("Sleep test") as t:
        time.sleep(0.1)
    print(f"   Elapsed: {t.elapsed:.3f}s")
    
    print("\n" + "=" * 60)
    print("Utility Functions Test: SUCCESS")
    print("=" * 60)
