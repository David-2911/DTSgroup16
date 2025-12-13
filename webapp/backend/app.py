"""
Flask Application for Brain Tumor Classification.

This is the main entry point for the backend API. It provides endpoints for:
- /api/classify: Upload and classify brain MRI images
- /api/health: Health check endpoint
- /api/classes: Get available classification classes

The application uses:
- TensorFlow/Keras for model inference
- Apache Spark for image preprocessing (consistency with training)
- Flask-CORS for cross-origin resource sharing
"""

import os
import sys
import time
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG, CORS_ORIGINS,
    MAX_CONTENT_LENGTH, LOG_LEVEL, LOG_FORMAT
)
from model_loader import ModelLoader, get_model_loader
from preprocessing import ImagePreprocessor, get_preprocessor, SparkSessionManager
from gradcam import GradCAM, get_gradcam
from analysis_generator import AnalysisGenerator
from utils import (
    validate_image, allowed_file,
    format_success_response, format_error_response,
    format_health_response, format_classes_response,
    format_visualization_response, load_original_image,
    setup_logging, log_request_info, Timer
)

# =============================================================================
# APPLICATION SETUP
# =============================================================================

# Configure logging
setup_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)

# Create Flask application
app = Flask(__name__)

# Configure Flask
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Enable CORS for frontend access
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

# Global instances (initialized at startup)
model_loader = None
preprocessor = None
spark_manager = None
gradcam = None
analysis_generator = None


def initialize_services():
    """
    Initialize model, preprocessor, Spark session, and Grad-CAM at application startup.
    
    This ensures all services are ready before handling requests.
    """
    global model_loader, preprocessor, spark_manager, gradcam, analysis_generator
    
    logger.info("=" * 60)
    logger.info("Initializing Brain Tumor Classification Backend")
    logger.info("=" * 60)
    
    try:
        # Initialize Spark session manager
        logger.info("Initializing Spark session...")
        spark_manager = SparkSessionManager()
        spark = spark_manager.get_or_create_session()
        logger.info(f"Spark session ready: {spark.sparkContext.applicationId}")
        
        # Initialize image preprocessor
        logger.info("Initializing image preprocessor...")
        preprocessor = get_preprocessor(use_hdfs=False)
        logger.info("Image preprocessor ready")
        
        # Initialize and load model
        logger.info("Loading classification model...")
        model_loader = get_model_loader()
        model_loader.load_model()
        logger.info("Model loaded successfully")
        
        # Warmup model for faster first prediction
        logger.info("Warming up model...")
        model_loader.warmup()
        
        # Initialize Grad-CAM for visualization
        logger.info("Initializing Grad-CAM visualization...")
        gradcam = GradCAM(model_loader.get_model_for_gradcam())
        logger.info(f"Grad-CAM ready with layer: {gradcam.layer_name}")
        
        # Initialize analysis generator
        logger.info("Initializing medical analysis generator...")
        analysis_generator = AnalysisGenerator()
        logger.info("Analysis generator ready")
        
        logger.info("=" * 60)
        logger.info("Backend initialization complete!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        logger.error("The server will start but may not function properly.")
        return False


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/classify', methods=['POST'])
def classify_image():
    """
    Classify a brain MRI image with Grad-CAM visualization.
    
    Accepts a multipart/form-data POST request with an image file.
    Returns classification results along with Grad-CAM heatmap visualization
    showing which regions of the MRI the model focused on.
    
    Request:
        - file: Image file (PNG, JPG, JPEG, etc.)
        
    Response:
        {
            "success": true,
            "prediction": {
                "class": "Glioma",
                "class_id": 0,
                "confidence": 94.5,
                "probabilities": {
                    "Glioma": 94.5,
                    "Meningioma": 3.2,
                    "No Tumor": 1.8,
                    "Pituitary": 0.5
                }
            },
            "visualization": {
                "heatmap_overlay": "base64_encoded_image...",
                "original_image": "base64_encoded_image...",
                "heatmap_only": "base64_encoded_image..."
            },
            "processing_time_ms": 1523.45,
            "timestamp": "2024-01-15T10:30:00.000Z"
        }
        
    Errors:
        - 400: No file provided or invalid file
        - 415: Unsupported file type
        - 500: Internal server error
    """
    start_time = time.time()
    log_request_info(request)
    
    try:
        # Check if file is in request
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify(*format_error_response(
                "No file provided. Use 'file' field in multipart/form-data.",
                "NO_FILE",
                400
            ))
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            logger.warning("Empty filename in request")
            return jsonify(*format_error_response(
                "No file selected.",
                "EMPTY_FILENAME",
                400
            ))
        
        # Check file extension
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify(*format_error_response(
                f"File type not allowed. Allowed: png, jpg, jpeg, gif, bmp, webp",
                "INVALID_FILE_TYPE",
                415
            ))
        
        # Read file data
        file_data = file.read()
        
        # Validate image
        is_valid, validation_message = validate_image(
            file_data,
            filename=file.filename,
            mime_type=file.content_type
        )
        
        if not is_valid:
            logger.warning(f"Image validation failed: {validation_message}")
            return jsonify(*format_error_response(
                validation_message,
                "INVALID_IMAGE",
                400
            ))
        
        logger.info(f"Processing image: {file.filename}")
        
        # Preprocess image using Spark-based preprocessing
        with Timer("Image preprocessing"):
            preprocessed_image = preprocessor.preprocess_image(file_data, file.filename)
        
        # Run model inference
        with Timer("Model inference"):
            prediction = model_loader.predict(preprocessed_image)
        
        # Generate Grad-CAM visualization
        with Timer("Grad-CAM visualization"):
            # Load original image for overlay
            original_image = load_original_image(file_data)
            
            # Generate heatmap and overlay
            visualization = gradcam.generate_visualization(
                preprocessed_image,
                original_image,
                class_index=prediction['class_id'],
                alpha=0.4
            )
        
        # Generate medical analysis
        with Timer("Medical analysis"):
            analysis = analysis_generator.generate_analysis(
                predicted_class=prediction['class'],
                confidence=prediction['confidence'],
                all_probabilities=prediction['probabilities']
            )
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        # Format response with visualization and analysis
        response = format_visualization_response(
            prediction, visualization, processing_time, analysis
        )
        
        logger.info(f"Classification complete: {prediction['class']} ({prediction['confidence']:.1f}%)")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        return jsonify(*format_error_response(
            f"Classification failed: {str(e)}",
            "CLASSIFICATION_ERROR",
            500
        ))


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns the status of the backend services.
    
    Response:
        {
            "status": "healthy",
            "timestamp": "2024-01-15T10:30:00.000Z",
            "services": {
                "model": {
                    "status": "up",
                    "loaded": true
                },
                "spark": {
                    "status": "up",
                    "active": true
                }
            },
            "details": {
                "model_path": "/path/to/model.keras",
                "num_classes": 4,
                "image_size": [224, 224]
            }
        }
    """
    try:
        # Check model status
        model_loaded = model_loader is not None and model_loader.is_loaded
        
        # Check Spark status
        spark_active = spark_manager is not None and spark_manager.is_active
        
        # Get additional details
        details = {}
        
        if model_loaded:
            model_info = model_loader.get_model_info()
            details["model_path"] = model_info.get("model_path")
            details["num_classes"] = model_info.get("num_classes")
            details["image_size"] = [224, 224]
        
        if spark_active:
            spark = spark_manager.get_or_create_session()
            details["spark_app_id"] = spark.sparkContext.applicationId
            details["spark_version"] = spark.version
        
        response = format_health_response(model_loaded, spark_active, details)
        
        # Return appropriate status code
        status_code = 200 if response["status"] == "healthy" else 503
        
        return jsonify(response), status_code
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """
    Get available classification classes.
    
    Returns information about all tumor classes the model can predict.
    
    Response:
        {
            "success": true,
            "classes": [
                {
                    "id": 0,
                    "name": "Glioma",
                    "description": "A type of tumor that occurs in the brain..."
                },
                ...
            ],
            "total_classes": 4
        }
    """
    try:
        response = format_classes_response()
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Get classes error: {e}")
        return jsonify(*format_error_response(
            f"Failed to get classes: {str(e)}",
            "CLASSES_ERROR",
            500
        ))


@app.route('/', methods=['GET'])
def root():
    """
    Root endpoint - API information.
    """
    return jsonify({
        "name": "Brain Tumor Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/api/classify": {
                "method": "POST",
                "description": "Upload and classify a brain MRI image"
            },
            "/api/health": {
                "method": "GET",
                "description": "Health check endpoint"
            },
            "/api/classes": {
                "method": "GET",
                "description": "Get available classification classes"
            }
        },
        "documentation": "See README.md for detailed API documentation"
    })


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file too large error."""
    max_mb = MAX_CONTENT_LENGTH / (1024 * 1024)
    return jsonify(*format_error_response(
        f"File too large. Maximum size: {max_mb:.0f}MB",
        "FILE_TOO_LARGE",
        413
    ))


@app.errorhandler(404)
def handle_not_found(e):
    """Handle 404 errors."""
    return jsonify(*format_error_response(
        "Endpoint not found",
        "NOT_FOUND",
        404
    ))


@app.errorhandler(405)
def handle_method_not_allowed(e):
    """Handle 405 errors."""
    return jsonify(*format_error_response(
        f"Method not allowed. Allowed methods for this endpoint.",
        "METHOD_NOT_ALLOWED",
        405
    ))


@app.errorhandler(500)
def handle_internal_error(e):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify(*format_error_response(
        "Internal server error",
        "INTERNAL_ERROR",
        500
    ))


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================

@app.before_request
def before_request():
    """Called before each request."""
    pass  # Add any pre-request logic here


@app.after_request
def after_request(response):
    """Called after each request."""
    # Add security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


def shutdown_services():
    """Clean up resources on shutdown."""
    logger.info("Shutting down services...")
    
    try:
        if spark_manager is not None:
            spark_manager.stop_session()
            logger.info("Spark session stopped")
    except Exception as e:
        logger.error(f"Error stopping Spark: {e}")
    
    logger.info("Shutdown complete")


import atexit
atexit.register(shutdown_services)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Initialize services before starting the server
    init_success = initialize_services()
    
    if not init_success:
        logger.warning("Some services failed to initialize. Server starting anyway...")
    
    # Start Flask server
    logger.info(f"Starting server on {FLASK_HOST}:{FLASK_PORT}")
    logger.info(f"Debug mode: {FLASK_DEBUG}")
    
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        use_reloader=False  # Disable reloader to prevent double initialization
    )
