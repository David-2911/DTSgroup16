#!/usr/bin/env python3
"""
Test Script for Grad-CAM Functionality.

This script tests the complete Grad-CAM visualization pipeline:
1. Send image to API
2. Verify classification response
3. Decode and save visualization images
4. Validate heatmap quality

Usage:
    python test_gradcam.py [image_path]
    
If no image path provided, uses a sample from the training data.
"""

import os
import sys
import json
import base64
import argparse
from io import BytesIO
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_gradcam_api(image_path: str, save_outputs: bool = True):
    """
    Test Grad-CAM via the API endpoint.
    
    Args:
        image_path: Path to test MRI image
        save_outputs: Whether to save output images
    """
    import requests
    from PIL import Image
    
    print("=" * 60)
    print("Testing Grad-CAM via API")
    print("=" * 60)
    
    # Check if server is running
    try:
        health = requests.get('http://localhost:5000/api/health', timeout=5)
        if health.status_code != 200:
            print("Server not healthy. Start the server first:")
            print("   python app.py")
            return False
        print("Server is running")
    except requests.exceptions.ConnectionError:
        print("Cannot connect to server. Start it with:")
        print("   python app.py")
        return False
    
    # Check image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return False
    
    print(f"\nTesting with image: {image_path}")
    
    # Send classification request
    print("\nSending classification request...")
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(
            'http://localhost:5000/api/classify',
            files=files,
            timeout=60
        )
    
    # Parse response
    result = response.json()
    
    if not result.get('success'):
        print(f"Classification failed: {result.get('error', 'Unknown error')}")
        return False
    
    # Print prediction results
    prediction = result['prediction']
    print("\nClassification Results:")
    print(f"   Predicted Class: {prediction['class']}")
    print(f"   Confidence: {prediction['confidence']:.2f}%")
    print(f"   Class ID: {prediction['class_id']}")
    print("\n   Probabilities:")
    for class_name, prob in prediction['probabilities'].items():
        bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
        print(f"     {class_name:12}: {bar} {prob:.2f}%")
    
    # Check processing time
    if 'processing_time_ms' in result:
        print(f"\nProcessing Time: {result['processing_time_ms']:.2f}ms")
    
    # Check visualization data
    visualization = result.get('visualization', {})
    
    if not visualization:
        print("\nNo visualization data in response!")
        return False
    
    print("\nVisualization Data:")
    
    # Decode and save images
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    
    for key in ['heatmap_overlay', 'original_image', 'heatmap_only']:
        if key in visualization and visualization[key]:
            b64_data = visualization[key]
            print(f"   {key}: {len(b64_data)} characters (base64)")
            
            if save_outputs:
                try:
                    img_bytes = base64.b64decode(b64_data)
                    img = Image.open(BytesIO(img_bytes))
                    
                    output_path = output_dir / f"test_{key}.png"
                    img.save(output_path)
                    print(f"   Saved: {output_path}")
                except Exception as e:
                    print(f"   Failed to decode {key}: {e}")
        else:
            print(f"   Missing: {key}")
    
    print("\n" + "=" * 60)
    print("Grad-CAM API Test: SUCCESS")
    print("=" * 60)
    print(f"\nOutput images saved to: {output_dir}")
    
    return True


def test_gradcam_module():
    """
    Test Grad-CAM module directly (without API).
    """
    import numpy as np
    from PIL import Image
    
    print("=" * 60)
    print("Testing Grad-CAM Module Directly")
    print("=" * 60)
    
    try:
        # Import modules
        print("\n1. Importing modules...")
        from model_loader import get_model_loader
        from gradcam import GradCAM
        print("   Imports successful")
        
        # Load model
        print("\n2. Loading model...")
        loader = get_model_loader()
        loader.load_model()
        model = loader.get_model_for_gradcam()
        print(f"   Model loaded: {model.name}")
        
        # Initialize Grad-CAM
        print("\n3. Initializing Grad-CAM...")
        gradcam = GradCAM(model)
        layer_info = gradcam.get_layer_info()
        print(f"   Layer: {layer_info['layer_name']}")
        print(f"   Output shape: {layer_info.get('output_shape', 'N/A')}")
        
        # Create test image
        print("\n4. Creating test image...")
        test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
        print(f"   Test image shape: {test_image.shape}")
        
        # Generate heatmap for each class
        print("\n5. Testing heatmap generation for all classes...")
        from config import CLASS_NAMES
        
        for class_id, class_name in CLASS_NAMES.items():
            heatmap = gradcam.generate_heatmap(test_image, class_id)
            
            # Validate heatmap
            assert heatmap.shape == (224, 224), f"Wrong shape: {heatmap.shape}"
            assert heatmap.min() >= 0, f"Min < 0: {heatmap.min()}"
            assert heatmap.max() <= 1, f"Max > 1: {heatmap.max()}"
            
            print(f"   Class {class_id} ({class_name}): shape={heatmap.shape}, range=[{heatmap.min():.3f}, {heatmap.max():.3f}]")
        
        # Test overlay
        print("\n6. Testing heatmap overlay...")
        original_pil = Image.fromarray((test_image[0] * 255).astype(np.uint8))
        heatmap = gradcam.generate_heatmap(test_image, 0)
        overlayed = gradcam.overlay_heatmap(heatmap, original_pil, alpha=0.4)
        
        assert overlayed.size == (224, 224), f"Wrong size: {overlayed.size}"
        assert overlayed.mode == 'RGB', f"Wrong mode: {overlayed.mode}"
        print(f"   Overlay: size={overlayed.size}, mode={overlayed.mode}")
        
        # Test base64 conversion
        print("\n7. Testing base64 conversion...")
        b64_string = gradcam.heatmap_to_base64(overlayed)
        
        # Verify we can decode it back
        img_bytes = base64.b64decode(b64_string)
        decoded = Image.open(BytesIO(img_bytes))
        
        print(f"   Base64 length: {len(b64_string)} chars")
        print(f"   Decoded back: {decoded.size}")
        
        # Test full visualization
        print("\n8. Testing complete visualization pipeline...")
        viz_data = gradcam.generate_visualization(
            test_image,
            original_pil,
            class_index=0,
            alpha=0.4
        )
        
        assert 'heatmap_overlay' in viz_data, "Missing heatmap_overlay"
        assert 'original_image' in viz_data, "Missing original_image"
        assert 'heatmap_only' in viz_data, "Missing heatmap_only"
        
        print(f"   Keys present: {list(viz_data.keys())}")
        print(f"   Overlay: {len(viz_data['heatmap_overlay'])} chars")
        print(f"   Original: {len(viz_data['original_image'])} chars")
        print(f"   Heatmap only: {len(viz_data['heatmap_only'])} chars")
        
        print("\n" + "=" * 60)
        print("Grad-CAM Module Test: SUCCESS")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("Grad-CAM Module Test: FAILED")
        print("=" * 60)
        return False


def test_with_real_image(image_path: str):
    """
    Test Grad-CAM with a real MRI image (no API).
    
    Args:
        image_path: Path to real MRI image
    """
    import numpy as np
    from PIL import Image
    
    print("=" * 60)
    print("Testing Grad-CAM with Real Image")
    print("=" * 60)
    
    try:
        from model_loader import get_model_loader
        from gradcam import GradCAM
        from preprocessing import preprocess_image
        
        # Load and preprocess image
        print(f"\n1. Loading image: {image_path}")
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        print("\n2. Preprocessing image...")
        preprocessed = preprocess_image(image_bytes, os.path.basename(image_path))
        preprocessed_batch = np.expand_dims(preprocessed, axis=0)
        print(f"   Shape: {preprocessed_batch.shape}")
        
        # Load model and initialize Grad-CAM
        print("\n3. Loading model...")
        loader = get_model_loader()
        loader.load_model()
        model = loader.get_model_for_gradcam()
        
        print("\n4. Running inference...")
        prediction = loader.predict(preprocessed)
        print(f"   Predicted: {prediction['class']} ({prediction['confidence']:.2f}%)")
        
        # Generate visualization
        print("\n5. Generating Grad-CAM visualization...")
        gradcam = GradCAM(model)
        
        original_image = Image.open(image_path)
        viz_data = gradcam.generate_visualization(
            preprocessed_batch,
            original_image,
            class_index=prediction['class_id'],
            alpha=0.4
        )
        
        # Save outputs
        output_dir = Path(__file__).parent / "test_outputs"
        output_dir.mkdir(exist_ok=True)
        
        print("\n6. Saving outputs...")
        for key in ['heatmap_overlay', 'original_image', 'heatmap_only']:
            if key in viz_data:
                img_bytes = base64.b64decode(viz_data[key])
                img = Image.open(BytesIO(img_bytes))
                output_path = output_dir / f"real_{key}.png"
                img.save(output_path)
                print(f"   Saved: {output_path}")
        
        print("\n" + "=" * 60)
        print("Real Image Test: SUCCESS")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_sample_image():
    """Find a sample image from the training data."""
    sample_dirs = [
        Path(__file__).parent.parent.parent / "brain_Tumor_Types" / "glioma",
        Path(__file__).parent.parent.parent / "brain_Tumor_Types" / "meningioma",
        Path(__file__).parent.parent.parent / "brain_Tumor_Types" / "pituitary",
        Path(__file__).parent.parent.parent / "brain_Tumor_Types" / "notumor",
    ]
    
    for sample_dir in sample_dirs:
        if sample_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                images = list(sample_dir.glob(ext))
                if images:
                    return str(images[0])
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Test Grad-CAM functionality')
    parser.add_argument('image', nargs='?', help='Path to test image')
    parser.add_argument('--module', action='store_true', help='Test module directly (no API)')
    parser.add_argument('--api', action='store_true', help='Test via API')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    # Find sample image if not provided
    if not args.image:
        args.image = find_sample_image()
        if args.image:
            print(f"Using sample image: {args.image}")
        else:
            print("No sample image found. Provide image path or run module test only.")
    
    results = []
    
    # Run tests
    if args.all or args.module:
        results.append(("Module Test", test_gradcam_module()))
    
    if args.all or args.api:
        if args.image:
            results.append(("API Test", test_gradcam_api(args.image)))
        else:
            print("Skipping API test: no image provided")
    
    if not args.module and not args.api and not args.all:
        # Default: run module test and API test if image available
        results.append(("Module Test", test_gradcam_module()))
        if args.image:
            results.append(("API Test", test_gradcam_api(args.image)))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
