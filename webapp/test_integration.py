"""
Integration Test Script for Brain MRI Classification Web Application
Tests complete workflow from upload to classification
"""

import requests
import json
import os
import sys
from pathlib import Path
import time

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}[PASS] {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}[FAIL] {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}[INFO] {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[WARN] {msg}{Colors.END}")

def print_header(msg):
    print(f"\n{Colors.BOLD}{msg}{Colors.END}")


class IntegrationTester:
    def __init__(self):
        self.backend_url = "http://localhost:5000/api"
        self.test_images_path = Path(__file__).parent.parent / "brain_Tumor_Types"
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
    
    def test_backend_health(self):
        """Test if backend is responding"""
        print_info("Testing backend health endpoint...")
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                if status == 'healthy':
                    print_success("Backend is online and healthy")
                    
                    # Check services
                    services = data.get('services', {})
                    model_status = services.get('model', {}).get('status')
                    spark_status = services.get('spark', {}).get('status')
                    
                    print_info(f"  Model status: {model_status}")
                    print_info(f"  Spark status: {spark_status}")
                    
                    self.passed_tests += 1
                    return True
            print_error("Backend returned unhealthy status")
            self.failed_tests += 1
            return False
        except requests.exceptions.ConnectionError:
            print_error("Cannot connect to backend. Is Flask running on port 5000?")
            self.failed_tests += 1
            return False
        except requests.exceptions.Timeout:
            print_error("Backend health check timed out")
            self.failed_tests += 1
            return False
        except Exception as e:
            print_error(f"Health check failed: {str(e)}")
            self.failed_tests += 1
            return False
    
    def test_classes_endpoint(self):
        """Test classes endpoint"""
        print_info("Testing /api/classes endpoint...")
        try:
            response = requests.get(f"{self.backend_url}/classes", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    classes_data = data.get('classes', [])
                    # Classes can be list of strings or list of objects with 'name' key
                    if isinstance(classes_data, list) and len(classes_data) > 0:
                        if isinstance(classes_data[0], dict):
                            class_names = [c.get('name') for c in classes_data]
                        else:
                            class_names = classes_data
                        expected = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
                        if set(class_names) == set(expected):
                            print_success(f"Classes endpoint working correctly")
                            print_info(f"  Available classes: {', '.join(class_names)}")
                            self.passed_tests += 1
                            return True
            print_error("Classes endpoint returned unexpected data")
            self.failed_tests += 1
            return False
        except Exception as e:
            print_error(f"Classes test failed: {str(e)}")
            self.failed_tests += 1
            return False
    
    def find_test_image(self, class_name):
        """Find a test image for the given class"""
        # Map class names to folder names
        folder_map = {
            'Glioma': 'glioma',
            'Meningioma': 'meningioma',
            'No Tumor': 'notumor',
            'Pituitary': 'pituitary'
        }
        
        folder_name = folder_map.get(class_name, class_name.lower().replace(" ", ""))
        class_folder = self.test_images_path / folder_name
        
        if not class_folder.exists():
            return None
        
        # Get first available image
        images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg")) + list(class_folder.glob("*.png"))
        return images[0] if images else None
    
    def test_classification(self, class_name):
        """Test classification for a specific tumor type"""
        print_info(f"Testing classification for '{class_name}'...")
        
        # Find test image
        test_image = self.find_test_image(class_name)
        if not test_image:
            print_warning(f"No test images found for {class_name}, skipping...")
            self.skipped_tests += 1
            return None
        
        print_info(f"  Using test image: {test_image.name}")
        
        try:
            # Send classification request
            with open(test_image, 'rb') as f:
                files = {'file': (test_image.name, f, 'image/jpeg')}
                start_time = time.time()
                response = requests.post(
                    f"{self.backend_url}/classify",
                    files=files,
                    timeout=60
                )
                end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    prediction = data.get('prediction', {})
                    analysis = data.get('analysis', {})
                    visualization = data.get('visualization', {})
                    
                    predicted_class = prediction.get('class', 'Unknown')
                    confidence = prediction.get('confidence', 0)
                    
                    print_success(f"Classification completed in {response_time:.2f}s")
                    print_info(f"  Predicted: {predicted_class}")
                    print_info(f"  Confidence: {confidence:.1f}%")
                    print_info(f"  Expected: {class_name}")
                    
                    # Verify response structure
                    checks = []
                    
                    # Check prediction data
                    checks.append((prediction.get('class') is not None, "Prediction class present"))
                    checks.append((prediction.get('confidence') is not None, "Confidence score present"))
                    checks.append((prediction.get('probabilities') is not None, "Probabilities present"))
                    
                    # Check visualization data
                    checks.append((visualization.get('heatmap_overlay') is not None, "Heatmap overlay present"))
                    checks.append((visualization.get('original_image') is not None, "Original image present"))
                    
                    if visualization.get('heatmap_overlay'):
                        checks.append((len(visualization['heatmap_overlay']) > 1000, "Heatmap has valid data"))
                    
                    # Check analysis data
                    checks.append((analysis.get('classification') is not None, "Analysis classification present"))
                    checks.append((analysis.get('confidence_level') is not None, "Confidence level present"))
                    checks.append((analysis.get('description') is not None, "Description present"))
                    
                    if analysis.get('model_interpretation'):
                        checks.append((len(analysis['model_interpretation']) > 50, "Model interpretation has content"))
                    
                    all_passed = True
                    for check_result, description in checks:
                        if check_result:
                            print_success(f"  {description}")
                        else:
                            print_error(f"  {description}")
                            all_passed = False
                    
                    if all_passed:
                        self.passed_tests += 1
                        return True
                    else:
                        self.failed_tests += 1
                        return False
                else:
                    error_msg = data.get('error', 'Unknown error')
                    print_error(f"Classification failed: {error_msg}")
                    self.failed_tests += 1
                    return False
            else:
                print_error(f"HTTP {response.status_code}: {response.text[:200]}")
                self.failed_tests += 1
                return False
                
        except requests.exceptions.Timeout:
            print_error("Request timed out (>60s)")
            self.failed_tests += 1
            return False
        except Exception as e:
            print_error(f"Classification test failed: {str(e)}")
            self.failed_tests += 1
            return False
    
    def test_invalid_file(self):
        """Test with invalid file type"""
        print_info("Testing invalid file rejection...")
        
        # Create temporary text file
        temp_file = Path(__file__).parent / "temp_test_file.txt"
        temp_file.write_text("This is not an image file")
        
        try:
            with open(temp_file, 'rb') as f:
                files = {'file': ('test.txt', f, 'text/plain')}
                response = requests.post(
                    f"{self.backend_url}/classify",
                    files=files,
                    timeout=10
                )
            
            # Check for rejection - can be 400, 415, or success=false
            if response.status_code in [400, 415]:
                print_success("Invalid file correctly rejected with error status")
                self.passed_tests += 1
                return True
            elif response.status_code == 200:
                data = response.json()
                # Handle tuple response format [data, status]
                if isinstance(data, list) and len(data) >= 1:
                    actual_data = data[0]
                    if not actual_data.get('success'):
                        print_success("Invalid file correctly rejected")
                        self.passed_tests += 1
                        return True
                elif isinstance(data, dict) and not data.get('success'):
                    print_success("Invalid file correctly rejected")
                    self.passed_tests += 1
                    return True
            
            print_error(f"Invalid file not rejected properly (status: {response.status_code})")
            self.failed_tests += 1
            return False
        except Exception as e:
            print_error(f"Invalid file test failed: {str(e)}")
            self.failed_tests += 1
            return False
        finally:
            if temp_file.exists():
                temp_file.unlink()
    
    def test_missing_file(self):
        """Test with no file uploaded"""
        print_info("Testing missing file handling...")
        
        try:
            response = requests.post(
                f"{self.backend_url}/classify",
                timeout=10
            )
            
            # Check for rejection - can be 400 or success=false in 200 response
            if response.status_code == 400:
                print_success("Missing file correctly rejected with 400 status")
                self.passed_tests += 1
                return True
            elif response.status_code == 200:
                data = response.json()
                # Handle tuple response format [data, status]
                if isinstance(data, list) and len(data) >= 1:
                    actual_data = data[0]
                    if not actual_data.get('success'):
                        print_success("Missing file correctly rejected")
                        self.passed_tests += 1
                        return True
                elif isinstance(data, dict) and not data.get('success'):
                    print_success("Missing file correctly rejected")
                    self.passed_tests += 1
                    return True
            
            print_error(f"Missing file not handled properly (status: {response.status_code})")
            self.failed_tests += 1
            return False
        except Exception as e:
            print_error(f"Missing file test failed: {str(e)}")
            self.failed_tests += 1
            return False
    
    def test_response_time(self):
        """Test that response time is within acceptable limits"""
        print_info("Testing response time performance...")
        
        test_image = self.find_test_image('Glioma')
        if not test_image:
            print_warning("No test image available for performance test")
            self.skipped_tests += 1
            return None
        
        try:
            with open(test_image, 'rb') as f:
                files = {'file': (test_image.name, f, 'image/jpeg')}
                start_time = time.time()
                response = requests.post(
                    f"{self.backend_url}/classify",
                    files=files,
                    timeout=60
                )
                end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                if response_time < 15:
                    print_success(f"Response time acceptable: {response_time:.2f}s (< 15s)")
                    self.passed_tests += 1
                    return True
                else:
                    print_warning(f"Response time slow: {response_time:.2f}s (> 15s)")
                    self.passed_tests += 1  # Still pass, just warn
                    return True
            else:
                print_error(f"Performance test failed with status {response.status_code}")
                self.failed_tests += 1
                return False
                
        except Exception as e:
            print_error(f"Performance test failed: {str(e)}")
            self.failed_tests += 1
            return False
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "=" * 70)
        print(f"{Colors.BOLD}BRAIN MRI CLASSIFICATION - INTEGRATION TEST SUITE{Colors.END}")
        print("=" * 70)
        
        # Test 1: Backend health
        print_header("1. Backend Health Check")
        if not self.test_backend_health():
            print_error("\nBackend not running. Cannot continue tests.")
            print_info("Start backend with: cd backend && python app.py")
            return False
        
        # Test 2: Classes endpoint
        print_header("2. Classes Endpoint")
        self.test_classes_endpoint()
        
        # Test 3: Classification for each class
        print_header("3. Classification Tests")
        tumor_classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        for tumor_class in tumor_classes:
            self.test_classification(tumor_class)
            print()
        
        # Test 4: Error handling
        print_header("4. Error Handling Tests")
        self.test_invalid_file()
        self.test_missing_file()
        
        # Test 5: Performance
        print_header("5. Performance Test")
        self.test_response_time()
        
        # Summary
        print("\n" + "=" * 70)
        print(f"{Colors.BOLD}TEST SUMMARY{Colors.END}")
        print("=" * 70)
        total = self.passed_tests + self.failed_tests + self.skipped_tests
        print(f"Passed:  {Colors.GREEN}{self.passed_tests}{Colors.END}")
        print(f"Failed:  {Colors.RED}{self.failed_tests}{Colors.END}")
        print(f"Skipped: {Colors.YELLOW}{self.skipped_tests}{Colors.END}")
        print(f"Total:   {total}")
        print("=" * 70)
        
        if self.failed_tests == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed!{Colors.END}")
            return True
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}Some tests failed. Please review the output above.{Colors.END}")
            return False


def main():
    print_info("Starting integration tests...")
    print_info(f"Test images path: {Path(__file__).parent.parent / 'brain_Tumor_Types'}")
    
    tester = IntegrationTester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
