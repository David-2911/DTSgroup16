"""
Verify complete project structure for Brain MRI Classification Web Application
Checks that all required files are present
"""

import os
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def check_structure():
    """Verify all required files exist"""
    
    # Get webapp directory
    webapp_dir = Path(__file__).parent
    
    required_files = {
        'backend': [
            'app.py',
            'model_loader.py',
            'preprocessing.py',
            'gradcam.py',
            'analysis_generator.py',
            'utils.py',
            'config.py',
            'requirements.txt',
            'README.md'
        ],
        'frontend/src': [
            'App.jsx',
            'index.js',
            'index.css'
        ],
        'frontend/src/components': [
            'Header.jsx',
            'ImageUpload.jsx',
            'LoadingSpinner.jsx',
            'ClassificationResult.jsx',
            'HeatmapVisualization.jsx',
            'MedicalAnalysis.jsx',
            'ErrorMessage.jsx'
        ],
        'frontend/src/styles': [
            'App.css',
            'Header.css',
            'ImageUpload.css',
            'LoadingSpinner.css',
            'ClassificationResult.css',
            'HeatmapVisualization.css',
            'MedicalAnalysis.css',
            'ErrorMessage.css'
        ],
        'frontend/src/utils': [
            'api.js'
        ],
        'frontend': [
            'package.json'
        ],
        'shared': [
            'tumor_descriptions.json'
        ],
        '.': [
            'README.md',
            'DEPLOYMENT_CHECKLIST.md',
            'test_integration.py'
        ]
    }
    
    # Also check for model file in parent directory
    model_file = webapp_dir.parent / 'best_model_stage1.keras'
    
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}BRAIN MRI CLASSIFICATION - STRUCTURE VERIFICATION{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"\nWebapp directory: {webapp_dir}\n")
    
    all_good = True
    missing_files = []
    found_files = 0
    
    for directory, files in required_files.items():
        dir_path = webapp_dir / directory if directory != '.' else webapp_dir
        print(f"{Colors.BLUE}[DIR]{Colors.END} {directory}/")
        
        for file in files:
            file_path = dir_path / file
            if file_path.exists():
                print(f"  {Colors.GREEN}[OK]{Colors.END} {file}")
                found_files += 1
            else:
                print(f"  {Colors.RED}[MISSING]{Colors.END} {file}")
                missing_files.append(f"{directory}/{file}")
                all_good = False
        print()
    
    # Check model file
    print(f"{Colors.BLUE}[DIR]{Colors.END} ../")
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  {Colors.GREEN}[OK]{Colors.END} best_model_stage1.keras ({size_mb:.1f} MB)")
        found_files += 1
    else:
        print(f"  {Colors.RED}[MISSING]{Colors.END} best_model_stage1.keras")
        missing_files.append("../best_model_stage1.keras")
        all_good = False
    
    # Summary
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")
    
    total_files = sum(len(files) for files in required_files.values()) + 1  # +1 for model
    print(f"Files found:   {Colors.GREEN}{found_files}{Colors.END}")
    print(f"Files missing: {Colors.RED}{len(missing_files)}{Colors.END}")
    print(f"Total expected: {total_files}")
    
    if all_good:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All required files present!{Colors.END}")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}Missing files:{Colors.END}")
        for f in missing_files:
            print(f"  - {f}")
        return False


def check_dependencies():
    """Check if key dependencies are importable"""
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}DEPENDENCY CHECK{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}\n")
    
    dependencies = [
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
        ('tensorflow', 'TensorFlow'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
        ('pyspark', 'PySpark'),
    ]
    
    all_installed = True
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  {Colors.GREEN}[OK]{Colors.END} {name}")
        except ImportError:
            print(f"  {Colors.RED}[MISSING]{Colors.END} {name}")
            all_installed = False
    
    if all_installed:
        print(f"\n{Colors.GREEN}All Python dependencies available.{Colors.END}")
    else:
        print(f"\n{Colors.YELLOW}Some dependencies missing. Run:{Colors.END}")
        print("  pip install -r backend/requirements.txt")
    
    return all_installed


def main():
    structure_ok = check_structure()
    deps_ok = check_dependencies()
    
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.END}")
    if structure_ok and deps_ok:
        print(f"{Colors.GREEN}{Colors.BOLD}Project is ready for deployment!{Colors.END}")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}Please fix the issues above before deployment.{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}\n")
    
    return structure_ok and deps_ok


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
