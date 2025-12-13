#!/bin/bash
# =============================================================================
# Cleanup Script for Academic Submission
# DTS 301 - Big Data Computing | Group 16
# =============================================================================
# This script removes all files that should NOT be included in the submission:
# - Virtual environment (3.3 GB) - reinstallable via requirements.txt
# - node_modules (418 MB) - reinstallable via npm install
# - Git history - not needed for submission
# - Python cache files - auto-generated
# - Log files - runtime artifacts
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "Brain MRI Classification - Submission Cleanup Script"
echo "============================================================"
echo ""

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}Current directory:${NC} $(pwd)"
echo ""

# Show current size
echo "Current project size:"
du -sh .
echo ""

# Confirm before proceeding
echo -e "${RED}WARNING: This will permanently delete the following:${NC}"
echo "  - dtsvenv/          (Python virtual environment)"
echo "  - webapp/frontend/node_modules/  (Node.js packages)"
echo "  - .git/             (Git repository history)"
echo "  - .gitignore        (Git ignore file)"
echo "  - __pycache__/      (Python bytecode cache)"
echo "  - *.pyc files       (Compiled Python files)"
echo "  - server.log        (Runtime logs)"
echo "  - nohup.out         (Background process output)"
echo ""

read -p "Are you sure you want to proceed? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Starting cleanup..."
echo ""

# Step 1: Remove virtual environment
if [ -d "dtsvenv" ]; then
    echo -e "${YELLOW}[1/6]${NC} Removing virtual environment (dtsvenv/)..."
    rm -rf dtsvenv/
    echo -e "${GREEN}      Done.${NC}"
else
    echo -e "${YELLOW}[1/6]${NC} Virtual environment not found, skipping."
fi

# Step 2: Remove node_modules
if [ -d "webapp/frontend/node_modules" ]; then
    echo -e "${YELLOW}[2/6]${NC} Removing node_modules..."
    rm -rf webapp/frontend/node_modules/
    echo -e "${GREEN}      Done.${NC}"
else
    echo -e "${YELLOW}[2/6]${NC} node_modules not found, skipping."
fi

# Step 3: Remove git history
if [ -d ".git" ]; then
    echo -e "${YELLOW}[3/6]${NC} Removing git history (.git/)..."
    rm -rf .git/
    rm -f .gitignore
    echo -e "${GREEN}      Done.${NC}"
else
    echo -e "${YELLOW}[3/6]${NC} Git history not found, skipping."
fi

# Step 4: Remove Python cache
echo -e "${YELLOW}[4/6]${NC} Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}      Done.${NC}"

# Step 5: Remove log files
echo -e "${YELLOW}[5/6]${NC} Removing log files..."
rm -f webapp/backend/server.log 2>/dev/null || true
rm -f webapp/backend/nohup.out 2>/dev/null || true
rm -f nohup.out 2>/dev/null || true
echo -e "${GREEN}      Done.${NC}"

# Step 6: Clear temp uploads (but keep directory)
echo -e "${YELLOW}[6/6]${NC} Clearing temp uploads..."
rm -f webapp/backend/temp_uploads/* 2>/dev/null || true
echo -e "${GREEN}      Done.${NC}"

echo ""
echo "============================================================"
echo -e "${GREEN}Cleanup complete!${NC}"
echo "============================================================"
echo ""

# Show new size
echo "New project size:"
du -sh .
echo ""

# List main directories
echo "Remaining directories:"
du -sh */ 2>/dev/null | sort -rh
echo ""

# Verify required files still exist
echo "Verifying essential files..."
essential_files=(
    "best_model_stage1.keras"
    "Brain_MRI_Distributed_DL.ipynb"
    "requirements.txt"
    "README.md"
    "webapp/backend/app.py"
    "webapp/frontend/package.json"
    "webapp/frontend/src/App.jsx"
    "brain_Tumor_Types/glioma"
    "brain_Tumor_Types/meningioma"
    "brain_Tumor_Types/notumor"
    "brain_Tumor_Types/pituitary"
)

missing=0
for file in "${essential_files[@]}"; do
    if [ ! -e "$file" ]; then
        echo -e "${RED}MISSING:${NC} $file"
        missing=$((missing + 1))
    fi
done

if [ $missing -eq 0 ]; then
    echo -e "${GREEN}All essential files present!${NC}"
else
    echo -e "${RED}Warning: $missing essential file(s) missing!${NC}"
fi

echo ""
echo "============================================================"
echo "Next steps:"
echo "1. Review the cleaned directory"
echo "2. Create ZIP file:"
echo "   cd .. && zip -r DTSgroup16_submission.zip DTSgroup16/"
echo "============================================================"
