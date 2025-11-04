#!/bin/bash

# Medical Insurance Cost Prediction - Installation Script
# Supports: Linux, macOS, Windows (Git Bash)

set -e  # Exit on error

echo "=============================================="
echo "Medical Insurance Cost Prediction Installer"
echo "=============================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or higher from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}Found Python $PYTHON_VERSION${NC}"

# Check if version is sufficient
MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
    echo -e "${RED}Error: Python 3.8 or higher required${NC}"
    exit 1
fi

# Check pip
echo ""
echo "Checking pip installation..."
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}Warning: pip not found, installing...${NC}"
    python3 -m ensurepip --upgrade
fi
echo -e "${GREEN}pip is available${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip to latest version..."
python3 -m pip install --upgrade pip --quiet

# Ask about virtual environment
echo ""
echo "Do you want to create a virtual environment? (recommended)"
read -p "Create virtual environment? [Y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows Git Bash
        source venv/Scripts/activate
    else
        # Linux/macOS
        source venv/bin/activate
    fi
    echo -e "${GREEN}Virtual environment created and activated${NC}"
fi

# Install requirements
echo ""
echo "Installing required packages..."
echo "This may take a few minutes..."
pip install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo -e "${GREEN}All packages installed successfully!${NC}"
else
    echo -e "${RED}Error during package installation${NC}"
    exit 1
fi

# Verify installation
echo ""
echo "Verifying installation..."

python3 << EOF
import sys
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import sklearn
import scipy

print(f"\nPython: {sys.version.split()[0]}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"seaborn: {sns.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"scipy: {scipy.__version__}")
print("\n✓ All packages verified successfully!")
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Package verification failed${NC}"
    exit 1
fi

# Check if data file exists
echo ""
if [ -f "insurance.csv" ]; then
    echo -e "${GREEN}✓ Dataset (insurance.csv) found${NC}"
else
    echo -e "${YELLOW}⚠ Dataset (insurance.csv) not found${NC}"
    echo "  Please download the dataset or ensure it's in the project directory"
fi

# Display usage instructions
echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
echo ""
echo "To run the analysis:"
echo ""
echo "  Foundation Analysis:"
echo "    python medical_costs_beginner.py"
echo ""
echo "  Intermediate Analysis:"
echo "    python medical_costs_intermediate.py"
echo ""
echo "  Exceptional Analysis:"
echo "    python medical_costs_exceptional.py"
echo ""
echo "  Ethical & Privacy Analysis:"
echo "    python ethical_privacy.py"
echo ""

if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "Remember to activate the virtual environment before running:"
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "  venv\\Scripts\\activate    (Windows)"
    else
        echo "  source venv/bin/activate  (Linux/macOS)"
    fi
    echo ""
fi

echo "For more information, see README.md or REQUIREMENTS.md"
echo ""
