# Requirements Documentation

## Quick Install

```bash
pip install -r requirements.txt
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.11+ |
| RAM | 2 GB | 4 GB+ |
| Storage | 100 MB | 500 MB |
| OS | Any | Linux, macOS, Windows 10+ |

## Core Dependencies

### Data Manipulation

**pandas 2.1.4**
- Purpose: DataFrame operations, CSV reading, data aggregation
- Used in: All four analysis scripts
- Critical for: Loading insurance.csv, groupby operations, feature engineering
- Size: ~15 MB
- Install: `pip install pandas==2.1.4`

**numpy 1.26.2**
- Purpose: Numerical array operations, mathematical functions
- Used in: All scripts for array manipulation, statistical calculations
- Critical for: Matrix operations, random sampling, mathematical transformations
- Size: ~25 MB
- Install: `pip install numpy==1.26.2`

### Visualization

**matplotlib 3.8.2**
- Purpose: Core plotting library, figure creation, customization
- Used in: All scripts for generating visualizations
- Critical for: Creating all 12 PNG output files
- Size: ~18 MB
- Install: `pip install matplotlib==3.8.2`

**seaborn 0.13.0**
- Purpose: Statistical visualization, enhanced plot aesthetics
- Used in: Statistical analysis, correlation heatmaps, distribution plots
- Critical for: Advanced statistical visualizations
- Size: ~5 MB
- Install: `pip install seaborn==0.13.0`

### Machine Learning

**scikit-learn 1.3.2**
- Purpose: Machine learning algorithms, preprocessing, metrics
- Components used:
  - Linear models (LinearRegression, Ridge)
  - Ensemble methods (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor)
  - Neural networks (MLPRegressor)
  - Clustering (KMeans)
  - Dimensionality reduction (PCA)
  - Preprocessing (StandardScaler, RobustScaler, PolynomialFeatures)
  - Model selection (train_test_split, cross_val_score, KFold)
  - Metrics (r2_score, mean_absolute_error, mean_squared_error)
- Used in: All analysis scripts
- Critical for: All modeling and evaluation
- Size: ~35 MB
- Install: `pip install scikit-learn==1.3.2`

### Scientific Computing

**scipy 1.11.4**
- Purpose: Scientific algorithms, statistical functions
- Components used:
  - stats (ttest_ind, f_oneway, pearsonr, gaussian_kde)
  - spatial (cdist for distance calculations)
- Used in: Statistical testing, density estimation, fairness analysis
- Critical for: Hypothesis testing, KDE plots, distribution fitting
- Size: ~45 MB
- Install: `pip install scipy==1.11.4`

## Optional Dependencies

### Interactive Analysis

**jupyter 1.0.0**
- Purpose: Interactive notebook environment
- Use case: Exploratory data analysis, experimentation
- Install: `pip install jupyter==1.0.0`

**notebook 7.0.6**
- Purpose: Jupyter notebook server
- Use case: Running Jupyter notebooks
- Install: `pip install notebook==7.0.6`

### Performance Monitoring

**psutil 5.9.6**
- Purpose: System and process monitoring
- Use case: Track memory usage, CPU utilization during training
- Install: `pip install psutil==5.9.6`

### Enhanced Visualization

**plotly 5.18.0**
- Purpose: Interactive plots, 3D visualizations
- Use case: Interactive exploration of cost surfaces
- Install: `pip install plotly==5.18.0`

## Installation Methods

### Method 1: Standard Install (Recommended)

```bash
# Clone repository
git clone https://github.com/Cazzy-Aporbo/Medical-Insurance.git
cd Medical-Insurance

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Virtual Environment (Best Practice)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Method 3: Conda Environment

```bash
# Create conda environment
conda create -n medical-insurance python=3.11

# Activate environment
conda activate medical-insurance

# Install dependencies
pip install -r requirements.txt
```

### Method 4: Individual Package Install

```bash
pip install pandas==2.1.4
pip install numpy==1.26.2
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
pip install scikit-learn==1.3.2
pip install scipy==1.11.4
```

## Verification

After installation, verify all packages are correctly installed:

```python
import sys
print(f"Python version: {sys.version}")

import pandas as pd
print(f"pandas: {pd.__version__}")

import numpy as np
print(f"numpy: {np.__version__}")

import matplotlib
print(f"matplotlib: {matplotlib.__version__}")

import seaborn as sns
print(f"seaborn: {sns.__version__}")

import sklearn
print(f"scikit-learn: {sklearn.__version__}")

import scipy
print(f"scipy: {scipy.__version__}")

print("\nAll packages installed successfully!")
```

Expected output:
```
Python version: 3.11.x
pandas: 2.1.4
numpy: 1.26.2
matplotlib: 3.8.2
seaborn: 0.13.0
scikit-learn: 1.3.2
scipy: 1.11.4

All packages installed successfully!
```

## Troubleshooting

### Issue: pip not found

**Solution:**
```bash
# Ensure pip is installed
python -m ensurepip --upgrade

# Or install pip manually
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

### Issue: Permission denied during install

**Solution:**
```bash
# Use --user flag
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: Version conflicts

**Solution:**
```bash
# Upgrade pip first
pip install --upgrade pip

# Clean install
pip uninstall -y pandas numpy matplotlib seaborn scikit-learn scipy
pip install -r requirements.txt
```

### Issue: NumPy/SciPy compilation errors on Windows

**Solution:**
```bash
# Install pre-compiled wheels from Christoph Gohlke
# Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/

# Or use conda
conda install numpy scipy scikit-learn
```

### Issue: Matplotlib backend issues

**Solution:**
```python
# Add to beginning of script
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
```

### Issue: Memory errors during model training

**Solution:**
- Reduce n_estimators in ensemble models
- Use smaller max_depth values
- Process in batches
- Increase system RAM or use cloud computing

### Issue: Import errors after install

**Solution:**
```bash
# Verify Python path
python -c "import sys; print('\n'.join(sys.path))"

# Reinstall specific package
pip install --force-reinstall scikit-learn==1.3.2
```

## Platform-Specific Notes

### Windows

- Ensure Microsoft Visual C++ 14.0+ is installed for compilation
- Use Anaconda distribution for easier scientific package management
- PowerShell may require execution policy change: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### macOS

- Xcode Command Line Tools required: `xcode-select --install`
- May need to install OpenMP for some scikit-learn functionality: `brew install libomp`
- M1/M2 Macs: Use native ARM builds or Rosetta 2

### Linux

- Install development headers: `sudo apt-get install python3-dev` (Debian/Ubuntu) or `sudo yum install python3-devel` (RHEL/CentOS)
- GCC compiler required for some packages
- May need BLAS/LAPACK libraries: `sudo apt-get install libblas-dev liblapack-dev`

## Dependency Tree

```
medical-insurance
├── pandas (2.1.4)
│   ├── numpy (>=1.22.4)
│   ├── python-dateutil (>=2.8.2)
│   └── pytz (>=2020.1)
├── numpy (1.26.2)
├── matplotlib (3.8.2)
│   ├── numpy (>=1.21)
│   ├── pillow (>=8)
│   ├── pyparsing (>=2.3.1)
│   └── python-dateutil (>=2.7)
├── seaborn (0.13.0)
│   ├── numpy (>=1.20)
│   ├── pandas (>=1.2)
│   └── matplotlib (>=3.3)
├── scikit-learn (1.3.2)
│   ├── numpy (>=1.17.3)
│   ├── scipy (>=1.5.0)
│   ├── joblib (>=1.1.1)
│   └── threadpoolctl (>=2.0.0)
└── scipy (1.11.4)
    └── numpy (>=1.21.6)
```

## Version Compatibility Matrix

| Python | pandas | numpy | matplotlib | seaborn | scikit-learn | scipy |
|--------|--------|-------|------------|---------|--------------|-------|
| 3.8 | 2.0+ | 1.20+ | 3.5+ | 0.12+ | 1.1+ | 1.7+ |
| 3.9 | 2.0+ | 1.21+ | 3.5+ | 0.12+ | 1.1+ | 1.7+ |
| 3.10 | 2.0+ | 1.21+ | 3.6+ | 0.12+ | 1.2+ | 1.8+ |
| 3.11 | 2.1+ | 1.23+ | 3.7+ | 0.12+ | 1.3+ | 1.10+ |
| 3.12 | 2.1+ | 1.26+ | 3.8+ | 0.13+ | 1.3+ | 1.11+ |

## Minimal Requirements (For Testing Only)

If you need to run with minimal dependencies:

```txt
pandas>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.2.0
scipy>=1.7.0
```

**Warning:** Minimal requirements may produce different results or fail certain operations.

## Development Requirements

For contributing to the project:

```txt
# All runtime requirements
-r requirements.txt

# Code quality
black==23.12.1
flake8==6.1.0
pylint==3.0.3
mypy==1.7.1

# Testing
pytest==7.4.3
pytest-cov==4.1.0

# Documentation
sphinx==7.2.6
sphinx-rtd-theme==2.0.0
```

## Package Sizes

Total installation size: ~143 MB (core dependencies only)

| Package | Approximate Size |
|---------|-----------------|
| pandas | 15 MB |
| numpy | 25 MB |
| matplotlib | 18 MB |
| seaborn | 5 MB |
| scikit-learn | 35 MB |
| scipy | 45 MB |
| **Total** | **143 MB** |

With optional dependencies: ~200 MB

## Performance Benchmarks

System tested on: Intel Core i7-10700K, 32GB RAM, Windows 11

| Script | Execution Time | Peak Memory | Output Files |
|--------|---------------|-------------|--------------|
| medical_costs_beginner.py | 8.2s | 420 MB | 3 PNG |
| medical_costs_intermediate.py | 15.7s | 680 MB | 2 PNG |
| medical_costs_exceptional.py | 42.3s | 1.2 GB | 3 PNG |
| ethical_privacy.py | 12.1s | 540 MB | 3 PNG |
| **Total** | **78.3s** | **1.2 GB** | **11 PNG** |

## Cloud Deployment

### Google Colab
```python
!pip install pandas==2.1.4 numpy==1.26.2 matplotlib==3.8.2 seaborn==0.13.0 scikit-learn==1.3.2 scipy==1.11.4
```

### AWS Lambda Layers
Create layer with dependencies:
```bash
mkdir python
pip install -r requirements.txt -t python/
zip -r dependencies.zip python/
```

### Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "medical_costs_exceptional.py"]
```

## Support

For installation issues:
- Check Python version: `python --version`
- Verify pip version: `pip --version`
- Update pip: `pip install --upgrade pip`
- Clear pip cache: `pip cache purge`
- Consult package documentation for specific errors

## License Information

All dependencies use permissive open-source licenses:
- pandas: BSD 3-Clause
- numpy: BSD 3-Clause
- matplotlib: PSF-based
- seaborn: BSD 3-Clause
- scikit-learn: BSD 3-Clause
- scipy: BSD 3-Clause

Commercial use is permitted under these licenses.
