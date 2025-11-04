# Quick Installation Reference

## One-Line Install

```bash
pip install -r requirements.txt
```

## Package Summary

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.1.4 | Data manipulation |
| numpy | 1.26.2 | Numerical computing |
| matplotlib | 3.8.2 | Visualization |
| seaborn | 0.13.0 | Statistical plots |
| scikit-learn | 1.3.2 | Machine learning |
| scipy | 1.11.4 | Scientific computing |

**Total Size:** ~143 MB

## Platform-Specific Quick Start

### Windows
```batch
install.bat
```

### Linux/macOS
```bash
chmod +x install.sh
./install.sh
```

### Conda
```bash
conda env create -f environment.yml
conda activate medical-insurance
```

## Verification

```python
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, scipy; print('✓ All packages installed')"
```

## Troubleshooting

**Problem:** Permission denied  
**Solution:** `pip install --user -r requirements.txt`

**Problem:** Version conflicts  
**Solution:** Use virtual environment

**Problem:** Compilation errors  
**Solution:** Use Anaconda or pre-compiled wheels

## System Requirements

- Python 3.8+
- 2 GB RAM minimum
- 500 MB disk space

## Support

See REQUIREMENTS.md for detailed installation guide and troubleshooting.
