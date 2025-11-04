from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="medical-insurance-cost-prediction",
    version="1.0.0",
    author="Cazandra Aporbo",
    author_email="cazandra.aporbo@foxxhealth.com",
    description="Comprehensive ML analysis predicting medical insurance costs with privacy and fairness auditing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cazzy-Aporbo/Medical-Insurance",
    project_urls={
        "Bug Tracker": "https://github.com/Cazzy-Aporbo/Medical-Insurance/issues",
        "Documentation": "https://htmlpreview.github.io/?https://github.com/Cazzy-Aporbo/Medical-Insurance/blob/main/medical_costs_complete_analysis.html",
        "Source Code": "https://github.com/Cazzy-Aporbo/Medical-Insurance",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.1.4",
        "numpy>=1.26.2",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "scikit-learn>=1.3.2",
        "scipy>=1.11.4",
    ],
    extras_require={
        "dev": [
            "black>=23.12.1",
            "flake8>=6.1.0",
            "pylint>=3.0.3",
            "mypy>=1.7.1",
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=2.0.0",
        ],
        "interactive": [
            "jupyter>=1.0.0",
            "notebook>=7.0.6",
            "plotly>=5.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medical-costs-beginner=medical_costs_beginner:main",
            "medical-costs-intermediate=medical_costs_intermediate:main",
            "medical-costs-exceptional=medical_costs_exceptional:main",
            "medical-costs-ethical=ethical_privacy:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.md", "*.html"],
    },
    keywords=[
        "machine learning",
        "healthcare",
        "insurance",
        "cost prediction",
        "random forest",
        "neural networks",
        "fairness",
        "privacy",
        "HIPAA",
        "GDPR",
        "ensemble methods",
        "data science",
    ],
)
