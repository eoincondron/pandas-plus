from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read version from __init__.py
version = {}
with open(os.path.join(this_directory, 'pandas_plus', '__init__.py')) as f:
    exec(f.read(), version)

setup(
    name="pandas-plus",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="High-performance extension package for pandas with fast groupby operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/pandas-plus",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "numba>=0.56.0",
        "polars>=0.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-xdist",
            "black",
            "flake8",
            "isort",
            "mypy",
            "bandit",
            "safety",
        ],
        "plotting": [
            "matplotlib>=3.0.0",
            "seaborn>=0.11.0",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-xdist",
        ],
    },
    entry_points={
        "console_scripts": [
            "pandas-plus=pandas_plus.__main__:main",
        ],
    },
    keywords="pandas groupby performance numba numpy data-analysis",
    project_urls={
        "Bug Reports": "https://github.com/your-username/pandas-plus/issues",
        "Source": "https://github.com/your-username/pandas-plus",
        "Documentation": "https://github.com/your-username/pandas-plus/blob/main/README.md",
    },
    include_package_data=True,
    zip_safe=False,
)