# Jupyter Notebook Installation and Setup Guide

## Prerequisites
- Python 3.7 or higher installed on your system
- pip package manager

## 1. Create and Activate Virtual Environment

### On Windows:
```bash
# Create virtual environment
python -m venv jupyter_env

# Activate virtual environment
jupyter_env\Scripts\activate
```

### On macOS/Linux:
```bash
# Create virtual environment
python3 -m venv jupyter_env

# Activate virtual environment
source jupyter_env/bin/activate
```

## 2. Install Jupyter Notebook

```bash
# Upgrade pip first
pip install --upgrade pip

# Install Jupyter Notebook
pip install jupyter
```

## 3. Install Essential Dependencies

Install the core data science and analysis libraries:

```bash
# Install numpy (as specified)
pip install numpy

# Install other common dependencies
pip install pandas matplotlib seaborn scikit-learn scipy

# Install additional useful packages
pip install ipykernel ipywidgets
```

## 4. Register Virtual Environment with Jupyter

```bash
# Make your virtual environment available as a kernel in Jupyter
python -m ipykernel install --user --name=jupyter_env --display-name="Python (jupyter_env)"
```

## 5. Launch Jupyter Notebook

```bash
# Start Jupyter Notebook server
jupyter notebook
```

This will open Jupyter Notebook in your default web browser at `http://localhost:8888`

## 6. Basic Usage Commands

### Creating a New Notebook
1. Click "New" → "Python (jupyter_env)" in the Jupyter interface
2. This creates a new notebook using your virtual environment

### Basic Cell Operations
- **Run cell**: `Shift + Enter`
- **Insert cell above**: `A` (in command mode)
- **Insert cell below**: `B` (in command mode)
- **Delete cell**: `DD` (in command mode)
- **Change to Markdown**: `M` (in command mode)
- **Change to Code**: `Y` (in command mode)

## 7. Verify Installation

Test your setup with this code in a new notebook cell:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Test numpy
arr = np.array([1, 2, 3, 4, 5])
print(f"NumPy array: {arr}")
print(f"NumPy version: {np.__version__}")

# Verify other packages
print(f"Pandas version: {pd.__version__}")
```

## 8. Deactivating Environment

When you're done working:

```bash
# Deactivate virtual environment
deactivate
```

## 9. Reactivating for Future Sessions

To work with your setup again:

### Windows:
```bash
jupyter_env\Scripts\activate
jupyter notebook
```

### macOS/Linux:
```bash
source jupyter_env/bin/activate
jupyter notebook
```

## Requirements File (Optional)

Create a `requirements.txt` file for easy dependency management:

```txt
jupyter
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
ipykernel
ipywidgets
```

Install all dependencies from requirements file:
```bash
pip install -r requirements.txt
```

## Troubleshooting

- If Jupyter doesn't start, ensure your virtual environment is activated
- If you can't see your kernel, re-run the ipykernel install command
- For permission issues on Windows, run terminal as administrator
- For macOS/Linux permission issues, avoid using `sudo` with pip in virtual environments