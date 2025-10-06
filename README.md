# 🤖 datafun-07_ml
In this project we explore machine learning (ML). We will consider supervised learning in the form of linear regression for a "best fit" understanding relative to our data.

## 📁 Project Structure

```text
datafun-07_ml/
├── data/                  # Datasets and raw data files
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code (Python scripts, modules)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore file
└── .venv/                 # Virtual environment (not tracked by git)
```

## ⚙️ Setup for the Project

Follow these steps to set up your project:

1. ## 🐙 Sign into GitHub.
2. ## 📁 Create a repository titled `datafun_07_ml` with a README.
3. ## 💻 Clone the repository to VS Code

    ```powershell
    git clone https://github.com/ksteele3712/datafun_07_ml
    ```

4. ## 🚫 Create a `.gitignore` file:
    - Purpose: Excludes files and folders from being tracked by git (e.g., virtual environments, temporary files, system files).
    - 
5. ## 📋 Create a `requirements.txt` file:
    - Purpose: Lists all Python packages needed for your project so others can install them easily with `pip install -r requirements.txt`.

6. ## 🖥️ Be sure to be in the right terminal directory before installing requirements:
    - 📂 Change to your project folder:
  
    ```powershell
    cd C:\Repos\datafun_07_ml
    ```

    - 🐍 Then use these commands to set up your Python environment:
  
    ```powershell
    py -m pip install --upgrade pip setuptools wheel
    py -m pip install --upgrade -r requirements.txt
    ```

    - 📦 Then use these commands to add in external requirements:
  
    ```powershell
    pip install jupyterlab numpy pandas pyarrow matplotlib seaborn scipy
    ```

7. ## 🚀 GitHub Commit Commands- Keep your Github up to Date 

Use these commands to commit and push your changes to GitHub:

```powershell
git add .
git commit -m "Initialize Repos in Github"
git push
```

8. ## 📓 Create a Jupyter Notebook 
   📝 First make a named ipynb file. Mine is kristinesteele-_ml.ipynb
   ⚙️ Then click 'select kernel'
   🐍 Click on the venv environment
   🔧 If not available and yet your venv has been activated in your terminal- then after clicking on 'select kernel' instead choosed 'select a different kernel', then 'Python environment' then click on 'venv' option

9. ## 📊 Delineate the Title and Meaning of Our Notebook
    This project demonstrates how we can make a "line of best fit" to model the meaning of our data. It is a sort of cause and effect pattern. Consequently, our line is relatively predictive in nature, helping us to see likely future results depending on our input.

10. ## 🐍 Python Imports for Machine Learning

Add these essential imports to your Jupyter notebook for machine learning analysis:

```python
# Core data analysis and numerical computing
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Standard library imports
import math
import statistics
from pathlib import Path

# Display settings for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('default')

print("All packages imported successfully! 📊🤖")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")
```

11. ## 📂 Importing Data Files


📁 To organize these files, we first created a `data` folder inside our `datafun_07_ml` project directory. We then downloaded the CSV files from their respective sources and placed them in the `data` folder:


🗂️ This structure keeps our data organized and makes it easy to load each file in our analysis notebooks using pandas.

