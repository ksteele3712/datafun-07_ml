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

---

## 🎯 **Machine Learning Project Accomplishments**

### **Project Overview: Linear Regression Analysis**
This project demonstrates practical machine learning through **supervised learning** using **linear regression** to analyze and predict NYC January temperatures. We built a complete data science pipeline from data acquisition to prediction and visualization.

---

### **🤖 Part 1 - Simple Linear Regression & Linear Relationships**

#### **What We Accomplished:**
- ✅ **Linear Regression Implementation**: Used scikit-learn's LinearRegression to model temperature trends
- ✅ **Real Data Analysis**: Analyzed NYC temperature data (1895-2018) for climate trend analysis  
- ✅ **Statistical Modeling**: Created predictive models showing temperature changes over time
- ✅ **Educational Examples**: Demonstrated linear relationships with Fahrenheit/Celsius conversion

#### **Key Machine Learning Concepts Demonstrated:**
- **🎯 Supervised Learning**: Training models with historical data to make predictions
- **📊 Independent/Dependent Variables**: Years (X) predict temperatures (y)
- **📈 Model Training**: `model.fit(X, y)` - machine "learned" from our data
- **🔮 Prediction Capability**: `model.predict()` - making future temperature forecasts
- **📉 Model Evaluation**: R-squared scores to assess prediction accuracy

#### **🌡️ Linear Relationships Theory:**
- Mathematical foundation with temperature conversion formulas
- Perfect linear relationship demonstration (F° to C° conversion)
- Understanding slope, intercept, and correlation concepts

---

### **🎯 Part 2 - Complete Machine Learning Pipeline**

#### **Full Data Science Workflow Implementation:**

**📊 Section 1 - Data Acquisition**
- Loaded NYC temperature dataset from CSV files
- Proper DataFrame creation and data structure setup

**🔍 Section 2 - Data Inspection**  
- Used `head()` and `tail()` methods for data exploration
- Analyzed data types and dataset characteristics

**🧹 Section 3 - Data Cleaning**
- Improved column names for clarity (Date→Year, Value→Temperature)
- Data type optimization and missing value assessment
- Data integrity verification

**📈 Section 4 - Descriptive Statistics**
- Comprehensive statistical analysis using `describe()`
- Mean, median, standard deviation calculations
- Temperature range and distribution analysis

**🤖 Section 5 - Model Building**
- **SciPy Linear Regression**: Used `stats.linregress()` for statistical modeling
- **Slope & Intercept Calculation**: Quantified temperature trend over time
- **Model Validation**: R-squared, p-values, and correlation coefficients

**🎯 Section 6 - Prediction**
- **2024 Temperature Forecast**: Applied trained model to predict future temperatures
- **Model Performance**: Compared predictions with historical data
- **Confidence Metrics**: Statistical significance assessment

**📊 Section 7 - Professional Visualizations**
- **Seaborn Integration**: Created publication-quality scatter plots with regression lines
- **Multi-panel Dashboards**: Comprehensive data visualization suite
- **Prediction Visualization**: Highlighted 2024 forecast on historical trends

---

### **🚀 Technical Skills Demonstrated**

#### **Machine Learning:**
- Linear regression modeling and training
- Supervised learning implementation
- Predictive modeling and forecasting
- Model evaluation and validation

#### **Data Science:**
- Complete data pipeline (acquisition → cleaning → modeling → visualization)
- Statistical analysis and hypothesis testing  
- Data preprocessing and feature engineering
- Professional data visualization

#### **Programming & Tools:**
- **Python Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy
- **Jupyter Notebooks**: Interactive development and documentation
- **Statistical Computing**: Advanced statistical analysis and modeling
- **Version Control**: Git workflow and project management

---

### **📈 Project Impact & Results**

**🌡️ Climate Analysis:**
- Quantified NYC temperature trends over 124 years (1895-2018)
- Identified measurable climate change patterns
- Created predictive models for future temperature forecasting

**🎓 Educational Value:**
- Demonstrated practical machine learning applications
- Connected mathematical theory with real-world data analysis
- Showcased professional data science workflow

**💼 Professional Development:**
- Industry-standard data science practices
- Statistical modeling expertise
- Professional documentation and presentation skills

---

## 📂 **Data Sources & File Organization**

### **🌡️ NYC Temperature Datasets**

This project utilizes historical NYC January temperature data spanning over 120 years for comprehensive climate trend analysis:

#### **📊 Data Files Used:**

**1. Part 1 Analysis:**
- **File**: `examples/ch10/ave_hi_nyc_jan_1895-2018.csv`
- **Purpose**: Initial linear regression demonstration and climate trend exploration
- **Data Structure**: Date (YYYYMM format), Temperature Value, Anomaly
- **Time Range**: January 1895 - January 2018 (124 years)

**2. Part 2 Complete Pipeline:**
- **File**: `examples/ch10/ave_hi_nyc2_jan_1895-2018.csv`  
- **Purpose**: Full data science workflow implementation and 2024 prediction
- **Data Structure**: Year, Temperature Value, Temperature Anomaly
- **Time Range**: 1895 - 2018 (124 years)
- **Features**: Cleaned data format optimized for machine learning analysis

#### **📁 Project Data Organization:**

```text
datafun-07_ml/
├── examples/
│   └── ch10/
│       ├── ave_hi_nyc_jan_1895-2018.csv      # Part 1 dataset
│       └── ave_hi_nyc2_jan_1895-2018.csv     # Part 2 dataset  
├── data/                                      # Additional datasets (if needed)
├── notebooks/
│   └── kristinesteele_ml.ipynb               # Main analysis notebook
└── README.md
```

#### **🌡️ Data Characteristics:**

- **Geographic Location**: New York City, NY
- **Measurement**: Average high temperatures in January
- **Units**: Degrees Fahrenheit (°F)
- **Data Quality**: Historical weather station records
- **Statistical Significance**: 124 years provides robust sample size for trend analysis
- **Climate Relevance**: January temperatures ideal for detecting long-term climate patterns

#### **📈 Data Usage in Analysis:**

**Independent Variable (X)**: Year (1895-2018)
- Used to predict temperature trends over time
- Represents temporal progression for climate change analysis

**Dependent Variable (y)**: Average High Temperature
- Target variable for machine learning predictions
- Measured in degrees Fahrenheit for NYC January conditions

**Additional Features**: Temperature Anomaly
- Deviation from long-term average temperatures
- Useful for identifying unusual climate patterns and variations

### **🔄 Data Loading Process:**

Both datasets are loaded using pandas for comprehensive analysis:

```python
# Part 1: Initial exploration
nyc_data = pd.read_csv('examples/ch10/ave_hi_nyc_jan_1895-2018.csv')

# Part 2: Full pipeline analysis  
nyc_df = pd.read_csv('examples/ch10/ave_hi_nyc2_jan_1895-2018.csv')
```

🗂️ This organized structure enables efficient data access and maintains clear separation between different analysis phases while ensuring reproducible research workflows.

