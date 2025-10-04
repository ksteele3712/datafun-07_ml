# datafun-07_ml
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

1. **Sign into GitHub.**
2. **Create a repository titled `datafun_07_ml` with a README.**
3. **Clone the repository to VS Code:**
    ```powershell
git clone https://github.com/ksteele3712/datafun_07_ml
    ```
4. **Create a `.gitignore` file:**
    - Purpose: Excludes files and folders from being tracked by git (e.g., virtual environments, temporary files, system files).
5. **Create a `requirements.txt` file:**
    - Purpose: Lists all Python packages needed for your project so others can install them easily with `pip install -r requirements.txt`.

6. **Be sure to be in the right terminal directory before installing requirements:**
    - Change to your project folder:
    ```powershell
cd C:\Repos\datafun_07_ml
    ```
    - Then use these commands to set up your Python environment:
    ```powershell
py -m pip install --upgrade pip setuptools wheel
py -m pip install --upgrade -r requirements.txt
    ```
    - Then use these commands to add in external requirements:
    ```powershell
pip install jupyterlab pandas pyarrow matplotlib seaborn
    ```

## 🚀 GitHub Commit Commands

Use these commands to commit and push your changes to GitHub:

```powershell
git add .
git commit -m "Initialize Repos in Github"
git push
```

## Create a file to work out of.

## 📂 Importing Data Files


To organize these files, we first created a `data` folder inside our `datafun_07_ml` project directory. We then downloaded the CSV files from their respective sources and placed them in the `data` folder:


This structure keeps our data organized and makes it easy to load each file in our analysis notebooks using pandas.

