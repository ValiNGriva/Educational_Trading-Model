# S&P 500 Machine Learning Trading Model — WSL & Miniconda Setup Guide

This guide provides step-by-step instructions to configure a **Windows Subsystem for Linux (WSL)** environment with **Miniconda** to execute the three Jupyter Notebooks for the S&P 500 Trading Signal Model project.

---

## Step 1: Install & Enable WSL (Ubuntu)

1. Open **PowerShell** as Administrator on your Windows machine.
2. Run the following command to install WSL and Ubuntu:
   ```bash
   wsl --install
   ```
3. Restart your computer if prompted.
4. Open the **Ubuntu** application from your Windows Start Menu and set up your Linux username and password.

---

## Step 2: Install Miniconda in WSL

Inside your Ubuntu WSL terminal, run the following commands to download and install Miniconda:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y
sudo apt install wget curl git -y

# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Run installer
bash miniconda.sh -b -p $HOME/miniconda

# Initialize Conda in bash shell
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc
```

Verify the installation:
```bash
conda --version
```

---

## Step 3: Create Conda Environment from `environment.yml`

Navigate to your project directory inside WSL where `environment.yml` is located:

```bash
# Clone or navigate to your project folder
cd /path/to/your/project

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate sp500-ml-trading
```

---

## Step 4: Launch JupyterLab / Jupyter Notebook

With the `sp500-ml-trading` environment activated, launch JupyterLab:

```bash
jupyter lab --no-browser
```

Copy the local URL provided in the terminal (e.g., `http://127.0.0.1:8888/lab?token=...`) and paste it into your Windows web browser (Chrome, Edge, or Firefox).

---

## Step 5: Sequential Execution Order of Notebooks

Run the deliverables in the following chronological sequence:

1. **`1_Exploratory_Data_Analysis.ipynb`**:
   * Fetches historical OHLCV pricing data for `SPY`.
   * Computes technical indicators (RSI, MACD, Bollinger Bands, Moving Average ratios).
   * Applies **strict 1-day feature lagging (`shift(1)`)** to prevent look-ahead bias.
   * Constructs discrete 5-day future return target labels (`Buy 1`, `Sell -1`, `Hold 0`).
   * Visualizes class distributions and feature correlation heatmaps.

2. **`2_Modeling_Pipeline.ipynb`**:
   * Applies strict chronological splits (Train: 2010–2020, Validation: 2021–2022, Test: 2023–2024).
   * Constructs `sklearn.pipeline.Pipeline` workflows with `StandardScaler` and `class_weight='balanced'`.
   * Evaluates Baseline Logistic Regression, Random Forest, and Tuned HistGradientBoosting (`TimeSeriesSplit` CV).
   * Trains an **Isolation Forest** unsupervised regime filter to gate high-noise market periods.

3. **`3_Backtesting_and_Evaluation.ipynb`**:
   * Executes vectorized portfolio backtests with **0.1% transaction fee penalties**.
   * Evaluates cumulative return curves of the raw ML model vs. the Isolation Forest filtered strategy vs. the **Buy & Hold SPY** benchmark.
