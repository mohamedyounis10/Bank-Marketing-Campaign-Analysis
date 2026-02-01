<div align="center">

# 🏦 Bank Marketing Campaign Analysis

### Predicting Term Deposit Subscriptions with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-green.svg)](https://scikit-learn.org/)
[![Power BI](https://img.shields.io/badge/Power-BI-yellow.svg)](https://powerbi.microsoft.com/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Results](#-results)
- [Dashboard](#-dashboard)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 🎯 Overview

This project analyzes a bank's marketing campaign data to predict whether customers will subscribe to a term deposit. The analysis includes comprehensive Exploratory Data Analysis (EDA), data cleaning, feature engineering, and machine learning model evaluation using multiple algorithms.

**Goal**: Build a predictive model to identify customers most likely to subscribe to term deposits, helping the bank optimize marketing efforts and improve campaign success rates.

---

## 📁 Project Structure

```
Bank Marketing Campaign/
│
├── 📊 Dashboard/
│   ├── Bank.pbix                    # Power BI Dashboard
│   └── Dataset/
│       └── bank_cleaned.csv         # Cleaned dataset for visualization
│
├── 📂 Dataset/
│   └── bank-additional-full.csv     # Original dataset (41,188 records)
│
├── 📓 notebook.ipynb                # Main analysis notebook
│
└── 📄 README.md                     # Project documentation
```

### Structure Details

- **[Dashboard/](#dashboard)** - Contains the Power BI dashboard and cleaned dataset
  - `Bank.pbix` - Interactive Power BI dashboard for data visualization
  - `Dataset/bank_cleaned.csv` - Processed dataset ready for visualization

- **[Dataset/](#dataset)** - Contains the original dataset
  - `bank-additional-full.csv` - Raw marketing campaign data with 21 features

- **[notebook.ipynb](#usage)** - Main Jupyter notebook containing:
  - Data loading and exploration
  - Exploratory Data Analysis (EDA)
  - Data cleaning and preprocessing
  - Feature engineering
  - Model training and evaluation

---

## 📊 Dataset

The dataset contains information about a bank's marketing campaigns, including:

- **Total Records**: 41,188
- **Features**: 21 attributes
- **Target Variable**: `y` (yes/no - subscription to term deposit)

### Key Features

- **Demographic**: age, job, marital status, education
- **Financial**: default, housing loan, personal loan
- **Campaign**: contact type, month, day of week, duration, campaign number
- **Previous Campaign**: pdays, previous contacts, previous outcome
- **Economic Indicators**: employment variation rate, consumer price index, consumer confidence index, euribor 3 month rate, number of employees

### Dataset Source

The dataset is based on direct marketing campaigns of a Portuguese banking institution.

---

## ✨ Features

- ✅ **Comprehensive EDA**: Statistical analysis, visualizations, and data insights
- ✅ **Data Cleaning**: Missing value handling, duplicate removal, outlier treatment
- ✅ **Feature Engineering**: Data transformation and encoding
- ✅ **Multiple ML Models**: Logistic Regression, Random Forest, Gradient Boosting
- ✅ **Model Evaluation**: Cross-validation with multiple metrics (Accuracy, Precision, Recall, F1-Score)
- ✅ **Interactive Dashboard**: Power BI visualization for business insights
- ✅ **Handling Imbalanced Data**: Random Under Sampling technique

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- Power BI Desktop (for dashboard)

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

### Required Packages

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning algorithms
- `imbalanced-learn` - Handling imbalanced datasets

---

## 💻 Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/bank-marketing-campaign.git
   cd bank-marketing-campaign
   ```

2. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook notebook.ipynb
   ```

3. **Run the cells sequentially** to:
   - Load and explore the data
   - Perform EDA
   - Clean and preprocess the data
   - Train and evaluate models

4. **View the Dashboard**:
   - Open `Dashboard/Bank.pbix` in Power BI Desktop
   - Explore interactive visualizations

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Data information and description
- Missing values and duplicate detection
- Distribution analysis (histograms, KDE, boxplots, violin plots)
- Correlation analysis
- Relationship analysis between features and target

### 2. Data Cleaning
- Duplicate removal
- Outlier handling using IQR method
- Feature engineering (pdays transformation)

### 3. Data Preprocessing
- Categorical encoding using Label Encoder
- Feature selection (removed: cons.conf.idx, campaign, duration)
- Handling imbalanced data with Random Under Sampling

### 4. Modeling
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble tree-based model
- **Gradient Boosting**: Advanced boosting algorithm

### 5. Evaluation
- 5-fold Stratified Cross-Validation
- Metrics: Accuracy, Precision, Recall, F1-Score
- Best model selection based on F1-Score

---

## 📈 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| Logistic Regression | - | - | - | - |
| Random Forest | - | - | - | - |
| **Gradient Boosting** | - | - | - | **0.4589** |

🏆 **Best Model**: Gradient Boosting Classifier with F1-Score of **0.4589**

### Key Insights

- The dataset is imbalanced (more "no" than "yes" responses)
- Economic indicators show correlation with subscription rates
- Previous campaign outcomes influence current campaign success
- Age groups and job types have varying subscription rates

---

## 📊 Dashboard

The Power BI dashboard (`Dashboard/Bank.pbix`) provides:

- Interactive visualizations of campaign data
- Customer segmentation analysis
- Subscription rate trends
- Feature importance insights
- Business metrics and KPIs

<img width="1287" height="719" alt="Screenshot 2026-02-02 010322" src="https://github.com/user-attachments/assets/d0754b88-0947-419c-bf47-4ec249de671a" />

---

## 🛠️ Technologies Used

- **Python 3.12** - Programming language
- **Jupyter Notebook** - Interactive development environment
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning
- **Imbalanced-learn** - Handling imbalanced datasets
- **Power BI** - Business intelligence and visualization

---

## 👤 Author

**Mohamed**

- GitHub: [@mohamedyounis10](https://github.com/mohamedyounis10)
- LinkedIn: [mohamedyounis15](https://www.linkedin.com/in/mohamedyounis15/)

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**⭐ If you find this project helpful, please give it a star! ⭐**

Made with ❤️ for Data Science

</div>

