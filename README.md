# 🤖 Machine Learning & NLP Lab Assignments

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive collection of machine learning and natural language processing lab assignments covering gradient descent optimization, classification algorithms, and large language model-based sentiment analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Assignments](#assignments)
- [Technologies Used](#technologies-used)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Results & Insights](#results--insights)
- [License](#license)
- [Author](#author)

## 🎯 Overview

This repository contains implementations and experiments from machine learning coursework, focusing on:

- **Optimization Algorithms**: Gradient descent with various learning rates
- **Classification Models**: Logistic and linear regression from scratch
- **Modern NLP**: LLM-based sentiment analysis with zero-shot and few-shot learning
- **Model Evaluation**: Comprehensive performance metrics and analysis

## 📚 Assignments

### Lab Assignment 3: Gradient Descent & Regression Models

**Objective**: Implement gradient descent optimization and build classification/regression models from scratch.

#### Key Components:

1. **Gradient Descent Analysis**
   - Explored multiple learning rate scenarios (α = 0.05, 0.08, 0.15, 0.19)
   - Analyzed convergence behavior and stability
   - Identified optimal learning rates for different cost functions

2. **Logistic Regression (Binary Classification)**
   - Dataset: Breast Cancer Wisconsin dataset (569 samples, 30 features)
   - Implementation: From-scratch sigmoid function, binary cross-entropy loss
   - Results: **92% training accuracy**
   - Validation: Compared with scikit-learn's SGDClassifier

3. **Linear Regression**
   - Dataset: Diabetes dataset (442 samples, 10 features)
   - Implementation: Mean squared error (MSE) optimization
   - Results: **MSE: 2878.27** | **R² Score: 0.51**
   - Validation: Benchmarked against SGDRegressor

#### Key Learnings:
- Learning rate selection critically impacts convergence
- Too high (α = 0.19): Model divergence with oscillations
- Too low (α = 0.05): Slow convergence requiring more iterations
- Optimal range (α = 0.08-0.15): Stable, efficient convergence

### Lab Assignment 6: LLM-Based Sentiment Analysis

**Objective**: Perform sentiment analysis on Yelp reviews using multiple large language models with various prompting strategies.

#### Key Components:

1. **Dataset Preparation**
   - Source: Yelp Restaurant Reviews (Arizona)
   - Sample: 100 reviews (50 positive, 50 negative)
   - Balanced dataset for unbiased evaluation

2. **Prompting Strategies**
   - **Zero-Shot Learning**: Direct sentiment prediction without examples
   - **Few-Shot Learning**: Guided predictions with 2-4 labeled examples
   - Comparative analysis of prompting effectiveness

3. **Multi-Model Comparison**
   - Primary Model: Claude 3 Sonnet
   - Alternative Models: LLaMA, GPT-based models
   - Cross-model performance evaluation

4. **Evaluation Metrics**
   - Precision, Recall, F1-Score
   - Accuracy across different prompting strategies
   - Error analysis and misclassification patterns

## 🛠️ Technologies Used

### Core Libraries
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Dataset loading, metrics, and model validation
- **Matplotlib**: Data visualization and loss curve plotting

### Machine Learning
- Custom implementations of gradient descent
- Logistic regression with sigmoid activation
- Linear regression with MSE optimization
- Model comparison and benchmarking

### NLP & LLMs
- Large Language Models (Claude 3, LLaMA)
- Prompt engineering (zero-shot, few-shot)
- Sentiment classification
- Performance evaluation frameworks

## 📁 Repository Structure

```
ml-sentiment-analysis-labs/
│
├── lab-assignment-3/
│   ├── LA3_Kondeti_Ravi_Teja.ipynb
│   ├── README.md
│   └── outputs/
│       ├── gradient_descent_curves.png
│       └── model_performance.png
│
├── lab-assignment-6/
│   ├── Lab-Assignment-6.pdf
│   ├── LA6_sentiment_analysis.ipynb
│   └── data/
│       └── restaurant_reviews_az.csv
│
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook or JupyterLab
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/RaviTeja-Kondeti/ml-sentiment-analysis-labs.git
cd ml-sentiment-analysis-labs
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

### Running the Assignments

**Lab Assignment 3:**
```bash
cd lab-assignment-3
jupyter notebook LA3_Kondeti_Ravi_Teja.ipynb
```

**Lab Assignment 6:**
```bash
cd lab-assignment-6
jupyter notebook LA6_sentiment_analysis.ipynb
```

## 📊 Results & Insights

### Gradient Descent Optimization

| Learning Rate (α) | Convergence | Iterations to Converge | Final Loss |
|------------------|-------------|------------------------|------------|
| 0.05 | ✅ Stable | ~13-14 | 3523.0 |
| 0.08 | ✅ Optimal | ~5-6 | 3523.0 |
| 0.15 | ✅ Fast | ~10-12 | 3523.0 |
| 0.19 | ❌ Diverged | N/A | Oscillating |

### Model Performance

#### Logistic Regression (Breast Cancer)
- **Training Accuracy**: 92%
- **Model Type**: Binary Classification
- **Features**: 30 clinical measurements
- **Implementation**: Custom gradient descent optimizer

#### Linear Regression (Diabetes)
- **Mean Squared Error**: 2878.27
- **R² Score**: 0.51
- **Features**: 10 baseline variables
- **Insights**: Moderate predictive power; potential for feature engineering

### Key Takeaways

1. **Learning Rate Selection**: Critical hyperparameter affecting training dynamics
2. **From-Scratch Implementation**: Deep understanding of algorithm internals
3. **Model Validation**: Importance of comparing custom implementations with established libraries
4. **LLM Prompting**: Few-shot learning significantly improves zero-shot performance
5. **Evaluation Rigor**: Multiple metrics provide comprehensive model assessment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Ravi Teja Kondeti**
- ASU ID: 1234434879
- University: Arizona State University
- Program: Master of Science in Business Analytics (MSBA)

---

### 📝 Acknowledgments

- Used GenAI tools for code optimization and documentation
- Breast Cancer Wisconsin dataset from scikit-learn
- Diabetes dataset from scikit-learn
- Yelp dataset for sentiment analysis experiments

### 🤝 Contributing

This repository is primarily for academic coursework. However, suggestions and feedback are welcome via issues or pull requests.

---

**Last Updated**: November 2025

*Part of MSBA coursework at Arizona State University*
