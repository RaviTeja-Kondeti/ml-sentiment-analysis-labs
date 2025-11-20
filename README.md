# Machine Learning Engineering: Optimization & Classification Systems

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

Production-ready implementations of ML optimization algorithms and classification systems. Built from scratch to understand algorithmic internals and performance characteristics critical for enterprise AI systems.

## 🔍 What's Inside

### Core ML Engineering Skills Demonstrated

**Gradient Descent Optimization**
- Hyperparameter tuning analysis across 4 learning rate configurations
- Convergence stability testing and divergence detection
- Performance benchmarking against production libraries
- Production-grade error handling and validation

**Binary Classification Pipeline**
- From-scratch logistic regression implementation
- Healthcare data: Breast cancer detection (569 samples, 30 features)
- **92% classification accuracy** validated against scikit-learn
- Cross-entropy loss optimization with sigmoid activation

**Regression Modeling**
- Custom linear regression with MSE optimization
- Healthcare predictions: Diabetes progression modeling
- **R² = 0.51** with room for feature engineering improvements
- Production comparison with SGDRegressor

**LLM Integration** (Coming Soon)
- Zero-shot and few-shot prompting strategies
- Multi-model sentiment analysis (Claude 3, LLaMA)
- Real-world NLP application: Yelp reviews classification

## 🎯 Business Value

### Real-World Applications

1. **Healthcare AI**: Binary classification for medical diagnosis support
2. **Predictive Analytics**: Disease progression modeling for patient care
3. **Model Optimization**: Learning rate tuning for production ML systems
4. **Algorithm Validation**: Benchmarking custom implementations vs industry standard libraries

### Technical Highlights

- **Algorithm Optimization**: Systematic hyperparameter search yielding 2.6x faster convergence
- **Production Validation**: All implementations benchmarked against scikit-learn for correctness
- **Scalable Design**: NumPy vectorization for efficient matrix operations
- **Code Quality**: Clean, documented, reproducible implementations

## 🛠️ Technology Stack

```python
# Core ML & Data Science
import numpy as np           # Vectorized computations
import pandas as pd          # Data manipulation  
import sklearn              # Validation & metrics
import matplotlib.pyplot as plt  # Visualization

# LLM Integration (Phase 2)
from anthropic import Claude
import llama_cpp
```

## 📊 Performance Metrics

### Gradient Descent Convergence Analysis

| Learning Rate (α) | Outcome | Iterations | Speedup |
|------------------|---------|------------|----------|
| 0.05 | ✅ Converged | 13-14 | Baseline |
| **0.08** | ✅ **Optimal** | **5-6** | **2.6x** |
| 0.15 | ✅ Converged | 10-12 | 1.3x |
| 0.19 | ❌ Diverged | N/A | Failed |

**Key Insight**: Optimal learning rate selection provides 2.6x faster convergence while maintaining stability.

### Model Performance

#### Binary Classification (Healthcare)
```
Dataset: Breast Cancer Wisconsin (569 samples)
Model: Logistic Regression (from scratch)
Accuracy: 92.0%
Validation: Matches scikit-learn SGDClassifier
```

#### Regression (Healthcare)
```
Dataset: Diabetes Progression (442 samples)
Model: Linear Regression (custom implementation)
MSE: 2878.27
R²: 0.51
Validation: Benchmarked vs SGDRegressor
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/RaviTeja-Kondeti/ml-sentiment-analysis-labs.git
cd ml-sentiment-analysis-labs

# Install dependencies
pip install numpy pandas scikit-learn matplotlib jupyter

# Launch Jupyter
jupyter notebook
```

### Run Gradient Descent Optimization

```python
# Navigate to lab-assignment-3 folder
# Open: LA3_Kondeti_Ravi_Teja.ipynb

# Execute gradient descent scenarios
for scenario in scenarios:
    alpha = scenario['alpha']
    w1, costs = gradient_descent(alpha, n_iterations=25)
    plot_convergence(costs)
```

### Binary Classification Pipeline

```python
# Load breast cancer dataset
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

# Train custom logistic regression
model = LogisticRegressionCustom(learning_rate=2.5e-6, epochs=5000)
model.fit(X, y)

# Evaluate
accuracy = model.score(X, y)  # 92%
```

## 📁 Project Structure

```
ml-sentiment-analysis-labs/
│
├── lab-assignment-3/          # Core ML implementations
│   ├── LA3_Kondeti_Ravi_Teja.ipynb
│   ├── README.md
│   └── outputs/
│       ├── gradient_descent_convergence.png
│       └── classification_performance.png
│
├── lab-assignment-6/          # LLM applications (coming soon)
│   ├── sentiment_analysis.ipynb
│   └── data/
│
├── .gitignore
├── LICENSE
└── README.md
```

## 📈 Technical Deep Dives

### 1. Learning Rate Optimization

**Problem**: Finding optimal learning rate for gradient descent convergence.

**Approach**: Systematic grid search across 4 configurations (0.05, 0.08, 0.15, 0.19).

**Results**:
- α = 0.08 achieved fastest convergence (5-6 iterations)
- α = 0.19 caused divergence (oscillating loss)
- Optimal range identified: 0.08-0.15

**Business Impact**: 2.6x faster model training = reduced compute costs.

### 2. From-Scratch Implementation Strategy

**Why build from scratch?**
- Deep understanding of algorithm internals
- Ability to customize for specific use cases
- Debug production ML systems effectively
- Optimize for edge cases and performance

**Validation**: All implementations benchmarked against scikit-learn to ensure correctness.

### 3. Binary Classification for Healthcare

**Use Case**: Medical diagnosis support system

**Technical Stack**:
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + 
                    (1 - y_true) * np.log(1 - y_pred))

def train(X, y, learning_rate, epochs):
    w = np.zeros((X.shape[1], 1))
    b = 0
    
    for epoch in range(epochs):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)
        
        # Gradient descent
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        
        w -= learning_rate * dw
        b -= learning_rate * db
    
    return w, b
```

**Production Considerations**:
- Numerical stability (clip sigmoid outputs)
- Vectorized operations for performance
- Early stopping for efficiency
- Cross-validation for generalization

## 🔧 Engineering Best Practices

### Code Quality
- ✅ Vectorized NumPy operations (10-100x faster than loops)
- ✅ Modular, reusable functions
- ✅ Comprehensive validation against production libraries
- ✅ Clean, documented code

### Performance Optimization
- Matrix operations instead of loops
- Efficient memory usage with NumPy arrays
- Batch processing for large datasets
- Learning rate scheduling for faster convergence

### Validation Strategy
- Benchmark against scikit-learn
- Cross-validation for generalization
- Multiple metrics (accuracy, precision, recall, F1)
- Visualize convergence and performance

## 🚀 Production Roadmap

### Phase 1: Core ML (Completed)
- [x] Gradient descent optimization
- [x] Binary classification system
- [x] Regression modeling
- [x] Performance benchmarking

### Phase 2: LLM Integration (In Progress)
- [ ] Zero-shot sentiment classification
- [ ] Few-shot learning with Claude 3
- [ ] Multi-model comparison (Claude, LLaMA, GPT)
- [ ] Production prompt engineering patterns

### Phase 3: Enterprise Features (Planned)
- [ ] API endpoint for model serving
- [ ] Docker containerization
- [ ] CI/CD pipeline with automated testing
- [ ] Model monitoring and drift detection
- [ ] A/B testing framework

## 💼 Real-World Impact

### Healthcare AI Applications
1. **Breast Cancer Detection**: 92% accuracy classification system
2. **Disease Progression**: Predictive modeling for patient outcomes
3. **Clinical Decision Support**: ML-powered diagnosis assistance

### ML Engineering Lessons
1. **Hyperparameter tuning is critical**: 2.6x performance improvement
2. **Validation matters**: Always benchmark against production libraries
3. **Understanding internals**: Build from scratch before using frameworks
4. **Performance vs Accuracy**: Balance convergence speed with model quality

## 💡 Key Takeaways

### For ML Engineers
- Understanding optimization algorithms is crucial for production ML
- Learning rate selection significantly impacts model training efficiency
- From-scratch implementations provide deep algorithmic intuition
- Always validate custom code against established libraries

### For Data Scientists
- Systematic hyperparameter search yields measurable improvements
- Model performance should be benchmarked against baselines
- Visualization helps debug convergence issues
- Feature engineering remains critical (R² = 0.51 suggests room for improvement)

## 🔗 Connect

**Ravi Teja Kondeti**  
📍 Currently: ML Engineering & Quantitative Finance  
🎯 Focus: AI/ML Systems, HFT, Enterprise AI  
📚 Education: MSBA (ASU, 3.83 GPA), Financial Engineering (WorldQuant)  

*Building production ML systems and exploring AI applications in finance and healthcare.*

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🚀 Future Work

- **Model Optimization**: Implement Adam, RMSprop optimizers
- **Feature Engineering**: Polynomial features, interaction terms
- **Ensemble Methods**: Random forests, gradient boosting
- **Deep Learning**: Neural network implementations from scratch
- **Production Deployment**: REST API, containerization, monitoring
- **Real-time Systems**: Streaming ML for live predictions

---

**Status**: Active Development | **Last Updated**: November 2025
