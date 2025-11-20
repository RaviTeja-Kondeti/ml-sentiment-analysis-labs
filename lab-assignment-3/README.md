# Lab Assignment 3: Gradient Descent & Regression Models

## 🎯 Objective

Implement gradient descent optimization from scratch and build logistic/linear regression models for classification and regression tasks.

## 📝 Assignment Overview

This assignment explores fundamental machine learning concepts through hands-on implementation:

### Part 1: Gradient Descent Analysis

Experimented with different learning rates to understand convergence behavior:
- **α = 0.05**: Slow but stable convergence (~13-14 iterations)
- **α = 0.08**: Optimal balance (5-6 iterations) ✅
- **α = 0.15**: Fast convergence with slight oscillations (10-12 iterations)
- **α = 0.19**: Divergence - model unstable ❌

### Part 2: Logistic Regression (Binary Classification)

**Dataset**: Breast Cancer Wisconsin
- **Samples**: 569
- **Features**: 30 clinical measurements
- **Classes**: Malignant vs Benign

**Implementation**:
- Sigmoid activation function
- Binary cross-entropy loss
- Custom gradient descent optimizer

**Results**:
- Training Accuracy: **92%**
- Validated against scikit-learn's SGDClassifier

### Part 3: Linear Regression

**Dataset**: Diabetes Dataset
- **Samples**: 442
- **Features**: 10 baseline variables (age, sex, BMI, etc.)
- **Target**: Disease progression measure

**Implementation**:
- Mean Squared Error (MSE) loss function
- Gradient-based optimization

**Results**:
- Mean Squared Error: **2878.27**
- R² Score: **0.51**
- Benchmarked against SGDRegressor

## 📊 Key Visualizations

### Gradient Descent Convergence

| Learning Rate | Status | Iterations | Final Loss |
|--------------|--------|-----------|------------|
| 0.05 | ✅ Stable | 13-14 | 3523.0 |
| 0.08 | ✅ Optimal | 5-6 | 3523.0 |
| 0.15 | ✅ Fast | 10-12 | 3523.0 |
| 0.19 | ❌ Diverged | N/A | Oscillating |

### Loss Curves

Both logistic and linear regression models showed smooth convergence with appropriate learning rates, demonstrating stable optimization behavior.

## 🛠️ Technologies

- **NumPy**: Matrix operations and numerical computation
- **Pandas**: Data manipulation
- **Scikit-learn**: Datasets and validation metrics
- **Matplotlib**: Visualization

## 📚 Key Learnings

1. **Learning Rate Selection**: Critical hyperparameter that significantly impacts:
   - Convergence speed
   - Training stability
   - Final model performance

2. **From-Scratch Implementation**: Building algorithms from ground up provides:
   - Deep understanding of mathematical foundations
   - Insight into gradient flow and optimization dynamics
   - Appreciation for library implementations

3. **Model Validation**: Comparing custom implementations with scikit-learn:
   - Validates correctness of implementation
   - Provides performance benchmarks
   - Builds confidence in results

4. **Gradient Descent Behavior**:
   - **Too low learning rate**: Slow convergence, requires many iterations
   - **Optimal learning rate**: Fast, stable convergence
   - **Too high learning rate**: Oscillations and potential divergence

## 💾 Files

- `LA3_Kondeti_Ravi_Teja.ipynb`: Complete Jupyter notebook with implementation
- Loss curve visualizations
- Model performance plots

## 📈 Results Summary

### Logistic Regression
- ✅ Successfully classified breast cancer cases
- ✅ 92% accuracy on training data
- ✅ Implementation matches scikit-learn performance

### Linear Regression
- ✅ Predicted diabetes progression
- ✅ R² score of 0.51 indicates moderate predictive power
- ✅ MSE validated against scikit-learn

## 🔧 Running the Code

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib jupyter

# Launch Jupyter
jupyter notebook LA3_Kondeti_Ravi_Teja.ipynb

# Run all cells
# Cell -> Run All
```

## 💡 Future Improvements

1. **Regularization**: Add L1/L2 regularization to prevent overfitting
2. **Feature Engineering**: Create polynomial features for better fit
3. **Cross-Validation**: Implement k-fold CV for robust evaluation
4. **Adaptive Learning Rates**: Try Adam, RMSprop optimizers
5. **Early Stopping**: Monitor validation loss to prevent overtraining

---

**Author**: Ravi Teja Kondeti  
**ASU ID**: 1234434879  
**Course**: MSBA - Arizona State University  
**Date**: February 2025  
