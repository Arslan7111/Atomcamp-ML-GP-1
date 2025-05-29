
# Admissions Optimizer 🚀

This project builds a decision-support tool for Global Tech University to flag high-probability admits and forecast scholarship allocations using machine learning regression models.

## Dataset 📊
- `data.csv`: Admissions data (GRE, TOEFL, SOP, LOR, CGPA, Research, Chance of Admit).

## Key Tasks 📝
✅ Multiple Linear Regression for baseline analysis.  
✅ Decision Tree & Random Forest Regressors with hyperparameter tuning.  
✅ Multicollinearity handling using VIF.  
✅ Feature selection based on Random Forest importance.  
✅ Retraining models on top 5 features for comparison.  
✅ Visualizations for feature relationships.

## How to Run ⚙️
```bash
python admissions_optimizer.py
```
Install the required libraries:
```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
```

## Results 📈
- Models are evaluated using R² score & RMSE.  
- Key features identified for accurate predictions.

