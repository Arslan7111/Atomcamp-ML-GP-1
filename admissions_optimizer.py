
# admissions_optimizer.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data.csv')

# Separate features and target
X = df.drop('Chance of Admit', axis=1)
y = df['Chance of Admit']

# 1️⃣ Check multicollinearity using VIF
def calculate_vif(df_features):
    vif_data = pd.DataFrame()
    vif_data['feature'] = df_features.columns
    vif_data['VIF'] = [variance_inflation_factor(df_features.values, i)
                        for i in range(df_features.shape[1])]
    return vif_data

vif_before = calculate_vif(X)
print('\nVariance Inflation Factor (VIF) before handling multicollinearity:\n', vif_before)

# 2️⃣ Multiple Linear Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mlr = LinearRegression()
mlr.fit(X_train, y_train)
y_pred_mlr = mlr.predict(X_test)
print('\nMultiple Linear Regression R2:', r2_score(y_test, y_pred_mlr))
print('MLR RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_mlr)))

# 3️⃣ Decision Tree Regressor with hyperparameter tuning
dtree = DecisionTreeRegressor(random_state=42)
params_dtree = {'max_depth': [2, 3, 4, 5, 6, None],
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 2, 4]}
grid_dtree = GridSearchCV(dtree, params_dtree, cv=5, scoring='r2')
grid_dtree.fit(X_train, y_train)
best_dtree = grid_dtree.best_estimator_
y_pred_dtree = best_dtree.predict(X_test)
print('\nBest Decision Tree Parameters:', grid_dtree.best_params_)
print('Decision Tree R2:', r2_score(y_test, y_pred_dtree))

# 4️⃣ Random Forest Regressor with hyperparameter tuning
rf = RandomForestRegressor(random_state=42)
params_rf = {'n_estimators': [50, 100, 200],
              'max_depth': [3, 5, 7, None],
              'min_samples_split': [2, 5],
              'min_samples_leaf': [1, 2]}
random_rf = RandomizedSearchCV(rf, params_rf, n_iter=10, cv=3, scoring='r2', random_state=42, n_jobs=-1)
random_rf.fit(X_train, y_train)
best_rf = random_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print('\nBest Random Forest Parameters:', random_rf.best_params_)
print('Random Forest R2:', r2_score(y_test, y_pred_rf))

# 5️⃣ Feature Importance & Top 5 features
feature_importances = pd.Series(best_rf.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)
print('\nFeature Importances:\n', feature_importances)

top5_features = feature_importances.index[:5]
X_top5 = X[top5_features]

# Retrain models with top 5 features
X_train5, X_test5, y_train5, y_test5 = train_test_split(X_top5, y, test_size=0.2, random_state=42)

# MLR with top 5
mlr_top5 = LinearRegression()
mlr_top5.fit(X_train5, y_train5)
y_pred_mlr5 = mlr_top5.predict(X_test5)
print('\nMLR (Top 5) R2:', r2_score(y_test5, y_pred_mlr5))

# Decision Tree with top 5
best_dtree.fit(X_train5, y_train5)
y_pred_dtree5 = best_dtree.predict(X_test5)
print('Decision Tree (Top 5) R2:', r2_score(y_test5, y_pred_dtree5))

# Random Forest with top 5
best_rf.fit(X_train5, y_train5)
y_pred_rf5 = best_rf.predict(X_test5)
print('Random Forest (Top 5) R2:', r2_score(y_test5, y_pred_rf5))

# Optional: Visualizations
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

feature_importances.plot(kind='bar', title='Feature Importances')
plt.ylabel('Importance Score')
plt.show()
