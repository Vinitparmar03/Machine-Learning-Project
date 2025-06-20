import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
import os
from joblib import parallel_backend

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

# 🔇 Suppress known harmless warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ✅ Custom transformer to convert sparse to dense
class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.toarray() if hasattr(X, "toarray") else X

# 1. Load and clean dataset
df = pd.read_csv("cardekho_dataset.csv")
df = df.drop_duplicates()
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# 2. Define categorical and numerical indices
categorical_indices = [0, 1, 2, 5, 6, 7]
numerical_indices = list(set(range(X.shape[1])) - set(categorical_indices))

# 3. Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_indices),
    ('num', StandardScaler(), numerical_indices)
])

# 4. Pipeline: Preprocessing → Dense → PCA → XGBoost
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("to_dense", DenseTransformer()),
    ("pca", PCA(n_components=0.95)),
    ("regressor", XGBRegressor(objective="reg:squarederror", random_state=42))
])

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Hyperparameter grid
param_grid = {
    "regressor__n_estimators": [100, 150, 200],
    "regressor__learning_rate": [0.05, 0.1, 0.2],
    "regressor__max_depth": [3, 4]
}

# ✅ Fix parallel processing warning by limiting CPU count
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# 7. GridSearch with safe backend
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
with parallel_backend('loky'):
    grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# 8. Evaluate on test set
y_pred = best_model.predict(X_test)
print(f"\n✅ Best Hyperparameters: {grid_search.best_params_}")
print(f"R² Score : {r2_score(y_test, y_pred):.4f}")
print(f"MSE      : {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE     : {math.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE      : {mean_absolute_error(y_test, y_pred):.2f}")

# 9. Predict a sample car
sample_car = pd.DataFrame([{
    "car_name": "Hyundai i20", "brand": "Hyundai", "model": "i20", "vehicle_age": 4, "km_driven": 35000,
    "seller_type": "Individual", "fuel_type": "Petrol", "transmission_type": "Manual",
    "mileage": 18.6, "engine": 1197, "max_power": 81.33, "seats": 5
}])
sample_car = sample_car[X.columns]
pred_price = best_model.predict(sample_car)[0]
print("\n🚗 Predicted Selling Price for Sample Car:")
print(f"Predicted Price: ₹{int(pred_price):,}")

# 10. Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price (XGBoost + PCA)')
plt.grid()
plt.tight_layout()
plt.show()

# 11. Check cross-validation score variance
cv_results = grid_search.cv_results_
mean_scores = cv_results["mean_test_score"]
std_scores = cv_results["std_test_score"]

# 🔥 Just show the best one
best_index = grid_search.best_index_
best_params = cv_results["params"][best_index]
best_mean = mean_scores[best_index]
best_std = std_scores[best_index]
best_var = best_std ** 2

print("\n📊 Best CV Result:")
print(f"✅ Best Params  : {best_params}")
print(f"🔹 Mean R²      : {best_mean:.4f}")
print(f"🔸 Std Dev      : {best_std:.4f}")
print(f"📈 Variance     : {best_var:.6f}")