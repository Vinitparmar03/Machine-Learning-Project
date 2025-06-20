import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("cardekho_dataset.csv")

df = df.drop_duplicates()

# Feature matrix and target
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Train-test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Categorical and numerical indices
categorical_indices = [0, 1, 2, 5, 6, 7]
numerical_indices = list(set(range(X.shape[1])) - set(categorical_indices))

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=10), categorical_indices),
    ('num', StandardScaler(), numerical_indices)
])

# Fit and transform
X_train_encoded = preprocessor.fit_transform(X_train)
X_val_encoded = preprocessor.transform(X_val)
X_test_encoded = preprocessor.transform(X_test)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% variance
X_train_pca = pca.fit_transform(X_train_encoded.toarray())
X_val_pca = pca.transform(X_val_encoded.toarray())
X_test_pca = pca.transform(X_test_encoded.toarray())

# Evaluation Function
def evaluate_model(model, X_val, y_val, name):
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)

    print(f"\n📌 {name}")
    print(f"R² Score : {r2:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"RMSE     : {rmse:.2f}")
    print(f"MAE      : {mae:.2f}")

    return r2, model

models = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_pca, y_train)
models["Linear Regression"] = evaluate_model(lr, X_val_pca, y_val, "Linear Regression")

# Polynomial Regression
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train_pca)
X_val_poly = poly_features.transform(X_val_pca)
poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train)
models["Polynomial Regression"] = evaluate_model(poly_lr, X_val_poly, y_val, "Polynomial Regression")

# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train_pca, y_train)
models["Decision Tree"] = evaluate_model(dt, X_val_pca, y_val, "Decision Tree")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)
models["Random Forest"] = evaluate_model(rf, X_val_pca, y_val, "Random Forest")

# SVR Linear
svr_linear = SVR(kernel='linear')
svr_linear.fit(X_train_pca, y_train)
models["SVR Linear"] = evaluate_model(svr_linear, X_val_pca, y_val, "SVR Linear")

# SVR RBF
svr_rbf = SVR(kernel='rbf')
svr_rbf.fit(X_train_pca, y_train)
models["SVR RBF"] = evaluate_model(svr_rbf, X_val_pca, y_val, "SVR RBF")

# XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb.fit(X_train_pca, y_train)
models["XGBoost"] = evaluate_model(xgb, X_val_pca, y_val, "XGBoost Regressor")

# Select best model
best_model_name = max(models, key=lambda k: models[k][0])
best_model = models[best_model_name][1]
print(f"\n✅ Best Model Based on Validation: {best_model_name}")

# Final Evaluation on Test Set
X_trainval_pca = np.vstack((X_train_pca, X_val_pca))
y_trainval = pd.concat([y_train, y_val])

# If poly, transform
if best_model_name == "Polynomial Regression":
    X_trainval_poly = poly_features.fit_transform(X_trainval_pca)
    X_test_poly = poly_features.transform(X_test_pca)
    best_model.fit(X_trainval_poly, y_trainval)
    y_test_pred = best_model.predict(X_test_poly)
else:
    best_model.fit(X_trainval_pca, y_trainval)
    y_test_pred = best_model.predict(X_test_pca)

# Final test evaluation
print(f"\n🧪 Final Evaluation on Test Set using {best_model_name}")
print(f"R² Score : {r2_score(y_test, y_test_pred):.4f}")
print(f"MSE      : {mean_squared_error(y_test, y_test_pred):.4f}")
print(f"RMSE     : {math.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
print(f"MAE      : {mean_absolute_error(y_test, y_test_pred):.2f}")


# ✅ Predict for a new unseen car (with auto column handling)
sample_car = pd.DataFrame([{
    "car_name": "Hyundai i20", "brand": "Hyundai", "model": "i20", "vehicle_age": 4, "km_driven": 35000,
    "seller_type": "Individual", "fuel_type": "Petrol", "transmission_type": "Manual",
    "mileage": 18.6, "engine": 1197, "max_power": 81.33, "seats": 5
}])

# Keep only columns that were used in training (X.columns)
sample_car = sample_car[X.columns]

# Transform and apply PCA
sample_encoded = preprocessor.transform(sample_car)
sample_pca = pca.transform(sample_encoded.toarray())

# Predict with best model
if best_model_name == "Polynomial Regression":
    sample_poly = poly_features.transform(sample_pca)
    pred_price = best_model.predict(sample_poly)[0]
else:
    pred_price = best_model.predict(sample_pca)[0]

# Print predicted price
print("\n🚗 Predicted Selling Price for Sample Car:")
print(f"Predicted Price: ₹{int(pred_price):,}")



# Plot
plt.scatter(y_test, y_test_pred, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Actual vs Predicted Price\n({best_model_name} on Test Set)')
plt.grid()
plt.show()
