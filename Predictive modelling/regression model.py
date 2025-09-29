import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseVal')

print("--- Dataset Head ---")
print(X.head())
print("\nTarget variable (Median House Value) head:")
print(y.head())
print("\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples.")

print("\n--- Training Linear Regression Model ---")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
print("Linear Regression model trained successfully.")

y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("\n--- Linear Regression Performance ---")
print(f"Mean Squared Error (MSE): {mse_linear:.4f}") 
print(f"R-squared (R²): {r2_linear:.4f}") 

print("\n--- Training Random Forest Regressor Model ---")
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)
print("Random Forest model trained successfully.")

y_pred_forest = forest_model.predict(X_test)
mse_forest = mean_squared_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)

print("\n--- Random Forest Performance ---")
print(f"Mean Squared Error (MSE): {mse_forest:.4f}")
print(f"R-squared (R²): {r2_forest:.4f}")

print("\n--- Model Comparison ---")
print(f"Linear Regression R²: {r2_linear:.4f}")
print(f"Random Forest R²: {r2_forest:.4f}")
print("\nConclusion: The Random Forest Regressor performed significantly better, explaining a larger portion of the variance in house prices.")