# study_hours.py - very simple linear regression example

# 1. libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 2. dataset (hours studied -> score)
# small sample data: hours, score
# FIXED: make hours and scores the same length
hours = np.array([1.5, 2.0, 2.5, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.5, 9.0]).reshape(-1, 1)
scores = np.array([20, 25, 30, 40, 45, 50, 55, 65, 72, 85, 90])

# 3. split into train and test
X_train, X_test, y_train, y_test = train_test_split(hours, scores, test_size=0.2, random_state=42)

# 4. create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. evaluate on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.2f}")
print(f"Test R^2: {r2:.2f}")

# 6. use the model to predict new values
# change this value to test different hours
new_hours = np.array([[6.5]])
predicted_score = model.predict(new_hours)[0]
print(f"\nIf a student studies {new_hours[0][0]} hours, predicted score ≈ {predicted_score:.1f}")
