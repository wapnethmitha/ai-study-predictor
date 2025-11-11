from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

# Train model once at startup
hours = np.array([1.5, 2.0, 2.5, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.5, 9.0]).reshape(-1, 1)
scores = np.array([20, 25, 30, 40, 45, 50, 55, 65, 72, 85, 90])
X_train, X_test, y_train, y_test = train_test_split(hours, scores, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

mse = mean_squared_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))

@app.route('/')
def index():
    return render_template('index.html', mse=mse, r2=r2)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    hours_value = float(data['hours'])
    predicted_score = model.predict(np.array([[hours_value]]))[0]
    return jsonify({'prediction': round(predicted_score, 1)})

if __name__ == '__main__':
    app.run(debug=True)