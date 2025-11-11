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
    return render_template('index.html', mse=round(mse, 2), r2=round(r2, 2))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    hours_value = float(data['hours'])
    predicted_score = model.predict(np.array([[hours_value]]))[0]
    return jsonify({'prediction': round(predicted_score, 1)})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message'].lower()
    
    if 'study' in user_message and 'hours' in user_message:
        try:
            words = user_message.split()
            for i, word in enumerate(words):
                if word in ['study', 'studying']:
                    try:
                        hours_val = float(words[i+1])
                        if 0 <= hours_val <= 10:
                            pred = model.predict(np.array([[hours_val]]))[0]
                            return jsonify({'response': f"If you study {hours_val} hours, your predicted score is **{pred:.1f}** 🎯"})
                    except:
                        pass
        except:
            pass
    
    if 'score' in user_message and ('for' in user_message or 'get' in user_message):
        try:
            words = user_message.replace('?', '').split()
            for word in words:
                try:
                    num = float(word)
                    if 0 <= num <= 10:
                        pred = model.predict(np.array([[num]]))[0]
                        return jsonify({'response': f"Studying **{num}** hours should give you around **{pred:.1f}** score 📊"})
                except:
                    pass
        except:
            pass
    
    if 'hours' in user_message and ('for' in user_message or 'get' in user_message):
        try:
            words = user_message.replace('?', '').split()
            for word in words:
                try:
                    score = float(word)
                    if 20 <= score <= 90:
                        hours_needed = (score - model.intercept_) / model.coef_[0]
                        return jsonify({'response': f"To get a score of **{score}**, you need to study approximately **{hours_needed:.1f} hours** ⏱️"})
                except:
                    pass
        except:
            pass
    
    return jsonify({'response': "I didn't understand. Try: 'What if I study 5 hours?' or 'How many hours for 80 score?'"})

if __name__ == '__main__':
    app.run(debug=True)