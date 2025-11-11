from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import json

app = Flask(__name__)

# Train model once at startup
hours = np.array([1.5, 2.0, 2.5, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.5, 9.0]).reshape(-1, 1)
scores = np.array([20, 25, 30, 40, 45, 50, 55, 65, 72, 85, 90])
X_train, X_test, y_train, y_test = train_test_split(hours, scores, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

mse = mean_squared_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))

# Generate visualization
def generate_chart():
    # Create line for predictions across range
    x_range = np.linspace(0, 10, 100).reshape(-1, 1)
    y_pred_range = model.predict(x_range)
    
    fig = go.Figure()
    
    # Scatter plot for training data
    fig.add_trace(go.Scatter(
        x=hours.flatten(), y=scores,
        mode='markers',
        name='Training Data',
        marker=dict(size=10, color='#667eea', symbol='circle')
    ))
    
    # Line for model prediction
    fig.add_trace(go.Scatter(
        x=x_range.flatten(), y=y_pred_range,
        mode='lines',
        name='Model Prediction',
        line=dict(color='#764ba2', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title='📈 Study Hours vs Test Scores',
        xaxis_title='Study Hours',
        yaxis_title='Test Score',
        hovermode='closest',
        plot_bgcolor='#f5f5f5',
        width=800,
        height=500
    )
    
    return fig.to_html(include_plotlyjs='cdn')

chart_html = generate_chart()

@app.route('/')
def index():
    return render_template('index.html', mse=mse, r2=r2, chart=chart_html)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    hours_value = float(data['hours'])
    predicted_score = model.predict(np.array([[hours_value]]))[0]
    return jsonify({'prediction': round(predicted_score, 1)})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message'].lower()
    
    # Pattern: "what if I study X hours?"
    if 'study' in user_message and 'hours' in user_message:
        try:
            # Extract number from message
            words = user_message.split()
            for i, word in enumerate(words):
                if word in ['study', 'studying']:
                    try:
                        hours_val = float(words[i+1])
                        pred = model.predict(np.array([[hours_val]]))[0]
                        return jsonify({
                            'response': f"If you study {hours_val} hours, your predicted score is **{pred:.1f}** 🎯",
                            'type': 'prediction'
                        })
                    except:
                        pass
        except:
            pass
    
    # Pattern: "what score for X hours?"
    if 'score' in user_message and ('for' in user_message or 'get' in user_message):
        try:
            words = user_message.replace('?', '').split()
            for word in words:
                try:
                    num = float(word)
                    if 0 <= num <= 10:
                        pred = model.predict(np.array([[num]]))[0]
                        return jsonify({
                            'response': f"Studying **{num}** hours should give you around **{pred:.1f}** score 📊",
                            'type': 'prediction'
                        })
                except:
                    pass
        except:
            pass
    
    # Pattern: "how many hours for X score?"
    if 'hours' in user_message and ('for' in user_message or 'get' in user_message):
        try:
            words = user_message.replace('?', '').split()
            for word in words:
                try:
                    score = float(word)
                    if 20 <= score <= 90:
                        # Reverse prediction: (score - intercept) / slope
                        hours_needed = (score - model.intercept_) / model.coef_[0]
                        return jsonify({
                            'response': f"To get a score of **{score}**, you need to study approximately **{hours_needed:.1f} hours** ⏱️",
                            'type': 'prediction'
                        })
                except:
                    pass
        except:
            pass
    
    return jsonify({
        'response': "I didn't understand. Try asking: 'What if I study 5 hours?' or 'How many hours for 80 score?'",
        'type': 'error'
    })

if __name__ == '__main__':
    app.run(debug=True)