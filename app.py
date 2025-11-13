from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import re

app = Flask(__name__)

# Train model once at startup
hours = np.array([1.5, 2.0, 2.5, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.5, 9.0]).reshape(-1, 1)
scores = np.array([20, 25, 30, 40, 45, 50, 55, 65, 72, 85, 90])
X_train, X_test, y_train, y_test = train_test_split(hours, scores, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

mse = mean_squared_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))

# Track reactions
reactions = {'thumbs_up': 0, 'thumbs_down': 0}

def extract_number(text):
    """Extract first number from text"""
    numbers = re.findall(r'\d+\.?\d*', text)
    return float(numbers[0]) if numbers else None

def get_study_tips(hours_val, score_val):
    """Generate AI study tips based on hours and predicted score"""
    tips = []
    
    # Tips based on study hours
    if hours_val <= 2:
        tips.extend([
            "💡 **Use the Pomodoro Technique**: Study for 25 mins, break for 5 mins",
            "📚 **Focus on key topics**: Quality > Quantity. Master the essentials first",
            "⚡ **Active recall**: Test yourself frequently instead of passive reading"
        ])
    elif hours_val <= 4:
        tips.extend([
            "🎯 **Mix study methods**: Combine reading, writing, and practice problems",
            "🧠 **Spaced repetition**: Review material multiple times over days",
            "✍️ **Make summaries**: Write condensed notes of complex topics"
        ])
    elif hours_val <= 6:
        tips.extend([
            "🔄 **Take regular breaks**: Every 50 mins, take a 10 min break",
            "📊 **Practice past papers**: Solve previous exam questions",
            "💪 **Stay hydrated**: Drink water! Your brain works better hydrated"
        ])
    else:
        tips.extend([
            "⚖️ **Balance is key**: Don't burn out. Take proper rest",
            "🎧 **Minimize distractions**: Use focus apps or study in quiet places",
            "😴 **Sleep well**: 7-8 hours sleep helps consolidate learning"
        ])
    
    # Tips based on predicted score
    if score_val < 40:
        tips.append("🚀 **You've got this!** Start with basics and build gradually")
    elif score_val < 60:
        tips.append("📈 **Good progress!** Review weak areas and practice more")
    elif score_val < 80:
        tips.append("🌟 **Almost there!** Fine-tune your knowledge and do mock tests")
    else:
        tips.append("🏆 **Excellent preparation!** Revise and stay confident!")
    
    return tips

@app.route('/')
def index():
    return render_template('index.html', mse=round(mse, 2), r2=round(r2, 2))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    hours_value = float(data['hours'])
    predicted_score = model.predict(np.array([[hours_value]]))[0]
    tips = get_study_tips(hours_value, predicted_score)
    return jsonify({'prediction': round(predicted_score, 1), 'tips': tips})

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message'].lower()
    
    # Pattern 1: "study X hours" -> predict score
    if 'study' in user_message and 'hours' in user_message:
        hours_val = extract_number(user_message)
        if hours_val and 0 <= hours_val <= 10:
            pred = model.predict(np.array([[hours_val]]))[0]
            tips = get_study_tips(hours_val, pred)
            tips_text = "\n\n".join(tips)
            return jsonify({'response': f"If you study {hours_val} hours, your predicted score is **{pred:.1f}** 🎯\n\n**Study Tips:**\n{tips_text}"})
    
    # Pattern 2: "score X" or "take X" or "need X" (predict required hours)
    if any(word in user_message for word in ['score', 'take', 'need', 'get', 'marks', 'mark']):
        score_val = extract_number(user_message)
        if score_val and 20 <= score_val <= 100:
            hours_needed = (score_val - model.intercept_) / model.coef_[0]
            hours_needed = max(0, min(10, hours_needed))
            tips = get_study_tips(hours_needed, score_val)
            tips_text = "\n\n".join(tips)
            return jsonify({'response': f"To get a score of **{score_val}**, you need to study approximately **{hours_needed:.1f} hours** ⏱️\n\n**Study Tips:**\n{tips_text}"})
    
    # Pattern 3: "hours for X" (score query)
    if 'hours' in user_message and any(word in user_message for word in ['for', 'get', 'score']):
        score_val = extract_number(user_message)
        if score_val and 20 <= score_val <= 100:
            hours_needed = (score_val - model.intercept_) / model.coef_[0]
            hours_needed = max(0, min(10, hours_needed))
            tips = get_study_tips(hours_needed, score_val)
            tips_text = "\n\n".join(tips)
            return jsonify({'response': f"To get a score of **{score_val}**, you need to study approximately **{hours_needed:.1f} hours** ⏱️\n\n**Study Tips:**\n{tips_text}"})
    
    # Pattern 4: "X hours" -> predict score (simple)
    hours_val = extract_number(user_message)
    if hours_val and 'hours' in user_message and 0 <= hours_val <= 10:
        pred = model.predict(np.array([[hours_val]]))[0]
        tips = get_study_tips(hours_val, pred)
        tips_text = "\n\n".join(tips)
        return jsonify({'response': f"Studying **{hours_val}** hours should give you around **{pred:.1f}** score 📊\n\n**Study Tips:**\n{tips_text}"})
    
    return jsonify({'response': "I didn't understand. Try: 'I study 5 hours', 'I need 90 score', or 'How many hours for 80?'"})

@app.route('/react', methods=['POST'])
def react():
    data = request.json
    reaction_type = data.get('type')  # 'thumbs_up' or 'thumbs_down'
    
    if reaction_type in reactions:
        reactions[reaction_type] += 1
        return jsonify({'success': True, 'reactions': reactions})
    
    return jsonify({'success': False})

@app.route('/reactions', methods=['GET'])
def get_reactions():
    return jsonify(reactions)

if __name__ == '__main__':
    app.run(debug=True)