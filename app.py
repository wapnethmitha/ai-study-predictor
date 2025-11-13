from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from supabase import create_client, Client
import numpy as np, re, os
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json
from pathlib import Path
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default-secret-key-change-this")

# Supabase setup
try:
    supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    SUPABASE_ENABLED = True
except:
    SUPABASE_ENABLED = False
    print("⚠️ Supabase not configured. Using local storage only.")

# Users database file (local fallback)
USERS_FILE = Path("users.json")

def load_users():
    """Load users from JSON file"""
    if USERS_FILE.exists():
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def sync_user_to_supabase(email, username):
    """Sync user to Supabase database"""
    if not SUPABASE_ENABLED:
        return
    
    try:
        # Check if user exists in Supabase
        result = supabase.table("users").select("*").eq("email", email).execute()
        
        if not result.data:
            # Insert new user
            supabase.table("users").insert({
                "email": email,
                "username": username,
                "created_at": datetime.now().isoformat()
            }).execute()
            print(f"✅ User {email} synced to Supabase")
    except Exception as e:
        print(f"⚠️ Failed to sync user to Supabase: {e}")

# Train model once
hours = np.array([1.5, 2.0, 2.5, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.5, 9.0]).reshape(-1, 1)
scores = np.array([20, 25, 30, 40, 45, 50, 55, 65, 72, 85, 90])
X_train, X_test, y_train, y_test = train_test_split(hours, scores, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
mse, r2 = mean_squared_error(y_test, model.predict(X_test)), r2_score(y_test, model.predict(X_test))

# Helper functions
def extract_number(text):
    nums = re.findall(r'\d+\.?\d*', text)
    return float(nums[0]) if nums else None

def get_study_tips(hours_val, score_val):
    tips = []
    if hours_val <= 2:
        tips += ["💡 Use the Pomodoro Technique", "📚 Focus on key topics"]
    elif hours_val <= 4:
        tips += ["🎯 Mix study methods", "🧠 Spaced repetition"]
    elif hours_val <= 6:
        tips += ["🔄 Take regular breaks", "📊 Practice past papers"]
    else:
        tips += ["⚖️ Balance is key", "😴 Sleep well"]
    if score_val < 40:
        tips.append("🚀 Start with basics")
    elif score_val < 80:
        tips.append("🌟 Fine-tune your knowledge")
    else:
        tips.append("🏆 Excellent preparation")
    return tips

# -------------- ROUTES ------------------

@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', mse=round(mse,2), r2=round(r2,2), username=session.get('username', 'Student'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not email or not password:
            return render_template('login.html', error="Email and password are required.")

        users = load_users()
        
        # Check if user exists and password matches (local)
        if email in users and users[email]['password'] == password:
            session['user'] = email
            session['username'] = users[email].get('username', email.split('@')[0])
            session.permanent = True
            app.permanent_session_lifetime = 86400  # 24 hours
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid email or password. Check your credentials or register first.")
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not email or not password:
            return render_template('login.html', error="Email and password are required.")
        
        if len(password) < 6:
            return render_template('login.html', error="Password must be at least 6 characters.")
        
        if '@' not in email:
            return render_template('login.html', error="Invalid email format.")

        users = load_users()
        
        # Check if user already exists
        if email in users:
            return render_template('login.html', error="Email already registered. Please login instead.")
        
        username = email.split('@')[0]
        
        # Create new user in local storage
        users[email] = {
            'password': password,
            'username': username,
            'created_at': str(np.datetime64('today'))
        }
        save_users(users)
        
        # Sync to Supabase
        sync_user_to_supabase(email, username)
        
        # Auto login after registration
        session['user'] = email
        session['username'] = username
        session.permanent = True
        app.permanent_session_lifetime = 86400
        return redirect(url_for('home'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return jsonify({'error': 'Please log in first.'})
    
    try:
        data = request.json
        hours_value = float(data['hours'])
        
        if hours_value < 0 or hours_value > 10:
            return jsonify({'error': 'Study hours should be between 0 and 10.'})
        
        predicted_score = model.predict(np.array([[hours_value]]))[0]
        tips = get_study_tips(hours_value, predicted_score)
        return jsonify({'prediction': round(predicted_score, 1), 'tips': tips})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    if 'user' not in session:
        return jsonify({'response': 'Please log in first.'})

    message = request.json.get('message', '').lower()
    hours_val = extract_number(message)
    response_text = "I didn't understand. Try: 'I study 5 hours', 'I need 80 score', or 'How many hours for 75?'"

    # Pattern 1: "study X hours"
    if 'study' in message and 'hours' in message and hours_val is not None:
        if 0 <= hours_val <= 10:
            pred = model.predict(np.array([[hours_val]]))[0]
            tips = get_study_tips(hours_val, pred)
            tips_str = " | ".join(tips)
            response_text = f"If you study {hours_val} hours, your predicted score is **{pred:.1f}** 🎯\n\n💡 Tips: {tips_str}"
    
    # Pattern 2: "need/score X"
    elif any(word in message for word in ['score', 'need', 'get', 'mark', 'for']) and hours_val is not None:
        if 20 <= hours_val <= 100:
            score_val = hours_val
            hours_needed = (score_val - model.intercept_) / model.coef_[0]
            hours_needed = max(0, min(10, hours_needed))
            tips = get_study_tips(hours_needed, score_val)
            tips_str = " | ".join(tips)
            response_text = f"To get a score of **{score_val}**, you need to study approximately **{hours_needed:.1f} hours** ⏱️\n\n💡 Tips: {tips_str}"

    # Save chat to Supabase
    if SUPABASE_ENABLED:
        try:
            supabase.table("chats").insert({
                "user_email": session['user'],
                "message": message,
                "response": response_text,
                "created_at": datetime.now().isoformat()
            }).execute()
        except Exception as e:
            print(f"⚠️ Failed to save chat: {e}")

    return jsonify({'response': response_text})

@app.route('/history', methods=['GET'])
def history():
    if 'user' not in session:
        return jsonify([])
    
    if not SUPABASE_ENABLED:
        return jsonify([])
    
    try:
        result = supabase.table("chats").select("*").eq("user_email", session['user']).order("created_at", desc=True).limit(50).execute()
        return jsonify(result.data if result.data else [])
    except Exception as e:
        print(f"⚠️ Failed to load chat history: {e}")
        return jsonify([])

@app.route('/react', methods=['POST'])
def react():
    return jsonify({'success': True, 'reactions': {'thumbs_up': 0, 'thumbs_down': 0}})

@app.route('/reactions', methods=['GET'])
def get_reactions():
    return jsonify({'thumbs_up': 0, 'thumbs_down': 0})

if __name__ == '__main__':
    app.run(debug=True)
