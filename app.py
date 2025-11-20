
import os
import json
import numpy as np
import re
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from supabase import create_client, Client
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
from datetime import datetime
from face_analyzer import capture_face_emotion
from pathlib import Path
import base64
import cv2
from face_analyzer import analyze_face
from flask import request, jsonify, session

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

        if SUPABASE_ENABLED:
            try:
                result = supabase.table("users").select("*").eq("email", email).execute()
                user = result.data[0] if result.data else None

                if user and user['password'] == password:
                    session['user'] = email
                    session['username'] = user.get('username', email.split('@')[0])
                    session.permanent = True
                    app.permanent_session_lifetime = 86400  # 24 hours
                    return redirect(url_for('home'))
                else:
                    return render_template('login.html', error="Invalid email or password. Please register first.")
            except Exception as e:
                return render_template('login.html', error=f"Supabase error: {e}")
        else:
            return render_template('login.html', error="Supabase not enabled.")

    return render_template('login.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        # Basic validation
        if not email or not password:
            return render_template('login.html', error="Email and password are required.")
        if len(password) < 6:
            return render_template('login.html', error="Password must be at least 6 characters.")
        if '@' not in email:
            return render_template('login.html', error="Invalid email format.")

        username = email.split('@')[0]

        if SUPABASE_ENABLED:
            try:
                # Check if user already exists
                result = supabase.table("users").select("*").eq("email", email).execute()
                if result.data:
                    return render_template('login.html', error="Email already registered. Please login instead.")

                # Insert new user
                supabase.table("users").insert({
                    "email": email,
                    "username": username,
                    "password": password,
                    "created_at": datetime.now().isoformat()
                }).execute()

                # Auto login
                session['user'] = email
                session['username'] = username
                session.permanent = True
                app.permanent_session_lifetime = 86400  # 24 hours
                return redirect(url_for('home'))

            except Exception as e:
                return render_template('login.html', error=f"Supabase error: {e}")
        else:
            return render_template('login.html', error="Supabase not enabled.")

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
        # Get JSON data from request
        data = request.json
        hours_value = float(data.get('hours', 0))

        # Validate hours
        if hours_value < 0 or hours_value > 10:
            return jsonify({'error': 'Study hours should be between 0 and 10.'})

        # Predict score using the trained model
        predicted_score = model.predict(np.array([[hours_value]]))[0]

        # Get study tips based on hours and predicted score
        tips = get_study_tips(hours_value, predicted_score)

        return jsonify({
            'prediction': round(predicted_score, 1),
            'tips': tips
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    if 'user' not in session:
        return jsonify({'response': 'Please log in first.'})

    message = request.json.get('message', '').strip().lower()
    hours_val = extract_number(message)
    response_text = (
        "I didn't understand. Try: 'I study 5 hours', "
        "'I need 80 score', or 'How many hours for 75?'"
    )

    # -----------------------------
    # AI PREDICTION LOGIC
    # -----------------------------
    if 'study' in message and 'hours' in message and hours_val is not None:
        if 0 <= hours_val <= 10:
            predicted_score = model.predict(np.array([[hours_val]]))[0]
            tips = get_study_tips(hours_val, predicted_score)
            tips_str = " | ".join(tips)
            response_text = (
                f"If you study {hours_val} hours, your predicted score is {predicted_score:.1f} 🎯\n"
                f"💡 Tips: {tips_str}"
            )
    elif any(word in message for word in ['score', 'need', 'get', 'mark', 'for']) and hours_val is not None:
        if 20 <= hours_val <= 100:
            score_val = hours_val
            hours_needed = (score_val - model.intercept_) / model.coef_[0]
            hours_needed = max(0, min(10, hours_needed))
            tips = get_study_tips(hours_needed, score_val)
            tips_str = " | ".join(tips)
            response_text = (
                f"To get a score of {score_val}, you need to study approximately {hours_needed:.1f} hours ⏱️\n"
                f"💡 Tips: {tips_str}"
            )

    # -----------------------------
    # SAVE TO SUPABASE USING user_id
    # -----------------------------
    if SUPABASE_ENABLED:
        try:
            # Fetch user_id once
            user_data = supabase.table("users").select("id").eq("email", session['user']).execute()
            if not user_data.data:
                print("❌ User not found in Supabase")
                return jsonify({'response': response_text})  # fallback if user not found

            user_id = str(user_data.data[0]["id"])  # convert UUID to string

            # Insert chat
            result = supabase.table("chats").insert({
                "user_id": user_id,
                "message": message,
                "response": response_text,
                "created_at": datetime.now().isoformat()
            }).execute()

            print("Supabase insert result:", result)

        except Exception as e:
            print("❌ Supabase save failed:", e)

    return jsonify({'response': response_text})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    if 'user' not in session:
        return jsonify({"success": False, "error": "User not logged in"})

    user_email = session['user']

    try:
        # Fetch user_id from users table
        user_data = supabase.table("users").select("id").eq("email", user_email).single().execute()
        user_id = user_data.data.get("id")

        print("🗑 Clearing chat history for user_id:", user_id)

        # Delete chats using user_id
        supabase.table("chats").delete().eq("user_id", user_id).execute()

        return jsonify({"success": True})

    except Exception as e:
        print("❌ Error clearing chat:", e)
        return jsonify({"success": False, "error": str(e)})






@app.route('/history')
def history():
    if 'user' not in session:
        return jsonify([])  # User not logged in

    if SUPABASE_ENABLED:
        try:
            # Get user_id from session email
            user_data = supabase.table("users").select("id").eq("email", session['user']).execute()
            if not user_data.data:
                print("❌ User not found in Supabase")
                return jsonify([])  # No user found

            user_id = str(user_data.data[0]["id"])  # convert UUID to string

            # Fetch chats by user_id
            chats = supabase.table("chats").select("*") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .execute()

            print(f"Fetched {len(chats.data)} chats for user {session['user']}")
            return jsonify(chats.data)

        except Exception as e:
            print("❌ Failed to load history:", e)
            return jsonify([])

    return jsonify([])

@app.route('/react', methods=['POST'])
def react():
    if 'user' not in session:
        return jsonify({'error': 'Please log in first.'})

    data = request.json
    chat_id = data.get('chat_id')
    thumbs_up = data.get('thumbs_up', 0)
    thumbs_down = data.get('thumbs_down', 0)

    if SUPABASE_ENABLED:
        try:
            # Get user_id from email
            user_data = supabase.table("users").select("id").eq("email", session['user']).execute()
            user_id = user_data.data[0]["id"] if user_data.data else None

            if user_id:
                supabase.table("reactions").insert({
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "thumbs_up": thumbs_up,
                    "thumbs_down": thumbs_down,
                    "created_at": datetime.now().isoformat()
                }).execute()
            else:
                print("❌ Could not find user ID for reaction insert")

        except Exception as e:
            print("❌ Failed to save reaction:", e)
            return jsonify({'success': False, 'error': str(e)})

    return jsonify({'success': True})


@app.route('/reactions', methods=['GET'])
def get_reactions():
    chat_id = request.args.get('chat_id')

    if SUPABASE_ENABLED:
        try:
            query = supabase.table("reactions").select("*")
            if chat_id:
                query = query.eq("chat_id", chat_id)
            result = query.execute()
            data = result.data

            thumbs_up_total = sum(item.get("thumbs_up", 0) for item in data)
            thumbs_down_total = sum(item.get("thumbs_down", 0) for item in data)

            return jsonify({'thumbs_up': thumbs_up_total, 'thumbs_down': thumbs_down_total})
        except Exception as e:
            print("❌ Failed to fetch reactions:", e)
            return jsonify({'thumbs_up': 0, 'thumbs_down': 0, 'error': str(e)})

    return jsonify({'thumbs_up': 0, 'thumbs_down': 0})


@app.route('/analyze_face', methods=['POST'])
def analyze_face_route():
    try:
        data = request.json
        img_data = data.get("image", "")

        if not img_data:
            return jsonify({"error": "No image received"}), 400

        # Remove "data:image/jpeg;base64,"
        img_str = img_data.split(",")[1]

        # Decode base64 → bytes
        img_bytes = base64.b64decode(img_str)

        # Convert bytes → numpy array
        np_arr = np.frombuffer(img_bytes, np.uint8)

        # Decode image
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Analyze face emotion
        emotion = analyze_face(frame)

        if emotion is None:
            return jsonify({
                "focus": "Unknown",
                "recommended_hours": 0
            })

        # Generate recommended hours based on emotion
        responses = {
    "happy": "You look happy! 😄 Great energy—use it to tackle challenging topics today!",
    "neutral": "You look calm and neutral 🙂. A steady mindset is perfect for focused studying!",
    "sad": "You seem a bit sad 😔 Take a short break, breathe, maybe listen to calm music, then start with light revision. You got this! 💪",
    "angry": "Looks like you're feeling frustrated 😤 Try relaxing for a bit, maybe take a walk. Start studying once you're calmer.",
    "fear": "You seem a bit anxious 😟 It's okay! Break tasks into smaller pieces and start slowly. You can do this! ",
    "surprise": "You look surprised 😯 Maybe something broke your focus—regain calm and continue at your pace!",
    "disgust": "You don't seem motivated 🤢 Try changing topics or your study environment for a fresh mindset."
    }
        message = responses.get(emotion, "Stay focused and do your best! 💡")
        return jsonify({
            "focus": emotion.capitalize(),
            "message": message  # <-- Updated field
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)