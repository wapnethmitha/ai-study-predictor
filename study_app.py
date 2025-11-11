import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import re
from word2number import w2n  # new import

st.title("🧠 Study Hours AI Predictor")
st.write("Ask questions like: 'What’s the score if a student studies 4.5 hours?' or 'two hours'")

# --- Train the model ---
hours = np.array([1.5, 2.0, 2.5, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.5, 9.0]).reshape(-1, 1)
scores = np.array([20, 25, 30, 40, 45, 50, 55, 65, 72, 85, 90])
X_train, X_test, y_train, y_test = train_test_split(hours, scores, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

# --- Performance ---
st.subheader("📊 Model Performance")
st.write(f"**Test MSE:** {mean_squared_error(y_test, model.predict(X_test)):.2f}")
st.write(f"**Test R²:** {r2_score(y_test, model.predict(X_test)):.2f}")

# --- Prompt input ---
st.subheader("💬 Ask a question")
user_input = st.text_input("Type your question here:")

if user_input:
    # Try to extract a number (digits first)
    match = re.search(r'(\d+(\.\d+)?)', user_input)
    
    if match:
        hours_value = float(match.group(1))
    else:
        try:
            # Try to convert number words to digits
            hours_value = w2n.word_to_num(user_input)
        except:
            hours_value = None
    
    if hours_value is not None:
        predicted_score = model.predict(np.array([[hours_value]]))[0]
        st.success(f"If a student studies {hours_value} hours, predicted score ≈ {predicted_score:.1f}")
    else:
        st.warning("Please mention the study hours in your question (digits or words).")
