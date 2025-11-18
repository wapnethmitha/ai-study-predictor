import os
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime

# Load .env variables
load_dotenv()

# Get Supabase credentials
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# Create Supabase client
supabase = create_client(url, key)

# --- INSERT A TEST USER ---
test_user = {
    "email": "test3@example.com",   # change if already exists
    "password": "test123",
    "username": "testuser3",
    "created_at": datetime.now().isoformat()
}

# Insert the test user
insert_result = supabase.table("users").insert(test_user).execute()
print("Inserted user:", insert_result.data)

# --- QUERY ALL USERS ---
all_users = supabase.table("users").select("*").execute()
print("All users:", all_users.data)

# --- INSERT A TEST CHAT ---
# Use the newly inserted user's ID
user_id = insert_result.data[0]["id"]

test_chat = {
    "user_id": user_id,  # reference the inserted user
    "message": "Hello, AI!",
    "response": "Hi there!",
    "created_at": datetime.now().isoformat()
}

insert_chat = supabase.table("chats").insert(test_chat).execute()
print("Inserted chat:", insert_chat.data)

# --- QUERY ALL CHATS ---
all_chats = supabase.table("chats").select("*").execute()
print("All chats:", all_chats.data)
