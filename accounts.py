import json
import os

USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def create_account(username, password):
    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = {"password": password}
    save_users(users)
    return True, "Account created!"

def login(username, password):
    users = load_users()
    if username in users and users[username]["password"] == password:
        return True, "Login successful!"
    return False, "Invalid credentials."

def get_user_data_file(username):
    return f"input_{username}.txt"
