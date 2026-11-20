import hashlib
import hmac
import os
import streamlit as st
from db import get_db_connection

def hash_password(password: str, salt: bytes = None) -> str:
    """Hash a password using PBKDF2 HMAC SHA-256 with a 16-byte random salt."""
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return f"{salt.hex()}:{key.hex()}"

def verify_password(stored_hash: str, provided_password: str) -> bool:
    """Verify password against stored salt:hash string."""
    try:
        salt_hex, key_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        key = hashlib.pbkdf2_hmac("sha256", provided_password.encode("utf-8"), salt, 100000)
        return hmac.compare_digest(key.hex(), key_hex)
    except Exception:
        return False

def seed_default_users():
    """Ensure staff and admin default accounts exist."""
    conn = get_db_connection()
    with conn:
        cur = conn.cursor()
        cur.execute("SELECT username FROM users WHERE username = 'staff'")
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("staff", hash_password("staff123"), "staff")
            )
        cur.execute("SELECT username FROM users WHERE username = 'admin'")
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("admin", hash_password("admin123"), "admin")
            )
    conn.close()

def authenticate_user(username: str, password: str):
    """Check credentials and return user info dict or None."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT username, password_hash, role FROM users WHERE username = ?", (username.strip(),))
    row = cur.fetchone()
    conn.close()
    
    if row and verify_password(row["password_hash"], password):
        return {"username": row["username"], "role": row["role"]}
    return None

PAGE_PERMISSIONS = {
    "Patient Check-in": ["patient", "staff", "admin"],
    "Track My Token": ["patient", "staff", "admin"],
    "Best Time to Visit": ["patient", "staff", "admin"],
    "QR Check-in": ["patient", "staff", "admin"],
    "Staff Panel": ["staff", "admin"],
    "Doctor Load Balancing": ["staff", "admin"],
    "Notifications Log": ["staff", "admin"],
    "Wait Forecast": ["staff", "admin"],
    "Analytics": ["admin"]
}

def get_allowed_pages(role: str):
    """Return list of page names accessible by the given role."""
    return [page for page, roles in PAGE_PERMISSIONS.items() if role in roles]

def can_access_page(role: str, page_name: str) -> bool:
    """Check if the given role is allowed to view a specific page."""
    allowed = PAGE_PERMISSIONS.get(page_name, [])
    return role in allowed

def get_current_user():
    """Retrieve current logged in user from session state or default to patient role."""
    if "user" not in st.session_state or not st.session_state["user"]:
        return {"username": "Guest", "role": "patient"}
    return st.session_state["user"]

def login_user(user_dict: dict):
    st.session_state["user"] = user_dict

def logout_user():
    st.session_state["user"] = None
    if "page" in st.session_state:
        st.session_state["page"] = "Patient Check-in"
