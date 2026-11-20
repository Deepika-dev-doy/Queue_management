import os
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

DB_FILE = "queue.db"

def get_db_connection():
    """Create a thread-safe connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def migrate_db():
    """
    Safely migrates the SQLite database without losing existing patient rows.
    Creates department_status, users, and notifications tables if missing.
    Adds phone, email, and expected_call_time columns to patients table if missing.
    Seeds department_status and default user credentials.
    """
    conn = get_db_connection()
    with conn:
        cur = conn.cursor()
        
        # 1. Ensure patients table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token INTEGER NOT NULL,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                visit_type TEXT NOT NULL,
                predicted_wait REAL NOT NULL,
                checkin_time TEXT NOT NULL,
                called_time TEXT,
                actual_wait_minutes REAL,
                status TEXT NOT NULL
            )
        """)
        
        # Check existing columns in patients
        cur.execute("PRAGMA table_info(patients)")
        columns = [row["name"] for row in cur.fetchall()]
        
        if "phone" not in columns:
            cur.execute("ALTER TABLE patients ADD COLUMN phone TEXT")
        if "email" not in columns:
            cur.execute("ALTER TABLE patients ADD COLUMN email TEXT")
        if "expected_call_time" not in columns:
            cur.execute("ALTER TABLE patients ADD COLUMN expected_call_time TEXT")
            
        # Backfill expected_call_time for existing rows if null
        cur.execute("SELECT id, checkin_time, predicted_wait FROM patients WHERE expected_call_time IS NULL")
        for row in cur.fetchall():
            try:
                checkin_dt = datetime.strptime(row["checkin_time"], "%Y-%m-%d %H:%M:%S")
                pred_wait = row["predicted_wait"] or 0.0
                exp_dt = checkin_dt + timedelta(minutes=float(pred_wait))
                exp_str = exp_dt.strftime("%Y-%m-%d %H:%M:%S")
                cur.execute("UPDATE patients SET expected_call_time = ? WHERE id = ?", (exp_str, row["id"]))
            except Exception:
                pass

        # 2. department_status table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS department_status (
                department TEXT PRIMARY KEY,
                doctors_available INTEGER NOT NULL DEFAULT 2
            )
        """)
        
        # Seed 4 departments if empty
        cur.execute("SELECT COUNT(*) FROM department_status")
        if cur.fetchone()[0] == 0:
            departments = ["General", "Cardiology", "Orthopedics", "Pediatrics"]
            for dept in departments:
                cur.execute(
                    "INSERT INTO department_status (department, doctors_available) VALUES (?, ?)",
                    (dept, 2)
                )

        # 3. users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL
            )
        """)
        
        # 4. notifications table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token INTEGER,
                channel TEXT NOT NULL,
                recipient TEXT,
                message TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
    conn.close()

# Database Helper Functions

def get_department_doctors(department: str) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT doctors_available FROM department_status WHERE department = ?", (department,))
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else 2

def get_all_department_doctors() -> dict:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT department, doctors_available FROM department_status")
    rows = cur.fetchall()
    conn.close()
    return {row["department"]: row["doctors_available"] for row in rows}

def update_department_doctors(department: str, doctors: int):
    conn = get_db_connection()
    with conn:
        conn.execute(
            "UPDATE department_status SET doctors_available = ? WHERE department = ?",
            (doctors, department)
        )
    conn.close()

def add_patient(token: int, name: str, department: str, visit_type: str,
                predicted_wait: float, checkin_time: str, expected_call_time: str,
                phone: str = "", email: str = "") -> int:
    conn = get_db_connection()
    with conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO patients (token, name, department, visit_type, predicted_wait, 
                                checkin_time, expected_call_time, phone, email, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'waiting')
        """, (token, name, department, visit_type, predicted_wait, 
              checkin_time, expected_call_time, phone or None, email or None))
        patient_id = cur.lastrowid
    conn.close()
    return patient_id

def get_max_token() -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COALESCE(MAX(token), 0) FROM patients")
    max_t = cur.fetchone()[0]
    conn.close()
    return int(max_t)

def get_waiting_queue_count(department: str) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM patients WHERE department = ? AND status = 'waiting'", (department,))
    count = cur.fetchone()[0]
    conn.close()
    return int(count)

def get_patient_by_token(token: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM patients WHERE token = ?", (token,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def get_waiting_patients_df(department: str = "All") -> pd.DataFrame:
    conn = get_db_connection()
    query = """
        SELECT token, name, department, visit_type, predicted_wait, expected_call_time, checkin_time, phone, email
        FROM patients 
        WHERE status = 'waiting'
    """
    params = ()
    if department != "All":
        query += " AND department = ?"
        params = (department,)
    query += " ORDER BY CASE WHEN visit_type = 'Emergency' THEN 0 ELSE 1 END, token ASC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_served_patients_df(department: str = "All") -> pd.DataFrame:
    conn = get_db_connection()
    query = """
        SELECT token, name, department, visit_type, predicted_wait, expected_call_time, checkin_time, called_time, actual_wait_minutes
        FROM patients 
        WHERE status = 'served'
    """
    params = ()
    if department != "All":
        query += " AND department = ?"
        params = (department,)
    query += " ORDER BY id DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_next_patient_to_call(department: str = "All"):
    conn = get_db_connection()
    cur = conn.cursor()
    if department == "All":
        cur.execute("""
            SELECT * FROM patients
            WHERE status = 'waiting'
            ORDER BY CASE WHEN visit_type = 'Emergency' THEN 0 ELSE 1 END, token ASC
            LIMIT 1
        """)
    else:
        cur.execute("""
            SELECT * FROM patients
            WHERE status = 'waiting' AND department = ?
            ORDER BY CASE WHEN visit_type = 'Emergency' THEN 0 ELSE 1 END, token ASC
            LIMIT 1
        """, (department,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def get_first_waiting_in_line(department: str, exclude_id: int = None):
    conn = get_db_connection()
    cur = conn.cursor()
    query = """
        SELECT * FROM patients
        WHERE status = 'waiting' AND department = ?
    """
    params = [department]
    if exclude_id is not None:
        query += " AND id != ?"
        params.append(exclude_id)
    query += " ORDER BY CASE WHEN visit_type = 'Emergency' THEN 0 ELSE 1 END, token ASC LIMIT 1"
    cur.execute(query, tuple(params))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def mark_patient_served(patient_id: int, called_time: str, actual_wait: float):
    conn = get_db_connection()
    with conn:
        conn.execute("""
            UPDATE patients
            SET status = 'served', called_time = ?, actual_wait_minutes = ?
            WHERE id = ?
        """, (called_time, actual_wait, patient_id))
    conn.close()

def update_patient_status(patient_id: int, status: str):
    conn = get_db_connection()
    with conn:
        conn.execute("UPDATE patients SET status = ? WHERE id = ?", (status, patient_id))
    conn.close()

def cancel_patient_by_token(token: int) -> bool:
    conn = get_db_connection()
    with conn:
        cur = conn.cursor()
        cur.execute("UPDATE patients SET status = 'cancelled' WHERE token = ? AND status = 'waiting'", (token,))
        rows = cur.rowcount
    conn.close()
    return rows > 0

def log_notification(token: int, channel: str, recipient: str, message: str, status: str):
    conn = get_db_connection()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        conn.execute("""
            INSERT INTO notifications (token, channel, recipient, message, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (token, channel, recipient, message, status, now_str))
    conn.close()

def get_all_notifications_df() -> pd.DataFrame:
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM notifications ORDER BY id DESC", conn)
    conn.close()
    return df
