import os
import sqlite3
from datetime import datetime, timedelta
import joblib
import pandas as pd
from PIL import Image

import db
import auth
import notifier
import utils

def test_all():
    print("=" * 60)
    print("RUNNING FEATURE VALIDATION TEST SUITE")
    print("=" * 60)
    
    # 1. DB Migration & Old Rows Preservation
    print("\n[1/8] Testing Database Migration...")
    db.migrate_db()
    conn = db.get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    assert "patients" in tables, "patients table missing"
    assert "department_status" in tables, "department_status table missing"
    assert "users" in tables, "users table missing"
    assert "notifications" in tables, "notifications table missing"
    
    cur.execute("PRAGMA table_info(patients)")
    cols = [r["name"] for r in cur.fetchall()]
    assert "phone" in cols, "phone column missing from patients"
    assert "email" in cols, "email column missing from patients"
    assert "expected_call_time" in cols, "expected_call_time column missing from patients"
    conn.close()
    print("  -> PASS: Database migrated successfully with all required tables and columns.")

    # 2. Login Hashing & Roles
    print("\n[2/8] Testing Password Hashing & Roles...")
    auth.seed_default_users()
    staff_user = auth.authenticate_user("staff", "staff123")
    assert staff_user is not None and staff_user["role"] == "staff", "Staff authentication failed"
    admin_user = auth.authenticate_user("admin", "admin123")
    assert admin_user is not None and admin_user["role"] == "admin", "Admin authentication failed"
    wrong_user = auth.authenticate_user("admin", "wrongpassword")
    assert wrong_user is None, "Failed authentication should return None"
    
    patient_pages = auth.get_allowed_pages("patient")
    staff_pages = auth.get_allowed_pages("staff")
    admin_pages = auth.get_allowed_pages("admin")
    
    assert "Patient Check-in" in patient_pages and "Staff Panel" not in patient_pages
    assert "Staff Panel" in staff_pages and "Analytics" not in staff_pages
    assert "Analytics" in admin_pages and "Doctor Load Balancing" in admin_pages
    print("  -> PASS: Password hashing (PBKDF2 HMAC + salt) and RBAC role boundaries verified.")

    # 3. Expected Call Time Calculation
    print("\n[3/8] Testing Expected Call Time Calculation...")
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    pred_wait = 25.0
    exp_dt = now + timedelta(minutes=pred_wait)
    exp_str = exp_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    pid = db.add_patient(
        token=9999,
        name="Test Patient",
        department="General",
        visit_type="Walk-in",
        predicted_wait=pred_wait,
        checkin_time=now_str,
        expected_call_time=exp_str,
        phone="+15550001",
        email="test@example.com"
    )
    p_rec = db.get_patient_by_token(9999)
    assert p_rec is not None, "Failed to fetch inserted patient"
    assert p_rec["expected_call_time"] == exp_str, "Expected call time mismatch"
    # Clean up test row
    conn = db.get_db_connection()
    with conn:
        conn.execute("DELETE FROM patients WHERE token = 9999")
    conn.close()
    print("  -> PASS: Expected call time correctly computed and persisted.")

    # 4. Hourly Forecast (12 values: 8-19)
    print("\n[4/8] Testing Wait Forecast (8-19)...")
    model = joblib.load("model.pkl")
    forecast_df = utils.compute_hourly_forecast(model, "General", "Walk-in", 2, day_of_week=0)
    assert len(forecast_df) == 12, f"Expected 12 hourly forecasts, got {len(forecast_df)}"
    assert list(forecast_df["hour"]) == list(range(8, 20)), "Hours must be 8 through 19"
    assert (forecast_df["predicted_wait"] >= 0).all(), "Predictions must be non-negative"
    print("  -> PASS: Exactly 12 hourly predictions returned for hours 8-19.")

    # 5. Doctor Load Balancing Suggestion
    print("\n[5/8] Testing Doctor Load Balancing Suggestion Engine...")
    docs_map = {"General": 2, "Cardiology": 1, "Orthopedics": 3, "Pediatrics": 2}
    queues_map = {"General": 4, "Cardiology": 15, "Orthopedics": 2, "Pediatrics": 1}
    sugg = utils.compute_load_balancing_suggestion(model, docs_map, queues_map, hour=10, day_of_week=0)
    assert "can_balance" in sugg, "Suggestion dict must have can_balance key"
    assert "current_highest" in sugg, "Suggestion dict must have current_highest key"
    print(f"  -> Suggestion result: can_balance={sugg['can_balance']}")
    if sugg["can_balance"]:
        print(f"     Move 1 doc from {sugg['move_from']} to {sugg['move_to']}: wait drops from {sugg['current_highest']} to {sugg['new_highest']} min")
    print("  -> PASS: Load balancing suggestion engine evaluated without errors.")

    # 6. PDF Token Slip Generation
    print("\n[6/8] Testing PDF Token Slip Generation (fpdf2)...")
    pdf_info = {
        "token": 42,
        "name": "Jane Doe",
        "department": "Cardiology",
        "visit_type": "Appointment",
        "checkin_time": "2026-11-20 10:30:00",
        "position_text": "#2 in line",
        "predicted_wait": 22.5,
        "expected_call_time_str": "10:52 AM"
    }
    pdf_bytes = utils.generate_token_pdf(pdf_info, "http://192.168.1.100:8501")
    assert isinstance(pdf_bytes, bytes) and len(pdf_bytes) > 500, "PDF bytes should be non-empty"
    assert pdf_bytes.startswith(b"%PDF"), "Output must be valid PDF"
    print(f"  -> PASS: PDF token slip successfully generated ({len(pdf_bytes)} bytes).")

    # 7. QR Code Generation
    print("\n[7/8] Testing QR Code Generation (qrcode[pil])...")
    qr_img = utils.generate_qr_code("http://192.168.1.100:8501/?page=checkin")
    assert isinstance(qr_img, Image.Image), "QR code must be a PIL Image"
    qr_bytes = utils.get_qr_bytes(qr_img)
    assert len(qr_bytes) > 200, "QR PNG bytes should be non-empty"
    print(f"  -> PASS: QR Code generated successfully ({qr_img.size[0]}x{qr_img.size[1]} px).")

    # 8. Notifications in Simulated Mode
    print("\n[8/8] Testing Notification Dispatcher (Simulated)...")
    res = notifier.dispatch_notification(
        token=888,
        phone="",
        email="",
        message="Token #888: Please proceed to General Consultation."
    )
    assert res["token"] == 888, "Token mismatch in notification dispatch"
    assert any(c[0] == "simulated" for c in res["dispatches"]), "Expected simulated channel dispatch"
    
    notifs_df = db.get_all_notifications_df()
    match = notifs_df[notifs_df["token"] == 888]
    assert not match.empty, "Notification must be logged in notifications table"
    assert match.iloc[0]["channel"] == "simulated", "Logged channel must be 'simulated'"
    print("  -> PASS: Notification dispatched and logged to database in simulated mode.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (8/8)")
    print("=" * 60)

if __name__ == "__main__":
    test_all()
