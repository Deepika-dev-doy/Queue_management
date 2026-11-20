import io
import os
import socket
from datetime import datetime
import numpy as np
import pandas as pd
import qrcode
from PIL import Image
from fpdf import FPDF

# --- Network & URL Utilities ---

def get_local_ip() -> str:
    """Auto-detect the host machine's local network IP using a standard socket trick."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        # Does not actually establish internet connection; determines outward interface
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def get_default_app_url() -> str:
    """Return default application base URL with detected IP and port 8501."""
    return f"http://{get_local_ip()}:8501"


# --- QR Code Utilities ---

def generate_qr_code(url: str) -> Image.Image:
    """Generate a high-contrast QR code image for a given URL."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=2
    )
    qr.add_data(url)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")

def get_qr_bytes(img: Image.Image) -> bytes:
    """Convert a PIL Image to PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- PDF Token Slip Utilities (fpdf2) ---

class TokenPDF(FPDF):
    def header(self):
        pass
    def footer(self):
        pass

def generate_token_pdf(token_data: dict, app_url: str = "") -> bytes:
    """
    Generate an in-memory hospital token slip PDF (A6 receipt format)
    containing patient info, predicted wait, expected call time, and a dynamic QR code.
    """
    pdf = TokenPDF(format=(105, 148))  # 105 x 148 mm receipt format
    pdf.set_auto_page_break(auto=False, margin=5)
    pdf.add_page()
    
    # 1. Hospital Header
    pdf.set_fill_color(24, 76, 120)  # Medical Navy
    pdf.rect(0, 0, 105, 18, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_xy(0, 4)
    pdf.cell(105, 5, "CITY GENERAL HOSPITAL", align="C")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(0, 10)
    pdf.cell(105, 4, "Intelligent Outpatient Queue Management", align="C")
    
    # 2. Token Number Highlight Box
    pdf.set_text_color(30, 30, 30)
    pdf.set_xy(10, 22)
    pdf.set_font("Helvetica", "B", 20)
    token_str = f"TOKEN #{token_data.get('token', 'N/A')}"
    pdf.cell(85, 10, token_str, border=1, align="C")
    
    # 3. Patient & Queue Details
    pdf.set_font("Helvetica", "", 9)
    y = 35
    line_h = 5.2
    
    fields = [
        ("Patient Name:", str(token_data.get("name", "N/A"))),
        ("Department:", str(token_data.get("department", "N/A"))),
        ("Visit Type:", str(token_data.get("visit_type", "N/A"))),
        ("Check-in Time:", str(token_data.get("checkin_time", "N/A"))),
        ("Queue Position:", str(token_data.get("position_text", "N/A"))),
        ("Predicted Wait:", f"{float(token_data.get('predicted_wait', 0.0)):.1f} mins"),
        ("Expected Call Time:", str(token_data.get("expected_call_time_str", "N/A"))),
    ]
    
    for label, val in fields:
        pdf.set_xy(10, y)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(36, line_h, label)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(49, line_h, val)
        y += line_h
        
    # 4. QR Code linking to Track My Token
    base_url = app_url or get_default_app_url()
    token_num = token_data.get("token", "")
    track_url = f"{base_url}/?page=track&token={token_num}"
    
    qr_img = generate_qr_code(track_url)
    qr_buf = io.BytesIO()
    qr_img.save(qr_buf, format="PNG")
    qr_buf.seek(0)
    
    qr_size = 32
    qr_x = (105 - qr_size) / 2
    pdf.image(qr_buf, x=qr_x, y=y + 2, w=qr_size, h=qr_size)
    
    # 5. QR Instructions & Footer Note
    pdf.set_xy(5, y + qr_size + 3)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(95, 3.5, "Scan with your phone to track queue status in real time.\nPlease keep this slip until your number is announced.", align="C")
    
    return bytes(pdf.output())


# --- Hourly Forecast & Best Time to Visit (Features 7 & 8) ---

def load_hourly_average_queues(data_path="data.csv") -> dict:
    """Calculate the average historical queue length for hours 8-19 from data.csv."""
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        avg_q = df.groupby("hour")["queue_length"].mean().round().astype(int).to_dict()
        return avg_q
    # Fallback to Poisson lambda values
    return {h: 9 if h in {9, 10, 11, 17, 18} else 6 for h in range(8, 20)}

def compute_hourly_forecast(model, department: str, visit_type: str, doctors: int, 
                            day_of_week: int = None, data_path: str = "data.csv") -> pd.DataFrame:
    """
    Predict wait time for each hour from 8 to 19 using the trained ML model.
    Uses historical average queue length per hour from data.csv.
    """
    if day_of_week is None:
        day_of_week = datetime.now().weekday()
        
    hourly_queues = load_hourly_average_queues(data_path)
    current_hour = datetime.now().hour
    
    rows = []
    for h in range(8, 20):
        q_len = hourly_queues.get(h, 6)
        input_df = pd.DataFrame([{
            "hour": h,
            "day_of_week": day_of_week,
            "department": department,
            "visit_type": visit_type,
            "queue_length": q_len,
            "doctors_available": doctors
        }])
        
        raw_pred = model.predict(input_df)[0]
        wait_pred = round(max(0.0, float(raw_pred)), 1)
        
        # Format 12-hour label
        period = "AM" if h < 12 else "PM"
        display_hour = h if h <= 12 else h - 12
        time_label = f"{display_hour}:00 {period}"
        
        rows.append({
            "hour": h,
            "time_label": time_label,
            "predicted_wait": wait_pred,
            "queue_length": q_len,
            "doctors": doctors,
            "is_current_hour": (h == current_hour)
        })
        
    return pd.DataFrame(rows)

def compute_best_times_to_visit(model, department: str, visit_type: str, doctors: int,
                                day_of_week: int = None, data_path: str = "data.csv") -> dict:
    """
    Identify the 3 hours with the lowest predicted wait time for today.
    """
    df = compute_hourly_forecast(model, department, visit_type, doctors, day_of_week, data_path)
    sorted_df = df.sort_values(by="predicted_wait", ascending=True)
    top_3 = sorted_df.head(3).to_dict(orient="records")
    
    return {
        "top_3": top_3,
        "full_forecast": df
    }


# --- Doctor Load Balancing Suggestion Engine (Feature 12) ---

def compute_load_balancing_suggestion(model, dept_doctors: dict, dept_queues: dict, 
                                      hour: int = None, day_of_week: int = None) -> dict:
    """
    Evaluates current queue conditions across the 4 departments.
    Tests moving 1 doctor from each department with 2+ doctors to each other department.
    Re-predicts waits and finds the single move that minimizes the highest predicted wait.
    """
    now = datetime.now()
    if hour is None:
        hour = max(8, min(19, now.hour))
    if day_of_week is None:
        day_of_week = now.weekday()
        
    departments = ["General", "Cardiology", "Orthopedics", "Pediatrics"]
    
    # 1. Calculate current predicted wait for a typical Walk-in patient per department
    current_waits = {}
    for dept in departments:
        q_len = dept_queues.get(dept, 0)
        docs = dept_doctors.get(dept, 2)
        inp = pd.DataFrame([{
            "hour": hour,
            "day_of_week": day_of_week,
            "department": dept,
            "visit_type": "Walk-in",
            "queue_length": q_len,
            "doctors_available": docs
        }])
        pred = round(max(0.0, float(model.predict(inp)[0])), 1)
        current_waits[dept] = pred
        
    current_highest = max(current_waits.values())
    
    # 2. Simulate all single doctor transfers
    best_move = None
    lowest_new_highest = current_highest
    best_simulated_waits = None
    
    for dept_from in departments:
        if dept_doctors.get(dept_from, 2) >= 2:  # must have at least 2 doctors to spare 1
            for dept_to in departments:
                if dept_from != dept_to and dept_doctors.get(dept_to, 2) < 5:
                    sim_docs = dict(dept_doctors)
                    sim_docs[dept_from] -= 1
                    sim_docs[dept_to] += 1
                    
                    # Recompute waits for all 4 departments under simulated allocation
                    sim_waits = {}
                    for dept in departments:
                        inp = pd.DataFrame([{
                            "hour": hour,
                            "day_of_week": day_of_week,
                            "department": dept,
                            "visit_type": "Walk-in",
                            "queue_length": dept_queues.get(dept, 0),
                            "doctors_available": sim_docs[dept]
                        }])
                        p = round(max(0.0, float(model.predict(inp)[0])), 1)
                        sim_waits[dept] = p
                        
                    sim_highest = max(sim_waits.values())
                    
                    # If this move lowers the maximum bottleneck wait significantly
                    if sim_highest < lowest_new_highest - 0.5:
                        lowest_new_highest = sim_highest
                        best_move = (dept_from, dept_to)
                        best_simulated_waits = sim_waits

    if best_move:
        return {
            "can_balance": True,
            "move_from": best_move[0],
            "move_to": best_move[1],
            "current_highest": current_highest,
            "new_highest": lowest_new_highest,
            "current_waits": current_waits,
            "simulated_waits": best_simulated_waits
        }
    else:
        return {
            "can_balance": False,
            "current_highest": current_highest,
            "current_waits": current_waits
        }
