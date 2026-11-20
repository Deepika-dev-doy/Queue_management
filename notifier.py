import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from db import log_notification

def send_email_notification(to_email: str, subject: str, message: str) -> bool:
    """Send an email using SMTP if environment variables are set."""
    host = os.environ.get("SMTP_HOST")
    port = os.environ.get("SMTP_PORT", "587")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    
    if not (host and user and password and to_email):
        return False
        
    try:
        msg = MIMEMultipart()
        msg["From"] = user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))
        
        server = smtplib.SMTP(host, int(port), timeout=10)
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"SMTP error: {e}")
        return False

def send_sms_notification(to_phone: str, message: str) -> bool:
    """Send an SMS using Twilio if credentials are set (lazy import)."""
    sid = os.environ.get("TWILIO_SID")
    token = os.environ.get("TWILIO_TOKEN")
    from_num = os.environ.get("TWILIO_FROM")
    
    if not (sid and token and from_num and to_phone):
        return False
        
    try:
        from twilio.rest import Client
        client = Client(sid, token)
        client.messages.create(body=message, from_=from_num, to=to_phone)
        return True
    except Exception as e:
        print(f"Twilio SMS error: {e}")
        return False

def dispatch_notification(token: int, phone: str, email: str, message: str) -> dict:
    """
    Sends notification via Email or SMS if configured, otherwise falls back to SIMULATE.
    Always logs the outcome to the notifications database table.
    """
    dispatched_channels = []
    
    # 1. Try Email if email provided and env vars present
    if email and os.environ.get("SMTP_HOST"):
        success = send_email_notification(email, f"Hospital Queue Update - Token #{token}", message)
        status = "sent" if success else "failed"
        log_notification(token, "email", email, message, status)
        dispatched_channels.append(("email", email, status))
        
    # 2. Try SMS if phone provided and twilio env vars present
    if phone and os.environ.get("TWILIO_SID"):
        success = send_sms_notification(phone, message)
        status = "sent" if success else "failed"
        log_notification(token, "sms", phone, message, status)
        dispatched_channels.append(("sms", phone, status))

    # 3. If no live channel was used, simulate
    if not dispatched_channels:
        recipient = phone or email or f"Patient (Token #{token})"
        channel = "simulated"
        status = "simulated"
        log_notification(token, channel, recipient, message, status)
        dispatched_channels.append((channel, recipient, status))
        
    return {
        "token": token,
        "message": message,
        "dispatches": dispatched_channels
    }

def notify_called_patient(patient: dict) -> dict:
    """Notify the patient who was just called."""
    token = patient.get("token")
    name = patient.get("name", "Patient")
    dept = patient.get("department", "Department")
    phone = patient.get("phone") or ""
    email = patient.get("email") or ""
    
    message = f"Token #{token} ({name}): Please proceed to {dept} for your consultation."
    return dispatch_notification(token, phone, email, message)

def notify_next_in_line(patient: dict) -> dict:
    """Notify the new first patient in line."""
    token = patient.get("token")
    name = patient.get("name", "Patient")
    dept = patient.get("department", "Department")
    phone = patient.get("phone") or ""
    email = patient.get("email") or ""
    
    message = f"Token #{token} ({name}): You are next in line for {dept}. Please stand by."
    return dispatch_notification(token, phone, email, message)
