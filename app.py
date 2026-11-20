import os
import json
import sqlite3
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import db
import auth
import notifier
import utils

# --- Initialize Application State and Database ---
db.migrate_db()
auth.seed_default_users()

MODEL_FILE = "model.pkl"
DATA_FILE = "data.csv"
METRICS_FILE = "metrics.json"

@st.cache_resource
def load_prediction_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

# Streamlit Page Configuration
st.set_page_config(
    page_title="Hospital Queue Management",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Role-Based Navigation & Query Params Handling ---
current_user = auth.get_current_user()
allowed_pages = auth.get_allowed_pages(current_user["role"])

# Map query params to page names
PARAM_PAGE_MAP = {
    "checkin": "Patient Check-in",
    "track": "Track My Token",
    "best_time": "Best Time to Visit",
    "qr": "QR Check-in",
    "staff": "Staff Panel",
    "balancing": "Doctor Load Balancing",
    "forecast": "Wait Forecast",
    "notifications": "Notifications Log",
    "analytics": "Analytics"
}

# Resolve starting page from query parameters or session state
query_page_param = st.query_params.get("page", None)
resolved_page = None
if query_page_param and query_page_param in PARAM_PAGE_MAP:
    target_page = PARAM_PAGE_MAP[query_page_param]
    if target_page in allowed_pages:
        resolved_page = target_page

if not resolved_page:
    if "selected_page" in st.session_state and st.session_state["selected_page"] in allowed_pages:
        resolved_page = st.session_state["selected_page"]
    else:
        resolved_page = allowed_pages[0]

# --- Sidebar UI: Authentication & Navigation ---
st.sidebar.title("🏥 Hospital Queue System")

# Display current role badge
role_colors = {"patient": "#4CAF50", "staff": "#2196F3", "admin": "#9C27B0"}
role_name = current_user["role"].upper()
st.sidebar.markdown(
    f"<div style='padding: 6px 12px; border-radius: 6px; background-color: #f0f2f6; border-left: 5px solid {role_colors.get(current_user['role'], '#9E9E9E')}; margin-bottom: 15px;'>"
    f"<strong>User:</strong> {current_user['username']} &nbsp;|&nbsp; <strong>Role:</strong> {role_name}</div>",
    unsafe_allow_html=True
)

# Authentication Box
if current_user["role"] == "patient":
    with st.sidebar.expander("🔐 Staff / Admin Login", expanded=False):
        with st.form("login_form", clear_on_submit=False):
            u_name = st.text_input("Username")
            u_pass = st.text_input("Password", type="password")
            btn_login = st.form_submit_button("Log In", use_container_width=True)
            if btn_login:
                user_record = auth.authenticate_user(u_name, u_pass)
                if user_record:
                    auth.login_user(user_record)
                    st.success(f"Welcome, {user_record['username']} ({user_record['role']})!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        st.caption("Default logins: `staff` / `staff123` | `admin` / `admin123`")
else:
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        auth.logout_user()
        st.query_params.clear()
        st.rerun()

st.sidebar.markdown("---")

# Navigation radio restricted strictly to allowed pages
selected_page = st.sidebar.radio(
    "Navigation",
    allowed_pages,
    index=allowed_pages.index(resolved_page) if resolved_page in allowed_pages else 0
)
st.session_state["selected_page"] = selected_page

# App URL Configuration for QR codes
st.sidebar.markdown("---")
if "app_url" not in st.session_state:
    st.session_state["app_url"] = utils.get_default_app_url()
sidebar_app_url = st.sidebar.text_input("Application URL (for QR codes):", value=st.session_state["app_url"])
st.session_state["app_url"] = sidebar_app_url

# Security Guard: Stop if unauthorized direct access attempted
if not auth.can_access_page(current_user["role"], selected_page):
    st.error(f"⛔ Access Denied: Role '{current_user['role']}' cannot access '{selected_page}'.")
    st.stop()

# Load Predictive ML Model
model = load_prediction_model()


# ===========================================================================
# PAGE 1: Patient Check-in
# ===========================================================================
if selected_page == "Patient Check-in":
    st.title("🩺 Patient Check-in & Waiting-Time Prediction")
    st.write("Register incoming patients, compute real-time queue conditions, and generate instant ML wait forecasts.")
    
    if model is None:
        st.warning("⚠️ Model not loaded. Please make sure `model.pkl` exists.")

    col_form, col_info = st.columns([1.2, 1])

    with col_form:
        st.subheader("New Patient Registration")
        
        # Best time hint calculation for selected department
        preview_dept = st.selectbox(
            "Department",
            ["General", "Cardiology", "Orthopedics", "Pediatrics"],
            key="checkin_dept_select"
        )
        
        if model is not None:
            docs_count = db.get_department_doctors(preview_dept)
            best_info = utils.compute_best_times_to_visit(model, preview_dept, "Walk-in", docs_count)
            if best_info["top_3"]:
                best_hr = best_info["top_3"][0]
                st.info(f"💡 **Best time today for {preview_dept}:** {best_hr['time_label']} (typical wait: ~{best_hr['predicted_wait']:.1f} mins with {docs_count} doctor(s)).")

        with st.form("checkin_form", clear_on_submit=True):
            name = st.text_input("Patient Full Name *", placeholder="e.g. John Doe")
            visit_type = st.selectbox("Visit Type", ["Walk-in", "Appointment", "Emergency"])
            
            c_ph, c_em = st.columns(2)
            with c_ph:
                phone = st.text_input("Phone Number (Optional)", placeholder="+1-555-0199")
            with c_em:
                email = st.text_input("Email Address (Optional)", placeholder="patient@example.com")
                
            submitted = st.form_submit_button("Check In Patient", use_container_width=True, type="primary")

        if submitted:
            if not name.strip():
                st.error("Patient name cannot be empty.")
            elif model is None:
                st.error("Model unavailable. Please train `model.pkl` first.")
            else:
                now = datetime.now()
                clamped_hour = max(8, min(19, now.hour))
                weekday = now.weekday()
                
                # Fetch doctors from department_status table
                dept_docs = db.get_department_doctors(preview_dept)
                current_queue = db.get_waiting_queue_count(preview_dept)
                new_token = db.get_max_token() + 1
                
                # Predict wait using trained model
                input_df = pd.DataFrame([{
                    "hour": clamped_hour,
                    "day_of_week": weekday,
                    "department": preview_dept,
                    "visit_type": visit_type,
                    "queue_length": current_queue,
                    "doctors_available": dept_docs
                }])
                
                raw_wait = model.predict(input_df)[0]
                pred_wait = round(max(0.0, float(raw_wait)), 1)
                
                checkin_time_dt = now
                checkin_time_str = checkin_time_dt.strftime("%Y-%m-%d %H:%M:%S")
                expected_call_dt = checkin_time_dt + timedelta(minutes=pred_wait)
                expected_call_time_str = expected_call_dt.strftime("%I:%M %p")
                
                # Insert into database
                db.add_patient(
                    token=new_token,
                    name=name.strip(),
                    department=preview_dept,
                    visit_type=visit_type,
                    predicted_wait=pred_wait,
                    checkin_time=checkin_time_str,
                    expected_call_time=expected_call_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    phone=phone.strip(),
                    email=email.strip()
                )
                
                st.session_state["last_checked_in"] = {
                    "token": new_token,
                    "name": name.strip(),
                    "department": preview_dept,
                    "visit_type": visit_type,
                    "checkin_time": checkin_time_str,
                    "patients_ahead": current_queue,
                    "position_text": f"#{current_queue + 1} in line",
                    "predicted_wait": pred_wait,
                    "expected_call_time_str": expected_call_time_str,
                    "phone": phone.strip(),
                    "email": email.strip()
                }
                st.success(f"✅ Token #{new_token} generated for {name.strip()}!")

    with col_info:
        st.subheader("Issued Token & Details")
        if "last_checked_in" in st.session_state:
            token_info = st.session_state["last_checked_in"]
            
            st.markdown(f"""
            <div style="background-color: #f0f7ff; padding: 20px; border-radius: 10px; border-left: 6px solid #1E88E5; margin-bottom: 15px;">
                <h1 style="margin: 0; color: #1565C0;">TOKEN #{token_info['token']}</h1>
                <h3 style="margin: 5px 0; color: #333;">{token_info['name']}</h3>
                <p style="margin: 0; font-size: 15px; color: #555;"><strong>Department:</strong> {token_info['department']} &nbsp;|&nbsp; <strong>Type:</strong> {token_info['visit_type']}</p>
                <p style="margin: 4px 0 0 0; font-size: 14px; color: #666;">Check-in: {token_info['checkin_time']}</p>
                <h4 style="margin: 12px 0 0 0; color: #2E7D32;">🕒 Expected to be seen at {token_info['expected_call_time_str']}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            m1, m2 = st.columns(2)
            m1.metric("Patients Ahead", f"{token_info['patients_ahead']}")
            m2.metric("Predicted Wait", f"{token_info['predicted_wait']:.1f} mins")
            
            # PDF Slip Download
            pdf_bytes = utils.generate_token_pdf(token_info, st.session_state.get("app_url", ""))
            st.download_button(
                label="📄 Download Token Slip (PDF)",
                data=pdf_bytes,
                file_name=f"Hospital_Token_{token_info['token']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.info("Submit the check-in form to issue a token, view expected call time, and download your printable slip.")


# ===========================================================================
# PAGE 2: Track My Token (Feature 20)
# ===========================================================================
elif selected_page == "Track My Token":
    st.title("🎫 Track My Token")
    st.write("Check your real-time position in queue, remaining wait time, and expected consultation hour.")
    
    # Query parameter auto-fill
    default_token_val = st.query_params.get("token", "")
    token_input = st.text_input("Enter your Token Number", value=str(default_token_val), placeholder="e.g. 12")
    
    if token_input.strip().isdigit():
        t_num = int(token_input.strip())
        patient = db.get_patient_by_token(t_num)
        
        if patient:
            st.markdown("---")
            col_left, col_right = st.columns([1.5, 1])
            
            status_colors = {
                "waiting": ("#E3F2FD", "#1565C0", "⏳ Waiting in Queue"),
                "served": ("#E8F5E9", "#2E7D32", "✅ Completed / Served"),
                "cancelled": ("#FFEBEE", "#C62828", "❌ Cancelled"),
                "no_show": ("#FFF3E0", "#E65100", "⚠️ Marked No-Show")
            }
            bg_c, text_c, label_c = status_colors.get(patient["status"], ("#EEE", "#333", patient["status"]))
            
            with col_left:
                st.markdown(f"""
                <div style="background-color: {bg_c}; padding: 18px; border-radius: 8px; border-left: 6px solid {text_c};">
                    <h2 style="margin: 0; color: {text_c};">Token #{patient['token']} - {patient['name']}</h2>
                    <h4 style="margin: 5px 0 0 0; color: {text_c};">{label_c}</h4>
                    <p style="margin: 6px 0 0 0; color: #444;">Department: <strong>{patient['department']}</strong> | Type: <strong>{patient['visit_type']}</strong></p>
                    <p style="margin: 2px 0 0 0; color: #666; font-size: 13px;">Checked in: {patient['checkin_time']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if patient["status"] == "waiting":
                    # Calculate queue position among waiting patients in this department
                    waiting_df = db.get_waiting_patients_df(patient["department"])
                    queue_tokens = list(waiting_df["token"])
                    pos = queue_tokens.index(patient["token"]) + 1 if patient["token"] in queue_tokens else 1
                    patients_ahead = pos - 1
                    
                    # Remaining wait calculation
                    now = datetime.now()
                    checkin_dt = datetime.strptime(patient["checkin_time"], "%Y-%m-%d %H:%M:%S")
                    elapsed_min = (now - checkin_dt).total_seconds() / 60.0
                    remaining_wait = max(1.0, round(patient["predicted_wait"] - elapsed_min, 1))
                    
                    # Expected call time
                    exp_str = patient.get("expected_call_time")
                    if exp_str:
                        try:
                            exp_dt = datetime.strptime(exp_str, "%Y-%m-%d %H:%M:%S")
                            exp_display = exp_dt.strftime("%I:%M %p")
                        except Exception:
                            exp_display = exp_str
                    else:
                        exp_display = (checkin_dt + timedelta(minutes=patient["predicted_wait"])).strftime("%I:%M %p")
                        
                    st.write("")
                    q1, q2, q3 = st.columns(3)
                    q1.metric("Queue Position", f"#{pos} in line")
                    q2.metric("Patients Ahead", f"{patients_ahead}")
                    q3.metric("Est. Remaining Wait", f"{remaining_wait:.1f} mins")
                    
                    st.success(f"🕒 **Expected to be seen at {exp_display}**")
                    
                    st.write("")
                    if st.button("❌ Cancel My Token", type="secondary"):
                        db.cancel_patient_by_token(patient["token"])
                        st.warning("Your token has been cancelled.")
                        st.rerun()

                elif patient["status"] == "served":
                    st.success(f"🎉 Consultation completed at **{patient.get('called_time', 'N/A')}**. Actual wait time: **{patient.get('actual_wait_minutes', 'N/A')} mins**.")
                elif patient["status"] == "cancelled":
                    st.error("This token was cancelled and will not be called.")
                elif patient["status"] == "no_show":
                    st.warning("This token was marked as a no-show by the department staff.")

            with col_right:
                # PDF Slip Download
                exp_time_val = patient.get("expected_call_time", "")
                try:
                    exp_formatted = datetime.strptime(exp_time_val, "%Y-%m-%d %H:%M:%S").strftime("%I:%M %p") if exp_time_val else "N/A"
                except Exception:
                    exp_formatted = exp_time_val or "N/A"
                    
                pdf_payload = {
                    "token": patient["token"],
                    "name": patient["name"],
                    "department": patient["department"],
                    "visit_type": patient["visit_type"],
                    "checkin_time": patient["checkin_time"],
                    "position_text": f"Token #{patient['token']}",
                    "predicted_wait": patient["predicted_wait"],
                    "expected_call_time_str": exp_formatted
                }
                token_pdf_bytes = utils.generate_token_pdf(pdf_payload, st.session_state.get("app_url", ""))
                st.download_button(
                    label="📄 Download Token Slip (PDF)",
                    data=token_pdf_bytes,
                    file_name=f"Token_{patient['token']}_Slip.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.warning(f"No patient record found for Token #{t_num}.")


# ===========================================================================
# PAGE 3: Best Time to Visit (Feature 8)
# ===========================================================================
elif selected_page == "Best Time to Visit":
    st.title("⏰ Best Time to Visit Today")
    st.write("Plan your hospital consultation to minimize waiting. Our ML model predicts expected wait times across operating hours (8:00 AM – 7:00 PM).")
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        sel_dept = st.selectbox("Select Department", ["General", "Cardiology", "Orthopedics", "Pediatrics"])
    with col_sel2:
        sel_vt = st.selectbox("Select Visit Type", ["Walk-in", "Appointment", "Emergency"])
        
    if model is not None:
        docs = db.get_department_doctors(sel_dept)
        best_data = utils.compute_best_times_to_visit(model, sel_dept, sel_vt, docs)
        top_3 = best_data["top_3"]
        forecast_df = best_data["full_forecast"]
        
        st.subheader("Optimal Arrival Windows Today")
        c1, c2, c3 = st.columns(3)
        medals = ["🥇 Best Time", "🥈 Second Best", "🥉 Third Best"]
        cards = [c1, c2, c3]
        
        for idx in range(min(3, len(top_3))):
            rec = top_3[idx]
            with cards[idx]:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; text-align: center;">
                    <h4 style="margin: 0; color: #1565C0;">{medals[idx]}</h4>
                    <h2 style="margin: 8px 0; color: #333;">{rec['time_label']}</h2>
                    <p style="margin: 0; color: #2E7D32; font-weight: bold; font-size: 16px;">~{rec['predicted_wait']:.1f} mins wait</p>
                    <p style="margin: 4px 0 0 0; color: #777; font-size: 12px;">Est. queue: {rec['queue_length']} patients ({docs} doctors)</p>
                </div>
                """, unsafe_allow_html=True)
                
        st.write("")
        st.subheader("Full Operating Hours Predicted Wait Curve")
        chart_df = forecast_df.set_index("time_label")[["predicted_wait"]]
        st.bar_chart(chart_df)
    else:
        st.warning("Model file `model.pkl` is missing.")


# ===========================================================================
# PAGE 4: QR Check-in (Feature 19)
# ===========================================================================
elif selected_page == "QR Check-in":
    st.title("📱 Contactless QR Code Check-in")
    st.write("Patients can scan this QR code with their mobile smartphone camera to directly access the check-in form.")
    
    col_qr, col_settings = st.columns([1, 1.2])
    
    with col_settings:
        st.subheader("Network & Host Settings")
        detected_ip = utils.get_local_ip()
        st.info(f"Detected Local Machine IP: **`{detected_ip}`**")
        
        app_url_input = st.text_input(
            "Application Base URL",
            value=st.session_state.get("app_url", utils.get_default_app_url()),
            help="Ensure this points to the machine's local Wi-Fi IP address so phones can connect."
        )
        st.session_state["app_url"] = app_url_input
        
        checkin_url = f"{app_url_input}/?page=checkin"
        st.code(checkin_url, language="text")
        
        st.warning("⚠️ **Wi-Fi Connectivity Requirement:** The patient's mobile smartphone must be connected to the same Wi-Fi / local network as this computer to open the page.")

    with col_qr:
        st.subheader("Scan to Check In")
        qr_image = utils.generate_qr_code(checkin_url)
        st.image(qr_image, caption="Point smartphone camera here to Check In", width=280)
        
        qr_bytes = utils.get_qr_bytes(qr_image)
        st.download_button(
            label="💾 Download QR Code (PNG)",
            data=qr_bytes,
            file_name="Hospital_CheckIn_QR.png",
            mime="image/png",
            use_container_width=True
        )


# ===========================================================================
# PAGE 5: Staff Panel (Staff & Admin)
# ===========================================================================
elif selected_page == "Staff Panel":
    st.title("👩‍⚕️ Clinical Staff Queue Management")
    st.write("Manage live department waiting lists, prioritize emergency arrivals, and notify patients.")
    
    col_filter, col_action = st.columns([2, 1])
    with col_filter:
        staff_dept = st.selectbox(
            "Filter Department",
            ["All", "General", "Cardiology", "Orthopedics", "Pediatrics"]
        )
    with col_action:
        st.write("")
        call_next_btn = st.button("🔔 Call Next Patient", use_container_width=True, type="primary")

    if call_next_btn:
        next_p = db.get_next_patient_to_call(staff_dept)
        if next_p:
            now_dt = datetime.now()
            called_time_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
            checkin_dt = datetime.strptime(next_p["checkin_time"], "%Y-%m-%d %H:%M:%S")
            actual_wait = round((now_dt - checkin_dt).total_seconds() / 60.0, 1)
            
            # Update DB
            db.mark_patient_served(next_p["id"], called_time_str, actual_wait)
            
            # 1. Notify the called patient
            res_called = notifier.notify_called_patient(next_p)
            st.toast(f"📢 Called #{next_p['token']} ({next_p['name']}): Notified via {res_called['dispatches'][0][0]}!")
            
            # 2. Notify new first person in line
            new_first = db.get_first_waiting_in_line(next_p["department"], exclude_id=next_p["id"])
            if new_first:
                res_next = notifier.notify_next_in_line(new_first)
                st.toast(f"🔔 Notified next in line: Token #{new_first['token']} ({new_first['name']})")
                
            st.success(
                f"📢 **Now Calling Token #{next_p['token']}**: {next_p['name']} "
                f"({next_p['department']} - {next_p['visit_type']}) | "
                f"Actual Wait: **{actual_wait} mins** (Predicted: {next_p['predicted_wait']} mins)"
            )
        else:
            st.warning(f"No waiting patients found for: **{staff_dept}**.")

    # Fetch waiting and served tables
    waiting_df = db.get_waiting_patients_df(staff_dept)
    served_df = db.get_served_patients_df(staff_dept)

    # Quick Patient Action (Cancel / No-show)
    if not waiting_df.empty:
        with st.expander("⚙️ Manage Waiting Patient Status (No-show / Cancel)", expanded=False):
            p_options = {f"Token #{row['token']} - {row['name']} ({row['department']})": row['token'] for _, row in waiting_df.iterrows()}
            selected_label = st.selectbox("Select Patient", list(p_options.keys()))
            selected_token = p_options[selected_label]
            
            b1, b2 = st.columns(2)
            with b1:
                if st.button("⚠️ Mark as No-Show", use_container_width=True):
                    p_rec = db.get_patient_by_token(selected_token)
                    if p_rec:
                        db.update_patient_status(p_rec["id"], "no_show")
                        st.warning(f"Token #{selected_token} marked as No-Show.")
                        st.rerun()
            with b2:
                if st.button("❌ Cancel Token", use_container_width=True):
                    db.cancel_patient_by_token(selected_token)
                    st.info(f"Token #{selected_token} cancelled.")
                    st.rerun()

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"⏳ Waiting Queue ({len(waiting_df)})")
        if not waiting_df.empty:
            # Format expected call time for display
            disp_wait_df = waiting_df.copy()
            st.dataframe(
                disp_wait_df[["token", "name", "department", "visit_type", "predicted_wait", "expected_call_time", "checkin_time"]],
                use_container_width=True,
                column_config={
                    "token": st.column_config.NumberColumn("Token #", format="%d"),
                    "predicted_wait": st.column_config.NumberColumn("Pred. Wait", format="%.1f min"),
                    "expected_call_time": st.column_config.TextColumn("Expected Call Time")
                }
            )
        else:
            st.info("No patients currently waiting.")

    with c2:
        st.subheader(f"✅ Served Patients ({len(served_df)})")
        if not served_df.empty:
            st.dataframe(
                served_df[["token", "name", "department", "visit_type", "predicted_wait", "expected_call_time", "called_time", "actual_wait_minutes"]],
                use_container_width=True,
                column_config={
                    "token": st.column_config.NumberColumn("Token #", format="%d"),
                    "predicted_wait": st.column_config.NumberColumn("Pred (min)", format="%.1f min"),
                    "actual_wait_minutes": st.column_config.NumberColumn("Actual (min)", format="%.1f min")
                }
            )
        else:
            st.info("No patients served yet.")


# ===========================================================================
# PAGE 6: Doctor Load Balancing (Feature 12)
# ===========================================================================
elif selected_page == "Doctor Load Balancing":
    st.title("⚖️ Doctor Load Balancing & Staff Allocation")
    st.write("Monitor bottleneck departments, simulate on-duty doctor reallocations, and optimize patient throughput.")

    departments = ["General", "Cardiology", "Orthopedics", "Pediatrics"]
    dept_docs = db.get_all_department_doctors()
    
    # Calculate current queue lengths and typical predicted wait
    dept_queues = {d: db.get_waiting_queue_count(d) for d in departments}
    
    now = datetime.now()
    cur_hour = max(8, min(19, now.hour))
    cur_weekday = now.weekday()
    
    current_waits = {}
    high_wait_alerts = []
    
    for d in departments:
        q_len = dept_queues[d]
        docs = dept_docs.get(d, 2)
        if model is not None:
            inp = pd.DataFrame([{
                "hour": cur_hour,
                "day_of_week": cur_weekday,
                "department": d,
                "visit_type": "Walk-in",
                "queue_length": q_len,
                "doctors_available": docs
            }])
            p_wait = round(max(0.0, float(model.predict(inp)[0])), 1)
        else:
            p_wait = 0.0
        current_waits[d] = p_wait
        if p_wait > 45.0:
            high_wait_alerts.append((d, p_wait))

    # Display high wait red alerts
    for alert_dept, alert_wait in high_wait_alerts:
        st.error(f"🚨 **High Wait Bottleneck Alert**: **{alert_dept}** estimated wait is **{alert_wait:.1f} mins** (exceeds the 45-minute threshold)!")

    # Current Status Cards
    st.subheader("Current Department Capacities & Wait Estimates")
    cols = st.columns(4)
    for i, d in enumerate(departments):
        with cols[i]:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border: 1px solid #ced4da; border-radius: 8px; padding: 15px;">
                <h3 style="margin: 0; color: #1E88E5;">{d}</h3>
                <h4 style="margin: 5px 0; color: #333;">{dept_docs.get(d, 2)} Doctor(s)</h4>
                <p style="margin: 2px 0; color: #555;">Waiting: <strong>{dept_queues[d]}</strong> patients</p>
                <p style="margin: 2px 0; color: {'#D32F2F' if current_waits[d] > 45 else '#2E7D32'}; font-weight: bold;">
                    Est. Walk-in Wait: {current_waits[d]:.1f} mins
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Suggestion Engine
    st.write("")
    st.subheader("🤖 AI Doctor Load Balancing Recommendation")
    if model is not None:
        rec = utils.compute_load_balancing_suggestion(model, dept_docs, dept_queues, cur_hour, cur_weekday)
        
        if rec["can_balance"]:
            st.markdown(f"""
            <div style="background-color: #E8F5E9; border-left: 6px solid #2E7D32; padding: 16px; border-radius: 6px; margin-bottom: 15px;">
                <h3 style="margin: 0; color: #1B5E20;">💡 Recommended Staff Transfer</h3>
                <p style="margin: 8px 0 0 0; font-size: 16px; color: #222;">
                    Move <strong>1 doctor</strong> from <strong>{rec['move_from']}</strong> to <strong>{rec['move_to']}</strong>: 
                    highest wait drops from <strong>{rec['current_highest']:.1f} min</strong> to <strong>{rec['new_highest']:.1f} min</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("⚡ Apply Suggestion", type="primary"):
                db.update_department_doctors(rec["move_from"], dept_docs[rec["move_from"]] - 1)
                db.update_department_doctors(rec["move_to"], dept_docs[rec["move_to"]] + 1)
                st.success(f"Applied! 1 doctor moved from {rec['move_from']} to {rec['move_to']}.")
                st.rerun()
        else:
            st.success("✅ The doctor load is currently balanced across all departments.")

    # Manual Doctor Allocation Editor
    st.write("")
    with st.expander("🛠️ Manually Adjust Doctor Allocations", expanded=False):
        with st.form("edit_doctors_form"):
            new_allocations = {}
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            d_cols = [col_d1, col_d2, col_d3, col_d4]
            for idx, d in enumerate(departments):
                with d_cols[idx]:
                    new_allocations[d] = st.number_input(f"{d} Doctors", min_value=1, max_value=5, value=dept_docs.get(d, 2))
                    
            if st.form_submit_button("Save Doctor Allocations", use_container_width=True):
                for d, count in new_allocations.items():
                    db.update_department_doctors(d, count)
                st.success("Updated doctor allocations saved successfully!")
                st.rerun()


# ===========================================================================
# PAGE 7: Wait Forecast (Staff & Admin) (Feature 7)
# ===========================================================================
elif selected_page == "Wait Forecast":
    st.title("📈 Hourly Wait Forecast Analysis")
    st.write("Simulate expected wait times throughout operating hours based on historical arrival distributions.")
    
    col_fc1, col_fc2, col_fc3 = st.columns(3)
    with col_fc1:
        fc_dept = st.selectbox("Department", ["General", "Cardiology", "Orthopedics", "Pediatrics"])
    with col_fc2:
        fc_vt = st.selectbox("Visit Type", ["Walk-in", "Appointment", "Emergency"])
    with col_fc3:
        default_docs = db.get_department_doctors(fc_dept)
        fc_docs = st.slider("Doctors On Duty", min_value=1, max_value=5, value=default_docs)

    if model is not None:
        today_weekday = datetime.now().weekday()
        forecast_df = utils.compute_hourly_forecast(model, fc_dept, fc_vt, fc_docs, today_weekday)
        
        st.subheader(f"Wait Time Forecast for {fc_dept} ({fc_vt})")
        chart_data = forecast_df.set_index("time_label")[["predicted_wait"]]
        st.line_chart(chart_data)
        
        st.subheader("Hourly Prediction Table")
        
        display_tbl = forecast_df.copy()
        display_tbl["Status"] = display_tbl["is_current_hour"].apply(lambda x: "⏰ CURRENT HOUR" if x else "")
        
        st.dataframe(
            display_tbl[["time_label", "predicted_wait", "queue_length", "doctors", "Status"]],
            use_container_width=True,
            column_config={
                "time_label": "Operating Hour",
                "predicted_wait": st.column_config.NumberColumn("Predicted Wait (min)", format="%.1f min"),
                "queue_length": "Avg Historical Queue",
                "doctors": "Doctors On Duty"
            }
        )
    else:
        st.warning("Model file `model.pkl` not found.")


# ===========================================================================
# PAGE 8: Notifications Log (Staff & Admin)
# ===========================================================================
elif selected_page == "Notifications Log":
    st.title("📬 Notifications Dispatch Log")
    st.write("Audit all outbound patient communications dispatched via Email, SMS (Twilio), or Local Simulation.")
    
    notif_df = db.get_all_notifications_df()
    
    if not notif_df.empty:
        col_nf1, col_nf2 = st.columns([1, 2])
        with col_nf1:
            chan_filter = st.selectbox("Filter Channel", ["All"] + list(notif_df["channel"].unique()))
        
        filtered_df = notif_df if chan_filter == "All" else notif_df[notif_df["channel"] == chan_filter]
        
        st.dataframe(
            filtered_df,
            use_container_width=True,
            column_config={
                "id": "ID",
                "token": st.column_config.NumberColumn("Token #", format="%d"),
                "channel": "Channel",
                "recipient": "Recipient",
                "message": "Message Content",
                "status": "Delivery Status",
                "created_at": "Timestamp"
            }
        )
        
        # CSV Export
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Notifications Log (CSV)",
            data=csv_data,
            file_name="notifications_log.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No notifications have been recorded yet.")


# ===========================================================================
# PAGE 9: Analytics (Admin Only)
# ===========================================================================
elif selected_page == "Analytics":
    st.title("📊 Hospital Operations & ML Analytics")
    st.write("Machine learning model benchmarks, historical queue dynamics, and live system metrics.")

    # 1. Model Comparison Table
    st.subheader("1. Predictive Model Benchmarks")
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        
        metrics_rows = []
        for m_name, scores in metrics_data.items():
            metrics_rows.append({
                "Model": m_name,
                "MAE (min)": scores["MAE"],
                "RMSE (min)": scores["RMSE"],
                "R² Score": scores["R2"]
            })
        m_df = pd.DataFrame(metrics_rows)
        st.dataframe(m_df, use_container_width=True)
        st.caption("Models trained and evaluated on 80/20 split of hospital dataset.")
    else:
        st.warning("`metrics.json` not found.")

    # 2. Historical Data Visualizations
    st.subheader("2. Historical Queue & Wait Time Analysis")
    if os.path.exists(DATA_FILE):
        hist_df = pd.read_csv(DATA_FILE)
        
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Average Wait Time by Hour (Minutes)**")
            wait_by_hour = hist_df.groupby("hour")["wait_time"].mean().round(1)
            st.bar_chart(wait_by_hour)

        with g2:
            st.markdown("**Average Queue Length by Hour (Patients)**")
            queue_by_hour = hist_df.groupby("hour")["queue_length"].mean().round(1)
            st.line_chart(queue_by_hour)

        st.markdown("**Average Wait Time by Department (Minutes)**")
        wait_by_dept = hist_df.groupby("department")["wait_time"].mean().round(1)
        st.bar_chart(wait_by_dept)
    else:
        st.warning("`data.csv` not found.")

    # 3. Live System Data
    st.subheader("3. Live System Telemetry (SQLite: `queue.db`)")
    conn = db.get_db_connection()
    live_df = pd.read_sql_query("SELECT * FROM patients", conn)
    conn.close()

    if not live_df.empty:
        total_p = len(live_df)
        waiting_p = (live_df["status"] == "waiting").sum()
        served_p = (live_df["status"] == "served").sum()
        cancelled_p = (live_df["status"] == "cancelled").sum()
        
        served_only = live_df[live_df["status"] == "served"]
        avg_pred = live_df["predicted_wait"].mean()
        avg_act = served_only["actual_wait_minutes"].mean() if not served_only.empty else None

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Registered", f"{total_p}")
        k2.metric("Waiting / Served", f"{waiting_p} / {served_p}")
        k3.metric("Avg Predicted Wait", f"{avg_pred:.1f} mins" if pd.notnull(avg_pred) else "N/A")
        k4.metric("Avg Actual Wait", f"{avg_act:.1f} mins" if (avg_act is not None and pd.notnull(avg_act)) else "N/A")

        st.markdown("**Live Patient Distribution by Department**")
        st.bar_chart(live_df["department"].value_counts())
    else:
        st.info("No patient records in `queue.db` yet.")
