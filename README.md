# Intelligent Queue Management and Waiting-Time Prediction (Hospital)

An end-to-end, lightweight Python application for hospital outpatient queue management and machine learning-powered patient waiting-time prediction. Built with Python 3.13, Scikit-learn, Streamlit, SQLite, `qrcode[pil]`, and `fpdf2`.

---

## 📌 Project Overview

Hospital emergency and outpatient departments frequently suffer from unpredictable bottlenecks, patient dissatisfaction, and staffing inefficiencies. This system provides:
1. **Synthetic Data Generator (`generate_data.py`)**: Simulates 5,000 realistic patient visits across multiple departments, visit priorities, Poisson-distributed queue lengths with peak-hour surges, and doctor allocations.
2. **Machine Learning Pipeline (`train_model.py`)**: Preprocesses categorical and numerical operational features using a `ColumnTransformer` (One-Hot Encoding + Passthrough) and trains three regressors (Baseline Mean, Linear Regression, and Random Forest), benchmarked with MAE, RMSE, and $R^2$.
3. **Database Architecture (`db.py` & `queue.db`)**: Lightweight embedded SQLite database tracking patients, tokens, timestamps, predicted wait times, actual consultation durations, on-duty department staffing, role-based users, and notification dispatch logs.
4. **Role-Based Authentication (`auth.py`)**: PBKDF2 HMAC SHA-256 salted password hashing supporting 3 roles: `patient` (public/guest), `staff`, and `admin`.
5. **Notification Dispatcher (`notifier.py`)**: Automatically alerts called patients and upcoming patients via Email (SMTP), SMS (Twilio), or Local In-App Simulation with complete audit logs.
6. **Hospital Utilities (`utils.py`)**: Network IP auto-discovery, contactless QR code check-in generation, in-memory PDF token slip generation, 12-hour wait forecasts, and automated doctor load balancing recommendations.
7. **Interactive Dashboard (`app.py`)**: Multi-page Streamlit application with query parameter routing (`st.query_params`) and role-based page protection.

---

## ⚙️ Setup & Installation

### Prerequisites
- Windows OS with PowerShell
- Python 3.13
- Existing virtual environment in `.\venv`

### Activation & Package Installation
Open PowerShell in the project directory:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Packages in `requirements.txt`:
- `pandas`
- `numpy`
- `scikit-learn`
- `streamlit`
- `joblib`
- `qrcode[pil]`
- `fpdf2`

---

## 🔐 User Roles & Default Logins

The application provides strict Role-Based Access Control (RBAC):

| Role | Default Username | Default Password | Accessible Pages |
| :--- | :--- | :--- | :--- |
| **Patient (Guest)** | *(No login needed)* | *(None)* | Patient Check-in, Track My Token, Best Time to Visit, QR Check-in |
| **Staff** | `staff` | `staff123` | *All Patient pages* + Staff Panel, Doctor Load Balancing, Notifications Log, Wait Forecast |
| **Admin** | `admin` | `admin123` | *All pages* (including Analytics and System Telemetry) |

Passwords are encrypted using `hashlib.pbkdf2_hmac` with SHA-256, 100,000 iterations, and a 16-byte random salt.

---

## 🔔 Live Notifications (Email & SMS Configuration)

The system automatically delivers notifications when staff call the next patient:
1. **To the Called Patient**: *"Please proceed to {department} for your consultation."*
2. **To the Next Patient in Line**: *"You are next in line for {department}. Please stand by."*

### Enabling Live Channels via Environment Variables
- **Email (SMTP)**: Set the following environment variables:
  ```powershell
  $env:SMTP_HOST = "smtp.gmail.com"
  $env:SMTP_PORT = "587"
  $env:SMTP_USER = "your-hospital@gmail.com"
  $env:SMTP_PASSWORD = "your-app-password"
  ```
- **SMS (Twilio)**: Set the following environment variables:
  ```powershell
  $env:TWILIO_SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  $env:TWILIO_TOKEN = "your_auth_token"
  $env:TWILIO_FROM = "+15551234567"
  ```
- **Simulation Fallback**: If these environment variables are not set or credentials fail, the system automatically runs in **SIMULATE** mode, displaying notifications via `st.toast` and logging them into the `notifications` SQLite table.

---

## 📱 QR Code Check-in & Mobile Wi-Fi Note

- The **QR Check-in** page auto-detects the host computer's local network IP (e.g., `http://192.168.x.x:8501/?page=checkin`).
- ⚠️ **Important Wi-Fi Requirement**: In order for a patient's mobile smartphone to open the check-in form via the QR code, the phone must be connected to the **same Wi-Fi network** as the computer hosting the app.
- An editable base URL field is provided in the sidebar and on the QR page for custom hostnames or reverse proxies.

---

## 🚀 Execution & Run Order

Run the modules in the following order:

### Step 1: Generate Synthetic Data
```powershell
python generate_data.py
```
- Creates `data.csv` with 5,000 synthetic patient visits.

### Step 2: Train & Evaluate Models
```powershell
python train_model.py
```
- Trains Dummy Baseline, Linear Regression, and Random Forest pipelines.
- Saves `model.pkl` and `metrics.json`.

### Step 3: Run Validation Suite
```powershell
python test_features.py
```
- Validates database migration, authentication, expected call time, forecast calculations, doctor load balancing, PDF generation, QR rendering, and simulated notifications.

### Step 4: Launch Streamlit Application
```powershell
streamlit run app.py
```
- Application Local URL: **`http://localhost:8501`**

---

## 📊 Summary of ML Results

In model benchmarks evaluated on the held-out 20% test dataset (1,000 visits), the **Random Forest Regressor** decisively outperformed both the Baseline Mean predictor and Linear Regression:

| Model | MAE (min) | RMSE (min) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Baseline (Mean)** | 25.5409 | 35.4743 | -0.0014 |
| **Linear Regression** | 11.6317 | 16.6167 | 0.7803 |
| **Random Forest (200 Trees)** | **3.2071** | **4.2944** | **0.9853** |

Random Forest achieved an **$R^2$ score of 0.9853** and reduced **MAE to 3.21 minutes**, approaching the theoretical limit set by the underlying Gaussian operational noise ($\sigma = 3.0$).

---

## 🗄️ Database Tables (`queue.db`)

1. **`patients`**: `id`, `token`, `name`, `department`, `visit_type`, `predicted_wait`, `checkin_time`, `called_time`, `actual_wait_minutes`, `status` (`'waiting'`, `'served'`, `'cancelled'`, `'no_show'`), `phone`, `email`, `expected_call_time`.
2. **`department_status`**: `department`, `doctors_available`.
3. **`users`**: `username`, `password_hash`, `role`.
4. **`notifications`**: `id`, `token`, `channel`, `recipient`, `message`, `status`, `created_at`.
