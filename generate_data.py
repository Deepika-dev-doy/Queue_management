import numpy as np
import pandas as pd

def generate_hospital_data(n_samples=5000, seed=42, output_path="data.csv"):
    np.random.seed(seed)
    
    # 1. Operational columns
    hours = np.random.randint(8, 20, size=n_samples)  # 8 to 19 inclusive
    days_of_week = np.random.randint(0, 7, size=n_samples)  # 0 to 6 (0=Monday)
    
    departments = ["General", "Cardiology", "Orthopedics", "Pediatrics"]
    chosen_departments = np.random.choice(departments, size=n_samples)
    
    visit_types = ["Walk-in", "Appointment", "Emergency"]
    visit_probs = [0.50, 0.40, 0.10]
    chosen_visit_types = np.random.choice(visit_types, size=n_samples, p=visit_probs)
    
    doctors_available = np.random.randint(1, 4, size=n_samples)  # 1 to 3
    
    # 2. Queue length (Poisson, mean 6, x1.5 in peak hours 9, 10, 11, 17, 18)
    peak_hours = {9, 10, 11, 17, 18}
    poisson_means = np.array([6.0 * 1.5 if h in peak_hours else 6.0 for h in hours])
    queue_lengths = np.random.poisson(poisson_means)
    
    # 3. Wait time calculation
    # service_time = General 8, Cardiology 15, Orthopedics 12, Pediatrics 10 minutes
    service_times = {
        "General": 8,
        "Cardiology": 15,
        "Orthopedics": 12,
        "Pediatrics": 10
    }
    st_arr = np.array([service_times[dept] for dept in chosen_departments])
    
    # Visit factor (Walk-in 1.0, Appointment 0.7, Emergency 0.2)
    visit_factors = {
        "Walk-in": 1.0,
        "Appointment": 0.7,
        "Emergency": 0.2
    }
    vf_arr = np.array([visit_factors[vt] for vt in chosen_visit_types])
    
    # Monday factor (1.2 on day 0)
    day_factors = np.where(days_of_week == 0, 1.2, 1.0)
    
    # Base calculation
    base_wait = (queue_lengths * st_arr) / doctors_available
    adjusted_wait = base_wait * vf_arr * day_factors
    
    # Add Gaussian noise (std 3), clip at 0, round to 1 decimal
    noise = np.random.normal(loc=0.0, scale=3.0, size=n_samples)
    final_wait = np.clip(adjusted_wait + noise, 0, None)
    final_wait = np.round(final_wait, 1)
    
    df = pd.DataFrame({
        "hour": hours,
        "day_of_week": days_of_week,
        "department": chosen_departments,
        "visit_type": chosen_visit_types,
        "queue_length": queue_lengths,
        "doctors_available": doctors_available,
        "wait_time": final_wait
    })
    
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows and saved to {output_path}")
    print("Sample records:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())
    return df

if __name__ == "__main__":
    generate_hospital_data()
