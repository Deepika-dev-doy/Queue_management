import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def train_and_evaluate(data_path="data.csv", model_path="model.pkl", metrics_path="metrics.json"):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    features = ["hour", "day_of_week", "department", "visit_type", "queue_length", "doctors_available"]
    target = "wait_time"
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")
    
    # Preprocessor: OneHotEncoder for categorical features, passthrough for numerical
    categorical_features = ["department", "visit_type"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"
    )
    
    # 1. Baseline Model (Mean predictor)
    baseline_model = DummyRegressor(strategy="mean")
    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)
    
    # 2. Linear Regression Pipeline
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    
    # 3. Random Forest Pipeline (200 trees, random_state=42)
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    
    # Metric evaluation
    def evaluate(y_true, y_pred):
        return {
            "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "RMSE": round(float(root_mean_squared_error(y_true, y_pred)), 4),
            "R2": round(float(r2_score(y_true, y_pred)), 4)
        }
    
    metrics = {
        "Baseline (Mean)": evaluate(y_test, y_pred_base),
        "Linear Regression": evaluate(y_test, y_pred_lr),
        "Random Forest": evaluate(y_test, y_pred_rf)
    }
    
    # Print results formatted nicely
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS (Test Set)")
    print("=" * 60)
    print(f"{'Model':<25} | {'MAE (min)':<10} | {'RMSE (min)':<10} | {'R2 Score':<10}")
    print("-" * 60)
    for model_name, score in metrics.items():
        print(f"{model_name:<25} | {score['MAE']:<10.4f} | {score['RMSE']:<10.4f} | {score['R2']:<10.4f}")
    print("=" * 60)
    
    # Save the Random Forest pipeline
    joblib.dump(rf_pipeline, model_path)
    print(f"\nRandom Forest pipeline saved to {model_path}")
    
    # Save metrics to JSON
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    return metrics

if __name__ == "__main__":
    train_and_evaluate()
